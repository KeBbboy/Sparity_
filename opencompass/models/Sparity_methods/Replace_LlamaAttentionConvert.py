# flake8: noqa
# yapf: disable
import argparse
from argparse import ArgumentParser
from typing import Dict, List, Optional, Union
import transformers
import torch
from mmengine.device import is_npu_available
import os
import csv
import builtins
from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]
import math
import pdb
import types
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Cache, GenerationConfig, LlamaConfig)
# from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      LlamaForCausalLM,
                                                      LlamaRotaryEmbedding,
                                                      apply_rotary_pos_emb,
                                                      rotate_half)
from typing_extensions import Unpack
import time
from opencompass.models import BaseModel
from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.utils import get_logger

from .cake.utils import CompressConfig
from .monkeypatch import replace_llama,replace_qwen2
import time
    

import os
from datetime import datetime

def extract_question_part(text):
    start_index = text.find("Question: ")
    return text[start_index:] if start_index != -1 else ""

def add_quest_args(parser: ArgumentParser):
    parser.add_argument('--dynamic-linear', action='store_true')
    parser.add_argument('--dynamic-ntk', type=float)
    parser.add_argument('--dynamic-part-ntk', action='store_true')
    parser.add_argument('--dynamic-yarn', action='store_true')
    parser.add_argument('--ntk', type=float)
    parser.add_argument('--part-ntk', type=float)
    parser.add_argument('--linear', type=float)
    parser.add_argument('--yarn', type=float)
    parser.add_argument('--rerope', type=float)
    parser.add_argument('--factor', type=float)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--finetuned', action='store_true')
    parser.add_argument('--gpt-neox-max-length', type=int)
    parser.add_argument('--adapter', type=str)
    parser.add_argument('--max-position-embeddings', type=int)
    parser.add_argument('--original-max-position-embeddings', type=int)
    parser.add_argument('--sliding-window-attention', type=int)
    parser.add_argument('--custom-model', action='store_true')
    parser.add_argument('--custom-model-together', action='store_true')
    parser.add_argument('--custom-model-mistral', action='store_true')
    parser.add_argument('--no-use-cache', action='store_true')
    return parser

class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        batch_size: int,
    ):
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence,
                                             add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # compare the last len(stop) tokens
        lookback_ids_batch = input_ids[:, -self.sequence_id_len:]
        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
        for i, done in enumerate(self.done_tracker):
            if done:
                continue
            self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker

@MODELS.register_module()
class Replace_LlamaAttentionConvert(HuggingFaceCausalLM):
    def __init__(self, *args, **kwargs):

        self.max_seq_len = kwargs.get('max_seq_len', 16384)

        self.method = kwargs.pop('method')
        self.arkvale_kwargs = kwargs.pop('arkvale_kwargs', {})
        self.infllm_kwargs = kwargs.pop('infllm_kwargs', {})
        self.cache_kwargs = kwargs.pop('cache_kwargs', {})
        self.past_key_values = None
        super().__init__(*args, **kwargs)

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        
        
        self._set_model_kwargs_torch_dtype(model_kwargs)

        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        if self.method == 'magicpig':
            # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            # self.model = LlamaForCausalLM.from_pretrained(path, torch_dtype = torch.bfloat16)
            # self.model = self.model.to("cuda")  # 或 self.model.to("cuda:0") 指定第一个 GPU
            return 
        # replace_llama(self=self, path=path, model_kwargs = model_kwargs)
        replace_qwen2(self=self, path=path, model_kwargs = model_kwargs)
        print(self.model)
        print('Model kwargs:')
        print(model_kwargs)


        self.model.eval()
        self.model.generation_config.do_sample = False
    
    def _single_generate(self,
                        inputs: List[str],
                        max_out_len: int,
                        min_out_len: Optional[int] = None,
                        stopping_criteria: List[str] = [],
                        **kwargs) -> List[str]:

        csv_filename = f'inference_stats_{max_out_len}.csv'

        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        if self.use_fastchat_template:
            try:
                from fastchat.model import get_conversation_template
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'Fastchat is not implemented. You can use '
                    '\'pip install "fschat[model_worker,webui]"\' '
                    'to implement fastchat.')
            conv = get_conversation_template('vicuna')
            conv.append_message(conv.roles[0], inputs[0])
            conv.append_message(conv.roles[1], None)
            inputs = [conv.get_prompt()]

        if self.mode == 'mid':
            input_ids = self.tokenizer(inputs, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            if len(input_ids[0]) > self.max_seq_len - max_out_len:
                half = int((self.max_seq_len - max_out_len) / 2)
                inputs = [
                    self.tokenizer.decode(input_ids[0][:half],
                                        skip_special_tokens=True) +
                    self.tokenizer.decode(input_ids[0][-half:],
                                        skip_special_tokens=True)
                ]
        # ============== 使用自动适应窗口大小 ==============
        # question_part = extract_question_part(inputs[0])
        # questiont_input = self.tokenizer(question_part,
        #                          truncation=True,
        #                          max_length=self.max_seq_len - max_out_len)['input_ids']

    
        # questiont_token_count = len(questiont_input)
        # # Update model config with the new window size
        # if hasattr(self.model, 'config'):
        #     self.model.config.window_size = questiont_token_count
        #     if hasattr(self.model.config, 'max_capacity_prompt'):
        #         self.model.config.max_capacity_prompt = max(self.model.config.max_capacity_prompt, questiont_token_count + 256)  # Add buffer to ensure it's larger

        # ============== 使用自动适应窗口大小 ==============
        torch.cuda.synchronize()

        
        input_ids = self.tokenizer(inputs,
                                truncation=True,
                                max_length=self.max_seq_len - 
                                max_out_len)['input_ids']
        
        # 计算输入token数量
        input_token_count = sum(len(ids) for ids in input_ids)
        
        input_ids = torch.tensor(input_ids, device=self.model.device)
        origin_stopping_criteria = stopping_criteria
        if stopping_criteria:
            # Construct huggingface stopping criteria
            if self.tokenizer.eos_token is not None:
                stopping_criteria = stopping_criteria + [
                    self.tokenizer.eos_token
                ]
            stopping_criteria = transformers.StoppingCriteriaList([
                *[
                    MultiTokenEOSCriteria(sequence, self.tokenizer,
                                    input_ids.shape[0])
                    for sequence in stopping_criteria
                ],
            ])
            kwargs['stopping_criteria'] = stopping_criteria

        if min_out_len is not None:
            kwargs['min_new_tokens'] = min_out_len

        # 创建CUDA事件来测量GPU时间
        start_event = torch.cuda.Event(enable_timing=True)
        first_token_event = None
        end_event = torch.cuda.Event(enable_timing=True)
        
        # 添加prefill和decode阶段的事件
        prefill_start_event = torch.cuda.Event(enable_timing=True)
        prefill_end_event = torch.cuda.Event(enable_timing=True)
        decode_start_event = torch.cuda.Event(enable_timing=True)
        
        # 记录总体开始时间（包括CPU+GPU时间）
        cpu_start_time = time.time()

        # GPU计时开始
        start_event.record()
        # Prefill阶段开始
        prefill_start_event.record()
        
        # 创建KV缓存内存监控器
        class KVCacheMemoryMonitor:
            def __init__(self):
                self.max_kv_cache_memory = 0
                
            def update_kv_cache_memory(self, past_key_values):
                """更新KV缓存内存统计"""
                if past_key_values is None:
                    return
                    
                total_kv_memory = 0
                try:
                    # 计算所有层的KV缓存内存
                    for layer_idx in range(len(past_key_values)):
                        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
                            # DynamicCache类型
                            if layer_idx < len(past_key_values.key_cache):
                                key_tensor = past_key_values.key_cache[layer_idx]
                                value_tensor = past_key_values.value_cache[layer_idx]
                                
                                # 计算tensor的内存占用（字节）
                                key_memory = key_tensor.numel() * key_tensor.element_size()
                                value_memory = value_tensor.numel() * value_tensor.element_size()
                                total_kv_memory += key_memory + value_memory
                        else:
                            # 传统的tuple格式
                            key_tensor, value_tensor = past_key_values[layer_idx]
                            key_memory = key_tensor.numel() * key_tensor.element_size()
                            value_memory = value_tensor.numel() * value_tensor.element_size()
                            total_kv_memory += key_memory + value_memory
                except Exception as e:
                    # 如果计算失败，返回当前值
                    pass
                
                current_kv_cache_memory = total_kv_memory / (1024 ** 2)  # 转换为MB
                self.max_kv_cache_memory = max(self.max_kv_cache_memory, current_kv_cache_memory)
        
        kv_monitor = KVCacheMemoryMonitor()
        prefill_kv_cache_memory = 0  # 新增字段
        # 使用自定义callback来捕获第一个token生成时间和后续token
        class TokenGenerationCallback(transformers.StoppingCriteria):
            def __init__(self, model_ref, kv_monitor):
                self.first_token_captured = False
                self.first_token_event = None
                self.token_count = 0
                self.max_memory_allocated = 0
                self.prefill_done = False
                self.model_ref = model_ref  # 模型引用，用于访问past_key_values
                self.kv_monitor = kv_monitor  # KV缓存监控器
                
            def __call__(self, input_ids, scores, **kwargs):
                self.token_count = input_ids.shape[1] - kwargs.get('input_length', 0)
                
                # 捕获第一个token的时间
                if not self.first_token_captured and self.token_count > 0:
                    self.first_token_captured = True
                    self.first_token_event = torch.cuda.Event(enable_timing=True)
                    self.first_token_event.record()
                    
                    # 标记prefill阶段结束和decode阶段开始
                    if not self.prefill_done:
                        prefill_end_event.record()
                        decode_start_event.record()
                        self.prefill_done = True
                    
                    torch.cuda.synchronize()
                
                # 监控总内存使用情况的峰值
                torch.cuda.synchronize()
                current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                self.max_memory_allocated = max(self.max_memory_allocated, current_memory)
                
                # 监控KV缓存内存占用峰值
                past_key_values = kwargs.get('past_key_values', None)
                self.kv_monitor.update_kv_cache_memory(past_key_values)
                
                return False

        token_callback = TokenGenerationCallback(self.model, kv_monitor)
        
        # 添加回调到stopping criteria
        if 'stopping_criteria' in kwargs:
            kwargs['stopping_criteria'].append(token_callback)
        else:
            kwargs['stopping_criteria'] = transformers.StoppingCriteriaList([token_callback])

        # 生成文本前清空缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 确保使用缓存并返回past_key_values
        kwargs['use_cache'] = True
        kwargs['return_dict_in_generate'] = True
        kwargs['output_scores'] = True
        
        # 生成文本

        generation_result = self.model.generate(input_ids=input_ids,
                                              max_new_tokens=max_out_len,
                                              **kwargs)
        
        # 从生成结果中提取outputs和past_key_values
        if hasattr(generation_result, 'sequences'):
            outputs = generation_result.sequences
        else:
            outputs = generation_result
            

            
        # 监控最终的KV缓存
        if hasattr(generation_result, 'past_key_values') and generation_result.past_key_values is not None:
            kv_monitor.update_kv_cache_memory(generation_result.past_key_values)
        
        # GPU计时结束
        end_event.record()
        
        # 等待所有CUDA操作完成
        torch.cuda.synchronize()
        
        peak_memory = token_callback.max_memory_allocated  # MB
        peak_kv_cache_memory = kv_monitor.max_kv_cache_memory  # MB
        
        # 计算GPU上的时间（毫秒）
        total_gpu_time_ms = start_event.elapsed_time(end_event)
        total_gpu_time = total_gpu_time_ms / 1000.0  # 转换为秒
        
        # 计算prefill时间和decode时间
        prefill_time = 0
        decode_time = 0
        if token_callback.prefill_done:
            prefill_time = prefill_start_event.elapsed_time(prefill_end_event) / 1000.0  # 秒
            decode_time = decode_start_event.elapsed_time(end_event) / 1000.0  # 秒
        
        # 计算总CPU+GPU时间
        total_cpu_time = time.time() - cpu_start_time
        
        # 计算输出token数量
        if not self.extract_pred_after_decode:
            output_tokens = outputs[:, input_ids.shape[1]:]
            output_token_count = output_tokens.numel()
        else:
            output_token_count = token_callback.token_count

        # 计算TTFT (如果有第一个token事件)
        ttft_gpu = 0
        if token_callback.first_token_event is not None:
            ttft_gpu = start_event.elapsed_time(token_callback.first_token_event) / 1000.0  # 转换为秒
        
        # 计算TPOT (Time Per Output Token)
        tpot_gpu = 0
        if output_token_count > 1:
            remaining_tokens = output_token_count - 1
            remaining_time = total_gpu_time - ttft_gpu
            tpot_gpu = remaining_time / remaining_tokens

        throughput = output_token_count / total_gpu_time if total_gpu_time > 0 else 0

        # 打印统计信息
        print(f"输入Token数: {input_token_count}")
        print(f"输出Token数: {output_token_count}")
        print(f"总Token数: {input_token_count + output_token_count}")
        print(f"Prefill时间: {prefill_time:.4f}秒")
        print(f"Decode时间: {decode_time:.4f}秒")
        print(f"GPU TTFT (GPU Time to First Token): {ttft_gpu:.4f}秒")
        print(f"GPU TPOT (GPU Time Per Output Token): {tpot_gpu:.4f}秒/token")
        print(f"总GPU时间: {total_gpu_time:.4f}秒")
        print(f"总CPU+GPU时间: {total_cpu_time:.4f}秒")
        print(f"GPU吞吐量: {throughput:.2f} tokens/秒")
        print(f"峰值GPU内存: {peak_memory:.2f} MB")
        print(f"峰值past_key_value缓存内存: {peak_kv_cache_memory:.2f} MB")

        file_exists = os.path.isfile(csv_filename)

        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 写入CSV文件
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'input_tokens', 'output_tokens', 'total_tokens', 
                        'prefill_time', 'decode_time', 'ttft', 'tpot', 'gpu_time', 
                        'cpu_gpu_time', 'throughput', 'peak_memory', 'peak_kv_cache_memory','prefill_kv_cache_memory']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 如果文件不存在，先写入表头
            if not file_exists:
                writer.writeheader()
            
            # 写入数据行
            writer.writerow({
                'timestamp': timestamp,
                'input_tokens': input_token_count,
                'output_tokens': output_token_count,
                'total_tokens': input_token_count + output_token_count,
                'prefill_time': f"{prefill_time:.4f}",
                'decode_time': f"{decode_time:.4f}",
                'ttft': f"{ttft_gpu:.4f}",
                'tpot': f"{tpot_gpu:.4f}",
                'gpu_time': f"{total_gpu_time:.4f}",
                'cpu_gpu_time': f"{total_cpu_time:.4f}",
                'throughput': f"{throughput:.2f}",
                'peak_memory': f"{peak_memory:.2f}",
                'peak_kv_cache_memory': f"{peak_kv_cache_memory:.2f}",
                'prefill_kv_cache_memory': f"{prefill_kv_cache_memory:.2f}",
            })

        if not self.extract_pred_after_decode:
            outputs = outputs[:, input_ids.shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs,
                                            skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        if self.end_str:
            decodeds = [token.split(self.end_str)[0] for token in decodeds]
        if origin_stopping_criteria:
            for t in origin_stopping_criteria:
                decodeds = [token.split(t)[0] for token in decodeds]
        
        return decodeds
    
    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))

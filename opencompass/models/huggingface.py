import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
import time
import csv

import os
from datetime import datetime
PromptType = Union[PromptList, str]


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
class HuggingFace(BaseModel):
    """Model wrapper around HuggingFace models.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        extract_pred_after_decode (bool): Whether to extract the prediction
            string from the decoded output string, instead of extract the
            prediction tokens before decoding. Defaults to False.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
        pad_token_id (int): The id of the padding token. Defaults to None. Use
            (#vocab + pad_token_id) if get negative value.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
        use_fastchat_template (str, optional): Whether to use fastchat to get
            the conversation template. If True, fastchat needs to be
            implemented first. Defaults to False.
        end_str (str, optional): Whether to trim generated strings with end_str
            if the model has special ending strings that are not handled well.
            Defaults to None.

    Note:
        About ``extract_pred_after_decode``: Commonly, we should extract the
        the prediction tokens before decoding. But for some tokenizers using
        ``sentencepiece``, like LLaMA,  this behavior may change the number of
        whitespaces, which is harmful for Python programming tasks.
    """

    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)
        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        assert mode in ['none', 'mid']
        self.mode = mode
        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path,
                             model_kwargs=model_kwargs,
                             peft_path=peft_path)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if self.pad_token_id is not None:
            if self.pad_token_id < 0:
                self.pad_token_id += self.tokenizer.vocab_size
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {self.pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != self.pad_token_id:
                self.logger.warning(
                    'pad_token_id is not consistent with the tokenizer. Using '
                    f'{self.pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = self.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.logger.warning('pad_token_id is not set for the tokenizer.')
            if self.tokenizer.eos_token is not None:
                self.logger.warning(
                    f'Using eos_token_id {self.tokenizer.eos_token} '
                    'as pad_token_id.')
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                from transformers.generation import GenerationConfig
                gcfg = GenerationConfig.from_pretrained(path)

                if gcfg.pad_token_id is not None:
                    self.logger.warning(
                        f'Using pad_token_id {gcfg.pad_token_id} '
                        'as pad_token_id.')
                    self.tokenizer.pad_token_id = gcfg.pad_token_id
                else:
                    raise ValueError(
                        'pad_token_id is not set for this tokenizer. Try to '
                        'set pad_token_id via passing '
                        '`pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path or \
                (tokenizer_path and
                 'decapoda-research/llama' in tokenizer_path):
            self.logger.warning('We set new pad_token_id for LLaMA model')
            # keep consistent with official LLaMA repo
            # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb  # noqa
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token_id = 0

    def _set_model_kwargs_torch_dtype(self, model_kwargs):
        if 'torch_dtype' not in model_kwargs:
            torch_dtype = torch.float16
        else:
            torch_dtype = {
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float': torch.float,
                'auto': 'auto',
                'None': None
            }.get(model_kwargs['torch_dtype'])
        self.logger.debug(f'HF using torch_dtype: {torch_dtype}')
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModel, AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path:
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.
            min_out_len (Optional[int]): The minimum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)
        if self.batch_padding and len(inputs) > 1:
            return self._batch_generate(inputs=inputs,
                                        max_out_len=max_out_len,
                                        min_out_len=min_out_len,
                                        stopping_criteria=stopping_criteria,
                                        **generation_kwargs)
        else:
            return sum(
                (self._single_generate(inputs=[input_],
                                       max_out_len=max_out_len,
                                       min_out_len=min_out_len,
                                       stopping_criteria=stopping_criteria,
                                       **generation_kwargs)
                 for input_ in inputs), [])

    def _batch_generate(self,
                        inputs: List[str],
                        max_out_len: int,
                        min_out_len: Optional[int] = None,
                        stopping_criteria: List[str] = [],
                        **kwargs) -> List[str]:
        """Support for batch prompts inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
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
            for i in range(len(inputs)):
                conv = get_conversation_template('vicuna')
                conv.append_message(conv.roles[0], inputs[i])
                conv.append_message(conv.roles[1], None)
                inputs[i] = conv.get_prompt()

        # step-1: tokenize the input with batch_encode_plus
        tokens = self.tokenizer.batch_encode_plus(inputs,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.max_seq_len)
        tokens = {
            k: torch.tensor(np.array(tokens[k]), device=self.model.device)
            for k in tokens if k in ['input_ids', 'attention_mask']
        }

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
                                          tokens['input_ids'].shape[0])
                    for sequence in stopping_criteria
                ],
            ])
            kwargs['stopping_criteria'] = stopping_criteria

        if min_out_len is not None:
            kwargs['min_new_tokens'] = min_out_len

        # step-2: conduct model forward to generate output
        outputs = self.model.generate(**tokens,
                                      max_new_tokens=max_out_len,
                                      **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, tokens['input_ids'].shape[1]:]

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

            torch.cuda.synchronize()
            print("=========================self.max_seq_len: ", self.max_seq_len)
            print("=========================max_out_len: ", max_out_len)
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
                            'cpu_gpu_time', 'throughput', 'peak_memory', 'peak_kv_cache_memory']
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
    def get_logits(self, inputs: List[str]):

        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(inputs,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_seq_len)

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens)

        else:
            input_ids = self.tokenizer(
                inputs,
                padding=False,
                truncation=True,
                max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            tokens = {'input_ids': input_ids}

            outputs = self.model(input_ids)
        return outputs[0], {'tokens': tokens}

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length)
        else:
            return np.concatenate([
                self._get_ppl(inputs=[text], mask_length=mask_length)
                for text in inputs
            ])

    def _get_ppl(self,
                 inputs: List[str],
                 mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[..., :-1, :].contiguous().float()

        shift_labels = inputs['tokens']['input_ids'][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs['tokens']['input_ids'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def get_loglikelihood(
            self,
            inputs: List[str],
            conts: List[str],
            mask_length: Optional[List[int]] = None) -> List[float]:
        """Get loglikelihood scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            conts (List[str]): A list of strings: slices after the space.
            NOT SUPPORT mask_length YET!
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of loglikelihood scores.
        """
        assert mask_length is None, 'Not support mask_length yet.'
        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_loglikelihood(inputs, conts)
        else:
            return np.concatenate([
                self._get_loglikelihood(inputs=[inputs[idx]],
                                        conts=[conts[idx]])
                for idx in range(len(inputs))
            ])

    def _get_loglikelihood(self, inputs: str, conts: str) -> float:
        """Get loglikelihood scores given input string and continuation string.

        Args:
            inputs (str): string.
            conts (str): strings: slices after the space.
        Returns:
            float: loglikelihood scores.
        """
        input_tokenizer_out = self.tokenizer(inputs,
                                             padding=True,
                                             truncation=False,
                                             return_length=True,
                                             return_tensors='pt').to(
                                                 self.model.device)

        input_ids = input_tokenizer_out['input_ids'][:, :self.max_seq_len]
        input_length = input_tokenizer_out['length']
        context_ids = [
            self.tokenizer(inputs[i].replace(conts[i], ''),
                           padding=False,
                           truncation=True,
                           max_length=self.max_seq_len)['input_ids']
            for i in range(len(inputs))
        ]
        # forward
        outputs = self.model(input_ids)['logits']
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        # calculate loglikelihood
        answer = np.zeros(len(inputs))
        for i in range(len(inputs)):
            if self.tokenizer.padding_side == 'right':
                cont_ids = input_ids[i, len(context_ids[i]):input_length[i]]
                logits = outputs[i,
                                 len(context_ids[i]) - 1:input_length[i] -
                                 1, :]  # noqa
            else:
                cont_ids = input_ids[i, len(context_ids[i]) - input_length[i]:]
                logits = outputs[i,
                                 len(context_ids[i]) - input_length[i] - 1:-1]
            # Reducing the dimension will lead to a wrong outcome
            logits_gather = torch.gather(
                logits.unsqueeze(0), 2,
                cont_ids.unsqueeze(0).unsqueeze(-1))  # [1, seq]
            # Answer: sum the likelihood of each token in continuation
            answer[i] = float(logits_gather.detach().cpu().sum())
        return answer

    def get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        """https://swj0419.github.io/detect-pretrain.github.io/"""

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_mink_percent(inputs, k=k)
        else:
            return np.concatenate([
                self._get_mink_percent(inputs=[text], k=k) for text in inputs
            ])

    def _get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = inputs['tokens']['input_ids'][:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        lens = (inputs['tokens']['input_ids'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        mink_percent = []
        for nloss, nlen in zip(loss, lens):
            nlen = int(nlen)
            minklen = max(nlen * k // 100, 1)
            nloss = torch.topk(loss[-nlen:], minklen, dim=-1)[0]
            nloss = -nloss.float().mean().cpu().detach().numpy()
            mink_percent.append(nloss)
        return np.array(mink_percent)

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))


@MODELS.register_module()
class HuggingFaceCausalLM(HuggingFace):
    """Model wrapper around HuggingFace CausalLM.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
    """

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False


class HuggingFaceChatGLM3(HuggingFace):
    """Model wrapper around HuggingFace's ChatGLM3. Details available in
    `https://huggingface.co/THUDM/chatglm3-6b`.

    model.chat() is used for inference.
    """

    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 num_extra_tokens: int = 50):
        super().__init__(path=path,
                         hf_cache_dir=hf_cache_dir,
                         max_seq_len=max_seq_len,
                         tokenizer_path=tokenizer_path,
                         tokenizer_kwargs=tokenizer_kwargs,
                         peft_path=peft_path,
                         tokenizer_only=tokenizer_only,
                         generation_kwargs=generation_kwargs,
                         model_kwargs=model_kwargs,
                         meta_template=meta_template,
                         extract_pred_after_decode=extract_pred_after_decode,
                         batch_padding=batch_padding,
                         pad_token_id=pad_token_id,
                         mode=mode)
        self.template_parser = APITemplateParser(meta_template)
        # used to compensate for #tokens occupied by sth like system prompt
        self.num_extra_tokens = num_extra_tokens

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512,
                 skip_overlength=False,
                 **kwargs) -> str:
        """Generate response from input prompt.

        Args:
            inputs (list): input prompt
            max_out_len (int): max output length
        """
        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)

        responses = []
        for _input in inputs:
            assert isinstance(_input, (str, PromptList))
            if isinstance(_input, str):
                history = [{'role': 'user', 'content': _input}]
            else:
                history = []
                for item in _input:
                    msg = {
                        'content': item['prompt'],
                        'role': {
                            'HUMAN': 'user',
                            'BOT': 'assistant',
                            'SYSTEM': 'system',
                        }[item['role'].upper()]
                    }
                    history.append(msg)
            user_content = history[-1]['content']
            history = history[:-1]

            if skip_overlength:
                # The model will report the following error
                # if the sequence length is greater than the maximum length:
                # "Input length of input_ids is {INPUT_IDS},
                # but `max_length` is set to 8192.
                # This can lead to unexpected behavior.
                # You should consider increasing `max_new_tokens`."
                # The following hardcode can fix this exception.
                len_user_content = len(self.tokenizer.encode(user_content))
                if len_user_content > 8192:
                    responses.append('')
                    continue

            response, history = self.model.chat(self.tokenizer,
                                                user_content,
                                                history=history,
                                                max_new_tokens=max_out_len,
                                                **generation_kwargs)
            # response will be dict sometime
            if isinstance(response, dict):
                response = response.get('content', '')
            responses.append(response)
        return responses

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt)) + self.num_extra_tokens

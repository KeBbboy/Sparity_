import argparse
import os
from importlib.metadata import version

import torch
import transformers
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, LlamaConfig)
import os
from dataclasses import dataclass, field
from typing import Optional

import transformers
from .cake.utils import CompressConfig
from .infllm_utils import GreedySearch, patch_hf, patch_model_center
from .llama_model.llama_model import (
    adaptive_LlamaModel_forward, llama_attn_forward_cake,
    llama_attn_forward_CAM, llama_attn_forward_H2O, llama_attn_forward_L2Norm,
    llama_attn_forward_PyramidKV, llama_attn_forward_SnapKV,
    llama_attn_forward_StreamingLLM, llama_flash_attn2_forward_AdaKV,
    llama_flash_attn2_forward_CAM, llama_flash_attn2_forward_H2O,
    llama_flash_attn2_forward_HeadKV, llama_flash_attn2_forward_L2Norm,
    llama_flash_attn2_forward_PyramidKV, llama_flash_attn2_forward_SnapKV,
    llama_flash_attn2_forward_StreamingLLM, llama_model_forward_cake,
    llama_sdpa_attn_forward_CAM, llama_sdpa_attn_forward_Flexprefill,
    llama_sdpa_attn_forward_H2O, llama_sdpa_attn_forward_L2Norm,
    llama_sdpa_attn_forward_PyramidKV, llama_sdpa_attn_forward_SnapKV,
    llama_sdpa_attn_forward_SparQ, llama_sdpa_attn_forward_StreamingLLM,
    prepare_inputs_for_generation_llama,llama_sdpa_attn_forward_FULL_KV,qwen_forward,
    prepare_inputs_for_generation_llama_new)
from .llama_model.llama_model_think import (llama_attn_forward_SnapKV_ThinK,
                                think_model_forward)



def replace_llama(self,
                  path=None,
                  model_kwargs=None,
                  model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
                  inputs=None,
                  max_new_tokens=0,
                  past_key_values=0):

    if self.method == 'pyramidkv':
        print('Using PyramidKV!')

        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_PyramidKV
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_PyramidKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_PyramidKV

    elif self.method == 'streamingllm':
        print('Using StreamingLLM!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_StreamingLLM
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_StreamingLLM

    elif self.method == 'h2o':
        print('Using H2O!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_H2O
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_H2O
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_H2O

    elif self.method == 'cam':
        print('Using CAM!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_CAM
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_CAM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_CAM

    elif self.method == 'snapkv':
        print('Using SnapKV!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_SnapKV

    elif self.method == 'minference':
        print('Using MInference!')
        # from .minference0.minference import init_minference, minference_attn_forward
        import sys
        sys.path.insert(0, "/home/yichen/experimental")

        from KVCache.pyramidkv.minference import init_minference, minference_attn_forward
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        init_minference(model_name)
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = minference_attn_forward
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = minference_attn_forward
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = minference_attn_forward

    elif self.method == 'minference1':
        print('Using MInference!')
        import sys
        import json
        from .MInference.minference.patch import new_patch
        from .MInference.minference.minference_configuration import MInferenceConfig
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        current_dir = os.path.dirname(os.path.abspath(__file__))  
        json_path = os.path.join(current_dir, "MInference", "minference", "configs", "Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern.json")
        self.config = MInferenceConfig(
            attn_type="minference",
            model_name=model_name,
            config_path=None,
            starting_layer=-1,
            kv_cache_cpu=False,
            kv_type="dense",
            is_search=False,
            attn_kwargs={}
        )

        with open(json_path, "r") as f:
            self.config.attn_kwargs.setdefault("best_pattern", json.load(f))
        self.model = new_patch(self.model, self.config)
        
    elif self.method == 'pyramidinfer':

        from .pyramidinfer.utils import get_llama_model, load_pyramid_config
        import json
        self.model = get_llama_model(
                path,
                torch_dtype=torch.float16,
                device_map="cuda:0",
                attn_implementation="eager",
                # cache_dir=args.cache_dir,
                load_in_8bit= False,
        )
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        print("Pyramidinfer Model GPU Memory Per GPU (MB): ", f"{torch.cuda.max_memory_allocated(device=self.model.device) / 1024 / 1024:.3f}")
        # pyramid_model = torch.compile(pyramid_model, mode="max-autotune")
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "pyramidinfer", "configs", "llama3_8b.json")
        with open(config_path, "r") as f:
            pyramid_config = json.load(f)
        self.model = load_pyramid_config(self.model, pyramid_config)

    elif self.method == 'l2norm':
        print('Using L2Norm!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_L2Norm
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_L2Norm
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_L2Norm

    elif self.method == 'adakv':
        print('Using AdaKV!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_AdaKV
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_AdaKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_flash_attn2_forward_AdaKV

    elif self.method == 'headkv':
        print('Using HeadKV!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_HeadKV
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_HeadKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_flash_attn2_forward_HeadKV

    elif self.method == 'think':
        print('Using Think!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        transformers.models.llama.modeling_llama.LlamaModel.forward = think_model_forward
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_attn_forward_SnapKV_ThinK


    elif self.method == 'sparq':
        print('Using SparQ!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_SparQ
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SparQ
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_SparQ
    
    elif self.method == 'sparq2':
        print('Using SparQ2!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_SparQ
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SparQ
        # transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_SparQ2


    elif self.method == 'flexprefill':
        print('Using flexprefill!')
        model_kwargs['torch_dtype'] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_Flexprefill

    elif self.method == 'cake':
        print('Using cake!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        compress_config = CompressConfig(True, True)
        gamma = 200
        compress_config.cache_size = 256
        compress_config.window_size = 32
        tau1 = 1.6
        tau2 = 0.4
        hyper = [tau1, tau2, gamma]
        compress_config.hyper = hyper

        config = AutoConfig.from_pretrained(path)
        if hasattr(config, 'num_hidden_layers'):
            layers = config.num_hidden_layers
        for i in range(layers):
            self.model.model.layers[i].self_attn.config.key_size = [
                compress_config.cache_size - compress_config.window_size
            ] * layers
            self.model.model.layers[i].self_attn.config.window_size = [
                compress_config.window_size
            ] * layers
            self.model.model.layers[i].self_attn.config.prefill = [True
                                                                   ] * layers
            self.model.model.layers[i].self_attn.config.decoding_evict = [
                None
            ] * layers
            self.model.model.layers[
                i].self_attn.config.tau1 = compress_config.hyper[0]
            self.model.model.layers[
                i].self_attn.config.tau2 = compress_config.hyper[1]
            self.model.model.layers[
                i].self_attn.config.gamma = compress_config.hyper[2]
            from .cake.cake_cache import CakeprefillKVCache
            self.model.model.layers[i].self_attn.config.prefill_cake_evict = [
                CakeprefillKVCache(
                    cache_size=compress_config.cache_size,
                    window_size=compress_config.window_size,
                    k_seq_dim=2,
                    v_seq_dim=2,
                    num_heads=self.model.model.layers[i].self_attn.num_heads,
                    num_layers=layers,
                    use_cascading=compress_config.cascading)
            ] * layers
        transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_cake
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_attn_forward_cake
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_attn_forward_cake

    elif self.method == 'infllm':
        print('Using infllm!')
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='cuda')
        self.model = patch_hf(self.model, self.infllm_kwargs.type,
                              **self.infllm_kwargs)

    elif self.method == 'quest':
        print('Using quest!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--model', action='append', nargs='+')
        parser.add_argument('--fixed-length', type=int)
        parser.add_argument('--max-tokens', type=int, default=8192)
        parser.add_argument('--min-tokens', type=int, default=256)
        parser.add_argument('--tokens-step', type=int)
        parser.add_argument('--length-step', type=int, default=128)
        parser.add_argument('--iterations', type=int, default=20)
        parser.add_argument('--output-file', type=str)

        parser.add_argument('--quest',action='store_true',help='Enable quest attention')
        parser.add_argument('--token_budget', type=int, default=1024)
        parser.add_argument('--chunk_size', type=int, default=16)
        from .Quest.evaluation.quest_attention import enable_quest_attention_eval
        args = parser.parse_args(
            [])  

        enable_quest_attention_eval(self.model, args)

    elif self.method == 'tova':
        print('Using tova!')
        from .tova.convert_models.convert import enable_tova_caching
        from .tova.convert_models.llama_custom import OLD_LlamaRotaryEmbedding
        from .tova.tova_cache import TOVACache

        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        enable_tova_caching(self.model)

        multi_state_size = 512
        cache = TOVACache(multi_state_size)
        self.past_key_values = cache

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

    elif self.method == 'qfilters':
        print('Using qfilters!')
        from .qfilters.q_cache import KNormCache, QFiltersCache
        model_kwargs['torch_dtype'] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        cache = QFiltersCache(
            self.model,
            window_length=64,
            max_length=128,
            model_name='meta-llama/Llama-3.1-8B-Instruct',
            **model_kwargs,
        )
        self.past_key_values = cache

    elif self.method == 'magicpig':
        print('Using magicpig!')
        from .MagicPIG.models.llama import LLM
        from .MagicPIG.models.template import Templates
        parser = argparse.ArgumentParser()
        parser.add_argument('--model',
                            type=str,
                            default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                            help='model')
        parser.add_argument('--M', type=int, default=5000, help='max length')
        parser.add_argument('--D', type=int, default=1, help='dec length')
        parser.add_argument('--G',
                            type=int,
                            default=256,
                            help='generation length')
        parser.add_argument('--t', type=float, default=0.6, help='temperature')
        parser.add_argument('--K', type=int, default=10, help='K')
        parser.add_argument('--L', type=int, default=75, help='K')
        parser.add_argument('--data',
                            type=str,
                            default='../data/story.txt',
                            help='source data file')
        parser.add_argument('--template',
                            type=str,
                            default='meta-llama3',
                            help='chat template')
        args = parser.parse_args([])
        print(args)
        MAX_LEN = args.M
        DEC_LEN = args.D
        GEN_LEN = args.G
        MODEL_NAME = path
        DTYPE = torch.bfloat16
        DEVICE = 'cuda:0'
        chat_template = Templates[args.template]
        llm = LLM(model=None,
                  K=args.K,
                  L=args.L,
                  max_length=MAX_LEN,
                  model_name=args.model,
                  batch_size=1,
                  device=DEVICE,
                  dtype=DTYPE,
                  generation_buffer=args.G + 32)
        print('inputs', inputs)
        generated = llm.generate(input_ids=inputs,
                                 max_tokens=args.G,
                                 verbose=True,
                                 temperature=args.t)
        return generated

    # elif self.method == 'arkvale':
    #     print('Using arkvale!')
    #     from arkvale import adapter
    #     print('model_kwargs', model_kwargs)
    #     self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
    #     dev = torch.device('cuda:0')
    #     dtype = torch.float16
    #     adapter.enable_arkvale(
    #         self.model,
    #         dtype=dtype,
    #         device=dev,
    #         page_size=self.arkvale_kwargs.page_size,
    #         # page_budgets=None, # page_budgets=None means "full" (no eviction & recall)
    #         page_budgets=self.arkvale_kwargs.page_budgets,
    #         page_topks=self.arkvale_kwargs.page_topks,
    #         n_max_bytes=self.arkvale_kwargs.n_max_bytes,
    #         n_max_cpu_bytes=self.arkvale_kwargs.n_max_cpu_bytes,
    #     )

    elif self.method == 'hipattention':
        print('Using hipattention!')

    elif self.method == 'keyformer':
        print('Using keyformer!')

    elif self.method == 'retrieval':
        print('Using RetrievalAttention!')

    elif self.method == 'full':
        print('Using fullkv!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_FULL_KV

        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)

    elif self.method == 'mustafar':
        print('Using Mustafar!')
        from .mustafar.models.llama_mustafar_Kt_Mag_Vt_Mag import LlamaForCausalLM_MUSTAFAR
        parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        )
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        config.k_sparsity = model_args.k_sparsity
        config.v_sparsity = model_args.v_sparsity
        config.group_size = model_args.group_size
        config.residual_length = model_args.residual_length
        config.use_flash = True
        self.model = LlamaForCausalLM_MUSTAFAR.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                #cache_dir=cache_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
        )


    if self.method not in ['fullkv']:
        print('Using prepare_inputs_for_generation_llama_new!')
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new


def replace_qwen2(self,
                  path=None,
                  model_kwargs=None,
                  model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
                  inputs=None,
                  max_new_tokens=0,
                  past_key_values=0):

    if self.method == 'pyramidkv':
        print('Using PyramidKV!')

        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_PyramidKV
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_PyramidKV
        transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention.forward = llama_sdpa_attn_forward_PyramidKV

    elif self.method == 'streamingllm':
        print('Using StreamingLLM!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_StreamingLLM
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_StreamingLLM

    elif self.method == 'h2o':
        print('Using H2O!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_H2O
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_H2O
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_H2O

    elif self.method == 'cam':
        print('Using CAM!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_CAM
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_CAM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_CAM

    elif self.method == 'snapkv':
        print('Using SnapKV!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
        transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention.forward = llama_sdpa_attn_forward_SnapKV

    elif self.method == 'minference':
        print('Using MInference!')
        # from .minference0.minference import init_minference, minference_attn_forward
        import sys
        sys.path.insert(0, "/home/yichen/experimental")

        from KVCache.pyramidkv.minference import init_minference, minference_attn_forward
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        init_minference(model_name)
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = minference_attn_forward
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = minference_attn_forward
        transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention.forward = minference_attn_forward

    elif self.method == 'minference1':
        print('Using MInference!')
        import sys
        import json
        from .MInference.minference.patch import new_patch
        from .MInference.minference.minference_configuration import MInferenceConfig
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        current_dir = os.path.dirname(os.path.abspath(__file__))  
        json_path = os.path.join(current_dir, "MInference", "minference", "configs", "Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern.json")
        self.config = MInferenceConfig(
            attn_type="minference",
            model_name=model_name,
            config_path=None,
            starting_layer=-1,
            kv_cache_cpu=False,
            kv_type="dense",
            is_search=False,
            attn_kwargs={}
        )

        with open(json_path, "r") as f:
            self.config.attn_kwargs.setdefault("best_pattern", json.load(f))
        self.model = new_patch(self.model, self.config)
        
    elif self.method == 'pyramidinfer':

        from .pyramidinfer.utils import get_llama_model, load_pyramid_config
        import json
        self.model = get_llama_model(
                path,
                torch_dtype=torch.float16,
                device_map="cuda:0",
                attn_implementation="eager",
                # cache_dir=args.cache_dir,
                load_in_8bit= False,
        )
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        print("Pyramidinfer Model GPU Memory Per GPU (MB): ", f"{torch.cuda.max_memory_allocated(device=self.model.device) / 1024 / 1024:.3f}")
        # pyramid_model = torch.compile(pyramid_model, mode="max-autotune")
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "pyramidinfer", "configs", "llama3_8b.json")
        with open(config_path, "r") as f:
            pyramid_config = json.load(f)
        self.model = load_pyramid_config(self.model, pyramid_config)

    elif self.method == 'l2norm':
        print('Using L2Norm!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_L2Norm
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_L2Norm
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_L2Norm

    elif self.method == 'adakv':
        print('Using AdaKV!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_AdaKV
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_AdaKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_flash_attn2_forward_AdaKV

    elif self.method == 'headkv':
        print('Using HeadKV!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_HeadKV
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_HeadKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_flash_attn2_forward_HeadKV

    elif self.method == 'think':
        print('Using Think!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        transformers.models.llama.modeling_llama.LlamaModel.forward = think_model_forward
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_attn_forward_SnapKV_ThinK


    elif self.method == 'sparq':
        print('Using SparQ!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_SparQ
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SparQ
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_SparQ
    
    elif self.method == 'sparq2':
        print('Using SparQ2!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_SparQ
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SparQ
        # transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_SparQ2


    elif self.method == 'flexprefill':
        print('Using flexprefill!')
        model_kwargs['torch_dtype'] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention.forward = llama_sdpa_attn_forward_Flexprefill

    elif self.method == 'cake':
        print('Using cake!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        compress_config = CompressConfig(True, True)
        gamma = 200
        compress_config.cache_size = 256
        compress_config.window_size = 32
        tau1 = 1.6
        tau2 = 0.4
        hyper = [tau1, tau2, gamma]
        compress_config.hyper = hyper

        config = AutoConfig.from_pretrained(path)
        if hasattr(config, 'num_hidden_layers'):
            layers = config.num_hidden_layers
        for i in range(layers):
            self.model.model.layers[i].self_attn.config.key_size = [
                compress_config.cache_size - compress_config.window_size
            ] * layers
            self.model.model.layers[i].self_attn.config.window_size = [
                compress_config.window_size
            ] * layers
            self.model.model.layers[i].self_attn.config.prefill = [True
                                                                   ] * layers
            self.model.model.layers[i].self_attn.config.decoding_evict = [
                None
            ] * layers
            self.model.model.layers[
                i].self_attn.config.tau1 = compress_config.hyper[0]
            self.model.model.layers[
                i].self_attn.config.tau2 = compress_config.hyper[1]
            self.model.model.layers[
                i].self_attn.config.gamma = compress_config.hyper[2]
            from .cake.cake_cache import CakeprefillKVCache
            self.model.model.layers[i].self_attn.config.prefill_cake_evict = [
                CakeprefillKVCache(
                    cache_size=compress_config.cache_size,
                    window_size=compress_config.window_size,
                    k_seq_dim=2,
                    v_seq_dim=2,
                    num_heads=self.model.model.layers[i].self_attn.num_heads,
                    num_layers=layers,
                    use_cascading=compress_config.cascading)
            ] * layers
        transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_cake
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_attn_forward_cake
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_attn_forward_cake

    elif self.method == 'infllm':
        print('Using infllm!')
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='cuda')
        self.model = patch_hf(self.model, self.infllm_kwargs.type,
                              **self.infllm_kwargs)

    elif self.method == 'quest':
        print('Using quest!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--model', action='append', nargs='+')
        parser.add_argument('--fixed-length', type=int)
        parser.add_argument('--max-tokens', type=int, default=8192)
        parser.add_argument('--min-tokens', type=int, default=256)
        parser.add_argument('--tokens-step', type=int)
        parser.add_argument('--length-step', type=int, default=128)
        parser.add_argument('--iterations', type=int, default=20)
        parser.add_argument('--output-file', type=str)

        parser.add_argument('--quest',action='store_true',help='Enable quest attention')
        parser.add_argument('--token_budget', type=int, default=1024)
        parser.add_argument('--chunk_size', type=int, default=16)
        from .Quest.evaluation.quest_attention import enable_quest_attention_eval
        args = parser.parse_args([])  

        enable_quest_attention_eval(self.model, args)

    elif self.method == 'tova':
        print('Using tova!')
        from .tova.convert_models.convert import enable_tova_caching
        from .tova.convert_models.llama_custom import OLD_LlamaRotaryEmbedding
        from .tova.tova_cache import TOVACache

        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        enable_tova_caching(self.model)

        multi_state_size = 512
        cache = TOVACache(multi_state_size)
        self.past_key_values = cache

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

    elif self.method == 'qfilters':
        print('Using qfilters!')
        from .qfilters.q_cache import KNormCache, QFiltersCache
        model_kwargs['torch_dtype'] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        cache = QFiltersCache(
            self.model,
            window_length=64,
            max_length=128,
            model_name='meta-llama/Llama-3.1-8B-Instruct',
            **model_kwargs,
        )
        self.past_key_values = cache

    elif self.method == 'magicpig':
        print('Using magicpig!')
        from .MagicPIG.models.llama import LLM
        from .MagicPIG.models.template import Templates
        parser = argparse.ArgumentParser()
        parser.add_argument('--model',
                            type=str,
                            default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                            help='model')
        parser.add_argument('--M', type=int, default=5000, help='max length')
        parser.add_argument('--D', type=int, default=1, help='dec length')
        parser.add_argument('--G',
                            type=int,
                            default=256,
                            help='generation length')
        parser.add_argument('--t', type=float, default=0.6, help='temperature')
        parser.add_argument('--K', type=int, default=10, help='K')
        parser.add_argument('--L', type=int, default=75, help='K')
        parser.add_argument('--data',
                            type=str,
                            default='../data/story.txt',
                            help='source data file')
        parser.add_argument('--template',
                            type=str,
                            default='meta-llama3',
                            help='chat template')
        args = parser.parse_args([])
        print(args)
        MAX_LEN = args.M
        DEC_LEN = args.D
        GEN_LEN = args.G
        MODEL_NAME = path
        DTYPE = torch.bfloat16
        DEVICE = 'cuda:0'
        chat_template = Templates[args.template]
        llm = LLM(model=None,
                  K=args.K,
                  L=args.L,
                  max_length=MAX_LEN,
                  model_name=args.model,
                  batch_size=1,
                  device=DEVICE,
                  dtype=DTYPE,
                  generation_buffer=args.G + 32)
        print('inputs', inputs)
        generated = llm.generate(input_ids=inputs,
                                 max_tokens=args.G,
                                 verbose=True,
                                 temperature=args.t)
        return generated

    # elif self.method == 'arkvale':
    #     print('Using arkvale!')
    #     from arkvale import adapter
    #     print('model_kwargs', model_kwargs)
    #     self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
    #     dev = torch.device('cuda:0')
    #     dtype = torch.float16
    #     adapter.enable_arkvale(
    #         self.model,
    #         dtype=dtype,
    #         device=dev,
    #         page_size=self.arkvale_kwargs.page_size,
    #         # page_budgets=None, # page_budgets=None means "full" (no eviction & recall)
    #         page_budgets=self.arkvale_kwargs.page_budgets,
    #         page_topks=self.arkvale_kwargs.page_topks,
    #         n_max_bytes=self.arkvale_kwargs.n_max_bytes,
    #         n_max_cpu_bytes=self.arkvale_kwargs.n_max_cpu_bytes,
    #     )

    elif self.method == 'hipattention':
        print('Using hipattention!')

    elif self.method == 'keyformer':
        print('Using keyformer!')

    elif self.method == 'retrieval':
        print('Using RetrievalAttention!')

    elif self.method == 'full':
        print('Using fullkv!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        # transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV
        # transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
        transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention.forward= llama_sdpa_attn_forward_FULL_KV

        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)


    elif self.method == 'qwen':
        print('Using qwen!')
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.model.config.window_size = self.cache_kwargs.window_size
        self.model.config.max_capacity_prompt = self.cache_kwargs.max_capacity_prompt
        transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention.forward= qwen_forward

    if self.method not in ['fullkv']:
        print('Using prepare_inputs_for_generation_llama_new!')
        transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new
        

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )
    k_bits: Optional[int] = field(
        default=2,
        metadata={"help": "KV_cache quantization bits."},
    )
    v_bits: Optional[int] = field(
        default=2,
        metadata={"help": "KV_cache quantization bits."},
    )
    k_sparsity: Optional[float] = field(
        default=0.5,
        metadata={"help": "KV_cache sparsity."},
    )
    v_sparsity: Optional[float] = field(
        default=0.5,
        metadata={"help": "KV_cache sparsity."},
    )
    mode: Optional[str] = field(
        default='mustafar',
        metadata={"help": "Operation mode: 'mustafar', 'dense', 'pruned' or 'kivi(X)'."},
    )
    k_quant_dim: Optional[str] = field(
        default='token',
        metadata={"help": "KV_cache quantization bits."},
    )
    v_quant_dim: Optional[str] = field(
        default='token',
        metadata={"help": "KV_cache quantization bits."},
    )
    group_size: Optional[int] = field(
        default=128,
        metadata={"help": "KV_cache quantization group size."},
    )
    residual_length: Optional[int] = field(
        default=128,
        metadata={"help": "KV_cache residual length."},
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    load_quant: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a quantized model"},
    )
    w_bit: Optional[int] = field(
        default=4,
        metadata={"help": "The model weight bit width."},
    )
    lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LoRA"},
    )
    lora_mode: Optional[str] = field(
        default="q",
        metadata={"help": "LoRA mode"},
    )
    lora_r: Optional[int] = field(
        default=1,
        metadata={"help": "LoRA r"},
    )
    lora_alpha: Optional[float] = field(
        default=1.,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: Optional[float] = field(
        default=0.,
        metadata={"help": "LoRA dropout"},
    )
    


@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default='c4',
        metadata={"help": "The dataset used for fine-tuning the model."},
    )
    eval_tasks: Optional[str] = field(
        default='wikitext',
        metadata={"help": "The dataset used for evaluation."},
    )
    tasks: Optional[str] = field(
        default='wikitext',
        metadata={"help": "The dataset used for evaluation."},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "The batch size."},
    )
    num_fewshot: Optional[int] = field(
        default=0,
        metadata={"help": "The number of fewshot examples."},
    )
    output_path: Optional[str] = field(
        default='./outputs',
        metadata={"help": "The output path."},
    )
    e: Optional[bool] = field(
        default=False,
        metadata={"help": "Evaluate on LongBench-E."},
    )
    use_our_imp: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use our KV cache quantization implementation."},
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="./outputs")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    num_train_epochs: Optional[int] = field(default=1)
    n_train_samples: Optional[int] = field(default=None)
    n_eval_samples: Optional[int] = field(default=None)
    qat: Optional[bool] = field(default=False)
    exp_name: Optional[str] = field(default="test")


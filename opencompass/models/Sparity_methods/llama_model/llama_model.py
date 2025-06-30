import math
import warnings
from tkinter import NO
from typing import List, Optional, Tuple, Union
import os
import time
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (apply_rotary_pos_emb,
                                                      logger, repeat_kv)
from transformers.utils import logging
import os
import pandas as pd
from torch import topk,softmax
from .apply_rope import triton_apply_rotary_pos_emb
from ..cake.cake_cache import CakeCache, CakeDecodingKVCache_LayerWise
from ..cake.utils import calculate_entropy
from .flex_prefill_attention import flex_prefill_attention
from .pyramidkv_utils import (DynamicCacheSplitHeadFlatten, init_adakv,
                              init_CAM, init_H2O, init_headkv, init_l2norm,
                              init_pyramidkv, init_snapkv, init_sparq,
                              init_StreamingLLM)

logger = logging.get_logger(__name__)


def _flash_attention_forward(self,
                             query_states,
                             key_states,
                             value_states,
                             attention_mask,
                             query_length,
                             dropout=0.0,
                             softmax_scale=None):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
    """
    # if not self._flash_attn_uses_top_left_mask:
    #     causal = self.is_causal
    # else:
    # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
    causal = self.is_causal and query_length != 1

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask,
            query_length)

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size,
                                query_length)
    else:
        attn_output = flash_attn_func(query_states,
                                      key_states,
                                      value_states,
                                      dropout,
                                      softmax_scale=softmax_scale,
                                      causal=causal)

    # if self.layer_idx == 0:
    #     import pdb; pdb.set_trace()

    return attn_output


def llama_attn_forward_PyramidKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    init_pyramidkv(self, num_hidden_layers=self.config.num_hidden_layers)

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads *
                             self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }

        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    attn_weights = torch.matmul(query_states, key_states.transpose(
        2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights,
                                         dim=-1,
                                         dtype=torch.float32).to(
                                             query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights,
                                         p=self.attention_dropout,
                                         training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
            f' {attn_output.size()}')

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size //
                                        self.config.pretraining_tp,
                                        dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size //
                                                 self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_sdpa_attn_forward_PyramidKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            'LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, '
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    init_pyramidkv(self, num_hidden_layers=self.config.num_hidden_layers)

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)

            if self.layer_idx == 31:
                estimate_kv_memory(past_key_value)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == 'cuda' and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
    # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def llama_flash_attn2_forward_PyramidKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:

    init_pyramidkv(self, num_hidden_layers=self.config.num_hidden_layers)
    # LlamaFlashAttention2 attention does not support output_attentions
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # print('kv_seq_len:', kv_seq_len)
        # print('key_states.shape:', key_states.shape)

        # print(f"self.layer_idx {self.layer_idx}")
        # print(f"key_states {key_states.device}")
        # print(f"value_states {value_states.device}")

        # if self.layer_idx < len(past_key_value.key_cache):
        #     for index in range(len(past_key_value.key_cache)):
        #         print(f"past_key_value.key_cache[{index}] {past_key_value.key_cache[index].device}")
        #         print(f"past_key_value.value_cache[{index}] {past_key_value.value_cache[index].device}")

        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
        # print(f"after self.key_cache[layer_idx] {past_key_value.key_cache[self.layer_idx].device}")
        # print(f"after self.value_states[layer_idx] {past_key_value.value_cache[self.layer_idx].device}")

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f'The input hidden states seems to be silently casted in float32, this might be related to'
            f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(self,
                                           query_states,
                                           key_states,
                                           value_states,
                                           attention_mask,
                                           q_len,
                                           dropout=dropout_rate)

    attn_output = attn_output.reshape(bsz, q_len,
                                      self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attn_forward_L2Norm(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    init_l2norm(self)

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads *
                             self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }

        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    attn_weights = torch.matmul(query_states, key_states.transpose(
        2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights,
                                         dim=-1,
                                         dtype=torch.float32).to(
                                             query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights,
                                         p=self.attention_dropout,
                                         training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
            f' {attn_output.size()}')

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size //
                                        self.config.pretraining_tp,
                                        dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size //
                                                 self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_sdpa_attn_forward_L2Norm(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            'LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, '
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    init_l2norm(self)

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == 'cuda' and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
    # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def llama_flash_attn2_forward_L2Norm(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:

    init_l2norm(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # print('kv_seq_len:', kv_seq_len)
        # print('key_states.shape:', key_states.shape)

        # print(f"self.layer_idx {self.layer_idx}")
        # print(f"key_states {key_states.device}")
        # print(f"value_states {value_states.device}")

        # if self.layer_idx < len(past_key_value.key_cache):
        #     for index in range(len(past_key_value.key_cache)):
        #         print(f"past_key_value.key_cache[{index}] {past_key_value.key_cache[index].device}")
        #         print(f"past_key_value.value_cache[{index}] {past_key_value.value_cache[index].device}")

        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
        # print(f"after self.key_cache[layer_idx] {past_key_value.key_cache[self.layer_idx].device}")
        # print(f"after self.value_states[layer_idx] {past_key_value.value_cache[self.layer_idx].device}")

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f'The input hidden states seems to be silently casted in float32, this might be related to'
            f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(self,
                                           query_states,
                                           key_states,
                                           value_states,
                                           attention_mask,
                                           q_len,
                                           dropout=dropout_rate)

    attn_output = attn_output.reshape(bsz, q_len,
                                      self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attn_forward_CAM(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    init_CAM(self)

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads *
                             self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }

        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    attn_weights = torch.matmul(query_states, key_states.transpose(
        2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights,
                                         dim=-1,
                                         dtype=torch.float32).to(
                                             query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights,
                                         p=self.attention_dropout,
                                         training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
            f' {attn_output.size()}')

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size //
                                        self.config.pretraining_tp,
                                        dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size //
                                                 self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_sdpa_attn_forward_CAM(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            'LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, '
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    init_CAM(self)

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
            if self.layer_idx == 31:
                estimate_kv_memory(past_key_value)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == 'cuda' and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
    # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def llama_flash_attn2_forward_CAM(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    # [SnapKV] register kv_cluster
    init_CAM(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):  #[SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # print('kv_seq_len:', kv_seq_len)
        # print('key_states.shape:', key_states.shape)
        if key_states.shape[-2] == kv_seq_len:  # [SnapKV] add kv_cluster
            self.kv_seq_len = kv_seq_len  # [SnapKV] register kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f'The input hidden states seems to be silently casted in float32, this might be related to'
            f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(self,
                                           query_states,
                                           key_states,
                                           value_states,
                                           attention_mask,
                                           q_len,
                                           dropout=dropout_rate)

    attn_output = attn_output.reshape(bsz, q_len,
                                      self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attn_forward_H2O(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    init_H2O(self)

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads *
                             self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }

        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    attn_weights = torch.matmul(query_states, key_states.transpose(
        2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights,
                                         dim=-1,
                                         dtype=torch.float32).to(
                                             query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights,
                                         p=self.attention_dropout,
                                         training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
            f' {attn_output.size()}')

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size //
                                        self.config.pretraining_tp,
                                        dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size //
                                                 self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_sdpa_attn_forward_H2O(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            'LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, '
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    init_H2O(self)
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
            
        else:
            # print("key_states.shape[-2]",key_states.shape[-2])
            # print("kv_seq_len",kv_seq_len)
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
            # print("key_states,value_states",key_states.shape, value_states.shape)
        past_key_value._seen_tokens = self.kv_seq_len

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == 'cuda' and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
    # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    # print("attn_output.shape",attn_output.shape)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def llama_flash_attn2_forward_H2O(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    # [SnapKV] register kv_cluster
    # print("===============replace successfully ===================")
    init_H2O(self)

    # LlamaFlashAttention2 attention does not support output_attentions
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    output_attentions = False
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):  #[SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # print('kv_seq_len:', kv_seq_len)
        # print('key_states.shape:', key_states.shape)
        if key_states.shape[-2] == kv_seq_len:  # [SnapKV] add kv_cluster
            self.kv_seq_len = kv_seq_len  # [SnapKV] register kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f'The input hidden states seems to be silently casted in float32, this might be related to'
            f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(self,
                                           query_states,
                                           key_states,
                                           value_states,
                                           attention_mask,
                                           q_len,
                                           dropout=dropout_rate)

    attn_output = attn_output.reshape(bsz, q_len,
                                      self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attn_forward_StreamingLLM(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    init_StreamingLLM(self)

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads *
                             self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }

        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    attn_weights = torch.matmul(query_states, key_states.transpose(
        2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights,
                                         dim=-1,
                                         dtype=torch.float32).to(
                                             query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights,
                                         p=self.attention_dropout,
                                         training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
            f' {attn_output.size()}')

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size //
                                        self.config.pretraining_tp,
                                        dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size //
                                                 self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_sdpa_attn_forward_StreamingLLM(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            'LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, '
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    init_StreamingLLM(self)

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == 'cuda' and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
    # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def llama_flash_attn2_forward_StreamingLLM(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    # [SnapKV] register kv_cluster
    init_StreamingLLM(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):  #[SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # print('kv_seq_len:', kv_seq_len)
        # print('key_states.shape:', key_states.shape)
        if key_states.shape[-2] == kv_seq_len:  # [SnapKV] add kv_cluster
            self.kv_seq_len = kv_seq_len  # [SnapKV] register kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f'The input hidden states seems to be silently casted in float32, this might be related to'
            f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(self,
                                           query_states,
                                           key_states,
                                           value_states,
                                           attention_mask,
                                           q_len,
                                           dropout=dropout_rate)

    attn_output = attn_output.reshape(bsz, q_len,
                                      self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attn_forward_SnapKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    print("===============replace successfully ===================")
    init_snapkv(self)

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads *
                             self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }

        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    attn_weights = torch.matmul(query_states, key_states.transpose(
        2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights,
                                         dim=-1,
                                         dtype=torch.float32).to(
                                             query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights,
                                         p=self.attention_dropout,
                                         training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
            f' {attn_output.size()}')

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size //
                                        self.config.pretraining_tp,
                                        dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size //
                                                 self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

# 例如：每层有一个 list 或 tensor 的压缩 KV
def estimate_kv_memory(past_key_value, csv_file="kv_mem_log.csv") -> float:
    total_kv_memory = 0

    key_cache = past_key_value.key_cache
    value_cache = past_key_value.value_cache

    print(f"KV dtype: {key_cache[0].dtype}")
    print(f"key_cache层数: {len(key_cache)}")

    for i, (k, v) in enumerate(zip(key_cache, value_cache)):
        if k is not None and v is not None:
            key_shape = k.shape
            value_shape = v.shape
            key_numel = k.numel()
            val_numel = v.numel()
            key_mem = key_numel * k.element_size()
            val_mem = val_numel * v.element_size()
            layer_mem = (key_mem + val_mem) / (1024 ** 2)

            print(f"[Layer {i}] key_cache shape: {key_shape}, value_cache shape: {value_shape}")
            print(f"  ↳ key.numel(): {key_numel}, value.numel(): {val_numel}")
            print(f"  ↳ key mem: {key_mem / (1024 ** 2):.2f} MB, value mem: {val_mem / (1024 ** 2):.2f} MB")

            total_kv_memory += key_mem + val_mem

    kv_mem_MB = total_kv_memory / (1024 ** 2)
    print(f"[KV Cache] 当前past_key_value占用内存: {kv_mem_MB:.2f} MB")

    # ===== ✅ 写入 CSV（自动创建） =====
    header = ["timestamp", "kv_memory_MB"]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{kv_mem_MB:.2f}"
    ]
    need_header = not os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow(row)
    # ==================================

    return kv_mem_MB



def llama_sdpa_attn_forward_SnapKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            'LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, '
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    
    init_snapkv(self)

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)

            
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)
            if self.layer_idx == 23:
                estimate_kv_memory(past_key_value)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == 'cuda' and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
    # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def llama_flash_attn2_forward_SnapKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    # [SnapKV] register kv_cluster
    print("===============replace successfully ===================")
    init_snapkv(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):  #[SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # print('kv_seq_len:', kv_seq_len)
        # print('key_states.shape:', key_states.shape)
        if key_states.shape[-2] == kv_seq_len:  # [SnapKV] add kv_cluster
            self.kv_seq_len = kv_seq_len  # [SnapKV] register kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress,self.layer_idx, cache_kwargs)
            
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f'The input hidden states seems to be silently casted in float32, this might be related to'
            f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(self,
                                           query_states,
                                           key_states,
                                           value_states,
                                           attention_mask,
                                           q_len,
                                           dropout=dropout_rate)

    attn_output = attn_output.reshape(bsz, q_len,
                                      self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_flash_attn2_forward_AdaKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:

    init_adakv(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models

        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)

            # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
            # to be able to avoid many of these transpose/reshape/view.
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            dropout_rate = self.attention_dropout if self.training else 0.0

            # In PEFT, usually we cast the layer norms in float32 for training stability reasons
            # therefore the input hidden states gets silently casted in float32. Hence, we need
            # cast them back in the correct dtype just to be sure everything works as expected.
            # This might slowdown training & inference so it is recommended to not cast the LayerNorms
            # in fp32. (LlamaRMSNorm handles it correctly)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, '_pre_quantization_dtype'):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f'The input hidden states seems to be silently casted in float32, this might be related to'
                    f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
                    f' {target_dtype}.')

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)

            attn_output = _flash_attention_forward(self,
                                                   query_states,
                                                   key_states,
                                                   value_states,
                                                   attention_mask,
                                                   q_len,
                                                   dropout=dropout_rate)

            attn_output = attn_output.reshape(bsz, q_len,
                                              self.hidden_size).contiguous()
            # attn_output = self.o_proj(attn_output)
            if self.layer_idx == 31:
                estimate_kv_memory(past_key_value)
        else:
            # modified by AdaKV
            self.kv_seq_len += q_len

            cache_kwargs['head_lens'] = self.kv_cluster.head_lens
            cache_kwargs['cu_klen'] = self.kv_cluster.cu_klen
            # ipdb.set_trace()
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

            # NOTE: update meta data
            self.kv_cluster.klen_sum += self.num_heads
            self.kv_cluster.max_seqlen_k += 1
            self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
            self.kv_cluster.head_lens += 1

            query_states = query_states.view(-1, 1, self.head_dim)
            key_states = key_states.view(-1, 1, self.head_dim)
            value_states = value_states.view(-1, 1, self.head_dim)

            cu_seqlens_q = self.kv_cluster.cu_qlen
            cu_seqlens_k = self.kv_cluster.cu_klen
            max_seqlen_q = 1
            max_seqlen_k = self.kv_cluster.max_seqlen_k

            attn_output = flash_attn_varlen_func(query_states,
                                                 key_states,
                                                 value_states,
                                                 cu_seqlens_q,
                                                 cu_seqlens_k,
                                                 max_seqlen_q,
                                                 max_seqlen_k,
                                                 causal=True).reshape(
                                                     bsz, self.num_heads,
                                                     q_len, self.head_dim)
            attn_output = attn_output.transpose(0, 1).reshape(
                bsz, q_len, self.hidden_size)

        past_key_value._seen_tokens = self.kv_seq_len

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_flash_attn2_forward_HeadKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:

    init_headkv(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models

        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states)
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)

            # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
            # to be able to avoid many of these transpose/reshape/view.
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            dropout_rate = self.attention_dropout if self.training else 0.0

            # In PEFT, usually we cast the layer norms in float32 for training stability reasons
            # therefore the input hidden states gets silently casted in float32. Hence, we need
            # cast them back in the correct dtype just to be sure everything works as expected.
            # This might slowdown training & inference so it is recommended to not cast the LayerNorms
            # in fp32. (LlamaRMSNorm handles it correctly)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, '_pre_quantization_dtype'):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f'The input hidden states seems to be silently casted in float32, this might be related to'
                    f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
                    f' {target_dtype}.')

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)

            attn_output = _flash_attention_forward(self,
                                                   query_states,
                                                   key_states,
                                                   value_states,
                                                   attention_mask,
                                                   q_len,
                                                   dropout=dropout_rate)

            attn_output = attn_output.reshape(bsz, q_len,
                                              self.hidden_size).contiguous()
            # attn_output = self.o_proj(attn_output)
        else:
            # modified by AdaKV
            self.kv_seq_len += q_len

            cache_kwargs['head_lens'] = self.kv_cluster.head_lens
            cache_kwargs['cu_klen'] = self.kv_cluster.cu_klen
            # ipdb.set_trace()
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

            # NOTE: update meta data
            self.kv_cluster.klen_sum += self.num_heads
            self.kv_cluster.max_seqlen_k += 1
            self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
            self.kv_cluster.head_lens += 1

            query_states = query_states.view(-1, 1, self.head_dim)
            key_states = key_states.view(-1, 1, self.head_dim)
            value_states = value_states.view(-1, 1, self.head_dim)

            cu_seqlens_q = self.kv_cluster.cu_qlen
            cu_seqlens_k = self.kv_cluster.cu_klen
            max_seqlen_q = 1
            max_seqlen_k = self.kv_cluster.max_seqlen_k

            attn_output = flash_attn_varlen_func(query_states,
                                                 key_states,
                                                 value_states,
                                                 cu_seqlens_q,
                                                 cu_seqlens_k,
                                                 max_seqlen_q,
                                                 max_seqlen_k,
                                                 causal=True).reshape(
                                                     bsz, self.num_heads,
                                                     q_len, self.head_dim)
            attn_output = attn_output.transpose(0, 1).reshape(
                bsz, q_len, self.hidden_size)

        past_key_value._seen_tokens = self.kv_seq_len

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


from transformers.models.llama.modeling_llama import StaticCache

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """Creates a causal 4D mask of shape `(batch_size, 1, query_length,
    key_value_length)` from a 2D mask of shape `(batch_size,
    key_value_length)`, or if the input `attention_mask` is already 4D, do
    nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length),
                                 fill_value=min_dtype,
                                 dtype=dtype,
                                 device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None,
                                  None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone(
            )  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :
                                       mask_length] + attention_mask[:, None,
                                                                     None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :
                        mask_length] = causal_mask[:, :, :, :
                                                   mask_length].masked_fill(
                                                       padding_mask, min_dtype)

    return causal_mask


def prepare_inputs_for_generation_llama_new(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    if not isinstance(past_key_values, tuple):
        if len(past_key_values.key_cache) == 0:
            for layer in self.model.layers:
                layer.self_attn.kv_seq_len = 0

    # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
    # Exception 1: when passing input_embeds, input_ids may be missing entries
    # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
    if past_key_values is not None:
        if inputs_embeds is not None:  # Exception 1
            input_ids = input_ids[:, -cache_position.shape[0]:]
        elif input_ids.shape[1] != cache_position.shape[
                0]:  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

            # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
            position_ids = position_ids.clone(
                memory_format=torch.contiguous_format)

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {'inputs_embeds': inputs_embeds, 'input_ids': None}
    else:
        # The clone here is for the same reason as for `position_ids`.
        model_inputs = {
            'input_ids':
            input_ids.clone(memory_format=torch.contiguous_format),
            'inputs_embeds': None
        }

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs['inputs_embeds'] is not None:
            batch_size, sequence_length, _ = model_inputs[
                'inputs_embeds'].shape
            device = model_inputs['inputs_embeds'].device
        else:
            batch_size, sequence_length = model_inputs['input_ids'].shape
            device = model_inputs['input_ids'].device

        dtype = self.lm_head.weight.dtype
        min_dtype = torch.finfo(dtype).min

        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=past_key_values.get_max_length(),
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=batch_size,
        )

    # import pdb;pdb.set_trace()

    model_inputs.update({
        'position_ids': position_ids,
        'cache_position': cache_position,
        'past_key_values': past_key_values,
        'use_cache': use_cache,
        'attention_mask': attention_mask,
    })
    return model_inputs


def prepare_inputs_for_generation_llama(self,
                                        input_ids,
                                        past_key_values=None,
                                        attention_mask=None,
                                        inputs_embeds=None,
                                        **kwargs):

    if past_key_values is None or len(past_key_values.key_cache) == 0:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            # cache_length = past_length = past_key_values[0][0].shape[2]
            # max_cache_length = None
            cache_length = past_length = self.model.layers[
                0].self_attn.kv_seq_len
            max_cache_length = None
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[
                1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (max_cache_length is not None and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:

            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update({
        'position_ids': position_ids,
        'past_key_values': past_key_values,
        'use_cache': kwargs.get('use_cache'),
        'attention_mask': attention_mask,
    })
    return model_inputs


#! For AdaKV and HeadKV
def adaptive_LlamaModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (output_hidden_states if output_hidden_states
                            is not None else self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            'You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one'
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.'
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if (use_cache
            and not (type(past_key_values) == DynamicCacheSplitHeadFlatten)
            and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
        # return_legacy_cache = True  #! For 4.41 version.
        past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(
            past_key_values)
        logger.warning_once(
            'We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. '
            'Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)'
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length(
        ) if past_key_values is not None else 0
        cache_position = torch.arange(past_seen_tokens,
                                      past_seen_tokens +
                                      inputs_embeds.shape[1],
                                      device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds,
                                           cache_position, past_key_values,
                                           output_attentions)
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1], )

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states, )

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(
            v for v in
            [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def create_causal_mask(seq_length, device=None, dtype=torch.float):
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device),
                      diagonal=1)
    mask = mask.masked_fill(mask == 1,
                            float('-inf')).masked_fill(mask == 0, 0.0)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.to(dtype)


def llama_sdpa_attn_forward_SparQ(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            'LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, '
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    init_snapkv(self)

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)


    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            past_key_value.update(key_states, value_states,self.layer_idx, cache_kwargs)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        else:
            past_key_value._seen_tokens = self.kv_seq_len
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # key_states = repeat_kv(key_states, self.num_key_value_groups)
            # value_states = repeat_kv(value_states, self.num_key_value_groups)
            batch_size = query_states.shape[0]
            n_heads =  query_states.shape[1]
            n_kv_heads = key_states.shape[1]
            head_size =  key_states.shape[3]
            seuqence = key_states.shape[2]

            query_states = query_states.reshape(batch_size,n_kv_heads, n_heads //  n_kv_heads, 1, head_size)
            key_states = key_states.reshape(batch_size, n_kv_heads, 1, seuqence,head_size)
            value_states = value_states.reshape(batch_size, n_kv_heads, 1, seuqence, head_size)
            # =========================== 进行sparq =====================
            Q = query_states
            K = key_states
            V = value_states


            r = self.config.window_size
            k = self.config.max_capacity_prompt

            i1 = topk(abs(Q).sum(dim=2, keepdim=True), r, -1).indices

            
            # 内联gather: Q_hat = gather(Q, -1, i1)
            dim_Q = -1 + (-1 < 0) * Q.ndim
            Q_hat = Q.gather(dim_Q, i1.expand(*Q.shape[:dim_Q], i1.shape[dim_Q], *Q.shape[dim_Q + 1:]))
            
            # 内联gather: K_hat = gather(K, -1, i1)
            dim_K = -1 + (-1 < 0) * K.ndim
            K_hat = K.gather(dim_K, i1.expand(*K.shape[:dim_K], i1.shape[dim_K], *K.shape[dim_K + 1:]))
            
            scale = torch.sqrt(
                Q.shape[-1]
                * abs(Q_hat).sum(dim=-1, keepdim=True)
                / abs(Q).sum(dim=-1, keepdim=True)
            )

            # 创建基本的因果掩码 (下三角矩阵)
            causal_mask = torch.tril(torch.ones(seuqence, seuqence))
            causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf'))
            last_token_mask = causal_mask[-1:, :]  # 形状: [1, seq_len]

            expanded_mask = last_token_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            M = expanded_mask.expand(batch_size, n_kv_heads,  n_heads // n_kv_heads, 1, seuqence)
            M = M.to(Q_hat.device)

            s_hat = softmax(Q_hat @ K_hat.transpose(-1, -2) / scale + M, dim=-1)
            # (batch_size, n_kv_heads, n_heads // n_kv_heads, 1, sequence)

            
            
            # # 2. Gather top k positions based on approximate attention scores
            actual_k = min(k, s_hat.shape[-1])  # 确保不会越界
            i2 = topk(s_hat.sum(dim=2, keepdim=True), actual_k, -1).indices
            iKV = i2[..., 0, :, None]

            
            # # 内联gather: K_top = gather(K, -2, iKV)
            dim_K_top = -2 + (-2 < 0) * K.ndim
            K_top = K.gather(dim_K_top, iKV.expand(*K.shape[:dim_K_top], iKV.shape[dim_K_top], *K.shape[dim_K_top + 1:]))
            
            # # 内联gather: V_top = gather(V, -2, iKV)
            dim_V_top = -2 + (-2 < 0) * V.ndim
            V_top = V.gather(dim_V_top, iKV.expand(*V.shape[:dim_V_top], iKV.shape[dim_V_top], *V.shape[dim_V_top + 1:]))
            
            # # 内联gather: M_top = gather(M, -1, i2)
            dim_M = -1 + (-1 < 0) * M.ndim
            M_top = M.gather(dim_M, i2.expand(*M.shape[:dim_M], i2.shape[dim_M], *M.shape[dim_M + 1:]))
            
            # # 内联attn函数
            s = (Q @ K_top.transpose(-1, -2)) / torch.sqrt(torch.tensor(Q.shape[-1], device=Q.device)) + M_top
            y_ = torch.softmax(s.to(V_top.dtype), dim=-1) @ V_top

            
            # # 3. Estimate the total score of the top k, and interpolate with V_mean
            # # 内联gather: alpha = gather(s_hat, -1, i2)
            dim_s_hat = -1 + (-1 < 0) * s_hat.ndim
            alpha = s_hat.gather(dim_s_hat, i2.expand(*s_hat.shape[:dim_s_hat], i2.shape[dim_s_hat], *s_hat.shape[dim_s_hat + 1:])).sum(-1, keepdim=True)
            
            V_mean = V.mean(dim=-2, keepdim=True)
            attn_output = alpha * y_ + (1 - alpha) * V_mean

            batch_size, n_heads, n_kv_heads, seq_len, head_size = attn_output.shape
            attn_output = attn_output.reshape(batch_size, n_kv_heads * n_heads, seq_len, head_size)  # -> (batch_size, n_heads, head_size)

            
            

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, seq_len, self.hidden_size)

            attn_output = attn_output.to(self.o_proj.weight.dtype)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

        past_key_value._seen_tokens = self.kv_seq_len

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == 'cuda' and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
    # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
    
def llama_sdpa_attn_forward_Flexprefill(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:

    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            '`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` '
            'make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers'
        )

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    if triton.__version__ == '3.0.0':
        query_states, key_states = triton_apply_rotary_pos_emb(
            query_states, key_states, cos, sin)
    else:
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f'The input hidden states seems to be silently casted in float32, this might be related to'
            f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # "block_size": 128,
    # "flex_prefill_gamma": 0.9,
    # "flex_prefill_tau": 0.1,
    # "flex_prefill_min_budget": 64,
    # "flex_prefill_max_budget": 256,

    # you can use flex_prefill_attention both in prefilling and decoding, which will use full attention while in decoding phase
    block_size = 32
    flex_prefill_min_budget = 64
    flex_prefill_max_budget = 4096
    
    if query_states.shape[-3] > 1:
        if self.layer_idx == 30:
            estimate_kv_memory(past_key_value)
            print("query_states", query_states.shape)
            print("key_states", key_states.shape)
            print("value_states", value_states.shape)
            torch.cuda.synchronize()
            Flexstart_time = time.time()
            attn_output = flex_prefill_attention(
                query_states,
                key_states,
                value_states,
                gamma=0.9,
                tau=0.1,
                block_size=block_size,
                min_budget=flex_prefill_min_budget,
                max_budget=flex_prefill_max_budget,
            )
            torch.cuda.synchronize()
            Flexend_time = time.time()
            flex_time = Flexend_time - Flexstart_time
            print("Flex prefill time: ", flex_time)
            torch.cuda.synchronize()
            start_time = time.time()
            num_key_value_groups = 4 # 计算重复次数，这里是4
            query_states = query_states.permute(0, 2, 1, 3)  # -> [B, H, L, D]
            key_states = key_states.permute(0, 2, 1, 3)      # -> [B, H, L, D]
            value_states = value_states.permute(0, 2, 1, 3)  # -> [B, H, L, D]
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)
            standard_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
            torch.cuda.synchronize()
            end_time = time.time()
            standard_time = end_time - start_time
            print("prefill time: ", standard_time)

            header = ['flex_prefill_time', 'standard_prefill_time']
            row = [flex_time, standard_time]

            csv_file = "prefill_timing.csv"
            need_header = not os.path.exists(csv_file)  # 如果文件不存在则需要写表头

            # 写入 CSV 文件
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                if need_header:
                    writer.writerow(header)
                writer.writerow(row)





        else:
            attn_output = flex_prefill_attention(
                query_states,
                key_states,
                value_states,
                gamma=0.9,
                tau=0.1,
                block_size=block_size,
                min_budget=flex_prefill_min_budget,
                max_budget=flex_prefill_max_budget,
            )
    else:
        attn_output = flex_prefill_attention(
            query_states,
            key_states,
            value_states,
            gamma=0.9,
            tau=0.1,
            block_size=block_size,
            min_budget=flex_prefill_min_budget ,
            max_budget=flex_prefill_max_budget,
        )
        

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attn_forward_cake(
        
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:

    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            '`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` '
            'make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers'
        )
    if isinstance(past_key_value, DynamicCache):
        past_key_value = CakeCache.from_dynamic_cache(past_key_value)

    if self.config.decoding_evict[self.layer_idx] is None and len(
            past_key_value.layer_budget) == self.config.prefill_cake_evict[
                self.layer_idx].num_layers:
        self.config.decoding_evict[
            self.layer_idx] = CakeDecodingKVCache_LayerWise(
                hh_size=past_key_value.layer_budget[self.layer_idx],
                window_size=self.config.window_size[self.layer_idx],
                k_seq_dim=2,
                v_seq_dim=2)
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    # # Flash attention requires the input to have the shape
    # # batch_size x seq_length x head_dim x hidden_dim
    # # therefore we just need to keep the original shape
    # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    if self.config.prefill[self.layer_idx]:
        tmp_attn_weights = torch.matmul(
            query_states[..., -self.config.window_size[self.layer_idx]:, :],
            key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if q_len != 1:
            mask = torch.full((self.config.window_size[self.layer_idx],
                               self.config.window_size[self.layer_idx]),
                              torch.finfo(tmp_attn_weights.dtype).min,
                              device=tmp_attn_weights.device)
            mask_cond = torch.arange(mask.size(-1),
                                     device=tmp_attn_weights.device)
            mask.masked_fill_(
                mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(tmp_attn_weights.device)
            tmp_attention_mask = mask[None, None, :, :]

            tmp_attn_weights[:, :, -self.config.window_size[self.layer_idx]:,
                             -self.config.window_size[
                                 self.layer_idx]:] += tmp_attention_mask

        tmp_attn_weights = nn.functional.softmax(tmp_attn_weights,
                                                 dim=-1,
                                                 dtype=torch.float32).to(
                                                     query_states.dtype)
        # tmp_attn_weights = nn.functional.softmax(tmp_attn_weights, dim=-1, dtype=torch.float64)

        disp = calculate_entropy(
            tmp_attn_weights[:, :,
                             -self.config.window_size[self.layer_idx]:, :-self.
                             config.window_size[self.layer_idx]])
        var = torch.var(
            tmp_attn_weights[:, :,
                             -self.config.window_size[self.layer_idx]:, :-self.
                             config.window_size[self.layer_idx]],
            dim=-2).sum(0).sum(0).sum(0)

        pref_score = (disp**(1 / self.config.tau1) *
                      var**(1 / self.config.tau2)).cpu().numpy()

        #compute preference score and hh score
        attention_score = tmp_attn_weights[:, :, -self.config.
                                           window_size[self.layer_idx]:, :]

        attn_mean = attention_score.mean(dim=-2)
        attn_var = attention_score.var(dim=-2)
        attn_cache = attn_mean + self.config.gamma * attn_var
        attn_cache = attn_cache[:, :, :-self.config.window_size[self.
                                                                layer_idx]]
        attn_cache = F.avg_pool1d(attn_cache,
                                  kernel_size=5,
                                  padding=5 // 2,
                                  stride=1)

        attn_cache = attn_cache.reshape(bsz, self.num_key_value_heads,
                                        self.num_key_value_groups, -1)
        hh_score = attn_cache.mean(dim=-2)
        past_key_value.update_score(pref_score, hh_score)

        past_key_value.layer_budget.append(
            self.config.key_size[self.layer_idx])
        self.config.prefill[self.layer_idx] = False
        past_key_value = self.config.prefill_cake_evict[self.layer_idx](
            past_key_value, q_len)

    if self.config.decoding_evict[self.layer_idx] is not None:

        tmp_attn_weights = torch.matmul(
            query_states[..., -self.config.window_size[self.layer_idx]:, :],
            key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        tmp_attn_weights = nn.functional.softmax(tmp_attn_weights,
                                                 dim=-1,
                                                 dtype=torch.float32).to(
                                                     query_states.dtype)

        past_key_value = self.config.decoding_evict[self.layer_idx](
            past_key_value, tmp_attn_weights, self.layer_idx)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f'The input hidden states seems to be silently casted in float32, this might be related to'
            f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        self=self,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        query_length=q_len,
        dropout=dropout_rate,
        # sliding_window=getattr(self, 'sliding_window', None),
        # use_top_left_mask=self._flash_attn_uses_top_left_mask,
        # is_causal=self.is_causal,
        # dropout_rate,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_model_forward_cake(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (output_hidden_states if output_hidden_states
                            is not None else self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            'You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one'
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.'
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if (use_cache and not isinstance(past_key_values, Cache)
            and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        logger.warning_once(
            'We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. '
            'Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)'
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length(
        ) if past_key_values is not None else 0
        cache_position = torch.arange(past_seen_tokens,
                                      past_seen_tokens +
                                      inputs_embeds.shape[1],
                                      device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds,
                                           cache_position, past_key_values,
                                           output_attentions)
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            past_key_values = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1], )

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states, )

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(
            v for v in
            [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )



def llama_sdpa_attn_forward_FULL_KV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            'LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, '
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')
        
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            
            past_key_value.update(key_states, value_states,
                                  self.layer_idx, cache_kwargs)
            if self.layer_idx == 23:
                estimate_kv_memory(past_key_value)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == 'cuda' and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
    # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def qwen_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
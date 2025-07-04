from transformers.models.llama.modeling_llama import *
from minference.configs.model2path import MODEL2PATH
import json
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from flash_attn import flash_attn_func
from datetime import datetime
import csv

minference_config = None

def init_minference(model_name):
    config_path = MODEL2PATH[model_name]
    global minference_config
    minference_config = json.load(open(config_path))

def estimate_kv_memory(past_key_value,csv_file="kv_mem_log.csv") -> float:
    total_kv_memory = 0
    print(f"key_cache层数: {len(past_key_value.key_cache)}")
    print(f"key_cache[0].shape: {past_key_value.key_cache[0].shape}")
    print(f"value_cache [0].shape: {past_key_value.value_cache[0].shape}")
    for i in range(31):
        key_shape = past_key_value.key_cache[i].shape if past_key_value.key_cache[i] is not None else None
        value_shape = past_key_value.value_cache[i].shape if past_key_value.value_cache[i] is not None else None
        print(f"[Layer {i}] key_cache shape: {key_shape}, value_cache shape: {value_shape}")

        if key_shape and value_shape:
            key_numel = past_key_value.key_cache[i].numel()
            val_numel = past_key_value.value_cache[i].numel()
            key_mem = past_key_value.key_cache[i].numel() * past_key_value.key_cache[i].element_size()
            val_mem = past_key_value.value_cache[i].numel() * past_key_value.value_cache[i].element_size()
             # 新增打印 numel 信息
            print(f"  ↳ key.numel(): {key_numel}, value.numel(): {val_numel}")
            print(f"  ↳ key mem: {key_mem / (1024 ** 2):.2f} MB, value mem: {val_mem / (1024 ** 2):.2f} MB")
            total_kv_memory += key_mem + val_mem
    print(f"KV dtype: {past_key_value.key_cache[0].dtype}")

    try:
        if hasattr(past_key_value, 'key_cache') and hasattr(past_key_value, 'value_cache'):
            key_cache = past_key_value.key_cache
            value_cache = past_key_value.value_cache

            for k, v in zip(key_cache, value_cache):
                if k is not None and v is not None:
                    key_mem = k.numel() * k.element_size()
                    val_mem = v.numel() * v.element_size()
                    total_kv_memory += key_mem + val_mem
    except Exception as e:
        print(f"[KV估算失败] {e}")
    kv_mem = total_kv_memory / (1024 ** 2)
    print(f"[KV Cache] 当前past_key_value占用内存: {kv_mem:.2f} MB")

    # ===== ✅ 写入 CSV（自动创建） =====
    header = ["timestamp",  "kv_memory_MB"]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{kv_mem:.2f}"
    ]

    need_header = not os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow(row)
    # ==================================
    return total_kv_memory / (1024 ** 2)


def minference_prefill_kernel(
    q, k, v, head_id, layer_idx,
    config,
):
    head_dim = q.size(-1)
    def vertical_and_slash_kernel(q, k, v, vertical_size, slash_size):
        vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
        last_q = min(64, q_len)
        qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:,:,-last_q:,:], k) / math.sqrt(head_dim)
        qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[...,-last_q:,-last_q:].to(q.device), qk[:, :, :, -last_q:], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[...,:30] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
        slash[...,-100:] = torch.inf
        slash_topk = slash
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)

    def block_sparse_kernel(q, k, v, vertical_size=None, slash_size=None):
        topk = 100
        return block_sparse_attention(q, k, v, topk)

    q_len = q.shape[2]
    ty, vertical_size, slash_size, _ = config["best_pattern"][layer_idx].get(str(head_id), ("vertical_and_slash", 1000, 6096, 1))

    if "minference_ratio" in config:
        vertical_size = int(vertical_size * config.get("minference_ratio", 1))
        slash_size = int(slash_size * config.get("minference_ratio", 1))
    fc = {
        "stream_llm": streaming_forward,
        "vertical_and_slash": vertical_and_slash_kernel,
        "block_sparse": block_sparse_kernel,
    }[ty]
    return fc(q, k, v, vertical_size, slash_size)

def minference_prefill_forward(
    query_states, key_states, value_states,
    prefill_kwargs,
):
    print("----------",prefill_kwargs)
    starting_layer = prefill_kwargs["attn_forward_config"].get("starting_layer", 0)
    layer_idx = prefill_kwargs["layer_idx"]

    output = torch.empty_like(query_states)
    bsz, _, q_len, head_dim = query_states.shape
    for head in range(query_states.size(1)):
        q = query_states[:, head, :, :].unsqueeze(1)
        k = key_states[:, head, :, :].unsqueeze(1)
        v = value_states[:, head, :, :].unsqueeze(1)
        if layer_idx >= starting_layer:
            attn_output = minference_prefill_kernel(q, k, v, head, layer_idx, prefill_kwargs["attn_forward_config"])
        else:
            attn_output = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1,2), 0.0, softmax_scale=None, causal=q_len != 1).view(bsz, 1, q_len, head_dim)
        output[:, head:head + 1] = attn_output
    return output

def minference_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    assert output_attentions == False, "output_attentions is not supported for MInference"

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if q_len != 1: # prefill
        minference_kwards = {
            "layer_idx": self.layer_idx,
            "attn_forward_config": {"best_pattern": minference_config},
        }
        attn_output = minference_prefill_forward(
            query_states,
            key_states,
            value_states,
            {"attn_forward_config": minference_config, "layer_idx": self.layer_idx},
        )

        if self.layer_idx == 31:
            estimate_kv_memory(past_key_value)
    else:
        attn_output = _flash_attention_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=self.attention_dropout,
            sliding_window=getattr(self, "sliding_window", None),
            is_causal=self.is_causal,
        )

    # print("attn_output.size(1)",attn_output.shape)
    # print("q_len==============",q_len)

    # assert attn_output.size(1) == q_len
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    # attn_output = attn_output.contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value



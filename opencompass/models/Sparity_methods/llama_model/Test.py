import torch
import time
from opencompass.models.Sparity_methods.llama_model.flex_prefill_attention import flex_prefill_attention

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

if __name__ == '__main__':
    torch.manual_seed(0)
    B, N, H, D = 1, 3228, 32, 128
    gamma = 0.9
    tau = 0.1

    q = torch.randn(B, N, H, D, device='cuda', dtype=torch.bfloat16) 
    k = torch.randn(B, N, H // 4, D, device='cuda', dtype=torch.bfloat16) 
    v = torch.randn(B, N, H // 4, D, device='cuda', dtype=torch.bfloat16)

    print(f"Original shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
    
    # 预热GPU
    for _ in range(5):
        num_key_value_groups = H // (H // 4)  # 计算重复次数，这里是4
        x = repeat_kv(k, num_key_value_groups)
        y = repeat_kv(v, num_key_value_groups)
       
        _ = torch.nn.functional.scaled_dot_product_attention(
            q, x, y, attn_mask=None, is_causal=True
        )
        _ = flex_prefill_attention(
            q, k, v, gamma, tau, min_budget=1024, max_budget=None,
            gqa_interleave=False, block_size=32, return_computational_ratio=False
        )
        break  # 只打印一次形状信息
    
    torch.cuda.synchronize()
    
    # 测量flex_prefill_attention
    num_runs = 10
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        flex_prefill_output, computational_ratio = flex_prefill_attention(
            q,
            k,
            v,
            gamma,
            tau,
            min_budget=1024,
            max_budget=None,
            gqa_interleave=False,
            block_size=32,
            return_computational_ratio=True,
        )
    
    torch.cuda.synchronize()
    flex_time = (time.time() - start_time) / num_runs
    
    # 测量标准scaled_dot_product_attention
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        num_key_value_groups = H // (H // 4)  # 计算重复次数，这里是4
        key_states = repeat_kv(k, num_key_value_groups)
        value_states = repeat_kv(v, num_key_value_groups)
        standard_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
    
    torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / num_runs
    
    print(f'Flex prefill attention time: {flex_time*1000:.2f} ms')
    print(f'Standard attention time: {standard_time*1000:.2f} ms')
    print(f'Speedup: {standard_time/flex_time:.2f}x')
    print(f'Computational ratio: {computational_ratio*100:.2f}%')
    print(f'Flex attention output norm: {flex_prefill_output.norm():.4f}')
    print(f'Standard attention output norm: {standard_output.norm():.4f}')
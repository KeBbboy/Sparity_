from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen2.5-14b-instruct-turbomind',
        path='Qwen/Qwen2.5-14B-Instruct',
        engine_config=dict(session_len=16384, max_batch_size=1, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=2048),
        max_seq_len=16384,
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]

from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content


from mmengine.config import read_base

with read_base():
    from .datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets  # noqa: F401, F403
    from .datasets.aime2025.aime2025_llmjudge_gen_5e9f4f import aime2025_datasets  # noqa: F401, F403

datasets = [
    *aime2024_datasets
]

from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
from opencompass.models import HuggingFace

models = [
    dict(
        type=HuggingFace,
        abbr='deepseek-r1-distill-llama-8b-hf',
        path='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        max_out_len=32768,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]


from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .weights import checkpoint_directory, is_native_checkpoint, prepare_weights


def load_pipeline(model, prepare_prompt=None):
    path = checkpoint_directory(model)
    with ThreadPoolExecutor(max_workers=1) as pool:
        upload = pool.submit(prepare_weights, model, path)
        if prepare_prompt is not None:
            prepare_prompt()
        pipe = _load_components(model, path)
        from .residency import Residency

        residency = Residency(pipe, model.cpu_offload, upload.result())
    return pipe, residency


def _load_components(model, path):
    import torch
    from diffusers import QwenImage21Transformer2DModel
    from ..inference.pipeline import ImagePipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen Image requires an available CUDA GPU/driver")
    checkpoint = path.parent / "qwen_image_2.1_bf16.safetensors"
    components = {}
    optimized = (
        is_native_checkpoint(model) and torch.cuda.get_device_capability() == (12, 0)
        and torch.__version__.partition("+")[0] == "2.14.0"
    )
    transformer_class = QwenImage21Transformer2DModel
    if optimized:
        from ..inference.transformer import ImageTransformer
        transformer_class = ImageTransformer
    if Path(model.model_path).is_dir() and checkpoint.is_file():
        components["transformer"] = transformer_class.from_single_file(
            str(checkpoint), config=str(path), subfolder="transformer",
            dtype=torch.bfloat16, local_files_only=True,
        )
    elif optimized:
        components["transformer"] = transformer_class.from_pretrained(
            str(path), subfolder="transformer",
            dtype=torch.bfloat16, local_files_only=True,
        )
    pipe = ImagePipeline.from_pretrained(
        str(path), dtype=torch.bfloat16,
        local_files_only=True, **components,
    )
    pipe.text_encoder.lm_head = torch.nn.Identity()
    if optimized:
        from ..inference.blocks import optimize_blocks
        optimize_blocks(pipe.transformer)
    pipe.vae.enable_tiling()
    return pipe

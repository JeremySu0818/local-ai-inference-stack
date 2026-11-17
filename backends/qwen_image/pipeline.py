"""Run the Qwen Image generation and editing pipeline."""
import base64
import gc
import io
from pathlib import Path
import math
import threading
import time


class ImageGenerationCancelled(Exception):
    pass


class QwenImageBackend:
    def __init__(self, model):
        self.model = model
        self.pipe = None
        self.lock = threading.Lock()
        from .size import SizeSelector
        self.size_selector = SizeSelector()

    def load(self):
        import torch
        from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen Image requires an available CUDA GPU/driver")
        kwargs = {}
        path = Path(self.model.model_path)
        checkpoint = path.parent / "qwen_image_2.1_bf16.safetensors"
        if path.is_dir() and checkpoint.is_file():
            kwargs["transformer"] = QwenImage21Transformer2DModel.from_single_file(
                str(checkpoint), config=str(path), subfolder="transformer",
                dtype=torch.bfloat16, local_files_only=True,
            )
        self.pipe = QwenImage21Pipeline.from_pretrained(
            self.model.model_path,
            revision=self.model.model_revision,
            dtype=torch.bfloat16,
            local_files_only=True, **kwargs,
        )
        self.pipe.vae.enable_tiling()
        if self.model.cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")

    def generate(self, options, images, cancelled):
        import torch

        def check_step(_pipeline, _step, _timestep, tensors):
            if cancelled.is_set():
                raise ImageGenerationCancelled
            return tensors

        with self.lock, torch.inference_mode():
            if cancelled.is_set():
                return None
            if options.width is None:
                ratio = self.size_selector.choose(options.prompt)
                options.width = round(options.output_resolution * math.sqrt(ratio) / 32) * 32
                options.height = round(options.output_resolution / math.sqrt(ratio) / 32) * 32
                options.validate_options()
            if cancelled.is_set():
                return None
            try:
                result = self.pipe(
                    prompt=options.prompt, image=images or None,
                    negative_prompt=options.negative_prompt,
                    true_cfg_scale=options.true_cfg_scale,
                    width=options.width, height=options.height,
                    num_inference_steps=options.num_inference_steps,
                    num_images_per_prompt=options.n,
                    output_resolution=options.output_resolution,
                    use_kv_cache=options.use_kv_cache,
                    generator=torch.Generator("cpu").manual_seed(options.seed),
                    callback_on_step_end=check_step,
                    callback_on_step_end_tensor_inputs=[],
                )
            except ImageGenerationCancelled:
                torch.cuda.synchronize()
                self.pipe.maybe_free_model_hooks()
                return None
            data = []
            for img in result.images:
                if cancelled.is_set():
                    return None
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                data.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})
            return {"created": int(time.time()), "seed": options.seed,
                    "size": f"{options.width}x{options.height}", "data": data}

    def unload(self):
        import torch

        self.pipe = None
        self.size_selector.model = None
        self.size_selector.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

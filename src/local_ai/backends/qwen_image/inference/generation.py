import logging
import threading

from ..api.response import encode_response
from ..loading.pipeline import load_pipeline
from .size import SizeSelector


log = logging.getLogger(__name__)


class ImageGenerationCancelled(Exception):
    pass


class QwenImageBackend:
    def __init__(self, model):
        self.model = model
        self.pipe = None
        self.lock = threading.Lock()
        self.size_selector = SizeSelector()
        self.first_size = None

    def load(self, first_prompt=None):
        def prepare_prompt():
            try:
                self.first_size = (first_prompt, self.size_selector.choose(first_prompt))
            except Exception as exc:
                log.warning("Automatic sizing deferred to generation: %s", exc)

        self.pipe, self.residency = load_pipeline(self.model, prepare_prompt if first_prompt is not None else None)

    def generate(self, options, images, cancelled):
        import torch

        with self.lock, torch.inference_mode():
            if cancelled.is_set():
                return None
            first_size, self.first_size = self.first_size, None
            if options.width is None:
                ratio = (first_size[1] if first_size and first_size[0] == options.prompt
                         else self.size_selector.choose(options.prompt))
                options.set_aspect_ratio(ratio)
            if cancelled.is_set():
                return None

            self.pipe.release_encoder = self.residency.for_request(options, images)
            try:
                for attempt in range(2):
                    try:
                        self.residency.restore_encoder()
                        result = self._infer(options, images, cancelled)
                        return encode_response(result.images, options, cancelled)
                    except torch.OutOfMemoryError:
                        if attempt or self.residency.offloaded or not self.residency.automatic:
                            raise
                    self.pipe.release_encoder = None
                    self.residency.enable_offload()
                    if cancelled.is_set():
                        return None
            except ImageGenerationCancelled:
                torch.cuda.synchronize()
                self.pipe.maybe_free_model_hooks()
                return None
            finally:
                self.pipe.release_encoder = None

    def _infer(self, options, images, cancelled):
        import torch

        def check_step(pipeline, step, timestep, tensors):
            if cancelled.is_set():
                raise ImageGenerationCancelled
            return tensors

        return self.pipe(
            prompt=options.prompt, image=images or None, negative_prompt=options.negative_prompt,
            true_cfg_scale=options.true_cfg_scale, width=options.width, height=options.height,
            num_inference_steps=options.num_inference_steps, num_images_per_prompt=options.n,
            output_resolution=options.output_resolution, use_kv_cache=options.use_kv_cache,
            generator=torch.Generator("cpu").manual_seed(options.seed),
            callback_on_step_end=check_step, callback_on_step_end_tensor_inputs=[],
        )

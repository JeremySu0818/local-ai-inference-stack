import logging
from functools import partial

import torch

from .encoder import PromptEncoder


log = logging.getLogger(__name__)


class Residency:
    def __init__(self, pipe, cpu_offload, staged):
        self.pipe = pipe
        self.automatic = cpu_offload == "auto"
        self.offloaded = cpu_offload is True
        self.encoder = PromptEncoder(pipe.text_encoder) if self.automatic else None
        if self.offloaded:
            self.pipe.enable_model_cpu_offload()
            return

        self._install(staged)
        try:
            for module in (pipe.transformer, pipe.vae, pipe.text_encoder):
                module.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            return
        except torch.OutOfMemoryError:
            if not self.automatic:
                raise
        self.enable_offload()

    def _install(self, staged):
        for path in sorted(staged):
            component = path.parent.name
            module = getattr(self.pipe, component)
            weights = staged.pop(path)
            expected = module.state_dict()
            if all(key in expected and tensor.shape == expected[key].shape and tensor.dtype == expected[key].dtype
                   for key, tensor in weights.items()):
                module.load_state_dict(weights, strict=False, assign=True)
                if component == "text_encoder" and self.encoder is not None:
                    self.encoder.select_shard(path, weights)

    def for_request(self, options, images):
        if self.encoder is None:
            return None
        pixels = options.width * options.height * options.n
        guided = options.true_cfg_scale > 1 and options.negative_prompt is not None
        if pixels <= 1024 * 1056 and not images and not guided:
            return None
        return partial(self.encoder.park, partial=pixels <= 2 * 1024 * 1056 and len(images) <= 1)

    def restore_encoder(self):
        if self.encoder is not None:
            self.encoder.restore()

    def enable_offload(self):
        self.encoder.park()
        self.pipe.vae.clear_cache()
        self.pipe.enable_model_cpu_offload()
        self.encoder = None
        self.offloaded = True
        log.warning("Using model CPU offload to fit available CUDA memory")

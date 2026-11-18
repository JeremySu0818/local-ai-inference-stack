from diffusers import QwenImage21Pipeline


class ImagePipeline(QwenImage21Pipeline):
    release_encoder = None

    def prepare_latents(self, *args, **kwargs):
        if self.release_encoder is not None:
            self.release_encoder()
        return super().prepare_latents(*args, **kwargs)

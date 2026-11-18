import math
import secrets

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str = Field(min_length=1, max_length=32000)
    negative_prompt: str | None = None
    width: int | None = Field(default=None, ge=256, le=4096)
    height: int | None = Field(default=None, ge=256, le=4096)
    size: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    seed: int = Field(default_factory=lambda: secrets.randbits(32), ge=0, le=2**63-1)
    num_inference_steps: int = Field(default=40, ge=1, le=100)
    true_cfg_scale: float = Field(default=1.0, ge=1, le=20, allow_inf_nan=False)
    use_kv_cache: bool = True
    output_resolution: int = Field(default=1024, ge=256, le=2048)
    response_format: str = "b64_json"
    images: list[str] = Field(default_factory=list, max_length=10)

    def set_aspect_ratio(self, ratio):
        root = math.sqrt(ratio)
        self.width = round(self.output_resolution * root / 32) * 32
        self.height = round(self.output_resolution / root / 32) * 32
        self.validate_options()

    @model_validator(mode="after")
    def validate_options(self):
        if not self.prompt.strip():
            raise ValueError("prompt cannot be blank")
        if self.size and self.size != "auto":
            try:
                self.width, self.height = map(int, self.size.lower().split("x"))
            except ValueError:
                raise ValueError("size must be WIDTHxHEIGHT")
        if (self.width is None) != (self.height is None):
            raise ValueError("provide both width and height, or omit both for automatic sizing")
        if self.width is None:
            return self
        if any(v < 256 or v > 4096 or v % 32 for v in (self.width, self.height)):
            raise ValueError("dimensions must be multiples of 32 between 256 and 4096")
        if self.width * self.height > 4096 * 2048:
            raise ValueError("maximum output area is 8388608 pixels")
        if self.response_format != "b64_json":
            raise ValueError("only response_format=b64_json is supported")
        return self

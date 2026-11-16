"""Download native configs/encoder/VAE into HF_HOME; reuse the moved DiT checkpoint."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
path = root / "downloads/Qwen-Image-2.1/diffusers"
snapshot_download(
    "Qwen/Qwen-Image-2.1",
    revision="d26bb61231c349cf6b7896fa83353113880e1ba3",
    local_dir=path,
    allow_patterns=["model_index.json", "processor/*", "scheduler/*",
                    "text_encoder/*", "vae/*", "transformer/config.json", "LICENSE"],
)
print(path)

# Keep the small CPU size selector in the standard global HF Hub cache.
snapshot_download(
    "Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
    allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE"],
)

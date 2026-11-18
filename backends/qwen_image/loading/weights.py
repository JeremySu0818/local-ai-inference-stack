import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


log = logging.getLogger(__name__)
NATIVE_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"


def checkpoint_directory(model):
    path = Path(model.model_path)
    if path.is_dir():
        return path
    from huggingface_hub import try_to_load_from_cache

    index = try_to_load_from_cache(model.model_path, "model_index.json", revision=model.model_revision)
    if isinstance(index, str):
        return Path(index).parent
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model.model_path, revision=model.model_revision, local_files_only=True))


def prefetch(files, head_shard=None):
    def read(path):
        buffer = bytearray(4 << 20)
        with path.open("rb", buffering=0) as source:
            if path == head_shard:
                header_length = int.from_bytes(source.read(8), "little")
                header = json.loads(source.read(header_length))
                begin, end = header["lm_head.weight"]["data_offsets"]
                if begin == 0:
                    source.seek(8 + header_length + end)
            while source.readinto(buffer):
                pass

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(read, files))


def is_native_checkpoint(model):
    return model.model_path == "Qwen/Qwen-Image-2.1" and model.model_revision == NATIVE_REVISION


def prepare_weights(model, path):
    single = path.parent / "qwen_image_2.1_bf16.safetensors"
    files = list((path / "vae").glob("*.safetensors")) + list((path / "text_encoder").glob("*.safetensors"))
    if Path(model.model_path).is_dir() and single.is_file():
        files.append(single)
    else:
        files.extend((path / "transformer").glob("*.safetensors"))

    if model.cpu_offload is not True and is_native_checkpoint(model):
        import torch

        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > sum(f.stat().st_size for f in files):
            index = json.loads((path / "text_encoder" / "model.safetensors.index.json").read_text())
            head_shard = path / "text_encoder" / index["weight_map"]["lm_head.weight"]
            direct = [f for f in files if f.parent.name != "vae" and f != head_shard]
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    remainder = pool.submit(prefetch, [f for f in files if f not in direct], head_shard)
                    states = upload_shards(direct)
                    remainder.result()
                return states
            except Exception as exc:
                log.warning("Checkpoint streaming unavailable, using local model loader: %s", exc)
    prefetch(files)
    return {}


def upload_shards(paths):
    from fastsafetensors import SafeTensorsFileLoader, cpp

    cpp.set_gil_release(True)

    loader = SafeTensorsFileLoader(None, "cuda:0", nogds=True, set_numa=False,
                                   bbuf_size_kb=8192, max_threads=16)
    try:
        loader.add_filenames({0: [str(path) for path in paths]})
        buffer = loader.copy_files_to_device(max_copy_block_size=64 << 20)
        try:
            states = {}
            for name in loader.get_keys():
                path = Path(buffer.get_filename(name))
                states.setdefault(path, {})[name] = buffer.get_tensor(name)
            return states
        finally:
            buffer.close()
    finally:
        loader.close()

import logging

import torch

from .weights import upload_shards


log = logging.getLogger(__name__)


class PromptEncoder:
    def __init__(self, module):
        self.module = module
        self.host = module.state_dict()
        self.shard = None
        self.subset = {}
        self.upload_paths = []
        self.parked = False
        self.partial = False

    def select_shard(self, path, keys):
        self.upload_paths.append(path)
        if self.shard is None and "model.language_model.embed_tokens.weight" not in keys:
            self.shard = path
            self.subset = {key: self.host[key] for key in keys}

    def park(self, partial=False):
        partial = partial and self.shard is not None
        if self.parked and (not self.partial or partial):
            return
        torch.cuda.current_stream().synchronize()
        self.module.load_state_dict(self.subset if partial else self.host, strict=not partial, assign=True)
        self.parked = True
        self.partial = partial

    def restore(self):
        if not self.parked:
            return
        if self.upload_paths:
            try:
                paths = [self.shard] if self.partial else self.upload_paths
                staged = upload_shards(paths)
            except Exception as exc:
                log.warning("Restoring prompt encoder from host memory: %s", exc)
            else:
                for weights in staged.values():
                    self.module.load_state_dict(weights, strict=False, assign=True)
        self.module.to("cuda", non_blocking=True)
        self.parked = False
        self.partial = False

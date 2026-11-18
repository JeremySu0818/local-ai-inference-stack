"""Populate the shared cache for the models in the configuration."""
import sys

from local_ai.backends import get_backend
from local_ai.router.config import load_config


def prepare(config_path):
    for model in load_config(config_path).models.values():
        hook = get_backend(model.backend).prepare
        if hook is not None:
            hook(model)


if __name__ == "__main__":
    prepare(sys.argv[1])

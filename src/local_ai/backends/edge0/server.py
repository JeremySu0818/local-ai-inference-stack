"""Launch a configured Edge0 model through the edge0-infer portal."""
import os

from local_ai.router.config import load_config

from . import INFER, STRATA, STRATA_PYTHON, check, engine_args, model_dir


def serve(config_path, model_id):
    model = load_config(config_path).models[model_id]
    if model.backend != "edge0":
        raise SystemExit(f"{model_id} is not an Edge0 model")
    for tool in (INFER, STRATA, STRATA_PYTHON):
        if not os.access(tool, os.X_OK):
            raise SystemExit(f"Edge0 {tool} is missing. Run ./setup.sh")
    path = model_dir(model)
    check(model, path)
    os.execv(str(INFER), engine_args(model, path, "serve"))

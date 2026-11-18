"""Launch a configured Edge0 model through the edge0-infer portal."""
import os

from local_ai.router.config import load_config

from . import check, engine_args, infer, model_dir, strata, strata_python


def serve(config_path, model_id):
    model = load_config(config_path).models[model_id]
    if model.backend != "edge0":
        raise SystemExit(f"{model_id} is not an Edge0 model")
    for tool in (infer(), strata(), strata_python()):
        if not os.access(tool, os.X_OK):
            raise SystemExit(f"Edge0 {tool} is missing. Run ./setup.sh")
    path = model_dir(model)
    check(model, path)
    os.execv(str(infer()), engine_args(model, path, "serve"))

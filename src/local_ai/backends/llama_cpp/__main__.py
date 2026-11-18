import sys

from . import source_dir
from .server import serve


if __name__ == "__main__":
    if sys.argv[1:] == ["source-dir"]:
        print(source_dir())
    elif len(sys.argv) == 4 and sys.argv[1] == "serve":
        serve(*sys.argv[2:])
    else:
        raise SystemExit("Usage: python -m local_ai.backends.llama_cpp source-dir | serve CONFIG MODEL_ID")

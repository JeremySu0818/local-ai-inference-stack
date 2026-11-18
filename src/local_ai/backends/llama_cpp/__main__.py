import sys

from .server import SOURCE_DIR, serve


if __name__ == "__main__":
    if sys.argv[1:] == ["source-dir"]:
        print(SOURCE_DIR)
    elif len(sys.argv) == 4 and sys.argv[1] == "serve":
        serve(*sys.argv[2:])
    else:
        raise SystemExit("Usage: python -m local_ai.backends.llama_cpp source-dir | serve CONFIG MODEL_ID")

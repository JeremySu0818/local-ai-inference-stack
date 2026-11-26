"""Mirror service output and keep the latest 1,000,000 lines for this run."""
import os
import sys

from local_ai.router.config import load_config


def main():
    path = load_config(os.environ["AI_ROUTER_CONFIG"]).state_dir / "server.log"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    limit = 1_000_000
    line_count = 0
    with path.open("w+b") as log:
        while chunk := sys.stdin.buffer.read1(65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
            log.flush()
            line_count += chunk.count(b"\n")
            if line_count > limit:
                log.seek(0)
                for _ in range(line_count - limit):
                    log.readline()
                remaining = log.read()
                log.seek(0)
                log.write(remaining)
                log.truncate()
                log.flush()
                line_count = limit


if __name__ == "__main__":
    main()

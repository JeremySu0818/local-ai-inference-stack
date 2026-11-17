"""Prevent multiple routers from owning the same state directory."""
import fcntl
import logging
import os

from .errors import StartupBlocked

log = logging.getLogger(__name__)


def acquire_state_lock(state_dir):
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    lockfile = open(state_dir / "router.lock", "a+")
    try:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lockfile.close()
        raise StartupBlocked("已有 router 使用此狀態目錄，略過啟動")
    marker = state_dir / "unclean"
    if marker.exists():
        try:
            owner = int(marker.read_text().strip())
            if owner <= 0:
                raise ValueError("invalid PID")
        except ValueError:
            owner = None
        if owner is not None:
            try:
                os.kill(owner, 0)
            except ProcessLookupError:
                pass
            except PermissionError:
                lockfile.close()
                raise StartupBlocked(f"unclean 記錄的程序 {owner} 仍存活，略過啟動")
            else:
                lockfile.close()
                raise StartupBlocked(f"unclean 記錄的程序 {owner} 仍存活，略過啟動")
        log.warning("removing stale shutdown marker: %s (PID %s)", marker, owner)
        marker.unlink()
    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"{os.getpid()}\n")
        f.flush()
        os.fsync(f.fileno())
    return lockfile


def release_state_lock(state_dir, lockfile):
    (state_dir / "unclean").unlink()
    fcntl.flock(lockfile, fcntl.LOCK_UN)
    lockfile.close()

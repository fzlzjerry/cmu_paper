"""Ready/release checkpoint for exact E00 CUDA-process auditing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import time


def _process_start_time_ticks() -> int:
    raw = Path("/proc/self/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise RuntimeError("cannot parse /proc/self/stat")
    fields = raw[closing + 2 :].split()
    if len(fields) <= 19:
        raise RuntimeError("cannot parse process start time")
    return int(fields[19])


def audit_checkpoint(
    *,
    ready_file: str | None,
    release_file: str | None,
    timeout_seconds: float,
) -> None:
    if ready_file is None and release_file is None:
        return
    if ready_file is None or release_file is None:
        raise ValueError("audit ready/release files must be supplied together")
    if timeout_seconds <= 0:
        raise ValueError("audit timeout must be positive")

    ready_path = Path(ready_file)
    release_path = Path(release_file)
    if not ready_path.is_absolute() or not release_path.is_absolute():
        raise ValueError("audit handshake paths must be absolute")
    if ready_path == release_path or ready_path.parent != release_path.parent:
        raise ValueError("audit handshake paths must be distinct siblings")
    ready_path.parent.resolve(strict=True)

    payload = {
        "protocol": "e00-process-audit-v1",
        "pid": os.getpid(),
        "process_start_time_ticks": _process_start_time_ticks(),
        "cuda_work_complete": True,
    }
    temporary = ready_path.with_name(
        f".{ready_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with temporary.open("xb") as handle:
        handle.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, ready_path, follow_symlinks=False)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    directory_fd = os.open(ready_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            release_stat = release_path.lstat()
        except FileNotFoundError:
            release_stat = None
        if release_stat is not None:
            if not stat.S_ISREG(release_stat.st_mode):
                raise RuntimeError("audit release path is not a regular file")
            if release_path.read_bytes() != b"release\n":
                raise RuntimeError("audit release file has invalid content")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for process-audit release")
        time.sleep(0.05)

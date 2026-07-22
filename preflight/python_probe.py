#!/usr/bin/env python3
"""Emit the isolated Python/CUDA inventory used by E00."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preflight.audit_checkpoint import audit_checkpoint


def distribution_version(name: str) -> dict[str, Any]:
    try:
        return {"installed": True, "version": importlib.metadata.version(name)}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-ready-file")
    parser.add_argument("--audit-release-file")
    parser.add_argument("--audit-timeout-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    import torch

    available = bool(torch.cuda.is_available())
    payload: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.splitlines()[0],
            "optimize_level": int(sys.flags.optimize),
        },
        "torch": {
            "installed": True,
            "version": torch.__version__,
            "built_cuda": torch.version.cuda,
        },
        "triton": distribution_version("triton"),
        "vllm": distribution_version("vllm"),
        "jsonschema": distribution_version("jsonschema"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_is_available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "current_device": None,
        "device_name": None,
        "device_uuid": None,
        "compute_capability": None,
        "total_memory_bytes": None,
    }
    if available:
        device = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device)
        major, minor = torch.cuda.get_device_capability(device)
        device_uuid = str(properties.uuid)
        if not device_uuid.startswith("GPU-"):
            device_uuid = f"GPU-{device_uuid}"
        payload.update(
            {
                "current_device": device,
                "device_name": torch.cuda.get_device_name(device),
                "device_uuid": device_uuid,
                "compute_capability": {"major": int(major), "minor": int(minor)},
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    if available:
        torch.cuda.synchronize()
        audit_checkpoint(
            ready_file=args.audit_ready_file,
            release_file=args.audit_release_file,
            timeout_seconds=args.audit_timeout_seconds,
        )
    print(json.dumps(payload, sort_keys=True))
    return 0 if available else 2


if __name__ == "__main__":
    raise SystemExit(main())

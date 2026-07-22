#!/usr/bin/env python3
"""Fresh-process native-SASS or forced-PTX E00 runtime probe."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preflight.audit_checkpoint import audit_checkpoint

MASK = 0x5A5A5A5A


def _require_jit_contract(mode: str) -> None:
    force_ptx = os.environ.get("CUDA_FORCE_PTX_JIT")
    disable_ptx = os.environ.get("CUDA_DISABLE_PTX_JIT")
    if mode == "native":
        if disable_ptx != "1" or force_ptx is not None:
            raise RuntimeError(
                "native mode requires CUDA_DISABLE_PTX_JIT=1 and an unset "
                "CUDA_FORCE_PTX_JIT"
            )
        return

    if force_ptx != "1" or disable_ptx is not None:
        raise RuntimeError(
            "forced-ptx mode requires CUDA_FORCE_PTX_JIT=1 and an unset "
            "CUDA_DISABLE_PTX_JIT"
        )
    if os.environ.get("CUDA_CACHE_DISABLE") == "1":
        return
    raw_cache_path = os.environ.get("CUDA_CACHE_PATH")
    if raw_cache_path is None:
        raise RuntimeError(
            "forced-ptx mode requires CUDA_CACHE_DISABLE=1 or a unique, "
            "initially empty CUDA_CACHE_PATH"
        )
    cache_path = Path(raw_cache_path).expanduser().resolve(strict=True)
    if not cache_path.is_dir():
        raise RuntimeError(f"CUDA_CACHE_PATH is not a directory: {cache_path}")
    if next(cache_path.iterdir(), None) is not None:
        raise RuntimeError(f"CUDA_CACHE_PATH is not initially empty: {cache_path}")


def _load_extension():
    module_name = os.environ.get("E00_EXTENSION_MODULE", "e00_cuda_cert")
    extension_path = os.environ.get("E00_EXTENSION_PATH")
    if extension_path is None:
        return importlib.import_module(module_name)
    path = Path(extension_path).expanduser().resolve(strict=True)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E00 extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_signed_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


def run_probe(
    mode: str,
    *,
    audit_ready_file: str | None = None,
    audit_release_file: str | None = None,
    audit_timeout_seconds: float = 60.0,
) -> dict[str, object]:
    # Validate the requested code-path environment before importing torch or
    # loading any CUDA module in this fresh process.
    _require_jit_contract(mode)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    extension = _load_extension()
    device = torch.device("cuda", 0)
    values = [
        _to_signed_int32(index * 0x9E3779B1 + 0x12345678)
        for index in range(257)
    ]
    expected_values = [
        _to_signed_int32((value & 0xFFFFFFFF) ^ MASK) for value in values
    ]
    expected = torch.tensor(expected_values, dtype=torch.int32)
    input_tensor = torch.tensor(values, dtype=torch.int32, device=device)
    output_tensor = torch.empty_like(input_tensor)
    output_pointer = output_tensor.data_ptr()

    result = extension.xor_out(input_tensor, output_tensor)
    torch.cuda.synchronize(device)
    if result is not None:
        raise AssertionError("xor_out returned an object instead of None")
    if output_tensor.data_ptr() != output_pointer:
        raise AssertionError("output pointer changed")
    if not torch.equal(output_tensor.cpu(), expected):
        raise AssertionError("runtime probe produced an incorrect XOR result")

    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    payload = {
        "capability": [capability[0], capability[1]],
        "device_name": properties.name,
        "length": len(values),
        "mode": mode,
        "status": "pass",
    }
    audit_checkpoint(
        ready_file=audit_ready_file,
        release_file=audit_release_file,
        timeout_seconds=audit_timeout_seconds,
    )
    return payload


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "forced-ptx"), required=True)
    parser.add_argument("--audit-ready-file")
    parser.add_argument("--audit-release-file")
    parser.add_argument("--audit-timeout-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    payload = run_probe(
        args.mode,
        audit_ready_file=args.audit_ready_file,
        audit_release_file=args.audit_release_file,
        audit_timeout_seconds=args.audit_timeout_seconds,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

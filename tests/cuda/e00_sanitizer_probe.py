#!/usr/bin/env python3
"""Minimal assertion-based E00 probe for Compute Sanitizer tools."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preflight.audit_checkpoint import audit_checkpoint


MASK = 0x5A5A5A5A


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


def run_sanitizer_probe(
    *,
    audit_ready_file: str | None = None,
    audit_release_file: str | None = None,
    audit_timeout_seconds: float = 60.0,
) -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA must be available under Compute Sanitizer")
    extension = _load_extension()
    device = torch.device("cuda", 0)
    values = [
        _to_signed_int32(index * 0x45D9F3B + 0x10203040)
        for index in range(4097)
    ]
    expected = torch.tensor(
        [_to_signed_int32((value & 0xFFFFFFFF) ^ MASK) for value in values],
        dtype=torch.int32,
    )
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
        raise AssertionError("sanitizer probe produced an incorrect XOR result")
    audit_checkpoint(
        ready_file=audit_ready_file,
        release_file=audit_release_file,
        timeout_seconds=audit_timeout_seconds,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-ready-file")
    parser.add_argument("--audit-release-file")
    parser.add_argument("--audit-timeout-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = _parse_args(sys.argv[1:])
    run_sanitizer_probe(
        audit_ready_file=arguments.audit_ready_file,
        audit_release_file=arguments.audit_release_file,
        audit_timeout_seconds=arguments.audit_timeout_seconds,
    )
    print(json.dumps({"length": 4097, "status": "pass"}, sort_keys=True))

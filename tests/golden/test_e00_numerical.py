#!/usr/bin/env python3
"""Exact CPU-golden tests for the E00 certification-only CUDA operation."""

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
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
LENGTHS = (1, 255, 256, 257, 4097)
ANCHORS = (
    INT32_MIN,
    INT32_MAX,
    -2_000_000_001,
    -65_537,
    -256,
    -1,
    0,
    1,
    255,
    256,
    65_537,
    2_000_000_001,
)


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


def _input_values(length: int) -> list[int]:
    values: list[int] = []
    for index in range(length):
        if index < len(ANCHORS):
            values.append(ANCHORS[index])
        else:
            mixed = (index * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFFFFFF
            values.append(_to_signed_int32(mixed))
    return values


def _golden(values: list[int]) -> list[int]:
    return [_to_signed_int32((value & 0xFFFFFFFF) ^ MASK) for value in values]


def run_numerical_test(
    *,
    audit_ready_file: str | None = None,
    audit_release_file: str | None = None,
    audit_timeout_seconds: float = 60.0,
) -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA must be available for E00 numerical tests")
    extension = _load_extension()
    device = torch.device("cuda", 0)

    for length in LENGTHS:
        values = _input_values(length)
        expected = torch.tensor(_golden(values), dtype=torch.int32)
        input_tensor = torch.tensor(values, dtype=torch.int32, device=device)
        output_tensor = torch.empty_like(input_tensor)
        output_pointer = output_tensor.data_ptr()

        result = extension.xor_out(input_tensor, output_tensor)
        torch.cuda.synchronize(device)

        if result is not None:
            raise AssertionError("xor_out must not construct or return an output")
        if output_tensor.data_ptr() != output_pointer:
            raise AssertionError("output pointer changed")
        if not torch.equal(output_tensor.cpu(), expected):
            raise AssertionError(f"exact XOR mismatch at length {length}")

    torch.cuda.synchronize(device)
    audit_checkpoint(
        ready_file=audit_ready_file,
        release_file=audit_release_file,
        timeout_seconds=audit_timeout_seconds,
    )


def test_e00_numerical() -> None:
    run_numerical_test()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-ready-file")
    parser.add_argument("--audit-release-file")
    parser.add_argument("--audit-timeout-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    run_numerical_test(
        audit_ready_file=args.audit_ready_file,
        audit_release_file=args.audit_release_file,
        audit_timeout_seconds=args.audit_timeout_seconds,
    )
    print(json.dumps({"lengths": list(LENGTHS), "status": "pass"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

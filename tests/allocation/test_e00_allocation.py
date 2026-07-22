#!/usr/bin/env python3
"""Eager and CUDA Graph allocation-stability tests for E00."""

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
LENGTH = 4097
MINIMUM_ITERATIONS = 1000
STAT_KEYS = (
    "allocation.all.current",
    "allocation.all.allocated",
    "segment.all.current",
    "segment.all.allocated",
    "active_bytes.all.current",
    "allocated_bytes.all.current",
    "allocated_bytes.all.allocated",
    "reserved_bytes.all.current",
    "reserved_bytes.all.allocated",
    "num_device_alloc",
    "num_device_free",
    "num_alloc_retries",
    "num_ooms",
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


def _allocator_snapshot(device: torch.device) -> dict[str, int]:
    statistics = torch.cuda.memory_stats(device)
    snapshot = {key: int(statistics.get(key, 0)) for key in STAT_KEYS}
    snapshot["memory_allocated"] = int(torch.cuda.memory_allocated(device))
    snapshot["memory_reserved"] = int(torch.cuda.memory_reserved(device))
    snapshot["max_memory_allocated"] = int(torch.cuda.max_memory_allocated(device))
    snapshot["max_memory_reserved"] = int(torch.cuda.max_memory_reserved(device))
    return snapshot


def _assert_stable(
    lane: str,
    before: dict[str, int],
    after: dict[str, int],
) -> None:
    if before != after:
        changes = {
            key: {"before": before[key], "after": after[key]}
            for key in before
            if before[key] != after[key]
        }
        raise AssertionError(f"{lane} allocator counters changed: {changes}")


def _iteration_count() -> int:
    raw = os.environ.get("E00_ALLOCATION_ITERATIONS", str(MINIMUM_ITERATIONS))
    iterations = int(raw)
    if iterations < MINIMUM_ITERATIONS:
        raise ValueError(
            f"E00_ALLOCATION_ITERATIONS must be >= {MINIMUM_ITERATIONS}"
        )
    return iterations


def run_allocation_test(
    *,
    audit_ready_file: str | None = None,
    audit_release_file: str | None = None,
    audit_timeout_seconds: float = 60.0,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA must be available for E00 allocation tests")
    extension = _load_extension()
    iterations = _iteration_count()
    device = torch.device("cuda", 0)
    values = [
        _to_signed_int32(index * 0x9E3779B1 + 0x13579BDF)
        for index in range(LENGTH)
    ]
    expected = torch.tensor(
        [_to_signed_int32((value & 0xFFFFFFFF) ^ MASK) for value in values],
        dtype=torch.int32,
    )
    input_tensor = torch.tensor(values, dtype=torch.int32, device=device)
    output_tensor = torch.empty_like(input_tensor)
    input_pointer = input_tensor.data_ptr()
    output_pointer = output_tensor.data_ptr()

    for _ in range(32):
        if extension.xor_out(input_tensor, output_tensor) is not None:
            raise AssertionError("xor_out returned an object during eager warmup")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    eager_before = _allocator_snapshot(device)
    for _ in range(iterations):
        extension.xor_out(input_tensor, output_tensor)
    torch.cuda.synchronize(device)
    eager_after = _allocator_snapshot(device)
    _assert_stable("eager", eager_before, eager_after)
    if input_tensor.data_ptr() != input_pointer:
        raise AssertionError("eager input pointer changed")
    if output_tensor.data_ptr() != output_pointer:
        raise AssertionError("eager output pointer changed")
    if not torch.equal(output_tensor.cpu(), expected):
        raise AssertionError("eager output does not match the exact golden")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_result = extension.xor_out(input_tensor, output_tensor)
    if captured_result is not None:
        raise AssertionError("xor_out returned an object during capture")
    for _ in range(32):
        graph.replay()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    graph_before = _allocator_snapshot(device)
    for _ in range(iterations):
        graph.replay()
    torch.cuda.synchronize(device)
    graph_after = _allocator_snapshot(device)
    _assert_stable("graph", graph_before, graph_after)
    if input_tensor.data_ptr() != input_pointer:
        raise AssertionError("Graph input pointer changed")
    if output_tensor.data_ptr() != output_pointer:
        raise AssertionError("Graph output pointer changed")
    if not torch.equal(output_tensor.cpu(), expected):
        raise AssertionError("Graph output does not match the exact golden")

    torch.cuda.synchronize(device)
    audit_checkpoint(
        ready_file=audit_ready_file,
        release_file=audit_release_file,
        timeout_seconds=audit_timeout_seconds,
    )
    return {
        "eager_iterations": iterations,
        "graph_replays": iterations,
        "length": LENGTH,
        "status": "pass",
    }


def test_e00_allocation_stability() -> None:
    run_allocation_test()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-ready-file")
    parser.add_argument("--audit-release-file")
    parser.add_argument("--audit-timeout-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    payload = run_allocation_test(
        audit_ready_file=args.audit_ready_file,
        audit_release_file=args.audit_release_file,
        audit_timeout_seconds=args.audit_timeout_seconds,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

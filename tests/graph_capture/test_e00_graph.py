#!/usr/bin/env python3
"""CUDA Graph capture/replay certification for the E00 XOR operation."""

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
REPLAY_COUNT = 3
LENGTH = 4097


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


def _values(seed: int) -> list[int]:
    return [
        _to_signed_int32(index * 0x9E3779B1 + seed)
        for index in range(LENGTH)
    ]


def _expected(values: list[int]) -> torch.Tensor:
    return torch.tensor(
        [_to_signed_int32((value & 0xFFFFFFFF) ^ MASK) for value in values],
        dtype=torch.int32,
    )


def run_graph_test(
    *,
    audit_ready_file: str | None = None,
    audit_release_file: str | None = None,
    audit_timeout_seconds: float = 60.0,
) -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA must be available for E00 Graph tests")
    extension = _load_extension()
    device = torch.device("cuda", 0)
    replay_values = [_values(seed) for seed in (0x01020304, 0x11223344, 0x55667788)]
    replay_inputs = [
        torch.tensor(values, dtype=torch.int32, device=device)
        for values in replay_values
    ]
    replay_expected = [_expected(values) for values in replay_values]
    static_input = torch.empty(LENGTH, dtype=torch.int32, device=device)
    static_output = torch.empty_like(static_input)
    input_pointer = static_input.data_ptr()
    output_pointer = static_output.data_ptr()

    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        static_input.copy_(replay_inputs[0])
        for _ in range(3):
            if extension.xor_out(static_input, static_output) is not None:
                raise AssertionError("xor_out returned an object during warmup")
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_result = extension.xor_out(static_input, static_output)
    if captured_result is not None:
        raise AssertionError("xor_out returned an object during capture")

    for source, expected in zip(replay_inputs, replay_expected, strict=True):
        static_input.copy_(source)
        graph.replay()
        torch.cuda.synchronize(device)
        if static_input.data_ptr() != input_pointer:
            raise AssertionError("static input pointer changed")
        if static_output.data_ptr() != output_pointer:
            raise AssertionError("static output pointer changed")
        if not torch.equal(static_output.cpu(), expected):
            raise AssertionError("Graph replay produced an incorrect XOR result")

    torch.cuda.synchronize(device)
    audit_checkpoint(
        ready_file=audit_ready_file,
        release_file=audit_release_file,
        timeout_seconds=audit_timeout_seconds,
    )


def test_e00_graph_capture_replay() -> None:
    run_graph_test()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-ready-file")
    parser.add_argument("--audit-release-file")
    parser.add_argument("--audit-timeout-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    run_graph_test(
        audit_ready_file=args.audit_ready_file,
        audit_release_file=args.audit_release_file,
        audit_timeout_seconds=args.audit_timeout_seconds,
    )
    print(
        json.dumps(
            {"length": LENGTH, "replays": REPLAY_COUNT, "status": "pass"},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

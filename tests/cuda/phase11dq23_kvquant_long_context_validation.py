#!/usr/bin/env python3
"""Deterministic long-context validation for KVQuant q3/q2 Value decode."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from kvbench.runtime.allocation import audit_cuda_allocations


AUTHORIZED_IMAGE = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
WIDTH = 4092
CAPACITY = WIDTH + 5
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
SPARSE_CAP = 12
TILE_WIDTH = 128
STORED_TILES = (WIDTH + TILE_WIDTH - 1) // TILE_WIDTH - 1
ATOL = 0.01
RTOL = 0.01
APIS = {
    bits: (
        f"vecquant{bits}matmul_nuq_perchannel_transposed_"
        "mha_batched_fused_opt2_deterministic_out"
    )
    for bits in (3, 2)
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("determinism", "fixtures", "stream", "graph", "sanitizer"),
    )
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().contiguous().view(torch.uint8).cpu()
    payload = bytes(contiguous.untyped_storage())[: int(contiguous.numel())]
    return hashlib.sha256(payload).hexdigest()


def _load_extension(path: Path, expected_sha256: str) -> ModuleType:
    resolved = path.resolve(strict=True)
    if _sha256(resolved) != expected_sha256:
        raise RuntimeError("Phase 11D-Q23 extension SHA-256 differs")
    specification = importlib.util.spec_from_file_location("quant_cuda", resolved)
    if specification is None or specification.loader is None:
        raise RuntimeError("Phase 11D-Q23 extension loader is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    missing = [name for name in APIS.values() if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"Phase 11D-Q23 deterministic APIs are absent: {missing}")
    return module


def _require_environment() -> None:
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST") != AUTHORIZED_IMAGE
        or os.environ.get("KVBENCH_EXECUTION_ENVIRONMENT")
        != "measurement_container"
    ):
        raise RuntimeError(
            "Phase 11D-Q23 CUDA validation requires the authorized container"
        )


def _synthetic_inputs(bits: int) -> dict[str, torch.Tensor]:
    device = torch.device("cuda:0")
    levels = 1 << bits
    packed_rows = bits * HEAD_DIM // 32
    score_storage = torch.linspace(
        -0.75,
        0.875,
        steps=QUERY_HEADS * CAPACITY,
        dtype=torch.float32,
        device=device,
    ).view(1, QUERY_HEADS, CAPACITY)
    vector = score_storage[:, :, :WIDTH]
    packed_seed = torch.arange(
        KV_HEADS * packed_rows * CAPACITY,
        dtype=torch.int64,
        device=device,
    ).view(KV_HEADS, packed_rows, CAPACITY)
    packed = (
        packed_seed.mul_(0x45D9F3B)
        .add_(0x13579BDF)
        .bitwise_and_(0xFFFFFFFF)
        .to(torch.int32)
    )
    lookup = torch.linspace(
        -1.25,
        1.5,
        steps=CAPACITY * levels,
        dtype=torch.float32,
        device=device,
    ).view(CAPACITY, levels)
    outliers = (
        torch.arange(
            CAPACITY * SPARSE_CAP,
            dtype=torch.float32,
            device=device,
        )
        .view(CAPACITY, SPARSE_CAP)
        .remainder_(97)
        .sub_(48)
        .mul_(0.0005)
    )
    indices = (
        torch.arange(SPARSE_CAP, dtype=torch.int32, device=device)
        .view(1, SPARSE_CAP)
        .expand(CAPACITY, SPARSE_CAP)
        .contiguous()
    )
    workspace_storage = torch.empty(
        (1, QUERY_HEADS, CAPACITY),
        dtype=torch.float32,
        device=device,
    )
    workspace = workspace_storage[
        :, :, : STORED_TILES * TILE_WIDTH
    ].view(1, QUERY_HEADS, STORED_TILES, HEAD_DIM)
    return {
        "score_storage": score_storage,
        "vector": vector,
        "packed": packed,
        "lookup": lookup,
        "outliers": outliers,
        "indices": indices,
        "output": torch.empty(
            (1, QUERY_HEADS, HEAD_DIM),
            dtype=torch.float32,
            device=device,
        ),
        "workspace_storage": workspace_storage,
        "workspace": workspace,
    }


def _call(
    runtime: ModuleType,
    bits: int,
    inputs: dict[str, torch.Tensor],
) -> None:
    getattr(runtime, APIS[bits])(
        inputs["vector"],
        inputs["packed"],
        inputs["output"],
        inputs["lookup"],
        WIDTH,
        inputs["outliers"],
        inputs["indices"],
        inputs["workspace"],
    )


def _pointers(inputs: dict[str, torch.Tensor]) -> dict[str, int]:
    return {
        name: int(tensor.data_ptr())
        for name, tensor in sorted(inputs.items())
    }


def _independent_control(
    bits: int,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    levels = 1 << bits
    vector = inputs["vector"].detach().cpu()
    packed = inputs["packed"].detach().cpu().to(torch.int64)
    lookup = inputs["lookup"].detach().cpu()
    outliers = inputs["outliers"].detach().cpu()
    indices = inputs["indices"].detach().cpu().to(torch.int64)
    tokens = torch.arange(WIDTH, dtype=torch.int64)
    dense = torch.empty((KV_HEADS, HEAD_DIM, WIDTH), dtype=torch.float32)
    for kv_head in range(KV_HEADS):
        for channel in range(HEAD_DIM):
            bit_offset = channel * bits
            packed_word = bit_offset // 32
            shift = bit_offset % 32
            words = torch.bitwise_and(
                packed[kv_head, packed_word, :WIDTH],
                0xFFFFFFFF,
            )
            codes = torch.bitwise_right_shift(words, shift)
            if shift + bits > 32:
                next_words = torch.bitwise_and(
                    packed[kv_head, packed_word + 1, :WIDTH],
                    0xFFFFFFFF,
                )
                codes = torch.bitwise_or(
                    codes,
                    torch.bitwise_left_shift(next_words, 32 - shift),
                )
            codes = torch.bitwise_and(codes, levels - 1)
            dense[kv_head, channel] = lookup[tokens, codes]

    sparse_rows = torch.zeros(
        (WIDTH, KV_HEADS * HEAD_DIM),
        dtype=torch.float32,
    )
    sparse_rows.scatter_add_(1, indices[:WIDTH], outliers[:WIDTH])
    reconstructed = dense + sparse_rows.transpose(0, 1).reshape(
        KV_HEADS,
        HEAD_DIM,
        WIDTH,
    )
    control = torch.empty((1, QUERY_HEADS, HEAD_DIM), dtype=torch.float64)
    for query_head in range(QUERY_HEADS):
        kv_head = query_head // 4
        control[0, query_head] = torch.sum(
            reconstructed[kv_head].to(torch.float64)
            * vector[0, query_head].to(torch.float64),
            dim=1,
        )
    return control.to(torch.float32)


def _control_result(
    observed: torch.Tensor,
    control: torch.Tensor,
) -> dict[str, Any]:
    observed_cpu = observed.detach().cpu()
    difference = (observed_cpu - control).abs()
    relative = difference / torch.maximum(
        control.abs(),
        torch.full_like(control, 1.0e-12),
    )
    passed = bool(torch.allclose(observed_cpu, control, atol=ATOL, rtol=RTOL))
    if not passed:
        raise AssertionError("q3/q2 independent numerical control differs")
    return {
        "atol": ATOL,
        "rtol": RTOL,
        "control_sha256": _tensor_sha256(control),
        "maximum_absolute_difference": float(difference.max()),
        "maximum_relative_difference": float(relative.max()),
        "passed": True,
    }


def _run_repetitions(
    runtime: ModuleType,
    bits: int,
    inputs: dict[str, torch.Tensor],
    repetitions: int,
) -> dict[str, Any]:
    pointers = _pointers(inputs)
    hashes: list[str] = []
    for _ in range(repetitions):
        _call(runtime, bits, inputs)
        torch.cuda.synchronize()
        hashes.append(_tensor_sha256(inputs["output"]))
    if len(set(hashes)) != 1:
        raise AssertionError(f"q{bits} output remains history-dependent")
    if not bool(torch.isfinite(inputs["output"]).all()):
        raise AssertionError(f"q{bits} output contains NaN or Inf")
    if pointers != _pointers(inputs):
        raise AssertionError(f"q{bits} caller-owned pointer changed")
    return {
        "bits": bits,
        "repetitions": repetitions,
        "output_sha256": hashes[0],
        "unique_output_sha256_count": 1,
        "caller_owned_pointers_stable": True,
        "all_finite": True,
    }


def _run_determinism(runtime: ModuleType) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for bits in (3, 2):
        inputs = _synthetic_inputs(bits)
        result = _run_repetitions(runtime, bits, inputs, 100)
        result["independent_control"] = _control_result(
            inputs["output"],
            _independent_control(bits, inputs),
        )
        result["score_stride"] = list(inputs["vector"].stride())
        result["workspace_stride"] = list(inputs["workspace"].stride())
        cases.append(result)
    return {
        "mode": "determinism",
        "quantized_context": WIDTH,
        "stored_workspace_tiles": STORED_TILES,
        "cases": cases,
        "passed": True,
    }


class _FixtureRuntime:
    def __init__(self, runtime: ModuleType) -> None:
        self._runtime = runtime
        self._workspaces = {
            bits: torch.empty(
                (1, QUERY_HEADS, 32 if bits == 4 else 31, HEAD_DIM),
                dtype=torch.float32,
                device="cuda",
            )
            for bits in (4, 3, 2)
        }
        self._pointers = {
            bits: int(value.data_ptr())
            for bits, value in self._workspaces.items()
        }

    def __getattr__(self, name: str) -> Any:
        for bits in (4, 3, 2):
            legacy = (
                f"vecquant{bits}matmul_nuq_perchannel_transposed_"
                "mha_batched_fused_opt2"
            )
            if name == legacy:
                return lambda *args, _bits=bits: self._deterministic(
                    _bits,
                    *args,
                )
        return getattr(self._runtime, name)

    def _deterministic(
        self,
        bits: int,
        vector: torch.Tensor,
        packed: torch.Tensor,
        output: torch.Tensor,
        lookup: torch.Tensor,
        width: int,
        outliers: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        api = (
            f"vecquant{bits}matmul_nuq_perchannel_transposed_"
            "mha_batched_fused_opt2_deterministic_out"
        )
        getattr(self._runtime, api)(
            vector,
            packed,
            output,
            lookup,
            width,
            outliers,
            indices,
            self._workspaces[bits],
        )
        if int(self._workspaces[bits].data_ptr()) != self._pointers[bits]:
            raise AssertionError("fixture workspace pointer changed")


def _run_fixtures(
    runtime: ModuleType,
    fixture_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    module_path = (
        source_root.resolve(strict=True)
        / "tests_phase11p"
        / "phase11p_cuda_validation.py"
    )
    specification = importlib.util.spec_from_file_location(
        "phase11p_cuda_validation",
        module_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Phase 11P fixture validator loader is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.run_all_fixtures(
        _FixtureRuntime(runtime),
        fixture_root.resolve(strict=True),
    )
    return {
        "mode": "fixtures",
        "cases": 9,
        "payload_metadata_sparse_sink_store_append": "EXACT",
        "decode_tolerance": {"atol": ATOL, "rtol": RTOL},
        "passed": True,
    }


def _run_stream(runtime: ModuleType) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for bits in (3, 2):
        canonical = _synthetic_inputs(bits)
        _call(runtime, bits, canonical)
        torch.cuda.synchronize()
        expected_sha = _tensor_sha256(canonical["output"])

        inputs = _synthetic_inputs(bits)
        expected_vector = inputs["vector"].clone()
        inputs["vector"].zero_()
        side = torch.cuda.Stream()
        complete = torch.cuda.Event()
        with torch.cuda.stream(side):
            torch.cuda._sleep(50_000_000)
            inputs["vector"].copy_(expected_vector)
            _call(runtime, bits, inputs)
            complete.record(side)
        torch.cuda.current_stream().wait_event(complete)
        torch.cuda.synchronize()
        observed_sha = _tensor_sha256(inputs["output"])
        if observed_sha != expected_sha:
            raise AssertionError(f"q{bits} non-default stream ordering failed")
        results.append(
            {
                "bits": bits,
                "output_sha256": observed_sha,
                "side_stream": int(side.cuda_stream),
                "default_stream": int(
                    torch.cuda.default_stream().cuda_stream
                ),
            }
        )
    return {"mode": "stream", "cases": results, "passed": True}


def _run_graph(runtime: ModuleType) -> dict[str, Any]:
    inputs = {bits: _synthetic_inputs(bits) for bits in (3, 2)}

    def operation() -> None:
        for bits in (3, 2):
            _call(runtime, bits, inputs[bits])

    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    eager_hashes = {
        str(bits): _tensor_sha256(inputs[bits]["output"])
        for bits in (3, 2)
    }
    pointers = {bits: _pointers(inputs[bits]) for bits in (3, 2)}
    eager_allocation = audit_cuda_allocations(
        operation,
        device=torch.device("cuda:0"),
    )
    if not eager_allocation.passed:
        raise AssertionError("q3/q2 eager decode allocated")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    graph.replay()
    torch.cuda.synchronize()
    graph_allocation = audit_cuda_allocations(
        graph.replay,
        device=torch.device("cuda:0"),
    )
    if not graph_allocation.passed:
        raise AssertionError("q3/q2 graph replay allocated")
    replay_hashes: dict[str, list[str]] = {"3": [], "2": []}
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize()
        for bits in (3, 2):
            replay_hashes[str(bits)].append(
                _tensor_sha256(inputs[bits]["output"])
            )
    for bits in (3, 2):
        observed = set(replay_hashes[str(bits)])
        if observed != {eager_hashes[str(bits)]}:
            raise AssertionError(f"q{bits} graph output differs")
        if pointers[bits] != _pointers(inputs[bits]):
            raise AssertionError(f"q{bits} graph pointer changed")
    graph.reset()
    return {
        "mode": "graph",
        "capture": "PASS",
        "replay": "PASS",
        "eager_output_sha256": eager_hashes,
        "replay_output_sha256": replay_hashes,
        "eager_allocation": eager_allocation.to_dict(),
        "replay_allocation": graph_allocation.to_dict(),
        "caller_owned_pointers_stable": True,
        "passed": True,
    }


def _reset_cuda_for_sanitizer() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()
    torch.cuda.synchronize()
    cudart = ctypes.CDLL("libcudart.so.13")
    reset = cudart.cudaDeviceReset
    reset.argtypes = []
    reset.restype = ctypes.c_int
    status = int(reset())
    if status != 0:
        raise RuntimeError(f"cudaDeviceReset failed with status {status}")


def _run_sanitizer(runtime: ModuleType) -> dict[str, Any]:
    inputs = {bits: _synthetic_inputs(bits) for bits in (3, 2)}
    for bits in (3, 2):
        _call(runtime, bits, inputs[bits])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for bits in (3, 2):
            _call(runtime, bits, inputs[bits])
    graph.replay()
    torch.cuda.synchronize()
    hashes = {
        str(bits): _tensor_sha256(inputs[bits]["output"])
        for bits in (3, 2)
    }
    graph.reset()
    del graph
    del inputs
    _reset_cuda_for_sanitizer()
    return {
        "mode": "sanitizer",
        "eager": "PASS",
        "graph_replay": "PASS",
        "output_sha256": hashes,
        "passed": True,
    }


def main() -> None:
    arguments = _parse_args()
    _require_environment()
    runtime = _load_extension(
        arguments.extension,
        arguments.extension_sha256,
    )
    if arguments.mode == "determinism":
        result = _run_determinism(runtime)
    elif arguments.mode == "fixtures":
        if arguments.fixture_root is None or arguments.source_root is None:
            raise RuntimeError("fixtures mode requires fixture and source roots")
        result = _run_fixtures(
            runtime,
            arguments.fixture_root,
            arguments.source_root,
        )
    elif arguments.mode == "stream":
        result = _run_stream(runtime)
    elif arguments.mode == "graph":
        result = _run_graph(runtime)
    else:
        result = _run_sanitizer(runtime)
    result.update(
        {
            "schema_version": (
                "kvbench-phase11dq23-long-context-cuda-validation-1.0.0"
            ),
            "authorized_image_digest": AUTHORIZED_IMAGE,
            "extension_sha256": arguments.extension_sha256,
            "performance_timing": False,
        }
    )
    print(
        json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


if __name__ == "__main__":
    main()

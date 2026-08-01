#!/usr/bin/env python3
"""Focused exact-container B=8 compressed-adapter sanitizer probe."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import torch

from kvbench.adapters.base import MethodRuntimeContext
from kvbench.adapters.kivi import KIVIMethodAdapter
from kvbench.adapters.kvquant import KVQuantMethodAdapter
from kvbench.adapters.turboquant import TurboQuantMethodAdapter
from kvbench.runtime.cuda_graph import capture_fixed_graph
from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VALUE,
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    require_authorized_cuda_environment,
)
from kvbench.third_party.vllm_turboquant.compat import _build_hadamard_cached
from kvbench.schema.phase13b import (
    PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
    PHASE13B_CONFIGURATIONS,
)


_CONFIGURATION_MAP = {
    "tq_4bit_nc": (TurboQuantMethodAdapter, "turboquant_4bit_nc"),
    "tq_k3v4_nc": (TurboQuantMethodAdapter, "turboquant_k3v4_nc"),
    "tq_3bit_nc": (TurboQuantMethodAdapter, "turboquant_3bit_nc"),
    "k4v4": (KIVIMethodAdapter, "k4v4"),
    "k2v4": (KIVIMethodAdapter, "k2v4"),
    "k2v2": (KIVIMethodAdapter, "k2v2"),
    "kvq4": (KVQuantMethodAdapter, "kvq4"),
    "kvq3": (KVQuantMethodAdapter, "kvq3"),
    "kvq2": (KVQuantMethodAdapter, "kvq2"),
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        required=True,
        choices=PHASE13B_CONFIGURATIONS,
    )
    parser.add_argument("--batch-size", required=True, type=int, choices=(8,))
    parser.add_argument("--image-config-digest", required=True)
    return parser.parse_args(argv)


def _require_exact_environment(declared_digest: str) -> dict[str, Any]:
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get(PHASE6_IMAGE_ENVIRONMENT_VARIABLE)
        != PHASE13B_AUTHORIZED_CONTAINER_DIGEST
        or os.environ.get(PHASE6_CONTAINER_ENVIRONMENT_VARIABLE)
        != PHASE6_CONTAINER_ENVIRONMENT_VALUE
    ):
        raise RuntimeError(
            "Phase 13B sanitizer execution requires the exact authorized "
            "Measurement Container"
        )
    return require_authorized_cuda_environment(declared_digest)


def _runtime_context(configuration: str) -> MethodRuntimeContext:
    return MethodRuntimeContext(
        model_id="phase13b-compressed-batch-sanitizer",
        model_revision="0e9e39f249a16976918f6564b8830bc894c89659",
        backend_id=f"phase13b-{configuration}",
        backend_fingerprint=hashlib.sha256(
            f"phase13b-{configuration}".encode("ascii")
        ).hexdigest(),
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


def _identical_inputs(
    *,
    batch: int,
    capacity: int,
    device: torch.device,
    seed: int,
) -> tuple[Any, Any, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    key = torch.randn(
        (1, 8, capacity, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).expand(batch, -1, -1, -1).clone()
    value = torch.randn(
        (1, 8, capacity, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).expand(batch, -1, -1, -1).clone()
    query = torch.randn(
        (1, 32, 1, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).expand(batch, -1, -1, -1).clone()
    return key, value, query


def _release_tracked_cuda_storages() -> None:
    storages: dict[int, object] = {}
    tracked = gc.get_objects()
    for value in tracked:
        if type(value) is not torch.Tensor or not value.is_cuda:
            continue
        try:
            storage = value.untyped_storage()
        except RuntimeError:
            continue
        if int(storage.nbytes()) != 0:
            storages.setdefault(int(storage._cdata), storage)
    del tracked
    for storage in storages.values():
        storage.resize_(0)
    storages.clear()
    gc.collect()


def _reset_cuda_for_memcheck() -> None:
    torch.cuda.synchronize()
    blas_handle = int(torch.cuda.current_blas_handle())
    _build_hadamard_cached.cache_clear()
    _release_tracked_cuda_storages()
    for _ in range(8):
        gc.collect()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()
        torch.cuda.synchronize()
        if (
            int(torch.cuda.memory_allocated()) == 0
            and int(torch.cuda.memory_reserved()) == 0
        ):
            break
    else:
        raise RuntimeError("Phase 13B sanitizer allocator did not drain")

    cublas = ctypes.CDLL("libcublas.so.13")
    destroy = cublas.cublasDestroy_v2
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_int
    if int(destroy(ctypes.c_void_p(blas_handle))) != 0:
        raise RuntimeError("cublasDestroy_v2 failed")
    cudart = ctypes.CDLL("libcudart.so.13")
    reset = cudart.cudaDeviceReset
    reset.argtypes = []
    reset.restype = ctypes.c_int
    if int(reset()) != 0:
        raise RuntimeError("cudaDeviceReset failed")


def _run(configuration: str, batch: int) -> dict[str, Any]:
    adapter_class, adapter_configuration = _CONFIGURATION_MAP[configuration]
    family = (
        "turboquant"
        if configuration.startswith("tq_")
        else "kvquant"
        if configuration.startswith("kvq")
        else "kivi"
    )
    capacity = 34 if family == "kivi" else 18
    prefix = 33 if family == "kivi" else 17
    layer = 2 if family == "turboquant" else 0
    device = torch.device("cuda:0")
    key, value, query = _identical_inputs(
        batch=batch,
        capacity=capacity,
        device=device,
        seed=13_000 + tuple(PHASE13B_CONFIGURATIONS).index(configuration),
    )
    position_dtype = torch.int32 if family == "turboquant" else torch.int64
    positions = torch.arange(capacity, dtype=position_dtype, device=device)
    method = adapter_class(_runtime_context(configuration), adapter_configuration)
    if hasattr(method, "prepare_runtime"):
        method.prepare_runtime()
    cache = method.allocate(batch_size=batch, capacity=capacity, device=device)
    if family == "kvquant":
        method.initialize_cache_untimed(cache)
    else:
        cache.initialize_deterministic()
    pointers_before = cache.pointers()
    cache.prepare_prefill(prefix)
    store_kwargs = (
        {"key_pre_rope_states": key[:, :, :prefix, :]}
        if family == "kvquant"
        else {}
    )
    method.store_prefill(
        cache,
        key[:, :, :prefix, :],
        value[:, :, :prefix, :],
        layer,
        positions[:prefix],
        **store_kwargs,
    )
    cache.complete_prefill()
    append_position = positions[prefix : prefix + 1]
    if family == "kvquant":
        cache.bind_fixed_position_tensor_untimed(
            append_position,
            logical_position=prefix,
        )
    cache.prepare_fixed(prefix)
    attention = SimpleNamespace(layer_idx=layer)

    def operation() -> Any:
        append_kwargs = (
            {"key_pre_rope_states": key[:, :, prefix : prefix + 1, :]}
            if family == "kvquant"
            else {}
        )
        handles = method.append_decode(
            cache,
            key[:, :, prefix : prefix + 1, :],
            value[:, :, prefix : prefix + 1, :],
            layer,
            append_position,
            **append_kwargs,
        )
        return method.decode_attention(
            attention,
            query,
            handles[0],
            handles[1],
            scaling=1.0 / math.sqrt(128),
        )

    operation()
    graph = capture_fixed_graph(operation, warmup_steps=1, device=device)
    first = graph.replay()
    second = graph.replay()
    torch.cuda.synchronize(device=device)
    output = second.detach().cpu()
    row_exact = all(torch.equal(output[0:1], output[index : index + 1]) for index in range(1, batch))
    finite = bool(torch.isfinite(output).all())
    graph_exact = bool(torch.equal(first.detach().cpu(), output))
    graph_record = graph.to_dict()
    graph.graph.reset()
    if (
        not row_exact
        or not finite
        or not graph_exact
        or graph_record["fallback"] is not False
        or cache.pointers() != pointers_before
    ):
        raise RuntimeError("Phase 13B sanitizer numerical/graph contract differs")
    geometry = cache.gqa_geometry()
    if (
        geometry["num_query_heads"] != 32
        or geometry["num_kv_heads"] != 8
        or geometry["gqa_group_size"] != 4
        or geometry["native_kv_head_storage"] is not True
        or geometry["query_head_sized_kv_cache"] is not False
    ):
        raise RuntimeError("Phase 13B sanitizer GQA geometry differs")
    return {
        "batch_size": batch,
        "configuration": configuration,
        "finite": True,
        "graph_capture": "PASS",
        "graph_replay": "PASS",
        "native_gqa": "32Q/8KV",
        "pointer_stability": "PASS",
        "row_exact": True,
        "timing_collected": False,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    failures: list[dict[str, str]] = []
    result: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    try:
        environment = _require_exact_environment(arguments.image_config_digest)
        result = _run(arguments.configuration, arguments.batch_size)
    except Exception as error:
        failures.append({"type": type(error).__name__, "message": str(error)})
        error.__traceback__ = None
        del error
    finally:
        if environment is not None:
            try:
                _reset_cuda_for_memcheck()
            except Exception as error:
                failures.append(
                    {"type": type(error).__name__, "message": str(error)}
                )
                error.__traceback__ = None
                del error
    if failures:
        print(
            json.dumps(
                {"failures": failures, "status": "FAIL"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "authorized_container_digest": environment["container_digest"],
                "result": result,
                "status": "PASS",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)

#!/usr/bin/env python3
"""Minimal store, append, and decode probe for Compute Sanitizer."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys

import torch

from kvbench.runtime.turboquant_admission import (
    evaluate_fixture_configuration,
    require_authorized_cuda_environment,
)
from kvbench.runtime.turboquant_cache import TURBOQUANT_MANDATORY_CONFIGS
from kvbench.third_party.vllm_turboquant.compat import (
    _build_hadamard_cached,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        required=True,
        choices=TURBOQUANT_MANDATORY_CONFIGS,
    )
    parser.add_argument("--image-config-digest", required=True)
    return parser.parse_args(argv)


def _release_sanitizer_cuda_state() -> None:
    """Release the isolated probe context before memcheck leak reporting."""

    torch.cuda.synchronize()
    blas_handle = int(torch.cuda.current_blas_handle())
    _build_hadamard_cached.cache_clear()
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()
    torch.cuda.synchronize()
    gc.collect()

    cublas = ctypes.CDLL("libcublas.so.13")
    destroy = cublas.cublasDestroy_v2
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_int
    destroy_result = int(destroy(ctypes.c_void_p(blas_handle)))
    if destroy_result != 0:
        raise RuntimeError(
            f"cublasDestroy_v2 failed with status {destroy_result}"
        )

    cudart = ctypes.CDLL("libcudart.so.13")
    reset = cudart.cudaDeviceReset
    reset.argtypes = []
    reset.restype = ctypes.c_int
    reset_result = int(reset())
    if reset_result != 0:
        raise RuntimeError(
            f"cudaDeviceReset failed with status {reset_result}"
        )


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    environment = require_authorized_cuda_environment(
        arguments.image_config_digest
    )
    result = evaluate_fixture_configuration(arguments.configuration)
    if result.get("passed") is not True:
        raise RuntimeError("TurboQuant sanitizer probe did not conform")
    del result
    _release_sanitizer_cuda_state()
    print(
        json.dumps(
            {
                "configuration": arguments.configuration,
                "container_digest": environment["container_digest"],
                "operation": "store_append_decode",
                "status": "pass",
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

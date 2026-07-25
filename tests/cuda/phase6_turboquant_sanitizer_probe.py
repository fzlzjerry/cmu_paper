#!/usr/bin/env python3
"""Minimal store, append, and decode probe for Compute Sanitizer."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys

import torch

from kvbench.runtime.turboquant_admission import (
    evaluate_fixture_configuration,
    require_authorized_cuda_environment,
)
from kvbench.runtime.turboquant_cache import TURBOQUANT_MANDATORY_CONFIGS


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
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()
    torch.cuda.synchronize()
    gc.collect()

    cudart = ctypes.CDLL("libcudart.so.13")
    reset = cudart.cudaDeviceReset
    reset.argtypes = []
    reset.restype = ctypes.c_int
    result = int(reset())
    if result != 0:
        raise RuntimeError(f"cudaDeviceReset failed with status {result}")


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
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the E00 CUDA certification extension for one supplied architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


MODULE_NAME = "e00_cuda_cert"
ARCH_PATTERN = re.compile(r"^(?P<major>[1-9][0-9]*)\.(?P<minor>[0-9]+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_architecture(raw_arch: str) -> tuple[str, str, str]:
    match = ARCH_PATTERN.fullmatch(raw_arch)
    if match is None:
        raise ValueError(
            "E00_DETECTED_ARCH must use canonical '<major>.<minor>' form "
            "(for example, '12.0')"
        )
    major = match.group("major")
    minor = match.group("minor")
    return major, minor, f"{major}{minor}"


def _require_empty_build_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"build path is not a directory: {path}")
        if next(path.iterdir(), None) is not None:
            raise ValueError(f"build directory must be initially empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def _write_json_exclusively(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def build_extension(build_directory: Path, *, verbose: bool) -> dict[str, Any]:
    raw_arch = os.environ.get("E00_DETECTED_ARCH")
    if raw_arch is None or not raw_arch.strip():
        raise ValueError("E00_DETECTED_ARCH is required and must not be empty")
    if raw_arch != raw_arch.strip():
        raise ValueError("E00_DETECTED_ARCH must not contain surrounding whitespace")
    major, minor, arch_code = _parse_architecture(raw_arch)

    raw_cuda_home = os.environ.get("CUDA_HOME")
    if raw_cuda_home is None or not raw_cuda_home.strip():
        raise ValueError("CUDA_HOME is required and must point to the CUDA toolkit")
    cuda_home = Path(raw_cuda_home).expanduser().resolve(strict=True)
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise ValueError(f"CUDA_HOME does not contain bin/nvcc: {cuda_home}")

    build_directory = build_directory.expanduser().resolve()
    _require_empty_build_directory(build_directory)

    # Import only after CUDA_HOME has been validated so cpp_extension observes
    # the explicit toolkit selection rather than guessing a floating default.
    from torch.utils import cpp_extension

    resolved_cpp_cuda_home = (
        Path(cpp_extension.CUDA_HOME).resolve()
        if cpp_extension.CUDA_HOME is not None
        else None
    )
    if resolved_cpp_cuda_home != cuda_home:
        raise RuntimeError(
            "PyTorch cpp_extension CUDA_HOME does not match the explicit "
            f"CUDA_HOME: {resolved_cpp_cuda_home!s} != {cuda_home!s}"
        )

    source_directory = Path(__file__).resolve().parent
    sources = [
        source_directory / "binding.cpp",
        source_directory / "xor_kernel.cu",
    ]
    header = source_directory / "xor_kernel.h"
    for source in [*sources, header]:
        if not source.is_file():
            raise FileNotFoundError(source)

    # Both flags are derived solely from the independently supplied detected
    # architecture: one exact native cubin plus one forward-compatible PTX.
    native_flag = f"-gencode=arch=compute_{arch_code},code=sm_{arch_code}"
    ptx_flag = f"-gencode=arch=compute_{arch_code},code=compute_{arch_code}"
    module = cpp_extension.load(
        name=MODULE_NAME,
        sources=[str(source) for source in sources],
        build_directory=str(build_directory),
        extra_cflags=["-O2", "-std=c++20"],
        extra_cuda_cflags=[
            "-O2",
            "-std=c++20",
            "-lineinfo",
            native_flag,
            ptx_flag,
        ],
        with_cuda=True,
        is_python_module=True,
        keep_intermediates=True,
        verbose=verbose,
    )
    module_path = Path(module.__file__).resolve(strict=True)

    return {
        "architecture": {
            "canonical": raw_arch,
            "compute": f"compute_{arch_code}",
            "sass": f"sm_{arch_code}",
        },
        "build_directory": str(build_directory),
        "cuda_home": str(cuda_home),
        "module_name": MODULE_NAME,
        "module_path": str(module_path),
        "module_sha256": _sha256(module_path),
        "nvcc": str(nvcc),
        "source_sha256": {
            source.name: _sha256(source) for source in [*sources, header]
        },
        "cxx_flags": ["-O2", "-std=c++20"],
        "cuda_flags": [
            "-O2",
            "-std=c++20",
            "-lineinfo",
            native_flag,
            ptx_flag,
        ],
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-directory",
        type=Path,
        required=True,
        help="new or empty directory that will contain the extension build",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional new file for the machine-readable build result",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="emit verbose cpp_extension/ninja build diagnostics",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        payload = build_extension(args.build_directory, verbose=args.verbose)
        if args.json_output is not None:
            _write_json_exclusively(args.json_output.resolve(), payload)
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as error:  # fail closed while preserving a concise cause
        print(f"E00 extension build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

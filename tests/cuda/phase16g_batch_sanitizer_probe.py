#!/usr/bin/env python3
"""Focused B=16 custom-kernel sanitizer probe for Phase 16G."""

from __future__ import annotations

import argparse
import json
import os
import sys

from tests.cuda import phase13b_batch_sanitizer_probe as phase13b
from scripts.phase16g_batch_geometry_admission import SANITIZER_CONFIGURATIONS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        required=True,
        choices=SANITIZER_CONFIGURATIONS,
    )
    parser.add_argument("--batch-size", required=True, type=int, choices=(16,))
    parser.add_argument("--image-config-digest", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    failures: list[dict[str, str]] = []
    result = None
    environment = None
    try:
        environment = phase13b._require_exact_environment(
            arguments.image_config_digest
        )
        result = phase13b._run(arguments.configuration, arguments.batch_size)
    except Exception as error:
        failures.append({"type": type(error).__name__, "message": str(error)})
        error.__traceback__ = None
        del error
    finally:
        if environment is not None:
            try:
                phase13b._reset_cuda_for_memcheck()
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

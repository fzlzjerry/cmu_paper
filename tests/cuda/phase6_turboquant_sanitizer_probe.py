#!/usr/bin/env python3
"""Minimal store, append, and decode probe for Compute Sanitizer."""

from __future__ import annotations

import argparse
import json
import sys

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


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    environment = require_authorized_cuda_environment(
        arguments.image_config_digest
    )
    result = evaluate_fixture_configuration(arguments.configuration)
    if result.get("passed") is not True:
        raise RuntimeError("TurboQuant sanitizer probe did not conform")
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

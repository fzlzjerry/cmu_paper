#!/usr/bin/env python3
"""Verify installed E00 wheel RECORD hashes and reject unowned executable files."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import site
import sys
from typing import Any


RELEVANT_SUFFIXES = (".py", ".pth", ".so")


def _is_relevant(path: Path) -> bool:
    return path.name.endswith(RELEVANT_SUFFIXES)


def _sha256_record_digest(path: Path) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
    return encoded.rstrip(b"=").decode("ascii")


def inspect_environment() -> dict[str, Any]:
    prefix = Path(sys.prefix).resolve(strict=True)
    site_roots = [
        Path(value).resolve(strict=True)
        for value in site.getsitepackages()
        if Path(value).is_dir()
    ]
    errors: list[str] = []
    owned_paths: set[Path] = set()
    distribution_count = 0
    record_entry_count = 0
    hashed_entry_count = 0
    permitted_unhashed_entry_count = 0

    distributions = sorted(
        importlib.metadata.distributions(),
        key=lambda item: str(item.metadata.get("Name", "")).lower(),
    )
    for distribution in distributions:
        distribution_count += 1
        name = str(distribution.metadata.get("Name", "<unnamed>"))
        record_text = distribution.read_text("RECORD")
        if record_text is None:
            errors.append(f"distribution has no RECORD: {name}")
            continue
        try:
            rows = list(csv.reader(io.StringIO(record_text)))
        except csv.Error as error:
            errors.append(f"cannot parse RECORD for {name}: {error}")
            continue

        for row_number, row in enumerate(rows, start=1):
            record_entry_count += 1
            if len(row) != 3 or not row[0]:
                errors.append(f"malformed RECORD row {name}:{row_number}")
                continue
            recorded_path, recorded_hash, recorded_size = row
            candidate = Path(distribution.locate_file(recorded_path))
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as error:
                errors.append(
                    f"missing RECORD file {name}:{recorded_path}: {error}"
                )
                continue
            try:
                resolved.relative_to(prefix)
            except ValueError:
                errors.append(
                    f"RECORD path escapes the Python environment "
                    f"{name}:{recorded_path}"
                )
                continue
            if not resolved.is_file():
                errors.append(f"RECORD path is not a file {name}:{recorded_path}")
                continue
            owned_paths.add(resolved)

            if not recorded_hash:
                permitted = (
                    recorded_path.endswith(".dist-info/RECORD")
                    or recorded_path.endswith(".pyc")
                )
                if not permitted or _is_relevant(resolved):
                    errors.append(
                        f"security-relevant RECORD entry is unhashed "
                        f"{name}:{recorded_path}"
                    )
                else:
                    permitted_unhashed_entry_count += 1
                continue

            algorithm, separator, expected_digest = recorded_hash.partition("=")
            if separator != "=" or algorithm != "sha256" or not expected_digest:
                errors.append(
                    f"unsupported RECORD hash {name}:{recorded_path}: "
                    f"{recorded_hash!r}"
                )
                continue
            observed_digest = _sha256_record_digest(resolved)
            if observed_digest != expected_digest:
                errors.append(
                    f"RECORD hash mismatch {name}:{recorded_path}"
                )
            try:
                expected_size = int(recorded_size)
            except ValueError:
                errors.append(
                    f"invalid RECORD size {name}:{recorded_path}: "
                    f"{recorded_size!r}"
                )
            else:
                if resolved.stat().st_size != expected_size:
                    errors.append(
                        f"RECORD size mismatch {name}:{recorded_path}"
                    )
            hashed_entry_count += 1

    relevant_file_count = 0
    for site_root in site_roots:
        for path in sorted(site_root.rglob("*")):
            if path.is_symlink():
                if _is_relevant(path):
                    errors.append(f"security-relevant site file is a symlink: {path}")
                continue
            if not path.is_file() or not _is_relevant(path):
                continue
            relevant_file_count += 1
            resolved = path.resolve(strict=True)
            if resolved not in owned_paths:
                errors.append(
                    f"unowned security-relevant site file: "
                    f"{path.relative_to(prefix)}"
                )

    return {
        "status": "PASS" if not errors else "FAIL",
        "python_prefix": str(prefix),
        "site_roots": [str(path) for path in site_roots],
        "distribution_count": distribution_count,
        "record_entry_count": record_entry_count,
        "hashed_entry_count": hashed_entry_count,
        "permitted_unhashed_entry_count": permitted_unhashed_entry_count,
        "relevant_file_count": relevant_file_count,
        "errors": errors,
    }


def main() -> int:
    payload = inspect_environment()
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

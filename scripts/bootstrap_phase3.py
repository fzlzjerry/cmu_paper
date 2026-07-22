#!/usr/bin/env python3
"""Install or verify the isolated, hash-locked Phase 3 Python additions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
E00_PYTHON: Final = ROOT / ".venv" / "bin" / "python"
E00_LOCK: Final = ROOT / "preflight" / "requirements-e00.txt"
PHASE3_LOCK: Final = ROOT / "preflight" / "requirements-phase3.txt"
PHASE3_ROOT: Final = ROOT / ".phase3"
TARGET: Final = PHASE3_ROOT / "site-packages"

EXPECTED_E00_LOCK_SHA256: Final = (
    "aafe68e54cb316d6bb673dbc42087b2f971ac94668973cc3f8cc555d8a0dbb29"
)
INHERITED: Final = {
    "filelock": "3.29.0",
    "fsspec": "2026.4.0",
    "torch": "2.12.1+cu130",
    "triton": "3.7.1",
    "typing-extensions": "4.15.0",
}
ADDITIONS: Final = {
    "certifi": "2026.7.22",
    "charset-normalizer": "3.4.9",
    "hf-xet": "1.5.2",
    "huggingface-hub": "0.36.2",
    "idna": "3.18",
    "numpy": "2.5.1",
    "packaging": "26.2",
    "pyyaml": "6.0.3",
    "regex": "2026.7.19",
    "requests": "2.34.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "tqdm": "4.69.0",
    "transformers": "4.57.6",
    "urllib3": "2.7.0",
}


class BootstrapError(RuntimeError):
    """Raised when the Phase 3 dependency boundary is not exact."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def distributions_at(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_dir():
        return result
    for distribution in importlib.metadata.distributions(path=[str(path)]):
        name = distribution.metadata.get("Name")
        if not name:
            raise BootstrapError(f"distribution without Name metadata under {path}")
        normalized = canonical_name(name)
        if normalized in result:
            raise BootstrapError(f"duplicate distribution in target: {normalized}")
        result[normalized] = distribution.version
    return result


def inherited_versions() -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json;"
        f"names={list(INHERITED)!r};"
        "print(json.dumps({n:m.version(n) for n in names},sort_keys=True))"
    )
    completed = subprocess.run(
        [str(E00_PYTHON), "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BootstrapError(
            "cannot inspect inherited E00 distributions: " + completed.stderr.strip()
        )
    return json.loads(completed.stdout)


def verify_base() -> None:
    if not E00_PYTHON.is_file():
        raise BootstrapError(f"certified E00 interpreter is missing: {E00_PYTHON}")
    actual_lock = sha256(E00_LOCK)
    if actual_lock != EXPECTED_E00_LOCK_SHA256:
        raise BootstrapError(
            f"E00 lock hash mismatch: expected {EXPECTED_E00_LOCK_SHA256}, got {actual_lock}"
        )
    actual_versions = inherited_versions()
    if actual_versions != INHERITED:
        raise BootstrapError(
            f"inherited E00 version mismatch: expected {INHERITED}, got {actual_versions}"
        )


def import_versions(path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    prior = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(path) if not prior else f"{path}:{prior}"
    code = (
        "import json,numpy,safetensors,tokenizers,torch,transformers,huggingface_hub;"
        "from huggingface_hub.cli.hf import main as hf_main;"
        "assert callable(hf_main);"
        "print(json.dumps({'numpy':numpy.__version__,'safetensors':safetensors.__version__,"
        "'tokenizers':tokenizers.__version__,'torch':torch.__version__,"
        "'transformers':transformers.__version__,'huggingface_hub':huggingface_hub.__version__},"
        "sort_keys=True))"
    )
    completed = subprocess.run(
        [str(E00_PYTHON), "-c", code],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BootstrapError("Phase 3 import check failed: " + completed.stderr.strip())
    return json.loads(completed.stdout)


def verify_target(path: Path) -> dict[str, object]:
    actual = distributions_at(path)
    if actual != ADDITIONS:
        raise BootstrapError(f"target mismatch: expected {ADDITIONS}, got {actual}")
    imports = import_versions(path)
    expected_imports = {
        "huggingface_hub": ADDITIONS["huggingface-hub"],
        "numpy": ADDITIONS["numpy"],
        "safetensors": ADDITIONS["safetensors"],
        "tokenizers": ADDITIONS["tokenizers"],
        "torch": INHERITED["torch"],
        "transformers": ADDITIONS["transformers"],
    }
    if imports != expected_imports:
        raise BootstrapError(
            f"import identity mismatch: expected {expected_imports}, got {imports}"
        )
    return {
        "additions": dict(sorted(actual.items())),
        "imports": imports,
    }


def install() -> None:
    verify_base()
    if TARGET.exists():
        raise BootstrapError(f"refusing to replace existing Phase 3 target: {TARGET}")
    PHASE3_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="site-packages.staging.", dir=PHASE3_ROOT))
    command = [
        str(E00_PYTHON),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--require-hashes",
        "--only-binary=:all:",
        "--target",
        str(staging),
        "-r",
        str(PHASE3_LOCK),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise BootstrapError(
            f"install failed with {completed.returncode}; retained staging at {staging}"
        )
    verify_target(staging)
    staging.rename(TARGET)


def verify() -> dict[str, object]:
    verify_base()
    target_report = verify_target(TARGET)
    return {
        "status": "pass",
        "e00_lock_sha256": sha256(E00_LOCK),
        "phase3_lock_sha256": sha256(PHASE3_LOCK),
        "target": str(TARGET),
        "inherited": dict(sorted(INHERITED.items())),
        **target_report,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "verify"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.action == "install":
            install()
        report = verify()
    except BootstrapError as error:
        print(json.dumps({"status": "fail", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

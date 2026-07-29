#!/usr/bin/env python3
"""Validate the Decision 0025 aggregate KVQuant source patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import scripts.validate_kvquant_gqa_patch as base


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "third_party/patches/kvquant/graphsafe-kvq3-manifest.json"
)
PATCH = (
    "third_party/patches/kvquant/"
    "0002-graphsafe-kvq3-deterministic.patch"
)
PARENT_COMMIT = "7fa389ecf5a5e198c76096d52fc2949dde844532"
CORRECTED_PATH = "deployment/kvquant/quant_cuda_kernel.cu"


def validate(source_root: Path) -> dict[str, object]:
    manifest = base._read_json(MANIFEST, "Decision 0025 patch manifest")
    if manifest.get("schema_version") != base.SCHEMA_VERSION:
        raise base.ValidationError("unsupported Decision 0025 manifest schema")
    authority = manifest.get("authority", {})
    if (
        authority.get("base_decision") != "0021"
        or authority.get("graphsafe_decision") != "0024"
        or authority.get("correction_decision") != "0025"
        or authority.get("official_gqa_support_claimed") is not False
    ):
        raise base.ValidationError("Decision 0025 authority mismatch")
    correction = manifest.get("correction", {})
    if (
        correction.get("path") != CORRECTED_PATH
        or correction.get("old_expression") != "deq2[val][k]"
        or correction.get("new_expression") != "deq2[val][off]"
        or correction.get("cross_thread_shared_dependency_after_correction")
        is not False
        or correction.get("barrier_added") is not False
    ):
        raise base.ValidationError("kvq3 correction contract mismatch")

    base.MANIFEST_PATH = MANIFEST
    base.PATCH_RELATIVE_PATH = PATCH
    patch_path, patch_bytes = base._resolve_patch(manifest)
    paths = base._validate_patch_structure(manifest, patch_bytes)
    reconstruction = base._validate_reconstruction(
        source_root,
        manifest,
        patch_path,
    )

    source_root = source_root.resolve(strict=True)
    head = base._run_git(source_root, "rev-parse", "HEAD").strip()
    tree = base._run_git(source_root, "rev-parse", "HEAD^{tree}").strip()
    if (
        head != manifest["source"]["patched_commit"]
        or tree != manifest["source"]["patched_tree"]
    ):
        raise base.ValidationError("local corrected source identity mismatch")
    status = base._run_git(
        source_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if status:
        raise base.ValidationError("corrected source checkout is not clean")

    delta = base._run_git(
        source_root,
        "diff",
        "--unified=0",
        PARENT_COMMIT,
        "HEAD",
        "--",
        CORRECTED_PATH,
    )
    if (
        delta.count("-                float sub = fabsf(newvecval - deq2[val][k]);")
        != 1
        or delta.count(
            "+                float sub = fabsf(newvecval - deq2[val][off]);"
        )
        != 1
    ):
        raise base.ValidationError("parent-relative kvq3 correction is not exact")
    changed = base._run_git(
        source_root,
        "diff",
        "--name-only",
        PARENT_COMMIT,
        "HEAD",
    ).splitlines()
    if changed != [CORRECTED_PATH]:
        raise base.ValidationError("parent-relative source scope is not one file")

    return {
        "status": "PASS",
        "patched_commit": head,
        "patched_tree": tree,
        "aggregate_patch_sha256": base._sha256(patch_bytes),
        "aggregate_changed_paths": paths,
        "parent_relative_changed_paths": changed,
        "reconstruction": reconstruction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = validate(arguments.source_root)
    except (OSError, ValueError, KeyError, base.ValidationError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

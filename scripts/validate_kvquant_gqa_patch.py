#!/usr/bin/env python3
"""Validate the in-repository KVQuant Llama-3.1 GQA patch authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY_ROOT / "third_party/patches/kvquant/manifest.json"
)
FROZEN_PHASE9P_MANIFEST_PATH = (
    REPOSITORY_ROOT / "docs/evidence/phase9p/patch-manifest.json"
)
LOCK_PATH = REPOSITORY_ROOT / "third_party/LOCK.json"
PATCH_RELATIVE_PATH = (
    "third_party/patches/kvquant/0001-llama31-native-gqa.patch"
)
FROZEN_PHASE9P_MANIFEST_SHA256 = (
    "c2390f52af2f4f6d4ef5731f64ed05b9f307e009391f7af7c79baee0209b5e5e"
)
CUSTODY_MANIFEST_SHA256 = (
    "85e76396f058844190620e1cc7d2eef6afba37e83aca87d44f7c5e99c79b7539"
)
SCHEMA_VERSION = "kvbench-kvquant-gqa-source-patch-manifest-1.0.0"
DIFF_HEADER = re.compile(
    rb"^diff --git a/([^\n]+) b/([^\n]+)$",
    flags=re.MULTILINE,
)
INDEX_HEADER = re.compile(
    rb"^index ([0-9a-f]{40})\.\.([0-9a-f]{40})(?: [0-7]{6})?$",
    flags=re.MULTILINE,
)


class ValidationError(RuntimeError):
    """Raised when the stored patch authority is inconsistent."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return environment


def _read_json(path: Path, description: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"{description} must be a JSON object")
    return value


def _read_manifest() -> dict[str, Any]:
    return _read_json(MANIFEST_PATH, "patch manifest")


def _kvquant_lock() -> dict[str, Any]:
    lock = _read_json(LOCK_PATH, "third-party lock")
    sources = lock.get("sources")
    if not isinstance(sources, list):
        raise ValidationError("third-party lock sources must be a list")
    matches = [
        record
        for record in sources
        if isinstance(record, dict) and record.get("id") == "kvquant"
    ]
    if len(matches) != 1:
        raise ValidationError("third-party lock must contain one KVQuant record")
    return matches[0]


def _validate_frozen_authority(
    manifest: dict[str, Any],
    patch_bytes: bytes,
) -> None:
    frozen_bytes = FROZEN_PHASE9P_MANIFEST_PATH.read_bytes()
    if _sha256(frozen_bytes) != FROZEN_PHASE9P_MANIFEST_SHA256:
        raise ValidationError("frozen Phase 9P manifest digest mismatch")
    frozen = _read_json(
        FROZEN_PHASE9P_MANIFEST_PATH,
        "frozen Phase 9P manifest",
    )
    if _sha256(MANIFEST_PATH.read_bytes()) != CUSTODY_MANIFEST_SHA256:
        raise ValidationError("custody manifest digest mismatch")

    for field in ("method_identifier", "human_name"):
        if manifest["authority"][field] != frozen["authority"][field]:
            raise ValidationError(f"authority {field} drifted from Phase 9P")
    for field in (
        "official_gqa_support_claimed",
        "root_license_status",
    ):
        if manifest["authority"][field] != frozen["authority"][field]:
            raise ValidationError(f"authority {field} drifted from Phase 9P")
    for field in (
        "repository",
        "base_commit",
        "base_tree",
        "patched_commit",
        "patched_tree",
    ):
        if manifest["source"][field] != frozen["source"][field]:
            raise ValidationError(f"source {field} drifted from Phase 9P")
    if manifest["source"]["patched_commit_role"] != (
        "historical_validation_identity_not_required_for_reconstruction"
    ):
        raise ValidationError("patched commit role is not explicit")
    if manifest["source"]["durable_execution_authority"] != (
        "base_commit_plus_patch_sha256_plus_patched_tree"
    ):
        raise ValidationError("durable execution authority drifted")

    patch = manifest["patch"]
    frozen_patch = frozen["patch"]
    if patch["path"] != PATCH_RELATIVE_PATH:
        raise ValidationError("patch path is not the fixed repository path")
    if patch["sha256"] != frozen_patch["aggregate_sha256"]:
        raise ValidationError("patch digest drifted from Phase 9P")
    if patch["size_bytes"] != frozen_patch["aggregate_bytes"]:
        raise ValidationError("patch size drifted from Phase 9P")
    if _sha256(patch_bytes) != frozen_patch["aggregate_sha256"]:
        raise ValidationError("stored patch bytes drifted from Phase 9P")

    records = manifest["patched_files"]
    frozen_records = frozen_patch["changed_files"]
    if not isinstance(records, list) or len(records) != len(frozen_records):
        raise ValidationError("patched-file inventory drifted from Phase 9P")
    for record, frozen_record in zip(records, frozen_records, strict=True):
        expected = {
            "path": frozen_record["path"],
            "change_type": frozen_record["change_type"],
            "base_sha256": frozen_record["before_sha256"],
            "patched_sha256": frozen_record["after_sha256"],
        }
        observed = {key: record[key] for key in expected}
        if observed != expected:
            raise ValidationError(
                f"patched-file authority drifted: {expected['path']}"
            )

    locked = _kvquant_lock()
    locked_patch = locked["phase9p_patch"]
    lock_expectations = {
        "repository": manifest["source"]["repository"],
        "revision": manifest["source"]["base_commit"],
        "tree": manifest["source"]["base_tree"],
        "license_verification_status": "unresolved_no_root_license",
    }
    for key, expected in lock_expectations.items():
        if locked[key] != expected:
            raise ValidationError(f"KVQuant lock {key} mismatch")
    patch_expectations = {
        "decision": manifest["authority"]["base_decision"],
        "custody_decision": manifest["authority"]["custody_decision"],
        "method_identifier": manifest["authority"]["method_identifier"],
        "human_name": manifest["authority"]["human_name"],
        "branch": "kvbench/llama31-gqa",
        "patched_commit": manifest["source"]["patched_commit"],
        "patched_tree": manifest["source"]["patched_tree"],
        "aggregate_patch_sha256": patch["sha256"],
        "patch_manifest": FROZEN_PHASE9P_MANIFEST_PATH.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "patch_manifest_sha256": FROZEN_PHASE9P_MANIFEST_SHA256,
        "in_repository_patch": PATCH_RELATIVE_PATH,
        "in_repository_patch_sha256": patch["sha256"],
        "in_repository_manifest": MANIFEST_PATH.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "in_repository_manifest_sha256": CUSTODY_MANIFEST_SHA256,
        "official_gqa_support_claimed": False,
    }
    for key, expected in patch_expectations.items():
        if locked_patch[key] != expected:
            raise ValidationError(f"KVQuant patch lock {key} mismatch")

    source_tree_published = locked_patch["source_tree_published"]
    if type(source_tree_published) is not bool:
        raise ValidationError("KVQuant source-tree publication must be boolean")
    if source_tree_published:
        raise ValidationError("KVQuant source tree must remain unpublished")

    published = locked_patch["patch_published"]
    if type(published) is not bool:
        raise ValidationError("KVQuant patch publication must be boolean")
    expected_status = (
        "main_repository_patch_published"
        if published
        else "main_repository_patch_publication_authorized_pending_push"
    )
    expected_scope = (
        "operator_authorized_public_patch_custody_published"
        if published
        else "operator_authorized_public_patch_custody_pending_push"
    )
    if locked["acquisition_status"] != expected_status:
        raise ValidationError("KVQuant publication status is inconsistent")
    if locked_patch["scope"] != expected_scope:
        raise ValidationError("KVQuant publication scope is inconsistent")


def _run_git(
    cwd: Path,
    *arguments: str,
    binary: bool = False,
) -> bytes | str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        env=_git_environment(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    if result.returncode != 0:
        stderr = (
            result.stderr.decode("utf-8", errors="replace")
            if binary
            else result.stderr
        )
        raise ValidationError(
            f"git {' '.join(arguments)} failed: {stderr.strip()}"
        )
    return result.stdout


def _resolve_patch(manifest: dict[str, Any]) -> tuple[Path, bytes]:
    relative = manifest["patch"]["path"]
    if not isinstance(relative, str):
        raise ValidationError("patch.path must be a string")
    if relative != PATCH_RELATIVE_PATH:
        raise ValidationError("patch path is not the fixed repository path")
    patch_path = (REPOSITORY_ROOT / relative).resolve(strict=True)
    if not patch_path.is_relative_to(REPOSITORY_ROOT):
        raise ValidationError("patch path escapes the repository")
    patch_bytes = patch_path.read_bytes()
    if len(patch_bytes) != manifest["patch"]["size_bytes"]:
        raise ValidationError("stored patch size does not match manifest")
    if _sha256(patch_bytes) != manifest["patch"]["sha256"]:
        raise ValidationError("stored patch SHA-256 does not match manifest")
    return patch_path, patch_bytes


def _validate_patch_structure(
    manifest: dict[str, Any],
    patch_bytes: bytes,
) -> list[str]:
    matches = list(DIFF_HEADER.finditer(patch_bytes))
    records = manifest["patched_files"]
    if len(records) != manifest["patch"]["changed_file_count"]:
        raise ValidationError("changed-file count does not match manifest")
    if len(matches) != len(records):
        raise ValidationError("patch section count does not match manifest")
    if b"GIT binary patch" in patch_bytes:
        raise ValidationError("Phase 9P patch unexpectedly contains binary data")

    expected_paths = [record["path"] for record in records]
    observed_paths: list[str] = []
    for index, match in enumerate(matches):
        left = match.group(1).decode("utf-8")
        right = match.group(2).decode("utf-8")
        if left != right:
            raise ValidationError("rename or copy patch sections are forbidden")
        normalized = PurePosixPath(left)
        if (
            not left
            or normalized.is_absolute()
            or ".." in normalized.parts
            or normalized.as_posix() != left
        ):
            raise ValidationError(f"unsafe or noncanonical patch path: {left}")
        observed_paths.append(left)
        section_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(patch_bytes)
        )
        section = patch_bytes[match.start() : section_end]
        index_match = INDEX_HEADER.search(section)
        if index_match is None:
            raise ValidationError(f"missing full-index header for {left}")
        record = records[index]
        forbidden_mode_or_copy_markers = (
            b"\nold mode ",
            b"\nnew mode ",
            b"\ndeleted file mode ",
            b"\nrename from ",
            b"\nrename to ",
            b"\ncopy from ",
            b"\ncopy to ",
        )
        if any(
            marker in section for marker in forbidden_mode_or_copy_markers
        ):
            raise ValidationError(f"mode, rename, or copy drift: {left}")
        old_blob = index_match.group(1).decode("ascii")
        new_blob = index_match.group(2).decode("ascii")
        expected_old = record["base_git_blob"] or ("0" * 40)
        if old_blob != expected_old:
            raise ValidationError(f"base Git blob mismatch in patch: {left}")
        if new_blob != record["patched_git_blob"]:
            raise ValidationError(f"patched Git blob mismatch in patch: {left}")
        if record["change_type"] == "added":
            if b"\nnew file mode 100644\n" not in section:
                raise ValidationError(f"added-file marker missing: {left}")
        elif record["change_type"] == "modified":
            if b"\nnew file mode " in section:
                raise ValidationError(f"modified file marked as added: {left}")
        else:
            raise ValidationError(f"unsupported change type for {left}")

    if observed_paths != expected_paths:
        raise ValidationError("patch path order does not match manifest")
    if len(set(observed_paths)) != len(observed_paths):
        raise ValidationError("patch contains duplicate paths")
    return observed_paths


def _validate_file(
    path: Path,
    *,
    expected_blob: str,
    expected_sha256: str,
    repository: Path,
) -> None:
    data = path.read_bytes()
    if _sha256(data) != expected_sha256:
        raise ValidationError(f"file SHA-256 mismatch: {path.name}")
    observed_blob = _run_git(repository, "hash-object", str(path))
    if not isinstance(observed_blob, str):
        raise AssertionError("text Git command returned bytes")
    if observed_blob.strip() != expected_blob:
        raise ValidationError(f"file Git blob mismatch: {path.name}")


def _validate_reconstruction(
    source_root: Path,
    manifest: dict[str, Any],
    patch_path: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    if not (source_root / ".git").exists():
        raise ValidationError("source root is not a Git checkout")

    source = manifest["source"]
    base_commit = source["base_commit"]
    base_tree = _run_git(source_root, "rev-parse", f"{base_commit}^{{tree}}")
    if not isinstance(base_tree, str):
        raise AssertionError("text Git command returned bytes")
    if base_tree.strip() != source["base_tree"]:
        raise ValidationError("source checkout does not contain the pinned tree")

    with tempfile.TemporaryDirectory(
        prefix="kvquant-gqa-patch-validation-"
    ) as directory:
        clone = Path(directory) / "source"
        clone_result = subprocess.run(
            (
                "git",
                "clone",
                "--quiet",
                "--no-checkout",
                "--shared",
                str(source_root),
                str(clone),
            ),
            env=_git_environment(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if clone_result.returncode != 0:
            raise ValidationError(
                f"temporary source clone failed: {clone_result.stderr.strip()}"
            )
        _run_git(clone, "config", "--local", "core.autocrlf", "false")
        _run_git(clone, "config", "--local", "core.eol", "lf")
        _run_git(clone, "config", "--local", "core.safecrlf", "false")
        _run_git(
            clone,
            "config",
            "--local",
            "core.attributesFile",
            "/dev/null",
        )
        _run_git(clone, "checkout", "--quiet", "--detach", base_commit)
        status = _run_git(clone, "status", "--porcelain", "--untracked-files=all")
        if not isinstance(status, str) or status:
            raise ValidationError("temporary base checkout is not clean")

        for record in manifest["patched_files"]:
            target = clone / record["path"]
            if record["change_type"] == "added":
                if target.exists():
                    raise ValidationError(
                        f"added path already exists at base: {record['path']}"
                    )
                continue
            _validate_file(
                target,
                expected_blob=record["base_git_blob"],
                expected_sha256=record["base_sha256"],
                repository=clone,
            )

        _run_git(
            clone,
            "apply",
            "--index",
            "--whitespace=nowarn",
            str(patch_path),
        )
        patched_tree = _run_git(clone, "write-tree")
        if not isinstance(patched_tree, str):
            raise AssertionError("text Git command returned bytes")
        if patched_tree.strip() != source["patched_tree"]:
            raise ValidationError("reconstructed tree does not match patched tree")

        changed = _run_git(
            clone,
            "diff",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            "--name-only",
            "-z",
            binary=True,
        )
        if not isinstance(changed, bytes):
            raise AssertionError("binary Git command returned text")
        observed_paths = [
            value.decode("utf-8")
            for value in changed.rstrip(b"\0").split(b"\0")
            if value
        ]
        expected_paths = [
            record["path"] for record in manifest["patched_files"]
        ]
        if observed_paths != expected_paths:
            raise ValidationError("reconstructed changed paths do not match")

        for record in manifest["patched_files"]:
            _validate_file(
                clone / record["path"],
                expected_blob=record["patched_git_blob"],
                expected_sha256=record["patched_sha256"],
                repository=clone,
            )

    return {
        "base_commit": base_commit,
        "base_tree": source["base_tree"],
        "patched_tree": source["patched_tree"],
        "applied_patch_sha256": manifest["patch"]["sha256"],
        "changed_file_count": len(manifest["patched_files"]),
    }


def validate(source_root: Path | None = None) -> dict[str, Any]:
    manifest = _read_manifest()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("unsupported patch-manifest schema")
    authority = manifest["authority"]
    if authority["root_license_status"] != "unresolved_no_root_license":
        raise ValidationError("root-license status must remain unresolved")
    if authority["official_gqa_support_claimed"]:
        raise ValidationError("official upstream GQA support must not be claimed")
    if not authority["public_patch_custody_authorized_by_operator"]:
        raise ValidationError("public patch custody lacks operator authority")
    if manifest["publication"]["upstream_source_tree_vendored"]:
        raise ValidationError("the full upstream source tree must not be vendored")

    patch_path, patch_bytes = _resolve_patch(manifest)
    _validate_frozen_authority(manifest, patch_bytes)
    paths = _validate_patch_structure(manifest, patch_bytes)
    report: dict[str, Any] = {
        "status": "PASS",
        "patch_path": patch_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "patch_sha256": _sha256(patch_bytes),
        "patch_size_bytes": len(patch_bytes),
        "changed_paths": paths,
        "reconstruction": "NOT_REQUESTED",
    }
    if source_root is not None:
        report["reconstruction"] = _validate_reconstruction(
            source_root,
            manifest,
            patch_path,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "optional local Git checkout containing the pinned base commit; "
            "validation uses only an ephemeral clone"
        ),
    )
    arguments = parser.parse_args()
    try:
        report = validate(arguments.source_root)
    except (OSError, ValueError, KeyError, ValidationError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build and validate the append-only Phase 6 R2 outer admission bundle."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any

from kvbench.runtime.artifacts import validate_run_id
from preflight.run_preflight import (
    json_bytes,
    rename_noreplace,
    sha256_file,
    write_exclusive,
)
from scripts.r2_artifact import (
    ValidatedArtifact,
    validate_local_artifact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT_SHA256 = (
    "f003bc3dc5de6b67a6d8f1b8bed7fa49b7f90f9d7edc4d1383e2d97c8aa19d6d"
)
ORIGINAL_BUNDLE_RUN_ID = (
    "phase6-20260726t035257468z-0df5bb4d-4139a6-4bit_nc-fixed-l128-eager"
)
ORIGINAL_BUNDLE_RELATIVE = (
    Path("artifacts") / "phase6" / ORIGINAL_BUNDLE_RUN_ID
)
ORIGINAL_R2_URI = (
    "r2://kvbench-artifacts/kvbench/sha256/"
    f"{ORIGINAL_ROOT_SHA256}/"
)
OUTER_ARTIFACT_ROOT_RELATIVE = Path("artifacts") / "phase6_r2_outer"
OUTER_RECEIPT_RELATIVE = (
    Path("docs")
    / "evidence"
    / "phase6"
    / "r2-admission-outer-publication.json"
)
METHOD_ADMISSION_RELATIVE = (
    Path("docs")
    / "evidence"
    / "phase6"
    / "turboquant-method-admission.json"
)
ORIGINAL_PUBLICATION_RELATIVE = (
    Path("docs")
    / "evidence"
    / "phase6"
    / "r2-admission-publication.json"
)
PASS_REPORT_RELATIVE = (
    Path("docs")
    / "phase_reports"
    / "phase6-turboquant-measurement-adapter-pass.md"
)
REQUIRED_REPOSITORY_FILES = (
    METHOD_ADMISSION_RELATIVE,
    ORIGINAL_PUBLICATION_RELATIVE,
    PASS_REPORT_RELATIVE,
)
ORIGINAL_COPY_PREFIX = PurePosixPath(
    "original", "sha256", ORIGINAL_ROOT_SHA256
)
ORIGINAL_REFERENCE_PATH = PurePosixPath(
    "references", "original-phase6-root.json"
)
ADMISSION_REFERENCES_PATH = PurePosixPath(
    "references", "phase6-admission-runs.json"
)
ADMISSION_RUN_IDS = (
    ORIGINAL_BUNDLE_RUN_ID,
    "phase6-20260726t035319367z-0df5bb4d-11ae26-4bit_nc-fixed-l128-graph",
    "phase6-20260726t035320420z-0df5bb4d-7b10b9-k3v4_nc-fixed-l128-eager",
    "phase6-20260726t035321226z-0df5bb4d-444aeb-k3v4_nc-fixed-l128-graph",
    "phase6-20260726t035322291z-0df5bb4d-bd5031-3bit_nc-fixed-l128-eager",
    "phase6-20260726t035323097z-0df5bb4d-220048-3bit_nc-fixed-l128-graph",
    "phase6-20260726t035325093z-0df5bb4d-2dd3d5-4bit_nc-fixed-l4096-eager",
    "phase6-20260726t035327097z-0df5bb4d-7c8bcd-4bit_nc-fixed-l4096-graph",
    "phase6-20260726t035329322z-0df5bb4d-5428a2-4bit_nc-growing-l128-eager",
)
MANIFEST_SCHEMA = "kvbench-phase6-r2-outer-bundle-1.0.0"
ROOT_REFERENCE_SCHEMA = "kvbench-phase6-r2-inner-root-reference-1.0.0"
RUN_REFERENCES_SCHEMA = "kvbench-phase6-admission-run-references-1.0.0"
INVENTORY_SCHEMA = "kvbench-artifact-inventory-1.0.0"
COMPLETION_SCHEMA = "kvbench-completion-1.0.0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


class OuterBundleError(RuntimeError):
    """The Phase 6 outer bundle failed a narrow build or validation rule."""


@dataclass(frozen=True)
class OuterBundleValidation:
    run_id: str
    root_sha256: str
    object_count: int
    original_object_count: int
    admission_run_count: int
    required_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": (
                "kvbench-phase6-r2-outer-bundle-validation-1.0.0"
            ),
            "status": "PASS",
            "run_id": self.run_id,
            "root_sha256": self.root_sha256,
            "object_count": self.object_count,
            "original_root_sha256": ORIGINAL_ROOT_SHA256,
            "original_object_count": self.original_object_count,
            "admission_run_count": self.admission_run_count,
            "required_paths": list(self.required_paths),
            "receipt_in_bundle": False,
            "performance_claim_eligible": False,
            "phase7_started": False,
            "phase8_started": False,
        }


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite value {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise OuterBundleError(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise OuterBundleError(f"{label} is invalid")
    return payload


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(
            repository_root.resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError) as error:
        raise OuterBundleError(
            "Phase 6 source bundle must be inside the repository"
        ) from error


def _regular_source_files(root: Path) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise OuterBundleError("source bundle is absent or unsafe")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            raise OuterBundleError("source bundle contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file() or metadata.st_nlink != 1:
            raise OuterBundleError("source bundle contains an unsafe entry")
        files[relative] = path
    return files


def _load_admission_run_ids(source_bundle: Path) -> tuple[str, ...]:
    bounded = _strict_json(
        source_bundle / "validation" / "bounded-grid.json",
        "Phase 6 bounded grid",
    )
    run_ids = bounded.get("run_ids")
    embedded = bounded.get("embedded_run_ids")
    if (
        not isinstance(run_ids, list)
        or any(not isinstance(item, str) for item in run_ids)
        or not isinstance(embedded, list)
        or embedded != run_ids[1:]
    ):
        raise OuterBundleError("Phase 6 admission run references are invalid")
    return tuple(run_ids)


def _original_reference(
    *,
    original: ValidatedArtifact,
    source_relative: str,
    copy_prefix: PurePosixPath,
    original_root_sha256: str,
    original_uri: str,
) -> dict[str, object]:
    by_path = original.by_path()
    return {
        "schema_version": ROOT_REFERENCE_SCHEMA,
        "root_sha256": original_root_sha256,
        "uri": original_uri,
        "object_count": len(original.files),
        "source_relative_path": source_relative,
        "copy_prefix": copy_prefix.as_posix(),
        "complete_content_copied": True,
        "manifest_sha256": by_path["manifest.json"].sha256,
        "inventory_sha256": by_path["artifact_inventory.json"].sha256,
        "checksum_ledger_sha256": by_path["checksums.sha256"].sha256,
        "complete_sha256": by_path["COMPLETE"].sha256,
    }


def _run_references(
    *,
    run_ids: Sequence[str],
    source_bundle: Path,
    copy_prefix: PurePosixPath,
) -> dict[str, object]:
    bounded_relative = "validation/bounded-grid.json"
    references: list[dict[str, object]] = []
    for index, run_id in enumerate(run_ids):
        embedded_prefix = (
            copy_prefix
            if index == 0
            else copy_prefix / "grid-runs" / run_id
        )
        source_manifest = (
            source_bundle / "manifest.json"
            if index == 0
            else source_bundle / "grid-runs" / run_id / "manifest.json"
        )
        source_complete = (
            source_bundle / "COMPLETE"
            if index == 0
            else source_bundle / "grid-runs" / run_id / "COMPLETE"
        )
        if not source_manifest.is_file() or not source_complete.is_file():
            raise OuterBundleError(
                "Phase 6 admission run reference lacks immutable controls"
            )
        references.append(
            {
                "index": index,
                "run_id": run_id,
                "role": "bundle_root" if index == 0 else "embedded_grid_run",
                "bundle_prefix": embedded_prefix.as_posix(),
                "manifest_sha256": sha256_file(source_manifest),
                "complete_sha256": sha256_file(source_complete),
            }
        )
    return {
        "schema_version": RUN_REFERENCES_SCHEMA,
        "run_count": len(references),
        "run_ids": list(run_ids),
        "bounded_grid_path": (copy_prefix / bounded_relative).as_posix(),
        "bounded_grid_sha256": sha256_file(
            source_bundle / bounded_relative
        ),
        "references": references,
    }


def _role(relative: str) -> str:
    if relative == "manifest.json":
        return "manifest"
    if relative.startswith(f"{ORIGINAL_COPY_PREFIX.as_posix()}/"):
        return "original_phase6_bundle"
    if relative == ORIGINAL_REFERENCE_PATH.as_posix():
        return "original_root_reference"
    if relative == ADMISSION_REFERENCES_PATH.as_posix():
        return "admission_run_references"
    if relative.startswith("docs/evidence/phase6/"):
        return "phase6_governance_evidence"
    if relative.startswith("docs/phase_reports/"):
        return "phase6_report"
    return "phase6_outer_bundle"


def _payload_paths(stage: Path, excluded: set[str]) -> list[Path]:
    return [
        path
        for path in sorted(
            stage.rglob("*"),
            key=lambda item: item.relative_to(stage).as_posix(),
        )
        if path.is_file()
        and path.relative_to(stage).as_posix() not in excluded
    ]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finalize_stage(
    *,
    stage: Path,
    final: Path,
    manifest: Mapping[str, object],
) -> None:
    write_exclusive(stage / "manifest.json", json_bytes(dict(manifest)))
    inventory_items = []
    for path in _payload_paths(
        stage,
        {"artifact_inventory.json", "checksums.sha256", "COMPLETE"},
    ):
        relative = path.relative_to(stage).as_posix()
        inventory_items.append(
            {
                "path": relative,
                "role": _role(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_exclusive(
        stage / "artifact_inventory.json",
        json_bytes(
            {
                "schema_version": INVENTORY_SCHEMA,
                "run_id": manifest["run_id"],
                "files": inventory_items,
                "excluded_control_files": [
                    "artifact_inventory.json",
                    "checksums.sha256",
                    "COMPLETE",
                ],
            }
        ),
    )
    ledger_paths = _payload_paths(stage, {"checksums.sha256", "COMPLETE"})
    ledger = "".join(
        f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n"
        for path in ledger_paths
    ).encode("utf-8")
    write_exclusive(stage / "checksums.sha256", ledger)
    write_exclusive(
        stage / "COMPLETE",
        json_bytes(
            {
                "schema_version": COMPLETION_SCHEMA,
                "run_id": manifest["run_id"],
                "status": manifest["status"],
                "manifest_sha256": sha256_file(stage / "manifest.json"),
                "artifact_inventory_sha256": sha256_file(
                    stage / "artifact_inventory.json"
                ),
                "checksum_ledger_path": "checksums.sha256",
                "checksum_ledger_sha256": sha256_file(
                    stage / "checksums.sha256"
                ),
                "written_last": True,
            }
        ),
    )
    for path in sorted(stage.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    stage.chmod(0o555)
    _fsync_directory(stage)
    validate_local_artifact(stage)
    rename_noreplace(stage, final)
    _fsync_directory(final.parent)


def _expected_required_paths(
    copy_prefix: PurePosixPath,
) -> tuple[str, ...]:
    return (
        ORIGINAL_REFERENCE_PATH.as_posix(),
        ADMISSION_REFERENCES_PATH.as_posix(),
        *(path.as_posix() for path in REQUIRED_REPOSITORY_FILES),
        (copy_prefix / "manifest.json").as_posix(),
        (copy_prefix / "artifact_inventory.json").as_posix(),
        (copy_prefix / "checksums.sha256").as_posix(),
        (copy_prefix / "COMPLETE").as_posix(),
    )


def validate_outer_bundle(
    artifact_root: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    source_bundle: Path | None = None,
    expected_original_root: str = ORIGINAL_ROOT_SHA256,
    expected_run_ids: Sequence[str] = ADMISSION_RUN_IDS,
    copy_prefix: PurePosixPath = ORIGINAL_COPY_PREFIX,
) -> OuterBundleValidation:
    """Validate exact source copies, controls, and Phase 6 references."""

    repository = repository_root.resolve(strict=True)
    source = (
        repository / ORIGINAL_BUNDLE_RELATIVE
        if source_bundle is None
        else source_bundle
    ).resolve(strict=True)
    original = validate_local_artifact(source)
    if original.root_sha256 != expected_original_root:
        raise OuterBundleError("original Phase 6 root digest differs")
    artifact = validate_local_artifact(artifact_root)
    manifest = _strict_json(artifact_root / "manifest.json", "outer manifest")
    run_id = manifest.get("run_id")
    required_paths = _expected_required_paths(copy_prefix)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or not isinstance(run_id, str)
        or manifest.get("status") != "completed"
        or _GIT_SHA_RE.fullmatch(str(manifest.get("source_git_sha"))) is None
        or manifest.get("original_root_sha256") != expected_original_root
        or manifest.get("original_r2_uri")
        != (
            ORIGINAL_R2_URI
            if expected_original_root == ORIGINAL_ROOT_SHA256
            else f"r2://test/sha256/{expected_original_root}/"
        )
        or manifest.get("original_object_count") != len(original.files)
        or manifest.get("admission_run_ids") != list(expected_run_ids)
        or manifest.get("required_paths") != list(required_paths)
        or manifest.get("performance_claim_eligible") is not False
        or manifest.get("quality_benchmark_executed") is not False
        or manifest.get("phase7_started") is not False
        or manifest.get("phase8_started") is not False
    ):
        raise OuterBundleError("outer manifest is invalid")

    outer_by_path = artifact.by_path()
    original_by_path = original.by_path()
    copied_paths = {
        relative[len(copy_prefix.as_posix()) + 1 :]: record
        for relative, record in outer_by_path.items()
        if relative.startswith(f"{copy_prefix.as_posix()}/")
    }
    if set(copied_paths) != set(original_by_path):
        raise OuterBundleError(
            "outer bundle does not contain the complete original bundle"
        )
    for relative, source_record in original_by_path.items():
        copied = copied_paths[relative]
        if (
            copied.size_bytes != source_record.size_bytes
            or copied.sha256 != source_record.sha256
        ):
            raise OuterBundleError("copied original bundle bytes differ")

    for relative in REQUIRED_REPOSITORY_FILES:
        bundled = outer_by_path.get(relative.as_posix())
        source_path = repository / relative
        if (
            bundled is None
            or not source_path.is_file()
            or bundled.size_bytes != source_path.stat().st_size
            or bundled.sha256 != sha256_file(source_path)
        ):
            raise OuterBundleError(
                f"required repository evidence differs: {relative.as_posix()}"
            )
    if OUTER_RECEIPT_RELATIVE.as_posix() in outer_by_path:
        raise OuterBundleError("outer publication receipt is self-included")

    source_relative = _repository_relative(source, repository)
    original_reference = _strict_json(
        artifact_root / ORIGINAL_REFERENCE_PATH,
        "original root reference",
    )
    expected_reference = _original_reference(
        original=original,
        source_relative=source_relative,
        copy_prefix=copy_prefix,
        original_root_sha256=expected_original_root,
        original_uri=(
            ORIGINAL_R2_URI
            if expected_original_root == ORIGINAL_ROOT_SHA256
            else f"r2://test/sha256/{expected_original_root}/"
        ),
    )
    if original_reference != expected_reference:
        raise OuterBundleError("original root reference differs")

    observed_run_ids = _load_admission_run_ids(source)
    if observed_run_ids != tuple(expected_run_ids):
        raise OuterBundleError("Phase 6 admission run list differs")
    run_references = _strict_json(
        artifact_root / ADMISSION_REFERENCES_PATH,
        "admission run references",
    )
    expected_references = _run_references(
        run_ids=expected_run_ids,
        source_bundle=source,
        copy_prefix=copy_prefix,
    )
    if run_references != expected_references:
        raise OuterBundleError("admission run references differ")
    if any(relative not in outer_by_path for relative in required_paths):
        raise OuterBundleError("outer bundle lacks a required path")

    return OuterBundleValidation(
        run_id=run_id,
        root_sha256=artifact.root_sha256,
        object_count=len(artifact.files),
        original_object_count=len(original.files),
        admission_run_count=len(expected_run_ids),
        required_paths=required_paths,
    )


def build_outer_bundle(
    *,
    repository_root: Path,
    source_bundle: Path,
    output_root: Path,
    run_id: str,
    source_git_sha: str,
    expected_original_root: str = ORIGINAL_ROOT_SHA256,
    expected_run_ids: Sequence[str] = ADMISSION_RUN_IDS,
    original_uri: str = ORIGINAL_R2_URI,
    copy_prefix: PurePosixPath = ORIGINAL_COPY_PREFIX,
) -> tuple[Path, OuterBundleValidation]:
    """Build one immutable no-replace outer bundle."""

    validate_run_id(run_id)
    if _GIT_SHA_RE.fullmatch(source_git_sha) is None:
        raise OuterBundleError("source Git SHA is invalid")
    repository = repository_root.resolve(strict=True)
    source = source_bundle.resolve(strict=True)
    original = validate_local_artifact(source)
    if original.root_sha256 != expected_original_root:
        raise OuterBundleError("original Phase 6 root digest differs")
    run_ids = _load_admission_run_ids(source)
    if run_ids != tuple(expected_run_ids):
        raise OuterBundleError("Phase 6 admission run list differs")
    source_relative = _repository_relative(source, repository)
    repository_files: dict[Path, bytes] = {}
    for relative in REQUIRED_REPOSITORY_FILES:
        path = repository / relative
        if not path.is_file() or path.is_symlink():
            raise OuterBundleError(
                f"required repository evidence is absent: {relative.as_posix()}"
            )
        repository_files[relative] = path.read_bytes()

    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise OuterBundleError("outer artifact root is unsafe")
    final = output_root / run_id
    if final.exists() or final.is_symlink():
        raise OuterBundleError("outer bundle run ID already exists")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{run_id}.",
            suffix=".staging",
            dir=output_root,
        )
    )
    original_files = _regular_source_files(source)
    for relative, path in original_files.items():
        write_exclusive(
            stage / copy_prefix / PurePosixPath(relative),
            path.read_bytes(),
        )
    for relative, data in repository_files.items():
        write_exclusive(stage / relative, data)
    write_exclusive(
        stage / ORIGINAL_REFERENCE_PATH,
        json_bytes(
            _original_reference(
                original=original,
                source_relative=source_relative,
                copy_prefix=copy_prefix,
                original_root_sha256=expected_original_root,
                original_uri=original_uri,
            )
        ),
    )
    write_exclusive(
        stage / ADMISSION_REFERENCES_PATH,
        json_bytes(
            _run_references(
                run_ids=run_ids,
                source_bundle=source,
                copy_prefix=copy_prefix,
            )
        ),
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": run_id,
        "status": "completed",
        "scope": (
            "Append-only Phase 6 R2 outer admission bundle; no experiment "
            "execution and no Phase 7 or Phase 8 work."
        ),
        "source_git_sha": source_git_sha,
        "original_root_sha256": expected_original_root,
        "original_r2_uri": original_uri,
        "original_object_count": len(original.files),
        "admission_run_ids": list(run_ids),
        "required_paths": list(_expected_required_paths(copy_prefix)),
        "performance_claim_eligible": False,
        "quality_benchmark_executed": False,
        "phase7_started": False,
        "phase8_started": False,
    }
    _finalize_stage(stage=stage, final=final, manifest=manifest)
    validation = validate_outer_bundle(
        final,
        repository_root=repository,
        source_bundle=source,
        expected_original_root=expected_original_root,
        expected_run_ids=expected_run_ids,
        copy_prefix=copy_prefix,
    )
    return final, validation


def _clean_git_sha(repository_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if status.stdout:
        raise OuterBundleError("source tree must be clean before bundle build")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if _GIT_SHA_RE.fullmatch(revision) is None:
        raise OuterBundleError("source Git SHA is invalid")
    return revision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    build = commands.add_parser("build")
    build.add_argument("--run-id", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("artifact", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.operation == "build":
            revision = _clean_git_sha(REPOSITORY_ROOT)
            final, validation = build_outer_bundle(
                repository_root=REPOSITORY_ROOT,
                source_bundle=REPOSITORY_ROOT / ORIGINAL_BUNDLE_RELATIVE,
                output_root=REPOSITORY_ROOT / OUTER_ARTIFACT_ROOT_RELATIVE,
                run_id=arguments.run_id,
                source_git_sha=revision,
            )
            payload = {
                **validation.to_dict(),
                "artifact_path": str(final),
            }
        else:
            validation = validate_outer_bundle(
                arguments.artifact.resolve(strict=True),
            )
            payload = validation.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (
        OSError,
        OuterBundleError,
        RuntimeError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build and validate the append-only Phase 8 R2 outer admission bundle."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
from typing import Any

from kvbench.runtime.artifacts import (
    validate_run_directory,
    validate_run_id,
)
from kvbench.runtime.kivi_admission import (
    build_phase8_method_admission_report,
    require_exact_phase8_grid,
)
from kvbench.schema.phase3 import GateDisposition
from kvbench.schema.phase8 import (
    Phase8MethodAdmissionReport,
    Phase8RunManifest,
)
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
OUTER_ARTIFACT_ROOT_RELATIVE = Path("artifacts") / "phase8_r2_outer"
INNER_RECEIPT_RELATIVE = (
    Path("docs")
    / "evidence"
    / "phase8"
    / "r2-admission-publication.json"
)
OUTER_RECEIPT_RELATIVE = (
    Path("docs")
    / "evidence"
    / "phase8"
    / "r2-admission-outer-publication.json"
)
METHOD_ADMISSION_RELATIVE = (
    Path("docs")
    / "evidence"
    / "phase8"
    / "kivi-method-admission.json"
)
METHOD_ADMISSION_CHECKSUM_RELATIVE = (
    Path("docs")
    / "evidence"
    / "phase8"
    / "kivi-method-admission.sha256"
)
PASS_REPORT_RELATIVE = (
    Path("docs")
    / "phase_reports"
    / "phase8-kivi-measurement-adapter.md"
)
REQUIRED_REPOSITORY_FILES = (
    INNER_RECEIPT_RELATIVE,
    METHOD_ADMISSION_RELATIVE,
    METHOD_ADMISSION_CHECKSUM_RELATIVE,
    PASS_REPORT_RELATIVE,
)
INNER_REFERENCE_PATH = PurePosixPath(
    "references", "original-phase8-root.json"
)
ADMISSION_REFERENCES_PATH = PurePosixPath(
    "references", "phase8-admission-runs.json"
)
MANIFEST_SCHEMA = "kvbench-phase8-r2-outer-bundle-1.0.0"
ROOT_REFERENCE_SCHEMA = "kvbench-phase8-r2-inner-root-reference-1.0.0"
RUN_REFERENCES_SCHEMA = (
    "kvbench-phase8-kivi-admission-run-references-1.0.0"
)
INNER_RECEIPT_SCHEMA = (
    "kvbench-phase8-kivi-admission-r2-publication-1.0.0"
)
OUTER_RECEIPT_SCHEMA = (
    "kvbench-phase8-kivi-admission-r2-outer-publication-1.0.0"
)
INVENTORY_SCHEMA = "kvbench-artifact-inventory-1.0.0"
COMPLETION_SCHEMA = "kvbench-completion-1.0.0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_UTC_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z"
)
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_PROHIBITED_CREDENTIAL_KEYS = frozenset(
    {
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "cloudflare_api_token",
        "hf_token",
        "hugging_face_hub_token",
        "r2_account_id",
        "authorization",
    }
)
_FORBIDDEN_INNER_REPORT_VALIDATION_KEYS = (
    "retrieved_" + "report_valid",
    "method_admission_" + "report_valid",
)


class Phase8OuterBundleError(RuntimeError):
    """The Phase 8 outer bundle failed a narrow build or validation rule."""


@dataclass(frozen=True)
class Phase8OuterBundleValidation:
    run_id: str
    root_sha256: str
    object_count: int
    inner_root_sha256: str
    inner_object_count: int
    admission_run_count: int
    method_admission_sha256: str
    required_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": (
                "kvbench-phase8-r2-outer-bundle-validation-1.0.0"
            ),
            "status": "PASS",
            "run_id": self.run_id,
            "root_sha256": self.root_sha256,
            "object_count": self.object_count,
            "inner_root_sha256": self.inner_root_sha256,
            "inner_object_count": self.inner_object_count,
            "admission_run_count": self.admission_run_count,
            "method_admission_sha256": self.method_admission_sha256,
            "required_paths": list(self.required_paths),
            "receipt_in_bundle": False,
            "performance_claim_eligible": False,
            "speedup_calculated": False,
            "r_hbm": None,
            "quality_execution": "LOCKED",
            "full_scan": "CLOSED",
            "phase9_started": False,
        }


@dataclass(frozen=True)
class Phase8OuterPublicationValidation:
    """Strict external receipt binding for the report-bearing outer bundle."""

    run_id: str
    root_sha256: str
    object_count: int
    r2_uri: str
    receipt_path: str
    receipt_sha256: str
    bucket_lock_identity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": (
                "kvbench-phase8-r2-outer-publication-validation-1.0.0"
            ),
            "status": "PASS",
            "run_id": self.run_id,
            "root_sha256": self.root_sha256,
            "object_count": self.object_count,
            "r2_uri": self.r2_uri,
            "receipt_path": self.receipt_path,
            "receipt_sha256": self.receipt_sha256,
            "bucket_lock_identity": self.bucket_lock_identity,
            "receipt_in_bundle": False,
            "complete_uploaded_last": True,
            "destination_initially_empty": True,
            "clean_retrieval": True,
            "performance_claim_eligible": False,
            "speedup_calculated": False,
            "r_hbm": None,
            "quality_execution": "LOCKED",
            "full_scan": "CLOSED",
            "phase9_started": False,
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
        raise Phase8OuterBundleError(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise Phase8OuterBundleError(f"{label} is invalid")
    return payload


def _reject_credential_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if (
                isinstance(key, str)
                and key.lower() in _PROHIBITED_CREDENTIAL_KEYS
            ):
                raise Phase8OuterBundleError(
                    "publication evidence contains a credential field"
                )
            if key in _FORBIDDEN_INNER_REPORT_VALIDATION_KEYS:
                raise Phase8OuterBundleError(
                    "inner publication evidence cannot validate the "
                    "MethodAdmissionReport"
                )
            _reject_credential_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_credential_keys(nested)


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(
            repository_root.resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError) as error:
        raise Phase8OuterBundleError(
            "Phase 8 inner bundle must be inside the repository"
        ) from error


def _regular_source_files(root: Path) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise Phase8OuterBundleError("source bundle is absent or unsafe")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase8OuterBundleError("source bundle contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase8OuterBundleError(
                "source bundle contains an unsafe entry"
            )
        files[relative] = path
    return files


def _inner_uri(root_sha256: str) -> str:
    return (
        "r2://kvbench-artifacts/kvbench/sha256/"
        f"{root_sha256}/"
    )


def _load_admission_manifests(
    source_bundle: Path,
) -> tuple[tuple[str, ...], tuple[Phase8RunManifest, ...]]:
    bounded = _strict_json(
        source_bundle / "validation" / "bounded-grid.json",
        "Phase 8 bounded grid",
    )
    run_ids = bounded.get("run_ids")
    embedded = bounded.get("embedded_run_ids")
    if (
        bounded.get("schema_version")
        != "kvbench-phase8-kivi-bounded-grid-1.0.0"
        or not isinstance(run_ids, list)
        or len(run_ids) != 10
        or len(set(run_ids)) != 10
        or any(not isinstance(item, str) for item in run_ids)
        or not isinstance(embedded, list)
        or embedded != run_ids[1:]
        or bounded.get("attempted") != 10
        or bounded.get("passed") != 10
        or bounded.get("failed") != 0
        or bounded.get("speedup_calculated") is not False
        or bounded.get("performance_claim_eligible") is not False
    ):
        raise Phase8OuterBundleError(
            "Phase 8 admission run references are invalid"
        )
    candidate = _strict_json(
        source_bundle / "validation" / "admission-candidate.json",
        "Phase 8 admission candidate",
    )
    if (
        candidate.get("status")
        != "LOCAL_CHECKS_PASS_PUBLICATION_PENDING"
        or candidate.get("g2_kivi") != "NOT_EVALUATED"
        or candidate.get("durable_publication") != "pending_host_side"
        or candidate.get("clean_retrieval") != "pending_host_side"
        or candidate.get("performance_claim_eligible") is not False
        or candidate.get("speedup_calculated") is not False
        or candidate.get("r_hbm") is not None
    ):
        raise Phase8OuterBundleError(
            "Phase 8 inner admission candidate differs"
        )

    manifests: list[Phase8RunManifest] = []
    for index, run_id in enumerate(run_ids):
        run_root = (
            source_bundle
            if index == 0
            else source_bundle / "grid-runs" / run_id
        )
        validation = validate_run_directory(
            run_root,
            expect_final_name=True,
        )
        if not validation.valid or not validation.complete:
            raise Phase8OuterBundleError(
                "Phase 8 admission run lacks valid immutable controls"
            )
        payload = _strict_json(
            run_root / "manifest.json",
            "Phase 8 admission run manifest",
        )
        manifest = Phase8RunManifest.from_dict(payload)
        if (
            manifest.run_id != run_id
            or manifest.status.value != "completed"
        ):
            raise Phase8OuterBundleError(
                "Phase 8 admission run identity differs"
            )
        manifests.append(manifest)
    try:
        require_exact_phase8_grid(manifests)
    except (RuntimeError, TypeError, ValueError) as error:
        raise Phase8OuterBundleError(
            "Phase 8 bounded grid is not exact"
        ) from error
    return tuple(run_ids), tuple(manifests)


def _require_mapping(
    value: object,
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase8OuterBundleError(f"{label} is invalid")
    return value


def _validate_inner_receipt(
    receipt: Mapping[str, Any],
    *,
    inner: ValidatedArtifact,
    source_run_id: str,
    execution_git_sha: str,
) -> None:
    _reject_credential_keys(receipt)
    local = _require_mapping(
        receipt.get("local_validation"),
        label="inner receipt local validation",
    )
    publication = _require_mapping(
        receipt.get("publication"),
        label="inner receipt publication",
    )
    retrieval = _require_mapping(
        receipt.get("clean_retrieval"),
        label="inner receipt clean retrieval",
    )
    lock = _require_mapping(
        receipt.get("bucket_lock"),
        label="inner receipt Bucket Lock",
    )
    lock_rule_name = lock.get("lock_rule_name")
    expected_uri = _inner_uri(inner.root_sha256)
    if (
        receipt.get("schema_version") != INNER_RECEIPT_SCHEMA
        or receipt.get("admission_status") != "PASS"
        or receipt.get("artifact_status") != "completed"
        or receipt.get("source_git_sha") != execution_git_sha
        or receipt.get("source_run_id") != source_run_id
        or receipt.get("credential_values_recorded") is not False
        or receipt.get("env_file_read") is not False
        or local.get("valid") is not True
        or local.get("complete") is not True
        or local.get("status") != "completed"
        or local.get("root_sha256") != inner.root_sha256
        or local.get("object_count") != len(inner.files)
        or local.get("complete_marker_valid") is not True
        or local.get("inventory_valid") is not True
        or local.get("checksum_ledger_valid") is not True
        or local.get("root_digest_valid") is not True
        or local.get("bundle_validation_valid") is not True
        or publication.get("result") != "PASS"
        or publication.get("root_sha256") != inner.root_sha256
        or publication.get("uri") != expected_uri
        or publication.get("object_count") != len(inner.files)
        or publication.get("content_addressed") is not True
        or publication.get("conditional_writes") is not True
        or publication.get("complete_last") is not True
        or retrieval.get("result") != "PASS"
        or retrieval.get("root_sha256") != inner.root_sha256
        or retrieval.get("object_count") != len(inner.files)
        or retrieval.get("destination_initially_empty") is not True
        or retrieval.get("complete_marker_valid") is not True
        or retrieval.get("inventory_valid") is not True
        or retrieval.get("checksum_ledger_valid") is not True
        or retrieval.get("root_digest_valid") is not True
        or retrieval.get("bundle_validation_valid") is not True
        or retrieval.get("unexpected_objects") is not False
        or lock.get("provider") != "cloudflare_r2"
        or lock.get("bucket") != "kvbench-artifacts"
        or lock.get("endpoint")
        != "https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com"
        or lock.get("endpoint_class") != "cloudflare_r2_s3"
        or lock.get("bucket_exists") is not True
        or lock.get("verification_result") != "PASS"
        or lock.get("enabled") is not True
        or lock.get("public_state_result") != "PASS"
        or lock.get("managed_r2_dev_enabled") is not False
        or lock.get("public_r2_dev") is not False
        or type(lock.get("custom_domain_count")) is not int
        or lock.get("custom_domain_count") < 0
        or type(lock.get("enabled_custom_domain_count")) is not int
        or lock.get("enabled_custom_domain_count") != 0
        or lock.get("public_custom_domain") is not False
        or not isinstance(lock.get("lock_rule_id"), str)
        or not lock.get("lock_rule_id")
        or lock.get("lock_rule_id") != lock.get("lock_rule_id").strip()
        or "lock_rule_name" not in lock
        or (
            lock_rule_name is not None
            and (
                not isinstance(lock_rule_name, str)
                or not lock_rule_name.strip()
            )
        )
        or lock.get("lock_scope") != "exact"
        or lock.get("covered_prefix") != "kvbench/sha256/"
        or lock.get("lock_prefix") != "kvbench/sha256/"
        or lock.get("retention_type") != "Indefinite"
        or lock.get("retention_condition") != "Indefinite"
        or lock.get("bucket_public") is not False
        or not isinstance(lock.get("verified_at_utc"), str)
        or _UTC_TIMESTAMP_RE.fullmatch(lock["verified_at_utc"]) is None
    ):
        raise Phase8OuterBundleError(
            "Phase 8 inner publication receipt does not bind the bundle"
        )


def _method_checksum_bytes(report_path: Path) -> bytes:
    return (
        f"{sha256_file(report_path)}  {METHOD_ADMISSION_RELATIVE.name}\n"
    ).encode("utf-8")


def _validate_report_join(
    *,
    repository_root: Path,
    source_bundle: Path,
    inner: ValidatedArtifact,
    manifests: Sequence[Phase8RunManifest],
) -> Phase8MethodAdmissionReport:
    report_path = repository_root / METHOD_ADMISSION_RELATIVE
    checksum_path = repository_root / METHOD_ADMISSION_CHECKSUM_RELATIVE
    receipt_path = repository_root / INNER_RECEIPT_RELATIVE
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or not checksum_path.is_file()
        or checksum_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
    ):
        raise Phase8OuterBundleError(
            "Phase 8 governance evidence is absent or unsafe"
        )
    try:
        expected_checksum = _method_checksum_bytes(report_path)
        observed_checksum = checksum_path.read_bytes()
    except OSError as error:
        raise Phase8OuterBundleError(
            "Phase 8 method report checksum is unreadable"
        ) from error
    if observed_checksum != expected_checksum:
        raise Phase8OuterBundleError(
            "Phase 8 method report checksum differs"
        )
    try:
        report = Phase8MethodAdmissionReport.from_dict(
            _strict_json(report_path, "Phase 8 MethodAdmissionReport")
        )
    except (TypeError, ValueError) as error:
        raise Phase8OuterBundleError(
            "Phase 8 MethodAdmissionReport is invalid"
        ) from error
    execution_shas = {manifest.git_sha for manifest in manifests}
    if (
        len(execution_shas) != 1
        or report.status is not GateDisposition.PASS
        or report.gates.g2_kivi is not GateDisposition.PASS
        or report.local_root_digest != inner.root_sha256
        or report.r2_uri != _inner_uri(inner.root_sha256)
        or report.creation_git_sha != next(iter(execution_shas))
        or report.clean_retrieval is not True
    ):
        raise Phase8OuterBundleError(
            "Phase 8 method report does not join the inner root"
        )
    receipt = _strict_json(
        receipt_path,
        "Phase 8 inner publication receipt",
    )
    _validate_inner_receipt(
        receipt,
        inner=inner,
        source_run_id=source_bundle.name,
        execution_git_sha=next(iter(execution_shas)),
    )
    source_relative = _repository_relative(
        source_bundle, repository_root
    )
    source_prefix = f"{source_relative}/"
    by_id = {
        reference.evidence_id: reference
        for reference in report.evidence_references
    }
    for reference in report.evidence_references:
        if reference.path == INNER_RECEIPT_RELATIVE.as_posix():
            evidence_path = receipt_path
        elif reference.path.startswith(source_prefix):
            inner_relative = reference.path[len(source_prefix) :]
            if not inner_relative:
                raise Phase8OuterBundleError(
                    "Phase 8 evidence reference is not a file"
                )
            evidence_path = source_bundle.joinpath(
                *PurePosixPath(inner_relative).parts
            )
        else:
            raise Phase8OuterBundleError(
                "Phase 8 report references evidence outside the inner closure"
            )
        if (
            not evidence_path.is_file()
            or evidence_path.is_symlink()
            or sha256_file(evidence_path) != reference.sha256
        ):
            raise Phase8OuterBundleError(
                "Phase 8 report evidence reference differs"
            )
    for check_id in ("durable_publication", "clean_retrieval"):
        checks = [check for check in report.checks if check.check_id == check_id]
        if (
            len(checks) != 1
            or not any(
                by_id[evidence_id].path
                == INNER_RECEIPT_RELATIVE.as_posix()
                for evidence_id in checks[0].evidence_ids
            )
        ):
            raise Phase8OuterBundleError(
                "Phase 8 durable checks do not cite the inner receipt"
            )
    rebuilt = build_phase8_method_admission_report(
        created_at_utc=report.created_at_utc,
        creation_git_sha=next(iter(execution_shas)),
        evidence_root=repository_root,
        inner_bundle_root=source_bundle,
        publication_receipt_path=receipt_path,
    )
    if rebuilt.to_dict() != report.to_dict():
        raise Phase8OuterBundleError(
            "Phase 8 method report was not produced by strict derivation"
        )
    return report


def _inner_reference(
    *,
    inner: ValidatedArtifact,
    source_relative: str,
    copy_prefix: PurePosixPath,
) -> dict[str, object]:
    by_path = inner.by_path()
    return {
        "schema_version": ROOT_REFERENCE_SCHEMA,
        "root_sha256": inner.root_sha256,
        "uri": _inner_uri(inner.root_sha256),
        "object_count": len(inner.files),
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
        source_run = (
            source_bundle
            if index == 0
            else source_bundle / "grid-runs" / run_id
        )
        references.append(
            {
                "index": index,
                "run_id": run_id,
                "role": "bundle_root" if index == 0 else "embedded_grid_run",
                "bundle_prefix": embedded_prefix.as_posix(),
                "manifest_sha256": sha256_file(
                    source_run / "manifest.json"
                ),
                "complete_sha256": sha256_file(source_run / "COMPLETE"),
            }
        )
    return {
        "schema_version": RUN_REFERENCES_SCHEMA,
        "run_count": len(references),
        "run_ids": list(run_ids),
        "bounded_grid_path": (
            copy_prefix / bounded_relative
        ).as_posix(),
        "bounded_grid_sha256": sha256_file(
            source_bundle / bounded_relative
        ),
        "references": references,
    }


def _role(relative: str, copy_prefix: PurePosixPath) -> str:
    if relative == "manifest.json":
        return "manifest"
    if relative.startswith(f"{copy_prefix.as_posix()}/"):
        return "phase8_inner_bundle"
    if relative == INNER_REFERENCE_PATH.as_posix():
        return "inner_root_reference"
    if relative == ADMISSION_REFERENCES_PATH.as_posix():
        return "admission_run_references"
    if relative == METHOD_ADMISSION_RELATIVE.as_posix():
        return "method_admission_report"
    if relative == METHOD_ADMISSION_CHECKSUM_RELATIVE.as_posix():
        return "method_admission_checksum"
    if relative == INNER_RECEIPT_RELATIVE.as_posix():
        return "inner_r2_publication_receipt"
    if relative == PASS_REPORT_RELATIVE.as_posix():
        return "phase8_report"
    return "phase8_outer_bundle"


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
    copy_prefix: PurePosixPath,
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
                "role": _role(relative, copy_prefix),
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
        INNER_REFERENCE_PATH.as_posix(),
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
    source_bundle: Path,
) -> Phase8OuterBundleValidation:
    """Validate exact source copies, report joins, and Phase 8 references."""

    if source_bundle.is_symlink():
        raise Phase8OuterBundleError(
            "Phase 8 inner bundle path is a symlink"
        )
    repository = repository_root.resolve(strict=True)
    source = source_bundle.resolve(strict=True)
    source_relative = _repository_relative(source, repository)
    inner = validate_local_artifact(source)
    if OUTER_RECEIPT_RELATIVE.as_posix() in inner.by_path():
        raise Phase8OuterBundleError(
            "Phase 8 inner bundle contains the outer publication receipt"
        )
    run_ids, manifests = _load_admission_manifests(source)
    report = _validate_report_join(
        repository_root=repository,
        source_bundle=source,
        inner=inner,
        manifests=manifests,
    )
    copy_prefix = PurePosixPath(
        "original", "sha256", inner.root_sha256
    )
    artifact = validate_local_artifact(artifact_root)
    manifest = _strict_json(
        artifact_root / "manifest.json", "Phase 8 outer manifest"
    )
    run_id = manifest.get("run_id")
    required_paths = _expected_required_paths(copy_prefix)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or not isinstance(run_id, str)
        or manifest.get("status") != "completed"
        or _GIT_SHA_RE.fullmatch(
            str(manifest.get("source_git_sha"))
        )
        is None
        or manifest.get("inner_root_sha256") != inner.root_sha256
        or manifest.get("inner_r2_uri") != _inner_uri(inner.root_sha256)
        or manifest.get("inner_object_count") != len(inner.files)
        or manifest.get("admission_run_ids") != list(run_ids)
        or manifest.get("method_admission_sha256")
        != sha256_file(repository / METHOD_ADMISSION_RELATIVE)
        or manifest.get("required_paths") != list(required_paths)
        or manifest.get("performance_claim_eligible") is not False
        or manifest.get("speedup_calculated") is not False
        or manifest.get("r_hbm") is not None
        or manifest.get("quality_execution") != "LOCKED"
        or manifest.get("quality_benchmark_executed") is not False
        or manifest.get("performance_data_frozen") is not False
        or manifest.get("full_scan") != "CLOSED"
        or manifest.get("global_g2") != "NOT_EVALUATED"
        or manifest.get("phase9_started") is not False
    ):
        raise Phase8OuterBundleError("Phase 8 outer manifest is invalid")

    outer_by_path = artifact.by_path()
    inner_by_path = inner.by_path()
    copied_paths = {
        relative[len(copy_prefix.as_posix()) + 1 :]: record
        for relative, record in outer_by_path.items()
        if relative.startswith(f"{copy_prefix.as_posix()}/")
    }
    if set(copied_paths) != set(inner_by_path):
        raise Phase8OuterBundleError(
            "outer bundle does not contain the complete Phase 8 inner bundle"
        )
    for relative, source_record in inner_by_path.items():
        copied = copied_paths[relative]
        if (
            copied.size_bytes != source_record.size_bytes
            or copied.sha256 != source_record.sha256
        ):
            raise Phase8OuterBundleError(
                "copied Phase 8 inner bundle bytes differ"
            )

    for relative in REQUIRED_REPOSITORY_FILES:
        bundled = outer_by_path.get(relative.as_posix())
        source_path = repository / relative
        if (
            bundled is None
            or not source_path.is_file()
            or source_path.is_symlink()
            or source_path.stat().st_nlink != 1
            or bundled.size_bytes != source_path.stat().st_size
            or bundled.sha256 != sha256_file(source_path)
        ):
            raise Phase8OuterBundleError(
                f"required repository evidence differs: {relative.as_posix()}"
            )
    if OUTER_RECEIPT_RELATIVE.as_posix() in outer_by_path:
        raise Phase8OuterBundleError(
            "Phase 8 outer publication receipt is self-included"
        )

    expected_inner_reference = _inner_reference(
        inner=inner,
        source_relative=source_relative,
        copy_prefix=copy_prefix,
    )
    observed_inner_reference = _strict_json(
        artifact_root / INNER_REFERENCE_PATH,
        "Phase 8 inner root reference",
    )
    if observed_inner_reference != expected_inner_reference:
        raise Phase8OuterBundleError(
            "Phase 8 inner root reference differs"
        )
    expected_run_references = _run_references(
        run_ids=run_ids,
        source_bundle=source,
        copy_prefix=copy_prefix,
    )
    observed_run_references = _strict_json(
        artifact_root / ADMISSION_REFERENCES_PATH,
        "Phase 8 admission run references",
    )
    if observed_run_references != expected_run_references:
        raise Phase8OuterBundleError(
            "Phase 8 admission run references differ"
        )
    if any(relative not in outer_by_path for relative in required_paths):
        raise Phase8OuterBundleError(
            "Phase 8 outer bundle lacks a required path"
        )

    source_prefix = f"{source_relative}/"
    for reference in report.evidence_references:
        if reference.path == INNER_RECEIPT_RELATIVE.as_posix():
            bundled_path = reference.path
        else:
            if not reference.path.startswith(source_prefix):
                raise Phase8OuterBundleError(
                    "Phase 8 report evidence escaped the outer closure"
                )
            bundled_path = (
                copy_prefix
                / PurePosixPath(reference.path[len(source_prefix) :])
            ).as_posix()
        bundled = outer_by_path.get(bundled_path)
        if bundled is None or bundled.sha256 != reference.sha256:
            raise Phase8OuterBundleError(
                "Phase 8 report evidence is not exact in the outer bundle"
            )

    return Phase8OuterBundleValidation(
        run_id=run_id,
        root_sha256=artifact.root_sha256,
        object_count=len(artifact.files),
        inner_root_sha256=inner.root_sha256,
        inner_object_count=len(inner.files),
        admission_run_count=len(run_ids),
        method_admission_sha256=sha256_file(
            repository / METHOD_ADMISSION_RELATIVE
        ),
        required_paths=required_paths,
    )


def validate_outer_publication_receipt(
    artifact_root: Path,
    *,
    receipt_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
    source_bundle: Path,
) -> Phase8OuterPublicationValidation:
    """Bind the external receipt to one exact, validated outer bundle."""

    repository = repository_root.resolve(strict=True)
    artifact = artifact_root.resolve(strict=True)
    expected_receipt = (repository / OUTER_RECEIPT_RELATIVE).resolve(
        strict=True
    )
    receipt = receipt_path.resolve(strict=True)
    if (
        receipt != expected_receipt
        or receipt_path.is_symlink()
        or not receipt.is_file()
        or receipt.stat().st_nlink != 1
    ):
        raise Phase8OuterBundleError(
            "Phase 8 outer publication receipt is absent or unsafe"
        )
    try:
        receipt.relative_to(artifact)
    except ValueError:
        pass
    else:
        raise Phase8OuterBundleError(
            "Phase 8 outer publication receipt is self-included"
        )

    validation = validate_outer_bundle(
        artifact,
        repository_root=repository,
        source_bundle=source_bundle,
    )
    validated_artifact = validate_local_artifact(artifact)
    manifest = _strict_json(
        artifact / "manifest.json",
        "Phase 8 outer manifest",
    )
    payload = _strict_json(
        receipt,
        "Phase 8 outer publication receipt",
    )
    self_reference = _require_mapping(
        payload.get("self_reference_control"),
        label="outer receipt self-reference control",
    )
    if (
        payload.get("schema_version") != OUTER_RECEIPT_SCHEMA
        or payload.get("performance_claim_eligible") is not False
        or payload.get("speedup_calculated") is not False
        or payload.get("r_hbm") is not None
        or payload.get("quality_execution") != "LOCKED"
        or payload.get("quality_benchmark_executed") is not False
        or payload.get("performance_data_frozen") is not False
        or payload.get("full_scan") != "CLOSED"
        or payload.get("global_g2") != "NOT_EVALUATED"
        or payload.get("phase9_started") is not False
        or self_reference.get("included_in_bundle") is not False
        or self_reference.get("receipt_path")
        != OUTER_RECEIPT_RELATIVE.as_posix()
    ):
        raise Phase8OuterBundleError(
            "Phase 8 outer publication receipt governance differs"
        )
    normalized = dict(payload)
    normalized["schema_version"] = INNER_RECEIPT_SCHEMA
    _validate_inner_receipt(
        normalized,
        inner=validated_artifact,
        source_run_id=validation.run_id,
        execution_git_sha=str(manifest["source_git_sha"]),
    )
    lock = _require_mapping(
        payload.get("bucket_lock"),
        label="outer receipt Bucket Lock",
    )
    return Phase8OuterPublicationValidation(
        run_id=validation.run_id,
        root_sha256=validation.root_sha256,
        object_count=validation.object_count,
        r2_uri=_inner_uri(validation.root_sha256),
        receipt_path=OUTER_RECEIPT_RELATIVE.as_posix(),
        receipt_sha256=sha256_file(receipt),
        bucket_lock_identity=str(lock["lock_rule_id"]),
    )


def build_outer_bundle(
    *,
    repository_root: Path,
    source_bundle: Path,
    output_root: Path,
    run_id: str,
    source_git_sha: str,
) -> tuple[Path, Phase8OuterBundleValidation]:
    """Build one immutable no-replace Phase 8 outer bundle."""

    validate_run_id(run_id)
    if _GIT_SHA_RE.fullmatch(source_git_sha) is None:
        raise Phase8OuterBundleError("source Git SHA is invalid")
    if source_bundle.is_symlink():
        raise Phase8OuterBundleError(
            "Phase 8 inner bundle path is a symlink"
        )
    repository = repository_root.resolve(strict=True)
    source = source_bundle.resolve(strict=True)
    source_relative = _repository_relative(source, repository)
    inner = validate_local_artifact(source)
    if OUTER_RECEIPT_RELATIVE.as_posix() in inner.by_path():
        raise Phase8OuterBundleError(
            "Phase 8 inner bundle contains the outer publication receipt"
        )
    run_ids, manifests = _load_admission_manifests(source)
    _validate_report_join(
        repository_root=repository,
        source_bundle=source,
        inner=inner,
        manifests=manifests,
    )
    repository_files: dict[Path, bytes] = {}
    for relative in REQUIRED_REPOSITORY_FILES:
        path = repository / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
        ):
            raise Phase8OuterBundleError(
                f"required repository evidence is unsafe: {relative.as_posix()}"
            )
        repository_files[relative] = path.read_bytes()

    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise Phase8OuterBundleError(
            "Phase 8 outer artifact root is unsafe"
        )
    final = output_root / run_id
    if final.exists() or final.is_symlink():
        raise Phase8OuterBundleError(
            "Phase 8 outer bundle run ID already exists"
        )
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{run_id}.",
            suffix=".staging",
            dir=output_root,
        )
    )
    copy_prefix = PurePosixPath(
        "original", "sha256", inner.root_sha256
    )
    for relative, path in _regular_source_files(source).items():
        write_exclusive(
            stage / copy_prefix / PurePosixPath(relative),
            path.read_bytes(),
        )
    for relative, data in repository_files.items():
        write_exclusive(stage / relative, data)
    write_exclusive(
        stage / INNER_REFERENCE_PATH,
        json_bytes(
            _inner_reference(
                inner=inner,
                source_relative=source_relative,
                copy_prefix=copy_prefix,
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
            "Append-only Phase 8 KIVI R2 outer admission bundle; no "
            "experiment execution, performance claim, quality execution, "
            "or Phase 9 work."
        ),
        "source_git_sha": source_git_sha,
        "inner_root_sha256": inner.root_sha256,
        "inner_r2_uri": _inner_uri(inner.root_sha256),
        "inner_object_count": len(inner.files),
        "admission_run_ids": list(run_ids),
        "method_admission_sha256": sha256_file(
            repository / METHOD_ADMISSION_RELATIVE
        ),
        "required_paths": list(_expected_required_paths(copy_prefix)),
        "performance_claim_eligible": False,
        "speedup_calculated": False,
        "r_hbm": None,
        "quality_execution": "LOCKED",
        "quality_benchmark_executed": False,
        "performance_data_frozen": False,
        "full_scan": "CLOSED",
        "global_g2": "NOT_EVALUATED",
        "phase9_started": False,
    }
    _finalize_stage(
        stage=stage,
        final=final,
        manifest=manifest,
        copy_prefix=copy_prefix,
    )
    validation = validate_outer_bundle(
        final,
        repository_root=repository,
        source_bundle=source,
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
        raise Phase8OuterBundleError(
            "source tree must be clean before outer bundle build"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if _GIT_SHA_RE.fullmatch(revision) is None:
        raise Phase8OuterBundleError("source Git SHA is invalid")
    return revision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-bundle", required=True, type=Path)
    build.add_argument("--run-id", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("artifact", type=Path)
    validate.add_argument("--source-bundle", required=True, type=Path)
    publication = commands.add_parser("validate-publication")
    publication.add_argument("artifact", type=Path)
    publication.add_argument("--source-bundle", required=True, type=Path)
    publication.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        source = arguments.source_bundle.resolve(strict=True)
        if arguments.operation == "build":
            revision = _clean_git_sha(REPOSITORY_ROOT)
            final, validation = build_outer_bundle(
                repository_root=REPOSITORY_ROOT,
                source_bundle=source,
                output_root=(
                    REPOSITORY_ROOT / OUTER_ARTIFACT_ROOT_RELATIVE
                ),
                run_id=arguments.run_id,
                source_git_sha=revision,
            )
            payload = {
                **validation.to_dict(),
                "artifact_path": str(final),
            }
        elif arguments.operation == "validate":
            validation = validate_outer_bundle(
                arguments.artifact.resolve(strict=True),
                repository_root=REPOSITORY_ROOT,
                source_bundle=source,
            )
            payload = validation.to_dict()
        else:
            publication = validate_outer_publication_receipt(
                arguments.artifact.resolve(strict=True),
                receipt_path=arguments.receipt,
                repository_root=REPOSITORY_ROOT,
                source_bundle=source,
            )
            payload = publication.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (
        OSError,
        Phase8OuterBundleError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
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

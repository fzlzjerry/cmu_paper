#!/usr/bin/env python3
"""Build and validate the self-reference-safe Phase 11 R2 outer bundle.

The CLI defaults to the immutable Decision 0027 authority.  The Decision 0029
successor is available only through its explicit, non-mixable authority
profile and separate repository evidence namespace.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
from typing import Any

from kvbench.runtime.artifacts import validate_run_directory, validate_run_id
from kvbench.schema import canonical_json_bytes
from kvbench.schema.phase11 import (
    PHASE11_AGGREGATE_PATCH_SHA256,
    PHASE11_CALIBRATION_ID,
    PHASE11_CALIBRATION_ROOT,
    PHASE11_CONFIGURATIONS,
    PHASE11_CORRECTED_COMMIT,
    PHASE11_CORRECTED_TREE,
    PHASE11_DECISIONS,
    PHASE11_EXECUTION_SOURCE_IDENTIFIER,
    PHASE11_EXTENSION_SHA256,
    PHASE11_FIXTURE_ID,
    PHASE11_FIXTURE_ROOT,
    PHASE11_HISTORICAL_FIXTURE_ROOT,
    PHASE11_METHOD_IDENTIFIER,
    PHASE11Q23_AGGREGATE_PATCH_SHA256,
    PHASE11Q23_CALIBRATION_ID,
    PHASE11Q23_CALIBRATION_ROOT,
    PHASE11Q23_CONFIGURATIONS,
    PHASE11Q23_CORRECTED_COMMIT,
    PHASE11Q23_CORRECTED_TREE,
    PHASE11Q23_DECISIONS,
    PHASE11Q23_EXECUTION_SOURCE_IDENTIFIER,
    PHASE11Q23_EXTENSION_SHA256,
    PHASE11Q23_FIXTURE_ID,
    PHASE11Q23_FIXTURE_ROOT,
    PHASE11Q23_HISTORICAL_FIXTURE_ROOT,
    PHASE11Q23_METHOD_IDENTIFIER,
    Phase11MethodAdmissionReport,
    Phase11RQ23MethodAdmissionReport,
    Phase11RQ23RunManifest,
    Phase11RunManifest,
    Phase11RunPoint,
    require_exact_phase11_grid,
)
from preflight.run_preflight import (
    json_bytes,
    rename_noreplace,
    sha256_file,
    write_exclusive,
)
from scripts.r2_artifact import (
    STATUS_VARIABLES,
    ValidatedArtifact,
    publication_order,
    validate_local_artifact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTER_ARTIFACT_ROOT_RELATIVE = Path("artifacts") / "phase11_r2_outer"
AUTHORITY_PROFILE_DECISION0027 = "decision0027"
AUTHORITY_PROFILE_DECISION0029 = "decision0029"
AUTHORITY_PROFILE_CHOICES = (
    AUTHORITY_PROFILE_DECISION0027,
    AUTHORITY_PROFILE_DECISION0029,
)
AUTHORITY_PROFILE = AUTHORITY_PROFILE_DECISION0027
INNER_RECEIPT_RELATIVE = (
    Path("docs")
    / "evidence"
    / "phase11"
    / "r2-admission-publication.json"
)
OUTER_RECEIPT_RELATIVE = (
    Path("docs")
    / "evidence"
    / "phase11"
    / "r2-admission-outer-publication.json"
)
INNER_PUBLISH_STDOUT_RELATIVE = Path(
    "docs/evidence/phase11/r2-admission-publish.stdout.json"
)
INNER_PUBLISH_STDERR_RELATIVE = Path(
    "docs/evidence/phase11/r2-admission-publish.stderr.txt"
)
INNER_VERIFY_STDOUT_RELATIVE = Path(
    "docs/evidence/phase11/r2-admission-verify.stdout.json"
)
INNER_VERIFY_STDERR_RELATIVE = Path(
    "docs/evidence/phase11/r2-admission-verify.stderr.txt"
)
OUTER_PUBLISH_STDOUT_RELATIVE = Path(
    "docs/evidence/phase11/r2-admission-outer-publish.stdout.json"
)
OUTER_PUBLISH_STDERR_RELATIVE = Path(
    "docs/evidence/phase11/r2-admission-outer-publish.stderr.txt"
)
OUTER_VERIFY_STDOUT_RELATIVE = Path(
    "docs/evidence/phase11/r2-admission-outer-verify.stdout.json"
)
OUTER_VERIFY_STDERR_RELATIVE = Path(
    "docs/evidence/phase11/r2-admission-outer-verify.stderr.txt"
)
METHOD_ADMISSION_RELATIVE = (
    Path("docs")
    / "evidence"
    / "phase11"
    / "kvquant-method-admission.json"
)
METHOD_ADMISSION_CHECKSUM_RELATIVE = (
    Path("docs")
    / "evidence"
    / "phase11"
    / "kvquant-method-admission.sha256"
)
PASS_REPORT_RELATIVE = (
    Path("docs")
    / "phase_reports"
    / "phase11r-kvquant-measurement-adapter.md"
)
REQUIRED_REPOSITORY_FILES = (
    INNER_RECEIPT_RELATIVE,
    INNER_PUBLISH_STDOUT_RELATIVE,
    INNER_PUBLISH_STDERR_RELATIVE,
    INNER_VERIFY_STDOUT_RELATIVE,
    INNER_VERIFY_STDERR_RELATIVE,
    METHOD_ADMISSION_RELATIVE,
    METHOD_ADMISSION_CHECKSUM_RELATIVE,
    PASS_REPORT_RELATIVE,
)
INNER_REFERENCE_PATH = PurePosixPath("source-inner", "binding.json")
ADMISSION_REFERENCES_PATH = INNER_REFERENCE_PATH
BUNDLED_INNER_RECEIPT_PATH = PurePosixPath(
    "admission", "inner-r2-publication.json"
)
BUNDLED_INNER_PUBLISH_STDOUT_PATH = PurePosixPath(
    "admission", "r2-publish.stdout.json"
)
BUNDLED_INNER_PUBLISH_STDERR_PATH = PurePosixPath(
    "admission", "r2-publish.stderr.txt"
)
BUNDLED_INNER_VERIFY_STDOUT_PATH = PurePosixPath(
    "admission", "r2-verify.stdout.json"
)
BUNDLED_INNER_VERIFY_STDERR_PATH = PurePosixPath(
    "admission", "r2-verify.stderr.txt"
)
BUNDLED_METHOD_ADMISSION_PATH = PurePosixPath(
    "admission", "method-admission.json"
)
BUNDLED_METHOD_CHECKSUM_PATH = PurePosixPath(
    "admission", "method-admission.sha256"
)
BUNDLED_PASS_REPORT_PATH = PurePosixPath(
    "reports", "phase11r-kvquant-measurement-adapter.md"
)
MANIFEST_SCHEMA = "kvbench-phase11-r2-outer-bundle-1.0.0"
OUTER_BUNDLE_VALIDATION_SCHEMA = (
    "kvbench-phase11-r2-outer-bundle-validation-1.0.0"
)
OUTER_PUBLICATION_VALIDATION_SCHEMA = (
    "kvbench-phase11-r2-outer-publication-validation-1.0.0"
)
INNER_RECEIPT_SCHEMA = (
    "kvbench-phase11-kvquant-admission-r2-publication-1.0.0"
)
OUTER_RECEIPT_SCHEMA = (
    "kvbench-phase11-kvquant-admission-r2-outer-publication-1.0.0"
)
BOUNDED_GRID_SCHEMA = "kvbench-phase11-bounded-grid-1.0.0"
POINT_SCHEMA = "kvbench-phase11-kvquant-point-1.0.0"
INVENTORY_SCHEMA = "kvbench-artifact-inventory-1.0.0"
COMPLETION_SCHEMA = "kvbench-completion-1.0.0"
PASS_REPORT_HEADING = "PHASE 11 REPORT"
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_UTC_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z"
)
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
_FORBIDDEN_CREDENTIAL_PATH_PARTS = frozenset(
    {".env", "credentials", "secrets"}
)
_FORBIDDEN_CLAIM_KEYS = frozenset(
    {
        "speedup",
        "comparative_latency",
        "performance_claim",
        "hbm_traffic",
        "capacity_improvement",
    }
)


@dataclass(frozen=True)
class _AuthorityProfile:
    name: str
    method_identifier: str
    execution_source_identifier: str
    aggregate_patch_sha256: str
    corrected_commit: str
    corrected_tree: str
    extension_sha256: str
    decisions: tuple[str, ...]
    calibration_id: str
    calibration_root: str
    historical_fixture_root: str
    fixture_id: str
    fixture_root: str
    configurations: tuple[str, ...]
    run_manifest_class: type[Any]
    method_admission_report_class: type[Any]
    evidence_namespace: str
    pass_report_name: str
    pass_report_heading: str
    manifest_schema: str
    outer_bundle_validation_schema: str
    inner_receipt_schema: str
    outer_receipt_schema: str
    outer_publication_validation_schema: str


_AUTHORITY_PROFILES = {
    AUTHORITY_PROFILE_DECISION0027: _AuthorityProfile(
        name=AUTHORITY_PROFILE_DECISION0027,
        method_identifier=PHASE11_METHOD_IDENTIFIER,
        execution_source_identifier=PHASE11_EXECUTION_SOURCE_IDENTIFIER,
        aggregate_patch_sha256=PHASE11_AGGREGATE_PATCH_SHA256,
        corrected_commit=PHASE11_CORRECTED_COMMIT,
        corrected_tree=PHASE11_CORRECTED_TREE,
        extension_sha256=PHASE11_EXTENSION_SHA256,
        decisions=PHASE11_DECISIONS,
        calibration_id=PHASE11_CALIBRATION_ID,
        calibration_root=PHASE11_CALIBRATION_ROOT,
        historical_fixture_root=PHASE11_HISTORICAL_FIXTURE_ROOT,
        fixture_id=PHASE11_FIXTURE_ID,
        fixture_root=PHASE11_FIXTURE_ROOT,
        configurations=PHASE11_CONFIGURATIONS,
        run_manifest_class=Phase11RunManifest,
        method_admission_report_class=Phase11MethodAdmissionReport,
        evidence_namespace="phase11",
        pass_report_name="phase11r-kvquant-measurement-adapter.md",
        pass_report_heading="PHASE 11 REPORT",
        manifest_schema="kvbench-phase11-r2-outer-bundle-1.0.0",
        outer_bundle_validation_schema=(
            "kvbench-phase11-r2-outer-bundle-validation-1.0.0"
        ),
        inner_receipt_schema=(
            "kvbench-phase11-kvquant-admission-r2-publication-1.0.0"
        ),
        outer_receipt_schema=(
            "kvbench-phase11-kvquant-admission-r2-outer-publication-1.0.0"
        ),
        outer_publication_validation_schema=(
            "kvbench-phase11-r2-outer-publication-validation-1.0.0"
        ),
    ),
    AUTHORITY_PROFILE_DECISION0029: _AuthorityProfile(
        name=AUTHORITY_PROFILE_DECISION0029,
        method_identifier=PHASE11Q23_METHOD_IDENTIFIER,
        execution_source_identifier=(
            PHASE11Q23_EXECUTION_SOURCE_IDENTIFIER
        ),
        aggregate_patch_sha256=PHASE11Q23_AGGREGATE_PATCH_SHA256,
        corrected_commit=PHASE11Q23_CORRECTED_COMMIT,
        corrected_tree=PHASE11Q23_CORRECTED_TREE,
        extension_sha256=PHASE11Q23_EXTENSION_SHA256,
        decisions=PHASE11Q23_DECISIONS,
        calibration_id=PHASE11Q23_CALIBRATION_ID,
        calibration_root=PHASE11Q23_CALIBRATION_ROOT,
        historical_fixture_root=PHASE11Q23_HISTORICAL_FIXTURE_ROOT,
        fixture_id=PHASE11Q23_FIXTURE_ID,
        fixture_root=PHASE11Q23_FIXTURE_ROOT,
        configurations=PHASE11Q23_CONFIGURATIONS,
        run_manifest_class=Phase11RQ23RunManifest,
        method_admission_report_class=Phase11RQ23MethodAdmissionReport,
        evidence_namespace="phase11rq23",
        pass_report_name=(
            "phase11rq23-kvquant-measurement-adapter.md"
        ),
        pass_report_heading="PHASE 11R-Q23 REPORT",
        manifest_schema=(
            "kvbench-phase11rq23-r2-outer-bundle-1.0.0"
        ),
        outer_bundle_validation_schema=(
            "kvbench-phase11rq23-r2-outer-bundle-validation-1.0.0"
        ),
        inner_receipt_schema=(
            "kvbench-phase11rq23-kvquant-admission-"
            "r2-publication-1.0.0"
        ),
        outer_receipt_schema=(
            "kvbench-phase11rq23-kvquant-admission-"
            "r2-outer-publication-1.0.0"
        ),
        outer_publication_validation_schema=(
            "kvbench-phase11rq23-r2-outer-publication-validation-1.0.0"
        ),
    ),
}


def _activate_profile(name: str) -> None:
    """Activate one exact, non-mixable Phase 11 authority profile."""

    try:
        profile = _AUTHORITY_PROFILES[name]
    except KeyError as error:
        raise Phase11OuterBundleError(
            "Phase 11 authority profile is invalid"
        ) from error

    evidence = Path("docs") / "evidence" / profile.evidence_namespace
    globals().update(
        {
            "AUTHORITY_PROFILE": profile.name,
            "PHASE11_METHOD_IDENTIFIER": profile.method_identifier,
            "PHASE11_EXECUTION_SOURCE_IDENTIFIER": (
                profile.execution_source_identifier
            ),
            "PHASE11_AGGREGATE_PATCH_SHA256": (
                profile.aggregate_patch_sha256
            ),
            "PHASE11_CORRECTED_COMMIT": profile.corrected_commit,
            "PHASE11_CORRECTED_TREE": profile.corrected_tree,
            "PHASE11_EXTENSION_SHA256": profile.extension_sha256,
            "PHASE11_DECISIONS": profile.decisions,
            "PHASE11_CALIBRATION_ID": profile.calibration_id,
            "PHASE11_CALIBRATION_ROOT": profile.calibration_root,
            "PHASE11_HISTORICAL_FIXTURE_ROOT": (
                profile.historical_fixture_root
            ),
            "PHASE11_FIXTURE_ID": profile.fixture_id,
            "PHASE11_FIXTURE_ROOT": profile.fixture_root,
            "PHASE11_CONFIGURATIONS": profile.configurations,
            "Phase11RunManifest": profile.run_manifest_class,
            "Phase11MethodAdmissionReport": (
                profile.method_admission_report_class
            ),
            "INNER_RECEIPT_RELATIVE": (
                evidence / "r2-admission-publication.json"
            ),
            "OUTER_RECEIPT_RELATIVE": (
                evidence / "r2-admission-outer-publication.json"
            ),
            "INNER_PUBLISH_STDOUT_RELATIVE": (
                evidence / "r2-admission-publish.stdout.json"
            ),
            "INNER_PUBLISH_STDERR_RELATIVE": (
                evidence / "r2-admission-publish.stderr.txt"
            ),
            "INNER_VERIFY_STDOUT_RELATIVE": (
                evidence / "r2-admission-verify.stdout.json"
            ),
            "INNER_VERIFY_STDERR_RELATIVE": (
                evidence / "r2-admission-verify.stderr.txt"
            ),
            "OUTER_PUBLISH_STDOUT_RELATIVE": (
                evidence / "r2-admission-outer-publish.stdout.json"
            ),
            "OUTER_PUBLISH_STDERR_RELATIVE": (
                evidence / "r2-admission-outer-publish.stderr.txt"
            ),
            "OUTER_VERIFY_STDOUT_RELATIVE": (
                evidence / "r2-admission-outer-verify.stdout.json"
            ),
            "OUTER_VERIFY_STDERR_RELATIVE": (
                evidence / "r2-admission-outer-verify.stderr.txt"
            ),
            "METHOD_ADMISSION_RELATIVE": (
                evidence / "kvquant-method-admission.json"
            ),
            "METHOD_ADMISSION_CHECKSUM_RELATIVE": (
                evidence / "kvquant-method-admission.sha256"
            ),
            "PASS_REPORT_RELATIVE": (
                Path("docs")
                / "phase_reports"
                / profile.pass_report_name
            ),
            "BUNDLED_PASS_REPORT_PATH": (
                PurePosixPath("reports") / profile.pass_report_name
            ),
            "MANIFEST_SCHEMA": profile.manifest_schema,
            "OUTER_BUNDLE_VALIDATION_SCHEMA": (
                profile.outer_bundle_validation_schema
            ),
            "INNER_RECEIPT_SCHEMA": profile.inner_receipt_schema,
            "OUTER_RECEIPT_SCHEMA": profile.outer_receipt_schema,
            "OUTER_PUBLICATION_VALIDATION_SCHEMA": (
                profile.outer_publication_validation_schema
            ),
            "PASS_REPORT_HEADING": profile.pass_report_heading,
        }
    )
    globals()["REQUIRED_REPOSITORY_FILES"] = (
        INNER_RECEIPT_RELATIVE,
        INNER_PUBLISH_STDOUT_RELATIVE,
        INNER_PUBLISH_STDERR_RELATIVE,
        INNER_VERIFY_STDOUT_RELATIVE,
        INNER_VERIFY_STDERR_RELATIVE,
        METHOD_ADMISSION_RELATIVE,
        METHOD_ADMISSION_CHECKSUM_RELATIVE,
        PASS_REPORT_RELATIVE,
    )


class Phase11OuterBundleError(RuntimeError):
    """The Phase 11 outer bundle failed a narrow validation rule."""


_activate_profile(AUTHORITY_PROFILE_DECISION0027)


@dataclass(frozen=True)
class Phase11OuterBundleValidation:
    run_id: str
    root_sha256: str
    object_count: int
    inner_root_sha256: str
    inner_object_count: int
    admission_run_count: int
    bounded_grid_sha256: str
    method_admission_sha256: str
    required_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OUTER_BUNDLE_VALIDATION_SCHEMA,
            "status": "PASS",
            "run_id": self.run_id,
            "root_sha256": self.root_sha256,
            "object_count": self.object_count,
            "inner_root_sha256": self.inner_root_sha256,
            "inner_object_count": self.inner_object_count,
            "admission_run_count": self.admission_run_count,
            "bounded_grid_sha256": self.bounded_grid_sha256,
            "method_admission_sha256": self.method_admission_sha256,
            "required_paths": list(self.required_paths),
            "receipt_in_bundle": False,
            "g2_kvq": "PASS",
            "global_g2": "NOT_EVALUATED",
            "performance_claim_eligible": False,
            "speedup_calculated": False,
            "r_hbm": None,
            "quality_execution": "LOCKED",
            "full_scan": "CLOSED",
            "phase12_started": False,
        }


@dataclass(frozen=True)
class Phase11OuterPublicationValidation:
    run_id: str
    root_sha256: str
    object_count: int
    r2_uri: str
    receipt_path: str
    receipt_sha256: str
    bucket_lock_identity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OUTER_PUBLICATION_VALIDATION_SCHEMA,
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
            "g2_kvq": "PASS",
            "global_g2": "NOT_EVALUATED",
            "performance_claim_eligible": False,
            "speedup_calculated": False,
            "r_hbm": None,
            "quality_execution": "LOCKED",
            "full_scan": "CLOSED",
            "phase12_started": False,
        }


@dataclass(frozen=True)
class _InnerClosure:
    artifact: ValidatedArtifact
    manifest: Phase11RunManifest
    points: tuple[Phase11RunPoint, ...]
    point_payloads: tuple[dict[str, Any], ...]
    point_paths: tuple[str, ...]
    bounded_grid_sha256: str
    method_fingerprints: dict[str, str]
    cache_layout_fingerprints: dict[str, str]


def _strict_json_value(path: Path, label: str) -> object:
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
        raise Phase11OuterBundleError(f"{label} is invalid") from error
    return payload


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    payload = _strict_json_value(path, label)
    if not isinstance(payload, dict):
        raise Phase11OuterBundleError(f"{label} is invalid")
    return payload


def _reject_governance_drift(
    value: object,
    *,
    allow_quality_execution_false: bool = False,
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold()
            if normalized in _PROHIBITED_CREDENTIAL_KEYS:
                raise Phase11OuterBundleError(
                    "Phase 11 evidence contains a credential field"
                )
            if key in _FORBIDDEN_INNER_REPORT_VALIDATION_KEYS:
                raise Phase11OuterBundleError(
                    "inner publication evidence cannot validate the "
                    "MethodAdmissionReport"
                )
            if normalized in _FORBIDDEN_CLAIM_KEYS:
                raise Phase11OuterBundleError(
                    "Phase 11 evidence contains a forbidden claim field"
                )
            if (
                normalized == "performance_claim_eligible"
                and nested is not False
            ):
                raise Phase11OuterBundleError(
                    "Phase 11 performance eligibility drifted"
                )
            if normalized == "speedup_calculated" and nested is not False:
                raise Phase11OuterBundleError(
                    "Phase 11 speedup state drifted"
                )
            if normalized == "r_hbm" and nested is not None:
                raise Phase11OuterBundleError(
                    "Phase 11 r_hbm must remain null"
                )
            if normalized == "performance_data_frozen" and nested is not False:
                raise Phase11OuterBundleError(
                    "Phase 11 performance freeze state drifted"
                )
            if normalized == "global_g2" and nested != "NOT_EVALUATED":
                raise Phase11OuterBundleError(
                    "Phase 11 Global G2 state drifted"
                )
            if (
                normalized in {"global_g2_g5", "global_g2_g5_state"}
                and nested != "NOT_EVALUATED"
            ):
                raise Phase11OuterBundleError(
                    "Phase 11 global gate state drifted"
                )
            if (
                normalized == "quality_execution"
                and not (
                    allow_quality_execution_false
                    and nested is False
                )
                and str(nested).upper() != "LOCKED"
            ):
                raise Phase11OuterBundleError(
                    "Phase 11 quality execution state drifted"
                )
            if (
                normalized in {"full_scan", "full_scan_state"}
                and str(nested).upper() != "CLOSED"
            ):
                raise Phase11OuterBundleError(
                    "Phase 11 Full Scan state drifted"
                )
            if normalized == "phase12_started" and nested is not False:
                raise Phase11OuterBundleError(
                    "Phase 12 must remain unstarted"
                )
            _reject_governance_drift(
                nested,
                allow_quality_execution_false=(
                    allow_quality_execution_false
                ),
            )
    elif isinstance(value, list):
        for nested in value:
            _reject_governance_drift(
                nested,
                allow_quality_execution_false=(
                    allow_quality_execution_false
                ),
            )


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(
            repository_root.resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError) as error:
        raise Phase11OuterBundleError(
            "Phase 11 inner bundle must be inside the repository"
        ) from error


def _safe_inner_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        not value
        or relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Phase11OuterBundleError(
            "Phase 11 evidence reference is not a safe inner path"
        )
    return relative


def _regular_source_files(root: Path) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise Phase11OuterBundleError("source bundle is absent or unsafe")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase11OuterBundleError(
                "source bundle contains a symlink"
            )
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase11OuterBundleError(
                "source bundle contains an unsafe entry"
            )
        parts = {part.casefold() for part in PurePosixPath(relative).parts}
        if parts & _FORBIDDEN_CREDENTIAL_PATH_PARTS:
            raise Phase11OuterBundleError(
                "source bundle contains a credential path"
            )
        files[relative] = path
    return files


def _r2_uri(root_sha256: str) -> str:
    return (
        "r2://kvbench-artifacts/kvbench/sha256/"
        f"{root_sha256}/"
    )


def _require_sha_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(PHASE11_CONFIGURATIONS)
        or any(
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in value.values()
        )
    ):
        raise Phase11OuterBundleError(f"{label} differs")
    return {str(key): str(digest) for key, digest in value.items()}


def _validate_point_payload(
    *,
    point: Phase11RunPoint,
    payload: dict[str, Any],
    path: Path,
) -> None:
    expected_keys = {
        "schema_version",
        "run_id",
        "configuration",
        "runner_kind",
        "graph_mode",
        "batch_size",
        "context_length",
        "output_steps",
        "quality_status",
        "claim_eligibility",
        "performance_claim_eligible",
        "measurement_scope",
        "speedup_calculated",
        "runner",
        "allocation_audits",
    }
    _reject_governance_drift(payload)
    canonical = canonical_json_bytes(payload)
    digest = hashlib.sha256(canonical).hexdigest()
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != POINT_SCHEMA
        or payload.get("run_id") != point.run_id
        or payload.get("configuration") != point.configuration
        or payload.get("runner_kind") != point.runner_kind.value
        or payload.get("graph_mode") != point.graph_mode.value
        or payload.get("batch_size") != point.batch_size
        or payload.get("context_length") != point.context_length
        or payload.get("output_steps") != point.output_steps
        or payload.get("quality_status") != "unvalidated"
        or payload.get("claim_eligibility") != "performance_only"
        or payload.get("performance_claim_eligible") is not False
        or payload.get("measurement_scope")
        != "measurement_container_admission"
        or payload.get("speedup_calculated") is not False
        or not isinstance(payload.get("runner"), Mapping)
        or not isinstance(payload.get("allocation_audits"), list)
        or digest != point.manifest_sha256
    ):
        raise Phase11OuterBundleError(
            "Phase 11 bounded point payload differs"
        )


def _validate_inner_bundle(source: Path) -> _InnerClosure:
    validation = validate_run_directory(source, expect_final_name=True)
    if not validation.valid or not validation.complete:
        raise Phase11OuterBundleError(
            "Phase 11 inner bundle lifecycle is invalid"
        )
    artifact = validate_local_artifact(source)
    try:
        manifest = Phase11RunManifest.from_dict(
            _strict_json(source / "manifest.json", "Phase 11 manifest")
        )
    except (TypeError, ValueError) as error:
        raise Phase11OuterBundleError(
            "Phase 11 inner manifest is invalid"
        ) from error
    if (
        manifest.status.value != "completed"
        or manifest.bounded_point_count != 9
        or manifest.g2_kvq_state
        != "NOT_EVALUATED_PUBLICATION_PENDING"
        or manifest.global_g2_g5_state != "NOT_EVALUATED"
        or manifest.full_scan_state != "CLOSED"
        or manifest.performance_claim_eligible
        or manifest.speedup_calculated
        or manifest.r_hbm is not None
    ):
        raise Phase11OuterBundleError(
            "Phase 11 inner manifest governance differs"
        )
    grid_path = source / "validation" / "bounded-grid.json"
    grid = _strict_json(grid_path, "Phase 11 bounded grid")
    _reject_governance_drift(grid)
    expected_grid_keys = {
        "schema_version",
        "points",
        "point_records",
        "attempted",
        "passed",
        "failed",
        "capacity_infeasible",
        "method_fingerprints",
        "cache_layout_fingerprints",
        "quality_status",
        "performance_claim_eligible",
        "measurement_scope",
        "speedup_calculated",
    }
    raw_points = grid.get("points")
    point_records = grid.get("point_records")
    if (
        set(grid) != expected_grid_keys
        or grid.get("schema_version") != BOUNDED_GRID_SCHEMA
        or not isinstance(raw_points, list)
        or not isinstance(point_records, list)
        or len(raw_points) != 9
        or len(point_records) != 9
        or grid.get("attempted") != 9
        or grid.get("passed") != 9
        or grid.get("failed") != 0
        or grid.get("capacity_infeasible") != 0
        or grid.get("quality_status") != "unvalidated"
        or grid.get("performance_claim_eligible") is not False
        or grid.get("measurement_scope")
        != "measurement_container_admission"
        or grid.get("speedup_calculated") is not False
    ):
        raise Phase11OuterBundleError(
            "Phase 11 bounded grid structure differs"
        )
    try:
        points = require_exact_phase11_grid(
            tuple(Phase11RunPoint.from_dict(item) for item in raw_points)
        )
    except (TypeError, ValueError) as error:
        raise Phase11OuterBundleError(
            "Phase 11 bounded run set differs"
        ) from error
    if (
        len({point.run_id for point in points}) != 9
        or len({point.manifest_sha256 for point in points}) != 9
    ):
        raise Phase11OuterBundleError(
            "Phase 11 bounded run identities are duplicated"
        )
    payloads: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, (point, record) in enumerate(zip(points, point_records)):
        relative = f"grid/{index:02d}-{point.run_id}/point.json"
        expected_record = {
            "index": index,
            "run_id": point.run_id,
            "path": relative,
            "sha256": point.manifest_sha256,
        }
        if not isinstance(record, dict) or record != expected_record:
            raise Phase11OuterBundleError(
                "Phase 11 bounded point record is invalid"
            )
        path = source / relative
        payload = _strict_json(path, "Phase 11 bounded point")
        _validate_point_payload(point=point, payload=payload, path=path)
        payloads.append(payload)
        paths.append(relative)
    source_json_paths = [
        source / record.relative_path
        for record in artifact.files
        if record.relative_path.endswith(".json")
    ]
    for path in source_json_paths:
        relative = path.relative_to(source).as_posix()
        _reject_governance_drift(
            _strict_json_value(path, "Phase 11 inner JSON evidence"),
            allow_quality_execution_false=(
                AUTHORITY_PROFILE == AUTHORITY_PROFILE_DECISION0029
                and relative == "authority/q23-evidence/summary.json"
            ),
        )
    return _InnerClosure(
        artifact=artifact,
        manifest=manifest,
        points=points,
        point_payloads=tuple(payloads),
        point_paths=tuple(paths),
        bounded_grid_sha256=sha256_file(grid_path),
        method_fingerprints=_require_sha_mapping(
            grid.get("method_fingerprints"),
            label="Phase 11 method fingerprints",
        ),
        cache_layout_fingerprints=_require_sha_mapping(
            grid.get("cache_layout_fingerprints"),
            label="Phase 11 cache layout fingerprints",
        ),
    )


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase11OuterBundleError(f"{label} is invalid")
    return value


def _validate_bucket_lock(lock: Mapping[str, Any]) -> str:
    expected_keys = {
        "provider",
        "bucket",
        "endpoint",
        "endpoint_class",
        "bucket_exists",
        "bucket_public",
        "public_state_result",
        "managed_r2_dev_enabled",
        "public_r2_dev",
        "custom_domain_count",
        "enabled_custom_domain_count",
        "public_custom_domain",
        "lock_rule_id",
        "lock_rule_name",
        "covered_prefix",
        "lock_prefix",
        "lock_scope",
        "enabled",
        "retention_type",
        "retention_condition",
        "verification_result",
        "verified_at_utc",
    }
    lock_rule_name = lock.get("lock_rule_name")
    lock_rule_id = lock.get("lock_rule_id")
    if (
        set(lock) != expected_keys
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
        or not isinstance(lock_rule_id, str)
        or not lock_rule_id.strip()
        or lock_rule_id != lock_rule_id.strip()
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
        or _UTC_TIMESTAMP_RE.fullmatch(str(lock["verified_at_utc"])) is None
    ):
        raise Phase11OuterBundleError(
            "Phase 11 publication Bucket Lock identity differs"
        )
    return lock_rule_id


def _stable_bucket_lock_identity(
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_bucket_lock(lock)
    return {
        key: value
        for key, value in lock.items()
        if key != "verified_at_utc"
    }


def _parse_utc(value: object, *, label: str) -> datetime:
    if (
        not isinstance(value, str)
        or _UTC_TIMESTAMP_RE.fullmatch(value) is None
    ):
        raise Phase11OuterBundleError(f"{label} is not a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Phase11OuterBundleError(
            f"{label} is not a UTC timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Phase11OuterBundleError(f"{label} is not a UTC timestamp")
    return parsed


def _publication_order_sha256(artifact: ValidatedArtifact) -> str:
    order = publication_order(artifact)
    payload = "".join(f"{item.relative_path}\n" for item in order).encode()
    return hashlib.sha256(payload).hexdigest()


def _raw_r2_paths(receipt_kind: str) -> tuple[Path, Path, Path, Path]:
    if receipt_kind == "inner":
        return (
            INNER_PUBLISH_STDOUT_RELATIVE,
            INNER_PUBLISH_STDERR_RELATIVE,
            INNER_VERIFY_STDOUT_RELATIVE,
            INNER_VERIFY_STDERR_RELATIVE,
        )
    if receipt_kind == "outer":
        return (
            OUTER_PUBLISH_STDOUT_RELATIVE,
            OUTER_PUBLISH_STDERR_RELATIVE,
            OUTER_VERIFY_STDOUT_RELATIVE,
            OUTER_VERIFY_STDERR_RELATIVE,
        )
    raise Phase11OuterBundleError("publication receipt kind is invalid")


def _validate_raw_r2_evidence(
    value: object,
    *,
    receipt_kind: str,
    repository_root: Path,
    artifact: ValidatedArtifact,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = _require_mapping(value, label="raw R2 tool evidence")
    if set(records) != {"publish", "verify"}:
        raise Phase11OuterBundleError("raw R2 tool evidence set differs")
    repository = repository_root.resolve(strict=True)
    artifact_relative = _repository_relative(artifact.directory, repository)
    expected_paths = _raw_r2_paths(receipt_kind)
    results: list[dict[str, Any]] = []
    for operation, stdout_relative, stderr_relative in (
        ("publish", expected_paths[0], expected_paths[1]),
        ("verify", expected_paths[2], expected_paths[3]),
    ):
        record = _require_mapping(
            records.get(operation),
            label=f"raw R2 {operation} evidence",
        )
        expected_argv = (
            [
                ".venv/bin/python",
                "scripts/r2_artifact.py",
                "publish",
                artifact_relative,
            ]
            if operation == "publish"
            else [
                ".venv/bin/python",
                "scripts/r2_artifact.py",
                "verify",
                artifact.root_sha256,
            ]
        )
        if (
            set(record)
            != {
                "command_argv",
                "working_directory",
                "returncode",
                "stdout_path",
                "stdout_sha256",
                "stderr_path",
                "stderr_sha256",
            }
            or record.get("command_argv") != expected_argv
            or record.get("working_directory") != "."
            or record.get("returncode") != 0
            or record.get("stdout_path") != stdout_relative.as_posix()
            or record.get("stderr_path") != stderr_relative.as_posix()
        ):
            raise Phase11OuterBundleError(
                f"raw R2 {operation} command evidence differs"
            )
        stdout = repository / stdout_relative
        stderr = repository / stderr_relative
        for path, expected_sha in (
            (stdout, record.get("stdout_sha256")),
            (stderr, record.get("stderr_sha256")),
        ):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or not isinstance(expected_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
                or sha256_file(path) != expected_sha
            ):
                raise Phase11OuterBundleError(
                    f"raw R2 {operation} output evidence differs"
                )
        if stderr.stat().st_size != 0:
            raise Phase11OuterBundleError(
                f"raw R2 {operation} stderr must be empty"
            )
        results.append(_require_tool_output(stdout, operation=operation))
    return results[0], results[1]


def _validate_publication_receipt_core(
    receipt: Mapping[str, Any],
    *,
    expected_schema: str,
    artifact: ValidatedArtifact,
    source_run_id: str,
    source_git_sha: str,
    repository_root: Path,
) -> str:
    _reject_governance_drift(receipt)
    core_keys = {
        "schema_version",
        "recorded_at_utc",
        "admission_status",
        "artifact_status",
        "source_git_sha",
        "source_run_id",
        "local_validation",
        "publication",
        "clean_retrieval",
        "bucket_lock",
        "raw_tool_evidence",
        "credential_values_recorded",
        "env_file_read",
    }
    outer_keys = {
        "g2_kvq",
        "global_g2",
        "performance_claim_eligible",
        "speedup_calculated",
        "r_hbm",
        "quality_execution",
        "quality_benchmark_executed",
        "performance_data_frozen",
        "full_scan",
        "phase12_started",
        "self_reference_control",
    }
    expected_keys = (
        core_keys | outer_keys
        if expected_schema == OUTER_RECEIPT_SCHEMA
        else core_keys
    )
    local = _require_mapping(
        receipt.get("local_validation"),
        label="publication local validation",
    )
    publication = _require_mapping(
        receipt.get("publication"),
        label="publication result",
    )
    retrieval = _require_mapping(
        receipt.get("clean_retrieval"),
        label="publication clean retrieval",
    )
    lock = _require_mapping(
        receipt.get("bucket_lock"),
        label="publication Bucket Lock",
    )
    expected_uri = _r2_uri(artifact.root_sha256)
    receipt_kind = (
        "outer" if expected_schema == OUTER_RECEIPT_SCHEMA else "inner"
    )
    raw_publish, raw_verify = _validate_raw_r2_evidence(
        receipt.get("raw_tool_evidence"),
        receipt_kind=receipt_kind,
        repository_root=repository_root,
        artifact=artifact,
    )
    raw_publication = _require_mapping(
        raw_publish.get("publish"),
        label="raw publisher result",
    )
    raw_retrieval = _require_mapping(
        raw_verify.get("verify"),
        label="raw verifier result",
    )
    raw_publish_lock = _require_mapping(
        raw_publish.get("bucket_lock"),
        label="raw publisher Bucket Lock",
    )
    raw_verify_lock = _require_mapping(
        raw_verify.get("bucket_lock"),
        label="raw verifier Bucket Lock",
    )
    receipt_lock_identity = _stable_bucket_lock_identity(lock)
    raw_verify_lock_identity = _stable_bucket_lock_identity(raw_verify_lock)
    published_at = _parse_utc(
        publication.get("published_at_utc"),
        label="publication timestamp",
    )
    retrieved_at = _parse_utc(
        retrieval.get("retrieved_at_utc"),
        label="retrieval timestamp",
    )
    recorded_at = _parse_utc(
        receipt.get("recorded_at_utc"),
        label="receipt timestamp",
    )
    expected_order_sha256 = _publication_order_sha256(artifact)
    if (
        set(receipt) != expected_keys
        or set(local)
        != {
            "valid",
            "complete",
            "status",
            "root_sha256",
            "object_count",
            "complete_marker_valid",
            "inventory_valid",
            "checksum_ledger_valid",
            "root_digest_valid",
            "bundle_validation_valid",
        }
        or set(publication)
        != {
            "result",
            "provider",
            "root_sha256",
            "uri",
            "object_count",
            "uploaded_count",
            "verified_existing_count",
            "content_addressed",
            "conditional_writes",
            "complete_last",
            "publication_order_sha256",
            "published_at_utc",
        }
        or set(retrieval)
        != {
            "result",
            "provider",
            "root_sha256",
            "uri",
            "object_count",
            "destination_initially_empty",
            "complete_marker_valid",
            "inventory_valid",
            "checksum_ledger_valid",
            "root_digest_valid",
            "bundle_validation_valid",
            "unexpected_objects",
            "retrieved_at_utc",
        }
        or receipt.get("schema_version") != expected_schema
        or receipt.get("admission_status") != "PASS"
        or receipt.get("artifact_status") != "completed"
        or receipt.get("source_git_sha") != source_git_sha
        or receipt.get("source_run_id") != source_run_id
        or receipt.get("credential_values_recorded") is not False
        or receipt.get("env_file_read") is not False
        or local.get("valid") is not True
        or local.get("complete") is not True
        or local.get("status") != "completed"
        or local.get("root_sha256") != artifact.root_sha256
        or local.get("object_count") != len(artifact.files)
        or local.get("complete_marker_valid") is not True
        or local.get("inventory_valid") is not True
        or local.get("checksum_ledger_valid") is not True
        or local.get("root_digest_valid") is not True
        or local.get("bundle_validation_valid") is not True
        or publication.get("result") != "PASS"
        or publication.get("provider") != "cloudflare_r2"
        or publication.get("root_sha256") != artifact.root_sha256
        or publication.get("uri") != expected_uri
        or publication.get("object_count") != len(artifact.files)
        or publication.get("content_addressed") is not True
        or publication.get("conditional_writes") is not True
        or publication.get("complete_last") is not True
        or publication.get("publication_order_sha256")
        != expected_order_sha256
        or type(publication.get("uploaded_count")) is not int
        or publication.get("uploaded_count") < 0
        or type(publication.get("verified_existing_count")) is not int
        or publication.get("verified_existing_count") < 0
        or (
            publication.get("uploaded_count")
            + publication.get("verified_existing_count")
            != len(artifact.files)
        )
        or retrieval.get("result") != "PASS"
        or retrieval.get("provider") != "cloudflare_r2"
        or retrieval.get("root_sha256") != artifact.root_sha256
        or retrieval.get("uri") != expected_uri
        or retrieval.get("object_count") != len(artifact.files)
        or retrieval.get("destination_initially_empty") is not True
        or retrieval.get("complete_marker_valid") is not True
        or retrieval.get("inventory_valid") is not True
        or retrieval.get("checksum_ledger_valid") is not True
        or retrieval.get("root_digest_valid") is not True
        or retrieval.get("bundle_validation_valid") is not True
        or retrieval.get("unexpected_objects") is not False
        or not (published_at <= retrieved_at <= recorded_at)
        or raw_publish_lock != lock
        or raw_verify_lock_identity != receipt_lock_identity
        or publication.get("provider") != raw_publication.get("provider")
        or publication.get("root_sha256")
        != raw_publication.get("root_sha256")
        or publication.get("uri") != raw_publication.get("uri")
        or publication.get("object_count")
        != raw_publication.get("object_count")
        or publication.get("uploaded_count")
        != raw_publication.get("uploaded_count")
        or publication.get("verified_existing_count")
        != raw_publication.get("verified_existing_count")
        or publication.get("complete_last")
        != raw_publication.get("complete_last")
        or publication.get("publication_order_sha256")
        != raw_publication.get("publication_order_sha256")
        or publication.get("published_at_utc")
        != raw_publication.get("published_at_utc")
        or retrieval.get("provider") != raw_retrieval.get("provider")
        or retrieval.get("root_sha256")
        != raw_retrieval.get("root_sha256")
        or retrieval.get("uri") != raw_retrieval.get("uri")
        or retrieval.get("object_count")
        != raw_retrieval.get("object_count")
        or retrieval.get("complete_marker_valid")
        != raw_retrieval.get("complete_marker_valid")
        or retrieval.get("inventory_valid")
        != raw_retrieval.get("inventory_valid")
        or retrieval.get("checksum_ledger_valid")
        != raw_retrieval.get("checksum_ledger_valid")
        or retrieval.get("unexpected_objects")
        != raw_retrieval.get("unexpected_objects")
        or retrieval.get("retrieved_at_utc")
        != raw_retrieval.get("retrieved_at_utc")
    ):
        raise Phase11OuterBundleError(
            "Phase 11 publication receipt does not bind the bundle"
        )
    return _validate_bucket_lock(lock)


def _require_tool_output(
    value: Mapping[str, Any] | Path,
    *,
    operation: str,
) -> dict[str, Any]:
    payload = (
        _strict_json(value, f"R2 {operation} output")
        if isinstance(value, Path)
        else dict(value)
    )
    result = payload.get(operation)
    lock = payload.get("bucket_lock")
    identity = payload.get("r2")
    variables = payload.get("required_variables")
    if (
        set(payload)
        != {
            "status",
            "required_variables",
            "r2",
            "bucket_lock",
            operation,
        }
        or payload.get("status") != "PASS"
        or not isinstance(result, Mapping)
        or not isinstance(lock, Mapping)
        or not isinstance(identity, Mapping)
        or not isinstance(variables, Mapping)
        or set(variables) != set(STATUS_VARIABLES)
        or any(
            status not in {"PRESENT", "MISSING"}
            for status in variables.values()
        )
        or identity
        != {
            "provider": "cloudflare_r2",
            "endpoint_class": "cloudflare_r2_s3",
            "endpoint": (
                "https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com"
            ),
            "bucket": "kvbench-artifacts",
            "prefix": "kvbench/sha256",
            "region": "auto",
        }
    ):
        raise Phase11OuterBundleError(
            f"R2 {operation} output is not an exact successful result"
        )
    return payload


def assemble_publication_receipt(
    *,
    artifact_root: Path,
    publish_output: Path,
    publish_stderr: Path,
    verify_output: Path,
    verify_stderr: Path,
    receipt_kind: str,
    source_run_id: str,
    source_git_sha: str,
    recorded_at_utc: str | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    """Assemble one strict receipt from the existing publisher/verifier output."""

    if receipt_kind not in {"inner", "outer"}:
        raise Phase11OuterBundleError("publication receipt kind is invalid")
    validate_run_id(source_run_id)
    if _GIT_SHA_RE.fullmatch(source_git_sha) is None:
        raise Phase11OuterBundleError("publication source Git SHA is invalid")
    if artifact_root.is_symlink():
        raise Phase11OuterBundleError("published artifact path is a symlink")
    repository = repository_root.resolve(strict=True)
    artifact_path = artifact_root.resolve(strict=True)
    artifact = validate_local_artifact(artifact_path)
    expected_raw_paths = _raw_r2_paths(receipt_kind)
    supplied_raw_paths = (
        publish_output,
        publish_stderr,
        verify_output,
        verify_stderr,
    )
    for supplied, relative in zip(
        supplied_raw_paths,
        expected_raw_paths,
        strict=True,
    ):
        if (
            supplied.is_symlink()
            or supplied.resolve(strict=True) != (repository / relative)
            or not supplied.is_file()
            or supplied.stat().st_nlink != 1
        ):
            raise Phase11OuterBundleError(
                "raw R2 tool-evidence path differs"
            )
    for operation, stderr in (
        ("publish", publish_stderr),
        ("verify", verify_stderr),
    ):
        if stderr.stat().st_size != 0:
            raise Phase11OuterBundleError(
                f"raw R2 {operation} stderr must be empty"
            )
    published = _require_tool_output(publish_output, operation="publish")
    verified = _require_tool_output(verify_output, operation="verify")
    publication = _require_mapping(
        published["publish"],
        label="publisher result",
    )
    retrieval = _require_mapping(
        verified["verify"],
        label="verifier result",
    )
    publish_lock = _require_mapping(
        published["bucket_lock"],
        label="publisher Bucket Lock",
    )
    verify_lock = _require_mapping(
        verified["bucket_lock"],
        label="verifier Bucket Lock",
    )
    publish_lock_identity = _stable_bucket_lock_identity(publish_lock)
    verify_lock_identity = _stable_bucket_lock_identity(verify_lock)
    if publish_lock_identity != verify_lock_identity:
        raise Phase11OuterBundleError(
            "publisher and verifier Bucket Lock identities differ"
        )
    expected_uri = _r2_uri(artifact.root_sha256)
    expected_order = _publication_order_sha256(artifact)
    if (
        set(publication)
        != {
            "provider",
            "root_sha256",
            "uri",
            "object_count",
            "uploaded_count",
            "verified_existing_count",
            "complete_last",
            "publication_order_sha256",
            "published_at_utc",
        }
        or set(retrieval)
        != {
            "provider",
            "root_sha256",
            "uri",
            "object_count",
            "complete_marker_valid",
            "inventory_valid",
            "checksum_ledger_valid",
            "unexpected_objects",
            "verification_result",
            "retrieved_at_utc",
        }
        or publication.get("provider") != "cloudflare_r2"
        or publication.get("root_sha256") != artifact.root_sha256
        or publication.get("uri") != expected_uri
        or publication.get("object_count") != len(artifact.files)
        or type(publication.get("uploaded_count")) is not int
        or type(publication.get("verified_existing_count")) is not int
        or publication.get("uploaded_count") < 0
        or publication.get("verified_existing_count") < 0
        or publication.get("uploaded_count")
        + publication.get("verified_existing_count")
        != len(artifact.files)
        or publication.get("complete_last") is not True
        or publication.get("publication_order_sha256") != expected_order
        or retrieval.get("provider") != "cloudflare_r2"
        or retrieval.get("root_sha256") != artifact.root_sha256
        or retrieval.get("uri") != expected_uri
        or retrieval.get("object_count") != len(artifact.files)
        or retrieval.get("complete_marker_valid") is not True
        or retrieval.get("inventory_valid") is not True
        or retrieval.get("checksum_ledger_valid") is not True
        or retrieval.get("unexpected_objects") is not False
        or retrieval.get("verification_result") != "PASS"
    ):
        raise Phase11OuterBundleError(
            "publisher or clean-retrieval output differs from the artifact"
        )
    timestamp = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if recorded_at_utc is None
        else recorded_at_utc
    )
    payload: dict[str, object] = {
        "schema_version": (
            INNER_RECEIPT_SCHEMA
            if receipt_kind == "inner"
            else OUTER_RECEIPT_SCHEMA
        ),
        "recorded_at_utc": timestamp,
        "admission_status": "PASS",
        "artifact_status": "completed",
        "source_git_sha": source_git_sha,
        "source_run_id": source_run_id,
        "local_validation": {
            "valid": True,
            "complete": True,
            "status": "completed",
            "root_sha256": artifact.root_sha256,
            "object_count": len(artifact.files),
            "complete_marker_valid": True,
            "inventory_valid": True,
            "checksum_ledger_valid": True,
            "root_digest_valid": True,
            "bundle_validation_valid": True,
        },
        "publication": {
            "result": "PASS",
            "provider": "cloudflare_r2",
            "root_sha256": artifact.root_sha256,
            "uri": expected_uri,
            "object_count": len(artifact.files),
            "uploaded_count": publication["uploaded_count"],
            "verified_existing_count": publication[
                "verified_existing_count"
            ],
            "content_addressed": True,
            "conditional_writes": True,
            "complete_last": True,
            "publication_order_sha256": expected_order,
            "published_at_utc": publication["published_at_utc"],
        },
        "clean_retrieval": {
            "result": "PASS",
            "provider": "cloudflare_r2",
            "root_sha256": artifact.root_sha256,
            "uri": expected_uri,
            "object_count": len(artifact.files),
            "destination_initially_empty": True,
            "complete_marker_valid": True,
            "inventory_valid": True,
            "checksum_ledger_valid": True,
            "root_digest_valid": True,
            "bundle_validation_valid": True,
            "unexpected_objects": False,
            "retrieved_at_utc": retrieval["retrieved_at_utc"],
        },
        "bucket_lock": dict(publish_lock),
        "raw_tool_evidence": {
            "publish": {
                "command_argv": [
                    ".venv/bin/python",
                    "scripts/r2_artifact.py",
                    "publish",
                    _repository_relative(artifact_path, repository),
                ],
                "working_directory": ".",
                "returncode": 0,
                "stdout_path": expected_raw_paths[0].as_posix(),
                "stdout_sha256": sha256_file(publish_output),
                "stderr_path": expected_raw_paths[1].as_posix(),
                "stderr_sha256": sha256_file(publish_stderr),
            },
            "verify": {
                "command_argv": [
                    ".venv/bin/python",
                    "scripts/r2_artifact.py",
                    "verify",
                    artifact.root_sha256,
                ],
                "working_directory": ".",
                "returncode": 0,
                "stdout_path": expected_raw_paths[2].as_posix(),
                "stdout_sha256": sha256_file(verify_output),
                "stderr_path": expected_raw_paths[3].as_posix(),
                "stderr_sha256": sha256_file(verify_stderr),
            },
        },
        "credential_values_recorded": False,
        "env_file_read": False,
    }
    if receipt_kind == "outer":
        payload.update(
            {
                "g2_kvq": "PASS",
                "global_g2": "NOT_EVALUATED",
                "performance_claim_eligible": False,
                "speedup_calculated": False,
                "r_hbm": None,
                "quality_execution": "LOCKED",
                "quality_benchmark_executed": False,
                "performance_data_frozen": False,
                "full_scan": "CLOSED",
                "phase12_started": False,
                "self_reference_control": {
                    "included_in_bundle": False,
                    "receipt_path": OUTER_RECEIPT_RELATIVE.as_posix(),
                },
            }
        )
    _validate_publication_receipt_core(
        payload,
        expected_schema=str(payload["schema_version"]),
        artifact=artifact,
        source_run_id=source_run_id,
        source_git_sha=source_git_sha,
        repository_root=repository,
    )
    return payload


def write_publication_receipt(
    *,
    output_path: Path,
    artifact_root: Path,
    publish_output: Path,
    publish_stderr: Path,
    verify_output: Path,
    verify_stderr: Path,
    receipt_kind: str,
    source_run_id: str,
    source_git_sha: str,
    recorded_at_utc: str | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Write a production receipt exactly once after strict assembly."""

    payload = assemble_publication_receipt(
        artifact_root=artifact_root,
        publish_output=publish_output,
        publish_stderr=publish_stderr,
        verify_output=verify_output,
        verify_stderr=verify_stderr,
        receipt_kind=receipt_kind,
        source_run_id=source_run_id,
        source_git_sha=source_git_sha,
        recorded_at_utc=recorded_at_utc,
        repository_root=repository_root,
    )
    write_exclusive(output_path, json_bytes(payload))
    return output_path


def validate_inner_publication_receipt(
    artifact_root: Path,
    *,
    receipt_path: Path,
    source_run_id: str,
    source_git_sha: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> str:
    """Validate one external inner receipt and return its lock identity."""

    if artifact_root.is_symlink() or receipt_path.is_symlink():
        raise Phase11OuterBundleError(
            "Phase 11 inner publication inputs are symlinks"
        )
    artifact_path = artifact_root.resolve(strict=True)
    receipt = receipt_path.resolve(strict=True)
    if not receipt.is_file() or receipt.stat().st_nlink != 1:
        raise Phase11OuterBundleError(
            "Phase 11 inner publication receipt is unsafe"
        )
    artifact = validate_local_artifact(artifact_path)
    payload = _strict_json(receipt, "Phase 11 inner publication receipt")
    return _validate_publication_receipt_core(
        payload,
        expected_schema=INNER_RECEIPT_SCHEMA,
        artifact=artifact,
        source_run_id=source_run_id,
        source_git_sha=source_git_sha,
        repository_root=repository_root,
    )


def _method_checksum_bytes(report_path: Path) -> bytes:
    return (
        f"{sha256_file(report_path)}  {METHOD_ADMISSION_RELATIVE.name}\n"
    ).encode("utf-8")


def _validate_phase_report(
    path: Path,
    *,
    report: Phase11MethodAdmissionReport,
    report_sha256: str,
) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise Phase11OuterBundleError(
            "Phase 11 final report is unreadable"
        ) from error
    def states(label: str) -> tuple[str, ...]:
        pattern = re.compile(
            rf"^\s*(?:-\s*)?{re.escape(label)}\s*:\s*"
            r"(?P<state>[^\r\n]+?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        return tuple(
            match.group("state").strip().rstrip(".").strip().upper()
            for match in pattern.finditer(text)
        )

    def exact_state(label: str, expected: str) -> bool:
        observed = states(label)
        return bool(observed) and all(item == expected for item in observed)

    required_states = {
        "Status": "PASS",
        "Working tree": "CLEAN",
        "Algorithm identifier": PHASE11_METHOD_IDENTIFIER,
        "Execution-source identifier": PHASE11_EXECUTION_SOURCE_IDENTIFIER,
        "Decisions": ", ".join(PHASE11_DECISIONS),
        "Aggregate patch SHA": PHASE11_AGGREGATE_PATCH_SHA256,
        "Corrected commit": PHASE11_CORRECTED_COMMIT,
        "Corrected tree": PHASE11_CORRECTED_TREE,
        "Extension SHA": PHASE11_EXTENSION_SHA256,
        "Calibration ID": PHASE11_CALIBRATION_ID,
        "Calibration root": PHASE11_CALIBRATION_ROOT,
        "Historical Phase 10 root": PHASE11_HISTORICAL_FIXTURE_ROOT,
        "Corrected fixture ID": PHASE11_FIXTURE_ID,
        "Corrected fixture root": PHASE11_FIXTURE_ROOT,
        "Adapter location": "src/kvbench/adapters/kvquant.py",
        "Supported configurations": "kvq4, kvq3, kvq2",
        "Boundary semantics": (
            "PRE-ROPE KEY QUANTIZATION; ATTENTION-READY SINK KEY"
        ),
        "Static cache": "PASS",
        "Fixture conformance": "9/9 PASS",
        "Execution-path and GQA audit": "PASS",
        "Eager allocation": "PASS",
        "CUDA Graph": "PASS",
        "Sanitizer": "PASS",
        "Bounded admission": "9/9 PASS",
        "Admission run IDs": ", ".join(
            point.run_id for point in report.bounded_runs
        ),
        "MethodAdmissionReport SHA-256": report_sha256,
        "Inner R2 URI": str(report.r2_uri),
        "G2-KVQ": "PASS",
        "Global G2": "NOT EVALUATED",
        "G3": "NOT EVALUATED",
        "G4": "NOT EVALUATED",
        "G5": "NOT EVALUATED",
        "Full Scan": "CLOSED",
        "Quality execution": "LOCKED",
        "PERFORMANCE_DATA_FROZEN": "ABSENT",
        "Performance claim eligible": "FALSE",
        "Speedup calculated": "NO",
        "r_hbm": "NULL",
        "Historical evidence changed": "NO",
        "Existing methods changed": "NO",
        "Measurement Container changed": "NO",
        "Phase 12 started": "NO",
    }
    if (
        re.match(
            rf"^\s*#?\s*{re.escape(PASS_REPORT_HEADING)}\s*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        is None
        or any(
            not exact_state(label, expected.upper())
            for label, expected in required_states.items()
        )
        or any(
            re.search(
                rf"(?im)^\s*(?:-\s*)?{re.escape(key)}\s*[:=]",
                text,
            )
            for key in _PROHIBITED_CREDENTIAL_KEYS
        )
    ):
        raise Phase11OuterBundleError(
            "Phase 11 final report governance differs"
        )


def _validate_report_join(
    *,
    repository_root: Path,
    source_bundle: Path,
    inner: _InnerClosure,
) -> Phase11MethodAdmissionReport:
    report_path = repository_root / METHOD_ADMISSION_RELATIVE
    checksum_path = repository_root / METHOD_ADMISSION_CHECKSUM_RELATIVE
    receipt_path = repository_root / INNER_RECEIPT_RELATIVE
    phase_report_path = repository_root / PASS_REPORT_RELATIVE
    for path in (
        report_path,
        checksum_path,
        receipt_path,
        phase_report_path,
    ):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
        ):
            raise Phase11OuterBundleError(
                "Phase 11 governance evidence is absent or unsafe"
            )
    if checksum_path.read_bytes() != _method_checksum_bytes(report_path):
        raise Phase11OuterBundleError(
            "Phase 11 MethodAdmissionReport checksum differs"
        )
    receipt = _strict_json(
        receipt_path,
        "Phase 11 inner publication receipt",
    )
    lock_identity = _validate_publication_receipt_core(
        receipt,
        expected_schema=INNER_RECEIPT_SCHEMA,
        artifact=inner.artifact,
        source_run_id=inner.manifest.run_id,
        source_git_sha=inner.manifest.git_sha,
        repository_root=repository_root,
    )
    report_payload = _strict_json(
        report_path,
        "Phase 11 MethodAdmissionReport",
    )
    _reject_governance_drift(report_payload)
    try:
        report = Phase11MethodAdmissionReport.from_dict(report_payload)
    except (TypeError, ValueError) as error:
        raise Phase11OuterBundleError(
            "Phase 11 MethodAdmissionReport is invalid"
        ) from error
    from scripts.phase11_kvquant_admission import (
        derive_phase11_method_admission_report,
    )

    expected_report = derive_phase11_method_admission_report(
        bundle_path=source_bundle,
        publication_receipt_path=receipt_path,
        created_at_utc=report.created_at_utc,
        repository_root=repository_root,
    )
    if report != expected_report:
        raise Phase11OuterBundleError(
            "Phase 11 MethodAdmissionReport was not derived exactly "
            "from validated inner evidence"
        )
    if report.bucket_lock_identity != lock_identity:
        raise Phase11OuterBundleError(
            "Phase 11 report Bucket Lock identity differs"
        )
    source_relative = _repository_relative(
        source_bundle,
        repository_root,
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
            safe_relative = _safe_inner_relative(inner_relative)
            if safe_relative.as_posix() not in inner.artifact.by_path():
                raise Phase11OuterBundleError(
                    "Phase 11 evidence reference is outside the "
                    "inner inventory"
                )
            evidence_path = source_bundle.joinpath(*safe_relative.parts)
        else:
            raise Phase11OuterBundleError(
                "Phase 11 report evidence escaped the inner closure"
            )
        if (
            not evidence_path.is_file()
            or evidence_path.is_symlink()
            or sha256_file(evidence_path) != reference.sha256
        ):
            raise Phase11OuterBundleError(
                "Phase 11 report evidence reference differs"
            )
    for check_id in ("durable_publication", "clean_retrieval"):
        checks = [
            check for check in report.checks if check.check_id == check_id
        ]
        if (
            len(checks) != 1
            or not any(
                by_id[evidence_id].path
                == INNER_RECEIPT_RELATIVE.as_posix()
                for evidence_id in checks[0].evidence_ids
            )
        ):
            raise Phase11OuterBundleError(
                "Phase 11 durable checks do not cite the inner receipt"
            )
    _validate_phase_report(
        phase_report_path,
        report=report,
        report_sha256=sha256_file(report_path),
    )
    return report


def _inner_binding(
    *,
    inner: _InnerClosure,
    copy_prefix: PurePosixPath,
) -> dict[str, object]:
    records = [
        {
            "index": index,
            "run_id": point.run_id,
            "path": relative,
            "sha256": point.manifest_sha256,
        }
        for index, (point, relative) in enumerate(
            zip(inner.points, inner.point_paths)
        )
    ]
    return {
        "run_id": inner.manifest.run_id,
        "root_sha256": inner.artifact.root_sha256,
        "bounded_grid_path": (
            copy_prefix / "validation" / "bounded-grid.json"
        ).as_posix(),
        "bounded_grid_sha256": inner.bounded_grid_sha256,
        "point_records": records,
    }


def _role(relative: str, copy_prefix: PurePosixPath) -> str:
    if relative == "manifest.json":
        return "manifest"
    if relative.startswith(f"{copy_prefix.as_posix()}/"):
        return "phase11_inner_bundle"
    if relative == INNER_REFERENCE_PATH.as_posix():
        return "inner_root_and_run_binding"
    if relative == BUNDLED_METHOD_ADMISSION_PATH.as_posix():
        return "method_admission_report"
    if relative == BUNDLED_METHOD_CHECKSUM_PATH.as_posix():
        return "method_admission_checksum"
    if relative == BUNDLED_INNER_RECEIPT_PATH.as_posix():
        return "inner_r2_publication_receipt"
    if relative in {
        BUNDLED_INNER_PUBLISH_STDOUT_PATH.as_posix(),
        BUNDLED_INNER_PUBLISH_STDERR_PATH.as_posix(),
        BUNDLED_INNER_VERIFY_STDOUT_PATH.as_posix(),
        BUNDLED_INNER_VERIFY_STDERR_PATH.as_posix(),
    }:
        return "inner_r2_raw_tool_evidence"
    if relative == BUNDLED_PASS_REPORT_PATH.as_posix():
        return "phase11_report"
    return "phase11_outer_bundle"


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
        BUNDLED_INNER_RECEIPT_PATH.as_posix(),
        BUNDLED_INNER_PUBLISH_STDOUT_PATH.as_posix(),
        BUNDLED_INNER_PUBLISH_STDERR_PATH.as_posix(),
        BUNDLED_INNER_VERIFY_STDOUT_PATH.as_posix(),
        BUNDLED_INNER_VERIFY_STDERR_PATH.as_posix(),
        BUNDLED_METHOD_ADMISSION_PATH.as_posix(),
        BUNDLED_METHOD_CHECKSUM_PATH.as_posix(),
        BUNDLED_PASS_REPORT_PATH.as_posix(),
        (copy_prefix / "manifest.json").as_posix(),
        (copy_prefix / "artifact_inventory.json").as_posix(),
        (copy_prefix / "checksums.sha256").as_posix(),
        (copy_prefix / "COMPLETE").as_posix(),
    )


def _bundled_repository_paths() -> dict[Path, PurePosixPath]:
    return {
        INNER_RECEIPT_RELATIVE: BUNDLED_INNER_RECEIPT_PATH,
        INNER_PUBLISH_STDOUT_RELATIVE: BUNDLED_INNER_PUBLISH_STDOUT_PATH,
        INNER_PUBLISH_STDERR_RELATIVE: BUNDLED_INNER_PUBLISH_STDERR_PATH,
        INNER_VERIFY_STDOUT_RELATIVE: BUNDLED_INNER_VERIFY_STDOUT_PATH,
        INNER_VERIFY_STDERR_RELATIVE: BUNDLED_INNER_VERIFY_STDERR_PATH,
        METHOD_ADMISSION_RELATIVE: BUNDLED_METHOD_ADMISSION_PATH,
        METHOD_ADMISSION_CHECKSUM_RELATIVE: (
            BUNDLED_METHOD_CHECKSUM_PATH
        ),
        PASS_REPORT_RELATIVE: BUNDLED_PASS_REPORT_PATH,
    }


def validate_outer_bundle(
    artifact_root: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    source_bundle: Path,
) -> Phase11OuterBundleValidation:
    """Validate the exact inner copy, nine runs, and report-bearing closure."""

    if source_bundle.is_symlink():
        raise Phase11OuterBundleError(
            "Phase 11 inner bundle path is a symlink"
        )
    repository = repository_root.resolve(strict=True)
    source = source_bundle.resolve(strict=True)
    inner = _validate_inner_bundle(source)
    if OUTER_RECEIPT_RELATIVE.as_posix() in inner.artifact.by_path():
        raise Phase11OuterBundleError(
            "Phase 11 inner bundle contains the outer receipt"
        )
    report = _validate_report_join(
        repository_root=repository,
        source_bundle=source,
        inner=inner,
    )
    copy_prefix = PurePosixPath(
        "source-inner", "sha256", inner.artifact.root_sha256
    )
    artifact = validate_local_artifact(artifact_root)
    manifest = _strict_json(
        artifact_root / "manifest.json",
        "Phase 11 outer manifest",
    )
    _reject_governance_drift(manifest)
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
        or manifest.get("inner_run_id") != inner.manifest.run_id
        or manifest.get("inner_root_sha256")
        != inner.artifact.root_sha256
        or manifest.get("inner_r2_uri")
        != _r2_uri(inner.artifact.root_sha256)
        or manifest.get("inner_object_count")
        != len(inner.artifact.files)
        or manifest.get("bounded_grid_sha256")
        != inner.bounded_grid_sha256
        or manifest.get("admission_run_ids")
        != [point.run_id for point in inner.points]
        or manifest.get("method_admission_sha256")
        != sha256_file(repository / METHOD_ADMISSION_RELATIVE)
        or manifest.get("required_paths") != list(required_paths)
        or manifest.get("g2_kvq") != "PASS"
        or manifest.get("global_g2") != "NOT_EVALUATED"
        or manifest.get("performance_claim_eligible") is not False
        or manifest.get("speedup_calculated") is not False
        or manifest.get("r_hbm") is not None
        or manifest.get("quality_execution") != "LOCKED"
        or manifest.get("quality_benchmark_executed") is not False
        or manifest.get("performance_data_frozen") is not False
        or manifest.get("full_scan") != "CLOSED"
        or manifest.get("phase12_started") is not False
    ):
        raise Phase11OuterBundleError(
            "Phase 11 outer manifest is invalid"
        )
    outer_by_path = artifact.by_path()
    inner_by_path = inner.artifact.by_path()
    copied_paths = {
        relative[len(copy_prefix.as_posix()) + 1 :]: record
        for relative, record in outer_by_path.items()
        if relative.startswith(f"{copy_prefix.as_posix()}/")
    }
    if set(copied_paths) != set(inner_by_path):
        raise Phase11OuterBundleError(
            "outer bundle does not contain the complete Phase 11 inner bundle"
        )
    for relative, source_record in inner_by_path.items():
        copied = copied_paths[relative]
        if (
            copied.size_bytes != source_record.size_bytes
            or copied.sha256 != source_record.sha256
        ):
            raise Phase11OuterBundleError(
                "copied Phase 11 inner bundle bytes differ"
            )
    for source_relative, bundled_relative in _bundled_repository_paths().items():
        bundled = outer_by_path.get(bundled_relative.as_posix())
        source_path = repository / source_relative
        if (
            bundled is None
            or not source_path.is_file()
            or source_path.is_symlink()
            or source_path.stat().st_nlink != 1
            or bundled.size_bytes != source_path.stat().st_size
            or bundled.sha256 != sha256_file(source_path)
        ):
            raise Phase11OuterBundleError(
                "required repository evidence differs: "
                f"{source_relative.as_posix()}"
            )
    if OUTER_RECEIPT_RELATIVE.as_posix() in outer_by_path:
        raise Phase11OuterBundleError(
            "Phase 11 outer publication receipt is self-included"
        )
    expected_binding = _inner_binding(
        inner=inner,
        copy_prefix=copy_prefix,
    )
    observed_binding = _strict_json(
        artifact_root / INNER_REFERENCE_PATH,
        "Phase 11 inner binding",
    )
    if observed_binding != expected_binding:
        raise Phase11OuterBundleError(
            "Phase 11 inner root or run binding differs"
        )
    if any(relative not in outer_by_path for relative in required_paths):
        raise Phase11OuterBundleError(
            "Phase 11 outer bundle lacks a required path"
        )
    expected_paths = {
        *(f"{copy_prefix.as_posix()}/{path}" for path in inner_by_path),
        INNER_REFERENCE_PATH.as_posix(),
        *(
            path.as_posix()
            for path in _bundled_repository_paths().values()
        ),
        "manifest.json",
        "artifact_inventory.json",
        "checksums.sha256",
        "COMPLETE",
    }
    if set(outer_by_path) != expected_paths:
        raise Phase11OuterBundleError(
            "Phase 11 outer bundle contains an unexpected object"
        )
    source_relative = _repository_relative(source, repository)
    source_prefix = f"{source_relative}/"
    for reference in report.evidence_references:
        if reference.path == INNER_RECEIPT_RELATIVE.as_posix():
            bundled_path = BUNDLED_INNER_RECEIPT_PATH.as_posix()
        else:
            if not reference.path.startswith(source_prefix):
                raise Phase11OuterBundleError(
                    "Phase 11 report evidence escaped the outer closure"
                )
            inner_relative = _safe_inner_relative(
                reference.path[len(source_prefix) :]
            )
            bundled_path = (
                copy_prefix
                / inner_relative
            ).as_posix()
        bundled = outer_by_path.get(bundled_path)
        if bundled is None or bundled.sha256 != reference.sha256:
            raise Phase11OuterBundleError(
                "Phase 11 report evidence is not exact in the outer bundle"
            )
    return Phase11OuterBundleValidation(
        run_id=run_id,
        root_sha256=artifact.root_sha256,
        object_count=len(artifact.files),
        inner_root_sha256=inner.artifact.root_sha256,
        inner_object_count=len(inner.artifact.files),
        admission_run_count=len(inner.points),
        bounded_grid_sha256=inner.bounded_grid_sha256,
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
) -> Phase11OuterPublicationValidation:
    """Bind one external receipt to the exact report-bearing outer bundle."""

    repository = repository_root.resolve(strict=True)
    if artifact_root.is_symlink():
        raise Phase11OuterBundleError(
            "Phase 11 outer artifact path is a symlink"
        )
    artifact_path = artifact_root.resolve(strict=True)
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
        raise Phase11OuterBundleError(
            "Phase 11 outer publication receipt is absent or unsafe"
        )
    try:
        receipt.relative_to(artifact_path)
    except ValueError:
        pass
    else:
        raise Phase11OuterBundleError(
            "Phase 11 outer publication receipt is self-included"
        )
    validation = validate_outer_bundle(
        artifact_path,
        repository_root=repository,
        source_bundle=source_bundle,
    )
    artifact = validate_local_artifact(artifact_path)
    manifest = _strict_json(
        artifact_path / "manifest.json",
        "Phase 11 outer manifest",
    )
    payload = _strict_json(
        receipt,
        "Phase 11 outer publication receipt",
    )
    self_reference = _require_mapping(
        payload.get("self_reference_control"),
        label="outer receipt self-reference control",
    )
    if (
        payload.get("g2_kvq") != "PASS"
        or payload.get("global_g2") != "NOT_EVALUATED"
        or payload.get("performance_claim_eligible") is not False
        or payload.get("speedup_calculated") is not False
        or payload.get("r_hbm") is not None
        or payload.get("quality_execution") != "LOCKED"
        or payload.get("quality_benchmark_executed") is not False
        or payload.get("performance_data_frozen") is not False
        or payload.get("full_scan") != "CLOSED"
        or payload.get("phase12_started") is not False
        or self_reference.get("included_in_bundle") is not False
        or self_reference.get("receipt_path")
        != OUTER_RECEIPT_RELATIVE.as_posix()
    ):
        raise Phase11OuterBundleError(
            "Phase 11 outer receipt governance differs"
        )
    lock_identity = _validate_publication_receipt_core(
        payload,
        expected_schema=OUTER_RECEIPT_SCHEMA,
        artifact=artifact,
        source_run_id=validation.run_id,
        source_git_sha=str(manifest["source_git_sha"]),
        repository_root=repository,
    )
    return Phase11OuterPublicationValidation(
        run_id=validation.run_id,
        root_sha256=validation.root_sha256,
        object_count=validation.object_count,
        r2_uri=_r2_uri(validation.root_sha256),
        receipt_path=OUTER_RECEIPT_RELATIVE.as_posix(),
        receipt_sha256=sha256_file(receipt),
        bucket_lock_identity=lock_identity,
    )


def build_outer_bundle(
    *,
    repository_root: Path,
    source_bundle: Path,
    output_root: Path,
    run_id: str,
    source_git_sha: str,
) -> tuple[Path, Phase11OuterBundleValidation]:
    """Build one immutable, no-replace Phase 11 outer bundle locally."""

    validate_run_id(run_id)
    if _GIT_SHA_RE.fullmatch(source_git_sha) is None:
        raise Phase11OuterBundleError("source Git SHA is invalid")
    if source_bundle.is_symlink():
        raise Phase11OuterBundleError(
            "Phase 11 inner bundle path is a symlink"
        )
    repository = repository_root.resolve(strict=True)
    source = source_bundle.resolve(strict=True)
    expected_output_root = repository / OUTER_ARTIFACT_ROOT_RELATIVE
    if output_root.resolve(strict=False) != expected_output_root:
        raise Phase11OuterBundleError(
            "Phase 11 outer artifact root is not the fixed path"
        )
    inner = _validate_inner_bundle(source)
    if OUTER_RECEIPT_RELATIVE.as_posix() in inner.artifact.by_path():
        raise Phase11OuterBundleError(
            "Phase 11 inner bundle contains the outer receipt"
        )
    _validate_report_join(
        repository_root=repository,
        source_bundle=source,
        inner=inner,
    )
    repository_files: dict[PurePosixPath, bytes] = {}
    for source_relative, bundled_relative in _bundled_repository_paths().items():
        path = repository / source_relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
        ):
            raise Phase11OuterBundleError(
                "required repository evidence is unsafe: "
                f"{source_relative.as_posix()}"
            )
        repository_files[bundled_relative] = path.read_bytes()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise Phase11OuterBundleError(
            "Phase 11 outer artifact root is unsafe"
        )
    final = output_root / run_id
    if final.exists() or final.is_symlink():
        raise Phase11OuterBundleError(
            "Phase 11 outer bundle run ID already exists"
        )
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{run_id}.",
            suffix=".staging",
            dir=output_root,
        )
    )
    copy_prefix = PurePosixPath(
        "source-inner", "sha256", inner.artifact.root_sha256
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
            _inner_binding(inner=inner, copy_prefix=copy_prefix)
        ),
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": run_id,
        "status": "completed",
        "scope": (
            "Append-only Phase 11 KVQuant R2 outer admission bundle; "
            "no experiment execution, performance claim, quality "
            "execution, or Phase 12 work."
        ),
        "source_git_sha": source_git_sha,
        "inner_run_id": inner.manifest.run_id,
        "inner_root_sha256": inner.artifact.root_sha256,
        "inner_r2_uri": _r2_uri(inner.artifact.root_sha256),
        "inner_object_count": len(inner.artifact.files),
        "bounded_grid_sha256": inner.bounded_grid_sha256,
        "admission_run_ids": [point.run_id for point in inner.points],
        "method_admission_sha256": sha256_file(
            repository / METHOD_ADMISSION_RELATIVE
        ),
        "required_paths": list(_expected_required_paths(copy_prefix)),
        "g2_kvq": "PASS",
        "global_g2": "NOT_EVALUATED",
        "performance_claim_eligible": False,
        "speedup_calculated": False,
        "r_hbm": None,
        "quality_execution": "LOCKED",
        "quality_benchmark_executed": False,
        "performance_data_frozen": False,
        "full_scan": "CLOSED",
        "phase12_started": False,
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
        raise Phase11OuterBundleError(
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
        raise Phase11OuterBundleError("source Git SHA is invalid")
    return revision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authority-profile",
        choices=AUTHORITY_PROFILE_CHOICES,
        default=AUTHORITY_PROFILE_DECISION0027,
        help=(
            "select the exact Decision 0027 historical or Decision 0029 "
            "successor authority"
        ),
    )
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
    receipt = commands.add_parser("assemble-receipt")
    receipt.add_argument("--kind", choices=("inner", "outer"), required=True)
    receipt.add_argument("--artifact", required=True, type=Path)
    receipt.add_argument("--publish-output", required=True, type=Path)
    receipt.add_argument("--publish-stderr", required=True, type=Path)
    receipt.add_argument("--verify-output", required=True, type=Path)
    receipt.add_argument("--verify-stderr", required=True, type=Path)
    receipt.add_argument("--output", required=True, type=Path)
    receipt.add_argument("--source-run-id", required=True)
    receipt.add_argument("--source-git-sha", required=True)
    receipt.add_argument("--recorded-at-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        _activate_profile(arguments.authority_profile)
        if arguments.operation == "build":
            source = arguments.source_bundle.resolve(strict=True)
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
            source = arguments.source_bundle.resolve(strict=True)
            validation = validate_outer_bundle(
                arguments.artifact.resolve(strict=True),
                repository_root=REPOSITORY_ROOT,
                source_bundle=source,
            )
            payload = validation.to_dict()
        elif arguments.operation == "validate-publication":
            source = arguments.source_bundle.resolve(strict=True)
            publication = validate_outer_publication_receipt(
                arguments.artifact.resolve(strict=True),
                receipt_path=arguments.receipt,
                repository_root=REPOSITORY_ROOT,
                source_bundle=source,
            )
            payload = publication.to_dict()
        else:
            expected = REPOSITORY_ROOT / (
                INNER_RECEIPT_RELATIVE
                if arguments.kind == "inner"
                else OUTER_RECEIPT_RELATIVE
            )
            if arguments.output.resolve(strict=False) != expected:
                raise Phase11OuterBundleError(
                    "publication receipt output path differs"
                )
            output = write_publication_receipt(
                output_path=arguments.output,
                artifact_root=arguments.artifact,
                publish_output=arguments.publish_output,
                publish_stderr=arguments.publish_stderr,
                verify_output=arguments.verify_output,
                verify_stderr=arguments.verify_stderr,
                receipt_kind=arguments.kind,
                source_run_id=arguments.source_run_id,
                source_git_sha=arguments.source_git_sha,
                recorded_at_utc=arguments.recorded_at_utc,
            )
            payload = {
                "status": "PASS",
                "receipt_kind": arguments.kind,
                "receipt_path": str(output.resolve(strict=True)),
                "receipt_sha256": sha256_file(output),
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (
        OSError,
        Phase11OuterBundleError,
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

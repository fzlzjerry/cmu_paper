"""Decision 0029 authority compatibility for the Phase 11 schemas."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import unittest

from kvbench.errors import SchemaValidationError
from kvbench.runtime.artifacts import _validate_manifest
from kvbench.schema.phase11 import (
    PHASE11_AGGREGATE_PATCH_SHA256,
    PHASE11_AUTHORIZED_CONTAINER_DIGEST,
    PHASE11_CORRECTED_TREE,
    PHASE11_EXTENSION_SHA256,
    PHASE11Q23_AGGREGATE_PATCH_SHA256,
    PHASE11Q23_AUTHORIZED_CONTAINER_DIGEST,
    PHASE11Q23_CALIBRATION_ID,
    PHASE11Q23_CALIBRATION_ROOT,
    PHASE11Q23_CORRECTED_COMMIT,
    PHASE11Q23_CORRECTED_CUDA_SHA256,
    PHASE11Q23_CORRECTED_TREE,
    PHASE11Q23_DECISION_0021_PATCH_SHA256,
    PHASE11Q23_DECISIONS,
    PHASE11Q23_EXECUTION_SOURCE_IDENTIFIER,
    PHASE11Q23_EXTENSION_SHA256,
    PHASE11Q23_FIXTURE_ID,
    PHASE11Q23_FIXTURE_ROOT,
    PHASE11Q23_HISTORICAL_FIXTURE_ID,
    PHASE11Q23_HISTORICAL_FIXTURE_ROOT,
    PHASE11Q23_METHOD_IDENTIFIER,
    PHASE11Q23_UPSTREAM_BASE_COMMIT,
    PHASE11Q23_UPSTREAM_BASE_TREE,
    Phase11Authority,
    Phase11MethodAdmissionReport,
    Phase11RQ23MethodAdmissionReport,
    Phase11RQ23RunManifest,
    Phase11RunManifest,
    Phase11SanitizerEvidence,
)


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_REPORT = (
    ROOT / "docs/evidence/phase11/kvquant-method-admission.json"
)
HISTORICAL_REPORT_SHA256 = (
    "59ef5bfc581a68cdc4d21c4c0a840f046e698633f7475f79906063c6e333ae6a"
)
HISTORICAL_CANONICAL_SHA256 = (
    "b394dfeb0b199bcdf9e234bb93d94e064943b2f4233e141ca26b8cc065eab2ae"
)


def _q23_authority_payload() -> dict[str, object]:
    return {
        "method_identifier": PHASE11Q23_METHOD_IDENTIFIER,
        "execution_source_identifier": (
            PHASE11Q23_EXECUTION_SOURCE_IDENTIFIER
        ),
        "upstream_base_commit": PHASE11Q23_UPSTREAM_BASE_COMMIT,
        "upstream_base_tree": PHASE11Q23_UPSTREAM_BASE_TREE,
        "decision_0021_patch_sha256": (
            PHASE11Q23_DECISION_0021_PATCH_SHA256
        ),
        "aggregate_patch_sha256": PHASE11Q23_AGGREGATE_PATCH_SHA256,
        "corrected_commit": PHASE11Q23_CORRECTED_COMMIT,
        "corrected_tree": PHASE11Q23_CORRECTED_TREE,
        "corrected_cuda_sha256": PHASE11Q23_CORRECTED_CUDA_SHA256,
        "extension_sha256": PHASE11Q23_EXTENSION_SHA256,
        "decisions": list(PHASE11Q23_DECISIONS),
        "calibration_id": PHASE11Q23_CALIBRATION_ID,
        "calibration_root": PHASE11Q23_CALIBRATION_ROOT,
        "historical_fixture_id": PHASE11Q23_HISTORICAL_FIXTURE_ID,
        "historical_fixture_root": PHASE11Q23_HISTORICAL_FIXTURE_ROOT,
        "fixture_id": PHASE11Q23_FIXTURE_ID,
        "fixture_root": PHASE11Q23_FIXTURE_ROOT,
        "authorized_container_digest": (
            PHASE11Q23_AUTHORIZED_CONTAINER_DIGEST
        ),
    }


def _historical_report_payload() -> dict[str, object]:
    return json.loads(HISTORICAL_REPORT.read_text(encoding="utf-8"))


def _q23_report_payload() -> dict[str, object]:
    payload = copy.deepcopy(_historical_report_payload())
    payload["schema_version"] = (
        Phase11RQ23MethodAdmissionReport.SCHEMA_VERSION
    )
    payload["authority"] = _q23_authority_payload()
    sanitizer = payload["sanitizer_evidence"]
    assert isinstance(sanitizer, dict)
    sanitizer["corrected_tree"] = PHASE11Q23_CORRECTED_TREE
    sanitizer["extension_sha256"] = PHASE11Q23_EXTENSION_SHA256
    return payload


def _q23_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": Phase11RQ23RunManifest.SCHEMA_VERSION,
        "artifact_schema_version": (
            Phase11RQ23RunManifest.ARTIFACT_SCHEMA_VERSION
        ),
        "run_id": "phase11rq23-schema-test",
        "status": "created",
        "created_at_utc": "2026-07-30T00:00:00Z",
        "started_at_utc": None,
        "finished_at_utc": None,
        "run_kind": "phase11_admission",
        "git_sha": PHASE11Q23_CORRECTED_COMMIT,
        "git_dirty": False,
        "authority": _q23_authority_payload(),
        "bounded_point_count": 9,
        "measurement_scope": "measurement_container_admission",
        "quality_status": "unvalidated",
        "claim_eligibility": "performance_only",
        "quality_execution": "locked",
        "performance_claim_eligible": False,
        "performance_data_frozen": False,
        "quality_benchmark_executed": False,
        "speedup_calculated": False,
        "r_hbm": None,
        "full_scan_state": "CLOSED",
        "g2_kvq_state": "NOT_EVALUATED_PUBLICATION_PENDING",
        "global_g2_g5_state": "NOT_EVALUATED",
        "inventory_path": None,
        "failure_reason": None,
    }


class Phase11RQ23SchemaTests(unittest.TestCase):
    def test_historical_report_and_canonical_hash_remain_unchanged(self) -> None:
        raw = HISTORICAL_REPORT.read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            HISTORICAL_REPORT_SHA256,
        )
        report = Phase11MethodAdmissionReport.from_dict(json.loads(raw))
        self.assertEqual(report.fingerprint(), HISTORICAL_CANONICAL_SHA256)
        self.assertEqual(
            hashlib.sha256(HISTORICAL_REPORT.read_bytes()).hexdigest(),
            HISTORICAL_REPORT_SHA256,
        )

    def test_exact_q23_authority_manifest_and_report_are_accepted(self) -> None:
        authority = Phase11Authority.from_dict(_q23_authority_payload())
        self.assertEqual(
            authority.aggregate_patch_sha256,
            PHASE11Q23_AGGREGATE_PATCH_SHA256,
        )
        manifest = Phase11RQ23RunManifest.from_dict(
            _q23_manifest_payload()
        )
        self.assertEqual(
            manifest.authority.corrected_tree,
            PHASE11Q23_CORRECTED_TREE,
        )
        report = Phase11RQ23MethodAdmissionReport.from_dict(
            _q23_report_payload()
        )
        self.assertEqual(
            report.sanitizer_evidence.extension_sha256,
            PHASE11Q23_EXTENSION_SHA256,
        )
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(report)),
            tuple(
                field.name
                for field in dataclasses.fields(
                    Phase11MethodAdmissionReport
                )
            ),
        )
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(manifest)),
            tuple(
                field.name for field in dataclasses.fields(Phase11RunManifest)
            ),
        )

    def test_q23_run_manifest_is_registered_with_artifact_lifecycle(
        self,
    ) -> None:
        parsed = _validate_manifest(_q23_manifest_payload())
        self.assertIsInstance(parsed, Phase11RQ23RunManifest)

    def test_mixed_or_tampered_authority_is_rejected(self) -> None:
        mixed = _q23_authority_payload()
        mixed["aggregate_patch_sha256"] = PHASE11_AGGREGATE_PATCH_SHA256
        with self.assertRaises(SchemaValidationError):
            Phase11Authority.from_dict(mixed)

        tampered_values: dict[str, object] = {
            "method_identifier": "tampered",
            "execution_source_identifier": "tampered",
            "upstream_base_commit": "0" * 40,
            "upstream_base_tree": "0" * 40,
            "decision_0021_patch_sha256": "0" * 64,
            "aggregate_patch_sha256": "0" * 64,
            "corrected_commit": "0" * 40,
            "corrected_tree": "0" * 40,
            "corrected_cuda_sha256": "0" * 64,
            "extension_sha256": "0" * 64,
            "decisions": ["0021", "0029"],
            "calibration_id": "tampered",
            "calibration_root": "0" * 64,
            "historical_fixture_id": "tampered",
            "historical_fixture_root": "0" * 64,
            "fixture_id": "tampered",
            "fixture_root": "0" * 64,
            "authorized_container_digest": f"sha256:{'0' * 64}",
        }
        for field_name, value in tampered_values.items():
            with self.subTest(field_name=field_name):
                payload = _q23_authority_payload()
                payload[field_name] = value
                with self.assertRaises(SchemaValidationError):
                    Phase11Authority.from_dict(payload)

    def test_schema_and_sanitizer_profiles_cannot_be_crossed(self) -> None:
        q23_with_legacy_sanitizer = _q23_report_payload()
        sanitizer = q23_with_legacy_sanitizer["sanitizer_evidence"]
        assert isinstance(sanitizer, dict)
        sanitizer["corrected_tree"] = PHASE11_CORRECTED_TREE
        sanitizer["extension_sha256"] = PHASE11_EXTENSION_SHA256
        with self.assertRaises(SchemaValidationError):
            Phase11RQ23MethodAdmissionReport.from_dict(
                q23_with_legacy_sanitizer
            )

        historical_with_q23_schema = _historical_report_payload()
        historical_with_q23_schema["schema_version"] = (
            Phase11RQ23MethodAdmissionReport.SCHEMA_VERSION
        )
        with self.assertRaises(SchemaValidationError):
            Phase11RQ23MethodAdmissionReport.from_dict(
                historical_with_q23_schema
            )

        historical_sanitizer = _historical_report_payload()[
            "sanitizer_evidence"
        ]
        assert isinstance(historical_sanitizer, dict)
        mixed_sanitizer = copy.deepcopy(historical_sanitizer)
        mixed_sanitizer["corrected_tree"] = PHASE11Q23_CORRECTED_TREE
        mixed_sanitizer["extension_sha256"] = PHASE11_EXTENSION_SHA256
        with self.assertRaises(SchemaValidationError):
            Phase11SanitizerEvidence.from_dict(mixed_sanitizer)

        mixed_sanitizer = copy.deepcopy(historical_sanitizer)
        mixed_sanitizer["corrected_tree"] = PHASE11_CORRECTED_TREE
        mixed_sanitizer["extension_sha256"] = PHASE11Q23_EXTENSION_SHA256
        with self.assertRaises(SchemaValidationError):
            Phase11SanitizerEvidence.from_dict(mixed_sanitizer)

        self.assertEqual(
            PHASE11Q23_AUTHORIZED_CONTAINER_DIGEST,
            PHASE11_AUTHORIZED_CONTAINER_DIGEST,
        )


if __name__ == "__main__":
    unittest.main()

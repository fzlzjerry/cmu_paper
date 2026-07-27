"""Focused tests for the append-only Phase 8 R2 outer bundle."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from kvbench.runtime.artifacts import phase8_artifact_store
from kvbench.schema import (
    ClaimEligibility,
    MeasurementScope,
    QualityExecutionState,
    QualityValidationState,
    RunStatus,
)
from kvbench.schema.method_admission import (
    MethodAdmissionEvidenceReference,
)
from kvbench.schema.phase3 import GateDisposition
from kvbench.schema.phase8 import (
    PHASE8_ADMISSION_CHECK_IDS,
    PHASE8_HELD_OUT_CONFIG,
    PHASE8_MANDATORY_CONFIGS,
    Phase8AdmissionCheck,
    Phase8AdmissionGates,
    Phase8MethodAdmissionReport,
)
from preflight.run_preflight import json_bytes, sha256_file
from scripts.phase8_r2_outer_bundle import (
    ADMISSION_REFERENCES_PATH,
    INNER_RECEIPT_RELATIVE,
    INNER_REFERENCE_PATH,
    METHOD_ADMISSION_CHECKSUM_RELATIVE,
    METHOD_ADMISSION_RELATIVE,
    OUTER_RECEIPT_RELATIVE,
    PASS_REPORT_RELATIVE,
    REQUIRED_REPOSITORY_FILES,
    Phase8OuterBundleError,
    build_outer_bundle,
    validate_outer_bundle,
    validate_outer_publication_receipt,
)
from scripts.r2_artifact import (
    publication_order,
    validate_local_artifact,
)
from tests.unit.test_phase8_kivi_admission import (
    _manifests,
)


SOURCE_GIT_SHA = "8" * 40
OUTER_GIT_SHA = "a" * 40


def _initial_manifest(manifest):
    return dataclasses.replace(
        manifest,
        status=RunStatus.CREATED,
        started_at_utc=None,
        finished_at_utc=None,
        inventory_path=None,
        failure_reason=None,
    )


def _write_required_point_payloads(run: object, index: int) -> None:
    run.write_json(
        "config/method.json",
        {"method": "kivi", "configuration_index": index},
    )
    run.write_json(
        "environment/container_identity.json",
        {"container": "authorized-phase8-test-identity"},
    )
    run.write_json("raw/runner.json", {"point_index": index})
    run.write_json("validation/point.json", {"passed": True})


def _artifact_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _make_inner_bundle(repository: Path):
    manifests = tuple(
        dataclasses.replace(
            manifest,
            accounting=dataclasses.replace(
                manifest.accounting,
                active_context=(
                    manifest.context_length
                    if manifest.runner_kind.value == "fixed_l"
                    else manifest.capacity
                ),
                logical_bf16_active_bytes=(
                    (
                        manifest.context_length
                        if manifest.runner_kind.value == "fixed_l"
                        else manifest.capacity
                    )
                    * 1024
                ),
            ),
        )
        for manifest in _manifests()
    )
    store = phase8_artifact_store(repository)
    bundle_manifest = manifests[0]
    bundle_run = store.create(
        bundle_manifest.run_id,
        _initial_manifest(bundle_manifest),
    )
    bundle_run.start()
    _write_required_point_payloads(bundle_run, 0)
    bundle_run.write_json(
        "validation/evidence.json",
        {
            "schema_version": "phase8-test-inner-evidence-1.0.0",
            "status": "PASS",
        },
    )

    embedded_ids: list[str] = []
    for index, manifest in enumerate(manifests[1:], start=1):
        run = store.create(manifest.run_id, _initial_manifest(manifest))
        run.start()
        _write_required_point_payloads(run, index)
        completed = run.finalize(manifest)
        for relative, source in _artifact_files(completed).items():
            bundle_run.write_bytes(
                f"grid-runs/{manifest.run_id}/{relative}",
                source.read_bytes(),
            )
        embedded_ids.append(manifest.run_id)

    run_ids = [manifest.run_id for manifest in manifests]
    bundle_run.write_json(
        "validation/bounded-grid.json",
        {
            "schema_version": "kvbench-phase8-kivi-bounded-grid-1.0.0",
            "run_ids": run_ids,
            "embedded_run_ids": embedded_ids,
            "bundle_root_point_run_id": run_ids[0],
            "attempted": 10,
            "passed": 10,
            "failed": 0,
            "speedup_calculated": False,
            "performance_claim_eligible": False,
        },
    )
    bundle_run.write_json(
        "validation/admission-candidate.json",
        {
            "schema_version": (
                "kvbench-phase8-kivi-admission-candidate-1.0.0"
            ),
            "status": "LOCAL_CHECKS_PASS_PUBLICATION_PENDING",
            "durable_publication": "pending_host_side",
            "clean_retrieval": "pending_host_side",
            "g2_kivi": "NOT_EVALUATED",
            "performance_claim_eligible": False,
            "speedup_calculated": False,
            "r_hbm": None,
        },
    )
    source = bundle_run.finalize(bundle_manifest)
    validate_local_artifact(source, environ={})
    return source, manifests


def _receipt_payload(source: Path, root_sha256: str, object_count: int):
    uri = (
        "r2://kvbench-artifacts/kvbench/sha256/"
        f"{root_sha256}/"
    )
    return {
        "schema_version": (
            "kvbench-phase8-kivi-admission-r2-publication-1.0.0"
        ),
        "admission_status": "PASS",
        "artifact_status": "completed",
        "source_git_sha": SOURCE_GIT_SHA,
        "source_run_id": source.name,
        "local_validation": {
            "valid": True,
            "complete": True,
            "status": "completed",
            "root_sha256": root_sha256,
            "object_count": object_count,
            "complete_marker_valid": True,
            "inventory_valid": True,
            "checksum_ledger_valid": True,
            "root_digest_valid": True,
            "bundle_validation_valid": True,
        },
        "publication": {
            "result": "PASS",
            "root_sha256": root_sha256,
            "uri": uri,
            "object_count": object_count,
            "content_addressed": True,
            "conditional_writes": True,
            "complete_last": True,
        },
        "clean_retrieval": {
            "result": "PASS",
            "root_sha256": root_sha256,
            "object_count": object_count,
            "destination_initially_empty": True,
            "complete_marker_valid": True,
            "inventory_valid": True,
            "checksum_ledger_valid": True,
            "root_digest_valid": True,
            "bundle_validation_valid": True,
            "unexpected_objects": False,
        },
        "bucket_lock": {
            "provider": "cloudflare_r2",
            "bucket": "kvbench-artifacts",
            "endpoint": (
                "https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com"
            ),
            "endpoint_class": "cloudflare_r2_s3",
            "bucket_exists": True,
            "verification_result": "PASS",
            "enabled": True,
            "public_state_result": "PASS",
            "managed_r2_dev_enabled": False,
            "public_r2_dev": False,
            "custom_domain_count": 0,
            "enabled_custom_domain_count": 0,
            "public_custom_domain": False,
            "lock_rule_id": "phase8-test-indefinite-lock",
            "lock_rule_name": "phase8-test-indefinite-lock",
            "lock_scope": "exact",
            "covered_prefix": "kvbench/sha256/",
            "lock_prefix": "kvbench/sha256/",
            "retention_type": "Indefinite",
            "retention_condition": "Indefinite",
            "bucket_public": False,
            "verified_at_utc": "2026-07-27T01:59:00Z",
        },
        "credential_values_recorded": False,
        "env_file_read": False,
    }


def _outer_receipt_payload(
    artifact: Path,
    root_sha256: str,
    object_count: int,
) -> dict[str, object]:
    payload = _receipt_payload(artifact, root_sha256, object_count)
    payload.update(
        {
            "schema_version": (
                "kvbench-phase8-kivi-admission-r2-outer-publication-1.0.0"
            ),
            "source_git_sha": OUTER_GIT_SHA,
            "source_run_id": artifact.name,
            "performance_claim_eligible": False,
            "speedup_calculated": False,
            "r_hbm": None,
            "quality_execution": "LOCKED",
            "quality_benchmark_executed": False,
            "performance_data_frozen": False,
            "full_scan": "CLOSED",
            "global_g2": "NOT_EVALUATED",
            "phase9_started": False,
            "self_reference_control": {
                "included_in_bundle": False,
                "receipt_path": OUTER_RECEIPT_RELATIVE.as_posix(),
            },
        }
    )
    return payload


def _write_governance(
    repository: Path,
    source: Path,
    manifests: tuple,
) -> None:
    inner = validate_local_artifact(source, environ={})
    receipt_path = repository / INNER_RECEIPT_RELATIVE
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(
        json_bytes(
            _receipt_payload(
                source,
                inner.root_sha256,
                len(inner.files),
            )
        )
    )
    _write_method_report(repository, source, manifests)
    phase_report = repository / PASS_REPORT_RELATIVE
    phase_report.parent.mkdir(parents=True, exist_ok=True)
    phase_report.write_text(
        "PHASE 8 REPORT\n\nStatus: PASS\n",
        encoding="utf-8",
    )


def _write_method_report(
    repository: Path,
    source: Path,
    manifests: tuple,
) -> None:
    inner = validate_local_artifact(source, environ={})
    receipt_path = repository / INNER_RECEIPT_RELATIVE

    source_relative = source.relative_to(repository).as_posix()
    evidence_path = source / "validation" / "evidence.json"
    ledger_path = source / "checksums.sha256"
    references = (
        MethodAdmissionEvidenceReference(
            evidence_id="phase8_inner_evidence",
            path=f"{source_relative}/validation/evidence.json",
            sha256=sha256_file(evidence_path),
        ),
        MethodAdmissionEvidenceReference(
            evidence_id="phase8_inner_ledger",
            path=f"{source_relative}/checksums.sha256",
            sha256=sha256_file(ledger_path),
        ),
        MethodAdmissionEvidenceReference(
            evidence_id="phase8_inner_publication",
            path=INNER_RECEIPT_RELATIVE.as_posix(),
            sha256=sha256_file(receipt_path),
        ),
    )
    evidence_by_check = {
        check_id: (
            ("phase8_inner_publication",)
            if check_id in {"durable_publication", "clean_retrieval"}
            else (
                ("phase8_inner_ledger",)
                if check_id == "immutable_checksums"
                else ("phase8_inner_evidence",)
            )
        )
        for check_id in PHASE8_ADMISSION_CHECK_IDS
    }
    checks = tuple(
        Phase8AdmissionCheck(
            check_id=check_id,
            status=GateDisposition.PASS,
            summary=f"{check_id} passed",
            evidence_ids=evidence_by_check[check_id],
        )
        for check_id in PHASE8_ADMISSION_CHECK_IDS
    )
    method_fingerprints = {
        configuration: next(
            manifest.method_fingerprint
            for manifest in manifests
            if manifest.method_configuration == configuration
        )
        for configuration in (
            *PHASE8_MANDATORY_CONFIGS,
            PHASE8_HELD_OUT_CONFIG,
        )
    }
    cache_layout_fingerprints = {
        configuration: next(
            manifest.cache_layout_fingerprint
            for manifest in manifests
            if manifest.method_configuration == configuration
        )
        for configuration in (
            *PHASE8_MANDATORY_CONFIGS,
            PHASE8_HELD_OUT_CONFIG,
        )
    }
    identity = manifests[0]
    report = Phase8MethodAdmissionReport(
        schema_version=Phase8MethodAdmissionReport.SCHEMA_VERSION,
        created_at_utc="2026-07-27T02:00:00Z",
        status=GateDisposition.PASS,
        mandatory_configurations=PHASE8_MANDATORY_CONFIGS,
        held_out_configuration=PHASE8_HELD_OUT_CONFIG,
        admitted_configurations=PHASE8_MANDATORY_CONFIGS,
        method_fingerprints=method_fingerprints,
        cache_layout_fingerprints=cache_layout_fingerprints,
        adapter_version=identity.adapter_version,
        adapter_source_sha256=identity.adapter_source_sha256,
        official_base_commit=identity.official_base_commit,
        official_base_tree=identity.official_base_tree,
        patched_tree=identity.patched_tree,
        decision_0018_patch_sha256=(
            identity.decision_0018_patch_sha256
        ),
        extension_sha256=identity.extension_sha256,
        fixture_root_digest=identity.fixture_root_digest,
        authorized_container_digest=identity.authorized_container_digest,
        checks=checks,
        evidence_references=references,
        gates=Phase8AdmissionGates(
            g0=GateDisposition.PASS,
            g1=GateDisposition.PASS,
            g2_tq=GateDisposition.PASS,
            g2_kivi=GateDisposition.PASS,
            global_g2=GateDisposition.NOT_EVALUATED,
            g3=GateDisposition.NOT_EVALUATED,
            g4=GateDisposition.NOT_EVALUATED,
            g5=GateDisposition.NOT_EVALUATED,
            full_scan_state="CLOSED",
        ),
        blockers=(),
        local_root_digest=inner.root_sha256,
        r2_uri=(
            "r2://kvbench-artifacts/kvbench/sha256/"
            f"{inner.root_sha256}/"
        ),
        bucket_lock_identity="phase8-test-indefinite-lock",
        clean_retrieval=True,
        claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
        quality_status=QualityValidationState.UNVALIDATED,
        quality_execution=QualityExecutionState.LOCKED,
        performance_claim_eligible=False,
        performance_data_frozen=False,
        quality_benchmark_executed=False,
        speedup_calculated=False,
        r_hbm=None,
        measurement_scope=(
            MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
        ),
        creation_git_sha=SOURCE_GIT_SHA,
    )
    report_path = repository / METHOD_ADMISSION_RELATIVE
    report_path.write_bytes(json_bytes(report.to_dict()))
    checksum_path = repository / METHOD_ADMISSION_CHECKSUM_RELATIVE
    checksum_path.write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o644)
        elif path.is_dir():
            path.chmod(0o755)


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


class Phase8R2OuterBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repository = Path(self.temporary.name) / "repository"
        self.repository.mkdir()
        self.source, self.manifests = _make_inner_bundle(self.repository)
        _write_governance(
            self.repository,
            self.source,
            self.manifests,
        )
        self.strict_report = Phase8MethodAdmissionReport.from_dict(
            json.loads(
                (
                    self.repository / METHOD_ADMISSION_RELATIVE
                ).read_text(encoding="utf-8")
            )
        )
        self.output_root = (
            self.repository / "artifacts" / "phase8_r2_outer"
        )

    def tearDown(self) -> None:
        _make_writable(self.repository)
        self.temporary.cleanup()

    def _build(self, run_id: str = "phase8-r2-outer-test") -> Path:
        with mock.patch(
            "scripts.phase8_r2_outer_bundle."
            "build_phase8_method_admission_report",
            return_value=self.strict_report,
        ):
            final, validation = build_outer_bundle(
                repository_root=self.repository,
                source_bundle=self.source,
                output_root=self.output_root,
                run_id=run_id,
                source_git_sha=OUTER_GIT_SHA,
            )
        self.assertEqual(validation.run_id, run_id)
        self.assertEqual(validation.admission_run_count, 10)
        return final

    def _validate(self, final: Path):
        with mock.patch(
            "scripts.phase8_r2_outer_bundle."
            "build_phase8_method_admission_report",
            return_value=self.strict_report,
        ):
            return validate_outer_bundle(
                final,
                repository_root=self.repository,
                source_bundle=self.source,
            )

    def _validate_publication(self, final: Path):
        with mock.patch(
            "scripts.phase8_r2_outer_bundle."
            "build_phase8_method_admission_report",
            return_value=self.strict_report,
        ):
            return validate_outer_publication_receipt(
                final,
                receipt_path=self.repository / OUTER_RECEIPT_RELATIVE,
                repository_root=self.repository,
                source_bundle=self.source,
            )

    def _write_receipt_and_refresh_report(
        self,
        payload: dict[str, object],
    ) -> None:
        (self.repository / INNER_RECEIPT_RELATIVE).write_bytes(
            json_bytes(payload)
        )
        _write_method_report(
            self.repository,
            self.source,
            self.manifests,
        )

    def test_complete_inner_report_and_receipt_are_exactly_bound(self) -> None:
        outer_receipt = self.repository / OUTER_RECEIPT_RELATIVE
        outer_receipt.parent.mkdir(parents=True, exist_ok=True)
        outer_receipt.write_text(
            '{"not_part_of_outer_bundle":true}\n',
            encoding="utf-8",
        )
        inner = validate_local_artifact(self.source, environ={})
        inner_hashes = {
            item.relative_path: item.sha256 for item in inner.files
        }
        final = self._build()
        validation = self._validate(final)
        self.assertEqual(validation.inner_root_sha256, inner.root_sha256)
        self.assertEqual(
            validation.object_count,
            len(inner.files) + len(REQUIRED_REPOSITORY_FILES) + 2 + 4,
        )
        self.assertEqual(
            publication_order(validate_local_artifact(final))[-1].relative_path,
            "COMPLETE",
        )
        copy_prefix = (
            final / "original" / "sha256" / inner.root_sha256
        )
        for relative, digest in inner_hashes.items():
            self.assertEqual(
                hashlib.sha256(
                    (copy_prefix / relative).read_bytes()
                ).hexdigest(),
                digest,
            )
        for relative in REQUIRED_REPOSITORY_FILES:
            self.assertEqual(
                (final / relative).read_bytes(),
                (self.repository / relative).read_bytes(),
            )
        self.assertTrue((final / INNER_REFERENCE_PATH).is_file())
        self.assertTrue((final / ADMISSION_REFERENCES_PATH).is_file())
        self.assertFalse((final / OUTER_RECEIPT_RELATIVE).exists())
        self.assertEqual(
            {
                item.relative_path: item.sha256
                for item in validate_local_artifact(
                    self.source, environ={}
                ).files
            },
            inner_hashes,
        )

    def test_existing_final_is_never_replaced(self) -> None:
        final = self._build()
        root_before = validate_local_artifact(final).root_sha256
        with self.assertRaises(Phase8OuterBundleError):
            self._build()
        self.assertEqual(
            validate_local_artifact(final).root_sha256,
            root_before,
        )

    def test_report_root_mismatch_and_required_symlink_fail_closed(self) -> None:
        report_path = self.repository / METHOD_ADMISSION_RELATIVE
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["local_root_digest"] = "e" * 64
        report["r2_uri"] = (
            "r2://kvbench-artifacts/kvbench/sha256/"
            f"{'e' * 64}/"
        )
        report_path.write_bytes(json_bytes(report))
        checksum_path = (
            self.repository / METHOD_ADMISSION_CHECKSUM_RELATIVE
        )
        checksum_path.write_text(
            f"{sha256_file(report_path)}  {report_path.name}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            Phase8OuterBundleError,
            "does not join the inner root",
        ):
            self._build("phase8-r2-outer-wrong-report-root")

        _write_governance(
            self.repository,
            self.source,
            self.manifests,
        )
        phase_report = self.repository / PASS_REPORT_RELATIVE
        phase_report.unlink()
        phase_report.symlink_to(self.repository / METHOD_ADMISSION_RELATIVE)
        with self.assertRaisesRegex(
            Phase8OuterBundleError,
            "required repository evidence is unsafe",
        ):
            self._build("phase8-r2-outer-symlink")

    def test_report_evidence_must_remain_inside_the_outer_closure(self) -> None:
        report_path = self.repository / METHOD_ADMISSION_RELATIVE
        report = json.loads(report_path.read_text(encoding="utf-8"))
        phase_report = self.repository / PASS_REPORT_RELATIVE
        for reference in report["evidence_references"]:
            if reference["evidence_id"] == "phase8_inner_evidence":
                reference["path"] = PASS_REPORT_RELATIVE.as_posix()
                reference["sha256"] = sha256_file(phase_report)
                break
        else:
            self.fail("inner evidence reference is absent")
        report_path.write_bytes(json_bytes(report))
        checksum_path = (
            self.repository / METHOD_ADMISSION_CHECKSUM_RELATIVE
        )
        checksum_path.write_text(
            f"{sha256_file(report_path)}  {report_path.name}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            Phase8OuterBundleError,
            "outside the inner closure",
        ):
            self._build("phase8-r2-outer-evidence-escape")

    def test_schema_valid_forged_pass_report_is_rejected(self) -> None:
        report_path = self.repository / METHOD_ADMISSION_RELATIVE
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["checks"][0]["summary"] = "caller asserted PASS"
        report_path.write_bytes(json_bytes(report))
        checksum_path = (
            self.repository / METHOD_ADMISSION_CHECKSUM_RELATIVE
        )
        checksum_path.write_text(
            f"{sha256_file(report_path)}  {report_path.name}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            Phase8OuterBundleError,
            "not produced by strict derivation",
        ):
            self._build("phase8-r2-outer-forged-report")

    def test_external_outer_publication_receipt_is_strictly_bound(self) -> None:
        final = self._build("phase8-r2-outer-published")
        validation = self._validate(final)
        receipt_path = self.repository / OUTER_RECEIPT_RELATIVE
        receipt_path.write_bytes(
            json_bytes(
                _outer_receipt_payload(
                    final,
                    validation.root_sha256,
                    validation.object_count,
                )
            )
        )
        publication = self._validate_publication(final)
        self.assertEqual(publication.root_sha256, validation.root_sha256)
        self.assertEqual(publication.object_count, validation.object_count)
        self.assertFalse((final / OUTER_RECEIPT_RELATIVE).exists())

        for section, field, value in (
            ("publication", "root_sha256", "0" * 64),
            ("publication", "object_count", validation.object_count + 1),
            (
                "publication",
                "uri",
                "r2://kvbench-artifacts/kvbench/sha256/"
                f"{'0' * 64}/",
            ),
            ("publication", "complete_last", False),
            ("clean_retrieval", "destination_initially_empty", False),
            ("bucket_lock", "enabled", False),
            ("self_reference_control", "included_in_bundle", True),
        ):
            with self.subTest(section=section, field=field):
                payload = _outer_receipt_payload(
                    final,
                    validation.root_sha256,
                    validation.object_count,
                )
                payload[section][field] = value
                receipt_path.write_bytes(json_bytes(payload))
                with self.assertRaises(Phase8OuterBundleError):
                    self._validate_publication(final)

    def test_receipt_requires_content_addressed_conditional_complete_last(
        self,
    ) -> None:
        inner = validate_local_artifact(self.source, environ={})
        for field in (
            "content_addressed",
            "conditional_writes",
            "complete_last",
        ):
            with self.subTest(field=field):
                receipt = _receipt_payload(
                    self.source,
                    inner.root_sha256,
                    len(inner.files),
                )
                del receipt["publication"][field]
                self._write_receipt_and_refresh_report(receipt)
                with self.assertRaisesRegex(
                    Phase8OuterBundleError,
                    "receipt does not bind the bundle",
                ):
                    self._build(
                        f"phase8-r2-outer-missing-publication-{field}"
                    )

    def test_receipt_requires_empty_destination_and_all_validations(
        self,
    ) -> None:
        inner = validate_local_artifact(self.source, environ={})
        fields = (
            ("local_validation", "complete_marker_valid"),
            ("local_validation", "inventory_valid"),
            ("local_validation", "checksum_ledger_valid"),
            ("local_validation", "root_digest_valid"),
            ("local_validation", "bundle_validation_valid"),
            ("clean_retrieval", "destination_initially_empty"),
            ("clean_retrieval", "complete_marker_valid"),
            ("clean_retrieval", "inventory_valid"),
            ("clean_retrieval", "checksum_ledger_valid"),
            ("clean_retrieval", "root_digest_valid"),
            ("clean_retrieval", "bundle_validation_valid"),
        )
        for section, field in fields:
            with self.subTest(section=section, field=field):
                receipt = _receipt_payload(
                    self.source,
                    inner.root_sha256,
                    len(inner.files),
                )
                del receipt[section][field]
                self._write_receipt_and_refresh_report(receipt)
                with self.assertRaisesRegex(
                    Phase8OuterBundleError,
                    "receipt does not bind the bundle",
                ):
                    self._build(
                        f"phase8-r2-outer-missing-{section}-{field}"
                    )

    def test_receipt_requires_full_bucket_lock_identity(self) -> None:
        inner = validate_local_artifact(self.source, environ={})
        required_lock_fields = tuple(
            _receipt_payload(
                self.source,
                inner.root_sha256,
                len(inner.files),
            )["bucket_lock"]
        )
        for field in required_lock_fields:
            with self.subTest(field=field):
                receipt = _receipt_payload(
                    self.source,
                    inner.root_sha256,
                    len(inner.files),
                )
                del receipt["bucket_lock"][field]
                self._write_receipt_and_refresh_report(receipt)
                with self.assertRaisesRegex(
                    Phase8OuterBundleError,
                    "receipt does not bind the bundle",
                ):
                    self._build(
                        f"phase8-r2-outer-missing-lock-{field}"
                    )

    def test_inner_receipt_cannot_claim_method_report_validation(self) -> None:
        inner = validate_local_artifact(self.source, environ={})
        for section in (
            None,
            "local_validation",
            "publication",
            "clean_retrieval",
            "bucket_lock",
        ):
            for forbidden in (
                "retrieved_" + "report_valid",
                "method_admission_" + "report_valid",
            ):
                with self.subTest(section=section, forbidden=forbidden):
                    receipt = _receipt_payload(
                        self.source,
                        inner.root_sha256,
                        len(inner.files),
                    )
                    target = receipt if section is None else receipt[section]
                    target[forbidden] = True
                    self._write_receipt_and_refresh_report(receipt)
                    with self.assertRaisesRegex(
                        Phase8OuterBundleError,
                        "cannot validate the MethodAdmissionReport",
                    ):
                        self._build(
                            "phase8-r2-outer-forbidden-"
                            f"{section or 'root'}-{forbidden}"
                        )

    def test_bucket_lock_integer_and_identifier_types_are_exact(self) -> None:
        inner = validate_local_artifact(self.source, environ={})
        invalid_fields = (
            ("custom_domain_count", False),
            ("enabled_custom_domain_count", False),
            ("lock_rule_id", " "),
        )
        for field, value in invalid_fields:
            with self.subTest(field=field):
                receipt = _receipt_payload(
                    self.source,
                    inner.root_sha256,
                    len(inner.files),
                )
                receipt["bucket_lock"][field] = value
                self._write_receipt_and_refresh_report(receipt)
                with self.assertRaisesRegex(
                    Phase8OuterBundleError,
                    "receipt does not bind the bundle",
                ):
                    self._build(
                        f"phase8-r2-outer-invalid-lock-{field}"
                    )

    def test_clean_retrieval_validates_and_tamper_fails(self) -> None:
        final = self._build()
        retrieved = self.repository / "retrieved-empty-destination"
        shutil.copytree(final, retrieved, copy_function=shutil.copy2)
        self.assertEqual(
            self._validate(retrieved).root_sha256,
            self._validate(final).root_sha256,
        )
        _make_writable(retrieved)
        report = retrieved / METHOD_ADMISSION_RELATIVE
        report.write_bytes(report.read_bytes() + b"tamper\n")
        _make_immutable(retrieved)
        with self.assertRaises(RuntimeError):
            self._validate(retrieved)


if __name__ == "__main__":
    unittest.main()

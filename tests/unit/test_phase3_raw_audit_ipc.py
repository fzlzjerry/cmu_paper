"""Adversarial CPU tests for the Phase 3 raw-audit IPC boundary."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from kvbench.runtime import phase3_coordinator, phase3_worker
from kvbench.runtime.gqa_device_dispatch import (
    REQUIRED_SUT_SOURCES,
    phase3_source_identity_sha256,
)
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.runtime.phase3_coordinator import (
    MAX_IPC_BYTES,
    Phase3ExecutionSourcePin,
    Phase3CoordinatorError,
    WORKER_EVIDENCE_V1,
    _cache_identity,
    _expected_phase3_raw_audit_operations,
    _ingest_worker_evidence_v2,
    _open_private_ipc_parent,
    _open_private_raw_audit_root,
    _parse_canonical_json,
    _pin_phase3_execution_sources,
    _preserve_phase3_worker_channel_artifacts,
    _read_pinned_ipc_file,
    _raw_audit_ingestion_outcome,
    _revalidate_phase3_execution_sources,
    _resolved_phase3_worker_result,
    _resolve_phase3_terminal_status,
    _validate_cache_source_join,
    _validate_worker_evidence_v1,
)
from kvbench.runtime.phase3_raw_audit_evidence import (
    PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
    PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND,
    RAW_AUDIT_STATUS_COMPLETED,
    RAW_AUDIT_STATUS_FAILED,
    RAW_AUDIT_STATUS_NOT_ATTEMPTED,
    REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS,
    Phase3RawAuditFile,
    Phase3RawAuditOperationRecord,
    Phase3RawAuditRunIndex,
)
from kvbench.runtime.phase3_worker import (
    _publish_phase3_worker_evidence_channels,
    build_phase3_worker_evidence_v2,
)
from kvbench.runtime.process_supervision import HandshakeStage
from kvbench.runtime.phase3_worker_channels import (
    PHASE3_RAW_AUDIT_OPERATION_PLAN_ENV,
    Phase3RawAuditProducerError,
    Phase3RawAuditProducerRegistry,
    build_phase3_raw_audit_operation_plan_bytes,
    build_phase3_worker_channel_commitment,
    parse_phase3_raw_audit_operation_plan_bytes,
    phase3_worker_channel_commitment_sha256,
    require_phase3_raw_audit_measurement_admission,
)
from kvbench.schema import (
    GraphMode,
    Phase3WorkerResult,
    RunStatus,
    RunnerKind,
    canonical_json_bytes,
    sha256_hex,
)
from kvbench.schema.phase3 import (
    BF16BackendIdentity,
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GROWING_PLAN_PATH,
    PHASE3_PLAN_FINGERPRINTS,
    Phase3ProcessPoint,
)
from tests.unit.test_phase3_gqa_device_dispatch_geometry import (
    allocation_raw_evidence,
    paired_allocator_raw_evidence,
    raw_replay_fixture,
)


RUN_ID = "phase3-raw-ipc-fixture"
POINT_ID = "fixed_l-b1-l128-eager-r1"
RAW_PAYLOAD = b"RAW-BYTES-MUST-NOT-CROSS-IPC-20260723"


class CapturingRun:
    def __init__(self) -> None:
        self.writes: dict[str, bytes] = {}

    def write_bytes(self, relative: str, payload: bytes) -> None:
        if relative in self.writes:
            raise AssertionError("test artifact path was replaced")
        self.writes[relative] = bytes(payload)


def _point() -> Phase3ProcessPoint:
    return Phase3ProcessPoint(
        point_id=POINT_ID,
        runner_kind=RunnerKind.FIXED_L,
        graph_mode=GraphMode.EAGER,
        batch_size=1,
        context_length=128,
        output_steps=1,
        process_replicate=1,
        stability_member=False,
    )


def _growing_operations(
    *,
    run_id: str = RUN_ID,
) -> tuple[Phase3AuditOperationKey, ...]:
    point = Phase3ProcessPoint(
        point_id="growing_context-b1-l128-eager-r1",
        runner_kind=RunnerKind.GROWING_CONTEXT,
        graph_mode=GraphMode.EAGER,
        batch_size=1,
        context_length=128,
        output_steps=16,
        process_replicate=1,
        stability_member=False,
    )
    return tuple(
        Phase3AuditOperationKey.from_point(
            run_id=run_id,
            point=point,
            decode_step=decode_step,
            cache_layout_fingerprint="1" * 64,
            execution_git_sha="2" * 40,
            plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[
                PHASE3_GROWING_PLAN_PATH
            ],
            hardware_identity_sha256="3" * 64,
            software_identity_sha256="4" * 64,
            model_identity_sha256="5" * 64,
            backend_identity_sha256="6" * 64,
            source_identity_sha256="7" * 64,
        )
        for decode_step in range(16)
    )


def _index(
    *,
    run_id: str = RUN_ID,
    payload: bytes = RAW_PAYLOAD,
) -> Phase3RawAuditRunIndex:
    operation = Phase3AuditOperationKey.from_point(
        run_id=run_id,
        point=_point(),
        decode_step=0,
        cache_layout_fingerprint="1" * 64,
        execution_git_sha="2" * 40,
        plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[PHASE3_FIXED_PLAN_PATH],
        hardware_identity_sha256="3" * 64,
        software_identity_sha256="4" * 64,
        model_identity_sha256="5" * 64,
        backend_identity_sha256="6" * 64,
        source_identity_sha256="7" * 64,
    )
    declaration = Phase3RawAuditFile.from_bytes(
        path="step-0000/partial.bin",
        kind="partial",
        payload=payload,
    )
    record = Phase3RawAuditOperationRecord(
        schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
        operation=operation,
        status=RAW_AUDIT_STATUS_FAILED,
        failure_reason="collector_failed",
        files=(declaration,),
    )
    return Phase3RawAuditRunIndex.create((record,))


def _write_declared(root: Path, payload: bytes = RAW_PAYLOAD) -> None:
    directory = root / "step-0000"
    directory.mkdir(mode=0o700)
    (directory / "partial.bin").write_bytes(payload)




def _completed_index(*, run_id: str = RUN_ID) -> Phase3RawAuditRunIndex:
    failed = _index(run_id=run_id)
    operation = failed.records[0].operation
    kinds = tuple(
        sorted(
            (
                *REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS,
                PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND,
            )
        )
    )
    files = tuple(
        Phase3RawAuditFile.from_bytes(
            path=f"step-0000/{kind}.bin",
            kind=kind,
            payload=f"{kind}-payload".encode("ascii"),
        )
        for kind in kinds
    )
    record = Phase3RawAuditOperationRecord(
        schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
        operation=operation,
        status=RAW_AUDIT_STATUS_COMPLETED,
        failure_reason=None,
        files=files,
    )
    return Phase3RawAuditRunIndex.create((record,))


def _completed_record(
    operation: Phase3AuditOperationKey,
    raw_root: Path,
) -> Phase3RawAuditOperationRecord:
    directory_name = f"step-{operation.decode_step:04d}"
    directory = raw_root / directory_name
    directory.mkdir(mode=0o700)
    declarations: list[Phase3RawAuditFile] = []
    kinds = REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS
    if operation.decode_step == 0:
        kinds = tuple(
            sorted((*kinds, PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND))
        )
    for kind in kinds:
        payload = f"{kind}-payload".encode("ascii")
        path = f"{directory_name}/{kind}.bin"
        (raw_root / path).write_bytes(payload)
        declarations.append(
            Phase3RawAuditFile.from_bytes(
                path=path,
                kind=kind,
                payload=payload,
            )
        )
    return Phase3RawAuditOperationRecord(
        schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
        operation=operation,
        status=RAW_AUDIT_STATUS_COMPLETED,
        failure_reason=None,
        files=tuple(declarations),
    )


def _write_index_files(root: Path, index: Phase3RawAuditRunIndex) -> None:
    for record in index.records:
        for declaration in record.files:
            target = root / declaration.path
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            payload = (
                RAW_PAYLOAD
                if declaration.kind == "partial"
                else f"{declaration.kind}-payload".encode("ascii")
            )
            target.write_bytes(payload)


def _ingest(
    *,
    root_fd: int,
    owner_uid: int,
    index: Phase3RawAuditRunIndex,
    run: CapturingRun,
    outcome: dict[str, object],
    expected_run_id: str = RUN_ID,
) -> None:
    _ingest_worker_evidence_v2(
        evidence=build_phase3_worker_evidence_v2(index),
        expected_run_id=expected_run_id,
        expected_point_id=POINT_ID,
        raw_audit_root_fd=root_fd,
        raw_audit_owner_uid=owner_uid,
        expected_operations=tuple(
            record.operation for record in index.records
        ),
        run=run,
        outcome=outcome,
    )


class Phase3RawAuditProducerRegistryTests(unittest.TestCase):
    def test_failure_reason_preserves_bounded_producer_exception(self) -> None:
        reason = phase3_worker._raw_audit_failure_reason(
            phase3_worker.WorkerProtocolError(
                "paired allocator controls did not verify"
            ),
            prefix="operation_audit_failed",
        )
        self.assertEqual(
            reason,
            "operation_audit_failed:workerprotocolerror:"
            "paired.allocator.controls.did.not.verify",
        )
        self.assertLessEqual(len(reason), 256)

    def test_operation_plan_round_trip_and_canonical_rejection(self) -> None:
        operations = tuple(
            record.operation for record in _completed_index().records
        )
        payload = build_phase3_raw_audit_operation_plan_bytes(operations)
        self.assertEqual(
            parse_phase3_raw_audit_operation_plan_bytes(payload),
            operations,
        )
        duplicate = payload.replace(
            b'"operations":',
            b'"operations":[],"operations":',
            1,
        )
        noncanonical = json.dumps(
            json.loads(payload),
            indent=2,
        ).encode("utf-8")
        for malformed in (duplicate, noncanonical):
            with self.subTest(payload=malformed[:48]):
                with self.assertRaises(Phase3RawAuditProducerError):
                    parse_phase3_raw_audit_operation_plan_bytes(malformed)

    def test_fingerprinted_environment_and_worker_plan_join(self) -> None:
        operations = tuple(
            record.operation for record in _completed_index().records
        )
        plan = build_phase3_raw_audit_operation_plan_bytes(operations)
        with tempfile.TemporaryDirectory() as directory:
            environment = phase3_coordinator._worker_environment(
                Path(directory),
                raw_audit_operations=operations,
            )
            with self.assertRaisesRegex(ValueError, "nonempty"):
                phase3_coordinator._worker_environment(
                    Path(directory),
                    raw_audit_operations=(),
                )
        self.assertEqual(
            environment[PHASE3_RAW_AUDIT_OPERATION_PLAN_ENV],
            plan.decode("utf-8"),
        )
        with (
            mock.patch.dict(
                os.environ,
                {
                    PHASE3_RAW_AUDIT_OPERATION_PLAN_ENV: plan.decode("utf-8")
                },
            ),
            mock.patch.object(
                phase3_worker,
                "_ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS",
                None,
            ),
        ):
            self.assertEqual(
                phase3_worker._initialize_phase3_raw_audit_operation_plan(
                    run_id=RUN_ID,
                    point_id=POINT_ID,
                ),
                operations,
            )

    def test_incomplete_registration_invokes_no_producer(self) -> None:
        operations = _growing_operations()
        observed: list[int] = []

        def producer(
            operation: Phase3AuditOperationKey,
            _: Path,
        ) -> Phase3RawAuditOperationRecord:
            observed.append(operation.decode_step)
            raise AssertionError("incomplete registry invoked a producer")

        registry = Phase3RawAuditProducerRegistry(operations)
        for operation in operations[:-1]:
            registry.register(operation, producer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            with self.assertRaisesRegex(
                Phase3RawAuditProducerError,
                "incomplete before measurement",
            ):
                registry.collect(root)
        self.assertEqual(observed, [])

    def test_completed_producer_index_is_registered_before_measurement(
        self,
    ) -> None:
        operations = tuple(
            record.operation for record in _completed_index().records
        )
        ordering: list[str] = []

        def producer(
            operation: Phase3AuditOperationKey,
            raw_root: Path,
        ) -> Phase3RawAuditOperationRecord:
            ordering.append("producer_completed")
            return _completed_record(operation, raw_root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            with (
                mock.patch.object(phase3_worker, "_ACTIVE_RUN_ID", RUN_ID),
                mock.patch.object(
                    phase3_worker,
                    "_ACTIVE_RAW_AUDIT_RUN_INDEX",
                    None,
                ),
                mock.patch.object(
                    phase3_worker,
                    "_ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS",
                    operations,
                ),
                mock.patch.object(
                    phase3_worker,
                    "_ACTIVE_WORKER_STAGES",
                    [
                        HandshakeStage.WORKER_STARTED,
                        HandshakeStage.CUDA_CONTEXT_CREATED,
                    ],
                ),
            ):
                index = phase3_worker._collect_and_register_phase3_raw_audits(
                    expected_operations=operations,
                    raw_audit_root=root,
                    producer_bindings=((operations[0], producer),),
                )
                ordering.append("measurement_started")
                self.assertIs(
                    phase3_worker._ACTIVE_RAW_AUDIT_RUN_INDEX,
                    index,
                )
                require_phase3_raw_audit_measurement_admission(
                    index,
                    operations,
                )
        self.assertEqual(
            ordering,
            ["producer_completed", "measurement_started"],
        )

    def test_failed_producer_preserves_unattempted_tail_and_no_retry(
        self,
    ) -> None:
        operations = _growing_operations()
        observed: list[int] = []

        def producer(
            operation: Phase3AuditOperationKey,
            _: Path,
        ) -> Phase3RawAuditOperationRecord:
            observed.append(operation.decode_step)
            if operation.decode_step != 0:
                raise AssertionError("producer ran after an earlier failure")
            return Phase3RawAuditOperationRecord(
                schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
                operation=operation,
                status=RAW_AUDIT_STATUS_FAILED,
                failure_reason="collector_failed",
                files=(),
            )

        registry = Phase3RawAuditProducerRegistry(operations)
        for operation in operations:
            registry.register(operation, producer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            index = registry.collect(root)
            with self.assertRaisesRegex(
                Phase3RawAuditProducerError,
                "cannot be retried",
            ):
                registry.collect(root)
        self.assertEqual(observed, [0])
        self.assertEqual(index.records[0].status, RAW_AUDIT_STATUS_FAILED)
        self.assertTrue(
            all(
                record.status == RAW_AUDIT_STATUS_NOT_ATTEMPTED
                for record in index.records[1:]
            )
        )
        with self.assertRaisesRegex(
            Phase3RawAuditProducerError,
            "decode step 0: collector_failed",
        ):
            require_phase3_raw_audit_measurement_admission(
                index,
                operations,
            )



class Phase3RawAuditIPCCompatibilityTests(unittest.TestCase):
    def test_legacy_v1_validation_is_nonmutating_and_unchanged(self) -> None:
        worker_result = {"status": "completed", "output_checksum": "abc"}
        evidence = {
            "schema_version": WORKER_EVIDENCE_V1,
            "run_id": RUN_ID,
            "point_id": POINT_ID,
            "worker_result": worker_result,
            "runtime": {"cache_layout_fingerprint": "8" * 64},
            "legacy_field": {"retained": True},
        }
        before = copy.deepcopy(evidence)
        result = SimpleNamespace(to_dict=lambda: dict(worker_result))
        _validate_worker_evidence_v1(
            evidence=evidence,
            expected_run_id=RUN_ID,
            expected_point_id=POINT_ID,
            result=result,
            cache_layout_fingerprint="8" * 64,
        )
        self.assertEqual(evidence, before)
        wrong = copy.deepcopy(evidence)
        wrong["point_id"] = "fixed_l-b4-l128-eager-r1"
        with self.assertRaisesRegex(
            Phase3CoordinatorError, "identity join failed"
        ):
            _validate_worker_evidence_v1(
                evidence=wrong,
                expected_run_id=RUN_ID,
                expected_point_id=POINT_ID,
                result=result,
                cache_layout_fingerprint="8" * 64,
            )

    def test_raw_index_registration_is_single_use_and_resettable(self) -> None:
        index = _index()
        operations = tuple(record.operation for record in index.records)
        pre_measurement_stages = [
            HandshakeStage.WORKER_STARTED,
            HandshakeStage.CUDA_CONTEXT_CREATED,
        ]
        with (
            mock.patch.object(phase3_worker, "_ACTIVE_RUN_ID", RUN_ID),
            mock.patch.object(
                phase3_worker, "_ACTIVE_RAW_AUDIT_RUN_INDEX", None
            ),
            mock.patch.object(
                phase3_worker,
                "_ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS",
                operations,
            ),
            mock.patch.object(
                phase3_worker,
                "_ACTIVE_WORKER_STAGES",
                pre_measurement_stages,
            ),
        ):
            phase3_worker.register_phase3_raw_audit_run_index(index)
            self.assertIs(phase3_worker._ACTIVE_RAW_AUDIT_RUN_INDEX, index)
            with self.assertRaisesRegex(
                phase3_worker.WorkerProtocolError, "already registered"
            ):
                phase3_worker.register_phase3_raw_audit_run_index(index)
            phase3_worker._reset_phase3_raw_audit_run_index()
            self.assertIsNone(phase3_worker._ACTIVE_RAW_AUDIT_RUN_INDEX)
            phase3_worker.register_phase3_raw_audit_run_index(index)
            self.assertIs(phase3_worker._ACTIVE_RAW_AUDIT_RUN_INDEX, index)

    def test_raw_index_registration_after_measurement_is_rejected(self) -> None:
        index = _index()
        operations = tuple(record.operation for record in index.records)
        with (
            mock.patch.object(phase3_worker, "_ACTIVE_RUN_ID", RUN_ID),
            mock.patch.object(
                phase3_worker, "_ACTIVE_RAW_AUDIT_RUN_INDEX", None
            ),
            mock.patch.object(
                phase3_worker,
                "_ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS",
                operations,
            ),
            mock.patch.object(
                phase3_worker,
                "_ACTIVE_WORKER_STAGES",
                [
                    HandshakeStage.WORKER_STARTED,
                    HandshakeStage.CUDA_CONTEXT_CREATED,
                    HandshakeStage.MEASUREMENT_STARTED,
                ],
            ),
        ):
            with self.assertRaisesRegex(
                phase3_worker.WorkerProtocolError,
                "before measurement",
            ):
                phase3_worker.register_phase3_raw_audit_run_index(index)

    def test_v2_ipc_is_minimal_and_contains_no_raw_evidence_bytes(self) -> None:
        index = _index()
        envelope = build_phase3_worker_evidence_v2(index)
        self.assertEqual(
            set(envelope),
            {
                "schema_version",
                "raw_audit_run_index",
                "raw_audit_run_index_sha256",
            },
        )
        index_bytes = canonical_json_bytes(index.to_dict())
        self.assertEqual(
            envelope["raw_audit_run_index_sha256"],
            sha256_hex(index_bytes),
        )
        ipc_bytes = canonical_json_bytes(envelope) + b"\n"
        self.assertNotIn(RAW_PAYLOAD, ipc_bytes)
        self.assertEqual(
            _parse_canonical_json(
                ipc_bytes,
                maximum_bytes=MAX_IPC_BYTES,
                label="worker evidence",
            ),
            envelope,
        )

    def test_duplicate_or_noncanonical_nested_index_is_rejected_at_ipc_parse(
        self,
    ) -> None:
        index_text = canonical_json_bytes(_index().to_dict()).decode("utf-8")
        duplicate_index = index_text.replace(
            '"point_id":',
            '"point_id":"duplicate","point_id":',
            1,
        )
        duplicate = (
            '{"raw_audit_run_index":'
            + duplicate_index
            + ',"raw_audit_run_index_sha256":"'
            + "0" * 64
            + '","schema_version":"kvbench-phase3-worker-evidence-2.0.0"}\n'
        ).encode("utf-8")
        noncanonical = (
            json.dumps(
                build_phase3_worker_evidence_v2(_index()),
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            + b"\n"
        )
        for payload in (duplicate, noncanonical):
            with self.subTest(payload=payload[:48]):
                with self.assertRaises(Phase3CoordinatorError):
                    _parse_canonical_json(
                        payload,
                        maximum_bytes=MAX_IPC_BYTES,
                        label="worker evidence",
                    )


class Phase3RawAuditIPCSecureIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-raw-ipc-test-"
        )
        self.base = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _empty_root(self, name: str = "raw") -> tuple[Path, int, int]:
        root = self.base / name
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        descriptor, owner_uid = _open_private_raw_audit_root(root)
        return root, descriptor, owner_uid

    def test_v2_success_secure_ingests_once_and_copies_exact_bytes(self) -> None:
        root, descriptor, owner_uid = self._empty_root()
        try:
            _write_declared(root)
            index = _index()
            run = CapturingRun()
            outcome = _raw_audit_ingestion_outcome()
            with mock.patch.object(
                phase3_coordinator,
                "ingest_phase3_raw_audit_evidence_fd",
                wraps=phase3_coordinator.ingest_phase3_raw_audit_evidence_fd,
            ) as ingest:
                _ingest(
                    root_fd=descriptor,
                    owner_uid=owner_uid,
                    index=index,
                    run=run,
                    outcome=outcome,
                )
            ingest.assert_called_once()
        finally:
            os.close(descriptor)
        self.assertEqual(
            run.writes["raw/audits/files/step-0000/partial.bin"],
            RAW_PAYLOAD,
        )
        self.assertEqual(
            run.writes["raw/audits/index.json"],
            canonical_json_bytes(index.to_dict()),
        )
        self.assertIs(outcome["passed"], False)
        self.assertIs(outcome["ingestion_passed"], True)
        self.assertIs(outcome["collection_validation_passed"], False)
        self.assertIs(outcome["collection_completion_passed"], False)
        self.assertIs(outcome["declaration_completion_observed"], False)
        self.assertIs(outcome["semantic_validation_pending"], False)
        self.assertIs(outcome["commitment_validation_passed"], False)
        self.assertIs(outcome["scientific_completion_passed"], False)
        self.assertIs(outcome["terminal_eligible"], False)
        self.assertEqual(outcome["status"], "ingested_failed_evidence")
        self.assertEqual(outcome["artifact_file_count"], 1)

    def test_sha_and_identity_mismatches_fail_before_source_ingestion(self) -> None:
        root, descriptor, owner_uid = self._empty_root()
        try:
            index = _index()
            envelope = build_phase3_worker_evidence_v2(index)
            envelope["raw_audit_run_index_sha256"] = "0" * 64
            with mock.patch.object(
                phase3_coordinator,
                "ingest_phase3_raw_audit_evidence_fd",
            ) as ingest:
                with self.assertRaisesRegex(
                    Phase3CoordinatorError, "SHA-256 differs"
                ):
                    _ingest_worker_evidence_v2(
                        evidence=envelope,
                        expected_run_id=RUN_ID,
                        expected_point_id=POINT_ID,
                        raw_audit_root_fd=descriptor,
                        raw_audit_owner_uid=owner_uid,
                        expected_operations=tuple(
                            record.operation for record in _index().records
                        ),
                        run=CapturingRun(),
                        outcome=_raw_audit_ingestion_outcome(),
                    )
                ingest.assert_not_called()
            with mock.patch.object(
                phase3_coordinator,
                "ingest_phase3_raw_audit_evidence_fd",
            ) as ingest:
                with self.assertRaisesRegex(
                    Phase3CoordinatorError, "identity join failed"
                ):
                    _ingest(
                        root_fd=descriptor,
                        owner_uid=owner_uid,
                        index=index,
                        run=CapturingRun(),
                        outcome=_raw_audit_ingestion_outcome(),
                        expected_run_id="phase3-different-run",
                    )
                ingest.assert_not_called()
        finally:
            os.close(descriptor)

    def test_pinned_descriptor_survives_root_path_replacement(self) -> None:
        root, descriptor, owner_uid = self._empty_root()
        try:
            _write_declared(root)
            moved = self.base / "original-raw-root"
            root.rename(moved)
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            (root / "attacker.bin").write_bytes(b"replacement-path-bytes")
            run = CapturingRun()
            outcome = _raw_audit_ingestion_outcome()
            _ingest(
                root_fd=descriptor,
                owner_uid=owner_uid,
                index=_index(),
                run=run,
                outcome=outcome,
            )
        finally:
            os.close(descriptor)
        self.assertEqual(
            run.writes["raw/audits/files/step-0000/partial.bin"],
            RAW_PAYLOAD,
        )
        self.assertIs(outcome["source_root_path_reopened_after_spawn"], False)

    def test_tamper_and_undeclared_files_fail_closed(self) -> None:
        for case in ("tamper", "undeclared"):
            with self.subTest(case=case):
                root, descriptor, owner_uid = self._empty_root(case)
                try:
                    _write_declared(
                        root,
                        b"X" * len(RAW_PAYLOAD) if case == "tamper" else RAW_PAYLOAD,
                    )
                    if case == "undeclared":
                        (root / "undeclared.bin").write_bytes(b"forbidden")
                    with self.assertRaisesRegex(
                        Phase3CoordinatorError,
                        "failed secure ingestion",
                    ):
                        _ingest(
                            root_fd=descriptor,
                            owner_uid=owner_uid,
                            index=_index(),
                            run=CapturingRun(),
                            outcome=_raw_audit_ingestion_outcome(),
                        )
                finally:
                    os.close(descriptor)

    def test_v2_rejects_any_extra_ipc_field_before_ingestion(self) -> None:
        root, descriptor, owner_uid = self._empty_root()
        try:
            envelope = build_phase3_worker_evidence_v2(_index())
            envelope["raw_evidence_bytes"] = RAW_PAYLOAD.decode("ascii")
            with mock.patch.object(
                phase3_coordinator,
                "ingest_phase3_raw_audit_evidence_fd",
            ) as ingest:
                with self.assertRaisesRegex(
                    Phase3CoordinatorError, "outside the raw-index envelope"
                ):
                    _ingest_worker_evidence_v2(
                        evidence=envelope,
                        expected_run_id=RUN_ID,
                        expected_point_id=POINT_ID,
                        raw_audit_root_fd=descriptor,
                        raw_audit_owner_uid=owner_uid,
                        expected_operations=tuple(
                            record.operation for record in _index().records
                        ),
                        run=CapturingRun(),
                        outcome=_raw_audit_ingestion_outcome(),
                    )
                ingest.assert_not_called()
        finally:
            os.close(descriptor)


class Phase3CoordinatorSemanticReplayTests(unittest.TestCase):
    def test_coordinator_rederives_verified_b011_b012_from_reduced_raw_files(
        self,
    ) -> None:
        fixture = raw_replay_fixture(production_allocation_identity=True)
        operation = fixture["operation_key"]
        self.assertIsInstance(operation, Phase3AuditOperationKey)
        gqa_allocator, mha_allocator, _ = paired_allocator_raw_evidence(
            fixture
        )
        _, allocation = allocation_raw_evidence(fixture)
        witness = json.loads(
            allocation.operation_witness_raw.decode("utf-8")
        )
        measured_before = witness["measured_before"]
        measured_after = witness["measured_after"]
        measured_output = witness["measured_output"]
        prefix_sha256 = measured_before["historical_prefix_sha256"]
        history_chain_sha256 = hashlib.sha256(
            (
                f"{prefix_sha256}:"
                f"{measured_after['destination_slot_sha256']}"
            ).encode("ascii")
        ).hexdigest()

        bundle = canonical_json_bytes(
            {
                "snapshot": json.loads(
                    allocation.snapshot_raw.decode("utf-8")
                ),
                "memory_stats_before": json.loads(
                    allocation.memory_stats_before_raw.decode("utf-8")
                ),
                "memory_stats_after": json.loads(
                    allocation.memory_stats_after_raw.decode("utf-8")
                ),
                "memory_accounting_before": json.loads(
                    allocation.memory_accounting_before_raw.decode("utf-8")
                ),
                "memory_accounting_after": json.loads(
                    allocation.memory_accounting_after_raw.decode("utf-8")
                ),
                "operation_witness": witness,
                "gqa_allocator_control": json.loads(
                    gqa_allocator.decode("utf-8")
                ),
                "mha_allocator_control": json.loads(
                    mha_allocator.decode("utf-8")
                ),
                "audit_sha256_ledger": (
                    allocation.audit_sha256_ledger_raw.decode("ascii")
                ),
            }
        )
        b011_raw = fixture["observation_raw"]
        gqa_raw = fixture["gqa_raw"]
        mha_raw = fixture["mha_raw"]
        self.assertIsInstance(b011_raw, bytes)
        self.assertIsInstance(gqa_raw, bytes)
        self.assertIsInstance(mha_raw, bytes)
        dispatch_sha256 = sha256_hex(b011_raw)
        allocation_sha256 = sha256_hex(allocation.audit_raw)
        provenance = canonical_json_bytes(
            {
                "schema_version": (
                    "kvbench-phase3-endpoint-session-1.0.0"
                ),
                "receipt_sha256": "a" * 64,
                "cache_pointers": {
                    "keys_data_ptr": measured_before["key_data_ptr"],
                    "values_data_ptr": measured_before["value_data_ptr"],
                    "keys_storage_ptr": measured_before["key_data_ptr"],
                    "values_storage_ptr": measured_before["value_data_ptr"],
                },
                "cache_layout_fingerprint": (
                    operation.cache_layout_fingerprint
                ),
                "operation_fingerprints": [
                    operation.operation_fingerprint_sha256
                ],
                "dispatch_audit_sha256": [dispatch_sha256],
                "allocation_audit_sha256": [allocation_sha256],
                "audit_output_sha256": [measured_output["sha256"]],
                "audit_output_finite": [measured_output["finite"]],
                "graph_retained": False,
                "prefix_sha256": prefix_sha256,
                "history_chain_sha256": history_chain_sha256,
            }
        )
        payloads = {
            "b011_audit": ("dispatch-audit.json", b011_raw),
            "b011_gqa_chrome_trace": (
                "gqa.geometry.chrome.json",
                gqa_raw,
            ),
            "b011_mha_chrome_trace": (
                "mha.geometry.chrome.json",
                mha_raw,
            ),
            "b012_allocation_audit": (
                "allocation-audit.json",
                allocation.audit_raw,
            ),
            "b012_allocator_snapshot": (
                "allocator-evidence.json",
                bundle,
            ),
            "b012_allocator_trace": (
                "allocator-trace.json",
                allocation.trace_raw,
            ),
            PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND: (
                "session-provenance.json",
                provenance,
            ),
        }

        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-semantic-replay-"
        ) as directory:
            root = Path(directory)
            root.chmod(0o700)
            descriptor, owner_uid = _open_private_raw_audit_root(root)
            step = root / "step-0000"
            step.mkdir(mode=0o700)
            declarations: list[Phase3RawAuditFile] = []
            for kind, (filename, payload) in sorted(payloads.items()):
                target = step / filename
                target.write_bytes(payload)
                declarations.append(
                    Phase3RawAuditFile.from_bytes(
                        path=f"step-0000/{filename}",
                        kind=kind,
                        payload=payload,
                    )
                )
            record = Phase3RawAuditOperationRecord(
                schema_version=(
                    PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION
                ),
                operation=operation,
                status=RAW_AUDIT_STATUS_COMPLETED,
                failure_reason=None,
                files=tuple(
                    sorted(
                        declarations,
                        key=lambda item: (item.kind, item.path),
                    )
                ),
            )
            raw_index = Phase3RawAuditRunIndex.create((record,))
            try:
                raw_sources = fixture["source_bytes_by_path"]
                self.assertIsInstance(raw_sources, dict)
                source_bytes = {
                    relative: (
                        raw_sources[relative]
                        if relative in REQUIRED_SUT_SOURCES
                        else f"fixture:{relative}\n".encode("utf-8")
                    )
                    for relative in (
                        phase3_coordinator.PHASE3_EXECUTION_SOURCE_PATHS
                    )
                }
                source_digests = {
                    relative: sha256_hex(payload)
                    for relative, payload in source_bytes.items()
                }
                pin = Phase3ExecutionSourcePin(
                    execution_git_sha=operation.execution_git_sha,
                    source_bytes_by_path=tuple(source_bytes.items()),
                    source_identity_sha256=phase3_source_identity_sha256(
                        {
                            relative: source_digests[relative]
                            for relative in REQUIRED_SUT_SOURCES
                        }
                    ),
                    execution_source_identity_sha256=(
                        phase3_coordinator
                        ._phase3_execution_source_identity_sha256(
                            source_digests
                        )
                    ),
                )
                backend_raw = fixture["backend_identity_raw"]
                self.assertIsInstance(backend_raw, bytes)
                backend = BF16BackendIdentity.from_dict(
                    json.loads(backend_raw.decode("utf-8"))
                )
                outcome = _raw_audit_ingestion_outcome()
                outcome.update(
                    {
                        "process_audit_passed": True,
                        "commitment_validation_passed": True,
                        (
                            "execution_source_revalidated_after_worker_exit"
                        ): True,
                    }
                )
                _ingest_worker_evidence_v2(
                    evidence=build_phase3_worker_evidence_v2(
                        raw_index
                    ),
                    expected_run_id=operation.run_id,
                    expected_point_id=operation.point_id,
                    raw_audit_root_fd=descriptor,
                    raw_audit_owner_uid=owner_uid,
                    expected_operations=(operation,),
                    run=CapturingRun(),
                    outcome=outcome,
                    execution_source_pin=pin,
                    backend_identity=backend,
                )
            finally:
                os.close(descriptor)

        self.assertIs(outcome["semantic_validation_passed"], True)
        self.assertIs(outcome["scientific_completion_passed"], True)
        self.assertIs(outcome["terminal_eligible"], True)
        self.assertIs(outcome["passed"], True)
        self.assertEqual(outcome["status"], "validated")
        self.assertEqual(
            outcome["semantic_operations"][0]["gqa_verdict"],
            "gqa_nonmaterialization_verified",
        )


class Phase3ExecutionSourcePinTests(unittest.TestCase):
    def _source_bytes(self) -> dict[str, bytes]:
        return {
            relative: f"frozen:{relative}\n".encode("utf-8")
            for relative in phase3_coordinator.PHASE3_EXECUTION_SOURCE_PATHS
        }

    def test_pin_binds_live_bytes_to_commit_and_keys_use_precomputed_digests(
        self,
    ) -> None:
        git_sha = "a" * 40
        source_bytes = self._source_bytes()
        with (
            mock.patch.object(
                phase3_coordinator,
                "_run_checked",
                return_value=git_sha,
            ),
            mock.patch.object(
                phase3_coordinator,
                "_read_declared_commit_source_bytes",
                side_effect=lambda _, relative: source_bytes[relative],
            ),
            mock.patch.object(
                phase3_coordinator,
                "_read_live_sut_source_bytes",
                side_effect=lambda relative: source_bytes[relative],
            ),
        ):
            pin = _pin_phase3_execution_sources(git_sha)
            _revalidate_phase3_execution_sources(pin)

        self.assertEqual(
            tuple(relative for relative, _ in pin.source_bytes_by_path),
            phase3_coordinator.PHASE3_EXECUTION_SOURCE_PATHS,
        )
        self.assertEqual(
            pin.source_identity_sha256,
            phase3_coordinator.phase3_source_identity_sha256(
                pin.sut_source_sha256_by_path
            ),
        )
        self.assertEqual(
            pin.execution_source_identity_sha256,
            phase3_coordinator._phase3_execution_source_identity_sha256(
                pin.source_sha256_by_path
            ),
        )
        cache = SimpleNamespace(layout_fingerprint="1" * 64)
        backend = SimpleNamespace(fingerprint=lambda: "6" * 64)
        with mock.patch.object(
            phase3_coordinator,
            "sha256_file",
            side_effect=AssertionError("operation keys reopened a source path"),
        ):
            operations = _expected_phase3_raw_audit_operations(
                point=_point(),
                run_id=RUN_ID,
                git_sha=git_sha,
                cache=cache,
                backend=backend,
                source_sha256_by_path=pin.sut_source_sha256_by_path,
            )
        self.assertEqual(len(operations), 1)
        self.assertEqual(
            operations[0].source_identity_sha256,
            pin.source_identity_sha256,
        )
        static_relative = phase3_coordinator.STATIC_CACHE_SOURCE.relative_to(
            phase3_coordinator.REPOSITORY_ROOT
        ).as_posix()
        cache_identity = _cache_identity(
            _point(),
            implementation_sha256=pin.source_sha256_by_path[
                static_relative
            ],
        )
        _validate_cache_source_join(cache_identity, pin)

        changed_sources = dict(source_bytes)
        changed_sources[static_relative] = b"different static cache bytes\n"
        changed_pin = Phase3ExecutionSourcePin(
            execution_git_sha=git_sha,
            source_bytes_by_path=tuple(changed_sources.items()),
            source_identity_sha256=(
                phase3_coordinator.phase3_source_identity_sha256(
                    {
                        relative: sha256_hex(changed_sources[relative])
                        for relative in phase3_coordinator.REQUIRED_SUT_SOURCES
                    }
                )
            ),
            execution_source_identity_sha256=(
                phase3_coordinator._phase3_execution_source_identity_sha256(
                    {
                        relative: sha256_hex(payload)
                        for relative, payload in changed_sources.items()
                    }
                )
            ),
        )
        with self.assertRaisesRegex(
            Phase3CoordinatorError,
            "cache identity differs from pinned",
        ):
            _validate_cache_source_join(cache_identity, changed_pin)

    def test_pin_rejects_live_source_that_differs_from_declared_commit(self) -> None:
        git_sha = "a" * 40
        source_bytes = self._source_bytes()
        changed_path = phase3_coordinator.REQUIRED_SUT_SOURCES[1]
        with (
            mock.patch.object(
                phase3_coordinator,
                "_run_checked",
                return_value=git_sha,
            ),
            mock.patch.object(
                phase3_coordinator,
                "_read_declared_commit_source_bytes",
                side_effect=lambda _, relative: source_bytes[relative],
            ),
            mock.patch.object(
                phase3_coordinator,
                "_read_live_sut_source_bytes",
                side_effect=lambda relative: (
                    b"mutated\n"
                    if relative == changed_path
                    else source_bytes[relative]
                ),
            ),
        ):
            with self.assertRaisesRegex(
                Phase3CoordinatorError,
                "differs from the declared execution commit",
            ):
                _pin_phase3_execution_sources(git_sha)

    def test_post_exit_revalidation_rejects_head_or_source_mutation(self) -> None:
        git_sha = "a" * 40
        source_bytes = self._source_bytes()
        pin = Phase3ExecutionSourcePin(
            execution_git_sha=git_sha,
            source_bytes_by_path=tuple(source_bytes.items()),
            source_identity_sha256=(
                phase3_coordinator.phase3_source_identity_sha256(
                    {
                        relative: sha256_hex(source_bytes[relative])
                        for relative in phase3_coordinator.REQUIRED_SUT_SOURCES
                    }
                )
            ),
            execution_source_identity_sha256=(
                phase3_coordinator._phase3_execution_source_identity_sha256(
                    {
                        relative: sha256_hex(payload)
                        for relative, payload in source_bytes.items()
                    }
                )
            ),
        )
        with mock.patch.object(
            phase3_coordinator,
            "_run_checked",
            return_value="b" * 40,
        ):
            with self.assertRaisesRegex(Phase3CoordinatorError, "HEAD changed"):
                _revalidate_phase3_execution_sources(pin)

        changed_path = phase3_coordinator.REQUIRED_SUT_SOURCES[-1]
        with (
            mock.patch.object(
                phase3_coordinator,
                "_run_checked",
                return_value=git_sha,
            ),
            mock.patch.object(
                phase3_coordinator,
                "_read_declared_commit_source_bytes",
                side_effect=lambda _, relative: source_bytes[relative],
            ),
            mock.patch.object(
                phase3_coordinator,
                "_read_live_sut_source_bytes",
                side_effect=lambda relative: (
                    b"mutated-after-spawn\n"
                    if relative == changed_path
                    else source_bytes[relative]
                ),
            ),
        ):
            with self.assertRaisesRegex(
                Phase3CoordinatorError,
                "changed during worker lifetime",
            ):
                _revalidate_phase3_execution_sources(pin)


class Phase3DualChannelCommitmentTests(unittest.TestCase):
    def test_publication_preserves_full_primary_and_binds_both_channel_roles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-dual-channel-"
        ) as directory:
            root = Path(directory)
            root.chmod(0o700)
            primary_path = root / "worker-evidence.json"
            sidecar_path = root / "raw-audit-index.json"
            primary = {
                "schema_version": WORKER_EVIDENCE_V1,
                "run_id": RUN_ID,
                "point_id": POINT_ID,
                "worker_result": {"status": "aborted"},
                "runtime": {"retained": True},
                "numerical": {"retained": True},
                "model_identity": {"retained": True},
            }
            index = _index()
            observed = _publish_phase3_worker_evidence_channels(
                primary_path=primary_path,
                raw_index_path=sidecar_path,
                primary_evidence=primary,
                raw_index=index,
                run_id=RUN_ID,
                point_id=POINT_ID,
            )
            primary_raw = primary_path.read_bytes()
            sidecar_raw = sidecar_path.read_bytes()
        self.assertEqual(primary_raw, canonical_json_bytes(primary) + b"\n")
        self.assertEqual(
            _parse_canonical_json(
                sidecar_raw,
                maximum_bytes=MAX_IPC_BYTES,
                label="sidecar",
            ),
            build_phase3_worker_evidence_v2(index),
        )
        self.assertNotIn(RAW_PAYLOAD, sidecar_raw)
        self.assertEqual(
            observed,
            phase3_worker_channel_commitment_sha256(
                run_id=RUN_ID,
                point_id=POINT_ID,
                primary_evidence_bytes=primary_raw,
                raw_audit_index_bytes=sidecar_raw,
            ),
        )
        self.assertNotEqual(
            observed,
            phase3_worker_channel_commitment_sha256(
                run_id=RUN_ID,
                point_id=POINT_ID,
                primary_evidence_bytes=sidecar_raw,
                raw_audit_index_bytes=primary_raw,
            ),
        )
        self.assertNotEqual(
            observed,
            phase3_worker_channel_commitment_sha256(
                run_id=RUN_ID,
                point_id=POINT_ID,
                primary_evidence_bytes=primary_raw + b"x",
                raw_audit_index_bytes=sidecar_raw,
            ),
        )

    def test_artifacts_preserve_exact_channels_and_canonical_commitment(self) -> None:
        index = _index()
        primary_bytes = (
            canonical_json_bytes(
                {
                    "schema_version": WORKER_EVIDENCE_V1,
                    "run_id": RUN_ID,
                    "point_id": POINT_ID,
                    "opaque": "primary-v1-is-preserved-exactly",
                }
            )
            + b"\n"
        )
        sidecar_bytes = (
            canonical_json_bytes(build_phase3_worker_evidence_v2(index)) + b"\n"
        )
        run = CapturingRun()
        commitment, digest = _preserve_phase3_worker_channel_artifacts(
            run=run,
            run_id=RUN_ID,
            point_id=POINT_ID,
            primary_evidence_bytes=primary_bytes,
            raw_audit_index_bytes=sidecar_bytes,
        )

        self.assertEqual(
            run.writes[phase3_coordinator.TRANSPORT_PRIMARY_CHANNEL_ARTIFACT],
            primary_bytes,
        )
        self.assertEqual(
            run.writes[phase3_coordinator.TRANSPORT_SIDECAR_CHANNEL_ARTIFACT],
            sidecar_bytes,
        )
        expected_commitment = build_phase3_worker_channel_commitment(
            run_id=RUN_ID,
            point_id=POINT_ID,
            primary_evidence_bytes=primary_bytes,
            raw_audit_index_bytes=sidecar_bytes,
        )
        self.assertEqual(commitment, expected_commitment)
        self.assertEqual(
            run.writes[phase3_coordinator.TRANSPORT_COMMITMENT_ARTIFACT],
            canonical_json_bytes(expected_commitment),
        )
        self.assertEqual(digest, sha256_hex(canonical_json_bytes(commitment)))
        self.assertEqual(
            run.writes[
                phase3_coordinator.TRANSPORT_COMMITMENT_DIGEST_ARTIFACT
            ],
            digest.encode("ascii") + b"\n",
        )


class Phase3PinnedIPCReadTests(unittest.TestCase):
    def test_missing_sidecar_and_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-pinned-ipc-"
        ) as directory:
            root = Path(directory)
            root.chmod(0o700)
            descriptor, owner_uid = _open_private_ipc_parent(root)
            try:
                with self.assertRaisesRegex(
                    Phase3CoordinatorError, "sidecar is absent"
                ):
                    _read_pinned_ipc_file(
                        descriptor,
                        "raw-audit-index.json",
                        expected_owner_uid=owner_uid,
                        maximum_bytes=MAX_IPC_BYTES,
                        label="sidecar",
                    )
                target = root / "target.json"
                target.write_bytes(b"{}\n")
                target.chmod(0o600)
                (root / "raw-audit-index.json").symlink_to(target.name)
                with self.assertRaisesRegex(
                    Phase3CoordinatorError, "unsafe or oversized"
                ):
                    _read_pinned_ipc_file(
                        descriptor,
                        "raw-audit-index.json",
                        expected_owner_uid=owner_uid,
                        maximum_bytes=MAX_IPC_BYTES,
                        label="sidecar",
                    )
            finally:
                os.close(descriptor)

    def test_pinned_parent_rejects_path_replacement_bytes(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-pinned-parent-"
        ) as directory:
            base = Path(directory)
            original = base / "ipc"
            original.mkdir(mode=0o700)
            descriptor, owner_uid = _open_private_ipc_parent(original)
            retained = canonical_json_bytes({"retained": True}) + b"\n"
            try:
                target = original / "worker-evidence.json"
                target.write_bytes(retained)
                target.chmod(0o600)
                moved = base / "original-ipc"
                original.rename(moved)
                original.mkdir(mode=0o700)
                replacement = original / "worker-evidence.json"
                replacement.write_bytes(
                    canonical_json_bytes({"attacker": True}) + b"\n"
                )
                replacement.chmod(0o600)
                observed = _read_pinned_ipc_file(
                    descriptor,
                    "worker-evidence.json",
                    expected_owner_uid=owner_uid,
                    maximum_bytes=MAX_IPC_BYTES,
                    label="primary",
                )
            finally:
                os.close(descriptor)
        self.assertEqual(observed, retained)


class Phase3RawAuditSemanticJoinTests(unittest.TestCase):
    def test_provenance_mismatch_fails_before_ingestion_or_artifact_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-provenance-"
        ) as directory:
            root = Path(directory)
            root.chmod(0o700)
            descriptor, owner_uid = _open_private_raw_audit_root(root)
            index = _index()
            expected = Phase3AuditOperationKey.from_point(
                run_id=RUN_ID,
                point=_point(),
                decode_step=0,
                cache_layout_fingerprint="1" * 64,
                execution_git_sha="2" * 40,
                plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[
                    PHASE3_FIXED_PLAN_PATH
                ],
                hardware_identity_sha256="3" * 64,
                software_identity_sha256="4" * 64,
                model_identity_sha256="5" * 64,
                backend_identity_sha256="6" * 64,
                source_identity_sha256="8" * 64,
            )
            run = CapturingRun()
            try:
                with mock.patch.object(
                    phase3_coordinator,
                    "ingest_phase3_raw_audit_evidence_fd",
                ) as ingest:
                    with self.assertRaisesRegex(
                        Phase3CoordinatorError,
                        "provenance differs",
                    ):
                        _ingest_worker_evidence_v2(
                            evidence=build_phase3_worker_evidence_v2(index),
                            expected_run_id=RUN_ID,
                            expected_point_id=POINT_ID,
                            raw_audit_root_fd=descriptor,
                            raw_audit_owner_uid=owner_uid,
                            expected_operations=(expected,),
                            run=run,
                            outcome=_raw_audit_ingestion_outcome(),
                        )
                    ingest.assert_not_called()
            finally:
                os.close(descriptor)
        self.assertEqual(run.writes, {})

    def test_completed_declarations_require_independent_semantic_replay(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-completed-index-"
        ) as directory:
            root = Path(directory)
            root.chmod(0o700)
            descriptor, owner_uid = _open_private_raw_audit_root(root)
            index = _completed_index()
            _write_index_files(root, index)
            run = CapturingRun()
            outcome = _raw_audit_ingestion_outcome()
            try:
                _ingest(
                    root_fd=descriptor,
                    owner_uid=owner_uid,
                    index=index,
                    run=run,
                    outcome=outcome,
                )
            finally:
                os.close(descriptor)
        self.assertIs(outcome["ingestion_passed"], True)
        self.assertIs(outcome["collection_validation_passed"], False)
        self.assertIs(outcome["collection_completion_passed"], False)
        self.assertIs(outcome["declaration_completion_observed"], True)
        self.assertIs(outcome["semantic_validation_pending"], True)
        self.assertIs(outcome["semantic_validation_attempted"], False)
        self.assertIs(outcome["semantic_validation_passed"], False)
        self.assertIs(outcome["scientific_completion_passed"], False)
        self.assertIs(outcome["passed"], False)
        self.assertIs(outcome["terminal_eligible"], False)
        self.assertEqual(
            outcome["status"],
            "ingested_declared_complete_pending_semantic_validation",
        )
        self.assertEqual(
            outcome["artifact_file_count"],
            len(REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS) + 1,
        )

    def test_failed_index_is_preserved_but_completed_result_becomes_aborted(
        self,
    ) -> None:
        bundle = phase3_coordinator.load_phase3_admission_bundle(
            Path(PHASE3_FIXED_PLAN_PATH)
        )
        expected_operations = (
            bundle.plan.measurement.measured_count
            * bundle.plan.measurement.measured_batches
        )
        result = Phase3WorkerResult(
            schema_version=Phase3WorkerResult.SCHEMA_VERSION,
            run_id=RUN_ID,
            point_id=POINT_ID,
            runner_kind=RunnerKind.FIXED_L,
            count_unit=bundle.plan.measurement.count_unit,
            status=RunStatus.COMPLETED,
            expected_operations=expected_operations,
            completed_operations=expected_operations,
            failed_operations=0,
            output_checksum="9" * 64,
            failure_reason=None,
        )
        outcome = _raw_audit_ingestion_outcome()
        outcome.update(
            {
                "passed": False,
                "ingestion_passed": True,
                "scientific_completion_passed": False,
                "terminal_eligible": False,
                "status": "ingested_failed_evidence",
            }
        )
        status, reason = _resolve_phase3_terminal_status(
            result=result,
            process_audit_passed=True,
            worker_evidence_valid=True,
            raw_audit_outcome=outcome,
            failure_reason=None,
        )
        self.assertIs(status, RunStatus.ABORTED)
        self.assertIn("without complete scientific raw-audit evidence", reason)
        assert reason is not None
        resolved = _resolved_phase3_worker_result(
            result,
            final_status=status,
            final_reason=reason,
        )
        self.assertIs(resolved.status, RunStatus.ABORTED)
        self.assertEqual(resolved.failure_reason, reason)
        self.assertEqual(
            resolved.completed_operations,
            result.completed_operations,
        )
        self.assertEqual(resolved.output_checksum, result.output_checksum)
        self.assertNotEqual(resolved.to_dict(), result.to_dict())

    def test_terminal_result_rejects_inconsistent_completion_reason(self) -> None:
        bundle = phase3_coordinator.load_phase3_admission_bundle(
            Path(PHASE3_FIXED_PLAN_PATH)
        )
        expected = (
            bundle.plan.measurement.measured_count
            * bundle.plan.measurement.measured_batches
        )
        result = Phase3WorkerResult(
            schema_version=Phase3WorkerResult.SCHEMA_VERSION,
            run_id=RUN_ID,
            point_id=POINT_ID,
            runner_kind=RunnerKind.FIXED_L,
            count_unit=bundle.plan.measurement.count_unit,
            status=RunStatus.COMPLETED,
            expected_operations=expected,
            completed_operations=expected,
            failed_operations=0,
            output_checksum="8" * 64,
            failure_reason=None,
        )
        with self.assertRaisesRegex(
            Phase3CoordinatorError, "completed terminal result has a reason"
        ):
            _resolved_phase3_worker_result(
                result,
                final_status=RunStatus.COMPLETED,
                final_reason="not allowed",
            )

if __name__ == "__main__":
    unittest.main()

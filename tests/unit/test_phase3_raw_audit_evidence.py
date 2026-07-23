"""CPU-only tests for strict Phase 3 raw-audit evidence ingestion."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any
import unittest

from kvbench.errors import SchemaValidationError
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.runtime.phase3_raw_audit_evidence import (
    MAX_RAW_AUDIT_FILES_PER_RUN,
    MAX_RAW_AUDIT_FILE_SIZE_BYTES,
    MAX_RAW_AUDIT_RUN_INDEX_SIZE_BYTES,
    MAX_RAW_AUDIT_RUN_SIZE_BYTES,
    PHASE3_RAW_AUDIT_FILE_SCHEMA_VERSION,
    PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
    PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND,
    PHASE3_RAW_AUDIT_RUN_INDEX_SCHEMA_VERSION,
    RAW_AUDIT_STATUS_COMPLETED,
    RAW_AUDIT_STATUS_FAILED,
    RAW_AUDIT_STATUS_NOT_ATTEMPTED,
    REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS,
    Phase3RawAuditEvidenceError,
    Phase3RawAuditFile,
    Phase3RawAuditOperationRecord,
    Phase3RawAuditRunIndex,
    ingest_phase3_raw_audit_evidence,
    ingest_phase3_raw_audit_evidence_fd,
    parse_phase3_raw_audit_run_index_bytes,
    validate_phase3_raw_audit_campaign_indices,
)
from kvbench.schema import GraphMode, RunnerKind
from kvbench.schema.base import canonical_json_bytes
from kvbench.schema.phase3 import (
    FROZEN_PHASE3_POINT_IDS,
    FROZEN_PHASE3_STABILITY_POINT_IDS,
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GROWING_PLAN_PATH,
    PHASE3_PLAN_FINGERPRINTS,
    Phase3ProcessPoint,
)


CACHE_SHA256 = "1" * 64
MODEL_SHA256 = "2" * 64
BACKEND_SHA256 = "3" * 64
SOURCE_SHA256 = "4" * 64
EXECUTION_GIT_SHA = "5" * 40
HARDWARE_SHA256 = "6" * 64
SOFTWARE_SHA256 = "7" * 64
RUN_ID = "phase3-raw-audit-fixture"

FIXED_L_CAMPAIGN_ID = "phase3-raw-audit-fixed-l-campaign"
GROWING_CAMPAIGN_ID = "phase3-raw-audit-growing-campaign"
_POINT_ID_RE = re.compile(
    r"\A(?P<runner>fixed_l|growing_context)-b(?P<batch>[1-9][0-9]*)-"
    r"l(?P<context>[1-9][0-9]*)-(?P<graph>eager|cuda_graph)-"
    r"r(?P<replicate>[1-9][0-9]*)\Z"
)


def fixed_point() -> Phase3ProcessPoint:
    return Phase3ProcessPoint(
        point_id="fixed_l-b1-l128-eager-r1",
        runner_kind=RunnerKind.FIXED_L,
        graph_mode=GraphMode.EAGER,
        batch_size=1,
        context_length=128,
        output_steps=1,
        process_replicate=1,
        stability_member=False,
    )


def growing_point() -> Phase3ProcessPoint:
    return Phase3ProcessPoint(
        point_id="growing_context-b1-l128-eager-r1",
        runner_kind=RunnerKind.GROWING_CONTEXT,
        graph_mode=GraphMode.EAGER,
        batch_size=1,
        context_length=128,
        output_steps=16,
        process_replicate=1,
        stability_member=False,
    )


def operation_key(
    point: Phase3ProcessPoint,
    decode_step: int,
    *,
    run_id: str = RUN_ID,
    hardware_identity_sha256: str = HARDWARE_SHA256,
    software_identity_sha256: str = SOFTWARE_SHA256,
    model_identity_sha256: str = MODEL_SHA256,
    backend_identity_sha256: str = BACKEND_SHA256,
    source_identity_sha256: str = SOURCE_SHA256,
) -> Phase3AuditOperationKey:
    plan_path = (
        PHASE3_FIXED_PLAN_PATH
        if point.runner_kind is RunnerKind.FIXED_L
        else PHASE3_GROWING_PLAN_PATH
    )
    return Phase3AuditOperationKey.from_point(
        run_id=run_id,
        point=point,
        decode_step=decode_step,
        cache_layout_fingerprint=CACHE_SHA256,
        execution_git_sha=EXECUTION_GIT_SHA,
        plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[plan_path],
        hardware_identity_sha256=hardware_identity_sha256,
        software_identity_sha256=software_identity_sha256,
        model_identity_sha256=model_identity_sha256,
        backend_identity_sha256=backend_identity_sha256,
        source_identity_sha256=source_identity_sha256,
    )


def point_from_id(point_id: str) -> Phase3ProcessPoint:
    match = _POINT_ID_RE.fullmatch(point_id)
    if match is None:  # pragma: no cover - frozen fixture invariant
        raise AssertionError("frozen point ID no longer matches its schema")
    runner_kind = RunnerKind(match.group("runner"))
    return Phase3ProcessPoint(
        point_id=point_id,
        runner_kind=runner_kind,
        graph_mode=GraphMode(match.group("graph")),
        batch_size=int(match.group("batch")),
        context_length=int(match.group("context")),
        output_steps=1 if runner_kind is RunnerKind.FIXED_L else 16,
        process_replicate=int(match.group("replicate")),
        stability_member=point_id in FROZEN_PHASE3_STABILITY_POINT_IDS,
    )


def campaign_fixture(
    *,
    identity_override_by_point: dict[str, str] | None = None,
    run_id_override_by_point: dict[str, str] | None = None,
) -> tuple[tuple[Phase3RawAuditRunIndex, ...], dict[str, str]]:
    identity_overrides = identity_override_by_point or {}
    run_id_overrides = run_id_override_by_point or {}
    indices: list[Phase3RawAuditRunIndex] = []
    membership: dict[str, str] = {}
    for ordinal, point_id in enumerate(FROZEN_PHASE3_POINT_IDS):
        point = point_from_id(point_id)
        run_id = run_id_overrides.get(
            point_id,
            f"phase3-raw-audit-campaign-run-{ordinal:02d}",
        )
        records = tuple(
            completed_record(
                operation_key(
                    point,
                    step,
                    run_id=run_id,
                    hardware_identity_sha256=identity_overrides.get(
                        point_id, HARDWARE_SHA256
                    ),
                )
            )
            for step in range(point.output_steps)
        )
        indices.append(Phase3RawAuditRunIndex.create(records))
        membership[run_id] = (
            FIXED_L_CAMPAIGN_ID
            if point.runner_kind is RunnerKind.FIXED_L
            else GROWING_CAMPAIGN_ID
        )
    return tuple(indices), membership


def file_declaration(
    *,
    path: str,
    kind: str,
    payload: bytes,
) -> Phase3RawAuditFile:
    return Phase3RawAuditFile.from_bytes(
        path=path,
        kind=kind,
        payload=payload,
    )


def write_file(root: Path, declaration: Phase3RawAuditFile, payload: bytes) -> None:
    target = root.joinpath(*Path(declaration.path).parts)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.write_bytes(payload)


def record_with_files(
    operation: Phase3AuditOperationKey,
    *,
    status: str,
    failure_reason: str | None,
    declared_payloads: tuple[tuple[str, str, bytes], ...],
    root: Path | None = None,
) -> Phase3RawAuditOperationRecord:
    declarations = tuple(
        sorted(
            (
                file_declaration(path=path, kind=kind, payload=payload)
                for kind, path, payload in declared_payloads
            ),
            key=lambda item: (item.kind, item.path),
        )
    )
    if root is not None:
        payload_by_path = {
            path: payload for _, path, payload in declared_payloads
        }
        for declaration in declarations:
            write_file(root, declaration, payload_by_path[declaration.path])
    return Phase3RawAuditOperationRecord(
        schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
        operation=operation,
        status=status,
        failure_reason=failure_reason,
        files=declarations,
    )


def completed_record(
    operation: Phase3AuditOperationKey,
    *,
    root: Path | None = None,
) -> Phase3RawAuditOperationRecord:
    prefix = f"step-{operation.decode_step:04d}"
    kinds = REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS
    if operation.decode_step == 0:
        kinds = tuple(
            sorted((*kinds, PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND))
        )
    payloads = tuple(
        (
            kind,
            f"{prefix}/{kind}.json",
            f"{operation.operation_fingerprint_sha256}:{kind}".encode("ascii"),
        )
        for kind in kinds
    )
    return record_with_files(
        operation,
        status=RAW_AUDIT_STATUS_COMPLETED,
        failure_reason=None,
        declared_payloads=payloads,
        root=root,
    )


def failed_record(
    operation: Phase3AuditOperationKey,
    *,
    root: Path | None = None,
    payloads: tuple[tuple[str, str, bytes], ...] = (),
) -> Phase3RawAuditOperationRecord:
    return record_with_files(
        operation,
        status=RAW_AUDIT_STATUS_FAILED,
        failure_reason="allocator_snapshot_collection_failed",
        declared_payloads=payloads,
        root=root,
    )


def unattempted_record(
    operation: Phase3AuditOperationKey,
) -> Phase3RawAuditOperationRecord:
    return record_with_files(
        operation,
        status=RAW_AUDIT_STATUS_NOT_ATTEMPTED,
        failure_reason="prior_operation_failed",
        declared_payloads=(),
    )


class Phase3RawAuditSchemaTests(unittest.TestCase):
    def test_completed_fixed_index_round_trips_strictly(self) -> None:
        record = completed_record(operation_key(fixed_point(), 0))
        index = Phase3RawAuditRunIndex.create((record,))
        self.assertEqual(
            index.schema_version,
            PHASE3_RAW_AUDIT_RUN_INDEX_SCHEMA_VERSION,
        )
        self.assertEqual(
            Phase3RawAuditRunIndex.from_dict(index.to_dict()),
            index,
        )
        self.assertEqual(
            tuple(item.kind for item in record.files),
            tuple(
                sorted(
                    (
                        *REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS,
                        PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND,
                    )
                )
            ),
        )

    def test_completed_file_kind_set_requires_session_provenance(self) -> None:
        expected = tuple(
            sorted(
                {
                    "b011_audit",
                    "b011_gqa_chrome_trace",
                    "b011_mha_chrome_trace",
                    "b012_allocation_audit",
                    "b012_allocator_snapshot",
                    "b012_allocator_trace",
                }
            )
        )
        self.assertEqual(len(expected), 6)
        self.assertEqual(REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS, expected)
        self.assertNotIn(
            PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND, expected
        )
        self.assertNotIn(
            "b011_source_audit", REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS
        )

    def test_run_index_byte_parser_requires_canonical_duplicate_free_json(
        self,
    ) -> None:
        index = Phase3RawAuditRunIndex.create(
            (completed_record(operation_key(fixed_point(), 0)),)
        )
        canonical = canonical_json_bytes(index.to_dict())
        self.assertEqual(parse_phase3_raw_audit_run_index_bytes(canonical), index)

        noncanonical = json.dumps(
            index.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertNotEqual(noncanonical, canonical)
        rejected = (
            canonical + b"\n",
            noncanonical,
            b'{"outer":{"key":1,"key":2}}',
            b'{"value":NaN}',
            canonical_json_bytes({"unknown": True}),
            b"\xff",
            b"",
        )
        for payload in rejected:
            with self.subTest(payload=payload[:32]):
                with self.assertRaises(Phase3RawAuditEvidenceError):
                    parse_phase3_raw_audit_run_index_bytes(payload)
        with self.assertRaises(TypeError):
            parse_phase3_raw_audit_run_index_bytes(
                bytearray(canonical)  # type: ignore[arg-type]
            )
        with self.assertRaises(Phase3RawAuditEvidenceError):
            parse_phase3_raw_audit_run_index_bytes(
                b" " * (MAX_RAW_AUDIT_RUN_INDEX_SIZE_BYTES + 1)
            )

    def test_run_index_schema_enforces_hard_file_count(self) -> None:
        payloads = tuple(
            (
                f"partial_{ordinal:04d}",
                f"partial/{ordinal:04d}.bin",
                b"",
            )
            for ordinal in range(MAX_RAW_AUDIT_FILES_PER_RUN + 1)
        )
        record = failed_record(
            operation_key(fixed_point(), 0),
            payloads=payloads,
        )
        with self.assertRaises(ValueError):
            Phase3RawAuditRunIndex.create((record,))

    def test_strict_parser_rejects_missing_unknown_and_wrong_types(self) -> None:
        index = Phase3RawAuditRunIndex.create(
            (completed_record(operation_key(fixed_point(), 0)),)
        )
        payload = index.to_dict()
        missing = copy.deepcopy(payload)
        del missing["point_id"]
        with self.assertRaises(SchemaValidationError):
            Phase3RawAuditRunIndex.from_dict(missing)
        unknown = copy.deepcopy(payload)
        unknown["trusted"] = True
        with self.assertRaises(SchemaValidationError):
            Phase3RawAuditRunIndex.from_dict(unknown)
        wrong_size = copy.deepcopy(payload)
        wrong_size["records"][0]["files"][0]["size_bytes"] = False
        with self.assertRaises(SchemaValidationError):
            Phase3RawAuditRunIndex.from_dict(wrong_size)

    def test_file_schema_is_versioned_and_path_is_strict(self) -> None:
        valid = file_declaration(path="step/a.json", kind="b011_audit", payload=b"x")
        self.assertEqual(valid.schema_version, PHASE3_RAW_AUDIT_FILE_SCHEMA_VERSION)
        for path in (
            "../outside",
            "/absolute",
            "a\\b",
            "a//b",
            "a/./b",
            "a/../b",
            "a/\x01b",
            "a/\udcffb",
        ):
            with self.subTest(path=repr(path)):
                with self.assertRaises(ValueError):
                    file_declaration(path=path, kind="partial", payload=b"x")
        with self.assertRaises(ValueError):
            Phase3RawAuditFile(
                schema_version=PHASE3_RAW_AUDIT_FILE_SCHEMA_VERSION,
                path="large",
                kind="partial",
                sha256=hashlib.sha256(b"").hexdigest(),
                size_bytes=MAX_RAW_AUDIT_FILE_SIZE_BYTES + 1,
            )

    def test_completed_record_requires_exact_nonempty_file_set(self) -> None:
        operation = operation_key(fixed_point(), 0)
        complete = completed_record(operation)
        with self.assertRaises(ValueError):
            Phase3RawAuditOperationRecord(
                schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
                operation=operation,
                status=RAW_AUDIT_STATUS_COMPLETED,
                failure_reason=None,
                files=complete.files[:-1],
            )
        extra = file_declaration(
            path="step-0000/unexpected.json",
            kind="unexpected_partial",
            payload=b"extra",
        )
        files = tuple(
            sorted(
                (*complete.files, extra),
                key=lambda item: (item.kind, item.path),
            )
        )
        with self.assertRaises(ValueError):
            Phase3RawAuditOperationRecord(
                schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
                operation=operation,
                status=RAW_AUDIT_STATUS_COMPLETED,
                failure_reason=None,
                files=files,
            )

    def test_status_and_failure_reason_contracts_are_fail_closed(self) -> None:
        operation = operation_key(fixed_point(), 0)
        with self.assertRaises(ValueError):
            Phase3RawAuditOperationRecord(
                schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
                operation=operation,
                status=RAW_AUDIT_STATUS_FAILED,
                failure_reason=None,
                files=(),
            )
        partial = file_declaration(path="partial.json", kind="partial", payload=b"x")
        with self.assertRaises(ValueError):
            Phase3RawAuditOperationRecord(
                schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
                operation=operation,
                status=RAW_AUDIT_STATUS_NOT_ATTEMPTED,
                failure_reason="prior_operation_failed",
                files=(partial,),
            )
        with self.assertRaises(ValueError):
            Phase3RawAuditOperationRecord(
                schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
                operation=operation,
                status="skipped",
                failure_reason="prior_operation_failed",
                files=(),
            )

    def test_duplicate_kinds_and_noncanonical_file_order_are_rejected(self) -> None:
        operation = operation_key(fixed_point(), 0)
        left = file_declaration(path="a", kind="partial", payload=b"a")
        right = file_declaration(path="b", kind="partial", payload=b"b")
        with self.assertRaises(ValueError):
            Phase3RawAuditOperationRecord(
                schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
                operation=operation,
                status=RAW_AUDIT_STATUS_FAILED,
                failure_reason="collector_failed",
                files=(left, right),
            )
        first = file_declaration(path="z", kind="z_partial", payload=b"z")
        second = file_declaration(path="a", kind="a_partial", payload=b"a")
        with self.assertRaises(ValueError):
            Phase3RawAuditOperationRecord(
                schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
                operation=operation,
                status=RAW_AUDIT_STATUS_FAILED,
                failure_reason="collector_failed",
                files=(first, second),
            )

    def test_growing_index_requires_all_sixteen_ordered_operations(self) -> None:
        point = growing_point()
        records = tuple(
            completed_record(operation_key(point, step)) for step in range(16)
        )
        self.assertEqual(len(Phase3RawAuditRunIndex.create(records).records), 16)
        with self.assertRaises(ValueError):
            Phase3RawAuditRunIndex.create(records[:-1])
        with self.assertRaises(ValueError):
            Phase3RawAuditRunIndex.create(tuple(reversed(records)))

    def test_failure_transition_requires_all_later_steps_unattempted(self) -> None:
        point = growing_point()
        keys = tuple(operation_key(point, step) for step in range(16))
        valid = (
            completed_record(keys[0]),
            failed_record(keys[1]),
            *(unattempted_record(key) for key in keys[2:]),
        )
        Phase3RawAuditRunIndex.create(valid)
        invalid_after = list(valid)
        invalid_after[2] = completed_record(keys[2])
        with self.assertRaises(ValueError):
            Phase3RawAuditRunIndex.create(invalid_after)
        invalid_before = list(valid)
        invalid_before[0] = unattempted_record(keys[0])
        with self.assertRaises(ValueError):
            Phase3RawAuditRunIndex.create(invalid_before)

    def test_duplicate_and_file_directory_conflicting_paths_are_rejected(self) -> None:
        point = growing_point()
        keys = tuple(operation_key(point, step) for step in range(16))
        first = completed_record(keys[0])
        second = completed_record(keys[1])
        duplicate_files = tuple(
            first.files[0] if item.kind == first.files[0].kind else item
            for item in second.files
        )
        duplicate_record = Phase3RawAuditOperationRecord(
            schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
            operation=keys[1],
            status=RAW_AUDIT_STATUS_COMPLETED,
            failure_reason=None,
            files=duplicate_files,
        )
        records = (
            first,
            duplicate_record,
            *(completed_record(key) for key in keys[2:]),
        )
        with self.assertRaises(ValueError):
            Phase3RawAuditRunIndex.create(records)

        first_prefix_files = tuple(
            file_declaration(path="prefix", kind=item.kind, payload=b"one")
            if item.kind == first.files[0].kind
            else item
            for item in first.files
        )
        first_prefix = Phase3RawAuditOperationRecord(
            schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
            operation=keys[0],
            status=RAW_AUDIT_STATUS_COMPLETED,
            failure_reason=None,
            files=first_prefix_files,
        )
        second_prefix_files = tuple(
            file_declaration(
                path="prefix/child",
                kind=item.kind,
                payload=b"two",
            )
            if item.kind == second.files[0].kind
            else item
            for item in second.files
        )
        second_prefix = Phase3RawAuditOperationRecord(
            schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
            operation=keys[1],
            status=RAW_AUDIT_STATUS_COMPLETED,
            failure_reason=None,
            files=second_prefix_files,
        )
        prefix_records = (
            first_prefix,
            second_prefix,
            *(completed_record(key) for key in keys[2:]),
        )
        with self.assertRaises(ValueError):
            Phase3RawAuditRunIndex.create(prefix_records)



class Phase3RawAuditCampaignValidationTests(unittest.TestCase):
    def test_exact_frozen_campaign_has_20_runs_and_80_unique_operations(
        self,
    ) -> None:
        indices, membership = campaign_fixture()
        observed = validate_phase3_raw_audit_campaign_indices(
            indices,
            campaign_membership=membership,
            fixed_l_campaign_id=FIXED_L_CAMPAIGN_ID,
            growing_context_campaign_id=GROWING_CAMPAIGN_ID,
        )
        self.assertEqual(observed, indices)
        self.assertEqual(len(observed), 20)
        operations = tuple(
            record.operation for index in observed for record in index.records
        )
        self.assertEqual(len(operations), 80)
        self.assertEqual(
            len(
                {
                    operation.operation_fingerprint_sha256
                    for operation in operations
                }
            ),
            80,
        )

    def test_campaign_rejects_grid_order_run_id_and_identity_tampering(
        self,
    ) -> None:
        indices, membership = campaign_fixture()
        for candidate in (indices[:-1], tuple(reversed(indices))):
            with self.subTest(run_count=len(candidate)):
                with self.assertRaises(ValueError):
                    validate_phase3_raw_audit_campaign_indices(
                        candidate,
                        campaign_membership=membership,
                        fixed_l_campaign_id=FIXED_L_CAMPAIGN_ID,
                        growing_context_campaign_id=GROWING_CAMPAIGN_ID,
                    )

        duplicate_indices, duplicate_membership = campaign_fixture(
            run_id_override_by_point={
                FROZEN_PHASE3_POINT_IDS[1]: indices[0].run_id
            }
        )
        with self.assertRaises(ValueError):
            validate_phase3_raw_audit_campaign_indices(
                duplicate_indices,
                campaign_membership=duplicate_membership,
                fixed_l_campaign_id=FIXED_L_CAMPAIGN_ID,
                growing_context_campaign_id=GROWING_CAMPAIGN_ID,
            )

        mixed_indices, mixed_membership = campaign_fixture(
            identity_override_by_point={
                FROZEN_PHASE3_POINT_IDS[-1]: "8" * 64
            }
        )
        with self.assertRaises(ValueError):
            validate_phase3_raw_audit_campaign_indices(
                mixed_indices,
                campaign_membership=mixed_membership,
                fixed_l_campaign_id=FIXED_L_CAMPAIGN_ID,
                growing_context_campaign_id=GROWING_CAMPAIGN_ID,
            )

    def test_campaign_membership_is_exact_and_lane_bound(self) -> None:
        indices, membership = campaign_fixture()
        missing = dict(membership)
        missing.pop(indices[0].run_id)
        extra = {**membership, "phase3-unregistered-run": FIXED_L_CAMPAIGN_ID}
        wrong_lane = dict(membership)
        wrong_lane[indices[0].run_id] = GROWING_CAMPAIGN_ID
        for candidate in (missing, extra, wrong_lane):
            with self.subTest(keys=len(candidate)):
                with self.assertRaises(ValueError):
                    validate_phase3_raw_audit_campaign_indices(
                        indices,
                        campaign_membership=candidate,
                        fixed_l_campaign_id=FIXED_L_CAMPAIGN_ID,
                        growing_context_campaign_id=GROWING_CAMPAIGN_ID,
                    )
        with self.assertRaises(ValueError):
            validate_phase3_raw_audit_campaign_indices(
                indices,
                campaign_membership=membership,
                fixed_l_campaign_id=FIXED_L_CAMPAIGN_ID,
                growing_context_campaign_id=FIXED_L_CAMPAIGN_ID,
            )

class Phase3RawAuditSecureIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-raw-audit-test-"
        )
        self.base = Path(self.temporary.name)
        self.root = self.base / "evidence"
        self.root.mkdir(mode=0o700)
        self.root.chmod(0o700)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_completed_fixed_evidence_returns_immutable_verified_bytes(self) -> None:
        record = completed_record(operation_key(fixed_point(), 0), root=self.root)
        index = Phase3RawAuditRunIndex.create((record,))
        observed = ingest_phase3_raw_audit_evidence(self.root, index)
        self.assertIsInstance(observed, MappingProxyType)
        self.assertEqual(set(observed), {item.path for item in record.files})
        for declaration in record.files:
            self.assertEqual(
                hashlib.sha256(observed[declaration.path]).hexdigest(),
                declaration.sha256,
            )
        with self.assertRaises(TypeError):
            mutable_view: Any = observed
            mutable_view["new"] = b"forbidden"

    def test_complete_growing_evidence_reads_all_sixteen_steps(self) -> None:
        point = growing_point()
        records = tuple(
            completed_record(operation_key(point, step), root=self.root)
            for step in range(16)
        )
        index = Phase3RawAuditRunIndex.create(records)
        observed = ingest_phase3_raw_audit_evidence(self.root, index)
        self.assertEqual(
            len(observed),
            16 * len(REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS) + 1,
        )

    def test_partial_failure_is_preserved_and_later_steps_are_absent(self) -> None:
        point = growing_point()
        keys = tuple(operation_key(point, step) for step in range(16))
        completed = tuple(
            completed_record(keys[step], root=self.root) for step in range(3)
        )
        partials = (
            (
                "b011_gqa_chrome_trace",
                "step-0003/b011_gqa_chrome_trace.json",
                b"partial trace",
            ),
            (
                "collector_failure_log",
                "step-0003/collector_failure_log.json",
                b"failure evidence",
            ),
        )
        failed = failed_record(keys[3], root=self.root, payloads=partials)
        later = tuple(unattempted_record(key) for key in keys[4:])
        index = Phase3RawAuditRunIndex.create((*completed, failed, *later))
        observed = ingest_phase3_raw_audit_evidence(self.root, index)
        self.assertEqual(
            set(observed),
            {
                item.path
                for record in (*completed, failed)
                for item in record.files
            },
        )

    def test_failed_fixed_operation_may_have_no_partial_files(self) -> None:
        index = Phase3RawAuditRunIndex.create(
            (failed_record(operation_key(fixed_point(), 0)),)
        )
        self.assertEqual(
            dict(ingest_phase3_raw_audit_evidence(self.root, index)),
            {},
        )

    def test_symlink_file_is_rejected(self) -> None:
        record = completed_record(operation_key(fixed_point(), 0), root=self.root)
        index = Phase3RawAuditRunIndex.create((record,))
        victim = self.root.joinpath(*Path(record.files[0].path).parts)
        payload = victim.read_bytes()
        outside = self.base / "outside.json"
        outside.write_bytes(payload)
        victim.unlink()
        victim.symlink_to(outside)
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(self.root, index)

    def test_symlink_directory_is_rejected(self) -> None:
        operation = operation_key(fixed_point(), 0)
        payload = b"partial"
        record = failed_record(
            operation,
            payloads=(("partial_trace", "linked/trace.json", payload),),
        )
        index = Phase3RawAuditRunIndex.create((record,))
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "trace.json").write_bytes(payload)
        (self.root / "linked").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(self.root, index)

    def test_hardlinked_file_is_rejected(self) -> None:
        record = completed_record(operation_key(fixed_point(), 0), root=self.root)
        index = Phase3RawAuditRunIndex.create((record,))
        victim = self.root.joinpath(*Path(record.files[0].path).parts)
        os.link(victim, self.base / "second-link.json")
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(self.root, index)

    def test_undeclared_file_and_directory_are_rejected(self) -> None:
        record = completed_record(operation_key(fixed_point(), 0), root=self.root)
        index = Phase3RawAuditRunIndex.create((record,))
        (self.root / "extra.bin").write_bytes(b"extra")
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(self.root, index)
        (self.root / "extra.bin").unlink()
        (self.root / "empty-extra-directory").mkdir()
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(self.root, index)

    def test_missing_size_changed_and_digest_changed_files_are_rejected(self) -> None:
        for mutation in ("missing", "size", "digest"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory(
                    prefix="kvbench-phase3-raw-mutation-"
                ) as temporary:
                    root = Path(temporary) / "evidence"
                    root.mkdir(mode=0o700)
                    root.chmod(0o700)
                    record = completed_record(
                        operation_key(fixed_point(), 0), root=root
                    )
                    index = Phase3RawAuditRunIndex.create((record,))
                    victim = root.joinpath(*Path(record.files[0].path).parts)
                    original = victim.read_bytes()
                    if mutation == "missing":
                        victim.unlink()
                    elif mutation == "size":
                        victim.write_bytes(original + b"x")
                    else:
                        replacement = bytes(
                            [original[0] ^ 0x01]
                        ) + original[1:]
                        victim.write_bytes(replacement)
                    with self.assertRaises(Phase3RawAuditEvidenceError):
                        ingest_phase3_raw_audit_evidence(root, index)

    def test_nonregular_declared_entry_is_rejected(self) -> None:
        operation = operation_key(fixed_point(), 0)
        declaration = Phase3RawAuditFile(
            schema_version=PHASE3_RAW_AUDIT_FILE_SCHEMA_VERSION,
            path="collector.pipe",
            kind="collector_pipe",
            sha256=hashlib.sha256(b"").hexdigest(),
            size_bytes=0,
        )
        record = Phase3RawAuditOperationRecord(
            schema_version=PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
            operation=operation,
            status=RAW_AUDIT_STATUS_FAILED,
            failure_reason="collector_failed",
            files=(declaration,),
        )
        index = Phase3RawAuditRunIndex.create((record,))
        os.mkfifo(self.root / declaration.path)
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(self.root, index)

    def test_private_real_root_is_required(self) -> None:
        record = completed_record(operation_key(fixed_point(), 0), root=self.root)
        index = Phase3RawAuditRunIndex.create((record,))
        self.root.chmod(0o755)
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(self.root, index)
        self.root.chmod(0o700)
        link = self.base / "evidence-link"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(link, index)

    def test_file_and_total_read_limits_are_enforced(self) -> None:
        operation = operation_key(fixed_point(), 0)
        payload = b"12345678"
        record = failed_record(
            operation,
            root=self.root,
            payloads=(("partial", "partial.bin", payload),),
        )
        index = Phase3RawAuditRunIndex.create((record,))
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(
                self.root,
                index,
                max_file_size_bytes=4,
            )
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(
                self.root,
                index,
                max_total_size_bytes=4,
            )
        with self.assertRaises(ValueError):
            ingest_phase3_raw_audit_evidence(
                self.root,
                index,
                max_file_size_bytes=MAX_RAW_AUDIT_FILE_SIZE_BYTES + 1,
            )
        with self.assertRaises(ValueError):
            ingest_phase3_raw_audit_evidence(
                self.root,
                index,
                max_total_size_bytes=MAX_RAW_AUDIT_RUN_SIZE_BYTES + 1,
            )

    def test_intermediate_symlink_in_root_path_is_rejected(self) -> None:
        real_parent = self.base / "real-parent"
        real_parent.mkdir(mode=0o700)
        real_root = real_parent / "evidence"
        real_root.mkdir(mode=0o700)
        real_root.chmod(0o700)
        record = completed_record(
            operation_key(fixed_point(), 0),
            root=real_root,
        )
        index = Phase3RawAuditRunIndex.create((record,))
        linked_parent = self.base / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(linked_parent / "evidence", index)

    def test_descriptor_ingestion_pins_renamed_root_and_keeps_caller_fd(
        self,
    ) -> None:
        record = completed_record(
            operation_key(fixed_point(), 0),
            root=self.root,
        )
        index = Phase3RawAuditRunIndex.create((record,))
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        moved = self.base / "moved-evidence"
        self.root.rename(moved)
        self.root.symlink_to(moved, target_is_directory=True)
        try:
            offset_before = os.lseek(descriptor, 0, os.SEEK_CUR)
            observed = ingest_phase3_raw_audit_evidence_fd(descriptor, index)
            self.assertEqual(set(observed), {item.path for item in record.files})
            self.assertEqual(os.lseek(descriptor, 0, os.SEEK_CUR), offset_before)
            os.fstat(descriptor)
            with self.assertRaises(Phase3RawAuditEvidenceError):
                ingest_phase3_raw_audit_evidence(self.root, index)
        finally:
            os.close(descriptor)

    def test_descriptor_ingestion_rejects_bad_fd_owner_and_root_mode(self) -> None:
        record = completed_record(
            operation_key(fixed_point(), 0),
            root=self.root,
        )
        index = Phase3RawAuditRunIndex.create((record,))
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with self.assertRaises(Phase3RawAuditEvidenceError):
                ingest_phase3_raw_audit_evidence_fd(
                    descriptor,
                    index,
                    expected_owner_uid=os.geteuid() + 1,
                )
            os.fstat(descriptor)
            self.root.chmod(0o755)
            with self.assertRaises(Phase3RawAuditEvidenceError):
                ingest_phase3_raw_audit_evidence_fd(descriptor, index)
        finally:
            os.close(descriptor)
            self.root.chmod(0o700)

        ordinary_file = self.base / "ordinary-file"
        ordinary_file.write_bytes(b"not a directory")
        file_descriptor = os.open(ordinary_file, os.O_RDONLY)
        try:
            with self.assertRaises(Phase3RawAuditEvidenceError):
                ingest_phase3_raw_audit_evidence_fd(file_descriptor, index)
        finally:
            os.close(file_descriptor)
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence_fd(file_descriptor, index)
        with self.assertRaises(TypeError):
            ingest_phase3_raw_audit_evidence_fd(True, index)  # type: ignore[arg-type]

    def test_ingestion_enforces_configured_and_hard_file_count_limits(self) -> None:
        record = failed_record(
            operation_key(fixed_point(), 0),
            root=self.root,
            payloads=(
                ("partial_a", "partial/a.bin", b"a"),
                ("partial_b", "partial/b.bin", b"b"),
            ),
        )
        index = Phase3RawAuditRunIndex.create((record,))
        with self.assertRaises(Phase3RawAuditEvidenceError):
            ingest_phase3_raw_audit_evidence(
                self.root,
                index,
                max_file_count=1,
            )
        with self.assertRaises(ValueError):
            ingest_phase3_raw_audit_evidence(
                self.root,
                index,
                max_file_count=MAX_RAW_AUDIT_FILES_PER_RUN + 1,
            )
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with self.assertRaises(Phase3RawAuditEvidenceError):
                ingest_phase3_raw_audit_evidence_fd(
                    descriptor,
                    index,
                    max_file_count=1,
                )
        finally:
            os.close(descriptor)


if __name__ == "__main__":
    unittest.main()

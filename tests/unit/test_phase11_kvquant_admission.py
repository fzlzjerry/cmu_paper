"""Focused strict Phase 11 KVQuant admission-schema tests."""

from __future__ import annotations

import dataclasses
import json
import unittest

from kvbench.schema import (
    ClaimEligibility,
    GraphMode,
    MeasurementScope,
    MethodAdmissionEvidenceReference,
    QualityExecutionState,
    QualityValidationState,
    RunnerKind,
    RunStatus,
    canonical_json_bytes,
)
from kvbench.schema.config import MethodName
from kvbench.schema.phase3 import GateDisposition
from kvbench.schema.phase11 import (
    PHASE11_ACCOUNTING_CONTEXTS,
    PHASE11_ADMISSION_CHECK_IDS,
    PHASE11_AGGREGATE_PATCH_SHA256,
    PHASE11_ALLOCATION_AUDIT_CONTEXTS,
    PHASE11_AUTHORIZED_CONTAINER_DIGEST,
    PHASE11_BOUNDED_POINT_SIGNATURES,
    PHASE11_CALIBRATION_ID,
    PHASE11_CALIBRATION_ROOT,
    PHASE11_CONFIGURATIONS,
    PHASE11_CORRECTED_COMMIT,
    PHASE11_CORRECTED_CUDA_SHA256,
    PHASE11_CORRECTED_TREE,
    PHASE11_DECISION_0021_PATCH_SHA256,
    PHASE11_DECISIONS,
    PHASE11_EXECUTION_SOURCE_IDENTIFIER,
    PHASE11_EXTENSION_SHA256,
    PHASE11_FIXTURE_CASES,
    PHASE11_FIXTURE_ID,
    PHASE11_FIXTURE_ROOT,
    PHASE11_GRAPH_POINT_SIGNATURES,
    PHASE11_HISTORICAL_FIXTURE_ID,
    PHASE11_HISTORICAL_FIXTURE_ROOT,
    PHASE11_METHOD_IDENTIFIER,
    PHASE11_SANITIZER_CASES,
    PHASE11_UPSTREAM_BASE_COMMIT,
    PHASE11_UPSTREAM_BASE_TREE,
    Phase11AdmissionCheck,
    Phase11AdmissionGates,
    Phase11AllocationEvidence,
    Phase11Authority,
    Phase11ByteAccounting,
    Phase11ByteBreakdown,
    Phase11ExecutionPathEvidence,
    Phase11FixtureEvidence,
    Phase11GraphEvidence,
    Phase11MethodAdmissionReport,
    Phase11MethodConfiguration,
    Phase11RunPoint,
    Phase11SanitizerEvidence,
    require_exact_phase11_grid,
)


def _authority() -> Phase11Authority:
    return Phase11Authority(
        method_identifier=PHASE11_METHOD_IDENTIFIER,
        execution_source_identifier=PHASE11_EXECUTION_SOURCE_IDENTIFIER,
        upstream_base_commit=PHASE11_UPSTREAM_BASE_COMMIT,
        upstream_base_tree=PHASE11_UPSTREAM_BASE_TREE,
        decision_0021_patch_sha256=PHASE11_DECISION_0021_PATCH_SHA256,
        aggregate_patch_sha256=PHASE11_AGGREGATE_PATCH_SHA256,
        corrected_commit=PHASE11_CORRECTED_COMMIT,
        corrected_tree=PHASE11_CORRECTED_TREE,
        corrected_cuda_sha256=PHASE11_CORRECTED_CUDA_SHA256,
        extension_sha256=PHASE11_EXTENSION_SHA256,
        decisions=PHASE11_DECISIONS,
        calibration_id=PHASE11_CALIBRATION_ID,
        calibration_root=PHASE11_CALIBRATION_ROOT,
        historical_fixture_id=PHASE11_HISTORICAL_FIXTURE_ID,
        historical_fixture_root=PHASE11_HISTORICAL_FIXTURE_ROOT,
        fixture_id=PHASE11_FIXTURE_ID,
        fixture_root=PHASE11_FIXTURE_ROOT,
        authorized_container_digest=PHASE11_AUTHORIZED_CONTAINER_DIGEST,
    )


def _configuration(configuration: str) -> Phase11MethodConfiguration:
    return Phase11MethodConfiguration(
        configuration=configuration,
        bit_width={"kvq4": 4, "kvq3": 3, "kvq2": 2}[configuration],
        layers=32,
        batch_size=1,
        query_heads=32,
        kv_heads=8,
        groups=4,
        head_dim=128,
        interface_dtype="bfloat16",
        sink_tokens=5,
        key_cap=12,
        value_cap=12,
        sparse_value_dtype="float32",
        sparse_index_dtype="int32",
        key_semantics="pre_rope",
        sink_key_semantics="attention_ready",
        query_to_kv_mapping="query_head//4",
    )


def _breakdown() -> Phase11ByteBreakdown:
    return Phase11ByteBreakdown(
        dense_k_payload=100,
        dense_v_payload=100,
        key_metadata=10,
        value_metadata=10,
        key_sparse_values=48,
        key_sparse_indices=48,
        value_sparse_values=48,
        value_sparse_indices=48,
        active_count_mask=8,
        sink_k=20,
        sink_v=20,
        staging=30,
        padding_alignment=2,
        persistent_workspace=8,
    )


def _accounting(
    configuration: str,
    context: int,
) -> Phase11ByteAccounting:
    breakdown = _breakdown()
    return Phase11ByteAccounting(
        configuration=configuration,
        capacity=context,
        active_context=context,
        allocated_bytes=breakdown.total,
        predicted_allocated_bytes=breakdown.total,
        active_storage_bytes=breakdown.total,
        logical_bf16_allocated_bytes=context * 4096,
        logical_bf16_active_bytes=context * 4096,
        rho_alloc=breakdown.total / (context * 4096),
        r_alloc=(context * 4096) / breakdown.total,
        predicted_relative_error=0.0,
        temporary_peak_bytes=32,
        breakdown=breakdown,
        r_hbm=None,
    )


def _fixture_evidence() -> Phase11FixtureEvidence:
    return Phase11FixtureEvidence(
        fixture_id=PHASE11_FIXTURE_ID,
        fixture_root=PHASE11_FIXTURE_ROOT,
        cases=PHASE11_FIXTURE_CASES,
        input_and_pre_rope_exact=True,
        dense_payload_exact=True,
        metadata_exact=True,
        sparse_values_indices_counts_exact=True,
        unused_slots_exact=True,
        sink_exact=True,
        store_exact=True,
        append_exact=True,
        byte_breakdown_exact=True,
        decode_atol=0.01,
        decode_rtol=0.01,
        decode_within_tolerance=True,
        finite_output=True,
        kvq4_phase10_byte_identical=True,
        kvq2_phase10_byte_identical=True,
        kvq3_decision_0025_corrected=True,
        reuse_proof_valid=True,
    )


def _path_evidence(configuration: str) -> Phase11ExecutionPathEvidence:
    return Phase11ExecutionPathEvidence(
        configuration=configuration,
        caller_owned_outputs=True,
        device_resident_value_parameters=True,
        current_cuda_stream=True,
        corrected_kvq3_pack=True,
        deterministic_q4_value_decode=True,
        caller_owned_q4_value_workspace=True,
        fixed_order_q4_value_reduction=True,
        direct_compressed_decode=True,
        native_gqa=True,
        value_fixed_12=True,
        no_cpu_topk=True,
        no_dynamic_sparse_allocation=True,
        no_tensor_to_host=True,
        no_host_synchronization=True,
        no_complete_prefix_materialization=True,
        no_gqa_expansion=True,
        no_repeat_kv=True,
        no_repeat_interleave=True,
        no_query_head_sized_cache=True,
        no_measured_torch_cat=True,
        stable_kernel_path=True,
        no_backend_fallback=True,
    )


def _allocation_evidence(configuration: str) -> Phase11AllocationEvidence:
    return Phase11AllocationEvidence(
        configuration=configuration,
        audited_contexts=PHASE11_ALLOCATION_AUDIT_CONTEXTS[configuration],
        cache_growth_bytes=0,
        dynamic_sparse_allocation_bytes=0,
        unknown_allocation_bytes=0,
        full_prefix_allocation_bytes=0,
        gqa_expanded_allocation_bytes=0,
        persistent_allocated_delta=0,
        persistent_reserved_delta=0,
        dense_pointer_stable=True,
        metadata_pointer_stable=True,
        sparse_pointer_stable=True,
        sink_pointer_stable=True,
        staging_pointer_stable=True,
        workspace_pointer_stable=True,
    )


def _graph_evidence(
    configuration: str,
    context: int,
) -> Phase11GraphEvidence:
    return Phase11GraphEvidence(
        configuration=configuration,
        context_length=context,
        capture_passed=True,
        replay_passed=True,
        eager_graph_agreement=True,
        repeated_replay_stable=True,
        pointer_stable=True,
        replay_allocation_events=0,
        persistent_allocated_delta=0,
        persistent_reserved_delta=0,
        eager_fallback=False,
    )


def _sanitizer_evidence() -> Phase11SanitizerEvidence:
    return Phase11SanitizerEvidence(
        cases=PHASE11_SANITIZER_CASES,
        container_digest=PHASE11_AUTHORIZED_CONTAINER_DIGEST,
        corrected_tree=PHASE11_CORRECTED_TREE,
        extension_sha256=PHASE11_EXTENSION_SHA256,
        command_argv=("compute-sanitizer", "--tool", "memcheck"),
        tool_version="Compute Sanitizer 2026.1",
        stdout_sha256="8" * 64,
        stderr_sha256="9" * 64,
        memory_errors=0,
        leaked_allocations=0,
        unsupported_architecture_fallback=False,
    )


def _bounded_runs() -> tuple[Phase11RunPoint, ...]:
    return tuple(
        Phase11RunPoint(
            run_id=f"phase11-admission-{index:02d}",
            configuration=configuration,
            runner_kind=runner_kind,
            graph_mode=graph_mode,
            batch_size=1,
            context_length=context,
            output_steps=output_steps,
            status=RunStatus.COMPLETED,
            manifest_sha256=f"{index + 1:064x}",
            quality_status=QualityValidationState.UNVALIDATED,
            claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
            performance_claim_eligible=False,
            measurement_scope=MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION,
            speedup_calculated=False,
        )
        for index, (
            configuration,
            runner_kind,
            graph_mode,
            context,
            output_steps,
        ) in enumerate(PHASE11_BOUNDED_POINT_SIGNATURES)
    )


def _gates(
    g2_kvq: GateDisposition = GateDisposition.PASS,
) -> Phase11AdmissionGates:
    return Phase11AdmissionGates(
        g0=GateDisposition.PASS,
        g1=GateDisposition.PASS,
        g2_tq=GateDisposition.PASS,
        g2_kivi=GateDisposition.PASS,
        g2_kvq=g2_kvq,
        global_g2=GateDisposition.NOT_EVALUATED,
        g3=GateDisposition.NOT_EVALUATED,
        g4=GateDisposition.NOT_EVALUATED,
        g5=GateDisposition.NOT_EVALUATED,
        full_scan_state="CLOSED",
    )


def _report() -> Phase11MethodAdmissionReport:
    evidence = MethodAdmissionEvidenceReference(
        evidence_id="phase11_admission_evidence",
        path="artifacts/phase11/validation/evidence.json",
        sha256="a" * 64,
    )
    return Phase11MethodAdmissionReport(
        schema_version=Phase11MethodAdmissionReport.SCHEMA_VERSION,
        created_at_utc="2026-07-29T10:00:00Z",
        status=GateDisposition.PASS,
        method_name=MethodName.KVQUANT,
        authority=_authority(),
        configurations=tuple(
            _configuration(configuration)
            for configuration in PHASE11_CONFIGURATIONS
        ),
        admitted_configurations=PHASE11_CONFIGURATIONS,
        method_fingerprints={
            configuration: f"{index + 1:064x}"
            for index, configuration in enumerate(PHASE11_CONFIGURATIONS)
        },
        cache_layout_fingerprints={
            configuration: f"{index + 10:064x}"
            for index, configuration in enumerate(PHASE11_CONFIGURATIONS)
        },
        adapter_version="phase11_kvquant_1",
        adapter_source_sha256="b" * 64,
        byte_accounting=tuple(
            _accounting(configuration, context)
            for configuration in PHASE11_CONFIGURATIONS
            for context in PHASE11_ACCOUNTING_CONTEXTS
        ),
        fixture_evidence=_fixture_evidence(),
        execution_path_evidence=tuple(
            _path_evidence(configuration)
            for configuration in PHASE11_CONFIGURATIONS
        ),
        allocation_evidence=tuple(
            _allocation_evidence(configuration)
            for configuration in PHASE11_CONFIGURATIONS
        ),
        graph_evidence=tuple(
            _graph_evidence(configuration, context)
            for configuration, context in PHASE11_GRAPH_POINT_SIGNATURES
        ),
        sanitizer_evidence=_sanitizer_evidence(),
        bounded_runs=_bounded_runs(),
        checks=tuple(
            Phase11AdmissionCheck(
                check_id=check_id,
                status=GateDisposition.PASS,
                summary=f"{check_id} passed",
                evidence_ids=(evidence.evidence_id,),
            )
            for check_id in PHASE11_ADMISSION_CHECK_IDS
        ),
        evidence_references=(evidence,),
        gates=_gates(),
        blockers=(),
        local_root_digest="c" * 64,
        r2_uri=f"r2://kvbench-artifacts/kvbench/sha256/{'c' * 64}/",
        complete_last=True,
        checksums_valid=True,
        bucket_lock_identity="phase11-indefinite-lock",
        clean_retrieval=True,
        historical_evidence_unchanged=True,
        existing_methods_unchanged=True,
        measurement_container_unchanged=True,
        claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
        quality_status=QualityValidationState.UNVALIDATED,
        quality_execution=QualityExecutionState.LOCKED,
        performance_claim_eligible=False,
        performance_data_frozen=False,
        quality_benchmark_executed=False,
        speedup_calculated=False,
        r_hbm=None,
        measurement_scope=MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION,
        creation_git_sha="d" * 40,
    )


class Phase11KVQuantAdmissionSchemaTests(unittest.TestCase):
    def test_exact_authority_and_configuration(self) -> None:
        self.assertEqual(_authority().fixture_root, PHASE11_FIXTURE_ROOT)
        self.assertEqual(
            tuple(
                _configuration(configuration).bit_width
                for configuration in PHASE11_CONFIGURATIONS
            ),
            (4, 3, 2),
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(_authority(), aggregate_patch_sha256="0" * 64)
        with self.assertRaises(ValueError):
            dataclasses.replace(_configuration("kvq3"), bit_width=4)

    def test_byte_accounting_is_physical_reciprocal_and_no_hbm(self) -> None:
        accounting = _accounting("kvq4", 128)
        self.assertEqual(accounting.breakdown.total, accounting.allocated_bytes)
        self.assertAlmostEqual(
            accounting.rho_alloc * accounting.r_alloc,
            1.0,
            places=12,
        )
        self.assertIsNone(accounting.r_hbm)
        with self.assertRaises(ValueError):
            dataclasses.replace(accounting, predicted_allocated_bytes=700)
        payload = accounting.to_dict()
        payload["r_hbm"] = 1.0
        with self.assertRaises(Exception):
            Phase11ByteAccounting.from_dict(payload)

    def test_exact_nine_point_grid_only(self) -> None:
        points = _bounded_runs()
        self.assertEqual(require_exact_phase11_grid(points), points)
        with self.assertRaises(ValueError):
            require_exact_phase11_grid((points[1], points[0], *points[2:]))
        with self.assertRaises(ValueError):
            require_exact_phase11_grid((*points, points[-1]))

    def test_pass_report_binds_all_local_and_durable_evidence(self) -> None:
        report = _report()
        self.assertIs(report.gates.g2_kvq, GateDisposition.PASS)
        self.assertEqual(report.admitted_configurations, PHASE11_CONFIGURATIONS)
        self.assertEqual(len(report.bounded_runs), 9)
        self.assertEqual(len(report.byte_accounting), 15)
        self.assertFalse(report.speedup_calculated)
        self.assertIsNone(report.r_hbm)
        parsed = Phase11MethodAdmissionReport.from_dict(
            json.loads(canonical_json_bytes(report))
        )
        self.assertEqual(parsed, report)

    def test_g2_pass_rejects_missing_publication_or_failed_run(self) -> None:
        report = _report()
        with self.assertRaises(ValueError):
            dataclasses.replace(report, clean_retrieval=False)
        failed_runs = (
            dataclasses.replace(
                report.bounded_runs[0],
                status=RunStatus.RUNTIME_FAILED,
            ),
            *report.bounded_runs[1:],
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(report, bounded_runs=failed_runs)
        with self.assertRaises(ValueError):
            dataclasses.replace(report, checksums_valid=False)

    def test_non_pass_requires_matching_gate_and_concrete_blocker(self) -> None:
        report = _report()
        blocked_checks = (
            dataclasses.replace(
                report.checks[0],
                status=GateDisposition.BLOCKED,
            ),
            *report.checks[1:],
        )
        blocked = dataclasses.replace(
            report,
            status=GateDisposition.BLOCKED,
            checks=blocked_checks,
            gates=_gates(GateDisposition.BLOCKED),
            blockers=("graph replay allocation was nonzero",),
            local_root_digest=None,
            r2_uri=None,
            complete_last=False,
            checksums_valid=False,
            bucket_lock_identity=None,
            clean_retrieval=False,
        )
        self.assertIs(blocked.status, GateDisposition.BLOCKED)
        with self.assertRaises(ValueError):
            dataclasses.replace(blocked, blockers=())

    def test_strict_report_rejects_unknown_fields_and_global_gate_drift(self) -> None:
        payload = _report().to_dict()
        payload["speedup"] = 1.5
        with self.assertRaises(Exception):
            Phase11MethodAdmissionReport.from_dict(payload)
        with self.assertRaises(ValueError):
            dataclasses.replace(
                _gates(),
                global_g2=GateDisposition.PASS,
            )


if __name__ == "__main__":
    unittest.main()

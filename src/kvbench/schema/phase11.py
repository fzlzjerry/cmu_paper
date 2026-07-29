"""Strict, narrow records for Phase 11 KVQuant admission."""

from __future__ import annotations

import dataclasses
from datetime import datetime
import math
from typing import ClassVar, Literal

from kvbench.schema.base import (
    ClaimEligibility,
    GraphMode,
    MeasurementScope,
    QualityExecutionState,
    QualityValidationState,
    RunnerKind,
    RunStatus,
    StrictModel,
    require_git_sha,
    require_identifier,
    require_oci_digest,
    require_relative_path,
    require_run_id,
    require_schema,
    require_sha256,
    require_utc_timestamp,
)
from kvbench.schema.config import MethodName
from kvbench.schema.method_admission import MethodAdmissionEvidenceReference
from kvbench.schema.phase3 import GateDisposition


PHASE11_METHOD_IDENTIFIER = "kvquant_gqa_upstream_patch_v1"
PHASE11_EXECUTION_SOURCE_IDENTIFIER = "kvquant_gqa_graphsafe_kvq3_v2"
PHASE11_UPSTREAM_BASE_COMMIT = "57a238357f0ffe50084670fcd5781c9848f80ea2"
PHASE11_UPSTREAM_BASE_TREE = "094e0f736f77ee327e5350cbd1eefb1c936aa77b"
PHASE11_DECISION_0021_PATCH_SHA256 = (
    "db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6"
)
PHASE11_AGGREGATE_PATCH_SHA256 = (
    "23a15db86790299392412c3ce2da7d971f4f073cfaf6839d82d3746c8b56b551"
)
PHASE11_CORRECTED_COMMIT = "0d9df350bd1788284e1ce76a8bf6e886beca5efa"
PHASE11_CORRECTED_TREE = "a85cf7bf093982a4bf89c33d4e6794d9a85f846d"
PHASE11_CORRECTED_CUDA_SHA256 = (
    "07ea018378e10ee80e0485e42225ab9903adcee0879af27c621289f147fabba1"
)
PHASE11_EXTENSION_SHA256 = (
    "46c41aad8f56d58608d4c1273bd3a72fd36c8f69f9ca2c5a046f0c811631bf51"
)
PHASE11_CALIBRATION_ID = "kvqcal-cdb724c806d64d095c040d2673a987a3"
PHASE11_CALIBRATION_ROOT = (
    "8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf"
)
PHASE11_HISTORICAL_FIXTURE_ID = "kvqref-a50af6511c314b6394e58a7f81ceefb8"
PHASE11_HISTORICAL_FIXTURE_ROOT = (
    "32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab"
)
PHASE11_FIXTURE_ID = "kvqref-2e0a0e9022c50cbc6fb497d88cae973e"
PHASE11_FIXTURE_ROOT = (
    "c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec"
)
PHASE11_AUTHORIZED_CONTAINER_DIGEST = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
PHASE11_DECISIONS = ("0021", "0023", "0024", "0025")
PHASE11_CONFIGURATIONS = ("kvq4", "kvq3", "kvq2")
PHASE11_ACCOUNTING_CONTEXTS = (5, 17, 18, 128, 4096)
PHASE11_ALLOCATION_AUDIT_CONTEXTS = {
    "kvq4": (128, 4096, 17),
    "kvq3": (128,),
    "kvq2": (128,),
}
PHASE11_DECODE_ATOL = 0.01
PHASE11_DECODE_RTOL = 0.01
PHASE11_RECIPROCAL_TOLERANCE = 1e-9

PHASE11_FIXTURE_CASES = tuple(
    f"{configuration}/{case}"
    for configuration in PHASE11_CONFIGURATIONS
    for case in (
        "key_zero_value_fixed12",
        "key_few_value_fixed12",
        "key_cap_value_fixed12",
    )
)

PHASE11_BOUNDED_POINT_SIGNATURES = (
    ("kvq4", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
    ("kvq4", RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 128, 1),
    ("kvq3", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
    ("kvq3", RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 128, 1),
    ("kvq2", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
    ("kvq2", RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 128, 1),
    ("kvq4", RunnerKind.FIXED_L, GraphMode.EAGER, 4096, 1),
    ("kvq4", RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 4096, 1),
    ("kvq4", RunnerKind.GROWING_CONTEXT, GraphMode.EAGER, 17, 4),
)

PHASE11_GRAPH_POINT_SIGNATURES = (
    ("kvq4", 128),
    ("kvq3", 128),
    ("kvq2", 128),
    ("kvq4", 4096),
)

PHASE11_SANITIZER_CASES = (
    "kvq4_cap_sparse",
    "kvq3_corrected",
    "kvq2",
    "sink",
    "native_32q_8kv_decode",
    "fixed_l_overwrite",
    "graph_replay",
)

PHASE11_ADMISSION_CHECK_IDS = (
    "fixture_conformance",
    "byte_accounting",
    "sparse_contract",
    "sink_storage",
    "store_append_correctness",
    "direct_compressed_decode",
    "native_gqa",
    "execution_path",
    "no_dynamic_or_unknown_allocation",
    "no_host_synchronization",
    "graph_capture_replay",
    "graph_zero_replay_allocation",
    "compute_sanitizer",
    "bounded_admission_grid",
    "immutable_checksums",
    "durable_publication",
    "clean_retrieval",
)


def _require_nonnegative_int(value: int, *, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


def _validate_phase11_lifecycle(
    *,
    status: RunStatus,
    created_at_utc: str,
    started_at_utc: str | None,
    finished_at_utc: str | None,
    inventory_path: str | None,
    failure_reason: str | None,
) -> None:
    require_utc_timestamp(created_at_utc, field_name="created_at_utc")
    if status is RunStatus.CREATED:
        if any(
            value is not None
            for value in (
                started_at_utc,
                finished_at_utc,
                inventory_path,
                failure_reason,
            )
        ):
            raise ValueError("created Phase 11 manifest has later lifecycle fields")
    elif status in {RunStatus.RUNNING, RunStatus.FINALIZING}:
        if started_at_utc is None:
            raise ValueError("nonterminal Phase 11 manifest requires started_at_utc")
        require_utc_timestamp(started_at_utc, field_name="started_at_utc")
        if any(
            value is not None
            for value in (finished_at_utc, inventory_path, failure_reason)
        ):
            raise ValueError(
                "nonterminal Phase 11 manifest has terminal-only fields"
            )
    else:
        if started_at_utc is None or finished_at_utc is None:
            raise ValueError("terminal Phase 11 manifest requires timestamps")
        require_utc_timestamp(started_at_utc, field_name="started_at_utc")
        require_utc_timestamp(finished_at_utc, field_name="finished_at_utc")
        if inventory_path != "artifact_inventory.json":
            raise ValueError("terminal Phase 11 manifest has wrong inventory")
        if status is RunStatus.COMPLETED and failure_reason is not None:
            raise ValueError("completed Phase 11 manifest cannot carry failure")
        if status.is_failure and not failure_reason:
            raise ValueError("failed Phase 11 manifest requires a reason")
    timestamps = [
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in (created_at_utc, started_at_utc, finished_at_utc)
        if value is not None
    ]
    if timestamps != sorted(timestamps):
        raise ValueError("Phase 11 lifecycle timestamps are out of order")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11RunManifest(StrictModel):
    """One append-only local Phase 11 admission-evidence bundle."""

    schema_version: str
    artifact_schema_version: str
    run_id: str
    status: RunStatus
    created_at_utc: str
    started_at_utc: str | None
    finished_at_utc: str | None
    run_kind: Literal["phase11_admission"]
    git_sha: str
    git_dirty: bool
    authority: "Phase11Authority"
    bounded_point_count: Literal[9]
    measurement_scope: MeasurementScope
    quality_status: QualityValidationState
    claim_eligibility: ClaimEligibility
    quality_execution: QualityExecutionState
    performance_claim_eligible: bool
    performance_data_frozen: bool
    quality_benchmark_executed: bool
    speedup_calculated: bool
    r_hbm: None
    full_scan_state: Literal["CLOSED"]
    g2_kvq_state: Literal["NOT_EVALUATED_PUBLICATION_PENDING"]
    global_g2_g5_state: Literal["NOT_EVALUATED"]
    inventory_path: str | None
    failure_reason: str | None

    SCHEMA_VERSION: ClassVar[str] = (
        "kvbench-phase11-kvquant-run-manifest-1.0.0"
    )
    ARTIFACT_SCHEMA_VERSION: ClassVar[str] = "kvbench-artifacts-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_schema(
            self.artifact_schema_version,
            self.ARTIFACT_SCHEMA_VERSION,
        )
        require_run_id(self.run_id)
        require_git_sha(self.git_sha)
        if self.git_dirty:
            raise ValueError("Phase 11 admission requires a clean Git tree")
        if (
            self.authority.authorized_container_digest
            != PHASE11_AUTHORIZED_CONTAINER_DIGEST
            or self.measurement_scope
            is not MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
            or self.quality_status is not QualityValidationState.UNVALIDATED
            or self.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            or self.quality_execution is not QualityExecutionState.LOCKED
            or self.performance_claim_eligible
            or self.performance_data_frozen
            or self.quality_benchmark_executed
            or self.speedup_calculated
            or self.r_hbm is not None
        ):
            raise ValueError("Phase 11 local bundle must remain a non-claim")
        if self.inventory_path is not None:
            require_relative_path(
                self.inventory_path,
                field_name="inventory_path",
            )
        _validate_phase11_lifecycle(
            status=self.status,
            created_at_utc=self.created_at_utc,
            started_at_utc=self.started_at_utc,
            finished_at_utc=self.finished_at_utc,
            inventory_path=self.inventory_path,
            failure_reason=self.failure_reason,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11Authority(StrictModel):
    """Exact immutable source, calibration, fixture, and container authority."""

    method_identifier: str
    execution_source_identifier: str
    upstream_base_commit: str
    upstream_base_tree: str
    decision_0021_patch_sha256: str
    aggregate_patch_sha256: str
    corrected_commit: str
    corrected_tree: str
    corrected_cuda_sha256: str
    extension_sha256: str
    decisions: tuple[str, ...]
    calibration_id: str
    calibration_root: str
    historical_fixture_id: str
    historical_fixture_root: str
    fixture_id: str
    fixture_root: str
    authorized_container_digest: str

    def __post_init__(self) -> None:
        require_git_sha(self.upstream_base_commit)
        require_git_sha(self.upstream_base_tree)
        require_git_sha(self.corrected_commit)
        require_git_sha(self.corrected_tree)
        for value, name in (
            (self.decision_0021_patch_sha256, "decision_0021_patch_sha256"),
            (self.aggregate_patch_sha256, "aggregate_patch_sha256"),
            (self.corrected_cuda_sha256, "corrected_cuda_sha256"),
            (self.extension_sha256, "extension_sha256"),
            (self.calibration_root, "calibration_root"),
            (self.historical_fixture_root, "historical_fixture_root"),
            (self.fixture_root, "fixture_root"),
        ):
            require_sha256(value, field_name=name)
        require_oci_digest(self.authorized_container_digest)
        for value, name in (
            (self.method_identifier, "method_identifier"),
            (self.execution_source_identifier, "execution_source_identifier"),
            (self.calibration_id, "calibration_id"),
            (self.historical_fixture_id, "historical_fixture_id"),
            (self.fixture_id, "fixture_id"),
        ):
            require_identifier(value, field_name=name)
        observed = (
            self.method_identifier,
            self.execution_source_identifier,
            self.upstream_base_commit,
            self.upstream_base_tree,
            self.decision_0021_patch_sha256,
            self.aggregate_patch_sha256,
            self.corrected_commit,
            self.corrected_tree,
            self.corrected_cuda_sha256,
            self.extension_sha256,
            self.decisions,
            self.calibration_id,
            self.calibration_root,
            self.historical_fixture_id,
            self.historical_fixture_root,
            self.fixture_id,
            self.fixture_root,
            self.authorized_container_digest,
        )
        expected = (
            PHASE11_METHOD_IDENTIFIER,
            PHASE11_EXECUTION_SOURCE_IDENTIFIER,
            PHASE11_UPSTREAM_BASE_COMMIT,
            PHASE11_UPSTREAM_BASE_TREE,
            PHASE11_DECISION_0021_PATCH_SHA256,
            PHASE11_AGGREGATE_PATCH_SHA256,
            PHASE11_CORRECTED_COMMIT,
            PHASE11_CORRECTED_TREE,
            PHASE11_CORRECTED_CUDA_SHA256,
            PHASE11_EXTENSION_SHA256,
            PHASE11_DECISIONS,
            PHASE11_CALIBRATION_ID,
            PHASE11_CALIBRATION_ROOT,
            PHASE11_HISTORICAL_FIXTURE_ID,
            PHASE11_HISTORICAL_FIXTURE_ROOT,
            PHASE11_FIXTURE_ID,
            PHASE11_FIXTURE_ROOT,
            PHASE11_AUTHORIZED_CONTAINER_DIGEST,
        )
        if observed != expected:
            raise ValueError("Phase 11 authority differs from the frozen record")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11MethodConfiguration(StrictModel):
    configuration: str
    bit_width: int
    layers: int
    batch_size: int
    query_heads: int
    kv_heads: int
    groups: int
    head_dim: int
    interface_dtype: Literal["bfloat16"]
    sink_tokens: int
    key_cap: int
    value_cap: int
    sparse_value_dtype: Literal["float32"]
    sparse_index_dtype: Literal["int32"]
    key_semantics: Literal["pre_rope"]
    sink_key_semantics: Literal["attention_ready"]
    query_to_kv_mapping: Literal["query_head//4"]

    def __post_init__(self) -> None:
        require_identifier(self.configuration, field_name="configuration")
        expected_bits = {"kvq4": 4, "kvq3": 3, "kvq2": 2}
        if (
            self.configuration not in expected_bits
            or self.bit_width != expected_bits[self.configuration]
            or (
                self.layers,
                self.batch_size,
                self.query_heads,
                self.kv_heads,
                self.groups,
                self.head_dim,
                self.sink_tokens,
                self.key_cap,
                self.value_cap,
            )
            != (32, 1, 32, 8, 4, 128, 5, 12, 12)
            or (
                self.interface_dtype,
                self.sparse_value_dtype,
                self.sparse_index_dtype,
                self.key_semantics,
                self.sink_key_semantics,
                self.query_to_kv_mapping,
            )
            != (
                "bfloat16",
                "float32",
                "int32",
                "pre_rope",
                "attention_ready",
                "query_head//4",
            )
        ):
            raise ValueError("Phase 11 KVQuant configuration or geometry differs")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11ByteBreakdown(StrictModel):
    dense_k_payload: int
    dense_v_payload: int
    key_metadata: int
    value_metadata: int
    key_sparse_values: int
    key_sparse_indices: int
    value_sparse_values: int
    value_sparse_indices: int
    active_count_mask: int
    sink_k: int
    sink_v: int
    staging: int
    padding_alignment: int
    persistent_workspace: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _require_nonnegative_int(
                getattr(self, field.name),
                field_name=field.name,
            )

    @property
    def total(self) -> int:
        return sum(
            int(getattr(self, field.name))
            for field in dataclasses.fields(self)
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11ByteAccounting(StrictModel):
    configuration: str
    capacity: int
    active_context: int
    allocated_bytes: int
    predicted_allocated_bytes: int
    active_storage_bytes: int
    logical_bf16_allocated_bytes: int
    logical_bf16_active_bytes: int
    rho_alloc: float
    r_alloc: float
    predicted_relative_error: float
    temporary_peak_bytes: int
    breakdown: Phase11ByteBreakdown
    r_hbm: None

    def __post_init__(self) -> None:
        if self.configuration not in PHASE11_CONFIGURATIONS:
            raise ValueError("unknown Phase 11 KVQuant configuration")
        for field_name in (
            "capacity",
            "active_context",
            "allocated_bytes",
            "predicted_allocated_bytes",
            "active_storage_bytes",
            "logical_bf16_allocated_bytes",
            "logical_bf16_active_bytes",
            "temporary_peak_bytes",
        ):
            _require_nonnegative_int(
                getattr(self, field_name),
                field_name=field_name,
            )
        if (
            self.capacity <= 0
            or self.active_context <= 0
            or self.active_context > self.capacity
            or self.allocated_bytes <= 0
            or self.predicted_allocated_bytes <= 0
            or self.active_storage_bytes > self.allocated_bytes
            or self.logical_bf16_allocated_bytes <= 0
            or self.logical_bf16_active_bytes <= 0
            or self.breakdown.total != self.allocated_bytes
            or self.r_hbm is not None
        ):
            raise ValueError("Phase 11 byte-accounting capacity or sum is invalid")
        expected_rho = self.allocated_bytes / self.logical_bf16_allocated_bytes
        expected_r = self.logical_bf16_allocated_bytes / self.allocated_bytes
        expected_error = abs(
            self.predicted_allocated_bytes - self.allocated_bytes
        ) / self.allocated_bytes
        for observed, expected, name in (
            (self.rho_alloc, expected_rho, "rho_alloc"),
            (self.r_alloc, expected_r, "r_alloc"),
            (
                self.predicted_relative_error,
                expected_error,
                "predicted_relative_error",
            ),
        ):
            if (
                not math.isfinite(observed)
                or observed < 0.0
                or not math.isclose(
                    observed,
                    expected,
                    rel_tol=PHASE11_RECIPROCAL_TOLERANCE,
                    abs_tol=PHASE11_RECIPROCAL_TOLERANCE,
                )
            ):
                raise ValueError(f"Phase 11 {name} is invalid")
        if (
            self.rho_alloc <= 0.0
            or self.r_alloc <= 0.0
            or abs(self.rho_alloc * self.r_alloc - 1.0)
            > PHASE11_RECIPROCAL_TOLERANCE
            or self.predicted_relative_error >= 0.01
        ):
            raise ValueError("Phase 11 ratios or prediction error are invalid")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11RunPoint(StrictModel):
    run_id: str
    configuration: str
    runner_kind: RunnerKind
    graph_mode: GraphMode
    batch_size: int
    context_length: int
    output_steps: int
    status: RunStatus
    manifest_sha256: str
    quality_status: QualityValidationState
    claim_eligibility: ClaimEligibility
    performance_claim_eligible: bool
    measurement_scope: MeasurementScope
    speedup_calculated: bool

    def __post_init__(self) -> None:
        require_run_id(self.run_id)
        require_sha256(self.manifest_sha256, field_name="manifest_sha256")
        if (
            (
                self.configuration,
                self.runner_kind,
                self.graph_mode,
                self.context_length,
                self.output_steps,
            )
            not in PHASE11_BOUNDED_POINT_SIGNATURES
            or self.batch_size != 1
            or not self.status.is_terminal
        ):
            raise ValueError("run is outside the bounded Phase 11 grid")
        if (
            self.quality_status is not QualityValidationState.UNVALIDATED
            or self.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            or self.performance_claim_eligible
            or self.measurement_scope
            is not MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
            or self.speedup_calculated
        ):
            raise ValueError("Phase 11 run must remain a non-claim")


def require_exact_phase11_grid(
    points: tuple[Phase11RunPoint, ...],
) -> tuple[Phase11RunPoint, ...]:
    signatures = tuple(
        (
            point.configuration,
            point.runner_kind,
            point.graph_mode,
            point.context_length,
            point.output_steps,
        )
        for point in points
    )
    if signatures != PHASE11_BOUNDED_POINT_SIGNATURES:
        raise ValueError("Phase 11 requires exactly the ordered nine-point grid")
    if len({point.run_id for point in points}) != len(points):
        raise ValueError("Phase 11 bounded run IDs must be unique")
    return points


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11FixtureEvidence(StrictModel):
    fixture_id: str
    fixture_root: str
    cases: tuple[str, ...]
    input_and_pre_rope_exact: bool
    dense_payload_exact: bool
    metadata_exact: bool
    sparse_values_indices_counts_exact: bool
    unused_slots_exact: bool
    sink_exact: bool
    store_exact: bool
    append_exact: bool
    byte_breakdown_exact: bool
    decode_atol: float
    decode_rtol: float
    decode_within_tolerance: bool
    finite_output: bool
    kvq4_phase10_byte_identical: bool
    kvq2_phase10_byte_identical: bool
    kvq3_decision_0025_corrected: bool
    reuse_proof_valid: bool

    def __post_init__(self) -> None:
        require_identifier(self.fixture_id, field_name="fixture_id")
        require_sha256(self.fixture_root, field_name="fixture_root")
        if (
            self.fixture_id != PHASE11_FIXTURE_ID
            or self.fixture_root != PHASE11_FIXTURE_ROOT
            or self.cases != PHASE11_FIXTURE_CASES
            or not math.isclose(self.decode_atol, PHASE11_DECODE_ATOL)
            or not math.isclose(self.decode_rtol, PHASE11_DECODE_RTOL)
            or not all(
                (
                    self.input_and_pre_rope_exact,
                    self.dense_payload_exact,
                    self.metadata_exact,
                    self.sparse_values_indices_counts_exact,
                    self.unused_slots_exact,
                    self.sink_exact,
                    self.store_exact,
                    self.append_exact,
                    self.byte_breakdown_exact,
                    self.decode_within_tolerance,
                    self.finite_output,
                    self.kvq4_phase10_byte_identical,
                    self.kvq2_phase10_byte_identical,
                    self.kvq3_decision_0025_corrected,
                    self.reuse_proof_valid,
                )
            )
        ):
            raise ValueError("Phase 11 fixture evidence is incomplete or drifted")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11ExecutionPathEvidence(StrictModel):
    configuration: str
    caller_owned_outputs: bool
    device_resident_value_parameters: bool
    current_cuda_stream: bool
    corrected_kvq3_pack: bool
    direct_compressed_decode: bool
    native_gqa: bool
    value_fixed_12: bool
    no_cpu_topk: bool
    no_dynamic_sparse_allocation: bool
    no_tensor_to_host: bool
    no_host_synchronization: bool
    no_complete_prefix_materialization: bool
    no_gqa_expansion: bool
    no_repeat_kv: bool
    no_repeat_interleave: bool
    no_query_head_sized_cache: bool
    no_measured_torch_cat: bool
    stable_kernel_path: bool
    no_backend_fallback: bool

    def __post_init__(self) -> None:
        if self.configuration not in PHASE11_CONFIGURATIONS or not all(
            getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "configuration"
        ):
            raise ValueError("Phase 11 execution-path evidence is incomplete")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11AllocationEvidence(StrictModel):
    configuration: str
    audited_contexts: tuple[int, ...]
    cache_growth_bytes: int
    dynamic_sparse_allocation_bytes: int
    unknown_allocation_bytes: int
    full_prefix_allocation_bytes: int
    gqa_expanded_allocation_bytes: int
    persistent_allocated_delta: int
    persistent_reserved_delta: int
    dense_pointer_stable: bool
    metadata_pointer_stable: bool
    sparse_pointer_stable: bool
    sink_pointer_stable: bool
    staging_pointer_stable: bool
    workspace_pointer_stable: bool

    def __post_init__(self) -> None:
        if self.configuration not in PHASE11_CONFIGURATIONS:
            raise ValueError("unknown Phase 11 allocation configuration")
        if (
            self.audited_contexts
            != PHASE11_ALLOCATION_AUDIT_CONTEXTS[self.configuration]
        ):
            raise ValueError("Phase 11 allocation contexts differ")
        for field_name in (
            "cache_growth_bytes",
            "dynamic_sparse_allocation_bytes",
            "unknown_allocation_bytes",
            "full_prefix_allocation_bytes",
            "gqa_expanded_allocation_bytes",
            "persistent_allocated_delta",
            "persistent_reserved_delta",
        ):
            if getattr(self, field_name) != 0:
                raise ValueError("Phase 11 allocation audit requires zero deltas")
        if not all(
            (
                self.dense_pointer_stable,
                self.metadata_pointer_stable,
                self.sparse_pointer_stable,
                self.sink_pointer_stable,
                self.staging_pointer_stable,
                self.workspace_pointer_stable,
            )
        ):
            raise ValueError("Phase 11 persistent pointers must remain stable")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11GraphEvidence(StrictModel):
    configuration: str
    context_length: int
    capture_passed: bool
    replay_passed: bool
    eager_graph_agreement: bool
    repeated_replay_stable: bool
    pointer_stable: bool
    replay_allocation_events: int
    persistent_allocated_delta: int
    persistent_reserved_delta: int
    eager_fallback: bool

    def __post_init__(self) -> None:
        if (
            (self.configuration, self.context_length)
            not in PHASE11_GRAPH_POINT_SIGNATURES
            or not all(
                (
                    self.capture_passed,
                    self.replay_passed,
                    self.eager_graph_agreement,
                    self.repeated_replay_stable,
                    self.pointer_stable,
                )
            )
            or self.replay_allocation_events != 0
            or self.persistent_allocated_delta != 0
            or self.persistent_reserved_delta != 0
            or self.eager_fallback
        ):
            raise ValueError("Phase 11 CUDA Graph evidence is incomplete")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11SanitizerEvidence(StrictModel):
    cases: tuple[str, ...]
    container_digest: str
    corrected_tree: str
    extension_sha256: str
    command_argv: tuple[str, ...]
    tool_version: str
    stdout_sha256: str
    stderr_sha256: str
    memory_errors: int
    leaked_allocations: int
    unsupported_architecture_fallback: bool

    def __post_init__(self) -> None:
        require_oci_digest(self.container_digest)
        require_git_sha(self.corrected_tree)
        require_sha256(self.extension_sha256, field_name="extension_sha256")
        require_sha256(self.stdout_sha256, field_name="stdout_sha256")
        require_sha256(self.stderr_sha256, field_name="stderr_sha256")
        if (
            self.cases != PHASE11_SANITIZER_CASES
            or self.container_digest != PHASE11_AUTHORIZED_CONTAINER_DIGEST
            or self.corrected_tree != PHASE11_CORRECTED_TREE
            or self.extension_sha256 != PHASE11_EXTENSION_SHA256
            or not self.command_argv
            or not all(item.strip() for item in self.command_argv)
            or not self.tool_version.strip()
            or self.memory_errors != 0
            or self.leaked_allocations != 0
            or self.unsupported_architecture_fallback
        ):
            raise ValueError("Phase 11 sanitizer evidence is incomplete")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11AdmissionCheck(StrictModel):
    check_id: str
    status: GateDisposition
    summary: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier(self.check_id, field_name="check_id")
        if self.check_id not in PHASE11_ADMISSION_CHECK_IDS:
            raise ValueError("unknown Phase 11 admission check")
        if not self.summary.strip() or not self.evidence_ids:
            raise ValueError("Phase 11 admission check requires summary and evidence")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Phase 11 check evidence IDs must be unique")
        for evidence_id in self.evidence_ids:
            require_identifier(evidence_id, field_name="evidence_id")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11AdmissionGates(StrictModel):
    g0: GateDisposition
    g1: GateDisposition
    g2_tq: GateDisposition
    g2_kivi: GateDisposition
    g2_kvq: GateDisposition
    global_g2: GateDisposition
    g3: GateDisposition
    g4: GateDisposition
    g5: GateDisposition
    full_scan_state: Literal["CLOSED"]

    def __post_init__(self) -> None:
        if (
            self.g0,
            self.g1,
            self.g2_tq,
            self.g2_kivi,
            self.global_g2,
            self.g3,
            self.g4,
            self.g5,
            self.full_scan_state,
        ) != (
            GateDisposition.PASS,
            GateDisposition.PASS,
            GateDisposition.PASS,
            GateDisposition.PASS,
            GateDisposition.NOT_EVALUATED,
            GateDisposition.NOT_EVALUATED,
            GateDisposition.NOT_EVALUATED,
            GateDisposition.NOT_EVALUATED,
            "CLOSED",
        ):
            raise ValueError("Phase 11 must preserve completed and global gates")
        if self.g2_kvq not in {
            GateDisposition.PASS,
            GateDisposition.PARTIAL,
            GateDisposition.BLOCKED,
            GateDisposition.FAIL,
        }:
            raise ValueError("G2-KVQ status must be terminal")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase11MethodAdmissionReport(StrictModel):
    """One G2-KVQ report joining the exact corrected admission evidence."""

    schema_version: str
    created_at_utc: str
    status: GateDisposition
    method_name: MethodName
    authority: Phase11Authority
    configurations: tuple[Phase11MethodConfiguration, ...]
    admitted_configurations: tuple[str, ...]
    method_fingerprints: dict[str, str]
    cache_layout_fingerprints: dict[str, str]
    adapter_version: str
    adapter_source_sha256: str
    byte_accounting: tuple[Phase11ByteAccounting, ...]
    fixture_evidence: Phase11FixtureEvidence
    execution_path_evidence: tuple[Phase11ExecutionPathEvidence, ...]
    allocation_evidence: tuple[Phase11AllocationEvidence, ...]
    graph_evidence: tuple[Phase11GraphEvidence, ...]
    sanitizer_evidence: Phase11SanitizerEvidence
    bounded_runs: tuple[Phase11RunPoint, ...]
    checks: tuple[Phase11AdmissionCheck, ...]
    evidence_references: tuple[MethodAdmissionEvidenceReference, ...]
    gates: Phase11AdmissionGates
    blockers: tuple[str, ...]
    local_root_digest: str | None
    r2_uri: str | None
    complete_last: bool
    checksums_valid: bool
    bucket_lock_identity: str | None
    clean_retrieval: bool
    historical_evidence_unchanged: bool
    existing_methods_unchanged: bool
    measurement_container_unchanged: bool
    claim_eligibility: ClaimEligibility
    quality_status: QualityValidationState
    quality_execution: QualityExecutionState
    performance_claim_eligible: bool
    performance_data_frozen: bool
    quality_benchmark_executed: bool
    speedup_calculated: bool
    r_hbm: None
    measurement_scope: MeasurementScope
    creation_git_sha: str

    SCHEMA_VERSION: ClassVar[str] = (
        "kvbench-phase11-kvquant-admission-report-1.0.0"
    )

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_utc_timestamp(self.created_at_utc)
        require_git_sha(self.creation_git_sha)
        if self.method_name is not MethodName.KVQUANT:
            raise ValueError("Phase 11 admission method must be KVQuant")
        require_identifier(self.adapter_version, field_name="adapter_version")
        require_sha256(
            self.adapter_source_sha256,
            field_name="adapter_source_sha256",
        )
        observed_configurations = tuple(
            item.configuration for item in self.configurations
        )
        if (
            observed_configurations != PHASE11_CONFIGURATIONS
            or len(set(observed_configurations)) != len(observed_configurations)
        ):
            raise ValueError("Phase 11 configuration records differ")
        for mapping, field_name in (
            (self.method_fingerprints, "method_fingerprint"),
            (self.cache_layout_fingerprints, "cache_layout_fingerprint"),
        ):
            if set(mapping) != set(PHASE11_CONFIGURATIONS):
                raise ValueError(f"Phase 11 {field_name} keys differ")
            for digest in mapping.values():
                require_sha256(digest, field_name=field_name)
        accounting_signatures = tuple(
            (record.configuration, record.active_context)
            for record in self.byte_accounting
        )
        expected_accounting = tuple(
            (configuration, context)
            for configuration in PHASE11_CONFIGURATIONS
            for context in PHASE11_ACCOUNTING_CONTEXTS
        )
        if accounting_signatures != expected_accounting:
            raise ValueError("Phase 11 byte accounting does not cover exact contexts")
        if tuple(
            record.configuration for record in self.execution_path_evidence
        ) != PHASE11_CONFIGURATIONS:
            raise ValueError("Phase 11 path evidence configuration set differs")
        if tuple(
            record.configuration for record in self.allocation_evidence
        ) != PHASE11_CONFIGURATIONS:
            raise ValueError("Phase 11 allocation evidence configuration set differs")
        if tuple(
            (record.configuration, record.context_length)
            for record in self.graph_evidence
        ) != PHASE11_GRAPH_POINT_SIGNATURES:
            raise ValueError("Phase 11 graph evidence point set differs")
        require_exact_phase11_grid(self.bounded_runs)
        if tuple(check.check_id for check in self.checks) != PHASE11_ADMISSION_CHECK_IDS:
            raise ValueError("Phase 11 admission checks are missing or reordered")
        references = {
            reference.evidence_id: reference
            for reference in self.evidence_references
        }
        if len(references) != len(self.evidence_references):
            raise ValueError("Phase 11 evidence reference IDs must be unique")
        referenced = {
            evidence_id
            for check in self.checks
            for evidence_id in check.evidence_ids
        }
        if referenced != set(references):
            raise ValueError("Phase 11 checks and evidence must join exactly")
        if self.status not in {
            GateDisposition.PASS,
            GateDisposition.PARTIAL,
            GateDisposition.BLOCKED,
            GateDisposition.FAIL,
        }:
            raise ValueError("G2-KVQ status must be terminal")
        if self.local_root_digest is not None:
            require_sha256(self.local_root_digest, field_name="local_root_digest")
        if self.status is GateDisposition.PASS:
            if (
                self.admitted_configurations != PHASE11_CONFIGURATIONS
                or any(
                    point.status is not RunStatus.COMPLETED
                    for point in self.bounded_runs
                )
                or any(
                    check.status is not GateDisposition.PASS
                    for check in self.checks
                )
                or self.gates.g2_kvq is not GateDisposition.PASS
                or self.blockers
                or self.local_root_digest is None
                or self.r2_uri is None
                or self.r2_uri
                != (
                    "r2://kvbench-artifacts/kvbench/sha256/"
                    f"{self.local_root_digest}/"
                )
                or not self.complete_last
                or not self.checksums_valid
                or not self.bucket_lock_identity
                or not self.bucket_lock_identity.strip()
                or not self.clean_retrieval
                or not self.historical_evidence_unchanged
                or not self.existing_methods_unchanged
                or not self.measurement_container_unchanged
            ):
                raise ValueError(
                    "G2-KVQ PASS requires every local, bounded, and durable gate"
                )
        elif (
            not self.blockers
            or len(set(self.blockers)) != len(self.blockers)
            or any(not blocker.strip() for blocker in self.blockers)
            or self.gates.g2_kvq is not self.status
        ):
            raise ValueError("non-PASS G2-KVQ requires exact blockers and status")
        if (
            self.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            or self.quality_status is not QualityValidationState.UNVALIDATED
            or self.quality_execution is not QualityExecutionState.LOCKED
            or self.performance_claim_eligible
            or self.performance_data_frozen
            or self.quality_benchmark_executed
            or self.speedup_calculated
            or self.r_hbm is not None
            or self.measurement_scope
            is not MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
        ):
            raise ValueError("Phase 11 report must remain a non-claim")


def parse_phase11_run_manifest(
    payload: dict[str, object],
) -> Phase11RunManifest:
    """Parse only the strict local Phase 11 lifecycle manifest."""

    return Phase11RunManifest.from_dict(payload)

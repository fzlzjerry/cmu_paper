"""Strict compact admission report for one cache-method adapter."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from kvbench.schema.base import (
    ClaimEligibility,
    MeasurementScope,
    QualityExecutionState,
    QualityValidationState,
    StrictModel,
    require_git_sha,
    require_identifier,
    require_relative_path,
    require_schema,
    require_sha256,
    require_utc_timestamp,
)
from kvbench.schema.config import MethodName
from kvbench.schema.phase3 import GateDisposition


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionEvidenceReference(StrictModel):
    evidence_id: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        require_identifier(self.evidence_id, field_name="evidence_id")
        require_relative_path(self.path, field_name="evidence path")
        require_sha256(self.sha256)


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionCheckResult(StrictModel):
    status: GateDisposition
    summary: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("admission check summary must be non-empty")
        if not self.evidence_ids:
            raise ValueError("admission check requires evidence references")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("admission check evidence IDs must be unique")
        for evidence_id in self.evidence_ids:
            require_identifier(evidence_id, field_name="evidence_id")


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionModelIdentity(StrictModel):
    model_id: str
    revision: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        require_git_sha(self.revision)
        require_sha256(self.fingerprint, field_name="model fingerprint")


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionBackendIdentity(StrictModel):
    backend_id: str
    fingerprint: str

    def __post_init__(self) -> None:
        require_identifier(self.backend_id, field_name="backend_id")
        require_sha256(self.fingerprint, field_name="backend fingerprint")


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionGates(StrictModel):
    g0: GateDisposition
    g1: GateDisposition
    g2: GateDisposition
    g3: GateDisposition
    g4: GateDisposition
    g5: GateDisposition
    full_scan_state: str

    def __post_init__(self) -> None:
        if self.full_scan_state != "closed":
            raise ValueError("Full Scan must remain closed")


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionReport(StrictModel):
    """References evidence; raw traces and samples remain external."""

    schema_version: str
    created_at_utc: str
    status: GateDisposition
    method_name: MethodName
    method_config_id: str
    method_config_fingerprint: str
    adapter_version: str
    adapter_config_fingerprint: str
    model_identity: MethodAdmissionModelIdentity
    backend_identity: MethodAdmissionBackendIdentity
    cache_layout_fingerprint: str
    correctness: MethodAdmissionCheckResult
    byte_accounting: MethodAdmissionCheckResult
    execution_path: MethodAdmissionCheckResult
    graph: MethodAdmissionCheckResult
    reproducibility_status: GateDisposition
    evidence_references: tuple[MethodAdmissionEvidenceReference, ...]
    gates: MethodAdmissionGates
    blockers: tuple[str, ...]
    claim_eligibility: ClaimEligibility
    quality_status: QualityValidationState
    quality_execution: QualityExecutionState
    performance_claim_eligible: bool
    performance_data_frozen: bool
    quality_benchmark_executed: bool
    measurement_scope: MeasurementScope
    creation_git_sha: str

    SCHEMA_VERSION: ClassVar[str] = "kvbench-method-admission-report-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_utc_timestamp(self.created_at_utc)
        require_identifier(
            self.method_config_id,
            field_name="method_config_id",
        )
        require_sha256(
            self.method_config_fingerprint,
            field_name="method_config_fingerprint",
        )
        require_identifier(self.adapter_version, field_name="adapter_version")
        require_sha256(
            self.adapter_config_fingerprint,
            field_name="adapter_config_fingerprint",
        )
        require_sha256(
            self.cache_layout_fingerprint,
            field_name="cache_layout_fingerprint",
        )
        require_git_sha(self.creation_git_sha)
        if self.status not in {
            GateDisposition.PASS,
            GateDisposition.PARTIAL,
            GateDisposition.BLOCKED,
            GateDisposition.FAIL,
        }:
            raise ValueError("method admission status must be terminal")
        check_results = (
            self.correctness,
            self.byte_accounting,
            self.execution_path,
            self.graph,
        )
        if self.status is GateDisposition.PASS and (
            any(item.status is not GateDisposition.PASS for item in check_results)
            or self.reproducibility_status is not GateDisposition.PASS
        ):
            raise ValueError("PASS requires every method admission check to pass")
        references = {
            item.evidence_id: item for item in self.evidence_references
        }
        if len(references) != len(self.evidence_references):
            raise ValueError("evidence reference IDs must be unique")
        referenced_ids = {
            evidence_id
            for result in check_results
            for evidence_id in result.evidence_ids
        }
        if referenced_ids != set(references):
            raise ValueError("check results and evidence references must join exactly")
        if (
            not self.blockers
            or len(set(self.blockers)) != len(self.blockers)
            or any(not blocker.strip() for blocker in self.blockers)
        ):
            raise ValueError("blockers must be non-empty and unique")
        if (
            self.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            or self.quality_status is not QualityValidationState.UNVALIDATED
            or self.quality_execution is not QualityExecutionState.LOCKED
            or self.performance_claim_eligible
            or self.performance_data_frozen
            or self.quality_benchmark_executed
            or self.measurement_scope is not MeasurementScope.NATIVE_HOST_ADMISSION
        ):
            raise ValueError("method admission must retain native-host non-claim state")
        if self.method_name is MethodName.BF16:
            expected_gates = (
                GateDisposition.PASS,
                GateDisposition.PASS,
                GateDisposition.NOT_EVALUATED,
                GateDisposition.NOT_EVALUATED,
                GateDisposition.NOT_EVALUATED,
                GateDisposition.NOT_EVALUATED,
            )
            observed_gates = (
                self.gates.g0,
                self.gates.g1,
                self.gates.g2,
                self.gates.g3,
                self.gates.g4,
                self.gates.g5,
            )
            if observed_gates != expected_gates:
                raise ValueError("BF16 report must preserve G0/G1 and leave G2-G5 open")
            if self.blockers != ("B-009", "B-010"):
                raise ValueError("BF16 report must retain B-009 and B-010")

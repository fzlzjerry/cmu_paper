"""Strict lifecycle records for Phase 13B compressed batch admission."""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Literal

from kvbench.schema.base import (
    RunStatus,
    StrictModel,
    require_git_sha,
    require_oci_digest,
    require_relative_path,
    require_run_id,
    require_schema,
    require_utc_timestamp,
)


PHASE13B_CONFIGURATIONS = (
    "tq_4bit_nc",
    "tq_k3v4_nc",
    "tq_3bit_nc",
    "k4v4",
    "k2v4",
    "k2v2",
    "kvq4",
    "kvq3",
    "kvq2",
)
PHASE13B_BATCH_SIZES = (1, 4, 8)
PHASE13B_CONTEXT_LENGTH = 128
PHASE13B_AUTHORIZED_CONTAINER_DIGEST = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)


@dataclasses.dataclass(frozen=True, slots=True)
class Phase13BBatchAdmissionManifest(StrictModel):
    schema_version: str
    run_id: str
    status: RunStatus
    created_at_utc: str
    started_at_utc: str | None
    finished_at_utc: str | None
    inventory_path: str | None
    failure_reason: str | None
    creation_git_sha: str
    authorized_container_digest: str
    decision_id: Literal["0030"]
    configurations: tuple[str, ...]
    batch_sizes: tuple[int, ...]
    context_length: Literal[128]
    timing_collected: Literal[False]
    performance_claim_eligible: Literal[False]
    quality_executed: Literal[False]
    full_scan_executed: Literal[False]

    SCHEMA_VERSION: ClassVar[str] = (
        "kvbench-phase13b-batch-admission-manifest-1.0.0"
    )

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_run_id(self.run_id)
        require_utc_timestamp(self.created_at_utc)
        require_git_sha(self.creation_git_sha)
        require_oci_digest(self.authorized_container_digest)
        if self.authorized_container_digest != PHASE13B_AUTHORIZED_CONTAINER_DIGEST:
            raise ValueError("Phase 13B container digest differs")
        if self.configurations != PHASE13B_CONFIGURATIONS:
            raise ValueError("Phase 13B configuration set differs")
        if self.batch_sizes != PHASE13B_BATCH_SIZES:
            raise ValueError("Phase 13B batch set differs")
        if self.status is RunStatus.CREATED:
            if any(
                value is not None
                for value in (
                    self.started_at_utc,
                    self.finished_at_utc,
                    self.inventory_path,
                    self.failure_reason,
                )
            ):
                raise ValueError("created Phase 13B manifest has terminal fields")
            return
        if self.started_at_utc is None or self.finished_at_utc is None:
            raise ValueError("terminal Phase 13B manifest lacks timestamps")
        require_utc_timestamp(self.started_at_utc)
        require_utc_timestamp(self.finished_at_utc)
        if self.inventory_path is None:
            raise ValueError("terminal Phase 13B manifest lacks inventory")
        require_relative_path(self.inventory_path, field_name="inventory_path")
        if self.status is RunStatus.COMPLETED:
            if self.failure_reason is not None:
                raise ValueError("completed Phase 13B manifest has a failure")
        elif self.failure_reason is None or not self.failure_reason.strip():
            raise ValueError("failed Phase 13B manifest lacks a reason")


def parse_phase13b_manifest(
    payload: dict[str, object],
) -> Phase13BBatchAdmissionManifest:
    return Phase13BBatchAdmissionManifest.from_dict(payload)

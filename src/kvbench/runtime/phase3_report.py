"""Evidence-derived, append-only Phase 3 BF16 G1 admission reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import secrets
import stat
import statistics
import subprocess
from typing import Any

from kvbench.config import REPOSITORY_ROOT, load_phase3_admission_bundle
from kvbench.errors import SchemaValidationError
from kvbench.runtime.artifacts import validate_run_directory
from kvbench.runtime.phase3_campaign import (
    assert_unique_plan_campaign,
    campaign_root,
    validate_phase3_campaign_directory,
)
from kvbench.runtime.process_supervision import (
    RunOwnedProcessRegistry,
    command_fingerprint,
)
from kvbench.runtime.phase3_worker_channels import (
    build_phase3_worker_channel_commitment,
)
from kvbench.schema import (
    ClaimEligibility,
    CompletionMarker,
    FROZEN_PHASE3_POINT_IDS,
    FROZEN_PHASE3_STABILITY_POINT_IDS,
    G1_CRITERIA,
    GateDisposition,
    GraphMode,
    MeasurementScope,
    Phase3G1AdmissionReport,
    Phase3G1Criterion,
    Phase3RunEvidence,
    Phase3RunManifest,
    Phase3StabilitySummary,
    Phase3WorkerResult,
    QualityExecutionState,
    QualityStatus,
    QualityValidationState,
    RunStatus,
    RunnerKind,
    canonical_json_bytes,
    g1_expected_point_ids,
    parse_run_manifest,
    sha256_hex,
    SourceDigest,
)
from kvbench.schema.phase3 import (
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GROWING_PLAN_PATH,
    PHASE3_PLAN_FINGERPRINTS,
)


_CAMPAIGN_ID = re.compile(
    r"phase3-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)
_REPORT_ID = re.compile(
    r"phase3-g1-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)
_PROCESS_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROCESS_HANDSHAKE_STAGES = (
    "worker_started",
    "cuda_context_created",
    "measurement_started",
    "measurement_finished",
    "evidence_flushed",
    "worker_exiting",
    "supervisor_reaped",
)
_PROCESS_WORKER_HANDSHAKE_STAGES = _PROCESS_HANDSHAKE_STAGES[:-1]
_PROCESS_READY_NOT_OBSERVED_V2 = {
    "schema_version": "kvbench-phase3-worker-ready-2.0.0",
    "readiness_observed": False,
    "pid": None,
    "process_start_time_ticks": None,
    "cuda_imported": None,
}
_PROCESS_AUDIT_V2 = "kvbench-phase3-process-audit-2.0.0"
_PROCESS_AUDIT_V3 = "kvbench-phase3-process-audit-3.0.0"
_PROCESS_REGISTRY_V2 = "kvbench-phase3-process-registry-2.0.0"
_PROCESS_HANDSHAKE_V2 = "kvbench-phase3-worker-handshake-2.0.0"
_PROCESS_HANDSHAKE_V3 = "kvbench-phase3-worker-handshake-3.0.0"
_PROCESS_COMMAND_FINGERPRINT_ENV = "KVBENCH_PHASE3_COMMAND_FINGERPRINT"
_MAX_JSON_BYTES = 64 * 1024 * 1024
_FIXED_POINT_IDS = tuple(
    point_id
    for point_id in FROZEN_PHASE3_POINT_IDS
    if point_id.startswith("fixed_l-")
)
_GROWING_POINT_IDS = tuple(
    point_id
    for point_id in FROZEN_PHASE3_POINT_IDS
    if point_id.startswith("growing_context-")
)
_SUT_SOURCE_PATHS = (
    "src/kvbench/runtime/backend.py",
    "src/kvbench/runtime/bf16_endpoint.py",
    "src/kvbench/runtime/cuda_graph.py",
    "src/kvbench/runtime/fixed_l_runner.py",
    "src/kvbench/runtime/growing_context_runner.py",
    "src/kvbench/runtime/static_cache.py",
    "src/kvbench/runtime/timing.py",
)
_FORBIDDEN_HOT_PATH_PATTERNS = (
    "torch.cat(",
    ".repeat_interleave(",
    "repeat_kv(",
    "DynamicCache(",
    ".expand(",
)
_REPORT_GENERATOR_CHANGE_PATHS = frozenset(
    {
        "docs/blockers.md",
        "docs/evidence/phase3/g1-admission.json",
        "docs/phase_reports/phase3.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/validate_phase2.py",
        "src/kvbench/runtime/phase3_report.py",
        "src/kvbench/runtime/phase3_report_publication.py",
        "tests/unit/test_phase3_report.py",
        "tests/unit/test_phase3_report_publication.py",
    }
)
_POST_REPORT_CHANGE_PATHS = frozenset(
    {
        "docs/blockers.md",
        "docs/evidence/phase3/g1-admission.json",
        "docs/phase_reports/phase3.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
    }
)
_PHASE3_REPORT_RAW_AUDIT_REPLAY_CONTRACT = (
    "phase3-report-raw-audit-replay-v1"
)
_PHASE3_EAGER_ALLOCATION_CRITERION = (
    "phase3_eager_attributed_ephemeral_v1"
)
_PHASE3_GRAPH_ALLOCATION_CRITERION = "phase3_graph_zero_allocation_v1"
_PHASE3_GQA_VERIFIED = "gqa_nonmaterialization_verified"
_PHASE3_EAGER_ALLOCATION_CLASSES = frozenset(
    {
        "context_scaled_workspace",
        "fixed_output",
        "fixed_shared_activation",
        "framework_bookkeeping",
    }
)


class Phase3ReportError(RuntimeError):
    """The selected campaigns cannot support a trustworthy G1 report."""


@dataclass(frozen=True, slots=True)
class ValidatedPhase3Run:
    """Cross-file joined immutable evidence for one exact process."""

    run_dir: Path
    manifest: Phase3RunManifest
    completion: CompletionMarker
    worker_result: Phase3WorkerResult
    process_audit: Mapping[str, Any]
    ready_process: Mapping[str, Any]
    worker_evidence: Mapping[str, Any] | None
    runtime: Mapping[str, Any] | None
    numerical: Mapping[str, Any] | None
    timing: Mapping[str, Any] | None
    telemetry: Mapping[str, Any] | None

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def point_id(self) -> str:
        return self.manifest.point_id


def _strict_json_object(
    path: Path,
    *,
    canonical: bool = False,
) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Phase3ReportError(f"required evidence is absent: {path}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > _MAX_JSON_BYTES
    ):
        raise Phase3ReportError(f"evidence path is unsafe: {path}")

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON number")

    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError) as error:
        raise Phase3ReportError(f"evidence is not strict JSON: {path}") from error
    if not isinstance(payload, dict):
        raise Phase3ReportError(f"evidence is not a JSON object: {path}")
    expected = (
        canonical_json_bytes(payload) + b"\n"
        if canonical
        else (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    )
    if raw != expected:
        raise Phase3ReportError(
            f"evidence does not use its deterministic JSON encoding: {path}"
        )
    return payload


def _optional_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _strict_json_object(path)


def _require_equal(label: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise Phase3ReportError(f"cross-file evidence join failed: {label}")


def _manifest_environment_join(run_dir: Path, manifest: Phase3RunManifest) -> None:
    recorded_environment = _strict_json_object(
        run_dir / "environment" / "worker_environment.json"
    )
    environment = dict(recorded_environment)
    observed_command_fingerprint = environment.pop(
        _PROCESS_COMMAND_FINGERPRINT_ENV,
        None,
    )
    digest = sha256_hex(canonical_json_bytes(environment))
    _require_equal(
        "worker environment SHA-256",
        digest,
        manifest.command.environment_sha256,
    )
    if observed_command_fingerprint is not None:
        expected_command_fingerprint = command_fingerprint(
            manifest.command.argv,
            working_directory=manifest.command.working_directory,
            environment_sha256=manifest.command.environment_sha256,
        )
        _require_equal(
            "worker command fingerprint environment",
            observed_command_fingerprint,
            expected_command_fingerprint,
        )


def _runtime_model_identity(manifest: Phase3RunManifest) -> dict[str, Any]:
    """Project the full manifest identity onto the loader's runtime evidence."""

    model = manifest.model_identity
    return {
        "model_id": model.model_id,
        "revision": model.revision,
        "snapshot_path": model.local_snapshot_path,
        "file_hashes": {
            artifact.path: artifact.sha256 for artifact in model.artifacts
        },
        "architecture": model.architecture,
        "num_hidden_layers": model.geometry.num_hidden_layers,
        "num_attention_heads": model.geometry.num_query_heads,
        "num_key_value_heads": model.geometry.num_kv_heads,
        "head_dim": model.geometry.head_dim,
        "max_position_embeddings": model.geometry.max_context_length,
        "weight_dtype": model.weight_dtype,
    }


def _split_runtime_join(
    run_dir: Path,
    runtime: Mapping[str, Any],
    numerical: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any], Mapping[str, Any]]:
    allocation = _strict_json_object(run_dir / "allocation" / "audit.json")
    expected_allocation = {
        "allocation": runtime.get("allocation"),
        "memory_evidence": runtime.get("memory_evidence"),
        "cache_accounting": runtime.get("cache_accounting"),
        "instrumented_duration_reported_as_timing": False,
    }
    _require_equal("allocation split", allocation, expected_allocation)

    gqa = _strict_json_object(run_dir / "gqa" / "audit.json")
    expected_gqa = {
        "source": runtime.get("gqa_source"),
        "cache_geometry": runtime.get("gqa_cache_geometry"),
        "operator": runtime.get("gqa_operator"),
        "operators": runtime.get("gqa_operators"),
        "mha_control": runtime.get("mha_control"),
        "prefill_backend": runtime.get("prefill_backend"),
        "decode_backend": runtime.get("backend"),
    }
    _require_equal("GQA split", gqa, expected_gqa)

    telemetry = _strict_json_object(run_dir / "telemetry" / "snapshots.json")
    expected_telemetry = {
        "before": runtime.get("telemetry_before"),
        "after": runtime.get("telemetry_after"),
        "sampling_interval_seconds": runtime.get(
            "telemetry_sampling_interval_seconds"
        ),
        "queried_inside_decode_hot_path": False,
        "stability_inference": False,
    }
    _require_equal("telemetry split", telemetry, expected_telemetry)

    timing = _optional_json_object(run_dir / "raw" / "timing.json")
    runtime_timing = runtime.get("timing")
    if runtime_timing is None:
        if timing is not None:
            raise Phase3ReportError("timing artifact exists for a skipped timing lane")
    else:
        if not isinstance(runtime_timing, Mapping) or timing is None:
            raise Phase3ReportError("runtime timing lacks its raw timing artifact")
        expected_timing = {
            **runtime_timing,
            "quality_status": "unvalidated",
            "claim_eligibility": "performance_only",
            "performance_claim_eligible": False,
            "measurement_scope": "native_host_admission",
            "profiler_instrumented": False,
        }
        _require_equal("timing split", timing, expected_timing)

    numerical_file = _optional_json_object(
        run_dir / "numerical" / "agreement.json"
    )
    _require_equal("numerical split", numerical_file, numerical)
    return timing, telemetry, allocation


def _process_snapshot_clean(
    value: object,
    *,
    allow_supervised: bool,
    ready: Mapping[str, Any] | None = None,
    gpu_uuid: str | None = None,
) -> bool:
    snapshot = _mapping(value)
    if snapshot is None:
        return False
    allowed = snapshot.get("allowed_compute_processes")
    subcommands = _sequence(snapshot.get("subcommands"))
    if set(snapshot) != {
        "captured_at_utc",
        "query_exit_code",
        "graphics_processes",
        "allowed_compute_processes",
        "foreign_compute_processes",
        "unknown_processes",
        "subcommands",
        "errors",
    }:
        return False
    try:
        captured_at = datetime.fromisoformat(
            str(snapshot.get("captured_at_utc")).replace("Z", "+00:00")
        )
    except ValueError:
        return False
    expected_subcommands = (
        (
            "gpu_index_uuid",
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
        ),
        (
            "compute_apps",
            [
                "/usr/bin/nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
        ),
        ("pmon", ["/usr/bin/nvidia-smi", "pmon", "-c", "1"]),
    )
    subcommands_valid = bool(
        subcommands is not None
        and len(subcommands) == 3
        and all(
            _mapping(command) is not None
            and set(_mapping(command))
            == {"name", "argv", "exit_code", "stdout", "stderr"}
            and _mapping(command).get("name") == expected_name
            and _mapping(command).get("argv") == expected_argv
            and _mapping(command).get("exit_code") == 0
            and isinstance(_mapping(command).get("stdout"), str)
            and isinstance(_mapping(command).get("stderr"), str)
            for command, (expected_name, expected_argv) in zip(
                subcommands,
                expected_subcommands,
            )
        )
    )
    if not (
        captured_at.tzinfo is not None
        and captured_at.utcoffset() == timezone.utc.utcoffset(None)
        and snapshot.get("query_exit_code") == 0
        and snapshot.get("errors") == []
        and isinstance(snapshot.get("graphics_processes"), list)
        and snapshot.get("foreign_compute_processes") == []
        and snapshot.get("unknown_processes") == []
        and isinstance(allowed, list)
        and (allow_supervised or allowed == [])
        and subcommands_valid
    ):
        return False
    if not allow_supervised:
        return True
    if ready is None or gpu_uuid is None:
        return False
    expected_root = {
        "pid": ready.get("pid"),
        "start_time_ticks": ready.get("process_start_time_ticks"),
    }
    return all(
        isinstance(process, Mapping)
        and set(process)
        == {
            "gpu_uuid",
            "pid",
            "process_start_time_ticks",
            "process_type",
            "process_name",
            "used_gpu_memory_mib",
            "relationship",
            "supervised_root_identity",
        }
        and process.get("gpu_uuid") == gpu_uuid
        and isinstance(process.get("pid"), int)
        and not isinstance(process.get("pid"), bool)
        and isinstance(process.get("process_start_time_ticks"), int)
        and not isinstance(process.get("process_start_time_ticks"), bool)
        and process.get("process_type") in {"C", "C+G"}
        and isinstance(process.get("process_name"), str)
        and bool(process.get("process_name"))
        and (
            process.get("used_gpu_memory_mib") is None
            or _finite_number(process.get("used_gpu_memory_mib"))
            and float(process.get("used_gpu_memory_mib")) >= 0.0
        )
        and process.get("relationship")
        in {"supervised_child", "supervised_descendant"}
        and process.get("supervised_root_identity") == expected_root
        for process in allowed
    )


def _validate_telemetry_evidence(
    telemetry: Mapping[str, Any],
    manifest: Phase3RunManifest,
) -> None:
    if set(telemetry) != {
        "before",
        "after",
        "sampling_interval_seconds",
        "queried_inside_decode_hot_path",
        "stability_inference",
    } or (
        telemetry.get("queried_inside_decode_hot_path") is not False
        or telemetry.get("stability_inference") is not False
    ):
        raise Phase3ReportError("telemetry envelope differs from the frozen policy")
    snapshots: list[Mapping[str, Any]] = []
    expected_keys = {
        "timestamp",
        "collected_at_utc",
        "host_query_started_ns",
        "host_query_finished_ns",
        "host_monotonic_ns",
        "gpu_name",
        "gpu_uuid",
        "power_watts",
        "temperature_celsius",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "vram_used_mib",
        "ecc_mode",
        "raw_snapshot",
        "stability_inference",
    }
    for label in ("before", "after"):
        snapshot = _mapping(telemetry.get(label))
        if snapshot is None or set(snapshot) != expected_keys:
            raise Phase3ReportError("telemetry snapshot fields differ")
        started = snapshot.get("host_query_started_ns")
        finished = snapshot.get("host_query_finished_ns")
        midpoint = snapshot.get("host_monotonic_ns")
        if (
            not isinstance(snapshot.get("timestamp"), str)
            or not snapshot.get("timestamp")
            or not isinstance(snapshot.get("collected_at_utc"), str)
            or snapshot.get("gpu_name") != manifest.gpu_full_name
            or snapshot.get("gpu_uuid") != manifest.gpu_uuid
            or not isinstance(started, int)
            or isinstance(started, bool)
            or not isinstance(finished, int)
            or isinstance(finished, bool)
            or not isinstance(midpoint, int)
            or isinstance(midpoint, bool)
            or started < 0
            or started > finished
            or midpoint != (started + finished) // 2
            or not all(
                _finite_number(snapshot.get(field))
                for field in (
                    "power_watts",
                    "temperature_celsius",
                    "sm_clock_mhz",
                    "memory_clock_mhz",
                    "vram_used_mib",
                )
            )
            or snapshot.get("raw_snapshot") is not True
            or snapshot.get("stability_inference") is not False
            or not (
                snapshot.get("ecc_mode") is None
                or isinstance(snapshot.get("ecc_mode"), str)
                and bool(snapshot.get("ecc_mode"))
            )
        ):
            raise Phase3ReportError("telemetry snapshot identity or values differ")
        try:
            parsed_time = datetime.fromisoformat(
                str(snapshot["collected_at_utc"]).replace("Z", "+00:00")
            )
        except ValueError as error:
            raise Phase3ReportError("telemetry collection timestamp is invalid") from error
        if parsed_time.tzinfo is None or parsed_time.utcoffset() != timezone.utc.utcoffset(None):
            raise Phase3ReportError("telemetry collection timestamp is not UTC")
        snapshots.append(snapshot)
    interval = telemetry.get("sampling_interval_seconds")
    derived_interval = (
        int(snapshots[1]["host_monotonic_ns"])
        - int(snapshots[0]["host_monotonic_ns"])
    ) / 1_000_000_000.0
    if (
        not _finite_number(interval)
        or float(interval) < 0.0
        or not math.isclose(
            float(interval),
            derived_interval,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise Phase3ReportError("telemetry sampling interval is not derived")


def _process_integer(value: object, *, positive: bool = False) -> bool:
    return bool(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= (1 if positive else 0)
    )


def _process_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(None)
    ):
        return None
    return parsed


def _v2_raw_process_record(
    value: object,
    *,
    category: str,
    registered_identity: Mapping[str, object],
) -> dict[str, object]:
    process = _mapping(value)
    base_keys = {
        "gpu_uuid",
        "pid",
        "process_start_time_ticks",
        "process_type",
        "process_name",
        "used_gpu_memory_mib",
    }
    expected_keys = (
        base_keys | {"relationship", "supervised_root_identity"}
        if category == "allowed_compute_processes"
        else base_keys
    )
    expected_types = {
        "allowed_compute_processes": {"C", "C+G"},
        "foreign_compute_processes": {"C", "C+G"},
        "unknown_processes": {"UNKNOWN"},
        "graphics_processes": {"G"},
    }[category]
    if (
        process is None
        or set(process) != expected_keys
        or not isinstance(process.get("gpu_uuid"), str)
        or re.fullmatch(r"GPU-[A-Za-z0-9-]+\Z", str(process.get("gpu_uuid")))
        is None
        or not _process_integer(process.get("pid"), positive=True)
        or not _process_integer(process.get("process_start_time_ticks"))
        or process.get("process_type") not in expected_types
        or not isinstance(process.get("process_name"), str)
        or not process.get("process_name")
        or (
            process.get("used_gpu_memory_mib") is not None
            and (
                not _finite_number(process.get("used_gpu_memory_mib"))
                or float(process["used_gpu_memory_mib"]) < 0.0
            )
        )
    ):
        raise Phase3ReportError("v2 raw GPU process record is malformed")
    if category == "allowed_compute_processes" and (
        process.get("relationship")
        not in {"supervised_child", "supervised_descendant"}
        or process.get("supervised_root_identity")
        != {
            "pid": registered_identity["pid"],
            "start_time_ticks": registered_identity["start_time_ticks"],
        }
    ):
        raise Phase3ReportError("v2 supervised process identity differs")
    return {
        "gpu_uuid": process["gpu_uuid"],
        "pid": process["pid"],
        "process_start_time_ticks": (
            None
            if process["process_start_time_ticks"] == 0
            else process["process_start_time_ticks"]
        ),
    }


def _v2_raw_process_snapshot(
    value: object,
    *,
    registered_identity: Mapping[str, object],
) -> tuple[Mapping[str, Any], tuple[dict[str, object], ...]]:
    snapshot = _mapping(value)
    if snapshot is None:
        raise Phase3ReportError("v2 raw process snapshot is not an object")
    query_exit_code = snapshot.get("query_exit_code")
    errors = snapshot.get("errors")
    subcommands = _sequence(snapshot.get("subcommands"))
    captured_at = _process_utc_timestamp(snapshot.get("captured_at_utc"))
    expected_subcommands = (
        (
            "gpu_index_uuid",
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
        ),
        (
            "compute_apps",
            [
                "/usr/bin/nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
        ),
        ("pmon", ["/usr/bin/nvidia-smi", "pmon", "-c", "1"]),
    )
    if (
        set(snapshot)
        != {
            "captured_at_utc",
            "query_exit_code",
            "graphics_processes",
            "allowed_compute_processes",
            "foreign_compute_processes",
            "unknown_processes",
            "subcommands",
            "errors",
        }
        or captured_at is None
        or not isinstance(query_exit_code, int)
        or isinstance(query_exit_code, bool)
        or query_exit_code < 0
        or query_exit_code > 255
        or not isinstance(errors, list)
        or any(not isinstance(item, str) or not item for item in errors)
        or subcommands is None
        or len(subcommands) not in {0, len(expected_subcommands)}
    ):
        raise Phase3ReportError("v2 process query outcome is malformed")
    if not subcommands:
        if (
            query_exit_code != 2
            or len(errors) != 1
            or any(
                snapshot.get(field) != []
                for field in (
                    "graphics_processes",
                    "allowed_compute_processes",
                    "foreign_compute_processes",
                    "unknown_processes",
                )
            )
        ):
            raise Phase3ReportError("v2 empty process query outcome differs")
    else:
        exit_codes: list[int] = []
        for raw_command, (expected_name, expected_argv) in zip(
            subcommands,
            expected_subcommands,
        ):
            command = _mapping(raw_command)
            exit_code = None if command is None else command.get("exit_code")
            if (
                command is None
                or set(command)
                != {"name", "argv", "exit_code", "stdout", "stderr"}
                or command.get("name") != expected_name
                or command.get("argv") != expected_argv
                or not isinstance(exit_code, int)
                or isinstance(exit_code, bool)
                or exit_code < 0
                or exit_code > 255
                or not isinstance(command.get("stdout"), str)
                or not isinstance(command.get("stderr"), str)
            ):
                raise Phase3ReportError("v2 process query subcommand differs")
            exit_codes.append(exit_code)
        nonzero_codes = [code for code in exit_codes if code != 0]
        if not errors:
            expected_query_exit_code = 0
        elif nonzero_codes:
            expected_query_exit_code = nonzero_codes[0]
        else:
            expected_query_exit_code = 2
        if query_exit_code != expected_query_exit_code:
            raise Phase3ReportError(
                "v2 aggregate process query exit code differs"
            )

    observations: list[dict[str, object]] = []
    seen: set[tuple[object, object, object]] = set()
    for category in (
        "allowed_compute_processes",
        "foreign_compute_processes",
        "unknown_processes",
    ):
        records = snapshot.get(category)
        if not isinstance(records, list):
            raise Phase3ReportError("v2 compute process list is malformed")
        for record in records:
            observation = _v2_raw_process_record(
                record,
                category=category,
                registered_identity=registered_identity,
            )
            key = (
                observation["gpu_uuid"],
                observation["pid"],
                observation["process_start_time_ticks"],
            )
            if key in seen:
                raise Phase3ReportError("v2 process snapshot repeats an identity")
            seen.add(key)
            observations.append(observation)

    graphics = snapshot.get("graphics_processes")
    if not isinstance(graphics, list):
        raise Phase3ReportError("v2 graphics process list is malformed")
    for record in graphics:
        _v2_raw_process_record(
            record,
            category="graphics_processes",
            registered_identity=registered_identity,
        )
    return snapshot, tuple(observations)


def _v2_expected_registry_verdict(
    observations: Sequence[Mapping[str, object]],
    *,
    registered_identity: Mapping[str, object],
    allow_missing_start_time: bool,
) -> dict[str, object]:
    owned: list[dict[str, object]] = []
    foreign: list[dict[str, object]] = []
    pid_reuse: list[dict[str, object]] = []
    unverified: list[dict[str, object]] = []
    registered_pid = registered_identity["pid"]
    registered_start = registered_identity["start_time_ticks"]
    registered_gpu = registered_identity["gpu_uuid"]
    for observation in observations:
        retained = dict(observation)
        pid = observation.get("pid")
        start_time = observation.get("process_start_time_ticks")
        gpu_uuid = observation.get("gpu_uuid")
        if pid == registered_pid and start_time is not None and start_time != registered_start:
            pid_reuse.append(retained)
        elif (
            pid == registered_pid
            and gpu_uuid == registered_gpu
            and start_time == registered_start
        ):
            owned.append(retained)
        elif (
            pid == registered_pid
            and gpu_uuid == registered_gpu
            and start_time is None
            and allow_missing_start_time
        ):
            owned.append(retained)
        elif pid == registered_pid and gpu_uuid == registered_gpu and start_time is None:
            unverified.append(retained)
        else:
            foreign.append(retained)
    if pid_reuse:
        disposition = "pid_reuse_detected"
    elif foreign:
        disposition = "foreign_process_detected"
    elif unverified:
        disposition = "unverified_registered_pid"
    elif owned:
        disposition = "owned_only"
    else:
        disposition = "clean"
    return {
        "disposition": disposition,
        "hard_failure": disposition not in {"clean", "owned_only"},
        "owned": owned,
        "foreign": foreign,
        "pid_reuse": pid_reuse,
        "unverified": unverified,
    }


def _v2_registry_snapshot_verdict(
    raw_snapshot: object,
    verdict_value: object,
    *,
    registered_identity: Mapping[str, object],
    terminal_resolution_allowed: bool,
    proc_disappeared_after_registration: bool,
) -> Mapping[str, Any]:
    snapshot, observations = _v2_raw_process_snapshot(
        raw_snapshot,
        registered_identity=registered_identity,
    )
    allow_missing = bool(
        terminal_resolution_allowed and proc_disappeared_after_registration
    )
    expected_registry = _v2_expected_registry_verdict(
        observations,
        registered_identity=registered_identity,
        allow_missing_start_time=allow_missing,
    )
    errors = snapshot["errors"]
    registered_pid = registered_identity["pid"]
    registered_gpu_uuid = registered_identity["gpu_uuid"]
    registered_pmon_gap_error = (
        f"compute_apps GPU {registered_gpu_uuid} "
        f"PID {registered_pid} has no pmon process type"
    )
    registered_proc_unavailable_prefix = (
        f"cannot read /proc/{registered_pid}/stat"
    )
    registered_query_race_owned = bool(
        errors
        and snapshot["query_exit_code"] == 2
        and expected_registry["disposition"] == "owned_only"
        and registered_pmon_gap_error in errors
        and all(
            error == registered_pmon_gap_error
            or (
                terminal_resolution_allowed
                and error.startswith(registered_proc_unavailable_prefix)
            )
            for error in errors
        )
    )
    terminal_resolution_used = bool(
        registered_query_race_owned
        and terminal_resolution_allowed
        and any(
            error.startswith(registered_proc_unavailable_prefix)
            for error in errors
        )
    )
    clean_query = snapshot["query_exit_code"] == 0 and errors == []
    expected = {
        "passed": bool(
            expected_registry["hard_failure"] is False
            and (clean_query or registered_query_race_owned)
        ),
        "terminal_registered_process_resolution": terminal_resolution_used,
        "query_evidence_hard_failure": bool(
            not clean_query and not registered_query_race_owned
        ),
        "registry_verdict": expected_registry,
        "raw_query_exit_code": snapshot["query_exit_code"],
        "raw_errors": errors,
    }
    verdict = _mapping(verdict_value)
    if verdict is None or dict(verdict) != expected:
        raise Phase3ReportError("v2 process registry verdict differs from raw evidence")
    return verdict


def _validate_v2_handshake(
    *,
    manifest: Phase3RunManifest,
    registry: Mapping[str, Any],
    handshake: Mapping[str, Any],
    worker_evidence: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    identity = _mapping(registry.get("identity"))
    events = _sequence(registry.get("handshake_events"))
    outcome = _mapping(registry.get("outcome"))
    if identity is None or events is None or outcome is None:
        raise Phase3ReportError("v2 process registry lifecycle is incomplete")
    spawned_at = _process_utc_timestamp(identity.get("spawned_at_utc"))
    expected_command_fingerprint = command_fingerprint(
        manifest.command.argv,
        working_directory=manifest.command.working_directory,
        environment_sha256=manifest.command.environment_sha256,
    )
    if (
        set(handshake)
        != {
            "schema_version",
            "run_id",
            "events",
            "terminal_outcome",
            "evidence_flushed_required_for_owned_completion",
        }
        or handshake.get("schema_version")
        != "kvbench-phase3-worker-handshake-2.0.0"
        or handshake.get("run_id") != manifest.run_id
        or handshake.get("events") != list(events)
        or handshake.get("terminal_outcome") != dict(outcome)
        or handshake.get("evidence_flushed_required_for_owned_completion") is not True
        or len(events) != len(_PROCESS_HANDSHAKE_STAGES)
        or spawned_at is None
        or identity.get("expected_command_fingerprint")
        != expected_command_fingerprint
    ):
        raise Phase3ReportError("v2 worker handshake envelope differs")

    previous_timestamp: datetime | None = None
    evidence_sha256: str | None = None
    for sequence, (raw_event, expected_stage) in enumerate(
        zip(events, _PROCESS_HANDSHAKE_STAGES),
        start=1,
    ):
        event = _mapping(raw_event)
        timestamp = (
            None if event is None else _process_utc_timestamp(event.get("recorded_at_utc"))
        )
        expected_digest = event.get("evidence_sha256") if event is not None else None
        if (
            event is None
            or set(event)
            != {
                "schema_version",
                "sequence",
                "stage",
                "recorded_at_utc",
                "run_id",
                "gpu_uuid",
                "pid",
                "process_start_time_ticks",
                "parent_pid",
                "command_fingerprint",
                "evidence_sha256",
            }
            or event.get("schema_version")
            != "kvbench-phase3-worker-handshake-event-1.0.0"
            or event.get("sequence") != sequence
            or event.get("stage") != expected_stage
            or timestamp is None
            or event.get("run_id") != identity.get("run_id")
            or event.get("gpu_uuid") != identity.get("gpu_uuid")
            or event.get("pid") != identity.get("pid")
            or event.get("process_start_time_ticks")
            != identity.get("start_time_ticks")
            or event.get("parent_pid") != identity.get("parent_pid")
            or event.get("command_fingerprint")
            != identity.get("expected_command_fingerprint")
            or (
                expected_stage == "evidence_flushed"
                and (
                    not isinstance(expected_digest, str)
                    or _PROCESS_SHA256.fullmatch(expected_digest) is None
                )
            )
            or expected_stage != "evidence_flushed"
            and expected_digest is not None
            or previous_timestamp is not None
            and timestamp < previous_timestamp
            or previous_timestamp is None
            and timestamp < spawned_at
        ):
            raise Phase3ReportError("v2 worker handshake event differs")
        previous_timestamp = timestamp
        if expected_stage == "evidence_flushed":
            evidence_sha256 = str(expected_digest)

    expected_outcome = {
        "disposition": "owned_completed",
        "reason": "registered worker completed the ordered handshake",
        "returncode": 0,
        "observed_stages": list(_PROCESS_HANDSHAKE_STAGES),
        "missing_worker_stages": [],
        "evidence_flushed": True,
        "worker_exiting_observed": True,
        "full_handshake_observed": True,
        "exclusivity_passed": True,
    }
    if dict(outcome) != expected_outcome or evidence_sha256 is None:
        raise Phase3ReportError("v2 worker ownership completion differs")
    canonical_evidence_sha256 = sha256_hex(
        canonical_json_bytes(worker_evidence) + b"\n"
    )
    if evidence_sha256 != canonical_evidence_sha256:
        raise Phase3ReportError("v2 worker evidence digest linkage differs")
    return identity, evidence_sha256



def _normalize_v3_process_evidence(
    run_dir: Path,
    manifest: Phase3RunManifest,
    process_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate v3 policy fields, then reuse the strict v2 raw-evidence join."""

    audit_v3_extra = {
        "owned_completion_basis",
        "owned_completion_policy",
        "worker_exiting_required_for_owned_completion",
        "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed",
        "worker_termination_resolved",
        "worker_termination_disposition",
    }
    registry_v3_extra = {
        "owned_completion_policy",
        "evidence_flushed_required_for_owned_completion",
        "zero_returncode_required_for_owned_completion",
        "worker_exiting_required_for_owned_completion",
        "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed",
    }
    handshake_v3_extra = {
        "zero_returncode_required_for_owned_completion",
        "worker_exiting_required_for_owned_completion",
        "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed",
    }
    registry = _strict_json_object(
        run_dir / "environment" / "process.registry.json"
    )
    handshake = _strict_json_object(
        run_dir / "environment" / "process.handshake.json"
    )
    outcome = _mapping(registry.get("outcome"))
    if (
        process_audit.get("schema_version") != _PROCESS_AUDIT_V3
        or process_audit.get("owned_completion_policy")
        != RunOwnedProcessRegistry.OWNED_COMPLETION_POLICY
        or process_audit.get("worker_exiting_required_for_owned_completion")
        is not False
        or process_audit.get(
            "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed"
        )
        is not True
        or process_audit.get("worker_termination_resolved") is not True
        or process_audit.get("worker_termination_disposition")
        not in {"registered_reaped", "registered_terminated_reaped"}
    ):
        raise Phase3ReportError("v3 process audit policy is malformed")
    if (
        registry.get("schema_version") != RunOwnedProcessRegistry.SCHEMA_VERSION
        or registry.get("owned_completion_policy")
        != RunOwnedProcessRegistry.OWNED_COMPLETION_POLICY
        or registry.get("evidence_flushed_required_for_owned_completion") is not True
        or registry.get("zero_returncode_required_for_owned_completion") is not True
        or registry.get("worker_exiting_required_for_owned_completion") is not False
        or registry.get(
            "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed"
        )
        is not True
    ):
        raise Phase3ReportError("v3 process registry policy is malformed")
    if (
        handshake.get("schema_version") != _PROCESS_HANDSHAKE_V3
        or handshake.get("run_id") != manifest.run_id
        or handshake.get("events") != registry.get("handshake_events")
        or handshake.get("terminal_outcome") != registry.get("outcome")
        or handshake.get("evidence_flushed_required_for_owned_completion") is not True
        or handshake.get("zero_returncode_required_for_owned_completion") is not True
        or handshake.get("worker_exiting_required_for_owned_completion") is not False
        or handshake.get(
            "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed"
        )
        is not True
        or outcome is None
        or process_audit.get("ownership_verdict") != outcome.get("disposition")
        or process_audit.get("exclusivity_passed")
        is not outcome.get("exclusivity_passed")
        or process_audit.get("evidence_flushed")
        is not outcome.get("evidence_flushed")
        or process_audit.get("worker_exiting_observed")
        is not outcome.get("worker_exiting_observed")
        or process_audit.get("owned_completion_basis")
        != outcome.get("owned_completion_basis")
    ):
        raise Phase3ReportError("v3 process handshake policy is malformed")
    normalized_events = [dict(event) for event in registry["handshake_events"]]
    if process_audit.get("passed") is True:
        if (
            outcome.get("disposition") != "owned_completed"
            or outcome.get("owned_completion_basis")
            != "full_ordered_handshake_zero_exit"
            or outcome.get("worker_exiting_observed") is not True
        ):
            raise Phase3ReportError(
                "v3 completed process evidence lacks the retained full handshake"
            )
        transport_root = run_dir / "raw" / "transport"
        primary_path = transport_root / "primary_worker_evidence.v1.jsonl"
        sidecar_path = transport_root / "raw_audit_index_sidecar.v2.jsonl"
        primary_payload = _strict_json_object(primary_path, canonical=True)
        _strict_json_object(sidecar_path, canonical=True)
        primary_bytes = primary_path.read_bytes()
        sidecar_bytes = sidecar_path.read_bytes()
        expected_commitment = build_phase3_worker_channel_commitment(
            run_id=manifest.run_id,
            point_id=manifest.point_id,
            primary_evidence_bytes=primary_bytes,
            raw_audit_index_bytes=sidecar_bytes,
        )
        commitment_bytes = (
            transport_root / "channel_commitment.json"
        ).read_bytes()
        expected_commitment_bytes = canonical_json_bytes(expected_commitment)
        commitment_sha256 = sha256_hex(expected_commitment_bytes)
        digest_bytes = (transport_root / "channel_commitment.sha256").read_bytes()
        evidence_events = [
            event
            for event in normalized_events
            if event.get("stage") == "evidence_flushed"
        ]
        worker_evidence = _strict_json_object(
            run_dir / "raw" / "worker_evidence.json"
        )
        if (
            commitment_bytes != expected_commitment_bytes
            or digest_bytes != commitment_sha256.encode("ascii") + b"\n"
            or len(evidence_events) != 1
            or evidence_events[0].get("evidence_sha256") != commitment_sha256
            or primary_payload != worker_evidence
        ):
            raise Phase3ReportError("v3 worker channel commitment differs")
        evidence_events[0]["evidence_sha256"] = sha256_hex(
            canonical_json_bytes(worker_evidence) + b"\n"
        )
    elif outcome.get("owned_completion_basis") is not None:
        raise Phase3ReportError("v3 failed process evidence claims completion")

    termination = _strict_json_object(
        run_dir / "validation" / "worker_termination.json"
    )
    termination_keys = {
        "schema_version",
        "run_id",
        "required",
        "registered",
        "resolved",
        "disposition",
        "returncode",
        "failure_reason",
        "source_revalidation_attempted_after_resolution",
        "source_revalidated_after_resolution",
        "pidfd_closed_after_resolution",
        "pidfd_closed_after_source_revalidation_attempt",
    }
    expected_pidfd_close = (
        True if process_audit.get("pidfd_opened") is True else None
    )
    if (
        set(termination) != termination_keys
        or termination.get("schema_version")
        != "kvbench-phase3-worker-termination-1.0.0"
        or termination.get("run_id") != manifest.run_id
        or termination.get("required") is not True
        or termination.get("registered") is not True
        or termination.get("resolved") is not True
        or termination.get("disposition")
        != process_audit.get("worker_termination_disposition")
        or termination.get("returncode") != outcome.get("returncode")
        or termination.get("failure_reason") is not None
        or termination.get("source_revalidation_attempted_after_resolution")
        is not True
        or termination.get("source_revalidated_after_resolution") is not True
        or termination.get("pidfd_closed_after_resolution")
        is not expected_pidfd_close
        or termination.get("pidfd_closed_after_source_revalidation_attempt")
        is not expected_pidfd_close
    ):
        raise Phase3ReportError(
            "v3 worker termination is not exactly joined to process evidence"
        )

    normalized_outcome = {
        key: value
        for key, value in outcome.items()
        if key != "owned_completion_basis"
    }
    normalized_audit = {
        key: value
        for key, value in process_audit.items()
        if key not in audit_v3_extra
    }
    normalized_audit["schema_version"] = _PROCESS_AUDIT_V2
    normalized_registry = {
        key: value for key, value in registry.items() if key not in registry_v3_extra
    }
    normalized_registry["schema_version"] = _PROCESS_REGISTRY_V2
    normalized_registry["handshake_events"] = normalized_events
    normalized_registry["outcome"] = normalized_outcome
    normalized_handshake = {
        key: value
        for key, value in handshake.items()
        if key not in handshake_v3_extra
    }
    normalized_handshake["schema_version"] = _PROCESS_HANDSHAKE_V2
    normalized_handshake["events"] = normalized_events
    normalized_handshake["terminal_outcome"] = normalized_outcome
    return normalized_audit, normalized_registry, normalized_handshake


def _validate_process_evidence_v2_pass(
    run_dir: Path,
    manifest: Phase3RunManifest,
    process_audit: Mapping[str, Any],
    ready: Mapping[str, Any],
    worker_result: Phase3WorkerResult,
    *,
    registry_evidence: Mapping[str, Any] | None = None,
    handshake_evidence: Mapping[str, Any] | None = None,
) -> None:
    del worker_result
    expected_command_fingerprint = command_fingerprint(
        manifest.command.argv,
        working_directory=manifest.command.working_directory,
        environment_sha256=manifest.command.environment_sha256,
    )
    expected_audit_keys = {
        "schema_version",
        "passed",
        "certified_helper",
        "registry_created",
        "ownership_verdict",
        "exclusivity_passed",
        "evidence_flushed",
        "worker_exiting_observed",
        "pid_start_time_protected",
        "pidfd_supported",
        "pidfd_opened",
        "pidfd_closed",
        "failure_reason",
        "foreign_compute_allowed",
        "unknown_compute_allowed",
    }
    if (
        set(process_audit) != expected_audit_keys
        or process_audit.get("schema_version")
        != "kvbench-phase3-process-audit-2.0.0"
        or process_audit.get("passed") is not True
        or process_audit.get("certified_helper") != "preflight/process_query.py"
        or process_audit.get("registry_created") is not True
        or process_audit.get("ownership_verdict") != "owned_completed"
        or process_audit.get("exclusivity_passed") is not True
        or process_audit.get("evidence_flushed") is not True
        or process_audit.get("worker_exiting_observed") is not True
        or process_audit.get("pid_start_time_protected") is not True
        or not isinstance(process_audit.get("pidfd_supported"), bool)
        or not isinstance(process_audit.get("pidfd_opened"), bool)
        or not isinstance(process_audit.get("pidfd_closed"), bool)
        or process_audit.get("pidfd_closed") != process_audit.get("pidfd_opened")
        or process_audit.get("failure_reason") is not None
        or process_audit.get("foreign_compute_allowed") is not False
        or process_audit.get("unknown_compute_allowed") is not False
    ):
        raise Phase3ReportError("v2 process audit outcome is not an exact pass")
    if (
        set(ready)
        != {
            "schema_version",
            "pid",
            "process_start_time_ticks",
            "cuda_imported",
        }
        or ready.get("schema_version")
        != "kvbench-phase3-worker-ready-1.0.0"
        or not _process_integer(ready.get("pid"), positive=True)
        or not _process_integer(ready.get("process_start_time_ticks"))
        or ready.get("cuda_imported") is not False
    ):
        raise Phase3ReportError("v2 worker readiness identity is invalid")

    registry = (
        _strict_json_object(run_dir / "environment" / "process.registry.json")
        if registry_evidence is None
        else dict(registry_evidence)
    )
    handshake = (
        _strict_json_object(run_dir / "environment" / "process.handshake.json")
        if handshake_evidence is None
        else dict(handshake_evidence)
    )
    worker_evidence = _strict_json_object(
        run_dir / "raw" / "worker_evidence.json"
    )
    identity = _mapping(registry.get("identity"))
    handle = _mapping(registry.get("handle"))
    outcome = _mapping(registry.get("outcome"))
    expected_registry_keys = {
        "schema_version",
        "identity",
        "handle",
        "handshake_events",
        "exit_observed_without_reaping",
        "supervisor_reaped",
        "proc_disappeared_after_registration",
        "device_snapshot_count",
        "registered_compute_observed",
        "outcome",
        "pidfd_closed_by_supervisor",
        "process_handle_reaped_by_supervisor",
    }
    expected_identity_keys = {
        "pid",
        "start_time_ticks",
        "parent_pid",
        "run_id",
        "gpu_uuid",
        "spawned_at_utc",
        "expected_command_fingerprint",
    }
    expected_handle_keys = {
        "process_handle_kind",
        "process_handle_retained",
        "pidfd_supported",
        "pidfd_opened",
        "pidfd",
    }
    if (
        set(registry) != expected_registry_keys
        or registry.get("schema_version")
        != "kvbench-phase3-process-registry-2.0.0"
        or identity is None
        or set(identity) != expected_identity_keys
        or identity.get("pid") != ready.get("pid")
        or identity.get("start_time_ticks")
        != ready.get("process_start_time_ticks")
        or not _process_integer(identity.get("parent_pid"), positive=True)
        or identity.get("run_id") != manifest.run_id
        or identity.get("gpu_uuid") != manifest.gpu_uuid
        or _process_utc_timestamp(identity.get("spawned_at_utc")) is None
        or identity.get("expected_command_fingerprint")
        != expected_command_fingerprint
        or handle is None
        or set(handle) != expected_handle_keys
        or not isinstance(handle.get("process_handle_kind"), str)
        or not handle.get("process_handle_kind")
        or handle.get("process_handle_retained") is not True
        or not isinstance(handle.get("pidfd_supported"), bool)
        or not isinstance(handle.get("pidfd_opened"), bool)
        or (
            handle.get("pidfd_opened") is True
            and not _process_integer(handle.get("pidfd"))
        )
        or (
            handle.get("pidfd_opened") is False
            and handle.get("pidfd") is not None
        )
        or registry.get("exit_observed_without_reaping") is not True
        or registry.get("supervisor_reaped") is not True
        or not isinstance(
            registry.get("proc_disappeared_after_registration"), bool
        )
        or not _process_integer(registry.get("device_snapshot_count"))
        or not isinstance(registry.get("registered_compute_observed"), bool)
        or outcome is None
        or registry.get("pidfd_closed_by_supervisor")
        is not handle.get("pidfd_opened")
        or registry.get("process_handle_reaped_by_supervisor") is not True
        or process_audit.get("pidfd_supported")
        is not handle.get("pidfd_supported")
        or process_audit.get("pidfd_opened") is not handle.get("pidfd_opened")
    ):
        raise Phase3ReportError("v2 process registry identity or handle differs")

    _validate_v2_handshake(
        manifest=manifest,
        registry=registry,
        handshake=handshake,
        worker_evidence=worker_evidence,
    )
    before = _strict_json_object(run_dir / "environment" / "process.before.json")
    release = _strict_json_object(
        run_dir / "environment" / "process.release_audit.json"
    )
    during = _strict_json_object(run_dir / "environment" / "process.during.json")
    after = _strict_json_object(run_dir / "environment" / "process.after.json")
    release_verdict = _strict_json_object(
        run_dir / "environment" / "process.release_registry_verdict.json"
    )
    after_verdict = _strict_json_object(
        run_dir / "environment" / "process.after_registry_verdict.json"
    )

    before_snapshot, before_observations = _v2_raw_process_snapshot(
        before,
        registered_identity=identity,
    )
    if (
        before_snapshot.get("query_exit_code") != 0
        or before_snapshot.get("errors") != []
        or before_observations
    ):
        raise Phase3ReportError("v2 pre-spawn process snapshot is not clean")
    release_join = _v2_registry_snapshot_verdict(
        release,
        release_verdict,
        registered_identity=identity,
        terminal_resolution_allowed=False,
        proc_disappeared_after_registration=bool(
            registry["proc_disappeared_after_registration"]
        ),
    )
    expected_monitor_keys = {
        "schema_version",
        "sampling_target_seconds",
        "samples",
        "sample_registry_verdicts",
        "saw_registered_compute",
        "fast_exit_before_first_telemetry_poll",
        "monitoring_stopped_before_worker_exit",
    }
    samples = _sequence(during.get("samples"))
    sample_verdicts = _sequence(during.get("sample_registry_verdicts"))
    fast_exit = during.get("fast_exit_before_first_telemetry_poll")
    if (
        set(during) != expected_monitor_keys
        or during.get("schema_version")
        != "kvbench-phase3-process-monitor-2.0.0"
        or during.get("sampling_target_seconds") != 2.0
        or samples is None
        or sample_verdicts is None
        or len(samples) != len(sample_verdicts)
        or not isinstance(during.get("saw_registered_compute"), bool)
        or not isinstance(fast_exit, bool)
        or during.get("monitoring_stopped_before_worker_exit") is not False
        or fast_exit != (len(samples) == 0)
        or (
            fast_exit
            and during.get("saw_registered_compute") is not False
        )
        or (
            not fast_exit
            and (
                not samples
                or during.get("saw_registered_compute") is not True
            )
        )
    ):
        raise Phase3ReportError("v2 continuous process monitor differs")

    joined_sample_verdicts: list[Mapping[str, Any]] = []
    for index, (sample, verdict) in enumerate(
        zip(samples, sample_verdicts)
    ):
        joined_sample_verdicts.append(
            _v2_registry_snapshot_verdict(
                sample,
                verdict,
                registered_identity=identity,
                terminal_resolution_allowed=index == len(samples) - 1,
                proc_disappeared_after_registration=bool(
                    registry["proc_disappeared_after_registration"]
                ),
            )
        )
    after_join = _v2_registry_snapshot_verdict(
        after,
        after_verdict,
        registered_identity=identity,
        terminal_resolution_allowed=True,
        proc_disappeared_after_registration=bool(
            registry["proc_disappeared_after_registration"]
        ),
    )
    all_joins = [release_join, *joined_sample_verdicts, after_join]
    observed_registered_compute = any(
        bool(_mapping(verdict.get("registry_verdict")).get("owned"))
        for verdict in all_joins
        if _mapping(verdict.get("registry_verdict")) is not None
    )
    during_registered_compute = any(
        bool(_mapping(verdict.get("registry_verdict")).get("owned"))
        for verdict in joined_sample_verdicts
        if _mapping(verdict.get("registry_verdict")) is not None
    )
    if (
        any(verdict.get("passed") is not True for verdict in all_joins)
        or during.get("saw_registered_compute") is not during_registered_compute
        or registry.get("registered_compute_observed")
        is not observed_registered_compute
        or registry.get("device_snapshot_count") != len(samples) + 2
    ):
        raise Phase3ReportError("v2 process audit and monitor linkage differs")


def _v2_optional_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    return _strict_json_object(path)


def _v2_failure_readiness_observed(ready: Mapping[str, Any]) -> bool:
    if dict(ready) == _PROCESS_READY_NOT_OBSERVED_V2:
        return False
    if (
        set(ready)
        != {
            "schema_version",
            "pid",
            "process_start_time_ticks",
            "cuda_imported",
        }
        or ready.get("schema_version")
        != "kvbench-phase3-worker-ready-1.0.0"
        or not _process_integer(ready.get("pid"), positive=True)
        or not _process_integer(ready.get("process_start_time_ticks"))
        or ready.get("cuda_imported") is not False
    ):
        raise Phase3ReportError("v2 failed-run readiness evidence is invalid")
    return True


def _validate_v2_failure_audit_join(
    manifest: Phase3RunManifest,
    process_audit: Mapping[str, Any],
    worker_result: Phase3WorkerResult,
) -> bool:
    expected_keys = {
        "schema_version",
        "passed",
        "certified_helper",
        "registry_created",
        "ownership_verdict",
        "exclusivity_passed",
        "evidence_flushed",
        "worker_exiting_observed",
        "pid_start_time_protected",
        "pidfd_supported",
        "pidfd_opened",
        "pidfd_closed",
        "failure_reason",
        "foreign_compute_allowed",
        "unknown_compute_allowed",
    }
    boolean_fields = (
        "registry_created",
        "exclusivity_passed",
        "evidence_flushed",
        "worker_exiting_observed",
        "pid_start_time_protected",
        "pidfd_supported",
        "pidfd_opened",
        "pidfd_closed",
    )
    failure_reason = process_audit.get("failure_reason")
    if (
        set(process_audit) != expected_keys
        or process_audit.get("schema_version")
        != "kvbench-phase3-process-audit-2.0.0"
        or process_audit.get("passed") is not False
        or process_audit.get("certified_helper")
        != "preflight/process_query.py"
        or any(not isinstance(process_audit.get(field), bool) for field in boolean_fields)
        or (
            process_audit.get("ownership_verdict") is not None
            and process_audit.get("ownership_verdict")
            not in {
                "owned_worker_failure",
                "foreign_process_detected",
                "pid_reuse_detected",
                "unverified_process_detected",
            }
        )
        or process_audit.get("pidfd_opened") is True
        and process_audit.get("pidfd_supported") is not True
        or process_audit.get("pidfd_closed")
        is not process_audit.get("pidfd_opened")
        or not isinstance(failure_reason, str)
        or not failure_reason
        or len(failure_reason) > 1000
        or " ".join(failure_reason.split()) != failure_reason
        or process_audit.get("foreign_compute_allowed") is not False
        or process_audit.get("unknown_compute_allowed") is not False
        or manifest.status is not RunStatus.ABORTED
        or worker_result.status is not RunStatus.ABORTED
        or manifest.failure_reason != failure_reason
        or worker_result.failure_reason != failure_reason
    ):
        raise Phase3ReportError(
            "v2 process audit failure is not exactly joined to an aborted run"
        )
    return bool(process_audit["registry_created"])


def _v2_validate_failure_after_artifact(
    run_dir: Path,
    *,
    registered_identity: Mapping[str, object],
) -> Mapping[str, Any] | None:
    after = _v2_optional_json_object(
        run_dir / "environment" / "process.after.json"
    )
    after_error = _v2_optional_json_object(
        run_dir / "environment" / "process.after_error.json"
    )
    if (after is None) == (after_error is None):
        raise Phase3ReportError(
            "v2 failed run must retain exactly one post-worker process outcome"
        )
    if after_error is not None:
        if (
            set(after_error) != {"type", "message"}
            or not isinstance(after_error.get("type"), str)
            or not after_error.get("type")
            or after_error.get("message")
            != "post-worker process snapshot failed"
        ):
            raise Phase3ReportError("v2 post-worker process error differs")
        return None
    assert after is not None
    _v2_raw_process_snapshot(
        after,
        registered_identity=registered_identity,
    )
    return after


def _validate_v2_registry_not_created_failure(
    run_dir: Path,
    manifest: Phase3RunManifest,
    process_audit: Mapping[str, Any],
    ready: Mapping[str, Any],
) -> None:
    if dict(ready) != _PROCESS_READY_NOT_OBSERVED_V2:
        raise Phase3ReportError(
            "v2 unregistered failure lacks the not-observed readiness sentinel"
        )
    registry = _strict_json_object(
        run_dir / "environment" / "process.registry.json"
    )
    expected_registry = {
        "schema_version": "kvbench-phase3-process-registry-2.0.0",
        "registry_created": False,
        "run_id": manifest.run_id,
        "pidfd_supported": process_audit["pidfd_supported"],
        "pidfd_closed_by_supervisor": process_audit["pidfd_opened"],
    }
    handshake = _strict_json_object(
        run_dir / "environment" / "process.handshake.json"
    )
    expected_handshake = {
        "schema_version": "kvbench-phase3-worker-handshake-2.0.0",
        "run_id": manifest.run_id,
        "events": [],
        "terminal_outcome": None,
        "evidence_flushed_required_for_owned_completion": True,
    }
    if (
        registry != expected_registry
        or handshake != expected_handshake
        or process_audit.get("ownership_verdict") is not None
        or process_audit.get("exclusivity_passed") is not False
        or process_audit.get("evidence_flushed") is not False
        or process_audit.get("worker_exiting_observed") is not False
        or process_audit.get("pid_start_time_protected") is not False
    ):
        raise Phase3ReportError("v2 unregistered process failure evidence differs")
    forbidden = (
        "process.release_audit.json",
        "process.release_registry_verdict.json",
        "process.during.json",
        "process.after_registry_verdict.json",
    )
    if any(
        _v2_optional_json_object(run_dir / "environment" / name) is not None
        for name in forbidden
    ) or _v2_optional_json_object(run_dir / "raw" / "worker_evidence.json") is not None:
        raise Phase3ReportError(
            "v2 unregistered failure retains impossible worker artifacts"
        )
    placeholder_identity = {
        "pid": 0,
        "start_time_ticks": 0,
        "gpu_uuid": manifest.gpu_uuid,
    }
    before = _v2_optional_json_object(
        run_dir / "environment" / "process.before.json"
    )
    if before is not None:
        before_snapshot, _ = _v2_raw_process_snapshot(
            before,
            registered_identity=placeholder_identity,
        )
        if before_snapshot.get("allowed_compute_processes") != []:
            raise Phase3ReportError(
                "v2 unregistered pre-spawn evidence claims a supervised process"
            )
    after = _v2_validate_failure_after_artifact(
        run_dir,
        registered_identity=placeholder_identity,
    )
    if after is not None and after.get("allowed_compute_processes") != []:
        raise Phase3ReportError(
            "v2 unregistered post-worker evidence claims a supervised process"
        )


def _v2_registered_failure_registry(
    registry: Mapping[str, Any],
    *,
    manifest: Phase3RunManifest,
    process_audit: Mapping[str, Any],
    ready: Mapping[str, Any],
    readiness_observed: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    expected_registry_keys = {
        "schema_version",
        "identity",
        "handle",
        "handshake_events",
        "exit_observed_without_reaping",
        "supervisor_reaped",
        "proc_disappeared_after_registration",
        "device_snapshot_count",
        "registered_compute_observed",
        "outcome",
        "pidfd_closed_by_supervisor",
        "process_handle_reaped_by_supervisor",
    }
    expected_identity_keys = {
        "pid",
        "start_time_ticks",
        "parent_pid",
        "run_id",
        "gpu_uuid",
        "spawned_at_utc",
        "expected_command_fingerprint",
    }
    expected_handle_keys = {
        "process_handle_kind",
        "process_handle_retained",
        "pidfd_supported",
        "pidfd_opened",
        "pidfd",
    }
    identity = _mapping(registry.get("identity"))
    handle = _mapping(registry.get("handle"))
    outcome = _mapping(registry.get("outcome"))
    expected_command_fingerprint = command_fingerprint(
        manifest.command.argv,
        working_directory=manifest.command.working_directory,
        environment_sha256=manifest.command.environment_sha256,
    )
    if (
        set(registry) != expected_registry_keys
        or registry.get("schema_version")
        != "kvbench-phase3-process-registry-2.0.0"
        or identity is None
        or set(identity) != expected_identity_keys
        or not _process_integer(identity.get("pid"), positive=True)
        or not _process_integer(identity.get("start_time_ticks"))
        or not _process_integer(identity.get("parent_pid"), positive=True)
        or identity.get("run_id") != manifest.run_id
        or identity.get("gpu_uuid") != manifest.gpu_uuid
        or _process_utc_timestamp(identity.get("spawned_at_utc")) is None
        or identity.get("expected_command_fingerprint")
        != expected_command_fingerprint
        or handle is None
        or set(handle) != expected_handle_keys
        or not isinstance(handle.get("process_handle_kind"), str)
        or not handle.get("process_handle_kind")
        or handle.get("process_handle_retained") is not True
        or not isinstance(handle.get("pidfd_supported"), bool)
        or not isinstance(handle.get("pidfd_opened"), bool)
        or handle.get("pidfd_opened") is True
        and handle.get("pidfd_supported") is not True
        or handle.get("pidfd_opened") is True
        and not _process_integer(handle.get("pidfd"))
        or handle.get("pidfd_opened") is False
        and handle.get("pidfd") is not None
        or registry.get("exit_observed_without_reaping") is not True
        or registry.get("supervisor_reaped") is not True
        or not isinstance(
            registry.get("proc_disappeared_after_registration"), bool
        )
        or not _process_integer(registry.get("device_snapshot_count"))
        or not isinstance(registry.get("registered_compute_observed"), bool)
        or outcome is None
        or registry.get("pidfd_closed_by_supervisor")
        is not handle.get("pidfd_opened")
        or registry.get("process_handle_reaped_by_supervisor") is not True
        or process_audit.get("registry_created") is not True
        or process_audit.get("pid_start_time_protected") is not True
        or process_audit.get("pidfd_supported")
        is not handle.get("pidfd_supported")
        or process_audit.get("pidfd_opened") is not handle.get("pidfd_opened")
    ):
        raise Phase3ReportError("v2 failed process registry identity differs")
    if readiness_observed and (
        ready.get("pid") != identity.get("pid")
        or ready.get("process_start_time_ticks")
        != identity.get("start_time_ticks")
    ):
        raise Phase3ReportError(
            "v2 failed-run readiness and registry identities differ"
        )
    return identity, outcome


def _validate_v2_failure_handshake(
    *,
    manifest: Phase3RunManifest,
    registry: Mapping[str, Any],
    handshake: Mapping[str, Any],
    identity: Mapping[str, Any],
    outcome: Mapping[str, Any],
    readiness_observed: bool,
) -> None:
    events = _sequence(registry.get("handshake_events"))
    if events is None or not events:
        raise Phase3ReportError("v2 registered failure lacks a terminal handshake")
    event_stages = [
        _mapping(event).get("stage") if _mapping(event) is not None else None
        for event in events
    ]
    worker_event_count = len(events) - 1
    expected_stages = [
        *_PROCESS_WORKER_HANDSHAKE_STAGES[:worker_event_count],
        "supervisor_reaped",
    ]
    expected_command_fingerprint = command_fingerprint(
        manifest.command.argv,
        working_directory=manifest.command.working_directory,
        environment_sha256=manifest.command.environment_sha256,
    )
    spawned_at = _process_utc_timestamp(identity.get("spawned_at_utc"))
    if (
        worker_event_count < 0
        or worker_event_count > len(_PROCESS_WORKER_HANDSHAKE_STAGES)
        or event_stages != expected_stages
        or spawned_at is None
        or identity.get("expected_command_fingerprint")
        != expected_command_fingerprint
        or set(handshake)
        != {
            "schema_version",
            "run_id",
            "events",
            "terminal_outcome",
            "evidence_flushed_required_for_owned_completion",
        }
        or handshake.get("schema_version")
        != "kvbench-phase3-worker-handshake-2.0.0"
        or handshake.get("run_id") != manifest.run_id
        or handshake.get("events") != list(events)
        or handshake.get("terminal_outcome") != dict(outcome)
        or handshake.get("evidence_flushed_required_for_owned_completion") is not True
        or readiness_observed
        and "worker_started" not in event_stages
    ):
        raise Phase3ReportError("v2 failed worker handshake envelope differs")

    previous_timestamp: datetime | None = None
    for raw_event, expected_stage in zip(events, expected_stages):
        event = _mapping(raw_event)
        timestamp = (
            None
            if event is None
            else _process_utc_timestamp(event.get("recorded_at_utc"))
        )
        expected_digest = event.get("evidence_sha256") if event is not None else None
        expected_sequence = _PROCESS_HANDSHAKE_STAGES.index(expected_stage) + 1
        if (
            event is None
            or set(event)
            != {
                "schema_version",
                "sequence",
                "stage",
                "recorded_at_utc",
                "run_id",
                "gpu_uuid",
                "pid",
                "process_start_time_ticks",
                "parent_pid",
                "command_fingerprint",
                "evidence_sha256",
            }
            or event.get("schema_version")
            != "kvbench-phase3-worker-handshake-event-1.0.0"
            or event.get("sequence") != expected_sequence
            or event.get("stage") != expected_stage
            or timestamp is None
            or timestamp < spawned_at
            or previous_timestamp is not None
            and timestamp < previous_timestamp
            or event.get("run_id") != identity.get("run_id")
            or event.get("gpu_uuid") != identity.get("gpu_uuid")
            or event.get("pid") != identity.get("pid")
            or event.get("process_start_time_ticks")
            != identity.get("start_time_ticks")
            or event.get("parent_pid") != identity.get("parent_pid")
            or event.get("command_fingerprint")
            != expected_command_fingerprint
            or expected_stage == "evidence_flushed"
            and (
                not isinstance(expected_digest, str)
                or _PROCESS_SHA256.fullmatch(expected_digest) is None
            )
            or expected_stage != "evidence_flushed"
            and expected_digest is not None
        ):
            raise Phase3ReportError("v2 failed worker handshake event differs")
        previous_timestamp = timestamp

    expected_outcome_keys = {
        "disposition",
        "reason",
        "returncode",
        "observed_stages",
        "missing_worker_stages",
        "evidence_flushed",
        "worker_exiting_observed",
        "full_handshake_observed",
        "exclusivity_passed",
    }
    disposition = outcome.get("disposition")
    returncode = outcome.get("returncode")
    evidence_flushed = "evidence_flushed" in event_stages
    worker_exiting = "worker_exiting" in event_stages
    full_handshake = worker_event_count == len(_PROCESS_WORKER_HANDSHAKE_STAGES)
    missing_stages = list(
        _PROCESS_WORKER_HANDSHAKE_STAGES[worker_event_count:]
    )
    hard_dispositions = {
        "foreign_process_detected",
        "pid_reuse_detected",
        "unverified_process_detected",
    }
    if (
        set(outcome) != expected_outcome_keys
        or disposition not in {"owned_worker_failure", *hard_dispositions}
        or not isinstance(returncode, int)
        or isinstance(returncode, bool)
        or outcome.get("observed_stages") != expected_stages
        or outcome.get("missing_worker_stages") != missing_stages
        or outcome.get("evidence_flushed") is not evidence_flushed
        or outcome.get("worker_exiting_observed") is not worker_exiting
        or outcome.get("full_handshake_observed") is not full_handshake
        or not isinstance(outcome.get("reason"), str)
        or not outcome.get("reason")
    ):
        raise Phase3ReportError("v2 failed worker ownership outcome differs")
    if disposition == "owned_worker_failure":
        expected_reason = (
            "registered worker exited before evidence_flushed"
            if not evidence_flushed
            else f"registered worker exited with return code {returncode}"
        )
        if (
            returncode == 0
            and evidence_flushed
            or outcome.get("reason") != expected_reason
            or outcome.get("exclusivity_passed") is not True
        ):
            raise Phase3ReportError("v2 owned worker failure outcome differs")
    elif outcome.get("exclusivity_passed") is not False:
        raise Phase3ReportError("v2 hard process failure passed exclusivity")


def _v2_hard_failure_from_verdict(
    verdict: Mapping[str, Any],
) -> tuple[int, str, str] | None:
    registry_verdict = _mapping(verdict.get("registry_verdict"))
    if registry_verdict is None:
        raise Phase3ReportError("v2 failed registry verdict is malformed")
    disposition = registry_verdict.get("disposition")
    candidate: tuple[int, str, str] | None = None
    if disposition == "pid_reuse_detected":
        candidate = (
            3,
            "pid_reuse_detected",
            "device snapshot observed registered PID with a new start time",
        )
    elif disposition == "foreign_process_detected":
        candidate = (
            2,
            "foreign_process_detected",
            "device snapshot contains an unregistered process",
        )
    elif disposition == "unverified_registered_pid":
        candidate = (
            1,
            "unverified_process_detected",
            "device snapshot PID lacks a retained identity basis",
        )
    if verdict.get("query_evidence_hard_failure") is True and (
        candidate is None or candidate[0] < 1
    ):
        candidate = (
            1,
            "unverified_process_detected",
            "GPU process query failed outside exact terminal worker resolution",
        )
    return candidate


def _v2_validate_failure_monitor(
    during: Mapping[str, Any],
    *,
    registered_identity: Mapping[str, object],
    proc_disappeared_after_registration: bool,
) -> tuple[list[Mapping[str, Any]], str]:
    common_keys = {
        "schema_version",
        "sampling_target_seconds",
        "samples",
        "sample_registry_verdicts",
    }
    completed_keys = {
        *common_keys,
        "saw_registered_compute",
        "fast_exit_before_first_telemetry_poll",
        "monitoring_stopped_before_worker_exit",
    }
    samples = _sequence(during.get("samples"))
    verdicts = _sequence(during.get("sample_registry_verdicts"))
    if (
        frozenset(during) not in {frozenset(common_keys), frozenset(completed_keys)}
        or during.get("schema_version")
        != "kvbench-phase3-process-monitor-2.0.0"
        or during.get("sampling_target_seconds") != 2.0
        or samples is None
        or verdicts is None
        or len(samples) != len(verdicts)
    ):
        raise Phase3ReportError("v2 failed process monitor envelope differs")
    joined: list[Mapping[str, Any]] = []
    for index, (sample, verdict) in enumerate(zip(samples, verdicts)):
        joined.append(
            _v2_registry_snapshot_verdict(
                sample,
                verdict,
                registered_identity=registered_identity,
                terminal_resolution_allowed=index == len(samples) - 1,
                proc_disappeared_after_registration=(
                    proc_disappeared_after_registration
                ),
            )
        )
    if set(during) == common_keys:
        if (
            not joined
            or any(item.get("passed") is not True for item in joined[:-1])
            or joined[-1].get("passed") is not False
        ):
            raise Phase3ReportError(
                "v2 truncated process monitor does not end at its first failure"
            )
        return joined, "failed"

    fast_exit = during.get("fast_exit_before_first_telemetry_poll")
    observed_compute = any(
        bool(_mapping(item.get("registry_verdict")).get("owned"))
        for item in joined
        if _mapping(item.get("registry_verdict")) is not None
    )
    if (
        any(item.get("passed") is not True for item in joined)
        or not isinstance(during.get("saw_registered_compute"), bool)
        or not isinstance(fast_exit, bool)
        or fast_exit != (len(joined) == 0)
        or during.get("saw_registered_compute") is not observed_compute
        or during.get("monitoring_stopped_before_worker_exit") is not False
    ):
        raise Phase3ReportError("v2 completed failed-run process monitor differs")
    return joined, "completed"


def _v2_validate_registered_failure_process_artifacts(
    run_dir: Path,
    *,
    identity: Mapping[str, Any],
    registry: Mapping[str, Any],
    readiness_observed: bool,
) -> tuple[tuple[int, str, str] | None, str | None]:
    before = _strict_json_object(
        run_dir / "environment" / "process.before.json"
    )
    before_snapshot, before_observations = _v2_raw_process_snapshot(
        before,
        registered_identity=identity,
    )
    if (
        before_snapshot.get("query_exit_code") != 0
        or before_snapshot.get("errors") != []
        or before_observations
    ):
        raise Phase3ReportError("v2 registered failure has dirty pre-spawn evidence")

    release = _v2_optional_json_object(
        run_dir / "environment" / "process.release_audit.json"
    )
    release_verdict = _v2_optional_json_object(
        run_dir / "environment" / "process.release_registry_verdict.json"
    )
    during = _v2_optional_json_object(
        run_dir / "environment" / "process.during.json"
    )
    after_verdict = _v2_optional_json_object(
        run_dir / "environment" / "process.after_registry_verdict.json"
    )
    after = _v2_validate_failure_after_artifact(
        run_dir,
        registered_identity=identity,
    )
    if not readiness_observed and any(
        item is not None
        for item in (release, release_verdict, during, after_verdict)
    ):
        raise Phase3ReportError(
            "v2 not-ready worker has post-readiness process artifacts"
        )
    if release_verdict is not None and release is None:
        raise Phase3ReportError("v2 release verdict lacks its raw snapshot")
    if release is not None and release_verdict is None:
        _v2_raw_process_snapshot(release, registered_identity=identity)

    proc_disappeared = bool(
        registry["proc_disappeared_after_registration"]
    )
    joined: list[Mapping[str, Any]] = []
    release_join: Mapping[str, Any] | None = None
    if release is not None and release_verdict is not None:
        release_join = _v2_registry_snapshot_verdict(
            release,
            release_verdict,
            registered_identity=identity,
            terminal_resolution_allowed=False,
            proc_disappeared_after_registration=proc_disappeared,
        )
        joined.append(release_join)

    monitor_kind: str | None = None
    monitor_joins: list[Mapping[str, Any]] = []
    if during is not None:
        if release_join is None or release_join.get("passed") is not True:
            raise Phase3ReportError(
                "v2 process monitoring started without a passed release audit"
            )
        monitor_joins, monitor_kind = _v2_validate_failure_monitor(
            during,
            registered_identity=identity,
            proc_disappeared_after_registration=proc_disappeared,
        )
        joined.extend(monitor_joins)

    after_join: Mapping[str, Any] | None = None
    if after_verdict is not None:
        if (
            after is None
            or release_join is None
            or release_join.get("passed") is not True
            or monitor_kind != "completed"
        ):
            raise Phase3ReportError(
                "v2 post-reap verdict lacks the completed monitoring prefix"
            )
        after_join = _v2_registry_snapshot_verdict(
            after,
            after_verdict,
            registered_identity=identity,
            terminal_resolution_allowed=True,
            proc_disappeared_after_registration=proc_disappeared,
        )
        joined.append(after_join)

    failure_location: str | None = None
    failed_joins: list[Mapping[str, Any]] = []
    if release_join is not None and release_join.get("passed") is False:
        failure_location = "release"
        failed_joins.append(release_join)
        if during is not None or after_verdict is not None:
            raise Phase3ReportError(
                "v2 process artifacts continue after a failed release audit"
            )
    if monitor_kind == "failed":
        if failure_location is not None or after_verdict is not None:
            raise Phase3ReportError(
                "v2 process artifacts continue after a failed monitor sample"
            )
        failure_location = "during"
        failed_joins.append(monitor_joins[-1])
    if after_join is not None and after_join.get("passed") is False:
        if failure_location is not None:
            raise Phase3ReportError("v2 process evidence contains two failures")
        failure_location = "after"
        failed_joins.append(after_join)
    if any(
        item.get("passed") is not True and item not in failed_joins
        for item in joined
    ):
        raise Phase3ReportError("v2 process verdict sequence is inconsistent")

    hard_failure: tuple[int, str, str] | None = None
    for verdict in joined:
        candidate = _v2_hard_failure_from_verdict(verdict)
        if candidate is not None and (
            hard_failure is None or candidate[0] > hard_failure[0]
        ):
            hard_failure = candidate
    observed_compute = any(
        bool(_mapping(item.get("registry_verdict")).get("owned"))
        for item in joined
        if _mapping(item.get("registry_verdict")) is not None
    )
    missing_start_time_owned = False
    for item in joined:
        registry_verdict = _mapping(item.get("registry_verdict"))
        owned = (
            None
            if registry_verdict is None
            else _sequence(registry_verdict.get("owned"))
        )
        if owned is not None and any(
            _mapping(observation) is not None
            and _mapping(observation).get("process_start_time_ticks") is None
            for observation in owned
        ):
            missing_start_time_owned = True
            break
    if (
        registry.get("device_snapshot_count") != len(joined)
        or registry.get("registered_compute_observed") is not observed_compute
        or registry.get("proc_disappeared_after_registration")
        is not missing_start_time_owned
    ):
        raise Phase3ReportError(
            "v2 failed process registry counters differ from raw evidence"
        )
    return hard_failure, failure_location


def _validate_process_evidence_v2_failure(
    run_dir: Path,
    manifest: Phase3RunManifest,
    process_audit: Mapping[str, Any],
    ready: Mapping[str, Any],
    worker_result: Phase3WorkerResult,
    *,
    registry_evidence: Mapping[str, Any] | None = None,
    handshake_evidence: Mapping[str, Any] | None = None,
) -> None:
    registry_created = _validate_v2_failure_audit_join(
        manifest,
        process_audit,
        worker_result,
    )
    readiness_observed = _v2_failure_readiness_observed(ready)
    if not registry_created:
        _validate_v2_registry_not_created_failure(
            run_dir,
            manifest,
            process_audit,
            ready,
        )
        return

    registry = (
        _strict_json_object(run_dir / "environment" / "process.registry.json")
        if registry_evidence is None
        else dict(registry_evidence)
    )
    handshake = (
        _strict_json_object(run_dir / "environment" / "process.handshake.json")
        if handshake_evidence is None
        else dict(handshake_evidence)
    )
    if _v2_optional_json_object(
        run_dir / "raw" / "worker_evidence.json"
    ) is not None:
        raise Phase3ReportError(
            "v2 process-supervision failure cannot retain parsed worker evidence"
        )
    identity, outcome = _v2_registered_failure_registry(
        registry,
        manifest=manifest,
        process_audit=process_audit,
        ready=ready,
        readiness_observed=readiness_observed,
    )
    _validate_v2_failure_handshake(
        manifest=manifest,
        registry=registry,
        handshake=handshake,
        identity=identity,
        outcome=outcome,
        readiness_observed=readiness_observed,
    )
    if (
        process_audit.get("ownership_verdict")
        != outcome.get("disposition")
        or process_audit.get("exclusivity_passed")
        is not outcome.get("exclusivity_passed")
        or process_audit.get("evidence_flushed")
        is not outcome.get("evidence_flushed")
        or process_audit.get("worker_exiting_observed")
        is not outcome.get("worker_exiting_observed")
    ):
        raise Phase3ReportError(
            "v2 failed process audit and ownership outcome differ"
        )

    hard_failure, failure_location = (
        _v2_validate_registered_failure_process_artifacts(
            run_dir,
            identity=identity,
            registry=registry,
            readiness_observed=readiness_observed,
        )
    )
    disposition = outcome.get("disposition")
    if disposition == "owned_worker_failure":
        if hard_failure is not None or failure_location is not None:
            raise Phase3ReportError(
                "v2 owned worker failure contains hard process evidence"
            )
        return

    if (
        hard_failure is None
        or failure_location is None
        or outcome.get("disposition") != hard_failure[1]
        or outcome.get("reason") != hard_failure[2]
    ):
        raise Phase3ReportError(
            "v2 hard ownership outcome is not derived from raw process evidence"
        )
    expected_failure_reasons = {
        "release": "Phase3CoordinatorError: worker release audit failed closed",
        "during": (
            "Phase3CoordinatorError: worker process audit detected foreign "
            "or unverified compute"
        ),
        "after": (
            "Phase3CoordinatorError: post-reap process audit detected foreign "
            "or unverified compute"
        ),
    }
    if (
        process_audit.get("failure_reason")
        != expected_failure_reasons[failure_location]
    ):
        raise Phase3ReportError(
            "v2 hard process failure reason differs from its evidence stage"
        )


def _validate_process_evidence_v2(
    run_dir: Path,
    manifest: Phase3RunManifest,
    process_audit: Mapping[str, Any],
    ready: Mapping[str, Any],
    worker_result: Phase3WorkerResult,
    *,
    registry_evidence: Mapping[str, Any] | None = None,
    handshake_evidence: Mapping[str, Any] | None = None,
) -> None:
    if process_audit.get("passed") is False:
        _validate_process_evidence_v2_failure(
            run_dir,
            manifest,
            process_audit,
            ready,
            worker_result,
            registry_evidence=registry_evidence,
            handshake_evidence=handshake_evidence,
        )
        return
    _validate_process_evidence_v2_pass(
        run_dir,
        manifest,
        process_audit,
        ready,
        worker_result,
        registry_evidence=registry_evidence,
        handshake_evidence=handshake_evidence,
    )


def _join_setup_and_process_evidence(
    run_dir: Path,
    manifest: Phase3RunManifest,
    process_audit: Mapping[str, Any],
    ready: Mapping[str, Any],
    worker_result: Phase3WorkerResult,
) -> None:
    plan_path = manifest.plan_source.path
    if plan_path is None:
        raise Phase3ReportError("manifest plan path is absent")
    bundle = load_phase3_admission_bundle(plan_path)
    plan_payload = _strict_json_object(run_dir / "config" / "plan.json")
    _require_equal("saved plan", plan_payload, bundle.plan.to_dict())
    references = _strict_json_object(
        run_dir / "config" / "referenced_fingerprints.json"
    )
    expected_references = {
        "schema_version": "kvbench-phase3-references-1.0.0",
        "fingerprints": [
            {"path": path, "sha256": digest}
            for path, digest in bundle.canonical_fingerprints
        ],
        "formal_blockers_retained": list(bundle.all_blockers),
    }
    _require_equal("referenced fingerprints", references, expected_references)
    live_hardware = _strict_json_object(
        run_dir / "environment" / "live_hardware.json"
    )
    expected_hardware = {
        "schema_version": "kvbench-phase3-live-hardware-1.0.0",
        "gpu_name": manifest.gpu_full_name,
        "gpu_uuid": manifest.gpu_uuid,
        "pci_bus_id": manifest.pci_bus_id,
        "pci_device_id": manifest.pci_device_id,
        "driver_version": manifest.driver_version,
        "compute_capability": "12.0",
        "native_g0_status": "PASS",
        "container_parity_status": "not_evaluated",
        "blocker_b010": "OPEN",
    }
    _require_equal("live hardware", live_hardware, expected_hardware)

    process_schema = process_audit.get("schema_version")
    if process_schema == _PROCESS_AUDIT_V3:
        normalized_audit, normalized_registry, normalized_handshake = (
            _normalize_v3_process_evidence(run_dir, manifest, process_audit)
        )
        _validate_process_evidence_v2(
            run_dir,
            manifest,
            normalized_audit,
            ready,
            worker_result,
            registry_evidence=normalized_registry,
            handshake_evidence=normalized_handshake,
        )
        return
    if process_schema == _PROCESS_AUDIT_V2:
        _validate_process_evidence_v2(
            run_dir, manifest, process_audit, ready, worker_result
        )
        return
    expected_process_audit = {
        "schema_version": "kvbench-phase3-process-audit-1.0.0",
        "certified_helper": "preflight/process_query.py",
        "foreign_compute_allowed": False,
        "unknown_compute_allowed": False,
    }
    audit_passed = process_audit.get("passed")
    if (
        set(process_audit) != {*expected_process_audit, "passed"}
        or any(
            process_audit.get(key) != value
            for key, value in expected_process_audit.items()
        )
        or not isinstance(audit_passed, bool)
    ):
        raise Phase3ReportError("process audit outcome is malformed")
    if not audit_passed and (
        manifest.status is not RunStatus.ABORTED
        or worker_result.status is not RunStatus.ABORTED
        or worker_result.failure_reason
        != "Phase3CoordinatorError: worker process audit detected foreign compute"
    ):
        raise Phase3ReportError("failed process audit is not joined to an aborted run")
    if (
        ready.get("schema_version") != "kvbench-phase3-worker-ready-1.0.0"
        or not isinstance(ready.get("pid"), int)
        or isinstance(ready.get("pid"), bool)
        or not isinstance(ready.get("process_start_time_ticks"), int)
        or isinstance(ready.get("process_start_time_ticks"), bool)
        or ready.get("process_start_time_ticks", -1) < 0
        or ready.get("cuda_imported") is not False
    ):
        raise Phase3ReportError("worker readiness identity is invalid")
    before = _strict_json_object(run_dir / "environment" / "process.before.json")
    release = _strict_json_object(
        run_dir / "environment" / "process.release_audit.json"
    )
    during = _strict_json_object(run_dir / "environment" / "process.during.json")
    after = _strict_json_object(run_dir / "environment" / "process.after.json")
    samples = _sequence(during.get("samples"))
    common_process_evidence_valid = bool(
        _process_snapshot_clean(before, allow_supervised=False)
        and _process_snapshot_clean(
            release,
            allow_supervised=True,
            ready=ready,
            gpu_uuid=manifest.gpu_uuid,
        )
        and _process_snapshot_clean(after, allow_supervised=False)
        and during.get("schema_version")
        == "kvbench-phase3-process-monitor-1.0.0"
        and during.get("sampling_target_seconds") == 2.0
        and samples is not None
        and bool(samples)
    )
    if not common_process_evidence_valid:
        raise Phase3ReportError("continuous GPU process evidence is malformed")
    assert samples is not None
    if audit_passed:
        passed = bool(
            set(during)
            == {
                "schema_version",
                "sampling_target_seconds",
                "samples",
                "saw_allowed_compute",
                "monitoring_stopped_before_worker_exit",
            }
            and during.get("saw_allowed_compute") is True
            and during.get("monitoring_stopped_before_worker_exit") is False
            and all(
                _process_snapshot_clean(
                    sample,
                    allow_supervised=True,
                    ready=ready,
                    gpu_uuid=manifest.gpu_uuid,
                )
                for sample in samples
            )
            and any(
                bool(_mapping(sample).get("allowed_compute_processes"))
                for sample in samples
                if _mapping(sample) is not None
            )
        )
        if not passed:
            raise Phase3ReportError(
                "continuous GPU process evidence is not an exact pass"
            )
        return

    terminal = _mapping(samples[-1])
    if terminal is None:
        raise Phase3ReportError("failed process audit terminal sample is malformed")
    sanitized_terminal = dict(terminal)
    sanitized_terminal["foreign_compute_processes"] = []
    sanitized_terminal["unknown_processes"] = []
    sanitized_terminal["errors"] = []
    sanitized_terminal["query_exit_code"] = 0
    failed_sequence_valid = bool(
        set(during) == {"schema_version", "sampling_target_seconds", "samples"}
        and all(
            _process_snapshot_clean(
                sample,
                allow_supervised=True,
                ready=ready,
                gpu_uuid=manifest.gpu_uuid,
            )
            for sample in samples[:-1]
        )
        and any(
            bool(_mapping(sample).get("allowed_compute_processes"))
            for sample in samples[:-1]
            if _mapping(sample) is not None
        )
        and _process_snapshot_clean(
            sanitized_terminal,
            allow_supervised=True,
            ready=ready,
            gpu_uuid=manifest.gpu_uuid,
        )
        and (
            bool(terminal.get("foreign_compute_processes"))
            or bool(terminal.get("unknown_processes"))
        )
        and isinstance(terminal.get("errors"), list)
        and bool(terminal.get("errors"))
    )
    if not failed_sequence_valid:
        raise Phase3ReportError("failed process audit evidence is not exact")


def _load_validated_run(run_dir: Path, expected_run_id: str) -> ValidatedPhase3Run:
    validation = validate_run_directory(run_dir)
    if not validation.valid or not validation.complete:
        raise Phase3ReportError(
            f"run is not complete and checksum-valid: {expected_run_id}: "
            + "; ".join(validation.errors)
        )
    if run_dir.name != expected_run_id:
        raise Phase3ReportError("run directory name differs from explicit campaign join")

    manifest_raw = _strict_json_object(run_dir / "manifest.json")
    parsed = parse_run_manifest(manifest_raw)
    if not isinstance(parsed, Phase3RunManifest):
        raise Phase3ReportError("selected campaign contains a non-Phase-3 manifest")
    manifest = parsed
    _require_equal("manifest run ID", manifest.run_id, expected_run_id)

    completion = CompletionMarker.from_dict(_strict_json_object(run_dir / "COMPLETE"))
    worker_result = Phase3WorkerResult.from_dict(
        _strict_json_object(run_dir / "validation" / "worker_result.json")
    )
    _require_equal("completion run ID", completion.run_id, manifest.run_id)
    _require_equal("completion status", completion.status, manifest.status)
    _require_equal("worker run ID", worker_result.run_id, manifest.run_id)
    _require_equal("worker point ID", worker_result.point_id, manifest.point_id)
    _require_equal("worker runner", worker_result.runner_kind, manifest.runner_kind)
    _require_equal("worker count unit", worker_result.count_unit, manifest.count_unit)
    _require_equal("worker/final status", worker_result.status, manifest.status)

    point_payload = _strict_json_object(run_dir / "validation" / "point.json")
    expected_point_payload = {
        "point_id": manifest.point_id,
        "runner_kind": manifest.runner_kind.value,
        "graph_mode": manifest.graph_mode.value,
        "batch_size": manifest.batch_size,
        "context_length": manifest.context_length,
        "output_steps": manifest.output_steps,
        "process_replicate": manifest.process_replicate,
        "stability_member": manifest.point_id
        in FROZEN_PHASE3_STABILITY_POINT_IDS,
        "point_fingerprint": manifest.point_fingerprint,
    }
    _require_equal("point payload", point_payload, expected_point_payload)
    _manifest_environment_join(run_dir, manifest)

    process_audit = _strict_json_object(
        run_dir / "validation" / "process_audit_outcome.json"
    )
    ready_process = _strict_json_object(
        run_dir / "environment" / "process.ready.json"
    )
    _join_setup_and_process_evidence(
        run_dir,
        manifest,
        process_audit,
        ready_process,
        worker_result,
    )
    worker_evidence = _optional_json_object(
        run_dir / "raw" / "worker_evidence.json"
    )
    runtime: Mapping[str, Any] | None = None
    numerical: Mapping[str, Any] | None = None
    timing: Mapping[str, Any] | None = None
    telemetry: Mapping[str, Any] | None = None
    if worker_evidence is not None:
        _require_equal("raw worker run ID", worker_evidence.get("run_id"), manifest.run_id)
        _require_equal("raw worker point ID", worker_evidence.get("point_id"), manifest.point_id)
        _require_equal(
            "raw worker result",
            worker_evidence.get("worker_result"),
            worker_result.to_dict(),
        )
        raw_runtime = worker_evidence.get("runtime")
        raw_numerical = worker_evidence.get("numerical")
        if raw_runtime is not None and not isinstance(raw_runtime, Mapping):
            raise Phase3ReportError("worker runtime evidence is not an object")
        if raw_numerical is not None and not isinstance(raw_numerical, Mapping):
            raise Phase3ReportError("worker numerical evidence is not an object")
        runtime = raw_runtime
        numerical = raw_numerical
        if runtime is not None:
            _require_equal(
                "runtime cache fingerprint",
                runtime.get("cache_layout_fingerprint"),
                manifest.cache_identity.layout_fingerprint,
            )
            timing, telemetry, _ = _split_runtime_join(
                run_dir,
                runtime,
                numerical,
            )
            _validate_telemetry_evidence(telemetry, manifest)
        elif any(
            (run_dir / relative).exists()
            for relative in (
                "allocation/audit.json",
                "gqa/audit.json",
                "raw/timing.json",
                "telemetry/snapshots.json",
            )
        ):
            raise Phase3ReportError("split runtime evidence exists without raw runtime")

    stdout_path = run_dir / "logs" / "worker.stdout.txt"
    stdout = stdout_path.read_bytes()
    if stdout:
        try:
            stdout_payload = json.loads(stdout.decode("utf-8"))
            stdout_result = Phase3WorkerResult.from_dict(stdout_payload)
        except (UnicodeError, ValueError, SchemaValidationError) as error:
            raise Phase3ReportError("worker stdout is not a strict result") from error
        if stdout != canonical_json_bytes(stdout_result) + b"\n":
            raise Phase3ReportError("worker stdout is not canonical JSON")
        if process_audit.get("passed") is False:
            if (
                stdout_result.run_id != worker_result.run_id
                or stdout_result.point_id != worker_result.point_id
                or stdout_result.runner_kind is not worker_result.runner_kind
                or stdout_result.count_unit is not worker_result.count_unit
                or stdout_result.expected_operations
                != worker_result.expected_operations
                or stdout_result.completed_operations
                != worker_result.completed_operations
                or stdout_result.failed_operations
                != worker_result.failed_operations
            ):
                raise Phase3ReportError(
                    "coordinator-aborted worker stdout identity differs"
                )
            if stdout_result.status is RunStatus.ABORTED:
                resolution = _strict_json_object(
                    run_dir / "validation" / "worker_terminal_resolution.json"
                )
                resolved_sha256 = sha256_hex(
                    canonical_json_bytes(worker_result)
                )
                if resolution != {
                    "schema_version": (
                        "kvbench-phase3-worker-terminal-resolution-1.0.0"
                    ),
                    "source_worker_status": "aborted",
                    "source_worker_result_sha256": resolved_sha256,
                    "source_primary_channel_preserved": False,
                    "resolved_terminal_status": "aborted",
                    "resolved_worker_result_sha256": resolved_sha256,
                    "status_overridden": False,
                    "resolution_reason": worker_result.failure_reason,
                }:
                    raise Phase3ReportError(
                        "worker abort resolution is not exactly joined"
                    )
        elif stdout_result != worker_result:
            raise Phase3ReportError("worker stdout differs from the strict worker result")

    if runtime is not None:
        model_file = _strict_json_object(
            run_dir / "validation" / "model_identity.json"
        )
        expected_runtime_model = _runtime_model_identity(manifest)
        _require_equal("loaded model identity", model_file, expected_runtime_model)
        _require_equal(
            "raw loaded model identity",
            worker_evidence.get("model_identity") if worker_evidence else None,
            expected_runtime_model,
        )
    return ValidatedPhase3Run(
        run_dir=run_dir,
        manifest=manifest,
        completion=completion,
        worker_result=worker_result,
        process_audit=process_audit,
        ready_process=ready_process,
        worker_evidence=worker_evidence,
        runtime=runtime,
        numerical=numerical,
        timing=timing,
        telemetry=telemetry,
    )


def expected_campaign_run_ids(
    fixed_campaign_id: str,
    growing_campaign_id: str,
) -> tuple[str, ...]:
    """Reconstruct the sole admissible 16+4 run selection without discovery."""

    if (
        not _CAMPAIGN_ID.fullmatch(fixed_campaign_id)
        or not _CAMPAIGN_ID.fullmatch(growing_campaign_id)
        or fixed_campaign_id == growing_campaign_id
    ):
        raise Phase3ReportError("two distinct exact Phase 3 campaign IDs are required")
    return tuple(
        f"{fixed_campaign_id}-{point_id}" for point_id in _FIXED_POINT_IDS
    ) + tuple(
        f"{growing_campaign_id}-{point_id}" for point_id in _GROWING_POINT_IDS
    )


def load_phase3_campaign_evidence(
    fixed_campaign_id: str,
    growing_campaign_id: str,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[ValidatedPhase3Run, ...]:
    """Load exactly the preregistered runs selected by two explicit campaigns."""

    root = Path(repository_root).resolve(strict=True)
    records = _validated_campaign_selection(
        fixed_campaign_id,
        growing_campaign_id,
        repository_root=root,
    )
    artifact_root = root / "artifacts" / "phase3"
    expected_ids = expected_campaign_run_ids(fixed_campaign_id, growing_campaign_id)
    runs = tuple(
        _load_validated_run(artifact_root / run_id, run_id)
        for run_id in expected_ids
    )
    _require_equal(
        "ordered 20-point selection",
        tuple(run.point_id for run in runs),
        FROZEN_PHASE3_POINT_IDS,
    )
    git_shas = {run.manifest.git_sha for run in runs}
    if len(git_shas) != 1:
        raise Phase3ReportError("selected campaigns do not share one exact Git SHA")
    identity_tuples = {
        (
            run.manifest.hardware_fingerprint,
            run.manifest.software_fingerprint,
            run.manifest.model_fingerprint,
            run.manifest.backend_fingerprint,
            run.manifest.contract_fingerprint,
            run.manifest.measurement_protocol_fingerprint,
        )
        for run in runs
    }
    if len(identity_tuples) != 1:
        raise Phase3ReportError("selected campaigns mix frozen scientific identities")
    runs_by_id = {run.run_id: run for run in runs}
    for record in records:
        result_runs = record["result"].get("runs")
        if not isinstance(result_runs, list):
            raise Phase3ReportError("campaign result run list is malformed")
        for item in result_runs:
            if not isinstance(item, Mapping):
                raise Phase3ReportError("campaign result run item is malformed")
            run_id = item.get("run_id")
            if not isinstance(run_id, str) or run_id not in runs_by_id:
                raise Phase3ReportError("campaign result references an unknown run")
            run = runs_by_id[run_id]
            if (
                item.get("point_id") != run.point_id
                or item.get("status") != run.manifest.status.value
                or item.get("run_dir")
                != run.run_dir.relative_to(root).as_posix()
            ):
                raise Phase3ReportError("campaign result differs from immutable run evidence")
    return runs


def _validated_campaign_selection(
    fixed_campaign_id: str,
    growing_campaign_id: str,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_run_ids = expected_campaign_run_ids(
        fixed_campaign_id,
        growing_campaign_id,
    )
    root = campaign_root(repository_root)
    declarations = (
        (
            fixed_campaign_id,
            PHASE3_FIXED_PLAN_PATH,
            PHASE3_PLAN_FINGERPRINTS[PHASE3_FIXED_PLAN_PATH],
            _FIXED_POINT_IDS,
            expected_run_ids[: len(_FIXED_POINT_IDS)],
        ),
        (
            growing_campaign_id,
            PHASE3_GROWING_PLAN_PATH,
            PHASE3_PLAN_FINGERPRINTS[PHASE3_GROWING_PLAN_PATH],
            _GROWING_POINT_IDS,
            expected_run_ids[len(_FIXED_POINT_IDS) :],
        ),
    )
    records: list[dict[str, Any]] = []
    shared_git_sha: str | None = None
    for campaign_id, plan_path, plan_fingerprint, point_ids, run_ids in declarations:
        validation = validate_phase3_campaign_directory(root / campaign_id)
        if not validation.get("valid"):
            raise Phase3ReportError("selected campaign record is not checksum-valid")
        preregistration = validation.get("preregistration")
        result = validation.get("result")
        if not isinstance(preregistration, dict) or not isinstance(result, dict):
            raise Phase3ReportError("selected campaign validation lacks payloads")
        git_sha = preregistration.get("git_sha")
        if not isinstance(git_sha, str):
            raise Phase3ReportError("campaign Git SHA is absent")
        if shared_git_sha is None:
            shared_git_sha = git_sha
        if (
            git_sha != shared_git_sha
            or preregistration.get("plan_path") != plan_path
            or preregistration.get("plan_fingerprint") != plan_fingerprint
            or preregistration.get("expected_process_count") != len(point_ids)
            or preregistration.get("point_ids") != list(point_ids)
            or preregistration.get("run_ids") != list(run_ids)
            or result.get("git_sha") != git_sha
            or result.get("expected_process_count") != len(point_ids)
            or result.get("attempted_process_count") != len(point_ids)
            or result.get("unattempted_point_ids") != []
            or result.get("unexpected_campaign_abort") is not False
            or result.get("unexpected_failure") is not None
            or result.get("preregistered_before_execution") is not True
        ):
            raise Phase3ReportError("campaign selection/result differs from frozen plan")
        result_runs = result.get("runs")
        if (
            not isinstance(result_runs, list)
            or [
                item.get("run_id") if isinstance(item, Mapping) else None
                for item in result_runs
            ]
            != list(run_ids)
        ):
            raise Phase3ReportError("campaign result does not preserve all preregistered runs")
        assert_unique_plan_campaign(
            root,
            plan_path=plan_path,
            git_sha=git_sha,
            selected_campaign_id=campaign_id,
        )
        records.append(
            {
                "campaign_id": campaign_id,
                "preregistration": preregistration,
                "result": result,
                "preregistration_sha256": sha256_hex(
                    (root / campaign_id / "preregistered.json").read_bytes()
                ),
                "result_sha256": sha256_hex(
                    (root / campaign_id / "result.json").read_bytes()
                ),
                "completion_sha256": sha256_hex(
                    (root / campaign_id / "COMPLETE").read_bytes()
                ),
            }
        )
    return records[0], records[1]


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: object) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return None


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _audit_zero_allocation(value: object) -> bool:
    audit = _mapping(value)
    if audit is None:
        return False
    return bool(
        audit.get("audit_available") is True
        and audit.get("passed") is True
        and audit.get("allocation_event_count") == 0
        and audit.get("allocation_event_bytes") == 0
        and audit.get("allocated_delta", 1) <= 0
        and audit.get("reserved_delta", 1) <= 0
        and audit.get("failure_reason") is None
    )


def _validated_timing_host_ms(run: ValidatedPhase3Run) -> list[float] | None:
    """Independently validate every raw host/CUDA sample for one process."""

    timing = run.timing
    if timing is None:
        return None
    expected_top_keys = {
        "samples",
        "sample_count",
        "paper_claim_eligible",
        "measurement_scope",
        "quality_status",
        "claim_eligibility",
        "performance_claim_eligible",
        "profiler_instrumented",
    }
    samples = _sequence(timing.get("samples"))
    if (
        set(timing) != expected_top_keys
        or samples is None
        or len(samples) != run.manifest.measured_batches
        or timing.get("sample_count") != run.manifest.measured_batches
        or timing.get("paper_claim_eligible") is not False
        or timing.get("measurement_scope") != "native_host_admission"
        or timing.get("quality_status") != "unvalidated"
        or timing.get("claim_eligibility") != "performance_only"
        or timing.get("performance_claim_eligible") is not False
        or timing.get("profiler_instrumented") is not False
    ):
        return None
    expected_operations = (
        run.manifest.measured_count
        if run.manifest.runner_kind is RunnerKind.FIXED_L
        else run.manifest.output_steps
    )
    values: list[float] = []
    for index, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample)
        expected_sample_keys = {
            "batch_index",
            "host_total_ns",
            "cuda_total_ms",
            "completed_operations",
            "failed_operations",
            "host_ns_per_operation",
            "cuda_ms_per_operation",
        }
        if sample is None or set(sample) != expected_sample_keys:
            return None
        host_total = sample.get("host_total_ns")
        cuda_total = sample.get("cuda_total_ms")
        host_per_operation = sample.get("host_ns_per_operation")
        cuda_per_operation = sample.get("cuda_ms_per_operation")
        if (
            sample.get("batch_index") != index
            or not isinstance(host_total, int)
            or isinstance(host_total, bool)
            or host_total <= 0
            or not _finite_number(cuda_total)
            or float(cuda_total) <= 0.0
            or sample.get("completed_operations") != expected_operations
            or sample.get("failed_operations") != 0
            or not _finite_number(host_per_operation)
            or not _finite_number(cuda_per_operation)
        ):
            return None
        expected_host = host_total / expected_operations
        expected_cuda = float(cuda_total) / expected_operations
        if not math.isclose(
            float(host_per_operation),
            expected_host,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(cuda_per_operation),
            expected_cuda,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return None
        values.append(expected_host / 1_000_000.0)
    expected_worker_operations = (
        run.manifest.measured_count * run.manifest.measured_batches
        if run.manifest.runner_kind is RunnerKind.FIXED_L
        else run.manifest.output_steps
        * run.manifest.measured_count
        * run.manifest.measured_batches
    )
    if (
        run.worker_result.expected_operations != expected_worker_operations
        or run.worker_result.completed_operations != expected_worker_operations
        or run.worker_result.failed_operations != 0
    ):
        return None
    return values


def _backend_flash(value: object) -> bool:
    backend = _mapping(value)
    return bool(
        backend is not None
        and backend.get("backend_name") == "FLASH_ATTENTION"
        and backend.get("enable_gqa") is True
        and backend.get("dtype") == "torch.bfloat16"
        and backend.get("device") == "cuda:0"
    )


def _operator_passes(value: object) -> bool:
    operator = _mapping(value)
    operations = None if operator is None else _sequence(operator.get("operations"))
    return bool(
        operator is not None
        and operator.get("passed") is True
        and operator.get("query_head_sized_kv_temporary") is False
        and operator.get("warnings") == []
        and operations
        and any(
            "scaled_dot_product_flash_attention" in str(operation)
            for operation in operations
        )
        and not any(
            fragment in str(operation)
            for operation in operations
            for fragment in ("repeat_interleave", "repeat", "clone")
        )
        and _backend_flash(operator.get("backend"))
    )


def _runtime_output_checksum(run: ValidatedPhase3Run) -> str | None:
    runtime = run.runtime
    if runtime is None:
        return None
    if run.manifest.runner_kind is RunnerKind.FIXED_L:
        value = runtime.get("output_checksum")
        return value if isinstance(value, str) else None
    steps = _sequence(runtime.get("step_evidence"))
    if steps is None:
        return None
    canonical_steps: list[dict[str, Any]] = []
    for item in steps:
        step = _mapping(item)
        if step is None:
            return None
        canonical_steps.append(
            {
                "step": step.get("step"),
                "historical_active_length": step.get("historical_active_length"),
                "output_checksum": step.get("output_checksum"),
                "output_finite": step.get("output_finite"),
            }
        )
    return sha256_hex(canonical_json_bytes(canonical_steps))


def _sha256_value(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _comparison_passes(
    comparison: Mapping[str, Any] | None,
    *,
    atol: float,
    rtol: float,
) -> bool:
    if comparison is None or set(comparison) != {
        "passed",
        "finite",
        "max_absolute_error",
        "max_relative_error",
        "atol",
        "rtol",
    }:
        return False
    absolute = comparison.get("max_absolute_error")
    relative = comparison.get("max_relative_error")
    return bool(
        comparison.get("passed") is True
        and comparison.get("finite") is True
        and comparison.get("atol") == atol
        and comparison.get("rtol") == rtol
        and _finite_number(absolute)
        and float(absolute) >= 0.0
        and _finite_number(relative)
        and float(relative) >= 0.0
    )


def _numerical_passes(run: ValidatedPhase3Run) -> bool:
    numerical = run.numerical
    if numerical is None:
        return False
    small = _mapping(numerical.get("small_tensor"))
    full = _mapping(numerical.get("full_model"))
    if (
        small is None
        or full is None
        or small.get("passed") is not True
        or small.get("reference") != "explicit_fp32_gqa_attention"
        or small.get("atol") != 0.02
        or small.get("rtol") != 0.02
        or small.get("timing_collected") is not False
        or full.get("passed") is not True
        or full.get("reference_implementation")
        != "transformers_eager_dynamic_cache"
        or full.get("tolerance_atol") != 0.125
        or full.get("tolerance_rtol") != 0.02
        or full.get("reference_cache_type") != "DynamicCache"
        or full.get("reference_implementation_restored") is not True
        or full.get("fixed_repeat_exact") is not True
        or full.get("fixed_historical_cache_unchanged") is not True
        or full.get("timing_collected") is not False
        or full.get("performance_claim_eligible") is not False
    ):
        return False
    records = _sequence(small.get("records"))
    fixed_steps = _sequence(full.get("fixed_steps"))
    growing_steps = _sequence(full.get("growing_steps"))
    if records is None or len(records) != 12 or fixed_steps is None or growing_steps is None:
        return False
    expected_small = tuple(
        (batch, length, mode)
        for batch in (1, 2)
        for length in (7, 17)
        for mode in ("causal_gqa", "decode_gqa", "causal_mha")
    )
    for record, expected in zip(records, expected_small):
        item = _mapping(record)
        if item is None:
            return False
        comparison = _mapping(item.get("comparison"))
        if (
            (item.get("batch_size"), item.get("context_length"), item.get("mode"))
            != expected
            or item.get("boundary_first_finite") is not True
            or item.get("boundary_last_finite") is not True
            or not _comparison_passes(comparison, atol=0.02, rtol=0.02)
        ):
            return False
    if len(fixed_steps) != 3 or len(growing_steps) != 3:
        return False
    for index, raw_step in enumerate(fixed_steps):
        step = _mapping(raw_step)
        if (
            step is None
            or step.get("mode") != "fixed_l"
            or step.get("step") != index
            or step.get("position") != 8
            or not _sha256_value(step.get("reference_checksum"))
            or not _sha256_value(step.get("observed_checksum"))
            or not _comparison_passes(
                _mapping(step.get("comparison")),
                atol=0.125,
                rtol=0.02,
            )
        ):
            return False
    for index, raw_step in enumerate(growing_steps):
        step = _mapping(raw_step)
        if (
            step is None
            or step.get("mode") != "growing_context"
            or step.get("step") != index
            or step.get("position") != 8 + index
            or not _sha256_value(step.get("reference_checksum"))
            or not _sha256_value(step.get("observed_checksum"))
            or not _comparison_passes(
                _mapping(step.get("comparison")),
                atol=0.125,
                rtol=0.02,
            )
        ):
            return False
    if run.manifest.graph_mode is GraphMode.CUDA_GRAPH:
        graph = _mapping(numerical.get("full_model_graph"))
        graph_details = None if graph is None else _mapping(graph.get("graph"))
        return bool(
            graph is not None
            and graph.get("passed") is True
            and graph.get("prefix_length") == 8
            and graph_details is not None
            and graph_details.get("captured") is True
            and graph_details.get("fallback") is False
            and _comparison_passes(
                _mapping(graph.get("eager_replay_comparison")),
                atol=0.02,
                rtol=0.02,
            )
            and graph.get("replay_outputs_exact") is True
            and graph.get("replay_copies_independent") is True
            and graph.get("cache_pointers_stable") is True
            and graph.get("historical_cache_unchanged") is True
            and _sha256_value(graph.get("eager_checksum"))
            and _sha256_value(graph.get("first_replay_checksum"))
            and graph.get("first_replay_checksum")
            == graph.get("second_replay_checksum")
            and _audit_zero_allocation(graph.get("replay_allocation"))
            and graph.get("timing_collected") is False
            and graph.get("performance_claim_eligible") is False
        )
    return numerical.get("full_model_graph") is None


def _gqa_passes(run: ValidatedPhase3Run) -> bool:
    runtime = run.runtime
    if runtime is None:
        return False
    source = _mapping(runtime.get("gqa_source"))
    geometry = _mapping(runtime.get("gqa_cache_geometry"))
    operators = (
        [runtime.get("gqa_operator")]
        if run.manifest.runner_kind is RunnerKind.FIXED_L
        else list(_sequence(runtime.get("gqa_operators")) or ())
    )
    expected_operator_count = (
        1
        if run.manifest.runner_kind is RunnerKind.FIXED_L
        else run.manifest.output_steps
    )
    return bool(
        source is not None
        and source.get("passed") is True
        and source.get("findings") == []
        and geometry is not None
        and geometry.get("uses_kv_head_geometry") is True
        and geometry.get("query_head_storage_detected") is False
        and geometry.get("num_query_heads") == 32
        and geometry.get("num_kv_heads") == 8
        and len(operators) == expected_operator_count
        and all(_operator_passes(item) for item in operators)
        and _operator_passes(runtime.get("mha_control"))
        and _backend_flash(runtime.get("prefill_backend"))
        and _backend_flash(runtime.get("backend"))
    )


def _cache_geometry_passes(run: ValidatedPhase3Run) -> bool:
    runtime = run.runtime
    if runtime is None:
        return False
    accounting = _mapping(runtime.get("cache_accounting"))
    geometry = _mapping(runtime.get("gqa_cache_geometry"))
    identity = run.manifest.cache_identity
    if accounting is None or geometry is None:
        return False
    expected_shape = [32, identity.batch_size, 8, identity.capacity, 128]
    return bool(
        geometry.get("cache_shape") == expected_shape
        and accounting.get("predicted_tensor_bytes") == identity.tensor_storage_bytes
        and accounting.get("measured_tensor_bytes") == identity.tensor_storage_bytes
        and accounting.get("padding_bytes") == identity.padding_bytes == 0
        and accounting.get("workspace_bytes") == identity.workspace_bytes
        and accounting.get("allocated_bytes")
        == identity.tensor_storage_bytes + identity.workspace_bytes
        and runtime.get("cache_layout_fingerprint") == identity.layout_fingerprint
    )


def _fixed_runner_passes(run: ValidatedPhase3Run) -> bool:
    runtime = run.runtime
    return bool(
        runtime is not None
        and run.manifest.status is RunStatus.COMPLETED
        and runtime.get("runner") == "fixed_l"
        and runtime.get("context_convention") == "historical_prefix_length"
        and runtime.get("context_length") == run.manifest.context_length
        and runtime.get("total_attended_length") == run.manifest.context_length + 1
        and runtime.get("historical_cache_unchanged") is True
        and runtime.get("historical_checksum_before")
        == runtime.get("historical_checksum_after")
        and runtime.get("cache_pointers_stable") is True
        and runtime.get("output_finite") is True
        and _validated_timing_host_ms(run) is not None
    )


def _growing_runner_passes(run: ValidatedPhase3Run) -> bool:
    runtime = run.runtime
    if runtime is None:
        return False
    expected_lengths = list(
        range(
            run.manifest.context_length,
            run.manifest.context_length + run.manifest.output_steps,
        )
    )
    steps = _sequence(runtime.get("step_evidence"))
    return bool(
        run.manifest.status is RunStatus.COMPLETED
        and runtime.get("runner") == "growing_context"
        and runtime.get("context_convention")
        == "historical_active_length_before_append"
        and runtime.get("active_lengths") == expected_lengths
        and steps is not None
        and len(steps) == run.manifest.output_steps == 16
        and all(
            _mapping(item) is not None
            and _mapping(item).get("step") == index
            and _mapping(item).get("historical_active_length")
            == expected_lengths[index]
            and _mapping(item).get("output_finite") is True
            for index, item in enumerate(steps)
        )
        and runtime.get("cache_pointers_stable") is True
        and runtime.get("output_finite") is True
        and _validated_timing_host_ms(run) is not None
    )


def _graph_passes(run: ValidatedPhase3Run) -> bool:
    runtime = run.runtime
    if runtime is None:
        return False
    graph = _mapping(runtime.get("graph"))
    control = None if run.numerical is None else _mapping(
        run.numerical.get("full_model_graph")
    )
    return bool(
        run.manifest.status is RunStatus.COMPLETED
        and graph is not None
        and graph.get("captured") is True
        and graph.get("fallback") is False
        and graph.get("consecutive_replay_outputs_exact") is True
        and graph.get("first_replay_checksum") == graph.get("second_replay_checksum")
        and runtime.get("cache_pointers_stable") is True
        and control is not None
        and control.get("passed") is True
    )


def _graph_agreement_passes(run: ValidatedPhase3Run) -> bool:
    runtime = run.runtime
    if runtime is None or run.numerical is None:
        return False
    actual = _mapping(runtime.get("eager_graph_comparison"))
    control = _mapping(run.numerical.get("full_model_graph"))
    reference = None if control is None else _mapping(
        control.get("eager_replay_comparison")
    )
    return bool(
        _comparison_passes(actual, atol=0.02, rtol=0.02)
        and _comparison_passes(reference, atol=0.02, rtol=0.02)
    )


def _graph_allocation_passes(run: ValidatedPhase3Run) -> bool:
    runtime = run.runtime
    if runtime is None or run.numerical is None:
        return False
    control = _mapping(run.numerical.get("full_model_graph"))
    return bool(
        _audit_zero_allocation(runtime.get("allocation"))
        and control is not None
        and _audit_zero_allocation(control.get("replay_allocation"))
    )


def _timing_process_median_ms(run: ValidatedPhase3Run) -> tuple[float, list[float]]:
    values = _validated_timing_host_ms(run)
    if values is None:
        raise Phase3ReportError(f"stability timing is absent: {run.point_id}")
    return float(statistics.median(values)), values


def _stability_summary(
    graph_mode: GraphMode,
    runs_by_point: Mapping[str, ValidatedPhase3Run],
) -> tuple[Phase3StabilitySummary, dict[str, Any]] | None:
    point_ids = tuple(
        point_id
        for point_id in FROZEN_PHASE3_STABILITY_POINT_IDS
        if f"-{graph_mode.value}-" in point_id
    )
    selected = tuple(runs_by_point[point_id] for point_id in point_ids)
    if any(run.timing is None for run in selected):
        return None
    process_records: list[dict[str, Any]] = []
    process_medians: list[float] = []
    temperatures: list[float] = []
    clocks: list[int] = []
    powers: list[float] = []
    for run in selected:
        median_ms, samples_ms = _timing_process_median_ms(run)
        process_medians.append(median_ms)
        if run.telemetry is None:
            raise Phase3ReportError("stability telemetry is absent")
        snapshots = []
        for key in ("before", "after"):
            snapshot = _mapping(run.telemetry.get(key))
            if snapshot is None:
                raise Phase3ReportError("stability telemetry snapshot is malformed")
            temperature = snapshot.get("temperature_celsius")
            clock = snapshot.get("sm_clock_mhz")
            power = snapshot.get("power_watts")
            if not all(_finite_number(value) for value in (temperature, clock, power)):
                raise Phase3ReportError("stability telemetry is non-finite")
            if float(clock) != int(float(clock)):
                raise Phase3ReportError("SM clock is not integral")
            temperatures.append(float(temperature))
            clocks.append(int(float(clock)))
            powers.append(float(power))
            snapshots.append(snapshot)
        process_records.append(
            {
                "run_id": run.run_id,
                "point_id": run.point_id,
                "pid": run.ready_process.get("pid"),
                "process_start_time_ticks": run.ready_process.get(
                    "process_start_time_ticks"
                ),
                "host_wall_ms_per_operation_samples": samples_ms,
                "process_median_host_wall_ms": median_ms,
                "telemetry": snapshots,
            }
        )
    identities = {
        (item["pid"], item["process_start_time_ticks"])
        for item in process_records
    }
    if len(identities) != 3 or any(
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or not isinstance(ticks, int)
        or isinstance(ticks, bool)
        for pid, ticks in identities
    ):
        raise Phase3ReportError("stability processes are not independently identified")
    median = float(statistics.median(process_medians))
    minimum = min(process_medians)
    maximum = max(process_medians)
    cv = statistics.stdev(process_medians) / statistics.mean(process_medians) * 100.0
    payload = {
        "schema_version": "kvbench-phase3-stability-source-1.0.0",
        "graph_mode": graph_mode.value,
        "point_ids": list(point_ids),
        "evidence_run_ids": [run.run_id for run in selected],
        "primary_endpoint": "host_observed_wall_time_per_decode_step",
        "profiler_instrumented": False,
        "processes": process_records,
        "derived": {
            "process_replicates": 3,
            "process_median_host_wall_ms": process_medians,
            "median_host_wall_ms": median,
            "minimum_host_wall_ms": minimum,
            "maximum_host_wall_ms": maximum,
            "coefficient_of_variation_percent": cv,
            "temperature_min_c": min(temperatures),
            "temperature_max_c": max(temperatures),
            "sm_clock_min_mhz": min(clocks),
            "sm_clock_max_mhz": max(clocks),
            "power_min_w": min(powers),
            "power_max_w": max(powers),
            "criterion_percent": 3.0,
            "passed": cv <= 3.0,
        },
        "performance_claim_eligible": False,
        "measurement_scope": "native_host_admission",
    }
    path = f"stability/{graph_mode.value}.json"
    digest = sha256_hex(canonical_json_bytes(payload) + b"\n")
    summary = Phase3StabilitySummary(
        graph_mode=graph_mode,
        point_ids=point_ids,
        evidence_run_ids=tuple(run.run_id for run in selected),
        process_replicates=3,
        process_median_host_wall_ms=tuple(process_medians),
        median_host_wall_ms=median,
        minimum_host_wall_ms=minimum,
        maximum_host_wall_ms=maximum,
        coefficient_of_variation_percent=cv,
        temperature_min_c=min(temperatures),
        temperature_max_c=max(temperatures),
        sm_clock_min_mhz=min(clocks),
        sm_clock_max_mhz=max(clocks),
        power_min_w=min(powers),
        power_max_w=max(powers),
        summary_artifact_path=path,
        summary_artifact_sha256=digest,
    )
    return summary, payload


def _git_source_audit(
    repository_root: Path,
    git_sha: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    findings: list[str] = []
    for relative in _SUT_SOURCE_PATHS:
        result = subprocess.run(
            ["git", "show", f"{git_sha}:{relative}"],
            cwd=repository_root,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise Phase3ReportError(f"cannot read recorded Git blob: {relative}")
        try:
            source = result.stdout.decode("utf-8")
        except UnicodeError as error:
            raise Phase3ReportError(f"recorded Git blob is not UTF-8: {relative}") from error
        matched = [pattern for pattern in _FORBIDDEN_HOT_PATH_PATTERNS if pattern in source]
        findings.extend(f"{relative}:{pattern}" for pattern in matched)
        records.append(
            {
                "path": relative,
                "sha256": sha256_hex(result.stdout),
                "forbidden_patterns": matched,
            }
        )
    return {
        "schema_version": "kvbench-phase3-git-source-audit-1.0.0",
        "git_sha": git_sha,
        "paths": records,
        "findings": findings,
        "passed": not findings,
    }


def _git_command(
    repository: Path,
    argv: Sequence[str],
    *,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", *argv],
        cwd=repository,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )


def _git_changed_paths(repository: Path, base: str, target: str) -> tuple[str, ...]:
    result = _git_command(
        repository,
        ["diff", "--name-only", "-z", f"{base}..{target}"],
    )
    if result.returncode != 0 or not isinstance(result.stdout, bytes):
        raise Phase3ReportError("cannot resolve report-generator Git diff")
    try:
        paths = tuple(
            item.decode("utf-8")
            for item in result.stdout.split(b"\0")
            if item
        )
    except UnicodeError as error:
        raise Phase3ReportError("report-generator Git diff path is not UTF-8") from error
    if len(paths) != len(set(paths)):
        raise Phase3ReportError("report-generator Git diff contains duplicate paths")
    return tuple(sorted(paths))


def _report_generator_provenance(
    repository: Path,
    source_execution_git_sha: str,
    requested_generator_git_sha: str | None,
) -> dict[str, Any]:
    head = _git_command(repository, ["rev-parse", "HEAD"], text=True)
    worktree = _git_command(repository, ["status", "--short"], text=True)
    if (
        head.returncode != 0
        or not isinstance(head.stdout, str)
        or worktree.returncode != 0
        or not isinstance(worktree.stdout, str)
        or bool(worktree.stdout.strip())
    ):
        raise Phase3ReportError("report derivation requires a clean Git worktree")
    current_git_sha = head.stdout.strip()
    generator_git_sha = requested_generator_git_sha or current_git_sha
    if not re.fullmatch(r"[0-9a-f]{40}", generator_git_sha):
        raise Phase3ReportError("report-generator Git SHA is invalid")
    source_ancestor = _git_command(
        repository,
        ["merge-base", "--is-ancestor", source_execution_git_sha, generator_git_sha],
    )
    if source_ancestor.returncode != 0:
        raise Phase3ReportError(
            "execution Git SHA is not an ancestor of the report generator"
        )
    changed_paths = _git_changed_paths(
        repository,
        source_execution_git_sha,
        generator_git_sha,
    )
    if any(path not in _REPORT_GENERATOR_CHANGE_PATHS for path in changed_paths):
        raise Phase3ReportError(
            "execution-to-report Git diff is not restricted to reporting code"
        )
    if requested_generator_git_sha is not None:
        generator_ancestor = _git_command(
            repository,
            ["merge-base", "--is-ancestor", generator_git_sha, current_git_sha],
        )
        if generator_ancestor.returncode != 0:
            raise Phase3ReportError(
                "recorded report generator is not an ancestor of current HEAD"
            )
        post_report_paths = _git_changed_paths(
            repository,
            generator_git_sha,
            current_git_sha,
        )
        if any(path not in _POST_REPORT_CHANGE_PATHS for path in post_report_paths):
            raise Phase3ReportError(
                "code changed after the recorded report generator"
            )
    return {
        "schema_version": "kvbench-phase3-report-git-provenance-1.0.0",
        "source_execution_git_sha": source_execution_git_sha,
        "report_generator_git_sha": generator_git_sha,
        "execution_to_generator_changed_paths": list(changed_paths),
        "source_execution_is_ancestor": True,
        "reporting_only_descendant": True,
    }


def _recorded_report_generator_provenance(
    repository: Path,
    source_execution_git_sha: str,
    recorded: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate immutable v1 provenance without comparing current implementation blobs."""

    expected_keys = {
        "schema_version",
        "source_execution_git_sha",
        "report_generator_git_sha",
        "execution_to_generator_changed_paths",
        "source_execution_is_ancestor",
        "reporting_only_descendant",
    }
    generator_git_sha = recorded.get("report_generator_git_sha")
    changed = _sequence(recorded.get("execution_to_generator_changed_paths"))
    if (
        set(recorded) != expected_keys
        or recorded.get("schema_version")
        != "kvbench-phase3-report-git-provenance-1.0.0"
        or recorded.get("source_execution_git_sha") != source_execution_git_sha
        or not isinstance(generator_git_sha, str)
        or re.fullmatch(r"[0-9a-f]{40}", generator_git_sha) is None
        or changed is None
        or any(not isinstance(path, str) or not path for path in changed)
        or tuple(changed) != tuple(sorted(set(changed)))
        or any(path not in _REPORT_GENERATOR_CHANGE_PATHS for path in changed)
        or recorded.get("source_execution_is_ancestor") is not True
        or recorded.get("reporting_only_descendant") is not True
    ):
        raise Phase3ReportError("recorded report-generator provenance is malformed")
    source_ancestor = _git_command(
        repository,
        ["merge-base", "--is-ancestor", source_execution_git_sha, generator_git_sha],
    )
    if source_ancestor.returncode != 0:
        raise Phase3ReportError(
            "recorded report generator does not descend from execution"
        )
    observed_changed = _git_changed_paths(
        repository,
        source_execution_git_sha,
        generator_git_sha,
    )
    if observed_changed != tuple(changed):
        raise Phase3ReportError(
            "recorded execution-to-generator Git diff differs"
        )
    return dict(recorded)


def _git_blob_bytes(
    repository: Path,
    git_sha: str,
    relative_path: str,
) -> bytes:
    result = _git_command(
        repository,
        ["cat-file", "blob", f"{git_sha}:{relative_path}"],
    )
    if (
        result.returncode != 0
        or not isinstance(result.stdout, bytes)
        or not result.stdout
    ):
        raise Phase3ReportError(
            f"report source blob is unavailable: {relative_path}"
        )
    return result.stdout


def _report_generator_requires_raw_audit_replay(
    repository: Path,
    generator_git_sha: str,
) -> bool:
    generator_source = _git_blob_bytes(
        repository,
        generator_git_sha,
        "src/kvbench/runtime/phase3_report.py",
    )
    return (
        _PHASE3_REPORT_RAW_AUDIT_REPLAY_CONTRACT.encode("ascii")
        in generator_source
    )


def _report_execution_source_pin(
    repository: Path,
    execution_git_sha: str,
) -> Any:
    from kvbench.runtime.gqa_device_dispatch import (
        phase3_source_identity_sha256,
    )
    from kvbench.runtime.phase3_coordinator import (
        PHASE3_EXECUTION_SOURCE_PATHS,
        Phase3ExecutionSourcePin,
        _phase3_execution_source_identity_sha256,
    )

    pinned = tuple(
        (
            relative_path,
            _git_blob_bytes(repository, execution_git_sha, relative_path),
        )
        for relative_path in PHASE3_EXECUTION_SOURCE_PATHS
    )
    source_digests = {
        relative_path: sha256_hex(payload)
        for relative_path, payload in pinned
    }
    sut_digests = {
        relative_path: source_digests[relative_path]
        for relative_path in (
            "src/kvbench/runtime/backend.py",
            "src/kvbench/runtime/bf16_endpoint.py",
            "src/kvbench/runtime/static_cache.py",
        )
    }
    return Phase3ExecutionSourcePin(
        execution_git_sha=execution_git_sha,
        source_bytes_by_path=pinned,
        source_identity_sha256=phase3_source_identity_sha256(sut_digests),
        execution_source_identity_sha256=(
            _phase3_execution_source_identity_sha256(source_digests)
        ),
    )


def _validate_report_execution_source_pin(
    run: ValidatedPhase3Run,
    source_pin: Any,
) -> None:
    before = _strict_json_object(
        run.run_dir / "validation" / "execution_source_pin.before_spawn.json"
    )
    after = _strict_json_object(
        run.run_dir
        / "validation"
        / "execution_source_pin.after_worker_exit.json"
    )
    if (
        before != source_pin.to_dict(verification_stage="before_spawn")
        or after
        != source_pin.to_dict(verification_stage="after_worker_exit")
    ):
        raise Phase3ReportError(
            f"execution source pin differs for run: {run.run_id}"
        )


def _raw_audit_index_bytes(run: ValidatedPhase3Run) -> bytes:
    index_path = run.run_dir / "raw" / "audits" / "index.json"
    try:
        index_bytes = index_path.read_bytes()
    except OSError as error:
        raise Phase3ReportError(
            f"consolidated raw-audit index is absent: {run.run_id}"
        ) from error
    sidecar = _strict_json_object(
        run.run_dir
        / "raw"
        / "transport"
        / "raw_audit_index_sidecar.v2.jsonl",
        canonical=True,
    )
    if (
        set(sidecar)
        != {
            "schema_version",
            "raw_audit_run_index",
            "raw_audit_run_index_sha256",
        }
        or sidecar.get("schema_version")
        != "kvbench-phase3-worker-evidence-2.0.0"
        or not isinstance(sidecar.get("raw_audit_run_index"), Mapping)
        or sidecar.get("raw_audit_run_index_sha256")
        != sha256_hex(index_bytes)
        or canonical_json_bytes(sidecar["raw_audit_run_index"])
        != index_bytes
    ):
        raise Phase3ReportError(
            f"raw-audit sidecar/index binding differs: {run.run_id}"
        )
    return index_bytes


def _retained_raw_audit_bytes(run: ValidatedPhase3Run, index: Any) -> dict[str, bytes]:
    retained: dict[str, bytes] = {}
    root = run.run_dir / "raw" / "audits" / "files"
    for record in index.records:
        for declaration in record.files:
            path = root / declaration.path
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise Phase3ReportError(
                    f"declared raw-audit file is absent: {run.run_id}: "
                    f"{declaration.path}"
                ) from error
            if (
                len(payload) != declaration.size_bytes
                or sha256_hex(payload) != declaration.sha256
            ):
                raise Phase3ReportError(
                    f"declared raw-audit file differs: {run.run_id}: "
                    f"{declaration.path}"
                )
            retained[declaration.path] = payload
    return retained


def _raw_audit_replay_facts(index: Any, outcome: Mapping[str, Any]) -> dict[str, bool]:
    semantic_operations = _sequence(outcome.get("semantic_operations"))
    semantic_passed = bool(
        outcome.get("semantic_validation_passed") is True
        and outcome.get("scientific_completion_passed") is True
        and semantic_operations is not None
        and len(semantic_operations) == len(index.records)
    )
    failed = {
        "semantic_validation_passed": False,
        "gqa_nonmaterialization_verified": False,
        "allocation_criterion_passed": False,
        "graph_zero_allocation_passed": False,
        "flash_backend_forced": False,
    }
    if not semantic_passed or semantic_operations is None:
        return failed
    gqa_verified = semantic_passed
    allocation_verified = semantic_passed
    graph_zero_allocation = semantic_passed
    flash_backend_forced = semantic_passed
    for record, raw_operation in zip(
        index.records,
        semantic_operations,
        strict=True,
    ):
        operation = _mapping(raw_operation)
        if operation is None:
            raise Phase3ReportError("raw-audit replay operation is malformed")
        class_counts = _mapping(operation.get("allocation_class_counts"))
        failure_reasons = _sequence(
            operation.get("allocation_failure_reasons")
        )
        kernel_families = _mapping(
            operation.get("device_kernel_families")
        )
        event_count = operation.get("allocation_event_count")
        if (
            operation.get("operation_fingerprint_sha256")
            != record.operation.operation_fingerprint_sha256
            or class_counts is None
            or failure_reasons is None
            or kernel_families is None
            or type(event_count) is not int
            or event_count < 0
            or any(
                type(name) is not str
                or type(count) is not int
                or count < 0
                for name, count in class_counts.items()
            )
            or sum(class_counts.values()) != event_count
        ):
            raise Phase3ReportError(
                "raw-audit replay operation differs from its operation key"
            )
        operation_gqa_verified = bool(
            operation.get("gqa_verdict") == _PHASE3_GQA_VERIFIED
            and operation.get("gqa_reasons") == []
        )
        gqa_verified = gqa_verified and operation_gqa_verified
        gqa_family = kernel_families.get("gqa")
        mha_family = kernel_families.get("mha_control")
        flash_backend_forced = bool(
            flash_backend_forced
            and operation_gqa_verified
            and isinstance(gqa_family, str)
            and gqa_family.startswith("pytorch_flash::flash_fwd")
            and isinstance(mha_family, str)
            and mha_family.startswith("pytorch_flash::flash_fwd")
        )

        is_graph = record.operation.graph_mode.value == "cuda_graph"
        expected_criterion = (
            _PHASE3_GRAPH_ALLOCATION_CRITERION
            if is_graph
            else _PHASE3_EAGER_ALLOCATION_CRITERION
        )
        operation_allocation_verified = bool(
            operation.get("allocation_criterion_id") == expected_criterion
            and failure_reasons == []
            and (
                class_counts == {} and event_count == 0
                if is_graph
                else set(class_counts).issubset(
                    _PHASE3_EAGER_ALLOCATION_CLASSES
                )
            )
        )
        allocation_verified = (
            allocation_verified and operation_allocation_verified
        )
        if is_graph:
            graph_zero_allocation = bool(
                graph_zero_allocation and operation_allocation_verified
            )
    return {
        "semantic_validation_passed": semantic_passed,
        "gqa_nonmaterialization_verified": gqa_verified,
        "allocation_criterion_passed": allocation_verified,
        "graph_zero_allocation_passed": graph_zero_allocation,
        "flash_backend_forced": flash_backend_forced,
    }


def _replay_report_raw_audit_run(
    run: ValidatedPhase3Run,
    source_pin: Any,
) -> dict[str, bool]:
    from kvbench.runtime.phase3_coordinator import (
        Phase3CoordinatorError,
        _expected_phase3_raw_audit_operations,
        _replay_phase3_raw_audit_semantics,
    )
    from kvbench.runtime.phase3_raw_audit_evidence import (
        Phase3RawAuditEvidenceError,
        parse_phase3_raw_audit_run_index_bytes,
    )
    from kvbench.schema.phase3 import Phase3ProcessPoint

    _validate_report_execution_source_pin(run, source_pin)
    try:
        index = parse_phase3_raw_audit_run_index_bytes(
            _raw_audit_index_bytes(run)
        )
        point = Phase3ProcessPoint(
            point_id=run.point_id,
            runner_kind=run.manifest.runner_kind,
            graph_mode=run.manifest.graph_mode,
            batch_size=run.manifest.batch_size,
            context_length=run.manifest.context_length,
            output_steps=run.manifest.output_steps,
            process_replicate=run.manifest.process_replicate,
            stability_member=(
                run.point_id in FROZEN_PHASE3_STABILITY_POINT_IDS
            ),
        )
        expected_operations = _expected_phase3_raw_audit_operations(
            point=point,
            run_id=run.run_id,
            git_sha=run.manifest.git_sha,
            cache=run.manifest.cache_identity,
            backend=run.manifest.backend_identity,
            source_sha256_by_path=source_pin.sut_source_sha256_by_path,
        )
        if (
            index.run_id != run.run_id
            or index.point_id != run.point_id
            or tuple(record.operation for record in index.records)
            != expected_operations
        ):
            raise Phase3ReportError(
                f"raw-audit operation join differs: {run.run_id}"
            )
        retained = _retained_raw_audit_bytes(run, index)
        outcome: dict[str, Any] = {
            "process_audit_passed": run.process_audit.get("passed") is True,
            "commitment_validation_passed": True,
            "execution_source_revalidated_after_worker_exit": True,
        }
        _replay_phase3_raw_audit_semantics(
            index=index,
            retained=retained,
            execution_source_pin=source_pin,
            backend_identity=run.manifest.backend_identity,
            outcome=outcome,
        )
        return _raw_audit_replay_facts(index, outcome)
    except Phase3ReportError:
        raise
    except (Phase3CoordinatorError, Phase3RawAuditEvidenceError) as error:
        raise Phase3ReportError(
            f"independent raw-audit replay failed: {run.run_id}"
        ) from error


def _replay_report_raw_audits(
    runs: tuple[ValidatedPhase3Run, ...],
    repository: Path,
) -> dict[str, dict[str, bool]]:
    execution_git_sha = runs[0].manifest.git_sha
    source_pin = _report_execution_source_pin(
        repository,
        execution_git_sha,
    )
    return {
        run.run_id: _replay_report_raw_audit_run(run, source_pin)
        for run in runs
    }


def _criterion(
    name: str,
    runs_by_point: Mapping[str, ValidatedPhase3Run],
    passed: bool,
    reason: str,
    *,
    disposition: GateDisposition = GateDisposition.FAIL,
) -> Phase3G1Criterion:
    evidence = tuple(
        runs_by_point[point_id].run_id
        for point_id in g1_expected_point_ids(name)
    )
    return Phase3G1Criterion(
        criterion=name,
        disposition=GateDisposition.PASS if passed else disposition,
        evidence_run_ids=evidence,
        reason=None if passed else reason,
    )


def _derive_criteria(
    runs: tuple[ValidatedPhase3Run, ...],
    stability: Mapping[GraphMode, Phase3StabilitySummary],
    source_audit: Mapping[str, Any],
    repository_root: Path,
    raw_audit_replays: Mapping[str, Mapping[str, bool]] | None = None,
) -> tuple[Phase3G1Criterion, ...]:
    by_point = {run.point_id: run for run in runs}
    fixed = tuple(run for run in runs if run.manifest.runner_kind is RunnerKind.FIXED_L)
    growing = tuple(
        run for run in runs if run.manifest.runner_kind is RunnerKind.GROWING_CONTEXT
    )
    eager = tuple(run for run in runs if run.manifest.graph_mode is GraphMode.EAGER)
    graph = tuple(
        run for run in runs if run.manifest.graph_mode is GraphMode.CUDA_GRAPH
    )
    model_payload = runs[0].manifest.model_identity.to_dict()
    backend_payload = runs[0].manifest.backend_identity.to_dict()
    exact_model = all(
        run.manifest.model_identity.to_dict() == model_payload
        and run.worker_evidence is not None
        and run.worker_evidence.get("model_identity")
        == _runtime_model_identity(run.manifest)
        for run in runs
    )
    exact_backend = all(
        run.manifest.backend_identity.to_dict() == backend_payload for run in runs
    )
    if raw_audit_replays is not None:
        if set(raw_audit_replays) != {run.run_id for run in runs}:
            raise Phase3ReportError(
                "raw-audit replay set differs from selected runs"
            )
        raw_gqa = all(
            raw_audit_replays[run.run_id].get(
                "gqa_nonmaterialization_verified"
            )
            is True
            for run in runs
        )
        allocation = all(
            raw_audit_replays[run.run_id].get(
                "allocation_criterion_passed"
            )
            is True
            for run in runs
        )
        graph_allocation = all(
            raw_audit_replays[run.run_id].get(
                "graph_zero_allocation_passed"
            )
            is True
            for run in graph
        )
        raw_flash_backend = all(
            raw_audit_replays[run.run_id].get("flash_backend_forced")
            is True
            for run in runs
        )
        source_non_growth = bool(source_audit.get("passed")) and raw_gqa
    else:
        raw_gqa = all(_gqa_passes(run) for run in runs)
        allocation = all(
            run.runtime is not None
            and _audit_zero_allocation(run.runtime.get("allocation"))
            and _mapping(run.runtime.get("memory_evidence")) is not None
            and _mapping(run.runtime.get("memory_evidence")).get(
                "timing_executed"
            )
            is True
            and _mapping(run.runtime.get("memory_evidence")).get(
                "timing_allocated_delta_bytes", 1
            )
            <= 0
            and _mapping(run.runtime.get("memory_evidence")).get(
                "timing_reserved_delta_bytes", 1
            )
            <= 0
            for run in runs
        )
        graph_allocation = all(_graph_allocation_passes(run) for run in graph)
        raw_flash_backend = False
        source_non_growth = bool(source_audit.get("passed")) and all(
            _mapping(run.runtime.get("gqa_source")) is not None
            and _mapping(run.runtime.get("gqa_source")).get("passed") is True
            for run in runs
            if run.runtime is not None
        ) and all(run.runtime is not None for run in runs)
    checksums = all(
        run.worker_result.output_checksum is not None
        and run.worker_result.output_checksum == _runtime_output_checksum(run)
        for run in runs
    )
    for mode in (GraphMode.EAGER, GraphMode.CUDA_GRAPH):
        replicate_checksums = {
            by_point[point_id].worker_result.output_checksum
            for point_id in FROZEN_PHASE3_STABILITY_POINT_IDS
            if f"-{mode.value}-" in point_id
        }
        checksums = checksums and len(replicate_checksums) == 1
    independent = True
    for mode in (GraphMode.EAGER, GraphMode.CUDA_GRAPH):
        selected = [
            by_point[point_id]
            for point_id in FROZEN_PHASE3_STABILITY_POINT_IDS
            if f"-{mode.value}-" in point_id
        ]
        identities = {
            (
                run.ready_process.get("pid"),
                run.ready_process.get("process_start_time_ticks"),
            )
            for run in selected
        }
        independent = independent and len(selected) == 3 and len(identities) == 3
    stability_pass = (
        set(stability) == {GraphMode.EAGER, GraphMode.CUDA_GRAPH}
        and all(
            summary.coefficient_of_variation_percent <= 3.0
            for summary in stability.values()
        )
    )
    process_audits = all(
        run.process_audit.get("passed") is True
        and run.process_audit.get("foreign_compute_allowed") is False
        and run.process_audit.get("unknown_compute_allowed") is False
        for run in runs
    )
    if raw_audit_replays is not None:
        no_fallback = process_audits and raw_flash_backend and all(
            run.manifest.status is not RunStatus.BACKEND_FALLBACK
            for run in runs
        )
    else:
        no_fallback = process_audits and all(
            run.manifest.status is not RunStatus.BACKEND_FALLBACK
            and run.runtime is not None
            and _backend_flash(run.runtime.get("prefill_backend"))
            and _backend_flash(run.runtime.get("backend"))
            and (
                run.manifest.graph_mode is GraphMode.EAGER
                or _mapping(run.runtime.get("graph")) is not None
                and _mapping(run.runtime.get("graph")).get("fallback") is False
            )
            for run in runs
        )
    no_claim = all(
        run.manifest.performance_claim_eligible is False
        and run.manifest.measurement_scope is MeasurementScope.NATIVE_HOST_ADMISSION
        and run.manifest.quality.quality_status
        is QualityValidationState.UNVALIDATED
        and run.manifest.quality.claim_eligibility
        is ClaimEligibility.PERFORMANCE_ONLY
        and run.manifest.quality.quality_execution
        is QualityExecutionState.LOCKED
        and not run.manifest.quality.performance_data_frozen
        and (
            run.timing is None
            or run.timing.get("performance_claim_eligible") is False
            and run.timing.get("profiler_instrumented") is False
        )
        for run in runs
    ) and not any(
        (repository_root / relative).exists()
        for relative in ("paper-results", "paper_results", "artifacts/quality", "artifacts/profiler")
    )
    all_immutable = all(
        validate_run_directory(run.run_dir).valid
        and validate_run_directory(run.run_dir).complete
        for run in runs
    )

    results = {
        "exact_model_and_tokenizer_identity": (exact_model, "model/tokenizer identity join failed"),
        "exact_bf16_backend_identity": (exact_backend, "backend identity join failed"),
        "numerical_reference_match": (all(_numerical_passes(run) for run in runs), "one or more numerical controls failed or are absent"),
        "no_torch_cat_growth": (
            source_non_growth,
            "raw dispatch/source proof found forbidden or unverified growth"
            if raw_audit_replays is not None
            else "recorded SUT source audit found forbidden growth",
        ),
        "no_unexplained_measured_region_allocation": (
            allocation,
            "one or more raw exact-operation allocation replays failed the frozen criterion"
            if raw_audit_replays is not None
            else "one or more exact decode operations issued allocator events or lacked normal timing",
        ),
        "kv_head_cache_geometry": (all(_cache_geometry_passes(run) for run in runs), "cache geometry or byte accounting failed"),
        "gqa_not_materialized": (
            raw_gqa,
            "raw GQA dispatch/allocation replay did not verify non-materialization"
            if raw_audit_replays is not None
            else "GQA source/operator/storage audit failed",
        ),
        "fixed_l_runner": (all(_fixed_runner_passes(run) for run in fixed), "one or more fixed-L processes did not complete its declared timing lane"),
        "growing_context_runner": (all(_growing_runner_passes(run) for run in growing), "one or more growing-context processes did not complete its declared trajectory"),
        "eager_lane": (all(run.manifest.status is RunStatus.COMPLETED and _validated_timing_host_ms(run) is not None for run in eager), "one or more eager processes did not complete valid timing"),
        "cuda_graph_capture_and_replay": (all(_graph_passes(run) for run in graph), "one or more CUDA Graph capture/replay controls failed"),
        "eager_graph_numerical_agreement": (all(_graph_agreement_passes(run) for run in graph), "one or more eager/graph comparisons failed"),
        "graph_replay_no_allocation": (
            graph_allocation,
            "one or more raw graph replay allocation controls were not strict zero"
            if raw_audit_replays is not None
            else "one or more graph replay allocation controls failed",
        ),
        "stable_output_checksums": (checksums, "output checksum derivation or replicate stability failed"),
        "independent_process_replicates": (independent, "exact independent process identities were not preserved"),
        "stability_threshold": (stability_pass, "both eager and graph host-wall CV summaries are required at CV <= 3%"),
        "no_backend_fallback": (
            no_fallback,
            "raw forced-Flash dispatch or process audit indicates fallback/ambiguity"
            if raw_audit_replays is not None
            else "backend dispatch or process audit indicates fallback/ambiguity",
        ),
        "no_model_substitution": (exact_model, "selected runs do not share the exact frozen checkpoint/tokenizer"),
        "no_formal_paper_claim": (no_claim, "Phase 3 claim/quality/profiler governance was violated"),
        "immutable_checksum_valid_artifacts": (all_immutable, "one or more selected runs failed immutable checksum validation"),
    }
    criteria: list[Phase3G1Criterion] = []
    for name in G1_CRITERIA:
        passed, reason = results[name]
        disposition = (
            GateDisposition.PARTIAL
            if name == "stability_threshold" and not passed
            else GateDisposition.FAIL
        )
        criteria.append(
            _criterion(
                name,
                by_point,
                passed,
                reason,
                disposition=disposition,
            )
        )
    return tuple(criteria)


def derive_phase3_g1_report(
    runs: tuple[ValidatedPhase3Run, ...],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    generated_at_utc: str | None = None,
    report_generator_git_sha: str | None = None,
    recorded_report_git_provenance: Mapping[str, Any] | None = None,
) -> tuple[Phase3G1AdmissionReport, dict[str, dict[str, Any]], dict[str, Any]]:
    """Derive G1 solely from validated immutable run evidence."""

    if tuple(run.point_id for run in runs) != FROZEN_PHASE3_POINT_IDS:
        raise Phase3ReportError("report input is not the exact ordered 20-point grid")
    if len({run.run_id for run in runs}) != 20:
        raise Phase3ReportError("report input contains duplicate run IDs")
    repository = Path(repository_root).resolve(strict=True)
    git_sha = runs[0].manifest.git_sha
    if any(run.manifest.git_sha != git_sha for run in runs):
        raise Phase3ReportError("report input mixes Git SHAs")
    if recorded_report_git_provenance is None:
        report_git_provenance = _report_generator_provenance(
            repository,
            git_sha,
            report_generator_git_sha,
        )
    else:
        report_git_provenance = _recorded_report_generator_provenance(
            repository,
            git_sha,
            recorded_report_git_provenance,
        )
        if (
            report_generator_git_sha is not None
            and report_git_provenance["report_generator_git_sha"]
            != report_generator_git_sha
        ):
            raise Phase3ReportError(
                "requested and recorded report generators differ"
            )
    raw_audit_replay_required = (
        _report_generator_requires_raw_audit_replay(
            repository,
            report_git_provenance["report_generator_git_sha"],
        )
    )
    if any(
        (repository / relative).exists()
        or (repository / relative).is_symlink()
        for relative in (
            "PERFORMANCE_DATA_FROZEN",
            "artifacts/quality",
            "artifacts/profiler",
            "paper-results",
            "paper_results",
        )
    ):
        raise Phase3ReportError("quality, profiler, paper, or freeze state is not closed")
    by_point = {run.point_id: run for run in runs}
    stability: dict[GraphMode, Phase3StabilitySummary] = {}
    stability_payloads: dict[str, dict[str, Any]] = {}
    for mode in (GraphMode.EAGER, GraphMode.CUDA_GRAPH):
        derived = _stability_summary(mode, by_point)
        if derived is not None:
            summary, payload = derived
            stability[mode] = summary
            stability_payloads[summary.summary_artifact_path] = payload
    source_audit = _git_source_audit(repository, git_sha)
    raw_audit_replays = (
        _replay_report_raw_audits(runs, repository)
        if raw_audit_replay_required
        else None
    )
    criteria = _derive_criteria(
        runs,
        stability,
        source_audit,
        repository,
        raw_audit_replays,
    )
    dispositions = {item.disposition for item in criteria}
    status = (
        GateDisposition.FAIL
        if GateDisposition.FAIL in dispositions
        else GateDisposition.BLOCKED
        if GateDisposition.BLOCKED in dispositions
        else GateDisposition.PARTIAL
        if GateDisposition.PARTIAL in dispositions
        else GateDisposition.PASS
    )
    timestamp = generated_at_utc or datetime.now(timezone.utc).isoformat()
    run_evidence = tuple(
        Phase3RunEvidence(
            run_id=run.run_id,
            point_id=run.point_id,
            point_fingerprint=run.manifest.point_fingerprint,
            plan_path=run.manifest.plan_source.path or "",
            plan_fingerprint=run.manifest.plan_fingerprint,
            status=run.manifest.status,
            manifest_sha256=run.completion.manifest_sha256,
            artifact_inventory_sha256=run.completion.artifact_inventory_sha256,
            checksum_ledger_sha256=run.completion.checksum_ledger_sha256,
            checksum_valid=True,
        )
        for run in runs
    )
    report = Phase3G1AdmissionReport(
        schema_version=Phase3G1AdmissionReport.SCHEMA_VERSION,
        generated_at_utc=timestamp,
        git_sha=git_sha,
        status=status,
        g0=GateDisposition.PASS,
        g1=status,
        g2=GateDisposition.NOT_EVALUATED,
        g3=GateDisposition.NOT_EVALUATED,
        g4=GateDisposition.NOT_EVALUATED,
        g5=GateDisposition.NOT_EVALUATED,
        full_scan_state="closed",
        quality=QualityStatus(
            schema_version=QualityStatus.SCHEMA_VERSION,
            quality_status=QualityValidationState.UNVALIDATED,
            claim_eligibility=ClaimEligibility.PERFORMANCE_ONLY,
            quality_execution=QualityExecutionState.LOCKED,
            performance_data_frozen=False,
        ),
        quality_benchmark_executed=False,
        quality_only_dependencies_installed=False,
        measurement_scope=MeasurementScope.NATIVE_HOST_ADMISSION,
        performance_claim_eligible=False,
        performance_data_frozen=False,
        blocker_b009="OPEN",
        blocker_b010="OPEN",
        expected_process_count=20,
        plan_sources=tuple(
            SourceDigest(
                path=path,
                sha256=digest,
            )
            for path, digest in PHASE3_PLAN_FINGERPRINTS.items()
        ),
        run_evidence=run_evidence,
        stability_summaries=tuple(
            stability[mode]
            for mode in (GraphMode.EAGER, GraphMode.CUDA_GRAPH)
            if mode in stability
        ),
        criteria=criteria,
        all_artifacts_checksum_valid=True,
        formal_paper_claim_generated=False,
    )
    derivation = {
        "schema_version": "kvbench-phase3-g1-derivation-1.0.0",
        "report_git_provenance": report_git_provenance,
        "git_source_audit": source_audit,
        "selected_run_ids": [run.run_id for run in runs],
        "selected_point_ids": [run.point_id for run in runs],
        "criteria": [item.to_dict() for item in criteria],
        "self_asserted_campaign_flags_used": False,
        "glob_or_latest_selection_used": False,
        "selective_rerun_performed": False,
    }
    return report, stability_payloads, derivation


def build_phase3_g1_report(
    fixed_campaign_id: str,
    growing_campaign_id: str,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    generated_at_utc: str | None = None,
    report_generator_git_sha: str | None = None,
    recorded_report_git_provenance: Mapping[str, Any] | None = None,
) -> tuple[Phase3G1AdmissionReport, dict[str, dict[str, Any]], dict[str, Any]]:
    """Load two explicit campaigns and derive their one admissible G1 report."""

    repository = Path(repository_root).resolve(strict=True)
    campaign_records = _validated_campaign_selection(
        fixed_campaign_id,
        growing_campaign_id,
        repository_root=repository,
    )
    runs = load_phase3_campaign_evidence(
        fixed_campaign_id,
        growing_campaign_id,
        repository_root=repository,
    )
    report, stability, derivation = derive_phase3_g1_report(
        runs,
        repository_root=repository,
        generated_at_utc=generated_at_utc,
        report_generator_git_sha=report_generator_git_sha,
        recorded_report_git_provenance=recorded_report_git_provenance,
    )
    derivation["campaign_preregistration"] = [
        {
            "campaign_id": record["campaign_id"],
            "preregistration_sha256": record["preregistration_sha256"],
            "result_sha256": record["result_sha256"],
            "completion_sha256": record["completion_sha256"],
            "unique_exact_plan_git_attempt": True,
        }
        for record in campaign_records
    ]
    derivation["selective_rerun_performed"] = False
    return report, stability, derivation


def _exclusive_write(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_phase3_g1_report(
    fixed_campaign_id: str,
    growing_campaign_id: str,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Publish a new report through the version-2 append-only lifecycle."""

    from kvbench.runtime.phase3_report_publication import (
        publish_phase3_g1_report,
    )

    return publish_phase3_g1_report(
        fixed_campaign_id,
        growing_campaign_id,
        repository_root=repository_root,
    )


def validate_phase3_g1_report_directory(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Independently validate one finalized report bundle and stability hashes."""

    candidate = Path(path).absolute()
    try:
        completion_payload = _strict_json_object(
            candidate / "COMPLETE",
            canonical=True,
        )
    except (OSError, ValueError, Phase3ReportError):
        completion_payload = {}
    if (
        completion_payload.get("schema_version")
        == "kvbench-phase3-g1-completion-2.0.0"
    ):
        from kvbench.runtime.phase3_report_publication import (
            validate_phase3_g1_report_directory_v2,
        )

        return validate_phase3_g1_report_directory_v2(
            candidate,
            repository_root=repository_root,
        )

    errors: list[str] = []
    report_sha = ""
    try:
        lexical = Path(path).absolute()
        lexical_metadata = lexical.lstat()
        if stat.S_ISLNK(lexical_metadata.st_mode) or not stat.S_ISDIR(
            lexical_metadata.st_mode
        ):
            raise Phase3ReportError("report directory is a symlink or non-directory")
        directory = lexical.resolve(strict=True)
        if directory.name == "" or not _REPORT_ID.fullmatch(directory.name):
            errors.append("report directory name is invalid")
        write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        for target in (directory, *sorted(directory.rglob("*"))):
            metadata = target.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or metadata.st_mode & write_bits
                or target.is_file()
                and metadata.st_nlink != 1
            ):
                errors.append("report bundle contains unsafe or writable content")
                break
        report_payload = _strict_json_object(
            directory / "report.json",
            canonical=True,
        )
        report = Phase3G1AdmissionReport.from_dict(report_payload)
        source_payload = _strict_json_object(
            directory / "source_campaigns.json",
            canonical=True,
        )
        fixed_campaign_id = source_payload.get("fixed_campaign_id")
        growing_campaign_id = source_payload.get("growing_campaign_id")
        if (
            source_payload.get("schema_version")
            != "kvbench-phase3-report-sources-1.0.0"
            or not isinstance(fixed_campaign_id, str)
            or not isinstance(growing_campaign_id, str)
            or source_payload.get("explicit_selection") is not True
        ):
            raise Phase3ReportError("report campaign sources are invalid")
        derivation_payload = _strict_json_object(
            directory / "derivation.json",
            canonical=True,
        )
        report_git_provenance = _mapping(
            derivation_payload.get("report_git_provenance")
        )
        recorded_generator_git_sha = (
            None
            if report_git_provenance is None
            else report_git_provenance.get("report_generator_git_sha")
        )
        if not isinstance(recorded_generator_git_sha, str):
            raise Phase3ReportError("report-generator provenance is absent")
        expected_report, expected_stability, expected_derivation = (
            build_phase3_g1_report(
                fixed_campaign_id,
                growing_campaign_id,
                repository_root=directory.parents[2],
                generated_at_utc=report.generated_at_utc,
                report_generator_git_sha=recorded_generator_git_sha,
                recorded_report_git_provenance=report_git_provenance,
            )
        )
        if expected_report.to_dict() != report_payload:
            errors.append("report differs from independently rederived run evidence")
        if derivation_payload != expected_derivation:
            errors.append("derivation artifact differs from source runs")
        expected_files = {
            "COMPLETE",
            "checksums.sha256",
            "derivation.json",
            "report.json",
            "source_campaigns.json",
            *expected_stability.keys(),
        }
        actual_files = {
            target.relative_to(directory).as_posix()
            for target in directory.rglob("*")
            if target.is_file()
        }
        if actual_files != expected_files:
            errors.append("report exact file set differs")
        for relative, expected_payload in expected_stability.items():
            observed_payload = _strict_json_object(
                directory / relative,
                canonical=True,
            )
            if observed_payload != expected_payload:
                errors.append(f"stability derivation differs: {relative}")
        completion = _strict_json_object(
            directory / "COMPLETE",
            canonical=True,
        )
        ledger_path = directory / "checksums.sha256"
        ledger_bytes = ledger_path.read_bytes()
        entries: dict[str, str] = {}
        for line in ledger_bytes.decode("utf-8").splitlines():
            digest, separator, relative = line.partition("  ")
            lexical = PurePosixPath(relative)
            if (
                not separator
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or lexical.is_absolute()
                or ".." in lexical.parts
                or relative in entries
            ):
                errors.append("checksum ledger is malformed")
                continue
            entries[relative] = digest
        actual = {
            target.relative_to(directory).as_posix()
            for target in directory.rglob("*")
            if target.is_file() and target.name not in {"checksums.sha256", "COMPLETE"}
        }
        if set(entries) != actual or list(entries) != sorted(entries):
            errors.append("checksum ledger coverage or ordering differs")
        for relative, digest in entries.items():
            target = directory / relative
            if not target.is_file() or sha256_hex(target.read_bytes()) != digest:
                errors.append(f"checksum mismatch: {relative}")
        report_sha = sha256_hex((directory / "report.json").read_bytes())
        if (
            completion.get("schema_version")
            != "kvbench-phase3-g1-completion-1.0.0"
            or completion.get("report_id") != directory.name
            or completion.get("status") != report.status.value
            or completion.get("report_sha256") != report_sha
            or completion.get("checksum_ledger_sha256") != sha256_hex(ledger_bytes)
            or completion.get("written_last") is not True
        ):
            errors.append("report completion marker differs")
        for summary in report.stability_summaries:
            target = directory / summary.summary_artifact_path
            if (
                not target.is_file()
                or sha256_hex(target.read_bytes())
                != summary.summary_artifact_sha256
            ):
                errors.append("stability source artifact hash differs")
    except (
        OSError,
        UnicodeError,
        ValueError,
        Phase3ReportError,
        SchemaValidationError,
    ) as error:
        errors.append(f"report validation failed closed: {type(error).__name__}")
    return {
        "schema_version": "kvbench-phase3-g1-validation-1.0.0",
        "valid": not errors,
        "report_sha256": report_sha,
        "errors": errors,
    }

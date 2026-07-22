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
    environment = _strict_json_object(
        run_dir / "environment" / "worker_environment.json"
    )
    digest = sha256_hex(canonical_json_bytes(environment))
    _require_equal("worker environment SHA-256", digest, manifest.command.environment_sha256)


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


def _join_setup_and_process_evidence(
    run_dir: Path,
    manifest: Phase3RunManifest,
    process_audit: Mapping[str, Any],
    ready: Mapping[str, Any],
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
    if process_audit != {
        "schema_version": "kvbench-phase3-process-audit-1.0.0",
        "passed": True,
        "certified_helper": "preflight/process_query.py",
        "foreign_compute_allowed": False,
        "unknown_compute_allowed": False,
    }:
        raise Phase3ReportError("process audit outcome is not an exact pass")
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
    if (
        not _process_snapshot_clean(before, allow_supervised=False)
        or not _process_snapshot_clean(
            release,
            allow_supervised=True,
            ready=ready,
            gpu_uuid=manifest.gpu_uuid,
        )
        or not _process_snapshot_clean(after, allow_supervised=False)
        or during.get("schema_version")
        != "kvbench-phase3-process-monitor-1.0.0"
        or during.get("sampling_target_seconds") != 2.0
        or during.get("saw_allowed_compute") is not True
        or during.get("monitoring_stopped_before_worker_exit") is not False
        or samples is None
        or not samples
        or not all(
            _process_snapshot_clean(
                sample,
                allow_supervised=True,
                ready=ready,
                gpu_uuid=manifest.gpu_uuid,
            )
            for sample in samples
        )
        or not any(
            bool(_mapping(sample).get("allowed_compute_processes"))
            for sample in samples
            if _mapping(sample) is not None
        )
    ):
        raise Phase3ReportError("continuous GPU process evidence is not an exact pass")


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
        if stdout != canonical_json_bytes(worker_result) + b"\n":
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
    allocation = all(
        run.runtime is not None
        and _audit_zero_allocation(run.runtime.get("allocation"))
        and _mapping(run.runtime.get("memory_evidence")) is not None
        and _mapping(run.runtime.get("memory_evidence")).get("timing_executed") is True
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
        "no_torch_cat_growth": (bool(source_audit.get("passed")) and all(_mapping(run.runtime.get("gqa_source")) is not None and _mapping(run.runtime.get("gqa_source")).get("passed") is True for run in runs if run.runtime is not None) and all(run.runtime is not None for run in runs), "recorded SUT source audit found forbidden growth"),
        "no_unexplained_measured_region_allocation": (allocation, "one or more exact decode operations issued allocator events or lacked normal timing"),
        "kv_head_cache_geometry": (all(_cache_geometry_passes(run) for run in runs), "cache geometry or byte accounting failed"),
        "gqa_not_materialized": (all(_gqa_passes(run) for run in runs), "GQA source/operator/storage audit failed"),
        "fixed_l_runner": (all(_fixed_runner_passes(run) for run in fixed), "one or more fixed-L processes did not complete its declared timing lane"),
        "growing_context_runner": (all(_growing_runner_passes(run) for run in growing), "one or more growing-context processes did not complete its declared trajectory"),
        "eager_lane": (all(run.manifest.status is RunStatus.COMPLETED and _validated_timing_host_ms(run) is not None for run in eager), "one or more eager processes did not complete valid timing"),
        "cuda_graph_capture_and_replay": (all(_graph_passes(run) for run in graph), "one or more CUDA Graph capture/replay controls failed"),
        "eager_graph_numerical_agreement": (all(_graph_agreement_passes(run) for run in graph), "one or more eager/graph comparisons failed"),
        "graph_replay_no_allocation": (all(_graph_allocation_passes(run) for run in graph), "one or more graph replay allocation controls failed"),
        "stable_output_checksums": (checksums, "output checksum derivation or replicate stability failed"),
        "independent_process_replicates": (independent, "exact independent process identities were not preserved"),
        "stability_threshold": (stability_pass, "both eager and graph host-wall CV summaries are required at CV <= 3%"),
        "no_backend_fallback": (no_fallback, "backend dispatch or process audit indicates fallback/ambiguity"),
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
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    worktree = subprocess.run(
        ["git", "status", "--short"],
        cwd=repository,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if (
        head.returncode != 0
        or head.stdout.strip() != git_sha
        or worktree.returncode != 0
        or bool(worktree.stdout.strip())
    ):
        raise Phase3ReportError("report derivation requires the exact clean recorded Git SHA")
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
    criteria = _derive_criteria(runs, stability, source_audit, repository)
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
    """Write a new immutable report bundle; never reuse an existing directory."""

    repository = Path(repository_root).resolve(strict=True)
    report, stability_payloads, derivation = build_phase3_g1_report(
        fixed_campaign_id,
        growing_campaign_id,
        repository_root=repository,
    )
    root = repository / "artifacts" / "phase3_reports"
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise Phase3ReportError("Phase 3 report root is unsafe")
    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fZ").lower()
    report_id = f"phase3-g1-{timestamp}-{report.git_sha[:8]}-{secrets.token_hex(3)}"
    if not _REPORT_ID.fullmatch(report_id):
        raise Phase3ReportError("generated report ID is invalid")
    directory = root / report_id
    os.mkdir(directory, 0o755)
    try:
        _exclusive_write(
            directory / "source_campaigns.json",
            canonical_json_bytes(
                {
                    "schema_version": "kvbench-phase3-report-sources-1.0.0",
                    "fixed_campaign_id": fixed_campaign_id,
                    "growing_campaign_id": growing_campaign_id,
                    "explicit_selection": True,
                }
            )
            + b"\n",
        )
        stability_root = directory / "stability"
        if stability_payloads:
            os.mkdir(stability_root, 0o755)
        for relative, payload in sorted(stability_payloads.items()):
            target = directory / relative
            _exclusive_write(target, canonical_json_bytes(payload) + b"\n")
        _exclusive_write(
            directory / "derivation.json",
            canonical_json_bytes(derivation) + b"\n",
        )
        _exclusive_write(
            directory / "report.json",
            canonical_json_bytes(report) + b"\n",
        )
        payload_paths = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
        )
        ledger = b"".join(
            f"{sha256_hex(path.read_bytes())}  {path.relative_to(directory).as_posix()}\n".encode(
                "utf-8"
            )
            for path in payload_paths
        )
        _exclusive_write(directory / "checksums.sha256", ledger)
        completion = {
            "schema_version": "kvbench-phase3-g1-completion-1.0.0",
            "report_id": report_id,
            "status": report.status.value,
            "report_sha256": sha256_hex((directory / "report.json").read_bytes()),
            "checksum_ledger_sha256": sha256_hex(ledger),
            "written_last": True,
        }
        _exclusive_write(
            directory / "COMPLETE",
            canonical_json_bytes(completion) + b"\n",
        )
        for path in sorted(directory.rglob("*"), reverse=True):
            path.chmod(0o444 if path.is_file() else 0o555)
        directory.chmod(0o555)
    except BaseException:
        # A partial directory is intentionally retained as immutable evidence.
        for path in sorted(directory.rglob("*"), reverse=True):
            try:
                path.chmod(0o444 if path.is_file() else 0o555)
            except OSError:
                pass
        directory.chmod(0o555)
        raise
    validation = validate_phase3_g1_report_directory(directory)
    if not validation["valid"]:
        raise Phase3ReportError("written Phase 3 report failed independent validation")
    return {
        "schema_version": "kvbench-phase3-g1-write-result-1.0.0",
        "ok": True,
        "report_id": report_id,
        "status": report.status.value,
        "report_dir": directory.relative_to(repository).as_posix(),
        "report_sha256": validation["report_sha256"],
        "execution_attempted": False,
        "timing_collected": False,
        "performance_claim_eligible": False,
    }


def validate_phase3_g1_report_directory(path: str | Path) -> dict[str, Any]:
    """Independently validate one finalized report bundle and stability hashes."""

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
        expected_report, expected_stability, expected_derivation = (
            build_phase3_g1_report(
                fixed_campaign_id,
                growing_campaign_id,
                repository_root=directory.parents[2],
                generated_at_utc=report.generated_at_utc,
            )
        )
        if expected_report.to_dict() != report_payload:
            errors.append("report differs from independently rederived run evidence")
        derivation_payload = _strict_json_object(
            directory / "derivation.json",
            canonical=True,
        )
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

"""Supervised single-point worker for the frozen Phase 3 BF16 campaign."""

from __future__ import annotations

import gc
import importlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import traceback
from typing import Any

from kvbench.config import load_phase3_admission_bundle
from kvbench.runtime.process_supervision import (
    HandshakeEvent,
    HandshakeStage,
    ProcessIdentity,
    ProcessSupervisionError,
    command_fingerprint,
    publish_bytes_no_replace,
    read_published_bytes,
    read_process_identity,
    write_handshake_event,
)
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.runtime.phase3_raw_audit_evidence import (
    Phase3RawAuditRunIndex,
    parse_phase3_raw_audit_run_index_bytes,
)
from kvbench.runtime.phase3_worker_channels import (
    PHASE3_RAW_AUDIT_OPERATION_PLAN_ENV,
    Phase3RawAuditProducerError,
    Phase3RawAuditProducerRegistry,
    RawAuditOperationProducer,
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
    expand_phase3_process_points,
    sha256_hex,
)
from kvbench.schema.phase3 import (
    PHASE3_GPU_UUID,
    PHASE3_PYTHON_EXECUTABLE,
    PHASE3_REPOSITORY_ROOT,
)


IPC_PATH_ENV = "KVBENCH_PHASE3_IPC_PATH"
READY_PATH_ENV = "KVBENCH_PHASE3_AUDIT_READY"
RELEASE_PATH_ENV = "KVBENCH_PHASE3_AUDIT_RELEASE"
HANDSHAKE_TIMEOUT_ENV = "KVBENCH_PHASE3_HANDSHAKE_TIMEOUT_SECONDS"
HANDSHAKE_DIR_ENV = "KVBENCH_PHASE3_HANDSHAKE_DIR"
COMMAND_FINGERPRINT_ENV = "KVBENCH_PHASE3_COMMAND_FINGERPRINT"
RAW_AUDIT_ROOT_ENV = "KVBENCH_PHASE3_RAW_AUDIT_ROOT"
RAW_AUDIT_INDEX_IPC_ENV = "KVBENCH_PHASE3_RAW_AUDIT_INDEX_IPC_PATH"
HANDSHAKE_TIMEOUT_SECONDS = 120.0
MAX_REASON_CHARACTERS = 1000
WORKER_EVIDENCE_V2 = "kvbench-phase3-worker-evidence-2.0.0"

_ACTIVE_HANDSHAKE_DIRECTORY: Path | None = None
_ACTIVE_PROCESS_IDENTITY: ProcessIdentity | None = None
_ACTIVE_RUN_ID: str | None = None
_ACTIVE_COMMAND_FINGERPRINT: str | None = None
_ACTIVE_WORKER_STAGES: list[HandshakeStage] = []
_ACTIVE_RAW_AUDIT_RUN_INDEX: Phase3RawAuditRunIndex | None = None
_ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS: (
    tuple[Phase3AuditOperationKey, ...] | None
) = None



class WorkerProtocolError(RuntimeError):
    """The coordinator/worker supervision contract was malformed."""


def build_phase3_worker_evidence_v2(
    index: Phase3RawAuditRunIndex,
) -> dict[str, Any]:
    """Build the minimal IPC envelope; raw evidence bytes stay out-of-band."""

    if type(index) is not Phase3RawAuditRunIndex:
        raise TypeError("worker evidence v2 requires a raw-audit run index")
    index_bytes = canonical_json_bytes(index.to_dict())
    try:
        reconstructed = parse_phase3_raw_audit_run_index_bytes(index_bytes)
    except Exception as error:
        raise WorkerProtocolError(
            "worker evidence v2 index failed canonical reconstruction"
        ) from error
    if reconstructed != index:
        raise WorkerProtocolError("worker evidence v2 index changed on reconstruction")
    return {
        "schema_version": WORKER_EVIDENCE_V2,
        "raw_audit_run_index": index.to_dict(),
        "raw_audit_run_index_sha256": sha256_hex(index_bytes),
    }


def _initialize_phase3_raw_audit_operation_plan(
    *,
    run_id: str,
    point_id: str,
) -> tuple[Phase3AuditOperationKey, ...]:
    """Load the exact coordinator-owned plan before importing CUDA."""

    global _ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS
    if _ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS is not None:
        raise WorkerProtocolError("raw-audit operation plan is already initialized")
    encoded = os.environ.get(PHASE3_RAW_AUDIT_OPERATION_PLAN_ENV)
    if not encoded:
        raise WorkerProtocolError("raw-audit operation plan is absent")
    try:
        operations = parse_phase3_raw_audit_operation_plan_bytes(
            encoded.encode("utf-8")
        )
    except (UnicodeEncodeError, Phase3RawAuditProducerError) as error:
        raise WorkerProtocolError("raw-audit operation plan is invalid") from error
    if (
        operations[0].run_id != run_id
        or operations[0].point_id != point_id
    ):
        raise WorkerProtocolError(
            "raw-audit operation plan differs from the active worker"
        )
    _ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS = operations
    return operations


def register_phase3_raw_audit_run_index(index: Phase3RawAuditRunIndex) -> None:
    """Register the sole plan-bound index before measurement starts."""

    global _ACTIVE_RAW_AUDIT_RUN_INDEX
    if type(index) is not Phase3RawAuditRunIndex:
        raise TypeError("raw-audit run index has the wrong type")
    if _ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS is None:
        raise WorkerProtocolError("raw-audit operation plan is not initialized")
    if _ACTIVE_WORKER_STAGES != [
        HandshakeStage.WORKER_STARTED,
        HandshakeStage.CUDA_CONTEXT_CREATED,
    ]:
        raise WorkerProtocolError(
            "raw-audit run index must be registered before measurement"
        )
    if _ACTIVE_RUN_ID is not None and index.run_id != _ACTIVE_RUN_ID:
        raise WorkerProtocolError("raw-audit run index differs from active run")
    observed = tuple(record.operation for record in index.records)
    if observed != _ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS:
        raise WorkerProtocolError(
            "raw-audit run index differs from the trusted operation plan"
        )
    if _ACTIVE_RAW_AUDIT_RUN_INDEX is not None:
        raise WorkerProtocolError("raw-audit run index is already registered")
    _ACTIVE_RAW_AUDIT_RUN_INDEX = index


def _reset_phase3_raw_audit_run_index() -> None:
    """Reset process-local index registration at the start of a worker run."""

    global _ACTIVE_RAW_AUDIT_RUN_INDEX
    _ACTIVE_RAW_AUDIT_RUN_INDEX = None


def _reset_phase3_raw_audit_operation_plan() -> None:
    """Reset the process-local trusted plan at the start of a worker run."""

    global _ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS
    _ACTIVE_RAW_AUDIT_EXPECTED_OPERATIONS = None


def _publish_phase3_worker_evidence_channels(
    *,
    primary_path: Path,
    raw_index_path: Path,
    primary_evidence: dict[str, Any],
    raw_index: Phase3RawAuditRunIndex,
    run_id: str,
    point_id: str,
) -> str:
    """Exclusively publish both channels and return their commitment digest."""

    primary_bytes = canonical_json_bytes(primary_evidence) + b"\n"
    raw_index_bytes = canonical_json_bytes(
        build_phase3_worker_evidence_v2(raw_index)
    ) + b"\n"
    _exclusive_write(primary_path, primary_bytes)
    _exclusive_write(raw_index_path, raw_index_bytes)
    return phase3_worker_channel_commitment_sha256(
        run_id=run_id,
        point_id=point_id,
        primary_evidence_bytes=primary_bytes,
        raw_audit_index_bytes=raw_index_bytes,
    )


def _safe_reason(error: BaseException) -> str:
    text = " ".join(str(error).split())
    if not text:
        text = type(error).__name__
    return f"{type(error).__name__}: {text}"[:MAX_REASON_CHARACTERS]


def _exclusive_write(path: Path, data: bytes) -> None:
    try:
        publish_bytes_no_replace(path, data)
    except ProcessSupervisionError as error:
        raise WorkerProtocolError(
            "worker IPC publication failed"
        ) from error


def _process_start_ticks() -> int:
    raw = Path("/proc/self/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    tail = raw[closing + 2 :].split()
    if closing < 0 or len(tail) < 20:
        raise WorkerProtocolError("cannot parse worker process identity")
    value = int(tail[19])
    if value < 0:
        raise WorkerProtocolError("worker process start ticks are invalid")
    return value


def _required_ipc_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise WorkerProtocolError(f"required worker environment is absent: {name}")
    path = Path(value)
    if not path.is_absolute():
        raise WorkerProtocolError("worker IPC path must be absolute")
    return path


def _worker_argv(
    plan_path: str,
    point_id: str,
    replicate: int,
    run_id: str,
) -> tuple[str, ...]:
    return (
        PHASE3_PYTHON_EXECUTABLE,
        "-m",
        "kvbench",
        "phase3-worker",
        "--plan",
        plan_path,
        "--point-id",
        point_id,
        "--replicate",
        str(replicate),
        "--run-id",
        run_id,
    )


def _initialize_worker_handshake(
    *,
    plan_path: str,
    point_id: str,
    replicate: int,
    run_id: str,
) -> None:
    global _ACTIVE_COMMAND_FINGERPRINT
    global _ACTIVE_HANDSHAKE_DIRECTORY
    global _ACTIVE_PROCESS_IDENTITY
    global _ACTIVE_RUN_ID
    global _ACTIVE_WORKER_STAGES

    handshake_directory = _required_ipc_path(HANDSHAKE_DIR_ENV)
    try:
        metadata = handshake_directory.lstat()
    except OSError as error:
        raise WorkerProtocolError("worker handshake directory is absent") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise WorkerProtocolError("worker handshake directory is unsafe")
    expected_fingerprint = os.environ.get(COMMAND_FINGERPRINT_ENV)
    if not expected_fingerprint:
        raise WorkerProtocolError("expected worker command fingerprint is absent")
    environment = dict(os.environ)
    environment.pop(COMMAND_FINGERPRINT_ENV, None)
    environment_sha256 = sha256_hex(canonical_json_bytes(environment))
    working_directory = str(Path.cwd().resolve(strict=True))
    if working_directory != PHASE3_REPOSITORY_ROOT:
        raise WorkerProtocolError("worker working directory differs")
    observed_fingerprint = command_fingerprint(
        _worker_argv(plan_path, point_id, replicate, run_id),
        working_directory=working_directory,
        environment_sha256=environment_sha256,
    )
    try:
        identity = read_process_identity(os.getpid())
    except ProcessSupervisionError as error:
        raise WorkerProtocolError("cannot register worker process identity") from error
    _reset_phase3_raw_audit_operation_plan()
    _ACTIVE_HANDSHAKE_DIRECTORY = handshake_directory
    _ACTIVE_PROCESS_IDENTITY = identity
    _ACTIVE_RUN_ID = run_id
    _ACTIVE_COMMAND_FINGERPRINT = observed_fingerprint
    _ACTIVE_WORKER_STAGES = []
    _reset_phase3_raw_audit_run_index()
    _emit_worker_stage(HandshakeStage.WORKER_STARTED)
    if observed_fingerprint != expected_fingerprint:
        raise WorkerProtocolError("worker command fingerprint differs")


def _handshake_timestamp() -> str:
    from datetime import datetime, timezone

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _emit_worker_stage(
    stage: HandshakeStage,
    *,
    evidence_sha256: str | None = None,
) -> None:
    if (
        _ACTIVE_HANDSHAKE_DIRECTORY is None
        or _ACTIVE_PROCESS_IDENTITY is None
        or _ACTIVE_RUN_ID is None
        or _ACTIVE_COMMAND_FINGERPRINT is None
    ):
        raise WorkerProtocolError("worker handshake is not initialized")
    worker_stages = tuple(HandshakeStage)[:-1]
    if len(_ACTIVE_WORKER_STAGES) >= len(worker_stages):
        raise WorkerProtocolError("worker handshake is already complete")
    expected = worker_stages[len(_ACTIVE_WORKER_STAGES)]
    if stage is not expected:
        raise WorkerProtocolError(
            f"worker handshake expected {expected.value}, not {stage.value}"
        )
    identity = _ACTIVE_PROCESS_IDENTITY
    event = HandshakeEvent(
        sequence=stage.sequence,
        stage=stage,
        recorded_at_utc=_handshake_timestamp(),
        run_id=_ACTIVE_RUN_ID,
        gpu_uuid=PHASE3_GPU_UUID,
        pid=identity.pid,
        process_start_time_ticks=identity.start_time_ticks,
        parent_pid=identity.parent_pid,
        command_fingerprint=_ACTIVE_COMMAND_FINGERPRINT,
        evidence_sha256=evidence_sha256,
    )
    try:
        write_handshake_event(_ACTIVE_HANDSHAKE_DIRECTORY, event)
    except ProcessSupervisionError as error:
        raise WorkerProtocolError("cannot publish worker handshake event") from error
    _ACTIVE_WORKER_STAGES.append(stage)


def _await_supervisor_release(
    *,
    plan_path: str,
    point_id: str,
    replicate: int,
    run_id: str,
) -> tuple[Path, Path, Path, Path]:
    ipc_path = _required_ipc_path(IPC_PATH_ENV)
    ready_path = _required_ipc_path(READY_PATH_ENV)
    release_path = _required_ipc_path(RELEASE_PATH_ENV)
    handshake_directory = _required_ipc_path(HANDSHAKE_DIR_ENV)
    raw_audit_root = _required_ipc_path(RAW_AUDIT_ROOT_ENV)
    raw_audit_index_path = _required_ipc_path(RAW_AUDIT_INDEX_IPC_ENV)
    parents = {
        path.parent.resolve(strict=True)
        for path in (
            ipc_path,
            ready_path,
            release_path,
            handshake_directory,
            raw_audit_root,
            raw_audit_index_path,
        )
    }
    if len(parents) != 1 or len(
        {
            ipc_path.name,
            ready_path.name,
            release_path.name,
            handshake_directory.name,
            raw_audit_root.name,
            raw_audit_index_path.name,
        }
    ) != 6:
        raise WorkerProtocolError("worker IPC paths must be distinct siblings")
    try:
        raw_metadata = raw_audit_root.lstat()
    except OSError as error:
        raise WorkerProtocolError("worker raw-audit root is absent") from error
    if (
        stat.S_ISLNK(raw_metadata.st_mode)
        or not stat.S_ISDIR(raw_metadata.st_mode)
        or raw_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(raw_metadata.st_mode) != 0o700
        or any(raw_audit_root.iterdir())
    ):
        raise WorkerProtocolError(
            "worker raw-audit root must be empty, private, and process-owned"
        )
    _initialize_worker_handshake(
        plan_path=plan_path,
        point_id=point_id,
        replicate=replicate,
        run_id=run_id,
    )
    ready_payload = {
        "schema_version": "kvbench-phase3-worker-ready-1.0.0",
        "pid": os.getpid(),
        "process_start_time_ticks": _process_start_ticks(),
        "cuda_imported": "torch" in sys.modules,
    }
    if ready_payload["cuda_imported"]:
        raise WorkerProtocolError("worker imported torch before the exclusivity audit")
    _exclusive_write(ready_path, canonical_json_bytes(ready_payload) + b"\n")
    raw_timeout = os.environ.get(
        HANDSHAKE_TIMEOUT_ENV,
        str(HANDSHAKE_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw_timeout)
    except ValueError as error:
        raise WorkerProtocolError("handshake timeout is invalid") from error
    if timeout <= 0.0 or timeout > 600.0:
        raise WorkerProtocolError("handshake timeout is outside the safe range")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            release_payload = read_published_bytes(
                release_path,
                maximum_bytes=16,
            )
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        except ProcessSupervisionError as error:
            raise WorkerProtocolError(
                "worker release path is unsafe"
            ) from error
        if release_payload != b"release\n":
            raise WorkerProtocolError("worker release token is invalid")
        return ipc_path, ready_path, release_path, raw_audit_root
    raise WorkerProtocolError("worker supervisor release timed out")


def _point_for_request(plan_path: str, point_id: str, replicate: int) -> tuple[Any, Any]:
    bundle = load_phase3_admission_bundle(plan_path)
    if not bundle.execution_ready:
        raise WorkerProtocolError("Phase 3 bundle is not narrowly execution-authorized")
    matches = [
        point
        for point in expand_phase3_process_points(bundle.plan)
        if point.point_id == point_id
    ]
    if len(matches) != 1 or matches[0].process_replicate != replicate:
        raise WorkerProtocolError("worker request is outside the frozen process grid")
    return bundle, matches[0]


def _small_attention_controls(torch: Any, *, device: Any) -> dict[str, Any]:
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.numerical import (
        compare_tensors_untimed,
        small_attention_reference,
    )

    torch.manual_seed(20260722)
    records: list[dict[str, Any]] = []
    for batch in (1, 2):
        for length in (7, 17):
            for mode in ("causal_gqa", "decode_gqa", "causal_mha"):
                causal = mode != "decode_gqa"
                query_heads = 32
                kv_heads = 32 if mode == "causal_mha" else 8
                query_length = length if causal else 1
                query = torch.randn(
                    (batch, query_heads, query_length, 128),
                    dtype=torch.bfloat16,
                    device=device,
                )
                key = torch.randn(
                    (batch, kv_heads, length, 128),
                    dtype=torch.bfloat16,
                    device=device,
                )
                value = torch.randn_like(key)
                with forced_flash_execution():
                    observed = torch.nn.functional.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=128**-0.5,
                        enable_gqa=True,
                    )
                reference = small_attention_reference(
                    query,
                    key,
                    value,
                    is_causal=causal,
                    scale=128**-0.5,
                )
                comparison = compare_tensors_untimed(
                    observed,
                    reference,
                    atol=0.02,
                    rtol=0.02,
                )
                records.append(
                    {
                        "batch_size": batch,
                        "context_length": length,
                        "mode": mode,
                        "boundary_first_finite": bool(
                            torch.isfinite(observed[..., 0, :]).all().item()
                        ),
                        "boundary_last_finite": bool(
                            torch.isfinite(observed[..., -1, :]).all().item()
                        ),
                        "comparison": comparison.to_dict(),
                    }
                )
    passed = all(
        item["boundary_first_finite"]
        and item["boundary_last_finite"]
        and item["comparison"]["passed"]
        for item in records
    )
    return {
        "passed": passed,
        "reference": "explicit_fp32_gqa_attention",
        "atol": 0.02,
        "rtol": 0.02,
        "records": records,
        "timing_collected": False,
    }


def _deterministic_ids(
    torch: Any,
    *,
    batch_size: int,
    length: int,
    offset: int,
    device: Any,
) -> Any:
    values = torch.arange(
        batch_size * length,
        dtype=torch.long,
        device=device,
    ).reshape(batch_size, length)
    return (values + offset) % 120_000 + 1_000


def _gqa_passed(runtime: dict[str, Any]) -> bool:
    source = runtime["gqa_source"]
    geometry = runtime["gqa_cache_geometry"]
    operators = runtime.get("gqa_operators", [runtime.get("gqa_operator")])
    return bool(
        source["passed"]
        and geometry["uses_kv_head_geometry"]
        and not geometry["query_head_storage_detected"]
        and operators
        and all(item is not None and item["passed"] for item in operators)
        and runtime["mha_control"]["passed"]
        and runtime["backend"]["backend_name"] == "FLASH_ATTENTION"
        and runtime["prefill_backend"]["backend_name"] == "FLASH_ATTENTION"
    )


def _classify_runtime(
    *,
    point: Any,
    runtime: dict[str, Any],
    numerical: dict[str, Any],
) -> tuple[RunStatus, str | None]:
    if not numerical["small_tensor"]["passed"] or not numerical["full_model"]["passed"]:
        return RunStatus.NUMERICAL_FAILED, "predeclared numerical reference failed"
    if point.graph_mode is GraphMode.CUDA_GRAPH:
        graph_control = numerical.get("full_model_graph")
        if not graph_control or not graph_control["passed"]:
            return RunStatus.GRAPH_REPLAY_FAILED, "full-model graph control failed"
        if (
            runtime.get("graph") is None
            or runtime["graph"].get("fallback") is not False
            or runtime["graph"].get("consecutive_replay_outputs_exact") is not True
            or runtime["graph"].get("first_replay_checksum")
            != runtime["graph"].get("second_replay_checksum")
            or runtime.get("eager_graph_comparison", {}).get("passed") is not True
        ):
            return RunStatus.GRAPH_REPLAY_FAILED, "runner graph replay evidence failed"
    if point.runner_kind is RunnerKind.FIXED_L:
        if not runtime["historical_cache_unchanged"] or not runtime["cache_pointers_stable"]:
            return RunStatus.STATE_DRIFT_DETECTED, "fixed-L cache state drift detected"
    elif not runtime["cache_pointers_stable"]:
        return RunStatus.STATE_DRIFT_DETECTED, "growing cache pointer drift detected"
    if not runtime["output_finite"]:
        return RunStatus.NUMERICAL_FAILED, "actual admission output was non-finite"
    if not _gqa_passed(runtime):
        return RunStatus.GQA_MATERIALIZATION_DETECTED, "GQA non-materialization audit failed"
    if "error" in runtime["telemetry_before"] or "error" in runtime["telemetry_after"]:
        return RunStatus.RUNTIME_FAILED, "required telemetry snapshot unavailable"
    if not runtime["allocation"]["passed"]:
        return RunStatus.ALLOCATION_FAILED, (
            runtime["allocation"].get("failure_reason")
            or "measured-region allocation audit failed"
        )
    return RunStatus.COMPLETED, None


def _aggregate_growing_checksum(runtime: dict[str, Any]) -> str:
    payload = [
        {
            "step": item["step"],
            "historical_active_length": item["historical_active_length"],
            "output_checksum": item["output_checksum"],
            "output_finite": item["output_finite"],
        }
        for item in runtime["step_evidence"]
    ]
    return sha256_hex(canonical_json_bytes(payload))


def _map_exception(error: BaseException) -> RunStatus:
    from kvbench.runtime.allocation import AllocationAuditError
    from kvbench.runtime.backend import BackendFallbackError, BackendUnsupportedError
    from kvbench.runtime.bf16_endpoint import EndpointGeometryError
    from kvbench.runtime.cuda_graph import GraphCaptureError, GraphReplayError
    from kvbench.runtime.model_loader import ModelAccessError, ModelIdentityError
    from kvbench.runtime.static_cache import CacheBoundsError, CacheStateError
    from kvbench.runtime.timing import TimingFailure

    if isinstance(error, ModelIdentityError):
        return RunStatus.MODEL_IDENTITY_UNRESOLVED
    if isinstance(error, ModelAccessError):
        return RunStatus.MODEL_ACCESS_BLOCKED
    if isinstance(error, BackendFallbackError):
        return RunStatus.BACKEND_FALLBACK
    if isinstance(error, BackendUnsupportedError):
        return RunStatus.BACKEND_UNSUPPORTED
    if isinstance(error, EndpointGeometryError):
        return RunStatus.UNSUPPORTED_GEOMETRY
    if isinstance(error, CacheBoundsError):
        return RunStatus.CAPACITY_INFEASIBLE
    if isinstance(error, CacheStateError):
        return RunStatus.STATE_DRIFT_DETECTED
    if isinstance(error, GraphCaptureError):
        return RunStatus.GRAPH_CAPTURE_FAILED
    if isinstance(error, GraphReplayError):
        return RunStatus.GRAPH_REPLAY_FAILED
    if isinstance(error, AllocationAuditError):
        return RunStatus.ALLOCATION_FAILED
    if isinstance(error, TimingFailure):
        return RunStatus.RUNTIME_FAILED
    try:
        torch = importlib.import_module("torch")
        if isinstance(error, torch.OutOfMemoryError):
            return RunStatus.CAPACITY_INFEASIBLE
    except ModuleNotFoundError:
        pass
    return RunStatus.ABORTED


def _phase3_raw_audit_producer_bindings(
    *,
    expected_operations: tuple[Phase3AuditOperationKey, ...],
    torch: Any,
    device: Any,
    loaded: Any,
    point: Any,
    prefix_input_ids: Any,
    decode_input_ids: Any,
) -> tuple[
    tuple[Phase3AuditOperationKey, RawAuditOperationProducer],
    ...,
]:
    """Return production producer bindings supplied by endpoint integration."""

    del (
        expected_operations,
        torch,
        device,
        loaded,
        point,
        prefix_input_ids,
        decode_input_ids,
    )
    raise WorkerProtocolError(
        "production Phase 3 raw-audit producer API is not installed"
    )


def _collect_and_register_phase3_raw_audits(
    *,
    expected_operations: tuple[Phase3AuditOperationKey, ...],
    raw_audit_root: Path,
    producer_bindings: tuple[
        tuple[Phase3AuditOperationKey, RawAuditOperationProducer],
        ...,
    ],
) -> Phase3RawAuditRunIndex:
    """Bind all producers, collect once, and enforce pre-timing admission."""

    if type(producer_bindings) is not tuple:
        raise WorkerProtocolError("raw-audit producer bindings must be a tuple")
    registry = Phase3RawAuditProducerRegistry(expected_operations)
    for binding in producer_bindings:
        if type(binding) is not tuple or len(binding) != 2:
            raise WorkerProtocolError("raw-audit producer binding is malformed")
        operation, producer = binding
        registry.register(operation, producer)
    index = registry.collect(raw_audit_root)
    register_phase3_raw_audit_run_index(index)
    require_phase3_raw_audit_measurement_admission(
        index,
        expected_operations,
    )
    return index


def execute_worker(
    *,
    plan_path: str,
    point_id: str,
    replicate: int,
    run_id: str,
) -> Phase3WorkerResult:
    """Execute one point after the supervisor has certified exclusivity."""

    ipc_path, _, _, raw_audit_root = _await_supervisor_release(
        plan_path=plan_path,
        point_id=point_id,
        replicate=replicate,
        run_id=run_id,
    )
    bundle, point = _point_for_request(plan_path, point_id, replicate)
    expected_operations = (
        bundle.plan.measurement.measured_count
        * bundle.plan.measurement.measured_batches
        if point.runner_kind is RunnerKind.FIXED_L
        else point.output_steps
        * bundle.plan.measurement.measured_count
        * bundle.plan.measurement.measured_batches
    )
    evidence: dict[str, Any] = {
        "schema_version": "kvbench-phase3-worker-evidence-1.0.0",
        "run_id": run_id,
        "point_id": point_id,
        "stage": "initializing_cuda",
        "runtime": None,
        "numerical": None,
        "model_identity": None,
        "raw_audit_root_initialized": raw_audit_root.is_dir(),
    }
    completed_operations = 0
    failed_operations = 0
    output_checksum: str | None = None
    status = RunStatus.ABORTED
    reason: str | None = None
    try:
        raw_audit_operations = _initialize_phase3_raw_audit_operation_plan(
            run_id=run_id,
            point_id=point_id,
        )
        torch = importlib.import_module("torch")
        torch.cuda.init()
        _emit_worker_stage(HandshakeStage.CUDA_CONTEXT_CREATED)

        from kvbench.runtime.cuda_graph import validate_full_model_fixed_graph
        from kvbench.runtime.fixed_l_runner import run_fixed_l
        from kvbench.runtime.growing_context_runner import run_growing_context
        from kvbench.runtime.model_loader import load_frozen_model
        from kvbench.runtime.numerical import validate_full_model_reference

        device = torch.device("cuda:0")
        loaded = load_frozen_model(device=str(device))
        evidence["model_identity"] = loaded.identity.to_dict()
        small = _small_attention_controls(torch, device=device)
        reference_prefix = torch.arange(
            1_000,
            1_008,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        reference_decode = torch.arange(
            2_000,
            2_003,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        full_reference = validate_full_model_reference(
            loaded.model,
            reference_prefix,
            reference_decode,
        )
        numerical: dict[str, Any] = {
            "small_tensor": small,
            "full_model": full_reference.to_dict(),
            "full_model_graph": None,
        }
        if point.graph_mode is GraphMode.CUDA_GRAPH:
            graph_control = validate_full_model_fixed_graph(
                loaded.model,
                reference_prefix,
                reference_decode[:, :1],
            )
            numerical["full_model_graph"] = graph_control.to_dict()
        evidence["numerical"] = numerical
        del reference_prefix, reference_decode
        gc.collect()
        torch.cuda.empty_cache()
        prefix = _deterministic_ids(
            torch,
            batch_size=point.batch_size,
            length=point.context_length,
            offset=10_000,
            device=device,
        )
        if point.runner_kind is RunnerKind.FIXED_L:
            decode_input_ids = _deterministic_ids(
                torch,
                batch_size=point.batch_size,
                length=1,
                offset=40_000,
                device=device,
            )
        else:
            decode_input_ids = _deterministic_ids(
                torch,
                batch_size=point.batch_size,
                length=point.output_steps,
                offset=50_000,
                device=device,
            )
        evidence["stage"] = "collecting_raw_audits"
        producer_bindings = _phase3_raw_audit_producer_bindings(
            expected_operations=raw_audit_operations,
            torch=torch,
            device=device,
            loaded=loaded,
            point=point,
            prefix_input_ids=prefix,
            decode_input_ids=decode_input_ids,
        )
        _collect_and_register_phase3_raw_audits(
            expected_operations=raw_audit_operations,
            raw_audit_root=raw_audit_root,
            producer_bindings=producer_bindings,
        )

        evidence["stage"] = "running_point"
        _emit_worker_stage(HandshakeStage.MEASUREMENT_STARTED)
        try:
            if point.runner_kind is RunnerKind.FIXED_L:
                result = run_fixed_l(
                    loaded.model,
                    prefix,
                    decode_input_ids,
                    context_length=point.context_length,
                    graph_mode=point.graph_mode.value,
                    warmup_steps=bundle.plan.measurement.warmup_count,
                    measured_steps=bundle.plan.measurement.measured_count,
                    measured_batches=bundle.plan.measurement.measured_batches,
                )
            else:
                result = run_growing_context(
                    loaded.model,
                    prefix,
                    decode_input_ids,
                    starting_context=point.context_length,
                    warmup_trajectories=bundle.plan.measurement.warmup_count,
                )
        finally:
            _emit_worker_stage(HandshakeStage.MEASUREMENT_FINISHED)
        runtime = result.to_dict()
        evidence["runtime"] = runtime
        completed_operations = (
            expected_operations if runtime.get("timing") is not None else 0
        )
        output_checksum = (
            runtime["output_checksum"]
            if point.runner_kind is RunnerKind.FIXED_L
            else _aggregate_growing_checksum(runtime)
        )
        status, reason = _classify_runtime(
            point=point,
            runtime=runtime,
            numerical=numerical,
        )
        evidence["stage"] = "completed" if status is RunStatus.COMPLETED else "failed"
    except BaseException as error:
        status = _map_exception(error)
        reason = _safe_reason(error)
        submitted = getattr(error, "submitted_operations", 0)
        if isinstance(submitted, int) and not isinstance(submitted, bool):
            completed_operations = max(0, min(submitted, expected_operations))
        if evidence.get("stage") == "running_point":
            failed_operations = 1
            completed_operations = min(
                completed_operations,
                expected_operations - failed_operations,
            )
        evidence["stage"] = "failed"
        evidence["exception_type"] = type(error).__name__
        traceback.print_exc(file=sys.stderr)
    worker_result = Phase3WorkerResult(
        schema_version=Phase3WorkerResult.SCHEMA_VERSION,
        run_id=run_id,
        point_id=point_id,
        runner_kind=point.runner_kind,
        count_unit=bundle.plan.measurement.count_unit,
        status=status,
        expected_operations=expected_operations,
        completed_operations=completed_operations,
        failed_operations=failed_operations,
        output_checksum=output_checksum,
        failure_reason=reason,
    )
    evidence["worker_result"] = worker_result.to_dict()
    raw_index_path = _required_ipc_path(RAW_AUDIT_INDEX_IPC_ENV)
    if _ACTIVE_RAW_AUDIT_RUN_INDEX is None:
        _exclusive_write(ipc_path, canonical_json_bytes(evidence) + b"\n")
    else:
        commitment_sha256 = _publish_phase3_worker_evidence_channels(
            primary_path=ipc_path,
            raw_index_path=raw_index_path,
            primary_evidence=evidence,
            raw_index=_ACTIVE_RAW_AUDIT_RUN_INDEX,
            run_id=run_id,
            point_id=point_id,
        )
    if (
        _ACTIVE_RAW_AUDIT_RUN_INDEX is not None
        and _ACTIVE_WORKER_STAGES == list(tuple(HandshakeStage)[:4])
    ):
        _emit_worker_stage(
            HandshakeStage.EVIDENCE_FLUSHED,
            evidence_sha256=commitment_sha256,
        )
    return worker_result


def emit_worker_result(result: Phase3WorkerResult) -> None:
    """Emit the sole canonical stdout record used by the coordinator."""

    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    sys.stdout.buffer.flush()
    if _ACTIVE_WORKER_STAGES == list(tuple(HandshakeStage)[:5]):
        _emit_worker_stage(HandshakeStage.WORKER_EXITING)

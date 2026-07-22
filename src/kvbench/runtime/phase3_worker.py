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
    read_process_identity,
    write_handshake_event,
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
HANDSHAKE_TIMEOUT_SECONDS = 120.0
MAX_REASON_CHARACTERS = 1000

_ACTIVE_HANDSHAKE_DIRECTORY: Path | None = None
_ACTIVE_PROCESS_IDENTITY: ProcessIdentity | None = None
_ACTIVE_RUN_ID: str | None = None
_ACTIVE_COMMAND_FINGERPRINT: str | None = None
_ACTIVE_WORKER_STAGES: list[HandshakeStage] = []


class WorkerProtocolError(RuntimeError):
    """The coordinator/worker supervision contract was malformed."""


def _safe_reason(error: BaseException) -> str:
    text = " ".join(str(error).split())
    if not text:
        text = type(error).__name__
    return f"{type(error).__name__}: {text}"[:MAX_REASON_CHARACTERS]


def _exclusive_write(path: Path, data: bytes) -> None:
    if not path.is_absolute() or path.parent.is_symlink():
        raise WorkerProtocolError("IPC paths must use a real absolute parent")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise WorkerProtocolError("IPC parent is not a directory")
    target = parent / path.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


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
    _ACTIVE_HANDSHAKE_DIRECTORY = handshake_directory
    _ACTIVE_PROCESS_IDENTITY = identity
    _ACTIVE_RUN_ID = run_id
    _ACTIVE_COMMAND_FINGERPRINT = observed_fingerprint
    _ACTIVE_WORKER_STAGES = []
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
) -> tuple[Path, Path, Path]:
    ipc_path = _required_ipc_path(IPC_PATH_ENV)
    ready_path = _required_ipc_path(READY_PATH_ENV)
    release_path = _required_ipc_path(RELEASE_PATH_ENV)
    handshake_directory = _required_ipc_path(HANDSHAKE_DIR_ENV)
    parents = {
        path.parent.resolve(strict=True)
        for path in (ipc_path, ready_path, release_path, handshake_directory)
    }
    if len(parents) != 1 or len(
        {ipc_path.name, ready_path.name, release_path.name, handshake_directory.name}
    ) != 4:
        raise WorkerProtocolError("worker IPC paths must be distinct siblings")
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
            metadata = release_path.lstat()
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise WorkerProtocolError("worker release path is unsafe")
        if release_path.read_bytes() != b"release\n":
            raise WorkerProtocolError("worker release token is invalid")
        return ipc_path, ready_path, release_path
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


def execute_worker(
    *,
    plan_path: str,
    point_id: str,
    replicate: int,
    run_id: str,
) -> Phase3WorkerResult:
    """Execute one point after the supervisor has certified exclusivity."""

    ipc_path, _, _ = _await_supervisor_release(
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
    }
    completed_operations = 0
    failed_operations = 0
    output_checksum: str | None = None
    status = RunStatus.ABORTED
    reason: str | None = None
    try:
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
        evidence["stage"] = "running_point"
        _emit_worker_stage(HandshakeStage.MEASUREMENT_STARTED)
        try:
            if point.runner_kind is RunnerKind.FIXED_L:
                current = _deterministic_ids(
                    torch,
                    batch_size=point.batch_size,
                    length=1,
                    offset=40_000,
                    device=device,
                )
                result = run_fixed_l(
                    loaded.model,
                    prefix,
                    current,
                    context_length=point.context_length,
                    graph_mode=point.graph_mode.value,
                    warmup_steps=bundle.plan.measurement.warmup_count,
                    measured_steps=bundle.plan.measurement.measured_count,
                    measured_batches=bundle.plan.measurement.measured_batches,
                )
            else:
                decode = _deterministic_ids(
                    torch,
                    batch_size=point.batch_size,
                    length=point.output_steps,
                    offset=50_000,
                    device=device,
                )
                result = run_growing_context(
                    loaded.model,
                    prefix,
                    decode,
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
    evidence_bytes = canonical_json_bytes(evidence) + b"\n"
    _exclusive_write(ipc_path, evidence_bytes)
    if _ACTIVE_WORKER_STAGES == list(tuple(HandshakeStage)[:4]):
        _emit_worker_stage(
            HandshakeStage.EVIDENCE_FLUSHED,
            evidence_sha256=sha256_hex(evidence_bytes),
        )
    return worker_result


def emit_worker_result(result: Phase3WorkerResult) -> None:
    """Emit the sole canonical stdout record used by the coordinator."""

    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    sys.stdout.buffer.flush()
    if _ACTIVE_WORKER_STAGES == list(tuple(HandshakeStage)[:5]):
        _emit_worker_stage(HandshakeStage.WORKER_EXITING)

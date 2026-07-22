"""Fail-closed coordinator for the two frozen Phase 3 BF16 plans."""

from __future__ import annotations

from collections import Counter
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import tempfile
import time
from typing import Any, Mapping

from kvbench.config import REPOSITORY_ROOT, ExperimentBundle, load_phase3_admission_bundle
from kvbench.errors import KVBenchError, SchemaValidationError
from kvbench.runtime.artifacts import (
    phase3_artifact_store,
    sha256_file,
    validate_run_directory,
)
from kvbench.runtime.phase3_campaign import Phase3CampaignRecorder
from kvbench.schema import (
    BF16BackendIdentity,
    BF16CacheIdentity,
    ClaimClass,
    ConfigSourceKind,
    GateDisposition,
    MeasurementScope,
    MethodConfigFingerprint,
    Phase3CommandSpec,
    Phase3RunManifest,
    Phase3WorkerResult,
    RunKind,
    RunStatus,
    canonical_json_bytes,
    derive_cache_layout_fingerprint,
    derive_phase3_point_fingerprint,
    expand_phase3_process_points,
    sha256_hex,
)
from kvbench.schema.phase3 import (
    PHASE3_CONTRACT_FINGERPRINT,
    PHASE3_DRIVER_VERSION,
    PHASE3_E00_MANIFEST_SHA256,
    PHASE3_E00_RUN_ID,
    PHASE3_GPU_FULL_NAME,
    PHASE3_GPU_UUID,
    PHASE3_HARDWARE_FINGERPRINT,
    PHASE3_HARDWARE_ID,
    PHASE3_MEASUREMENT_PROTOCOL_FINGERPRINT,
    PHASE3_PCI_BUS_ID,
    PHASE3_PCI_DEVICE_ID,
    PHASE3_PYTHON_EXECUTABLE,
    PHASE3_REPOSITORY_ROOT,
    PHASE3_SOFTWARE_ENVIRONMENT_ID,
    PHASE3_SOFTWARE_FINGERPRINT,
)


PYTHON_EXECUTABLE = Path(PHASE3_PYTHON_EXECUTABLE)
PROCESS_QUERY = REPOSITORY_ROOT / "preflight" / "process_query.py"
STATIC_CACHE_SOURCE = REPOSITORY_ROOT / "src/kvbench/runtime/static_cache.py"
TORCH_PACKAGE_ROOT = (
    REPOSITORY_ROOT / ".venv/lib/python3.12/site-packages/torch"
)
E00_MANIFEST = (
    REPOSITORY_ROOT
    / "docs/evidence/e00"
    / PHASE3_E00_RUN_ID
    / "manifest.json"
)
FAILED_E00_MANIFEST = (
    REPOSITORY_ROOT
    / "docs/evidence/e00/e00-20260722T041628.190813Z-980eff7b6f59-0dd71f2d/manifest.json"
)
FAILED_E00_LEDGER = FAILED_E00_MANIFEST.with_name("checksums.sha256")
SUCCESSFUL_E00_LEDGER = E00_MANIFEST.with_name("checksums.sha256")
PHASE3_DEPENDENCY_LOCK = REPOSITORY_ROOT / "preflight/requirements-phase3.txt"
PHASE2_FINAL_SHA = "c16139b0f365eaa052b17cff2fd19c1d4c62a4d1"
PERFORMANCE_FREEZE_MARKER = REPOSITORY_ROOT / "PERFORMANCE_DATA_FROZEN"
MAX_STDOUT_BYTES = 1024 * 1024
MAX_IPC_BYTES = 64 * 1024 * 1024
WORKER_TIMEOUT_SECONDS = 3600
READY_TIMEOUT_SECONDS = 120
SENSITIVE_ENV_FRAGMENTS = (
    "token",
    "secret",
    "password",
    "credential",
    "api_key",
    "proxy",
)
SENSITIVE_ENV_KEY_EXEMPTIONS = frozenset({"TOKENIZERS_PARALLELISM"})


class Phase3CoordinatorError(RuntimeError):
    """Campaign coordination failed before a trustworthy worker result."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _run_checked(argv: tuple[str, ...]) -> str:
    result = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "TZ": "UTC",
        },
    )
    if result.returncode != 0:
        raise Phase3CoordinatorError(f"command failed: {argv[0]}")
    return result.stdout.strip()


def _git_identity() -> tuple[str, bool]:
    sha = _run_checked(("/usr/bin/git", "rev-parse", "HEAD"))
    dirty = bool(_run_checked(("/usr/bin/git", "status", "--porcelain=v1")))
    if dirty:
        raise Phase3CoordinatorError("Phase 3 execution requires a clean Git tree")
    ancestor = subprocess.run(
        ("/usr/bin/git", "merge-base", "--is-ancestor", PHASE2_FINAL_SHA, sha),
        cwd=REPOSITORY_ROOT,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
    )
    if ancestor.returncode != 0:
        raise Phase3CoordinatorError("accepted Phase 2 final commit is not an ancestor")
    return sha, dirty


def _validate_entry_evidence() -> None:
    expected_files = {
        FAILED_E00_MANIFEST: "0720734d29c90f609e51cf4c5e4f0b1fadce220e23e146e566f860bb962c0035",
        FAILED_E00_LEDGER: "8716fc317747e7e9b5c06017cb8e5339df610c5a89d0d7fbee82ad07fbc68b52",
        E00_MANIFEST: PHASE3_E00_MANIFEST_SHA256,
        SUCCESSFUL_E00_LEDGER: "5a610162163979aca97beb2b7b0b480befb85d0b4e63b77c26ec46c36864eca8",
        PHASE3_DEPENDENCY_LOCK: "cebe254a3e03a48e3e67100ce11d5623fc0dc722dc43e2f482152beb644a08e9",
    }
    if any(
        not path.is_file() or sha256_file(path) != expected
        for path, expected in expected_files.items()
    ):
        raise Phase3CoordinatorError("certified native-host E00 manifest changed")
    if PERFORMANCE_FREEZE_MARKER.exists() or PERFORMANCE_FREEZE_MARKER.is_symlink():
        raise Phase3CoordinatorError("quality freeze marker must remain absent")
    for forbidden in (
        REPOSITORY_ROOT / "paper-results",
        REPOSITORY_ROOT / "paper_results",
        REPOSITORY_ROOT / "artifacts/quality",
        REPOSITORY_ROOT / "artifacts/profiler",
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise Phase3CoordinatorError("forbidden result root exists")


def _live_hardware() -> dict[str, Any]:
    argv = (
        "/usr/bin/nvidia-smi",
        "--query-gpu=name,uuid,pci.bus_id,pci.device_id,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    )
    rows = [line.strip() for line in _run_checked(argv).splitlines() if line.strip()]
    if len(rows) != 1:
        raise Phase3CoordinatorError("Phase 3 requires exactly one visible GPU")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 6:
        raise Phase3CoordinatorError("live GPU identity field count differs")
    observed = tuple(values)
    expected = (
        PHASE3_GPU_FULL_NAME,
        PHASE3_GPU_UUID,
        PHASE3_PCI_BUS_ID,
        PHASE3_PCI_DEVICE_ID,
        PHASE3_DRIVER_VERSION,
        "12.0",
    )
    if observed != expected:
        raise Phase3CoordinatorError("live GPU differs from certified native-host G0")
    return {
        "schema_version": "kvbench-phase3-live-hardware-1.0.0",
        "gpu_name": values[0],
        "gpu_uuid": values[1],
        "pci_bus_id": values[2],
        "pci_device_id": values[3],
        "driver_version": values[4],
        "compute_capability": values[5],
        "native_g0_status": "PASS",
        "container_parity_status": "not_evaluated",
        "blocker_b010": "OPEN",
    }


def _worker_environment(temp_root: Path) -> dict[str, str]:
    environment = {
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "PYTHONPATH": (
            f"{REPOSITORY_ROOT / '.phase3/site-packages'}:"
            f"{REPOSITORY_ROOT / 'src'}"
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "KVBENCH_PHASE3_IPC_PATH": str(temp_root / "worker-evidence.json"),
        "KVBENCH_PHASE3_AUDIT_READY": str(temp_root / "worker-ready.json"),
        "KVBENCH_PHASE3_AUDIT_RELEASE": str(temp_root / "worker-release"),
        "KVBENCH_PHASE3_HANDSHAKE_TIMEOUT_SECONDS": str(READY_TIMEOUT_SECONDS),
    }
    if any(
        fragment in key.lower()
        for key in environment
        if key not in SENSITIVE_ENV_KEY_EXEMPTIONS
        for fragment in SENSITIVE_ENV_FRAGMENTS
    ):
        raise Phase3CoordinatorError("worker environment contains a forbidden key")
    return environment


def _parse_canonical_json(raw: bytes, *, maximum_bytes: int, label: str) -> dict[str, Any]:
    if not raw or len(raw) > maximum_bytes or not raw.endswith(b"\n"):
        raise Phase3CoordinatorError(f"{label} is absent, oversized, or unterminated")
    body = raw[:-1]
    if b"\n" in body or b"\r" in body:
        raise Phase3CoordinatorError(f"{label} must be one canonical JSON line")

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON")

    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Phase3CoordinatorError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != body:
        raise Phase3CoordinatorError(f"{label} is not a canonical JSON object")
    return value


def _process_snapshot(*, pid: int | None = None, start_ticks: int | None = None) -> dict[str, Any]:
    argv = ["/usr/bin/python3", str(PROCESS_QUERY)]
    if pid is not None or start_ticks is not None:
        if pid is None or start_ticks is None:
            raise Phase3CoordinatorError("supervised process identity is incomplete")
        argv.extend(
            [
                "--supervised-root-pid",
                str(pid),
                "--supervised-root-start-ticks",
                str(start_ticks),
            ]
        )
    result = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    payload = _parse_canonical_json(
        result.stdout,
        maximum_bytes=16 * 1024 * 1024,
        label="GPU process snapshot",
    )
    if result.returncode != payload.get("query_exit_code"):
        raise Phase3CoordinatorError("GPU process snapshot exit-code mismatch")
    return payload


def _snapshot_clean(snapshot: Mapping[str, Any], *, allow_supervised: bool) -> bool:
    allowed = snapshot.get("allowed_compute_processes")
    return bool(
        snapshot.get("query_exit_code") == 0
        and snapshot.get("errors") == []
        and snapshot.get("foreign_compute_processes") == []
        and snapshot.get("unknown_processes") == []
        and isinstance(allowed, list)
        and (allow_supervised or allowed == [])
    )


def _exclusive_release(path: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, b"release\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _wait_for_ready(process: subprocess.Popen[bytes], ready_path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if ready_path.exists():
            metadata = ready_path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise Phase3CoordinatorError("worker readiness path is unsafe")
            payload = _parse_canonical_json(
                ready_path.read_bytes(),
                maximum_bytes=16 * 1024,
                label="worker readiness",
            )
            if (
                payload.get("pid") != process.pid
                or not isinstance(payload.get("process_start_time_ticks"), int)
                or payload.get("process_start_time_ticks", -1) < 0
                or payload.get("cuda_imported") is not False
            ):
                raise Phase3CoordinatorError("worker readiness identity differs")
            return payload
        if process.poll() is not None:
            raise Phase3CoordinatorError("worker exited before readiness")
        time.sleep(0.05)
    raise Phase3CoordinatorError("worker readiness timed out")


def _terminate_worker(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def _cache_identity(point: Any) -> BF16CacheIdentity:
    implementation_sha256 = sha256_file(STATIC_CACHE_SOURCE)
    capacity = point.context_length + point.output_steps
    workspace_bytes = 32 * point.batch_size * (32 + 8) * 1 * 64 * 2
    payload = {
        "schema_version": BF16CacheIdentity.SCHEMA_VERSION,
        "layout_name": "layers_batch_kv_heads_context_head_dim",
        "dtype": "bfloat16",
        "num_layers": 32,
        "batch_size": point.batch_size,
        "num_kv_heads": 8,
        "capacity": capacity,
        "head_dim": 128,
        "tensor_storage_bytes": (
            2 * 32 * point.batch_size * 8 * capacity * 128 * 2
        ),
        "padding_bytes": 0,
        "workspace_bytes": workspace_bytes,
        "device": "cuda:0",
        "implementation_sha256": implementation_sha256,
        "layout_fingerprint": derive_cache_layout_fingerprint(
            num_layers=32,
            batch_size=point.batch_size,
            num_kv_heads=8,
            capacity=capacity,
            head_dim=128,
            device="cuda:0",
            workspace_bytes=workspace_bytes,
            implementation_sha256=implementation_sha256,
        ),
    }
    return BF16CacheIdentity.from_dict(payload)


def _backend_identity_stdlib() -> BF16BackendIdentity:
    """Verify backend bytes without importing PyTorch in the coordinator."""

    expected_sources = {
        "include/ATen/native/transformers/cuda/flash_attn/flash_api.h": (
            "1474aa79d8aa6ce39984dbc3c0aad9dba283ab819f034370e5cfb70980524ee7"
        ),
        "lib/libtorch_cuda.so": (
            "b248fb7e9935440965e4736eea48868b315ba41012734b7ce058fc0a2d0b1984"
        ),
        "nn/attention/__init__.py": (
            "56e10b6f965cc050db782dd4dc472097c9b02ec5b5fe3ab2c8b04055c0b0bbe0"
        ),
        "nn/attention/varlen.py": (
            "2f5384e0bc8ce371d00a1c09d38ad019517009798e7cb3434f56cf4b9fa351ea"
        ),
        "nn/functional.py": (
            "27493186ee22f811b553e31d9c804d4d46716d1be62d034d731537f66f27ef19"
        ),
    }
    for relative, expected in expected_sources.items():
        path = TORCH_PACKAGE_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3CoordinatorError("frozen backend source bytes differ")
    return BF16BackendIdentity.from_dict(
        {
            "schema_version": BF16BackendIdentity.SCHEMA_VERSION,
            "backend_id": "torch_sdpa_flash_gqa",
            "torch_version": "2.12.1+cu130",
            "torch_git_sha": "7269437d655783a26cba32aa88195b741ff496aa",
            "cuda_runtime_version": "13.0",
            "cudnn_version": "9.20.0",
            "triton_version": "3.7.1",
            "flash_generation": "FA2",
            "flash_version": "2.5.7",
            "dispatch_api": "torch.nn.functional.scaled_dot_product_attention",
            "selected_backend": "flash_attention",
            "enable_gqa": True,
            "compile_mode": "disabled",
            "source_artifacts": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(expected_sources.items())
            ],
        }
    )


def _worker_argv(plan_path: str, point: Any, run_id: str) -> tuple[str, ...]:
    return (
        PHASE3_PYTHON_EXECUTABLE,
        "-m",
        "kvbench",
        "phase3-worker",
        "--plan",
        plan_path,
        "--point-id",
        point.point_id,
        "--replicate",
        str(point.process_replicate),
        "--run-id",
        run_id,
    )


def _initial_manifest(
    *,
    bundle: ExperimentBundle,
    plan_path: str,
    point: Any,
    run_id: str,
    created_at: str,
    git_sha: str,
    environment: Mapping[str, str],
    backend: BF16BackendIdentity,
    cache: BF16CacheIdentity,
) -> Phase3RunManifest:
    method_fingerprint = MethodConfigFingerprint.from_config(
        bundle.methods[0],
        "bf16",
    )
    command = Phase3CommandSpec(
        schema_version=Phase3CommandSpec.SCHEMA_VERSION,
        argv=_worker_argv(plan_path, point, run_id),
        working_directory=PHASE3_REPOSITORY_ROOT,
        environment_sha256=sha256_hex(canonical_json_bytes(dict(environment))),
        dry_run=False,
    )
    payload = {
        "schema_version": Phase3RunManifest.SCHEMA_VERSION,
        "artifact_schema_version": Phase3RunManifest.ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "created",
        "created_at_utc": created_at,
        "started_at_utc": None,
        "finished_at_utc": None,
        "run_kind": RunKind.PHASE3_ADMISSION.value,
        "runner_kind": point.runner_kind.value,
        "graph_mode": point.graph_mode.value,
        "claim_class": ClaimClass.NONE.value,
        "measurement_scope": MeasurementScope.NATIVE_HOST_ADMISSION.value,
        "performance_claim_eligible": False,
        "plan_source": {
            "kind": ConfigSourceKind.PATH.value,
            "path": plan_path,
            "canonical_inline_json": None,
            "sha256": bundle.plan.fingerprint(),
        },
        "plan_fingerprint": bundle.plan.fingerprint(),
        "point_id": point.point_id,
        "point_fingerprint": derive_phase3_point_fingerprint(point.point_id),
        "git_sha": git_sha,
        "git_dirty": False,
        "container_digest": None,
        "hardware_id": PHASE3_HARDWARE_ID,
        "hardware_fingerprint": PHASE3_HARDWARE_FINGERPRINT,
        "native_g0_status": GateDisposition.PASS.value,
        "e00_run_id": PHASE3_E00_RUN_ID,
        "e00_manifest_sha256": PHASE3_E00_MANIFEST_SHA256,
        "blocker_b010": "OPEN",
        "gpu_uuid": PHASE3_GPU_UUID,
        "gpu_full_name": PHASE3_GPU_FULL_NAME,
        "pci_bus_id": PHASE3_PCI_BUS_ID,
        "pci_device_id": PHASE3_PCI_DEVICE_ID,
        "driver_version": PHASE3_DRIVER_VERSION,
        "software_environment_id": PHASE3_SOFTWARE_ENVIRONMENT_ID,
        "software_fingerprint": PHASE3_SOFTWARE_FINGERPRINT,
        "model_identity": bundle.model.to_dict(),
        "model_fingerprint": bundle.model.fingerprint(),
        "method": "bf16",
        "method_config_id": "bf16",
        "method_config_fingerprint": method_fingerprint.to_dict(),
        "contract_fingerprint": PHASE3_CONTRACT_FINGERPRINT,
        "measurement_protocol_fingerprint": (
            PHASE3_MEASUREMENT_PROTOCOL_FINGERPRINT
        ),
        "backend_identity": backend.to_dict(),
        "backend_fingerprint": backend.fingerprint(),
        "cache_identity": cache.to_dict(),
        "batch_size": point.batch_size,
        "context_length": point.context_length,
        "output_steps": point.output_steps,
        "warmup_count": bundle.plan.measurement.warmup_count,
        "measured_count": bundle.plan.measurement.measured_count,
        "measured_batches": bundle.plan.measurement.measured_batches,
        "count_unit": bundle.plan.measurement.count_unit.value,
        "random_seed": bundle.plan.measurement.seed,
        "process_replicate": point.process_replicate,
        "quality": bundle.plan.quality.to_dict(),
        "command": command.to_dict(),
        "inventory_path": None,
        "failure_reason": None,
    }
    return Phase3RunManifest.from_dict(payload)


def _failed_result(
    *,
    bundle: ExperimentBundle,
    point: Any,
    run_id: str,
    reason: str,
) -> Phase3WorkerResult:
    expected = (
        bundle.plan.measurement.measured_count
        * bundle.plan.measurement.measured_batches
        if point.runner_kind.value == "fixed_l"
        else point.output_steps
        * bundle.plan.measurement.measured_count
        * bundle.plan.measurement.measured_batches
    )
    return Phase3WorkerResult(
        schema_version=Phase3WorkerResult.SCHEMA_VERSION,
        run_id=run_id,
        point_id=point.point_id,
        runner_kind=point.runner_kind,
        count_unit=bundle.plan.measurement.count_unit,
        status=RunStatus.ABORTED,
        expected_operations=expected,
        completed_operations=0,
        failed_operations=0,
        output_checksum=None,
        failure_reason=reason[:1000],
    )


def _write_runtime_artifacts(run: Any, evidence: Mapping[str, Any]) -> None:
    run.write_json("raw/worker_evidence.json", evidence)
    runtime = evidence.get("runtime")
    numerical = evidence.get("numerical")
    if isinstance(numerical, Mapping):
        run.write_json("numerical/agreement.json", numerical)
    if not isinstance(runtime, Mapping):
        return
    timing = runtime.get("timing")
    if isinstance(timing, Mapping):
        run.write_json(
            "raw/timing.json",
            {
                **timing,
                "quality_status": "unvalidated",
                "claim_eligibility": "performance_only",
                "performance_claim_eligible": False,
                "measurement_scope": "native_host_admission",
                "profiler_instrumented": False,
            },
        )
    run.write_json(
        "allocation/audit.json",
        {
            "allocation": runtime.get("allocation"),
            "memory_evidence": runtime.get("memory_evidence"),
            "cache_accounting": runtime.get("cache_accounting"),
            "instrumented_duration_reported_as_timing": False,
        },
    )
    run.write_json(
        "gqa/audit.json",
        {
            "source": runtime.get("gqa_source"),
            "cache_geometry": runtime.get("gqa_cache_geometry"),
            "operator": runtime.get("gqa_operator"),
            "operators": runtime.get("gqa_operators"),
            "mha_control": runtime.get("mha_control"),
            "prefill_backend": runtime.get("prefill_backend"),
            "decode_backend": runtime.get("backend"),
        },
    )
    run.write_json(
        "telemetry/snapshots.json",
        {
            "before": runtime.get("telemetry_before"),
            "after": runtime.get("telemetry_after"),
            "sampling_interval_seconds": runtime.get(
                "telemetry_sampling_interval_seconds"
            ),
            "queried_inside_decode_hot_path": False,
            "stability_inference": False,
        },
    )


def _terminal_manifest(
    initial: Phase3RunManifest,
    *,
    started_at: str,
    status: RunStatus,
    failure_reason: str | None,
) -> Phase3RunManifest:
    payload = initial.to_dict()
    payload.update(
        {
            "status": status.value,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "inventory_path": "artifact_inventory.json",
            "failure_reason": None if status is RunStatus.COMPLETED else failure_reason,
        }
    )
    return Phase3RunManifest.from_dict(payload)


def _run_point(
    *,
    bundle: ExperimentBundle,
    plan_path: str,
    point: Any,
    run_id: str,
    git_sha: str,
    backend: BF16BackendIdentity,
    live_hardware: Mapping[str, Any],
) -> dict[str, Any]:
    store = phase3_artifact_store(REPOSITORY_ROOT)
    created_at = _utc_now()
    with tempfile.TemporaryDirectory(prefix=f"kvbench-{run_id}-", dir="/tmp") as raw_temp:
        temp_root = Path(raw_temp).resolve(strict=True)
        environment = _worker_environment(temp_root)
        cache = _cache_identity(point)
        initial = _initial_manifest(
            bundle=bundle,
            plan_path=plan_path,
            point=point,
            run_id=run_id,
            created_at=created_at,
            git_sha=git_sha,
            environment=environment,
            backend=backend,
            cache=cache,
        )
        run = store.create(run_id, initial)
        run.start()
        started_at = _utc_now()
        run.write_json("config/plan.json", bundle.plan.to_dict())
        run.write_json(
            "config/referenced_fingerprints.json",
            {
                "schema_version": "kvbench-phase3-references-1.0.0",
                "fingerprints": [
                    {"path": path, "sha256": digest}
                    for path, digest in bundle.canonical_fingerprints
                ],
                "formal_blockers_retained": list(bundle.all_blockers),
            },
        )
        run.write_json(
            "validation/point.json",
            {
                **point.to_dict(),
                "point_fingerprint": derive_phase3_point_fingerprint(
                    point.point_id
                ),
            },
        )
        run.write_json("environment/worker_environment.json", environment)
        run.write_json("environment/live_hardware.json", live_hardware)
        process_snapshots: dict[str, Any] = {}
        result: Phase3WorkerResult | None = None
        evidence: dict[str, Any] | None = None
        process: subprocess.Popen[bytes] | None = None
        stdout_path = temp_root / "worker.stdout"
        stderr_path = temp_root / "worker.stderr"
        failure_reason: str | None = None
        audit_passed = False
        try:
            before = _process_snapshot()
            process_snapshots["before"] = before
            if not _snapshot_clean(before, allow_supervised=False):
                raise Phase3CoordinatorError("foreign or unknown compute before worker")
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                process = subprocess.Popen(
                    _worker_argv(plan_path, point, run_id),
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
                ready = _wait_for_ready(
                    process,
                    Path(environment["KVBENCH_PHASE3_AUDIT_READY"]),
                )
                process_snapshots["ready"] = ready
                pid = int(ready["pid"])
                start_ticks = int(ready["process_start_time_ticks"])
                release_audit = _process_snapshot(pid=pid, start_ticks=start_ticks)
                process_snapshots["release_audit"] = release_audit
                if not _snapshot_clean(release_audit, allow_supervised=True):
                    raise Phase3CoordinatorError("worker release audit failed closed")
                _exclusive_release(Path(environment["KVBENCH_PHASE3_AUDIT_RELEASE"]))
                during_samples: list[dict[str, Any]] = []
                saw_allowed_compute = False
                worker_deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
                while process.poll() is None:
                    if time.monotonic() >= worker_deadline:
                        raise Phase3CoordinatorError("worker execution timed out")
                    candidate = _process_snapshot(pid=pid, start_ticks=start_ticks)
                    if not _snapshot_clean(candidate, allow_supervised=True):
                        if (
                            process.poll() is not None
                            and candidate.get("foreign_compute_processes") == []
                            and candidate.get("unknown_processes") == []
                        ):
                            break
                        process_snapshots["during"] = {
                            "schema_version": "kvbench-phase3-process-monitor-1.0.0",
                            "sampling_target_seconds": 2.0,
                            "samples": [*during_samples, candidate],
                        }
                        raise Phase3CoordinatorError("worker process audit detected foreign compute")
                    if candidate.get("allowed_compute_processes"):
                        saw_allowed_compute = True
                    during_samples.append(candidate)
                    time.sleep(2.0)
                if not during_samples or not saw_allowed_compute:
                    raise Phase3CoordinatorError(
                        "worker process monitoring did not observe clean compute"
                    )
                process_snapshots["during"] = {
                    "schema_version": "kvbench-phase3-process-monitor-1.0.0",
                    "sampling_target_seconds": 2.0,
                    "samples": during_samples,
                    "saw_allowed_compute": saw_allowed_compute,
                    "monitoring_stopped_before_worker_exit": False,
                }
                process.wait(timeout=WORKER_TIMEOUT_SECONDS)
            after = _process_snapshot()
            process_snapshots["after"] = after
            if not _snapshot_clean(after, allow_supervised=False):
                raise Phase3CoordinatorError("compute process leaked after worker")
            stdout = stdout_path.read_bytes()
            parsed_result = _parse_canonical_json(
                stdout,
                maximum_bytes=MAX_STDOUT_BYTES,
                label="worker stdout",
            )
            result = Phase3WorkerResult.from_dict(parsed_result)
            if (
                result.run_id != run_id
                or result.point_id != point.point_id
                or result.runner_kind is not point.runner_kind
                or result.count_unit is not bundle.plan.measurement.count_unit
            ):
                raise Phase3CoordinatorError("worker result differs from requested point")
            if process.returncode != 0:
                raise Phase3CoordinatorError("worker returned nonzero with a result")
            ipc_path = Path(environment["KVBENCH_PHASE3_IPC_PATH"])
            ipc_metadata = ipc_path.lstat()
            if stat.S_ISLNK(ipc_metadata.st_mode) or not stat.S_ISREG(ipc_metadata.st_mode):
                raise Phase3CoordinatorError("worker IPC path is unsafe")
            evidence = _parse_canonical_json(
                ipc_path.read_bytes(),
                maximum_bytes=MAX_IPC_BYTES,
                label="worker evidence",
            )
            if (
                evidence.get("run_id") != run_id
                or evidence.get("point_id") != point.point_id
                or evidence.get("worker_result") != result.to_dict()
            ):
                raise Phase3CoordinatorError("worker evidence identity join failed")
            runtime = evidence.get("runtime")
            if isinstance(runtime, Mapping):
                if runtime.get("cache_layout_fingerprint") != cache.layout_fingerprint:
                    raise Phase3CoordinatorError("runtime cache fingerprint differs")
            if result.status is RunStatus.COMPLETED and not saw_allowed_compute:
                raise Phase3CoordinatorError("completed worker lacked active compute audit")
            audit_passed = True
        except BaseException as error:
            failure_reason = f"{type(error).__name__}: {' '.join(str(error).split())}"[:1000]
            if process is not None:
                _terminate_worker(process)
            try:
                if "after" not in process_snapshots:
                    process_snapshots["after"] = _process_snapshot()
            except BaseException as after_error:
                process_snapshots["after_error"] = {
                    "type": type(after_error).__name__,
                    "message": "post-worker process snapshot failed",
                }
            result = _failed_result(
                bundle=bundle,
                point=point,
                run_id=run_id,
                reason=failure_reason,
            )
        stdout = stdout_path.read_bytes() if stdout_path.is_file() else b""
        stderr = stderr_path.read_bytes() if stderr_path.is_file() else b""
        run.write_bytes("logs/worker.stdout.txt", stdout)
        run.write_bytes("logs/worker.stderr.txt", stderr)
        for name, snapshot in process_snapshots.items():
            run.write_json(f"environment/process.{name}.json", snapshot)
        run.write_json(
            "validation/process_audit_outcome.json",
            {
                "schema_version": "kvbench-phase3-process-audit-1.0.0",
                "passed": audit_passed,
                "certified_helper": "preflight/process_query.py",
                "foreign_compute_allowed": False,
                "unknown_compute_allowed": False,
            },
        )
        run.write_json("validation/worker_result.json", result.to_dict())
        if evidence is not None:
            _write_runtime_artifacts(run, evidence)
            model_identity = evidence.get("model_identity")
            if isinstance(model_identity, Mapping):
                run.write_json("validation/model_identity.json", model_identity)
        final_status = result.status if audit_passed else RunStatus.ABORTED
        final_reason = (
            result.failure_reason if audit_passed else failure_reason or "process audit failed"
        )
        final = _terminal_manifest(
            initial,
            started_at=started_at,
            status=final_status,
            failure_reason=final_reason,
        )
        final_path = run.finalize(final)
    validation = validate_run_directory(final_path)
    if not validation.valid or not validation.complete:
        raise Phase3CoordinatorError("final Phase 3 run failed checksum validation")
    return {
        "run_id": run_id,
        "point_id": point.point_id,
        "status": final_status.value,
        "run_dir": str(final_path.relative_to(REPOSITORY_ROOT)),
        "checksum_valid": True,
        "timing_collected": bool(
            evidence is not None
            and isinstance(evidence.get("runtime"), Mapping)
            and evidence["runtime"].get("timing") is not None
        ),
    }


def run_phase3_campaign(plan_path: str | Path) -> dict[str, Any]:
    """Run every frozen point once, preserving failures and never retrying."""

    _validate_entry_evidence()
    git_sha, _ = _git_identity()
    live_hardware = _live_hardware()
    bundle = load_phase3_admission_bundle(plan_path)
    if not bundle.execution_ready:
        raise Phase3CoordinatorError("Phase 3 bundle is not narrowly execution-authorized")
    relative_plan = bundle.plan_path.relative_to(REPOSITORY_ROOT).as_posix()
    points = expand_phase3_process_points(bundle.plan)
    backend = _backend_identity_stdlib()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fZ").lower()
    campaign_id = f"phase3-{timestamp}-{git_sha[:8]}-{secrets.token_hex(3)}"
    planned_run_ids = tuple(
        f"{campaign_id}-{point.point_id}" for point in points
    )
    recorder = Phase3CampaignRecorder.create(
        repository_root=REPOSITORY_ROOT,
        campaign_id=campaign_id,
        created_at_utc=_utc_now(),
        git_sha=git_sha,
        plan_path=relative_plan,
        plan_fingerprint=bundle.plan.fingerprint(),
        point_ids=tuple(point.point_id for point in points),
        run_ids=planned_run_ids,
    )
    runs: list[dict[str, Any]] = []
    unexpected_error: BaseException | None = None
    try:
        for point, run_id in zip(points, planned_run_ids):
            runs.append(
                _run_point(
                    bundle=bundle,
                    plan_path=relative_plan,
                    point=point,
                    run_id=run_id,
                    git_sha=git_sha,
                    backend=backend,
                    live_hardware=live_hardware,
                )
            )
    except BaseException as error:
        unexpected_error = error
    counts = Counter(item["status"] for item in runs)
    result = {
        "schema_version": "kvbench-phase3-campaign-result-1.0.0",
        "ok": unexpected_error is None
        and len(runs) == len(points)
        and all(item["status"] == RunStatus.COMPLETED.value for item in runs),
        "campaign_id": campaign_id,
        "git_sha": git_sha,
        "plan": relative_plan,
        "plan_fingerprint": bundle.plan.fingerprint(),
        "expected_process_count": bundle.plan.expected_process_count,
        "attempted_process_count": len(runs),
        "unattempted_point_ids": [
            point.point_id for point in points[len(runs) :]
        ],
        "status_counts": dict(sorted(counts.items())),
        "runs": runs,
        "execution_attempted": True,
        "timing_collected": any(item["timing_collected"] for item in runs),
        "profiler_executed": False,
        "quality_executed": False,
        "performance_claim_eligible": False,
        "measurement_scope": "native_host_admission",
        "selective_rerun_performed": False,
        "preregistered_before_execution": True,
        "unexpected_campaign_abort": unexpected_error is not None,
        "unexpected_failure": (
            None
            if unexpected_error is None
            else f"{type(unexpected_error).__name__}: "
            f"{' '.join(str(unexpected_error).split())}"[:1000]
        ),
        "finished_at_utc": _utc_now(),
    }
    campaign_path = recorder.finalize(result)
    result["campaign_record"] = campaign_path.relative_to(
        REPOSITORY_ROOT
    ).as_posix()
    if unexpected_error is not None:
        raise unexpected_error
    return result

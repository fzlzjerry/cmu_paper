"""Append-only continuation for the interrupted Phase 13D campaign.

This coordinator deliberately leaves the original worker, timing, graph,
telemetry, adapters, CUDA sources, candidate table, and execution order
untouched.  It persists every GPU-process query before classification and
executes only the failed logical record plus the frozen suffix.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import statistics
import subprocess
import sys
import time
from typing import Any

from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from kvbench.runtime.artifacts import sha256_file
from kvbench.runtime.process_supervision import run_stage_supervised_command
from scripts.r2_artifact import validate_local_artifact
from scripts import phase12_unified_admission as phase12
from scripts import phase13_pilot as pilot
from scripts import phase13d_knee_densification as phase13d


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "phase13d-20260825t030556684636z-a06837a3-83761a"
ORIGINAL_EXECUTION_HEAD = "a06837a341f58936b21f721af3fe30dce69660d4"
ORIGINAL_STAGE = (
    REPOSITORY_ROOT
    / "artifacts/phase13d/.kvbench-staging/"
    "phase13d-20260825t030556684636z-a06837a3-83761a.8f4bb4ac.staging"
)
PREFIX_ROOT = Path(
    "/home/rockrock/phase13d_prefix_states/"
    "phase13d-20260825t030556684636z-a06837a3-83761a"
)
SEGMENT_A_ID = "segment-a-original"
SEGMENT_A_VALID_RUNS = 51
SEGMENT_A_TREE_SHA256 = (
    "d3ac81494ea7e1e32109350477f2b972f9e963a2a273f611979beaa96e234243"
)
SEGMENT_A_FILE_COUNT = 1_275
SEGMENT_A_BYTES = 740_209_661
ORIGINAL_RUNS_TREE_SHA256 = (
    "eb3c820a202ef9ac503aec62fe53921bf9d4000b9b70083d00ead418743a4d33"
)
ORIGINAL_RUNS_FILE_COUNT = 1_291
ORIGINAL_RUNS_BYTES = 793_498_965
FAILED_SEQUENCE_INDEX = 51
FAILED_RUN_ID = (
    "phase13d-20260825t030556684636z-a06837a3-83761a-"
    "r0-o051-kvq3-b8-l5120"
)
FAILED_RUN_TREE_SHA256 = (
    "cd13c6701d6546e00d07345b963f0ba804fd1d751b1da08a497fd2d503f806c7"
)
FAILED_RUN_FILE_COUNT = 16
FAILED_RUN_BYTES = 53_289_304
PREFIX_CATALOG_SHA256 = (
    "8646f3fe60edfea5472676113549bb01917c9d785c2c7527141985696599848a"
)
PREFIX_STATE_INDEX_SHA256 = (
    "997ceba83e17a8710bdcdb6288b4d40575242b76b47328d5b1492acc2d393b85"
)
PREFIX_STATE_COUNT = 84
SEGMENT_B_RECORDS = 201
TOTAL_LOGICAL_RECORDS = 252
SNAPSHOT_STATES = frozenset(
    {"clean", "foreign_process_detected", "query_failed"}
)
SNAPSHOT_RETRIES = 2
SNAPSHOT_RETRY_DELAY_SECONDS = 1.0
SNAPSHOT_TIMEOUT_SECONDS = 30
SNAPSHOT_COMMAND = (
    sys.executable,
    str(REPOSITORY_ROOT / "preflight/process_query.py"),
)
SEGMENT_SCHEMA = "kvbench-phase13d-continuation-segment-1.0.0"
RUN_SCHEMA = "kvbench-phase13d-continuation-run-1.0.0"
SNAPSHOT_RAW_SCHEMA = "kvbench-gpu-process-snapshot-raw-1.0.0"
SNAPSHOT_CLASSIFICATION_SCHEMA = (
    "kvbench-gpu-process-snapshot-classification-1.0.0"
)
_SEGMENT_RE = re.compile(
    r"phase13dseg-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)


TIMING_CRITICAL_TRACKED_PATHS = (
    "configs/models/primary_gqa_model.yaml",
    "configs/methods/bf16.yaml",
    "configs/methods/turboquant.yaml",
    "configs/methods/kivi.yaml",
    "configs/methods/kvquant.yaml",
    "configs/plans/pilot.yaml",
    "scripts/phase12_unified_admission.py",
    "scripts/phase13_pilot.py",
    "scripts/phase13d_knee_densification.py",
    "src/kvbench/adapters/base.py",
    "src/kvbench/adapters/factory.py",
    "src/kvbench/adapters/bf16.py",
    "src/kvbench/adapters/turboquant.py",
    "src/kvbench/adapters/kivi.py",
    "src/kvbench/adapters/kvquant.py",
    "src/kvbench/runtime/allocation.py",
    "src/kvbench/runtime/allocation_attribution.py",
    "src/kvbench/runtime/backend.py",
    "src/kvbench/runtime/bf16_endpoint.py",
    "src/kvbench/runtime/cuda_graph.py",
    "src/kvbench/runtime/fixed_l_runner.py",
    "src/kvbench/runtime/gqa_audit.py",
    "src/kvbench/runtime/gqa_device_dispatch.py",
    "src/kvbench/runtime/gqa_taxonomy.py",
    "src/kvbench/runtime/kivi_cache.py",
    "src/kvbench/runtime/kivi_session.py",
    "src/kvbench/runtime/kvquant_cache.py",
    "src/kvbench/runtime/kvquant_session.py",
    "src/kvbench/runtime/method_harness.py",
    "src/kvbench/runtime/model_loader.py",
    "src/kvbench/runtime/numerical.py",
    "src/kvbench/runtime/static_cache.py",
    "src/kvbench/runtime/telemetry.py",
    "src/kvbench/runtime/timing.py",
    "src/kvbench/runtime/turboquant_audit.py",
    "src/kvbench/runtime/turboquant_cache.py",
    "src/kvbench/runtime/turboquant_session.py",
    "src/kvbench/third_party/vllm_turboquant/centroids.py",
    "src/kvbench/third_party/vllm_turboquant/compat.py",
    "src/kvbench/third_party/vllm_turboquant/config.py",
    "src/kvbench/third_party/vllm_turboquant/provenance.json",
    "src/kvbench/third_party/vllm_turboquant/triton_decode_attention.py",
    "src/kvbench/third_party/vllm_turboquant/triton_turboquant_decode.py",
    "src/kvbench/third_party/vllm_turboquant/triton_turboquant_store.py",
)

EXTERNAL_EXECUTION_HASHES = {
    "kvquant:deployment/kvquant/setup_cuda.py": (
        "27cdeb876e0c6276c1538a68eb83991efe89ddd4c4606be3339f0d68a764c68b"
    ),
    "kvquant:deployment/kvquant/quant_cuda.cpp": (
        "854831d282626fa9d695f4f017ca2537c48a09072f9f0dee4f38573caed4437c"
    ),
    "kvquant:deployment/kvquant/quant_cuda_kernel.cu": (
        "457a1d8d7bd07ba2e7e8420cbe3b722b52471f131ecc88f7f5442584a8b212f9"
    ),
    "kvquant:deployment/kvquant/measurement_cuda_kernel.cu": (
        "0b9a840820038e72487987689433d41b29c332e21a146bfd02bc089cd5ea9137"
    ),
    "kivi:quant/csrc/gemv_cuda.cu": (
        "850b9a33b1e320b3d77dab61a423cea6c73473ab7f33b064dd2e34a682dfef63"
    ),
    "kivi:quant/csrc/gemv_cuda_backup.cu": (
        "bf90b7ee23d54cad470ce443185400b81ca7319eff9f75db6a2fdf80d3fe24ae"
    ),
    "kivi:quant/csrc/pybind.cpp": (
        "75e7837bcd9a6aac8156ceba986a6efd9b64d0d88a60da0a799723b8ae1bfaff"
    ),
    "kivi_extension": (
        "45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9"
    ),
    "kvquant_extension": (
        "b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d"
    ),
}


class Phase13DContinuationError(RuntimeError):
    """The append-only continuation contract failed closed."""


@dataclass(frozen=True)
class SnapshotOutcome:
    state: str
    attempt: int
    parsed_payload: dict[str, Any] | None
    raw_path: str
    classification_path: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13DContinuationError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise Phase13DContinuationError(f"JSON evidence is not an object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_durable_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    write_exclusive(path, json_bytes(dict(payload)))
    _fsync_directory(path.parent)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _tree_evidence(paths: Sequence[Path], *, base: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(paths, key=lambda value: value.relative_to(base).as_posix()):
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase13DContinuationError("immutable evidence contains unsafe file")
        relative = path.relative_to(base).as_posix()
        file_digest = sha256_file(path)
        digest.update(f"{file_digest}  {relative}\n".encode("utf-8"))
        count += 1
        size += metadata.st_size
    return {"file_count": count, "size_bytes": size, "tree_sha256": digest.hexdigest()}


def _segment_a_paths(stage: Path) -> list[Path]:
    failed_root = stage / "runs" / FAILED_RUN_ID
    return [
        path
        for path in (stage / "runs").glob("*/*")
        if path.is_file() and failed_root not in path.parents
    ] + [
        path
        for path in (stage / "runs").glob("*/stage-progress/*")
        if path.is_file() and failed_root not in path.parents
    ]


def _validate_segment_a(stage: Path) -> dict[str, Any]:
    results = sorted((stage / "runs").glob("*/result.json"))
    if len(results) != SEGMENT_A_VALID_RUNS:
        raise Phase13DContinuationError("Segment A finalized-run count differs")
    for result_path in results:
        root = result_path.parent
        result = _strict_json(result_path)
        manifest = _strict_json(root / "manifest.json")
        binding = _strict_json(root / "result-binding.json")
        if (
            result.get("run_id") != root.name
            or result.get("execution_git_sha") != ORIGINAL_EXECUTION_HEAD
            or manifest.get("run_id") != root.name
            or manifest.get("status") != "completed"
            or binding.get("run_id") != root.name
            or binding.get("result_sha256") != sha256_file(result_path)
            or result.get("finite_output") is not True
            or result.get("gpu_exclusive") is not True
            or result.get("no_backend_fallback") is not True
            or result.get("allocation_stable") is not True
            or result.get("kernel_path_stable") is not True
            or result.get("graph_replay_allocation", {}).get("passed") is not True
            or result.get("graph_replay_allocation", {}).get(
                "allocation_event_count"
            )
            != 0
        ):
            raise Phase13DContinuationError("Segment A finalized run differs")
    evidence = _tree_evidence(_segment_a_paths(stage), base=stage)
    if evidence != {
        "file_count": SEGMENT_A_FILE_COUNT,
        "size_bytes": SEGMENT_A_BYTES,
        "tree_sha256": SEGMENT_A_TREE_SHA256,
    }:
        raise Phase13DContinuationError("Segment A immutable tree differs")
    all_runs = _tree_evidence(
        [path for path in (stage / "runs").rglob("*") if path.is_file()],
        base=stage,
    )
    if all_runs != {
        "file_count": ORIGINAL_RUNS_FILE_COUNT,
        "size_bytes": ORIGINAL_RUNS_BYTES,
        "tree_sha256": ORIGINAL_RUNS_TREE_SHA256,
    }:
        raise Phase13DContinuationError("original run tree differs")
    return evidence


def _validate_failed_run(stage: Path) -> dict[str, Any]:
    root = stage / "runs" / FAILED_RUN_ID
    paths = [path for path in root.rglob("*") if path.is_file()]
    evidence = _tree_evidence(paths, base=stage)
    if evidence != {
        "file_count": FAILED_RUN_FILE_COUNT,
        "size_bytes": FAILED_RUN_BYTES,
        "tree_sha256": FAILED_RUN_TREE_SHA256,
    }:
        raise Phase13DContinuationError("original failed-run evidence differs")
    if (
        (root / "result.json").exists()
        or (root / "manifest.json").exists()
        or not (root / "stage-progress/10-measurement-completed.json").is_file()
        or not (root / "stage-progress/11-finalization-started.json").is_file()
    ):
        raise Phase13DContinuationError("original failed-run lifecycle differs")
    return evidence


def _prefix_index_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            (
                "\t".join(
                    str(value)
                    for value in (
                        entry["snapshot_id"],
                        entry["method_config_id"],
                        entry["batch_size"],
                        entry["historical_context"],
                        entry["method_config_fingerprint"],
                        entry["state_file_bytes"],
                        entry["state_file_sha256"],
                    )
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def validate_prefix_reuse(prefix_root: Path, *, verify_state_bytes: bool) -> dict[str, Any]:
    catalog_path = prefix_root / "catalog.json"
    if sha256_file(catalog_path) != PREFIX_CATALOG_SHA256:
        raise Phase13DContinuationError("prefix catalog checksum differs")
    catalog = _strict_json(catalog_path)
    entries = catalog.get("entries")
    if not isinstance(entries, list) or len(entries) != PREFIX_STATE_COUNT:
        raise Phase13DContinuationError("prefix catalog cardinality differs")
    if _prefix_index_digest(entries) != PREFIX_STATE_INDEX_SHA256:
        raise Phase13DContinuationError("prefix state index differs")
    total_bytes = 0
    for entry in entries:
        root = prefix_root / str(entry["snapshot_relative_path"])
        state = root / "state.safetensors"
        manifest = _strict_json(root / "manifest.json")
        expected = str(entry["state_file_sha256"])
        try:
            complete = (root / "COMPLETE").read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as error:
            raise Phase13DContinuationError("prefix COMPLETE is invalid") from error
        if (
            root.is_symlink()
            or not root.is_dir()
            or not state.is_file()
            or state.is_symlink()
            or manifest.get("state_file_sha256") != expected
            or manifest.get("configuration") != entry["method_config_id"]
            or manifest.get("snapshot_key", {}).get("batch_size")
            != entry["batch_size"]
            or manifest.get("historical_context") != entry["historical_context"]
            or manifest.get("batch_reuse_policy") != "exact_target_batch_only"
            or manifest.get("runtime_prefix_sharing") is not False
            or complete != expected
            or state.stat().st_size != entry["state_file_bytes"]
            or state.stat().st_mode & 0o222
        ):
            raise Phase13DContinuationError("prefix state authority differs")
        if verify_state_bytes and sha256_file(state) != expected:
            raise Phase13DContinuationError("prefix state bytes differ")
        total_bytes += state.stat().st_size
    return {
        "status": "PASS",
        "prefix_root": str(prefix_root),
        "catalog_sha256": PREFIX_CATALOG_SHA256,
        "state_index_sha256": PREFIX_STATE_INDEX_SHA256,
        "state_count": len(entries),
        "state_bytes": total_bytes,
        "state_bytes_verified": verify_state_bytes,
        "read_only": True,
        "regenerated": False,
    }


def _order() -> dict[str, Any]:
    order = _strict_json(phase13d.ORDER_PATH)
    staged = _strict_json(ORIGINAL_STAGE / "execution_order.json")
    if order != staged or len(order.get("records", [])) != TOTAL_LOGICAL_RECORDS:
        raise Phase13DContinuationError("frozen execution order differs")
    return order


def continuation_records(order: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = order.get("records")
    if not isinstance(records, list) or len(records) != TOTAL_LOGICAL_RECORDS:
        raise Phase13DContinuationError("frozen record cardinality differs")
    selected = []
    for sequence_index, raw in enumerate(records[FAILED_SEQUENCE_INDEX:], start=FAILED_SEQUENCE_INDEX):
        record = dict(raw)
        record["original_sequence_index"] = sequence_index
        record["original_logical_run_id"] = phase13d._run_id(CAMPAIGN_ID, record)
        record["replacement_for_failed_finalization"] = (
            sequence_index == FAILED_SEQUENCE_INDEX
        )
        selected.append(record)
    first = selected[0]
    if (
        len(selected) != SEGMENT_B_RECORDS
        or first["original_logical_run_id"] != FAILED_RUN_ID
        or first["method_config_id"] != "kvq3"
        or first["batch_size"] != 8
        or first["historical_context"] != 5120
        or first["replicate_index"] != 0
        or first["seed"] != 20260823
    ):
        raise Phase13DContinuationError("replacement logical record differs")
    return selected


def _git_blob(commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ("/usr/bin/git", "show", f"{commit}:{relative}"),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Phase13DContinuationError(f"Git blob is unavailable: {relative}")
    return result.stdout


def timing_critical_equivalence(execution_head: str) -> dict[str, Any]:
    files = []
    for relative in TIMING_CRITICAL_TRACKED_PATHS:
        before = _git_blob(ORIGINAL_EXECUTION_HEAD, relative)
        after = _git_blob(execution_head, relative)
        before_sha = hashlib.sha256(before).hexdigest()
        after_sha = hashlib.sha256(after).hexdigest()
        files.append(
            {
                "path": relative,
                "original_sha256": before_sha,
                "continuation_sha256": after_sha,
                "size_bytes": len(after),
                "unchanged": before == after,
            }
        )
    external: list[dict[str, Any]] = []
    kvquant_root = Path(os.environ.get("KVBENCH_KVQUANT_SOURCE_ROOT", ""))
    kivi_root = Path(os.environ.get("KVBENCH_KIVI_SOURCE_ROOT", ""))
    extension = Path(os.environ.get("KVBENCH_KVQUANT_EXTENSION", ""))
    kivi_extension = Path(
        "/opt/kvbench/.phase3/site-packages/"
        "kivi_gemv.cpython-312-x86_64-linux-gnu.so"
    )
    external_paths = {
        "kvquant:deployment/kvquant/setup_cuda.py": kvquant_root
        / "deployment/kvquant/setup_cuda.py",
        "kvquant:deployment/kvquant/quant_cuda.cpp": kvquant_root
        / "deployment/kvquant/quant_cuda.cpp",
        "kvquant:deployment/kvquant/quant_cuda_kernel.cu": kvquant_root
        / "deployment/kvquant/quant_cuda_kernel.cu",
        "kvquant:deployment/kvquant/measurement_cuda_kernel.cu": kvquant_root
        / "deployment/kvquant/measurement_cuda_kernel.cu",
        "kivi:quant/csrc/gemv_cuda.cu": kivi_root / "quant/csrc/gemv_cuda.cu",
        "kivi:quant/csrc/gemv_cuda_backup.cu": kivi_root
        / "quant/csrc/gemv_cuda_backup.cu",
        "kivi:quant/csrc/pybind.cpp": kivi_root / "quant/csrc/pybind.cpp",
        "kivi_extension": kivi_extension,
        "kvquant_extension": extension,
    }
    for name, path in external_paths.items():
        if not path.is_file() or path.is_symlink():
            raise Phase13DContinuationError(f"external execution source absent: {name}")
        observed = sha256_file(path)
        expected = EXTERNAL_EXECUTION_HASHES[name]
        external.append(
            {
                "name": name,
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "unchanged": observed == expected,
            }
        )
    order = _order()
    result = {
        "schema_version": "kvbench-phase13d-timing-equivalence-1.0.0",
        "status": "PASS",
        "original_execution_head": ORIGINAL_EXECUTION_HEAD,
        "continuation_execution_head": execution_head,
        "tracked_files": files,
        "external_execution_files": external,
        "authorized_container_digest": phase13d.AUTHORIZED_CONTAINER_DIGEST,
        "method_fingerprints": dict(phase13d.CONFIG_FINGERPRINTS),
        "warmup_steps": phase13d.WARMUP_STEPS,
        "measured_steps": phase13d.MEASURED_STEPS,
        "measured_batches": phase13d.MEASURED_BATCHES,
        "candidate_table_sha256": sha256_file(phase13d.CANDIDATE_PATH),
        "execution_order_sha256": sha256_file(phase13d.ORDER_PATH),
        "records_sha256": order["records_sha256"],
        "seeds": order["seeds"],
        "timing_critical_changed": False,
    }
    if (
        any(item["unchanged"] is not True for item in files)
        or any(item["unchanged"] is not True for item in external)
        or result["warmup_steps"] != 64
        or result["measured_steps"] != 128
        or result["seeds"] != [20260823, 20260824, 20260825]
    ):
        raise Phase13DContinuationError("timing-critical identity changed")
    return result


def _snapshot_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{REPOSITORY_ROOT / 'src'}:{REPOSITORY_ROOT}",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
    }


def _invoke_snapshot_command(command: Sequence[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            tuple(command),
            cwd=REPOSITORY_ROOT,
            env=_snapshot_environment(),
            check=False,
            capture_output=True,
            text=True,
            timeout=SNAPSHOT_TIMEOUT_SECONDS,
        )
        return {
            "return_code": result.returncode,
            "raw_stdout": result.stdout,
            "raw_stderr": result.stderr,
            "invocation_exception": None,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "return_code": None,
            "raw_stdout": error.stdout if isinstance(error.stdout, str) else "",
            "raw_stderr": error.stderr if isinstance(error.stderr, str) else "",
            "invocation_exception": (
                f"TimeoutExpired: snapshot query exceeded {SNAPSHOT_TIMEOUT_SECONDS}s"
            ),
        }
    except OSError as error:
        return {
            "return_code": None,
            "raw_stdout": "",
            "raw_stderr": "",
            "invocation_exception": f"{type(error).__name__}: {error}",
        }


def _classify_snapshot_attempt(
    *, invocation: Mapping[str, Any]
) -> tuple[str, dict[str, Any] | None, str | None, dict[str, Any] | None]:
    parsed: dict[str, Any] | None = None
    parser_exception: str | None = None
    try:
        candidate = json.loads(str(invocation["raw_stdout"]))
        if not isinstance(candidate, dict):
            raise ValueError("JSON root is not an object")
        parsed = candidate
    except (json.JSONDecodeError, ValueError) as error:
        parser_exception = f"{type(error).__name__}: {error}"
    process_list = None
    if parsed is not None:
        process_list = {
            name: parsed.get(name)
            for name in (
                "graphics_processes",
                "allowed_compute_processes",
                "foreign_compute_processes",
                "unknown_processes",
            )
        }
    required_lists_valid = bool(
        parsed is not None
        and all(isinstance(value, list) for value in process_list.values())
        and isinstance(parsed.get("errors"), list)
        and isinstance(parsed.get("subcommands"), list)
        and type(parsed.get("query_exit_code")) is int
    )
    query_failed = bool(
        invocation.get("invocation_exception") is not None
        or invocation.get("return_code") != 0
        or parser_exception is not None
        or not required_lists_valid
    )
    if parsed is not None:
        query_failed = bool(
            query_failed
            or parsed.get("query_exit_code") != 0
            or parsed.get("errors") != []
            or parsed.get("unknown_processes") != []
        )
    if query_failed:
        state = "query_failed"
    elif parsed is not None and parsed.get("foreign_compute_processes") != []:
        state = "foreign_process_detected"
    else:
        state = "clean"
    if state not in SNAPSHOT_STATES:
        raise AssertionError("unknown snapshot state")
    return state, parsed, parser_exception, process_list


def capture_persisted_snapshot(
    *, evidence_root: Path, phase: str, retries: int = SNAPSHOT_RETRIES
) -> SnapshotOutcome:
    if phase not in {"preflight", "postflight"} or retries != SNAPSHOT_RETRIES:
        raise Phase13DContinuationError("snapshot retry contract differs")
    evidence_root.mkdir(parents=True, exist_ok=False)
    for attempt in range(retries + 1):
        command = list(SNAPSHOT_COMMAND)
        invocation = _invoke_snapshot_command(command)
        raw_path = evidence_root / f"attempt-{attempt:02d}.raw.json"
        raw = {
            "schema_version": SNAPSHOT_RAW_SCHEMA,
            "timestamp": _utc_now(),
            "phase": phase,
            "attempt": attempt,
            "command_api": "preflight/process_query.py",
            "command": command,
            "return_code": invocation["return_code"],
            "raw_stdout": invocation["raw_stdout"],
            "raw_stderr": invocation["raw_stderr"],
            "invocation_exception": invocation["invocation_exception"],
        }
        # This immutable raw record is durably written before JSON parsing or
        # process classification.
        _write_durable_exclusive(raw_path, raw)
        state, parsed, parser_exception, process_list = _classify_snapshot_attempt(
            invocation=invocation
        )
        classification_path = evidence_root / f"attempt-{attempt:02d}.classification.json"
        classification = {
            **raw,
            "schema_version": SNAPSHOT_CLASSIFICATION_SCHEMA,
            "raw_record_path": raw_path.name,
            "parser_result": parsed,
            "parser_exception": parser_exception,
            "process_list": process_list,
            "state": state,
        }
        _write_durable_exclusive(classification_path, classification)
        if state != "query_failed" or attempt == retries:
            return SnapshotOutcome(
                state=state,
                attempt=attempt,
                parsed_payload=parsed,
                raw_path=raw_path.relative_to(evidence_root.parent.parent).as_posix(),
                classification_path=classification_path.relative_to(
                    evidence_root.parent.parent
                ).as_posix(),
            )
        time.sleep(SNAPSHOT_RETRY_DELAY_SECONDS)
    raise AssertionError("snapshot retry loop is unreachable")


def snapshot_policy(*, phase: str, state: str) -> str:
    if state not in SNAPSHOT_STATES:
        raise Phase13DContinuationError("snapshot state differs")
    if phase == "preflight":
        return {
            "clean": "start_run",
            "foreign_process_detected": "abort_campaign",
            "query_failed": "fail_current_before_measurement",
        }[state]
    if phase == "postflight":
        return {
            "clean": "finalize_run",
            "foreign_process_detected": "invalidate_current_and_abort",
            "query_failed": "exclude_current_and_continue",
        }[state]
    raise Phase13DContinuationError("snapshot phase differs")


def _segment_run_id(segment_id: str, record: Mapping[str, Any]) -> str:
    value = (
        f"{segment_id}-g{int(record['original_sequence_index']):03d}-"
        f"r{record['replicate_index']}-o{int(record['order_index']):03d}-"
        f"{record['method_config_id']}-b{record['batch_size']}-"
        f"l{record['context_label']}"
    )
    if phase13d._RUN_RE.fullmatch(value) is None:
        raise Phase13DContinuationError("continuation run ID is invalid")
    return value


def _run_manifest(
    *,
    root: Path,
    segment_id: str,
    execution_head: str,
    run_id: str,
    record: Mapping[str, Any],
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "kvbench-phase13d-continuation-run-manifest-1.0.0",
        "campaign_id": CAMPAIGN_ID,
        "source_segment_id": segment_id,
        "execution_head": execution_head,
        "run_id": run_id,
        "original_logical_run_id": record["original_logical_run_id"],
        "original_sequence_index": record["original_sequence_index"],
        "replacement_for_failed_finalization": record[
            "replacement_for_failed_finalization"
        ],
        "status": status,
        "reason": reason,
        "target_id": record["target_id"],
        "method_config_id": record["method_config_id"],
        "method_config_fingerprint": record["method_config_fingerprint"],
        "replicate_index": record["replicate_index"],
        "seed": record["seed"],
        "order_index": record["order_index"],
        "batch_size": record["batch_size"],
        "context_label": record["context_label"],
        "historical_context": record["historical_context"],
        "total_attended_context": record["total_attended_context"],
        "runner_kind": "fixed_l",
        "graph_mode": "cuda_graph",
        "warmup_steps": 64,
        "measured_steps": 128,
        "measured_batches": phase13d.MEASURED_BATCHES,
        "authorized_container_digest": phase13d.AUTHORIZED_CONTAINER_DIGEST,
        "selective_rerun": False,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    _write_durable_exclusive(root / "manifest.json", manifest)
    return manifest


def _record_disposition(
    *, root: Path, run_id: str, status: str, reason: str, launched: bool
) -> None:
    _write_durable_exclusive(
        root / "disposition.json",
        {
            "schema_version": "kvbench-phase13d-continuation-disposition-1.0.0",
            "run_id": run_id,
            "status": status,
            "reason": reason,
            "cuda_process_launched": launched,
            "record_preserved": True,
        },
    )


def _write_worker_evidence(
    *, root: Path, result: Any, pre: SnapshotOutcome, post: SnapshotOutcome
) -> None:
    write_exclusive(root / "worker.stdout.txt", result.stdout)
    write_exclusive(root / "worker.stderr.txt", result.stderr)
    _write_durable_exclusive(root / "worker.supervision.json", result.to_dict())
    _write_durable_exclusive(
        root / "worker.snapshot-summary.json",
        {
            "schema_version": "kvbench-phase13d-snapshot-summary-1.0.0",
            "preflight_state": pre.state,
            "preflight_attempt": pre.attempt,
            "preflight_classification_path": pre.classification_path,
            "postflight_state": post.state,
            "postflight_attempt": post.attempt,
            "postflight_classification_path": post.classification_path,
        },
    )
    if pre.parsed_payload is not None:
        _write_durable_exclusive(root / "worker.gpu-before.json", pre.parsed_payload)
    if post.parsed_payload is not None:
        _write_durable_exclusive(root / "worker.gpu-after.json", post.parsed_payload)


def _extract_worker_payload(result: Any, *, run_id: str) -> dict[str, Any]:
    matches = [
        line[len(phase13d.WORKER_PREFIX) :]
        for line in result.stdout.decode("utf-8", errors="strict").splitlines()
        if line.startswith(phase13d.WORKER_PREFIX)
    ]
    if len(matches) != 1:
        raise Phase13DContinuationError("worker result channel differs")
    payload = json.loads(matches[0])
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise Phase13DContinuationError("worker result identity differs")
    return payload


def _run_one(
    *,
    segment_root: Path,
    segment_id: str,
    execution_head: str,
    record: Mapping[str, Any],
    prefix_entry: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    run_id = _segment_run_id(segment_id, record)
    run_root = segment_root / "runs" / run_id
    run_root.mkdir()
    (run_root / "stage-progress").mkdir()
    _write_durable_exclusive(
        run_root / "started.json",
        {
            "schema_version": "kvbench-phase13d-continuation-run-start-1.0.0",
            "run_id": run_id,
            "source_segment_id": segment_id,
            "execution_head": execution_head,
            "started_at_utc": _utc_now(),
            "record": dict(record),
        },
    )
    pre = capture_persisted_snapshot(
        evidence_root=run_root / "snapshots/preflight", phase="preflight"
    )
    action = snapshot_policy(phase="preflight", state=pre.state)
    if action != "start_run":
        status = (
            "foreign_process_detected"
            if pre.state == "foreign_process_detected"
            else "preflight_snapshot_unavailable"
        )
        reason = f"preflight_snapshot:{pre.state}"
        _record_disposition(
            root=run_root, run_id=run_id, status=status, reason=reason, launched=False
        )
        manifest = _run_manifest(
            root=run_root,
            segment_id=segment_id,
            execution_head=execution_head,
            run_id=run_id,
            record=record,
            status=status,
            reason=reason,
        )
        return manifest, action == "abort_campaign"
    key = (
        str(record["method_config_id"]),
        int(record["batch_size"]),
        int(record["historical_context"]),
    )
    if key != (
        prefix_entry["method_config_id"],
        prefix_entry["batch_size"],
        prefix_entry["historical_context"],
    ):
        raise Phase13DContinuationError("prefix geometry differs")
    command = (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/phase13d_knee_densification.py"),
        "--run-worker",
        "--run-id",
        run_id,
        "--configuration",
        str(record["method_config_id"]),
        "--batch-size",
        str(record["batch_size"]),
        "--context-label",
        str(record["context_label"]),
        "--replicate-index",
        str(record["replicate_index"]),
        "--order-index",
        str(record["order_index"]),
        "--git-sha",
        execution_head,
        "--run-artifact-root",
        str(run_root),
        "--prefix-state-root",
        str(prefix_entry["snapshot_root"]),
        "--prefix-state-sha256",
        str(prefix_entry["state_file_sha256"]),
    )
    timeouts = pilot.stage_timeout_contract(
        batch=int(record["batch_size"]),
        historical=int(record["historical_context"]),
    )
    result = run_stage_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=phase12._child_environment(),
        stage_timeouts=timeouts,
        stage_observer=lambda: pilot._read_stage_observations(
            root=run_root / "stage-progress", run_id=run_id
        ),
        startup_stage="startup",
        transition_stage="transition",
        observer_poll_seconds=pilot.STAGE_OBSERVER_POLL_SECONDS,
    )
    post = capture_persisted_snapshot(
        evidence_root=run_root / "snapshots/postflight", phase="postflight"
    )
    _write_worker_evidence(root=run_root, result=result, pre=pre, post=post)
    if not pilot._supervision_passed(result):
        reason = (
            f"supervisor_stage_timeout:{result.timeout_stage}"
            if result.timeout_stage is not None
            else "supervised_worker_failed"
        )
        _record_disposition(
            root=run_root,
            run_id=run_id,
            status="runtime_failed",
            reason=reason,
            launched=True,
        )
        manifest = _run_manifest(
            root=run_root,
            segment_id=segment_id,
            execution_head=execution_head,
            run_id=run_id,
            record=record,
            status="runtime_failed",
            reason=reason,
        )
        return manifest, True
    payload = _extract_worker_payload(result, run_id=run_id)
    payload.update(
        {
            "schema_version": RUN_SCHEMA,
            "source_segment_id": segment_id,
            "execution_head": execution_head,
            "original_logical_run_id": record["original_logical_run_id"],
            "original_sequence_index": record["original_sequence_index"],
            "replacement_for_failed_finalization": record[
                "replacement_for_failed_finalization"
            ],
        }
    )
    _write_durable_exclusive(run_root / "result.json", payload)
    action = snapshot_policy(phase="postflight", state=post.state)
    if action == "finalize_run":
        status, reason, abort = "completed", None, False
    elif action == "exclude_current_and_continue":
        status = "postflight_snapshot_unavailable"
        reason = "postflight_snapshot:query_failed"
        abort = False
        _record_disposition(
            root=run_root,
            run_id=run_id,
            status=status,
            reason=reason,
            launched=True,
        )
    else:
        status = "foreign_process_detected"
        reason = "postflight_snapshot:foreign_process_detected"
        abort = True
        _record_disposition(
            root=run_root,
            run_id=run_id,
            status=status,
            reason=reason,
            launched=True,
        )
    manifest = _run_manifest(
        root=run_root,
        segment_id=segment_id,
        execution_head=execution_head,
        run_id=run_id,
        record=record,
        status=status,
        reason=reason,
    )
    _write_durable_exclusive(
        run_root / "result-binding.json",
        {
            "schema_version": "kvbench-phase13d-continuation-result-binding-1.0.0",
            "run_id": run_id,
            "result_sha256": sha256_file(run_root / "result.json"),
        },
    )
    return manifest, abort


def new_segment_id(execution_head: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", execution_head) is None:
        raise Phase13DContinuationError("execution HEAD is invalid")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")
    return f"phase13dseg-{stamp}z-{execution_head[:8]}-{secrets.token_hex(3)}"


def reserve_segment(*, segment_id: str, execution_head: str) -> Path:
    if _SEGMENT_RE.fullmatch(segment_id) is None:
        raise Phase13DContinuationError("segment ID is invalid")
    phase13d._require_clean_execution_sha(execution_head)
    _validate_segment_a(ORIGINAL_STAGE)
    _validate_failed_run(ORIGINAL_STAGE)
    records = continuation_records(_order())
    segments = ORIGINAL_STAGE / "segments"
    segments.mkdir(exist_ok=True)
    root = segments / segment_id
    root.mkdir()
    (root / "runs").mkdir()
    _write_durable_exclusive(
        root / "segment-reservation.json",
        {
            "schema_version": "kvbench-phase13d-continuation-reservation-1.0.0",
            "campaign_id": CAMPAIGN_ID,
            "segment_id": segment_id,
            "execution_head": execution_head,
            "created_at_utc": _utc_now(),
            "original_campaign_manifest_sha256": sha256_file(
                ORIGINAL_STAGE / "campaign_manifest.json"
            ),
            "original_execution_order_sha256": sha256_file(
                ORIGINAL_STAGE / "execution_order.json"
            ),
            "replacement_records": 1,
            "remaining_original_records": 200,
            "planned_segment_records": len(records),
            "append_only": True,
        },
    )
    return root


def _prefix_catalog_index(prefix_root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    catalog = _strict_json(prefix_root / "catalog.json")
    index = {}
    for raw in catalog["entries"]:
        entry = dict(raw)
        entry["snapshot_root"] = prefix_root / str(entry["snapshot_relative_path"])
        key = (
            str(entry["method_config_id"]),
            int(entry["batch_size"]),
            int(entry["historical_context"]),
        )
        index[key] = entry
    return index


def run_segment(
    *, segment_root: Path, segment_id: str, execution_head: str, prefix_root: Path
) -> dict[str, Any]:
    if segment_root != ORIGINAL_STAGE / "segments" / segment_id:
        raise Phase13DContinuationError("segment path differs")
    phase12._require_authorized_container_runtime()
    if any(name in os.environ for name in pilot._FORBIDDEN_ENVIRONMENT):
        raise Phase13DContinuationError("credentials entered Measurement Container")
    phase13d._configure_pilot_contexts()
    phase13d._require_clean_execution_sha(execution_head)
    phase13d.validate_preregistration(replay_source=False)
    prefix = validate_prefix_reuse(prefix_root, verify_state_bytes=True)
    _write_durable_exclusive(segment_root / "prefix-reuse-validation.json", prefix)
    equivalence = timing_critical_equivalence(execution_head)
    _write_durable_exclusive(
        segment_root / "timing-critical-equivalence.json", equivalence
    )
    records = continuation_records(_order())
    _write_durable_exclusive(
        segment_root / "continuation-order.json",
        {
            "schema_version": "kvbench-phase13d-continuation-order-1.0.0",
            "campaign_id": CAMPAIGN_ID,
            "segment_id": segment_id,
            "records": records,
            "records_sha256": _canonical_sha256(records),
            "randomization_regenerated": False,
            "valid_segment_a_runs_rerun": False,
        },
    )
    _write_durable_exclusive(
        segment_root / "segment_manifest.json",
        {
            "schema_version": SEGMENT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "segment_id": segment_id,
            "execution_head": execution_head,
            "original_execution_head": ORIGINAL_EXECUTION_HEAD,
            "source_campaign_manifest_path": "../../campaign_manifest.json",
            "source_execution_order_path": "../../execution_order.json",
            "segment_a_id": SEGMENT_A_ID,
            "segment_a_valid_runs": SEGMENT_A_VALID_RUNS,
            "segment_a_tree_sha256": SEGMENT_A_TREE_SHA256,
            "original_failed_run_id": FAILED_RUN_ID,
            "original_failed_run_tree_sha256": FAILED_RUN_TREE_SHA256,
            "original_failed_run_excluded": True,
            "replacement_records": 1,
            "remaining_records": 200,
            "planned_records": len(records),
            "snapshot_states": sorted(SNAPSHOT_STATES),
            "snapshot_retries_after_initial_attempt": SNAPSHOT_RETRIES,
            "prefix_state_count": PREFIX_STATE_COUNT,
            "prefix_states_reused_read_only": True,
            "prefix_states_regenerated": False,
            "warmup_steps": 64,
            "measured_steps": 128,
            "measured_batches": phase13d.MEASURED_BATCHES,
            "candidate_table_unchanged": True,
            "randomization_unchanged": True,
            "authorized_container_digest": phase13d.AUTHORIZED_CONTAINER_DIGEST,
        },
    )
    index = _prefix_catalog_index(prefix_root)
    manifests: list[dict[str, Any]] = []
    abort_reason: str | None = None
    for record in records:
        if abort_reason is not None:
            run_id = _segment_run_id(segment_id, record)
            run_root = segment_root / "runs" / run_id
            run_root.mkdir()
            _record_disposition(
                root=run_root,
                run_id=run_id,
                status="aborted",
                reason=abort_reason,
                launched=False,
            )
            manifests.append(
                _run_manifest(
                    root=run_root,
                    segment_id=segment_id,
                    execution_head=execution_head,
                    run_id=run_id,
                    record=record,
                    status="aborted",
                    reason=abort_reason,
                )
            )
            continue
        key = (
            str(record["method_config_id"]),
            int(record["batch_size"]),
            int(record["historical_context"]),
        )
        manifest, abort = _run_one(
            segment_root=segment_root,
            segment_id=segment_id,
            execution_head=execution_head,
            record=record,
            prefix_entry=index[key],
        )
        manifests.append(manifest)
        if abort or manifest["status"] == "runtime_failed":
            abort_reason = str(manifest["reason"])
    counts = Counter(str(item["status"]) for item in manifests)
    completed = counts.get("completed", 0)
    status = "LOCAL_COMPLETE" if completed == SEGMENT_B_RECORDS else "PARTIAL"
    result = {
        "schema_version": "kvbench-phase13d-continuation-local-1.0.0",
        "campaign_id": CAMPAIGN_ID,
        "segment_id": segment_id,
        "execution_head": execution_head,
        "execution_heads": [execution_head],
        "planned_records": SEGMENT_B_RECORDS,
        "status_counts": dict(sorted(counts.items())),
        "segment_status": status,
        "abort_reason": abort_reason,
        "selective_reruns": 0,
        "segment_a_runs_rerun": 0,
    }
    _write_durable_exclusive(segment_root / "segment-result.json", result)
    _write_durable_exclusive(
        segment_root / "SEGMENT_COMPLETE",
        {
            "schema_version": "kvbench-phase13d-continuation-complete-1.0.0",
            "segment_id": segment_id,
            "status": status,
            "segment_result_sha256": sha256_file(segment_root / "segment-result.json"),
            "written_last": True,
        },
    )
    return result


def _validate_completed_continuation_run(
    *,
    run_root: Path,
    record: Mapping[str, Any],
    require_manifest: bool,
) -> dict[str, Any]:
    run_id = _segment_run_id(run_root.parents[1].name, record)
    result = _strict_json(run_root / "result.json")
    summary = _strict_json(run_root / "worker.snapshot-summary.json")
    required_stages = {
        f"{sequence:02d}-{stage}-{state}.json"
        for sequence, stage, state in (
            (1, "model_load", "started"),
            (2, "model_load", "completed"),
            (3, "prefix_construction", "started"),
            (4, "prefix_construction", "completed"),
            (5, "graph_capture", "started"),
            (6, "graph_capture", "completed"),
            (7, "warmup_and_audit", "started"),
            (8, "warmup_and_audit", "completed"),
            (9, "measurement", "started"),
            (10, "measurement", "completed"),
            (11, "finalization", "started"),
            (12, "finalization", "completed"),
        )
    }
    observed_stages = {
        path.name for path in (run_root / "stage-progress").glob("*.json")
    }
    if (
        result.get("run_id") != run_id
        or result.get("source_segment_id") != run_root.parents[1].name
        or result.get("original_sequence_index")
        != record["original_sequence_index"]
        or result.get("original_logical_run_id")
        != record["original_logical_run_id"]
        or result.get("finite_output") is not True
        or result.get("gpu_exclusive") is not True
        or result.get("no_backend_fallback") is not True
        or result.get("allocation_stable") is not True
        or result.get("kernel_path_stable") is not True
        or result.get("graph_replay_allocation", {}).get("passed") is not True
        or result.get("graph_replay_allocation", {}).get(
            "allocation_event_count"
        )
        != 0
        or summary.get("preflight_state") != "clean"
        or summary.get("postflight_state") != "clean"
        or observed_stages != required_stages
    ):
        raise Phase13DContinuationError(
            "completed continuation run evidence differs"
        )
    if require_manifest:
        manifest = _strict_json(run_root / "manifest.json")
        binding = _strict_json(run_root / "result-binding.json")
        if (
            manifest.get("run_id") != run_id
            or manifest.get("status") != "completed"
            or manifest.get("reason") is not None
            or manifest.get("original_sequence_index")
            != record["original_sequence_index"]
            or manifest.get("execution_head") != result.get("execution_head")
            or binding.get("run_id") != run_id
            or binding.get("result_sha256") != sha256_file(run_root / "result.json")
            or (run_root / "disposition.json").exists()
        ):
            raise Phase13DContinuationError(
                "completed continuation manifest differs"
            )
    elif (
        (run_root / "manifest.json").exists()
        or (run_root / "result-binding.json").exists()
        or (run_root / "disposition.json").exists()
    ):
        raise Phase13DContinuationError(
            "repairable finalization has unexpected control files"
        )
    return result


def _repair_completed_finalization(
    *,
    run_root: Path,
    segment_id: str,
    record: Mapping[str, Any],
    repair_execution_head: str,
) -> dict[str, Any]:
    result = _validate_completed_continuation_run(
        run_root=run_root, record=record, require_manifest=False
    )
    measurement_execution_head = str(result.get("execution_head"))
    if re.fullmatch(r"[0-9a-f]{40}", measurement_execution_head) is None:
        raise Phase13DContinuationError(
            "repairable result execution HEAD differs"
        )
    _write_durable_exclusive(
        run_root / "finalization-repair.json",
        {
            "schema_version": (
                "kvbench-phase13d-continuation-finalization-repair-1.0.0"
            ),
            "run_id": result["run_id"],
            "measurement_execution_head": measurement_execution_head,
            "repair_execution_head": repair_execution_head,
            "measurement_reexecuted": False,
            "reason": "manifest_optional_q4_field_keyerror_after_clean_postflight",
            "result_sha256": sha256_file(run_root / "result.json"),
            "snapshot_summary_sha256": sha256_file(
                run_root / "worker.snapshot-summary.json"
            ),
        },
    )
    manifest = _run_manifest(
        root=run_root,
        segment_id=segment_id,
        execution_head=measurement_execution_head,
        run_id=str(result["run_id"]),
        record=record,
        status="completed",
        reason=None,
    )
    _write_durable_exclusive(
        run_root / "result-binding.json",
        {
            "schema_version": (
                "kvbench-phase13d-continuation-result-binding-1.0.0"
            ),
            "run_id": result["run_id"],
            "result_sha256": sha256_file(run_root / "result.json"),
        },
    )
    _validate_completed_continuation_run(
        run_root=run_root, record=record, require_manifest=True
    )
    return manifest


def resume_segment(
    *,
    segment_root: Path,
    segment_id: str,
    execution_head: str,
    prefix_root: Path,
) -> dict[str, Any]:
    if segment_root != ORIGINAL_STAGE / "segments" / segment_id:
        raise Phase13DContinuationError("segment path differs")
    phase12._require_authorized_container_runtime()
    if any(name in os.environ for name in pilot._FORBIDDEN_ENVIRONMENT):
        raise Phase13DContinuationError(
            "credentials entered Measurement Container"
        )
    phase13d._configure_pilot_contexts()
    phase13d._require_clean_execution_sha(execution_head)
    phase13d.validate_preregistration(replay_source=False)
    if (segment_root / "SEGMENT_COMPLETE").exists():
        raise Phase13DContinuationError("completed segment cannot resume")
    initial_manifest = _strict_json(segment_root / "segment_manifest.json")
    stored_order = _strict_json(segment_root / "continuation-order.json")
    records = continuation_records(_order())
    if (
        initial_manifest.get("segment_id") != segment_id
        or initial_manifest.get("execution_head")
        != "93d60abad3f71a9872f86849753312d95a105be1"
        or stored_order.get("records") != records
        or stored_order.get("records_sha256") != _canonical_sha256(records)
    ):
        raise Phase13DContinuationError("resumable segment authority differs")
    prefix = validate_prefix_reuse(prefix_root, verify_state_bytes=True)
    equivalence = timing_critical_equivalence(execution_head)
    _write_durable_exclusive(
        segment_root / f"timing-critical-equivalence-{execution_head}.json",
        equivalence,
    )
    existing_roots = sorted(
        path.name
        for path in (segment_root / "runs").iterdir()
        if path.is_dir()
    )
    expected_roots: list[str] = []
    preserved = 0
    repaired = 0
    start_index: int | None = None
    for index, record in enumerate(records):
        run_id = _segment_run_id(segment_id, record)
        run_root = segment_root / "runs" / run_id
        if not run_root.exists():
            start_index = index
            break
        expected_roots.append(run_id)
        if (run_root / "manifest.json").is_file():
            _validate_completed_continuation_run(
                run_root=run_root, record=record, require_manifest=True
            )
            preserved += 1
            continue
        _repair_completed_finalization(
            run_root=run_root,
            segment_id=segment_id,
            record=record,
            repair_execution_head=execution_head,
        )
        repaired += 1
        preserved += 1
    if start_index is None:
        start_index = len(records)
    if existing_roots != expected_roots or repaired != 1 or preserved != 13:
        raise Phase13DContinuationError("resumable prefix differs")
    _write_durable_exclusive(
        segment_root / f"resume-manifest-{execution_head}.json",
        {
            "schema_version": (
                "kvbench-phase13d-continuation-resume-1.0.0"
            ),
            "campaign_id": CAMPAIGN_ID,
            "segment_id": segment_id,
            "initial_execution_head": initial_manifest["execution_head"],
            "resume_execution_head": execution_head,
            "preserved_completed_runs": preserved,
            "repaired_finalization_records": repaired,
            "measurement_records_rerun": 0,
            "resume_original_sequence_index": records[start_index][
                "original_sequence_index"
            ],
            "remaining_records": len(records) - start_index,
            "frozen_order_regenerated": False,
            "prefix_states_regenerated": False,
        },
    )
    index = _prefix_catalog_index(prefix_root)
    manifests: list[dict[str, Any]] = [
        _strict_json(
            segment_root
            / "runs"
            / _segment_run_id(segment_id, record)
            / "manifest.json"
        )
        for record in records[:start_index]
    ]
    abort_reason: str | None = None
    for record in records[start_index:]:
        if abort_reason is not None:
            run_id = _segment_run_id(segment_id, record)
            run_root = segment_root / "runs" / run_id
            run_root.mkdir()
            _record_disposition(
                root=run_root,
                run_id=run_id,
                status="aborted",
                reason=abort_reason,
                launched=False,
            )
            manifests.append(
                _run_manifest(
                    root=run_root,
                    segment_id=segment_id,
                    execution_head=execution_head,
                    run_id=run_id,
                    record=record,
                    status="aborted",
                    reason=abort_reason,
                )
            )
            continue
        key = (
            str(record["method_config_id"]),
            int(record["batch_size"]),
            int(record["historical_context"]),
        )
        manifest, abort = _run_one(
            segment_root=segment_root,
            segment_id=segment_id,
            execution_head=execution_head,
            record=record,
            prefix_entry=index[key],
        )
        manifests.append(manifest)
        if abort or manifest["status"] == "runtime_failed":
            abort_reason = str(manifest["reason"])
    counts = Counter(str(item["status"]) for item in manifests)
    completed = counts.get("completed", 0)
    status = "LOCAL_COMPLETE" if completed == SEGMENT_B_RECORDS else "PARTIAL"
    execution_heads = sorted(
        {str(item["execution_head"]) for item in manifests}
    )
    result = {
        "schema_version": "kvbench-phase13d-continuation-local-1.0.0",
        "campaign_id": CAMPAIGN_ID,
        "segment_id": segment_id,
        "execution_heads": execution_heads,
        "planned_records": SEGMENT_B_RECORDS,
        "status_counts": dict(sorted(counts.items())),
        "segment_status": status,
        "abort_reason": abort_reason,
        "selective_reruns": 0,
        "measurement_records_rerun": 0,
        "segment_a_runs_rerun": 0,
        "preserved_resume_runs": preserved,
        "repaired_finalization_records": repaired,
    }
    _write_durable_exclusive(segment_root / "segment-result.json", result)
    _write_durable_exclusive(
        segment_root / "SEGMENT_COMPLETE",
        {
            "schema_version": (
                "kvbench-phase13d-continuation-complete-1.0.0"
            ),
            "segment_id": segment_id,
            "status": status,
            "segment_result_sha256": sha256_file(
                segment_root / "segment-result.json"
            ),
            "written_last": True,
        },
    )
    return result


def _run_record(
    *,
    root: Path,
    manifest_path: Path,
    source_segment_id: str,
    execution_head: str,
    original_sequence_index: int,
    original_logical_run_id: str,
) -> dict[str, Any]:
    manifest = _strict_json(manifest_path)
    run_root = manifest_path.parent
    result_path = run_root / "result.json"
    result = _strict_json(result_path) if result_path.is_file() else None
    return {
        **manifest,
        "source_segment_id": source_segment_id,
        "execution_head": execution_head,
        "original_sequence_index": original_sequence_index,
        "original_logical_run_id": original_logical_run_id,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "result_path": (
            result_path.relative_to(root).as_posix() if result is not None else None
        ),
        "result_sha256": sha256_file(result_path) if result is not None else None,
        "process_median_ms": (
            float(result["process_median_ms"]) if result is not None else None
        ),
        "host_wall_cuda_event_ratio": (
            float(result["host_wall_cuda_event_ratio"])
            if result is not None
            else None
        ),
        "kernel_count": result.get("kernel_count") if result else None,
        "finite_output": result.get("finite_output") if result else None,
        "no_backend_fallback": result.get("no_backend_fallback") if result else None,
        "allocation_stable": result.get("allocation_stable") if result else None,
        "kernel_path_stable": result.get("kernel_path_stable") if result else None,
        "gpu_exclusive": result.get("gpu_exclusive") if result else None,
        "output_checksum": result.get("output_checksum") if result else None,
        "kernel_path_fingerprint": (
            result.get("kernel_path_fingerprint") if result else None
        ),
        "allocation_fingerprint": (
            result.get("allocation_fingerprint") if result else None
        ),
        "temperature_min_c": result.get("temperature_min_c") if result else None,
        "temperature_max_c": result.get("temperature_max_c") if result else None,
        "sm_clock_min_mhz": result.get("sm_clock_min_mhz") if result else None,
        "sm_clock_max_mhz": result.get("sm_clock_max_mhz") if result else None,
        "memory_clock_min_mhz": (
            result.get("memory_clock_min_mhz") if result else None
        ),
        "memory_clock_max_mhz": (
            result.get("memory_clock_max_mhz") if result else None
        ),
        "power_min_w": result.get("power_min_w") if result else None,
        "power_max_w": result.get("power_max_w") if result else None,
        "runner": result.get("runner") if result else None,
    }


def _require_combined_logical_coverage(
    records: Sequence[Mapping[str, Any]],
) -> None:
    segment_a = [
        item for item in records if item["source_segment_id"] == SEGMENT_A_ID
    ]
    replacements = [
        item
        for item in records
        if item["source_segment_id"] != SEGMENT_A_ID
        and item["original_logical_run_id"] == FAILED_RUN_ID
        and item.get("replacement_for_failed_finalization") is True
    ]
    if (
        len(records) != TOTAL_LOGICAL_RECORDS
        or [int(item["original_sequence_index"]) for item in records]
        != list(range(TOTAL_LOGICAL_RECORDS))
        or len(segment_a) != SEGMENT_A_VALID_RUNS
        or [int(item["original_sequence_index"]) for item in segment_a]
        != list(range(SEGMENT_A_VALID_RUNS))
        or len(replacements) != 1
        or int(replacements[0]["original_sequence_index"])
        != FAILED_SEQUENCE_INDEX
        or any(
            item.get("manifest_path")
            == f"runs/{FAILED_RUN_ID}/manifest.json"
            for item in records
        )
    ):
        raise Phase13DContinuationError("combined logical coverage differs")


def combined_run_records(root: Path, *, segment_id: str) -> list[dict[str, Any]]:
    order = _order()["records"]
    records: list[dict[str, Any]] = []
    for sequence_index in range(SEGMENT_A_VALID_RUNS):
        logical = order[sequence_index]
        run_id = phase13d._run_id(CAMPAIGN_ID, logical)
        records.append(
            _run_record(
                root=root,
                manifest_path=root / "runs" / run_id / "manifest.json",
                source_segment_id=SEGMENT_A_ID,
                execution_head=ORIGINAL_EXECUTION_HEAD,
                original_sequence_index=sequence_index,
                original_logical_run_id=run_id,
            )
        )
    segment_root = root / "segments" / segment_id
    segment_order = _strict_json(segment_root / "continuation-order.json")["records"]
    for logical in segment_order:
        run_id = _segment_run_id(segment_id, logical)
        manifest_path = segment_root / "runs" / run_id / "manifest.json"
        run_manifest = _strict_json(manifest_path)
        records.append(
            _run_record(
                root=root,
                manifest_path=manifest_path,
                source_segment_id=segment_id,
                execution_head=str(run_manifest["execution_head"]),
                original_sequence_index=int(logical["original_sequence_index"]),
                original_logical_run_id=str(logical["original_logical_run_id"]),
            )
        )
    records.sort(key=lambda item: int(item["original_sequence_index"]))
    _require_combined_logical_coverage(records)
    return records


def _enrich_summaries(
    summaries: list[dict[str, Any]], records: Sequence[Mapping[str, Any]]
) -> None:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["method_config_id"]),
                int(record["batch_size"]),
                int(record["historical_context"]),
            )
        ].append(record)
    for summary in summaries:
        matching = grouped[
            (
                str(summary["method_config_id"]),
                int(summary["batch_size"]),
                int(summary["historical_context"]),
            )
        ]
        summary["source_segment_ids"] = sorted(
            {str(item["source_segment_id"]) for item in matching}
        )
        summary["execution_heads"] = sorted(
            {str(item["execution_head"]) for item in matching}
        )
        summary["original_sequence_indices"] = sorted(
            int(item["original_sequence_index"]) for item in matching
        )


def materialize_combined(root: Path, *, segment_id: str) -> dict[str, Any]:
    _validate_segment_a(root)
    _validate_failed_run(root)
    segment = _strict_json(root / "segments" / segment_id / "segment-result.json")
    records = combined_run_records(root, segment_id=segment_id)
    candidate = _strict_json(phase13d.CANDIDATE_PATH)
    summaries = phase13d._new_point_summaries(records, candidate)
    _enrich_summaries(summaries, records)
    source = phase13d._source_summaries()
    phase13d._mark_monotonicity_warnings(source=source, densified=summaries)
    fits = phase13d._combined_fit_records(
        source=source,
        densified=summaries,
        candidate=candidate,
        campaign_id=CAMPAIGN_ID,
    )
    for fit in fits:
        matching = [
            row
            for row in summaries
            if row["method_config_id"] == fit["method_config_id"]
            and int(row["batch_size"]) == int(fit["batch_size"])
        ]
        fit["densification_source_segment_ids"] = sorted(
            {value for row in matching for value in row["source_segment_ids"]}
        )
        fit["densification_execution_heads"] = sorted(
            {value for row in matching for value in row["execution_heads"]}
        )
    combined_index = phase13d._combined_source_index(source, summaries, CAMPAIGN_ID)
    ratios = phase13d._ratio_records(source, summaries)
    targets = [dict(row) for row in candidate["targets"]]
    proposals = [dict(row) for row in candidate["proposals"]]
    feasibility = _strict_json(root / "unified/feasibility.json")["records"]
    exclusions = [
        {
            "kind": "original_failed_finalization",
            "run_id": FAILED_RUN_ID,
            "original_sequence_index": FAILED_SEQUENCE_INDEX,
            "reason": "post_worker_gpu_process_snapshot_rejected_without_payload",
            "tree_sha256": FAILED_RUN_TREE_SHA256,
            "timing_used": False,
        }
    ] + [
        {
            "kind": "continuation_run",
            "run_id": row["run_id"],
            "original_sequence_index": row["original_sequence_index"],
            "reason": row["reason"],
            "status": row["status"],
            "timing_used": False,
        }
        for row in records
        if row["status"] != "completed"
    ]
    phase13d._write_parquet(root / "target_table.parquet", targets)
    phase13d._write_parquet(root / "candidate_generation.parquet", proposals)
    phase13d._write_parquet(root / "feasibility.parquet", feasibility)
    phase13d._write_parquet(root / "raw_run_index.parquet", records)
    phase13d._write_parquet(root / "point_summary.parquet", summaries)
    phase13d._write_parquet(root / "combined_source_index.parquet", combined_index)
    phase13d._write_parquet(root / "refined_knees.parquet", fits)
    phase13d._write_parquet(root / "exclusions.parquet", exclusions)
    status_counts = Counter(str(row["status"]) for row in records)
    stable = [row for row in summaries if row["disposition"] == "stable"]
    unstable = [row for row in summaries if row["disposition"] == "unstable"]
    target_fits = [row for row in fits if row["was_original_densification_target"]]
    unresolved = [
        row
        for row in target_fits
        if row["resolution_status"]
        in {"densification_required", "unstable_data", "fit_failed"}
    ]
    resolution_counts = Counter(str(row["resolution_status"]) for row in target_fits)
    local_pass = bool(
        segment.get("segment_status") == "LOCAL_COMPLETE"
        and status_counts == {"completed": TOTAL_LOGICAL_RECORDS}
        and len(stable) == 84
        and not unstable
        and not unresolved
    )
    qc = {
        "schema_version": "kvbench-phase13d-continuation-qc-1.0.0",
        "campaign_id": CAMPAIGN_ID,
        "segment_id": segment_id,
        "phase13d_status": "LOCAL_PASS_PENDING_PUBLICATION" if local_pass else "PARTIAL",
        "segment_a_valid_runs": SEGMENT_A_VALID_RUNS,
        "segment_b_records": SEGMENT_B_RECORDS,
        "original_failed_run_excluded": True,
        "total_logical_records": TOTAL_LOGICAL_RECORDS,
        "status_counts": dict(sorted(status_counts.items())),
        "stable_new_points": len(stable),
        "unstable_new_points": len(unstable),
        "maximum_cv": max(
            (float(row["cv"]) for row in summaries if row["cv"] is not None),
            default=None,
        ),
        "resolution_status_counts": dict(sorted(resolution_counts.items())),
        "remaining_unresolved_targets": [row["target_id"] for row in unresolved],
        "timing_critical_equivalence": "PASS",
        "prefix_states_reused_read_only": True,
        "selective_reruns": 0,
        "r_hbm": None,
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "performance_claim_eligible": False,
        "phase14_readiness": "READY" if local_pass else "NOT_READY",
        "pilot_only_ratio_records": len(ratios),
        "pilot_only_ratios_calculated": sum(row["calculated"] for row in ratios),
    }
    _write_durable_exclusive(root / "densification_qc.json", qc)
    report = [
        "# Phase 13D Continuation",
        "",
        f"- Campaign: `{CAMPAIGN_ID}`",
        f"- Segment: `{segment_id}`",
        f"- Local status: `{qc['phase13d_status']}`",
        f"- Segment A / Segment B: {SEGMENT_A_VALID_RUNS}/{SEGMENT_B_RECORDS}",
        f"- Final logical coverage: {status_counts.get('completed', 0)}/{TOTAL_LOGICAL_RECORDS}",
        f"- Maximum CV: `{qc['maximum_cv']}`",
        f"- Refined resolution statuses: `{json.dumps(qc['resolution_status_counts'], sort_keys=True)}`",
        f"- Remaining unresolved: `{json.dumps(qc['remaining_unresolved_targets'])}`",
        f"- Phase 14 readiness: `{qc['phase14_readiness']}`",
        "- Original failed-finalization run excluded; its bytes remain unchanged.",
        "- Full Scan remains CLOSED and quality execution remains LOCKED.",
        "- No final speedup, HBM, capacity, knee, or quality claim is made.",
        "",
    ]
    write_exclusive(root / "densification_report.md", "\n".join(report).encode())
    (root / "plots").mkdir()
    phase13d._render_plots(root, source=source, densified=summaries, fits=fits)
    _write_durable_exclusive(
        root / "inventory.json",
        {
            "schema_version": "kvbench-phase13d-continuation-scientific-inventory-1.0.0",
            "campaign_id": CAMPAIGN_ID,
            "raw_run_records": len(records),
            "segment_a_run_records": SEGMENT_A_VALID_RUNS,
            "segment_b_run_records": SEGMENT_B_RECORDS,
            "original_failed_runs_excluded": 1,
            "new_point_summaries": len(summaries),
            "refined_fit_records": len(fits),
            "source_bundle_copied": False,
            "r_hbm": None,
        },
    )
    _write_durable_exclusive(
        root / "unified/combined-campaign.json",
        {
            "schema_version": "kvbench-phase13d-combined-campaign-1.0.0",
            "campaign_id": CAMPAIGN_ID,
            "segment_a_id": SEGMENT_A_ID,
            "segment_b_id": segment_id,
            "segment_a_execution_head": ORIGINAL_EXECUTION_HEAD,
            "segment_b_execution_heads": _strict_json(
                root / "segments" / segment_id / "segment-result.json"
            )["execution_heads"],
            "logical_coverage": len(records),
            "original_failed_run_excluded": FAILED_RUN_ID,
            "status": qc["phase13d_status"],
        },
    )
    return qc


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    paths = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase13DContinuationError("campaign contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase13DContinuationError("campaign contains unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            paths.append(path)
    return paths


def seal_combined(root: Path, *, segment_id: str) -> Path:
    qc = _strict_json(root / "densification_qc.json")
    segment_result = _strict_json(root / "segments" / segment_id / "segment-result.json")
    _write_durable_exclusive(
        root / "manifest.json",
        {
            "schema_version": "kvbench-phase13d-continuation-artifact-1.0.0",
            "run_id": CAMPAIGN_ID,
            "campaign_id": CAMPAIGN_ID,
            "status": qc["phase13d_status"],
            "created_at_utc": _utc_now(),
            "original_execution_head": ORIGINAL_EXECUTION_HEAD,
            "continuation_execution_heads": segment_result["execution_heads"],
            "segment_id": segment_id,
            "authorized_container_digest": phase13d.AUTHORIZED_CONTAINER_DIGEST,
            "append_only": True,
            "complete_written_last": True,
            "quality_status": "unvalidated",
            "performance_claim_eligible": False,
            "r_hbm": None,
        },
    )
    excluded = {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
    items = [
        {
            "path": path.relative_to(root).as_posix(),
            "role": "phase13d_continuation_evidence",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _payload_paths(root, excluded)
    ]
    _write_durable_exclusive(
        root / "artifact_inventory.json",
        {
            "schema_version": "kvbench-artifact-inventory-1.0.0",
            "run_id": CAMPAIGN_ID,
            "files": items,
            "excluded_control_files": [
                "artifact_inventory.json",
                "checksums.sha256",
                "COMPLETE",
            ],
        },
    )
    ledger = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in _payload_paths(root, {"checksums.sha256", "COMPLETE"})
    ).encode()
    write_exclusive(root / "checksums.sha256", ledger)
    _fsync_directory(root)
    _write_durable_exclusive(
        root / "COMPLETE",
        {
            "schema_version": "kvbench-completion-1.0.0",
            "run_id": CAMPAIGN_ID,
            "status": qc["phase13d_status"],
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "artifact_inventory_sha256": sha256_file(root / "artifact_inventory.json"),
            "checksum_ledger_path": "checksums.sha256",
            "checksum_ledger_sha256": sha256_file(root / "checksums.sha256"),
            "written_last": True,
        },
    )
    final = REPOSITORY_ROOT / "artifacts/phase13d" / CAMPAIGN_ID
    if final.exists() or final.is_symlink():
        raise Phase13DContinuationError("final campaign path already exists")
    rename_noreplace(root, final)
    return final


def validate_combined(root: Path, *, segment_id: str) -> dict[str, Any]:
    artifact = validate_local_artifact(root, environ={})
    _validate_segment_a(root)
    _validate_failed_run(root)
    manifest = _strict_json(root / "manifest.json")
    complete = _strict_json(root / "COMPLETE")
    qc = _strict_json(root / "densification_qc.json")
    records = combined_run_records(root, segment_id=segment_id)
    if (
        manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("segment_id") != segment_id
        or manifest.get("original_execution_head") != ORIGINAL_EXECUTION_HEAD
        or manifest.get("authorized_container_digest")
        != phase13d.AUTHORIZED_CONTAINER_DIGEST
        or complete.get("written_last") is not True
        or qc.get("total_logical_records") != TOTAL_LOGICAL_RECORDS
        or qc.get("segment_a_valid_runs") != SEGMENT_A_VALID_RUNS
        or qc.get("segment_b_records") != SEGMENT_B_RECORDS
        or qc.get("original_failed_run_excluded") is not True
        or len(records) != TOTAL_LOGICAL_RECORDS
        or qc.get("full_scan") != "CLOSED"
        or qc.get("quality_execution") != "LOCKED"
        or qc.get("r_hbm") is not None
    ):
        raise Phase13DContinuationError("combined campaign semantics differ")
    return {
        "status": "PASS",
        "campaign_id": CAMPAIGN_ID,
        "segment_id": segment_id,
        "phase13d_status": qc["phase13d_status"],
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "logical_coverage": len(records),
        "maximum_cv": qc["maximum_cv"],
        "remaining_unresolved_targets": qc["remaining_unresolved_targets"],
    }


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--new-segment-id", action="store_true")
    actions.add_argument("--reserve-segment", action="store_true")
    actions.add_argument("--validate-entry", action="store_true")
    actions.add_argument("--run-segment", action="store_true")
    actions.add_argument("--resume-segment", action="store_true")
    actions.add_argument("--materialize-combined", action="store_true")
    actions.add_argument("--seal-combined", action="store_true")
    actions.add_argument("--validate-combined", action="store_true")
    parser.add_argument("--segment-id")
    parser.add_argument("--execution-head")
    parser.add_argument("--segment-root", type=Path)
    parser.add_argument("--campaign-root", type=Path, default=ORIGINAL_STAGE)
    parser.add_argument("--prefix-root", type=Path, default=PREFIX_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    if args.new_segment_id:
        if args.execution_head is None:
            raise Phase13DContinuationError("execution HEAD is required")
        print(new_segment_id(args.execution_head))
        return 0
    if args.reserve_segment:
        if args.segment_id is None or args.execution_head is None:
            raise Phase13DContinuationError("segment identity is required")
        print(
            reserve_segment(
                segment_id=args.segment_id, execution_head=args.execution_head
            )
        )
        return 0
    if args.validate_entry:
        result = {
            "segment_a": _validate_segment_a(args.campaign_root),
            "failed_run": _validate_failed_run(args.campaign_root),
            "prefix": validate_prefix_reuse(
                args.prefix_root, verify_state_bytes=True
            ),
            "continuation_records": len(continuation_records(_order())),
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.run_segment:
        if (
            args.segment_id is None
            or args.execution_head is None
            or args.segment_root is None
        ):
            raise Phase13DContinuationError("segment execution arguments are required")
        print(
            json.dumps(
                run_segment(
                    segment_root=args.segment_root,
                    segment_id=args.segment_id,
                    execution_head=args.execution_head,
                    prefix_root=args.prefix_root,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.resume_segment:
        if (
            args.segment_id is None
            or args.execution_head is None
            or args.segment_root is None
        ):
            raise Phase13DContinuationError(
                "segment resume arguments are required"
            )
        print(
            json.dumps(
                resume_segment(
                    segment_root=args.segment_root,
                    segment_id=args.segment_id,
                    execution_head=args.execution_head,
                    prefix_root=args.prefix_root,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.materialize_combined:
        if args.segment_id is None:
            raise Phase13DContinuationError("segment ID is required")
        print(json.dumps(materialize_combined(args.campaign_root, segment_id=args.segment_id), sort_keys=True))
        return 0
    if args.seal_combined:
        if args.segment_id is None:
            raise Phase13DContinuationError("segment ID is required")
        print(seal_combined(args.campaign_root, segment_id=args.segment_id))
        return 0
    if args.validate_combined:
        if args.segment_id is None:
            raise Phase13DContinuationError("segment ID is required")
        print(json.dumps(validate_combined(args.campaign_root, segment_id=args.segment_id), sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

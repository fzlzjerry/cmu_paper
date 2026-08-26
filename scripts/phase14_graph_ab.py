"""Phase 14 paired eager/CUDA-Graph mechanism experiment.

This coordinator reuses the admitted fixed-L runner, timing implementation,
Phase-13 feasibility formulas, exact-batch prefix states, stage supervision,
append-only lifecycle, and provisional knee fitter.  Its only new behavior is
the preregistered paired Graph OFF/ON schedule and paired analysis.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import stat
import statistics
import subprocess
import sys
from typing import Any, Iterator

from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from kvbench.runtime.artifacts import sha256_file
from kvbench.runtime.process_supervision import run_stage_supervised_command
from kvbench.schema import GraphMode, RunnerKind
from scripts.r2_artifact import validate_local_artifact
from scripts import phase12_unified_admission as phase12
from scripts import phase13_pilot as pilot
from scripts import phase13d_continuation as continuation
from scripts.phase13_prefix_state import validate_prefix_state


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase14"
STAGING_ROOT = ARTIFACT_ROOT / ".kvbench-staging"
PLAN_PATH = REPOSITORY_ROOT / "docs/plans/phase14-graph-ab.md"
ORDER_PATH = REPOSITORY_ROOT / "docs/plans/phase14-graph-ab-execution-order.json"
SOURCE_CAMPAIGN_ID = "phase13-20260822t150835736582z-4ddd7b17-3a8fb3"
SOURCE_CAMPAIGN_ROOT = REPOSITORY_ROOT / "artifacts/phase13" / SOURCE_CAMPAIGN_ID
SOURCE_CAMPAIGN_ROOT_SHA256 = (
    "feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531"
)
SOURCE_PREFIX_ROOT = (
    REPOSITORY_ROOT
    / "artifacts/phase13_prefix_catalogs"
    / SOURCE_CAMPAIGN_ID
    / "catalog"
)
PHASE13D_REPORT = REPOSITORY_ROOT / "docs/phase_reports/phase13d-continuation.md"

CONFIGURATIONS = pilot.CONFIGURATIONS
CONFIG_FINGERPRINTS = pilot.CONFIG_FINGERPRINTS
BATCH_SIZES = (1, 4)
CONTEXT_LABELS = (4096, 16384, 24576, 32768, 65536, 131072)
GRAPH_MODES = ("eager", "cuda_graph")
SEEDS = (20260826, 20260827, 20260828)
REPLICATES = 3
WARMUP_STEPS = 64
MEASURED_STEPS = 128
MEASURED_BATCHES = pilot.MEASURED_BATCHES
CV_THRESHOLD = 0.03
PAIR_COUNT_PER_REPLICATE = len(CONFIGURATIONS) * len(BATCH_SIZES) * len(CONTEXT_LABELS)
PLANNED_PAIR_RECORDS = PAIR_COUNT_PER_REPLICATE * REPLICATES
PLANNED_RUN_RECORDS = PLANNED_PAIR_RECORDS * len(GRAPH_MODES)
AUTHORIZED_CONTAINER_DIGEST = pilot.AUTHORIZED_CONTAINER_DIGEST
GPU_UUID = pilot.GPU_UUID

CAMPAIGN_SCHEMA = "kvbench-phase14-campaign-1.0.0"
ORDER_SCHEMA = "kvbench-phase14-execution-order-1.0.0"
RUN_SCHEMA = "kvbench-phase14-process-run-1.0.0"
WORKER_PREFIX = "PHASE14_WORKER_RESULT="
STAGE_SEQUENCE = (
    ("model_load", "started"),
    ("model_load", "completed"),
    ("prefix_restore", "started"),
    ("prefix_restore", "completed"),
    ("mode_setup", "started"),
    ("mode_setup", "completed"),
    ("warmup_and_audit", "started"),
    ("warmup_and_audit", "completed"),
    ("measurement", "started"),
    ("measurement", "completed"),
    ("finalization", "started"),
    ("finalization", "completed"),
)
_CAMPAIGN_RE = re.compile(
    r"phase14-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)
_RUN_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,179}\Z")


class Phase14Error(RuntimeError):
    """The Phase 14 paired mechanism contract failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


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


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase14Error(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise Phase14Error(f"JSON evidence is not an object: {path}")
    return value


def actual_historical_context(label: int) -> int:
    if label not in CONTEXT_LABELS:
        raise Phase14Error("unknown Phase 14 context label")
    return 131071 if label == 131072 else label


def _point_seed(seed: int, configuration: str) -> int:
    digest = hashlib.sha256(f"{seed}:{configuration}:phase14".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _mode_seed(seed: int, configuration: str, batch: int, label: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{configuration}:{batch}:{label}:mode".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _execution_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    pair_orders: list[dict[str, Any]] = []
    sequence_index = 0
    for replicate_index, seed in enumerate(SEEDS):
        blocks = list(CONFIGURATIONS)
        random.Random(seed).shuffle(blocks)
        pair_order_index = 0
        block_record = {
            "replicate_index": replicate_index,
            "seed": seed,
            "configuration_order": blocks,
            "pairs": [],
        }
        for block_index, configuration in enumerate(blocks):
            points = [
                (batch, label)
                for batch in BATCH_SIZES
                for label in CONTEXT_LABELS
            ]
            random.Random(_point_seed(seed, configuration)).shuffle(points)
            for within_block_index, (batch, label) in enumerate(points):
                modes = list(GRAPH_MODES)
                random.Random(
                    _mode_seed(seed, configuration, batch, label)
                ).shuffle(modes)
                historical = actual_historical_context(label)
                pair_key = (
                    f"r{replicate_index}-{configuration}-b{batch}-l{label}"
                )
                block_record["pairs"].append(
                    {
                        "pair_order_index": pair_order_index,
                        "pair_key": pair_key,
                        "method_config_id": configuration,
                        "batch_size": batch,
                        "context_label": label,
                        "mode_order": modes,
                    }
                )
                for member_index, mode in enumerate(modes):
                    records.append(
                        {
                            "sequence_index": sequence_index,
                            "order_index": pair_order_index * 2 + member_index,
                            "replicate_index": replicate_index,
                            "seed": seed,
                            "block_index": block_index,
                            "within_block_index": within_block_index,
                            "pair_order_index": pair_order_index,
                            "pair_member_index": member_index,
                            "pair_key": pair_key,
                            "method_config_id": configuration,
                            "method_config_fingerprint": CONFIG_FINGERPRINTS[
                                configuration
                            ],
                            "batch_size": batch,
                            "context_label": label,
                            "historical_context": historical,
                            "total_attended_context": historical + 1,
                            "runner_kind": "fixed_l",
                            "graph_mode": mode,
                        }
                    )
                    sequence_index += 1
                pair_order_index += 1
        pair_orders.append(block_record)
    identity = {
        (
            row["replicate_index"],
            row["method_config_id"],
            row["batch_size"],
            row["context_label"],
            row["graph_mode"],
        )
        for row in records
    }
    if (
        len(records) != PLANNED_RUN_RECORDS
        or len(identity) != PLANNED_RUN_RECORDS
        or any(
            records[index]["pair_key"] != records[index + 1]["pair_key"]
            for index in range(0, len(records), 2)
        )
    ):
        raise Phase14Error("Phase 14 paired execution order differs")
    return records, pair_orders


def derive_execution_order() -> dict[str, Any]:
    records, pair_orders = _execution_records()
    payload = {
        "schema_version": ORDER_SCHEMA,
        "seeds": list(SEEDS),
        "blocked_randomization": True,
        "configuration_blocks": True,
        "pair_members_back_to_back": True,
        "mode_first_randomized": True,
        "pair_orders": pair_orders,
        # The complete 360-pair order, including each randomized first mode,
        # is committed directly.  The two adjacent run records per pair are
        # reconstructed deterministically and bound by records_sha256.
        "planned_pair_records": PLANNED_PAIR_RECORDS,
        "planned_run_records": PLANNED_RUN_RECORDS,
    }
    payload["records_sha256"] = _canonical_sha256(records)
    return payload


def validate_execution_order(value: Mapping[str, Any]) -> None:
    if dict(value) != derive_execution_order():
        raise Phase14Error("committed Phase 14 execution order differs")


def write_execution_order(path: Path = ORDER_PATH) -> None:
    if path.exists() or path.is_symlink():
        raise Phase14Error("Phase 14 execution order already exists")
    write_exclusive(path, json_bytes(derive_execution_order()))


def _pilot_feasibility_input(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "replicate_index": record["replicate_index"],
        "seed": record["seed"],
        "block_index": record["block_index"],
        "within_block_index": record["within_block_index"],
        "order_index": record["sequence_index"],
        "method_config_id": record["method_config_id"],
        "method_config_fingerprint": record["method_config_fingerprint"],
        "batch_size": record["batch_size"],
        "context_label": record["context_label"],
        "historical_context": record["historical_context"],
        "total_attended_context": record["total_attended_context"],
        "runner_kind": "fixed_l",
        "graph_mode": "cuda_graph",
    }


def build_feasibility_records(order: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_execution_order(order)
    records: list[dict[str, Any]] = []
    pair_status: dict[str, str] = {}
    expanded, _ = _execution_records()
    for raw in expanded:
        base = pilot.feasibility_record(_pilot_feasibility_input(raw))
        status = (
            "pair_feasible"
            if base["status"] == "feasible"
            else "pair_capacity_infeasible"
        )
        prior = pair_status.setdefault(str(raw["pair_key"]), status)
        if prior != status:
            raise Phase14Error("A/B pair feasibility is asymmetric")
        records.append(
            {
                **base,
                **dict(raw),
                "status": status,
                "pair_feasibility_requires_both_modes": True,
                "feasibility_mode": "cuda_graph_reserve_bounds_pair",
            }
        )
    if (
        len(records) != PLANNED_RUN_RECORDS
        or sum(row["status"] == "pair_feasible" for row in records) != 660
        or sum(
            row["status"] == "pair_capacity_infeasible" for row in records
        )
        != 60
    ):
        raise Phase14Error("Phase 14 feasibility cardinality differs")
    return records


def _source_prefix_index() -> dict[tuple[str, int, int], dict[str, Any]]:
    internal = _strict_json(SOURCE_PREFIX_ROOT / "catalog.json")
    published = _strict_json(
        SOURCE_CAMPAIGN_ROOT / "unified" / "prefix-catalog.json"
    )
    entries = internal.get("entries")
    published_entries = published.get("entries")
    if (
        not isinstance(entries, list)
        or not isinstance(published_entries, list)
        or len(entries) != 228
        or [
            {key: value for key, value in row.items() if key != "snapshot_relative_path"}
            for row in entries
        ]
        != published_entries
    ):
        raise Phase14Error("Phase 13 prefix catalog authority differs")
    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in entries:
        row = dict(raw)
        key = (
            str(row["method_config_id"]),
            int(row["batch_size"]),
            int(row["context_label"]),
        )
        if key[1] not in BATCH_SIZES or key[2] not in CONTEXT_LABELS:
            continue
        if key in index:
            raise Phase14Error("Phase 13 prefix catalog contains duplicates")
        root = SOURCE_PREFIX_ROOT / str(row["snapshot_relative_path"])
        manifest = validate_prefix_state(
            root,
            configuration=key[0],
            family=phase12._method_family(key[0]),
            batch=key[1],
            historical=actual_historical_context(key[2]),
            method_config_fingerprint=CONFIG_FINGERPRINTS[key[0]],
            verify_state_bytes=False,
        )
        state = root / "state.safetensors"
        if (
            manifest.get("state_file_sha256") != row["state_file_sha256"]
            or state.stat().st_size != row["state_file_bytes"]
            or state.stat().st_mode & 0o222
            or root.stat().st_mode & 0o222
        ):
            raise Phase14Error("Phase 13 prefix state authority differs")
        row["snapshot_root"] = str(root)
        index[key] = row
    if len(index) != 110:
        raise Phase14Error("Phase 14 exact prefix-state coverage differs")
    return index


def _source_path_index() -> dict[tuple[str, int, int], dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((SOURCE_CAMPAIGN_ROOT / "runs").glob("*/result.json")):
        result = _strict_json(path)
        key = (
            str(result["method_config_id"]),
            int(result["batch_size"]),
            int(result["context_label"]),
        )
        if key[1] in BATCH_SIZES and key[2] in CONTEXT_LABELS:
            grouped[key].append(result)
    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for key, results in grouped.items():
        witnesses = {
            (
                row["output_checksum"],
                row["graph_path_before"]["normalized_sha256"],
                int(row["kernel_count"]),
                row["runner"]["cache_layout_fingerprint"],
                int(row["runner"]["cache_accounting"]["allocated_bytes"]),
            )
            for row in results
        }
        if len(results) != REPLICATES or len(witnesses) != 1:
            raise Phase14Error("Phase 13 path witness is not replicate-stable")
        output, topology, kernels, layout, allocated = next(iter(witnesses))
        index[key] = {
            "method_config_id": key[0],
            "batch_size": key[1],
            "context_label": key[2],
            "source_campaign_id": SOURCE_CAMPAIGN_ID,
            "source_run_ids": sorted(str(row["run_id"]) for row in results),
            "output_checksum": output,
            "graph_topology_sha256": topology,
            "kernel_count": kernels,
            "cache_layout_fingerprint": layout,
            "allocated_bytes": allocated,
            "timing_reused": False,
            "path_identity_reused": True,
        }
    if len(index) != 110:
        raise Phase14Error("Phase 13 source path witness coverage differs")
    return index


def validate_lightweight_entry() -> dict[str, Any]:
    status = (REPOSITORY_ROOT / "docs/status.md").read_text(encoding="utf-8")
    report = PHASE13D_REPORT.read_text(encoding="utf-8")
    source_manifest = _strict_json(SOURCE_CAMPAIGN_ROOT / "campaign_manifest.json")
    source_qc = _strict_json(SOURCE_CAMPAIGN_ROOT / "pilot_qc.json")
    if (
        "Status: PASS" not in report
        or "Phase 14 readiness: READY" not in report
        or "| G1-G5 unified admission | PASS |" not in status
        or "| Pilot/full-scan gates | PHASE 14 READY / CLOSED |" not in status
        or "quality is LOCKED" not in status
        or source_manifest.get("fingerprints") != CONFIG_FINGERPRINTS
        or source_qc.get("phase13_status") != "LOCAL_PASS_PENDING_PUBLICATION"
    ):
        raise Phase14Error("recorded Phase 14 entry state differs")
    image = subprocess.run(
        ("docker", "image", "inspect", AUTHORIZED_CONTAINER_DIGEST, "--format", "{{.Id}}"),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if image.returncode != 0 or image.stdout.strip() != AUTHORIZED_CONTAINER_DIGEST:
        raise Phase14Error("authorized Measurement Container is unavailable")
    return {
        "status": "PASS",
        "phase13d_continuation": "PASS",
        "phase14_readiness": "READY",
        "gates": "G0-G5 PASS",
        "full_scan": "CLOSED",
        "quality": "LOCKED",
        "container": AUTHORIZED_CONTAINER_DIGEST,
        "fingerprints": dict(CONFIG_FINGERPRINTS),
    }


@dataclasses.dataclass(frozen=True, slots=True)
class Phase14OperationKey:
    configuration: str
    runner_kind: RunnerKind
    graph_mode: GraphMode
    historical_context: int
    attended_context: int
    batch_size: int
    capacity: int
    decode_step: int
    operation_fingerprint_sha256: str

    @classmethod
    def create(
        cls, configuration: str, batch: int, historical: int, graph_mode: str
    ) -> "Phase14OperationKey":
        try:
            mode = GraphMode(graph_mode)
        except ValueError as error:
            raise Phase14Error("Phase 14 graph mode differs") from error
        payload = {
            "schema_version": "kvbench-phase14-operation-key-1.0.0",
            "configuration": configuration,
            "runner_kind": "fixed_l",
            "graph_mode": mode.value,
            "historical_context": historical,
            "attended_context": historical + 1,
            "batch_size": batch,
            "capacity": historical + 1,
            "decode_step": 0,
            "input_recipe_schema": pilot.INPUT_RECIPE_SCHEMA,
        }
        return cls(
            configuration=configuration,
            runner_kind=RunnerKind.FIXED_L,
            graph_mode=mode,
            historical_context=historical,
            attended_context=historical + 1,
            batch_size=batch,
            capacity=historical + 1,
            decode_step=0,
            operation_fingerprint_sha256=_canonical_sha256(payload),
        )


class _EagerPassthrough:
    """Phase-14-local stand-in proving no CUDAGraph is created for Graph OFF."""

    def __init__(self, operation: Any) -> None:
        self._operation = operation

    def replay(self) -> Any:
        return self._operation()

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured": False,
            "fallback": False,
            "warmup_steps": 0,
            "graph_mode": "eager",
            "phase14_eager_passthrough": True,
        }


@contextmanager
def _graph_capture_disabled_for_eager() -> Iterator[None]:
    import kvbench.runtime.cuda_graph as module

    original = module.capture_fixed_graph

    def eager_passthrough(
        operation: Any, *, warmup_steps: int, device: Any
    ) -> _EagerPassthrough:
        del device
        if warmup_steps != 0:
            raise Phase14Error("eager setup attempted Graph warmup")
        return _EagerPassthrough(operation)

    module.capture_fixed_graph = eager_passthrough
    try:
        yield
    finally:
        module.capture_fixed_graph = original


def _build_mode_session(
    *,
    loaded: Any,
    operation: Phase14OperationKey,
    prefix: Any,
    decode: Any,
    snapshot_root: Path,
    state_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    session, receipt = pilot._build_restored_session(
        loaded=loaded,
        operation=operation,
        prefix=prefix,
        decode=decode,
        snapshot_root=snapshot_root,
        expected_state_sha256=state_sha256,
    )
    if operation.graph_mode is GraphMode.EAGER:
        fake = session.graph
        if not isinstance(fake, _EagerPassthrough):
            raise Phase14Error("eager setup created a CUDA Graph")
        session._graph = None
        session.graph_evidence = None
    return session, receipt


def _optional_telemetry_range(
    before: Mapping[str, Any], after: Mapping[str, Any], key: str
) -> tuple[float | None, float | None]:
    values = (before.get(key), after.get(key))
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in values
    ):
        return None, None
    converted = tuple(float(value) for value in values)
    return min(converted), max(converted)


def _worker_owned_snapshot(
    *, run_root: Path, pid: int, start_ticks: int
) -> dict[str, Any]:
    evidence = run_root / "snapshots" / "worker-owned"
    evidence.mkdir(parents=True)
    command = (
        sys.executable,
        str(REPOSITORY_ROOT / "preflight/process_query.py"),
        "--supervised-root-pid",
        str(pid),
        "--supervised-root-start-ticks",
        str(start_ticks),
    )
    final: dict[str, Any] | None = None
    for attempt in range(continuation.SNAPSHOT_RETRIES + 1):
        invocation = continuation._invoke_snapshot_command(command)
        state, parsed, parser_exception, process_list = (
            continuation._classify_snapshot_attempt(invocation=invocation)
        )
        if state == "clean":
            allowed = parsed.get("allowed_compute_processes") if parsed else None
            owned = bool(
                isinstance(allowed, list)
                and len(allowed) == 1
                and allowed[0].get("gpu_uuid") == GPU_UUID
                and allowed[0].get("pid") == pid
                and allowed[0].get("process_start_time_ticks") == start_ticks
                and allowed[0].get("relationship") == "supervised_child"
            )
            if not owned:
                state = "foreign_process_detected"
        payload = {
            "schema_version": "kvbench-phase14-worker-snapshot-1.0.0",
            "timestamp": _utc_now(),
            "attempt": attempt,
            "command": list(command),
            "return_code": invocation["return_code"],
            "raw_stdout": invocation["raw_stdout"],
            "raw_stderr": invocation["raw_stderr"],
            "invocation_exception": invocation["invocation_exception"],
            "parser_result": parsed,
            "parser_exception": parser_exception,
            "process_list": process_list,
            "state": state,
        }
        write_exclusive(evidence / f"attempt-{attempt:02d}.json", json_bytes(payload))
        final = payload
        if state != "query_failed":
            break
    assert final is not None
    return final


def _stage_timeouts(*, batch: int, historical: int) -> dict[str, float]:
    return {
        "startup": 600.0,
        "transition": 600.0,
        "model_load": 900.0,
        "prefix_restore": pilot.prefix_construction_timeout_seconds(
            batch=batch, historical=historical
        ),
        "mode_setup": pilot.graph_capture_timeout_seconds(
            batch=batch, historical=historical
        ),
        "warmup_and_audit": pilot.warmup_audit_timeout_seconds(
            batch=batch, historical=historical
        ),
        "measurement": 7200.0,
        "finalization": 1800.0,
    }


def _run_worker(
    *,
    run_id: str,
    configuration: str,
    batch: int,
    context_label: int,
    graph_mode: str,
    replicate_index: int,
    order_index: int,
    git_sha: str,
    run_artifact_root: Path,
    prefix_state_root: Path,
    prefix_state_sha256: str,
    source_witness_path: Path,
) -> dict[str, Any]:
    attestation = phase12._require_authorized_container_runtime()
    pid = os.getpid()
    start_ticks = phase12._process_start_ticks(pid)

    import torch

    from kvbench.runtime.allocation import audit_cuda_allocations
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.fixed_l_runner import run_fixed_l
    from kvbench.runtime.model_loader import load_frozen_model
    from kvbench.runtime.numerical import tensor_sha256_untimed
    from kvbench.runtime.timing import warmup_operations

    observed_head = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    observed_status = subprocess.run(
        ("/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or observed_head != git_sha
        or observed_status
    ):
        raise Phase14Error("worker execution authority differs")
    historical = actual_historical_context(context_label)
    pilot._patch_phase12_point_globals(batch=batch, historical=historical)
    recorder = pilot._WorkerStageRecorder(
        root=run_artifact_root / "stage-progress",
        run_id=run_id,
        stage_sequence=STAGE_SEQUENCE,
    )
    recorder.record("model_load", "started")
    loaded = load_frozen_model(device=torch.device("cuda:0"))
    recorder.record("model_load", "completed")
    prefix, decode = pilot._point_inputs(
        batch=batch, historical=historical, device=torch.device("cuda:0")
    )
    operation = Phase14OperationKey.create(
        configuration, batch, historical, graph_mode
    )
    source = _strict_json(source_witness_path)
    source_key = f"{configuration}/B{batch}/L{context_label}"
    source_witness = source.get("records", {}).get(source_key)
    if not isinstance(source_witness, dict):
        raise Phase14Error("source path witness is absent")
    recorder.record("prefix_restore", "started")
    import kvbench.runtime.cuda_graph as graph_module

    original_capture = graph_module.capture_fixed_graph
    capture_started = False

    def observed_capture(*args: Any, **kwargs: Any) -> Any:
        nonlocal capture_started
        if capture_started:
            raise Phase14Error("mode setup capture invoked more than once")
        recorder.record("prefix_restore", "completed")
        recorder.record("mode_setup", "started")
        capture_started = True
        if graph_mode == "cuda_graph":
            return original_capture(*args, **kwargs)
        warmup_steps = kwargs.get("warmup_steps")
        if warmup_steps is None and len(args) >= 2:
            warmup_steps = args[1]
        if warmup_steps != 0:
            raise Phase14Error("eager setup attempted Graph warmup")
        if not args:
            raise Phase14Error("eager setup operation is absent")
        return _EagerPassthrough(args[0])

    graph_module.capture_fixed_graph = observed_capture
    try:
        with torch.inference_mode(), forced_flash_execution():
            if graph_mode == "cuda_graph":
                with phase12._observable_cuda_graph_factory(torch) as observed_graphs:
                    session, restore = _build_mode_session(
                        loaded=loaded,
                        operation=operation,
                        prefix=prefix,
                        decode=decode,
                        snapshot_root=prefix_state_root,
                        state_sha256=prefix_state_sha256,
                    )
            else:
                observed_graphs = []
                session, restore = _build_mode_session(
                    loaded=loaded,
                    operation=operation,
                    prefix=prefix,
                    decode=decode,
                    snapshot_root=prefix_state_root,
                    state_sha256=prefix_state_sha256,
                )
    finally:
        graph_module.capture_fixed_graph = original_capture
    if graph_mode == "cuda_graph":
        if (
            not capture_started
            or session.graph is None
            or len(observed_graphs) != 1
            or session.graph.graph is not observed_graphs[0]
        ):
            raise Phase14Error("CUDA Graph setup differs")
    elif not capture_started or session.graph is not None or observed_graphs:
        raise Phase14Error("Graph OFF initialized a CUDA Graph")
    recorder.record("mode_setup", "completed")
    recorder.record("warmup_and_audit", "started")
    pointers_before = phase12._phase12_session_pointers(session)
    history_before = session.current_historical_prefix_sha256()
    graph_path_before = graph_path_after = None
    graph_exec_before = None
    if graph_mode == "cuda_graph":
        graph_exec_before = int(session.graph.graph.raw_cuda_graph_exec())
        graph_path_before = phase12._write_cuda_graph_path_witness(
            graph=session.graph.graph, run_root=run_artifact_root, phase="before"
        )
        operation_callable = session.graph.replay
    else:
        operation_callable = lambda: session.execute_audit_step(0)
    warm_output = warmup_operations(
        operation_callable, count=WARMUP_STEPS, device=session.cache_device
    )
    warm_cpu = warm_output.detach().to(device="cpu", copy=True).clone()
    warm_checksum = tensor_sha256_untimed(warm_cpu)
    warm_finite = bool(torch.isfinite(warm_cpu).all())
    allocation_audit = audit_cuda_allocations(
        operation_callable, device=session.cache_device
    )
    audit_output = operation_callable().detach().to(device="cpu", copy=True).clone()
    torch.cuda.synchronize(device=session.cache_device)
    audit_checksum = tensor_sha256_untimed(audit_output)
    audit_finite = bool(torch.isfinite(audit_output).all())
    allocation_record = allocation_audit.to_dict()
    allocation_passed = bool(
        allocation_audit.audit_available
        and allocation_audit.passed
        and allocation_audit.allocated_after == allocation_audit.allocated_before
        and allocation_audit.reserved_after == allocation_audit.reserved_before
    )
    if graph_mode == "cuda_graph":
        graph_passed = bool(
            session.graph_evidence is not None
            and session.graph_evidence.get("captured") is True
            and session.graph_evidence.get("fallback") is False
            and session.graph_evidence.get("consecutive_replay_outputs_exact") is True
            and session.eager_graph_comparison is not None
            and session.eager_graph_comparison.passed
            and allocation_audit.allocation_event_count == 0
            and allocation_audit.allocation_event_bytes == 0
        )
    else:
        graph_passed = True
    family = phase12._method_family(configuration)
    geometry = session.gqa_cache_geometry()
    live_fingerprint = phase12._validate_runtime_adapter_fingerprint(
        method=session.method,
        cache=session.cache,
        observed=session.adapter_config_fingerprint,
    )
    runtime_context = session.method.runtime_context
    backend_verified = bool(
        runtime_context.model_id == phase12.PHASE12_MODEL_ID
        and runtime_context.model_revision == phase12.PHASE12_MODEL_REVISION
        and runtime_context.num_layers == 32
        and runtime_context.num_query_heads == 32
        and runtime_context.num_kv_heads == 8
        and runtime_context.head_dim == 128
        and re.fullmatch(r"[0-9a-f]{64}", runtime_context.backend_fingerprint)
    )
    path_passed = bool(
        backend_verified
        and live_fingerprint
        and phase12._gqa_geometry_passes(geometry, family=family)
        and pointers_before == phase12._phase12_session_pointers(session)
        and graph_passed
    )
    if (
        not warm_finite
        or not audit_finite
        or warm_checksum != audit_checksum
        or audit_checksum != source_witness["output_checksum"]
        or pointers_before != phase12._phase12_session_pointers(session)
        or history_before != session.current_historical_prefix_sha256()
        or session.cache_layout_fingerprint()
        != source_witness["cache_layout_fingerprint"]
        or session.method_cache_accounting()["allocated_bytes"]
        != source_witness["allocated_bytes"]
    ):
        raise Phase14Error("mode setup numerical, layout, or allocation drifted")
    if session.graph_evidence is not None:
        session.graph_evidence["replay_allocation"] = allocation_record
        session.graph_evidence["phase14_warmup_replays"] = WARMUP_STEPS
    session.admit(
        observed_outputs=((audit_checksum, audit_finite),),
        execution_path_passed=path_passed,
        allocation_passed=allocation_passed,
        graph_passed=graph_passed,
    )
    if graph_mode == "eager":
        session.eager_graph_comparison = None
    recorder.record("warmup_and_audit", "completed")
    recorder.record("measurement", "started")
    raw_runner = run_fixed_l(
        session,
        measured_steps=MEASURED_STEPS,
        measured_batches=MEASURED_BATCHES,
    ).to_dict()
    recorder.record("measurement", "completed")
    recorder.record("finalization", "started")
    runner = phase12._normalize_runner_result(raw_runner)
    if graph_mode == "cuda_graph":
        graph_exec_after = int(session.graph.graph.raw_cuda_graph_exec())
        graph_path_after = phase12._write_cuda_graph_path_witness(
            graph=session.graph.graph, run_root=run_artifact_root, phase="after"
        )
        if (
            graph_exec_before is None
            or graph_exec_before <= 0
            or graph_exec_after != graph_exec_before
            or graph_path_before["normalized_sha256"]
            != graph_path_after["normalized_sha256"]
            or graph_path_before["normalized_sha256"]
            != source_witness["graph_topology_sha256"]
        ):
            raise Phase14Error("measured CUDA Graph topology drifted")
    memory = runner.get("memory_evidence")
    if (
        runner.get("graph_mode") != graph_mode
        or runner.get("output_finite") is not True
        or runner.get("output_checksum") != audit_checksum
        or runner.get("cache_pointers_stable") is not True
        or runner.get("historical_cache_unchanged") is not True
        or not isinstance(memory, Mapping)
        or memory.get("timing_allocated_delta_bytes") != 0
        or memory.get("timing_reserved_delta_bytes") != 0
        or runner.get("r_hbm") is not None
    ):
        raise Phase14Error("Phase 14 timing stability or allocation failed")
    samples = runner["timing"]["samples"]
    wall_values = [
        float(item["host_ns_per_operation"]) / 1_000_000.0 for item in samples
    ]
    cuda_values = [float(item["cuda_ms_per_operation"]) for item in samples]
    ratios = [wall / cuda for wall, cuda in zip(wall_values, cuda_values)]
    if any(not math.isfinite(value) or value <= 0 for value in (*wall_values, *cuda_values)):
        raise Phase14Error("Phase 14 timing sample is invalid")
    telemetry = {}
    for key, out_key in (
        ("temperature_celsius", "temperature"),
        ("sm_clock_mhz", "sm_clock"),
        ("memory_clock_mhz", "memory_clock"),
        ("power_watts", "power"),
    ):
        telemetry[out_key] = _optional_telemetry_range(
            runner["telemetry_before"], runner["telemetry_after"], key
        )
    owned = _worker_owned_snapshot(
        run_root=run_artifact_root, pid=pid, start_ticks=start_ticks
    )
    if owned["state"] == "foreign_process_detected":
        raise Phase14Error("foreign GPU process detected after measurement")
    cache_identity = _canonical_sha256(
        {
            "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
            "cache_layout_fingerprint": runner["cache_layout_fingerprint"],
            "cache_accounting": runner["cache_accounting"],
            "cache_byte_breakdown": runner["cache_byte_breakdown"],
        }
    )
    semantic_path = _canonical_sha256(
        {
            "configuration": configuration,
            "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
            "batch_size": batch,
            "context_label": context_label,
            "backend_fingerprint": runtime_context.backend_fingerprint,
            "cache_layout_fingerprint": runner["cache_layout_fingerprint"],
            "source_graph_topology_sha256": source_witness[
                "graph_topology_sha256"
            ],
            "kernel_count": source_witness["kernel_count"],
            "prefix_state_sha256": prefix_state_sha256,
        }
    )
    payload = {
        "schema_version": RUN_SCHEMA,
        "run_id": run_id,
        "method_config_id": configuration,
        "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
        "method_family": family,
        "replicate_index": replicate_index,
        "order_index": order_index,
        "batch_size": batch,
        "context_label": context_label,
        "historical_context": historical,
        "total_attended_context": historical + 1,
        "graph_mode": graph_mode,
        "execution_git_sha": git_sha,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "gpu_uuid": GPU_UUID,
        "operation_fingerprint_sha256": operation.operation_fingerprint_sha256,
        "process_wall_median_ms": statistics.median(wall_values),
        "process_cuda_median_ms": statistics.median(cuda_values),
        "host_wall_cuda_event_ratio": statistics.median(ratios),
        "kernel_count": int(source_witness["kernel_count"]),
        "kernel_count_source": (
            "current_graph_debug_witness"
            if graph_mode == "cuda_graph"
            else "paired_source_graph_path_authority"
        ),
        "output_checksum": runner["output_checksum"],
        "semantic_kernel_path_fingerprint": semantic_path,
        "cache_identity_fingerprint": cache_identity,
        "mode_allocation_fingerprint": _canonical_sha256(
            {"graph_mode": graph_mode, "audit": allocation_record, "cache": cache_identity}
        ),
        "temperature_min_c": telemetry["temperature"][0],
        "temperature_max_c": telemetry["temperature"][1],
        "sm_clock_min_mhz": telemetry["sm_clock"][0],
        "sm_clock_max_mhz": telemetry["sm_clock"][1],
        "memory_clock_min_mhz": telemetry["memory_clock"][0],
        "memory_clock_max_mhz": telemetry["memory_clock"][1],
        "power_min_w": telemetry["power"][0],
        "power_max_w": telemetry["power"][1],
        "telemetry_available": all(value[0] is not None for value in telemetry.values()),
        "finite_output": True,
        "no_backend_fallback": path_passed,
        "allocation_stable": True,
        "kernel_path_stable": True,
        "gpu_exclusive": owned["state"] == "clean",
        "worker_owned_snapshot_state": owned["state"],
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "measured_batches": MEASURED_BATCHES,
        "allocation_audit": allocation_record,
        "prefix_state_restore": restore,
        "source_path_witness": source_witness,
        "graph_capture": graph_mode == "cuda_graph",
        "graph_replay_allocation_zero": (
            graph_mode != "cuda_graph"
            or (
                allocation_audit.allocation_event_count == 0
                and allocation_audit.allocation_event_bytes == 0
            )
        ),
        "graph_path_before": graph_path_before,
        "graph_path_after": graph_path_after,
        "runner": runner,
        "container_runtime_attestation": attestation,
        "mechanism_experiment": True,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    recorder.record("finalization", "completed")
    return payload


def new_campaign_id(git_sha: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise Phase14Error("execution Git SHA is invalid")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:21]
    return f"phase14-{stamp}z-{git_sha[:8]}-{secrets.token_hex(3)}"


def _validate_campaign_id(value: str) -> str:
    if _CAMPAIGN_RE.fullmatch(value) is None:
        raise Phase14Error("Phase 14 campaign ID is invalid")
    return value


def reserve_campaign(*, campaign_id: str, git_sha: str) -> Path:
    identifier = _validate_campaign_id(campaign_id)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    stage = STAGING_ROOT / f"{identifier}.{secrets.token_hex(4)}.staging"
    stage.mkdir()
    for relative in ("runs", "unified", "plots"):
        (stage / relative).mkdir()
    write_exclusive(
        stage / "campaign-reservation.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase14-reservation-1.0.0",
                "campaign_id": identifier,
                "execution_git_sha": git_sha,
                "reserved_at_utc": _utc_now(),
                "append_only": True,
            }
        ),
    )
    return stage.resolve(strict=True)


def _run_id(campaign_id: str, record: Mapping[str, Any]) -> str:
    value = (
        f"{campaign_id}-r{record['replicate_index']}-"
        f"o{int(record['order_index']):03d}-{record['method_config_id']}-"
        f"b{record['batch_size']}-l{record['context_label']}-{record['graph_mode']}"
    )
    if _RUN_RE.fullmatch(value) is None:
        raise Phase14Error("Phase 14 run ID is invalid")
    return value


def _write_manifest(
    *,
    run_root: Path,
    campaign_id: str,
    run_id: str,
    record: Mapping[str, Any],
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "kvbench-phase14-run-manifest-1.0.0",
        "campaign_id": campaign_id,
        "run_id": run_id,
        "status": status,
        "reason": reason,
        "sequence_index": record["sequence_index"],
        "pair_key": record["pair_key"],
        "pair_order_index": record["pair_order_index"],
        "pair_member_index": record["pair_member_index"],
        "method_config_id": record["method_config_id"],
        "method_config_fingerprint": record["method_config_fingerprint"],
        "replicate_index": record["replicate_index"],
        "seed": record["seed"],
        "batch_size": record["batch_size"],
        "context_label": record["context_label"],
        "historical_context": record["historical_context"],
        "total_attended_context": record["total_attended_context"],
        "runner_kind": "fixed_l",
        "graph_mode": record["graph_mode"],
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "measured_batches": MEASURED_BATCHES,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "selective_rerun": False,
        "mechanism_experiment": True,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    write_exclusive(run_root / "manifest.json", json_bytes(payload))
    return payload


def _write_disposition(
    *, run_root: Path, run_id: str, status: str, reason: str, launched: bool
) -> None:
    write_exclusive(
        run_root / "disposition.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase14-disposition-1.0.0",
                "run_id": run_id,
                "status": status,
                "reason": reason,
                "cuda_process_launched": launched,
                "record_preserved": True,
            }
        ),
    )


def _write_nonlaunched(
    *,
    stage: Path,
    campaign_id: str,
    record: Mapping[str, Any],
    status: str,
    reason: str,
) -> dict[str, Any]:
    run_id = _run_id(campaign_id, record)
    root = stage / "runs" / run_id
    root.mkdir()
    _write_disposition(
        run_root=root, run_id=run_id, status=status, reason=reason, launched=False
    )
    return _write_manifest(
        run_root=root,
        campaign_id=campaign_id,
        run_id=run_id,
        record=record,
        status=status,
        reason=reason,
    )


def _write_supervision(
    *, root: Path, result: Any, pre: Any, post: Any
) -> None:
    write_exclusive(root / "worker.stdout", result.stdout)
    write_exclusive(root / "worker.stderr", result.stderr)
    write_exclusive(
        root / "worker.supervision.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase14-worker-supervision-1.0.0",
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "timeout_stage": result.timeout_stage,
                "timeout_stage_elapsed_seconds": result.timeout_stage_elapsed_seconds,
                "preflight_state": pre.state,
                "preflight_attempt": pre.attempt,
                "preflight_classification_path": pre.classification_path,
                "postflight_state": post.state,
                "postflight_attempt": post.attempt,
                "postflight_classification_path": post.classification_path,
            }
        ),
    )


def _extract_worker(result: Any, run_id: str) -> dict[str, Any]:
    matches = [
        line[len(WORKER_PREFIX) :]
        for line in result.stdout.decode("utf-8", errors="strict").splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(matches) != 1:
        raise Phase14Error("worker result channel differs")
    payload = json.loads(matches[0])
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise Phase14Error("worker result identity differs")
    return payload


def _run_one(
    *,
    stage: Path,
    campaign_id: str,
    record: Mapping[str, Any],
    git_sha: str,
    prefix_entry: Mapping[str, Any],
    source_witness_path: Path,
) -> tuple[dict[str, Any], bool]:
    run_id = _run_id(campaign_id, record)
    root = stage / "runs" / run_id
    root.mkdir()
    (root / "stage-progress").mkdir()
    write_exclusive(
        root / "started.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase14-run-start-1.0.0",
                "run_id": run_id,
                "started_at_utc": _utc_now(),
                "record": dict(record),
            }
        ),
    )
    pre = continuation.capture_persisted_snapshot(
        evidence_root=root / "snapshots" / "preflight", phase="preflight"
    )
    pre_action = continuation.snapshot_policy(phase="preflight", state=pre.state)
    if pre_action != "start_run":
        status = (
            "foreign_process_detected"
            if pre.state == "foreign_process_detected"
            else "preflight_snapshot_unavailable"
        )
        reason = f"preflight_snapshot:{pre.state}"
        _write_disposition(
            run_root=root, run_id=run_id, status=status, reason=reason, launched=False
        )
        manifest = _write_manifest(
            run_root=root,
            campaign_id=campaign_id,
            run_id=run_id,
            record=record,
            status=status,
            reason=reason,
        )
        return manifest, pre_action == "abort_campaign"
    command = (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/phase14_graph_ab.py"),
        "--run-worker",
        "--run-id",
        run_id,
        "--configuration",
        str(record["method_config_id"]),
        "--batch-size",
        str(record["batch_size"]),
        "--context-label",
        str(record["context_label"]),
        "--graph-mode",
        str(record["graph_mode"]),
        "--replicate-index",
        str(record["replicate_index"]),
        "--order-index",
        str(record["order_index"]),
        "--git-sha",
        git_sha,
        "--run-artifact-root",
        str(root),
        "--prefix-state-root",
        str(prefix_entry["snapshot_root"]),
        "--prefix-state-sha256",
        str(prefix_entry["state_file_sha256"]),
        "--source-witness-path",
        str(source_witness_path),
    )
    result = run_stage_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=phase12._child_environment(),
        stage_timeouts=_stage_timeouts(
            batch=int(record["batch_size"]),
            historical=int(record["historical_context"]),
        ),
        stage_observer=lambda: pilot._read_stage_observations(
            root=root / "stage-progress", run_id=run_id, stage_sequence=STAGE_SEQUENCE
        ),
        startup_stage="startup",
        transition_stage="transition",
        observer_poll_seconds=pilot.STAGE_OBSERVER_POLL_SECONDS,
    )
    post = continuation.capture_persisted_snapshot(
        evidence_root=root / "snapshots" / "postflight", phase="postflight"
    )
    _write_supervision(root=root, result=result, pre=pre, post=post)
    if not pilot._supervision_passed(result):
        stderr = result.stderr.decode("utf-8", errors="replace")
        reason = (
            f"supervisor_stage_timeout:{result.timeout_stage}"
            if result.timeout_stage is not None
            else "foreign_gpu_process_detected"
            if "foreign GPU process" in stderr
            else "supervised_worker_failed"
        )
        _write_disposition(
            run_root=root,
            run_id=run_id,
            status="runtime_failed",
            reason=reason,
            launched=True,
        )
        manifest = _write_manifest(
            run_root=root,
            campaign_id=campaign_id,
            run_id=run_id,
            record=record,
            status="runtime_failed",
            reason=reason,
        )
        return manifest, True
    payload = _extract_worker(result, run_id)
    write_exclusive(root / "result.json", json_bytes(payload))
    post_action = continuation.snapshot_policy(phase="postflight", state=post.state)
    worker_snapshot = payload.get("worker_owned_snapshot_state")
    if worker_snapshot == "foreign_process_detected" or post_action == "invalidate_current_and_abort":
        status, reason, abort = (
            "foreign_process_detected",
            "post_measurement_foreign_process",
            True,
        )
    elif worker_snapshot == "query_failed" or post_action == "exclude_current_and_continue":
        status, reason, abort = (
            "postflight_snapshot_unavailable",
            "post_measurement_snapshot_query_failed",
            False,
        )
        _write_disposition(
            run_root=root, run_id=run_id, status=status, reason=reason, launched=True
        )
    else:
        status, reason, abort = "completed", None, False
    manifest = _write_manifest(
        run_root=root,
        campaign_id=campaign_id,
        run_id=run_id,
        record=record,
        status=status,
        reason=reason,
    )
    write_exclusive(
        root / "result-binding.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase14-result-binding-1.0.0",
                "run_id": run_id,
                "result_sha256": sha256_file(root / "result.json"),
            }
        ),
    )
    return manifest, abort


def run_campaign(
    *, stage: Path, campaign_id: str, git_sha: str, prefix_root: Path
) -> dict[str, Any]:
    identifier = _validate_campaign_id(campaign_id)
    phase12._require_authorized_container_runtime()
    if any(name in os.environ for name in pilot._FORBIDDEN_ENVIRONMENT):
        raise Phase14Error("credentials entered the Measurement Container")
    root = stage.resolve(strict=True)
    head = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != git_sha or status:
        raise Phase14Error("container source is not the clean execution commit")
    order = _strict_json(ORDER_PATH)
    validate_execution_order(order)
    feasibility = build_feasibility_records(order)
    if prefix_root.resolve(strict=True) != SOURCE_PREFIX_ROOT.resolve(strict=True):
        raise Phase14Error("Phase 14 prefix root differs")
    prefix_index = _source_prefix_index()
    source_index = _source_path_index()
    source_payload = {
        "schema_version": "kvbench-phase14-source-path-authority-1.0.0",
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "source_campaign_root_sha256": SOURCE_CAMPAIGN_ROOT_SHA256,
        "timing_reused": False,
        "records": {
            f"{key[0]}/B{key[1]}/L{key[2]}": value
            for key, value in sorted(source_index.items())
        },
    }
    source_path = root / "unified" / "source-path-authority.json"
    write_exclusive(source_path, json_bytes(source_payload))
    write_exclusive(root / "execution_order.json", json_bytes(order))
    write_exclusive(root / "feasibility.json", json_bytes({"records": feasibility}))
    write_exclusive(
        root / "campaign_manifest.json",
        json_bytes(
            {
                "schema_version": CAMPAIGN_SCHEMA,
                "campaign_id": identifier,
                "execution_git_sha": git_sha,
                "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
                "source_phase13_campaign_id": SOURCE_CAMPAIGN_ID,
                "source_phase13_root_sha256": SOURCE_CAMPAIGN_ROOT_SHA256,
                "source_timing_reused": False,
                "prefix_states_reused_read_only": True,
                "same_prefix_state_within_pair": True,
                "fresh_session_per_mode_process": True,
                "configurations": list(CONFIGURATIONS),
                "fingerprints": dict(CONFIG_FINGERPRINTS),
                "batch_sizes": list(BATCH_SIZES),
                "context_labels": list(CONTEXT_LABELS),
                "graph_modes": list(GRAPH_MODES),
                "warmup_steps": WARMUP_STEPS,
                "measured_steps": MEASURED_STEPS,
                "measured_batches": MEASURED_BATCHES,
                "replicates": REPLICATES,
                "seeds": list(SEEDS),
                "planned_pair_records": PLANNED_PAIR_RECORDS,
                "planned_run_records": PLANNED_RUN_RECORDS,
                "pair_members_back_to_back": True,
                "mode_first_randomized": True,
                "execution_order_sha256": sha256_file(root / "execution_order.json"),
                "source_path_authority_sha256": sha256_file(source_path),
                "selective_reruns": 0,
                "full_scan": "CLOSED",
                "quality_execution": "LOCKED",
                "performance_data_frozen": False,
            }
        ),
    )
    manifests: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for record in feasibility:
        if record["status"] == "pair_capacity_infeasible":
            manifests.append(
                _write_nonlaunched(
                    stage=root,
                    campaign_id=identifier,
                    record=record,
                    status="pair_capacity_infeasible",
                    reason=str(record["reason"]),
                )
            )
            continue
        if stop_reason is not None:
            manifests.append(
                _write_nonlaunched(
                    stage=root,
                    campaign_id=identifier,
                    record=record,
                    status="aborted",
                    reason=stop_reason,
                )
            )
            continue
        key = (
            str(record["method_config_id"]),
            int(record["batch_size"]),
            int(record["context_label"]),
        )
        entry = prefix_index.get(key)
        if entry is None:
            raise Phase14Error("feasible pair lacks an exact prefix state")
        manifest, abort = _run_one(
            stage=root,
            campaign_id=identifier,
            record=record,
            git_sha=git_sha,
            prefix_entry=entry,
            source_witness_path=source_path,
        )
        manifests.append(manifest)
        if abort:
            stop_reason = str(manifest["reason"])
    counts: dict[str, int] = defaultdict(int)
    for manifest in manifests:
        counts[str(manifest["status"])] += 1
    result = {
        "schema_version": "kvbench-phase14-local-campaign-result-1.0.0",
        "campaign_id": identifier,
        "execution_git_sha": git_sha,
        "planned_pair_records": PLANNED_PAIR_RECORDS,
        "planned_run_records": PLANNED_RUN_RECORDS,
        "status_counts": dict(sorted(counts.items())),
        "stop_reason": stop_reason,
        "selective_reruns": 0,
        "phase14_status": "BLOCKED" if stop_reason else "LOCAL_COMPLETE",
        "durable_publication": "PENDING_HOST_SIDE",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "r_hbm": None,
    }
    write_exclusive(root / "unified" / "local-campaign.json", json_bytes(result))
    return result


def _run_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "runs").glob("*/manifest.json")):
        manifest = _strict_json(manifest_path)
        result_path = manifest_path.parent / "result.json"
        result = _strict_json(result_path) if result_path.is_file() else None
        records.append(
            {
                **manifest,
                "manifest_path": manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "result_path": (
                    result_path.relative_to(root).as_posix() if result else None
                ),
                "result_sha256": sha256_file(result_path) if result else None,
                **(
                    {
                        key: result.get(key)
                        for key in (
                            "process_wall_median_ms",
                            "process_cuda_median_ms",
                            "host_wall_cuda_event_ratio",
                            "kernel_count",
                            "output_checksum",
                            "semantic_kernel_path_fingerprint",
                            "cache_identity_fingerprint",
                            "mode_allocation_fingerprint",
                            "temperature_min_c",
                            "temperature_max_c",
                            "sm_clock_min_mhz",
                            "sm_clock_max_mhz",
                            "memory_clock_min_mhz",
                            "memory_clock_max_mhz",
                            "power_min_w",
                            "power_max_w",
                            "telemetry_available",
                            "finite_output",
                            "no_backend_fallback",
                            "allocation_stable",
                            "kernel_path_stable",
                            "gpu_exclusive",
                            "graph_replay_allocation_zero",
                            "runner",
                        )
                    }
                    if result
                    else {}
                ),
            }
        )
    if len(records) != PLANNED_RUN_RECORDS:
        raise Phase14Error("Phase 14 run-record cardinality differs")
    return records


def _range(rows: Sequence[Mapping[str, Any]], low: str, high: str) -> list[float] | None:
    lows = [float(row[low]) for row in rows if isinstance(row.get(low), (int, float))]
    highs = [float(row[high]) for row in rows if isinstance(row.get(high), (int, float))]
    return [min(lows), max(highs)] if lows and highs else None


def _point_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[
            (
                str(row["method_config_id"]),
                int(row["batch_size"]),
                int(row["context_label"]),
                str(row["graph_mode"]),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        for batch in BATCH_SIZES:
            for label in CONTEXT_LABELS:
                for mode in GRAPH_MODES:
                    matching = grouped[(configuration, batch, label, mode)]
                    completed = [row for row in matching if row["status"] == "completed"]
                    statuses = [str(row["status"]) for row in matching]
                    byte_features = pilot._point_byte_features(completed)
                    if len(completed) == REPLICATES:
                        wall_stats = pilot.point_statistics(
                            [float(row["process_wall_median_ms"]) for row in completed]
                        )
                        cuda_stats = pilot.point_statistics(
                            [float(row["process_cuda_median_ms"]) for row in completed]
                        )
                        agreements = bool(
                            len({row["output_checksum"] for row in completed}) == 1
                            and len(
                                {
                                    row["semantic_kernel_path_fingerprint"]
                                    for row in completed
                                }
                            )
                            == 1
                            and len(
                                {row["cache_identity_fingerprint"] for row in completed}
                            )
                            == 1
                            and len(
                                {row["mode_allocation_fingerprint"] for row in completed}
                            )
                            == 1
                            and all(row["finite_output"] is True for row in completed)
                            and all(
                                row["no_backend_fallback"] is True for row in completed
                            )
                            and all(row["allocation_stable"] is True for row in completed)
                            and all(row["kernel_path_stable"] is True for row in completed)
                            and all(row["gpu_exclusive"] is True for row in completed)
                            and all(
                                row["graph_replay_allocation_zero"] is True
                                for row in completed
                            )
                        )
                        disposition = pilot.classify_point(
                            statistics_record=wall_stats, agreements=agreements
                        )
                    else:
                        wall_stats = {
                            "median_ms": None,
                            "mean_ms": None,
                            "standard_deviation_ms": None,
                            "minimum_ms": None,
                            "maximum_ms": None,
                            "cv": None,
                        }
                        cuda_stats = dict(wall_stats)
                        agreements = False
                        disposition = (
                            "pair_capacity_infeasible"
                            if all(status == "pair_capacity_infeasible" for status in statuses)
                            else "runtime_failed"
                            if "runtime_failed" in statuses
                            else "excluded"
                        )
                    summaries.append(
                        {
                            "method_config_id": configuration,
                            "method_config_fingerprint": CONFIG_FINGERPRINTS[
                                configuration
                            ],
                            "batch_size": batch,
                            "context_label": label,
                            "historical_context": actual_historical_context(label),
                            "graph_mode": mode,
                            "replicate_count": len(matching),
                            "completed_replicates": len(completed),
                            **wall_stats,
                            "cuda_median_ms": cuda_stats["median_ms"],
                            "cuda_mean_ms": cuda_stats["mean_ms"],
                            "cuda_standard_deviation_ms": cuda_stats[
                                "standard_deviation_ms"
                            ],
                            "cuda_cv": cuda_stats["cv"],
                            "process_wall_medians_ms": [
                                float(row["process_wall_median_ms"])
                                for row in sorted(
                                    completed,
                                    key=lambda item: int(item["replicate_index"]),
                                )
                            ],
                            "process_cuda_medians_ms": [
                                float(row["process_cuda_median_ms"])
                                for row in sorted(
                                    completed,
                                    key=lambda item: int(item["replicate_index"]),
                                )
                            ],
                            "host_wall_cuda_event_ratio": (
                                statistics.median(
                                    float(row["host_wall_cuda_event_ratio"])
                                    for row in completed
                                )
                                if completed
                                else None
                            ),
                            "temperature_range_c": _range(
                                completed, "temperature_min_c", "temperature_max_c"
                            ),
                            "sm_clock_range_mhz": _range(
                                completed, "sm_clock_min_mhz", "sm_clock_max_mhz"
                            ),
                            "memory_clock_range_mhz": _range(
                                completed,
                                "memory_clock_min_mhz",
                                "memory_clock_max_mhz",
                            ),
                            "power_range_w": _range(
                                completed, "power_min_w", "power_max_w"
                            ),
                            **byte_features,
                            "agreements": agreements,
                            "disposition": disposition,
                            "mechanism_experiment": True,
                            "quality_status": "unvalidated",
                            "performance_claim_eligible": False,
                            "r_hbm": None,
                        }
                    )
    if len(summaries) != 240:
        raise Phase14Error("Phase 14 point-summary cardinality differs")
    return summaries


def _pair_records(
    summaries: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {
        (
            str(row["method_config_id"]),
            int(row["batch_size"]),
            int(row["context_label"]),
            str(row["graph_mode"]),
        ): row
        for row in summaries
    }
    raw_group: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if row["status"] == "completed":
            raw_group[
                (
                    str(row["method_config_id"]),
                    int(row["batch_size"]),
                    int(row["context_label"]),
                )
            ].append(row)
    pairs: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        for batch in BATCH_SIZES:
            for label in CONTEXT_LABELS:
                eager = by_key[(configuration, batch, label, "eager")]
                graph = by_key[(configuration, batch, label, "cuda_graph")]
                completed = raw_group[(configuration, batch, label)]
                if (
                    eager["disposition"] == "pair_capacity_infeasible"
                    and graph["disposition"] == "pair_capacity_infeasible"
                ):
                    status = "pair_capacity_infeasible"
                elif eager["disposition"] == "stable" and graph["disposition"] == "stable":
                    output_exact = len({row["output_checksum"] for row in completed}) == 1
                    path_exact = len(
                        {row["semantic_kernel_path_fingerprint"] for row in completed}
                    ) == 1
                    cache_exact = len(
                        {row["cache_identity_fingerprint"] for row in completed}
                    ) == 1
                    status = "stable" if output_exact and path_exact and cache_exact else "failed"
                elif "unstable" in {eager["disposition"], graph["disposition"]}:
                    status = "unstable"
                else:
                    status = "failed"
                graph_ratio = (
                    float(eager["median_ms"]) / float(graph["median_ms"])
                    if status == "stable"
                    else None
                )
                eager_proxy = (
                    float(eager["median_ms"]) - float(eager["cuda_median_ms"])
                    if status == "stable"
                    else None
                )
                graph_proxy = (
                    float(graph["median_ms"]) - float(graph["cuda_median_ms"])
                    if status == "stable"
                    else None
                )
                pairs.append(
                    {
                        "method_config_id": configuration,
                        "method_config_fingerprint": CONFIG_FINGERPRINTS[
                            configuration
                        ],
                        "batch_size": batch,
                        "context_label": label,
                        "historical_context": actual_historical_context(label),
                        "pair_status": status,
                        "eager_wall_median_ms": eager["median_ms"],
                        "graph_wall_median_ms": graph["median_ms"],
                        "eager_cuda_median_ms": eager["cuda_median_ms"],
                        "graph_cuda_median_ms": graph["cuda_median_ms"],
                        "graph_ratio": graph_ratio,
                        "wall_minus_gpu_eager_ms": eager_proxy,
                        "wall_minus_gpu_graph_ms": graph_proxy,
                        "output_agreement": (
                            len({row["output_checksum"] for row in completed}) == 1
                            if completed
                            else False
                        ),
                        "kernel_path_agreement": (
                            len(
                                {
                                    row["semantic_kernel_path_fingerprint"]
                                    for row in completed
                                }
                            )
                            == 1
                            if completed
                            else False
                        ),
                        "cache_identity_agreement": (
                            len(
                                {row["cache_identity_fingerprint"] for row in completed}
                            )
                            == 1
                            if completed
                            else False
                        ),
                        "replicate_pairs": len(completed) // 2,
                        "host_minus_device_is_proxy_only": True,
                        "mechanism_experiment": True,
                        "quality_status": "unvalidated",
                        "performance_claim_eligible": False,
                        "not_quality_preserving_speedup": True,
                    }
                )
    return pairs


def _residual_summary(fit: Mapping[str, Any]) -> dict[str, Any]:
    model = fit.get("knee_model")
    residuals = model.get("residuals") if isinstance(model, Mapping) else None
    if not isinstance(residuals, list) or not residuals:
        return {"count": 0, "mean": None, "rmse": None, "minimum": None, "maximum": None}
    values = [float(value) for value in residuals]
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "rmse": math.sqrt(statistics.mean(value * value for value in values)),
        "minimum": min(values),
        "maximum": max(values),
    }


def _mode_fits(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fits: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        for batch in BATCH_SIZES:
            for mode in GRAPH_MODES:
                all_rows = [
                    row
                    for row in summaries
                    if row["method_config_id"] == configuration
                    and row["batch_size"] == batch
                    and row["graph_mode"] == mode
                ]
                stable = [row for row in all_rows if row["disposition"] == "stable"]
                if any(row["disposition"] == "unstable" for row in all_rows):
                    fit = {"fit_status": "unstable_data"}
                else:
                    fit = pilot.provisional_knee_fit(
                        [
                            (float(row["context_label"]), float(value))
                            for row in stable
                            for value in row["process_wall_medians_ms"]
                        ]
                    )
                bootstrap_rows = [
                    {**row, "process_medians_ms": row["process_wall_medians_ms"]}
                    for row in stable
                ]
                bootstrap = pilot._session_bootstrap_knee(
                    bootstrap_rows,
                    seed=20260826
                    + CONFIGURATIONS.index(configuration) * 1000
                    + batch * 10
                    + GRAPH_MODES.index(mode),
                )
                knee = fit.get("knee_model", {})
                fits.append(
                    {
                        "method_config_id": configuration,
                        "batch_size": batch,
                        "graph_mode": mode,
                        "stable_point_count": len(stable),
                        "session_observation_count": sum(
                            len(row["process_wall_medians_ms"]) for row in stable
                        ),
                        "fit_status": fit["fit_status"],
                        "tau": knee.get("tau"),
                        "a": knee.get("a"),
                        "s": knee.get("s"),
                        "L_star": knee.get("L_star"),
                        "r_squared": knee.get("r_squared"),
                        "bootstrap_estimable": bootstrap["estimable"],
                        "bootstrap_knee_lower_95": bootstrap["lower_95"],
                        "bootstrap_knee_upper_95": bootstrap["upper_95"],
                        "bootstrap_valid_draws": bootstrap["valid_draws"],
                        "residual_summary": _residual_summary(fit),
                        "fit_json": fit,
                        "quality_status": "unvalidated",
                        "performance_claim_eligible": False,
                    }
                )
    return fits


def _graph_effects(
    fits: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    index = {
        (str(row["method_config_id"]), int(row["batch_size"]), str(row["graph_mode"])): row
        for row in fits
    }
    results: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        for batch in BATCH_SIZES:
            eager = index[(configuration, batch, "eager")]
            graph = index[(configuration, batch, "cuda_graph")]
            tau_e = eager.get("tau")
            tau_g = graph.get("tau")
            slope_e = eager.get("s")
            slope_g = graph.get("s")
            knee_e = eager.get("L_star")
            knee_g = graph.get("L_star")
            matching = [
                row
                for row in pairs
                if row["method_config_id"] == configuration
                and row["batch_size"] == batch
                and row["pair_status"] == "stable"
            ]
            proxy_lower = bool(
                matching
                and statistics.median(
                    float(row["wall_minus_gpu_graph_ms"]) for row in matching
                )
                < statistics.median(
                    float(row["wall_minus_gpu_eager_ms"]) for row in matching
                )
            )
            floor_ratio = (
                float(tau_e) / float(tau_g)
                if isinstance(tau_e, (int, float))
                and isinstance(tau_g, (int, float))
                and float(tau_g) > 0
                else None
            )
            slope_ratio = (
                float(slope_g) / float(slope_e)
                if isinstance(slope_e, (int, float))
                and isinstance(slope_g, (int, float))
                and float(slope_e) > 0
                else None
            )
            floor_material = bool(floor_ratio is not None and floor_ratio >= 1.05)
            slope_similar = bool(
                slope_ratio is not None and 0.80 <= slope_ratio <= 1.20
            )
            semantic_match = all(
                row["output_agreement"] is True
                and row["kernel_path_agreement"] is True
                and row["cache_identity_agreement"] is True
                for row in matching
            )
            results.append(
                {
                    "method_config_id": configuration,
                    "batch_size": batch,
                    "eager_fit_status": eager["fit_status"],
                    "graph_fit_status": graph["fit_status"],
                    "floor_delta_ms": (
                        float(tau_e) - float(tau_g)
                        if isinstance(tau_e, (int, float))
                        and isinstance(tau_g, (int, float))
                        else None
                    ),
                    "floor_ratio": floor_ratio,
                    "slope_ratio": slope_ratio,
                    "knee_shift_tokens": (
                        float(knee_g) - float(knee_e)
                        if isinstance(knee_e, (int, float))
                        and isinstance(knee_g, (int, float))
                        else None
                    ),
                    "eager_knee": knee_e,
                    "graph_knee": knee_g,
                    "floor_material_threshold": 1.05,
                    "slope_similarity_interval": [0.80, 1.20],
                    "material_floor_reduction": floor_material,
                    "slope_similar": slope_similar,
                    "host_minus_device_proxy_lower": proxy_lower,
                    "semantics_unchanged": semantic_match,
                    "launch_floor_interpretation_supported": bool(
                        floor_material and slope_similar and proxy_lower and semantic_match
                    ),
                    "direct_launch_gap_measured": False,
                    "quality_status": "unvalidated",
                    "performance_claim_eligible": False,
                }
            )
    return results


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pilot._parquet_rows(path, rows)


def _plots(
    root: Path,
    summaries: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
) -> None:
    stable = [row for row in summaries if row["disposition"] == "stable"]
    for batch in BATCH_SIZES:
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in stable:
            if row["batch_size"] == batch:
                series[f"{row['method_config_id']}/{row['graph_mode']}"] .append(
                    (float(row["context_label"]), float(row["median_ms"]))
                )
        pilot._svg_line_plot(
            root / "plots" / f"eager-vs-graph-b{batch}.svg",
            title=f"Phase 14 eager versus Graph, B={batch}",
            y_label="median host-wall ms per operation",
            series=series,
            note="Mechanism-only; quality unvalidated; claim ineligible",
        )
    indexed = {str(i + 1): row for i, row in enumerate(effects)}
    for filename, title, key in (
        ("fitted-floor-comparison.svg", "Fitted floor ratio", "floor_ratio"),
        ("slope-ratio.svg", "Graph/eager slope ratio", "slope_ratio"),
        ("knee-shift.svg", "Graph minus eager knee", "knee_shift_tokens"),
    ):
        series = {
            row["method_config_id"] + f"/B{row['batch_size']}": [
                (float(index), float(row[key]))
            ]
            for index, row in indexed.items()
            if isinstance(row.get(key), (int, float))
        }
        pilot._svg_line_plot(
            root / "plots" / filename,
            title=title,
            y_label=key,
            series=series,
            note="Mechanism-only; quality unvalidated; claim ineligible",
        )
    pair_stable = [row for row in pairs if row["pair_status"] == "stable"]
    for filename, title, key in (
        ("graph-ratio-vs-context.svg", "Graph ratio versus context", "graph_ratio"),
        (
            "host-minus-device-proxy.svg",
            "Host-minus-device overhead proxy",
            "wall_minus_gpu_graph_ms",
        ),
    ):
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in pair_stable:
            series[f"{row['method_config_id']}/B{row['batch_size']}"] .append(
                (float(row["context_label"]), float(row[key]))
            )
        pilot._svg_line_plot(
            root / "plots" / filename,
            title=title,
            y_label=key,
            series=series,
            note="Proxy only; no direct launch-gap or quality claim",
        )
    series = defaultdict(list)
    for row in stable:
        series[f"{row['method_config_id']}/B{row['batch_size']}/{row['graph_mode']}"] .append(
            (float(row["context_label"]), float(row["cv"]))
        )
    pilot._svg_line_plot(
        root / "plots" / "cv-by-mode.svg",
        title="Process-median CV by mode",
        y_label="CV",
        series=series,
        note="Phase 14 paired process stability",
    )


def materialize_analysis(root: Path) -> dict[str, Any]:
    campaign = _strict_json(root / "unified" / "local-campaign.json")
    records = _run_records(root)
    summaries = _point_summaries(records)
    pairs = _pair_records(summaries, records)
    fits = _mode_fits(summaries)
    effects = _graph_effects(fits, pairs)
    feasibility = _strict_json(root / "feasibility.json")["records"]
    exclusions = [
        {
            "run_id": row["run_id"],
            "method_config_id": row["method_config_id"],
            "batch_size": row["batch_size"],
            "context_label": row["context_label"],
            "graph_mode": row["graph_mode"],
            "status": row["status"],
            "reason": row["reason"],
            "machine_readable": True,
        }
        for row in records
        if row["status"] != "completed"
    ]
    _write_parquet(root / "feasibility.parquet", feasibility)
    _write_parquet(root / "raw_run_index.parquet", records)
    _write_parquet(root / "point_summary.parquet", summaries)
    _write_parquet(root / "graph_ab_pairs.parquet", pairs)
    _write_parquet(root / "mode_fits.parquet", fits)
    _write_parquet(root / "graph_effects.parquet", effects)
    _write_parquet(root / "exclusions.parquet", exclusions)
    stable = [row for row in summaries if row["disposition"] == "stable"]
    unstable = [row for row in summaries if row["disposition"] == "unstable"]
    failed = [
        row
        for row in summaries
        if row["disposition"] in {"runtime_failed", "excluded", "failed"}
    ]
    stable_pairs = [row for row in pairs if row["pair_status"] == "stable"]
    bad_pairs = [row for row in pairs if row["pair_status"] in {"failed", "unstable"}]
    counts = {str(key): int(value) for key, value in campaign["status_counts"].items()}
    complete = bool(
        counts == {"completed": 660, "pair_capacity_infeasible": 60}
        and len(stable) == 220
        and not unstable
        and not failed
        and len(stable_pairs) == 110
        and not bad_pairs
    )
    local_pass = campaign["phase14_status"] == "LOCAL_COMPLETE" and complete
    qc = {
        "schema_version": "kvbench-phase14-qc-1.0.0",
        "campaign_id": campaign["campaign_id"],
        "phase14_status": (
            "LOCAL_PASS_PENDING_PUBLICATION" if local_pass else "BLOCKED"
        ),
        "planned_pair_records": PLANNED_PAIR_RECORDS,
        "planned_run_records": PLANNED_RUN_RECORDS,
        "pair_feasible_records": 330,
        "pair_capacity_infeasible_records": 30,
        "completed_runs": counts.get("completed", 0),
        "capacity_infeasible_run_records": counts.get(
            "pair_capacity_infeasible", 0
        ),
        "failed_runs": sum(
            count
            for status, count in counts.items()
            if status not in {"completed", "pair_capacity_infeasible"}
        ),
        "stable_mode_points": len(stable),
        "unstable_mode_points": len(unstable),
        "stable_ab_points": len(stable_pairs),
        "failed_ab_points": len(bad_pairs),
        "maximum_eager_cv": max(
            (
                float(row["cv"])
                for row in stable
                if row["graph_mode"] == "eager"
            ),
            default=None,
        ),
        "maximum_graph_cv": max(
            (
                float(row["cv"])
                for row in stable
                if row["graph_mode"] == "cuda_graph"
            ),
            default=None,
        ),
        "output_mismatches": sum(
            row["output_agreement"] is not True for row in stable_pairs
        ),
        "kernel_path_mismatches": sum(
            row["kernel_path_agreement"] is not True for row in stable_pairs
        ),
        "cache_identity_mismatches": sum(
            row["cache_identity_agreement"] is not True for row in stable_pairs
        ),
        "fit_status_counts": {
            status: sum(row["fit_status"] == status for row in fits)
            for status in sorted({str(row["fit_status"]) for row in fits})
        },
        "mode_fit_records": len(fits),
        "graph_effect_records": len(effects),
        "launch_floor_interpretation_supported_cases": sum(
            row["launch_floor_interpretation_supported"] is True for row in effects
        ),
        "direct_launch_gap_measured": False,
        "selective_reruns": 0,
        "mechanism_experiment": True,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "phase15_readiness": "READY" if local_pass else "NOT_READY",
        "blocker": None if local_pass else "Phase 14 paired QC is incomplete",
    }
    write_exclusive(root / "phase14_qc.json", json_bytes(qc))
    report = "\n".join(
        (
            "# Phase 14 CUDA Graph OFF/ON Mechanism Experiment",
            "",
            f"- Campaign: `{campaign['campaign_id']}`",
            f"- Status: `{qc['phase14_status']}`",
            f"- Planned pairs/runs: {PLANNED_PAIR_RECORDS}/{PLANNED_RUN_RECORDS}",
            f"- Completed/capacity-infeasible runs: {qc['completed_runs']}/{qc['capacity_infeasible_run_records']}",
            f"- Stable/unstable mode points: {len(stable)}/{len(unstable)}",
            f"- Maximum eager/Graph CV: `{qc['maximum_eager_cv']}` / `{qc['maximum_graph_cv']}`",
            f"- Fit statuses: `{json.dumps(qc['fit_status_counts'], sort_keys=True)}`",
            f"- Launch-floor interpretation cases: {qc['launch_floor_interpretation_supported_cases']}/20",
            "- Host-minus-device values are proxies, not direct CPU launch-gap measurements.",
            "- Quality remains unvalidated and performance claims remain ineligible.",
            "- Full Scan: `CLOSED`",
            "",
        )
    )
    write_exclusive(root / "phase14_report.md", report.encode("utf-8"))
    _plots(root, summaries, pairs, effects)
    write_exclusive(
        root / "inventory.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase14-scientific-inventory-1.0.0",
                "campaign_id": campaign["campaign_id"],
                "run_records": len(records),
                "point_summaries": len(summaries),
                "ab_pairs": len(pairs),
                "mode_fits": len(fits),
                "graph_effects": len(effects),
                "exclusions": len(exclusions),
                "plot_count": 8,
                "r_hbm": None,
            }
        ),
    )
    return qc


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise Phase14Error("Phase 14 campaign contains a symlink")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode) or path.stat().st_nlink != 1:
            raise Phase14Error("Phase 14 campaign contains an unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            files.append(path)
    return files


def seal_campaign(stage: Path, *, campaign_id: str) -> Path:
    identifier = _validate_campaign_id(campaign_id)
    root = stage.resolve(strict=True)
    qc = _strict_json(root / "phase14_qc.json")
    if qc.get("campaign_id") != identifier:
        raise Phase14Error("Phase 14 QC campaign identity differs")
    write_exclusive(
        root / "manifest.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase14-artifact-manifest-1.0.0",
                "run_id": identifier,
                "campaign_id": identifier,
                "status": qc["phase14_status"],
                "created_at_utc": _utc_now(),
                "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
                "planned_run_records": PLANNED_RUN_RECORDS,
                "append_only": True,
                "complete_written_last": True,
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
                "r_hbm": None,
            }
        ),
    )
    items = [
        {
            "path": path.relative_to(root).as_posix(),
            "role": "phase14_graph_ab_evidence",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _payload_paths(
            root, {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
        )
    ]
    write_exclusive(
        root / "artifact_inventory.json",
        json_bytes(
            {
                "schema_version": "kvbench-artifact-inventory-1.0.0",
                "run_id": identifier,
                "files": items,
                "excluded_control_files": [
                    "artifact_inventory.json",
                    "checksums.sha256",
                    "COMPLETE",
                ],
            }
        ),
    )
    ledger = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in _payload_paths(root, {"checksums.sha256", "COMPLETE"})
    ).encode("utf-8")
    write_exclusive(root / "checksums.sha256", ledger)
    write_exclusive(
        root / "COMPLETE",
        json_bytes(
            {
                "schema_version": "kvbench-completion-1.0.0",
                "run_id": identifier,
                "status": qc["phase14_status"],
                "manifest_sha256": sha256_file(root / "manifest.json"),
                "artifact_inventory_sha256": sha256_file(
                    root / "artifact_inventory.json"
                ),
                "checksum_ledger_path": "checksums.sha256",
                "checksum_ledger_sha256": sha256_file(root / "checksums.sha256"),
                "written_last": True,
            }
        ),
    )
    final = ARTIFACT_ROOT / identifier
    if final.exists() or final.is_symlink():
        raise Phase14Error("Phase 14 finalized campaign already exists")
    rename_noreplace(root, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    validate_campaign(final, expected_campaign_id=identifier)
    return final


def validate_campaign(
    root: Path, *, expected_campaign_id: str | None = None
) -> dict[str, Any]:
    artifact = validate_local_artifact(root, environ={})
    manifest = _strict_json(root / "manifest.json")
    identifier = str(manifest.get("campaign_id"))
    _validate_campaign_id(identifier)
    if expected_campaign_id is not None and identifier != expected_campaign_id:
        raise Phase14Error("Phase 14 campaign identity differs")
    order = _strict_json(root / "execution_order.json")
    validate_execution_order(order)
    records = _run_records(root)
    qc = _strict_json(root / "phase14_qc.json")
    campaign = _strict_json(root / "campaign_manifest.json")
    if (
        campaign.get("execution_order_sha256") != sha256_file(root / "execution_order.json")
        or campaign.get("planned_pair_records") != PLANNED_PAIR_RECORDS
        or campaign.get("planned_run_records") != PLANNED_RUN_RECORDS
        or campaign.get("source_timing_reused") is not False
        or campaign.get("pair_members_back_to_back") is not True
        or campaign.get("mode_first_randomized") is not True
        or qc.get("planned_run_records") != PLANNED_RUN_RECORDS
        or qc.get("full_scan") != "CLOSED"
        or qc.get("quality_execution") != "LOCKED"
        or qc.get("r_hbm") is not None
        or len(records) != PLANNED_RUN_RECORDS
    ):
        raise Phase14Error("Phase 14 campaign validation differs")
    return {
        "status": "PASS",
        "campaign_id": identifier,
        "phase14_status": qc["phase14_status"],
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
    }


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--validate-entry", action="store_true")
    actions.add_argument("--write-execution-order", action="store_true")
    actions.add_argument("--validate-execution-order", action="store_true")
    actions.add_argument("--print-feasibility-summary", action="store_true")
    actions.add_argument("--new-campaign-id", action="store_true")
    actions.add_argument("--reserve-campaign", action="store_true")
    actions.add_argument("--run-campaign", action="store_true")
    actions.add_argument("--run-worker", action="store_true")
    actions.add_argument("--materialize-analysis", action="store_true")
    actions.add_argument("--finalize-staged-campaign", action="store_true")
    actions.add_argument("--validate-campaign", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--git-sha")
    parser.add_argument("--campaign-id")
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--configuration", choices=CONFIGURATIONS)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-label", type=int)
    parser.add_argument("--graph-mode", choices=GRAPH_MODES)
    parser.add_argument("--replicate-index", type=int)
    parser.add_argument("--order-index", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--run-artifact-root", type=Path)
    parser.add_argument("--prefix-root", type=Path)
    parser.add_argument("--prefix-state-root", type=Path)
    parser.add_argument("--prefix-state-sha256")
    parser.add_argument("--source-witness-path", type=Path)
    return parser.parse_args(argv)


def _require(value: Any, label: str) -> Any:
    if value is None:
        raise Phase14Error(f"{label} is required")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.validate_entry:
        print(json.dumps(validate_lightweight_entry(), sort_keys=True))
    elif args.write_execution_order:
        write_execution_order(_require(args.output, "output"))
    elif args.validate_execution_order:
        validate_execution_order(_strict_json(_require(args.output, "output")))
        print(json.dumps({"status": "PASS", "records": PLANNED_RUN_RECORDS}, sort_keys=True))
    elif args.print_feasibility_summary:
        rows = build_feasibility_records(_strict_json(ORDER_PATH))
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "planned_run_records": len(rows),
                    "pair_feasible_run_records": sum(
                        row["status"] == "pair_feasible" for row in rows
                    ),
                    "pair_capacity_infeasible_run_records": sum(
                        row["status"] == "pair_capacity_infeasible" for row in rows
                    ),
                },
                sort_keys=True,
            )
        )
    elif args.new_campaign_id:
        print(new_campaign_id(_require(args.git_sha, "git-sha")))
    elif args.reserve_campaign:
        print(
            reserve_campaign(
                campaign_id=_require(args.campaign_id, "campaign-id"),
                git_sha=_require(args.git_sha, "git-sha"),
            )
        )
    elif args.run_campaign:
        result = run_campaign(
            stage=_require(args.stage, "stage"),
            campaign_id=_require(args.campaign_id, "campaign-id"),
            git_sha=_require(args.git_sha, "git-sha"),
            prefix_root=_require(args.prefix_root, "prefix-root"),
        )
        print(json.dumps(result, sort_keys=True))
    elif args.run_worker:
        result = _run_worker(
            run_id=_require(args.run_id, "run-id"),
            configuration=_require(args.configuration, "configuration"),
            batch=_require(args.batch_size, "batch-size"),
            context_label=_require(args.context_label, "context-label"),
            graph_mode=_require(args.graph_mode, "graph-mode"),
            replicate_index=_require(args.replicate_index, "replicate-index"),
            order_index=_require(args.order_index, "order-index"),
            git_sha=_require(args.git_sha, "git-sha"),
            run_artifact_root=_require(args.run_artifact_root, "run-artifact-root"),
            prefix_state_root=_require(args.prefix_state_root, "prefix-state-root"),
            prefix_state_sha256=_require(
                args.prefix_state_sha256, "prefix-state-sha256"
            ),
            source_witness_path=_require(
                args.source_witness_path, "source-witness-path"
            ),
        )
        print(WORKER_PREFIX + json.dumps(result, sort_keys=True))
    elif args.materialize_analysis:
        print(json.dumps(materialize_analysis(_require(args.stage, "stage")), sort_keys=True))
    elif args.finalize_staged_campaign:
        print(
            seal_campaign(
                _require(args.stage, "stage"),
                campaign_id=_require(args.campaign_id, "campaign-id"),
            )
        )
    elif args.validate_campaign:
        print(
            json.dumps(
                validate_campaign(_require(args.artifact, "artifact")),
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Narrow Phase 13 Pilot coordinator, QC, and provisional analysis.

The module owns no adapter, CUDA kernel, cache layout, or timing boundary.  It
reuses the Phase 12 authority/session bridge and the common fixed-L runner,
adds the preregistered Pilot grid, and keeps every planned point append-only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
from typing import Any

from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from kvbench.runtime.artifacts import sha256_file
from kvbench.runtime.method_harness import execution_path_audit_facade
from kvbench.runtime.process_supervision import run_supervised_command
from kvbench.schema import GraphMode, RunnerKind, canonical_json_bytes, sha256_hex
from scripts.r2_artifact import validate_local_artifact
import scripts.phase12_unified_admission as phase12


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase13"
STAGING_ROOT = ARTIFACT_ROOT / ".kvbench-staging"
PLAN_PATH = Path("docs/plans/phase13-pilot-scan.md")
ORDER_PATH = Path("docs/plans/phase13-pilot-execution-order.json")

AUTHORIZED_CONTAINER_DIGEST = phase12.PHASE12_AUTHORIZED_CONTAINER_DIGEST
GPU_UUID = phase12.PHASE12_GPU_UUID
CONFIGURATIONS = phase12.MAIN_CONFIG_IDS
CONFIG_FINGERPRINTS = dict(phase12.EXPECTED_CONFIG_FINGERPRINTS)
HELD_OUT_CONFIGURATIONS = phase12.HELD_OUT_CONFIG_IDS
BATCH_SIZES = (1, 4, 8)
CONTEXT_LABELS = (4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072)
SEEDS = (20260801, 20260802, 20260803)
REPLICATES = 3
WARMUP_STEPS = 64
MEASURED_STEPS = 128
MEASURED_BATCHES = 5
CV_THRESHOLD = 0.03
MAX_MEMORY_FRACTION = 0.88
GPU_TOTAL_MEMORY_BYTES = 101_970_345_984
MODEL_WEIGHT_BYTES = 16_060_556_288
REFERENCE_CAPACITY = 4097
PLANNED_RECORD_COUNT = 810
INPUT_RECIPE_SCHEMA = "kvbench-phase13-pilot-input-1.0.0"
CAMPAIGN_SCHEMA = "kvbench-phase13-pilot-campaign-1.0.0"
RUN_SCHEMA = "kvbench-phase13-pilot-process-run-1.0.0"
WORKER_PREFIX = "PHASE13_WORKER_RESULT="
CHILD_TIMEOUT_SECONDS = 7_200.0

_CAMPAIGN_RE = re.compile(
    r"phase13-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)
_RUN_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,159}\Z")
_CONTROL_FILES = frozenset(
    {"manifest.json", "inventory.json", "checksums.sha256", "COMPLETE"}
)
_FORBIDDEN_ENVIRONMENT = phase12._FORBIDDEN_CHILD_ENVIRONMENT


class Phase13PilotError(RuntimeError):
    """The Pilot contract or evidence failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_sha256(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value))


def actual_historical_context(label: int) -> int:
    """Map the requested top label to the fixed-L historical-prefix convention."""

    if label not in CONTEXT_LABELS:
        raise Phase13PilotError("unknown Phase 13 context label")
    return 131071 if label == 131072 else label


def _point_seed(seed: int, configuration: str) -> int:
    digest = hashlib.sha256(f"{seed}:{configuration}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def derive_execution_order() -> dict[str, Any]:
    """Derive all 810 immutable planned records with blocked randomization."""

    records: list[dict[str, Any]] = []
    block_orders: list[dict[str, Any]] = []
    for replicate_index, seed in enumerate(SEEDS):
        blocks = list(CONFIGURATIONS)
        random.Random(seed).shuffle(blocks)
        block_orders.append(
            {
                "replicate_index": replicate_index,
                "seed": seed,
                "configuration_order": blocks,
            }
        )
        global_index = 0
        for block_index, configuration in enumerate(blocks):
            points = [(batch, label) for batch in BATCH_SIZES for label in CONTEXT_LABELS]
            random.Random(_point_seed(seed, configuration)).shuffle(points)
            for within_block_index, (batch, label) in enumerate(points):
                historical = actual_historical_context(label)
                records.append(
                    {
                        "replicate_index": replicate_index,
                        "seed": seed,
                        "block_index": block_index,
                        "within_block_index": within_block_index,
                        "order_index": global_index,
                        "method_config_id": configuration,
                        "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                        "batch_size": batch,
                        "context_label": label,
                        "historical_context": historical,
                        "total_attended_context": historical + 1,
                        "runner_kind": "fixed_l",
                        "graph_mode": "cuda_graph",
                    }
                )
                global_index += 1
    if len(records) != PLANNED_RECORD_COUNT:
        raise Phase13PilotError("Phase 13 execution order cardinality differs")
    identity_keys = (
        "replicate_index",
        "method_config_id",
        "batch_size",
        "context_label",
    )
    if len({tuple(record[key] for key in identity_keys) for record in records}) != len(
        records
    ):
        raise Phase13PilotError("Phase 13 execution order contains duplicates")
    payload = {
        "schema_version": "kvbench-phase13-pilot-execution-order-1.0.0",
        "seeds": list(SEEDS),
        "blocked_randomization": True,
        "configuration_blocks": True,
        "point_order_within_block_randomized": True,
        "block_orders": block_orders,
        "records": records,
    }
    payload["records_sha256"] = _canonical_sha256(records)
    return payload


def validate_execution_order(payload: Mapping[str, Any]) -> None:
    expected = derive_execution_order()
    if dict(payload) != expected:
        raise Phase13PilotError("committed Phase 13 execution order differs")


def _phase12_reference_runs() -> dict[str, Mapping[str, Any]]:
    root = (
        REPOSITORY_ROOT
        / "artifacts/phase12/phase12-20260731t062914664948z-6165f78d-c78b9a/runs"
    )
    selected: dict[str, Mapping[str, Any]] = {}
    for path in sorted(root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        configuration = payload.get("method_config_id")
        if configuration in CONFIGURATIONS and configuration not in selected:
            if payload.get("method_config_fingerprint") != CONFIG_FINGERPRINTS[configuration]:
                raise Phase13PilotError("Phase 12 reference fingerprint differs")
            selected[str(configuration)] = payload
    if tuple(selected) != CONFIGURATIONS:
        missing = sorted(set(CONFIGURATIONS) - set(selected))
        if missing:
            raise Phase13PilotError(f"Phase 12 reference runs absent: {missing}")
        selected = {configuration: selected[configuration] for configuration in CONFIGURATIONS}
    return selected


def _reference_graph_reserve_bytes(configuration: str) -> int:
    worker = _phase12_reference_runs()[configuration]
    memory = worker["runner"]["memory_evidence"]
    cache = worker["runner"]["cache_accounting"]
    reserve = (
        int(memory["post_setup"]["allocated_bytes"])
        - int(memory["model_baseline"]["allocated_bytes"])
        - int(cache["allocated_bytes"])
    )
    if reserve <= 0:
        raise Phase13PilotError("Phase 12 graph reserve is invalid")
    return reserve


def _turboquant_cache_bytes(configuration: str, batch: int, capacity: int) -> int:
    slot_size = {"tq_4bit_nc": 134, "tq_k3v4_nc": 118, "tq_3bit_nc": 102}[
        configuration
    ]
    rounded = math.ceil(capacity / 16) * 16
    packed = 28 * rounded * 8 * slot_size
    skipped_bf16 = 2 * 4 * 8 * rounded * 128 * 2
    mapping = math.ceil(capacity / 16) * 4 + rounded * 8
    hadamard = 2 * 128 * 128 * 4
    levels = {"tq_4bit_nc": 16, "tq_k3v4_nc": 16, "tq_3bit_nc": 8}[
        configuration
    ]
    quantizer = (levels + max(0, levels - 1)) * 4
    store = 3 * capacity * 8 * 128 * 4 + 2 * capacity * 8 * 4
    decode = batch * (
        2 * 32 * 128 * 4 + 32 * 4 * 129 * 4 + 32 * 128 * 2 + 32 * 4
    )
    # Compressed storage is source-defined only for B=1.  The batch multiplier
    # is a conservative memory projection; the separate authority flag remains
    # false for B>1 and the real factory attempt is still required.
    batch_scaled_history = batch * (packed + skipped_bf16 + mapping + store)
    return batch_scaled_history + hadamard + quantizer + decode


def _kivi_cache_bytes(configuration: str, batch: int, capacity: int) -> int:
    k_bits, v_bits = {"k4v4": (4, 4), "k2v4": (2, 4), "k2v2": (2, 2)}[
        configuration
    ]
    layers, kv_heads, query_heads, dimension = 32, 8, 32, 128
    residual, group = 32, 32
    key_history_capacity = (capacity // group) * group
    value_history_capacity = max(0, capacity - residual)
    key_groups = key_history_capacity // group
    value_head_groups = dimension // group
    key_words = key_history_capacity * k_bits // 32
    value_words = dimension * v_bits // 32
    key_residual = layers * batch * kv_heads * residual * dimension * 2
    value_residual = key_residual
    ordered_value = value_residual
    fp16_staging = sum(
        (
            ordered_value,
            batch * query_heads * dimension * 2,
            2 * batch * kv_heads * dimension * 2,
            2 * batch * kv_heads * dimension * 2,
            2 * batch * kv_heads * value_head_groups * 2,
            batch * query_heads * dimension * 2,
        )
    )
    quant_elements = batch * kv_heads * dimension * residual
    quant_staging = (
        quant_elements * 2
        + quant_elements * 4
        + batch * kv_heads * dimension * 8 * 4
    )
    workspace = (
        2 * batch * query_heads * capacity * 2
        + batch * query_heads * capacity * 4
        + 2 * batch * query_heads * dimension * 2
    )
    categories = (
        layers * batch * kv_heads * key_words * dimension * 4,
        layers * batch * kv_heads * value_words * value_history_capacity * 4,
        2 * layers * batch * kv_heads * key_groups * dimension * 2,
        2 * layers * batch * kv_heads * value_head_groups * value_history_capacity * 2,
        key_residual,
        value_residual,
        layers
        * (key_history_capacity + residual + value_history_capacity + residual)
        * 8,
        fp16_staging,
        quant_staging,
        workspace,
    )
    return sum(categories)


def _kvquant_cache_bytes(configuration: str, batch: int, capacity: int) -> int:
    bits = {"kvq4": 4, "kvq3": 3, "kvq2": 2}[configuration]
    levels = 1 << bits
    layers, heads, query_heads, dimension = 32, 8, 32, 128
    packed_rows = bits * dimension // 32
    query_elements = batch * query_heads * dimension
    kv_elements = batch * heads * dimension
    dense = batch * layers * heads * packed_rows * capacity * 4
    key_metadata = (
        layers * levels * 4
        + layers * heads * dimension * levels * 4
        + 3 * layers * heads * dimension * 4
        + 64 * 4
    )
    value_metadata = batch * (layers * levels * 4 + layers * capacity * levels * 4)
    sparse = batch * layers * capacity * 12 * 4
    count_mask = batch * (2 * layers * capacity * 4 + capacity)
    sink = layers * batch * heads * dimension * 5 * 2
    staging = (
        3 * kv_elements * 2
        + 4 * kv_elements * 4
        + query_elements * 2
        + query_elements * 4
        + query_elements * 2
        + 2 * 12 * 4
        + 3 * 4
        + 1
        + kv_elements * 4
        + 2 * kv_elements * 4
        + levels * 4
        + 5 * 4
        + 2 * capacity * 4
        + 3 * 4
        + 8
    )
    workspace = (
        2 * batch * query_heads * capacity * 4
        + batch * query_heads * capacity * 2
        + batch * query_heads * 5 * 2
        + 4 * query_elements * 4
        + query_elements * 2
        + (batch * 32 * 32 * 128 * 4 if configuration == "kvq4" else 0)
    )
    return (
        2 * dense
        + key_metadata
        + value_metadata
        + 4 * sparse
        + count_mask
        + 2 * sink
        + staging
        + workspace
    )


def cache_allocated_bytes(configuration: str, batch: int, capacity: int) -> int:
    if configuration == "bf16":
        return 2 * 32 * batch * 8 * capacity * 128 * 2 + 163_840
    if configuration.startswith("tq_"):
        return _turboquant_cache_bytes(configuration, batch, capacity)
    if configuration in {"k4v4", "k2v4", "k2v2"}:
        return _kivi_cache_bytes(configuration, batch, capacity)
    return _kvquant_cache_bytes(configuration, batch, capacity)


def adapter_geometry_supported(configuration: str, batch: int) -> bool:
    """Reflect the current factory/cache boundary without weakening the grid."""

    return configuration == "bf16" or batch == 1


def feasibility_record(order_record: Mapping[str, Any]) -> dict[str, Any]:
    configuration = str(order_record["method_config_id"])
    batch = int(order_record["batch_size"])
    historical = int(order_record["historical_context"])
    capacity = historical + 1
    cache_bytes = cache_allocated_bytes(configuration, batch, capacity)
    reference_reserve = _reference_graph_reserve_bytes(configuration)
    graph_reserve = math.ceil(
        reference_reserve * batch * capacity / REFERENCE_CAPACITY
    )
    limit = math.floor(GPU_TOTAL_MEMORY_BYTES * MAX_MEMORY_FRACTION)
    required = MODEL_WEIGHT_BYTES + cache_bytes + graph_reserve
    feasible = required <= limit
    return {
        **dict(order_record),
        "schema_version": "kvbench-phase13-feasibility-record-1.0.0",
        "capacity": capacity,
        "model_weight_bytes": MODEL_WEIGHT_BYTES,
        "cache_allocated_bytes": cache_bytes,
        "persistent_workspace_included_in_cache": True,
        "graph_pool_or_capture_reserve_bytes": graph_reserve,
        "graph_reserve_reference_bytes": reference_reserve,
        "gpu_total_memory_bytes": GPU_TOTAL_MEMORY_BYTES,
        "max_memory_fraction": MAX_MEMORY_FRACTION,
        "configured_safety_margin_bytes": GPU_TOTAL_MEMORY_BYTES - limit,
        "limit_bytes": limit,
        "predicted_required_bytes": required,
        "status": "feasible" if feasible else "capacity_infeasible",
        "reason": None if feasible else "predicted_required_bytes_exceed_0.88_limit",
        "adapter_geometry_supported_at_entry": adapter_geometry_supported(
            configuration, batch
        ),
        "unsupported_geometry_is_not_reclassified_as_capacity": True,
        "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
    }


def build_feasibility_records(order: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_execution_order(order)
    records = [feasibility_record(record) for record in order["records"]]
    if len(records) != PLANNED_RECORD_COUNT:
        raise Phase13PilotError("Phase 13 feasibility cardinality differs")
    return records


@dataclasses.dataclass(frozen=True, slots=True)
class Phase13OperationKey:
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
    def create(cls, configuration: str, batch: int, historical: int) -> "Phase13OperationKey":
        payload = {
            "schema_version": "kvbench-phase13-operation-key-1.0.0",
            "configuration": configuration,
            "runner_kind": "fixed_l",
            "graph_mode": "cuda_graph",
            "historical_context": historical,
            "attended_context": historical + 1,
            "batch_size": batch,
            "capacity": historical + 1,
            "decode_step": 0,
            "input_recipe_schema": INPUT_RECIPE_SCHEMA,
        }
        return cls(
            configuration=configuration,
            runner_kind=RunnerKind.FIXED_L,
            graph_mode=GraphMode.CUDA_GRAPH,
            historical_context=historical,
            attended_context=historical + 1,
            batch_size=batch,
            capacity=historical + 1,
            decode_step=0,
            operation_fingerprint_sha256=_canonical_sha256(payload),
        )


def point_statistics(process_medians: Sequence[float]) -> dict[str, float]:
    if len(process_medians) != REPLICATES:
        raise Phase13PilotError("Pilot point requires exactly three processes")
    values = [float(value) for value in process_medians]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise Phase13PilotError("Pilot process median is invalid")
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    return {
        "median_ms": statistics.median(values),
        "mean_ms": mean,
        "standard_deviation_ms": standard_deviation,
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "cv": standard_deviation / mean,
    }


def classify_point(*, statistics_record: Mapping[str, Any], agreements: bool) -> str:
    if not agreements:
        return "failed"
    return "stable" if float(statistics_record["cv"]) <= CV_THRESHOLD else "unstable"


def provisional_knee_fit(observations: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Fit constant, linear, and fixed-order hinge candidates without claiming a knee."""

    if len(observations) < 4 or len({x for x, _ in observations}) < 4:
        return {"fit_status": "insufficient_feasible_span"}
    points = sorted((float(x), float(y)) for x, y in observations)
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    if any(not math.isfinite(value) for value in (*xs, *ys)):
        return {"fit_status": "fit_failed"}
    tau0 = statistics.mean(ys)
    floor_sse = sum((value - tau0) ** 2 for value in ys)
    x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator <= 0:
        return {"fit_status": "insufficient_feasible_span"}
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    linear_sse = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    best: tuple[float, float, float, float] | None = None
    candidates = sorted(set(xs + [(left + right) / 2 for left, right in zip(xs, xs[1:])]))
    for knee in candidates:
        hinge = [max(0.0, x - knee) for x in xs]
        h_mean = statistics.mean(hinge)
        h_denom = sum((h - h_mean) ** 2 for h in hinge)
        candidate_slope = (
            0.0
            if h_denom == 0
            else sum((h - h_mean) * (y - y_mean) for h, y in zip(hinge, ys))
            / h_denom
        )
        if candidate_slope < 0:
            candidate_slope = 0.0
        candidate_tau = y_mean - candidate_slope * h_mean
        sse = sum(
            (y - (candidate_tau + candidate_slope * h)) ** 2
            for h, y in zip(hinge, ys)
        )
        candidate = (sse, knee, candidate_tau, candidate_slope)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    knee_sse, knee, tau, knee_slope = best
    total_sse = sum((value - y_mean) ** 2 for value in ys)
    r_squared = None if total_sse == 0 else 1.0 - knee_sse / total_sse
    if knee_slope <= 0:
        status = "no_positive_slope"
    elif knee < min(xs):
        status = "knee_below_range"
    elif knee > max(xs):
        status = "knee_above_range"
    else:
        status = "knee_observed"
    return {
        "fit_status": status,
        "constant_floor": {"tau": tau0, "sse": floor_sse},
        "linear": {"a": intercept, "s": slope, "sse": linear_sse},
        "knee_model": {
            "tau": tau,
            "a": tau - knee_slope * knee,
            "s": knee_slope,
            "L_star": knee if knee_slope > 0 else None,
            "sse": knee_sse,
            "r_squared": r_squared,
            "residuals": [
                y - (tau + knee_slope * max(0.0, x - knee)) for x, y in points
            ],
        },
    }


def knee_density(contexts: Sequence[int], knee: float | None) -> dict[str, Any]:
    if knee is None or not math.isfinite(knee) or knee <= 0:
        return {"assessed": False, "sufficient": False, "missing_interval": None}
    below = [value for value in contexts if value < 0.75 * knee]
    near = [value for value in contexts if 0.75 * knee <= value <= 1.25 * knee]
    above = [value for value in contexts if value > 1.25 * knee]
    sufficient = bool(below and near and above)
    return {
        "assessed": True,
        "below_count": len(below),
        "near_count": len(near),
        "above_count": len(above),
        "near_definition": "[0.75*L_star,1.25*L_star]",
        "sufficient": sufficient,
        "missing_interval": None if sufficient else [0.75 * knee, 1.25 * knee],
    }


def new_campaign_id(git_sha: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise Phase13PilotError("execution Git SHA is invalid")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:21]
    return f"phase13-{stamp}z-{git_sha[:8]}-{secrets.token_hex(3)}"


def _validate_campaign_id(value: str) -> str:
    if _CAMPAIGN_RE.fullmatch(value) is None:
        raise Phase13PilotError("Phase 13 campaign ID is invalid")
    return value


def reserve_campaign(*, campaign_id: str, git_sha: str) -> Path:
    identifier = _validate_campaign_id(campaign_id)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    stage = STAGING_ROOT / f"{identifier}.{secrets.token_hex(4)}.staging"
    stage.mkdir()
    for relative in ("runs", "unified", "pilot_plots"):
        (stage / relative).mkdir()
    write_exclusive(
        stage / "campaign-reservation.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13-campaign-reservation-1.0.0",
                "campaign_id": identifier,
                "execution_git_sha": git_sha,
                "reserved_at_utc": _utc_now(),
                "append_only": True,
                "overwrite": False,
            }
        ),
    )
    return stage.resolve(strict=True)


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13PilotError(f"invalid JSON evidence: {path}") from error
    if not isinstance(payload, dict):
        raise Phase13PilotError(f"JSON evidence is not an object: {path}")
    return payload


def _patch_phase12_point_globals(*, batch: int, historical: int) -> None:
    recipe = {
        "schema_version": INPUT_RECIPE_SCHEMA,
        "prefix": (
            f"(arange({batch}*{historical}).reshape({batch},{historical})"
            "+12000)%120000+1000"
        ),
        "decode": (
            f"(arange({batch}).reshape({batch},1)+12000+{historical}+257)"
            "%120000+1000"
        ),
    }
    phase12.PHASE12_BATCH_SIZE = batch
    phase12.PHASE12_CONTEXT_LENGTH = historical
    phase12.PHASE12_WARMUP_STEPS = WARMUP_STEPS
    phase12.PHASE12_MEASURED_STEPS = MEASURED_STEPS
    phase12.PHASE12_MEASURED_BATCHES = MEASURED_BATCHES
    phase12.PHASE12_INPUT_RECIPE = recipe
    phase12.PHASE12_INPUT_RECIPE_SHA256 = _canonical_sha256(recipe)


def _run_worker(
    *,
    run_id: str,
    configuration: str,
    batch: int,
    context_label: int,
    replicate_index: int,
    order_index: int,
    git_sha: str,
    run_artifact_root: Path,
) -> dict[str, Any]:
    """Execute one feasible Pilot point inside the authorized container."""

    attestation = phase12._require_authorized_container_runtime()
    pre_snapshot = phase12._capture_process_snapshot()
    phase12._require_idle_snapshot(pre_snapshot)
    pid = os.getpid()
    start_ticks = phase12._process_start_ticks(pid)

    import torch

    from kvbench.runtime.allocation import audit_cuda_allocations
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.fixed_l_runner import run_fixed_l
    from kvbench.runtime.model_loader import load_frozen_model
    from kvbench.runtime.numerical import tensor_sha256_untimed
    from kvbench.runtime.timing import warmup_operations

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Phase13PilotError("worker execution authority differs")
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
    if observed_head != git_sha or observed_status:
        raise Phase13PilotError("worker Git source authority or cleanliness differs")

    historical = actual_historical_context(context_label)
    _patch_phase12_point_globals(batch=batch, historical=historical)
    device = torch.device("cuda:0")
    loaded = load_frozen_model(device=device)
    prefix = (
        torch.arange(batch * historical, dtype=torch.long, device=device)
        .reshape(batch, historical)
        .add(12_000)
        .remainder(120_000)
        .add(1_000)
    )
    decode = (
        torch.arange(batch, dtype=torch.long, device=device)
        .reshape(batch, 1)
        .add(12_000 + historical + 257)
        .remainder(120_000)
        .add(1_000)
    )
    operation = Phase13OperationKey.create(configuration, batch, historical)
    with torch.inference_mode(), forced_flash_execution():
        with phase12._observable_cuda_graph_factory(torch) as observed_graphs:
            session = phase12._build_phase12_session(
                loaded=loaded,
                operation_key=operation,
                prefix_input_ids=prefix,
                decode_input_ids=decode,
            )
        if session.graph is None or len(observed_graphs) != 1:
            raise Phase13PilotError("captured graph is absent or ambiguous")
        graph_exec_before = int(session.graph.graph.raw_cuda_graph_exec())
        graph_path_before = phase12._write_cuda_graph_path_witness(
            graph=session.graph.graph,
            run_root=run_artifact_root,
            phase="before",
        )
        pointers_before = phase12._phase12_session_pointers(session)
        history_before = session.current_historical_prefix_sha256()
        warm_output = warmup_operations(
            session.graph.replay,
            count=WARMUP_STEPS,
            device=session.cache_device,
        )
        warm_cpu = warm_output.detach().to(device="cpu", copy=True).clone()
        warm_checksum = tensor_sha256_untimed(warm_cpu)
        warm_finite = bool(torch.isfinite(warm_cpu).all())
        replay_allocation = audit_cuda_allocations(
            session.graph.replay,
            device=session.cache_device,
        )
        audit_output = (
            session.graph.replay().detach().to(device="cpu", copy=True).clone()
        )
        torch.cuda.synchronize(device=session.cache_device)
        audit_checksum = tensor_sha256_untimed(audit_output)
        audit_finite = bool(torch.isfinite(audit_output).all())
        allocation_record = replay_allocation.to_dict()
        graph_passed = bool(
            session.graph_evidence is not None
            and session.graph_evidence.get("captured") is True
            and session.graph_evidence.get("fallback") is False
            and session.graph_evidence.get("consecutive_replay_outputs_exact") is True
            and session.eager_graph_comparison is not None
            and session.eager_graph_comparison.passed
        )
        allocation_passed = bool(
            replay_allocation.audit_available
            and replay_allocation.passed
            and replay_allocation.allocation_event_count == 0
            and replay_allocation.allocation_event_bytes == 0
            and replay_allocation.allocated_after == replay_allocation.allocated_before
            and replay_allocation.reserved_after == replay_allocation.reserved_before
        )
        family = phase12._method_family(configuration)
        geometry = session.gqa_cache_geometry()
        prior_g3 = phase12._expected_prior_g3_binding(family)
        phase12._validate_prior_g3_binding(prior_g3, family=family)
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
        geometry_passed = phase12._gqa_geometry_passes(geometry, family=family)
        path_audit = execution_path_audit_facade(
            backend_identity_verified=backend_verified,
            device_kernel_family_verified=bool(live_fingerprint),
            allocation_categories_verified=allocation_passed,
            temporary_tensor_shapes_verified=(
                pointers_before == phase12._phase12_session_pointers(session)
                and geometry_passed
            ),
            gqa_replication_detected=not geometry_passed,
            full_prefix_temporary_detected=False,
            host_synchronization_detected=False,
            backend_fallback_detected=not graph_passed,
            full_prefix_dequantization=(
                "not_applicable" if family == "bf16" else "verified_false"
            ),
        )
        phase12._validate_execution_path_record(path_audit.to_dict(), family=family)
        if (
            not warm_finite
            or not audit_finite
            or warm_checksum != audit_checksum
            or pointers_before != phase12._phase12_session_pointers(session)
            or history_before != session.current_historical_prefix_sha256()
        ):
            raise Phase13PilotError("post-capture warmup or audit drifted")
        session.graph_evidence["replay_allocation"] = allocation_record
        session.graph_evidence["phase13_warmup_replays"] = WARMUP_STEPS
        session.admit(
            observed_outputs=((audit_checksum, audit_finite),),
            execution_path_passed=path_audit.passed,
            allocation_passed=allocation_passed,
            graph_passed=graph_passed,
        )
        raw_runner = run_fixed_l(
            session,
            measured_steps=MEASURED_STEPS,
            measured_batches=MEASURED_BATCHES,
        ).to_dict()
    runner = phase12._normalize_runner_result(raw_runner)
    graph_exec_after = int(session.graph.graph.raw_cuda_graph_exec())
    graph_path_after = phase12._write_cuda_graph_path_witness(
        graph=session.graph.graph,
        run_root=run_artifact_root,
        phase="after",
    )
    if (
        graph_exec_before <= 0
        or graph_exec_after != graph_exec_before
        or graph_path_before["normalized_sha256"]
        != graph_path_after["normalized_sha256"]
    ):
        raise Phase13PilotError("measured graph topology changed")
    memory = runner.get("memory_evidence")
    if (
        runner.get("output_finite") is not True
        or runner.get("cache_pointers_stable") is not True
        or runner.get("historical_cache_unchanged") is not True
        or runner.get("output_checksum") != audit_checksum
        or not isinstance(memory, Mapping)
        or memory.get("timing_allocated_delta_bytes") != 0
        or memory.get("timing_reserved_delta_bytes") != 0
        or runner.get("r_hbm") is not None
    ):
        raise Phase13PilotError("Pilot run stability or allocation failed")
    timing_samples = runner["timing"]["samples"]
    process_median = statistics.median(
        float(sample["cuda_ms_per_operation"]) for sample in timing_samples
    )
    telemetry_before = runner["telemetry_before"]
    telemetry_after = runner["telemetry_after"]
    temperature = phase12._telemetry_range(
        telemetry_before, telemetry_after, "temperature_celsius"
    )
    sm_clock = phase12._telemetry_range(
        telemetry_before, telemetry_after, "sm_clock_mhz"
    )
    memory_clock = phase12._telemetry_range(
        telemetry_before, telemetry_after, "memory_clock_mhz"
    )
    power = phase12._telemetry_range(telemetry_before, telemetry_after, "power_watts")
    owned_snapshot = phase12._capture_process_snapshot(
        supervised_pid=pid,
        supervised_start_ticks=start_ticks,
    )
    phase12._require_owned_snapshot(owned_snapshot, pid=pid, start_ticks=start_ticks)
    kernel_path_fingerprint = _canonical_sha256(
        {
            "configuration": configuration,
            "operation": operation.operation_fingerprint_sha256,
            "prior_g3": prior_g3,
            "runtime_adapter_fingerprint": runner["adapter_config_fingerprint"],
            "cache_layout_fingerprint": runner["cache_layout_fingerprint"],
            "backend": runtime_context.backend_fingerprint,
            "graph_topology": graph_path_before["normalized_sha256"],
            "path_audit": path_audit.to_dict(),
        }
    )
    allocation_fingerprint = _canonical_sha256(
        {
            "cache_accounting": runner["cache_accounting"],
            "cache_byte_breakdown": runner["cache_byte_breakdown"],
            "replay_allocation": allocation_record,
        }
    )
    return {
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
        "execution_git_sha": git_sha,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "gpu_uuid": GPU_UUID,
        "operation_fingerprint_sha256": operation.operation_fingerprint_sha256,
        "process_median_ms": process_median,
        "output_checksum": runner["output_checksum"],
        "kernel_path_fingerprint": kernel_path_fingerprint,
        "allocation_fingerprint": allocation_fingerprint,
        "temperature_min_c": temperature[0],
        "temperature_max_c": temperature[1],
        "sm_clock_min_mhz": int(sm_clock[0]),
        "sm_clock_max_mhz": int(sm_clock[1]),
        "memory_clock_min_mhz": int(memory_clock[0]),
        "memory_clock_max_mhz": int(memory_clock[1]),
        "power_min_w": power[0],
        "power_max_w": power[1],
        "finite_output": True,
        "no_backend_fallback": path_audit.passed,
        "allocation_stable": True,
        "kernel_path_stable": True,
        "gpu_exclusive": True,
        "warmup_replays": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "measured_batches": MEASURED_BATCHES,
        "graph_replay_allocation": allocation_record,
        "graph_path_before": graph_path_before,
        "graph_path_after": graph_path_after,
        "runner": runner,
        "container_runtime_attestation": attestation,
        "gpu_process_before_cuda": pre_snapshot,
        "gpu_process_owned_after_measurement": owned_snapshot,
        "pilot_only": True,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }


def _run_id(campaign_id: str, record: Mapping[str, Any]) -> str:
    value = (
        f"{campaign_id}-r{record['replicate_index']}-o{int(record['order_index']):03d}-"
        f"{record['method_config_id']}-b{record['batch_size']}-l{record['context_label']}"
    )
    if _RUN_RE.fullmatch(value) is None:
        raise Phase13PilotError("Phase 13 run ID is invalid")
    return value


def _write_status_manifest(
    *,
    run_root: Path,
    campaign_id: str,
    run_id: str,
    record: Mapping[str, Any],
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "kvbench-phase13-run-manifest-1.0.0",
        "campaign_id": campaign_id,
        "run_id": run_id,
        "status": status,
        "reason": reason,
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
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "measured_batches": MEASURED_BATCHES,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "selective_rerun": False,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    write_exclusive(run_root / "manifest.json", json_bytes(manifest))
    return manifest


def _write_nonlaunched_record(
    *,
    stage: Path,
    campaign_id: str,
    record: Mapping[str, Any],
    status: str,
    reason: str,
) -> dict[str, Any]:
    run_id = _run_id(campaign_id, record)
    run_root = stage / "runs" / run_id
    run_root.mkdir()
    write_exclusive(
        run_root / "disposition.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13-nonlaunch-disposition-1.0.0",
                "run_id": run_id,
                "status": status,
                "reason": reason,
                "cuda_process_launched": False,
                "record_preserved": True,
            }
        ),
    )
    return _write_status_manifest(
        run_root=run_root,
        campaign_id=campaign_id,
        run_id=run_id,
        record=record,
        status=status,
        reason=reason,
    )


def _supervision_passed(result: Any) -> bool:
    return phase12._supervision_passed(result)


def _run_one_process(
    *,
    stage: Path,
    campaign_id: str,
    record: Mapping[str, Any],
    git_sha: str,
) -> dict[str, Any]:
    run_id = _run_id(campaign_id, record)
    run_root = stage / "runs" / run_id
    run_root.mkdir()
    write_exclusive(
        run_root / "started.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13-run-start-1.0.0",
                "run_id": run_id,
                "started_at_utc": _utc_now(),
                "record": dict(record),
            }
        ),
    )
    pre = phase12._capture_process_snapshot()
    phase12._require_idle_snapshot(pre)
    command = (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/phase13_pilot.py"),
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
        git_sha,
        "--run-artifact-root",
        str(run_root),
    )
    result = run_supervised_command(
        command,
        working_directory=str(REPOSITORY_ROOT),
        environment=phase12._child_environment(),
        timeout_seconds=CHILD_TIMEOUT_SECONDS,
    )
    post = phase12._capture_process_snapshot()
    phase12._require_idle_snapshot(post)
    phase12._write_supervised_command_evidence(
        root=run_root,
        prefix="worker",
        result=result,
        pre_snapshot=pre,
        post_snapshot=post,
    )
    if not _supervision_passed(result):
        stderr = result.stderr.decode("utf-8", errors="replace")
        if "requires frozen B=1" in stderr or "requires frozen layers=32 B=1" in stderr:
            reason = "adapter_static_cache_rejects_batch_greater_than_one"
        else:
            reason = "supervised_worker_failed"
        write_exclusive(
            run_root / "failure.json",
            json_bytes(
                {
                    "schema_version": "kvbench-phase13-run-failure-1.0.0",
                    "run_id": run_id,
                    "failure_reason": reason,
                    "worker_returncode": result.returncode,
                    "selective_retry_permitted": False,
                    "campaign_preserved": True,
                }
            ),
        )
        return _write_status_manifest(
            run_root=run_root,
            campaign_id=campaign_id,
            run_id=run_id,
            record=record,
            status="runtime_failed",
            reason=reason,
        )
    matches = [
        line[len(WORKER_PREFIX) :]
        for line in result.stdout.decode("utf-8", errors="strict").splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(matches) != 1:
        raise Phase13PilotError("worker result channel differs")
    payload = json.loads(matches[0])
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise Phase13PilotError("worker result identity differs")
    write_exclusive(run_root / "result.json", json_bytes(payload))
    manifest = _write_status_manifest(
        run_root=run_root,
        campaign_id=campaign_id,
        run_id=run_id,
        record=record,
        status="completed",
        reason=None,
    )
    manifest["result_sha256"] = sha256_file(run_root / "result.json")
    # The initial write is immutable.  Bind the result in a separate receipt.
    write_exclusive(
        run_root / "result-binding.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13-run-result-binding-1.0.0",
                "run_id": run_id,
                "result_sha256": manifest["result_sha256"],
            }
        ),
    )
    return manifest


def run_campaign(*, stage: Path, campaign_id: str, git_sha: str) -> dict[str, Any]:
    identifier = _validate_campaign_id(campaign_id)
    phase12._require_authorized_container_runtime()
    if any(name in os.environ for name in _FORBIDDEN_ENVIRONMENT):
        raise Phase13PilotError("R2 or model credentials entered the measurement container")
    resolved = stage.resolve(strict=True)
    if not (resolved / "campaign-reservation.json").is_file():
        raise Phase13PilotError("Phase 13 campaign reservation is absent")
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
        raise Phase13PilotError("container source is not the clean execution commit")
    order = _strict_json(REPOSITORY_ROOT / ORDER_PATH)
    validate_execution_order(order)
    feasibility = build_feasibility_records(order)
    write_exclusive(resolved / "execution_order.json", json_bytes(order))
    write_exclusive(
        resolved / "unified" / "feasibility.json",
        json_bytes({"records": feasibility}),
    )
    write_exclusive(
        resolved / "campaign_manifest.json",
        json_bytes(
            {
                "schema_version": CAMPAIGN_SCHEMA,
                "campaign_id": identifier,
                "execution_git_sha": git_sha,
                "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
                "configurations": list(CONFIGURATIONS),
                "fingerprints": CONFIG_FINGERPRINTS,
                "batch_sizes": list(BATCH_SIZES),
                "context_labels": list(CONTEXT_LABELS),
                "top_context_mapping": {
                    "label": 131072,
                    "historical_prefix": 131071,
                    "total_attended": 131072,
                },
                "warmup_steps": WARMUP_STEPS,
                "measured_steps": MEASURED_STEPS,
                "measured_batches": MEASURED_BATCHES,
                "replicates": REPLICATES,
                "seeds": list(SEEDS),
                "planned_point_records": PLANNED_RECORD_COUNT,
                "execution_order_sha256": sha256_file(resolved / "execution_order.json"),
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
        if record["status"] == "capacity_infeasible":
            manifests.append(
                _write_nonlaunched_record(
                    stage=resolved,
                    campaign_id=identifier,
                    record=record,
                    status="capacity_infeasible",
                    reason=str(record["reason"]),
                )
            )
            continue
        if stop_reason is not None:
            manifests.append(
                _write_nonlaunched_record(
                    stage=resolved,
                    campaign_id=identifier,
                    record=record,
                    status="aborted",
                    reason=stop_reason,
                )
            )
            continue
        manifest = _run_one_process(
            stage=resolved,
            campaign_id=identifier,
            record=record,
            git_sha=git_sha,
        )
        manifests.append(manifest)
        if manifest["status"] == "runtime_failed":
            stop_reason = str(manifest["reason"])
    counts: dict[str, int] = defaultdict(int)
    for manifest in manifests:
        counts[str(manifest["status"])] += 1
    result = {
        "schema_version": "kvbench-phase13-local-campaign-result-1.0.0",
        "campaign_id": identifier,
        "execution_git_sha": git_sha,
        "planned_point_records": PLANNED_RECORD_COUNT,
        "status_counts": dict(sorted(counts.items())),
        "stopped_after_method_bug": stop_reason is not None,
        "stop_reason": stop_reason,
        "selective_reruns": 0,
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "phase13_status": "BLOCKED" if stop_reason else "LOCAL_COMPLETE",
        "durable_publication": "PENDING_HOST_SIDE",
        "r_hbm": None,
    }
    write_exclusive(resolved / "unified" / "local-campaign.json", json_bytes(result))
    return result


def _run_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "runs").glob("*/manifest.json")):
        manifest = _strict_json(manifest_path)
        run_root = manifest_path.parent
        result_path = run_root / "result.json"
        result = _strict_json(result_path) if result_path.is_file() else None
        records.append(
            {
                **manifest,
                "manifest_path": manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "result_path": (
                    result_path.relative_to(root).as_posix() if result is not None else None
                ),
                "result_sha256": sha256_file(result_path) if result is not None else None,
                "process_median_ms": (
                    float(result["process_median_ms"]) if result is not None else None
                ),
                "output_checksum": result.get("output_checksum") if result else None,
                "kernel_path_fingerprint": (
                    result.get("kernel_path_fingerprint") if result else None
                ),
                "allocation_fingerprint": (
                    result.get("allocation_fingerprint") if result else None
                ),
                "temperature_min_c": (
                    result.get("temperature_min_c") if result else None
                ),
                "temperature_max_c": (
                    result.get("temperature_max_c") if result else None
                ),
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
        )
    if len(records) != PLANNED_RECORD_COUNT:
        raise Phase13PilotError("Phase 13 run-record cardinality differs")
    return records


def _point_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["method_config_id"]),
                int(record["batch_size"]),
                int(record["context_label"]),
            )
        ].append(record)
    summaries: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        for batch in BATCH_SIZES:
            for label in CONTEXT_LABELS:
                matching = grouped[(configuration, batch, label)]
                statuses = [str(item["status"]) for item in matching]
                completed = [item for item in matching if item["status"] == "completed"]
                if len(completed) == REPLICATES:
                    stats = point_statistics(
                        [float(item["process_median_ms"]) for item in completed]
                    )
                    output_agreement = len(
                        {str(item["output_checksum"]) for item in completed}
                    ) == 1
                    path_agreement = len(
                        {str(item["kernel_path_fingerprint"]) for item in completed}
                    ) == 1
                    allocation_agreement = len(
                        {str(item["allocation_fingerprint"]) for item in completed}
                    ) == 1
                    disposition = classify_point(
                        statistics_record=stats,
                        agreements=output_agreement and path_agreement and allocation_agreement,
                    )
                else:
                    stats = {
                        "median_ms": None,
                        "mean_ms": None,
                        "standard_deviation_ms": None,
                        "minimum_ms": None,
                        "maximum_ms": None,
                        "cv": None,
                    }
                    output_agreement = path_agreement = allocation_agreement = False
                    if all(status == "capacity_infeasible" for status in statuses):
                        disposition = "capacity_infeasible"
                    elif "runtime_failed" in statuses:
                        disposition = "runtime_failed"
                    else:
                        disposition = "aborted"
                summaries.append(
                    {
                        "method_config_id": configuration,
                        "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                        "batch_size": batch,
                        "context_label": label,
                        "historical_context": actual_historical_context(label),
                        "replicate_count": len(matching),
                        "completed_replicates": len(completed),
                        **stats,
                        "output_checksum_agreement": output_agreement,
                        "kernel_path_agreement": path_agreement,
                        "allocation_agreement": allocation_agreement,
                        "disposition": disposition,
                        "monotonicity_warning_only": True,
                        "quality_status": "unvalidated",
                        "performance_claim_eligible": False,
                        "r_hbm": None,
                    }
                )
    if len(summaries) != 270:
        raise Phase13PilotError("Phase 13 point-summary cardinality differs")
    return summaries


def _fit_records(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        for batch in BATCH_SIZES:
            matching = [
                item
                for item in summaries
                if item["method_config_id"] == configuration
                and item["batch_size"] == batch
                and item["disposition"] == "stable"
            ]
            fit = provisional_knee_fit(
                [
                    (float(item["context_label"]), float(item["median_ms"]))
                    for item in matching
                ]
            )
            knee_model = fit.get("knee_model")
            knee = (
                knee_model.get("L_star")
                if isinstance(knee_model, Mapping)
                else None
            )
            density = knee_density(
                [int(item["context_label"]) for item in matching],
                float(knee) if isinstance(knee, (int, float)) else None,
            )
            records.append(
                {
                    "method_config_id": configuration,
                    "batch_size": batch,
                    "stable_point_count": len(matching),
                    "fit_status": fit["fit_status"],
                    "fit_json": json.dumps(fit, sort_keys=True, separators=(",", ":")),
                    "density_json": json.dumps(
                        density, sort_keys=True, separators=(",", ":")
                    ),
                    "pilot_only": True,
                    "quality_status": "unvalidated",
                    "performance_claim_eligible": False,
                }
            )
    return records


def _ratio_records(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(item["method_config_id"]), int(item["batch_size"]), int(item["context_label"])): item
        for item in summaries
    }
    rows: list[dict[str, Any]] = []
    for item in summaries:
        if item["method_config_id"] == "bf16":
            continue
        baseline = by_key[("bf16", int(item["batch_size"]), int(item["context_label"]))]
        calculated = bool(
            item["disposition"] == "stable" and baseline["disposition"] == "stable"
        )
        ratio = (
            float(baseline["median_ms"]) / float(item["median_ms"])
            if calculated
            else None
        )
        rows.append(
            {
                "method_config_id": item["method_config_id"],
                "batch_size": item["batch_size"],
                "context_label": item["context_label"],
                "provisional_same_work_ratio": ratio,
                "calculated": calculated,
                "pilot_only": True,
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
                "not_quality_preserving_speedup": True,
            }
        )
    return rows


def _parquet_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise Phase13PilotError("pinned pyarrow analysis environment is required") from error
    normalized = []
    for row in rows:
        normalized.append(
            {
                key: (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
                if key != "runner"
            }
        )
    table = pa.Table.from_pylist(normalized)
    pq.write_table(
        table,
        path,
        compression="zstd",
        version="2.6",
        use_dictionary=False,
        write_statistics=True,
    )


def _svg_plot(path: Path, *, title: str, message: str) -> None:
    safe_title = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    safe_message = (
        message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" '
        'viewBox="0 0 960 540">'
        '<rect width="960" height="540" fill="white"/>'
        f'<text x="48" y="72" font-family="sans-serif" font-size="28">{safe_title}</text>'
        f'<text x="48" y="140" font-family="sans-serif" font-size="18">{safe_message}</text>'
        '<text x="48" y="500" font-family="sans-serif" font-size="14">'
        'Pilot-only; quality unvalidated; not claim eligible</text></svg>'
    )
    write_exclusive(path, svg.encode("utf-8"))


def materialize_analysis(root: Path) -> dict[str, Any]:
    campaign = _strict_json(root / "unified" / "local-campaign.json")
    runs = _run_records(root)
    summaries = _point_summaries(runs)
    fits = _fit_records(summaries)
    ratios = _ratio_records(summaries)
    feasibility = _strict_json(root / "unified" / "feasibility.json")["records"]
    exclusions = [
        {
            "run_id": record["run_id"],
            "method_config_id": record["method_config_id"],
            "batch_size": record["batch_size"],
            "context_label": record["context_label"],
            "status": record["status"],
            "reason": record["reason"],
            "machine_readable": True,
        }
        for record in runs
        if record["status"] != "completed"
    ]
    _parquet_rows(root / "feasibility.parquet", feasibility)
    _parquet_rows(root / "raw_run_index.parquet", runs)
    _parquet_rows(root / "point_summary.parquet", summaries)
    _parquet_rows(root / "provisional_knees.parquet", fits)
    _parquet_rows(root / "exclusions.parquet", exclusions)
    stable = [item for item in summaries if item["disposition"] == "stable"]
    unstable = [item for item in summaries if item["disposition"] == "unstable"]
    maximum_cv = max((float(item["cv"]) for item in stable + unstable), default=None)
    qc = {
        "schema_version": "kvbench-phase13-pilot-qc-1.0.0",
        "campaign_id": campaign["campaign_id"],
        "phase13_status": campaign["phase13_status"],
        "planned_point_records": PLANNED_RECORD_COUNT,
        "status_counts": campaign["status_counts"],
        "stable_points": len(stable),
        "unstable_points": len(unstable),
        "maximum_cv": maximum_cv,
        "fit_status_counts": dict(
            sorted(
                {
                    status: sum(item["fit_status"] == status for item in fits)
                    for status in {str(item["fit_status"]) for item in fits}
                }.items()
            )
        ),
        "provisional_ratio_records": len(ratios),
        "provisional_ratios_calculated": sum(item["calculated"] for item in ratios),
        "selective_reruns": 0,
        "monotonicity_warning_only": True,
        "r_hbm": None,
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data_frozen": False,
        "performance_claim_eligible": False,
        "blocker": campaign["stop_reason"],
    }
    write_exclusive(root / "pilot_qc.json", json_bytes(qc))
    report = "\n".join(
        (
            "# Phase 13 Pilot QC",
            "",
            f"- Campaign: `{campaign['campaign_id']}`",
            f"- Status: `{campaign['phase13_status']}`",
            f"- Planned point records: {PLANNED_RECORD_COUNT}",
            f"- Status counts: `{json.dumps(campaign['status_counts'], sort_keys=True)}`",
            f"- Blocker: `{campaign['stop_reason']}`",
            "- Full Scan: `CLOSED`",
            "- Quality execution: `LOCKED`",
            "- No speedup, HBM, capacity, quality, or final-knee claim is made.",
            "",
        )
    )
    write_exclusive(root / "pilot_qc_report.md", report.encode("utf-8"))
    no_data = "No complete stable point set is available because the Pilot stopped on the recorded method boundary."
    for batch in BATCH_SIZES:
        _svg_plot(
            root / "pilot_plots" / f"latency-vs-context-b{batch}.svg",
            title=f"Pilot T versus L, B={batch}",
            message=no_data,
        )
    for method in ("bf16", "turboquant", "kivi", "kvquant"):
        _svg_plot(
            root / "pilot_plots" / f"method-{method}.svg",
            title=f"Pilot method view: {method}",
            message=no_data,
        )
    _svg_plot(root / "pilot_plots" / "cv-vs-context.svg", title="Pilot CV versus context", message=no_data)
    _svg_plot(
        root / "pilot_plots" / "allocated-ratio-vs-context.svg",
        title="Pilot allocated-byte ratio versus context",
        message=no_data,
    )
    _svg_plot(
        root / "pilot_plots" / "pilot-only-same-work-ratio.svg",
        title="Pilot-only same-work ratio versus context",
        message=no_data,
    )
    inventory = {
        "schema_version": "kvbench-phase13-scientific-inventory-1.0.0",
        "campaign_id": campaign["campaign_id"],
        "planned_records": PLANNED_RECORD_COUNT,
        "raw_run_records": len(runs),
        "point_summaries": len(summaries),
        "fit_records": len(fits),
        "exclusions": len(exclusions),
        "plot_count": 10,
        "r_hbm": None,
    }
    write_exclusive(root / "inventory.json", json_bytes(inventory))
    return qc


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase13PilotError("Phase 13 campaign contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase13PilotError("Phase 13 campaign contains an unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            files.append(path)
    return files


def seal_campaign(stage: Path, *, campaign_id: str) -> Path:
    identifier = _validate_campaign_id(campaign_id)
    root = stage.resolve(strict=True)
    qc = _strict_json(root / "pilot_qc.json")
    if qc.get("campaign_id") != identifier:
        raise Phase13PilotError("Phase 13 QC campaign identity differs")
    manifest = {
        "schema_version": "kvbench-phase13-artifact-manifest-1.0.0",
        "run_id": identifier,
        "campaign_id": identifier,
        "status": qc["phase13_status"],
        "created_at_utc": _utc_now(),
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "planned_point_records": PLANNED_RECORD_COUNT,
        "append_only": True,
        "complete_written_last": True,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    write_exclusive(root / "manifest.json", json_bytes(manifest))
    inventory_items = [
        {
            "path": path.relative_to(root).as_posix(),
            "role": "phase13_pilot_evidence",
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
                "files": inventory_items,
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
                "status": qc["phase13_status"],
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
        raise Phase13PilotError("Phase 13 finalized campaign already exists")
    rename_noreplace(root, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    validate_campaign(final, expected_campaign_id=identifier)
    return final


def validate_campaign(root: Path, *, expected_campaign_id: str | None = None) -> dict[str, Any]:
    artifact = validate_local_artifact(root, environ={})
    manifest = _strict_json(root / "manifest.json")
    identifier = str(manifest.get("campaign_id"))
    _validate_campaign_id(identifier)
    if expected_campaign_id is not None and identifier != expected_campaign_id:
        raise Phase13PilotError("Phase 13 campaign identity differs")
    runs = _run_records(root)
    order = _strict_json(root / "execution_order.json")
    validate_execution_order(order)
    qc = _strict_json(root / "pilot_qc.json")
    if (
        qc.get("campaign_id") != identifier
        or qc.get("planned_point_records") != PLANNED_RECORD_COUNT
        or qc.get("full_scan") != "CLOSED"
        or qc.get("quality_execution") != "LOCKED"
        or qc.get("r_hbm") is not None
        or sum(qc.get("status_counts", {}).values()) != PLANNED_RECORD_COUNT
        or len(runs) != PLANNED_RECORD_COUNT
    ):
        raise Phase13PilotError("Phase 13 QC or run topology differs")
    return {
        "status": "PASS",
        "campaign_id": identifier,
        "phase13_status": qc["phase13_status"],
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
    }


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
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
    parser.add_argument("--replicate-index", type=int)
    parser.add_argument("--order-index", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--run-artifact-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    if args.write_execution_order:
        if args.output is None:
            raise Phase13PilotError("--output is required")
        write_exclusive(args.output, json_bytes(derive_execution_order()))
        return 0
    if args.validate_execution_order:
        path = ORDER_PATH if args.output is None else args.output
        validate_execution_order(json.loads(path.read_text(encoding="utf-8")))
        print(json.dumps({"status": "PASS", "records": PLANNED_RECORD_COUNT}))
        return 0
    if args.print_feasibility_summary:
        order = json.loads((ORDER_PATH if args.output is None else args.output).read_text())
        records = build_feasibility_records(order)
        print(
            json.dumps(
                {
                    "planned": len(records),
                    "feasible": sum(item["status"] == "feasible" for item in records),
                    "capacity_infeasible": sum(
                        item["status"] == "capacity_infeasible" for item in records
                    ),
                    "entry_geometry_unsupported": sum(
                        not item["adapter_geometry_supported_at_entry"] for item in records
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.new_campaign_id:
        if args.git_sha is None:
            raise Phase13PilotError("--git-sha is required")
        print(new_campaign_id(args.git_sha))
        return 0
    if args.reserve_campaign:
        if args.campaign_id is None or args.git_sha is None:
            raise Phase13PilotError("campaign ID and Git SHA are required")
        print(reserve_campaign(campaign_id=args.campaign_id, git_sha=args.git_sha))
        return 0
    if args.run_campaign:
        if args.stage is None or args.campaign_id is None or args.git_sha is None:
            raise Phase13PilotError("stage, campaign ID, and Git SHA are required")
        print(
            json.dumps(
                run_campaign(
                    stage=args.stage,
                    campaign_id=args.campaign_id,
                    git_sha=args.git_sha,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.run_worker:
        required = (
            args.run_id,
            args.configuration,
            args.batch_size,
            args.context_label,
            args.replicate_index,
            args.order_index,
            args.git_sha,
            args.run_artifact_root,
        )
        if any(value is None for value in required):
            raise Phase13PilotError("worker identity arguments are required")
        payload = _run_worker(
            run_id=str(args.run_id),
            configuration=str(args.configuration),
            batch=int(args.batch_size),
            context_label=int(args.context_label),
            replicate_index=int(args.replicate_index),
            order_index=int(args.order_index),
            git_sha=str(args.git_sha),
            run_artifact_root=Path(args.run_artifact_root),
        )
        print(WORKER_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    if args.materialize_analysis:
        if args.stage is None:
            raise Phase13PilotError("--stage is required")
        print(json.dumps(materialize_analysis(args.stage), sort_keys=True))
        return 0
    if args.finalize_staged_campaign:
        if args.stage is None or args.campaign_id is None:
            raise Phase13PilotError("stage and campaign ID are required")
        print(seal_campaign(args.stage, campaign_id=args.campaign_id))
        return 0
    if args.validate_campaign:
        if args.artifact is None:
            raise Phase13PilotError("--artifact is required")
        print(json.dumps(validate_campaign(args.artifact), sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

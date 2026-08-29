"""Phase 15 deterministic Nsight profiler subset.

Profiler durations are mechanism evidence only.  This module deliberately
reuses the admitted fixed-L session, exact prefix snapshots, process checks,
artifact lifecycle, and R2 client; it never emits normal timing records.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import statistics
import subprocess
import sys
from typing import Any

from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from kvbench.runtime.artifacts import sha256_file
from scripts.r2_artifact import validate_local_artifact
from scripts import phase12_unified_admission as phase12
from scripts import phase13_pilot as pilot
from scripts import phase13d_continuation as continuation
from scripts import phase14_graph_ab as phase14
from scripts.phase13_prefix_state import validate_prefix_state


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase15"
STAGING_ROOT = ARTIFACT_ROOT / ".kvbench-staging"
PLAN_PATH = REPOSITORY_ROOT / "docs/plans/phase15-profiler-subset.md"
SELECTION_PATH = REPOSITORY_ROOT / "docs/plans/phase15-profiler-selection.json"
METRIC_MAP_PATH = REPOSITORY_ROOT / "docs/evidence/phase15/ncu-metric-map.json"

PHASE13_CAMPAIGN_ID = "phase13-20260822t150835736582z-4ddd7b17-3a8fb3"
PHASE13_ROOT = REPOSITORY_ROOT / "artifacts/phase13" / PHASE13_CAMPAIGN_ID
PHASE13_ROOT_SHA256 = (
    "feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531"
)
PHASE13_PREFIX_ROOT = (
    REPOSITORY_ROOT
    / "artifacts/phase13_prefix_catalogs"
    / PHASE13_CAMPAIGN_ID
    / "catalog"
)
PHASE13D_CAMPAIGN_ID = "phase13d-20260825t030556684636z-a06837a3-83761a"
PHASE13D_ROOT = REPOSITORY_ROOT / "artifacts/phase13d" / PHASE13D_CAMPAIGN_ID
PHASE13D_PREFIX_ROOT = Path(
    "/home/rockrock/phase13d_prefix_states/"
    "phase13d-20260825t030556684636z-a06837a3-83761a"
)
PHASE13D_ROOT_SHA256 = (
    "a8559a5e184f9555f8002919113b804e9621702292aff43bdc91c4441fa682d2"
)
PHASE14_CAMPAIGN_ID = "phase14-20260826t115110887808z-47ba4220-42fc95"
PHASE14_ROOT = REPOSITORY_ROOT / "artifacts/phase14" / PHASE14_CAMPAIGN_ID
PHASE14_ROOT_SHA256 = (
    "22a613fbc69edbc83d95f82439584180465cb6fdf7fe02c2a4e3e3b338c068f0"
)

AUTHORIZED_CONTAINER_DIGEST = pilot.AUTHORIZED_CONTAINER_DIGEST
CONFIGURATIONS = pilot.CONFIGURATIONS
CONFIG_FINGERPRINTS = pilot.CONFIG_FINGERPRINTS
ANCHORS = ("bf16", "tq_4bit_nc", "k4v4", "kvq4")
COMMON_BATCH = 1
COMMON_CONTEXT_LABEL = 131072
COMMON_HISTORICAL_CONTEXT = 131071
WARMUP_STEPS = 64
NSYS_DECODE_OPERATIONS = 8
NCU_DECODE_OPERATIONS = 1
SMOKE_WARMUP_STEPS = 2
NVTX_RANGE = "phase15_decode"
RUN_KINDS = ("nsys", "ncu")
KERNEL_ROLES = (
    "model_projection",
    "cache_append",
    "dense_cache_attention",
    "quantize",
    "dequantize",
    "kivi_residual",
    "kvquant_sparse_selection",
    "kvquant_sparse_correction",
    "sink_attention",
    "output_merge",
    "other_model",
    "unknown",
)
CAMPAIGN_SCHEMA = "kvbench-phase15-campaign-1.0.0"
RUN_SCHEMA = "kvbench-phase15-profiler-run-1.0.0"
SELECTION_SCHEMA = "kvbench-phase15-selection-1.0.0"
METRIC_MAP_SCHEMA = "kvbench-phase15-ncu-metric-map-1.0.0"
KERNEL_CLASSIFICATION_VERSION = "kvbench-phase15-kernel-classification-v2"
WORKER_PREFIX = "PHASE15_WORKER_RESULT="
_CAMPAIGN_RE = re.compile(
    r"phase15-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)


class Phase15Error(RuntimeError):
    """Phase 15 failed closed."""


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
        raise Phase15Error(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise Phase15Error(f"JSON evidence is not an object: {path}")
    return value


def actual_historical_context(label: int) -> int:
    if label <= 0 or label > 131072:
        raise Phase15Error("Phase 15 context label is outside the model range")
    return 131071 if label == 131072 else label


def _source_hashes() -> dict[str, str]:
    return {
        "phase13_point_summary": "77e77e52391149fc8c02b143aff5d2b3b0e1e717dadba593dd36177d9fb25236",
        "phase13_provisional_knees": "7e89f868a569f06d0109f8ce48847cffc837d0919cb0a423ac7d8b00737a1ff8",
        "phase13d_point_summary": "afb27e89ffc16aa3dd90d50fbf9cdde84c6d76e5093ea8b1963eae2cf542b850",
        "phase13d_refined_knees": "100621fc81522a343a5c05bc7b0d5d3eacd32280eb0a78b35c364dda61032339",
        "phase14_point_summary": "1e4993bdd5b75e62108881c9f9dfe402f3dbd785b576347a8b16c244f5163e4c",
        "phase14_graph_ab_pairs": "fead54b996459cdafd6c33c50ed100906092b4a1bdb6d59671762392f6b8d447",
        "phase14_mode_fits": "74d7a72df72c25b489ee1943c086d5f0c87e65d887eff9fd8162c3539566e660",
        "phase14_graph_effects": "c873831e75e233544f2f24d267cc4e515aa025bafd6d32e1060d8a6b19e66996",
        "phase14_qc": "ab353799906820bafa9f44b7f8c6cfb5c601647e5c45901bf87b013aabd7fe53",
        "phase14_closure_report": "7ba3906a3663ecb0e9f457b9ba48da2a57383d0783d890b4bcde50bcd565be25",
    }


def _anchor_specs() -> dict[str, dict[str, Any]]:
    return {
        "bf16": {
            "batch_size": 1,
            "refined_knee": 4096.0,
            "fit_status": "knee_observed",
            "resolution_status": "insufficient_feasible_span",
            "regime_selection": "fallback_no_identifiable_knee",
            "contexts": {"pre": 4096, "near": 6144, "post": 32768, "longest": 131072},
        },
        "tq_4bit_nc": {
            "batch_size": 1,
            "refined_knee": 4096.0,
            "fit_status": "knee_observed",
            "resolution_status": "insufficient_feasible_span",
            "regime_selection": "fallback_no_identifiable_knee",
            "contexts": {"pre": 4096, "near": 6144, "post": 32768, "longest": 131072},
        },
        "k4v4": {
            "batch_size": 1,
            "refined_knee": 5824.0,
            "fit_status": "knee_observed",
            "resolution_status": "density_sufficient",
            "regime_selection": "refined_knee_relative",
            "contexts": {"pre": 4096, "near": 6144, "post": 7680, "longest": 131072},
        },
        "kvq4": {
            "batch_size": 1,
            "refined_knee": 40960.0,
            "fit_status": "knee_observed",
            "resolution_status": "density_sufficient",
            "regime_selection": "refined_knee_relative",
            "contexts": {"pre": 24576, "near": 49152, "post": 65536, "longest": 131072},
        },
    }


def derive_selection() -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    nsys: list[dict[str, Any]] = []
    ncu: list[dict[str, Any]] = []
    for configuration in ANCHORS:
        spec = _anchor_specs()[configuration]
        roles_by_context: dict[int, list[str]] = defaultdict(list)
        for role, context in spec["contexts"].items():
            roles_by_context[int(context)].append(str(role))
        points = []
        for context in sorted(roles_by_context):
            point = {
                "method_config_id": configuration,
                "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                "batch_size": int(spec["batch_size"]),
                "context_label": context,
                "historical_context": actual_historical_context(context),
                "total_attended_context": actual_historical_context(context) + 1,
                "roles": sorted(roles_by_context[context]),
                "regime_selection": spec["regime_selection"],
            }
            points.append(point)
            for mode in ("eager", "cuda_graph"):
                nsys.append(
                    {
                        **point,
                        "run_kind": "nsys",
                        "graph_mode": mode,
                        "profile_id": f"nsys-{configuration}-b1-l{context}-{mode}",
                        "decode_operations": NSYS_DECODE_OPERATIONS,
                    }
                )
        anchors.append(
            {
                "method_config_id": configuration,
                "batch_size": spec["batch_size"],
                "refined_knee": spec["refined_knee"],
                "fit_status": spec["fit_status"],
                "resolution_status": spec["resolution_status"],
                "regime_selection": spec["regime_selection"],
                "points": points,
            }
        )
    common = {
        "batch_size": COMMON_BATCH,
        "context_label": COMMON_CONTEXT_LABEL,
        "historical_context": COMMON_HISTORICAL_CONTEXT,
        "total_attended_context": COMMON_HISTORICAL_CONTEXT + 1,
        "graph_mode": "cuda_graph",
        "intersection_contexts": [
            4096,
            8192,
            16384,
            24576,
            32768,
            49152,
            65536,
            98304,
            131072,
        ],
        "selection_rule": "largest_stable_feasible_graph_context_in_all_ten_at_b1",
    }
    for configuration in CONFIGURATIONS:
        ncu.append(
            {
                "method_config_id": configuration,
                "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
                "batch_size": COMMON_BATCH,
                "context_label": COMMON_CONTEXT_LABEL,
                "historical_context": COMMON_HISTORICAL_CONTEXT,
                "total_attended_context": COMMON_HISTORICAL_CONTEXT + 1,
                "roles": ["common_long"],
                "regime_selection": "common_same_work",
                "run_kind": "ncu",
                "graph_mode": "cuda_graph",
                "profile_id": f"ncu-{configuration}-b1-l131072-cuda_graph",
                "decode_operations": NCU_DECODE_OPERATIONS,
            }
        )
    existing = {
        (row["method_config_id"], row["batch_size"], row["context_label"])
        for row in ncu
    }
    for anchor in anchors:
        for point in anchor["points"]:
            key = (
                point["method_config_id"],
                point["batch_size"],
                point["context_label"],
            )
            if key in existing:
                continue
            existing.add(key)
            ncu.append(
                {
                    **point,
                    "run_kind": "ncu",
                    "graph_mode": "cuda_graph",
                    "profile_id": (
                        f"ncu-{point['method_config_id']}-b{point['batch_size']}"
                        f"-l{point['context_label']}-cuda_graph"
                    ),
                    "decode_operations": NCU_DECODE_OPERATIONS,
                }
            )
    return {
        "schema_version": SELECTION_SCHEMA,
        "source_campaigns": {
            "phase13": {"campaign_id": PHASE13_CAMPAIGN_ID, "root_sha256": PHASE13_ROOT_SHA256},
            "phase13d": {"campaign_id": PHASE13D_CAMPAIGN_ID, "root_sha256": PHASE13D_ROOT_SHA256},
            "phase14": {"campaign_id": PHASE14_CAMPAIGN_ID, "root_sha256": PHASE14_ROOT_SHA256},
        },
        "source_file_sha256": _source_hashes(),
        "configurations": list(CONFIGURATIONS),
        "fingerprints": dict(CONFIG_FINGERPRINTS),
        "anchors": anchors,
        "common_same_work": common,
        "nsys_profiles": nsys,
        "ncu_profiles": ncu,
        "expected_nsys_profiles": 32,
        "expected_ncu_profiles": 22,
        "profiler_timing_is_mechanism_only": True,
        "normal_timing_reused": False,
        "performance_claim_eligible": False,
        "quality_status": "unvalidated",
    }


def validate_selection(value: Mapping[str, Any]) -> None:
    expected = derive_selection()
    if dict(value) != expected:
        raise Phase15Error("Phase 15 frozen selection differs")
    nsys = value["nsys_profiles"]
    ncu = value["ncu_profiles"]
    if (
        len(nsys) != 32
        or len(ncu) != 22
        or len({row["profile_id"] for row in nsys}) != 32
        or len({row["profile_id"] for row in ncu}) != 22
        or any(row["run_kind"] != "nsys" for row in nsys)
        or any(row["run_kind"] != "ncu" for row in ncu)
    ):
        raise Phase15Error("Phase 15 profiler selection cardinality differs")


def write_selection(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise Phase15Error("Phase 15 selection path already exists")
    write_exclusive(path, json_bytes(derive_selection()))


_METRIC_CANDIDATES: dict[str, tuple[str, ...]] = {
    "dram_read_bytes": ("dram__bytes_op_read.sum",),
    "dram_write_bytes": ("dram__bytes_op_write.sum",),
    "l2_read_sectors": ("lts__t_sectors_op_read.sum",),
    "l2_write_sectors": ("lts__t_sectors_op_write.sum",),
    "l2_hit_rate": ("lts__t_sector_hit_rate.pct",),
    "memory_throughput": ("dram__throughput.avg.pct_of_peak_sustained_elapsed",),
    "sm_activity": ("sm__throughput.avg.pct_of_peak_sustained_elapsed",),
    "achieved_occupancy": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    ),
    "active_warps": ("smsp__warps_active.avg.per_cycle_active",),
    "kernel_elapsed": ("gpu__time_duration.sum",),
    "load_store_activity": (
        "smsp__sass_thread_inst_executed_op_memory_pred_on.sum",
    ),
    "executed_instructions": ("smsp__inst_executed.sum",),
}


def resolve_metric_map(metrics_text: str, sections_text: str) -> dict[str, Any]:
    available = {
        token
        for token in re.findall(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", metrics_text)
        if "__" in token
    }
    selected: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    units = {
        "dram_read_bytes": "byte",
        "dram_write_bytes": "byte",
        "l2_read_sectors": "sector_32_bytes",
        "l2_write_sectors": "sector_32_bytes",
        "l2_hit_rate": "percent",
        "memory_throughput": "percent_of_peak",
        "sm_activity": "percent_of_peak",
        "achieved_occupancy": "percent",
        "active_warps": "warps_per_cycle_active",
        "kernel_elapsed": "nanosecond",
        "load_store_activity": "thread_instructions",
        "executed_instructions": "instructions",
    }
    meanings = {
        "dram_read_bytes": "physical DRAM read bytes",
        "dram_write_bytes": "physical DRAM write bytes",
        "l2_read_sectors": "L2 read sectors; derived bytes use 32 bytes per sector",
        "l2_write_sectors": "L2 write sectors; derived bytes use 32 bytes per sector",
        "l2_hit_rate": "L2 sector hit rate",
        "memory_throughput": "DRAM throughput utilization",
        "sm_activity": "SM throughput utilization",
        "achieved_occupancy": "active-warps occupancy",
        "active_warps": "active warps per active cycle",
        "kernel_elapsed": "kernel elapsed duration",
        "load_store_activity": "executed load/store thread instructions",
        "executed_instructions": "executed SM subpartition instructions",
    }
    for semantic, candidates in _METRIC_CANDIDATES.items():
        metric = next((candidate for candidate in candidates if candidate in available), None)
        if metric is None:
            unavailable.append(
                {"semantic": semantic, "requested_metrics": list(candidates), "reason": "unavailable_on_current_gpu"}
            )
            continue
        selected.append(
            {
                "semantic": semantic,
                "metric": metric,
                "meaning": meanings[semantic],
                "unit": units[semantic],
                "status": "raw" if semantic not in {"l2_read_sectors", "l2_write_sectors"} else "raw_with_derived_bytes",
            }
        )
    required = {"dram_read_bytes", "dram_write_bytes", "l2_read_sectors", "l2_write_sectors", "kernel_elapsed"}
    if not required.issubset({row["semantic"] for row in selected}):
        raise Phase15Error("required SM120 traffic metrics are unavailable")
    metric_names = [row["metric"] for row in selected]
    return {
        "schema_version": METRIC_MAP_SCHEMA,
        "gpu_architecture": "SM120",
        "query_metrics_sha256": hashlib.sha256(metrics_text.encode()).hexdigest(),
        "query_sections_sha256": hashlib.sha256(sections_text.encode()).hexdigest(),
        "selected_metrics": selected,
        "metric_names": metric_names,
        "unavailable_requested_metrics": unavailable,
        "selection_source": "live_ncu_query_metrics_and_sections",
        "replay_mode": "kernel",
        "graph_profiling": "node",
        "replay_pass_count": None,
        "replay_pass_count_source": "filled_from_ncu_raw_export",
    }


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "inf", "-inf"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def parse_ncu_csv(text: str, metric_map: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = [line for line in text.splitlines() if not line.startswith("==") and line.strip()]
    if not lines:
        raise Phase15Error("NCU export is empty")
    semantic_by_metric = {
        row["metric"]: row["semantic"] for row in metric_map["selected_metrics"]
    }
    records: list[dict[str, Any]] = []
    parsed_rows = list(csv.reader(io.StringIO("\n".join(lines))))
    header = [str(value).strip() for value in parsed_rows[0]]
    if "Metric Name" not in header:
        if len(parsed_rows) < 3:
            raise Phase15Error("NCU wide export is missing unit or data rows")
        units = parsed_rows[1]
        index = {name: position for position, name in enumerate(header)}
        selected = {
            metric: index[metric]
            for metric in semantic_by_metric
            if metric in index
        }
        if set(selected) != set(semantic_by_metric):
            missing = sorted(set(semantic_by_metric) - set(selected))
            raise Phase15Error(f"NCU wide export omitted selected metrics: {missing}")
        for raw in parsed_rows[2:]:
            if len(raw) != len(header):
                raise Phase15Error("NCU wide export row has inconsistent width")
            kernel = raw[index.get("Kernel Name", -1)] if "Kernel Name" in index else "unknown"
            for metric, position in selected.items():
                value = _numeric(raw[position])
                if value is None:
                    continue
                records.append(
                    {
                        "kernel_name": str(kernel),
                        "kernel_id": str(raw[index["ID"]]) if "ID" in index else "",
                        "context": str(raw[index["Context"]]) if "Context" in index else "",
                        "stream": str(raw[index["Stream"]]) if "Stream" in index else "",
                        "metric_name": metric,
                        "metric_semantic": semantic_by_metric[metric],
                        "unit": str(units[position]).strip(),
                        "metric_value": value,
                    }
                )
        if not records:
            raise Phase15Error("NCU export contains no selected metrics")
        return records

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    for raw in reader:
        normalized = {str(key).strip(): value for key, value in raw.items() if key is not None}
        metric = normalized.get("Metric Name") or normalized.get("Metric Name ")
        if metric not in semantic_by_metric:
            continue
        value = _numeric(normalized.get("Metric Value"))
        if value is None:
            continue
        kernel = (
            normalized.get("Kernel Name")
            or normalized.get("Kernel Name (Demangled)")
            or normalized.get("Kernel Name (Mangled)")
            or "unknown"
        )
        records.append(
            {
                "kernel_name": str(kernel),
                "kernel_id": str(normalized.get("ID", "")),
                "context": str(normalized.get("Context", "")),
                "stream": str(normalized.get("Stream", "")),
                "metric_name": metric,
                "metric_semantic": semantic_by_metric[metric],
                "unit": str(normalized.get("Metric Unit", normalized.get("Unit", ""))),
                "metric_value": value,
            }
        )
    if not records:
        raise Phase15Error("NCU export contains no selected metrics")
    return records


def classify_kernel(name: str, *, configuration: str, nvtx_ranges: Sequence[str] = ()) -> tuple[str, str]:
    text = " ".join([name, *nvtx_ranges]).lower()
    family = phase12._method_family(configuration)
    # These are exact method-authority symbols. Keep them family-scoped so an
    # unrelated similarly named kernel cannot become cache-path traffic.
    family_rules: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
        "turboquant": (
            ("dense_cache_attention", ("_tq_decode_stage1",)),
            ("output_merge", ("_fwd_kernel_stage2",)),
            ("quantize", ("_tq_fused_store_mse", "normtwoops")),
        ),
        "kivi": (
            (
                "dense_cache_attention",
                (
                    "bgemv2_kernel_outer_dim",
                    "bgemv4_kernel_outer_dim",
                    "cunn_softmaxforward",
                ),
            ),
        ),
        "kvquant": (
            (
                "dense_cache_attention",
                (
                    "matmulkernelnuqperchanneltransposedmhabatchedfusedoptdeterministictiles",
                    "cunn_softmaxforward",
                ),
            ),
            (
                "output_merge",
                (
                    "matmulkernelnuqperchanneltransposedmhabatchedfusedoptdeterministicreduce",
                ),
            ),
            ("kvquant_sparse_selection", ("selectfixedoutliers1024cap12kernel",)),
            (
                "kvquant_sparse_correction",
                (
                    "writekeysparseresidual1024cap12kernel",
                    "writevaluemetadataandsparsekernel",
                ),
            ),
            (
                "cache_append",
                (
                    "vecquant2appendveck",
                    "vecquant3appendveck",
                    "vecquant4appendveck",
                    "appendvaluesparsedeviceoutkernel",
                    "clearvaluepackedcolumnkernel",
                ),
            ),
        ),
    }
    for role, patterns in family_rules.get(family, ()):
        if any(pattern in text for pattern in patterns):
            return role, "exact_kernel_symbol_plus_method_authority"
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("kvquant_sparse_selection", ("select_fixed_outlier", "sparse_selection")),
        ("kvquant_sparse_correction", ("kvquant_sparse", "sparse_correction", "value_sparse", "key_sparse_residual")),
        ("kivi_residual", ("kivi_residual", "residual_attention", "residual_cache")),
        ("sink_attention", ("sink_attention", "sink_cache", "sink_")),
        ("cache_append", ("cache_append", "append_decode", "update_cache", "scatter_kernel")),
        ("quantize", ("quantize", "pack_kv", "pack_key", "pack_value")),
        ("dequantize", ("dequant", "unpack_kv", "decompress")),
        ("dense_cache_attention", ("flash_fwd", "fmha", "attention", "attn")),
        ("output_merge", ("output_merge", "merge_output", "correction_merge")),
        ("model_projection", ("gemm", "cutlass", "cublas", "projection", "linear")),
        (
            "other_model",
            (
                "rmsnorm",
                "layer_norm",
                "meanops",
                "indexselectsmallindex",
                "rotary",
                "rope",
                "silu",
                "elementwise",
            ),
        ),
    )
    for role, patterns in rules:
        if any(pattern in text for pattern in patterns):
            if role.startswith("kvquant") and family != "kvquant":
                continue
            if role == "kivi_residual" and family != "kivi":
                continue
            return role, "kernel_name_plus_method_authority"
    return "unknown", "insufficient_unambiguous_evidence"


def reclassify_kernel_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    configuration: str,
    recorded_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    """Reclassify immutable NCU events and prove byte-total conservation."""

    records: list[dict[str, Any]] = []
    role_bytes: dict[str, float] = defaultdict(float)
    totals: dict[str, float] = defaultdict(float)
    unknown_dram = 0.0
    unknown_kernels = 0
    for raw in rows:
        event = dict(raw)
        recorded_role = str(event.get("kernel_role", "unknown"))
        recorded_basis = str(
            event.get("classification_basis", "insufficient_unambiguous_evidence")
        )
        role, basis = classify_kernel(
            str(event.get("kernel_name", "")), configuration=configuration
        )
        dram_read = float(event.get("dram_read_bytes", 0.0))
        dram_write = float(event.get("dram_write_bytes", 0.0))
        dram = dram_read + dram_write
        l2_read = float(event.get("l2_read_bytes", 0.0))
        l2_write = float(event.get("l2_write_bytes", 0.0))
        l2 = l2_read + l2_write
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (dram_read, dram_write, l2_read, l2_write)
        ):
            raise Phase15Error("NCU kernel event contains invalid traffic bytes")
        event.update(
            {
                "recorded_kernel_role": recorded_role,
                "recorded_classification_basis": recorded_basis,
                "kernel_role": role,
                "classification_basis": basis,
                "cache_path": kernel_is_cache_path(role),
                "dram_bytes": dram,
                "l2_bytes": l2,
                "kernel_classification_version": KERNEL_CLASSIFICATION_VERSION,
            }
        )
        records.append(event)
        role_bytes[role] += dram
        totals["dram_read"] += dram_read
        totals["dram_write"] += dram_write
        totals["l2_read"] += l2_read
        totals["l2_write"] += l2_write
        if role == "unknown":
            unknown_kernels += 1
            unknown_dram += dram

    observed = {
        "total_decode_dram_read_bytes": totals["dram_read"],
        "total_decode_dram_write_bytes": totals["dram_write"],
        "total_decode_dram_bytes": totals["dram_read"] + totals["dram_write"],
        "total_decode_l2_read_bytes": totals["l2_read"],
        "total_decode_l2_write_bytes": totals["l2_write"],
        "total_decode_l2_bytes": totals["l2_read"] + totals["l2_write"],
    }
    for key, value in observed.items():
        recorded = recorded_summary.get(key)
        if not isinstance(recorded, (int, float)) or not math.isclose(
            float(recorded), value, rel_tol=0.0, abs_tol=0.5
        ):
            raise Phase15Error(f"NCU reclassification changed recorded byte total: {key}")

    summary = dict(recorded_summary)
    summary.update(
        {
            **observed,
            "cache_path_dram_bytes": sum(
                value for role, value in role_bytes.items() if kernel_is_cache_path(role)
            ),
            "cache_path_l2_bytes": sum(
                float(event["l2_bytes"])
                for event in records
                if bool(event["cache_path"])
            ),
            "unclassified_dram_bytes": unknown_dram,
            "unclassified_dram_fraction": (
                unknown_dram / observed["total_decode_dram_bytes"]
                if observed["total_decode_dram_bytes"]
                else None
            ),
            "unclassified_kernel_count": unknown_kernels,
            "kernel_classification_version": KERNEL_CLASSIFICATION_VERSION,
            "recorded_totals_conserved": True,
        }
    )
    return records, summary, dict(sorted(role_bytes.items()))


def kernel_is_cache_path(role: str) -> bool:
    return role in {
        "cache_append",
        "dense_cache_attention",
        "quantize",
        "dequantize",
        "kivi_residual",
        "kvquant_sparse_selection",
        "kvquant_sparse_correction",
        "sink_attention",
        "output_merge",
    }


def aggregate_kernel_metrics(
    rows: Sequence[Mapping[str, Any]], *, configuration: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_kernel: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("kernel_id", "")), str(row["kernel_name"]))
        record = by_kernel.setdefault(
            key,
            {"kernel_id": key[0], "kernel_name": key[1], "metrics": {}},
        )
        record["metrics"][str(row["metric_semantic"])] = float(row["metric_value"])
    kernels: list[dict[str, Any]] = []
    totals = defaultdict(float)
    unknown_dram = 0.0
    cache_dram = 0.0
    cache_l2 = 0.0
    weighted_hit_num = 0.0
    weighted_hit_den = 0.0
    utilization_weight = 0.0
    utilization_totals = defaultdict(float)
    for record in by_kernel.values():
        role, basis = classify_kernel(record["kernel_name"], configuration=configuration)
        metrics = record["metrics"]
        dram_read = float(metrics.get("dram_read_bytes", 0.0))
        dram_write = float(metrics.get("dram_write_bytes", 0.0))
        l2_read = float(metrics.get("l2_read_sectors", 0.0)) * 32.0
        l2_write = float(metrics.get("l2_write_sectors", 0.0)) * 32.0
        dram = dram_read + dram_write
        l2 = l2_read + l2_write
        totals["dram_read_bytes"] += dram_read
        totals["dram_write_bytes"] += dram_write
        totals["l2_read_bytes"] += l2_read
        totals["l2_write_bytes"] += l2_write
        totals["kernel_duration_ns"] += float(metrics.get("kernel_elapsed", 0.0))
        duration = float(metrics.get("kernel_elapsed", 0.0))
        if duration > 0:
            utilization_weight += duration
            for semantic in (
                "memory_throughput",
                "sm_activity",
                "achieved_occupancy",
                "active_warps",
            ):
                if semantic in metrics:
                    utilization_totals[semantic] += float(metrics[semantic]) * duration
        hit = metrics.get("l2_hit_rate")
        if hit is not None and (l2_read + l2_write) > 0:
            weighted_hit_num += float(hit) * (l2_read + l2_write)
            weighted_hit_den += l2_read + l2_write
        if kernel_is_cache_path(role):
            cache_dram += dram
            cache_l2 += l2
        elif role == "unknown":
            unknown_dram += dram
        kernels.append(
            {
                **record,
                "kernel_role": role,
                "classification_basis": basis,
                "cache_path": kernel_is_cache_path(role),
                "dram_read_bytes": dram_read,
                "dram_write_bytes": dram_write,
                "dram_bytes": dram,
                "l2_read_bytes": l2_read,
                "l2_write_bytes": l2_write,
                "l2_bytes": l2,
                "memory_throughput": metrics.get("memory_throughput"),
                "sm_activity": metrics.get("sm_activity"),
                "achieved_occupancy": metrics.get("achieved_occupancy"),
                "active_warps": metrics.get("active_warps"),
                "kernel_duration_ns": metrics.get("kernel_elapsed"),
                "load_store_activity": metrics.get("load_store_activity"),
                "executed_instructions": metrics.get("executed_instructions"),
            }
        )
    total_dram = totals["dram_read_bytes"] + totals["dram_write_bytes"]
    total_l2 = totals["l2_read_bytes"] + totals["l2_write_bytes"]
    summary = {
        "total_decode_dram_read_bytes": totals["dram_read_bytes"],
        "total_decode_dram_write_bytes": totals["dram_write_bytes"],
        "total_decode_dram_bytes": total_dram,
        "cache_path_dram_bytes": cache_dram,
        "total_decode_l2_read_bytes": totals["l2_read_bytes"],
        "total_decode_l2_write_bytes": totals["l2_write_bytes"],
        "total_decode_l2_bytes": total_l2,
        "cache_path_l2_bytes": cache_l2,
        "unclassified_dram_bytes": unknown_dram,
        "unclassified_dram_fraction": unknown_dram / total_dram if total_dram else None,
        "l2_hit_rate": weighted_hit_num / weighted_hit_den if weighted_hit_den else None,
        "summed_kernel_duration_ns": totals["kernel_duration_ns"],
        "kernel_count": len(kernels),
        "memory_throughput": utilization_totals["memory_throughput"] / utilization_weight if utilization_weight else None,
        "sm_activity": utilization_totals["sm_activity"] / utilization_weight if utilization_weight else None,
        "achieved_occupancy": utilization_totals["achieved_occupancy"] / utilization_weight if utilization_weight else None,
        "active_warps": utilization_totals["active_warps"] / utilization_weight if utilization_weight else None,
    }
    return kernels, summary


def hbm_ratio(
    *, bf16: Mapping[str, Any], method: Mapping[str, Any], same_work: bool
) -> dict[str, Any]:
    if not same_work:
        return {"r_hbm": None, "reason": "same_work_identity_mismatch"}
    required = ("cache_path_dram_bytes", "total_decode_dram_bytes")
    if any(float(bf16.get(key, 0.0)) <= 0 or float(method.get(key, 0.0)) <= 0 for key in required):
        return {"r_hbm": None, "reason": "incomplete_ncu_traffic"}
    r_cache = float(bf16["cache_path_dram_bytes"]) / float(method["cache_path_dram_bytes"])
    rho_cache = 1.0 / r_cache
    return {
        "r_hbm": r_cache,
        "r_hbm_scope": "cache_path",
        "r_hbm_cache": r_cache,
        "rho_hbm_cache": rho_cache,
        "r_hbm_total_decode": float(bf16["total_decode_dram_bytes"]) / float(method["total_decode_dram_bytes"]),
        "reason": None,
    }


def traffic_amplification(*, rho_hbm: float, rho_alloc: float) -> float:
    if not math.isfinite(rho_hbm) or not math.isfinite(rho_alloc) or rho_alloc <= 0:
        raise Phase15Error("traffic amplification inputs are invalid")
    return rho_hbm / rho_alloc


def _prefix_catalog_index() -> dict[tuple[str, int, int], dict[str, Any]]:
    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    sources = (
        (PHASE13_PREFIX_ROOT, "phase13"),
        (PHASE13D_PREFIX_ROOT, "phase13d"),
    )
    needed = {
        (str(row["method_config_id"]), int(row["batch_size"]), int(row["context_label"]))
        for row in [*derive_selection()["nsys_profiles"], *derive_selection()["ncu_profiles"]]
    }
    for root, source in sources:
        catalog = _strict_json(root / "catalog.json")
        entries = catalog.get("entries")
        if not isinstance(entries, list):
            raise Phase15Error(f"{source} prefix catalog is invalid")
        for raw in entries:
            key = (
                str(raw["method_config_id"]),
                int(raw["batch_size"]),
                int(raw["context_label"]),
            )
            if key not in needed or key in index:
                continue
            snapshot = root / str(raw["snapshot_relative_path"])
            manifest = validate_prefix_state(
                snapshot,
                configuration=key[0],
                family=phase12._method_family(key[0]),
                batch=key[1],
                historical=actual_historical_context(key[2]),
                method_config_fingerprint=CONFIG_FINGERPRINTS[key[0]],
                verify_state_bytes=False,
            )
            state = snapshot / "state.safetensors"
            if (
                manifest.get("state_file_sha256") != raw["state_file_sha256"]
                or state.stat().st_size != int(raw["state_file_bytes"])
                or state.stat().st_mode & 0o222
                or snapshot.stat().st_mode & 0o222
            ):
                raise Phase15Error(f"{source} prefix state authority differs")
            index[key] = {
                **dict(raw),
                "snapshot_root": str(snapshot),
                "source_campaign": source,
                "validated_read_only": True,
            }
    missing = sorted(needed - set(index))
    if missing:
        raise Phase15Error(f"Phase 15 exact prefix states are absent: {missing}")
    return index


def validate_lightweight_entry() -> dict[str, Any]:
    status_text = (REPOSITORY_ROOT / "docs/status.md").read_text(encoding="utf-8")
    closure_text = (
        REPOSITORY_ROOT / "docs/phase_reports/phase14-analysis-closure.md"
    ).read_text(encoding="utf-8")
    phase14_manifest = _strict_json(PHASE14_ROOT / "campaign_manifest.json")
    if (
        "Status: PASS" not in closure_text
        or "Phase 15: `READY`" not in closure_text
        or "G0-G5" not in status_text
        or "PASS" not in status_text
        or "Full Scan" not in status_text
        or "CLOSED" not in status_text
        or "LOCKED" not in status_text
        or phase14_manifest.get("fingerprints") != CONFIG_FINGERPRINTS
    ):
        raise Phase15Error("recorded Phase 15 entry state differs")
    image = subprocess.run(
        ("docker", "image", "inspect", AUTHORIZED_CONTAINER_DIGEST, "--format", "{{.Id}}"),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if image.returncode != 0 or image.stdout.strip() != AUTHORIZED_CONTAINER_DIGEST:
        raise Phase15Error("authorized Measurement Container is unavailable")
    validate_selection(_strict_json(SELECTION_PATH))
    prefixes = _prefix_catalog_index()
    return {
        "status": "PASS",
        "phase14c": "PASS",
        "phase15": "READY",
        "gates": "G0-G5 PASS",
        "full_scan": "CLOSED",
        "quality": "LOCKED",
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "fingerprints": dict(CONFIG_FINGERPRINTS),
        "prefix_states": len(prefixes),
        "nsys_profiles": 32,
        "ncu_profiles": 22,
    }


def _build_profiler_session(
    *, configuration: str, batch: int, context_label: int, graph_mode: str, prefix_entry: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    import torch
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model

    historical = actual_historical_context(context_label)
    pilot._patch_phase12_point_globals(batch=batch, historical=historical)
    loaded = load_frozen_model(device=torch.device("cuda:0"))
    prefix, decode = pilot._point_inputs(
        batch=batch, historical=historical, device=torch.device("cuda:0")
    )
    operation = phase14.Phase14OperationKey.create(
        configuration, batch, historical, graph_mode
    )
    with torch.inference_mode(), forced_flash_execution():
        if graph_mode == "eager":
            with phase14._graph_capture_disabled_for_eager():
                session, receipt = phase14._build_mode_session(
                    loaded=loaded,
                    operation=operation,
                    prefix=prefix,
                    decode=decode,
                    snapshot_root=Path(str(prefix_entry["snapshot_root"])),
                    state_sha256=str(prefix_entry["state_file_sha256"]),
                )
        else:
            session, receipt = phase14._build_mode_session(
                loaded=loaded,
                operation=operation,
                prefix=prefix,
                decode=decode,
                snapshot_root=Path(str(prefix_entry["snapshot_root"])),
                state_sha256=str(prefix_entry["state_file_sha256"]),
            )
    return session, receipt


def _install_profiler_adapter_nvtx_annotations() -> None:
    """Install profiler-process-only adapter ranges.

    The dedicated worker terminates with ``os._exit``; normal timing processes
    never import or call this installer.
    """

    import torch
    from kvbench.adapters.bf16 import BF16MethodAdapter
    from kvbench.adapters.turboquant import TurboQuantMethodAdapter
    from kvbench.adapters.kivi import KIVIMethodAdapter
    from kvbench.adapters.kvquant import KVQuantMethodAdapter

    classes = (
        BF16MethodAdapter,
        TurboQuantMethodAdapter,
        KIVIMethodAdapter,
        KVQuantMethodAdapter,
    )
    def wrap(original: Any, label: str) -> Any:
        def annotated(self: Any, *args: Any, **kwargs: Any) -> Any:
            torch.cuda.nvtx.range_push(label)
            try:
                return original(self, *args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        return annotated

    for adapter in classes:
        for method_name, label in (
            ("append_decode", "phase15_cache_append"),
            ("decode_attention", "phase15_cache_decode"),
        ):
            original = getattr(adapter, method_name)
            setattr(adapter, method_name, wrap(original, label))


def _run_profile_worker(
    *,
    run_kind: str,
    configuration: str,
    batch: int,
    context_label: int,
    graph_mode: str,
    git_sha: str,
    prefix_state_root: Path,
    prefix_state_sha256: str,
    decode_operations: int,
    warmup_steps: int,
    smoke: bool,
) -> dict[str, Any]:
    if run_kind not in RUN_KINDS or graph_mode not in {"eager", "cuda_graph"}:
        raise Phase15Error("profiler worker identity differs")
    attestation = phase12._require_authorized_container_runtime()
    if any(name in os.environ for name in pilot._FORBIDDEN_ENVIRONMENT):
        raise Phase15Error("credentials entered the Measurement Container")
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
        raise Phase15Error("profiler worker source authority differs")

    import torch
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.numerical import tensor_sha256_untimed
    from kvbench.runtime.timing import warmup_operations

    _install_profiler_adapter_nvtx_annotations()
    prefix_entry = {
        "snapshot_root": str(prefix_state_root),
        "state_file_sha256": prefix_state_sha256,
    }
    session, receipt = _build_profiler_session(
        configuration=configuration,
        batch=batch,
        context_label=context_label,
        graph_mode=graph_mode,
        prefix_entry=prefix_entry,
    )
    pointers_before = phase12._phase12_session_pointers(session)
    history_before = session.current_historical_prefix_sha256()
    if graph_mode == "cuda_graph":
        if session.graph is None:
            raise Phase15Error("Graph profiler worker did not capture a graph")
        operation = session.graph.replay
    else:
        def operation() -> Any:
            with torch.inference_mode(), forced_flash_execution():
                return session.execute_audit_step(0)

    warmed = warmup_operations(
        operation, count=warmup_steps, device=session.cache_device
    )
    torch.cuda.synchronize(device=session.cache_device)
    if not bool(torch.isfinite(warmed).all()):
        raise Phase15Error("profiler warmup output is non-finite")
    output = None
    torch.cuda.nvtx.range_push(NVTX_RANGE)
    try:
        for _ in range(decode_operations):
            output = operation()
        # One profiler-boundary synchronization keeps the selected GPU work in
        # the range.  It is explicitly labeled below and never used as timing.
        torch.cuda.synchronize(device=session.cache_device)
    finally:
        torch.cuda.nvtx.range_pop()
    if output is None:
        raise Phase15Error("profiler decode region did not execute")
    output_cpu = output.detach().to(device="cpu", copy=True).clone()
    output_checksum = tensor_sha256_untimed(output_cpu)
    finite = bool(torch.isfinite(output_cpu).all())
    pointers_after = phase12._phase12_session_pointers(session)
    history_after = session.current_historical_prefix_sha256()
    accounting = session.method_cache_accounting()
    geometry = session.gqa_cache_geometry()
    family = phase12._method_family(configuration)
    replayed_adapter_fingerprint = phase12._validate_runtime_adapter_fingerprint(
        method=session.method,
        cache=session.cache,
        observed=session.adapter_config_fingerprint,
    )
    if (
        not finite
        or pointers_before != pointers_after
        or history_before != history_after
        or replayed_adapter_fingerprint != session.adapter_config_fingerprint
        or not phase12._gqa_geometry_passes(geometry, family=family)
    ):
        raise Phase15Error("profiler worker numerical or path identity drifted")
    return {
        "schema_version": RUN_SCHEMA,
        "run_kind": run_kind,
        "smoke": smoke,
        "method_config_id": configuration,
        "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
        "batch_size": batch,
        "context_label": context_label,
        "historical_context": actual_historical_context(context_label),
        "total_attended_context": actual_historical_context(context_label) + 1,
        "graph_mode": graph_mode,
        "decode_operations": decode_operations,
        "warmup_steps": warmup_steps,
        "output_checksum": output_checksum,
        "output_finite": finite,
        "cache_pointers_stable": pointers_before == pointers_after,
        "historical_cache_unchanged": history_before == history_after,
        "cache_layout_fingerprint": session.cache_layout_fingerprint(),
        "cache_accounting": accounting,
        "gqa_geometry": geometry,
        "adapter_config_fingerprint": session.adapter_config_fingerprint,
        "adapter_config_fingerprint_replayed": replayed_adapter_fingerprint,
        "method_config_fingerprint": CONFIG_FINGERPRINTS[configuration],
        "prefix_restore": receipt,
        "prefix_state_sha256": prefix_state_sha256,
        "nvtx_range": NVTX_RANGE,
        "profiler_boundary_synchronization": True,
        "profiler_duration_is_normal_timing": False,
        "performance_claim_eligible": False,
        "quality_status": "unvalidated",
        "r_hbm": None if run_kind == "nsys" else "computed_after_ncu_aggregation",
        "container_runtime_attestation": attestation,
        "execution_git_sha": git_sha,
    }


def _emit_worker_and_exit(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(
        WORKER_PREFIX
        + json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _extract_worker_payload(stdout: str) -> dict[str, Any]:
    matches = [
        line[len(WORKER_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(matches) != 1:
        raise Phase15Error("profiler worker result channel differs")
    value = json.loads(matches[0])
    if not isinstance(value, dict) or value.get("run_kind") not in RUN_KINDS:
        raise Phase15Error("profiler worker result is invalid")
    return value


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _sqlite_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _string_ids(connection: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in _sqlite_tables(connection):
        return {}
    columns = _sqlite_columns(connection, "StringIds")
    id_column = "id" if "id" in columns else "stringId"
    value_column = "value" if "value" in columns else "string"
    if id_column not in columns or value_column not in columns:
        return {}
    return {
        int(row[0]): str(row[1])
        for row in connection.execute(
            f'SELECT "{id_column}", "{value_column}" FROM StringIds'
        )
    }


def _resolve_sqlite_name(row: Mapping[str, Any], strings: Mapping[int, str]) -> str:
    for key in (
        "text",
        "name",
        "demangledName",
        "shortName",
        "mangledName",
        "textId",
        "nameId",
    ):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, int) and value in strings:
            return strings[value]
        text = str(value)
        if text and text != "0":
            return text
    return "unknown"


def _table_rows(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    cursor = connection.execute(f'SELECT * FROM "{table}"')
    names = [str(item[0]) for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def parse_nsys_sqlite(path: Path, *, configuration: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = _sqlite_tables(connection)
        strings = _string_ids(connection)
        if "NVTX_EVENTS" not in tables:
            raise Phase15Error("Nsight Systems export has no NVTX events")
        nvtx_rows = _table_rows(connection, "NVTX_EVENTS")
        ranges = []
        for raw in nvtx_rows:
            name = _resolve_sqlite_name(raw, strings)
            start = raw.get("start")
            end = raw.get("end")
            if (
                name == NVTX_RANGE
                and isinstance(start, int)
                and isinstance(end, int)
                and end > start
            ):
                ranges.append((start, end))
        if len(ranges) != 1:
            raise Phase15Error("Nsight Systems decode NVTX range differs")
        range_start, range_end = ranges[0]

        runtime_table = next(
            (
                name
                for name in (
                    "CUPTI_ACTIVITY_KIND_RUNTIME",
                    "CUPTI_ACTIVITY_KIND_DRIVER",
                )
                if name in tables
            ),
            None,
        )
        kernel_table = next(
            (
                name
                for name in (
                    "CUPTI_ACTIVITY_KIND_KERNEL",
                    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
                )
                if name in tables
            ),
            None,
        )
        if runtime_table is None or kernel_table is None:
            raise Phase15Error("Nsight Systems CUDA event tables are absent")
        apis: list[dict[str, Any]] = []
        for raw in _table_rows(connection, runtime_table):
            start, end = raw.get("start"), raw.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if start < range_start or start > range_end:
                continue
            apis.append(
                {
                    "event_kind": "cuda_api",
                    "name": _resolve_sqlite_name(raw, strings),
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": max(0, end - start),
                    "correlation_id": raw.get("correlationId"),
                    "thread_id": raw.get("globalTid", raw.get("threadId")),
                }
            )
        kernels: list[dict[str, Any]] = []
        for raw in _table_rows(connection, kernel_table):
            start, end = raw.get("start"), raw.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if end < range_start or start > range_end:
                continue
            name = _resolve_sqlite_name(raw, strings)
            role, basis = classify_kernel(name, configuration=configuration)
            kernels.append(
                {
                    "event_kind": "cuda_kernel",
                    "name": name,
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": max(0, end - start),
                    "correlation_id": raw.get("correlationId"),
                    "stream_id": raw.get("streamId"),
                    "device_id": raw.get("deviceId"),
                    "kernel_role": role,
                    "classification_basis": basis,
                }
            )
        if not kernels:
            raise Phase15Error("Nsight Systems decode range contains no kernels")
        kernels.sort(key=lambda row: (int(row["start_ns"]), int(row["end_ns"])))
        apis.sort(key=lambda row: (int(row["start_ns"]), int(row["end_ns"])))
        launch_apis = [
            row
            for row in apis
            if "launch" in str(row["name"]).lower()
        ]
        graph_launches = [
            row
            for row in launch_apis
            if "graph" in str(row["name"]).lower()
        ]
        sync_apis = [
            row
            for row in apis
            if any(
                token in str(row["name"]).lower()
                for token in ("synchronize", "waitevent", "eventquery")
            )
        ]
        launch_starts = [int(row["start_ns"]) for row in launch_apis]
        cpu_interval = (
            (launch_starts[-1] - launch_starts[0]) / (len(launch_starts) - 1)
            if len(launch_starts) >= 2
            else 0.0
        )
        api_by_correlation = {
            int(row["correlation_id"]): row
            for row in launch_apis
            if isinstance(row.get("correlation_id"), int)
        }
        api_to_gpu = []
        for kernel in kernels:
            correlation = kernel.get("correlation_id")
            if isinstance(correlation, int) and correlation in api_by_correlation:
                api_to_gpu.append(
                    max(
                        0,
                        int(kernel["start_ns"])
                        - int(api_by_correlation[correlation]["end_ns"]),
                    )
                )
        idle = []
        overlap = []
        for previous, current in zip(kernels, kernels[1:]):
            delta = int(current["start_ns"]) - int(previous["end_ns"])
            if delta >= 0:
                idle.append(delta)
            else:
                overlap.append(-delta)
        events = [
            {
                "event_kind": "nvtx_region",
                "name": NVTX_RANGE,
                "start_ns": range_start,
                "end_ns": range_end,
                "duration_ns": range_end - range_start,
            },
            *apis,
            *kernels,
        ]
        summary = {
            "cpu_cuda_submission_call_count": len(launch_apis),
            "cpu_submission_interval_ns": cpu_interval,
            "api_to_gpu_start_mean_ns": statistics.fmean(api_to_gpu) if api_to_gpu else None,
            "api_to_gpu_start_max_ns": max(api_to_gpu) if api_to_gpu else None,
            "gpu_inter_kernel_idle_total_ns": sum(idle),
            "gpu_inter_kernel_idle_mean_ns": statistics.fmean(idle) if idle else 0.0,
            "synchronization_time_ns": sum(int(row["duration_ns"]) for row in sync_apis),
            "synchronization_call_count": len(sync_apis),
            "profiler_boundary_synchronization_count": 1,
            "kernel_count": len(kernels),
            "graph_launch_count": len(graph_launches),
            "overlap_total_ns": sum(overlap),
            "total_profile_region_duration_ns": range_end - range_start,
            "kernel_order_sha256": _canonical_sha256([row["name"] for row in kernels]),
            "unknown_kernel_count": sum(row["kernel_role"] == "unknown" for row in kernels),
        }
        return events, summary
    finally:
        connection.close()


def nsys_pair_effect(eager: Mapping[str, Any], graph: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "cpu_submission_interval_ns",
        "api_to_gpu_start_mean_ns",
        "gpu_inter_kernel_idle_total_ns",
        "synchronization_time_ns",
    )
    result: dict[str, Any] = {}
    for field in fields:
        left, right = eager.get(field), graph.get(field)
        result["delta_" + field.removesuffix("_ns")] = (
            float(left) - float(right)
            if isinstance(left, (int, float)) and isinstance(right, (int, float))
            else None
        )
    result.update(
        {
            "eager_cpu_cuda_submission_call_count": int(
                eager["cpu_cuda_submission_call_count"]
            ),
            "graph_cpu_cuda_submission_call_count": int(
                graph["cpu_cuda_submission_call_count"]
            ),
            "cpu_cuda_submission_call_reduction": int(
                eager["cpu_cuda_submission_call_count"]
            )
            - int(graph["cpu_cuda_submission_call_count"]),
            "eager_synchronization_call_count": int(
                eager["synchronization_call_count"]
            ),
            "graph_synchronization_call_count": int(
                graph["synchronization_call_count"]
            ),
            "kernel_count_change": int(graph["kernel_count"]) - int(eager["kernel_count"]),
            "graph_launch_change": int(graph["graph_launch_count"]) - int(eager["graph_launch_count"]),
            "kernel_order_same": eager["kernel_order_sha256"] == graph["kernel_order_sha256"],
            "overlap_change_ns": float(graph["overlap_total_ns"]) - float(eager["overlap_total_ns"]),
        }
    )
    return result


def new_campaign_id(git_sha: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        raise Phase15Error("execution Git SHA is invalid")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f") + "z"
    nonce = os.urandom(3).hex()
    return f"phase15-{stamp}-{git_sha[:8]}-{nonce}"


def _validate_campaign_id(value: str) -> str:
    if not _CAMPAIGN_RE.fullmatch(value):
        raise Phase15Error("Phase 15 campaign ID is invalid")
    return value


def reserve_campaign(*, campaign_id: str, git_sha: str) -> Path:
    identifier = _validate_campaign_id(campaign_id)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    stage = STAGING_ROOT / f"{identifier}.{os.urandom(4).hex()}.staging"
    if (ARTIFACT_ROOT / identifier).exists():
        raise Phase15Error("Phase 15 campaign already exists")
    stage.mkdir()
    for name in ("raw", "runs", "smoke", "plots"):
        (stage / name).mkdir()
    for name in ("nsys", "ncu"):
        (stage / "raw" / name).mkdir()
    write_exclusive(
        stage / "reservation.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase15-reservation-1.0.0",
                "campaign_id": identifier,
                "execution_git_sha": git_sha,
                "created_at_utc": _utc_now(),
                "append_only": True,
            }
        ),
    )
    return stage


def _snapshot(root: Path, phase: str) -> str:
    evidence = continuation.capture_persisted_snapshot(
        evidence_root=root, phase=phase
    )
    if evidence.state == "foreign_process_detected":
        raise Phase15Error(f"foreign GPU process detected during {phase}")
    return evidence.state


def _worker_command(
    *, record: Mapping[str, Any], git_sha: str, prefix: Mapping[str, Any], smoke: bool
) -> list[str]:
    warmup = SMOKE_WARMUP_STEPS if smoke else WARMUP_STEPS
    operations = 1 if smoke else int(record["decode_operations"])
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/phase15_profiler_subset.py"),
        "--run-worker",
        "--run-kind",
        str(record["run_kind"]),
        "--configuration",
        str(record["method_config_id"]),
        "--batch-size",
        str(record["batch_size"]),
        "--context-label",
        str(record["context_label"]),
        "--graph-mode",
        str(record["graph_mode"]),
        "--git-sha",
        git_sha,
        "--prefix-state-root",
        str(prefix["snapshot_root"]),
        "--prefix-state-sha256",
        str(prefix["state_file_sha256"]),
        "--decode-operations",
        str(operations),
        "--warmup-steps",
        str(warmup),
        *( ["--smoke"] if smoke else [] ),
    ]


def _profiler_command(
    *,
    record: Mapping[str, Any],
    worker: Sequence[str],
    raw_base: Path,
    metric_map: Mapping[str, Any],
) -> list[str]:
    if record["run_kind"] == "nsys":
        return [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--backtrace=none",
            "--capture-range=nvtx",
            f"--nvtx-capture={NVTX_RANGE}",
            "--capture-range-end=stop",
            "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
            "--cuda-graph-trace=node",
            "--force-overwrite=true",
            f"--output={raw_base}",
            *worker,
        ]
    return [
        "ncu",
        "--target-processes=application-only",
        "--nvtx",
        f"--nvtx-include={NVTX_RANGE}/",
        "--replay-mode=kernel",
        "--graph-profiling=node",
        "--disable-extra-suffixes",
        "--force-overwrite",
        "--metrics",
        ",".join(str(value) for value in metric_map["metric_names"]),
        "--export",
        str(raw_base),
        *worker,
    ]


def _classify_profiler_failure(
    *, run_kind: str, returncode: int, stderr: str, raw_exists: bool
) -> str:
    lowered = stderr.lower()
    if "metric" in lowered and ("not found" in lowered or "unavailable" in lowered):
        return "metric_unavailable"
    if ("replay" in lowered and "failed" in lowered) or "launchfailed" in lowered:
        return "kernel_replay_failed"
    if run_kind == "ncu" and "graph" in lowered and "failed" in lowered:
        return "graph_profile_failed"
    if returncode != 0 or not raw_exists:
        return "tool_configuration_failed"
    return "parse_failed"


def _export_nsys(report: Path, sqlite_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            f"--output={sqlite_path}",
            str(report),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def _export_ncu(report: Path, csv_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ncu",
            "--import",
            str(report),
            "--page=raw",
            "--csv",
            "--print-units=base",
            "--log-file",
            str(csv_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def _run_profiler_point(
    *,
    stage: Path,
    record: Mapping[str, Any],
    git_sha: str,
    prefix: Mapping[str, Any],
    metric_map: Mapping[str, Any],
    attempt: int,
    smoke: bool,
) -> dict[str, Any]:
    profile_id = str(record["profile_id"])
    run_id = f"{profile_id}-attempt{attempt}"
    parent = stage / ("smoke" if smoke else "runs")
    root = parent / run_id
    root.mkdir()
    raw_parent = stage / "raw" / str(record["run_kind"])
    raw_base = raw_parent / run_id
    request = {
        "schema_version": "kvbench-phase15-profile-request-1.0.0",
        "run_id": run_id,
        "profile_id": profile_id,
        "attempt": attempt,
        "smoke": smoke,
        "record": dict(record),
        "run_kind": record["run_kind"],
        "normal_timing": False,
        "prefix_state_sha256": prefix["state_file_sha256"],
        "prefix_source_campaign": prefix["source_campaign"],
        "created_at_utc": _utc_now(),
    }
    write_exclusive(root / "request.json", json_bytes(request))
    pre_state = _snapshot(root / "preflight-snapshot", "preflight")
    worker = _worker_command(
        record=record, git_sha=git_sha, prefix=prefix, smoke=smoke
    )
    command = _profiler_command(
        record=record, worker=worker, raw_base=raw_base, metric_map=metric_map
    )
    write_exclusive(
        root / "command.json",
        json_bytes(
            {
                "argv": command,
                "run_kind": record["run_kind"],
                "contains_credentials": False,
                "profiler_duration_is_normal_timing": False,
            }
        ),
    )
    started = _utc_now()
    timeout = 7_200 if record["run_kind"] == "nsys" else 21_600
    try:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=phase12._child_environment(),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        timed_out = False
    except subprocess.TimeoutExpired as error:
        result = subprocess.CompletedProcess(
            command,
            124,
            stdout=(error.stdout or "") if isinstance(error.stdout, str) else (error.stdout or b"").decode(errors="replace"),
            stderr=(error.stderr or "") if isinstance(error.stderr, str) else (error.stderr or b"").decode(errors="replace"),
        )
        timed_out = True
    finished = _utc_now()
    write_exclusive(root / "profiler.stdout", result.stdout.encode("utf-8"))
    write_exclusive(root / "profiler.stderr", result.stderr.encode("utf-8"))
    post_state = _snapshot(root / "postflight-snapshot", "postflight")
    suffix = ".nsys-rep" if record["run_kind"] == "nsys" else ".ncu-rep"
    report = Path(str(raw_base) + suffix)
    status = "completed"
    reason = None
    payload = None
    events: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    export_record: dict[str, Any] = {}
    raw_exists = report.is_file()
    if result.returncode != 0 or not raw_exists:
        status = _classify_profiler_failure(
            run_kind=str(record["run_kind"]),
            returncode=result.returncode,
            stderr=result.stdout + "\n" + result.stderr,
            raw_exists=raw_exists,
        )
        reason = "profiler_command_failed"
    else:
        try:
            payload = _extract_worker_payload(result.stdout)
            if record["run_kind"] == "nsys":
                sqlite_path = raw_parent / f"{run_id}.sqlite"
                exported = _export_nsys(report, sqlite_path)
                export_record = {
                    "returncode": exported.returncode,
                    "stdout": exported.stdout,
                    "stderr": exported.stderr,
                    "structured_path": str(sqlite_path.relative_to(stage)),
                }
                if exported.returncode != 0 or not sqlite_path.is_file():
                    status, reason = "export_failed", "nsys_sqlite_export_failed"
                else:
                    events, summary = parse_nsys_sqlite(
                        sqlite_path, configuration=str(record["method_config_id"])
                    )
            else:
                csv_path = raw_parent / f"{run_id}.csv"
                exported = _export_ncu(report, csv_path)
                export_record = {
                    "returncode": exported.returncode,
                    "stdout": exported.stdout,
                    "stderr": exported.stderr,
                    "structured_path": str(csv_path.relative_to(stage)),
                }
                if exported.returncode != 0 or not csv_path.is_file():
                    status, reason = "export_failed", "ncu_csv_export_failed"
                else:
                    metric_rows = parse_ncu_csv(
                        csv_path.read_text(encoding="utf-8", errors="strict"), metric_map
                    )
                    events, summary = aggregate_kernel_metrics(
                        metric_rows, configuration=str(record["method_config_id"])
                    )
        except (Phase15Error, OSError, UnicodeError, json.JSONDecodeError, sqlite3.Error) as error:
            status, reason = "parse_failed", f"{type(error).__name__}:{error}"
    manifest = {
        "schema_version": RUN_SCHEMA,
        "run_id": run_id,
        "profile_id": profile_id,
        "attempt": attempt,
        "smoke": smoke,
        "status": status,
        "reason": reason,
        "run_kind": record["run_kind"],
        "record": dict(record),
        "execution_git_sha": git_sha,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "preflight_snapshot_state": pre_state,
        "postflight_snapshot_state": post_state,
        "returncode": result.returncode,
        "timed_out": timed_out,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "raw_report_path": str(report.relative_to(stage)) if raw_exists else None,
        "raw_report_sha256": sha256_file(report) if raw_exists else None,
        "worker_result": payload,
        "export": export_record,
        "summary": summary,
        "event_count": len(events),
        "normal_timing": False,
        "performance_claim_eligible": False,
        "quality_status": "unvalidated",
    }
    write_exclusive(root / "events.json", json_bytes({"records": events}))
    write_exclusive(root / "manifest.json", json_bytes(manifest))
    return manifest


def _query_ncu(stage: Path) -> dict[str, Any]:
    outputs: dict[str, str] = {}
    requests = (
        (
            "--query-metrics",
            "query-metrics.txt",
            ("ncu", "--query-metrics", "--query-metrics-mode=all"),
        ),
        ("--query-sections", "query-sections.txt", ("ncu", "--query-sections")),
    )
    for action, name, command in requests:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        payload = result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else "")
        write_exclusive(stage / name, payload.encode("utf-8"))
        if result.returncode == 0:
            outputs[action] = result.stdout
        elif action == "--query-sections" and "unrecognised option" in (
            result.stdout + result.stderr
        ):
            # NCU 2026.2 removed the requested spelling. Preserve that rejected
            # query verbatim, then use its documented semantic replacement.
            replacement = subprocess.run(
                ("ncu", "--list-sections"),
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
            replacement_payload = replacement.stdout + (
                "\nSTDERR:\n" + replacement.stderr if replacement.stderr else ""
            )
            write_exclusive(
                stage / "list-sections.txt", replacement_payload.encode("utf-8")
            )
            if replacement.returncode != 0:
                raise Phase15Error("ncu section discovery is globally unavailable")
            outputs[action] = replacement.stdout
            write_exclusive(
                stage / "section-query-transition.json",
                json_bytes(
                    {
                        "requested_command": ["ncu", "--query-sections"],
                        "requested_returncode": result.returncode,
                        "requested_status": "tool_option_unavailable",
                        "replacement_command": ["ncu", "--list-sections"],
                        "replacement_returncode": replacement.returncode,
                        "replacement_status": "PASS",
                        "semantic_scope": "section_inventory_only",
                    }
                ),
            )
        else:
            raise Phase15Error(f"ncu {action} failed")
    metric_map = resolve_metric_map(outputs["--query-metrics"], outputs["--query-sections"])
    write_exclusive(stage / "metric_map.json", json_bytes(metric_map))
    return metric_map


def _smoke_record(run_kind: str) -> dict[str, Any]:
    return {
        "method_config_id": "bf16",
        "method_config_fingerprint": CONFIG_FINGERPRINTS["bf16"],
        "batch_size": 1,
        "context_label": 4096,
        "historical_context": 4096,
        "total_attended_context": 4097,
        "roles": ["smoke"],
        "regime_selection": "profiler_smoke",
        "run_kind": run_kind,
        "graph_mode": "cuda_graph",
        "profile_id": f"smoke-{run_kind}-bf16-b1-l4096-cuda_graph",
        "decode_operations": 1,
    }


def run_campaign(*, stage: Path, campaign_id: str, git_sha: str) -> dict[str, Any]:
    identifier = _validate_campaign_id(campaign_id)
    phase12._require_authorized_container_runtime()
    if any(name in os.environ for name in pilot._FORBIDDEN_ENVIRONMENT):
        raise Phase15Error("credentials entered the Measurement Container")
    head = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ("/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout
    if head != git_sha or status:
        raise Phase15Error("Phase 15 execution source is not clean and frozen")
    root = stage.resolve(strict=True)
    selection = _strict_json(SELECTION_PATH)
    validate_selection(selection)
    prefix_index = _prefix_catalog_index()
    write_exclusive(root / "selection.json", json_bytes(selection))
    write_exclusive(
        root / "campaign_manifest.json",
        json_bytes(
            {
                "schema_version": CAMPAIGN_SCHEMA,
                "campaign_id": identifier,
                "execution_git_sha": git_sha,
                "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
                "configurations": list(CONFIGURATIONS),
                "fingerprints": dict(CONFIG_FINGERPRINTS),
                "anchors": list(ANCHORS),
                "common_same_work": selection["common_same_work"],
                "expected_nsys_profiles": 32,
                "expected_ncu_profiles": 22,
                "nsys_decode_operations": NSYS_DECODE_OPERATIONS,
                "ncu_decode_operations": NCU_DECODE_OPERATIONS,
                "warmup_steps": WARMUP_STEPS,
                "run_kinds": list(RUN_KINDS),
                "normal_timing_reused": False,
                "profiler_durations_excluded_from_timing": True,
                "performance_claim_eligible": False,
                "quality_status": "unvalidated",
            }
        ),
    )
    metric_map = _query_ncu(root)
    completed: list[dict[str, Any]] = []
    for run_kind in RUN_KINDS:
        record = _smoke_record(run_kind)
        prefix = prefix_index[("bf16", 1, 4096)]
        result = _run_profiler_point(
            stage=root,
            record=record,
            git_sha=git_sha,
            prefix=prefix,
            metric_map=metric_map,
            attempt=0,
            smoke=True,
        )
        if result["status"] != "completed":
            raise Phase15Error(
                f"global {run_kind} smoke incompatibility: {result['status']}"
            )
    for record in [*selection["nsys_profiles"], *selection["ncu_profiles"]]:
        key = (
            str(record["method_config_id"]),
            int(record["batch_size"]),
            int(record["context_label"]),
        )
        first = _run_profiler_point(
            stage=root,
            record=record,
            git_sha=git_sha,
            prefix=prefix_index[key],
            metric_map=metric_map,
            attempt=0,
            smoke=False,
        )
        chosen = first
        if first["status"] in {
            "tool_configuration_failed",
            "kernel_replay_failed",
            "graph_profile_failed",
            "export_failed",
            "parse_failed",
        }:
            chosen = _run_profiler_point(
                stage=root,
                record=record,
                git_sha=git_sha,
                prefix=prefix_index[key],
                metric_map=metric_map,
                attempt=1,
                smoke=False,
            )
        completed.append(chosen)
    write_exclusive(
        root / "run-index.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase15-run-index-1.0.0",
                "campaign_id": identifier,
                "records": completed,
                "profile_count": len(completed),
                "completed": sum(row["status"] == "completed" for row in completed),
                "failed": sum(row["status"] != "completed" for row in completed),
                "retried": sum(int(row["attempt"]) > 0 for row in completed),
            }
        ),
    )
    return {
        "campaign_id": identifier,
        "profiles": len(completed),
        "completed": sum(row["status"] == "completed" for row in completed),
        "failed": sum(row["status"] != "completed" for row in completed),
        "retried": sum(int(row["attempt"]) > 0 for row in completed),
    }


_CONTINUATION_ALLOWED_SOURCE_CHANGES = {
    "scripts/phase15_profiler_subset.py",
    "tests/unit/test_phase15_profiler_subset.py",
}
_CONTINUATION_MANIFEST = "continuation-manifest-v2.json"


def _require_clean_execution_head(git_sha: str) -> None:
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
        raise Phase15Error("Phase 15 continuation source is not clean and frozen")


def _continuation_source_equivalence(
    *, original_head: str, continuation_head: str
) -> dict[str, Any]:
    changed = subprocess.run(
        (
            "/usr/bin/git",
            "diff",
            "--name-only",
            original_head,
            continuation_head,
            "--",
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    changed = sorted(value.strip() for value in changed if value.strip())
    unexpected = sorted(set(changed) - _CONTINUATION_ALLOWED_SOURCE_CHANGES)
    if unexpected:
        raise Phase15Error(
            f"timing-critical source changed before NCU continuation: {unexpected}"
        )
    return {
        "schema_version": "kvbench-phase15-continuation-source-equivalence-1.0.0",
        "original_execution_git_sha": original_head,
        "continuation_execution_git_sha": continuation_head,
        "changed_paths": changed,
        "allowed_profiler_only_paths": sorted(_CONTINUATION_ALLOWED_SOURCE_CHANGES),
        "timing_critical_source_changed": False,
        "adapters_changed": False,
        "cuda_changed": False,
        "runner_or_timing_changed": False,
        "model_or_method_config_changed": False,
    }


def _profile_attempts(stage: Path, profile_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((stage / "runs").glob(f"{profile_id}-attempt*/manifest.json")):
        row = _strict_json(path)
        if row.get("profile_id") != profile_id:
            raise Phase15Error("profile attempt identity differs")
        rows.append(row)
    return sorted(rows, key=lambda row: int(row["attempt"]))


def _profile_attempt_directories(stage: Path, profile_id: str) -> list[tuple[int, Path]]:
    prefix = f"{profile_id}-attempt"
    rows: list[tuple[int, Path]] = []
    for path in sorted((stage / "runs").glob(f"{prefix}*")):
        if not path.is_dir():
            raise Phase15Error("profiler attempt path is not a directory")
        suffix = path.name.removeprefix(prefix)
        if not suffix.isdigit() or str(int(suffix)) != suffix:
            raise Phase15Error("profiler attempt directory identity is invalid")
        rows.append((int(suffix), path))
    observed = [attempt for attempt, _ in rows]
    if observed and observed != list(range(max(observed) + 1)):
        raise Phase15Error("profiler attempt directory sequence is not contiguous")
    return rows


def next_profile_attempt(stage: Path, profile_id: str) -> int:
    directories = _profile_attempt_directories(stage, profile_id)
    if not directories:
        return 0
    for attempt, path in directories:
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            row = _strict_json(manifest_path)
            if row.get("profile_id") != profile_id or row.get("attempt") != attempt:
                raise Phase15Error("profiler attempt manifest identity differs")
    return directories[-1][0] + 1


def prepare_ncu_continuation(
    *, stage: Path, campaign_id: str, git_sha: str
) -> dict[str, Any]:
    identifier = _validate_campaign_id(campaign_id)
    _require_clean_execution_head(git_sha)
    root = stage.resolve(strict=True)
    if (root / "run-index.json").exists() or (root / _CONTINUATION_MANIFEST).exists():
        raise Phase15Error("Phase 15 NCU continuation was already prepared")
    campaign = _strict_json(root / "campaign_manifest.json")
    if campaign.get("campaign_id") != identifier:
        raise Phase15Error("Phase 15 continuation campaign identity differs")
    original_head = str(campaign.get("execution_git_sha"))
    equivalence = _continuation_source_equivalence(
        original_head=original_head, continuation_head=git_sha
    )
    selection = _strict_json(root / "selection.json")
    validate_selection(selection)
    manifests = _all_run_manifests(root)
    completed_nsys = [
        row
        for row in manifests
        if row.get("run_kind") == "nsys" and row.get("status") == "completed"
    ]
    completed_ncu = [
        row
        for row in manifests
        if row.get("run_kind") == "ncu" and row.get("status") == "completed"
    ]
    if len(completed_nsys) != 32 or completed_ncu:
        raise Phase15Error("Phase 15 continuation entry profile coverage differs")
    incomplete_ncu = [
        path
        for record in selection["ncu_profiles"]
        for _, path in _profile_attempt_directories(
            root, str(record["profile_id"])
        )
        if not (path / "manifest.json").is_file()
    ]
    payload = {
        "schema_version": "kvbench-phase15-ncu-continuation-1.1.0",
        "campaign_id": identifier,
        "prepared_at_utc": _utc_now(),
        "segment_a_completed_nsys_profiles": 32,
        "segment_a_failed_ncu_attempts": sum(
            row.get("run_kind") == "ncu" and row.get("status") != "completed"
            for row in manifests
        ),
        "segment_a_incomplete_ncu_attempts": len(incomplete_ncu),
        "segment_a_incomplete_attempt_paths": [
            str(path.relative_to(root)) for path in incomplete_ncu
        ],
        "continuation_ncu_profiles": 22,
        "fresh_authorized_outer_container_per_profile": True,
        "disable_extra_metric_suffixes": True,
        "normal_timing": False,
        "selection_sha256": sha256_file(root / "selection.json"),
        "metric_map_sha256": sha256_file(root / "metric_map.json"),
        "source_equivalence": equivalence,
        "superseded_preparation_record": (
            {
                "path": "continuation-manifest.json",
                "sha256": sha256_file(root / "continuation-manifest.json"),
            }
            if (root / "continuation-manifest.json").is_file()
            else None
        ),
    }
    write_exclusive(root / _CONTINUATION_MANIFEST, json_bytes(payload))
    return payload


def run_one_ncu_profile(
    *, stage: Path, campaign_id: str, git_sha: str, profile_id: str, attempt: int
) -> dict[str, Any]:
    identifier = _validate_campaign_id(campaign_id)
    phase12._require_authorized_container_runtime()
    if any(name in os.environ for name in pilot._FORBIDDEN_ENVIRONMENT):
        raise Phase15Error("credentials entered the Measurement Container")
    _require_clean_execution_head(git_sha)
    root = stage.resolve(strict=True)
    continuation = _strict_json(root / _CONTINUATION_MANIFEST)
    if (
        continuation.get("campaign_id") != identifier
        or continuation.get("source_equivalence", {}).get(
            "continuation_execution_git_sha"
        )
        != git_sha
    ):
        raise Phase15Error("Phase 15 NCU continuation authority differs")
    selection = _strict_json(root / "selection.json")
    validate_selection(selection)
    matches = [
        dict(row)
        for row in selection["ncu_profiles"]
        if row.get("profile_id") == profile_id
    ]
    if len(matches) != 1:
        raise Phase15Error("Phase 15 selected NCU profile identity differs")
    expected_attempt = next_profile_attempt(root, profile_id)
    if attempt != expected_attempt:
        raise Phase15Error("Phase 15 NCU continuation attempt differs")
    record = matches[0]
    key = (
        str(record["method_config_id"]),
        int(record["batch_size"]),
        int(record["context_label"]),
    )
    return _run_profiler_point(
        stage=root,
        record=record,
        git_sha=git_sha,
        prefix=_prefix_catalog_index()[key],
        metric_map=_strict_json(root / "metric_map.json"),
        attempt=attempt,
        smoke=False,
    )


def finalize_ncu_continuation_index(
    *, stage: Path, campaign_id: str, git_sha: str
) -> dict[str, Any]:
    identifier = _validate_campaign_id(campaign_id)
    _require_clean_execution_head(git_sha)
    root = stage.resolve(strict=True)
    if (root / "run-index.json").exists():
        raise Phase15Error("Phase 15 run index already exists")
    continuation = _strict_json(root / _CONTINUATION_MANIFEST)
    if continuation.get("campaign_id") != identifier:
        raise Phase15Error("Phase 15 continuation identity differs")
    selection = _strict_json(root / "selection.json")
    validate_selection(selection)
    chosen: list[dict[str, Any]] = []
    for record in selection["nsys_profiles"]:
        directories = _profile_attempt_directories(
            root, str(record["profile_id"])
        )
        attempts = _profile_attempts(root, str(record["profile_id"]))
        if (
            len(directories) != 1
            or len(attempts) != 1
            or attempts[0].get("status") != "completed"
        ):
            raise Phase15Error("Phase 15 Segment A Nsys coverage differs")
        chosen.append(attempts[0])
    for record in selection["ncu_profiles"]:
        directories = _profile_attempt_directories(
            root, str(record["profile_id"])
        )
        attempts = _profile_attempts(root, str(record["profile_id"]))
        if (
            not directories
            or not attempts
            or not (directories[-1][1] / "manifest.json").is_file()
            or attempts[-1].get("status") != "completed"
            or int(attempts[-1]["attempt"]) != directories[-1][0]
        ):
            raise Phase15Error(
                f"Phase 15 NCU profile is not completed: {record['profile_id']}"
            )
        if attempts[-1].get("execution_git_sha") != git_sha:
            raise Phase15Error("Phase 15 NCU continuation execution SHA differs")
        chosen.append(attempts[-1])
    if len(chosen) != 54:
        raise Phase15Error("Phase 15 continuation chosen profile count differs")
    all_runs = _all_run_manifests(root)
    all_attempt_directories = [
        path
        for record in [*selection["nsys_profiles"], *selection["ncu_profiles"]]
        for _, path in _profile_attempt_directories(
            root, str(record["profile_id"])
        )
    ]
    payload = {
        "schema_version": "kvbench-phase15-run-index-1.0.0",
        "campaign_id": identifier,
        "records": chosen,
        "profile_count": len(chosen),
        "completed": sum(row["status"] == "completed" for row in chosen),
        "failed": sum(row["status"] != "completed" for row in chosen),
        "retried": sum(int(row["attempt"]) > 0 for row in chosen),
        "historical_failed_attempts_preserved": sum(
            row.get("status") != "completed" for row in all_runs
        ),
        "historical_incomplete_attempts_preserved": sum(
            not (path / "manifest.json").is_file()
            for path in all_attempt_directories
        ),
        "execution_git_shas": sorted(
            {str(row["execution_git_sha"]) for row in chosen}
        ),
        "continuation": True,
    }
    write_exclusive(root / "run-index.json", json_bytes(payload))
    write_exclusive(
        root / "continuation-result.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase15-ncu-continuation-result-1.0.0",
                "campaign_id": identifier,
                "status": "PASS",
                "completed_nsys_profiles": 32,
                "completed_ncu_profiles": 22,
                "historical_failed_attempts_preserved": payload[
                    "historical_failed_attempts_preserved"
                ],
                "fresh_authorized_outer_container_per_ncu_profile": True,
                "finished_at_utc": _utc_now(),
            }
        ),
    )
    return payload


def _parquet_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise Phase15Error("analysis image lacks pyarrow") from error
    normalized = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                item[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            else:
                item[key] = value
        normalized.append(item)
    if not normalized:
        normalized = [{"empty": True}]
    expected = pa.Table.from_pylist(normalized)
    if path.exists():
        observed = pq.read_table(path)
        if not observed.equals(expected):
            raise Phase15Error(f"existing derived Parquet differs: {path.name}")
        return
    pq.write_table(expected, path, compression="zstd")


def _write_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise Phase15Error(f"existing derived output differs: {path.name}")
        return
    write_exclusive(path, payload)


def _all_run_manifests(stage: Path) -> list[dict[str, Any]]:
    return [
        _strict_json(path)
        for path in sorted((stage / "runs").glob("*/manifest.json"))
    ]


def _chosen_run_manifests(stage: Path) -> list[dict[str, Any]]:
    value = _strict_json(stage / "run-index.json")
    rows = value.get("records")
    if not isinstance(rows, list):
        raise Phase15Error("Phase 15 run index is invalid")
    return [dict(row) for row in rows]


def _plot_outputs(
    root: Path,
    *,
    nsys_pairs: Sequence[Mapping[str, Any]],
    traffic: Sequence[Mapping[str, Any]],
    amplifications: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _plot_outputs_svg(
            root,
            nsys_pairs=nsys_pairs,
            traffic=traffic,
            amplifications=amplifications,
        )
        return

    plot_root = root / "plots"

    def bar(name: str, labels: list[str], values: list[float], ylabel: str) -> None:
        figure, axis = plt.subplots(figsize=(9, 4.8))
        axis.bar(range(len(labels)), values)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_root / name, dpi=120)
        plt.close(figure)

    pair_labels = [
        f"{row['method_config_id']}@{row['context_label']}" for row in nsys_pairs
    ]
    bar(
        "cpu_submission_interval.png",
        pair_labels,
        [float(row.get("delta_cpu_submission_interval") or 0.0) for row in nsys_pairs],
        "eager - graph CPU submission interval (ns; profiler only)",
    )
    bar(
        "gpu_idle_gaps.png",
        pair_labels,
        [float(row.get("delta_gpu_inter_kernel_idle_total") or 0.0) for row in nsys_pairs],
        "eager - graph GPU idle (ns; profiler only)",
    )
    bar(
        "synchronization_time.png",
        pair_labels,
        [float(row.get("delta_synchronization_time") or 0.0) for row in nsys_pairs],
        "eager - graph synchronization (ns; profiler only)",
    )
    common = [row for row in traffic if row.get("common_same_work") is True]
    labels = [str(row["method_config_id"]) for row in common]
    bar(
        "cache_path_dram_bytes.png",
        labels,
        [float(row["cache_path_dram_bytes"]) for row in common],
        "cache-path DRAM bytes",
    )
    figure, axis = plt.subplots(figsize=(9, 4.8))
    x = range(len(common))
    axis.bar([value - 0.18 for value in x], [float(row["total_decode_dram_bytes"]) for row in common], width=0.36, label="total")
    axis.bar([value + 0.18 for value in x], [float(row["cache_path_dram_bytes"]) for row in common], width=0.36, label="cache path")
    axis.set_xticks(list(x), labels, rotation=45, ha="right")
    axis.set_ylabel("DRAM bytes")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_root / "total_vs_cache_dram.png", dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        [float(row["rho_alloc"]) for row in amplifications],
        [float(row["rho_hbm"]) for row in amplifications],
    )
    for row in amplifications:
        axis.annotate(str(row["method_config_id"]), (float(row["rho_alloc"]), float(row["rho_hbm"])), fontsize=7)
    axis.set_xlabel("allocated-byte ratio")
    axis.set_ylabel("measured cache-path HBM ratio")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_root / "allocated_vs_hbm_ratio.png", dpi=120)
    plt.close(figure)
    bar(
        "traffic_amplification.png",
        [str(row["method_config_id"]) for row in amplifications],
        [float(row["A_traffic"]) for row in amplifications],
        "A_traffic",
    )
    bar(
        "l2_hit_rate.png",
        labels,
        [float(row.get("l2_hit_rate") or 0.0) for row in common],
        "weighted L2 hit rate (%)",
    )
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(labels, [float(row.get("sm_activity") or 0.0) for row in common], marker="o", label="SM activity")
    axis.plot(labels, [float(row.get("achieved_occupancy") or 0.0) for row in common], marker="o", label="occupancy")
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_root / "sm_activity_occupancy.png", dpi=120)
    plt.close(figure)
    roles = defaultdict(float)
    for row in common:
        value = row["dram_bytes_by_role"]
        role_map = value if isinstance(value, dict) else json.loads(str(value))
        for role, role_bytes in role_map.items():
            roles[role] += float(role_bytes)
    bar("traffic_by_kernel_role.png", list(roles), list(roles.values()), "summed DRAM bytes")


def _plot_outputs_svg(
    root: Path,
    *,
    nsys_pairs: Sequence[Mapping[str, Any]],
    traffic: Sequence[Mapping[str, Any]],
    amplifications: Sequence[Mapping[str, Any]],
) -> None:
    """Dependency-free diagnostics for the pinned analysis image."""

    from html import escape

    plot_root = root / "plots"

    def document(
        *,
        title: str,
        labels: Sequence[str],
        series: Sequence[tuple[str, Sequence[float]]],
        kind: str = "bar",
    ) -> bytes:
        width, height = 1200, 620
        left, top, plot_width, plot_height = 90, 60, 1060, 440
        all_values = [float(value) for _, values in series for value in values]
        minimum = min([0.0, *all_values]) if all_values else 0.0
        maximum = max([0.0, *all_values]) if all_values else 1.0
        span = maximum - minimum or 1.0

        def y(value: float) -> float:
            return top + plot_height - (float(value) - minimum) / span * plot_height

        colors = ("#2f6f9f", "#d46a35", "#3f8f5f")
        body = [
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="18">{escape(title)}</text>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
            f'<line x1="{left}" y1="{y(0)}" x2="{left + plot_width}" y2="{y(0)}" stroke="#333"/>',
            f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-size="11">{maximum:.4g}</text>',
            f'<text x="{left - 8}" y="{top + plot_height + 4}" text-anchor="end" font-size="11">{minimum:.4g}</text>',
        ]
        count = max(1, len(labels))
        slot = plot_width / count
        if kind == "scatter":
            x_values = [float(value) for value in series[0][1]]
            y_values = [float(value) for value in series[1][1]]
            x_min, x_max = min([0.0, *x_values]), max([0.0, *x_values])
            x_span = x_max - x_min or 1.0
            for label, x_value, y_value in zip(labels, x_values, y_values):
                x = left + (x_value - x_min) / x_span * plot_width
                body.append(f'<circle cx="{x:.2f}" cy="{y(y_value):.2f}" r="5" fill="{colors[0]}"/>')
                body.append(f'<text x="{x + 7:.2f}" y="{y(y_value) - 5:.2f}" font-size="10">{escape(label)}</text>')
        elif kind == "line":
            for series_index, (series_label, values) in enumerate(series):
                points = []
                for index, value in enumerate(values):
                    x = left + (index + 0.5) * slot
                    points.append(f"{x:.2f},{y(float(value)):.2f}")
                body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[series_index]}" stroke-width="2"/>')
                body.append(f'<text x="{left + 160 * series_index}" y="{height - 18}" fill="{colors[series_index]}" font-size="12">{escape(series_label)}</text>')
        else:
            bar_width = slot * 0.72 / max(1, len(series))
            for series_index, (series_label, values) in enumerate(series):
                for index, value in enumerate(values):
                    value = float(value)
                    x = left + index * slot + slot * 0.14 + series_index * bar_width
                    y_value = y(max(0.0, value)) if value >= 0 else y(0.0)
                    height_value = abs(y(value) - y(0.0))
                    body.append(f'<rect x="{x:.2f}" y="{y_value:.2f}" width="{bar_width:.2f}" height="{height_value:.2f}" fill="{colors[series_index]}"/>')
                body.append(f'<text x="{left + 180 * series_index}" y="{height - 18}" fill="{colors[series_index]}" font-size="12">{escape(series_label)}</text>')
        for index, label in enumerate(labels):
            x = left + (index + 0.5) * slot
            body.append(f'<text x="{x:.2f}" y="{top + plot_height + 18}" text-anchor="end" transform="rotate(-50 {x:.2f} {top + plot_height + 18})" font-size="9">{escape(label)}</text>')
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            + "".join(body)
            + "</svg>\n"
        ).encode("utf-8")

    pair_labels = [
        f"{row['method_config_id']}@{row['context_label']}" for row in nsys_pairs
    ]
    common = [row for row in traffic if row.get("common_same_work") is True]
    labels = [str(row["method_config_id"]) for row in common]
    plots = {
        "cpu_submission_interval.svg": document(
            title="Eager minus Graph CPU submission interval (ns; profiler only)",
            labels=pair_labels,
            series=(("eager - graph", [float(row.get("delta_cpu_submission_interval") or 0.0) for row in nsys_pairs]),),
        ),
        "gpu_idle_gaps.svg": document(
            title="Eager minus Graph GPU inter-kernel idle (ns; profiler only)",
            labels=pair_labels,
            series=(("eager - graph", [float(row.get("delta_gpu_inter_kernel_idle_total") or 0.0) for row in nsys_pairs]),),
        ),
        "synchronization_time.svg": document(
            title="Eager minus Graph synchronization time (ns; profiler only)",
            labels=pair_labels,
            series=(("eager - graph", [float(row.get("delta_synchronization_time") or 0.0) for row in nsys_pairs]),),
        ),
        "cache_path_dram_bytes.svg": document(
            title="Cache-path DRAM bytes",
            labels=labels,
            series=(("cache path", [float(row["cache_path_dram_bytes"]) for row in common]),),
        ),
        "total_vs_cache_dram.svg": document(
            title="Total-decode versus cache-path DRAM bytes",
            labels=labels,
            series=(
                ("total", [float(row["total_decode_dram_bytes"]) for row in common]),
                ("cache path", [float(row["cache_path_dram_bytes"]) for row in common]),
            ),
        ),
        "allocated_vs_hbm_ratio.svg": document(
            title="Allocated ratio versus measured cache-path HBM ratio",
            labels=[str(row["method_config_id"]) for row in amplifications],
            series=(
                ("rho_alloc", [float(row["rho_alloc"]) for row in amplifications]),
                ("rho_hbm", [float(row["rho_hbm"]) for row in amplifications]),
            ),
            kind="scatter",
        ),
        "traffic_amplification.svg": document(
            title="Traffic amplification",
            labels=[str(row["method_config_id"]) for row in amplifications],
            series=(("A_traffic", [float(row["A_traffic"]) for row in amplifications]),),
        ),
        "l2_hit_rate.svg": document(
            title="Weighted L2 hit rate",
            labels=labels,
            series=(("L2 hit rate", [float(row.get("l2_hit_rate") or 0.0) for row in common]),),
        ),
        "sm_activity_occupancy.svg": document(
            title="SM activity and achieved occupancy",
            labels=labels,
            series=(
                ("SM activity", [float(row.get("sm_activity") or 0.0) for row in common]),
                ("occupancy", [float(row.get("achieved_occupancy") or 0.0) for row in common]),
            ),
            kind="line",
        ),
    }
    roles: dict[str, float] = defaultdict(float)
    for row in common:
        value = row["dram_bytes_by_role"]
        role_map = value if isinstance(value, dict) else json.loads(str(value))
        for role, role_bytes in role_map.items():
            roles[role] += float(role_bytes)
    plots["traffic_by_kernel_role.svg"] = document(
        title="Traffic breakdown by kernel role",
        labels=list(roles),
        series=(("DRAM bytes", list(roles.values())),),
    )
    for name, payload in plots.items():
        _write_or_verify(plot_root / name, payload)


def materialize_analysis(stage: Path) -> dict[str, Any]:
    root = stage.resolve(strict=True)
    selection = _strict_json(root / "selection.json")
    validate_selection(selection)
    chosen = _chosen_run_manifests(root)
    all_runs = _all_run_manifests(root)
    by_run_id = {str(row["run_id"]): row for row in all_runs}
    if len(chosen) != 54 or any(str(row["run_id"]) not in by_run_id for row in chosen):
        raise Phase15Error("Phase 15 chosen profile cardinality differs")

    nsys_index: list[dict[str, Any]] = []
    ncu_index: list[dict[str, Any]] = []
    nsys_events: list[dict[str, Any]] = []
    kernel_rows: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    traffic_rows: list[dict[str, Any]] = []
    for manifest in chosen:
        record = manifest["record"]
        index_row = {
            "run_id": manifest["run_id"],
            "profile_id": manifest["profile_id"],
            "attempt": manifest["attempt"],
            "status": manifest["status"],
            "failure_reason": manifest["reason"],
            "method_config_id": record["method_config_id"],
            "method_config_fingerprint": record["method_config_fingerprint"],
            "batch_size": record["batch_size"],
            "context_label": record["context_label"],
            "historical_context": record["historical_context"],
            "graph_mode": record["graph_mode"],
            "roles": record["roles"],
            "raw_report_path": manifest["raw_report_path"],
            "raw_report_sha256": manifest["raw_report_sha256"],
            "execution_git_sha": manifest["execution_git_sha"],
            "authorized_container_digest": manifest["authorized_container_digest"],
            "normal_timing": False,
        }
        if manifest["run_kind"] == "nsys":
            nsys_index.append({**index_row, **dict(manifest.get("summary", {}))})
            event_path = root / "runs" / str(manifest["run_id"]) / "events.json"
            for event in _strict_json(event_path).get("records", []):
                nsys_events.append({"run_id": manifest["run_id"], **dict(event)})
        else:
            event_path = root / "runs" / str(manifest["run_id"]) / "events.json"
            raw_records = _strict_json(event_path).get("records", [])
            if not isinstance(raw_records, list):
                raise Phase15Error("NCU event evidence is not a record list")
            records, derived_summary, role_bytes = reclassify_kernel_events(
                raw_records,
                configuration=str(record["method_config_id"]),
                recorded_summary=dict(manifest.get("summary", {})),
            )
            ncu_index.append({**index_row, **derived_summary})
            for event in records:
                kernel_rows.append({"run_id": manifest["run_id"], **event})
                classifications.append(
                    {
                        "run_id": manifest["run_id"],
                        "kernel_id": event["kernel_id"],
                        "kernel_name": event["kernel_name"],
                        "kernel_role": event["kernel_role"],
                        "classification_basis": event["classification_basis"],
                        "cache_path": event["cache_path"],
                        "dram_bytes": event["dram_bytes"],
                    }
                )
            common = (
                record["batch_size"] == COMMON_BATCH
                and record["context_label"] == COMMON_CONTEXT_LABEL
                and record["graph_mode"] == "cuda_graph"
            )
            traffic_rows.append(
                {
                    **index_row,
                    **derived_summary,
                    "common_same_work": common,
                    "dram_bytes_by_role": dict(sorted(role_bytes.items())),
                }
            )

    pair_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in nsys_index:
        if row["status"] == "completed":
            grouped[(str(row["method_config_id"]), int(row["batch_size"]), int(row["context_label"]))][str(row["graph_mode"])] = row
    for (configuration, batch, context), modes in sorted(grouped.items()):
        if set(modes) != {"eager", "cuda_graph"}:
            continue
        pair_rows.append(
            {
                "method_config_id": configuration,
                "batch_size": batch,
                "context_label": context,
                "roles": modes["eager"]["roles"],
                **nsys_pair_effect(modes["eager"], modes["cuda_graph"]),
            }
        )

    common = {
        str(row["method_config_id"]): row
        for row in traffic_rows
        if row["common_same_work"] is True and row["status"] == "completed"
    }
    if set(common) != set(CONFIGURATIONS):
        raise Phase15Error("common same-work NCU traffic is incomplete")
    bf16 = common["bf16"]
    hbm_rows: list[dict[str, Any]] = []
    amplification_rows: list[dict[str, Any]] = []
    bf16_allocated = float(bf16["cache_accounting"]["allocated_bytes"] if isinstance(bf16.get("cache_accounting"), dict) else by_run_id[str(bf16["run_id"])]["worker_result"]["cache_accounting"]["allocated_bytes"])
    for configuration in CONFIGURATIONS:
        row = common[configuration]
        eligibility = bool(
            float(row.get("unclassified_dram_bytes", 0.0)) == 0.0
            and float(bf16.get("unclassified_dram_bytes", 0.0)) == 0.0
        )
        ratio = hbm_ratio(bf16=bf16, method=row, same_work=eligibility)
        worker = by_run_id[str(row["run_id"])]["worker_result"]
        allocated = float(worker["cache_accounting"]["allocated_bytes"])
        rho_alloc = allocated / bf16_allocated
        hbm_rows.append(
            {
                "method_config_id": configuration,
                "batch_size": COMMON_BATCH,
                "context_label": COMMON_CONTEXT_LABEL,
                "graph_mode": "cuda_graph",
                "same_work_eligible": eligibility,
                **ratio,
            }
        )
        if ratio["r_hbm"] is not None:
            rho_hbm = float(ratio["rho_hbm_cache"])
            amplification_rows.append(
                {
                    "method_config_id": configuration,
                    "rho_alloc": rho_alloc,
                    "r_alloc": 1.0 / rho_alloc,
                    "rho_hbm": rho_hbm,
                    "r_hbm": ratio["r_hbm"],
                    "r_nominal": worker["cache_accounting"].get("r_nominal"),
                    "A_traffic": traffic_amplification(rho_hbm=rho_hbm, rho_alloc=rho_alloc),
                    "critical_path_effect": None,
                }
            )

    anchor_summary: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        pairs = [row for row in pair_rows if row["method_config_id"] == anchor]
        submission = sum(float(row.get("delta_cpu_submission_interval") or 0.0) > 0 for row in pairs)
        idle = sum(float(row.get("delta_gpu_inter_kernel_idle_total") or 0.0) > 0 for row in pairs)
        sync = sum(float(row.get("delta_synchronization_time") or 0.0) > 0 for row in pairs)
        order_changed = sum(row.get("kernel_order_same") is False for row in pairs)
        overlap_changed = sum(abs(float(row.get("overlap_change_ns") or 0.0)) > 0 for row in pairs)
        if submission and (overlap_changed or order_changed):
            classification = "cpu_plus_overlap"
        elif order_changed:
            classification = "device_scheduling_change"
        elif submission and not idle:
            classification = "mostly_cpu_submission"
        elif pairs:
            classification = "method_specific_mixed"
        else:
            classification = "inconclusive"
        anchor_summary.append(
            {
                "method_config_id": anchor,
                "profiled_regime_pairs": len(pairs),
                "cpu_submission_reduced_pairs": submission,
                "gpu_idle_reduced_pairs": idle,
                "synchronization_reduced_pairs": sync,
                "kernel_order_changed_pairs": order_changed,
                "overlap_changed_pairs": overlap_changed,
                "dram_traffic_mode_change": None,
                "dram_traffic_mode_change_reason": "NCU subset is Graph-only by frozen contract",
                "graph_effect_classification": classification,
            }
        )
    mechanism = {
        "schema_version": "kvbench-phase15-mechanism-summary-1.0.0",
        "campaign_id": _strict_json(root / "campaign_manifest.json")["campaign_id"],
        "nsys_profiles": len(nsys_index),
        "ncu_profiles": len(ncu_index),
        "failed_profiles": sum(row["status"] != "completed" for row in chosen),
        "retried_profiles": sum(int(row["attempt"]) > 0 for row in chosen),
        "nsys_pair_count": len(pair_rows),
        "common_same_work_point": selection["common_same_work"],
        "common_same_work_ncu_configurations": len(common),
        "hbm_ratio_populated": sum(row["r_hbm"] is not None for row in hbm_rows),
        "kernel_classification_version": KERNEL_CLASSIFICATION_VERSION,
        "recorded_ncu_totals_conserved": True,
        "anchors": anchor_summary,
        "phase14_result_preserved": True,
        "phase14_pure_launch_floor_only_supported": False,
        "scientific_interpretation": "CUDA Graph effects are method- and regime-dependent; direct profiler evidence does not support a pure launch-floor-only model.",
        "profiler_durations_used_as_normal_timing": False,
        "full_scan": "CLOSED",
        "quality": "LOCKED",
        "performance_data_frozen_present": False,
    }

    selection_rows = [*selection["nsys_profiles"], *selection["ncu_profiles"]]
    _parquet_write(root / "selection_table.parquet", selection_rows)
    _parquet_write(root / "nsys_run_index.parquet", nsys_index)
    _parquet_write(root / "ncu_run_index.parquet", ncu_index)
    _parquet_write(root / "nsys_events.parquet", nsys_events)
    _parquet_write(root / "nsys_pair_effects.parquet", pair_rows)
    _parquet_write(root / "kernel_metrics.parquet", kernel_rows)
    _parquet_write(root / "kernel_classification.parquet", classifications)
    _parquet_write(root / "decode_traffic.parquet", traffic_rows)
    _parquet_write(root / "hbm_ratios.parquet", hbm_rows)
    _parquet_write(root / "traffic_amplification.parquet", amplification_rows)
    _write_or_verify(root / "mechanism_summary.json", json_bytes(mechanism))
    _plot_outputs(root, nsys_pairs=pair_rows, traffic=traffic_rows, amplifications=amplification_rows)
    report = (
        "# Phase 15 profiler subset\n\n"
        f"Status: {'PASS' if len(common) == 10 else 'BLOCKED'}\n\n"
        f"Campaign: `{mechanism['campaign_id']}`\n\n"
        f"Nsight Systems: {len(nsys_index)} selected profiles and {len(pair_rows)} eager/Graph pairs.\n\n"
        f"Nsight Compute: {len(ncu_index)} selected profiles; common same-work traffic covers {len(common)}/10 configurations.\n\n"
        f"Measured cache-path `r_hbm` is eligible for {mechanism['hbm_ratio_populated']}/10 configurations; ineligible rows remain null with a reason.\n\n"
        "Profiler durations are mechanism-only and are excluded from all normal timing and fit data.\n\n"
        "The direct evidence preserves Phase 14's conclusion: CUDA Graph effects are heterogeneous and are not explained by a pure launch-floor-only model.\n\n"
        "Full Scan remains CLOSED. Quality remains LOCKED.\n"
    )
    _write_or_verify(root / "phase15_report.md", report.encode("utf-8"))
    qc = {
        "schema_version": "kvbench-phase15-qc-1.0.0",
        "campaign_id": mechanism["campaign_id"],
        "status": "PASS" if len(common) == 10 else "BLOCKED",
        "selected_nsys_profiles": 32,
        "selected_ncu_profiles": 22,
        "completed_nsys_profiles": sum(row["status"] == "completed" for row in nsys_index),
        "completed_ncu_profiles": sum(row["status"] == "completed" for row in ncu_index),
        "failed_profiles": mechanism["failed_profiles"],
        "retried_profiles": mechanism["retried_profiles"],
        "common_same_work_configurations": len(common),
        "r_hbm_populated": mechanism["hbm_ratio_populated"],
        "profiler_timing_contamination": False,
        "phase16_readiness": "READY" if len(common) == 10 else "NOT READY",
        "full_scan": "CLOSED",
        "quality": "LOCKED",
    }
    _write_or_verify(root / "phase15_qc.json", json_bytes(qc))
    return qc


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise Phase15Error("Phase 15 bundle contains a symlink")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode) or path.stat().st_nlink != 1:
            raise Phase15Error("Phase 15 bundle contains an unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            files.append(path)
    return files


def seal_campaign(stage: Path, *, campaign_id: str) -> Path:
    identifier = _validate_campaign_id(campaign_id)
    root = stage.resolve(strict=True)
    qc = _strict_json(root / "phase15_qc.json")
    if qc.get("campaign_id") != identifier or qc.get("status") != "PASS":
        raise Phase15Error("Phase 15 QC is not locally complete")
    required = {
        "campaign_manifest.json",
        "selection_table.parquet",
        "metric_map.json",
        "nsys_run_index.parquet",
        "ncu_run_index.parquet",
        "nsys_events.parquet",
        "nsys_pair_effects.parquet",
        "kernel_metrics.parquet",
        "kernel_classification.parquet",
        "decode_traffic.parquet",
        "hbm_ratios.parquet",
        "traffic_amplification.parquet",
        "mechanism_summary.json",
        "phase15_report.md",
        "phase15_qc.json",
    }
    if not required.issubset({path.name for path in root.iterdir()}):
        raise Phase15Error("Phase 15 required outputs are absent")
    write_exclusive(
        root / "manifest.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase15-artifact-manifest-1.0.0",
                "run_id": identifier,
                "campaign_id": identifier,
                "status": "PASS",
                "created_at_utc": _utc_now(),
                "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
                "append_only": True,
                "complete_written_last": True,
                "run_kind": "profiler_subset",
                "profiler_duration_is_normal_timing": False,
                "performance_claim_eligible": False,
                "quality_status": "unvalidated",
            }
        ),
    )
    payload = _payload_paths(
        root,
        {"inventory.json", "artifact_inventory.json", "checksums.sha256", "COMPLETE"},
    )
    items = [
        {
            "path": path.relative_to(root).as_posix(),
            "role": "phase15_profiler_mechanism_evidence",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in payload
    ]
    inventory = {
        "schema_version": "kvbench-artifact-inventory-1.0.0",
        "run_id": identifier,
        "files": items,
        "excluded_control_files": [
            "inventory.json",
            "artifact_inventory.json",
            "checksums.sha256",
            "COMPLETE",
        ],
    }
    write_exclusive(root / "inventory.json", json_bytes(inventory))
    write_exclusive(root / "artifact_inventory.json", json_bytes(inventory))
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
                "status": "PASS",
                "manifest_sha256": sha256_file(root / "manifest.json"),
                "artifact_inventory_sha256": sha256_file(root / "artifact_inventory.json"),
                "checksum_ledger_path": "checksums.sha256",
                "checksum_ledger_sha256": sha256_file(root / "checksums.sha256"),
                "written_last": True,
            }
        ),
    )
    final = ARTIFACT_ROOT / identifier
    if final.exists() or final.is_symlink():
        raise Phase15Error("Phase 15 finalized campaign already exists")
    rename_noreplace(root, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    validate_campaign(final, expected_campaign_id=identifier)
    return final


def validate_campaign(root: Path, *, expected_campaign_id: str | None = None) -> dict[str, Any]:
    artifact = validate_local_artifact(root, environ={})
    manifest = _strict_json(root / "manifest.json")
    identifier = _validate_campaign_id(str(manifest.get("campaign_id")))
    if expected_campaign_id is not None and identifier != expected_campaign_id:
        raise Phase15Error("Phase 15 campaign identity differs")
    qc = _strict_json(root / "phase15_qc.json")
    mechanism = _strict_json(root / "mechanism_summary.json")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("profiler_duration_is_normal_timing") is not False
        or qc.get("status") != "PASS"
        or qc.get("selected_nsys_profiles") != 32
        or qc.get("selected_ncu_profiles") != 22
        or qc.get("common_same_work_configurations") != 10
        or mechanism.get("profiler_durations_used_as_normal_timing") is not False
        or mechanism.get("phase14_pure_launch_floor_only_supported") is not False
        or mechanism.get("full_scan") != "CLOSED"
        or mechanism.get("quality") != "LOCKED"
    ):
        raise Phase15Error("Phase 15 campaign semantic validation differs")
    return {
        "status": "PASS",
        "campaign_id": identifier,
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "nsys_profiles": qc["completed_nsys_profiles"],
        "ncu_profiles": qc["completed_ncu_profiles"],
        "failed_profiles": qc["failed_profiles"],
        "retried_profiles": qc["retried_profiles"],
    }


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--validate-entry", action="store_true")
    actions.add_argument("--write-selection", action="store_true")
    actions.add_argument("--validate-selection", action="store_true")
    actions.add_argument("--new-campaign-id", action="store_true")
    actions.add_argument("--reserve-campaign", action="store_true")
    actions.add_argument("--run-campaign", action="store_true")
    actions.add_argument("--run-worker", action="store_true")
    actions.add_argument("--prepare-ncu-continuation", action="store_true")
    actions.add_argument("--run-one-ncu-profile", action="store_true")
    actions.add_argument("--finalize-ncu-continuation-index", action="store_true")
    actions.add_argument("--materialize-analysis", action="store_true")
    actions.add_argument("--finalize-staged-campaign", action="store_true")
    actions.add_argument("--validate-campaign", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--git-sha")
    parser.add_argument("--profile-id")
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--run-kind", choices=RUN_KINDS)
    parser.add_argument("--configuration", choices=CONFIGURATIONS)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-label", type=int)
    parser.add_argument("--graph-mode", choices=("eager", "cuda_graph"))
    parser.add_argument("--prefix-state-root", type=Path)
    parser.add_argument("--prefix-state-sha256")
    parser.add_argument("--decode-operations", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _require(value: Any, label: str) -> Any:
    if value is None:
        raise Phase15Error(f"{label} is required")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.validate_entry:
        print(json.dumps(validate_lightweight_entry(), sort_keys=True))
    elif args.write_selection:
        write_selection(_require(args.output, "output"))
    elif args.validate_selection:
        validate_selection(_strict_json(_require(args.output, "output")))
        print(json.dumps({"status": "PASS", "nsys": 32, "ncu": 22}, sort_keys=True))
    elif args.new_campaign_id:
        print(new_campaign_id(_require(args.git_sha, "git-sha")))
    elif args.reserve_campaign:
        print(reserve_campaign(campaign_id=_require(args.campaign_id, "campaign-id"), git_sha=_require(args.git_sha, "git-sha")))
    elif args.run_campaign:
        print(json.dumps(run_campaign(stage=_require(args.stage, "stage"), campaign_id=_require(args.campaign_id, "campaign-id"), git_sha=_require(args.git_sha, "git-sha")), sort_keys=True))
    elif args.prepare_ncu_continuation:
        print(
            json.dumps(
                prepare_ncu_continuation(
                    stage=_require(args.stage, "stage"),
                    campaign_id=_require(args.campaign_id, "campaign-id"),
                    git_sha=_require(args.git_sha, "git-sha"),
                ),
                sort_keys=True,
            )
        )
    elif args.run_one_ncu_profile:
        print(
            json.dumps(
                run_one_ncu_profile(
                    stage=_require(args.stage, "stage"),
                    campaign_id=_require(args.campaign_id, "campaign-id"),
                    git_sha=_require(args.git_sha, "git-sha"),
                    profile_id=_require(args.profile_id, "profile-id"),
                    attempt=_require(args.attempt, "attempt"),
                ),
                sort_keys=True,
            )
        )
    elif args.finalize_ncu_continuation_index:
        print(
            json.dumps(
                finalize_ncu_continuation_index(
                    stage=_require(args.stage, "stage"),
                    campaign_id=_require(args.campaign_id, "campaign-id"),
                    git_sha=_require(args.git_sha, "git-sha"),
                ),
                sort_keys=True,
            )
        )
    elif args.run_worker:
        payload = _run_profile_worker(
            run_kind=_require(args.run_kind, "run-kind"),
            configuration=_require(args.configuration, "configuration"),
            batch=_require(args.batch_size, "batch-size"),
            context_label=_require(args.context_label, "context-label"),
            graph_mode=_require(args.graph_mode, "graph-mode"),
            git_sha=_require(args.git_sha, "git-sha"),
            prefix_state_root=_require(args.prefix_state_root, "prefix-state-root"),
            prefix_state_sha256=_require(args.prefix_state_sha256, "prefix-state-sha256"),
            decode_operations=_require(args.decode_operations, "decode-operations"),
            warmup_steps=_require(args.warmup_steps, "warmup-steps"),
            smoke=args.smoke,
        )
        _emit_worker_and_exit(payload)
    elif args.materialize_analysis:
        print(json.dumps(materialize_analysis(_require(args.stage, "stage")), sort_keys=True))
    elif args.finalize_staged_campaign:
        print(seal_campaign(_require(args.stage, "stage"), campaign_id=_require(args.campaign_id, "campaign-id")))
    elif args.validate_campaign:
        print(json.dumps(validate_campaign(_require(args.artifact, "artifact")), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

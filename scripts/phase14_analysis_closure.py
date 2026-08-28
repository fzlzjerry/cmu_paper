#!/usr/bin/env python3
"""Close Phase 14 from its immutable, already-published analysis tables.

This module is deliberately analysis-only.  It never imports torch, opens a
CUDA device, regenerates a fit, or writes into the source Phase 14 campaign.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import stat
import subprocess
from typing import Any

from kvbench.runtime.artifacts import sha256_file
from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from scripts.r2_artifact import validate_local_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CAMPAIGN_ID = "phase14-20260826t115110887808z-47ba4220-42fc95"
SOURCE_CAMPAIGN_ROOT_SHA256 = (
    "22a613b07c1ee6d3e9a0a7fc81df6065ccc8a10bf783b1a69aded3c2eb8068f0"
)
SOURCE_R2_URI = (
    "r2://kvbench-artifacts/kvbench/sha256/"
    f"{SOURCE_CAMPAIGN_ROOT_SHA256}/"
)
SOURCE_CAMPAIGN_ROOT = (
    REPOSITORY_ROOT / "artifacts" / "phase14" / SOURCE_CAMPAIGN_ID
)
SOURCE_REPORT_PATH = REPOSITORY_ROOT / "docs/phase_reports/phase14-graph-ab.md"
SOURCE_RECEIPT_PATH = REPOSITORY_ROOT / "docs/evidence/phase14/r2-publication.json"
PHASE13_KNEES_PATH = (
    REPOSITORY_ROOT
    / "artifacts/phase13"
    / "phase13-20260822t150835736582z-4ddd7b17-3a8fb3"
    / "provisional_knees.parquet"
)
PHASE13D_KNEES_PATH = (
    REPOSITORY_ROOT
    / "artifacts/phase13d"
    / "phase13d-20260825t030556684636z-a06837a3-83761a"
    / "refined_knees.parquet"
)
CLOSURE_JSON_PATH = REPOSITORY_ROOT / "docs/evidence/phase14/analysis-closure.json"
CLOSURE_REPORT_PATH = (
    REPOSITORY_ROOT / "docs/phase_reports/phase14-analysis-closure.md"
)
CLOSURE_RECEIPT_PATH = (
    REPOSITORY_ROOT
    / "docs/evidence/phase14/analysis-closure-r2-publication.json"
)
CLOSURE_ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts/phase14"

CV_THRESHOLD = 0.03
EXPECTED_COMPLETED_RUNS = 660
EXPECTED_STABLE_CONDITIONS = 105
EXPECTED_UNSTABLE_CONDITIONS = 5
EXPECTED_IDENTIFIABLE_COMPARISONS = 14
EXPECTED_UNSTABLE_EAGER_COMPARISONS = 4
EXPECTED_NO_POSITIVE_SLOPE_COMPARISONS = 2

SOURCE_FILE_HASHES = {
    "docs/evidence/phase14/r2-publication.json": (
        "f5de7bc57f79ab40a9299d7f5c8bf9b90f0f7a8968db3666e8f0bab460050114"
    ),
    "docs/phase_reports/phase14-graph-ab.md": (
        "96210af393acfeaa00a92ed5e41e41f654bc9c7c67205db29abe81a1e9e5acb3"
    ),
    f"artifacts/phase14/{SOURCE_CAMPAIGN_ID}/campaign_manifest.json": (
        "fd84f79c6822e108b934af7a3dd2dfb5530843fe0634182ba9299062e82ed535"
    ),
    f"artifacts/phase14/{SOURCE_CAMPAIGN_ID}/raw_run_index.parquet": (
        "7fc4ef98354682b78d27114dd2f5365eb3217afb4f78dc6934e4c2903bd739cf"
    ),
    f"artifacts/phase14/{SOURCE_CAMPAIGN_ID}/point_summary.parquet": (
        "1e4993bdd5b75e62108881c9f9dfe402f3dbd785b576347a8b16c244f5163e4c"
    ),
    f"artifacts/phase14/{SOURCE_CAMPAIGN_ID}/graph_ab_pairs.parquet": (
        "fead54b996459cdafd6c33c50ed100906092b4a1bdb6d59671762392f6b8d447"
    ),
    f"artifacts/phase14/{SOURCE_CAMPAIGN_ID}/mode_fits.parquet": (
        "74d7a72df72c25b489ee1943c086d5f0c87e65d887eff9fd8162c3539566e660"
    ),
    f"artifacts/phase14/{SOURCE_CAMPAIGN_ID}/graph_effects.parquet": (
        "c873831e75e233544f2f24d267cc4e515aa025bafd6d32e1060d8a6b19e66996"
    ),
    f"artifacts/phase14/{SOURCE_CAMPAIGN_ID}/phase14_qc.json": (
        "ab353799906820bafa9f44b7f8c6cfb5c601647e5c45901bf87b013aabd7fe53"
    ),
    f"artifacts/phase14/{SOURCE_CAMPAIGN_ID}/phase14_report.md": (
        "8ca872a4ef2ba0caaa3d00c60da317806a73aa89111c67c44adced70abb0b53b"
    ),
    (
        "artifacts/phase13/phase13-20260822t150835736582z-4ddd7b17-3a8fb3/"
        "provisional_knees.parquet"
    ): "7e89f868a569f06d0109f8ce48847cffc837d0919cb0a423ac7d8b00737a1ff8",
    (
        "artifacts/phase13d/phase13d-20260825t030556684636z-a06837a3-83761a/"
        "refined_knees.parquet"
    ): "100621fc81522a343a5c05bc7b0d5d3eacd32280eb0a78b35c364dda61032339",
}

_BUNDLE_ID_RE = re.compile(
    r"phase14c-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)


class Phase14ClosureError(RuntimeError):
    """The immutable Phase 14 closure contract failed closed."""


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite value {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise Phase14ClosureError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise Phase14ClosureError(f"JSON evidence is not an object: {path}")
    return value


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise Phase14ClosureError(
            "the pinned pyarrow analysis environment is required"
        ) from error
    try:
        return [dict(row) for row in parquet.read_table(path).to_pylist()]
    except (OSError, TypeError, ValueError) as error:
        raise Phase14ClosureError(f"invalid Parquet evidence: {path}") from error


def _counter(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def _comparison_category(row: Mapping[str, Any]) -> str:
    eager = row.get("eager_fit_status")
    graph = row.get("graph_fit_status")
    if eager == "knee_observed" and graph == "knee_observed":
        return "fully_identifiable"
    if eager == "unstable_data":
        return "unstable_eager"
    if eager == "no_positive_slope":
        return "no_positive_eager_slope"
    return "other_non_identifiable"


def criterion_supported(row: Mapping[str, Any]) -> bool:
    """Recompute the frozen complete launch-floor-only criterion."""

    return bool(
        _comparison_category(row) == "fully_identifiable"
        and row.get("material_floor_reduction") is True
        and row.get("slope_similar") is True
        and row.get("semantics_unchanged") is True
        and row.get("host_minus_device_proxy_lower") is True
    )


def summarize_tables(
    point_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    fit_rows: Sequence[Mapping[str, Any]],
    effect_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and summarize the stored Phase 14 analysis without refitting."""

    if (len(point_rows), len(pair_rows), len(fit_rows), len(effect_rows)) != (
        240,
        120,
        40,
        20,
    ):
        raise Phase14ClosureError("Phase 14 analysis-table cardinality differs")

    point_counts = _counter(point_rows, "disposition")
    if point_counts != {
        "pair_capacity_infeasible": 20,
        "stable": 215,
        "unstable": 5,
    }:
        raise Phase14ClosureError("Phase 14 point dispositions differ")
    completed_runs = sum(
        int(row.get("completed_replicates", 0))
        for row in point_rows
        if row.get("disposition") in {"stable", "unstable"}
    )
    if completed_runs != EXPECTED_COMPLETED_RUNS:
        raise Phase14ClosureError("Phase 14 completed-run count differs")

    unstable = sorted(
        (
            {
                "method_config_id": str(row["method_config_id"]),
                "batch_size": int(row["batch_size"]),
                "context_label": int(row["context_label"]),
                "graph_mode": str(row["graph_mode"]),
                "cv": float(row["cv"]),
                "status": "unstable",
            }
            for row in point_rows
            if row.get("disposition") == "unstable"
        ),
        key=lambda row: (
            row["method_config_id"],
            row["batch_size"],
            row["context_label"],
        ),
    )
    if len(unstable) != EXPECTED_UNSTABLE_CONDITIONS or any(
        row["graph_mode"] != "eager" or row["cv"] <= CV_THRESHOLD
        for row in unstable
    ):
        raise Phase14ClosureError("unstable eager conditions differ")
    if any(
        float(row["cv"]) > CV_THRESHOLD
        for row in point_rows
        if row.get("disposition") == "stable"
    ):
        raise Phase14ClosureError("a stable point exceeds the frozen CV threshold")

    pair_counts = _counter(pair_rows, "pair_status")
    if pair_counts != {
        "pair_capacity_infeasible": 10,
        "stable": 105,
        "unstable": 5,
    }:
        raise Phase14ClosureError("Phase 14 A/B condition counts differ")
    stable_pairs = [row for row in pair_rows if row.get("pair_status") == "stable"]
    semantic_mismatches = sum(
        not (
            row.get("output_agreement") is True
            and row.get("kernel_path_agreement") is True
            and row.get("cache_identity_agreement") is True
        )
        for row in stable_pairs
    )
    if semantic_mismatches:
        raise Phase14ClosureError("stable Phase 14 pair semantics differ")
    proxy_lower = sum(
        float(row["wall_minus_gpu_graph_ms"])
        < float(row["wall_minus_gpu_eager_ms"])
        for row in stable_pairs
    )

    fit_counts = _counter(fit_rows, "fit_status")
    if fit_counts != {
        "knee_observed": 34,
        "no_positive_slope": 2,
        "unstable_data": 4,
    }:
        raise Phase14ClosureError("Phase 14 mode-fit statuses differ")

    categories = Counter(_comparison_category(row) for row in effect_rows)
    expected_categories = {
        "fully_identifiable": EXPECTED_IDENTIFIABLE_COMPARISONS,
        "unstable_eager": EXPECTED_UNSTABLE_EAGER_COMPARISONS,
        "no_positive_eager_slope": EXPECTED_NO_POSITIVE_SLOPE_COMPARISONS,
        "other_non_identifiable": 0,
    }
    if {key: categories.get(key, 0) for key in expected_categories} != expected_categories:
        raise Phase14ClosureError("Phase 14 comparison categories differ")
    if any(
        bool(row.get("launch_floor_interpretation_supported"))
        != criterion_supported(row)
        for row in effect_rows
    ):
        raise Phase14ClosureError("stored launch-floor criterion differs")

    identifiable = [
        row
        for row in effect_rows
        if _comparison_category(row) == "fully_identifiable"
    ]
    support_count = sum(criterion_supported(row) for row in identifiable)
    floor_decreases = sum(float(row["floor_delta_ms"]) > 0 for row in identifiable)
    material_floor_reductions = sum(
        row.get("material_floor_reduction") is True for row in identifiable
    )
    slope_similar = sum(row.get("slope_similar") is True for row in identifiable)
    knee_shifts = Counter(float(row["knee_shift_tokens"]) for row in identifiable)
    graph_knee_disappeared = sum(
        row.get("eager_fit_status") == "knee_observed"
        and row.get("graph_fit_status") != "knee_observed"
        for row in identifiable
    )

    return {
        "completed_mode_runs": completed_runs,
        "point_status_counts": point_counts,
        "stable_ab_condition_count": len(stable_pairs),
        "pair_status_counts": pair_counts,
        "unstable_eager_conditions": unstable,
        "fit_status_counts": fit_counts,
        "comparison_categories": expected_categories,
        "fully_identifiable_comparisons": [
            {
                "method_config_id": str(row["method_config_id"]),
                "batch_size": int(row["batch_size"]),
            }
            for row in identifiable
        ],
        "unstable_eager_comparisons": [
            {
                "method_config_id": str(row["method_config_id"]),
                "batch_size": int(row["batch_size"]),
            }
            for row in effect_rows
            if _comparison_category(row) == "unstable_eager"
        ],
        "no_positive_eager_slope_comparisons": [
            {
                "method_config_id": str(row["method_config_id"]),
                "batch_size": int(row["batch_size"]),
            }
            for row in effect_rows
            if _comparison_category(row) == "no_positive_eager_slope"
        ],
        "launch_floor_only": {
            "support_count": support_count,
            "eligible_denominator": len(identifiable),
            "floor_decreases": floor_decreases,
            "material_floor_reductions": material_floor_reductions,
            "slope_similar": slope_similar,
            "semantics_unchanged": sum(
                row.get("semantics_unchanged") is True for row in identifiable
            ),
        },
        "knee": {
            "shift_counts": {
                str(int(shift)): count for shift, count in sorted(knee_shifts.items())
            },
            "graph_knee_disappeared": graph_knee_disappeared,
        },
        "host_minus_device_proxy": {
            "graph_lower_count": proxy_lower,
            "stable_pair_denominator": len(stable_pairs),
            "graph_not_lower_count": len(stable_pairs) - proxy_lower,
            "direct_launch_gap_measured": False,
        },
        "semantic_mismatches": semantic_mismatches,
    }


def _validate_source_hashes() -> None:
    for relative, expected in SOURCE_FILE_HASHES.items():
        path = REPOSITORY_ROOT / relative
        if sha256_file(path) != expected:
            raise Phase14ClosureError(f"immutable source hash differs: {relative}")


def source_summary() -> dict[str, Any]:
    """Read and validate every table required by the Phase 14 closure."""

    _validate_source_hashes()
    campaign = _strict_json(SOURCE_CAMPAIGN_ROOT / "campaign_manifest.json")
    qc = _strict_json(SOURCE_CAMPAIGN_ROOT / "phase14_qc.json")
    receipt = _strict_json(SOURCE_RECEIPT_PATH)
    if (
        campaign.get("campaign_id") != SOURCE_CAMPAIGN_ID
        or campaign.get("planned_pair_records") != 360
        or campaign.get("planned_run_records") != 720
        or qc.get("completed_runs") != EXPECTED_COMPLETED_RUNS
        or qc.get("selective_reruns") != 0
        or receipt.get("status") != "PASS"
        or receipt.get("publication", {}).get("root_sha256")
        != SOURCE_CAMPAIGN_ROOT_SHA256
        or receipt.get("clean_retrieval", {}).get("result") != "PASS"
    ):
        raise Phase14ClosureError("immutable Phase 14 campaign authority differs")

    summary = summarize_tables(
        _read_parquet(SOURCE_CAMPAIGN_ROOT / "point_summary.parquet"),
        _read_parquet(SOURCE_CAMPAIGN_ROOT / "graph_ab_pairs.parquet"),
        _read_parquet(SOURCE_CAMPAIGN_ROOT / "mode_fits.parquet"),
        _read_parquet(SOURCE_CAMPAIGN_ROOT / "graph_effects.parquet"),
    )
    phase13 = _read_parquet(PHASE13_KNEES_PATH)
    phase13d = _read_parquet(PHASE13D_KNEES_PATH)
    if len(phase13) != 30 or _counter(phase13, "fit_status") != {
        "knee_observed": 30
    }:
        raise Phase14ClosureError("Phase 13 knee authority differs")
    if (
        len(phase13d) != 30
        or _counter(phase13d, "fit_status") != {"knee_observed": 30}
        or _counter(phase13d, "resolution_status")
        != {"density_sufficient": 6, "insufficient_feasible_span": 24}
    ):
        raise Phase14ClosureError("Phase 13D knee authority differs")
    summary["upstream_knee_tables"] = {
        "phase13_rows": len(phase13),
        "phase13_fit_status_counts": _counter(phase13, "fit_status"),
        "phase13d_rows": len(phase13d),
        "phase13d_fit_status_counts": _counter(phase13d, "fit_status"),
        "phase13d_resolution_status_counts": _counter(
            phase13d, "resolution_status"
        ),
    }
    return summary


def build_closure_payload(
    summary: Mapping[str, Any],
    *,
    analysis_git_sha: str,
    recorded_at_utc: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", analysis_git_sha) is None:
        raise Phase14ClosureError("analysis Git SHA is invalid")
    payload = {
        "schema_version": "kvbench-phase14-analysis-closure-1.0.0",
        "status": "PASS",
        "recorded_at_utc": recorded_at_utc,
        "analysis_execution_git_sha": analysis_git_sha,
        "analysis_scope": {
            "analysis_only": True,
            "cuda_run": False,
            "timing_rerun": False,
            "source_campaign_modified": False,
            "cv_threshold_changed": False,
            "fits_regenerated": False,
        },
        "source_phase14": {
            "campaign_id": SOURCE_CAMPAIGN_ID,
            "r2_root_sha256": SOURCE_CAMPAIGN_ROOT_SHA256,
            "r2_uri": SOURCE_R2_URI,
            "object_count": 18628,
            "original_status": "BLOCKED",
            "planned_ab_replicate_pairs": 360,
            "feasible_ab_replicate_pairs": 330,
            "capacity_infeasible_ab_replicate_pairs": 30,
            "completed_mode_runs": summary["completed_mode_runs"],
            "runtime_failures": 0,
            "selective_reruns": 0,
            "report_path": "docs/phase_reports/phase14-graph-ab.md",
            "report_sha256": SOURCE_FILE_HASHES[
                "docs/phase_reports/phase14-graph-ab.md"
            ],
            "campaign_report_sha256": SOURCE_FILE_HASHES[
                f"artifacts/phase14/{SOURCE_CAMPAIGN_ID}/phase14_report.md"
            ],
            "file_sha256": dict(sorted(SOURCE_FILE_HASHES.items())),
        },
        "stability": {
            "cv_threshold": CV_THRESHOLD,
            "stable_ab_condition_count": summary["stable_ab_condition_count"],
            "unstable_eager_condition_count": len(
                summary["unstable_eager_conditions"]
            ),
            "unstable_eager_conditions": summary["unstable_eager_conditions"],
            "unstable_points_reclassified": 0,
        },
        "mechanism_summary": {
            "comparison_categories": summary["comparison_categories"],
            "fully_identifiable_comparisons": summary[
                "fully_identifiable_comparisons"
            ],
            "unstable_eager_comparisons": summary[
                "unstable_eager_comparisons"
            ],
            "no_positive_eager_slope_comparisons": summary[
                "no_positive_eager_slope_comparisons"
            ],
            "launch_floor_only": summary["launch_floor_only"],
            "knee": summary["knee"],
            "host_minus_device_proxy": summary["host_minus_device_proxy"],
            "semantic_mismatches": summary["semantic_mismatches"],
            "fit_status_counts": summary["fit_status_counts"],
        },
        "upstream_knee_tables": summary["upstream_knee_tables"],
        "scientific_interpretation": (
            "The observed CUDA Graph effect is heterogeneous and is not "
            "explained by a pure launch-floor reduction model."
        ),
        "limitations": [
            (
                "Five eager condition groups remain unstable and their eager "
                "mechanism parameters are not claim-bearing."
            ),
            (
                "Phase 15 profiling is required for direct launch-gap and "
                "physical traffic attribution."
            ),
        ],
        "claims": {
            "mechanism_experiment": True,
            "quality_status": "unvalidated",
            "performance_claim_eligible": False,
            "direct_launch_gap_measured": False,
            "hbm_traffic_claim": False,
        },
        "gates": {
            "phase14_analysis_closure": "PASS",
            "phase15": "READY",
            "full_scan": "CLOSED",
            "quality": "LOCKED",
            "performance_data_frozen": False,
        },
        "durable_publication": {
            "bundle_scope": "closure_only",
            "source_phase14_objects_reuploaded": False,
            "source_phase14_root_reference": SOURCE_CAMPAIGN_ROOT_SHA256,
            "receipt_path": (
                "docs/evidence/phase14/analysis-closure-r2-publication.json"
            ),
            "self_reference_safe": True,
        },
    }
    validate_closure_payload(payload, summary=summary)
    return payload


def validate_closure_payload(
    payload: Mapping[str, Any],
    *,
    summary: Mapping[str, Any] | None = None,
) -> None:
    mechanism = payload.get("mechanism_summary")
    gates = payload.get("gates")
    source = payload.get("source_phase14")
    stability = payload.get("stability")
    if (
        payload.get("schema_version")
        != "kvbench-phase14-analysis-closure-1.0.0"
        or payload.get("status") != "PASS"
        or not isinstance(mechanism, Mapping)
        or not isinstance(gates, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(stability, Mapping)
        or source.get("campaign_id") != SOURCE_CAMPAIGN_ID
        or source.get("r2_root_sha256") != SOURCE_CAMPAIGN_ROOT_SHA256
        or source.get("report_sha256")
        != SOURCE_FILE_HASHES["docs/phase_reports/phase14-graph-ab.md"]
        or stability.get("cv_threshold") != CV_THRESHOLD
        or stability.get("unstable_points_reclassified") != 0
        or gates
        != {
            "phase14_analysis_closure": "PASS",
            "phase15": "READY",
            "full_scan": "CLOSED",
            "quality": "LOCKED",
            "performance_data_frozen": False,
        }
    ):
        raise Phase14ClosureError("Phase 14 closure payload differs")
    launch = mechanism.get("launch_floor_only")
    categories = mechanism.get("comparison_categories")
    proxy = mechanism.get("host_minus_device_proxy")
    if (
        not isinstance(launch, Mapping)
        or launch.get("support_count") != 0
        or launch.get("eligible_denominator")
        != EXPECTED_IDENTIFIABLE_COMPARISONS
        or categories
        != {
            "fully_identifiable": 14,
            "unstable_eager": 4,
            "no_positive_eager_slope": 2,
            "other_non_identifiable": 0,
        }
        or not isinstance(proxy, Mapping)
        or proxy.get("graph_lower_count") != 23
        or proxy.get("stable_pair_denominator") != 105
    ):
        raise Phase14ClosureError("Phase 14 closure aggregation differs")
    if summary is not None and (
        stability.get("unstable_eager_conditions")
        != summary["unstable_eager_conditions"]
        or mechanism.get("fully_identifiable_comparisons")
        != summary["fully_identifiable_comparisons"]
    ):
        raise Phase14ClosureError("closure payload does not match source tables")


def render_report(payload: Mapping[str, Any]) -> str:
    validate_closure_payload(payload)
    mechanism = payload["mechanism_summary"]
    launch = mechanism["launch_floor_only"]
    proxy = mechanism["host_minus_device_proxy"]
    unstable = payload["stability"]["unstable_eager_conditions"]
    lines = [
        "# Phase 14C Analysis Closure",
        "",
        "Status: PASS",
        "",
        f"- Original campaign: `{SOURCE_CAMPAIGN_ID}`",
        f"- Original R2 root: `{SOURCE_CAMPAIGN_ROOT_SHA256}`",
        f"- Original report SHA-256: `{payload['source_phase14']['report_sha256']}`",
        "- Completed execution: 660/660 feasible mode runs; zero runtime failures; zero selective reruns.",
        "- Stable A/B conditions: 105.",
        "- The frozen CV threshold remains 3%; five eager conditions remain `unstable`:",
    ]
    lines.extend(
        f"  - `{row['method_config_id']}` B={row['batch_size']} "
        f"L={row['context_label']}: CV `{row['cv']}`"
        for row in unstable
    )
    lines.extend(
        [
            "- Fully identifiable floor+slope+knee comparisons: 14.",
            "- Inconclusive because eager data are unstable: 4 comparisons (`bf16` B=1/B=4 and `tq_3bit_nc` B=1/B=4).",
            "- Ineligible because the eager fit has no positive slope: 2 comparisons (`k4v4` B=4 and `k2v2` B=4).",
            "- Other non-identifiable comparisons: 0.",
            f"- Complete launch-floor-only support: {launch['support_count']} of {launch['eligible_denominator']} fully identifiable comparisons.",
            f"- Floors decrease numerically in {launch['floor_decreases']} of 14 comparisons; {launch['material_floor_reductions']} meet the frozen meaningful-floor threshold.",
            f"- Slopes satisfy the frozen similarity interval in {launch['slope_similar']} of 14 comparisons.",
            "- Knee shifts among the 14 identifiable comparisons: 9 unchanged, 2 at -6144 tokens, and 3 at -61440 tokens; Graph removes no observed knee.",
            f"- Host-minus-device proxy is lower under Graph in {proxy['graph_lower_count']} of {proxy['stable_pair_denominator']} stable A/B conditions. This is not a direct launch-gap measurement.",
            "- Output, backend, cache layout, kernel path, and allocation mismatches: 0.",
            "",
            "Scientific interpretation: The observed CUDA Graph effect is heterogeneous and is not explained by a pure launch-floor reduction model. This is a negative mechanism result, not a failed experiment.",
            "",
            "Limitations:",
            "- The five unstable eager conditions remain excluded from stable fitted mechanism comparisons; their eager mechanism parameters are not claim-bearing.",
            "- Phase 15 profiling is required for direct launch-gap and physical traffic attribution.",
            "",
            "- Phase 15: `READY`",
            "- Full Scan: `CLOSED`",
            "- Quality: `LOCKED`",
            "- Performance claims: ineligible; quality remains unvalidated.",
            "- Closure publication receipt: `docs/evidence/phase14/analysis-closure-r2-publication.json` (self-reference-safe and outside the closure bundle).",
            "",
        ]
    )
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def new_bundle_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"phase14c-{stamp}z-{head[:8]}-{secrets.token_hex(3)}"


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase14ClosureError("closure bundle contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase14ClosureError("closure bundle contains an unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            result.append(path)
    return result


def seal_bundle(bundle_id: str) -> Path:
    if _BUNDLE_ID_RE.fullmatch(bundle_id) is None:
        raise Phase14ClosureError("closure bundle ID is invalid")
    summary = source_summary()
    closure = _strict_json(CLOSURE_JSON_PATH)
    validate_closure_payload(closure, summary=summary)
    report = CLOSURE_REPORT_PATH.read_text(encoding="utf-8")
    if report != render_report(closure):
        raise Phase14ClosureError("closure report does not match closure JSON")

    CLOSURE_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    stage = CLOSURE_ARTIFACT_ROOT / f".{bundle_id}.{secrets.token_hex(8)}.staging"
    final = CLOSURE_ARTIFACT_ROOT / bundle_id
    if final.exists() or final.is_symlink():
        raise Phase14ClosureError("closure bundle already exists")
    stage.mkdir(mode=0o700)
    write_exclusive(stage / "analysis-closure.json", CLOSURE_JSON_PATH.read_bytes())
    write_exclusive(stage / "analysis-closure.md", CLOSURE_REPORT_PATH.read_bytes())
    write_exclusive(
        stage / "source-reference.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase14c-source-reference-1.0.0",
                "source_campaign_id": SOURCE_CAMPAIGN_ID,
                "source_root_sha256": SOURCE_CAMPAIGN_ROOT_SHA256,
                "source_r2_uri": SOURCE_R2_URI,
                "source_object_count": 18628,
                "source_objects_reuploaded": False,
                "source_file_sha256": dict(sorted(SOURCE_FILE_HASHES.items())),
            }
        ),
    )
    write_exclusive(
        stage / "manifest.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase14c-artifact-manifest-1.0.0",
                "run_id": bundle_id,
                "status": "PASS",
                "created_at_utc": _utc_now(),
                "append_only": True,
                "complete_written_last": True,
                "scope": "analysis_closure_only",
                "source_campaign_id": SOURCE_CAMPAIGN_ID,
                "source_root_sha256": SOURCE_CAMPAIGN_ROOT_SHA256,
                "source_objects_reuploaded": False,
                "quality_status": "unvalidated",
                "performance_claim_eligible": False,
            }
        ),
    )
    items = [
        {
            "path": path.relative_to(stage).as_posix(),
            "role": "phase14c_analysis_closure",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _payload_paths(
            stage, {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
        )
    ]
    write_exclusive(
        stage / "artifact_inventory.json",
        json_bytes(
            {
                "schema_version": "kvbench-artifact-inventory-1.0.0",
                "run_id": bundle_id,
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
        f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n"
        for path in _payload_paths(stage, {"checksums.sha256", "COMPLETE"})
    ).encode("utf-8")
    write_exclusive(stage / "checksums.sha256", ledger)
    write_exclusive(
        stage / "COMPLETE",
        json_bytes(
            {
                "schema_version": "kvbench-completion-1.0.0",
                "run_id": bundle_id,
                "status": "PASS",
                "manifest_sha256": sha256_file(stage / "manifest.json"),
                "artifact_inventory_sha256": sha256_file(
                    stage / "artifact_inventory.json"
                ),
                "checksum_ledger_path": "checksums.sha256",
                "checksum_ledger_sha256": sha256_file(stage / "checksums.sha256"),
                "written_last": True,
            }
        ),
    )
    rename_noreplace(stage, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    validate_bundle(final)
    return final


def validate_bundle(root: Path) -> dict[str, Any]:
    artifact = validate_local_artifact(root, environ={})
    manifest = _strict_json(root / "manifest.json")
    closure = _strict_json(root / "analysis-closure.json")
    reference = _strict_json(root / "source-reference.json")
    validate_closure_payload(closure, summary=source_summary())
    if (
        manifest.get("status") != "PASS"
        or manifest.get("source_root_sha256") != SOURCE_CAMPAIGN_ROOT_SHA256
        or manifest.get("source_objects_reuploaded") is not False
        or reference.get("source_root_sha256") != SOURCE_CAMPAIGN_ROOT_SHA256
        or reference.get("source_objects_reuploaded") is not False
        or (root / "analysis-closure.json").read_bytes()
        != CLOSURE_JSON_PATH.read_bytes()
        or (root / "analysis-closure.md").read_bytes()
        != CLOSURE_REPORT_PATH.read_bytes()
        or len(artifact.files) != 7
    ):
        raise Phase14ClosureError("closure bundle validation differs")
    return {
        "status": "PASS",
        "bundle_id": manifest["run_id"],
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "source_objects_reuploaded": False,
    }


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--new-bundle-id", action="store_true")
    action.add_argument("--seal-bundle", action="store_true")
    action.add_argument("--validate-bundle", type=Path)
    parser.add_argument("--analysis-git-sha")
    parser.add_argument("--recorded-at-utc")
    parser.add_argument("--bundle-id")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    if arguments.new_bundle_id:
        print(new_bundle_id())
        return 0
    if arguments.validate_bundle is not None:
        print(
            json.dumps(
                validate_bundle(arguments.validate_bundle),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if arguments.seal_bundle:
        if not arguments.bundle_id:
            raise Phase14ClosureError("--bundle-id is required")
        print(
            json.dumps(
                validate_bundle(seal_bundle(arguments.bundle_id)),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    summary = source_summary()
    if arguments.analysis_git_sha:
        payload = build_closure_payload(
            summary,
            analysis_git_sha=arguments.analysis_git_sha,
            recorded_at_utc=arguments.recorded_at_utc or _utc_now(),
        )
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Recover checksum-bound Full Scan result files for host-wall analysis only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

from scripts import r2_artifact as r2

SOURCE_OUTER_ROOT = "d74587675dd59b464d81c6e82885d3c9706c681a9da216ad1a7fe4c6ccd88daa"


def parquet_values(rows: list[dict]) -> list[dict]:
    """Mirror the existing artifact writer's nested-value encoding."""
    return [{key: json.dumps(value, sort_keys=True, separators=(",", ":"))
             if isinstance(value, (dict, list, tuple)) else value
             for key, value in row.items() if key != "runner"} for row in rows]


def exact_wall_median(data: bytes, *, digest: str, row: dict) -> float:
    """Use only checksum-bound raw host samples, never the host/device ratio."""
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("source result checksum differs")
    result = json.loads(data)
    for key in ("run_id", "method_config_id", "batch_size", "context_label"):
        if result.get(key) != row[key]:
            raise ValueError("source result identity differs: " + key)
    samples = result["runner"]["timing"]["samples"]
    if len(samples) != row["measured_batches"]:
        raise ValueError("raw timing sample count differs")
    values = []
    device = []
    for sample in samples:
        if sample["completed_operations"] != 256 or sample["failed_operations"] != 0:
            raise ValueError("raw timing operation count differs")
        value = float(sample["host_ns_per_operation"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid raw host timing")
        if not math.isclose(value * 256, sample["host_total_ns"], rel_tol=1e-12):
            raise ValueError("raw host timing totals differ")
        values.append(value / 1_000_000)
        device.append(float(sample["cuda_ms_per_operation"]))
    if statistics.median(device) != row["process_median_ms"]:
        raise ValueError("source device median differs from immutable index")
    return statistics.median(values)


def analyze(family: Path) -> dict:
    from scripts import phase16_full_scan as scan

    source = r2.validate_local_artifact(family / "outer", environ={})
    if source.root_sha256 != SOURCE_OUTER_ROOT:
        raise ValueError("immutable source outer root differs")
    records = []
    sources = []
    segment_index = [scan._publication_record(family, i) for i in range(5)]
    for i, publication in enumerate(segment_index):
        for original in scan._retained_segment_records(family, i, publication):
            row = dict(original)
            row["latency_basis"] = "host_wall"
            if row["status"] == "completed":
                data = (family / "wall-analysis" / "source-results" / (row["run_id"] + ".json")).read_bytes()
                wall = exact_wall_median(data, digest=row["result_sha256"], row=row)
                row["cuda_process_median_ms"] = row["process_median_ms"]
                row["process_median_ms"] = wall
                row["host_wall_process_median_ms"] = wall
                sources.append({"run_id": row["run_id"], "segment_id": row["segment_id"],
                                "source_root_sha256": publication["root_sha256"],
                                "source_result_path": row["result_path"],
                                "source_result_sha256": row["result_sha256"]})
            records.append(row)
    if len(records) != 2670 or len(sources) != 2205:
        raise ValueError("source record coverage differs")
    summaries = scan.point_summaries(records)
    for row in summaries:
        row["latency_basis"] = "host_wall"
    ratios = scan.same_work_ratios(summaries)
    for row in ratios:
        row["latency_basis"] = "host_wall"
    coverage = scan._coverage_status(summaries)
    qc = {
        "schema_version": "kvbench-phase16-host-wall-closure-1.0.0",
        "family_id": family.name,
        "analysis_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "execution_git_sha": scan._strict_json(family / "family-reservation.json")["execution_git_sha"],
        "source_outer_root": SOURCE_OUTER_ROOT,
        "latency_basis": "host_wall",
        "planned_process_records": len(records),
        "completed_records": len(sources),
        "capacity_infeasible_records": sum(r["status"] == "capacity_infeasible" for r in records),
        "stable_logical_points": sum(r["disposition"] == "stable" for r in summaries),
        "unstable_logical_points": sum(r["disposition"] == "unstable" for r in summaries),
        "failed_logical_points": sum(r["disposition"] == "failed" for r in summaries),
        "maximum_cv": max(r["cv"] for r in summaries if r["cv"] is not None),
        "same_work_ratio_count": sum(r["calculated"] for r in ratios),
        "phase17_coverage": coverage,
        "status": "PASS" if coverage["sufficient"] else "PARTIAL",
        "phase17": "READY" if coverage["sufficient"] else "NOT_READY",
        "quality_execution": "LOCKED", "performance_data_frozen": False,
        "timing_rows_r_hbm_null": all(r["r_hbm"] is None for r in records),
        "raw_samples_duplicated": False, "timing_reruns": 0,
        "cv_threshold": 0.03, "standard_deviation": "sample",
    }
    output = family / "wall-closure"
    output.mkdir(exist_ok=False)
    for name, rows in (("raw_run_index", records), ("source_result_index", sources),
                       ("point_summary", summaries), ("same_work_ratios", ratios),
                       ("segment_index", segment_index)):
        scan.phase13._parquet_rows(output / (name + ".parquet"), rows)
    scan.write_exclusive(output / "full_scan_qc.json", scan.json_bytes(qc))
    scan.write_exclusive(output / "source-references.json", scan.json_bytes({
        "source_outer_root": SOURCE_OUTER_ROOT,
        "source_outer_uri": "r2://kvbench-artifacts/kvbench/sha256/" + SOURCE_OUTER_ROOT + "/",
        "unchanged_tables": ["base_grid.parquet", "adaptive_grid.parquet", "feasibility.parquet",
                             "capacity_amplification.parquet", "profiler_feature_join.parquet"],
        "superseded_analysis_fields": ["host-wall summaries", "same-work ratios", "timing plot latency basis"],
        "source_data_modified": False,
    }))
    plots = output / "plots"
    plots.mkdir()
    scan.phase13._svg_line_plot(
        plots / "host-wall-timing.svg", title="Full Scan host-wall timing (quality unvalidated)",
        y_label="median host-wall ms", series={
            f"{config}/B{batch}": [(float(r["historical_context"]), float(r["median_ms"]))
                                  for r in summaries if r["method_config_id"] == config
                                  and r["batch_size"] == batch and r["disposition"] == "stable"]
            for config in scan.CONFIGURATIONS for batch in scan.BATCH_SIZES})
    for name, title, field, rows in (
        ("host-wall-cv", "Host-wall process CV", "cv", summaries),
        ("host-wall-ratios", "Host-wall same-work ratio (quality unvalidated)",
         "performance_only_ratio", ratios),
    ):
        scan.phase13._svg_line_plot(
            plots / (name + ".svg"), title=title, y_label=field,
            series={f"{config}/B{batch}": [
                (float(r["historical_context"]), float(r[field])) for r in rows
                if r["method_config_id"] == config and r["batch_size"] == batch
                and r.get(field) is not None]
                for config in scan.CONFIGURATIONS for batch in scan.BATCH_SIZES},
            note="Performance-only; quality unvalidated; not claim eligible")
    scan.write_exclusive(output / "full_scan_report.md", (
        "# Phase 16 host-wall analysis closure\n\n"
        f"Status: {qc['status']} (durability requires publication receipt).\n\n"
        "Uses exact raw host-wall process medians from checksum-bound result files. "
        "The source outer bundle preserves CUDA-event analysis; its ratios must not be called host-wall ratios. "
        "No timing was rerun, and source artifacts remain immutable.\n\n"
        f"Records: {len(records)}; completed: {len(sources)}; stable points: {qc['stable_logical_points']}; "
        f"unstable: {qc['unstable_logical_points']}; maximum CV: {qc['maximum_cv']:.9%}.\n\n"
        "Quality LOCKED; r_hbm remains null in timing rows; Phase 17 is not started.\n"
    ).encode())
    return scan._seal_artifact(output, run_id=family.name + "-wall-closure",
                               status=qc["status"], role="phase16_host_wall_analysis_closure")


def validate(family: Path) -> dict:
    from scripts import phase16_full_scan as scan

    scan.validate_full_scan(family / "outer")
    artifact = r2.validate_local_artifact(family / "wall-closure", environ={})
    output = family / "wall-closure"
    qc = scan._strict_json(output / "full_scan_qc.json")
    sources = scan._read_parquet(output / "source_result_index.parquet")
    records = scan._read_parquet(output / "raw_run_index.parquet")
    indexed = {r["run_id"]: r for r in records}
    expected = []
    expected_sources = []
    for i in range(5):
        publication = scan._publication_record(family, i)
        for original in scan._retained_segment_records(family, i, publication):
            derived = indexed[original["run_id"]]
            for key, value in parquet_values([original])[0].items():
                if key == "process_median_ms":
                    continue
                if derived.get(key) != value:
                    raise ValueError("derived source field differs: " + key)
            row = dict(original)
            row["latency_basis"] = "host_wall"
            if row["status"] == "completed":
                expected_sources.append({"run_id": row["run_id"], "segment_id": row["segment_id"],
                                         "source_root_sha256": publication["root_sha256"],
                                         "source_result_path": row["result_path"],
                                         "source_result_sha256": row["result_sha256"]})
                data = (family / "wall-analysis" / "source-results" / (row["run_id"] + ".json")).read_bytes()
                wall = exact_wall_median(data, digest=row["result_sha256"], row=row)
                if derived["process_median_ms"] != wall or derived["cuda_process_median_ms"] != row["process_median_ms"]:
                    raise ValueError("derived process median differs")
                row["process_median_ms"] = wall
            expected.append(row)
    if len(records) != 2670 or len(indexed) != 2670 or len(sources) != 2205:
        raise ValueError("derived coverage differs")
    if sources != expected_sources:
        raise ValueError("source result references differ")
    summaries = scan.point_summaries(expected)
    for row in summaries:
        row["latency_basis"] = "host_wall"
    ratios = scan.same_work_ratios(summaries)
    for row in ratios:
        row["latency_basis"] = "host_wall"
    if parquet_values(summaries) != scan._read_parquet(output / "point_summary.parquet"):
        raise ValueError("host-wall summary rederivation differs")
    if parquet_values(ratios) != scan._read_parquet(output / "same_work_ratios.parquet"):
        raise ValueError("host-wall ratio rederivation differs")
    coverage = scan._coverage_status(summaries)
    expected_qc = {
        "planned_process_records": len(records), "completed_records": len(sources),
        "capacity_infeasible_records": sum(r["status"] == "capacity_infeasible" for r in records),
        "stable_logical_points": sum(r["disposition"] == "stable" for r in summaries),
        "unstable_logical_points": sum(r["disposition"] == "unstable" for r in summaries),
        "failed_logical_points": sum(r["disposition"] == "failed" for r in summaries),
        "same_work_ratio_count": sum(r["calculated"] for r in ratios),
        "phase17_coverage": coverage,
        "status": "PASS" if coverage["sufficient"] else "PARTIAL",
        "phase17": "READY" if coverage["sufficient"] else "NOT_READY",
        "standard_deviation": "sample", "timing_reruns": 0,
        "execution_git_sha": scan._strict_json(family / "family-reservation.json")["execution_git_sha"],
    }
    if any(qc.get(key) != value for key, value in expected_qc.items()):
        raise ValueError("host-wall QC rederivation differs")
    if (qc["maximum_cv"] != max(r["cv"] for r in summaries if r["cv"] is not None)
            or qc["cv_threshold"] != 0.03 or qc["latency_basis"] != "host_wall"
            or qc["source_outer_root"] != SOURCE_OUTER_ROOT
            or qc["quality_execution"] != "LOCKED"
            or qc["performance_data_frozen"] is not False
            or any(r["r_hbm"] is not None for r in records)):
        raise ValueError("host-wall QC authority differs")
    return {"status": "PASS", "root_sha256": artifact.root_sha256,
            "source_results_rederived": len(sources), "process_records": len(records),
            "point_summaries": len(summaries), "timing_reruns": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("requests", "fetch", "analyze", "validate"))
    parser.add_argument("family", type=Path)
    args = parser.parse_args()
    if args.operation == "analyze":
        print(json.dumps(analyze(args.family), sort_keys=True))
        return
    if args.operation == "validate":
        print(json.dumps(validate(args.family), sort_keys=True))
        return
    stage = args.family / "wall-analysis"
    if args.operation == "requests":
        from scripts import phase16_full_scan as scan
        stage.mkdir(exist_ok=False)
        requests = []
        for replicate in range(5):
            publication = scan._publication_record(args.family, replicate)
            for row in scan._retained_segment_records(args.family, replicate, publication):
                if row["status"] == "completed":
                    requests.append({
                        "run_id": row["run_id"],
                        "root_sha256": publication["root_sha256"],
                        "path": row["result_path"],
                        "sha256": row["result_sha256"],
                    })
        if len(requests) != 2205 or len({r["run_id"] for r in requests}) != 2205:
            raise ValueError("completed source cardinality differs")
        with (stage / "requests.json").open("x") as stream:
            json.dump(requests, stream, sort_keys=True, indent=2)
        print(json.dumps({"requested_results": len(requests)}), flush=True)
        return
    requests = json.loads((stage / "requests.json").read_text())
    config = r2.R2Config.from_environment()
    client = r2.R2S3Client(config)
    results = stage / "source-results"
    results.mkdir(exist_ok=True)
    for index, record in enumerate(requests):
        name = record["run_id"]
        if Path(name).name != name:
            raise ValueError("unsafe run ID")
        target = results / (name + ".json")
        if target.exists():
            data = target.read_bytes()
        else:
            key = r2.artifact_object_key(config.prefix, record["root_sha256"], record["path"])
            data = client.get_object_or_none(key)
        if data is None or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("source result checksum differs: " + name)
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(data)
        if (index + 1) % 25 == 0 or index + 1 == len(requests):
            print(json.dumps({"verified_results": index + 1, "total": len(requests)}), flush=True)


if __name__ == "__main__":
    main()

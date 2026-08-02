#!/usr/bin/env python3
"""Phase 13F end-to-end Pilot feasibility evidence.

This module performs no timing and launches no model workload.  It recomputes
the frozen 810-record design from source-owned cache formulas plus the complete
prefix-construction and CUDA Graph lifecycle peak, then seals one append-only
diagnostic bundle.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

from kvbench.runtime.artifacts import sha256_file
from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from scripts import phase13_pilot as pilot
from scripts.r2_artifact import validate_local_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase13f"
DECISION_PATH = Path(
    "docs/decisions/0031-phase13-end-to-end-prefix-feasibility.md"
)
AUTHORIZED_CONTAINER_DIGEST = pilot.AUTHORIZED_CONTAINER_DIGEST
CAMPAIGN_PATTERN = re.compile(
    r"phase13f-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)

STOPPED_CAMPAIGNS = (
    {
        "campaign_id": "phase13-20260801t080641686374z-009dfd71-14b7e3",
        "path": Path(
            "artifacts/phase13/"
            "phase13-20260801t080641686374z-009dfd71-14b7e3"
        ),
        "root_sha256": (
            "be8680d3d94dba35d58a98ac13aa5ae3aa2ba47e767c301418b129060466babc"
        ),
        "ledger_sha256": (
            "baec9e845ff8310e6ef27bf3bfc983c816a08353ca61e8efcb263602c14b541c"
        ),
        "report_path": Path("docs/phase_reports/phase13-pilot-scan.md"),
        "report_sha256": (
            "8064c21622cc1f7c61b259de9ca901ecb9f1cf26b3eb4ee428cf3a6b5a98a6f6"
        ),
        "qc_path": Path("docs/evidence/phase13/pilot_qc.json"),
        "qc_sha256": (
            "442bdeb112ce19a8822de9eb1851e159de61461f657cab35c56f72fa463202e4"
        ),
        "receipt_path": Path("docs/evidence/phase13/r2-publication.json"),
        "receipt_sha256": (
            "d72f082991cf7869c8d668331b8482c11a4cb29499fd89a4918271a9000b9570"
        ),
    },
    {
        "campaign_id": "phase13-20260801t184243094922z-e886592f-99b6ca",
        "path": Path(
            "artifacts/phase13/"
            "phase13-20260801t184243094922z-e886592f-99b6ca"
        ),
        "root_sha256": (
            "94104865452017fbfd3c87fffd34e82b12248e379ef2680b22153dd5b71d90b8"
        ),
        "ledger_sha256": (
            "b7e533e367612298f50bf2ba24ea0017bff25dbcdb838c1bb2901d21283fa4ae"
        ),
        "report_path": Path("docs/phase_reports/phase13r-pilot-scan.md"),
        "report_sha256": (
            "d7587691a3c4553517a0187054fe7c39992aaf9f5a3b4610fd98877a1591fd77"
        ),
        "qc_path": Path("docs/evidence/phase13r/pilot_qc.json"),
        "qc_sha256": (
            "e4b6af36d418e889e81d6324aea4c8f2783e1671b0f8459ef94ff3dc485d3eb9"
        ),
        "receipt_path": Path("docs/evidence/phase13r/r2-publication.json"),
        "receipt_sha256": (
            "a70d7a86ebb94710332011e2c04afe17c73b08d729c0ce07de9eeeba0558eefa"
        ),
    },
)

FAILED_RUN = Path(
    "artifacts/phase13/phase13-20260801t184243094922z-e886592f-99b6ca/"
    "runs/phase13-20260801t184243094922z-e886592f-99b6ca-r0-o014-"
    "tq_3bit_nc-b8-l98304"
)
FAILED_STDERR_SHA256 = (
    "5b9c9b87884df65ef3ade4ebbe635e2b9511d65ea5062f0d33bb29973de7ccf4"
)
FAILED_RECORD_SHA256 = (
    "382c0fa9f81e31fbcf98f7c6adcddf8d0babf6c37b204b4855db272c0819487b"
)
SOURCE_HASHES = {
    "src/kvbench/runtime/bf16_endpoint.py": (
        "9095e9a2a9c01e1ea6afb2f1cefcee46a964a82caae7b819a125757b59244a9b"
    ),
    "src/kvbench/runtime/backend.py": (
        "6934cd0fba0674dc2e2ef764afecffb7a958d3be407b463ab88b222347de6dd1"
    ),
    "scripts/phase12_unified_admission.py": (
        "3cc0d8e9096de3ba528cfeae470915a098eb08aded1d3571fc9aa405dd636d59"
    ),
}


class Phase13FFeasibilityError(RuntimeError):
    """The corrected feasibility contract or its evidence failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13FFeasibilityError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise Phase13FFeasibilityError(f"JSON evidence is not an object: {path}")
    return value


def validate_historical_custody() -> list[dict[str, Any]]:
    """Prove both stopped campaigns and their receipts remain byte-exact."""

    records: list[dict[str, Any]] = []
    for authority in STOPPED_CAMPAIGNS:
        root = REPOSITORY_ROOT / authority["path"]
        artifact = validate_local_artifact(root, environ={})
        if artifact.root_sha256 != authority["root_sha256"]:
            raise Phase13FFeasibilityError("stopped Phase 13 root differs")
        checks = (
            (root / "checksums.sha256", authority["ledger_sha256"]),
            (REPOSITORY_ROOT / authority["report_path"], authority["report_sha256"]),
            (REPOSITORY_ROOT / authority["qc_path"], authority["qc_sha256"]),
            (
                REPOSITORY_ROOT / authority["receipt_path"],
                authority["receipt_sha256"],
            ),
        )
        for path, expected in checks:
            if sha256_file(path) != expected:
                raise Phase13FFeasibilityError(
                    f"stopped Phase 13 custody file differs: {path}"
                )
        receipt = _strict_json(REPOSITORY_ROOT / authority["receipt_path"])
        retrieval = receipt.get("clean_retrieval")
        if (
            not isinstance(retrieval, Mapping)
            or retrieval.get("result") != "PASS"
            or retrieval.get("root_sha256") != authority["root_sha256"]
        ):
            raise Phase13FFeasibilityError("stopped campaign R2 receipt differs")
        records.append(
            {
                "campaign_id": authority["campaign_id"],
                "local_path": authority["path"].as_posix(),
                "root_sha256": authority["root_sha256"],
                "checksum_ledger_sha256": authority["ledger_sha256"],
                "report_sha256": authority["report_sha256"],
                "qc_sha256": authority["qc_sha256"],
                "receipt_sha256": authority["receipt_sha256"],
                "clean_retrieval": "PASS",
                "changed": False,
            }
        )
    return records


def _source_authority() -> dict[str, Any]:
    for relative, expected in SOURCE_HASHES.items():
        if sha256_file(REPOSITORY_ROOT / relative) != expected:
            raise Phase13FFeasibilityError(
                f"prefix-construction authority differs: {relative}"
            )
    decision = REPOSITORY_ROOT / DECISION_PATH
    if not decision.is_file():
        raise Phase13FFeasibilityError("Decision 0031 is absent")
    return {
        "schema_version": "kvbench-phase13f-source-authority-1.0.0",
        "decision_id": "0031",
        "decision_path": DECISION_PATH.as_posix(),
        "decision_sha256": sha256_file(decision),
        "source_hashes": dict(SOURCE_HASHES),
        "adapter_or_cuda_source_changed": False,
    }


def recompute() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute and strictly classify every frozen planned record."""

    order = pilot.derive_execution_order()
    records = pilot.build_feasibility_records(order)
    if len(records) != pilot.PLANNED_RECORD_COUNT:
        raise Phase13FFeasibilityError("corrected record cardinality differs")
    status_counts = Counter(str(record["status"]) for record in records)
    point_statuses: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    point_required: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for record in records:
        key = (
            str(record["method_config_id"]),
            int(record["batch_size"]),
            int(record["context_label"]),
        )
        point_statuses[key].add(str(record["status"]))
        point_required[key].add(int(record["predicted_required_bytes"]))
        components = (
            int(record["model_weight_bytes"])
            + int(record["cache_allocated_bytes"])
            + int(record["endpoint_workspace_bytes"])
            + int(record["prefix_control_tensor_bytes"])
            + int(record["prefix_compute_peak_bytes"])
            + int(record["graph_pool_or_capture_reserve_bytes"])
        )
        if components != record["predicted_required_bytes"]:
            raise Phase13FFeasibilityError("feasibility component sum differs")
        if record["max_memory_fraction"] != 0.88:
            raise Phase13FFeasibilityError("frozen memory fraction differs")
    if any(len(value) != 1 for value in point_statuses.values()) or any(
        len(value) != 1 for value in point_required.values()
    ):
        raise Phase13FFeasibilityError("replicate feasibility is nondeterministic")
    if len(point_statuses) != 270:
        raise Phase13FFeasibilityError("unique feasibility point count differs")
    summary = {
        "schema_version": "kvbench-phase13f-feasibility-summary-1.0.0",
        "planned_records": len(records),
        "unique_points": len(point_statuses),
        "replicates_per_point": 3,
        "status_counts": dict(sorted(status_counts.items())),
        "unique_point_status_counts": dict(
            sorted(
                Counter(next(iter(value)) for value in point_statuses.values()).items()
            )
        ),
        "max_memory_fraction": pilot.MAX_MEMORY_FRACTION,
        "limit_bytes": int(records[0]["limit_bytes"]),
        "deterministic": True,
        "grid_changed": False,
        "pilot_executed": False,
    }
    return records, summary


def former_point_proof(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bind the corrected target classification to the immutable OOM trace."""

    target = [
        dict(record)
        for record in records
        if record["method_config_id"] == "tq_3bit_nc"
        and record["batch_size"] == 8
        and record["context_label"] == 98304
    ]
    if len(target) != 3 or any(
        record["status"] != "capacity_infeasible" for record in target
    ):
        raise Phase13FFeasibilityError("former failed point classification differs")
    stderr = REPOSITORY_ROOT / FAILED_RUN / "worker.stderr.txt"
    failure = REPOSITORY_ROOT / FAILED_RUN / "failure.json"
    if (
        sha256_file(stderr) != FAILED_STDERR_SHA256
        or sha256_file(failure) != FAILED_RECORD_SHA256
    ):
        raise Phase13FFeasibilityError("former failed point evidence differs")
    stderr_text = stderr.read_text(encoding="utf-8")
    required_markers = (
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.00 GiB.",
        "of which 5.27 GiB is free",
        "attention.o_proj(output)",
        "endpoint.prefill(prefix_input_ids)",
    )
    if any(marker not in stderr_text for marker in required_markers):
        raise Phase13FFeasibilityError("former failed point semantics differ")
    record = target[0]
    attempted = 8 * 98304 * pilot.PREFIX_HIDDEN_WIDTH * pilot.PREFIX_DTYPE_BYTES
    if attempted != 6 * 1024**3 or attempted != record["hidden_bf16_bytes"]:
        raise Phase13FFeasibilityError("output-projection allocation proof differs")
    old_records = _strict_json(
        REPOSITORY_ROOT
        / STOPPED_CAMPAIGNS[1]["path"]
        / "unified/feasibility.json"
    )["records"]
    old = [
        item
        for item in old_records
        if item["method_config_id"] == "tq_3bit_nc"
        and item["batch_size"] == 8
        and item["context_label"] == 98304
        and item["replicate_index"] == 0
    ]
    if len(old) != 1 or old[0]["status"] != "feasible":
        raise Phase13FFeasibilityError("former underestimation record differs")
    return {
        "schema_version": "kvbench-phase13f-former-point-proof-1.0.0",
        "configuration": "tq_3bit_nc",
        "batch_size": 8,
        "context_label": 98304,
        "replicate_count": 3,
        "old_status": "feasible",
        "old_predicted_required_bytes": old[0]["predicted_required_bytes"],
        "corrected_status": "capacity_infeasible",
        "corrected_predicted_required_bytes": record["predicted_required_bytes"],
        "limit_bytes": record["limit_bytes"],
        "attention_output_projection_allocation_bytes": attempted,
        "attention_output_projection_allocation_gib": 6.0,
        "attention_output_projection_peak_bytes": record[
            "attention_output_projection_peak_bytes"
        ],
        "mlp_peak_bytes": record["mlp_peak_bytes"],
        "failure_stage": "prefix_construction_before_graph_warmup_or_timing",
        "failure_stderr_sha256": FAILED_STDERR_SHA256,
        "failure_record_sha256": FAILED_RECORD_SHA256,
        "source_faithful": True,
        "pilot_rerun": False,
    }


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase13FFeasibilityError("Phase 13F evidence contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase13FFeasibilityError("Phase 13F evidence contains an unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            paths.append(path)
    return paths


def generate(root: Path, *, git_sha: str) -> dict[str, Any]:
    """Create one finalized bundle from inside the authorized container."""

    resolved = root.resolve(strict=True)
    if any(resolved.iterdir()):
        raise Phase13FFeasibilityError("Phase 13F output directory is not empty")
    if os.environ.get("KVBENCH_EXECUTION_ENVIRONMENT") != "measurement_container":
        raise Phase13FFeasibilityError("Phase 13F diagnostic is outside the container")
    if os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST") != AUTHORIZED_CONTAINER_DIGEST:
        raise Phase13FFeasibilityError("Phase 13F container digest differs")
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
        raise Phase13FFeasibilityError("Phase 13F source checkout differs")
    records, summary = recompute()
    custody = validate_historical_custody()
    source = _source_authority()
    proof = former_point_proof(records)
    write_exclusive(
        resolved / "feasibility.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13f-feasibility-matrix-1.0.0",
                "records": records,
            }
        ),
    )
    write_exclusive(resolved / "summary.json", json_bytes(summary))
    write_exclusive(resolved / "former-point-proof.json", json_bytes(proof))
    write_exclusive(
        resolved / "historical-custody.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13f-historical-custody-1.0.0",
                "campaigns": custody,
                "both_stopped_campaigns_unchanged": True,
            }
        ),
    )
    write_exclusive(resolved / "source-authority.json", json_bytes(source))
    manifest = {
        "schema_version": "kvbench-phase13f-artifact-manifest-1.0.0",
        "run_id": resolved.name,
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "source_git_sha": git_sha,
        "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
        "diagnostic_only": True,
        "pilot_executed": False,
        "planned_records": pilot.PLANNED_RECORD_COUNT,
        "complete_written_last": True,
        "append_only": True,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    write_exclusive(resolved / "manifest.json", json_bytes(manifest))
    inventory = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "role": "phase13f_feasibility_evidence",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _payload_paths(
            resolved, {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
        )
    ]
    write_exclusive(
        resolved / "artifact_inventory.json",
        json_bytes(
            {
                "schema_version": "kvbench-artifact-inventory-1.0.0",
                "run_id": resolved.name,
                "files": inventory,
                "excluded_control_files": [
                    "artifact_inventory.json",
                    "checksums.sha256",
                    "COMPLETE",
                ],
            }
        ),
    )
    ledger = "".join(
        f"{sha256_file(path)}  {path.relative_to(resolved).as_posix()}\n"
        for path in _payload_paths(resolved, {"checksums.sha256", "COMPLETE"})
    ).encode("utf-8")
    write_exclusive(resolved / "checksums.sha256", ledger)
    write_exclusive(
        resolved / "COMPLETE",
        json_bytes(
            {
                "schema_version": "kvbench-completion-1.0.0",
                "run_id": resolved.name,
                "status": "PASS",
                "manifest_sha256": sha256_file(resolved / "manifest.json"),
                "artifact_inventory_sha256": sha256_file(
                    resolved / "artifact_inventory.json"
                ),
                "checksum_ledger_path": "checksums.sha256",
                "checksum_ledger_sha256": sha256_file(
                    resolved / "checksums.sha256"
                ),
                "written_last": True,
            }
        ),
    )
    return validate(resolved)


def validate(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    artifact = validate_local_artifact(resolved, environ={})
    records, expected_summary = recompute()
    matrix = _strict_json(resolved / "feasibility.json")
    if matrix.get("records") != records:
        raise Phase13FFeasibilityError("sealed feasibility matrix differs")
    if _strict_json(resolved / "summary.json") != expected_summary:
        raise Phase13FFeasibilityError("sealed feasibility summary differs")
    expected_proof = former_point_proof(records)
    if _strict_json(resolved / "former-point-proof.json") != expected_proof:
        raise Phase13FFeasibilityError("sealed former-point proof differs")
    custody = _strict_json(resolved / "historical-custody.json")
    if custody.get("campaigns") != validate_historical_custody():
        raise Phase13FFeasibilityError("sealed historical custody differs")
    if _strict_json(resolved / "source-authority.json") != _source_authority():
        raise Phase13FFeasibilityError("sealed source authority differs")
    manifest = _strict_json(resolved / "manifest.json")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("authorized_container_digest") != AUTHORIZED_CONTAINER_DIGEST
        or manifest.get("diagnostic_only") is not True
        or manifest.get("pilot_executed") is not False
        or manifest.get("planned_records") != 810
        or manifest.get("r_hbm") is not None
    ):
        raise Phase13FFeasibilityError("Phase 13F manifest differs")
    return {
        "status": "PASS",
        "artifact_path": resolved.as_posix(),
        "root_sha256": artifact.root_sha256,
        "object_count": artifact.object_count,
        "planned_records": 810,
        "feasible_records": expected_summary["status_counts"]["feasible"],
        "capacity_infeasible_records": expected_summary["status_counts"][
            "capacity_infeasible"
        ],
        "former_point": "capacity_infeasible",
    }


def promote(stage: Path) -> Path:
    resolved = stage.resolve(strict=True)
    validate(resolved)
    if not CAMPAIGN_PATTERN.fullmatch(resolved.name):
        raise Phase13FFeasibilityError("Phase 13F evidence ID differs")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    final = ARTIFACT_ROOT / resolved.name
    if final.exists() or final.is_symlink():
        raise Phase13FFeasibilityError("Phase 13F evidence ID already exists")
    rename_noreplace(resolved, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    validate(final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--output", required=True, type=Path)
    generate_parser.add_argument("--git-sha", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("artifact", type=Path)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("stage", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "generate":
        result = generate(arguments.output, git_sha=arguments.git_sha)
    elif arguments.command == "validate":
        result = validate(arguments.artifact)
    else:
        result = {"status": "PASS", "artifact_path": promote(arguments.stage).as_posix()}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

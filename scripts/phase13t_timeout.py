#!/usr/bin/env python3
"""Seal and validate Phase 13T stage-supervision remediation evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Any

from kvbench.runtime.artifacts import sha256_file
from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from scripts import phase13_pilot as pilot
from scripts import phase13f_feasibility as phase13f
from scripts.r2_artifact import validate_local_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase13t"
DECISION_PATH = Path(
    "docs/decisions/0032-phase13-stage-aware-supervision.md"
)
PLAN_PATH = Path("docs/plans/phase13t-supervisor-timeout.md")
AUTHORIZED_CONTAINER_DIGEST = pilot.AUTHORIZED_CONTAINER_DIGEST
DIAGNOSTIC_SOURCE_SHA = "fb6a1fedc0cc01abbfbd974005f7567fbd291fd7"
ARTIFACT_PATTERN = re.compile(
    r"phase13t-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)
DIAGNOSTIC_FILES = frozenset(
    {
        "container-runtime.json",
        "kernel-path.after.normalized.dot",
        "kernel-path.after.raw.dot",
        "kernel-path.before.normalized.dot",
        "kernel-path.before.raw.dot",
        "probe.py",
        "stage-events.jsonl",
        "stderr.txt",
        "stdout.txt",
        "worker-result.json",
    }
)
REQUIRED_DIAGNOSTIC_FILES = frozenset(
    {
        "container-runtime.json",
        "kernel-path.after.normalized.dot",
        "kernel-path.after.raw.dot",
        "kernel-path.before.normalized.dot",
        "kernel-path.before.raw.dot",
        "probe.py",
        "stage-events.jsonl",
        "worker-result.json",
    }
)
PHASE13R2_AUTHORITY = {
    "campaign_id": "phase13-20260802t045837322693z-3127f1d1-486dcb",
    "path": Path(
        "artifacts/phase13/"
        "phase13-20260802t045837322693z-3127f1d1-486dcb"
    ),
    "root_sha256": (
        "581b02a6ca1a09c976a899b2b5d7eeb7897c0ad8f7ed8ad9fb11be5f6475f327"
    ),
    "ledger_sha256": (
        "9a25a98cc9083ba91ea5f14fc9bbcb0d7359aae3ac0d2843e9cd1b48f30c5dc9"
    ),
    "report_path": Path("docs/phase_reports/phase13r2-pilot-scan.md"),
    "report_sha256": (
        "5bfe1b892e97e3854e008915bcb55c6f735969736bbd40066645e13570bb9d43"
    ),
    "qc_path": Path("docs/evidence/phase13r2/pilot_qc.json"),
    "qc_sha256": (
        "fad8370aaab5426b14ee699a038f928f8558ed67eef9270ff5f7ff907358fbec"
    ),
    "receipt_path": Path("docs/evidence/phase13r2/r2-publication.json"),
    "receipt_sha256": (
        "6f39f30ed1e0491b71799ad8223507bd49cc48de3226c23bcf5a34f45dc29a31"
    ),
}
STOPPED_STAGING = {
    (
        "artifacts/phase13/.kvbench-staging/"
        "phase13-20260801t075729414590z-6f321e9a-22d46b.f9b5e77e.staging/"
        "campaign-reservation.json"
    ): "11d5e406dbebf4d54f5c5c450dfb03311ef491fd221400d8dc34a2410dec23e2",
    (
        "artifacts/phase13/.kvbench-staging/"
        "phase13-20260801t080147230025z-883cbf67-1f0f75.3c979b98.staging/"
        "campaign-reservation.json"
    ): "de05d87649336d4163283cb51e4e0c1e0a112221c73c7202b0cd7bdafa4237bc",
}
FAILED_RUN_FILES = {
    "failure.json": "7cbc7911300f4934da7b3875494efe0d7f1ce5b809e953bc3b2df5581997a98a",
    "manifest.json": "4ee7c00ea59156c16bbfe3c179fd9f6bea388034b3d9488d8c903937ff285b52",
    "started.json": "b851bee883f596e477886b2201befc154588acd004f6935ce49938defd732798",
    "worker.stderr.txt": "1138681b5042a58e0762da9fd90d6d418c8a42e3743e7934821baf83cf6376bc",
    "worker.stdout.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "worker.supervision.json": "859f5b680f24eb2117bd24bcf3a39c05363da287ac1e7c185174e35b35bcded3",
}


class Phase13TTimeoutError(RuntimeError):
    """The supervisor remediation or its evidence failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13TTimeoutError(f"invalid JSON evidence: {path}") from error
    if not isinstance(payload, dict):
        raise Phase13TTimeoutError(f"JSON evidence is not an object: {path}")
    return payload


def validate_historical_custody() -> list[dict[str, Any]]:
    """Prove all stopped completed and staging campaigns remain unchanged."""

    records = phase13f.validate_historical_custody()
    authority = PHASE13R2_AUTHORITY
    root = REPOSITORY_ROOT / authority["path"]
    artifact = validate_local_artifact(root, environ={})
    if artifact.root_sha256 != authority["root_sha256"]:
        raise Phase13TTimeoutError("stopped Phase 13R2 root differs")
    for path, expected in (
        (root / "checksums.sha256", authority["ledger_sha256"]),
        (REPOSITORY_ROOT / authority["report_path"], authority["report_sha256"]),
        (REPOSITORY_ROOT / authority["qc_path"], authority["qc_sha256"]),
        (REPOSITORY_ROOT / authority["receipt_path"], authority["receipt_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise Phase13TTimeoutError(f"stopped Phase 13R2 custody differs: {path}")
    receipt = _strict_json(REPOSITORY_ROOT / authority["receipt_path"])
    retrieval = receipt.get("clean_retrieval")
    if (
        not isinstance(retrieval, Mapping)
        or retrieval.get("result") != "PASS"
        or retrieval.get("root_sha256") != authority["root_sha256"]
    ):
        raise Phase13TTimeoutError("stopped Phase 13R2 receipt differs")
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
    for relative, expected in STOPPED_STAGING.items():
        path = REPOSITORY_ROOT / relative
        if sha256_file(path) != expected:
            raise Phase13TTimeoutError("stopped Phase 13 staging evidence differs")
        records.append(
            {
                "campaign_id": path.parent.name,
                "local_path": path.parent.relative_to(REPOSITORY_ROOT).as_posix(),
                "reservation_sha256": expected,
                "finalized": False,
                "changed": False,
            }
        )
    failed_root = (
        root
        / "runs"
        / (
            "phase13-20260802t045837322693z-3127f1d1-486dcb-r0-o027-"
            "kvq2-b1-l131072"
        )
    )
    for name, expected in FAILED_RUN_FILES.items():
        if sha256_file(failed_root / name) != expected:
            raise Phase13TTimeoutError("immutable timeout-run evidence differs")
    supervision = _strict_json(failed_root / "worker.supervision.json")
    if (
        supervision.get("timeout", {}).get("timeout_seconds") != 7200.0
        or supervision.get("timeout", {}).get("timed_out") is not True
        or supervision.get("returncode") != -15
    ):
        raise Phase13TTimeoutError("immutable timeout semantics differ")
    return records


def _diagnostic_events(root: Path) -> list[dict[str, Any]]:
    path = root / "stage-events.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise Phase13TTimeoutError("diagnostic stage log is unreadable") from error
    events: list[dict[str, Any]] = []
    last_elapsed = -1.0
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise Phase13TTimeoutError("diagnostic stage log is malformed") from error
        if (
            not isinstance(event, dict)
            or event.get("schema_version")
            != "kvbench-phase13t-stage-event-1.0.0"
            or not isinstance(event.get("stage"), str)
            or event.get("event")
            not in {"started", "completed", "attention_completed"}
            or not isinstance(event.get("elapsed_seconds"), (int, float))
            or float(event["elapsed_seconds"]) < last_elapsed
        ):
            raise Phase13TTimeoutError("diagnostic stage event differs")
        last_elapsed = float(event["elapsed_seconds"])
        events.append(event)
    if not events:
        raise Phase13TTimeoutError("diagnostic stage log is empty")
    return events


def _unique_duration(
    events: Sequence[Mapping[str, Any]], stage: str
) -> dict[str, float]:
    starts = [
        float(event["elapsed_seconds"])
        for event in events
        if event.get("stage") == stage and event.get("event") == "started"
    ]
    completions = [
        float(event["elapsed_seconds"])
        for event in events
        if event.get("stage") == stage and event.get("event") == "completed"
    ]
    if len(starts) != 1 or len(completions) != 1 or completions[0] <= starts[0]:
        raise Phase13TTimeoutError(f"diagnostic {stage} boundaries differ")
    return {
        "started_elapsed_seconds": starts[0],
        "completed_elapsed_seconds": completions[0],
        "duration_seconds": completions[0] - starts[0],
    }


def _boundary_span(
    events: Sequence[Mapping[str, Any]],
    *,
    start_stage: str,
    start_event: str,
    end_stage: str,
    end_event: str,
) -> dict[str, float]:
    starts = [
        float(event["elapsed_seconds"])
        for event in events
        if event.get("stage") == start_stage
        and event.get("event") == start_event
    ]
    completions = [
        float(event["elapsed_seconds"])
        for event in events
        if event.get("stage") == end_stage
        and event.get("event") == end_event
    ]
    if len(starts) != 1 or len(completions) != 1 or completions[0] <= starts[0]:
        raise Phase13TTimeoutError("diagnostic supervisor-stage boundaries differ")
    return {
        "started_elapsed_seconds": starts[0],
        "completed_elapsed_seconds": completions[0],
        "duration_seconds": completions[0] - starts[0],
    }


def diagnostic_summary(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    names = frozenset(path.name for path in resolved.iterdir() if path.is_file())
    if (
        not REQUIRED_DIAGNOSTIC_FILES.issubset(names)
        or not names.issubset(DIAGNOSTIC_FILES)
        or any(path.is_symlink() for path in resolved.iterdir())
    ):
        raise Phase13TTimeoutError("diagnostic file set differs")
    runtime = _strict_json(resolved / "container-runtime.json")
    if (
        runtime.get("authorized_container_digest") != AUTHORIZED_CONTAINER_DIGEST
        or runtime.get("diagnostic_source_git_sha") != DIAGNOSTIC_SOURCE_SHA
        or runtime.get("native_host_cuda_execution") is not False
    ):
        raise Phase13TTimeoutError("diagnostic execution authority differs")
    result = _strict_json(resolved / "worker-result.json")
    if (
        result.get("method_config_id") != "kvq2"
        or result.get("batch_size") != 1
        or result.get("context_label") != 131072
        or result.get("historical_context") != 131071
        or result.get("execution_git_sha") != DIAGNOSTIC_SOURCE_SHA
        or result.get("authorized_container_digest") != AUTHORIZED_CONTAINER_DIGEST
        or result.get("finite_output") is not True
        or result.get("no_backend_fallback") is not True
        or result.get("allocation_stable") is not True
        or result.get("r_hbm") is not None
    ):
        raise Phase13TTimeoutError("diagnostic worker result differs")
    events = _diagnostic_events(resolved)
    if any(event.get("event") == "failed" for event in events):
        raise Phase13TTimeoutError("diagnostic reports a failed stage")
    stages = {
        name: _unique_duration(events, name)
        for name in (
            "model_load",
            "prefix_construction",
            "graph_capture",
            "pilot_warmup",
            "measurement",
            "finalization",
        )
    }
    supervised_stages = {
        "model_load": stages["model_load"],
        "prefix_construction": _boundary_span(
            events,
            start_stage="prefix_construction",
            start_event="started",
            end_stage="graph_capture",
            end_event="started",
        ),
        "graph_capture": _boundary_span(
            events,
            start_stage="graph_capture",
            start_event="started",
            end_stage="session_setup",
            end_event="completed",
        ),
        "warmup_and_audit": _boundary_span(
            events,
            start_stage="session_setup",
            start_event="completed",
            end_stage="measurement",
            end_event="started",
        ),
        "measurement": stages["measurement"],
        "finalization": stages["finalization"],
    }
    layer_starts = [
        event for event in events
        if event.get("stage") == "prefix_layer"
        and event.get("event") == "started"
    ]
    layer_completions = [
        event for event in events
        if event.get("stage") == "prefix_layer"
        and event.get("event") == "attention_completed"
    ]
    if (
        [event.get("layer") for event in layer_starts] != list(range(32))
        or [event.get("layer") for event in layer_completions] != list(range(32))
    ):
        raise Phase13TTimeoutError("diagnostic layer progression differs")
    layer_durations = [
        float(completed["elapsed_seconds"]) - float(started["elapsed_seconds"])
        for started, completed in zip(layer_starts, layer_completions, strict=True)
    ]
    prefix_duration = supervised_stages["prefix_construction"][
        "duration_seconds"
    ]
    frozen_budget = pilot.prefix_construction_timeout_seconds(
        batch=1,
        historical=131071,
    )
    frozen_contract = pilot.stage_timeout_contract(
        batch=1,
        historical=131071,
    )
    if (
        prefix_duration <= 7200.0
        or prefix_duration >= frozen_budget
        or any(duration <= 0 or duration >= 1800 for duration in layer_durations)
    ):
        raise Phase13TTimeoutError("diagnostic progression or bound differs")
    for stage, boundary in supervised_stages.items():
        if boundary["duration_seconds"] >= frozen_contract[stage]:
            raise Phase13TTimeoutError(
                f"diagnostic exceeded the Decision 0032 {stage} bound"
            )
    return {
        "schema_version": "kvbench-phase13t-diagnostic-summary-1.0.0",
        "configuration": "kvq2",
        "batch_size": 1,
        "context_label": 131072,
        "historical_context": 131071,
        "old_global_timeout_seconds": 7200.0,
        "source_observed_stages": stages,
        "supervisor_stages": supervised_stages,
        "prefix_layers_completed": 32,
        "prefix_layer_duration_min_seconds": min(layer_durations),
        "prefix_layer_duration_max_seconds": max(layer_durations),
        "prefix_layer_duration_mean_seconds": sum(layer_durations) / 32,
        "decision_0032_prefix_budget_seconds": frozen_budget,
        "decision_0032_stage_contract_seconds": frozen_contract,
        "old_timeout_expired_during": "prefix_construction",
        "normal_forward_progress": True,
        "stalled": False,
        "deadlocked": False,
        "worker_completed": True,
        "output_finite": True,
        "backend_fallback": False,
        "timing_claim": False,
        "performance_claim_eligible": False,
        "raw_stage_log_sha256": sha256_file(resolved / "stage-events.jsonl"),
        "worker_result_sha256": sha256_file(resolved / "worker-result.json"),
    }


def timeout_matrix() -> dict[str, Any]:
    order = pilot.derive_execution_order()
    feasibility = pilot.build_feasibility_records(order)
    records: list[dict[str, Any]] = []
    for record in feasibility:
        launched = record["status"] == "feasible"
        records.append(
            {
                "method_config_id": record["method_config_id"],
                "method_config_fingerprint": record[
                    "method_config_fingerprint"
                ],
                "batch_size": record["batch_size"],
                "context_label": record["context_label"],
                "historical_context": record["historical_context"],
                "replicate_index": record["replicate_index"],
                "order_index": record["order_index"],
                "feasibility_status": record["status"],
                "stage_timeouts_seconds": (
                    pilot.stage_timeout_contract(
                        batch=int(record["batch_size"]),
                        historical=int(record["historical_context"]),
                    )
                    if launched
                    else None
                ),
            }
        )
    counts = Counter(record["feasibility_status"] for record in records)
    if len(records) != 810 or counts != {"feasible": 684, "capacity_infeasible": 126}:
        raise Phase13TTimeoutError("frozen 810-record classification differs")
    return {
        "schema_version": "kvbench-phase13t-timeout-matrix-1.0.0",
        "records": records,
        "planned_records": 810,
        "feasible_records": 684,
        "capacity_infeasible_records": 126,
        "grid_changed": False,
        "timing_boundaries_changed": False,
        "heartbeat_extends_deadline": False,
    }


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase13TTimeoutError("Phase 13T evidence contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase13TTimeoutError("Phase 13T evidence contains an unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            paths.append(path)
    return paths


def _source_authority() -> dict[str, Any]:
    paths = (
        DECISION_PATH,
        PLAN_PATH,
        Path("scripts/phase13_pilot.py"),
        Path("scripts/phase13t_timeout.py"),
        Path("scripts/validate_phase2.py"),
        Path("src/kvbench/runtime/process_supervision.py"),
        Path("tests/unit/test_phase13t_timeout.py"),
    )
    return {
        "schema_version": "kvbench-phase13t-source-authority-1.0.0",
        "decision_id": "0032",
        "source_hashes": {
            path.as_posix(): sha256_file(REPOSITORY_ROOT / path) for path in paths
        },
        "adapters_changed": False,
        "cuda_changed": False,
        "grid_changed": False,
        "timing_boundaries_changed": False,
    }


def generate(root: Path, *, diagnostic: Path, git_sha: str) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    diagnostic_root = diagnostic.resolve(strict=True)
    if any(resolved.iterdir()):
        raise Phase13TTimeoutError("Phase 13T output directory is not empty")
    if (
        os.environ.get("KVBENCH_EXECUTION_ENVIRONMENT") != "measurement_container"
        or os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST")
        != AUTHORIZED_CONTAINER_DIGEST
    ):
        raise Phase13TTimeoutError("Phase 13T validation is outside the container")
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
        raise Phase13TTimeoutError("Phase 13T source checkout differs")
    summary = diagnostic_summary(diagnostic_root)
    matrix = timeout_matrix()
    custody = validate_historical_custody()
    source = _source_authority()
    copied = resolved / "diagnostic"
    copied.mkdir()
    for name in sorted(path.name for path in diagnostic_root.iterdir() if path.is_file()):
        shutil.copyfile(diagnostic_root / name, copied / name)
    write_exclusive(resolved / "diagnostic-summary.json", json_bytes(summary))
    write_exclusive(resolved / "timeout-matrix.json", json_bytes(matrix))
    write_exclusive(
        resolved / "historical-custody.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13t-historical-custody-1.0.0",
                "campaigns": custody,
                "all_stopped_campaigns_unchanged": True,
            }
        ),
    )
    write_exclusive(resolved / "source-authority.json", json_bytes(source))
    write_exclusive(
        resolved / "manifest.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13t-manifest-1.0.0",
                "run_id": resolved.name,
                "status": "PASS",
                "created_at_utc": _utc_now(),
                "source_git_sha": git_sha,
                "authorized_container_digest": AUTHORIZED_CONTAINER_DIGEST,
                "diagnostic_source_git_sha": DIAGNOSTIC_SOURCE_SHA,
                "diagnostic_only": True,
                "pilot_executed": False,
                "complete_written_last": True,
                "append_only": True,
                "full_scan": "CLOSED",
                "quality_execution": "LOCKED",
                "performance_claim_eligible": False,
                "r_hbm": None,
            }
        ),
    )
    inventory = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "role": "phase13t_timeout_remediation",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _payload_paths(
            resolved,
            {"artifact_inventory.json", "checksums.sha256", "COMPLETE"},
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
    for path in sorted(resolved.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    resolved.chmod(0o555)
    return validate(resolved)


def validate(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    artifact = validate_local_artifact(resolved, environ={})
    summary = diagnostic_summary(resolved / "diagnostic")
    if _strict_json(resolved / "diagnostic-summary.json") != summary:
        raise Phase13TTimeoutError("sealed diagnostic summary differs")
    if _strict_json(resolved / "timeout-matrix.json") != timeout_matrix():
        raise Phase13TTimeoutError("sealed timeout matrix differs")
    custody = _strict_json(resolved / "historical-custody.json")
    if custody.get("campaigns") != validate_historical_custody():
        raise Phase13TTimeoutError("sealed historical custody differs")
    source = _strict_json(resolved / "source-authority.json")
    expected_source = _source_authority()
    if source != expected_source:
        raise Phase13TTimeoutError("sealed source authority differs")
    manifest = _strict_json(resolved / "manifest.json")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("authorized_container_digest") != AUTHORIZED_CONTAINER_DIGEST
        or manifest.get("diagnostic_only") is not True
        or manifest.get("pilot_executed") is not False
        or manifest.get("r_hbm") is not None
    ):
        raise Phase13TTimeoutError("Phase 13T manifest differs")
    return {
        "status": "PASS",
        "artifact_path": resolved.as_posix(),
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "diagnostic": summary,
        "planned_records": 810,
        "feasible_records": 684,
        "capacity_infeasible_records": 126,
    }


def promote(stage: Path) -> Path:
    resolved = stage.resolve(strict=True)
    validate(resolved)
    if not ARTIFACT_PATTERN.fullmatch(resolved.name):
        raise Phase13TTimeoutError("Phase 13T evidence ID differs")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    final = ARTIFACT_ROOT / resolved.name
    if final.exists() or final.is_symlink():
        raise Phase13TTimeoutError("Phase 13T evidence ID already exists")
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
    generate_parser.add_argument("--diagnostic", required=True, type=Path)
    generate_parser.add_argument("--git-sha", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("artifact", type=Path)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("stage", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "generate":
        result = generate(
            arguments.output,
            diagnostic=arguments.diagnostic,
            git_sha=arguments.git_sha,
        )
    elif arguments.command == "validate":
        result = validate(arguments.artifact)
    else:
        result = {
            "status": "PASS",
            "artifact_path": promote(arguments.stage).as_posix(),
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

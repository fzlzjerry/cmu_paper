#!/usr/bin/env python3
"""Run, seal, and validate Phase 13P-C CUDA-context isolation evidence."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from typing import Any

from kvbench.runtime.artifacts import sha256_file
from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from scripts import phase13_pilot as pilot
from scripts.phase13pb_prefix_remediation import validate_equivalence
from scripts.r2_artifact import validate_local_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase13pc"
DECISION_PATH = Path(
    "docs/decisions/0035-phase13-equivalence-cuda-context-isolation.md"
)
SOURCE_PATHS = (
    DECISION_PATH,
    Path("scripts/phase13_pilot.py"),
    Path("scripts/phase13pc_cuda_context_isolation.py"),
)
EVIDENCE_ID_PATTERN = re.compile(
    r"phase13pc-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)


class Phase13PCIsolationError(RuntimeError):
    """The CUDA-context isolation remediation failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13PCIsolationError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise Phase13PCIsolationError("JSON evidence is not an object")
    return value


def new_evidence_id(git_sha: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise Phase13PCIsolationError("execution Git SHA differs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:21]
    return f"phase13pc-{stamp}z-{git_sha[:8]}-{secrets.token_hex(3)}"


def _snapshot_is_idle(snapshot: Any) -> bool:
    return bool(
        isinstance(snapshot, dict)
        and snapshot.get("query_exit_code") == 0
        and snapshot.get("errors") == []
        and snapshot.get("allowed_compute_processes") == []
        and snapshot.get("foreign_compute_processes") == []
        and snapshot.get("unknown_processes") == []
    )


def validate_handoff(
    handoff: Mapping[str, Any],
    *,
    equivalence_path: Path,
) -> dict[str, Any]:
    equivalence = _strict_json(equivalence_path)
    matrix = validate_equivalence(equivalence)
    git_sha = matrix["execution_git_sha"]
    if (
        handoff.get("schema_version")
        != "kvbench-phase13-prefix-equivalence-handoff-1.0.0"
        or handoff.get("status") != "PASS"
        or handoff.get("decision_id") != "0035"
        or handoff.get("execution_git_sha") != git_sha
        or handoff.get("authorized_container_digest")
        != pilot.AUTHORIZED_CONTAINER_DIGEST
        or handoff.get("dedicated_child_process") is not True
        or handoff.get("parent_loaded_model") is not False
        or handoff.get("parent_initialized_cuda") is not False
        or handoff.get("child_exit_code") != 0
        or handoff.get("child_exit_passed") is not True
        or handoff.get("child_evidence_validated") is not True
        or handoff.get("equivalence_case_count") != 30
        or handoff.get("equivalence_cross_batch_rejection_count") != 20
        or handoff.get("equivalence_sha256") != sha256_file(equivalence_path)
        or handoff.get("pre_idle_gpu") is not True
        or handoff.get("post_idle_gpu") is not True
        or handoff.get("remaining_cuda_processes") != []
        or handoff.get("foreign_cuda_processes") != []
        or handoff.get("runtime_prefix_cache_sharing") is not False
        or handoff.get("timing_collected") is not False
        or not _snapshot_is_idle(handoff.get("pre_snapshot"))
        or not _snapshot_is_idle(handoff.get("post_snapshot"))
    ):
        raise Phase13PCIsolationError("CUDA-context isolation handoff differs")
    for field in ("child_stdout_sha256", "child_stderr_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(handoff.get(field))) is None:
            raise Phase13PCIsolationError("child stream digest differs")
    return {
        "status": "PASS",
        "execution_git_sha": git_sha,
        "case_count": matrix["case_count"],
        "cross_batch_rejection_count": matrix["cross_batch_rejection_count"],
        "pre_idle_gpu": True,
        "post_idle_gpu": True,
    }


def _source_authority(git_sha: str) -> dict[str, Any]:
    decision = REPOSITORY_ROOT / DECISION_PATH
    if not decision.is_file() or "- Status: Accepted" not in decision.read_text(
        encoding="utf-8"
    ):
        raise Phase13PCIsolationError("Decision 0035 authority differs")
    return {
        "schema_version": "kvbench-phase13pc-source-authority-1.0.0",
        "execution_git_sha": git_sha,
        "decision_id": "0035",
        "decision_path": DECISION_PATH.as_posix(),
        "decision_sha256": sha256_file(decision),
        "source_hashes": {
            relative.as_posix(): sha256_file(REPOSITORY_ROOT / relative)
            for relative in SOURCE_PATHS
        },
        "authorized_container_digest": pilot.AUTHORIZED_CONTAINER_DIGEST,
        "adapter_changed": False,
        "cuda_changed": False,
        "cache_layout_changed": False,
        "grid_changed": False,
        "timing_boundary_changed": False,
    }


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase13PCIsolationError("Phase 13P-C evidence contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase13PCIsolationError("Phase 13P-C evidence contains unsafe file")
        if path.relative_to(root).as_posix() not in excluded:
            files.append(path)
    return files


def run(root: Path, *, scratch_root: Path, git_sha: str) -> dict[str, Any]:
    """Execute the isolated handoff and seal its focused evidence."""

    resolved = root.resolve(strict=True)
    scratch = scratch_root.resolve(strict=True)
    if any(resolved.iterdir()) or any(scratch.iterdir()):
        raise Phase13PCIsolationError("Phase 13P-C output roots must be empty")
    if os.environ.get("KVBENCH_EXECUTION_ENVIRONMENT") != "measurement_container":
        raise Phase13PCIsolationError("Phase 13P-C ran outside the container")
    if (
        os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST")
        != pilot.AUTHORIZED_CONTAINER_DIGEST
    ):
        raise Phase13PCIsolationError("Phase 13P-C container digest differs")
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
        raise Phase13PCIsolationError("Phase 13P-C source checkout differs")
    equivalence_path = resolved / "prefix-equivalence.json"
    handoff_path = resolved / "isolation-handoff.json"
    pilot.run_prefix_equivalence_isolated(
        output=equivalence_path,
        scratch_root=scratch,
        handoff_output=handoff_path,
        git_sha=git_sha,
    )
    result = validate_handoff(
        _strict_json(handoff_path), equivalence_path=equivalence_path
    )
    write_exclusive(
        resolved / "source-authority.json", json_bytes(_source_authority(git_sha))
    )
    write_exclusive(
        resolved / "preservation.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13pc-preservation-1.0.0",
                "stopped_campaigns_changed": False,
                "stopped_campaigns_resumed": False,
                "stopped_campaigns_published": False,
                "stopped_timing_data_read_or_analyzed": False,
                "pilot_executed": False,
                "timing_collected": False,
            }
        ),
    )
    write_exclusive(
        resolved / "manifest.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13pc-artifact-manifest-1.0.0",
                "run_id": resolved.name,
                "status": "PASS",
                "created_at_utc": _utc_now(),
                "source_git_sha": git_sha,
                "authorized_container_digest": pilot.AUTHORIZED_CONTAINER_DIGEST,
                "decision_id": "0035",
                "case_count": 30,
                "dedicated_child_process": True,
                "parent_initialized_cuda": False,
                "pre_idle_gpu": True,
                "post_idle_gpu": True,
                "diagnostic_only": True,
                "pilot_executed": False,
                "timing_collected": False,
                "performance_claim_eligible": False,
                "r_hbm": None,
                "complete_written_last": True,
                "append_only": True,
            }
        ),
    )
    inventory = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "role": "phase13pc_cuda_context_isolation_evidence",
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
    for path in sorted(resolved.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    resolved.chmod(0o555)
    return {**validate(resolved), **result}


def validate(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    artifact = validate_local_artifact(resolved, environ={})
    equivalence_path = resolved / "prefix-equivalence.json"
    result = validate_handoff(
        _strict_json(resolved / "isolation-handoff.json"),
        equivalence_path=equivalence_path,
    )
    git_sha = result["execution_git_sha"]
    if _strict_json(resolved / "source-authority.json") != _source_authority(
        git_sha
    ):
        raise Phase13PCIsolationError("sealed source authority differs")
    preservation = _strict_json(resolved / "preservation.json")
    manifest = _strict_json(resolved / "manifest.json")
    if (
        any(
            preservation.get(field) is not False
            for field in (
                "stopped_campaigns_changed",
                "stopped_campaigns_resumed",
                "stopped_campaigns_published",
                "stopped_timing_data_read_or_analyzed",
                "pilot_executed",
                "timing_collected",
            )
        )
        or manifest.get("status") != "PASS"
        or manifest.get("source_git_sha") != git_sha
        or manifest.get("decision_id") != "0035"
        or manifest.get("dedicated_child_process") is not True
        or manifest.get("parent_initialized_cuda") is not False
        or manifest.get("pre_idle_gpu") is not True
        or manifest.get("post_idle_gpu") is not True
        or manifest.get("pilot_executed") is not False
        or manifest.get("timing_collected") is not False
        or manifest.get("r_hbm") is not None
    ):
        raise Phase13PCIsolationError("Phase 13P-C sealed evidence differs")
    return {
        "status": "PASS",
        "artifact_path": resolved.as_posix(),
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        **result,
    }


def promote(stage: Path) -> Path:
    resolved = stage.resolve(strict=True)
    validate(resolved)
    if EVIDENCE_ID_PATTERN.fullmatch(resolved.name) is None:
        raise Phase13PCIsolationError("Phase 13P-C evidence ID differs")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    final = ARTIFACT_ROOT / resolved.name
    if final.exists() or final.is_symlink():
        raise Phase13PCIsolationError("Phase 13P-C evidence ID already exists")
    rename_noreplace(resolved, final)
    validate(final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    new_id = subparsers.add_parser("new-id")
    new_id.add_argument("--git-sha", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", required=True, type=Path)
    run_parser.add_argument("--scratch-root", required=True, type=Path)
    run_parser.add_argument("--git-sha", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("artifact", type=Path)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("stage", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "new-id":
        print(new_evidence_id(arguments.git_sha))
        return 0
    if arguments.command == "run":
        print(
            json.dumps(
                run(
                    arguments.output,
                    scratch_root=arguments.scratch_root,
                    git_sha=arguments.git_sha,
                ),
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "validate":
        print(json.dumps(validate(arguments.artifact), sort_keys=True))
        return 0
    if arguments.command == "promote":
        print(
            json.dumps(
                {"artifact_path": promote(arguments.stage).as_posix()},
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

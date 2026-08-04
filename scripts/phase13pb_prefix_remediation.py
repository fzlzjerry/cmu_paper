#!/usr/bin/env python3
"""Seal and validate focused Phase 13P-B batch-exact prefix evidence."""

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
from scripts.r2_artifact import validate_local_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "phase13pb"
DECISION_PATH = Path(
    "docs/decisions/0034-phase13-batch-exact-prefix-state.md"
)
SOURCE_PATHS = (
    DECISION_PATH,
    Path("scripts/phase13_prefix_state.py"),
    Path("scripts/phase13_pilot.py"),
)
CAMPAIGN_PATTERN = re.compile(
    r"phase13pb-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)
EXPECTED_CASES = {
    (configuration, batch)
    for configuration in pilot.CONFIGURATIONS
    for batch in pilot.BATCH_SIZES
}
EXPECTED_CROSS_BATCH_REJECTIONS = {
    (configuration, 1, batch)
    for configuration in pilot.CONFIGURATIONS
    for batch in (4, 8)
}


class Phase13PBBatchExactError(RuntimeError):
    """The exact-batch prefix remediation or its evidence failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase13PBBatchExactError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise Phase13PBBatchExactError("JSON evidence is not an object")
    return value


def _require_sha256(value: Any, message: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise Phase13PBBatchExactError(message)
    return value


def new_evidence_id(git_sha: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise Phase13PBBatchExactError("execution Git SHA differs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")[:21]
    return f"phase13pb-{stamp}z-{git_sha[:8]}-{secrets.token_hex(3)}"


def validate_equivalence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete 30-case same-batch matrix and negative controls."""

    records = payload.get("records")
    rejections = payload.get("cross_batch_rejections")
    if (
        payload.get("schema_version")
        != "kvbench-phase13-prefix-equivalence-2.0.0"
        or payload.get("status") != "PASS"
        or payload.get("decision_id") != "0034"
        or payload.get("authorized_container_digest")
        != pilot.AUTHORIZED_CONTAINER_DIGEST
        or payload.get("configuration_count") != len(pilot.CONFIGURATIONS)
        or payload.get("batch_sizes") != list(pilot.BATCH_SIZES)
        or payload.get("historical_context") != 17
        or payload.get("case_count") != 30
        or payload.get("all_configurations_passed") is not True
        or payload.get("all_cases_passed") is not True
        or payload.get("cross_batch_restore_fail_closed") is not True
        or payload.get("cross_batch_rejection_count") != 20
        or payload.get("fresh_target_allocation") is not True
        or payload.get("restore_outside_timing") is not True
        or payload.get("runtime_prefix_cache_sharing") is not False
        or payload.get("timing_collected") is not False
        or not isinstance(records, list)
        or len(records) != 30
        or not isinstance(rejections, list)
        or len(rejections) != 20
    ):
        raise Phase13PBBatchExactError("prefix equivalence header differs")
    git_sha = str(payload.get("execution_git_sha"))
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise Phase13PBBatchExactError("prefix equivalence Git SHA differs")

    exact_fields = (
        "cache_layout_fingerprint",
        "cache_accounting",
        "cache_byte_breakdown",
        "pointer_labels",
        "pointer_count",
        "pointers_stable",
        "pointers_unique",
        "raw_pointer_values_unique",
        "nonallocated_null_pointer_labels",
        "null_pointer_tensor_contract_verified",
        "recognized_same_tensor_alias_groups",
        "unexpected_pointer_alias_groups",
        "output_checksum",
        "kernel_path_fingerprint",
        "kernel_count",
        "graph_capture",
        "graph_fallback",
        "graph_replay_exact",
        "eager_graph_agreement",
    )
    observed_cases: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise Phase13PBBatchExactError("prefix equivalence record differs")
        configuration = str(record.get("configuration"))
        batch = record.get("target_batch")
        key = (configuration, batch)
        direct = record.get("direct")
        restored = record.get("restored")
        state_sha256 = _require_sha256(
            record.get("source_state_sha256"),
            "prefix state SHA differs",
        )
        if (
            key in observed_cases
            or key not in EXPECTED_CASES
            or record.get("case_id") != f"{configuration}/B{batch}/L17"
            or record.get("method_config_fingerprint")
            != pilot.CONFIG_FINGERPRINTS.get(configuration)
            or record.get("source_batch") != batch
            or record.get("historical_context") != 17
            or record.get("batch_reuse_policy") != "exact_target_batch_only"
            or record.get("passed") is not True
            or any(
                record.get(field) is not True
                for field in (
                    "state_bytes_exact",
                    "lifecycle_exact",
                    "layout_exact",
                    "allocation_exact",
                    "output_checksum_exact",
                    "kernel_path_exact",
                    "cuda_graph_result_exact",
                    "fresh_target_allocation",
                    "direct_and_restored_pointers_disjoint",
                    "restore_outside_timing",
                )
            )
            or record.get("runtime_prefix_cache_sharing") is not False
            or state_sha256 != record.get("target_state_sha256")
            or state_sha256 != record.get("restored_state_sha256")
            or not isinstance(direct, dict)
            or not isinstance(restored, dict)
            or any(direct.get(field) != restored.get(field) for field in exact_fields)
            or {
                pointer
                for pointer in direct.get("pointer_values", [])
                if isinstance(pointer, int) and pointer > 0
            }.intersection(
                pointer
                for pointer in restored.get("pointer_values", [])
                if isinstance(pointer, int) and pointer > 0
            )
        ):
            raise Phase13PBBatchExactError(
                f"prefix equivalence case differs: {configuration}/B{batch}"
            )
        for session in (direct, restored):
            expected_null_labels = sorted(
                pilot._EXPECTED_EQUIVALENCE_NONALLOCATED_NULL_POINTER_LABELS[
                    configuration
                ]
            )
            pointer_values = session.get("pointer_values")
            if (
                session.get("pointers_stable") is not True
                or session.get("pointers_unique") is not True
                or session.get("graph_capture") is not True
                or session.get("graph_fallback") is not False
                or session.get("graph_replay_exact") is not True
                or session.get("eager_graph_agreement") is not True
                or session.get("null_pointer_tensor_contract_verified") is not True
                or session.get("nonallocated_null_pointer_labels")
                != expected_null_labels
                or not isinstance(pointer_values, list)
                or any(
                    not isinstance(pointer, int)
                    or isinstance(pointer, bool)
                    or pointer < 0
                    for pointer in pointer_values
                )
                or sum(pointer == 0 for pointer in pointer_values)
                != len(expected_null_labels)
                or session.get("unexpected_pointer_alias_groups") != []
            ):
                raise Phase13PBBatchExactError(
                    "prefix equivalence session invariant differs"
                )
        observed_cases.add(key)
    if observed_cases != EXPECTED_CASES:
        raise Phase13PBBatchExactError("prefix equivalence case set differs")

    observed_rejections: set[tuple[str, int, int]] = set()
    for record in rejections:
        if not isinstance(record, dict):
            raise Phase13PBBatchExactError("cross-batch rejection record differs")
        key = (
            str(record.get("configuration")),
            int(record.get("source_batch", -1)),
            int(record.get("target_batch", -1)),
        )
        if (
            key in observed_rejections
            or key not in EXPECTED_CROSS_BATCH_REJECTIONS
            or record.get("historical_context") != 17
            or record.get("rejected_before_cache_mutation") is not True
            or record.get("pointers_unchanged") is not True
            or record.get("error") != "prefix state exact batch differs"
        ):
            raise Phase13PBBatchExactError(
                "cross-batch restoration rejection differs"
            )
        observed_rejections.add(key)
    if observed_rejections != EXPECTED_CROSS_BATCH_REJECTIONS:
        raise Phase13PBBatchExactError("cross-batch rejection set differs")
    return {
        "status": "PASS",
        "execution_git_sha": git_sha,
        "case_count": len(observed_cases),
        "cross_batch_rejection_count": len(observed_rejections),
    }


def _source_authority(git_sha: str) -> dict[str, Any]:
    decision = REPOSITORY_ROOT / DECISION_PATH
    if not decision.is_file() or "- Status: Accepted" not in decision.read_text(
        encoding="utf-8"
    ):
        raise Phase13PBBatchExactError("Decision 0034 authority differs")
    return {
        "schema_version": "kvbench-phase13pb-source-authority-1.0.0",
        "execution_git_sha": git_sha,
        "decision_id": "0034",
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
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise Phase13PBBatchExactError("Phase 13P-B evidence contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Phase13PBBatchExactError(
                "Phase 13P-B evidence contains an unsafe file"
            )
        if path.relative_to(root).as_posix() not in excluded:
            files.append(path)
    return files


def generate(
    root: Path,
    *,
    equivalence_path: Path,
    git_sha: str,
) -> dict[str, Any]:
    """Seal only the focused remediation evidence inside the authorized image."""

    resolved = root.resolve(strict=True)
    if any(resolved.iterdir()):
        raise Phase13PBBatchExactError("Phase 13P-B output is not empty")
    if os.environ.get("KVBENCH_EXECUTION_ENVIRONMENT") != "measurement_container":
        raise Phase13PBBatchExactError("Phase 13P-B ran outside the container")
    if (
        os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST")
        != pilot.AUTHORIZED_CONTAINER_DIGEST
    ):
        raise Phase13PBBatchExactError("Phase 13P-B container digest differs")
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
        raise Phase13PBBatchExactError("Phase 13P-B source checkout differs")
    equivalence = _strict_json(equivalence_path)
    result = validate_equivalence(equivalence)
    if result["execution_git_sha"] != git_sha:
        raise Phase13PBBatchExactError("equivalence execution commit differs")
    write_exclusive(
        resolved / "prefix-equivalence.json",
        equivalence_path.read_bytes(),
    )
    write_exclusive(
        resolved / "source-authority.json",
        json_bytes(_source_authority(git_sha)),
    )
    write_exclusive(
        resolved / "preservation.json",
        json_bytes(
            {
                "schema_version": "kvbench-phase13pb-preservation-1.0.0",
                "stopped_phase13_campaigns_mounted": False,
                "stopped_phase13_campaigns_changed": False,
                "timing_data_read_or_analyzed": False,
                "pilot_executed": False,
                "timing_collected": False,
                "historical_evidence_mount_mode": "repository_read_only",
                "runtime_prefix_cache_sharing": False,
            }
        ),
    )
    manifest = {
        "schema_version": "kvbench-phase13pb-artifact-manifest-1.0.0",
        "run_id": resolved.name,
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "source_git_sha": git_sha,
        "authorized_container_digest": pilot.AUTHORIZED_CONTAINER_DIGEST,
        "decision_id": "0034",
        "case_count": 30,
        "cross_batch_rejection_count": 20,
        "diagnostic_only": True,
        "pilot_executed": False,
        "timing_collected": False,
        "complete_written_last": True,
        "append_only": True,
        "quality_status": "not_run",
        "performance_claim_eligible": False,
        "r_hbm": None,
    }
    write_exclusive(resolved / "manifest.json", json_bytes(manifest))
    inventory = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "role": "phase13pb_batch_exact_prefix_evidence",
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
        for path in _payload_paths(
            resolved,
            {"checksums.sha256", "COMPLETE"},
        )
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
    equivalence = _strict_json(resolved / "prefix-equivalence.json")
    result = validate_equivalence(equivalence)
    git_sha = str(equivalence["execution_git_sha"])
    if _strict_json(resolved / "source-authority.json") != _source_authority(
        git_sha
    ):
        raise Phase13PBBatchExactError("sealed source authority differs")
    preservation = _strict_json(resolved / "preservation.json")
    manifest = _strict_json(resolved / "manifest.json")
    if (
        preservation.get("stopped_phase13_campaigns_mounted") is not False
        or preservation.get("stopped_phase13_campaigns_changed") is not False
        or preservation.get("timing_data_read_or_analyzed") is not False
        or preservation.get("pilot_executed") is not False
        or manifest.get("status") != "PASS"
        or manifest.get("source_git_sha") != git_sha
        or manifest.get("authorized_container_digest")
        != pilot.AUTHORIZED_CONTAINER_DIGEST
        or manifest.get("decision_id") != "0034"
        or manifest.get("case_count") != 30
        or manifest.get("cross_batch_rejection_count") != 20
        or manifest.get("pilot_executed") is not False
        or manifest.get("timing_collected") is not False
        or manifest.get("r_hbm") is not None
    ):
        raise Phase13PBBatchExactError("Phase 13P-B sealed evidence differs")
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
    if CAMPAIGN_PATTERN.fullmatch(resolved.name) is None:
        raise Phase13PBBatchExactError("Phase 13P-B evidence ID differs")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    final = ARTIFACT_ROOT / resolved.name
    if final.exists() or final.is_symlink():
        raise Phase13PBBatchExactError("Phase 13P-B evidence ID already exists")
    rename_noreplace(resolved, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    validate(final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    new_id = subparsers.add_parser("new-id")
    new_id.add_argument("--git-sha", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--output", required=True, type=Path)
    generate_parser.add_argument("--equivalence", required=True, type=Path)
    generate_parser.add_argument("--git-sha", required=True)
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
    if arguments.command == "generate":
        print(
            json.dumps(
                generate(
                    arguments.output,
                    equivalence_path=arguments.equivalence,
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

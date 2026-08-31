#!/usr/bin/env python3
"""Focused container-only Phase 16R logical-prefix reconstruction probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from preflight.run_preflight import json_bytes, write_exclusive
import scripts.phase12_unified_admission as phase12
import scripts.phase13_pilot as phase13
import scripts.phase16_full_scan as phase16


class Phase16RValidationError(RuntimeError):
    """Focused logical-prefix reconstruction validation failed closed."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=phase16.CONFIGURATIONS, required=True)
    parser.add_argument("--batch", type=int, choices=(2, 16), required=True)
    parser.add_argument("--context-label", type=int, required=True)
    parser.add_argument("--logical-prefix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("reconstruct", "exact_snapshot"), required=True)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--snapshot-sha256")
    parser.add_argument("--snapshot-schema")
    parser.add_argument("--timing-smoke", action="store_true")
    return parser.parse_args()


def _entry(arguments: argparse.Namespace) -> dict[str, Any]:
    manifest, _, _ = phase16.logical_prefix.validate_logical_prefix_artifact(
        arguments.logical_prefix_root,
        expected={
            "batch_size": arguments.batch,
            "configured_context_label": arguments.context_label,
            "actual_historical_context": phase16.actual_historical_context(
                arguments.context_label
            ),
        },
        load_tensors=False,
    )
    logical = {
        "batch_size": arguments.batch,
        "configured_context_label": arguments.context_label,
        "actual_historical_context": manifest["actual_historical_context"],
        "logical_prefix_id": manifest["logical_prefix_id"],
        "artifact_root": str(arguments.logical_prefix_root),
        "token_checksum": manifest["token_tensor"]["sha256"],
        "decode_token_checksum": manifest["current_decode_token"]["sha256"],
        "token_file_sha256": manifest["token_file_sha256"],
    }
    if arguments.mode == "reconstruct":
        return {
            "kind": "logical_prefix",
            "logical_prefix": logical,
            "restore_mode": "logical_reconstruct",
            "optional_snapshot": {
                "readability": "snapshot_absent",
                "restore_compatibility": "not_applicable",
            },
            "schema_version": None,
            "snapshot_root": None,
            "state_file_sha256": None,
        }
    if (
        arguments.snapshot_root is None
        or arguments.snapshot_sha256 is None
        or re.fullmatch(r"[0-9a-f]{64}", arguments.snapshot_sha256) is None
        or arguments.snapshot_schema is None
    ):
        raise Phase16RValidationError("exact snapshot arguments are absent")
    return {
        "kind": "logical_prefix",
        "logical_prefix": logical,
        "restore_mode": "optional_exact_snapshot",
        "optional_snapshot": {
            "readability": "readable_current_layout",
            "restore_compatibility": "restore_allowed",
        },
        "schema_version": arguments.snapshot_schema,
        "snapshot_root": str(arguments.snapshot_root),
        "state_file_sha256": arguments.snapshot_sha256,
    }


def main() -> int:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise Phase16RValidationError("focused validation output already exists")
    attestation = phase12._require_authorized_container_runtime()
    before = phase12._capture_process_snapshot()
    phase12._require_idle_snapshot(before)
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
        raise Phase16RValidationError("focused CUDA authority differs")
    observed_head = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"),
        cwd=phase16.REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    historical = phase16.actual_historical_context(arguments.context_label)
    entry = _entry(arguments)
    phase13._patch_phase12_point_globals(batch=arguments.batch, historical=historical)
    loaded = load_frozen_model(device=torch.device("cuda:0"))
    operation = phase13.Phase13OperationKey.create(
        arguments.configuration, arguments.batch, historical
    )
    evidence_root = arguments.output.parent / f"{arguments.output.stem}-graph"
    evidence_root.mkdir(parents=False, exist_ok=False)
    with (
        phase16._worker_overrides(batch=arguments.batch, entry=entry),
        phase16._worker_context_label_override(arguments.context_label),
        torch.inference_mode(),
        forced_flash_execution(),
    ):
        prefix, decode = phase13._point_inputs(
            batch=arguments.batch,
            historical=historical,
            device=torch.device("cuda:0"),
        )
        with phase12._observable_cuda_graph_factory(torch) as observed_graphs:
            session, receipt = phase13._build_restored_session(
                loaded=loaded,
                operation=operation,
                prefix=prefix,
                decode=decode,
                snapshot_root=(
                    arguments.snapshot_root
                    if arguments.snapshot_root is not None
                    else evidence_root / "no-snapshot"
                ),
                expected_state_sha256=(
                    arguments.snapshot_sha256
                    if arguments.snapshot_sha256 is not None
                    else "0" * 64
                ),
            )
        if len(observed_graphs) != 1:
            raise Phase16RValidationError("focused Graph capture is ambiguous")
        pointers_before = phase12._phase12_session_pointers(session)
        history_before = session.current_historical_prefix_sha256()
        graph_path = phase12._write_cuda_graph_path_witness(
            graph=session.graph.graph,
            run_root=evidence_root,
            phase="before",
        )
        allocation = audit_cuda_allocations(
            session.graph.replay, device=session.cache_device
        )
        validation_output = (
            session.graph.replay().detach().to(device="cpu", copy=True).clone()
        )
        torch.cuda.synchronize(device=session.cache_device)
        output_checksum = tensor_sha256_untimed(validation_output)
        finite = bool(torch.isfinite(validation_output).all())
        graph_passed = bool(
            session.graph_evidence.get("captured") is True
            and session.graph_evidence.get("fallback") is False
            and session.graph_evidence.get("consecutive_replay_outputs_exact") is True
            and session.eager_graph_comparison is not None
            and session.eager_graph_comparison.passed
        )
        allocation_passed = bool(
            allocation.audit_available
            and allocation.passed
            and allocation.allocation_event_count == 0
            and allocation.allocation_event_bytes == 0
            and allocation.allocated_after == allocation.allocated_before
            and allocation.reserved_after == allocation.reserved_before
        )
        if (
            not finite
            or not graph_passed
            or not allocation_passed
            or pointers_before != phase12._phase12_session_pointers(session)
            or history_before != session.current_historical_prefix_sha256()
            or receipt.get("logical_input_validated") is not True
            or receipt.get("cache_build_or_restore_outside_timing") is not True
            or receipt.get("materialized_snapshot_required") is not False
        ):
            raise Phase16RValidationError("focused reconstruction invariant failed")
        timing: dict[str, Any] | None = None
        if arguments.timing_smoke:
            warm = warmup_operations(
                session.graph.replay, count=64, device=session.cache_device
            )
            warm_cpu = warm.detach().to(device="cpu", copy=True).clone()
            warm_checksum = tensor_sha256_untimed(warm_cpu)
            warm_finite = bool(torch.isfinite(warm_cpu).all())
            session.graph_evidence["replay_allocation"] = allocation.to_dict()
            session.admit(
                observed_outputs=((warm_checksum, warm_finite),),
                execution_path_passed=True,
                allocation_passed=allocation_passed,
                graph_passed=graph_passed,
            )
            runner = run_fixed_l(
                session, measured_steps=8, measured_batches=1
            ).to_dict()
            timing = {
                "run_kind": "validation_smoke",
                "claim_eligible": False,
                "measured_steps": 8,
                "measured_batches": 1,
                "output_checksum": runner["output_checksum"],
                "cache_layout_fingerprint": runner[
                    "cache_layout_fingerprint"
                ],
                "adapter_config_fingerprint": runner[
                    "adapter_config_fingerprint"
                ],
                "cache_pointers_stable": runner["cache_pointers_stable"],
                "historical_cache_unchanged": runner[
                    "historical_cache_unchanged"
                ],
                "r_hbm": runner.get("r_hbm"),
            }
    owned = phase12._capture_process_snapshot(
        supervised_pid=pid, supervised_start_ticks=start_ticks
    )
    phase12._require_owned_snapshot(owned, pid=pid, start_ticks=start_ticks)
    payload = {
        "schema_version": "kvbench-phase16r-reconstruction-validation-1.0.0",
        "status": "PASS",
        "configuration": arguments.configuration,
        "batch_size": arguments.batch,
        "context_label": arguments.context_label,
        "historical_context": historical,
        "mode": arguments.mode,
        "execution_git_sha": observed_head,
        "authorized_container_digest": phase16.PHASE16G_CONTAINER_DIGEST,
        "logical_prefix_id": entry["logical_prefix"]["logical_prefix_id"],
        "token_checksum": entry["logical_prefix"]["token_checksum"],
        "prefix_receipt": receipt,
        "output_checksum": output_checksum,
        "finite_output": finite,
        "cache_layout_fingerprint": session.cache_layout_fingerprint(),
        "cache_accounting": session.method_cache_accounting(),
        "kernel_path_fingerprint": graph_path["normalized_sha256"],
        "kernel_count": graph_path["kernel_node_count"],
        "pointers_stable": pointers_before
        == phase12._phase12_session_pointers(session),
        "graph_passed": graph_passed,
        "graph_replay_allocation_zero": allocation_passed,
        "materialized_cache_snapshot_written": False,
        "timing_smoke": timing,
        "container_attestation": attestation,
        "gpu_process_before": before,
        "gpu_process_owned_after": owned,
        "performance_claim_eligible": False,
        "quality_status": "unvalidated",
        "r_hbm": None,
    }
    write_exclusive(arguments.output, json_bytes(payload))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

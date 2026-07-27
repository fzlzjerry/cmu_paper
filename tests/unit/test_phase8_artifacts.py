"""Append-only lifecycle tests for strict Phase 8 KIVI run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kvbench.errors import (
    ArtifactConflictError,
    ArtifactStateError,
)
from kvbench.runtime.artifacts import (
    phase8_artifact_store,
    validate_run_directory,
)
from kvbench.schema.phase8 import (
    PHASE8_AUTHORIZED_CONTAINER_DIGEST,
    PHASE8_BASE_TREE,
    PHASE8_DECISION_0018_PATCH_SHA256,
    PHASE8_EXTENSION_SHA256,
    PHASE8_FIXTURE_ROOT_DIGEST,
    PHASE8_OFFICIAL_COMMIT,
    PHASE8_PATCHED_TREE,
    Phase8RunManifest,
)


def _payload(
    *,
    status: str = "created",
    failure_reason: str | None = None,
) -> dict[str, object]:
    breakdown = {
        "quantized_historical_k_payload": 10,
        "quantized_historical_v_payload": 10,
        "k_scales": 2,
        "k_zeros": 2,
        "v_scales": 2,
        "v_zeros": 2,
        "other_metadata": 0,
        "residual_k": 10,
        "residual_v": 10,
        "fp16_staging": 10,
        "quantization_staging": 10,
        "padding_alignment": 0,
        "persistent_workspace": 10,
        "value_rollover_shift_scratch": 10,
        "block_group_rounding": 2,
    }
    allocated = sum(breakdown.values())
    terminal = status not in {"created", "running", "finalizing"}
    started = None if status == "created" else "2026-07-27T00:00:30Z"
    finished = "2026-07-27T00:01:00Z" if terminal else None
    return {
        "schema_version": Phase8RunManifest.SCHEMA_VERSION,
        "artifact_schema_version": Phase8RunManifest.ARTIFACT_SCHEMA_VERSION,
        "run_id": "phase8-k4v4-fixed-128-eager-artifact-001",
        "status": status,
        "git_sha": "5" * 40,
        "git_dirty": False,
        "created_at_utc": "2026-07-27T00:00:00Z",
        "started_at_utc": started,
        "finished_at_utc": finished,
        "runner_kind": "fixed_l",
        "graph_mode": "eager",
        "method_configuration": "k4v4",
        "k_bits": 4,
        "v_bits": 4,
        "method_config_fingerprint": "1" * 64,
        "method_fingerprint": "2" * 64,
        "adapter_version": "phase8_kivi_1",
        "adapter_source_sha256": "3" * 64,
        "official_base_commit": PHASE8_OFFICIAL_COMMIT,
        "official_base_tree": PHASE8_BASE_TREE,
        "patched_tree": PHASE8_PATCHED_TREE,
        "decision_0018_patch_sha256": PHASE8_DECISION_0018_PATCH_SHA256,
        "extension_sha256": PHASE8_EXTENSION_SHA256,
        "fixture_root_digest": PHASE8_FIXTURE_ROOT_DIGEST,
        "group_size": 32,
        "residual_length": 32,
        "dtype_boundary": "bf16_to_fp16_official_kivi_to_bf16",
        "cache_layout_fingerprint": "4" * 64,
        "authorized_container_digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        "batch_size": 1,
        "context_length": 128,
        "output_steps": 1,
        "capacity": 129,
        "accounting": {
            "capacity": 129,
            "active_context": 128,
            "allocated_bytes": allocated,
            "predicted_allocated_bytes": allocated,
            "active_storage_bytes": 64,
            "logical_bf16_allocated_bytes": 516,
            "logical_bf16_active_bytes": 512,
            "rho_alloc": allocated / 516,
            "r_alloc": 516 / allocated,
            "predicted_relative_error": 0.0,
            "temporary_peak_bytes": 0,
            "breakdown": breakdown,
            "r_hbm": None,
        },
        "quality_status": "unvalidated",
        "claim_eligibility": "performance_only",
        "performance_claim_eligible": False,
        "measurement_scope": "measurement_container_admission",
        "quality_execution": "locked",
        "quality_benchmark_executed": False,
        "performance_data_frozen": False,
        "speedup_calculated": False,
        "full_scan_state": "CLOSED",
        "inventory_path": "artifact_inventory.json" if terminal else None,
        "failure_reason": failure_reason,
    }


def _write_required_payloads(run: object) -> None:
    run.write_json("config/method.json", {"method": "kivi", "config": "k4v4"})
    run.write_json(
        "environment/container_identity.json",
        {"digest": PHASE8_AUTHORIZED_CONTAINER_DIGEST},
    )
    run.write_json("raw/runner.json", {"runner": "fixed_l"})
    run.write_json("validation/point.json", {"passed": True})


def _restore_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass
    try:
        root.chmod(0o700)
    except OSError:
        pass


class Phase8ArtifactTests(unittest.TestCase):
    def test_manifest_accepts_created_running_and_terminal_states(self) -> None:
        created = Phase8RunManifest.from_dict(_payload())
        running = Phase8RunManifest.from_dict(_payload(status="running"))
        terminal = Phase8RunManifest.from_dict(_payload(status="completed"))

        self.assertEqual(created.status.value, "created")
        self.assertEqual(running.status.value, "running")
        self.assertEqual(terminal.status.value, "completed")

    def test_complete_is_last_and_no_path_or_run_id_is_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = phase8_artifact_store(root)
            initial = Phase8RunManifest.from_dict(_payload())
            run = store.create(initial.run_id, initial)
            run.start()
            _write_required_payloads(run)
            self.assertFalse((run.stage / "COMPLETE").exists())
            with self.assertRaises(ArtifactConflictError):
                run.write_json("raw/runner.json", {"replacement": True})

            final = run.finalize(
                Phase8RunManifest.from_dict(_payload(status="completed"))
            )
            self.addCleanup(_restore_writable, root)
            validation = validate_run_directory(final)
            self.assertTrue(validation.valid, validation.errors)
            self.assertTrue(validation.complete)
            completion = json.loads(
                (final / "COMPLETE").read_text(encoding="utf-8")
            )
            self.assertTrue(completion["written_last"])
            self.assertNotIn(
                "COMPLETE",
                (final / "checksums.sha256").read_text(encoding="utf-8"),
            )
            with self.assertRaises(ArtifactConflictError):
                store.create(initial.run_id, initial)

    def test_terminal_failure_is_finalized_and_preserves_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = phase8_artifact_store(root)
            initial = Phase8RunManifest.from_dict(_payload())
            run = store.create(initial.run_id, initial)
            run.start()
            _write_required_payloads(run)
            run.write_json(
                "validation/runtime_failure.json",
                {"failure": "official kernel launch failed"},
            )
            final = run.finalize(
                Phase8RunManifest.from_dict(
                    _payload(
                        status="runtime_failed",
                        failure_reason="official kernel launch failed",
                    )
                )
            )
            self.addCleanup(_restore_writable, root)

            validation = validate_run_directory(final)
            self.assertTrue(validation.valid, validation.errors)
            self.assertTrue(validation.complete)
            self.assertEqual(validation.status, "runtime_failed")
            self.assertTrue(
                (final / "validation/runtime_failure.json").is_file()
            )
            manifest = json.loads(
                (final / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["failure_reason"],
                "official kernel launch failed",
            )

    def test_failed_finalization_keeps_staging_and_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = phase8_artifact_store(root)
            initial = Phase8RunManifest.from_dict(_payload())
            run = store.create(initial.run_id, initial)
            run.start()
            run.write_json("raw/runner.json", {"partial": True})

            with self.assertRaises(ArtifactStateError):
                run.finalize(
                    Phase8RunManifest.from_dict(_payload(status="completed"))
                )

            self.assertTrue(run.stage.is_dir())
            self.assertTrue((run.stage / "raw/runner.json").is_file())
            self.assertFalse((run.stage / "COMPLETE").exists())
            with self.assertRaises(ArtifactConflictError):
                store.create(initial.run_id, initial)


if __name__ == "__main__":
    unittest.main()

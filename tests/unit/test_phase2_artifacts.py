"""Append-only lifecycle, manifest, and provenance tests for Phase 2."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from kvbench.errors import (
    ArtifactConflictError,
    ArtifactSafetyError,
    ArtifactStateError,
    ProvenanceError,
    SchemaValidationError,
)
from kvbench.runtime.artifacts import (
    AppendOnlyArtifactStore,
    summarize_run_directory,
    validate_run_directory,
)
from kvbench.runtime.command import reconstruct_command
from kvbench.schema import RunManifest, SampleRecord, canonical_json_bytes


ZERO_SHA256 = "0" * 64
ONE_SHA256 = "1" * 64
TWO_SHA256 = "2" * 64
THREE_SHA256 = "3" * 64
ZERO_GIT_SHA = "0" * 40
CREATED_AT = "2026-07-22T00:00:00Z"
STARTED_AT = "2026-07-22T00:00:01Z"
FINISHED_AT = "2026-07-22T00:00:02Z"
PLAN_PATH = "configs/plans/smoke.yaml"


def _method_fingerprint() -> dict[str, object]:
    return {
        "schema_version": "kvbench.method-fingerprint.v1",
        "method": "bf16",
        "variant_id": "bf16",
        "canonicalization": "kvbench-json-v1",
        "algorithm": "sha256",
        "sha256": ZERO_SHA256,
        "execution_ready": False,
    }


def _quality_not_applicable() -> dict[str, object]:
    return {
        "schema_version": "kvbench.quality-status.v1",
        "quality_status": "not_applicable",
        "claim_eligibility": "none",
        "quality_execution": "locked",
        "performance_data_frozen": False,
    }


def created_manifest(run_id: str, *, git_dirty: bool = True) -> dict[str, object]:
    return {
        "schema_version": "kvbench-run-manifest-1.0.0",
        "artifact_schema_version": "kvbench-artifacts-1.0.0",
        "run_id": run_id,
        "status": "created",
        "created_at_utc": CREATED_AT,
        "started_at_utc": None,
        "finished_at_utc": None,
        "run_kind": "synthetic",
        "runner_kind": "fixed_l",
        "graph_mode": "eager",
        "claim_class": "none",
        "plan_source": {
            "kind": "path",
            "path": PLAN_PATH,
            "canonical_inline_json": None,
            "sha256": ONE_SHA256,
        },
        "git_sha": ZERO_GIT_SHA,
        "git_dirty": git_dirty,
        "container_digest": None,
        "hardware_id": "synthetic-hardware",
        "hardware_fingerprint": ZERO_SHA256,
        "software_environment_id": "synthetic-software",
        "software_fingerprint": ONE_SHA256,
        "model_id": "synthetic/model",
        "model_revision": "unresolved-synthetic-revision",
        "model_fingerprint": TWO_SHA256,
        "method": "bf16",
        "method_config_id": "bf16-placeholder",
        "method_config_fingerprint": _method_fingerprint(),
        "contract_fingerprint": THREE_SHA256,
        "attention_backend": None,
        "cache_layout": None,
        "random_seed": 20260722,
        "process_replicate": 1,
        "quality": _quality_not_applicable(),
        "command": {
            "schema_version": "kvbench-command-1.0.0",
            "argv": ["kvbench", "run", "--plan", PLAN_PATH, "--dry-run"],
            "dry_run": True,
        },
        "inventory_path": None,
        "failure_reason": None,
    }


def terminal_manifest(
    run_id: str,
    *,
    status: str = "completed",
    failure_reason: str | None = None,
) -> dict[str, object]:
    manifest = created_manifest(run_id)
    manifest.update(
        {
            "status": status,
            "started_at_utc": STARTED_AT,
            "finished_at_utc": FINISHED_AT,
            "inventory_path": "artifact_inventory.json",
            "failure_reason": failure_reason,
        }
    )
    return manifest


def synthetic_sample(run_id: str) -> dict[str, object]:
    return {
        "schema_version": "kvbench-sample-1.0.0",
        "run_id": run_id,
        "timestamp_utc": FINISHED_AT,
        "git_sha": ZERO_GIT_SHA,
        "git_dirty": True,
        "container_digest": None,
        "gpu_uuid": "synthetic-gpu",
        "gpu_full_name": "synthetic fixture GPU",
        "pci_device_id": "synthetic",
        "driver_version": "not-collected",
        "cuda_runtime_version": "not-collected",
        "cuda_toolkit_version": "not-collected",
        "torch_version": "not-collected",
        "triton_version": "not-collected",
        "hardware_fingerprint": ZERO_SHA256,
        "software_fingerprint": ONE_SHA256,
        "contract_fingerprint": THREE_SHA256,
        "method": "bf16",
        "method_config_id": "bf16-placeholder",
        "method_config_fingerprint": _method_fingerprint(),
        "model_id": "synthetic/model",
        "model_revision": "unresolved-synthetic-revision",
        "model_fingerprint": TWO_SHA256,
        "weight_dtype": "synthetic",
        "batch_size": 1,
        "context_length": 1,
        "decode_step": 1,
        "runner_kind": "fixed_l",
        "graph_mode": "eager",
        "attention_backend": "synthetic",
        "cache_layout": "synthetic",
        "r_nominal": None,
        "r_alloc": None,
        "r_hbm": None,
        "hbm_evidence": None,
        "logical_bf16_bytes": None,
        "cache_allocated_bytes": None,
        "cache_data_bytes": None,
        "metadata_bytes": None,
        "scale_zero_bytes": None,
        "norm_bytes": None,
        "residual_bytes": None,
        "sink_bytes": None,
        "outlier_value_bytes": None,
        "outlier_index_bytes": None,
        "padding_bytes": None,
        "workspace_bytes": None,
        "peak_memory_bytes": None,
        "wall_time_ms": None,
        "gpu_event_ms": None,
        "kernel_count": None,
        "sm_clock_mhz": None,
        "memory_clock_mhz": None,
        "power_w": None,
        "temperature_c": None,
        "replicate": 1,
        "step_index": 0,
        "random_seed": 20260722,
        "run_kind": "synthetic",
        "status": "completed",
        "failure_reason": None,
        "claim_class": "none",
        "quality": _quality_not_applicable(),
    }


class ManifestAndReconstructionTests(unittest.TestCase):
    def test_manifest_validates_independently_and_retains_provenance(self) -> None:
        parsed = RunManifest.from_dict(created_manifest("manifest-independent"))
        serialized = parsed.to_dict()
        self.assertTrue(serialized["git_dirty"])
        self.assertEqual(serialized["hardware_fingerprint"], ZERO_SHA256)
        self.assertEqual(serialized["software_fingerprint"], ONE_SHA256)
        self.assertEqual(serialized["model_fingerprint"], TWO_SHA256)
        self.assertEqual(serialized["contract_fingerprint"], THREE_SHA256)
        self.assertEqual(
            serialized["method_config_fingerprint"]["sha256"], ZERO_SHA256
        )
        self.assertIsNone(serialized["container_digest"])
        self.assertIsNone(serialized["attention_backend"])
        self.assertIsNone(serialized["cache_layout"])

    def test_command_reconstruction_is_deterministic_and_state_free(self) -> None:
        manifest = created_manifest("command-reconstruction")
        expected = ("kvbench", "run", "--plan", PLAN_PATH, "--dry-run")
        self.assertEqual(reconstruct_command(manifest), expected)
        self.assertEqual(reconstruct_command(copy.deepcopy(manifest)), expected)
        parsed = RunManifest.from_dict(manifest)
        self.assertEqual(reconstruct_command(parsed), expected)
        self.assertEqual(parsed.command.argv, expected)

    def test_missing_provenance_fingerprint_is_rejected(self) -> None:
        manifest = created_manifest("missing-provenance")
        del manifest["hardware_fingerprint"]
        with self.assertRaises(SchemaValidationError):
            RunManifest.from_dict(manifest)

    def test_null_metric_semantics_round_trip(self) -> None:
        parsed = SampleRecord.from_dict(synthetic_sample("null-metrics"))
        serialized = parsed.to_dict()
        for field in (
            "r_nominal",
            "r_alloc",
            "r_hbm",
            "hbm_evidence",
            "wall_time_ms",
            "gpu_event_ms",
            "logical_bf16_bytes",
            "cache_allocated_bytes",
        ):
            self.assertIn(field, serialized)
            self.assertIsNone(serialized[field])
        self.assertEqual(parsed.canonical_bytes(), canonical_json_bytes(serialized))

    def test_estimated_or_unsupported_r_hbm_cannot_masquerade_as_measured(self) -> None:
        sample = synthetic_sample("invalid-hbm")
        sample["r_hbm"] = 2.0
        with self.assertRaises(SchemaValidationError) as caught:
            SampleRecord.from_dict(sample)
        self.assertIn("r_hbm requires direct ncu evidence", str(caught.exception))


class ArtifactLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "synthetic-runs"
        self.formal = self.base / "formal-evidence"
        self.store = AppendOnlyArtifactStore(
            self.root, formal_evidence_roots=[self.formal]
        )

    def _completed_run(self, run_id: str = "completed-run") -> tuple[object, Path]:
        run = self.store.create(run_id, created_manifest(run_id))
        run.start()
        run.write_json(
            "raw/synthetic.json",
            {"fixture": True, "scientific_measurement": False},
        )
        final = run.finalize(terminal_manifest(run_id))
        return run, final

    def test_new_run_stages_then_atomically_finalizes_with_inventory(self) -> None:
        run_id = "atomic-finalization"
        run = self.store.create(run_id, created_manifest(run_id))
        self.assertEqual(run.state, "created")
        self.assertTrue(run.stage.is_dir())
        self.assertEqual(run.stage.parent.name, ".kvbench-staging")
        self.assertFalse((self.root / run_id).exists())
        run.start()
        run.write_bytes("raw/payload.bin", b"synthetic-only")
        stage = run.stage
        final = run.finalize(terminal_manifest(run_id))
        self.assertFalse(stage.exists())
        self.assertEqual(final, self.root / run_id)
        self.assertEqual(run.state, "completed")
        for control in (
            "manifest.initial.json",
            "manifest.json",
            "artifact_inventory.json",
            "checksums.sha256",
            "COMPLETE",
        ):
            self.assertTrue((final / control).is_file())
        validation = validate_run_directory(final)
        self.assertTrue(validation.valid, validation.errors)
        self.assertTrue(validation.complete)
        inventory = json.loads(
            (final / "artifact_inventory.json").read_text(encoding="utf-8")
        )
        declared = {item["path"] for item in inventory["files"]}
        actual = {
            path.relative_to(final).as_posix()
            for path in final.rglob("*")
            if path.is_file()
            and path.name not in {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
        }
        self.assertEqual(declared, actual)
        self.assertIn("raw/payload.bin", declared)

    def test_existing_run_id_rejected_and_completed_api_is_locked(self) -> None:
        run, _ = self._completed_run("immutable-completed")
        with self.assertRaises(ArtifactConflictError):
            self.store.create(
                "immutable-completed", created_manifest("immutable-completed")
            )
        with self.assertRaises(ArtifactStateError):
            run.write_bytes("raw/late.bin", b"late")
        with self.assertRaises(ArtifactStateError):
            run.finalize(terminal_manifest("immutable-completed"))

    def test_invalid_initial_manifest_creates_no_artifact_state(self) -> None:
        invalid = created_manifest("invalid-initial")
        del invalid["hardware_fingerprint"]
        with self.assertRaises(SchemaValidationError):
            self.store.create("invalid-initial", invalid)
        self.assertFalse(self.root.exists())

    def test_final_manifest_cannot_change_initial_provenance(self) -> None:
        run_id = "immutable-provenance"
        run = self.store.create(run_id, created_manifest(run_id))
        run.start()
        final = terminal_manifest(run_id)
        final["git_sha"] = "1" * 40
        with self.assertRaises(ProvenanceError):
            run.finalize(final)
        self.assertEqual(run.state, "running")
        self.assertTrue(run.stage.is_dir())

    def test_failed_run_is_finalized_and_preserved(self) -> None:
        run_id = "preserved-failure"
        run = self.store.create(run_id, created_manifest(run_id))
        run.start()
        run.write_json("validation/failure.json", {"reason": "synthetic failure"})
        final = run.finalize(
            terminal_manifest(
                run_id,
                status="runtime_failed",
                failure_reason="synthetic fixture failure",
            )
        )
        self.assertTrue(final.is_dir())
        self.assertEqual(run.state, "runtime_failed")
        validation = validate_run_directory(final)
        self.assertTrue(validation.valid, validation.errors)
        self.assertEqual(validation.status, "runtime_failed")
        summary = summarize_run_directory(final)
        self.assertEqual(summary["status"], "runtime_failed")
        self.assertFalse(summary["scientific_conclusions_generated"])

    def test_interrupted_run_remains_distinguishable_and_reserved(self) -> None:
        run_id = "interrupted-run"
        run = self.store.create(run_id, created_manifest(run_id))
        run.start()
        run.write_bytes("logs/interrupted.log", b"synthetic interruption")
        validation = validate_run_directory(run.stage, expect_final_name=False)
        self.assertFalse(validation.valid)
        self.assertFalse(validation.complete)
        self.assertIn("COMPLETE is absent", " ".join(validation.errors))
        self.assertTrue(run.stage.is_dir())
        with self.assertRaises(ArtifactConflictError):
            self.store.create(run_id, created_manifest(run_id))

    def test_checksum_and_inventory_tampering_is_detected(self) -> None:
        _, final = self._completed_run("tamper-detection")
        payload = final / "raw" / "synthetic.json"
        payload.chmod(0o644)
        payload.write_bytes(b'{"fixture":false}\n')
        validation = validate_run_directory(final)
        self.assertFalse(validation.valid)
        errors = "\n".join(validation.errors)
        self.assertIn("checksum mismatch: raw/synthetic.json", errors)
        self.assertIn("artifact inventory", errors)

    def test_unsafe_run_ids_and_artifact_paths_are_rejected(self) -> None:
        for unsafe in ("../escape", "/absolute", ".hidden", "UPPERCASE", ""):
            with self.subTest(run_id=unsafe):
                with self.assertRaises(ArtifactSafetyError):
                    self.store.create(unsafe, created_manifest("safe-placeholder"))
        run = self.store.create("path-safety", created_manifest("path-safety"))
        run.start()
        for unsafe_path in (
            "../escape",
            "/absolute",
            "a/../b",
            "a//b",
            "manifest.json",
            "raw/bad\nname",
        ):
            with self.subTest(path=unsafe_path):
                with self.assertRaises(ArtifactSafetyError):
                    run.write_bytes(unsafe_path, b"unsafe")

    def test_concurrent_duplicate_run_id_has_exactly_one_winner(self) -> None:
        run_id = "concurrent-reservation"
        barrier = threading.Barrier(2)

        def attempt() -> str:
            barrier.wait(timeout=5)
            try:
                self.store.create(run_id, created_manifest(run_id))
            except ArtifactConflictError:
                return "conflict"
            return "created"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = sorted(executor.map(lambda _: attempt(), range(2)))
        self.assertEqual(outcomes, ["conflict", "created"])
        stages = list((self.root / ".kvbench-staging").glob(f"{run_id}.*.staging"))
        self.assertEqual(len(stages), 1)

    def test_test_root_cannot_overlap_formal_evidence_boundary(self) -> None:
        for unsafe_root in (self.formal, self.formal / "child", self.base):
            with self.subTest(root=unsafe_root):
                with self.assertRaises(ArtifactSafetyError):
                    AppendOnlyArtifactStore(
                        unsafe_root, formal_evidence_roots=[self.formal]
                    )

    def test_internal_control_directory_symlink_is_rejected(self) -> None:
        self.root.mkdir()
        redirected = self.base / "redirected-staging"
        redirected.mkdir()
        (self.root / ".kvbench-staging").symlink_to(
            redirected, target_is_directory=True
        )
        with self.assertRaises(ArtifactSafetyError):
            self.store.create("symlink-control", created_manifest("symlink-control"))
        self.assertEqual(tuple(redirected.iterdir()), ())

    def test_hardlinked_payload_is_rejected_before_promotion(self) -> None:
        run_id = "hardlink-rejection"
        run = self.store.create(run_id, created_manifest(run_id))
        run.start()
        payload = run.write_bytes("raw/payload.bin", b"fixture")
        os.link(payload, run.stage / "raw" / "alias.bin")
        with self.assertRaises(ArtifactSafetyError):
            run.finalize(terminal_manifest(run_id))
        self.assertFalse((self.root / run_id).exists())
        self.assertTrue(run.stage.is_dir())

    def test_symlink_run_directory_returns_invalid_without_following(self) -> None:
        _, final = self._completed_run("symlink-validation")
        link = self.base / "run-link"
        link.symlink_to(final, target_is_directory=True)
        result = validate_run_directory(link)
        self.assertFalse(result.valid)
        self.assertFalse(result.complete)

    def test_lifecycle_and_completion_schemas_are_independently_validated(self) -> None:
        _, lifecycle_final = self._completed_run("lifecycle-tamper")
        lifecycle = lifecycle_final / "lifecycle"
        lifecycle.chmod(0o755)
        (lifecycle / "0004-completed.json").unlink()
        lifecycle_result = validate_run_directory(lifecycle_final)
        self.assertFalse(lifecycle_result.valid)
        self.assertIn("exactly four lifecycle", "\n".join(lifecycle_result.errors))

        _, completion_final = self._completed_run("completion-tamper")
        completion = completion_final / "COMPLETE"
        completion.chmod(0o644)
        payload = json.loads(completion.read_text(encoding="utf-8"))
        payload["unknown"] = "rejected"
        completion.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        completion_result = validate_run_directory(completion_final)
        self.assertFalse(completion_result.valid)
        self.assertIn(
            "completion marker failed independent schema validation",
            "\n".join(completion_result.errors),
        )


if __name__ == "__main__":
    unittest.main()

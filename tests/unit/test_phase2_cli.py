"""Black-box command-line skeleton tests for Phase 2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from kvbench.runtime.artifacts import AppendOnlyArtifactStore


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ZERO_SHA256 = "0" * 64
ONE_SHA256 = "1" * 64
TWO_SHA256 = "2" * 64
THREE_SHA256 = "3" * 64
PLAN_PATH = "configs/plans/smoke.yaml"


def _manifest(run_id: str, *, terminal: bool = False) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "kvbench-run-manifest-1.0.0",
        "artifact_schema_version": "kvbench-artifacts-1.0.0",
        "run_id": run_id,
        "status": "created",
        "created_at_utc": "2026-07-22T00:00:00Z",
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
        "git_sha": "0" * 40,
        "git_dirty": True,
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
        "method_config_fingerprint": {
            "schema_version": "kvbench.method-fingerprint.v1",
            "method": "bf16",
            "variant_id": "bf16",
            "canonicalization": "kvbench-json-v1",
            "algorithm": "sha256",
            "sha256": ZERO_SHA256,
            "execution_ready": False,
        },
        "contract_fingerprint": THREE_SHA256,
        "attention_backend": None,
        "cache_layout": None,
        "random_seed": 20260722,
        "process_replicate": 1,
        "quality": {
            "schema_version": "kvbench.quality-status.v1",
            "quality_status": "not_applicable",
            "claim_eligibility": "none",
            "quality_execution": "locked",
            "performance_data_frozen": False,
        },
        "command": {
            "schema_version": "kvbench-command-1.0.0",
            "argv": ["kvbench", "run", "--plan", PLAN_PATH, "--dry-run"],
            "dry_run": True,
        },
        "inventory_path": None,
        "failure_reason": None,
    }
    if terminal:
        manifest.update(
            {
                "status": "completed",
                "started_at_utc": "2026-07-22T00:00:01Z",
                "finished_at_utc": "2026-07-22T00:00:02Z",
                "inventory_path": "artifact_inventory.json",
            }
        )
    return manifest


def _invoke(argv: list[str]) -> tuple[int, str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": "src",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    result = subprocess.run(
        ["/usr/bin/python3", "-m", "kvbench", *argv],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return result.returncode, result.stdout, result.stderr


class Phase2CliTests(unittest.TestCase):
    def test_validate_config_success(self) -> None:
        hardware = next((REPOSITORY_ROOT / "configs" / "hardware").glob("*.yaml"))
        code, output, error = _invoke(["validate-config", str(hardware)])
        self.assertEqual(code, 0, error)
        payload = json.loads(output)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["command"], "validate-config")
        self.assertEqual(payload["document_type"], "hardware")
        self.assertFalse(payload["execution_attempted"])

    def test_validate_config_failure_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.yaml"
            invalid.write_text(
                '{"document_type":"hardware","unknown":"rejected"}\n',
                encoding="utf-8",
            )
            code, output, error = _invoke(["validate-config", str(invalid)])
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        payload = json.loads(error)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "schema_validation_error")

    def test_run_dry_run_validates_and_never_executes_any_lane(self) -> None:
        plan = REPOSITORY_ROOT / PLAN_PATH
        code, output, error = _invoke(["run", "--plan", str(plan), "--dry-run"])
        self.assertEqual(code, 0, error)
        payload = json.loads(output)
        self.assertTrue(payload["ok"])
        self.assertEqual(
            payload["intended_argv"],
            ["kvbench", "run", "--plan", PLAN_PATH, "--dry-run"],
        )
        self.assertFalse(payload["performance_execution_implemented"])
        self.assertFalse(payload["execution_attempted"])
        self.assertFalse(payload["timing_collected"])
        self.assertFalse(payload["profiler_executed"])
        self.assertFalse(payload["quality_executed"])
        self.assertEqual(payload["admission"]["quality_execution"], "locked")
        self.assertFalse(payload["admission"]["performance_data_frozen"])

    def test_run_without_dry_run_fails_closed_for_phase_three(self) -> None:
        plan = REPOSITORY_ROOT / PLAN_PATH
        code, output, error = _invoke(["run", "--plan", str(plan)])
        self.assertEqual(code, 3)
        self.assertEqual(output, "")
        payload = json.loads(error)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "phase_not_implemented")
        self.assertFalse(payload["execution_attempted"])
        self.assertFalse(payload["timing_collected"])
        self.assertFalse(payload["profiler_executed"])
        self.assertFalse(payload["quality_executed"])

    def test_validate_run_success_and_checksum_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            store = AppendOnlyArtifactStore(
                root,
                formal_evidence_roots=[Path(directory) / "formal"],
            )
            run = store.create("cli-valid-run", _manifest("cli-valid-run"))
            run.start()
            run.write_bytes("raw/synthetic.bin", b"fixture")
            final = run.finalize(_manifest("cli-valid-run", terminal=True))

            code, output, error = _invoke(["validate-run", str(final)])
            self.assertEqual(code, 0, error)
            payload = json.loads(output)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["result"]["valid"])
            self.assertTrue(payload["result"]["complete"])

            artifact = final / "raw" / "synthetic.bin"
            artifact.chmod(0o644)
            artifact.write_bytes(b"tampered fixture")
            code, output, error = _invoke(["validate-run", str(final)])
            self.assertEqual(code, 1, error)
            payload = json.loads(output)
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["result"]["valid"])
            self.assertTrue(
                any("checksum mismatch" in item for item in payload["result"]["errors"])
            )

    def test_summarize_handles_incomplete_run_without_conclusions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AppendOnlyArtifactStore(
                Path(directory) / "runs",
                formal_evidence_roots=[Path(directory) / "formal"],
            )
            run = store.create("cli-incomplete", _manifest("cli-incomplete"))
            run.start()
            run.write_bytes("logs/fixture.log", b"incomplete")
            code, output, error = _invoke(["summarize", str(run.stage)])
        self.assertEqual(code, 0, error)
        payload = json.loads(output)
        summary = payload["summary"]
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["status"], "running")
        self.assertFalse(summary["scientific_conclusions_generated"])


if __name__ == "__main__":
    unittest.main()

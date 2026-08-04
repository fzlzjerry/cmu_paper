"""Focused fail-closed tests for Phase 13P-C process isolation."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import phase13_pilot as pilot
from scripts import phase13pc_cuda_context_isolation as remediation
from tests.unit.test_phase13pr_prefix_state import _equivalence_payload


def _idle_snapshot() -> dict[str, object]:
    return {
        "query_exit_code": 0,
        "errors": [],
        "allowed_compute_processes": [],
        "foreign_compute_processes": [],
        "unknown_processes": [],
    }


def _run_with_payload(payload: dict[str, object], returncode: int = 0):
    def fake(command, **kwargs):
        del kwargs
        output = Path(command[command.index("--output") + 1])
        if returncode == 0:
            output.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(
            command, returncode, stdout=b"child-result", stderr=b""
        )

    return fake


class Phase13PCIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = _equivalence_payload()
        self.git_sha = self.payload["execution_git_sha"]
        child_environment = mock.patch.object(
            pilot.phase12, "_child_environment", return_value={}
        )
        child_environment.start()
        self.addCleanup(child_environment.stop)

    def _paths(self, temporary: str) -> tuple[Path, Path, Path]:
        root = Path(temporary)
        scratch = root / "scratch"
        scratch.mkdir()
        return root / "equivalence.json", scratch, root / "handoff.json"

    def test_campaign_parent_uses_dedicated_child_and_never_calls_cuda_gate(
        self,
    ) -> None:
        campaign = inspect.getsource(pilot.run_campaign)
        isolated = inspect.getsource(pilot.run_prefix_equivalence_isolated)
        self.assertIn("run_prefix_equivalence_isolated(", campaign)
        self.assertNotIn("run_prefix_equivalence(", campaign)
        self.assertIn('"--validate-prefix-equivalence"', isolated)
        self.assertNotIn("import torch", isolated)
        self.assertNotIn("load_frozen_model", isolated)
        self.assertNotIn("empty_cache", isolated)
        self.assertNotIn("supervised_pid", isolated)

    def test_successful_30_case_child_handoff_is_idle_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, scratch, handoff = self._paths(temporary)
            with mock.patch.object(
                pilot.phase12,
                "_capture_process_snapshot",
                side_effect=[_idle_snapshot(), _idle_snapshot()],
            ) as snapshots, mock.patch.object(
                pilot.subprocess,
                "run",
                side_effect=_run_with_payload(self.payload),
            ):
                result = pilot.run_prefix_equivalence_isolated(
                    output=output,
                    scratch_root=scratch,
                    handoff_output=handoff,
                    git_sha=self.git_sha,
                )
            self.assertEqual(result["case_count"], 30)
            self.assertEqual(snapshots.call_count, 2)
            evidence = json.loads(handoff.read_text(encoding="utf-8"))
            self.assertTrue(evidence["dedicated_child_process"])
            self.assertFalse(evidence["parent_initialized_cuda"])
            self.assertEqual(evidence["remaining_cuda_processes"], [])
            self.assertEqual(evidence["foreign_cuda_processes"], [])
            validated = remediation.validate_handoff(
                evidence, equivalence_path=output
            )
            self.assertEqual(validated["case_count"], 30)

    def test_child_failure_fails_closed_after_post_exit_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, scratch, handoff = self._paths(temporary)
            with mock.patch.object(
                pilot.phase12,
                "_capture_process_snapshot",
                side_effect=[_idle_snapshot(), _idle_snapshot()],
            ) as snapshots, mock.patch.object(
                pilot.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    ("child",), 7, stdout=b"", stderr=b"failed"
                ),
            ):
                with self.assertRaisesRegex(
                    pilot.Phase13PilotError, "child failed"
                ):
                    pilot.run_prefix_equivalence_isolated(
                        output=output,
                        scratch_root=scratch,
                        handoff_output=handoff,
                        git_sha=self.git_sha,
                    )
            self.assertEqual(snapshots.call_count, 2)
            self.assertFalse(handoff.exists())

    def test_residual_cuda_context_is_rejected(self) -> None:
        residual = _idle_snapshot()
        residual["allowed_compute_processes"] = [{"pid": 1234}]
        with tempfile.TemporaryDirectory() as temporary:
            output, scratch, handoff = self._paths(temporary)
            with mock.patch.object(
                pilot.phase12,
                "_capture_process_snapshot",
                side_effect=[_idle_snapshot(), residual],
            ), mock.patch.object(
                pilot.subprocess,
                "run",
                side_effect=_run_with_payload(self.payload),
            ):
                with self.assertRaisesRegex(Exception, "active CUDA process"):
                    pilot.run_prefix_equivalence_isolated(
                        output=output,
                        scratch_root=scratch,
                        handoff_output=handoff,
                        git_sha=self.git_sha,
                    )
            self.assertFalse(handoff.exists())

    def test_child_evidence_tampering_is_rejected(self) -> None:
        tampered = json.loads(json.dumps(self.payload))
        tampered["records"][0]["output_checksum_exact"] = False
        with tempfile.TemporaryDirectory() as temporary:
            output, scratch, handoff = self._paths(temporary)
            with mock.patch.object(
                pilot.phase12,
                "_capture_process_snapshot",
                side_effect=[_idle_snapshot(), _idle_snapshot()],
            ), mock.patch.object(
                pilot.subprocess,
                "run",
                side_effect=_run_with_payload(tampered),
            ):
                with self.assertRaises(Exception):
                    pilot.run_prefix_equivalence_isolated(
                        output=output,
                        scratch_root=scratch,
                        handoff_output=handoff,
                        git_sha=self.git_sha,
                    )
            self.assertFalse(handoff.exists())

    def test_handoff_field_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "equivalence.json"
            output.write_text(
                json.dumps(self.payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            base = {
                "schema_version": "kvbench-phase13-prefix-equivalence-handoff-1.0.0",
                "status": "PASS",
                "decision_id": "0035",
                "execution_git_sha": self.git_sha,
                "authorized_container_digest": pilot.AUTHORIZED_CONTAINER_DIGEST,
                "dedicated_child_process": True,
                "parent_loaded_model": False,
                "parent_initialized_cuda": False,
                "child_exit_code": 0,
                "child_exit_passed": True,
                "child_evidence_validated": True,
                "equivalence_case_count": 30,
                "equivalence_cross_batch_rejection_count": 20,
                "equivalence_sha256": pilot.sha256_file(output),
                "child_stdout_sha256": "a" * 64,
                "child_stderr_sha256": "b" * 64,
                "pre_idle_gpu": True,
                "post_idle_gpu": True,
                "pre_snapshot": _idle_snapshot(),
                "post_snapshot": _idle_snapshot(),
                "remaining_cuda_processes": [],
                "foreign_cuda_processes": [],
                "runtime_prefix_cache_sharing": False,
                "timing_collected": False,
            }
            for field, value in (
                ("dedicated_child_process", False),
                ("parent_initialized_cuda", True),
                ("child_exit_code", 1),
                ("post_idle_gpu", False),
                ("remaining_cuda_processes", [{"pid": 1}]),
            ):
                changed = dict(base)
                changed[field] = value
                with self.assertRaises(remediation.Phase13PCIsolationError):
                    remediation.validate_handoff(changed, equivalence_path=output)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from scripts import phase16_full_scan as phase16
from scripts import phase16_full_scan_continuation as continuation


class Phase16FullScanContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records = phase16.derive_execution_orders()["segments"][0]["records"]
        cls.first = dict(records[0])
        cls.second = dict(records[1])

    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="phase16-continuation-test."))
        self.segment = self.temporary / "replicate-0"
        (self.segment / "raw").mkdir(parents=True)
        self.segment_id = "phase16-20260831t000000000000z-12345678-abcdef-replicate-0"
        (self.segment / "segment_manifest.json").write_text(
            json.dumps({"segment_id": self.segment_id}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def _write_manifest(
        self,
        *,
        record: dict[str, object],
        status: str,
        reason: str | None = None,
        replacement_number: int | None = None,
    ) -> Path:
        logical_id = phase16._logical_run_id(self.segment_id, record)
        if replacement_number is None:
            run_id = logical_id
            replacement_of = None
        else:
            run_id = (
                f"{logical_id}{continuation.REPLACEMENT_SUFFIX}"
                f"{replacement_number:03d}"
            )
            replacement_of = logical_id
        root = self.segment / "raw" / run_id
        root.mkdir()
        manifest = phase16._run_manifest(
            family_id="phase16-20260831t000000000000z-12345678-abcdef",
            segment_id=self.segment_id,
            run_id=run_id,
            record=record,
            status=status,
            reason=reason,
            result_path=None,
            replacement_of=replacement_of,
        )
        manifest["logical_record_id"] = logical_id
        (root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return root

    def test_valid_original_is_preserved_and_absent_suffix_is_pending(self) -> None:
        self._write_manifest(record=self.first, status="completed")
        effective, pending, replaced = continuation._effective_and_pending(
            segment_root=self.segment,
            order={"records": [self.first, self.second]},
        )
        self.assertEqual([item["status"] for item in effective], ["completed"])
        self.assertEqual(len(pending), 1)
        self.assertFalse(pending[0]["replace"])
        self.assertEqual(replaced, [])

    def test_infrastructure_failure_gets_explicit_replacement_only(self) -> None:
        original = self._write_manifest(
            record=self.first,
            status="infrastructure_failed",
            reason="preflight_snapshot_unavailable",
        )
        effective, pending, _ = continuation._effective_and_pending(
            segment_root=self.segment, order={"records": [self.first]}
        )
        self.assertEqual(effective, [])
        self.assertEqual(len(pending), 1)
        self.assertTrue(pending[0]["replace"])
        self.assertEqual(pending[0]["replacement_number"], 1)
        self.assertTrue(original.is_dir())
        replacement = self._write_manifest(
            record=self.first, status="completed", replacement_number=1
        )
        effective, pending, replaced = continuation._effective_and_pending(
            segment_root=self.segment, order={"records": [self.first]}
        )
        self.assertEqual([item["run_id"] for item in effective], [replacement.name])
        self.assertEqual(pending, [])
        self.assertEqual(replaced[0]["replacement_run_id"], replacement.name)
        self.assertTrue(original.is_dir())

    def test_valid_original_cannot_be_selectively_rerun(self) -> None:
        self._write_manifest(record=self.first, status="completed")
        self._write_manifest(
            record=self.first, status="completed", replacement_number=1
        )
        with self.assertRaisesRegex(
            continuation.Phase16ContinuationError, "valid original run was rerun"
        ):
            continuation._effective_and_pending(
                segment_root=self.segment, order={"records": [self.first]}
            )

    def test_foreign_process_in_incomplete_attempt_fails_closed(self) -> None:
        logical_id = phase16._logical_run_id(self.segment_id, self.first)
        root = self.segment / "raw" / logical_id
        snapshots = root / "gpu-snapshots" / "preflight"
        snapshots.mkdir(parents=True)
        (snapshots / "attempt-00.classification.json").write_text(
            json.dumps({"state": "foreign_process_detected"}), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            continuation.Phase16ContinuationError, "foreign GPU process"
        ):
            continuation._effective_and_pending(
                segment_root=self.segment, order={"records": [self.first]}
            )

    def test_exact_telemetry_finalization_failure_is_replacement_eligible(self) -> None:
        root = self._write_manifest(
            record=self.first,
            status="runtime_failed",
            reason="supervised_worker_failed",
        )
        (root / "failure.json").write_text(
            json.dumps({"returncode": 1}), encoding="utf-8"
        )
        (root / "worker.stderr.txt").write_text(
            "scripts.phase12_unified_admission.Phase12UnifiedAdmissionError: "
            "telemetry temperature_celsius is unavailable\n",
            encoding="utf-8",
        )
        stages = root / "stage-progress"
        stages.mkdir()
        for name in (
            "10-measurement-completed.json",
            "11-finalization-started.json",
        ):
            (stages / name).write_text("{}", encoding="utf-8")
        postflight = root / "gpu-snapshots" / "postflight"
        postflight.mkdir(parents=True)
        for attempt in range(3):
            (postflight / f"attempt-{attempt:02d}.classification.json").write_text(
                json.dumps({"state": "query_failed"}), encoding="utf-8"
            )
        self.assertTrue(continuation._is_telemetry_finalization_failure(root))
        effective, pending, _ = continuation._effective_and_pending(
            segment_root=self.segment, order={"records": [self.first]}
        )
        self.assertEqual(effective, [])
        self.assertTrue(pending[0]["replace"])

    def test_other_runtime_failure_is_not_retried(self) -> None:
        self._write_manifest(
            record=self.first,
            status="runtime_failed",
            reason="supervised_worker_failed",
        )
        effective, pending, _ = continuation._effective_and_pending(
            segment_root=self.segment, order={"records": [self.first]}
        )
        self.assertEqual(len(effective), 1)
        self.assertEqual(pending, [])

    def test_replacement_identity_keeps_logical_id_and_link(self) -> None:
        logical_id = phase16._logical_run_id(self.segment_id, self.first)
        replacement_id = f"{logical_id}{continuation.REPLACEMENT_SUFFIX}001"
        with continuation._replacement_identity(
            logical_id=logical_id,
            replacement_run_id=replacement_id,
            controller_git_sha="2" * 40,
        ):
            manifest = phase16._run_manifest(
                family_id="phase16-20260831t000000000000z-12345678-abcdef",
                segment_id=self.segment_id,
                run_id=replacement_id,
                record={**self.first, "timing_execution_git_sha": "1" * 40},
                status="completed",
                reason=None,
                result_path="result.json",
            )
        self.assertEqual(manifest["run_id"], replacement_id)
        self.assertEqual(manifest["logical_record_id"], logical_id)
        self.assertEqual(manifest["replacement_of"], logical_id)
        self.assertFalse(manifest["selective_rerun"])

    def test_timing_critical_equivalence_rejects_blob_change(self) -> None:
        with mock.patch.object(
            continuation,
            "_git_blob",
            side_effect=lambda commit, relative: (
                b"same"
                if commit == "1" * 40 or relative != phase16.TIMING_CRITICAL_PATHS[0]
                else b"changed"
            ),
        ):
            with self.assertRaisesRegex(
                continuation.Phase16ContinuationError,
                "timing-critical code changed",
            ):
                continuation.timing_critical_equivalence(
                    timing_git_sha="1" * 40, controller_git_sha="2" * 40
                )


if __name__ == "__main__":
    unittest.main()

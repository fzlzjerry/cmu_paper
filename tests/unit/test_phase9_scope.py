"""Focused tests for the exact Phase 9 scope validator additions."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import validate_phase2


class Phase9ScopeTests(unittest.TestCase):
    def test_phase9_allowlist_is_exact_and_rejects_near_paths(self) -> None:
        expected = {
            "docs/plans/phase9-kvquant-calibration.md",
            "docker/calibration-kvquant.Dockerfile",
            "scripts/phase9_kvquant_calibration.py",
            "scripts/phase9_kvquant_worker.py",
            "scripts/r2_artifact.py",
            "tests/unit/test_phase9_scope.py",
            "tests/schema/test_config_schema.py",
            "tests/unit/test_r2_artifact.py",
            "docs/evidence/phase9/r2-publication.json",
            "docs/phase_reports/phase9-kvquant-calibration.md",
        }
        self.assertLessEqual(expected, validate_phase2.PHASE9_ALLOWED_PATHS)
        for rejected in (
            "docs/plans/phase9-kvquant-calibration-copy.md",
            "docker/calibration-kvquant.Dockerfile.dev",
            "scripts/calibration_framework.py",
            "scripts/r2_artifact_phase9.py",
            "tests/unit/test_r2_artifact_copy.py",
            "tests/schema/test_config_schema_copy.py",
            "reference/kvquant/fixture.json",
            "src/kvbench/adapters/kvquant.py",
            "calibration/kvquant/kvqcal-deadbeef/fisher.safetensors",
            "docs/evidence/phase9/layer_stats.parquet",
        ):
            with self.subTest(rejected=rejected):
                self.assertNotIn(
                    rejected,
                    validate_phase2.PHASE9_ALLOWED_PATHS,
                )

    def test_raw_calibration_payload_suffixes_remain_git_forbidden(self) -> None:
        for suffix in (
            ".parquet",
            ".safetensors",
            ".pt",
            ".pth",
            ".bin",
        ):
            with self.subTest(suffix=suffix):
                self.assertIn(suffix, validate_phase2.RAW_RESULT_SUFFIXES)

    def test_absent_calibration_root_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                validate_phase2,
                "ROOT",
                Path(temporary),
            ):
                self.assertEqual(
                    validate_phase2.validate_phase9_calibration_root(),
                    [],
                )

    def test_completed_exact_calibration_child_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_root = root / "calibration" / "kvquant"
            (calibration_root / ".kvbench-staging").mkdir(
                parents=True,
            )
            (calibration_root / ".kvbench-reservations").mkdir()
            completed = calibration_root / f"kvqcal-{'a' * 32}"
            completed.mkdir()
            for name in (
                "manifest.json",
                "artifact_inventory.json",
                "checksums.sha256",
                "COMPLETE",
            ):
                (completed / name).touch()
            os.chmod(completed, 0o555)
            with mock.patch.object(validate_phase2, "ROOT", root):
                self.assertEqual(
                    validate_phase2.validate_phase9_calibration_root(),
                    [],
                )

    def test_malformed_child_and_incomplete_staging_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_root = root / "calibration" / "kvquant"
            staging = calibration_root / ".kvbench-staging"
            staging.mkdir(parents=True)
            (staging / "attempt.json").touch()
            (calibration_root / "kvqcal-near-match").mkdir()
            with mock.patch.object(validate_phase2, "ROOT", root):
                errors = (
                    validate_phase2.validate_phase9_calibration_root()
                )
            self.assertIn(
                "Phase 9 contains incomplete calibration staging",
                errors,
            )
            self.assertIn(
                "unsafe Phase 9 calibration child: kvqcal-near-match",
                errors,
            )


if __name__ == "__main__":
    unittest.main()

"""Focused exact-path tests for Phase 12E."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest

from scripts import validate_phase2


ROOT = Path(__file__).resolve().parents[2]


class Phase12EScopeTests(unittest.TestCase):
    def test_entry_freezes_phase11r(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE12E_ENTRY_COMMIT,
            "7c7af7cd1efe4a8befa36ceaedb11e2b47733276",
        )
        self.assertTrue(
            validate_phase2.commit_is_ancestor(
                validate_phase2.PHASE12E_ENTRY_COMMIT
            )
        )
        self.assertEqual(
            validate_phase2.current_phase11r_paths(),
            validate_phase2.historical_phase11r_paths(),
        )
        self.assertIn(
            "docs/phase_reports/phase11r-kvquant-measurement-adapter.md",
            validate_phase2.historical_phase11r_paths(),
        )
        self.assertLessEqual(
            validate_phase2.historical_phase11r_paths(),
            validate_phase2.PHASE11R_ALLOWED_PATHS,
        )

    def test_phase12e_allowlist_is_exact(self) -> None:
        expected = {
            "Makefile",
            (
                "docs/decisions/"
                "0028-phase12e-kivi-historical-source-validation.md"
            ),
            "scripts/validate_phase2.py",
            "src/kvbench/runtime/kivi_admission.py",
            "tests/unit/test_phase8_kivi_admission.py",
            (
                "tests/unit/"
                "test_phase12e_kivi_historical_authority.py"
            ),
            "tests/unit/test_phase12e_scope.py",
        }
        self.assertEqual(validate_phase2.PHASE12E_ALLOWED_PATHS, expected)
        self.assertFalse(
            any("*" in path for path in validate_phase2.PHASE12E_ALLOWED_PATHS)
        )

    def test_current_phase12e_segment_is_exactly_scoped(self) -> None:
        current = validate_phase2.current_phase12e_paths()
        required = {
            "Makefile",
            (
                "docs/decisions/"
                "0028-phase12e-kivi-historical-source-validation.md"
            ),
            "scripts/validate_phase2.py",
            "tests/unit/test_phase12e_scope.py",
        }
        self.assertLessEqual(required, current)
        self.assertLessEqual(current, validate_phase2.PHASE12E_ALLOWED_PATHS)
        self.assertEqual(
            current,
            validate_phase2.historical_phase12e_paths(),
        )
        self.assertNotEqual(validate_phase2.changed_paths(), current)

    def test_method_evidence_and_future_work_are_rejected(self) -> None:
        rejected = {
            "src/kvbench/adapters/bf16.py",
            "src/kvbench/adapters/turboquant.py",
            "src/kvbench/adapters/kivi.py",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/kivi_cache.py",
            "scripts/phase8_r2_outer_bundle.py",
            "tests/unit/test_phase8_r2_outer_bundle.py",
            "docs/evidence/phase8/kivi-method-admission.json",
            "docs/evidence/phase8/r2-admission-outer-publication.json",
            "artifacts/phase8/rewritten/manifest.json",
            "reference/kivi/fixtures/k4v4/fixture.json",
            "calibration/kvquant/changed/COMPLETE",
            "docker/measurement.Dockerfile",
            "configs/plans/pilot.yaml",
            "configs/plans/full_scan.yaml",
            "artifacts/phase12/g5/run.json",
            "artifacts/profiler/phase12e/result.json",
            "artifacts/quality/phase12e/result.json",
        }
        self.assertFalse(rejected & validate_phase2.PHASE12E_ALLOWED_PATHS)

    def test_entry_did_not_contain_phase12e_files(self) -> None:
        for relative in (
            (
                "docs/decisions/"
                "0028-phase12e-kivi-historical-source-validation.md"
            ),
            (
                "tests/unit/"
                "test_phase12e_kivi_historical_authority.py"
            ),
            "tests/unit/test_phase12e_scope.py",
        ):
            with self.subTest(relative=relative):
                result = subprocess.run(
                    [
                        "git",
                        "cat-file",
                        "-e",
                        f"{validate_phase2.PHASE12E_ENTRY_COMMIT}:{relative}",
                    ],
                    cwd=ROOT,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()

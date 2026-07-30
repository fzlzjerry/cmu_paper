"""Focused exact-path tests for the Phase 11D CUDA remediation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest

from scripts import validate_phase2


ROOT = Path(__file__).resolve().parents[2]


class Phase11DScopeTests(unittest.TestCase):
    def test_entry_and_historical_phase11_segment_are_frozen(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE11D_ENTRY_COMMIT,
            "69e99389b548e82e65e027cc0ea7b86c9fbe43dd",
        )
        self.assertEqual(
            validate_phase2.current_phase11_paths(),
            validate_phase2.historical_phase11_paths(),
        )
        self.assertIn(
            "src/kvbench/adapters/kvquant.py",
            validate_phase2.historical_phase11_paths(),
        )
        self.assertNotIn(
            (
                "docs/decisions/"
                "0027-kvquant-deterministic-long-context-value-decode.md"
            ),
            validate_phase2.historical_phase11_paths(),
        )
        self.assertLessEqual(
            validate_phase2.historical_phase11_paths(),
            validate_phase2.PHASE11_ALLOWED_PATHS,
        )

    def test_phase11d_allowlist_is_exact(self) -> None:
        expected = {
            "Makefile",
            (
                "docs/decisions/"
                "0027-kvquant-deterministic-long-context-value-decode.md"
            ),
            "docs/evidence/phase11d/cuda-validation.json",
            (
                "docs/phase_reports/"
                "phase11d-kvquant-deterministic-long-context-cuda.md"
            ),
            "scripts/phase11d_kvquant_validation.py",
            "scripts/validate_kvquant_long_context_patch.py",
            "scripts/validate_phase2.py",
            (
                "tests/cuda/"
                "phase11d_kvquant_long_context_validation.py"
            ),
            "tests/unit/test_phase11d_scope.py",
            "tests/unit/test_phase9p_patch_custody.py",
            (
                "third_party/patches/kvquant/"
                "0003-deterministic-long-context-value-decode.patch"
            ),
            (
                "third_party/patches/kvquant/"
                "deterministic-long-context-manifest.json"
            ),
        }
        self.assertEqual(
            validate_phase2.PHASE11D_ALLOWED_PATHS,
            expected,
        )
        self.assertFalse(
            any(
                "*" in path
                for path in validate_phase2.PHASE11D_ALLOWED_PATHS
            )
        )

    def test_adapter_fixture_calibration_and_campaign_paths_are_rejected(
        self,
    ) -> None:
        rejected = {
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/kvquant_cache.py",
            (
                "reference/kvquant_phase11pr/fixtures/kvq4/"
                "key_cap_value_fixed12/dense_payload.safetensors"
            ),
            (
                "calibration/kvquant/"
                "kvqcal-cdb724c806d64d095c040d2673a987a3/"
                "COMPLETE"
            ),
            "scripts/phase11_kvquant_admission.py",
            "scripts/phase11_r2_outer_bundle.py",
            "scripts/r2_artifact_phase11d.py",
            "docs/evidence/phase11d/r2-publication.json",
            "tests/cuda/phase11d_benchmark_grid.py",
            "artifacts/profiler/phase11d/result.json",
            "artifacts/quality/phase11d/result.json",
            "docs/quality/phase11d.md",
        }
        self.assertFalse(
            rejected & validate_phase2.PHASE11D_ALLOWED_PATHS
        )

    def test_current_phase11d_segment_includes_untracked_files(self) -> None:
        current = validate_phase2.current_phase11d_paths()
        self.assertIn("tests/unit/test_phase11d_scope.py", current)
        self.assertIn(
            "tests/cuda/phase11d_kvquant_long_context_validation.py",
            current,
        )
        self.assertLessEqual(
            current,
            validate_phase2.PHASE11D_ALLOWED_PATHS,
        )

    def test_phase11d_entry_did_not_contain_remediation_files(self) -> None:
        for relative in (
            (
                "docs/decisions/"
                "0027-kvquant-deterministic-long-context-value-decode.md"
            ),
            (
                "third_party/patches/kvquant/"
                "0003-deterministic-long-context-value-decode.patch"
            ),
            (
                "third_party/patches/kvquant/"
                "deterministic-long-context-manifest.json"
            ),
        ):
            with self.subTest(relative=relative):
                observed = subprocess.run(
                    [
                        "git",
                        "cat-file",
                        "-e",
                        f"{validate_phase2.PHASE11D_ENTRY_COMMIT}:{relative}",
                    ],
                    cwd=ROOT,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.assertNotEqual(observed.returncode, 0)


if __name__ == "__main__":
    unittest.main()

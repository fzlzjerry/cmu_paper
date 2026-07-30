"""Focused exact-path tests for the Phase 11R admission rerun."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest

from scripts import validate_phase2


ROOT = Path(__file__).resolve().parents[2]


class Phase11RScopeTests(unittest.TestCase):
    def test_phase11r_boundary_freezes_phase11d(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE11R_ENTRY_COMMIT,
            "f0f02364a556da70e67b3107a0c0afad5f75eae9",
        )
        self.assertTrue(
            validate_phase2.commit_is_ancestor(
                validate_phase2.PHASE11R_ENTRY_COMMIT
            )
        )
        self.assertEqual(
            validate_phase2.current_phase11d_paths(),
            validate_phase2.historical_phase11d_paths(),
        )
        historical = validate_phase2.historical_phase11d_paths()
        self.assertIn(
            (
                "docs/phase_reports/"
                "phase11d-kvquant-deterministic-long-context-cuda.md"
            ),
            historical,
        )
        self.assertIn(
            (
                "third_party/patches/kvquant/"
                "0003-deterministic-long-context-value-decode.patch"
            ),
            historical,
        )
        self.assertLessEqual(
            historical,
            validate_phase2.PHASE11D_ALLOWED_PATHS,
        )

    def test_phase11r_allowlist_is_exact(self) -> None:
        expected = {
            "Makefile",
            "docs/evidence/phase11/kvquant-method-admission.json",
            "docs/evidence/phase11/kvquant-method-admission.sha256",
            "docs/evidence/phase11/r2-admission-outer-publish.stderr.txt",
            "docs/evidence/phase11/r2-admission-outer-publish.stdout.json",
            "docs/evidence/phase11/r2-admission-outer-publication.json",
            "docs/evidence/phase11/r2-admission-outer-verify.stderr.txt",
            "docs/evidence/phase11/r2-admission-outer-verify.stdout.json",
            "docs/evidence/phase11/r2-admission-publish.stderr.txt",
            "docs/evidence/phase11/r2-admission-publish.stdout.json",
            "docs/evidence/phase11/r2-admission-publication.json",
            "docs/evidence/phase11/r2-admission-verify.stderr.txt",
            "docs/evidence/phase11/r2-admission-verify.stdout.json",
            "docs/method_notes/kvquant.md",
            (
                "docs/phase_reports/"
                "phase11r-kvquant-measurement-adapter.md"
            ),
            "docs/plans/phase11r-kvquant-admission-rerun.md",
            "docs/risk_register.md",
            "docs/status.md",
            "docs/tasks.md",
            "scripts/phase11_kvquant_admission.py",
            "scripts/phase11_r2_outer_bundle.py",
            "scripts/validate_kvquant_long_context_patch.py",
            "scripts/validate_phase2.py",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/kvquant_cache.py",
            "src/kvbench/runtime/kvquant_session.py",
            "src/kvbench/schema/phase11.py",
            "tests/cuda/phase11_kvquant_sanitizer_probe.py",
            "tests/cuda/test_phase11_kvquant_cuda.py",
            "tests/graph/test_phase11_kvquant_graph.py",
            "tests/unit/test_phase11_kvquant_adapter.py",
            "tests/unit/test_phase11_kvquant_admission.py",
            "tests/unit/test_phase11_kvquant_admission_driver.py",
            "tests/unit/test_phase11_kvquant_cache.py",
            "tests/unit/test_phase11_kvquant_factory.py",
            "tests/unit/test_phase11_kvquant_session.py",
            "tests/unit/test_phase11_make_targets.py",
            "tests/unit/test_phase11_r2_outer_bundle.py",
            "tests/unit/test_phase11r_scope.py",
        }
        self.assertEqual(validate_phase2.PHASE11R_ALLOWED_PATHS, expected)
        self.assertFalse(
            any("*" in path for path in validate_phase2.PHASE11R_ALLOWED_PATHS)
        )

    def test_current_phase11r_segment_includes_untracked_files(self) -> None:
        current = validate_phase2.current_phase11r_paths()
        self.assertIn("scripts/validate_phase2.py", current)
        self.assertIn("tests/unit/test_phase11r_scope.py", current)
        self.assertLessEqual(
            current,
            validate_phase2.PHASE11R_ALLOWED_PATHS,
        )

    def test_immutable_and_out_of_scope_paths_are_rejected(self) -> None:
        rejected = {
            "docs/phase_reports/phase11-kvquant-measurement-adapter.md",
            "docs/plans/phase11-kvquant-measurement-adapter.md",
            "docs/evidence/phase11d/cuda-validation.json",
            (
                "docs/phase_reports/"
                "phase11d-kvquant-deterministic-long-context-cuda.md"
            ),
            "docs/decisions/0016-phase6a-measurement-container.md",
            "docs/decisions/0021-kvquant-patch-main-repository-custody.md",
            (
                "docs/decisions/"
                "0023-phase10-kvquant-source-faithful-sparse-fixture-semantics.md"
            ),
            (
                "docs/decisions/"
                "0024-kvquant-graph-safe-caller-owned-cuda-apis.md"
            ),
            "docs/decisions/0025-kvquant-deterministic-kvq3-value-pack.md",
            "docs/decisions/0026-kvquant-pre-rope-adapter-boundary.md",
            (
                "docs/decisions/"
                "0027-kvquant-deterministic-long-context-value-decode.md"
            ),
            (
                "third_party/patches/kvquant/"
                "0001-llama31-native-gqa.patch"
            ),
            (
                "third_party/patches/kvquant/"
                "0002-graphsafe-kvq3-deterministic.patch"
            ),
            (
                "third_party/patches/kvquant/"
                "0003-deterministic-long-context-value-decode.patch"
            ),
            "third_party/patches/kvquant/manifest.json",
            (
                "third_party/patches/kvquant/"
                "graphsafe-kvq3-manifest.json"
            ),
            (
                "third_party/patches/kvquant/"
                "deterministic-long-context-manifest.json"
            ),
            (
                "reference/kvquant_phase11pr/fixtures/kvq4/"
                "key_cap_value_fixed12/dense_payload.safetensors"
            ),
            (
                "calibration/kvquant/"
                "kvqcal-cdb724c806d64d095c040d2673a987a3/"
                "COMPLETE"
            ),
            "src/kvbench/adapters/bf16.py",
            "src/kvbench/adapters/turboquant.py",
            "src/kvbench/adapters/kivi.py",
            "src/kvbench/adapters/factory.py",
            "src/kvbench/runtime/kvquant_fixture.py",
            "docker/measurement.Dockerfile",
            "docs/blockers.md",
            "configs/plans/pilot.yaml",
            "configs/plans/full_scan.yaml",
            "scripts/r2_artifact_phase11r.py",
            "artifacts/phase11/rewritten-run/manifest.json",
            "artifacts/profiler/phase11r/result.json",
            "artifacts/quality/phase11r/result.json",
            "docs/quality/phase11r.md",
        }
        self.assertFalse(rejected & validate_phase2.PHASE11R_ALLOWED_PATHS)

    def test_phase11r_entry_did_not_contain_new_report_or_scope_test(
        self,
    ) -> None:
        for relative in (
            (
                "docs/phase_reports/"
                "phase11r-kvquant-measurement-adapter.md"
            ),
            "docs/plans/phase11r-kvquant-admission-rerun.md",
            "tests/unit/test_phase11r_scope.py",
        ):
            with self.subTest(relative=relative):
                observed = subprocess.run(
                    [
                        "git",
                        "cat-file",
                        "-e",
                        f"{validate_phase2.PHASE11R_ENTRY_COMMIT}:{relative}",
                    ],
                    cwd=ROOT,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.assertNotEqual(observed.returncode, 0)


if __name__ == "__main__":
    unittest.main()

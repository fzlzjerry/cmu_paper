"""Focused exact-path and preservation tests for Phase 11R-Q23."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import unittest

from scripts import validate_phase2


ROOT = Path(__file__).resolve().parents[2]


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


class Phase11RQ23ScopeTests(unittest.TestCase):
    def test_entry_freezes_completed_phase11dq23(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE11RQ23_ENTRY_COMMIT,
            "d99920e5dd7ea94bce7c98b4301bd035c073dfea",
        )
        self.assertTrue(
            validate_phase2.commit_is_ancestor(
                validate_phase2.PHASE11RQ23_ENTRY_COMMIT
            )
        )
        self.assertEqual(
            validate_phase2.current_phase11dq23_paths(),
            validate_phase2.historical_phase11dq23_paths(),
        )
        self.assertLessEqual(
            validate_phase2.historical_phase11dq23_paths(),
            validate_phase2.PHASE11DQ23_ALLOWED_PATHS,
        )

    def test_phase11rq23_allowlist_is_exact(self) -> None:
        expected = {
            "Makefile",
            "docs/evidence/phase11rq23/kvquant-method-admission.json",
            "docs/evidence/phase11rq23/kvquant-method-admission.sha256",
            (
                "docs/evidence/phase11rq23/"
                "r2-admission-outer-publish.stderr.txt"
            ),
            (
                "docs/evidence/phase11rq23/"
                "r2-admission-outer-publish.stdout.json"
            ),
            (
                "docs/evidence/phase11rq23/"
                "r2-admission-outer-publication.json"
            ),
            (
                "docs/evidence/phase11rq23/"
                "r2-admission-outer-verify.stderr.txt"
            ),
            (
                "docs/evidence/phase11rq23/"
                "r2-admission-outer-verify.stdout.json"
            ),
            "docs/evidence/phase11rq23/r2-admission-publish.stderr.txt",
            "docs/evidence/phase11rq23/r2-admission-publish.stdout.json",
            "docs/evidence/phase11rq23/r2-admission-publication.json",
            "docs/evidence/phase11rq23/r2-admission-verify.stderr.txt",
            "docs/evidence/phase11rq23/r2-admission-verify.stdout.json",
            "docs/method_notes/kvquant.md",
            (
                "docs/phase_reports/"
                "phase11rq23-kvquant-measurement-adapter.md"
            ),
            "docs/plans/phase11rq23-kvquant-admission-rerun.md",
            "docs/risk_register.md",
            "docs/status.md",
            "docs/tasks.md",
            "scripts/phase11_kvquant_admission.py",
            "scripts/phase11_r2_outer_bundle.py",
            "scripts/validate_phase2.py",
            "src/kvbench/schema/__init__.py",
            "src/kvbench/schema/phase11.py",
            "tests/unit/test_phase11rq23_kvquant_admission.py",
            "tests/unit/test_phase11rq23_kvquant_admission_driver.py",
            "tests/unit/test_phase11rq23_make_targets.py",
            "tests/unit/test_phase11rq23_r2_outer_bundle.py",
            "tests/unit/test_phase11rq23_r2_outer_profile.py",
            "tests/unit/test_phase11rq23_schema.py",
            "tests/unit/test_phase11rq23_scope.py",
        }
        self.assertEqual(validate_phase2.PHASE11RQ23_ALLOWED_PATHS, expected)
        self.assertFalse(any("*" in path for path in expected))

    def test_current_segment_is_exactly_scoped(self) -> None:
        current = validate_phase2.current_phase11rq23_paths()
        required = {
            "docs/plans/phase11rq23-kvquant-admission-rerun.md",
            "scripts/validate_phase2.py",
            "tests/unit/test_phase11rq23_scope.py",
        }
        self.assertLessEqual(required, current)
        self.assertLessEqual(
            current,
            validate_phase2.PHASE11RQ23_ALLOWED_PATHS,
        )

    def test_implementations_historical_evidence_and_phase12_are_rejected(
        self,
    ) -> None:
        rejected = {
            "docs/blockers.md",
            "docs/decisions/0021-kvquant-patch-main-repository-custody.md",
            (
                "docs/decisions/"
                "0029-kvquant-deterministic-long-context-q3-q2-"
                "value-decode.md"
            ),
            "docs/evidence/phase11/kvquant-method-admission.json",
            "docs/evidence/phase11/kvquant-method-admission.sha256",
            "docs/evidence/phase11/r2-admission-publication.json",
            "docs/evidence/phase11/r2-admission-outer-publication.json",
            "docs/evidence/phase11dq23/cuda-validation.json",
            (
                "docs/phase_reports/"
                "phase11r-kvquant-measurement-adapter.md"
            ),
            (
                "docs/phase_reports/"
                "phase11dq23-kvquant-deterministic-long-context-q3-q2.md"
            ),
            "src/kvbench/adapters/bf16.py",
            "src/kvbench/adapters/turboquant.py",
            "src/kvbench/adapters/kivi.py",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/kvquant_cache.py",
            "src/kvbench/runtime/kvquant_session.py",
            "tests/cuda/phase11_kvquant_sanitizer_probe.py",
            "tests/cuda/test_phase11_kvquant_cuda.py",
            "tests/graph/test_phase11_kvquant_graph.py",
            "tests/unit/test_phase11_kvquant_admission.py",
            "tests/unit/test_phase11_kvquant_admission_driver.py",
            "tests/unit/test_phase11_make_targets.py",
            "tests/unit/test_phase11_r2_outer_bundle.py",
            (
                "third_party/patches/kvquant/"
                "0004-deterministic-long-context-q3-q2-value-decode.patch"
            ),
            (
                "third_party/patches/kvquant/"
                "deterministic-long-context-q3-q2-manifest.json"
            ),
            (
                "reference/kvquant_phase11pr/fixtures/kvq3/"
                "key_cap_value_fixed12/dense_payload.safetensors"
            ),
            (
                "calibration/kvquant/"
                "kvqcal-cdb724c806d64d095c040d2673a987a3/COMPLETE"
            ),
            "configs/methods/kvquant.yaml",
            "docker/measurement.Dockerfile",
            "artifacts/phase12/new-campaign/manifest.json",
            "docs/evidence/phase12/unified-admission.json",
            "scripts/phase12_unified_admission.py",
            "artifacts/profiler/phase11rq23/result.json",
            "artifacts/quality/phase11rq23/result.json",
            "docs/quality/phase11rq23.md",
        }
        self.assertFalse(
            rejected & validate_phase2.PHASE11RQ23_ALLOWED_PATHS
        )

    def test_entry_did_not_contain_new_phase_paths(self) -> None:
        for relative in (
            "docs/plans/phase11rq23-kvquant-admission-rerun.md",
            (
                "docs/phase_reports/"
                "phase11rq23-kvquant-measurement-adapter.md"
            ),
            "docs/evidence/phase11rq23/kvquant-method-admission.json",
            "tests/unit/test_phase11rq23_scope.py",
        ):
            with self.subTest(relative=relative):
                observed = subprocess.run(
                    [
                        "git",
                        "cat-file",
                        "-e",
                        (
                            f"{validate_phase2.PHASE11RQ23_ENTRY_COMMIT}:"
                            f"{relative}"
                        ),
                    ],
                    cwd=ROOT,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.assertNotEqual(observed.returncode, 0)

    def test_historical_and_q23_authorities_remain_byte_exact(self) -> None:
        expected = {
            (
                "docs/phase_reports/"
                "phase11r-kvquant-measurement-adapter.md"
            ): "bf24d235cb3f1fa80fbe53ae40690c3e4b0451a3b6038eb50b7b60cde19c8b64",
            (
                "docs/evidence/phase11/"
                "kvquant-method-admission.json"
            ): "59ef5bfc581a68cdc4d21c4c0a840f046e698633f7475f79906063c6e333ae6a",
            (
                "docs/evidence/phase11/"
                "r2-admission-outer-publication.json"
            ): "2d4fcbc05a5e39a11b4c0382ce700f61122bebab417f3fe31447240ce942b42c",
            (
                "docs/decisions/"
                "0029-kvquant-deterministic-long-context-q3-q2-"
                "value-decode.md"
            ): "28245640635b5b1e2e28aaca5728a8030f83596dc72d3676891802057468a0bd",
            (
                "docs/evidence/phase11dq23/cuda-validation.json"
            ): "04759580cf6ddbd6d5108f5069058ce71994a12c0ce6b951b36093ab222b934c",
            (
                "docs/phase_reports/"
                "phase11dq23-kvquant-deterministic-long-context-q3-q2.md"
            ): "1100063f6ee2383e0c7a0d98446df413f90d9599d6b625a29a329cde9cbd8d9e",
            (
                "third_party/patches/kvquant/"
                "0004-deterministic-long-context-q3-q2-value-decode.patch"
            ): "7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a",
            (
                "src/kvbench/adapters/kvquant.py"
            ): "6fd5f73f7af3ef5c6d5accd16bc241780b7d679bdf5b98d1396a85d26965af18",
            (
                "src/kvbench/runtime/kvquant_cache.py"
            ): "5819e9fb716bae9f172f84d82367d3648c1492ecde66da9bd9b9dacf3b49c4c0",
            (
                "src/kvbench/runtime/kvquant_session.py"
            ): "088a4212b36e8f24ef4a7ee8d4a6e0487fc9f3eb61ac9582c5f5551d46d4c836",
            (
                "reference/kvquant_phase11pr/fixtures/manifest.json"
            ): "60950cac4c7aa107552260e16e6917fbfac275882dd8b45d56116ce3da214049",
            (
                "reference/kvquant_phase11pr/fixtures/reuse_proof.json"
            ): "3fc25acb331cac7f4f40a0b8f6b52c5d0d73cffdd562ef33c06f5050c50f689d",
        }
        for relative, digest in expected.items():
            with self.subTest(relative=relative):
                self.assertEqual(_sha256(relative), digest)

    def test_stopped_phase12_staging_remains_frozen(self) -> None:
        self.assertEqual(
            validate_phase2.validate_phase12_blocked_artifact_root(),
            [],
        )


if __name__ == "__main__":
    unittest.main()

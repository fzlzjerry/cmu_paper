"""Focused exact-path and authority tests for Phase 11D-Q23."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts import phase11dq23_kvquant_validation
from scripts import validate_phase2


ROOT = Path(__file__).resolve().parents[2]


class Phase11DQ23ScopeTests(unittest.TestCase):
    def test_entry_freezes_completed_phase12e(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE11DQ23_ENTRY_COMMIT,
            "2bc6aaa1d05b08d50f4c01bbc0b2863dd8689fe1",
        )
        self.assertTrue(
            validate_phase2.commit_is_ancestor(
                validate_phase2.PHASE11DQ23_ENTRY_COMMIT
            )
        )
        self.assertEqual(
            validate_phase2.current_phase12e_paths(),
            validate_phase2.historical_phase12e_paths(),
        )
        self.assertLessEqual(
            validate_phase2.historical_phase12e_paths(),
            validate_phase2.PHASE12E_ALLOWED_PATHS,
        )

    def test_phase11dq23_allowlist_is_exact(self) -> None:
        expected = {
            "Makefile",
            (
                "docs/decisions/"
                "0029-kvquant-deterministic-long-context-q3-q2-"
                "value-decode.md"
            ),
            "docs/evidence/phase11dq23/cuda-validation.json",
            (
                "docs/phase_reports/"
                "phase11dq23-kvquant-deterministic-long-context-q3-q2.md"
            ),
            "scripts/phase11dq23_kvquant_validation.py",
            "scripts/validate_kvquant_q23_long_context_patch.py",
            "scripts/validate_phase2.py",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/kvquant_cache.py",
            "src/kvbench/runtime/kvquant_session.py",
            "tests/cuda/phase11_kvquant_sanitizer_probe.py",
            (
                "tests/cuda/"
                "phase11dq23_kvquant_long_context_validation.py"
            ),
            "tests/cuda/test_phase11_kvquant_cuda.py",
            "tests/graph/test_phase11_kvquant_graph.py",
            "tests/unit/test_phase11_kvquant_adapter.py",
            "tests/unit/test_phase11_kvquant_cache.py",
            "tests/unit/test_phase11_kvquant_session.py",
            "tests/unit/test_phase11dq23_scope.py",
            "tests/unit/test_phase12e_scope.py",
            "tests/unit/test_phase9p_patch_custody.py",
            (
                "third_party/patches/kvquant/"
                "0004-deterministic-long-context-q3-q2-value-decode.patch"
            ),
            (
                "third_party/patches/kvquant/"
                "deterministic-long-context-q3-q2-manifest.json"
            ),
        }
        self.assertEqual(validate_phase2.PHASE11DQ23_ALLOWED_PATHS, expected)
        self.assertFalse(any("*" in path for path in expected))

    def test_current_segment_is_exactly_scoped(self) -> None:
        current = validate_phase2.current_phase11dq23_paths()
        required = {
            "Makefile",
            (
                "docs/decisions/"
                "0029-kvquant-deterministic-long-context-q3-q2-"
                "value-decode.md"
            ),
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/kvquant_cache.py",
            "tests/cuda/phase11dq23_kvquant_long_context_validation.py",
            "tests/unit/test_phase11dq23_scope.py",
        }
        self.assertLessEqual(required, current)
        self.assertLessEqual(
            current,
            validate_phase2.PHASE11DQ23_ALLOWED_PATHS,
        )
        self.assertEqual(validate_phase2.changed_paths(), current)

    def test_failed_phase12_staging_is_exactly_frozen(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE12_BLOCKED_ARTIFACT_ROOT_NAMES,
            {"phase12"},
        )
        self.assertEqual(
            validate_phase2.validate_phase12_blocked_artifact_root(),
            [],
        )
        self.assertEqual(
            len(validate_phase2.PHASE12_BLOCKED_STAGING_FILE_SHA256S),
            3,
        )

    def test_future_campaign_and_method_changes_are_rejected(self) -> None:
        rejected = {
            "src/kvbench/adapters/bf16.py",
            "src/kvbench/adapters/turboquant.py",
            "src/kvbench/adapters/kivi.py",
            "reference/kvquant_phase11pr/fixtures/kvq3/changed.bin",
            "calibration/kvquant/changed/COMPLETE",
            "docker/measurement.Dockerfile",
            "artifacts/phase12/g5/run.json",
            "docs/evidence/phase12/unified-admission.json",
            "scripts/phase12_unified_admission.py",
            "artifacts/profiler/phase11dq23/result.json",
            "artifacts/quality/phase11dq23/result.json",
        }
        self.assertFalse(
            rejected & validate_phase2.PHASE11DQ23_ALLOWED_PATHS
        )

    def test_manifest_and_make_target_bind_exact_identities(self) -> None:
        manifest = json.loads(
            (
                ROOT
                / "third_party/patches/kvquant/"
                "deterministic-long-context-q3-q2-manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["source"]["patched_commit"],
            "34b0bdfa83082e1f30387d9ac5cca369006e089c",
        )
        self.assertEqual(
            manifest["source"]["patched_tree"],
            "1f85af65fe03061583ffe8bd91e47d7ecffdd312",
        )
        self.assertEqual(
            manifest["patch"]["sha256"],
            "7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a",
        )
        self.assertEqual(
            manifest["extension"]["sha256"],
            "b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d",
        )
        self.assertEqual(
            manifest["validation"]["evidence_sha256"],
            "04759580cf6ddbd6d5108f5069058ce71994a12c0ce6b951b36093ab222b934c",
        )
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("validate-kvquant-phase11dq23:", makefile)
        self.assertIn(
            '/usr/bin/strip --strip-unneeded "$$extension"',
            makefile,
        )
        self.assertIn("--network=none", makefile)
        self.assertIn(
            "scripts/phase11dq23_kvquant_validation.py",
            makefile,
        )
        self.assertIn(
            "dst=/opt/kvquant-calibration,readonly",
            makefile,
        )
        self.assertIn(
            "--calibration-root /opt/kvquant-calibration",
            makefile,
        )

    def test_multiline_source_validator_json_is_parsed(self) -> None:
        payload = b'{\n  "status": "PASS",\n  "value": 1\n}\n'
        self.assertEqual(
            phase11dq23_kvquant_validation._last_json(payload),
            {"status": "PASS", "value": 1},
        )

    def test_mha_gqa_regression_uses_exact_existing_cuda_tests(self) -> None:
        expected = {
            (
                "tests_phase9p.test_deployment_gqa.DeploymentCudaTests."
                "test_4_3_2_bit_native_gqa_matches_explicit_repeat_reference"
            ),
            (
                "tests_phase9p.test_deployment_gqa.DeploymentCudaTests."
                "test_mha_groups_one_matches_direct_unpacked_4_3_2_bit_reference"
            ),
            (
                "tests_phase9p.test_deployment_gqa.DeploymentCudaTests."
                "test_cap_reached_sparse_4_3_2_bit_gqa_matches_native_kv_reference"
            ),
            (
                "tests_phase9p.test_deployment_gqa.DeploymentCudaTests."
                "test_all_changed_kernels_capture_and_allocate_nothing"
            ),
        }
        self.assertEqual(
            set(
                phase11dq23_kvquant_validation.EXISTING_MHA_GQA_CUDA_TESTS
            ),
            expected,
        )
        self.assertNotIn(
            (
                "tests_phase9p.test_deployment_gqa.DeploymentCudaTests."
                "test_python_eager_prefill_gqa_and_mha_match_explicit_repeat"
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()

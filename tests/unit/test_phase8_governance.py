"""Focused Phase 8 scope and hot-path governance regressions."""

from __future__ import annotations

from pathlib import Path
import unittest

from scripts.validate_phase2 import (
    HOT_PATH_FUNCTIONS,
    PHASE7_ALLOWED_PATHS,
    PHASE8_ALLOWED_PATHS,
    PHASE8_APPROVED_ARTIFACT_ROOT_NAMES,
    PHASE8_ENTRY_COMMIT,
    PHASE8_HOT_PATH_SOURCES,
    current_phase8_paths,
    historical_phase7_paths,
    repository_python_paths,
)


class Phase8GovernanceTests(unittest.TestCase):
    def test_entry_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            PHASE8_ENTRY_COMMIT,
            "8d6d766a34a15bd40bd42cc47c5482b0dd052cc0",
        )
        expected = frozenset(
            {
                "Makefile",
                "docs/blockers.md",
                "docs/decisions/0019-phase7-allocation-ratio-terminology-erratum.md",
                "docs/evidence/phase8/kivi-method-admission.json",
                "docs/evidence/phase8/kivi-method-admission.sha256",
                "docs/evidence/phase8/r2-admission-outer-publication.json",
                "docs/evidence/phase8/r2-admission-publication.json",
                "docs/method_notes/kivi.md",
                "docs/phase_reports/phase8-kivi-measurement-adapter.md",
                "docs/plans/phase8-kivi-measurement-adapter.md",
                "docs/risk_register.md",
                "docs/status.md",
                "docs/tasks.md",
                "scripts/phase8_kivi_admission.py",
                "scripts/phase8_r2_outer_bundle.py",
                "scripts/validate_phase2.py",
                "src/kvbench/adapters/__init__.py",
                "src/kvbench/adapters/factory.py",
                "src/kvbench/adapters/kivi.py",
                "src/kvbench/runtime/allocation.py",
                "src/kvbench/runtime/allocation_attribution.py",
                "src/kvbench/runtime/kivi_admission.py",
                "src/kvbench/runtime/kivi_allocation.py",
                "src/kvbench/runtime/artifacts.py",
                "src/kvbench/runtime/kivi_cache.py",
                "src/kvbench/runtime/kivi_fixture.py",
                "src/kvbench/runtime/numerical.py",
                "src/kvbench/runtime/kivi_session.py",
                "src/kvbench/runtime/process_supervision.py",
                "src/kvbench/schema/phase8.py",
                "tests/cuda/phase8_kivi_sanitizer_probe.py",
                "tests/cuda/test_phase8_kivi_cuda.py",
                "tests/graph/test_phase8_kivi_graph.py",
                "tests/unit/test_phase7_kivi_b019_remediation.py",
                "tests/unit/test_phase7_kivi_reference.py",
                "tests/unit/test_phase7_kivi_source_audit.py",
                "tests/unit/test_phase8_governance.py",
                "tests/unit/test_phase8_artifacts.py",
                "tests/unit/test_phase8_kivi_adapter.py",
                "tests/unit/test_phase8_kivi_admission.py",
                "tests/unit/test_phase8_kivi_admission_driver.py",
                "tests/unit/test_phase8_kivi_allocation.py",
                "tests/unit/test_phase8_kivi_cache.py",
                "tests/unit/test_phase8_kivi_fixture.py",
                "tests/unit/test_phase8_kivi_schema.py",
                "tests/unit/test_phase8_kivi_session.py",
                "tests/unit/test_phase8_make_targets.py",
                "tests/unit/test_phase8_process_supervision.py",
                "tests/unit/test_phase8_r2_outer_bundle.py",
                "tests/unit/test_phase8_ratio_terminology.py",
            }
        )
        self.assertEqual(PHASE8_ALLOWED_PATHS, expected)

    def test_phase8_rejects_deferred_or_historical_changes(self) -> None:
        for rejected in (
            "configs/methods/kivi.yaml",
            "docker/measurement.Dockerfile",
            "docs/phase_reports/phase7-kivi-reference.md",
            "reference/kivi/fixtures/k4v4/fixture.json",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/adapters/turboquant.py",
            "third_party/LOCK.json",
            "third_party/NOTICE.md",
        ):
            with self.subTest(rejected=rejected):
                self.assertNotIn(rejected, PHASE8_ALLOWED_PATHS)

    def test_phase7_history_and_phase8_current_state_are_separate(self) -> None:
        self.assertLessEqual(historical_phase7_paths(), PHASE7_ALLOWED_PATHS)
        self.assertLessEqual(current_phase8_paths(), PHASE8_ALLOWED_PATHS)

    def test_kivi_hot_path_names_and_conditional_sources_are_exact(self) -> None:
        self.assertEqual(
            PHASE8_HOT_PATH_SOURCES,
            frozenset(
                {
                    "src/kvbench/adapters/kivi.py",
                    "src/kvbench/runtime/kivi_cache.py",
                }
            ),
        )
        self.assertEqual(
            HOT_PATH_FUNCTIONS["src/kvbench/adapters/kivi.py"],
            {
                "_commit_token",
                "_layer_context",
                "_quantize_into",
                "_store_historical_k",
                "_store_historical_v",
                "store_prefill",
                "append_decode",
                "_decode_compressed",
                "decode_attention",
                "launch_into",
            },
        )
        self.assertEqual(
            HOT_PATH_FUNCTIONS["src/kvbench/runtime/kivi_cache.py"],
            {"update"},
        )

    def test_phase8_driver_outer_graph_and_artifact_roots_are_governed(
        self,
    ) -> None:
        repository = Path(__file__).parents[2]
        governed = {
            path.relative_to(repository).as_posix()
            for path in repository_python_paths()
        }
        self.assertTrue(
            {
                "scripts/phase8_kivi_admission.py",
                "scripts/phase8_r2_outer_bundle.py",
                "tests/graph/test_phase8_kivi_graph.py",
                "tests/unit/test_phase8_kivi_admission_driver.py",
                "tests/unit/test_phase8_r2_outer_bundle.py",
            }.issubset(governed)
        )
        self.assertEqual(
            PHASE8_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase8", "phase8_r2_outer"}),
        )


if __name__ == "__main__":
    unittest.main()

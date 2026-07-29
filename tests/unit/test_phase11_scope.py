"""Focused exact-path and preservation tests for Phase 11."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from scripts import validate_phase2


ROOT = Path(__file__).resolve().parents[2]


class Phase11ScopeTests(unittest.TestCase):
    def test_entry_and_current_segment_are_separate(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE11_ENTRY_COMMIT,
            "72f1897af78b738cc8c74fd335a8957a8e8f5d6c",
        )
        self.assertEqual(
            validate_phase2.current_phase11pr_paths(),
            validate_phase2.historical_phase11pr_paths(),
        )
        self.assertNotIn(
            "docs/plans/phase11-kvquant-measurement-adapter.md",
            validate_phase2.current_phase11pr_paths(),
        )
        self.assertIn(
            "docs/plans/phase11-kvquant-measurement-adapter.md",
            validate_phase2.current_phase11_paths(),
        )

    def test_allowlist_is_exact_and_small(self) -> None:
        required = {
            "docs/decisions/0026-kvquant-pre-rope-adapter-boundary.md",
            "docs/evidence/phase11/r2-admission-outer-publish.stderr.txt",
            "docs/evidence/phase11/r2-admission-outer-publish.stdout.json",
            "docs/evidence/phase11/r2-admission-outer-verify.stderr.txt",
            "docs/evidence/phase11/r2-admission-outer-verify.stdout.json",
            "docs/evidence/phase11/r2-admission-publish.stderr.txt",
            "docs/evidence/phase11/r2-admission-publish.stdout.json",
            "docs/evidence/phase11/r2-admission-verify.stderr.txt",
            "docs/evidence/phase11/r2-admission-verify.stdout.json",
            "docs/plans/phase11-kvquant-measurement-adapter.md",
            "src/kvbench/adapters/base.py",
            "src/kvbench/adapters/factory.py",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/allocation_attribution.py",
            "src/kvbench/runtime/bf16_endpoint.py",
            "src/kvbench/runtime/kvquant_cache.py",
            "src/kvbench/runtime/kvquant_session.py",
            "scripts/phase11_kvquant_admission.py",
            "tests/cuda/test_phase11_kvquant_cuda.py",
            "tests/graph/test_phase11_kvquant_graph.py",
            "tests/unit/test_phase11_scope.py",
        }
        self.assertLessEqual(required, validate_phase2.PHASE11_ALLOWED_PATHS)
        rejected = {
            "docker/measurement.Dockerfile",
            "src/kvbench/methods/kvquant/framework.py",
            "src/kvbench/runtime/kvquant.py",
            "src/kvbench/runtime/kvquant_scheduler.py",
            "scripts/r2_artifact_phase11.py",
            "tests/cuda/phase11_benchmark_grid.py",
            "artifacts/profiler/phase11/result.json",
            "artifacts/quality/phase11/result.json",
            "reference/kvquant_phase11pr/fixtures/kvq3/key_zero_value_fixed12/copy.safetensors",
        }
        self.assertFalse(
            rejected & validate_phase2.PHASE11_ALLOWED_PATHS
        )
        prohibited_or_unused = {
            "configs/methods/kvquant.yaml",
            "docs/blockers.md",
            "src/kvbench/runtime/kvquant_admission.py",
            "src/kvbench/runtime/kvquant_allocation.py",
            "tests/unit/test_phase11_artifacts.py",
            "tests/unit/test_phase11_governance.py",
        }
        self.assertFalse(
            prohibited_or_unused & validate_phase2.PHASE11_ALLOWED_PATHS
        )

    def test_historical_authorities_remain_byte_exact(self) -> None:
        immutable = {
            "docs/decisions/0021-kvquant-patch-main-repository-custody.md": (
                "e09cb0f7c59c07eb04ec28319d6705c436c9c25d466bbe63e2f1859cf75d4daf"
            ),
            (
                "docs/decisions/"
                "0023-phase10-kvquant-source-faithful-sparse-fixture-semantics.md"
            ): (
                "e212b01fb286013a054567b7707a375447849986281d934c7ab17a73156bada3"
            ),
            (
                "docs/decisions/"
                "0024-kvquant-graph-safe-caller-owned-cuda-apis.md"
            ): (
                "1117bea675bbc74af873674a3c0757d93e20bb4297ec0ee3ce99418f0fc46111"
            ),
            (
                "docs/decisions/"
                "0025-kvquant-deterministic-kvq3-value-pack.md"
            ): (
                "06655f71fa5aef2077adeb40f2c1362efc27be9b42961dc8586c34d366eb0e5e"
            ),
            "reference/kvquant/fixtures/COMPLETE": (
                "b31b2621f83e95e19e7dd77701c85ab6fbf7b8ff7befd742899d6fa539918430"
            ),
            "reference/kvquant_phase11pr/fixtures/COMPLETE": (
                "0fbf63ef30a1ded14ce8dc518ea4f2b84f05ed27452f852d74bd0b07997357e1"
            ),
        }
        for relative, expected in immutable.items():
            with self.subTest(relative=relative):
                observed = hashlib.sha256(
                    (ROOT / relative).read_bytes()
                ).hexdigest()
                self.assertEqual(observed, expected)

    def test_no_broad_phase11_artifact_or_fixture_paths(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE11_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase11", "phase11_r2_outer"}),
        )
        self.assertNotIn(
            "phase11_profiler",
            validate_phase2.PHASE11_APPROVED_ARTIFACT_ROOT_NAMES,
        )
        self.assertFalse(
            any(
                path.startswith("reference/kvquant_phase11pr/")
                for path in validate_phase2.PHASE11_ALLOWED_PATHS
            )
        )
        self.assertFalse(
            any("*" in path for path in validate_phase2.PHASE11_ALLOWED_PATHS)
        )


if __name__ == "__main__":
    unittest.main()

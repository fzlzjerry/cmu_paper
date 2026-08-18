"""Exact-path scope tests for Phase 13R q4 workspace remediation."""

from __future__ import annotations

import unittest

from scripts import validate_phase2


EXPECTED = frozenset(
    {
        "Makefile",
        "docs/blockers.md",
        "docs/decisions/0036-kvquant-q4-value-decode-workspace-geometry.md",
        "docs/evidence/phase13rq4/cuda-validation.json",
        "docs/evidence/phase13rq4/kvquant-q4-method-admission.json",
        "docs/evidence/phase13rq4/r2-publication.json",
        "docs/evidence/phase13rq4/unified-admission.json",
        "docs/evidence/phase13rq4/workspace-boundaries.json",
        "docs/phase_reports/phase13r-kvquant-q4-workspace.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase12_unified_admission.py",
        "scripts/phase11_kvquant_admission.py",
        "scripts/phase13_pilot.py",
        "scripts/phase13r_q4_workspace.py",
        "scripts/validate_phase2.py",
        "src/kvbench/adapters/kvquant.py",
        "src/kvbench/runtime/kvquant_cache.py",
        "src/kvbench/runtime/kvquant_session.py",
        "tests/cuda/phase11_kvquant_sanitizer_probe.py",
        "tests/cuda/phase13r_q4_workspace_sanitizer_probe.py",
        "tests/cuda/test_phase11_kvquant_cuda.py",
        "tests/graph/test_phase11_kvquant_graph.py",
        "tests/unit/test_phase11_kvquant_cache.py",
        "tests/unit/test_phase11_kvquant_session.py",
        "tests/unit/test_phase13r_q4_workspace.py",
        "tests/unit/test_phase13rq4_scope.py",
    }
)


class Phase13RQ4ScopeTests(unittest.TestCase):
    def test_exact_entry_and_allowlist(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE13RQ4_ENTRY_COMMIT,
            "a127b0d12f1815f9e51f21767686d4536b9b35e3",
        )
        self.assertEqual(validate_phase2.PHASE13RQ4_ALLOWED_PATHS, EXPECTED)
        self.assertEqual(
            validate_phase2.PHASE13RQ4_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase13rq4"}),
        )
        for relative in EXPECTED:
            self.assertTrue(
                validate_phase2.phase13rq4_path_is_allowed(relative)
            )

    def test_near_miss_and_broad_paths_are_rejected(self) -> None:
        for relative in (
            "docs/evidence/phase13rq4",
            "docs/evidence/phase13rq4/unified-admission.json.backup",
            "artifacts/phase13rq4",
            "artifacts/phase13rq4/*",
            "src/kvbench/runtime/*.py",
            "../src/kvbench/runtime/kvquant_cache.py",
            "/src/kvbench/runtime/kvquant_cache.py",
            "src\\kvbench\\runtime\\kvquant_cache.py",
        ):
            self.assertFalse(
                validate_phase2.phase13rq4_path_is_allowed(relative)
            )

    def test_current_changes_remain_within_exact_scope(self) -> None:
        self.assertLessEqual(
            validate_phase2.current_phase13rq4_paths(),
            EXPECTED,
        )


if __name__ == "__main__":
    unittest.main()

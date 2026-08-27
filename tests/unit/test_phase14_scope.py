"""Exact-path scope tests for Phase 14."""

from __future__ import annotations

import unittest

from scripts import validate_phase2


EXPECTED_PHASE14_PATHS = frozenset(
    {
        "Makefile",
        "docs/evidence/phase14/r2-publication.json",
        "docs/phase_reports/phase14-graph-ab.md",
        "docs/plans/phase14-graph-ab-execution-order.json",
        "docs/plans/phase14-graph-ab.md",
        "docs/risk_register.md",
        "docs/status.md",
        "docs/tasks.md",
        "scripts/phase14_graph_ab.py",
        "scripts/r2_artifact.py",
        "scripts/validate_phase2.py",
        "tests/unit/test_phase14_graph_ab.py",
        "tests/unit/test_phase14_scope.py",
        "tests/unit/test_r2_artifact.py",
    }
)


class Phase14ScopeTests(unittest.TestCase):
    def test_phase14_entry_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE14_ENTRY_COMMIT,
            "7b14043c0563607d92a9dc03d0069144db65508d",
        )
        self.assertEqual(
            validate_phase2.PHASE14_ALLOWED_PATHS,
            EXPECTED_PHASE14_PATHS,
        )
        self.assertEqual(
            validate_phase2.PHASE14_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase14"}),
        )
        for relative in EXPECTED_PHASE14_PATHS:
            self.assertTrue(validate_phase2.phase14_path_is_allowed(relative))

    def test_phase14_near_miss_and_broad_paths_are_rejected(self) -> None:
        rejected = (
            "docs/evidence/phase14",
            "docs/evidence/phase14/r2-publication.json.backup",
            "docs/phase_reports/phase14-graph-ab.md.backup",
            "docs/plans/phase14-*.md",
            "scripts/phase14_graph_ab.py.backup",
            "scripts/r2_artifact.py.backup",
            "tests/unit/test_phase14_extra.py",
            "tests/unit/test_r2_artifact_extra.py",
            "src/kvbench/runtime/fixed_l_runner.py",
            "src/kvbench/adapters/kvquant.py",
            "artifacts/phase14",
            "artifacts/phase14/*",
            "../scripts/phase14_graph_ab.py",
            "/scripts/phase14_graph_ab.py",
            "scripts\\phase14_graph_ab.py",
        )
        for relative in rejected:
            self.assertFalse(
                validate_phase2.phase14_path_is_allowed(relative),
                relative,
            )


if __name__ == "__main__":
    unittest.main()

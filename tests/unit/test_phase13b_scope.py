"""Exact-path scope tests for Phase 13B batch-geometry remediation."""

from __future__ import annotations

import unittest

from scripts import validate_phase2


class Phase13BScopeTests(unittest.TestCase):
    def test_entry_and_artifact_root_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE13B_ENTRY_COMMIT,
            "c853acf65048b957a713f67b05ee560b845cd37f",
        )
        self.assertEqual(
            validate_phase2.PHASE13B_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase13b"}),
        )

    def test_every_authorized_path_is_canonical_and_exact(self) -> None:
        self.assertTrue(validate_phase2.PHASE13B_ALLOWED_PATHS)
        self.assertFalse(
            any("*" in path for path in validate_phase2.PHASE13B_ALLOWED_PATHS)
        )
        for relative in validate_phase2.PHASE13B_ALLOWED_PATHS:
            self.assertTrue(validate_phase2.phase13b_path_is_allowed(relative))

    def test_near_miss_and_broad_paths_are_rejected(self) -> None:
        for relative in (
            "src/kvbench/adapters",
            "src/kvbench/adapters/kivi.py.backup",
            "scripts/phase12_unified_admission.py.backup",
            "tests/unit/test_phase12_unified_admission.py.backup",
            "tests/cuda/*.py",
            "artifacts/phase13b",
            "../src/kvbench/adapters/kivi.py",
            "/src/kvbench/adapters/kivi.py",
            "src\\kvbench\\adapters\\kivi.py",
        ):
            self.assertFalse(validate_phase2.phase13b_path_is_allowed(relative))


if __name__ == "__main__":
    unittest.main()

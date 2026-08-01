"""Focused tests for the preregistered Phase 13 Pilot contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

from scripts import phase13_pilot


ROOT = Path(__file__).resolve().parents[2]


class Phase13PilotTests(unittest.TestCase):
    def test_exact_grid_configuration_set_and_top_context(self) -> None:
        self.assertEqual(len(phase13_pilot.CONFIGURATIONS), 10)
        self.assertEqual(phase13_pilot.BATCH_SIZES, (1, 4, 8))
        self.assertEqual(len(phase13_pilot.CONTEXT_LABELS), 9)
        self.assertEqual(phase13_pilot.actual_historical_context(131072), 131071)
        self.assertEqual(phase13_pilot.actual_historical_context(98304), 98304)
        self.assertNotIn("turboquant_k8v4", phase13_pilot.CONFIGURATIONS)
        self.assertNotIn("k4v2", phase13_pilot.CONFIGURATIONS)

    def test_execution_order_is_deterministic_complete_and_tamper_evident(self) -> None:
        first = phase13_pilot.derive_execution_order()
        second = phase13_pilot.derive_execution_order()
        self.assertEqual(first, second)
        self.assertEqual(len(first["records"]), 810)
        self.assertEqual(first["seeds"], [20260801, 20260802, 20260803])
        phase13_pilot.validate_execution_order(first)
        tampered = copy.deepcopy(first)
        tampered["records"][0]["batch_size"] = 1
        with self.assertRaisesRegex(
            phase13_pilot.Phase13PilotError, "execution order differs"
        ):
            phase13_pilot.validate_execution_order(tampered)

    def test_committed_order_is_exact(self) -> None:
        path = ROOT / phase13_pilot.ORDER_PATH
        payload = json.loads(path.read_text(encoding="utf-8"))
        phase13_pilot.validate_execution_order(payload)
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "e82c2ba5502a373989e6e82fa03fb8a30e4f104aa70320c9f4ff9cf87f0c0342",
        )

    def test_feasibility_has_every_record_and_never_masks_geometry(self) -> None:
        records = phase13_pilot.build_feasibility_records(
            phase13_pilot.derive_execution_order()
        )
        self.assertEqual(len(records), 810)
        self.assertEqual(sum(item["status"] == "feasible" for item in records), 792)
        self.assertEqual(
            sum(item["status"] == "capacity_infeasible" for item in records), 18
        )
        unsupported = [
            item for item in records if not item["adapter_geometry_supported_at_entry"]
        ]
        self.assertEqual(len(unsupported), 486)
        self.assertTrue(
            all(item["unsupported_geometry_is_not_reclassified_as_capacity"] for item in records)
        )
        self.assertTrue(
            any(item["status"] == "feasible" for item in unsupported),
            "unsupported geometry must still receive the real factory attempt",
        )

    def test_cv_uses_sample_standard_deviation_and_frozen_boundary(self) -> None:
        equal = phase13_pilot.point_statistics((1.0, 1.0, 1.0))
        self.assertEqual(equal["cv"], 0.0)
        boundary = {"cv": 0.03}
        self.assertEqual(
            phase13_pilot.classify_point(
                statistics_record=boundary, agreements=True
            ),
            "stable",
        )
        self.assertEqual(
            phase13_pilot.classify_point(
                statistics_record={"cv": 0.0300001}, agreements=True
            ),
            "unstable",
        )
        self.assertEqual(
            phase13_pilot.classify_point(
                statistics_record=boundary, agreements=False
            ),
            "failed",
        )
        with self.assertRaisesRegex(
            phase13_pilot.Phase13PilotError, "exactly three"
        ):
            phase13_pilot.point_statistics((1.0, 1.0))

    def test_fit_and_density_fail_closed(self) -> None:
        insufficient = phase13_pilot.provisional_knee_fit(
            ((4096.0, 1.0), (8192.0, 1.1), (16384.0, 1.2))
        )
        self.assertEqual(insufficient["fit_status"], "insufficient_feasible_span")
        fitted = phase13_pilot.provisional_knee_fit(
            (
                (4096.0, 1.0),
                (8192.0, 1.0),
                (16384.0, 1.2),
                (32768.0, 1.8),
                (65536.0, 3.0),
            )
        )
        self.assertIn(
            fitted["fit_status"],
            {"knee_observed", "knee_below_range", "knee_above_range"},
        )
        self.assertIn("residuals", fitted["knee_model"])
        density = phase13_pilot.knee_density((4096, 8192, 16384), 32768.0)
        self.assertFalse(density["sufficient"])
        self.assertIsNotNone(density["missing_interval"])

    def test_governance_and_historical_phase12_evidence_are_unchanged(self) -> None:
        config = json.loads((ROOT / "configs/plans/pilot.yaml").read_text())
        self.assertEqual(config["admission"]["full_scan_state"], "closed")
        self.assertEqual(config["quality"]["quality_execution"], "locked")
        self.assertFalse(config["quality"]["performance_data_frozen"])
        self.assertEqual(config["measurement"]["seed"], 20260721)
        self.assertEqual(
            hashlib.sha256((ROOT / "configs/plans/pilot.yaml").read_bytes()).hexdigest(),
            "6eb8ee48a9a569d0378879e40a6c4c965ad568e0dbf025b4ed9ca69f7ab39ea1",
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / "docs/evidence/phase12/unified-admission.json").read_bytes()
            ).hexdigest(),
            "3e337e883baaf055d307e97f25a001fb0c9a5b8a8bc14dab6230fd3d8823b4bb",
        )


if __name__ == "__main__":
    unittest.main()

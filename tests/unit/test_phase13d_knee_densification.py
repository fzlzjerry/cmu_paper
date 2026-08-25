"""Focused tests for the preregistered Phase 13D densification contract."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import phase13d_knee_densification as phase13d


ROOT = Path(__file__).resolve().parents[2]


class Phase13DCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = json.loads(
            (ROOT / "docs/plans/phase13d-candidate-table.json").read_text(
                encoding="utf-8"
            )
        )
        self.order = json.loads(
            (ROOT / "docs/plans/phase13d-execution-order.json").read_text(
                encoding="utf-8"
            )
        )

    def test_exact_target_candidate_and_run_counts(self) -> None:
        phase13d._validate_candidate_table_structure(self.candidate)
        self.assertEqual(self.candidate["target_count"], 25)
        self.assertEqual(self.candidate["proposal_count"], 175)
        self.assertEqual(self.candidate["included_candidate_count"], 84)
        self.assertEqual(self.order, phase13d.derive_execution_order(self.candidate))
        self.assertEqual(self.order["planned_run_records"], 252)
        self.assertEqual(self.order["seeds"], [20260823, 20260824, 20260825])

    def test_half_up_rounding_is_not_bankers_rounding(self) -> None:
        self.assertEqual(phase13d.round_context_half_up(Decimal("64")), 128)
        self.assertEqual(phase13d.round_context_half_up(Decimal("192")), 256)
        self.assertEqual(phase13d.round_context_half_up(Decimal("4480")), 4480)

    def test_boundary_and_midrange_candidate_sets_are_exact(self) -> None:
        by_target: dict[str, list[int]] = {}
        for row in self.candidate["proposals"]:
            if row["final_inclusion_status"] is True:
                by_target.setdefault(row["target_id"], []).append(
                    row["historical_context"]
                )
        self.assertEqual(by_target["bf16-b1"], [4480, 5120, 6144])
        self.assertEqual(
            by_target["k4v4-b1"],
            [4608, 5504, 6144, 6784, 7680, 9216],
        )
        self.assertTrue(
            all(
                4096 <= value <= 131071
                for values in by_target.values()
                for value in values
            )
        )

    def test_candidate_and_order_tampering_fail_closed(self) -> None:
        candidate = deepcopy(self.candidate)
        candidate["proposals"][0]["historical_context"] += 128
        with self.assertRaises(phase13d.Phase13DError):
            phase13d._validate_candidate_table_structure(candidate)
        order = deepcopy(self.order)
        order["records"][0]["context_label"] += 128
        self.assertNotEqual(order, phase13d.derive_execution_order(self.candidate))

    def test_randomization_is_deterministic_and_blocked(self) -> None:
        derived = phase13d.derive_execution_order(self.candidate)
        self.assertEqual(derived, phase13d.derive_execution_order(self.candidate))
        for replicate in range(3):
            records = [
                row for row in derived["records"] if row["replicate_index"] == replicate
            ]
            seen_blocks: list[str] = []
            for row in records:
                if row["target_id"] not in seen_blocks:
                    seen_blocks.append(row["target_id"])
                self.assertEqual(seen_blocks[-1], row["target_id"])
            self.assertEqual(len(seen_blocks), 25)

    def test_read_only_nested_mountpoints_are_precreated_exactly(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        marker = (
            '"$$task_root/repository/artifacts/phase13/'
            'phase13-20260822t150835736582z-4ddd7b17-3a8fb3"'
        )
        blocked = (
            '"$$task_root/repository/artifacts/phase13/'
            'phase13-20260804t111810342595z-a127b0d1-8649c3"'
        )
        phase13b = (
            '"$$task_root/repository/artifacts/phase13b/'
            'phase13b-20260801t143138050263z-b862af64-batch-admission"'
        )
        phase13rq4 = (
            '"$$task_root/repository/artifacts/phase13rq4/'
            'phase13rq4-20260820t094629495794z-ab4e0b84-b8c7bd"'
        )
        final = '"$$task_root/repository/artifacts/phase13d/$$campaign_id"'
        for value in (marker, blocked, phase13b, phase13rq4, final):
            self.assertIn(value, makefile)
        self.assertLess(makefile.index(marker), makefile.index("--validate-source"))


class Phase13DResolutionTests(unittest.TestCase):
    def _target(self) -> dict[str, object]:
        return {
            "target_id": "bf16-b1",
            "method_config_id": "bf16",
            "batch_size": 1,
        }

    def _proposals(self) -> list[dict[str, object]]:
        return [
            {
                "target_id": "bf16-b1",
                "final_inclusion_status": True,
                "outside_valid_range": False,
                "existing_point_status": False,
                "duplicate_status": False,
            }
        ]

    def test_density_sufficient_and_boundary_statuses_resolve(self) -> None:
        summary = [{"target_id": "bf16-b1", "disposition": "stable"}]
        resolution, _ = phase13d._target_resolution(
            target=self._target(),
            fit_status="knee_observed",
            density={"sufficient": True},
            new_summaries=summary,
            proposals=self._proposals(),
        )
        self.assertEqual(resolution, "density_sufficient")
        for status in ("knee_below_range", "knee_above_range", "no_positive_slope"):
            resolution, _ = phase13d._target_resolution(
                target=self._target(),
                fit_status=status,
                density={"sufficient": False},
                new_summaries=summary,
                proposals=self._proposals(),
            )
            self.assertEqual(resolution, status)

    def test_exhausted_span_is_explicit_and_unstable_is_unresolved(self) -> None:
        resolution, limitation = phase13d._target_resolution(
            target=self._target(),
            fit_status="knee_observed",
            density={"sufficient": False},
            new_summaries=[{"target_id": "bf16-b1", "disposition": "stable"}],
            proposals=self._proposals(),
        )
        self.assertEqual(resolution, "insufficient_feasible_span")
        self.assertTrue(
            limitation["all_preregistered_feasible_candidates_exhausted"]
        )
        resolution, _ = phase13d._target_resolution(
            target=self._target(),
            fit_status="unstable_data",
            density={"sufficient": False},
            new_summaries=[{"target_id": "bf16-b1", "disposition": "unstable"}],
            proposals=self._proposals(),
        )
        self.assertEqual(resolution, "unstable_data")
        resolution, _ = phase13d._target_resolution(
            target=self._target(),
            fit_status="knee_observed",
            density={"sufficient": False},
            new_summaries=[{"target_id": "bf16-b1", "disposition": "unstable"}],
            proposals=self._proposals(),
        )
        self.assertEqual(resolution, "unstable_data")

    def test_combined_monotonicity_is_warning_only(self) -> None:
        source = [
            {
                "method_config_id": "bf16",
                "batch_size": 1,
                "context_label": 4096,
                "median_ms": 2.0,
                "disposition": "stable",
            }
        ]
        densified = [
            {
                "method_config_id": "bf16",
                "batch_size": 1,
                "context_label": 4480,
                "median_ms": 1.5,
                "disposition": "stable",
                "monotonicity_warning": False,
                "monotonicity_warning_only": True,
            }
        ]
        phase13d._mark_monotonicity_warnings(
            source=source,
            densified=densified,
        )
        self.assertTrue(densified[0]["monotonicity_warning"])
        self.assertTrue(densified[0]["monotonicity_warning_only"])

    def test_source_target_count_mismatch_fails_closed(self) -> None:
        rows = [
            {
                "density_json": json.dumps(
                    {
                        "assessed": True,
                        "sufficient": False,
                        "below_count": 0,
                        "near_count": 1,
                        "above_count": 8,
                    }
                ),
                "fit_status": "knee_observed",
                "L_star": 4096.0,
                "method_config_id": "bf16",
                "batch_size": 1,
                "bootstrap_estimable": True,
                "bootstrap_knee_lower_95": 4096.0,
                "bootstrap_knee_upper_95": 4096.0,
            }
        ]
        with mock.patch.object(phase13d, "_read_parquet", return_value=rows):
            with self.assertRaisesRegex(
                phase13d.Phase13DError, "source target count differs"
            ):
                phase13d.derive_targets()

    def test_preregistration_validator_rejects_path_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidate.json"
            order_path = root / "order.json"
            candidate_path.write_bytes(phase13d.CANDIDATE_PATH.read_bytes())
            order_path.write_bytes(phase13d.ORDER_PATH.read_bytes())
            with mock.patch.object(phase13d, "CANDIDATE_PATH", candidate_path), mock.patch.object(
                phase13d, "ORDER_PATH", order_path
            ):
                result = phase13d.validate_preregistration(replay_source=False)
                self.assertEqual(result["status"], "PASS")
                payload = json.loads(order_path.read_text(encoding="utf-8"))
                payload["seeds"][0] += 1
                order_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(phase13d.Phase13DError):
                    phase13d.validate_preregistration(replay_source=False)


if __name__ == "__main__":
    unittest.main()

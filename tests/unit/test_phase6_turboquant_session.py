"""Focused bounded-grid and execution-authority tests for Phase 6."""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest import mock

from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    TurboQuantAdmissionError,
    require_authorized_cuda_environment,
)
from kvbench.runtime.turboquant_session import (
    build_turboquant_operation_keys,
    phase6_backend_fingerprint,
    session_measurement_scope,
)
from kvbench.schema import GraphMode, MeasurementScope, RunnerKind
from kvbench.schema.phase6 import AUTHORIZED_CONTAINER_DIGEST
from scripts.phase6_turboquant_admission import (
    GRID,
    _full_model_allocation_criterion,
    _memcheck_summaries_pass,
    _method_config_fingerprint,
)


class Phase6TurboQuantSessionTests(unittest.TestCase):
    def test_fixed_operation_is_one_pointer_stable_scratch_step(self) -> None:
        keys = build_turboquant_operation_keys(
            configuration="turboquant_4bit_nc",
            runner_kind=RunnerKind.FIXED_L,
            graph_mode=GraphMode.CUDA_GRAPH,
            starting_context=4096,
            output_steps=1,
        )
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0].historical_context, 4096)
        self.assertEqual(keys[0].attended_context, 4097)
        self.assertEqual(keys[0].capacity, 4097)
        self.assertEqual(keys[0].decode_step, 0)
        self.assertEqual(len(keys[0].operation_fingerprint_sha256), 64)

    def test_growing_operation_has_exact_four_step_trajectory(self) -> None:
        keys = build_turboquant_operation_keys(
            configuration="turboquant_4bit_nc",
            runner_kind=RunnerKind.GROWING_CONTEXT,
            graph_mode=GraphMode.EAGER,
            starting_context=128,
            output_steps=4,
        )
        self.assertEqual(
            [item.historical_context for item in keys],
            [128, 129, 130, 131],
        )
        self.assertEqual(
            [item.attended_context for item in keys],
            [129, 130, 131, 132],
        )
        self.assertEqual({item.capacity for item in keys}, {132})
        self.assertEqual(
            [item.decode_step for item in keys],
            [0, 1, 2, 3],
        )

    def test_operation_grid_rejects_unapproved_forms(self) -> None:
        rejected = (
            {
                "configuration": "turboquant_k8v4",
                "runner_kind": RunnerKind.FIXED_L,
                "graph_mode": GraphMode.EAGER,
                "starting_context": 128,
                "output_steps": 1,
            },
            {
                "configuration": "turboquant_k3v4_nc",
                "runner_kind": RunnerKind.GROWING_CONTEXT,
                "graph_mode": GraphMode.CUDA_GRAPH,
                "starting_context": 128,
                "output_steps": 4,
            },
            {
                "configuration": "turboquant_3bit_nc",
                "runner_kind": RunnerKind.GROWING_CONTEXT,
                "graph_mode": GraphMode.EAGER,
                "starting_context": 128,
                "output_steps": 3,
            },
            {
                "configuration": "turboquant_4bit_nc",
                "runner_kind": RunnerKind.FIXED_L,
                "graph_mode": GraphMode.EAGER,
                "starting_context": 128,
                "output_steps": 2,
            },
        )
        for parameters in rejected:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    build_turboquant_operation_keys(**parameters)

    def test_backend_fingerprint_is_deterministic_and_source_bound(self) -> None:
        first = phase6_backend_fingerprint()
        second = phase6_backend_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, "0" * 64)

    def test_phase6_scope_is_explicit_while_legacy_default_is_preserved(
        self,
    ) -> None:
        self.assertEqual(
            session_measurement_scope(SimpleNamespace()),
            MeasurementScope.NATIVE_HOST_ADMISSION,
        )
        self.assertEqual(
            session_measurement_scope(
                SimpleNamespace(
                    measurement_scope=(
                        MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
                    )
                )
            ),
            MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION,
        )

    def test_admission_grid_is_exact_and_contains_no_campaign_expansion(
        self,
    ) -> None:
        self.assertEqual(len(GRID), 9)
        self.assertEqual(
            sum(item[3] == 4096 for item in GRID),
            2,
        )
        self.assertEqual(
            sum(item[1] is RunnerKind.GROWING_CONTEXT for item in GRID),
            1,
        )
        self.assertEqual(
            {
                item[0]
                for item in GRID
                if item[3] == 4096
                or item[1] is RunnerKind.GROWING_CONTEXT
            },
            {"turboquant_4bit_nc"},
        )

    def test_method_config_fingerprints_are_exact_and_distinct(self) -> None:
        configurations = (
            "turboquant_4bit_nc",
            "turboquant_k3v4_nc",
            "turboquant_3bit_nc",
        )
        first = {
            item: _method_config_fingerprint(item)
            for item in configurations
        }
        second = {
            item: _method_config_fingerprint(item)
            for item in configurations
        }
        self.assertEqual(first, second)
        self.assertEqual(len(set(first.values())), 3)
        self.assertTrue(all(len(item) == 64 for item in first.values()))
        with self.assertRaises(ValueError):
            _method_config_fingerprint("turboquant_k8v4")

    def test_frozen_allocation_criteria_distinguish_eager_and_graph(
        self,
    ) -> None:
        eager = {
            "audit_available": True,
            "allocation_event_count": 898,
            "allocation_event_bytes": 9_802_604,
            "allocated_delta": 0,
            "reserved_delta": 0,
            "event_counts": {
                "alloc": 898,
                "free_requested": 898,
                "free_completed": 898,
            },
        }
        eager_result = _full_model_allocation_criterion(
            eager,
            graph_required=False,
            turboquant_hot_path_zero_allocation=True,
            attended_context=129,
        )
        self.assertTrue(eager_result["passed"])
        self.assertTrue(
            eager_result["fully_attributed_bounded_ephemeral"]
        )
        self.assertEqual(eager_result["unknown_allocation_count"], 0)
        drifted = dict(eager)
        drifted["allocation_event_count"] = 899
        self.assertFalse(
            _full_model_allocation_criterion(
                drifted,
                graph_required=False,
                turboquant_hot_path_zero_allocation=True,
                attended_context=129,
            )["passed"]
        )
        graph = {
            "audit_available": True,
            "allocation_event_count": 0,
            "allocation_event_bytes": 0,
            "allocated_delta": 0,
            "reserved_delta": 0,
            "event_counts": {},
        }
        graph_result = _full_model_allocation_criterion(
            graph,
            graph_required=True,
            turboquant_hot_path_zero_allocation=True,
            attended_context=129,
        )
        self.assertTrue(graph_result["passed"])
        self.assertTrue(graph_result["strict_graph_zero_events"])

    def test_memcheck_summaries_require_unique_zero_error_and_leak(self) -> None:
        passing = (
            b"========= LEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
            b"========= ERROR SUMMARY: 0 errors\n"
        )
        self.assertTrue(_memcheck_summaries_pass(passing, b""))
        self.assertFalse(
            _memcheck_summaries_pass(
                b"========= ERROR SUMMARY: 0 errors\n",
                b"",
            )
        )
        self.assertFalse(
            _memcheck_summaries_pass(
                passing
                + b"========= ERROR SUMMARY: 0 errors\n",
                b"",
            )
        )
        self.assertFalse(
            _memcheck_summaries_pass(
                b"========= LEAK SUMMARY: 1 bytes leaked in 1 allocation\n"
                b"========= ERROR SUMMARY: 0 errors\n",
                b"",
            )
        )

    def test_native_or_undeclared_cuda_environment_fails_closed(self) -> None:
        cleared = {
            PHASE6_IMAGE_ENVIRONMENT_VARIABLE: "",
            PHASE6_CONTAINER_ENVIRONMENT_VARIABLE: "",
        }
        with mock.patch.dict(os.environ, cleared, clear=False):
            with self.assertRaisesRegex(
                TurboQuantAdmissionError,
                "exact authorized Measurement Container",
            ):
                require_authorized_cuda_environment(
                    AUTHORIZED_CONTAINER_DIGEST
                )


if __name__ == "__main__":
    unittest.main()

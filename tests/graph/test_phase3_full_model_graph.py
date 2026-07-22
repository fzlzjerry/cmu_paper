"""Untimed exact-checkpoint full-model CUDA Graph validation."""

from __future__ import annotations

import unittest

import torch

from kvbench.runtime.cuda_graph import validate_full_model_fixed_graph
from kvbench.runtime.fixed_l_runner import run_fixed_l
from kvbench.runtime.model_loader import load_frozen_model


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Phase3FullModelGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_frozen_model(device="cuda:0")

    def test_short_fixed_shape_capture_replay_is_exact_and_allocation_free(
        self,
    ) -> None:
        prefix = torch.arange(
            3_000,
            3_008,
            dtype=torch.long,
            device="cuda:0",
        ).unsqueeze(0)
        current = torch.tensor([[4_000]], dtype=torch.long, device="cuda:0")
        result = validate_full_model_fixed_graph(
            self.loaded.model,
            prefix,
            current,
        )
        self.assertTrue(result.passed, result.to_dict())
        self.assertTrue(result.eager_replay_comparison["passed"])
        self.assertTrue(result.replay_outputs_exact)
        self.assertTrue(result.replay_copies_independent)
        self.assertEqual(
            result.first_replay_checksum,
            result.second_replay_checksum,
        )
        self.assertTrue(result.cache_pointers_stable)
        self.assertTrue(result.historical_cache_unchanged)
        self.assertTrue(result.replay_allocation["passed"])
        self.assertEqual(result.replay_allocation["allocation_event_count"], 0)
        serialized = result.to_dict()
        self.assertFalse(serialized["timing_collected"])
        self.assertFalse(serialized["performance_claim_eligible"])

    def test_exact_model_fixed_runner_graph_lane_times_only_after_zero_alloc(
        self,
    ) -> None:
        prefix = torch.arange(
            8_000,
            8_008,
            dtype=torch.long,
            device="cuda:0",
        ).unsqueeze(0)
        current = torch.tensor([[9_000]], dtype=torch.long, device="cuda:0")
        result = run_fixed_l(
            self.loaded.model,
            prefix,
            current,
            context_length=8,
            graph_mode="cuda_graph",
            warmup_steps=1,
            measured_steps=1,
            measured_batches=1,
        )
        self.assertTrue(result.allocation.passed, result.allocation.to_dict())
        self.assertIsNotNone(result.timing)
        self.assertIsNone(result.timing_skipped_reason)
        self.assertTrue(result.memory_evidence.timing_executed)
        self.assertIsNotNone(result.graph)
        assert result.graph is not None
        self.assertFalse(result.graph["fallback"])
        self.assertTrue(result.graph["consecutive_replay_outputs_exact"])
        self.assertTrue(result.eager_graph_comparison.passed)
        self.assertTrue(result.output_finite)


if __name__ == "__main__":
    unittest.main()

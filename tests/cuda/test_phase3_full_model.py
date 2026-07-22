"""Untimed exact-checkpoint numerical controls for the Phase 3 BF16 SUT."""

from __future__ import annotations

import unittest

import torch

from kvbench.runtime.fixed_l_runner import run_fixed_l
from kvbench.runtime.growing_context_runner import run_growing_context
from kvbench.runtime.model_loader import load_frozen_model
from kvbench.runtime.numerical import validate_full_model_reference


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Phase3FullModelReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        loaded = load_frozen_model(device="cuda:0")
        cls.model = loaded.model
        cls.tokenizer = loaded.tokenizer
        cls.identity = loaded.identity

    def test_exact_model_fixed_and_growing_match_dynamic_cache_reference(
        self,
    ) -> None:
        self.assertEqual(self.model.__class__.__name__, "LlamaForCausalLM")
        self.assertEqual(
            self.tokenizer.__class__.__name__,
            "PreTrainedTokenizerFast",
        )
        self.assertEqual(len(self.identity.file_hashes), 11)
        prefix = torch.arange(
            1_000,
            1_008,
            dtype=torch.long,
            device="cuda:0",
        ).unsqueeze(0)
        decode = torch.arange(
            2_000,
            2_003,
            dtype=torch.long,
            device="cuda:0",
        ).unsqueeze(0)
        result = validate_full_model_reference(self.model, prefix, decode)
        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(result.reference_cache_type, "DynamicCache")
        self.assertTrue(result.reference_implementation_restored)
        self.assertTrue(result.fixed_repeat_exact)
        self.assertTrue(result.fixed_historical_cache_unchanged)
        self.assertEqual(len(result.fixed_steps), 3)
        self.assertEqual(len(result.growing_steps), 3)
        for evidence in (*result.fixed_steps, *result.growing_steps):
            self.assertTrue(evidence.comparison.passed, evidence.to_dict())
            self.assertTrue(evidence.comparison.finite, evidence.to_dict())
        serialized = result.to_dict()
        self.assertFalse(serialized["timing_collected"])
        self.assertFalse(serialized["performance_claim_eligible"])

    def test_exact_model_eager_runners_fail_closed_before_normal_timing(
        self,
    ) -> None:
        prefix = torch.arange(
            5_000,
            5_008,
            dtype=torch.long,
            device="cuda:0",
        ).unsqueeze(0)
        current = torch.tensor([[6_000]], dtype=torch.long, device="cuda:0")
        fixed = run_fixed_l(
            self.model,
            prefix,
            current,
            context_length=8,
            graph_mode="eager",
            warmup_steps=1,
            measured_steps=1,
            measured_batches=1,
        )
        self.assertFalse(fixed.allocation.passed)
        self.assertIsNone(fixed.timing)
        self.assertIsNotNone(fixed.timing_skipped_reason)
        self.assertFalse(fixed.memory_evidence.timing_executed)
        self.assertTrue(fixed.historical_cache_unchanged)
        self.assertTrue(fixed.output_finite)

        decode = torch.arange(
            7_000,
            7_003,
            dtype=torch.long,
            device="cuda:0",
        ).unsqueeze(0)
        growing = run_growing_context(
            self.model,
            prefix,
            decode,
            starting_context=8,
            warmup_trajectories=1,
        )
        self.assertFalse(growing.allocation.passed)
        self.assertIsNone(growing.timing)
        self.assertIsNotNone(growing.timing_skipped_reason)
        self.assertFalse(growing.memory_evidence.timing_executed)
        self.assertEqual(growing.active_lengths, (8, 9, 10))
        self.assertTrue(growing.cache_pointers_stable)
        self.assertTrue(growing.output_finite)


if __name__ == "__main__":
    unittest.main()

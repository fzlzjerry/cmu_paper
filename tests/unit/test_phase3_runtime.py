"""Focused CPU controls for the Phase 3 runtime core."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from kvbench.runtime.allocation import MemorySnapshot, NormalTimingMemoryEvidence
from kvbench.runtime.backend import (
    BackendFallbackError,
    backend_identity,
    flash_attention_forward,
)
from kvbench.runtime.bf16_endpoint import rotate_half_in_place
from kvbench.runtime.model_loader import (
    EXPECTED_HASHES,
    EXPECTED_SIZES,
    ModelAccessError,
    verify_frozen_snapshot,
)
from kvbench.runtime.phase3_coordinator import (
    SENSITIVE_ENV_FRAGMENTS,
    SENSITIVE_ENV_KEY_EXEMPTIONS,
    _worker_environment,
)
from kvbench.runtime.numerical import (
    compare_tensors_untimed,
    small_attention_reference,
    tensor_sha256_untimed,
)
from kvbench.runtime.static_cache import (
    BF16StaticCache,
    CacheBoundsError,
    cache_accounting_for_geometry,
    layout_fingerprint_for_geometry,
)
from kvbench.runtime.telemetry import (
    TelemetrySnapshot,
    telemetry_sampling_interval_seconds,
)
from kvbench.schema import BF16BackendIdentity


class Phase3StaticCacheTests(unittest.TestCase):
    def make_cache(self, *, capacity: int = 5) -> BF16StaticCache:
        return BF16StaticCache(
            num_layers=2,
            batch_size=2,
            num_kv_heads=2,
            capacity=capacity,
            head_dim=4,
            device="cpu",
        )

    def prefill(self, cache: BF16StaticCache, length: int) -> None:
        cache.prepare_prefill(length)
        for layer in range(cache.num_layers):
            base = torch.arange(
                cache.batch_size * cache.num_kv_heads * length * cache.head_dim,
                dtype=torch.bfloat16,
            ).reshape(
                cache.batch_size,
                cache.num_kv_heads,
                length,
                cache.head_dim,
            )
            cache.update(base + layer, base + layer + 1, layer)
        cache.complete_prefill()

    def test_shape_bytes_and_pure_fingerprint_match(self) -> None:
        cache = self.make_cache()
        self.assertEqual(tuple(cache.keys.shape), (2, 2, 2, 5, 4))
        self.assertEqual(cache.predicted_tensor_bytes, 640)
        self.assertEqual(cache.tensor_storage_bytes, 640)
        self.assertEqual(cache.accounting().padding_bytes, 0)
        pure = cache_accounting_for_geometry(
            num_layers=2,
            batch_size=2,
            num_kv_heads=2,
            capacity=5,
            head_dim=4,
        )
        self.assertEqual(pure["predicted_tensor_bytes"], 640)
        self.assertEqual(
            cache.layout_fingerprint(),
            layout_fingerprint_for_geometry(
                num_layers=2,
                batch_size=2,
                num_kv_heads=2,
                capacity=5,
                head_dim=4,
                device="cpu",
            ),
        )

    def test_fixed_scratch_preserves_historical_storage(self) -> None:
        cache = self.make_cache()
        self.prefill(cache, 4)
        historical_keys = cache.keys[:, :, :, :4, :].clone()
        historical_values = cache.values[:, :, :, :4, :].clone()
        pointers = cache.pointers()
        cache.prepare_fixed(4)
        for iteration in range(5):
            key = torch.full((2, 2, 1, 4), iteration, dtype=torch.bfloat16)
            value = torch.full((2, 2, 1, 4), iteration + 1, dtype=torch.bfloat16)
            for layer in range(2):
                attended_key, attended_value = cache.update(key, value, layer)
                self.assertEqual(tuple(attended_key.shape), (2, 2, 5, 4))
                self.assertEqual(tuple(attended_value.shape), (2, 2, 5, 4))
        self.assertTrue(torch.equal(cache.keys[:, :, :, :4, :], historical_keys))
        self.assertTrue(
            torch.equal(cache.values[:, :, :, :4, :], historical_values)
        )
        self.assertEqual(cache.active_context, 4)
        self.assertEqual(cache.pointers(), pointers)

    def test_growing_progression_and_bounds(self) -> None:
        cache = self.make_cache(capacity=4)
        self.prefill(cache, 2)
        cache.prepare_growing(2, 2)
        for step in range(2):
            cache.select_growing_step(step)
            key = torch.full((2, 2, 1, 4), step + 5, dtype=torch.bfloat16)
            for layer in range(2):
                attended, _ = cache.update(key, key, layer)
                self.assertEqual(int(attended.shape[-2]), 3 + step)
            cache.finish_growing_step()
            self.assertEqual(cache.active_context, 3 + step)
        with self.assertRaises(CacheBoundsError):
            cache.select_growing_step(2)
        cache.reset_active_length(0)
        self.assertEqual(cache.active_context, 0)

    def test_capacity_rejection_is_prewrite(self) -> None:
        cache = self.make_cache(capacity=3)
        self.prefill(cache, 2)
        original = cache.keys.clone()
        with self.assertRaises(CacheBoundsError):
            cache.prepare_growing(2, 2)
        self.assertTrue(torch.equal(cache.keys, original))


class Phase3NumericalTests(unittest.TestCase):
    def test_cat_free_half_rotation_matches_explicit_formula(self) -> None:
        original = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0]]]],
            dtype=torch.bfloat16,
        )
        states = original.clone()
        cos = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.bfloat16)
        sin = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]], dtype=torch.bfloat16)
        scratch = torch.empty_like(states[..., :2])
        expected = torch.empty_like(states)
        expected[..., :2] = (
            original[..., :2] * cos.unsqueeze(1)[..., :2]
            - original[..., 2:] * sin.unsqueeze(1)[..., :2]
        )
        expected[..., 2:] = (
            original[..., 2:] * cos.unsqueeze(1)[..., 2:]
            + original[..., :2] * sin.unsqueeze(1)[..., 2:]
        )
        rotate_half_in_place(states, cos, sin, scratch)
        self.assertTrue(torch.equal(states, expected))

    def test_small_gqa_reference_causal_and_finite(self) -> None:
        torch.manual_seed(7)
        query = torch.randn((2, 4, 3, 8), dtype=torch.bfloat16)
        key = torch.randn((2, 2, 3, 8), dtype=torch.bfloat16)
        value = torch.randn((2, 2, 3, 8), dtype=torch.bfloat16)
        reference = small_attention_reference(
            query,
            key,
            value,
            is_causal=True,
            scale=8**-0.5,
        )
        self.assertEqual(tuple(reference.shape), (2, 4, 3, 8))
        comparison = compare_tensors_untimed(
            reference,
            reference.clone(),
            atol=0.02,
            rtol=0.02,
        )
        self.assertTrue(comparison.passed)
        self.assertTrue(comparison.finite)

    def test_checksum_binds_shape_and_dtype(self) -> None:
        value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        self.assertEqual(
            tensor_sha256_untimed(value),
            tensor_sha256_untimed(value.clone()),
        )
        self.assertNotEqual(
            tensor_sha256_untimed(value),
            tensor_sha256_untimed(value.reshape(4, 2)),
        )


class Phase3IdentityAndBackendTests(unittest.TestCase):
    def test_frozen_snapshot_contract_has_exact_eleven_sized_entries(self) -> None:
        self.assertEqual(len(EXPECTED_HASHES), 11)
        self.assertEqual(set(EXPECTED_HASHES), set(EXPECTED_SIZES))
        self.assertEqual(EXPECTED_SIZES["LICENSE"], 7_627)
        self.assertEqual(
            EXPECTED_HASHES["LICENSE"],
            "64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369",
        )

    def test_backend_identity_is_verified_and_schema_consumable(self) -> None:
        identity = backend_identity()
        self.assertEqual(identity["torch_version"], "2.12.1+cu130")
        self.assertEqual(identity["selected_backend"], "flash_attention")
        self.assertIs(identity["enable_gqa"], True)
        self.assertEqual(len(identity["source_artifacts"]), 5)
        parsed = BF16BackendIdentity.from_dict(identity)
        self.assertEqual(parsed.backend_id, "torch_sdpa_flash_gqa")

    def test_backend_fails_outside_forced_context(self) -> None:
        query = torch.zeros((1, 4, 1, 8), dtype=torch.bfloat16)
        key = torch.zeros((1, 2, 1, 8), dtype=torch.bfloat16)
        with self.assertRaises(BackendFallbackError):
            flash_attention_forward(
                object(),
                query,
                key,
                key,
                None,
                8**-0.5,
            )

    def test_missing_exact_snapshot_fails_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-the-frozen-revision"
            with self.assertRaises(ModelAccessError):
                verify_frozen_snapshot(missing)


class Phase3EvidenceSerializationTests(unittest.TestCase):
    def test_sanitized_worker_environment_is_constructible_and_secret_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = _worker_environment(Path(directory))

        self.assertEqual(environment["TOKENIZERS_PARALLELISM"], "false")
        self.assertNotIn("HF_TOKEN", environment)
        self.assertFalse(
            any(
                fragment in key.lower()
                for key in environment
                if key not in SENSITIVE_ENV_KEY_EXEMPTIONS
                for fragment in SENSITIVE_ENV_FRAGMENTS
            )
        )

    def test_normal_timing_memory_evidence_keeps_audit_separate(self) -> None:
        def snapshot(label: str, allocated: int, peak: int) -> MemorySnapshot:
            return MemorySnapshot(
                label=label,
                host_timestamp_ns=1,
                allocated_bytes=allocated,
                reserved_bytes=allocated + 10,
                peak_allocated_bytes=peak,
                peak_reserved_bytes=peak + 10,
            )

        evidence = NormalTimingMemoryEvidence(
            model_baseline=snapshot("model_baseline", 100, 100),
            post_cache_allocation=snapshot("post_cache_allocation", 150, 150),
            post_setup=snapshot("post_setup", 175, 200),
            timing_before=snapshot("normal_timing_before", 175, 175),
            timing_after=snapshot("normal_timing_after", 180, 210),
            timing_executed=True,
        ).to_dict()
        self.assertEqual(evidence["timing_allocated_delta_bytes"], 5)
        self.assertEqual(evidence["timing_peak_allocated_bytes"], 210)
        self.assertTrue(evidence["instrumented_audit_separate"])
        self.assertTrue(evidence["timing_executed"])
        self.assertFalse(evidence["peak_reset_inside_measured_boundary"])
        self.assertFalse(evidence["profiler_duration_reported"])

    def test_telemetry_interval_is_raw_and_monotonic(self) -> None:
        def snapshot(host_ns: int) -> TelemetrySnapshot:
            return TelemetrySnapshot(
                timestamp="2026/07/22 00:00:00.000",
                collected_at_utc="2026-07-22T00:00:00+00:00",
                host_query_started_ns=host_ns - 5,
                host_query_finished_ns=host_ns + 5,
                host_monotonic_ns=host_ns,
                gpu_name="GPU",
                gpu_uuid="GPU-0000",
                power_watts=100.0,
                temperature_celsius=40.0,
                sm_clock_mhz=1000.0,
                memory_clock_mhz=2000.0,
                vram_used_mib=3000.0,
                ecc_mode="Enabled",
            )

        before = snapshot(1_000_000_000)
        after = snapshot(2_500_000_000)
        self.assertEqual(telemetry_sampling_interval_seconds(before, after), 1.5)
        serialized = before.to_dict()
        self.assertTrue(serialized["raw_snapshot"])
        self.assertFalse(serialized["stability_inference"])


if __name__ == "__main__":
    unittest.main()

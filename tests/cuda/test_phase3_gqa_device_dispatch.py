"""Actual CUDA operator controls for the Phase 3 B-011 dispatch gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from kvbench.runtime.gqa_device_dispatch import (
    FLASH_FORWARD_FAMILY,
    MATERIALIZATION_CLASSIFICATIONS,
    collect_gqa_mha_device_dispatch,
)
from kvbench.schema import GQAVerdict


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Phase3GQADeviceDispatchCudaTests(unittest.TestCase):
    def test_public_flash_gqa_and_mha_controls_expose_device_kernels(self) -> None:
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(20260722)
        query = torch.randn(
            (1, 32, 1, 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        gqa_key = torch.randn(
            (1, 8, 128, 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        gqa_value = torch.randn_like(gqa_key, generator=generator)
        mha_key = torch.randn(
            (1, 32, 128, 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        mha_value = torch.randn_like(mha_key, generator=generator)
        repository_root = Path(__file__).resolve().parents[2]
        output_directory = Path(
            tempfile.mkdtemp(prefix="kvbench-phase3-dispatch-audit-", dir="/tmp")
        )
        audit = collect_gqa_mha_device_dispatch(
            gqa_query=query,
            gqa_key=gqa_key,
            gqa_value=gqa_value,
            mha_query=query,
            mha_key=mha_key,
            mha_value=mha_value,
            output_directory=output_directory,
            artifact_relative_root="dispatch/traces",
            source_root=repository_root,
            source_paths=(
                Path("src/kvbench/runtime/backend.py"),
                Path("src/kvbench/runtime/bf16_endpoint.py"),
                Path("src/kvbench/runtime/static_cache.py"),
            ),
            is_causal=False,
            scale=128**-0.5,
            warmup_count=3,
            # Task C supplies the independent allocator-history proof. This
            # operator gate must remain unproven, rather than inventing it.
            allocation_verified=False,
        )
        payload = audit.to_dict()
        (output_directory / "dispatch-audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"preserved_dispatch_audit={output_directory}")

        self.assertTrue(audit.raw_trace_bytes_verified)
        self.assertTrue(audit.gqa.backend.passed)
        self.assertTrue(audit.mha.backend.passed)
        self.assertEqual(audit.gqa.backend.enabled_backends, ("FLASH_ATTENTION",))
        self.assertEqual(audit.mha.backend.enabled_backends, ("FLASH_ATTENTION",))
        self.assertTrue(audit.evaluation.dispatch_verified)
        self.assertTrue(audit.evaluation.no_replication_kernel_verified)
        self.assertEqual(
            audit.evaluation.verdict,
            GQAVerdict.NONMATERIALIZATION_UNPROVEN,
        )
        self.assertEqual(audit.gqa.attention_families, (FLASH_FORWARD_FAMILY,))
        self.assertEqual(audit.mha.attention_families, (FLASH_FORWARD_FAMILY,))
        self.assertFalse(audit.gqa.events_before_attention)
        self.assertFalse(audit.mha.events_before_attention)
        self.assertFalse(
            any(
                event.classification in MATERIALIZATION_CLASSIFICATIONS
                for event in (*audit.gqa.device_events, *audit.mha.device_events)
            )
        )
        self.assertTrue(audit.gqa_source_shape.source_verified)
        self.assertTrue(audit.mha_source_shape.source_verified)
        self.assertTrue(audit.gqa_source_shape.shape_verified_for(audit.gqa))
        self.assertTrue(audit.mha_source_shape.shape_verified_for(audit.mha))
        self.assertEqual(audit.gqa.byte_evidence.native_kv_bytes, 524_288)
        self.assertEqual(audit.gqa.byte_evidence.expanded_kv_bytes, 2_097_152)
        for role, artifact in (
            ("gqa", audit.gqa.raw_trace),
            ("mha", audit.mha.raw_trace),
        ):
            self.assertIsNotNone(artifact)
            assert artifact is not None
            path = output_directory / f"{role}.chrome.json"
            raw = path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), artifact.sha256)
            self.assertEqual(len(raw), artifact.size_bytes)
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in ('"latency"', '"duration"', '"wall_time_ms"'):
            self.assertNotIn(forbidden, rendered)


if __name__ == "__main__":
    unittest.main()

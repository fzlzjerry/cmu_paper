"""Focused tests for Phase 13P-R untimed deterministic prefix restoration."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import torch

from kvbench.runtime.static_cache import BF16StaticCache
from scripts import phase13_pilot as pilot
from scripts.phase13_prefix_state import (
    Phase13PrefixStateError,
    restore_prefix_state,
    save_prefix_state,
    validate_prefix_state,
)


ROOT = Path(__file__).resolve().parents[2]


def _cache(batch: int) -> BF16StaticCache:
    cache = BF16StaticCache(
        num_layers=2,
        batch_size=batch,
        num_kv_heads=2,
        capacity=18,
        head_dim=4,
        device="cpu",
    )
    cache.initialize_deterministic()
    return cache


class PrefixStateTests(unittest.TestCase):
    def test_catalog_plan_is_deterministic_and_covers_228_unique_points(self) -> None:
        feasibility = pilot.build_feasibility_records(pilot.derive_execution_order())
        first = pilot.derive_prefix_catalog_plan(feasibility)
        second = pilot.derive_prefix_catalog_plan(feasibility)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 90)
        self.assertEqual(sum(len(item["target_batches"]) for item in first), 228)
        for item in first:
            self.assertEqual(item["source_batch"], max(item["target_batches"]))
            self.assertTrue(item["leading_rows_are_source_faithful"])

    def test_restore_copies_leading_rows_into_fresh_caller_owned_buffers(self) -> None:
        source = _cache(8)
        source.prepare_prefill(17)
        values = torch.arange(source.keys[:, :, :, :17, :].numel()).reshape(
            source.keys[:, :, :, :17, :].shape
        )
        source.keys[:, :, :, :17, :].copy_(values.to(torch.bfloat16))
        source.values[:, :, :, :17, :].copy_((values + 37).to(torch.bfloat16))
        source.complete_prefill()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshot"
            manifest = save_prefix_state(
                cache=source,
                family="bf16",
                configuration="bf16",
                historical=17,
                source_batch=8,
                output=root,
                authority={"test": True},
            )
            target = _cache(1)
            target_pointers = target.pointers()
            receipt = restore_prefix_state(
                cache=target,
                family="bf16",
                configuration="bf16",
                historical=17,
                root=root,
                expected_state_sha256=manifest["state_file_sha256"],
            )
            self.assertTrue(
                torch.equal(target.keys[:, 0], source.keys[:, 0])
            )
            self.assertTrue(
                torch.equal(target.values[:, 0], source.values[:, 0])
            )
            self.assertTrue(torch.count_nonzero(target.keys[:, :, :, 17:, :]) == 0)
            self.assertEqual(target.pointers(), target_pointers)
            self.assertTrue(set(target.pointers().values()).isdisjoint(source.pointers().values()))
            self.assertTrue(receipt["fresh_target_allocation"])
            self.assertFalse(receipt["runtime_prefix_sharing"])
            self.assertTrue(receipt["restore_outside_timing"])

    def test_state_and_manifest_tampering_fail_closed(self) -> None:
        source = _cache(1)
        source.prepare_prefill(17)
        source.keys[:, :, :, :17, :].fill_(1)
        source.values[:, :, :, :17, :].fill_(2)
        source.complete_prefill()
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            save_prefix_state(
                cache=source,
                family="bf16",
                configuration="bf16",
                historical=17,
                source_batch=1,
                output=first,
                authority={"test": True},
            )
            state = first / "state.safetensors"
            with state.open("r+b") as handle:
                handle.seek(-1, 2)
                original = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([original[0] ^ 1]))
            with self.assertRaisesRegex(Phase13PrefixStateError, "checksum"):
                validate_prefix_state(first, verify_state_bytes=True)

            second = Path(temporary) / "second"
            save_prefix_state(
                cache=source,
                family="bf16",
                configuration="bf16",
                historical=17,
                source_batch=1,
                output=second,
                authority={"test": True},
            )
            manifest_path = second / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_batch"] = 8
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            target = _cache(1)
            with self.assertRaisesRegex(Phase13PrefixStateError, "geometry"):
                restore_prefix_state(
                    cache=target,
                    family="bf16",
                    configuration="bf16",
                    historical=17,
                    root=second,
                    expected_state_sha256=manifest["state_file_sha256"],
                )

    def test_formal_worker_uses_restore_and_omits_full_per_point_audits(self) -> None:
        worker = inspect.getsource(pilot._run_worker)
        self.assertIn("_build_restored_session", worker)
        self.assertNotIn("execution_path_audit_facade", worker)
        self.assertNotIn("fixture", worker.lower())
        self.assertNotIn("sanitizer", worker.lower())
        self.assertIn("prefix_state_restore", worker)
        session_bridge = inspect.getsource(pilot._bind_session_prefix_witness)
        self.assertIn("current_historical_prefix_sha256", session_bridge)
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn(
            "src=$$task_root/prefix-states,dst=/opt/kvbench-prefix-states",
            makefile,
        )

    def test_equivalence_loads_model_before_inference_mode(self) -> None:
        source = inspect.getsource(pilot.run_prefix_equivalence)
        self.assertLess(
            source.index("loaded = load_frozen_model"),
            source.index("with torch.inference_mode()"),
        )

    def test_equivalence_uses_existing_graph_witness_phase(self) -> None:
        source = inspect.getsource(pilot._equivalence_session_record)
        self.assertIn('phase="before"', source)
        self.assertNotIn('phase="equivalence"', source)


if __name__ == "__main__":
    unittest.main()

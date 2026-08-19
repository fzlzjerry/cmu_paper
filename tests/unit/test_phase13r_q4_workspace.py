"""Focused Decision 0036 q4 Value-decode workspace geometry tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import torch

from kvbench.adapters.kvquant import KVQuantMethodAdapter
from kvbench.runtime.kvquant_cache import (
    KVQUANT_HEAD_DIM,
    KVQUANT_NUM_KV_HEADS,
    KVQUANT_NUM_LAYERS,
    KVQUANT_NUM_QUERY_HEADS,
    KVQUANT_Q4_VALUE_DECODE_TILE_WIDTH,
    KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION,
    KVQuantStaticCache,
    kvquant_q4_value_decode_quantized_capacity,
    kvquant_q4_value_decode_tile_capacity,
    kvquant_q4_value_decode_workspace_bytes,
    kvquant_q4_value_decode_workspace_shape,
)
from kvbench.runtime.kvquant_session import kvquant_runtime_context
from kvbench.runtime.static_cache import CacheStateError
from scripts import phase13_pilot
from scripts import phase13r_q4_workspace


class Phase13RQ4WorkspaceTests(unittest.TestCase):
    def _cache(
        self,
        *,
        batch_size: int = 1,
        capacity: int = 18,
        configuration: str = "kvq4",
    ) -> KVQuantStaticCache:
        return KVQuantStaticCache(
            config_name=configuration,
            num_layers=KVQUANT_NUM_LAYERS,
            batch_size=batch_size,
            num_query_heads=KVQUANT_NUM_QUERY_HEADS,
            num_kv_heads=KVQUANT_NUM_KV_HEADS,
            capacity=capacity,
            head_dim=KVQUANT_HEAD_DIM,
            device="cpu",
        )

    def test_fixed_l_capacity_boundaries_are_exact(self) -> None:
        cases = (
            (4096, 4097, 4092, 32),
            (8192, 8193, 8188, 64),
            (16384, 16385, 16380, 128),
            (32768, 32769, 32764, 256),
            (65536, 65537, 65532, 512),
            (131071, 131072, 131067, 1024),
        )
        for historical, total, quantized, tiles in cases:
            with self.subTest(historical=historical):
                self.assertEqual(
                    kvquant_q4_value_decode_quantized_capacity(total),
                    quantized,
                )
                self.assertEqual(
                    kvquant_q4_value_decode_tile_capacity(total),
                    tiles,
                )
        self.assertEqual(KVQUANT_Q4_VALUE_DECODE_TILE_WIDTH, 128)
        self.assertEqual(
            KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION,
            "kvbench-kvquant-q4-value-workspace-capacity-v1",
        )

    def test_batch_scales_bytes_but_not_tile_count(self) -> None:
        total = 16385
        self.assertEqual(kvquant_q4_value_decode_tile_capacity(total), 128)
        for batch in (1, 4, 8):
            with self.subTest(batch=batch):
                shape = kvquant_q4_value_decode_workspace_shape(
                    batch_size=batch,
                    total_attended_capacity=total,
                )
                self.assertEqual(shape, (batch, 32, 128, 128))
                self.assertEqual(
                    kvquant_q4_value_decode_workspace_bytes(
                        batch_size=batch,
                        total_attended_capacity=total,
                    ),
                    batch * 32 * 128 * 128 * 4,
                )

    def test_tile_boundary_has_no_off_by_one(self) -> None:
        self.assertEqual(kvquant_q4_value_decode_tile_capacity(133), 1)
        self.assertEqual(kvquant_q4_value_decode_tile_capacity(134), 2)
        self.assertEqual(
            kvquant_q4_value_decode_quantized_capacity(133),
            128,
        )
        self.assertEqual(
            kvquant_q4_value_decode_quantized_capacity(134),
            129,
        )

    def test_workspace_is_caller_owned_stable_and_exact(self) -> None:
        cache = self._cache(capacity=134)
        workspace = cache.q4_value_decode_workspace
        self.assertIsNotNone(workspace)
        assert workspace is not None
        pointer = workspace.untyped_storage().data_ptr()
        self.assertIs(
            cache.require_q4_value_decode_workspace(quantized_length=129),
            workspace,
        )
        self.assertEqual(
            cache.q4_value_decode_workspace.untyped_storage().data_ptr(),
            pointer,
        )
        self.assertEqual(
            cache.q4_value_decode_workspace_bytes,
            workspace.untyped_storage().nbytes(),
        )
        geometry = cache.q4_value_decode_workspace_geometry()
        self.assertEqual(geometry["workspace_shape"], [1, 32, 2, 128])
        self.assertEqual(geometry["workspace_bytes"], 32 * 2 * 128 * 4)
        self.assertFalse(geometry["resize_during_decode"])
        self.assertFalse(geometry["counted_as_cache_payload"])

    def test_decode_path_rejects_undersized_workspace(self) -> None:
        source = inspect.getsource(KVQuantMethodAdapter._decode_quantized_value)
        self.assertIn("require_q4_value_decode_workspace", source)
        cache = self._cache(capacity=134)
        cache.q4_value_decode_workspace = torch.empty(
            (1, 32, 1, 128),
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(CacheStateError, "capacity differs"):
            cache.require_q4_value_decode_workspace(quantized_length=129)

    def test_q3_q2_admitted_fingerprints_are_unchanged(self) -> None:
        expected = {
            "kvq3": (
                "995d0629e0f14c2472789345115524c887453c087435da767d37c35d0570bf7f",
                "54089c1908844c1e1d6119c58c27f0fc1ba2eb4bb6dadb9ca57191e81a7d38c9",
            ),
            "kvq2": (
                "5c851a6fdc4aea323d248f172bf50604d54b6e105911a715bb420305a6c587fe",
                "ec961123e0e429f72e78716a0e28998c739c806cc7dbd70033b40c3ca4a232f1",
            ),
        }
        for configuration, (adapter_expected, layout_expected) in expected.items():
            with self.subTest(configuration=configuration):
                cache = self._cache(
                    capacity=129,
                    configuration=configuration,
                )
                layout = cache.layout_fingerprint()
                adapter = KVQuantMethodAdapter(
                    kvquant_runtime_context(configuration),
                    configuration,
                )
                self.assertEqual(layout, layout_expected)
                self.assertEqual(
                    adapter.config_fingerprint(layout),
                    adapter_expected,
                )
                self.assertIsNone(cache.q4_value_decode_workspace)

    def test_pilot_feasibility_includes_exact_q4_workspace(self) -> None:
        order = phase13_pilot.derive_execution_order()
        records = phase13_pilot.build_feasibility_records(order)
        selected = {
            (record["batch_size"], record["context_label"]): record
            for record in records
            if record["replicate_index"] == 0
            and record["method_config_id"] == "kvq4"
            and record["batch_size"] == 8
            and record["context_label"] in {16_384, 65_536}
        }
        admitted = selected[(8, 16_384)]
        rejected = selected[(8, 65_536)]
        self.assertEqual(admitted["status"], "feasible")
        self.assertEqual(
            admitted["q4_value_decode_workspace"]["workspace_shape"],
            [8, 32, 128, 128],
        )
        self.assertEqual(
            admitted["q4_value_decode_workspace"]["workspace_bytes"],
            16_777_216,
        )
        self.assertEqual(rejected["status"], "capacity_infeasible")
        self.assertEqual(
            rejected["q4_value_decode_workspace"]["tile_capacity"],
            512,
        )

    def test_target_harness_warms_post_capture_eager_reserve_before_audit(
        self,
    ) -> None:
        source = inspect.getsource(
            phase13r_q4_workspace._target_session_record
        )
        warmup = source.index("session._fixed_operation()\n")
        synchronize = source.index(
            "torch.cuda.synchronize(device=session.cache_device)",
            warmup,
        )
        pointers = source.index("pointers_before =", synchronize)
        audit = source.index("eager_allocation = audit_cuda_allocations", pointers)
        self.assertLess(warmup, synchronize)
        self.assertLess(synchronize, pointers)
        self.assertLess(pointers, audit)

    def test_successor_report_binds_probe_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cuda_path = root / "cuda.json"
            sanitizer_path = root / "sanitizer.json"
            probe_path = root / "probe.json"
            cuda_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "execution_git_sha": "6" * 40,
                        "target_records": [
                            {
                                "batch_size": batch,
                                "method_config_fingerprint": "1" * 64,
                                "cache_layout_fingerprint": "2" * 64,
                            }
                            for batch in (4, 8)
                        ],
                        "source_hashes": {"cache": "3" * 64},
                    }
                ),
                encoding="utf-8",
            )
            sanitizer_path.write_text(
                json.dumps(
                    {"status": "PASS", "execution_git_sha": "6" * 40}
                ),
                encoding="utf-8",
            )
            probe_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "execution_git_sha": "6" * 40,
                        "method_config_fingerprint": "4" * 64,
                        "cache_layout_fingerprint": "5" * 64,
                    }
                ),
                encoding="utf-8",
            )
            report = phase13r_q4_workspace.successor_report(
                git_sha="6" * 40,
                cuda_validation=cuda_path,
                sanitizer=sanitizer_path,
                standardized_probe=probe_path,
            )
            phase13r_q4_workspace.validate_successor_report(report)
            self.assertEqual(
                report["standardized_phase12_method_config_fingerprint"],
                "4" * 64,
            )
            tampered = dict(report)
            tampered["q3_changed"] = True
            with self.assertRaisesRegex(
                phase13r_q4_workspace.Phase13RQ4Error,
                "MethodAdmissionReport differs",
            ):
                phase13r_q4_workspace.validate_successor_report(tampered)


if __name__ == "__main__":
    unittest.main()

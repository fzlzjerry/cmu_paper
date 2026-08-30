"""Focused tests for Phase 16G exact B=2/B=16 geometry admission."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from kvbench.runtime.static_cache import BF16StaticCache
from kvbench.runtime.kivi_cache import KIVIStaticCache
from kvbench.runtime.kivi_cache import _raw_tensor_bytes_untimed
from kvbench.runtime.kvquant_cache import KVQuantStaticCache
from kvbench.runtime.kvquant_session import _key_active_entries_untimed
from kvbench.runtime.turboquant_cache import TurboQuantStaticCache
from kvbench.schema.phase16g import (
    PHASE16G_ADMITTED_BATCH_SIZES,
    PHASE16G_CONFIGURATIONS,
    PHASE16G_CONTAINER_DIGEST,
    PHASE16G_INDEX_SCHEMA,
    PHASE16G_NEW_BATCH_SIZES,
    PHASE16G_PREFIX_SCHEMA,
    Phase16GGeometryError,
    geometry_key,
    require_admitted_geometry,
    validate_geometry_index,
)
from scripts import phase16g_batch_geometry_admission as phase16g
from scripts import validate_phase2
from scripts.phase13_prefix_state import (
    PREFIX_STATE_SCHEMA,
    PREFIX_STATE_SCHEMA_V3,
    Phase13PrefixStateError,
    restore_prefix_state,
    save_prefix_state,
    validate_prefix_state,
)


ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT = "a" * 64
MODEL = {"id": "model", "revision": "model-revision"}
TOKENIZER = {"id": "tokenizer", "revision": "tokenizer-revision"}


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
    cache.prepare_prefill(17)
    values = torch.arange(
        cache.keys[:, :, :, :17, :].numel(),
        dtype=torch.float32,
    ).reshape(cache.keys[:, :, :, :17, :].shape)
    cache.keys[:, :, :, :17, :].copy_(values.to(torch.bfloat16))
    cache.values[:, :, :, :17, :].copy_((values + 7).to(torch.bfloat16))
    cache.complete_prefill()
    return cache


def _empty_cache(batch: int) -> BF16StaticCache:
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


def _inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.arange(batch * 17, dtype=torch.long).reshape(batch, 17)
    positions = torch.arange(17, dtype=torch.long).reshape(1, 17).expand(batch, -1).clone()
    return tokens, positions


def _save_v3(root: Path, batch: int) -> tuple[BF16StaticCache, dict[str, object]]:
    source = _cache(batch)
    tokens, positions = _inputs(batch)
    manifest = save_prefix_state(
        cache=source,
        family="bf16",
        configuration="bf16",
        historical=17,
        source_batch=batch,
        method_config_fingerprint=FINGERPRINT,
        output=root,
        authority={"test": True},
        schema_version=PREFIX_STATE_SCHEMA_V3,
        prefix_token_ids=tokens,
        prefix_positions=positions,
        model_identity=MODEL,
        tokenizer_identity=TOKENIZER,
    )
    return source, manifest


def _synthetic_index() -> dict[str, object]:
    records = {}
    for configuration in PHASE16G_CONFIGURATIONS:
        for batch in PHASE16G_NEW_BATCH_SIZES:
            records[geometry_key(configuration, batch)] = {
                "configuration": configuration,
                "batch_size": batch,
                "status": "PASS",
                "method_config_fingerprint": "a" * 64,
                "adapter_config_fingerprint": "b" * 64,
                "cache_layout_fingerprint": "c" * 64,
                "short_eager": "PASS",
                "short_cuda_graph": "PASS",
                "prefix_restore": "PASS",
                "allocation": "PASS",
                "execution_path": "PASS",
            }
    return {
        "schema_version": PHASE16G_INDEX_SCHEMA,
        "status": "PASS",
        "decision_id": "0039",
        "authorized_container_digest": PHASE16G_CONTAINER_DIGEST,
        "prefix_format_version": PHASE16G_PREFIX_SCHEMA,
        "admitted_full_scan_batches": list(PHASE16G_ADMITTED_BATCH_SIZES),
        "existing_geometry_batches_unchanged": [1, 4, 8],
        "existing_geometry_authorities": {
            configuration: {"path": "authority.json"}
            for configuration in PHASE16G_CONFIGURATIONS
        },
        "new_geometry_records": records,
    }


class Phase16GBatchGeometryTests(unittest.TestCase):
    def test_b2_and_b16_prefix_schema_round_trip_exactly(self) -> None:
        for batch in PHASE16G_NEW_BATCH_SIZES:
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "state"
                source, manifest = _save_v3(root, batch)
                validated = validate_prefix_state(
                    root,
                    configuration="bf16",
                    family="bf16",
                    batch=batch,
                    historical=17,
                    method_config_fingerprint=FINGERPRINT,
                    model_identity=MODEL,
                    tokenizer_identity=TOKENIZER,
                    verify_state_bytes=True,
                )
                self.assertEqual(validated["schema_version"], PREFIX_STATE_SCHEMA_V3)
                target = _empty_cache(batch)
                tokens, positions = _inputs(batch)
                receipt = restore_prefix_state(
                    cache=target,
                    family="bf16",
                    configuration="bf16",
                    historical=17,
                    root=root,
                    expected_state_sha256=manifest["state_file_sha256"],
                    expected_method_config_fingerprint=FINGERPRINT,
                    expected_prefix_token_ids=tokens,
                    expected_prefix_positions=positions,
                    expected_model_identity=MODEL,
                    expected_tokenizer_identity=TOKENIZER,
                )
                self.assertTrue(torch.equal(target.keys, source.keys))
                self.assertTrue(torch.equal(target.values, source.values))
                self.assertTrue(receipt["restored_tensor_checksums_exact"])
                self.assertTrue(receipt["stored_input_tensors_exact"])

    def test_legacy_b1_b4_b8_prefixes_remain_compatible(self) -> None:
        for batch in (1, 4, 8):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "state"
                source = _cache(batch)
                manifest = save_prefix_state(
                    cache=source,
                    family="bf16",
                    configuration="bf16",
                    historical=17,
                    source_batch=batch,
                    method_config_fingerprint=FINGERPRINT,
                    output=root,
                    authority={"legacy": True},
                )
                self.assertEqual(manifest["schema_version"], PREFIX_STATE_SCHEMA)
                self.assertEqual(validate_prefix_state(root)["source_batch"], batch)

    def test_manifest_tensor_batch_mismatch_and_broadcast_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "state"
            _, manifest = _save_v3(root, 2)
            target = _empty_cache(2)
            wrong_tokens, wrong_positions = _inputs(1)
            with self.assertRaisesRegex(Phase13PrefixStateError, "input tensor"):
                restore_prefix_state(
                    cache=target,
                    family="bf16",
                    configuration="bf16",
                    historical=17,
                    root=root,
                    expected_state_sha256=manifest["state_file_sha256"],
                    expected_method_config_fingerprint=FINGERPRINT,
                    expected_prefix_token_ids=wrong_tokens,
                    expected_prefix_positions=wrong_positions,
                    expected_model_identity=MODEL,
                    expected_tokenizer_identity=TOKENIZER,
                )
            manifest_path = root / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["batch_size"] = 16
            manifest_path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Phase13PrefixStateError, "manifest batch"):
                validate_prefix_state(root)

    def test_geometry_gate_admits_exact_five_batches_only(self) -> None:
        payload = _synthetic_index()
        validate_geometry_index(payload)
        for configuration in PHASE16G_CONFIGURATIONS:
            for batch in PHASE16G_ADMITTED_BATCH_SIZES:
                self.assertEqual(
                    require_admitted_geometry(
                        payload,
                        configuration=configuration,
                        batch_size=batch,
                    ),
                    f"{configuration}/B{batch}",
                )
        with self.assertRaises(Phase16GGeometryError):
            require_admitted_geometry(payload, configuration="bf16", batch_size=3)

    def test_compressed_cache_constructors_admit_b2_and_b16_only(self) -> None:
        for batch in PHASE16G_NEW_BATCH_SIZES:
            with self.subTest(family="turboquant", batch=batch):
                cache = TurboQuantStaticCache(
                    config_name="turboquant_4bit_nc",
                    num_layers=32,
                    batch_size=batch,
                    num_query_heads=32,
                    num_kv_heads=8,
                    capacity=17,
                    head_dim=128,
                    device="cpu",
                )
                self.assertEqual(cache.batch_size, batch)
            with self.subTest(family="kivi", batch=batch):
                cache = KIVIStaticCache(
                    config_name="k4v4",
                    num_layers=1,
                    batch_size=batch,
                    num_query_heads=32,
                    num_kv_heads=8,
                    capacity=64,
                    head_dim=128,
                    device="cpu",
                )
                self.assertEqual(cache.batch_size, batch)
            with self.subTest(family="kvquant", batch=batch):
                cache = KVQuantStaticCache(
                    config_name="kvq4",
                    num_layers=32,
                    batch_size=batch,
                    num_query_heads=32,
                    num_kv_heads=8,
                    capacity=18,
                    head_dim=128,
                    device="cpu",
                )
                self.assertEqual(cache.batch_size, batch)
        for constructor in (
            lambda: TurboQuantStaticCache(
                config_name="turboquant_4bit_nc",
                num_layers=32,
                batch_size=3,
                num_query_heads=32,
                num_kv_heads=8,
                capacity=17,
                head_dim=128,
                device="cpu",
            ),
            lambda: KIVIStaticCache(
                config_name="k4v4",
                num_layers=1,
                batch_size=3,
                num_query_heads=32,
                num_kv_heads=8,
                capacity=64,
                head_dim=128,
                device="cpu",
            ),
            lambda: KVQuantStaticCache(
                config_name="kvq4",
                num_layers=32,
                batch_size=3,
                num_query_heads=32,
                num_kv_heads=8,
                capacity=18,
                head_dim=128,
                device="cpu",
            ),
        ):
            with self.assertRaisesRegex(ValueError, "B in"):
                constructor()

    def test_short_matrix_and_maximum_selection_contracts_are_exact(self) -> None:
        expected = {
            (configuration, batch, mode)
            for configuration in PHASE16G_CONFIGURATIONS
            for batch in PHASE16G_NEW_BATCH_SIZES
            for mode in phase16g.MODES
        }
        self.assertEqual(len(expected), 40)
        first = phase16g.b16_feasibility_records()
        second = phase16g.b16_feasibility_records()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 90)
        maxima = phase16g.largest_feasible_contexts(first)
        self.assertEqual(set(maxima), set(PHASE16G_CONFIGURATIONS))
        for record in first:
            expected_cache = phase16g.phase13.cache_allocated_bytes(
                str(record["method_config_id"]),
                16,
                int(record["capacity"]),
            )
            self.assertEqual(record["cache_allocated_bytes"], expected_cache)

    def test_new_geometry_eager_allocation_control_is_exact_and_tamper_closed(
        self,
    ) -> None:
        counts = {
            "alloc": 898,
            "free_completed": 898,
            "free_requested": 898,
        }
        fields = {
            "audit_available": True,
            "allocation_event_count": 898,
            "allocation_event_bytes": 21_586_552,
            "event_counts": counts,
            "allocated_before": 100,
            "allocated_after": 100,
            "reserved_before": 200,
            "reserved_after": 200,
        }
        first = SimpleNamespace(**fields)
        repeat = SimpleNamespace(**fields)
        passed, control = phase16g._eager_allocation_passed(
            first,
            repeat,
            family="turboquant",
            batch=2,
        )
        self.assertTrue(passed)
        self.assertTrue(control["repeat_exact"])
        self.assertFalse(control["event_byte_extrapolation_used"])
        tampered = SimpleNamespace(
            **{**fields, "allocation_event_bytes": 21_586_553}
        )
        self.assertFalse(
            phase16g._eager_allocation_passed(
                first,
                tampered,
                family="turboquant",
                batch=2,
            )[0]
        )

    def test_eager_allocator_is_primed_before_both_admission_audits(self) -> None:
        calls: list[str] = []

        class Session:
            cache_device = torch.device("cpu")

            @staticmethod
            def _fixed_operation() -> torch.Tensor:
                calls.append("operation")
                return torch.zeros(1)

        audit = SimpleNamespace()
        session = Session()
        session.graph = SimpleNamespace(replay=lambda: torch.zeros(1))

        def fake_audit(operation, *, device):
            del device
            calls.append("audit")
            operation()
            return audit

        with (
            patch(
                "kvbench.runtime.allocation.audit_cuda_allocations",
                side_effect=fake_audit,
            ),
            patch("torch.cuda.synchronize"),
        ):
            phase16g._session_outputs_and_audits(session)

        self.assertEqual(
            calls[:5],
            ["operation", "audit", "operation", "audit", "operation"],
        )

    def test_untimed_kivi_raw_buffer_preserves_exact_checksum_bytes(self) -> None:
        tensor = torch.arange(257, dtype=torch.int32).reshape(257, 1)
        legacy = bytes(tensor.untyped_storage())[
            : tensor.numel() * tensor.element_size()
        ]
        current = bytes(_raw_tensor_bytes_untimed(tensor))
        self.assertEqual(current, legacy)

    def test_untimed_kvquant_count_buffer_preserves_exact_values(self) -> None:
        counts = torch.tensor([[[3, 0, 7, 2, 5]]], dtype=torch.int32)
        cache = SimpleNamespace(sink_tokens=1, key_active_counts=counts)
        self.assertEqual(_key_active_entries_untimed(cache, 5), 12)

    def test_allocator_reclaim_occurs_after_caller_drops_session(self) -> None:
        events: list[str] = []

        class Graph:
            @staticmethod
            def reset() -> None:
                events.append("graph_reset")

        class Session:
            graph = SimpleNamespace(graph=Graph())

            def __del__(self) -> None:
                events.append("session_dropped")

        session = Session()
        with (
            patch("gc.collect", side_effect=lambda: events.append("gc")),
            patch(
                "torch.cuda.empty_cache",
                side_effect=lambda: events.append("empty_cache"),
            ),
        ):
            phase16g._release_session(session)
            session = None
            phase16g._reclaim_cuda_allocator()

        self.assertEqual(
            events,
            ["graph_reset", "session_dropped", "gc", "empty_cache"],
        )

    def test_compute_sanitizer_version_banner_is_exact_and_tamper_closed(
        self,
    ) -> None:
        genuine = (
            "NVIDIA (R) Compute Sanitizer\n"
            "Copyright (c) 2020-2025 NVIDIA Corporation\n"
            "Version 2025.3.1.0 (build 36400806) (public-release)"
        )
        self.assertTrue(phase16g._compute_sanitizer_version_valid(genuine))
        self.assertFalse(
            phase16g._compute_sanitizer_version_valid(
                genuine.replace("NVIDIA (R) Compute Sanitizer", "other tool")
            )
        )

    def test_compute_sanitizer_result_channel_is_unique_and_tamper_closed(
        self,
    ) -> None:
        result = '{"result":{"batch_size":16},"status":"PASS"}'
        genuine = (
            "========= COMPUTE-SANITIZER\n"
            f"{result}\n"
            "========= ERROR SUMMARY: 0 errors\n"
        )
        self.assertEqual(
            phase16g._compute_sanitizer_result_channel(genuine)["status"],
            "PASS",
        )
        with self.assertRaises(ValueError):
            phase16g._compute_sanitizer_result_channel(genuine + result)
        with self.assertRaises(ValueError):
            phase16g._compute_sanitizer_result_channel(
                genuine + "unexpected trailing text\n"
            )
        self.assertFalse(
            phase16g._compute_sanitizer_version_valid(
                genuine.replace("Version 2025.3.1.0", "Version unknown")
            )
        )

    def test_historical_method_admission_reports_are_unchanged(self) -> None:
        expected = {
            "docs/evidence/phase4/method-admission.json": "1362fd1817b8bb5706baaa09ed6e5115789fbc4d35d394f184d0b132a0e58d22",
            "docs/evidence/phase13b/turboquant-method-admission.json": "49799ef89646ec008a530c5180fdcef6cd4af9ca0d5772fe2b01d6e775e3b1c0",
            "docs/evidence/phase13b/kivi-method-admission.json": "1e91730ac56af37e03d80edce7979a509d52049428faad89f61e61dc6bd48c51",
            "docs/evidence/phase13b/kvquant-method-admission.json": "e1cee8e1c514f9cf6323b5e710480c1fefab2804e5f4eafe6c473b29f4768481",
            "docs/evidence/phase13rq4/kvquant-q4-method-admission.json": "75605637f460a309081e1e0a4065e90e8e14194d365250cb513092616ef89ec7",
        }
        for relative, digest in expected.items():
            self.assertEqual(phase16g.sha256_file(ROOT / relative), digest)

    def test_phase16g_scope_is_exact_and_full_scan_is_not_allowed(self) -> None:
        expected = frozenset(
            {
                "docs/decisions/0039-full-scan-batch-geometry-admission.md",
                "docs/evidence/phase16g/batch-geometry-admission.json",
                "docs/evidence/phase16g/r2-publication.json",
                "docs/phase_reports/phase16g-batch-geometry-admission.md",
                "docs/risk_register.md",
                "docs/status.md",
                "docs/tasks.md",
                "scripts/phase13_prefix_state.py",
                "scripts/phase16g_batch_geometry_admission.py",
                "scripts/validate_phase2.py",
                "src/kvbench/runtime/kivi_cache.py",
                "src/kvbench/runtime/kvquant_cache.py",
                "src/kvbench/runtime/kvquant_session.py",
                "src/kvbench/runtime/turboquant_cache.py",
                "src/kvbench/schema/phase16g.py",
                "tests/cuda/phase16g_batch_sanitizer_probe.py",
                "tests/unit/test_phase16g_batch_geometry.py",
                "tests/unit/test_phase13b_batch_geometry.py",
            }
        )
        self.assertEqual(validate_phase2.PHASE16G_ALLOWED_PATHS, expected)
        self.assertEqual(
            validate_phase2.PHASE16G_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase16g"}),
        )
        self.assertNotIn("configs/plans/full_scan.yaml", expected)
        self.assertNotIn("artifacts/full_scan", expected)


if __name__ == "__main__":
    unittest.main()

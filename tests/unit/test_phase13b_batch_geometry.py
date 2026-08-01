"""Focused CPU contracts for Phase 13B static compressed-cache batching."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import tempfile
import unittest

from kvbench.errors import ArtifactConflictError
from kvbench.runtime.artifacts import (
    AppendOnlyArtifactStore,
    validate_run_directory,
)
from kvbench.runtime.kivi_cache import KIVIStaticCache
from kvbench.runtime.kvquant_cache import KVQuantStaticCache
from kvbench.runtime.turboquant_cache import (
    TURBOQUANT_SLOT_SIZES,
    TurboQuantStaticCache,
)
from kvbench.schema import RunStatus
from kvbench.schema.phase13b import (
    PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
    PHASE13B_BATCH_SIZES,
    PHASE13B_CONFIGURATIONS,
    Phase13BBatchAdmissionManifest,
)
from scripts.phase13b_compressed_batch_admission import (
    MATRIX_SCHEMA,
    Phase13BBatchAdmissionError,
    validate_cuda_matrix,
)


SUPPORTED_BATCH_SIZES = (1, 4, 8)
TURBOQUANT_CONFIGS = (
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
)
KIVI_CONFIGS = ("k4v4", "k2v4", "k2v2")
KVQUANT_BITS = {"kvq4": 4, "kvq3": 3, "kvq2": 2}


class Phase13BStaticBatchGeometryTests(unittest.TestCase):
    def test_exact_compressed_configuration_and_batch_sets(self) -> None:
        self.assertEqual(len(TURBOQUANT_CONFIGS + KIVI_CONFIGS + tuple(KVQUANT_BITS)), 9)
        self.assertEqual(SUPPORTED_BATCH_SIZES, (1, 4, 8))

    def test_turboquant_native_batch_banks_and_exact_accounting(self) -> None:
        for configuration in TURBOQUANT_CONFIGS:
            for batch in SUPPORTED_BATCH_SIZES:
                with self.subTest(configuration=configuration, batch=batch):
                    cache = TurboQuantStaticCache(
                        config_name=configuration,
                        num_layers=32,
                        batch_size=batch,
                        num_query_heads=32,
                        num_kv_heads=8,
                        capacity=17,
                        head_dim=128,
                        device="cpu",
                    )
                    self.assertEqual(
                        tuple(cache.packed_cache.shape),
                        (28, batch * 2, 16, 8, TURBOQUANT_SLOT_SIZES[configuration]),
                    )
                    self.assertEqual(tuple(cache.block_table.shape), (batch, 2))
                    self.assertEqual(tuple(cache.store_key_float.shape), (17, batch, 8, 128))
                    self.assertEqual(tuple(cache.decode_output.shape), (batch, 32, 128))
                    self.assertEqual(cache.slot_mapping.numel(), batch * 32)
                    if batch == 1:
                        self.assertEqual(cache.block_table.tolist(), [[0, 1]])
                        self.assertEqual(cache.slot_mapping.tolist(), list(range(32)))
                    accounting = cache.accounting()
                    self.assertEqual(sum(cache.byte_breakdown().values()), accounting.allocated_bytes)
                    self.assertEqual(accounting.predicted_tensor_bytes, accounting.measured_tensor_bytes)
                    self.assertEqual(cache.layout_fingerprint(), cache.layout_fingerprint())
                    del cache
                    gc.collect()

    def test_kivi_native_batch_axis_and_exact_accounting(self) -> None:
        for configuration in KIVI_CONFIGS:
            for batch in SUPPORTED_BATCH_SIZES:
                with self.subTest(configuration=configuration, batch=batch):
                    cache = KIVIStaticCache(
                        config_name=configuration,
                        num_layers=1,
                        batch_size=batch,
                        num_query_heads=32,
                        num_kv_heads=8,
                        capacity=64,
                        head_dim=128,
                        device="cpu",
                    )
                    self.assertEqual(cache.packed_key_history.shape[:3], (1, batch, 8))
                    self.assertEqual(cache.packed_value_history.shape[:3], (1, batch, 8))
                    self.assertEqual(tuple(cache.key_residual.shape), (1, batch, 8, 32, 128))
                    self.assertEqual(tuple(cache.decode_output_fp16.shape), (batch, 32, 128))
                    accounting = cache.accounting()
                    self.assertEqual(cache.byte_breakdown(), cache.predicted_byte_breakdown())
                    self.assertEqual(sum(cache.byte_breakdown().values()), accounting.allocated_bytes)
                    relative_error = abs(
                        accounting.predicted_tensor_bytes
                        - accounting.measured_tensor_bytes
                    ) / accounting.measured_tensor_bytes
                    self.assertLess(relative_error, 0.01)
                    self.assertEqual(cache.layout_fingerprint(), cache.layout_fingerprint())
                    del cache
                    gc.collect()

    def test_kvquant_native_batch_banks_and_exact_accounting(self) -> None:
        for configuration, bits in KVQUANT_BITS.items():
            for batch in SUPPORTED_BATCH_SIZES:
                with self.subTest(configuration=configuration, batch=batch):
                    cache = KVQuantStaticCache(
                        config_name=configuration,
                        num_layers=32,
                        batch_size=batch,
                        num_query_heads=32,
                        num_kv_heads=8,
                        capacity=18,
                        head_dim=128,
                        device="cpu",
                    )
                    self.assertEqual(
                        tuple(cache.packed_key_cache.shape),
                        (32, batch, 8, bits * 4, 18),
                    )
                    self.assertEqual(tuple(cache.key_sparse_values.shape), (32, batch, 18, 12))
                    self.assertEqual(tuple(cache.value_sparse_indices.shape), (32, batch, 18, 12))
                    self.assertEqual(tuple(cache.sink_key.shape), (32, batch, 8, 128, 5))
                    self.assertEqual(tuple(cache.decode_quantized_output.shape), (batch, 32, 128))
                    accounting = cache.accounting()
                    self.assertEqual(cache.byte_breakdown(), cache.predicted_byte_breakdown())
                    self.assertEqual(sum(cache.byte_breakdown().values()), accounting.allocated_bytes)
                    self.assertLess(accounting.relative_error, 0.01)
                    ratios = cache.ratios()
                    self.assertLessEqual(abs(ratios.rho_alloc * ratios.r_alloc - 1.0), 1e-9)
                    self.assertIsNone(ratios.r_hbm)
                    self.assertEqual(cache.layout_fingerprint(), cache.layout_fingerprint())
                    del cache
                    gc.collect()

    def test_unadmitted_batch_geometry_fails_closed(self) -> None:
        constructors = (
            lambda: TurboQuantStaticCache(
                config_name="turboquant_4bit_nc",
                num_layers=32,
                batch_size=2,
                num_query_heads=32,
                num_kv_heads=8,
                capacity=17,
                head_dim=128,
                device="cpu",
            ),
            lambda: KIVIStaticCache(
                config_name="k4v4",
                num_layers=1,
                batch_size=2,
                num_query_heads=32,
                num_kv_heads=8,
                capacity=64,
                head_dim=128,
                device="cpu",
            ),
            lambda: KVQuantStaticCache(
                config_name="kvq4",
                num_layers=32,
                batch_size=2,
                num_query_heads=32,
                num_kv_heads=8,
                capacity=18,
                head_dim=128,
                device="cpu",
            ),
        )
        for constructor in constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaisesRegex(ValueError, "B in"):
                    constructor()

    def test_cuda_matrix_validation_is_exact_and_fail_closed(self) -> None:
        records = [
            {
                "configuration": configuration,
                "batch_size": batch,
                "status": "PASS",
                "r_hbm": None,
                "timing_collected": False,
            }
            for configuration in PHASE13B_CONFIGURATIONS
            for batch in PHASE13B_BATCH_SIZES
        ]
        payload = {
            "schema_version": MATRIX_SCHEMA,
            "status": "PASS",
            "authorized_container_digest": PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
            "configurations": list(PHASE13B_CONFIGURATIONS),
            "batch_sizes": list(PHASE13B_BATCH_SIZES),
            "point_count": 27,
            "records": records,
        }
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "matrix.json"
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(validate_cuda_matrix(artifact)["status"], "PASS")
            payload["records"][0]["batch_size"] = 2
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(Phase13BBatchAdmissionError):
                validate_cuda_matrix(artifact)

    def test_phase13b_uses_existing_complete_last_lifecycle(self) -> None:
        created_at = "2026-08-01T00:00:00Z"
        run_id = "phase13b-test-batch-admission"
        created = Phase13BBatchAdmissionManifest(
            schema_version=Phase13BBatchAdmissionManifest.SCHEMA_VERSION,
            run_id=run_id,
            status=RunStatus.CREATED,
            created_at_utc=created_at,
            started_at_utc=None,
            finished_at_utc=None,
            inventory_path=None,
            failure_reason=None,
            creation_git_sha="1" * 40,
            authorized_container_digest=PHASE13B_AUTHORIZED_CONTAINER_DIGEST,
            decision_id="0030",
            configurations=PHASE13B_CONFIGURATIONS,
            batch_sizes=PHASE13B_BATCH_SIZES,
            context_length=128,
            timing_collected=False,
            performance_claim_eligible=False,
            quality_executed=False,
            full_scan_executed=False,
        )
        completed = Phase13BBatchAdmissionManifest.from_dict(
            {
                **created.to_dict(),
                "status": RunStatus.COMPLETED.value,
                "started_at_utc": created_at,
                "finished_at_utc": "2026-08-01T00:01:00Z",
                "inventory_path": "artifact_inventory.json",
            }
        )
        required = (
            "admission/kivi-method-admission.json",
            "admission/kvquant-method-admission.json",
            "admission/turboquant-method-admission.json",
            "allocation/audit.json",
            "config/authority.json",
            "environment/container_identity.json",
            "gqa/audit.json",
            "numerical/batch-control.json",
            "validation/cuda-graph.json",
            "validation/historical-preservation.json",
            "validation/matrix.json",
            "validation/non-default-stream.json",
            "validation/sanitizer.json",
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = AppendOnlyArtifactStore(Path(temporary) / "phase13b")
            run = store.create(run_id, created)
            run.start()
            for relative in required:
                run.write_json(relative, {"status": "PASS"})
            final = run.finalize(completed)
            validation = validate_run_directory(final)
            self.assertTrue(validation.valid)
            self.assertTrue(validation.complete)
            self.assertTrue((final / "COMPLETE").is_file())
            with self.assertRaises(ArtifactConflictError):
                store.create(run_id, created)


if __name__ == "__main__":
    unittest.main()

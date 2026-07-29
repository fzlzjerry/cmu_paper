"""Focused exact-path and preservation tests for Phase 11P-R."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import unittest

from scripts import validate_phase2


ROOT = Path(__file__).resolve().parents[2]


class Phase11PRScopeTests(unittest.TestCase):
    def test_entry_and_static_allowlist_are_exact(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE11PR_ENTRY_COMMIT,
            "1cb2c95be61a328f88a031ae4ce91784dddec736",
        )
        static = {
            "Makefile",
            "docs/decisions/0025-kvquant-deterministic-kvq3-value-pack.md",
            "docs/evidence/phase11pr/cuda-validation.json",
            "docs/evidence/phase11pr/r2-publication.json",
            "docs/phase_reports/phase11p-r-kvq3-value-pack.md",
            "reference/kvquant_phase11pr/generate_corrected_bundle.py",
            "reference/kvquant_phase11pr/validate_corrected_bundle.py",
            "scripts/validate_kvquant_graphsafe_patch.py",
            "scripts/validate_phase2.py",
            "tests/cuda/phase11pr_kvq3_pack_validation.py",
            "tests/unit/test_phase9p_patch_custody.py",
            "tests/unit/test_phase11pr_scope.py",
            (
                "third_party/patches/kvquant/"
                "0002-graphsafe-kvq3-deterministic.patch"
            ),
            (
                "third_party/patches/kvquant/"
                "graphsafe-kvq3-manifest.json"
            ),
        }
        self.assertEqual(
            validate_phase2.PHASE11PR_ALLOWED_PATHS,
            static
            | validate_phase2.PHASE11PR_FIXTURE_PATHS
            | validate_phase2.PHASE11PR_FIXTURE_ROOT_PATHS,
        )

    def test_fixture_matrix_members_and_root_paths_are_exact(self) -> None:
        expected_fixture_paths = {
            (
                "reference/kvquant_phase11pr/fixtures/"
                f"{family}/{case}/{member}"
            )
            for family in ("kvq4", "kvq3", "kvq2")
            for case in (
                "key_zero_value_fixed12",
                "key_few_value_fixed12",
                "key_cap_value_fixed12",
            )
            for member in (
                "fixture_manifest.json",
                "inputs.safetensors",
                "dense_payload.safetensors",
                "metadata.safetensors",
                "sparse_values.safetensors",
                "sparse_indices.safetensors",
                "sink.safetensors",
                "store_state.safetensors",
                "append_state.safetensors",
                "decode_output.safetensors",
                "byte_breakdown.json",
                "checksums.sha256",
            )
        }
        self.assertEqual(
            validate_phase2.PHASE11PR_FIXTURE_PATHS,
            expected_fixture_paths,
        )
        self.assertEqual(len(expected_fixture_paths), 3 * 3 * 12)
        self.assertEqual(
            validate_phase2.PHASE11PR_FIXTURE_ROOT_PATHS,
            {
                "reference/kvquant_phase11pr/fixtures/COMPLETE",
                (
                    "reference/kvquant_phase11pr/fixtures/"
                    "artifact_inventory.json"
                ),
                "reference/kvquant_phase11pr/fixtures/checksums.sha256",
                "reference/kvquant_phase11pr/fixtures/manifest.json",
                "reference/kvquant_phase11pr/fixtures/reference_trace.json",
                "reference/kvquant_phase11pr/fixtures/reuse_proof.json",
                (
                    "reference/kvquant_phase11pr/fixtures/authority/"
                    "build_manifest.json"
                ),
                (
                    "reference/kvquant_phase11pr/fixtures/authority/"
                    "calibration_manifest.json"
                ),
                (
                    "reference/kvquant_phase11pr/fixtures/authority/"
                    "environment.json"
                ),
                (
                    "reference/kvquant_phase11pr/fixtures/authority/"
                    "source_manifest.json"
                ),
            },
        )

    def test_only_exact_fixture_tensors_escape_raw_artifact_rejection(
        self,
    ) -> None:
        expected = {
            path
            for path in validate_phase2.PHASE11PR_FIXTURE_PATHS
            if path.endswith(".safetensors")
        }
        self.assertEqual(
            validate_phase2.PHASE11PR_SAFE_TENSOR_PATHS,
            expected,
        )
        self.assertEqual(len(expected), 3 * 3 * 9)

    def test_near_paths_and_deferred_work_are_rejected(self) -> None:
        rejected = {
            (
                "reference/kvquant_phase11pr/__pycache__/"
                "generate_corrected_bundle.cpython-312.pyc"
            ),
            "reference/kvquant_phase11pr/README.md",
            "reference/kvquant_phase11pr/framework.py",
            (
                "reference/kvquant_phase11pr/fixtures/kvq3/no_outlier/"
                "fixture_manifest.json"
            ),
            (
                "reference/kvquant_phase11pr/fixtures/kvq3/"
                "key_zero_value_fixed12/copy.safetensors"
            ),
            (
                "reference/kvquant_phase11pr/fixtures/kvq3/"
                "key_zero_value_fixed12/debug.pt"
            ),
            (
                "reference/kvquant_phase11pr/fixtures/authority/"
                "source_checkout.tar"
            ),
            (
                "reference/kvquant/fixtures/kvq3/"
                "key_zero_value_fixed12/dense_payload.safetensors"
            ),
            "scripts/r2_artifact_phase11pr.py",
            "third_party/patches/kvquant/0003-kvq3.patch",
            "docs/evidence/phase11pr/r2-publication-v2.json",
            "docker/measurement.Dockerfile",
            "src/kvbench/adapters/kvquant.py",
            "artifacts/profiler/phase11pr/result.json",
            "artifacts/quality/phase11pr/result.json",
        }
        self.assertFalse(
            rejected & validate_phase2.PHASE11PR_ALLOWED_PATHS
        )

    def test_phase10_and_phase11p_authority_remain_byte_exact(self) -> None:
        immutable = {
            "docs/phase_reports/phase10-kvquant-reference.md": (
                "6d7d3bb3f40e20f6af16e9c41d1ee4173b19c4ef171854830b286fa64b10b0e8"
            ),
            "docs/phase_reports/phase10-kvquant-reference-blocked.md": (
                "0362ac36f03ec8b92c0b154ad85fd4e810c3d45332f7c063b1da00d7adc94e4d"
            ),
            "docs/decisions/0021-kvquant-patch-main-repository-custody.md": (
                "e09cb0f7c59c07eb04ec28319d6705c436c9c25d466bbe63e2f1859cf75d4daf"
            ),
            (
                "docs/decisions/"
                "0023-phase10-kvquant-source-faithful-sparse-fixture-semantics.md"
            ): (
                "e212b01fb286013a054567b7707a375447849986281d934c7ab17a73156bada3"
            ),
            (
                "docs/decisions/"
                "0024-kvquant-graph-safe-caller-owned-cuda-apis.md"
            ): (
                "1117bea675bbc74af873674a3c0757d93e20bb4297ec0ee3ce99418f0fc46111"
            ),
            "reference/kvquant/fixtures/COMPLETE": (
                "b31b2621f83e95e19e7dd77701c85ab6fbf7b8ff7befd742899d6fa539918430"
            ),
            "reference/kvquant/fixtures/artifact_inventory.json": (
                "a133a5deb438922354970cc05ef38bd44bf44f5f00d00ee7ec4c5d026cadf1d6"
            ),
            "reference/kvquant/fixtures/checksums.sha256": (
                "8c2a5119703261bc7c37dd9e893f15a65cbd937b06e81110aa7761e5b01c4937"
            ),
            "reference/kvquant/fixtures/manifest.json": (
                "ff984277bf44624b742eed399dba7c827bd66d66350d7b2889824a8c00bc5079"
            ),
        }
        for relative, expected in immutable.items():
            with self.subTest(relative=relative):
                observed = hashlib.sha256(
                    (ROOT / relative).read_bytes()
                ).hexdigest()
                self.assertEqual(observed, expected)

    def test_phase_boundary_and_adapter_deferral_are_explicit(self) -> None:
        frozen = validate_phase2.current_phase10_paths()
        self.assertIn(
            "docs/decisions/0024-kvquant-graph-safe-caller-owned-cuda-apis.md",
            frozen,
        )
        self.assertNotIn(
            "docs/decisions/0025-kvquant-deterministic-kvq3-value-pack.md",
            frozen,
        )
        for relative in (
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/kvquant.py",
        ):
            self.assertNotEqual(
                subprocess.run(
                    [
                        "git",
                        "cat-file",
                        "-e",
                        f"{validate_phase2.PHASE11_ENTRY_COMMIT}:{relative}",
                    ],
                    cwd=ROOT,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode,
                0,
            )


if __name__ == "__main__":
    unittest.main()

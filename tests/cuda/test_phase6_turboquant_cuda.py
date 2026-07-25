"""Fixture, native-GQA, and skipped-layer CUDA checks for Phase 6."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VALUE,
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    evaluate_fixture_configuration,
    require_authorized_cuda_environment,
)
from kvbench.runtime.turboquant_cache import TURBOQUANT_MANDATORY_CONFIGS
from kvbench.schema.phase6 import AUTHORIZED_CONTAINER_DIGEST


def _authorized_environment_declared() -> bool:
    return (
        Path("/.dockerenv").is_file()
        and os.environ.get(PHASE6_IMAGE_ENVIRONMENT_VARIABLE)
        == AUTHORIZED_CONTAINER_DIGEST
        and os.environ.get(PHASE6_CONTAINER_ENVIRONMENT_VARIABLE)
        == PHASE6_CONTAINER_ENVIRONMENT_VALUE
    )


@unittest.skipUnless(
    _authorized_environment_declared(),
    "Phase 6 CUDA is authorized only in the exact Measurement Container",
)
class Phase6TurboQuantCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = require_authorized_cuda_environment(
            AUTHORIZED_CONTAINER_DIGEST
        )

    def test_all_mandatory_fixture_paths_and_bf16_boundaries(self) -> None:
        for configuration in TURBOQUANT_MANDATORY_CONFIGS:
            with self.subTest(configuration=configuration):
                result = evaluate_fixture_configuration(configuration)
                self.assertTrue(result["passed"], result)
                self.assertTrue(result["store"]["passed"])
                self.assertTrue(result["append"]["passed"])
                self.assertTrue(result["appended_slot"]["passed"])
                self.assertTrue(result["decode"]["passed"])
                self.assertTrue(result["decode"]["finite"])
                self.assertTrue(result["slot_layout"]["passed"])
                self.assertTrue(result["bf16_boundary_store"]["passed"])
                self.assertTrue(result["bf16_boundary_append"]["passed"])
                self.assertTrue(result["pointers_stable"])
                self.assertLess(
                    result["predicted_allocated_relative_error"],
                    0.01,
                )
                self.assertEqual(
                    result["gqa_geometry"]["num_kv_heads"],
                    8,
                )
                self.assertEqual(
                    result["gqa_geometry"]["num_query_heads"],
                    32,
                )
                self.assertFalse(
                    result["gqa_geometry"]["gqa_materialized"]
                )
                self.assertIsNone(result["r_hbm"])


if __name__ == "__main__":
    unittest.main()

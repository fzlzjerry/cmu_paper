"""Checksum-bound replay of the exact-container Phase 13B CUDA matrix."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import unittest

from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VALUE,
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    require_authorized_cuda_environment,
)
from kvbench.schema.phase13b import PHASE13B_AUTHORIZED_CONTAINER_DIGEST
from scripts.phase13b_compressed_batch_admission import validate_cuda_matrix


def _authorized_environment_declared() -> bool:
    return (
        Path("/.dockerenv").is_file()
        and os.environ.get(PHASE6_IMAGE_ENVIRONMENT_VARIABLE)
        == PHASE13B_AUTHORIZED_CONTAINER_DIGEST
        and os.environ.get(PHASE6_CONTAINER_ENVIRONMENT_VARIABLE)
        == PHASE6_CONTAINER_ENVIRONMENT_VALUE
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@unittest.skipUnless(
    _authorized_environment_declared(),
    "Phase 13B CUDA is authorized only in the exact Measurement Container",
)
class Phase13BBatchGeometryCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = require_authorized_cuda_environment(
            PHASE13B_AUTHORIZED_CONTAINER_DIGEST
        )
        matrix_path = os.environ.get("KVBENCH_PHASE13B_CUDA_MATRIX")
        if matrix_path is None:
            raise AssertionError("KVBENCH_PHASE13B_CUDA_MATRIX is required")
        cls.matrix_path = Path(matrix_path).resolve(strict=True)
        cls.matrix = validate_cuda_matrix(cls.matrix_path)

    def test_all_twenty_seven_geometry_records_pass(self) -> None:
        self.assertEqual(self.matrix["point_count"], 27)
        for record in self.matrix["records"]:
            with self.subTest(
                configuration=record["configuration"],
                batch=record["batch_size"],
            ):
                self.assertEqual(record["status"], "PASS")
                self.assertTrue(all(record["checks"].values()))
                self.assertTrue(record["output_finite"])
                self.assertTrue(record["pointers_stable"])
                self.assertTrue(record["historical_prefix_unchanged"])
                self.assertIsNone(record["r_hbm"])
                self.assertFalse(record["timing_collected"])

    def test_matrix_source_hashes_match_the_container_checkout(self) -> None:
        for relative, expected in self.matrix["source_hashes"].items():
            with self.subTest(relative=relative):
                self.assertEqual(_sha256_file(Path(relative)), expected)

    def test_b1_and_batched_controls_are_explicit(self) -> None:
        for record in self.matrix["records"]:
            control = record["batch_numerical_control"]
            if record["batch_size"] == 1:
                self.assertEqual(
                    control["reference"],
                    "b1_frozen_fixture_and_source_preservation",
                )
            else:
                self.assertEqual(
                    control["reference"],
                    "identical_rows_within_same_batch_execution",
                )
                self.assertTrue(record["batch_bank_isolation"]["passed"])


if __name__ == "__main__":
    unittest.main()

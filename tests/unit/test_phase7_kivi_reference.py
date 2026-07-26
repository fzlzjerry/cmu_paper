"""Focused tests for the Phase 7 KIVI reference lane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from reference.kivi.generate_fixtures import (
    ReferenceGenerationError,
    _canonical_json,
    _write_no_replace,
)
from reference.kivi.validate_fixtures import (
    FixtureValidationError,
    validate_all,
)
from scripts.validate_phase2 import (
    APPROVED_ARTIFACT_ROOT_NAMES,
    PHASE7_APPROVED_ARTIFACT_ROOT_NAMES,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPOSITORY_ROOT / "reference/kivi"
FIXTURE_ROOT = REFERENCE_ROOT / "fixtures"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_variant(
    root: Path,
    variant: str,
    mutate: object,
) -> None:
    fixture_path = root / variant / "fixture.json"
    fixture = _load(fixture_path)
    mutate(fixture)  # type: ignore[operator]
    fixture_raw = _canonical_json(fixture)
    fixture_path.write_bytes(fixture_raw)

    manifest_path = root / variant / "manifest.json"
    manifest = _load(manifest_path)
    fixture_identity = manifest["fixture"]
    assert isinstance(fixture_identity, dict)
    fixture_identity["nbytes"] = len(fixture_raw)
    fixture_identity["sha256"] = _sha256(fixture_raw)
    manifest_raw = _canonical_json(manifest)
    manifest_path.write_bytes(manifest_raw)

    fixture_set_path = root / "fixture_set.json"
    fixture_set = _load(fixture_set_path)
    records = fixture_set["variant_manifests"]
    assert isinstance(records, list)
    record = next(
        item
        for item in records
        if isinstance(item, dict) and item["variant"] == variant
    )
    record["nbytes"] = len(manifest_raw)
    record["sha256"] = _sha256(manifest_raw)
    fixture_set_path.write_bytes(_canonical_json(fixture_set))
    _rewrite_ledger(root)


def _rewrite_ledger(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    ledger = "".join(
        f"{_sha256(path.read_bytes())}  {path.relative_to(root).as_posix()}\n"
        for path in files
    )
    (root / "checksums.sha256").write_text(ledger, encoding="utf-8")


class Phase7KiviReferenceTests(unittest.TestCase):
    def test_phase7_artifact_root_allowlist_is_exact(self) -> None:
        self.assertEqual(
            PHASE7_APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset({"phase7_kivi_reference"}),
        )
        allowed = (
            APPROVED_ARTIFACT_ROOT_NAMES
            | PHASE7_APPROVED_ARTIFACT_ROOT_NAMES
        )
        self.assertIn("phase7_kivi_reference", allowed)
        for rejected in (
            "phase7",
            "phase7_kivi",
            "phase7_kivi_reference_backup",
            "phase8_kivi_reference",
            "quality",
        ):
            with self.subTest(rejected=rejected):
                self.assertNotIn(rejected, allowed)

    def test_frozen_fixture_set_validates_without_regeneration(self) -> None:
        result = validate_all(FIXTURE_ROOT, check_image=False)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["configuration_count"], 4)
        self.assertEqual(result["mandatory_count"], 3)
        self.assertEqual(result["held_out_count"], 1)
        self.assertEqual(result["ledger_entries"], 9)
        self.assertFalse(result["performance_measurement"])
        self.assertFalse(result["r_hbm_populated"])

    def test_configuration_geometry_and_asymmetry_are_frozen(self) -> None:
        fixture_set = _load(FIXTURE_ROOT / "fixture_set.json")
        self.assertEqual(
            fixture_set["configurations"],
            ["k4v4", "k2v4", "k2v2", "k4v2"],
        )
        self.assertEqual(
            fixture_set["mandatory_configurations"],
            ["k4v4", "k2v4", "k2v2"],
        )
        self.assertEqual(
            fixture_set["held_out_configurations"],
            ["k4v2"],
        )
        for variant in fixture_set["configurations"]:
            fixture = _load(FIXTURE_ROOT / str(variant) / "fixture.json")
            configuration = fixture["configuration"]
            assert isinstance(configuration, dict)
            geometry = fixture["geometry"]
            assert isinstance(geometry, dict)
            self.assertEqual(geometry["batch_size"], 1)
            self.assertEqual(geometry["num_query_heads"], 32)
            self.assertEqual(geometry["num_kv_heads"], 8)
            self.assertEqual(geometry["head_dim"], 128)
            self.assertEqual(configuration["group_size"], 32)
            self.assertEqual(configuration["residual_length"], 32)
            self.assertEqual(geometry["seed"], 20260726)
            self.assertEqual(geometry["input_dtype"], "bfloat16")

    def test_rollover_boundary_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixtures"
            shutil.copytree(FIXTURE_ROOT, root)

            def mutate(fixture: dict[str, object]) -> None:
                rollover = fixture["rollover"]
                assert isinstance(rollover, dict)
                boundary = rollover["boundary"]
                assert isinstance(boundary, dict)
                state = boundary["state"]
                assert isinstance(state, dict)
                state["residual_key_tokens"] = [31]

            _rewrite_variant(root, "k4v4", mutate)
            with self.assertRaisesRegex(
                FixtureValidationError,
                "incorrect rollover boundary|missing or duplicated cache token",
            ):
                validate_all(root, check_image=False)

    def test_gqa_materialization_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixtures"
            shutil.copytree(FIXTURE_ROOT, root)

            def mutate(fixture: dict[str, object]) -> None:
                gqa = fixture["gqa"]
                assert isinstance(gqa, dict)
                gqa["expanded_temporary"] = True

            _rewrite_variant(root, "k2v4", mutate)
            with self.assertRaisesRegex(
                FixtureValidationError,
                "GQA contract mismatch",
            ):
                validate_all(root, check_image=False)

    def test_byte_accounting_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixtures"
            shutil.copytree(FIXTURE_ROOT, root)

            def mutate(fixture: dict[str, object]) -> None:
                records = fixture["byte_accounting"]
                assert isinstance(records, list)
                record = records[2]
                assert isinstance(record, dict)
                record["actual_total"] = int(record["actual_total"]) + 1

            _rewrite_variant(root, "k2v2", mutate)
            with self.assertRaisesRegex(
                FixtureValidationError,
                "byte accounting identity mismatch",
            ):
                validate_all(root, check_image=False)

    def test_r_hbm_population_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixtures"
            shutil.copytree(FIXTURE_ROOT, root)
            fixture_set_path = root / "fixture_set.json"
            fixture_set = _load(fixture_set_path)
            fixture_set["r_hbm_populated"] = True
            fixture_set_path.write_bytes(_canonical_json(fixture_set))
            _rewrite_ledger(root)
            with self.assertRaisesRegex(
                FixtureValidationError,
                "fixture-set manifest mismatch",
            ):
                validate_all(root, check_image=False)

    def test_no_replace_accepts_identity_and_refuses_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixtures"
            self.assertEqual(
                _write_no_replace(root, {"fixture.json": b"one\n"}),
                "created",
            )
            self.assertEqual(
                _write_no_replace(root, {"fixture.json": b"one\n"}),
                "existing_identical",
            )
            with self.assertRaisesRegex(
                ReferenceGenerationError,
                "overwrite refused",
            ):
                _write_no_replace(root, {"fixture.json": b"two\n"})
            self.assertEqual((root / "fixture.json").read_bytes(), b"one\n")

    def test_build_evidence_binds_required_cuda_controls(self) -> None:
        environment = _load(REFERENCE_ROOT / "environment.json")
        build = _load(REFERENCE_ROOT / "build_manifest.json")
        results = build["results"]
        assert isinstance(results, dict)
        for key in (
            "source_probe",
            "native_sm120",
            "sm120_cubin",
            "compute120_ptx",
            "forced_ptx_jit",
            "compute_sanitizer",
        ):
            self.assertTrue(results[key])
        self.assertFalse(results["no_kernel_image"])
        self.assertFalse(results["unsupported_fallback"])
        sanitizer = build["compute_sanitizer"]
        assert isinstance(sanitizer, dict)
        self.assertEqual(sanitizer["status"], "PASS")
        self.assertEqual(sanitizer["error_summary"], "0 errors")
        self.assertEqual(
            build["image"]["config_digest"],  # type: ignore[index]
            environment["image"]["config_digest"],  # type: ignore[index]
        )
        self.assertFalse(environment["measurement_container_modified"])
        self.assertFalse(environment["credentials_in_image"])
        self.assertFalse(environment["model_weights_in_image"])


if __name__ == "__main__":
    unittest.main()

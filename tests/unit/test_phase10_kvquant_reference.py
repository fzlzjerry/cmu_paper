"""Focused governance and numerical-fixture tests for Phase 10."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from reference.kvquant import generate_fixtures
from reference.kvquant import validate_fixtures
from scripts import validate_phase2


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "reference/kvquant/fixtures"
BLOCKED_REPORT_SHA256 = (
    "0362ac36f03ec8b92c0b154ad85fd4e810c3d45332f7c063b1da00d7adc94e4d"
)
DECISION_0021_SHA256 = (
    "e09cb0f7c59c07eb04ec28319d6705c436c9c25d466bbe63e2f1859cf75d4daf"
)
CALIBRATION_ROOT = (
    "8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf"
)
PHASE10_PASS_COMMIT = "dab34fc2671ad58b695a993b204bbc0b83a3d651"


class Phase10KVQuantReferenceTests(unittest.TestCase):
    def test_source_faithful_contract_is_exact(self) -> None:
        self.assertEqual(
            generate_fixtures.CASES,
            (
                ("key_zero_value_fixed12", 0),
                ("key_few_value_fixed12", 6),
                ("key_cap_value_fixed12", 12),
            ),
        )
        self.assertEqual(
            generate_fixtures.FAMILIES,
            (("kvq4", 4), ("kvq3", 3), ("kvq2", 2)),
        )
        source = (ROOT / "reference/kvquant/generate_fixtures.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("select_fixed_outliers", source)
        self.assertIn('"value_sparse_selection_mode": "fixed_extrema"', source)
        self.assertNotIn('"no_outlier"', source)
        self.assertNotIn('"few_outliers"', source)
        self.assertNotIn('"cap_reached"', source)
        self.assertNotIn("repeat_kv(", source)

    def test_reference_environment_adds_only_locked_tokenizers(self) -> None:
        dockerfile = (
            ROOT / "docker/reference-kvquant.Dockerfile"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "BASE_IMAGE=sha256:"
            "127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d",
            dockerfile,
        )
        self.assertIn("tokenizers-0.15.2-", dockerfile)
        self.assertIn(
            "9e0480c452217edd35eca56fafe2029fb4d368b7c0475f8dfa3c5c9c400a7456",
            dockerfile,
        )
        self.assertIn("--no-deps", dockerfile)
        self.assertIn("--no-index", dockerfile)
        self.assertNotIn("model", dockerfile.lower())
        self.assertNotIn("credential", dockerfile.lower())

    def test_authority_and_historical_blocked_report_are_unchanged(self) -> None:
        decision = (
            ROOT
            / "docs/decisions/0021-kvquant-patch-main-repository-custody.md"
        )
        self.assertEqual(
            hashlib.sha256(decision.read_bytes()).hexdigest(),
            DECISION_0021_SHA256,
        )
        report = ROOT / "docs/phase_reports/phase10-kvquant-reference-blocked.md"
        self.assertEqual(
            hashlib.sha256(report.read_bytes()).hexdigest(),
            BLOCKED_REPORT_SHA256,
        )
        config = (
            ROOT / "configs/methods/kvquant.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn(CALIBRATION_ROOT, config)

    def test_measurement_adapter_remains_fail_closed(self) -> None:
        factory = subprocess.check_output(
            [
                "git",
                "show",
                f"{PHASE10_PASS_COMMIT}:src/kvbench/adapters/factory.py",
            ],
            cwd=ROOT,
            text=True,
        )
        self.assertIn('_DEFERRED_METHODS = frozenset({"kvquant"})', factory)
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
                        f"{PHASE10_PASS_COMMIT}:{relative}",
                    ],
                    cwd=ROOT,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode,
                0,
            )

    def test_phase_boundaries_and_protected_methods_are_preserved(self) -> None:
        changed = validate_phase2.current_phase10_paths()
        forbidden_prefixes = (
            "src/kvbench/adapters/bf16",
            "src/kvbench/adapters/turboquant",
            "src/kvbench/adapters/kivi",
            "reference/turboquant/",
            "reference/kivi/",
            "docker/measurement.Dockerfile",
            "artifacts/profiler/",
            "artifacts/quality/",
        )
        self.assertFalse(
            any(
                path == prefix or path.startswith(prefix)
                for path in changed
                for prefix in forbidden_prefixes
            )
        )
        status = (ROOT / "docs/status.md").read_text(encoding="utf-8")
        self.assertIn("G2-KVQ remains NOT EVALUATED", status)
        self.assertIn("Full-scan admission: CLOSED", status)
        self.assertIn("Quality execution: LOCKED", status)
        self.assertFalse((ROOT / "PERFORMANCE_DATA_FROZEN").exists())

    def test_finalized_bundle_validates_without_regeneration(self) -> None:
        result = validate_fixtures.validate_fixture_bundle(FIXTURE_ROOT)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["fixture_count"], 9)
        self.assertEqual(result["g2_kvq"], "NOT_EVALUATED")
        self.assertFalse(result["performance_data"])
        self.assertFalse(result["profiler_data"])
        self.assertFalse(result["quality_data"])

    def test_all_value_and_key_counts_are_explicit(self) -> None:
        for family, _ in generate_fixtures.FAMILIES:
            for case_name, key_count in generate_fixtures.CASES:
                manifest = json.loads(
                    (
                        FIXTURE_ROOT
                        / family
                        / case_name
                        / "fixture_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                sparse = manifest["sparse_contract"]
                self.assertEqual(sparse["key_active_count"], key_count)
                self.assertEqual(sparse["key_capacity"], 12)
                self.assertEqual(sparse["value_active_count_non_sink"], 12)
                self.assertEqual(sparse["value_active_count_sink"], 0)
                self.assertEqual(
                    sparse["value_sparse_selection_mode"],
                    "fixed_extrema",
                )
                self.assertFalse(sparse["value_occupancy_data_dependent"])

    def test_byte_contract_keeps_hbm_null_and_ratios_reciprocal(self) -> None:
        for family, _ in generate_fixtures.FAMILIES:
            allocated_sparse = set()
            active_sparse = []
            for case_name, _ in generate_fixtures.CASES:
                payload = json.loads(
                    (
                        FIXTURE_ROOT
                        / family
                        / case_name
                        / "byte_breakdown.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertIsNone(payload["r_hbm"])
                self.assertLessEqual(
                    abs(payload["rho_alloc"] * payload["r_alloc"] - 1.0),
                    1e-9,
                )
                allocated_sparse.add(
                    payload["key_sparse_value_bytes"]
                    + payload["key_sparse_index_bytes"]
                    + payload["value_sparse_value_bytes"]
                    + payload["value_sparse_index_bytes"]
                )
                active_sparse.append(
                    payload["fixed_capacity"]["key_active_entries"]
                )
            self.assertEqual(len(allocated_sparse), 1)
            self.assertEqual(active_sparse, [0, 78, 156])

    def test_existing_finalized_bundle_cannot_be_overwritten(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                str(ROOT / "reference/kvquant/generate_fixtures.py"),
                "fixtures",
                "--source-root",
                "/unneeded/source",
                "--calibration-root",
                "/unneeded/calibration",
                "--patch-manifest",
                "/unneeded/manifest.json",
                "--extension",
                str(ROOT / "reference/kvquant/generate_fixtures.py"),
                "--reference-root",
                str(ROOT / "reference/kvquant"),
            ),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "FAIL")
        self.assertIn("already exists", payload["reason"])


if __name__ == "__main__":
    unittest.main()

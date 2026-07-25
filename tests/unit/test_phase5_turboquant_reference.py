"""Focused Phase 5 TurboQuant reference-lane regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from kvbench.adapters import MethodRuntimeContext, build_method_adapter
from kvbench.errors import ArtifactConflictError, ConfigLoadError
from reference.turboquant.generate_fixtures import publish_staged
from reference.turboquant.validate_fixtures import validate_reference
from scripts.validate_phase2 import PHASE5_ALLOWED_PATHS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPOSITORY_ROOT / "reference" / "turboquant"
PHASE5_ENTRY_SHA = "9eeabe787060e84c20cd7f88da8f7bca68eae1d4"
SOURCE_SHA = "752a3a504485790a2e8491cacbb35c137339ad34"
MANDATORY = [
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
]
OPTIONAL = ["turboquant_k8v4"]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context() -> MethodRuntimeContext:
    return MethodRuntimeContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="0" * 40,
        backend_id="torch_sdpa_flash_gqa",
        backend_fingerprint="a" * 64,
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


class Phase5SourceAndFixtureTests(unittest.TestCase):
    def test_source_lock_and_configuration_names_match(self) -> None:
        lock = _json(REPOSITORY_ROOT / "third_party" / "LOCK.json")
        source_manifest = _json(REFERENCE_ROOT / "source_manifest.json")
        source = source_manifest["source"]
        locked = next(item for item in lock["sources"] if item["id"] == "vllm_turboquant")
        self.assertEqual(source["commit"], SOURCE_SHA)
        self.assertEqual(source["commit"], locked["revision"])
        self.assertEqual(source["tree"], locked["tree"])
        self.assertEqual(source["repository"], locked["repository"])
        expected_files = {
            item["path"]: (item["git_blob"], item["sha256"])
            for item in locked["relevant_source_files"]
        }
        actual_files = {
            item["path"]: (item["git_blob"], item["sha256"])
            for item in source["relevant_source_files"]
        }
        self.assertEqual(actual_files, expected_files)
        configurations = source_manifest["configurations"]
        self.assertEqual(
            [item["cache_dtype"] for item in configurations],
            ["turboquant_k8v4", *MANDATORY],
        )
        self.assertEqual(
            [item["cache_dtype"] for item in configurations if item["phase5_role"] == "mandatory"],
            MANDATORY,
        )
        project = _json(REPOSITORY_ROOT / "configs/methods/turboquant.yaml")
        self.assertEqual(
            {item["variant_id"] for item in project["variants"]},
            {item["cache_dtype"] for item in configurations},
        )

    def test_fixture_schema_checksums_layout_and_actual_storage(self) -> None:
        result = validate_reference()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["mandatory_fixture_count"], 3)
        self.assertEqual(result["optional_fixture_count"], 1)
        self.assertFalse(result["timing_data_present"])
        self.assertFalse(result["performance_claim_eligible"])
        slots = {
            item["configuration"]: item["slot_size"]
            for item in result["configurations"]
        }
        self.assertEqual(
            slots,
            {
                "turboquant_k8v4": 196,
                "turboquant_4bit_nc": 134,
                "turboquant_k3v4_nc": 118,
                "turboquant_3bit_nc": 102,
            },
        )

    def test_finalized_fixture_set_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged = root / "staged"
            target = root / "target"
            staged.mkdir()
            target.mkdir()
            (staged / "manifest.json").write_bytes(b"new")
            (target / "manifest.json").write_bytes(b"final")
            before = _sha256(target / "manifest.json")
            with self.assertRaises(ArtifactConflictError):
                publish_staged(staged, target)
            self.assertEqual(_sha256(target / "manifest.json"), before)
            self.assertEqual((target / "manifest.json").read_bytes(), b"final")

    def test_identical_regeneration_verifies_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged = root / "staged"
            target = root / "target"
            staged.mkdir()
            target.mkdir()
            (staged / "manifest.json").write_bytes(b"same")
            (target / "manifest.json").write_bytes(b"same")
            before_inode = (target / "manifest.json").stat().st_ino
            self.assertEqual(publish_staged(staged, target), "verified_existing")
            self.assertEqual((target / "manifest.json").stat().st_ino, before_inode)


class Phase5GovernanceTests(unittest.TestCase):
    def test_phase6_blocked_report_has_exact_scope_exception(self) -> None:
        blocked_report = (
            "docs/phase_reports/phase6-turboquant-measurement-blocked.md"
        )
        self.assertIn(blocked_report, PHASE5_ALLOWED_PATHS)
        self.assertNotIn(
            "docs/phase_reports/arbitrary-phase-report.md",
            PHASE5_ALLOWED_PATHS,
        )
        self.assertNotIn(
            "src/kvbench/methods/turboquant/adapter.py",
            PHASE5_ALLOWED_PATHS,
        )
        for alternate in (
            f"../{blocked_report}",
            f"./{blocked_report}",
            "docs/phase_reports/../phase_reports/"
            "phase6-turboquant-measurement-blocked.md",
            "docs//phase_reports/phase6-turboquant-measurement-blocked.md",
            "docs/phase_reports/phase6-turboquant-measurement-blocked.md.bak",
            "docs/phase_reports/phase6-turboquant-measurement-blocked-extra.md",
        ):
            with self.subTest(alternate=alternate):
                self.assertNotIn(alternate, PHASE5_ALLOWED_PATHS)
        for existing in (
            "docs/phase_reports/phase5-turboquant-reference.md",
            "docs/status.md",
            "scripts/validate_phase2.py",
            "tests/unit/test_phase5_turboquant_reference.py",
        ):
            with self.subTest(existing=existing):
                self.assertIn(existing, PHASE5_ALLOWED_PATHS)

    def test_turboquant_requires_an_explicit_phase6_configuration(self) -> None:
        with self.assertRaises(ConfigLoadError):
            build_method_adapter("turboquant", _context())
        self.assertFalse((REPOSITORY_ROOT / "src/kvbench/methods/turboquant").exists())

    def test_quality_locked_full_scan_closed_and_no_formal_timing(self) -> None:
        status = (REPOSITORY_ROOT / "docs/status.md").read_text(encoding="utf-8")
        self.assertIn("Quality execution: LOCKED", status)
        self.assertIn("Full Scan remains CLOSED", status)
        self.assertFalse(any(REPOSITORY_ROOT.rglob("PERFORMANCE_DATA_FROZEN")))
        self.assertFalse((REPOSITORY_ROOT / "artifacts/quality").exists())
        for path in (REFERENCE_ROOT / "fixtures").rglob("*.json"):
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn('"latency"', text)
            self.assertNotIn('"throughput"', text)

    def test_old_phase4_phase3_and_e00_evidence_are_unchanged(self) -> None:
        result = subprocess.run(
            (
                "git",
                "diff",
                "--name-only",
                PHASE5_ENTRY_SHA,
                "--",
                "docs/evidence/e00",
                "docs/evidence/phase3",
                "docs/evidence/phase4",
                "artifacts/phase3",
                "artifacts/phase3_campaigns",
                "artifacts/phase3_reports",
                "docs/phase_reports/phase0.md",
                "docs/phase_reports/phase1-e00.md",
                "docs/phase_reports/phase1-e00-remediation.md",
                "docs/phase_reports/phase2.md",
                "docs/phase_reports/phase3.md",
                "docs/phase_reports/phase3-remediation.md",
                "docs/phase_reports/phase4.md",
            ),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()

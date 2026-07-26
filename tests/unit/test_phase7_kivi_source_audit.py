"""Focused regressions for the blocked Phase 7 KIVI source audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest

from kvbench.adapters.factory import build_method_adapter
from kvbench.errors import ErrorCode, PhaseNotImplementedError
from scripts.validate_phase2 import (
    PHASE6_ALLOWED_PATHS,
    PHASE7_ALLOWED_PATHS,
    PHASE7_ENTRY_COMMIT,
    current_phase7_paths,
    historical_phase6_paths,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = (
    REPOSITORY_ROOT / "docs/evidence/phase7/kivi-source-audit.json"
)
LOCK_PATH = REPOSITORY_ROOT / "third_party/LOCK.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
B019_ENTRY_COMMIT = "755c1bdb87af3e7becda792bd5d300ab877fee7e"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_at_commit(commit: str, path: str) -> str:
    result = subprocess.run(
        ("git", "show", f"{commit}:{path}"),
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot read {path} at {commit}")
    return hashlib.sha256(result.stdout).hexdigest()


class Phase7KiviSourceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = _load_json(AUDIT_PATH)
        cls.lock = _load_json(LOCK_PATH)

    def test_entry_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            PHASE7_ENTRY_COMMIT,
            "0974bbc98f8f941b09800786591108292dc4e0dd",
        )
        expected = frozenset(
            {
                "Makefile",
                "docs/blockers.md",
                (
                    "docs/decisions/"
                    "0017-kivi-source-authority-and-gqa-materialization.md"
                ),
                "docs/decisions/0018-kivi-b019-native-gqa-patch-authority.md",
                "docs/evidence/phase7/kivi-b019-remediation.json",
                "docs/evidence/phase7/kivi-source-audit.json",
                "docs/method_notes/kivi.md",
                "docs/phase_reports/phase7-kivi-b019-remediation.md",
                "docs/phase_reports/phase7-kivi-reference-blocked.md",
                "docs/plans/phase7-kivi-b019-remediation.md",
                "docs/plans/phase7-kivi-reference.md",
                "docs/risk_register.md",
                "docs/status.md",
                "docs/tasks.md",
                "scripts/validate_kivi_b019_patch.py",
                "scripts/validate_phase2.py",
                "tests/unit/test_phase7_kivi_b019_remediation.py",
                "tests/unit/test_phase7_kivi_source_audit.py",
                "third_party/LOCK.json",
                "third_party/NOTICE.md",
                "third_party/patches/kivi/0001-preserve-native-gqa-kv-storage.patch",
                "third_party/patches/kivi/manifest.json",
            }
        )
        self.assertEqual(PHASE7_ALLOWED_PATHS, expected)
        for rejected in (
            "docker/reference-kivi.Dockerfile",
            "reference/kivi/generate_fixtures.py",
            "reference/registry.py",
            "src/kvbench/adapters/kivi.py",
            "src/kvbench/runtime/kivi_cache.py",
            "artifacts/phase7/COMPLETE",
            "artifacts/quality/kivi.json",
            "docs/evidence/quality/kivi.json",
        ):
            with self.subTest(rejected=rejected):
                self.assertNotIn(rejected, PHASE7_ALLOWED_PATHS)

    def test_historical_and_current_scope_are_separate(self) -> None:
        self.assertLessEqual(historical_phase6_paths(), PHASE6_ALLOWED_PATHS)
        self.assertLessEqual(current_phase7_paths(), PHASE7_ALLOWED_PATHS)

    def test_source_authority_and_hash_bindings(self) -> None:
        audit = self.audit
        self.assertIsInstance(audit, dict)
        source = audit["source"]
        self.assertEqual(audit["status"], "BLOCKED")
        self.assertEqual(audit["run_kind"], "source_audit")
        self.assertEqual(
            source["repository"], "https://github.com/jy-yuan/KIVI.git"
        )
        self.assertEqual(
            source["revision"],
            "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6",
        )
        self.assertEqual(
            source["tree"],
            "c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b",
        )
        self.assertFalse(source["floating_branch_used"])
        self.assertFalse(source["unofficial_fork_used"])
        self.assertEqual(
            source["source_lock_sha256"],
            _sha256_at_commit(
                B019_ENTRY_COMMIT, "third_party/LOCK.json"
            ),
        )

        lock = self.lock
        self.assertIsInstance(lock, dict)
        kivi = next(
            item for item in lock["sources"] if item.get("id") == "kivi"
        )
        self.assertEqual(kivi["repository"], source["repository"])
        self.assertEqual(kivi["revision"], source["revision"])
        self.assertEqual(kivi["tree"], source["tree"])
        self.assertEqual(kivi["declared_license"], "MIT")
        files = kivi["relevant_source_files"]
        self.assertEqual(len(files), 15)
        self.assertEqual(
            len({item["path"] for item in files}),
            len(files),
        )
        for item in files:
            with self.subTest(path=item["path"]):
                self.assertRegex(item["git_blob"], GIT_OBJECT_PATTERN)
                self.assertRegex(item["sha256"], SHA256_PATTERN)

        for record_name in ("authority_decision", "plan"):
            record = source[record_name]
            self.assertEqual(
                record["sha256"],
                _sha256(REPOSITORY_ROOT / record["path"]),
            )

    def test_algorithm_and_rollover_source_findings_are_frozen(self) -> None:
        algorithm = self.audit["algorithm_source_audit"]
        self.assertEqual(
            algorithm["key_quantization_direction"],
            "per_channel_grouped_along_token_dimension",
        )
        self.assertEqual(
            algorithm["value_quantization_direction"],
            "per_token_grouped_along_head_dimension",
        )
        self.assertEqual(algorithm["packing_container"], "int32")
        self.assertEqual(algorithm["supported_cuda_bits"], [2, 4])
        self.assertEqual(algorithm["frozen_group_size"], 32)
        self.assertEqual(algorithm["frozen_residual_length"], 32)
        self.assertIn("reaches 32", algorithm["key_rollover"])
        self.assertIn("becomes 33", algorithm["value_rollover"])
        self.assertTrue(algorithm["dynamic_cache_growth_uses_torch_cat"])
        self.assertEqual(
            algorithm["runtime_layout_verification"], "NOT_EXECUTED"
        )

    def test_gqa_materialization_is_positive_blocking_evidence(self) -> None:
        gqa = self.audit["gqa_audit"]
        storage = gqa["storage_semantic_audit"]
        self.assertEqual((gqa["h_q"], gqa["h_kv"], gqa["n_rep"]), (32, 8, 4))
        self.assertTrue(gqa["repeat_kv"])
        self.assertTrue(gqa["repeat_interleave_equivalent"])
        self.assertTrue(gqa["expand_reshape"])
        self.assertTrue(gqa["residual_kv_expanded_to_h_q"])
        self.assertEqual(storage["input_shape"], [1, 8, 32, 128])
        self.assertEqual(storage["output_shape"], [1, 32, 32, 128])
        self.assertEqual(storage["input_storage_bytes"], 65_536)
        self.assertEqual(storage["output_storage_bytes"], 262_144)
        self.assertEqual(
            storage["output_storage_bytes"],
            4 * storage["input_storage_bytes"],
        )
        self.assertFalse(storage["same_storage"])
        self.assertTrue(storage["output_contiguous"])
        self.assertFalse(storage["performance_measurement"])
        self.assertEqual(
            gqa["final_verdict"], "BLOCKED_GQA_MATERIALIZATION"
        )
        self.assertEqual(self.audit["blocker"]["id"], "B-019")

    def test_runtime_work_was_not_started(self) -> None:
        execution = self.audit["execution"]
        self.assertTrue(execution)
        self.assertTrue(all(value is False for value in execution.values()))
        preservation = self.audit["preservation"]
        for key in (
            "kivi_measurement_adapter_created",
            "measurement_container_modified",
            "turboquant_adapter_modified",
            "turboquant_fixtures_modified",
            "common_runners_modified",
            "historical_evidence_modified",
            "phase8_started",
        ):
            with self.subTest(key=key):
                self.assertFalse(preservation[key])
        self.assertEqual(preservation["factory_status"], "phase_not_implemented")
        self.assertFalse((REPOSITORY_ROOT / "reference/kivi").exists())
        self.assertFalse(
            (REPOSITORY_ROOT / "docker/reference-kivi.Dockerfile").exists()
        )
        self.assertFalse(
            (REPOSITORY_ROOT / "src/kvbench/adapters/kivi.py").exists()
        )
        self.assertFalse((REPOSITORY_ROOT / "artifacts/phase7").exists())

    def test_kivi_adapter_remains_fail_closed(self) -> None:
        with self.assertRaises(PhaseNotImplementedError) as raised:
            build_method_adapter("kivi", None)  # type: ignore[arg-type]
        self.assertEqual(raised.exception.code, ErrorCode.PHASE_NOT_IMPLEMENTED)
        self.assertEqual(
            str(raised.exception),
            (
                "phase_not_implemented: "
                "kivi method adapter is deferred beyond Phase 6"
            ),
        )

    def test_protected_phase6_and_measurement_paths_are_unchanged(self) -> None:
        protected = (
            "docker/measurement.Dockerfile",
            "reference/turboquant",
            "src/kvbench/adapters/turboquant.py",
            "src/kvbench/runtime/cuda_graph.py",
            "src/kvbench/runtime/fixed_l_runner.py",
            "src/kvbench/runtime/growing_context_runner.py",
            "src/kvbench/runtime/timing.py",
            "docs/evidence/phase6",
        )
        result = subprocess.run(
            (
                "git",
                "diff",
                "--quiet",
                "--no-ext-diff",
                PHASE7_ENTRY_COMMIT,
                "--",
                *protected,
            ),
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        self.assertEqual(result.returncode, 0)

    def test_governance_and_claim_boundaries_remain_closed(self) -> None:
        entry = self.audit["entry"]
        self.assertEqual(entry["phase6"], "PASS")
        self.assertEqual(entry["g0"], "PASS")
        self.assertEqual(entry["g1"], "PASS")
        self.assertEqual(entry["g2_tq"], "PASS")
        self.assertEqual(entry["global_g2_g5"], "NOT_EVALUATED")
        self.assertEqual(entry["full_scan"], "CLOSED")
        self.assertEqual(entry["quality_execution"], "LOCKED")
        self.assertEqual(entry["performance_data_frozen"], "absent")
        self.assertFalse(
            (REPOSITORY_ROOT / "PERFORMANCE_DATA_FROZEN").exists()
        )
        self.assertTrue(all(value is False for value in self.audit["claims"].values()))
        self.assertFalse(self.audit["execution"]["r_hbm_populated"])
        self.assertEqual(
            set(entry["r2_required_variables"].values()), {"PRESENT"}
        )

        status = (REPOSITORY_ROOT / "docs/status.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Quality execution: LOCKED", status)
        self.assertIn("Full-scan admission: CLOSED", status)
        self.assertIn("Phase 7 status: PARTIAL", status)
        self.assertIn("B-019 is RESOLVED under patched-source authority", status)
        self.assertIn("Phase 8, Pilot", status)
        self.assertIn("KIVI Measurement Adapter remains fail-closed", status)

    def test_report_and_decision_state_the_stop_condition(self) -> None:
        report = (
            REPOSITORY_ROOT
            / "docs/phase_reports/phase7-kivi-reference-blocked.md"
        ).read_text(encoding="utf-8")
        decision = (
            REPOSITORY_ROOT
            / "docs/decisions/"
            "0017-kivi-source-authority-and-gqa-materialization.md"
        ).read_text(encoding="utf-8")
        plan = (
            REPOSITORY_ROOT / "docs/plans/phase7-kivi-reference.md"
        ).read_text(encoding="utf-8")
        self.assertTrue(report.startswith("PHASE 7 REPORT\n\nStatus: BLOCKED"))
        self.assertIn("BLOCKED_GQA_MATERIALIZATION", report)
        self.assertIn("Do not begin Phase 8", report)
        self.assertIn("Do not substitute `develop`, `lmeval`", decision)
        self.assertIn("Do not patch the official algorithm", decision)
        self.assertIn("Status: BLOCKED", plan)


if __name__ == "__main__":
    unittest.main()

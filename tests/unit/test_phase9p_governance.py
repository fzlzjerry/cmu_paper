"""Focused Phase 9P authority, scope, and preservation regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from scripts.validate_phase2 import (
    PHASE8_ALLOWED_PATHS,
    PHASE9P_ALLOWED_PATHS,
    PHASE9P_ENTRY_COMMIT,
    current_phase9p_paths,
    historical_phase8_paths,
    repository_python_paths,
)


ROOT = Path(__file__).resolve().parents[2]
PHASE9P_PASS_COMMIT = "1b3a98160ba4760007ca861c1a280def698b2027"
PATCH_MANIFEST_PATH = ROOT / "docs/evidence/phase9p/patch-manifest.json"
TEST_REPORT_PATH = ROOT / "docs/evidence/phase9p/test-report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


class Phase9PGovernanceTests(unittest.TestCase):
    def test_entry_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            PHASE9P_ENTRY_COMMIT,
            "f2c6475f09cdf6e9660552eb23c91b03e386aa59",
        )
        self.assertEqual(
            PHASE9P_ALLOWED_PATHS,
            frozenset(
                {
                    "Makefile",
                    "docs/decisions/0020-kvquant-upstream-gqa-patch.md",
                    "docs/evidence/phase9p/patch-manifest.json",
                    "docs/evidence/phase9p/test-report.json",
                    "docs/method_notes/kvquant.md",
                    "docs/phase_reports/phase9p-kvquant-upstream-gqa-patch.md",
                    "scripts/validate_phase2.py",
                    "tests/unit/test_phase7_kivi_b019_remediation.py",
                    "tests/unit/test_phase9p_governance.py",
                    "third_party/LOCK.json",
                    "third_party/NOTICE.md",
                }
            ),
        )
        self.assertLessEqual(historical_phase8_paths(), PHASE8_ALLOWED_PATHS)
        self.assertLessEqual(current_phase9p_paths(), PHASE9P_ALLOWED_PATHS)

    def test_phase9p_rejects_deferred_implementation_and_results(self) -> None:
        for rejected in (
            "configs/methods/kvquant.yaml",
            "docker/calibration-kvquant.Dockerfile",
            "docker/measurement.Dockerfile",
            "calibration/kvquant/example/COMPLETE",
            "reference/kvquant/fixture.json",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/kvquant.py",
            "third_party/patches/kvquant/0001.patch",
            "docs/phase_reports/phase9-kvquant-calibration.md",
            "artifacts/quality/kvquant.json",
            "artifacts/profiler/kvquant.ncu-rep",
        ):
            with self.subTest(rejected=rejected):
                self.assertNotIn(rejected, PHASE9P_ALLOWED_PATHS)

    def test_decision_fixes_private_patched_upstream_authority(self) -> None:
        decision = (
            ROOT / "docs/decisions/0020-kvquant-upstream-gqa-patch.md"
        ).read_text(encoding="utf-8")
        for required in (
            "57a238357f0ffe50084670fcd5781c9848f80ea2",
            "094e0f736f77ee327e5350cbd1eefb1c936aa77b",
            "kvquant_gqa_upstream_patch_v1",
            "KVQuant-GQA patched upstream",
            "not an official author-released GQA implementation",
            "local/private",
            "six entries per tail",
            "capacity of 12",
            "five initial K/V positions",
        ):
            self.assertIn(required, decision)
        self.assertIn("must not be committed to the main", decision)
        self.assertIn("uploaded to R2", decision)

    def test_historical_authorities_remain_byte_and_tree_identical(self) -> None:
        expected_hashes = {
            "configs/methods/bf16.yaml": (
                "fdffda79ca294ca7592f6ffc6033698b5875f3af9824970ccf08cf61af841fd8"
            ),
            "configs/methods/turboquant.yaml": (
                "a7e69050097820a455bb5086adf22d2adeb44068bec408504d8525f709260ec2"
            ),
            "configs/methods/kivi.yaml": (
                "5c48e8f0380f2c17750b25f91c721bbdbbb68385fa3d958b2b28cdd922716c81"
            ),
            "src/kvbench/adapters/factory.py": (
                "31e482c39f54319f2dbef814fdfa212283a8a3004dabda2248e14ea75cdf7672"
            ),
            "docker/measurement.Dockerfile": (
                "333a1e4264e8dc7798c5af06622fc97871371c3fc1e063f4a3b88cfb25389ace"
            ),
            "docs/phase_reports/phase8-kivi-measurement-adapter.md": (
                "63043780d7618ad684e1f64f28e785a941b578cd8b5fda59dd6e1b182b3a1dd2"
            ),
        }
        for relative, expected in expected_hashes.items():
            with self.subTest(relative=relative):
                if relative == "src/kvbench/adapters/factory.py":
                    payload = subprocess.check_output(
                        [
                            "git",
                            "show",
                            f"{PHASE9P_PASS_COMMIT}:{relative}",
                        ],
                        cwd=ROOT,
                    )
                    self.assertEqual(
                        hashlib.sha256(payload).hexdigest(),
                        expected,
                    )
                else:
                    self.assertEqual(_sha256(ROOT / relative), expected)

        expected_trees = {
            "docs/evidence/e00": "d1d38baddc5de52ca623f16c04327fcb2829369e",
            "docs/evidence/phase6": "539996d350278436f3a7ef68ee964fc9a2d6248f",
            "docs/evidence/phase7": "14ff49484fd9c43547f822e5607b62a854bc824e",
            "docs/evidence/phase8": "2c2910d68703741834ee54e1e5ed1a9298cb8e46",
            "reference/turboquant": "7b4297645774a2faaca79a8a86a47d3308f9faf1",
            "reference/kivi": "046d25c1f3669c5e9657065a27ed1ebf90e88186",
        }
        for relative, expected in expected_trees.items():
            with self.subTest(relative=relative):
                self.assertEqual(
                    _git("rev-parse", f"{PHASE9P_ENTRY_COMMIT}:{relative}"),
                    expected,
                )
                self.assertEqual(
                    subprocess.run(
                        [
                            "git",
                            "diff",
                            "--quiet",
                            PHASE9P_ENTRY_COMMIT,
                            "--",
                            relative,
                        ],
                        cwd=ROOT,
                        check=False,
                    ).returncode,
                    0,
                )

    def test_kvquant_factory_remains_fail_closed_after_reference_lane(
        self,
    ) -> None:
        factory = _git(
            "show",
            f"{PHASE9P_PASS_COMMIT}:src/kvbench/adapters/factory.py",
        )
        self.assertIn('_DEFERRED_METHODS = frozenset({"kvquant"})', factory)
        self.assertFalse((ROOT / "PERFORMANCE_DATA_FROZEN").exists())
        self.assertTrue((ROOT / "reference/kvquant").is_dir())
        self.assertTrue(
            (ROOT / "reference/kvquant/fixtures/COMPLETE").is_file()
        )
        for relative in (
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/kvquant.py",
        ):
            result = subprocess.run(
                [
                    "git",
                    "cat-file",
                    "-e",
                    f"{PHASE9P_PASS_COMMIT}:{relative}",
                ],
                cwd=ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_phase9p_evidence_is_compact_checksum_bound_and_non_claiming(self) -> None:
        manifest = _load_json(PATCH_MANIFEST_PATH)
        report = _load_json(TEST_REPORT_PATH)

        self.assertEqual(manifest["schema"], "kvbench.phase9p.patch_manifest.v1")
        self.assertEqual(report["schema"], "kvbench.phase9p.test_report.v1")
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            manifest["authority"]["method_identifier"],
            "kvquant_gqa_upstream_patch_v1",
        )
        self.assertFalse(manifest["authority"]["official_gqa_support_claimed"])
        self.assertEqual(
            manifest["authority"]["root_license_status"],
            "unresolved_no_root_license",
        )
        self.assertEqual(
            manifest["source"]["base_commit"],
            "57a238357f0ffe50084670fcd5781c9848f80ea2",
        )
        self.assertEqual(
            manifest["source"]["base_tree"],
            "094e0f736f77ee327e5350cbd1eefb1c936aa77b",
        )
        self.assertEqual(manifest["source"]["branch"], "kvbench/llama31-gqa")
        self.assertTrue(manifest["source"]["clean"])
        self.assertEqual(len(manifest["source"]["patched_commit"]), 40)
        self.assertEqual(len(manifest["source"]["patched_tree"]), 40)
        self.assertEqual(len(manifest["patch"]["aggregate_sha256"]), 64)
        self.assertGreater(len(manifest["patch"]["changed_files"]), 0)
        for entry in manifest["patch"]["changed_files"]:
            if entry["change_type"] == "added":
                self.assertIsNone(entry["before_sha256"])
            else:
                self.assertEqual(len(entry["before_sha256"]), 64)
            self.assertEqual(len(entry["after_sha256"]), 64)
            self.assertNotIn("content", entry)
            self.assertNotIn("patch", entry)

        target = manifest["target"]
        self.assertEqual(target["model_revision"], "0e9e39f249a16976918f6564b8830bc894c89659")
        self.assertEqual(target["num_query_heads"], 32)
        self.assertEqual(target["num_kv_heads"], 8)
        self.assertEqual(target["num_kv_groups"], 4)
        self.assertEqual(target["head_dim"], 128)
        self.assertEqual(target["kv_width"], 1024)
        self.assertEqual(target["key_outlier_cap"], 12)
        self.assertEqual(target["value_outlier_cap"], 12)
        self.assertEqual(target["sink_tokens"], 5)

        self.assertEqual(
            manifest["test_evidence"]["sha256"], _sha256(TEST_REPORT_PATH)
        )
        publication = manifest["publication"]
        self.assertFalse(publication["source_published"])
        self.assertFalse(publication["patch_published"])
        self.assertFalse(publication["r2_attempted"])
        self.assertFalse(publication["credentials_forwarded"])

        self.assertFalse(report["run_kind"]["performance_measurement"])
        self.assertFalse(report["run_kind"]["quality_evaluation"])
        self.assertFalse(report["run_kind"]["full_phase9_calibration"])
        for name in (
            "loader",
            "rope",
            "geometry",
            "hooks",
            "outlier",
            "gqa_numerical",
            "mha_regression",
            "cache",
            "sink",
            "fisher_smoke",
            "quantizer_smoke",
            "sm120_native",
            "forced_ptx_jit",
            "sanitizer_mha",
            "sanitizer_gqa",
            "sanitizer_cap_reached",
            "cuda_graph",
            "allocation",
            "historical_regression",
        ):
            with self.subTest(name=name):
                self.assertEqual(report["tests"][name]["status"], "PASS")

        self.assertEqual(report["gates"]["g0"], "PASS")
        self.assertEqual(report["gates"]["g1"], "PASS")
        self.assertEqual(report["gates"]["g2_tq"], "PASS")
        self.assertEqual(report["gates"]["g2_kivi"], "PASS")
        self.assertEqual(report["gates"]["g2_kvq"], "NOT_EVALUATED")
        self.assertEqual(report["gates"]["global_g2_g5"], "NOT_EVALUATED")
        self.assertEqual(report["gates"]["full_scan"], "CLOSED")
        self.assertEqual(report["gates"]["quality_execution"], "LOCKED")

    def test_phase9p_python_test_is_governed(self) -> None:
        governed = {
            path.relative_to(ROOT).as_posix() for path in repository_python_paths()
        }
        self.assertIn("tests/unit/test_phase9p_governance.py", governed)


if __name__ == "__main__":
    unittest.main()

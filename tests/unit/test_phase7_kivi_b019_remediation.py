"""Focused regressions for the Phase 7 KIVI B-019 source remediation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from kvbench.adapters.factory import build_method_adapter
from kvbench.errors import ErrorCode, PhaseNotImplementedError
from scripts.validate_kivi_b019_patch import _run_git, validate


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPOSITORY_ROOT / "third_party/patches/kivi/manifest.json"
)
PATCH_PATH = (
    REPOSITORY_ROOT
    / "third_party/patches/kivi/"
    "0001-preserve-native-gqa-kv-storage.patch"
)
EVIDENCE_PATH = (
    REPOSITORY_ROOT / "docs/evidence/phase7/kivi-b019-remediation.json"
)
LOCK_PATH = REPOSITORY_ROOT / "third_party/LOCK.json"
B019_ENTRY_COMMIT = "755c1bdb87af3e7becda792bd5d300ab877fee7e"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase7KiviB019RemediationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _load_json(MANIFEST_PATH)
        cls.evidence = _load_json(EVIDENCE_PATH)
        cls.lock = _load_json(LOCK_PATH)

    def test_manifest_lock_and_evidence_bind_the_exact_patch(self) -> None:
        manifest = self.manifest
        self.assertEqual(
            manifest["source"]["base_commit"],
            "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6",
        )
        self.assertEqual(
            manifest["source"]["base_tree"],
            "c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b",
        )
        self.assertEqual(
            manifest["source"]["patched_tree"],
            "b617493dea5aff1a754cd27ad6be12ac512b2aee",
        )
        self.assertEqual(manifest["patch"]["sha256"], _sha256(PATCH_PATH))

        kivi_lock = next(
            item
            for item in self.lock["sources"]
            if item.get("id") == "kivi"
        )
        source_patch = kivi_lock["source_patch"]
        self.assertEqual(
            source_patch["manifest_sha256"], _sha256(MANIFEST_PATH)
        )
        self.assertEqual(
            source_patch["patch_sha256"], _sha256(PATCH_PATH)
        )
        self.assertEqual(
            source_patch["patched_tree"],
            manifest["source"]["patched_tree"],
        )
        self.assertFalse(source_patch["unofficial_fork"])
        self.assertEqual(source_patch["upstream_status"], "not_merged")

        authority = self.evidence["patch_authority"]
        self.assertEqual(authority["manifest"]["sha256"], _sha256(MANIFEST_PATH))
        self.assertEqual(authority["patch"]["sha256"], _sha256(PATCH_PATH))
        self.assertEqual(
            authority["source_lock"]["sha256"], _sha256(LOCK_PATH)
        )

    def test_patch_is_parseable_narrow_and_forbidden_operation_free(self) -> None:
        result = subprocess.run(
            ("git", "apply", "--numstat", str(PATCH_PATH)),
            cwd=REPOSITORY_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "120\t0\tmodels/kivi_gqa.py",
                "12\t3\tmodels/llama_kivi.py",
            ],
        )

        added_lines = [
            line[1:]
            for line in PATCH_PATH.read_text(encoding="utf-8").splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        added_source = "\n".join(added_lines)
        self.assertNotIn("repeat_kv(", added_source)
        self.assertNotIn("repeat_interleave(", added_source)
        self.assertNotIn(".expand(", added_source)
        self.assertEqual(added_source.count("torch.bmm("), 2)

    def test_git_output_preserves_porcelain_status_prefix(self) -> None:
        status = _run_git(
            REPOSITORY_ROOT,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        if status:
            self.assertIn(
                status[0], {" ", "M", "A", "D", "R", "C", "U", "?"}
            )

    def test_cpu_semantics_head_mapping_and_native_kv_operands(self) -> None:
        result = validate(device="cpu")
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["performance_measurement"])
        semantics = result["semantics"]
        self.assertEqual((semantics["h_q"], semantics["h_kv"]), (32, 8))
        self.assertEqual(
            semantics["head_mapping"],
            [head // 4 for head in range(32)],
        )
        self.assertTrue(semantics["unsupported_geometry_rejected"])
        self.assertEqual(
            {
                (case["query_length"], case["key_length"])
                for case in semantics["cases"]
            },
            {(3, 17), (1, 33)},
        )
        for case in semantics["cases"]:
            with self.subTest(case=case):
                self.assertTrue(case["query_key_bf16_exact"])
                self.assertTrue(case["attention_value_bf16_exact"])
                self.assertFalse(case["expanded_kv_operand"])
                self.assertEqual(case["key_shape"][1], 8)
                self.assertEqual(case["value_shape"][1], 8)
                for operands in case["bmm_operands"]:
                    self.assertEqual(operands["left_shape"][0], 8)
                    self.assertEqual(operands["right_shape"][0], 8)

    def test_only_gqa_glue_changed_and_phase8_remains_unstarted(self) -> None:
        semantic_scope = self.manifest["semantic_scope"]
        for key in (
            "quantization_changed",
            "packing_changed",
            "cache_layout_changed",
            "rollover_changed",
            "cuda_extension_changed",
            "attention_score_geometry_changed",
            "attention_output_geometry_changed",
        ):
            with self.subTest(key=key):
                self.assertFalse(semantic_scope[key])
        self.assertTrue(semantic_scope["residual_kv_materialization_removed"])

        boundary = self.evidence["execution_boundary"]
        self.assertTrue(all(value is False for value in boundary.values()))
        outcome = self.evidence["outcome"]
        self.assertEqual(
            outcome["blocker_status"],
            "RESOLVED_UNDER_PATCHED_SOURCE_AUTHORITY",
        )
        self.assertEqual(outcome["phase7_status"], "PARTIAL_REMEDIATION_ONLY")
        self.assertEqual(outcome["g2_kivi"], "NOT_EVALUATED")
        self.assertFalse(outcome["phase8_started"])

    def test_measurement_turboquant_and_historical_evidence_are_unchanged(
        self,
    ) -> None:
        protected = (
            "docker/measurement.Dockerfile",
            "reference/turboquant",
            "src/kvbench/adapters/turboquant.py",
            "src/kvbench/runtime/cuda_graph.py",
            "src/kvbench/runtime/fixed_l_runner.py",
            "src/kvbench/runtime/growing_context_runner.py",
            "src/kvbench/runtime/timing.py",
            "docs/evidence/phase6",
            "docs/evidence/phase7/kivi-source-audit.json",
            "docs/phase_reports/phase7-kivi-reference-blocked.md",
            "docs/decisions/0017-kivi-source-authority-and-gqa-materialization.md",
        )
        result = subprocess.run(
            (
                "git",
                "diff",
                "--quiet",
                "--no-ext-diff",
                B019_ENTRY_COMMIT,
                "--",
                *protected,
            ),
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        self.assertEqual(result.returncode, 0)

    def test_kivi_measurement_adapter_remains_fail_closed(self) -> None:
        with self.assertRaises(PhaseNotImplementedError) as raised:
            build_method_adapter("kivi", None)  # type: ignore[arg-type]
        self.assertEqual(raised.exception.code, ErrorCode.PHASE_NOT_IMPLEMENTED)
        self.assertFalse(
            (REPOSITORY_ROOT / "src/kvbench/adapters/kivi.py").exists()
        )
        self.assertTrue(
            (REPOSITORY_ROOT / "docker/reference-kivi.Dockerfile").exists()
        )
        self.assertTrue((REPOSITORY_ROOT / "reference/kivi").is_dir())


if __name__ == "__main__":
    unittest.main()

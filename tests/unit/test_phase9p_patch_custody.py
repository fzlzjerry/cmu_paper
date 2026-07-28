"""Focused regressions for durable KVQuant patch custody in main."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import scripts.validate_kvquant_gqa_patch as patch_validator
from scripts.validate_phase2 import (
    KVQUANT_PATCH_CUSTODY_ALLOWED_PATHS,
    PHASE9P_FINAL_COMMIT,
    commit_is_ancestor,
    current_kvquant_patch_custody_paths,
    historical_phase9p_paths,
    PHASE9P_ALLOWED_PATHS,
)
from scripts.validate_kvquant_gqa_patch import (
    _git_environment,
    _validate_frozen_authority,
    ValidationError,
    validate,
)


ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = ROOT / "third_party/patches/kvquant"
PATCH_PATH = PATCH_DIR / "0001-llama31-native-gqa.patch"
MANIFEST_PATH = PATCH_DIR / "manifest.json"
FROZEN_PHASE9P_MANIFEST = (
    ROOT / "docs/evidence/phase9p/patch-manifest.json"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase9PPatchCustodyTests(unittest.TestCase):
    def test_historical_phase9p_scope_is_frozen(self) -> None:
        self.assertEqual(
            PHASE9P_FINAL_COMMIT,
            "1b3a98160ba4760007ca861c1a280def698b2027",
        )
        self.assertLessEqual(historical_phase9p_paths(), PHASE9P_ALLOWED_PATHS)
        self.assertTrue(commit_is_ancestor(PHASE9P_FINAL_COMMIT))
        scope_source = (ROOT / "scripts/validate_phase2.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "if not commit_is_ancestor(PHASE9P_FINAL_COMMIT):",
            scope_source,
        )
        self.assertLessEqual(
            current_kvquant_patch_custody_paths(),
            KVQUANT_PATCH_CUSTODY_ALLOWED_PATHS,
        )

    def test_patch_bytes_match_frozen_phase9p_authority(self) -> None:
        manifest = _load_json(MANIFEST_PATH)
        frozen = _load_json(FROZEN_PHASE9P_MANIFEST)
        self.assertEqual(
            _sha256(PATCH_PATH),
            "db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6",
        )
        self.assertEqual(PATCH_PATH.stat().st_size, 289239)
        self.assertEqual(
            _sha256(MANIFEST_PATH),
            "85e76396f058844190620e1cc7d2eef6afba37e83aca87d44f7c5e99c79b7539",
        )
        self.assertEqual(
            manifest["patch"]["sha256"],
            frozen["patch"]["aggregate_sha256"],
        )
        self.assertEqual(
            manifest["source"]["patched_commit"],
            frozen["source"]["patched_commit"],
        )
        self.assertEqual(
            manifest["source"]["patched_tree"],
            frozen["source"]["patched_tree"],
        )
        self.assertEqual(
            manifest["source"]["patched_commit_role"],
            "historical_validation_identity_not_required_for_reconstruction",
        )
        self.assertEqual(
            manifest["source"]["durable_execution_authority"],
            "base_commit_plus_patch_sha256_plus_patched_tree",
        )

        current_files = manifest["patched_files"]
        frozen_files = frozen["patch"]["changed_files"]
        self.assertEqual(
            [record["path"] for record in current_files],
            [record["path"] for record in frozen_files],
        )
        for current, original in zip(
            current_files,
            frozen_files,
            strict=True,
        ):
            self.assertEqual(
                current["base_sha256"],
                original["before_sha256"],
            )
            self.assertEqual(
                current["patched_sha256"],
                original["after_sha256"],
            )

    def test_static_validator_passes_without_separate_checkout(self) -> None:
        report = validate()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["reconstruction"], "NOT_REQUESTED")
        self.assertEqual(len(report["changed_paths"]), 15)

        completed = subprocess.run(
            [sys.executable, "scripts/validate_kvquant_gqa_patch.py"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "PASS")

    def test_static_validator_rejects_frozen_authority_drift(self) -> None:
        manifest = _load_json(MANIFEST_PATH)
        patch_bytes = PATCH_PATH.read_bytes()
        mutations = (
            ("repository", "source", "repository", "https://example.invalid"),
            ("base commit", "source", "base_commit", "0" * 40),
            ("base tree", "source", "base_tree", "1" * 40),
            ("patched commit", "source", "patched_commit", "2" * 40),
            ("patched tree", "source", "patched_tree", "3" * 40),
            ("method", "authority", "method_identifier", "wrong"),
        )
        for label, section, key, value in mutations:
            with self.subTest(label=label):
                candidate = deepcopy(manifest)
                candidate[section][key] = value
                with self.assertRaises(ValidationError):
                    _validate_frozen_authority(candidate, patch_bytes)

        candidate = deepcopy(manifest)
        candidate["patched_files"][0]["base_sha256"] = "4" * 64
        with self.assertRaises(ValidationError):
            _validate_frozen_authority(candidate, patch_bytes)

    def test_git_environment_rejects_command_scope_injection(self) -> None:
        hostile = {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.autocrlf",
            "GIT_CONFIG_VALUE_0": "true",
            "GIT_INDEX_FILE": "/tmp/hostile-index",
            "GIT_DIR": "/tmp/hostile-git-dir",
            "GIT_WORK_TREE": "/tmp/hostile-work-tree",
            "GIT_CONFIG_GLOBAL": "/tmp/hostile-global-config",
            "LC_ALL": "hostile",
            "LANG": "hostile",
        }
        with patch.dict(os.environ, hostile, clear=False):
            environment = _git_environment()

        for key in (
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_KEY_0",
            "GIT_CONFIG_VALUE_0",
            "GIT_INDEX_FILE",
            "GIT_DIR",
            "GIT_WORK_TREE",
        ):
            self.assertNotIn(key, environment)
        self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertEqual(environment["GIT_CONFIG_GLOBAL"], "/dev/null")
        self.assertEqual(environment["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(environment["LC_ALL"], "C")
        self.assertEqual(environment["LANG"], "C")

    def test_lock_join_rejects_authority_and_type_drift(self) -> None:
        manifest = _load_json(MANIFEST_PATH)
        patch_bytes = PATCH_PATH.read_bytes()
        lock = _load_json(ROOT / "third_party/LOCK.json")
        locked = next(
            record for record in lock["sources"] if record["id"] == "kvquant"
        )
        mutations = (
            (("repository",), "https://example.invalid"),
            (("phase9p_patch", "decision"), "docs/decisions/wrong.md"),
            (
                ("phase9p_patch", "custody_decision"),
                "docs/decisions/wrong.md",
            ),
            (("phase9p_patch", "branch"), "main"),
            (
                ("phase9p_patch", "patch_manifest"),
                "docs/evidence/phase9p/wrong.json",
            ),
            (("phase9p_patch", "patch_manifest_sha256"), "0" * 64),
            (("phase9p_patch", "scope"), "wrong"),
            (("phase9p_patch", "official_gqa_support_claimed"), True),
            (("phase9p_patch", "source_tree_published"), 0),
            (("phase9p_patch", "patch_published"), "false"),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                candidate = deepcopy(locked)
                target = candidate
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with patch.object(
                    patch_validator,
                    "_kvquant_lock",
                    return_value=candidate,
                ):
                    with self.assertRaises(ValidationError):
                        _validate_frozen_authority(manifest, patch_bytes)

    def test_only_patch_and_manifest_are_vendored(self) -> None:
        self.assertEqual(
            {
                path.relative_to(PATCH_DIR).as_posix()
                for path in PATCH_DIR.rglob("*")
                if path.is_file()
            },
            {
                "0001-llama31-native-gqa.patch",
                "manifest.json",
            },
        )
        self.assertFalse((PATCH_DIR / ".git").exists())
        self.assertFalse(any((ROOT / "third_party").rglob("*.so")))
        self.assertFalse((ROOT / "third_party/kvquant-gqa").exists())

    def test_operator_authority_and_unresolved_license_are_explicit(self) -> None:
        manifest = _load_json(MANIFEST_PATH)
        authority = manifest["authority"]
        publication = manifest["publication"]
        self.assertTrue(
            authority["public_patch_custody_authorized_by_operator"]
        )
        self.assertEqual(
            authority["root_license_status"],
            "unresolved_no_root_license",
        )
        self.assertFalse(authority["official_gqa_support_claimed"])
        self.assertEqual(
            publication["destination"],
            "https://github.com/fzlzjerry/cmu_paper",
        )
        self.assertEqual(publication["branch"], "main")
        self.assertTrue(publication["patch_publication_authorized"])
        self.assertFalse(publication["upstream_source_tree_vendored"])
        self.assertFalse(publication["license_or_lineage_resolved"])

    def test_completed_phase9p_evidence_remains_unchanged(self) -> None:
        protected = (
            "docs/decisions/0020-kvquant-upstream-gqa-patch.md",
            "docs/evidence/phase9p/patch-manifest.json",
            "docs/evidence/phase9p/test-report.json",
            "docs/phase_reports/phase9p-kvquant-upstream-gqa-patch.md",
        )
        completed = subprocess.run(
            ["git", "diff", "--quiet", PHASE9P_FINAL_COMMIT, "--", *protected],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)

    def test_phase_boundaries_remain_closed(self) -> None:
        manifest = _load_json(MANIFEST_PATH)
        boundaries = manifest["phase_boundaries"]
        self.assertFalse(boundaries["full_phase9_calibration_started"])
        self.assertFalse(boundaries["phase10_started"])
        self.assertFalse(boundaries["kvquant_adapter_enabled"])
        self.assertFalse(boundaries["performance_or_profiler_data_created"])
        self.assertFalse(boundaries["quality_data_created"])
        factory = (ROOT / "src/kvbench/adapters/factory.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('_DEFERRED_METHODS = frozenset({"kvquant"})', factory)
        self.assertFalse((ROOT / "PERFORMANCE_DATA_FROZEN").exists())


if __name__ == "__main__":
    unittest.main()

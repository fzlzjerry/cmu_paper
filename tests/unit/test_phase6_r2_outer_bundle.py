"""Focused tests for the append-only Phase 6 R2 outer bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import unittest

from scripts.phase6_r2_outer_bundle import (
    ADMISSION_REFERENCES_PATH,
    ORIGINAL_REFERENCE_PATH,
    OUTER_RECEIPT_RELATIVE,
    REQUIRED_REPOSITORY_FILES,
    OuterBundleError,
    build_outer_bundle,
    validate_outer_bundle,
)
from scripts.r2_artifact import (
    publication_order,
    validate_local_artifact,
)
from tests.unit.test_r2_artifact import finalized_artifact


RUN_IDS = tuple(f"phase6-test-run-{index}" for index in range(9))
SOURCE_GIT_SHA = "a" * 40


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _make_writable(root: Path) -> None:
    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o644)
        elif path.is_dir():
            path.chmod(0o755)


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


class Phase6R2OuterBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repository = Path(self.temporary.name) / "repository"
        source_parent = self.repository / "artifacts" / "phase6"
        source_parent.mkdir(parents=True)
        bounded = {
            "schema_version": "phase6-test-bounded-grid-1.0.0",
            "run_ids": list(RUN_IDS),
            "embedded_run_ids": list(RUN_IDS[1:]),
        }
        extra_files = {
            "validation/bounded-grid.json": _json_bytes(bounded),
        }
        for run_id in RUN_IDS[1:]:
            extra_files[f"grid-runs/{run_id}/manifest.json"] = _json_bytes(
                {"run_id": run_id, "status": "completed"}
            )
            extra_files[f"grid-runs/{run_id}/COMPLETE"] = _json_bytes(
                {"run_id": run_id, "status": "completed"}
            )
        self.source = finalized_artifact(
            source_parent / RUN_IDS[0],
            extra_files=extra_files,
        )
        for index, relative in enumerate(REQUIRED_REPOSITORY_FILES):
            target = self.repository / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                f"phase6 outer required evidence {index}\n".encode("utf-8")
            )
        self.original = validate_local_artifact(self.source)
        self.copy_prefix = PurePosixPath(
            "original", "sha256", self.original.root_sha256
        )
        self.original_uri = (
            f"r2://test/sha256/{self.original.root_sha256}/"
        )
        self.output_root = (
            self.repository / "artifacts" / "phase6_r2_outer"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, run_id: str = "phase6-r2-outer-test") -> Path:
        final, validation = build_outer_bundle(
            repository_root=self.repository,
            source_bundle=self.source,
            output_root=self.output_root,
            run_id=run_id,
            source_git_sha=SOURCE_GIT_SHA,
            expected_original_root=self.original.root_sha256,
            expected_run_ids=RUN_IDS,
            original_uri=self.original_uri,
            copy_prefix=self.copy_prefix,
        )
        self.assertEqual(validation.run_id, run_id)
        self.assertEqual(validation.admission_run_count, 9)
        return final

    def _validate(self, final: Path):
        return validate_outer_bundle(
            final,
            repository_root=self.repository,
            source_bundle=self.source,
            expected_original_root=self.original.root_sha256,
            expected_run_ids=RUN_IDS,
            copy_prefix=self.copy_prefix,
        )

    def test_complete_original_and_required_evidence_are_bound(self) -> None:
        original_hashes = {
            item.relative_path: item.sha256 for item in self.original.files
        }
        final = self._build()
        validation = self._validate(final)
        expected_count = len(self.original.files) + 3 + 2 + 4
        self.assertEqual(validation.object_count, expected_count)
        self.assertEqual(
            publication_order(validate_local_artifact(final))[-1].relative_path,
            "COMPLETE",
        )
        for relative, digest in original_hashes.items():
            copied = final / self.copy_prefix / relative
            self.assertEqual(
                hashlib.sha256(copied.read_bytes()).hexdigest(),
                digest,
            )
        for relative in REQUIRED_REPOSITORY_FILES:
            self.assertEqual(
                (final / relative).read_bytes(),
                (self.repository / relative).read_bytes(),
            )
        self.assertTrue((final / ORIGINAL_REFERENCE_PATH).is_file())
        self.assertTrue((final / ADMISSION_REFERENCES_PATH).is_file())
        self.assertFalse((final / OUTER_RECEIPT_RELATIVE).exists())
        self.assertEqual(
            {
                item.relative_path: item.sha256
                for item in validate_local_artifact(self.source).files
            },
            original_hashes,
        )

    def test_existing_final_is_never_replaced(self) -> None:
        final = self._build()
        root_before = validate_local_artifact(final).root_sha256
        with self.assertRaises(OuterBundleError):
            self._build()
        self.assertEqual(
            validate_local_artifact(final).root_sha256,
            root_before,
        )

    def test_incorrect_admission_run_list_is_rejected(self) -> None:
        with self.assertRaises(OuterBundleError):
            build_outer_bundle(
                repository_root=self.repository,
                source_bundle=self.source,
                output_root=self.output_root,
                run_id="phase6-r2-outer-wrong-runs",
                source_git_sha=SOURCE_GIT_SHA,
                expected_original_root=self.original.root_sha256,
                expected_run_ids=tuple(reversed(RUN_IDS)),
                original_uri=self.original_uri,
                copy_prefix=self.copy_prefix,
            )

    def test_clean_retrieval_name_validates_and_tamper_fails(self) -> None:
        final = self._build()
        retrieved = self.repository / "retrieved-empty-destination"
        shutil.copytree(final, retrieved, copy_function=shutil.copy2)
        self.assertEqual(
            self._validate(retrieved).root_sha256,
            self._validate(final).root_sha256,
        )
        _make_writable(retrieved)
        required = retrieved / REQUIRED_REPOSITORY_FILES[0]
        required.write_bytes(required.read_bytes() + b"tamper\n")
        _make_immutable(retrieved)
        with self.assertRaises(RuntimeError):
            self._validate(retrieved)


if __name__ == "__main__":
    unittest.main()

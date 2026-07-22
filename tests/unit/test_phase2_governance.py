"""Read-only governance regressions for certified E00 and quality lock."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import unittest

from kvbench.errors import ArtifactSafetyError
from kvbench.runtime.artifacts import AppendOnlyArtifactStore


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
E00_ROOT = REPOSITORY_ROOT / "docs" / "evidence" / "e00"
FAILED_RUN_ID = "e00-20260722T041628.190813Z-980eff7b6f59-0dd71f2d"
SUCCESSFUL_RUN_ID = "e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32"
FAILED_MANIFEST_SHA256 = (
    "0720734d29c90f609e51cf4c5e4f0b1fadce220e23e146e566f860bb962c0035"
)
SUCCESSFUL_MANIFEST_SHA256 = (
    "d054df714bb5eea1f114bf10a03a2879f56ec8d17d3b07e24fe6efcaba6b7aca"
)
QUALITY_ADDENDUM_SHA256 = (
    "62a8978e04732caff101487275d8b22f14358254538a7b377db2153597a1f332"
)
POST_PERFORMANCE_SHA256 = (
    "b6178566f239ca6ae598b477754f2ebb9d34d0f44c4fd25593b7ea58aa844620"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.stat().st_size,
            _sha256(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _verify_checksum_ledger(run_dir: Path) -> None:
    ledger = run_dir / "checksums.sha256"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise AssertionError(f"empty checksum ledger: {ledger}")
    seen: set[str] = set()
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise AssertionError(f"malformed checksum line in {ledger}: {line!r}")
        expected, relative = parts
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise AssertionError(f"unsafe checksum path in {ledger}: {relative!r}")
        if relative in seen:
            raise AssertionError(f"duplicate checksum path in {ledger}: {relative!r}")
        seen.add(relative)
        target = run_dir.joinpath(*pure.parts)
        if not target.is_file():
            raise AssertionError(f"missing checksummed E00 file: {target}")
        if _sha256(target) != expected:
            raise AssertionError(f"E00 checksum mismatch: {target}")


class E00PreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.before = _tree_snapshot(E00_ROOT)

    @classmethod
    def tearDownClass(cls) -> None:
        if _tree_snapshot(E00_ROOT) != cls.before:
            raise AssertionError("Phase 2 governance tests modified immutable E00 evidence")

    def test_original_failed_e00_manifest_and_ledger_are_unchanged(self) -> None:
        run_dir = E00_ROOT / FAILED_RUN_ID
        self.assertEqual(_sha256(run_dir / "manifest.json"), FAILED_MANIFEST_SHA256)
        _verify_checksum_ledger(run_dir)

    def test_successful_e00_manifest_and_ledger_are_valid(self) -> None:
        run_dir = E00_ROOT / SUCCESSFUL_RUN_ID
        self.assertEqual(
            _sha256(run_dir / "manifest.json"), SUCCESSFUL_MANIFEST_SHA256
        )
        _verify_checksum_ledger(run_dir)

    def test_phase2_writer_rejects_formal_e00_evidence_root(self) -> None:
        before = _tree_snapshot(E00_ROOT)
        with self.assertRaises(ArtifactSafetyError):
            AppendOnlyArtifactStore(E00_ROOT)
        with self.assertRaises(ArtifactSafetyError):
            AppendOnlyArtifactStore(E00_ROOT, formal_evidence_roots=[])
        self.assertEqual(_tree_snapshot(E00_ROOT), before)


class QualityGovernanceTests(unittest.TestCase):
    def test_quality_protocols_retain_preregistered_hashes(self) -> None:
        self.assertEqual(
            _sha256(REPOSITORY_ROOT / "CODEX_QUALITY_EVALUATION_ADDENDUM.md"),
            QUALITY_ADDENDUM_SHA256,
        )
        self.assertEqual(
            _sha256(REPOSITORY_ROOT / "CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md"),
            POST_PERFORMANCE_SHA256,
        )

    def test_quality_execution_is_locked_and_freeze_marker_absent(self) -> None:
        status = (REPOSITORY_ROOT / "docs" / "status.md").read_text(encoding="utf-8")
        decision = (
            REPOSITORY_ROOT
            / "docs"
            / "decisions"
            / "0005-quality-protocol-precedence-and-execution-lock.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Quality execution: LOCKED", status)
        self.assertIn("Quality execution remains `LOCKED`", decision)
        self.assertFalse(
            any(REPOSITORY_ROOT.rglob("PERFORMANCE_DATA_FROZEN")),
            "PERFORMANCE_DATA_FROZEN must not exist during Phase 2",
        )

    def test_no_quality_result_directories_exist(self) -> None:
        for relative in (
            "artifacts/quality",
            "results/quality",
            "docs/evidence/quality",
        ):
            self.assertFalse((REPOSITORY_ROOT / relative).exists(), relative)


if __name__ == "__main__":
    unittest.main()

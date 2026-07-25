"""Append-only lifecycle tests for strict Phase 6 run artifacts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from kvbench.errors import (
    ArtifactConflictError,
    ArtifactSafetyError,
    ArtifactStateError,
)
from kvbench.runtime.artifacts import (
    AppendOnlyArtifactStore,
    phase6_artifact_store,
    validate_run_directory,
)
from kvbench.schema import Phase6RunManifest
from tests.schema.test_phase6_schema import _run_payload


def _completed(payload: dict[str, object]) -> dict[str, object]:
    result = deepcopy(payload)
    result.update(
        {
            "status": "completed",
            "started_at_utc": "2026-07-25T01:02:04Z",
            "finished_at_utc": "2026-07-25T01:02:05Z",
            "inventory_path": "artifact_inventory.json",
        }
    )
    return result


def _restore_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass
    try:
        root.chmod(0o700)
    except OSError:
        pass


class Phase6ArtifactTests(unittest.TestCase):
    def test_complete_last_integrity_and_duplicate_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = phase6_artifact_store(root)
            payload = _run_payload()
            run = store.create(
                str(payload["run_id"]),
                Phase6RunManifest.from_dict(payload),
            )
            run.start()
            run.write_json("config/method.json", {"method": "turboquant"})
            run.write_json(
                "environment/container_identity.json",
                {"verified": True},
            )
            run.write_json("raw/runner.json", {"runner": "fixed_l"})
            run.write_json("validation/point.json", {"passed": True})
            final = run.finalize(
                Phase6RunManifest.from_dict(_completed(payload))
            )
            self.addCleanup(_restore_writable, root)
            validation = validate_run_directory(final)
            self.assertTrue(validation.valid, validation.errors)
            self.assertTrue(validation.complete)
            self.assertTrue((final / "COMPLETE").is_file())
            self.assertEqual(
                (final / "manifest.json").stat().st_mode & 0o222,
                0,
            )
            with self.assertRaises(ArtifactConflictError):
                store.create(
                    str(payload["run_id"]),
                    Phase6RunManifest.from_dict(payload),
                )

    def test_missing_required_payload_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _run_payload()
            run = phase6_artifact_store(root).create(
                str(payload["run_id"]),
                Phase6RunManifest.from_dict(payload),
            )
            run.start()
            run.write_json("config/method.json", {"method": "turboquant"})
            with self.assertRaises(ArtifactStateError):
                run.finalize(
                    Phase6RunManifest.from_dict(_completed(payload))
                )

    def test_checksum_tamper_and_formal_evidence_overlap_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = phase6_artifact_store(root)
            payload = _run_payload()
            run = store.create(
                str(payload["run_id"]),
                Phase6RunManifest.from_dict(payload),
            )
            run.start()
            for relative in (
                "config/method.json",
                "environment/container_identity.json",
                "raw/runner.json",
                "validation/point.json",
            ):
                run.write_json(relative, {"relative": relative})
            final = run.finalize(
                Phase6RunManifest.from_dict(_completed(payload))
            )
            self.addCleanup(_restore_writable, root)
            target = final / "raw/runner.json"
            target.chmod(0o644)
            target.write_text("{}\n", encoding="utf-8")
            validation = validate_run_directory(final)
            self.assertFalse(validation.valid)
            self.assertTrue(
                any("checksum mismatch" in item for item in validation.errors)
            )
            with self.assertRaises(ArtifactSafetyError):
                AppendOnlyArtifactStore(
                    root / "docs/evidence/phase6",
                    formal_evidence_roots=(root / "docs/evidence",),
                )


if __name__ == "__main__":
    unittest.main()

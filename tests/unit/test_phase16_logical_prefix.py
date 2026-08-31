from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import phase13_pilot as phase13
from scripts import phase16_full_scan as phase16
from scripts import phase16_logical_prefix as logical


class Phase16LogicalPrefixTests(unittest.TestCase):
    def test_one_logical_input_is_shared_across_all_methods(self) -> None:
        matching = [
            item
            for item in phase16.logical_points()
            if item["batch_size"] == 2 and item["context_label"] == 4096
        ]
        self.assertEqual(len(matching), 10)
        identifiers = {
            logical.logical_prefix_id(
                batch_size=item["batch_size"],
                configured_context_label=item["context_label"],
                historical_context=item["historical_context"],
            )
            for item in matching
        }
        self.assertEqual(len(identifiers), 1)
        self.assertEqual(len(phase16.logical_prefix_specs()), 59)

    def test_compact_artifact_round_trip_matches_frozen_generator(self) -> None:
        temporary = Path(tempfile.mkdtemp(prefix="phase16-logical-prefix-test."))
        root = temporary / logical.logical_prefix_id(
            batch_size=2,
            configured_context_label=4096,
            historical_context=4096,
        )
        try:
            manifest = logical.create_logical_prefix_artifact(
                root,
                batch_size=2,
                configured_context_label=4096,
                historical_context=4096,
            )
            observed, prefix, decode = logical.validate_logical_prefix_artifact(root)
            expected_prefix, expected_decode = phase13._point_inputs(
                batch=2, historical=4096, device="cpu"
            )
            self.assertEqual(observed, manifest)
            self.assertEqual(str(prefix.dtype), "torch.int32")
            self.assertTrue(prefix.to(dtype=expected_prefix.dtype).equal(expected_prefix))
            self.assertTrue(decode.to(dtype=expected_decode.dtype).equal(expected_decode))
            self.assertEqual(
                {path.name for path in root.iterdir()}, logical.ARTIFACT_FILES
            )
            self.assertFalse(manifest["cache_snapshot_stored"])
            self.assertNotIn("state.safetensors", {path.name for path in root.iterdir()})
        finally:
            self._remove_read_only_tree(temporary)

    def test_token_tampering_fails_closed(self) -> None:
        temporary = Path(tempfile.mkdtemp(prefix="phase16-logical-tamper-test."))
        root = temporary / logical.logical_prefix_id(
            batch_size=16,
            configured_context_label=16384,
            historical_context=16384,
        )
        try:
            logical.create_logical_prefix_artifact(
                root,
                batch_size=16,
                configured_context_label=16384,
                historical_context=16384,
            )
            token_file = root / logical.TOKEN_FILE
            token_file.chmod(0o644)
            with token_file.open("ab") as handle:
                handle.write(b"tamper")
                handle.flush()
                os.fsync(handle.fileno())
            token_file.chmod(0o444)
            with self.assertRaises(logical.Phase16LogicalPrefixError):
                logical.validate_logical_prefix_artifact(root)
        finally:
            self._remove_read_only_tree(temporary)

    def test_reconstruction_precedes_measurement_and_writes_no_cache(self) -> None:
        source = Path("scripts/phase16_full_scan.py").read_text(encoding="utf-8")
        logical_source = Path("scripts/phase16_logical_prefix.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("logical_prefix_untimed_reconstruction", source)
        self.assertIn("cache_build_or_restore_outside_timing", source)
        self.assertNotIn("save_prefix_state(", source)
        self.assertNotIn("state.safetensors", logical_source)
        self.assertIn('"cache_snapshot_stored": False', logical_source)

    @staticmethod
    def _remove_read_only_tree(root: Path) -> None:
        for path in sorted(root.rglob("*"), reverse=True):
            try:
                path.chmod(0o755 if path.is_dir() else 0o644)
            except FileNotFoundError:
                pass
        shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()

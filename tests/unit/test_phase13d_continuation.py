"""Focused tests for append-only Phase 13D continuation semantics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from kvbench.runtime.artifacts import sha256_file
from scripts import phase13d_continuation as continuation


def _snapshot_payload(*, foreign: bool = False) -> dict[str, object]:
    return {
        "captured_at_utc": "2026-08-25T00:00:00.000000Z",
        "query_exit_code": 0,
        "graphics_processes": [],
        "allowed_compute_processes": [],
        "foreign_compute_processes": ([{"pid": 1234}] if foreign else []),
        "unknown_processes": [],
        "subcommands": [],
        "errors": [],
    }


def _invocation(payload: object, *, return_code: int = 0) -> dict[str, object]:
    return {
        "return_code": return_code,
        "raw_stdout": json.dumps(payload, sort_keys=True),
        "raw_stderr": "",
        "invocation_exception": None,
    }


class Phase13DSnapshotContinuationTests(unittest.TestCase):
    def test_rejected_snapshot_payload_is_persisted_before_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            evidence = run_root / "snapshots" / "postflight"
            invalid = {
                "return_code": 7,
                "raw_stdout": "not-json",
                "raw_stderr": "query unavailable",
                "invocation_exception": None,
            }
            writes: list[Path] = []
            original_write = continuation._write_durable_exclusive

            def observed_write(path: Path, payload: object) -> None:
                writes.append(path)
                original_write(path, payload)

            with (
                mock.patch.object(
                    continuation, "_invoke_snapshot_command", return_value=invalid
                ),
                mock.patch.object(continuation.time, "sleep"),
                mock.patch.object(
                    continuation,
                    "_write_durable_exclusive",
                    side_effect=observed_write,
                ),
            ):
                outcome = continuation.capture_persisted_snapshot(
                    evidence_root=evidence,
                    phase="postflight",
                )

            self.assertEqual(outcome.state, "query_failed")
            self.assertEqual(outcome.attempt, 2)
            self.assertEqual(len(writes), 6)
            for attempt in range(3):
                raw_path = evidence / f"attempt-{attempt:02d}.raw.json"
                classification_path = (
                    evidence / f"attempt-{attempt:02d}.classification.json"
                )
                self.assertLess(writes.index(raw_path), writes.index(classification_path))
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                classified = json.loads(
                    classification_path.read_text(encoding="utf-8")
                )
                self.assertEqual(raw["return_code"], 7)
                self.assertEqual(raw["raw_stdout"], "not-json")
                self.assertEqual(raw["raw_stderr"], "query unavailable")
                self.assertEqual(classified["state"], "query_failed")
                self.assertIsNone(classified["parser_result"])
                self.assertIn("JSONDecodeError", classified["parser_exception"])
                self.assertIsNone(classified["process_list"])

    def test_query_failure_is_distinct_from_foreign_process(self) -> None:
        failed = continuation._classify_snapshot_attempt(
            invocation=_invocation("invalid", return_code=1)
        )
        foreign = continuation._classify_snapshot_attempt(
            invocation=_invocation(_snapshot_payload(foreign=True))
        )
        clean = continuation._classify_snapshot_attempt(
            invocation=_invocation(_snapshot_payload())
        )
        self.assertEqual(failed[0], "query_failed")
        self.assertEqual(foreign[0], "foreign_process_detected")
        self.assertEqual(clean[0], "clean")

    def test_postflight_query_failure_excludes_only_current_run(self) -> None:
        self.assertEqual(
            continuation.snapshot_policy(
                phase="postflight", state="query_failed"
            ),
            "exclude_current_and_continue",
        )
        self.assertEqual(
            continuation.snapshot_policy(phase="preflight", state="query_failed"),
            "fail_current_before_measurement",
        )

    def test_true_foreign_process_still_aborts_campaign(self) -> None:
        self.assertEqual(
            continuation.snapshot_policy(
                phase="preflight", state="foreign_process_detected"
            ),
            "abort_campaign",
        )
        self.assertEqual(
            continuation.snapshot_policy(
                phase="postflight", state="foreign_process_detected"
            ),
            "invalidate_current_and_abort",
        )


class Phase13DContinuationOrderTests(unittest.TestCase):
    def test_q4_manifest_does_not_require_nonexistent_order_workspace_field(self) -> None:
        record = next(
            row
            for row in continuation.continuation_records(continuation._order())
            if row["method_config_id"] == "kvq4"
        )
        self.assertNotIn("q4_value_decode_workspace", record)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = continuation._run_manifest(
                root=root,
                segment_id="phase13dseg-20260825t000000000000z-aaaaaaaa-aaaaaa",
                execution_head="a" * 40,
                run_id="phase13dseg-20260825t000000000000z-aaaaaaaa-aaaaaa-g063-r0-o063-kvq4-b8-l5120",
                record=record,
                status="completed",
                reason=None,
            )
        self.assertEqual(manifest["status"], "completed")
        self.assertNotIn("q4_value_decode_workspace", manifest)

    def test_continuation_preserves_exact_frozen_suffix(self) -> None:
        order = continuation._order()
        records = continuation.continuation_records(order)
        self.assertEqual(len(records), 201)
        self.assertEqual(
            [row["original_sequence_index"] for row in records],
            list(range(51, 252)),
        )
        for sequence_index, record in enumerate(records, start=51):
            expected = dict(order["records"][sequence_index])
            for key, value in expected.items():
                self.assertEqual(record[key], value)
        self.assertTrue(records[0]["replacement_for_failed_finalization"])
        self.assertTrue(
            all(
                not row["replacement_for_failed_finalization"]
                for row in records[1:]
            )
        )

    def test_valid_segment_a_runs_are_not_selected(self) -> None:
        records = continuation.continuation_records(continuation._order())
        selected = {row["original_sequence_index"] for row in records}
        self.assertTrue(selected.isdisjoint(range(51)))
        self.assertEqual(min(selected), 51)

    def test_combined_coverage_excludes_only_original_failed_directory(self) -> None:
        records = []
        for sequence_index in range(252):
            segment_a = sequence_index < 51
            records.append(
                {
                    "source_segment_id": (
                        continuation.SEGMENT_A_ID if segment_a else "segment-b"
                    ),
                    "original_sequence_index": sequence_index,
                    "original_logical_run_id": (
                        continuation.FAILED_RUN_ID
                        if sequence_index == 51
                        else f"logical-{sequence_index}"
                    ),
                    "replacement_for_failed_finalization": sequence_index == 51,
                    "manifest_path": (
                        f"runs/logical-{sequence_index}/manifest.json"
                        if segment_a
                        else f"segments/segment-b/runs/run-{sequence_index}/manifest.json"
                    ),
                }
            )
        continuation._require_combined_logical_coverage(records)
        records[51]["manifest_path"] = (
            f"runs/{continuation.FAILED_RUN_ID}/manifest.json"
        )
        with self.assertRaises(continuation.Phase13DContinuationError):
            continuation._require_combined_logical_coverage(records)

    def test_timing_critical_tracked_git_blobs_are_unchanged(self) -> None:
        for relative in continuation.TIMING_CRITICAL_TRACKED_PATHS:
            self.assertEqual(
                continuation._git_blob(
                    continuation.ORIGINAL_EXECUTION_HEAD, relative
                ),
                continuation._git_blob("HEAD", relative),
                relative,
            )


class Phase13DPrefixReuseTests(unittest.TestCase):
    def test_prefix_state_is_checksum_verified_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshots" / "bf16-b1-l17"
            snapshot.mkdir(parents=True)
            state = snapshot / "state.safetensors"
            state.write_bytes(b"immutable-prefix-state")
            digest = sha256_file(state)
            state.chmod(0o444)
            manifest = {
                "state_file_sha256": digest,
                "configuration": "bf16",
                "snapshot_key": {"batch_size": 1},
                "historical_context": 17,
                "batch_reuse_policy": "exact_target_batch_only",
                "runtime_prefix_sharing": False,
            }
            (snapshot / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (snapshot / "COMPLETE").write_text(digest + "\n", encoding="ascii")
            entry = {
                "snapshot_id": "bf16-b1-l17",
                "method_config_id": "bf16",
                "batch_size": 1,
                "historical_context": 17,
                "method_config_fingerprint": "f" * 64,
                "state_file_bytes": state.stat().st_size,
                "state_file_sha256": digest,
                "snapshot_relative_path": "snapshots/bf16-b1-l17",
            }
            catalog = {"entries": [entry]}
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            with (
                mock.patch.object(
                    continuation, "PREFIX_CATALOG_SHA256", sha256_file(catalog_path)
                ),
                mock.patch.object(
                    continuation,
                    "PREFIX_STATE_INDEX_SHA256",
                    continuation._prefix_index_digest([entry]),
                ),
                mock.patch.object(continuation, "PREFIX_STATE_COUNT", 1),
            ):
                result = continuation.validate_prefix_reuse(
                    root, verify_state_bytes=True
                )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["read_only"])
            self.assertFalse(result["regenerated"])
            self.assertEqual(state.stat().st_mode & 0o222, 0)

            os.chmod(state, 0o644)
            with (
                mock.patch.object(
                    continuation, "PREFIX_CATALOG_SHA256", sha256_file(catalog_path)
                ),
                mock.patch.object(
                    continuation,
                    "PREFIX_STATE_INDEX_SHA256",
                    continuation._prefix_index_digest([entry]),
                ),
                mock.patch.object(continuation, "PREFIX_STATE_COUNT", 1),
                self.assertRaises(continuation.Phase13DContinuationError),
            ):
                continuation.validate_prefix_reuse(root, verify_state_bytes=True)


if __name__ == "__main__":
    unittest.main()

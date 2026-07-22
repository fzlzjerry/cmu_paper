"""CPU-only tests for evidence-derived Phase 3 G1 reporting."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.validate_phase2 import validate_phase3_campaign_and_report_roots
from kvbench.cli import build_parser, command_phase3_report
from kvbench.runtime.phase3_report import (
    Phase3ReportError,
    ValidatedPhase3Run,
    _audit_zero_allocation,
    _strict_json_object,
    _stability_summary,
    expected_campaign_run_ids,
    validate_phase3_g1_report_directory,
    write_phase3_g1_report,
)
from kvbench.schema import (
    FROZEN_PHASE3_POINT_IDS,
    G1_CRITERIA,
    GraphMode,
    Phase3G1AdmissionReport,
    RunnerKind,
)
from kvbench.schema.phase3 import PHASE3_PLAN_FINGERPRINTS


ZERO_SHA256 = "0" * 64
ZERO_GIT_SHA = "0" * 40


def blocked_report() -> Phase3G1AdmissionReport:
    return Phase3G1AdmissionReport.from_dict(
        {
            "schema_version": "kvbench-phase3-g1-admission-report-1.0.0",
            "generated_at_utc": "2026-07-22T00:00:00+00:00",
            "git_sha": ZERO_GIT_SHA,
            "status": "BLOCKED",
            "g0": "PASS",
            "g1": "BLOCKED",
            "g2": "NOT_EVALUATED",
            "g3": "NOT_EVALUATED",
            "g4": "NOT_EVALUATED",
            "g5": "NOT_EVALUATED",
            "full_scan_state": "closed",
            "quality": {
                "schema_version": "kvbench.quality-status.v1",
                "quality_status": "unvalidated",
                "claim_eligibility": "performance_only",
                "quality_execution": "locked",
                "performance_data_frozen": False,
            },
            "quality_benchmark_executed": False,
            "quality_only_dependencies_installed": False,
            "measurement_scope": "native_host_admission",
            "performance_claim_eligible": False,
            "performance_data_frozen": False,
            "blocker_b009": "OPEN",
            "blocker_b010": "OPEN",
            "expected_process_count": 20,
            "plan_sources": [
                {"path": path, "sha256": digest}
                for path, digest in PHASE3_PLAN_FINGERPRINTS.items()
            ],
            "run_evidence": [],
            "stability_summaries": [],
            "criteria": [
                {
                    "criterion": criterion,
                    "disposition": "BLOCKED",
                    "evidence_run_ids": [],
                    "reason": "unit-test fixture has no campaign evidence",
                }
                for criterion in G1_CRITERIA
            ],
            "all_artifacts_checksum_valid": False,
            "formal_paper_claim_generated": False,
        }
    )


def timing_run(
    point_id: str,
    *,
    process_index: int,
    host_ns_per_operation: int,
) -> ValidatedPhase3Run:
    samples = [
        {
            "batch_index": index,
            "host_total_ns": host_ns_per_operation * 32,
            "cuda_total_ms": 1.0,
            "completed_operations": 32,
            "failed_operations": 0,
            "host_ns_per_operation": float(host_ns_per_operation),
            "cuda_ms_per_operation": 1.0 / 32.0,
        }
        for index in range(5)
    ]
    telemetry = {
        "before": {
            "temperature_celsius": 40.0 + process_index,
            "sm_clock_mhz": 1800.0 + process_index,
            "power_watts": 120.0 + process_index,
        },
        "after": {
            "temperature_celsius": 45.0 + process_index,
            "sm_clock_mhz": 1900.0 + process_index,
            "power_watts": 180.0 + process_index,
        },
    }
    manifest = SimpleNamespace(
        measured_batches=5,
        measured_count=32,
        runner_kind=RunnerKind.FIXED_L,
        output_steps=1,
        run_id=f"phase3-report-test-{process_index}-{point_id}",
        point_id=point_id,
    )
    return ValidatedPhase3Run(
        run_dir=Path("/tmp") / point_id,
        manifest=manifest,
        completion=SimpleNamespace(),
        worker_result=SimpleNamespace(
            expected_operations=160,
            completed_operations=160,
            failed_operations=0,
        ),
        process_audit={},
        ready_process={
            "pid": 1000 + process_index,
            "process_start_time_ticks": 5000 + process_index,
        },
        worker_evidence={},
        runtime={},
        numerical={},
        timing={
            "samples": samples,
            "sample_count": 5,
            "paper_claim_eligible": False,
            "measurement_scope": "native_host_admission",
            "quality_status": "unvalidated",
            "claim_eligibility": "performance_only",
            "performance_claim_eligible": False,
            "profiler_instrumented": False,
        },
        telemetry=telemetry,
    )


class Phase3ReportSelectionTests(unittest.TestCase):
    def test_campaign_ids_reconstruct_exact_frozen_order(self) -> None:
        fixed = "phase3-20260722t010203000001z-12345678-abcdef"
        growing = "phase3-20260722t010203000002z-12345678-fedcba"
        run_ids = expected_campaign_run_ids(fixed, growing)
        self.assertEqual(len(run_ids), 20)
        self.assertEqual(
            tuple(run_id.removeprefix(f"{fixed}-") if index < 16 else run_id.removeprefix(f"{growing}-") for index, run_id in enumerate(run_ids)),
            FROZEN_PHASE3_POINT_IDS,
        )

    def test_campaign_selection_rejects_same_or_floating_ids(self) -> None:
        campaign = "phase3-20260722t010203000001z-12345678-abcdef"
        with self.assertRaises(Phase3ReportError):
            expected_campaign_run_ids(campaign, campaign)
        with self.assertRaises(Phase3ReportError):
            expected_campaign_run_ids("latest", campaign)

    def test_cli_requires_both_explicit_campaign_ids(self) -> None:
        parser = build_parser()
        fixed = "phase3-20260722t010203000001z-12345678-abcdef"
        growing = "phase3-20260722t010203000002z-12345678-fedcba"
        args = parser.parse_args(
            [
                "phase3-report",
                "--fixed-campaign-id",
                fixed,
                "--growing-campaign-id",
                growing,
            ]
        )
        self.assertIs(args.handler, command_phase3_report)
        self.assertEqual(args.fixed_campaign_id, fixed)
        self.assertEqual(args.growing_campaign_id, growing)


class StabilityDerivationTests(unittest.TestCase):
    def test_host_wall_summary_is_derived_from_raw_process_samples(self) -> None:
        point_ids = (
            "fixed_l-b1-l4096-cuda_graph-r1",
            "fixed_l-b1-l4096-cuda_graph-r2",
            "fixed_l-b1-l4096-cuda_graph-r3",
        )
        process_values = (1_000_000, 1_010_000, 990_000)
        runs = {
            point_id: timing_run(
                point_id,
                process_index=index,
                host_ns_per_operation=process_values[index],
            )
            for index, point_id in enumerate(point_ids)
        }
        derived = _stability_summary(GraphMode.CUDA_GRAPH, runs)
        self.assertIsNotNone(derived)
        assert derived is not None
        summary, payload = derived
        expected = (1.0, 1.01, 0.99)
        self.assertEqual(summary.process_median_host_wall_ms, expected)
        self.assertAlmostEqual(summary.median_host_wall_ms, 1.0)
        self.assertAlmostEqual(summary.minimum_host_wall_ms, 0.99)
        self.assertAlmostEqual(summary.maximum_host_wall_ms, 1.01)
        self.assertAlmostEqual(
            summary.coefficient_of_variation_percent,
            statistics.stdev(expected) / statistics.mean(expected) * 100.0,
        )
        self.assertEqual(payload["primary_endpoint"], "host_observed_wall_time_per_decode_step")
        self.assertFalse(payload["profiler_instrumented"])

    def test_missing_timing_cannot_create_a_stability_summary(self) -> None:
        point_ids = (
            "fixed_l-b1-l4096-eager-r1",
            "fixed_l-b1-l4096-eager-r2",
            "fixed_l-b1-l4096-eager-r3",
        )
        runs = {
            point_id: timing_run(
                point_id,
                process_index=index,
                host_ns_per_operation=1_000_000,
            )
            for index, point_id in enumerate(point_ids)
        }
        missing = copy.copy(runs[point_ids[1]])
        object.__setattr__(missing, "timing", None)
        runs[point_ids[1]] = missing
        self.assertIsNone(_stability_summary(GraphMode.EAGER, runs))

    def test_allocation_audit_requires_zero_events_and_nonpositive_deltas(self) -> None:
        passing = {
            "audit_available": True,
            "passed": True,
            "allocation_event_count": 0,
            "allocation_event_bytes": 0,
            "allocated_delta": 0,
            "reserved_delta": 0,
            "failure_reason": None,
        }
        self.assertTrue(_audit_zero_allocation(passing))
        failing = dict(passing, allocation_event_count=1, allocation_event_bytes=8)
        self.assertFalse(_audit_zero_allocation(failing))


class ImmutableReportBundleTests(unittest.TestCase):
    def test_campaign_and_report_roots_validate_without_legacy_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            artifacts = Path(raw_root) / "artifacts"
            (artifacts / "phase3_campaigns" / "invalid-campaign").mkdir(
                parents=True
            )
            (artifacts / "phase3_reports" / "invalid-report").mkdir(parents=True)
            self.assertFalse((artifacts / "phase3").exists())
            errors = validate_phase3_campaign_and_report_roots(artifacts)

        self.assertIn("invalid Phase 3 campaign: invalid-campaign", errors)
        self.assertIn("invalid Phase 3 report: invalid-report", errors)

    def test_run_and_report_json_readers_match_their_exact_writers(self) -> None:
        payload = {"b": [2, 3], "a": 1}
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "evidence.json"
            path.write_bytes(
                (
                    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                    + "\n"
                ).encode("utf-8")
            )
            self.assertEqual(_strict_json_object(path), payload)
            with self.assertRaises(Phase3ReportError):
                _strict_json_object(path, canonical=True)

            path.write_bytes(b'{"a":1,"b":[2,3]}\n')
            self.assertEqual(
                _strict_json_object(path, canonical=True),
                payload,
            )
            with self.assertRaises(Phase3ReportError):
                _strict_json_object(path)

    def test_report_bundle_is_checksummed_and_tampering_fails(self) -> None:
        fixed = "phase3-20260722t010203000001z-12345678-abcdef"
        growing = "phase3-20260722t010203000002z-12345678-fedcba"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            derivation = {
                "schema_version": "kvbench-phase3-g1-derivation-1.0.0",
                "report_git_provenance": {
                    "schema_version": "kvbench-phase3-report-git-provenance-1.0.0",
                    "source_execution_git_sha": ZERO_GIT_SHA,
                    "report_generator_git_sha": ZERO_GIT_SHA,
                    "execution_to_generator_changed_paths": [],
                    "source_execution_is_ancestor": True,
                    "reporting_only_descendant": True,
                },
                "selected_run_ids": [],
            }
            with mock.patch(
                "kvbench.runtime.phase3_report.build_phase3_g1_report",
                return_value=(blocked_report(), {}, derivation),
            ):
                result = write_phase3_g1_report(
                    fixed,
                    growing,
                    repository_root=root,
                )
                report_dir = root / result["report_dir"]
                self.assertTrue(
                    validate_phase3_g1_report_directory(report_dir)["valid"]
                )
                report_path = report_dir / "report.json"
                report_path.chmod(0o644)
                report_path.write_bytes(report_path.read_bytes() + b" ")
                report_path.chmod(0o444)
                self.assertFalse(
                    validate_phase3_g1_report_directory(report_dir)["valid"]
                )
            for path in sorted(report_dir.rglob("*"), reverse=True):
                path.chmod(0o755 if path.is_dir() else 0o644)
            report_dir.chmod(0o755)


if __name__ == "__main__":
    unittest.main()

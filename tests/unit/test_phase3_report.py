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
from kvbench.runtime.process_supervision import command_fingerprint
from kvbench.runtime.phase3_report import (
    Phase3ReportError,
    ValidatedPhase3Run,
    _manifest_environment_join,
    _validate_process_evidence_v2,
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
    canonical_json_bytes,
    sha256_hex,
    Phase3G1AdmissionReport,
    RunStatus,
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
            ), mock.patch(
                "kvbench.runtime.phase3_report_publication.capture_phase3_source_index",
                return_value={
                    "schema_version": "kvbench-phase3-report-source-index-2.0.0",
                    "fixed_campaign_id": fixed,
                    "growing_campaign_id": growing,
                    "run_count": 20,
                    "runs": [
                        {"run_id": f"fixture-run-{index}"}
                        for index in range(20)
                    ],
                    "campaigns": [],
                    "all_checksums_valid": True,
                    "all_sources_immutable": True,
                },
            ), mock.patch(
                "kvbench.runtime.phase3_report_publication._select_report_id",
                return_value=(
                    "phase3-g1-20260723t010116000000z-"
                    "457123b1-a00016"
                ),
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


class Phase3ProcessEvidenceV2Tests(unittest.TestCase):
    RUN_ID = "phase3-unit-process-v2"
    GPU_UUID = "GPU-unit-test"
    PID = 4242
    START_TICKS = 987654
    PARENT_PID = 3131
    COMMAND_ARGV = ("/venv/bin/python", "-m", "kvbench.runtime.phase3_worker")
    WORKING_DIRECTORY = "/home/rockrock/cmu_paper"
    ENVIRONMENT_SHA = "e" * 64
    COMMAND_SHA = command_fingerprint(
        COMMAND_ARGV,
        working_directory=WORKING_DIRECTORY,
        environment_sha256=ENVIRONMENT_SHA,
    )

    @staticmethod
    def _subcommands() -> list[dict[str, object]]:
        return [
            {
                "name": "gpu_index_uuid",
                "argv": [
                    "/usr/bin/nvidia-smi",
                    "--query-gpu=index,uuid",
                    "--format=csv,noheader,nounits",
                ],
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
            },
            {
                "name": "compute_apps",
                "argv": [
                    "/usr/bin/nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
            },
            {
                "name": "pmon",
                "argv": ["/usr/bin/nvidia-smi", "pmon", "-c", "1"],
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
            },
        ]

    @classmethod
    def _raw_process(
        cls,
        *,
        pid: int | None = None,
        start_ticks: int | None = None,
    ) -> dict[str, object]:
        return {
            "gpu_uuid": cls.GPU_UUID,
            "pid": cls.PID if pid is None else pid,
            "process_start_time_ticks": (
                cls.START_TICKS if start_ticks is None else start_ticks
            ),
            "process_type": "C",
            "process_name": "/venv/bin/python",
            "used_gpu_memory_mib": 512,
        }

    @classmethod
    def _snapshot(
        cls,
        *processes: dict[str, object],
    ) -> dict[str, object]:
        return {
            "captured_at_utc": "2026-07-22T00:00:10Z",
            "query_exit_code": 0,
            "graphics_processes": [],
            "allowed_compute_processes": [],
            "foreign_compute_processes": list(processes),
            "unknown_processes": [],
            "subcommands": cls._subcommands(),
            "errors": [],
        }

    @staticmethod
    def _registry_verdict(
        *,
        owned: tuple[dict[str, object], ...] = (),
    ) -> dict[str, object]:
        return {
            "passed": True,
            "terminal_registered_process_resolution": False,
            "query_evidence_hard_failure": False,
            "registry_verdict": {
                "disposition": "owned_only" if owned else "clean",
                "hard_failure": False,
                "owned": list(owned),
                "foreign": [],
                "pid_reuse": [],
                "unverified": [],
            },
            "raw_query_exit_code": 0,
            "raw_errors": [],
        }

    @classmethod
    def _bundle(cls, *, fast_exit: bool = False) -> dict[str, object]:
        worker_evidence = {
            "run_id": cls.RUN_ID,
            "point_id": "fixed_l-b1-l128-eager-r1",
            "worker_result": {"status": "completed"},
        }
        evidence_sha = sha256_hex(
            canonical_json_bytes(worker_evidence) + b"\n"
        )
        stages = (
            "worker_started",
            "cuda_context_created",
            "measurement_started",
            "measurement_finished",
            "evidence_flushed",
            "worker_exiting",
            "supervisor_reaped",
        )
        events = [
            {
                "schema_version": (
                    "kvbench-phase3-worker-handshake-event-1.0.0"
                ),
                "sequence": sequence,
                "stage": stage,
                "recorded_at_utc": (
                    f"2026-07-22T00:00:{sequence:02d}Z"
                ),
                "run_id": cls.RUN_ID,
                "gpu_uuid": cls.GPU_UUID,
                "pid": cls.PID,
                "process_start_time_ticks": cls.START_TICKS,
                "parent_pid": cls.PARENT_PID,
                "command_fingerprint": cls.COMMAND_SHA,
                "evidence_sha256": (
                    evidence_sha if stage == "evidence_flushed" else None
                ),
            }
            for sequence, stage in enumerate(stages, start=1)
        ]
        outcome = {
            "disposition": "owned_completed",
            "reason": "registered worker completed the ordered handshake",
            "returncode": 0,
            "observed_stages": list(stages),
            "missing_worker_stages": [],
            "evidence_flushed": True,
            "worker_exiting_observed": True,
            "full_handshake_observed": True,
            "exclusivity_passed": True,
        }
        owned = {
            "gpu_uuid": cls.GPU_UUID,
            "pid": cls.PID,
            "process_start_time_ticks": cls.START_TICKS,
        }
        samples = [] if fast_exit else [cls._snapshot(cls._raw_process())]
        sample_verdicts = (
            []
            if fast_exit
            else [cls._registry_verdict(owned=(owned,))]
        )
        return {
            "audit": {
                "schema_version": "kvbench-phase3-process-audit-2.0.0",
                "passed": True,
                "certified_helper": "preflight/process_query.py",
                "registry_created": True,
                "ownership_verdict": "owned_completed",
                "exclusivity_passed": True,
                "evidence_flushed": True,
                "worker_exiting_observed": True,
                "pid_start_time_protected": True,
                "pidfd_supported": True,
                "pidfd_opened": True,
                "pidfd_closed": True,
                "failure_reason": None,
                "foreign_compute_allowed": False,
                "unknown_compute_allowed": False,
            },
            "ready": {
                "schema_version": "kvbench-phase3-worker-ready-1.0.0",
                "pid": cls.PID,
                "process_start_time_ticks": cls.START_TICKS,
                "cuda_imported": False,
            },
            "registry": {
                "schema_version": (
                    "kvbench-phase3-process-registry-2.0.0"
                ),
                "identity": {
                    "pid": cls.PID,
                    "start_time_ticks": cls.START_TICKS,
                    "parent_pid": cls.PARENT_PID,
                    "run_id": cls.RUN_ID,
                    "gpu_uuid": cls.GPU_UUID,
                    "spawned_at_utc": "2026-07-22T00:00:00Z",
                    "expected_command_fingerprint": cls.COMMAND_SHA,
                },
                "handle": {
                    "process_handle_kind": "subprocess.Popen",
                    "process_handle_retained": True,
                    "pidfd_supported": True,
                    "pidfd_opened": True,
                    "pidfd": 9,
                },
                "handshake_events": events,
                "exit_observed_without_reaping": True,
                "supervisor_reaped": True,
                "proc_disappeared_after_registration": False,
                "device_snapshot_count": 2 + len(samples),
                "registered_compute_observed": not fast_exit,
                "outcome": outcome,
                "pidfd_closed_by_supervisor": True,
                "process_handle_reaped_by_supervisor": True,
            },
            "handshake": {
                "schema_version": (
                    "kvbench-phase3-worker-handshake-2.0.0"
                ),
                "run_id": cls.RUN_ID,
                "events": events,
                "terminal_outcome": outcome,
                "evidence_flushed_required_for_owned_completion": True,
            },
            "worker_evidence": worker_evidence,
            "before": cls._snapshot(),
            "release": cls._snapshot(),
            "release_verdict": cls._registry_verdict(),
            "during": {
                "schema_version": (
                    "kvbench-phase3-process-monitor-2.0.0"
                ),
                "sampling_target_seconds": 2.0,
                "samples": samples,
                "sample_registry_verdicts": sample_verdicts,
                "saw_registered_compute": not fast_exit,
                "fast_exit_before_first_telemetry_poll": fast_exit,
                "monitoring_stopped_before_worker_exit": False,
            },
            "after": cls._snapshot(),
            "after_verdict": cls._registry_verdict(),
        }

    @classmethod
    def _failure_bundle(
        cls,
        *,
        worker_stages: tuple[str, ...],
        returncode: int,
        disposition: str,
        outcome_reason: str,
        failure_reason: str,
        readiness_observed: bool = True,
    ) -> dict[str, object]:
        bundle = cls._bundle(fast_exit=True)
        worker_stage_order = (
            "worker_started",
            "cuda_context_created",
            "measurement_started",
            "measurement_finished",
            "evidence_flushed",
            "worker_exiting",
        )
        events_by_stage = {
            event["stage"]: event
            for event in bundle["registry"]["handshake_events"]
        }
        events = [
            copy.deepcopy(events_by_stage[stage])
            for stage in (*worker_stages, "supervisor_reaped")
        ]
        evidence_flushed = "evidence_flushed" in worker_stages
        worker_exiting = "worker_exiting" in worker_stages
        outcome = {
            "disposition": disposition,
            "reason": outcome_reason,
            "returncode": returncode,
            "observed_stages": [*worker_stages, "supervisor_reaped"],
            "missing_worker_stages": [
                stage for stage in worker_stage_order if stage not in worker_stages
            ],
            "evidence_flushed": evidence_flushed,
            "worker_exiting_observed": worker_exiting,
            "full_handshake_observed": worker_stages == worker_stage_order,
            "exclusivity_passed": disposition == "owned_worker_failure",
        }
        bundle["registry"]["handshake_events"] = events
        bundle["registry"]["outcome"] = outcome
        bundle["handshake"]["events"] = events
        bundle["handshake"]["terminal_outcome"] = outcome
        bundle["audit"].update(
            {
                "passed": False,
                "ownership_verdict": disposition,
                "exclusivity_passed": disposition == "owned_worker_failure",
                "evidence_flushed": evidence_flushed,
                "worker_exiting_observed": worker_exiting,
                "failure_reason": failure_reason,
            }
        )
        if not readiness_observed:
            bundle["ready"] = {
                "schema_version": "kvbench-phase3-worker-ready-2.0.0",
                "readiness_observed": False,
                "pid": None,
                "process_start_time_ticks": None,
                "cuda_imported": None,
            }
        bundle.pop("worker_evidence")
        return bundle

    @classmethod
    def _registry_not_created_bundle(cls) -> dict[str, object]:
        failure_reason = "Phase3CoordinatorError: worker process registration failed"
        bundle = cls._bundle(fast_exit=True)
        bundle["audit"].update(
            {
                "passed": False,
                "registry_created": False,
                "ownership_verdict": None,
                "exclusivity_passed": False,
                "evidence_flushed": False,
                "worker_exiting_observed": False,
                "pid_start_time_protected": False,
                "failure_reason": failure_reason,
            }
        )
        bundle["ready"] = {
            "schema_version": "kvbench-phase3-worker-ready-2.0.0",
            "readiness_observed": False,
            "pid": None,
            "process_start_time_ticks": None,
            "cuda_imported": None,
        }
        bundle["registry"] = {
            "schema_version": "kvbench-phase3-process-registry-2.0.0",
            "registry_created": False,
            "run_id": cls.RUN_ID,
            "pidfd_supported": True,
            "pidfd_closed_by_supervisor": True,
        }
        bundle["handshake"] = {
            "schema_version": "kvbench-phase3-worker-handshake-2.0.0",
            "run_id": cls.RUN_ID,
            "events": [],
            "terminal_outcome": None,
            "evidence_flushed_required_for_owned_completion": True,
        }
        for key in (
            "worker_evidence",
            "release",
            "release_verdict",
            "during",
            "after_verdict",
        ):
            bundle.pop(key)
        return bundle

    @classmethod
    def _owned_worker_failure_bundle(
        cls,
        kind: str,
    ) -> dict[str, object]:
        if kind == "early":
            bundle = cls._failure_bundle(
                worker_stages=("worker_started",),
                returncode=7,
                disposition="owned_worker_failure",
                outcome_reason="registered worker exited before evidence_flushed",
                failure_reason=(
                    "Phase3CoordinatorError: registered worker exited before "
                    "worker_started readiness completed"
                ),
                readiness_observed=False,
            )
            for key in ("release", "release_verdict", "during", "after_verdict"):
                bundle.pop(key)
            bundle["registry"]["device_snapshot_count"] = 0
            bundle["registry"]["registered_compute_observed"] = False
            return bundle
        if kind == "incomplete":
            worker_stages = (
                "worker_started",
                "cuda_context_created",
                "measurement_started",
                "measurement_finished",
            )
            returncode = 0
            outcome_reason = "registered worker exited before evidence_flushed"
        elif kind == "abnormal":
            worker_stages = (
                "worker_started",
                "cuda_context_created",
                "measurement_started",
                "measurement_finished",
                "evidence_flushed",
                "worker_exiting",
            )
            returncode = 7
            outcome_reason = "registered worker exited with return code 7"
        else:
            raise AssertionError(f"unsupported worker failure kind: {kind}")
        bundle = cls._failure_bundle(
            worker_stages=worker_stages,
            returncode=returncode,
            disposition="owned_worker_failure",
            outcome_reason=outcome_reason,
            failure_reason=(
                "Phase3CoordinatorError: worker ownership failed: "
                "owned_worker_failure"
            ),
        )
        bundle.pop("after_verdict")
        bundle["registry"]["device_snapshot_count"] = 1
        bundle["registry"]["registered_compute_observed"] = False
        return bundle

    @classmethod
    def _hard_failure_verdict(
        cls,
        kind: str,
    ) -> tuple[dict[str, object], dict[str, object], str, str]:
        if kind == "foreign":
            process = cls._raw_process(pid=9999, start_ticks=111)
            observation = {
                "gpu_uuid": cls.GPU_UUID,
                "pid": 9999,
                "process_start_time_ticks": 111,
            }
            snapshot = cls._snapshot(process)
            registry_disposition = "foreign_process_detected"
            ownership_disposition = "foreign_process_detected"
            ownership_reason = "device snapshot contains an unregistered process"
            registry_hard_failure = True
            lists = {
                "owned": [],
                "foreign": [observation],
                "pid_reuse": [],
                "unverified": [],
            }
        elif kind == "pid_reuse":
            process = cls._raw_process(start_ticks=cls.START_TICKS + 1)
            observation = {
                "gpu_uuid": cls.GPU_UUID,
                "pid": cls.PID,
                "process_start_time_ticks": cls.START_TICKS + 1,
            }
            snapshot = cls._snapshot(process)
            registry_disposition = "pid_reuse_detected"
            ownership_disposition = "pid_reuse_detected"
            ownership_reason = (
                "device snapshot observed registered PID with a new start time"
            )
            registry_hard_failure = True
            lists = {
                "owned": [],
                "foreign": [],
                "pid_reuse": [observation],
                "unverified": [],
            }
        elif kind == "unverified":
            process = cls._raw_process(start_ticks=0)
            process["process_type"] = "UNKNOWN"
            snapshot = cls._snapshot()
            snapshot["unknown_processes"] = [process]
            observation = {
                "gpu_uuid": cls.GPU_UUID,
                "pid": cls.PID,
                "process_start_time_ticks": None,
            }
            registry_disposition = "unverified_registered_pid"
            ownership_disposition = "unverified_process_detected"
            ownership_reason = (
                "device snapshot PID lacks a retained identity basis"
            )
            registry_hard_failure = True
            lists = {
                "owned": [],
                "foreign": [],
                "pid_reuse": [],
                "unverified": [observation],
            }
        elif kind == "query_unverified":
            snapshot = cls._snapshot()
            snapshot["query_exit_code"] = 1
            snapshot["errors"] = ["pmon exited with status 1"]
            snapshot["subcommands"][2]["exit_code"] = 1
            registry_disposition = "clean"
            ownership_disposition = "unverified_process_detected"
            ownership_reason = (
                "GPU process query failed outside exact terminal worker resolution"
            )
            registry_hard_failure = False
            lists = {
                "owned": [],
                "foreign": [],
                "pid_reuse": [],
                "unverified": [],
            }
        else:
            raise AssertionError(f"unsupported hard failure kind: {kind}")
        verdict = {
            "passed": False,
            "terminal_registered_process_resolution": False,
            "query_evidence_hard_failure": kind == "query_unverified",
            "registry_verdict": {
                "disposition": registry_disposition,
                "hard_failure": registry_hard_failure,
                **lists,
            },
            "raw_query_exit_code": snapshot["query_exit_code"],
            "raw_errors": snapshot["errors"],
        }
        return snapshot, verdict, ownership_disposition, ownership_reason

    @classmethod
    def _hard_failure_bundle(cls, kind: str) -> dict[str, object]:
        snapshot, verdict, disposition, outcome_reason = (
            cls._hard_failure_verdict(kind)
        )
        location = "release" if kind == "unverified" else "during"
        failure_reason = (
            "Phase3CoordinatorError: worker release audit failed closed"
            if location == "release"
            else (
                "Phase3CoordinatorError: worker process audit detected foreign "
                "or unverified compute"
            )
        )
        bundle = cls._failure_bundle(
            worker_stages=("worker_started", "cuda_context_created"),
            returncode=-15,
            disposition=disposition,
            outcome_reason=outcome_reason,
            failure_reason=failure_reason,
        )
        bundle.pop("after_verdict")
        bundle["registry"]["registered_compute_observed"] = False
        if location == "release":
            bundle["release"] = snapshot
            bundle["release_verdict"] = verdict
            bundle.pop("during")
            bundle["registry"]["device_snapshot_count"] = 1
        else:
            bundle["during"] = {
                "schema_version": "kvbench-phase3-process-monitor-2.0.0",
                "sampling_target_seconds": 2.0,
                "samples": [snapshot],
                "sample_registry_verdicts": [verdict],
            }
            bundle["registry"]["device_snapshot_count"] = 2
        return bundle

    @classmethod
    def _validate_bundle(
        cls,
        root: Path,
        bundle: dict[str, object],
    ) -> None:
        expected_status = (
            RunStatus.COMPLETED
            if bundle["audit"]["passed"]
            else RunStatus.ABORTED
        )
        manifest_status = bundle.get("_manifest_status", expected_status)
        worker_status = bundle.get("_worker_status", expected_status)
        manifest_failure_reason = bundle.get(
            "_manifest_failure_reason",
            bundle["audit"]["failure_reason"],
        )
        worker_failure_reason = bundle.get(
            "_worker_failure_reason",
            bundle["audit"]["failure_reason"],
        )
        paths = {
            "environment/process.registry.json": "registry",
            "environment/process.handshake.json": "handshake",
            "raw/worker_evidence.json": "worker_evidence",
            "environment/process.before.json": "before",
            "environment/process.release_audit.json": "release",
            "environment/process.release_registry_verdict.json": (
                "release_verdict"
            ),
            "environment/process.during.json": "during",
            "environment/process.after.json": "after",
            "environment/process.after_registry_verdict.json": (
                "after_verdict"
            ),
        }
        for relative, key in paths.items():
            if key not in bundle:
                continue
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(bundle[key], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _validate_process_evidence_v2(
            root,
            SimpleNamespace(
                run_id=cls.RUN_ID,
                gpu_uuid=cls.GPU_UUID,
                status=manifest_status,
                failure_reason=manifest_failure_reason,
                command=SimpleNamespace(
                    argv=cls.COMMAND_ARGV,
                    working_directory=cls.WORKING_DIRECTORY,
                    environment_sha256=cls.ENVIRONMENT_SHA,
                ),
            ),
            bundle["audit"],
            bundle["ready"],
            SimpleNamespace(
                status=worker_status,
                failure_reason=worker_failure_reason,
            ),
        )

    def _assert_rejected(self, bundle: dict[str, object]) -> None:
        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-report-v2-",
            dir="/tmp",
        ) as raw_root:
            with self.assertRaises(Phase3ReportError):
                self._validate_bundle(Path(raw_root), bundle)

    def _assert_accepted(self, bundle: dict[str, object]) -> None:
        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-report-v2-",
            dir="/tmp",
        ) as raw_root:
            self._validate_bundle(Path(raw_root), bundle)

    def test_registered_process_and_explicit_fast_exit_pass(self) -> None:
        for fast_exit in (False, True):
            with self.subTest(fast_exit=fast_exit):
                with tempfile.TemporaryDirectory(
                    prefix="kvbench-phase3-report-v2-",
                    dir="/tmp",
                ) as raw_root:
                    self._validate_bundle(
                        Path(raw_root),
                        self._bundle(fast_exit=fast_exit),
                    )

    def test_registry_not_created_failure_uses_explicit_sentinel(self) -> None:
        self._assert_accepted(self._registry_not_created_bundle())
        malformed_sentinel = self._registry_not_created_bundle()
        malformed_sentinel["ready"]["cuda_imported"] = False
        self._assert_rejected(malformed_sentinel)

    def test_registered_worker_exit_failures_remain_reportable(self) -> None:
        for kind in ("early", "incomplete", "abnormal"):
            with self.subTest(kind=kind):
                self._assert_accepted(self._owned_worker_failure_bundle(kind))

    def test_hard_process_failures_remain_reportable(self) -> None:
        for kind in (
            "foreign",
            "unverified",
            "query_unverified",
            "pid_reuse",
        ):
            with self.subTest(kind=kind):
                self._assert_accepted(self._hard_failure_bundle(kind))

    def test_failed_process_audit_requires_exact_abort_join(self) -> None:
        worker_reason_mismatch = self._owned_worker_failure_bundle("incomplete")
        worker_reason_mismatch["_worker_failure_reason"] = "different failure"
        manifest_status_mismatch = self._owned_worker_failure_bundle("incomplete")
        manifest_status_mismatch["_manifest_status"] = RunStatus.COMPLETED
        hard_stage_mismatch = self._hard_failure_bundle("foreign")
        hard_stage_mismatch["audit"]["failure_reason"] = (
            "Phase3CoordinatorError: worker release audit failed closed"
        )
        for label, bundle in (
            ("worker_reason", worker_reason_mismatch),
            ("manifest_status", manifest_status_mismatch),
            ("hard_stage", hard_stage_mismatch),
        ):
            with self.subTest(label=label):
                self._assert_rejected(bundle)

    def test_manifest_command_and_spawn_order_fail_closed(self) -> None:
        command_mismatch = self._bundle()
        command_mismatch["registry"]["identity"][
            "expected_command_fingerprint"
        ] = "c" * 64
        for event in command_mismatch["registry"]["handshake_events"]:
            event["command_fingerprint"] = "c" * 64
        spawn_after_first_event = self._bundle()
        spawn_after_first_event["registry"]["identity"]["spawned_at_utc"] = (
            "2026-07-22T00:00:02Z"
        )
        failed_command_mismatch = self._owned_worker_failure_bundle("early")
        failed_command_mismatch["registry"]["identity"][
            "expected_command_fingerprint"
        ] = "c" * 64
        for event in failed_command_mismatch["registry"]["handshake_events"]:
            event["command_fingerprint"] = "c" * 64
        failed_spawn_after_first_event = self._owned_worker_failure_bundle("early")
        failed_spawn_after_first_event["registry"]["identity"][
            "spawned_at_utc"
        ] = "2026-07-22T00:00:02Z"
        for label, bundle in (
            ("manifest_command", command_mismatch),
            ("spawn_order", spawn_after_first_event),
            ("failed_manifest_command", failed_command_mismatch),
            ("failed_spawn_order", failed_spawn_after_first_event),
        ):
            with self.subTest(label=label):
                self._assert_rejected(bundle)

    def test_manifest_command_uses_pre_injection_environment_digest(self) -> None:
        base_environment = {
            "LANG": "C",
            "PATH": "/usr/bin:/bin",
        }
        environment_sha = sha256_hex(canonical_json_bytes(base_environment))
        command = SimpleNamespace(
            argv=self.COMMAND_ARGV,
            working_directory=self.WORKING_DIRECTORY,
            environment_sha256=environment_sha,
        )
        expected_fingerprint = command_fingerprint(
            command.argv,
            working_directory=command.working_directory,
            environment_sha256=command.environment_sha256,
        )
        recorded_environment = {
            **base_environment,
            "KVBENCH_PHASE3_COMMAND_FINGERPRINT": expected_fingerprint,
        }
        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase3-report-v2-environment-",
            dir="/tmp",
        ) as raw_root:
            root = Path(raw_root)
            path = root / "environment" / "worker_environment.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(recorded_environment, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest = SimpleNamespace(command=command)
            _manifest_environment_join(root, manifest)
            recorded_environment[
                "KVBENCH_PHASE3_COMMAND_FINGERPRINT"
            ] = "f" * 64
            path.write_text(
                json.dumps(recorded_environment, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Phase3ReportError):
                _manifest_environment_join(root, manifest)

    def test_identity_handshake_and_evidence_linkage_fail_closed(self) -> None:
        registry_mismatch = self._bundle()
        registry_mismatch["registry"]["identity"]["start_time_ticks"] += 1
        incomplete = self._bundle()
        incomplete["registry"]["handshake_events"].pop()
        incomplete["handshake"]["events"].pop()
        digest_mismatch = self._bundle()
        digest_mismatch["worker_evidence"]["unexpected"] = True
        for label, bundle in (
            ("registry_start_time", registry_mismatch),
            ("incomplete_handshake", incomplete),
            ("evidence_sha", digest_mismatch),
        ):
            with self.subTest(label=label):
                self._assert_rejected(bundle)

    def test_monitor_foreign_process_and_pid_reuse_fail_closed(self) -> None:
        bad_fast_exit = self._bundle(fast_exit=True)
        bad_fast_exit["during"][
            "fast_exit_before_first_telemetry_poll"
        ] = False
        foreign = self._bundle()
        foreign["during"]["samples"][0]["foreign_compute_processes"].append(
            self._raw_process(pid=9999, start_ticks=111)
        )
        pid_reuse = self._bundle()
        pid_reuse["during"]["samples"][0]["foreign_compute_processes"][0][
            "process_start_time_ticks"
        ] = self.START_TICKS + 1
        for label, bundle in (
            ("fast_exit", bad_fast_exit),
            ("foreign", foreign),
            ("pid_reuse", pid_reuse),
        ):
            with self.subTest(label=label):
                self._assert_rejected(bundle)

if __name__ == "__main__":
    unittest.main()

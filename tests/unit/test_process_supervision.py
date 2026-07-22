"""Deterministic controls for Phase 3 run-owned process supervision."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from kvbench.runtime.process_supervision import (
    DeviceProcessObservation,
    HandshakeEvent,
    HandshakeStage,
    OwnershipDisposition,
    ProcObservationDisposition,
    ProcessIdentity,
    ProcessIdentityUnavailable,
    ProcessSupervisionError,
    RunOwnedProcessRegistry,
    SnapshotDisposition,
    command_fingerprint,
    read_process_identity,
)


ENVIRONMENT_SHA256 = "1" * 64
EVIDENCE_SHA256 = "2" * 64
OTHER_SHA256 = "3" * 64
RECORDED_AT = "2026-07-22T12:00:00.000000Z"


class FakeProcessHandle:
    """Non-reaping process handle used by deterministic unit tests."""


def expected_fingerprint() -> str:
    return command_fingerprint(
        ("/usr/bin/python3", "-m", "kvbench", "phase3-worker"),
        working_directory="/home/rockrock/cmu_paper",
        environment_sha256=ENVIRONMENT_SHA256,
    )


def make_registry(
    *,
    pidfd_supported: bool = True,
    pidfd: int | None = 9,
) -> RunOwnedProcessRegistry:
    return RunOwnedProcessRegistry.register_spawn(
        process_identity=ProcessIdentity(
            pid=432362,
            start_time_ticks=10973359,
            parent_pid=431000,
        ),
        expected_supervisor_pid=431000,
        process_handle=FakeProcessHandle(),
        pidfd_supported=pidfd_supported,
        pidfd=pidfd,
        run_id="phase3-unit-run",
        gpu_uuid="GPU-unit-0001",
        spawned_at_utc=RECORDED_AT,
        expected_command_fingerprint=expected_fingerprint(),
    )


def record_through(
    registry: RunOwnedProcessRegistry,
    terminal_stage: HandshakeStage,
) -> None:
    for stage in tuple(HandshakeStage)[:-1]:
        registry.record_worker_stage(
            stage,
            recorded_at_utc=RECORDED_AT,
            evidence_sha256=(
                EVIDENCE_SHA256
                if stage is HandshakeStage.EVIDENCE_FLUSHED
                else None
            ),
        )
        if stage is terminal_stage:
            return
    raise AssertionError("terminal stage is not worker-owned")


def reap(registry: RunOwnedProcessRegistry, returncode: int) -> None:
    registry.note_exit_observed()
    registry.record_supervisor_reaped(
        returncode,
        recorded_at_utc=RECORDED_AT,
    )


class ProcessIdentityTests(unittest.TestCase):
    def test_proc_stat_parser_handles_parentheses_and_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            process_root = root / "77"
            process_root.mkdir()
            tail = ["S", "55", *(["0"] * 17), "123456"]
            (process_root / "stat").write_text(
                f"77 (worker ) name) {' '.join(tail)}\n",
                encoding="utf-8",
            )
            identity = read_process_identity(77, proc_root=root)

        self.assertEqual(identity.pid, 77)
        self.assertEqual(identity.parent_pid, 55)
        self.assertEqual(identity.start_time_ticks, 123456)

    def test_missing_proc_identity_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            with self.assertRaises(ProcessIdentityUnavailable):
                read_process_identity(77, proc_root=Path(raw_root))

    def test_command_fingerprint_binds_environment_and_argv(self) -> None:
        first = expected_fingerprint()
        second = command_fingerprint(
            ("/usr/bin/python3", "-m", "kvbench", "different-worker"),
            working_directory="/home/rockrock/cmu_paper",
            environment_sha256=ENVIRONMENT_SHA256,
        )
        self.assertNotEqual(first, second)


class HandshakeTests(unittest.TestCase):
    def test_fast_normal_exit_before_telemetry_poll_is_owned_completed(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.WORKER_EXITING)
        reap(registry, 0)

        outcome = registry.terminal_outcome()
        self.assertIs(outcome.disposition, OwnershipDisposition.OWNED_COMPLETED)
        self.assertTrue(outcome.full_handshake_observed)
        self.assertEqual(registry.to_evidence()["device_snapshot_count"], 0)

    def test_exit_before_first_poll_and_required_handshake_is_owned_failure(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.WORKER_STARTED)
        reap(registry, 1)

        outcome = registry.terminal_outcome()
        self.assertIs(
            outcome.disposition,
            OwnershipDisposition.OWNED_WORKER_FAILURE,
        )
        self.assertTrue(outcome.exclusivity_passed)
        self.assertFalse(outcome.evidence_flushed)

    def test_exit_after_evidence_flush_without_exiting_event_is_completed(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        reap(registry, 0)

        outcome = registry.terminal_outcome()
        self.assertIs(outcome.disposition, OwnershipDisposition.OWNED_COMPLETED)
        self.assertTrue(outcome.evidence_flushed)
        self.assertFalse(outcome.worker_exiting_observed)
        self.assertFalse(outcome.full_handshake_observed)

    def test_abnormal_exit_after_evidence_flush_is_owned_failure(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        reap(registry, -9)

        outcome = registry.terminal_outcome()
        self.assertIs(
            outcome.disposition,
            OwnershipDisposition.OWNED_WORKER_FAILURE,
        )
        self.assertIn("return code -9", outcome.reason)

    def test_out_of_order_and_duplicate_stages_fail_closed(self) -> None:
        registry = make_registry()
        with self.assertRaises(ProcessSupervisionError):
            registry.record_worker_stage(
                HandshakeStage.CUDA_CONTEXT_CREATED,
                recorded_at_utc=RECORDED_AT,
            )
        registry.record_worker_stage(
            HandshakeStage.WORKER_STARTED,
            recorded_at_utc=RECORDED_AT,
        )
        with self.assertRaises(ProcessSupervisionError):
            registry.record_worker_stage(
                HandshakeStage.WORKER_STARTED,
                recorded_at_utc=RECORDED_AT,
            )

    def test_command_fingerprint_and_identity_mismatch_fail_closed(self) -> None:
        registry = make_registry()
        with self.assertRaises(ProcessSupervisionError):
            registry.record_worker_stage(
                HandshakeStage.WORKER_STARTED,
                recorded_at_utc=RECORDED_AT,
                observed_command_fingerprint=OTHER_SHA256,
            )
        wrong_identity = HandshakeEvent(
            sequence=1,
            stage=HandshakeStage.WORKER_STARTED,
            recorded_at_utc=RECORDED_AT,
            run_id="phase3-different-run",
            gpu_uuid="GPU-unit-0001",
            pid=432362,
            process_start_time_ticks=10973359,
            parent_pid=431000,
            command_fingerprint=expected_fingerprint(),
        )
        with self.assertRaises(ProcessSupervisionError):
            registry.ingest_worker_event(wrong_identity)

    def test_malformed_evidence_stage_and_worker_reap_event_are_rejected(self) -> None:
        with self.assertRaises(ProcessSupervisionError):
            HandshakeEvent(
                sequence=5,
                stage=HandshakeStage.EVIDENCE_FLUSHED,
                recorded_at_utc=RECORDED_AT,
                run_id="phase3-unit-run",
                gpu_uuid="GPU-unit-0001",
                pid=432362,
                process_start_time_ticks=10973359,
                parent_pid=431000,
                command_fingerprint=expected_fingerprint(),
            )
        registry = make_registry()
        supervisor_event = HandshakeEvent(
            sequence=7,
            stage=HandshakeStage.SUPERVISOR_REAPED,
            recorded_at_utc=RECORDED_AT,
            run_id="phase3-unit-run",
            gpu_uuid="GPU-unit-0001",
            pid=432362,
            process_start_time_ticks=10973359,
            parent_pid=431000,
            command_fingerprint=expected_fingerprint(),
        )
        with self.assertRaises(ProcessSupervisionError):
            registry.ingest_worker_event(supervisor_event)

    def test_reap_requires_prior_non_reaping_exit_observation(self) -> None:
        registry = make_registry()
        with self.assertRaises(ProcessSupervisionError):
            registry.record_supervisor_reaped(0, recorded_at_utc=RECORDED_AT)


class DeviceProcessOwnershipTests(unittest.TestCase):
    def test_pid_disappearance_retains_registered_child_ownership(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        disposition = registry.observe_proc_start_time(None)
        self.assertIs(
            disposition,
            ProcObservationDisposition.DISAPPEARED_RETAINED,
        )
        verdict = registry.classify_device_snapshot(
            (
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=432362,
                    process_start_time_ticks=None,
                ),
            )
        )
        self.assertIs(verdict.disposition, SnapshotDisposition.OWNED_ONLY)
        reap(registry, 0)
        self.assertIs(
            registry.terminal_outcome().disposition,
            OwnershipDisposition.OWNED_COMPLETED,
        )

    def test_simulated_pid_reuse_is_a_sticky_hard_failure(self) -> None:
        registry = make_registry()
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        self.assertIs(
            registry.observe_proc_start_time(10973360),
            ProcObservationDisposition.PID_REUSE_DETECTED,
        )
        reap(registry, 0)
        outcome = registry.terminal_outcome()
        self.assertIs(
            outcome.disposition,
            OwnershipDisposition.PID_REUSE_DETECTED,
        )
        self.assertFalse(outcome.exclusivity_passed)

    def test_same_pid_with_new_start_time_in_snapshot_is_pid_reuse(self) -> None:
        registry = make_registry()
        verdict = registry.classify_device_snapshot(
            (
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=432362,
                    process_start_time_ticks=10973360,
                ),
            )
        )
        self.assertIs(
            verdict.disposition,
            SnapshotDisposition.PID_REUSE_DETECTED,
        )
        self.assertTrue(verdict.hard_failure)

    def test_unregistered_foreign_process_is_a_hard_failure(self) -> None:
        registry = make_registry()
        verdict = registry.classify_device_snapshot(
            (
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=777777,
                    process_start_time_ticks=900,
                ),
            )
        )
        self.assertIs(
            verdict.disposition,
            SnapshotDisposition.FOREIGN_PROCESS_DETECTED,
        )
        record_through(registry, HandshakeStage.EVIDENCE_FLUSHED)
        reap(registry, 0)
        self.assertIs(
            registry.terminal_outcome().disposition,
            OwnershipDisposition.FOREIGN_PROCESS_DETECTED,
        )

    def test_worker_and_foreign_process_coexistence_remains_a_hard_failure(self) -> None:
        registry = make_registry()
        verdict = registry.classify_device_snapshot(
            (
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=432362,
                    process_start_time_ticks=10973359,
                ),
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=777777,
                    process_start_time_ticks=900,
                ),
            )
        )
        self.assertIs(
            verdict.disposition,
            SnapshotDisposition.FOREIGN_PROCESS_DETECTED,
        )
        self.assertEqual(len(verdict.owned), 1)
        self.assertEqual(len(verdict.foreign), 1)

    def test_missing_start_time_without_disappearance_basis_fails_closed(self) -> None:
        registry = make_registry()
        verdict = registry.classify_device_snapshot(
            (
                DeviceProcessObservation(
                    gpu_uuid="GPU-unit-0001",
                    pid=432362,
                    process_start_time_ticks=None,
                ),
            )
        )
        self.assertIs(
            verdict.disposition,
            SnapshotDisposition.UNVERIFIED_REGISTERED_PID,
        )
        self.assertTrue(verdict.hard_failure)

    def test_process_handle_and_pidfd_metadata_are_preserved(self) -> None:
        registry = make_registry(pidfd=9)
        evidence = registry.to_evidence()
        handle = evidence["handle"]
        self.assertIsInstance(handle, dict)
        assert isinstance(handle, dict)
        self.assertTrue(handle["process_handle_retained"])
        self.assertTrue(handle["pidfd_supported"])
        self.assertTrue(handle["pidfd_opened"])
        self.assertEqual(handle["pidfd"], 9)
        self.assertIsInstance(registry.process_handle, FakeProcessHandle)

    def test_process_handle_fallback_is_retained_without_pidfd_support(self) -> None:
        registry = make_registry(pidfd_supported=False, pidfd=None)
        evidence = registry.to_evidence()
        handle = evidence["handle"]
        self.assertIsInstance(handle, dict)
        assert isinstance(handle, dict)
        self.assertTrue(handle["process_handle_retained"])
        self.assertFalse(handle["pidfd_supported"])
        self.assertFalse(handle["pidfd_opened"])
        self.assertIsNone(handle["pidfd"])


if __name__ == "__main__":
    unittest.main()

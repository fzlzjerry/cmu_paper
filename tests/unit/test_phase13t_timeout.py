"""Focused tests for Phase 13T stage-aware Pilot supervision."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from kvbench.runtime.process_supervision import (
    ProcessSupervisionError,
    run_stage_supervised_command,
)
from scripts import phase13_pilot as pilot
from scripts import phase13t_timeout as timeout_evidence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _event(
    sequence: int,
    stage: str,
    state: str,
    monotonic_ns: int,
) -> dict[str, object]:
    encoded = f"{sequence}:{stage}:{state}:{monotonic_ns}".encode("ascii")
    return {
        "sequence": sequence,
        "stage": stage,
        "state": state,
        "monotonic_ns": monotonic_ns,
        "event_sha256": hashlib.sha256(encoded).hexdigest(),
    }


class StageSupervisionTests(unittest.TestCase):
    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
        }

    def test_stage_supervisor_completes_and_preserves_common_evidence(self) -> None:
        observations: list[dict[str, object]] = []
        first_ns: int | None = None

        def observer() -> list[dict[str, object]]:
            nonlocal first_ns
            now_ns = time.monotonic_ns()
            if not observations:
                first_ns = now_ns
                observations.append(_event(1, "work", "started", now_ns))
            elif (
                len(observations) == 1
                and first_ns is not None
                and now_ns - first_ns > 40_000_000
            ):
                observations.append(_event(2, "work", "completed", now_ns))
            return list(observations)

        result = run_stage_supervised_command(
            (
                sys.executable,
                "-c",
                "import time; time.sleep(0.12); print('stage-ok')",
            ),
            working_directory=str(REPOSITORY_ROOT),
            environment=self._environment(),
            stage_timeouts={
                "startup": 1.0,
                "transition": 1.0,
                "work": 1.0,
            },
            stage_observer=observer,
            startup_stage="startup",
            transition_stage="transition",
            observer_poll_seconds=0.01,
        )

        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.stdout, b"stage-ok\n")
        evidence = result.to_dict()
        self.assertTrue(evidence["direct_child"]["verified"])
        self.assertEqual(evidence["final_reap"]["count"], 1)
        self.assertTrue(
            evidence["stage_supervision"]["no_heartbeat_extension"]
        )
        self.assertEqual(
            [item["state"] for item in evidence["stage_supervision"]["observations"]],
            ["started", "completed"],
        )

    def test_unchanged_observation_cannot_extend_a_stage_deadline(self) -> None:
        observations: list[dict[str, object]] = []

        def observer() -> list[dict[str, object]]:
            if not observations:
                observations.append(
                    _event(1, "work", "started", time.monotonic_ns())
                )
            return list(observations)

        result = run_stage_supervised_command(
            (sys.executable, "-c", "import time; time.sleep(2)"),
            working_directory=str(REPOSITORY_ROOT),
            environment=self._environment(),
            stage_timeouts={
                "startup": 1.0,
                "transition": 1.0,
                "work": 0.08,
            },
            stage_observer=observer,
            startup_stage="startup",
            transition_stage="transition",
            observer_poll_seconds=0.01,
        )

        self.assertTrue(result.timed_out)
        self.assertEqual(result.timeout_stage, "work")
        self.assertGreaterEqual(result.timeout_stage_elapsed_seconds, 0.08)
        self.assertNotEqual(result.returncode, 0)

    def test_observed_stage_evidence_is_immutable(self) -> None:
        observations: list[dict[str, object]] = []
        calls = 0

        def observer() -> list[dict[str, object]]:
            nonlocal calls
            calls += 1
            if not observations:
                observations.append(
                    _event(1, "work", "started", time.monotonic_ns())
                )
            elif calls >= 2:
                observations[0] = {
                    **observations[0],
                    "event_sha256": "0" * 64,
                }
            return list(observations)

        with self.assertRaisesRegex(
            ProcessSupervisionError,
            "changed after observation",
        ):
            run_stage_supervised_command(
                (sys.executable, "-c", "import time; time.sleep(2)"),
                working_directory=str(REPOSITORY_ROOT),
                environment=self._environment(),
                stage_timeouts={
                    "startup": 1.0,
                    "transition": 1.0,
                    "work": 1.0,
                },
                stage_observer=observer,
                startup_stage="startup",
                transition_stage="transition",
                observer_poll_seconds=0.01,
            )


class Phase13StageContractTests(unittest.TestCase):
    @staticmethod
    def _synthetic_diagnostic(root: Path) -> None:
        events: list[dict[str, object]] = []

        def add(stage: str, event: str, elapsed: float, **extra: object) -> None:
            events.append(
                {
                    "schema_version": "kvbench-phase13t-stage-event-1.0.0",
                    "stage": stage,
                    "event": event,
                    "elapsed_seconds": elapsed,
                    "timestamp_ns": int(elapsed * 1_000_000_000),
                    **extra,
                }
            )

        add("diagnostic", "started", 0.0)
        add("worker", "started", 0.5)
        add("model_load", "started", 1.0)
        add("model_load", "completed", 12.0)
        add("session_setup", "started", 12.1)
        add("prefix_construction", "started", 20.0)
        for layer in range(32):
            started = 20.1 + layer * 300.0
            add("prefix_layer", "started", started, layer=layer)
            add(
                "prefix_layer",
                "attention_completed",
                started + 250.0,
                layer=layer,
            )
        add("prefix_construction", "completed", 9_620.0)
        add("graph_capture", "started", 9_621.0)
        add("graph_capture", "completed", 9_631.0)
        add("session_setup", "completed", 9_632.0)
        add("pilot_warmup", "started", 9_633.0)
        add("pilot_warmup", "completed", 9_643.0)
        add("measurement", "started", 9_644.0)
        add("measurement", "completed", 9_654.0)
        add("worker", "completed", 9_655.0)
        add("finalization", "started", 9_656.0)
        add("finalization", "completed", 9_657.0)
        (root / "stage-events.jsonl").write_text(
            "".join(
                json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for item in events
            ),
            encoding="utf-8",
        )
        (root / "worker-result.json").write_text(
            json.dumps(
                {
                    "method_config_id": "kvq2",
                    "batch_size": 1,
                    "context_label": 131072,
                    "historical_context": 131071,
                    "execution_git_sha": timeout_evidence.DIAGNOSTIC_SOURCE_SHA,
                    "authorized_container_digest": pilot.AUTHORIZED_CONTAINER_DIGEST,
                    "finite_output": True,
                    "no_backend_fallback": True,
                    "allocation_stable": True,
                    "r_hbm": None,
                }
            ),
            encoding="utf-8",
        )
        (root / "container-runtime.json").write_text(
            json.dumps(
                {
                    "authorized_container_digest": pilot.AUTHORIZED_CONTAINER_DIGEST,
                    "diagnostic_source_git_sha": timeout_evidence.DIAGNOSTIC_SOURCE_SHA,
                    "native_host_cuda_execution": False,
                }
            ),
            encoding="utf-8",
        )
        for name in (
            "kernel-path.after.normalized.dot",
            "kernel-path.after.raw.dot",
            "kernel-path.before.normalized.dot",
            "kernel-path.before.raw.dot",
            "probe.py",
        ):
            (root / name).write_text("fixture\n", encoding="utf-8")

    def test_prefix_deadline_is_finite_point_scaled_and_deterministic(self) -> None:
        self.assertEqual(
            pilot.prefix_construction_timeout_seconds(
                batch=1,
                historical=131071,
            ),
            34_568.0,
        )
        self.assertEqual(
            pilot.stage_timeout_contract(batch=1, historical=131071),
            pilot.stage_timeout_contract(batch=1, historical=131071),
        )
        contract = pilot.stage_timeout_contract(batch=8, historical=49152)
        self.assertEqual(contract["prefix_construction"], 100_104.0)
        self.assertTrue(all(value > 0 for value in contract.values()))
        with self.assertRaisesRegex(
            pilot.Phase13PilotError,
            "prefix timeout geometry is invalid",
        ):
            pilot.stage_timeout_contract(batch=1, historical=131072)

    def test_stage_files_are_exact_ordered_and_tamper_rejected(self) -> None:
        run_id = "phase13-test-stage-events"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "stage-progress"
            root.mkdir()
            recorder = pilot._WorkerStageRecorder(root=root, run_id=run_id)
            for stage, state in pilot.STAGE_SEQUENCE:
                recorder.record(stage, state)
            observations = pilot._read_stage_observations(
                root=root,
                run_id=run_id,
            )
            self.assertEqual(len(observations), len(pilot.STAGE_SEQUENCE))
            self.assertEqual(
                [(item["stage"], item["state"]) for item in observations],
                list(pilot.STAGE_SEQUENCE),
            )

            first = root / "01-model_load-started.json"
            tampered = json.loads(first.read_text(encoding="utf-8"))
            tampered["unexpected"] = True
            first.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(pilot.Phase13PilotError):
                pilot._read_stage_observations(root=root, run_id=run_id)

    def test_pilot_uses_stage_supervision_without_changing_the_grid(self) -> None:
        source = inspect.getsource(pilot._run_one_process)
        self.assertIn("run_stage_supervised_command", source)
        self.assertNotIn("run_supervised_command(", source)
        order = pilot.derive_execution_order()
        records = pilot.build_feasibility_records(order)
        self.assertEqual(len(records), 810)
        self.assertEqual(
            sum(item["status"] == "feasible" for item in records),
            684,
        )
        self.assertEqual(
            sum(item["status"] == "capacity_infeasible" for item in records),
            126,
        )

    def test_diagnostic_summary_proves_progress_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._synthetic_diagnostic(root)
            summary = timeout_evidence.diagnostic_summary(root)
            self.assertTrue(summary["normal_forward_progress"])
            self.assertFalse(summary["stalled"])
            self.assertFalse(summary["deadlocked"])
            self.assertEqual(summary["prefix_layers_completed"], 32)
            self.assertGreater(
                summary["stages"]["prefix_construction"]["duration_seconds"],
                7_200.0,
            )

            lines = (root / "stage-events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            (root / "stage-events.jsonl").write_text(
                "\n".join(lines[:-1]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(timeout_evidence.Phase13TTimeoutError):
                timeout_evidence.diagnostic_summary(root)


if __name__ == "__main__":
    unittest.main()

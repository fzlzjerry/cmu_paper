"""Focused Phase 6 scope and frozen-governance regressions."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest import mock

from kvbench.schema import MethodAdmissionReportV2
from scripts import phase6_turboquant_admission
from scripts.validate_phase2 import (
    APPROVED_ARTIFACT_ROOT_NAMES,
    PHASE6_ALLOWED_PATHS,
    PHASE6_ENTRY_COMMIT,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _FakeB018Run:
    def __init__(self, name: str) -> None:
        self.stage = Path("/tmp") / name
        self.payloads: dict[str, object] = {}
        self.finalized: list[object] = []

    def write_json(self, relative: str, payload: object) -> None:
        self.payloads[relative] = payload

    def finalize(self, manifest: object) -> Path:
        self.finalized.append(manifest)
        return self.stage


class Phase6GovernanceTests(unittest.TestCase):
    def test_entry_and_allowlist_are_exact(self) -> None:
        self.assertEqual(
            PHASE6_ENTRY_COMMIT,
            "e06f638f4b913f9bd1be2975a478657f5bf2338e",
        )
        required = {
            "docs/plans/phase6-turboquant-measurement-adapter.md",
            "src/kvbench/adapters/turboquant.py",
            "src/kvbench/runtime/phase3_coordinator.py",
            "src/kvbench/runtime/turboquant_cache.py",
            "src/kvbench/runtime/turboquant_session.py",
            "tests/unit/test_process_supervision.py",
            "tests/unit/test_phase6_governance.py",
        }
        self.assertLessEqual(required, PHASE6_ALLOWED_PATHS)
        for rejected in (
            "src/kvbench/adapters/kivi.py",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/plugins/turboquant.py",
            "scripts/phase7_kivi.py",
            "artifacts/quality/result.json",
            "results/turboquant.json",
        ):
            with self.subTest(rejected=rejected):
                self.assertNotIn(rejected, PHASE6_ALLOWED_PATHS)

    def test_artifact_root_allowlist_is_exact(self) -> None:
        self.assertEqual(
            APPROVED_ARTIFACT_ROOT_NAMES,
            frozenset(
                {
                    "README.md",
                    "phase3",
                    "phase3_campaigns",
                    "phase3_reports",
                    "phase4_smoke",
                    "phase6",
                    "phase6a",
                }
            ),
        )

    def test_admission_runtime_venv_stays_inside_ignored_directory(self) -> None:
        makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn(
            'mkdir "$$task_root/source/.venv"',
            makefile,
        )
        self.assertIn(
            'ln -s /opt/kvbench/.venv/bin '
            '"$$task_root/source/.venv/bin"',
            makefile,
        )
        self.assertIn(
            'ln -s /opt/kvbench/.venv/lib '
            '"$$task_root/source/.venv/lib"',
            makefile,
        )
        self.assertIn(
            'ln -s /opt/kvbench/.venv/pyvenv.cfg '
            '"$$task_root/source/.venv/pyvenv.cfg"',
            makefile,
        )
        self.assertNotIn(
            'ln -s /opt/kvbench/.venv "$$task_root/source/.venv"',
            makefile,
        )

    def test_admission_rehydrates_e00_immutable_modes_in_clone(self) -> None:
        makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertEqual(
            makefile.count(
                'chmod -R a-w "$$task_root/source/docs/evidence/e00"'
            ),
            1,
        )
        self.assertEqual(
            makefile.count(
                'find "$$task_root/source/docs/evidence/e00" '
                "-perm /222 -print -quit"
            ),
            1,
        )

    def test_admission_uses_only_the_locked_container_python(self) -> None:
        makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        command = (
            "make PHASE2_PYTHON=/opt/kvbench/.venv/bin/python "
            "PHASE3_PYTHON=/opt/kvbench/.venv/bin/python "
        )
        self.assertEqual(makefile.count(f"{command}test-cuda"), 1)
        self.assertEqual(makefile.count(f"{command}test-graph"), 1)

    def test_validation_target_imports_from_the_repository_root(self) -> None:
        makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn(
            "$(PHASE3_PYTHON) -m scripts.phase6_turboquant_admission "
            "--validate-only",
            makefile,
        )
        self.assertNotIn(
            "$(PHASE3_PYTHON) scripts/phase6_turboquant_admission.py "
            "--validate-only",
            makefile,
        )

    def test_b018_mode_cannot_enter_the_bounded_grid(self) -> None:
        with (
            mock.patch.object(
                phase6_turboquant_admission,
                "run_b018_sanitizer_only",
                return_value={"status": "PASS"},
            ) as sanitizer_only,
            mock.patch.object(
                phase6_turboquant_admission,
                "run_admission",
            ) as full_admission,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = phase6_turboquant_admission.main(
                ["--b018-sanitizer-only"]
            )
        self.assertEqual(exit_code, 0)
        sanitizer_only.assert_called_once_with()
        full_admission.assert_not_called()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            phase6_turboquant_admission._parse_args(
                ["--b018-sanitizer-only", "--validate-only"]
            )

    def test_b018_first_sanitizer_failure_stops_later_configs(self) -> None:
        run = _FakeB018Run("b018-first-failure")
        initial = mock.Mock(run_id="b018-first-failure")
        failed = {
            "passed": False,
            "probe_passed": True,
            "exit_code": 99,
            "memcheck_summaries_passed": False,
        }
        with (
            mock.patch.object(
                phase6_turboquant_admission,
                "_require_clean_git",
                return_value="1" * 40,
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "require_authorized_cuda_environment",
                return_value={"container_digest": "authorized"},
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "phase6_artifact_store",
                return_value=object(),
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "_sanitizer_tool_identity",
                return_value={"sha256": "2" * 64},
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "_b018_cache_identity",
                return_value=("3" * 64, "4" * 64),
            ) as cache_identity,
            mock.patch.object(
                phase6_turboquant_admission,
                "_create_started_run",
                return_value=(run, initial, "2026-07-25T00:00:00Z"),
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "_run_sanitizer_configuration",
                return_value=failed,
            ) as sanitizer,
            mock.patch.object(
                phase6_turboquant_admission,
                "_terminal_manifest",
                side_effect=lambda _initial, **kwargs: kwargs["status"],
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "turboquant_4bit_nc",
            ):
                phase6_turboquant_admission.run_b018_sanitizer_only()
        self.assertEqual(cache_identity.call_count, 1)
        sanitizer.assert_called_once()
        self.assertEqual(
            sanitizer.call_args.args[1],
            "turboquant_4bit_nc",
        )
        self.assertEqual(
            run.finalized,
            [phase6_turboquant_admission.RunStatus.RUNTIME_FAILED],
        )

    def test_b018_success_finalizes_one_artifact_per_configuration(
        self,
    ) -> None:
        configurations = (
            "turboquant_4bit_nc",
            "turboquant_k3v4_nc",
            "turboquant_3bit_nc",
        )
        runs = tuple(_FakeB018Run(f"b018-pass-{index}") for index in range(3))
        initials = tuple(
            mock.Mock(run_id=f"b018-pass-{index}") for index in range(3)
        )
        created = tuple(
            (run, initial, "2026-07-25T00:00:00Z")
            for run, initial in zip(runs, initials, strict=True)
        )
        passed = {
            "passed": True,
            "probe_passed": True,
            "exit_code": 0,
            "memcheck_summaries_passed": True,
        }
        with (
            mock.patch.object(
                phase6_turboquant_admission,
                "_require_clean_git",
                return_value="1" * 40,
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "require_authorized_cuda_environment",
                return_value={"container_digest": "authorized"},
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "phase6_artifact_store",
                return_value=object(),
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "_sanitizer_tool_identity",
                return_value={"sha256": "2" * 64},
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "_b018_cache_identity",
                return_value=("3" * 64, "4" * 64),
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "_create_started_run",
                side_effect=created,
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "_run_sanitizer_configuration",
                return_value=passed,
            ) as sanitizer,
            mock.patch.object(
                phase6_turboquant_admission,
                "_terminal_manifest",
                side_effect=lambda _initial, **kwargs: kwargs["status"],
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "validate_run_directory",
                return_value=mock.Mock(valid=True, complete=True),
            ),
            mock.patch.object(
                phase6_turboquant_admission,
                "_b018_artifact_checksums",
                return_value={"checksums.sha256": "5" * 64},
            ),
        ):
            result = phase6_turboquant_admission.run_b018_sanitizer_only()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            tuple(item["configuration"] for item in result["configurations"]),
            configurations,
        )
        self.assertEqual(sanitizer.call_count, 3)
        self.assertTrue(
            all(
                run.finalized
                == [phase6_turboquant_admission.RunStatus.ABORTED]
                for run in runs
            )
        )

    def test_b018_target_is_exact_digest_offline_and_sanitizer_only(
        self,
    ) -> None:
        makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        start = makefile.index(
            "remediate-b018-turboquant: verify-measurement-container"
        )
        end = makefile.index(
            "\nadmit-turboquant: verify-measurement-container",
            start,
        )
        target = makefile[start:end]
        self.assertIn("--network=none", target)
        self.assertIn(
            'test "$(MEASUREMENT_IMAGE_CONFIG_DIGEST)" = '
            '"$(PHASE6_AUTHORIZED_IMAGE_CONFIG_DIGEST)"',
            target,
        )
        self.assertIn("--b018-sanitizer-only", target)
        self.assertNotIn("test-cuda", target)
        self.assertNotIn("test-graph", target)
        self.assertNotIn("run_admission", target)

    def test_sanitizer_probe_cleans_up_after_evaluator_exception(self) -> None:
        from tests.cuda import phase6_turboquant_sanitizer_probe as probe

        partial = object()

        def fail_evaluator(
            configuration: str,
            *,
            release_cuda_resources_for_sanitizer: bool,
            sanitizer_resources: dict[str, object],
        ) -> dict[str, object]:
            self.assertTrue(release_cuda_resources_for_sanitizer)
            sanitizer_resources["partial"] = partial
            raise ValueError(f"evaluator failure: {configuration}")

        with (
            mock.patch.object(
                probe,
                "require_authorized_cuda_environment",
                return_value={"container_digest": "authorized"},
            ),
            mock.patch.object(
                probe,
                "evaluate_fixture_configuration",
                side_effect=fail_evaluator,
            ),
            mock.patch.object(
                probe,
                "release_fixture_cuda_resources_for_sanitizer",
            ) as release_resources,
            mock.patch.object(
                probe,
                "_release_sanitizer_cuda_state",
            ) as release,
        ):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = probe.main(
                    [
                        "--configuration",
                        "turboquant_4bit_nc",
                        "--image-config-digest",
                        "authorized",
                    ]
                )
        self.assertEqual(exit_code, 2)
        self.assertIn("evaluator failure", stderr.getvalue())
        release_resources.assert_called_once_with(
            {"partial": partial}
        )
        release.assert_called_once_with()

    def test_sanitizer_allocator_drain_reaches_zero_fixed_point(self) -> None:
        from tests.cuda import phase6_turboquant_sanitizer_probe as probe

        fake_torch = mock.Mock()
        fake_torch.cuda.memory_allocated.side_effect = [1, 0]
        fake_torch.cuda.memory_reserved.side_effect = [1, 0]
        with (
            mock.patch.object(probe, "torch", fake_torch),
            mock.patch.object(probe, "gc") as fake_gc,
        ):
            probe._drain_sanitizer_allocator(max_passes=2)

        self.assertEqual(fake_gc.collect.call_count, 2)
        self.assertEqual(fake_torch.cuda.synchronize.call_count, 4)
        self.assertEqual(fake_torch.cuda.empty_cache.call_count, 4)

    def test_sanitizer_allocator_drain_fails_closed_if_nonzero(self) -> None:
        from tests.cuda import phase6_turboquant_sanitizer_probe as probe

        fake_torch = mock.Mock()
        fake_torch.cuda.memory_allocated.return_value = 4
        fake_torch.cuda.memory_reserved.return_value = 8
        with (
            mock.patch.object(probe, "torch", fake_torch),
            mock.patch.object(probe, "gc") as fake_gc,
        ):
            with self.assertRaisesRegex(RuntimeError, "allocated=4 reserved=8"):
                probe._drain_sanitizer_allocator(max_passes=2)

        self.assertEqual(fake_gc.collect.call_count, 2)

    def test_sanitizer_release_orders_drain_destroy_and_reset(self) -> None:
        from tests.cuda import phase6_turboquant_sanitizer_probe as probe

        events: list[str] = []
        fake_torch = mock.Mock()
        fake_torch.cuda.synchronize.side_effect = lambda: events.append("sync")
        fake_torch.cuda.current_blas_handle.side_effect = (
            lambda: events.append("handle") or 7
        )
        cublas = mock.Mock()
        cublas.cublasDestroy_v2.side_effect = (
            lambda _handle: events.append("destroy") or 0
        )
        cudart = mock.Mock()
        cudart.cudaDeviceReset.side_effect = (
            lambda: events.append("reset") or 0
        )
        with (
            mock.patch.object(probe, "torch", fake_torch),
            mock.patch.object(
                probe._build_hadamard_cached,
                "cache_clear",
                side_effect=lambda: events.append("cache_clear"),
            ),
            mock.patch.object(
                probe,
                "_drain_sanitizer_allocator",
                side_effect=lambda: events.append("drain"),
            ),
            mock.patch.object(
                probe.ctypes,
                "CDLL",
                side_effect=[cublas, cudart],
            ),
        ):
            probe._release_sanitizer_cuda_state()

        self.assertEqual(
            events,
            ["sync", "handle", "cache_clear", "drain", "destroy", "reset"],
        )

    def test_sanitizer_probe_resets_only_its_isolated_cuda_context(
        self,
    ) -> None:
        probe = (
            REPOSITORY_ROOT
            / "tests"
            / "cuda"
            / "phase6_turboquant_sanitizer_probe.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(probe.count('ctypes.CDLL("libcudart.so.13")'), 1)
        self.assertEqual(probe.count('ctypes.CDLL("libcublas.so.13")'), 1)
        self.assertEqual(probe.count("cudaDeviceReset"), 2)
        self.assertEqual(probe.count("cublasDestroy_v2"), 2)
        self.assertIn("torch._C._cuda_clearCublasWorkspaces()", probe)
        self.assertIn("torch._C._host_emptyCache()", probe)
        self.assertIn("_build_hadamard_cached.cache_clear()", probe)
        self.assertIn(
            "release_cuda_resources_for_sanitizer=True",
            probe,
        )
        self.assertIn("finally:", probe)
        self.assertLess(
            probe.index("release_cuda_resources_for_sanitizer=True"),
            probe.rindex("_release_sanitizer_cuda_state()"),
        )
        self.assertIn("os._exit(exit_code)", probe)

    def test_plan_freezes_tolerance_and_later_phases(self) -> None:
        plan = (
            REPOSITORY_ROOT
            / "docs"
            / "plans"
            / "phase6-turboquant-measurement-adapter.md"
        ).read_text(encoding="utf-8")
        self.assertIn("`atol=0.02, rtol=0.02`", plan)
        self.assertIn("Phase 7 is explicitly deferred", plan)
        self.assertIn("Full Scan remains closed", plan)
        self.assertIn("`r_hbm` null", plan)

    def test_blocked_method_report_is_strict_and_evidence_backed(
        self,
    ) -> None:
        report_path = (
            REPOSITORY_ROOT
            / "docs"
            / "evidence"
            / "phase6"
            / "turboquant-method-admission.json"
        )
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = MethodAdmissionReportV2.from_dict(payload)
        self.assertEqual(report.status.value, "BLOCKED")
        self.assertEqual(report.blockers, ("B-018",))
        self.assertEqual(report.admitted_config_ids, ())
        self.assertFalse(report.performance_claim_eligible)
        publication = json.loads(
            (
                REPOSITORY_ROOT
                / "docs"
                / "evidence"
                / "phase6"
                / "r2-publication.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(publication["admission_status"], "BLOCKED")
        self.assertEqual(publication["clean_retrieval"]["result"], "PASS")
        self.assertFalse(publication["credential_values_recorded"])

    def test_quality_and_full_scan_remain_locked(self) -> None:
        status = (REPOSITORY_ROOT / "docs" / "status.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Quality execution: LOCKED", status)
        self.assertIn("Full Scan remains CLOSED", status)
        self.assertFalse(any(REPOSITORY_ROOT.rglob("PERFORMANCE_DATA_FROZEN")))


if __name__ == "__main__":
    unittest.main()

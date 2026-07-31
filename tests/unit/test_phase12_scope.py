"""Focused exact-path tests for Phase 12 unified admission."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import validate_phase2
from scripts import phase12_unified_admission as phase12


ROOT = Path(__file__).resolve().parents[2]


def _copy_stopped_phase12_root(destination: Path) -> None:
    source = ROOT / "artifacts" / "phase12"
    reservations = destination / ".kvbench-reservations"
    staging = destination / ".kvbench-staging"
    reservations.mkdir(parents=True)
    staging.mkdir()
    stopped_id = validate_phase2.PHASE12_STOPPED_CAMPAIGN_ID
    shutil.copytree(
        source / ".kvbench-reservations" / stopped_id,
        reservations / stopped_id,
    )
    stopped_stage = next(
        relative.split("/", 1)[1]
        for relative in validate_phase2.PHASE12_BLOCKED_STAGING_DIRECTORIES
        if relative.startswith(".kvbench-staging/")
        and relative.count("/") == 1
    )
    shutil.copytree(
        source / ".kvbench-staging" / stopped_stage,
        staging / stopped_stage,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(phase12.json_bytes(payload))


def _create_active_stage(
    artifact_root: Path,
    *,
    campaign_id: str,
    execution_git_sha: str,
    entry_authority: object,
    entry_gates: object,
) -> Path:
    stage_name = f"{campaign_id}.{('ab' * 12)}.staging"
    stage = artifact_root / ".kvbench-staging" / stage_name
    (artifact_root / ".kvbench-reservations" / campaign_id).mkdir()
    _write_json(
        stage / "campaign-reservation.json",
        {
            "schema_version": (
                "kvbench-phase12-campaign-reservation-1.0.0"
            ),
            "campaign_id": campaign_id,
            "execution_git_sha": execution_git_sha,
            "created_at_utc": "2026-07-31T01:02:03Z",
            "staging_directory": stage_name,
            "reservation_directory": campaign_id,
            "append_only": True,
            "reuse_permitted": False,
        },
    )
    _write_json(stage / "unified" / "entry-authority.json", entry_authority)
    _write_json(stage / "unified" / "entry-g1-g4.json", entry_gates)
    return stage


def _write_completed_container_test(stage: Path, target: str) -> None:
    root = stage / "validation" / target
    root.mkdir(parents=True)
    (root / "command.stdout.txt").write_bytes(b"")
    (root / "command.stderr.txt").write_bytes(b"")
    _write_json(
        root / "command.supervision.json",
        {
            "schema_version": (
                "kvbench-generic-supervised-command-result-1.0.0"
            ),
            "returncode": 0,
            "timeout": {"timed_out": False},
            "direct_child": {
                "verified": True,
                "process_handle_retained": True,
            },
            "final_reap": {"completed": True, "count": 1},
        },
    )
    idle = {
        "query_exit_code": 0,
        "errors": [],
        "allowed_compute_processes": [],
        "foreign_compute_processes": [],
        "unknown_processes": [],
    }
    _write_json(root / "command.gpu-before.json", idle)
    _write_json(root / "command.gpu-after.json", idle)
    _write_json(
        root / "verdict.json",
        {
            "schema_version": (
                "kvbench-phase12-container-test-verdict-1.0.0"
            ),
            "target": target,
            "authorized_container_digest": (
                phase12.PHASE12_AUTHORIZED_CONTAINER_DIGEST
            ),
            "passed": True,
            "cuda_executed_on_native_host": False,
        },
    )


class Phase12ScopeTests(unittest.TestCase):
    def test_entry_freezes_phase11rq23(self) -> None:
        self.assertEqual(
            validate_phase2.PHASE12_ENTRY_COMMIT,
            "845a9293877121187a383c2c7aeab67912c856bd",
        )
        self.assertTrue(
            validate_phase2.commit_is_ancestor(
                validate_phase2.PHASE12_ENTRY_COMMIT
            )
        )
        self.assertEqual(
            validate_phase2.current_phase11rq23_paths(),
            validate_phase2.historical_phase11rq23_paths(),
        )
        self.assertLessEqual(
            validate_phase2.historical_phase11rq23_paths(),
            validate_phase2.PHASE11RQ23_ALLOWED_PATHS,
        )

    def test_phase12_allowlist_is_exact(self) -> None:
        expected = {
            "Makefile",
            "docs/evidence/phase12/r2-publication.json",
            "docs/evidence/phase12/unified-admission.json",
            "docs/phase_reports/phase12-unified-admission.md",
            "docs/plans/phase12-unified-admission.md",
            "docs/risk_register.md",
            "docs/status.md",
            "docs/tasks.md",
            "scripts/phase12_unified_admission.py",
            "scripts/validate_phase2.py",
            "src/kvbench/schema/phase12.py",
            "tests/unit/test_phase12_artifact_lifecycle.py",
            "tests/unit/test_phase12_schema.py",
            "tests/unit/test_phase12_scope.py",
            "tests/unit/test_phase12_unified_admission.py",
        }
        self.assertEqual(validate_phase2.PHASE12_ALLOWED_PATHS, expected)
        self.assertFalse(
            any("*" in path for path in validate_phase2.PHASE12_ALLOWED_PATHS)
        )
        for relative in expected:
            with self.subTest(relative=relative):
                self.assertTrue(
                    validate_phase2.phase12_path_is_allowed(relative)
                )

    def test_current_phase12_segment_is_exactly_scoped(self) -> None:
        current = validate_phase2.current_phase12_paths()
        self.assertLessEqual(
            {
                "scripts/validate_phase2.py",
                "tests/unit/test_phase12_scope.py",
            },
            current,
        )
        self.assertTrue(
            all(
                validate_phase2.phase12_path_is_allowed(relative)
                for relative in current
            )
        )
        self.assertEqual(validate_phase2.changed_paths(), current)

    def test_siblings_traversal_and_out_of_scope_paths_are_rejected(
        self,
    ) -> None:
        rejected = {
            "",
            "/docs/plans/phase12-unified-admission.md",
            "./docs/plans/phase12-unified-admission.md",
            "docs//plans/phase12-unified-admission.md",
            "docs/plans/../plans/phase12-unified-admission.md",
            "docs/plans/phase12-unified-admission.md/../copy.md",
            r"docs\plans\phase12-unified-admission.md",
            "docs/plans/phase12-unified-admission-copy.md",
            "docs/evidence/phase12/publication.json",
            "docs/evidence/phase12/r2-publication-copy.json",
            "docs/evidence/phase12/raw-run.json",
            "docs/blockers.md",
            "src/kvbench/adapters/bf16.py",
            "src/kvbench/adapters/turboquant.py",
            "src/kvbench/adapters/kivi.py",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/fixed_l_runner.py",
            "src/kvbench/runtime/growing_context_runner.py",
            "src/kvbench/runtime/cuda_graph.py",
            "tests/cuda/test_phase12_cuda.py",
            "tests/graph/test_phase12_graph.py",
            "configs/plans/pilot.yaml",
            "configs/plans/full_scan.yaml",
            "artifacts/phase12/campaign/run.json",
            "artifacts/profiler/phase12/result.json",
            "artifacts/quality/phase12/result.json",
            "reference/kvquant_phase11pr/fixtures/COMPLETE",
            "calibration/kvquant/changed/COMPLETE",
            "docker/measurement.Dockerfile",
        }
        for relative in rejected:
            with self.subTest(relative=relative):
                self.assertFalse(
                    validate_phase2.phase12_path_is_allowed(relative)
                )

    def test_entry_did_not_contain_phase12_files(self) -> None:
        for relative in (
            "docs/plans/phase12-unified-admission.md",
            "scripts/phase12_unified_admission.py",
            "src/kvbench/schema/phase12.py",
            "tests/unit/test_phase12_scope.py",
        ):
            with self.subTest(relative=relative):
                result = subprocess.run(
                    [
                        "git",
                        "cat-file",
                        "-e",
                        f"{validate_phase2.PHASE12_ENTRY_COMMIT}:{relative}",
                    ],
                    cwd=ROOT,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.assertNotEqual(result.returncode, 0)

    def test_new_campaign_requires_reservation_and_semantic_replay(
        self,
    ) -> None:
        campaign_id = (
            "phase12-20260731t010203040506z-845a9293-a1b2c3"
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            artifact_root = temporary / "artifacts" / "phase12"
            _copy_stopped_phase12_root(artifact_root)
            campaign = artifact_root / campaign_id
            campaign.mkdir()
            (campaign / "manifest.json").write_text(
                (
                    '{"schema_version":'
                    '"kvbench-phase12-campaign-bundle-1.0.0"}\n'
                ),
                encoding="utf-8",
            )
            (artifact_root / ".kvbench-reservations" / campaign_id).mkdir()
            with (
                mock.patch.object(validate_phase2, "ROOT", temporary),
                mock.patch.object(
                    phase12,
                    "validate_phase12_campaign",
                    return_value={
                        "campaign_id": campaign_id,
                        "status": "PASS",
                    },
                ) as validator,
            ):
                self.assertEqual(
                    validate_phase2.validate_phase12_blocked_artifact_root(),
                    [],
                )
            validator.assert_called_once_with(campaign)

            shutil.rmtree(
                artifact_root / ".kvbench-reservations" / campaign_id
            )
            with (
                mock.patch.object(validate_phase2, "ROOT", temporary),
                mock.patch.object(
                    phase12,
                    "validate_phase12_campaign",
                ),
            ):
                self.assertIn(
                    "Phase 12 reservation/campaign identities differ",
                    validate_phase2.validate_phase12_blocked_artifact_root(),
                )

    def test_new_campaign_tamper_and_stopped_id_reuse_fail_closed(
        self,
    ) -> None:
        campaign_id = (
            "phase12-20260731t010203040506z-845a9293-d4e5f6"
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            artifact_root = temporary / "artifacts" / "phase12"
            _copy_stopped_phase12_root(artifact_root)
            campaign = artifact_root / campaign_id
            campaign.mkdir()
            (campaign / "manifest.json").write_text(
                (
                    '{"schema_version":'
                    '"kvbench-phase12-campaign-bundle-1.0.0"}\n'
                ),
                encoding="utf-8",
            )
            (artifact_root / ".kvbench-reservations" / campaign_id).mkdir()
            with (
                mock.patch.object(validate_phase2, "ROOT", temporary),
                mock.patch.object(
                    phase12,
                    "validate_phase12_campaign",
                    side_effect=phase12.Phase12UnifiedAdmissionError(
                        "tampered campaign"
                    ),
                ),
            ):
                self.assertIn(
                    (
                        "Phase 12 campaign semantic validation failed: "
                        f"{campaign_id}"
                    ),
                    validate_phase2.validate_phase12_blocked_artifact_root(),
                )

            stopped = artifact_root / validate_phase2.PHASE12_STOPPED_CAMPAIGN_ID
            stopped.mkdir()
            with (
                mock.patch.object(validate_phase2, "ROOT", temporary),
                mock.patch.object(
                    phase12,
                    "validate_phase12_campaign",
                ),
            ):
                errors = (
                    validate_phase2.validate_phase12_blocked_artifact_root()
                )
            self.assertTrue(
                any(
                    "invalid campaign roots" in error
                    and validate_phase2.PHASE12_STOPPED_CAMPAIGN_ID in error
                    for error in errors
                )
            )

    def test_historical_stopped_stage_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            artifact_root = temporary / "artifacts" / "phase12"
            _copy_stopped_phase12_root(artifact_root)
            relative = next(
                iter(
                    validate_phase2.PHASE12_BLOCKED_STAGING_FILE_SHA256S
                )
            )
            target = artifact_root / relative
            target.write_bytes(target.read_bytes() + b"\n")
            with mock.patch.object(validate_phase2, "ROOT", temporary):
                self.assertTrue(
                    any(
                        "blocked artifact checksum differs" in error
                        for error in (
                            validate_phase2
                            .validate_phase12_blocked_artifact_root()
                        )
                    )
                )

            extra = target.parent / "unexpected.json"
            extra.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(validate_phase2, "ROOT", temporary):
                self.assertIn(
                    "historical Phase 12 blocked artifact files differ",
                    validate_phase2.validate_phase12_blocked_artifact_root(),
                )

    def test_one_active_stage_is_valid_during_container_tests(self) -> None:
        campaign_id = (
            "phase12-20260731t010203040506z-845a9293-112233"
        )
        execution_git_sha = (
            "845a9293877121187a383c2c7aeab67912c856bd"
        )
        authority = {"authority": "expected"}
        gates = {"gates": "expected"}
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            artifact_root = temporary / "artifacts" / "phase12"
            _copy_stopped_phase12_root(artifact_root)
            stage = _create_active_stage(
                artifact_root,
                campaign_id=campaign_id,
                execution_git_sha=execution_git_sha,
                entry_authority=authority,
                entry_gates=gates,
            )
            with (
                mock.patch.object(validate_phase2, "ROOT", temporary),
                mock.patch.object(
                    phase12,
                    "_expected_entry_authority",
                    return_value=authority,
                ),
                mock.patch.object(
                    phase12,
                    "_expected_entry_g1_g4",
                    return_value=gates,
                ),
            ):
                self.assertEqual(
                    validate_phase2.validate_phase12_blocked_artifact_root(),
                    [],
                )
                _write_completed_container_test(stage, "test-cuda")
                self.assertEqual(
                    validate_phase2.validate_phase12_blocked_artifact_root(),
                    [],
                )

    def test_active_stage_tamper_and_extra_stage_fail_closed(self) -> None:
        campaign_id = (
            "phase12-20260731t010203040506z-845a9293-445566"
        )
        execution_git_sha = (
            "845a9293877121187a383c2c7aeab67912c856bd"
        )
        authority = {"authority": "expected"}
        gates = {"gates": "expected"}
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            artifact_root = temporary / "artifacts" / "phase12"
            _copy_stopped_phase12_root(artifact_root)
            stage = _create_active_stage(
                artifact_root,
                campaign_id=campaign_id,
                execution_git_sha=execution_git_sha,
                entry_authority=authority,
                entry_gates=gates,
            )
            _write_json(
                stage / "unified" / "entry-authority.json",
                {"authority": "tampered"},
            )
            with (
                mock.patch.object(validate_phase2, "ROOT", temporary),
                mock.patch.object(
                    phase12,
                    "_expected_entry_authority",
                    return_value=authority,
                ),
                mock.patch.object(
                    phase12,
                    "_expected_entry_g1_g4",
                    return_value=gates,
                ),
            ):
                self.assertIn(
                    (
                        "Phase 12 active staging semantic validation failed: "
                        f"{campaign_id}"
                    ),
                    validate_phase2.validate_phase12_blocked_artifact_root(),
                )

            _write_json(
                stage / "unified" / "entry-authority.json",
                authority,
            )
            (stage / "unexpected.bin").write_bytes(b"unexpected")
            with (
                mock.patch.object(validate_phase2, "ROOT", temporary),
                mock.patch.object(
                    phase12,
                    "_expected_entry_authority",
                    return_value=authority,
                ),
                mock.patch.object(
                    phase12,
                    "_expected_entry_g1_g4",
                    return_value=gates,
                ),
            ):
                self.assertIn(
                    (
                        "Phase 12 active staging semantic validation failed: "
                        f"{campaign_id}"
                    ),
                    validate_phase2.validate_phase12_blocked_artifact_root(),
                )
            (stage / "unexpected.bin").unlink()

            orphan_id = (
                "phase12-20260731t010203040506z-845a9293-aabbcc"
            )
            (
                artifact_root / ".kvbench-reservations" / orphan_id
            ).mkdir()
            with mock.patch.object(validate_phase2, "ROOT", temporary):
                self.assertIn(
                    "Phase 12 reservation/campaign identities differ",
                    validate_phase2.validate_phase12_blocked_artifact_root(),
                )

            extra_id = (
                "phase12-20260731t010203040506z-845a9293-778899"
            )
            extra = (
                artifact_root
                / ".kvbench-staging"
                / f"{extra_id}.{('cd' * 12)}.staging"
            )
            extra.mkdir()
            with mock.patch.object(validate_phase2, "ROOT", temporary):
                self.assertIn(
                    "Phase 12 staging contains multiple active campaigns",
                    validate_phase2.validate_phase12_blocked_artifact_root(),
                )


if __name__ == "__main__":
    unittest.main()

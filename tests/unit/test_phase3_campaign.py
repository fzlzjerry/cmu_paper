"""CPU-only append-only campaign preregistration tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from kvbench.runtime.phase3_campaign import (
    Phase3CampaignError,
    Phase3CampaignRecorder,
    assert_unique_plan_campaign,
    campaign_root,
    validate_phase3_campaign_directory,
)


CAMPAIGN_ID = "phase3-20260722t010203000001z-12345678-abcdef"
PLAN_PATH = "configs/plans/phase3_bf16_fixed_l.yaml"
PLAN_SHA = "1" * 64
GIT_SHA = "2" * 40
POINTS = ("point-a", "point-b")
RUNS = tuple(f"{CAMPAIGN_ID}-{point}" for point in POINTS)


def campaign_result() -> dict[str, object]:
    return {
        "schema_version": "kvbench-phase3-campaign-result-1.0.0",
        "ok": False,
        "campaign_id": CAMPAIGN_ID,
        "git_sha": GIT_SHA,
        "plan": PLAN_PATH,
        "plan_fingerprint": PLAN_SHA,
        "expected_process_count": 2,
        "attempted_process_count": 2,
        "unattempted_point_ids": [],
        "status_counts": {"allocation_failed": 2},
        "runs": [
            {"run_id": run_id, "point_id": point_id, "status": "allocation_failed"}
            for point_id, run_id in zip(POINTS, RUNS)
        ],
        "execution_attempted": True,
        "timing_collected": False,
        "profiler_executed": False,
        "quality_executed": False,
        "performance_claim_eligible": False,
        "measurement_scope": "native_host_admission",
        "selective_rerun_performed": False,
        "preregistered_before_execution": True,
        "unexpected_campaign_abort": False,
        "unexpected_failure": None,
        "finished_at_utc": "2026-07-22T01:02:04+00:00",
    }


class Phase3CampaignTests(unittest.TestCase):
    def test_preregistration_prevents_same_plan_sha_rerun_and_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            recorder = Phase3CampaignRecorder.create(
                repository_root=root,
                campaign_id=CAMPAIGN_ID,
                created_at_utc="2026-07-22T01:02:03+00:00",
                git_sha=GIT_SHA,
                plan_path=PLAN_PATH,
                plan_fingerprint=PLAN_SHA,
                point_ids=POINTS,
                run_ids=RUNS,
            )
            with self.assertRaises(Phase3CampaignError):
                assert_unique_plan_campaign(
                    campaign_root(root),
                    plan_path=PLAN_PATH,
                    git_sha=GIT_SHA,
                )
            directory = recorder.finalize(campaign_result())
            validation = validate_phase3_campaign_directory(directory)
            self.assertTrue(validation["valid"], validation["errors"])
            for path in sorted(directory.iterdir(), reverse=True):
                path.chmod(0o644)
            directory.chmod(0o755)

    def test_campaign_tampering_fails_independent_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            recorder = Phase3CampaignRecorder.create(
                repository_root=root,
                campaign_id=CAMPAIGN_ID,
                created_at_utc="2026-07-22T01:02:03+00:00",
                git_sha=GIT_SHA,
                plan_path=PLAN_PATH,
                plan_fingerprint=PLAN_SHA,
                point_ids=POINTS,
                run_ids=RUNS,
            )
            directory = recorder.finalize(campaign_result())
            result_path = directory / "result.json"
            directory.chmod(0o755)
            result_path.chmod(0o644)
            result_path.write_bytes(result_path.read_bytes() + b" ")
            result_path.chmod(0o444)
            directory.chmod(0o555)
            self.assertFalse(validate_phase3_campaign_directory(directory)["valid"])
            for path in sorted(directory.iterdir(), reverse=True):
                path.chmod(0o644)
            directory.chmod(0o755)


if __name__ == "__main__":
    unittest.main()

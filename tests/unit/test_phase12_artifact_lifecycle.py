"""Focused tests for the append-only Phase 12 campaign artifact lifecycle."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from kvbench.schema.base import QualityExecutionState
from kvbench.schema.phase3 import GateDisposition
from kvbench.schema.phase12 import (
    PHASE12_AUTHORIZED_CONTAINER_DIGEST,
    PHASE12_RANDOMIZATION_SEEDS,
    PHASE12_RANDOMIZED_ORDERS,
    Phase12GlobalGates,
    Phase12PublicationState,
    Phase12UnifiedAdmissionReport,
)
from scripts import phase12_unified_admission as phase12
from scripts.r2_artifact import ArtifactValidationError
from tests.unit.test_phase12_schema import _report


CAMPAIGN_ID = (
    "phase12-20260730t000000000000z-2bc6aaa1-abcdef"
)
GIT_SHA = "2bc6aaa1d05b08d50f4c01bbc0b2863dd8689fe1"
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def _pending_gates() -> Phase12GlobalGates:
    return Phase12GlobalGates(
        g0=GateDisposition.PASS,
        g1=GateDisposition.PASS,
        g2=GateDisposition.PASS,
        g3=GateDisposition.PASS,
        g4=GateDisposition.PASS,
        g5=GateDisposition.NOT_EVALUATED,
        pilot_state="NOT_READY",
        full_scan_state="CLOSED",
        quality_execution=QualityExecutionState.LOCKED,
        performance_data_frozen=False,
    )


def _pending_report(
    campaign_id: str = CAMPAIGN_ID,
) -> Phase12UnifiedAdmissionReport:
    original = _report(
        publication_state=Phase12PublicationState.PENDING,
        gates=_pending_gates(),
    )
    renamed_runs = tuple(
        dataclasses.replace(
            item,
            run_id=(
                f"{campaign_id}-r{item.replicate_index}-"
                f"{item.order_index:02d}-{item.method_config_id}"
            ),
            manifest_path=(
                "runs/"
                f"{campaign_id}-r{item.replicate_index}-"
                f"{item.order_index:02d}-{item.method_config_id}"
                "/manifest.json"
            ),
        )
        for item in original.runs
    )
    run_ids = {
        item.method_config_id: tuple(
            run.run_id
            for run in renamed_runs
            if run.method_config_id == item.method_config_id
        )
        for item in original.g5_statistics
    }
    renamed_statistics = tuple(
        dataclasses.replace(
            item,
            run_ids=run_ids[item.method_config_id],
        )
        for item in original.g5_statistics
    )
    return dataclasses.replace(
        original,
        campaign_id=campaign_id,
        runs=renamed_runs,
        g5_statistics=renamed_statistics,
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, payload: object) -> None:
    _write_bytes(path, phase12.json_bytes(payload))


def _idle_snapshot() -> dict[str, object]:
    return {
        "query_exit_code": 0,
        "errors": [],
        "allowed_compute_processes": [],
        "foreign_compute_processes": [],
        "unknown_processes": [],
    }


def _supervision_result() -> dict[str, object]:
    return {
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
    }


def _bucket_lock() -> dict[str, object]:
    return {
        "provider": "cloudflare_r2",
        "endpoint_class": "cloudflare_r2_s3",
        "bucket": "kvbench-artifacts",
        "bucket_exists": True,
        "bucket_public": False,
        "managed_r2_dev_enabled": False,
        "public_r2_dev": False,
        "custom_domain_count": 0,
        "enabled_custom_domain_count": 0,
        "public_custom_domain": False,
        "public_state_result": "PASS",
        "verification_result": "PASS",
        "enabled": True,
        "lock_rule_id": "kvbench-evidence-indefinite",
        "lock_rule_name": None,
        "covered_prefix": "kvbench/sha256/",
        "lock_prefix": "kvbench/sha256/",
        "lock_scope": "exact",
        "retention_type": "Indefinite",
        "retention_condition": "Indefinite",
        "endpoint": (
            "https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com"
        ),
        "verified_at_utc": "2026-07-30T00:00:00Z",
    }


def _r2_result(
    *,
    operation: str,
    root_sha256: str,
    object_count: int,
) -> dict[str, object]:
    uri = (
        "r2://kvbench-artifacts/kvbench/sha256/"
        f"{root_sha256}/"
    )
    common = {
        "provider": "cloudflare_r2",
        "root_sha256": root_sha256,
        "uri": uri,
        "object_count": object_count,
    }
    result = (
        {
            **common,
            "complete_last": True,
            "uploaded_count": object_count,
            "verified_existing_count": 0,
            "published_at_utc": "2026-07-30T00:00:00Z",
            "publication_order_sha256": "a" * 64,
        }
        if operation == "publish"
        else {
            **common,
            "verification_result": "PASS",
            "checksum_ledger_valid": True,
            "complete_marker_valid": True,
            "inventory_valid": True,
            "unexpected_objects": False,
            "retrieved_at_utc": "2026-07-30T00:00:01Z",
        }
    )
    return {
        "status": "PASS",
        "required_variables": {
            name: "PRESENT"
            for name in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "CLOUDFLARE_API_TOKEN",
                "KVBENCH_R2_PREFIX",
                "R2_ACCOUNT_ID",
                "R2_BUCKET",
                "R2_ENDPOINT",
            )
        },
        "r2": {
            "bucket": "kvbench-artifacts",
            "endpoint": (
                "https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com"
            ),
            "endpoint_class": "cloudflare_r2_s3",
            "prefix": "kvbench/sha256",
            "provider": "cloudflare_r2",
            "region": "auto",
        },
        "bucket_lock": _bucket_lock(),
        operation: result,
    }


def _materialize_payload(
    root: Path,
    report: Phase12UnifiedAdmissionReport | None = None,
) -> Phase12UnifiedAdmissionReport:
    selected = _pending_report() if report is None else report
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "campaign-reservation.json",
        {
            "schema_version": (
                "kvbench-phase12-campaign-reservation-1.0.0"
            ),
            "campaign_id": selected.campaign_id,
            "execution_git_sha": selected.execution_git_sha,
            "created_at_utc": "2026-07-30T00:00:00Z",
            "append_only": True,
            "reuse_permitted": False,
        },
    )
    entry = phase12._expected_entry_g1_g4()
    _write_json(root / "unified" / "entry-g1-g4.json", entry)
    _write_json(
        root / "unified" / "entry-authority.json",
        phase12._expected_serialized_entry_authority(
            campaign_id=selected.campaign_id,
            execution_git_sha=selected.execution_git_sha,
        ),
    )
    entry_authority_sha256 = phase12.sha256_file(
        root / "unified" / "entry-authority.json"
    )
    bridged_configurations = []
    for configuration in selected.configurations:
        method = phase12._method_family(
            configuration.method_config_id
        )
        bridged_configurations.append(
            dataclasses.replace(
                configuration,
                prior_gates=tuple(
                    dataclasses.replace(
                        gate,
                        evidence=(
                            phase12.Phase12EvidenceReference(
                                evidence_id=(
                                    f"{configuration.method_config_id}_"
                                    f"{gate.gate.lower()}_method_admission"
                                ),
                                path=phase12.PRIOR_ADMISSION_REPORT_BINDINGS[
                                    method
                                ].as_posix(),
                                sha256=phase12.EXPECTED_REPORT_SHA256S[method],
                            ),
                        )
                        + (
                            (
                                phase12.Phase12EvidenceReference(
                                    evidence_id=(
                                        f"{configuration.method_config_id}_"
                                        f"{gate.gate.lower()}_"
                                        "historical_source_bridge"
                                    ),
                                    path="unified/entry-authority.json",
                                    sha256=entry_authority_sha256,
                                ),
                            )
                            if method in {"bf16", "turboquant"}
                            else ()
                        ),
                    )
                    for gate in configuration.prior_gates
                ),
            )
        )
    selected = dataclasses.replace(
        selected,
        configurations=tuple(bridged_configurations),
    )

    for target in ("test-cuda", "test-graph"):
        base = root / "validation" / target
        _write_bytes(base / "command.stdout.txt", b"")
        _write_bytes(base / "command.stderr.txt", b"")
        _write_json(
            base / "command.supervision.json",
            _supervision_result(),
        )
        _write_json(base / "command.gpu-before.json", _idle_snapshot())
        _write_json(base / "command.gpu-after.json", _idle_snapshot())
        _write_json(
            base / "verdict.json",
            {
                "target": target,
                "authorized_container_digest": (
                    PHASE12_AUTHORIZED_CONTAINER_DIGEST
                ),
                "passed": True,
                "cuda_executed_on_native_host": False,
            },
        )

    updated_runs = []
    for compact in selected.runs:
        run_root = root / compact.manifest_path
        run_root = run_root.parent
        _write_json(run_root / "started.json", {})
        _write_bytes(run_root / "worker.stdout.txt", b"")
        _write_bytes(run_root / "worker.stderr.txt", b"")
        _write_json(
            run_root / "worker.supervision.json",
            _supervision_result(),
        )
        _write_json(
            run_root / "worker.gpu-before.json",
            _idle_snapshot(),
        )
        _write_json(
            run_root / "worker.gpu-after.json",
            _idle_snapshot(),
        )
        path_records = {}
        for phase, pointer in (("before", "0xAAA0"), ("after", "0xBBB0")):
            raw = (
                "digraph dot {\n"
                "subgraph cluster_4 {\n"
                '\"graph_4_node_0\"[shape=\"record\" '
                f'label=\"{{KERNEL|{{node handle | {pointer}}}}}\"];\n'
                "}\n"
                "}\n"
            ).encode("utf-8")
            normalized, nodes, kernels, edges = (
                phase12._normalize_cuda_graph_debug_dot(raw)
            )
            raw_path = run_root / f"kernel-path.{phase}.raw.dot"
            normalized_path = (
                run_root / f"kernel-path.{phase}.normalized.dot"
            )
            _write_bytes(raw_path, raw)
            _write_bytes(normalized_path, normalized)
            path_records[phase] = {
                "schema_version": (
                    "kvbench-phase12-cuda-graph-path-witness-1.0.0"
                ),
                "phase": phase,
                "observation_kind": "cuda_graph_debug_dot",
                "raw_path": raw_path.name,
                "raw_sha256": phase12.sha256_file(raw_path),
                "normalized_path": normalized_path.name,
                "normalized_sha256": phase12.sha256_file(
                    normalized_path
                ),
                "node_count": nodes,
                "kernel_node_count": kernels,
                "edge_count": edges,
                "process_local_handles_normalized": True,
                "timing_collected": False,
                "profiler_used": False,
            }
        path_observation = {
            "schema_version": (
                "kvbench-phase12-cuda-graph-path-observation-1.0.0"
            ),
            "observation_kind": "cuda_graph_debug_dot",
            "before": path_records["before"],
            "after": path_records["after"],
            "normalized_sha256": path_records["before"][
                "normalized_sha256"
            ],
            "node_count": path_records["before"]["node_count"],
            "kernel_node_count": path_records["before"][
                "kernel_node_count"
            ],
            "edge_count": path_records["before"]["edge_count"],
            "graph_exec_pointer_stable": True,
            "topology_stable_within_process": True,
            "timing_collected": False,
            "profiler_used": False,
        }
        _write_json(
            run_root / "result.json",
            {"kernel_path_observation": path_observation},
        )
        result_relative = (
            run_root / "result.json"
        ).relative_to(root).as_posix()
        manifest = {
            "schema_version": (
                "kvbench-phase12-g5-run-manifest-1.0.0"
            ),
            "run_id": compact.run_id,
            "campaign_id": selected.campaign_id,
            "status": "completed",
            "method_config_id": compact.method_config_id,
            "method_config_fingerprint": (
                phase12.EXPECTED_CONFIG_FINGERPRINTS[
                    compact.method_config_id
                ]
            ),
            "replicate_index": compact.replicate_index,
            "seed": compact.seed,
            "order_index": compact.order_index,
            "execution_git_sha": selected.execution_git_sha,
            "authorized_container_digest": (
                PHASE12_AUTHORIZED_CONTAINER_DIGEST
            ),
            "runner_kind": phase12.PHASE12_RUNNER_KIND,
            "graph_mode": phase12.PHASE12_GRAPH_MODE,
            "batch_size": phase12.PHASE12_BATCH_SIZE,
            "context_length": phase12.PHASE12_CONTEXT_LENGTH,
            "warmup_steps": phase12.PHASE12_WARMUP_STEPS,
            "measured_steps": phase12.PHASE12_MEASURED_STEPS,
            "measured_batches": phase12.PHASE12_MEASURED_BATCHES,
            "result_path": result_relative,
            "result_sha256": phase12.sha256_file(
                run_root / "result.json"
            ),
            "stdout_sha256": phase12.sha256_file(
                run_root / "worker.stdout.txt"
            ),
            "stderr_sha256": phase12.sha256_file(
                run_root / "worker.stderr.txt"
            ),
            "supervision_sha256": phase12.sha256_file(
                run_root / "worker.supervision.json"
            ),
            "gpu_before_sha256": phase12.sha256_file(
                run_root / "worker.gpu-before.json"
            ),
            "gpu_after_sha256": phase12.sha256_file(
                run_root / "worker.gpu-after.json"
            ),
            "kernel_path_before_raw_path": (
                run_root / "kernel-path.before.raw.dot"
            ).relative_to(root).as_posix(),
            "kernel_path_before_raw_sha256": phase12.sha256_file(
                run_root / "kernel-path.before.raw.dot"
            ),
            "kernel_path_before_normalized_path": (
                run_root / "kernel-path.before.normalized.dot"
            ).relative_to(root).as_posix(),
            "kernel_path_before_normalized_sha256": phase12.sha256_file(
                run_root / "kernel-path.before.normalized.dot"
            ),
            "kernel_path_after_raw_path": (
                run_root / "kernel-path.after.raw.dot"
            ).relative_to(root).as_posix(),
            "kernel_path_after_raw_sha256": phase12.sha256_file(
                run_root / "kernel-path.after.raw.dot"
            ),
            "kernel_path_after_normalized_path": (
                run_root / "kernel-path.after.normalized.dot"
            ).relative_to(root).as_posix(),
            "kernel_path_after_normalized_sha256": phase12.sha256_file(
                run_root / "kernel-path.after.normalized.dot"
            ),
            "speedup_calculated": False,
            "r_hbm": None,
            "selective_rerun": False,
        }
        _write_json(run_root / "manifest.json", manifest)
        updated_runs.append(
            dataclasses.replace(
                compact,
                manifest_sha256=phase12.sha256_file(
                    run_root / "manifest.json"
                ),
            )
        )

    selected = dataclasses.replace(
        selected,
        runs=tuple(updated_runs),
    )
    for admission, summary in zip(
        selected.configurations,
        selected.g5_statistics,
        strict=True,
    ):
        configuration = admission.method_config_id
        method = phase12._method_family(configuration)
        _write_json(
            root / "admission" / configuration / "report.json",
            {
                "schema_version": phase12.PHASE12_PER_CONFIG_SCHEMA,
                "campaign_id": selected.campaign_id,
                "method_config_id": configuration,
                "method_family": method,
                "configuration_admission": admission.to_dict(),
                "prior_gate_check_ids": entry["configurations"][
                    configuration
                ]["evidence"],
                "g5_statistics": summary.to_dict(),
                "prior_method_admission_path": (
                    phase12.PRIOR_ADMISSION_REPORT_BINDINGS[
                        method
                    ].as_posix()
                ),
                "prior_method_admission_sha256": (
                    phase12.EXPECTED_REPORT_SHA256S[method]
                ),
                "speedup_calculated": False,
                "r_hbm": None,
            },
        )
    _write_json(
        root / "unified" / "local-admission.json",
        selected.to_dict(),
    )
    _write_json(
        root / "unified" / "campaign-result.json",
        {
            "schema_version": (
                "kvbench-phase12-local-campaign-result-1.0.0"
            ),
            "campaign_id": selected.campaign_id,
            "execution_git_sha": selected.execution_git_sha,
            "container_digest": PHASE12_AUTHORIZED_CONTAINER_DIGEST,
            "expected_runs": 30,
            "completed_runs": 30,
            "failed_runs": 0,
            "local_g1_g4": "PASS",
            "local_g5_reproducibility": "PASS",
            "durable_publication": "PENDING_HOST_SIDE",
            "global_g5": "NOT_EVALUATED",
            "pilot": "NOT_READY",
            "full_scan": "CLOSED",
            "quality_execution": "LOCKED",
            "performance_data_frozen": False,
            "selective_reruns": 0,
            "speedup_calculated": False,
            "r_hbm": None,
            "local_admission_path": "unified/local-admission.json",
            "local_admission_sha256": phase12.sha256_file(
                root / "unified" / "local-admission.json"
            ),
        },
    )
    return selected


def _compact_replay(
    report: Phase12UnifiedAdmissionReport,
):
    by_path = {item.manifest_path: item for item in report.runs}

    def replay(
        *,
        payload: object,
        manifest_path: str,
        manifest_sha256: str,
    ):
        del payload
        compact = by_path[manifest_path]
        if manifest_sha256 != compact.manifest_sha256:
            raise AssertionError("test fixture manifest SHA differs")
        return compact

    return replay


def _make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            path.chmod(0o700)
        elif path.is_file():
            path.chmod(0o600)
    root.chmod(0o700)


class Phase12ArtifactLifecycleTests(unittest.TestCase):
    def test_failed_campaign_is_terminally_sealed_and_cannot_resume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            artifact_root = temporary / "artifacts" / "phase12"
            stage = (
                artifact_root
                / ".kvbench-staging"
                / f"{CAMPAIGN_ID}.fedcba9876543210.staging"
            )
            run_id = f"{CAMPAIGN_ID}-r0-00-k2v4"
            _write_json(
                stage / "campaign-reservation.json",
                {
                    "schema_version": (
                        "kvbench-phase12-campaign-reservation-1.0.0"
                    ),
                    "campaign_id": CAMPAIGN_ID,
                    "execution_git_sha": GIT_SHA,
                    "created_at_utc": "2026-07-30T00:00:00Z",
                    "append_only": True,
                    "reuse_permitted": False,
                },
            )
            _write_json(
                stage / "runs" / run_id / "started.json",
                {
                    "schema_version": "kvbench-phase12-run-start-1.0.0",
                    "run_id": run_id,
                },
            )
            try:
                with mock.patch.object(
                    phase12,
                    "PHASE12_ARTIFACT_ROOT",
                    artifact_root,
                ):
                    result = phase12.finalize_failed_phase12_campaign(
                        stage=stage,
                        campaign_id=CAMPAIGN_ID,
                        git_sha=GIT_SHA,
                        failure_code=2,
                    )
                    final = artifact_root / CAMPAIGN_ID
                    validated = phase12.validate_local_artifact(
                        final,
                        environ={},
                    )
                    self.assertEqual(result["status"], "FAILED_PRESERVED")
                    self.assertFalse(stage.exists())
                    self.assertTrue(final.is_dir())
                    self.assertGreater(len(validated.files), 0)
                    self.assertEqual(
                        json.loads(
                            (final / "failure.json").read_text(
                                encoding="utf-8"
                            )
                        )["preserved_run_ids"],
                        [run_id],
                    )
                    self.assertFalse(
                        any(
                            path.lstat().st_mode & WRITE_BITS
                            for path in (final, *final.rglob("*"))
                        )
                    )
                    with self.assertRaises(
                        phase12.Phase12UnifiedAdmissionError
                    ):
                        phase12.finalize_failed_phase12_campaign(
                            stage=stage,
                            campaign_id=CAMPAIGN_ID,
                            git_sha=GIT_SHA,
                            failure_code=2,
                        )
                    with self.assertRaises(
                        phase12.Phase12UnifiedAdmissionError
                    ):
                        phase12.reserve_campaign(
                            campaign_id=CAMPAIGN_ID,
                            git_sha=GIT_SHA,
                        )
            finally:
                _make_tree_writable(temporary)

    def test_exact_payload_topology_validates_and_rejects_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "payload"
            report = _materialize_payload(root)
            expected = phase12._expected_payload_paths(
                CAMPAIGN_ID,
                report,
            )
            actual = {
                path.relative_to(root).as_posix()
                for path in phase12._payload_paths(
                    root,
                    set(phase12._CONTROL_FILES),
                )
            }
            self.assertEqual(actual, expected)
            self.assertEqual(len(expected), 387)
            self.assertFalse(
                any("r2-publication" in path for path in expected)
            )
            with (
                mock.patch.object(
                    phase12,
                    "_validate_worker_result",
                ),
                mock.patch.object(
                    phase12,
                    "_compact_g5_run",
                    side_effect=_compact_replay(report),
                ),
                mock.patch.object(
                    phase12,
                    "_expected_entry_authority",
                    side_effect=AssertionError("live replay attempted"),
                ),
                mock.patch.object(
                    phase12,
                    "REPOSITORY_ROOT",
                    Path(directory) / "missing-repository",
                ),
            ):
                observed, result = phase12.validate_phase12_payload(
                    root,
                    expected_campaign_id=CAMPAIGN_ID,
                )
            self.assertEqual(observed, report)
            self.assertEqual(result["completed_runs"], 30)

            _write_json(
                root
                / "docs"
                / "evidence"
                / "phase12"
                / "r2-publication.json",
                {"status": "PASS"},
            )
            with self.assertRaisesRegex(
                phase12.Phase12UnifiedAdmissionError,
                "payload topology",
            ):
                phase12.validate_phase12_payload(
                    root,
                    expected_campaign_id=CAMPAIGN_ID,
                )

    def test_synthetic_authority_bridges_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            first = _materialize_payload(temporary / "first")
            second = _materialize_payload(temporary / "second", first)
            for configuration in second.configurations:
                method = phase12._method_family(
                    configuration.method_config_id
                )
                for gate in configuration.prior_gates:
                    expected_paths = [
                        phase12.PRIOR_ADMISSION_REPORT_BINDINGS[
                            method
                        ].as_posix()
                    ]
                    if method in {"bf16", "turboquant"}:
                        expected_paths.append(
                            "unified/entry-authority.json"
                        )
                    self.assertEqual(
                        [item.path for item in gate.evidence],
                        expected_paths,
                    )

    def test_reservation_and_child_run_ids_are_no_replace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            plan = repository / phase12.PHASE12_PLAN_PATH
            _write_bytes(plan, b"# synthetic committed Phase 12 plan\n")
            plan_sha256 = phase12.sha256_file(plan)
            with (
                mock.patch.object(
                    phase12,
                    "PHASE12_PLAN_SHA256",
                    plan_sha256,
                ),
                mock.patch.object(
                    phase12,
                    "load_and_validate_prior_admission_evidence",
                    return_value={
                        "methods": {
                            "bf16": {"authority_bridge": {}},
                            "turboquant": {"authority_bridge": {}},
                        }
                    },
                ),
                mock.patch.object(
                    phase12,
                    "_expected_historical_source_bridges",
                    return_value={"bf16": {}, "turboquant": {}},
                ),
                mock.patch.object(
                    phase12,
                    "_expected_entry_authority",
                    return_value={},
                ),
                mock.patch.object(
                    phase12,
                    "aggregate_g1_g4",
                    return_value=phase12._expected_entry_g1_g4(),
                ),
            ):
                stage = phase12.reserve_campaign(
                    campaign_id=CAMPAIGN_ID,
                    git_sha=GIT_SHA,
                    repo_root=repository,
                )
                self.assertTrue(stage.is_dir())
                with self.assertRaisesRegex(
                    phase12.Phase12UnifiedAdmissionError,
                    "already used",
                ):
                    phase12.reserve_campaign(
                        campaign_id=CAMPAIGN_ID,
                        git_sha=GIT_SHA,
                        repo_root=repository,
                    )

            configuration = PHASE12_RANDOMIZED_ORDERS[0][0]
            run_id = f"{CAMPAIGN_ID}-r0-00-{configuration}"
            (stage / "runs" / run_id).mkdir(parents=True)
            with self.assertRaisesRegex(
                phase12.Phase12UnifiedAdmissionError,
                "run ID already exists",
            ):
                phase12._run_one_process(
                    stage=stage,
                    campaign_id=CAMPAIGN_ID,
                    configuration=configuration,
                    replicate_index=0,
                    seed=PHASE12_RANDOMIZATION_SEEDS[0],
                    order_index=0,
                    git_sha=GIT_SHA,
                )

    def test_tampered_child_manifest_and_config_report_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "payload"
            report = _materialize_payload(root)
            first_manifest = root / report.runs[0].manifest_path
            manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
            manifest["status"] = "unstable"
            _write_bytes(first_manifest, phase12.json_bytes(manifest))
            with self.assertRaisesRegex(
                phase12.Phase12UnifiedAdmissionError,
                "run-manifest checksum",
            ):
                phase12.validate_phase12_payload(
                    root,
                    expected_campaign_id=CAMPAIGN_ID,
                )

            _write_bytes(
                first_manifest,
                phase12.json_bytes(
                    {
                        **manifest,
                        "status": "completed",
                    }
                ),
            )
            first_report = (
                root
                / "admission"
                / report.configurations[0].method_config_id
                / "report.json"
            )
            payload = json.loads(first_report.read_text(encoding="utf-8"))
            payload["speedup_calculated"] = True
            _write_bytes(first_report, phase12.json_bytes(payload))
            with (
                mock.patch.object(
                    phase12,
                    "_validate_worker_result",
                ),
                mock.patch.object(
                    phase12,
                    "_compact_g5_run",
                    side_effect=_compact_replay(report),
                ),
                self.assertRaisesRegex(
                    phase12.Phase12UnifiedAdmissionError,
                    "per-configuration report",
                ),
            ):
                phase12.validate_phase12_payload(
                    root,
                    expected_campaign_id=CAMPAIGN_ID,
                )

            payload["speedup_calculated"] = False
            payload["prior_gate_check_ids"]["G3"] = ["wrong_check"]
            _write_bytes(first_report, phase12.json_bytes(payload))
            with (
                mock.patch.object(
                    phase12,
                    "_validate_worker_result",
                ),
                mock.patch.object(
                    phase12,
                    "_compact_g5_run",
                    side_effect=_compact_replay(report),
                ),
                self.assertRaisesRegex(
                    phase12.Phase12UnifiedAdmissionError,
                    "per-configuration report",
                ),
            ):
                phase12.validate_phase12_payload(
                    root,
                    expected_campaign_id=CAMPAIGN_ID,
                )

    def test_finalization_is_immutable_and_checksum_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            artifact_root = temporary / "artifacts" / "phase12"
            stage = (
                artifact_root
                / ".kvbench-staging"
                / f"{CAMPAIGN_ID}.0123456789abcdef.staging"
            )
            report = _materialize_payload(stage)
            try:
                with (
                    mock.patch.object(
                        phase12,
                        "PHASE12_ARTIFACT_ROOT",
                        artifact_root,
                    ),
                    mock.patch.object(
                        phase12,
                        "_validate_worker_result",
                    ),
                    mock.patch.object(
                        phase12,
                        "_compact_g5_run",
                        side_effect=_compact_replay(report),
                    ),
                ):
                    final, root_sha256, object_count = (
                        phase12._finalize_campaign_stage(
                            stage=stage,
                            campaign_id=CAMPAIGN_ID,
                        )
                    )
                self.assertEqual(final, artifact_root / CAMPAIGN_ID)
                self.assertRegex(root_sha256, r"\A[0-9a-f]{64}\Z")
                self.assertGreater(object_count, 267)
                self.assertEqual(
                    {
                        "manifest.json",
                        "artifact_inventory.json",
                        "checksums.sha256",
                        "COMPLETE",
                    },
                    {
                        name
                        for name in (
                            "manifest.json",
                            "artifact_inventory.json",
                            "checksums.sha256",
                            "COMPLETE",
                        )
                        if (final / name).is_file()
                    },
                )
                self.assertFalse(
                    any(
                        path.lstat().st_mode & WRITE_BITS
                        for path in (final, *final.rglob("*"))
                    )
                )
                self.assertFalse(
                    (
                        final
                        / "docs"
                        / "evidence"
                        / "phase12"
                        / "r2-publication.json"
                    ).exists()
                )

                child = final / report.runs[0].manifest_path
                child.chmod(0o644)
                child.write_bytes(child.read_bytes() + b"\n")
                with (
                    mock.patch.object(
                        phase12,
                        "PHASE12_ARTIFACT_ROOT",
                        artifact_root,
                    ),
                    self.assertRaises(ArtifactValidationError),
                ):
                    phase12.validate_phase12_campaign(final)
            finally:
                _make_tree_writable(temporary)


    def test_r2_tool_evidence_rejects_every_unexpected_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for operation in ("publish", "verify"):
                for location in ("top_level", "bucket_lock", "result"):
                    with self.subTest(
                        operation=operation,
                        location=location,
                    ):
                        payload = _r2_result(
                            operation=operation,
                            root_sha256="d" * 64,
                            object_count=7,
                        )
                        if location == "top_level":
                            payload["unexpected"] = False
                        elif location == "bucket_lock":
                            bucket_lock = payload["bucket_lock"]
                            assert isinstance(bucket_lock, dict)
                            bucket_lock["unexpected"] = False
                        else:
                            result = payload[operation]
                            assert isinstance(result, dict)
                            result["unexpected"] = False
                        path = temporary / f"{operation}-{location}.json"
                        _write_json(path, payload)
                        with self.assertRaises(
                            phase12.Phase12UnifiedAdmissionError
                        ):
                            phase12._validate_r2_tool_result(
                                path=path,
                                operation=operation,
                                root_sha256="d" * 64,
                                object_count=7,
                            )

    def test_copied_r2_evidence_rejects_configured_secret_value(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "publish.json"
            payload = _r2_result(
                operation="publish",
                root_sha256="e" * 64,
                object_count=3,
            )
            secret = "phase12-test-secret-value"
            publish = payload["publish"]
            assert isinstance(publish, dict)
            publish["published_at_utc"] = secret
            _write_json(path, payload)
            with (
                mock.patch.dict(
                    "os.environ",
                    {"AWS_ACCESS_KEY_ID": secret},
                    clear=True,
                ),
                self.assertRaises(
                    phase12.Phase12UnifiedAdmissionError
                ) as caught,
            ):
                phase12._secret_free_r2_evidence_bytes(path)
            self.assertNotIn(secret, str(caught.exception))

    def test_single_campaign_publication_closes_once_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            artifact_root = temporary / "artifacts" / "phase12"
            stage = (
                artifact_root
                / ".kvbench-staging"
                / f"{CAMPAIGN_ID}.0123456789abcdef.staging"
            )
            report = _materialize_payload(stage)
            try:
                with (
                    mock.patch.object(
                        phase12,
                        "REPOSITORY_ROOT",
                        temporary,
                    ),
                    mock.patch.object(
                        phase12,
                        "PHASE12_ARTIFACT_ROOT",
                        artifact_root,
                    ),
                    mock.patch.object(
                        phase12,
                        "_validate_current_entry_g1_g4",
                    ),
                    mock.patch.object(
                        phase12,
                        "_validate_worker_result",
                    ),
                    mock.patch.object(
                        phase12,
                        "_compact_g5_run",
                        side_effect=_compact_replay(report),
                    ),
                ):
                    campaign, root_sha256, object_count = (
                        phase12._finalize_campaign_stage(
                            stage=stage,
                            campaign_id=CAMPAIGN_ID,
                        )
                    )
                    publish_path = temporary / "publish.json"
                    verify_path = temporary / "verify.json"
                    _write_json(
                        publish_path,
                        _r2_result(
                            operation="publish",
                            root_sha256=root_sha256,
                            object_count=object_count,
                        ),
                    )
                    _write_json(
                        verify_path,
                        _r2_result(
                            operation="verify",
                            root_sha256=root_sha256,
                            object_count=object_count,
                        ),
                    )
                    receipt = (
                        temporary
                        / "docs"
                        / "evidence"
                        / "phase12"
                        / "r2-publication.json"
                    )
                    final_report = receipt.with_name(
                        "unified-admission.json"
                    )
                    markdown = (
                        temporary
                        / "docs"
                        / "phase_reports"
                        / "phase12-unified-admission.md"
                    )
                    result = phase12.close_phase12_campaign_publication(
                        artifact=campaign,
                        publish_result_path=publish_path,
                        verify_result_path=verify_path,
                        receipt_output=receipt,
                        report_output=final_report,
                        markdown_output=markdown,
                    )
                    self.assertEqual(result["status"], "PASS")
                    payload = json.loads(receipt.read_text(encoding="utf-8"))
                    self.assertEqual(payload["publication_attempt_count"], 1)
                    self.assertEqual(payload["clean_retrieval_count"], 1)
                    replay = (
                        phase12.validate_phase12_campaign_final_evidence(
                            artifact=campaign,
                            receipt=receipt,
                            report=final_report,
                            markdown=markdown,
                        )
                    )
                    self.assertEqual(replay, result)
            finally:
                _make_tree_writable(temporary)



if __name__ == "__main__":
    unittest.main()

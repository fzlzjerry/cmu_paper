"""Focused Phase 12E tests for KIVI historical source validation."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from kvbench.runtime.kivi_admission import (
    KIVIAdmissionError,
    PHASE8_ADAPTER_PATH,
    PHASE8_DECISION_0026_ENDPOINT_COMMIT,
    PHASE8_DECISION_0026_ENDPOINT_SHA256,
    PHASE8_ENDPOINT_PATH,
    PHASE8_EXECUTION_GIT_SHA,
    PHASE8_HISTORICAL_ADAPTER_SHA256,
    PHASE8_HISTORICAL_CACHE_SHA256,
    PHASE8_HISTORICAL_ENDPOINT_SHA256,
    PHASE13B_DECISION_0030_PATH,
    PHASE13B_DECISION_0030_SHA256,
    PHASE13B_KIVI_REPORT_PATH,
    PHASE13B_KIVI_REPORT_SHA256,
    _phase8_git_path_history,
    _parse_publication_receipt,
    resolve_phase8_historical_source_authority,
)
from kvbench.runtime.kivi_allocation import (
    KIVIAllocationBinding,
    KIVIAllocationError,
    replay_preserved_kivi_allocation_attribution,
)
from scripts.phase8_r2_outer_bundle import validate_outer_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHASE8_INNER = (
    REPOSITORY_ROOT
    / "artifacts"
    / "phase8"
    / "phase8-20260727t113020276z-462325e9-0edc5a-k4v4-fixed-l128-eager"
)
PHASE8_OUTER = (
    REPOSITORY_ROOT
    / "artifacts"
    / "phase8_r2_outer"
    / "phase8-r2-outer-20260727t123744540656675z-7ff9f36"
)
PHASE8_INNER_RECEIPT = (
    REPOSITORY_ROOT
    / "docs"
    / "evidence"
    / "phase8"
    / "r2-admission-publication.json"
)
PHASE8_OUTER_ROOT = (
    "de7d41f151af9fe1e716f27ae0f1fc24d2ef0a4b16e8e5c3ecf45d5f9983e132"
)


def _resolve():
    return resolve_phase8_historical_source_authority(
        repository_root=REPOSITORY_ROOT,
        execution_git_sha=PHASE8_EXECUTION_GIT_SHA,
        manifest_adapter_sha256=PHASE8_HISTORICAL_ADAPTER_SHA256,
    )


def _require_local_phase8_evidence(*roots: Path) -> None:
    if all(root.is_dir() for root in roots):
        return
    message = "local immutable Phase 8 bundle unavailable"
    if os.environ.get("KVBENCH_PHASE12E_REQUIRE_LOCAL_EVIDENCE") == "1":
        raise AssertionError(message)
    raise unittest.SkipTest(message)


class Phase12EKIVIHistoricalAuthorityTests(unittest.TestCase):
    def test_full_history_exposes_discarded_side_branch_change(self) -> None:
        environment = {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LC_ALL": "C",
            "LANG": "C",
            "PATH": "/usr/bin:/bin",
        }

        with tempfile.TemporaryDirectory(
            prefix="kvbench-phase12e-history-"
        ) as temporary:
            repository = Path(temporary)

            def git(*arguments: str, input_bytes: bytes = b"") -> str:
                result = subprocess.run(
                    ("/usr/bin/git", *arguments),
                    cwd=repository,
                    env=environment,
                    input=input_bytes,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    shell=False,
                    timeout=10,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    result.stderr.decode("utf-8", errors="replace"),
                )
                return result.stdout.decode("ascii").strip()

            git("init", "--quiet")
            blob_a = git("hash-object", "-w", "--stdin", input_bytes=b"A\n")
            blob_b = git("hash-object", "-w", "--stdin", input_bytes=b"B\n")
            tree_a = git(
                "mktree",
                input_bytes=f"100644 blob {blob_a}\tf\n".encode("ascii"),
            )
            tree_b = git(
                "mktree",
                input_bytes=f"100644 blob {blob_b}\tf\n".encode("ascii"),
            )
            identity = ("-c", "user.name=x", "-c", "user.email=x@y")
            base = git(
                *identity,
                "commit-tree",
                tree_a,
                input_bytes=b"base\n",
            )
            main = git(
                *identity,
                "commit-tree",
                tree_a,
                "-p",
                base,
                input_bytes=b"main\n",
            )
            side = git(
                *identity,
                "commit-tree",
                tree_b,
                "-p",
                base,
                input_bytes=b"side changes f\n",
            )
            merge = git(
                *identity,
                "commit-tree",
                tree_a,
                "-p",
                main,
                "-p",
                side,
                input_bytes=b"ours merge\n",
            )

            default_history = git(
                "rev-list",
                "--reverse",
                f"{base}..{merge}",
                "--",
                "f",
            )
            full_history = _phase8_git_path_history(
                repository,
                start_commit=base,
                end_commit=merge,
                relative_paths=("f",),
            )

        self.assertEqual(default_history, "")
        self.assertEqual(full_history.splitlines(), [side, merge])

    def test_genuine_phase8_outer_bundle_replays_and_passes(self) -> None:
        _require_local_phase8_evidence(PHASE8_INNER, PHASE8_OUTER)
        validation = validate_outer_bundle(
            PHASE8_OUTER,
            repository_root=REPOSITORY_ROOT,
            source_bundle=PHASE8_INNER,
        )
        self.assertEqual(validation.root_sha256, PHASE8_OUTER_ROOT)
        self.assertEqual(validation.admission_run_count, 10)

    def test_exact_execution_blobs_and_transition_are_bound(self) -> None:
        authority = _resolve()
        self.assertEqual(
            authority.execution_git_sha,
            PHASE8_EXECUTION_GIT_SHA,
        )
        self.assertEqual(
            authority.adapter_source_sha256,
            PHASE8_HISTORICAL_ADAPTER_SHA256,
        )
        self.assertEqual(
            authority.cache_source_sha256,
            PHASE8_HISTORICAL_CACHE_SHA256,
        )
        self.assertEqual(
            authority.endpoint_source_sha256,
            PHASE8_HISTORICAL_ENDPOINT_SHA256,
        )
        self.assertEqual(
            authority.endpoint_transition_commit,
            PHASE8_DECISION_0026_ENDPOINT_COMMIT,
        )

    def test_phase13b_successor_decision_and_report_are_checksum_bound(
        self,
    ) -> None:
        from kvbench.runtime import kivi_admission

        self.assertEqual(
            kivi_admission.sha256_file(
                REPOSITORY_ROOT / PHASE13B_DECISION_0030_PATH
            ),
            PHASE13B_DECISION_0030_SHA256,
        )
        self.assertEqual(
            kivi_admission.sha256_file(
                REPOSITORY_ROOT / PHASE13B_KIVI_REPORT_PATH
            ),
            PHASE13B_KIVI_REPORT_SHA256,
        )
        original = kivi_admission.sha256_file
        for relative_path, message in (
            (PHASE13B_DECISION_0030_PATH, "Decision 0030 checksum differs"),
            (
                PHASE13B_KIVI_REPORT_PATH,
                "successor report checksum differs",
            ),
        ):
            with self.subTest(relative_path=relative_path):

                def tampered(
                    path: Path,
                    *,
                    target: Path = REPOSITORY_ROOT / relative_path,
                ) -> str:
                    if path == target:
                        return "0" * 64
                    return original(path)

                with mock.patch(
                    "kvbench.runtime.kivi_admission.sha256_file",
                    side_effect=tampered,
                ), self.assertRaisesRegex(KIVIAdmissionError, message):
                    _resolve()

    def test_missing_or_tampered_execution_commit_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            KIVIAdmissionError,
            "execution commit differs",
        ):
            resolve_phase8_historical_source_authority(
                repository_root=REPOSITORY_ROOT,
                execution_git_sha="0" * 40,
                manifest_adapter_sha256=PHASE8_HISTORICAL_ADAPTER_SHA256,
            )
        with self.assertRaisesRegex(
            KIVIAdmissionError,
            "execution commit differs",
        ):
            resolve_phase8_historical_source_authority(
                repository_root=REPOSITORY_ROOT,
                execution_git_sha=(
                    "7ff9f36055eb71d581d5849822487f6589cdc6e8"
                ),
                manifest_adapter_sha256=PHASE8_HISTORICAL_ADAPTER_SHA256,
            )

    def test_endpoint_or_transition_tampering_fails_closed(self) -> None:
        from kvbench.runtime import kivi_admission

        original_blob = kivi_admission._phase8_git_blob_sha256

        def tampered_blob(
            repository_root: Path,
            *,
            revision: str,
            relative_path: str,
        ) -> str:
            if (
                revision == PHASE8_EXECUTION_GIT_SHA
                and relative_path == PHASE8_ENDPOINT_PATH
            ):
                return "0" * 64
            return original_blob(
                repository_root,
                revision=revision,
                relative_path=relative_path,
            )

        with mock.patch(
            "kvbench.runtime.kivi_admission._phase8_git_blob_sha256",
            side_effect=tampered_blob,
        ), self.assertRaisesRegex(
            KIVIAdmissionError,
            "endpoint blobs do not match",
        ):
            _resolve()

        original_git = kivi_admission._phase8_git

        def extra_transition(
            repository_root: Path,
            *arguments: str,
            binary: bool = False,
        ):
            if arguments and arguments[0] == "rev-list":
                return (
                    f"{PHASE8_DECISION_0026_ENDPOINT_COMMIT}\n"
                    f"{'f' * 40}"
                )
            return original_git(
                repository_root,
                *arguments,
                binary=binary,
            )

        with mock.patch(
            "kvbench.runtime.kivi_admission._phase8_git",
            side_effect=extra_transition,
        ), self.assertRaisesRegex(
            KIVIAdmissionError,
            "exact Decision 0026 commit",
        ):
            _resolve()

    def test_adapter_or_cache_authority_tampering_fails_closed(self) -> None:
        from kvbench.runtime import kivi_admission

        with self.assertRaisesRegex(
            KIVIAdmissionError,
            "adapter authority changed",
        ):
            resolve_phase8_historical_source_authority(
                repository_root=REPOSITORY_ROOT,
                execution_git_sha=PHASE8_EXECUTION_GIT_SHA,
                manifest_adapter_sha256="0" * 64,
            )

        original = kivi_admission._phase8_regular_source_sha256
        for relative, message in (
            ("src/kvbench/adapters/kivi.py", "adapter authority changed"),
            ("src/kvbench/runtime/kivi_cache.py", "cache authority changed"),
        ):
            with self.subTest(relative=relative):

                def tampered(
                    repository_root: Path,
                    relative_path: str,
                    *,
                    target: str = relative,
                ) -> str:
                    if relative_path == target:
                        return "0" * 64
                    return original(repository_root, relative_path)

                with mock.patch(
                    "kvbench.runtime.kivi_admission."
                    "_phase8_regular_source_sha256",
                    side_effect=tampered,
                ), self.assertRaisesRegex(KIVIAdmissionError, message):
                    _resolve()

        original_git = kivi_admission._phase8_git

        def changed_then_reverted(
            repository_root: Path,
            *arguments: str,
            binary: bool = False,
        ):
            if (
                arguments
                and arguments[0] == "rev-list"
                and PHASE8_ADAPTER_PATH in arguments
            ):
                return "f" * 40
            return original_git(
                repository_root,
                *arguments,
                binary=binary,
            )

        with mock.patch(
            "kvbench.runtime.kivi_admission._phase8_git",
            side_effect=changed_then_reverted,
        ), self.assertRaisesRegex(
            KIVIAdmissionError,
            "adapter or cache changed",
        ):
            _resolve()

    def test_raw_allocation_binding_and_hash_tampering_fails(self) -> None:
        _require_local_phase8_evidence(PHASE8_INNER)
        operation = (
            PHASE8_INNER
            / "allocation"
            / "operations"
            / "step-0000"
        )
        point = json.loads(
            (PHASE8_INNER / "validation" / "point.json").read_text(
                encoding="utf-8"
            )
        )
        envelope = point["allocation"]["operation_allocations"][0]
        audit = json.loads(
            (operation / "allocation_audit.json").read_text(encoding="utf-8")
        )
        binding = KIVIAllocationBinding.from_mapping(audit["binding"])
        replay = replay_preserved_kivi_allocation_attribution(
            operation,
            raw_files=envelope["raw_files"],
            expected_binding=binding,
        )
        self.assertTrue(replay.summary["criterion"]["passed"])

        with self.assertRaisesRegex(
            KIVIAllocationError,
            "binding differs",
        ):
            replay_preserved_kivi_allocation_attribution(
                operation,
                raw_files=envelope["raw_files"],
                expected_binding=dataclasses.replace(
                    binding,
                    endpoint_source_sha256="0" * 64,
                ),
            )
        with self.assertRaisesRegex(
            KIVIAllocationError,
            "binding differs",
        ):
            replay_preserved_kivi_allocation_attribution(
                operation,
                raw_files=envelope["raw_files"],
                expected_binding=dataclasses.replace(
                    binding,
                    cache_source_sha256="0" * 64,
                ),
            )
        tampered_files = dict(envelope["raw_files"])
        tampered_files["trace_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            KIVIAllocationError,
            "checksum verification",
        ):
            replay_preserved_kivi_allocation_attribution(
                operation,
                raw_files=tampered_files,
                expected_binding=binding,
            )

    def test_receipt_tampering_fails_closed(self) -> None:
        receipt = json.loads(
            PHASE8_INNER_RECEIPT.read_text(encoding="utf-8")
        )
        receipt["source_git_sha"] = "0" * 40
        with tempfile.TemporaryDirectory(
            prefix=".phase12e-receipt-",
            dir=REPOSITORY_ROOT,
        ) as temporary:
            receipt_path = Path(temporary) / "receipt.json"
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KIVIAdmissionError,
                "receipt does not bind",
            ):
                _parse_publication_receipt(
                    receipt_path=receipt_path,
                    evidence_root=REPOSITORY_ROOT,
                    inner_root_digest=receipt["local_validation"][
                        "root_sha256"
                    ],
                    inner_object_count=receipt["local_validation"][
                        "object_count"
                    ],
                    source_run_id=receipt["source_run_id"],
                    source_git_sha=PHASE8_EXECUTION_GIT_SHA,
                )

    def test_current_kivi_method_admission_remains_pass(self) -> None:
        report = json.loads(
            (
                REPOSITORY_ROOT
                / "docs"
                / "evidence"
                / "phase8"
                / "kivi-method-admission.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["gates"]["g2_kivi"], "PASS")
        self.assertEqual(
            report["adapter_source_sha256"],
            PHASE8_HISTORICAL_ADAPTER_SHA256,
        )
        self.assertEqual(
            _resolve().endpoint_source_sha256,
            PHASE8_HISTORICAL_ENDPOINT_SHA256,
        )
        self.assertEqual(
            PHASE8_DECISION_0026_ENDPOINT_SHA256,
            "9095e9a2a9c01e1ea6afb2f1cefcee46a964a82caae7b819a125757b59244a9b",
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import runpy
import shutil
from pathlib import Path
import tempfile
import unittest

from scripts import phase11_kvquant_admission as admission


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
Q23_EVIDENCE = (
    REPOSITORY_ROOT
    / "artifacts/phase11/phase11dq23-launch.ssAO5M"
)


class Phase11RQ23AdmissionDriverTests(unittest.TestCase):
    def tearDown(self) -> None:
        admission._activate_authority_profile("decision0027")

    def test_q23_profile_binds_exact_current_authority(self) -> None:
        admission._activate_authority_profile("decision0029")
        authority = admission._authority()
        self.assertEqual(
            authority.execution_source_identifier,
            "kvquant_gqa_longctx_deterministic_q23_v4",
        )
        self.assertEqual(
            authority.corrected_commit,
            "34b0bdfa83082e1f30387d9ac5cca369006e089c",
        )
        self.assertEqual(
            authority.corrected_tree,
            "1f85af65fe03061583ffe8bd91e47d7ecffdd312",
        )
        self.assertEqual(
            authority.extension_sha256,
            "b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d",
        )
        self.assertEqual(authority.decisions[-1], "0029")
        self.assertIs(
            admission.Phase11RunManifest,
            admission.Phase11RQ23RunManifest,
        )
        self.assertTrue(
            admission._run_id("d" * 40).startswith("phase11rq23-")
        )

    def test_q23_execution_path_requires_all_deterministic_apis(self) -> None:
        admission._activate_authority_profile("decision0029")
        records = admission._static_execution_path()
        self.assertEqual(
            tuple(record.configuration for record in records),
            ("kvq4", "kvq3", "kvq2"),
        )
        self.assertTrue(all(record.direct_compressed_decode for record in records))
        self.assertTrue(all(record.native_gqa for record in records))
        self.assertTrue(all(record.no_dynamic_sparse_allocation for record in records))
        self.assertTrue(all(record.no_host_synchronization for record in records))
        self.assertTrue(all(record.no_backend_fallback for record in records))

    def test_sanitizer_probe_binds_current_q23_authority(self) -> None:
        namespace = runpy.run_path(
            str(
                REPOSITORY_ROOT
                / "tests/cuda/phase11_kvquant_sanitizer_probe.py"
            )
        )
        namespace["_require_exact_authority"]()
        self.assertEqual(
            namespace["_AUTHORITY"]["aggregate_patch_sha256"],
            "7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a",
        )
        self.assertEqual(
            namespace["_AUTHORITY"]["extension_sha256"],
            "b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d",
        )

    def test_q23_evidence_replays_and_tamper_fails(self) -> None:
        admission._activate_authority_profile("decision0029")
        binding = admission._validate_q23_evidence_bundle(
            Q23_EVIDENCE.resolve(strict=True)
        )
        self.assertEqual(binding["object_count"], 46)
        self.assertEqual(
            binding["evidence_root_sha256"],
            "8b65112ea2d49b58ee07c1533b429fac1a8af7466e09adad073d9a22ae2ec790",
        )
        self.assertTrue(binding["sanitizer_memcheck_passed"])
        self.assertTrue(binding["sanitizer_initcheck_passed"])
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "q23"
            shutil.copytree(Q23_EVIDENCE, copy)
            path = copy / "checks/source.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaises(admission.Phase11KVQuantDriverError):
                admission._validate_q23_evidence_bundle(copy)

    def test_q23_report_references_close_q23_evidence(self) -> None:
        admission._activate_authority_profile("decision0029")
        evidence_ids = dict(admission._active_report_evidence_paths())
        self.assertEqual(
            evidence_ids["q23_binding"],
            "config/q23-validation-binding.json",
        )
        self.assertEqual(
            evidence_ids["q23_complete"],
            "authority/q23-evidence/COMPLETE",
        )
        checks = admission._active_check_evidence()
        self.assertIn("q23_summary", checks["compute_sanitizer"])
        self.assertIn("q23_binding", checks["execution_path"])
        self.assertIn("q23_checksum_ledger", checks["immutable_checksums"])

    def test_legacy_profile_remains_default_and_restorable(self) -> None:
        admission._activate_authority_profile("decision0029")
        admission._activate_authority_profile("decision0027")
        self.assertEqual(
            admission._authority().corrected_commit,
            "4b8533b29b04f8c4bf55f688a41fefe20487637b",
        )
        self.assertIs(
            admission.Phase11RunManifest,
            admission._LEGACY_RUN_MANIFEST_CLASS,
        )
        self.assertTrue(admission._run_id("d" * 40).startswith("phase11-"))


if __name__ == "__main__":
    unittest.main()

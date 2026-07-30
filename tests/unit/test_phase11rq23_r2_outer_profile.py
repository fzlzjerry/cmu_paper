from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from kvbench.schema import phase11 as phase11_schema
from scripts import phase11_r2_outer_bundle as outer


class Phase11RQ23OuterProfileTests(unittest.TestCase):
    def tearDown(self) -> None:
        outer._activate_profile(outer.AUTHORITY_PROFILE_DECISION0027)

    def test_default_profile_preserves_decision0027_exactly(self) -> None:
        outer._activate_profile(outer.AUTHORITY_PROFILE_DECISION0027)

        self.assertEqual(
            outer.AUTHORITY_PROFILE,
            outer.AUTHORITY_PROFILE_DECISION0027,
        )
        self.assertIs(
            outer.Phase11RunManifest,
            phase11_schema.Phase11RunManifest,
        )
        self.assertIs(
            outer.Phase11MethodAdmissionReport,
            phase11_schema.Phase11MethodAdmissionReport,
        )
        self.assertEqual(
            outer.PHASE11_EXECUTION_SOURCE_IDENTIFIER,
            phase11_schema.PHASE11_EXECUTION_SOURCE_IDENTIFIER,
        )
        self.assertEqual(
            outer.INNER_RECEIPT_RELATIVE,
            Path("docs/evidence/phase11/r2-admission-publication.json"),
        )
        self.assertEqual(
            outer.OUTER_RECEIPT_RELATIVE,
            Path(
                "docs/evidence/phase11/"
                "r2-admission-outer-publication.json"
            ),
        )
        self.assertEqual(
            outer.METHOD_ADMISSION_RELATIVE,
            Path("docs/evidence/phase11/kvquant-method-admission.json"),
        )
        self.assertEqual(
            outer.PASS_REPORT_RELATIVE,
            Path(
                "docs/phase_reports/"
                "phase11r-kvquant-measurement-adapter.md"
            ),
        )
        self.assertEqual(outer.PASS_REPORT_HEADING, "PHASE 11 REPORT")
        self.assertEqual(
            outer.MANIFEST_SCHEMA,
            "kvbench-phase11-r2-outer-bundle-1.0.0",
        )
        self.assertEqual(
            outer.INNER_RECEIPT_SCHEMA,
            (
                "kvbench-phase11-kvquant-admission-"
                "r2-publication-1.0.0"
            ),
        )

    def test_decision0029_profile_is_exact_and_separate(self) -> None:
        outer._activate_profile(outer.AUTHORITY_PROFILE_DECISION0029)

        self.assertEqual(
            outer.AUTHORITY_PROFILE,
            outer.AUTHORITY_PROFILE_DECISION0029,
        )
        self.assertIs(
            outer.Phase11RunManifest,
            phase11_schema.Phase11RQ23RunManifest,
        )
        self.assertIs(
            outer.Phase11MethodAdmissionReport,
            phase11_schema.Phase11RQ23MethodAdmissionReport,
        )
        self.assertNotEqual(
            outer.Phase11RunManifest.SCHEMA_VERSION,
            phase11_schema.Phase11RunManifest.SCHEMA_VERSION,
        )
        self.assertNotEqual(
            outer.Phase11MethodAdmissionReport.SCHEMA_VERSION,
            phase11_schema.Phase11MethodAdmissionReport.SCHEMA_VERSION,
        )
        self.assertEqual(
            outer.PHASE11_EXECUTION_SOURCE_IDENTIFIER,
            phase11_schema.PHASE11Q23_EXECUTION_SOURCE_IDENTIFIER,
        )
        self.assertEqual(
            outer.PHASE11_AGGREGATE_PATCH_SHA256,
            phase11_schema.PHASE11Q23_AGGREGATE_PATCH_SHA256,
        )
        self.assertEqual(
            outer.PHASE11_CORRECTED_COMMIT,
            phase11_schema.PHASE11Q23_CORRECTED_COMMIT,
        )
        self.assertEqual(
            outer.PHASE11_CORRECTED_TREE,
            phase11_schema.PHASE11Q23_CORRECTED_TREE,
        )
        self.assertEqual(
            outer.PHASE11_EXTENSION_SHA256,
            phase11_schema.PHASE11Q23_EXTENSION_SHA256,
        )
        self.assertEqual(
            outer.PHASE11_DECISIONS,
            phase11_schema.PHASE11Q23_DECISIONS,
        )
        self.assertEqual(
            outer.INNER_RECEIPT_RELATIVE,
            Path(
                "docs/evidence/phase11rq23/"
                "r2-admission-publication.json"
            ),
        )
        self.assertEqual(
            outer.OUTER_RECEIPT_RELATIVE,
            Path(
                "docs/evidence/phase11rq23/"
                "r2-admission-outer-publication.json"
            ),
        )
        self.assertEqual(
            outer.METHOD_ADMISSION_RELATIVE,
            Path(
                "docs/evidence/phase11rq23/"
                "kvquant-method-admission.json"
            ),
        )
        self.assertEqual(
            outer.PASS_REPORT_RELATIVE,
            Path(
                "docs/phase_reports/"
                "phase11rq23-kvquant-measurement-adapter.md"
            ),
        )
        self.assertEqual(
            outer.BUNDLED_PASS_REPORT_PATH.as_posix(),
            "reports/phase11rq23-kvquant-measurement-adapter.md",
        )
        self.assertEqual(
            outer.PASS_REPORT_HEADING,
            "PHASE 11R-Q23 REPORT",
        )
        self.assertEqual(
            outer.MANIFEST_SCHEMA,
            "kvbench-phase11rq23-r2-outer-bundle-1.0.0",
        )
        self.assertEqual(
            outer.INNER_RECEIPT_SCHEMA,
            (
                "kvbench-phase11rq23-kvquant-admission-"
                "r2-publication-1.0.0"
            ),
        )
        self.assertEqual(
            outer.OUTER_RECEIPT_SCHEMA,
            (
                "kvbench-phase11rq23-kvquant-admission-"
                "r2-outer-publication-1.0.0"
            ),
        )
        self.assertEqual(
            outer.OUTER_ARTIFACT_ROOT_RELATIVE,
            Path("artifacts/phase11_r2_outer"),
        )

    def test_profile_switch_restores_historical_namespace(self) -> None:
        outer._activate_profile(outer.AUTHORITY_PROFILE_DECISION0029)
        outer._activate_profile(outer.AUTHORITY_PROFILE_DECISION0027)

        self.assertEqual(
            outer.REQUIRED_REPOSITORY_FILES,
            (
                Path(
                    "docs/evidence/phase11/"
                    "r2-admission-publication.json"
                ),
                Path(
                    "docs/evidence/phase11/"
                    "r2-admission-publish.stdout.json"
                ),
                Path(
                    "docs/evidence/phase11/"
                    "r2-admission-publish.stderr.txt"
                ),
                Path(
                    "docs/evidence/phase11/"
                    "r2-admission-verify.stdout.json"
                ),
                Path(
                    "docs/evidence/phase11/"
                    "r2-admission-verify.stderr.txt"
                ),
                Path(
                    "docs/evidence/phase11/"
                    "kvquant-method-admission.json"
                ),
                Path(
                    "docs/evidence/phase11/"
                    "kvquant-method-admission.sha256"
                ),
                Path(
                    "docs/phase_reports/"
                    "phase11r-kvquant-measurement-adapter.md"
                ),
            ),
        )

    def test_parser_defaults_historical_and_accepts_only_exact_profiles(
        self,
    ) -> None:
        parser = outer._parser()
        historical = parser.parse_args(
            ["validate", "artifact", "--source-bundle", "inner"]
        )
        successor = parser.parse_args(
            [
                "--authority-profile",
                "decision0029",
                "validate",
                "artifact",
                "--source-bundle",
                "inner",
            ]
        )

        self.assertEqual(
            historical.authority_profile,
            outer.AUTHORITY_PROFILE_DECISION0027,
        )
        self.assertEqual(
            successor.authority_profile,
            outer.AUTHORITY_PROFILE_DECISION0029,
        )
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--authority-profile",
                        "decision0028",
                        "validate",
                        "artifact",
                        "--source-bundle",
                        "inner",
                    ]
                )
        with self.assertRaises(outer.Phase11OuterBundleError):
            outer._activate_profile("decision0028")

    def test_q23_report_heading_and_authority_fail_closed(self) -> None:
        outer._activate_profile(outer.AUTHORITY_PROFILE_DECISION0029)
        report_sha256 = "a" * 64
        report = SimpleNamespace(
            bounded_runs=tuple(
                SimpleNamespace(run_id=f"phase11rq23-point-{index}")
                for index in range(9)
            ),
            r2_uri=(
                "r2://kvbench-artifacts/kvbench/sha256/"
                f"{'b' * 64}/"
            ),
        )
        valid = self._report_text(
            report=report,
            report_sha256=report_sha256,
        )

        self._validate_report_text(
            text=valid,
            report=report,
            report_sha256=report_sha256,
        )
        for drifted in (
            valid.replace("PHASE 11R-Q23 REPORT", "PHASE 11 REPORT", 1),
            valid.replace(
                outer.PHASE11_EXECUTION_SOURCE_IDENTIFIER,
                phase11_schema.PHASE11_EXECUTION_SOURCE_IDENTIFIER,
                1,
            ),
            valid.replace(
                outer.PHASE11_AGGREGATE_PATCH_SHA256,
                phase11_schema.PHASE11_AGGREGATE_PATCH_SHA256,
                1,
            ),
            valid.replace(
                ", ".join(outer.PHASE11_DECISIONS),
                ", ".join(phase11_schema.PHASE11_DECISIONS),
                1,
            ),
        ):
            with self.subTest(drifted=drifted[:80]):
                with self.assertRaises(outer.Phase11OuterBundleError):
                    self._validate_report_text(
                        text=drifted,
                        report=report,
                        report_sha256=report_sha256,
                    )
        outer._activate_profile(outer.AUTHORITY_PROFILE_DECISION0027)
        with self.assertRaises(outer.Phase11OuterBundleError):
            self._validate_report_text(
                text=valid,
                report=report,
                report_sha256=report_sha256,
            )

    @staticmethod
    def _report_text(
        *,
        report: SimpleNamespace,
        report_sha256: str,
    ) -> str:
        states = {
            "Status": "PASS",
            "Working tree": "CLEAN",
            "Algorithm identifier": outer.PHASE11_METHOD_IDENTIFIER,
            "Execution-source identifier": (
                outer.PHASE11_EXECUTION_SOURCE_IDENTIFIER
            ),
            "Decisions": ", ".join(outer.PHASE11_DECISIONS),
            "Aggregate patch SHA": outer.PHASE11_AGGREGATE_PATCH_SHA256,
            "Corrected commit": outer.PHASE11_CORRECTED_COMMIT,
            "Corrected tree": outer.PHASE11_CORRECTED_TREE,
            "Extension SHA": outer.PHASE11_EXTENSION_SHA256,
            "Calibration ID": outer.PHASE11_CALIBRATION_ID,
            "Calibration root": outer.PHASE11_CALIBRATION_ROOT,
            "Historical Phase 10 root": (
                outer.PHASE11_HISTORICAL_FIXTURE_ROOT
            ),
            "Corrected fixture ID": outer.PHASE11_FIXTURE_ID,
            "Corrected fixture root": outer.PHASE11_FIXTURE_ROOT,
            "Adapter location": "src/kvbench/adapters/kvquant.py",
            "Supported configurations": "kvq4, kvq3, kvq2",
            "Boundary semantics": (
                "PRE-ROPE KEY QUANTIZATION; ATTENTION-READY SINK KEY"
            ),
            "Static cache": "PASS",
            "Fixture conformance": "9/9 PASS",
            "Execution-path and GQA audit": "PASS",
            "Eager allocation": "PASS",
            "CUDA Graph": "PASS",
            "Sanitizer": "PASS",
            "Bounded admission": "9/9 PASS",
            "Admission run IDs": ", ".join(
                point.run_id for point in report.bounded_runs
            ),
            "MethodAdmissionReport SHA-256": report_sha256,
            "Inner R2 URI": report.r2_uri,
            "G2-KVQ": "PASS",
            "Global G2": "NOT EVALUATED",
            "G3": "NOT EVALUATED",
            "G4": "NOT EVALUATED",
            "G5": "NOT EVALUATED",
            "Full Scan": "CLOSED",
            "Quality execution": "LOCKED",
            "PERFORMANCE_DATA_FROZEN": "ABSENT",
            "Performance claim eligible": "FALSE",
            "Speedup calculated": "NO",
            "r_hbm": "NULL",
            "Historical evidence changed": "NO",
            "Existing methods changed": "NO",
            "Measurement Container changed": "NO",
            "Phase 12 started": "NO",
        }
        return "\n".join(
            [
                f"# {outer.PASS_REPORT_HEADING}",
                "",
                *(f"- {key}: {value}" for key, value in states.items()),
                "",
            ]
        )

    @staticmethod
    def _validate_report_text(
        *,
        text: str,
        report: SimpleNamespace,
        report_sha256: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.md"
            path.write_text(text, encoding="utf-8")
            outer._validate_phase_report(
                path,
                report=report,
                report_sha256=report_sha256,
            )


if __name__ == "__main__":
    unittest.main()

"""Focused static tests for the Phase 8 exact-container Make targets."""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
AUTHORIZED_MEASUREMENT_CONFIG = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
KIVI_REFERENCE_MANIFEST = (
    "sha256:f27e4cdef6bd15f18ab76b1fe0e4413ede004b42538c74e3dd90d04172406f75"
)
KIVI_REFERENCE_CONFIG = (
    "sha256:0915dc8488fd6c9a150a3b4f56bb4b97b5dbdb7c51d96cda2d431df20e856ce3"
)
KIVI_EXTENSION_SHA256 = (
    "45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9"
)
KIVI_NEW_PACK_SHA256 = (
    "3678af0e34a0ba18e5d80a4128acf11d4070667c800a15540a16d07253a4f75e"
)
KIVI_ADMISSION_RUN_ID = (
    "phase8-20260727t113020276z-462325e9-0edc5a-k4v4-fixed-l128-eager"
)
KIVI_ADMISSION_ARTIFACT = (
    f"$(CURDIR)/artifacts/phase8/{KIVI_ADMISSION_RUN_ID}"
)


def _assignment(name: str) -> str:
    match = re.search(
        rf"^(?:override )?{re.escape(name)}\s*:?=\s*(\S+)\s*$",
        MAKEFILE,
        flags=re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"missing Make assignment: {name}")
    return match.group(1)


def _recipe(target: str) -> str:
    lines = MAKEFILE.splitlines()
    prefix = f"{target}:"
    for index, line in enumerate(lines):
        if (
            line.startswith(prefix)
            and index + 1 < len(lines)
            and lines[index + 1].startswith("\t")
        ):
            body: list[str] = []
            for candidate in lines[index + 1 :]:
                if not candidate.startswith("\t"):
                    break
                body.append(candidate)
            return "\n".join(body)
    raise AssertionError(f"missing Make recipe: {target}")


class Phase8MakeTargetTests(unittest.TestCase):
    def test_phase8_authorities_are_exact_and_not_floating(self) -> None:
        self.assertEqual(
            _assignment("PHASE8_AUTHORIZED_IMAGE_CONFIG_DIGEST"),
            AUTHORIZED_MEASUREMENT_CONFIG,
        )
        self.assertEqual(
            _assignment("PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST"),
            KIVI_REFERENCE_MANIFEST,
        )
        self.assertEqual(
            _assignment("PHASE8_KIVI_REFERENCE_CONFIG_DIGEST"),
            KIVI_REFERENCE_CONFIG,
        )
        self.assertEqual(
            _assignment("PHASE8_KIVI_EXTENSION_SHA256"),
            KIVI_EXTENSION_SHA256,
        )
        self.assertEqual(
            _assignment("PHASE8_KIVI_NEW_PACK_SHA256"),
            KIVI_NEW_PACK_SHA256,
        )
        for name in (
            "PHASE8_AUTHORIZED_IMAGE_CONFIG_DIGEST",
            "PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST",
            "PHASE8_KIVI_REFERENCE_CONFIG_DIGEST",
            "PHASE8_KIVI_EXTENSION_SHA256",
            "PHASE8_KIVI_NEW_PACK_SHA256",
        ):
            self.assertIsNotNone(
                re.search(
                    rf"^override {name} :=",
                    MAKEFILE,
                    flags=re.MULTILINE,
                )
            )

    def test_admission_uses_a_clean_detached_clone_and_exact_images(self) -> None:
        recipe = _recipe("admit-kivi")
        self.assertIn(
            "admit-kivi: override MEASUREMENT_IMAGE_CONFIG_DIGEST := "
            "$(PHASE8_AUTHORIZED_IMAGE_CONFIG_DIGEST)",
            MAKEFILE,
        )
        self.assertIn("admit-kivi: verify-measurement-container", MAKEFILE)
        self.assertIn(
            "git status --porcelain=v1 --untracked-files=all",
            recipe,
        )
        self.assertIn("git clone --quiet --no-local --no-checkout", recipe)
        self.assertIn("checkout --quiet --detach", recipe)
        self.assertIn("remote remove origin", recipe)
        self.assertIn(
            'chmod -R a-w "$$task_root/source/docs/evidence/e00"',
            recipe,
        )
        self.assertIn(
            'ln -s /opt/kvbench/.venv/bin "$$task_root/source/.venv/bin"',
            recipe,
        )
        self.assertIn(
            "ln -s /opt/kvbench/.phase3/site-packages "
            '"$$task_root/source/.phase3/site-packages"',
            recipe,
        )
        self.assertIn(
            'image_id="$(PHASE8_AUTHORIZED_IMAGE_CONFIG_DIGEST)"',
            recipe,
        )
        self.assertIn(
            '"$(KIVI_REFERENCE_IMAGE)@'
            '$(PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST)"',
            recipe,
        )
        self.assertIn(
            "$(PHASE8_KIVI_REFERENCE_CONFIG_DIGEST)",
            recipe,
        )
        self.assertIn(
            '= "$(PHASE8_KIVI_REFERENCE_MANIFEST_DIGEST)"',
            recipe,
        )
        self.assertNotIn("docker build", recipe)
        self.assertNotIn("docker run", recipe)
        self.assertNotIn("apt-get", recipe)
        self.assertNotIn("pip install", recipe)

    def test_reference_source_and_extension_are_extracted_and_verified(
        self,
    ) -> None:
        recipe = _recipe("admit-kivi")
        self.assertIn(
            'docker cp "$$reference_cid:/opt/kivi-source/."',
            recipe,
        )
        self.assertIn(
            "kivi_gemv.cpython-312-x86_64-linux-gnu.so",
            recipe,
        )
        self.assertIn(
            'git -C "$$task_root/kivi-source" clean -fdx',
            recipe,
        )
        self.assertIn("kivi-gqa.authority.py", recipe)
        self.assertIn(
            'scripts/validate_kivi_b019_patch.py',
            recipe,
        )
        self.assertIn(
            '--source-root "$$task_root/kivi-source"',
            recipe,
        )
        self.assertIn("$(PHASE8_KIVI_NEW_PACK_SHA256)", recipe)
        self.assertIn("$(PHASE8_KIVI_EXTENSION_SHA256)", recipe)
        self.assertIn(
            "src=$$task_root/kivi-source,dst=/opt/kivi-source,readonly",
            recipe,
        )
        self.assertIn(
            "src=$$task_root/kivi-extension/"
            "kivi_gemv.cpython-312-x86_64-linux-gnu.so,"
            "dst=/opt/kvbench/.phase3/site-packages/"
            "kivi_gemv.cpython-312-x86_64-linux-gnu.so,readonly",
            recipe,
        )

    def test_measurement_container_is_fail_closed_and_secret_free(
        self,
    ) -> None:
        recipe = _recipe("admit-kivi")
        for required in (
            "--read-only",
            "--network=none",
            '--gpus "device=$(MEASUREMENT_GPU_UUID)"',
            "src=$$task_root/source,"
            "dst=/home/rockrock/cmu_paper,readonly",
            "src=$$artifact_root,"
            "dst=/home/rockrock/cmu_paper/artifacts/phase8",
            "src=$$model_root,"
            "dst=/root/.cache/huggingface/hub/"
            "models--meta-llama--Llama-3.1-8B-Instruct,readonly",
            "snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
            "KVBENCH_KIVI_SOURCE_ROOT=/opt/kivi-source",
            'KVBENCH_AUTHORIZED_IMAGE_DIGEST="$$image_id"',
            "KVBENCH_EXECUTION_ENVIRONMENT=measurement_container",
        ):
            self.assertIn(required, recipe)
        writable_bind_mounts = [
            line
            for line in recipe.splitlines()
            if "--mount" in line and ",readonly" not in line
        ]
        self.assertEqual(len(writable_bind_mounts), 1)
        self.assertIn("artifacts/phase8", writable_bind_mounts[0])
        for forbidden in (
            "--env-file",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "CLOUDFLARE_API_TOKEN",
            "R2_ACCOUNT_ID",
            "R2_ENDPOINT",
        ):
            self.assertNotIn(forbidden, recipe)

    def test_required_cuda_graph_checks_and_driver_are_launched(self) -> None:
        recipe = _recipe("admit-kivi")
        self.assertIn(
            "/usr/bin/mkdir -p /tmp/kivi-b019-objects",
            recipe,
        )
        self.assertIn(
            "GIT_OBJECT_DIRECTORY=/tmp/kivi-b019-objects "
            "GIT_ALTERNATE_OBJECT_DIRECTORIES="
            "/opt/kivi-source/.git/objects",
            recipe,
        )
        self.assertIn(
            "/opt/kvbench/.venv/bin/python "
            "scripts/validate_kivi_b019_patch.py --device cpu "
            "--source-root /opt/kivi-source",
            recipe,
        )
        self.assertIn(
            "PHASE3_PYTHON=/opt/kvbench/.venv/bin/python test-cuda",
            recipe,
        )
        self.assertIn(
            "PHASE3_PYTHON=/opt/kvbench/.venv/bin/python test-graph",
            recipe,
        )
        self.assertIn(
            "/opt/kvbench/.venv/bin/python "
            "scripts/phase8_kivi_admission.py",
            recipe,
        )
        for forbidden in (
            "pilot",
            "full-scan",
            "profile-subset",
            "nsight",
            "quality",
            "phase9",
            "kvquant",
            "$(R2_ARTIFACT)",
        ):
            self.assertNotIn(forbidden, recipe.casefold())
        self.assertIn(
            "FAILED_PHASE8_LAUNCH_EVIDENCE_PRESERVED",
            recipe,
        )
        self.assertIn("preserve=0", recipe)

    def test_validation_and_outer_bundle_targets_are_host_side_only(
        self,
    ) -> None:
        validation = _recipe("validate-admission-kivi")
        self.assertIn(
            "-m scripts.phase8_kivi_admission --validate-only "
            '--artifact "$(PHASE8_KIVI_ADMISSION_ARTIFACT)"',
            validation,
        )
        self.assertIn(
            "PHASE8_KIVI_ADMISSION_ARTIFACT_required",
            validation,
        )
        self.assertNotIn("docker", validation)
        self.assertNotIn("publish", validation)

        build = _recipe("phase8-r2-outer-bundle")
        self.assertIn(
            "-m scripts.phase8_r2_outer_bundle build "
            '--source-bundle "$(PHASE8_R2_INNER_ARTIFACT)" '
            '--run-id "$(PHASE8_R2_OUTER_RUN_ID)"',
            build,
        )
        self.assertNotIn("r2_artifact.py", build)
        self.assertNotIn(" publish ", build)

        outer_validation = _recipe("validate-phase8-r2-outer-bundle")
        self.assertIn(
            "-m scripts.phase8_r2_outer_bundle validate "
            '"$(PHASE8_R2_OUTER_ARTIFACT)" '
            '--source-bundle "$(PHASE8_R2_INNER_ARTIFACT)"',
            outer_validation,
        )
        self.assertNotIn("docker", outer_validation)
        self.assertNotIn("r2_artifact.py", outer_validation)

        publication_validation = _recipe(
            "validate-phase8-r2-outer-publication"
        )
        self.assertIn(
            "-m scripts.phase8_r2_outer_bundle "
            "validate-publication "
            '"$(PHASE8_R2_OUTER_ARTIFACT)" '
            '--source-bundle "$(PHASE8_R2_INNER_ARTIFACT)" '
            '--receipt "$(PHASE8_R2_OUTER_RECEIPT)"',
            publication_validation,
        )
        self.assertNotIn("docker", publication_validation)
        self.assertNotIn("r2_artifact.py", publication_validation)

    def test_bare_validation_is_bound_to_published_admission(self) -> None:
        evidence_root = REPOSITORY_ROOT / "docs" / "evidence" / "phase8"
        method_report = json.loads(
            (evidence_root / "kivi-method-admission.json").read_text(
                encoding="utf-8"
            )
        )
        publication = json.loads(
            (evidence_root / "r2-admission-publication.json").read_text(
                encoding="utf-8"
            )
        )
        referenced_run_ids = {
            reference["path"].split("/")[2]
            for reference in method_report["evidence_references"]
            if reference["path"].startswith("artifacts/phase8/")
        }

        self.assertEqual(
            publication["source_run_id"],
            KIVI_ADMISSION_RUN_ID,
        )
        self.assertEqual(referenced_run_ids, {KIVI_ADMISSION_RUN_ID})
        self.assertEqual(
            publication["local_validation"]["root_sha256"],
            method_report["local_root_digest"],
        )
        self.assertEqual(
            _assignment("PHASE8_KIVI_ADMISSION_ARTIFACT"),
            KIVI_ADMISSION_ARTIFACT,
        )
        self.assertIsNotNone(
            re.search(
                r"^PHASE8_KIVI_ADMISSION_ARTIFACT := "
                rf"{re.escape(KIVI_ADMISSION_ARTIFACT)}$",
                MAKEFILE,
                flags=re.MULTILINE,
            )
        )


if __name__ == "__main__":
    unittest.main()

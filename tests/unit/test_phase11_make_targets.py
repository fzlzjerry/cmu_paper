"""Static tests for the exact-container Phase 11 KVQuant Make targets."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
AUTHORIZED_CONTAINER = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
CORRECTED_COMMIT = "4b8533b29b04f8c4bf55f688a41fefe20487637b"
CORRECTED_TREE = "46f2149a0369d5c97d9a6bc77d57b5f3a5a5fb3b"
AGGREGATE_PATCH = (
    "bae63bced549479709b10d7f6a8ee35a8f21ec18cc040a7424591cee47c1b0a6"
)
AUTHORITY_EXTENSION = (
    "a79644923ba131e56abe95029e669346dbbb11fd210d2b9f8b2086819ffeaad1"
)
CALIBRATION_ROOT = (
    "8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf"
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
            body = []
            for candidate in lines[index + 1 :]:
                if not candidate.startswith("\t"):
                    break
                body.append(candidate)
            return "\n".join(body)
    raise AssertionError(f"missing Make recipe: {target}")


class Phase11KVQuantMakeTargetTests(unittest.TestCase):
    def test_authorities_are_exact_overrides(self) -> None:
        expected = {
            "PHASE11_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST": (
                AUTHORIZED_CONTAINER
            ),
            "PHASE11_KVQUANT_CORRECTED_COMMIT": CORRECTED_COMMIT,
            "PHASE11_KVQUANT_CORRECTED_TREE": CORRECTED_TREE,
            "PHASE11_KVQUANT_AGGREGATE_PATCH_SHA256": AGGREGATE_PATCH,
            "PHASE11_KVQUANT_EXTENSION_SHA256": AUTHORITY_EXTENSION,
            "PHASE11_KVQUANT_CALIBRATION_ROOT_SHA256": CALIBRATION_ROOT,
        }
        for name, value in expected.items():
            self.assertEqual(_assignment(name), value)
            self.assertIsNotNone(
                re.search(
                    rf"^override {name} :=",
                    MAKEFILE,
                    flags=re.MULTILINE,
                )
            )

    def test_calibration_mount_is_exact_lifecycle_validated_authority(
        self,
    ) -> None:
        recipe = _recipe("admit-kvquant")
        self.assertIsNotNone(
            re.search(
                r"^override PHASE11_KVQUANT_CALIBRATION :=",
                MAKEFILE,
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(
            _assignment("PHASE11_KVQUANT_CALIBRATION"),
            "$(CURDIR)/calibration/kvquant/"
            "kvqcal-cdb724c806d64d095c040d2673a987a3",
        )
        for required in (
            'calibration_root="$(PHASE11_KVQUANT_CALIBRATION)"',
            'test "$$calibration_root" = "$$repository_root/calibration/'
            'kvquant/kvqcal-cdb724c806d64d095c040d2673a987a3"',
            'test "$$(realpath -e "$$calibration_root")" = '
            '"$$calibration_root"',
            "validate_local_artifact(sys.argv[1], environ={})",
            "$(PHASE11_KVQUANT_CALIBRATION_ROOT_SHA256)",
            "src=$$calibration_root,dst=/opt/kvquant-calibration/"
            "kvqcal-cdb724c806d64d095c040d2673a987a3,readonly",
            "KVBENCH_KVQUANT_CALIBRATION_ROOT=/opt/kvquant-calibration/"
            "kvqcal-cdb724c806d64d095c040d2673a987a3",
        ):
            self.assertIn(required, recipe)
        self.assertGreaterEqual(recipe.count("/usr/bin/env -i"), 3)
        self.assertIn("/usr/bin/python3", recipe)
        self.assertNotIn("$(PHASE2_ENV)", recipe)
        self.assertNotIn("$(PHASE2_PYTHON)", recipe)
        self.assertNotIn(
            "src=$(PHASE11_KVQUANT_CALIBRATION),"
            "dst=/opt/kvquant-calibration/",
            recipe,
        )

    def test_admission_uses_clean_detached_repository_and_source(self) -> None:
        recipe = _recipe("admit-kvquant")
        self.assertIn(
            "admit-kvquant: override MEASUREMENT_IMAGE_CONFIG_DIGEST := "
            "$(PHASE11_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)",
            MAKEFILE,
        )
        self.assertIn(
            "admit-kvquant: verify-measurement-container",
            MAKEFILE,
        )
        self.assertIn(
            "git status --porcelain=v1 --untracked-files=all",
            recipe,
        )
        self.assertEqual(
            recipe.count("git clone --quiet --no-local --no-checkout"),
            2,
        )
        self.assertEqual(recipe.count("checkout --quiet --detach"), 2)
        self.assertEqual(recipe.count("remote remove origin"), 2)
        self.assertIn("$(PHASE11_KVQUANT_CORRECTED_COMMIT)", recipe)
        self.assertIn("$(PHASE11_KVQUANT_CORRECTED_TREE)", recipe)
        self.assertIn(
            "-m scripts.validate_kvquant_long_context_patch",
            recipe,
        )
        self.assertIn(
            '"$(PHASE11D_KVQUANT_SOURCE_ROOT)"',
            recipe,
        )
        self.assertIn(
            "$(PHASE11_KVQUANT_AGGREGATE_PATCH_SHA256)",
            recipe,
        )
        self.assertIn(
            'repository_root="$$(git rev-parse --show-toplevel)"',
            recipe,
        )
        self.assertIn(
            'test "$$repository_root" = "$(CURDIR)"',
            recipe,
        )
        for fixture_root in (
            "reference/kvquant/fixtures",
            "reference/kvquant_phase11pr/fixtures",
        ):
            self.assertIn(
                f'"$$task_root/repository/{fixture_root}"',
                recipe,
            )
        self.assertIn('chmod -R a-w -- "$$fixture_root"', recipe)
        self.assertIn(
            'test -z "$$(find "$$fixture_root" -perm /222 '
            '-print -quit)"',
            recipe,
        )

    def test_executed_repository_and_cuda_build_sources_bind_to_git(
        self,
    ) -> None:
        recipe = _recipe("admit-kvquant")
        for relative in (
            "scripts/phase11_kvquant_admission.py",
            "scripts/validate_kvquant_long_context_patch.py",
            "src/kvbench/adapters/kvquant.py",
            "src/kvbench/runtime/bf16_endpoint.py",
            "src/kvbench/runtime/kvquant_cache.py",
            "src/kvbench/runtime/kvquant_session.py",
            "src/kvbench/schema/phase11.py",
            "tests/cuda/phase11_kvquant_sanitizer_probe.py",
            "tests/cuda/test_phase11_kvquant_cuda.py",
            "tests/graph/test_phase11_kvquant_graph.py",
        ):
            self.assertIn(relative, recipe)
        self.assertIn(
            'git -C "$$task_root/repository" cat-file -e '
            '"$$head:$$file"',
            recipe,
        )
        self.assertIn(
            'git -C "$$task_root/repository" cat-file blob '
            '"$$head:$$file"',
            recipe,
        )
        self.assertIn(
            'git -C "$$task_root/kvquant-source" cat-file -e '
            '"$(PHASE11_KVQUANT_CORRECTED_COMMIT):$$relative"',
            recipe,
        )
        self.assertIn(
            'git -C "$$task_root/kvquant-source" cat-file blob '
            '"$(PHASE11_KVQUANT_CORRECTED_COMMIT):$$relative"',
            recipe,
        )
        self.assertGreaterEqual(
            recipe.count('= "$$committed_sha256"'),
            3,
        )

    def test_stripped_fresh_build_is_the_runtime_authority(self) -> None:
        recipe = _recipe("admit-kvquant")
        self.assertIsNone(
            re.search(
                r"^PHASE11_KVQUANT_EXTENSION\s*\?=",
                MAKEFILE,
                flags=re.MULTILINE,
            )
        )
        self.assertNotIn("$(PHASE11_KVQUANT_EXTENSION)", recipe)
        self.assertNotIn("/opt/kvquant-authority", recipe)
        self.assertIn("$(PHASE11_KVQUANT_EXTENSION_SHA256)", recipe)
        self.assertIn("setup_cuda.py build_ext --inplace", recipe)
        self.assertIn(
            '/usr/bin/strip --strip-unneeded "$$fresh_extension"',
            recipe,
        )
        self.assertIn(
            'sha256sum "$$fresh_extension"',
            recipe,
        )
        self.assertIn(".sm_120.cubin", recipe)
        self.assertIn(".target sm_120", recipe)
        self.assertIn(
            'KVBENCH_KVQUANT_EXTENSION="$$fresh_extension"',
            recipe,
        )
        self.assertIn(
            'KVBENCH_KVQUANT_FRESH_BUILD_EXTENSION="$$fresh_extension"',
            recipe,
        )
        self.assertIn(
            "-m scripts.validate_kvquant_long_context_patch "
            "--source-root /opt/kvquant-source",
            recipe,
        )

    def test_measurement_container_is_readonly_offline_and_secret_free(
        self,
    ) -> None:
        recipe = _recipe("admit-kvquant")
        for required in (
            "--read-only",
            "--network=none",
            '--gpus "device=$(MEASUREMENT_GPU_UUID)"',
            "dst=/home/rockrock/cmu_paper,readonly",
            "dst=/opt/kvquant-source,readonly",
            "dst=/opt/kvquant-calibration/"
            "kvqcal-cdb724c806d64d095c040d2673a987a3,readonly",
            "models--meta-llama--Llama-3.1-8B-Instruct,readonly",
            "KVBENCH_AUTHORIZED_IMAGE_DIGEST",
            "KVBENCH_EXECUTION_ENVIRONMENT=measurement_container",
        ):
            self.assertIn(required, recipe)
        self.assertNotIn("--pid=host", recipe)
        writable = [
            line
            for line in recipe.splitlines()
            if "--mount" in line and ",readonly" not in line
        ]
        self.assertEqual(len(writable), 2)
        self.assertTrue(
            any("artifacts/phase11" in line for line in writable)
        )
        self.assertTrue(any("/opt/kvquant-build" in line for line in writable))
        for forbidden in (
            "--env-file",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "CLOUDFLARE_API_TOKEN",
            "R2_ACCOUNT_ID",
            "R2_ENDPOINT",
            "pip install",
            "apt-get",
            "docker build",
        ):
            self.assertNotIn(forbidden, recipe)

    def test_artifact_mount_is_fresh_empty_and_symlink_free(self) -> None:
        recipe = _recipe("admit-kvquant")
        for required in (
            'artifact_parent="$$repository_root/artifacts"',
            'test -d "$$artifact_parent" && test ! -L "$$artifact_parent"',
            'artifact_store_root="$$artifact_parent/phase11"',
            "test -d \"$$artifact_store_root\" && "
            'test ! -L "$$artifact_store_root"',
            'test "$$(realpath -e "$$artifact_store_root")" = '
            '"$$repository_root/artifacts/phase11"',
            'artifact_root="$$(mktemp -d "$$artifact_store_root/'
            'phase11-launch.XXXXXX")"',
            'test -d "$$artifact_root" && test ! -L "$$artifact_root"',
            'test "$$(realpath -e "$$artifact_root/..")" = '
            '"$$artifact_store_root"',
            'test ! -e "$$artifact_root/.env"',
            'test ! -L "$$artifact_root/.env"',
            'find "$$artifact_root" -mindepth 1 -maxdepth 1 '
            "-print -quit",
            "PHASE11_LAUNCH_ARTIFACT_ROOT_PRESERVED",
        ):
            self.assertIn(required, recipe)
        self.assertNotIn(
            'artifact_root="$$(realpath "$(CURDIR)/artifacts/phase11")"',
            recipe,
        )
        self.assertNotIn(
            'test -z "$$(find "$$artifact_store_root"',
            recipe,
        )
        self.assertNotIn('rm -rf -- "$$artifact_root"', recipe)

    def test_admission_launches_only_narrow_driver_work(self) -> None:
        recipe = _recipe("admit-kvquant")
        self.assertIn(
            "cd /home/rockrock/cmu_paper && "
            "/opt/kvbench/.venv/bin/python -m "
            "scripts.phase11_kvquant_admission",
            recipe,
        )
        self.assertIn(
            "FAILED_PHASE11_LAUNCH_EVIDENCE_PRESERVED",
            recipe,
        )
        for forbidden in (
            "pilot",
            "full-scan",
            "profile-subset",
            "nsight",
            "quality",
            "phase12",
            "publish-artifact-r2",
        ):
            self.assertNotIn(forbidden, recipe.casefold())

    def test_final_validation_is_exact_container_and_four_way_bound(
        self,
    ) -> None:
        recipe = _recipe("validate-admission-kvquant")
        self.assertIn(
            "validate-admission-kvquant: override "
            "MEASUREMENT_IMAGE_CONFIG_DIGEST := "
            "$(PHASE11_KVQUANT_AUTHORIZED_IMAGE_CONFIG_DIGEST)",
            MAKEFILE,
        )
        self.assertIn(
            "validate-admission-kvquant: verify-measurement-container",
            MAKEFILE,
        )
        for required in (
            "git status --porcelain=v1 --untracked-files=all",
            "source_tree_not_clean",
            "PHASE11_KVQUANT_INNER_ARTIFACT_required",
            "PHASE11_KVQUANT_OUTER_ARTIFACT_required",
            "phase11_method_admission_checksum_required",
            "docker image inspect",
            "validate_local_artifact",
            "git clone --quiet --no-local --no-checkout",
            "checkout --quiet --detach",
            "remote remove origin",
            'test ! -e "$$task_root/repository/.env"',
            "/usr/bin/sha256sum -c",
            "docker create --read-only --network=none",
            "src=$$task_root/repository,"
            "dst=/home/rockrock/cmu_paper,readonly",
            "phase11_inner_artifact_must_be_repository_relative",
            "phase11_outer_artifact_must_be_repository_relative",
            'inner_relative="$${inner_source#$$repository_root/}"',
            'outer_relative="$${outer_source#$$repository_root/}"',
            'mkdir -p "$$task_root/repository/'
            '$$(dirname "$$inner_relative")"',
            "src=$$inner_source,dst=/home/rockrock/cmu_paper/"
            "$$inner_relative,readonly",
            "src=$$outer_source,dst=/home/rockrock/cmu_paper/"
            "$$outer_relative,readonly",
            "--entrypoint /opt/kvbench/.venv/bin/python",
            "PYTHONPATH=/home/rockrock/cmu_paper/src:"
            "/home/rockrock/cmu_paper:"
            "/opt/kvbench/.phase3/site-packages",
            "--validate-only",
            '--artifact "/home/rockrock/cmu_paper/$$inner_relative"',
            '--outer-artifact "/home/rockrock/cmu_paper/$$outer_relative"',
            "--method-admission-report "
            '"/home/rockrock/cmu_paper/'
            '$(PHASE11_KVQUANT_METHOD_ADMISSION_REPORT)"',
            "--publication-receipt "
            '"/home/rockrock/cmu_paper/'
            '$(PHASE11_KVQUANT_PUBLICATION_RECEIPT)"',
        ):
            self.assertIn(required, recipe)
        for exact_output in (
            "PHASE11_KVQUANT_METHOD_ADMISSION_REPORT",
            "PHASE11_KVQUANT_METHOD_ADMISSION_CHECKSUM",
            "PHASE11_KVQUANT_PUBLICATION_RECEIPT",
        ):
            self.assertIsNotNone(
                re.search(
                    rf"^override {exact_output} :=",
                    MAKEFILE,
                    flags=re.MULTILINE,
                )
            )
        for forbidden in (
            "--gpus",
            "--pid=host",
            "--env-file",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "CLOUDFLARE_API_TOKEN",
            "$(PHASE3_ENV)",
            "$(PHASE3_PYTHON)",
            "publish-artifact-r2",
        ):
            self.assertNotIn(forbidden, recipe)


if __name__ == "__main__":
    unittest.main()

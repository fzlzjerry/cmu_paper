"""Static safety checks for the Phase 6A Measurement Container definition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPOSITORY_ROOT / "docker" / "measurement.Dockerfile"
DOCKERIGNORE = REPOSITORY_ROOT / ".dockerignore"
MAKEFILE = REPOSITORY_ROOT / "Makefile"
OBSERVED_CONTAINER_LOCK = (
    REPOSITORY_ROOT
    / "preflight"
    / "measurement-container-system-packages.lock.json"
)
OBSERVED_CONTAINER_IDENTITY = (
    REPOSITORY_ROOT / "artifacts" / "phase6a" / "container_g0"
)
BASE_MANIFEST = (
    "sha256:"
    "0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c"
)
PYTHON_LOCK_SHA256 = {
    "preflight/requirements-e00.txt": (
        "aafe68e54cb316d6bb673dbc42087b2f971ac94668973cc3f8cc555d8a0dbb29"
    ),
    "preflight/requirements-phase3.txt": (
        "cebe254a3e03a48e3e67100ce11d5623fc0dc722dc43e2f482152beb644a08e9"
    ),
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MeasurementContainerDefinitionTests(unittest.TestCase):
    def test_base_is_exact_linux_amd64_manifest(self) -> None:
        text = _text(DOCKERFILE)
        from_lines = [
            line.strip() for line in text.splitlines() if line.startswith("FROM ")
        ]
        self.assertEqual(
            from_lines,
            [f"FROM --platform=linux/amd64 nvidia/cuda@{BASE_MANIFEST}"],
        )
        self.assertIn(
            f'org.kvbench.measurement.base.manifest="{BASE_MANIFEST}"',
            text,
        )

    def test_only_exact_lock_inputs_are_copied(self) -> None:
        text = _text(DOCKERFILE)
        copies = {
            tuple(line.split()[1:])
            for line in text.splitlines()
            if line.startswith("COPY ")
        }
        expected = {(path, path) for path in PYTHON_LOCK_SHA256}
        self.assertEqual(copies, expected)
        self.assertNotIn("\nADD ", f"\n{text}")
        self.assertNotRegex(text, r"(?m)^COPY\s+\.\s")

    def test_lock_labels_match_current_lock_bytes(self) -> None:
        text = _text(DOCKERFILE)
        label_names = {
            "preflight/requirements-e00.txt": "requirements-e00",
            "preflight/requirements-phase3.txt": "requirements-phase3",
        }
        for relative, expected in PYTHON_LOCK_SHA256.items():
            with self.subTest(relative=relative):
                self.assertEqual(_sha256(REPOSITORY_ROOT / relative), expected)
                label = label_names[relative]
                self.assertIn(
                    f'org.kvbench.measurement.{label}.sha256="{expected}"',
                    text,
                )

    def test_native_lock_is_not_presented_as_container_authority(self) -> None:
        text = _text(DOCKERFILE)
        self.assertNotIn("preflight/system-packages.lock.json", text)
        self.assertNotIn("org.kvbench.measurement.system-lock", text)
        self.assertNotIn("org.kvbench.measurement.image-id", text)
        self.assertNotIn("org.kvbench.measurement.image.digest", text)

    def test_build_derived_identity_is_never_fabricated(self) -> None:
        if OBSERVED_CONTAINER_LOCK.exists():
            metadata = OBSERVED_CONTAINER_LOCK.lstat()
            self.assertTrue(stat.S_ISREG(metadata.st_mode))
            payload = json.loads(OBSERVED_CONTAINER_LOCK.read_text())
            self.assertEqual(
                payload["scope"]["execution_environment_kind"],
                "measurement_container",
            )
            self.assertTrue(payload["dpkg_packages"])
            self.assertTrue(payload["tools"])
        if OBSERVED_CONTAINER_IDENTITY.exists():
            for entry in OBSERVED_CONTAINER_IDENTITY.iterdir():
                with self.subTest(entry=entry.name):
                    self.assertFalse(entry.name.startswith("."))
                    self.assertTrue(entry.is_dir())
                    self.assertTrue((entry / "COMPLETE").is_file())
                    self.assertEqual(
                        entry.stat().st_mode
                        & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH),
                        0,
                    )

    def test_frozen_measurement_stack_inputs_are_explicit(self) -> None:
        dockerfile = _text(DOCKERFILE)
        e00_lock = _text(
            REPOSITORY_ROOT / "preflight" / "requirements-e00.txt"
        )
        phase3_lock = _text(
            REPOSITORY_ROOT / "preflight" / "requirements-phase3.txt"
        )
        for token in (
            "python3.12=3.12.3-1ubuntu0.15",
            "gcc-13=13.3.0-6ubuntu2~24.04.1",
            "g++-13=13.3.0-6ubuntu2~24.04.1",
            "cuda-nvcc-13-0=13.0.88-1",
            "cuda-nvdisasm-13-0=13.0.85-1",
            "cuda-sanitizer-13-0=13.0.85-1",
            "compute-sanitizer --version",
            "--require-hashes",
            "--no-deps",
        ):
            with self.subTest(token=token):
                self.assertIn(token, dockerfile)
        self.assertIn("torch==2.12.1+cu130", e00_lock)
        self.assertIn("triton==3.7.1", e00_lock)
        self.assertIn("transformers==4.57.6", phase3_lock)

    def test_sm120_native_and_ptx_targets_are_frozen(self) -> None:
        text = _text(DOCKERFILE)
        self.assertIn("TORCH_CUDA_ARCH_LIST=12.0+PTX", text)
        self.assertIn("CUDAARCHS=120", text)
        self.assertIn("CMAKE_CUDA_ARCHITECTURES=120", text)
        self.assertRegex(text, r"(?m)^\s*LD_LIBRARY_PATH=\s*\\?$")

    def test_context_is_deny_by_default_and_secret_safe(self) -> None:
        rules = [
            line.strip()
            for line in _text(DOCKERIGNORE).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        allowed = {rule for rule in rules if rule.startswith("!")}
        self.assertEqual(
            allowed,
            {
                "!docker/",
                "!docker/measurement.Dockerfile",
                "!preflight/",
                "!preflight/requirements-e00.txt",
                "!preflight/requirements-phase3.txt",
            },
        )
        self.assertEqual(rules[0], "**")
        self.assertIn(".env", rules)
        self.assertIn(".env.*", rules)

    def test_no_secret_model_artifact_reference_or_driver_package(self) -> None:
        text = _text(DOCKERFILE)
        forbidden = (
            ".env",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "CLOUDFLARE_API_TOKEN",
            "R2_ACCOUNT_ID",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "nvidia-utils-",
            "libnvidia-",
            "reference/turboquant",
            "vllm/",
            "artifacts/",
            "model weights",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token.lower(), text.lower())
        secret_arg = re.compile(
            r"(?im)^ARG\s+\S*(?:SECRET|TOKEN|PASSWORD|ACCESS_KEY)\S*"
        )
        self.assertIsNone(secret_arg.search(text))

    def test_make_targets_use_exact_image_and_secret_free_clean_source(self) -> None:
        text = _text(MAKEFILE)
        for target in (
            "measurement-container:",
            "observe-measurement-container-lock:",
            "measurement-container-lock-review:",
            "verify-measurement-container:",
            "preflight-container:",
            "publish-artifact-r2:",
            "verify-artifact-r2:",
        ):
            with self.subTest(target=target):
                self.assertIn(target, text)
        self.assertIn("git archive --format=tar", text)
        self.assertIn("git clone --quiet --no-local --no-checkout", text)
        self.assertIn('test ! -e "$$task_root/source/.env"', text)
        self.assertIn(
            "PYTHONPATH=$(CURDIR):$(CURDIR)/src $(PHASE2_PYTHON) "
            "scripts/r2_artifact.py",
            text,
        )
        self.assertGreaterEqual(text.count("docker image save --output"), 2)
        self.assertGreaterEqual(
            text.count("validate_docker_image_save_archive"),
            2,
        )
        self.assertIn("--network=none --pid=host", text)
        self.assertIn('--gpus "device=$(MEASUREMENT_GPU_UUID)"', text)
        self.assertIn('--workdir /workspace "$$image_id"', text)
        self.assertIn("--container-image-config-digest \"$$image_id\"", text)
        self.assertIn(
            "--container-runtime-inspect-json "
            "/run/kvbench/runtime-inspect.json",
            text,
        )
        self.assertIn(
            '--container-source-revision "$$image_revision"',
            text,
        )
        self.assertIn(
            "--container-image-history-jsonl "
            "/run/kvbench/image-history.jsonl",
            text,
        )
        self.assertIn(
            "--container-image-build-scan-json "
            "/run/kvbench/image-build-scan.json",
            text,
        )
        self.assertGreaterEqual(
            text.count('git merge-base --is-ancestor "$$image_revision" "$$head"'),
            2,
        )
        self.assertGreaterEqual(
            text.count(
                'git diff --quiet "$$image_revision" "$$head" -- '
                "docker/measurement.Dockerfile "
                "preflight/requirements-e00.txt "
                "preflight/requirements-phase3.txt"
            ),
            2,
        )
        self.assertGreaterEqual(
            text.count('git cat-file -e "$$image_revision:$$lock_path"'),
            2,
        )
        self.assertIn('rm -f -- "$$task_root/image-save.tar"', text)
        self.assertIn('image_id="$(MEASUREMENT_IMAGE_CONFIG_DIGEST)"', text)
        self.assertNotIn("--env-file", text)
        reference_recipe = text.split("\nreference-kivi:", 1)[1].split(
            "\nvalidate-reference-kivi:", 1
        )[0]
        self.assertIn(
            '--build-arg KIVI_BUILD_REVISION="$(KIVI_REFERENCE_BUILD_REVISION)"',
            reference_recipe,
        )
        self.assertEqual(reference_recipe.count("--build-arg"), 1)
        kvquant_reference_recipe = text.split(
            "\nreference-kvquant:", 1
        )[1].split("\nvalidate-reference-kvquant:", 1)[0]
        self.assertEqual(
            kvquant_reference_recipe.count("--build-arg"),
            2,
        )
        self.assertNotIn(
            "--build-arg",
            text.replace(reference_recipe, "").replace(
                kvquant_reference_recipe,
                "",
            ),
        )

    def test_container_lock_bootstrap_is_two_pass_and_fail_closed(self) -> None:
        text = _text(MAKEFILE)
        self.assertIn(
            "observe-measurement-container-lock: phase6a-source-safety",
            text,
        )
        self.assertIn(
            "measurement-container-lock-review: phase6a-source-safety",
            text,
        )
        self.assertIn(
            "verify-measurement-container: measurement-container-lock-review",
            text,
        )
        observation = text.split(
            "observe-measurement-container-lock:", 1
        )[1].split("measurement-container-lock-review:", 1)[0]
        self.assertIn("--read-only --network=none", observation)
        self.assertIn(
            "--observe-measurement-container-system-lock", observation
        )
        self.assertIn("publish_atomic_exclusive", observation)
        self.assertIn(
            "measurement-container-system-packages.lock.json", observation
        )
        self.assertNotIn(
            "cp preflight/measurement-container-system-packages.lock.json",
            observation,
        )
        review = text.split(
            "measurement-container-lock-review:", 1
        )[1].split("verify-measurement-container:", 1)[0]
        self.assertIn("git ls-files --error-unmatch", review)
        self.assertIn("cmp -s", review)
        self.assertIn("load_measurement_container_system_lock", review)
        self.assertIn("REVIEWED_LOCK_BYTES_EQUAL", review)


if __name__ == "__main__":
    unittest.main()

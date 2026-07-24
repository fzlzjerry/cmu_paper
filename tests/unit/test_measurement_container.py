"""Static safety checks for the Phase 6A Measurement Container definition."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPOSITORY_ROOT / "docker" / "measurement.Dockerfile"
DOCKERIGNORE = REPOSITORY_ROOT / ".dockerignore"
BASE_MANIFEST = (
    "sha256:"
    "0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c"
)
LOCK_SHA256 = {
    "preflight/system-packages.lock.json": (
        "9283ef0f7edc23a07a1943afc014adb5b5e45973e305ccd5cb22d9ccc29e9b7a"
    ),
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
        expected = {(path, path) for path in LOCK_SHA256}
        self.assertEqual(copies, expected)
        self.assertNotIn("\nADD ", f"\n{text}")
        self.assertNotRegex(text, r"(?m)^COPY\s+\.\s")

    def test_lock_labels_match_current_lock_bytes(self) -> None:
        text = _text(DOCKERFILE)
        label_names = {
            "preflight/system-packages.lock.json": "system-lock",
            "preflight/requirements-e00.txt": "requirements-e00",
            "preflight/requirements-phase3.txt": "requirements-phase3",
        }
        for relative, expected in LOCK_SHA256.items():
            with self.subTest(relative=relative):
                self.assertEqual(_sha256(REPOSITORY_ROOT / relative), expected)
                label = label_names[relative]
                self.assertIn(
                    f'org.kvbench.measurement.{label}.sha256="{expected}"',
                    text,
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
                "!preflight/system-packages.lock.json",
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


if __name__ == "__main__":
    unittest.main()

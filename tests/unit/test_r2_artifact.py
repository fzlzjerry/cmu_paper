"""Focused offline tests for the Phase 6A Cloudflare R2 artifact path."""

from __future__ import annotations

from contextlib import redirect_stdout
import copy
from datetime import datetime, timezone
from io import BytesIO, StringIO
import json
from pathlib import Path
import shutil
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

from kvbench.runtime.artifacts import AppendOnlyArtifactStore
from preflight import run_preflight
from scripts.r2_artifact import (
    _output,
    ArtifactValidationError,
    CloudflareReadClient,
    ObjectConflictError,
    R2ArtifactError,
    R2Config,
    R2S3Client,
    RemoteRequestError,
    artifact_object_key,
    artifact_object_prefix,
    main,
    publish_artifact,
    redact_text,
    required_variable_status,
    sha256_bytes,
    sha256_file,
    validate_local_artifact,
    verify_cloudflare_bucket_lock,
    verify_remote_artifact,
)
from tests.unit.test_phase2_artifacts import (
    created_manifest,
    terminal_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def finalized_artifact(
    root: Path,
    *,
    payload: bytes = b"synthetic artifact bytes\n",
    extra_files: dict[str, bytes] | None = None,
    manifest_extra: dict[str, object] | None = None,
    inventory_extra: dict[str, object] | None = None,
    inventory_item_extra: dict[str, object] | None = None,
    completion_extra: dict[str, object] | None = None,
) -> Path:
    root.mkdir()
    _write(root / "raw" / "payload.bin", payload)
    extras = {} if extra_files is None else extra_files
    for relative, data in extras.items():
        _write(root / relative, data)

    manifest: dict[str, object] = {
        "schema_version": "kvbench-r2-synthetic-manifest-1.0.0",
        "run_id": "r2-synthetic",
        "scientific_measurement": False,
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    _write(root / "manifest.json", _json_bytes(manifest))

    inventory_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    inventory = {
        "schema_version": "kvbench-artifact-inventory-1.0.0",
        "run_id": "r2-synthetic",
        "files": [
            {
                "path": relative,
                "role": "artifact",
                "size_bytes": (root / relative).stat().st_size,
                "sha256": sha256_file(root / relative),
            }
            for relative in inventory_paths
        ],
        "excluded_control_files": [
            "artifact_inventory.json",
            "checksums.sha256",
            "COMPLETE",
        ],
    }
    if inventory_item_extra:
        inventory["files"][0].update(inventory_item_extra)
    if inventory_extra:
        inventory.update(inventory_extra)
    _write(root / "artifact_inventory.json", _json_bytes(inventory))

    ledger_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    ledger = "".join(
        f"{sha256_file(root / relative)}  {relative}\n"
        for relative in ledger_paths
    ).encode()
    _write(root / "checksums.sha256", ledger)
    complete = {
        "schema_version": "kvbench-completion-1.0.0",
        "run_id": "r2-synthetic",
        "status": "completed",
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "artifact_inventory_sha256": sha256_file(
            root / "artifact_inventory.json"
        ),
        "checksum_ledger_path": "checksums.sha256",
        "checksum_ledger_sha256": sha256_file(root / "checksums.sha256"),
        "written_last": True,
    }
    if completion_extra:
        complete.update(completion_extra)
    _write(root / "COMPLETE", _json_bytes(complete))
    _make_immutable(root)
    return root


class FakeR2:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.put_calls: list[tuple[str, str, str | None]] = []

    def get_object_or_none(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def put_object_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None,
    ) -> str:
        self.put_calls.append((key, "*", content_type))
        if key in self.objects:
            raise RemoteRequestError(status=412, code="PreconditionFailed")
        self.objects[key] = data
        # Deliberately not a checksum.  The publisher must ignore this value.
        return '"not-a-scientific-checksum"'

    def list_keys(self, prefix: str) -> list[str]:
        return sorted(key for key in self.objects if key.startswith(prefix))


class FakeCloudflare:
    def __init__(
        self,
        *,
        public: bool = False,
        bucket_name: str = "phase6a-bucket",
        custom_domains: list[object] | None = None,
        rules: list[dict[str, object]] | None = None,
    ) -> None:
        self.public = public
        self.bucket_name = bucket_name
        self.custom_domains = [] if custom_domains is None else custom_domains
        self.rules = (
            [
                {
                    "id": "phase6a-indefinite",
                    "name": "phase6a-indefinite-lock",
                    "enabled": True,
                    "prefix": "kvbench/sha256/",
                    "condition": {"type": "Indefinite"},
                }
            ]
            if rules is None
            else rules
        )
        self.paths: list[str] = []
        self.checks: list[str] = []

    def get_json(
        self,
        path: str,
        *,
        check: str = "cloudflare_management",
    ) -> dict[str, object]:
        self.paths.append(path)
        self.checks.append(check)
        if path.endswith("/domains/managed"):
            result: object = {
                "bucketId": "not-recorded",
                "domain": "not-recorded",
                "enabled": self.public,
            }
        elif path.endswith("/domains/custom"):
            result = {"domains": self.custom_domains}
        elif path.endswith("/lock"):
            result = {"rules": self.rules}
        else:
            result = {"name": self.bucket_name}
        return {"success": True, "result": result}


class FakeResponse:
    def __init__(self, data: bytes = b"") -> None:
        self.data = data

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, amount: int | None = None) -> bytes:
        return self.data if amount is None else self.data[:amount]


class R2ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.environ = {
            "R2_ACCOUNT_ID": "a" * 32,
            "R2_BUCKET": "phase6a-bucket",
            "R2_ENDPOINT": (
                "https://" + "a" * 32 + ".r2.cloudflarestorage.com"
            ),
            "AWS_ACCESS_KEY_ID": "access-key-id-not-for-network",
            "AWS_SECRET_ACCESS_KEY": "secret-access-key-not-for-network",
            "KVBENCH_R2_PREFIX": "kvbench/sha256",
            "CLOUDFLARE_API_TOKEN": "cloudflare-token-not-for-network",
        }
        self.config = R2Config.from_environment(self.environ)

    def artifact(self, name: str = "artifact", **kwargs: object) -> Path:
        return finalized_artifact(self.base / name, **kwargs)

    def container_e00_artifact(self) -> Path:
        source = next(
            path.parent
            for path in sorted(
                (ROOT / "docs" / "evidence" / "e00").glob(
                    "e00-*/manifest.json"
                )
            )
            if json.loads(path.read_text(encoding="utf-8"))["gate"][
                "aggregate_status"
            ]
            == "PASS"
        )
        stage = self.base / "e00-container-stage"
        stage.mkdir()
        for child in source.iterdir():
            if child.name in {"manifest.json", "checksums.sha256", "COMPLETE"}:
                continue
            destination = stage / child.name
            if child.is_dir():
                shutil.copytree(child, destination)
            else:
                shutil.copy2(child, destination)

        manifest = copy.deepcopy(
            json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        )
        run_id = manifest["run"]["id"]
        digest = "sha256:" + "a" * 64
        evidence_reference = manifest["evidence"]["commands"][0]["stdout"]
        manifest["schema_version"] = "e00-manifest-1.1.0"
        manifest["execution_environment"] = {
            "kind": "measurement_container",
            "verification_status": "PASS",
            "verification_evidence_file_id": manifest[
                "execution_environment"
            ]["verification_evidence_file_id"],
            "performance_claim_eligible": False,
            "performance_ineligibility_reasons": [
                "container_bf16_parity_not_yet_established",
                "hardware_certification_not_benchmark_timing",
            ],
            "container": {
                "runtime": "docker",
                "image_reference": "kvbench-measurement:phase6a",
                "image_config_digest": digest,
                "base_image_digest": digest,
                "image_inspect_validation": evidence_reference,
                "image_inspect": evidence_reference,
                "runtime_container_id": "b" * 64,
                "runtime_inspect_validation": evidence_reference,
                "runtime_inspect": evidence_reference,
                "digest_status": "verified_against_sanitized_image_inspect",
            },
            "container_parity_required_before_e02": True,
        }
        manifest["evidence"]["storage"]["root"] = (
            "artifacts/phase6a/container_g0"
        )
        manifest["evidence"]["storage"]["run_directory"] = (
            f"artifacts/phase6a/container_g0/{run_id}"
        )
        manifest["gate"]["checks"][1]["name"] = (
            "measurement_container_environment_verified"
        )
        final = self.base / run_id
        run_preflight.finalize_stage(
            stage=stage,
            final=final,
            manifest=manifest,
        )
        return final

    def test_content_addressed_keys_are_canonical_and_traversal_is_rejected(
        self,
    ) -> None:
        digest = "1" * 64
        self.assertEqual(
            artifact_object_prefix("kvbench/sha256/", digest),
            f"kvbench/sha256/{digest}/",
        )
        self.assertEqual(
            artifact_object_key(
                "kvbench/sha256", digest, "raw/payload.bin"
            ),
            f"kvbench/sha256/{digest}/raw/payload.bin",
        )
        for path in ("../escape", "/absolute", "a/../b", "a//b", r"a\b"):
            with self.subTest(path=path), self.assertRaises(
                ArtifactValidationError
            ):
                artifact_object_key("kvbench/sha256", digest, path)
        with self.assertRaises(R2ArtifactError):
            artifact_object_prefix("../outside", digest)
        with self.assertRaises(R2ArtifactError):
            artifact_object_prefix("kvbench/sha256", "not-a-digest")

    def test_config_requires_the_normalized_production_prefix(self) -> None:
        trailing = {**self.environ, "KVBENCH_R2_PREFIX": "kvbench/sha256/"}
        self.assertEqual(
            R2Config.from_environment(trailing).prefix,
            "kvbench/sha256",
        )
        for value in ("kvbench/evidence", "kvbench/sha256//"):
            with self.subTest(value=value), self.assertRaises(R2ArtifactError):
                R2Config.from_environment(
                    {**self.environ, "KVBENCH_R2_PREFIX": value}
                )

    def test_config_binds_the_s3_endpoint_to_the_management_account(
        self,
    ) -> None:
        without_endpoint = dict(self.environ)
        del without_endpoint["R2_ENDPOINT"]
        expected = self.environ["R2_ENDPOINT"]
        self.assertEqual(
            R2Config.from_environment(without_endpoint).endpoint,
            expected,
        )
        self.assertEqual(
            R2Config.from_environment(self.environ).endpoint,
            expected,
        )
        for endpoint in (
            "https://example.invalid",
            "https://" + "b" * 32 + ".r2.cloudflarestorage.com",
            expected + ":443",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(
                R2ArtifactError,
                "does not match",
            ):
                R2Config.from_environment(
                    {**self.environ, "R2_ENDPOINT": endpoint}
                )
        for account_id in ("A" * 32, "a" * 31, "not-an-account"):
            with self.subTest(account_id=account_id), self.assertRaisesRegex(
                R2ArtifactError,
                "R2_ACCOUNT_ID is invalid",
            ):
                R2Config.from_environment(
                    {**self.environ, "R2_ACCOUNT_ID": account_id}
                )

    def test_publish_uses_conditional_creation_complete_last_and_exact_republish(
        self,
    ) -> None:
        artifact = validate_local_artifact(
            self.artifact(), environ=self.environ
        )
        client = FakeR2()
        first = publish_artifact(client, self.config, artifact)
        self.assertEqual(set(first.uploaded), set(first.ordered_paths))
        self.assertFalse(first.verified_existing)
        self.assertEqual(first.ordered_paths[-1], "COMPLETE")
        self.assertTrue(client.put_calls)
        self.assertTrue(all(if_none_match == "*" for _, if_none_match, _ in client.put_calls))
        self.assertTrue(client.put_calls[-1][0].endswith("/COMPLETE"))

        second = publish_artifact(client, self.config, artifact)
        self.assertFalse(second.uploaded)
        self.assertEqual(set(second.verified_existing), set(second.ordered_paths))
        self.assertEqual(len(client.put_calls), len(first.ordered_paths))

    def test_local_mutation_after_validation_is_rejected_before_put(self) -> None:
        root = self.artifact("mutation-race")
        artifact = validate_local_artifact(root, environ=self.environ)
        existing_client = FakeR2()
        publish_artifact(existing_client, self.config, artifact)
        existing_put_count = len(existing_client.put_calls)
        payload = root / "raw" / "payload.bin"
        payload.chmod(0o644)
        payload.write_bytes(b"changed after validation\n")

        missing_client = FakeR2()
        for client in (missing_client, existing_client):
            with self.subTest(existing=bool(client.objects)), self.assertRaisesRegex(
                ArtifactValidationError,
                "changed after final validation",
            ):
                publish_artifact(client, self.config, artifact)

        self.assertEqual(missing_client.put_calls, [])
        self.assertEqual(missing_client.objects, {})
        self.assertEqual(len(existing_client.put_calls), existing_put_count)

    def test_existing_mismatched_bytes_are_rejected(self) -> None:
        artifact = validate_local_artifact(
            self.artifact(), environ=self.environ
        )
        client = FakeR2()
        publish_artifact(client, self.config, artifact)
        key = artifact_object_key(
            self.config.prefix, artifact.root_sha256, "raw/payload.bin"
        )
        client.objects[key] = b"different bytes"
        with self.assertRaises(ObjectConflictError):
            publish_artifact(client, self.config, artifact)

    def test_clean_retrieval_revalidates_every_file_and_root_digest(self) -> None:
        artifact = validate_local_artifact(
            self.artifact(), environ=self.environ
        )
        client = FakeR2()
        publish_artifact(client, self.config, artifact)
        destination = self.base / "clean-download"
        result = verify_remote_artifact(
            client,
            self.config,
            artifact.root_sha256,
            destination,
            environ=self.environ,
        )
        self.assertEqual(result.root_sha256, artifact.root_sha256)
        reconstructed = validate_local_artifact(
            destination, environ=self.environ
        )
        self.assertEqual(reconstructed.root_sha256, artifact.root_sha256)
        self.assertFalse(destination.stat().st_mode & stat.S_IWUSR)

    def test_clean_retrieval_accepts_a_new_name_for_writer_artifact(self) -> None:
        run_id = "r2-writer-artifact"
        store = AppendOnlyArtifactStore(self.base / "writer-runs")
        run = store.create(run_id, created_manifest(run_id))
        run.start()
        run.write_bytes("raw/payload.txt", b"writer lifecycle fixture\n")
        source = run.finalize(terminal_manifest(run_id))
        artifact = validate_local_artifact(source, environ=self.environ)
        client = FakeR2()
        publish_artifact(client, self.config, artifact)

        destination = self.base / "clean-destination-name"
        result = verify_remote_artifact(
            client,
            self.config,
            artifact.root_sha256,
            destination,
            environ=self.environ,
        )

        self.assertEqual(result.root_sha256, artifact.root_sha256)
        self.assertNotEqual(destination.name, run_id)
        self.assertFalse(destination.stat().st_mode & stat.S_IWUSR)

    def test_nonrepository_inventory_contract_is_exact(self) -> None:
        invalid_cases = (
            {
                "inventory_extra": {
                    "schema_version": "kvbench-artifact-inventory-9.9.9"
                }
            },
            {"inventory_extra": {"unexpected": True}},
            {
                "inventory_extra": {
                    "excluded_control_files": [
                        "artifact_inventory.json",
                        "checksums.sha256",
                    ]
                }
            },
            {"inventory_item_extra": {"unexpected": True}},
            {"inventory_item_extra": {"role": "   "}},
        )
        for index, arguments in enumerate(invalid_cases):
            with self.subTest(index=index), self.assertRaisesRegex(
                ArtifactValidationError,
                "artifact inventory is invalid",
            ):
                validate_local_artifact(
                    self.artifact(f"invalid-inventory-{index}", **arguments),
                    environ=self.environ,
                )

    def test_finalization_control_run_ids_and_status_must_agree(self) -> None:
        invalid_cases = (
            {"manifest_extra": {"run_id": "different-manifest-run"}},
            {"inventory_extra": {"run_id": "different-inventory-run"}},
            {"completion_extra": {"run_id": "different-completion-run"}},
            {
                "manifest_extra": {"status": "completed"},
                "completion_extra": {"status": "FAIL"},
            },
        )
        for index, arguments in enumerate(invalid_cases):
            with self.subTest(index=index), self.assertRaisesRegex(
                ArtifactValidationError,
                "identities do not agree|statuses do not agree",
            ):
                validate_local_artifact(
                    self.artifact(f"identity-mismatch-{index}", **arguments),
                    environ=self.environ,
                )

    def test_e00_artifacts_reuse_manifest_and_reference_validators(self) -> None:
        root = self.container_e00_artifact()
        with (
            patch(
                "preflight.run_preflight.validate_manifest",
                wraps=run_preflight.validate_manifest,
            ) as manifest_validator,
            patch(
                "preflight.run_preflight.evidence_reference_errors",
                wraps=run_preflight.evidence_reference_errors,
            ) as reference_validator,
        ):
            validate_local_artifact(root, environ={})

        manifest_validator.assert_called_once()
        reference_validator.assert_called_once()
        projected_manifest = reference_validator.call_args.args[1]
        self.assertIn(
            "artifact_inventory.json",
            {
                item["path"]
                for item in projected_manifest["evidence"]["files"]
            },
        )
        original_manifest = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertNotIn(
            "artifact_inventory.json",
            {
                item["path"]
                for item in original_manifest["evidence"]["files"]
            },
        )

        with (
            patch(
                "preflight.run_preflight.validate_manifest",
                return_value=["invalid"],
            ),
            patch(
                "preflight.run_preflight.evidence_reference_errors",
            ) as reference_validator,
            self.assertRaisesRegex(
                ArtifactValidationError,
                "E00 manifest validation failed",
            ),
        ):
            validate_local_artifact(root, environ={})
        reference_validator.assert_not_called()

        with (
            patch(
                "preflight.run_preflight.validate_manifest",
                wraps=run_preflight.validate_manifest,
            ),
            patch(
                "preflight.run_preflight.evidence_reference_errors",
                return_value=["invalid"],
            ),
            self.assertRaisesRegex(
                ArtifactValidationError,
                "E00 evidence-reference validation failed",
            ),
        ):
            validate_local_artifact(root, environ={})

    def test_incomplete_prefix_is_rejected(self) -> None:
        artifact = validate_local_artifact(
            self.artifact(), environ=self.environ
        )
        client = FakeR2()
        publish_artifact(client, self.config, artifact)
        del client.objects[
            artifact_object_key(
                self.config.prefix, artifact.root_sha256, "COMPLETE"
            )
        ]
        with self.assertRaisesRegex(
            ArtifactValidationError, "prefix is incomplete"
        ):
            verify_remote_artifact(
                client,
                self.config,
                artifact.root_sha256,
                self.base / "incomplete-download",
                environ=self.environ,
            )

    def test_unexpected_remote_object_is_rejected(self) -> None:
        artifact = validate_local_artifact(
            self.artifact(), environ=self.environ
        )
        client = FakeR2()
        publish_artifact(client, self.config, artifact)
        client.objects[
            artifact_object_key(
                self.config.prefix, artifact.root_sha256, "unexpected.bin"
            )
        ] = b"unexpected"
        with self.assertRaises(ArtifactValidationError):
            verify_remote_artifact(
                client,
                self.config,
                artifact.root_sha256,
                self.base / "unexpected-download",
                environ=self.environ,
            )

    def test_remote_checksum_tamper_is_rejected_without_using_etag(self) -> None:
        artifact = validate_local_artifact(
            self.artifact(), environ=self.environ
        )
        client = FakeR2()
        publish_artifact(client, self.config, artifact)
        key = artifact_object_key(
            self.config.prefix, artifact.root_sha256, "raw/payload.bin"
        )
        client.objects[key] = b"tampered"
        with self.assertRaises(ArtifactValidationError):
            verify_remote_artifact(
                client,
                self.config,
                artifact.root_sha256,
                self.base / "tampered-download",
                environ=self.environ,
            )

    def test_sigv4_put_sends_if_none_match_wildcard(self) -> None:
        requests: list[object] = []

        def opener(request: object, *, timeout: int) -> FakeResponse:
            self.assertEqual(timeout, 120)
            requests.append(request)
            return FakeResponse()

        client = R2S3Client(
            self.config,
            opener=opener,
            clock=lambda: datetime(2026, 7, 24, tzinfo=timezone.utc),
        )
        client.put_object_if_absent(
            "kvbench/sha256/" + "2" * 64 + "/payload.bin",
            b"payload",
            content_type="application/octet-stream",
        )
        self.assertEqual(len(requests), 1)
        request = requests[0]
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(headers["if-none-match"], "*")
        self.assertIn("if-none-match", headers["authorization"].lower())
        self.assertIn(", SignedHeaders=", headers["authorization"])
        self.assertIn(", Signature=", headers["authorization"])
        self.assertEqual(
            headers["x-amz-content-sha256"], sha256_bytes(b"payload")
        )

    def test_secret_presence_and_redaction_never_return_values(self) -> None:
        statuses = required_variable_status(self.environ)
        self.assertEqual(set(statuses.values()), {"PRESENT"})
        without_endpoint = dict(self.environ)
        del without_endpoint["R2_ENDPOINT"]
        self.assertEqual(
            required_variable_status(without_endpoint)["R2_ENDPOINT"],
            "MISSING",
        )
        serialized = json.dumps(statuses)
        representation = repr(self.config)
        public_identity = json.dumps(self.config.public_identity())
        for name in REQUIRED_VARIABLE_NAMES:
            self.assertIn(name, statuses)
        for value in (
            self.environ["AWS_ACCESS_KEY_ID"],
            self.environ["AWS_SECRET_ACCESS_KEY"],
            self.environ["CLOUDFLARE_API_TOKEN"],
            self.environ["R2_ACCOUNT_ID"],
        ):
            self.assertNotIn(value, serialized)
            self.assertNotIn(value, representation)
        self.assertNotIn(self.environ["R2_ACCOUNT_ID"], public_identity)
        self.assertEqual(
            self.config.public_identity()["endpoint"],
            "https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com",
        )
        raw = "failure " + " ".join(
            self.environ[name]
            for name in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "CLOUDFLARE_API_TOKEN",
                "R2_ACCOUNT_ID",
            )
        )
        redacted = redact_text(raw, self.environ)
        for value in self.environ.values():
            if value in {
                self.environ["R2_BUCKET"],
                self.environ["KVBENCH_R2_PREFIX"],
            }:
                continue
            self.assertNotIn(value, redacted)

    def test_public_output_has_a_final_account_id_redaction_boundary(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            _output(
                {"provider_field": self.environ["R2_ACCOUNT_ID"]},
                environ=self.environ,
            )
        self.assertNotIn(self.environ["R2_ACCOUNT_ID"], output.getvalue())
        self.assertIn("<redacted:R2_ACCOUNT_ID>", output.getvalue())

    def test_local_artifact_with_secret_bytes_or_env_file_is_rejected(self) -> None:
        secret = self.environ["AWS_SECRET_ACCESS_KEY"].encode()
        secret_root = self.artifact("secret-artifact", payload=secret)
        with self.assertRaises(ArtifactValidationError) as caught:
            validate_local_artifact(secret_root, environ=self.environ)
        self.assertNotIn(secret.decode(), str(caught.exception))

        env_root = self.artifact(
            "env-artifact",
            extra_files={".env.backup": b"not-even-a-real-secret"},
        )
        with self.assertRaises(ArtifactValidationError):
            validate_local_artifact(env_root, environ=self.environ)

    def test_manifest_credential_fields_are_rejected(self) -> None:
        for index, name in enumerate(
            (
                "AWS_ACCESS_KEY_ID",
                "AWS_SESSION_TOKEN",
                "CLOUDFLARE_API_TOKEN",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
            )
        ):
            with self.subTest(name=name):
                artifact = self.artifact(
                    f"credential-field-{index}",
                    manifest_extra={name: "redacted-placeholder"},
                )
                with self.assertRaisesRegex(
                    ArtifactValidationError,
                    "prohibited credential material",
                ):
                    validate_local_artifact(artifact, environ={})

    def test_hugging_face_secret_bytes_are_rejected_and_redacted(self) -> None:
        secret_environ = {
            **self.environ,
            "HF_TOKEN": "hf_secret-value-not-for-network",
        }
        artifact = self.artifact(
            "hf-secret-artifact",
            payload=secret_environ["HF_TOKEN"].encode(),
        )
        with self.assertRaises(ArtifactValidationError) as caught:
            validate_local_artifact(artifact, environ=secret_environ)
        self.assertNotIn(secret_environ["HF_TOKEN"], str(caught.exception))
        self.assertNotIn(
            secret_environ["HF_TOKEN"],
            redact_text(
                f"failure {secret_environ['HF_TOKEN']}",
                secret_environ,
            ),
        )

    def test_bucket_lock_verification_is_read_only_and_prefers_exact_rule(
        self,
    ) -> None:
        reader = FakeCloudflare(
            custom_domains=[{"domain": "disabled.example", "enabled": False}],
            rules=[
                {
                    "id": "whole-bucket",
                    "enabled": True,
                    "condition": {"type": "Indefinite"},
                },
                {
                    "id": "exact",
                    "name": "exact-evidence-lock",
                    "enabled": True,
                    "prefix": "kvbench/sha256",
                    "condition": {"type": "Indefinite"},
                },
            ]
        )
        evidence = verify_cloudflare_bucket_lock(self.config, reader)
        self.assertEqual(evidence.rule_id, "exact")
        self.assertEqual(evidence.rule_name, "exact-evidence-lock")
        self.assertEqual(evidence.scope_kind, "exact")
        self.assertEqual(evidence.retention_type, "Indefinite")
        rendered = evidence.to_dict()
        self.assertTrue(rendered["bucket_exists"])
        self.assertEqual(rendered["endpoint_class"], "cloudflare_r2_s3")
        self.assertFalse(rendered["managed_r2_dev_enabled"])
        self.assertFalse(rendered["public_r2_dev"])
        self.assertEqual(rendered["custom_domain_count"], 1)
        self.assertEqual(rendered["enabled_custom_domain_count"], 0)
        self.assertFalse(rendered["public_custom_domain"])
        self.assertEqual(rendered["public_state_result"], "PASS")
        self.assertEqual(rendered["verification_result"], "PASS")
        self.assertEqual(rendered["lock_rule_name"], "exact-evidence-lock")
        self.assertEqual(rendered["lock_prefix"], "kvbench/sha256")
        self.assertNotIn(
            self.environ["R2_ACCOUNT_ID"],
            json.dumps(evidence.to_dict()),
        )
        self.assertTrue(reader.paths)
        self.assertTrue(all("/r2/buckets/" in path for path in reader.paths))
        self.assertEqual(
            reader.checks,
            [
                "bucket_exists",
                "managed_r2_dev_public_access",
                "custom_domain_public_access",
                "bucket_lock",
            ],
        )

    def test_public_bucket_or_insufficient_lock_is_rejected(self) -> None:
        with self.assertRaisesRegex(R2ArtifactError, "public access"):
            verify_cloudflare_bucket_lock(
                self.config, FakeCloudflare(public=True)
            )
        with self.assertRaisesRegex(R2ArtifactError, "no enabled indefinite"):
            verify_cloudflare_bucket_lock(
                self.config,
                FakeCloudflare(
                    rules=[
                        {
                            "id": "finite",
                            "enabled": True,
                            "prefix": "kvbench/sha256/",
                            "condition": {
                                "type": "Age",
                                "maxAgeSeconds": 86400,
                            },
                        }
                    ]
                ),
            )
        for broader_prefix in ("", "kvbench/"):
            with self.subTest(broader_prefix=broader_prefix), self.assertRaisesRegex(
                R2ArtifactError, "exactly matches"
            ):
                verify_cloudflare_bucket_lock(
                    self.config,
                    FakeCloudflare(
                        rules=[
                            {
                                "id": "broader",
                                "enabled": True,
                                "prefix": broader_prefix,
                                "condition": {"type": "Indefinite"},
                            }
                        ]
                    ),
                )

        with self.assertRaisesRegex(R2ArtifactError, "no enabled indefinite"):
            verify_cloudflare_bucket_lock(
                self.config,
                FakeCloudflare(
                    rules=[
                        {
                            "id": "disabled",
                            "enabled": False,
                            "prefix": "kvbench/sha256/",
                            "condition": {"type": "Indefinite"},
                        }
                    ]
                ),
            )

    def test_bucket_and_every_custom_domain_fail_closed(self) -> None:
        with self.assertRaisesRegex(R2ArtifactError, "does not exist"):
            verify_cloudflare_bucket_lock(
                self.config,
                FakeCloudflare(bucket_name="different-bucket"),
            )
        for domains in (
            [{"domain": "public.example", "enabled": True}],
            [{"domain": "malformed.example"}],
            ["not-an-object"],
        ):
            with self.subTest(domains=domains), self.assertRaisesRegex(
                R2ArtifactError,
                "public access|failed validation",
            ):
                verify_cloudflare_bucket_lock(
                    self.config,
                    FakeCloudflare(custom_domains=domains),
                )

    def test_untrusted_s3_xml_error_code_is_not_exposed(self) -> None:
        account_id = self.environ["R2_ACCOUNT_ID"]

        def opener(request: object, *, timeout: int) -> FakeResponse:
            raise urllib.error.HTTPError(
                "https://example.invalid/object",
                412,
                "precondition failed",
                {},
                BytesIO(
                    (
                        "<Error><Code>"
                        f"{account_id}"
                        "</Code></Error>"
                    ).encode()
                ),
            )

        client = R2S3Client(self.config, opener=opener)
        with self.assertRaises(RemoteRequestError) as caught:
            client.get_object_or_none("safe/key")
        self.assertEqual(caught.exception.code, "Unspecified")
        self.assertNotIn(account_id, str(caught.exception))

    def test_verify_output_is_emitted_before_temporary_cleanup(self) -> None:
        events: list[str] = []
        temporary_path = self.base / "tracked-temporary"
        lock = verify_cloudflare_bucket_lock(
            self.config, FakeCloudflare()
        )

        class TrackingTemporaryDirectory:
            def __init__(self, *, prefix: str) -> None:
                self.prefix = prefix

            def __enter__(self) -> str:
                temporary_path.mkdir()
                events.append("enter")
                return str(temporary_path)

            def __exit__(self, *args: object) -> None:
                events.append("cleanup")

        def fake_verify(*args: object, **kwargs: object) -> object:
            events.append("verify")
            return SimpleNamespace(
                to_dict=lambda: {"verification_result": "PASS"}
            )

        def fake_output(payload: object) -> None:
            self.assertNotIn("cleanup", events)
            events.append("output")

        with (
            patch(
                "scripts.r2_artifact.R2Config.from_environment",
                return_value=self.config,
            ),
            patch(
                "scripts.r2_artifact.verify_cloudflare_bucket_lock",
                return_value=lock,
            ),
            patch("scripts.r2_artifact.R2S3Client", return_value=object()),
            patch(
                "scripts.r2_artifact.verify_remote_artifact",
                side_effect=fake_verify,
            ),
            patch(
                "scripts.r2_artifact.tempfile.TemporaryDirectory",
                TrackingTemporaryDirectory,
            ),
            patch("scripts.r2_artifact._output", side_effect=fake_output),
        ):
            self.assertEqual(main(["verify", "3" * 64]), 0)

        self.assertEqual(events, ["enter", "verify", "output", "cleanup"])

    def test_cloudflare_http_error_body_cannot_enter_exception(self) -> None:
        token = self.environ["CLOUDFLARE_API_TOKEN"]
        account_id = self.environ["R2_ACCOUNT_ID"]

        def opener(request: object, *, timeout: int) -> FakeResponse:
            raise urllib.error.HTTPError(
                "https://api.cloudflare.com/client/v4/example",
                403,
                "forbidden",
                {},
                BytesIO(
                    _json_bytes(
                        {
                            "success": False,
                            "errors": [
                                {
                                    "code": 10000,
                                    "message": (
                                        f"invalid token {token} for account "
                                        f"{account_id}"
                                    ),
                                }
                            ],
                        }
                    )
                ),
            )

        client = CloudflareReadClient(
            token,
            opener=opener,
            redaction_environment=self.environ,
        )
        with self.assertRaises(RemoteRequestError) as caught:
            client.get_json(
                f"/accounts/{account_id}/r2/buckets/example",
                check="bucket_exists",
            )
        self.assertEqual(caught.exception.status, 403)
        self.assertEqual(caught.exception.code, "10000")
        self.assertEqual(caught.exception.failed_check, "bucket_exists")
        self.assertIn(
            "<redacted:R2_ACCOUNT_ID>",
            caught.exception.failed_path or "",
        )
        self.assertIn(
            "<redacted:CLOUDFLARE_API_TOKEN>",
            caught.exception.provider_message or "",
        )
        self.assertNotIn(token, str(caught.exception))
        self.assertNotIn(account_id, str(caught.exception))

    def test_management_evidence_survives_a_later_s3_failure(self) -> None:
        output = StringIO()
        lock = verify_cloudflare_bucket_lock(self.config, FakeCloudflare())
        with (
            patch(
                "scripts.r2_artifact.R2Config.from_environment",
                return_value=self.config,
            ),
            patch(
                "scripts.r2_artifact.verify_cloudflare_bucket_lock",
                return_value=lock,
            ),
            patch("scripts.r2_artifact.R2S3Client", return_value=object()),
            patch(
                "scripts.r2_artifact.verify_remote_artifact",
                side_effect=RemoteRequestError(
                    status=503,
                    code="ServiceUnavailable",
                ),
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(main(["verify", "4" * 64]), 1)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "FAIL")
        self.assertEqual(
            payload["bucket_lock"]["verification_result"],
            "PASS",
        )
        self.assertTrue(payload["bucket_lock"]["bucket_exists"])
        self.assertFalse(payload["bucket_lock"]["public_r2_dev"])
        self.assertFalse(payload["bucket_lock"]["public_custom_domain"])
        self.assertEqual(
            payload["remote_error"]["provider_error_code"],
            "ServiceUnavailable",
        )

    def test_nonempty_retrieval_destination_is_rejected(self) -> None:
        artifact = validate_local_artifact(
            self.artifact(), environ=self.environ
        )
        client = FakeR2()
        publish_artifact(client, self.config, artifact)
        destination = self.base / "not-empty"
        destination.mkdir()
        (destination / "existing").write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(ArtifactValidationError, "must be empty"):
            verify_remote_artifact(
                client,
                self.config,
                artifact.root_sha256,
                destination,
                environ=self.environ,
            )
        self.assertEqual(
            (destination / "existing").read_text(encoding="utf-8"), "preserve"
        )


REQUIRED_VARIABLE_NAMES = (
    "R2_ACCOUNT_ID",
    "R2_BUCKET",
    "R2_ENDPOINT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "KVBENCH_R2_PREFIX",
    "CLOUDFLARE_API_TOKEN",
)


if __name__ == "__main__":
    unittest.main()

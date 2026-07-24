"""Focused offline tests for the Phase 6A Cloudflare R2 artifact path."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
import urllib.error

from scripts.r2_artifact import (
    ArtifactValidationError,
    CloudflareReadClient,
    ObjectConflictError,
    R2ArtifactError,
    R2Config,
    R2S3Client,
    RemoteRequestError,
    artifact_object_key,
    artifact_object_prefix,
    publish_artifact,
    redact_text,
    required_variable_status,
    sha256_bytes,
    sha256_file,
    validate_local_artifact,
    verify_cloudflare_bucket_lock,
    verify_remote_artifact,
)


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
        rules: list[dict[str, object]] | None = None,
    ) -> None:
        self.public = public
        self.rules = (
            [
                {
                    "id": "phase6a-indefinite",
                    "enabled": True,
                    "prefix": "kvbench/evidence/",
                    "condition": {"type": "Indefinite"},
                }
            ]
            if rules is None
            else rules
        )
        self.paths: list[str] = []

    def get_json(self, path: str) -> dict[str, object]:
        self.paths.append(path)
        if path.endswith("/domains/managed"):
            result: object = {
                "bucketId": "not-recorded",
                "domain": "not-recorded",
                "enabled": self.public,
            }
        elif path.endswith("/domains/custom"):
            result = {"domains": []}
        elif path.endswith("/lock"):
            result = {"rules": self.rules}
        else:
            result = {"name": "phase6a-bucket"}
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
            "AWS_ACCESS_KEY_ID": "access-key-id-not-for-network",
            "AWS_SECRET_ACCESS_KEY": "secret-access-key-not-for-network",
            "KVBENCH_R2_PREFIX": "kvbench/evidence",
            "CLOUDFLARE_API_TOKEN": "cloudflare-token-not-for-network",
        }
        self.config = R2Config.from_environment(self.environ)

    def artifact(self, name: str = "artifact", **kwargs: object) -> Path:
        return finalized_artifact(self.base / name, **kwargs)

    def test_content_addressed_keys_are_canonical_and_traversal_is_rejected(
        self,
    ) -> None:
        digest = "1" * 64
        self.assertEqual(
            artifact_object_prefix("kvbench/evidence/", digest),
            f"kvbench/evidence/{digest}/",
        )
        self.assertEqual(
            artifact_object_key(
                "kvbench/evidence", digest, "raw/payload.bin"
            ),
            f"kvbench/evidence/{digest}/raw/payload.bin",
        )
        for path in ("../escape", "/absolute", "a/../b", "a//b", r"a\b"):
            with self.subTest(path=path), self.assertRaises(
                ArtifactValidationError
            ):
                artifact_object_key("kvbench/evidence", digest, path)
        with self.assertRaises(R2ArtifactError):
            artifact_object_prefix("../outside", digest)
        with self.assertRaises(R2ArtifactError):
            artifact_object_prefix("kvbench/evidence", "not-a-digest")

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
            "kvbench/evidence/" + "2" * 64 + "/payload.bin",
            b"payload",
            content_type="application/octet-stream",
        )
        self.assertEqual(len(requests), 1)
        request = requests[0]
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(headers["if-none-match"], "*")
        self.assertIn("if-none-match", headers["authorization"].lower())
        self.assertEqual(
            headers["x-amz-content-sha256"], sha256_bytes(b"payload")
        )

    def test_secret_presence_and_redaction_never_return_values(self) -> None:
        statuses = required_variable_status(self.environ)
        self.assertEqual(set(statuses.values()), {"PRESENT"})
        serialized = json.dumps(statuses)
        representation = repr(self.config)
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
        artifact = self.artifact(
            "credential-field",
            manifest_extra={"AWS_ACCESS_KEY_ID": "redacted-placeholder"},
        )
        with self.assertRaisesRegex(
            ArtifactValidationError, "prohibited credential material"
        ):
            validate_local_artifact(artifact, environ={})

    def test_bucket_lock_verification_is_read_only_and_prefers_exact_rule(
        self,
    ) -> None:
        reader = FakeCloudflare(
            rules=[
                {
                    "id": "whole-bucket",
                    "enabled": True,
                    "condition": {"type": "Indefinite"},
                },
                {
                    "id": "exact",
                    "enabled": True,
                    "prefix": "kvbench/evidence/",
                    "condition": {"type": "Indefinite"},
                },
            ]
        )
        evidence = verify_cloudflare_bucket_lock(self.config, reader)
        self.assertEqual(evidence.rule_id, "exact")
        self.assertEqual(evidence.scope_kind, "exact")
        self.assertEqual(evidence.retention_type, "Indefinite")
        self.assertTrue(reader.paths)
        self.assertTrue(all("/r2/buckets/" in path for path in reader.paths))

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
                            "prefix": "kvbench/evidence/",
                            "condition": {
                                "type": "Age",
                                "maxAgeSeconds": 86400,
                            },
                        }
                    ]
                ),
            )

    def test_cloudflare_http_error_body_cannot_enter_exception(self) -> None:
        token = self.environ["CLOUDFLARE_API_TOKEN"]

        def opener(request: object, *, timeout: int) -> FakeResponse:
            raise urllib.error.HTTPError(
                "https://api.cloudflare.com/client/v4/example",
                403,
                "forbidden",
                {},
                BytesIO(f"secret={token}".encode()),
            )

        client = CloudflareReadClient(token, opener=opener)
        with self.assertRaises(RemoteRequestError) as caught:
            client.get_json("/accounts/example/r2/buckets/example")
        self.assertNotIn(token, str(caught.exception))

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
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "KVBENCH_R2_PREFIX",
    "CLOUDFLARE_API_TOKEN",
)


if __name__ == "__main__":
    unittest.main()

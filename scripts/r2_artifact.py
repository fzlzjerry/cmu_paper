#!/usr/bin/env python3
"""Publish and verify finalized kvbench artifacts in Cloudflare R2.

This is deliberately provider-specific.  It implements only the two Phase 6A
operations, ``publish`` and ``verify``, and uses the existing finalized
artifact controls as the authority for file identity.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


REGION = "auto"
SERVICE = "s3"
CONTROL_FILES = (
    "manifest.json",
    "artifact_inventory.json",
    "checksums.sha256",
    "COMPLETE",
)
OBJECT_ACCESS_VARIABLES = (
    "R2_ACCOUNT_ID",
    "R2_BUCKET",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "KVBENCH_R2_PREFIX",
)
LOCK_VARIABLE = "CLOUDFLARE_API_TOKEN"
REQUIRED_VARIABLES = (*OBJECT_ACCESS_VARIABLES, LOCK_VARIABLE)
SECRET_VARIABLES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "CLOUDFLARE_API_TOKEN",
    "R2_ACCOUNT_ID",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]")
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_PROHIBITED_JSON_KEYS = frozenset(
    {
        "aws_access_key_id",
        "aws_secret_access_key",
        "cloudflare_api_token",
        "r2_account_id",
        "authorization",
    }
)


class R2ArtifactError(RuntimeError):
    """Base class for safe, redacted Phase 6A R2 failures."""


class ArtifactValidationError(R2ArtifactError):
    """The local or reconstructed artifact is not safely finalized."""


class ObjectConflictError(R2ArtifactError):
    """An existing content-addressed object has different bytes."""


class RemoteRequestError(R2ArtifactError):
    """A provider request failed without retaining a secret-bearing message."""

    def __init__(self, *, status: int | None, code: str) -> None:
        self.status = status
        self.code = code
        status_text = str(status) if status is not None else "unavailable"
        super().__init__(
            f"Cloudflare R2 request failed (HTTP {status_text}, code {code})"
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_variable_status(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return names and PRESENT/MISSING only; values are never returned."""

    source = os.environ if environ is None else environ
    return {
        name: "PRESENT" if bool(source.get(name)) else "MISSING"
        for name in REQUIRED_VARIABLES
    }


def redact_text(
    text: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Redact configured credentials without echoing their values."""

    source = os.environ if environ is None else environ
    redacted = str(text)
    replacements = sorted(
        (
            (value, f"<redacted:{name}>")
            for name in SECRET_VARIABLES
            if (value := source.get(name))
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for value, marker in replacements:
        redacted = redacted.replace(value, marker)
    redacted = re.sub(
        r"(?i)(authorization\s*:\s*(?:bearer|aws4-hmac-sha256)\s+)\S+",
        r"\1<redacted>",
        redacted,
    )
    return redacted


def _require_safe_relative(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ArtifactValidationError("artifact path is not a safe POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ArtifactValidationError("artifact path is not a safe POSIX path")
    return path.as_posix()


def normalize_prefix(value: str) -> str:
    """Return a canonical, non-empty R2 prefix without a trailing slash."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise R2ArtifactError("KVBENCH_R2_PREFIX is invalid")
    if value.startswith(("/", "\\")) or "\\" in value:
        raise R2ArtifactError("KVBENCH_R2_PREFIX is invalid")
    candidate = value[:-1] if value.endswith("/") else value
    if not candidate or candidate.endswith("/"):
        raise R2ArtifactError("KVBENCH_R2_PREFIX is invalid")
    try:
        normalized = _require_safe_relative(candidate)
    except ArtifactValidationError as error:
        raise R2ArtifactError("KVBENCH_R2_PREFIX is invalid") from error
    return normalized


def validate_root_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise R2ArtifactError("root SHA-256 must be 64 lowercase hexadecimal digits")
    return value


def artifact_object_prefix(base_prefix: str, root_sha256: str) -> str:
    return f"{normalize_prefix(base_prefix)}/{validate_root_sha256(root_sha256)}/"


def artifact_object_key(
    base_prefix: str,
    root_sha256: str,
    relative_path: str,
) -> str:
    prefix = artifact_object_prefix(base_prefix, root_sha256)
    relative = _require_safe_relative(relative_path)
    key = f"{prefix}{relative}"
    if not key.startswith(prefix):
        raise R2ArtifactError("object key escaped the configured prefix")
    return key


def _validate_endpoint(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise R2ArtifactError("R2_ENDPOINT must be an HTTPS origin without credentials")
    return f"https://{parsed.netloc}"


@dataclass(frozen=True)
class R2Config:
    endpoint: str
    bucket: str
    prefix: str
    account_id: str = field(repr=False)
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    cloudflare_api_token: str = field(repr=False)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "R2Config":
        source = os.environ if environ is None else environ
        if any(not source.get(name) for name in REQUIRED_VARIABLES):
            raise R2ArtifactError("required R2 variables are missing")
        account_id = source["R2_ACCOUNT_ID"]
        endpoint = source.get(
            "R2_ENDPOINT",
            f"https://{account_id}.r2.cloudflarestorage.com",
        )
        bucket = source["R2_BUCKET"]
        if _BUCKET_RE.fullmatch(bucket) is None:
            raise R2ArtifactError("R2_BUCKET is invalid")
        return cls(
            endpoint=_validate_endpoint(endpoint),
            bucket=bucket,
            prefix=normalize_prefix(source["KVBENCH_R2_PREFIX"]),
            account_id=account_id,
            access_key_id=source["AWS_ACCESS_KEY_ID"],
            secret_access_key=source["AWS_SECRET_ACCESS_KEY"],
            cloudflare_api_token=source["CLOUDFLARE_API_TOKEN"],
        )

    def public_identity(self) -> dict[str, str]:
        return {
            "provider": "cloudflare_r2",
            "endpoint": self.endpoint,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "region": REGION,
        }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite value {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ArtifactValidationError(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{label} is invalid")
    return payload


def _reject_credential_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in _PROHIBITED_JSON_KEYS:
                raise ArtifactValidationError(
                    "artifact contains prohibited credential material"
                )
            _reject_credential_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_credential_keys(child)


def _contains_bytes(path: Path, needle: bytes) -> bool:
    if not needle:
        return False
    overlap = max(0, len(needle) - 1)
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            data = tail + chunk
            if needle in data:
                return True
            tail = data[-overlap:] if overlap else b""
    return False


def _is_env_path(relative: str) -> bool:
    return any(part == ".env" or part.startswith(".env.") for part in PurePosixPath(relative).parts)


@dataclass(frozen=True)
class ArtifactFile:
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ValidatedArtifact:
    directory: Path
    root_sha256: str
    files: tuple[ArtifactFile, ...]

    def by_path(self) -> dict[str, ArtifactFile]:
        return {item.relative_path: item for item in self.files}


def _collect_files(directory: Path, *, require_immutable: bool) -> list[Path]:
    try:
        root_metadata = directory.lstat()
    except OSError as error:
        raise ArtifactValidationError("artifact directory is missing or unsafe") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ArtifactValidationError("artifact directory is missing or unsafe")
    if require_immutable and root_metadata.st_mode & _WRITE_BITS:
        raise ArtifactValidationError("finalized artifact remains writable")

    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        metadata = path.lstat()
        relative = path.relative_to(directory).as_posix()
        _require_safe_relative(relative)
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactValidationError("artifact contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            if require_immutable and metadata.st_mode & _WRITE_BITS:
                raise ArtifactValidationError("finalized artifact remains writable")
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ArtifactValidationError("artifact contains an unsafe file")
        if require_immutable and metadata.st_mode & _WRITE_BITS:
            raise ArtifactValidationError("finalized artifact remains writable")
        if _is_env_path(relative):
            raise ArtifactValidationError("artifact contains a prohibited secret file")
        files.append(path)
    return files


def _parse_checksum_ledger(path: Path) -> dict[str, str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactValidationError("checksum ledger is unreadable") from error
    if raw and not raw.endswith(b"\n"):
        raise ArtifactValidationError("checksum ledger is not canonical")
    entries: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or _SHA256_RE.fullmatch(parts[0]) is None:
            raise ArtifactValidationError("checksum ledger is malformed")
        relative = _require_safe_relative(parts[1])
        if relative in entries:
            raise ArtifactValidationError("checksum ledger contains duplicate paths")
        entries[relative] = parts[0]
    if not entries or list(entries) != sorted(entries):
        raise ArtifactValidationError("checksum ledger is not canonical")
    return entries


def _tree_root_sha256(files: Sequence[ArtifactFile]) -> str:
    canonical = "".join(
        f"{item.sha256}  {item.relative_path}\n"
        for item in sorted(files, key=lambda item: item.relative_path)
    ).encode("utf-8")
    return sha256_bytes(canonical)


def _validate_artifact(
    directory: str | Path,
    *,
    environ: Mapping[str, str] | None,
    require_immutable: bool,
) -> ValidatedArtifact:
    root = Path(directory).absolute()
    files = _collect_files(root, require_immutable=require_immutable)
    relatives = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(set(CONTROL_FILES) - relatives)
    if missing:
        raise ArtifactValidationError("artifact lacks required finalization controls")

    manifest = _load_json_object(root / "manifest.json", "manifest")
    inventory = _load_json_object(
        root / "artifact_inventory.json", "artifact inventory"
    )
    completion = _load_json_object(root / "COMPLETE", "completion marker")
    for payload in (manifest, inventory, completion):
        _reject_credential_keys(payload)

    source = os.environ if environ is None else environ
    secret_values = tuple(
        value.encode("utf-8")
        for name in SECRET_VARIABLES
        if (value := source.get(name))
    )
    for path in files:
        if any(_contains_bytes(path, value) for value in secret_values):
            raise ArtifactValidationError(
                "artifact contains prohibited credential material"
            )

    ledger = _parse_checksum_ledger(root / "checksums.sha256")
    ledger_expected = relatives - {"checksums.sha256", "COMPLETE"}
    if set(ledger) != ledger_expected:
        raise ArtifactValidationError(
            "checksum ledger does not exactly cover artifact files"
        )
    for relative, expected in ledger.items():
        if sha256_file(root / relative) != expected:
            raise ArtifactValidationError("artifact checksum verification failed")

    inventory_files = inventory.get("files")
    if not isinstance(inventory_files, list):
        raise ArtifactValidationError("artifact inventory is invalid")
    declared: dict[str, tuple[str, int]] = {}
    for item in inventory_files:
        if not isinstance(item, Mapping):
            raise ArtifactValidationError("artifact inventory is invalid")
        relative_value = item.get("path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if (
            not isinstance(relative_value, str)
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ArtifactValidationError("artifact inventory is invalid")
        relative = _require_safe_relative(relative_value)
        if relative in declared:
            raise ArtifactValidationError("artifact inventory has duplicate paths")
        declared[relative] = (digest, size)
    inventory_expected = relatives - {
        "artifact_inventory.json",
        "checksums.sha256",
        "COMPLETE",
    }
    if set(declared) != inventory_expected or list(declared) != sorted(declared):
        raise ArtifactValidationError(
            "artifact inventory does not exactly cover payload files"
        )
    for relative, (digest, size) in declared.items():
        target = root / relative
        if target.stat().st_size != size or sha256_file(target) != digest:
            raise ArtifactValidationError("artifact inventory verification failed")

    if completion.get("written_last") is not True:
        raise ArtifactValidationError("COMPLETE is not a valid final marker")
    expected_controls = {
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "artifact_inventory_sha256": sha256_file(
            root / "artifact_inventory.json"
        ),
        "checksum_ledger_sha256": sha256_file(root / "checksums.sha256"),
    }
    if any(completion.get(key) != value for key, value in expected_controls.items()):
        raise ArtifactValidationError("COMPLETE control hashes do not match")
    if completion.get("checksum_ledger_path", "checksums.sha256") != "checksums.sha256":
        raise ArtifactValidationError("COMPLETE references an unexpected ledger")

    if "manifest.initial.json" in relatives and require_immutable:
        try:
            from kvbench.runtime.artifacts import validate_run_directory

            repository_validation = validate_run_directory(root)
        except (ImportError, OSError) as error:
            raise ArtifactValidationError(
                "repository artifact validator is unavailable"
            ) from error
        if not repository_validation.valid or not repository_validation.complete:
            raise ArtifactValidationError(
                "artifact failed repository lifecycle validation"
            )

    records = tuple(
        ArtifactFile(
            relative_path=path.relative_to(root).as_posix(),
            path=path,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in files
    )
    return ValidatedArtifact(
        directory=root.resolve(strict=True),
        root_sha256=_tree_root_sha256(records),
        files=records,
    )


def validate_local_artifact(
    directory: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> ValidatedArtifact:
    """Validate a finalized immutable artifact before any network operation."""

    return _validate_artifact(
        directory,
        environ=environ,
        require_immutable=True,
    )


def publication_order(artifact: ValidatedArtifact) -> tuple[ArtifactFile, ...]:
    by_path = artifact.by_path()
    payload = sorted(
        (
            item
            for item in artifact.files
            if item.relative_path not in CONTROL_FILES
        ),
        key=lambda item: item.relative_path,
    )
    ordered = (
        *payload,
        by_path["manifest.json"],
        by_path["artifact_inventory.json"],
        by_path["checksums.sha256"],
        by_path["COMPLETE"],
    )
    if len({item.relative_path for item in ordered}) != len(artifact.files):
        raise ArtifactValidationError("publication order does not cover artifact")
    return ordered


def _content_type(path: str) -> str | None:
    if path.endswith(".json"):
        return "application/json"
    if path.endswith((".txt", ".sha256")) or path == "COMPLETE":
        return "text/plain"
    guessed, _ = mimetypes.guess_type(path, strict=True)
    return guessed


@dataclass(frozen=True)
class PublicationResult:
    root_sha256: str
    uri: str
    ordered_paths: tuple[str, ...]
    uploaded: tuple[str, ...]
    verified_existing: tuple[str, ...]
    published_at_utc: str

    def to_dict(self) -> dict[str, object]:
        order_bytes = "".join(f"{item}\n" for item in self.ordered_paths).encode()
        return {
            "provider": "cloudflare_r2",
            "root_sha256": self.root_sha256,
            "uri": self.uri,
            "object_count": len(self.ordered_paths),
            "uploaded_count": len(self.uploaded),
            "verified_existing_count": len(self.verified_existing),
            "complete_last": bool(
                self.ordered_paths and self.ordered_paths[-1] == "COMPLETE"
            ),
            "publication_order_sha256": sha256_bytes(order_bytes),
            "published_at_utc": self.published_at_utc,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def publish_artifact(
    client: "R2S3Client",
    config: R2Config,
    artifact: ValidatedArtifact,
) -> PublicationResult:
    """Conditionally create every object, publishing COMPLETE last."""

    uploaded: list[str] = []
    verified_existing: list[str] = []
    ordered = publication_order(artifact)
    for item in ordered:
        key = artifact_object_key(
            config.prefix,
            artifact.root_sha256,
            item.relative_path,
        )
        existing = client.get_object_or_none(key)
        if existing is not None:
            if sha256_bytes(existing) != item.sha256:
                raise ObjectConflictError(
                    "existing content-addressed object has different bytes"
                )
            verified_existing.append(item.relative_path)
            continue
        data = item.path.read_bytes()
        try:
            client.put_object_if_absent(
                key,
                data,
                content_type=_content_type(item.relative_path),
            )
        except RemoteRequestError as error:
            if error.status != 412 and error.code != "ObjectLockedByBucketPolicy":
                raise
            raced = client.get_object_or_none(key)
            if raced is None or sha256_bytes(raced) != item.sha256:
                raise ObjectConflictError(
                    "conditional object creation did not preserve exact bytes"
                ) from error
            verified_existing.append(item.relative_path)
            continue
        retrieved = client.get_object_or_none(key)
        if retrieved is None or sha256_bytes(retrieved) != item.sha256:
            raise R2ArtifactError("uploaded object failed authoritative SHA-256 check")
        uploaded.append(item.relative_path)

    return PublicationResult(
        root_sha256=artifact.root_sha256,
        uri=(
            f"r2://{config.bucket}/"
            f"{artifact_object_prefix(config.prefix, artifact.root_sha256)}"
        ),
        ordered_paths=tuple(item.relative_path for item in ordered),
        uploaded=tuple(uploaded),
        verified_existing=tuple(verified_existing),
        published_at_utc=_utc_now(),
    )


def _ensure_empty_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactValidationError("retrieval destination is unsafe")
        if any(path.iterdir()):
            raise ArtifactValidationError("retrieval destination must be empty")
    else:
        path.mkdir(mode=0o700, parents=False)
    return path.resolve(strict=True)


def _write_downloaded_file(root: Path, relative: str, data: bytes) -> None:
    safe = PurePosixPath(_require_safe_relative(relative))
    current = root
    for part in safe.parent.parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactValidationError("retrieval path is unsafe") from None
    target = root.joinpath(*safe.parts)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as error:
        raise ArtifactValidationError("retrieval produced a duplicate path") from error
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_tree_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


@dataclass(frozen=True)
class RetrievalResult:
    root_sha256: str
    uri: str
    object_count: int
    retrieved_at_utc: str

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": "cloudflare_r2",
            "root_sha256": self.root_sha256,
            "uri": self.uri,
            "object_count": self.object_count,
            "complete_marker_valid": True,
            "inventory_valid": True,
            "checksum_ledger_valid": True,
            "unexpected_objects": False,
            "verification_result": "PASS",
            "retrieved_at_utc": self.retrieved_at_utc,
        }


def verify_remote_artifact(
    client: "R2S3Client",
    config: R2Config,
    root_sha256: str,
    destination: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> RetrievalResult:
    """Download a remote artifact into a clean directory and fully revalidate it."""

    digest = validate_root_sha256(root_sha256)
    remote_prefix = artifact_object_prefix(config.prefix, digest)
    keys = client.list_keys(remote_prefix)
    if len(keys) != len(set(keys)):
        raise ArtifactValidationError("remote listing contains duplicate objects")
    relatives: list[str] = []
    for key in keys:
        if not key.startswith(remote_prefix):
            raise ArtifactValidationError("remote object escaped the configured prefix")
        relative = key[len(remote_prefix) :]
        relatives.append(_require_safe_relative(relative))
    if "COMPLETE" not in relatives:
        raise ArtifactValidationError("remote artifact prefix is incomplete")

    target_root = _ensure_empty_directory(Path(destination).absolute())
    for relative in sorted(relatives, key=lambda item: (item == "COMPLETE", item)):
        key = artifact_object_key(config.prefix, digest, relative)
        data = client.get_object_or_none(key)
        if data is None:
            raise ArtifactValidationError("remote listing changed during retrieval")
        _write_downloaded_file(target_root, relative, data)

    reconstructed = _validate_artifact(
        target_root,
        environ=environ,
        require_immutable=False,
    )
    if reconstructed.root_sha256 != digest:
        raise ArtifactValidationError("reconstructed root digest does not match prefix")
    if set(relatives) != {item.relative_path for item in reconstructed.files}:
        raise ArtifactValidationError("remote prefix contains unexpected objects")
    _make_tree_immutable(target_root)
    immutable = validate_local_artifact(target_root, environ=environ)
    if immutable.root_sha256 != digest:
        raise ArtifactValidationError("immutable retrieval digest changed")
    return RetrievalResult(
        root_sha256=digest,
        uri=f"r2://{config.bucket}/{remote_prefix}",
        object_count=len(relatives),
        retrieved_at_utc=_utc_now(),
    )


def _xml_error_code(data: bytes) -> str:
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return "Unspecified"
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "Code" and element.text:
            return element.text[:80]
    return "Unspecified"


class R2S3Client:
    """Small SigV4 client for the exact R2 object operations Phase 6A needs."""

    def __init__(
        self,
        config: R2Config,
        *,
        opener: Callable[..., Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._opener = urllib.request.urlopen if opener is None else opener
        self._clock = (
            (lambda: datetime.now(timezone.utc)) if clock is None else clock
        )
        parsed = urllib.parse.urlsplit(config.endpoint)
        self._host = parsed.netloc

    @staticmethod
    def _canonical_query(parameters: Sequence[tuple[str, str]]) -> str:
        encoded = [
            (
                urllib.parse.quote(key, safe="-_.~"),
                urllib.parse.quote(value, safe="-_.~"),
            )
            for key, value in parameters
        ]
        return "&".join(f"{key}={value}" for key, value in sorted(encoded))

    def _object_path(self, key: str | None = None) -> str:
        raw = f"/{self._config.bucket}"
        if key is not None:
            raw += f"/{key}"
        return urllib.parse.quote(raw, safe="/-_.~")

    def _authorization_headers(
        self,
        *,
        method: str,
        canonical_path: str,
        canonical_query: str,
        body: bytes,
        extra_headers: Mapping[str, str],
    ) -> dict[str, str]:
        now = self._clock().astimezone(timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        short_date = now.strftime("%Y%m%d")
        payload_hash = sha256_bytes(body)
        headers = {
            "host": self._host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
            **{key.lower(): " ".join(value.split()) for key, value in extra_headers.items()},
        }
        signed_header_names = sorted(headers)
        canonical_headers = "".join(
            f"{name}:{headers[name]}\n" for name in signed_header_names
        )
        signed_headers = ";".join(signed_header_names)
        canonical_request = "\n".join(
            (
                method,
                canonical_path,
                canonical_query,
                canonical_headers,
                signed_headers,
                payload_hash,
            )
        )
        scope = f"{short_date}/{REGION}/{SERVICE}/aws4_request"
        string_to_sign = "\n".join(
            (
                "AWS4-HMAC-SHA256",
                amz_date,
                scope,
                sha256_bytes(canonical_request.encode("utf-8")),
            )
        )
        key_date = hmac.new(
            f"AWS4{self._config.secret_access_key}".encode(),
            short_date.encode(),
            hashlib.sha256,
        ).digest()
        key_region = hmac.new(key_date, REGION.encode(), hashlib.sha256).digest()
        key_service = hmac.new(key_region, SERVICE.encode(), hashlib.sha256).digest()
        signing_key = hmac.new(
            key_service, b"aws4_request", hashlib.sha256
        ).digest()
        signature = hmac.new(
            signing_key, string_to_sign.encode(), hashlib.sha256
        ).hexdigest()
        authorization = (
            "AWS4-HMAC-SHA256 "
            f"Credential={self._config.access_key_id}/{scope},"
            f"SignedHeaders={signed_headers},Signature={signature}"
        )
        output = {name: value for name, value in headers.items() if name != "host"}
        output["Authorization"] = authorization
        return output

    def _request(
        self,
        method: str,
        *,
        key: str | None = None,
        query: Sequence[tuple[str, str]] = (),
        body: bytes = b"",
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        canonical_path = self._object_path(key)
        canonical_query = self._canonical_query(query)
        extra = {} if headers is None else dict(headers)
        signed = self._authorization_headers(
            method=method,
            canonical_path=canonical_path,
            canonical_query=canonical_query,
            body=body,
            extra_headers=extra,
        )
        url = f"{self._config.endpoint}{canonical_path}"
        if canonical_query:
            url += f"?{canonical_query}"
        request = urllib.request.Request(
            url,
            data=body if method == "PUT" else None,
            headers=signed,
            method=method,
        )
        try:
            with self._opener(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            try:
                response_body = error.read(65536)
            except OSError:
                response_body = b""
            raise RemoteRequestError(
                status=error.code,
                code=_xml_error_code(response_body),
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RemoteRequestError(status=None, code="TransportError") from None

    def get_object_or_none(self, key: str) -> bytes | None:
        try:
            return self._request("GET", key=key)
        except RemoteRequestError as error:
            if error.status == 404:
                return None
            raise

    def put_object_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None,
    ) -> None:
        headers = {"If-None-Match": "*"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        self._request("PUT", key=key, body=data, headers=headers)

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        continuation: str | None = None
        seen_tokens: set[str] = set()
        while True:
            query = [("list-type", "2"), ("prefix", prefix)]
            if continuation is not None:
                query.append(("continuation-token", continuation))
            raw = self._request("GET", query=query)
            try:
                root = ET.fromstring(raw)
            except ET.ParseError as error:
                raise RemoteRequestError(
                    status=200, code="MalformedListResponse"
                ) from error
            truncated = False
            next_token: str | None = None
            for element in root.iter():
                name = element.tag.rsplit("}", 1)[-1]
                if name == "Contents":
                    for child in element:
                        if child.tag.rsplit("}", 1)[-1] == "Key" and child.text:
                            keys.append(child.text)
                elif name == "IsTruncated":
                    truncated = (element.text or "").lower() == "true"
                elif name == "NextContinuationToken":
                    next_token = element.text
            if not truncated:
                return keys
            if not next_token or next_token in seen_tokens:
                raise RemoteRequestError(
                    status=200, code="InvalidContinuationToken"
                )
            seen_tokens.add(next_token)
            continuation = next_token


class CloudflareReadClient:
    """Read-only Cloudflare management API client for bucket certification."""

    def __init__(
        self,
        api_token: str,
        *,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._api_token = api_token
        self._opener = urllib.request.urlopen if opener is None else opener

    def get_json(self, path: str) -> dict[str, Any]:
        if not path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise R2ArtifactError("Cloudflare API path is unsafe")
        request = urllib.request.Request(
            f"https://api.cloudflare.com/client/v4{path}",
            headers={"Authorization": f"Bearer {self._api_token}"},
            method="GET",
        )
        try:
            with self._opener(request, timeout=60) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            raise RemoteRequestError(
                status=error.code, code="CloudflareAPIError"
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RemoteRequestError(status=None, code="TransportError") from None
        if len(raw) > 4 * 1024 * 1024:
            raise RemoteRequestError(status=200, code="OversizedAPIResponse")
        try:
            payload = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RemoteRequestError(status=200, code="MalformedAPIResponse") from error
        if (
            not isinstance(payload, dict)
            or payload.get("success") is not True
            or not isinstance(payload.get("result"), dict)
        ):
            raise RemoteRequestError(status=200, code="RejectedAPIResponse")
        return payload


@dataclass(frozen=True)
class BucketLockEvidence:
    bucket: str
    endpoint: str
    rule_id: str
    covered_prefix: str
    rule_prefix: str
    scope_kind: str
    retention_type: str
    verified_at_utc: str

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": "cloudflare_r2",
            "bucket": self.bucket,
            "endpoint": self.endpoint,
            "bucket_public": False,
            "lock_rule_id": self.rule_id,
            "lock_rule_name": self.rule_id,
            "covered_prefix": self.covered_prefix,
            "lock_prefix": self.rule_prefix,
            "lock_scope": self.scope_kind,
            "enabled": True,
            "retention_type": self.retention_type,
            "verified_at_utc": self.verified_at_utc,
        }


def _cloudflare_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("success") is not True or not isinstance(
        payload.get("result"), Mapping
    ):
        raise R2ArtifactError("Cloudflare API response failed validation")
    return payload["result"]


def _lock_scope(rule_prefix: str, evidence_prefix: str) -> str | None:
    exact = f"{evidence_prefix}/"
    if rule_prefix == exact:
        return "exact"
    if rule_prefix == "":
        return "whole_bucket"
    if rule_prefix.endswith("/") and exact.startswith(rule_prefix):
        return "parent_prefix"
    return None


def verify_cloudflare_bucket_lock(
    config: R2Config,
    client: CloudflareReadClient | None = None,
) -> BucketLockEvidence:
    """Read and verify bucket existence, direct public state, and Bucket Lock."""

    reader = (
        CloudflareReadClient(config.cloudflare_api_token)
        if client is None
        else client
    )
    account = urllib.parse.quote(config.account_id, safe="")
    bucket = urllib.parse.quote(config.bucket, safe="")
    base = f"/accounts/{account}/r2/buckets/{bucket}"

    bucket_result = _cloudflare_result(reader.get_json(base))
    if bucket_result.get("name") != config.bucket:
        raise R2ArtifactError("configured R2 bucket does not exist")

    managed = _cloudflare_result(reader.get_json(f"{base}/domains/managed"))
    custom = _cloudflare_result(reader.get_json(f"{base}/domains/custom"))
    domains = custom.get("domains")
    if not isinstance(managed.get("enabled"), bool) or not isinstance(domains, list):
        raise R2ArtifactError("R2 public-access response failed validation")
    if managed["enabled"] or any(
        isinstance(item, Mapping) and item.get("enabled") is True
        for item in domains
    ):
        raise R2ArtifactError("R2 bucket direct public access is enabled")
    if any(
        not isinstance(item, Mapping) or not isinstance(item.get("enabled"), bool)
        for item in domains
    ):
        raise R2ArtifactError("R2 public-access response failed validation")

    lock_result = _cloudflare_result(reader.get_json(f"{base}/lock"))
    rules = lock_result.get("rules")
    if not isinstance(rules, list):
        raise R2ArtifactError("R2 Bucket Lock response failed validation")
    candidates: list[tuple[int, int, str, str, str]] = []
    scope_rank = {"exact": 0, "parent_prefix": 1, "whole_bucket": 2}
    for rule in rules:
        if not isinstance(rule, Mapping) or rule.get("enabled") is not True:
            continue
        rule_id = rule.get("id")
        condition = rule.get("condition")
        raw_prefix = rule.get("prefix", "")
        if (
            not isinstance(rule_id, str)
            or not rule_id
            or not isinstance(condition, Mapping)
            or condition.get("type") != "Indefinite"
            or not isinstance(raw_prefix, str)
        ):
            continue
        scope = _lock_scope(raw_prefix, config.prefix)
        if scope is not None:
            candidates.append(
                (
                    scope_rank[scope],
                    -len(raw_prefix),
                    rule_id,
                    raw_prefix,
                    scope,
                )
            )
    if not candidates:
        raise R2ArtifactError(
            "no enabled indefinite Bucket Lock rule covers the evidence prefix"
        )
    _, _, rule_id, rule_prefix, scope = sorted(candidates)[0]
    return BucketLockEvidence(
        bucket=config.bucket,
        endpoint=config.endpoint,
        rule_id=rule_id,
        covered_prefix=f"{config.prefix}/",
        rule_prefix=rule_prefix,
        scope_kind=scope,
        retention_type="Indefinite",
        verified_at_utc=_utc_now(),
    )


def _output(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish or verify one finalized artifact in Cloudflare R2"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("artifact", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("root_sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    statuses = required_variable_status()
    try:
        config = R2Config.from_environment()
        lock = verify_cloudflare_bucket_lock(config)
        client = R2S3Client(config)
        if arguments.operation == "publish":
            artifact = validate_local_artifact(arguments.artifact)
            result: Mapping[str, object] = publish_artifact(
                client, config, artifact
            ).to_dict()
        else:
            with tempfile.TemporaryDirectory(prefix="kvbench-r2-verify-") as temporary:
                result = verify_remote_artifact(
                    client,
                    config,
                    arguments.root_sha256,
                    Path(temporary) / "artifact",
                ).to_dict()
        _output(
            {
                "status": "PASS",
                "required_variables": statuses,
                "r2": config.public_identity(),
                "bucket_lock": lock.to_dict(),
                arguments.operation: result,
            }
        )
        return 0
    except (R2ArtifactError, OSError, ValueError) as error:
        _output(
            {
                "status": "FAIL",
                "required_variables": statuses,
                "error": redact_text(str(error)),
                "error_type": type(error).__name__,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

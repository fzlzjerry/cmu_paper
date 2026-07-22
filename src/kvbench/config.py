"""Strict loading for JSON-compatible YAML configuration documents."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, TypeAlias

from kvbench.errors import ConfigLoadError, SchemaValidationError
from kvbench.schema import (
    DocumentType,
    ExperimentConfig,
    HardwareManifest,
    MethodConfig,
    ModelIdentity,
    ResolutionState,
    StrictModel,
    canonical_json_bytes,
    sha256_hex,
)


MAX_CONFIG_BYTES = 2 * 1024 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

ConfigDocument: TypeAlias = (
    HardwareManifest | ModelIdentity | MethodConfig | ExperimentConfig
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigLoadError("duplicate object key is not allowed")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ConfigLoadError("non-finite JSON numbers are not allowed")


def load_json_compatible_yaml(path: Path) -> dict[str, Any]:
    """Load the documented dependency-free YAML 1.2 JSON subset."""

    try:
        size = path.stat().st_size
    except OSError as error:
        raise ConfigLoadError("configuration path is not readable") from error
    if size > MAX_CONFIG_BYTES:
        raise ConfigLoadError("configuration exceeds the 2 MiB safety limit")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigLoadError("configuration must be readable UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except ConfigLoadError:
        raise
    except json.JSONDecodeError as error:
        raise ConfigLoadError(
            f"configuration is not valid JSON-compatible YAML at line {error.lineno}"
        ) from error
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ConfigLoadError("configuration root must be an object")
    return value


def parse_config(value: dict[str, Any]) -> ConfigDocument:
    """Dispatch a strict configuration document by its declared type."""

    raw_type = value.get("document_type")
    try:
        document_type = DocumentType(raw_type)
    except (TypeError, ValueError) as error:
        raise SchemaValidationError(
            "missing or invalid document_type", path=("document_type",)
        ) from error
    models: dict[DocumentType, type[StrictModel]] = {
        DocumentType.HARDWARE: HardwareManifest,
        DocumentType.MODEL: ModelIdentity,
        DocumentType.METHOD: MethodConfig,
        DocumentType.EXPERIMENT: ExperimentConfig,
    }
    return models[document_type].from_dict(value)


def load_config(path: str | Path) -> ConfigDocument:
    return parse_config(load_json_compatible_yaml(Path(path)))


@dataclasses.dataclass(frozen=True, slots=True)
class ExperimentBundle:
    """A plan and all referenced typed identities, loaded as one contract."""

    plan_path: Path
    plan: ExperimentConfig
    hardware: HardwareManifest
    model: ModelIdentity
    methods: tuple[MethodConfig, ...]
    canonical_fingerprints: tuple[tuple[str, str], ...]
    blockers: tuple[str, ...]

    @property
    def execution_ready(self) -> bool:
        return not self.blockers


def _resolve_reference(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=False)
    resolved_root = root.resolve(strict=True)
    if not candidate.is_relative_to(resolved_root):
        raise ConfigLoadError("configuration reference escapes repository root")
    return candidate


def _load_reference(
    root: Path,
    relative: str,
    expected_sha256: str | None,
) -> tuple[ConfigDocument, str]:
    document = load_config(_resolve_reference(root, relative))
    fingerprint = sha256_hex(canonical_json_bytes(document))
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise SchemaValidationError(
            "referenced configuration fingerprint does not match",
            path=(relative,),
        )
    return document, fingerprint


def load_experiment_bundle(
    plan_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> ExperimentBundle:
    """Load a plan and verify every typed repository-relative reference."""

    root = Path(repository_root).resolve(strict=True)
    path = Path(plan_path)
    if not path.is_absolute():
        path = root / path
    plan_document = load_config(path)
    if not isinstance(plan_document, ExperimentConfig):
        raise SchemaValidationError("--plan must reference an experiment document")

    hardware_document, hardware_sha = _load_reference(
        root, plan_document.hardware.path, plan_document.hardware.sha256
    )
    if not isinstance(hardware_document, HardwareManifest):
        raise SchemaValidationError("hardware reference has the wrong document type")
    model_document, model_sha = _load_reference(
        root, plan_document.model.path, plan_document.model.sha256
    )
    if not isinstance(model_document, ModelIdentity):
        raise SchemaValidationError("model reference has the wrong document type")

    methods: list[MethodConfig] = []
    fingerprints: list[tuple[str, str]] = [
        (plan_document.hardware.path, hardware_sha),
        (plan_document.model.path, model_sha),
    ]
    blockers = set(plan_document.resolution.blockers)
    blockers.update(plan_document.admission.blockers)
    blockers.update(plan_document.software_environment.resolution.blockers)
    blockers.update(hardware_document.resolution.blockers)
    blockers.update(model_document.resolution.blockers)
    for selection in plan_document.methods:
        loaded, fingerprint = _load_reference(
            root, selection.config.path, selection.config.sha256
        )
        if not isinstance(loaded, MethodConfig):
            raise SchemaValidationError("method reference has the wrong document type")
        known_variants = {variant.variant_id for variant in loaded.variants}
        missing = sorted(set(selection.variants) - known_variants)
        if missing:
            raise SchemaValidationError(
                "selected variant is absent from its method config",
                path=(selection.config.path, "variants"),
            )
        methods.append(loaded)
        fingerprints.append((selection.config.path, fingerprint))
        blockers.update(loaded.resolution.blockers)
        blockers.update(loaded.implementation.resolution.blockers)
        for variant in loaded.variants:
            if variant.variant_id in selection.variants:
                blockers.update(variant.resolution.blockers)

    for reference in (
        plan_document.experiment_contract,
        plan_document.measurement_protocol,
    ):
        referenced_path = _resolve_reference(root, reference.path)
        try:
            raw = referenced_path.read_bytes()
        except OSError as error:
            raise ConfigLoadError("referenced contract is not readable") from error
        digest = sha256_hex(raw)
        if reference.sha256 is not None and digest != reference.sha256:
            raise SchemaValidationError(
                "referenced contract fingerprint does not match",
                path=(reference.path,),
            )
        fingerprints.append((reference.path, digest))

    if plan_document.resolution.status is ResolutionState.RESOLVED and blockers:
        raise SchemaValidationError("resolved plan cannot retain admission blockers")
    return ExperimentBundle(
        plan_path=path.resolve(strict=True),
        plan=plan_document,
        hardware=hardware_document,
        model=model_document,
        methods=tuple(methods),
        canonical_fingerprints=tuple(sorted(fingerprints)),
        blockers=tuple(sorted(blockers)),
    )


def validate_all_example_configs(
    *, repository_root: str | Path = REPOSITORY_ROOT
) -> tuple[tuple[str, str], ...]:
    """Validate every versioned example and return stable fingerprints."""

    root = Path(repository_root).resolve(strict=True)
    paths = sorted((root / "configs").glob("*/*.yaml"))
    if not paths:
        raise ConfigLoadError("no example configurations were found")
    results: list[tuple[str, str]] = []
    for path in paths:
        document = load_config(path)
        if isinstance(document, ExperimentConfig):
            load_experiment_bundle(path, repository_root=root)
        results.append((path.relative_to(root).as_posix(), document.fingerprint()))
    return tuple(results)

"""Deterministic command reconstruction from saved manifest fields."""

from __future__ import annotations

from kvbench.errors import ProvenanceError
from kvbench.schema import RunManifest


def _parsed_manifest(manifest: object) -> RunManifest:
    if isinstance(manifest, RunManifest):
        return manifest
    if isinstance(manifest, dict):
        return RunManifest.from_dict(manifest)
    to_dict = getattr(manifest, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return RunManifest.from_dict(result)
    raise TypeError("manifest must be a mapping or expose to_dict()")


def reconstruct_command(manifest: object) -> tuple[str, ...]:
    """Rebuild the declared dry-run command without consulting ambient state."""

    parsed = _parsed_manifest(manifest)
    if parsed.plan_source.kind.value != "path":
        raise ProvenanceError(
            "only a saved path config source is reconstructable in Phase 2",
        )
    if parsed.plan_source.path is None:
        raise ProvenanceError(
            "manifest.plan_source.path must be a non-empty string",
        )
    return parsed.command.argv

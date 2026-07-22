"""Exact offline model/tokenizer identity verification and loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

from kvbench.runtime.backend import (
    ATTENTION_IMPLEMENTATION,
    BackendUnsupportedError,
    register_transformers_attention,
)


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
DEFAULT_SNAPSHOT = Path(
    "/root/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-8B-Instruct/"
    f"snapshots/{MODEL_REVISION}"
)
TRANSFORMERS_VERSION = "4.57.6"
EXPECTED_HASHES = {
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "generation_config.json": "189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd",
    "model.safetensors.index.json": "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b",
    "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}


class ModelIdentityError(RuntimeError):
    """The local checkpoint differs from the frozen identity."""


class ModelAccessError(RuntimeError):
    """The exact gated checkpoint is not locally accessible."""


@dataclass(frozen=True, slots=True)
class FrozenModelIdentity:
    """Fully verified identity used by performance and later quality work."""

    model_id: str
    revision: str
    snapshot_path: str
    file_hashes: dict[str, str]
    architecture: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    weight_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "snapshot_path": self.snapshot_path,
            "file_hashes": dict(sorted(self.file_hashes.items())),
            "architecture": self.architecture,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "weight_dtype": self.weight_dtype,
        }


@dataclass(frozen=True, slots=True)
class LoadedFrozenModel:
    """Loaded exact checkpoint and tokenizer plus verified identity."""

    model: Any
    tokenizer: Any
    identity: FrozenModelIdentity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelIdentityError(f"invalid frozen JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise ModelIdentityError(f"frozen JSON artifact is not an object: {path.name}")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ModelIdentityError(f"frozen model field mismatch: {label}")


def verify_frozen_snapshot(
    snapshot_path: str | Path = DEFAULT_SNAPSHOT,
) -> FrozenModelIdentity:
    """Hash every required byte and validate the frozen Llama geometry."""

    snapshot = Path(snapshot_path).expanduser().resolve(strict=False)
    if not snapshot.is_dir():
        raise ModelAccessError("the exact frozen model snapshot is absent")
    if snapshot.name != MODEL_REVISION:
        raise ModelIdentityError("snapshot directory is not the frozen revision")
    observed: dict[str, str] = {}
    for name, expected_hash in EXPECTED_HASHES.items():
        path = snapshot / name
        if not path.is_file():
            raise ModelAccessError(f"required frozen model artifact is absent: {name}")
        digest = sha256_file(path)
        if digest != expected_hash:
            raise ModelIdentityError(f"SHA-256 mismatch for frozen artifact: {name}")
        observed[name] = digest

    config = _load_json(snapshot / "config.json")
    _require_equal(config.get("architectures"), ["LlamaForCausalLM"], "architectures")
    _require_equal(config.get("model_type"), "llama", "model_type")
    _require_equal(config.get("hidden_size"), 4096, "hidden_size")
    _require_equal(config.get("num_hidden_layers"), 32, "num_hidden_layers")
    _require_equal(config.get("num_attention_heads"), 32, "num_attention_heads")
    _require_equal(config.get("num_key_value_heads"), 8, "num_key_value_heads")
    _require_equal(config.get("max_position_embeddings"), 131072, "max_position_embeddings")
    _require_equal(config.get("rope_theta"), 500000.0, "rope_theta")
    _require_equal(config.get("torch_dtype"), "bfloat16", "torch_dtype")
    _require_equal(config.get("sliding_window"), None, "sliding_window")
    rope = config.get("rope_scaling")
    if not isinstance(rope, dict):
        raise ModelIdentityError("rope_scaling is not a mapping")
    for key, expected in {
        "rope_type": "llama3",
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    }.items():
        _require_equal(rope.get(key), expected, f"rope_scaling.{key}")

    index = _load_json(snapshot / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelIdentityError("weight index has no weight_map")
    declared_shards = {
        value for value in weight_map.values() if isinstance(value, str)
    }
    expected_shards = {
        name for name in EXPECTED_HASHES if name.endswith(".safetensors")
    }
    _require_equal(declared_shards, expected_shards, "weight shards")
    return FrozenModelIdentity(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        snapshot_path=str(snapshot),
        file_hashes=observed,
        architecture="LlamaForCausalLM",
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=131072,
        weight_dtype="bfloat16",
    )


def load_frozen_model(
    snapshot_path: str | Path = DEFAULT_SNAPSHOT,
    *,
    device: str = "cuda:0",
) -> LoadedFrozenModel:
    """Load only the hash-verified local snapshot with all network paths closed."""

    identity = verify_frozen_snapshot(snapshot_path)
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise ModelAccessError(
            "the locked Phase 3 model-loading dependency target is unavailable"
        ) from error
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ModelIdentityError("Transformers version differs from 4.57.6")
    register_transformers_attention()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    snapshot = identity.snapshot_path
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=ATTENTION_IMPLEMENTATION,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ModelAccessError("exact local model/tokenizer loading failed") from error
    model.to(device=torch.device(device), dtype=torch.bfloat16)
    model.eval()
    model.requires_grad_(False)
    if model.__class__.__name__ != "LlamaForCausalLM":
        raise ModelIdentityError("loaded architecture is not LlamaForCausalLM")
    config = model.config
    for name, expected in {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "max_position_embeddings": 131072,
    }.items():
        _require_equal(getattr(config, name, None), expected, name)
    _require_equal(
        config._attn_implementation,
        ATTENTION_IMPLEMENTATION,
        "attention implementation",
    )
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ModelIdentityError("loaded model parameter dtype is not uniformly BF16")
    if len(tokenizer) != 128256:
        raise ModelIdentityError("loaded tokenizer vocabulary size differs")
    return LoadedFrozenModel(model=model, tokenizer=tokenizer, identity=identity)


def require_local_model() -> FrozenModelIdentity:
    """Validate the default offline snapshot without loading framework objects."""

    try:
        return verify_frozen_snapshot(DEFAULT_SNAPSHOT)
    except BackendUnsupportedError:
        raise

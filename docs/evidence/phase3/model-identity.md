# Phase 3 primary model identity evidence

- Recorded: 2026-07-22
- Repository: `meta-llama/Llama-3.1-8B-Instruct`
- Exact revision: `0e9e39f249a16976918f6564b8830bc894c89659`
- Local snapshot: `/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`
- Network policy for execution: offline/local-files-only
- Access state: manually gated repository access verified; no credential value
  was printed, copied, or stored in the repository

## Byte identity

The snapshot contains exactly the 11 requested files. No `.incomplete` file
remained after acquisition. The pinned Hugging Face 0.36.2 cache scanner
registered one 16.1 GB repository revision with 11 files at the exact commit.
That CLI release has no `cache verify` subcommand, so byte verification used
SHA-256 over every resolved snapshot file.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `LICENSE` | 7,627 | `64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369` |
| `config.json` | 855 | `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e` |
| `generation_config.json` | 184 | `189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd` |
| `model.safetensors.index.json` | 23,950 | `146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b` |
| `model-00001-of-00004.safetensors` | 4,976,698,672 | `2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668` |
| `model-00002-of-00004.safetensors` | 4,999,802,720 | `09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15` |
| `model-00003-of-00004.safetensors` | 4,915,916,176 | `fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa` |
| `model-00004-of-00004.safetensors` | 1,168,138,808 | `92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b` |
| `special_tokens_map.json` | 296 | `6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec` |
| `tokenizer.json` | 9,085,657 | `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4` |
| `tokenizer_config.json` | 55,351 | `177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424` |

## Deterministic offline load

With `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, local-files-only loading,
Transformers 4.57.6, and BF16 dtype, the exact snapshot loaded as:

```text
architecture: LlamaForCausalLM
parameter count: 8,030,261,248
parameter dtype set: {torch.bfloat16}
layers: 32
query heads: 32
KV heads: 8
head dimension: 128
maximum positions: 131072
vocabulary/tokenizer size: 128256
embedding shape: [128256, 4096]
LM-head shape: [128256, 4096]
tokenizer class: PreTrainedTokenizerFast
```

The load did not access the cached base checkpoint and did not substitute a
model, tokenizer, backend, dtype, or revision. Model weights remain outside
Git. This evidence resolves B-004 only; B-009 and B-010 remain open.

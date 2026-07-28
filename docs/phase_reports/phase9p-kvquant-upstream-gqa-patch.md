# Phase 9P — KVQuant upstream Llama-3.1 GQA compatibility patch

Status: **PASS**

Date: 2026-07-28

## Scope and entry

Phase 9P started from clean main-project commit
`f2c6475f09cdf6e9660552eb23c91b03e386aa59`, the documented Phase 8 PASS
descendant that repairs only the KIVI admission-validator authority binding.
The Phase 9 BLOCKED report remains an immutable conversation handoff; there is
no repository path to rewrite. Its six source and compatibility blockers were
treated as entry evidence, not silently replaced.

This phase changed no BF16, TurboQuant, KIVI, Measurement Container, existing
adapter, historical evidence, performance artifact, profiler artifact, or
quality artifact. It did not run full Phase 9 calibration, create 4/3/2-bit
calibration artifacts, start Phase 10, or enable KVQuant in the explicit method
factory.

## Authority

Decision 0020 establishes:

- repository: `https://github.com/SqueezeAILab/KVQuant.git`;
- base commit: `57a238357f0ffe50084670fcd5781c9848f80ea2`;
- base tree: `094e0f736f77ee327e5350cbd1eefb1c936aa77b`;
- method identifier: `kvquant_gqa_upstream_patch_v1`;
- name: **KVQuant-GQA patched upstream**;
- local branch: `kvbench/llama31-gqa`;
- patched commit: `4ad80bc8c942d0a05516d2be8f8d443a77a05900`;
- patched tree: `c4f1490c9c0c4ec46099f1e95c092516df2adb4e`;
- aggregate patch SHA-256:
  `db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6`.

The clean private checkout is retained at
`/home/rockrock/third_party_worktrees/kvquant-gqa`. Repository-wide licensing
and embedded/adapted-source lineage remain unresolved. This project does not
claim official author-released GQA support or redistribution authority. The
modified checkout, patch contents, source archive, Docker source layer, and
extension were not committed to this repository or uploaded to R2.

## Frozen target and compatibility

The patch supports only `meta-llama/Llama-3.1-8B-Instruct` and its tokenizer at
revision `0e9e39f249a16976918f6564b8830bc894c89659`. The checksum-bound local
snapshot manifest has SHA-256
`ab9f6a32a41934c9e49881db68022827b6aca35f4f644627c77e3420978d1336`.
The exact config, tokenizer, and BF16 eager 32-layer model loaded with networking
disabled. Native Llama-3.1 `llama3` RoPE was retained; legacy linear
substitution was rejected.

The validated geometry is hidden size 4096, 32 query heads, 8 KV heads, four
query groups per KV head, head dimension 128, KV width 1024, and maximum context
131072. Calibration hooks capture pre-RoPE Keys and Values in native eight-head
geometry for all 32 layers. The full Fisher and quantizer entry points require
the same model authority and frozen 16-example, 2048-token policy. Model forward
is BF16, Fisher accumulation is FP32, and fitting activations are FP16. A
one-layer small-sequence Fisher and 4-bit NUQ quantizer smoke passed; no full
calibration was executed.

The project-defined sparse policy uses tail fraction 0.005, six entries from
each tail, and a shared Key/Value cap of 12 for 4-, 3-, and 2-bit operation.
Selection is deterministic by value then flat index, with no duplicate or
overlapping index, `float32` values, `int32` indices, and zero padding. This cap
is not an author-provided default. The sink policy is exactly five initial FP16
K/V positions in native eight-head storage, excluded from quantized and sparse
history.

Persistent deployment K/V remains native eight-head data. Each query head maps
to `query_head / 4`; MHA remains the groups=1 special case. The exact eager path
contains no `repeat_kv`, `repeat_interleave`, physical K/V expansion,
query-head-sized K/V temporary, or full-prefix dequantization. Unsupported
geometry and alternate attention implementations fail closed.

## Isolated validation

The source was mounted read-only into
`kvbench-phase9p-validation@sha256:16ee632c5ac029deca5859f3da4c74f9e5e55f5080c10745a59653e95d8e5b44`,
which derives from the unchanged authorized Measurement Container. Modern
calibration used Transformers 4.57.6/tokenizers 0.22.2. Vendored deployment ran
in a separate fresh process with Transformers 4.38.0.dev0/tokenizers 0.15.2.

The canonical extension SHA-256 is
`280496acd435361245d349b7af92210ea1d9eca7488873523116f7a84b087a71`.
It contains an `sm_120` cubin and compute-120 PTX (PTX ISA 9.0). The following
focused checks passed:

- 29 modern loader, geometry, hook, Fisher, quantizer, and outlier tests with
  every opt-in exact-loader/integration test enabled;
- 8 vendored deployment source and cross-stack RoPE tests;
- native 4/3/2-bit dense/sparse GQA and MHA numerical controls;
- native cap-reached sparse controls;
- CUDA Graph capture/replay and zero replay allocation for every changed
  kernel path;
- extension-only forced PTX/JIT with CUDA cache disabled;
- exact full-model vendored loader smoke joined to the final extension; and
- separate Compute Sanitizer MHA, native GQA, and cap-reached runs, each with
  `ERROR SUMMARY: 0 errors`.

The complete compact identities, file hashes, commands, environment, and test
results are in `docs/evidence/phase9p/patch-manifest.json` and
`docs/evidence/phase9p/test-report.json`.

## Gates and non-claims

- G0: PASS
- G1: PASS
- G2-TQ: PASS
- G2-KIVI: PASS
- G2-KVQ: NOT EVALUATED
- Global G2-G5: NOT EVALUATED
- Full Scan: CLOSED
- Quality execution: LOCKED
- `PERFORMANCE_DATA_FROZEN`: absent

Phase 9P establishes only patched-upstream compatibility and reproducible
source/test authority. It makes no KVQuant accuracy, speedup, timing, physical
HBM, capacity, knee, memory-benefit, PPL, LongBench, or quality claim.

## Next action

A separate task may restart full Phase 9 calibration using exactly the patched
commit/tree and frozen GQA/cap/sink policy above. Phase 10 must not start until
that separately authorized Phase 9 task completes.

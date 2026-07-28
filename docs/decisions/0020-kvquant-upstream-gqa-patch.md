# Decision 0020: KVQuant upstream Llama-3.1 GQA compatibility patch

- Status: Accepted
- Date: 2026-07-28
- Authority: explicit Phase 9P authorization, AGENTS.md,
  CODEX_WORKFLOW.md, and the frozen primary-model contract
- Supersedes: no prior KVQuant source-authority decision
- Superseded by: none

## Context

The author-maintained KVQuant repository is pinned to commit
`57a238357f0ffe50084670fcd5781c9848f80ea2`, tree
`094e0f736f77ee327e5350cbd1eefb1c936aa77b`, at
`https://github.com/SqueezeAILab/KVQuant`. The immutable Phase 9 handoff found
that this revision rejects Llama-3.1 RoPE, forces an unsuitable FP16 boundary,
rejects native GQA deployment, uses an MHA-specific 42-slot sparse row, and
does not provide deterministic sparse tie-breaking.

No repository-wide root license or file-level redistribution authority for
the KVQuant-specific and adapted source has been verified. The source may be
used only as a local/private research input under this decision. This record
does not grant redistribution authority.

## Decision

1. The project will modify the exact pinned implementation directly in one
   local checkout and branch. An unofficial fork, floating branch, or model
   substitution is not authorized.
2. The patched authority is the pinned upstream KVQuant revision plus one
   checksum-bound project GQA compatibility patch. Its method identifier is
   `kvquant_gqa_upstream_patch_v1` and its human-readable name is
   **KVQuant-GQA patched upstream**.
3. This authority is not an official author-released GQA implementation. All
   future technical records and papers must retain the patched-upstream label.
4. The patch must preserve the upstream method's pre-RoPE per-channel Key
   quantization, per-token Value quantization, non-uniform quantization,
   Fisher/sensitivity weighting, dense-and-sparse decomposition,
   attention-sink FP16 retention, and fixed-cap sparse storage.
5. The permitted compatibility changes are limited to:

   - exact modern loading for
     `meta-llama/Llama-3.1-8B-Instruct` at revision
     `0e9e39f249a16976918f6564b8830bc894c89659`, including its tokenizer;
   - native Llama-3.1 `llama3` RoPE without linear substitution;
   - native 32-query-head, 8-KV-head, four-group, 128-dimensional-head
     geometry;
   - a deterministic geometry-aware fixed outlier cap;
   - deterministic sparse tie-breaking; and
   - explicit BF16 model-forward, FP32 Fisher/codebook, FP16 fitting, and FP16
     attention-sink boundaries.

6. Persistent K/V data remains native `H_KV=8`. Query head `h` addresses KV
   head `h // 4`; implementation-side `repeat_kv`, physical K/V expansion, and
   query-head-sized persistent or temporary K/V storage are forbidden.
7. The project-defined capped-outlier generalization uses native
   `kv_width = 8 * 128 = 1024`, tail fraction `0.005`, six entries per tail,
   and a shared Key/Value capacity of 12 for 4-, 3-, and 2-bit variants. This
   value is not an author-provided default.
8. Lower-tail ordering is value ascending then flat index ascending. Upper-tail
   ordering is value descending then flat index ascending. The stored order is
   lower then upper, with no overlap or duplicate index, `float32` values,
   `int32` indices, and zero padding. The numeric cap is a validated integer
   policy, not a Boolean switch.
9. The frozen attention-sink policy is five initial K/V positions in FP16,
   stored at native KV-head geometry and excluded from dense quantized history
   and sparse selection.
10. Core Fisher, NUQ, packing, lookup-table, scale, zero-point, and sparse
    correction mathematics are not authorized to change. The cap
    generalization, deterministic tie policy, exact RoPE interpretation, and
    GQA address mapping are explicit project compatibility semantics.
11. Because root licensing and adapted-source lineage remain unresolved, the
    modified checkout, Git patch, source archive, patch hunks, and Docker source
    layer must remain local/private. They must not be committed to the main
    project, uploaded to R2, or published to any public artifact store until a
    later decision resolves licensing authority.
12. Compact project records may contain only source and patched commit/tree
    identities, changed paths, per-file before/after hashes, aggregate patch
    digest, environment and extension digests, commands, and test evidence.

## Consequences

Phase 9P can establish only a checksum-bound local compatibility authority.
It does not create a KVQuant reference lane, Measurement Adapter, calibration
bundle, G2-KVQ result, performance result, HBM result, capacity result, or
quality result. Full Phase 9 calibration remains a separate task and may use
the patched authority only after every Phase 9P loader, geometry, numerical,
CUDA, graph, allocation, PTX/JIT, and sanitizer check passes.

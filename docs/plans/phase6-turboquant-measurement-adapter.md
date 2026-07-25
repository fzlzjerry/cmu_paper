# Phase 6 TurboQuant Measurement Adapter

## Authority and scope

Phase 6 starts at clean commit
`e06f638f4b913f9bd1be2975a478657f5bf2338e`. CUDA execution is authorized
only in Measurement Container
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
The image will not be rebuilt, mutated, or supplemented. Pilot, Full Scan,
profiling, fitting, figures, quality evaluation, KIVI, KVQuant, and Phase 7
remain out of scope.

## Source integration

The authority is vLLM `v0.25.1`, commit
`752a3a504485790a2e8491cacbb35c137339ad34`, tree
`3ec7a4eb00f9bc8fec399bea6cf7de27a7936372`. The container has neither vLLM
nor its package-level dependencies, so direct package import is not possible.
Carry only these pinned source bodies and record their upstream paths, Git blob
IDs, and SHA-256 values in a local provenance manifest:

- `turboquant/config.py`: `TQ_PRESETS`, `TurboQuantConfig`;
- `turboquant/centroids.py`: `solve_lloyd_max`, `get_centroids`;
- `triton_turboquant_store.py`: `_tq_fused_store_mse`;
- `triton_turboquant_decode.py`: `_tq_decode_stage1` and its layout helpers;
- `triton_decode_attention.py`: `_fwd_kernel_stage2`;
- the serving-path `_build_hadamard` body from `turboquant_attn.py`.

Keep kernel and algorithmic bodies unchanged. A narrow compatibility module
may replace vLLM package imports with the container's pinned Torch/Triton APIs.
Because the public launchers allocate rotated-query/store intermediates, the
adapter will preallocate those tensors and invoke the exact pinned kernels.
This is allocation plumbing only; any required algorithmic change blocks the
phase and requires a decision record.

## Adapter and cache

Add one `TurboQuantMethodAdapter`, one `TurboQuantStaticCache`, one narrow
Phase 6 session facade, and one explicit factory mapping. The same adapter
supports `turboquant_4bit_nc`, `turboquant_k3v4_nc`, and
`turboquant_3bit_nc`; `turboquant_k8v4` remains held out. Unknown methods,
KIVI, KVQuant, unsupported presets, and unsupported geometry fail closed.

The dense 32-layer layout is fixed:

- BF16 layers: `0, 1, 30, 31`, using the unchanged BF16 cache/forced-Flash
  path;
- compressed layers: `2..29`, using one combined packed K/V tensor;
- block size: 16, with a preallocated block count, contiguous deterministic
  block table, precomputed slot mappings, and pointer-stable tensors;
- exact slot sizes: 134, 118, and 102 bytes per KV head/token respectively.

Expose active, allocated, and rounded capacity; block count/table; slot
mapping; layer lists; slot size; and a deterministic layout fingerprint.
Reuse the existing fixed-L and growing-context runners through the session
boundary. The runner has no TurboQuant-specific branch. Phase 3 retains its
16-step default while Phase 6 explicitly requests four growing steps.

## Workspace and accounting

Preallocate and reuse centroids, midpoint boundaries, Hadamard matrices,
rotated key/query buffers, normalization buffers, store input buffers,
decode split workspace, LSE, output, sequence lengths, mappings, and skipped
layer BF16 storage before warmup. Record temporary peak separately.

For each preset, report compressed key/value payload, key norm metadata,
value scale/zero metadata, alignment padding, block rounding, skipped-layer
BF16 K/V, and persistent workspace. The persistent breakdown must equal
adapter-owned storage exactly; predicted-versus-allocated error must be below
1%. Record logical BF16 bytes, allocated bytes, `r_nominal`, and `r_alloc`.
Leave `r_hbm` null.

## Conformance and admission

Replay all three mandatory Phase 5 fixtures at their frozen geometry and
require exact input, store, append, appended-slot, slot-size, and byte-layout
agreement. Freeze decode and eager/graph agreement at
`atol=0.02, rtol=0.02`; this tolerance may not be loosened after any admission
run. Reject NaN or Inf.

Complete imports, Triton compilation, any tuning, centroid construction,
mappings, outputs, workspace, skipped-layer storage, and exact-shape warmup
before measured execution or graph capture. Reuse the existing execution-path,
allocation, graph, artifact, and supervision machinery. Audit kernel families
`_tq_fused_store_mse`, `_tq_decode_stage1`, and `_fwd_kernel_stage2`, tensor
geometry, ordering, no full-prefix dequantization, no GQA materialization,
no host synchronization, no cache growth, no fallback, and stable pointers.
Run the existing sanitizer contract once per mandatory preset.

Only after fixture, CUDA, path, allocation, graph, and sanitizer checks pass,
run the bounded non-claim-bearing grid:

- all mandatory presets: fixed-L `B=1,L=128`, eager and CUDA Graph;
- `turboquant_4bit_nc`: fixed-L `B=1,L=4096`, eager and CUDA Graph;
- `turboquant_4bit_nc`: growing context `B=1,L=128,O=4`, eager.

Every manifest records the method/config fingerprints, pinned source and
adapter identities, fixture set, cache layout, slot/layer policy, backend,
graph mode, and authorized image digest. No speedup is computed.

Create one TurboQuant `MethodAdmissionReport`. G2-TQ can pass only when all
three mandatory presets satisfy every frozen criterion. Global G2-G5 remain
not evaluated, Full Scan remains closed, quality remains locked, and
`PERFORMANCE_DATA_FROZEN` remains absent.

After local evidence is finalized and validated, publish the complete bundle
with the existing R2 tool using conditional writes and `COMPLETE` last.
Retrieve into a new empty directory and verify every object and checksum.
Record the content-addressed URI, root digest, Bucket Lock identity, and
verification timestamp. Phase 7 is explicitly deferred to a separate task.

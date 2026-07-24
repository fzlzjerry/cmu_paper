# Phase 6 TurboQuant Measurement Adapter Plan

## Scope and authority

Phase 6 adds one explicit TurboQuant adapter to the existing
`KVCacheMethod` boundary. It reuses the existing fixed-L, growing-context,
correctness, allocation, CUDA Graph, path-audit, artifact, and admission
infrastructure. It does not add a runner, server, scheduler, plugin system, or
general cache framework.

Authority is vLLM `v0.25.1` at commit
`752a3a504485790a2e8491cacbb35c137339ad34`, tree
`3ec7a4eb00f9bc8fec399bea6cf7de27a7936372`, plus the checksum-bound Phase 5
fixture set.

## Upstream source integration

The frozen Measurement Lane has no importable `vllm` package, so direct import
is unavailable. Carry only the source needed by the adapter, with upstream
license headers, original blob hashes, and line provenance:

- `TurboQuantConfig` and preset layout from
  `vllm/model_executor/layers/quantization/turboquant/config.py`;
- Lloyd-Max centroids from
  `vllm/model_executor/layers/quantization/turboquant/centroids.py`;
- `_tq_fused_store_mse` from
  `vllm/v1/attention/ops/triton_turboquant_store.py`;
- `_tq_decode_stage1` from
  `vllm/v1/attention/ops/triton_turboquant_decode.py`;
- `_fwd_kernel_stage2` from
  `vllm/v1/attention/ops/triton_decode_attention.py`.

Keep those algorithmic kernels unchanged. Isolate import adaptation and
preallocated-buffer dispatch in a small Measurement Lane bridge. Prefill may
use the source-equivalent launcher outside measurement. Measured one-token
store and decode use the same formulas and exact upstream kernels with
preallocated FP32 rotation/normalization, split-KV, output, and LSE buffers.
Any fixture-byte drift or numerical-semantic change stops Phase 6 and requires
a decision record; it is not repaired by changing the algorithm.

## Cache state and full-model policy

- Block size: 16.
- Dense attention layers: 0 through 31.
- BF16 skipped layers: 0, 1, 30, 31.
- Compressed layers: 2 through 29.
- Compressed storage per layer:
  `[total_blocks, 16, 8, slot_size]` `uint8`, with combined packed K/V slots.
- Skipped storage: separate static BF16 K and V tensors.
- Mapping: `ceil(capacity / 16)` blocks per request, contiguous deterministic
  request-local block ranges, one precomputed block table, and precomputed
  slot mappings.
- Capacity: rounded to whole blocks; active context remains distinct from
  allocated capacity.
- Workspace: one shared, pointer-stable store scratch set and one shared
  split-KV decode workspace because layers execute serially. Centroids,
  Hadamard matrices, sequence metadata, output, and LSE are constructed before
  warmup or capture.

The adapter fingerprint records the selected configuration, exact compressed
and BF16 layer lists, upstream source identities, backend identity, block
mapping, slot size, fixture-set SHA, and cache-layout fingerprint.

## Conformance, accounting, and graph strategy

For `turboquant_4bit_nc`, `turboquant_k3v4_nc`, and
`turboquant_3bit_nc`, replay the frozen B=1, Hq=32, Hkv=8, D=128,
store-17/append-1 fixtures. Require exact store, append, appended-slot, slot
size, checksum, and byte-layout matches. Freeze decode tolerance in a concise
decision before the smoke grid.

Persistent accounting separates compressed key data, value data, key norms,
value scales, value zero points, compressed alignment, skipped BF16 K/V,
block-rounding capacity, metadata, centroids, and persistent workspace.
The category sum must equal adapter-owned storage exactly; temporary peak is
reported separately; predicted-versus-allocated error must be below 1%.
Report `r_nominal` and `r_alloc`, never `r_hbm`.

Warm the exact fixed-L shape after all compilation and allocations, then use
the existing CUDA Graph harness. Capture and replay must use stable cache,
mapping, centroid, workspace, and output pointers, agree with eager execution
at the frozen tolerance, and produce zero replay allocation. Growing-context
graph support is deferred.

## Bounded admission

After fixture, unit, CUDA, path, allocation, graph, and one-operation
Compute Sanitizer checks pass for all mandatory configurations, run only:

- each mandatory configuration: fixed-L B=1, L=128, eager and CUDA Graph;
- `turboquant_4bit_nc`: fixed-L B=1, L=4096, eager and CUDA Graph;
- `turboquant_4bit_nc`: growing-context B=1, L=128, O=4, eager.

Use new immutable run IDs. This is native-host admission evidence with no
speedup calculation and no formal performance eligibility.

## Expected files

- add the minimal carried-source module, source manifest, cache state, adapter,
  Phase 6 tests, admission command, tolerance decision, and Phase 6 report;
- modify only the explicit adapter factory, the smallest generic common-cache
  seams needed by both BF16 and TurboQuant, method configuration, Makefile,
  and current governance indexes;
- preserve all Phase 5 fixtures and all Phase 4, Phase 3, and E00 evidence.

## Deferred

Phase 7 KIVI work, KVQuant, held-out `turboquant_k8v4` admission, Pilot, Full
Scan, profiler collection, fitting, figures, performance claims, and quality
execution remain out of scope.

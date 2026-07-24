# TurboQuant source note

Status: Phase 5 source authority and deterministic reference fixtures
validated on SM120. Not admitted for benchmarking or quality execution.

## Paper

- Title: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
- Local input: literature/raw/2504.19874.pdf
- Version: arXiv:2504.19874v1, 2025-04-28
- SHA-256: 431eb13926e10491f5fbd0bebd0813c51bd6c1e884426a1500c5db640b2997ab
- Paper-reported hardware: a single NVIDIA A100 GPU.
- The paper does not identify an author-owned implementation repository.

## Pinned reference authority

- Repository: https://github.com/vllm-project/vllm
- Release: v0.25.1
- Commit: 752a3a504485790a2e8491cacbb35c137339ad34
- Commit date: 2026-07-12T16:40:12-07:00
- Git tree: 3ec7a4eb00f9bc8fec399bea6cf7de27a7936372
- License: Apache-2.0, verified from the root `LICENSE`.
- Role: authoritative implementation for the Phase 5 TurboQuant/vLLM
  Reference Lane plus inspected source for a later Measurement Lane adapter.
- Authority boundary: upstream vLLM is an official vLLM project source, but
  there is no evidence that it is the paper authors' official repository.
  Fixtures therefore establish conformance to this pinned vLLM implementation,
  not to every algorithmic variant described by the paper.

## Algorithm and current source semantics

The paper normalizes and rotates vectors, then applies optimized scalar
codebooks for MSE or inner-product distortion. The inspected vLLM candidate
uses a Hadamard rotation and Lloyd-Max scalar quantization for MSE-key modes,
stores a per-vector FP16 key norm, uses uniform per-vector value quantization,
and supports optional norm correction. Source comments trace this implementation
to DRIVE/EDEN/HIGGS-style work predating the TurboQuant paper, and it explicitly
omits QJL.

That implementation choice must not be silently equated with every algorithmic
variant in the paper. A decision record is required before changing the chosen
reference semantics or substituting another implementation.

## Verified vLLM cache dtype names

| Name | Key storage | Value storage | Norm correction | Planned role |
|---|---|---|---|---|
| turboquant_4bit_nc | 4-bit MSE indices + FP16 norm | 4-bit uniform + FP16 scale/zero | yes | main regression |
| turboquant_k3v4_nc | 3-bit MSE indices + FP16 norm | 4-bit uniform + FP16 scale/zero | yes | main regression |
| turboquant_3bit_nc | 3-bit MSE indices + FP16 norm | 3-bit uniform + FP16 scale/zero | yes | main regression |
| turboquant_k8v4 | FP8 key | 4-bit uniform + FP16 scale/zero | no | held-out validation |

The FP8-key path is categorically different from the MSE-key paths and cannot
be treated as one continuous bitwidth curve without an explicit path feature.

## Layout and metadata

For head dimension D in the inspected source:

- MSE key data bytes: ceil(D times key_bits / 8)
- MSE key metadata: one FP16 vector norm
- FP8 key data bytes: D
- Value data bytes: ceil(D times value_bits / 8)
- Value metadata: one FP16 scale plus one FP16 zero
- Slot: key bytes followed by value bytes, padded to an even byte count
- Dense models may skip the first and last two layers for aggressive presets;
  skipped-layer policy must be fixed in config and included in byte accounting.

There is no residual window, sink-token region, or sparse-outlier structure in
this source candidate. Codebooks and rotation matrices are method metadata and
their allocation/lifetime must still be accounted for separately.

## Reference kernels and graph path

Inspected paths at the pinned commit:

- vllm/model_executor/layers/quantization/turboquant/config.py
- vllm/v1/attention/ops/triton_turboquant_store.py
- vllm/v1/attention/ops/triton_turboquant_decode.py
- vllm/v1/attention/backends/turboquant_attn.py
- tests/quantization/test_turboquant.py

The backend declares uniform-batch CUDA Graph support and reserves decode
workspace. The fixed-L decode path uses a fused Triton split-KV attention path.
The continuation-prefill fallback can fully dequantize cached K/V and create
full-size FP16 buffers. That fallback is outside the intended fixed-L decode
measurement path and must be proven absent there; it cannot be accepted by
assumption.

## Porting and admission risks

1. The vLLM reference authority is fixed, but equivalence to every paper
   variant remains outside the demonstrated scope.
2. The small official reference path executed on Blackwell/RTX PRO 6000;
   Measurement Lane compatibility and admission remain untested.
3. Decode performs query conversion/rotation operations whose allocation
   behavior must be measured under eager and graph replay.
4. The source includes full-prefix dequantization for large continuation
   prefill, so execution-path auditing must distinguish runner kinds.
5. GQA indexing appears native in the decode kernel through query-head to
   KV-head mapping, but non-materialization must be demonstrated by trace and
   allocation tests.
6. Layer-skipping changes effective cache bytes and must be included in the
   method fingerprint; it cannot be inferred from nominal bitwidth.
7. Current source unit tests are not project golden fixtures and do not replace
   numerical reference, Compute Sanitizer, graph replay, or allocation gates.

## Phase 5 reference fixture result

The isolated reference environment is recorded in
`reference/turboquant/environment.json` with its complete
`python-freeze.txt`; the installed vLLM runtime files match the hashes of the
pinned Git tree. CUDA was available on the NVIDIA RTX PRO 6000 Blackwell,
compute capability 12.0, using driver 595.71.05, PyTorch 2.11.0+cu130, CUDA
13.0, Triton 3.6.0, and vLLM 0.25.1. This reference environment is separate
from the Measurement Lane and does not resolve B-010.

The frozen fixture geometry is batch 1, 32 query heads, 8 KV heads, head
dimension 128, a 17-token store, one append, block size 16, seed 20260724, and
BF16 inputs. The official vLLM store function and compressed-cache decode
function produced these per-head, per-token layouts:

| Cache dtype | Packed key | Packed value | Key norm | Scale | Zero | Padding | Slot |
|---|---:|---:|---:|---:|---:|---:|---:|
| turboquant_4bit_nc | 64 | 64 | 2 | 2 | 2 | 0 | 134 |
| turboquant_k3v4_nc | 48 | 64 | 2 | 2 | 2 | 0 | 118 |
| turboquant_3bit_nc | 48 | 48 | 2 | 2 | 2 | 0 | 102 |
| turboquant_k8v4 | 128 | 64 | 0 | 2 | 2 | 0 | 196 |

For each fixture, the source-derived slot multiplied by the allocated storage
shape agrees with the actual packed cache file size. Store, append-slot, and
decode outputs plus checksums are under `reference/turboquant/fixtures/`.
Repeated generation in the same frozen environment returned
`verified_existing`: all bytes matched and the finalized fixture set was not
replaced.

Kernel-name-only `torch.profiler` traces identify `_tq_fused_store_mse` for
the three mandatory configurations, `_tq_fused_store_fp8` for the held-out
configuration, and `_tq_decode_stage1` plus `_fwd_kernel_stage2` for decode.
No `_tq_full_dequant_kv`, GQA repeat/materialization name, or backend fallback
was observed in this minimal direct path. The trace retains no duration and
cannot support a latency or physical-HBM claim. Source-declared CUDA Graph
support is `AttentionCGSupport.UNIFORM_BATCH`; direct graph smoke was not
exercised because the minimal API would require a separate graph harness, so
unified capture/replay remains Phase 6 work.

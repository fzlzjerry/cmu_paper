# TurboQuant source note

Status: Phase 0 source audit; not admitted for benchmarking.

## Paper

- Title: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
- Local input: literature/raw/2504.19874.pdf
- Version: arXiv:2504.19874v1, 2025-04-28
- SHA-256: 431eb13926e10491f5fbd0bebd0813c51bd6c1e884426a1500c5db640b2997ab
- Paper-reported hardware: a single NVIDIA A100 GPU.
- The paper does not identify an author-owned implementation repository.

## Pinned source candidate

- Repository: https://github.com/vllm-project/vllm
- Release: v0.25.1
- Commit: 752a3a504485790a2e8491cacbb35c137339ad34
- Role: candidate Reference Lane plus source for a later Measurement Lane adapter.
- Authority caveat: upstream vLLM is an official vLLM project source, but Phase 0
  found no evidence that it is the paper authors' official reference
  implementation. Phase 5 must resolve or explicitly accept this distinction.

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

1. Reference authority and exact equivalence to the paper are unresolved.
2. Blackwell/RTX PRO 6000 compatibility is untested; no G0 evidence exists.
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

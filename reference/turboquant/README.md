# TurboQuant reference lane

This directory freezes the TurboQuant implementation in official vLLM release
`v0.25.1` at commit `752a3a504485790a2e8491cacbb35c137339ad34`.
It is a correctness/reference lane, not the Measurement Lane. It calls the
pinned upstream store and compressed-cache decode functions directly; it does
not contain a local TurboQuant algorithm implementation.

Run the complete reference command from the repository root:

```bash
make reference-turboquant
```

The command verifies the exact Git commit/tree and relevant source blobs,
selects or creates `.reference/turboquant-v0.25.1`, checks the complete package
freeze and CUDA/SM120 identity, regenerates all fixtures in a temporary
directory, validates them, and atomically publishes only when no finalized set
exists. If the committed fixture set already exists, exact regeneration is
verified and the existing bytes are left untouched. A difference is a hard
failure; there is no overwrite flag.

Validation without CUDA execution is:

```bash
make validate-reference-turboquant
```

The digest-pinned alternative container definition is
`docker/reference-turboquant.Dockerfile`. It uses a source checkout at the exact
commit and an isolated Python environment. It does not alter the active
Measurement Lane and does not resolve B-010.

## Frozen fixture contract

The input geometry is batch 1, 32 query heads, 8 KV heads, head dimension 128,
17 stored tokens, one append, block size 16, and seed `20260724`, using BF16
inputs generated on CPU. Each configuration directory contains the packed
cache after store and append, the appended slot, the official decode output, a
strict manifest, a checksum ledger, and a kernel-name-only reference trace.
All binary tensors are raw contiguous bytes with dtype, shape, byte size, and
SHA-256 recorded in the manifest.

The mandatory MSE+NC fixtures are `turboquant_4bit_nc`,
`turboquant_k3v4_nc`, and `turboquant_3bit_nc`. The same pinned official path
also supports the held-out `turboquant_k8v4`, so it is included as one optional
fixture and remains separate from the continuous MSE family.

`torch.profiler` is used only to identify CUDA kernel names. Durations are
discarded. The fixtures contain no benchmark samples, latency fields,
throughput data, quality results, or performance claims. Upstream declares
`AttentionCGSupport.UNIFORM_BATCH`; direct graph smoke is not exercised because
that would require a separate graph harness and is deferred to Phase 6.

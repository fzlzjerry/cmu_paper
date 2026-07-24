# Phase 5 TurboQuant reference plan

Status: frozen before fixture generation on 2026-07-24.

## Authority and environment

- Reference authority: the official vLLM project implementation in
  `https://github.com/vllm-project/vllm.git`.
- Release and source identity: tag `v0.25.1`, commit
  `752a3a504485790a2e8491cacbb35c137339ad34`, commit date
  `2026-07-12T16:40:12-07:00`, tree
  `3ec7a4eb00f9bc8fec399bea6cf7de27a7936372`, Apache-2.0.
- Scope of authority: the pinned TurboQuant/vLLM implementation only. It is
  not asserted to be an author-owned implementation of arXiv:2504.19874 or an
  implementation of every paper variant; the source explicitly omits QJL.
- Container definition: official `vllm/vllm-openai:v0.25.1` image pinned by
  linux/amd64 digest
  `sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`
  (multi-arch manifest
  `sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089`).
- Direct dependency strategy: preserve the official image by digest and record
  a complete package freeze from the selected isolated reference environment.
  The source pins Python 3.12, CUDA 13.0.2, PyTorch 2.11.0, and PyTorch pins
  Triton 3.6.0. No package is installed into the Measurement Lane.
- Source acquisition is by exact commit only. The generator refuses a source
  checkout whose remote, commit, or tree differs from the lock.

## Source-supported configurations

The pinned `TQ_PRESETS`, `CacheDType`, backend declaration, and upstream tests
all contain:

- `turboquant_4bit_nc`: 4-bit Lloyd-Max/MSE keys, 4-bit uniform values,
  FP16 key norm and value scale/zero, norm correction enabled.
- `turboquant_k3v4_nc`: 3-bit Lloyd-Max/MSE keys, 4-bit uniform values,
  FP16 key norm and value scale/zero, norm correction enabled.
- `turboquant_3bit_nc`: 3-bit Lloyd-Max/MSE keys, 3-bit uniform values,
  FP16 key norm and value scale/zero, norm correction enabled.
- `turboquant_k8v4`: FP8 E4M3 keys, 4-bit uniform values, FP16 value
  scale/zero, no key norm, norm correction disabled; held out from the
  continuous MSE bitwidth family.

No configuration differs from preregistration, so no decision record is
needed. The backend accepts FP16/BF16 inputs, block sizes 16/32/64/128, and
declares `UNIFORM_BATCH` CUDA Graph support. The fixture uses block size 16.

## Fixture

- Geometry: batch 1, 32 query heads, 8 KV heads, head dimension 128.
- Sequence: 17-token initial store, one append at position 17, compressed-cache
  decode at total context 18. The odd initial length crosses a 16-token page.
- Inputs: deterministic CPU-generated values copied as BF16 to CUDA.
- Seed: `20260724`.
- Rotation/codebook: the official serving-path Hadamard matrix and official
  Lloyd-Max centroids/midpoints.
- Contents per configuration: manifest, input checksums, packed cache after
  store and append, source-derived and actual byte layout, decode output,
  checksums, append delta, environment/source identity, kernel-name-only
  reference trace, and source-declared graph metadata.
- Trace policy: `torch.profiler` CUDA activities identify kernel names only;
  durations are discarded and no latency claim is permitted.
- Graph policy: record the upstream `UNIFORM_BATCH` declaration. A direct graph
  smoke is attempted only if the minimal official API supports it without a
  new graph harness; otherwise it is explicitly deferred to Phase 6.

## Files and commands

Create:

- `docker/reference-turboquant.Dockerfile`
- `reference/turboquant/README.md`
- `reference/turboquant/bootstrap_environment.py`
- `reference/turboquant/generate_fixtures.py`
- `reference/turboquant/validate_fixtures.py`
- `reference/turboquant/environment.json`
- `reference/turboquant/python-freeze.txt`
- `reference/turboquant/source_manifest.json`
- `reference/turboquant/fixtures/<configuration>/`
- focused Phase 5 unit tests and two Make targets.

Primary command: `make reference-turboquant`.

Validation-only command: `make validate-reference-turboquant`.

Generation validates the frozen source/environment, runs the official store,
append, and decode APIs, stages all fixture directories, and promotes them
with no-replace semantics. Validation re-hashes every payload, checks layout
sums and actual storage, and rejects timing fields.

## Tests

Focused tests cover source lock and file hashes, exact configuration names,
manifest/schema/checksum validation, layout arithmetic and actual allocation,
deterministic metadata and output tolerance, no-overwrite behavior,
TurboQuant Measurement Lane rejection, unchanged BF16 regressions, locked
quality/closed Full Scan governance, absence of formal timing, and unchanged
Phase 3/Phase 4/E00 evidence commitments.

## Deferred

Phase 6 owns the TurboQuant Measurement Lane adapter, common protocol
integration, allocation/sanitizer/graph admission, measurement-container
parity, and G2-TQ. Performance scans, broad profiling, latency comparison,
quality evaluation, KIVI/KVQuant abstractions, and any optimization are
explicitly outside Phase 5.

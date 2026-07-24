# Phase 5 TurboQuant Reference Lane report

Status: PASS

Date: 2026-07-24

## Entry

- Starting HEAD: `9eeabe787060e84c20cd7f88da8f7bca68eae1d4`.
- Working tree: clean before entry validation.
- Phase 4 report: PASS and unchanged.
- G0: PASS.
- G1: PASS.
- Quality execution: LOCKED.
- Full Scan: CLOSED.

The authorized three-file provenance-baseline update at the starting HEAD was
accepted unchanged. The old and new provenance commits have identical trees
and matching subjects. No historical evidence, run ID, artifact checksum, or
phase report was edited.

The first package-lock entry check independently exposed NVIDIA environment
drift: `nvidia-smi` and `nvidia-utils-595` had moved from locked 595.71.05 to
595.84. This was not attributed to the Git author-email rewrite. Phase 5 did
not begin until every locally cached 595.71.05 package was version- and
integrity-checked and the complete installed NVIDIA 595 stack was restored
offline to 595.71.05. No reboot was required. The loaded kernel module,
on-disk module, user-space libraries, and NVML then agreed at 595.71.05;
`nvidia-smi` worked and matched locked SHA-256
`7896b7cdd9cb84b1e0fbc4baf91dfa3039b44ab6641cd08d7cc1dd23f81d6deb`.
`make package-lock-check`, entry `make test`, and entry `make checks` passed.
No package lock was changed and no long-term apt hold or pin was installed.

## Minimal scope

- Plan: `docs/plans/phase5-turboquant-reference.md`.
- General framework added: no; one TurboQuant/vLLM reference lane only.
- Existing Measurement Lane modified: no.
- Phase 6 work started: no.

No plugin system, dynamic discovery, runner framework, artifact lifecycle,
generic container orchestrator, fixture database, profiling framework, or
KIVI/KVQuant abstraction was added. BF16 semantics, adapters, runners, timing
boundaries, model/tokenizer identity, and quality protocols were not changed.

## Pinned reference

- Repository: `https://github.com/vllm-project/vllm.git`.
- Release tag: `v0.25.1`.
- Exact commit: `752a3a504485790a2e8491cacbb35c137339ad34`.
- Exact tree: `3ec7a4eb00f9bc8fec399bea6cf7de27a7936372`.
- Commit date: `2026-07-12T16:40:12-07:00`.
- License: Apache-2.0.
- Relevant source files: `vllm/config/cache.py`, TurboQuant `config.py` and
  `centroids.py`, `triton_turboquant_store.py`,
  `triton_turboquant_decode.py`, `turboquant_attn.py`, and
  `tests/quantization/test_turboquant.py`.
- Source lock: `third_party/LOCK.json`, `third_party/NOTICE.md`, and
  `reference/turboquant/source_manifest.json` bind the commit/tree, exact Git
  blobs, and per-file SHA-256 values.
- Floating branch used: no.

The tag is an official vLLM release. The source explicitly documents lineage
that predates the TurboQuant paper and omits QJL, so the authority is bounded
to this exact vLLM implementation. No full source copy was vendored and no
local algorithm rewrite was used.

## Reference environment

- Definition: isolated `.reference/turboquant-v0.25.1` environment selected by
  the primary command; complete freeze in
  `reference/turboquant/python-freeze.txt`; digest-pinned alternative
  definition in `docker/reference-turboquant.Dockerfile`.
- Base image digest: linux/amd64
  `sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`
  (multi-architecture manifest
  `sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089`).
- Python: 3.12.3.
- PyTorch: 2.11.0+cu130.
- CUDA: runtime 13.0; toolkit package 13.0.2.
- Triton: 3.6.0.
- vLLM: 0.25.1; exact wheel SHA-256
  `16fc7a28df1576eb6f7ca0455026551b8f9adb674c19c66059359ef3e964bd1e`.
- SM120 execution: PASS on NVIDIA RTX PRO 6000 Blackwell Workstation Edition,
  compute capability 12.0, driver 595.71.05. CUDA availability, GPU visibility,
  imports, official store/decode execution, source identity, and absence of CPU
  fallback passed.
- Measurement environment modified: no; this lane does not resolve B-010.

The environment manifest SHA-256 is
`b8a44c6769a17eb3c1de6e3ce129563bca2d338a7702b94e9256d443b89fcdb4`;
the complete freeze SHA-256 is
`5cffaea6de3bc701bf4bf28b53e2432f28b84ba6a735aa572aee38e40cfa3774`.
The installed runtime source files match the pinned Git source bytes.

## Verified configurations

- `turboquant_4bit_nc`: source-supported mandatory MSE+NC preset; K4/V4,
  Lloyd-Max MSE keys, per-vector uniform values, FP16 metadata, norm correction.
- `turboquant_k3v4_nc`: source-supported mandatory MSE+NC preset; K3/V4,
  Lloyd-Max MSE keys, per-vector uniform values, FP16 metadata, norm correction.
- `turboquant_3bit_nc`: source-supported mandatory MSE+NC preset; K3/V3,
  Lloyd-Max MSE keys, per-vector uniform values, FP16 metadata, norm correction.
- `turboquant_k8v4`: source-supported held-out preset on the same path; FP8 E4M3
  keys, V4 per-vector uniform values, FP16 metadata, no norm correction.
- Config differences from preregistration: none.
- Decision record: none required.

All accept FP16/BF16 inputs and block sizes 16, 32, 64, and 128. The backend
accepts any positive head dimension; upstream tests cover 64, 96, 128, and
256. Dense models skip the first two and last two layers; hybrid models do not
apply boundary skips. The held-out k8v4 result is not part of the MSE+NC
continuous bit-width family.

## Fixture geometry

- Batch: 1.
- Query heads: 32.
- KV heads: 8.
- Head dimension: 128.
- Context: 17-token store, one-token append, 18 total; block size 16 and two
  blocks.
- Seed: 20260724.
- Input dtype: BF16, generated deterministically on CPU and copied to CUDA.

## Fixtures

- Fixture root: `reference/turboquant/fixtures`.
- Mandatory fixture count: 3.
- Optional fixture count: 1 (`turboquant_k8v4`).
- Store outputs: exact packed `cache_after_store` binaries and metadata.
- Append outputs: exact packed `cache_after_append` and appended-slot binaries.
- Decode outputs: BF16 tensors from the official compressed-cache decode path.
- Checksums: per-fixture ledgers plus a 34-entry root ledger. Fixture-set
  SHA-256 is
  `774ec946a8839d4de012bc6fba0ee5a933ab1488ecc43354d8573b4481b12f76`;
  root-ledger SHA-256 is
  `d4dbf7933c417a956c3789af404b38aac146f705ec7f8e2a03cad999fc294b38`.
- Existing-fixture overwrite prevention: the first command returned
  `published_new`; a second exact generation returned `verified_existing`
  after byte-for-byte comparison and did not replace any file. A differing
  existing fixture is rejected.

The generator calls official upstream
`triton_turboquant_store` for both store and append and
`triton_turboquant_decode_attention` for decode. It contains no local
quantization, packing, or attention reimplementation.

## Byte layouts

All values below are source-derived bytes per KV head per token. Page and
allocated-cache values use the frozen two-block geometry.

| Configuration | Key bytes | Value bytes | Norm / scale / zero bytes | Padding | Total slot | Page bytes | Actual cache bytes | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `turboquant_4bit_nc` | 64 | 64 | 2 / 2 / 2 | 0 | 134 | 17,152 | 34,304 | PASS |
| `turboquant_k3v4_nc` | 48 | 64 | 2 / 2 / 2 | 0 | 118 | 15,104 | 30,208 | PASS |
| `turboquant_3bit_nc` | 48 | 48 | 2 / 2 / 2 | 0 | 102 | 13,056 | 26,112 | PASS |
| `turboquant_k8v4` | 128 | 64 | 0 / 2 / 2 | 0 | 196 | 25,088 | 50,176 | PASS |

Every byte-breakdown sum equals its slot size. Store-owned bytes equal
`17 * 8 * slot`, append-owned bytes equal `18 * 8 * slot`, and every complete
packed-cache file equals the source allocation formula. Nominal bit width was
not treated as allocation and no physical HBM traffic was estimated.

## Reference trace

- Trace mechanism: `torch.profiler` CUDA activities; kernel names retained,
  all duration fields discarded; `run_kind: reference_trace`.
- Store kernels: `_tq_fused_store_mse` for the mandatory family and
  `_tq_fused_store_fp8` for k8v4, plus recorded norm/divide/cast/fill/GEMM
  auxiliaries where applicable.
- Append kernels: the same official MSE or FP8 store kernel for the one-token
  append.
- Decode kernels: `_tq_decode_stage1` and `_fwd_kernel_stage2`, plus recorded
  auxiliary cast/fill/GEMM kernels where applicable.
- Full-prefix dequantization observed: no kernel-name evidence in this path.
- GQA materialization observed: no; source maps query head to KV head by integer
  group division.
- Backend fallback: none observed.
- Trace used for performance claim: no.

The trace identifies implementation kernels only. It does not prove physical
memory traffic and contains no reportable timing.

## Graph information

- Upstream support: `AttentionCGSupport.UNIFORM_BATCH`.
- Reference graph smoke: not exercised in the minimal direct API because it
  would require a separate graph harness.
- Deferred to Phase 6: unified Measurement Lane capture/replay correctness,
  allocation, and admission validation.

## Commands

- Primary reproduction command: `make reference-turboquant`.
- Validation command: `make validate-reference-turboquant`.
- Other commands executed: NVIDIA cache/package integrity checks and offline
  restoration; `nvidia-smi`; `make package-lock-check`; entry and final
  `make test`; entry and final `make checks`; focused Phase 5 tests; two exact
  reference generations; source/tree/wheel verification; `pip check`; and
  validation-only fixture replay.

No pilot, Full Scan, profile subset, fit, figure, serving throughput,
comparative latency, quality evaluation, or Nsight Compute command ran.

## Tests

- `make test`: PASS; schema 41, Phase 2 unit 31, Phase 3 unit 226,
  remediation controls 167, Phase 4 unit 10, Phase 5 unit 7, repository checks,
  and immutable evidence validation all passed.
- `make checks`: PASS.
- Source-lock test: PASS, including exact commit/tree/tag/blob/source and wheel
  byte identity.
- Config-name tests: PASS for all three mandatory names and same-path k8v4.
- Fixture-schema tests: PASS.
- Checksum tests: PASS.
- Byte-layout tests: PASS for formulas, sums, owned bytes, and actual files.
- Determinism tests: PASS; same-seed regeneration was byte-identical in the
  frozen environment and existing IDs were not overwritten.
- BF16 regression tests: PASS; BF16 production sources and tests were unchanged.
- Governance tests: PASS; TurboQuant factory rejection, quality lock, Full Scan
  closure, absence of formal timing, and historical evidence preservation.

## Admission gates

- G0: PASS.
- G1: PASS.
- G2: NOT EVALUATED.
- G3: NOT EVALUATED.
- G4: NOT EVALUATED.
- G5: NOT EVALUATED.
- Full Scan: CLOSED.

## Quality governance

- Quality execution: LOCKED.
- Quality benchmark run: no.
- `PERFORMANCE_DATA_FROZEN`: absent.

## Preservation

- Phase 4 evidence unchanged: yes.
- Phase 3 evidence unchanged: yes.
- E00 evidence unchanged: yes.
- Existing run overwritten: no.
- Formal performance data created: no.
- Profiler timing created: no; reference profiler durations were discarded.
- Quality data created: no.

The scope and immutable-evidence validators confirm that completed evidence and
historical phase reports are unchanged. The TurboQuant adapter factory still
rejects execution; no BF16 adapter, fixed-L runner, growing-context runner, or
quality implementation was modified.

## Changed files

- Source pin and plan: `docs/plans/phase5-turboquant-reference.md`,
  `docs/method_notes/turboquant.md`, `third_party/LOCK.json`, and
  `third_party/NOTICE.md`.
- Environment and commands: `.gitignore`, `Makefile`,
  `docker/reference-turboquant.Dockerfile`, and the scripts/manifests/README
  under `reference/turboquant/`.
- Fixtures and tests: the compact binaries, manifests, traces, and ledgers under
  `reference/turboquant/fixtures/`, plus
  `tests/unit/test_phase5_turboquant_reference.py`.
- Governance: `scripts/validate_phase2.py`, `docs/status.md`,
  `docs/blockers.md`, `docs/risk_register.md`, `docs/tasks.md`, and this report.

## Commits

- `6163b77`: `docs: freeze Phase 5 TurboQuant source`.
- `6c3a576`: `ref: add TurboQuant fixture environment`.
- `ae6d86c`: `test: freeze TurboQuant reference fixtures`.
- Final governance and this Phase 5 report: this commit.

## Risks

- R-007 remains explicit: conformance is to the pinned vLLM implementation,
  not an unverified broader paper implementation.
- R-032 records environment drift, overwrite, trace-interpretation, and
  deferred graph-admission risks.
- NVIDIA unattended-upgrade recurrence remains an operational risk. The
  proposed mitigation is a reviewed exact-version apt preference plus an
  `unattended-upgrades` NVIDIA exclusion, verified first with a dry run; an
  explicit hold of the complete package set is the fallback. No persistent
  policy was installed without approval.

## Blockers

- B-009 remains OPEN.
- B-010 remains OPEN; the reference environment is not the Measurement Lane.

## Scientific interpretation

The evidence supports only that the pinned upstream vLLM TurboQuant
implementation and generated fixtures are reproducible to the demonstrated
extent on the recorded SM120 reference environment. It supports no speedup,
compression-benefit, HBM-traffic, knee, capacity, formal performance, or
quality claim.

## Next action

Phase 6 TurboQuant Measurement Adapter may be proposed in a new task. It has
not begun and is not authorized by this report.

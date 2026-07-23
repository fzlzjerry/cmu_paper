# Phase 4 common BF16 adapter report

Status: PASS

Date: 2026-07-24

## Entry verification

- Starting HEAD:
  `a3a56e45354ac93ab3c25f82a82e8e6096b513b9`.
- The working tree was clean.
- Authoritative Phase 3 report:
  `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb`, PASS.
- Phase 3 report SHA-256:
  `c29aef1d9f22b328201599b3e6cdf9efe7c069e78abaf6b37bc3cb12931414c9`.
- The report checksum ledger, source-run indexes, and all source-run checksum
  ledgers independently validated.
- G0 and G1 were PASS. G2-G5 were NOT EVALUATED. Full Scan was CLOSED.
- Quality execution was LOCKED and `PERFORMANCE_DATA_FROZEN` was absent.
- B-009 and B-010 were OPEN.
- Entry `make test` and `make checks` passed.
- The protected E00 and Phase 3 evidence baseline contained 4,497 files with
  aggregate SHA-256
  `790de0d535f1ad6fe0d5363f133d83c92610e6f60b3ab34deae8b58b74641dda`.

## Scope and implementation

The concise implementation plan is
`docs/plans/phase4-common-adapter.md`. Phase 4 adds a small
`KVCacheMethod` protocol, one BF16 adapter, and one immutable explicit factory.
The protocol contains only `allocate`, `store_prefill`, `append_decode`,
`decode_attention`, `allocated_bytes`, `byte_breakdown`,
`logical_bf16_bytes`, `config_fingerprint`, and
`supports_cuda_graph`.

The adapter delegates to the existing `BF16StaticCache`, cache `update`
operation, forced-Flash `flash_attention_forward`, Phase 3 endpoint, output
buffers, and graph path. It preserves BF16 storage, 32 query heads, 8 KV
heads, head dimension 128, fixed-L scratch-slot semantics, growing-context
append semantics, model and tokenizer identity, and the frozen numerical
tolerances. It does not copy or replace the Phase 3 runtime.

The factory contains only the explicit `bf16` builder. TurboQuant, KIVI, and
KVQuant retain `phase_not_implemented`; unknown names are rejected. There is
no dynamic discovery, registration API, global mutable registry, plugin
framework, dependency-injection container, second schema framework, or new
campaign engine.

Fixed-L and growing-context runners construct the method before measured
execution and call the common endpoint/session facade. No method-specific
branch remains after construction, and timing boundaries are unchanged. New
manifests retain every prior field and add the adapter configuration
fingerprint. Historical manifests and artifacts were not rewritten.

No scientific semantic decision changed, so no decision record was added.
Compatibility changes were limited to adapter passthrough parameters,
manifest/schema support for the fingerprint, focused validation tests, and
allowlisting the checksum-bound Phase 4 smoke root.

The static cache, backend implementation, timing implementation, allocation
attribution, GQA device-dispatch parser, process supervision, artifact
lifecycle, CLI command surface, Phase 3 plans, and all old evidence were
reused unchanged.

## Adapter identity and accounting

- Adapter: `kvbench-bf16-method-adapter-1.0.0`.
- Fixed-L adapter configuration fingerprint:
  `ad2d39da5d2cd147faa03690f8c7f192ee4d1ec315ea3c51bc6cf4c4b40ba14f`.
- Fixed-L cache-layout fingerprint:
  `deaccec80b89045fc9c9507fe2bf0ddf8ce482d885a24ac89fbec4ee2ff3394e`.
- Growing-context adapter configuration fingerprint:
  `3b3898c23c6c3282e85db88dc032ca256531545f1ec3bceb939649fa9d9f48e5`.
- Growing-context cache-layout fingerprint:
  `d0f44530529ae6f7f5068cd10b462bfe85a36d99d0e228df833481337f23733a`.
- Method configuration fingerprint:
  `81ca6a0d74727a9c8a54f14d6d222dee502ee5e9a9ab6e8c9a03b4fed98f371b`.
- Backend fingerprint:
  `0841ae768cf05df38adbf803b5019460491572b9bf205d87f703428d2cfbc354`.

Canonical JSON and the repository SHA-256 utility bind method name, BF16
dtype, cache layout, exact model revision, backend identity, 32/8/128
geometry, graph capability, adapter version, and adapter implementation source.
The byte breakdown is deterministic and sums to actual cache-owned storage.
The adapter introduces no measured-operation allocation and changes no
numerical behavior.

## Common validation harnesses

`src/kvbench/runtime/method_harness.py` composes the existing Phase 3
validators:

- correctness covers the small-tensor reference, static cache, fixed-L,
  growing-context, eager, graph, eager/graph agreement, and finite-output
  controls;
- allocation combines adapter-owned bytes with the frozen eager allocation
  verdict and strict graph replay zero-allocation verdict;
- graph reuses capture/replay, pointer stability, output agreement, replay
  allocation, and eager-fallback checks;
- the execution-path facade exposes the existing backend, device-kernel,
  allocation, temporary-shape, GQA, full-prefix-temporary, host-sync, and
  fallback results without collecting a second trace. BF16 full-prefix
  dequantization is explicitly `not_applicable`.

## Method admission report

The strict compact schema is
`src/kvbench/schema/method_admission.py`. The validated BF16 instance is
`docs/evidence/phase4/method-admission.json`, SHA-256
`1362fd1817b8bb5706baaa09ed6e5115789fbc4d35d394f184d0b132a0e58d22`.
All evidence references join exactly and their SHA-256 values reconstruct.

The report records PASS for correctness, byte accounting, execution path,
graph behavior, and reproducibility. It records G0/G1 PASS, G2-G5 NOT
EVALUATED, Full Scan CLOSED, B-009/B-010 OPEN, quality LOCKED,
`quality_status=unvalidated`, `claim_eligibility=performance_only`,
`performance_claim_eligible=false`,
`measurement_scope=native_host_admission`, and
`performance_data_frozen=false`. It is a BF16 framework report, not a
quantized-method admission.

## Bounded functional smoke

Exactly three untimed functional smokes ran from clean execution SHA
`0cf160caa532c7cac23275c8a14fd8694789a86f`:

- fixed-L B=1, L=128, eager:
  `phase4-smoke-fixed-l-eager-20260723t184024203883z-0cf160ca-429f62`,
  PASS;
- fixed-L B=1, L=128, CUDA Graph:
  `phase4-smoke-fixed-l-cuda-graph-20260723t184024865242z-0cf160ca-2a4d27`,
  PASS;
- growing-context B=1, L=128, O=4, eager:
  `phase4-smoke-growing-context-eager-20260723t184025220693z-0cf160ca-9b6819`,
  PASS.

Their append-only directories have valid checksum ledgers. The smoke index is
`docs/evidence/phase4/smoke-index.json`, SHA-256
`6611a1f547a3d5d17cab6b2b36cad241a2aa8d437478fa4a24a4b173da594ba8`.
No record contains latency or formal timing samples. No independent timing
replicates, profiler campaign, pilot, Full Scan, or quality run occurred.

## Validation

- `make test`: PASS, including 10 focused Phase 4 tests.
- `make checks`: PASS, including format, lint, hot-path, annotation,
  configuration, provenance, scope, immutable evidence, and package locks.
- `make test-cuda`: PASS, 15/15.
- `make test-graph`: PASS, 4/4.
- Fingerprint reconstruction, byte-breakdown sum, runner delegation,
  fail-closed factory behavior, manifest retention, admission-report
  validation, evidence-reference checksums, quality lock, Full Scan lock, and
  non-overwrite governance tests passed.
- Existing Phase 3 CLI plans remain valid. BF16 dry-run constructs the
  adapter, while quantized methods, pilot, Full Scan, profiling, and quality
  remain fail-closed.

The required CUDA test targets exercised existing untimed Phase 3 audit
instrumentation; no profiler campaign or profiler artifact was created.

## Preservation and gates

After Phase 4 execution, the protected E00 and Phase 3 evidence set still
contains the same 4,497 files and the same aggregate SHA-256
`790de0d535f1ad6fe0d5363f133d83c92610e6f60b3ab34deae8b58b74641dda`.
No completed run was edited, overwritten, or deleted. No formal performance,
profiler, pilot, Full Scan, or quality data was created.

- G0: PASS.
- G1: PASS.
- G2: NOT EVALUATED.
- G3: NOT EVALUATED.
- G4: NOT EVALUATED.
- G5: NOT EVALUATED.
- Full Scan: CLOSED.
- Quality execution: LOCKED.
- `PERFORMANCE_DATA_FROZEN`: absent.
- B-009: OPEN.
- B-010: OPEN.

## Commits

- `5c44e43295733d67ad018bd2346ee4541c451029`:
  `phase4: add common BF16 method adapter`.
- `0cf160caa532c7cac23275c8a14fd8694789a86f`:
  `phase4: add focused adapter validation`.
- `9ec1bf00c309a7dbae105b574bc720a5a1a74003`:
  `phase4: record adapter evidence and governance`.
- This report and the final scope-allowlist correction are recorded in the
  final Phase 4 report commit.

## Risks, blockers, and scientific interpretation

R-031 records adapter-drift risk and its BF16 mitigations. B-009 durable
evidence storage and B-010 digest-pinned container parity remain open; neither
was closed or weakened.

The evidence supports only that the already validated native-host BF16
implementation preserves its numerical, allocation, GQA execution-path, graph,
and runner behavior through the common adapter boundary under the focused
tests and three untimed smokes. It supports no speedup, compression, capacity,
knee, HBM-traffic, formal performance, quantized-method, or quality claim.

Phase 5 has not begun. A TurboQuant Reference Lane may be proposed only as a
separate new task.

# PHASE 11 REPORT

Status: BLOCKED

## Entry and authority

- Starting HEAD: `72f1897af78b738cc8c74fd335a8957a8e8f5d6c`.
- Admission execution HEAD:
  `4a001b22739986dac34279c6ae2a0b06c8bb74a6`.
- Working tree at admission: clean.
- Final reporting HEAD and clean-tree state are emitted by the external
  handoff after this tracked report is committed; a tracked report cannot
  contain its own commit identity.
- Algorithm identifier: `kvquant_gqa_upstream_patch_v1`.
- Execution-source identifier: `kvquant_gqa_graphsafe_kvq3_v2`.
- Decisions: 0021, 0023, 0024, 0025 unchanged; Decision 0026 Accepted
  only for the backward-compatible pre-RoPE adapter boundary.
- Aggregate patch:
  `23a15db86790299392412c3ce2da7d971f4f073cfaf6839d82d3746c8b56b551`.
- Corrected commit/tree:
  `0d9df350bd1788284e1ce76a8bf6e886beca5efa` /
  `a85cf7bf093982a4bf89c33d4e6794d9a85f846d`.
- Corrected extension:
  `46c41aad8f56d58608d4c1273bd3a72fd36c8f69f9ca2c5a046f0c811631bf51`.
- Calibration ID/root:
  `kvqcal-cdb724c806d64d095c040d2673a987a3` /
  `8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`.
- Historical Phase 10 root:
  `32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab`.
- Corrected oracle ID/root:
  `kvqref-2e0a0e9022c50cbc6fb497d88cae973e` /
  `c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec`.
- Authorized Measurement Container:
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.

## Adapter result

- Adapter/cache:
  `src/kvbench/adapters/kvquant.py` and
  `src/kvbench/runtime/kvquant_cache.py`.
- Configurations: `kvq4`, `kvq3`, and `kvq2`.
- Boundary: pre-RoPE K is quantized; attention-ready K is retained for
  full-precision sink semantics; V remains the native projection output.
- Static layout: caller-owned packed K/V, metadata, float32 sparse values,
  int32 sparse indices, counts/masks, native-eight-head sink K/V, staging,
  output, softmax, merge, and correction workspaces.
- Layout fingerprints:
  `kvq4=07a2d4b1c2df473e272c5513fa989ab150229c594b9b9e92f28e0953f39b382f`,
  `kvq3=679dffa1414fe2fdd2102e2e71edd6335363b9cce5c52c00b36192f82d3d173b`,
  and
  `kvq2=b6d154a07402e8c3bce05462aab482323dfd606bcc69215468e4f83231a188c6`.
- Sparse semantics: Key count 0..12; non-sink Value count 12; sink Value count
  0; capacity 12; no CPU top-k or dynamic sparse output.
- Store/append: fixed-slot writes and overwrite semantics conform to the
  corrected fixtures.
- Decode: direct compressed-cache path with
  `kv_head = query_head // 4`; no complete-prefix materialization, GQA
  expansion, cache growth, or fallback was observed in completed points.
- Accounting formulas cover L=5, 17, 18, 128, and 4096 with exact category
  sums, allocation error below 1%, reciprocal `rho_alloc`/`r_alloc`, and
  `r_hbm=null`; final admission accounting evidence was not finalized after
  the mandatory grid stopped.

## Validation and stopped admission

- All nine corrected fixtures: PASS for dense payload, metadata, sparse
  values/indices/counts, sink, store, append, byte records, finite output, and
  frozen decode tolerance.
- Focused CUDA Graph tests: PASS for all three bit widths, pointer stability,
  eager/graph agreement, repeated replay, and zero replay allocation.
- Compute Sanitizer pre-grid matrix: PASS with zero reported errors and zero
  leaked allocations for the required short fixture/path cases.
- Bounded grid: six of nine points completed and passed:
  - `phase11-20260729t190343762z-4a001b22-c81b0c-kvquant-p00-kvq4-fixed-l128-eager`;
  - `phase11-20260729t190343762z-4a001b22-c81b0c-kvquant-p01-kvq4-fixed-l128-graph`;
  - `phase11-20260729t190343762z-4a001b22-c81b0c-kvquant-p02-kvq3-fixed-l128-eager`;
  - `phase11-20260729t190343762z-4a001b22-c81b0c-kvquant-p03-kvq3-fixed-l128-graph`;
  - `phase11-20260729t190343762z-4a001b22-c81b0c-kvquant-p04-kvq2-fixed-l128-eager`;
  - `phase11-20260729t190343762z-4a001b22-c81b0c-kvquant-p05-kvq2-fixed-l128-graph`.
- Point 7, `...-p06-kvq4-fixed-l4096-eager`, was attempted but stopped
  before point finalization. Points 8 and 9 (`kvq4` L=4096 CUDA Graph and
  `kvq4` growing-context L=17/O=4 eager) were not attempted.
- Stop reason: `EndpointSessionError: post-warmup output is unstable`.
- Failed append-only run:
  `phase11-20260729t190343762z-4a001b22-c81b0c-kvquant`, local root
  `227e37abc8649433bf806bbc527829c4062dd2ca6da5cbeafc8f152dbfb9a982`
  with 100 checksum-valid objects.
- R2 publication, clean retrieval, and MethodAdmissionReport: not attempted
  because the local admission gate did not pass.

## Concrete blocker

An untimed exact-container control repeatedly invoked the corrected kvq4
Value-decode kernel on identical caller-owned inputs:

- quantized width 124: 20/20 runs produced one exact output SHA-256;
- quantized width 4092: dense-only produced 20 distinct SHA-256 values in
  20 runs;
- quantized width 4092: dense plus sparse correction also produced 20
  distinct SHA-256 values in 20 runs;
- all values were finite and remained within the previously frozen
  `atol=0.01, rtol=0.01`, but were not byte deterministic.

Raw local diagnostic:
`artifacts/phase11/phase11-l4096-diagnostic.gg2dGt/result.json`, SHA-256
`3a23331999dd8787729e18c537529bfff6242f2af29b638052ade25991a493da`.
Its source SHA-256 is
`2ace6951286bb3a8628ab900a5c0f3ff6421d123f3ff99b6bb929d231aeef43c`.

The corrected source launches 32 Value-decode width tiles at this context.
Dense and sparse Value paths contain inter-block floating `atomicAdd`
reductions, and the tail tile contains a shared-memory initialization hazard
for inactive columns. The dynamic control proves execution-history-dependent
output under source containing both risks; it does not isolate their
individual causal shares. Clearing caller-owned buffers cannot make this
source byte deterministic.

Changing the post-warmup exact gate to tolerate the newly observed drift would
weaken admission after seeing the result. A valid repair requires a separately
authorized, checksum-bound deterministic CUDA Value-reduction and tail
initialization correction, followed by new extension identity, targeted
numerical/Graph/allocation/sanitizer tests, and a fresh nine-point admission.
That source change is prohibited in Phase 11.

## Gate and preservation state

- G0, G1, G2-TQ, G2-KIVI: PASS.
- Phase 11: BLOCKED.
- G2-KVQ and Global G2-G5: NOT EVALUATED.
- Full Scan: CLOSED.
- Quality execution: LOCKED.
- `PERFORMANCE_DATA_FROZEN`: absent.
- Phase 9 calibration, Phase 10 fixtures, corrected Phase 11P-R fixtures,
  Decisions 0021/0023/0024/0025, BF16, TurboQuant, KIVI, and the Measurement
  Container: unchanged.
- Phase 12: not started.
- Performance, speedup, HBM, knee, capacity, profiler, and quality claims:
  none.

## Commits and next action

- `1188f1d`: plan and Decision 0026.
- `781b416`: pre-RoPE boundary.
- `96b2520`: static cache and corrected-oracle binding.
- `4a001b2`: factory, runner, audit, fixture, graph, sanitizer, and admission
  integration.
- Final governance/report commit: emitted by the external handoff.

Tracked files changed from the starting HEAD:

```text
Makefile
docs/decisions/0026-kvquant-pre-rope-adapter-boundary.md
docs/phase_reports/phase11-kvquant-measurement-adapter.md
docs/plans/phase11-kvquant-measurement-adapter.md
docs/risk_register.md
docs/status.md
docs/tasks.md
scripts/phase11_kvquant_admission.py
scripts/phase11_r2_outer_bundle.py
scripts/validate_phase2.py
src/kvbench/adapters/__init__.py
src/kvbench/adapters/base.py
src/kvbench/adapters/factory.py
src/kvbench/adapters/kvquant.py
src/kvbench/runtime/allocation_attribution.py
src/kvbench/runtime/artifacts.py
src/kvbench/runtime/bf16_endpoint.py
src/kvbench/runtime/kvquant_cache.py
src/kvbench/runtime/kvquant_fixture.py
src/kvbench/runtime/kvquant_session.py
src/kvbench/schema/__init__.py
src/kvbench/schema/phase11.py
tests/cuda/phase11_kvquant_sanitizer_probe.py
tests/cuda/test_phase11_kvquant_cuda.py
tests/graph/test_phase11_kvquant_graph.py
tests/unit/test_phase10_kvquant_reference.py
tests/unit/test_phase11_kvquant_adapter.py
tests/unit/test_phase11_kvquant_admission.py
tests/unit/test_phase11_kvquant_admission_driver.py
tests/unit/test_phase11_kvquant_cache.py
tests/unit/test_phase11_kvquant_factory.py
tests/unit/test_phase11_kvquant_fixture.py
tests/unit/test_phase11_kvquant_session.py
tests/unit/test_phase11_make_targets.py
tests/unit/test_phase11_r2_outer_bundle.py
tests/unit/test_phase11_scope.py
tests/unit/test_phase11pr_scope.py
tests/unit/test_phase6_governance.py
tests/unit/test_phase9_governance.py
tests/unit/test_phase9p_governance.py
tests/unit/test_phase9p_patch_custody.py
```

Scientific interpretation is limited to this result: the static KVQuant
adapter conforms to the corrected short reference fixtures, but the frozen
long-context Value-decode CUDA reduction is not byte deterministic and G2-KVQ
is not satisfied.

Next action: propose a separate narrow CUDA-source remediation for the
deterministic long-context Value reduction and tail initialization. Do not
begin Phase 12.

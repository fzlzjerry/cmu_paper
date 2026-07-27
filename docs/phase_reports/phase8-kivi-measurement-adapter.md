# PHASE 8 REPORT

Status: PASS

## Entry

- Starting HEAD: `8d6d766a34a15bd40bd42cc47c5482b0dd052cc0`
- Final HEAD: execution and admission HEAD
  `462325e9df809d3bcf24a06361bf004bc7383d73`; the report-only descendant
  containing this file is recorded in the final handoff.
- Working tree: clean at entry and at the execution/admission HEAD; final
  report-only evidence is committed before the outer artifact lifecycle
  proceeds.
- Phase 7 report: PASS,
  `docs/phase_reports/phase7-kivi-reference.md`, unchanged.
- Phase 7 R2 root:
  `abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302`;
  clean retrieval PASS.
- Authorized container:
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`,
  available and unchanged.
- G0: PASS.
- G1: PASS.
- G2-TQ: PASS.
- G2-KIVI at entry: NOT EVALUATED.
- Quality: execution LOCKED; benchmark NOT RUN.
- Full Scan: CLOSED.

## Ratio terminology

- Phase 7 legacy field: the immutable Phase 7 field named `r_alloc` is
  interpreted as `rho_alloc_legacy`.
- Canonical `rho_alloc`:
  `C_method_allocated / C_BF16_allocated`.
- Canonical `r_alloc`:
  `C_BF16_allocated / C_method_allocated`.
- Erratum/decision:
  `docs/decisions/0019-phase7-allocation-ratio-terminology-erratum.md`;
  reciprocal absolute tolerance `1e-9`.
- Historical fixtures modified: no; Phase 7 bytes, values, checksums,
  manifests, traces, and report remain unchanged.
- Reciprocal tests: PASS. Every bounded-admission point has
  `abs(r_alloc * rho_alloc - 1) = 0.0`, within the frozen tolerance. Legacy
  values are normalized before use and cannot be consumed directly by model
  fitting.

## Minimal scope

- Plan: `docs/plans/phase8-kivi-measurement-adapter.md`.
- New framework: no.
- New runner: no benchmark runner; the existing fixed-L and growing-context
  runners are used through one narrow KIVI session bridge and admission
  driver.
- Measurement Container changed: no.
- TurboQuant changed: no; its adapter, configuration, fixtures, and admission
  evidence are unchanged.
- Phase 9 started: no.

## Source authority

- Official commit: `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`.
- Base tree: `c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b`.
- Patched tree: `b617493dea5aff1a754cd27ad6be12ac512b2aee`.
- Decision 0018 patch SHA-256:
  `c9c2dd52d4c81b844d1d1d7218ad2cd60a5b31574a387f716d466cb01310423d`.
- Extension SHA-256:
  `45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9`.
- Algorithmic source changed: no Phase 8 algorithmic or CUDA source change.
  Quantization, packing, scale/minimum semantics, rollover formulas, and the
  official CUDA kernels remain those of the exact Decision 0018 authority.
- Conformance authority wording: checksum-bound patched official KIVI source,
  not an unmodified official system and not a paper-era equivalence claim.
- Exact upstream kernel families reused:
  `bgemv2_kernel_outer_dim` and `bgemv4_kernel_outer_dim`, launched from the
  checksum-bound extension into caller-owned output storage. The bound host
  stub offsets are `0xab70` and `0xa970`, respectively.

## Adapter

- Location: `src/kvbench/adapters/kivi.py`; the single cache-state class is
  in `src/kvbench/runtime/kivi_cache.py`.
- Configurations: mandatory `k4v4`, `k2v4`, and `k2v2`; held-out validation
  control `k4v2`.
- Group size: 32.
- Residual length: 32.
- BF16-FP16 boundary: BF16 common-runner input is copied/cast in place into
  preallocated FP16 staging, the exact half-only official KIVI operations and
  residual math execute in FP16, and output is copied/cast in place into the
  BF16 common output.
- Factory: the existing explicit factory maps only the four authorized KIVI
  configuration names after validating the frozen geometry.
- KVQuant fail-closed: PASS; KVQuant remains deferred and rejected.

## Static cache

- Historical K: preallocated packed key history with capacity rounded to
  complete 32-token key groups.
- Historical V: preallocated packed value history with capacity for every
  token outside the recent residual window.
- Metadata: preallocated K/V scales and additive minimum offsets plus fixed
  token-index ledgers; all actual owned metadata is counted.
- Residual K: one preallocated 32-token FP16 region per layer.
- Residual V: one preallocated KIVI-specific 32-slot FP16 circular region per
  layer, plus a fixed ordered staging view.
- Staging: preallocated FP16 query, K, V, metadata operand, quantization, and
  decode-output staging.
- Workspace: preallocated logits, softmax, merge, common output, kernel
  output, mapping, and index workspaces.
- Allocation: every persistent adapter-owned buffer is allocated before the
  measured operation for the declared maximum capacity; actual physical bytes
  and active source-faithful bytes are reported separately.
- Dynamic growth: none.
- `torch.cat`: absent from the measured KIVI path.
- Layout fingerprints:
  - `k4v4`:
    `2250f4971092a2712ed0a5d9872f1b2ac7d5073c9492df46d18fb0a587159c3e`
  - `k2v4`:
    `937152211917de10a077ec53f67a6b4ddea3eb4afaa643d9a9aa7d20a1f4bc69`
  - `k2v2`:
    `cc17163a48dd0e10f22dc6725082bba1349ba59a7a58747f3b025b288cf4d2e1`
  - `k4v2`:
    `cc3b5958d9146af758e18f95b68e265c8592ebcc84f06a3ec5400d359ce4355a`

## Rollover

- K boundary: after token 31 is committed at active length 32, key tokens
  0-31 are quantized into the next preallocated historical group and the key
  residual becomes empty.
- V boundary: at active length 33, value token 0 is quantized into the next
  preallocated historical position and residual tokens 1-32 retain logical
  order; at active length 34, value token 1 moves and residual tokens 2-33
  remain.
- L31: no historical K/V; residual K/V tokens 0-30. PASS.
- L32: historical K tokens 0-31; empty K residual; V residual tokens 0-31.
  PASS.
- L33: historical K tokens 0-31; K residual token 32; historical V token 0;
  V residual tokens 1-32. PASS.
- L34: historical K tokens 0-31; K residual tokens 32-33; historical V tokens
  0-1; V residual tokens 2-33. PASS.
- Missing tokens: none.
- Duplicate tokens: none.
- Reallocation: none.
- Growing-context result: PASS for B=1, starting L=31, O=4, eager. The audited
  rollover active lengths are 31, 32, 33, and 34, and the runner finishes
  with committed context 35 without changing cache pointers.

## Byte accounting

The duplicated eager/Graph points have identical accounting. `Logical BF16`
below is allocated-capacity BF16 K/V; active-context BF16 bytes are also
retained in each manifest.

| Configuration/context | Allocated bytes | Active storage | Logical BF16 | rho_alloc | r_alloc | Relative error |
|---|---:|---:|---:|---:|---:|---:|
| k4v4 fixed L=128, capacity 129 | 11,276,928 | 6,684,672 | 16,908,288 | 0.6669467659883721 | 1.4993700412027104 | 0.0 |
| k2v4 fixed L=128, capacity 129 | 10,228,352 | 5,636,096 | 16,908,288 | 0.6049312621124031 | 1.653080378931034 | 0.0 |
| k2v2 fixed L=128, capacity 129 | 9,433,728 | 4,849,664 | 16,908,288 | 0.5579351380813954 | 1.7923230349656043 | 0.0 |
| k4v2 fixed L=128, capacity 129 | 10,482,304 | 5,898,240 | 16,908,288 | 0.6199506419573644 | 1.613031638845811 | 0.0 |
| k4v4 fixed L=4096, capacity 4097 | 176,853,632 | 169,213,952 | 537,001,984 | 0.3293351556779351 | 3.0364204451283197 | 0.0 |
| k4v4 growing L=31/O=4, committed L=35 | 7,313,024 | 3,010,560 | 4,587,520 | 1.5941127232142858 | 0.6273082106663399 | 0.0 |

- Product check: PASS for every point; maximum observed reciprocal product
  error is 0.0.
- Persistent breakdown: packed historical K, packed historical V, K
  scales/minima, V scales/minima, other metadata, K/V residuals, FP16 staging,
  quantization staging, padding/alignment, persistent workspace, fixed
  rollover scratch, and block/group rounding sum exactly to actual owned
  storage.
- Predicted-versus-allocated error: 0.0 for every bounded point, below 1%.
- Temporary peak: recorded separately as 0 bytes for these points.
- `r_hbm`: null; no HBM traffic estimate is made.
- Piecewise tests: PASS at L=31, 32, 33, 64, 128, and 4096 without requiring
  a smooth or strictly monotonic curve.

## Fixture conformance

- Fixture root:
  `abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302`.
- `k4v4`: PASS.
- `k2v4`: PASS.
- `k2v2`: PASS.
- `k4v2`: PASS as held-out validation only.
- Input checksums: PASS.
- Store: exact tensor/state match, PASS.
- Append: exact tensor/state match, PASS.
- Packed K/V: exact match, PASS.
- Metadata: exact scale/minimum tensor match, PASS.
- Residual: exact tensor and token-index match, PASS.
- Rollover: exact L31/L32/L33/L34 state match, PASS.
- Byte breakdown: exact match after interpreting the Phase 7 legacy ratio name
  through Decision 0019, PASS.
- Decode tolerance: frozen before admission at `atol=0.02`, `rtol=0.02`;
  FP16 official-boundary and BF16 common-runner outputs PASS.
- NaN/Inf: none.
- Source, patch, extension, and fixture authority: exact match.

## Execution path

- 2-bit kernels: `bgemv2_kernel_outer_dim`, verified.
- 4-bit kernels: `bgemv4_kernel_outer_dim`, verified.
- Full-prefix dequantization: absent.
- GQA materialization: absent; persistent and operand K/V geometry remains
  H_KV=8 with `kv_head = query_head // 4`.
- `repeat_kv` / `repeat_interleave`: absent.
- H_Q-sized K/V temporary: absent.
- Host synchronization: absent from the hot path.
- Dynamic allocation: no cache growth and no unknown or persistent measured
  allocation. Eager common-path ephemeral allocation is fully attributed;
  Graph replay has zero allocation events.
- Backend fallback: absent.
- Stable post-warmup path: PASS across ten independently observed launcher
  sequences.
- Instrumented durations: discarded and not reported as benchmark timing.

## Allocation

- Eager cache growth: zero.
- Eager unknown allocation: zero.
- Eager criterion: all nine eager operation endpoints, including four
  growing-context steps, pass
  `phase8_kivi_decision_0013_composed_eager_attribution_v1`. Each operation
  has 874 fully attributed ephemeral events totaling 9,637,132 bytes, with no
  context-dependent, cache-growth, GQA-expanded, full-prefix, or unknown
  allocation.
- Persistent allocated delta: zero for every audited eager and Graph
  operation.
- Persistent reserved delta: zero for every audited eager and Graph operation.
- Graph events: zero allocation events and zero allocation bytes for every
  bounded Graph point.
- Graph delta: zero allocated and zero reserved delta.
- Raw evidence: 13 operation bundles retain canonical allocator snapshot,
  trace, before/after stats, accounting, operation witness, strict audit, and
  checksum records.

## CUDA Graph

- Capture: PASS for all three mandatory configurations at L=128 and for k4v4
  at L=4096.
- Replay: PASS.
- Agreement: eager/Graph output agreement PASS under the frozen
  `atol=0.02`, `rtol=0.02`.
- Stability: repeated replay is bit-stable and cache, staging, metadata,
  workspace, input, and output pointers remain stable.
- Replay allocation: zero events, zero bytes, zero allocated delta, and zero
  reserved delta.
- Fallback: absent.
- Growing-context Graph: not run and not required in Phase 8.

## Sanitizer

- Kernel families: distinct 2-bit and 4-bit
  `bgemv*_kernel_outer_dim` families.
- Rollover covered: yes, including k4v4 L31-to-L33 token movement.
- Result: PASS; exit code 0, `ERROR SUMMARY: 0 errors`, and
  `LEAK SUMMARY: 0 bytes leaked in 0 allocations`.
- Evidence:
  `artifacts/phase8/phase8-20260727t113020276z-462325e9-0edc5a-k4v4-fixed-l128-eager/validation/sanitizer/`;
  result, stdout, stderr, command, source/extension identity, process
  supervision, and checksums are retained.

## Bounded admission

- Plan:
  - each mandatory configuration at B=1/L=128, eager and CUDA Graph;
  - k4v4 at B=1/L=4096, eager and CUDA Graph;
  - k4v4 growing-context B=1, starting L=31, O=4, eager;
  - held-out k4v2 at B=1/L=128, eager.
- Run IDs:
  1. `phase8-20260727t113020276z-462325e9-0edc5a-k4v4-fixed-l128-eager`
  2. `phase8-20260727t113219795z-462325e9-4500db-k4v4-fixed-l128-graph`
  3. `phase8-20260727t113303541z-462325e9-1e9f53-k2v4-fixed-l128-eager`
  4. `phase8-20260727t113346546z-462325e9-638d16-k2v4-fixed-l128-graph`
  5. `phase8-20260727t113425952z-462325e9-fb8991-k2v2-fixed-l128-eager`
  6. `phase8-20260727t113504872z-462325e9-d39761-k2v2-fixed-l128-graph`
  7. `phase8-20260727t114925068z-462325e9-697123-k4v4-fixed-l4096-eager`
  8. `phase8-20260727t120333372z-462325e9-37371a-k4v4-fixed-l4096-graph`
  9. `phase8-20260727t120354423z-462325e9-bf994c-k4v4-growing-l31-eager`
  10. `phase8-20260727t120439529z-462325e9-f2f221-k4v2-fixed-l128-eager`
- Attempted: 10.
- Passed: 10.
- Failed: 0.
- Speedup calculated: no.
- Measurement scope: `measurement_container_admission`.
- Claim eligibility: `performance_only`; `performance_claim_eligible=false`.

## Durable publication

- Local inner root:
  `f0c72b5330d2f1f0ab4c6a1594d223fdf068a32cf58cdec63f4e254ef8aed515`.
- R2 URI:
  `r2://kvbench-artifacts/kvbench/sha256/f0c72b5330d2f1f0ab4c6a1594d223fdf068a32cf58cdec63f4e254ef8aed515/`.
- Initial publication: PASS; 331/331 objects uploaded through
  content-addressed conditional writes, with zero pre-existing objects.
- COMPLETE-last: PASS; publication-order SHA-256
  `0a8831003dc24af2d7b9d082bcc01d53e42bef5720f72b74b539147f607f7914`.
- Clean retrieval: PASS into a new empty directory; 331/331 objects and no
  unexpected object.
- Checksum: COMPLETE, inventory, checksum ledger, bundle validation, and root
  digest all PASS.
- Bucket Lock: Cloudflare R2 bucket `kvbench-artifacts`, exact enabled
  indefinite rule identity `kvbench-evidence-indefinite`, covering
  `kvbench/sha256/`; bucket private.
- Credential leakage: none. Credential values were not recorded, `.env` was
  not read, and credentials were not passed into the Measurement Container.
- Receipt:
  `docs/evidence/phase8/r2-admission-publication.json`.

## Tests

- Package lock: PASS.
- `make test`: PASS at the clean execution lineage; all focused Phase 8 unit
  tests PASS.
- `make checks`: PASS.
- `make test-cuda`: PASS inside the exact authorized Measurement Container.
- `make test-graph`: PASS inside the exact authorized Measurement Container.
- TurboQuant regression: `make validate-admission-turboquant` PASS; adapter
  and evidence unchanged.
- KIVI reference: `make validate-reference-kivi` PASS; fixtures unchanged.
- Ratio tests: PASS for canonical direction, legacy compatibility, swapped
  value rejection, reciprocal tolerance, and fitting prohibition.
- Fixture tests: PASS for all four configurations.
- Rollover tests: PASS for L31/L32/L33/L34, token movement, bounds, fixed
  scratch, and no reallocation.
- Byte tests: PASS for actual-owned sum, prediction error, canonical ratios,
  and block-rounded L31/L32/L33/L64/L128/L4096 behavior.
- GQA tests: PASS for native eight-head storage, `query_head // 4`, no
  replication, and no H_Q-sized K/V temporary.
- Allocation tests: PASS for raw evidence replay, complete attribution,
  mutation rejection, zero unknown allocation, and zero persistent deltas.
- Graph tests: PASS for capture, replay, agreement, stability, pointer
  identity, and zero replay allocation.
- Sanitizer tests: PASS for the bounded two-bit/four-bit plus rollover matrix.
- Governance tests: PASS; Full Scan and quality remain locked, KVQuant remains
  fail-closed, and `r_hbm` remains null.
- Historical evidence: PASS; Phase 7 report, fixtures, ledgers, TurboQuant
  evidence, source locks, and authorized image remain unchanged.

## Method admission

- G2-KIVI: PASS.
- Mandatory configurations: `k4v4`, `k2v4`, and `k2v2`, all admitted.
- Held-out configuration: `k4v2`, PASS as validation-only and not included in
  `admitted_configurations`.
- MethodAdmissionReport:
  `docs/evidence/phase8/kivi-method-admission.json`.
- Strict checks: 17/17 PASS, derived from 261 checksum-bound evidence
  references:
  fixture conformance, byte accounting, residual rollover, token integrity,
  static cache, no measured `torch.cat`, direct compressed decode, native GQA,
  no unknown allocation, Graph capture/replay, Graph zero replay allocation,
  no fallback, Compute Sanitizer, bounded grid, immutable checksums, durable
  publication, and clean retrieval.
- Report checksum:
  `3a4b63b9da0eab12db9a916ebdc1cffd788ea6f93678d87964a8332ae7cec83a`.

## Admission gates

- G0: PASS.
- G1: PASS.
- G2-TQ: PASS.
- G2-KIVI: PASS.
- Global G2: NOT EVALUATED.
- G3: NOT EVALUATED.
- G4: NOT EVALUATED.
- G5: NOT EVALUATED.
- Full Scan: CLOSED.

## Quality governance

- Quality execution: LOCKED.
- Quality benchmark: NOT RUN.
- `PERFORMANCE_DATA_FROZEN`: absent / false.
- Quality status: `unvalidated`.

## Preservation

- Phase 7 report changed: no.
- Phase 7 fixtures changed: no.
- TurboQuant changed: no.
- Authorized image changed: no.
- Historical evidence changed: no.
- Existing run overwritten: no.
- Formal performance data: none.
- Profiler data: none; allocation/path instrumentation is separate and no
  instrumented duration is reported as benchmark timing.
- Quality data: none.
- Prior Phase 8 attempt: the append-only `96cd5248` attempt stopped at
  `bounded_grid_8` with
  `EndpointSessionError: session audits did not pass`; its artifacts remain
  preserved, were not overwritten, and were not promoted or published as the
  passing admission bundle.

## Changed files

- Plan and terminology:
  `docs/plans/phase8-kivi-measurement-adapter.md`;
  `docs/decisions/0019-phase7-allocation-ratio-terminology-erratum.md`.
- Adapter and factory:
  `src/kvbench/adapters/kivi.py`;
  `src/kvbench/adapters/__init__.py`;
  `src/kvbench/adapters/factory.py`.
- Static state and runtime:
  `src/kvbench/runtime/kivi_cache.py`;
  `src/kvbench/runtime/kivi_fixture.py`;
  `src/kvbench/runtime/kivi_session.py`;
  `src/kvbench/runtime/kivi_admission.py`;
  `src/kvbench/runtime/kivi_allocation.py`.
- Reused common infrastructure extensions:
  `src/kvbench/runtime/allocation.py`;
  `src/kvbench/runtime/allocation_attribution.py`;
  `src/kvbench/runtime/artifacts.py`;
  `src/kvbench/runtime/numerical.py`;
  `src/kvbench/runtime/process_supervision.py`.
- Schema and narrow entrypoints:
  `src/kvbench/schema/phase8.py`;
  `scripts/phase8_kivi_admission.py`;
  `scripts/phase8_r2_outer_bundle.py`;
  `scripts/validate_phase2.py`;
  `Makefile`.
- CUDA and Graph validation:
  `tests/cuda/phase8_kivi_sanitizer_probe.py`;
  `tests/cuda/test_phase8_kivi_cuda.py`;
  `tests/graph/test_phase8_kivi_graph.py`.
- Focused unit validation:
  `tests/unit/test_phase8_artifacts.py`;
  `tests/unit/test_phase8_governance.py`;
  `tests/unit/test_phase8_kivi_adapter.py`;
  `tests/unit/test_phase8_kivi_admission.py`;
  `tests/unit/test_phase8_kivi_admission_driver.py`;
  `tests/unit/test_phase8_kivi_allocation.py`;
  `tests/unit/test_phase8_kivi_cache.py`;
  `tests/unit/test_phase8_kivi_fixture.py`;
  `tests/unit/test_phase8_kivi_schema.py`;
  `tests/unit/test_phase8_kivi_session.py`;
  `tests/unit/test_phase8_make_targets.py`;
  `tests/unit/test_phase8_process_supervision.py`;
  `tests/unit/test_phase8_r2_outer_bundle.py`;
  `tests/unit/test_phase8_ratio_terminology.py`;
  focused Phase 7 authority regressions in
  `tests/unit/test_phase7_kivi_b019_remediation.py` and
  `tests/unit/test_phase7_kivi_source_audit.py`.
- Finalized governance evidence:
  `docs/evidence/phase8/r2-admission-publication.json`;
  `docs/evidence/phase8/kivi-method-admission.json`;
  `docs/evidence/phase8/kivi-method-admission.sha256`;
  this report.

## Commits

- `fea216ff1c58d3c117a5a270268b952c09ec7383` —
  `docs: freeze phase 8 kivi adapter plan`.
- `0faeb68956f2641f9618a2e53c04e8dd3a1cf284` —
  `feat: add static KIVI measurement adapter`.
- `462325e9df809d3bcf24a06361bf004bc7383d73` —
  `test: integrate strict KIVI admission evidence`.
- One report/evidence commit contains the durable receipt,
  MethodAdmissionReport, checksum, and this report; its exact SHA is recorded
  in the final handoff.
- No push, tag, or pull request was performed.

## Risks

- Conformance is to the exact official KIVI base plus the checksum-bound
  Decision 0018 patch, not unmodified upstream or paper-era behavior.
- The official CUDA ABI remains FP16-only; the adapter's frozen
  BF16-to-FP16-to-BF16 staging boundary is part of the method fingerprint and
  must not be generalized to another boundary.
- The exact extension digest and direct host-stub offsets are execution
  authority. Any binary, source, ABI, container, or offset drift requires a
  new audit and admission.
- Short-capacity allocation can have `rho_alloc > 1` because all static
  staging and workspace are physically owned and counted. This is correct
  accounting, not a compression, capacity, or HBM claim.
- Phase 7's legacy ratio name remains immutable and must be normalized through
  Decision 0019 before later analysis.
- The earlier failed Phase 8 attempt remains append-only evidence and cannot
  be substituted for the successful `462325e9` bundle.

## Blockers

- None for the static KIVI Measurement Adapter or method-specific G2-KIVI.
- Global G2-G5 remain NOT EVALUATED.
- Phase 9, KVQuant calibration/reference/adapter work, Pilot, profiling,
  Full Scan, fitting, figures, performance claims, and quality evaluation
  remain outside this phase.

## Scientific interpretation

The static KIVI Measurement Adapter conforms to the checksum-bound patched
official KIVI reference for all three mandatory configurations and the
held-out asymmetry control, and it satisfies method-specific G2-KIVI. No
speedup, physical-HBM-traffic, knee, capacity, performance, or quality claim is
made.

## Next action

The report-bearing outer artifact may now package this immutable report,
MethodAdmissionReport, receipt, and complete inner bundle for conditional
COMPLETE-last publication and clean retrieval; no outer root is claimed in
this report before that lifecycle occurs. After durable report closure, Phase
9 KVQuant Calibration may be proposed only as a separate new task. Phase 9
has not started.

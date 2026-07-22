# Phase 3 BF16 baseline report

Status: FAIL

Date: 2026-07-22

Execution Git SHA: `457123b12220aa4a724968c1b4dd04340cf34a54`

Report-generator Git SHA: `ade0e86d2243ff193f684e008f99f35403dca293`

## Entry and scope

Phase 3 entered from clean Phase 2 final SHA
`c16139b0f365eaa052b17cff2fd19c1d4c62a4d1`. The Phase 2 report, commits,
11 configuration bundles, CLI fail-closed behavior, append-only lifecycle,
command reconstruction, both E00 ledgers/manifests, immutable E00 byte
identity, quality lock, absent `PERFORMANCE_DATA_FROZEN`, and closed Full Scan
were verified. Entry `make test` and `make checks` passed. No Phase 3 artifact
existed and no unrelated local change was present.

Only the BF16 static-cache baseline was implemented and evaluated. Phase 4,
quantized methods, pilot/full scan, profilers, fitting, figures, capacity
experiments, and quality execution remained closed.

## Frozen identities

- Model/tokenizer: `meta-llama/Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`, loaded offline from the exact
  content-addressed local snapshot.
- Config SHA-256:
  `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e`.
- Tokenizer SHA-256: `tokenizer.json`
  `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`,
  `tokenizer_config.json`
  `177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424`,
  and `special_tokens_map.json`
  `6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec`.
- Architecture: `LlamaForCausalLM`, 32 layers, 32 query heads, 8 KV heads,
  head dimension 128, maximum context 131072, BF16 weights, Llama-3 RoPE
  scaling with factor 8 and theta 500000.
- Backend: direct PyTorch 2.12.1+cu130 Flash SDPA, bundled FA2 2.5.7,
  `enable_gqa=True`, CUDA 13.0, cuDNN 9.20.0. No other SDPA backend was enabled
  and no fallback was accepted.
- The same exact checkpoint and tokenizer remain reserved for later quality
  validation; quality execution remains locked.

## Implementation

The cache owns two contiguous BF16 tensors, K and V, with shape
`[layers,batch,kv_heads,capacity,head_dim]`. Tensor bytes are
`2 * layers * batch * kv_heads * capacity * head_dim * 2`. Admission capacities
ranged from 129 through 16385 positions; tensor storage ranged from 16,908,288
to 8,590,458,880 bytes, with zero padding and 163,840-655,360 bytes of declared
workspace. Bounds, deterministic positions, reset, fixed scratch overwrite,
growing append, pointer stability, byte accounting, and no-resize behavior are
tested. The measured SUT source contains no `torch.cat`, `repeat_kv`,
`repeat_interleave`, or dynamic cache.

`context_length=L` means historical prefix length. Fixed-L attends `L+1`
including scratch position `L`, overwrites that slot, and preserves historical
cache bytes. Growing context records pre-append lengths `L..L+O-1`; all four
runs preserved the exact 16-step progressions without reallocation. Tokens,
positions, sampling exclusion, timing boundaries, CUDA events, host wall time,
memory evidence, and before/after NVML telemetry are explicit.

Small-tensor tolerances were frozen at `atol=0.02,rtol=0.02`; full-model logits
at `atol=0.125,rtol=0.02`; eager/graph agreement at
`atol=0.02,rtol=0.02`. Nineteen runtime-complete processes passed the small,
static-cache, and full-model numerical controls with finite outputs. The one
aborted process lacks complete runtime evidence, so the all-run numerical and
identity criteria correctly fail.

## Bounded campaigns and immutable report

- Fixed-L campaign:
  `phase3-20260722t112917207390z-457123b1-36731e`, 16/16 processes attempted.
- Growing campaign:
  `phase3-20260722t113532869819z-457123b1-694228`, 4/4 processes attempted.
- Final statuses: 19 `gqa_materialization_detected`, one `aborted`; no point was
  omitted or rerun.
- Full machine report:
  `artifacts/phase3_reports/phase3-g1-20260722t115413439499z-457123b1-e225cd`,
  report SHA-256
  `060a88283f083e281692a2c471d279da9bfc635e0f513e2dca588ed729d85c7d`,
  checksum-ledger SHA-256
  `9590525b5f3afa53ab55729e123300a14082077457a5ad8ebd4ddd424f0a2077`.
- Independent source-backed validation: PASS, no errors.

## G1 findings

Nineteen runs had exact KV-head cache geometry, no query-head storage, passing
source audit, and no observed query-head-sized KV temporary. Across 15 fixed-L
decode audits plus 64 growing step audits, however, all 79 operator traces
exposed only `aten.scaled_dot_product_attention.default`; none satisfied the
frozen direct fused-kernel proof predicate. The terminal
`gqa_materialization_detected` status is therefore a fail-closed taxonomy. It
does not support a positive claim that physical KV materialization was
observed.

All 19 same-run MHA controls also exposed only
`aten.scaled_dot_product_attention.default`; all 19 failed the frozen
lower-level fused-operator proof and recorded no query-head-sized KV
temporary. Thus the required MHA control did not supply the missing dispatch
evidence either.

All 11 runtime-complete eager audits issued allocator traffic: 1,066 events in
each fixed-L audit and 17,056 in each 16-step growing-context audit; both batch
sizes occur in each group. Cumulative event bytes ranged from 10,960,908 to
1,005,810,432. Allocated and reserved before/after deltas were zero, which does
not waive the event failure. Eager normal timing was not collected.

All eight fixed-L graph processes, covering six distinct `(batch, context)`
shapes with three processes at B1/L4096, captured and replayed with stable
pointers, exact consecutive replay checksums, eager/graph agreement inside the
frozen tolerance, and zero replay allocation events/deltas. Graph timing
remains claim-ineligible admission evidence. The three fixed-L B1/L4096 graph
process medians were 11.74265846875, 11.748611125, and 11.74608940625 ms; median
11.74608940625 ms and CV 0.02543787699280443%, with temperature 48-56 C,
SM clock 2610-2827 MHz, and power 277.87-409.01 W. No eager stability summary
exists, so the Phase 3 stability criterion is PARTIAL and G5 is NOT EVALUATED.

The six stability-point `validation/worker_result.json` records contain the
same output-tensor checksum, computed outside timing:
`f53f3bfcb12f6bc3fd8d64c3fe03c08939fbc0fd0fae5fa5b499ac660c4ae8e5`.
Only the three graph records have normal timing. The derived graph-stability
artifact SHA-256 is
`53b6fcb0fa1fb7a8283f5022cf9ed52d03c6f22e3d28492a27eb8455aee79312`.
The all-run checksum criterion still fails because the aborted run has no
final runtime checksum.

One fixed-L B1/L16384 eager worker produced its internal GQA-failure result,
then the final during-process sample reported PID 432362 as `[No data]` in
`compute_apps` while `pmon` no longer listed it, so it was recorded unknown.
The preceding five compute samples identified the same PID/start ticks as the
allowed supervised child; after/release snapshots contained no compute
process. The coordinator preserved stdout/stderr and finalized the point as
`aborted` with `process_audit.passed=false`. This failure was not rerun.

The machine report's passing criteria are exact BF16 backend identity,
eager/graph numerical agreement, graph replay no-allocation, independent
process identities, no formal paper claim, and immutable checksum-valid
artifacts. Overall G1 is FAIL. G0 remains PASS; G2-G5 remain NOT EVALUATED;
Full Scan remains CLOSED.

## Governance and next action

Every Phase 3 result records `quality_status=unvalidated`,
`claim_eligibility=performance_only`, `performance_claim_eligible=false`, and
`measurement_scope=native_host_admission`. No quality dependency, quality
benchmark, profiler, paper-result directory, speedup, knee, compression, HBM,
capacity, or model-comparison claim was produced. B-009 and B-010 remain open.

Phase 4 must not begin. Minimum remediation is B-011 through B-013: directly
prove the geometry-specific fused native-GQA dispatch and corresponding MHA
control, eliminate all eager decode allocation events without changing the
endpoint, and make terminal process monitoring race-safe. Report-lifecycle
hardening in R-026 is also required before publishing the next admission
report. Then use a new Git SHA and run both complete bounded campaigns with
new run IDs. Preserve every current artifact.

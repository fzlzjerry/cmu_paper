# Decision 0011: Phase 3 run session and isolated-control join

- Status: Accepted
- Date: 2026-07-23
- Authority: AGENTS.md, the Phase 3 remediation instruction, Decisions 0007
  through 0010, and the immutable Phase 3 G1 report
- Supersedes: Decision 0010 only where it can be read to require the B-011
  direct-SDPA control trace to invoke the full-model measurement callable
- Superseded by: none

## Context

Decision 0010 correctly requires the allocation audit and normal timing to use
one exact endpoint, cache, fixture set, and graph. Its wording also grouped the
device-dispatch trace with those exact-callable requirements. The Phase 3
remediation contract, however, preregisters an isolated GQA attention control
and an MHA control with frozen geometry, backend, build, device, dtype, causal
semantics, and warmup. The purpose of that trace is to expose the actual CUDA
attention kernel without the thousands of unrelated full-model kernels. It is
not a benchmark timing operation.

These proofs answer different questions. The B-011 control proves the forced
Flash device dispatch and absence of a preceding replication/materialization
kernel for the exact attention geometry and actual cache views. The B-012
full-endpoint audit proves that the measured model operation does not allocate
or retain an expanded K/V representation. A valid GQA verdict requires both,
joined to one operation key, backend, source set, cache backing storage, cache
views, and geometry. Requiring the isolated control to be the full-model
callable would obscure rather than strengthen the kernel proof.

The current endpoint audit also models each decode step as an independently
constructed endpoint and restores it from a full-capacity device clone. That
cannot preserve the frozen growing trajectory, makes fixed eager timing
one-shot, captures graphs before the preregistered warmup, and scales the
largest growing witness to about 1.51 TB of logical D2H traffic. The proof must
be run-level and stage-bounded.

## Decision

### Isolated B-011 control

B-011 continues to use a direct, isolated PyTorch SDPA control for GQA and an
MHA control. Both controls must:

- use the exact operation key, frozen public SDPA API, forced Flash backend,
  software build, GPU, batch, attended context, dtype, head dimension, query
  length, scale, and causal semantics;
- use GQA key/value views into the run session's actual cache backing storage;
- record and independently revalidate the cache base pointers, view pointers,
  shapes, strides, offsets, capacity, and layout fingerprint;
- use a deterministic, session-owned query control and a separately labelled
  MHA key/value control;
- preserve raw CUDA Chrome traces and parsed kernel/correlation/stream order;
- remain `run_kind=dispatch_audit`, with no duration admitted to benchmark
  timing; and
- fail on backend fallback, absent device kernels, preceding replication/copy
  activity, or an unrelated kernel family.

The direct control is not evidence that the full model used the same Python
callable. The exact full-model allocation audit, endpoint/source binding, and
session provenance supply that separate proof. The combined verdict may be
`gqa_nonmaterialization_verified` only after independent replay joins both
proofs by raw checksums and the exact identities above.

### One run-level execution state

Every worker process constructs one factory-sealed run session:

- fixed-L owns one model, endpoint, cache, fixture set, and repeatable decode
  callable;
- CUDA Graph fixed-L additionally owns one captured graph, inner graph, output
  tensor, stream, and replay callable; and
- growing-context owns one model, endpoint, cache, complete 16-token fixture,
  all precreated positions/RoPE values/views, and 16 ordered step adapters.

Per-operation audit adapters may carry an operation key, step-specific views,
production allocation binding, and witness callbacks. They must not construct
or own a separate endpoint or cache.

The one-way state order is:

`constructing -> normal_warmup_complete -> graph_captured_if_needed ->`
`auditing -> audits_semantically_verified -> restored ->`
`audit_buffers_released -> provenance_committed -> ready -> measuring ->`
`measured -> postvalidated`.

Any failure before `ready` permanently invalidates the session. Readiness is a
private-factory, one-shot live validation and is never granted by a serialized
boolean.

### Frozen warmup and timing order

Fixed-L performs exactly 16 normal eager decode warmups after prefill and
fixed-lane preparation. Graph capture occurs only after those 16 warmups. The
allocation audit replays the exact retained graph, and timing replays that
same graph object without recapture. Eager timing repeats the exact session
decode callable for the preregistered 32 operations by 5 batches.

Growing-context performs exactly one complete 16-step warmup trajectory, then
executes the frozen reset/prefill preparation outside timing. Its 16 audit
adapters execute in order over the same endpoint and cache. After all audits,
the session resets/prefills once outside timing and measures one ordered
16-step trajectory. No reset, prefix copy, fixture construction, D2H transfer,
host synchronization, or audit operation may enter either measured region.

The existing `measure_fixed_batches` and `measure_growing_trajectory` timing
boundaries remain unchanged.

### Restoration without a full cache clone

A full-capacity device clone is forbidden as normal production audit state.
Fixed-L restores only the destination slot and exact lifecycle metadata needed
between instrumented audit attempts. Growing audit preparation uses the
frozen reset/prefill path and replays prior steps outside instrumentation.
Neither preparation path is callable from normal timing.

Growing reset intentionally does not zero inactive future cache bytes because
the frozen runner did not do so. The proof instead records the run-boundary
full-cache state, requires exact base/view identities, checksum-chains every
destination transition, and proves that each stale future slot is overwritten
before it can enter the attended prefix.

One initial and one final run-level full-capacity witness are retained. Per-step
witnesses contain only the exact destination, historical-state digest,
lifecycle transition, output, and chained predecessor digest. Repeating a
full-capacity D2H witness for every growing step is forbidden.

### Stage-specific resource admission

Resource admission uses the maximum live set for each stage, then the maximum
of those mutually exclusive peaks. It must not sum allocations that cannot be
live together. The calculation includes:

- resident model, cache, endpoint workspace, fixtures, output, and retained
  graph storage where applicable;
- only the restoration storage live in the current audit stage;
- MHA/control tensors only while their control is live;
- cgroup-aware host availability;
- descriptor-pinned staging filesystem availability; and
- cumulative limits of 256 MiB per raw file, 1 GiB per run, a 16 MiB run index,
  and 512 files per run.

Audit-only query/MHA tensors, profiler objects, allocator snapshots, and
restoration buffers are released before timing. Device allocated/reserved and
host/staging state are recorded before construction, after cache allocation,
after audit setup, and after cleanup. PyTorch reserved blocks may remain only
when they arise from the same frozen warmup/control path and are explicitly
reported; no `empty_cache` or other allocator-state change may silently
invalidate the warmed allocation proof.

### Raw evidence and independent replay

Every operation retains the exact 18 B-011/B-012 raw artifacts plus
`phase3_session_provenance`. Provenance contains a checksum ledger over the
other 18 kinds and the run/session, receipt, source, endpoint/cache/view,
callable, graph, fixture, restoration, dispatch, allocation-join, paired
control, verdict, and criterion identities. Provenance is written last, before
the final live readiness validation, and is explicitly ineligible for normal
timing.

The coordinator preserves all raw bytes first, then independently rebuilds
dispatch, paired controls, allocation facts, the combined GQA verdict, and
provenance. For growing runs it additionally proves one session ID, one
endpoint/cache/model/tokenizer and base storage identity, and exactly 16
ordered chained operations. Only successful independent replay can set the
scientific raw-audit outcome and terminal eligibility to true.

## Consequences

- The current one-operation measurement handoff cannot enter timing.
- The current per-step endpoint construction and full-cache clone/witness plan
  must be replaced, not waived.
- B-011 remains an isolated direct device-kernel proof exactly as required by
  the remediation contract; B-012 remains an exact full-endpoint proof.
- The empty production eager allocation catalog remains fail-closed until
  source-backed event templates and raw paired-control verification are
  frozen in a later decision amendment.
- No campaign may run until run-session CPU, CUDA, graph, coordinator replay,
  maximum-point resource, process, report, make, and immutability gates pass.
- This decision changes no model, tokenizer, weights, dtype, backend, grid,
  numerical tolerance, graph semantics, or timing boundary.

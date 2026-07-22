# Decision 0010: Phase 3 audit-to-measurement session identity

- Status: Accepted
- Date: 2026-07-22
- Authority: AGENTS.md, the Phase 3 remediation instruction, Decisions 0007,
  0008, and 0009, and the immutable Phase 3 G1 report
- Supersedes: none
- Superseded by: none

## Context

The immutable Phase 3 evidence cannot establish that the decode operation
examined by the dispatch and allocation audits was the operation later used by
the benchmark timing runner. Reconstructing an equivalent endpoint, cache,
fixture, graph, or callable after audit would leave an identity gap even if its
shapes and values happened to match. That gap is scientifically material: GQA
non-materialization and allocation attribution are properties of an exact
execution target, not merely of a geometry.

The frozen runners have two different state machines. A fixed-L point repeats
one decode geometry into a fixed destination slot. A growing-context point is
one ordered 16-step trajectory over a shared endpoint and cache. Treating its
steps as 16 independently constructed endpoints would change the benchmark
semantics. CUDA Graph points likewise require the exact captured graph to be
audited and replayed; recapture after audit would create a different target.

The factory-sealed frozen-model receipt is also an in-process identity proof.
Storing only the public model identity and execution-source manifest digest in
primary evidence does not let the coordinator independently prove that the
loaded model receipt named the exact loader source in that manifest.

## Decision

### Sealed session

Every Phase 3 process point must construct exactly one factory-sealed
audit-to-measurement session before any measurement begins. The session owns
strong references to the loaded model and tokenizer, endpoint, cache, worker
fixtures, position tensors and embeddings, output storage, and, when
applicable, the captured CUDA Graph and its replay callable.

The session is measurement-ready only after all of the following have passed:

1. the complete execution-source manifest is revalidated against both Git
   blobs and live bytes;
2. the full factory-sealed model-load receipt is revalidated against the live
   model, tokenizer, parameters, storage, and snapshot ledger;
3. the receipt loader-source digest equals the manifest's model-loader digest;
4. fixture, storage, tensor-view, cache-layout, backend, source, and resource
   witnesses all bind to the same operation key and live objects;
5. required device-dispatch evidence, allocator evidence, paired GQA/MHA
   controls, and their raw files have completed and passed semantic replay;
6. Decision 0008's combined GQA verdict is
   `gqa_nonmaterialization_verified` and Decision 0009's lane-specific
   allocation criterion passes;
7. the exact source manifest, object/storage identities, content digests,
   cache state, operation key, and raw-evidence checksums are committed into a
   canonical session-provenance record; and
8. a final live validation immediately before `measurement_started` confirms
   that none of those identities or admitted resources changed.

The session seal is private to the factory. Public dataclass construction,
deserialization, copied dictionaries, matching object IDs without retained
strong references, or shape/value equivalence cannot create a ready session.
Readiness is one-way and occurs only after the final validation. Any mutation,
replacement, garbage collection, source change, resource failure, or evidence
join mismatch invalidates the session and closes timing.

### Fixed-L lane

The fixed-L audit record represents the single frozen decode geometry for that
process point. Audit warmup, device tracing, allocator collection, graph
capture when selected, and normal timing use the exact session-owned endpoint,
cache, fixtures, destination slot, and callable. Preparation and cache reset
occur only outside the measured region. Normal timing may repeat the sealed
callable according to the preregistered count and batches; it must not rebuild
or recapture it.

For CUDA Graph mode, allocation admission applies to replay of the exact
captured graph and Decision 0009's strict zero-allocation replay criterion.
Normal timing replays that same captured graph object. A post-audit recapture
is forbidden.

### Growing-context lane

All 16 operation keys belong to one ordered session over one endpoint and one
cache. Each raw audit record binds its step-specific token view, cache
position, position embedding, active cache views, and expected state
transition. Collection follows decode-step order. Per-step evidence may not be
obtained from separately constructed endpoints or caches.

Audit collection may advance the shared cache while examining the trajectory.
After all 16 steps pass, the session restores and verifies the frozen prefix
state once, outside timing. The measured region then executes the same ordered
16 step callables with no reset, prefix copy, fixture construction, host
synchronization, tensor-to-host conversion, or audit instrumentation inside
that region. The existing one-trajectory timing boundary is unchanged.

### Evidence and independent join

The completed raw-audit kind set gains a required canonical session-provenance
record for every operation. It identifies the session, operation key, exact
model receipt digest, loader-source digest, source-manifest digest, fixture and
cache-state digests, object/storage/view identities, callable identity, graph
identity where applicable, and checksums of the raw evidence used by the
combined verdict. The record is audit provenance, never benchmark timing.

Primary worker evidence must contain a strict model-load receipt summary and
session summary. The coordinator independently validates at least:

- receipt digest and frozen-identity digest against the serialized model
  identity;
- receipt loader-source digest against the execution-source manifest;
- session source-manifest, model, backend, hardware, software, plan, point,
  cache-layout, and operation identities against coordinator-owned values; and
- session raw-evidence digests against the no-follow ingested raw files.

Serialized `passed`, `verified`, or `measurement_ready` booleans are never
trusted without these independent derivations.

### Resource admission

Resource checks are conservative safety bounds, not claims that Python has
reserved physical memory or disk space. They use live CUDA availability,
cgroup-aware host limits, and a no-follow, descriptor-pinned staging directory
identity. The bound includes the maximum permitted raw-audit file, run-index,
and per-run evidence sizes. Device and host bounds include the exact endpoint,
cache, baseline-witness, graph, trace, and transient terms that remain live at
each collection stage. Unknown terms fail admission.

The staging root and every source/fixture file used for identity capture must
be a regular, singly linked file or directory owned as required by the current
process, opened without symlink traversal, and bound by descriptor metadata.
Replacement or aliasing between admission, collection, final readiness, and
raw-evidence flush is a failure.

### Benchmark and governance boundaries

Profiler- or allocator-instrumented operations are `dispatch_audit` or
allocation-audit evidence only. Their durations never enter normal timing.
This decision does not change the model, tokenizer, weights, dtype, backend,
grid, graph semantics, timing boundary, or numerical tolerance. It does not
authorize a pilot, full scan, performance profiling, quality execution, or a
campaign rerun. All targeted remediation gates must pass first.

## Consequences

- Existing Phase 3 artifacts remain immutable and cannot satisfy this new
  proof contract retroactively.
- A producer API that only emits raw files but cannot return the same sealed
  execution session to the runner fails closed.
- The legacy high-level operator and allocator summaries may remain as
  diagnostics, but they cannot override the combined raw-evidence verdicts.
- Any need to reconstruct an endpoint, cache, fixture, or graph between audit
  and timing is a blocker requiring a new scientific decision.
- The full fresh campaigns remain prohibited until CPU, CUDA, graph, source,
  process, report-lifecycle, and immutable-evidence admission tests pass.

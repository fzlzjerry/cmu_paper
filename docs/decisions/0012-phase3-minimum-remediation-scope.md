# Decision 0012: Minimum Phase 3 remediation scope

- Status: Accepted
- Date: 2026-07-23
- Authority: the Phase 3 scope-reduction directive, AGENTS.md, the immutable Phase 3 G1 report, and Decisions 0007 through 0011
- Supersedes: the architectural and evidence-packaging portions of Decisions 0010 and 0011 where they require generic session machinery, adversarial object sealing, or exactly 19 raw files per operation
- Superseded by: none

## Context

The remediation branch accumulated an uncommitted endpoint, session, and source-governance stack of more than twelve thousand lines. Much of it addressed adversarial Python object forgery, filesystem races outside formal publication, generic dependency injection, duplicated cache witnesses, and exact per-operation evidence sharding. Those mechanisms are not required to answer the scientific G1 questions and made the measurement path harder to audit.

This decision narrows architecture and packaging only. It does not weaken GQA device-kernel proof, exact endpoint allocation attribution, graph zero-allocation, process ownership, report finalization, numerical validation, immutability, or campaign admission.

## Decision

### Components kept

- Forced-Flash isolated GQA and MHA CUDA traces, parsed device kernels, kernel-family evidence, no-preceding-materialization checks, and source/shape/stride/native-KV evidence.
- Exact full-endpoint allocator history and snapshot evidence, Python and C++ stacks, lifetimes, requested and block sizes, segment and device allocation facts, persistent allocator deltas, and device-used-memory deltas.
- Expanded-KV, cache-growth, context-scaled-unknown, unknown-event, and graph zero-allocation failures.
- The frozen model/tokenizer receipt digest, execution Git SHA, existing source pinning, operation keys, raw checksums, and coordinator-owned semantic replay.
- The committed B-013 registry, PID start-time/pidfd/handshake behavior and tests.
- The committed R-026 staging, validation, COMPLETE-last, checksum, immutability, no-replace publication, and tests.

### Components simplified

- One concrete endpoint-specific session replaces generic run-session and per-operation handoff frameworks.
- The session retains only the receipt digest, one endpoint/cache, base pointers and required views, operation keys and fixtures, exact measurement callable(s), minimal state witnesses, evidence references, and a small one-way lifecycle.
- Fixed eager uses 16 warmups, one destination-slot restoration buffer, audit, restoration, and the same repeatable callable for timing.
- Fixed graph uses 16 eager warmups, one capture, and the same retained graph for audit and timing.
- Growing context uses one endpoint/cache, one 16-step warmup trajectory, reset/prefill, ordered audits, reset/prefill, and one ordered measured trajectory.
- Raw evidence is consolidated into raw GQA/MHA traces, one dispatch JSON, raw allocator history/snapshot, one allocation JSON, one run-level session/provenance JSON, and one index/checksum ledger. Mandatory raw inputs remain checksum-bound and independently replayed.

### Components removed

- Generic dependency-injection session frameworks and reusable sealing machinery.
- Anti-copy, anti-pickle, deep object-graph sealing, and adversarial Python forgery tests.
- Identity records for every intermediate tensor and duplicate cache witnesses.
- Full-capacity device clones and full-cache D2H evidence per growing step.
- Dead diagnostic readiness paths and isolated per-step measurement handoffs.
- New full-package filesystem/source threat defenses beyond existing source pinning and formal artifact publication.
- The exact 19-file-per-operation packaging requirement.

### Minimum lifecycle

The concrete session has only these effective phases:

`constructed -> warmed -> audited -> restored -> ready -> measured`

Timing cannot begin before audited evidence passes, exact state is restored, and audit-only buffers are released. Any failure before ready permanently rejects the point.

### Measured-region exclusions

The exact timed callable contains no prepare, reset, copy, D2H transfer, host synchronization, profiler, allocator instrumentation, or evidence serialization. Forced-Flash execution encloses the complete lane, and fallback remains an error.

## Consequences

- Current uncommitted generic endpoint/session/source work must be reduced before integration.
- No campaign runs until the original scientific pre-campaign gates pass and the tree is clean.
- If raw CUDA dispatch or exact endpoint allocation evidence cannot prove G1 for the frozen backend, B-011 or B-012 remains open; no additional framework or criterion waiver is permitted.
- Decisions 0010 and 0011 remain historical records but no longer mandate their superseded architecture or exact evidence-file count.

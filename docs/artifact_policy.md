# Append-only artifact audit policy

Status: Phase 2 local controls implemented and tested; durable backing,
external attestation, and immutable locator publication remain open as B-009.

## Scope

`/artifacts/` is ignored by Git because the research contract excludes raw
experiment outputs from ordinary code PRs. Ignoring the directory does not
waive provenance, retention, or append-only requirements. No claim-bearing run
may begin until B-009 is closed. Phase 2 implements the local artifact
lifecycle below, but durable storage and publication location remain
unresolved and must be recorded and exercised before the first claim-bearing
run.

## E00 boundary

Decision 0002 permits the prerequisite, non-performance G0 certification to
write append-only Git-tracked evidence under `docs/evidence/e00/<run_id>/`.
This narrow path must use unique IDs, atomic finalization, checksums, and a
completion marker. It does not authorize G1-G5 admission or performance runs,
which remain blocked by B-009 until durable backing and locator publication
are demonstrated.

## Phase 2 local implementation

`src/kvbench/runtime/artifacts.py` provides a caller-configurable local store.
It validates run IDs and roots; permanently reserves IDs; creates unique
same-filesystem staging directories; writes files exclusively with flush and
`fsync`; records strict initial and lifecycle documents; prevents final
manifests from changing initial command/provenance; rejects symlinks,
hardlinks, traversal, unsafe control paths, and formal-evidence-root overlap;
and uses Linux `renameat2(RENAME_NOREPLACE)` for atomic promotion.

Finalization schema-validates the terminal manifest, builds a complete
artifact inventory and canonical SHA-256 ledger, writes a strict completion
marker last, independently revalidates the whole staging tree, removes write
bits, and atomically promotes it. Successful and failed terminal runs use the
same preservation path. Interrupted stages and reservations remain visible
and are never silently reused. Supported APIs expose no edit operation after
finalization. Tests exercise success, failure, interruption, concurrency,
tampering, identity changes, unsafe paths/links, and formal-root rejection
using temporary directories only.

## Control disposition

1. Unique staging plus no-replace same-filesystem promotion: implemented and
   unit tested locally.
2. Existing/reserved ID rejection and no completed-run edit API: implemented
   and unit tested locally.
3. Complete SHA-256 ledger, artifact inventory, and completion marker written
   last: implemented and tamper tested locally.
4. Code/dirty/container/hardware/software/model/method/contract/config/run/seed/
   replicate provenance: required by strict manifests; formal execution remains
   impossible while referenced identities are unresolved.
5. Independent final validation before promotion and CLI validation before
   analysis: implemented; checksum failure preserves and invalidates evidence.
6. Append-only durable copy, external attestation, immutable locator, retention,
   retrieval verification, and report/release publication: not implemented;
   this is the remaining B-009 acceptance condition.
7. Separate timing, nsys, and ncu schemas and profiler-timing exclusion:
   implemented as contracts only; no such run has executed.

## Audit boundary

Git records the writer, schemas, validation code, phase reports, and immutable
artifact locators/digests, but not normal raw run payloads. Every admission or
scientific result must remain traceable from its report to a complete durable
run directory whose checksum ledger verifies independently. Local chmod may be
used as defense in depth after finalization, but it is not treated as the sole
immutability or retention mechanism. Local ledgers are not external
attestations: a storage principal capable of coherently replacing every local
control file remains in scope for B-009.

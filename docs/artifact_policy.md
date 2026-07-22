# Append-only artifact audit policy

Status: Phase 0 policy; implementation and durable backing are E01 work.

## Scope

`/artifacts/` is ignored by Git because the research contract excludes raw
experiment outputs from ordinary code PRs. Ignoring the directory does not
waive provenance, retention, or append-only requirements. No claim-bearing run
may begin until E01 implements the controls below and B-009 is closed. Durable
storage and publication location are intentionally not selected in Phase 0;
that operational choice must be recorded before the first such run.

## Required E01 controls

1. Write each run into a unique staging directory and atomically rename it on
   the same filesystem only after validation succeeds.
2. Reject an existing run ID; never edit, overwrite, or delete a completed run
   through project tooling.
3. Generate a per-run SHA-256 ledger covering every file and a completion marker
   written last.
4. Record code SHA, dirty-state refusal, container digest, hardware manifest,
   complete config, run kind, seeds, timestamps, and process-replicate identity.
5. Verify the ledger after finalization and before analysis, transfer, or
   publication; checksum failure invalidates the run without deleting it.
6. Copy finalized runs to the selected append-only durable store and record its
   immutable locator plus manifest digest in the phase report or release index.
7. Keep timing, nsys, and ncu run kinds separate in storage and analysis; only
   timing runs may feed ordinary benchmark latency claims.

## Audit boundary

Git records the writer, schemas, validation code, phase reports, and immutable
artifact locators/digests, but not normal raw run payloads. Every admission or
scientific result must remain traceable from its report to a complete durable
run directory whose checksum ledger verifies independently. Local chmod may be
used as defense in depth after finalization, but it is not treated as the sole
immutability or retention mechanism.

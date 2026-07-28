# Decision 0022: Phase 9 BLOCKED report Git custody

- Status: Accepted
- Date: 2026-07-28
- Authority: explicit operator request after public Phase 9P patch custody
- Superseded by: none

## Context

The Phase 9 calibration attempt ended with a `BLOCKED` final report at clean
commit `f2c6475f09cdf6e9660552eb23c91b03e386aa59`. Phase 9P treated that report as
an immutable conversation handoff and correctly left the completed Phase 9P
records unchanged. The report itself nevertheless had no repository path, so a
fresh clone did not retain its full text.

The operator requires all durable Phase 9/9P handoff material to survive loss
of the current server and explicitly asked for this missing report to be kept
on public `main`.

## Decision

1. Store the original assistant final at `docs/phase_reports/phase9-kvquant-calibration-blocked.md`.
2. Select the unique 8,605-character report beginning with starting HEAD
   `f2c6475f09cdf6e9660552eb23c91b03e386aa59`, not the earlier Phase 9 entry
   blocker report beginning at `1568de5...`.
3. Bind the original conversation bytes by SHA-256 `1d3d0b9d921aa49eeb4c81cb94099f0fe1386e806732d5238baacf9e6e74d4cd`.
4. Repository storage changes no report body bytes and appends only the one
   POSIX terminal newline required by repository text policy. The stored
   SHA-256 is `05bbc9d21fe4bff900bd141ddc7f6daec226848178f8c0b78b7ecdaba2c180b7`.
5. Preserve the original transport metadata block as part of the recovered
   assistant output; do not rewrite or summarize it.
6. Record extraction identity and both digests in `docs/evidence/phase9/blocked-report-custody.json`.
7. Do not rewrite Decision 0020, the Phase 9P report, or completed Phase 9P
   evidence. This later custody action does not retroactively alter Phase 9P.
8. This action does not restart calibration, begin Phase 10, enable KVQuant,
   create performance/profiler/quality data, or change any existing adapter or
   Measurement Container authority.

## Consequences

The full Phase 9 BLOCKED report is now recoverable from `main`, independently
checksum-verifiable, and distinct from the later Phase 9P PASS report. Its
scientific conclusion remains unchanged: no Phase 9 calibration artifact was
produced.

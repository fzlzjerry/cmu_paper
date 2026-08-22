# Decision 0037 — Phase 13 persistent chunked prefix construction

- Status: Accepted
- Date: 2026-08-22
- Scope: untimed Phase 13 prefix construction and exact state reuse only

## Context

The stopped non-claim-bearing campaign
`phase13-20260821t151036274823z-b27442c4-8a39d6` completed
`prefix-kvq4-b8-l16384` before its coordinator was terminated. Together with
its previously completed states, it supplied 168 exact
configuration-by-batch-by-context snapshots. All 168 state payloads were
validated byte-for-byte and frozen locally as
`phase13-prefix-seed-20260822t185619z-b27442c4-168`.

The seed contains 883,518,934,464 logical state bytes. Its catalog SHA-256 is
`c432013892d571d32e8cc6ff0c2c3cefb7415285742bf3c5682ef8fa67020f3e` and its
COMPLETE digest is
`db218e817b80aa4a91c2567bcd141afab1fff3ee52ce0247730ceecfe8b27ee4`.
The q4 B=8/L=16384 state SHA-256 remains
`175cbb47cdf81b96fa6ca8c8fbf3011413ba78e385bde7ea41cea01b83cec08d`.

## Decision

A fresh Pilot campaign may hardlink these read-only states into its own
persistent catalog and must construct only the 60 absent KVQuant snapshots.
Reuse is allowed only for an exact configuration, batch, context, layout, and
method-fingerprint match. Timing samples, cache objects, pointers, CUDA Graphs,
and runtime prefix state are never reused. Missing, altered, or mismatched
states fail closed and are not recomputed under a reused snapshot identity.

KVQuant setup packing processes one batch row and at most 128 tokens at a time
through caller-owned fixed scratch. The scratch is preallocated for prefix
construction, is independent of context length, and is included in the frozen
end-to-end 0.88 feasibility calculation. It must not materialize a complete
context FP32 Key or Value copy. Prefix restoration and this construction both
remain outside formal timing.

The preserved q4 B=8/L=16384 snapshot is the exact oracle for cache bytes,
lifecycle, layout, allocation accounting, output checksum, kernel path, and
CUDA Graph behavior. The optimized build's observed duration is recorded and
used only to estimate remaining setup time; no duration is a PASS threshold.

## Preservation

This decision changes no adapter, CUDA source, quantization, cache layout,
input, grid, feasibility limit, decode path, or timing boundary. All stopped
campaigns remain immutable, and no old timing sample enters the successor
Pilot. Phase 14 remains deferred.

# Phase 13P-C CUDA-context isolation remediation

- Status: PASS
- Execution Git SHA: `6ddd5ead6d6375e6ed38cd574c32cc1a24f02596`
- Decision: 0035 (Accepted)
- Authorized container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`

The complete 30-case exact-batch prefix-equivalence matrix ran in one dedicated
child process. The parent coordinator did not load the model or initialize
CUDA. All 30 direct/restored cases and all 20 cross-batch fail-closed controls
passed. The child exited successfully, its evidence validated, and both the
pre-child and post-child GPU snapshots were idle with no remaining, foreign,
or unknown CUDA process.

The append-only remediation artifact is
`artifacts/phase13pc/phase13pc-20260804t104145767130z-6ddd5ead-d637d9`, root
`da93d4a38b79129885d25e2d8d763ccc04c4826ae679460d168d825817e753f4`.
It was published COMPLETE-last to
`r2://kvbench-artifacts/kvbench/sha256/da93d4a38b79129885d25e2d8d763ccc04c4826ae679460d168d825817e753f4/`;
one clean retrieval passed inventory, checksum-ledger, COMPLETE, and root
validation under the indefinite Bucket Lock rule.

This remediation changed only the Pilot harness process boundary. It changed
no adapter, CUDA source, cache layout, grid, feasibility result, timing
boundary, calibration, fixture, or admission evidence. No timing was collected.
Every stopped Phase 13 campaign remains immutable, unpublished, and
non-claim-bearing. A wholly new Pilot campaign is now permitted.

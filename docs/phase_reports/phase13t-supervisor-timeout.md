# Phase 13T supervisor-timeout remediation

Status: PASS

Decision 0032 replaces the stopped Phase 13R2 campaign's single 7,200-second
worker deadline with ordered, finite, geometry-bound stage deadlines. The
exact authorized-container diagnostic of `kvq2/B=1/L=131072` completed
normally: it was neither stalled nor deadlocked. The supervisor-observed stage
durations were 11.56 seconds for model load, 27,550.86 seconds for prefix and
pre-capture validation, 3,590.45 seconds for Graph/correctness, 10,802.08
seconds for warmup/audit, 3,825.24 seconds for measurement plus untimed
validation, and 0.002 seconds for finalization.

The old deadline expired during normal prefix work. No adapter, CUDA, cache
layout, grid, feasibility result, calibration, fixture, admission evidence,
timing boundary, or analysis rule changed. The three finalized stopped Pilot
campaigns and two stopped staging reservations remain checksum-identical.

The remediation artifact
`phase13t-20260803t003012058490z-aad35807-7d1bf2` contains 16 objects and has
root `5e98c8103c6e15ca0877e39f7b9363a208ba8722bb6993ef67a04a06c9f795cb`.
It reproduces the frozen 810-record matrix as 684 feasible and 126
capacity-infeasible records. Package lock, full tests, full checks, and the
authorized-container Phase 13T tests pass.

R2 publication used conditional content-addressed writes with `COMPLETE` last.
Exactly one clean retrieval validated every object, inventory, ledger, marker,
and root at
`r2://kvbench-artifacts/kvbench/sha256/5e98c8103c6e15ca0877e39f7b9363a208ba8722bb6993ef67a04a06c9f795cb/`.

B-020 is resolved for process supervision. Phase 13 remains incomplete until a
wholly new Pilot campaign completes. Phase 14, Full Scan, profiling, quality,
and performance claims remain closed.

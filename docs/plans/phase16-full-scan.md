# Phase 16 Full Scan Plan

Phase 16 freezes the ten admitted configurations `bf16`, `tq_4bit_nc`,
`tq_k3v4_nc`, `tq_3bit_nc`, `k4v4`, `k2v4`, `k2v2`, `kvq4`, `kvq3`, and
`kvq2` at exact current fingerprints. Decision 0039 admits batches
`{1,2,4,8,16}` and prefix schema `kvbench-phase16g-prefix-state-3.0.0`.

The base grid is the nine labels `{4096,8192,16384,24576,32768,49152,
65536,98304,131072}` at every configuration and batch. Label 131072 maps to
historical prefix 131071 and total attended length 131072. The exact 84
included Phase 13D adaptive rows are added without regeneration, yielding 534
logical points and 2670 process records. Execution is fixed-L CUDA Graph with
64 warmups, 256 measured operations, five independent processes, and seeds
20260830, 20260831, 20260901, 20260902, and 20260903. The complete blocked
orders are checksum-frozen in `phase16-full-scan-execution-orders.json`.

Feasibility is recomputed from current owned-byte formulas at the frozen 0.88
memory limit, including model, cache payload and metadata, residual/sink/sparse
storage, padding, persistent and q4 workspaces, Graph reserve, and safety
margin. The entry calculation yields 441 feasible and 93 capacity-infeasible
logical points; any drift fails before CUDA. Known-infeasible points receive
five terminal records and are not launched.

An exact checksum-valid Phase 13, Phase 13D, or Phase 16G prefix is restored
read-only only when its layout-bound cache implementation blob is byte-exact at
the current execution HEAD. Decision 0039 changed the compressed cache source
identity, so legacy compressed snapshots fail that exact check and are rebuilt
deterministically outside timing; unchanged BF16 and current Phase 16G snapshots
remain reusable. KVQuant direct construction uses the already admitted fixed
bounded chunk prefill. Each timing process owns fresh cache and workspace
buffers. No runtime cache, pointer, or Graph sharing occurs, and no new
persistent prefix cache is created. The Phase 13 stage-timeout scaling formulas
are retained exactly and their input scope is extended only to the frozen Phase
16 batch/context set. The reused worker's fixed-L context mapper is bound per
process to that process record's exact frozen base or Phase 13D adaptive label;
the mapping is restored when the worker exits.

Each replicate is an independently sealed segment. Isolated point failures are
preserved and execution continues. An actual foreign GPU process, source,
container, fingerprint, or timing-critical drift stops only the current
segment. Only genuine incomplete or infrastructure-failed records are eligible
for explicitly linked replacement; slow, unstable, non-monotonic, and valid
records are never rerun. A point is stable at CV <= 0.03 using at least three
valid independent processes. Phase 17 coverage requires at least three stable
logical contexts per configuration/batch, or every feasible context when fewer
than three exist.

The final outer bundle references five segment roots and contains the frozen
grids, feasibility, indexes, QC, same-work ratios, capacity amplification,
Phase 15 common-point-only profiler features, diagnostics, checksums, and
`COMPLETE`. Each segment and the outer bundle are published content-addressed
to R2 with `COMPLETE` last and a clean retrieval. Timing rows retain
`r_hbm=null`, quality is unvalidated and locked, and performance claims remain
ineligible. Phase 17 is explicitly deferred.

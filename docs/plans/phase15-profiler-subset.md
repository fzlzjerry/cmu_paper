# Phase 15 profiler subset plan

Phase 15 is mechanism-only. Profiler-observed durations use `run_kind=nsys` or
`run_kind=ncu`; they never enter normal timing, Phase 13/14 tables, fits, or
performance claims. Full Scan and quality execution remain closed/locked, and
Phase 16 is deferred.

The detailed anchors are `bf16`, `tq_4bit_nc`, `k4v4`, and `kvq4`. Selection is
checksum-bound in `phase15-profiler-selection.json`. B=1 is sufficient for all
four anchors. For BF16 and TurboQuant the refined knee has insufficient feasible
span, so the deterministic fallback uses the smallest, lower-middle,
upper-middle, and longest stable contexts. KIVI and KVQuant use the frozen
pre/near/post/longest knee-relative rules. This yields 16 unique anchor points
and 32 Nsys eager/Graph traces.

The all-ten Graph-mode B=1 stable-context intersection is
`[4096,8192,16384,24576,32768,49152,65536,98304,131072]`; therefore the common
same-work point is label L=131072, actual historical prefix 131071, total
attended 131072. NCU profiles all ten there plus the 12 nonduplicate anchor
regime points, for 22 profiles total.

One BF16 B=1/L=4096 Graph smoke is required for each Nsight tool before the
full subset. Nsys captures eight warmed decode operations in one profiler NVTX
range; NCU captures one. Warmup is 64 for formal profiles. Model load, prefix
restore, compile/capture, and graph warmup remain outside the range. The single
end-of-range synchronization is explicitly classified as a profiler boundary,
not a method synchronization or timing sample.

NCU metrics are resolved from live SM120 `--query-metrics` and
`--query-sections` output. Unavailable metrics remain explicit. Kernel roles use
profiler-only adapter NVTX evidence, kernel identity, source authority, and the
method manifest; ambiguous kernels remain `unknown`. Canonical `r_hbm` is
BF16/method cache-path DRAM traffic at the exact common point and remains null
unless same-work identity, metrics, and classification are complete.

Isolated tool/export/parse failures are preserved and may receive one targeted
profiler-only retry with a new run ID. CUDA correctness, authority drift, or a
true foreign GPU process fails closed. The append-only bundle is finalized with
`COMPLETE` last, published content-addressed to R2, and cleanly retrieved.


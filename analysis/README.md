# Analysis boundary

This directory is reserved for offline validation, fitting, mechanism analysis,
and figure generation in later phases. Phase 2 intentionally contains no
fitting implementation and produces no scientific result or paper figure.

Future analysis code must:

- read only finalized run directories whose schemas, completion marker,
  inventory, provenance, and SHA-256 ledger validate;
- leave raw samples and completed manifests unchanged;
- reject or explicitly report incomplete, tampered, fallback, and unsupported
  runs;
- admit only normal timing run kinds to ordinary latency analysis;
- keep nsys and ncu evidence separate from timing;
- keep graph-on and graph-off comparisons separate;
- keep same-work latency and capacity amplification separate;
- retain every exclusion and failed/unstable point;
- derive every figure from immutable source artifacts without hand-editing
  data points.

r_hbm may be analyzed only where direct profiler-supported evidence exists.
Nominal or allocated bytes and latency are not physical HBM measurements.

Quality remains a separate, LOCKED lane. No performance/quality join may occur
until PERFORMANCE_DATA_FROZEN exists and the post-performance quality protocol
authorizes execution and exact-fingerprint joining.

# Phase 11R KVQuant Adapter Binding and Admission Rerun

## Scope

Bind the existing static KVQuant Measurement Adapter to Decision 0027 without
changing CUDA source, calibration, fixtures, existing methods, common runners,
or the authorized Measurement Container.

## Binding

- Execution source: commit
  `4b8533b29b04f8c4bf55f688a41fefe20487637b`, tree
  `46f2149a0369d5c97d9a6bc77d57b5f3a5a5fb3b`, aggregate patch
  `bae63bced549479709b10d7f6a8ee35a8f21ec18cc040a7424591cee47c1b0a6`.
- Extension:
  `a79644923ba131e56abe95029e669346dbbb11fd210d2b9f8b2086819ffeaad1`.
- Numerical oracle:
  `kvqref-2e0a0e9022c50cbc6fb497d88cae973e`, root
  `c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec`.
- `kvq4` Value decode uses the Decision 0027 deterministic out-API and one
  caller-owned FP32 workspace `[1, 32, 32, 128]`. `kvq3` and `kvq2` keep their
  existing corrected direct-compressed paths.

## Validation and Admission

Replay all nine corrected fixtures, then reuse the existing execution-path,
allocation, non-default-stream, CUDA Graph, Compute Sanitizer, GQA, artifact,
MethodAdmissionReport, and R2 controls. Run only the frozen nine-point bounded
grid. Publish the finalized admission evidence with `COMPLETE` last and perform
one clean retrieval.

This phase makes no speedup, latency, HBM, capacity, or quality claim. Global
G2-G5, Full Scan, and quality execution remain closed. Phase 12 is deferred.

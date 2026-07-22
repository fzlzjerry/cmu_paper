# Research invariants

1. Raw experiment data is append-only.
2. Never edit, overwrite, or delete artifacts from a completed run.
3. Never report profiler-instrumented timing as normal benchmark timing.
4. No performance claim is valid without:
   - git SHA
   - container digest
   - hardware manifest
   - config
   - raw samples
   - independent process replicates
5. Do not change benchmark semantics and implementation in the same PR.
6. Every CUDA kernel change requires:
   - numerical golden tests
   - Compute Sanitizer
   - CUDA Graph capture/replay test
   - memory-allocation test
7. Do not introduce torch.cat, dynamic allocation, CPU synchronization,
   or tensor-to-host conversion inside the measured decode region.
8. A method must not enter the full scan until all admission gates pass.
9. Same-work speedup and capacity-amplification results must remain separate.
10. Never compare graph-on results against graph-off results.
11. Never cherry-pick the fastest run or selectively rerun slow points.
12. Every exclusion must be recorded with a machine-readable reason.
13. Do not silently fall back to another attention backend or cache dtype.
14. Do not infer physical HBM traffic only from nominal bytes and latency.
15. Do not alter the preregistered grid after seeing performance results.

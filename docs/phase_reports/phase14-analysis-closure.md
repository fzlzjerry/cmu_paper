# Phase 14C Analysis Closure

Status: PASS

- Original campaign: `phase14-20260826t115110887808z-47ba4220-42fc95`
- Original R2 root: `22a613b07c1ee6d3e9a0a7fc81df6065ccc8a10bf783b1a69aded3c2eb8068f0`
- Original report SHA-256: `96210af393acfeaa00a92ed5e41e41f654bc9c7c67205db29abe81a1e9e5acb3`
- Completed execution: 660/660 feasible mode runs; zero runtime failures; zero selective reruns.
- Stable A/B conditions: 105.
- The frozen CV threshold remains 3%; five eager conditions remain `unstable`:
  - `bf16` B=1 L=16384: CV `0.038078505745157795`
  - `bf16` B=1 L=65536: CV `0.038436836137135284`
  - `bf16` B=4 L=4096: CV `0.03157440999181538`
  - `tq_3bit_nc` B=1 L=4096: CV `0.03448706000458669`
  - `tq_3bit_nc` B=4 L=4096: CV `0.039634094975259174`
- Fully identifiable floor+slope+knee comparisons: 14.
- Inconclusive because eager data are unstable: 4 comparisons (`bf16` B=1/B=4 and `tq_3bit_nc` B=1/B=4).
- Ineligible because the eager fit has no positive slope: 2 comparisons (`k4v4` B=4 and `k2v2` B=4).
- Other non-identifiable comparisons: 0.
- Complete launch-floor-only support: 0 of 14 fully identifiable comparisons.
- Floors decrease numerically in 14 of 14 comparisons; 8 meet the frozen meaningful-floor threshold.
- Slopes satisfy the frozen similarity interval in 10 of 14 comparisons.
- Knee shifts among the 14 identifiable comparisons: 9 unchanged, 2 at -6144 tokens, and 3 at -61440 tokens; Graph removes no observed knee.
- Host-minus-device proxy is lower under Graph in 23 of 105 stable A/B conditions. This is not a direct launch-gap measurement.
- Output, backend, cache layout, kernel path, and allocation mismatches: 0.

Scientific interpretation: The observed CUDA Graph effect is heterogeneous and is not explained by a pure launch-floor reduction model. This is a negative mechanism result, not a failed experiment.

Limitations:
- The five unstable eager conditions remain excluded from stable fitted mechanism comparisons; their eager mechanism parameters are not claim-bearing.
- Phase 15 profiling is required for direct launch-gap and physical traffic attribution.

- Phase 15: `READY`
- Full Scan: `CLOSED`
- Quality: `LOCKED`
- Performance claims: ineligible; quality remains unvalidated.
- Closure publication receipt: `docs/evidence/phase14/analysis-closure-r2-publication.json` (self-reference-safe and outside the closure bundle).

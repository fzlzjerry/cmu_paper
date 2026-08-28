# PHASE 14 REPORT

Status: BLOCKED

- Starting HEAD: `7b14043c0563607d92a9dc03d0069144db65508d`
- Execution HEAD: `47ba42209a7e78f159d400e103c0a152bfa40559`
- Campaign ID: `phase14-20260826t115110887808z-47ba4220-42fc95`
- Design: 360 A/B pairs / 720 mode-run records. Of these, 330 pairs / 660
  mode runs were feasible and completed; 30 pairs / 60 mode records were
  predeclared `pair_capacity_infeasible`. Runtime failures and selective
  reruns were both zero.
- Stability: 215 mode points passed and five eager mode points exceeded the
  frozen 3% CV threshold. Maximum overall eager CV was
  `0.039634094975259174` (3.9634095%); maximum Graph CV was
  `0.0019523078093870096` (0.1952308%). The unstable points were BF16 B=1 at
  L=16384 and L=65536, BF16 B=4 at L=4096, and `tq_3bit_nc` B=1/B=4 at
  L=4096. All raw runs are preserved and none was selectively rerun.
- Consistency: output, backend, cache identity, kernel path, and allocation
  mismatches were all zero. Quality remains unvalidated and these mechanism
  observations are not performance-claim eligible.
- Fits: 40 mode fits yielded 34 `knee_observed`, four `unstable_data`, and two
  `no_positive_slope`. Graph mode retained an observed knee in all 20
  configuration/batch rows; eager BF16 and `tq_3bit_nc` rows were unidentified
  because of unstable data, while KIVI `k4v4` B=4 and `k2v2` B=4 had no
  positive eager slope.
- Floor comparison: 16 rows were identifiable. Fitted floor deltas ranged from
  `0.9650276923` to `173.7809129735` ms and floor ratios from
  `1.0031782345` to `10.3069166250`. No row satisfied the complete
  preregistered launch-floor interpretation (`0/20`).
- Slope comparison: 14 rows were identifiable; slope ratios ranged from
  `0.2916538373` to `253.3449826406`, with median `0.9988835442`. The large
  extremes arise where an eager slope is nearly zero, so they do not support
  a launch-only interpretation.
- Knee shifts: among 14 identifiable comparisons, nine were zero, two were
  `-6144` tokens, and three were `-61440` tokens. No Graph knee disappeared.
- Host-minus-device proxy: among 105 stable A/B points, the Graph proxy was
  lower in only 23. Median eager and Graph proxy values were
  `-0.0009800498` and `0.0002675182` ms, respectively. These are timing
  proxies, not direct CPU launch-gap measurements.
- Durable evidence: local root
  `22a613b07c1ee6d3e9a0a7fc81df6065ccc8a10bf783b1a69aded3c2eb8068f0`
  contains 18,628 objects. It was conditionally published with `COMPLETE`
  last to
  `r2://kvbench-artifacts/kvbench/sha256/22a613b07c1ee6d3e9a0a7fc81df6065ccc8a10bf783b1a69aded3c2eb8068f0/`
  and passed one new-empty-directory clean retrieval with exact Bucket Lock.
- Phase 15 readiness: `NOT_READY`. Full Scan remains `CLOSED`, quality remains
  `LOCKED`, and `PERFORMANCE_DATA_FROZEN` remains absent.
- Next action: preregister a separate Phase 14 stability investigation and, if
  authorized, a wholly fresh complete A/B campaign. Do not selectively rerun
  only the five unstable points and do not begin Phase 15.

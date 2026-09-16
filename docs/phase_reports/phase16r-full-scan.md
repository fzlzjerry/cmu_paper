# Phase 16R Full Scan Report

Status: PASS. Phase 17 READY, not started. Quality LOCKED;
`PERFORMANCE_DATA_FROZEN` absent. No quality-preserving or final model claim.

Starting HEAD: `08a19c0f73bdd83ca1c2a939a77323710f9deba9`.
Timing execution HEAD: `ec534d9958d5616eef981c628b87a92c7c809872`.
Controller recovery HEAD: `ad78e30b97cedb505bf3998abaf106d621e418b1`.
Host-wall analysis HEAD: `1fb85eda2cb67f438ea1b0d65d7ca0dfad5c600f`.
Final HEAD is the closure commit containing this report (no self-referential hash).

Family: `phase16-20260831t123029614620z-ec534d99-de80ac`.
Measurement Container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.

## Execution and QC

450 base + 84 unchanged adaptive points = 534 logical points, with five
independent processes each: 2670 terminal records. Recomputed feasibility is
441 feasible / 93 capacity-infeasible logical points. Effective records are
2205 completed / 465 capacity-infeasible / 0 unresolved failed.
38 explicitly linked infrastructure replacements are preserved (35/1/0/2/0
by replicate), including one originally incomplete attempt; no valid, slow,
or unstable timing record was selectively rerun.

The primary host-wall analysis has 441 stable / 0 unstable / 0 failed logical
points; maximum sample-standard-deviation CV is **1.305276915%**, below 3%.
Output, kernel-path, and allocation mismatches: 0 each. All 50 config/batch
groups have sufficient stable coverage. Counts below are stable logical points:

| Configuration(s), each | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---:|---:|---:|---:|---:|
| bf16 | 12 | 9 | 10 | 8 | 3 |
| tq_4bit_nc, tq_k3v4_nc, tq_3bit_nc | 12 | 9 | 11 | 9 | 4 |
| k4v4, k2v4, k2v2 | 15 | 9 | 11 | 9 | 4 |
| kvq4, kvq3 | 9 | 9 | 8 | 9 | 4 |
| kvq2 | 9 | 9 | 11 | 9 | 4 |

357 eligible host-wall same-work ratios remain performance-only and quality
unvalidated. 27 capacity-amplification rows are separate. Timing `r_hbm` is
null; ten Phase 15 features retain `phase15_common_point_only` scope.

## Immutable analysis correction

The first outer bundle contains secondary CUDA-event summaries (maximum CV
1.308072913%). Its ratios must not be described as host-wall ratios.
An append-only host-wall closure derives exact medians from all 2205
checksum-bound raw result files, revalidates all 534 summaries, and supplies
the primary summaries, ratios, and diagnostic plots. Neither the sealed
outer bundle nor any source sample was edited. No timing was rerun.

## Durability

All roots below use `r2://kvbench-artifacts/kvbench/sha256/<root>/`.
Every bundle was uploaded COMPLETE-last and passed one clean retrieval.
Bucket Lock: `kvbench-evidence-indefinite`, exact `kvbench/sha256/` prefix,
private bucket, indefinite retention.

| Bundle | Objects | Root |
|---|---:|---|
| Replicate 0 | 15089 | `7f683da8c427ffa78c3617a102dd47bf6cc533da6cc10c35f8dee50aa7b68f4d` |
| Replicate 1 | 14789 | `a9e99b51a6d35c14733266bf34718fdb936ec144d4e4598f9c830b4d8213a355` |
| Replicate 2 | 14754 | `c859c77ebe820e743f23d53400fb36768bf8cc3953244285ecd244459d3511fd` |
| Replicate 3 | 14824 | `56152a5a3151170c1e3919bb24af35adafcd39aec5cf91b610f3c5445364b168` |
| Replicate 4 | 14754 | `78695bf14b4262debbc145a379a5d7f3c5936d1c6e8a295c55b125d8effaa09d` |
| Original outer (CUDA-event analysis) | 26 | `d74587675dd59b464d81c6e82885d3c9706c681a9da216ad1a7fe4c6ccd88daa` |
| Primary host-wall closure | 15 | `5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e` |

Host-wall URI:
`r2://kvbench-artifacts/kvbench/sha256/5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e/`.
Verification completed 2026-09-16T13:17:35.238202Z. The small outer/closure
reference segment roots rather than duplicating raw segment objects.
Receipt index: `docs/evidence/phase16r/full-scan-publication.json`.

## Prefix, preservation, and validation

59 compact logical prefixes total 54,425,655 bytes. No materialized snapshot
was created for this family; per-process reconstruction stayed outside timing.
Legacy mismatched layouts remain readable but restore-incompatible. Historical
prefix catalogs remain local and unchanged. All five large raw segment
stagings were evicted only after remote publication and clean verification;
local manifests, indices, summaries, and receipts remain.

Stopped family `phase16-20260830t172802896531z-08a19c0f-0c0e2f` stays
non-claim-bearing, with no timing reuse. Its STOPPED checksum remains
`9f39dfa09211cf3a0270a9ba1995199712a0bc530e568c6216bd7c6f8e1eeca3`.
Adapters, CUDA, configurations, calibration, fixtures, admission, and timing
semantics are unchanged. NVIDIA recovery stayed on kernel 7.0.0-31 without
reboot; no Phase 17, profiling, or quality execution occurred.

Validation: 27 focused Full Scan/statistics tests PASS; `make validate-full-scan`
PASS; exact host-wall rederivation PASS (2205 results, 2670 records, 534 points).
Initial test-container mount and test-global-state errors were corrected only
in test setup; no experiment was repeated. Final closure is committed with a
clean working tree. Remaining blocker: none. Next: propose Phase 17 separately.

# Research task breakdown

Status values in this research ledger are planning states, not admission
results. No remote issue tracker is configured, so this file is the
authoritative local task index until issues are created elsewhere.

| ID | Scope | Depends on | Required evidence / gate | Status |
|---|---|---|---|---|
| E00 | Hardware preflight | Phase 0 PASS | hardware manifest; native extension; PTX/JIT; Compute Sanitizer; G0 | complete: native-host G0 PASS in `e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32` |
| E01 | Repository scaffold and schemas | G0 | strict schemas; append-only writer; durable artifact policy; digest-pinned container; parity preflight | partial: Phase 2 local scaffold/contracts/writer PASS; durable storage and container-parity items remain open as B-009/B-010 |
| E02 | BF16 static-cache baseline | E01 and container-parity G0 for formal closure; Decision 0007 for Phase 3 engineering scope | reference numerical match; static allocation; GQA audit | Phase 3 remains engineering G1 FAIL. Latest complete campaigns preserved 5 completed and 15 pre-measurement aborts; B-013/B-014 are resolved, while B-011/B-012/B-015 and formal B-009/B-010 remain open |
| E03 | Fixed-L benchmark | E02; Decision 0007 permits only bounded Phase 3 admission runner | fixed-L and growing-context runners; timing-boundary tests | Original and first-remediation evidence remain immutable. Latest fixed-L and growing campaigns attempted all 20 frozen points once with new IDs, preserving 5 completed and 15 aborted runs; no selective rerun occurred and ordinary timing remains closed |
| E04 | CUDA Graph harness | E02-E03 | capture/replay correctness; no replay allocation; eager/graph lanes | Original 8/8 bounded graph controls remain immutable; three latest completed graph operations also passed strict zero allocation and numerical agreement, but five graph points aborted before measurement and overall G1 remains FAIL |
| E05 | TurboQuant reference lane | E00-E04 | authoritative pinned source; isolated container; golden fixtures | pending |
| E06 | TurboQuant measurement adapter | E05 | numerical, byte, graph, path, sanitizer, smoke evidence; G2-TQ | pending |
| E07 | KIVI reference lane | G2-TQ | pinned legacy container; rollover and K/V asymmetry fixtures | pending |
| E08 | KIVI measurement adapter | E07 | static buffers; r_alloc(L); GQA indexing; G2-KIVI | pending |
| E09 | KVQuant calibration | G2-KIVI | frozen dataset/revision/seed/cap/artifacts/checksums | pending |
| E10 | KVQuant reference lane | E09 | dense/sparse/sink fixtures for 4/3/2-bit and cap cases | pending |
| E11 | KVQuant measurement adapter | E10 | fixed sparse buffers; byte breakdown; graph/path tests; G2-KVQ | pending |
| E12 | Admission gates | E02-E11 | machine-readable G1-G5 report for every main configuration | Original report remains immutable FAIL. Latest full-campaign evidence produced independently valid immutable FAIL report `phase3-g1-20260723t060636246041z-3f2c365a-26bf3c`; unified E12 and G2-G5 remain pending |
| E13 | Pilot scan | E12 PASS | immutable randomized samples; QC; provisional knees; pilot gate | pending |
| E14 | Nsight Systems integration | E13 | nsys-only runs around knees; launch/sync/kernel evidence | pending |
| E15 | Nsight Compute integration | E13 | current-SM metric discovery; measured traffic; ncu-only runs | pending |
| E16 | Full scan | pilot gate, M14-GRAPH-AB, E14-E15 | preregistered grid; feasibility/exclusion records; immutable samples | pending |
| E17 | Knee and response-surface fitting | E16 | candidate models; strict holdouts; session bootstrap CIs | pending |
| E18 | Reproducibility package | E17 and all gates | pinned containers; reproduction commands; figures; final report | pending |

## Required post-pilot milestone

| ID | Scope | Owner and schedule | Required evidence | Status |
|---|---|---|---|---|
| M14-GRAPH-AB | Phase 14 CUDA Graph A/B mechanism experiment | E04 harness owner; execute after E13 pilot admission and before E16 | same method/cache/backend/shape with only Graph mode changed; output/cache identity; floor, slope, knee, launch-gap, and backend evidence | pending |

M14-GRAPH-AB is a named milestone, not a renumbering of the contract's E00-E18
task list. E16 remains closed until its evidence is reviewed.

## Cross-cutting subtasks

- Phase 2 implements the local portion of docs/artifact_policy.md; no
  claim-bearing run starts while B-009's durable-store/locator requirements
  remain open.
- E01 task closure still requires a pinned measurement-container digest and
  identical E00 execution inside it. Decision 0007 narrowly permits bounded
  BF16 Phase 3 implementation and non-claim native-host admission evidence;
  B-010 keeps formal E02 closure, ordinary timing, later methods, and claims
  closed until parity passes.
- E03 includes growing-context request validation but does not mix those
  samples with fixed-L fitting.
- E04 owns the capture/replay harness used by M14-GRAPH-AB; that milestone runs
  only after pilot admission and keeps method/cache/backend/shape fixed.
- E12 includes an operator-level MHA control with identical head dimension and
  no GQA repetition.
- B-013 and B-014 are resolved. Phase 3 failures B-011, B-012, and B-015
  require passing controls, a new clean Git SHA, and entirely new complete
  bounded campaigns. Existing completed/failed/aborted runs remain immutable;
  selective reruns are prohibited.
- E13 and E16 use blocked randomization and retain every failed, unstable, and
  capacity-infeasible point with a machine-readable reason.
- E14/E15 must set run_kind to nsys or ncu; only run_kind=timing enters latency
  analysis.
- Every issue that changes experimental semantics must link a decision record,
  and the semantics change must not share a PR with method implementation.

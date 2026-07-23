# Research task breakdown

Status values in this research ledger are planning states, not admission
results. No remote issue tracker is configured, so this file is the
authoritative local task index until issues are created elsewhere.

| ID | Scope | Depends on | Required evidence / gate | Status |
|---|---|---|---|---|
| E00 | Hardware preflight | Phase 0 PASS | hardware manifest; native extension; PTX/JIT; Compute Sanitizer; G0 | complete: native-host G0 PASS in `e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32` |
| E01 | Repository scaffold and schemas | G0 | strict schemas; append-only writer; durable artifact policy; digest-pinned container; parity preflight | partial: Phase 2 local scaffold/contracts/writer PASS; durable storage and container-parity items remain open as B-009/B-010 |
| E02 | BF16 static-cache baseline | E01 and container-parity G0 for formal closure; Decision 0007 for Phase 3 engineering scope | reference numerical match; static allocation; GQA audit | Native-host engineering G1 PASS in independently validated immutable report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb`. All 20 criteria pass from the unchanged complete campaigns and raw audit bundles. Formal E02 closure remains blocked by B-009/B-010. |
| E03 | Fixed-L benchmark | E02; Decision 0007 permits only bounded Phase 3 admission runner | fixed-L and growing-context runners; timing-boundary tests | Execution SHA `9def265ab613cde7a06b0e51850f066d0564d635` completed all 16 fixed-L and all 4 growing-context runs with new IDs, no abort/failure, no unattempted point, and no selective rerun. The results remain non-claim admission evidence. |
| E04 | CUDA Graph harness | E02-E03 | capture/replay correctness; no replay allocation; eager/graph lanes | Native-host BF16 harness admitted: Decision 0015 preserves asynchronous launch correlation and all 8 graph operations independently replay strict zero allocation. The new G1 report passes capture/replay, numerical agreement, and graph-allocation criteria. Later methods require their own G2 evidence and remain unstarted. |
| E05 | TurboQuant reference lane | E00-E04 | authoritative pinned source; isolated container; golden fixtures | pending |
| E06 | TurboQuant measurement adapter | E05 | numerical, byte, graph, path, sanitizer, smoke evidence; G2-TQ | pending |
| E07 | KIVI reference lane | G2-TQ | pinned legacy container; rollover and K/V asymmetry fixtures | pending |
| E08 | KIVI measurement adapter | E07 | static buffers; r_alloc(L); GQA indexing; G2-KIVI | pending |
| E09 | KVQuant calibration | G2-KIVI | frozen dataset/revision/seed/cap/artifacts/checksums | pending |
| E10 | KVQuant reference lane | E09 | dense/sparse/sink fixtures for 4/3/2-bit and cap cases | pending |
| E11 | KVQuant measurement adapter | E10 | fixed sparse buffers; byte breakdown; graph/path tests; G2-KVQ | pending |
| E12 | Admission gates | E02-E11 | machine-readable G1-G5 report for every main configuration | BF16 native-host G1 PASS in new no-replace report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb`, SHA-256 `c29aef1d9f22b328201599b3e6cdf9efe7c069e78abaf6b37bc3cb12931414c9`. G2-G5 remain NOT EVALUATED; E12 is not complete for later methods or formal/unified admission. |
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
- B-011 through B-017 are resolved for native-host BF16 G1. The reporting-only
  repair reused the same immutable campaigns and did not rerun any point.
  Preserve every failed and passing run/report.
- E13 and E16 use blocked randomization and retain every failed, unstable, and
  capacity-infeasible point with a machine-readable reason.
- E14/E15 must set run_kind to nsys or ncu; only run_kind=timing enters latency
  analysis.
- Every issue that changes experimental semantics must link a decision record,
  and the semantics change must not share a PR with method implementation.

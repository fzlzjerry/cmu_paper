# Research task breakdown

Status values in this research ledger are planning states, not admission
results. No remote issue tracker is configured, so this file is the
authoritative local task index until issues are created elsewhere.

| ID | Scope | Depends on | Required evidence / gate | Status |
|---|---|---|---|---|
| E00 | Hardware preflight | Phase 0 PASS | hardware manifest; native extension; PTX/JIT; Compute Sanitizer; G0 | complete: native-host G0 PASS in `e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32` |
| E01 | Repository scaffold and schemas | G0 | strict schemas; append-only writer; durable artifact policy; digest-pinned container; parity preflight | complete: local contracts/writer PASS; exact Decision 0016 image built/scanned; container G0 plus BF16 eager/graph parity PASS; private indefinite-locked R2 state verified; synthetic and container-G0 roots cleanly retrieved. B-009/B-010 RESOLVED. |
| E02 | BF16 static-cache baseline | E01 and container-parity G0 for formal closure; Decision 0007 for Phase 3 engineering scope | reference numerical match; static allocation; GQA audit | Native-host engineering G1 PASS in independently validated immutable report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb`. All 20 criteria pass from unchanged campaigns/raw audits. Phase 6A eager/graph runs establish container parity only and create no new unified or claim-bearing G1 result. |
| E03 | Fixed-L benchmark | E02; Decision 0007 permits only bounded Phase 3 admission runner | fixed-L and growing-context runners; timing-boundary tests | Execution SHA `9def265ab613cde7a06b0e51850f066d0564d635` completed all 16 fixed-L and all 4 growing-context runs with new IDs, no abort/failure, no unattempted point, and no selective rerun. The results remain non-claim admission evidence. |
| E04 | Common cache-method and CUDA Graph harness | E02-E03 | stable method protocol; correctness/allocation/path facades; capture/replay correctness; no replay allocation | complete for Phase 4 BF16 at `0cf160caa532c7cac23275c8a14fd8694789a86f`; `docs/evidence/phase4/method-admission.json` validates. TurboQuant reference work is complete, but later Measurement Lane adapters still require their own G2 evidence. |
| E05 | TurboQuant reference lane | E00-E04 | authoritative pinned source; isolated container; golden fixtures | complete: official vLLM v0.25.1 commit `752a3a504485790a2e8491cacbb35c137339ad34`; isolated SM120 environment; 3 mandatory MSE+NC and 1 held-out deterministic fixture; exact regeneration and validation PASS |
| E06 | TurboQuant measurement adapter | E05 | numerical, byte, graph, path, sanitizer, smoke evidence; G2-TQ | complete: execution HEAD `0df5bb4d445d48e6cba17e30723733f8de35cb14`; current-HEAD sanitizer 3/3 PASS; frozen bounded grid 9/9 PASS; 167-object root `f003bc3dc5de6b67a6d8f1b8bed7fa49b7f90f9d7edc4d1383e2d97c8aa19d6d` COMPLETE-last and cleanly retrieved; G2-TQ PASS |
| E07 | KIVI reference lane | G2-TQ | official source authority; isolated environment; rollover and K/V asymmetry fixtures; native eight-head GQA storage | BLOCKED at source audit by B-019: official `main` commit `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6` calls `repeat_kv` for recent K/V and physically materializes H_Q=32 storage from H_KV=8; no environment, CUDA build, fixture, trace, or R2 publication was started |
| E08 | KIVI measurement adapter | E07 | static buffers; r_alloc(L); GQA indexing; G2-KIVI | blocked by E07/B-019; Phase 8 remains unstarted |
| E09 | KVQuant calibration | G2-KIVI | frozen dataset/revision/seed/cap/artifacts/checksums | pending |
| E10 | KVQuant reference lane | E09 | dense/sparse/sink fixtures for 4/3/2-bit and cap cases | pending |
| E11 | KVQuant measurement adapter | E10 | fixed sparse buffers; byte breakdown; graph/path tests; G2-KVQ | pending |
| E12 | Admission gates | E02-E11 | machine-readable G1-G5 report for every main configuration | BF16 native-host G1 PASS in new no-replace report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb`, SHA-256 `c29aef1d9f22b328201599b3e6cdf9efe7c069e78abaf6b37bc3cb12931414c9`. Method-specific G2-TQ PASS; global G2-G5 remain NOT EVALUATED; E12 is not complete for later methods or formal/unified admission. |
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

- Phase 2 implements the local portion of docs/artifact_policy.md. Phase 6A
  resolves B-009 with verified private, indefinite-locked Cloudflare R2 plus
  clean synthetic and container-G0 retrieval. Every future publication still
  repeats the management and clean-retrieval checks.
- E01 is complete for exact Decision 0016 Docker image ID / OCI image-index
  digest
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
  Tags are non-authoritative; a changed digest requires full recertification.
  Native-host admission remains non-claim-bearing.
- E03 includes growing-context request validation but does not mix those
  samples with fixed-L fitting.
- E04 owns the capture/replay harness used by M14-GRAPH-AB; that milestone runs
  only after pilot admission and keeps method/cache/backend/shape fixed.
- Phase 4 adds only the BF16 adapter/factory and reusable admission facades.
  Phase 5 adds only the TurboQuant reference lane. Phase 6A remediation is
  PASS. Phase 6 has a minimal implementation; current-HEAD sanitizer passed
  3/3, the frozen bounded grid passed 9/9, durable publication and clean
  retrieval passed, and G2-TQ is PASS. Phase 7/E07 stopped BLOCKED at its
  official-source GQA audit under B-019 before any environment, CUDA, fixture,
  trace, or publication work. E08 and later tasks, KIVI adapter work, KVQuant,
  Pilot, performance profiler, Full Scan, and quality execution remain unopened
  and unauthorized.
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

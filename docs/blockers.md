# Blockers

Last updated: 2026-07-22.

## Phase 0 disposition

No unresolved item prevents the Phase 0 audit from meeting its stated
acceptance criteria. The items below block later tasks and must not be bypassed.
None has been resolved by a silent fallback.

| ID | Blocking condition | Blocks | Evidence / next action | Status |
|---|---|---|---|---|
| B-001 | No Git commit or code SHA existed after Phase 0. | E00 durable gate evidence and every later run | Resolved by reviewed root commit 9569d938d9023a3e71d98f12234efa1897004533. | resolved 2026-07-22 |
| B-002 | Current hardware and CUDA toolchain are uncertified. | E01 and all CUDA work | Run E00 and require G0 native, forced-PTX, sanitizer, SKU, clock, and process-isolation evidence. | expected next-phase work |
| B-003 | TurboQuant paper has no identified author-owned code repository; vLLM v0.25.1 is only a pinned candidate. | E05 golden fixtures | Establish reference authority/equivalence and record a decision if upstream semantics differ. | open |
| B-004 | Primary model ID, immutable revision, config hash, geometry, and context limit are unset. | E02 and method fixtures | Select only after G0, verify all method geometry, and record a model-selection decision. | open |
| B-005 | KVQuant calibration dataset/revision, preprocessing, seed, cap, and artifact do not exist. | E09-E11 | Freeze and checksum them in Phase 9; no calibration during scans. | open |
| B-006 | KVQuant root licensing and embedded/adapted-source lineage are incomplete. | E10 reference execution, copying, or redistribution | Map recorded embedded trees and attributed files to exact upstream commits/patch deltas; resolve repository-wide license authority. | open |
| B-007 | Archive acquisition URL and pre-workspace provenance were not supplied. | Exact external reacquisition of literature bundle | Ask the operator for origin metadata if available; retain the local archive/file hashes meanwhile. | open, non-gating for local audit |
| B-008 | qpdf is not installed for an additional PDF structural scan. | Optional defense-in-depth literature check | Install only in a reviewed environment or use an equivalent static scanner; current pdfinfo/pdfdetach checks are recorded. | open, non-gating |
| B-009 | `/artifacts/` is Git-ignored and no durable append-only store or immutable locator scheme is selected or implemented. | E01 acceptance, G1-G5 admission, and every claim-bearing run | Implement and test docs/artifact_policy.md; record the durable store and locator/digest publication mechanism before closing E01. Decision 0002 defines the narrower Git-tracked E00 evidence path. | open |
| B-010 | No digest-pinned measurement container exists and host G0 cannot certify one. | E02, method CUDA implementation, G1-G5, and all timing | After E01, pin the measurement image by digest and rerun the identical preflight inside it before E02. | open |

PHASE 7 REPORT

Status: BLOCKED

Entry:
- Starting HEAD: 0974bbc98f8f941b09800786591108292dc4e0dd
- Final HEAD: commit containing this report; exact SHA recorded in task handoff
- Working tree at entry: clean
- Final working tree: expected clean after the report commit
- Phase 6 report: PASS
- Phase 6 admission root: outer 8c4cf76f76bb17e648dfd911f11e268235ed827a9983814b774b9e95405496b0; original f003bc3dc5de6b67a6d8f1b8bed7fa49b7f90f9d7edc4d1383e2d97c8aa19d6d unchanged
- Complete Phase 6 bundle verified: yes; 176-object outer and 167-object original roots both cleanly retrieved
- G0: PASS
- G1: PASS
- G2-TQ: PASS
- Quality execution: LOCKED
- Full Scan: CLOSED

Minimal scope:
- Plan: docs/plans/phase7-kivi-reference.md
- General reference framework added: no
- Measurement Container modified: no
- TurboQuant adapter modified: no
- Phase 8 work started: no

Pinned KIVI source:
- Repository: https://github.com/jy-yuan/KIVI.git
- Exact commit: 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
- Exact tree: c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b
- Commit date: 2025-11-20T12:34:32-08:00
- License: MIT
- Relevant files: 15 files bound by Git blob and SHA-256 in third_party/LOCK.json
- Source lock: third_party/LOCK.json; SHA-256 3ba75f303a1e0320ec3adabc00469d650ee5077a3d41065b90498322e6b174a4
- Floating branch: no
- Unofficial fork used: no

Reference environment:
- Definition: NOT CREATED; hard source gate failed first
- Base image digest: NOT SELECTED
- Dockerfile SHA-256: NOT CREATED
- Image ID/config digest: NOT BUILT
- Python: NOT FROZEN
- PyTorch: NOT FROZEN
- CUDA: NOT FROZEN
- Compiler: NOT FROZEN
- Extension build command: NOT RUN; upstream path would be quant/setup.py
- Architecture flags: NOT SET
- Extension SHA-256: NOT BUILT
- Measurement Container modified: no
- Credentials/model weights in image: no image created

CUDA compatibility:
- Native SM120: NOT EXECUTED
- sm_120 cubin: NOT BUILT
- compute_120 PTX: NOT BUILT
- Forced PTX/JIT: NOT EXECUTED
- Compute Sanitizer: NOT EXECUTED
- Unsupported fallback: NOT EVALUATED
- no kernel image: NOT EVALUATED

Verified algorithm:
- Key quantization direction: source-verified per-channel, grouped along tokens
- Value quantization direction: source-verified per-token, grouped along head dimension
- Group semantics: last-dimension min/max groups; key tensor is transposed so the grouped dimension is token position
- Metadata precision: source allocates scale/minimum in the input tensor dtype; runtime identity not verified
- Packing: source-verified int32 bit packing; runtime bytes not generated
- Residual policy: recent K/V remain full precision; historical K/V are quantized
- Rollover trigger: keys move a full 32-token block at active length 32; values move the oldest token when active length becomes 33
- GQA support: advertised but unacceptable; recent K/V are repeated from 8 to 32 heads
- Full-prefix dequantization: source historical path uses quantized GEMV; runtime behavior NOT EXECUTED

Configurations:
- K4/V4: source-supported; fixture NOT GENERATED
- K2/V4: source-supported; fixture NOT GENERATED
- K2/V2: source-supported; fixture NOT GENERATED
- K4/V2 held-out: source-supported; fixture NOT GENERATED
- group_size: 32 frozen in plan; runtime NOT EXECUTED
- residual_length: 32 frozen in plan; runtime NOT EXECUTED
- Unsupported behavior: CUDA quantized GEMV accepts only 2-bit and 4-bit; runtime rejection tests NOT EXECUTED

Fixture geometry:
- Batch: 1 planned
- Query heads: 32
- KV heads: 8
- Head dimension: 128
- Basic context: 17 store + 1 append planned
- Rollover contexts: 31, 32, 33 selected from source
- Seed: 20260726 planned
- Dtype: BF16 planned

Fixtures:
- Root: NOT CREATED
- Mandatory count: 0
- Held-out count: 0
- Store: NOT EXECUTED
- Append: NOT EXECUTED
- Decode: NOT EXECUTED
- Quantized K: NOT GENERATED
- Quantized V: NOT GENERATED
- Metadata: NOT GENERATED
- Residual K/V: NOT GENERATED
- Rollover records: NOT GENERATED
- Checksums: source-audit JSON and source lock only; no fixture ledger
- Existing-fixture overwrite prevention: NOT IMPLEMENTED because no fixture generator was authorized

Byte accounting:
- Configuration/context: NOT EXECUTED
- Quantized K bytes: NOT MEASURED
- Quantized V bytes: NOT MEASURED
- Metadata bytes: NOT MEASURED
- Residual K bytes: NOT MEASURED
- Residual V bytes: NOT MEASURED
- Padding: NOT MEASURED
- Workspace: NOT MEASURED
- Actual total: NOT MEASURED
- Logical BF16 bytes: NOT CALCULATED
- r_alloc: NOT POPULATED
- Storage agreement: NOT EVALUATED
- r_hbm: NOT POPULATED

Residual rollover:
- Before: source audit identifies L=31
- Boundary: source audit identifies key block transition at L=32
- After: source audit identifies L=33 and value oldest-token transition
- Tokens moved: source semantics only; no fixture
- Missing tokens: NOT EVALUATED
- Duplicate tokens: NOT EVALUATED
- Reallocation observed: source uses torch.cat/contiguous; runtime NOT EXECUTED
- Source-faithful: source audit only; no fixture verdict

GQA:
- H_Q: 32
- H_KV: 8
- Storage geometry: persistent source cache begins at 8 KV heads, but recent K/V are expanded to 32-head temporaries
- Source audit: FAIL; models/llama_kivi.py calls Transformers repeat_kv for recent K and V
- Runtime audit: non-timing CPU storage-semantics proof for the exact helper; no KIVI CUDA execution
- repeat_kv: present and called
- repeat_interleave: helper documents equivalent behavior
- Expanded temporary: confirmed; BF16 storage 65,536 bytes to 262,144 bytes, distinct and contiguous
- Head mapping: quantized-history CUDA source maps query heads to KV heads by nh/nh_kv ratio, but residual expansion violates the gate
- Final verdict: BLOCKED_GQA_MATERIALIZATION

Reference trace:
- Mechanism: NOT COLLECTED
- Quantize/store kernels: NOT EXECUTED
- Append kernels: NOT EXECUTED
- Decode/dequant kernels: NOT EXECUTED
- Rollover operations: NOT EXECUTED
- Full-prefix temporary: NOT EVALUATED at runtime
- Backend fallback: NOT EVALUATED
- Timing claim: none

Graph information:
- Upstream support: no reference Graph guarantee found
- Reference graph smoke: NOT EXECUTED
- Deferred to Phase 8: no; Phase 7 must first be restarted and pass GQA

Durable publication:
- Local root digest: NOT CREATED
- R2 URI: NOT CREATED
- Initial publication: NOT RUN
- COMPLETE-last: NOT APPLICABLE
- Clean retrieval: Phase 6 entry roots PASS; no Phase 7 fixture root exists
- Checksum result: Phase 6 entry roots PASS; Phase 7 source audit is repository evidence only
- Bucket Lock: PASS at exact kvbench/sha256/ prefix during entry
- Credential leakage: none; required variables were reported only as PRESENT

Commands:
- git status --short
- git rev-parse HEAD
- make package-lock-check
- make test
- make checks
- make validate-reference-turboquant
- make validate-admission-turboquant
- make validate-phase6-r2-outer-bundle PHASE6_R2_OUTER_ARTIFACT=artifacts/phase6_r2_outer/phase6-r2-outer-20260726t083456863336315z-498dd83
- make verify-artifact-r2 ROOT_SHA256=8c4cf76f76bb17e648dfd911f11e268235ed827a9983814b774b9e95405496b0
- make verify-artifact-r2 ROOT_SHA256=f003bc3dc5de6b67a6d8f1b8bed7fa49b7f90f9d7edc4d1383e2d97c8aa19d6d
- git ls-remote --heads https://github.com/jy-yuan/KIVI.git
- non-timing BF16 repeat_kv storage-semantics audit

Tests:
- package-lock: PASS at entry and final rerun
- make test: PASS at entry and final rerun; Phase 7 focused tests 10/10 PASS
- make checks: PASS at entry and final rerun
- TurboQuant admission regression: PASS at entry and final rerun; 9/9 unchanged
- source-lock: PASS; exact commit/tree, 15 Git blobs, and 15 SHA-256 values verified
- build: NOT RUN
- fixture: NOT RUN
- rollover: source audit only
- byte-layout: NOT RUN
- GQA: BLOCKED as designed by positive materialization evidence
- sanitizer: NOT RUN
- governance: PASS; exact-path scope, quality lock, Full Scan closure, and fail-closed factory verified
- historical evidence: unchanged from Phase 7 entry; immutable validation PASS

Admission gates:
- G0: PASS
- G1: PASS
- G2-TQ: PASS
- G2-KIVI: NOT EVALUATED / BLOCKED
- Global G2: NOT EVALUATED
- G3: NOT EVALUATED
- G4: NOT EVALUATED
- G5: NOT EVALUATED
- Full Scan: CLOSED

Quality governance:
- Quality execution: LOCKED
- Quality benchmark: NOT RUN
- PERFORMANCE_DATA_FROZEN: absent

Preservation:
- Authorized Measurement Container changed: no
- TurboQuant fixtures changed: no
- TurboQuant adapter changed: no
- Historical evidence changed: no
- Existing run overwritten: no
- Formal performance data: none
- Profiler timing: none
- Quality data: none

Changed files:
- Makefile
- docs/blockers.md
- docs/decisions/0017-kivi-source-authority-and-gqa-materialization.md
- docs/evidence/phase7/kivi-source-audit.json
- docs/method_notes/kivi.md
- docs/phase_reports/phase7-kivi-reference-blocked.md
- docs/plans/phase7-kivi-reference.md
- docs/risk_register.md
- docs/status.md
- docs/tasks.md
- scripts/validate_phase2.py
- tests/unit/test_phase7_kivi_source_audit.py
- third_party/LOCK.json
- third_party/NOTICE.md

Commits:
- one Phase 7 BLOCKED source-audit commit; exact SHA recorded in task handoff

Risks:
- R-003 realized for the selected official KIVI GQA residual path
- R-021 resolved as a source-authority decision but does not cure GQA materialization

Blockers:
- B-019 OPEN: the selected official KIVI primary-model GQA path materializes recent 8-head K/V as 32-head temporaries

Scientific interpretation:
- The official KIVI repository, exact source commit/tree, and relevant source bytes are reproducibly pinned.
- The build, CUDA compatibility, fixtures, rollover runtime, byte layout, and acceptable GQA reference path are not established.
- The selected official GQA reference path is reproducibly noncompliant because it materializes H_Q-sized recent K/V tensors.
- No speedup, HBM, knee, capacity, or quality claim is made.

Next action:
- Obtain an exact author-maintained KIVI revision whose official primary-model GQA path preserves native eight-head K/V storage without H_Q-sized K/V temporaries, then start a separate new Phase 7 task from the full entry check. Do not begin Phase 8.

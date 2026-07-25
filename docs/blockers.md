# Blockers

Last updated: 2026-07-25.

## Current disposition

Phase 2 remains PASS, native-host Phase 3 BF16 G1 remains PASS, the Phase 4
common BF16 adapter is PASS, and the Phase 5 TurboQuant Reference Lane is PASS.
The immutable Phase 3 report
`phase3-g1-20260723t132609515797z-7f72c95f-f31ccb`, SHA-256
`c29aef1d9f22b328201599b3e6cdf9efe7c069e78abaf6b37bc3cb12931414c9`.
The report independently replays all 80 consolidated raw B-011/B-012 operation
bundles from the unchanged 20-run campaigns. Every earlier failed report,
campaign, and run remains immutable. B-011 through B-017 are resolved without
a silent fallback or weakened scientific gate. Phase 5 resolves B-003 only
for the explicitly bounded vLLM reference authority. It does not close or
weaken B-009 or B-010. Phase 6A remediation separately resolves both: the exact
Decision 0016 image passed container G0 and both BF16 parity smokes, and the
private indefinite-locked R2 path passed synthetic plus container-G0 clean
retrieval. The TurboQuant Measurement Adapter is implemented. Its original
final run
`phase6-20260725t065153714z-ace9261a-083f14-4bit_nc-fixed-l128-eager`
remains immutable failed evidence. The narrow B-018 remediation produced one
new sanitizer-only artifact for each mandatory configuration; all three probes
pass with exit code 0, `ERROR SUMMARY: 0 errors`, and
`LEAK SUMMARY: 0 bytes leaked`. B-018 is RESOLVED. The bounded admission grid
was intentionally not attempted, so G2-TQ remains BLOCKED; global G2-G5 remain
NOT EVALUATED.

| ID | Blocking condition | Blocks | Evidence / next action | Status |
|---|---|---|---|---|
| B-001 | No Git commit or code SHA existed after Phase 0. | E00 durable gate evidence and every later run | Resolved by reviewed root commit 9569d938d9023a3e71d98f12234efa1897004533. | resolved 2026-07-22 |
| B-002 | Formal G0 failed because required SASS inspection could not find `nvdisasm`; runtime and sanitizer lanes were not admitted in that run. | E01 and all non-E00 CUDA or timing work | Resolved without changing E00 semantics: the exact `cuda-nvdisasm-13-0=13.0.85-1` package/tool identity is locked at 6442ba1f7554ea0ebf0b3bb1a920c94567cab689, and new immutable run `e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32` passed every G0 lane. The original failed run remains unchanged. | resolved 2026-07-22 |
| B-003 | TurboQuant paper has no identified author-owned code repository; a bounded implementation authority was required for E05. | E05 golden fixtures | Resolved by selecting official vLLM release v0.25.1 at exact commit `752a3a504485790a2e8491cacbb35c137339ad34` as authority for this lane only. Source inspection found the preregistered names unchanged, so no decision record was needed. Fixtures claim conformance only to this pinned vLLM implementation, not every paper variant. | resolved 2026-07-24 |
| B-004 | Primary model ID, immutable revision, config hash, geometry, and context limit were unset. | E02 and method fixtures | Resolved by Decision 0007 and `docs/evidence/phase3/model-identity.md`: all 11 exact-revision artifacts match their frozen SHA-256 values and the full checkpoint/tokenizer load offline with the required BF16 GQA geometry. | resolved 2026-07-22 |
| B-005 | KVQuant calibration dataset/revision, preprocessing, seed, cap, and artifact do not exist. | E09-E11 | Freeze and checksum them in Phase 9; no calibration during scans. | open |
| B-006 | KVQuant root licensing and embedded/adapted-source lineage are incomplete. | E10 reference execution, copying, or redistribution | Map recorded embedded trees and attributed files to exact upstream commits/patch deltas; resolve repository-wide license authority. | open |
| B-007 | Archive acquisition URL and pre-workspace provenance were not supplied. | Exact external reacquisition of literature bundle | Ask the operator for origin metadata if available; retain the local archive/file hashes meanwhile. | open, non-gating for local audit |
| B-008 | qpdf is not installed for an additional PDF structural scan. | Optional defense-in-depth literature check | Install only in a reviewed environment or use an equivalent static scanner; current pdfinfo/pdfdetach checks are recorded. | open, non-gating |
| B-009 | Durable acceptance required verified private Cloudflare R2 management state, an active indefinite lock covering `kvbench/sha256/`, synthetic verification, and container-G0 publication plus clean retrieval. | E01 durable-store closure and any future claim-bearing artifact publication. | Read-only REST verification confirms bucket `kvbench-artifacts`, disabled r2.dev, zero custom domains, and exact enabled indefinite rule `kvbench-evidence-indefinite`. Synthetic root `bbb80210dc729dedc9dd25a24d61cfbedbbe9d05661b1f95e6af278df3d0c11e` cleanly reverified. Container-G0 root `85e1f49dea76d08b2cba4477d089a71759d529f03b2bc3538da3d15d8639455c` published 222 conditional objects with COMPLETE last and cleanly retrieved with every checksum and no unexpected object. | resolved 2026-07-25 |
| B-010 | Measurement Lane execution required one exact built/scanned image, full container G0, both bounded BF16 parity smokes, and digest-bound authority. | Measurement Lane CUDA environment authority; method-specific and global admission remain separate gates. | Docker image ID / OCI image-index digest `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e` passed exact package/tool/secret/model-weight verification. G0 run `e00-20260724T195014.679255Z-a6025ae023e1-23dbe853` and both eager/graph parity runs pass. Decision 0016 authorizes only that digest, disallows floating tags, and requires full recertification after a digest change. | resolved 2026-07-25 |
| B-011 | The immutable original report failed because only high-level SDPA names were visible and found no positive materialization evidence. Both complete campaigns at execution SHA `9def265ab613cde7a06b0e51850f066d0564d635` preserve direct CUDA traces for all 80 operations. | BF16 G1, Phase 4, every later method baseline comparison. | Reporting commit `7f72c95f9932c608f9bd68f1971d6e86378596a2` independently replays the checksum-bound raw traces, source/shape evidence, allocation join, and exact operation keys. New immutable report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb` derives `gqa_nonmaterialization_verified` for all 80 operations, identifies `pytorch_flash::flash_fwd_splitkv` for both controls, and passes `gqa_not_materialized`, `no_torch_cat_growth`, and `no_backend_fallback`. Preserve the corrected taxonomy and direct device-kernel contract. | resolved 2026-07-23 |
| B-012 | The immutable original eager audits had unexplained transient traffic. Decisions 0013/0014 froze the source-backed eager criterion and geometry-specific split control. | BF16 G1 eager/graph lanes, Phase 4 common baseline. | The new report independently replays every allocator event: all 72 eager operations pass `phase3_eager_attributed_ephemeral_v1` with 1,066 fully attributed events each and no failure reason; all 8 graph operations pass `phase3_graph_zero_allocation_v1` with zero events. `no_unexplained_measured_region_allocation` and `graph_replay_no_allocation` both PASS. Retain Decisions 0013/0014 and strict graph-zero-allocation. | resolved 2026-07-23 |
| B-013 | The original fixed-L terminal query race and three first-remediation recurrences remain immutable. The exact live race recurred in eight snapshots in the latest complete campaigns; every row joined to the registered GPU/PID/start time and passed as `owned_only`. Targeted tests keep unregistered foreign processes and PID reuse as hard failures. | Complete BF16 G1 process ownership. | Resolved by commit `eb908f6e372d6b232e6079e9344c2103bc90cdea`; retain the exact registered-row rule and all foreign/PID-reuse controls. | resolved 2026-07-23 |
| B-014 | The first remediation campaign exposed audit/measured checksum mismatch. Untimed exact-endpoint diagnostics showed stable eager and retained-graph outputs; the defect was inconsistent checksum serialization, not a model-output difference. Five completed latest-campaign runs preserved exact audit/measured checksum equality under the unchanged tolerances and measured region. | BF16 G1 fixed-L eager/graph lanes and every later baseline comparison. | Resolved by canonicalizing the audit witness to the frozen tensor checksum in commit `fe28f5e`; retain full-repeat eager/graph regression controls. | resolved 2026-07-23 |
| B-015 | Fifteen second-remediation workers aborted before measurement because raw-audit production did not complete; two exposed the old 1 GiB run bound and thirteen hid their lower producer cause. Untimed worst-case diagnostics preserved `WorkerProtocolError: paired allocator controls did not verify`, proved geometry-specific GQA=11/MHA=5 split counts, measured the 16-step bundle, and supported Decision 0014's 1,152 MiB envelope. The new fixed-L campaign had no size-bound or generic-wrapper failure, and every new producer failure retained its exact decode step and `ChromeTraceValidationError` cause. | Complete B-011/B-012 coverage, BF16 G1, Phase 4, every later baseline comparison. | Resolved by commit `52f41ce9d9be4edc07a833e00fe3404fbfa80b89`; retain the hard-bound/+1-byte, geometry-specific split, and exception-preservation tests. | resolved 2026-07-23 |
| B-016 | Three B-015 fixed-L graph controls (`B1/L16384`, `B4/L4096`, and `B4/L16384`) aborted before measurement with preserved `ChromeTraceValidationError: graph GPU marker is not contained by its host marker`. Decision 0015's untimed `B1/L16384` diagnostic reproduced the exact error: the host contained the unique asynchronous `cudaGraphLaunch`, while the valid MHA GPU range extended 150.023 microseconds past host return. In-memory removal of only that invalid relation recovered the exact two correlated Flash split-K nodes. | Complete B-011/B-012 graph coverage, BF16 G1, the growing campaign, Phase 4, and every later baseline comparison. | Resolved by commit `e7219e0dd714149e3eea783ce7a8602c4bf9bc54`: allow asynchronous GPU completion while retaining and strengthening launch-correlation, stream, External-ID, graph/node, unknown-category, kernel-family, and materialization checks. Deterministic parser tests, the real `B1/L16385` CUDA control, `make checks`, `make test`, `make test-cuda` (14/14), and `make test-graph` (3/3) pass. Retain these controls in both new complete campaigns. | resolved 2026-07-23 |
| B-017 | Immutable report `phase3-g1-20260723t123322160580z-9def265a-08dc69` remains valid FAIL evidence for the former legacy-summary/raw-bundle disconnect. | BF16 G1, B-011/B-012 closure, Phase 4, every later method baseline comparison. | Commit `7f72c95f9932c608f9bd68f1971d6e86378596a2` binds report behavior to its recorded generator SHA, reuses coordinator raw replay, rejects missing/tampered/mismatched evidence, ignores serialized worker `passed` booleans, and derives the five criteria from raw bytes. `make checks`, `make test`, `make test-cuda` (14/14), and `make test-graph` (3/3) pass. New no-replace report `phase3-g1-20260723t132609515797z-7f72c95f-f31ccb` independently validates PASS with no errors; the two campaigns and 20 source runs were not rerun or modified. | resolved 2026-07-23 |

| B-018 | The exact pinned TurboQuant store-append-decode probe completed functionally, but Compute Sanitizer memcheck reported 2,093,260 leaked bytes in 28 allocations and exited 99 in final run `phase6-20260725t065153714z-ace9261a-083f14-4bit_nc-fixed-l128-eager`. | G2-TQ and the bounded admission grid | Resolved without changing the algorithm, fixtures, container, runner, timing, or sanitizer criteria. Commit `aac794c21b01b3e43ff93e317286285d21dbcd47` explicitly releases sanitizer-only CUDA storages and library resources. New runs `phase6-b018-20260725t141636785z-aac794c2-6444b4-4bit_nc-sanitizer`, `phase6-b018-20260725t141640440z-aac794c2-eb5675-k3v4_nc-sanitizer`, and `phase6-b018-20260725t141643545z-aac794c2-4d7433-3bit_nc-sanitizer` each pass the probe, exit 0, and record zero errors and zero leaked bytes with valid COMPLETE/inventory/checksum ledgers. The grid remains separately unevaluated. | resolved 2026-07-25 |

## Phase 4 disposition

The adapter boundary adds no new blocker and closes none. Its report and three
functional smoke records remain local, checksum-bound, non-claim evidence. At
Phase 4 completion, B-009 still required durable append-only publication and
B-010 still required a digest-pinned Measurement Container with parity G0.
The separate Phase 5 reference environment did not satisfy container parity;
Phase 6A remediation later resolved both blockers independently.

## Phase 5 disposition

B-003 is resolved for the narrowly selected vLLM authority. The exact source,
environment, fixtures, and checksum ledgers reproduce on SM120; no source or
configuration substitution occurred. Direct CUDA Graph smoke was not exercised
in the minimal official API and remains Phase 6 work, but is not a Phase 5
blocker under the frozen acceptance criteria. At Phase 5 completion, B-009 and
B-010 remained OPEN; Phase 6A remediation later resolved them.

## Phase 6 entry-blocked disposition

Phase 6 was attempted and BLOCKED at entry because B-009 and B-010 were both
unresolved. At that entry, B-009 was OPEN without durable append-only
publication and clean retrieval verification, and B-010 was OPEN without an
authorized digest-pinned Measurement Container and container G0 parity.
Neither blocker is narrowed or resolved by the retrospective record. No Phase
6 implementation is present. The initial Phase 6A prerequisite attempt did not
alter this retrospective record and left both blockers open.

## Phase 6A remediation disposition

Phase 6A remediation reused the existing implementation and directly resolved
B-009 and B-010. Decision 0016 binds Measurement Lane CUDA execution to exact
Docker image ID / OCI image-index digest
`sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`.
Container G0 and both bounded BF16 parity smokes pass. Cloudflare management
state is private, and exact enabled rule `kvbench-evidence-indefinite` retains
`kvbench/sha256/` indefinitely. The synthetic root and container-G0 root
`85e1f49dea76d08b2cba4477d089a71759d529f03b2bc3538da3d15d8639455c`
both cleanly retrieve and validate.

At Phase 6A completion, blocker closure supplied environment and durable-store
authority only; Phase 6 had not yet been restarted and G2-TQ was
`NOT EVALUATED / READY`. The later Phase 6 disposition below supersedes that
historical entry state.


## Phase 6 measurement-adapter disposition

The minimal adapter and all three frozen fixture audits pass, including exact
packed bytes, byte accounting, execution path, eager allocation, and fixture
Graph replay. The mandatory sanitizer gate fails in finalized run `phase6-20260725t065153714z-ace9261a-083f14-4bit_nc-fixed-l128-eager`.
The stop rule prohibited the remaining sanitizer configurations and the entire
bounded grid. The failed artifact is immutable, independently valid, and
durably published at `r2://kvbench-artifacts/kvbench/sha256/f319c4b05054ee2f31bdcbfe15fa67850ea784c17718192b82e527567c5cf343/`. This durable failure does not admit any
configuration.

The subsequent B-018-only remediation preserved that evidence and used new
run IDs under the same authorized image. All three mandatory sanitizer probes
now pass with zero errors and zero leaked bytes, and each finalized
sanitizer-only bundle independently validates. Their `aborted` lifecycle is
intentional: it records that the authorized B-018-only scope completed without
entering the bounded grid. B-018 is RESOLVED. G2-TQ remains BLOCKED only because
the bounded admission grid is NOT EVALUATED; global G2-G5 remain NOT EVALUATED,
Full Scan remains CLOSED, and quality execution remains LOCKED.

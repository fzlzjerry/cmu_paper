# PHASE 6A REMEDIATION REPORT

Status: PASS

Date: 2026-07-25

## Entry

- Starting HEAD: `4f38b0ed4997ae29dc5c95a42db733ffdf56f903`
- Final HEAD: the commit containing this report; the exact non-self-referential
  SHA is recorded in the operator handoff
- Working tree: clean at entry; required clean state after report commit
- Previous Phase 6A report verified: yes, unchanged SHA-256
  `000da9d64bf362c19fd3486fc07dab12cf4e5f12fee0146de37ee31ce316b99f`
- G0 native host: PASS
- G1 native host: PASS, `native_host_admission` only and non-claim-bearing
- B-009 at entry: OPEN
- B-010 at entry: OPEN
- Quality execution: LOCKED
- Full Scan: CLOSED

## Operator prerequisites

- docker CLI: `/usr/bin/docker`, version 29.6.1
- docker daemon: available, version 29.6.1
- NVIDIA Container Toolkit: `/usr/bin/nvidia-ctk`, version 1.19.1
- NVIDIA Docker runtime: PASS; exact RTX PRO 6000 visible through the requested
  GPU runtime path
- R2 S3 credentials: PRESENT; object access verified
- Cloudflare management token: PRESENT; read-only management verification PASS
- Missing prerequisites: none

`.env` remained untracked, ignored, excluded by `.dockerignore`, and mode
`0600`. No secret value was printed, hashed, uploaded, committed, or passed to
Docker.

## Measurement Container

- Dockerfile: `docker/measurement.Dockerfile`, SHA-256
  `333a1e4264e8dc7798c5af06622fc97871371c3fc1e063f4a3b88cfb25389ace`
- Build: PASS; `make measurement-container`; clean `git archive` context;
  linux/amd64; no build secret or credential argument
- Base digest:
  `sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c`
- Image ID/config digest: Docker image ID / OCI image-index digest
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`;
  the repository's `image_config_digest` field records this same runtime
  identity; linux/amd64 platform manifest descriptor
  `sha256:75b8ee6d204b6af0ab29f05c6b8cdc5b7bcaa214d52f6df13f8b9b37a317eafc`;
  OCI config descriptor
  `sha256:8f0736d3545e7ee1d79be85feceb462441bb7fe69af1b40144ca28e0a722b44c`
- RepoDigest: unavailable; no approved remote registry was configured or
  introduced. Docker's local, non-authoritative `RepoDigests` metadata reports
  `kvbench-measurement@sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- Image secret scan: PASS; 17 layers, 60,732 members, zero configured-secret,
  operator-env, or credential findings
- Package/tool identities: PASS; exact 333-package dpkg inventory, 35 base
  Python distributions, 15 Phase 3 distributions, and 25 tool hashes. Python
  3.12.3; PyTorch 2.12.1+cu130; compiled CUDA 13.0; Triton 3.7.1; nvcc
  13.0.88; nvdisasm/cuobjdump 13.0.85; Compute Sanitizer 2025.3.1.0;
  GCC/G++ 13.3.0; ncu 2026.2.1.0; nsys
  2026.1.3.425-261338342291v0
- Model weights in image: none
- Credentials in image: none

The observed and reviewed system lock files are byte-identical at SHA-256
`ebca524cf7d43fca376ebede5ff8f3cc8129b0b6fcaefc20a91816c48db3db65`.

## Container G0

- Run ID: `e00-20260724T195014.679255Z-a6025ae023e1-23dbe853`
- Evidence directory:
  `artifacts/phase6a/container_g0/e00-20260724T195014.679255Z-a6025ae023e1-23dbe853`
- Manifest SHA-256:
  `62c70aa390cf699699c80080efb399eed0177a529c07955a571d25526616c617`
- Native execution: PASS in a separate process with PTX JIT disabled
- Forced PTX/JIT: PASS in a separate process and fresh cache
- Compute Sanitizer: PASS; memcheck, initcheck, racecheck, and synccheck; zero
  errors
- nvdisasm/SASS: PASS
- cubin/PTX: `sm_120` cubin and `compute_120` PTX present
- GPU exclusivity: PASS; exact GPU UUID
  `GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b`; no foreign or unknown compute
  process and no query failure
- Result: PASS; all 17 G0 checks, schema, checksum ledger, inventory, and
  COMPLETE-last finalization pass

The preceding failed run
`e00-20260724T193628.460762Z-64e4e9f4d22b-b87cadba` remains immutable. It
exposed two narrow preflight defects: the pinned image's deliberate dpkg
documentation exclusions and the current driver's canonical all-dash idle
`pmon` row. The fixes retain fail-closed native-host and unexpected-path/process
behavior; the failed evidence was not overwritten.

## BF16 parity

- Eager run ID:
  `phase6a-bf16-parity-eager-20260724t195409003183z-a6025ae0-d47cea`
- Graph run ID:
  `phase6a-bf16-parity-cuda_graph-20260724t195438755178z-a6025ae0-71ec78`
- Numerical result: PASS; finite full-model and small-tensor controls; output
  checksum
  `03224a04945ab6985152ffe57f4b490d406c40b26fe01e8d4997f8f84b4e8cf2`
- Backend: exact forced Flash `torch_sdpa_flash_gqa`; fallback prohibited
- Cache geometry: B=1, L=128, BF16, 32 query heads, 8 KV heads, head dimension
  128; KV-head storage geometry verified
- GQA: non-materialization verified; no query-head storage
- Eager allocation: PASS under the frozen attributed-ephemeral criterion
- Graph capture/replay: PASS; two exact independent replays and stable pointers
- Graph allocation: zero events and zero allocated/reserved delta during replay
- Fallback: none

Both artifacts use model and tokenizer revision
`0e9e39f249a16976918f6564b8830bc894c89659`, the exact BF16 adapter, image
digest above, `quality_status: unvalidated`,
`claim_eligibility: performance_only`,
`performance_claim_eligible: false`, and
`measurement_scope: measurement_container_parity`. No timing was collected or
reported as a scientific result.

## Cloudflare R2 management

- Bucket: `kvbench-artifacts`
- Bucket exists: yes
- Public r2.dev: disabled
- Public custom domain: none; zero custom domains configured
- Bucket Lock rule: ID `kvbench-evidence-indefinite`; name absent
- Prefix: `kvbench/sha256/`, exact coverage
- Enabled: yes
- Retention: Indefinite
- REST result: PASS; provider `cloudflare_r2`; endpoint class
  `cloudflare_r2_s3`; public-state PASS; verified at
  `2026-07-24T20:02:29.647800Z`

No bucket, public-domain, or lock setting was created, edited, deleted, or
weakened. No AWS S3 Object Lock header was used.

## R2 durable publication

- Existing synthetic root:
  `bbb80210dc729dedc9dd25a24d61cfbedbbe9d05661b1f95e6af278df3d0c11e`
- Existing synthetic verification: PASS; 11 objects; clean retrieval at
  `2026-07-24T19:46:45.631778Z`; valid COMPLETE, inventory, checksums, root,
  and no unexpected object
- Container-G0 root:
  `85e1f49dea76d08b2cba4477d089a71759d529f03b2bc3538da3d15d8639455c`
- Container-G0 URI:
  `r2://kvbench-artifacts/kvbench/sha256/85e1f49dea76d08b2cba4477d089a71759d529f03b2bc3538da3d15d8639455c/`
- Initial publish: PASS at `2026-07-24T20:02:06.605171Z`; 222 conditional
  object creations, zero replacements
- COMPLETE-last: yes
- Clean retrieval: PASS at `2026-07-24T20:03:33.015715Z` into a new empty
  temporary directory; 222 objects
- Checksum result: PASS; COMPLETE, inventory, checksum ledger, exact object
  set, local E00 validation, and root digest all valid
- Credential leakage: none detected

The first publication invocation failed before upload because the Make target
did not expose the repository-root E00 validator. The minimal launcher fix was
focused-tested; the successful initial publication then created all 222
objects. No provisional prefix or overwritten object resulted from the failed
invocation.

## Execution authority

- Decision:
  `docs/decisions/0016-measurement-container-execution-authority.md`, Accepted
- Authorized image digest:
  `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- Floating tag: disallowed as authority
- Digest-change rule: new container G0, both BF16 parity smokes, durable
  publication/retrieval, and a new authority decision are required

## Blockers

- B-009: RESOLVED 2026-07-25
- B-010: RESOLVED 2026-07-25
- Remaining: none for Phase 6A; later method-specific and global gates remain
  unevaluated

## Gates

- G0: PASS on native host and PASS in the authorized Measurement Container
- G1: PASS for native-host BF16 engineering admission; container BF16 parity
  PASS; no new claim-bearing or unified G1 result
- G2-TQ: NOT EVALUATED / READY
- Global G2: NOT EVALUATED
- G3: NOT EVALUATED
- G4: NOT EVALUATED
- G5: NOT EVALUATED
- Full Scan: CLOSED

## Quality

- Quality execution: LOCKED
- Quality benchmark: not run
- PERFORMANCE_DATA_FROZEN: absent

## Preservation

- Phase 5 fixtures changed: no
- Historical evidence changed: no
- Existing run overwritten: no
- Formal performance data: none
- Profiler data: none; no Nsight execution or profiler result
- Quality data: none

The retrospective Phase 6 report remains SHA-256
`18015eeda156eeb1718d551919605244b6ff21c819182fcd5d66fa316558ed74`;
the Phase 5 report remains
`070b80cd8cd19c934006f14398d0f3312d11f54d5c5d0be276a595ba3051421b`.

## Changed files

- `Makefile`
- `preflight/measurement-container-system-packages.expected.json`
- `preflight/measurement-container-system-packages.lock.json`
- `preflight/process_query.py`
- `preflight/run_preflight.py`
- `scripts/phase6a_bf16_parity.py`
- `scripts/r2_artifact.py`
- `scripts/validate_phase2.py`
- focused Measurement Container, parity, governance, preflight, and R2 tests
- Decision 0016, this report, and current status/blocker/risk/task ledgers

No Dockerfile, Phase 5 fixture, historical evidence, retrospective Phase 6
report, or initial Phase 6A blocked report byte changed during remediation.

## Commands

- `make package-lock-check`
- `make test`
- `make checks`
- `make measurement-container`
- `docker image inspect kvbench-measurement:phase6a --format '{{json .Id}} {{json .RepoDigests}} {{json .Descriptor}}'`
- `ctr -n moby content get sha256:75b8ee6d204b6af0ab29f05c6b8cdc5b7bcaa214d52f6df13f8b9b37a317eafc`
- `make verify-measurement-container MEASUREMENT_IMAGE_CONFIG_DIGEST=...`
- `make preflight-container MEASUREMENT_IMAGE_CONFIG_DIGEST=...`
- `make phase6a-bf16-container-parity MEASUREMENT_IMAGE_CONFIG_DIGEST=... CONTAINER_G0_ARTIFACT=...`
- `make verify-artifact-r2 ROOT_SHA256=bbb80210...c11e`
- `make publish-artifact-r2 ARTIFACT=...`
- `make verify-artifact-r2 ROOT_SHA256=85e1f49d...9455c`
- `make validate-reference-turboquant`

## Tests

- package-lock: PASS
- make test: PASS
- make checks: PASS
- container tests: PASS, including 110 focused Phase 6A tests and exact-image
  verification
- R2 tests: PASS; management, synthetic verification, publication, and clean
  retrieval passed
- historical evidence: PASS; Phase 5 fixtures, retrospective Phase 6 report,
  initial Phase 6A blocked report, and completed evidence unchanged

## Commits

- `73ccba5775f23c9e0e99a0a73fb50e183025ab32` — close existing container
  certification paths exposed by live build preparation
- `64e4e9f4d22bd180c3e711bd31cf0004eba258db` — freeze the reviewed exact
  container system lock
- `a6025ae023e152db0a0813ea8923ffe0fcef3d44` — correct the two preserved live
  G0 observation defects
- final report/governance commit — launcher correction, authority decision,
  blocker closure, and this report

The preferred two-commit limit was exceeded because the two-pass observed-lock
workflow required a reviewed lock commit before the lock-bearing image, and
two later defects were directly exposed only by live G0 and R2 publication.
The existing Phase 6A commits were not amended or rewritten.

## Scientific interpretation

- B-009 was satisfied.
- B-010 was satisfied.

## Next action

Restart the Phase 6 TurboQuant Measurement Adapter only in a separate new
task. Do not treat Phase 6A parity artifacts as performance results or begin
Pilot, Full Scan, profiling, fitting, figures, or quality execution.

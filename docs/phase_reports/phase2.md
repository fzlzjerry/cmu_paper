# PHASE 2 REPORT

Status: PASS

## Entry verification

- Starting HEAD: `aba70be8220972c068c6fbeac279d54e34cddbde`
- Working tree at entry: clean (`git status --short` produced no output)
- Phase 1 remediation report verified: PASS in
  `docs/phase_reports/phase1-e00-remediation.md`; successful run ID, manifest
  hash, and evidence commit matched the repository
- G0: PASS, native-host scope only
- B-002: RESOLVED
- Original failed E00 evidence unchanged: yes; manifest and ledger identity
  verified and its complete checksum ledger passes
- Successful E00 evidence validated: yes; manifest and ledger identity
  verified and its complete checksum ledger passes
- Quality execution: LOCKED
- PERFORMANCE_DATA_FROZEN: absent
- Full scan: CLOSED

## Completed

- Inspected the pre-existing repository, certified E00 path, dependencies,
  Makefile, tests, documentation, configs, generated evidence, and absent
  Phase 2 implementation.
- Committed a plan and tooling/boundary decision before implementation.
- Added the intentional scaffold, experiment and measurement contracts,
  strict schemas, 11 versioned templates, CLI, local append-only writer,
  deterministic command reconstruction, Makefile interfaces, tests, and
  governance records.
- Kept Phase 3, timing, profiler, model download/execution, CUDA method work,
  and every quality benchmark out of scope.

## Repository scaffold

- Directories created: `configs/hardware/`, `configs/models/`,
  `configs/methods/`, `configs/plans/`, `src/kvbench/schema/`,
  `src/kvbench/runtime/`, `tests/schema/`, `analysis/`, `calibration/`, and
  tracked role markers for `artifacts/`, `reference/`, and
  `docs/method_notes/`
- Existing structure reused: `scripts/`, `tests/unit/`, `docs/decisions/`,
  `docs/phase_reports/`, the certified `preflight/` package, E00 tests, and
  immutable `docs/evidence/e00/`
- Deviations from CODEX_WORKFLOW.md: Pydantic was preferred but not present in
  the certified lock; Decision 0006 selects frozen standard-library models and
  JSON-compatible YAML. No durable artifact backend or measurement Dockerfile
  was invented; B-009/B-010 remain open. A method-notes README was added to
  make the otherwise-empty required role visible.
- Decision records created:
  `docs/decisions/0006-phase2-schema-tooling-and-artifact-boundary.md`

## Experiment contracts

- docs/experiment_contract.md: created; freezes identity, comparison, lane,
  provenance, lifecycle, exclusion, fallback, dirty-tree, quality-lock, and
  claim-eligibility semantics
- docs/measurement_protocol.md: created; freezes future warmup/measured-step,
  timing boundary, process-replicate, randomization, telemetry, Graph/JIT,
  profiler separation, finalization, checksum, and no-selective-rerun behavior
- Key semantics frozen: same-work and capacity claims remain separate;
  timing/nsys/ncu and graph-on/graph-off lanes cannot be mixed; profiler timing
  is not benchmark timing; `r_hbm` requires direct NCU evidence; pre-quality
  performance metadata is `unvalidated`/`performance_only`; quality requires
  `PERFORMANCE_DATA_FROZEN` and remains locked

## Schemas

- Schema system/version: frozen typed standard-library dataclasses with strict
  parsers; each top-level/control model has an explicit `kvbench...v1` or
  `...1.0.0` schema version; canonicalization is `kvbench-json-v1`
- Models implemented: ExperimentConfig, MethodConfig,
  HardwareManifest, ModelIdentity, SoftwareEnvironment,
  MethodConfigFingerprint, SampleRecord, RunSummary, ExclusionRecord,
  RunManifest, ArtifactInventory, ArtifactEntry, LifecycleRecord,
  CompletionMarker, QualityStatus, RunStatus, RunKind, ClaimClass, plan/grid/
  measurement models, and method-specific parameter models
- Unknown-field policy: reject every unknown field; there is no implicit
  extension namespace in schema version 1
- Canonical serialization: deterministic compact sorted UTF-8 JSON, no NaN,
  SHA-256 over canonical bytes
- Schema tests: 23 PASS, including unknown fields, enums, positive shapes,
  method completeness/geometry, incompatible lanes, locked admission controls,
  preregistered grids, required fingerprints, versions, duplicate keys,
  non-finite numbers, quality state, and deterministic serialization

## Example configurations

- Files created: one certified-host identity; one unresolved primary-model
  template; BF16, TurboQuant, KIVI, and KVQuant method templates; smoke, pilot,
  Graph A/B, profiler-subset, and full-scan plans
- Validation status: 11/11 strict documents and all plan reference bundles PASS
- Unresolved identities represented: explicit `unresolved` or `blocked`
  Resolution records with null model/container/backend/cache/calibration/runtime
  fields and named blockers; no revision, commit, artifact hash, dtype, or
  backend support was invented

## CLI

- Commands implemented: `kvbench validate-config`, `kvbench preflight`,
  `kvbench run --plan`, `kvbench validate-run`, and `kvbench summarize`
- Dry-run behavior: validates a complete referenced bundle, evaluates and
  reports blockers, reconstructs canonical argv, and explicitly reports zero
  execution/timing/profiler/quality activity
- Fail-closed behavior: `run` without `--dry-run` exits nonzero with
  `phase_not_implemented`; it creates no run or measurement artifact
- Commands intentionally deferred: all model/cache/decode/performance
  execution, profiler integration, quality execution, fitting, figures, and
  reproduction execution

## Artifact lifecycle

- Staging strategy: validated ID plus permanent exclusive reservation and a
  unique same-filesystem `.kvbench-staging/<id>.<token>.staging` directory
- Atomic finalization: strict prevalidation, fsync, independent staged
  validation, removal of write bits, then Linux
  `renameat2(RENAME_NOREPLACE)`; unsupported no-replace semantics fail closed
- Existing-run protection: final path, reservation, or incomplete stage rejects
  the ID; concurrent duplicates have exactly one winner
- Failed-run preservation: every defined terminal failure uses the same
  inventory/checksum/completion/promotion path and is never deleted
- Completion marker: strict versioned marker written last and independently
  correlated with run ID, status, manifest, inventory, and ledger hashes
- Checksums: canonical lexically ordered SHA-256 ledger with exact payload
  coverage and tamper detection
- Artifact inventory: strict sorted path/role/size/SHA-256 entries with exact
  payload coverage
- Path-safety controls: canonical lowercase run IDs, repository/formal-evidence
  overlap rejection that callers cannot disable, traversal/C0/symlink/hardlink
  rejection, real control-directory checks, exclusive writes, and no finalized
  edit API

## Command reconstruction

- Manifest reconstruction: saved plan source, command argv, Git/dirty state,
  hardware/software/model/method/contract fingerprints, runner/graph/run kind,
  seed, replicate, and artifact version are independently schema validated;
  finalization rejects any initial-to-final provenance change
- Determinism test: PASS; reconstruction from saved mappings and typed manifests
  is identical without consulting mutable global state

## Makefile

- Targets added or updated: `bootstrap`, `preflight` (preserved),
  `preflight-unit` (preserved), `test`, `test-cuda`, `test-graph`, `smoke`,
  `pilot`, `full-scan`, `profile-subset`, `fit`, `figures`, and `reproduce`
- Later-phase targets: validation-only dry runs or explicit
  `phase_not_implemented` nonzero exits; smoke reports that its plan scope is
  all methods even when a method contract is selected for validation
- Confirmation no timing was collected: yes; no target invoked a model,
  profiler, quality tool, or timing collector

## Changed files

- Root/tooling: `.gitignore`, `Makefile`, `README.md`, `pyproject.toml`,
  `scripts/validate_phase2.py`
- Scaffold markers: `analysis/README.md`, `artifacts/README.md`,
  `calibration/README.md`, `reference/README.md`,
  `docs/method_notes/README.md`
- Configs: all 11 files under `configs/{hardware,models,methods,plans}/`
- Contracts/governance: `docs/experiment_contract.md`,
  `docs/measurement_protocol.md`, `docs/artifact_policy.md`,
  `docs/status.md`, `docs/blockers.md`, `docs/risk_register.md`,
  `docs/tasks.md`, Decision 0006, the Phase 2 plan, and this report
- Package: `src/kvbench/{__init__,__main__,cli,config,errors,validation}.py`,
  `src/kvbench/schema/{__init__,base,config,result}.py`, and
  `src/kvbench/runtime/{__init__,artifacts,command}.py`
- Tests: `tests/schema/{__init__,test_config_schema}.py` and
  `tests/unit/test_phase2_{artifacts,cli,governance}.py`

## Commands executed

- Entry: `git status --short`, `git rev-parse HEAD`, report/status/blocker/lock
  inspection, SHA-256 checks, and E00 ledger validation
- Implementation validation: dependency-free import/compile, config-bundle
  validation, AST/lint, annotation resolution, `git diff --check`, scoped Git
  checks, and direct unit discoveries
- Acceptance: `make test` (final PASS), `make preflight-unit` (33 PASS), and
  `make bootstrap` (PASS, installed=false)
- Preservation: both `sha256sum --check --quiet checksums.sha256` commands from
  their respective E00 run directories; manifest/protocol SHA-256; E00 Git diff
  and status; freeze-marker search
- A combined ledger command was once invoked from the repository root and
  failed because ledger paths are run-relative. No evidence changed. The exact
  commands were rerun from each run directory and both returned exit 0.
- Formal `make preflight` was not run because certified E00 code/environment did
  not change. A managed safety control declined direct `make pilot` invocation
  before execution; the inspected validation-only recipe is covered by config
  and black-box CLI dry-run tests, and no bypass was attempted.

## Tests and evidence

- Unit tests: 31/31 artifact, manifest, CLI, and governance tests PASS
- Schema tests: 23/23 PASS
- CLI tests: 6/6 PASS, including config success/failure, dry-run, non-dry
  fail-closed, valid/tampered run validation, and incomplete summary
- Artifact immutability tests: PASS for atomic completion, existing/concurrent
  IDs, API lock, failed/interrupted preservation, formal-root protection,
  symlink/hardlink/control safety, and initial/final provenance lock
- Checksum/tamper tests: PASS for payload, inventory, lifecycle, completion,
  and ledger validation
- Command reconstruction tests: PASS and ambient-state independent
- Formatting: PASS; dependency-free UTF-8/newline/whitespace check
- Lint: PASS; dependency/scope-aware AST compilation check
- Type checking: repository annotation-resolution check PASS; Pyright/mypy are
  not installed, so no third-party static type-analysis result is claimed
- Provenance validation: PASS
- Staged-scope validation: PASS against Phase 2 entry commit and allowlist
- E00 byte-identity validation: PASS; Git diff/status empty, exact manifests and
  complete checksum ledgers validated

## Admission gates

- G0: PASS (native-host scope retained)
- G1: NOT EVALUATED
- G2: NOT EVALUATED for TurboQuant, KIVI, and KVQuant
- G3: NOT EVALUATED
- G4: NOT EVALUATED
- G5: NOT EVALUATED
- Full scan: CLOSED

## Quality governance

- Quality protocol files unchanged except approved documentation fixes:
  unchanged byte-for-byte; no protocol fix was made
- Quality execution: LOCKED
- Quality-only dependencies installed: none
- Quality benchmark executed: none
- PERFORMANCE_DATA_FROZEN: absent

## Preservation checks

- Original failed E00 manifest SHA-256 still:
  `0720734d29c90f609e51cf4c5e4f0b1fadce220e23e146e566f860bb962c0035`
- Successful E00 manifest SHA-256 still:
  `d054df714bb5eea1f114bf10a03a2879f56ec8d17d3b07e24fe6efcaba6b7aca`
- Completed E00 evidence modified: no
- Raw benchmark data created: no; mutable tests used synthetic temporary data
  that was not scientific evidence and was removed with its temporary roots
- Profiler timing created: no

## Commits created

- `f313817cabed42291f7a6f8ebb08bb68fbd30233` — plan and Decision 0006
- `e3917c0c044639c17bd4de9d0ad712253bd2beca` — scaffold, contracts, schemas,
  and templates
- `140754e8406ba275ebb2bcf8ba2c8ccc672e6d79` — CLI, artifact lifecycle,
  command reconstruction, validator, and Makefile
- `f731b7ba9e56d9231a8b7a721459f932a44de3f4` — admission/method completeness
  hardening and schema regressions
- `407583154cc860869b335b9c6781a95241510a01` — artifact/CLI/governance tests
  and governance ledgers
- Final report/completion-record commit: this report's containing commit

## Observed risks

- Local chmod, ledgers, and completion markers are not an external append-only
  attestation; B-009 remains open for durable retention and publication.
- No digest-pinned measurement container exists; B-010 remains open, and host
  G0 cannot be inferred to certify a future container.
- Local atomic promotion requires Linux `renameat2(RENAME_NOREPLACE)` and fails
  closed if unavailable.
- Neither Pyright nor mypy is installed. Runtime annotation resolution passed,
  but a third-party static type-analysis result is not asserted.
- Model, method authority/equivalence, calibration, and method execution
  identities remain explicitly unresolved under B-003 through B-006.

## Blockers

- No blocker prevents this Phase 2 PASS.
- B-009 remains open for durable artifact storage, external attestation,
  immutable locators, retention, and retrieval verification.
- B-010 remains open for a digest-pinned measurement container and
  container-parity G0. It blocks E02 and all later CUDA/timing work.
- Other downstream blockers remain unchanged; no admission gate was inferred
  from documentation or schema existence.

## Scientific interpretation

Current evidence supports only that the Phase 2 repository contracts,
validation behavior, local append-only lifecycle, deterministic reconstruction,
and governance regressions operate as tested on synthetic temporary fixtures.
It does not support any latency, capacity, physical-HBM, CUDA-method, model,
profiler, quality, or paper-result claim.

## Next action

Phase 3 may be proposed in a new task, but must not start automatically. Any
proposal must preserve the open B-009/B-010 gates and cannot begin E02 CUDA or
timing work before digest-pinned container parity is established.

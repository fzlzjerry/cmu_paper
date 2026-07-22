# Phase 2 implementation plan

- Status: approved for implementation by the operator's Phase 2 instruction
- Phase: E01 repository scaffold and schemas only
- Starting commit: `aba70be8220972c068c6fbeac279d54e34cddbde`
- Entry tree: clean
- Date: 2026-07-22

## Scope and sequencing

Work follows `Inspect -> Plan -> Implement -> Test -> Record -> Phase Report`.
This plan is committed before implementation. Phase 3, model execution, CUDA
method work, timing, profiling, model downloads, and every quality benchmark
remain out of scope.

## Inventory and reuse

The repository currently contains the Phase 0 records and the independently
certified E00 implementation. There is no `pyproject.toml`, `src/kvbench/`,
general CLI, general run schema, plan/config tree, or general artifact writer.
The existing `Makefile`, `scripts/preflight.sh`, `preflight/`, E00 tests, and
both finalized `docs/evidence/e00/` runs are retained unchanged for certified
preflight behavior. E00's exclusive writes, SHA-256 helpers, completion marker,
and Linux `renameat2(RENAME_NOREPLACE)` strategy inform the new implementation,
but E00 code will not be refactored because doing so would alter the certified
Phase 1 path.

Existing directories reused unchanged include `scripts/`, `tests/unit/`,
`docs/decisions/`, `docs/method_notes/`, `reference/`'s intended role from the
workflow, and the narrow Git-tracked E00 evidence boundary. New general raw
artifacts remain under ignored `artifacts/`; tests use only temporary roots.

## Exact files

### Create

- `README.md`
- `pyproject.toml`
- `analysis/README.md`
- `artifacts/README.md`
- `calibration/README.md`
- `reference/README.md`
- `configs/hardware/rtx_pro_6000.yaml`
- `configs/models/primary_gqa_model.yaml`
- `configs/methods/bf16.yaml`
- `configs/methods/turboquant.yaml`
- `configs/methods/kivi.yaml`
- `configs/methods/kvquant.yaml`
- `configs/plans/smoke.yaml`
- `configs/plans/pilot.yaml`
- `configs/plans/graph_ab.yaml`
- `configs/plans/profiler_subset.yaml`
- `configs/plans/full_scan.yaml`
- `docs/experiment_contract.md`
- `docs/measurement_protocol.md`
- `docs/decisions/0006-phase2-schema-tooling-and-artifact-boundary.md`
- `src/kvbench/__init__.py`
- `src/kvbench/__main__.py`
- `src/kvbench/errors.py`
- `src/kvbench/config.py`
- `src/kvbench/validation.py`
- `src/kvbench/cli.py`
- `src/kvbench/schema/__init__.py`
- `src/kvbench/schema/base.py`
- `src/kvbench/schema/config.py`
- `src/kvbench/schema/result.py`
- `src/kvbench/runtime/__init__.py`
- `src/kvbench/runtime/artifacts.py`
- `src/kvbench/runtime/command.py`
- `scripts/validate_phase2.py`
- `tests/schema/__init__.py`
- `tests/schema/test_config_schema.py`
- `tests/unit/test_phase2_artifacts.py`
- `tests/unit/test_phase2_cli.py`
- `tests/unit/test_phase2_governance.py`
- `docs/phase_reports/phase2.md`

### Modify

- `.gitignore`
- `Makefile`
- `docs/artifact_policy.md`
- `docs/status.md`
- `docs/blockers.md`
- `docs/risk_register.md`
- `docs/tasks.md`
- this plan, only to record completion and actual validation evidence

The quality protocol files and completed E00 directories are not modified.

## Schema implementation

The repository will use frozen, typed standard-library models with explicit
parsers and validators. Decision 0006 records why Pydantic is not introduced
into the exact E00 environment. All models reject unknown fields and emit
field-path errors without echoing values. Versioned models include:

- `ExperimentConfig`
- `MethodConfig` and `MethodConfigFingerprint`
- `HardwareManifest`
- `ModelIdentity`
- `SoftwareEnvironment`
- `SampleRecord`
- `RunSummary`
- `ExclusionRecord`
- `RunManifest`
- `ArtifactInventory`
- `QualityStatus`
- `RunStatus`, `RunKind`, and `ClaimClass`

Canonical serialization is sorted, compact UTF-8 JSON with no NaN values and a
terminal newline only for stored pretty documents. Hashes use SHA-256 over the
compact canonical bytes. Versioned `.yaml` examples use the JSON-compatible
YAML 1.2 subset, so validation requires no new parser dependency.

Validation covers positive batch/context/step counts, method-specific allowed
fields and value domains, status/run-kind/metric compatibility, required
fingerprints for resolved runs, explicit unresolved identity states for
templates, and direct-measurement provenance for `r_hbm`. Estimated traffic
can never populate `r_hbm`.

## CLI contract

Expose:

- `kvbench validate-config <path>`
- `kvbench preflight`
- `kvbench run --plan <yaml> [--dry-run]`
- `kvbench validate-run <run_dir>`
- `kvbench summarize <run_dir>`

`preflight` delegates to the existing sanitized E00 launcher. It is implemented
but not invoked during Phase 2 validation. `run --dry-run` validates, reports
admission blockers, and reconstructs the intended command without execution.
`run` without `--dry-run` exits nonzero with `phase_not_implemented` before
creating artifacts. No command installs packages, downloads models, silently
runs preflight, invokes a profiler, or reaches quality code.

## Artifact lifecycle

The supported state flow is:

```text
created -> running -> finalizing -> completed
                            `-> terminal failure status
```

Terminal failures are `build_failed`, `runtime_failed`, `numerical_failed`,
`graph_capture_failed`, `profiler_failed`, `capacity_infeasible`, `unstable`,
`backend_fallback`, `unsupported_geometry`, and `aborted`.

Creation reserves the validated run ID with exclusive `mkdir`, creates a
unique same-filesystem staging directory, and writes immutable initial/lifecycle
records. Writes use exclusive creation, reject absolute/traversing/symlinked
paths, flush, and `fsync`. Finalization transitions once, validates the final
manifest, creates a complete payload inventory and SHA-256 ledger, writes the
authenticated completion marker last, verifies every byte, removes write bits,
and uses no-replace atomic rename. Existing final IDs, reservations, and
incomplete stages are conflicts and are never reused. Unexpected interruption
leaves a distinguishable staging directory and reservation for audit. Supported
APIs have no operation that edits finalized runs. Both successful and failed
terminal runs are finalized and preserved.

Artifact roots are caller-configurable. Root validation rejects the repository
root, `docs/evidence`, either E00 run, and any ancestor/descendant overlap with
formal immutable evidence. Mutable tests use `TemporaryDirectory` only.

## Command reconstruction

`RunManifest` stores the normalized plan source, canonical config hash, Git SHA,
dirty-tree state, software and hardware identities/fingerprints, method and
model identities/fingerprints, runner kind, Graph mode, run kind, seed, and
artifact schema version. Reconstruction consumes only manifest fields and
returns the original `kvbench run --plan ... --dry-run` argv. A test changes
ambient process state and proves reconstruction is unchanged.

## Error taxonomy

- `config_load_error`
- `schema_validation_error`
- `admission_blocked`
- `phase_not_implemented`
- `artifact_safety_error`
- `artifact_conflict`
- `artifact_state_error`
- `checksum_mismatch`
- `provenance_error`

CLI errors are structured, actionable, nonzero, and do not echo secret values.

## Unit-test matrix

Schema/config tests cover valid minimal documents; unknown fields; invalid
enums, batch size, and context length; malformed method settings; incompatible
run fields; required fingerprints; schema version; deterministic canonical
serialization; all example files; unresolved identities; and `r_hbm` evidence.

Artifact tests cover staging creation; atomic success and terminal-failure
finalization; duplicate/concurrent IDs; immutable completed APIs; interrupted
stages; complete inventories; checksum generation and tamper detection; unsafe
IDs and paths; formal-root rejection; and no writes to E00 evidence.

Manifest/CLI tests cover independent manifest validation; deterministic command
reconstruction; retained fingerprints/dirty state/null metrics; config success
and failure; run validation success and checksum failure; incomplete/failed
summaries; Phase 3 fail-closed behavior; and absence of quality execution paths.

Governance regression tests verify both E00 manifest hashes and checksum
ledgers, quality protocol hashes and `LOCKED` state, absent
`PERFORMANCE_DATA_FROZEN`, closed full scan, and Phase 2 scope exclusions.

## Makefile and validation

Keep `make preflight` and `make preflight-unit` semantics unchanged. Add
`bootstrap`, `test`, `test-cuda`, `test-graph`, `smoke`, `pilot`, `full-scan`,
`profile-subset`, `fit`, `figures`, and `reproduce`. Later-phase validation
targets either run only a declared dry run or fail with
`phase_not_implemented`; none collect timing/profiler/quality data or install
packages. Repository checks will cover formatting, lint/AST compilation,
annotation resolution, schema/example validation, provenance, staged scope,
immutable E00 identity, and E00 package-lock consistency.

## Deferred to Phase 3 or later

No adapter, model loader, static KV cache, decode runner, Graph benchmark,
method kernel, timing collector, profiler integration, quality runner, model
download, container build, fitting code, or figure generation is implemented.
Phase 3 remains only a proposed next task.

## Blockers B-009 and B-010

B-009 remains open after Phase 2 because local append-only enforcement is not
the missing durable store: no immutable remote locator, retention policy, or
publication mechanism has been selected or exercised. The blocker may be
narrowed to that unsatisfied operational acceptance criterion, never closed.

B-010 remains open because no digest-pinned measurement container is created
and native-host G0 does not certify a future container. No container-parity
claim or G1 admission follows from this phase.

## Planned commit scopes

1. Plan and Decision 0006.
2. Contracts, schemas, scaffold, and example configs.
3. CLI, artifact lifecycle, command reconstruction, and Makefile interfaces.
4. Tests and governance updates.
5. Final Phase 2 report and recorded validation evidence.

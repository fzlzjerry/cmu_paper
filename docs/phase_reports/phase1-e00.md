# PHASE 1 / E00 REPORT

Status: BLOCKED

## Completed

- Implemented and committed the E00 native-host CUDA certification collector,
  strict manifest schema, source and dependency audits, process supervision,
  numerical/Graph/allocation probes, and four Compute Sanitizer lanes at
  commit `980eff7b6f5904c4828aa79d684c01a8dc45320d`.
- Recorded the failure-evidence and sanitizer interpretation policy in Decision
  0004 without changing benchmark semantics.
- Passed the E00 unit and static-validation suite before the formal run.
- Ran formal E00 once from the clean implementation commit and finalized
  append-only run `e00-20260722T041628.190813Z-980eff7b6f59-0dd71f2d` as FAIL.
- Independently verified the finalized checksum ledger, strict manifest schema,
  recorded code SHA, completion marker, and absence of benchmark timing.
- Diagnosed the gate failure using only finalized evidence and read-only host
  inspection; the matching CUDA installation lacks `nvdisasm`.

## Changed files

- `Makefile`
- `preflight/` collector, schema, lock, CUDA extension, and probes
- `scripts/preflight.sh`
- `tests/unit/`, `tests/golden/`, `tests/graph_capture/`, `tests/allocation/`,
  and `tests/cuda/` E00 coverage
- `docs/decisions/0004-e00-failure-evidence-and-memcheck.md`
- `docs/evidence/e00/e00-20260722T041628.190813Z-980eff7b6f59-0dd71f2d/`
- `docs/status.md`, `docs/blockers.md`, `docs/tasks.md`, and
  `docs/risk_register.md`
- `docs/phase_reports/phase1-e00.md`

No completed-run artifact was edited or deleted.

## Commands executed

- `make preflight-unit`
- JSON Schema Draft 2020-12 validation, duplicate-key audits, Bash syntax
  checking, Python compilation, and Git whitespace/status checks
- `make preflight`
- `sha256sum --check --quiet checksums.sha256` inside the finalized run
- Read-only manifest inspection and host searches for `nvdisasm`

No benchmark, profiler, package installer, or replacement backend was run.

## Tests and evidence

- E00 unit suite: 32/32 tests passed.
- Formal run status: FAIL; `benchmark_timing_collected=false`.
- Code SHA: `980eff7b6f5904c4828aa79d684c01a8dc45320d`.
- Manifest SHA-256:
  `0720734d29c90f609e51cf4c5e4f0b1fadce220e23e146e566f860bb962c0035`.
- Checksum-ledger SHA-256:
  `8716fc317747e7e9b5c06017cb8e5339df610c5a89d0d7fbee82ad07fbc68b52`.
- The extension build exited successfully and produced a binary. Target listing
  found `xor_kernel.sm_120.cubin` and PTX; PTX dumping succeeded with target
  `sm_120`.
- SASS dumping exited nonzero with: `Could not find executable file
  'nvdisasm'`; `command -v`, expected CUDA 13.0 path, `/usr/bin`, and installed
  package ownership checks found no `nvdisasm`.
- Native certification execution, forced-PTX/JIT execution, and all Compute
  Sanitizer lanes are truthfully recorded as NOT_RUN.

## Admission gates

- Phase 0 acceptance: PASS.
- G0: FAIL.
- G1: NOT EVALUATED.
- G2-TQ: NOT EVALUATED.
- G2-KIVI: NOT EVALUATED.
- G2-KVQ: NOT EVALUATED.
- G3-G5: NOT EVALUATED.
- Full-scan admission: CLOSED.

## Observed risks

- R-001 materialized: the installed CUDA toolchain is incomplete for required
  SASS inspection even though compilation and target listing succeeded.
- Runtime, numerical, Graph, allocation, forced-PTX, and sanitizer behavior on
  this host remain uncertified.
- A later native-host G0 PASS still cannot certify the measurement container;
  B-010 requires container-parity G0 after E01 pins an image digest.

## Blockers

- B-002 is active: matching CUDA 13.0 `nvdisasm` is absent, so G0 failed.
- E01 and every non-E00 CUDA or timing task remain closed.
- B-009 and B-010 continue to block claim-bearing work even after a future
  native-host G0 PASS.

## Scientific interpretation

The evidence supports only that the clean E00 implementation produced a CUDA
binary containing the expected native and PTX target names and that required
SASS inspection could not complete because `nvdisasm` is absent. Certification
runtime did not execute, so this phase supports no claim about numerical
correctness, CUDA Graph capture, allocation behavior, sanitizer cleanliness,
native-versus-PTX execution, latency, throughput, memory, or method performance.

## Next action

Provision the CUDA 13.0-matched `nvdisasm` (expected package
`cuda-nvdisasm-13-0`), add its exact package identity, executable path, hash,
and version to the reviewed system/tool lock, commit that environment change,
and rerun the full E00 workflow under a new run ID. Preserve the failed run
unchanged and do not begin E01 unless the new G0 result passes.

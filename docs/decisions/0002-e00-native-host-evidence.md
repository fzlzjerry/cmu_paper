# Decision 0002: E00 native-host evidence boundary

- Status: Accepted
- Date: 2026-07-22
- Authority: CODEX_WORKFLOW.md phase ordering and the operator instruction to
  continue automatically without skipping gates
- Supersedes: none
- Superseded by: none

## Context

Phase 1 requires G0 hardware/toolchain certification before Phase 2 creates the
general artifact writer and schemas. B-009 previously blocked every admission
run on that Phase 2 writer, creating a circular dependency. The current process
runs directly on the host, not in an OCI container, so no honest container image
digest exists.

## Decision

1. E00 is a native-host, non-performance hardware certification. Its manifest
   records `execution_environment.kind=native_host`; null container runtime,
   image-reference, and image-digest fields; the explicit status
   `container.digest_status=not_applicable_native_host`; and
   `performance_claim_eligible=false`. No host hash is presented as a digest.
2. A technically passing G0 certifies only this GPU, driver, host, CUDA compiler,
   PyTorch extension path, PTX/JIT path, and sanitizer path. It does not make any
   later performance run eligible.
3. E00 raw logs are finalized under `docs/evidence/e00/<run_id>/` using a unique
   run ID, exclusively created temporary sibling directory, atomic rename,
   completion marker, and a SHA-256 ledger. Successful and failed completed E00
   directories are retained append-only and Git-tracked.
4. The implementation commit must be clean before any E00 run directory is
   created. Raw evidence and the Phase 1 report are committed afterward.
5. B-009 blocks E01 acceptance, G1-G5 admission, and every claim-bearing run,
   but not the prerequisite G0 certification. Phase 2 must still implement the
   general durable artifact policy before those later gates.
6. E00 collects no benchmark timing and supports no performance claim.
7. After E01 creates the digest-pinned measurement container, the identical
   preflight must pass inside it before E02, method CUDA implementation, or any
   timing work begins.
8. Graphics-only (`G`) GPU processes are allowed but recorded. A foreign `C`
   or `C+G` process fails G0; only an exact supervised child or descendant,
   identified by PID and process start time rather than executable name, is
   allowed. Unknown or unclassifiable GPU process types fail closed.

## Consequences

- A null container digest is transparent evidence of native-host execution, not
  a reproducibility substitute.
- A pinned measurement-container digest remains mandatory before G1 or any
  performance data collection.
- Container-parity G0 is a prerequisite to E02 and all later CUDA or timing
  work, even if native-host G0 passes.
- Failure of build, native execution, forced PTX/JIT, CUDA Graph replay,
  allocation stability, Compute Sanitizer, or required manifest collection
  fails G0 without fallback.

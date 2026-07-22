# E00 hardware preflight

This directory implements Phase 1 / Gate G0 from `CODEX_WORKFLOW.md`.
It certifies the recorded RTX PRO 6000 host, CUDA toolchain, native SASS path,
forced-PTX path, and sanitizer path. It collects no benchmark timing and cannot
support a performance claim.

## Reproducibility boundary

- `requirements-e00.txt` pins the complete Python closure with wheel SHA-256
  hashes.
- `system-packages.lock.json` pins the observed dpkg packages and executable
  hashes.
- `e00_manifest.schema.json` is validated offline during every completed run.
- `make preflight` performs no install, download, profiler collection, or
  benchmark measurement.
- The collector refuses a dirty tree before creating a run directory.

Decision 0002 permits native-host G0 only as non-performance certification.
Decision 0003 fixes the certification operation and test semantics. Decision
0004 requires truthful completed-FAIL evidence and fixes the exact sanitizer
summary policy. E02 and all method CUDA or timing work remain blocked until
E01 supplies a digest-pinned measurement container and the identical preflight
passes inside it.

## Usage

From a clean committed tree with the pinned project virtual environment and
system packages already installed:

```bash
make preflight-unit
make preflight
```

The formal collector writes a uniquely named temporary sibling only after its
clean-tree and initial process checks. It records commands, allowlisted
environment values, raw stdout/stderr, GPU process snapshots, build products,
test results, and checksums. It then writes `COMPLETE` last and atomically
renames to:

```text
docs/evidence/e00/<run_id>/
```

Both completed PASS and completed FAIL directories are append-only. An
unexpected collector failure retains its `.<run_id>.tmp` directory for audit
and must be diagnosed rather than overwritten or deleted.

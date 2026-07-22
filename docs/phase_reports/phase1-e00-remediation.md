# PHASE 1 REMEDIATION REPORT

Status: PASS

## Scope and preservation

- Continued Phase 1 only; Phase 0 was not repeated and Phase 2 was not begun.
- Preserved failed run
  `e00-20260722T041628.190813Z-980eff7b6f59-0dd71f2d` byte-for-byte. Its
  manifest SHA-256 remains
  `0720734d29c90f609e51cf4c5e4f0b1fadce220e23e146e566f860bb962c0035`.
- Ran no benchmark, profiler, latency measurement, performance scan, quality
  benchmark, or quality-only dependency installation.
- Kept quality execution LOCKED because no approved `PERFORMANCE_DATA_FROZEN`
  milestone exists.

## Quality protocol preregistration

- Read the workflow and both quality documents in full and recorded their
  precedence in Decision 0005.
- Retained the older addendum's existing supersession notice: its scientific
  requirements remain applicable, while
  `CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md` controls execution order.
- Committed the two quality documents, Decision 0005, and the status update
  separately at 6535a6f6a4e5caa53213e917e9fcf8fc9c0f0190.

## Environment remediation

- Verified APT candidate `cuda-nvdisasm-13-0=13.0.85-1` for `amd64` from the
  NVIDIA CUDA Ubuntu 24.04 x86_64 repository before installation.
- The proposed transaction contained one new package, zero upgrades, and zero
  removals. No unrelated CUDA, driver, compiler, PyTorch, Triton, or system
  package was changed.
- Installed only the authorized package and locked its package, version,
  architecture, binary path, version output, ownership, and SHA-256 at commit
  6442ba1f7554ea0ebf0b3bb1a920c94567cab689.
- Installed binary: `/usr/local/cuda-13.0/bin/nvdisasm`, owned by
  `cuda-nvdisasm-13-0`, SHA-256
  `3c27bded09bd877807207b62db8186a0a9a359d10311ab6e2c885f9b418c9f41`.

## New immutable E00 run

- Run ID: `e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32`.
- Source Git SHA: `6442ba1f7554ea0ebf0b3bb1a920c94567cab689`; the source worktree was clean.
- Evidence directory:
  `docs/evidence/e00/e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32/`.
- Manifest SHA-256:
  `d054df714bb5eea1f114bf10a03a2879f56ec8d17d3b07e24fe6efcaba6b7aca`.
- Checksum-ledger SHA-256:
  `5a610162163979aca97beb2b7b0b480befb85d0b4e63b77c26ec46c36864eca8`.
- The finalized manifest records `status=PASS`, `gate=G0`,
  `run_kind=hardware_preflight`, and `benchmark_timing_collected=false`.

## Commands and validation

- `make preflight-unit`: 33/33 tests passed.
- Repository-defined JSON Schema, Python syntax, Bash syntax, dependency/tool
  identity, source-integrity, scope, and immutable-evidence checks passed.
- `make preflight`: completed and finalized the new run as PASS.
- The new and original `checksums.sha256` ledgers validate against their
  respective finalized directories.
- The manifest schema, source provenance, staged scope, artifact immutability,
  and staged byte identity were validated before the evidence commit.

## G0 evidence

- The minimal PyTorch CUDA extension built and contains native `sm_120` cubin
  and `compute_120` PTX targets.
- Native execution passed with PTX JIT disabled; forced PTX/JIT execution
  passed in a separate process and fresh CUDA cache.
- Numerical golden tests passed exactly for all five recorded cases.
- CUDA Graph capture/replay passed, and the allocation test recorded zero
  allocated/reserved deltas with stable preallocated pointers.
- Compute Sanitizer memcheck, initcheck, racecheck, and synccheck all passed;
  memcheck also recorded zero leaked bytes.
- `nvdisasm` inspected the produced SM120 code; cubin/PTX listing and dumps
  passed, and no no-kernel-image or unsupported-architecture fallback was
  detected.
- The manifest captures the RTX PRO 6000 Blackwell Workstation Edition,
  compute capability 12.0, clocks, power, VRAM, ECC, driver, CUDA, compiler,
  PyTorch, Triton, host, and native-host/container provenance.
- The process audit recorded zero foreign or unknown CUDA compute workloads.

## Admission result

- G0: PASS for this native host.
- G1-G5: NOT EVALUATED.
- Full scan: CLOSED.
- B-002: resolved by the exact tool lock and the new passing evidence.
- R-001: mitigated for native-host G0; container parity remains tracked by
  R-022/B-010.

## Scientific interpretation

The current evidence directly supports only native-host G0 certification for
the recorded hardware, software, source, and tool fingerprints. It makes no
performance or quality claim and does not certify the future measurement
container. Phase 2 may be proposed in a new task, but was not started here.

# Phase 3 Remediation Report

- Status: BLOCKED
- Date: 2026-07-23
- Execution Git SHA:
  `7bd6dd48c1d88ac2b61684b02cc636f66b121054`
- Scope: native-host BF16 G1 remediation only

## Entry verification

- The working tree was clean at the execution SHA.
- `make checks`, `make test`, `make test-cuda`, and `make test-graph`
  passed before campaign execution.
- The immutable prior G1 report remained valid with SHA-256
  `060a88283f083e281692a2c471d279da9bfc635e0f513e2dca588ed729d85c7d`.
- Both prior campaigns and all 20 prior runs independently validated as
  complete and checksum-valid. A 600-file pre-execution checksum baseline had
  aggregate manifest SHA-256
  `e3f6cfaf7bf20728e82503ad5c2905750ea7afe4a4dbf0ac6692b67674095518`.
- G0 remained PASS, G1 remained FAIL, quality execution remained LOCKED, Full
  Scan remained CLOSED, and `PERFORMANCE_DATA_FROZEN` remained absent.

## Pre-campaign admission

All required controls passed before execution:

- failure-taxonomy, allocation-attribution, process-ownership, and report
  publication unit controls;
- forced-Flash GQA/MHA CUDA device-kernel controls;
- exact endpoint eager audit with fully attributed allocations;
- retained CUDA Graph audit with strict zero allocation;
- repository checks and immutable-evidence validation.

The CUDA suite passed 12/12 tests and the Graph suite passed 3/3 tests.

## Fresh fixed-L campaign

Campaign
`phase3-20260723t042422417332z-7bd6dd48-8a9cb6` preregistered and attempted
all 16 frozen points exactly once. It used new run IDs, performed no selective
rerun, executed no quality or profiler timing lane, and finalized every run.

The campaign result was 16 `aborted`, zero completed, and zero timing-bearing
runs:

- six workers reached a checksum-bound endpoint audit. All six independently
  derived `gqa_nonmaterialization_verified`, identified
  `pytorch_flash::flash_fwd_splitkv` for both GQA and MHA controls, observed no
  preceding replication/copy kernel, and observed no expanded-KV allocation;
- the two eager audits among those six each attributed all 1,066 allocation
  lifetimes under Decision 0013, with no unknown, cache-growth, expanded-KV,
  segment/device-allocation, or persistent-memory-delta failure;
- the four graph audits among those six retained strict zero allocation;
- those six workers then failed because the last measured output checksum did
  not equal the admitted audit-output checksum;
- seven runs finalized with `owned_worker_failure` after the registered worker
  exited before `evidence_flushed`; and
- three runs failed exclusivity when `compute_apps` reported the exact
  registered PID/start-time as `[No data]` while `pmon` omitted its process
  type. The registry still classified the PID as owned, but raw query evidence
  was treated as a hard failure.

The fixed-L campaign independently validates with no errors. All 16 run
directories are checksum-valid and complete, read-only, and have no file newer
than `COMPLETE`. The campaign and source-run checksum manifest has SHA-256
`501832c9af981d812928a7c6f9e82c3635170bea240e344581bb783c7c283438`.

## Stop condition

The fixed-L runner and process-ownership requirements are G1 gates. Their
campaign failures triggered the operator's immediate-stop instruction.
Therefore:

- the growing-context campaign was not started;
- no point was rerun;
- no G1 report bundle was published;
- quality, pilot, full scan, fitting, figures, and Phase 4 remained closed.

## Gate interpretation

- B-011: OPEN. Direct proof passed for six fixed-L operations, but complete
  fixed-L and growing-context campaign coverage does not exist.
- B-012: OPEN. Exact eager attribution and graph-zero-allocation passed for the
  operations that completed audit, but campaign-wide coverage does not exist.
- B-013: OPEN. Registered-worker ownership worked for normal owned rows, but
  the real `[No data]`/missing-`pmon` terminal query race still caused three
  hard failures.
- R-026: OPEN for actual remediation report publication. Publication unit
  controls passed, but the missing growing campaign correctly prevented a new
  G1 report from being created.
- B-014: OPEN. The retained fixed-L callable did not pass its post-audit
  output-equivalence gate in the six runs that reached it.

G1 remains FAIL. No old timing or partial remediation evidence is used to
claim G1 PASS.

## Required next action

1. Determine why the retained fixed-L callable's post-audit output differs
   from its admitted output, using an untimed diagnostic and the already frozen
   numerical tolerances. Do not weaken the numerical contract or alter the
   measured region.
2. Correct the registered-PID query join so an exact PID/start-time-owned child
   is not made unverified solely by the terminal `[No data]`/missing-`pmon`
   race; foreign and PID-reuse detection must remain hard failures.
3. Add targeted fixed eager/graph repeat-equivalence and real query-race
   regressions, then rerun all admission controls.
4. Commit a new execution SHA and, only from a clean tree with every gate
   passing, rerun both complete campaigns under entirely new IDs.

The current fixed-L campaign is immutable failed evidence and must never be
overwritten or selectively retried.

# PHASE 13D REPORT

Status: BLOCKED

## Entry and preregistration

- Starting HEAD: `345d805126587f318e3180ef75356ec38600e9f0`
- Execution HEAD: `a06837a341f58936b21f721af3fe30dce69660d4`
- Working tree at launch: clean
- Source campaign: `phase13-20260822t150835736582z-4ddd7b17-3a8fb3`
- Source root: `feb2e5a8ebba8b729c182fc8170107c9acf8128edd3e5618c8f1b90530557531`
- Source validation and clean-retrieval receipt: PASS
- Densification targets: 25
- Included unique candidates: 84
- Feasible process records: 252
- Capacity-infeasible process records: 0
- Seeds: 20260823, 20260824, 20260825
- Authorized container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`

The deterministic candidate table and complete execution order were committed
before CUDA execution. All 84 exact-batch prefix states completed before the
first timing run.

## Execution blocker

Campaign `phase13d-20260825t030556684636z-a06837a3-83761a` stopped
fail-closed during replicate 0:

- Fully finalized results: 51
- Failed coordinator transition: 1
- Unlaunched scheduled records: 200
- Selective reruns: 0
- Failure point: `kvq3`, B=8, historical L=5120, order index 51
- Failure stage: post-worker GPU-process snapshot during finalization
- Error: `Phase12UnifiedAdmissionError: GPU-process snapshot failed closed`

The affected worker completed model load, exact-batch prefix restoration,
CUDA Graph capture, warmup/allocation audit, and all 128 measured operations.
It entered finalization at `2026-08-25T05:44:09.009291Z`, but the common
post-run process snapshot failed before `result.json` and supervised-command
evidence were written. The failure predicate combines query failure, malformed
process evidence, foreign processes, and unknown processes; the rejected
snapshot payload was not durably recorded, so the exact subpredicate cannot be
reconstructed. After the stop, native `nvidia-smi` returned success, reported
an idle GPU, and the kernel log contained no NVIDIA Xid record.

The append-only evidence remains at:

`artifacts/phase13d/.kvbench-staging/phase13d-20260825t030556684636z-a06837a3-83761a.8f4bb4ac.staging/`

and the 84 prefix states remain at:

`/home/rockrock/phase13d_prefix_states/phase13d-20260825t030556684636z-a06837a3-83761a/`

The staging campaign was not resumed, finalized, assigned a root, or
published. It has no `COMPLETE` marker. The 51 finalized result records all
retain finite output, GPU exclusivity, no fallback, stable allocation and
kernel-path identities, and passing zero-allocation Graph replay evidence.

## Analysis and custody

- New three-process point summaries: 0
- New maximum CV: not estimable
- Combined refits: not run
- Original fit statuses: 30 `knee_observed`
- Refined fit statuses: not evaluated
- Original density sufficient: 5/30
- Original densification required: 25/30
- Resolved Phase 13D targets: 0/25
- R2 root/URI: none; incomplete staging evidence was not published as a
  complete campaign

The successful source campaign, its R2 objects, the first blocked Phase 13
campaign, Phase 13R evidence, adapters, CUDA, cache layouts, timing boundaries,
calibration, fixtures, admission evidence, and Measurement Container remain
unchanged.

## Phase decision

- Phase 13D: BLOCKED
- Phase 14 readiness: NOT READY
- Full Scan: CLOSED
- Quality execution: LOCKED
- `PERFORMANCE_DATA_FROZEN`: absent

The minimum remaining blocker is a separately authorized, evidence-preserving
remediation of the post-worker GPU-process snapshot handoff that records the
rejected snapshot payload while retaining fail-closed foreign/unknown-process
checks. A successor densification campaign must use a new campaign ID and new
run IDs; this staging campaign must never be resumed or selectively repaired.
No speedup, HBM, capacity, final-knee, or quality claim is made.

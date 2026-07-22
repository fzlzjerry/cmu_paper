# Phase 3 BF16 baseline implementation plan

- Status: preregistered
- Date: 2026-07-22
- Entry HEAD: `c16139b0f365eaa052b17cff2fd19c1d4c62a4d1`
- Governing decision: `docs/decisions/0007-phase3-primary-model-and-bf16-backend.md`
- Scope: Phase 3 only; Phase 4 and all formal scans remain closed

## Entry evidence and authority boundary

The entry tree was clean at the exact final Phase 2 HEAD. The complete Phase 2
commit sequence and report were verified. `make test` and `make checks` passed;
all 11 configuration bundles validated; the CLI remained fail-closed; local
append-only lifecycle, command reconstruction, and unique-run protection
passed; both E00 checksum ledgers and manifest hashes were recomputed; the E00
directories are byte-identical to their certified commits; quality execution
is locked; `PERFORMANCE_DATA_FROZEN` is absent; and Full Scan is CLOSED.

G0 remains PASS in native-host scope. B-009 and B-010 remain open. Therefore
Phase 3 output is engineering/admission evidence only and cannot support a
paper performance claim. No Phase 0, Phase 1, or Phase 2 work is restarted.

## Exact model and tokenizer identity

The baseline uses only:

- model/tokenizer repository: `meta-llama/Llama-3.1-8B-Instruct`
- immutable revision: `0e9e39f249a16976918f6564b8830bc894c89659`
- local source after acquisition:
  `/home/rockrock/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`
- loader/reference: Transformers 4.57.6, wheel SHA-256
  `4c9e9de11333ddfe5114bc872c9f370509198acf0b87a832a0ab9458e2bd0550`
- network policy during execution: local files only

Frozen hashes:

| Artifact | SHA-256 |
|---|---|
| `config.json` | `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e` |
| `generation_config.json` | `189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd` |
| `model.safetensors.index.json` | `146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b` |
| `special_tokens_map.json` | `6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec` |
| `tokenizer.json` | `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4` |
| `tokenizer_config.json` | `177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424` |
| weight shard 1 | `2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668` |
| weight shard 2 | `09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15` |
| weight shard 3 | `fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa` |
| weight shard 4 | `92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b` |

The architecture is decoder-only Llama full attention with 32 layers, hidden
size 4096, 32 query heads, 8 KV heads, head dimension 128, BF16 weights, and
131072 maximum positions. RoPE is Llama 3 scaling with factor 8,
high-frequency factor 4, low-frequency factor 1, original context 8192, and
theta 500000. This exact checkpoint and tokenizer are reserved for later
quality validation; no quality benchmark executes in Phase 3.

The acquisition stage must validate every hash before B-004 is marked
resolved. A missing file, access failure, or mismatch records
`model_access_blocked` or `model_identity_unresolved`; no base-model or other
checkpoint fallback is permitted. Weights remain outside Git.

## Dependency isolation

The certified E00 `.venv` remains byte-for-byte unchanged. A separate,
hash-locked Phase 3 dependency target under ignored local state will supply
Transformers, Hugging Face Hub, tokenizers, safetensors, NumPy, and their exact
runtime closure while importing the already certified PyTorch wheel. The lock
and bootstrap/verification command are committed separately from the plan and
from runtime implementation. No quality-only dependency is installed and no
unrelated package is upgraded.

## Exact BF16 attention backend

The SUT calls PyTorch 2.12.1+cu130 bundled FA2 2.5.7 directly through forced
Flash SDPA with `enable_gqa=True`. Only `SDPBackend.FLASH_ATTENTION` is enabled.
The runtime records the PyTorch git SHA, source/binary hashes from Decision
0007, fused backend choice, actual Q/K/V shapes, and any warning or exception.
Unsupported dispatch is `backend_unsupported`; no fallback is attempted.

The SUT never calls Transformers eager attention or its SDPA wrapper. A narrow
Transformers attention-interface registration supplies the direct function to
the loaded Llama layers. The unrecognized custom mask route intentionally
returns no external mask: square prefill invokes full causal Flash attention,
and one-token decode invokes noncausal Flash attention over an already bounded
past/current K/V view. No sliding-window path is allowed. `torch.compile` is
disabled.

## Cache layout and lifecycle

`BF16StaticCache` owns two preallocated tensors:

```text
K: [num_layers, batch, num_kv_heads, capacity, head_dim]
V: [num_layers, batch, num_kv_heads, capacity, head_dim]
dtype: torch.bfloat16
```

The exact tensor-storage formula is:

```text
2 (K,V) * layers * batch * KV_heads * capacity * head_dim * 2 bytes
```

Capacity is `L + 1` for fixed-L and `L + O` for growing-context. No query-head
storage, quantized storage, padding, or implicit resizing is allowed. Padding
bytes are zero for the selected layout. Workspace is measured and reported
separately.

The class provides explicit allocation, deterministic zero/test
initialization, prefill store/update, one-token append, active-length reset,
fixed scratch overwrite, growing append, metadata/accounting, pointer report,
and layout fingerprint operations. Positions are checked before any write.
Finalization forbids resize. Layer views share the backing storage; cache
updates use in-place indexed copies or direct one-position copies. There is no
`torch.cat`, list conversion, host copy, whole-prefix copy, wrap, or truncation
in decode.

The layout fingerprint is SHA-256 over canonical geometry, dtype, strides,
capacity, alignment/padding, workspace declaration, and implementation source
identity. Allocated bytes, tensor bytes, padding bytes, workspace bytes,
maximum context, active context, and storage pointers are exposed.

## Fixed-L semantics

`context_length` means historical prefix length `L`; total attended length is
`L + 1`. Model loading, deterministic token-fixture creation, cache allocation,
and a causal prefix prefill of exactly L tokens occur outside timing. The model
base is used for prefill so a full-vocabulary logit tensor is not created for
every prefix position.

The current token is fixed before measurement. Every operation uses the same
input token, position ID L, cache position L, output shape, and attended length.
It overwrites scratch slot L without advancing active historical length. No
full cache is restored between iterations. A short exact state-drift test
compares all historical K/V bytes before and after repeats. Admission records
pointer and deterministic device-checksum evidence before and after the
measured batch, and an untimed replay produces the output SHA-256.

Excluded from timing: model/tokenizer load, weight transfer, fixture creation,
prefix prefill, cache allocation, backend initialization, compilation/JIT,
autotune, warmup, graph capture, graph correctness replay, sampling, text
decoding, telemetry calls, allocation-stat setup, checksum generation,
serialization, environment collection, and log flushing.

## Growing-context semantics

The runner prefills exactly L tokens outside timing and decodes exactly O=16
pre-generated input tokens. The recorded historical active lengths are
`L, L+1, ..., L+O-1`; the current token is appended at that index and attends
active length plus one. It never samples, performs argmax/top-k, tokenizes, or
decodes text in the measured region.

Capacity is exactly at least `L + O`; over-capacity requests fail before a
write. The timed trajectory uses the preallocated cache without reset or
reallocation. A separate untimed deterministic replay after a fresh prefill
records the full active-length sequence and per-step output checksums, avoiding
per-step synchronization or checksum kernels in the timed trajectory. Raw
step records are preserved.

Growing-context CUDA Graph behavior is intentionally not invented in Phase 3:
the admission runner is eager only. Fixed-shape CUDA Graph support is evaluated
by the fixed-L runner.

## Eager and CUDA Graph execution

Eager invokes the forced backend and loaded model directly. All shapes and
inputs are prepared before timing, and warmup is complete. CUDA Graph uses
static input IDs, position IDs, cache position, cache backing storage, and
output references. Each `(batch,L)` shape is warmed on a side stream and
captured separately after model/backend/cache initialization. Capture time and
correctness synchronization are excluded. Timing includes replay only.

Graph capture failure is finalized as `graph_capture_failed`; replay failure is
`graph_replay_failed`. There is no eager fallback. Eager and graph samples,
summaries, filenames, and stability calculations remain separate.

## Timing boundaries and raw samples

For each fixed-L run:

1. perform 16 untimed warmup operations;
2. take telemetry/allocation observations outside the boundary;
3. synchronize once to establish the start boundary;
4. record a host `time.perf_counter_ns()` start and a CUDA start event;
5. execute exactly 32 eager operations or graph replays without per-step
   synchronization, logging, telemetry, or tensor-to-host conversion;
6. record the CUDA end event;
7. perform one completion synchronization and then record host completion;
8. divide each total by exactly 32 completed operations;
9. retain the batch-level host total, CUDA-event total, operation count, and
   failure count.

Five predeclared measured batches are retained per ordinary point. The full
count runs even if an intermediate batch is slow. Growing-context retains its
single exact O=16 trajectory as one batch sample because resetting/prefilling
inside the boundary would change its semantics. Failed or incomplete
operations are explicit and never omitted. Host-wall is primary; CUDA-event
time is secondary. No speedup is calculated.

Every timing record states:

```text
quality_status: unvalidated
claim_eligibility: performance_only
performance_claim_eligible: false
measurement_scope: native_host_admission
claim_class: none
```

No Phase 3 output is written beneath a paper-results directory.

## Allocation audit

Before measurement the audit records model baseline allocation, cache tensor
bytes, workspace declaration, storage pointers, reserved bytes, setup peak,
and warmup-steady allocation. Peak statistics are reset only outside timing
after setup and immediately before the measured batch; the reset itself is not
timed and does not alter execution.

For eager and graph separately, the audit records allocated/reserved bytes
before and after, peak bytes during the batch, allocator counters, and pointer
stability. It fails on cache resizing, pointer changes, positive persistent
allocated or reserved delta, monotonic repeated growth, or unexplained
workspace. Graph replay additionally requires zero new allocation delta and a
stable peak after capture. Predicted cache bytes must equal storage bytes
exactly. A small declared allocator-accounting tolerance of zero bytes applies
to persistent/cache deltas; any known transient third-party workspace is
reported at its observed byte maximum rather than hidden by a tolerance.

The direct Flash operator inspection observed a bounded transient workspace
risk in eager mode and a batch-4 issue in an optional preallocated-output
variant. The implementation will not use that faulty variant. G1 is not marked
PASS if the end-to-end eager audit cannot reconcile the measured workspace
with the frozen gate.

## GQA non-materialization audit

The audit has five independent parts:

1. source scan of the selected SUT modules for `repeat_kv`,
   `repeat_interleave`, `expand(...).reshape(...)`, `torch.cat`, DynamicCache,
   and full-prefix copies;
2. runtime Q/K/V and cache tensor shapes proving 32/8 head geometry;
3. exact cache-storage byte comparison against both the KV-head and forbidden
   query-head formulas;
4. allocator/operator audit for query-head-sized temporary K/V;
5. a forced-Flash operator control comparing GQA and MHA at the same batch,
   context, head dimension, and dtype, including actual fused backend choice.

Any detected expansion, Transformers fallback, or competing backend dispatch
finalizes the run as `gqa_materialization_detected` or `backend_fallback`.

## Numerical reference strategy

The tolerances in Decision 0007 are fixed before broad execution.

- Small tensors: an explicit FP32 einsum/matmul + causal mask + softmax
  reference, independent of forced Flash, covers GQA, batch 1 and 2, at least
  two context lengths, and boundary positions.
- Static cache: exact write/append/reset/bounds tests; BF16 attention comparison;
  fixed-L output/state stability; growing progression; eager/graph comparison.
- Full model: a short deterministic prefix and several one-token steps compare
  the SUT against Transformers 4.57.6 eager reference with the same exact
  weights, tokenizer, position IDs, RoPE, attention semantics, and cache
  positions. The eager reference may materialize GQA only in this untimed
  correctness control; it is never the measured SUT.

All comparisons record maximum absolute/relative differences, finite checks,
and output/logit SHA-256 values. Tolerances are not changed to obtain a pass.

## Telemetry and exclusivity

The unchanged certified `preflight/process_query.py` is reused as a subprocess
for before/during/after GPU process snapshots. A Phase 3 supervisor preserves
PID and `/proc` start-time identity, worker handshake evidence, stdout/stderr,
and process-group cleanup. The live device must match the certified UUID
`GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b`, full SKU, PCI identity,
capability 12.0, driver 595.71.05, and single-GPU inventory. Foreign or unknown
compute fails closed.

Outside each measured boundary, `/usr/bin/nvidia-smi` supplies timestamp, GPU
name, UUID, power, temperature, SM clock, memory clock, VRAM used, and ECC state
when available. The raw cadence and command result are retained separately
from timing. No NVML/nvidia-smi call occurs inside decode, and no stability
claim is inferred from one snapshot.

## Bounded admission grid

The grid is frozen before model execution.

Fixed-L ordinary points:

```text
batch_size: [1, 4]
context_length: [128, 4096, 16384]  # historical prefix L
graph_mode: [eager, cuda_graph]
warmup_steps: 16
measured_steps_per_batch: 32
measured_batches: 5
ordinary_process_replicates: 1
```

Growing-context ordinary points:

```text
batch_size: [1, 4]
starting_context: [128, 4096]
output_tokens: 16
graph_mode: [eager]
warmup_trajectories: 1
measured_trajectories: 1
ordinary_process_replicates: 1
```

Independent-process stability subset:

```text
runner: fixed_l
batch_size: 1
context_length: 4096
graph_mode: [eager, cuda_graph]
total_process_replicates_per_mode: 3
criterion: CV <= 3%
```

The ordinary grid contains 12 fixed-L and 4 growing points. Four additional
processes complete the two stability points to three total replicates each,
for 20 attempted process runs if capacity and setup succeed. Every configured
point remains in the plan after failure. Memory feasibility is checked from
the exact formula before launch. An infeasible point is recorded as
`capacity_infeasible` and is not replaced. No 32K-131K search point is added.

## CLI, schema, and artifact integration

The existing Phase 2 schema is extended with a distinct
`phase3_admission` plan/run type rather than mislabeling native evidence as
formal timing. Two new plan files independently select fixed-L and
growing-context. Real `kvbench run --plan <plan>` dispatch is admitted only
when method is BF16, the plan kind is Phase 3 admission, the runner and graph
mode are explicit and supported, exact identities resolve, native G0 evidence
still validates, quality stays locked, the freeze marker is absent, and Full
Scan is not requested.

TurboQuant, KIVI, KVQuant, pilot, full-scan, profiler, fit, figures, and quality
remain fail-closed with `phase_not_implemented`. The existing append-only
writer creates a unique run ID and writes beneath `artifacts/phase3`; every
success or failure is finalized with manifest, config, stdout, stderr,
inventory, ledger, and COMPLETE marker. An existing run is never reused.

The Phase 3 manifest freezes model/tokenizer/backend/cache identities, Git SHA
and dirty state, host/hardware/software identity, runner/mode/shape/counts,
seed, exact command, quality/claim fields, and B-010 scope before execution.
Only lifecycle/outcome fields may change at finalization. Command
reconstruction must exactly reproduce the saved argv. Timing, telemetry,
allocation, numerical, GQA, exclusivity, and raw step evidence are separate
machine-readable files. Local finalization does not close B-009.

Required terminal statuses include `model_identity_unresolved`,
`model_access_blocked`, `backend_unsupported`, `backend_fallback`,
`unsupported_geometry`, `numerical_failed`, `allocation_failed`,
`state_drift_detected`, `gqa_materialization_detected`,
`graph_capture_failed`, `graph_replay_failed`, `runtime_failed`,
`capacity_infeasible`, `unstable`, and `aborted`.

## Files planned

Documentation/configuration:

- `docs/decisions/0007-phase3-primary-model-and-bf16-backend.md`
- `docs/plans/phase3-bf16-baseline.md`
- `configs/models/primary_gqa_model.yaml`
- `configs/methods/bf16.yaml`
- `configs/plans/phase3_bf16_fixed_l.yaml`
- `configs/plans/phase3_bf16_growing.yaml`
- separate Phase 3 dependency lock and verification helper

Narrow runtime implementation (names may be combined only where it reduces
duplication without creating a Phase 4 adapter):

- model/tokenizer identity verifier and loader
- forced Flash BF16 attention function/backend verifier
- `src/kvbench/runtime/bf16_static_cache.py`
- `src/kvbench/runtime/fixed_l_runner.py`
- `src/kvbench/runtime/growing_context_runner.py`
- timing, allocation, telemetry, CUDA Graph, GPU-exclusivity, hardware, and
  admission-coordinator modules
- Phase 3 schema additions plus minimal CLI/config/artifact integration

Tests:

- CPU/unit and governance tests under new Phase 3 paths
- new `tests/phase3_cuda`, `tests/phase3_graph`, and
  `tests/phase3_allocation` paths, leaving certified E00 sources untouched
- short exact-checkpoint full-model smoke tests

Records after execution:

- machine-readable BF16 G1 admission report
- `docs/phase_reports/phase3.md`
- updates to status, blockers, risk register, and tasks supported by evidence

## Test and command plan

CPU/governance tests cover cache shapes/bytes/bounds/positions/reset,
fixed-state invariance, growing progression, plan/config/manifest validation,
failure finalization, command reconstruction, exact identity validation,
unique IDs, existing-run protection, locked quality/full scan/quantized methods,
and E00/paper-results preservation.

CUDA tests cover cache allocation/read/write/append, BF16/GQA attention,
forbidden query-head allocation, fixed/growing outputs, eager, graph,
eager/graph agreement, replay allocation, measured-region persistent
allocation, and capacity rejection. Full-model tests cover exact load,
tokenizer identity, short prefill, both runners, both fixed execution modes,
and deterministic checksums.

Commands after implementation include:

```text
make test
make checks
make test-cuda
make test-graph
kvbench run --plan configs/plans/phase3_bf16_fixed_l.yaml
kvbench run --plan configs/plans/phase3_bf16_growing.yaml
```

The pilot, full-scan, profiler-subset, fit, figures, Nsight, and all quality
targets are forbidden. No custom CUDA/C++ extension is planned; consequently
no Phase 3 claim of sanitizer coverage for third-party PyTorch kernels will be
made. If a custom extension becomes necessary, this plan stops and a new
decision plus the full golden/sanitizer/PTX/Graph/allocation protocol is
required before implementation.

## Failure and preservation policy

Every failure receives a new immutable run ID and preserves initial/final
manifest, resolved config, stdout, stderr, and machine-readable evidence. No
failed point is removed or selectively rerun. A code fix creates a new commit
and new run ID; it never appends to a finalized run. Every exclusion has a
machine-readable reason.

The original failed and successful E00 evidence, their ledgers, manifest
hashes, COMPLETE markers, Phase 2 artifacts, and completed Phase 3 runs are
read-only. Tests write only to temporary stores. No model weight, token,
framework cache, formal raw benchmark table, profiler timing, or quality data
is committed.

## Intentionally deferred Phase 4 and later work

The plan does not create the general `KVCacheMethod` protocol, method adapters,
TurboQuant, KIVI, KVQuant, method reference lanes, calibration, pilot/full
scan, profiler integration, response/knee fitting, capacity amplification,
continuous batching, multi-GPU execution, paper figures, or quality execution.
Growing-context dynamic/bucketed graph semantics are also deferred. Phase 4
does not begin automatically after this report.

## Gate disposition rule

Only G1 is evaluated. G1 passes only when every condition in the operator's
Phase 3 instruction and Decision 0007 passes with immutable checksum-valid
evidence. G0 remains PASS; G2-G5 remain NOT EVALUATED; Full Scan remains CLOSED;
quality remains LOCKED; `PERFORMANCE_DATA_FROZEN` remains absent; B-009 and
B-010 remain OPEN. Any unresolved identity, backend, numerical, allocation,
GQA, runner, graph, checksum, replicate, stability, fallback, substitution, or
artifact condition makes Phase 3 PARTIAL, BLOCKED, or FAIL rather than being
silently relaxed.

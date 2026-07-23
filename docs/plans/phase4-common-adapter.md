# Phase 4 common adapter plan

Status: implementation plan only. Phase 5, performance scans, profiling, and
quality execution remain closed.

## Reuse unchanged

- `BF16StaticCache` remains the sole BF16 cache implementation.
- `BF16DecodeEndpoint` retains model execution, RoPE, output, and timing-boundary
  semantics; it delegates only cache and attention operations to the adapter.
- The fixed-L and growing-context lifecycle, timing helpers, CUDA Graph capture,
  process supervision, artifact lifecycle, and Phase 3 trace parsers remain
  unchanged.
- Existing numerical, allocation-attribution, backend, and GQA audit primitives
  remain authoritative.

## Minimal interface and factory

Add a small `KVCacheMethod` protocol with `allocate`, `store_prefill`,
`append_decode`, `decode_attention`, `allocated_bytes`, `byte_breakdown`,
`logical_bf16_bytes`, `config_fingerprint`, and `supports_cuda_graph`.

Add one `BF16MethodAdapter` that wraps `BF16StaticCache` and
`flash_attention_forward`. Add an explicit immutable factory mapping containing
only `bf16`. Known quantized methods fail with `phase_not_implemented`; unknown
methods are rejected. There is no discovery, registration, or plugin API.

## Integration

- Construct the adapter from the existing resolved method config before the
  Phase 3 endpoint session is prepared.
- Route endpoint prefill writes, decode appends, and Flash attention through the
  adapter without moving the measured boundary.
- Expose adapter accounting, layout, pointer, geometry, and fingerprint through
  the existing endpoint session. Fixed-L and growing-context runners consume
  those common facades and contain no method-specific branch.
- Add the deterministic adapter fingerprint to newly created manifests and
  runtime evidence. Historical manifests remain parseable and immutable.

## Shared validation and report

Add thin common harness functions that compose the existing full-model
correctness, allocation-attribution, graph, backend, cache-geometry, and GQA
audits. They do not collect a second trace or define new admission criteria.

Add a strict compact `MethodAdmissionReport` schema containing method/model/
backend identity, adapter and cache fingerprints, four validation results,
evidence references, G0-G5 and Full Scan state, blockers, claim/quality state,
and creation Git SHA. Raw evidence remains external and checksum referenced.

## Tests and bounded evidence

Add focused unit, runner, schema, CUDA, graph, and governance tests using
existing fixtures. Run at most three untimed functional smokes at B=1, L=128:
fixed-L eager, fixed-L graph, and growing-context eager with O=4. Smoke records
contain no latency samples and retain native-host non-claim governance fields.

## Explicitly deferred

TurboQuant, KIVI, KVQuant, method-specific reference lanes, calibration,
quantized storage, pilot/full scans, profiler integration, quality evaluation,
and all performance/compression/knee claims are Phase 5 or later work.

# Decision 0036 — KVQuant q4 Value-decode workspace geometry

- Status: Accepted
- Date: 2026-08-18
- Scope: KVQuant q4 caller-owned deterministic Value-decode workspace only

## Context

The Decision 0027 deterministic q4 Value-decode API consumes one FP32 partial
per 128-token quantized-Value tile.  The admitted cache allocated 32 tiles,
which is exact only for the historical \`L=4096\` admission geometry.  The
immutable stopped Pilot campaign
\`phase13-20260804t111810342595z-a127b0d1-8649c3\` therefore failed closed at
\`kvq4\`, \`B=8\`, \`L=16384\` before graph capture.

## Decision

For a cache with declared maximum total-attended capacity \`C\`, the q4 cache
state allocates exactly once, during construction:

\`\`\`
quantized_value_capacity = max(0, C - sink_tokens)
q4_tile_capacity = ceil_div(quantized_value_capacity, 128)
workspace_shape = [batch, query_heads, q4_tile_capacity, head_dim]
\`\`\`

The frozen values remain \`sink_tokens=5\`, \`query_heads=32\`, \`head_dim=128\`, and
workspace dtype \`float32\`.  Allocation uses declared capacity, never active
context, and the workspace is neither resized nor replaced during prefill,
warmup, graph capture, or measured decode.  Its physical bytes are
\`batch * query_heads * q4_tile_capacity * head_dim * 4\` and are recorded as
persistent workspace, not cache payload.

The formula identifier is
\`kvbench-kvquant-q4-value-workspace-capacity-v1\`.

## Preservation

This correction changes no CUDA source, q4 numerical reduction, quantization,
packing, metadata, sparse selection, sink semantics, calibration, fixture, or
GQA mapping.  KVQuant q3/q2 behavior and their admitted fingerprints remain
unchanged.  The old Phase 11R and Phase 12 evidence and every stopped Phase 13
campaign remain immutable.  A successor q4 admission and a minimal q4-only G5
refresh are required before Pilot readiness is restored.

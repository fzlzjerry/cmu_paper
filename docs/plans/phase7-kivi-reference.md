# Phase 7 KIVI reference plan

Status: BLOCKED during source audit on 2026-07-26.

## Authority and source

Use only the author-maintained repository
`https://github.com/jy-yuan/KIVI.git` at commit
`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`, tree
`c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b`. Decision 0017 selects the
default `main` implementation for this audit because it is the official branch
that advertises Llama 3 and GQA support. The `develop` and `lmeval` branch
heads are recorded but are not substitutes.

Relevant source is limited to the root license, README, dependency/build
files, Llama and Mistral integrations, quantization/packing modules, and the
`kivi_gemv` CUDA/C++ extension. `third_party/LOCK.json` binds each relevant
file by Git blob and SHA-256.

## Intended reference lane

If the source audit passed, one isolated, digest-pinned reference container
would build the official extension through `quant/setup.py` with explicit
native `sm_120` and `compute_120` targets. It would not modify or install into
the authorized Measurement Container.

The frozen configuration mapping is direct:

- `k4v4`: `k_bits=4`, `v_bits=4`;
- `k2v4`: `k_bits=2`, `v_bits=4`;
- `k2v2`: `k_bits=2`, `v_bits=2`;
- `k4v2`: `k_bits=4`, `v_bits=2`, held-out asymmetry control;
- `group_size=32`, `residual_length=32`.

The planned fixture geometry is batch 1, 32 query heads, 8 KV heads, head
dimension 128, BF16 inputs, and seed 20260726. The basic store/append/decode
case is L=17 to L=18. Source inspection selects L=31, L=32, and L=33 for the
first rollover: the key path moves a full 32-token residual block at the
boundary, while the value path moves the oldest token after its residual
length exceeds 32.

Byte accounting would derive payload, scale/minimum metadata, residual,
padding, workspace, and total allocation directly from source-owned tensors.
Only a finalized COMPLETE-last fixture bundle would be published through the
existing R2 tool and cleanly retrieved.

## Hard stop

The selected source fails the mandatory GQA proof before environment or CUDA
work. Its advertised Llama GQA decode path calls the exact Transformers
`repeat_kv` helper for recent key/value regions. That helper expands 8 KV
heads by four and reshapes to a separately stored 32-head tensor. This violates
the Phase 7 prohibition on `repeat_kv` and any H_Q-sized K/V temporary.

Therefore no reference container, extension build, fixture generator,
fixture validator, sanitizer run, trace, graph smoke, byte-layout fixture, or
R2 fixture publication is authorized in this attempt. Phase 8 and the KIVI
Measurement Adapter remain unstarted and fail-closed.

Minimum remediation is an author-maintained source revision whose official
Llama GQA path consumes eight-head residual and quantized storage without
materialization. A new task must restart Phase 7 entry and source audit against
that exact revision.

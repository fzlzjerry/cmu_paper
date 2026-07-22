# Phase 3 BF16 backend identity evidence

- Recorded: 2026-07-22
- Evidence class: bounded operator-level engineering control
- Measurement timing recorded: no
- Performance-claim eligible: no
- Measurement scope: native_host_admission
- GPU UUID: GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b
- Certified pre-control process query: no foreign or unknown compute processes

## Selected implementation

The selected SUT operator is direct
torch.nn.functional.scaled_dot_product_attention under an enclosing
sdpa_kernel(SDPBackend.FLASH_ATTENTION) context, with enable_gqa=True,
dropout zero, and explicit scale 128**-0.5. No other SDPA backend is enabled.
Transformers is the loader and independent reference, not the SUT wrapper.

| Identity field | Frozen value |
|---|---|
| PyTorch | 2.12.1+cu130 |
| PyTorch Git SHA | 7269437d655783a26cba32aa88195b741ff496aa |
| CUDA runtime | 13.0 |
| cuDNN | 9.20.0 |
| bundled FlashAttention | FA2 2.5.7 |
| Triton environment | 3.7.1 (not the selected backend) |
| torch/nn/functional.py | 27493186ee22f811b553e31d9c804d4d46716d1be62d034d731537f66f27ef19 |
| torch/nn/attention/__init__.py | 56e10b6f965cc050db782dd4dc472097c9b02ec5b5fe3ab2c8b04055c0b0bbe0 |
| torch/nn/attention/varlen.py | 2f5384e0bc8ce371d00a1c09d38ad019517009798e7cb3434f56cf4b9fa351ea |
| flash_api.h | 1474aa79d8aa6ce39984dbc3c0aad9dba283ab819f034370e5cfb70980524ee7 |
| libtorch_cuda.so | b248fb7e9935440965e4736eea48868b315ba41012734b7ce058fc0a2d0b1984 |

The bundled flash_api.h declares Q with num_heads and K/V independently with
num_heads_k for forward, varlen-forward, and backward APIs. This is direct
source evidence that the selected kernel interface accepts KV-head geometry.

## Forced-dispatch controls

All BF16 SM120 controls returned fused backend choice 1, the installed enum
value for FLASH_ATTENTION.

| Lane | Batch | Q shape | K/V shape | Output shape | Finite |
|---|---:|---|---|---|---|
| causal prefill GQA | 1 | [1,32,128,128] | [1,8,128,128] | [1,32,128,128] | yes |
| causal prefill GQA | 4 | [4,32,128,128] | [4,8,128,128] | [4,32,128,128] | yes |
| one-token decode GQA | 1 | [1,32,1,128] | [1,8,128,128] | [1,32,1,128] | yes |
| one-token decode GQA | 4 | [4,32,1,128] | [4,8,128,128] | [4,32,1,128] | yes |
| one-token MHA control | 1 | [1,32,1,128] | [1,32,128,128] | [1,32,1,128] | yes |
| one-token MHA control | 4 | [4,32,1,128] | [4,32,128,128] | [4,32,1,128] | yes |

The Python dispatch boundary reported aten.scaled_dot_product_attention.default;
the fused choice is the lower-level selection evidence. Admission execution
must repeat the choice and shape check for every actual geometry and lane and
fail closed on a warning, competing choice, or unsupported dispatch.

## CUDA Graph control

A fixed-shape batch-1 decode operator captured successfully with stable
pointers. Capture-time output is not a correctness result. The first replay
was bit-exact to eager forced Flash, and consecutive replays were bit-exact:

    eager/replay SHA-256:
    1fb63e2f1a8f892c7eb0335526ba344024b30293ff53a509bbbab06453629a4a
    replay 1/replay 2 maximum absolute difference: 0

This is operator-level support evidence only. Full-model graph correctness and
allocation remain admission requirements.

## Allocation control

A separate instrumented control ran after warmup and recorded no duration. It
is ineligible for normal timing. One eager forced-Flash decode issued four
caching-allocator requests totaling 9,728 cumulative allocated bytes, with an
8,192-byte persistent output delta. One warmed graph replay issued zero
allocation requests and zero cumulative/current/reservation/device-allocation
deltas.

The eager requests are direct evidence of the frozen G1 risk: stable reserved
memory or a later free cannot convert them into a pass. Their size is far below
a query-head-expanded K/V pair for this geometry, but that observation is not
used alone to prove non-materialization.

## Rejected source paths

Transformers 4.57.6 source hashes:

| Source | SHA-256 | Rejection evidence |
|---|---|---|
| models/llama/modeling_llama.py | 31bf660a663259134324bc65da4e155951dc89c5ca46471d2325a9938e859e26 | rotate_half uses torch.cat; eager attention calls repeat_kv |
| integrations/sdpa_attention.py | dc5abe49a98dec3b9026739dfbf2e9a8f3e5272b2916b3c2d404727ac931a013 | conditional repeat_kv uses expand |
| integrations/flex_attention.py | e5201fe312ad19142f7d14cf3e256b2c82f45ae14e733fa2e47e8a73bbcf4158 | repeat_kv uses expand |

These paths are permitted only for untimed correctness reference work. Neither
is permitted in the measured SUT. No custom CUDA/C++ extension was introduced,
so this record does not claim Compute Sanitizer coverage for PyTorch's
third-party kernel.

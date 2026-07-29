# Phase 10 — KVQuant Reference Lane Plan

Status: frozen for the Phase 10R rerun.

## Authority and environment

- Source authority is Decision 0021: upstream base
  `57a238357f0ffe50084670fcd5781c9848f80ea2` /
  `094e0f736f77ee327e5350cbd1eefb1c936aa77b`, patch SHA-256
  `db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6`,
  and patched commit/tree `4ad80bc8c942d0a05516d2be8f8d443a77a05900` /
  `c4f1490c9c0c4ec46099f1e95c092516df2adb4e`.
- Calibration authority is
  `kvqcal-cdb724c806d64d095c040d2673a987a3`, root
  `8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`,
  mounted read-only. Fisher, quantizers, and calibration metadata are never
  regenerated or rewritten.
- The exact Phase 9 calibration image
  `sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d`
  contains modern `tokenizers==0.22.2`, while the authorized vendored
  Transformers reference source requires `tokenizers==0.15.2`. Use one thin
  `docker/reference-kvquant.Dockerfile` derived from that exact image. It adds
  only the checksum-locked `tokenizers==0.15.2` wheel in an isolated dependency
  directory, without modifying the Phase 9 image or its installed packages.
  Source, calibration, and output mounts are separated; credentials are
  host-only.

## Frozen fixture contract

- Geometry: batch 1, 32 query heads, 8 KV heads, 4 groups, head dimension 128,
  BF16 interface, seed `20260729`, store positions 0–16, append position 17,
  and total context 18.
- Positions 0–4 are full-precision sink K/V. Positions 5–16 are stored
  non-sink history; position 17 is appended.
- For each of `kvq4`, `kvq3`, and `kvq2`, generate exactly:
  `key_zero_value_fixed12`, `key_few_value_fixed12`, and
  `key_cap_value_fixed12`.
- Key construction uses each frozen coordinate threshold. Midpoints produce
  count 0; three deterministic lower and three upper normalized excursions
  produce count 6; six lower and six upper excursions produce count 12.
- Every non-sink Value row uses source-defined `fixed_extrema` selection:
  six lowest then six highest entries, active count 12. Sink rows have count 0.
  An equal-value control freezes stable tie ordering without adding a fourth
  full fixture.

## Semantics and outputs

- Preserve K immediately after `k_proj` and before RoPE, V immediately after
  `v_proj`, packed pre-RoPE K, native Llama-3.1 attention-ready K, position
  IDs, and the RoPE fingerprint as distinct fields.
- Retain native H_KV=8 dense, sparse, and sink storage. Query mapping is
  `kv_head = query_head // 4`; implementation-side `repeat_kv`,
  `repeat_interleave`, and query-head-sized K/V storage are forbidden.
- Each fixture records deterministic inputs, dense K/V payloads and metadata,
  fixed-cap sparse values/indices/counts, sink K/V, cache after store and
  append, appended slot, reference decode output, an independent explicit
  reconstruction control, finite-value verdicts, manifests, and checksums.
- Byte accounting separates dense K/V, K/V metadata, K/V sparse values and
  indices, sink K/V, padding, and reference workspace. Physical sparse
  allocation always includes 12 slots; active logical bytes vary only for Key.
  Record `rho_alloc`, reciprocal `r_alloc`, and null `r_hbm`.

## Execution and custody

- Use the patched 4/3/2-bit pack/decode paths. Record dynamic reference
  allocation and any dense temporary or fallback without optimizing it.
- Collect only `reference_trace` evidence needed to cover 4-bit, 2-bit,
  distinct 3-bit behavior, append, sink, sparse correction, decode, and GQA.
  Discard duration fields.
- Build the exact extension for native SM120 and required compute_120 PTX.
  Run forced PTX/JIT and the minimum Compute Sanitizer cases covering dense,
  few-sparse, cap-sparse, 2-bit, 4-bit, and distinct 3-bit paths.
- Finalize the compact bundle append-only under `reference/kvquant/fixtures`
  with exact inventory, checksum ledger, and `COMPLETE` last. Refuse differing
  overwrite.
- Publish from the host through the existing R2 client using conditional
  content-addressed writes, then clean-retrieve and validate the complete root.

Phase 11, the KVQuant Measurement Adapter, G2-KVQ, performance measurement,
profiling, fitting, figures, and quality evaluation remain deferred.

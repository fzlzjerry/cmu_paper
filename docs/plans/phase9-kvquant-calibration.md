# Phase 9 — KVQuant Calibration Plan

Status: entry checks passed; calibration not yet executed.

## Frozen authority

- Method identifier: `kvquant_gqa_upstream_patch_v1` (“KVQuant-GQA patched
  upstream”).
- Upstream: `https://github.com/SqueezeAILab/KVQuant.git`, base commit
  `57a238357f0ffe50084670fcd5781c9848f80ea2`, base tree
  `094e0f736f77ee327e5350cbd1eefb1c936aa77b`.
- Decision 0021 patch SHA-256:
  `db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6`;
  patched commit `4ad80bc8c942d0a05516d2be8f8d443a77a05900`; patched tree
  `c4f1490c9c0c4ec46099f1e95c092516df2adb4e`.
- Reconstruction/validation command:
  `make KVQUANT_GQA_SOURCE_ROOT=/home/rockrock/third_party_worktrees/kvquant-gqa validate-kvquant-gqa-patch`.
  Calibration must stop if this identity or the clean patched tree changes.

## Model, container, and data

- Model and tokenizer: `meta-llama/Llama-3.1-8B-Instruct` at revision
  `0e9e39f249a16976918f6564b8830bc894c89659`, from the checksum-bound offline
  snapshot. Model forward is BF16; geometry is 32 layers, 32 query heads,
  8 KV heads, head dimension 128, and GQA group size 4. Keys are captured
  immediately after `k_proj` and before RoPE; Values remain native H_KV=8.
- Container: `docker/calibration-kvquant.Dockerfile`, derived from immutable
  Phase 9P image
  `kvbench-phase9p-validation@sha256:16ee632c5ac029deca5859f3da4c74f9e5e55f5080c10745a59653e95d8e5b44`.
  It retains Python 3.12.3, PyTorch 2.12.1+cu130, CUDA userspace 13.0.2,
  Transformers 4.57.6, Accelerate 1.10.1, NumPy 2.5.1, SciPy 1.16.1, and
  scikit-learn 1.7.1, and adds only a hash-locked `datasets==5.0.0` parquet
  stack. The Dockerfile SHA-256 and built image config digest are frozen before
  calibration. Source and model snapshots are mounted read-only; credentials,
  caches, model weights, source, and outputs are not copied into the image.
- Dataset: public `Salesforce/wikitext`, upstream dataset revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`, config
  `wikitext-2-raw-v1`, split `train` only. The sole input object is train
  parquet from immutable conversion commit
  `3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e`, size 6,357,543 bytes,
  SHA-256 `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`.
- Selection: join train rows in source order with `"\n\n"`, tokenize once with
  the frozen fast tokenizer and its default special-token behavior (one BOS,
  no EOS), with no padding or truncation. A local `random.Random(20260721)`
  draws 16 ordered inclusive start offsets with
  `randint(0, token_count - 2048 - 1)`; each exact 2,048-token window, its
  global offsets, and overlapping train row IDs are recorded. Test and
  validation splits are never requested.

## Calibration outputs

- Precision and randomness: BF16 model forward, FP32 Fisher/gradient
  accumulation, FP16 fitting activations, and FP32 thresholds/codebooks.
  Python, NumPy, PyTorch CPU/CUDA, and clustering use base seed `20260721`;
  one thread, deterministic algorithms, disabled TF32, model `eval`, and
  scikit-learn `KMeans(n_init="auto", max_iter=50)` are frozen before execution.
- Fisher: one safe `fisher.safetensors` containing exactly 64 FP32 tensors:
  32 `k_proj` tensors for pre-RoPE Key and 32 `v_proj` tensors for Value. Each
  tensor has shape `[1, 32768, 1024]`; the manifest records per-tensor shape,
  dtype, finite-value counts, and checksum.
- Quantizers: exactly `kvq4`, `kvq3`, and `kvq2`, all linked to the same tokens
  and Fisher root. Trusted source-native pickle is converted once inside the
  calibration container to explicit safe tensors containing upper/lower
  thresholds and FP32 NUQ codebooks; only the safe form is published.
- Shared policies: `sink_tokens=5`; full-precision sink dtype FP16; sink rows
  are excluded from dense history and sparse selection. Key and Value use a
  fixed capacity of 12 native-width entries (six lower, six upper), float32
  values, int32 indices, lexicographic stable ties, no overlap, and zero-filled
  unused slots.
- `layer_stats.parquet` has exactly 192 rows
  (32 layers × K/V × 4/3/2 bits), including Fisher/codebook checksums,
  threshold summaries, outlier-count distributions, cap hits, clipping or
  saturation counts, sink policy, dtypes, and NaN/Inf counts.

## Reproducibility, bundle, and publication

- Rebuild the 16×2048 token tensor exactly; regenerate all three safe
  quantizers in fresh processes; replay one K and one V Fisher layer; and
  verify deterministic tied outliers and capacity 12. Exact byte identity is
  required for canonical safe serialization; frozen numerical tolerances are
  used only where exact bytes are not promised.
- Derive `calibration_id` from the canonical contract plus authority and built
  image fingerprints. Finalize once under
  `calibration/kvquant/<calibration_id>/` through the existing append-only,
  no-replace, checksum/inventory, `COMPLETE`-last lifecycle. Large tensors stay
  outside normal Git history.
- Publish the finalized root through the existing Cloudflare R2 client using
  conditional content-addressed writes and `COMPLETE` last. Retrieve into a
  new empty directory and revalidate every object and root digest before
  recording the external receipt.

Phase 10 reference fixtures, the KVQuant Measurement Adapter, G2-KVQ, Pilot,
Full Scan, profiling, performance fitting, PPL, LongBench, and all quality
evaluation remain deferred.

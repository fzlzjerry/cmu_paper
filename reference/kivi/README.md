# KIVI Phase 7 reference lane

This directory is one source-authoritative correctness lane for KIVI. It is
not a Measurement Adapter or a general reference framework.

Authority is the official `jy-yuan/KIVI` commit
`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6` plus the single
checksum-bound Decision 0018 patch. The resulting tree is
`b617493dea5aff1a754cd27ad6be12ac512b2aee`.

`generate_fixtures.py` invokes the pinned upstream quantize/pack and CUDA GEMV
operations and the patched native-GQA helpers. `validate_fixtures.py` validates
the frozen files without regenerating them. Fixture inputs are BF16 values
chosen from an exactly FP16-representable grid; the generator proves an exact
BF16 to FP16 to BF16 round trip before entering the upstream half-only CUDA
ABI.

Use:

```text
make reference-kivi
make validate-reference-kivi
```

Both commands are correctness-only. They create no latency, throughput,
physical-HBM, capacity, or quality evidence. R2 publication is host-side and
uses the repository's existing content-addressed artifact tool only after the
local bundle is complete.

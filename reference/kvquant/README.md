# KVQuant reference fixtures

This directory is the narrow Phase 10 KVQuant numerical Reference Lane. It is
bound to Decision 0021's patched-source identity, Decision 0023's
source-faithful sparse contract, and calibration
`kvqcal-cdb724c806d64d095c040d2673a987a3` at root
`8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`.

One thin reference image derives from the exact Phase 9 calibration image. It
adds only checksum-locked `tokenizers==0.15.2` in an isolated dependency
directory because the authorized vendored Transformers source cannot execute
against the calibration image's modern `tokenizers==0.22.2`. The Phase 9 image
and its installed packages are unchanged. The patched checkout and calibration
bundle are mounted read-only; the generated fixture destination is the only
writable mount. No model weights, calibration payload, reconstructed checkout,
credentials, executable pickle, performance result, or quality result is
stored here.

The fixture matrix is exactly three bit widths by three cases:

- `key_zero_value_fixed12`
- `key_few_value_fixed12`
- `key_cap_value_fixed12`

Key active counts are 0, 6, and 12. Every finite non-sink Value row contains
six lowest plus six highest entries, for a fixed active count of 12. Sink Value
rows contain zero sparse entries. The ambiguous legacy no/few/cap names are
not aliases.

Run `make reference-kvquant` to verify authority and calibration, verify or
build the exact SM120 extension, run native and forced-PTX controls, run the
minimal Compute Sanitizer matrix, generate the immutable nine-fixture bundle,
validate it, and compare it with a second deterministic generation. If the
bundle already exists, the target validates it and never replaces it.

Run `make validate-reference-kvquant` to validate the finalized fixtures
without regeneration. The validator checks exact paths, inventories,
checksums, source/calibration fingerprints, dense and sparse payloads,
pre-/post-RoPE semantics, native 32Q/8KV GQA storage, store/append/decode
outputs, independent numerical controls, and byte accounting.

The 3-bit Value parallel-store payload is preserved exactly as produced by the
frozen patched source and is validated by decode-and-repack round trip. This is
recorded as an observed source-path property; the reference lane does not
change or conceal it.

These fixtures do not implement the KVQuant Measurement Adapter and do not
evaluate G2-KVQ, performance, HBM traffic, capacity, or quality.

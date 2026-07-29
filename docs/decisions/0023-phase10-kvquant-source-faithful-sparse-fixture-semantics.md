# Decision 0023: Phase 10 KVQuant source-faithful sparse fixture semantics

- Status: Accepted
- Date: 2026-07-29
- Authority: explicit operator correction after the immutable Phase 10 BLOCKED report
- Superseded by: none

## Context

The original Phase 10 fixture contract required both Key and Value sparse
occupancy to cover zero, below-cap, and cap-reached states. The frozen patched
deployment instead calls `select_fixed_outliers` for Value without thresholds.
Every finite non-sink Value row therefore selects six lowest plus six highest
entries, while sink rows select none.

An exact-image, read-only probe of the frozen Key thresholds confirmed that the
unchanged Key path can deterministically produce counts 0, 6, and 12 for each
of `kvq4`, `kvq3`, and `kvq2`.

## Decision

1. Do not change the frozen Value-selection implementation.
2. Treat the original Phase 10 no/few/cap language as over-specified because it
   implicitly required variable occupancy for both Key and Value.
3. Key exercises source-faithful sparse counts 0, 6, and 12. The below-cap
   count is frozen at 6.
4. Every non-sink Value row uses fixed capacity 12 with six lowest entries
   followed by six highest entries. Every sink Value row has active count zero
   and zero-filled sparse storage.
5. Generate exactly this matrix:

   - `kvq4/key_zero_value_fixed12`
   - `kvq4/key_few_value_fixed12`
   - `kvq4/key_cap_value_fixed12`
   - `kvq3/key_zero_value_fixed12`
   - `kvq3/key_few_value_fixed12`
   - `kvq3/key_cap_value_fixed12`
   - `kvq2/key_zero_value_fixed12`
   - `kvq2/key_few_value_fixed12`
   - `kvq2/key_cap_value_fixed12`

6. This is a fixture-contract correction, not an algorithm change.
7. Decision 0021, patch SHA-256
   `db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6`,
   patched commit `4ad80bc8c942d0a05516d2be8f8d443a77a05900`, patched tree
   `c4f1490c9c0c4ec46099f1e95c092516df2adb4e`, and Phase 9 calibration root
   `8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf`
   remain unchanged.
8. Report fixed Value sparse allocation as a method property and represent it
   explicitly in byte accounting.
9. Do not claim that Value sparse occupancy is data-dependent in the admitted
   deployment path.
10. Preserve the prior Phase 10 BLOCKED report byte-for-byte under its custody
    record.

## Consequences

Phase 10 may resume with nine source-faithful fixtures. Phase 11, the KVQuant
Measurement Adapter, admission, performance, profiling, and quality work remain
outside this decision.

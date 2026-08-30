# Phase 16G — B=2/B=16 Geometry Admission

- Status: PASS
- Execution HEAD: `6c829edad5cd0401051a20ee42fc61eeb612a9ba`
- Authorized container: `sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e`
- Prefix format: `kvbench-phase16g-prefix-state-3.0.0`
- Geometry authority: Decision 0039
- Admitted Full Scan batches: `{1,2,4,8,16}`

The exact-shape prefix schema accepts positive integer batches while the
successor execution index separately admits only the five Full Scan batches.
Legacy B=1/4/8 artifacts and geometry records remain unchanged and readable.

All 40 B=2/B=16 L=4096 eager/Graph admissions passed numerical, finite-output,
prefix restoration, allocation, native-GQA/path, pointer-stability, and CUDA
Graph checks. All 20 exact-batch prefix cases passed without broadcasting,
repeat, or truncation. Selected custom-kernel memcheck/initcheck coverage passed.

## B=16 maximum-feasible Graph smokes

- `bf16`: L=16384 Graph PASS
- `tq_4bit_nc`: L=24576 Graph PASS
- `tq_k3v4_nc`: L=24576 Graph PASS
- `tq_3bit_nc`: L=24576 Graph PASS
- `k4v4`: L=24576 Graph PASS
- `k2v4`: L=24576 Graph PASS
- `k2v2`: L=24576 Graph PASS
- `kvq4`: L=24576 Graph PASS
- `kvq3`: L=24576 Graph PASS
- `kvq2`: L=24576 Graph PASS

G0–G5 remain PASS. Full Scan is READY but was not started. Quality remains
LOCKED. No timing, profiling, quality execution, speedup, or HBM claim was made.
Durable publication identity is recorded separately in the checksum-bound
Phase 16G R2 receipt after this report is sealed.

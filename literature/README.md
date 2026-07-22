# Literature input provenance

Status: audited, checksummed, and locally write-protected for Phase 0 on
2026-07-21T17:13:29Z.

## Source

- Operator-provided archive: /home/rockrock/cmu_paper/Archive.zip
- Archive size: 85,557,591 bytes
- Archive SHA-256: 20e5b6be5c3060012c48446d1b51067996cd4f13df1d6a73ee8eeb8f855e3ab1
- Acquisition URL and pre-workspace history: not provided; tracked in docs/blockers.md.
- Extracted destination: literature/raw/
- Extraction policy: unzip with no-overwrite after central-directory path/type review.
- Local protection: Archive.zip and all files/directories below literature/raw/
  have write bits removed.

This chmod state is not filesystem immutability and can be reversed by a
privileged operator. The checksum and manifest records also remain uncommitted

The archive contains 46 regular, unencrypted entries: 23 research PDFs and
23 matching macOS AppleDouble sidecars. There are no absolute paths, parent
traversal components, duplicate names, symlinks, special files, executable
permission bits, executable filename types, or Office macro formats. The ZIP
integrity test passed.

The PDF checks were static and non-interactive. pdfinfo reported JavaScript=no
and encrypted=no for every paper, and pdfdetach reported zero embedded files
for every paper. qpdf was not installed, so a qpdf structural check was not
available. PDFs remain untrusted binary documents; they were parsed only with
command-line metadata/text extractors, with no archive content executed.

AppleDouble sidecars are retained and hashed as raw provenance but excluded
from paper parsing and scientific interpretation.

## Paper inventory

| arXiv version | Title | Project role |
|---|---|---|
| 2306.14048v3 | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models | related work |
| 2309.17453v4 | Efficient Streaming Language Models with Attention Sinks | related work |
| 2310.07240v6 | CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving | related work |
| 2401.18079v6 | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization | primary method |
| 2402.02750v2 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache | primary method |
| 2402.18096v1 | No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization | related work |
| 2403.05527v4 | GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM | related work |
| 2404.00456v2 | QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs | related work |
| 2404.14469v2 | SnapKV: LLM Knows What You are Looking for Before Generation | related work |
| 2405.06219v3 | SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models | related work |
| 2405.14256v1 | ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification | related work |
| 2406.02069v4 | PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling | related work |
| 2406.03482v2 | QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead | related work |
| 2407.21118v2 | Palu: KV-Cache Compression with Low-Rank Projection | related work |
| 2410.10819v1 | DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads | related work |
| 2503.24000v1 | Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving | survey/evaluation |
| 2503.24358v2 | SQuat: Subspace-orthogonal KV Cache Quantization | related work |
| 2504.19874v1 | TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate | primary method |
| 2507.08143v2 | Compactor: Calibrated KV Cache Compression with Approximate Leverage Scores | related work |
| 2510.00636v1 | Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution | related work |
| 2602.23200v2 | InnerQ: Hardware-Aware Tuning-Free Quantization of KV Cache for Large Language Models | related work |
| 2605.13734v1 | KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving | related work |
| 2605.13810v1 | Provable Quantization with Randomized Hadamard Transform | related work |

## Verification

From the repository root:

    sha256sum -c literature/checksums.sha256

The machine-readable inventory is literature/manifest.csv. Do not modify files
under literature/raw/. A changed or additional input must be appended as a new
provenance event and must not overwrite the current files.

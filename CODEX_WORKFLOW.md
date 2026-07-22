# Codex 主工作流提示词：RTX PRO 6000 上的 KV Cache Quantization 定量研究

> 建议文件名：`CODEX_WORKFLOW.md`  
> 研究对象：TurboQuant-vLLM、KIVI、KVQuant  
> 目标硬件：单张 NVIDIA RTX PRO 6000 Blackwell 96GB  
> 目标产物：可运行代码、可复现实验、原始数据、机制分析、每种方法的定量性能模型

---

## 使用方式

1. 将本文件复制到项目仓库根目录，命名为 `CODEX_WORKFLOW.md`。
2. 将论文资料压缩包放在 Codex 可读取的位置。默认路径可设为：

   ```text
   REPO/Archive.zip
   ```

3. 启动 Codex 后发送以下一句话：

   ```text
   Read CODEX_WORKFLOW.md in full. Treat it as the authoritative research and engineering contract. Start from Phase 0, do not skip admission gates, and do not change experimental semantics without recording a decision document.
   ```

4. 本文件既是 Codex 的主提示词，也是项目执行合同。Codex 应逐阶段执行、测试、记录和汇报，而不是一次性大改整个仓库。

---

# 0. 你的角色

你是本项目的 **Codex research engineer、CUDA performance engineer 和 reproducibility maintainer**。

你的职责不是只把代码“跑起来”，而是构建一个能经受论文审稿的测量系统。你必须同时保证：

1. **算法正确性**：移植后的 TurboQuant、KIVI、KVQuant 与各自参考实现一致。
2. **测量公平性**：同模型、同工作量、同执行模式、同计时边界。
3. **机制可解释性**：区分名义 bitwidth、实际 cache bytes、实际 HBM traffic 和 wall time。
4. **数据可追溯性**：每个结果能追溯到代码 SHA、容器、硬件状态、配置和原始样本。
5. **工程可复现性**：从干净环境可以重新构建、运行 smoke test、复现实验和生成图表。

你不得为了获得更好的数字而修改实验语义、选择性重跑慢点、删除异常样本或混合不公平配置。

---

# 1. 研究目标

在单张 RTX PRO 6000 Blackwell 96GB 上，为三种 KV cache quantization method 建立统一的、method-conditioned 定量关系：

\[
T_m = F_m(B,L,r_{\mathrm{eff}}),
\]

以及同工作量 decode speedup：

\[
S_m(B,L,r_{\mathrm{eff}})
=
\frac{T_{\mathrm{BF16}}(B,L)}{T_m(B,L,r_{\mathrm{eff}})}.
\]

其中：

- \(m\)：量化方法；
- \(B\)：静态 decode batch size；
- \(L\)：当前有效 context length；
- \(r_{\mathrm{eff}}\)：由真实 cache allocation 得到的有效压缩倍数；
- \(T_m\)：固定工作量下的单 token decode wall time。

主模型不要求是统一 closed form。每个方法使用同一个 knee-aware 结构，但分别学习：

\[
\tau_m(B,r),\qquad
L_m^*(B,r),\qquad
s_m(B,r),
\]

分别表示：

- context-insensitive latency floor；
- floor 与 context-dependent 区间的拐点；
- 拐点后的 context slope。

基础重建式为：

\[
T_m(B,L,r)
=
\tau_m(B,r)
+
s_m(B,r)[L-L_m^*(B,r)]_+.
\]

这里：

\[
[x]_+=\max(x,0).
\]

---

# 2. 核心研究问题与可证伪假设

## RQ1：decode 时间是否具有 floor–knee–slope 结构？

检验：

\[
T(L)\approx \tau+s[L-L^*]_+.
\]

不能只凭高 \(R^2\) 将 floor 归因于 launch latency。必须通过 CUDA Graph A/B、Nsight Systems 和 hardware counters 排除其他解释。

## RQ2：固定 quantization method 后，\(B,L,r\) 是否足以预测 decode performance？

检验三类模型：

1. 只使用 \(BL/r\) 的简化模型；
2. 完整三变量响应面 \(F_m(B,L,r)\)；
3. knee-aware 响应面。

## RQ3：名义 bitwidth 是否能预测真实 cache bytes？

同时记录：

\[
r_{\mathrm{nominal}},\qquad
r_{\mathrm{alloc}},\qquad
r_{\mathrm{HBM}}.
\]

其中：

\[
r_{\mathrm{nominal}}=\frac{16}{q},
\]

\[
r_{\mathrm{alloc}}
=
\frac{C_{\mathrm{BF16,allocated}}}
{C_{m,\mathrm{allocated}}},
\]

\[
r_{\mathrm{HBM}}
=
\frac{C_{\mathrm{BF16,HBM}}}
{C_{m,\mathrm{HBM}}}.
\]

## RQ4：GQA shared KV 是否在某些路径中被物化？

在相同 tensor geometry 上比较：

- GQA：\(H_Q>H_{KV}\)；
- MHA control：\(H_Q=H_{KV}\)。

必须检查是否存在：

```text
expand().reshape()
repeat_interleave
repeat_kv
full-prefix temporary tensor
```

不能仅由 latency slope 反推 physical traffic amplification。

## RQ5：不同方法偏离简单 byte law 的原因是什么？

方法角色如下：

- **TurboQuant-vLLM**：固定、稠密、可精确计算 packed layout 的 anchor；
- **KIVI**：带 FP16 recent window，检验 \(r_{\mathrm{eff}}\) 随 \(L\) 变化；
- **KVQuant**：带 sparse outliers、indices 和 metadata，检验名义 bitwidth 是否仍具有预测力。

---

# 3. 明确范围

## 3.1 第一阶段纳入范围

- 单张 RTX PRO 6000；
- tensor parallel size = 1；
- decoder-only full-attention model；
- GQA 主模型；
- 静态 batch；
- 固定输出长度；
- fixed-\(L\) decode benchmark；
- growing-context request validation；
- eager 和 CUDA Graph 两条 lane；
- BF16 baseline；
- TurboQuant、KIVI、KVQuant；
- Nsight Systems 和 Nsight Compute 子集分析；
- 每方法独立响应面与跨方法比较。

## 3.2 第一阶段不纳入主结论

- continuous batching scheduler；
- queueing latency；
- multi-GPU tensor parallel；
- CPU/NVMe offload；
- speculative decoding；
- prefix cache sharing；
- 不同模型之间的绝对 latency 横向排名；
- capacity amplification 与 same-work latency 混合比较；
- 单层 kernel speedup 直接代替全模型 speedup。

这些可以作为扩展或附录，但不得混入主响应面。

---

# 4. 两条实现路线

必须同时维护 Reference Lane 和 Measurement Lane。

## 4.1 Reference Lane

目的：验证算法和数值，不用于主性能横向比较。

```text
Official TurboQuant/vLLM environment
Official KIVI environment
Official KVQuant environment
```

每种方法允许有独立容器和独立依赖，只输出：

- quantized cache 小 tensor；
- attention output；
- cache byte breakdown；
- calibration artifacts；
- golden test fixture。

## 4.2 Measurement Lane

目的：生成主论文性能结果。

必须固定：

- 同一模型；
- 同一权重 dtype；
- 同一 fixed-\(L\) runner；
- 同一计时边界；
- 同一 graph mode；
- 同一静态 cache 生命周期；
- 同一输出工作量；
- 同一进程隔离和随机化协议。

只有 method adapter、cache representation 和 method-specific attention decode path 可以变化。

---

# 5. 输入与来源处理

## 5.1 论文压缩包

默认输入：

```text
/Archive.zip
```

处理规则：

1. 只读解压到 `literature/raw/`。
2. 不执行压缩包内任何脚本、二进制或宏。
3. 为每个文件计算 SHA256。
4. 生成：

   ```text
   literature/manifest.csv
   literature/README.md
   literature/checksums.sha256
   ```

5. 对 TurboQuant、KIVI、KVQuant 建立 source note：

   ```text
   docs/method_notes/turboquant.md
   docs/method_notes/kivi.md
   docs/method_notes/kvquant.md
   ```

6. 每个 note 至少记录：

   - 论文标题和版本；
   - 官方仓库与 commit；
   - 算法定义；
   - bitwidth/configuration；
   - metadata；
   - residual/sink/outlier 结构；
   - reference kernel；
   - 已报告的 hardware/backend；
   - 当前移植风险。

## 5.2 第三方代码

所有第三方依赖必须固定 commit，并记录到：

```text
third_party/LOCK.json
third_party/NOTICE.md
```

不得引用浮动 `main` 作为正式实验依赖。

---

# 6. 不可违反的科研与工程规则

在仓库根目录创建 `AGENTS.md`，至少包含以下规则：

```markdown
# Research invariants

1. Raw experiment data is append-only.
2. Never edit, overwrite, or delete artifacts from a completed run.
3. Profiler-instrumented timing must never be reported as normal timing.
4. No performance claim is valid without code SHA, container digest,
   hardware manifest, config, raw samples, and independent process replicates.
5. Do not change benchmark semantics and implementation in the same PR.
6. Every CUDA kernel change requires numerical tests, Compute Sanitizer,
   CUDA Graph capture/replay, and allocation tests.
7. Do not introduce torch.cat, dynamic allocation, CPU synchronization,
   or tensor-to-host conversion inside the measured decode region.
8. No method enters the full scan until all admission gates pass.
9. Same-work speedup and capacity amplification must remain separate.
10. Never compare graph-on results against graph-off results.
11. Never cherry-pick the fastest run or selectively rerun slow points.
12. Every exclusion must be recorded with a machine-readable reason.
13. Do not silently fall back to another attention backend or cache dtype.
14. Do not infer physical HBM traffic only from nominal bytes and latency.
15. Do not alter the preregistered grid after seeing performance results.
```

---

# 7. 推荐仓库结构

创建或整理为：

```text
kv-compression-bench/
├── AGENTS.md
├── CODEX_WORKFLOW.md
├── README.md
├── Makefile
├── pyproject.toml
├── uv.lock
│
├── docker/
│   ├── measurement.Dockerfile
│   ├── reference-turboquant.Dockerfile
│   ├── reference-kivi.Dockerfile
│   └── reference-kvquant.Dockerfile
│
├── configs/
│   ├── hardware/
│   │   └── rtx_pro_6000.yaml
│   ├── models/
│   │   └── primary_gqa_model.yaml
│   ├── methods/
│   │   ├── bf16.yaml
│   │   ├── turboquant.yaml
│   │   ├── kivi.yaml
│   │   └── kvquant.yaml
│   └── plans/
│       ├── smoke.yaml
│       ├── pilot.yaml
│       ├── graph_ab.yaml
│       ├── full_scan.yaml
│       └── profiler_subset.yaml
│
├── src/kvbench/
│   ├── adapters/
│   │   ├── base.py
│   │   ├── bf16.py
│   │   ├── turboquant.py
│   │   ├── kivi.py
│   │   └── kvquant.py
│   ├── model/
│   │   ├── model_loader.py
│   │   ├── llama_decode.py
│   │   └── attention_reference.py
│   ├── runtime/
│   │   ├── static_cache.py
│   │   ├── cuda_graph.py
│   │   ├── fixed_l_runner.py
│   │   ├── growing_context_runner.py
│   │   └── gpu_lock.py
│   ├── metrics/
│   │   ├── timing.py
│   │   ├── memory.py
│   │   ├── nvml.py
│   │   ├── nsys.py
│   │   └── ncu.py
│   ├── schema/
│   │   ├── config.py
│   │   ├── result.py
│   │   └── validation.py
│   └── cli.py
│
├── reference/
│   ├── turboquant/
│   ├── kivi/
│   └── kvquant/
│
├── calibration/
│   └── kvquant/
│
├── tests/
│   ├── unit/
│   ├── golden/
│   ├── cuda/
│   ├── graph_capture/
│   ├── allocation/
│   └── perf_sanity/
│
├── analysis/
│   ├── validate_runs.py
│   ├── fit_linear.py
│   ├── fit_segmented.py
│   ├── fit_knee.py
│   ├── fit_response_surface.py
│   ├── cross_validation.py
│   ├── mechanism_analysis.py
│   └── make_figures.py
│
├── scripts/
│   ├── preflight.sh
│   ├── build_extensions.sh
│   ├── run_plan.sh
│   ├── collect_nsys.sh
│   ├── collect_ncu.sh
│   └── reproduce.sh
│
├── docs/
│   ├── experiment_contract.md
│   ├── measurement_protocol.md
│   ├── status.md
│   ├── blockers.md
│   ├── risk_register.md
│   ├── decisions/
│   └── method_notes/
│
├── literature/
│   ├── raw/
│   ├── manifest.csv
│   └── checksums.sha256
│
└── artifacts/
    └── <run_id>/
        ├── manifest.json
        ├── samples.parquet
        ├── summary.json
        ├── exclusions.parquet
        ├── stdout.log
        ├── stderr.log
        ├── telemetry.parquet
        ├── nsys/
        ├── ncu/
        └── checksums.sha256
```

`artifacts/` 默认不由普通代码 PR 修改。正式实验产物必须 append-only。

---

# 8. 统一 Adapter 接口

实现一个明确、类型安全的 method adapter：

```python
from typing import Protocol

class KVCacheMethod(Protocol):
    name: str

    def allocate(
        self,
        *,
        batch_size: int,
        max_context: int,
        model_config: "ModelConfig",
        device: "torch.device",
    ) -> "CacheState":
        ...

    def store_prefill(
        self,
        *,
        cache: "CacheState",
        key: "torch.Tensor",
        value: "torch.Tensor",
        positions: "torch.Tensor",
    ) -> None:
        ...

    def append_decode(
        self,
        *,
        cache: "CacheState",
        key: "torch.Tensor",
        value: "torch.Tensor",
        positions: "torch.Tensor",
    ) -> None:
        ...

    def decode_attention(
        self,
        *,
        query: "torch.Tensor",
        cache: "CacheState",
        seq_lens: "torch.Tensor",
        output: "torch.Tensor",
    ) -> "torch.Tensor":
        ...

    def allocated_bytes(self, cache: "CacheState") -> int:
        ...

    def byte_breakdown(self, cache: "CacheState") -> dict[str, int]:
        ...

    def logical_bf16_bytes(
        self,
        *,
        batch_size: int,
        context_length: int,
        model_config: "ModelConfig",
    ) -> int:
        ...

    def config_fingerprint(self) -> dict[str, object]:
        ...

    def supports_cuda_graph(self) -> bool:
        ...
```

硬约束：

- 所有大 buffer 在计时前分配；
- `decode_attention` 内不得进行 host sync；
- 不得在 measured region 内重新编译或 autotune；
- 不得完整反量化整个 prefix 到 BF16 临时 tensor；
- 不得将 GQA KV 复制到 query-head 数量；
- `allocated_bytes()` 必须与真实 storage 一致；
- `byte_breakdown()` 必须能拆分 data、scale、zero、norm、metadata、outlier、index、residual、sink 和 padding。

---

# 9. 主模型与控制变量

## 9.1 主模型选择规则

选择一个满足以下条件的 7B–9B、decoder-only、GQA、长上下文模型：

- 权重可在单张 96GB 卡上以 BF16 运行；
- head dimension 和 GQA geometry 被三个 adapter 支持；
- 不包含会改变 attention 路径的混合 SSM/Mamba 层；
- 不使用 sliding-window 作为主路径；
- tokenizer 和 config 可固定；
- 能构造至少 128K 的合成 token context，或明确记录模型最大长度限制。

将最终模型 ID、revision、config SHA 固定到：

```text
configs/models/primary_gqa_model.yaml
```

## 9.2 MHA 控制

创建 operator-level MHA control：

```text
num_query_heads == num_kv_heads
n_rep == 1
```

必须保持：

- head dimension 相同；
- batch 和 context 相同；
- dtype 相同；
- kernel family 尽量相同。

---

# 10. 方法配置

任何方法配置名和参数都必须先从固定 commit 的源代码验证。若上游名称不同，更新本项目配置并写 decision record，不得静默猜测。

## 10.1 TurboQuant-vLLM

主回归使用同一个 MSE+NC family：

```yaml
variants:
  - turboquant_4bit_nc
  - turboquant_k3v4_nc
  - turboquant_3bit_nc
```

held-out validation：

```yaml
- turboquant_k8v4
```

记录：

- key bits；
- value bits；
- key packed bytes；
- value packed bytes；
- norm/scale/zero bytes；
- slot alignment；
- skipped layers；
- block size；
- decode split count；
- graph support mode。

不得将 FP8-key path 和 MSE-key path 当作同一条连续 bitwidth 曲线，除非模型包含 method-path categorical feature。

## 10.2 KIVI

初始 canonical configuration：

```yaml
group_size: 32
residual_length: 32
variants:
  - {k_bits: 4, v_bits: 4}
  - {k_bits: 2, v_bits: 4}
  - {k_bits: 2, v_bits: 2}
```

held-out falsification pair：

```yaml
- {k_bits: 4, v_bits: 2}
- {k_bits: 2, v_bits: 4}
```

两者名义总 bit 近似，但 Key/Value 分配不同。用于检验单一总压缩率是否充分。

KIVI 的有效 byte ratio 必须按 \(L\) 计算：

\[
\rho_{\mathrm{KIVI}}(L)
=
\frac{
\min(L,W)c_{16}
+
\max(L-W,0)c_q
}{Lc_{16}}.
\]

这里 \(W\) 是 residual FP16 window。

## 10.3 KVQuant

主回归配置：

```yaml
sink_tokens: 5
outlier_cap: fixed_from_calibration
variants:
  - {bits: 4}
  - {bits: 3}
  - {bits: 2}
```

固定：

- calibration dataset；
- quantizer artifacts；
- outlier cap；
- sink token 数；
- LUT/scale precision；
- sparse index dtype；
- buffer layout；
- quantization threshold policy。

实际 bytes 拆分为：

\[
C_{\mathrm{KVQ}}
=
C_{\mathrm{dense}}
+C_{\mathrm{metadata}}
+C_{\mathrm{outlier\ value}}
+C_{\mathrm{outlier\ index}}
+C_{\mathrm{sink}}
+C_{\mathrm{padding}}.
\]

主扫描期间不得重新 calibration。

---

# 11. 分阶段执行流程

每个 Phase 必须遵循：

```text
Inspect → Plan → Implement → Test → Record → Phase Report
```

不得跨过 gate。若遇到 blocker，先记录证据和最小复现，再决定修复，不得用 silent fallback 掩盖。

---

## Phase 0：仓库与输入审计

### 目标

建立当前状态、依赖、输入和风险清单，不修改 benchmark hot path。

### 任务

1. 读取本文件和 `AGENTS.md`。
2. 检查是否已有仓库；若为空则初始化。
3. 检查 Git working tree。
4. 只读解压并清点 `Archive.zip`。
5. 搜索现有代码、模型配置、CUDA extensions、Dockerfiles。
6. 建立：

   ```text
   docs/status.md
   docs/risk_register.md
   docs/blockers.md
   docs/decisions/0001-initial-scope.md
   ```

7. 创建 issue/task breakdown：

   ```text
   E00 Hardware preflight
   E01 Repository scaffold and schemas
   E02 BF16 static-cache baseline
   E03 Fixed-L benchmark
   E04 CUDA Graph harness
   E05 TurboQuant reference lane
   E06 TurboQuant measurement adapter
   E07 KIVI reference lane
   E08 KIVI measurement adapter
   E09 KVQuant calibration
   E10 KVQuant reference lane
   E11 KVQuant measurement adapter
   E12 Admission gates
   E13 Pilot scan
   E14 Nsight Systems integration
   E15 Nsight Compute integration
   E16 Full scan
   E17 Knee and response-surface fitting
   E18 Reproducibility package
   ```

### 验收标准

- 没有执行压缩包中的未知代码；
- 所有输入有 checksum；
- 所有第三方仓库有 commit 计划；
- 风险登记至少覆盖 CUDA compatibility、Graph support、GQA、full dequantization、旧依赖和 OOM。

### Phase Report

输出：

```text
PHASE 0 REPORT
Status:
Repository state:
Inputs found:
Checksums generated:
Major risks:
Blockers:
Files created:
Next phase:
```

---

## Phase 1：RTX PRO 6000 硬件认证

### 目标

证明当前机器、驱动、toolchain 和扩展编译路径可以可靠运行 Blackwell CUDA kernel。

### 必须实现

```bash
make preflight
```

它应采集：

```text
nvidia-smi -L
nvidia-smi -q
nvcc --version
ncu --version
nsys --version
python/torch/triton/vllm versions
GPU full name
GPU UUID
PCI ID
compute capability
VRAM
ECC
power limit
SM clock
memory clock
temperature
vBIOS
CPU model
host kernel
container digest
```

### CUDA 认证

1. 编译最小 PyTorch CUDA extension。
2. 验证 native execution。
3. 验证 `CUDA_FORCE_PTX_JIT=1`。
4. 运行 Compute Sanitizer。
5. 检查 extension 不出现 `no kernel image`。
6. 检查不存在其他活跃 CUDA 进程。
7. 验证：

   ```python
   torch.cuda.is_available()
   torch.cuda.get_device_capability()
   ```

8. 将实际 compute capability 写入 manifest，不在代码中盲目硬编码。

### Gate G0

- extension build 通过；
- native run 通过；
- PTX/JIT 检查通过；
- Compute Sanitizer 通过；
- 完整 SKU 和功耗/时钟状态已记录；
- 失败时明确给出 toolchain blocker，不得继续 full implementation。

---

## Phase 2：仓库骨架、配置和 Schema

### 目标

在写性能代码前固定实验合同和数据结构。

### 任务

1. 创建推荐目录结构。
2. 创建：

   ```text
   docs/experiment_contract.md
   docs/measurement_protocol.md
   ```

3. 创建 Pydantic 或等价严格 schema：

   ```text
   ExperimentConfig
   MethodConfig
   HardwareManifest
   SampleRecord
   RunSummary
   ExclusionRecord
   ```

4. 创建 CLI：

   ```bash
   kvbench validate-config
   kvbench preflight
   kvbench run --plan <yaml>
   kvbench validate-run <run_dir>
   kvbench summarize <run_dir>
   ```

5. 创建 append-only artifact writer：

   - 临时目录写入；
   - 成功后原子 rename；
   - 已存在 run ID 时拒绝覆盖；
   - 生成 SHA256；
   - 写完成标记。

6. 创建基础 Makefile：

   ```bash
   make bootstrap
   make preflight
   make test
   make test-cuda
   make test-graph
   make smoke METHOD=<method>
   make pilot
   make full-scan
   make profile-subset
   make fit
   make figures
   make reproduce RUN_ID=<id>
   ```

### 验收标准

- invalid config 被拒绝；
- 结果目录不可覆盖；
- run manifest 可以独立重建实验命令；
- schema 有 unit tests；
- 尚未产生论文性能数字。

---

## Phase 3：BF16 静态 cache baseline

### 目标

建立所有量化方法的公平基线。

### 两种 runner

#### A. Fixed-\(L\) decode runner

- 预先构造长度为 \(L\) 的 cache；
- 每次重复相同形状的 one-token decode；
- 有效 \(L\) 不增长；
- 用于拟合 \(T(B,L)\)、floor、knee、slope；
- 支持 CUDA Graph replay。

#### B. Growing-context runner

- 从 \(L\) 开始生成固定 \(O\) 个 token；
- context 依次为：

  \[
  L,L+1,\ldots,L+O-1.
  \]

- 用于验证 fixed-\(L\) 模型能否还原 request latency。

### 计时边界

主 endpoint：

```text
host-observed wall time per decode step
```

同时记录：

```text
CUDA event GPU time
kernel count
allocated bytes
peak memory
NVML telemetry
```

计时区不得包含：

- model load；
- JIT compile；
- Triton autotune；
- graph capture；
- cache construction；
- tokenizer；
- sampling；
- logging flush；
- profiler initialization。

### Baseline Gate G1

- BF16 output 与 reference attention 一致；
- 无 `torch.cat` cache growth；
- 无 measured-region allocation；
- GQA 不物化为 query heads；
- fixed-\(L\) 和 growing-context runner 均可运行；
- eager 和 graph 两条 lane 均可运行；
- 三个独立进程复现稳定。

---

## Phase 4：统一 adapter 与测试框架

### 目标

让四个实现共享 runner，而不是每个方法拥有自己的 benchmark 脚本。

### 任务

1. 实现 `KVCacheMethod` Protocol。
2. 实现 BF16 adapter 作为 reference adapter。
3. 创建通用 golden test harness。
4. 创建 allocation accounting harness。
5. 创建 graph capture/replay harness。
6. 创建 execution-path audit：

   - 捕获 allocation；
   - 检查 host sync；
   - 检查 kernel names；
   - 检查临时 tensor shape；
   - 检查 GQA replication；
   - 检查 full-prefix dequantization。

7. 建立 `MethodAdmissionReport`。

### 验收标准

- 任一 adapter 能通过同一 CLI 运行；
- runner 不包含 method-specific 分支，或分支被限制在 adapter construction；
- adapter config fingerprint 写入 manifest；
- BF16 adapter 通过所有框架测试。

---

## Phase 5：TurboQuant Reference Lane

### 目标

固定官方/上游实现，生成 golden fixtures。

### 任务

1. 固定 vLLM/TurboQuant commit。
2. 创建独立 reference container。
3. 验证实际支持的 cache dtype 名称。
4. 对每个主配置生成：

   - cache slot layout；
   - packed bytes；
   - small-tensor store output；
   - small-tensor decode output；
   - graph support 信息；
   - kernel trace。

5. 保存为版本化 fixtures，不保存完整版权代码副本。

### 验收标准

- reference command 可复现；
- 每个 fixture 有 code SHA、config、tensor seed 和 checksum；
- 实际配置名与源码一致；
- 未将 reference 性能直接当成主横向数据。

---

## Phase 6：TurboQuant Measurement Adapter

### 目标

将 TurboQuant kernel 接入统一 Measurement Lane。

### 任务

1. 封装 store、append、decode kernel。
2. 使用固定 cache layout。
3. 预分配 workspace。
4. graph capture 前完成 Triton compile/autotune。
5. 查询真实 slot size 和 alignment。
6. 禁止 full-prefix BF16 dequantization。
7. 禁止隐式 backend fallback。
8. 对比 reference fixture。
9. 对每个配置实现 byte breakdown。
10. 实现 CUDA Graph capture/replay test。

### Gate G2-TQ

- reference numerical test 通过；
- predicted bytes 与 allocated bytes 偏差小于 1%；
- graph replay 正确；
- no full-prefix dequantization；
- no GQA replication；
- no measured-region allocation；
- Compute Sanitizer 通过；
- smoke grid 通过。

---

## Phase 7：KIVI Reference Lane

### 目标

固定官方算法路径，生成不同 K/V bitwidth 的 golden fixtures。

### 任务

1. 固定 KIVI commit 和依赖。
2. 在 RTX PRO 6000 toolchain 上重新编译 extension。
3. 记录实际编译 flags 和 architecture。
4. 验证 GQA 支持路径。
5. 生成：

   - K4/V4；
   - K2/V4；
   - K2/V2；
   - held-out K4/V2。

6. 固定 group size 和 residual length。
7. 对 recent FP16 window rollover 做 reference trace。

### 验收标准

- reference runner 可运行；
- extension 在当前 GPU 上无 compatibility fallback；
- K/V quantization direction 被明确记录；
- residual window 行为有 fixture；
- GQA 不通过 `repeat_kv` 物化。

---

## Phase 8：KIVI Measurement Adapter

### 目标

将 KIVI 变为 graph-safe、static-cache、可计量的 adapter。

### 任务

1. 抽离 quantize/dequantize kernel。
2. 用预分配 ring/static buffers 替代 dynamic cache growth。
3. 分离：

   ```text
   recent FP16 region
   quantized historical region
   metadata region
   workspace
   ```

4. 实现 residual rollover，且 measured region 不分配。
5. 使用 KV-head indexing 支持 GQA。
6. 不允许先完整反量化历史 prefix。
7. 实现按 \(L\) 计算的真实 `r_alloc`。
8. 添加 K2/V4 与 K4/V2 held-out test。
9. 完成 graph capture/replay。

### Gate G2-KIVI

同 G2-TQ，并额外要求：

- residual window byte accounting 正确；
- \(r_{\mathrm{alloc}}(L)\) 随 \(L\) 的变化可被 unit test 验证；
- rollover 前后输出和 storage 正确；
- held-out K/V asymmetry 配置可运行。

---

## Phase 9：KVQuant Calibration

### 目标

将所有数据依赖的量化器和 outlier 规则冻结在主实验之前。

### 任务

1. 固定 calibration dataset 与 revision。
2. 固定 preprocessing 和 seed。
3. 运行官方 calibration pipeline。
4. 确定固定 outlier cap。
5. 导出：

   ```text
   calibration/kvquant/<calibration_id>/
   ├── quantizers.bin
   ├── calibration_config.json
   ├── dataset_manifest.json
   ├── layer_stats.parquet
   └── checksums.sha256
   ```

6. 保存 outlier value/index dtype。
7. 保存 sink-token policy。
8. 主扫描期间对 calibration artifact 只读。

### 验收标准

- calibration 可由固定 seed 重建；
- artifact 有 checksum；
- outlier cap 固定；
- 不在 benchmark measured region 中运行 calibration、top-k 或 CPU selection。

---

## Phase 10：KVQuant Reference Lane

### 目标

生成 dense quantization、metadata 和 sparse correction 的 golden fixtures。

### 任务

1. 固定 KVQuant commit。
2. 固定旧 reference environment，不要求其成为主 measurement runtime。
3. 对 4/3/2-bit 生成：

   - dense packed values；
   - lookup/scale metadata；
   - outlier values；
   - outlier indices；
   - sink FP16 storage；
   - attention output。

4. 对没有 outlier、少量 outlier、达到 cap 三种情况建 fixture。
5. 验证 pre-RoPE/post-RoPE 语义。

### 验收标准

- fixtures 包含所有 sparse components；
- cap 行为确定；
- reference 输出稳定；
- 所有依赖有版本锁定。

---

## Phase 11：KVQuant Measurement Adapter

### 目标

实现固定形状、预分配、graph-safe 的 dense+sparse adapter。

### 任务

1. 为 dense quantized cache 预分配 buffer。
2. 为 outlier values 和 indices 预分配 cap-sized buffer。
3. 为 sink FP16 cache 单独分配。
4. 保证 decode hot path 中：

   - 无 CPU top-k；
   - 无动态 sparse allocation；
   - 无完整 BF16 prefix；
   - 无 GQA expansion；
   - 无 Python list conversion；
   - 无 host synchronization。

5. 逐项记录 byte breakdown。
6. 实现 sparse correction kernel 或封装 reference CUDA kernel。
7. graph capture 前固定指针和 workspace。
8. 对照 reference fixture。

### Gate G2-KVQ

同 G2-TQ，并额外要求：

- dense、metadata、outlier value、outlier index、sink 和 padding 可独立计量；
- outlier count 不超过 cap；
- cap reached case 正确；
- graph replay 对 sparse buffers 正确；
- 未发生隐式 full dequantization。

---

## Phase 12：统一 Admission Gates

每个方法必须产生机器可读报告：

```text
artifacts/admission/<method_config_id>/report.json
```

### G1 数值正确性

- BF16 baseline 与 reference attention 一致；
- 量化 adapter 与该方法 official reference 一致到预定义容差；
- 无 NaN/Inf；
- 短 generation 无灾难性崩溃。

不要要求量化结果与 BF16 完全一致；应要求与同方法 reference 一致。

### G2 内存正确性

目标：

\[
\left|
\frac{C_{\mathrm{predicted}}-C_{\mathrm{allocated}}}
{C_{\mathrm{allocated}}}
\right|<1\%.
\]

必须记录：

- allocator bytes；
- tensor storage bytes；
- padding；
- workspace；
- temporary peak。

### G3 执行路径正确性

- 无完整 BF16 prefix；
- 无 GQA repeat materialization；
- 无 measured-region `torch.cat`；
- 无 measured-region allocation；
- 无 CPU sync；
- kernel count 稳定；
- 无 silent backend fallback。

### G4 CUDA Graph

- capture 成功；
- replay 成功；
- 输出正确；
- replay 不新分配；
- 每个 context bucket 的 graph 策略被记录。

若某方法暂时不能 graph capture：

- 它只能进入统一 eager lane；
- graph-on 结果不得与其 eager 结果直接横比；
- blocker 必须记录。

### G5 重复性

默认目标：

```text
至少 3 个独立进程
每进程多个 measured steps
同一点 CV <= 3%
```

CV 超阈值时标记 `unstable`，不得只保留最快 replicate。

---

## Phase 13：Pilot Scan

### 目标

用较小预算验证测量系统、定位初步 knee，并决定是否进入 full scan。

### Pilot Grid

```yaml
batch_size: [1, 4, 8]
context_length:
  - 4096
  - 8192
  - 16384
  - 24576
  - 32768
  - 49152
  - 65536
  - 98304
  - 131072
warmup_steps: 64
measured_steps: 128
process_replicates: 3
```

### 执行顺序

使用 blocked randomization：

1. method/config 为 block；
2. block 内随机化 \((B,L)\)；
3. replicate 之间轮换 block 顺序；
4. 所有失败点、OOM 点和 unstable 点完整记录。

### Pilot QC

自动检查：

- memory feasibility；
- temperature drift；
- power/clock drift；
- CV；
- graph/eager mode；
- latency monotonicity 仅作为告警，不作为删除依据；
- kernel count stability；
- output checksum；
- allocation consistency。

### Pilot 输出

```text
pilot_qc_report.md
pilot_qc.json
provisional_knees.parquet
pilot_plots/
```

### 进入 Full Scan 的 Gate

- 所有主配置通过 admission；
- 没有系统性 silent fallback；
- 原始数据完整；
- 至少能拟合 provisional segmented model；
- knee 附近有足够密度；
- 若 graph lane 的 knee 消失，记录为研究结果，不得强行保留原叙事。

---

## Phase 14：CUDA Graph A/B 机制实验

### 目标

区分 launch/runtime floor 与 device-side context cost。

### 固定条件

同一方法、同一 cache、同一 backend、同一 shape，仅改变：

```text
CUDA Graph OFF
CUDA Graph ON
```

### 子集

```yaml
batch_size: [1, 4]
context_length: [4096, 16384, 24576, 32768, 65536, 131072]
```

### 检验

- floor 是否下降；
- post-knee slope 是否近似不变；
- knee 是否左移；
- kernel launch gaps 是否消失；
- graph capture 是否改变 kernel/backend；
- static cache 是否完全一致。

不能同时改变 DynamicCache/StaticCache、attention backend 或 model compile mode。

---

## Phase 15：Profiler 子集

### 15.1 Nsight Systems

选择：

```text
knee 前
knee 附近
knee 后
最长 context
```

分析：

- CPU kernel submission；
- launch gaps；
- kernel count；
- synchronization；
- graph replay；
- cache update；
- quantize/dequantize/sparse correction；
- overlap。

### 15.2 Nsight Compute

先执行：

```bash
ncu --query-metrics
```

为当前 SM 生成可用 metric map，不硬编码旧架构 metric 名称。

重点采集：

- DRAM read/write bytes；
- L2 read/write bytes；
- L2 hit rate；
- memory throughput；
- SM utilization；
- occupancy；
- kernel duration；
- instruction mix；
- temporary traffic。

### 严格分离 timing 与 profiler

```text
run_kind: timing | nsys | ncu
```

只有 `timing` 进入正式 latency 分析。

---

## Phase 16：Full Scan

### 主配置

```yaml
batch_size: [1, 2, 4, 8, 16]
context_length:
  - 4096
  - 8192
  - 16384
  - 24576
  - 32768
  - 49152
  - 65536
  - 98304
  - 131072
warmup_steps: 64
measured_steps: 256
process_replicates: 5
```

### Adaptive Knee Densification

依据 pilot 中每个 method/config 的 provisional \(L^*\)，在预注册规则下增加：

\[
\{0.6,0.75,0.9,1.0,1.1,1.25,1.5\}\times L^*.
\]

增加点的规则必须在看到 full-scan 结果之前固定，并记录 rounding 与最大长度限制。

### Memory Feasibility

运行前计算：

\[
M_{weights}+M_{cache}+M_{workspace}+M_{margin}<M_{GPU}.
\]

不可运行点写入：

```json
{
  "status": "capacity_infeasible",
  "reason": "predicted memory exceeds configured safety margin"
}
```

不得将 OOM 点静默删除。

### same-work 限制

BF16 和 compressed 的 speedup 只在相同：

```text
model
B
L
output work
graph mode
sampling mode
```

下计算。

capacity amplification 单独报告：

```text
claim_class = capacity_amplification
```

---

# 12. 计时与运行协议

每个正式点严格执行：

```text
1. 获取 GPU 独占锁
2. 检查无其他活跃 CUDA 进程
3. 检查 git tree clean 或记录 dirty diff
4. 加载固定 model revision
5. 构造 method adapter
6. 完成 JIT/compile/autotune
7. 完成 graph capture
8. 构造长度 L 的 cache
9. warmup
10. 等待或验证 thermal/clock 稳定
11. 运行 measured steps
12. 采集 NVML telemetry
13. 写原始 samples
14. 验证 schema
15. 原子关闭 run directory
16. 计算 checksums
```

不得：

- 只重跑慢点；
- 丢弃首个“看起来慢”的正式 replicate；
- 在同一 run 中改变 GPU power limit；
- 在 timing 期间启动 profiler；
- 在测量后修改 run manifest；
- 把平均值当作唯一结果而丢弃 raw samples。

---

# 13. 数据 Schema

每个 sample 至少包含：

```text
run_id
timestamp_utc
git_sha
git_dirty
container_digest
gpu_uuid
gpu_full_name
pci_device_id
driver_version
cuda_runtime_version
cuda_toolkit_version
torch_version
triton_version
method
method_config_id
method_config_fingerprint
model_id
model_revision
weight_dtype
batch_size
context_length
decode_step
runner_kind
graph_mode
attention_backend
cache_layout
r_nominal
r_alloc
r_hbm
logical_bf16_bytes
cache_allocated_bytes
cache_data_bytes
metadata_bytes
scale_zero_bytes
norm_bytes
residual_bytes
sink_bytes
outlier_value_bytes
outlier_index_bytes
padding_bytes
workspace_bytes
peak_memory_bytes
wall_time_ms
gpu_event_ms
kernel_count
sm_clock_mhz
memory_clock_mhz
power_w
temperature_c
replicate
step_index
random_seed
run_kind
status
failure_reason
```

说明：

- `r_hbm` 只在 profiler 点存在；
- 缺失应为 null，并有 `run_kind`；
- 不能用估计 HBM traffic 填充 `r_hbm`；
- byte breakdown 总和必须可检查。

---

# 14. 数学建模与统计分析

## 14.1 预定义候选模型

### Model A：线性模型

\[
T=\alpha+\beta L.
\]

### Model B：自由 segmented model

\[
T=\alpha+\beta_1L+\beta_2(L-L^*)_+.
\]

### Model C：floor/max model

\[
T=\max(\tau,a+cL).
\]

### Model D：method-specific knee-aware response surface

\[
T_m(B,L,r)
=
\tau_m(B,r)+s_m(B,r)[L-L_m^*(B,r)]_+.
\]

### Model E：naive byte law

\[
T=G(BL/r_{\mathrm{alloc}}).
\]

### Model F：feature-augmented model

\[
T=F(
B,L,r_{\mathrm{alloc}},
\text{metadata bytes},
\text{FP16 fraction},
\text{outlier bytes},
\text{kernel count}
).
\]

## 14.2 拟合方法

优先顺序：

1. segmented nonlinear regression；
2. GAM/tensor-product spline；
3. Gaussian process；
4. monotonic boosting，作为预测 baseline；
5. 不使用无法解释且无法校准不确定性的复杂深度网络作为首选。

## 14.3 模型比较

报告：

- AIC/BIC，若适用；
- held-out MAE/MAPE；
- P95 relative error；
- residual plot；
- calibration curve；
- knee confidence interval；
- speedup sign accuracy；
- method ranking accuracy。

## 14.4 严格交叉验证

不得随机打散相邻 \(L\) 点作为唯一验证。

必须运行：

### Leave-one-batch-out

例如用 \(B=1,2,8,16\) 训练，预测 \(B=4\)。

### Leave-one-config-out

对每种方法留出一个压缩配置。

### Leave-context-band-out

例如留出：

```text
24K–48K
```

检验 knee 区域预测。

### Session holdout

测试 session 必须来自不同进程/独立 run。

## 14.5 Knee 置信区间

bootstrap 的 resampling unit 必须是 process/session，而不是单个 decode step。

报告：

\[
L^*,\quad 95\%\ \mathrm{CI}.
\]

## 14.6 目标指标

预注册目标，不得看到结果后随意修改：

```text
Median relative error <= 5%
P95 relative error <= 10%
Speedup sign accuracy >= 95%
Method ranking accuracy >= 90%
Knee relative error <= 10%
```

若未达到，诚实报告失败，并分析是否需要 metadata/outlier/kernel features。

---

# 15. 关键机制派生量

## 15.1 Byte amplification

\[
A_{bytes}
=
\frac{\rho_{alloc}}{\rho_{nominal}},
\]

其中：

\[
\rho=1/r.
\]

## 15.2 Traffic amplification

\[
A_{traffic}
=
\frac{\rho_{HBM}}{\rho_{alloc}}.
\]

## 15.3 Critical-path elasticity

\[
e_{KV}^{crit}
=
\frac{\partial\ln T}{\partial\ln K_{KV}}.
\]

用于区分：

- KV bytes 存在；
- KV traffic 存在；
- KV 是否处于 wall-time critical path。

## 15.4 Knee scaling tests

检验：

\[
BL^*\approx \mathrm{constant},
\]

以及理想 byte law：

\[
B\rho L^*\approx \mathrm{constant}.
\]

这些是待检验假设，不得在拟合中强制成立后再声称发现。

---

# 16. 图表与最终分析产物

自动生成至少以下图：

1. `T vs L`，标出 floor、knee、slope 和 CI；
2. Graph OFF vs ON；
3. GQA vs synthetic MHA control；
4. \(f^{bytes}\)、physical traffic share、critical-path elasticity；
5. nominal compression vs allocated compression vs HBM compression；
6. 三种方法的 knee surface \(L_m^*(B,r)\)；
7. predicted vs measured latency；
8. leave-one-B / leave-one-config error；
9. same-work speedup heatmap；
10. metadata/outlier 对 KVQuant 偏差的分解；
11. KIVI residual window 对 \(r_{alloc}(L)\) 的影响；
12. TurboQuant fixed-layout 的 slot-byte 与 slope 关系。

每张图必须由脚本从 immutable raw data 生成，不得手工修改数据点。

---

# 17. Codex 的 Git/PR 工作方式

每个 issue 对应一个范围清晰的 commit 或 PR。

## PR 模板

```markdown
## Scope

## Research semantics changed?
- [ ] No
- [ ] Yes, recorded in docs/decisions/<id>.md

## Correctness evidence
- Unit tests:
- Golden tests:
- Numerical tolerances:

## CUDA evidence
- SM120/current-architecture build:
- PTX/JIT test:
- Compute Sanitizer:
- CUDA Graph capture/replay:

## Memory evidence
- Predicted bytes:
- Allocated bytes:
- Peak bytes:
- Temporary allocations:

## Execution-path audit
- Full-prefix dequantization:
- GQA replication:
- Host synchronization:
- Dynamic allocation:
- Backend fallback:

## Commands run

## Known limitations

## Raw benchmark data
No paper benchmark data is created or edited by this PR unless the issue is an experiment-run issue.
```

## 代码变更原则

- 一次 PR 不同时改变 benchmark semantics 和 method implementation；
- kernel optimization 与 correctness port 分开；
- 优化前先有 golden test；
- 不为通过性能 sanity test 而放宽数值容差；
- 不把微基准数字写入论文结果目录；
- 不自动 push 或发布，除非操作者明确授权。

---

# 18. Codex 每阶段汇报格式

每个 Phase 结束后必须输出：

```text
PHASE <N> REPORT

Status: PASS | PARTIAL | BLOCKED | FAIL

Completed:
- ...

Changed files:
- ...

Commands executed:
- ...

Tests and evidence:
- ...

Admission gates:
- G0: ...
- G1: ...

Observed risks:
- ...

Blockers:
- ...

Scientific interpretation:
- Only facts directly supported by current evidence.

Next action:
- ...
```

不得仅回复“done”。

---

# 19. 失败处理协议

遇到失败时执行：

1. 保留 stdout/stderr、config、manifest 和最小复现。
2. 将状态标为：

   ```text
   build_failed
   runtime_failed
   numerical_failed
   graph_capture_failed
   profiler_failed
   capacity_infeasible
   unstable
   backend_fallback
   unsupported_geometry
   ```

3. 写入 `docs/blockers.md`。
4. 不将失败点从计划中删除。
5. 不用另一个 backend 静默替代。
6. 不在同一正式 run 中修代码后继续追加数据；修复后创建新 run ID。
7. 若官方算法无法在统一 Measurement Lane 忠实实现，明确降级为 reference-only，不伪称公平横比。

---

# 20. 最终交付物

项目完成时必须存在：

```text
1. 固定、可构建的 measurement container
2. 三个独立 reference containers 或可复现 reference environments
3. RTX PRO 6000 preflight 和硬件 manifest
4. BF16 static-cache baseline
5. TurboQuant/KIVI/KVQuant 三个统一 adapters
6. eager 和 CUDA Graph execution lanes
7. fixed-L 和 growing-context runners
8. admission reports
9. pilot 和 full-scan immutable raw data
10. Nsight Systems / Nsight Compute 子集
11. 每方法 knee-aware model
12. 严格 cross-validation 结果
13. 自动图表脚本
14. decision log、risk register、blockers
15. 从干净环境复现指定 run 和 figure 的命令
16. 最终 README 和 research report
```

最终复现入口应尽量接近：

```bash
make bootstrap
make preflight
make test
make reproduce RUN_ID=<published_run_id>
make figures
```

---

# 21. 正式运行配置模板

创建 `configs/plans/full_scan.yaml`：

```yaml
experiment:
  name: main_b_l_r_scan
  model_config: configs/models/primary_gqa_model.yaml
  weight_dtype: bfloat16
  tensor_parallel_size: 1
  runtime: static_single_gpu
  graph_mode: cuda_graph
  runner_kind: fixed_l
  output_tokens_for_request_validation: 256

hardware:
  expected_gpu_family: RTX_PRO_6000_BLACKWELL
  exclusive_gpu: true
  max_memory_fraction: 0.88
  require_stable_clocks: true
  record_full_sku: true

methods:
  - name: bf16
    configs:
      - id: bf16

  - name: turboquant
    configs:
      - id: tq_4bit_nc
      - id: tq_k3v4_nc
      - id: tq_3bit_nc

  - name: kivi
    common:
      group_size: 32
      residual_length: 32
    configs:
      - id: k4v4
        k_bits: 4
        v_bits: 4
      - id: k2v4
        k_bits: 2
        v_bits: 4
      - id: k2v2
        k_bits: 2
        v_bits: 2

  - name: kvquant
    common:
      sink_tokens: 5
      calibration_artifact: calibration/kvquant/default/calibration_config.json
    configs:
      - id: kvq4
        bits: 4
      - id: kvq3
        bits: 3
      - id: kvq2
        bits: 2

grid:
  batch_size: [1, 2, 4, 8, 16]
  context_length:
    - 4096
    - 8192
    - 16384
    - 24576
    - 32768
    - 49152
    - 65536
    - 98304
    - 131072

measurement:
  warmup_steps: 64
  measured_steps: 256
  process_replicates: 5
  randomize_within_method_block: true
  rotate_method_block_order: true
  seed: 20260721

validation:
  require_admission_pass: true
  max_cv: 0.03
  allocation_error_limit: 0.01
  reject_silent_backend_fallback: true

outputs:
  format: parquet
  raw_samples: true
  environment_manifest: true
  telemetry: true
  checksum: sha256
  overwrite: false
```

Codex 必须验证这些配置名与当前固定源码一致。若不一致，更新 YAML 和 decision log。

---

# 22. Codex 直接执行的启动指令

从现在开始执行以下顺序：

```text
1. Read this entire file and AGENTS.md.
2. Do not edit benchmark implementation yet.
3. Execute Phase 0 repository and input audit.
4. Produce the Phase 0 report.
5. Continue to Phase 1 only when Phase 0 acceptance criteria pass.
6. Do not skip any admission gate.
7. Treat raw experiment artifacts as immutable.
8. Do not make scientific claims stronger than the current evidence.
9. When source APIs differ from this document, inspect the pinned source,
   record the discrepancy, and create a decision document before adapting.
10. Keep docs/status.md updated after every phase.
```

优先级严格为：

```text
Hardware preflight
→ BF16 baseline
→ Common harness
→ TurboQuant
→ KIVI
→ KVQuant
→ Admission gates
→ Pilot
→ Graph A/B and profiler subset
→ Full scan
→ Modeling
→ Reproduction package
```

不要同时大规模移植三个方法。先让 TurboQuant 跑通整个 Measurement Lane，再接入 KIVI，最后接入 KVQuant。

---

# 23. 首次 Codex 对话可直接粘贴的简短入口

```text
Read CODEX_WORKFLOW.md and AGENTS.md in full. This is a scientific measurement project, not a generic optimization task. Follow the phases and gates exactly. Start with Phase 0 only: inventory the repository, safely unpack and hash /mnt/data/Archive.zip without executing its contents, freeze source inputs, create the status/risk/decision documents, and return the required Phase 0 report. Do not implement or benchmark CUDA kernels yet.
```

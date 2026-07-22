# Codex 后置质量验证主提示词

## 在完成既有性能计划后，对 TurboQuant、KIVI、KVQuant 执行 PPL 与 LongBench 质量准入

> **建议仓库文件名**：`CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md`  
> **适用项目**：单张 NVIDIA RTX PRO 6000 Blackwell 96GB 上的 BF16、TurboQuant-vLLM、KIVI、KVQuant KV-cache quantization 研究  
> **前置主工作流**：`CODEX_WORKFLOW.md`  
> **相关补充文件**：`CODEX_QUALITY_EVALUATION_ADDENDUM.md`  
> **本文件的作用**：允许既有性能与机制实验完整结束，然后以严格、独立、可追溯的 Quality Lane 对每个精确配置执行质量准入。  
> **核心原则**：性能实验可以先完成，但在质量验证通过以前，所有配置只能标记为 `performance_only`，不得被称为“有效部署加速方案”。

---

# 0. 给 Codex 的首次指令

在仓库根目录保存本文件后，向 Codex 发送：

```text
Read CODEX_WORKFLOW.md, CODEX_QUALITY_EVALUATION_ADDENDUM.md, and
CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md in full.

Treat CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md as the authoritative
scheduling and execution contract for post-performance quality validation.

Do not interrupt or mutate an active performance run. First perform only the
pre-registration and integrity steps that are safe while the run is active.
After the existing performance plan is fully closed and checksummed, execute
the quality phases in order. Do not skip gates, do not tune quality thresholds
after viewing method results, and do not modify completed raw performance data.
```

---

# 1. 决策与科研定位

## 1.1 允许先完成性能计划

当前既有计划可以继续完成，包括：

- BF16 baseline；
- TurboQuant、KIVI、KVQuant 的 adapter 集成；
- fixed-length decode scan；
- growing-context decode scan；
- CUDA Graph A/B；
- Nsight Systems 与 Nsight Compute 子集；
- cache bytes、HBM traffic、floor、knee、slope；
- 完整的 `B × L × r` 性能网格；
- provisional performance model。

这批数据仍然可以回答：

```text
decode 实际移动了什么
哪里出现 floor/knee
logical bytes 如何映射到 HBM traffic
相同工作量下 latency 如何随 B、L、r 变化
不同方法的 implementation overhead 是什么
```

## 1.2 但性能结果在质量验证前必须降级标记

所有既有性能结果在 Quality Lane 完成前的状态是：

```yaml
quality_status: unvalidated
claim_eligibility: performance_only
```

不得表述为：

```text
该方法在不损失质量的情况下加速 X 倍
该配置具有部署价值
该配置是最佳量化配置
该压缩率是推荐设置
```

允许表述为：

```text
在尚未进行任务级质量准入的固定工作量实验中，
该精确实现配置的 decode latency / traffic / cache bytes 为……
```

## 1.3 质量验证失败不会自动使全部性能数据作废

必须区分两种失败：

### A. 算法质量失败

例如：

- PPL 明显上升；
- LongBench 下降超过预注册门槛；
- 长上下文能力下降；
- 量化配置本身过于激进。

处理：

```text
性能数据仍然是有效的系统测量数据；
该配置保留在 mechanism / appendix / quality-failing Pareto 区域；
但不能进入质量约束下的主加速结论。
```

### B. 实现正确性失败

例如：

- cache 索引错误；
- GQA head mapping 错误；
- CUDA Graph 与 eager 输出不一致；
- 量化 adapter 与 reference 算法不一致；
- batch 变化导致答案变化；
- 实际没有读取压缩 cache；
- 出现 full-prefix 错误反量化或 stale cache。

处理：

```text
对应 method_config_fingerprint 下的性能数据失效；
修复后必须生成新的 fingerprint；
重新运行所有受影响的 correctness、quality 和 performance points。
```

---

# 2. 调度规则：现在冻结，之后执行

## 2.1 当前性能任务仍在运行时，只允许做这些事情

Codex 可以立即执行：

1. 将本文件加入版本控制；
2. 创建质量合同模板；
3. 固定 benchmark 名称、revision、样本选择规则和门槛；
4. 创建空的 quality 目录结构和 schema；
5. 不读取量化配置的最终质量结果，因为尚未运行；
6. 不改变正在运行的性能代码、配置或 raw artifacts；
7. 不启动占用同一张 GPU 的质量任务。

不得执行：

- 修改当前性能网格；
- 因看到某配置速度快而降低质量门槛；
- 因看到某配置速度慢而跳过它的质量快速筛选；
- 在性能进程仍运行时抢占 GPU；
- 修改已完成 run 的 raw samples 或 manifest；
- 把 provisional 性能排名写成最终结论。

## 2.2 最佳科研做法

即使 Quality Lane 之后才运行，也应当**现在就提交**以下文件：

```text
CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md
configs/quality/quality_contract.template.yaml
docs/decisions/quality_protocol_preregistered.md
```

并记录：

```text
commit SHA
提交时间
制定质量协议时已知的性能信息范围
是否已查看完整方法排名
```

如果完整性能排名已经被研究者查看，也不禁止后续质量验证，但必须在报告中声明：

```text
quality protocol was specified after partial/full performance results were known
```

此时应把质量分析称为 `post-specified confirmatory validation`，而不是完全盲的预注册实验。

---

# 3. 你的角色

你是本项目的：

- Codex research engineer；
- evaluation engineer；
- CUDA correctness reviewer；
- reproducibility maintainer；
- statistical quality-gate implementer。

你的任务不是让所有量化配置“看起来通过”，而是建立一个能够回答以下问题的独立系统：


a. 量化 cache 是否真的参与了质量评测？  
b. next-token probability 是否发生系统性退化？  
c. 长上下文真实任务能力是否发生退化？  
d. 哪些精确配置能够在预注册质量约束下进入性能主结论？

你不得：

- 根据最终结果调整门槛；
- 只挑有利任务；
- 删除不利样本；
- 用普通 full-sequence PPL 冒充 KV-cache PPL；
- 让 LongBench 的答案在压缩 cache 尚未被读取前产生；
- 混用不同 checkpoint、tokenizer 或 method configuration；
- 将 `no statistically significant difference` 解释成 `equivalent quality`；
- 直接修改完成的性能 raw artifacts。

---

# 4. 不可违反的研究不变量

1. **同一个 checkpoint**：性能、PPL 和 LongBench 必须使用同一模型 checkpoint。
2. **同一个 tokenizer revision**：不得静默升级 tokenizer。
3. **同一个 method configuration**：bitwidth、group size、residual window、outlier cap、sink tokens、skip layers、backend 必须完全相同。
4. **同一个 adapter/kernel 语义**：Quality Lane 只能增加评测 harness，不得静默改变性能 hot path。
5. **BF16 配对**：每个量化样本必须和同一个 BF16 样本配对。
6. **质量主实验 batch size 为 1**：另做小型 batch invariance test。
7. **确定性解码**：`temperature=0`、`do_sample=false`、固定 seed。
8. **原始数据 append-only**：禁止删除或覆盖。
9. **固定样本集合**：任何方法都使用完全相同的样本 ID。
10. **固定可行集合**：超长样本在所有方法运行前统一过滤，不能 method-specific truncation。
11. **PPL 必须读取压缩 cache**：普通 full-sequence forward 不合格。
12. **LongBench 的主答案预测前必须至少执行一次量化 decode**。
13. **质量结论使用三状态**：`pass`、`fail`、`inconclusive`。
14. **性能数据与质量状态通过 fingerprint 连接**，不得通过方法名称模糊连接。
15. **实现 bug 与算法质量损失必须分开记录**。

---

# 5. 仓库中新增的结构

在不破坏既有结构的前提下新增：

```text
configs/quality/
├── quality_contract.yaml
├── quality_contract.template.yaml
├── datasets/
│   ├── wikitext2.yaml
│   ├── c4.yaml
│   ├── longbench_e.yaml
│   └── longbench_v2.yaml
├── fixtures/
│   ├── c4_sample_ids.txt
│   ├── ppl_anchor_ids.json
│   ├── longbench_e_sample_ids.json
│   └── longbench_v2_feasible_ids.json
├── prompts/
│   ├── longbench_e_prompt_hashes.json
│   └── longbench_v2_prompt_hashes.json
└── gates/
    └── quality_margins.yaml

src/kvbench/quality/
├── cache_sensitive_runner.py
├── ppl_streaming.py
├── longbench_runner.py
├── prompt_split.py
├── logits_metrics.py
├── batch_invariance.py
├── graph_invariance.py
├── parsers.py
├── bootstrap.py
├── admission.py
└── schemas.py

tests/quality/
├── test_ppl_uses_decode_cache.py
├── test_longbench_uses_decode_cache.py
├── test_prompt_split_token_identity.py
├── test_bf16_pairing.py
├── test_batch_invariance.py
├── test_graph_invariance.py
├── test_dataset_fixture_hashes.py
├── test_quality_contract_immutable.py
└── test_admission_logic.py

analysis/quality/
├── summarize_ppl.py
├── summarize_longbench.py
├── paired_bootstrap.py
├── make_quality_figures.py
└── join_performance_quality.py

docs/quality/
├── protocol.md
├── dataset_manifest.md
├── metric_definitions.md
├── exclusions_policy.md
└── final_report_template.md

artifacts/quality/
└── <quality_run_id>/
    ├── manifest.json
    ├── config_snapshot.yaml
    ├── method_config_fingerprint.json
    ├── raw/
    ├── summaries/
    ├── logs/
    └── checksums.sha256
```

完成的性能数据不允许被修改。质量状态通过单独 sidecar registry 连接：

```text
artifacts/quality_registry/config_quality_status.parquet
```

字段至少包括：

```text
method_config_fingerprint
performance_run_ids
quality_contract_id
q0_status
ppl_status
longbench_e_status
longbench_v2_status
joint_quality_status
claim_eligibility
reason
quality_run_ids
```

---

# 6. Phase QP-0：结束并冻结既有性能计划

只有在以下条件全部满足后才能启动 GPU Quality Lane：

```text
当前性能计划无 active process
所有预定 run 均为 completed / failed / capacity_infeasible
raw samples 已写入
所有 artifacts 已生成 checksum
git working tree 状态已记录
container digest 已记录
GPU hardware manifest 已记录
```

Codex 必须执行：

## 6.1 建立性能冻结清单

生成：

```text
artifacts/performance_freeze/<freeze_id>/performance_inventory.parquet
artifacts/performance_freeze/<freeze_id>/manifest.json
artifacts/performance_freeze/<freeze_id>/checksums.sha256
```

`performance_inventory.parquet` 每行至少包括：

```text
run_id
method
method_config_id
method_config_fingerprint
model_checkpoint
model_revision
tokenizer_revision
adapter_source_hash
kernel_source_hash
kernel_binary_hash
container_digest
git_sha
batch_size
context_length
graph_mode
run_status
raw_artifact_path
raw_artifact_sha256
quality_status = unvalidated
claim_eligibility = performance_only
```

## 6.2 建立冻结 Git tag

建议：

```bash
git tag -a perf-freeze-<YYYYMMDD>-<short_sha> -m "Performance dataset frozen before quality validation"
```

## 6.3 锁定 hot-path 文件

生成：

```text
artifacts/performance_freeze/<freeze_id>/locked_paths.txt
```

至少锁定：

```text
src/kvbench/adapters/
src/kvbench/runtime/
src/kvbench/model/
所有 CUDA / Triton kernel
configs/methods/
configs/models/
```

质量分支允许新增评测文件，但不得修改这些路径。

在每次质量运行前执行：

```bash
git diff --exit-code <perf_freeze_tag> -- $(cat locked_paths.txt)
```

如果有差异：

```text
STOP
标记 code-path mismatch
不得把质量结果与旧性能结果直接连接
```

## 6.4 构建 Quality image

Quality image 应当：

- 以冻结的 measurement image 为基础；
- 保持相同 CUDA、PyTorch、Triton、driver-visible runtime；
- 只增加 dataset、LongBench、evaluation、统计依赖；
- 不重新编译或替换方法 kernel，除非 binary hash 保持一致；
- 记录 base image digest 与 quality image digest。

---

# 7. Phase QP-1：冻结质量合同

在任何量化质量结果产生前，生成正式文件：

```text
configs/quality/quality_contract.yaml
artifacts/quality_contract/<contract_id>/contract_snapshot.yaml
artifacts/quality_contract/<contract_id>/checksums.sha256
```

合同必须由人类明确批准后才可运行正式质量实验。

## 7.1 模型一致性检查

Codex 从性能冻结清单读取：

```text
model checkpoint
revision
tokenizer revision
chat template
max model length
RoPE configuration
weight dtype
```

不得自行从 base 模型切换到 instruct 模型，或反向切换。

如果性能使用 base checkpoint：

- PPL 和 LongBench 仍然使用该 base checkpoint 做配对相对退化；
- 绝对 LongBench 分数可能较低，报告中要说明；
- 如需 Instruct checkpoint，应作为新的独立研究 track，不能为旧性能数据提供质量背书。

## 7.2 配置一致性检查

质量合同列出所有精确配置，例如：

```text
BF16
TurboQuant 4bit_nc
TurboQuant k3v4_nc
TurboQuant 3bit_nc
KIVI k4v4 + fixed group size + fixed residual window
KIVI k2v4 + fixed group size + fixed residual window
KIVI k2v2 + fixed group size + fixed residual window
KVQuant 4bit + fixed sink + fixed outlier cap + frozen quantizer
KVQuant 3bit + fixed sink + fixed outlier cap + frozen quantizer
KVQuant 2bit + fixed sink + fixed outlier cap + frozen quantizer
```

每个配置必须生成唯一：

```text
method_config_fingerprint
```

至少哈希：

```text
method name
bitwidths
group size
residual window
outlier cap
sink tokens
skipped layers
quantizer artifact hash
backend
graph mode
cache layout
model revision
tokenizer revision
adapter source hash
kernel binary hash
```

## 7.3 禁止使用性能排名选择初始质量配置

以下阶段必须覆盖所有精确配置：

```text
Q0 correctness
Fast PPL
```

不得只跑最快配置。

LongBench-E 的进入条件只能是 PPL 质量 Gate，而不是速度排名。

LongBench v2 finalist 的选择规则必须预注册为：

```text
每种方法选择：
1. PPL + LongBench-E 通过者中质量最好的配置；
2. PPL + LongBench-E 通过者中 r_alloc 最大的配置。
若二者相同，则该方法只运行一个 finalist。
选择时不得使用 speedup。
```

---

# 8. 一个关键问题：质量评测必须真正读取压缩 KV cache

这是本流程最重要的工程检查。

## 8.1 为什么普通 PPL 不合格

以下代码通常不合格：

```python
outputs = model(input_ids, labels=input_ids)
```

原因：

- full-sequence attention 可能直接使用当前 forward 内的 BF16 K/V；
- cache 可能只被写入而未被重新读取；
- 得到的 PPL 可能完全没有测量 KV quantization。

## 8.2 为什么普通一 token LongBench-v2 答案也可能不合格

在常见 serving 路径中：

```text
完整 prompt prefill
→ prefill logits 直接预测第一个输出 token
```

第一个输出 token 可能在压缩 cache 被 decode kernel 读取以前就产生。

如果 LongBench-v2 的答案仅为 `A/B/C/D` 一个 token，那么普通协议可能几乎没有测试 KV cache quantization。

因此必须使用下面的 **cache-sensitive execution protocol**。

---

# 9. Cache-sensitive PPL protocol

## 9.1 定义

使用 anchor-based、teacher-forced、逐 token decode-conditioned PPL。

对 token stream：

\[
x_1,\ldots,x_L,x_{L+1},x_{L+2},\ldots,x_{L+H+1}
\]

执行：

1. 用 `x_1...x_L` prefill；
2. 确认 K/V 按该方法真实规则写入压缩 cache；
3. 将 `x_{L+1}` 作为 **不计分 burn-in decode token**；
4. 该 decode step 必须读取压缩 cache，并产生预测 `x_{L+2}` 的 logits；
5. 开始对 `x_{L+2}...x_{L+H+1}` 累积 NLL；
6. 每一步输入真实 token，而不是模型自己生成的 token。

伪代码：

```python
prefix = ids[:L]
burn_in = ids[L]
targets = ids[L + 1 : L + 1 + H]

cache = method.prefill(prefix)
logits, cache = method.decode_one(burn_in, cache)

losses = []
for target in targets:
    losses.append(cross_entropy(logits, target))
    logits, cache = method.decode_one(target, cache)
```

注意最后一次多执行出的 logits 不计分。

## 9.2 必须证明压缩 cache 被读取

测试至少包括：

- method-specific decode kernel launch counter；
- NVTX range；
- debug 模式下的 adapter read counter；
- 对 cache 内容做受控扰动时 logits 应改变；
- 将 compressed cache read path 禁用时测试必须失败；
- 对 BF16 与 quantized 使用相同 prefill/decode split。

新增测试：

```text
test_ppl_uses_decode_cache.py
```

该测试不能仅依赖函数名，必须证明输出实际依赖 cache 内容。

## 9.3 数据集

主要使用：

```text
WikiText-2 test
固定 revision 的 C4 validation 子集
```

作用：

- WikiText-2：和已有 KV quantization 文献常见协议对齐，适合 regression；
- C4：覆盖更广的文本分布，降低单数据集偶然性。

固定：

```text
dataset revision
split
sample IDs
sample order
text hashes
tokenizer revision
BOS/EOS 规则
document separator
anchor positions
scored horizon H
```

## 9.4 文档边界

若将文档拼接成长 stream：

```text
doc_1 + EOS + doc_2 + EOS + ...
```

规则：

- anchor 不得落在文档边界后的忽略区；
- 默认忽略新文档起始后的 32 个 token；
- 所有方法使用完全相同的 token stream；
- 保存 stream SHA256。

## 9.5 PPL 长度设计

### Fast PPL：所有配置

```yaml
prefix_lengths:
  - 4096
  - 24576
  - 32768
  - 65536
anchors_per_length: 16
scored_tokens_per_anchor: 128
burn_in_decode_tokens: 1
```

目的：

- 快速发现灾难性质量损失；
- 覆盖 knee 前、附近和后方；
- 淘汰明显不合格配置。

### Full PPL：Fast PPL 通过者

```yaml
prefix_lengths:
  - 4096
  - 16384
  - 24576
  - 28672
  - 32768
  - 65536
  - 98304
  - 130560
anchors_per_length: 64
scored_tokens_per_anchor: 256
burn_in_decode_tokens: 1
```

必须检查：

```text
prefix_length + burn_in + scored_horizon + safety_margin <= max_model_length
```

若模型最大长度不同，按合同统一调整，并在任何方法运行前冻结。

## 9.6 PPL 主要统计量

主要分析 NLL 差：

\[
\Delta \mathrm{NLL}_{m,c,L}
=
\mathrm{NLL}_{m,c,L}
-
\mathrm{NLL}_{BF16,L}.
\]

PPL 相对变化：

\[
R_{PPL}
=
\exp(\Delta \mathrm{NLL})-1.
\]

所有 CI 以 anchor/document 为重采样单位，禁止把高度相关的单 token 当成独立样本。

## 9.7 PPL Gate

默认预注册门槛：

```yaml
global_relative_ppl_increase_max: 0.01
length_bucket_review_threshold: 0.02
length_bucket_hard_fail: 0.05
```

等价全局 NLL margin：

\[
\delta_{NLL}=\log(1.01).
\]

状态：

### PASS

- WikiText-2 与 C4 的主要 aggregate 上，95% paired bootstrap CI 上界均不超过 `log(1.01)`；
- 任一长度桶没有超过 5% PPL hard fail；
- 无 NaN/Inf/cache corruption。

### FAIL

满足任一：

- 95% CI 下界超过全局 margin；
- 任一长度桶 PPL 增幅超过 5%；
- 发生 NaN/Inf/非法 cache 状态；
- PPL runner 未真实读取压缩 cache。

### INCONCLUSIVE

- CI 跨过 margin；
- 样本不足；
- 两数据集结论不一致且无预注册裁决规则。

处理：

```text
增加预注册数量的 anchors；
不得仅选择有利数据集；
不得直接宣称质量相同。
```

---

# 10. Cache-sensitive LongBench protocol

## 10.1 执行分割原则

对 LongBench 完整 tokenized prompt：

\[
p_1,\ldots,p_N
\]

选择固定数量 `D` 个末尾 prompt token 作为 decode-conditioning suffix：

```text
prefill tokens: p_1 ... p_(N-D)
decode-conditioning tokens: p_(N-D+1) ... p_N
answer generation starts after p_N
```

所有 token 内容、顺序和官方 prompt 完全不变，只改变执行路径。

默认：

```yaml
decode_conditioning_tokens: 16
```

伪代码：

```python
prompt_ids = tokenizer(full_official_prompt)
prefill_ids = prompt_ids[:-D]
conditioning_ids = prompt_ids[-D:]

cache = method.prefill(prefill_ids)
for token in conditioning_ids:
    logits, cache = method.decode_one(token, cache)

# 此时 logits 预测第一个答案 token，且之前已读取压缩 cache。
answer = generate_from_logits_and_cache(logits, cache, generation_config)
```

BF16 baseline 必须使用完全相同的 split。

## 10.2 token identity test

新增测试：

```text
test_prompt_split_token_identity.py
```

必须证明：

```python
concat(prefill_ids, conditioning_ids) == original_prompt_ids
```

不得：

- 改写 prompt；
- 插入语义不同的 dummy text；
- 删除官方指令；
- 为不同方法选择不同 split；
- 在方法之间改变 chat template。

## 10.3 证明答案依赖压缩 cache

新增：

```text
test_longbench_uses_decode_cache.py
```

至少检查：

- 在答案 logits 产生前，method-specific decode kernel 已执行；
- cache read counter > 0；
- 受控修改旧 cache 会改变答案 logits；
- 将 `D=0` 与 `D=16` 区分开；
- runner manifest 保存实际 `D`；
- 对 KIVI 的 residual window、KVQuant sink/outlier、TurboQuant compressed slot 使用真实实现路径。

## 10.4 Native-serving secondary protocol

为了和真实 serving 行为对齐，finalist 另跑一个 secondary protocol：

```text
完整 prompt 一次性 prefill
直接生成答案
```

命名：

```text
longbench_native_prefill
```

主质量 Gate 使用：

```text
longbench_cache_sensitive
```

原因：

- cache-sensitive protocol 检验 KV quantization 本身；
- native protocol 检验实际 serving 中的最终行为；
- 两者不能混为同一指标。

---

# 11. LongBench-E：主要广覆盖任务 Gate

## 11.1 任务集合

使用固定 revision 的 LongBench-E 13-task suite：

```text
qasper
multifieldqa_en
hotpotqa
2wikimqa
gov_report
multi_news
trec
triviaqa
samsum
passage_count
passage_retrieval_en
lcc
repobench-p
```

覆盖：

- 单文档 QA；
- 多文档 QA；
- 摘要；
- few-shot；
- 合成检索；
- 代码任务。

## 11.2 进入条件

只有 Full PPL 状态为 `pass` 的配置进入 LongBench-E。

不得根据速度选择。

## 11.3 prompt 与生成参数

固定：

```text
official task prompt
model-specific chat template
few-shot examples
task-specific max_new_tokens
stop conditions
output parser
seed
decode_conditioning_tokens = 16
```

使用：

```yaml
temperature: 0.0
top_p: 1.0
do_sample: false
batch_size: 1
```

## 11.4 超长输入处理

在任何方法运行前：

1. 使用同一 tokenizer 计算完整 prompt token length；
2. 定义：

\[
N + G + M \le L_{max},
\]

其中：

- `N`：prompt tokens；
- `G`：任务生成预算；
- `M`：安全 margin。

3. 形成统一 feasible sample set；
4. 保存 sample IDs 和 hash；
5. 主结果不做 method-specific truncation。

若需要复现官方截断结果，作为 secondary report，不能替代主 feasible-set 结果。

## 11.5 指标

每个任务使用官方 metric，例如：

- F1；
- exact match；
- ROUGE-L；
- classification accuracy；
- code similarity；
- task-specific metric。

主要 aggregate 使用 task macro average：

\[
Q_{LB-E}=\frac{1}{13}\sum_{d=1}^{13}Q_d.
\]

禁止直接 sample-micro average 取代 task macro average。

报告：

```text
per-task score
category macro score
overall task-macro score
per-length bucket score
invalid output rate
truncation rate
output token count
BF16 paired difference
```

## 11.6 配对统计

使用 task-stratified paired bootstrap：

- 同一个 sample 的 BF16 与 quantized score 配对；
- 在 task 内重采样 sample；
- 再对 task 做 macro aggregation；
- 生成 95% CI。

定义下降：

\[
D_{LB-E}=Q_{BF16}-Q_{quant}.
\]

## 11.7 LongBench-E Gate

默认：

```yaml
macro_score_drop_max_points: 2.0
category_drop_review_points: 3.0
category_drop_hard_fail_points: 5.0
invalid_output_increase_max_pp: 1.0
```

### PASS

- 95% CI 上界不超过 2.0 score points；
- 无 category hard fail；
- invalid output 增加不超过 1 pp。

### FAIL

- 95% CI 下界超过 2.0 points；或
- 任一 category 下降超过 5 points；或
- invalid output 增加超过 1 pp；或
- cache-sensitive protocol 未被执行。

### INCONCLUSIVE

- CI 跨过门槛；
- 样本不足；
- parser 问题影响大量样本。

不得以“没有显著下降”替代 non-inferiority 结论。

---

# 12. LongBench v2：32K–128K 现实长上下文主 Gate

## 12.1 finalist 选择

每种方法从 `PPL pass + LongBench-E pass` 配置中选择：

1. 质量最好的配置；
2. `r_alloc` 最大的配置。

不得使用性能速度排序。

## 12.2 可行样本集合

使用与性能模型相同 checkpoint/tokenizer 重新计算 token length。

长度桶：

```text
8K–16K
16K–32K
32K–64K
64K–128K
```

定义：

```yaml
max_input_plus_generation_tokens: model_max_length - safety_margin
safety_margin: 128
generation_budget_no_cot: 8
```

在所有方法运行前生成：

```text
configs/quality/fixtures/longbench_v2_feasible_ids.json
```

不得：

- 只删除某种方法失败的样本；
- 为不同方法使用不同截断；
- 将 >128K 样本混入主 128K 结果。

## 12.3 主模式：no-CoT cache-sensitive

```yaml
cot: false
temperature: 0.0
do_sample: false
max_new_tokens: 8
decode_conditioning_tokens: 16
valid_answers: [A, B, C, D]
```

主结果是：

```text
cache-sensitive no-CoT accuracy
```

## 12.4 secondary：CoT finalist stress test

每种方法最终推荐配置再运行：

```text
LongBench v2 CoT subset
```

用途：

- 检查长生成过程中量化误差是否累积；
- 不作为所有配置的快速 Gate；
- 单独报告输出长度和截断率。

## 12.5 指标

报告：

```text
overall accuracy
per-category accuracy
per-token-length-bucket accuracy
invalid output rate
BF16-correct retention
correct→wrong count
wrong→correct count
paired 95% CI
McNemar discordant-pair table
```

定义：

\[
Retention=
\frac{N(BF16\ correct\land Quant\ correct)}{N(BF16\ correct)}.
\]

量化配置不能通过“偶然修复部分 BF16 错误”掩盖大量 `correct→wrong`。

## 12.6 Gate

默认：

```yaml
accuracy_drop_max_pp: 2.0
any_length_bucket_drop_max_pp: 5.0
any_category_drop_hard_fail_pp: 5.0
invalid_output_increase_max_pp: 1.0
baseline_correct_retention_min: 0.95
```

### PASS

- 95% paired CI 上界不超过 2 pp；
- 任一长度桶下降不超过 5 pp；
- 任一类别下降不超过 5 pp；
- invalid rate 增加不超过 1 pp；
- baseline-correct retention 点估计至少 95%，并报告 CI。

### FAIL

- 主要 accuracy 非劣性明确失败；或
- 任一 guardrail hard fail；或
- cache-sensitive path 未执行。

### INCONCLUSIVE

- CI 跨过门槛；
- feasible sample 数不足；
- retention CI 过宽且主要 accuracy 也不明确。

---

# 13. Phase QP-2：Q0 实现与执行保真度

在 PPL 和 LongBench 前，对所有配置运行 Q0。

## 13.1 cache 与 attention

检查：

```text
cache encode/decode round-trip
K/V MSE and cosine
attention output relative error
attention score divergence
NaN/Inf
out-of-bounds / sanitizer
cache pointer stability
no stale pages
GQA head mapping
```

## 13.2 logits

在固定 prompts、多个长度上报告：

```text
next-token KL
logit cosine
top-1 agreement
top-5 overlap
rank of BF16 top-1 under quantized logits
```

长度至少：

```text
512
4K
16K
24K
32K
64K
near max length
```

## 13.3 teacher-forced 与 free-running

### Teacher-forced

用于隔离条件分布偏差。

### Free-running greedy

记录：

```text
first divergence token
generation edit distance
repetition rate
invalid text rate
termination behavior
```

不把“首次 token 分叉”本身作为质量 hard fail，因为确定性生成对细小 logit 变化敏感；重点用于诊断。

## 13.4 Graph invariance

对同一配置比较：

```text
eager
CUDA Graph
```

要求：

- 同一输入的 logits 在数值容差内一致；
- task answer 不发生系统性变化；
- cache-sensitive kernel 均执行。

## 13.5 Batch invariance

主质量使用 `B=1`。

另取固定 100 个样本：

```yaml
batch_sizes: [1, 4, 8]
```

检查：

- tokenized input 一致；
- padding 不改变有效答案；
- logits 差异在容差内；
- A/B/C/D 选择一致；
- invalid rate 不随 batch 系统性增加。

若失败，视为 implementation bug，而不是正常质量损失。

## 13.6 Q0 hard fail

任一以下情况立即失败：

```text
NaN/Inf
非法 token 或 cache corruption
batch-dependent semantic failure
graph/eager 大规模不一致
reference mismatch 超过预定义容差
答案 logits 产生前未读取 compressed cache
方法配置与性能 fingerprint 不一致
```

---

# 14. 质量实验的逐级执行顺序

严格按顺序：

```text
QP-0 关闭并冻结性能计划
QP-1 冻结质量合同
Q0 所有配置 correctness
Q1A 所有配置 Fast PPL
Q1B Fast PPL 通过者 Full PPL
Q2A Full PPL 通过者 LongBench-E
Q2B 每方法 finalist LongBench v2 no-CoT
Q2C 最终推荐配置 LongBench v2 CoT stress
Q3 Joint admission report
Q4 Quality-constrained performance join
```

不得跳过前一 Gate 直接把某个最快配置送入最终质量报告。

---

# 15. 联合质量准入逻辑

每个精确 configuration 的状态：

| Q0 | PPL | LongBench-E | LongBench-v2 | 最终状态 |
|---|---|---|---|---|
| fail | 任意 | 任意 | 任意 | implementation_invalid |
| pass | fail | 未运行 | 未运行 | quality_fail_ppl |
| pass | pass | fail | 未运行 | quality_fail_longbench_e |
| pass | pass | pass | pass | quality_pass |
| pass | pass | pass | inconclusive | quality_inconclusive |
| pass | inconclusive | 未运行 | 未运行 | quality_inconclusive |

对未进入 LongBench-v2 的非-finalist：

```text
如果 Q0 + PPL + LongBench-E 均通过，
状态为 quality_pass_primary，
但不能声称已通过 LongBench-v2 finalist validation。
```

建议最终状态：

```text
implementation_invalid
quality_fail
quality_inconclusive
quality_pass_primary
quality_pass_finalist
```

---

# 16. 和既有性能数据连接

通过 `method_config_fingerprint` 连接：

```text
performance_inventory
JOIN
config_quality_status
ON method_config_fingerprint
```

生成：

```text
artifacts/joint_results/<joint_id>/performance_quality_join.parquet
artifacts/joint_results/<joint_id>/admission_table.csv
artifacts/joint_results/<joint_id>/pareto_points.parquet
```

不得通过：

```text
method == "KIVI"
```

这种粗粒度名称连接。

必须是：

```text
KIVI k2v4, group=32, residual=32, exact kernel hash, exact model revision
```

---

# 17. 质量约束下的主性能结论

定义可用配置集合：

\[
\mathcal C_{quality}=
\{c:\ c\text{ 通过预注册质量 Gate}\}.
\]

主结果：

\[
S^*(B,L)=\max_{c\in\mathcal C_{quality}} S_c(B,L).
\]

分别报告：

1. 所有 implementation-valid 配置的性能；
2. 质量失败配置；
3. 质量通过配置；
4. 质量约束下的 Pareto frontier。

主论文 headline 不得来自 `quality_fail` 配置。

允许保留的机制结论：

```text
某配置移动的物理字节
某配置的 knee
某配置的 launch/traffic 行为
某配置即使质量失败也可用于解释 bitwidth→traffic→time 的链条
```

但必须明确标记：

```text
not quality-eligible
```

---

# 18. 质量发现后的重跑规则

## 18.1 不需要重跑性能的情况

- PPL/LongBench 证明算法质量不达标；
- 质量 runner 自身只增加数据加载、prompt、统计代码；
- adapter、kernel、cache layout、graph mode 没有变化；
- exact fingerprint 与性能冻结清单一致。

处理：

```text
保留性能数据
更新 sidecar quality registry
从质量约束主结果中排除
```

## 18.2 必须重跑性能的情况

- 修改 adapter；
- 修改 kernel；
- 修改 quantizer artifact；
- 修改 cache layout；
- 修改 graph mode；
- 修复 GQA indexing；
- 修复 cache update；
- 发现旧性能 runner 实际没有执行预期方法；
- 任何 locked hot-path hash 改变。

处理：

```text
生成新 method_config_fingerprint
旧数据标记 superseded_or_invalid
重新运行受影响的 smoke、correctness、quality、performance
不得覆盖旧 artifacts
```

---

# 19. 统计要求

## 19.1 所有比较配对

PPL：同一 anchor、同一目标 tokens。  
LongBench：同一 sample、同一 prompt、同一 parser。

## 19.2 重采样单位

- PPL：anchor/document；
- LongBench-E：task-stratified sample；
- LongBench-v2：sample；
- 禁止把单 token 当成完全独立样本。

## 19.3 三状态判定

必须输出：

```text
pass
fail
inconclusive
```

`p > 0.05` 不等于 `pass`。

## 19.4 不允许的分析

- 看到结果后修改 margin；
- 只报告平均、不报告长度桶；
- 只报告 PPL、不报告 NLL；
- 只报告 LongBench 总平均、不报告任务和长度；
- 删除 BF16 正确但 quantized 错误的样本；
- 只运行量化方法擅长的任务；
- 以输出长度变短造成的低成本冒充质量保持。

---

# 20. 推荐的 `quality_contract.yaml`

```yaml
quality_contract:
  id: kv_quality_post_perf_v1
  status: requires_human_approval

  provenance:
    performance_freeze_tag: PERF_FREEZE_TAG
    performance_freeze_manifest: PATH_TO_MANIFEST
    protocol_commit_sha: TO_BE_FILLED
    performance_results_known_when_specified: true_or_false

  model:
    checkpoint: READ_FROM_PERFORMANCE_MANIFEST
    revision: READ_FROM_PERFORMANCE_MANIFEST
    tokenizer_revision: READ_FROM_PERFORMANCE_MANIFEST
    chat_template_hash: READ_FROM_PERFORMANCE_MANIFEST
    max_model_length: READ_FROM_PERFORMANCE_MANIFEST
    require_same_checkpoint_as_performance: true

  execution:
    quality_batch_size: 1
    temperature: 0.0
    top_p: 1.0
    do_sample: false
    seed: 20260722
    primary_protocol: cache_sensitive
    secondary_protocol: native_prefill_finalists_only
    decode_conditioning_tokens: 16
    require_compressed_cache_read_assertion: true

  configurations:
    source: performance_inventory
    require_exact_method_config_fingerprint: true
    run_q0_for_all: true
    run_fast_ppl_for_all: true

  ppl:
    mode: incremental_teacher_forcing
    burn_in_decode_tokens: 1
    datasets:
      - name: wikitext-2-raw-v1
        split: test
        revision: PINNED
      - name: c4
        split: validation
        revision: PINNED
        sample_ids_file: configs/quality/fixtures/c4_sample_ids.txt

    document_boundary:
      separator: eos
      ignored_tokens_after_boundary: 32

    fast:
      prefix_lengths: [4096, 24576, 32768, 65536]
      anchors_per_length: 16
      scored_tokens_per_anchor: 128

    full:
      prefix_lengths: [4096, 16384, 24576, 28672, 32768, 65536, 98304, 130560]
      anchors_per_length: 64
      scored_tokens_per_anchor: 256

    primary_metric: delta_nll
    report:
      - nll
      - delta_nll
      - ppl
      - relative_ppl_change
      - per_length
      - paired_ci

  longbench_e:
    revision: PINNED
    cache_sensitive: true
    decode_conditioning_tokens: 16
    official_prompts: true
    official_metrics: true
    truncate_overlength_primary: false
    tasks:
      - qasper
      - multifieldqa_en
      - hotpotqa
      - 2wikimqa
      - gov_report
      - multi_news
      - trec
      - triviaqa
      - samsum
      - passage_count
      - passage_retrieval_en
      - lcc
      - repobench-p
    primary_aggregation: task_macro_average

  longbench_v2:
    revision: PINNED
    cache_sensitive: true
    decode_conditioning_tokens: 16
    cot_primary: false
    cot_finalist_stress: true
    max_new_tokens_no_cot: 8
    valid_answers: [A, B, C, D]
    truncate_overlength_primary: false
    token_length_buckets:
      - [8192, 16384]
      - [16384, 32768]
      - [32768, 65536]
      - [65536, 131072]
    finalist_rule:
      - best_quality_among_primary_passers
      - highest_r_alloc_among_primary_passers
      - do_not_use_speed_for_selection

  invariance:
    graph_eager_sample_count: 100
    batch_sample_count: 100
    batch_sizes: [1, 4, 8]

  gates:
    ppl:
      global_relative_increase_max: 0.01
      bucket_review_threshold: 0.02
      bucket_hard_fail: 0.05

    longbench_e:
      macro_score_drop_max_points: 2.0
      category_review_points: 3.0
      category_hard_fail_points: 5.0
      invalid_output_increase_max_pp: 1.0

    longbench_v2:
      accuracy_drop_max_pp: 2.0
      any_length_bucket_drop_max_pp: 5.0
      any_category_drop_max_pp: 5.0
      invalid_output_increase_max_pp: 1.0
      baseline_correct_retention_min: 0.95

    decision_states: [pass, fail, inconclusive]

  data_policy:
    raw_append_only: true
    no_selective_reruns: true
    record_all_failures: true
    record_all_exclusions: true
    checksum_all_artifacts: true
```

Codex 必须用实际性能 manifest 填充 `READ_FROM_PERFORMANCE_MANIFEST`，不得猜测。

---

# 21. Makefile / CLI 接口

新增：

```bash
make quality-preregister
make performance-freeze
make quality-preflight
make quality-q0
make quality-ppl-fast
make quality-ppl-full
make quality-longbench-e
make quality-longbench-v2
make quality-longbench-v2-cot
make quality-invariance
make quality-admission
make quality-join-performance
make quality-report
```

示例：

```bash
python -m kvbench.quality.cache_sensitive_runner \
  --contract configs/quality/quality_contract.yaml \
  --stage ppl_fast \
  --method-config all
```

所有正式命令必须读取版本化合同，禁止临时在命令行覆盖关键门槛和样本集合。

允许命令行覆盖：

```text
output directory
worker count
resume from completed atomic unit
logging verbosity
```

不允许覆盖：

```text
dataset revision
sample IDs
quality margins
model checkpoint
method configuration
prompt template
decode conditioning tokens
```

---

# 22. Codex issue 拆分

建议每个 issue 一个可审查 PR：

```text
Q00 Freeze performance inventory and hashes
Q01 Add quality contract schema and immutability checks
Q02 Add dataset fixture pinning and hashes
Q03 Implement cache-sensitive prompt split
Q04 Implement cache-sensitive streaming PPL
Q05 Add proof that PPL reads compressed cache
Q06 Implement Q0 logit/cache correctness suite
Q07 Implement Fast PPL runner and report
Q08 Implement Full PPL paired bootstrap and Gate
Q09 Integrate LongBench-E official tasks and metrics
Q10 Add cache-sensitive LongBench execution
Q11 Implement LongBench-E paired task-stratified bootstrap
Q12 Build LongBench-v2 tokenizer-based feasible set
Q13 Implement LongBench-v2 no-CoT and retention metrics
Q14 Add graph/eager and batch invariance suites
Q15 Implement joint admission registry
Q16 Join quality status with frozen performance data
Q17 Generate quality-constrained Pareto figures
Q18 Write final reproducibility report
```

不得让单个 Codex task 同时：

```text
重写 adapter + 改 benchmark + 跑 full quality + 解释论文结论
```

---

# 23. 每个 Quality PR 的模板

```markdown
## Scope

## Frozen performance reference
- Performance freeze tag:
- Performance manifest:
- Locked hot-path diff clean: yes/no

## Quality contract
- Contract ID:
- Contract hash:
- Human-approved: yes/no

## Cache-sensitive evidence
- Compressed-cache read assertion:
- Decode kernel trace:
- Prompt token identity:

## Correctness
- Numerical tests:
- Graph/eager:
- Batch invariance:
- Compute Sanitizer:

## Dataset integrity
- Revision:
- Sample fixture hash:
- Prompt hash:

## Commands run

## Raw artifacts

## Known limitations

## Does this change a method adapter or kernel?
- [ ] No
- [ ] Yes — old performance data must not be reused
```

---

# 24. 运行与恢复规则

## 24.1 原子单位

建议最小原子单位：

```text
method_config × dataset × length_bucket × stage
```

完成后：

- 原子写入结果；
- 生成 checksum；
- 标记 completed；
- 支持从未完成单位恢复。

## 24.2 不允许选择性重跑

若出现失败：

```text
记录 failed 状态和完整 stderr
按预注册 retry policy 自动重试固定次数
不得只重跑分数低的配置
不得保留多次结果中最好的一次
```

## 24.3 retry policy

建议：

```yaml
max_retries: 1
retry_only_for:
  - infrastructure_error
  - transient_io_error
  - process_crash_without_output
never_retry_for:
  - low_quality_score
  - valid_model_output
  - quality_gate_failure
```

---

# 25. 必须生成的报告

## 25.1 `QUALITY_VALIDATION_REPORT.md`

至少包括：

1. 性能冻结信息；
2. 质量合同与是否后置指定；
3. 模型和 tokenizer；
4. exact method configurations；
5. cache-sensitive PPL 设计；
6. cache-sensitive LongBench 设计；
7. Q0 correctness；
8. WikiText-2 / C4 NLL 与 PPL；
9. LongBench-E task/category/length 结果；
10. LongBench-v2 accuracy/retention/length 结果；
11. batch 和 graph invariance；
12. pass/fail/inconclusive 表；
13. 与性能数据的 fingerprint join；
14. 哪些配置可进入主结论；
15. 哪些性能数据因 implementation bug 失效；
16. 所有限制。

## 25.2 结果表

```text
quality_admission_table.csv
ppl_summary.parquet
longbench_e_summary.parquet
longbench_v2_summary.parquet
batch_invariance_summary.parquet
graph_invariance_summary.parquet
performance_quality_join.parquet
```

## 25.3 图

至少：

### 图 Q1：PPL/NLL 随 context length

横轴：`L`  
纵轴：`ΔNLL` 或 relative PPL increase。

### 图 Q2：LongBench 质量保留随长度

横轴：长度桶  
纵轴：quantized − BF16 score。

### 图 Q3：BF16-correct retention

按方法、配置、长度报告。

### 图 Q4：质量约束下的性能 Pareto

点包含：

```text
decode speedup
r_alloc
LongBench drop
PPL increase
quality status
```

### 图 Q5：所有配置的 admission map

```text
quality pass
quality fail
inconclusive
implementation invalid
```

---

# 26. Codex 阶段汇报格式

每完成一个阶段，Codex 输出：

```markdown
## Stage completed

## Frozen inputs

## Commands executed

## Tests passed

## Raw artifacts and checksums

## Gate result
- pass / fail / inconclusive

## Configurations advancing

## Configurations stopped

## Implementation bugs found

## Does any prior performance data require rerun?

## Next authorized stage
```

不得只说：

```text
looks good
quality is similar
no significant difference
```

---

# 27. 开始执行时的完整 Codex Prompt

当既有性能计划全部结束后，向 Codex 发送：

```text
Read CODEX_WORKFLOW.md, CODEX_QUALITY_EVALUATION_ADDENDUM.md, and
CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md in full.

The previously scheduled performance and mechanism plan has now finished.
Begin the post-performance quality-validation workflow.

First:
1. Verify there is no active performance process.
2. Freeze and checksum the complete performance inventory.
3. Create the performance freeze tag and locked hot-path hash list.
4. Populate quality_contract.yaml from the frozen manifest without guessing.
5. Verify that the quality branch has not changed any adapter, runtime, model,
   CUDA, Triton, cache-layout, or method-config path relative to the frozen
   performance tag.
6. Materialize and hash all dataset fixtures and prompt templates.
7. Produce the quality contract summary and stop at the human-approval gate.

After the contract is approved, execute in this exact order:
Q0 correctness for all exact configurations;
Fast cache-sensitive PPL for all exact configurations;
Full cache-sensitive PPL for Fast-PPL passers;
LongBench-E for Full-PPL passers;
LongBench-v2 no-CoT for pre-registered per-method finalists;
LongBench-v2 CoT stress for final recommended configurations;
graph/eager and batch invariance;
joint quality admission;
quality-constrained performance join and Pareto report.

Critical requirements:
- Ordinary full-sequence PPL is forbidden.
- PPL must use a prefilled compressed cache, one unscored decode burn-in token,
  and teacher-forced one-token decode scoring.
- LongBench must preserve the exact official prompt tokens but feed the final
  16 prompt tokens through the decode path before answer generation, so the
  answer logits depend on the compressed cache.
- BF16 must use the identical execution split.
- Do not select configurations for quality based on speed.
- Do not change quality margins after seeing results.
- Do not modify completed raw performance artifacts.
- If a method implementation bug is found, invalidate the affected frozen
  fingerprint and report exactly which performance points require rerun.
- If an algorithmic quality Gate fails without an implementation bug, retain
  the performance data as performance-only but exclude it from quality-eligible
  claims.
- Use pass/fail/inconclusive, paired confidence intervals, and append-only raw
  outputs.

Do not begin a later stage until the preceding Gate and artifacts are complete.
```

---

# 28. Definition of Done

本后置质量流程只有在以下条件全部满足时才算完成：

- [ ] 完整性能计划被冻结并有 checksum；
- [ ] 质量协议有固定 commit、contract ID 和 hash；
- [ ] 模型、tokenizer、adapter、kernel 与性能 fingerprint 一致；
- [ ] 所有配置完成 Q0；
- [ ] 所有配置完成 Fast PPL；
- [ ] 通过者完成 Full PPL；
- [ ] Full PPL 通过者完成 LongBench-E；
- [ ] 每种方法的预注册 finalist 完成 LongBench-v2；
- [ ] cache-sensitive PPL 与 LongBench 均有“确实读取压缩 cache”的自动测试；
- [ ] 完成 graph/eager 与 batch invariance；
- [ ] 每个配置有 pass/fail/inconclusive 或 implementation-invalid 状态；
- [ ] 质量状态通过 exact fingerprint 连接到性能数据；
- [ ] 发现的实现 bug 已明确列出需重跑的性能范围；
- [ ] 质量失败配置仍被保留但不进入主有效加速结论；
- [ ] 生成质量约束下的 speedup/Pareto 结果；
- [ ] 所有 raw artifacts append-only 且有 checksum；
- [ ] 从固定环境可复现质量报告与图。

---

# 29. 最终研究表述约束

只有 `quality_pass_finalist` 配置可以支持：

```text
在预注册 PPL 和 LongBench 质量约束下，该配置实现了……
```

`quality_pass_primary` 可以支持：

```text
该配置通过了 PPL 与 LongBench-E 主质量 Gate，
但尚未完成 LongBench-v2 finalist validation。
```

`quality_fail` 只能支持：

```text
该配置展示了某种性能/流量机制，
但质量退化超过预注册门槛，不属于有效部署加速。
```

`implementation_invalid` 不得支持任何性能或质量结论，直至修复并重跑。

最终论文的核心结果应写成：

\[
\boxed{
\text{在给定质量损失约束下，KV cache compression 能提供多少 latency 与 capacity 收益？}
}
\]

而不是只写：

```text
哪一个低比特配置最快。
```


本文件保留质量评测的科学要求、指标定义和分析原则，
但不再是当前项目的质量执行顺序合同。
当前项目采用：
完成既有性能实验；
校验并冻结性能数据；
执行后置质量验证；
将质量结果与性能结果按 exact configuration fingerprint 连接。


运行顺序、数据冻结、PPL/LongBench 启动条件和重跑规则，
以 CODEX_POST_PERFORMANCE_QUALITY_VALIDATION.md 为最高优先级。
如两个文件发生冲突，不执行本文件中要求在性能实验结束前
启动质量验证的旧规则。
Codex Quality-Evaluation Addendum for KV Cache Quantization
适用项目：RTX PRO 6000 Blackwell 单卡上的 BF16、TurboQuant-vLLM、KIVI、KVQuant 统一测量项目  
插入位置：将本文件作为主工作流的强制补充，逻辑上插入在 Phase 12：统一 Admission Gates 与 Phase 13：Pilot Scan 之间。  
优先级：本文件中的质量规则高于“先跑完性能网格”的便利性。任何未通过质量 Gate 的配置，都不得被描述为有效加速方案。



0. 为什么必须新增 Quality Lane
当前工作流能够证明：
CUDA kernel 可以运行；
adapter 与小张量 reference fixture 基本一致；
cache 字节、HBM 流量和 decode latency 可以测量；
性能函数可以针对 B,L,r 拟合。


但这些条件不能证明：
量化后的模型仍然具有可接受的语言建模、长上下文检索、长上下文推理和生成质量。


小张量 golden test 只能发现实现错误，不能排除如下情况：
每层都只有很小的误差，但误差在长序列和多步生成中累积；
next-token logits 发生足以改变生成轨迹的微小排序变化；
NIAH 仍然成功，但多跳、聚合和全上下文推理显著下降；
平均 perplexity 基本不变，但某些长度区间或任务类型崩溃；
CUDA Graph、batch layout 或长上下文 backend 在特定形状下产生错误输出；
量化配置速度更快，但质量损失大到没有实际意义。


因此项目最终研究对象必须由：
$$
T_m(B,L,r)
$$



扩展为：
$$
\boxed{ \left(T_m(B,L,r),\;M_m(B,L,r),\;Q_m(L,r)\right) }
$$



其中：
T_m：时间；
M_m：容量或物理字节；
Q_m：模型质量；
质量主网格不需要完整遍历 B，但必须检查 batch invariance。



1. 立即执行规则：正在运行的实验如何处理
1.1 不必删除当前性能数据
已经完成或正在进行的 timing 数据继续保存，但 manifest 必须新增：
quality_status: unvalidated
claim_eligibility: performance_only


这些数据可以用于：
调试 runner；
验证重复性；
查找 provisional knee；
选择 profiler 点；
估算完整实验成本。


但不得用于以下结论：
“该配置实现了 X× 有效加速”
“该方法是最佳配置”
“该压缩率具有实际部署价值”


1.2 暂停条件
允许继续：
硬件 preflight；
BF16 baseline；
adapter port；
numerical golden tests；
allocation accounting；
CUDA Graph correctness；
performance smoke/pilot。


应暂停或标记为 provisional：
Full Scan 的论文级结论；
方法排名；
Pareto 最优配置声明；
对外报告的 speedup headline。


1.3 先冻结评测集，再看量化质量结果
为了避免看到结果后挑选有利数据，必须先冻结：
benchmark list
benchmark revision
sample IDs
prompt templates
chat template
few-shot examples
random seeds
decoding configuration
output parser
non-inferiority margins


冻结后写入：
configs/quality/quality_contract.yaml
artifacts/quality_contract/<contract_id>/checksums.sha256



2. 核心质量原则
2.1 质量不是单一数字
至少区分四层质量：
表示与执行保真度：cache、attention、logits 是否出现异常；
语言建模保真度：next-token likelihood/perplexity 是否退化；
长上下文能力：检索、多跳、聚合、全上下文理解是否退化；
生成任务能力：推理、问答和自由生成结果是否退化。


不得用单一 NIAH 或单一 perplexity 代替全部质量结论。
2.2 质量比较必须与 BF16 配对
每个量化样本必须和同一个 BF16 样本配对：
same model weights
same tokenizer revision
same prompt text
same tokenized prompt
same chat template
same few-shot examples
same decoding parameters
same max_new_tokens
same stop conditions
same output parser


质量差定义为：
$$
\Delta Q_{m,c,i}=Q_{m,c,i}-Q_{BF16,i},
$$



其中：
m：方法；
c：量化配置；
i：同一个评测样本。


所有置信区间优先基于 paired samples 计算。
2.3 性能与质量运行必须分离
质量 runner 不进入正式 timing 统计，原因包括：
tokenizer 和 output parsing 会污染 latency；
benchmark 的输出长度不同；
generation task 不具有固定工作量；
lm-eval/HELMET/RULER 的调度方式不是主 fixed-L runner。


数据中必须区分：
run_family: timing | profiler | quality


2.4 Batch size 不是质量主自变量
理论上，同一 prompt 在确定性解码下，batch size 不应改变语义输出。完整质量网格主要研究：
$$
Q_m(L,r),
$$



而不是昂贵地研究：
$$
Q_m(B,L,r)
$$



但必须执行 batch invariance test：
B = 1
B = 当前配置可运行的较大 batch 端点


若相同样本在不同 B 下质量系统性变化，则优先判定为：
implementation/backend correctness failure


而不是“算法的自然质量变化”。

3. 质量评估架构
新增独立目录：
src/kvbench/quality/
├── runner.py
├── paired_generation.py
├── logit_fidelity.py
├── perplexity.py
├── batch_invariance.py
├── parsers/
│   ├── gsm8k.py
│   ├── qa.py
│   └── multiple_choice.py
└── reporting.py

configs/quality/
├── quality_contract.yaml
├── smoke.yaml
├── fast_gate.yaml
├── long_context.yaml
└── final_suite.yaml

analysis/quality/
├── paired_bootstrap.py
├── noninferiority.py
├── quality_surface.py
├── pareto.py
└── make_quality_figures.py

artifacts/quality/
└── <quality_run_id>/
    ├── manifest.json
    ├── sample_results.parquet
    ├── aggregate_results.parquet
    ├── generations.jsonl.zst
    ├── logits_summary.parquet
    ├── quality_gate.json
    ├── quality_report.md
    └── checksums.sha256



4. Quality Gate 分层
任何配置进入正式 Full Scan 结论前，必须依次通过 Q0、Q1、Q2。Q3 用于最终论文级验证。

Phase 12Q-0：实现保真度与长序列数值检查
目标
发现小张量 golden test 无法发现的累积误差、backend 错误和形状相关错误。
输入网格
methods:
  - bf16
  - turboquant/*
  - kivi/*
  - kvquant/*

batch_size: [1, batch_endpoint]
context_length: [512, 4096, 16384, 32768, 65536, 131072]
continuation_tokens: 32
prompt_count_per_length: 16


若某长度超过模型或显存能力，记录 capacity_infeasible，不能静默删除。
指标
A. Cache round-trip
分别记录：
$$
\mathrm{MSE}(K,\hat K),\quad \mathrm{MSE}(V,\hat V),
$$



$$
\cos(K,\hat K),\quad \cos(V,\hat V).
$$



这些指标只用于机制分析，不直接等价于模型质量。
B. Attention score divergence
对选定层和 head 记录：
$$
D_{KL} \left( A_{BF16}\;\|\;A_m \right),
$$



其中 A 是 softmax 后的 attention distribution。
C. Attention output error
$$
E_O= \frac{\|O_m-O_{BF16}\|_2} {\|O_{BF16}\|_2+\epsilon}.
$$



D. Next-token logit fidelity
对相同 prefix 记录：
logit cosine similarity
KL(BF16 || quantized)
top-1 agreement
top-5 overlap
BF16 selected-token log-prob delta
quantized selected-token log-prob delta


E. Multi-step trajectory divergence
执行两种模式：
Teacher-forced mode：每步继续输入 BF16 选出的 token，测纯条件分布偏差；
Free-running mode：每个实现使用自己的 greedy token，测生成轨迹首次分叉位置。


记录：
first_divergence_step
prefix_agreement_length
sequence_exact_match
token_edit_distance
invalid_token_rate


F. Execution equivalence
相同量化方法和配置比较：
eager vs CUDA Graph
B=1 vs B=batch_endpoint
fixed-L vs growing-context 的重叠步骤
reference lane vs measurement lane


Q0 硬失败条件
以下任一项发生即失败：
NaN/Inf
illegal token ID
cache corruption
outlier cap violation
same-method graph/eager 出现大规模输出不一致
batch-dependent systematic output corruption
reference/measurement 语义不一致
长上下文下输出突然退化到无效文本


Q0 软门槛
下面数值只作为默认起点，必须在查看量化结果前由研究者冻结：
median_logit_kl_max: 0.02
p95_logit_kl_max: 0.10
next_token_top1_agreement_min: 0.95
median_attention_output_relative_error_max: 0.05


这些不是普遍定律。若论文选择其他门槛，必须记录理由并预注册。
Codex 任务
E12Q-00 Define quality schema and immutable prompt fixtures
E12Q-01 Implement teacher-forced logit comparison
E12Q-02 Implement free-running divergence runner
E12Q-03 Implement graph/eager and batch invariance checks
E12Q-04 Produce per-layer/head diagnostic report


验收输出
q0_fidelity_report.md
q0_fidelity_samples.parquet
q0_layer_heatmaps/
q0_gate.json



Phase 12Q-1：快速语言建模与回归 Gate
目标
以较低成本检测整体 language-modeling quality 是否已经明显退化。
推荐数据
主数据：
WikiText-2 test split


补充数据：
固定 revision 的 C4 validation 子集


必须保存：
dataset name
dataset revision
split
sample IDs
text hashes
tokenizer revision
stride
window size


评测方式
使用 teacher forcing 计算 NLL：
$$
\mathrm{NLL} = -\frac{1}{N} \sum_{t=1}^{N} \log p(x_t\mid x_{<t}).
$$



Perplexity：
$$
\mathrm{PPL}=\exp(\mathrm{NLL}).
$$



质量差同时报告：
$$
\Delta\mathrm{NLL} = \mathrm{NLL}_m- \mathrm{NLL}_{BF16},
$$



$$
R_{PPL} = \frac{\mathrm{PPL}_m} {\mathrm{PPL}_{BF16}}-1.
$$



NLL 是统计建模的首选量，因为 PPL 经过指数变换后不对称。
长度分桶
不要只报告全数据平均。按有效 prefix length 分桶：
0–4K
4K–16K
16K–32K
32K–64K
64K–128K


只有数据实际覆盖的桶才报告。
默认 non-inferiority margin
在看到量化结果前冻结：
relative_ppl_increase_max: 0.01
absolute_ppl_increase_max: 0.10


建议判定规则：
只有当 relative 和 absolute 两项均超过门槛时，才因 PPL 单项失败；
但任何长度桶出现显著灾难性退化，都可触发人工审查或失败。


Fast Gate 规模
开发期：
wikitext2_fraction: 0.20
c4_sample_count: 256


正式期：
wikitext2_fraction: 1.00
c4_sample_count: 2048


Codex 任务
E12Q-10 Pin datasets and tokenizer revisions
E12Q-11 Implement paired NLL/PPL runner
E12Q-12 Add prefix-length bucketing
E12Q-13 Add paired bootstrap confidence intervals
E12Q-14 Generate Q1 gate report



Phase 12Q-2：长上下文能力 Gate
目标
验证量化后的 cache 不只是保持表面 NIAH，而是仍能完成：
single/multi-needle retrieval
variable tracking
multi-hop tracing
aggregation
long-context application tasks


A. RULER 作为可控长度主轴
RULER 允许控制长度和任务复杂度。主网格建议：
context_length: [4096, 16384, 32768, 65536, 131072]
tasks:
  - single_needle
  - multi_needle
  - variable_tracking
  - common_words_extraction
  - frequent_words_extraction
  - multi_hop_tracing
  - aggregation


若使用完整 RULER 13-task suite，应在 final suite 中执行。
每个长度、任务、配置使用相同 seed 和相同生成样本。
B. HELMET RAG subset 作为快速真实任务代理
开发期优先采用 HELMET 推荐的 RAG 类任务子集，因为它比单纯 NIAH 更贴近下游长上下文应用。
建议长度：
context_length: [32768, 65536, 131072]


C. LongBench/LongBench v2 作为外部验证
预算允许时，最终报告至少增加一个现实长上下文 benchmark：
LongBench subset
或
LongBench v2 compatible subset


选择和 sample IDs 必须在结果前冻结，不得依据方法表现挑题。
指标
按 benchmark 官方定义记录：
exact match
F1
accuracy
ROUGE or task-specific score
invalid response rate


统一报告：
$$
\Delta Q_{task,length} = Q_{m,task,length} - Q_{BF16,task,length}.
$$



默认 non-inferiority margin
overall_accuracy_or_f1_drop_max_pp: 2.0
single_length_band_drop_max_pp: 5.0
invalid_response_rate_increase_max_pp: 1.0


判定必须使用 paired bootstrap CI，而不只比较点估计。
重要规则
NIAH 通过 ≠ 长上下文质量通过。


如果 NIAH 100%，但 multi-hop、aggregation 或 HELMET RAG 明显下降，配置仍然失败。
Codex 任务
E12Q-20 Integrate pinned RULER revision
E12Q-21 Build deterministic length/task generation manifests
E12Q-22 Integrate HELMET RAG subset or approved realistic benchmark
E12Q-23 Implement paired sample scoring
E12Q-24 Produce quality-vs-length curves
E12Q-25 Produce Q2 gate report



Phase 12Q-3：生成与推理任务最终验证
目标
验证量化不会在多步生成中破坏推理和答案格式。
最低推荐套件
GSM8K：多步数学推理
ARC-Challenge：知识与推理，多项选择
HellaSwag：常识续写
TruthfulQA（固定一种 scoring 口径）：真实性/错误倾向


若论文范围强调长上下文而非通用能力，可以将 ARC/HellaSwag/TruthfulQA 放入 appendix，但 GSM8K 和至少一个现实长上下文任务应保留。
两阶段预算
开发 Gate
gsm8k_sample_count: 200
arc_challenge_sample_count: 200
hellaswag_sample_count: 200
truthfulqa_sample_count: 200


样本 ID 固定。
最终论文 Gate
使用完整测试集或预注册的统计功效足够的固定子集。
解码配置
必须完全冻结：
temperature: 0
top_p: 1.0
top_k: disabled
seed: fixed
max_new_tokens: task_specific_but_fixed
stop_sequences: fixed
chat_template_revision: pinned
thinking_mode: explicitly_enabled_or_disabled


若模型具有显式 reasoning/thinking mode，必须保证所有实现使用相同模式，并单独记录 reasoning token 数。
附加指标
accuracy/EM/F1
invalid response rate
answer extraction failure rate
mean output tokens
median output tokens
truncation rate
first-token divergence against BF16
sequence exact match against BF16（诊断量，不是任务质量）


输出长度变化必须记录，因为它同时影响质量和请求延迟。
默认 margin
gsm8k_accuracy_drop_max_pp: 2.0
other_task_accuracy_drop_max_pp: 2.0
invalid_rate_increase_max_pp: 1.0
truncation_rate_increase_max_pp: 1.0


Codex 任务
E12Q-30 Pin lm-evaluation-harness revision
E12Q-31 Implement adapter/backend bridge for all cache methods
E12Q-32 Freeze prompt and output parser manifests
E12Q-33 Run development subsets
E12Q-34 Run full final suite only after configs are selected by predeclared rules
E12Q-35 Produce paired non-inferiority report



5. Config 级质量 Admission Gate
为每个方法配置生成：
{
  "method": "kivi",
  "config_id": "k2v2_g32_r32",
  "q0_fidelity": "pass",
  "q1_perplexity": "pass",
  "q2_long_context": "pass",
  "q3_generation": "pass",
  "overall_quality_gate": "pass",
  "eligible_for_performance_claim": true
}


正式规则：
Q0 fail  → 配置是实现错误或不可用，停止性能结论。
Q1 fail  → 不得称为质量保持的压缩。
Q2 fail  → 不得用于长上下文性能结论。
Q3 fail  → 可以报告系统性能，但必须明确为质量退化配置，不得进入推荐 Pareto front。


主论文推荐配置原则上必须：
Q0 + Q1 + Q2 + Q3 全部通过。



6. 质量网格设计：避免成本爆炸
6.1 不对所有 B 重复完整质量评测
完整质量评测使用：
batch_size: 1


batch invariance 只使用小样本检查：
batch_size: [1, batch_endpoint]


6.2 Config 选择
第一层：所有配置跑 Q0/Q1
BF16
TurboQuant 3 configs
KIVI 3 configs
KVQuant 3 configs


第二层：Q1 通过者跑 Q2 fast subset
失败配置不进入昂贵长上下文套件，但保留失败数据。
第三层：Q2 通过者跑 Q3 development subset
第四层：预声明选出的代表配置跑最终完整套件
建议每种方法最多选择：
一个 highest-quality 配置
一个 highest-compression 且通过质量门槛的配置


选取规则必须在结果前定义，例如：
在通过 Q0–Q2 的配置中，选择 r_alloc 最大者；
若并列，选择 Q1 NLL 增量更小者。



7. 从“速度模型”升级为“质量约束性能模型”
7.1 质量响应面
对每个方法拟合：
$$
D_m(L,r) = Q_{BF16}(L)-Q_m(L,r),
$$



其中 D 是 quality degradation。
不必强求 closed form，可用：
GAM
tensor-product spline
Gaussian process
monotonic/shape-constrained regression


质量通常不必以 B 为主输入；若 batch invariance 失败，先修实现。
7.2 质量约束下的最优速度
定义可接受质量集合：
$$
\mathcal{C}_{\delta} = \{(m,r):D_m(L,r)\le\delta\}.
$$



则真正有意义的 speedup 是：
$$
\boxed{ S^*(B,L;\delta) = \max_{(m,r)\in\mathcal{C}_{\delta}} S_m(B,L,r) }
$$



而不是无条件选择速度最快的配置。
7.3 Pareto front
每个配置是一点：
$$
\left( \text{latency}, \text{allocated bytes}, \text{quality degradation} \right).
$$



若配置 A 满足：
latency 不高于 B
memory 不高于 B
quality 不差于 B
且至少一项严格更好


则 B 被 A Pareto-dominate，不应作为推荐配置。
不要在没有预先定义权重时，将三者随意压成单一“综合分数”。

8. 统计方法
8.1 配对重采样
质量 CI 的 resampling unit 是评测样本：
QA question
RULER instance
GSM8K problem
text document/window


不得把生成 token 当作独立样本。
8.2 推荐检验
连续指标/NLL/F1：paired bootstrap
二元正确率：paired bootstrap；必要时 McNemar test
多长度曲线：按 task × length 分层 bootstrap
多配置比较：报告 multiplicity，并使用 Holm correction 或明确 exploratory


8.3 报告点估计和区间
每个结果至少包含：
BF16 score
quantized score
paired difference
95% CI
sample count
invalid count


不得只报告“下降 0.3%”而没有样本数和不确定性。

9. 质量数据 Schema
sample_results.parquet 至少包含：
quality_run_id
quality_contract_id
git_sha
container_digest
gpu_uuid
model_id
model_revision
tokenizer_id
tokenizer_revision
chat_template_hash
method
method_config_id
benchmark
benchmark_revision
task
sample_id
sample_hash
seed
batch_size
requested_context_length
actual_prompt_tokens
actual_output_tokens
truncated_prompt
truncated_output
decoding_config_hash
raw_prediction_path
normalized_prediction
reference_answer
sample_score
bf16_sample_score
paired_score_delta
invalid_response
parse_failure
first_divergence_step
logit_kl_median
logit_kl_p95
top1_agreement
status
failure_reason


原始 generations 必须压缩保存并做 checksum，不能只保留 aggregate。

10. 质量配置模板
quality_contract:
  id: quality_v1

  model:
    id: meta-llama/Meta-Llama-3.1-8B-Instruct
    revision: PIN_REQUIRED
    tokenizer_revision: PIN_REQUIRED
    chat_template_hash: COMPUTE_AND_FREEZE

  decoding:
    temperature: 0.0
    top_p: 1.0
    do_sample: false
    seed: 20260722
    thinking_mode: disabled

  methods:
    - bf16
    - turboquant_4bit_nc
    - turboquant_k3v4_nc
    - turboquant_3bit_nc
    - kivi_k4v4
    - kivi_k2v4
    - kivi_k2v2
    - kvquant_4bit
    - kvquant_3bit
    - kvquant_2bit

  q0:
    context_length: [512, 4096, 16384, 32768, 65536, 131072]
    batch_size: [1, 8]
    prompts_per_length: 16
    continuation_tokens: 32
    median_logit_kl_max: 0.02
    p95_logit_kl_max: 0.10
    next_token_top1_agreement_min: 0.95

  q1:
    wikitext2:
      revision: PIN_REQUIRED
      split: test
    c4:
      revision: PIN_REQUIRED
      split: validation
      sample_ids_file: configs/quality/fixtures/c4_ids.txt
    relative_ppl_increase_max: 0.01
    absolute_ppl_increase_max: 0.10

  q2:
    ruler:
      revision: PIN_REQUIRED
      lengths: [4096, 16384, 32768, 65536, 131072]
      tasks:
        - single_needle
        - multi_needle
        - variable_tracking
        - common_words_extraction
        - frequent_words_extraction
        - multi_hop_tracing
        - aggregation
    helmet:
      revision: PIN_REQUIRED
      suite: rag_fast
      lengths: [32768, 65536, 131072]
    overall_drop_max_pp: 2.0
    band_drop_max_pp: 5.0

  q3:
    lm_eval_revision: PIN_REQUIRED
    tasks:
      gsm8k:
        fewshot: PREDECLARE
        sample_ids_file: configs/quality/fixtures/gsm8k_ids.txt
      arc_challenge:
        fewshot: PREDECLARE
      hellaswag:
        fewshot: PREDECLARE
      truthfulqa_mc2:
        fewshot: PREDECLARE
    accuracy_drop_max_pp: 2.0
    invalid_rate_increase_max_pp: 1.0


不要直接使用模板中的 PIN_REQUIRED 运行。Codex 必须解析为明确 SHA/revision 后才允许执行。

11. Makefile/CLI 新增接口
make quality-contract-validate
make quality-fixtures

make quality-q0 METHOD=bf16
make quality-q0 METHOD=turboquant
make quality-q0 METHOD=kivi
make quality-q0 METHOD=kvquant

make quality-q1-fast
make quality-q1-full

make quality-ruler-fast
make quality-ruler-full
make quality-helmet-fast

make quality-lm-eval-fast
make quality-lm-eval-full

make quality-batch-invariance
make quality-graph-invariance

make quality-report
make quality-pareto
make quality-gate METHOD=<method> CONFIG=<config_id>


所有正式命令必须读取版本化 YAML，禁止临时改变任务、样本或门槛。

12. AGENTS.md 必须追加的规则
## Quality invariants

1. No speedup is deployment-relevant until the exact method/config passes its quality gate.
2. Numerical golden tests do not substitute for end-to-end quality evaluation.
3. NIAH alone is not sufficient evidence of long-context quality.
4. Perplexity alone is not sufficient evidence of generation or long-context quality.
5. Quality prompts, sample IDs, parsing rules, and margins must be frozen before method results are inspected.
6. Every quantized quality sample must be paired with the identical BF16 sample.
7. Temperature, chat template, tokenizer, stop rules, and thinking mode must be identical across methods.
8. Batch-size-dependent semantic output is treated as a correctness bug until disproven.
9. Raw generations and per-sample scores are immutable and must be checksummed.
10. Failed quality configurations remain in the dataset; they are never silently excluded.
11. The performance and quality runners are separate run families.
12. Final recommendations are made on a latency-memory-quality Pareto front, not speed alone.



13. Codex issue/PR 拆分
E12Q-00 Quality contract, schemas, and frozen fixtures
E12Q-01 Paired BF16/quantized generation harness
E12Q-02 Teacher-forced logit and attention fidelity
E12Q-03 Graph/eager and batch invariance
E12Q-10 PPL/NLL evaluation pipeline
E12Q-11 Prefix-length quality bucketing
E12Q-20 RULER integration
E12Q-21 HELMET RAG integration
E12Q-30 lm-evaluation-harness integration
E12Q-31 Deterministic output parsers
E12Q-40 Paired bootstrap and non-inferiority
E12Q-41 Quality gate report
E12Q-42 Joint Pareto analysis


每个 PR 不得同时改变：
quantization implementation
quality prompt semantics
quality threshold
scoring parser


其中两个或以上。否则无法审计质量变化来源。

14. Codex 可直接执行的主 Prompt
Read AGENTS.md, the main KV quantization workflow, and this quality addendum.

The current performance data is quality-unvalidated. Do not delete or rerun it merely because quality evaluation was missing. Add a separate Quality Lane and preserve all existing raw data.

First complete only E12Q-00 through E12Q-03.

Required work:
1. Add immutable quality schemas and a versioned quality contract.
2. Freeze model/tokenizer/chat-template revisions and prompt fixture hashes.
3. Implement paired BF16 vs quantized teacher-forced logit evaluation.
4. Implement free-running greedy divergence measurement.
5. Add graph-vs-eager and B=1-vs-batch-endpoint invariance tests.
6. Add raw per-sample output storage and checksums.
7. Add a MethodQualityAdmissionReport.
8. Mark existing timing runs as quality_status=unvalidated without editing their raw samples.

Constraints:
- Do not run the final benchmark suite yet.
- Do not change quantization kernels in the same PR as quality infrastructure.
- Do not choose prompts or thresholds after inspecting method results.
- Do not treat NIAH or perplexity as the sole quality metric.
- Do not include quality-run wall time in the fixed-L performance dataset.
- Do not make deployment or paper speedup claims.

Before implementation, write docs/decisions/quality_evaluation_plan.md containing:
- exact benchmark revisions to pin;
- exact task/sample selection rules;
- exact deterministic decoding settings;
- proposed non-inferiority margins;
- estimated GPU and storage cost;
- unresolved compatibility risks for TurboQuant, KIVI, and KVQuant.

Then implement the infrastructure, run only the quality smoke tests, and return:
- files changed;
- commands run;
- gate results;
- any graph/batch/reference mismatch;
- the next recommended issue.



15. Full quality run Codex Prompt
只有 Q0 infrastructure 通过并且质量合同已经由研究者批准后，才执行：
Read the approved quality contract and execute the versioned quality plan exactly as written.

Order:
1. Run BF16 quality baselines first and freeze their outputs.
2. Run Q0 for all method/config pairs.
3. Run Q1 only for Q0-pass configurations.
4. Run the fast RULER/HELMET Q2 suite only for Q1-pass configurations.
5. Run the development lm-eval Q3 subset only for Q2-pass configurations.
6. Select final configurations using the predeclared selection rule.
7. Run the full final suite only for selected configurations and BF16.
8. Compute paired confidence intervals and non-inferiority decisions.
9. Build the latency-memory-quality Pareto report.

Rules:
- Never overwrite a completed quality run.
- Never retry only failed or low-scoring samples without recording the retry policy.
- Never change output parsers after inspecting method-specific answers.
- Record all OOM, truncation, parse failure, invalid response, and backend fallback events.
- Preserve raw generations and sample-level scores.
- Report failures honestly; do not average away a collapsed length band.



16. 最终论文级报告必须同时回答的问题
对每一种方法和配置，最终表格至少回答：
实际 r_alloc 是多少？
实际 r_HBM 是多少？
相同 B,L 下 latency speedup 是多少？
PPL/NLL 变化是多少？
长上下文得分随 L 如何变化？
推理/生成任务下降是多少？
无效输出率是否变化？
输出长度是否变化？
是否通过预注册的质量门槛？
是否位于 latency-memory-quality Pareto front？


最终推荐语句应类似：
在 RTX PRO 6000、固定模型与软件栈下，配置 X 在 L=\cdots、B=\cdots 时实现 S\times 同工作量 decode speedup 和 r_{alloc}\times cache compression；其预注册质量套件相对 BF16 的最差长度桶下降为 \Delta，通过/未通过 non-inferiority Gate。


不得只写：
配置 X 加速了 S\times。



17. 完成标准
Quality Lane 只有在以下条件全部满足时才算完成：
质量合同已冻结并有 checksum
BF16 baseline 可复现
所有方法有 paired sample-level outputs
Q0/Q1/Q2/Q3 gate 状态明确
长上下文有 length-resolved results
graph/eager 和 batch invariance 已检查
原始 generations 被保存
统计区间基于 paired samples
性能结果与质量结果通过 config fingerprint 精确关联
最终推荐基于 Pareto front



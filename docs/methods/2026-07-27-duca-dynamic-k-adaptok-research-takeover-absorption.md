# DUCA 动态 K / AdapTok 研究接管回复吸收与独立审计

## 原始记录

- 日期：`2026-07-27`
- 原始附件：
  `C:/Users/skywalker/.codex/attachments/370a2c39-6571-4a98-af52-1445fd6fc21e/pasted-text.txt`
- 字节一致归档：
  `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-research-takeover-raw.txt`
- 大小：`152,867` bytes
- 物理行数：`4,589`（末行换行符计数为 `4,588`）
- SHA-256：
  `5ae7850662d726d91c4b3dc7f362ad223d33c35e3cbad9bb87771e939e07e031`
- 字节一致性：`true`

## 项目裁决

```text
SUBSTANTIAL_ACCEPT_RESEARCH_DIRECTION
MAJOR_CORRECTION_BEFORE_DESIGN_FREEZE
DYNAMIC_K_IS_A_REQUIRED_CANDIDATE, NOT AN EMPIRICAL_FACT
RECOMMEND_REGRET_GATED_HIERARCHICAL_ACQUISITION
NO_MODEL_CODE_OR_LONG_TRAINING_AUTHORIZED
```

这份回复覆盖面很广，正确抓住了下一阶段最有价值的科学中心：不是继续优化
actionness/transition 分数，而是判断新增物理帧对离线 TAD 分类、回归、高 tIoU 和双端
边界的真实边际价值，并在真实平均成本下联合决定 `K` 与帧位置。它也正确拒绝了
test-batch ILP、`Kmax` padding、post-NMS 才做物理映射、仅以 soft gradient 证明 hard
utility、以及把 90 轮结果放进公平主表。

但该回复不能逐字冻结为实现规范。它的短版和扩展版对 nestedness 给出相反裁决，
多个数值门槛互相冲突，若干数学式损坏或缺少成立条件，实验矩阵没有完全隔离动态
预算与 mixed-K/位置策略，数据划分和统计单位也没有闭环。正确状态是：

```text
scientific_direction = accepted_with_major_corrections
candidate_model = discussed
mathematical_contract = not_frozen
implementation = not_started
experiment = not_started
paper_claim = not_allowed
```

## 是否回答了关键问题

| 问题 | 覆盖度 | 独立审计 |
|---|---|---|
| 当前 DUCA 证据与公平性 | 高 | 方向正确；K=192 证据标签前后冲突，native uniform 与 wrapper uniform 仍需更清楚分离 |
| AdapTok 机制与迁移边界 | 高 | 主要判断正确；PDF 端点未逐页读取的限制已诚实披露 |
| 动态 K 是否值得成为主候选 | 高 | 回答了研究优先级，但“必须成为最终主创新”是设计判断，不是已验证事实 |
| 固定 K / 动态 K / train-free 统一关系 | 高 | 概念上成立；strict train-free 仍完全没有任务结果 |
| 内外层数学模型 | 中高 | 给出完整轮廓，但 nested/independent 冲突、公式损坏和若干假设缺失，不能直接实现 |
| 总 60 轮公平实验 | 中高 | 给出大体依赖图；缺同 K 直方图的 uniform-position 对照、严格 split manifest 和统一门槛 |
| 代码可实施性 | 中 | 没有把 RIME 缺口完整映射到主树；当前主树仍是旧 online/selected-axis DUCA |
| 新颖性与相关工作 | 中高 | 正确排除“动态 K/ILP/inverse-CDF 首创”；遗漏若干直接近邻，组合 claim 仍需实验证明 |
| 定理与保证 | 中 | batch invariance/价格单调可保留；其他命题需补条件或降为经验假设 |
| 可发表性与停止条件 | 高 | 结构合理，但具体 `pp`、gap recovery、成本比例须先做功效分析再预注册 |

因此，这份回复回答了大多数关键问题，但没有回答完以下决定性问题：

1. 主解码器究竟是 independent-per-K、strict nested，还是 weakly nested；
2. dynamic Oracle 的真实 headroom 是否存在；
3. nestedness 相对 independent optimum 的真实 regret；
4. hard utility 是否能在视频级、多尺度扰动下预测真实检测收益；
5. dynamic K 的收益是否独立于 mixed-K 训练、K 直方图和 uniform-position 策略；
6. 哪个视频级 split 分别用于 detector 训练、utility 拟合、dual/risk 校准、认证和最终评估；
7. 当前代码如何以最小改动实现 raw-proposal `q -> t -> official NMS`；
8. 真实执行是否按 K 分桶而非 padding，并在完整端到端成本上净省。

## 完全吸收的部分

1. 任务始终是 **offline TAD**，不能称 Online TAD。
2. 主候选是重型 backbone 之前的真实物理帧采集，不是 backbone 内 token pruning。
3. dynamic K、质量曲线、ILP、inverse-CDF、学习选帧都不能单独声称首创。
4. 固定 K 是动态模型的必要因果锚点；固定 K 内层失败时不得用动态 K“救结果”。
5. 外层应使用训练侧冻结的 per-video dual price；test-batch ILP 只能是
   AdapTok-style baseline 或 transductive Oracle。
6. 效用必须由 train-only、frozen-detector hard counterfactual 检验，优先使用
   cls/reg/high-IoU/pair-boundary 向量而非未经校准的 raw scalar。
7. K=192 的局部边界改善与宽双端、最大洞恶化是当前最高价值负证据；新模型必须直接
   针对 paired endpoints，而不是继续 action enrichment。
8. 非线性时间重参数化下，raw proposals 必须先从 `q` 映回物理时间 `t`，再调用
   参数不变的官方 NMS。
9. 所有 trainable 主臂共享最多 6,000 次成功 detector updates，development seed
   不进入正式均值。
10. 动态执行必须记录 requested/effective/unique/backbone K，并同时报告 batch=1
    latency、bucket throughput、memory、energy 和完整前端成本。

## 必须校正的 P0 问题

### 1. Nested 与 independent 直接矛盾

短版在原文 `203--221` 和 `779--790` 明确裁决：

```text
不强制 S(K1) subset S(K2)
共享密度，各 K 独立 exact-K 解码
nested-addition 仅作诊断
```

扩展版却在 `1593--1614`、`2481--2491`、`2738--2798`、
`2940--2973`、`3431--3438` 和 `3493--3495` 把 strict nested
写成主模型要求。二者对应不同效用：

```text
budget-policy value:
U(K2, S*(K2)) - U(K1, S*(K1))

addition value:
U(K2, S(K1) union A) - U(K1, S(K1))
```

它们不能共用同一个“新增帧边际收益”名称。离线插件在 heavy backbone 前一次性决定
`K`，没有 progressive heavy decoding 复用刚需，因此 nestedness 必须由 train-only
Oracle regret 决定，而不是由论文叙事决定。

### 2. 证据等级不一致

原文 `45--54` 将 K=192 终点和机制数值列为 `PARTNER_CLAIM`，但
`1357--1363` 又列为 `DOCUMENTED_RESULT / PROVIDED_AFTER_PUBLIC_SNAPSHOT`。
当前项目记录已确认这些是 snapshot 之后登记的终端诊断；正确标签是：

```text
documented_post_snapshot_terminal_diagnostic
over_budget_90_epochs
no_clean_native_uniform
not_paper_support
```

### 3. 当前代码事实与目标合同混淆

当前旧路径的确是 selected-axis NMS 后再映射到物理时间；目标合同则必须是
`raw q proposals -> physical t -> official NMS`。前者是 `CODE_FACT / known bug`，
后者是 `designed target contract / not implemented`，不能混成“当前已经满足”。

### 4. 基线和因果对照不完整

动态方法使用多 K 集合，而原矩阵只列固定 `U50=K384`、`U25=K192`。至少还需要：

- 每个候选 K 的 clean native uniform；
- 与 full model 完全相同逐视频 K 序列的 uniform-position 对照；
- 保持 K 直方图、只打乱 K 与视频配对的 histogram-shuffle；
- 相同 mixed-K exposure、warmup/ramp、总 heavy frames 和 checkpoint 规则；
- MGSampler-like / motion-uniform 与 AdapTok-TAD 直接迁移基线。

否则 dynamic gain 可能来自 mixed-K 训练、某个更优平均 K、或位置策略，而非内容感知
预算分配。

### 5. 数据划分与泄漏边界未闭环

必须冻结基于完整视频 ID 的不重叠 manifest，至少区分：

1. detector/selector training；
2. hard-utility label generation；
3. utility/risk head fitting；
4. dual price 与风险阈值 calibration；
5. certification/development；
6. official final evaluation。

当前项目已知 THUMOS 的若干 `val/test` 配置实际复用 validation videos；改变 overlap
不能制造独立测试总体。用于生成 Oracle/hard utility 的 detector 也不能在同一
utility holdout 上选阈值或结构。

### 6. 门槛互相冲突

回复同时出现 Oracle `+0.75pp` 与 `+1.0pp`、20% 与 40% gap recovery、25% 与 30%
成本下降、不同的 development/final 增益阈值。它们全部降级为
`designed_reviewer_proposal`。正确流程是先得到 clean baseline 方差与视频聚类功效，
再一次性预注册阈值；不得在主结果出现后选最有利版本。

### 7. 数学与公式不能直接复制

原文多处出现 `====`、`##`、`z!`、错误下标和缺失运算符。除此之外还需修正：

- 密度可行域至少要求 `alpha <= 1 <= beta`，并定义空可行域的 fail-closed 行为；
- 离散 `F^{-1}` 必须定义 generalized inverse/插值和确定性 tie-break；
- 0-based gap 应统一为 `s1 + 1`、`s[j+1]-s[j]`、`T-sK`；
- independent exact-K DP 与 nested insertion DP 是两种求解器，复杂度不可混写；
- `B`、`bar B`、`N B` 的总成本/均值成本口径必须统一；
- Hoeffding/i.i.d. 预算界不能直接把同一视频的滑窗当独立样本，应以完整视频聚类；
- endpoint bound 中物理时间与 frame index 必须使用同一单位；
- paired-boundary 互补性可能违反次模性，不能无条件声称 `1-1/e`；
- endpoint-to-IoU 只是带条件的单区间保守界，不是 mAP 定理。

可以保留的简单命题是：固定 dual price 与确定性 tie-break 下的 batch invariance，
以及成本单调时最优预算随价格非增。二者主要是协议/优化性质，不足以单独支撑新颖性。

### 8. 风险 fallback 会破坏平均预算

“所有 K 均不安全时回退 Kmax”会提高 realized mean K。必须把
`risk_infeasible` 作为显式事件计入 calibration、预算证书和成本表；不能既回退 Kmax
又宣称平均预算严格满足。

### 9. 新颖性地图仍不完整

除回复已有的 AdaFrame、MGSampler、AdapTok、DynamicViT、TE-TAD 等，还必须覆盖：

- AdaFocus / AdaFocusV3 / Uni-AdaFocus：cheap global evidence、sparse heavy
  computation 与 per-sample dynamic compute；
- SMART、Search-Map-Search、Dynamic Inference：联合/任务感知帧选择；
- ETAD：高效端到端 TAD 训练与动态 proposal sampling；
- GAP：固定 snippet 下采样的时间量化与边界损伤。

因此不能声称“首次 pre-backbone 动态帧预算”“首次 cheap-to-heavy 动态视频计算”
或“首次风险校准”。最可防守的候选组合命题是：

> 面向离线区间检测，用 train-only hard counterfactual 估计新增物理帧对分类、回归和
> paired endpoints 的预算条件价值，在有界物理时间 exact-K 采集中用 batch-invariant
> 平均成本策略分配 heavy backbone 观测，并以 pre-NMS 物理坐标和真实端到端成本闭环。

该命题仍需 AdapTok-TAD、AdaFocus-like/MGSampler-like、risk/no-risk、
physical-time/no-physical-time 和第二 detector 实验证明。

## 三种可执行路线

| 路线 | 数学语义 | 优点 | 风险 | 当前裁决 |
|---|---|---|---|---|
| Independent-per-K | 共享低成本证据/密度，每个 K 独立 exact-K；预测 budget-policy value | 与离线一次性决策一致，定位自由度最高，最容易先闭环 | “incremental addition”叙事较弱 | 安全基线与默认实现底座 |
| Strict nested ladder | 低预算集合是高预算集合真子集；预测 group-add utility | 边际新增语义清楚、mixed-K 共享自然 | 可能造成高 nested regret，且更接近 AdaFrame/AdapTok 前缀 | 只在 Oracle 通过后可冻结 |
| Regret-gated weak nesting | 先比较 independent/strict nested；必要时只约束高重叠率，不要求完全包含 | 保留部分稳定性并控制定位损失 | 多一个训练侧模型选择门，必须预注册 | **推荐的当前决策协议** |

这里“推荐”不是把三套模型都送进主表，而是先用 train-only Oracle 选择最终唯一
decoder family：

1. independent 是无争议的共同底座；
2. strict nested 若在所有关键 K 上的 paired video regret 低于预注册容忍度，才升级为
   最终 ladder；
3. strict nested 失败时，只允许一次 weak-overlap 备选；若仍有明显 regret，正式模型
   使用 independent-per-K，并把效用准确称为 budget-policy value；
4. 模型结构冻结后不得依据 official test mAP 重新切换。

## 推荐的下一步实验顺序

### Phase 0：可信测量零点

1. 冻结视频级 split manifest、seed、6,000-update、EMA/checkpoint、真实成本口径。
2. 复评 released dense，并建立 local dense。
3. 建立 clean native uniform：至少 K=384、K=192；dynamic 面板启用前补齐所有候选 K
   或以完全相同 K 序列运行 uniform-position 对照。
4. clean path 与 wrapper path 做 frame tensor、raw proposal、坐标和 mAP parity。
5. 实现并验证 `raw q proposals -> physical t -> unchanged official NMS`。

任一 parity 失败时只修测量与接口，不解释 selector gain。

### Phase 1：模型无关内核

只实现可独立测试的基础设施：

- bounded density projection；
- deterministic exact-K decoder；
- constant-density exact-uniform identity；
- independent-per-K decoder；
- optional nested/weak-overlap Oracle decoder；
- q/t coordinate adapter；
- requested/effective/unique/backbone-K ledger；
- K-bucket execution，不 pad 到全局 Kmax。

此阶段允许 focused tests，不启动 learned-selector 长训练。

### Phase 2：最高信息增益 Oracle 与 hard utility

全部只使用训练侧冻结 detector 和不重叠视频：

1. `Oracle-Dynamic`：在相同 realized mean cost 下比较 best fixed 与 per-video
   budget-policy Oracle；
2. `Oracle-Nested-Regret`：independent、strict nested、weak overlap 逐 K 比较；
3. `G_rank`：单帧、2/4/8/16 帧、1%/5%/10%、连续片段、start/end pair hard swaps；
4. null controls：random、score shuffle、score reverse、matched geometry；
5. `Pair-Risk`：验证 risk score 是否比 actionness/transition 更能预测
   start/end joint failure 与 mAP@0.7 proxy；
6. utility-label 生成的额外 detector forwards/GPU-hours 单列。

样本量和通过阈值由视频聚类功效模拟冻结。dynamic Oracle、最终 decoder family、
`G_rank` 和 pair-risk 任一关键门失败，都停止相应复杂模块。

### Phase 3：一枚 development seed 的总 60 轮因果矩阵

最小而充分的正式候选矩阵：

| Arm | 定义 | 唯一问题 |
|---|---|---|
| U-fixed | clean uniform K=384 | 固定预算性能底线 |
| U-same-K | 与 full 完全相同逐视频 K，但位置 uniform | 隔离动态 K 与位置策略 |
| F-bound | 固定 K，有界 exact-K + hard utility/risk | 内层是否成立 |
| D-shuffle | 保持 full K 直方图，打乱 K 与视频 | 内容感知预算是否有效 |
| D-no-risk | 相同动态预算框架，删除 pair/high-IoU risk | TAD 风险是否必要 |
| AdapTok-TAD | 同候选 K、total-loss curve、test-batch ILP | 排除直接任务平移 |
| RIME-full | 冻结后的唯一 decoder + utility/risk + fixed dual | 完整候选 |

MGSampler-like/AdaFocus-like 可以复用同一 detector 与预算面板作为强外部机制对照，
但不应让 development 矩阵无限膨胀。所有臂必须共享 mixed-K exposure、warmup/ramp、
heavy-frame training total、6,000 updates、EMA 和评估次数。

### Phase 4：冻结后论文证据

仅当 development seed 同时显示 inner selector、dynamic assignment、high-IoU 和真实
成本均为正向时：

1. 冻结方法、阈值、K 集合和统计方案；
2. 使用三枚未参与开发的新种子；
3. 扩展第二 detector；
4. 扩展 mean-K≈192 或第二预算面板；
5. 报告 Avg-mAP、mAP@0.6/0.7、short-action、boundary error、pair support、
   max-gap、realized K、latency/energy/memory；
6. 最后运行 strict frozen-detector/train-free 模式，单列声明。

## 当前代码状态与建议实现切片

主树已有的 `C3CoarseProbeActionnessSource`、`DucaAcquisitionAdapter`、
`PrefixMarginalUtilityBudgetController` 和 `DucaOnlineFrameSelector` 属于旧
online/selected-axis 路径。现有 `budgeted_center_radius_decode` 是 greedy
center-radius union + residual fill，不是本候选的 bounded-density exact-K solver。

主树尚缺：

- bounded density / exact-K decoder；
- independent/nested-regret Oracle；
- hard counterfactual `G_rank` / pair-risk labels；
- raw-proposal pre-NMS physical inverse map；
- variable-K bucket execution；
- RIME total-60 configs、validators、launchers 和完整 cost ledger。

用户批准设计后，建议按以下最小切片实现：

1. `geometry kernel`：投影、解码、坐标与 property tests；
2. `oracle/evidence kernel`：hard utility、nested regret、null controls；
3. `budget policy`：finite-K utility/risk heads、frozen dual、risk-infeasible ledger；
4. `runtime`：K bucketing、actual-K profiler、pre-NMS inverse map；
5. `training`：单一 total-60 development config；
6. 通过门后才添加多种子、第二 detector 和 train-free mode。

不应从 `.codex_tmp` 历史实现整树复制；它们不是当前主树的 RIME 实现。

## 可发表性判据

具体数值须在 clean baseline 和功效分析后预注册，但逻辑上必须同时成立：

1. inner policy 在相同 K 下超过 clean uniform；
2. dynamic policy 在相同 realized cost 下超过最强 fixed policy；
3. 与 full 使用相同 K 序列时，learned positions 超过 uniform positions；
4. K-video pairing 超过 histogram-shuffle；
5. pair-risk 对 mAP@0.7、short actions 或双端支持有独立贡献；
6. raw-proposal physical-time-before-NMS 是正确、可复核的坐标合同；
7. 第二 detector 方向一致；
8. 完整端到端成本低于 dense，且不存在 Kmax padding 幻觉。

若 dynamic Oracle 无 headroom，动态主线停止；若 inner fixed-K 失败，整个当前 DUCA
采集路线停止；若只有 detector 内部 true-time 改造有效，则应诚实改称 time-aware TAD
integration，不再声称 pure pre-backbone plugin。

## 当前不允许的动作

- 不把回复中的任一门槛直接注册为正式门槛；
- 不把 strict nestedness 或 DUCA-RIME 名称视为已冻结；
- 不启动完整 dynamic-K 60 轮训练；
- 不把旧 MUST controller 当作新 RIME；
- 不用 official test/GT 调 K 集合、dual、risk 或 nestedness；
- 不声称最终模型已实现、已验证或 paper-ready。


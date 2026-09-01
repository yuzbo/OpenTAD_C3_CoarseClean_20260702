# DUCA 总计 60 轮纯前置插件 Pro 审阅吸收与项目裁决

## 原始记录

- 审阅日期：`2026-07-27`
- 原始附件：
  `C:/Users/skywalker/.codex/attachments/975d262f-00ea-4639-a85c-a9c45aa03f9a/pasted-text.txt`
- 字节一致归档：
  `docs/methods/reviews/2026-07-27-duca-total60-prebackbone-pro-review-raw.txt`
- 原文/归档大小：`52,824` bytes
- 原文/归档物理行数：`886`
- 原文/归档 SHA-256：
  `D493FD3497D412B3B873940447F1C743F3A1A50418EBCFC20B9FCE16945A4E11`
- 字节一致性：`true`

## 总体裁决

```text
ACCEPT_MAJOR_REVISION
ACCEPT_SCIENTIFIC_CORE
DO_NOT_ACCEPT_EVERY_PROPOSED_FORMULA_OR_THRESHOLD_AS_FROZEN_FACT
CORRECT_STALE_EVIDENCE_STATUS
MODEL_FIRST_PR_CLEANUP_NONBLOCKING
```

我不“完全认可”到逐字、逐公式、逐阈值照搬的程度，但高度认可这份审阅的主要科学
诊断和大修方向。它准确指出了当前路线距离论文级模型仍缺少的四个核心环节：

1. 干净官方基线和包装路径等价性；
2. 唯一、可测试的 exact-K 有界时间重参数化；
3. 连续贡献教师与直接检测梯度各自对真实硬换帧收益的有效性；
4. 公平总计 60 轮、视频级统计和跨检测器证据。

不直接照搬的部分主要是：未经实验支持的具体密度常数、动态规划形式、RDD 唯一性、
数值发表门槛、统一禁止中间 checkpoint、将免训练路线完全延期，以及把 PR 清理放到
模型关键路径。这些属于有价值的 reviewer proposal，不是已经成立的模型事实。

## 审阅时点后的事实校正

### PR #3

`2026-07-27` 复核 GitHub 后，PR
[`#3`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/pull/3)
仍为 `OPEN/DRAFT`，head 为
`codex/duca-total60-plugin-cvpr-20260727@63a726a4aaf48ecbf6780bb196de43a890c6b4df`，
base 为 `codex/c3-coarse-clean-20260702`。当前 diff 确为 `133` 个文件、
`27,627` 行新增、`116` 行删除和 `28` 个提交；正文中的 “Documentation only”
不能准确描述整个 PR diff。

该判断成立，但它是审阅载体问题，不是模型科学门。重建干净文档 PR 可以并行处理，
不得阻塞官方基线、数学模型和硬收益门的实施。

### 官方 AdaTAD

上游固定提交
`sming256/OpenTAD@1aa8ca4ac5e846b1e8ff69298dd6607121a01589`
的 THUMOS14 VideoMAE-S `768x160` 表报告：

```text
mAP@0.3/0.4/0.5/0.6/0.7 =
83.90 / 79.01 / 72.38 / 61.57 / 48.27
Average-mAP = 69.03
```

配置启用 EMA，`workflow.end_epoch=60`，从 epoch 40 起每两轮验证。学习率调度器参数
写为 `max_epoch=100`，但实际训练终止轮次仍是 60。审阅对“官方结果为 69.03、实际
训练为 60 轮”的核心判断正确。

### K=384 与 K=192 历史课程

- K=384 已有封存终点 `65.385724%` 和原始评估/预测/更新审计回执，不再只是固定文档
  中的口述证据；但它消耗 30+60 共 90 轮完整模型训练，只能作超预算诊断。
- K=192 已在审阅成稿后得到封存终点 `57.967272%`：
  `73.907179/68.926135/61.194230/49.841145/35.967670%`
  （tIoU 0.3 至 0.7），覆盖 211 个视频和 422,000 条预测。因此原审阅将其列为
  “中间值、证据 D”已经过时。
- K=192 后续几何诊断显示：动作富集和半径 1 的局部边界邻近逐渐改善，但 R4Q5
  宽双侧边界支持和最大未覆盖区间持续恶化。这与高 tIoU 仍弱相符，但不是因果证明。
- K=192 同样消耗 90 轮、缺少干净 K=192 均匀终点，且恢复后的全局 RNG 不能位精确
  延续。因此证据等级提升不改变论文裁决：它仍不是公平 total-60 结果，也不能证明
  learned selector 优于 uniform。

### 最终模型状态

总计 60 轮的纯 pre-backbone 有界模型仍处于 `designed / implementation_not_started`。
当前不存在可称为 CVPR 最终模型的实现或公平主结果。

## 完全吸收的科学意见

1. **逆 CDF 和学习选帧本身不是新颖性。**
   系统不等概率抽样、AdaFrame 和 MGSampler 已覆盖累计质量或自适应选帧近邻。
   论文新意必须来自 TAD 特有的高 IoU 几何、有界 exact-K 时间变换、硬策略验证的
   任务效用学习和严格纯插件证据。
2. **必须冻结唯一数学映射。**
   从低成本证据到密度、累计映射、连续位置和整数集合的
   `e -> p -> F -> y -> S` 必须定义正性、可行域、首尾覆盖、最大洞、冲突处理、
   tie-break 和不可行行为，并由 property tests 验证。
3. **干净 A0 是所有 selector 结论的前提。**
   必须恢复 released-weight dense、本地同口径 dense、clean native K=384/K=192
   uniform，并验证相同索引下 clean path 与 wrapper path 的输入、坐标和预测一致性。
4. **纯插件必须有对称坐标合同。**
   非均匀选帧对应 uniform warped time `q=F(t)`。训练真值映到 q，推理预测从 q
   映回 t；检测器的 backbone、projection、assignment、head、loss 和 NMS 算法不改。
5. **贡献教师与直接梯度必须拆成两道门。**
   `G_rank` 回答 detached 贡献排序能否预测真实硬收益；
   `G_direct` 回答连续梯度能否稳定组合并支持直接端到端更新。
   `G_rank` 失败时，当前梯度贡献教师不得进入主模型；`G_direct` 失败但
   `G_rank` 通过时，保留 detached 排序监督并删除直接梯度。
6. **统计单位是视频。**
   同一视频或窗口中的大量扰动不是独立样本。候选数不能替代独立视频数；样本规模
   应由视频聚类方差和预先功效模拟决定。
7. **开发种子不能进入最终统计。**
   第一粒种子只作结构筛选。模型冻结后使用未参与开发的预登记种子完成最终均值、
   方差和视频级配对置信区间。
8. **必须报告定位敏感指标。**
   平均 mAP 之外，至少包括 mAP@0.7、短动作、起止边界误差、最大洞、累计偏移、
   密度熵、动作内/边界/背景采样比例和往返坐标误差。
9. **69 至 70 是强目标，不是信息论保证。**
   项目继续追求该性能，但不能通过增加训练预算、测试集调参或后端改造倒推结果。
   第一科学标准是同预算稳定超过干净 uniform，并恢复可解释的 dense gap。
10. **免训练模式必须与任务适配模式分开声明。**
    二者共用可行域、解码器和坐标适配器；免训练模式的 selector 与发布 detector
    均冻结，不使用目标域优化、教师、缓存或 GT。

## 修正后吸收的意见

### 1. 密度常数是候选，不是定理

审阅提出：

```text
1/(2T) <= p_i <= 2/T
|CDF(i) - i/T| <= 4/K
G_K = ceil(2T/K)
D_K = ceil(4T/K)
```

这些值形成了清楚、可测试的首个候选，但没有理论或实验证据证明它们对 THUMOS14、
K=384 和 K=192 最优，也没有证明其可行域不会过度限制边界微簇。项目吸收“必须有
显式界和预先冻结”的原则，不把这组数值直接升级为 canonical constant。它们应在
训练侧开发数据和几何可达性分析中一次性冻结，正式结果后不得回调。

### 2. 动态规划是实现候选，不是论文贡献

确定性 DP 可以同时满足 exact-K、唯一性、最大洞、首尾覆盖和 anchor 偏移，因此是
合理首选实现。但论文贡献是约束集合与几何证书，不是“用了 DP”。若更简单的单调
量化能够在同一合同下给出唯一解和零违反，也允许替代。

“任何 repair 都说明模型失败”表述过强。正确边界是：不得在正式解码器之后再做
未写入数学合同的临时补洞、膨胀或 uniform scaffold；若冲突处理本身是唯一、确定、
预先定义的解码算法组成部分，则不属于事后 repair。

### 3. 坐标逆映射必须位于物理时间 NMS 之前

非线性单调变换不保持区间 IoU，因此如果先在 q 轴做 NMS、再把保留结果映回 t 轴，
抑制关系可能已经改变。纯插件合同应为：

```text
detector raw proposals in q
  -> plugin inverse map q to t
  -> unchanged official NMS in physical time t
```

NMS 算法和参数不改，但坐标适配必须发生在 NMS 之前。若当前后端无法暴露 raw
proposals，这一接口问题必须在 parity gate 中显式裁决。

### 4. RDD 是优先候选，不是未经验证的唯一答案

rank-normalized distribution distillation 比原始 `|x * grad|` 幅值拟合更抗尺度，
也与密度输出更匹配，因此列为主候选。审阅给出的线性 rank 分布和等权 cls/reg KL
仍需验证 ties、重尾、teacher 非平稳和分类/回归冲突。

`G_rank` 失败应删除当前“连续梯度产生的贡献教师”，但不逻辑性地否定所有任务效用
学习。允许在一次明确的模型重设计中考察硬 counterfactual utility、pairwise/listwise
监督或部署可见运动先验；不得通过不断替换 loss 维持原主张。

### 5. `G_rank` 与 A1/A2 的依赖必须说清

审阅一方面说 `G_rank` 只阻塞 RDD，另一方面又把 A1/A2 定义为
`density + RDD`。因此：

- 投影、解码器、坐标适配和 property tests 可以与 `G_rank` 并行实现；
- 以 RDD 为学习信号的 A1/A2/A3 长训练必须等待 `G_rank`；
- 若 `G_rank` 失败，不能运行没有有效 selector 学习目标的同名长训练；
- 必须先删除当前 RDD 故事或形成一次新的硬效用监督设计。

### 6. checkpoint 规则保留条件分支

公平主表的首选是所有臂在第 6,000 次成功更新后的 EMA，尤其在没有独立训练侧
模型选择集合时。项目不接受“任何中间 checkpoint 一律禁止”的绝对规则：

- 若存在与官方测试完全隔离、预先冻结的训练侧留出集合；
- 所有臂使用相同最大 6,000 updates、评估频率、指标和选择规则；
- 官方测试只在设计冻结后评估一次；

则允许选择中间 checkpoint。否则统一使用 terminal EMA。旧 80/90 轮中间结果仍
绝不能与 60 轮基线比较。

### 7. 数值发表门槛暂不冻结

审阅提出的 `+1.0pp`、恢复 40% dense gap、第二检测器 `+0.5pp`、端到端成本下降
30% 和 selector 开销低于节省主干成本 15%，均是合理目标，但不是理论事实。它们
状态为 `designed_reviewer_proposal`。最终门槛应在 clean baseline、方差和功效模拟
可用后、查看正式主结果前冻结。

### 8. A4 短程分叉只能是开发诊断

从 A3 checkpoint 再训练 1,000 updates 的 on/off 分叉只能判断 direct gradient
是否值得进入模型。它不能把 A4 变成 7,000-update 主结果。若通过，正式 A4 必须从
统一初始化开始，在总计 6,000 updates 内按预登记日程训练。

### 9. 免训练模式不无限延期

免训练结果不与 task-adapted 主表混称，但它共享解码器和坐标合同，实施成本较低。
在 E1 parity 和 E2 decoder/coordinate contract 通过后，可以并行建立一个冻结
released detector 的最小 train-free 基线；不等待所有 task-adapted 多种子完成，
也不让它阻塞主模型。

### 10. PR 清理不进入模型关键路径

PR #3 的 diff 描述必须修正，才能形成清晰公开审阅面。但这不是 CVPR 模型贡献，
也不应排在 clean baseline、唯一数学模型、`G_rank` 和 `G_direct` 之前。

## 吸收后的候选主模型

当前唯一论文候选可概括为：

> 用低成本时序证据预测任务效用密度，将其投影到包含均匀恒等路径、局部伸缩上界、
> 累计偏移上界和 exact-K 覆盖约束的可行域；固定分位生成连续采样位置，确定性
> 解码器产生 K 个真实帧；训练监督只使用通过真实多尺度硬换帧验证的分类/回归效用，
> 后续 TAD 检测器内部保持不变。

它目前仍是 `designed_revised_candidate`，不是 `implemented`、`tested` 或
`empirically_supported`。

论文可能成立的三项贡献是：

1. TAD 定位友好的有界 exact-K 单调时间重参数化；
2. 经过硬策略验证的任务效用 rank-to-density 学习；
3. clean/wrapper parity、warped-time I/O、第二检测器和高 IoU/短动作共同支持的
   严格纯插件证据。

课程、DP、Slurm、日志、哈希、PR 和 checkpoint 管理都不是论文贡献。

## 吸收后的实验逻辑

### 硬前提

1. `E0`：released official dense checkpoint 的同口径评估。
2. `E1`：clean K=384/K=192 uniform 与 wrapper parity。
3. `E2`：exact-K、唯一性、首尾覆盖、最大洞、anchor 偏移、constant-uniform identity、
   q/t round-trip 和 pre-NMS inverse-map property tests。
4. `G_rank`：视频级多尺度硬 swap 是否支持当前贡献排序。
5. `G_direct`：在 `G_rank` 之上，直接梯度的可组合性、共享参数稳定性和短程配对收益。

`E0/E1`、`E2` 和统计工具可以并行。任何 learned-selector 正式长训练至少等待
`E1/E2`；使用 RDD 的臂还必须等待 `G_rank`；A4 还必须等待 `G_direct`。

### 发展阶段因果臂

| 臂 | 定义 | 回答的问题 |
|---|---|---|
| A0 | clean exact-uniform | 公平性能底线 |
| A1 | 无累计/局部界的 monotone density + 已通过 `G_rank` 的 RDD，从零释放 | 学习密度是否有价值 |
| A2 | bounded density + 同一 RDD，从零释放 | 几何约束是否有价值 |
| A3 | A2 + 总预算内的 uniform warmup/ramp | 课程是否有价值 |
| A4 | A3 + 通过 `G_direct` 的 direct gradient | 直接检测梯度是否有额外价值 |

`shuffled-RDD`、简单 motion prior 和 `no-RDD` 只作为固定的短程机制控制，不扩张成
无边界工程矩阵。若 `G_rank` 失败，A1 至 A3 按上述定义均不启动。

第一枚运行是 development screening seed，不进入最终统计。结构冻结后再用三枚
未参与开发的种子比较 A0 与主模型。K=384 成立后扩展 K=192 和第二检测器；真正
train-free 模式使用同一理论表面单列。

## 当前优先级

1. 恢复 official dense、clean K=384/K=192 uniform 和 wrapper parity。
2. 将可行域、连续映射、确定性解码、pre-NMS 逆映射和可证明命题写成唯一数学规范。
3. 运行视频级 `G_rank`，决定贡献教师是否还有资格存在。
4. 运行独立 `G_direct`，决定 direct gradient 是否进入 A4。
5. 只在上述门闭合后运行公平 total-60 development arms。
6. 第一枚 development seed 明确正向后，冻结结构并运行新种子、K=192 和第二检测器。
7. 共享解码器稳定后并行运行最小 frozen-detector train-free 模式。

## 当前状态表

| 项目 | 状态 |
|---|---|
| 原始审阅归档 | `completed / byte_identical` |
| 审阅科学裁决 | `major_revision_accepted_with_corrections` |
| PR #3 纯文档真实性 | `failed_non_scientific_nonblocking` |
| 官方 69.03 发布参考 | `verified_upstream` |
| 本地 released-weight dense 等价评估 | `open` |
| clean native K=384/K=192 uniform | `open` |
| clean/wrapper parity | `open` |
| 有界 exact-K 数学模型 | `designed_revised_candidate` |
| 有界 exact-K 生产实现 | `not_started` |
| `G_rank` | `designed / not_run` |
| `G_direct` | `designed / not_run` |
| 公平 total-60 主实验 | `not_started` |
| K=384 30+60 历史课程 | `terminal_over_budget_diagnostic` |
| K=192 30+60 历史课程 | `terminal_over_budget_diagnostic` |
| 真正 frozen-detector train-free | `open` |
| CVPR 最终模型 | `not_implemented / not_paper_ready` |

## 外部来源复核

- MGSampler, ICCV 2021：累计运动分布驱动的可解释帧采样，证明“累计质量采样”
  不是 DUCA 独有。
  `https://openaccess.thecvf.com/content/ICCV2021/html/Zhi_MGSampler_An_Explainable_Sampling_Strategy_for_Video_Action_Recognition_ICCV_2021_paper.html`
- AdaFrame, CVPR 2019：学习逐视频帧选择与 future utility，证明“学习选帧”
  本身不是新颖性。
  `https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html`
- Benchmarking TAD Robustness Against Temporal Corruptions, CVPR 2024：
  单帧扰动即可明显影响 TAD，主要伤害定位，支持高 IoU、短动作和边界诊断的必要性。
  `https://openaccess.thecvf.com/content/CVPR2024/html/Zeng_Benchmarking_the_Robustness_of_Temporal_Action_Detection_Models_Against_Temporal_CVPR_2024_paper.html`
- Hartley, JASA 1966：累计 size 后进行不等概率系统抽样，支持 inverse-CDF/systematic
  sampling 的经典近邻判断。
  `https://www.tandfonline.com/doi/abs/10.1080/01621459.1966.10480902`


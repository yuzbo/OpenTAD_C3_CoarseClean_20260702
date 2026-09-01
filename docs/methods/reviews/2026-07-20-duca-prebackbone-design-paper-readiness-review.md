---
type: method_review
status: hold_and_revise
date: 2026-07-20
route: DUCA Protected-E2E
---

# DUCA pre-backbone 设计与论文准备度评审

## 结论

当前 Protected-E2E pre-backbone 方法是一个**设计合理、值得做决定性实验的研究假设**，
但还不是一个被证明正确的方法，更不是已经能支撑完整论文的结果。

- 方法层面：`REVISE`。物理 exact-K、同图 hard/soft、hard-forward、
  protected gradient ownership 和原生坐标检测这条主线是合理的。
- 实验层面：`HOLD`。当前只有组件级 `tested_focused`，没有同一精确提交上的
  P0-P3、正式 official-60 mAP、重复种子或完整成本证据。
- 论文层面：`NOT PAPER READY`。现有 LaTeX 仍描述 zero-shot、teacher utility、
  HardTopK、gap repair 和 selected-axis remap 等旧路线，且结果表为空。
- 路线层面：保留已经冻结的四臂裁决，不增加新方法臂；但 P1 必须显式加入
  VideoMAE 不规则时间输入语义门禁。该门禁失败时应先修正表示契约，而不是启动
  长训练。

因此，对三个核心问题的直接回答是：

1. **设计是否合理：合理，但存在尚未排除的结构风险。**
2. **路线和方案是否完全正确：不完全正确；当前正确的是证伪顺序，不是方法结论。**
3. **能否支撑完整论文：有潜力，但现有证据不能；只有在关键门禁、mAP、统计与
   全流程成本全部闭环后才可能。**

## 审阅对象与当前方法重建

本评审以研究 wiki、当前论文草稿和隔离构建树
`.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720` 为准。隔离树仍是未提交
草稿，不是可引用的精确实验提交。

当前冻结候选可重建为：

1. 对完整离线窗口的 768 个候选帧运行低分辨率 coarse probe；
2. 使用 official-source ASFormer 的二分类 action logits 与时序 hidden，
   通过独立的 197-to-64 selector adapter/head 形成转移分数；
3. 在包含 exact-uniform 的同一物理 exact-K DAG 上，以 Viterbi 得到 384 个
   hard 帧，并以 Gibbs slot marginals 提供 soft backward；
4. detector 前向实际接收 hard 帧；protected 主臂只允许 detector loss 更新
   selector adapter/head，rho 臂才以固定 0.01 打开最后一个 ASFormer block；
5. ActionFormer head 在 dense/native 物理轴上训练、解码和 NMS，不重映射 GT
   到 selected axis；
6. 推理不读取 GT、teacher、raw prediction cache 或 counterfactual ledger。

这应被称为“离线、固定预算、内容自适应位置分配”，而不是 Online TAD，也不是
已经实现动态预算的动态计算方法。

## 设计中合理且应保留的部分

### 1. 问题定义与因果归因是清楚的

固定 `K=384` 先只比较“把相同重计算预算放在哪里”，能够把位置分配效果与预算
变化解耦。exact-uniform 又被包含在合法 exact-K family 中，使 learned arm 与
uniform anchor 可以使用同一约束和后端。这是目前路线最强的实验设计资产。

### 2. 物理 exact-K 可行集比旧 TopK/repair 路线严谨

source、internal、sink edge 共用一个物理时间图，hard Viterbi 与 soft Gibbs
不再来自两个不同的可行集；最大物理间隔是求解约束而不是事后补洞。这消除了旧
local bridge、candidate hole 和 post-hoc repair 的主要归因歧义。

### 3. hard-forward 与部署路径一致

当前 straight-through 写法保证 detector 前向张量与真实 hard gather 完全相等，
推理也只使用 hard 帧。它避免了“训练看软混合、部署看硬帧”的直接前向分布差异。
但 backward 是否代表真实离散交换效用仍需 P3，不能由 hard-forward 等式推出。

### 4. protected gradient ownership 是有价值的论文问题

主臂把 action-state representation 与 detector-specific selector adaptation
分开，rho 臂又能检验少量 detector gradient 是否需要进入 coarse trunk。若四臂
结果成立，它会给出比“联合训练有效”更强的机制归因。

### 5. 原生物理坐标与 fail-closed 检查方向正确

不规则采样后仍在 dense/native 时间轴生成 target、decode proposal 与 NMS，是
高 tIoU TAD 的必要条件。当前 metadata 一致性、整数位置、单调唯一、范围和
selected-axis 禁用检查，均应保留。

## 尚未闭环的关键风险

| 优先级 | 风险 | 为什么会动摇结论 | 必须先得到的证据 |
|---|---|---|---|
| P0 | 不规则采样与 VideoMAE 时间语义错配 | 选出的帧按 rank 被重新打包进 16-frame clip/tubelet；相邻输入的真实时间间隔可变。physical head 只在 backbone 之后改 proposal 点坐标，不能自动修复已经按等间隔位置编码产生的特征 | 完整模型 build；uniform-384 端到端 parity；短动作 support；timestamp-spacing 反事实；raw gather→VideoMAE→projection→head roundtrip |
| P0 | hard/soft 梯度方向可能不代表真实 hard swap | 同一 DAG 只证明可行集一致；soft RGB resample 经过非线性 VideoMAE 的局部导数，仍可能与离散替换的 detector-loss 差分反向 | 冻结的 48-window、576-swap、至少 512 effective swaps、video-cluster bootstrap P3 |
| P0 | 完整配置尚不存在 | 现有 768-frame base 使用 48 个 16-frame chunks；384-frame selector 路径需要明确 chunk、mask、feature length、projection 和 physical-head 契约，不能假定继承后自然正确 | 同一正式配置上的 shape、mask、chunk、AMP/DDP 和 official-backend identity gate |
| P1 | coarse supervision 叙述可能过强 | action head 是 binary actionness，但共享 ASFormer trunk 还接收 transition/boundary auxiliary gradients | 分 loss 反向的参数组梯度表；论文只声称 action head binary-supervised，除非另有严格路由实验 |
| P1 | 固定 coverage floor 容易被误称为覆盖保证 | 概率 floor 只是 soft 分布平滑；真正 hard coverage 来自物理 DAG/max-gap | 方法文本分别定义 soft regularizer 与 hard feasibility，不混写 |
| P1 | 全流程成本可能不降或降幅有限 | 当前仍解码、预处理、传输全部 768 帧，低分辨率 probe 也遍历 768 帧；只减少重 backbone/head 输入。训练期 soft resample 还很大 | trained-checkpoint 的 decode-to-output p50/p95、能耗、显存、GPU-hours 和 break-even |
| P1 | 现有代理证据偏负 | allocation ceiling 中 deploy transition geometry 和 frozen detector loss 均劣于 uniform；GT oracle headroom也有限 | 同 checkpoint、同后处理的 exact-uniform 与 deploy transition final mAP replay；最终仍以 mAP 裁决 |
| P2 | 数据与统计外推不足 | 当前主要是 THUMOS、单 seed；现有 transition `+0.4161` 点只是 raw one-seed fact | 正向 pilot 后再冻结种子/CI；至少第二数据集，最好再有一个 backbone 或 detector setting |

以当前 `3×160×160×768` 输入和 `K=384` 估算，训练期
`einsum("bcthw,bkt->bckhw")` 单样本约需 `22.65G` 次乘加，仅是 soft-resample
本身，不含 probe 与 detector。这个数是静态估算而非实测，但足以说明训练成本
不能省略。推理不运行该 soft bridge，仍需单独测量完整路径。

最严重的不是输出坐标，而是 **backbone 的输入时间语义**：即使 head 的点坐标
完全物理化，VideoMAE 仍可能把跨越不同真实时长的 16 个所选帧当作规则 clip。
如果该风险在 P1 中失败，继续调 selector 分数不会解决根因。

## 现有证据能说明什么

| 证据 | 可以说明 | 不可以说明 |
|---|---|---|
| physical DAG focused tests `9 passed` | hard/soft 图在合成输入上的构造与梯度可运行 | 官方 full model、TAD mAP、真实成本 |
| selector/contract focused tests，最终 `24 passed` | selector arm、metadata、部分梯度 ownership 与 fail-closed 契约可运行 | P0-P3、AMP/DDP、official-60 |
| Job `1176948` 的旧候选 P1/P2 connectivity | detector gradient 曾能到达旧 selector 路径 | 当前 physical-DAG 候选通过 P1/P2；P3；性能 |
| CellCF matched seed-0：uniform `63.8594`、transition `64.2755`、CellCF `64.0610` | transition arm 有 `+0.4161` 点单 seed 原始信号；CellCF 不优于 transition | 稳健增益、Protected-E2E 增益、论文主表 |
| allocation ceiling | 当前 deploy proxy 与 frozen loss 存在明显错配风险 | decoded/NMS mAP 的最终 KILL |
| 当前论文空表 | 已有问题定义与审计意识 | 已有完整论文证据 |

当前 claim 状态因此不变：

| Claim | 状态 | 论文前最低闭环 |
|---|---|---|
| C1：低成本 coarse state 足以指导稀疏选择 | `unproven` | deploy-visible selector 在 matched mAP 上稳定优于 uniform/random，并有成本证据 |
| C3：DUCA fixed-K 优于 matched uniform | `unproven` | 四臂 official-60 后的重复种子、置信区间与高 tIoU 分解 |
| C4：direct detector gradient 改善 selector | `unproven` | protected-E2E 显著优于 transition-no-bridge，且 P2/P3 通过 |
| C7：真实总成本下降 | `unproven` | dense-768 对比的完整 decode-to-output 时间、能耗、显存与 break-even |
| 论文最终方法 | `designed/partially tested components` | 不得在 C3/C4/C7 前升级为 `paper_ready` |

## 路线是否正确

### 应继续执行的顺序

1. **P0 协议封存**：精确 train split/loader/update count、train-only
   construction、配置与源码 hash、无泄漏和 official backend 身份。
2. **P1 完整模型语义门禁**：除 exact-K/DAG 外，新增 chunk/feature/mask、
   uniform parity、physical roundtrip、短动作 support 和时间间隔反事实。
3. **P2 四个独立 backward**：detector、action BCE、transition/boundary 与
   rho 路径分别记录 selector/action-head/ASFormer/detector 的梯度所有权。
4. **P3 梯度对齐**：按已冻结的 48/576/≥512/video-cluster protocol 比较
   soft gradient 与真实 legal hard-swap detector-loss 差分。
5. **单次四臂 official-60**：exact-uniform、transition-no-bridge、
   protected-E2E、protected-E2E-rho001，以 terminal EMA final mAP 裁决。
6. **仅在正向结果后扩展**：补 dense-768、random-384、重复种子、完整成本、
   第二数据集与泛化设置。

### 决策规则

- P1 时间语义或 full-model parity 失败：`STOP_AND_REVISE_REPRESENTATION`。
- P3 失败：停止 direct-gradient 主张；same-DAG 不能挽救无效 surrogate。
- protected-E2E 不优于 transition-no-bridge：C4 失败；不能把联合训练写成贡献。
- mAP 正向但全流程成本不降：只能讨论固定预算采样的准确率，不是效率论文。
- 成本下降但高 tIoU/短动作显著退化：不满足项目目标。
- mAP、P3、重复种子、完整成本和泛化都成立：路线才具备完整方法论文条件。

这个顺序总体正确。需要修正的是：不能把四臂训练前的 P1 仅理解为代码和坐标
检查；它还必须验证 backbone 是否理解不规则时间。

## 论文可行性与新颖性边界

广义“为动作检测选择少量帧”不是空白。Yeung 等人在 CVPR 2016 已使用 recurrent
agent 选择 frame glimpses 并输出动作检测；ETAD 研究了 TAD 训练时的 snippet/
proposal 采样；TE-TAD 明确使用实际时间轴坐标；近年的 TAD compression 与视频
识别 frame pruning/selection 也覆盖了效率与任务感知采样。因此不能声称：

- 首个 action detection frame selection；
- 首个 task-aware video frame selection；
- 首个 actual-time/physical-coordinate TAD；
- 仅凭“pre-backbone”就具有充分新颖性。

若实验闭环，较可辩护的贡献是：

> 面向离线高-IoU TAD 的固定预算、物理 exact-K pre-backbone acquisition：
> 以同一合法图统一 hard inference 与 soft learning，并用受保护的
> detector-to-selector gradient 做因果归因，同时报告完整系统 Pareto。

这是一种**组合与实验协议层面的新颖性**。是否足够取决于结果强度：如果只是
THUMOS 单 seed 的小幅增益，通常不足以支撑完整主会方法论文；如果能在两类数据
分布上保持高 tIoU、减少真实总成本，并通过强控制实验解释 protected gradient，
则有完整论文潜力。

相关先例：

- Yeung et al., CVPR 2016:
  https://openaccess.thecvf.com/content_cvpr_2016/html/Yeung_End-To-End_Learning_of_CVPR_2016_paper.html
- ETAD, CVPR Workshops 2023:
  https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Liu_ETAD_Training_Action_Detection_End_to_End_on_a_Laptop_CVPRW_2023_paper.html
- TE-TAD, CVPR 2024:
  https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html
- Temporal Action Detection Model Compression by Progressive Block Drop,
  CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Temporal_Action_Detection_Model_Compression_by_Progressive_Block_Drop_CVPR_2025_paper.html
- Search-Map-Search, CVPR 2023:
  https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Search-Map-Search_A_Frame_Selection_Paradigm_for_Action_Recognition_CVPR_2023_paper.html
- TAPS, ACCV 2024:
  https://openaccess.thecvf.com/content/ACCV2024/html/Dinai_TAPS_Temporal_Attention-based_Pruning_and_Scaling_for_Efficient_Video_Action_ACCV_2024_paper.html

## 现有论文草稿的处理边界

现有 paper 不是对当前 Protected-E2E 的轻量更新对象，而是旧方法叙事：

- abstract/introduction 仍以 zero-shot actionness、motion/energy、teacher utility
  warm-up 和 `window-online` 为主；
- method 仍包含 HardTopK、gap repair、teacher loss 与 selected-axis remap；
- experiments 的核心表全部为空，且 ablation arm 与当前四臂不一致；
- “dynamic compute”与 fixed-K placement 尚未严格区分。

在四臂 mAP 前不应重写为完成稿。当前只可冻结新的论文骨架：

1. offline fixed-budget acquisition 问题；
2. physical exact-K hard/soft selection；
3. protected detector-gradient ownership；
4. native-time TAD contract；
5. four-arm causal matrix；
6. high-tIoU、短动作和 full-stack Pareto。

若路线失败，也不能直接把现有单路线负结果包装成完整论文。负结果论文至少需要
跨 sampler、backbone 和 dataset 的系统性边界研究。

## 审阅限制

本轮未获授权启动并行代理或外部二次审稿，因此这是基于当前仓库、证据登记和
公开原始论文的本地零假设评审。没有运行新训练，也没有把任何 claim 升级为
`empirically_supported` 或 `paper_ready`。

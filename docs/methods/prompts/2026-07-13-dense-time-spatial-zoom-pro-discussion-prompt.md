# Dense-Time Spatial Zoom for Offline TAD: Pro Discussion Prompt

请以最高推理强度完成一次**路线级研究审查**。你不是来替现有想法辩护，也不是来把
Uni-AdaFocus 接入 AdaTAD 后宣布完成。你的任务是判断：在当前 DUCA 选帧路线暴露出
边界覆盖和时间几何问题后，是否应转向“保留密集时间、稀疏分配空间分辨率”的帧内
裁剪/Zoom 路线；若值得，给出一套真正 TAD-specific、成本诚实且可发表的最终候选。

## 0. 强制技能加载与可见性证书

开始分析前，先发现并加载当前环境中所有与本任务相关的 skill。至少检查并在回复开头
逐项报告 `loaded / unavailable / emulated`，不得虚构已加载：

1. 工作流与发散：`superpowers:using-superpowers`、`brainstorming`、`conductor`、
   `divergent-route-orchestrator`；
2. 研究记忆与查新：`research-wiki`、`academic-research-suite`、`research-lit`、
   `systematic-review`、`novelty-check`、`arxiv`、`semantic-scholar`、`openalex`；
3. 严厉评审：`gpt-5-pro`、`research-review`、`ccf-a-editorial-review`、
   `kill-argument`、`paper-claim-audit`；
4. 方法与实验：`innovation-validity-loop`、`formula-derivation`、
   `experimental-design`、`experiment-plan`、`experiment-audit`、
   `ablation-planner`、`result-to-claim`；
5. 实现与成本：`oss-audit`、`pytorch-training`、`cv-detection`、`data-loading`、
   `mixed-precision`、`system-profile`。

若支持并行 reviewer，至少并行设置四个互不共享结论的角色：`prior-art/novelty`、
`TAD method`、`PyTorch/code`、`systems/cost`，最后由一个 adversarial reviewer 统一裁决。
本轮仅做只读审查和设计，不修改仓库、不运行训练、不部署远端、不提交或推送代码。

## 1. 固定审查对象

- 仓库：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- 当前公开代码锚点：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/1f5f7254a390f183121e6c4b7cebcebd2f2954d1`
- 当前本地分支名：`codex/chronotransport-pro-review`

先验证仓库、分支、commit 和逐文件内容是否可见。若 commit 不可见，停止逐行结论并明确
列出不可见内容；不得用 prompt 摘要冒充代码证据。优先读取：

- `AGENTS.md`、`RTK.md`；
- `research-wiki/query_pack.md`、`research-wiki/anti_repetition.md`、
  `research-wiki/ideas/dense-time-spatial-zoom-tad.md`（若公开 commit 尚无此页，以本 prompt
  的设计状态为准，但不得称其已实现）；
- `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`；
- `opentad/models/backbones/vit_adapter.py`；
- official-derived AdaTAD detector、ActionFormerHead、数据 pipeline、raw-video transform、
  evaluator、cost profiler 和 DUCA selector/diagnostic 相关代码。

所有代码判断必须给出 GitHub `file:line` 链接。区分：`代码事实`、`实验事实`、
`文献事实`、`推断`、`设计建议`。

## 2. 不得改写的研究背景

1. 任务是**离线 TAD**，完整窗口可见，不是 Online/Streaming TAD。
2. 当前空间路线状态仅为 `discussed/oracle_gate_required`，没有实现、训练或性能证据，
   也尚未正式取代 DUCA-CellCF。
3. 当前 DUCA 的历史诊断显示：粗动作状态质量只属中等；learned selection 相对 exact
   uniform 损失 boundary radius-1 coverage，并面临 max-gap、selected-axis/physical-time
   geometry 和 hard/soft utility 对齐问题。旧 P0 协议又有 invalid uniform，不能把旧数字
   当 matched 论文结论。
4. 空间 Zoom 的动机不是“裁剪看起来更直观”，而是假设：**高 tIoU TAD 需要密集时间
   证据，但高空间分辨率可能只需集中在少量、时间连续区域。**该假设必须先被证伪检验。
5. 当前 raw-video AdaTAD 配置实际处理 160x160 输入；VideoMAE patch size 为 16，并对
   运行时空间网格插值位置编码。请逐行复核，不要默认基线是 224x224。
6. 112x112 和 96x96 原生 crop 分别对应 7x7=49、6x6=36 个空间 token；160x160 为
   10x10=100。token 减少不是实测 latency，不能直接等同为总成本节省。
7. 若 crop 被 resize 回与 full frame 相同的 heavy-backbone 输入尺寸，则通常只增加局部
   细节，不构成可信 heavy-compute reduction。

## 3. 强制查新的近邻与检索范围

至少阅读官方论文与官方代码：

- Uni-AdaFocus：`https://arxiv.org/abs/2412.11228`；
  `https://github.com/LeapLabTHU/Uni-AdaFocus`
- AdaSpot：
  `https://openaccess.thecvf.com/content/CVPR2026/html/Xarles_AdaSpot_Spend_Resolution_Where_It_Matters_for_Precise_Event_Spotting_CVPR_2026_paper.html`；
  `https://github.com/arturxe2/AdaSpot`
- AdaTAD：
  `https://openaccess.thecvf.com/content/CVPR2024/html/Liu_End-to-End_Temporal_Action_Detection_with_1B_Parameters_Across_1000_Frames_CVPR_2024_paper.html`

继续系统检索 2021-2026 的 spatially adaptive video inference、glance-and-focus、dynamic
resolution、ROI tube/crop policy、token pruning/merging、event spotting、TAD/TAL/TAS/AQA
高效计算、weakly supervised spatial localization。技术结论只能引用论文、官方代码或
官方文档；给出直接链接和检索截止日期。

输出 prior-art collision matrix：`工作`、`任务`、`决策变量`、`监督`、`时间是否稠密`、
`是否实测全成本`、`与本路线重叠`、`剩余不可还原 delta`。重点回答：在 AdaSpot 已存在
后，`low-res global + high-res ROI` 还剩多少创新空间？

## 4. 必须裁决的核心问题

1. 与继续做 temporal frame selection 相比，spatial zoom 是否更符合 THUMOS14 和高
   tIoU TAD 的信息结构？哪些动作需要全局场景、多人物或多个 ROI？
2. 当前 160 输入是否已经保留足够空间信息？若 dense 224/256 没有增益，Zoom 是否应
   立即 KILL？
3. 真正省算需要怎样的 source resolution、decode、resize、H2D 和 native crop pipeline？
   高清源帧常驻是否会吞掉 backbone 节省？
4. ROI 应按 frame、tubelet、clip 还是 temporal track 决策？一 ROI、两 ROI、可变 ROI
   数量与 deformable patch 的稳定性/硬件代价如何？
5. 没有空间框监督时，ROI center 应来自 training-free saliency、人物/运动 proposal、
   differentiable policy、reinforcement learning，还是 train-only counterfactual teacher？
6. 低分辨率 global branch 是否同时承担 context、action-state 和 temporal transition，
   从而避免另加 MobileNet+ASFormer；怎样防止 scout 本身变成第二个昂贵 backbone？
7. official-derived AdaTAD/ActionFormer detector 能否保持完全规则的物理时间网格和目标
   生成？global/local features 应在哪一级融合，才不修改 detector 语义？
8. 边界风险应控制 ROI 的位置、尺度、分辨率、数量还是计算深度？为什么该变量比再次
   丢帧更可识别、更稳定？
9. detector gradient 如何训练 ROI policy？直接穿过 `grid_sample` 是否与最终 hard crop
   utility 对齐？是否更应使用离散候选 ROI 的 hard counterfactual detector regret 蒸馏？
10. Uni-AdaFocus 和 AdaSpot 哪些构件可合法复用，哪些必须作为 baseline，哪些原样搬移
    会使论文因缺乏新颖性被拒？

## 5. 要求发散但互斥的路线

至少提出并比较四条互斥路线，不得只是同一模型调权重：

- A：AdaSpot-like training-free saliency zoom，作为稳定强 baseline；
- B：Boundary-risk-conditioned ROI scale/count，保留全部时间位置；
- C：离散 ROI tube/scale 候选 + train-only dense EMA detector counterfactual regret 蒸馏；
- D：由你从文献与代码中提出的更前沿路线，可质疑 ROI crop 本身，考虑 spatial token
  routing、foveated tokenization、multi-resolution feature transport 或其他更合理变量。

每条路线必须写清：输入、输出、可见信息、训练期特权信息、推理期信息、数学目标、
hard forward、gradient/surrogate、时间与空间坐标、复杂度、失败模式、创新碰撞和 kill
condition。禁止把 GT temporal segment、teacher/cache 或 validation prediction 用作推理
决策。

## 6. 两道先验生死门

在推荐任何 full model 前，先给出可执行、匹配且最小的 falsification：

### Gate S1：Spatial-resolution headroom

同一数据、时间网格、official-derived detector、训练 steps、augmentation 和 evaluator，
比较 dense 160/224/256；报告 Avg-mAP、各 tIoU、短/中/长动作、latency、峰值显存。
给出统计单位、seed、置信区间和明确 GO/KILL 阈值。没有稳定高 tIoU headroom 就停止。

### Gate S2：Oracle ROI sufficiency at equal total cost

定义不依赖测试标签的 deployable baselines，以及仅用于 headroom 的 privileged oracle。
比较 low-res only、fixed center、random、motion/person、AdaSpot saliency、1 ROI、2 ROI、
oracle candidate ROI 与 dense high-resolution。说明没有空间框时 oracle 如何由离散候选的
teacher feature/detector regret定义，防止把 test GT 泄漏包装成方法。匹配的是实测总成本，
不是仅匹配 crop 面积或理论 token。

任一 gate 失败时，必须给出明确 KILL，不得用新增 policy/loss 延长路线。

## 7. 若 Gate 通过，必须给出的最终候选

不要只给逐步实验计划。选择且只选择一个最终候选，给出：

1. 一句话 thesis 与相对 Uni-AdaFocus/AdaSpot 的不可还原 TAD-specific delta；
2. 完整 forward：高清原帧如何产生低分辨率 global view、ROI tube、native 96/112 local
   tokens、global/local fusion 和 official-derived AdaTAD detector 输入；
3. 张量形状、时间/空间坐标、padding/mask、位置编码与 batch 组织；
4. 数学目标：global/local/fused TAD loss、ROI stability、budget/cost、boundary-risk 或
   counterfactual utility；逐项写明监督来源和梯度所有权；
5. 单 checkpoint 的训练方案。允许稳定 warm-up/连续解冻，但不得包装成三个独立模型；
6. 推理合同：无 GT、无 teacher、无 dense prediction cache、规则时间网格、真实执行预算；
7. 与现有 OpenTAD/AdaTAD 的最小改动文件清单、关键类接口、伪代码和 focused tests；
8. full-stack cost equation：
   `decode + preprocessing + H2D + global scout + ROI policy + crop + local heavy backbone + fusion + detector + postprocess`；
9. 最可能失败的五个原因，以及每个原因最小诊断。

## 8. 论文与实验闭环

给出最小但足以审稿的矩阵：dense160、dense high-res、low-res only、fixed/random ROI、
Uni-AdaFocus direct port、AdaSpot-like、最终候选、当前强 uniform temporal baseline；再给
组件消融、第二 detector/第二数据集、三 seed、mAP@0.3:0.7、短动作、ROI jitter、context
retention、真实 p50/p95、显存、能耗与 accuracy-cost Pareto。

生成 result-to-claim matrix：S1/S2/主实验分别成功或失败时，哪些 claim 可以写、必须降级
或必须删除。不要用“有潜力”代替证据。

## 9. 强制输出顺序

1. Skill Loading Certificate；
2. Repository Visibility Certificate；
3. 已核验事实与未核验假设；
4. 严厉总裁决：`GO / HOLD / KILL`，以及“现在是否应放弃选帧主线”；
5. Prior-art collision matrix；
6. 当前代码/数据/成本条件是否支持空间 Zoom；
7. 四条互斥路线及红队批评；
8. Gate S1/S2 的精确定义与 kill rules；
9. 唯一最终候选的完整架构、公式、训练和推理合同；
10. 实现映射与关键伪代码；
11. 最小论文实验矩阵与 full-stack cost protocol；
12. Result-to-claim matrix；
13. 模拟 CVPR/ICCV 审稿：Summary、Strengths、Weaknesses、Questions、Score、Confidence、
    以及什么证据能真正提高评分；
14. 最后用不超过 12 条列出确定执行顺序。

全文使用清晰中文，必要公式和代码标识保留英文。不得使用空泛的“可进一步探索”；每个
结论必须有代码、文献、实验设计或明确推理依据。允许彻底否定本构想，也允许提出比
空间裁剪更合理的替代路线，但必须给出同样严格的成本与可证伪合同。

# GPT-5 Pro Prompt: GeoRoute-AdaTAD Design Audit and Optimization

Use this prompt only through a verified GPT-5 Pro route. It requests a
code-grounded audit and a single executable design verdict, not a generic idea
list.

---

你是一个极其严格的 CVPR/ICCV 级视频理解、稀疏 Transformer、TAD 与系统效率审稿人。请审计下面的离线 TAD 新路线，并且只在你真实读取了指定 GitHub commit 与源码后作答。

## 0. 可见性门槛

仓库：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
分支：`codex/spatial-zoom-s1-audit-fix-20260715`
当前可见基线 commit：`8ebe5f069494dc2efb3d4f9dc1ea3a2fbb51f89c`

先打开仓库、分支和该 commit。明确写出：可访问或不可访问。若不可访问，立即停止，不得根据本 prompt 假装做逐行审计。若可访问，列出你实际阅读的每一个文件及 `file:line`，至少包括：

1. `AGENTS.md`、`RTK.md`；
2. `research-wiki/query_pack.md`、`research-wiki/anti_repetition.md`、`research-wiki/ideas/geo-route-adatad.md`；
3. `docs/methods/2026-07-23-georoute-adatad-design-and-experiment-plan.md`；
4. AdaTAD/ActionFormer detector、projection、head、loss、NMS 路径；
5. `opentad/models/backbones/vit_adapter.py` 的 patch embedding、attention、adapter；
6. 当前 Continuous ROI/native crop wrapper 与 transform；
7. 相关配置、训练器、数据 transform、focused tests。

若第 3 项在该 commit 尚不可见，明确写“设计报告未被 GitHub commit 固化”，把它当作外部附件阅读，不得虚构 GitHub 行号。若你发现实际最新 commit 不同，也必须报告差异并停止把旧 commit 当作最终模型实现。

## 1. 研究边界与最终目标

任务是 **离线** Temporal Action Detection，不是 Online TAD，不做时序选帧，不降低整幅原图分辨率，也不使用手动/GT 特权裁剪结果决定路线。目标是：

> 在完整 AdaTAD-derived 检测通路中，以当前真实 Focal 分类与 DIoU 边界回归损失学习连续可变 `(cx, cy, w, h)` 的空间计算分配；主干仍保留原始分辨率的原生 VideoMAE `2 x 16 x 16` 时空 patch，不使用固定输出尺寸的 `grid_sample` 重采样；同时只允许一次重 VideoMAE 前向。当前配置没有独立 quality head 或 quality loss，不得虚构该项。

模型候选称为 `GeoRoute-AdaTAD`：

```text
light global scout
  -> continuous ROI trajectory
  -> ROI structured native-token scores + free residual TokenSelect + context allowance
  -> exact-K packed native tokens
  -> one VideoMAE forward
  -> geometry/mask-aware sparse temporal adapter
  -> unchanged AdaTAD projection, ActionFormerHead, losses, NMS
```

完整 768 detector 时间轴必须保留。当前 Continuous ROI/S1/S2 代码严格视为历史实现或可复用部件，而不是已经证明此模型的证据。

## 2. 参考方法必须逐项核查

### Uni-AdaFocus

以论文 `https://arxiv.org/html/2412.11228` 与官方仓库
`https://github.com/LeapLabTHU/Uni-AdaFocus` 为直接参照。核查：

- 全局观察器、连续 `cx,cy,w,h`、时序插值、几何反退化是否被正确借鉴；
- 原方法是否固定输出尺寸重采样、是否以分类代理训练策略、局部动作是否 detach；
- 为什么这些机制不能原样迁移到“原生 patch + 真实 TAD detector loss”；
- `16 observation / 48 segment` 的含义是否被误写成“每 16 帧一个 ROI”。

### A-MoD

以论文 `https://arxiv.org/html/2412.20875` 为直接参照。请独立查证是否存在作者公开官方实现；若不存在，明确写“paper-exact reproduction”，不要假称官方代码对齐。逐项审计：

- 前层 attention **列均值**重要度、exact capacity、identity bypass、不得将 routing score 乘回输出；
- 最终要求是 `dense prefix -> Dense -> MoD -> Dense -> MoD -> ...` 的间隔层 Dense-MoD，不是后半段连续 all-MoD；每个 MoD 必须从紧邻前一 dense block 的完整 attention 得分；
- SDPA/FlashAttention 下如何避免物化全 `N x N` attention map；
- A-MoD 评分不能被夸大为对硬排序的完整直接 detector gradient；
- 它在 VideoMAE/AdaTAD 上是否真正节省端到端成本仍是待验证问题。

同时对照 MoD、ToMe、DynamicViT/TokenSelect 的原始定义，指出它们与 ROI 的公平比较边界。

## 3. 必须严厉审计的核心问题

请逐条给出 `P0/P1/P2/P3` 级别、`file:line`、可复现理由和最小修复，不要泛泛而谈：

1. **可微性诚实性**：原生整数 patch 的 exact-K gather 对连续 ROI 是离散支持集。所谓“真实检测梯度学习 ROI”是否成立？必须区分 dense relaxed pathwise gradient、硬策略的 score-function estimator 与有偏 straight-through surrogate；不得把后两者写成相同的精确硬路径梯度。判断何者在单重主干下具有可接受的方差、成本与 detector utility，并要求对应 known-answer test。
2. **真正原生裁剪**：是否在任何地方偷偷发生整图 resize、固定局部 resize、全量 patch embedding 后再 mask、硬编码 `180x320`、坐标失真或 padding leakage？
3. **时空 token 语义**：一个 `2x16x16` token 是否在连续两帧保持同一绝对空间 patch？ROI 在两帧间移动时是否破坏 VideoMAE tubelet 语义？完整 768 时间轴如何保持？还应核查 Uni-AdaFocus 的 `16` observation / `48` segment 术语不能被误写为“每 16 帧一个 ROI”。
4. **AdaTAD 完整性**：新 backbone 是否真的能保留 projection/head/loss/NMS；哪些修改使“official AdaTAD”表述不再成立；实际 detector loss 是否能回到 policy 参数；梯度是否只经被选 token 而遗漏未选成员？
5. **稀疏 adapter**：现有 dense adapter 的哪些 shape/reshape/position 假设失效？最小的 mask-aware、absolute-coordinate、ROI-relative-geometry 设计是什么？是否应采用 geometry-conditioned deformable temporal alignment，还是这只是无必要工程复杂化？
6. **ROI 与自由 token 选择**：ROI 的结构性在 TAD 中是否有独特、可证伪优势？`ROI-only`、`free TokenSelect`、`ROI+residual` 如何配置成同预算、同输入、同训练分布的对照？若 free TokenSelect 胜出，应如何收窄或放弃 ROI 主张？
7. **A-MoD integration**：Dense-MoD 间隔层是否能保持“前一 dense attention -> 后一 MoD”关系？如何设计 `C=1` parity、random exact-K、linear MoD 和 A-MoD 数值 KAT？若不能维持 fused attention，是否应从主模型删除 A-MoD？
8. **ToMe/Token merging**：ToMe 在 patch embedding 之后发生时，为什么不能被当作 pre-backbone 空间裁剪的等价效率对照？如何公平报告 total cost 与内部 token FLOPs？
9. **成本与论文主张**：列出必须计入的 decode、preprocess、H2D、scout、router、gather、patch embed、backbone、adapter、detector、NMS、同步、显存、能耗。指出任何可能让“稀疏 token”反而更慢的 gather/padding/packing 风险。
10. **实验完整性**：检查 split、seed、checkpoint selection、development-only 设计、official-test sealing、无 GT/teacher/oracle/raw-prediction leakage，以及历史 S2 training-only receipt 不得被误用。

## 4. 唯一交付

不要给泛泛新想法清单。请输出一份可直接编码和排队的 **GeoRoute-AdaTAD v1 实施裁决**，包含：

1. `ACCEPT_WITH_FIXES`、`HOLD` 或 `KILL` 三选一；
2. 最终最小模型图和模块接口/shape；
3. 明确的 gradient estimator、损失项、warm-up/finetune 日程；
4. Dense-MoD 精确层级调度和 capacity 定义；
5. 最小代码改动列表，按 `file:line` 和优先级排序；
6. P0、P1、P2、P3 的唯一顺序、每阶段精确 baseline、通过/HOLD/KILL 条件；数值阈值、knot 数、K、scout 分辨率与 CPU/GPU gather 方案若无结果盲校准或代码证据，不得伪装成不可变理论常数；
7. 所有必须保持不变的 AdaTAD/数据/训练/成本设置；
8. 最窄、诚实的论文 claim 与明确禁止的过度主张；
9. 只给一个下一步，不得因为新颖性降低证据标准。

请把可证伪性、检测性能和真实成本置于工程框架复杂度之前。若发现这一方向不能胜过 free TokenSelect，或不能在单次重主干下建立诚实的 detector-gradient 学习，请明确 KILL 或提出最小收缩，而不是以复杂工程掩盖问题。

---

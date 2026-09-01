# DUCA SCOPE-DK 第三份外部审查：对照与核验（2026-08-21）

## 来源与证据边界

- 本文记录用户提供的《总评》副本（752 行）。副本建议 `PIVOT` 并命名
  `SCOPE-DK`，但未附可复核的审查会话、header 或原始转录。因此其机制和实验设计是
  **外部建议**，不是自动生效的科学裁决。
- 该文本能独立核验的公开父版本是
  `42dba3f90b37243e7965d18b6707e88e81bf7109`。本地对象库同时保有 Fovea
  `4ae5067100c4490c7110c00a1ad406230ba603cd` 与 UVT
  `df544c78ce515d925dc7019f106fce09a53c09f8`，但相应分支未见于 origin refs。
  所以“本次 GitHub 审查不能逐行复核后二者”成立；不能推论代码、训练或其单 seed
  数值不存在。
- 本页只记录审查和本地代码核验。没有新增实现、数据访问、训练、成本、性能或论文 claim。

## 本地复核后接受的事实

1. 历史 `42dba3f` 的 selector 在 backbone 前 gather 非均匀 RGB，`TwoStageDetector`
   以 selector 输出替换输入、mask 与训练 target 后再调用 backbone。因此历史
   `65.385724` 确实涉及间接、非均匀的 pre-backbone RGB 输入，而不是 uniform 结果。
2. 该版本把按 selected rank 排序的帧 reshape 为 24 个 16-frame VideoMAE clip；
   `BackboneWrapper` 只接受 frames/mask，并不接收 source index、timestamp 或间隔。
   selected-axis proposal 虽会在 NMS 前回映 true-time，但这不能使 VideoMAE 或 detector
   内部获得真实物理时间度量。
3. 历史输入流水线先对 768 帧完成 decode、resize、crop/augmentation 与 tensorization，
   selector 只位于该阶段之后。因此减少 VideoMAE 输入不等于自动降低完整 decode、预处理、
   H2D 或 scout 成本。
4. 代码没有在紧邻实际 `self.model.backbone(...)` 的调用点强制产出 `executed_k` 收据；
   `requested_budget`、`effective_budget` 或 metadata 中的 selected count 不能替代真实
   重骨干执行量。当前新候选的 `executed_k` 表述仍是设计/静态主张，未成为运行事实。
5. 历史恢复脚本已校验 clean revision、checkpoint、optimizer、scheduler、EMA 与 grad scaler，
   并从 `tools/train.py --resume` 恢复；但 RNG 仅为可选 presence 记录，DataLoader cursor、
   累积梯度与逐 step replay 没有被强制封存。不能称为完整确定性恢复闭环。
6. `65.385724` 是 30-epoch uniform warm start 加 60-epoch joint course，且同时带
   `density_transport_st`、detector gradient/contribution bridge、uniform companion 与
   full ASFormer adaptation。它是值得追查的历史输入信号，不能归因为一个 selector 机制。
7. UVT `1244840` 与 Fovea `1244851` 是已完成的真实、单 seed 开发训练；二者都是复合
   干预，尚未形成同提交、可归因、可写入论文主表的 mAP 或成本证据。

## 三份审查的一致内核：条件接受

- 不应原样继续 UVT/Fovea 的直接选择或多机制捆绑训练；这不等于删除其 Query 上下文或
  训练期语义知识迁移。它们只能作为 action/start/end 语义 scout 的单变量候选。
- 主方法身份仍是：低成本 scout 预测 action/start/end（及校准后的边界风险），确定性规则
  产生物理采样与每视频/窗口 dynamic K；fixed-K 仅为公平控制、归因和回退。
- 历史 selected-rank 伪连续 VideoMAE 输入、输出端 true-time remap、以及配置 K 不等于
  真实工作量，是必须正面解决的三个运行时问题。
- 任何新主张都须以真实重骨干执行量、完整路径成本、official split/evaluator 与预注册的
  fixed-K/内容—预算错配控制来检验；目前没有 dynamic-K efficacy 结论。

## 第三份与前两份并不完全一致

| 决策维度 | RiskClip | BSC-DK | SCOPE-DK | 需要先冻结的含义 |
| --- | --- | --- | --- | --- |
| dynamic-K 命题 | 达到残余风险阈值所需的最小预算 | 固定总预算下向复杂窗口重分配 | 达到语义覆盖的最小 cliplet 数 | 成本节省、固定成本重分配与覆盖阈值是不同主张，不能合用结果。 |
| 重骨干/检测器合同 | 倾向冻结插件诊断 | 所有臂同等 full training | Stage 2 sealed 初始化，Stage 3 训练一个 variable-support detector | 必须二选一：冻结 detector 的机制诊断，或所有新 runtime arms 等更新数 full training。 |
| 物理重建 | 无参数 scatter/interpolate | masked-zero scatter | `PhysicalLatticeSplat` 加 support mask/evidence-density channel | 三者改变 detector 输入分布的方式不同；splat 与新增通道本身可能造成增益，必须隔离。 |
| 独有归因控制 | same-K uniform、K-shuffle、actionness-only | BSC-FIX vs BSC-DYN | 同一 RGB 帧集的 `GAPPACK` vs `CONTIG` | SCOPE 的连续性对照有价值，但必须证明两臂真用同一 source frame set、同一执行 K。 |
| Query/KD 位置 | 语义 residual / detached 语义 KD | 同左，但 BSC 设计把 SQ/SQD 作为较强门 | S0/SQ/SQD 语义门后再进 TAD | 若 SQ 失败，只否定 Query 机制；不应自动否定 S0 的语义间接 dynamic-K 主线。 |

## 对 SCOPE-DK 的具体判断

### 认可并应保留的建议

1. 把 16-frame VideoMAE 单元限制为真实连续原始帧、让 timestamp/physical index 贯穿到
   proposal/NMS，具有明确的预训练—边界定位机制，值得作为下一个运行时合同候选。
2. 在 backbone 调用处而非配置/metadata 中记录 `executed_cliplets`、`executed_k_frames`、
   dummy/padding/重复执行及分段耗时，是判断“真实减少 VideoMAE 计算”的必要条件。
3. `GAPPACK` 对 `CONTIG` 可成为新的、不重复官方 dense/uniform/random receipt 的输入合同
   归因实验；但它仅解释连续性，不能替代官方 baseline 或 dynamic-K 证据。
4. K-shuffle（保持 K 分布、打乱内容—预算对应）是 dynamic-K 阶段不可缺少的强控制；此前
   BSC-DK 方案确实缺少这一项。

### 不能直接接受或需收缩的建议

1. `J=16..32`、48×16 cliplet、风险公式、coverage 阈值、`0.85×384`、`-0.30 pp` 等均是
   proposal；没有本地实现和真实数据证据，不能写成既定成功门。
2. `PhysicalLatticeSplat` 的 support mask/evidence-density channel 会改变 detector 输入，
   因而不能在没有“相同 sampler、仅改变重建器”的对照时被视为纯物理时间修复。
3. Stage 3 训练一个 variable-support detector 可以检验完整自适应系统，但不能被表述成
   selector 单因素收益。它与“所有臂冻结 detector”的机制诊断是不同实验，不能共用结论。
4. “Query 语义门失败即停止整个路线”过强。SQ 失败应停止 Query 协同 claim；SQD 失败应
   停止 teacher claim。只有 S0 的语义覆盖或 dynamic-K 对相同平均 K 的比较失败，才触及
   核心机制。
5. 不能把“selector 后的 VideoMAE 输入变短”误写成端到端加速。新的收据必须分别报告
   decode、preprocess、H2D、VideoMAE、splat、detector、NMS 与 end-to-end 成本。
6. 审查列出的 prior-art 风险尚未构成覆盖性文献核验；不能声称新颖性已通过或已失败。

## 本次吸收后的事实边界

当前没有采纳 `SCOPE-DK` 为新方法名或已冻结实现。三份意见共同支持的、且仍然只是路线
内核是“语义预测 → 确定性物理采样 → dynamic K”。下一个实现合同必须先明确预算命题、
detector 训练合同与重建器，再将 physical-contiguity 和真实 `executed_k` 作为可测试约束。
现有开发训练和历史 65 只能作为诊断证据；不构成 SCOPE、RiskClip、BSC 或任何动态预算方法的
效能支持。

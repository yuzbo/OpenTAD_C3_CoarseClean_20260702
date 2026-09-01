# DUCA BSC-DK 独立审查：对照与核验（2026-08-21）

## 来源与适用边界

本文记录用户提供的《独立审查结论》副本。该副本推荐
`BSC-DK`（Boundary-Semantic Certificate for Dynamic K）并给出 `PIVOT`，但未附可复核的
会话标识、header 或原始转录；故它是外部建议，不是自动生效的路线裁决。

它与前一份 `DUCA-RiskClip` 审查共享诊断框架，但在预算机制、训练合同、重建方式与实验
门槛上存在实质差异。两者的具体参数均未冻结。

## 本地核验：接受的事实

1. 历史 `42dba3f` 的 selector 在 backbone 前按 `selected_positions` gather RGB，
   `TwoStageDetector` 以 selector 输出替换 `inputs/masks/gt_segments/gt_labels` 后才进入
   backbone；这是 pre-backbone 的非均匀 RGB 路径。
2. 推理 proposal 在 `batched_nms` 前由 selected-axis 回映 true-time；该顺序保证几何 NMS
   使用物理坐标，却不能让 VideoMAE 追溯已失去的真实相邻帧间隔。
3. `_gather_time`、历史 BackboneWrapper 都不携带 source frame interval/timestamp；故已选择
   帧的 rank 邻接会被 VideoMAE 当作 clip 内时间邻接。
4. `65.385724` 的 30+60 epoch 课程、uniform warm start、`density_transport_st`、贡献蒸馏/
   gradient bridge、ASFormer adaptation 与 companion exposure 共同存在；不能归因为 selector
   或非均匀位置的单因素增益。
5. UVT `1244840` 与 Fovea `1244851` 确有 60-epoch 单 seed 开发训练，分别给出负的 UVT
   bundle 结果和未分解的 `query_cycle` 信号；二者没有构成论文级 matched evidence。

审查所谓“UVT/Fovea 无法独立 GitHub 复核”应精确理解为：`4ae50671`、`df544c78` 在本地
对象库存在，但相关 2026-08-19 分支未见于 origin refs。它不能被写成代码不存在或这些训练
从未发生。

## 两份审查的一致核心：条件接受

- 不能原样延续 UVT/Fovea 的直接选择与复合训练 bundle；负结果只否定各自 bundle，不否定
  “语义间接采样 + dynamic K”问题。
- 历史 65 是混杂的 pre-backbone 非均匀输入信号，不是公平主表结果。
- Query 与 teacher 只能改善 action/start/end 语义 scout，不能输出 index/K、读取 detector
  feedback、读取部署期 GT 或成为额外 detector 更新来源。
- fixed-K 必须是归因/回退，dynamic K 是最终核心；selected-rank 伪连续必须被新的物理时间
 运行时合同正面检验；`requested_k` 或配置字段不能代替实际 VideoMAE 计算审计。

## 两份审查不是同一实现路线

| 维度 | RiskClip 审查 | BSC-DK 审查 | 影响 |
| --- | --- | --- | --- |
| 核心预算机制 | 残余风险阈值决定每窗口最小 M | 在每视频总 K 固定时，把 cliplet 从简单窗口重分配到边界复杂窗口 | 前者检验成本—性能，后者检验固定成本下的重分配收益；不可混称同一主张。 |
| 动态预算集合 | K=`{256,320,384,448,512}` | K=`{256,272,...,512}` | K histogram、可行性和成本不同。 |
| 重型模型合同 | VideoMAE/detector 全冻结的插件范式 | 允许 VideoMAE/adapter/detector 接受相同原生 TAD loss、训练 60 epoch | 这是最重要的身份冲突，必须在实现前二选一并预注册。 |
| 未观察特征 | 无参数 scatter/interpolate | masked-zero scatter、禁止 selected-rank interpolation | 会改变 detector mask、跨 clip 上下文与输入分布，不能假定等价。 |
| 语义 P0 | calibration/Brier/PR-AUC 等，较低数值门槛 | 160/40 划分、20 epoch、BHit 与更高的固定门槛 | 分割、指标和阈值均为候选，不可混用或事后择优。 |
| 动态归因控制 | same-K uniform、K-shuffle、actionness-only | BSC-FIX 对 BSC-DYN | BSC 还缺显式 K-shuffle；否则难以排除 K histogram 或固定的窗口位置策略。 |

## 不接受为既定决定的部分

1. 不直接采纳 `BSC-DK` 名称、48×16 cliplet、证书公式、`rho`、160/40 split、20/60 epoch、
   K 范围、百分点门槛、1% infeasible 门槛或 checkpoint 规则；它们均是可检验 proposal。
2. 不接受“SQ 或 SQD 不通过就终止整个 DUCA 路线”。Query/KD 在审查自身也被降为辅助语义
   机制：若 SQ 不通过，应该停止 Query claim、保留 S0 的语义间接 dynamic-K 可检验性；若 SQD
   不通过，只停止 teacher claim。只有核心 fixed-K/dynamic-K 预测失败才可能终止相应主方法。
3. 不接受 SQD 默认成为后续 TAD 主臂。S0、SQ、SQD 必须先在冻结的训练 population 语义门中
   以预注册规则选择；teacher 的训练成本单列，且不应因增加训练资源而自动进入 headline。
4. 不接受 masked-zero/scatter 已与 AdaTAD 相容。它要先验证 shape/mask、真实连续 VideoMAE
   输入、timestamp metamorphic、pre-NMS trace、无 Kmax padding 与 runtime `executed_k`；否则
   性能变化可能来自丢失跨 clip 上下文。
5. 不接受审查中 prior-art 条目已经完成 novelty 清除。它们提示新颖性风险，但仍需独立、覆盖性
   文献核验，尤其应检索是否已有“边界语义 + pre-backbone physical cliplet + fixed-total-budget
   dynamic reallocation”的完整 TAD 系统。

## 本次吸收后的未冻结决策

当前路线内核不变：语义 scout -> 确定性物理采样 -> dynamic K；fixed-K 是 control。下一份
实现合同必须先锁定一个问题：新的 physical runtime 是 (A) 绑定冻结 detector 的严格插件诊断，
还是 (B) 所有臂完全同训练预算的 full-training TAD 方法。两者都合法，但不能在同一主表混用。

对于 BSC 的固定总预算主张，最小动态阶段还需加入预注册 K-shuffle（保持每视频/window K
序列或 histogram，只破坏内容—预算对应）或等价的强控制；否则无法证明收益来自“边界复杂度
驱动的预算重分配”。没有新增代码、数据、训练、性能、成本或论文 claim。


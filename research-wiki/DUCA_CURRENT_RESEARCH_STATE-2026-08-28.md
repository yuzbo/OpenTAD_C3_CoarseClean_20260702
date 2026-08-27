---
type: current_research_state
project: DUCA efficient temporal acquisition for temporal action detection
updated: 2026-08-28
evidence_status: point_estimates_available_paired_interval_missing
---

# DUCA 当前研究状态（2026-08-28）

本文是已有代码、研究 Wiki、原始实验结果和终态回执的压缩索引，不产生新的科学事实。若本文与原始实验日志、固定代码版本或评估产物冲突，应以原始证据为准，并显式记录冲突。

## 1. 论文问题与当前科学边界

DUCA 面向离线时序动作检测（Temporal Action Detection, TAD），研究如何在重型视频骨干网络之前，用低成本模型预测逐时刻动作性和动作边界重要性，再由确定性规则选择非均匀的原始视频帧，从而减少 VideoMAE 的真实高分辨率输入，同时保护高时间交并比下的动作边界定位性能。

当前长期目标包含两个层次：

1. **语义间接选帧：** 侦察模型学习动作性和边界语义，帧位置由确定性采集规则间接产生；直接学习离散帧索引只能作为对照。
2. **动态预算：** 根据逐视频或逐窗口的语义证据决定保留帧数，使重型路径真实执行不同工作量。固定 `K=384` 仅用于机制归因、公平对照和回退，不构成最终动态预算主张。

目前尚无证据证明动态预算能够提高性能或形成性能—成本联合优势。当前完成度最高的受控子问题，是固定并重放 H65 的 `K=384` 非均匀选帧结果，只改变重型视频编码器中的物理时间表示。

## 2. 代码与实验身份

| 对象 | 固定代码身份 | 作用与边界 |
|---|---|---|
| 共享官方 AdaTAD | `01c58b9f2370e914150cf94d392208a4e211c053` | 只读共享官方复现；DUCA 不重复训练 |
| H65 30+60 参考 | `04c35a3b76897e6c1569eeede41ed3aecaf7f854` | 历史 ASFormer 语义预测、确定性非均匀逐帧选择、固定 `K=384` |
| H65 60 轮学习率诊断 | `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f` | 只调整第二阶段学习率日程；未恢复 H65 30+60 |
| First-Mixing SingleClock | `b2ccfccab5b4912b59954afcc9b0364955327f7c` | 物理时间表示的早期单时钟候选 |
| PJST-D1 匹配训练 | `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508` | 固定并重放同一 H65 选择结果；只改变首次 tubelet 混合前的时间导数表示 |
| PJST-D1 只读终态评估 | `7bd120f0d342bf175c97c365fba7cbd359df055e` | OFF/ON epoch-59 EMA 推理和预登记配对统计入口 |
| UVT | `df544c78ce515d925dc7019f106fce09a53c09f8` | Utility Value Token 首版诊断；与 H65 不是同提交匹配归因 |
| Fovea/Query-Bridge | `4ae5067100c4490c7110c00a1ad406230ba603cd` | Foveated Sampler 与 Query-Bridge 首版诊断；与 H65 不是同提交匹配归因 |

项目根目录包含多路线协调材料和未归档修改，不能作为论文实验代码身份。任何性能结果必须绑定上表中的独立干净提交、配置、数据划分、检查点选择和官方评估器。

## 3. 已完成实验与真实结果

除特别说明外，平均检测精度（Avg-mAP）是时间交并比阈值 0.3、0.4、0.5、0.6、0.7 下 mAP 的平均值。

| 实验 | Avg-mAP | mAP@0.7 | 可以得出的结论 |
|---|---:|---:|---|
| 共享官方 dense AdaTAD | 68.73 | 未在当前材料重复摘录 | 一次共享官方复现；公开论文锚点为 69.03 |
| H65 30+60 | 65.13 | 43.31 | 当前最强干净 H65 参考；单种子 |
| H65 20+40 | 62.46 | 39.94 | 简单压缩训练明显下降；不否定 H65 选帧机制 |
| H65 30+30，AM-RPCH25 | 63.22 | 41.25 | 只改变第二阶段学习率日程，仍未恢复 30+60 |
| H65 30+30，LongCosine-H6000 | 63.56 | 41.01 | 延缓衰减仍未恢复 30+60 |
| RankPack `K=384` | 61.57 | 37.10 | 单种子物理时间表示对照 |
| TrueTime `K=384` | 62.19 | 37.89 | 相对 RankPack 小幅提高，尚无配对区间 |
| PJST-D1 OFF | 65.063283 | 43.646027 | 211/211 视频，固定 H65 选择和原表示 |
| PJST-D1 ON | 64.590802 | 43.768766 | 211/211 视频，只启用 PJST-D1 表示 |
| 连续片段 FZ | 49.89 | 29.68 | 真实完整训练的明显负结果 |
| 连续片段 JT | 47.24 | 26.52 | 联合训练没有恢复连续片段路线 |
| UVT legacy / geometry / geometry+EMA | 57.35 / 55.93 / 55.92 | 33.84 / 30.02 / 30.49 | 首版新价值头没有超过 legacy；存在选择分数和预算证据同时变化的混杂 |
| Fovea/Query-Bridge 最佳 `query_cycle` | 54.67 | 31.63 | 单种子首波最佳臂；缺同提交 H65 匹配归因，不能证明 Query-Bridge 的独立效应 |

历史 `65.3857` 属于 H65 语义间接非均匀逐帧选择的 30+60 诊断结果，并非均匀选帧。历史 `65.696` 来自改变物理检测网格的探索实现；它同时改变了检测器时间几何，因而不是与官方原生检测器严格匹配的纯输入采样比较。当前 `65.13` 更适合作为后续同代码、同训练协议的 H65 参考。

## 4. PJST-D1 当前证据

PJST-D1 的目标是隔离“同一组选中 RGB 帧如何被重型视频编码器解释时间”这一表示变量。OFF 与 ON 使用相同 H65 选择结果、相同 `K=384`、相同检测器、损失、非极大值抑制、数据划分、训练更新数和官方评估器。

两臂的 epoch-59 指数移动平均检查点均完成官方 validation 211 视频推理，每臂产生 422,000 条预测，视频标识集合一致。ON 相对 OFF 的点估计为：

- Avg-mAP：`-0.47248126` 个百分点；
- mAP@0.3：`-0.79522098`；
- mAP@0.4：`-1.25244448`；
- mAP@0.5：`-0.14698444`；
- mAP@0.6：`-0.29049525`；
- mAP@0.7：`+0.12273884`。

预登记的 10,000 次整视频配对自助法没有开始。统计终结器寻找 `work/result_detection.json`，实际预测位于 `work/gpu1_id0/result_detection.json`，因此在任何 bootstrap shard 和任何重复抽样之前退出：`0/16` shards，`0/10000` replicates。

因此，目前可以说 PJST-D1 的平均性能点估计没有正向支持，但不能说总体效应已经显著为负；`mAP@0.7` 的小幅正差也不能解释为真实收益。路径故障是证据生成失败，不是模型本身的科学结果。上一轮执行已经关闭，本文不构成自动修复或重试授权。

## 5. 已停止或不能重复解释的方向

1. H65 60 轮压缩、AM-RPCH25 和 LongCosine-H6000 均未恢复 30+60，压缩训练搜索已经停止。
2. 固定连续 16 帧片段采样导致明显定位性能下降，不再作为当前主路线。
3. H65、UVT、Fovea/Query-Bridge 位于不同提交并改变不同变量，不能用跨版本性能差直接归因某个组件。
4. 历史 65.xx、66.xx 结果不能冒充共享官方 dense AdaTAD；`65.696` 不能冒充严格采样对照。
5. 已经完成的 dense、uniform、random 和固定 K 归因对照不应在没有新科学问题时机械重复。

## 6. 数据、训练与评估资源

- THUMOS14 规范视频入口为 `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`：411 个有效 MP4 软链接，training 200、validation 211，0 个断链。
- 注释为 `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json`，类别映射为 `/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt`。
- VideoMAE-S 预训练权重为 `/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`。
- 正式实验使用官方数据划分和评估器；validation/test 标注不得参与推理期选择。
- 完整训练至少每 5 个 epoch 保存一次可恢复 `.pth`；恢复状态包含模型、指数移动平均模型、优化器、学习率调度器、混合精度缩放器、epoch/update 和随机状态。最终模型或最终指数移动平均模型的选择规则必须预先固定。
- 共享官方 AdaTAD 结果只读引用，不得为 DUCA 重复训练相同官方模型。

## 7. 当前未知与需要科学负责人裁决的问题

截至本文日期，没有正在运行的 DUCA 训练。当前仍未知：

1. 完成既有 PJST-D1 配对区间是否仍会改变路线判断，还是当前点估计已足以停止扩大该表示改动；
2. 物理时间表示的损失是否集中于动作时长、边界阈值或特定采样间隔；尚无可支持因果归因的分层统计；
3. 是否存在比 PJST-D1 更忠实于 H65、同时不引入新的选择器或训练混杂的表示改动；
4. 在进入动态预算前，固定 K 的语义间接选帧还缺哪一个最有决策价值的证据；
5. 长期动态预算如何在真实减少 VideoMAE 工作量的同时保持公平的检测训练与评估；目前没有实验证据。

下一次 Pro 同步必须先核对上述代码身份和结果边界，再独立决定当前论文问题是否成立、是否应继续当前表示子问题，并下达一个能最短改变论文判断的任务。候选历史不是穷尽列表；Pro 可以拒绝当前问题表述并提出更好的单一路径，但必须说明它如何吸收现有正负证据、避免重复实验，并给出明确的可证伪预测。

## 8. 主要证据入口

- `PAPER_PROGRESS.md`：对外论文缩略报告；
- `research-wiki/index.md`：研究记忆入口；
- `research-wiki/decision_history.md`：路线选择与否定原因；
- `research-wiki/log.md`：实验流水与原始结果索引；
- `research-wiki/anti_repetition.md`：不得机械重复的实验和误报；
- `research-wiki/duca_model_version_registry.md`：历史模型身份；
- `research-wiki/DUCA_WIKI_MEMORY_AUDIT-2026-08-17.md`：间接选帧完整历史审计；
- `research-wiki/duca_prebackbone_plugin_and_baseline_recovery_contract.md`：pre-backbone 方法身份和 baseline 边界；
- `docs/methods/2026-08-25-b2ccfcca-duca-pjst-pro-review-absorption.md`：PJST 外部审查的本地核验与吸收。

# DUCA H65：90→60 轮压缩性能下降诊断与后续调参裁决

Nonce: `DUCA-H65-60-COMPRESSION-DIAGNOSIS-v003-20260824`

你是 DUCA 项目的独立 Scientific First-Author Agent 与最严厉的训练动力学审稿人。请直接阅读本文件以及附带的冻结配置/收据，给出一份可执行的中文终稿。不要把问题交回给人类，也不要重新设计 DUCA 模型或改变数据、检测器、采样语义和评估协议。

## 1. 唯一问题

解释为什么历史 H65 从 90 轮（30+60）压缩到 60 轮后出现约 2.66 pp Avg-mAP、3.37 pp mAP@0.7 的下降；区分哪些原因已有证据支持，哪些仍是待检验假设。然后冻结当前两条 30+30 学习率实验结束后的唯一调整决策树。

## 2. 已冻结事实

### 历史 H65 90 轮

- Stage-1：30 epochs / 3000 successful optimizer updates。
- Stage-2：60 epochs / 6000 successful optimizer updates。
- seed=3407，K=384，H65 语义间接非均匀逐帧选择。
- epoch-59 EMA 官方验证：Avg-mAP=65.1257；mAP@0.3/0.4/0.5/0.6/0.7 = 80.2808/75.7109/68.5475/57.7757/43.3137。
- 该 checkpoint 缺 RNG/DataLoader 状态，因此不是完整恢复合同，但终态评估本身有效。

### 已失败的 60 轮压缩

- Stage-1：20 epochs / 2000 updates。
- Stage-2：40 epochs / 4000 updates。
- 其余数据、seed、K、检测器和 evaluator 保持同一实验族。
- epoch-39 EMA：Avg-mAP=62.4648；mAP@0.3/0.4/0.5/0.6/0.7 = 78.0914/73.4479/65.0772/55.7639/39.9434。
- 相对 H65：Avg-mAP -2.6609 pp；mAP@0.7 -3.3703 pp。
- Stage-1 终点：30-epoch EMA Avg-mAP=59.4231；20-epoch EMA=49.5389，差 -9.8842 pp。同为 epoch-20 时，原日程=50.8707、压缩日程=49.5389，差 -1.3318 pp。

因此旧 20+40 失败不能简单归因于“少 30 轮”或“峰值 LR 太小”。它同时改变了 Stage-1 成熟度、Stage-2 暴露、semantic/policy transition、feedback warmup/decay、full-joint tail、cosine horizon 和 EMA exposure。

### 当前正在运行的 30+30 A/B（尚无终态，不得推断）

两臂复用同一个成熟 Stage-1 epoch-29 EMA；固定模型、K384、数据、seed=3407、损失、检测器、官方 evaluator、参数组与各组基础 LR；Stage-2 均为 30 epochs / 3000 successful updates，每 5 epoch 保存恢复 checkpoint。

1. `AM-RPCH25`：500-update warmup；501–1500 为 1.0×；之后衰减到 0.25× 并保留平坦尾段；累计相对 LR 面积 1999.625。
2. `LongCosine-H6000`：沿 6000-update 历史 horizon 只执行前 3000 updates；第 3000 update 仍为 0.571157×；累计相对 LR 面积约 2366.228，比 AM 高 18.33%。

两臂共有：2000-step semantic/policy transition、1000-step feedback warmup + 1000-step cosine、约 1000-step full-joint tail。二者仍相对历史 90 轮共同缩短 Stage-2 6000→3000、semantic/policy 3000→2000、feedback 2000→1000、joint tail 3000→1000、EMA exposure 6000→3000。

Jobs 1252979/1252980 当前仍 RUNNING；中间 validation 不能用于挑 checkpoint 或宣称终态。当前已经冻结：在它们结束前禁止第三 scheduler、参数组 LR 微调或中途优胜者选择。

### 已冻结的终态门

- “恢复到 H65 邻域”：Avg-mAP ≥64.6257 且 mAP@0.7 ≥42.8137。
- “明确失败”：Avg-mAP <64.1257 或 mAP@0.7 <42.3137。
- 介于两者之间为灰区，不得包装成无损压缩。

## 3. 不允许改变的科学身份

- H65 模型结构、间接语义选帧、选中 RGB、K=384、VideoMAE-S/Adapter/ActionFormer、loss、NMS、split、官方 evaluator。
- fixed-K 本轮只做训练日程归因；不引入 dynamic K、TrueTime、Query-Bridge、Fovea 或新 selector。
- 不重复 dense/uniform/random 基线，不以中间 checkpoint 选最优。
- 不把单 seed 的调度结果升级为论文级效率或稳定性结论。

## 4. 你必须给出的裁决

请输出唯一 `CONTINUE / REVISE / STOP_60_EPOCH_COMPRESSION`，并逐项回答：

1. 对 20+40 大幅下降给出按证据强弱排序的因果诊断：Stage-1 handoff、Stage-2 总更新数、LR 面积/曲线、课程时钟、反馈时钟、联合训练尾段、EMA 暴露、欠拟合/优化不稳定，各自哪些已被证据支持，哪些只是推测。
2. 解释为什么“保持基础 LR 不变、保留 30-epoch Stage-1、调整 Stage-2 decay 而非整体抬高峰值 LR、保留非零尾段”是或不是正确策略。
3. 对当前 A/B 的结果分别冻结动作：
   - 若某臂进入恢复邻域；
   - 若两臂都在灰区；
   - 若两臂都明确失败且终端仍在上升；
   - 若两臂都失败且已平台或下降。
4. 如确需下一项实验，只能选择一个最便宜、最能区分机制的正式 H65-compatible 训练。给出精确 Stage-1/Stage-2 epochs、成功 update 数、scheduler 公式或关键拐点、semantic/feedback/joint-tail 时钟、EMA、基础 LR 是否变化、唯一停止规则。不得列无界网格。
5. 明确是否存在“60 epoch 在保持其他结构不变时原则上无法等价 90 epoch”的证据；若没有，请说明目前能支持的最窄结论。
6. 给出终态数据需要如何分析：训练/验证曲线、梯度与 LR、selector/semantic losses、EMA-vs-final、high-IoU，哪些是诊断，哪些可决定下一步；禁止 post-hoc cherry-pick。
7. 输出 `next_owner / next_action / dependency / expected_return_at`。

请以严肃但清晰的科研语言书写，首先给一句裁决，再给因果诊断、结果分支和唯一后续动作。不要声称 Jobs 1252979/1252980 已产生终态 mAP。

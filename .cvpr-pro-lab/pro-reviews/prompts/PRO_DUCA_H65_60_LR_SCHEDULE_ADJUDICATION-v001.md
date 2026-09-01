# DUCA H65：90→60 epoch 压缩后性能下降的训练动力学裁决

Nonce: `DUCA-H65-60-LR-SCHEDULE-v001-20260824`

你是 DUCA 项目的独立 Scientific First-Author Agent 和最严厉的优化/实验审稿人。请基于下列冻结代码、配置与真实终态结果，分析为什么 H65 从历史 30+60 epoch 压缩为 20+40 epoch 后性能下降，并给出唯一、最小、可直接实现的后续训练方案。不要把路线选择交回人类或 Codex；必须返回一个 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`。

## 项目与代码身份

- Exact ChatGPT Project: `DUCA`, `g-p-6a796fef9a00819194024cf1de3bd697`
- GitHub: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- H65 current branch: `codex/duca-h65-60-curriculum-20260823`
- Clean HEAD: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- 60-epoch Stage-2 training revision: `87ff0883651a631d48468ab4f9d6392f587c15e4`
- Historical H65 evidence anchor: `42dba3f90b37243e7965d18b6707e88e81bf7109`

本次附加四个精确配置文件。模型代码没有在 90→60 的归因实验中改变。

## 冻结模型合同

H65 是低成本 ASFormer 语义侦察器学习动作性与边界/状态转移证据，再由确定性 sampling-rate/density transport 间接形成非均匀逐帧位置；不是小模型直接预测 frame index。固定 `K=384`，按原始时间排序选中的 RGB 帧，进入同一 VideoMAE-S、Adapter、ActionFormer、loss、NMS、THUMOS14 training/validation split 与官方 evaluator。基础 optimizer 仍为 AdamW，base LR `1e-4`、weight decay `0.05`、backbone frozen、adapter LR `2e-4`。Stage-1 的 selector 参数组 LR 为 `5e-5/1e-4/5e-5`；Stage-2 为 `1e-5/2e-5/5e-5`。这些基础 LR、最终损失权重、模型结构、输入方式、冻结策略均未因压缩而改变。

## 两个终态结果

1. 历史日程：Stage-1 uniform K384 训练 30 epoch / 3000 successful updates；Stage-2 learned/joint 训练 60 epoch / 6000 successful updates。终态 epoch-59 EMA：Avg-mAP `65.1257`，mAP@0.7 `43.3137`。
2. 压缩日程：Stage-1 20 epoch / 2000 updates；Stage-2 40 epoch / 4000 updates，其中 20 epoch transition + 20 epoch joint。终态 epoch-39 EMA：Avg-mAP `62.4648`，mAP@0.7 `39.9434`。
3. 差值：Avg-mAP `-2.6609 pp`，mAP@0.7 `-3.3703 pp`。这是同一模型结构下的课程/优化负结果，不是 H65 科学机制被否定。
4. Stage-1 本身也显示成熟度差异：30-epoch endpoint Avg-mAP `59.4231`，20-epoch endpoint `49.5389`。请判断这是否主要说明 uniform detector/scout warmup 尚未成熟，还是存在更具体的 LR/curriculum 交互。

## 实际改变的训练动力学

- Stage-1 scheduler 从 warmup 3 / cosine horizon 30 改为 warmup 2 / horizon 20。
- Stage-2 从 warmup 5 / cosine horizon 60 改为 warmup 3 / horizon 40。
- Stage-2 所有 actionness、transition、boundary、policy-alpha、ASFormer-adapt 的 cosine transition 从 3000 updates 压到 2000。
- detector-gradient 与 detector-contribution 从历史 warmup 1000 + transition 2000 改为 667 + 1333。
- 因而压缩并非只少训练 30 epoch：它同时让 LR 更早衰减、语义监督更快减弱、learned sampling/feedback 更早增大，并缩短 full-joint 后段。
- 仍在运行的 First-Mixing SingleClock TrueTime 实验是独立表示实验，不得用它解释上述 90→60 的性能下降，也不得等待它才能给出本训练裁决。

## 用户冻结的优化原则

1. 保持所有参数组基础 LR 不变；不允许统一乘倍数。
2. 保留原始 30-epoch uniform Stage-1，不再压缩。
3. 若总训练预算必须为 60 epoch，则只调整 Stage-2 的 decay 速度/时间尺度，不提高峰值 LR。
4. 联合训练后段必须保留足够的非零 LR，可通过延长 cosine horizon 或短暂平坦尾段实现。
5. 只有明确证据表明某一参数组欠拟合且梯度稳定，才允许后续单独小幅调整该组 LR。
6. 不改变 H65 模型、selected RGB、K、损失终值、检测器、数据、seed=3407、official evaluator、final/final-EMA 规则。
7. 不重复 dense/uniform/random 等已经多次完成的通用对照；本轮只做 H65 schedule attribution。

## 请重点攻击并裁决

1. 对 `-2.6609 pp` 的原因给出按证据排序的机制解释：Stage-1 不成熟、Stage-2 总更新不足、cosine 过早接近零、语义监督退火过快、policy/feedback 开启过快、joint tail 不足，各自应该如何用已有曲线或无需训练的诊断区分？不要凭直觉定量归因。
2. “把 90 epoch 挤成 60 epoch”是否在固定每 epoch 100 updates 时，本质上减少了 1/3 optimizer exposure，因此无法仅靠 LR schedule 无损恢复？请明确可恢复上限与不可恢复部分。
3. 在保留 30-epoch Stage-1 的前提下，冻结一个 30-epoch Stage-2 方案。必须写出逐 update 的：warmup、cosine/hold 区间、policy alpha、actionness、transition、boundary、detector gradient/contribution、ASFormer adapt 时间轴，以及 terminal LR fraction。不得只说“调慢一点”。
4. 严厉比较下列思路并选择唯一主方案，必要时可提出一个更好的单一方案：
   - `LongCosine-90`：Stage-2 只跑 3000 updates，但沿用历史 6000-update LR horizon，使终点仍保留非零 LR；
   - `RelativeCosine+Hold`：在 3000 updates 内完成较慢的 transition，并给 full-joint 留一个短的非零 LR 平坦尾段；
   - 其他不改峰值 LR、可严格说明的 schedule。
5. 冻结最多 2–3 个正式完整训练臂。优先复用已经完成的 30-epoch Stage-1 epoch-29 EMA，禁止重复 Stage-1；每个臂必须给出唯一变量、预计信息增益、成功/失败阈值和何时停止继续调参。
6. 判断是否需要 matched `30+60` 只读 anchor 之外的 OFF 重训；若不需要，明确禁止重复。
7. 冻结训练正确性检查：实际 successful updates、每参数组 LR 曲线、各 curriculum 权重曲线、梯度范数/非有限更新、checkpoint 恢复、每 5 epoch 可恢复 `.pth`、final/final-EMA 不按中间验证挑选。
8. 给出结果解释边界：单 seed 达到什么程度只能说明 schedule recovery，何时才值得增加 seeds；不得把训练日程恢复宣称为新科学贡献。

## 必须返回的终稿

- 唯一裁决与一句话原因；
- 对性能下降的证据排序诊断；
- 精确 30-epoch Stage-2 schedule 表/公式；
- 最小配置改动（允许文件、字段、禁止改动）；
- 最多 2–3 臂实验矩阵、seed、6000? 或 3000 successful-update 解释、checkpoint/cost/stop rule；
- Critic 与 Evaluator PRE_RUN 必须核验的事实；
- 失败后下一步是增加训练 exposure、单独 LR 微调，还是承认 60-epoch 预算不可无损压缩；
- `next_owner / next_action / dependency / expected_return_at`。

请用严肃、直接、外部评审可理解的中文作答。不要引入新 selector、Query、Bridge、dynamic K、连续 cliplet 或新的论文故事。

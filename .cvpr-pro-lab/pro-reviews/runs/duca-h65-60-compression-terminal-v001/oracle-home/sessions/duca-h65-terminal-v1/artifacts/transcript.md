# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697-duca/c/6a8c1c88-5680-83ea-8170-910401f870af

## Prompt

请依据附加材料对 H65 90→60 轮压缩的终态结果作独立训练动力学裁决。Nonce: DUCA-H65-60-COMPRESSION-TERMINAL-v001-20260824

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_H65_60_COMPRESSION_TERMINAL_ADJUDICATION-v001.md
Lines: 1-40
```md
 1 | # DUCA H65 90→60 轮压缩：终态结果裁决与训练动力学解释
 2 | 
 3 | Nonce: `DUCA-H65-60-COMPRESSION-TERMINAL-v001-20260824`
 4 | 
 5 | 你是 DUCA 项目的独立 Scientific First-Author Agent、训练动力学专家和最严厉的审稿人。请直接阅读附加的历史比较收据、上一轮冻结裁决、两条真实配置与本次终态结果，给出唯一、可执行的中文终稿。不要把选择交还给人类，不要假定未提供的诊断数据存在，也不要为了“继续实验”而发明无界调参网格。
 6 | 
 7 | ## 唯一任务
 8 | 
 9 | 解释为什么 H65 从历史 `30+60` 压缩后显著下降，并用刚完成的两个 `30+30` LR 归因实验判断：
10 | 
11 | 1. 性能下降的主因是否仍可归为学习率衰减过快，还是证据已经更支持 Stage-2 更新数、课程/反馈时钟、full-joint 暴露或其他训练动力学不足；
12 | 2. `LongCosine-H6000` 比 `AM-RPCH25` 多保留 LR 面积，却只获得有限 Avg-mAP 改善且高 IoU 不占优，这一模式能排除什么、不能排除什么；
13 | 3. 是否应严格执行上一轮预注册的 `STOP_60_EPOCH_COMPRESSION`，保留历史 `30+60` 作为 H65 训练参考；
14 | 4. 后续调整方法应是什么。这里必须区分：
15 |    - 当前材料已经支持、无需再训练即可做的诊断；
16 |    - 只有获得新证据后才值得开启的新实验；
17 |    - 明确不应继续的 scheduler/LR 搜索。
18 | 
19 | ## 不可改变的事实边界
20 | 
21 | - H65 的模型结构、语义间接非均匀逐帧选择、K=384、选中 RGB、VideoMAE-S/Adapter/ActionFormer、loss、NMS、split 和官方 evaluator 在本次归因中不变。
22 | - `20+40` 压缩同时改变了 Stage-1 成熟度、Stage-2 更新数、课程/反馈时钟、full-joint tail、cosine horizon 与 EMA 暴露，不能把其下降全部归因于一个因素。
23 | - 两个新 arm 均复用成熟 30-epoch Stage-1 epoch-29 EMA，仅改变 Stage-2 LR 曲线；每个 arm 只有 3000 Stage-2 successful updates，semantic/policy transition 为 2000 steps，feedback 为 1000 warmup + 1000 transition，full-joint tail 约 1000 updates。
24 | - 两个新 arm 的 terminal EMA 都明确失败，且 epoch 29 Avg-mAP 均比 epoch 24 略低；因此不满足上一轮唯一 `+1000` 延长所需的 rising-tail 条件。
25 | - 不允许使用 intermediate checkpoint 挑最好，不允许把 terminal online 替代 EMA，不允许重新解释已冻结阈值。
26 | 
27 | ## 你必须给出的终稿
28 | 
29 | 首先给唯一裁决：`STOP_60_EPOCH_COMPRESSION / REVISE_WITH_ONE_NEW_MECHANISM_TEST / CONTINUE` 三选一。若不选择 STOP，必须明确指出哪条新证据足以推翻上一轮冻结的 flat/falling stop branch；仅凭“也许更多训练有用”不成立。
30 | 
31 | 随后逐项给出：
32 | 
33 | 1. 按证据强弱排序的因果解释，严禁把相互耦合因素机械分配为若干百分点；
34 | 2. 对两个 arm 的 terminal 与 epoch19/24/29 曲线作联合解释，特别分析 Avg-mAP 与 mAP@0.7 的不同走势；
35 | 3. 说明为何只调整 LR 曲线未恢复 H65，以及这是否意味着“60 轮原则上不可能”等价 90 轮；
36 | 4. 给出最小后续动作：优先列出无需训练的日志/梯度/selector/online-vs-EMA/更新轨迹诊断。若仍建议一个新训练，必须是单一、H65-compatible、能区分机制且不构成第三个 scheduler 搜索的实验，并给出严格触发条件、唯一终点和停止规则；
37 | 5. 明确禁止项：峰值 LR 全局抬高、参数组无依据微调、第三个 30+30 scheduler、20+40 重跑、intermediate-best 挽救、模型/selector/dynamic-K/TrueTime 混入本归因问题；
38 | 6. 输出 `next_owner / next_action / dependency / expected_return_at`。
39 | 
40 | 请用论文级、严肃但易懂的中文写作。区分“观察事实”“合理推断”“尚未测量”。不要声称单 seed 结果具有统计稳定性或论文级训练效率结论。
```

### File: .cvpr-pro-lab/receipts/H65_LR_SCHEDULE_TERMINAL_RESULTS-v001.md
Lines: 1-42
```md
 1 | # H65 30+30 LR-schedule terminal results — evidence packet
 2 | 
 3 | - evidence_class: `FULL_TRAINING / OFFICIAL_VALIDATION / SINGLE_SEED / TERMINAL_EMA / SCHEDULE_ATTRIBUTION_ONLY`
 4 | - frozen_revision: `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f`
 5 | - seed: `3407`
 6 | - shared_stage1: `30 epochs / 3000 successful updates / epoch-29 state_dict_ema`
 7 | - stage2_per_arm: `30 epochs / 3000 successful updates`
 8 | - run_root: `/data/run01/sczc063/yuzibo/duca_h65_lr60_formal_ae3642a1_20260824`
 9 | - official_validation: canonical THUMOS14 validation, 211 videos, 3325 ground-truth instances
10 | - primary_checkpoint: Stage-2 `epoch_29.pth / state_dict_ema`
11 | - checkpoint_interval: every 5 epochs
12 | 
13 | ## Arm A — AM-RPCH25
14 | 
15 | - Slurm job: `1252979`
16 | - schedule: 500-update warmup, 1000-update plateau, 1000-update decay, 500-update hold, terminal factor `0.25`
17 | - epoch 19 EMA: Avg-mAP `62.70`, mAP@0.7 `40.34`
18 | - epoch 24 EMA: Avg-mAP `63.35`, mAP@0.7 `40.98`
19 | - epoch 29 terminal EMA: Avg-mAP `63.22`; mAP@0.3/0.4/0.5/0.6/0.7 = `79.29/73.86/66.09/55.59/41.25`
20 | - terminal status: training and official validation completed (`Training Over`)
21 | 
22 | ## Arm B — LongCosine-H6000
23 | 
24 | - Slurm job: `1252980`
25 | - schedule: historical 6000-update cosine horizon truncated after its first 3000 updates; terminal factor `0.571157`
26 | - epoch 19 EMA: Avg-mAP `62.62`, mAP@0.7 `39.97`
27 | - epoch 24 EMA: Avg-mAP `63.58`, mAP@0.7 `40.95`
28 | - epoch 29 terminal EMA: Avg-mAP `63.56`; mAP@0.3/0.4/0.5/0.6/0.7 = `79.66/74.55/66.58/56.01/41.01`
29 | - terminal status: training and official validation completed (`Training Over`)
30 | 
31 | ## Frozen comparison references
32 | 
33 | - Historical 30+60 reproduction terminal EMA: Avg-mAP `65.1257`, mAP@0.7 `43.3137`.
34 | - Previous 20+40 compression terminal EMA: Avg-mAP `62.4648`, mAP@0.7 `39.9434`.
35 | - Recovery neighborhood: Avg-mAP `>=64.6257` and mAP@0.7 `>=42.8137`.
36 | - Clear failure: Avg-mAP `<64.1257` or mAP@0.7 `<42.3137`.
37 | 
38 | Both new 30+30 arms are clear failures. Neither meets the preregistered rising-tail condition because epoch 29 Avg-mAP is below epoch 24 (`63.22<63.35` and `63.56<63.58`), even though mAP@0.7 rises slightly. Under the prior frozen decision tree this selects `STOP_60_EPOCH_COMPRESSION` and does not authorize a third scheduler or a `+1000` continuation.
39 | 
40 | ## Evidence boundary
41 | 
42 | These results isolate two Stage-2 LR-tail shapes after a mature Stage-1 handoff. They do not isolate curriculum-clock, feedback-clock, full-joint-exposure, or EMA-lag effects. They do not falsify H65 semantic indirect frame selection, establish multi-seed stability, or support a training-efficiency claim.
```

### File: .cvpr-pro-lab/receipts/H65_CURRICULUM_TERMINAL_COMPARISON-v001.md
Lines: 1-38
```md
 1 | # H65 curriculum terminal comparison — MATERIAL RESULT
 2 | 
 3 | - completed_at: `2026-08-24T06:15:13+08:00`
 4 | - evidence_class: `FULL_TRAINING / OFFICIAL_VALIDATION / SINGLE_SEED / EMA_TERMINAL`
 5 | - dataset: canonical THUMOS14 validation, `211` videos
 6 | - evaluator_sha256: `e855e70d41d087d039a90ecdb8f3cc3efece209130417320edf35062b8503fd4`
 7 | - seed: `3407`
 8 | - independent_evaluator: `evaluate_h65_curriculum_terminal`, terminal read-only verification accepted
 9 | 
10 | ## Original 30+60 schedule
11 | 
12 | - job: `1251782`, `COMPLETED 0:0`
13 | - source revision: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
14 | - run root: `/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823`
15 | - frozen result: `gpu1_id0/intermediate_validation/epoch_060_ema.json`
16 | - checkpoint: `gpu1_id0/checkpoint/epoch_59.pth`, `state_dict_ema`
17 | - Avg-mAP / mAP@0.3/0.4/0.5/0.6/0.7: `65.1257 / 80.2808 / 75.7109 / 68.5475 / 57.7757 / 43.3137`
18 | 
19 | ## Compressed 20+40 schedule
20 | 
21 | - job: `1251622`, `COMPLETED 0:0`
22 | - source revision: `87ff0883651a631d48468ab4f9d6392f587c15e4`
23 | - run root: `/data/run01/sczc063/yuzibo/duca_h65_60_stage2_transition20_joint20_87ff0883_20260823`
24 | - frozen result: `gpu1_id0/intermediate_validation/epoch_040_ema.json`
25 | - checkpoint: `gpu1_id0/checkpoint/epoch_39.pth`, `state_dict_ema`
26 | - Avg-mAP / mAP@0.3/0.4/0.5/0.6/0.7: `62.4648 / 78.0914 / 73.4479 / 65.0772 / 55.7639 / 39.9434`
27 | 
28 | ## Frozen comparison
29 | 
30 | The compressed schedule changes training duration/curriculum but not the H65 selector mechanism. Relative to the original schedule, it changes Avg-mAP / mAP@0.3/0.4/0.5/0.6/0.7 by `-2.6609 / -2.1894 / -2.2630 / -3.4703 / -2.0117 / -3.3703` percentage points. Therefore the 20+40 compression does not preserve the 30+60 H65 endpoint under this seed. This is negative evidence about schedule compression, not a falsification of semantic indirect selection or of the separately trained SingleClock representation gate.
31 | 
32 | The independent Evaluator reproduced the two Slurm terminal states, config seeds, checkpoint epochs/update counts, video counts, evaluator identity, metrics and all reported deltas from the raw artifacts. Both immutable terminal checkpoints lack `rng_state` and `data_loader_state`. Their frozen EMA inference is usable for this diagnostic comparison, but the missing recovery fields prevent paper-level replication admission and must not be fabricated.
33 | 
34 | - next_owner: `DUCA Coordinator`
35 | - next_action: preserve this negative schedule result; continue the already frozen SingleClock/legacy-bootstrap evidence chain without changing the model
36 | - dependency: terminal Jobs `1252482` and `1252515`
37 | - expected_return_at: their formal terminal events
38 | - single_recovery: `none`
```

### File: .cvpr-pro-lab/pro-reviews/runs/duca-h65-60-compression-diagnosis-v003/PRO_DUCA_H65_60_COMPRESSION_DIAGNOSIS-v003.md
Lines: 1-479
````md
  1 | # 裁决：`CONTINUE`
  2 | 
  3 | **Nonce：`DUCA-H65-60-COMPRESSION-DIAGNOSIS-v003-20260824`**
  4 | 
  5 | 这里的 `CONTINUE` 仅表示：**让已经冻结的 `AM-RPCH25` 与 `LongCosine-H6000` 两个 30+30 作业运行到各自唯一的 terminal EMA，并按预先定义的门自动裁决。**它不表示 60 轮压缩已经成功，也不授权第三个 60 轮 scheduler、参数组 LR 微调、峰值 LR 上调或中间 checkpoint 选择。Jobs `1252979/1252980` 仍无已知终态，本报告不推断其 mAP。
  6 | 
  7 | ---
  8 | 
  9 | ## 一、核心科学判断
 10 | 
 11 | 历史 20+40 的下降不是一个可被“学习率太小”概括的单因素现象，而是一个**耦合压缩失败**：
 12 | 
 13 | [
 14 | \text{weaker Stage-1 handoff}
 15 | \times
 16 | \text{fewer Stage-2 updates}
 17 | \times
 18 | \text{shorter curriculum/feedback clocks}
 19 | \times
 20 | \text{shorter full-joint tail}
 21 | \times
 22 | \text{earlier LR decay}
 23 | \times
 24 | \text{shorter EMA exposure}.
 25 | ]
 26 | 
 27 | 现有证据中，**Stage‑1 交接成熟度不足是最强的直接信号**；Stage‑2 更新与 full-joint 暴露不足是下一高优先级解释；LR 曲线/面积是当前 A/B 正在直接检验的因素；课程、反馈和 EMA 目前仍是未隔离假设。
 28 | 
 29 | 严禁把各项差异机械地加成“各贡献多少 pp”。这些因素在同一训练轨迹中相互作用，现有设计不是因子实验。
 30 | 
 31 | ---
 32 | 
 33 | # 二、20+40 下降的因果诊断排序
 34 | 
 35 | ## 1. Stage‑1 handoff 不成熟——**最强支持**
 36 | 
 37 | 历史 30 轮 Stage‑1 terminal EMA 为 `59.4231`，压缩 20 轮仅为 `49.5389`，交接点相差 `−9.8842 pp`。而在同为第 20 轮时，原 30 轮日程为 `50.8707`，压缩日程为 `49.5389`，只差 `−1.3318 pp`。〔任务文件 L21–30〕
 38 | 
 39 | 这支持两个结论：
 40 | 
 41 | 1. 压缩日程自身的较短 cosine horizon 已经在相同更新数下造成约 `1.33 pp` 差异；
 42 | 2. 更大的 `9.88 pp` terminal handoff 差距主要在于压缩模型没有经历后续 10 轮成熟过程。
 43 | 
 44 | 因此，20+40 的 Stage‑2 是从明显更弱的 detector/coarse-semantic 状态启动的。这会迫使联合阶段同时承担：
 45 | 
 46 | * 补 detector 基础能力；
 47 | * 学习非均匀选择；
 48 | * 开启 detector-to-selector feedback；
 49 | * 完成 ASFormer/selector 适配。
 50 | 
 51 | 但不能据此宣称最终 `−2.6609 pp` 全由 Stage‑1 造成。Stage‑2 明显补回了部分 handoff 差距，故 Stage‑1 是强因果候选而不是完整解释。
 52 | 
 53 | ## 2. Stage‑2 successful updates 减少——**中强支持，尚未隔离**
 54 | 
 55 | 历史 Stage‑2 有 `6000` 次成功更新，旧压缩只有 `4000` 次；当前 30+30 A/B 更进一步只有 `3000` 次。〔Stage‑2 历史配置 L17–29；20+40 配置 L73–94；任务文件 L34–39〕
 56 | 
 57 | “更新数确实减少”是事实；“这造成了多少性能损失”尚未单独测得。但它是非常有力的解释，因为减少的不只是一般训练时间，而是 learned sampling、detector feedback 和最终 detector-led joint optimization 共同生效后的更新数。
 58 | 
 59 | 如果当前 A/B 在 terminal 前仍持续上升，则这一解释将明显增强；若已经平台或下降，则单纯增加更新的解释减弱。
 60 | 
 61 | ## 3. full-joint tail 缩短——**重要且可操作，但未直接证明**
 62 | 
 63 | 在历史 30+60 中：
 64 | 
 65 | * semantic/policy 与 feedback 在约 update 3000 完成；
 66 | * 随后还有约 `3000` 次完整联合更新。
 67 | 
 68 | 旧 20+40 中：
 69 | 
 70 | -相关时钟在 update 2000 左右完成；
 71 | -剩余约 `2000` 次 full-joint tail。
 72 | 
 73 | 当前 30+30 A/B 中：
 74 | 
 75 | -时钟同样在 update 2000 左右完成；
 76 | -只剩约 `1000` 次 full-joint tail。〔任务文件 L36–39〕
 77 | 
 78 | 因此，当前 A/B 即使修复了 Stage‑1，也仍把完整联合阶段压缩到历史的三分之一。该因素与 Stage‑2 总更新数高度相关，但科学含义更具体：问题可能不是“总共少训练”，而是**selector 已 fully active 后，detector 与采样策略共同沉降的时间不足**。
 79 | 
 80 | ## 4. LR horizon、面积与尾部非零性——**配置事实明确，性能因果正在由 A/B 检验**
 81 | 
 82 | 旧 20+40 把 Stage‑2 cosine horizon 从 60 轮缩成 40 轮；Stage‑1 同轮次比较已经表明，较短 horizon 会造成可观差异。因此，“过早进入低 LR 区域”有直接旁证，但还不是 Stage‑2 的独立结果。
 83 | 
 84 | 当前 A/B 在相同 Stage‑1、相同 3000 updates、相同课程时钟下，只改变 LR 曲线：
 85 | 
 86 | * `AM-RPCH25`：累计相对 LR 面积 `1999.625`；
 87 | * `LongCosine-H6000`：面积约 `2366.228`，高 `18.33%`，update 3000 仍为 `0.571157×`。〔任务文件 L34–39；AM 配置 L72–84；LongCosine 配置 L11–17〕
 88 | 
 89 | 所以：
 90 | 
 91 | * 若 LongCosine 单独恢复，支持“过早衰减/面积不足”；
 92 | * 若 AM 单独恢复，说明较高总面积并非必要，平稳的非零 floor/tail 更重要；
 93 | * 若两者均恢复，不能宣称某一 LR 形状是唯一原因；
 94 | * 若两者均失败，只能说明**在 3000 Stage‑2 updates 下，调整 LR 曲线不足以恢复 H65**。
 95 | 
 96 | 累计 LR 面积不是有效参数位移的等价物；后者还依赖梯度尺度、方向和 clipping。
 97 | 
 98 | ## 5. semantic/policy curriculum clock——**已发生改变，因果未隔离**
 99 | 
100 | 历史 transition 为 `3000` steps，旧压缩和当前 A/B 为 `2000` steps。actionness、transition、boundary 权重与 policy alpha 因此更快完成迁移。〔历史 Stage‑2 配置 L58–105；20+40 配置 L17–69；AM 配置 L18–68〕
101 | 
102 | 它可能造成：
103 | 
104 | * policy 在 detector 尚未适应时过早主导输入；
105 | * coarse supervision 过早减弱；
106 | * detector 所见输入分布变化过快。
107 | 
108 | 但没有 selector loss、梯度或策略熵证据时，这仍是机制假设。当前 A/B 两臂使用相同 2000-step clock，因此不能通过两臂差异隔离该因素。
109 | 
110 | ## 6. feedback clock——**已发生改变，因果未隔离**
111 | 
112 | 历史 feedback 为 `1000 warmup + 2000 transition`；旧 20+40 为 `667+1333`；当前 A/B 为 `1000+1000`。〔历史配置 L88–99；20+40 配置 L49–62；AM 配置 L50–61〕
113 | 
114 | 旧压缩是按总长度近似比例加速；当前 A/B 则保留 1000-step warmup、压缩 transition。两者都改变了 detector gradient 和 detector contribution 进入 selector 的时间结构。
115 | 
116 | 目前不能判断它是“过早反馈”“反馈变化过陡”还是无关因素。必须依赖梯度范数、loss 分量和 selector 行为分析，而不能凭最终 mAP 讲故事。
117 | 
118 | ## 7. EMA exposure/滞后——**弱假设**
119 | 
120 | 历史 Stage‑2 EMA 经历 6000 次联合更新，旧压缩 4000 次，当前 A/B 3000 次。若模型在后期仍快速移动，terminal EMA 可能落后于 online model。
121 | 
122 | 但当前材料没有给出：
123 | 
124 | * EMA decay 的有效时间常数；
125 | * online-final 与 EMA-final 的差异；
126 | * 后期权重变化率。
127 | 
128 | 因此 EMA 目前只能列为待检验假设。即使 online-final 更好，冻结 primary 仍是 terminal EMA，不能 post-hoc 改用 online checkpoint。
129 | 
130 | ## 8. 欠拟合还是优化不稳定——**当前无法判定；欠拟合先验更强**
131 | 
132 | 目前只有 terminal 指标，没有足够训练曲线、梯度或 entropy 收据。
133 | 
134 | * 若 loss 持续下降、梯度有限、validation terminal 仍上升、selector 未塌缩，支持欠拟合/暴露不足；
135 | * 若出现梯度尖峰、loss 振荡、selector 熵骤降、online–EMA 强烈分离或高 IoU 反复退化，才支持不稳定。
136 | 
137 | 不能仅凭 20+40 性能低就称其“训练不稳定”。
138 | 
139 | ## 9. 实现差异——**当前证据不支持其为主因**
140 | 
141 | 独立终态收据复现了 seed、checkpoint epoch、更新数、211 个 validation videos、evaluator 身份及所有 delta，并把变化界定为训练时长/课程变化，而非 H65 selector 机制变化。〔终态收据 L28–32〕
142 | 
143 | 两个历史 terminal checkpoint 缺 `rng_state` 与 `data_loader_state`，这阻止完整 resume/论文级复现，但不否定其冻结 terminal EMA 比较。没有证据表明 `−2.6609 pp` 是 evaluator 或 selector 实现漂移造成的。
144 | 
145 | ---
146 | 
147 | # 三、为什么当前策略是正确的
148 | 
149 | ## 1. 保留 30 轮 Stage‑1：正确且必要
150 | 
151 | 这是对最强 confound 的直接修复。继续从 20 轮 Stage‑1 调 LR，只会把“较弱初始化”和“Stage‑2 scheduler”继续混在一起。
152 | 
153 | ## 2. 保持基础 LR 不变：正确
154 | 
155 | 历史 30+60 已经证明这组基础 LR 至少能够到达 H65 邻域。当前没有任何证据显示峰值 LR 太小。
156 | 
157 | 整体抬高峰值 LR 会同时改变：
158 | 
159 | * detector/backbone/adapter 的绝对更新尺度；
160 | * coarse trunk/action head/transition scorer 的尺度；
161 | * feedback 开启时的耦合强度；
162 | * AMP、clipping 和 EMA 行为。
163 | 
164 | 这不是对压缩机制的最小修复，而是开启新的优化网格。
165 | 
166 | ## 3. 先改变 decay/horizon，而非峰值：正确
167 | 
168 | 旧压缩同时缩短 horizon 和训练暴露。当前 A/B 保持峰值与参数组比例，直接测试：
169 | 
170 | * 更大的累计 LR 面积是否必要；
171 | * 非零 tail 是否能防止联合阶段过早冻结。
172 | 
173 | `RelativeSuccessfulUpdateLR` 对各参数组乘同一个相对因子，能够保持原有组间 LR 比例，优于使用一个绝对 `eta_min` 使不同 base-LR 组发生比例扭曲。
174 | 
175 | ## 4. 保留非零尾段：正确，但不能替代更新数
176 | 
177 | AM 的 `0.25×` hold 与 LongCosine 在 update 3000 的 `0.571157×` 都避免 terminal 前 LR 近零。它们可以检验“旧压缩是否过早失去可学习性”，但不能凭空替代少掉的 3000 次 historical Stage‑2 updates，也不能恢复被压缩的 full-joint/EMA exposure。
178 | 
179 | 因此当前策略是**正确的第一诊断**，不是预先保证成功的解决方案。
180 | 
181 | ---
182 | 
183 | # 四、当前 A/B 的冻结终态决策树
184 | 
185 | 令每臂 terminal EMA 的：
186 | 
187 | [
188 | A_i=\text{Avg-mAP},\qquad H_i=\text{mAP@0.7}.
189 | ]
190 | 
191 | 分类严格沿用冻结门：
192 | 
193 | * **恢复邻域**：(A_i\ge64.6257) 且 (H_i\ge42.8137)；
194 | * **明确失败**：(A_i<64.1257) 或 (H_i<42.3137)；
195 | * **灰区**：其余情况。
196 | 
197 | ## 分支 1：至少一臂进入恢复邻域
198 | 
199 | **动作：接受一条 30+30 schedule 作为单 seed 的 H65-60 schedule-feasibility 结果；不再运行第三个 scheduler。**
200 | 
201 | 若两臂均通过，按以下预冻结规则选择：
202 | 
203 | [
204 | S_i=\min(A_i-64.6257,\ H_i-42.8137),
205 | ]
206 | 
207 | 选择 (S_i) 较大者，即对两个门具有更大最小安全余量的 arm。若两者 (S_i) 相差不超过 `0.10 pp`，选择 `LongCosine-H6000`，因为它最接近历史 horizon、非历史拐点更少。
208 | 
209 | 允许结论仅为：
210 | 
211 | > 在 seed 3407、冻结 H65 结构和协议下，该 30+30 schedule 恢复到了预注册 H65 邻域。
212 | 
213 | 不允许升级为：
214 | 
215 | * 60 轮与 90 轮严格等价；
216 | * 多 seed 稳定；
217 | * 训练效率论文结论；
218 | * selector 或 dynamic-K 论文结论。
219 | 
220 | ## 分支 2：没有 arm 通过，但至少一臂处于灰区
221 | 
222 | 包括“两臂均灰区”以及“一臂灰区、一臂明确失败”。
223 | 
224 | **动作：拒绝“60 轮无损压缩”表述；不再做第三个 60 轮 scheduler。仅允许下文唯一的 `+1000 full-joint updates` 延长诊断。**
225 | 
226 | 理由是：灰区已经未通过预注册恢复门。再启动一个全新的 30+30 clock/scheduler 组合，会成为 validation-guided 的第三次 60 轮搜索，而且同时修改 semantic、feedback 和 tail，无法干净隔离原因。
227 | 
228 | ## 分支 3：两臂均明确失败，但最佳臂 terminal 仍在上升
229 | 
230 | “仍在上升”现在冻结为：
231 | 
232 | * 使用 Stage‑2 epoch `19/24/29` 三个预定 EMA validation 点；
233 | * 对 Avg-mAP 和 mAP@0.7 分别作关于 epoch 的三点最小二乘直线；
234 | * 两条斜率均大于 0；
235 | * epoch 29 在两项指标上均不低于 epoch 24。
236 | 
237 | **动作：同样只允许唯一的 `+1000 full-joint updates` 延长诊断。**
238 | 
239 | 这时最诚实的假设是 3000 个 Stage‑2 updates 截断了尚未完成的优化，而不是继续发明一个 60 轮 scheduler。
240 | 
241 | ## 分支 4：两臂均明确失败，且最佳臂已经平台或下降
242 | 
243 | **动作：`STOP_60_EPOCH_COMPRESSION`。**
244 | 
245 | 不再允许：
246 | 
247 | * 第三个 30+30 scheduler；
248 | * 整体抬高 peak LR；
249 | * 单参数组 LR 微调；
250 | * 改 semantic/feedback clock；
251 | * 用 intermediate best checkpoint 挽救；
252 | * 复跑 20+40。
253 | 
254 | 保留 30+60 为当前 H65 训练参考，把本轮结论记为 schedule-compression negative evidence。
255 | 
256 | ---
257 | 
258 | # 五、唯一允许的后续训练：`H65-TAIL-EXPOSURE-EXT1000-v001`
259 | 
260 | 这不是第三个 60 轮方案，而是**在 60 轮未通过后，最便宜地判定“是否只是 Stage‑2/full-joint 暴露不足”**。一旦触发该实验，60-epoch no-loss compression 本身已经被停止；该实验最多建立一个 70-epoch 工程折中。
261 | 
262 | ## 1. Parent arm 的确定
263 | 
264 | 在没有通过 arm 时，定义恢复缺口：
265 | 
266 | [
267 | D_i=\max(64.6257-A_i,\ 42.8137-H_i).
268 | ]
269 | 
270 | * 灰区分支：只在非明确失败 arm 中选 (D_i) 最小者；
271 | * 双失败上升分支：只在满足上述 `RISING` 定义的 arm 中选 (D_i) 最小者；
272 | * 若差值不超过 `0.10 pp`，选择 `LongCosine-H6000`。
273 | 
274 | 这是一项预注册的 parent 选择规则，不允许人工看曲线后改选。
275 | 
276 | ## 2. 精确训练合同
277 | 
278 | | 项目                             | 冻结值                                                                        |
279 | | ------------------------------ | -------------------------------------------------------------------------- |
280 | | Stage‑1                        | 30 epochs / 3000 successful updates；同一 epoch‑29 EMA handoff                |
281 | | Stage‑2                        | 从选定 arm 的完整 update‑3000 checkpoint 继续到 40 epochs / 4000 successful updates |
282 | | 总训练                            | 30+40，即 70 epochs                                                          |
283 | | seed                           | 3407                                                                       |
284 | | 模型/数据/K/检测器/loss/NMS/evaluator | 全部不变                                                                       |
285 | | semantic/policy/asformer clock | 保持 2000 steps，禁止 reset                                                     |
286 | | feedback clock                 | 1000 warmup + 1000 transition，update 2000 完成，禁止 reset                      |
287 | | full-joint tail                | 从原 1000 扩展为 2000 updates                                                   |
288 | | base LR                        | 不变                                                                         |
289 | | checkpoint                     | 每 5 epochs；primary 为 Stage‑2 epoch 39 terminal EMA                         |
290 | | intermediate validation        | learning-curve only，不选 checkpoint                                          |
291 | | online/EMA                     | terminal EMA 唯一主结果；online 只诊断                                              |
292 | 
293 | ## 3. Scheduler 必须是原 arm 的原生连续延伸
294 | 
295 | 不得重启 scheduler，不得在 update 3001 重新 warmup。
296 | 
297 | * 若 parent 为 `AM-RPCH25`：前 3000 update 的因子轨迹必须逐点一致；update `2501–4000` 均保持 `0.25×`。
298 | * 若 parent 为 `LongCosine-H6000`：继续同一个 6000-update historical horizon；前 3000 update 逐点一致，update 3000 为 `0.571157×`，之后沿同一 cosine 继续衰减。Builder 必须导出 update `1–4000` 的权威 factor trace；不接受根据 epoch 重新计算的近似曲线。
299 | 
300 | ## 4. Resume fidelity 是硬门
301 | 
302 | 延长训练必须恢复：
303 | 
304 | * online model；
305 | * EMA shadow；
306 | * optimizer；
307 | * scheduler；
308 | * AMP scaler；
309 | * successful-update counter；
310 | * curriculum schedule step；
311 | * RNG；
312 | * DataLoader/sampler 状态。
313 | 
314 | 不得把 terminal `state_dict_ema` 重新载入模型后重置 optimizer。若完整状态缺失，延长实验 fail-closed，不得改为一次新的 30+40 全量重跑来扩大本轮搜索。
315 | 
316 | ## 5. 唯一停止规则
317 | 
318 | 只读取 update 4000 / Stage‑2 epoch 39 terminal EMA：
319 | 
320 | * 若同时满足 `Avg-mAP ≥64.6257` 和 `mAP@0.7 ≥42.8137`：保留 30+40 作为单 seed 工程候选，并把“Stage‑2/full-joint 暴露不足”提升为主要解释；
321 | * 否则：停止全部 H65 schedule compression 调参，保留 30+60 参考。
322 | 
323 | 无第四个 scheduler，无中间 checkpoint 挽救，无峰值 LR 调整。
324 | 
325 | ---
326 | 
327 | # 六、是否已有证据表明“60 轮原则上无法等价 90 轮”
328 | 
329 | **没有。**
330 | 
331 | 当前材料只能支持最窄的结论：
332 | 
333 | > 在 seed 3407 和冻结 H65 实验族下，具体的 20+40 压缩日程不能保持 30+60 terminal EMA，下降为 `−2.6609 pp Avg-mAP` 和 `−3.3703 pp mAP@0.7`。〔终态收据 L19–32〕
334 | 
335 | 它不能证明不存在另一条 60 轮优化轨迹。
336 | 
337 | 即使当前两条 30+30 均失败，也只能说明：
338 | 
339 | > 已测试的 20+40、AM-RPCH25 30+30 和 LongCosine-H6000 30+30 在该单 seed 下不足以恢复 H65 邻域。
340 | 
341 | 届时停止 60 轮压缩，是基于**实验资源纪律与避免无界 scheduler 搜索**，不是数学上的不可能性证明。
342 | 
343 | ---
344 | 
345 | # 七、终态数据分析与决策权限
346 | 
347 | ## 1. 结果读取前的 admission gate
348 | 
349 | 在读取 mAP 前，独立 Evaluator 必须确认：
350 | 
351 | * 正确的 Stage‑1 epoch‑29 EMA checkpoint 及 hash；
352 | * seed 3407；
353 | * Stage‑2 正好 3000 successful updates；
354 | * resolved config 与 scheduler mode；
355 | * 参数组 base LR 未变；
356 | * scheduler factor/LR trace 与累计面积；
357 | * terminal epoch‑29 online 与 EMA 均存在；
358 | * evaluator hash、211 videos、split 与 NMS 一致；
359 | * 无 intermediate checkpoint selection；
360 | * checkpoint 具备完整 resume state。
361 | 
362 | 任一身份不一致，结果是 `INVALID`，不是科学成功或失败。
363 | 
364 | ## 2. 能决定分支的量
365 | 
366 | 只有以下信息可以决定下一步：
367 | 
368 | 1. terminal EMA Avg-mAP；
369 | 2. terminal EMA mAP@0.7；
370 | 3. 若两臂明确失败，预冻结的 epoch 19/24/29 terminal-slope 分类；
371 | 4. 运行身份、更新数或 evaluator 失败可直接使结果无效。
372 | 
373 | ## 3. 只用于机制诊断、不能选 checkpoint 的量
374 | 
375 | ### LR 与梯度
376 | 
377 | 逐 successful update 记录：
378 | 
379 | * 每个参数组实际 LR；
380 | * detector/backbone/adapter/coarse trunk/action head/transition scorer 的梯度 L2 范数；
381 | * 建议同时报告 (L_2/\sqrt{\text{parameter count}})；
382 | * median、p95、max、zero-gradient rate；
383 | * gradient clipping 次数；
384 | * AMP retry/nonfinite 事件。
385 | 
386 | 用途：
387 | 
388 | * LongCosine 是否真的提供更大有效更新，而不只是名义面积；
389 | * feedback 开启处是否出现梯度冲击；
390 | * terminal 是欠拟合还是不稳定。
391 | 
392 | ### semantic/selector
393 | 
394 | 记录：
395 | 
396 | * actionness loss；
397 | * transition loss；
398 | * transition-boundary loss；
399 | * detector loss；
400 | * policy alpha、detector gradient、detector contribution 与 ASFormer adapt 的实际 schedule 值；
401 | * actionness/transition score entropy；
402 | * top-mass concentration；
403 | * selected-frame 相对 exact-uniform 的平均、p95、最大位移；
404 | * gap 分布；
405 | * selector 与 uniform 的重合率；
406 | * 是否出现位置塌缩或大幅来回摆动。
407 | 
408 | 除非发生非有限值、K384 身份破坏或实现合同失败，这些量不能改变预注册终态门。
409 | 
410 | ### EMA 与 online
411 | 
412 | 必须同时报告 terminal online 与 terminal EMA 的全部阈值 mAP。
413 | 
414 | * online 显著优于 EMA：支持 EMA lag；
415 | * online 与 EMA 均低：反对“仅 EMA 滞后”解释；
416 | * EMA 优于 online：支持后期轨迹仍有噪声。
417 | 
418 | 无论何种情况，primary 仍是 terminal EMA，禁止改用 online 挽救。
419 | 
420 | ### high-IoU
421 | 
422 | mAP@0.7 是恢复门的一部分，不只是附加分析。mAP@0.3–0.6、类别、动作时长和 boundary error 可用于解释，但不能替代 Avg-mAP 与 mAP@0.7 的双门。
423 | 
424 | 历史阈值下降并非严格随 IoU 单调增加——最大损失出现在 0.5 和 0.7 附近——所以不能未经边界误差分析就宣称这是“纯 boundary failure”。
425 | 
426 | ---
427 | 
428 | # 八、独立审查链
429 | 
430 | 若触发唯一延长实验：
431 | 
432 | **Builder** 只能增加一个 extension config/resume entry，冻结 update 1–4000 的 LR trace、第一步 continuation 一致性和完整 resume 字段；不得改模型、loss、selector 或基础 LR。
433 | 
434 | **Independent Critic** 必须验证：
435 | 
436 | * update 1–3000 与 parent arm 逐点相同；
437 | * optimizer/scheduler/EMA/RNG/DataLoader 未 reset；
438 | * curriculum step 没有重新从 0 开始；
439 | * update 3001 起已处于 full-joint 状态；
440 | * 只增加 1000 successful updates；
441 | * terminal epoch 39 EMA 是唯一 checkpoint；
442 | * 无 intermediate metric-driven 选择。
443 | 
444 | **Evaluator PRE_RUN** 必须在训练前封存 revision、config、parent checkpoint、resume state、evaluator、输出根、成功更新数和停止规则。Critic 或 PRE_RUN 任一不通过，不执行该延长。
445 | 
446 | ---
447 | 
448 | # 九、最终反主张
449 | 
450 | 本轮不得声称：
451 | 
452 | * 20+40 否定了 H65 语义间接选帧；
453 | * 60 轮在原则上不可能恢复；
454 | * LR 是唯一原因；
455 | * mAP@0.7 下降证明了单一边界机制；
456 | * 单 seed scheduler 结果证明训练效率或稳定性；
457 | * intermediate peak 可以替代 terminal EMA；
458 | * 当前 A/B 已有终态结果。
459 | 
460 | ---
461 | 
462 | ```text
463 | next_owner: Independent DUCA Evaluator
464 | 
465 | next_action:
466 |   等待 Jobs 1252979/1252980 各自形成正式 terminal event；
467 |   先完成只读身份、3000-successful-update、scheduler/LR trace、
468 |   terminal online/EMA、evaluator 与 resumable-state 审计；
469 |   再按本报告的双门、灰区和 terminal-slope 规则自动分类。
470 |   在此之前不得读取中间结果作路线选择，也不得创建第三个 scheduler。
471 | 
472 | dependency:
473 |   两个 A/B 作业的正式 terminal checkpoints、terminal official-validation JSON、
474 |   successful-update audit、完整 LR/gradient/selector diagnostics，
475 |   以及 full resume-state receipt。
476 | 
477 | expected_return_at:
478 |   两个 A/B 作业均形成正式 terminal event并完成独立只读封存之时。
479 | ```
````

### File: ../OpenTAD_DUCA_H65_LRSchedule_20260824/configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py
Lines: 1-142
```python
  1 | _base_ = ["./duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py"]
  2 | 
  3 | import os
  4 | 
  5 | 
  6 | def _required(name):
  7 |     value = os.environ.get(name, "")
  8 |     if not value:
  9 |         raise ValueError(f"{name} is required for DUCA rate curriculum stage 2")
 10 |     return value
 11 | 
 12 | 
 13 | duca_stage1_checkpoint = _required("DUCA_STAGE1_CHECKPOINT")
 14 | duca_stage1_checkpoint_sha256 = _required("DUCA_STAGE1_CHECKPOINT_SHA256")
 15 | duca_stage1_checkpoint_epoch = int(_required("DUCA_STAGE1_CHECKPOINT_EPOCH"))
 16 | 
 17 | # Stage 2 has a single, fresh 6,000-step optimizer schedule.  The first half
 18 | # retains coarse supervision while smoothly turning on learned sampling and
 19 | # detector feedback; the second half is TAD-led but never drops semantic and
 20 | # transition supervision to zero.
 21 | duca_stage2_half_steps = 3000
 22 | 
 23 | # Explicit admission metadata.  These values already come from the inherited
 24 | # official-60 training recipe; spelling them out lets the fail-closed Stage-2
 25 | # validator verify the frozen seed and successful-update budget before loading
 26 | # the Stage-1 handoff.
 27 | seed = 3407
 28 | total_epochs = 60
 29 | max_updates = 6000
 30 | 
 31 | duca_sampling_rate_contract = dict(
 32 |     route="DUCA_RATE_CURRICULUM_STAGE2_JOINT384",
 33 |     task="offline_temporal_action_detection",
 34 |     stage="low_lr_joint_rate_adaptation_then_tad_led_joint_training",
 35 |     pre_backbone_plugin=True,
 36 |     stage1_initialization="full_uniform_k384_ema_model",
 37 |     optimizer_scheduler_amp_state_reset=True,
 38 |     detector_gradient="density_transport_st",
 39 |     final_loss_emphasis=dict(
 40 |         detector=1.0,
 41 |         actionness=0.25,
 42 |         transition=0.10,
 43 |         transition_boundary=0.25,
 44 |     ),
 45 |     paper_claim_allowed=False,
 46 | )
 47 | 
 48 | model = dict(
 49 |     frame_selector=dict(
 50 |         coarse_trunk_lr=1.0e-5,
 51 |         action_head_lr=2.0e-5,
 52 |         transition_scorer_lr=5.0e-5,
 53 |         loss_weights=dict(
 54 |             actionness=1.0,
 55 |             transition=0.50,
 56 |             transition_boundary=2.0,
 57 |         ),
 58 |         loss_weight_schedule=dict(
 59 |             _delete_=True,
 60 |             type="progressive_joint",
 61 |             shape="cosine",
 62 |             warmup_steps=0,
 63 |             transition_steps=duca_stage2_half_steps,
 64 |             actionness=dict(
 65 |                 start=1.0,
 66 |                 end=0.25,
 67 |                 warmup_steps=0,
 68 |                 transition_steps=duca_stage2_half_steps,
 69 |             ),
 70 |             transition=dict(
 71 |                 start=0.50,
 72 |                 end=0.10,
 73 |                 warmup_steps=0,
 74 |                 transition_steps=duca_stage2_half_steps,
 75 |             ),
 76 |             transition_boundary=dict(
 77 |                 start=2.0,
 78 |                 end=0.25,
 79 |                 warmup_steps=0,
 80 |                 transition_steps=duca_stage2_half_steps,
 81 |             ),
 82 |             policy_alpha=dict(
 83 |                 start=0.0,
 84 |                 end=1.0,
 85 |                 warmup_steps=0,
 86 |                 transition_steps=duca_stage2_half_steps,
 87 |             ),
 88 |             detector_gradient=dict(
 89 |                 start=0.0,
 90 |                 end=0.25,
 91 |                 warmup_steps=1000,
 92 |                 transition_steps=2000,
 93 |             ),
 94 |             detector_contribution=dict(
 95 |                 start=0.0,
 96 |                 end=1.0,
 97 |                 warmup_steps=1000,
 98 |                 transition_steps=2000,
 99 |             ),
100 |             asformer_adapt=dict(
101 |                 start=0.0,
102 |                 end=1.0,
103 |                 warmup_steps=0,
104 |                 transition_steps=duca_stage2_half_steps,
105 |             ),
106 |         ),
107 |         actionness_source_cfg=dict(
108 |             policy_hidden_gradient_scope="asformer_full_encoder",
109 |         ),
110 |     ),
111 | )
112 | 
113 | workflow = dict(
114 |     # This curriculum candidate uses the same full THUMOS training and
115 |     # validation protocol as the official-60 arms, but its phase boundary is
116 |     # deliberately outside the frozen selected-axis evidence runtime.  It
117 |     # becomes paper-comparable only after the measured model result is sealed.
118 |     formal_protocol="",
119 |     # This is a new curriculum candidate, not one of the sealed legacy P0
120 |     # variants.  Leaving the inherited P0 contract enabled routes it through
121 |     # the legacy variant binder before model initialization.
122 |     formal_successful_update_contract=False,
123 |     training_profile="duca_rate_curriculum_stage2_joint384",
124 |     # Keep five-epoch validation strictly diagnostic.  The Stage-2 course is
125 |     # always judged by epoch-59 EMA, never by a curve-best checkpoint.
126 |     intermediate_validation_role="learning_curve_only",
127 |     intermediate_validation_selects_checkpoint=False,
128 |     # A pre-AMP NaN/Inf may be replayed only from the untouched batch state.
129 |     # The engine records every event and still fails closed after this bound.
130 |     max_nonfinite_loss_retries=8,
131 |     training_update_audit_json=os.environ.get("DUCA_STAGE2_UPDATE_AUDIT_JSON", ""),
132 |     model_initialization=dict(
133 |         enabled=True,
134 |         checkpoint_path=duca_stage1_checkpoint,
135 |         checkpoint_sha256=duca_stage1_checkpoint_sha256,
136 |         state_key="state_dict_ema",
137 |         expected_checkpoint_epoch=duca_stage1_checkpoint_epoch,
138 |         reset_state_keys=["frame_selector._loss_weight_schedule_step"],
139 |     ),
140 | )
141 | 
142 | work_dir = "exps/thumos/adatad/duca_sampling_rate_curriculum_stage2_joint384"
```

### File: ../OpenTAD_DUCA_H65_LRSchedule_20260824/configs/adatad/thumos/duca_h65_60_stage2_am_rpch25.py
Lines: 1-113
```python
  1 | """H65-60 attribution: mature Stage-1 plus AM-RPCH25 Stage-2."""
  2 | 
  3 | _base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]
  4 | 
  5 | 
  6 | duca_stage2_transition_steps = 2000
  7 | duca_stage2_total_steps = 3000
  8 | 
  9 | duca_sampling_rate_contract = dict(
 10 |     route="DUCA_H65_60_STAGE2_AM_RPCH25",
 11 |     stage="mature_stage1_then_area_matched_relative_plateau_cosine_hold",
 12 |     stage1_initialization="full_uniform_k384_epoch29_ema_model",
 13 |     optimizer_scheduler_amp_state_reset=True,
 14 |     curriculum_only_change=True,
 15 |     model_change_allowed=False,
 16 | )
 17 | 
 18 | model = dict(
 19 |     frame_selector=dict(
 20 |         loss_weight_schedule=dict(
 21 |             _delete_=True,
 22 |             type="progressive_joint",
 23 |             shape="cosine",
 24 |             warmup_steps=0,
 25 |             transition_steps=duca_stage2_transition_steps,
 26 |             actionness=dict(
 27 |                 start=1.0,
 28 |                 end=0.25,
 29 |                 warmup_steps=0,
 30 |                 transition_steps=duca_stage2_transition_steps,
 31 |             ),
 32 |             transition=dict(
 33 |                 start=0.50,
 34 |                 end=0.10,
 35 |                 warmup_steps=0,
 36 |                 transition_steps=duca_stage2_transition_steps,
 37 |             ),
 38 |             transition_boundary=dict(
 39 |                 start=2.0,
 40 |                 end=0.25,
 41 |                 warmup_steps=0,
 42 |                 transition_steps=duca_stage2_transition_steps,
 43 |             ),
 44 |             policy_alpha=dict(
 45 |                 start=0.0,
 46 |                 end=1.0,
 47 |                 warmup_steps=0,
 48 |                 transition_steps=duca_stage2_transition_steps,
 49 |             ),
 50 |             detector_gradient=dict(
 51 |                 start=0.0,
 52 |                 end=0.25,
 53 |                 warmup_steps=1000,
 54 |                 transition_steps=1000,
 55 |             ),
 56 |             detector_contribution=dict(
 57 |                 start=0.0,
 58 |                 end=1.0,
 59 |                 warmup_steps=1000,
 60 |                 transition_steps=1000,
 61 |             ),
 62 |             asformer_adapt=dict(
 63 |                 start=0.0,
 64 |                 end=1.0,
 65 |                 warmup_steps=0,
 66 |                 transition_steps=duca_stage2_transition_steps,
 67 |             ),
 68 |         ),
 69 |     ),
 70 | )
 71 | 
 72 | scheduler = dict(
 73 |     _delete_=True,
 74 |     type="RelativeSuccessfulUpdateLR",
 75 |     mode="am_rpch25",
 76 |     max_epoch=30,
 77 |     total_updates=duca_stage2_total_steps,
 78 |     warmup_updates=500,
 79 |     plateau_updates=1000,
 80 |     decay_updates=1000,
 81 |     hold_updates=500,
 82 |     terminal_factor=0.25,
 83 |     horizon_updates=6000,
 84 | )
 85 | 
 86 | workflow = dict(
 87 |     training_profile="duca_h65_60_stage2_am_rpch25",
 88 |     checkpoint_interval=5,
 89 |     require_resumable_training_state=True,
 90 |     end_epoch=30,
 91 |     expected_train_batches_per_epoch=100,
 92 |     expected_successful_optimizer_updates=duca_stage2_total_steps,
 93 |     primary_checkpoint_epoch=29,
 94 |     primary_checkpoint_state_key="state_dict_ema",
 95 |     checkpoint_criterion="terminal_epoch_29_state_dict_ema",
 96 |     intermediate_validation_role="learning_curve_only",
 97 |     intermediate_validation_selects_checkpoint=False,
 98 |     model_initialization=dict(expected_checkpoint_epoch=29),
 99 | )
100 | 
101 | seed = 3407
102 | total_epochs = 30
103 | max_updates = duca_stage2_total_steps
104 | checkpoint_interval_epochs = 5
105 | checkpoint_policy = dict(
106 |     resumable=True,
107 |     keep_latest=3,
108 |     milestones=True,
109 |     final=True,
110 |     final_ema=True,
111 | )
112 | paper_claim_allowed = False
113 | work_dir = "exps/thumos/adatad/duca_h65_60_stage2_am_rpch25"
```

### File: ../OpenTAD_DUCA_H65_LRSchedule_20260824/configs/adatad/thumos/duca_h65_60_stage2_longcosine_h6000.py
Lines: 1-19
```python
 1 | """H65-60 attribution: mature Stage-1 plus truncated historical cosine."""
 2 | 
 3 | _base_ = ["./duca_h65_60_stage2_am_rpch25.py"]
 4 | 
 5 | 
 6 | duca_sampling_rate_contract = dict(
 7 |     route="DUCA_H65_60_STAGE2_LONGCOSINE_H6000",
 8 |     stage="mature_stage1_then_truncated_historical_cosine_horizon",
 9 | )
10 | 
11 | scheduler = dict(
12 |     mode="longcosine_h6000",
13 | )
14 | 
15 | workflow = dict(
16 |     training_profile="duca_h65_60_stage2_longcosine_h6000",
17 | )
18 | 
19 | work_dir = "exps/thumos/adatad/duca_h65_60_stage2_longcosine_h6000"
```

## Answer

# 唯一裁决：`STOP_60_EPOCH_COMPRESSION`

**Nonce：`DUCA-H65-60-COMPRESSION-TERMINAL-v001-20260824`**

严格执行上一轮预注册的 **flat/falling stop branch**。停止 H65 的 `90→60` 轮训练压缩搜索，保留历史 `30+60` 终态 EMA 作为当前 H65 训练参考。

这项裁决只停止：

* `20+40`、`30+30` 及其 scheduler/LR/课程时钟搜索；
* 任何以 intermediate checkpoint、terminal online 或额外 `+1000` updates 挽救 60 轮压缩的尝试。

它**不停止 H65 模型本身**，不否定语义间接非均匀逐帧选择，不裁决 dynamic-K、SingleClock 或 TrueTime；这些必须使用独立实验身份，不能混入本次压缩归因。

---

## 一、为什么停止分支已经被机械触发

终态证据如下。所有数字均为 seed `3407`、官方 THUMOS14 validation、terminal EMA；因此只能作为本实验身份下的单 seed 终态证据。

| 训练身份                     | Avg-mAP | mAP@0.7 |        相对历史 30+60 |
| ------------------------ | ------: | ------: | ----------------: |
| 历史 `30+60`               | 65.1257 | 43.3137 |                参考 |
| 压缩 `20+40`               | 62.4648 | 39.9434 | −2.6609 / −3.3703 |
| `30+30` AM-RPCH25        |   63.22 |   41.25 | −1.9057 / −2.0637 |
| `30+30` LongCosine-H6000 |   63.56 |   41.01 | −1.5657 / −2.3037 |

预注册恢复邻域是：

[
\text{Avg-mAP}\ge64.6257,\qquad \text{mAP@0.7}\ge42.8137.
]

明确失败条件是任一：

[
\text{Avg-mAP}<64.1257
\quad\text{或}\quad
\text{mAP@0.7}<42.3137.
]

两臂不仅没有进入恢复邻域，而且在两项指标上均属于明确失败：

* AM-RPCH25 距恢复门仍差 `1.4057` Avg-mAP 和 `1.5637` mAP@0.7；
* LongCosine-H6000 距恢复门仍差 `1.0657` Avg-mAP 和 `1.8037` mAP@0.7。

更关键的是，两臂 epoch 29 的 Avg-mAP 都低于 epoch 24：

* AM：`63.35 → 63.22`；
* LongCosine：`63.58 → 63.56`。

虽然 epoch 19 到 29 的三点最小二乘总体斜率仍为正，且 mAP@0.7 在最后五轮略升，但上一轮冻结的 `RISING` 定义是合取条件：除正斜率外，还要求 epoch 29 在 **Avg-mAP 与 mAP@0.7 两项上都不低于 epoch 24**。Avg-mAP 条件明确失败，因此必须进入分支 4，而不是“仍在上升”的分支 3。
（依据：`H65_LR_SCHEDULE_TERMINAL_RESULTS-v001`, L17–38；上一轮裁决 L228–254。）

即使违反协议、非法选择 epoch 24，两臂的 epoch 24 指标也仍然是明确失败。因此本裁决并非被 terminal-only 规则人为制造；intermediate-best 同样无法进入恢复邻域。

---

# 二、因果解释：证据现在指向什么

必须区分两个问题：

1. 原始 `20+40` 为什么比 `30+60` 下降；
2. 在恢复成熟 Stage-1 后，两个 `30+30` 为什么仍然失败。

## 1. Stage-1 成熟度不足：解释原始 20+40 下降的最强直接信号

上一轮材料显示：

* 历史 30-epoch Stage-1 terminal EMA：`59.4231`；
* 压缩 20-epoch Stage-1 terminal EMA：`49.5389`；
* 交接点相差 `−9.8842 pp`。

而在同为第 20 轮时，原日程和压缩日程仅相差约 `−1.3318 pp`。这说明原始 `20+40` 最大的问题之一不是简单的 LR 末端过低，而是 Stage-2 从一个明显不成熟的 detector/coarse-semantic 状态开始。

两个新 arm 复用成熟的 30-epoch Stage-1，尽管 Stage-2 只有 `3000` updates、比 `20+40` 的 `4000` 更少，却仍相对 `20+40` 部分恢复：

* AM：`+0.7552` Avg-mAP、`+1.3066` mAP@0.7；
* LongCosine：`+1.0952` Avg-mAP、`+1.0666` mAP@0.7。

这与“Stage-1 成熟度是原始 20+40 下降的重要原因”一致。但它不是干净的单因素估计，因为新 arms 同时改变了 Stage-2 更新数、LR 曲线和课程时钟，不能把上述恢复量全部归给 Stage-1。

## 2. Stage-2 的有效联合适配窗口不足：当前残余差距的首要解释

历史 `30+60` 有：

* `6000` 个 Stage-2 successful updates；
* semantic/policy transition 约在 update 3000 完成；
* feedback 为 `1000 warmup + 2000 transition`，也约在 update 3000 完成；
* 随后约有 `3000` 个 fully active、full-joint updates。

两个新 `30+30` arms 有：

* 总共仅 `3000` 个 Stage-2 updates；
* semantic/policy transition 在 update 2000 完成；
* feedback 为 `1000 warmup + 1000 transition`；
* fully active 的 full-joint tail 只有约 `1000` updates。

因此，恢复成熟 Stage-1 后，仍然缺少的不是单纯“最后 LR 不够大”，而是 selector、detector feedback、ASFormer adaptation 和 detector 已全部开启后，共同沉降的训练暴露。历史日程有约 3000 次这种更新，新 arms 只有约 1000 次。

这是当前**最有力的残余机制解释**，但仍不是已隔离的因果事实。两个 arms 没有直接操纵 full-joint 暴露而保持其他变量不变。

同时，terminal Avg-mAP 已经平台或略降，意味着不能把这一解释简化成：

> “沿当前轨迹机械增加 1000 updates 就一定恢复。”

更准确的推断是：

> 压缩改变了完整联合适配的数量与发生时机，使优化轨迹在 3000 updates 内进入了一个低于历史终点的平衡或权衡区域；当前数据不支持盲目延长同一轨迹。

这正是为何“暴露不足仍是合理主因”与“禁止 `+1000` continuation”并不矛盾。

## 3. 课程与 feedback 时钟压缩：高优先级共因，尚未测量

历史日程和新 arms 不仅总更新数不同，目标权重与输入分布变化速度也不同：

* semantic/policy transition：`3000 → 2000`；
* feedback transition：`1000+2000 → 1000+1000`；
* full-joint tail：约 `3000 → 1000`。

这可能导致：

* learned sampling 在 detector 尚未适应时过快成为主导输入；
* coarse supervision 下降过快；
* detector-to-selector feedback 在较短区间内完成；
* selector 所定义的输入分布与 detector 表征共同变化过急。

但当前材料没有给出转折点附近的梯度、loss、selector entropy 或位置轨迹，因此只能标记为**高优先级机制假设**，不能宣称已经证明“课程太快”。

## 4. LR 衰减速度与累计面积：从主因降级为次要调节因素

LongCosine-H6000 相比 AM-RPCH25：

* 累计相对 LR 面积高约 `18.33%`；
* update 3000 的因子仍为 `0.571157×`，而 AM 为 `0.25×`；
* terminal Avg-mAP 仅高 `0.34 pp`；
* terminal mAP@0.7 反而低 `0.24 pp`。

逐阈值差异为：

| LongCosine − AM |  @0.3 |  @0.4 |  @0.5 |  @0.6 |  @0.7 |
| --------------- | ----: | ----: | ----: | ----: | ----: |
| mAP 差异          | +0.37 | +0.69 | +0.49 | +0.42 | −0.24 |

因此，更大的 LR 面积对低至中高 IoU 有有限帮助，但没有恢复最严格的 `@0.7`，也没有接近双恢复门。LR 不是完全无关，但证据已经不支持把“衰减过快”继续视为主要且充分的解释。

## 5. EMA lag：仍未测量，不能用于挽救

目前只提供 terminal EMA，没有 terminal online 指标、EMA decay 有效时间常数或 online–EMA 权重距离。

所以：

* 不能声称 terminal EMA 落后导致失败；
* 也不能声称 online 与 EMA 一样失败；
* 即使后续只读诊断发现 online 更好，primary checkpoint 仍必须是 terminal EMA，不能 post-hoc 用 online 替代。

## 6. 优化不稳定或实现漂移：当前无支持

两 arms 完成正式训练和官方验证，模型、K、RGB、VideoMAE-S、Adapter、ActionFormer、loss、NMS、split、evaluator 均冻结。没有非有限值、梯度尖峰、selector collapse 或 evaluator 漂移证据。

因此当前不应把失败解释为“训练崩溃”或“实现有 bug”。这些只能由现有日志审计确认或否定。

---

# 三、epoch 19/24/29 曲线的联合解释

| Arm                | epoch 19 | epoch 24 | epoch 29 terminal |
| ------------------ | -------: | -------: | ----------------: |
| AM Avg-mAP         |    62.70 |    63.35 |             63.22 |
| AM mAP@0.7         |    40.34 |    40.98 |             41.25 |
| LongCosine Avg-mAP |    62.62 |    63.58 |             63.56 |
| LongCosine mAP@0.7 |    39.97 |    40.95 |             41.01 |

两臂在 epoch 19→24 都有明显提升，说明训练并非从一开始完全失效。真正决定停止的是 epoch 24→29：

### AM-RPCH25

* Avg-mAP：`−0.13`；
* mAP@0.7：`+0.27`。

由于 Avg-mAP 是五个阈值的均值，按已报告的两位小数近似反推，`0.3–0.6` 四个阈值的平均变化约为 `−0.23 pp`。后段更新可能在略微改善严格边界命中的同时，损失了较宽松阈值下的总体检测质量。

### LongCosine-H6000

* Avg-mAP：`−0.02`；
* mAP@0.7：`+0.06`。

对应 `0.3–0.6` 的平均变化约为 `−0.04 pp`。这更接近平台，而不是仍有全面上升空间。

因此，末段不是“所有指标仍同步改善但恰好被截断”，而是出现了：

* AM：明显的 aggregate–high-IoU 权衡；
* LongCosine：aggregate 平台、high-IoU 微升；
* 两者：均没有形成恢复所需的联合上升。

这不证明训练已经达到全局最优，但足以否定预注册的“当前轨迹明显仍在上升，因此值得无条件继续”的前提。

---

# 四、LongCosine 的结果能排除什么、不能排除什么

## 可以排除或显著削弱

在本冻结实验族和 3000-update 条件内，可以排除以下解释作为**充分解释**：

1. **“失败主要因为 terminal LR 接近零。”**
   LongCosine 在终点仍有 `0.571157×`，依旧明确失败。

2. **“只要增加累计 LR 面积即可恢复历史 H65。”**
   面积增加约 18.33%，但恢复远不足，且 @0.7 不占优。

3. **“AM-RPCH25 的特定拐点是唯一问题。”**
   更接近历史 cosine horizon 的 LongCosine 同样失败。

4. **“第三个 30+30 scheduler 有充分科学依据。”**
   两个有实质区别、均避免低 LR 冻结的方案已覆盖主要 LR-tail 假设；继续搜索将成为 validation-guided scheduler tuning。

5. **“最佳 intermediate checkpoint 能挽救压缩。”**
   epoch 24 仍明显低于失败门，且本就不允许选取。

## 不能排除

当前实验不能排除：

* Stage-2 successful updates 与 full-joint 暴露不足；
* semantic/policy 与 feedback 时钟的交互；
* 名义 LR 面积没有转化为实际参数位移，例如梯度过小、裁剪或 Adam 状态导致的有效步长差异；
* EMA lag；
* 特定的 boundary、动作时长或类别失败；
* 某条尚未测试的 60-epoch 优化轨迹在原则上可能成功；
* 单 seed 随机性。

尤其要注意：AM 与 LongCosine 不只改变“面积”一个标量，而是改变了完整 LR 形状。因此 `+0.34/−0.24` 是两条 LR 轨迹的描述性差异，不能被解释成 LR 面积的纯因果效应。

---

# 五、为何只调整 LR 未恢复 H65

LR 曲线只能改变每个已有更新的名义尺度，不能补回以下缺失：

* 少掉的 3000 个历史 Stage-2 updates；
* 少掉的约 2000 个 fully active full-joint updates；
* 历史课程与 feedback 的绝对时间结构；
* 更长时间的 selector–detector 共同适配；
* 更长的 EMA 暴露。

而且 compressed clocks 使目标权重、采样分布与 detector feedback 在较短区间内同时变化。即使保留更多 LR 面积，这些更新仍发生在与历史训练不同的非平稳目标上。

因此，本次结果支持的不是“LR 完全无效”，而是：

> **在成熟 Stage-1、3000 个 Stage-2 updates 和冻结的 2000-step compressed clocks 下，单独改变 LR tail/area 不足以恢复 H65。**

---

# 六、这是否证明 60 轮原则上不可能等价 90 轮

**没有。**

当前只有 seed `3407` 下三条失败的 60 轮轨迹：

1. `20+40`；
2. `30+30 AM-RPCH25`；
3. `30+30 LongCosine-H6000`。

它们不是完整因子实验，也没有多 seed 稳定性证据。因此不能声称：

* 所有 60-epoch 日程都不可能恢复；
* 60 轮与 90 轮存在理论上的不可等价性；
* H65 必然需要恰好 90 个 calendar epochs。

正确结论是：

> 在 seed 3407、冻结 H65 模型与官方评估协议下，已测试的三条 60-epoch 训练轨迹均未保持历史 30+60 terminal EMA；其中两条成熟 Stage-1 的 LR-tail 归因实验表明，单独修复 LR 衰减与累计面积仍不足以恢复。

停止是基于**预注册纪律、终态曲线和避免无界调参**，不是数学不可能性证明。

---

# 七、最小后续动作

## A. 现在即可执行：只读训练动力学尸检，不再训练

以下分析只能使用已经存在的 checkpoint、日志、预测和 telemetry。不存在的字段必须记录为 `NOT_MEASURED`，不得补算成仿真事实或声称曾被记录。

### 1. matched-successful-update 轨迹对齐——最高优先级

若历史 `30+60` 保存了相应 checkpoints/logs，将历史 arm 与两个新 arms 在相同 Stage-2 successful update 上对齐，至少比较：

* update 0；
* update 1000；
* update 2000；
* update 2500；
* update 3000。

对齐内容包括 terminal/intermediate online 与 EMA、训练 loss、selector loss 和参数位移。目标是判断差距：

* 在 accelerated curriculum/feedback 完成前已经产生；
* 恰好在 update 1000–2000 的时钟切换附近产生；
* 还是在 fully active tail 中持续扩大。

不得用这些点选择 checkpoint。

### 2. 名义 LR 面积与实际更新量审计

从现有日志或 checkpoint 计算：

* 每个参数组实际 LR trace；
* optimizer successful-update counter；
* blockwise 参数位移；
* update 2000→3000 的增量位移；
* 若已记录，梯度 L2、归一化梯度范数、zero-gradient rate、clipping、AMP retry/nonfinite；
* online–EMA 权重距离。

Adam 下累计 LR 面积不是有效位移。该审计应直接回答 LongCosine 的额外面积是否真正作用到了 detector、adapter、coarse trunk、action head 和 transition scorer。

### 3. curriculum/feedback 时钟审计

核验实际 successful-update 轴上的：

* policy alpha；
* actionness、transition、transition-boundary 权重；
* detector gradient；
* detector contribution；
* ASFormer adaptation；
* Stage-1→Stage-2 时 schedule-step 是否按合同 reset；
* retry 是否错误推进或跳过时钟。

重点检查 update 1000 和 2000 前后的 loss、梯度和 selector 行为是否发生同步突变。

### 4. selector 轨迹审计

若已有 telemetry，报告：

* actionness/transition score entropy；
* top-mass concentration；
* 与 exact-uniform 的位置重合率；
* selected-frame displacement；
* gap 分布；
* 是否出现位置塌缩、频繁反转或后期漂移；
* 相同冻结样本在 update 1000/2000/3000 的选择变化。

这可以区分“detector 尚未适配”与“selector 自身退化”，但不能改变终态裁决。

### 5. terminal online 与 EMA 诊断

使用相同官方 evaluator，只读评估已保存的 epoch 19/24/29 online 和 EMA：

* online 与 EMA 均低：反对“仅 EMA lag”；
* online 显著高而 EMA 低：支持 EMA lag；
* EMA 高于 online：支持后期 online 轨迹噪声较大。

无论结果如何，terminal EMA 仍是唯一 primary，不能以 online 挽救。

### 6. 高 IoU 与边界误差分解

从已经封存的 prediction JSON 分析：

* start/end boundary error；
* 动作时长分层；
* short-action 子集；
* 类别与视频分层；
* proposal recall 与 score/calibration。

LongCosine 在 `0.3–0.6` 占优而 `0.7` 落后，最多提示 tight-localization 残余问题；在没有上述误差分解前，不得称为已证明的 boundary failure。

## B. 当前不授权任何新训练

本轮没有足以推翻 flat/falling branch 的新证据，因此：

* 不运行 `+1000 full-joint updates`；
* 不运行 clock-only arm；
* 不修改 EMA；
* 不运行第三个 scheduler；
* 不增加 seed 来继续挽救 60 轮压缩。

只有只读尸检首先得到一个明确、可重复定位的机制签名，例如压缩 arms 在同一 successful update 和同一时钟边界发生模块级梯度中断或 selector collapse，而历史轨迹没有，才值得提交一份**新的、单机制预注册请求**。在该证据出现前，新训练集合为空。

即使未来产生这种证据，也必须另立实验身份；它不能被视为当前 stop branch 的自动续期。

## C. H65 的立即训练合同

当前 H65 后续方法研究应使用：

* Stage-1 `30 epochs / 3000 successful updates`；
* Stage-2 `60 epochs / 6000 successful updates`；
* 历史 absolute curriculum/feedback clocks；
* terminal EMA；
* 同一官方 evaluator。

也就是保留 `30+60` 为训练参考。该参考仍是单 seed 诊断身份，不因此升级为多 seed 稳定性、训练效率或论文级结果。

---

# 八、明确禁止项

本归因问题至此关闭，禁止：

* 全局提高 peak LR；
* 无梯度或有效步长证据的参数组 LR 微调；
* 第三个 `30+30` scheduler；
* 重新运行 `20+40`；
* `+1000` continuation；
* 选择 epoch 24 或其他 intermediate-best；
* 用 terminal online 替代 terminal EMA；
* 修改或重释冻结恢复阈值；
* 调整 EMA decay 作为压缩挽救；
* 把模型、selector、dynamic-K、SingleClock、TrueTime 或 Bridge 混入该压缩归因；
* 用单 seed 结果宣称统计稳定、训练效率或 60 轮原则不可能。

---

# 九、最终论文级表述

允许写入研究账本的结论是：

> H65 的 `20+40` 压缩同时削弱 Stage-1 handoff，并减少 Stage-2 更新、课程/反馈时间和 fully active joint exposure。恢复成熟 Stage-1 后，两条具有显著不同 LR tail 与累计面积的 `30+30` arms 均在 terminal EMA 上明确失败，且 terminal Avg-mAP 已平台或略降。LongCosine 的额外 LR 面积仅带来有限 aggregate 改善，并未改善 mAP@0.7。因此，过快 LR 衰减不能再被视为主要且充分的解释；剩余证据更支持压缩后的联合适配数量与时钟结构不足，但这些因素尚未被单独隔离。依据预注册规则，停止全部 60-epoch compression 调参，并保留 `30+60` 作为当前 H65 训练参考。

```text
next_owner:
  Independent DUCA Evaluator / Training-Dynamics Analyst

next_action:
  1. 封存 STOP_60_EPOCH_COMPRESSION 终态裁决；
  2. 仅使用既有 artifacts，完成 historical 30+60、AM-RPCH25、
     LongCosine-H6000 的 matched-successful-update 轨迹审计；
  3. 输出 online-vs-EMA、实际 LR/参数位移、课程/feedback 时钟、
     selector 动态和 high-IoU 误差的只读 postmortem；
  4. 缺失日志统一标记 NOT_MEASURED；
  5. 不创建任何训练配置或 Slurm 作业。

dependency:
  历史 30+60 与两个 30+30 arms 已存在的 checkpoints、official-validation
  JSON、训练日志、successful-update/LR trace、selector telemetry、
  optimizer/EMA 状态及 prediction artifacts。
  不要求也不允许为了补齐诊断重新训练。

expected_return_at:
  H65_COMPRESSION_POSTMORTEM-v001 的只读证据收据完成并封存之时；
  不以任何新训练终态作为本裁决的后续返回条件。
```

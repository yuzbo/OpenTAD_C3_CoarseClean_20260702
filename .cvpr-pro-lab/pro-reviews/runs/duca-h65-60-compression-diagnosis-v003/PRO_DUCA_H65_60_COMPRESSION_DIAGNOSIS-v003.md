# 裁决：`CONTINUE`

**Nonce：`DUCA-H65-60-COMPRESSION-DIAGNOSIS-v003-20260824`**

这里的 `CONTINUE` 仅表示：**让已经冻结的 `AM-RPCH25` 与 `LongCosine-H6000` 两个 30+30 作业运行到各自唯一的 terminal EMA，并按预先定义的门自动裁决。**它不表示 60 轮压缩已经成功，也不授权第三个 60 轮 scheduler、参数组 LR 微调、峰值 LR 上调或中间 checkpoint 选择。Jobs `1252979/1252980` 仍无已知终态，本报告不推断其 mAP。

---

## 一、核心科学判断

历史 20+40 的下降不是一个可被“学习率太小”概括的单因素现象，而是一个**耦合压缩失败**：

[
\text{weaker Stage-1 handoff}
\times
\text{fewer Stage-2 updates}
\times
\text{shorter curriculum/feedback clocks}
\times
\text{shorter full-joint tail}
\times
\text{earlier LR decay}
\times
\text{shorter EMA exposure}.
]

现有证据中，**Stage‑1 交接成熟度不足是最强的直接信号**；Stage‑2 更新与 full-joint 暴露不足是下一高优先级解释；LR 曲线/面积是当前 A/B 正在直接检验的因素；课程、反馈和 EMA 目前仍是未隔离假设。

严禁把各项差异机械地加成“各贡献多少 pp”。这些因素在同一训练轨迹中相互作用，现有设计不是因子实验。

---

# 二、20+40 下降的因果诊断排序

## 1. Stage‑1 handoff 不成熟——**最强支持**

历史 30 轮 Stage‑1 terminal EMA 为 `59.4231`，压缩 20 轮仅为 `49.5389`，交接点相差 `−9.8842 pp`。而在同为第 20 轮时，原 30 轮日程为 `50.8707`，压缩日程为 `49.5389`，只差 `−1.3318 pp`。〔任务文件 L21–30〕

这支持两个结论：

1. 压缩日程自身的较短 cosine horizon 已经在相同更新数下造成约 `1.33 pp` 差异；
2. 更大的 `9.88 pp` terminal handoff 差距主要在于压缩模型没有经历后续 10 轮成熟过程。

因此，20+40 的 Stage‑2 是从明显更弱的 detector/coarse-semantic 状态启动的。这会迫使联合阶段同时承担：

* 补 detector 基础能力；
* 学习非均匀选择；
* 开启 detector-to-selector feedback；
* 完成 ASFormer/selector 适配。

但不能据此宣称最终 `−2.6609 pp` 全由 Stage‑1 造成。Stage‑2 明显补回了部分 handoff 差距，故 Stage‑1 是强因果候选而不是完整解释。

## 2. Stage‑2 successful updates 减少——**中强支持，尚未隔离**

历史 Stage‑2 有 `6000` 次成功更新，旧压缩只有 `4000` 次；当前 30+30 A/B 更进一步只有 `3000` 次。〔Stage‑2 历史配置 L17–29；20+40 配置 L73–94；任务文件 L34–39〕

“更新数确实减少”是事实；“这造成了多少性能损失”尚未单独测得。但它是非常有力的解释，因为减少的不只是一般训练时间，而是 learned sampling、detector feedback 和最终 detector-led joint optimization 共同生效后的更新数。

如果当前 A/B 在 terminal 前仍持续上升，则这一解释将明显增强；若已经平台或下降，则单纯增加更新的解释减弱。

## 3. full-joint tail 缩短——**重要且可操作，但未直接证明**

在历史 30+60 中：

* semantic/policy 与 feedback 在约 update 3000 完成；
* 随后还有约 `3000` 次完整联合更新。

旧 20+40 中：

-相关时钟在 update 2000 左右完成；
-剩余约 `2000` 次 full-joint tail。

当前 30+30 A/B 中：

-时钟同样在 update 2000 左右完成；
-只剩约 `1000` 次 full-joint tail。〔任务文件 L36–39〕

因此，当前 A/B 即使修复了 Stage‑1，也仍把完整联合阶段压缩到历史的三分之一。该因素与 Stage‑2 总更新数高度相关，但科学含义更具体：问题可能不是“总共少训练”，而是**selector 已 fully active 后，detector 与采样策略共同沉降的时间不足**。

## 4. LR horizon、面积与尾部非零性——**配置事实明确，性能因果正在由 A/B 检验**

旧 20+40 把 Stage‑2 cosine horizon 从 60 轮缩成 40 轮；Stage‑1 同轮次比较已经表明，较短 horizon 会造成可观差异。因此，“过早进入低 LR 区域”有直接旁证，但还不是 Stage‑2 的独立结果。

当前 A/B 在相同 Stage‑1、相同 3000 updates、相同课程时钟下，只改变 LR 曲线：

* `AM-RPCH25`：累计相对 LR 面积 `1999.625`；
* `LongCosine-H6000`：面积约 `2366.228`，高 `18.33%`，update 3000 仍为 `0.571157×`。〔任务文件 L34–39；AM 配置 L72–84；LongCosine 配置 L11–17〕

所以：

* 若 LongCosine 单独恢复，支持“过早衰减/面积不足”；
* 若 AM 单独恢复，说明较高总面积并非必要，平稳的非零 floor/tail 更重要；
* 若两者均恢复，不能宣称某一 LR 形状是唯一原因；
* 若两者均失败，只能说明**在 3000 Stage‑2 updates 下，调整 LR 曲线不足以恢复 H65**。

累计 LR 面积不是有效参数位移的等价物；后者还依赖梯度尺度、方向和 clipping。

## 5. semantic/policy curriculum clock——**已发生改变，因果未隔离**

历史 transition 为 `3000` steps，旧压缩和当前 A/B 为 `2000` steps。actionness、transition、boundary 权重与 policy alpha 因此更快完成迁移。〔历史 Stage‑2 配置 L58–105；20+40 配置 L17–69；AM 配置 L18–68〕

它可能造成：

* policy 在 detector 尚未适应时过早主导输入；
* coarse supervision 过早减弱；
* detector 所见输入分布变化过快。

但没有 selector loss、梯度或策略熵证据时，这仍是机制假设。当前 A/B 两臂使用相同 2000-step clock，因此不能通过两臂差异隔离该因素。

## 6. feedback clock——**已发生改变，因果未隔离**

历史 feedback 为 `1000 warmup + 2000 transition`；旧 20+40 为 `667+1333`；当前 A/B 为 `1000+1000`。〔历史配置 L88–99；20+40 配置 L49–62；AM 配置 L50–61〕

旧压缩是按总长度近似比例加速；当前 A/B 则保留 1000-step warmup、压缩 transition。两者都改变了 detector gradient 和 detector contribution 进入 selector 的时间结构。

目前不能判断它是“过早反馈”“反馈变化过陡”还是无关因素。必须依赖梯度范数、loss 分量和 selector 行为分析，而不能凭最终 mAP 讲故事。

## 7. EMA exposure/滞后——**弱假设**

历史 Stage‑2 EMA 经历 6000 次联合更新，旧压缩 4000 次，当前 A/B 3000 次。若模型在后期仍快速移动，terminal EMA 可能落后于 online model。

但当前材料没有给出：

* EMA decay 的有效时间常数；
* online-final 与 EMA-final 的差异；
* 后期权重变化率。

因此 EMA 目前只能列为待检验假设。即使 online-final 更好，冻结 primary 仍是 terminal EMA，不能 post-hoc 改用 online checkpoint。

## 8. 欠拟合还是优化不稳定——**当前无法判定；欠拟合先验更强**

目前只有 terminal 指标，没有足够训练曲线、梯度或 entropy 收据。

* 若 loss 持续下降、梯度有限、validation terminal 仍上升、selector 未塌缩，支持欠拟合/暴露不足；
* 若出现梯度尖峰、loss 振荡、selector 熵骤降、online–EMA 强烈分离或高 IoU 反复退化，才支持不稳定。

不能仅凭 20+40 性能低就称其“训练不稳定”。

## 9. 实现差异——**当前证据不支持其为主因**

独立终态收据复现了 seed、checkpoint epoch、更新数、211 个 validation videos、evaluator 身份及所有 delta，并把变化界定为训练时长/课程变化，而非 H65 selector 机制变化。〔终态收据 L28–32〕

两个历史 terminal checkpoint 缺 `rng_state` 与 `data_loader_state`，这阻止完整 resume/论文级复现，但不否定其冻结 terminal EMA 比较。没有证据表明 `−2.6609 pp` 是 evaluator 或 selector 实现漂移造成的。

---

# 三、为什么当前策略是正确的

## 1. 保留 30 轮 Stage‑1：正确且必要

这是对最强 confound 的直接修复。继续从 20 轮 Stage‑1 调 LR，只会把“较弱初始化”和“Stage‑2 scheduler”继续混在一起。

## 2. 保持基础 LR 不变：正确

历史 30+60 已经证明这组基础 LR 至少能够到达 H65 邻域。当前没有任何证据显示峰值 LR 太小。

整体抬高峰值 LR 会同时改变：

* detector/backbone/adapter 的绝对更新尺度；
* coarse trunk/action head/transition scorer 的尺度；
* feedback 开启时的耦合强度；
* AMP、clipping 和 EMA 行为。

这不是对压缩机制的最小修复，而是开启新的优化网格。

## 3. 先改变 decay/horizon，而非峰值：正确

旧压缩同时缩短 horizon 和训练暴露。当前 A/B 保持峰值与参数组比例，直接测试：

* 更大的累计 LR 面积是否必要；
* 非零 tail 是否能防止联合阶段过早冻结。

`RelativeSuccessfulUpdateLR` 对各参数组乘同一个相对因子，能够保持原有组间 LR 比例，优于使用一个绝对 `eta_min` 使不同 base-LR 组发生比例扭曲。

## 4. 保留非零尾段：正确，但不能替代更新数

AM 的 `0.25×` hold 与 LongCosine 在 update 3000 的 `0.571157×` 都避免 terminal 前 LR 近零。它们可以检验“旧压缩是否过早失去可学习性”，但不能凭空替代少掉的 3000 次 historical Stage‑2 updates，也不能恢复被压缩的 full-joint/EMA exposure。

因此当前策略是**正确的第一诊断**，不是预先保证成功的解决方案。

---

# 四、当前 A/B 的冻结终态决策树

令每臂 terminal EMA 的：

[
A_i=\text{Avg-mAP},\qquad H_i=\text{mAP@0.7}.
]

分类严格沿用冻结门：

* **恢复邻域**：(A_i\ge64.6257) 且 (H_i\ge42.8137)；
* **明确失败**：(A_i<64.1257) 或 (H_i<42.3137)；
* **灰区**：其余情况。

## 分支 1：至少一臂进入恢复邻域

**动作：接受一条 30+30 schedule 作为单 seed 的 H65-60 schedule-feasibility 结果；不再运行第三个 scheduler。**

若两臂均通过，按以下预冻结规则选择：

[
S_i=\min(A_i-64.6257,\ H_i-42.8137),
]

选择 (S_i) 较大者，即对两个门具有更大最小安全余量的 arm。若两者 (S_i) 相差不超过 `0.10 pp`，选择 `LongCosine-H6000`，因为它最接近历史 horizon、非历史拐点更少。

允许结论仅为：

> 在 seed 3407、冻结 H65 结构和协议下，该 30+30 schedule 恢复到了预注册 H65 邻域。

不允许升级为：

* 60 轮与 90 轮严格等价；
* 多 seed 稳定；
* 训练效率论文结论；
* selector 或 dynamic-K 论文结论。

## 分支 2：没有 arm 通过，但至少一臂处于灰区

包括“两臂均灰区”以及“一臂灰区、一臂明确失败”。

**动作：拒绝“60 轮无损压缩”表述；不再做第三个 60 轮 scheduler。仅允许下文唯一的 `+1000 full-joint updates` 延长诊断。**

理由是：灰区已经未通过预注册恢复门。再启动一个全新的 30+30 clock/scheduler 组合，会成为 validation-guided 的第三次 60 轮搜索，而且同时修改 semantic、feedback 和 tail，无法干净隔离原因。

## 分支 3：两臂均明确失败，但最佳臂 terminal 仍在上升

“仍在上升”现在冻结为：

* 使用 Stage‑2 epoch `19/24/29` 三个预定 EMA validation 点；
* 对 Avg-mAP 和 mAP@0.7 分别作关于 epoch 的三点最小二乘直线；
* 两条斜率均大于 0；
* epoch 29 在两项指标上均不低于 epoch 24。

**动作：同样只允许唯一的 `+1000 full-joint updates` 延长诊断。**

这时最诚实的假设是 3000 个 Stage‑2 updates 截断了尚未完成的优化，而不是继续发明一个 60 轮 scheduler。

## 分支 4：两臂均明确失败，且最佳臂已经平台或下降

**动作：`STOP_60_EPOCH_COMPRESSION`。**

不再允许：

* 第三个 30+30 scheduler；
* 整体抬高 peak LR；
* 单参数组 LR 微调；
* 改 semantic/feedback clock；
* 用 intermediate best checkpoint 挽救；
* 复跑 20+40。

保留 30+60 为当前 H65 训练参考，把本轮结论记为 schedule-compression negative evidence。

---

# 五、唯一允许的后续训练：`H65-TAIL-EXPOSURE-EXT1000-v001`

这不是第三个 60 轮方案，而是**在 60 轮未通过后，最便宜地判定“是否只是 Stage‑2/full-joint 暴露不足”**。一旦触发该实验，60-epoch no-loss compression 本身已经被停止；该实验最多建立一个 70-epoch 工程折中。

## 1. Parent arm 的确定

在没有通过 arm 时，定义恢复缺口：

[
D_i=\max(64.6257-A_i,\ 42.8137-H_i).
]

* 灰区分支：只在非明确失败 arm 中选 (D_i) 最小者；
* 双失败上升分支：只在满足上述 `RISING` 定义的 arm 中选 (D_i) 最小者；
* 若差值不超过 `0.10 pp`，选择 `LongCosine-H6000`。

这是一项预注册的 parent 选择规则，不允许人工看曲线后改选。

## 2. 精确训练合同

| 项目                             | 冻结值                                                                        |
| ------------------------------ | -------------------------------------------------------------------------- |
| Stage‑1                        | 30 epochs / 3000 successful updates；同一 epoch‑29 EMA handoff                |
| Stage‑2                        | 从选定 arm 的完整 update‑3000 checkpoint 继续到 40 epochs / 4000 successful updates |
| 总训练                            | 30+40，即 70 epochs                                                          |
| seed                           | 3407                                                                       |
| 模型/数据/K/检测器/loss/NMS/evaluator | 全部不变                                                                       |
| semantic/policy/asformer clock | 保持 2000 steps，禁止 reset                                                     |
| feedback clock                 | 1000 warmup + 1000 transition，update 2000 完成，禁止 reset                      |
| full-joint tail                | 从原 1000 扩展为 2000 updates                                                   |
| base LR                        | 不变                                                                         |
| checkpoint                     | 每 5 epochs；primary 为 Stage‑2 epoch 39 terminal EMA                         |
| intermediate validation        | learning-curve only，不选 checkpoint                                          |
| online/EMA                     | terminal EMA 唯一主结果；online 只诊断                                              |

## 3. Scheduler 必须是原 arm 的原生连续延伸

不得重启 scheduler，不得在 update 3001 重新 warmup。

* 若 parent 为 `AM-RPCH25`：前 3000 update 的因子轨迹必须逐点一致；update `2501–4000` 均保持 `0.25×`。
* 若 parent 为 `LongCosine-H6000`：继续同一个 6000-update historical horizon；前 3000 update 逐点一致，update 3000 为 `0.571157×`，之后沿同一 cosine 继续衰减。Builder 必须导出 update `1–4000` 的权威 factor trace；不接受根据 epoch 重新计算的近似曲线。

## 4. Resume fidelity 是硬门

延长训练必须恢复：

* online model；
* EMA shadow；
* optimizer；
* scheduler；
* AMP scaler；
* successful-update counter；
* curriculum schedule step；
* RNG；
* DataLoader/sampler 状态。

不得把 terminal `state_dict_ema` 重新载入模型后重置 optimizer。若完整状态缺失，延长实验 fail-closed，不得改为一次新的 30+40 全量重跑来扩大本轮搜索。

## 5. 唯一停止规则

只读取 update 4000 / Stage‑2 epoch 39 terminal EMA：

* 若同时满足 `Avg-mAP ≥64.6257` 和 `mAP@0.7 ≥42.8137`：保留 30+40 作为单 seed 工程候选，并把“Stage‑2/full-joint 暴露不足”提升为主要解释；
* 否则：停止全部 H65 schedule compression 调参，保留 30+60 参考。

无第四个 scheduler，无中间 checkpoint 挽救，无峰值 LR 调整。

---

# 六、是否已有证据表明“60 轮原则上无法等价 90 轮”

**没有。**

当前材料只能支持最窄的结论：

> 在 seed 3407 和冻结 H65 实验族下，具体的 20+40 压缩日程不能保持 30+60 terminal EMA，下降为 `−2.6609 pp Avg-mAP` 和 `−3.3703 pp mAP@0.7`。〔终态收据 L19–32〕

它不能证明不存在另一条 60 轮优化轨迹。

即使当前两条 30+30 均失败，也只能说明：

> 已测试的 20+40、AM-RPCH25 30+30 和 LongCosine-H6000 30+30 在该单 seed 下不足以恢复 H65 邻域。

届时停止 60 轮压缩，是基于**实验资源纪律与避免无界 scheduler 搜索**，不是数学上的不可能性证明。

---

# 七、终态数据分析与决策权限

## 1. 结果读取前的 admission gate

在读取 mAP 前，独立 Evaluator 必须确认：

* 正确的 Stage‑1 epoch‑29 EMA checkpoint 及 hash；
* seed 3407；
* Stage‑2 正好 3000 successful updates；
* resolved config 与 scheduler mode；
* 参数组 base LR 未变；
* scheduler factor/LR trace 与累计面积；
* terminal epoch‑29 online 与 EMA 均存在；
* evaluator hash、211 videos、split 与 NMS 一致；
* 无 intermediate checkpoint selection；
* checkpoint 具备完整 resume state。

任一身份不一致，结果是 `INVALID`，不是科学成功或失败。

## 2. 能决定分支的量

只有以下信息可以决定下一步：

1. terminal EMA Avg-mAP；
2. terminal EMA mAP@0.7；
3. 若两臂明确失败，预冻结的 epoch 19/24/29 terminal-slope 分类；
4. 运行身份、更新数或 evaluator 失败可直接使结果无效。

## 3. 只用于机制诊断、不能选 checkpoint 的量

### LR 与梯度

逐 successful update 记录：

* 每个参数组实际 LR；
* detector/backbone/adapter/coarse trunk/action head/transition scorer 的梯度 L2 范数；
* 建议同时报告 (L_2/\sqrt{\text{parameter count}})；
* median、p95、max、zero-gradient rate；
* gradient clipping 次数；
* AMP retry/nonfinite 事件。

用途：

* LongCosine 是否真的提供更大有效更新，而不只是名义面积；
* feedback 开启处是否出现梯度冲击；
* terminal 是欠拟合还是不稳定。

### semantic/selector

记录：

* actionness loss；
* transition loss；
* transition-boundary loss；
* detector loss；
* policy alpha、detector gradient、detector contribution 与 ASFormer adapt 的实际 schedule 值；
* actionness/transition score entropy；
* top-mass concentration；
* selected-frame 相对 exact-uniform 的平均、p95、最大位移；
* gap 分布；
* selector 与 uniform 的重合率；
* 是否出现位置塌缩或大幅来回摆动。

除非发生非有限值、K384 身份破坏或实现合同失败，这些量不能改变预注册终态门。

### EMA 与 online

必须同时报告 terminal online 与 terminal EMA 的全部阈值 mAP。

* online 显著优于 EMA：支持 EMA lag；
* online 与 EMA 均低：反对“仅 EMA 滞后”解释；
* EMA 优于 online：支持后期轨迹仍有噪声。

无论何种情况，primary 仍是 terminal EMA，禁止改用 online 挽救。

### high-IoU

mAP@0.7 是恢复门的一部分，不只是附加分析。mAP@0.3–0.6、类别、动作时长和 boundary error 可用于解释，但不能替代 Avg-mAP 与 mAP@0.7 的双门。

历史阈值下降并非严格随 IoU 单调增加——最大损失出现在 0.5 和 0.7 附近——所以不能未经边界误差分析就宣称这是“纯 boundary failure”。

---

# 八、独立审查链

若触发唯一延长实验：

**Builder** 只能增加一个 extension config/resume entry，冻结 update 1–4000 的 LR trace、第一步 continuation 一致性和完整 resume 字段；不得改模型、loss、selector 或基础 LR。

**Independent Critic** 必须验证：

* update 1–3000 与 parent arm 逐点相同；
* optimizer/scheduler/EMA/RNG/DataLoader 未 reset；
* curriculum step 没有重新从 0 开始；
* update 3001 起已处于 full-joint 状态；
* 只增加 1000 successful updates；
* terminal epoch 39 EMA 是唯一 checkpoint；
* 无 intermediate metric-driven 选择。

**Evaluator PRE_RUN** 必须在训练前封存 revision、config、parent checkpoint、resume state、evaluator、输出根、成功更新数和停止规则。Critic 或 PRE_RUN 任一不通过，不执行该延长。

---

# 九、最终反主张

本轮不得声称：

* 20+40 否定了 H65 语义间接选帧；
* 60 轮在原则上不可能恢复；
* LR 是唯一原因；
* mAP@0.7 下降证明了单一边界机制；
* 单 seed scheduler 结果证明训练效率或稳定性；
* intermediate peak 可以替代 terminal EMA；
* 当前 A/B 已有终态结果。

---

```text
next_owner: Independent DUCA Evaluator

next_action:
  等待 Jobs 1252979/1252980 各自形成正式 terminal event；
  先完成只读身份、3000-successful-update、scheduler/LR trace、
  terminal online/EMA、evaluator 与 resumable-state 审计；
  再按本报告的双门、灰区和 terminal-slope 规则自动分类。
  在此之前不得读取中间结果作路线选择，也不得创建第三个 scheduler。

dependency:
  两个 A/B 作业的正式 terminal checkpoints、terminal official-validation JSON、
  successful-update audit、完整 LR/gradient/selector diagnostics，
  以及 full resume-state receipt。

expected_return_at:
  两个 A/B 作业均形成正式 terminal event并完成独立只读封存之时。
```

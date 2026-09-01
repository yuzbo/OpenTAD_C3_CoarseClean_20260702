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

# DUCA H65 90→60 训练压缩终态科学复盘

**Nonce：`DUCA-H65-60-COMPRESSION-POSTMORTEM-v001-20260824`**
**Frozen Git revision：`ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f`**
**裁决对象：60-epoch H65 无损压缩子问题；不是 H65 语义间接选帧本身。**

# SCIENTIFIC_DECISION：**STOP**

停止继续寻找“总训练 60 epochs / 6000 successful updates 下，与 30+60 无损等价的 H65 日程”。

冻结 **30+60，即 Stage-1 3000 updates + Stage-2 6000 updates，terminal EMA** 为当前 H65 性能训练 recipe。停止新的参数组学习率、decay、warmup、plateau、hold、terminal-factor、Stage-1/Stage-2 比例扫描。

这不是停止 H65，也不是否定固定 K=384 的语义间接非均匀选帧；它只是否定以下可执行假设：

> 在成熟 Stage-1 后，仅靠重新设计 3000-update Stage-2 的共享 LR decay，就能使 30+30 无损等价于 30+60。

冻结提交已确认，两条 30+30 配置共享模型、课程时钟、基础参数组 LR、Stage-1 EMA handoff、3000 次成功更新和 terminal epoch-29 EMA，只在相对 LR schedule 模式及其标识字段上不同。代码测试也显式检查了解析后配置除这些归因字段外相同。

---

## 一、终态证据的直接含义

相对 30+60：

| 压缩方案                   | Avg-mAP 差距 | mAP@0.7 差距 | 终端状态                          |
| ---------------------- | ---------: | ---------: | ----------------------------- |
| 20+40                  | −2.6609 pp | −3.3703 pp | 明确失败                          |
| 30+30 AM-RPCH25        | −1.9057 pp | −2.0637 pp | epoch24 63.35 → epoch29 63.22 |
| 30+30 LongCosine-H6000 | −1.5657 pp | −2.3037 pp | epoch24 63.58 → epoch29 63.56 |

两个 30+30 方案均同时满足：

* 没有进入恢复邻域：`Avg ≥64.6257` 且 `@0.7 ≥42.8137`；
* 触发 clear failure；
* terminal Avg 不高于 epoch24；
* 最优的 LongCosine 仍差 `1.5657 pp Avg` 和 `2.3037 pp @0.7`。

因此不能把结果解释成“只差一点，继续换一个 cosine 即可”。截至 terminal EMA，剩余差距仍是结构性的，并且高 IoU 缺口没有随更高 LR 剂量同步改善。

---

# 二、因果诊断

## 1. 首要原因：缺失的 3000 个 Stage-2 成功更新

这是证据最强、无需推测的事实。

训练引擎只有在 optimizer step 确实成功后，才依次推进：

1. DUCA semantic/policy schedule；
2. LR scheduler；
3. Stage-2 EMA；
4. successful-update audit。

AMP 跳过的 step 不推进这些时钟。

所以 30+30 相比 30+60 缺失的不是抽象的“三十个 epoch 标签”，而是：

* 3000 个额外 minibatch 与数据增强观察；
* 3000 次 AdamW 一、二阶矩状态更新；
* 3000 次参数更新机会；
* 3000 次 decoupled weight-decay 应用；
* 3000 次 EMA 观测；
* 以及历史后半段长期处于成熟联合状态的优化路径。

必须注意：有效 weight-decay 总量还依赖每步 LR，不能简单说“正好少一半正则化”；但“少了 3000 次随机梯度与 AdamW 状态演化”是不可由增大 LR 面积替代的。

### 更关键的阶段分解

历史 Stage-2 的语义/策略与 detector-feedback 在大约 update 3000 才达到终值，之后还有约 **3000-update full-joint tail**。冻结的 30+30 配置将 transition 压到 2000 updates，整个 Stage-2 只剩 3000 updates，因此完全联合尾段仅约 **1000 updates**。历史基础配置明确将 Stage-2 设为 6000 updates，并以 3000 为课程半程。

相对于历史，30+30 实际上同时缺失：

* 约 1000 个更渐进的 transition updates；
* 约 2000 个已经完全开启 semantic/policy、feedback 和 joint adaptation 后的稳定尾段 updates。

这比“cosine 是否衰减太快”更接近核心解释。

---

## 2. Stage-1 handoff 成熟度确实重要，但不能单独量化

30+30 两条路线虽然比 20+40 **少 1000 个 Stage-2 updates**，却都明显优于 20+40：

* AM-RPCH25：相对 20+40，Avg `+0.7552 pp`，@0.7 `+1.3066 pp`；
* LongCosine：Avg `+1.0952 pp`，@0.7 `+1.0666 pp`。

这说明恢复成熟的 30-epoch Stage-1 handoff 有真实价值，尤其对高 IoU 有帮助。

但这不是纯净的 Stage-1 单变量实验，因为 30+30 同时改变了 Stage-2 LR package、transition clock 和 Stage-2 长度。因此正确结论是：

> Stage-1 不成熟是 20+40 失败的重要组成部分；恢复 Stage-1 成熟度可以追回约一部分损失，但仍不足以替代完整 6000-update Stage-2。

不能把上述差值直接写成“Stage-1 成熟度贡献了多少 pp”。

---

## 3. LR 曲线和累计剂量：有影响，但已被证明不是充分解释

AM-RPCH25 与 LongCosine 提供了有意义的 schedule 对照：

* AM-RPCH25 累计相对 exposure：`1999.625`，terminal factor `0.25`；
* LongCosine exposure：约 `2366.228`，高 `18.33%`，terminal factor `0.571157`。

实现按照成功更新直接计算倍率，不按 dataloader 长度二次缩放；测试固定了关键 update 点、累计 exposure、组间 LR 比例以及恢复后轨迹一致性。

更高 exposure 带来的终态变化只有：

* Avg：LongCosine 比 AM 高 `+0.34 pp`；
* @0.7：LongCosine反而低 `−0.24 pp`。

这说明：

1. 更大的累计 LR 剂量可能改善一部分平均检测质量；
2. 它没有恢复高 IoU；
3. 它没有把 terminal 结果带入恢复邻域；
4. 它没有形成持续向上的末端趋势。

因此，“原压缩失败只是 decay 太快”已不再成立。

---

## 4. 新 Stage-2 EMA 滞后：合理的次要假设，但尚未被测量

Stage-2 从 Stage-1 terminal EMA whole-model 权重初始化，但重新建立 optimizer、scheduler、AMP 和新的 Stage-2 EMA。

在 30+30 中：

* 新 EMA 总共只观察 3000 次成功更新；
* semantic/policy 和 feedback 大约到 update 2000 才全部到位；
* EMA 对成熟 full-joint 状态只获得约 1000 次尾段观察；
* LongCosine 在末端仍保持 `0.571` 的相对 LR，更可能产生 online–EMA lag。

所以 EMA 滞后可能贡献终态差距，尤其对 LongCosine。

但当前证据不支持把 1.5–2.3 pp 缺口主要归因给 EMA：

* AM 的 terminal LR 已降到 0.25，仍明确失败；
* 两条路线 epoch24→29 均无恢复趋势；
* 尚未提供 online 与 EMA 的同 checkpoint 官方评估差距。

结论只能是：

> EMA lag 是值得只读核验的次要原因，不是继续训练一个 EMA sweep 的充分理由。

---

## 5. 高 IoU 更明显下降：支持 boundary-support / feedback-clock underexposure，但不是证明

三个压缩结果都有相同方向：

* 20+40：@0.7 缺口 `3.3703 pp`，大于 Avg 缺口 `2.6609 pp`；
* AM：@0.7 缺口 `2.0637 pp`，大于 Avg 缺口 `1.9057 pp`；
* LongCosine：@0.7 缺口 `2.3037 pp`，明显大于 Avg 缺口 `1.5657 pp`。

这一重复模式与以下机制相容：

* transition/boundary semantic head 未充分稳定；
* detector-feedback 对选择策略的校正暴露不足；
* full-joint tail 太短，无法把动作性选择继续校准到精细边界；
* 新 EMA 对成熟边界状态平均不足。

但 aggregate mAP 模式不能证明是哪一个原因。必须通过未加权 boundary loss、边界覆盖率和 selector 几何诊断核验，不能直接声称“已证实 boundary clock 失败”。

---

# 三、是否已足以拒绝“只改 LR decay 即可让 30+30 无损等价”

## **是，作为一个可执行科学假设，已经足以拒绝。**

这里拒绝的是：

> 在保持成熟 Stage-1、相同模型、相同基础参数组 LR、相同 3000 个 Stage-2 successful updates、新建 optimizer/EMA 和 terminal-EMA 判定的前提下，只调整共享 LR decay，就能恢复 30+60。

理由完整且闭合：

1. 已恢复成熟 Stage-1；
2. 已比较两条明显不同的 tail/exposure 路径；
3. LongCosine 比 AM 多 18.33% exposure 并保留更高尾 LR；
4. 两者都触发 clear failure；
5. 两者 terminal 均不超过 epoch24；
6. 高 IoU 没有随更高 LR exposure 改善；
7. 两者之间的差异远小于它们与 30+60 的共同差距。

这不是一个关于“宇宙中不存在任何神奇 LR 函数”的数学证明。它是一个科研路线判定：继续从单 seed 官方 validation 上搜索更多 decay、warmup 或 terminal factor，会变成未注册的验证集日程调参，而不是因果实验。

---

# 四、为什么不能再做一个 60-epoch 小实验

在总预算 6000 updates 内，现在已经暴露出不可消除的资源冲突：

* 保留成熟 Stage-1 需要 3000 updates，于是 Stage-2 只剩 3000，已经由两条 LR 路线明确失败；
* 给 Stage-2 更多 updates，就必须压缩 Stage-1，而 20+40 已显示 handoff 成熟度损失；
* 25+35 等中间比例只能形成另一轮比例插值或网格搜索，不能区分 Stage-1 maturity、joint exposure、EMA 和 LR；
* 继承 optimizer/EMA、重叠课程、加速 Stage-1 或更换优化器都将成为新的优化机制，而不再是“只压缩训练日程”。

因此，同一 60-epoch 预算下不存在一个尚未运行、又能单独区分“joint exposure 不足”与“LR/EMA 路径错误”的最小单臂实验。

增加 epoch 当然可能恢复，但那会直接回到 30+60，不能冒充压缩成功。

---

# 五、冻结后的科学处置

## 冻结性能 recipe

当前 H65 性能训练 recipe 固定为：

* Stage-1：exact-uniform full-model，30 epochs / 3000 successful updates；
* handoff：Stage-1 terminal EMA whole-model；
* Stage-2：joint H65，60 epochs / 6000 successful updates；
* Stage-2 optimizer/scheduler/AMP/EMA：重建；
* primary checkpoint：terminal epoch-59 EMA；
* 不允许 intermediate checkpoint 选优；
* seed=3407 的 `65.1257 / 43.3137` 仅为单 seed 官方 validation 收据，不是统计结论或论文终值。

## 冻结负证据

以下结果永久保留为 schedule-compression negative evidence：

* 20+40 compressed curriculum；
* 30+30 AM-RPCH25；
* 30+30 LongCosine-H6000。

不得在未来只报告 epoch24，也不得删除 terminal failure 后将其中间峰值重新命名为压缩成功。

## 明确禁止

不再运行：

* 参数组独立 LR sweep；
* 新 warmup/plateau/decay/hold 比例；
* terminal factor sweep；
* 25+35、28+32 等课程比例插值；
* EMA decay sweep；
* 为恢复压缩结果而引入 Query-Bridge、UVT、Fovea、SingleClock、TrueTime 或 dynamic K；
* 任何根据本轮终态指标反向设计的新 checkpoint 选择规则。

训练日程不是论文创新点。DUCA 的论文价值来自 acquisition、物理时间、动态预算和真实重型计算下降，而不是把 90 个训练 epoch 压成 60。

---

# 六、最多五项只读诊断

这些诊断只完善因果复盘，不自动重开训练。

| 优先级 | 观察量                                                                                                                            | 支持或反驳的原因                                                        | 会改变下一步的结果                                                                                                                                         |
| --- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 对齐 Stage-2 successful update=`2000/2500/3000` 的 30+60、20+40、两条 30+30 official trajectory；保持同一 EMA state key                    | 最直接区分“3000 后 full-joint exposure 仍继续带来增益”与“历史在 update3000 已经领先” | 若历史 update3000 约等于 30+30，随后到6000持续升至65.1257，锁定 joint-tail underexposure；若历史 update3000 已达到恢复邻域而30+30远低，则转为 schedule/handoff/identity 审计，但不做 LR 网格 |
| 2   | 每个对齐 checkpoint 的 online `state_dict` 与 `state_dict_ema` 官方差值                                                                  | 检验新 Stage-2 EMA 是否严重滞后                                          | 只有 terminal online 同时达到 `Avg≥64.6257`、`@0.7≥42.8137` 而 EMA 未达到，才允许 fresh Pro 讨论一次 EMA-path 假设；不得事后改用 online 作为本轮 primary                          |
| 3   | 分类、回归、start/end 或 transition-boundary 的**未加权** loss 轨迹，而非加权 total cost                                                         | 检验高 IoU 下降是否来自 boundary/feedback clock 未成熟                      | 若历史在 update3000 后 boundary loss 继续系统下降而30+30停止，强化 joint exposure；若 boundary loss 已匹配但@0.7仍差，转向表示/支持几何，而不是 LR                                      |
| 4   | 从 Stage-1 handoff 到 update2000/2500/3000 的参数组归一化位移、Adam 一/二阶矩范数；优先 detector、adapter、coarse trunk、action head、transition scorer | 区分“参数尚未走到历史状态”与“走了相似距离但性能路径错误”                                  | 若 LongCosine 的位移/矩状态仍明显不足，支持 underexposure；若已接近历史 terminal 但性能仍差，说明累计 LR 面积和参数距离都不足以定义等价，不再做 schedule sweep                                       |
| 5   | selector entropy、最大物理空洞、GT start/end 邻域覆盖率、短动作覆盖率，在对齐 updates 上比较                                                              | 检验高 IoU 缺口是否由 selector boundary support 未成熟造成                   | 若历史尾段持续改善边界覆盖而30+30没有，归因于 semantic/feedback/full-joint exposure；若几何已匹配但@0.7仍差，下一独立主线应研究表示/物理时间，而非选帧日程                                             |

### 最便宜的核心 falsifier

最便宜、信息量最大的 falsifier 是第 1 项：

> 查看历史 30+60 在 Stage-2 update=3000 时处于哪里。

* 若它与 30+30 terminal 大致同级，随后 update3000→6000 才升至 65.1257，则“缺失 full-joint exposure”得到直接支持；
* 若它在 update3000 已经进入恢复邻域，则当前主解释被反驳，应检查 LR/handoff/EMA/代码身份，但依然没有理由开展无界 LR sweep。

---

# 七、可执行科研合同

## Active claim

> 在当前冻结 H65 机制、seed=3407 和官方 validation 下，成熟 Stage-1 加两个显著不同的 3000-update Stage-2 LR schedule 均不能无损复现 6000-update Stage-2；缺失的联合优化与 EMA 暴露是当前最强解释。

## Anti-claim

本轮不主张：

* 90 epochs 在所有 seed 上都是理论最小训练长度；
* 每个压缩损失都由缺失 updates 单独造成；
* Stage-1、LR、EMA、boundary clock 各自贡献了确定 pp；
* H65 机制失败；
* 65.1257 是论文级、多 seed 或显著性结果；
* 30+60 比任何未来优化器都更优；
* schedule 优化构成论文贡献。

## Cheapest falsifier

`historical 30+60 Stage-2 update3000 terminal-EMA trajectory alignment`，辅以同 checkpoint online–EMA gap。

---

## Builder

**状态：`IDLE`。**

Builder 不得提交新的训练配置、LR schedule、launcher 或压缩实验。

唯一允许的工程动作是必要时提供一个纯只读分析器，从已存在的日志、checkpoint 和 selected-position ledger 中导出上述五项诊断；该分析器不得：

* 加载训练模式；
* 修改 checkpoint；
* 选择最佳 epoch；
* 改 evaluator；
* 生成新模型结果；
* 启动 Slurm/GPU 训练。

---

## Independent Critic

Critic 必须核验：

1. 四组 run 的 exact Git revision、resolved config、seed、split、211-video validation 和 evaluator 身份；
2. 30+30 两臂除 schedule attribution 字段外模型和课程配置一致；
3. Stage-1 handoff 都来自正确 epoch-29 terminal EMA whole-model；
4. successful optimizer update、scheduler、EMA 和 DUCA clock 数完全一致；
5. 没有 AMP skip、resume、data-order 或 checkpoint-state 漂移；
6. primary 结果确实是 terminal EMA，不是 epoch24 或其他中间最优；
7. 20+40 与 30+30 的差异不得被误写成纯 Stage-1 单变量效应；
8. 65.696 或其他非 matched 结果没有进入本裁决。

Critic 只能返回：

* `H65_COMPRESSION_STOP_CAUSAL_AUDIT_PASS`；
* `H65_COMPRESSION_STOP_CAUSAL_AUDIT_BLOCKED`。

---

## Evaluator 准入门

### 对新的压缩训练

`PRE_RUN_STATUS = CLOSED_BY_SCIENTIFIC_STOP`

不再接受 60-epoch H65 compression run。

### 对只读诊断

只有以下信息全部锁定，诊断才准入：

* checkpoint SHA 与 state key；
* exact successful-update count；
* online/EMA 身份；
* official evaluator hash；
* annotation、class-map、NMS 和 split 身份；
* aligned update 的定义；
* 不允许通过诊断选择 checkpoint；
* 所有结果同时公开，不先看一条再决定是否提取另一条。

诊断仅能升级因果解释，不能将任一 30+30 中间 checkpoint 升格为正式结果。

---

# 八、下一步所有权

```text
next_owner:
  Independent Evaluator / Read-only Analysis Owner

next_action:
  从四个冻结 run 的日志、checkpoint 与 update audit 中提取五项只读诊断，
  生成终结性 schedule postmortem；随后把 30+60 作为唯一 H65 训练底座，
  转入一个与训练压缩正交的物理时间/表示归因门。

dependency:
  四组 immutable checkpoint/log/update-audit；
  terminal 与对齐中间 checkpoint；
  official evaluator、split、NMS、annotation 和 config identity receipts；
  不需要任何新训练。

expected_return_at:
  H65_COMPRESSION_READONLY_DIAGNOSTICS-v001
  + CRITIC_H65_COMPRESSION_STOP_CLOSURE-v001
  + H65_COMPRESSION_STOP_RECEIPT-v001
```

下一项论文主线动作不是继续压缩训练，而是在**冻结 30+60 优化 recipe** 后，单独研究 H65 的物理时间/表示因素；该因素不得与 LR、课程长度或 EMA 同时变化。Dynamic K 也必须保持为后续独立因素，不能用来掩盖本轮 schedule failure。

---

# 最终指令

**停止 60-epoch H65 无损压缩。**

保留 20+40、AM-RPCH25 和 LongCosine 的完整负结果；冻结 30+60 terminal-EMA 为性能 recipe；完成一次只读因果闭环后，将科研资源转回与论文主张直接相关的 acquisition、物理时间和动态预算问题。

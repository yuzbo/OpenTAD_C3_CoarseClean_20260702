ZOOMTOKEN-BPNS-R1-V004-TERMINAL-PRO-v001-20260829T130000+0800::g-p-6a79701398bc8191a9ef61db6302b24b

**request_id：** `PRO_BPNS_R1_V004_TERMINAL_COST_AND_DIAGNOSTIC_REVIEW_REQUEST-v001`
**Project ID：** `g-p-6a79701398bc8191a9ef61db6302b24b`
**请求恢复身份：** 按用户给定事实，此前浏览器调用在上传和提交前终止，实际科学提交数为 0；本回答是同一 request、nonce 和 prompt 的恢复裁决，不是第二个科学请求。

**实际使用的七个附件：**

1. `PRO_BPNS_R1_V004_TERMINAL_COST_AND_DIAGNOSTIC_REVIEW_REQUEST-v001.md` 
2. `ZOOMTOKEN_BPNS_R1_DECOUPLED_DIAGNOSTIC_AND_COST_CLOSURE_V004_TERMINAL_RECEIPT-2026-08-29.md` 
3. `profile.json` 
4. `terminal_receipt.json` 
5. `PAPER_PROGRESS.md` 
6. `query_pack.md` 
7. `ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md` 

**界面中观察到的模型路由：** `GPT-5.6 Pro`；当前界面没有暴露更细粒度的内部路由或思考强度日志。

# 总裁决：`PIVOT`

```text
v004_engineering_evidence=PASS_STRONG
v004_protocol_evidence=VALID_WITH_DISCLOSED_POWER_UNCERTAINTY
v004_scientific_evidence=VALID_NEGATIVE_FOR_STANDALONE_FULL_STACK_LATENCY_HEADLINE
frozen_joint_gate_decision=STOP_BPNS_R1_EFFICIENCY_HEADLINE
bpns_r1_disposition=FREEZE_AS_ATTRIBUTION_AND_NEGATIVE_SYSTEMS_EVIDENCE
role_contract_decision=KEEP
```

**`STOP_BPNS_R1_EFFICIENCY_HEADLINE` 是冻结合同唯一正确的直接结论。** 该门要求 R1/K100 的四-pass 中位数 p50 比值和完整 pass gross-energy 比值都不高于 `0.95`；实际分别为 `0.9849289616` 和 `0.9350002508`。能耗通过，p50 失败，因此不得通过显存、token 数、模型阶段计时或能耗单项来“投票推翻”联合门。冻结阈值不得后验放宽，也不存在 v005、BPNS-R1 重放或额外 seed 的补救授权。

这不等于 BPNS-R1 的全部科学价值归零。BPNS-R1 应从“论文效率主方法”降级并冻结为：

> **准确率可行的空间支持原语、真实 GPU 阶段与显存/能耗归因证据，以及“结构计算下降未必转化为串行全栈延迟下降”的有效负系统结果。**

---

# 一、工程、协议与科学证据必须分开

## 1. 工程证据：`PASS_STRONG`

v004 是一个有效完成的工程执行：

* candidate `a4694019fd4cbbdc74885e160163e23d947dc05f` 为最小 clean descendant；
* N16R4 `27 passed`，独立 Critic `PASS`，结果盲 Evaluator `PRE_RUN_READY`；
* precheck job `1260092` 完成；
* 唯一正式 job `1260095` 在 `g0059` 以 `COMPLETED 0:0` 终态；
* 八个 pass 顺序、每 pass 792 项、6,336 条成本记录、929,889 条功耗记录、prediction identity 和全部终态产物均存在；
* 没有 retry、resume、第二 seed、辅助臂、训练、阈值修改或 official test 访问。

因此，v004 不是工程失败。

## 2. 协议证据：`VALID_WITH_DISCLOSED_POWER_UNCERTAINTY`

执行身份、population、顺序、checkpoint、prediction SHA、成本范围和 frozen median-of-four 聚合均闭合；主估计可由 raw rows 独立重建。协议完成是真实的。

但协议完成不等于所有测量都同等无条件可靠。K100 pass 3 存在 `2804.8199 ms` 的局部功耗采样间隙，其余 pass 的局部间隙约为 `52–71 ms`。冻结协议事前没有最大 gap 阈值，coverage ratio 仍为 1.0，故不能在看到结果后追溯宣布整轮无效，也不能据此授权重跑；它只能成为能耗结论的显式不确定性。GPU temperature 未测量，因此不得把顺序差异称为热漂移。

还有一个需要明确记录的**导出语义问题**：原始 `profile.json` 在 pass 3 之后的 pass 5、6、7 中仍继续显示同一个 `2804.8199 ms` 的 `max_trace_gap_ms`，尽管这些 pass 的 measurement interval 已经位于该异常之后。这表明该字段至少不能直接被解释为每个 pass 的局部最大 gap，更像是累计 trace 的历史最大值；durable receipt 中“仅 pass 3 局部异常”的判断依赖另行按 pass 时间区间重算。

这是下一成本工具必须修正的局部测量语义，不把 v004 改写成协议失败。

## 3. 科学证据：对精确定义 headline 是有效负结果

v004 已经回答了冻结问题：

> 在当前串行 `decode→H2D→model→postprocess→Soft-NMS` 路径、单 seed、单 checkpoint、单 RTX 4090 条件下，连续 K64 空间支持虽然降低了 GPU 阶段工作、显存和 gross energy，但**没有形成预注册的至少 5% 端到端 p50 改善**。

这是对 **“BPNS-R1 单独成为真实全栈效率 headline”** 的有效科学否定，不是对所有空间压缩、深度压缩或组合压缩的否定。

---

# 二、各成本指标允许和禁止的最窄表述

| 指标               |                       固定条件观察 | 可以写                                                          | 绝不能写                       |
| ---------------- | ---------------------------: | ------------------------------------------------------------ | -------------------------- |
| decode→NMS p50   | R1/K100=`0.98493`，下降 `1.51%` | “端到端 p50 略降，但未达到预注册 5% 门”                                    | “R1 实现了显著/实用加速”            |
| decode→NMS p95   |         `0.97802`，下降 `2.20%` | “固定回放中 p95 方向性下降 2.20%”                                      | “稳定改善尾延迟”或跨工作负载结论          |
| gross GPU energy |         `0.93500`，下降 `6.50%` | “同一 RTX 4090、该串行回放下完整 pass gross energy 下降 6.50%，带功耗 gap 披露” | “总体能效提升”“跨硬件节能”            |
| throughput       |         `1.01711`，增加 `1.71%` | “该单作业串行吞吐提高 1.71%”                                           | “提高系统吞吐能力”或批量扩展结论          |
| peak allocated   |        `0.75130`，下降 `24.87%` | “该配置峰值 allocated 显存下降约 24.9%”                                | “可把 batch 增大多少”或跨硬件 OOM 结论 |
| peak reserved    |        `0.68966`，下降 `31.03%` | “该配置 reserved 峰值下降约 31.0%”                                   | “通用内存效率提升”                 |

上述主结果均来自四个完整 pass 的 arm-level 中位数，而不是把所有窗口简单池化。

## 模型阶段收益是真实的，但不能替代全栈结论

组件中位数显示：

* `model_forward_cuda`：约 `108.54 → 84.43 ms`，下降 `22.21%`；
* `heavy_backbone_cuda`：约 `71.17 → 50.89 ms`，下降 `28.50%`；
* `input_pipeline_serial`：约 `2289.97 → 2279.80 ms`，仅下降 `0.44%`。

按组件中位数的量级比较——注意这些中位数不能逐项严格相加——K100 的 input pipeline 已约为端到端 p50 的 `92%`，model-forward 仅约 `4.4%`。这构成目前最强、且与数据一致的解释：

> **BPNS-R1 确实减少了模型阶段工作，但串行输入/解码占主导，Amdahl 瓶颈吞没了绝大多数 wall-clock 收益。**

因此，36% token 减少不是“没有计算效果”，而是“没有转化为被冻结的全栈延迟效果”。

---

# 三、2.805 秒功耗 gap 的最终处置

我的裁决是：

1. **不追溯作废 v004。** 事前没有 gap 上限，coverage 完整，不能结果后增设规则。
2. **不授权重跑。** p50 门与功耗 trace 无关，已经确定失败。
3. **能耗结果保留，但必须带限定语。** 应写成“固定条件下估计下降 6.50%，其中一个 K100 pass 含一次 2.805 秒插值区间”。
4. **不得使用 temperature/thermal 解释。** temperature 未测量。
5. **未来成本任务必须输出 pass-local gap，而不是累计最大 gap。**

我还进行了一个不改变 frozen estimator、只检验结论敏感性的重算：若完全排除异常的 K100 pass 3，三个正常 K100 pass 的能耗中位数为约 `144176.21 J`，此时 R1/K100 仍约为 `0.93512`。进一步地，在接受“只有 pass 3 局部异常”这一 durable-receipt 归属的前提下，即使把该 pass 的能耗看作任意未知值，另外三个正常 K100 pass 仍使四点中位数的最低可能分母约为 `144159.99 J`，对应比值约 `0.93522`。因此能耗门通过对这一单点异常是稳健的；不确定性影响的是估计精度，而不是联合门的最终停止结论。

---

# 四、准确率、短动作与边界裁决

R1−K100 的 Avg-mAP 与 mAP@0.3–0.7 差值为：

```text
+0.5353 / +0.7520 / +0.1518 / +1.6238 / -0.1042 / +0.2528 pp
```

这支持的最窄结论是：

> 在 seed 42、两个冻结 final-EMA checkpoint 和本次 evaluator 路径下，R1 没有出现广泛准确率崩塌。

它不支持多 seed 非劣性或总体准确率提升。

边界结果不能支持“保护动作边界”：

* matched boundary：`3293 → 3287`，略少；
* start error：`0.149970 → 0.149101`，略好；
* end error：`0.130938 → 0.133124`，略差；
* short-action Avg-mAP：`0.411292 → 0.411555`，近乎不变；
* short matched：`2176 → 2172`，略少；
* short recall@0.7：`0.651268 → 0.654891`，略好。

方向混合、幅度很小、只有一个 seed，也没有不确定性区间。允许的表述最多是：

> “未观察到灾难性的短动作或边界退化，但各边界指标方向不一致。”

**“boundary-preserving”“protects boundaries”“improves boundary localization” 全部关闭。**

---

# 五、BPNS-R1 的最窄论文位置与可发表性

## BPNS-R1 不完全删除，但永久退出独立效率 headline

应冻结为三类证据：

1. **空间支持可行性证据：** 连续无孔洞 K64 在该单 seed 下保持了有竞争力的检测质量。
2. **真实局部成本证据：** 模型阶段 CUDA 时间、显存和 gross energy 确实下降。
3. **负系统归因：** 单纯空间 token 裁剪在当前串行离线 TAD pipeline 中不足以获得至少 5% 的全栈 p50 收益。

不再允许：

* 对 BPNS-R1 做 v005、阈值调整、额外 seed 或成本重放以“救回”独立 headline；
* 把它放在标题、摘要或主结果表中称为端到端高效方法；
* 将能耗和显存单项结果概括成“总体效率提升”；
* 使用边界保护语言。

## 当前论文主张应改写为

> **连续原生支持是一种准确率可行、显存与 GPU 能耗友好的空间压缩原语，但空间裁剪本身不足以显著降低串行离线 TAD 的 decode-to-NMS 延迟；有效方法还必须削减更大比例的模型工作，或以独立、预注册的系统贡献改变占主导的输入执行路径。**

其中后半句是研究动机，不是已完成正结果。

## 当前可发表性

* **BPNS-R1 单独作为效率论文：不具备可发表 headline。**
* **作为更广方法论文中的关键 ablation/负结果：有价值。**
* **作为系统瓶颈分析论文：当前单 seed、单硬件和单数据集仍不足。**
* 即使后续组合探针通过，也仍需 fresh Pro 决定机制归因、closest prior art、新颖性重构和多 seed，而不能直接把 composition-first probe 宣布为论文方法。

---

# 六、角色合同：`KEEP`

现行角色合同已经明确规定：

* Pro 冻结唯一科学任务；
* Codex 只做最小忠实实现；
* 一次 Critic、一次结果盲 Evaluator；
* 正式终态后必须 fresh post-result Pro；
* 不得自动 retry、resume 或创建 successor；
* 工程、协议和科学证据必须分开；
* 失败路线只做有界根因分析，不能无限审计。

v004 最终能够形成完整、可重建、可否定 headline 的证据，说明规则已经发挥作用。功耗 gap 的问题属于**具体 profiler 的局部测量定义缺口**，应在下一任务的 focused tests 和 measurement gate 中修正，不需要扩张角色制度。

```text
role_contract_decision=KEEP
role_contract_replacement_text=NONE
role_rules_sync_required=false
```

---

# 七、唯一下一任务

## 任务 ID

`ZOOMTOKEN-R1-TAR32-FKV-TERMINAL-VALIDATION-AND-K100-MATCHED-FULL-STACK-COST-CLOSURE-v001`

这是一个原子任务，包含“只读训练终态验收 → 最小成本工具适配 → 一次 matched cost job → fresh Pro 返回”四个连续阶段；不是四个可独立选择的任务。

## 为什么选择它，而不是默认接受附件路线

我选择 TAR32-FKV 不是因为 Codex 把它列出，而是因为它满足目前最高的信息增益条件：

* BPNS-R1 已证明模型阶段有明显可削减工作，但单纯空间裁剪的全栈幅度不足；
* TAR32-FKV 在同一 K64 支持上进一步削减奇数块的 Query/output/MLP 更新，直接针对已测得的模型阶段，而不改变 decode-to-NMS 成本定义；
* 其唯一正式训练已经发生，当前只需验收终态和完成一次成本闭环，不需要开启新的设计搜索或第二次训练；
* 它仍是 composition-first falsifier，不预设新颖性或论文成功；
* 立即改做解码/pipeline 工程会改变论文贡献与执行合同，也容易成为在 p50 失败后更换测量问题的后验补救，因此本轮不选择。

## 科学问题

> 已冻结的连续 K64 空间支持与 `[K64,K32]×6` 交替深度更新组合，在保留原冻结准确率条件的同时，能否相对 matched K100 同时实现至少 5% 的完整 decode→Soft-NMS p50 和 gross GPU energy 改善？

R1 v004 只作为只读空间消融背景，不作为新成本作业的第三臂。新的 headline 主比较必须在同一新作业中重新测量 `K100 ↔ TAR32-FKV`，不得把不同日期作业的 R1/K100 与 TAR/R1 比值相乘。

## 阶段 A：只读验收 job `1260166`

Builder 必须先定位并绑定原始 TAR32-FKV 授权文件、request/nonce、训练准确率门和成本规则的路径与 SHA，然后才可读取终态指标。

必须验证：

* exact revision `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`；
* 唯一正式 seed-42 job `1260166`；
* 无 retry、resume、第二 seed 或 checkpoint 挑选；
* 60 epoch、final epoch-59 EMA；
* official THUMOS14 training→validation 路径、批量、AMP、EMA、evaluator/NMS；
* 实际 route ledger 为 `[64,32]×6`；
* 偶数块 K64 完整更新，奇数块仅 K32 Query/output/MLP；
* 全 K64 始终提供 K/V，全 K64 Adapter 仍执行；
* 无 cache、新参数、新 loss、动态基数或 fallback；
* final checkpoint、raw evaluator vector、prediction 和路由统计完整。

若原冻结准确率合同无法在读取指标前以权威文件和 SHA 唯一恢复，或 job/model/output 身份无效，任务终态为：

```text
STOP_BEFORE_COST__TAR32_TERMINAL_PROTOCOL_OR_AUTHORITY_INVALID
```

不得后验制定准确率阈值，不提交成本作业。

## 阶段 B：允许与禁止的开发改动

**执行基线：** 从 `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7` 建立最小 clean/pushed descendant，仅移植或适配 v004 已验证的成本测量逻辑。

**只允许：**

* 新增或修改 TAR32/K100 专用 profiler；
* 新增对应 N16R4 launcher；
* 新增 focused tests；
* 修正 pass-local power-gap 计算、成功侧 anomaly 序列化和终态 receipt；
* 新 JobName、新 result root 和不可变启动回执。

**禁止：**

* 修改 backbone、TAR route、R1 selector、Adapter、detector、训练配置或 loss；
* 改变 K64/K32、block 交替顺序、attention-score 来源或 fallback；
* 训练、resume、重新选 checkpoint、读取中间 checkpoint；
* 增加 R1、DSR6、MOD32、DROP32 或任何第三臂；
* 改变数据、211-video/792-item population、evaluator、Soft-NMS、50-window warmup；
* 改变原冻结准确率门或本轮成本门；
* 使用 official test；
* 重用 v004 的能耗作为新作业 K100 基线；
* retry、requeue、第二次正式提交或在失败后自动建立 successor。

## Builder 必须交付

1. `tar32_terminal_audit_receipt.json`：训练身份、规则权威、checkpoint SHA、final-EMA raw evaluator vector、prediction SHA、route ledger 和全部偏差。
2. 最小 clean/pushed candidate、完整 diff 和 clean status。
3. focused-test 原始输出。
4. 一次 Critic 终态。
5. 一次 result-blind Evaluator `PRE_RUN_READY` 或明确拒绝。
6. 唯一 formal-cost start receipt。
7. 终态 raw cost、power、predictions、evaluator vectors、pass receipts、profile 和 terminal receipt。
8. 一份 paper-claim map，分别标记 direct measured、reconstructed、diagnostic 和 unmeasured evidence。

## 最小 focused tests

必须至少区分以下错误：

1. `[K64,K32]×6` ledger、偶/奇块语义、全 K64 K/V 与全 K64 Adapter；
2. K100 dense 路径不受 TAR 路由污染；
3. strict final-EMA 加载以及 job/checkpoint/source 身份；
4. `K100,TAR32,TAR32,K100,TAR32,K100,K100,TAR32` 顺序、四 pass/arm、792 项和 50 warmup；
5. 主估计先计算每 pass p50/total joules，再取四-pass median，拒绝 pooled-window 替代；
6. 功耗 coverage 必须按各 pass measurement interval 局部切片；
7. 前一 pass 出现 2.805 秒 gap 时，后一正常 pass 的 local gap 不得继承该值；
8. nominal interval `20 ms`，每 pass `coverage_ratio=1.0` 且 `local_max_gap<=100 ms`；`100.001 ms` 必须 fail closed；
9. 缺失、非有限功耗或 gap 超限不得生成有效 energy decision；
10. 每 pass 原子持久化、prediction SHA 重复一致、禁止旧 namespace 恢复；
11. 完整成本范围确实覆盖 decode、H2D、model、postprocess 与 Soft-NMS，并明确标注 video-level NMS 的窗口摊销。

这里的 `100 ms` 是本新任务事前冻结的测量质量上限：它是 nominal 20 ms 的五倍，高于 v004 正常观察的 52–71 ms，同时排除多秒线性插值。它不追溯适用于 v004。

## 一次 Independent Critic

Critic 只检查会改变准入或解释的问题：

* 模型代码相对 `b0a1ca…` 是否零变化；
* 机制与 `[64,32]×6` 是否一致；
* 原始冻结 accuracy/cost authority 是否在读取结果前绑定；
* K100 是否为同作业公平主对照；
* profiler 是否使用 pass-level median；
* power gap 是否真正局部计算；
* 是否存在结果泄漏、checkpoint 选择、第三臂或后验阈值；
* 论文表述是否仍把 probe、创新和正结果分开。

## 一次 Result-blind Evaluator

Evaluator 在读取新成本结果前核验：

* exact clean SHA、clean checkout；
* job `1260166` 的有效终态与唯一 final-EMA；
* K100 与 TAR checkpoint；
* 411-video inventory、211 validation videos、792 ordered items；
* 同一 Slurm-visible RTX 4090、同一软件栈、1 GPU/5 CPU；
* ABBA+BAAB 顺序、warmup、pass 数；
* 新 JobName、新 root、旧 root 封存；
* 20 ms sidecar、100 ms pass-local gap 硬门；
* 只有一次正式提交；
* `sbatch --test-only` 和存储/路径准入。

Evaluator 不得根据 TAR 的准确率高低改变成本顺序、资源、阈值或测量范围。

## 唯一正式实验

若阶段 A 合法且产生有效 final model output，严格依既有冻结要求提交一次 formal cost job：

```text
K100,TAR32,TAR32,K100,TAR32,K100,K100,TAR32
```

每 pass 使用完整 792-item validation population、50 个 warmup；同一 GPU 串行完成八 pass，测量完整 decode→Soft-NMS p50/p95、各阶段时间、throughput、peak allocated/reserved memory 和 gross GPU energy。

## 结果与停止规则

1. **训练/终态协议无效：** 不测成本，返回 fresh Pro；没有科学方向。
2. **有效模型输出：** 按原冻结合同完成唯一成本作业，不因看到准确率不理想而取消已预注册的 Pareto 测量。
3. **headline 生存条件：**

```text
原冻结 TAR32 accuracy gate 全部通过
AND median4(TAR32 p50)/median4(K100 p50) <= 0.95
AND median4(TAR32 gross energy)/median4(K100 gross energy) <= 0.95
AND 无系统性 short-action/high-tIoU/boundary 冲突
```

4. p50 或 energy 任一比值 `>0.95`：

```text
STOP_R1_TAR32_FKV_EFFICIENCY_HEADLINE
```

5. 成本通过但原冻结准确率门失败：

```text
PARETO_DIAGNOSTIC_ONLY__NO_ACCURACY_PRESERVING_HEADLINE
```

6. 功耗 coverage/gap 无效：

```text
INVALID_POWER_MEASUREMENT__NO_JOINT_COST_DECISION
```

保留合法 p50 作为诊断，但不自动重跑。

7. 全部通过：

```text
SURVIVE_COMPOSITE_PROBE_FOR_FRESH_PRO_ONLY
```

这只允许新的 post-result Pro 判断；不自动授权第二 seed、official test、新结构、机制归因或论文主张。

p95、throughput、memory 和边界指标必须完整报告，但不能挽救失败的 p50/energy 主门。

## 必须保存的终态证据

* exact source/config/command/checkpoint/data/software/hardware identity；
* job ID、node、GPU UUID、driver、power limit、起止时间和 Slurm 终态；
* TAR final-EMA 原始六项 evaluator vector、prediction 和 route ledger；
* 八个 pass 的有序 population identity；
* 每 pass raw latency rows、prediction SHA、evaluator vector、peak memory；
* 每 pass raw power trace、local gap、coverage、积分方法和异常；
* median-of-four 重算及 K100/TAR 比值；
* short-action、start/end error、matched 数和 recall@0.7；
* `profile.json`、`terminal_receipt.json`、原始日志及 SHA；
* produced、reconstructible、diagnostic 和 unmeasured evidence 的分栏清单。

---

# 八、精确北京时间期限

```text
builder_plan_due_at=2026-08-29T20:45:00+08:00
terminal_audit_due_at=2026-08-29T22:00:00+08:00
builder_candidate_due_at=2026-08-30T01:00:00+08:00
critic_due_at=2026-08-30T02:15:00+08:00
evaluator_due_at=2026-08-30T03:15:00+08:00
formal_action_due_at=2026-08-30T03:45:00+08:00
queue_check_at=2026-08-30T12:00:00+08:00
queue_blocker_due_at=2026-08-30T12:15:00+08:00
scientific_return_due_at=2026-08-30T22:00:00+08:00
expected_return_bound=正式 cost job 从 RUNNING 起 PT8H30M 内完成终态摄取；终态后 30 分钟内发起 fresh post-result Pro，且绝对不得晚于 scientific_return_due_at
```

若至 `queue_check_at` 仍未开始运行，只返回唯一 Job ID、队列状态和不可变启动回执；不得改变资源、顺序、阈值或重复提交。

---

# 可直接摄取的执行块

```text
decision=PIVOT
v004_engineering_evidence=PASS_STRONG
v004_protocol_evidence=VALID_WITH_DISCLOSED_POWER_UNCERTAINTY
v004_scientific_evidence=VALID_NEGATIVE_FOR_STANDALONE_FULL_STACK_LATENCY_HEADLINE
frozen_joint_gate_decision=STOP_BPNS_R1_EFFICIENCY_HEADLINE
bpns_r1_disposition=FREEZE_AS_ATTRIBUTION_AND_NEGATIVE_SYSTEMS_EVIDENCE
bpns_v005_authorized=false
bpns_replay_authorized=false
bpns_boundary_claim_allowed=false
role_contract_decision=KEEP
role_contract_replacement_text=NONE
selected_task=ZOOMTOKEN-R1-TAR32-FKV-TERMINAL-VALIDATION-AND-K100-MATCHED-FULL-STACK-COST-CLOSURE-v001
scientific_purpose=验收唯一 TAR32-FKV seed-42 训练终态，并在有效模型输出条件下用一次同GPU K100/TAR32 八pass完整回放判断组合空间与深度压缩能否同时跨过5% p50和5% gross-energy门
execution_base=b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7 的最小 clean/pushed cost descendant；训练模型代码零修改
primary_control=K100，同一新作业内重新测量
archival_ablation=BPNS-R1 v004，只读背景，不进入新作业，不跨作业相乘比值
formal_order=K100,TAR32,TAR32,K100,TAR32,K100,K100,TAR32
population=THUMOS14 validation；211 videos；792 ordered items/pass
warmup=50 windows/pass
power_rule=20ms target；coverage_ratio=1.0；pass-local max gap<=100ms
primary_estimator=每pass独立计算p50与完整pass gross joules；每臂取四pass中位数；再形成TAR32/K100比值
headline_acceptance=原冻结accuracy gate通过 AND p50_ratio<=0.95 AND energy_ratio<=0.95 AND无系统性次要冲突
headline_stop=accuracy gate失败或p50_ratio>0.95或energy_ratio>0.95
automatic_retry=false
second_seed_authorized=false
auxiliary_arm_authorized=false
official_test_authorized=false
post_result_pro_trigger=任何COMPLETED/FAILED/CANCELLED/TIMEOUT或客观blocker后立即进入一次全新的exact-Project Pro复盘
experiments_before_post_result_pro_decision=0
builder_plan_due_at=2026-08-29T20:45:00+08:00
terminal_audit_due_at=2026-08-29T22:00:00+08:00
builder_candidate_due_at=2026-08-30T01:00:00+08:00
critic_due_at=2026-08-30T02:15:00+08:00
evaluator_due_at=2026-08-30T03:15:00+08:00
formal_action_due_at=2026-08-30T03:45:00+08:00
queue_check_at=2026-08-30T12:00:00+08:00
queue_blocker_due_at=2026-08-30T12:15:00+08:00
scientific_return_due_at=2026-08-30T22:00:00+08:00
expected_return_bound=RUNNING起PT8H30M；终态后30分钟内fresh Pro
next_owner=Codex Builder
```

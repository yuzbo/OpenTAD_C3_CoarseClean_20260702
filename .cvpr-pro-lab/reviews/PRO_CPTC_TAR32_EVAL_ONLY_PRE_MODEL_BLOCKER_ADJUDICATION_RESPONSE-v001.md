ZOOMTOKEN-CPTC-TAR32-EVAL-BLOCKER-PRO-v001-20260829T203500+0800::g-p-6a79701398bc8191a9ef61db6302b24b
request_id=`PRO_CPTC_TAR32_EVAL_ONLY_PRE_MODEL_BLOCKER_ADJUDICATION-v001`
exact Project ID=`g-p-6a79701398bc8191a9ef61db6302b24b`
browser-visible model route=`GPT-5.6 Pro`
说明：我可以确认当前会话向我暴露的模型身份，但不能独立审计浏览器前端模型选择器的内部路由日志。

实际完整使用的九个附件：

1. `PRO_CPTC_TAR32_EVAL_ONLY_PRE_MODEL_BLOCKER_ADJUDICATION-v001.md` 
2. `ZOOMTOKEN_R1_TAR32_FKV_EVAL_ONLY_SUBMISSION_BLOCKER_RECEIPT-2026-08-29.md` 
3. `ZOOMTOKEN_R1_TAR32_FKV_TERMINAL_AUTHORITY_BINDING_RECEIPT-2026-08-29.md` 
4. `pasted-text.txt` 
5. `ZOOMTOKEN_R1_TAR32_FKV_TERMINAL_CRITIC_RECEIPT-2026-08-29.md` 
6. `run_zoomtoken_r1_tar32_fkv_eval_only_n16r4.sh` 
7. `PAPER_PROGRESS.md` 
8. `query_pack.md` 
9. `ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md` 

# 总裁决

```text
overall_decision=REVISE_AND_CONTINUE
replacement_evaluation_authorized=YES
role_contract_decision=REVISE
unique_next_task=ZT-CPTC-TAR32-TERMINAL-001
authorized_action=ONE_REPLACEMENT_EVALUATION_ONLY_COMPLETION
successor_task_status=ZT-CPTC-RP-K100-v001 remains FROZEN
```

我授权**一次且仅一次替代性 evaluation-only completion**。

这不是重新训练、模型 rescue、第二个科学候选、第二个 seed，也不是对已经观察到的性能结果进行重试。它是对一个在模型、数据加载器和 evaluator 全部启动前终止的调度级作业进行一次受控替代，以完成原本从未发生的第一次科学评测。

---

# 一、对 job `1261121` 的证据分类

## 1. 工程证据

`1261121` 暴露了一个确定性的外部 launcher 缺陷：

* launcher 检查的是视频根顶层普通 MP4；
* canonical 根实际由 `training/validation` 子目录中的 411 个 MP4 符号链接组成；
* 顶层普通 MP4 数为 `0`；
* 递归符号链接数及 `find -L` 后的普通文件数均为 `411`；
* 正确的只读检查是：

```bash
find -L "$VIDEO_ROOT" -type f -name '*.mp4'
```

附件中的修正版脚本确实采用了这一递归、跟随符号链接的检查。 

因此，工程分类是：

```text
PRE_MODEL_EXTERNAL_LAUNCHER_DEFECT
```

## 2. 协议证据

这次作业说明原协议对“一次提交”的计数单位没有充分区分：

* Slurm 调度提交次数；
* 真正进入模型、数据和 evaluator 的科学尝试次数。

原规则同时要求基础设施失败只做最小修复并迅速恢复实验链，又禁止正式提交后的静默 retry；在本例中，两条原则发生了真实歧义。

因此，协议分类是：

```text
PROTOCOL_INCOMPLETE_BEFORE_SCIENTIFIC_EXECUTION
```

它不是一个协议有效但结果失败的 TAR32 实验。

## 3. 科学证据

`1261121` 在 4 秒内终止，且：

* 结果根未创建；
* 模型和 checkpoint 未加载；
* canonical validation loader 未启动；
* evaluator 与 Soft-NMS 未启动；
* 没有 prediction；
* 没有任何 metric；
* 没有训练、resume 或参数更新。

这些事实由 blocker receipt 明确冻结。

所以它对以下问题均提供**零信息**：

* TAR32 是否保持 Avg-mAP；
* TAR32 是否保持 mAP@0.6/0.7；
* TAR32 是否损害短动作；
* TAR32 是否损害起止边界；
* CPTC 的 transformation-compression 假设是否成立；
* TAR32 是否具有真实成本优势。

科学分类必须是：

```text
NO_TAR32_SCIENTIFIC_EVIDENCE
```

不得把它称为负结果、零效应、模型失败或 TAR32 路线失败。

---

# 二、“一次提交”是否已不可逆耗尽

## 裁决：没有耗尽科学尝试，但已经消耗一次调度提交

必须同时保留两个真实计数：

```text
scheduler_submission_count_before_replacement=1
scientific_evaluation_attempt_count_before_replacement=0
```

若执行本次授权的替代作业，计数将成为：

```text
scheduler_submission_count_after_replacement=2
scientific_evaluation_attempt_count_after_replacement=1
```

不能把 `1261121` 从历史中删除，也不能继续声称总 Slurm 提交数为一；但它不应消耗原规则真正要保护的**一次科学评测机会**。

“一次 evaluation-only completion”的科学目的，是避免：

* 重复查看 validation 结果后调整候选；
* 尝试多个 checkpoint；
* 改变 evaluator 或 NMS；
* 重复随机评测后挑选有利结果；
* 借 evaluation-only 名义做训练或模型 rescue。

`1261121` 没有触及上述任何一项。允许修正外部文件清单检查，不会增加模型选择自由度，不会产生第二次指标观察，也不会改变统计机会。

反过来，若按纯字面把一次 `sbatch` 调用视为不可逆耗尽，就会在没有任何模型证据的情况下永久丢弃一个已经完成训练、通过机制审查且 checkpoint 有效的候选。这不是更严格的科学控制，而是让调度语法取代科学问题。

因此，本次是：

```text
AUTHORIZED_REPLACEMENT_COMPLETION
```

而不是：

```text
SILENT_RETRY
SECOND_SCIENTIFIC_EVALUATION
RESCUE_RUN
```

此次授权只覆盖当前已知的一个结果盲、模型前缺陷，不构成无限预模型重提许可。替代作业之后不再自动产生第三次提交权。

---

# 三、替代评测的精确冻结范围

## 3.1 不变的科学身份

以下全部保持冻结：

```text
model_candidate=b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7
branch=codex/zoomtoken-r1-tar32-fkv-v001
training_job=1260166
seed=42
checkpoint=epoch-59 state_dict_ema
checkpoint_sha256=fc70557ef00788f8e788d59464d8c392943638c446d949d586fefc68c6d9390b
config_sha256=b372d759c402bd82dbc758faa4b69e89351d757e57c8f76d1369f5fee7edc8ec
annotation_sha256=ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad
class_map_sha256=a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31
pretrained_sha256=4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251
validation_population=211 videos / 792 ordered items
evaluator=official evaluator
postprocess=configured Soft-NMS
resources=2 GPUs / 8 CPUs
training=false
resume=false
parameter_update=false
```

机制仍然是 `[64,32]×6`：偶数块 K64 完整更新，奇数块 K32 Query/output/MLP，所有 K64 保留当前 K/V、全部通过 Adapter，未选 Token 使用 identity residual bypass。既有 Critic 已对这些机制、梯度、上下文和 checkpoint 身份给出 `PASS_WITH_BLOCKER`，不需要重新审查整个模型。 

## 3.2 唯一允许的运行时修改

运行时唯一允许的语义修改是把错误的视频清单命令换成：

```bash
find -L "$VIDEO_ROOT" -type f -name '*.mp4'
```

修正版 runtime launcher 必须与附件脚本字节一致；本轮附件脚本的 SHA-256 为：

```text
5b157901598782aeb62a95803ff4f8955c8402bfdcfcd2d1d6f9acf89b46e34e
```

以 `e311804f` 中该 launcher 内容为冻结执行表面。文档或 blocker receipt 的同步提交不构成模型代码变化。

## 3.3 结果根与 JobName

由于 `1261121` 没有创建结果根，修正版脚本原本冻结的结果根仍满足“fresh root”条件。不得为了替代作业再引入第三个可选路径。

允许：

* 保持 JobName `zt-r1-tar32-eval-b0a1`；
* 获得一个新的、唯一的 Slurm Job ID；
* 在启动/终态 receipt 中明确写入：

```text
replacement_for_job=1261121
replacement_authority_request_id=PRO_CPTC_TAR32_EVAL_ONLY_PRE_MODEL_BLOCKER_ADJUDICATION-v001
scheduler_submission_ordinal=2
scientific_evaluation_attempt_ordinal=1
```

若原冻结结果根在提交前意外存在，必须 fail closed 并返回 blocker；不得临时换根继续提交。

## 3.4 提交前必须完成的最小检查

Builder 只需完成：

1. 证明模型 source 仍为 clean `b0a1ca...`；
2. 证明 runtime launcher 相对失败版本的科学执行差异只有 `find` 行；
3. `bash -n` 通过；
4. 在 canonical 根只读执行修正后的命令并得到精确 `411`；
5. 重新核对五类输入 SHA；
6. 核对结果根不存在；
7. 核对命令中没有训练、resume、optimizer 或参数更新入口；
8. 通过 `sbatch --test-only`，但不得另建 GPU smoke job。

## 3.5 Critic 与 Evaluator 边界

本轮允许一个**变更面 Critic**，范围只包括：

* `find -L` 是否准确反映 canonical 数据布局；
* launcher 是否发生其它科学漂移；
* candidate/config/checkpoint/data/evaluator/资源是否完全不变；
* 新作业是否确实为 replacement，而不是新候选。

这不是第二次模型科学 Critic，既有 terminal Critic 结论继续有效。

Result-blind Evaluator 在提交前只核对冻结身份、411 清单、结果根、命令和资源，不得读取或推测 TAR32 性能。正式作业完成后，由同一冻结 Evaluator brief 消费 immutable prediction，输出官方完整向量、短动作与边界诊断。短动作和边界分析只允许读取最终 prediction 与 annotation，不能重新运行或修改模型。

## 3.6 明确禁止

本轮禁止：

* 修改候选、模型公式、K、稀疏层位置或选择规则；
* 修改 config、数据、subset、checkpoint、seed、batch 或 GPU 数；
* 修改 evaluator、Soft-NMS、类别映射或预训练权重；
* 训练、resume、补 epoch、参数更新；
* 换用 raw `state_dict`；
* fallback、动态 K、第二 seed、辅助 arm；
* 根据结果修改准确率门；
* 启动 matched cost；
* 实现或运行 `ZT-CPTC-RP-K100-v001`；
* 再次发生错误后自动进行第三次提交。

---

# 四、准确率门和成本基线裁决

## 4.1 准确率门：原样保留

没有任何 TAR32 terminal metric 被观察，因此没有科学理由修改门槛。原冻结门继续采用：

```text
Avg-mAP >= 68.07
mAP@0.6 >= 60.14
mAP@0.7 >= 45.82
short-action mAP decrease <= 1.50 pp
start-boundary median absolute error worsening <= 10%
end-boundary median absolute error worsening <= 10%
```

参考臂仍为：

```text
R1/FULL64 = 69.07 / 61.14 / 46.57
```

这些条件在 TAR32 结果读取前已经被 authority receipt 冻结。

主指标必须以 evaluator 的**未舍入值**与上述固定数值比较；显示用四舍五入不能替代判定。短动作和边界必须使用冻结的 R1/FULL64 定义与基准产物。若该基准产物或诊断定义无法定位，分类为：

```text
MISSING_FROZEN_DIAGNOSTIC_BASELINE
```

不得把缺失项自动视为通过，也不得用 BPNS-v004 的观测重新设计阈值。

## 4.2 成本基线：原样保留，但本任务不启动成本

若 TAR32 通过全部准确率门，正确的当前臂成本比较仍是：

```text
R1/FULL64 versus R1-TAR32-FKV
```

原因是两臂具有相同 R1 K64 空间支持，能隔离“完整更新”与“奇数块 TAR32 transformation compression”的差异。改用 K100 会同时改变空间支持和变换更新，破坏当前阶段的因果解释。

BPNS-v004 的 K100/R1 延迟、能耗和显存结果仍是已有论文证据，但不能成为 TAR32 的事后准入门。最终路线对此已有明确区分。 

即使替代评测通过，本轮也只能返回：

```text
ACCURACY_ADMITTED_PENDING_FRESH_PRO
```

不得直接启动成本。fresh post-result Pro 再决定是否执行已经冻结的 matched full-stack cost。

---

# 五、后继 K100 Residual Probe 裁决

```text
ZT-CPTC-RP-K100-v001=FROZEN_NOT_AUTHORIZED
```

现在解冻它是不合理的，因为：

1. TAR32 已完成训练且 checkpoint 有效；
2. 目前仍没有一项 TAR32 科学指标；
3. 只需一次无模型变化的 evaluation completion 就能裁决这一候选；
4. 并行启动 Residual Probe 会破坏单线程路线，并把未完成的硬选择证据与预测型机制混在一起。

最终路线明确规定先闭合 TAR32，再决定是否进入 K100 residual predictability probe。

若 TAR32 有效但准确率失败，fresh Pro 可将 `ZT-CPTC-RP-K100-v001` 设为下一唯一任务；不能由 Codex自动启动。

---

# 六、Pro/Codex 角色合同裁决

## 决定：`REVISE`

现有合同的科学所有权、单任务原则、最小实现、独立 Critic/Evaluator、正式结果后 fresh Pro、后台静默监控等主体内容全部保留。只在“冻结任务的连续执行责任”中追加以下条款：

> **预科学执行失败与替代提交。** “一次正式实验”或“一次提交”默认约束一次真正进入冻结科学执行边界的尝试，而不单纯等同于一次 `sbatch` 调度调用。若作业在模型/检查点加载、canonical 数据加载器、evaluator、prediction、metric 和任何参数更新全部发生前，因外部 launcher 的确定性且结果盲缺陷终止；且修复不改变候选、模型、配置、数据、checkpoint、seed、资源语义、evaluator/NMS、阈值或主张范围，则原作业必须永久保留并计入调度提交数，但不消耗科学尝试名额。Codex 可在对变更面完成一次最小 Critic 和 result-blind Evaluator 后，连续执行至多一次 replacement submission，无需逐步请求权限。该 replacement 无论在哪一阶段再次失败，都不得自动产生第三次提交，必须返回 fresh Pro。若某项任务明确规定“`sbatch` 调用次数本身即硬上限”，则以该更严格字面规则为准。

同步到 `RTK.md` 的简短原则：

```text
一次科学实验限制的是一次进入模型/数据/evaluator的科学尝试；纯外部launcher的零科学执行失败可在无科学漂移、一次变更面复核后自动替代一次，但总Slurm提交数必须如实记录，replacement后不得再自动重提。
```

这项修订解决本次歧义，不增加通用编排系统，也不放松正式结果后的 Pro 裁决要求。

---

# 七、唯一下一任务与连续执行命令

## 唯一任务

```text
selected_task=ZT-CPTC-TAR32-TERMINAL-001
task_action=RPL1_EVALUATION_ONLY_COMPLETION
```

任务目的只有一个：

> 从同一 epoch-59 `state_dict_ema` 产生一次完整、官方、不可变的 TAR32 validation 结果，并应用原冻结准确率、短动作和边界门。

Codex 应连续完成：

```text
role-rule sync
→ Builder最小冻结与检查
→ 变更面Critic
→ result-blind Evaluator准入
→ 一次replacement sbatch
→ 单一机器侧终态等待
→ 一次终态摄取与结果Evaluator
→ fresh Project Pro交接
```

普通步骤之间不再询问权限。

正式提交后：

* 记录新 Job ID、输入身份、预期产物和最晚返回界限；
* 只允许一个机器侧后台等待进程执行 sleep/终态判断；
* 不使用前台 LLM heartbeat；
* 不反复消费 `squeue/sacct`；
* 只在终态信号、已有异常或硬返回界限触发时恢复；
* 不创建 cost、successor、retry、resume 或第三个 job。

---

# 八、北京时间截止时间

| 环节                                                 | 截止时间                        |
| -------------------------------------------------- | --------------------------- |
| 角色条款同步及 Builder 冻结计划                               | `2026-08-29T21:45:00+08:00` |
| Builder clean/pushed runtime identity、最小检查与证据      | `2026-08-29T22:15:00+08:00` |
| 变更面 independent Critic 终态                          | `2026-08-29T22:45:00+08:00` |
| result-blind Evaluator `PRE_RUN_READY_REPLACEMENT` | `2026-08-29T23:15:00+08:00` |
| 唯一 replacement 正式提交                                | `2026-08-29T23:30:00+08:00` |
| 终态证据及结果 Evaluator 返回                               | `2026-08-30T12:00:00+08:00` |
| mandatory fresh post-result Project Pro 提交         | `2026-08-30T13:00:00+08:00` |

运行边界：

```text
expected_execution_bound=Slurm START 后 3 小时内形成终态
terminal_ingestion_bound=终态后 45 分钟内
hard_return_rule=若 2026-08-30T12:00:00+08:00 仍无终态，则按客观 scheduler blocker 返回；不得追加作业
```

---

# 九、终态分支

## A. 协议完整且六项准确率门全部通过

```text
terminal_classification=ACCURACY_ADMITTED_PENDING_FRESH_PRO
```

保存全部结果，但不启动成本。fresh Pro 判断是否把唯一下一任务设为冻结的 `R1/FULL64` versus `R1-TAR32-FKV` matched full-stack cost。

## B. 协议完整但任一准确率、短动作或边界门失败

```text
terminal_classification=STOP_R1_TAR32_FKV_EXACT_COMPOSITION
```

这只否定当前 `[K64,K32]×6` 硬选择组合。不得改 K、层数、loss、seed 或训练。fresh Pro 决定是否正式解冻 K100 Residual Predictability Probe。

## C. 结果、身份或诊断不完整

```text
terminal_classification=ENGINEERING_OR_PROTOCOL_BLOCKER
```

不作科学解释，不启动成本，也不自动进行第三次提交。

## D. 替代作业已经产生部分 metric 后失败

所有已产生结果必须原样保存，但不得据部分 validation 作方向判断，也不得再次运行模型。直接返回 fresh Pro 裁决。

---

# 可直接摄取的执行块

```text
decision=REVISE_AND_CONTINUE
replacement_authorized=true
selected_task=ZT-CPTC-TAR32-TERMINAL-001
authorized_action=RPL1_EVALUATION_ONLY_COMPLETION
scientific_purpose=完成此前从未进入模型/数据/evaluator的唯一TAR32 final-EMA科学评测

job1261121_engineering_class=PRE_MODEL_EXTERNAL_LAUNCHER_DEFECT
job1261121_protocol_class=PROTOCOL_INCOMPLETE_BEFORE_SCIENTIFIC_EXECUTION
job1261121_scientific_class=NO_TAR32_SCIENTIFIC_EVIDENCE

scheduler_submission_count_before_replacement=1
scientific_evaluation_attempt_count_before_replacement=0
scheduler_submission_count_after_replacement=2
scientific_evaluation_attempt_count_after_replacement=1

model_candidate=b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7
training_job=1260166
checkpoint=epoch-59 state_dict_ema
checkpoint_sha256=fc70557ef00788f8e788d59464d8c392943638c446d949d586fefc68c6d9390b
corrected_launcher_commit=e311804f
corrected_launcher_attachment_sha256=5b157901598782aeb62a95803ff4f8955c8402bfdcfcd2d1d6f9acf89b46e34e
only_runtime_change=find -L "$VIDEO_ROOT" -type f -name '*.mp4'

validation_population=211 videos / 792 ordered items
evaluator=official evaluator
nms=configured Soft-NMS
resources=2 GPUs / 8 CPUs
training=false
resume=false
parameter_update=false

accuracy_gate=Avg-mAP>=68.07 AND mAP@0.6>=60.14 AND mAP@0.7>=45.82
short_action_gate=decrease<=1.50pp
start_boundary_gate=median_absolute_error_worsening<=10%
end_boundary_gate=median_absolute_error_worsening<=10%
accuracy_reference=R1/FULL64 69.07/61.14/46.57
metric_decision_precision=unrounded evaluator values

cost_baseline_if_later_authorized=R1/FULL64 versus R1-TAR32-FKV
cost_action_in_current_task=forbidden
successor_task=ZT-CPTC-RP-K100-v001
successor_task_status=FROZEN_NOT_AUTHORIZED

role_contract_decision=REVISE
automatic_replacement_limit=one
third_submission=forbidden

builder_plan_due_at=2026-08-29T21:45:00+08:00
builder_candidate_due_at=2026-08-29T22:15:00+08:00
critic_due_at=2026-08-29T22:45:00+08:00
evaluator_due_at=2026-08-29T23:15:00+08:00
formal_action_due_at=2026-08-29T23:30:00+08:00
terminal_return_due_at=2026-08-30T12:00:00+08:00
post_result_pro_due_at=2026-08-30T13:00:00+08:00
expected_execution_bound=START+03:00
terminal_ingestion_bound=terminal+00:45

pass_classification=ACCURACY_ADMITTED_PENDING_FRESH_PRO
accuracy_fail_classification=STOP_R1_TAR32_FKV_EXACT_COMPOSITION
invalid_classification=ENGINEERING_OR_PROTOCOL_BLOCKER
post_result_pro_trigger=any terminal state, complete result, partial metric, or objective blocker
next_owner=Codex
```

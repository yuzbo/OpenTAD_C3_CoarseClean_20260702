---
title: DUCA R0-R5 共享前置修正版正式实验
status: experiment_running
updated: 2026-07-23
---

# 目的

修正旧 `cd68d89/a00498e` 部署中的两项纯执行错误，不改变 selector、decoder、损失、AdaTAD/ActionFormer、TemporalMaxer 或数据协议：

1. R4 与 R5 不应各自重复 R0、P0、U/G0、terminal aggregate 和 hard-swap alignment。
2. 多 GPU bundle 的每个 `srun` step 必须只占一张 GPU，不能继承整个作业的 GPU 数而把其余子臂串行阻塞。

# 唯一身份

- Branch: `codex/duca-boundary-burst-20260722`
- Commit: `9f97f2c7f081b10fbf1f63d0602a621c6b43a780`
- GitHub: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/9f97f2c7f081b10fbf1f63d0602a621c6b43a780`
- Clean snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_9f97f2c_20260722`
- Formal root: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343`
- Local focused tests: `22 passed, 4 skipped`
- Remote Linux focused tests and both `bash -n` checks: exit 0

# 正式作业

| 角色 | Job | GPU | 依赖 | 内容 |
|---|---:|---:|---|---|
| R1 contracts | `1180490` | 1 | none | focused production/no-leak/runtime contracts |
| R2/R3 core | `1180491` | 2 | none | R2Q3 soft-detached 与 hard-detached，两臂并行 |
| R2/R3 adapted | `1180492` | 2 | none | R2Q3 soft-adapted 与 R4Q5-G0，两臂并行 |
| shared bootstrap | `1180493` | 2 | none | 唯一 R0、P0、gate、exact-uniform、R2Q3-G0、terminal U/G0 与 alignment |
| R4 | `1180494` | 2 | `afterok:1180493` | 只消费共享封存证据，并行训练 G1/G2 |
| R5 | `1180495` | 4 | `afterok:1180493` | 五预算、两后端、两策略、三种子完整矩阵与成本 |
| Aggregate | `1180496` | 1 | `afterok:1180495` | fail-closed R5 聚合 |

五预算固定为 `K/G = 384/2, 320/2, 256/3, 192/4, 128/6`。R5 共 `2 backends x 2 policies x 5 budgets x 3 seeds = 60` 个完整 official-validation terminal-EMA 单元。

# 已验证的修复行为

这里的“共享证据”不是跨模型共享 GT、特征或梯度。它指 R0/P0/gate/exact-uniform/G0/alignment 等相同前置只计算一次，随后以带路径、提交和哈希的只读产物供 R4/R5 消费。这样避免每个下游实验重复训练同一前端和重复评估，也不允许后续模型反向修改该封存证据。

`1180491` 与 `1180492` 各出现两个同时 RUNNING 的 job step。四个 step 的 Slurm 记录均为 `TRES=gres/gpu=1` 且 `TresPerStep=gres/gpu:1`，证明不再是假并行。`1180493` 当前只运行唯一 frontend/R0 step；其完成后 exact-uniform 与 R2Q3-G0 才在同一两卡 allocation 内并行。

# 被替代的作业

`1180336--1180341` 与 `1180356--1180358` 于 2026-07-22 23:34 +08:00 主动取消。原因是重复 bootstrap 和 step 资源继承错误；其日志及 epoch 4/9/14 部分 checkpoint 原样保留，只能作为执行故障历史，不能作为模型失败、正式 mAP 或论文证据。旧代码的可见训练损失有限且无 Traceback/OOM/non-finite loss，但没有 terminal epoch-59 EMA mAP。

# 当前裁决

状态仅为 `experiment_running`。代码与真实 Slurm 并发合同已验证；R4/R5 尚未开始，因为它们必须消费一次合法 U/G0 official mAP 与 hard-swap alignment。该依赖是模型证据依赖，不是重复工程门禁。任何论文性能结论仍必须等待完整 validation、OpenTAD tIoU 0.3--0.7、terminal epoch-59 EMA 的原始结果。

## 2026-07-22 23:46 进度

- `1180490` 已 `COMPLETED/0:0`，真实 focused 结果为 `96 passed in 26.93s`。
- `1180491/1180492` 的四个 P0 子臂均已进入 epoch 0，日志可见有限损失、K384 和约 3719 MB；孤立 AMP replay 处于 1/8，不构成失败。
- `1180493` 正在执行唯一共享 frontend/R0 step；`1180494/1180495/1180496` 正确等待封存依赖。
- 四个顶层 stderr 仍全部为 0 行；当前无 terminal mAP。

## 2026-07-22 23:50 进度

- `1180491/1180492` 的四个 P0 子臂仍真实并行；其中三个日志已到 epoch 0 batch 40，另一个已到 batch 20，损失均有限。
- 四臂均保持 `K=384`、显存约 `3719 MB`；每臂目前至多一次 AMP replay，未出现 Traceback、OOM、non-finite loss、ValueError 或 FAIL。
- `1180493` 仍只执行一次共享前置；`1180494/1180495` 不再重复 R0，待共享 U/G0 与 hard-swap alignment 封存后直接进入 R4/R5 正式阶段。
- 当前仍没有 terminal epoch-59 EMA mAP，严禁把 P0 损失或旧 R0 replay mAP 当作论文性能。

## 2026-07-23 a00498e Pro 审查复核

原始 Pro 回复已逐字归档为
`docs/methods/reviews/2026-07-23-a00498e-duca-selected-axis-ttdi-pro-review-raw.txt`，
SHA-256 为 `36523b2f1a7456f8d4a4314ea445971f8066eec59611f9632d7bc1d33e31a884`；
结构化复核见
`docs/methods/2026-07-23-a00498e-duca-selected-axis-ttdi-pro-review-absorption.md`。

远端 diff 证明 `a00498e -> 9f97f2c` 只修改并行 bundle 启动器及测试，模型本体未变；
`9f97f2c -> 4f81299` 也只新增粗分类后端 P0 启动脚本和测试。因此以下模型风险仍存在：

- R5 detector 仍以 `selected_axis_index` 作为内部时间轴，尚无 TTDI；
- mandatory boundary groups 进入 DP 前被合并成 bool union，且接纳前没有 exact-K/G
  completeability 检查；
- protected detector bridge 仍是 expected-position 与局部 RGB 时间斜率的 surrogate；
- 五预算 mAP 矩阵已存在，但正式成本 parser 仍只接受 K384/K256。

项目裁决为 `SUBSTANTIAL_ACCEPT_MODEL_DIAGNOSIS / PARTIAL_ACCEPT_TTDI_REMEDY`。
TTDI 只登记为 terminal mAP 后的首选候选，不是当前最终模型。若 learned hard selection
质量已经提高但高 tIoU mAP 未提高，先做 K384/ActionFormer/seed3407 的
`uniform / current learned / learned + zero-init true-time feature residual` 三臂；physical-coordinate
head 是后续第二变量，禁止与 feature residual 一次性混入。当前 Jobs `1180490--1180496`
继续运行，不因这份后验审查取消或重提。

## 2026-07-23 01:16 进度

- `1180490` 保持 `COMPLETED/0:0`；`1180491/1180492/1180493` 已连续运行约 1 小时 40 分钟。
- `1180494/1180495/1180496` 仍按一次共享证据依赖等待，没有重复 bootstrap 或伪并行回退。
- 当前仍无 terminal epoch-59 EMA mAP；同晚新稀疏粗扫描 Job `1180557` 是独立变量实验，不改变或替代本页 R0--R5 身份。

## 2026-07-23 03:42 R2/R3 门禁失败、根因与恢复

- `1180491/1180492` 顶层状态为 `FAILED`，但四个 P0 子臂均已正常完成 20 epoch、每臂
  `6000` 次有效 optimizer update，损失有限且无 OOM/non-finite collapse。失败发生在 P0
  结束后的真实整模门禁，尚未产生 official-60 mAP，因此不是模型性能负结果。
- 四臂同一根因：OpenTAD P0 checkpoint 的 `state_dict` 与 `state_dict_ema` 都保留了全部
  71 个 selector 学习状态，却没有保存 BatchNorm 的六个非参数运行缓冲；旧初始化器把这些
  缓冲误报为学习权重缺失。检查点本身及其 epoch-19 哈希均仍完整。
- 修复分支为 `codex/duca-boundary-burst-gatefix-20260723`，精确提交
  `487a1784554b8c07cbaf8e3948c5aea785a2d8e1`。初始化器现在仍严格拒绝任何缺失的可训练
  参数，只对 checkpoint 未提供的非参数 buffer 显式保留新阶段初值，并在 receipt 中逐项记录。
  Linux focused regression 为 `64 passed`；恢复入口同时强制核验 P0 路径、SHA-256、epoch=19
  与 `state_dict_ema`，不会重复 P0 训练。
- 四个互不依赖的恢复作业已提交：`1180671` soft-detached、`1180672` hard-detached、
  `1180673` soft-adapted、`1180674` R4Q5-G0。run root 为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_487a178_recovery_20260723_034220`；
  deployment manifest SHA-256 为
  `ca3e42e1bc624faaa592f632a77c8452143b444a19e1ec03486cfa9bf288cc25`。
- 即时状态：`1180671/1180672` 已在 GPU 上执行真实整模门禁，日志已越过旧 buffer mismatch
  位置且无新错误；`1180673/1180674` 因 `AssocGrpGRES` 等待资源。每臂只有门禁通过才会继续
  60-epoch 联合训练与 terminal-EMA 官方评测。

## 2026-07-23 03:58 运行时策略修复与精确恢复

- `1180671--1180673` 的六个缺失 BatchNorm buffer 门禁均已通过，但随后在 optimizer 前因
  `invalid selected-axis variant` 退出。根因是独立入口把 P0/实验标签
  `boundary_burst_r2q3_{soft,hard,adapted}_*` 直接当成生产运行时策略 ID；标签用于区分实验，
  运行时策略必须映射到实际存在的 soft-G0 或 hard-G0 配置。该退出没有训练更新或 mAP，
  不是方法性能证据。
- 精确修复提交为 `ca40c9c5a097e8ab083ba3ffd2ff7f5709841010`：soft-detached 与
  soft-adapted 映射到新增的正式 `boundary_burst_r2q3_soft_g0`，hard-detached 映射到既有
  `boundary_burst_r2q3_g0`，R4Q5 映射保持不变。远端不可变快照为
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_ca40c9c_20260723`，Linux focused
  regression 为 `84 passed in 69.38s`，bash 语法、精确 HEAD 与 clean-tree 均通过。
- 首次 ca40 恢复 Jobs `1180682--1180684` 又在 1--2 秒内由 fail-closed 入口拒绝，因为临时
  提交脚本提前创建了本应保持不存在的 `ARM_ROOT`。删除该预创建后重新提交；该问题同样发生在
  模型构建和 optimizer 前，不是数值或性能失败。
- 当前有效恢复 Jobs 为 `1180685` soft-detached、`1180686` hard-detached、`1180687`
  soft-adapted，三者无依赖，均从原 9f97 的已哈希 epoch-19 P0 恢复。run root 为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756`；
  deployment manifest SHA-256 为
  `f6c053a9452c26eefc427f375318fcec03689041b8829fbe1d6fc11d9e88268f`，jobs ledger SHA-256 为
  `62b71e2f5c66f5cb5cc462e0f70f384aaeadda7f841340f4a0a5e070057ab7c3`。即时状态为
  `PENDING(Priority)`，不是失败。
- `1180674` R4Q5-G0 已通过整模门禁，并完成 official-60 epoch 0 的 99 个 batch；日志中的
  `cls_loss/reg_loss` 与总损失有限，K=384，现已进入 epoch 1。两次孤立 AMP replay 后均继续
  有效更新，没有 OOM、Traceback 或 non-finite collapse。
- 当前仍没有 terminal epoch-59 EMA OpenTAD mAP。状态保持 `experiment_running`；禁止把门禁通过、
  P0 损失或 epoch 0 损失写成性能结论。

## 2026-07-23 04:21 恢复训练健康点

- ca40 的三个 R2Q3 恢复 Jobs `1180685--1180687` 均已通过整模门禁并进入 official-60；三臂
  已到 epoch 2，batch-50 总损失约 `0.9514--0.9516`，`cls/reg` 有限，K384，未见新错误。
- R4Q5 `1180674` 已到 epoch 4 batch 50，总损失 `0.9638`，同样有限且继续更新。
- 这证明 buffer 与 runtime-policy 两项修复已经越过原失败点；仍需 terminal epoch-59 EMA 官方
  mAP 才能评价 soft/hard/adapted/R4Q5 的性能。

## 2026-07-23 04:25 自动巡检

- `1180685/1180686/1180687` 均已进入 official-60 epoch 3，`1180674` 已进入 epoch 5；活动
  日志没有新的 Traceback、OOM、ValueError、non-finite collapse 或 FAIL。
- 原始共享 Job `1180493` 仍在运行；`1180494--1180496` 保持依赖等待。其后续若复现已知
  BatchNorm buffer 门禁，只允许从哈希封存的 P0 恢复，不重跑 P0。
- 尚无 terminal epoch-59 EMA OpenTAD mAP；当前只证明修复后的模型能够持续产生有限更新，
  不能据此判断 learned selection 是否超过 exact-uniform。

## 2026-07-23 05:58 自动巡检

- R4Q5 `1180674` 已进入 official-60 epoch 17；R2Q3 soft/hard/adapted
  `1180685--1180687` 已进入 epoch 15，当前损失有限且未见新 Traceback、OOM、ValueError、
  non-finite collapse 或 FAIL。
- 共享前置 `1180493` 的 paired bootstrap 已推进到 `800/1000`，05:46 仍有产物写入，因此是正常
  慢执行而非静默卡死；`1180494--1180496` 继续合法依赖等待。
- 当前仍无 terminal OpenTAD mAP，不能依据训练 loss 或 bootstrap 进度作性能判断。

## 2026-07-23 06:54 R0 终局负证据

`1180493` 完成全部 `1000/1000` paired bootstrap 后按预注册停止条件退出 `2:0`。产物
`r0_summary.json` 的 `status=KILL_PROJECTED_FEASIBLE_SET`、`ok=false`、
`paper_claim_allowed=false`、`selected_weakest_projected_family=null`；summary SHA-256 为
`8f7f79d45bc3fa7697d5d87fe9f770a3c3fd1e661e62c43eb9128a5d61e4a3df`。这不是 Traceback、
OOM 或数值崩溃，而是方法证据未通过门槛。

| R0 family | 内部 holdout Avg-mAP | 相对 uniform | paired 95% CI |
|---|---:|---:|---:|
| exact-uniform | 93.5871 | 0 | - |
| R2Q3 privileged burst | 94.1905 | +0.6034 pp | [-0.5300, +1.3305] pp |
| R4Q5 privileged burst | 93.9992 | +0.4122 pp | [-1.3154, +1.0950] pp |
| unrestricted GT Oracle | 93.9701 | +0.3830 pp | [-2.0322, +1.2146] pp |

预注册要求 projected family 的 CI 下界严格超过 `+0.20 pp`；没有一项合格。因此原后续 Jobs
`1180494/1180495` 已变为 `DependencyNeverSatisfied`，`1180496` 也不能形成合法聚合。禁止把
exit 2 修成“成功”或绕过门槛强选 R2Q3。上表是冻结 detector、training-internal holdout replay，
不是完整 validation official terminal mAP；独立训练中的 `1180674/1180685--1180687` 继续保留，
用于回答学习模型的最终 mAP，而不是反向改写本次 R0 统计裁决。

## 2026-07-23 09:18 科学停止的正确解释

- R0 只有 40 个 frontend-holdout videos；被冻结的 `transition_beta0/epoch_131` detector 曾在完整
  THUMOS training subset 上训练，因此这是 `training_internal_holdout`，不是 detector-unseen 泛化集。
- 1000 次 paired-video bootstrap 中，R2Q3 相对 uniform 的差值为正占 `85.2%`，超过 `+0.2 pp`
  占 `72.3%`；中位数 `+0.4553 pp`，但 2.5% 分位为 `-0.5300 pp`。R4Q5 对应正差值
  `73.3%`、超过 `+0.2 pp` 为 `59.8%`，2.5% 分位为 `-1.3154 pp`。
- 因而 `KILL_PROJECTED_FEASIBLE_SET` 只表示“该小样本冻结重放不足以稳定选择 projected family”，
  不表示 R2Q3 的均值为负，也不能证明联合重训后的 official-validation mAP 必然失败。
- R0 本身标记 `diagnostic_only=true`、`absolute_map_paper_claim_allowed=false`。把它作为原 DAG 的
  算力停止条件是预注册的保守调度选择；独立 official-60 训练臂必须继续完成，才能裁决方法性能。

## 2026-07-23 09:38 official-60 终点裁决合同

- R0 内部重放停止不得取消或替代独立正式臂。必须跑到 terminal
  `epoch_59.pth/state_dict_ema` 完整 THUMOS validation 的五臂为：R2Q3 `1180717`、
  R4Q5 `1180674`、soft-detached `1180685`、hard-detached `1180686`、
  soft-adapted `1180687`。
- 09:38 五臂均为 `RUNNING`，训练进度约为 epoch `34/46/45/42/44`；有限损失和
  optimizer update 持续，未发现 OOM、non-finite collapse、replay 耗尽或新 Traceback。
- 最终只使用各臂官方完整 validation 的 Avg-mAP 与 mAP@0.3--0.7，并与同协议
  exact-uniform `64.49%` 比较。中间 loss、P0 指标和内部 replay 都不得代替该裁决。

## 2026-07-23 09:44 中间性能边界

- 五臂的 official-60 协议关闭训练中 validation，只在 epoch 59 终点用 EMA 完整评估。
  当前所有日志中都没有 `Average-mAP` 或 `mAP at tIoU`，因此五臂的中间 mAP
  都是“尚不存在”，不是“暂时较低”。
- 09:44 最新健康记录：R2Q3 epoch 34 末 `Loss/cls/reg=2.2545/0.2497/0.2376`；
  R4Q5 epoch 46 末 `2.2026/0.2167/0.2163`；soft-detached epoch 45 中
  `2.1851/0.2207/0.2298`；hard-detached epoch 43 中 `2.2702/0.2316/0.2416`；
  soft-adapted epoch 44 末 `2.1474/0.2133/0.2095`。这些数值只证明训练有限稳定，
  不能用于 mAP 排名。

## 2026-07-23 10:02 每十轮完整验证诊断已部署

- 为避免只用训练 loss 猜测学习优劣，新增六条严格匹配的只读学习曲线：exact-uniform、R2Q3、R4Q5、
  soft-detached、hard-detached、soft-adapted。
- 诊断检查点为 one-based epoch 10/20/30/40/50，即 `epoch_9/19/29/39/49.pth` 的
  `state_dict_ema`；epoch 60 直接复用各臂原正式 terminal EMA 评估。
- 所有检查点均使用同一 OpenTAD `tools/test.py`、seed 3407、完整 THUMOS validation 和独立 work dir，
  不读取 raw-prediction cache，也不覆盖训练目录。中间 mAP 只解释收敛速度、峰值回落和臂间学习差异，
  禁止据此选择论文 checkpoint。
- 两个无依赖 Slurm bundle 已实际运行：`1180868` 负责 exact-uniform/R2Q3/R4Q5，`1180869`
  负责 soft-detached/hard-detached/soft-adapted。run root 为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_checkpoint_curve_20260723_100219`；
  deployment manifest SHA-256 为 `c63fa4742c58028a8b9ce44a4e70d434440baf2da8aeeba46ec762a2b414fc19`。

## 2026-07-23 14:04 Remote official-60 terminal recovery (single seed)

- Remote `sacct` and the raw `terminal_evaluation.json` confirm that R4Q5 `1180674`,
  R2Q3 soft-detached `1180685`, hard-detached `1180686`, and soft-adapted `1180687`
  completed 6000 optimizer/scheduler/EMA updates and evaluated `epoch_59.pth/state_dict_ema`
  on the full 211-video THUMOS validation with the OpenTAD tIoU 0.3--0.7 evaluator.
  Their top-level wrappers are marked `FAILED/1:0`, but each arm has a hash-bound terminal
  evaluation JSON and complete evaluator output; the table records the raw terminal artifacts.

| Arm | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 | vs matched uniform 64.49 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R4Q5-G0 (`487a178`) | 63.4211 | 79.5541 | 74.0180 | 66.3352 | 56.2227 | 40.9754 | -1.0689 pp |
| R2Q3 soft-detached (`ca40c9c`) | 63.9794 | 80.2437 | 75.1685 | 66.6974 | 56.3886 | 41.3991 | -0.5106 pp |
| R2Q3 hard-detached (`ca40c9c`) | 64.0002 | 80.3724 | 75.6258 | 67.0555 | 55.8074 | 41.1398 | -0.4898 pp |
| R2Q3 soft-adapted (`ca40c9c`) | 64.1539 | 80.0504 | 75.2577 | 67.2655 | 56.4131 | 41.7828 | -0.3361 pp |

- No single-seed mainline arm exceeds matched exact-uniform. This is negative evidence for C3,
  not a multi-seed family verdict. Jobs `1180868/1180869` remain read-only learning-curve
  diagnostics; their periodic mAP cannot replace the sealed epoch-59 results.

## 2026-07-23 14:13 Read-only convergence-curve interim results

The two diagnostic bundles are still running.  The completed, full-THUMOS
validation `Average-mAP` points are below; the checkpoint directory uses
zero-based names, so `epoch_9/19/29/39/49` are the one-based epochs
10/20/30/40/50 specified by the diagnostic contract.

| Bundle | Arm | epoch 10 | epoch 20 | epoch 30 | epoch 40 | epoch 50 |
|---|---|---:|---:|---:|---:|---:|
| `1180868` | exact-uniform | 30.86 | 48.56 | 60.16 | 63.13 | 64.30 |
| `1180868` | R2Q3 | 30.67 | 48.81 | 59.70 | 62.64 | evaluating |
| `1180869` | soft-detached | 30.34 | 47.67 | 59.99 | 62.64 | 63.65 |
| `1180869` | hard-detached | 29.83 | 47.51 | 59.58 | 62.88 | evaluating |

At the audit time, bundle A had not yet started R4Q5 and bundle B had not yet
started soft-adapted.  These are `experiment_running` / diagnostic-only
results: they show that both completed DUCA trajectories remain below the
matched uniform trajectory at epoch 40, but cannot select a checkpoint,
replace each arm's sealed epoch-59 EMA evaluation, or establish a cost claim.

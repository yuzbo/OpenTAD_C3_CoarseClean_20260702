# DUCA-Marginal-v1：Pro 路线转向与冻结任务

## 裁决

Pro 在摄取 `DUCA-Coverage-v1` 的真实运行前结果后给出 **PIVOT**。固定 `K=384` 内继续优化
时间覆盖不再是当前论文主问题；新的问题是：低成本侦察模型能否预测一个时间窗口再增加一组重型
视觉观察对时序动作检测损失的边际价值，并在每个视频严格不变的总观察预算下，把 VideoMAE 计算
从低价值窗口转移到高价值窗口。

原始可见报告保存在：

`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/pro-reviews/runs/duca-marginal-v1-pivot-user-supplied/visible-report.md`

## 证据解释

- H65 的真实对照不是全时间轴动作性 Top-K，而是预算校准后的确定性系统采样。
- `DUCA-Coverage-v1` 只否定当前 96 个时间锚点的设施位置干预：锚点覆盖相对增益只有
  `3.32%`，最大时间空洞第 95 百分位从 `2` 恶化到 `8`。它没有产生 mAP 或成本结果，也不否定
  所有覆盖、多样性或动态预算方法。
- H65 `65.13%` 与共享 dense AdaTAD `68.73%` 之间仍有约 `3.6` 个百分点差距，但现有证据不能把
  差距单独归因于帧覆盖、物理时间或预算分配。

## 冻结的唯一当前实验

先做冻结 H65 检测器的反事实边际预算实验，不启动新的 60 轮 detector 训练：

1. `Fixed-H65-384`：每个窗口固定 384 个非连续 H65-ranked observations，逐预测复现既有 H65。
2. `Oracle-Reallocate-384`：仅在训练侧 controller holdout 上使用真实三预算反事实效用，检验是否存在
   跨窗口重分配的理论空间；不得在官方 test 上使用 GT 或 oracle。
3. `Learned-Reallocate-384`：侦察特征预测 `384→256` 的损失代价和 `384→512` 的收益；每个视频总
   observation 数严格等于该视频所有窗口的 `sum(min(V_i, 384))`，其中 `V_i` 是窗口可用的唯一
   observation 数。
4. `Fixed-320` 与 `Learned-Allocate-320` 只在前述机制门通过后用于效率 operating point。

`K=256/384/512` 表示选中的非连续原始 observations；16 是预算变化的固定 packet 粒度，不表示连续
16 帧 clip。三种预算必须形成真实不同长度的 VideoMAE 输入，禁止统一补齐到 512 后声称动态计算。

## 代码与数据身份

- clean base：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- branch：`feature/duca-marginal-budget-v1-20260830`
- H65 terminal checkpoint：
  `/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth`
- checkpoint SHA-256：`dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c`
- state key：`state_dict_ema`
- utility controller 的 160/40 划分来自 THUMOS14 训练侧 200 个视频，按视频 ID 和 seed `3407`
  确定性划分；这 40 个视频只对 utility head 留出，不应表述为 H65 detector 未见数据。
- official test 标签对 learned allocator、utility head 和阈值不可见。

exact base 已包含 `PrefixMarginalUtilityBudgetController`、`counterfactual_utility.py`、
`budget_calibrated_sampling_rate` 与 H65 acquisition path；允许在这些既有表面上做最小修改。不得修改
VideoMAE、ActionFormer/AdaTAD head、检测损失、proposal decode、NMS、数据划分、注释、类别映射、
官方评估器、H65 Scout 与 terminal detector checkpoint。

## 运行前机制门

正式官方 test 前依次回答三个问题：

1. 实现是否有效：K384 selection 与 prediction 必须复现 H65，nested prefix 满足
   `prefix256 ⊂ prefix384 ⊂ prefix512`，无 GT 泄漏、统一 padding 或物理时间错位。
2. 是否存在 oracle headroom：controller holdout 上 equal-budget oracle 相对 fixed K384 的
   `ΔAvg-mAP ≥ +0.8` 个百分点且 `ΔmAP@0.7 ≥ +1.0` 个百分点。
3. 是否可预测：两个边际目标的 Spearman 相关系数均至少 `0.25`，符号准确率至少 `60%`，learned
   policy 恢复至少 `40%` 的 oracle Avg-mAP 增益，并实际产生 K256/K512 窗口且总预算误差为零。

只有全部通过才允许一次官方 test 与 10,000 次整视频配对自助法。落在 Pro 报告中未给出唯一动作的
门槛灰区时，证据保持未决并返回 Pro，不由 Codex自行放宽或重新定义阈值。

## 当前主张边界

现在可以记录的是：H65 不是纯 Top-K；当前 96-anchor Coverage 干预没有实现预注册中间目标；
下一项决定性问题是冻结 H65 下跨窗口边际计算价值是否存在并可被 Scout 预测。现在仍不能声称动态预算
有效、DUCA 优于 dense、已经降低真实成本，或 utility controller 已正确训练。

## 2026-08-31：短窗口合同修订与实现身份

Pro 对首个可运行候选做了第二次独立裁决并返回 **REVISE**。本轮不是路线转换，而是冻结短窗口和真实
执行口径：`c_i(K)=min(V_i,K)`；当非基线请求与 K384 的实际 observation 数相同时，该请求折叠为
K384，直接复用 K384 的 loss、prediction、位置和执行，不进行第二次 detector forward。历史 K384
始终执行 384 个 slots；只有实际不同的非基线预算按 16 个 observation 的 packet 向上取整，末包
padding 少于 16。没有正向的精确预算转移时，确定性回退为全 K384。

本轮 Pro 输入明确绑定公开代码：

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-budget-v1-20260830>
- Pro 审阅的首个候选：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/e45dda787a6880da4cbde0b6436ffd2a2b9df218>
- 吸收修订后的权威实现：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/be5bb8033c0b11c628394d268c1923ab398c04ed>

完整 Pro 报告保存在
`.cvpr-pro-lab/pro-reviews/runs/duca-marginal-short-window-contract-v002/visible-report.md`。权威提交
`be5bb8033c0b11c628394d268c1923ab398c04ed` 已推送；独立只读 Critic 对短窗口 accounting、折叠复用、
真实 packetized 执行、masked utility targets、K384 parity 和 PRE_RUN 证据闭环给出通过。N16R4 同提交
首个运行前检查 Slurm Job `1262073` 在零秒内退出：`sbatch --wrap` 使用 `/bin/sh`，无法执行 Bash
内建命令 `source`，因此模型入口从未运行。同一 clean commit、参数、数据与输出根只把批处理入口改为
带 Bash shebang 的脚本后，重提为 Job `1262075`。该作业进入 Linux 测试后发现测试把冻结的每视频
changed-window 上限从 `floor(0.5N)` 放宽成了 `N`，因此在多解情况下预期了非最优分配。Pro 原文、
配置、runner 和分配器均使用 `0.5`；最小修复只把测试恢复为 `0.5`，形成最新公开 commit
[`f87555f7da362fe1a20d4ca08f7a68c975ed8280`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f87555f7da362fe1a20d4ca08f7a68c975ed8280)。
该 commit 已部署至 `/data/run01/sczc063/yuzibo/duca_marginal_f87555f7_20260831`，PRE_RUN Job `1262076`
使用输出根 `/data/run01/sczc063/yuzibo/duca_marginal_prerun_f87555f7_20260831`。唯一条件后继 Job
`1262077` 绑定同一 commit 和输出根：前置终态后先读取 `pre_run_receipt.json`，只有状态为
`PRE_RUN_PASS` 才调用既有 `--stage all`，否则立即退出。在 PRE_RUN 通过并实际执行 probe 前，没有
counterfactual probe、headroom、可预测性、mAP 或成本结果。

### PRE_RUN 终态

Job `1262076` 在 11 分 16 秒后以 `COMPLETED 0:0` 结束并写出 `PRE_RUN_PASS`。收据绑定 clean commit
`f87555f7da362fe1a20d4ca08f7a68c975ed8280`、epoch-59 `state_dict_ema` 和既定 checkpoint/data/config
身份。它覆盖全部 200 个训练侧视频与 720 个窗口，160/40 video split 完整且互斥；短窗口保留，47 个
collapsed arm 正确别名 K384；逐窗口 K384 完整 384-slot tensor 相等，每视频实际 K384 成本与
`sum_i min(V_i,384)` 目标完全一致。真实冻结 forward 覆盖 historical short K384、explicit full K384、
K256-256 slots 以及 K512 的 400/448/464/480/496/512 slots。收据同时确认没有 detector/Scout 梯度、
utility-head fitting、detector training、official evaluator 或 official-test 访问。

这只证明冻结 probe 的实现与运行准入合同通过，不构成 headroom、可预测性、mAP 或效率证据。唯一条件
后继仍是 Job `1262077`，不得另行重复提交。

## 2026-08-31：冻结反事实终态与灰区

Job `1262077` 完成了 `select-k384`、`counterfactual-k256` 和 `counterfactual-k512`，分别封存了
selection 与两种反事实的 JSONL 产物和 receipt。它随后在 `summarize` 的首次官方训练侧 holdout 评估前
退出：数据划分工具写出的 block-list 是换行文本，而 OpenTAD mAP evaluator 把该路径按 JSON 读取。
这是汇总输入格式绑定错误；三个冻结 producer 已完成，但当时没有 `probe_result.json`，因此不能把该作业
写成 headroom 或机制结果。

最新权威提交为
[`f67d96fdf68a295eaa7f678f3dfc125530828889`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f67d96fdf68a295eaa7f678f3dfc125530828889)。
它只把同一换行 block-list 确定性序列化成 evaluator 接受的 JSON 数组，并加入 focused regression；没有
修改 `dynamic_budget.py`、`counterfactual_utility.py`、模型 forward、selection、预测、损失、数据、NMS、
指标或科学门槛。独立 Critic 通过，N16R4 clean snapshot 的 11 项聚焦测试通过。恢复 Job `1262098`
仅在当前提交下重新执行 PRE_RUN 身份核验与 `summarize`，没有重跑三个 producer 或训练 detector，最终
以 `COMPLETED 0:0` 结束。

恢复 receipt 绑定 clean `f67d96fd...`、同一 epoch-59 `state_dict_ema` checkpoint、同一 config SHA、
annotation、类别映射、训练视频与 VideoMAE 预训练权重。三个 producer artifact 保留其原始 `f87555f7...`
来源；这两个提交之间唯一生产代码差异是上述 summary block-list 适配，producer 依赖的配置与所有模型/
数据哈希一致。该跨提交来源必须在 Pro 材料与后续证据说明中保留，不能被改写成单提交重算。

40 个训练侧 utility holdout 视频、124 个窗口上的固定 K384 诊断为 Avg-mAP `88.131197%`、mAP@0.7
`76.270583%`；使用真实反事实效用的等预算 oracle 为 `88.856786%` 和 `76.999587%`。差值分别为
`+0.725589` 与 `+0.729004` 个百分点。oracle 分配 102 个 K384、11 个 K256 和 11 个 K512 窗口，逐视频
及总体实际 observation 预算误差均为零。

预注册强 headroom 门要求 `+0.8/+1.0` 个百分点，本次未通过；预定义无 headroom 边界为低于
`+0.3/+0.5`，本次也未落入。因此终态是 **灰区，返回 Pro**。按冻结代码，只有强 headroom 通过才训练
utility head，所以本轮没有 utility predictor、predictability 指标、learned allocation、official test、
配对区间或端到端成本结论。`K=320` 也未运行，因为奇数窗口视频仅用 K256/K384 无法保证精确均值且没有
预先冻结的补充规则。现在不得由 Codex 放宽阈值、补训 predictor、运行 official test 或自行选择路线。

## 2026-08-31：Pro 对灰区的修订与唯一后继

Fresh exact-Project Pro 完整摄取最新 GitHub 代码、原始 JSON、三个 producer receipts 与失败日志后给出
`REVISE`。Pro 接受双阶段来源：K256/K384/K512 producer 保持 `f87555f7...` 来源，最终汇总保持
`f67d96fd...` 来源；二者之间只有 block-list 格式适配，不需要重跑 producer。`+0.725589/+0.729004`
个百分点只准入为训练侧 40-video holdout 的机制诊断，不是论文主结果、官方 validation/test 结果或显著性
结论。

唯一后继是一次只读的 50% 改变窗口上限释放心证：在完全相同的密封 K256/K384/K512 产物、真实反事实
效用、逐视频实际 observation 总预算、NMS 和 evaluator 下，只把 `max_changed_fraction` 从 `0.5` 改为
`1.0`。不得新增预算档位、detector forward、utility-head 训练或 official-test 访问。若点估计未同时达到
`ΔAvg-mAP >= +0.8` 与 `ΔmAP@0.7 >= +1.0` 个百分点，则停止当前 Marginal-v1 机制；只有两项点门均
通过时才执行 seed 3407、10,000 次整视频配对 bootstrap，并且仍需返回 Pro 后才能讨论 predictor。

实现位于公开分支
[`feature/duca-marginal-cap-release-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-falsifier-v1-20260831)，
精确提交为
[`d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f)。
提交只修改 probe runner 与聚焦测试；默认 `0.5` 路径保持不变，新入口独立写入
`oracle_cap_release_result.json`。N16R4 的 14 项聚焦测试通过，独立 Critic 返回 PASS。唯一 Evaluator
Job `1262117` 已提交；在它终态前没有新的 mAP、区间或机制结论。

## 2026-08-31：cap-release 终态与预注册停止条件

唯一 Evaluator Job `1262117` 已以 `COMPLETED 0:0` 结束。它在 CPU 上只读复用原有密封产物，没有执行
detector/Scout forward、模型训练或 official test。原始固定 K384 与 50% 上限 oracle 点值被精确复现，
误差均为 `0.0` 个百分点。解除改变窗口上限后，oracle 分配由 K256/K384/K512 的 `11/102/11`
变为 `17/90/17`，改变 `11` 个视频、`34` 个窗口，实际 observation 总成本仍为 `47110`，预算误差为零。

解除上限后的 Avg-mAP/mAP@0.7 为 `88.558507%/76.720863%`，相对固定 K384 仅提高
`+0.427310/+0.450280` 个百分点，并且比原 50% 上限 oracle 低 `-0.298279/-0.278724` 个百分点。
两项点估计都没有达到预注册强门 `+0.8/+1.0`，因此按冻结规则没有运行配对 bootstrap，并停止当前
Marginal-v1 机制。这个结果只否定当前三档真实效用重分配机制在该训练侧 holdout 上具有足够强的 oracle
headroom；它不等价于否定所有动态预算方法，也不是 official validation/test 或论文性能结论。下一项科学
任务必须由 Pro 在摄取本终态和最新公开代码后独立裁决。

## 2026-08-31：Pro 终止加性 Marginal-v1，并冻结联合 mAP 邻域诊断

Fresh exact-Project Pro 在读取最新公开分支、精确提交 `d2fad7c0...`、runner、allocator、聚焦测试永久链接
以及全部密封产物后给出 `PIVOT`。Pro 终止的是“三档预算 + 独立窗口反事实损失相加 + 每视频等总
observation 重分配”这一具体机制，不终止 DUCA 的任务感知动态计算方向。直接证据是：解除 cap 扩大了
allocator 的可行集并提高其内部加性目标，却使最终 mAP 下降，因此独立窗口损失不是视频级滑窗预测经
合并、Soft-NMS、排序和 AP 聚合后的充分加性效用。粗预算档位与成对等成本转移可能放大该错配；可利用
空间本身很小仍是未排除的替代解释。

唯一当前任务是 **cap-release 差分邻域的联合 mAP 穷举诊断**。它只在 `d2fad7c0...` 的 runner 与聚焦
测试内增加只读 stage；`dynamic_budget.py` 必须逐字不变。分析从 capped 与 released allocation 自动导出
12 个差分窗口、6 组等成本转移，并对每视频所有合法平衡子集做笛卡尔积；当前密封输入应由数据推导出
96 个唯一状态。每个状态必须保持逐视频实际 observation 成本和全局 `47110` 不变，只复用原预测并运行
相同 NMS/评估器，不执行 forward、训练、official test 或 bootstrap。

继续门保持 `ΔAvg-mAP >= +0.8 pp` 且 `ΔmAP@0.7 >= +1.0 pp`。若 96 个状态均未同时通过，则停止“用
视频级联合效用修复本次 cap-release 差分”的路线；若存在通过状态，也只证明该开发集局部 action space
存在 metric-level headroom，结果仍须返回 Pro，不能自动训练 predictor。完整终态报告保存在
`.cvpr-pro-lab/pro-reviews/runs/duca-marginal-cap-release-terminal-v001/visible-report.md`。

## 2026-08-31：联合 mAP 邻域诊断终态

联合邻域实现已推送到公开仓库：

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831>
- 精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d>
- runner：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tools/bata/run_duca_marginal_frozen_h65_probe.py>
- 未修改的 allocator：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/opentad/models/duca/dynamic_budget.py>
- 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tests/test_duca_marginal_budget.py>

实现从 sealed capped/released allocation 和逐窗口实际 observation 成本自动推导出 5 个差分视频、12 个差分
窗口、6 个净转移组、8 个最小合法转移以及 96 个唯一联合状态。对于存在多种合法配对的
`video_validation_0000419`，实现保留全部等成本分解，没有预先指定某一种配对。`dynamic_budget.py` 相对父提交
保持逐字不变。N16R4 上 16 项聚焦测试、23 项既有回归测试和独立 Critic 均通过。

唯一 Evaluator Job `1262121` 以 `COMPLETED 0:0` 结束，完成 96 次 CPU evaluator 调用。集群只有要求申请
GPU 的分区，因此作业申请了 1 张调度占位 GPU，但通过清空 `CUDA_VISIBLE_DEVICES` 和固定 `--device cpu`
确保计算只在 CPU 上执行。stderr 只有与本诊断无关的 `requests` 依赖版本警告。作业没有执行 detector/Scout
forward、训练、utility-head fitting、梯度、official test 或 bootstrap。fixed/capped/released 三个参考结果的
复现误差均为 `0.0` 个百分点，所有联合状态保持逐视频预算和全局实际 observation 成本 `47110`。

没有状态同时达到冻结门 `ΔAvg-mAP >= +0.8 pp` 与 `ΔmAP@0.7 >= +1.0 pp`。按联合门最优的
`state_014` 为 `+0.553972/+0.933234` 个百分点；Avg-mAP 最优的 `state_020` 为
`+0.732990/+0.479291`；mAP@0.7 最优的 `state_001` 为 `+0.548669/+0.933539`。8 个最小合法转移中
没有一个同时改善 Avg-mAP 与 mAP@0.7，也没有观察到满足冻结定义的交互反转见证，因此诊断把当前失败
归为以单项误排序为主，而不是以窗口组合交互为主。

按照预注册规则，使用视频级联合效用修复本次 cap-release 差分邻域的路线停止。这个结论仍只是同一训练侧
40-video holdout 上的 metric-oracle 机制诊断：96 个状态中的最优状态不可部署、不可作为确认性结果，也不能
在选择后补做 bootstrap。它没有否定更广义的任务感知动态计算。原始结果保存在
`.cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-neighborhood-46812fac-job1262121/oracle_cap_release_neighborhood_result.json`，
SHA-256 为 `a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`。下一项科学问题和唯一可证伪任务
必须由 Pro 在摄取本终态及以上最新 GitHub 永久链接后独立裁决。

## 2026-08-31：Pro 最终裁决 STOP

精确 DUCA Project 的新对话完整绑定 nonce
`DUCA-MARGINAL-CAP-RELEASE-NEIGHBORHOOD-TERMINAL-ADJUDICATION-v001-20260831`、浏览器 `Pro` 模型、最新
GitHub 分支与精确提交 `46812fac...`。完整报告保存于
`.cvpr-pro-lab/pro-reviews/runs/duca-marginal-cap-release-neighborhood-terminal-compact-v002/visible-report.md`。
本轮传输没有挂载原始 JSON 字节文件，但提示词完整给出终态事实、路径和 SHA；Coordinator 随后独立重算并
确认 JSON SHA-256 为 `a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`。

Pro 裁决为 `STOP`：停止 capped→released 联合邻域修复，并终结现有加性 DUCA-Marginal-v1。直接支持的
科学解释不是内部状态名“single-item misranking”，而是：窗口级加性反事实检测损失不足以排序视频级联合
检测效用；失败已经出现在最小等成本预算转移层面，没有一个最小转移能同时改善 Avg-mAP 与 mAP@0.7，
所以不存在一组各自联合有益、仅在组合后被负交互破坏的转移来挽救该邻域。

Pro 明确没有另一项值得在这一冻结机制内执行的低成本实验。Builder、Critic、Evaluator PRE_RUN 和正式运行
全部为空；不得重跑 96 状态、改变门槛、挑选折中状态、更换配对、补 bootstrap、训练 utility head 或访问
official test。分支 `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831` 标记为只读负证据。

该 `STOP` 不否定三档预算本身、H65 priority sequence 或任务感知动态计算的一般问题。未来任何后继必须是
Pro 独立提出的新机制假设与新任务，不能作为 Marginal-v1 的小修。现有负结果只进入失败机制分析或补充材料，
不得表述为动态预算总体无效、统计显著的负总体效应、official validation/test 结果或可部署 oracle。

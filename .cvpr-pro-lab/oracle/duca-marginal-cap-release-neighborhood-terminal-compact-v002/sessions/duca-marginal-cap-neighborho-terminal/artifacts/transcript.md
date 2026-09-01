# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a94bbaf-b2f8-83ea-81bc-5c0b6b23bdb5

## Prompt

Read every attached file completely and answer the authoritative prompt as one independent scientific adjudication. Treat the GitHub repository, branch, exact commit 46812facc8773d9b4a9c21833cbe397c8aaa5a2d, runner, allocator and test permalinks in the prompt as the latest code truth. Preserve nonce DUCA-MARGINAL-CAP-RELEASE-NEIGHBORHOOD-TERMINAL-ADJUDICATION-v001-20260831 verbatim in the response.

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_MARGINAL_CAP_RELEASE_NEIGHBORHOOD_TERMINAL_ADJUDICATION-v001.md
Lines: 1-123
```md
  1 | # DUCA capped→released 联合 mAP 邻域终态：请独立裁决下一项科学任务
  2 | 
  3 | **Nonce：`DUCA-MARGINAL-CAP-RELEASE-NEIGHBORHOOD-TERMINAL-ADJUDICATION-v001-20260831`**
  4 | 
  5 | 你是本项目的独立科学负责人、路线设计者和论文主张审查者。Codex 只负责在你冻结的任务内完成最小实现、
  6 | 独立代码审查、正式评估和证据回传。请不要接受 Codex 预设的替代路线；基于本轮完整代码与原始终态，独立
  7 | 给出且只给出一个 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`，并决定是否还有一项值得执行的可证伪任务。
  8 | 
  9 | ## 1. 最新公开代码真值
 10 | 
 11 | 本轮讨论必须以以下已经推送的 GitHub 实现为代码真值：
 12 | 
 13 | - 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
 14 | - 分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-falsifier-v1-20260831>
 15 | - 精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d>
 16 | - 联合邻域 runner：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tools/bata/run_duca_marginal_frozen_h65_probe.py>
 17 | - 保持不变的 allocator：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/opentad/models/duca/dynamic_budget.py>
 18 | - 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tests/test_duca_marginal_budget.py>
 19 | 
 20 | 精确提交的父提交是 `d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`。本轮只修改 runner 与聚焦测试；
 21 | `opentad/models/duca/dynamic_budget.py` 相对父提交的 Git blob SHA-1 保持
 22 | `268c26cf41ae8a0d33c5a1b849ebff2adf0b388e`，没有修改 allocator、模型、预测、数据、损失、NMS、评估器
 23 | 或门槛。附件包含完整终态 JSON；runner、allocator 和聚焦测试由上面的精确提交永久 GitHub 链接绑定。
 24 | 请完整读取附件并审阅链接中的最新代码，不以项目中更旧 Source 或旧聊天覆盖它们。
 25 | 
 26 | ## 2. 上一轮由你冻结的问题与停止规则
 27 | 
 28 | 此前 cap release 扩大了三档 K256/K384/K512 allocator 的可行集，并改善其独立窗口加性损失目标，却使
 29 | 最终 mAP 下降。你据此停止当前加性 Marginal-v1，并冻结唯一后继：只在 capped 与 released allocation
 30 | 不同的窗口上枚举所有逐视频等实际 observation 成本的联合状态，以判断视频级联合预测集合效用是否仍能
 31 | 修复这一本次差分。
 32 | 
 33 | 冻结规则是：
 34 | 
 35 | 1. 从密封分配与逐窗口实际成本自动导出差分，不硬编码配对；
 36 | 2. 当前数据应得到 12 个差分窗口、6 个净转移组和 96 个唯一联合状态；
 37 | 3. 所有状态保持每个视频的实际 observation 总成本和全局成本 `47110`；
 38 | 4. 复用相同密封预测、Soft-NMS 和评估器；不执行模型前向、训练、official test 或 bootstrap；
 39 | 5. 只有至少一个状态同时达到相对 Fixed-H65-384 的 `ΔAvg-mAP >= +0.8 pp` 与
 40 |    `ΔmAP@0.7 >= +1.0 pp` 才允许继续，而且仍须返回你裁决；否则停止用视频级联合效用修复本次差分邻域；
 41 | 6. 从同一开发集 96 个状态中选择出的最佳状态不可部署、不可作为确认性或论文主结果，也不得选择后补做
 42 |    bootstrap。
 43 | 
 44 | ## 3. 实现、审查和执行事实
 45 | 
 46 | - 实现自动导出 5 个差分视频、12 个差分窗口、6 个净转移组、8 个最小合法转移和 96 个唯一状态。
 47 | - `video_validation_0000419` 存在两种完整等成本分解；实现保留四个合法最小配对和两种完整分解，没有任意
 48 |   固定一种配对。
 49 | - N16R4 exact clean snapshot 上：16 项聚焦测试通过；23 项既有回归测试通过；独立 Critic 返回
 50 |   `TERMINATOR_STATIC_PASS`。
 51 | - 唯一 Evaluator 是 Slurm Job `1262121`，终态 `COMPLETED 0:0`，共调用 CPU evaluator 96 次。
 52 | - 集群只有强制申请 GPU 的分区，所以作业申请 1 张调度占位 GPU；运行时清空 `CUDA_VISIBLE_DEVICES` 并
 53 |   固定 `--device cpu`，没有使用 GPU 计算。
 54 | - stderr 只有无关的 `requests` 依赖版本警告；没有 evaluator 失败。
 55 | - fixed、50% capped、cap-released 三个历史结果复现最大误差均为 `0.0 pp`。
 56 | - 没有 detector/Scout forward、模型训练、utility-head fitting、梯度计算、official test 或 bootstrap。
 57 | 
 58 | 原始终态：
 59 | 
 60 | - 本地：`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-neighborhood-46812fac-job1262121/oracle_cap_release_neighborhood_result.json`
 61 | - 远端：`/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/oracle_cap_release_neighborhood_result.json`
 62 | - SHA-256：`a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`
 63 | 
 64 | ## 4. 终态结果
 65 | 
 66 | 参考结果：
 67 | 
 68 | | 状态 | Avg-mAP | mAP@0.7 | 相对 Fixed-H65-384 |
 69 | |---|---:|---:|---:|
 70 | | Fixed-H65-384 | 88.131197% | 76.270583% | — |
 71 | | 50% capped oracle | 88.856786% | 76.999587% | +0.725589 / +0.729004 pp |
 72 | | cap-released oracle | 88.558507% | 76.720863% | +0.427310 / +0.450280 pp |
 73 | 
 74 | 96-state 联合邻域：
 75 | 
 76 | - 通过状态数量：`0/96`；
 77 | - 按联合门 margin 最优 `state_014`：Avg-mAP `88.685169%`，mAP@0.7 `77.203817%`，相对 fixed
 78 |   `+0.553972/+0.933234 pp`；联合门 margin 为 `-0.246028 pp`；
 79 | - Avg-mAP 最优 `state_020`：`88.864187%/76.749874%`，相对 fixed `+0.732990/+0.479291 pp`；
 80 | - mAP@0.7 最优 `state_001`：`88.679866%/77.204123%`，相对 fixed `+0.548669/+0.933539 pp`；
 81 | - 8 个最小合法转移中，没有一个同时提高 Avg-mAP 和 mAP@0.7；
 82 | - 按上一轮冻结定义，`interaction_witness_count=0`，root-cause classification 为
 83 |   `single-item misranking primary`；
 84 | - 终态状态为 `JOINT_NEIGHBORHOOD_GATE_FAILED_STOP_DIFFERENCE_REPAIR`，没有运行 bootstrap。
 85 | 
 86 | 最小转移的相对 capped 变化（Avg-mAP / mAP@0.7，百分点）为：
 87 | 
 88 | 1. video 0059：`-0.176920 / +0.204536`；
 89 | 2. video 0206：`+0.004503 / -0.249650`；
 90 | 3. video 0267：`-0.106811 / -0.151798`；
 91 | 4. video 0419 的四种合法最小配对：分别为
 92 |    `-0.029264/-0.014718`、`-0.028038/-0.013730`、`-0.006230/-0.015282`、
 93 |    `-0.006529/-0.019337`；
 94 | 5. video 0905：`-0.062820 / -0.132893`。
 95 | 
 96 | ## 5. 请独立回答的科学问题
 97 | 
 98 | 请不要因为需要“给下一项任务”而强行延续路线。先判断本轮证据是否足以停止更大的研究族；若不足，再选择
 99 | 唯一最有信息增益的后继。必须明确区分直接证据、最合理解释和仍未排除的替代解释。
100 | 
101 | 1. 本轮是否忠实执行了你冻结的联合邻域任务？是否存在会改变结论的实现、预算、评估或选择偏差？
102 | 2. `0/96` 联合门通过、Avg 最优与 @0.7 最优明显分离、且所有最小转移都不能同时改善两项指标，最直接支持
103 |    什么失败机制？`single-item misranking primary` 的冻结分类应如何转写成论文可理解的科学结论？
104 | 3. 这是否只停止“本次 capped→released 差分邻域的联合效用修复”，还是足以停止更广的三档预算、H65
105 |    priority sequence、或任务感知动态计算路线？请准确划定停止边界。
106 | 4. 在不重跑已停止诊断、不对 96 状态事后调参、不访问 official test、不扩大工程系统的前提下，是否还存在
107 |    一项能真正改变论文结论的低成本实验？若没有，请明确 `STOP` 并说明论文应如何保存这组负结果。
108 | 5. 若选择继续、修订或转向，请只冻结一项当前任务，并给出：
109 |    - 一句话科学问题与机制；
110 |    - 为什么它不是对已失败思路的换名重试；
111 |    - 最小 Builder 修改表面和禁止修改内容；
112 |    - 最便宜的决定性 falsifier、数据划分、公平比较、指标、阈值和停止规则；
113 |    - 独立 Critic 要核验的少数实质风险；
114 |    - Evaluator 的 PRE_RUN 与唯一正式运行；
115 |    - 可发表主张边界、失败后保留的负结果；
116 |    - `next_owner / next_action / dependency / absolute_deadline`。
117 | 
118 | ## 6. 输出要求
119 | 
120 | 输出第一行必须是且只能是 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP` 之一。随后必须原样写出 nonce。
121 | 先给终态科学解释与停止边界，再给唯一下一任务（若有）。不要提供多路线菜单、超参数网格、泛化工作流平台、
122 | 复杂合同代码或多个并行实验。不要把本轮开发集 oracle 结果写成部署效果、官方验证/测试结果、统计显著性或
123 | 论文主性能。Pro 对科学路线负责；Codex 不应替你预选机制。
```

## Answer

STOP
DUCA-MARGINAL-CAP-RELEASE-NEIGHBORHOOD-TERMINAL-ADJUDICATION-v001-20260831

## 终态科学裁决

**停止当前 capped→released 联合邻域修复，并据此终结现有加性 DUCA-Marginal-v1。不存在另一项值得在这一冻结机制内执行的科学实验。**

这不是停止整个 DUCA 项目，也不是否定三档预算、H65 时序优先序列或任务感知动态计算的一般可能性。它停止的是一个已经被完整检验的具体主张：

> 由逐窗口 K256/K384/K512 反事实检测损失构成的加性效用，在保持逐视频实际 observation 成本不变时，能够通过 cap release 或 capped→released 邻域内的联合预算重组，找到同时改善 Avg-mAP 与高 tIoU 定位的配置。

该主张现已达到终局负证据。

## 1. 实现忠实度与证据有效性

### 1.1 代码身份正确

精确提交 `46812facc8773d9b4a9c21833cbe397c8aaa5a2d` 相对父提交 `d2fad7c0dfc4a5efe98b10b9eee4723c6805699f` 只有一个提交差异，并且只修改：

* `tools/bata/run_duca_marginal_frozen_h65_probe.py`
* `tests/test_duca_marginal_budget.py`

allocator、模型、预测生成、损失、NMS 和评估器均未修改。GitHub 的精确比较与提交内容一致。 ([GitHub][1])

allocator 仍然以逐视频动态规划最大化分离后的窗口级预测效用，要求总实际 observation 成本严格等于全 K384 基线成本；`max_changed_fraction` 只限制允许改变的窗口数量，释放该上限不会改变效用函数、预算层级或成本口径。短窗口折叠、实际 observation 成本和 packet 化执行槽位也仍被明确区分。

### 1.2 联合邻域忠实于冻结任务

runner 没有硬编码六个预算转移配对。它先从 capped 和 released 分配的真实差分自动取得窗口，再逐视频枚举每个差分窗口采用 capped 或 released 状态的所有组合，只保留满足该视频原始实际成本的状态，最后对各视频状态做笛卡尔积并再次检查全局成本。它还要求：

* 不得改变非差分窗口；
* 每个视频成本分别守恒；
* 全局成本保持不变；
* capped 与完整 released 状态都必须存在；
* 每个联合状态必须能够由不重叠的最小合法转移分解。

`video_validation_0000419` 的四种最小配对和两种完整分解因此都被保留，而不是任意选定其中一种。

代码中的 `5 个视频、12 个窗口、6 个净转移组、96 个状态、47110 总成本` 是**推导完成后的失败即停形状核验**，不是生成状态的硬编码模板；如果密封输入推导出不同结构，程序会拒绝运行，而不会把数据强行改成预期结构。随后每个状态均通过同一个 `_official_holdout_metrics` 入口重新组合密封预测、执行相同 Soft-NMS 和评估，共 96 次。

新增测试明确覆盖了：

* 5 个差分视频和 12 个窗口；
* `2×2×2×2×6=96` 个唯一状态；
* 8 个最小合法转移；
* 0419 的四种配对与两种完整分解；
* 每个状态严格保持成本；
* 新结果文件不得覆盖旧 probe 与 cap-release 结果。

结合 16 项聚焦测试、23 项既有回归测试、独立 Critic 的静态通过、三组历史指标零误差复现和作业中的 96 次真实 evaluator 调用，我没有发现会改变本轮有限邻域结论的实现、成本或评估缺陷。

### 1.3 存在的限制不是结论逆转因素

第一，96 个状态是在同一开发 holdout 上评估和排序的，因此 `state_014`、`state_020` 或 `state_001` 都不能被部署，也不能作为确认性结果。这种事后选择原本会偏向产生虚假的正结果；但本轮在这种有利于“找到好状态”的条件下仍然是 `0/96`，所以它不解释门槛失败。

第二，runner 对已存在的终态结果采用幂等返回。不过作业记录显示实际调用 evaluator 96 次，说明正式运行没有通过已有结果文件跳过枚举。

第三，没有 bootstrap 是冻结规则的正确执行。bootstrap 可以估计点差的不确定性，却不能把未达到预注册实用效应门槛的点估计变成达到门槛的效应；更不能对从同一 96 状态中事后选出的最佳状态提供确认性推断。

第四，本会话实际挂载的是九个 Markdown 附件，没有出现终态 JSON 的原始字节文件。因此我没有在本地重新计算 `a8020892…` 的 SHA-256，也没有逐条重算 96 个 JSON 记录；这部分数值按权威提示词和绑定的正式运行事实接纳。终态归档必须保留原 JSON，不能只保留本裁决摘要。

九个附件之间没有形成影响本裁决的事实冲突：旧代码库存只具有索引意义，旧提示词、2026-08-28 状态和历史版本注册表不得覆盖当前精确提交；项目规则、角色规则和论文优先流程则要求把点估计、统计结论、代码通过和科学主张严格分开，并在没有新信息增益时终止重复实验。PJST-D1 的旧证据只提供这种证据边界原则，不参与本次 Marginal-v1 效果归因。        

## 2. 直接证据支持的失败机制

### 直接观察

1. **联合门为 `0/96`。**
   最接近门槛的 `state_014` 只有 `+0.553972 pp` Avg-mAP 和 `+0.933234 pp` mAP@0.7。它距离 Avg-mAP 门槛仍差 `0.246028 pp`，不是一个只差数值噪声的联合通过状态。

2. **两个指标的最优状态明显分离。**
   `state_020` 给出最高 Avg-mAP，但只有 `+0.479291 pp` 的 mAP@0.7；`state_001` 给出最高 mAP@0.7，但只有 `+0.548669 pp` 的 Avg-mAP。不存在一个状态继承两者优势。

3. **所有八个最小合法转移都不是联合有益转移。**

   * video 0059：牺牲 Avg-mAP 换取 mAP@0.7；
   * video 0206：Avg-mAP 几乎不变而 mAP@0.7 明显下降；
   * video 0267、0419 的四种配对和 video 0905：两项指标均下降。

4. **释放 cap 改善了 allocator 的加性损失目标，却使最终 mAP 由 capped 的 `+0.725589/+0.729004 pp` 降为 `+0.427310/+0.450280 pp`。**
   因此，问题不再是 capped 可行集过窄；扩大可行集反而让加性目标选择了对最终检测指标更差的预算转移。

### 最合理的科学解释

冻结状态码 `single-item misranking primary` 不应直接进入论文。更准确、可理解的表述是：

> **窗口级加性反事实检测损失不是视频级联合检测效用的充分排序统计量。失败已经出现在最小等成本预算转移层面：没有一个最小转移能够同时改善平均检测性能和高 tIoU 定位，因此不存在一组“各自联合有益、只是组合后被交互破坏”的转移来挽救该邻域。**

这里的基本单位其实是“最小等成本转移”，通常包含一个降档窗口与一个升档窗口，而不是字面上的单个窗口。因此论文不应使用“single-item”这种容易误导的内部术语。

机制上，窗口级 detector loss 是可加的局部代理，而最终 mAP 是经过预测合并、类别排序、Soft-NMS 和全数据集精确率—召回率计算后的非可加指标。一个降低局部损失的预算移动，可能改变重复预测、置信度排序或边界定位，使 Avg-mAP 与 mAP@0.7 呈现相反方向。当前结果直接支持的是**效用目标与最终联合指标失配**，而不是 cap 约束不足。

代码中的 `interaction_witness_count=0` 也必须谨慎解释：其定义只在至少两个最小转移分别对两项指标都为正时，才检查它们的组合是否反转或退化；本轮根本不存在这样的转移，所以 witness 为零在逻辑上是必然的。它不能证明 Soft-NMS、AP 或窗口组合完全没有交互，只能说明“由若干已经联合有益的原子转移组成、却被交互破坏”的特定解释不成立。

## 3. 精确停止边界

### 本轮证据足以停止

1. **capped→released 差分邻域的联合效用修复。**
2. **现有加性 DUCA-Marginal-v1：**

   * 同一密封 H65 priority sequence；
   * K256/K384/K512 三档；
   * 逐窗口反事实 detector loss 作为效用；
   * 逐视频等实际成本加性分配；
   * 通过改变 `max_changed_fraction`、配对、组合或 tie-break 寻找收益。
3. 在同一 96 状态上继续：

   * 改联合门；
   * 挑另一个折中状态；
   * 更换配对；
   * 重新排序；
   * 对事后最佳状态补 bootstrap；
   * 访问 official test；
   * 训练 utility head；
   * 再次运行同一 evaluator。

此前 cap release 已经停止当前加性机制，只留下联合邻域作为唯一可能的差分修复。本轮又穷尽并否定了该修复，所以现有加性 Marginal-v1 到此终结，而不仅是“暂时调低 cap”。

### 本轮证据不足以停止

1. **K256/K384/K512 三档本身。**
   它们只是计算动作空间；失败的是当前效用与分配依据，而不是三个预算数字的存在。

2. **H65 priority sequence 本身。**
   本轮固定并复用了该序列，没有与另一种合法优先序列做匹配比较，因而不能把失败归因于 H65 的排序质量。

3. **任务感知动态计算的一般研究问题。**
   本轮没有检验类别感知、proposal 集合感知、边界风险感知或直接以视频级集合效用为目标的其他机制，也没有检验其他独立数据或 detector。

4. **所有可能的三档预算分配。**
   96 状态是 capped 与 released 两个分配之间、逐差分窗口二选一所形成的完整邻域，不是全部窗口的 `3^W` 全空间。因此可以停止这条已冻结的修复路线，但不能发表“所有三档动态预算都不可能有效”的普遍结论。

不过，这些更宽的方向目前都只是**未被否定**，并不等于获得继续授权或正面证据。未来若重新研究，必须作为新的机制假设和新的 Pro 科学任务启动，不能把它描述成 Marginal-v1 的小修。

## 4. 是否还有一项能改变结论的低成本实验

**没有。**

原因不是缺少实验想象力，而是当前决策变量已被穷尽：

* 96 个冻结合法状态已经全部评估；
* 联合门没有通过状态；
* 最优 Avg-mAP 与最优 mAP@0.7 无法合一；
* 没有一个最小转移同时改善两项指标；
* capped 与 released 两个端点也都未达到门槛；
* bootstrap 不会改变实用效应门槛失败；
* 对同一开发集继续选择或拟合只会增加事后偏差；
* official test 被正确禁止；
* 任何真正不同的效用目标、优先序列或训练方法都已是新机制，而不是本轮低成本 falsifier。

因此：

* **Builder：无新修改。**
* **Critic：无新代码复审。**
* **Evaluator PRE_RUN：无。**
* **唯一正式运行：无。**
* **GPU/CPU 作业：均不再提交。**

## 5. 论文应如何保存这组负结果

这组结果适合进入论文的“失败机制分析”或补充材料，而不是主性能表中的候选方法成绩。应完整保存：

* 精确 commit、父提交和 allocator blob；
* Slurm Job `1262121`；
* 原始 JSON 路径与 SHA-256；
* 预注册门槛；
* 5 个差分视频、12 个差分窗口、6 个净转移组、8 个最小转移和 96 个状态；
* 每个状态的实际 observation 总成本 `47110`；
* Fixed、capped、released 三个零误差复现锚点；
* `0/96`、三个最佳状态以及全部最小转移结果；
* 没有训练、模型前向、official test 或 bootstrap 的事实。

论文可写的结论是：

> 在一个密封的开发 holdout 上，我们穷尽了 capped 与 released 三档等成本分配之间的全部 96 个合法联合状态。虽然释放分配上限改善了窗口级加性反事实损失目标，但没有状态同时达到预注册的 Avg-mAP 与 mAP@0.7 实用增益门槛；所有最小预算转移也都未同时改善两项指标。这表明窗口级加性 detector-loss 效用不足以指导该邻域内的视频级动态预算分配。

论文不可声称：

* 动态预算在总体上无效；
* H65 优先序列无效；
* 三档预算设计无效；
* 效应具有统计显著的负总体结论；
* 最佳 96-state oracle 是可部署方法；
* 本结果来自 official validation 或 official test；
* `interaction_witness_count=0` 证明不存在任何预测或 NMS 交互。

## 6. 唯一终结动作

`next_owner`: **Coordinator**

`next_action`: 在不改代码、不重跑 evaluator 的前提下，完成一次终态归档：把精确 commit、父提交、allocator blob、Job、JSON SHA、预注册门槛、`0/96`、三类最优状态、八个最小转移、停止边界和上述论文表述写入对应实验页、`decision_history.md` 与 append-only `log.md`；将当前分支标记为只读负证据；不得把内部分类 `single-item misranking primary` 原样写入论文。

`dependency`: 取得并验证原始 `oracle_cap_release_neighborhood_result.json`，确认其 SHA-256 为 `a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`；除此之外无科学或算力依赖。

`absolute_deadline`: **2026-09-01T18:00:00+08:00**

完成该归档后，本任务关闭，不再返回 Marginal-v1 做恢复性实验。

[1]: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d"

# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a9501ec-3cc4-83ea-ba60-b8302e6e2632

## Prompt

Read every provided file completely and answer the authoritative prompt as one independent scientific adjudication. Treat the repository, actual remote branch, exact commit 33e4ed137c33eef07f0452b44506a6993bdf7535, runner, test and unchanged allocator permalinks in the prompt as the latest code truth. Do not choose among routes proposed by Codex; make the project-level decision yourself. Preserve nonce DUCA-WHOLE-VIDEO-TERMINAL-ADJUDICATION-v001-20260831 verbatim in the response.

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_WHOLE_VIDEO_TERMINAL_ADJUDICATION-v001.md
Lines: 1-147
```md
  1 | # DUCA whole-video consistent-budget terminal scientific adjudication
  2 | 
  3 | Nonce: `DUCA-WHOLE-VIDEO-TERMINAL-ADJUDICATION-v001-20260831`
  4 | 
  5 | ## Your role
  6 | 
  7 | Act as the continuing scientific head and primary research owner of DUCA. Independently own the scientific question,
  8 | mechanism, falsifiable prediction, evidence interpretation, claim scope, failure diagnosis and research direction. Treat
  9 | this message as an evidence handoff, not a request to ratify Codex. Do not delegate the scientific choice back to Codex or
 10 | the human when the evidence permits a decision. Codex is the implementation and evaluation executor after you freeze one
 11 | task.
 12 | 
 13 | Use ordinary research language. Do not create workflow status codes, engineering contracts, proof systems or a new
 14 | coordination framework. Do not assume that any alternatives mentioned in prior discussions are exhaustive or endorsed.
 15 | 
 16 | ## Latest public implementation — authoritative code truth
 17 | 
 18 | The current implementation has been pushed and independently verified against its upstream branch. Use these permanent
 19 | GitHub links rather than local paths, historical branches or Project memory:
 20 | 
 21 | - Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
 22 | - Actual remote branch:
 23 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>
 24 | - Exact clean commit:
 25 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
 26 | - Frozen evaluator runner:
 27 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>
 28 | - Focused regression test:
 29 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py>
 30 | - Unchanged three-tier budget implementation:
 31 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>
 32 | 
 33 | The local clean worktree, its upstream branch and the remote branch all resolve to
 34 | `33e4ed137c33eef07f0452b44506a6993bdf7535`. The commit preserves the sealed producer proposal-row order; it does not
 35 | change predictions, candidate definitions, observation costs, Soft-NMS, evaluator, budget tiers or the frozen gate.
 36 | Twenty-eight focused tests and an independent Critic passed on this exact commit.
 37 | 
 38 | ## Paper question and negative history
 39 | 
 40 | DUCA asks whether low-cost semantic evidence can reduce real high-resolution VideoMAE computation for offline temporal
 41 | action detection while protecting high-tIoU localization. The current clean H65 30+60 reference reaches 65.13% Avg-mAP
 42 | and 43.31% mAP@0.7 on official THUMOS14 validation; the shared dense AdaTAD reproduction reaches 68.73% Avg-mAP. These
 43 | numbers are context only and are not directly comparable with the training-side controller holdout used below.
 44 | 
 45 | Material results preceding the present terminal experiment are:
 46 | 
 47 | 1. At fixed K384, the task-state native-tubelet coreset reached 62.81% Avg-mAP versus 64.13% for its matched uniform
 48 |    control, so fine-grained coreset score tuning stopped.
 49 | 2. Temporal facility-location Coverage-v1 failed its pre-registered unlabeled intervention gate before training; it
 50 |    provided no mAP result.
 51 | 3. With the H65 detector and K256/K384/K512 sealed counterfactual predictions frozen, the 50%-capped marginal-utility
 52 |    oracle improved the training-side holdout by only `+0.726/+0.729` percentage points in Avg-mAP/mAP@0.7. Releasing the
 53 |    cap reduced this to `+0.427/+0.450`.
 54 | 4. Exhaustive evaluation of the 96 capped-to-released joint states found no state passing the frozen `+0.8/+1.0` gate;
 55 |    the best joint-gate state was only `+0.554/+0.933`. You previously stopped that additive window-level Marginal-v1.
 56 | 5. You then made one project-level PIVOT: test whether the decision unit was wrong by giving all windows of one donor
 57 |    video K256, all windows of one recipient video K512, and all other videos K384. You froze the consequence that zero
 58 |    passing candidates would stop DUCA method innovation within the current THUMOS14/H65/three-tier/resource boundary.
 59 | 
 60 | ## Frozen whole-video experiment
 61 | 
 62 | - Data: 40 training-side controller-holdout videos, 124 overlapping windows.
 63 | - Fixed baseline: every window requests K384.
 64 | - Candidate: for every ordered pair of distinct videos, every donor window requests K256, every recipient window requests
 65 |   K512 and every other window requests K384.
 66 | - A legal candidate requires actual non-baseline execution in both changed videos and total actual observation cost no
 67 |   greater than the fixed baseline cost `47110`. Short windows retain `min(valid_observations, K)` cost accounting.
 68 | - Candidate identities were completely generated and sealed before labels, ground truth or metrics were read.
 69 | - Evaluation reused the same sealed predictions, physical-coordinate reconstruction, Soft-NMS, annotation, class map and
 70 |   evaluator. It ran no detector or Scout forward, training, gradient, bootstrap, official validation or official test.
 71 | - Frozen gate: at least `+0.8` percentage points in Avg-mAP and `+1.0` percentage points in mAP@0.7 simultaneously, at
 72 |   actual cost no greater than `47110`.
 73 | 
 74 | Corrected PRE_RUN Job `1262161` passed on the exact public implementation: 40 videos, 124 windows, 1560 ordered pairs,
 75 | 704 legal candidates, 1330 candidates with any actual intervention, and `0.0` percentage-point reproduction error for
 76 | the fixed, capped and released anchors. PRE_RUN receipt SHA-256 is
 77 | `734b178bfb7bdaa05879edfeb8e129263c9e2c4cf80867415eec6d41df3c12a3`; candidate-manifest SHA-256 is
 78 | `c4a02c47be1ab7e73dc81c18b32635d3347ece2f0d26b0d96de3ec4af053f69a`.
 79 | 
 80 | Formal Job `1262162` stopped after 500 candidates only because Slurm node `g0022` went down; it produced no terminal
 81 | result and is not scientific evidence. The one exact same-task infrastructure recovery, Job `1262190`, reused the same
 82 | clean snapshot, sealed manifest, predictions, evaluator and gate. It completed `704/704` candidates with Slurm
 83 | `COMPLETED 0:0`; no third job was run.
 84 | 
 85 | ## Authoritative terminal result
 86 | 
 87 | Full terminal artifact:
 88 | `/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_consistent_budget_result.json`
 89 | 
 90 | SHA-256: `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`
 91 | 
 92 | The fixed K384 anchor on this training-side holdout is:
 93 | 
 94 | - Avg-mAP `88.131197%`
 95 | - mAP@0.3/0.4/0.5/0.6/0.7:
 96 |   `95.833793/93.684529/89.581223/85.285857/76.270583%`
 97 | - actual observation cost `47110`
 98 | - reproduction error `0.0` percentage points for every reported metric
 99 | 
100 | All 704 legal candidates completed, with 705 evaluator calls including the fixed anchor. No candidate passed both gates.
101 | The passing-candidate count is zero.
102 | 
103 | The three most informative extrema are:
104 | 
105 | 1. Best Avg-mAP candidate, `video_validation_0000158=>video_validation_0000173`:
106 |    - delta Avg-mAP/mAP@0.7: `+0.694215/-0.043632` percentage points
107 |    - actual cost `46982`
108 | 2. Best mAP@0.7 candidate, `video_validation_0000490=>video_validation_0000173`:
109 |    - delta Avg-mAP/mAP@0.7: `-0.235922/+0.496998` percentage points
110 |    - actual cost `46854`
111 | 3. Best candidate under the pre-registered joint-gate ordering,
112 |    `video_validation_0000419=>video_validation_0000173`:
113 |    - delta Avg-mAP/mAP@0.7: `+0.147383/+0.489786` percentage points
114 |    - actual cost `45830`
115 |    - joint-gate margin `-0.652617` percentage points
116 | 
117 | The result explicitly records: no training, detector/Scout forward, gradients, bootstrap, official validation or official
118 | test; `paper_claim_allowed=false`, `deployable_policy_claim_allowed=false`, and development-holdout oracle selection.
119 | There is no uncertainty interval. The result is a deterministic, exhaustive diagnosis of the frozen development action
120 | space, not a population-level performance estimate or deployable policy.
121 | 
122 | ## Required independent adjudication
123 | 
124 | First verify whether the implementation and evidence are faithful enough to apply the pre-registered consequence. Then
125 | make your own single scientific decision: continue, narrow the claim, revise the scientific question, pivot to a genuinely
126 | new mechanism, stop, or escalate only a real human authority/resource boundary.
127 | 
128 | Your response must do all of the following:
129 | 
130 | 1. State whether the zero-pass result triggers the previously frozen stop boundary, and define exactly what is stopped
131 |    without overgeneralizing to all dynamic computation, all low-cost Scouts or all budget spaces.
132 | 2. Diagnose the mechanism-level failure before considering any successor. Separate facts supported by the exhaustive
133 |    action-space result from competing explanations that remain untested, and identify the strongest alternative
134 |    explanation.
135 | 3. State the strongest honest paper claim and the claims that remain prohibited. Address whether this negative sequence
136 |    is scientifically publishable as a result, supplementary analysis or only internal evidence.
137 | 4. Decide independently whether a scientifically distinct paper question with material expected information gain remains.
138 |    Do not revive Marginal-v1, extend this candidate search, tune the gate, or present a larger search as a new mechanism.
139 | 5. Issue exactly one current task if continued work is justified. Freeze its falsifiable prediction, control, data split,
140 |    metrics, compute/fairness treatment, uncertainty requirement, stop rule, minimal implementation surface, role order and
141 |    an absolute expected-return deadline. If no task is justified, explicitly state that no Builder/Critic/Evaluator work
142 |    remains and what evidence would be required before reopening research.
143 | 
144 | End with `next_owner`, `next_action`, `dependency`, and `expected_return_at`. Stop after the one decision and one task. Do
145 | not provide a menu of routes or ask Codex to choose between them.
146 | 
147 | Repeat nonce `DUCA-WHOLE-VIDEO-TERMINAL-ADJUDICATION-v001-20260831` verbatim in the final response.
```

### File: PAPER_PROGRESS.md
Lines: 1-122
```md
  1 | # DUCA 论文缩略报告
  2 | 
  3 | - 更新日期：2026-08-31
  4 | - 名称说明：DUCA 是本项目沿用的方法路线名称。
  5 | - 当前结论：固定 `K=384` 的任务状态时序 coreset 低于匹配均匀选择，`DUCA-Coverage-v1` 也未通过预注册中间机制门。冻结 H65 的三档边际预算诊断随后表明：50% 改变窗口上限的真实效用 oracle 相对固定 K384 为 `+0.726/+0.729` 个百分点；解除上限后反而降为 `+0.427/+0.450`，未达到 `+0.8/+1.0` 强门。capped→released 96-state 联合 mAP 邻域也没有通过状态，最佳联合门状态仅为 `+0.554/+0.933`。Pro 随后只允许一次整视频一致预算、跨视频一降一升的开发集证伪；704 个合法候选现已全部完成，没有候选同时达到 `+0.8/+1.0`。最佳 Avg-mAP 候选为 `+0.694/-0.044`，最佳 mAP@0.7 候选为 `-0.236/+0.497`。因此此前冻结的当前 THUMOS14、H65 优先序列、三档真实 observation 动作空间和资源范围内的项目级停止条件已经触发；结果仍需返回 Pro 归档停止范围并独立决定是否存在新的科学问题。
  6 | 
  7 | ## 1. 论文问题与应用价值
  8 | 
  9 | 离线时序动作检测（Temporal Action Detection, TAD）通常对长视频进行密集的重型视频编码。DUCA 研究一个更节省计算的问题：能否先用低成本模型预测逐时刻动作性和边界重要性，再由确定性规则选择少量高分辨率帧，并进一步为不同视频或窗口分配不同预算，在真实端到端计算下降时保持高时间交并比（temporal Intersection over Union, tIoU）下的边界定位性能。
 10 | 
 11 | 这个问题的应用价值在于：如果低成本语义证据能够可靠地替代大量冗余帧，就可以减少 VideoMAE 等视频骨干网络的实际输入，而不必修改下游检测头或官方评估器。
 12 | 
 13 | ## 2. 当前科学路线
 14 | 
 15 | 长期路线包含两个层次，二者不能混写：
 16 | 
 17 | 1. **语义间接选帧。** 低成本侦察模型学习二元动作性与动作起止边界；确定性采集规则根据这些预测产生有序、非均匀的原始帧位置。侦察模型不直接把帧索引当作主要学习目标。
 18 | 2. **动态预算。** 长期论文主张要求根据逐视频或逐窗口的语义证据决定保留帧数，使重型 VideoMAE 路径真实执行不同工作量。固定 `K=384` 仅用于机制归因、公平对照和回退。
 19 | 
 20 | 已完成的归因实验把 768 帧输入组成 384 个 VideoMAE 原生两帧 tubelet，并固定选择其中 192 个。对照臂在原生 tubelet 网格上均匀选择；候选臂使用冻结侦察器的动作性、边界强度和时序新颖性进行确定性选择。该候选在所有报告阈值上均低于均匀选择，现已终止。
 21 | 
 22 | 当前路线不再继续修改固定 K 内的时间覆盖。`DUCA-Marginal-v1` 把研究问题改为：在冻结 H65 Scout 与检测器时，能否预测一个窗口从 K384 降到 K256 的损失代价，以及从 K384 升到 K512 的收益，并在同一视频严格不变的真实 observation 预算 `sum_i min(V_i,384)` 下跨窗口转移 VideoMAE 计算。K384 保持 H65 的预算校准系统采样；K256/K512 必须来自同一非连续 H65 priority sequence 的嵌套集合。相同实际成本的短窗口请求折叠并复用 K384；只有实际不同的非基线预算按 16-observation packet 执行。第一步只是冻结检测器的反事实机制实验，不是完整训练后的论文主模型。
 23 | 
 24 | ## 3. 与官方基线的真实差异
 25 | 
 26 | 共享的未修改 AdaTAD 基线使用官方代码 revision `01c58b9f2370e914150cf94d392208a4e211c053`、seed 42、60 个训练轮次和官方评估器，平均检测精度（Avg-mAP）为 `68.73`；论文公开锚点为 `69.03`。DUCA 只读引用这一共享复现，不重复训练官方 dense 模型。
 27 | 
 28 | DUCA 与官方 dense 模型的目标差异只应来自输入采样与预算：前者先以低成本语义模型选帧，再让 VideoMAE 只处理被选中的高分辨率帧。下游 ActionFormer 检测头、损失、NMS 和官方评估器原则上不变。历史 65.xx 或 66.xx 结果不属于官方 dense 复现，不能代替 `68.73`，也不能在协议不匹配时直接计算方法增益。
 29 | 
 30 | ## 4. 已完成实现与代码身份
 31 | 
 32 | - H65 干净复现使用历史 ASFormer 语义预测、确定性非均匀逐帧选择和固定 `K=384`。当前可审计的 30+60 训练参考冻结于源码 revision `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。
 33 | - PJST-D1 是当前物理时间表示候选的代码名称。它不改变选择器，只在 VideoMAE 首次二帧 tubelet 混合前校正导数分量所使用的时间间隔。匹配训练冻结于 clean revision `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`；只读重推理与统计入口冻结于 `7bd120f0d342bf175c97c365fba7cbd359df055e`。
 34 | - UVT 诊断分支为 `codex/duca-uvt-utility-value-20260819@df544c78ce515d925dc7019f106fce09a53c09f8`；Fovea/Query-Bridge 诊断分支为 `codex/duca-fovea-query-bridge-20260819@4ae5067100c4490c7110c00a1ad406230ba603cd`。它们与 H65/PJST 不是同提交匹配比较，不能用跨版本差值归因。
 35 | - 物理连续片段路线已完成真实训练，但性能明显下降，现作为负结果保存，不再是当前论文主线。
 36 | - 为支持下一次科学路线与代码主线裁决，当前模型、配置、启动器、测试和研究记忆已同步到 `codex/duca-research-sync-20260828`；代码库存固定提交为 `5136011ed57df8a639427a633a488a592ba95924`。该快照用于检查实现重叠与历史分支，不是经过运行前检查的实验候选，也不能替代各实验的 clean revision。
 37 | - 先前的动态窗口预算候选冻结于 `codex/duca-semantic-budget-matched-20260828@36d75c146492a38eb8966c66ff6b2881938cf3c6`。它在取得效能结果前被后续科学裁决暂缓，保留为下一阶段的实现参考，不能作为动态预算证据。
 38 | - 当前原生 tubelet 时序 coreset 候选位于独立干净分支 `codex/duca-native-tubelet-coreset-20260828`，冻结提交为 `b33391126eac05e3353d322b973dda91741f0732`。实现包含固定 192 个原生 tubelet、任务状态驱动选择、端点/空洞覆盖、低分辨率上下文回收、物理时间重建、严格匹配的均匀对照、配置、聚焦测试与 N16R4 启动器。N16R4 环境的 20 项相关测试通过，独立只读审查未发现阻断性缺陷；两臂均完成 60 轮训练和官方验证，但结构化证据保存失败。
 39 | - 窗口级动态预算候选位于独立干净分支 `codex/duca-dynamic-native-tubelet-budget-20260829`，冻结提交为 `d127c2b2ceea7ff8a6932aa4a1925e1ff86cf610`。实现按视频生成需求排序和 16/20/24-clip 预算，在各预算内执行确定性均匀 tubelet 选择，并按真实 clip 数分组调用 VideoMAE；较短预算不会在重型骨干前补齐到 24。短窗口若不能容纳分配预算会明确退出。Python 编译、启动器语法、纯启动器测试和独立静态审查已通过；尚无运行前检查或正式实验结果。
 40 | - 当前 `DUCA-Coverage-v1` 候选位于独立干净分支 `feature/duca-coverage-only-v1-20260829`，当前提交为 `048143124e2a36a76575200ae17d6f42ec79ea3a`，基于 H65 正式提交 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。实现新增固定预算设施位置选择器，并提供 matched H65 对照、真实训练样本无标签重放门、60 轮配置、恢复合同和 N16R4 启动器。修正后的 PRE_RUN Job `1261679` 已执行 27 项测试和 200 个真实 training 样本重放，但因预注册覆盖/空洞干预条件未满足而在 smoke 前停止。代码已同步 GitHub并部署到 N16R4 干净目录 `/data/run01/sczc063/yuzibo/duca_coverage_v1_04814312_20260830`。
 41 | - 最新 Pro 裁决指定从 H65 clean revision `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 建立 `feature/duca-marginal-budget-v1-20260830`。吸收短窗口修订后的实现为 `be5bb8033c0b11c628394d268c1923ab398c04ed`；测试合同修复为 `f87555f7da362fe1a20d4ca08f7a68c975ed8280`。该提交完成 K384、K256 和 K512 冻结反事实产物后，汇总器因把换行文本 block-list 当作 JSON 读取而退出。最新提交 [`f67d96fdf68a295eaa7f678f3dfc125530828889`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f67d96fdf68a295eaa7f678f3dfc125530828889) 只为官方评估器生成确定性的 JSON block-list 适配文件并加入回归测试；没有修改模型、分配器、预测、损失、数据、NMS、指标或科学门槛。恢复作业没有重跑三个冻结 producer 阶段，只在当前 clean commit 上重做运行前身份核验并汇总既有产物。producer 产物保留其 `f87555f7...` 来源，汇总与最终结果绑定 `f67d96fd...`；两者的配置、checkpoint、annotation、类别映射、预训练权重及其哈希一致，这一跨提交来源会在后续 Pro 材料中明确披露。
 42 | - Pro 对灰区的后继实现位于 [`feature/duca-marginal-cap-release-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-falsifier-v1-20260831)，精确提交为 [`d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f)。它只增加独立的 `max_changed_fraction=1.0` 只读汇总入口和聚焦测试，默认 `0.5` 汇总路径、三档 producer、模型、数据、NMS 与门槛均不变。N16R4 的 14 项聚焦测试和独立 Critic 已通过。
 43 | - 最新的联合邻域诊断实现位于 [`feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831)，精确提交为 [`46812facc8773d9b4a9c21833cbe397c8aaa5a2d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d)。它只修改 probe runner 与聚焦测试；`dynamic_budget.py` 相对父提交逐字不变。实现从密封分配和真实 observation 成本自动导出 8 个最小合法转移、6 个净转移组及 96 个唯一联合状态，没有为多解视频硬编码配对。N16R4 上 16 项聚焦测试、23 项既有回归测试和独立 Critic 均通过。
 44 | - 项目级 Pro 在读取上述最新 GitHub 对象后裁决 `PIVOT`。唯一新任务位于 [`feature/duca-whole-video-consistent-budget-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831)，当前权威公开提交为 [`33e4ed137c33eef07f0452b44506a6993bdf7535`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535)。父提交 `c27d77...` 的首次 PRE_RUN `1262147` 暴露密封 proposal 被额外排序、从而改变 Soft-NMS 并破坏锚点复现的确定性证据错误；`33e4ed...` 只恢复密封 producer 原始顺序并增加回归测试，不改变候选、成本、预测、NMS、评估器或三档预算。28 项聚焦测试与独立 Critic 通过。修正后 PRE_RUN `1262161` 已 `COMPLETED 0:0`：40 个视频、124 个窗口、固定成本 `47110`、1560 个有序对中 704 个合法候选，三个既有锚点复现误差均为 `0.0` 个百分点。正式 Job `1262162` 因节点下线而中断；完全相同的唯一恢复 Job `1262190` 随后完成全部 704 个候选。终态 JSON SHA-256 为 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`，通过候选数为零。
 45 | 
 46 | 项目根目录承担多路线协调，工作区可能包含尚未归档的修改。论文实验身份必须引用上述独立 clean revision，而不能用项目根当前 `HEAD` 代替。
 47 | 
 48 | ## 5. 正式实验协议
 49 | 
 50 | 正式比较使用 THUMOS14 规范数据入口、官方训练与验证划分、同一检测器、损失、NMS、评估器和预先登记的模型选择规则。当前已有实验主要使用 seed `3407`；单种子结果不能支持稳定性或显著性结论。
 51 | 
 52 | 当前与后续实验矩阵按以下顺序组织：
 53 | 
 54 | 1. 只读引用共享的官方 dense AdaTAD 结果；
 55 | 2. H65 固定 `K=384` 的 30+60 参考训练；
 56 | 3. PJST-D1 固定/重放选择结果的物理时间表示 OFF/ON 配对已完成点估计，但平均点差为负且配对区间缺失；它不再是当前优先开发项；
 57 | 4. 固定 `K=384` 的原生 tubelet 均匀选择与任务状态驱动时序 coreset 已完成；细粒度 coreset 候选因一致的负向点估计终止；
 58 | 5. 同一 H65 代码基座上的 matched allocation control 与 `DUCA-Coverage-v1` 已完成真实训练样本无标签重放门；该门未通过，因此两个 60 轮完整训练臂没有启动。当前对照实现是预算校准系统采样，而冻结设计曾概括为 Top-K；在 Pro 裁决该基线身份和失败机制前不得重提正式训练。
 59 | 6. 冻结 H65 的 K256/K384/K512 反事实边际预算实验、cap-release 诊断和 capped→released 差分邻域的 96 个联合状态枚举均已完成。50% 上限 oracle 的 Avg-mAP/mAP@0.7 增益为 `+0.726/+0.729` 个百分点；解除上限后反而降为 `+0.427/+0.450`。96-state 中没有状态同时达到 `+0.8/+1.0`；当前加性 Marginal-v1 及其本次视频级联合效用修复均按冻结规则停止。
 60 | 7. 整视频一致预算的跨视频单次转移 oracle 已完成：donor 视频所有窗口请求 K256，recipient 视频所有窗口请求 K512，其余视频保持 K384；候选总实际 observation 成本不超过固定 K384 的 `47110`。候选集合在读取标签或指标前完整生成，随后仅复用密封预测和相同评估器。1560 个有序对中 704 个合法候选完成评估，没有候选通过联合门；没有执行模型前向、训练、bootstrap 或 official test。
 61 | 
 62 | 主要指标为 tIoU 0.3、0.4、0.5、0.6、0.7 下的 mAP，以及五个阈值的平均值；同时报告短动作、边界定位和完整端到端计算成本。任何计算节省主张都必须来自实际执行的 VideoMAE 工作量和相同硬件条件下的测量，不能由 padding 后的名义帧数推断。
 63 | 
 64 | 完整训练至少每 5 个训练轮次保存一次可恢复的 PyTorch `.pth` 检查点；如果未修改的官方配置保存更频繁，则保留官方间隔。恢复包应包含模型、指数移动平均模型、优化器、学习率调度器、混合精度缩放器、训练轮次或更新计数及随机状态，并至少保留最近三个有效恢复点、预定义里程碑和最终检查点。最终结果使用预先登记的最终模型或最终指数移动平均模型，不能根据中间验证结果事后挑选。
 65 | 
 66 | ## 6. 已取得的性能结果
 67 | 
 68 | 除特别说明外，Avg-mAP 是 tIoU 0.3 至 0.7 五个阈值下 mAP 的平均值。
 69 | 
 70 | | 实验 | Avg-mAP | mAP@0.7 | 证据边界 |
 71 | |---|---:|---:|---|
 72 | | 共享官方 dense AdaTAD | 68.73 | 未在本报告重复摘录 | 一次共享复现；论文公开 Avg-mAP 为 69.03 |
 73 | | H65 30+60 | 65.13 | 43.31 | 单种子、完整训练；当前 H65 参考 |
 74 | | H65 20+40 | 62.46 | 39.94 | 同 seed 的训练压缩诊断；比 30+60 低 2.66/3.37 点 |
 75 | | H65 30+30，AM-RPCH25 | 63.22 | 41.25 | 只改变第二阶段学习率日程 |
 76 | | H65 30+30，LongCosine-H6000 | 63.56 | 41.01 | 更慢衰减仍未恢复 30+60 |
 77 | | RankPack K384 | 61.57 | 37.10 | 单种子物理时间表示对照 |
 78 | | TrueTime K384 | 62.19 | 37.89 | 相对 RankPack 为 +0.62/+0.79 点；尚无配对区间 |
 79 | | PJST-D1 OFF | 65.063 | 43.646 | 211/211 视频，冻结 H65 选择结果 |
 80 | | PJST-D1 ON | 64.591 | 43.769 | 211/211 视频；相对 OFF 为 -0.472/+0.123 点 |
 81 | | 原生 tubelet uniform K384 | 64.13 | 42.45 | 单 seed；60 轮训练和官方验证完成；结构化指标文件未封存 |
 82 | | 原生 tubelet coreset K384 | 62.81 | 40.56 | 单 seed；相对匹配 uniform 为 -1.32/-1.89 点；结构化指标文件未封存 |
 83 | | 连续片段 FZ | 49.89 | 29.68 | 完整单种子训练；明显负结果 |
 84 | | 连续片段 JT | 47.24 | 26.52 | 完整单种子训练；联合训练未恢复性能 |
 85 | 
 86 | PJST-D1 两臂各产生 422,000 条预测，视频集合一致，重推理逐项复现了原始点估计。预先登记的 10,000 次整视频配对自助法在任何抽样前退出，因为统计程序指向了错误的预测文件目录。因此当前只有点估计，没有置信区间；不能据此宣布总体效应显著为负，也不能把 mAP@0.7 的 `+0.123` 点解释为真实收益。
 87 | 
 88 | UVT 的 legacy/geometry/geometry+EMA 三臂 Avg-mAP 为 `57.35/55.93/55.92`。Fovea/Query-Bridge 第一波中最佳 `query_cycle` 为 `54.67`。这些结果来自不同代码版本与实验合同，只能说明相应首版设计未恢复 H65 性能，不能用来量化某一个组件的因果影响。
 89 | 
 90 | 历史 `65.3857` 是 H65 语义间接非均匀逐帧选择的 30+60 诊断结果；`65.696` 来自改变物理检测网格的探索实现。后者同时改变了检测器时间几何，因此不是与官方原生检测器严格匹配的输入采样对照。当前干净 H65 复现 `65.13` 是更适合作为后续同代码、同训练协议比较的参考。
 91 | 
 92 | 当前还没有与主路线匹配、可用于论文的完整端到端成本结果，也没有动态预算保持性能或提高效率的实验证据。`DUCA-Coverage-v1` 已完成代码实现、独立审查和真实 training 数据运行前门；该门给出不满足预注册干预条件的诊断结果，但没有进入训练，因此不是 mAP 或成本结果。
 93 | 
 94 | `DUCA-Marginal-v1` 的冻结 detector 诊断在训练侧 40 个 utility holdout 视频、124 个窗口上比较了固定 K384 与使用真实反事实效用的等预算 oracle。固定臂 Avg-mAP/mAP@0.7 为 `88.131/76.271`，50% 上限 oracle 为 `88.857/77.000`。解除上限后，分配从 K256/K384/K512 的 `11/102/11` 变为 `17/90/17`，但结果降为 `88.559/76.721`；相对固定臂只有 `+0.427/+0.450` 个百分点。总 observation 预算仍精确为 `47110`。这里的百分数只描述训练侧 controller holdout，不可与 official validation/test 表直接比较。强 headroom 门未通过，因此 utility predictor、正式测试、配对区间和端到端成本均未运行。
 95 | 
 96 | 随后只在 capped 与 released 分配不同的 12 个窗口上枚举了全部 96 个逐视频等成本联合状态。最佳 Avg-mAP 状态相对固定 K384 为 `+0.733` 个百分点，但 mAP@0.7 仅 `+0.479`；最佳 mAP@0.7 状态为 `+0.549/+0.934`；按两项联合门最优的状态为 `+0.554/+0.933`。没有状态同时满足 `+0.8/+1.0`，也没有单个最小合法转移同时改善 Avg-mAP 与 mAP@0.7。该诊断没有执行模型前向、训练、official test 或 bootstrap；从 96 个开发集状态中事后选出的最优状态不能当作可部署策略或论文主结果。
 97 | 
 98 | 整视频一致预算的最终开发集 falsifier 在相同固定 K384 锚点 `88.1312%/76.2706%`（Avg-mAP/mAP@0.7）和真实成本 `47110` 下完成 704 个合法状态。Avg-mAP 最优状态的变化为 `+0.6942/-0.0436` 个百分点，mAP@0.7 最优状态为 `-0.2359/+0.4970`，联合门余量最优状态为 `+0.1474/+0.4898`；通过候选数为零。该结果只属于训练侧 controller holdout 的事后 privileged oracle 证伪，没有模型前向、训练、官方验证/测试、配对区间或可部署策略含义。
 99 | 
100 | ## 7. 结果解释与已停止的方向
101 | 
102 | - 20+40 训练压缩、AM-RPCH25 和 LongCosine-H6000 均未恢复 H65 30+60 的性能。现有证据说明简单压缩预热或只修改第二阶段学习率尾部不足以保持性能；它不否定 H65 的语义间接选帧机制。
103 | - 连续 16 帧片段采样在真实训练中造成大幅定位损失，联合训练也未恢复，因此该采样单元不再作为当前主路线。这个结果不否定低成本语义侦察或物理时间一致性的一般问题。
104 | - TrueTime 相对 RankPack 有小幅单种子提升，但证据不足以形成论文主张。
105 | - PJST-D1 的当前点估计没有显示平均性能收益；缺少配对置信区间意味着总体效应仍未完成统计裁决。统计程序的路径错误是证据生成失败，不是模型的科学失败。
106 | - UVT 与 Fovea/Query-Bridge 同时改变了选择分数、预算证据或训练信息流，且缺少与 H65 同提交的严格隔离，因此其性能下降不能归因于单一组件。
107 | 
108 | ## 8. 当前证据缺口与下一动作
109 | 
110 | THUMOS14 原始视频、注释、类别映射、VideoMAE-S 预训练权重、H65 Stage-1 侦察器检查点和共享官方 AdaTAD 结果均已核验。PJST-D1 的 OFF/ON 完整推理已经结束。
111 | 
112 | 当前固定预算归因已经得到负向点估计：在完全相同的 `K=384` 高分辨率帧预算、训练日程和检测器下，任务状态驱动 coreset 没有优于原生 tubelet 均匀选择，并在 tIoU 0.5 至 0.7 下降 `1.54/2.03/1.89` 个百分点。这个结果优先要求分析端点覆盖、最大空洞、跨 tubelet 打包、低分辨率上下文回收和选择分数是否共同损害重型表示，而不能直接把固定预算重命名为最终方法。
113 | 
114 | PJST-D1 的配对区间仍属未完成证据，但它不会改变当前路线，因此不作为当前任务补齐。原生 tubelet 两臂已从干净 H65 基座完成实现、N16R4 聚焦测试、独立审查和 60 轮训练。uniform `1260184` 与 coreset `1260185` 都写出 epoch-59 检查点并完成官方 211 视频评估，日志点估计分别为 `64.13%` 和 `62.81%`。两臂随后因同一个证据封存错误退出：配置没有保存预测，结构化指标入口因此拒绝写出 `metrics_epoch59_ema.json`。当前没有配对区间或成本结果；训练成功、日志点估计和结构化证据缺失必须分别陈述。
115 | 
116 | Pro 冻结的 96-state 联合 mAP 邻域诊断已经在公开提交 `46812fac...` 上完成。原 fixed/capped/released 三结果复现误差为 `0.0` 个百分点，96 次评估保持逐视频预算和全局成本 `47110`；没有状态通过 `+0.8/+1.0` 联合门，也没有单个最小合法转移同时改善两个门指标。Pro 已据此最终裁决 `STOP`：现有加性 Marginal-v1 及本次邻域修复关闭，不再重跑、改门、补 bootstrap、训练 utility head 或访问 official test。当前分支只作为负证据读取；未来若重新研究动态计算，必须由 Pro 以新的机制假设和独立任务启动，不能作为 Marginal-v1 的恢复。
117 | 
118 | 新的项目级 Pro 对话随后独立裁决 `PIVOT`：逐窗口混合预算与最终视频级 proposal 集合的决策单位可能错位。它只冻结一次整视频双向转移 oracle，以检查“视频内统一预算、视频间转移计算”是否存在足够开发集 headroom。该实验现已完成，704 个合法候选中没有候选在不超过 `47110` 实际 observation 成本时同时达到 `+0.8` Avg-mAP 和 `+1.0` mAP@0.7，因而触发此前冻结的当前边界内项目级停止条件。下一项动作不是调整本实验或由 Codex 选择路线，而是把终态 JSON、全部边界和最新已推送仓库、实际远端分支、精确提交 `33e4ed...`、runner、测试及未修改 allocator 的永久 GitHub 链接返回 Pro，由 Pro 独立归档停止范围并判断是否存在新的项目级科学问题。
119 | 
120 | ## 9. 可发表性边界
121 | 
122 | 目前可以写入论文的事实是：H65 的 30+60 训练参考明显优于已测试的 60 轮压缩日程；连续片段采样是明确负结果；PJST-D1 的匹配点估计没有平均收益；原生 tubelet coreset 的单种子点估计比匹配均匀选择低 `1.32` 个百分点；当前 96-anchor Coverage 干预没有达到预注册中间机制目标。DUCA-Marginal 的训练侧 oracle 在 50% 上限下仅有灰区增益，解除上限后增益进一步降至 `+0.427/+0.450` 个百分点；其 96-state 联合差分邻域和最终 704-state 整视频单次转移 falsifier 均没有达到 `+0.8/+1.0` 的预注册门，因此当前动作空间内没有足够开发集 headroom。它尚未证明 Scout 能预测这种效用，也没有 official test、统计区间或端到端成本结果。不能把该停止条件扩展为所有 coverage、动态预算或物理时间方法无效，也不能声称 DUCA 有效、优于 dense，或已经获得性能—成本联合优势。
```

### File: research-wiki/experiments/duca-whole-video-consistent-budget-v1.md
Lines: 1-117
```md
  1 | ---
  2 | type: experiment
  3 | status: tested
  4 | updated: 2026-08-31
  5 | project: DUCA
  6 | ---
  7 | 
  8 | # DUCA 整视频一致预算的跨视频转移 oracle
  9 | 
 10 | ## 科学问题
 11 | 
 12 | 在全局实际 observation 成本不高于固定 K384 时，如果同一个视频的全部重叠窗口使用同一预算档位，只在
 13 | 不同视频之间转移计算，能否避免逐窗口混合预算造成的 proposal 质量与置信度不一致，并同时改善 Avg-mAP
 14 | 与 mAP@0.7？
 15 | 
 16 | 这是 Pro 在终止加性 DUCA-Marginal-v1 后独立冻结的新机制假设。它不是旧 allocator 的恢复：旧机制在
 17 | 一个视频内部按窗口损失转移预算；新机制保持视频内预算一致，直接枚举一次跨视频 donor–recipient 转移。
 18 | 
 19 | ## 代码基座与最新公开真值
 20 | 
 21 | - 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
 22 | - 只读父分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831>
 23 | - 父提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d>
 24 | - 实现分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>
 25 | - 精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
 26 | - 独立 runner：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>
 27 | - 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py>
 28 | - 未修改的三档预算分配器：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>
 29 | 
 30 | 新分支只允许新增独立 runner、聚焦测试和必要时的薄 CPU Slurm 入口。不得修改旧 Marginal runner、
 31 | `opentad/models/duca/dynamic_budget.py`、selector、acquisition、detector、训练器、配置或既有预测。
 32 | 
 33 | 父提交 `c27d77aa...` 只新增上述 runner 与测试。候选清单由 `sample_id`、`video_id` 和密封的
 34 | `budget_accounting.actual_cost` 生成并落盘，此后才读取终态指标、annotation 或调用评估器。新 runner
 35 | 没有调用旧 Marginal allocator。首次 PRE_RUN Job `1262147` 发现 runner 对密封 proposal 行做了额外
 36 | 字典序排序；这会在分数并列时改变既有 Soft-NMS 的确定性输入顺序，使 fixed/capped/released 锚点无法
 37 | 按 `1e-6` 个百分点复现。该问题发生在候选性能计算前，是证据重放顺序错误，不是机制负结果。
 38 | 
 39 | 最小修正提交 `33e4ed13...` 只保留密封 producer 的原始 proposal 顺序，并增加相应回归测试；没有修改
 40 | 候选定义、成本、预测值、Soft-NMS、评估器、三档预算分配器或科学门槛。28 项聚焦测试通过，全新独立
 41 | Critic 对精确干净提交返回 `PASS`。修正提交已部署为 N16R4 干净快照
 42 | `/data/run01/sczc063/yuzibo/duca_whole_video_33e4ed13_20260831`。
 43 | 
 44 | 修正后的 PRE_RUN Job `1262161` 已 `COMPLETED 0:0` 并返回 `PRE_RUN_PASS`：40 个视频、124 个窗口，
 45 | 固定 K384 实际成本 `47110`；在全部 `40×39=1560` 个有序对中有 704 个合法候选、1330 个实际改变
 46 | 预算的候选；候选生成阶段未读取标签、GT 或指标，fixed/capped/released 三个锚点复现误差均为
 47 | `0.0` 个百分点。回执与候选清单分别位于
 48 | `/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_pre_run_receipt.json`
 49 | 和同目录 `whole_video_candidate_manifest.json`，SHA-256 分别为
 50 | `734b178bfb7bdaa05879edfeb8e129263c9e2c4cf80867415eec6d41df3c12a3` 与
 51 | `c4a02c47be1ab7e73dc81c18b32635d3347ece2f0d26b0d96de3ec4af053f69a`。
 52 | 
 53 | 首次正式 Evaluator Job `1262162` 只运行冻结的 `--stage evaluate` 并复用相同密封预测、Soft-NMS 与评估器，
 54 | 但在完成 `500/704` 个候选后因计算节点 `g0022` 下线而被 Slurm 标为 `NODE_FAIL`。运行器没有异常回执，
 55 | 也没有生成终态结果，因此它不是性能证据。完全相同的唯一基础设施恢复 Job `1262190` 随后从同一干净
 56 | 快照、脚本、候选清单和输出目录完成 `704/704` 个候选，Slurm 状态为 `COMPLETED 0:0`。没有创建第三份作业。
 57 | 
 58 | ## 终态结果
 59 | 
 60 | 固定 K384 在 40 个训练侧 controller holdout 视频上的 Avg-mAP/mAP@0.7 为 `88.1312%/76.2706%`，实际
 61 | observation 成本为 `47110`。全部 704 个合法 donor-recipient 候选均完成相同 Soft-NMS 与评估器计算，
 62 | 共调用评估器 705 次（固定臂一次、候选 704 次），但没有候选同时达到预先登记的两个性能门槛：
 63 | 
 64 | - Avg-mAP 最优候选 `video_validation_0000158=>video_validation_0000173` 的变化为
 65 |   `+0.6942/-0.0436` 个百分点（Avg-mAP/mAP@0.7），实际成本 `46982`；
 66 | - mAP@0.7 最优候选 `video_validation_0000490=>video_validation_0000173` 的变化为
 67 |   `-0.2359/+0.4970` 个百分点，实际成本 `46854`；
 68 | - 按联合门余量最优的候选 `video_validation_0000419=>video_validation_0000173` 的变化为
 69 |   `+0.1474/+0.4898` 个百分点，实际成本 `45830`，离联合门最近的余量仍为 `-0.6526` 个百分点。
 70 | 
 71 | 因此，通过候选数为零。该结果触发此前由 Pro 冻结的停止条件：在当前 THUMOS14 训练侧 holdout、H65
 72 | priority sequence、K256/K384/K512 三档真实 observation 动作空间和本轮资源边界内，不再扩大搜索、组合更多
 73 | 视频、修改预算档位或降低门槛。结果必须返回 Pro，由 Pro 归档这一停止范围并独立决定项目层面的后续科学任务；
 74 | Codex 不从该负结果自行选择替代路线。
 75 | 
 76 | 终态文件为
 77 | `/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_consistent_budget_result.json`，
 78 | SHA-256 为 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`。固定臂复现误差在全部六个
 79 | 指标上均为 `0.0` 个百分点；没有生成 failure receipt。
 80 | 
 81 | ## 冻结干预
 82 | 
 83 | 只使用 40 个训练侧 controller holdout 视频、124 个窗口，以及密封 K256/K384/K512 窗口预测和真实成本：
 84 | 
 85 | 1. 全 K384 是固定基线；
 86 | 2. 对每个不同视频的有序对 `(d,u)`，donor `d` 的全部窗口请求 K256；recipient `u` 的全部窗口请求
 87 |    K512；其余视频全部请求 K384；
 88 | 3. donor 和 recipient 都必须至少有一个窗口的实际成本不同于 K384；
 89 | 4. 短窗口继续按 `min(valid_observations,K)` 计费和折叠；
 90 | 5. 只保留全局实际成本不高于 `47110` 的候选；
 91 | 6. 候选集合必须在读取标签、GT 或指标前完整生成；标签只用于随后寻找开发集特权 oracle 最优状态。
 92 | 
 93 | 最多有 `40×39=1560` 个有序候选。所有状态复用相同密封预测、坐标逆映射、Soft-NMS、annotation、类别
 94 | 映射和评估器；不执行 detector/Scout forward、训练、梯度、bootstrap 或 official test。
 95 | 
 96 | ## 通过与停止规则
 97 | 
 98 | 只有至少一个合法候选相对 fixed K384 同时满足：
 99 | 
100 | - `ΔAvg-mAP >= +0.8` 个百分点；
101 | - `ΔmAP@0.7 >= +1.0` 个百分点；
102 | - 全局实际 observation 成本 `<=47110`；
103 | 
104 | 才说明该开发集 whole-video action space 有足够 headroom，并返回 Pro 裁决可预测性。最优候选先最大化
105 | `min(ΔAvg-mAP-0.8, ΔmAP@0.7-1.0)`，再选择成本更低者，最后按 donor、recipient 视频 ID 字典序。
106 | 
107 | 若身份复现误差超过 `1e-6` 个百分点、输入身份不一致、没有合法候选或通过候选为零，则停止本 falsifier。
108 | 前述前三种确定性问题只允许一次不改变机制和门槛的最小修复。若通过候选为零，不增加第三个视频、不组合
109 | 多个 transfer、不降低门槛、不改变预算档位、不补 bootstrap、不训练控制器、不访问 official test；在当前
110 | THUMOS14、H65 priority sequence、三档真实 observation 动作空间和资源边界下项目级停止 DUCA 方法创新。
111 | 
112 | ## 证据边界
113 | 
114 | 这是从 704 个合法开发集状态中事后选择的 privileged oracle 负结果。它没有执行 detector/Scout forward、
115 | 训练、梯度、bootstrap 或 official validation/test，也没有确认性统计区间；因此不能证明可部署控制器、真实
116 | 端到端速度收益或 DUCA 优于 dense AdaTAD。它只否定当前冻结边界内的整视频单 donor-recipient 三档转移
117 | 是否具有足够的开发集联合性能 headroom，不能外推为所有动态计算、所有预算空间或所有低成本 Scout 无效。
```

## Answer

Nonce: `DUCA-WHOLE-VIDEO-TERMINAL-ADJUDICATION-v001-20260831`

# 1. 唯一科学裁决

## **STOP**

**零通过结果触发此前冻结的项目级停止边界。**

停止的准确范围是：

> 在当前 THUMOS14 训练侧 controller holdout、冻结 H65 检测器与优先序列、密封的 K256/K384/K512 窗口预测、真实 observation 成本口径和现有资源范围内，停止继续开发基于“在视频或窗口之间转移三档输入 observation 预算”的 DUCA 方法。

因此，不再进行：

* Marginal-v1 的恢复、改写或重新训练；
* capped、released、96-state 或 whole-video 候选空间的扩展；
* 增加第三个视频、多次 transfer、组合更多 donor–recipient 对；
* 改变 K256/K384/K512 档位或事后降低 `+0.8/+1.0` 门槛；
* 训练 utility predictor、controller 或其他对当前 oracle 状态进行模仿的预测头；
* 为这条路线补 bootstrap、访问 official validation/test，或把更大的开发集搜索包装成新机制。

停止边界**不**扩展为以下一般性结论：

* 所有动态计算都无效；
* 所有低成本 Scout 都无效；
* 所有预算空间都缺少 headroom；
* 所有逐视频方法都无效；
* 预算条件训练、token/layer 级动态计算或其他数据集上的动态计算都无效。

这些问题没有被本实验检验。

# 2. 实现与证据是否足以执行停止条件

## 结论：足够，并且没有发现会阻止科学裁决的实现偏差

公开远端分支当前确实指向精确提交 `33e4ed137c33eef07f0452b44506a6993bdf7535`。

该提交相对父提交只修改了两个文件：whole-video runner 和对应聚焦测试；核心修改是删除对密封 proposal 行的额外排序，恢复 producer 原始顺序，并增加顺序保持回归测试。它没有修改预测、候选定义、成本、Soft-NMS、评估器或预算机制。([GitHub][1])

公开 runner 与冻结协议一致：

1. 它从密封 K384、K256、K512 产物合并预测，并保留 K384 producer 的原始行顺序；同时核对三个产物具有相同 sample 集、40 个 holdout 视频和124个窗口。
2. 每个候选严格定义为：一个 donor 视频全部请求 K256，一个不同的 recipient 视频全部请求 K512，其余全部请求 K384。合法性要求两个被改变视频都存在真实非基线执行，并且总实际成本不超过 `47110`。
3. 它完整枚举 `40×39=1560` 个有序视频对，并在候选清单中保留全部候选及其合法性；候选生成只依赖 `sample_id`、`video_id` 和密封 actual-cost accounting。
4. runner 先生成并写出完整 candidate manifest，之后才加载终态指标、annotation 和 evaluator。聚焦测试也显式检查了这一顺序。
5. 正式阶段重新生成候选空间并与 PRE_RUN manifest 的 canonical hash 比较，重新评估 fixed K384 anchor，要求所有指标复现误差不超过 `1e-6` 个百分点，随后逐一评估全部合法候选。
6. 联合通过条件在代码中固定为 `ΔAvg-mAP ≥ +0.8`、`ΔmAP@0.7 ≥ +1.0` 且成本不超过 `47110`；零通过时直接记录 whole-video 项目级停止状态。
7. 聚焦测试检查了完整有序对、真实短窗口成本折叠、视频内统一预算、未调用旧 Marginal allocator、密封 proposal 顺序以及确定性联合排序。

`dynamic_budget.py` 在邻域提交 `46812fac…` 和当前提交 `33e4ed13…` 上具有相同 blob SHA `268c26cf41ae8a0d33c5a1b849ebff2adf0b388e`，确认分配器没有被本轮修改。

物理时间恢复和评估路径也不是由 whole-video runner 重新发明。密封 counterfactual producer 明确保存 selected-axis 到原始真实时间坐标的映射并要求预测在 NMS 前逆映射；终态 runner只是按照密封顺序组合这些已经生成的预测。  `_official_holdout_metrics` 对组合后的 proposals 使用配置中不变的 Soft-NMS，并调用同一 THUMOS14 evaluator。

因此，首次正式作业的节点故障应继续排除在科学证据之外；使用同一快照、manifest、预测和 evaluator 完成 `704/704` 的唯一恢复作业可以作为本轮终态执行。根据你指定为权威的终态 artifact、SHA、`0.0` anchor 复现误差和 `705` 次 evaluator 调用，没有证据表明零通过来自候选遗漏、成本错误、proposal 重排、评估漂移或旧 allocator 介入。

我没有在此界面重新读取集群本地 JSON 的原始字节；其 SHA 和数值按本轮明确指令作为权威终态证据接受。公开代码与所报告的结果结构完全相容。

旧的 2026-08-28 材料包也已完整读取。其 prompt、旧研究状态、库存边界、PJST 原始证据、项目规则、角色规则、通用科研 skill、历史实现注册表和 README 只用于解释历史与证据边界；其中“仍待路线裁决”的内容已被本轮 2026-08-31 终态覆盖。        

# 3. 机制层面的失败诊断

## 3.1 被证据支持的事实

这不是“预算变化完全没有影响”。实际结果显示，预算转移能够改变检测结果，但变化没有形成足够大且方向一致的联合收益：

* 最佳 Avg-mAP 候选达到 `+0.694215`，但 mAP@0.7 为 `-0.043632`；
* 最佳 mAP@0.7 候选达到 `+0.496998`，但 Avg-mAP 为 `-0.235922`；
* 联合门排序最优候选只有 `+0.147383/+0.489786`，其联合门余量仍为 `-0.652617`。

这支持三个机制结论：

**第一，逐窗口预算混合不是失败的充分解释。**
本实验已经把同一视频的全部窗口统一到一个档位，消除了“同一视频内不同窗口预算导致 proposal 分数或质量不一致”这一主要候选解释，但联合 headroom 仍未出现。

**第二，当前三档转移的主要问题不是 allocator 没有找到正确的视频，而是候选动作所产生的效用不一致。**
提升 Avg-mAP 的状态和提升高 tIoU 定位的状态不是同一状态；计算转移有时改善较宽松阈值下的总体检测，却没有同步保护精确边界，反之亦然。

**第三，当前 action space 缺少达到预登记论文价值门槛的联合效用。**
这一结论适用于已完整枚举的 whole-video 单 donor–recipient 状态，以及此前 capped、released 和 96-state 差分邻域。它不是从一个失败策略推断出来，而是 oracle 直接读取开发集指标后仍无法找到通过状态。

因此，被否定的机制命题是：

> 仅把 H65 同一嵌套优先序列上的 K256/K384/K512 observation，在窗口之间或视频之间重新分配，就足以在不增加实际 observation 成本时产生可利用的 Avg-mAP 与高 tIoU 联合 headroom。

## 3.2 仍未被检验的解释

本实验没有检验：

* 多个 donor、多个 recipient 的任意全局组合；
* K256/K384/K512 之外的预算或连续预算；
* 重新训练后的 budget-conditioned detector；
* 不以 H65 同一 priority sequence 构造的 K256/K512；
* 改变重型编码器内部 token、层数、特征刷新频率或空间分辨率的动态计算；
* 其他数据集、检测器或 Scout；
* population-level 稳定性。

但这些未知项不能成为继续扩大当前搜索的理由。尤其是多个 transfer 的更大组合空间，仍只是对同一密封预测进行更强的开发集事后优化；它会增加 oracle 选择自由度，却没有引入新的机制或可部署预测信号。

## 3.3 最强的替代解释

最强的未检验解释是：

> **当前 H65 priority sequence 与只在 K384 下训练的检测器，没有形成跨预算兼容、单调且边界敏感的表示。**

K256 是从同一优先序列中删除 observations，K512 是向同一优先序列尾部增加 observations；但检测器是在 K384 分布下训练的。因而：

* K512 新增 observations 不一定包含检测器能利用的新增边界证据；
* K256 删除的 observations 可能改变 proposal 置信度和边界回归校准；
* 不同档位产生的收益和损失未必可加，也未必在 Avg-mAP 与 mAP@0.7 上同向。

whole-video 一致预算可以消除视频内部的档位混合，却不能修复这种**表示与训练分布不兼容**。这比“allocator 还不够聪明”更符合完整负结果。

这一解释仍然是解释，不是已证实结论。验证它需要新的训练机制或新的计算动作空间；那已经越过当前冻结边界，不能作为 Marginal-v1 的补救实验。

# 4. 最强诚实论文主张

当前最强且可核验的表述是：

> 在40个 THUMOS14 训练侧 controller-holdout 视频、冻结 H65 检测器、密封的嵌套 K256/K384/K512 预测和不高于 fixed K384 实际 observation 成本的条件下，704个合法整视频单 donor–recipient 预算转移状态中，没有状态同时达到预登记的 `+0.8` Avg-mAP 和 `+1.0` mAP@0.7 headroom。结合此前 capped、released 和96-state 邻域结果，当前三档 frozen-detector 预算转移动作空间缺少预登记水平的联合开发集 headroom。

还可以进一步写明：

* whole-video 一致预算没有挽救逐窗口 Marginal-v1 的失败；
* 最优 Avg-mAP 与最优高 tIoU 状态存在明显权衡；
* 当前停止是 action-space headroom 停止，而不是 predictor 训练失败。

## 仍然禁止的主张

不得声称：

* DUCA 或动态预算普遍无效；
* Scout 无法预测预算效用；
* 当前结果具有总体显著性或 population-level 置信区间；
* 当前 oracle 是可部署策略；
* 已获得端到端速度、显存、能耗或吞吐收益；
* 已在 official validation/test 上证明负结果；
* DUCA 优于 dense AdaTAD、保持 dense 性能，或获得性能—成本联合优势；
* 任意多视频组合也不可能通过；
* budget-conditioned 训练或其他动态计算动作空间也会失败。

# 5. 可发表性裁决

该结果具有良好的**内部科学终止价值**：候选集合预先生成、有限空间被完整枚举、anchor 精确复现、停止门预先冻结，而且实现没有明显混杂。它足以结束当前研究路线。

但它目前**不能作为一篇 CVPR 方法论文的独立主要结果**，原因是：

* 只使用训练侧 development holdout；
* oracle 事后读取标签和指标；
* 没有 official validation/test；
* 没有 population uncertainty；
* 没有可部署 controller；
* 没有端到端成本测量；
* 没有跨数据集或跨检测器复现；
* H65 本身仍明显低于共享 dense 参考。

因此，本轮负结果的发表地位应定为：

> **可以作为未来其他主论文中的补充失败分析；在当前项目中主要作为内部终态证据。**

若未来要将其发展为独立的“动态时序采集局限性”论文，需要预先设计跨检测器、跨数据集和多个预注册 action space 的系统研究。那将是一个新的研究项目，不是当前 DUCA 的下一实验。

# 6. 是否还存在应立即启动的全新科学任务

## 当前没有

概念上仍可提出与当前路线不同的问题，例如在不删除 temporal observations 的情况下，对 VideoMAE 内部的 token、层或特征刷新进行条件计算。但当前证据只说明输入 observation 转移动作空间失败，并没有提供足够证据表明某个新的内部计算机制具有较高成功概率。

立即选择并实现另一个机制，会把本次明确的停止边界变成无限连续的路线救援。它需要：

* 新的科学问题；
* 新的 action space；
* 新的 matched control；
* 新的实际计算口径；
* 很可能还需要新的完整训练。

在没有便宜、预注册的机制 headroom 证据之前，其预期信息增益不足以合理化新的 Builder、Critic、Evaluator 周期。

## 重新开放研究所需的证据

未来只有在以下条件出现时，才应以一份全新的项目级科学问询重新开放：

> 一个明确位于当前 H65/K256-K384-K512 observation-transfer 边界之外的新机制，先在不使用 official validation/test 的训练侧独立开发划分上证明：其 oracle action space 在匹配实际计算下能够同时超过预先冻结的 Avg-mAP 与高 tIoU headroom 门，并且这种 headroom 不依赖事后扩张候选集合或改变门槛。

在此证据出现之前，不应让 Codex自行实现候选机制，也不应把历史 ChronoTransport、UVT、Fovea、Query-Bridge 或其他分支自动恢复为主线。

# 7. 当前任务

**没有新的科学实验任务。**

Builder、Critic 和 Evaluator 在当前 DUCA 路线上均无剩余工作：

* Builder 不再修改模型、selector、allocator、runner 或配置；
* Critic 不再为同一机制发起新的审查循环；
* Evaluator 不再运行候选、bootstrap、训练、official validation 或 official test。

唯一剩余动作是由 Coordinator 完成一次简短研究归档：

1. 将本裁决、准确停止边界、最强替代解释和论文主张边界写入 decision history、whole-video 实验页和研究日志；
2. 绑定公开分支、精确提交、终态 JSON 路径与 SHA；
3. 把 Marginal-v1、cap-release、96-state neighborhood 和 whole-video 分支标为只读负证据；
4. 不提交新的 Slurm 作业，不新建候选分支。

`next_owner`: DUCA Coordinator
`next_action`: 归档本次项目级 STOP、冻结准确停止范围并确认无新的 Builder、Critic 或 Evaluator 工作被调度
`dependency`: 无；公开代码身份、终态 artifact、候选枚举和停止门均已完成
`expected_return_at`: `2026-08-31T18:00:00+08:00`

`DUCA-WHOLE-VIDEO-TERMINAL-ADJUDICATION-v001-20260831`

[1]: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535 "Fix sealed proposal replay order · yuzbo/OpenTAD_C3_CoarseClean_20260702@33e4ed1 · GitHub"

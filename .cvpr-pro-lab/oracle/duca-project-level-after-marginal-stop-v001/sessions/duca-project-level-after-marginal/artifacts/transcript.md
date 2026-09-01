# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a94c0ae-5388-83e9-afd3-6b8f1e596e1e

## Prompt

Read every provided file completely and answer the authoritative prompt as one independent project-level scientific adjudication. Treat the repository, actual remote branch, exact commit 46812facc8773d9b4a9c21833cbe397c8aaa5a2d, runner, allocator and test permalinks in the prompt as the latest code truth. This is a fresh scientific question, not a Marginal-v1 recovery or follow-up. Preserve nonce DUCA-PROJECT-LEVEL-AFTER-MARGINAL-STOP-v001-20260831 verbatim in the response.

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_PROJECT_LEVEL_AFTER_MARGINAL_STOP-v001.md
Lines: 1-100
```md
  1 | # DUCA 项目级后继裁决：在 Marginal-v1 终止后决定继续、转向或停止
  2 | 
  3 | **Nonce：`DUCA-PROJECT-LEVEL-AFTER-MARGINAL-STOP-v001-20260831`**
  4 | 
  5 | 你是 DUCA 的独立科学负责人、机制设计者和论文主张审查者。Codex 只负责执行你冻结的最小实现、独立代码
  6 | 审查、正式实验评估和证据回传。当前没有已授权的后继机制；请不要接受 Codex 预设路线，也不要因为流程需要
  7 | “下一项任务”而强行延续研究。
  8 | 
  9 | 这是一轮新的项目级科学裁决，不是对已经终止的 DUCA-Marginal-v1 的追问、恢复或调参。请基于最新公开代码、
 10 | 附带的完整阶段报告、终态负结果记录和上一轮 Pro 终态，独立判断 DUCA 是否仍有一条值得投稿的新机制路线。
 11 | 
 12 | ## 1. 最新公开实现与 GitHub 代码真值
 13 | 
 14 | 本轮必须以以下已推送对象为最新实现，不能以本地路径、旧 Project Source 或旧聊天中的分支名覆盖：
 15 | 
 16 | - 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
 17 | - 最新只读证据分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831>
 18 | - 精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d>
 19 | - 联合邻域诊断入口：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tools/bata/run_duca_marginal_frozen_h65_probe.py>
 20 | - 当前动态预算 allocator：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/opentad/models/duca/dynamic_budget.py>
 21 | - 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tests/test_duca_marginal_budget.py>
 22 | - H65 30+60 正式基座提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>
 23 | 
 24 | `46812fac...` 的本地 clean HEAD、远端 upstream 和远端分支头已重新核对为一致。它只为终态只读诊断修改
 25 | runner 与测试；allocator、模型和训练代码没有随诊断改变。附件中的阶段报告和实验页给出所有历史代码身份与
 26 | 证据边界。
 27 | 
 28 | ## 2. 仍然有效的论文目标
 29 | 
 30 | 研究对象是离线时序动作检测（Temporal Action Detection, TAD）。长期目标是在重型 VideoMAE 路径之前，利用
 31 | 低成本动作与状态证据，为不同时间位置分配真实不同的高分辨率视觉计算；在相同或更低的真实端到端计算下，
 32 | 尽量保护或提高 THUMOS14 上的平均检测精度，尤其是高时间交并比下的边界定位。
 33 | 
 34 | 任何候选都必须满足：
 35 | 
 36 | 1. 真正改变 VideoMAE 执行的 observation/clip 数，而不是 padding 后的名义稀疏；
 37 | 2. 使用官方数据划分、注释、类别映射、检测器评估和相同 NMS；
 38 | 3. 与公平的 dense 或匹配计算基线比较，分别报告性能、真实成本和不确定性；
 39 | 4. validation/test 标签、teacher 或事后挑选不得参与分配决策；
 40 | 5. 先用最便宜的决定性 falsifier 验证机制，再考虑完整训练；
 41 | 6. 保留负结果，避免把协调系统、合同代码或防御性框架当作科研贡献。
 42 | 
 43 | ## 3. 已经取得的主要证据
 44 | 
 45 | 请完整读取附件；以下只给出项目级必要摘要：
 46 | 
 47 | - 共享官方 dense AdaTAD Avg-mAP 为 `68.73`，论文公开锚点为 `69.03`；当前 H65 30+60 单种子参考为
 48 |   Avg-mAP `65.13`、mAP@0.7 `43.31`。两者不构成同一计算预算下的完整论文比较，当前也没有可发表的
 49 |   端到端成本曲线。
 50 | - 把 H65 的 30+60 压缩为 20+40 或 30+30，并调整第二阶段学习率衰减，均未恢复性能。它终止的是已测试的
 51 |   压缩日程，不否定 H65 的低成本语义证据。
 52 | - 连续高分辨率片段采样的完整训练明显下降；联合训练没有恢复。这条采样单元已停止。
 53 | - TrueTime/PJST-D1 没有显示平均性能收益；配对区间又因统计收尾路径错误缺失，因此不能升级成总体负效应。
 54 | - 固定 K=384 原生 tubelet 的任务状态 coreset 相比匹配 uniform 下降 `1.32` Avg-mAP 和 `1.89` mAP@0.7；
 55 |   结构化预测和配对区间没有封存。细粒度 coreset 已停止。
 56 | - DUCA-Coverage-v1 在训练前机制门失败：集合变化和覆盖增益不足，最大时间空洞反而恶化；没有启动完整训练。
 57 | - DUCA-Marginal-v1 冻结 H65、检测器和 K256/K384/K512 真实反事实。50% capped 真实效用 oracle 相对固定
 58 |   K384 为 `+0.726/+0.729` 个百分点；解除 cap 后只有 `+0.427/+0.450`，未达到 `+0.8/+1.0` 门。
 59 | - 最终 96-state 逐视频等成本联合邻域中通过状态为 `0/96`，联合门最优仅 `+0.554/+0.933`；8 个最小合法
 60 |   转移没有一个同时改善 Avg-mAP 与 mAP@0.7。上一轮 Pro 因此终止现有加性 Marginal-v1，并明确该结论
 61 |   不能外推为 H65 priority sequence、三档预算或任务感知动态计算总体无效。
 62 | 
 63 | 目前没有正在运行或已授权的 DUCA Builder、Critic、PRE_RUN、训练、Evaluator 或 official-test 任务。
 64 | 
 65 | ## 4. 已停止且不得换名重试的内容
 66 | 
 67 | 不要把以下内容作为新任务：
 68 | 
 69 | - 重跑或扩展 Marginal-v1 的 96 状态，改变 `+0.8/+1.0` 门，选择折中状态，更换配对或 tie-break；
 70 | - 为事后最佳状态补 bootstrap，训练旧 utility head，继续 cap/预算档位/加性损失小修；
 71 | - 恢复 fixed-K coreset 的分数调权、Coverage-v1 的 `M/sigma/K/M` 调参或最大空洞修补；
 72 | - 重做连续片段、PJST-D1、60 轮压缩学习率扫描，或在旧负结果上追加无决定性的种子；
 73 | - 直接访问 official test 来寻找路线；
 74 | - 用新术语包装同一机制，或者设计多个并行候选菜单。
 75 | 
 76 | ## 5. 请独立完成的项目级裁决
 77 | 
 78 | 第一行必须且只能是 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`。本轮的 `STOP` 表示：在当前任务、证据和
 79 | 可接受资源边界下停止 DUCA 项目级方法创新，而不只是停止 Marginal-v1。
 80 | 
 81 | 请依次回答：
 82 | 
 83 | 1. 当前证据是否已经足以结束 DUCA 的任务感知稀疏重型计算研究？分别说明直接证据、最合理的跨路线失败
 84 |    机制，以及仍未排除但也尚无正证据的可能性。
 85 | 2. 现有结果能否形成可投稿贡献？若不能，缺少的是新机制、严格公平的计算—性能比较、统计证据，还是更根本
 86 |    的问题定义；不要把工程完成度当成科学贡献。
 87 | 3. 如果你选择 `CONTINUE`、`REVISE` 或 `PIVOT`，只提出一个真正新的、最有信息增益的机制假设。它必须解释
 88 |    为什么能够越过已经观察到的失败，而不是 Marginal-v1、Coverage、coreset、连续片段或 PJST 的换名重试。
 89 | 4. 对这唯一机制冻结：一句话科学问题、因果机制、可证伪预测、最便宜的决定性实验、公平对照、真实计算
 90 |    定义、数据划分、指标、阈值、停止规则和论文主张边界。
 91 | 5. 只给完成该 falsifier 所需的最小 Builder 代码表面、必须保持不变的组件、独立 Critic 的少数实质检查、
 92 |    Evaluator 的 PRE_RUN 与唯一正式运行。不要生成平台、工作流框架、复杂合同或超参数网格。
 93 | 6. 明确 `next_owner / next_action / dependency / absolute_deadline`。如果没有一项实验能以合理成本改变论文
 94 |    结论，请选择项目级 `STOP`，说明应如何归档已有负结果，而不是虚构下一任务。
 95 | 
 96 | ## 6. 输出约束
 97 | 
 98 | 紧接第一行原样输出 nonce。只给一个科学裁决和至多一个当前任务，不给路线菜单。明确区分开发集 oracle、
 99 | official validation/test、点估计、置信区间、运行成本和因果解释。不得把未运行的机制写成有效，也不得把当前
100 | 负结果外推到证据没有覆盖的研究族。Pro 对问题和路线负责；Codex 只能执行你明确冻结的任务。
```

### File: PAPER_PROGRESS.md
Lines: 1-116
```md
  1 | # DUCA 论文缩略报告
  2 | 
  3 | - 更新日期：2026-08-31
  4 | - 名称说明：DUCA 是本项目沿用的方法路线名称。
  5 | - 当前结论：固定 `K=384` 的任务状态时序 coreset 低于匹配均匀选择，`DUCA-Coverage-v1` 也未通过预注册中间机制门。冻结 H65 的三档边际预算诊断随后表明：50% 改变窗口上限的真实效用 oracle 相对固定 K384 为 `+0.726/+0.729` 个百分点；解除上限后反而降为 `+0.427/+0.450`，未达到 `+0.8/+1.0` 强门。capped→released 96-state 联合 mAP 邻域也没有通过状态，最佳联合门状态仅为 `+0.554/+0.933`。Pro 最终裁决 `STOP`：现有加性 DUCA-Marginal-v1 及其本次联合邻域修复获得终局负证据，不再有代码、评估或算力任务。该结果只来自训练侧密封预测的开发集 oracle 诊断，不是可部署策略、官方验证/测试结果或论文主性能结论，也不否定 H65、三档预算或任务感知动态计算的一般问题。
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
 44 | 
 45 | 项目根目录承担多路线协调，工作区可能包含尚未归档的修改。论文实验身份必须引用上述独立 clean revision，而不能用项目根当前 `HEAD` 代替。
 46 | 
 47 | ## 5. 正式实验协议
 48 | 
 49 | 正式比较使用 THUMOS14 规范数据入口、官方训练与验证划分、同一检测器、损失、NMS、评估器和预先登记的模型选择规则。当前已有实验主要使用 seed `3407`；单种子结果不能支持稳定性或显著性结论。
 50 | 
 51 | 当前与后续实验矩阵按以下顺序组织：
 52 | 
 53 | 1. 只读引用共享的官方 dense AdaTAD 结果；
 54 | 2. H65 固定 `K=384` 的 30+60 参考训练；
 55 | 3. PJST-D1 固定/重放选择结果的物理时间表示 OFF/ON 配对已完成点估计，但平均点差为负且配对区间缺失；它不再是当前优先开发项；
 56 | 4. 固定 `K=384` 的原生 tubelet 均匀选择与任务状态驱动时序 coreset 已完成；细粒度 coreset 候选因一致的负向点估计终止；
 57 | 5. 同一 H65 代码基座上的 matched allocation control 与 `DUCA-Coverage-v1` 已完成真实训练样本无标签重放门；该门未通过，因此两个 60 轮完整训练臂没有启动。当前对照实现是预算校准系统采样，而冻结设计曾概括为 Top-K；在 Pro 裁决该基线身份和失败机制前不得重提正式训练。
 58 | 6. 冻结 H65 的 K256/K384/K512 反事实边际预算实验、cap-release 诊断和 capped→released 差分邻域的 96 个联合状态枚举均已完成。50% 上限 oracle 的 Avg-mAP/mAP@0.7 增益为 `+0.726/+0.729` 个百分点；解除上限后反而降为 `+0.427/+0.450`。96-state 中没有状态同时达到 `+0.8/+1.0`；当前加性 Marginal-v1 及其本次视频级联合效用修复均按冻结规则停止。
 59 | 
 60 | 主要指标为 tIoU 0.3、0.4、0.5、0.6、0.7 下的 mAP，以及五个阈值的平均值；同时报告短动作、边界定位和完整端到端计算成本。任何计算节省主张都必须来自实际执行的 VideoMAE 工作量和相同硬件条件下的测量，不能由 padding 后的名义帧数推断。
 61 | 
 62 | 完整训练至少每 5 个训练轮次保存一次可恢复的 PyTorch `.pth` 检查点；如果未修改的官方配置保存更频繁，则保留官方间隔。恢复包应包含模型、指数移动平均模型、优化器、学习率调度器、混合精度缩放器、训练轮次或更新计数及随机状态，并至少保留最近三个有效恢复点、预定义里程碑和最终检查点。最终结果使用预先登记的最终模型或最终指数移动平均模型，不能根据中间验证结果事后挑选。
 63 | 
 64 | ## 6. 已取得的性能结果
 65 | 
 66 | 除特别说明外，Avg-mAP 是 tIoU 0.3 至 0.7 五个阈值下 mAP 的平均值。
 67 | 
 68 | | 实验 | Avg-mAP | mAP@0.7 | 证据边界 |
 69 | |---|---:|---:|---|
 70 | | 共享官方 dense AdaTAD | 68.73 | 未在本报告重复摘录 | 一次共享复现；论文公开 Avg-mAP 为 69.03 |
 71 | | H65 30+60 | 65.13 | 43.31 | 单种子、完整训练；当前 H65 参考 |
 72 | | H65 20+40 | 62.46 | 39.94 | 同 seed 的训练压缩诊断；比 30+60 低 2.66/3.37 点 |
 73 | | H65 30+30，AM-RPCH25 | 63.22 | 41.25 | 只改变第二阶段学习率日程 |
 74 | | H65 30+30，LongCosine-H6000 | 63.56 | 41.01 | 更慢衰减仍未恢复 30+60 |
 75 | | RankPack K384 | 61.57 | 37.10 | 单种子物理时间表示对照 |
 76 | | TrueTime K384 | 62.19 | 37.89 | 相对 RankPack 为 +0.62/+0.79 点；尚无配对区间 |
 77 | | PJST-D1 OFF | 65.063 | 43.646 | 211/211 视频，冻结 H65 选择结果 |
 78 | | PJST-D1 ON | 64.591 | 43.769 | 211/211 视频；相对 OFF 为 -0.472/+0.123 点 |
 79 | | 原生 tubelet uniform K384 | 64.13 | 42.45 | 单 seed；60 轮训练和官方验证完成；结构化指标文件未封存 |
 80 | | 原生 tubelet coreset K384 | 62.81 | 40.56 | 单 seed；相对匹配 uniform 为 -1.32/-1.89 点；结构化指标文件未封存 |
 81 | | 连续片段 FZ | 49.89 | 29.68 | 完整单种子训练；明显负结果 |
 82 | | 连续片段 JT | 47.24 | 26.52 | 完整单种子训练；联合训练未恢复性能 |
 83 | 
 84 | PJST-D1 两臂各产生 422,000 条预测，视频集合一致，重推理逐项复现了原始点估计。预先登记的 10,000 次整视频配对自助法在任何抽样前退出，因为统计程序指向了错误的预测文件目录。因此当前只有点估计，没有置信区间；不能据此宣布总体效应显著为负，也不能把 mAP@0.7 的 `+0.123` 点解释为真实收益。
 85 | 
 86 | UVT 的 legacy/geometry/geometry+EMA 三臂 Avg-mAP 为 `57.35/55.93/55.92`。Fovea/Query-Bridge 第一波中最佳 `query_cycle` 为 `54.67`。这些结果来自不同代码版本与实验合同，只能说明相应首版设计未恢复 H65 性能，不能用来量化某一个组件的因果影响。
 87 | 
 88 | 历史 `65.3857` 是 H65 语义间接非均匀逐帧选择的 30+60 诊断结果；`65.696` 来自改变物理检测网格的探索实现。后者同时改变了检测器时间几何，因此不是与官方原生检测器严格匹配的输入采样对照。当前干净 H65 复现 `65.13` 是更适合作为后续同代码、同训练协议比较的参考。
 89 | 
 90 | 当前还没有与主路线匹配、可用于论文的完整端到端成本结果，也没有动态预算保持性能或提高效率的实验证据。`DUCA-Coverage-v1` 已完成代码实现、独立审查和真实 training 数据运行前门；该门给出不满足预注册干预条件的诊断结果，但没有进入训练，因此不是 mAP 或成本结果。
 91 | 
 92 | `DUCA-Marginal-v1` 的冻结 detector 诊断在训练侧 40 个 utility holdout 视频、124 个窗口上比较了固定 K384 与使用真实反事实效用的等预算 oracle。固定臂 Avg-mAP/mAP@0.7 为 `88.131/76.271`，50% 上限 oracle 为 `88.857/77.000`。解除上限后，分配从 K256/K384/K512 的 `11/102/11` 变为 `17/90/17`，但结果降为 `88.559/76.721`；相对固定臂只有 `+0.427/+0.450` 个百分点。总 observation 预算仍精确为 `47110`。这里的百分数只描述训练侧 controller holdout，不可与 official validation/test 表直接比较。强 headroom 门未通过，因此 utility predictor、正式测试、配对区间和端到端成本均未运行。
 93 | 
 94 | 随后只在 capped 与 released 分配不同的 12 个窗口上枚举了全部 96 个逐视频等成本联合状态。最佳 Avg-mAP 状态相对固定 K384 为 `+0.733` 个百分点，但 mAP@0.7 仅 `+0.479`；最佳 mAP@0.7 状态为 `+0.549/+0.934`；按两项联合门最优的状态为 `+0.554/+0.933`。没有状态同时满足 `+0.8/+1.0`，也没有单个最小合法转移同时改善 Avg-mAP 与 mAP@0.7。该诊断没有执行模型前向、训练、official test 或 bootstrap；从 96 个开发集状态中事后选出的最优状态不能当作可部署策略或论文主结果。
 95 | 
 96 | ## 7. 结果解释与已停止的方向
 97 | 
 98 | - 20+40 训练压缩、AM-RPCH25 和 LongCosine-H6000 均未恢复 H65 30+60 的性能。现有证据说明简单压缩预热或只修改第二阶段学习率尾部不足以保持性能；它不否定 H65 的语义间接选帧机制。
 99 | - 连续 16 帧片段采样在真实训练中造成大幅定位损失，联合训练也未恢复，因此该采样单元不再作为当前主路线。这个结果不否定低成本语义侦察或物理时间一致性的一般问题。
100 | - TrueTime 相对 RankPack 有小幅单种子提升，但证据不足以形成论文主张。
101 | - PJST-D1 的当前点估计没有显示平均性能收益；缺少配对置信区间意味着总体效应仍未完成统计裁决。统计程序的路径错误是证据生成失败，不是模型的科学失败。
102 | - UVT 与 Fovea/Query-Bridge 同时改变了选择分数、预算证据或训练信息流，且缺少与 H65 同提交的严格隔离，因此其性能下降不能归因于单一组件。
103 | 
104 | ## 8. 当前证据缺口与下一动作
105 | 
106 | THUMOS14 原始视频、注释、类别映射、VideoMAE-S 预训练权重、H65 Stage-1 侦察器检查点和共享官方 AdaTAD 结果均已核验。PJST-D1 的 OFF/ON 完整推理已经结束。
107 | 
108 | 当前固定预算归因已经得到负向点估计：在完全相同的 `K=384` 高分辨率帧预算、训练日程和检测器下，任务状态驱动 coreset 没有优于原生 tubelet 均匀选择，并在 tIoU 0.5 至 0.7 下降 `1.54/2.03/1.89` 个百分点。这个结果优先要求分析端点覆盖、最大空洞、跨 tubelet 打包、低分辨率上下文回收和选择分数是否共同损害重型表示，而不能直接把固定预算重命名为最终方法。
109 | 
110 | PJST-D1 的配对区间仍属未完成证据，但它不会改变当前路线，因此不作为当前任务补齐。原生 tubelet 两臂已从干净 H65 基座完成实现、N16R4 聚焦测试、独立审查和 60 轮训练。uniform `1260184` 与 coreset `1260185` 都写出 epoch-59 检查点并完成官方 211 视频评估，日志点估计分别为 `64.13%` 和 `62.81%`。两臂随后因同一个证据封存错误退出：配置没有保存预测，结构化指标入口因此拒绝写出 `metrics_epoch59_ema.json`。当前没有配对区间或成本结果；训练成功、日志点估计和结构化证据缺失必须分别陈述。
111 | 
112 | Pro 冻结的 96-state 联合 mAP 邻域诊断已经在公开提交 `46812fac...` 上完成。原 fixed/capped/released 三结果复现误差为 `0.0` 个百分点，96 次评估保持逐视频预算和全局成本 `47110`；没有状态通过 `+0.8/+1.0` 联合门，也没有单个最小合法转移同时改善两个门指标。Pro 已据此最终裁决 `STOP`：现有加性 Marginal-v1 及本次邻域修复关闭，不再重跑、改门、补 bootstrap、训练 utility head 或访问 official test。当前分支只作为负证据读取；未来若重新研究动态计算，必须由 Pro 以新的机制假设和独立任务启动，不能作为 Marginal-v1 的恢复。
113 | 
114 | ## 9. 可发表性边界
115 | 
116 | 目前可以写入论文的事实是：H65 的 30+60 训练参考明显优于已测试的 60 轮压缩日程；连续片段采样是明确负结果；PJST-D1 的匹配点估计没有平均收益；原生 tubelet coreset 的单种子点估计比匹配均匀选择低 `1.32` 个百分点；当前 96-anchor Coverage 干预没有达到预注册中间机制目标。DUCA-Marginal 的训练侧 oracle 在 50% 上限下仅有灰区增益，解除上限后增益进一步降至 `+0.427/+0.450` 个百分点；其 96-state 联合差分邻域也没有达到 `+0.8/+1.0` 的预注册门，因此当前机制及本次联合修复停止。它尚未证明 Scout 能预测这种效用，也没有 official test、统计区间或端到端成本结果。不能把该停止条件扩展为所有 coverage、动态预算或物理时间方法无效，也不能声称 DUCA-Marginal 有效、优于 dense，或已经获得性能—成本联合优势。
```

### File: research-wiki/experiments/duca-marginal-v1.md
Lines: 1-83
```md
 1 | ---
 2 | type: experiment
 3 | status: empirically_supported
 4 | updated: 2026-08-31
 5 | project: DUCA
 6 | ---
 7 | 
 8 | # DUCA-Marginal-v1：窗口级加性边际效用的终态负结果
 9 | 
10 | ## 科学问题
11 | 
12 | 本实验检验：在冻结 H65 时序优先序列、epoch-59 指数移动平均检测器和 K256/K384/K512 三档真实重型
13 | observation 预算后，逐窗口反事实检测损失能否作为加性效用，在每个视频实际 observation 总成本严格不变时，
14 | 把计算从低价值窗口转移到高价值窗口，并同时改善 Avg-mAP 与高时间交并比定位。
15 | 
16 | 40 个训练侧 utility holdout 视频只从 utility-head fitting 中留出，并不是 H65 detector 的未见测试集。本实验
17 | 从未访问 official test，因此所有百分数都属于开发集机制诊断，不能与 official validation/test 主表直接比较。
18 | 
19 | ## 实现与证据身份
20 | 
21 | - 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
22 | - 最终只读证据分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831>
23 | - 最终精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d>
24 | - 父提交：`d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`
25 | - allocator Git blob：`268c26cf41ae8a0d33c5a1b849ebff2adf0b388e`
26 | - runner：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tools/bata/run_duca_marginal_frozen_h65_probe.py>
27 | - 未修改的 allocator：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/opentad/models/duca/dynamic_budget.py>
28 | - 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tests/test_duca_marginal_budget.py>
29 | - 唯一联合邻域 Evaluator：Slurm Job `1262121`，`COMPLETED 0:0`
30 | - 原始终态：`.cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-neighborhood-46812fac-job1262121/oracle_cap_release_neighborhood_result.json`
31 | - 原始终态 SHA-256：`a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`
32 | 
33 | 最终提交只修改 probe runner 与聚焦测试；allocator、模型、预测生成、损失、数据、Soft-NMS 和评估器均未
34 | 改变。N16R4 上 16 项聚焦测试、23 项既有回归测试和独立 Critic 通过。正式作业完成 96 次 CPU evaluator
35 | 调用；集群分区虽要求申请一张 GPU 作为调度资源，但作业清空 CUDA 可见性并固定 CPU。没有 detector/Scout
36 | forward、模型训练、utility-head fitting、梯度、official test 或 bootstrap。
37 | 
38 | ## 结果
39 | 
40 | 固定 K384 为 Avg-mAP `88.131197%`、mAP@0.7 `76.270583%`。50% 改变窗口上限的真实效用 oracle 为
41 | `88.856786%/76.999587%`，相对固定为 `+0.725589/+0.729004` 个百分点；解除上限后降为
42 | `88.558507%/76.720863%`，相对固定仅 `+0.427310/+0.450280`。三项参考结果的复现误差均为 `0.0`
43 | 个百分点，全局实际 observation 成本始终为 `47110`。
44 | 
45 | 最终诊断从 capped/released 差分自动导出 5 个视频、12 个窗口、6 个净转移组、8 个最小合法转移和 96 个
46 | 逐视频等成本联合状态，并保留 `video_validation_0000419` 的全部四种最小配对与两种完整分解。结果为：
47 | 
48 | - 联合继续门通过状态：`0/96`；
49 | - 联合门最优 `state_014`：相对固定 `+0.553972/+0.933234` 个百分点；
50 | - Avg-mAP 最优 `state_020`：`+0.732990/+0.479291`；
51 | - mAP@0.7 最优 `state_001`：`+0.548669/+0.933539`；
52 | - 8 个最小等成本转移中，没有一个同时改善 Avg-mAP 与 mAP@0.7。
53 | 
54 | 冻结继续门为 `+0.8/+1.0` 个百分点，因而本实验按预注册规则停止。没有对同一开发集事后最佳状态运行
55 | bootstrap；这类 bootstrap 也不能把未达到实用效应门槛的点估计变成达到门槛的确认性效应。
56 | 
57 | ## Pro 终态裁决与停止边界
58 | 
59 | 精确 DUCA Project 的 Pro 终态报告保存于
60 | `.cvpr-pro-lab/pro-reviews/runs/duca-marginal-cap-release-neighborhood-terminal-compact-v002/visible-report.md`，裁决为
61 | `STOP`。Pro 认为直接支持的科研表述是：
62 | 
63 | > 窗口级加性反事实检测损失不是视频级联合检测效用的充分排序统计量。失败已经出现在最小等成本预算转移
64 | > 层面：没有一个最小转移能够同时改善平均检测性能和高时间交并比定位，因此不存在一组各自联合有益、只是
65 | > 组合后被交互破坏的转移来挽救该邻域。
66 | 
67 | 这里的“最小项”是一个等成本预算转移，通常包含一降一升两个窗口；论文不使用内部分类
68 | `single-item misranking primary`。`interaction_witness_count=0` 只否定“若干已各自联合有益的转移在组合后发生
69 | 反转”这一特定解释，不能证明 Soft-NMS、排序或窗口组合完全没有交互。
70 | 
71 | 本终态足以停止：当前 capped→released 联合邻域修复，以及由同一 H65 priority sequence、三档预算、逐窗口
72 | 反事实 detector loss、逐视频等成本加性分配和 cap/配对/tie-break 调整组成的现有 DUCA-Marginal-v1。它不足
73 | 以否定 K256/K384/K512 三档本身、H65 priority sequence、所有三档分配或任务感知动态计算的一般问题。
74 | 
75 | ## 只读归档状态
76 | 
77 | 分支 `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831` 从本记录起只作为负证据读取，不再接受恢复性
78 | 修改、阈值调整、配对重排、事后状态选择、bootstrap、utility-head 训练或 official-test 运行。本机制没有后继
79 | Builder、Critic、PRE_RUN、Evaluator 或算力任务。若未来重新研究动态计算，必须由 Pro 提出新的机制假设和
80 | 独立任务，不能描述成 Marginal-v1 的小修或恢复。
81 | 
82 | 这组结果可进入论文的失败机制分析或补充材料，但不能进入主性能表作为候选方法成绩，也不能表述为动态预算
83 | 总体无效、H65 无效、三档预算无效、统计显著的负总体结论或可部署 oracle。
```

### File: .cvpr-pro-lab/pro-reviews/runs/duca-marginal-cap-release-neighborhood-terminal-compact-v002/visible-report.md
Lines: 1-200
```md
  1 | STOP
  2 | DUCA-MARGINAL-CAP-RELEASE-NEIGHBORHOOD-TERMINAL-ADJUDICATION-v001-20260831
  3 | 
  4 | ## 终态科学裁决
  5 | 
  6 | **停止当前 capped→released 联合邻域修复，并据此终结现有加性 DUCA-Marginal-v1。不存在另一项值得在这一冻结机制内执行的科学实验。**
  7 | 
  8 | 这不是停止整个 DUCA 项目，也不是否定三档预算、H65 时序优先序列或任务感知动态计算的一般可能性。它停止的是一个已经被完整检验的具体主张：
  9 | 
 10 | > 由逐窗口 K256/K384/K512 反事实检测损失构成的加性效用，在保持逐视频实际 observation 成本不变时，能够通过 cap release 或 capped→released 邻域内的联合预算重组，找到同时改善 Avg-mAP 与高 tIoU 定位的配置。
 11 | 
 12 | 该主张现已达到终局负证据。
 13 | 
 14 | ## 1. 实现忠实度与证据有效性
 15 | 
 16 | ### 1.1 代码身份正确
 17 | 
 18 | 精确提交 `46812facc8773d9b4a9c21833cbe397c8aaa5a2d` 相对父提交 `d2fad7c0dfc4a5efe98b10b9eee4723c6805699f` 只有一个提交差异，并且只修改：
 19 | 
 20 | * `tools/bata/run_duca_marginal_frozen_h65_probe.py`
 21 | * `tests/test_duca_marginal_budget.py`
 22 | 
 23 | allocator、模型、预测生成、损失、NMS 和评估器均未修改。GitHub 的精确比较与提交内容一致。 ([GitHub][1])
 24 | 
 25 | allocator 仍然以逐视频动态规划最大化分离后的窗口级预测效用，要求总实际 observation 成本严格等于全 K384 基线成本；`max_changed_fraction` 只限制允许改变的窗口数量，释放该上限不会改变效用函数、预算层级或成本口径。短窗口折叠、实际 observation 成本和 packet 化执行槽位也仍被明确区分。
 26 | 
 27 | ### 1.2 联合邻域忠实于冻结任务
 28 | 
 29 | runner 没有硬编码六个预算转移配对。它先从 capped 和 released 分配的真实差分自动取得窗口，再逐视频枚举每个差分窗口采用 capped 或 released 状态的所有组合，只保留满足该视频原始实际成本的状态，最后对各视频状态做笛卡尔积并再次检查全局成本。它还要求：
 30 | 
 31 | * 不得改变非差分窗口；
 32 | * 每个视频成本分别守恒；
 33 | * 全局成本保持不变；
 34 | * capped 与完整 released 状态都必须存在；
 35 | * 每个联合状态必须能够由不重叠的最小合法转移分解。
 36 | 
 37 | `video_validation_0000419` 的四种最小配对和两种完整分解因此都被保留，而不是任意选定其中一种。
 38 | 
 39 | 代码中的 `5 个视频、12 个窗口、6 个净转移组、96 个状态、47110 总成本` 是**推导完成后的失败即停形状核验**，不是生成状态的硬编码模板；如果密封输入推导出不同结构，程序会拒绝运行，而不会把数据强行改成预期结构。随后每个状态均通过同一个 `_official_holdout_metrics` 入口重新组合密封预测、执行相同 Soft-NMS 和评估，共 96 次。
 40 | 
 41 | 新增测试明确覆盖了：
 42 | 
 43 | * 5 个差分视频和 12 个窗口；
 44 | * `2×2×2×2×6=96` 个唯一状态；
 45 | * 8 个最小合法转移；
 46 | * 0419 的四种配对与两种完整分解；
 47 | * 每个状态严格保持成本；
 48 | * 新结果文件不得覆盖旧 probe 与 cap-release 结果。
 49 | 
 50 | 结合 16 项聚焦测试、23 项既有回归测试、独立 Critic 的静态通过、三组历史指标零误差复现和作业中的 96 次真实 evaluator 调用，我没有发现会改变本轮有限邻域结论的实现、成本或评估缺陷。
 51 | 
 52 | ### 1.3 存在的限制不是结论逆转因素
 53 | 
 54 | 第一，96 个状态是在同一开发 holdout 上评估和排序的，因此 `state_014`、`state_020` 或 `state_001` 都不能被部署，也不能作为确认性结果。这种事后选择原本会偏向产生虚假的正结果；但本轮在这种有利于“找到好状态”的条件下仍然是 `0/96`，所以它不解释门槛失败。
 55 | 
 56 | 第二，runner 对已存在的终态结果采用幂等返回。不过作业记录显示实际调用 evaluator 96 次，说明正式运行没有通过已有结果文件跳过枚举。
 57 | 
 58 | 第三，没有 bootstrap 是冻结规则的正确执行。bootstrap 可以估计点差的不确定性，却不能把未达到预注册实用效应门槛的点估计变成达到门槛的效应；更不能对从同一 96 状态中事后选出的最佳状态提供确认性推断。
 59 | 
 60 | 第四，本会话实际挂载的是九个 Markdown 附件，没有出现终态 JSON 的原始字节文件。因此我没有在本地重新计算 `a8020892…` 的 SHA-256，也没有逐条重算 96 个 JSON 记录；这部分数值按权威提示词和绑定的正式运行事实接纳。终态归档必须保留原 JSON，不能只保留本裁决摘要。
 61 | 
 62 | 九个附件之间没有形成影响本裁决的事实冲突：旧代码库存只具有索引意义，旧提示词、2026-08-28 状态和历史版本注册表不得覆盖当前精确提交；项目规则、角色规则和论文优先流程则要求把点估计、统计结论、代码通过和科学主张严格分开，并在没有新信息增益时终止重复实验。PJST-D1 的旧证据只提供这种证据边界原则，不参与本次 Marginal-v1 效果归因。        
 63 | 
 64 | ## 2. 直接证据支持的失败机制
 65 | 
 66 | ### 直接观察
 67 | 
 68 | 1. **联合门为 `0/96`。**
 69 |    最接近门槛的 `state_014` 只有 `+0.553972 pp` Avg-mAP 和 `+0.933234 pp` mAP@0.7。它距离 Avg-mAP 门槛仍差 `0.246028 pp`，不是一个只差数值噪声的联合通过状态。
 70 | 
 71 | 2. **两个指标的最优状态明显分离。**
 72 |    `state_020` 给出最高 Avg-mAP，但只有 `+0.479291 pp` 的 mAP@0.7；`state_001` 给出最高 mAP@0.7，但只有 `+0.548669 pp` 的 Avg-mAP。不存在一个状态继承两者优势。
 73 | 
 74 | 3. **所有八个最小合法转移都不是联合有益转移。**
 75 | 
 76 |    * video 0059：牺牲 Avg-mAP 换取 mAP@0.7；
 77 |    * video 0206：Avg-mAP 几乎不变而 mAP@0.7 明显下降；
 78 |    * video 0267、0419 的四种配对和 video 0905：两项指标均下降。
 79 | 
 80 | 4. **释放 cap 改善了 allocator 的加性损失目标，却使最终 mAP 由 capped 的 `+0.725589/+0.729004 pp` 降为 `+0.427310/+0.450280 pp`。**
 81 |    因此，问题不再是 capped 可行集过窄；扩大可行集反而让加性目标选择了对最终检测指标更差的预算转移。
 82 | 
 83 | ### 最合理的科学解释
 84 | 
 85 | 冻结状态码 `single-item misranking primary` 不应直接进入论文。更准确、可理解的表述是：
 86 | 
 87 | > **窗口级加性反事实检测损失不是视频级联合检测效用的充分排序统计量。失败已经出现在最小等成本预算转移层面：没有一个最小转移能够同时改善平均检测性能和高 tIoU 定位，因此不存在一组“各自联合有益、只是组合后被交互破坏”的转移来挽救该邻域。**
 88 | 
 89 | 这里的基本单位其实是“最小等成本转移”，通常包含一个降档窗口与一个升档窗口，而不是字面上的单个窗口。因此论文不应使用“single-item”这种容易误导的内部术语。
 90 | 
 91 | 机制上，窗口级 detector loss 是可加的局部代理，而最终 mAP 是经过预测合并、类别排序、Soft-NMS 和全数据集精确率—召回率计算后的非可加指标。一个降低局部损失的预算移动，可能改变重复预测、置信度排序或边界定位，使 Avg-mAP 与 mAP@0.7 呈现相反方向。当前结果直接支持的是**效用目标与最终联合指标失配**，而不是 cap 约束不足。
 92 | 
 93 | 代码中的 `interaction_witness_count=0` 也必须谨慎解释：其定义只在至少两个最小转移分别对两项指标都为正时，才检查它们的组合是否反转或退化；本轮根本不存在这样的转移，所以 witness 为零在逻辑上是必然的。它不能证明 Soft-NMS、AP 或窗口组合完全没有交互，只能说明“由若干已经联合有益的原子转移组成、却被交互破坏”的特定解释不成立。
 94 | 
 95 | ## 3. 精确停止边界
 96 | 
 97 | ### 本轮证据足以停止
 98 | 
 99 | 1. **capped→released 差分邻域的联合效用修复。**
100 | 2. **现有加性 DUCA-Marginal-v1：**
101 | 
102 |    * 同一密封 H65 priority sequence；
103 |    * K256/K384/K512 三档；
104 |    * 逐窗口反事实 detector loss 作为效用；
105 |    * 逐视频等实际成本加性分配；
106 |    * 通过改变 `max_changed_fraction`、配对、组合或 tie-break 寻找收益。
107 | 3. 在同一 96 状态上继续：
108 | 
109 |    * 改联合门；
110 |    * 挑另一个折中状态；
111 |    * 更换配对；
112 |    * 重新排序；
113 |    * 对事后最佳状态补 bootstrap；
114 |    * 访问 official test；
115 |    * 训练 utility head；
116 |    * 再次运行同一 evaluator。
117 | 
118 | 此前 cap release 已经停止当前加性机制，只留下联合邻域作为唯一可能的差分修复。本轮又穷尽并否定了该修复，所以现有加性 Marginal-v1 到此终结，而不仅是“暂时调低 cap”。
119 | 
120 | ### 本轮证据不足以停止
121 | 
122 | 1. **K256/K384/K512 三档本身。**
123 |    它们只是计算动作空间；失败的是当前效用与分配依据，而不是三个预算数字的存在。
124 | 
125 | 2. **H65 priority sequence 本身。**
126 |    本轮固定并复用了该序列，没有与另一种合法优先序列做匹配比较，因而不能把失败归因于 H65 的排序质量。
127 | 
128 | 3. **任务感知动态计算的一般研究问题。**
129 |    本轮没有检验类别感知、proposal 集合感知、边界风险感知或直接以视频级集合效用为目标的其他机制，也没有检验其他独立数据或 detector。
130 | 
131 | 4. **所有可能的三档预算分配。**
132 |    96 状态是 capped 与 released 两个分配之间、逐差分窗口二选一所形成的完整邻域，不是全部窗口的 `3^W` 全空间。因此可以停止这条已冻结的修复路线，但不能发表“所有三档动态预算都不可能有效”的普遍结论。
133 | 
134 | 不过，这些更宽的方向目前都只是**未被否定**，并不等于获得继续授权或正面证据。未来若重新研究，必须作为新的机制假设和新的 Pro 科学任务启动，不能把它描述成 Marginal-v1 的小修。
135 | 
136 | ## 4. 是否还有一项能改变结论的低成本实验
137 | 
138 | **没有。**
139 | 
140 | 原因不是缺少实验想象力，而是当前决策变量已被穷尽：
141 | 
142 | * 96 个冻结合法状态已经全部评估；
143 | * 联合门没有通过状态；
144 | * 最优 Avg-mAP 与最优 mAP@0.7 无法合一；
145 | * 没有一个最小转移同时改善两项指标；
146 | * capped 与 released 两个端点也都未达到门槛；
147 | * bootstrap 不会改变实用效应门槛失败；
148 | * 对同一开发集继续选择或拟合只会增加事后偏差；
149 | * official test 被正确禁止；
150 | * 任何真正不同的效用目标、优先序列或训练方法都已是新机制，而不是本轮低成本 falsifier。
151 | 
152 | 因此：
153 | 
154 | * **Builder：无新修改。**
155 | * **Critic：无新代码复审。**
156 | * **Evaluator PRE_RUN：无。**
157 | * **唯一正式运行：无。**
158 | * **GPU/CPU 作业：均不再提交。**
159 | 
160 | ## 5. 论文应如何保存这组负结果
161 | 
162 | 这组结果适合进入论文的“失败机制分析”或补充材料，而不是主性能表中的候选方法成绩。应完整保存：
163 | 
164 | * 精确 commit、父提交和 allocator blob；
165 | * Slurm Job `1262121`；
166 | * 原始 JSON 路径与 SHA-256；
167 | * 预注册门槛；
168 | * 5 个差分视频、12 个差分窗口、6 个净转移组、8 个最小转移和 96 个状态；
169 | * 每个状态的实际 observation 总成本 `47110`；
170 | * Fixed、capped、released 三个零误差复现锚点；
171 | * `0/96`、三个最佳状态以及全部最小转移结果；
172 | * 没有训练、模型前向、official test 或 bootstrap 的事实。
173 | 
174 | 论文可写的结论是：
175 | 
176 | > 在一个密封的开发 holdout 上，我们穷尽了 capped 与 released 三档等成本分配之间的全部 96 个合法联合状态。虽然释放分配上限改善了窗口级加性反事实损失目标，但没有状态同时达到预注册的 Avg-mAP 与 mAP@0.7 实用增益门槛；所有最小预算转移也都未同时改善两项指标。这表明窗口级加性 detector-loss 效用不足以指导该邻域内的视频级动态预算分配。
177 | 
178 | 论文不可声称：
179 | 
180 | * 动态预算在总体上无效；
181 | * H65 优先序列无效；
182 | * 三档预算设计无效；
183 | * 效应具有统计显著的负总体结论；
184 | * 最佳 96-state oracle 是可部署方法；
185 | * 本结果来自 official validation 或 official test；
186 | * `interaction_witness_count=0` 证明不存在任何预测或 NMS 交互。
187 | 
188 | ## 6. 唯一终结动作
189 | 
190 | `next_owner`: **Coordinator**
191 | 
192 | `next_action`: 在不改代码、不重跑 evaluator 的前提下，完成一次终态归档：把精确 commit、父提交、allocator blob、Job、JSON SHA、预注册门槛、`0/96`、三类最优状态、八个最小转移、停止边界和上述论文表述写入对应实验页、`decision_history.md` 与 append-only `log.md`；将当前分支标记为只读负证据；不得把内部分类 `single-item misranking primary` 原样写入论文。
193 | 
194 | `dependency`: 取得并验证原始 `oracle_cap_release_neighborhood_result.json`，确认其 SHA-256 为 `a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`；除此之外无科学或算力依赖。
195 | 
196 | `absolute_deadline`: **2026-09-01T18:00:00+08:00**
197 | 
198 | 完成该归档后，本任务关闭，不再返回 Marginal-v1 做恢复性实验。
199 | 
200 | [1]: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d"
```

## Answer

PIVOT
DUCA-PROJECT-LEVEL-AFTER-MARGINAL-STOP-v001-20260831

## 一、项目级科学裁决

**不恢复 DUCA-Marginal-v1，也不停止整个 DUCA 问题；项目只转向一次“整视频预算一致、跨视频转移计算”的终局低成本证伪。**

这是目前唯一同时满足以下条件的新机制假设：

1. 没有被现有固定预算、Coverage、coreset、连续片段、PJST-D1 或 Marginal-v1 直接检验；
2. 能解释为什么逐窗口预算转移改善局部代理，却破坏最终检测指标；
3. 可以完全复用已经密封的 K256/K384/K512 预测，不训练、不访问 official validation/test、不执行新的 VideoMAE 前向；
4. 若在这种对假设极为有利的开发集 oracle 条件下仍失败，就没有合理依据继续投入完整训练。

远端证据分支头确为 `46812facc8773d9b4a9c21833cbe397c8aaa5a2d`；它比父提交只前进一个提交，唯一修改面是联合诊断 runner 与聚焦测试，没有修改 allocator、模型或训练代码。

## 二、当前证据是否足以终止整个研究问题

### 1. 已被直接证据否定的内容

本轮附件支持以下直接结论：

* 连续高分辨率片段作为采样单元已经得到明显负结果，联合训练没有恢复；
* 固定 K384 的任务状态 coreset 低于匹配 uniform，不能再把更细粒度的语义重排默认视为收益来源；
* Coverage-v1 连最基本的集合变化、覆盖改善和最大空洞机制门都没有通过；
* PJST-D1 的匹配点估计没有 Avg-mAP 正向支持；由于整视频配对区间没有生成，它不是已确认的总体负效应，但也不再提供路线级正证据。
* Marginal-v1 的 50% capped oracle 只有 `+0.725589/+0.729004` 个百分点；解除 cap 后下降到 `+0.427310/+0.450280`；
* capped→released 的全部 96 个合法联合状态中，`0/96` 达到 `+0.8/+1.0` 门，八个最小等成本转移没有一个同时改善 Avg-mAP 与 mAP@0.7。

因此，**逐窗口反事实 detector loss、逐视频内部等成本、加性预算分配**已经终结。当前 allocator 确实逐视频独立求解，目标成本固定为该视频全 K384 的实际 observation 总和；短窗口按照 `min(valid,K)` 计费，无法通过名义 padding 制造虚假稀疏。

### 2. 最合理的跨路线失败机制

以下是基于结果作出的因果解释，而不是已被直接证明的事实：

**第一，动作性或边界显著性不等于重型表示的边际价值。**
Scout 可以指出“哪里像动作或边界”，却未必能判断减少哪些上下文不会改变类别置信度、重复 proposal、边界回归和最终排序。H65 与 dense 之间仍约有 3.6 点 Avg-mAP 差距，也表明当前稀疏观察本身已经损失了相当的表示容量；旧状态报告同样明确，现有实验还没有形成动态预算或端到端成本优势。

**第二，逐窗口预算决策与最终检测单位错位。**
THUMOS14 滑窗检测中，同一动作常由多个重叠窗口共同产生 proposal。一个视频内部同时存在 K256、K384 和 K512 窗口，会让重叠预测具有不同的表示质量、置信度校准和边界偏差。最终 Avg-mAP 又经过视频内预测合并、Soft-NMS 以及跨数据集排序，不是窗口 loss 的可加函数。cap release 改善加性目标却降低最终 mAP，正是这种错位的直接迹象。

**第三，旧路线反复在“提高边界密度”和“保护时间覆盖”之间交换误差。**
连续片段过度聚集，coreset 和 Coverage 又没有找到同时保护上下文与边界的固定预算结构；动态窗口转移则在两个最终指标之间产生冲突。这更像决策单位与目标失配，而不是尚缺一个学习率、cap 或 tie-break。

### 3. 尚未排除、但也没有正证据的部分

现有 Marginal allocator禁止视频之间转移计算：每个视频必须独立守恒自身 K384 成本。
因此，它没有检验：

> **一个视频的全部重叠窗口保持同一预算档位，同时把总计算从低价值视频转移到高价值视频。**

这是本轮唯一保留的未决机制。其他更宽泛的类别感知、proposal 集合效用或新 Scout 训练方式只是逻辑上未被否定，并没有足够正证据或一个同样便宜的决定性实验，本轮不授权。

## 三、当前结果能否形成可投稿贡献

**不能形成一篇以 DUCA 为有效方法的主论文贡献。**

欠缺并非主要是工程完成度，而是以下科学要件：

1. **缺少通过机制门的新方法。** 当前可写的是一组约束明确的负结果，而不是一个有效候选。
2. **缺少严格匹配的性能—真实成本曲线。** 共享 dense AdaTAD 为 `68.73`，H65 为 `65.13`，但现有材料明确指出它们尚未构成完整的同计算预算论文比较。
3. **缺少可部署动态策略。** Marginal 只完成训练侧 40 视频 oracle，没有证明 Scout 能预测最终预算价值。
4. **缺少足够统计证据。** 多项历史结果是单种子；PJST-D1 没有完成预注册配对区间；Marginal 的事后 oracle 状态也不能补 bootstrap 后冒充确认性结果。
5. **缺少 official validation/test 主结果和实测端到端开销。**

现有材料最多适合作为论文的失败机制分析、补充材料或一篇以系统性负证据为主题的独立研究记录。代码、测试、Job 和归档完整度不能替代方法贡献；项目规则也明确要求只有能改变论文判断的实验进入关键路径。 

## 四、唯一新机制：整视频预算一致、跨视频计算转移

### 一句话科学问题

> 在相同或更低的全局实际 observation 成本下，给一个视频的全部重叠窗口使用同一预算档位，并在视频之间转移计算，能否避免逐窗口混合预算造成的 proposal 不一致，从而同时改善 Avg-mAP 和 mAP@0.7？

### 因果机制

* **视频内一致性：** 一个视频的所有窗口统一请求 K256、K384 或 K512，不再让同一动作的不同重叠视图使用不同表示容量。
* **视频间动态性：** 从整体较容易的视频释放计算，交给整体边界风险更高的视频。
* **与检测单位对齐：** 计算决策单位由局部窗口提升为完整视频 proposal 集合，更接近 Soft-NMS 和最终 AP 所作用的单位。
* **H65 保持不变：** 各预算档仍使用同一密封 H65 priority sequence 的嵌套集合，因此不把新 selector 混入本轮。

### 为什么不是 Marginal-v1 换名重试

本机制同时改变了三个根本合同：

* Marginal-v1 在**同一视频内部**移动预算；新机制只在**视频之间**移动预算；
* Marginal-v1 允许同视频窗口混用三档；新机制要求整视频统一请求档位；
* Marginal-v1 以窗口 detector loss 为加性效用；本轮直接用最终 proposal 集合的 Avg-mAP 与 mAP@0.7 作**开发集特权 oracle 裁决**。

它不调用旧 `allocate_equal_budget_marginal_reallocation`，不改变 cap，不扩展 96-state，不训练旧 utility head，也不修改 K 档位或 H65 priority sequence。

## 五、冻结的唯一当前任务：整视频双向转移 oracle falsifier

### 1. 数据和证据等级

只使用现有密封的：

* 40 个训练侧 controller holdout 视频；
* 124 个窗口；
* K256、K384、K512 三套窗口预测；
* 每个窗口的真实 observation 成本；
* 固定 split、annotation、class map、epoch-59 EMA detector、Soft-NMS 和评估器。

这是**开发集机制 oracle**，不是 official validation，不是 official test，也不是部署策略。

### 2. 唯一干预

固定全 K384 为基线。对每一对不同视频 `(d,u)` 构造一个候选：

* donor 视频 `d` 的全部窗口统一请求 K256；
* recipient 视频 `u` 的全部窗口统一请求 K512；
* 其余视频全部保持 K384；
* 因短窗口而与 K384 实际成本相同的请求，继续按现有 sealed accounting 折叠；
* donor 与 recipient 都必须至少包含一个实际非基线窗口。

候选生成不读取标签、GT 或 mAP。标签只在随后调用既有评估器、从所有候选中找开发集 oracle 最优状态时使用。

最多有 `40×39=1560` 个有序候选；只保留满足

$$
C(d,u)\leq C_{\mathrm{fixed}}=47110
$$

的候选，其中成本始终是

$$
\sum_i \min(V_i,K_i)
$$

而不是 requested K、execution slots 或 padding。

### 3. 公平对照与不变量

必须保持逐字或逐对象不变：

* H65 priority sequence 和所有密封三档预测；
* detector、Scout、checkpoint、模型参数和配置；
* 数据划分、annotation、类别映射；
* detector grid、坐标逆映射；
* loss、Soft-NMS、官方评估器；
* K256/K384/K512；
* `+0.8` Avg-mAP 与 `+1.0` mAP@0.7 项目级实用门。

本任务不重复 dense、uniform、coreset 或完整训练。

### 4. 指标与唯一通过规则

报告：

* tIoU 0.3、0.4、0.5、0.6、0.7 的 mAP；
* Avg-mAP；
* mAP@0.7；
* 每个候选的 actual observation 总数及相对 K384 的变化；
* legal candidate 数与 passing candidate 数。

**只有存在至少一个合法 `(d,u)` 候选同时满足**

* `ΔAvg-mAP ≥ +0.8` 个百分点；
* `ΔmAP@0.7 ≥ +1.0` 个百分点；
* actual observation 总成本不高于 `47110`；

才判定开发集 whole-video action space 有足够 headroom，并返回 Pro。

最佳状态按以下顺序确定：

1. 最大化
   `min(ΔAvg-mAP − 0.8, ΔmAP@0.7 − 1.0)`；
2. 成本更低；
3. donor、recipient 视频 ID 字典序。

### 5. 停止规则

以下任一情况直接终止本 falsifier：

* 密封 fixed/capped/released 锚点复现误差超过 `1e-6` 个百分点；
* 任何输入 SHA、视频集合、窗口集合或成本口径不一致；
* 没有合法的 donor–recipient 候选；
* `passing_candidate_count=0`。

前三类身份或确定性实现问题只允许在**不改变机制与门槛**的情况下作一次最小修复。

若正式结果为 `0` 个通过候选，裁决自动转为：

> **项目级 STOP：在当前 THUMOS14、H65 priority sequence、三档真实 observation 动作空间和可接受资源边界下，停止 DUCA 方法创新。**

不得随后增加第三个视频、组合多个 transfer、降低门槛、改变 K 档位、访问 official test、补 bootstrap 或训练控制器。那会把一个失败的 privileged falsifier 变成同开发集上的组合搜索。

本任务不做 bootstrap。若点门通过，最佳候选是从最多 1560 个开发集状态中事后选出，普通 bootstrap 也不能使它成为确认性或可部署结果；它只能证明值得另行讨论 predictability。若点门失败，bootstrap 同样不能改变实用效应门失败。

## 六、最小 Builder、Critic 与 Evaluator 表面

### Builder

从只读证据提交 `46812facc8773d9b4a9c21833cbe397c8aaa5a2d` 建立新分支：

`feature/duca-whole-video-consistent-budget-falsifier-v1-20260831`

只允许新增：

1. `tools/bata/run_duca_whole_video_consistent_budget_falsifier.py`
2. `tests/test_duca_whole_video_consistent_budget_falsifier.py`
3. 一个只负责绑定输入和启动该 CPU evaluator 的薄 N16R4 sbatch 文件；若现有通用提交方式足够，则不新增该文件。

不得修改：

* `tools/bata/run_duca_marginal_frozen_h65_probe.py`
* `opentad/models/duca/dynamic_budget.py`
* selector、acquisition、detector、训练器或配置；
* 既有预测与结果文件。

当前提交的测试已经覆盖 K384 精确保持、真实短窗口成本、packet padding、cap tie-break、全部 96 状态自动推导及旧结果不可覆盖，因此新测试只补 whole-video 候选特有的不变量，不再复制旧合同。

### 独立 Critic 只检查四项

1. 每个 changed video 是否真正对全部窗口使用同一 requested tier，且没有调用旧 Marginal allocator；
2. 候选集合是否在读取 GT 或指标前完整生成；
3. 是否使用真实 observation 成本并满足 `≤47110`；
4. 是否完全复用密封预测、相同 Soft-NMS/评估器，且没有 official validation/test、模型前向或训练。

通过即停止审查，不增加风格、schema 或通用完备性修复。

### Evaluator PRE_RUN

只执行：

* exact HEAD、clean tree 和两项 focused tests；
* 三个 producer、receipts、split 与终态 JSON 的身份检查；
* 原 fixed/capped/released 指标零误差复现；
* `40` 视频、`124` 窗口、`47110` 基线实际成本核对；
* 生成但不评价完整候选清单，确认候选唯一且非空。

### 唯一正式运行

一次 CPU-only evaluator 作业：

* 先评估 fixed K384；
* 再评估全部合法 donor–recipient 候选；
* 写出每个候选的预算、成本、六项指标、相对基线点差和最终通过数；
* 不执行 detector/Scout forward、梯度、训练、bootstrap 或 official test。

即使调度分区要求申请 GPU，该作业也必须清空 CUDA 可见性；调度资源不能被描述为模型计算成本。

## 七、负责人、依赖与截止时间

`next_owner`: **Builder**

`next_action`: 在新分支上实现上述独立 whole-video falsifier 与聚焦测试；完成后依次交给独立 Critic 和独立 Evaluator，Evaluator 只提交一次正式运行。

`dependency`:

* exact commit `46812facc8773d9b4a9c21833cbe397c8aaa5a2d`；
* 三个密封 K256/K384/K512 producer 产物及 receipts；
* `probe_result.json`、`oracle_cap_release_result.json`；
* 终态 `oracle_cap_release_neighborhood_result.json`，SHA-256
  `a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`；
* 已绑定的 split manifest、block list、annotation、class map 和 evaluator；
* 无新数据、无模型训练、无 official-test 权限、无 GPU 科学计算依赖。

`absolute_deadline`: **2026-09-02T18:00:00+08:00**

## 八、论文主张边界

### 若点门通过

只允许写：

> 在训练侧密封开发 holdout 上，一个从全部整视频 donor–recipient 转移中事后选择的特权 oracle，在不超过固定 K384 实际 observation 成本时达到了预注册的 Avg-mAP 与 mAP@0.7 headroom 门。这支持继续检验“视频内预算一致性”是否可由低成本 Scout 预测。

仍不得声称：

* 存在可部署控制器；
* Scout 能预测 donor 或 recipient；
* official validation/test 有收益；
* 结果具有确认性置信区间；
* 已获得真实端到端速度或能耗收益；
* DUCA 优于 dense AdaTAD。

### 若点门失败

应归档：

* 新 exact commit、作业、输入 SHA；
* legal candidate 总数和全部成本；
* fixed 锚点；
* `passing_candidate_count=0`；
* 最优 Avg-mAP、最优 mAP@0.7 和最优联合门候选；
* 没有模型前向、训练、bootstrap 或 official test 的事实。

论文可陈述：

> 在窗口级加性分配失败后，我们进一步穷尽了全部合法的单次整视频 donor–recipient 转移。即使保持视频内预算一致并允许跨视频移动计算，也没有开发集 oracle 状态达到预注册的联合实用门。

这足以在当前资源边界下停止 DUCA 项目级方法创新，但仍不能外推为“所有任务感知动态计算、所有 priority sequence 或所有视频骨干上的动态预算均无效”。

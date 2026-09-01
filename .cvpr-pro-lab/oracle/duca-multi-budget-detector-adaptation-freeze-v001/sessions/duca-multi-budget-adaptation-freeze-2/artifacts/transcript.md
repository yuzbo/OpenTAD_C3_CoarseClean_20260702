# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a9521de-d020-83e9-a0b9-19045c8d5390

## Prompt

Read every provided file completely and independently adjudicate the one new detector-adaptation question. Preserve the old stopped-route boundary, make every scientific choice yourself, and freeze the two missing execution decisions. Preserve nonce DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831 verbatim in the response.

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_MULTI_BUDGET_DETECTOR_ADAPTATION_FREEZE-v001.md
Lines: 1-113
```md
  1 | # DUCA 多预算检测器适应：科学裁决与执行冻结
  2 | 
  3 | Nonce：`DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`
  4 | 
  5 | 你是本课题的科学第一负责人、整体科研流程维护者和论文首脑。你独立负责科学问题、创新机制、可证伪预测、
  6 | 实验路线、结果解释与论文主张。Codex 只执行你冻结后的最小实现、独立代码审查、正式实验评估和证据回传。
  7 | 请独立判断，不要采纳 Codex 预设的路线，也不要为了继续项目而勉强批准实验。
  8 | 
  9 | 本轮只处理一个项目级问题：此前冻结 K384 检测器的三档预算转移已经终止；用户随后提供一份 `REVISE`，建议
 10 | 在旧停止边界之外检验“检测器只在 K384 上训练导致跨预算不适应”这一假说。请审查该新问题是否值得执行，并在
 11 | 值得执行时补齐唯一、可直接交给 Codex 的训练与开发集合同。
 12 | 
 13 | ## 路由与代码身份
 14 | 
 15 | - 精确 ChatGPT Project：`g-p-6a91061f789881918ccd8357ca3d6c92`
 16 | - Project URL：<https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92/project?tab=chats>
 17 | - 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
 18 | - H65 干净模型基座：
 19 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>
 20 | - 最新三档 whole-video 诊断分支：
 21 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>
 22 | - 最新诊断提交：
 23 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
 24 | - whole-video runner：
 25 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>
 26 | - focused test：
 27 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py>
 28 | - 未修改的三档预算实现：
 29 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>
 30 | 
 31 | `33e4ed...` 的科学身份只是终态 whole-video 诊断。它唯一修改密封 proposal 的重放顺序，避免 Soft-NMS
 32 | 平分时的确定性漂移；没有改变 Scout、VideoMAE、Temporal Adapter、检测头或选择机制。若批准新训练，模型
 33 | 基座必须是 `04c35a3b...`；只允许从诊断实现移植真实变长 VideoMAE 执行、packet 对齐、actual observation
 34 | 计数、K384 bit-exact parity、whole-video 评价以及原始生成顺序保持。
 35 | 
 36 | ## 已完成事实与停止边界
 37 | 
 38 | H65 固定 K384 的 30+60 参考为 Avg-mAP `65.13%`、mAP@0.7 `43.31%`。它使用
 39 | `budget_calibrated_sampling_rate`：H65 优先级调制、带覆盖下限的预算校准确定性系统采样；不是普通全局 Top-K。
 40 | 
 41 | 最新训练侧 controller holdout 终态使用 40 个视频、124 个窗口、密封 K256/K384/K512 预测与真实 observation
 42 | 成本。fixed K384 为 Avg-mAP `88.1312%`、mAP@0.7 `76.2706%`、成本 `47110`。全部 704 个合法整视频
 43 | donor-recipient 候选均完成评价：
 44 | 
 45 | - Avg-mAP 最优变化：`+0.6942/-0.0436` 个百分点；
 46 | - mAP@0.7 最优变化：`-0.2359/+0.4970` 个百分点；
 47 | - 联合门余量最优变化：`+0.1474/+0.4898` 个百分点；
 48 | - 同时通过 `+0.8/+1.0` 门的候选：`0/704`。
 49 | 
 50 | 这次 falsifier 没有训练控制器、模型前向、梯度、bootstrap 或 official validation/test。它支持的结论只是：
 51 | 冻结 K384 检测器后，在当前嵌套三档密封预测动作空间中没有足够的 Avg-mAP/高 tIoU 联合开发集 headroom。
 52 | 它不证明动态预算一般问题失败，不证明 Top-K 不可微是根因，不证明控制器过拟合，也不证明分类天然需要低预算、
 53 | 边界定位天然需要高预算。
 54 | 
 55 | 此前 Pro 已对这一旧动作空间裁决 `STOP`。Marginal-v1、cap-release、96-state 和 whole-video 分支继续只读，
 56 | 不得重跑、扩张、改门、训练旧 controller、选择后补 bootstrap 或访问 official test。
 57 | 
 58 | ## 用户提供的新 REVISE
 59 | 
 60 | 当前建议的新科学问题是：保持现有嵌套 K256/K384/K512 位置构造、Scout、物理时间逆映射、检测器结构、损失、
 61 | Soft-NMS、数据、评价器、checkpoint 规则和实际成本口径不变，只比较：
 62 | 
 63 | 1. 固定预算控制：训练只使用 K384；
 64 | 2. 多预算适应：训练时使用 K256/K384/K512，初始名义概率 `0.25/0.50/0.25`，再按短窗口折叠后的实际
 65 |    observation 数校准，使平均训练成本尽量匹配固定 K384。
 66 | 
 67 | 两臂必须从同一 H65 checkpoint 开始，匹配成功更新数、优化器、学习率日程、随机种子、可训练参数集合和最终
 68 | EMA 选择规则。第一轮明确不加入预算条件嵌入、蒸馏、Gumbel-Softmax、新 Scout/head/selector、DFT、Mamba、
 69 | Block Drop、CUDA/TensorRT 或跨数据集扩展。
 70 | 
 71 | 同一用户消息中的较早附件另有“预算原生 H65 选点 + 多预算训练”的提案；这会同时改变选点和训练分布。当前
 72 | 正文要求第一轮保留现有嵌套位置构造。请你独立裁决冲突；Codex 当前没有把两者合并。
 73 | 
 74 | 建议中的输出与门槛为：
 75 | 
 76 | - 分别报告 K256/K384/K512 的 Avg-mAP、mAP@0.3--0.7、proposal recall、起点/终点误差、proposal 数、
 77 |   NMS 前后假阳性和短/中/长动作结果；
 78 | - K384 安全门相对同更新数固定控制：`ΔAvg-mAP >= -0.2` 个百分点且
 79 |   `ΔmAP@0.7 >= -0.2` 个百分点；
 80 | - 在未参与参数学习和规则选择的训练侧开发集上重算等成本 whole-video oracle：
 81 |   `ΔAvg-mAP >= +0.8`、`ΔmAP@0.7 >= +1.0` 个百分点，且实际 observation 成本不高于固定 K384；
 82 | - oracle 若失败，停止当前 K256/K384/K512 动态转移路线；若 Avg-mAP 恢复而高 tIoU 仍失败，才另行讨论
 83 |   K、相邻物理间隔和局部采样密度条件，不在本轮预埋。
 84 | 
 85 | ## 必须由你独立冻结的未决项
 86 | 
 87 | 当前建议仍缺少两个会改变科学解释的关键决定，Codex 不得自行选择：
 88 | 
 89 | 1. **匹配训练日程。** 请在“从 H65 terminal checkpoint 进行两臂完全相同的短期继续训练，仅作机制诊断”与
 90 |    “从冻结起点进行匹配完整训练，形成论文级比较”之间作出唯一选择，并给出精确起点 checkpoint/state key、
 91 |    每臂成功更新数或轮数、训练阶段、优化器/学习率/EMA规则、随机种子、checkpoint 与中间验证用途。若你认为
 92 |    应采用另一种最小日程，请明确说明理由和完整数值。
 93 | 2. **独立训练侧开发划分。** 旧 40-video controller holdout 已被用于多轮 oracle 和规则裁决，不能被静默称为
 94 |    未参与规则选择。请给出一个不会使用 official test、且未参与本次参数学习或规则选择的确切方案：最好给出
 95 |    可复现的 ID 清单生成规则、seed、视频数量以及 train/dev 的用途与封存边界。若现有 200 个训练视频不足以
 96 |    同时满足该要求，请明确裁决可接受的替代证据设计，而不是让 Codex自行重切数据。
 97 | 
 98 | ## 你的返回合同
 99 | 
100 | 请输出一份可保存的科学裁决，包含：
101 | 
102 | 1. `SESSION_ASSERTION`：原样回显 nonce、Project ID、H65 base `04c35a3b...` 和诊断提交 `33e4ed...`。
103 | 2. `SCIENTIFIC_DECISION`：只选 `CONTINUE / REVISE / PIVOT / STOP` 之一，并说明旧 STOP 与新问题的边界。
104 | 3. `CAUSAL_ISOLATION`：判断第一轮应保留嵌套选点还是改预算原生选点；只允许冻结一个干预变量。
105 | 4. `TRAINING_FREEZE`：给出无歧义的两臂起点、训练长度、优化、随机性、checkpoint/EMA 与中间验证规则。
106 | 5. `DEVELOPMENT_SPLIT_FREEZE`：给出无歧义、可复现、无 official-test 泄漏的训练侧开发划分和使用边界。
107 | 6. `EVALUATION_AND_GATES`：确认或修订每档诊断、K384 安全门、等成本 oracle 门、停止规则和论文证据边界。
108 | 7. `CODEX_TASK_ORDER`：只在实验值得执行时，依次给出一个最小 Builder、一个独立 Critic、一个独立 Evaluator
109 |    的范围、禁止项、终态产物和失败返回条件。不要要求工作流平台、复杂合同代码、哈希体系或额外审计框架。
110 | 8. `NEXT_RETURN`：明确 Codex 必须把哪些实现与结果证据返回给你，才能进行下一轮科学裁决。
111 | 
112 | 没有新代码、训练或结果已经产生。不要把本 Prompt、用户建议或文档状态当作模型有效性证据。若关键上下文不足，
113 | 请直接冻结阻断条件；不要猜测。请在本轮独立完成科学取舍，不把路线选择交回 Codex。
```

### File: PAPER_PROGRESS.md
Lines: 1-126
```md
  1 | # DUCA 论文缩略报告
  2 | 
  3 | - 更新日期：2026-08-31
  4 | - 名称说明：DUCA 是本项目沿用的方法路线名称。
  5 | - 当前结论：固定 `K=384` 的任务状态时序 coreset 低于匹配均匀选择，`DUCA-Coverage-v1` 也未通过预注册中间机制门。冻结 H65 的三档边际预算诊断、96-state 联合邻域和最终 704-state 整视频枚举均未达到 `+0.8/+1.0` 联合门；Pro 对这一冻结检测器的旧动作空间作出的 `STOP` 继续有效。用户随后提供的最新 `REVISE` 在该边界之外冻结了一个新科学问题：保留现有嵌套 K256/K384/K512 位置构造，只比较固定 K384 训练与三档多预算检测器适应，检验跨预算表示不匹配这一尚未证实的假说。该路线目前仅完成设计；训练成功更新数和独立训练侧开发视频清单尚未冻结，没有新代码、作业或性能结果。
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
 22 | 固定 K 内的 coverage 和其后的三档预算转移均已结束。后者冻结 H65 Scout 与 detector，并从同一非连续 H65 priority sequence 构造 K256/K384/K512 的嵌套集合，在逐窗口、差分联合邻域和整视频跨视频三个层次搜索不增加真实 observation 成本的重分配。完整开发集 oracle 枚举没有产生预登记的 Avg-mAP 与高 tIoU 联合 headroom。Pro 因而关闭这一动作空间；它不是完整训练后的论文主模型，也不授权 Codex 自动恢复历史路线或自行选择新机制。
 23 | 
 24 | 最新 `REVISE` 将下一项机制检验限定为检测器的多预算适应：第一轮继续使用上述嵌套位置构造，只改变训练时检测器是否同时见到 K256/K384/K512。它不把输入分布不匹配当作已证实根因，也不同时加入预算条件嵌入、蒸馏、Gumbel-Softmax、新 Scout、Mamba、Block Drop 或部署优化。较早附件提出的“预算原生选点 + 多预算训练”与该单变量要求冲突，因而不进入第一轮。
 25 | 
 26 | ## 3. 与官方基线的真实差异
 27 | 
 28 | 共享的未修改 AdaTAD 基线使用官方代码 revision `01c58b9f2370e914150cf94d392208a4e211c053`、seed 42、60 个训练轮次和官方评估器，平均检测精度（Avg-mAP）为 `68.73`；论文公开锚点为 `69.03`。DUCA 只读引用这一共享复现，不重复训练官方 dense 模型。
 29 | 
 30 | DUCA 与官方 dense 模型的目标差异只应来自输入采样与预算：前者先以低成本语义模型选帧，再让 VideoMAE 只处理被选中的高分辨率帧。下游 ActionFormer 检测头、损失、NMS 和官方评估器原则上不变。历史 65.xx 或 66.xx 结果不属于官方 dense 复现，不能代替 `68.73`，也不能在协议不匹配时直接计算方法增益。
 31 | 
 32 | ## 4. 已完成实现与代码身份
 33 | 
 34 | - H65 干净复现使用历史 ASFormer 语义预测、确定性非均匀逐帧选择和固定 `K=384`。当前可审计的 30+60 训练参考冻结于源码 revision `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。
 35 | - PJST-D1 是当前物理时间表示候选的代码名称。它不改变选择器，只在 VideoMAE 首次二帧 tubelet 混合前校正导数分量所使用的时间间隔。匹配训练冻结于 clean revision `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`；只读重推理与统计入口冻结于 `7bd120f0d342bf175c97c365fba7cbd359df055e`。
 36 | - UVT 诊断分支为 `codex/duca-uvt-utility-value-20260819@df544c78ce515d925dc7019f106fce09a53c09f8`；Fovea/Query-Bridge 诊断分支为 `codex/duca-fovea-query-bridge-20260819@4ae5067100c4490c7110c00a1ad406230ba603cd`。它们与 H65/PJST 不是同提交匹配比较，不能用跨版本差值归因。
 37 | - 物理连续片段路线已完成真实训练，但性能明显下降，现作为负结果保存，不再是当前论文主线。
 38 | - 为支持下一次科学路线与代码主线裁决，当前模型、配置、启动器、测试和研究记忆已同步到 `codex/duca-research-sync-20260828`；代码库存固定提交为 `5136011ed57df8a639427a633a488a592ba95924`。该快照用于检查实现重叠与历史分支，不是经过运行前检查的实验候选，也不能替代各实验的 clean revision。
 39 | - 先前的动态窗口预算候选冻结于 `codex/duca-semantic-budget-matched-20260828@36d75c146492a38eb8966c66ff6b2881938cf3c6`。它在取得效能结果前被后续科学裁决暂缓，保留为下一阶段的实现参考，不能作为动态预算证据。
 40 | - 当前原生 tubelet 时序 coreset 候选位于独立干净分支 `codex/duca-native-tubelet-coreset-20260828`，冻结提交为 `b33391126eac05e3353d322b973dda91741f0732`。实现包含固定 192 个原生 tubelet、任务状态驱动选择、端点/空洞覆盖、低分辨率上下文回收、物理时间重建、严格匹配的均匀对照、配置、聚焦测试与 N16R4 启动器。N16R4 环境的 20 项相关测试通过，独立只读审查未发现阻断性缺陷；两臂均完成 60 轮训练和官方验证，但结构化证据保存失败。
 41 | - 窗口级动态预算候选位于独立干净分支 `codex/duca-dynamic-native-tubelet-budget-20260829`，冻结提交为 `d127c2b2ceea7ff8a6932aa4a1925e1ff86cf610`。实现按视频生成需求排序和 16/20/24-clip 预算，在各预算内执行确定性均匀 tubelet 选择，并按真实 clip 数分组调用 VideoMAE；较短预算不会在重型骨干前补齐到 24。短窗口若不能容纳分配预算会明确退出。Python 编译、启动器语法、纯启动器测试和独立静态审查已通过；尚无运行前检查或正式实验结果。
 42 | - 当前 `DUCA-Coverage-v1` 候选位于独立干净分支 `feature/duca-coverage-only-v1-20260829`，当前提交为 `048143124e2a36a76575200ae17d6f42ec79ea3a`，基于 H65 正式提交 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。实现新增固定预算设施位置选择器，并提供 matched H65 对照、真实训练样本无标签重放门、60 轮配置、恢复合同和 N16R4 启动器。修正后的 PRE_RUN Job `1261679` 已执行 27 项测试和 200 个真实 training 样本重放，但因预注册覆盖/空洞干预条件未满足而在 smoke 前停止。代码已同步 GitHub并部署到 N16R4 干净目录 `/data/run01/sczc063/yuzibo/duca_coverage_v1_04814312_20260830`。
 43 | - 最新 Pro 裁决指定从 H65 clean revision `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 建立 `feature/duca-marginal-budget-v1-20260830`。吸收短窗口修订后的实现为 `be5bb8033c0b11c628394d268c1923ab398c04ed`；测试合同修复为 `f87555f7da362fe1a20d4ca08f7a68c975ed8280`。该提交完成 K384、K256 和 K512 冻结反事实产物后，汇总器因把换行文本 block-list 当作 JSON 读取而退出。最新提交 [`f67d96fdf68a295eaa7f678f3dfc125530828889`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f67d96fdf68a295eaa7f678f3dfc125530828889) 只为官方评估器生成确定性的 JSON block-list 适配文件并加入回归测试；没有修改模型、分配器、预测、损失、数据、NMS、指标或科学门槛。恢复作业没有重跑三个冻结 producer 阶段，只在当前 clean commit 上重做运行前身份核验并汇总既有产物。producer 产物保留其 `f87555f7...` 来源，汇总与最终结果绑定 `f67d96fd...`；两者的配置、checkpoint、annotation、类别映射、预训练权重及其哈希一致，这一跨提交来源会在后续 Pro 材料中明确披露。
 44 | - Pro 对灰区的后继实现位于 [`feature/duca-marginal-cap-release-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-falsifier-v1-20260831)，精确提交为 [`d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f)。它只增加独立的 `max_changed_fraction=1.0` 只读汇总入口和聚焦测试，默认 `0.5` 汇总路径、三档 producer、模型、数据、NMS 与门槛均不变。N16R4 的 14 项聚焦测试和独立 Critic 已通过。
 45 | - 最新的联合邻域诊断实现位于 [`feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831)，精确提交为 [`46812facc8773d9b4a9c21833cbe397c8aaa5a2d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d)。它只修改 probe runner 与聚焦测试；`dynamic_budget.py` 相对父提交逐字不变。实现从密封分配和真实 observation 成本自动导出 8 个最小合法转移、6 个净转移组及 96 个唯一联合状态，没有为多解视频硬编码配对。N16R4 上 16 项聚焦测试、23 项既有回归测试和独立 Critic 均通过。
 46 | - 整视频最终 falsifier 位于 [`feature/duca-whole-video-consistent-budget-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831)，权威公开提交为 [`33e4ed137c33eef07f0452b44506a6993bdf7535`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535)。父提交 `c27d77...` 的首次 PRE_RUN `1262147` 暴露密封 proposal 被额外排序、从而改变 Soft-NMS 并破坏锚点复现的确定性证据错误；`33e4ed...` 只恢复密封 producer 原始顺序并增加回归测试，不改变候选、成本、预测、NMS、评估器或三档预算。28 项聚焦测试与独立 Critic 通过。修正后 PRE_RUN `1262161` 复现全部锚点；唯一恢复 Job `1262190` 完成 704 个候选。终态 JSON SHA-256 为 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`，通过候选数为零。Pro 随后使用该最新 GitHub 身份作出项目级 `STOP`；该分支与 Marginal-v1、cap-release、96-state 分支均只作为负证据读取。
 47 | - 新设计的多预算检测器适应实验以 H65 干净提交 `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 为模型基座，并且只允许从 `33e4ed...` 移植真实变长执行、packet 对齐、实际 observation 计数、K384 parity、whole-video 评价和原始生成顺序保持。当前尚未建立实现分支或提交；`33e4ed...` 不是新模型的科学基座。
 48 | 
 49 | 项目根目录承担多路线协调，工作区可能包含尚未归档的修改。论文实验身份必须引用上述独立 clean revision，而不能用项目根当前 `HEAD` 代替。
 50 | 
 51 | ## 5. 正式实验协议
 52 | 
 53 | 正式比较使用 THUMOS14 规范数据入口、官方训练与验证划分、同一检测器、损失、NMS、评估器和预先登记的模型选择规则。当前已有实验主要使用 seed `3407`；单种子结果不能支持稳定性或显著性结论。
 54 | 
 55 | 当前与后续实验矩阵按以下顺序组织：
 56 | 
 57 | 1. 只读引用共享的官方 dense AdaTAD 结果；
 58 | 2. H65 固定 `K=384` 的 30+60 参考训练；
 59 | 3. PJST-D1 固定/重放选择结果的物理时间表示 OFF/ON 配对已完成点估计，但平均点差为负且配对区间缺失；它不再是当前优先开发项；
 60 | 4. 固定 `K=384` 的原生 tubelet 均匀选择与任务状态驱动时序 coreset 已完成；细粒度 coreset 候选因一致的负向点估计终止；
 61 | 5. 同一 H65 代码基座上的 matched allocation control 与 `DUCA-Coverage-v1` 已完成真实训练样本无标签重放门；该门未通过，因此两个 60 轮完整训练臂没有启动。当前对照实现是预算校准系统采样，而冻结设计曾概括为 Top-K；在 Pro 裁决该基线身份和失败机制前不得重提正式训练。
 62 | 6. 冻结 H65 的 K256/K384/K512 反事实边际预算实验、cap-release 诊断和 capped→released 差分邻域的 96 个联合状态枚举均已完成。50% 上限 oracle 的 Avg-mAP/mAP@0.7 增益为 `+0.726/+0.729` 个百分点；解除上限后反而降为 `+0.427/+0.450`。96-state 中没有状态同时达到 `+0.8/+1.0`；当前加性 Marginal-v1 及其本次视频级联合效用修复均按冻结规则停止。
 63 | 7. 整视频一致预算的跨视频单次转移 oracle 已完成：donor 视频所有窗口请求 K256，recipient 视频所有窗口请求 K512，其余视频保持 K384；候选总实际 observation 成本不超过固定 K384 的 `47110`。候选集合在读取标签或指标前完整生成，随后仅复用密封预测和相同评估器。1560 个有序对中 704 个合法候选完成评估，没有候选通过联合门；没有执行模型前向、训练、bootstrap 或 official test。
 64 | 8. 新设计的单变量实验比较固定 K384 训练与 K256/K384/K512 多预算检测器适应。两臂必须从同一 H65 checkpoint 开始，并匹配成功更新数、优化器、学习率日程、随机种子、可训练参数集合和最终 EMA 规则。候选的初始名义抽样概率为 `0.25/0.50/0.25`，但须按短窗口折叠后的实际 observation 成本校准。确切更新数和独立开发划分尚未冻结，所以当前不进入实现或训练。
 65 | 
 66 | 主要指标为 tIoU 0.3、0.4、0.5、0.6、0.7 下的 mAP，以及五个阈值的平均值；同时报告短动作、边界定位和完整端到端计算成本。任何计算节省主张都必须来自实际执行的 VideoMAE 工作量和相同硬件条件下的测量，不能由 padding 后的名义帧数推断。
 67 | 
 68 | 完整训练至少每 5 个训练轮次保存一次可恢复的 PyTorch `.pth` 检查点；如果未修改的官方配置保存更频繁，则保留官方间隔。恢复包应包含模型、指数移动平均模型、优化器、学习率调度器、混合精度缩放器、训练轮次或更新计数及随机状态，并至少保留最近三个有效恢复点、预定义里程碑和最终检查点。最终结果使用预先登记的最终模型或最终指数移动平均模型，不能根据中间验证结果事后挑选。
 69 | 
 70 | ## 6. 已取得的性能结果
 71 | 
 72 | 除特别说明外，Avg-mAP 是 tIoU 0.3 至 0.7 五个阈值下 mAP 的平均值。
 73 | 
 74 | | 实验 | Avg-mAP | mAP@0.7 | 证据边界 |
 75 | |---|---:|---:|---|
 76 | | 共享官方 dense AdaTAD | 68.73 | 未在本报告重复摘录 | 一次共享复现；论文公开 Avg-mAP 为 69.03 |
 77 | | H65 30+60 | 65.13 | 43.31 | 单种子、完整训练；当前 H65 参考 |
 78 | | H65 20+40 | 62.46 | 39.94 | 同 seed 的训练压缩诊断；比 30+60 低 2.66/3.37 点 |
 79 | | H65 30+30，AM-RPCH25 | 63.22 | 41.25 | 只改变第二阶段学习率日程 |
 80 | | H65 30+30，LongCosine-H6000 | 63.56 | 41.01 | 更慢衰减仍未恢复 30+60 |
 81 | | RankPack K384 | 61.57 | 37.10 | 单种子物理时间表示对照 |
 82 | | TrueTime K384 | 62.19 | 37.89 | 相对 RankPack 为 +0.62/+0.79 点；尚无配对区间 |
 83 | | PJST-D1 OFF | 65.063 | 43.646 | 211/211 视频，冻结 H65 选择结果 |
 84 | | PJST-D1 ON | 64.591 | 43.769 | 211/211 视频；相对 OFF 为 -0.472/+0.123 点 |
 85 | | 原生 tubelet uniform K384 | 64.13 | 42.45 | 单 seed；60 轮训练和官方验证完成；结构化指标文件未封存 |
 86 | | 原生 tubelet coreset K384 | 62.81 | 40.56 | 单 seed；相对匹配 uniform 为 -1.32/-1.89 点；结构化指标文件未封存 |
 87 | | 连续片段 FZ | 49.89 | 29.68 | 完整单种子训练；明显负结果 |
 88 | | 连续片段 JT | 47.24 | 26.52 | 完整单种子训练；联合训练未恢复性能 |
 89 | 
 90 | PJST-D1 两臂各产生 422,000 条预测，视频集合一致，重推理逐项复现了原始点估计。预先登记的 10,000 次整视频配对自助法在任何抽样前退出，因为统计程序指向了错误的预测文件目录。因此当前只有点估计，没有置信区间；不能据此宣布总体效应显著为负，也不能把 mAP@0.7 的 `+0.123` 点解释为真实收益。
 91 | 
 92 | UVT 的 legacy/geometry/geometry+EMA 三臂 Avg-mAP 为 `57.35/55.93/55.92`。Fovea/Query-Bridge 第一波中最佳 `query_cycle` 为 `54.67`。这些结果来自不同代码版本与实验合同，只能说明相应首版设计未恢复 H65 性能，不能用来量化某一个组件的因果影响。
 93 | 
 94 | 历史 `65.3857` 是 H65 语义间接非均匀逐帧选择的 30+60 诊断结果；`65.696` 来自改变物理检测网格的探索实现。后者同时改变了检测器时间几何，因此不是与官方原生检测器严格匹配的输入采样对照。当前干净 H65 复现 `65.13` 是更适合作为后续同代码、同训练协议比较的参考。
 95 | 
 96 | 当前还没有与主路线匹配、可用于论文的完整端到端成本结果，也没有动态预算保持性能或提高效率的实验证据。`DUCA-Coverage-v1` 已完成代码实现、独立审查和真实 training 数据运行前门；该门给出不满足预注册干预条件的诊断结果，但没有进入训练，因此不是 mAP 或成本结果。
 97 | 
 98 | `DUCA-Marginal-v1` 的冻结 detector 诊断在训练侧 40 个 utility holdout 视频、124 个窗口上比较了固定 K384 与使用真实反事实效用的等预算 oracle。固定臂 Avg-mAP/mAP@0.7 为 `88.131/76.271`，50% 上限 oracle 为 `88.857/77.000`。解除上限后，分配从 K256/K384/K512 的 `11/102/11` 变为 `17/90/17`，但结果降为 `88.559/76.721`；相对固定臂只有 `+0.427/+0.450` 个百分点。总 observation 预算仍精确为 `47110`。这里的百分数只描述训练侧 controller holdout，不可与 official validation/test 表直接比较。强 headroom 门未通过，因此 utility predictor、正式测试、配对区间和端到端成本均未运行。
 99 | 
100 | 随后只在 capped 与 released 分配不同的 12 个窗口上枚举了全部 96 个逐视频等成本联合状态。最佳 Avg-mAP 状态相对固定 K384 为 `+0.733` 个百分点，但 mAP@0.7 仅 `+0.479`；最佳 mAP@0.7 状态为 `+0.549/+0.934`；按两项联合门最优的状态为 `+0.554/+0.933`。没有状态同时满足 `+0.8/+1.0`，也没有单个最小合法转移同时改善 Avg-mAP 与 mAP@0.7。该诊断没有执行模型前向、训练、official test 或 bootstrap；从 96 个开发集状态中事后选出的最优状态不能当作可部署策略或论文主结果。
101 | 
102 | 整视频一致预算的最终开发集 falsifier 在相同固定 K384 锚点 `88.1312%/76.2706%`（Avg-mAP/mAP@0.7）和真实成本 `47110` 下完成 704 个合法状态。Avg-mAP 最优状态的变化为 `+0.6942/-0.0436` 个百分点，mAP@0.7 最优状态为 `-0.2359/+0.4970`，联合门余量最优状态为 `+0.1474/+0.4898`；通过候选数为零。该结果只属于训练侧 controller holdout 的事后 privileged oracle 证伪，没有模型前向、训练、官方验证/测试、配对区间或可部署策略含义。
103 | 
104 | ## 7. 结果解释与已停止的方向
105 | 
106 | - 20+40 训练压缩、AM-RPCH25 和 LongCosine-H6000 均未恢复 H65 30+60 的性能。现有证据说明简单压缩预热或只修改第二阶段学习率尾部不足以保持性能；它不否定 H65 的语义间接选帧机制。
107 | - 连续 16 帧片段采样在真实训练中造成大幅定位损失，联合训练也未恢复，因此该采样单元不再作为当前主路线。这个结果不否定低成本语义侦察或物理时间一致性的一般问题。
108 | - TrueTime 相对 RankPack 有小幅单种子提升，但证据不足以形成论文主张。
109 | - PJST-D1 的当前点估计没有显示平均性能收益；缺少配对置信区间意味着总体效应仍未完成统计裁决。统计程序的路径错误是证据生成失败，不是模型的科学失败。
110 | - UVT 与 Fovea/Query-Bridge 同时改变了选择分数、预算证据或训练信息流，且缺少与 H65 同提交的严格隔离，因此其性能下降不能归因于单一组件。
111 | 
112 | ## 8. 当前证据缺口与下一动作
113 | 
114 | THUMOS14 原始视频、注释、类别映射、VideoMAE-S 预训练权重、H65 Stage-1 侦察器检查点和共享官方 AdaTAD 结果均已核验。PJST-D1 的 OFF/ON 完整推理已经结束。
115 | 
116 | 当前固定预算归因已经得到负向点估计：在完全相同的 `K=384` 高分辨率帧预算、训练日程和检测器下，任务状态驱动 coreset 没有优于原生 tubelet 均匀选择，并在 tIoU 0.5 至 0.7 下降 `1.54/2.03/1.89` 个百分点。这个结果优先要求分析端点覆盖、最大空洞、跨 tubelet 打包、低分辨率上下文回收和选择分数是否共同损害重型表示，而不能直接把固定预算重命名为最终方法。
117 | 
118 | PJST-D1 的配对区间仍属未完成证据，但它不会改变当前路线，因此不作为当前任务补齐。原生 tubelet 两臂已从干净 H65 基座完成实现、N16R4 聚焦测试、独立审查和 60 轮训练。uniform `1260184` 与 coreset `1260185` 都写出 epoch-59 检查点并完成官方 211 视频评估，日志点估计分别为 `64.13%` 和 `62.81%`。两臂随后因同一个证据封存错误退出：配置没有保存预测，结构化指标入口因此拒绝写出 `metrics_epoch59_ema.json`。当前没有配对区间或成本结果；训练成功、日志点估计和结构化证据缺失必须分别陈述。
119 | 
120 | Pro 冻结的 96-state 联合 mAP 邻域诊断已经在公开提交 `46812fac...` 上完成。原 fixed/capped/released 三结果复现误差为 `0.0` 个百分点，96 次评估保持逐视频预算和全局成本 `47110`；没有状态通过 `+0.8/+1.0` 联合门，也没有单个最小合法转移同时改善两个门指标。Pro 已据此最终裁决 `STOP`：现有加性 Marginal-v1 及本次邻域修复关闭，不再重跑、改门、补 bootstrap、训练 utility head 或访问 official test。当前分支只作为负证据读取；未来若重新研究动态计算，必须由 Pro 以新的机制假设和独立任务启动，不能作为 Marginal-v1 的恢复。
121 | 
122 | 项目级 Pro 对旧三档预算转移动作空间的 `STOP` 继续有效。最新 `REVISE` 已把“检测器只适应 K384”改写成一项新的、边界之外的可证伪实验，但仍只是假说。下一步不是恢复旧 oracle 或直接训练控制器，而是先由 Pro 冻结两臂相同的额外成功更新数（或明确采用匹配完整训练）以及未参与参数学习和规则选择的训练侧开发视频清单。冻结后才建立 Builder；独立 Critic 必须验证只有训练预算分布发生变化，Evaluator 再执行匹配训练和预登记输出。K384 安全门为相对同更新数控制的 Avg-mAP 与 mAP@0.7 均不低于 `-0.2` 个百分点；等成本 oracle 继续门仍为 `+0.8/+1.0` 且实际成本不高于固定 K384。
123 | 
124 | ## 9. 可发表性边界
125 | 
126 | 目前可以写入论文的事实是：H65 的 30+60 训练参考明显优于已测试的 60 轮压缩日程；连续片段采样是明确负结果；PJST-D1 的匹配点估计没有平均收益；原生 tubelet coreset 的单种子点估计比匹配均匀选择低 `1.32` 个百分点；当前 96-anchor Coverage 干预没有达到预注册中间机制目标。DUCA-Marginal 的训练侧 oracle 在 50% 上限下仅有灰区增益，解除上限后增益进一步降至 `+0.427/+0.450` 个百分点；其 96-state 联合差分邻域和最终 704-state 整视频单次转移 falsifier 均没有达到 `+0.8/+1.0` 的预注册门，因此当前冻结动作空间没有足够开发集 headroom。Pro 将其定位为可用于未来论文补充失败分析的内部终态证据，而不是独立 CVPR 主结果。它没有证明 Scout 能预测预算效用，也没有 official test、统计区间、可部署 controller 或端到端成本结果。不能把该停止扩展为所有 coverage、动态预算、budget-conditioned training、内部 token/层级条件计算或物理时间方法无效，也不能声称 DUCA 有效、优于 dense，或已经获得性能—成本联合优势。
```

### File: research-wiki/experiments/duca-whole-video-consistent-budget-v1.md
Lines: 1-137
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
 73 | 视频、修改预算档位或降低门槛。该结果随后已经返回 Pro；下文记录其独立终态裁决，Codex 未从负结果自行选择
 74 | 替代路线。
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
118 | 
119 | ## Pro 终态裁决
120 | 
121 | Fresh exact DUCA Project 对话完整绑定了最新公开代码：
122 | 
123 | - 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
124 | - 实际远端分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>
125 | - 精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
126 | - runner：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>
127 | - 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py>
128 | - 未修改的三档 allocator：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>
129 | 
130 | Pro 裁决为 `STOP`。在当前 THUMOS14 训练侧 holdout、冻结 H65 detector/priority sequence、密封三档预测、
131 | 真实 observation 成本与现有资源边界内，不再开发在窗口或视频之间转移 K256/K384/K512 预算的方法；不再
132 | 扩大状态空间、改门、训练 controller、补 bootstrap 或访问 official validation/test。该停止不否定所有动态
133 | 计算、Scout、预算空间、budget-conditioned training、内部 token/层级条件计算或其他数据集。
134 | 
135 | 最强诚实结论是：704 个合法整视频单 donor-recipient 状态中没有状态同时达到预登记联合门，且 Avg-mAP 与
136 | 高 tIoU 的最优状态分离。最强未验证解释是当前 H65 优先序列与仅在 K384 下训练的 detector 缺少跨预算兼容、
137 | 单调且边界敏感的表示。当前没有新的 Builder、Critic、Evaluator 或 Slurm 任务；该分支只作为负证据保留。
```

### File: research-wiki/experiments/duca-multi-budget-detector-adaptation.md
Lines: 1-53
```md
 1 | ---
 2 | type: experiment
 3 | status: designed
 4 | updated: 2026-08-31
 5 | project: DUCA
 6 | ---
 7 | 
 8 | # DUCA 多预算检测器适应
 9 | 
10 | ## 科学问题
11 | 
12 | 在保持 H65 的当前嵌套 K256/K384/K512 位置构造、Scout、检测器结构和评价协议不变时，仅让检测器在三种预算
13 | 分布上共同训练，能否恢复每档预算自身的检测质量，并重新建立匹配真实 observation 成本的 Avg-mAP 与高时间
14 | 交并比联合 oracle headroom？
15 | 
16 | 这是位于此前项目级 `STOP` 边界之外的新机制检验。旧结论仍成立：冻结 K384 检测器后进行三档预算转移没有
17 | 达到预登记联合门。新实验不重跑旧 oracle，也不把跨预算表示不匹配预先写成已证实根因。
18 | 
19 | ## 单一干预
20 | 
21 | - 固定预算控制只在 K384 上训练。
22 | - 候选在 K256/K384/K512 上共同训练；名义抽样概率从 `0.25/0.50/0.25` 开始，并按短窗口折叠后的实际
23 |   observation 成本校准。
24 | - 两臂使用同一 H65 起点、相同成功更新数、优化器、学习率日程、随机种子、可训练参数集合和最终指数移动平均
25 |   模型选择规则。
26 | 
27 | 第一轮保留当前嵌套位置构造。不得同时改成预算原生 H65 采样；否则将同时改变选点和训练分布，无法把结果归因
28 | 于检测器预算适应。
29 | 
30 | ## 代码边界
31 | 
32 | - 模型基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`。
33 | - `33e4ed137c33eef07f0452b44506a6993bdf7535` 仅提供真实变长执行、packet 对齐、实际 observation 计数、K384
34 |   parity、whole-video 评价和原始生成顺序保持等已验证功能。
35 | - 禁止预算条件嵌入、蒸馏、Gumbel-Softmax、新 Scout/head/selector、DFT、Mamba、Block Drop 和 TensorRT。
36 | 
37 | ## 决定性输出
38 | 
39 | 1. K256、K384、K512 各自的 Avg-mAP、mAP@0.3--0.7、proposal recall、起终点误差、proposal 数、NMS 前后
40 |    假阳性和动作长度分层结果。
41 | 2. 多预算模型在 K384 上相对同更新数固定控制满足 `ΔAvg-mAP >= -0.2` 和 `ΔmAP@0.7 >= -0.2` 个百分点。
42 | 3. 在未参与训练和规则选择的训练侧开发划分上，等成本 oracle 同时满足 `+0.8/+1.0` 个百分点且实际成本不高于
43 |    固定 K384。
44 | 
45 | 若第三项不通过，则停止当前三档动态转移路线。若仅高时间交并比仍失败，后续是否增加预算和采样密度条件必须由
46 | Pro 另行冻结，不能在本实验中预埋。
47 | 
48 | ## 执行前未决项
49 | 
50 | - 机制诊断采用多少相同额外成功更新，或是否直接进入匹配完整训练；
51 | - 未参与参数学习和规则选择的训练侧开发视频 ID 清单。
52 | 
53 | 这两项未冻结前不建立 Builder 分支、不提交 PRE_RUN 或训练。当前没有实验结果。
```

### File: research-wiki/sources/2026-08-31-duca-multi-budget-detector-adaptation-revise.md
Lines: 1-75
```md
 1 | # DUCA 多预算检测器适应：REVISE 裁决记录
 2 | 
 3 | 日期：2026-08-31  
 4 | 来源：用户在当前 Codex 对话中提供的裁决文本  
 5 | 来源身份边界：按用户要求作为当前科学裁决摄取；本记录未独立核验其外部生成会话或模型身份  
 6 | 
 7 | ## 唯一裁决
 8 | 
 9 | `REVISE`。
10 | 
11 | 此前的 `STOP` 继续约束冻结 K384 检测器、再在窗口或视频间转移 K256/K384/K512 密封预测的旧动作空间。
12 | 本裁决没有恢复 Marginal-v1、cap-release、96-state 或 whole-video 分支，而是在该边界之外提出一个新的、单变量的
13 | 机制检验：保持当前嵌套预算位置构造和评价协议不变，只让同一检测器在 K256、K384、K512 三种输入预算上共同
14 | 适应，检验此前缺少联合 headroom 是否主要来自检测器只见过 K384 的训练分布。
15 | 
16 | 跨预算表示不匹配仍是最强机制假说，不是已经证明的根因。当前负结果也不能被解释为动态预算一般问题失败、
17 | 不可微选择失败、控制器过拟合，或分类天然需要低预算而边界定位天然需要高预算。
18 | 
19 | ## 与较早附件版本的冲突处置
20 | 
21 | 同一用户消息附带的较早文本还建议把每个 K 改成预算原生的 H65 系统采样，并同时进行多预算训练。当前裁决正文
22 | 明确要求第一轮保留现有嵌套 K256/K384/K512 位置构造，只改变检测器训练分布。两者不能合并；当前裁决正文在
23 | 冲突处优先，预算原生选择不进入第一轮。
24 | 
25 | ## 唯一下一实验
26 | 
27 | ### 科学问题
28 | 
29 | 当前三档预算没有形成有效整视频联合 headroom，主要是否因为检测器只适应 K384，而未适应 K256/K512？
30 | 
31 | ### 代码与功能边界
32 | 
33 | - 模型基座：H65 干净提交 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。
34 | - `33e4ed137c33eef07f0452b44506a6993bdf7535` 只作为已验证功能来源，允许移植真实变长 VideoMAE 执行、
35 |   packet 对齐、实际 observation 计数、K384 bit-exact parity、whole-video 评价和原始生成顺序保持。
36 | - `33e4ed...` 不是新模型的科学基座。
37 | - 第一轮冻结现有嵌套 K256/K384/K512 位置构造、Scout、物理时间逆映射、检测器结构、分类与回归损失、
38 |   Soft-NMS、数据划分、官方评价器、checkpoint 选择规则和实际 observation 成本口径。
39 | 
40 | ### 匹配训练臂
41 | 
42 | 1. 固定预算控制：训练只使用 K384。
43 | 2. 多预算适应：训练时使用 K256/K384/K512；初始名义概率为 `0.25/0.50/0.25`，但必须依据短窗口折叠后的
44 |    实际 observation 数校准，使平均真实训练成本尽量匹配固定 K384。
45 | 
46 | 两臂必须从同一 H65 checkpoint 开始，并使用相同成功更新数、优化器、学习率日程、随机种子、可训练参数集合和
47 | 最终指数移动平均模型选择规则。
48 | 
49 | ### 当前禁止项
50 | 
51 | 第一轮不加入预算条件嵌入、知识蒸馏、Gumbel-Softmax、新 Scout 或分类/边界双头、新 selector、DFT、Mamba、
52 | Block Drop、CUDA/TensorRT 优化，也不扩展到其他数据集。
53 | 
54 | ## 决定性输出与门槛
55 | 
56 | - 分别报告 K256、K384、K512 的 Avg-mAP、mAP@0.3--0.7、proposal recall、起点误差、终点误差、proposal 数、
57 |   NMS 前后假阳性以及短、中、长动作结果。
58 | - K384 安全门相对同更新数固定 K384 控制为：`ΔAvg-mAP >= -0.2` 个百分点且
59 |   `ΔmAP@0.7 >= -0.2` 个百分点。
60 | - 在未参与参数学习和规则选择的训练侧开发集上重新计算等成本 whole-video oracle；继续门为
61 |   `ΔAvg-mAP >= +0.8` 个百分点、`ΔmAP@0.7 >= +1.0` 个百分点且实际 observation 成本不高于固定 K384。
62 | - 若 oracle 不通过，停止当前 K256/K384/K512 动态转移路线。若 Avg-mAP 恢复而 mAP@0.7 仍失败，才允许 Pro
63 |   另行裁决是否进行第二个单变量实验，为检测器提供 K、相邻物理时间间隔和局部采样密度条件。
64 | 
65 | ## 尚未冻结、因此阻断实现的事项
66 | 
67 | 当前文本允许两种训练定位：从 H65 terminal checkpoint 做两臂同更新数的短期继续训练，仅作为机制诊断；或在
68 | 设计冻结后做匹配完整训练，形成论文比较。它没有给出唯一成功更新数或训练轮数。它也没有给出新的、未参与参数
69 | 学习和规则选择的训练侧开发视频 ID 清单。上述两项会改变实验解释与执行成本，必须在 Builder 开始前由 Pro 明确
70 | 冻结；Codex 不自行沿用旧 40-video oracle 划分，也不把较早附件中的 `30+60` 静默并入当前裁决。
71 | 
72 | ## 当前状态
73 | 
74 | 该路线处于“完成科学设计、尚未实现”。没有新分支、Builder、Critic、Evaluator、PRE_RUN、Slurm 作业或性能
75 | 结果。文档记录不构成机制有效性证据。
```

## Answer

以下为可直接保存并交付 Codex 的冻结版科学裁决。

# DUCA 多预算检测器适应：科学裁决与执行冻结

## 1. `SESSION_ASSERTION`

**Nonce：`DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`**

* ChatGPT Project ID：`g-p-6a91061f789881918ccd8357ca3d6c92`
* 仓库：`yuzbo/OpenTAD_C3_CoarseClean_20260702`
* 新实验的唯一模型与代码基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
* 只读 whole-video 诊断提交：`33e4ed137c33eef07f0452b44506a6993bdf7535`

`04c35a3b...` 是 H65 训练身份。它的 Stage‑1 合同为 30 个训练轮次、每轮 100 次成功更新、终态 `epoch_29/state_dict_ema`；Stage‑2 合同明确使用一个重新开始的 6,000 次成功更新日程，并从 Stage‑1 EMA 初始化。

`33e4ed...` 不是新模型基座。该提交只修改 whole-video runner 和聚焦测试，把密封 proposal 的重放顺序恢复为 producer 原始顺序，避免 Soft‑NMS 分数并列时因额外排序产生确定性漂移；它没有修改模型、预测值、预算候选、损失、NMS 或评价器。

本轮材料权重冻结如下：

1. 用户本轮提供的 2026‑08‑31 prompt、`PAPER_PROGRESS.md`、两个实验页和来源记录是当前科学状态；
2. 2026‑08‑28 材料包已完整读取，只用于代码库存边界、角色约束、原始检查点证据和历史谱系，不覆盖本轮更新后的停止边界与科学问题。库存提交不是实验身份，Pro 负责科学选择，Builder、Critic、Evaluator 只执行冻结任务。  
3. PJST 原始评估材料用于确认 H65 Stage‑1 检查点路径、epoch、SHA‑256 和状态键；旧项目 prompt、README、旧研究状态和历史版本注册表仅作历史背景。     

---

## 2. `SCIENTIFIC_DECISION`

**唯一裁决：`CONTINUE`**

批准执行一次有界、单变量、完整匹配的多预算训练实验。

### 2.1 为什么值得执行

此前 704 个 whole-video 候选的终态负结果只检验了：

> 一个只在固定 `K=384` 协议下训练并冻结的模型，在密封的 `K256/K384/K512` 预测之间重新分配预算时，是否已经具有足够的开发集联合性能空间。

该实验没有重新训练 detector，没有执行梯度更新，也没有让重型表示在训练阶段见到 `K256` 或 `K512`。因此，“训练输入支持不足导致跨预算不兼容”仍是一个尚未接受直接检验、但可以通过单次匹配实验裁决的机制假说。

本轮批准不表示该假说已经成立，也不表示为了延续项目而放宽门槛。批准的理由只有一个：它改变了旧实验从未改变过的变量——**训练时的预算分布**——并能以两条完整匹配训练臂直接产生判别性证据。

### 2.2 旧 `STOP` 完整保留

以下旧路线继续永久只读：

* Marginal‑v1；
* cap‑release；
* 96-state 联合邻域；
* whole-video 704-state 分支；
* 旧 40-video controller holdout 上的密封预测与 oracle 结果。

不得：

* 重跑或扩大旧候选空间；
* 修改旧门槛；
* 加入第三个视频或组合多个 transfer；
* 训练旧 controller 或 utility head；
* 在旧结果中事后选择候选再补 bootstrap；
* 使用旧 40 个视频作为本轮“未参与规则选择”的开发集；
* 访问 official test；
* 把新结果写成旧 Marginal 路线的恢复。

新实验从 `04c35a3b...` 建立独立分支。旧分支唯一可移植的是已经验证的真实变长执行、packet 对齐、实际 observation 计数、K384 parity、whole-video 评价和原始 proposal 顺序保持。

---

## 3. `CAUSAL_ISOLATION`

### 3.1 冻结的唯一干预变量

第一轮继续使用当前**嵌套**的 `K256/K384/K512` 位置构造。

* `K256`、`K384`、`K512` 必须来自同一 H65 priority sequence；
* 不改成三个预算分别重新运行的“预算原生 H65 选点”；
* 不修改 Scout 输入、评分字段、优先级调制、覆盖下限、确定性累计采样或短窗口折叠；
* 不修改物理时间逆映射、VideoMAE 语义、Temporal Adapter、ActionFormer/AdaTAD 检测头、损失、Soft‑NMS 或评价器。

**唯一干预变量是：训练时送入相同 H65 模型的预算分布。**

两臂为：

1. **固定预算控制臂**

   * 每个训练窗口请求 `K384`；
   * 短窗口实际成本为 `min(valid_observations,384)`。

2. **多预算适应臂**

   * 每个训练窗口请求 `K256`、`K384` 或 `K512`；
   * 使用下文冻结的、按实际 observation 成本校准的概率；
   * 短窗口继续按 `min(valid_observations,K)` 折叠。

### 3.2 不进入第一轮的变量

明确禁止：

* 预算条件嵌入；
* 显式输入 `K`、相邻物理时间间隔或局部采样密度；
* 蒸馏；
* Gumbel‑Softmax；
* 新 Scout、新分类头或边界头；
* 新 selector 或预算原生位置构造；
* DFT、Mamba、Block Drop；
* CUDA、TensorRT 或其他部署优化；
* 新数据集；
* controller 训练；
* 根据开发结果改变预算概率。

### 3.3 可归因结论的精确边界

两臂使用完全相同的 H65 Stage‑2 可训练参数集合。不得只在候选臂冻结或解冻某些参数。

因此，正结果支持的是：

> 在相同 H65 架构、初始化、监督、优化和嵌套位置构造下，训练时接触多种预算输入可以建立更好的跨预算兼容性。

它不能被写成：

* “ActionFormer 检测头单独是根因”；
* “Scout 完全没有参与适应”；
* “不可微选点已被解决”；
* “动态预算一般问题已经成立”。

---

## 4. `TRAINING_FREEZE`

### 4.1 训练定位

选择：

> **从冻结的 H65 Stage‑1 终态进行两臂匹配的完整 Stage‑2 训练。**

拒绝从 H65 terminal detector 进行短期继续训练。

原因是：短期继续训练从一个已经被 `K384` 输入分布塑形的优化盆地出发。若候选失败，无法区分“多预算训练无效”和“学习率太小、更新不足或无法离开既有盆地”；若候选成功，也只能说明事后微调可部分修复，不能形成干净的完整训练比较。

### 4.2 唯一起点

两臂共同加载：

* 路径：
  `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
* 期望 epoch：`29`
* 状态键：`state_dict_ema`
* SHA‑256：
  `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`

该检查点身份在已有终态评估材料中已绑定；H65 Stage‑1 配置也明确指定 terminal `epoch_29/state_dict_ema`。

加载模型权重后，两臂都必须重新初始化：

* optimizer state；
* learning-rate scheduler state；
* AMP scaler；
* EMA accumulator；
* successful-update counter；
* `frame_selector._loss_weight_schedule_step`；
* 数据采样与随机增强随机状态。

不得从 H65 Stage‑2 terminal optimizer、scheduler 或 EMA 状态恢复。

### 4.3 训练长度

每臂：

* **恰好 6,000 次成功 optimizer update**；
* 非有限损失或 AMP 重放不计为成功更新；
* 两臂成功更新数必须完全相等；
* 以成功更新索引作为训练主时钟。

由于新 Stage‑2 训练集为 160 个视频，不能把“60 个实际数据轮次”静默当作与原 H65 相同的训练量。训练器应循环数据直到完成 6,000 次成功更新。

为保持 H65 原日程：

* 每 100 次成功更新定义一个 H65 **schedule epoch**；
* 总计 60 个 schedule epoch；
* warmup 5 个 schedule epoch，即前 500 次成功更新；
* Stage‑2 前后半程边界保持在第 3,000 次成功更新；
* 所有原来按 step 定义的 loss、policy 和 detector-gradient 日程保持原值。

H65 Stage‑2 原配置明确冻结 `max_updates=6000`、`duca_stage2_half_steps=3000`，并从 `state_dict_ema` 初始化后重置训练状态。

### 4.4 优化器、调度器与数值合同

两臂逐项相同：

* optimizer：`AdamW`
* 主学习率：`1e-4`
* weight decay：`0.05`
* VideoMAE 主体保持 H65 参数分组；基础 backbone 的冻结规则不变，adapter 学习率为 `2e-4`
* H65 Stage‑2 Scout 相关学习率：

  * coarse trunk：`1e-5`
  * action head：`2e-5`
  * transition scorer：`5e-5`
* scheduler：`LinearWarmupCosineAnnealingLR`
* warmup：前 500 次成功更新
* 总日程：6,000 次成功更新
* gradient clipping：global norm `1`
* AMP：开启
* EMA：开启
* batch size：每卡 `2`
* seed：`3407`
* DDP、unused-parameter、损失归一化和梯度累积规则不得因预算档位而改变。

这些数值来自 `04c35a3b...` 所继承的 H65/AdaTAD 配置；精确配置优先于任何文字摘要。

若 `K512` 在 batch size 2 下发生显存不足，只允许对**两臂同时**采用 microbatch 1、累计 2 个样本后完成一次 optimizer update。不得只降低候选臂 batch size，也不得改变总成功更新数。

### 4.5 随机性匹配

两臂必须共享：

* 视频与窗口采样次序；
* random truncation；
* resize/crop/flip/color augmentation；
* 模型初始化；
* dropout/drop-path 随机种子；
* optimizer update 次序。

候选臂的预算随机性使用独立命名空间，不得消耗数据增强或模型随机流：

`SHA256("DUCA-MBDA-BUDGET-v001|3407|<successful_update>|<sample_id>|<occurrence>")`

固定预算臂忽略该值并始终使用 `K384`。

### 4.6 多预算概率与训练成本校准

先在**160-video Stage‑2 训练集合**上重放完整的无标签训练窗口计划，按短窗口折叠计算：

* `μ256`：请求 K256 时的平均实际 observation；
* `μ384`：请求 K384 时的平均实际 observation；
* `μ512`：请求 K512 时的平均实际 observation。

不得读取动作类别、边界或检测指标。

保持中央预算概率固定为：

`p384 = 0.50`

保持两侧总暴露为：

`p256 + p512 = 0.50`

并唯一求解：

`p256 = 0.5 × (μ512 − μ384) / (μ512 − μ256)`

`p512 = 0.5 × (μ384 − μ256) / (μ512 − μ256)`

这样候选臂的期望实际 observation 成本等于固定 K384：

`p256·μ256 + p384·μ384 + p512·μ512 = μ384`

当 `μ384` 正好位于两端均值中点时，概率自然恢复为 `0.25/0.50/0.25`。

阻断条件：

* `μ512 == μ256`；
* 任一概率不在 `[0,1]`；
* 实际成本不是逐档单调；
* 概率是在读取开发标签或模型指标后调整。

6,000 次更新中的离散预算数量采用最大余数法从上述概率一次性确定，再由冻结哈希打乱。候选臂最终实际训练 observation 总量与控制臂的偏差必须报告；预运行时若超过 `0.5%`，只允许在不改变概率公式和不读取标签的情况下重新排列预算与样本 occurrence，使差值最小。仍超过 `0.5%` 时返回阻断，不开始正式训练。

### 4.7 checkpoint 与中间验证

保存点：

* 每 500 次成功更新保存可恢复 checkpoint；
* 强制保留 update 3,000；
* 保留最近三个有效 checkpoint；
* 强制保留 update 6,000 terminal checkpoint。

恢复包必须包含：

* model；
* `state_dict_ema`；
* optimizer；
* scheduler；
* AMP scaler；
* successful-update index；
* 数据 sampler 和全部随机状态。

**唯一结果模型：update 6,000 的 `state_dict_ema`。**

不得：

* 按中间 Avg-mAP 选择 checkpoint；
* early stopping；
* 事后选择 best validation EMA；
* 因某一预算曲线较好而修改概率或损失。

新的 40-video 开发集在两臂 terminal checkpoint 和配置全部密封前不得进行任何有标签评价。训练中可以查看损失、梯度、数值有限性和无标签实际成本，但不能查看开发 mAP。

---

## 5. `DEVELOPMENT_SPLIT_FREEZE`

### 5.1 为什么不能继续使用旧 40 个视频

旧 40-video controller holdout 已经参与：

* capped oracle；
* released oracle；
* 96-state 联合邻域；
* 704-state whole-video 枚举；
* 门槛与路线裁决。

因此它不再是未参与规则选择的开发集。

本轮将这 40 个视频明确归入训练可用集合，不再承担评价功能。

### 5.2 可复现划分规则

输入：

1. THUMOS14 规范训练侧的 200 个视频 ID；
2. 旧 whole-video 实验实际读取的 40 个 `holdout_videos` ID；
3. split seed：`20260831`。

步骤：

1. 从 annotation 中读取规范训练侧视频 ID，去重后按完整视频 ID 字典序排序；

2. 要求数量严格为 `200`；

3. 读取旧 40-video manifest，要求数量严格为 `40`，且全部属于上述 200；

4. 从 200 中移除旧 40，得到 160 个新的开发候选；

5. 对每个候选视频 ID 计算：

   `SHA256("DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-DEV-v001|20260831|" + video_id)`

6. 按 `(完整 SHA‑256 十六进制值, video_id)` 升序排序；

7. 前 `40` 个视频作为新开发集；

8. 其余 `120` 个新候选与旧 `40` 个历史 holdout 合并，形成 `160` 个 Stage‑2 训练视频。

最终数量：

* Stage‑2 train：`160`
* Stage‑2 development：`40`
* official evaluation/test：`0` 个被访问

必须落盘：

* `train_video_ids.txt`
* `development_video_ids.txt`
* `historical_40_excluded_from_dev.txt`
* 一个简短 JSON，记录 seed、生成字符串、数量和来源 annotation。

不要求新的哈希治理框架；普通文本清单和一个 JSON 足够。

### 5.3 使用边界

160-video train 可用于：

* 两臂参数学习；
* 多预算概率校准；
* 实际训练成本估计；
* 动作时长分层阈值；
* 数值与运行检查。

40-video development 仅用于：

* 两臂 terminal EMA 的三档预算评价；
* 预登记的诊断指标；
* 预登记的等成本 whole-video oracle；
* 本轮最终科学门槛。

development 不得用于：

* 超参数、概率、损失权重或训练长度选择；
* checkpoint 选择；
* NMS 或分数阈值选择；
* 修改 K 档位；
* 选择不同的 whole-video 候选定义；
* 决定是否补训某一臂。

### 5.4 证据独立性的诚实边界

该 40-video development 不参与**本轮 Stage‑2 参数更新或本轮规则选择**。

但是，现有 H65 Stage‑1 checkpoint 并不是针对该新划分重新训练和封存的。现有材料不能证明新开发视频从未影响过历史初始化。因此，本划分是：

> 条件于冻结 H65 Stage‑1 初始化的独立 Stage‑2 机制开发集。

它不是完全独立于所有历史训练的外部确认集。

这是在不重训 Scout、不改变另一项科学变量、又不访问 official test 的前提下可接受的最小证据设计。即使实验通过，也只能进入下一轮确认性实验裁决，不能直接形成最终泛化或显著性主张。

若旧 40-video manifest 无法精确恢复、200 个训练 ID 数量不符、train/dev 发生交集，Evaluator 必须返回阻断，不得自行换 seed、改变视频数量或重新切分。

---

## 6. `EVALUATION_AND_GATES`

### 6.1 实验准入检查

正式训练前必须通过：

1. 基座精确为 `04c35a3b...`；
2. `dynamic_budget.py` 的三档预算语义未修改；
3. Stage‑1 checkpoint 路径、SHA‑256、epoch 和状态键完全匹配；
4. 新 train/dev ID 无交集；
5. 两臂初始化参数逐张量相同；
6. 两臂 optimizer 参数组、学习率、trainable flags 和 EMA 配置相同；
7. 固定控制的 K384 前向与基座 K384 在相同 checkpoint、相同输入和相同 RNG 下：

   * selected positions bit-exact；
   * detector tensor 输出在现有数值容差内一致；
   * proposal 原始生成顺序一致；
   * actual observation 数一致；
8. `K256/K512` 真正执行变长 VideoMAE，不得在重型骨干前补齐为 K384 或 K512；
9. mixed-budget packet 分组后，输出重新对齐到原 batch/sample 顺序；
10. 候选训练期预期成本满足上文校准合同。

任何一项失败都属于实现或协议问题，不是模型负结果。

### 6.2 三档终态指标

两臂 terminal EMA 都必须分别在 `K256/K384/K512` 上产生密封预测并报告：

* mAP@0.3、0.4、0.5、0.6、0.7；
* Avg-mAP；
* pre-NMS class-agnostic proposal recall：

  * Recall@100、Recall@200；
  * tIoU 0.5 和 0.7；
* 每个 GT 与同类别最高 tIoU pre-NMS proposal 匹配后的：

  * 起点绝对误差；
  * 终点绝对误差；
  * 物理秒误差的均值和中位数；
  * 相对 GT duration 归一化误差；
* 每视频 pre-NMS、post-NMS proposal 数的均值和中位数；
* top‑200 proposal 中的假阳性：

  * 若不存在同类别、tIoU≥0.3 的 GT 匹配则记为假阳性；
  * 分别报告 NMS 前后；
* 短、中、长动作的 Avg-mAP 与 mAP@0.7。

短、中、长阈值只能由 160-video train 的 GT duration 三分位数预先确定，不得用 development duration 调整。

上述指标为解释性诊断，不新增通过门槛。

### 6.3 K384 安全门

在新 40-video development 上，比较多预算模型与同更新数固定控制，二者都以 K384 推理：

* `ΔAvg-mAP >= -0.2` 个百分点；
* `ΔmAP@0.7 >= -0.2` 个百分点。

两项必须同时满足。

该门防止多预算模型通过降低自身 K384 锚点来人为放大后续 oracle 增益。

### 6.4 等成本 whole-video oracle

对两臂分别使用各自密封的 K256/K384/K512 terminal predictions。

候选空间保持旧 whole-video 定义，但使用新的 40-video development：

1. 全 K384 为该模型自己的固定基线；
2. 对每个不同视频的有序对 `(donor,recipient)`：

   * donor 的全部窗口请求 K256；
   * recipient 的全部窗口请求 K512；
   * 其余视频全部请求 K384；
3. donor 和 recipient 都必须至少有一个窗口的实际成本不同于 K384；
4. 总实际 observation 成本不得超过全 K384；
5. 最多生成 `40×39=1560` 个有序状态；
6. 候选清单只依赖 video ID、sample ID 和 actual observation 成本，必须在读取标签和指标前生成；
7. 两臂使用完全相同的候选清单；
8. proposal 合并必须保持 producer 原始生成顺序。

`33e4ed...` 的必要价值仅在于已经证明额外重排 proposal 会破坏 Soft‑NMS 的确定性重放，因此新 Evaluator 必须保留原顺序。

每个模型的最优候选先最大化：

`min(ΔAvg-mAP − 0.8, ΔmAP@0.7 − 1.0)`

再选择实际成本更低者，最后按 donor、recipient ID 字典序。

### 6.5 Oracle 继续门

多预算模型相对其自身全 K384 基线必须同时满足：

* `ΔAvg-mAP >= +0.8` 个百分点；
* `ΔmAP@0.7 >= +1.0` 个百分点；
* actual observation cost `<=` 其全 K384 成本。

同时必须通过 K384 安全门。

固定控制也运行相同 oracle，但只作为因果对照，不改变预登记阈值。

结果解释冻结为：

1. **多预算通过、固定控制不通过**

   * 支持“训练时多预算暴露建立了新的跨预算联合 headroom”；
   * 返回 Pro 决定是否进入可预测 budget controller 与确认性实验；
   * 不自动训练 controller，不访问 official test。

2. **多预算与固定控制都通过**

   * 说明新 development 本身存在 whole-video headroom；
   * 不能把通过归因于多预算训练；
   * 结果为因果未决，返回 Pro，不自动继续。

3. **多预算未通过**

   * 无论固定控制结果如何，当前多预算检测器适应假说未达到预登记目标；
   * 停止当前 H65 嵌套 `K256/K384/K512` detector-adaptation 路线；
   * 不补训练、不换 seed、不改概率、不降低门槛。

4. **K384 安全门未通过**

   * 即使 oracle 通过，也判定本路线失败；
   * 不接受以损害标准预算性能换取 oracle 增益。

5. **Avg-mAP 达门而 mAP@0.7 未达门**

   * 当前路线仍停止；
   * 只允许下一轮由 Pro 重新判断是否值得单独检验 K、相邻物理间隔或局部采样密度条件；
   * 本轮不得预埋这些输入。

### 6.6 统计与论文证据边界

本轮使用一个 seed `3407` 和一个条件于历史 Stage‑1 初始化的训练侧 development。

因此：

* 不作总体显著性主张；
* 不把 oracle 最优候选称为可部署策略；
* 不把通过结果称为 official validation/test 结果；
* 不补事后 bootstrap 来放大结论；
* 不声称优于 dense AdaTAD；
* 不声称已经获得端到端性能—成本优势。

通过结果最多是强机制开发证据；失败结果可以在冻结范围内停止本三档适应路线，但不能外推否定所有动态计算或预算条件模型。

---

## 7. `CODEX_TASK_ORDER`

### 7.1 Builder

建立唯一分支：

`feature/duca-multi-budget-detector-adaptation-v1-20260831`

基于：

`04c35a3b76897e6c1569eeede41ed3aecaf7f854`

#### Builder 允许完成的最小实现

1. 新增两份 matched Stage‑2 配置：

   * fixed K384 control；
   * calibrated K256/K384/K512 adaptation。
2. 新增一个确定性 train/dev ID 生成入口；
3. 新增训练侧实际 observation 概率校准；
4. 移植并接入：

   * 真实变长 VideoMAE packet 执行；
   * packet 输出回填；
   * actual observation 计数；
   * K384 parity；
   * whole-video terminal evaluator；
   * producer 原始 proposal 顺序保持；
5. 新增一个最小 Slurm launcher；
6. 新增聚焦测试。

#### Builder 禁止修改

* H65 Scout 结构或输入；
* H65 priority、selector、位置嵌套关系；
* `dynamic_budget.py` 的三档语义；
* detector head；
* 分类或回归损失；
* 物理时间逆映射；
* Soft‑NMS；
* annotation、类别映射或评价器；
* official test 入口；
* 任何旧 Marginal、cap-release、96-state 或 whole-video 产物；
* 新 controller、部署优化或额外模型组件。

#### Builder 必须证明

* split 数量与无交集；
* 旧 40 个视频不进入新 development；
* 两臂初始 state_dict 一致；
* optimizer/trainable/EMA 配置一致；
* K384 bit-exact parity；
* 三档实际 observation 单调；
* 短窗口正确折叠；
* mixed packet 对齐；
* proposal 顺序不被排序；
* 6,000 次成功更新是唯一停止时钟；
* development 标签在 terminal 密封前不可访问。

#### Builder 失败返回

遇到以下任一情况立即返回客观阻断，不自行设计替代协议：

* Stage‑1 checkpoint 或 SHA 不匹配；
* 旧 40-video ID 清单无法恢复；
* 规范训练视频不是 200 个；
* K384 无法复现；
* 变长执行必须修改 selector、loss、NMS 或 evaluator；
* 成本校准无法满足合同；
* K512 只能通过候选臂专属 batch size 才能运行。

Builder 不提交正式训练。

### 7.2 独立 Critic

Critic 在独立上下文中只读审查 Builder 的唯一精确提交。

审查范围仅限：

* 科学变量是否仍只有训练预算分布；
* 位置集合是否仍嵌套；
* Scout、selector、detector、loss、NMS、评价器有无隐性变化；
* train/dev 泄漏；
* checkpoint 与状态重置；
* 两臂 trainable 参数、优化器和更新数公平性；
* variable-length forward 是否真实；
* K384 parity；
* packet 和 proposal 顺序；
* 训练成本口径；
* development 是否可能在 terminal 前参与选择。

Critic 输出只能是：

* `PASS`；或
* 一次有界 blocker 清单。

不得因代码风格、日志美化、通用完备性、缺少工作流框架或假设性边角条件制造修复循环。角色边界应继续服从论文优先、最短科学闭环原则。 

### 7.3 独立 Evaluator

Critic `PASS` 后，Evaluator 依次执行：

1. 真实数据 PRE_RUN：

   * checkpoint 身份；
   * split；
   * K384 parity；
   * 每档至少两个真实 batch；
   * 梯度、AMP、EMA、actual cost；
   * 不计算 development mAP。
2. 两臂正式 6,000-update 训练：

   * 相同 seed；
   * 相同硬件类别；
   * 相同样本与增强流；
   * 独立输出根；
   * 不访问 official test。
3. 两臂 terminal `state_dict_ema` 的六次推理：

   * control × K256/K384/K512；
   * adaptation × K256/K384/K512。
4. 密封三档预测后才开放 development 标签；
5. 运行预登记诊断、K384 安全门和两臂 whole-video oracle；
6. 生成一次终态结果，不根据结果回改训练或规则。

确定性的启动器、路径、节点或存储故障只允许一次不改变模型、数据、split、门槛和结果选择的最小恢复。它们必须与科学结果分开报告。

---

## 8. `NEXT_RETURN`

Codex 下一次必须返回以下证据，缺一项不得要求新的科学路线裁决。

### 8.1 实现与身份

* branch；
* exact commit；
* parent commit；
* clean-tree 状态；
* 修改文件清单；
* 从 `33e4ed...` 移植的具体符号；
* `dynamic_budget.py` 未修改证明；
* 两份最终配置全文或稳定链接。

### 8.2 数据划分

* 规范 200-video ID 清单；
* 旧 40-video 清单；
* 新 40-video development 清单；
* 新 160-video train 清单；
* seed 和完整 SHA‑256 生成字符串；
* train/dev 交集为空的测试结果。

### 8.3 起点与训练合同

* Stage‑1 checkpoint 实际路径；
* SHA‑256；
* epoch；
* state key；
* 两臂初始化逐张量一致性结果；
* optimizer 参数组；
* trainable 参数名集合；
* scheduler；
* AMP、EMA、gradient clipping；
* 预算概率 `p256/p384/p512`；
* `μ256/μ384/μ512`；
* 预期与实际训练 observation 成本；
* 6,000 次成功更新审计。

### 8.4 Builder 与 Critic 证据

* focused test 结果；
* K384 parity 数值；
* 三档真实 batch 的 selected count 与 packet shape；
* Critic 对 exact commit 的完整终态结论；
* 所有未关闭 blocker。

### 8.5 正式作业与终态模型

* PRE_RUN Job ID、状态和输出根；
* 两臂训练 Job ID、状态和输出根；
* 基础设施恢复记录；
* update‑6,000 checkpoint 路径与 SHA‑256；
* `state_dict_ema` 存在性；
* 两臂实际成功更新数；
* 两臂最终 observation 总量。

### 8.6 结果

对两臂、三档预算逐项返回：

* 五个 tIoU mAP；
* Avg-mAP；
* proposal recall；
* 起终点误差；
* proposal 数；
* NMS 前后假阳性；
* 短、中、长动作结果；
* 原始预测和结构化指标路径。

### 8.7 Oracle 与门槛

* 无标签候选 manifest；
* 合法候选数；
* 固定 K384 成本；
* 每臂最优 Avg-mAP 候选；
* 每臂最优 mAP@0.7 候选；
* 每臂最优联合门余量候选；
* K384 安全门两项差值；
* 多预算 oracle 两项差值和成本；
* fixed-control oracle 结果；
* 根据本裁决规则得到的唯一分类：

  * 支持适应；
  * 因果未决；
  * 停止当前路线；
  * 证据无效。

在这些实现与结果证据返回前，不得训练 controller，不得访问 official test，不得另行提出预算条件嵌入或新 selector，也不得把代码完成、测试通过或训练作业成功当作模型有效性证据。

以上即为本轮冻结版科学裁决，可直接作为 Builder、Critic 与 Evaluator 的唯一执行依据。

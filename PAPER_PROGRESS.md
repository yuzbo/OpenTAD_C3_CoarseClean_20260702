# DUCA 论文缩略报告

- 更新日期：2026-08-31
- 名称说明：DUCA 是本项目沿用的方法路线名称。
- 当前结论：固定 `K=384` 的任务状态时序 coreset 低于匹配均匀选择，`DUCA-Coverage-v1` 也未通过预注册中间机制门。冻结 H65 的三档边际预算诊断、96-state 联合邻域和最终 704-state 整视频枚举均未达到 `+0.8/+1.0` 联合门；Pro 对这一冻结检测器的旧动作空间作出的 `STOP` 继续有效。新的 Pro 数据准入裁决在该边界之外选择 `CONTINUE`：完整 200-video `training` 与完整 211-video OpenTAD `validation` 已正式准入，ActionFormer 212 只保留为来源差异。唯一模型任务是比较固定 K384 暴露与嵌套 K256/K384/K512 暴露下的完整 H65 Stage-2 系统；三种子按 `3407 → 3408 → 3409` 全部盲执行，所有训练和 prediction 封存后才一次性读取 held-out 指标。Builder 的初始实现 `0d67d49c...` 和真时间元数据修复 `409f370a...` 均通过聚焦测试与独立审查。首个完整训练链的 seed-3407 Jobs `1262696/1262697` 在任何训练前因不适用的 legacy P0 启动绑定收到空 formal variant 而退出，依赖 Jobs `1262698`–`1262701` 随后取消且从未启动。最小恢复提交 `2b3b3243...` 只停用该 legacy binder，同时保留 6,000 successful updates、update audit、terminal EMA 和 held-out sealing；N16R4 同款测试为 `26 passed`，新的独立 Critic 返回 `PASS`。PRE_RUN `1262715` 已以 4/4 次有限损失、有限梯度和成功 optimizer/scheduler/EMA/DUCA 更新通过。随后 Jobs `1262719/1262720` 同时在节点 `g0030` 的 batch 脚本启动前被信号 53 取消，依赖 Jobs `1262721`–`1262724` 亦已取消；没有模型进程、更新或结果。当前权威完整训练链排除该节点并显式绑定提交目录：seed 3407 Jobs `1262743/1262744`、seed 3408 Jobs `1262745/1262746`、seed 3409 Jobs `1262747/1262748`，后两组保持严格 `afterok` 盲序列。上述启动故障均不是模型或性能结果；当前没有 prediction、mAP、区间或成本结果。

## 1. 论文问题与应用价值

离线时序动作检测（Temporal Action Detection, TAD）通常对长视频进行密集的重型视频编码。DUCA 研究一个更节省计算的问题：能否先用低成本模型预测逐时刻动作性和边界重要性，再由确定性规则选择少量高分辨率帧，并进一步为不同视频或窗口分配不同预算，在真实端到端计算下降时保持高时间交并比（temporal Intersection over Union, tIoU）下的边界定位性能。

这个问题的应用价值在于：如果低成本语义证据能够可靠地替代大量冗余帧，就可以减少 VideoMAE 等视频骨干网络的实际输入，而不必修改下游检测头或官方评估器。

## 2. 当前科学路线

长期路线包含两个层次，二者不能混写：

1. **语义间接选帧。** 低成本侦察模型学习二元动作性与动作起止边界；确定性采集规则根据这些预测产生有序、非均匀的原始帧位置。侦察模型不直接把帧索引当作主要学习目标。
2. **动态预算。** 长期论文主张要求根据逐视频或逐窗口的语义证据决定保留帧数，使重型 VideoMAE 路径真实执行不同工作量。固定 `K=384` 仅用于机制归因、公平对照和回退。

已完成的归因实验把 768 帧输入组成 384 个 VideoMAE 原生两帧 tubelet，并固定选择其中 192 个。对照臂在原生 tubelet 网格上均匀选择；候选臂使用冻结侦察器的动作性、边界强度和时序新颖性进行确定性选择。该候选在所有报告阈值上均低于均匀选择，现已终止。

固定 K 内的 coverage 和其后的三档预算转移均已结束。后者冻结 H65 Scout 与 detector，并从同一非连续 H65 priority sequence 构造 K256/K384/K512 的嵌套集合，在逐窗口、差分联合邻域和整视频跨视频三个层次搜索不增加真实 observation 成本的重分配。完整开发集 oracle 枚举没有产生预登记的 Avg-mAP 与高 tIoU 联合 headroom。Pro 因而关闭这一动作空间；它不是完整训练后的论文主模型，也不授权 Codex 自动恢复历史路线或自行选择新机制。

最新 `REVISE` 将下一项机制检验限定为 H65 系统的多预算暴露适应：第一轮继续使用上述嵌套位置构造，只改变 Stage-2 训练是否同时见到 K256/K384/K512。由于 Stage-2 还继续适配 Scout/selector 相关路径和 detector feedback，该结果不能称为纯检测器适应。它不把输入分布不匹配当作已证实根因，也不同时加入渐进解冻、STE 温度退火、五档预算曲线、预算条件嵌入、蒸馏、Gumbel-Softmax、新 Scout、Mamba、Block Drop、部署优化或跨数据集扩展。较早附件提出的“预算原生选点 + 多预算训练”和同轮未落盘 `research_project_analysis.md` 摘要中的新增路线均不进入第一轮。

## 3. 与官方基线的真实差异

共享的未修改 AdaTAD 基线使用官方代码 revision `01c58b9f2370e914150cf94d392208a4e211c053`、seed 42、60 个训练轮次和官方评估器，平均检测精度（Avg-mAP）为 `68.73`；论文公开锚点为 `69.03`。DUCA 只读引用这一共享复现，不重复训练官方 dense 模型。

DUCA 与官方 dense 模型的目标差异只应来自输入采样与预算：前者先以低成本语义模型选帧，再让 VideoMAE 只处理被选中的高分辨率帧。下游 ActionFormer 检测头、损失、NMS 和官方评估器原则上不变。历史 65.xx 或 66.xx 结果不属于官方 dense 复现，不能代替 `68.73`，也不能在协议不匹配时直接计算方法增益。

## 4. 已完成实现与代码身份

- H65 干净复现使用历史 ASFormer 语义预测、确定性非均匀逐帧选择和固定 `K=384`。当前可审计的 30+60 训练参考冻结于源码 revision `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。
- PJST-D1 是当前物理时间表示候选的代码名称。它不改变选择器，只在 VideoMAE 首次二帧 tubelet 混合前校正导数分量所使用的时间间隔。匹配训练冻结于 clean revision `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`；只读重推理与统计入口冻结于 `7bd120f0d342bf175c97c365fba7cbd359df055e`。
- UVT 诊断分支为 `codex/duca-uvt-utility-value-20260819@df544c78ce515d925dc7019f106fce09a53c09f8`；Fovea/Query-Bridge 诊断分支为 `codex/duca-fovea-query-bridge-20260819@4ae5067100c4490c7110c00a1ad406230ba603cd`。它们与 H65/PJST 不是同提交匹配比较，不能用跨版本差值归因。
- 物理连续片段路线已完成真实训练，但性能明显下降，现作为负结果保存，不再是当前论文主线。
- 为支持下一次科学路线与代码主线裁决，当前模型、配置、启动器、测试和研究记忆已同步到 `codex/duca-research-sync-20260828`；代码库存固定提交为 `5136011ed57df8a639427a633a488a592ba95924`。该快照用于检查实现重叠与历史分支，不是经过运行前检查的实验候选，也不能替代各实验的 clean revision。
- 先前的动态窗口预算候选冻结于 `codex/duca-semantic-budget-matched-20260828@36d75c146492a38eb8966c66ff6b2881938cf3c6`。它在取得效能结果前被后续科学裁决暂缓，保留为下一阶段的实现参考，不能作为动态预算证据。
- 当前原生 tubelet 时序 coreset 候选位于独立干净分支 `codex/duca-native-tubelet-coreset-20260828`，冻结提交为 `b33391126eac05e3353d322b973dda91741f0732`。实现包含固定 192 个原生 tubelet、任务状态驱动选择、端点/空洞覆盖、低分辨率上下文回收、物理时间重建、严格匹配的均匀对照、配置、聚焦测试与 N16R4 启动器。N16R4 环境的 20 项相关测试通过，独立只读审查未发现阻断性缺陷；两臂均完成 60 轮训练和官方验证，但结构化证据保存失败。
- 窗口级动态预算候选位于独立干净分支 `codex/duca-dynamic-native-tubelet-budget-20260829`，冻结提交为 `d127c2b2ceea7ff8a6932aa4a1925e1ff86cf610`。实现按视频生成需求排序和 16/20/24-clip 预算，在各预算内执行确定性均匀 tubelet 选择，并按真实 clip 数分组调用 VideoMAE；较短预算不会在重型骨干前补齐到 24。短窗口若不能容纳分配预算会明确退出。Python 编译、启动器语法、纯启动器测试和独立静态审查已通过；尚无运行前检查或正式实验结果。
- 当前 `DUCA-Coverage-v1` 候选位于独立干净分支 `feature/duca-coverage-only-v1-20260829`，当前提交为 `048143124e2a36a76575200ae17d6f42ec79ea3a`，基于 H65 正式提交 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。实现新增固定预算设施位置选择器，并提供 matched H65 对照、真实训练样本无标签重放门、60 轮配置、恢复合同和 N16R4 启动器。修正后的 PRE_RUN Job `1261679` 已执行 27 项测试和 200 个真实 training 样本重放，但因预注册覆盖/空洞干预条件未满足而在 smoke 前停止。代码已同步 GitHub并部署到 N16R4 干净目录 `/data/run01/sczc063/yuzibo/duca_coverage_v1_04814312_20260830`。
- 最新 Pro 裁决指定从 H65 clean revision `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 建立 `feature/duca-marginal-budget-v1-20260830`。吸收短窗口修订后的实现为 `be5bb8033c0b11c628394d268c1923ab398c04ed`；测试合同修复为 `f87555f7da362fe1a20d4ca08f7a68c975ed8280`。该提交完成 K384、K256 和 K512 冻结反事实产物后，汇总器因把换行文本 block-list 当作 JSON 读取而退出。最新提交 [`f67d96fdf68a295eaa7f678f3dfc125530828889`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f67d96fdf68a295eaa7f678f3dfc125530828889) 只为官方评估器生成确定性的 JSON block-list 适配文件并加入回归测试；没有修改模型、分配器、预测、损失、数据、NMS、指标或科学门槛。恢复作业没有重跑三个冻结 producer 阶段，只在当前 clean commit 上重做运行前身份核验并汇总既有产物。producer 产物保留其 `f87555f7...` 来源，汇总与最终结果绑定 `f67d96fd...`；两者的配置、checkpoint、annotation、类别映射、预训练权重及其哈希一致，这一跨提交来源会在后续 Pro 材料中明确披露。
- Pro 对灰区的后继实现位于 [`feature/duca-marginal-cap-release-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-falsifier-v1-20260831)，精确提交为 [`d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f)。它只增加独立的 `max_changed_fraction=1.0` 只读汇总入口和聚焦测试，默认 `0.5` 汇总路径、三档 producer、模型、数据、NMS 与门槛均不变。N16R4 的 14 项聚焦测试和独立 Critic 已通过。
- 最新的联合邻域诊断实现位于 [`feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831)，精确提交为 [`46812facc8773d9b4a9c21833cbe397c8aaa5a2d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d)。它只修改 probe runner 与聚焦测试；`dynamic_budget.py` 相对父提交逐字不变。实现从密封分配和真实 observation 成本自动导出 8 个最小合法转移、6 个净转移组及 96 个唯一联合状态，没有为多解视频硬编码配对。N16R4 上 16 项聚焦测试、23 项既有回归测试和独立 Critic 均通过。
- 整视频最终 falsifier 位于 [`feature/duca-whole-video-consistent-budget-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831)，权威公开提交为 [`33e4ed137c33eef07f0452b44506a6993bdf7535`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535)。父提交 `c27d77...` 的首次 PRE_RUN `1262147` 暴露密封 proposal 被额外排序、从而改变 Soft-NMS 并破坏锚点复现的确定性证据错误；`33e4ed...` 只恢复密封 producer 原始顺序并增加回归测试，不改变候选、成本、预测、NMS、评估器或三档预算。28 项聚焦测试与独立 Critic 通过。修正后 PRE_RUN `1262161` 复现全部锚点；唯一恢复 Job `1262190` 完成 704 个候选。终态 JSON SHA-256 为 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`，通过候选数为零。Pro 随后使用该最新 GitHub 身份作出项目级 `STOP`；该分支与 Marginal-v1、cap-release、96-state 分支均只作为负证据读取。
- 新设计的 H65 系统多预算暴露实验以 H65 干净提交 `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 为模型基座，并且只允许从 `33e4ed...` 移植真实变长执行、packet 对齐、实际 observation 计数、K384 parity、whole-video 评价和原始生成顺序保持。`33e4ed...` 不是新模型的科学基座。Pro 授权的 `feature/duca-h65-system-multibudget-exposure-v1-20260831` 已从 `04c35a3b...` 完成最小实现，公开精确提交为 [`0d67d49c2fc4a5f50aa784f7809c0dd936492109`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/0d67d49c2fc4a5f50aa784f7809c0dd936492109)。它实现嵌套三档真实变长 VideoMAE 执行、K384 原路径一致性、完整成本分解、九份 prediction 封存和一次性官方评估。初始 PRE_RUN `1262690` 在任何成功更新前发现短窗口折叠预算把 padded `-1` 写入真时间映射；修复提交 [`409f370a7ed14e7077bc87138196ab6abe459f99`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/409f370a7ed14e7077bc87138196ab6abe459f99) 只按既有 detector mask 写出活动映射，不改执行张量或科学合同。首个完整训练链随后在训练前触发 legacy P0 binder 错配。当前权威恢复提交为 [`2b3b3243066a89e5a4be5acdb178c318fbeceac0`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/2b3b3243066a89e5a4be5acdb178c318fbeceac0)：它只让该实验不再进入不适用的 legacy binder，同时保留成功更新审计、6,000-update 预算、terminal EMA 与 held-out sealing。N16R4 `26 passed`、独立 Critic `PASS`，PRE_RUN `1262715` 完成 4/4 次成功更新并通过 checkpoint/probe validator。第一次从该提交建立的训练根 Jobs `1262719/1262720` 在 `g0030` 的 batch 脚本启动前被信号 53 取消，没有模型进程或产物；依赖 Jobs `1262721`–`1262724` 已取消。当前六个完整训练单元为 `1262743`–`1262748`，从同一提交与同一 calibration 排除 `g0030` 后按 seed 3407→3408→3409 的严格依赖链执行。这仍只是实现、运行准入与训练部署证据；正式结果尚未产生。
- 2026-08-31 的 GitHub 全历史 Pro 复核已逐项读取公开 Wiki、Gemini 预审和关键精确提交，并再次裁决 `REVISE`。它没有授权模型实现：当前仍只执行完整数据身份审计。只有数据身份返回 Pro 并获准后，才解锁固定 K384 与 K256/K384/K512 多预算暴露两个训练臂；两臂使用完整 200-video training、种子 `3407/3408/3409`、每种子 6,000 次成功更新、一次性完整 held-out 评估和整视频配对 bootstrap。完整原文见 `research-wiki/sources/2026-08-31-pro-github-wiki-comprehensive-review-v002.md`。
- 完整数据身份审计实现位于 [`feature/duca-full-data-identity-audit-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-full-data-identity-audit-v1-20260831)，精确提交为 [`fdd2bcdddf3f23f3546244adf90c4427ed022837`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdd2bcdddf3f23f3546244adf90c4427ed022837)，父提交为 H65 `04c35a3b...`。独立 Critic `PASS`，N16R4 CPU Evaluator 得到 `DATA_IDENTITY_PASS_211`；完整报告 SHA-256 为 `d7251c11935644cf8661e6bfdcfb857e29d2357cb894b7de9d8b2bd7eaf6f1ab`。这只解决数据事实，须返回 Pro 后才可能解除模型实现阻断。
- 精确 DUCA Project Pro 对话 `6a956592-da38-83e9-b50c-fd3906c0ec41` 已正式准入上述数据边界并解锁唯一模型 Builder。完整裁决见 `research-wiki/sources/2026-08-31-pro-duca-full-data-identity-admission-v001.md`；该授权不构成模型有效性证据。

项目根目录承担多路线协调，工作区可能包含尚未归档的修改。论文实验身份必须引用上述独立 clean revision，而不能用项目根当前 `HEAD` 代替。

## 5. 正式实验协议

正式比较要求两臂使用完整 THUMOS14 训练集，并在设计与全部选择规则冻结后，于完整官方 held-out evaluation
split 上使用同一检测器、损失、Soft-NMS、评估器和预登记模型选择规则作最终比较。训练侧子集只可用于前置诊断，
不能替代正式证据；官方留出评估不得参与调参、checkpoint/阈值/规则选择或路线迭代。当前项目同时存在 OpenTAD
`training/validation` 与 ActionFormer `Validation/Test` 两种命名。字面审计证明 OpenTAD 的 annotation、loader、
physical、evaluator 与历史 prediction 均使用相同 211 个 held-out 视频；ActionFormer `Test` 为 212，唯一额外 ID
是 `video_test_0000270`，OpenTAD 源码说明其因错误标注被删除。Pro 已正式准入这一 200/211 边界。当前已有实验主要使用
seed `3407`；单种子结果不能支持稳定性或显著性结论。

当前与后续实验矩阵按以下顺序组织：

1. 只读引用共享的官方 dense AdaTAD 结果；
2. H65 固定 `K=384` 的 30+60 参考训练；
3. PJST-D1 固定/重放选择结果的物理时间表示 OFF/ON 配对已完成点估计，但平均点差为负且配对区间缺失；它不再是当前优先开发项；
4. 固定 `K=384` 的原生 tubelet 均匀选择与任务状态驱动时序 coreset 已完成；细粒度 coreset 候选因一致的负向点估计终止；
5. 同一 H65 代码基座上的 matched allocation control 与 `DUCA-Coverage-v1` 已完成真实训练样本无标签重放门；该门未通过，因此两个 60 轮完整训练臂没有启动。当前对照实现是预算校准系统采样，而冻结设计曾概括为 Top-K；在 Pro 裁决该基线身份和失败机制前不得重提正式训练。
6. 冻结 H65 的 K256/K384/K512 反事实边际预算实验、cap-release 诊断和 capped→released 差分邻域的 96 个联合状态枚举均已完成。50% 上限 oracle 的 Avg-mAP/mAP@0.7 增益为 `+0.726/+0.729` 个百分点；解除上限后反而降为 `+0.427/+0.450`。96-state 中没有状态同时达到 `+0.8/+1.0`；当前加性 Marginal-v1 及其本次视频级联合效用修复均按冻结规则停止。
7. 整视频一致预算的跨视频单次转移 oracle 已完成：donor 视频所有窗口请求 K256，recipient 视频所有窗口请求 K512，其余视频保持 K384；候选总实际 observation 成本不超过固定 K384 的 `47110`。候选集合在读取标签或指标前完整生成，随后仅复用密封预测和相同评估器。1560 个有序对中 704 个合法候选完成评估，没有候选通过联合门；没有执行模型前向、训练、bootstrap 或 official test。
8. 新设计的单变量实验比较固定 K384 暴露与 K256/K384/K512 多预算暴露下的完整 H65 Stage-2 系统。Pro 已冻结两臂从同一 H65 Stage-1 `epoch_29/state_dict_ema` 开始、在完整 200-video `training` 集合上各完成 6,000 次成功更新，并让候选概率按冻结 occurrence 计划的实际 observation 成本匹配 K384。旧 160/40 划分、有标签训练侧 mAP 门和 oracle 已撤销。三种子全部训练并正式预测封存后，两个模型在同一个无标签 fixed mixed-budget manifest 上直接比较，并在完整 211-video `validation` held-out 集合上一次性执行统一评测和 10,000 次整视频配对 bootstrap。Builder 初始实现 `0d67d49c...` 的 PRE_RUN `1262690` 暴露短窗口真时间元数据缺陷；`409f370a...` 修复该元数据后，旧训练链又在训练前暴露不适用的 legacy P0 binder。当前最小恢复 `2b3b3243...` 已通过 N16R4 `26 passed`、独立 Critic 和 PRE_RUN `1262715`。第一次恢复训练提交在节点 `g0030` 的脚本启动前终止，未产生训练证据；当前完整训练单元为 `1262743`–`1262748`，排除该节点并按三种子盲顺序执行。仍没有 prediction、mAP、区间或成本结果。

主要指标为 tIoU 0.3、0.4、0.5、0.6、0.7 下的 mAP，以及五个阈值的平均值；同时报告短动作、边界定位和完整端到端计算成本。任何计算节省主张都必须来自实际执行的 VideoMAE 工作量和相同硬件条件下的测量，不能由 padding 后的名义帧数推断。

完整训练至少每 5 个训练轮次保存一次可恢复的 PyTorch `.pth` 检查点；如果未修改的官方配置保存更频繁，则保留官方间隔。恢复包应包含模型、指数移动平均模型、优化器、学习率调度器、混合精度缩放器、训练轮次或更新计数及随机状态，并至少保留最近三个有效恢复点、预定义里程碑和最终检查点。最终结果使用预先登记的最终模型或最终指数移动平均模型，不能根据中间验证结果事后挑选。

## 6. 已取得的性能结果

除特别说明外，Avg-mAP 是 tIoU 0.3 至 0.7 五个阈值下 mAP 的平均值。

| 实验 | Avg-mAP | mAP@0.7 | 证据边界 |
|---|---:|---:|---|
| 共享官方 dense AdaTAD | 68.73 | 未在本报告重复摘录 | 一次共享复现；论文公开 Avg-mAP 为 69.03 |
| H65 30+60 | 65.13 | 43.31 | 单种子、完整训练；当前 H65 参考 |
| H65 20+40 | 62.46 | 39.94 | 同 seed 的训练压缩诊断；比 30+60 低 2.66/3.37 点 |
| H65 30+30，AM-RPCH25 | 63.22 | 41.25 | 只改变第二阶段学习率日程 |
| H65 30+30，LongCosine-H6000 | 63.56 | 41.01 | 更慢衰减仍未恢复 30+60 |
| RankPack K384 | 61.57 | 37.10 | 单种子物理时间表示对照 |
| TrueTime K384 | 62.19 | 37.89 | 相对 RankPack 为 +0.62/+0.79 点；尚无配对区间 |
| PJST-D1 OFF | 65.063 | 43.646 | 211/211 视频，冻结 H65 选择结果 |
| PJST-D1 ON | 64.591 | 43.769 | 211/211 视频；相对 OFF 为 -0.472/+0.123 点 |
| 原生 tubelet uniform K384 | 64.13 | 42.45 | 单 seed；60 轮训练和官方验证完成；结构化指标文件未封存 |
| 原生 tubelet coreset K384 | 62.81 | 40.56 | 单 seed；相对匹配 uniform 为 -1.32/-1.89 点；结构化指标文件未封存 |
| 连续片段 FZ | 49.89 | 29.68 | 完整单种子训练；明显负结果 |
| 连续片段 JT | 47.24 | 26.52 | 完整单种子训练；联合训练未恢复性能 |

PJST-D1 两臂各产生 422,000 条预测，视频集合一致，重推理逐项复现了原始点估计。预先登记的 10,000 次整视频配对自助法在任何抽样前退出，因为统计程序指向了错误的预测文件目录。因此当前只有点估计，没有置信区间；不能据此宣布总体效应显著为负，也不能把 mAP@0.7 的 `+0.123` 点解释为真实收益。

UVT 的 legacy/geometry/geometry+EMA 三臂 Avg-mAP 为 `57.35/55.93/55.92`。Fovea/Query-Bridge 第一波中最佳 `query_cycle` 为 `54.67`。这些结果来自不同代码版本与实验合同，只能说明相应首版设计未恢复 H65 性能，不能用来量化某一个组件的因果影响。

历史 `65.3857` 是 H65 语义间接非均匀逐帧选择的 30+60 诊断结果；`65.696` 来自改变物理检测网格的探索实现。后者同时改变了检测器时间几何，因此不是与官方原生检测器严格匹配的输入采样对照。当前干净 H65 复现 `65.13` 是更适合作为后续同代码、同训练协议比较的参考。

当前还没有与主路线匹配、可用于论文的完整端到端成本结果，也没有动态预算保持性能或提高效率的实验证据。`DUCA-Coverage-v1` 已完成代码实现、独立审查和真实 training 数据运行前门；该门给出不满足预注册干预条件的诊断结果，但没有进入训练，因此不是 mAP 或成本结果。

`DUCA-Marginal-v1` 的冻结 detector 诊断在训练侧 40 个 utility holdout 视频、124 个窗口上比较了固定 K384 与使用真实反事实效用的等预算 oracle。固定臂 Avg-mAP/mAP@0.7 为 `88.131/76.271`，50% 上限 oracle 为 `88.857/77.000`。解除上限后，分配从 K256/K384/K512 的 `11/102/11` 变为 `17/90/17`，但结果降为 `88.559/76.721`；相对固定臂只有 `+0.427/+0.450` 个百分点。总 observation 预算仍精确为 `47110`。这里的百分数只描述训练侧 controller holdout，不可与 official validation/test 表直接比较。强 headroom 门未通过，因此 utility predictor、正式测试、配对区间和端到端成本均未运行。

随后只在 capped 与 released 分配不同的 12 个窗口上枚举了全部 96 个逐视频等成本联合状态。最佳 Avg-mAP 状态相对固定 K384 为 `+0.733` 个百分点，但 mAP@0.7 仅 `+0.479`；最佳 mAP@0.7 状态为 `+0.549/+0.934`；按两项联合门最优的状态为 `+0.554/+0.933`。没有状态同时满足 `+0.8/+1.0`，也没有单个最小合法转移同时改善 Avg-mAP 与 mAP@0.7。该诊断没有执行模型前向、训练、official test 或 bootstrap；从 96 个开发集状态中事后选出的最优状态不能当作可部署策略或论文主结果。

整视频一致预算的最终开发集 falsifier 在相同固定 K384 锚点 `88.1312%/76.2706%`（Avg-mAP/mAP@0.7）和真实成本 `47110` 下完成 704 个合法状态。Avg-mAP 最优状态的变化为 `+0.6942/-0.0436` 个百分点，mAP@0.7 最优状态为 `-0.2359/+0.4970`，联合门余量最优状态为 `+0.1474/+0.4898`；通过候选数为零。该结果只属于训练侧 controller holdout 的事后 privileged oracle 证伪，没有模型前向、训练、官方验证/测试、配对区间或可部署策略含义。

## 7. 结果解释与已停止的方向

- 20+40 训练压缩、AM-RPCH25 和 LongCosine-H6000 均未恢复 H65 30+60 的性能。现有证据说明简单压缩预热或只修改第二阶段学习率尾部不足以保持性能；它不否定 H65 的语义间接选帧机制。
- 连续 16 帧片段采样在真实训练中造成大幅定位损失，联合训练也未恢复，因此该采样单元不再作为当前主路线。这个结果不否定低成本语义侦察或物理时间一致性的一般问题。
- TrueTime 相对 RankPack 有小幅单种子提升，但证据不足以形成论文主张。
- PJST-D1 的当前点估计没有显示平均性能收益；缺少配对置信区间意味着总体效应仍未完成统计裁决。统计程序的路径错误是证据生成失败，不是模型的科学失败。
- UVT 与 Fovea/Query-Bridge 同时改变了选择分数、预算证据或训练信息流，且缺少与 H65 同提交的严格隔离，因此其性能下降不能归因于单一组件。

## 8. 当前证据缺口与下一动作

THUMOS14 原始视频、注释、类别映射、VideoMAE-S 预训练权重、H65 Stage-1 侦察器检查点和共享官方 AdaTAD 结果均已核验。PJST-D1 的 OFF/ON 完整推理已经结束。

当前固定预算归因已经得到负向点估计：在完全相同的 `K=384` 高分辨率帧预算、训练日程和检测器下，任务状态驱动 coreset 没有优于原生 tubelet 均匀选择，并在 tIoU 0.5 至 0.7 下降 `1.54/2.03/1.89` 个百分点。这个结果优先要求分析端点覆盖、最大空洞、跨 tubelet 打包、低分辨率上下文回收和选择分数是否共同损害重型表示，而不能直接把固定预算重命名为最终方法。

PJST-D1 的配对区间仍属未完成证据，但它不会改变当前路线，因此不作为当前任务补齐。原生 tubelet 两臂已从干净 H65 基座完成实现、N16R4 聚焦测试、独立审查和 60 轮训练。uniform `1260184` 与 coreset `1260185` 都写出 epoch-59 检查点并完成官方 211 视频评估，日志点估计分别为 `64.13%` 和 `62.81%`。两臂随后因同一个证据封存错误退出：配置没有保存预测，结构化指标入口因此拒绝写出 `metrics_epoch59_ema.json`。当前没有配对区间或成本结果；训练成功、日志点估计和结构化证据缺失必须分别陈述。

Pro 冻结的 96-state 联合 mAP 邻域诊断已经在公开提交 `46812fac...` 上完成。原 fixed/capped/released 三结果复现误差为 `0.0` 个百分点，96 次评估保持逐视频预算和全局成本 `47110`；没有状态通过 `+0.8/+1.0` 联合门，也没有单个最小合法转移同时改善两个门指标。Pro 已据此最终裁决 `STOP`：现有加性 Marginal-v1 及本次邻域修复关闭，不再重跑、改门、补 bootstrap、训练 utility head 或访问 official test。当前分支只作为负证据读取；未来若重新研究动态计算，必须由 Pro 以新的机制假设和独立任务启动，不能作为 Marginal-v1 的恢复。

项目级 Pro 对旧三档预算转移动作空间的 `STOP` 继续有效。Pro 已把“K384-only 训练是否缺少跨预算兼容性”冻结成一项新的、边界之外的可证伪实验，并用完整数据协议撤销 160/40 与有标签训练侧 oracle。身份 Builder `fdd2bcdd...`、独立 Critic 和 N16R4 CPU Evaluator 已证明 `training` 的 annotation/loader/物理文件均为同一 200 个视频，`validation` 的 annotation/loader/physical/evaluator/historical prediction 均为同一 211；ActionFormer 212 唯一额外 ID `video_test_0000270` 有 OpenTAD 源码解释。Pro 已签发数据准入和模型 Builder；三种子必须全部盲执行后再统一读取 held-out。K384 安全门为相对同更新数控制的 Avg-mAP 与 mAP@0.7 均不低于 `-0.2` 个百分点；相同 fixed mixed-budget manifest 上的直接继续门为 `+0.8/+1.0` 且实际成本不高于全 K384，并要求两项配对区间下界均大于零。

本轮 Builder 提交 `0d67d49c...` 把这一协议落实为可运行代码。K256/K384/K512 每次成功 update 的冻结次数为
`1454/3000/1546`，实际成本校准概率为 `0.24235161911751213/0.5/0.25764838088248787`。完整 held-out
推理只读取移除动作类别和时间段的 211-video annotation；所有九份预测与成本封存后，唯一 evaluator 才解析完整
held-out annotation 一次。成本口径包含真实进入 VideoMAE 的 observation、packet 执行、逐视频 wall-clock、
Scout/VideoMAE/detector 分项和 GPU 峰值显存。当前这些仍是实现合同；Critic 通过和正式作业终态之前不得写成
性能、效率或可发表性结论。

## 9. 可发表性边界

目前可以写入论文的事实是：H65 的 30+60 训练参考明显优于已测试的 60 轮压缩日程；连续片段采样是明确负结果；PJST-D1 的匹配点估计没有平均收益；原生 tubelet coreset 的单种子点估计比匹配均匀选择低 `1.32` 个百分点；当前 96-anchor Coverage 干预没有达到预注册中间机制目标。DUCA-Marginal 的训练侧 oracle 在 50% 上限下仅有灰区增益，解除上限后增益进一步降至 `+0.427/+0.450` 个百分点；其 96-state 联合差分邻域和最终 704-state 整视频单次转移 falsifier 均没有达到 `+0.8/+1.0` 的预注册门，因此当前冻结动作空间没有足够开发集 headroom。Pro 将其定位为可用于未来论文补充失败分析的内部终态证据，而不是独立 CVPR 主结果。它没有证明 Scout 能预测预算效用，也没有 official test、统计区间、可部署 controller 或端到端成本结果。不能把该停止扩展为所有 coverage、动态预算、budget-conditioned training、内部 token/层级条件计算或物理时间方法无效，也不能声称 DUCA 有效、优于 dense，或已经获得性能—成本联合优势。

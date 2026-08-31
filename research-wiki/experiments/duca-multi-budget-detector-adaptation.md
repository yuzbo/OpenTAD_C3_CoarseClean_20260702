---
type: experiment
status: tested
updated: 2026-08-31
project: DUCA
---

# DUCA H65 系统多预算暴露适应

## 科学问题

在保持 H65 的当前嵌套 K256/K384/K512 位置构造、模型结构和评价协议不变时，只改变 Stage-2 训练所见的预算
分布，能否建立跨预算兼容性，并在相同、无标签、等价预算的 fixed mixed-budget workload 下，相对只见 K384
的匹配控制提高 Avg-mAP 与高时间交并比性能？

这是位于此前项目级 `STOP` 边界之外的新机制检验。旧结论仍成立：冻结 K384 检测器后进行三档预算转移没有
达到预登记联合门。新实验不重跑旧 oracle，也不把跨预算表示不匹配预先写成已证实根因。

## 单一干预

- 固定预算控制只在 K384 上训练。
- 候选在 K256/K384/K512 上共同训练；冻结 `p384=0.50`，其余两档概率根据完整 200-video 训练集上的冻结
  6,000-update sample occurrence 计划和短窗口折叠后的实际 observation 成本唯一校准。
- 两臂使用同一 H65 起点、相同成功更新数、优化器、学习率日程、随机种子、可训练参数集合和最终指数移动平均
  模型选择规则。H65 Stage-2 同时适配 coarse trunk、action head、transition scorer、ASFormer policy path 和
  detector-feedback path，因此该总效应不能称为“纯检测器适应”。

第一轮保留当前嵌套位置构造。不得同时改成预算原生 H65 采样；否则将同时改变选点和训练分布，无法把结果归因
于预算暴露。

## 代码边界

- 模型基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`。
- `33e4ed137c33eef07f0452b44506a6993bdf7535` 仅提供真实变长执行、packet 对齐、实际 observation 计数、K384
  parity、whole-video 评价和原始生成顺序保持等已验证功能。
- 禁止预算条件嵌入、蒸馏、Gumbel-Softmax、新 Scout/head/selector、DFT、Mamba、Block Drop 和 TensorRT。

## 决定性输出

1. K256、K384、K512 各自的 Avg-mAP、mAP@0.3--0.7、proposal recall、起终点误差、proposal 数、NMS 前后
   假阳性和动作长度分层结果。
2. 多预算模型在 K384 上相对同更新数固定控制满足 `ΔAvg-mAP >= -0.2` 和 `ΔmAP@0.7 >= -0.2` 个百分点。
3. 两臂在同一个预注册、无标签、等价预算的 fixed mixed-budget manifest 上直接比较；候选相对控制必须同时满足
   `ΔAvg-mAP >= +0.8`、`ΔmAP@0.7 >= +1.0` 个百分点，且实际 observation 成本不高于全 K384。
4. 两臂都使用完整 200-video `training` 集合完成 6,000 次成功 update，并在设计与预测全部密封后，在完整
   `validation` held-out 集合上执行一次统一评测和 10,000 次整视频配对 bootstrap。训练子集、旧 40-video
   holdout、pilot、smoke 或有标签训练侧 oracle 均不能替代这一证据。

若 K384 安全门、mixed 任一实际效应门或成本条件失败，则停止当前三档适应路线；点估计通过但配对区间包含零时，
当前实验同样终结且不前进。任何 controller、预算条件或新结构都必须由后续 Pro 另行冻结，不能在本实验中预埋。

## 数据身份准入与当前执行边界

只读 split identity audit 已在 `fdd2bcdd...` 上完成并由独立 Critic 与 N16R4 CPU Evaluator 检查。完整训练侧的
annotation、loader 与 physical 集合均为相同 200 个视频；OpenTAD `validation` 的 annotation、loader、physical、
evaluator 与历史 prediction 集合均为相同 211 个视频。ActionFormer 原始 `Test` annotation 为 212，唯一额外 ID
是 `video_test_0000270`；OpenTAD README 明确记录其因错误标注被排除。411 个期望视频均可基本解码。

审计结论为 `DATA_IDENTITY_PASS_211`，完整报告 SHA-256 为
`d7251c11935644cf8661e6bfdcfb857e29d2357cb894b7de9d8b2bd7eaf6f1ab`。Pro 已在 nonce
`DUCA-FULL-DATA-IDENTITY-ADMISSION-v001-20260831` 下正式准入完整 200-video `training` 与完整 211-video OpenTAD
`validation`，并解锁本页模型 Builder。当前仍没有新模型提交、PRE_RUN、正式作业、性能或成本结果。

种子冲突已裁决：按 `3407 → 3408 → 3409` 顺序盲执行全部三种子；六个训练单元和全部 prediction 封存前不得
读取 held-out 指标，也不得根据 3407 决定是否运行后两枚种子。

当前唯一 Builder 分支必须从 `04c35a3b...` 建立为
`feature/duca-h65-system-multibudget-exposure-v1-20260831`。只允许最小实现三档嵌套变长执行、真实 observation
计数、K384 parity、两臂配置、prediction 封存与 focused tests；禁止增加新 selector、controller、预算条件、蒸馏、
Gumbel、Mamba、Block Drop、频域模块、detector wrapper 或工作流平台。完整 Pro 任务单见
`research-wiki/sources/2026-08-31-pro-duca-full-data-identity-admission-v001.md`。

## Pro 完整数据协议终态与当前处置

新的精确 Project Pro turn `DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831` 已验证完成并选择 `REVISE`。
它撤销了上一轮 160/40 正式协议和有标签训练侧 oracle，冻结完整 200-video `training`、完整 `validation` held-out、
两臂 6,000-update 日程、一次性评测、fixed mixed-budget 直接差异和配对区间。完整报告保存在
`.cvpr-pro-lab/pro-reviews/runs/duca-full-data-comparable-protocol-v001/visible-report.md`；规范化记录位于
`research-wiki/sources/2026-08-31-duca-full-data-comparable-protocol-v001.md`。

## Gemini 只读实现咨询

`agy` CLI 的 Gemini 3.7 Flash High 已在 `effort=high`、只读 plan 模式下检查最新 Pro 原文、完整 Wiki、H65
`04c35a3b...`、数据身份审计 `fdd2bcdd...`、冻结检测器证伪 `33e4ed...` 和当前尚未实现的 Builder 分支。
它支持继续保持本页唯一实验，不引入 controller、预算嵌入、蒸馏或新架构。

对 Builder 有直接价值的风险提示是：每个成功 update 使用单一预算档；预算采样使用独立随机数流；短窗口按真实
observation 折叠；变长 packet 不得伪装成固定最大长度重型执行；K384 必须与 H65 路径一致；AMP 重放不得推进
optimizer、schedule 或 EMA 时钟。这些仍须按基座代码逐项核验，不能因 Gemini 建议而改变 Pro 冻结变量。

Gemini 另外提出的 `0.1%` 成本误差、固定文件清单、PASS 后的具体 controller 消融、FAIL 后转向空间裁剪等均是
咨询建议，不是已冻结实验合同。其“PASS 证明跨预算机制”“FAIL 证明固有非连续性瓶颈并触发项目级停止”的表述
过强；正式结论仍只能由 Pro 根据完整三种子结果在本实验的预注册边界内裁决。完整原文保存于
`research-wiki/sources/2026-08-31-agy-gemini-duca-post-admission-optimization-v001.md`。

## 最小实现与提交前验证

Builder 已在精确父提交 `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 上完成并公开实现：

- 分支：`feature/duca-h65-system-multibudget-exposure-v1-20260831`；
- 精确提交：`0d67d49c2fc4a5f50aa784f7809c0dd936492109`；
- GitHub：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/0d67d49c2fc4a5f50aa784f7809c0dd936492109`。

实现保持 K384 的历史 H65 选择与重型骨干执行路径不变；候选只从同一 H65 priority sequence 构造嵌套的
K256/K384/K512 集合。每个成功 optimizer update 使用一个 batch 内同质预算，预算时钟不消费数据增强随机数流；
6,000 次更新的冻结 occurrence 计划为 K256/K384/K512 `1454/3000/1546`。实际概率由完整训练集合中的短窗口
折叠和真实 observation 数唯一校准，当前无标签准备结果为
`p256=0.24235161911751213`、`p384=0.5`、`p512=0.25764838088248787`。

非 K384 路径按 16 帧 packet 真实缩短 VideoMAE 执行，不把补齐位置计作有效 observation；最后一个 packet 的
人工 padding 对应特征在恢复 384 点 detector 网格前被裁掉。预处理和后处理与 H65 一致：packet 化、segment 与
空间维聚合、时间拼接和线性插值。K384 或短窗口折叠到 K384 的样本继续走原始路径。

完整 211-video prediction 使用不含动作类别与时间段的推理 annotation。三个种子的
Control-K384、Candidate-K384 和 Candidate-mixed 共九份预测必须连同 checkpoint、配置、实际 observation、真实
执行时延和输入身份先封存；一次性 evaluator 在全部封存通过后才解析完整 held-out annotation 一次，并对所有视图
复用同一官方 evaluator 与同一组 10,000 次整视频 bootstrap 索引。成本报告包括数据消费等待、模型与窗口处理、
逐视频 Soft-NMS、完整 population wall-clock、Scout/VideoMAE/detector 分项时间和 GPU 峰值显存；名义 K 与 packet
padding 不替代真实成本。

提交前检查已经完成：本地 `git diff --check`、Python 编译和两个 Slurm 启动器语法通过；N16R4 的当前实现运行
25 项聚焦与恢复合同测试，结果为 `25 passed`。完整 200-video training、211-video held-out 身份、校准产物、
固定 mixed manifest 和 bootstrap 索引的无标签准备与验证通过。上述仅证明实现可执行和实验身份闭合，不是模型
有效性、mAP 或成本收益证据。尚未运行正式 PRE_RUN 或任何新训练；独立 Critic 的终态如下。

## 独立 Critic 终态

新的无实现上下文 Critic 已对精确提交 `0d67d49c...` 完成只读审查并返回 `PASS`。它确认工作树 clean、sole parent
为 `04c35a3b...`，并核验唯一变量、成功 update 预算时钟、K384 原路径、嵌套集合、真实 packet 执行、padding 特征
裁剪、完整数据边界、九份封存、一次性 annotation 解析、官方 evaluator、配对 bootstrap 和成本口径。Critic 在当前
Windows Python 中受 PyTorch `c10.dll` 初始化限制而未重复运行 pytest；该环境限制不覆盖 Builder 已在 N16R4 得到的
`25 passed` Linux 结果。当前已解除 PRE_RUN 审查门，但仍没有模型性能或效率证据。

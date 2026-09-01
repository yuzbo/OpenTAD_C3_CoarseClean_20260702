# DUCA 多预算适应：完整训练与完整留出评测协议裁决

Nonce：`DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831`

你是本课题的科学第一负责人、整体科研流程维护者和论文首脑。你独立负责科学问题、创新机制、可证伪预测、
实验路线、结果解释和论文主张。Codex 只执行你冻结后的最小实现、独立代码审查、正式实验评估和证据回传。
本轮不要沿用 Codex 的预设选择；请独立解决一项会改变论文证据有效性的正式数据协议冲突。

## 精确项目与最新 GitHub 代码身份

- ChatGPT Project：`g-p-6a91061f789881918ccd8357ca3d6c92`
- Project URL：<https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92/project?tab=chats>
- GitHub 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- H65 干净训练基座：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>
- 最新 whole-video 诊断分支：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>
- 最新诊断提交：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
- 真实变长执行与 whole-video runner：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>
- 三档预算实现：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>

`33e4ed...` 只提供已经验证的真实变长 VideoMAE 执行、packet 对齐、actual-observation 计数、K384 parity、
whole-video 评价和 producer 原始 proposal 顺序保持；新模型科学基座仍是 `04c35a3b...`。

## 已完成的上一轮 Pro 裁决

请完整读取随本 prompt 提供的上一轮 Pro 报告。它在同一精确 Project 中由 Pro 模型生成，nonce 为
`DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`。上一轮选择 `CONTINUE`，冻结了一个单变量实验：

- 固定控制只训练 K384；候选使用当前嵌套 K256/K384/K512 位置构造进行多预算训练；
- Scout、位置构造、检测器架构、损失、物理时间映射、Soft-NMS 和评价器不变；
- 两臂从同一 H65 Stage-1 `epoch_29/state_dict_ema` 开始；
- 每臂 6,000 次成功 optimizer update，训练状态重新初始化，terminal `state_dict_ema` 是唯一结果模型；
- 多预算概率按实际 observation 成本校准；
- 旧 704-state 冻结检测器路线继续 `STOP`；不加入 embedding、蒸馏、Gumbel、新 Scout/head、Mamba、
  Block Drop 或部署优化。

这些机制和训练选择可以保留、修订或停止，但请自己判断。没有代码、训练或模型结果由上一轮报告自动产生。

## 必须解决的后到人类约束

上一轮 prompt 提交后，人类明确要求：

1. 两个正式可比训练臂都必须使用项目定义的**完整 THUMOS14 训练集**，不能把 160/40、旧 40-video holdout、
   pilot、smoke 或 shortened run 作为论文主比较。
2. 设计、超参数、规则、模型选择和停止条件冻结后，两臂必须在**完整官方 held-out evaluation/test split** 上使用
   相同数据语义、annotation、类别映射、Soft-NMS、评价器和结果保存协议完成最终比较。
3. 官方留出集只用于冻结方案的最终评价，不能参与训练、checkpoint 选择、阈值/规则选择、路线选择、候选设计或
   反复窥视后修改方法。
4. 训练侧子集若仍有必要，只能作为正式实验前的诊断，且之后必须用完整训练集重新训练两条匹配臂；诊断结果不得
   冒充论文主结果。

上一轮 Pro 报告却冻结了 160-video train / 40-video development，并明确本轮不访问 official test。该数据协议因而
不能直接交给 Builder。

## 真实 split 身份冲突

当前仓库和历史记录至少存在两套不能由 Codex静默等同的口径：

- OpenTAD/DUCA 实验常把 THUMOS14 `training` 用于训练、`validation` 用于官方评估，并记录 211 个成功评估视频；
- ActionFormer 官方协议常把 `validation` 用于训练、`test` 用于评估，历史材料记录 212 个带标注测试视频。

请基于随附完整材料和代码身份，独立冻结：

- 两臂完整训练所使用的精确 config subset 名称、annotation 文件、完整视频 ID 集合和数量；
- 最终完整 held-out evaluation/test 所使用的精确 subset 名称、annotation、类别映射、完整视频 ID 集合和排除规则；
- 为什么这一选择与当前 H65/AdaTAD 公平基线可比；
- 对 211/212 差异的事实解释和唯一处理，不允许把未知信息写成已知；若材料不足，冻结一个在实现前必须完成的
  只读身份核验和明确阻断条件，而不是让 Codex猜测。

## 你必须独立冻结的正式实验顺序

请设计最小、论文优先且无测试集泄漏的顺序。至少明确：

1. 是否保留任何训练侧诊断。如果保留，说明它只回答什么机制问题、何时封存，以及为什么不影响最终全量训练；
2. 两条正式臂如何使用完整训练集，从哪个 checkpoint/state key 开始，是否保留每臂 6,000 次成功更新、相同
   optimizer/LR/EMA/seed/trainable-set 与 observation 成本匹配；
3. 在打开完整 held-out 标签或指标之前，哪些代码、配置、checkpoint、概率、门槛和停止规则必须不可变；
4. 完整 held-out 评估是一次性最终比较还是还需要另一外部确认集；不得要求用 held-out 结果继续调参；
5. 对单种子结果、配对不确定性、可发表性和后续是否训练 controller 的证据边界；
6. 明确的停止条件：什么结果关闭当前 K256/K384/K512 适应路线，什么结果才允许下一轮 controller 研究。

不要为了形式完整而加入工作流平台、复杂合同代码、通用审计框架、额外哈希系统、Mamba、Gumbel、蒸馏、
新 selector、Block Drop、TensorRT 或跨数据集实验。本轮只解决正式数据与证据协议，并给出唯一当前任务。

## 返回合同

请输出一份可直接保存的裁决，并包含：

1. `SESSION_ASSERTION`：原样回显 nonce、Project ID、H65 base 和诊断提交；
2. `SCIENTIFIC_DECISION`：只选 `CONTINUE / REVISE / PIVOT / STOP` 之一；
3. `FULL_TRAIN_IDENTITY`：精确训练 subset、annotation、完整 ID 集、数量和 H65 可比性；
4. `HELD_OUT_EVALUATION_IDENTITY`：精确留出 subset、annotation、ID 集、数量、211/212 处置和禁止用途；
5. `DIAGNOSTIC_TO_FORMAL_SEQUENCE`：训练侧诊断与全量正式训练的唯一关系，或明确删除诊断；
6. `MATCHED_TRAINING_FREEZE`：两臂起点、成功更新数、优化、随机性、成本匹配、checkpoint/EMA；
7. `FINAL_EVALUATION_AND_STATISTICS`：一次性完整评估、指标、不确定性、继续/停止门和论文证据边界；
8. `CODEX_TASK_ORDER`：只给一个当前 Builder、一个独立 Critic、一个独立 Evaluator 的最小顺序与禁止项；
9. `NEXT_RETURN`：Codex 必须返回哪些实现、正式全量训练和完整评测证据，才能再次请求科学裁决。

若现有材料不足以诚实冻结完整 ID 集，请列出唯一只读身份核验任务、确切输入和通过/阻断输出；除此之外不要把
路线选择交回 Codex。不要把文档、代码完成或作业成功当作模型有效性证据。

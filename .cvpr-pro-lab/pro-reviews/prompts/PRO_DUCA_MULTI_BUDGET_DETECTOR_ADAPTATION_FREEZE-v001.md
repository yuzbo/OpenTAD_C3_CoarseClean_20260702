# DUCA 多预算检测器适应：科学裁决与执行冻结

Nonce：`DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`

你是本课题的科学第一负责人、整体科研流程维护者和论文首脑。你独立负责科学问题、创新机制、可证伪预测、
实验路线、结果解释与论文主张。Codex 只执行你冻结后的最小实现、独立代码审查、正式实验评估和证据回传。
请独立判断，不要采纳 Codex 预设的路线，也不要为了继续项目而勉强批准实验。

本轮只处理一个项目级问题：此前冻结 K384 检测器的三档预算转移已经终止；用户随后提供一份 `REVISE`，建议
在旧停止边界之外检验“检测器只在 K384 上训练导致跨预算不适应”这一假说。请审查该新问题是否值得执行，并在
值得执行时补齐唯一、可直接交给 Codex 的训练与开发集合同。

## 路由与代码身份

- 精确 ChatGPT Project：`g-p-6a91061f789881918ccd8357ca3d6c92`
- Project URL：<https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92/project?tab=chats>
- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- H65 干净模型基座：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>
- 最新三档 whole-video 诊断分支：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>
- 最新诊断提交：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
- whole-video runner：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>
- focused test：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py>
- 未修改的三档预算实现：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>

`33e4ed...` 的科学身份只是终态 whole-video 诊断。它唯一修改密封 proposal 的重放顺序，避免 Soft-NMS
平分时的确定性漂移；没有改变 Scout、VideoMAE、Temporal Adapter、检测头或选择机制。若批准新训练，模型
基座必须是 `04c35a3b...`；只允许从诊断实现移植真实变长 VideoMAE 执行、packet 对齐、actual observation
计数、K384 bit-exact parity、whole-video 评价以及原始生成顺序保持。

## 已完成事实与停止边界

H65 固定 K384 的 30+60 参考为 Avg-mAP `65.13%`、mAP@0.7 `43.31%`。它使用
`budget_calibrated_sampling_rate`：H65 优先级调制、带覆盖下限的预算校准确定性系统采样；不是普通全局 Top-K。

最新训练侧 controller holdout 终态使用 40 个视频、124 个窗口、密封 K256/K384/K512 预测与真实 observation
成本。fixed K384 为 Avg-mAP `88.1312%`、mAP@0.7 `76.2706%`、成本 `47110`。全部 704 个合法整视频
donor-recipient 候选均完成评价：

- Avg-mAP 最优变化：`+0.6942/-0.0436` 个百分点；
- mAP@0.7 最优变化：`-0.2359/+0.4970` 个百分点；
- 联合门余量最优变化：`+0.1474/+0.4898` 个百分点；
- 同时通过 `+0.8/+1.0` 门的候选：`0/704`。

这次 falsifier 没有训练控制器、模型前向、梯度、bootstrap 或 official validation/test。它支持的结论只是：
冻结 K384 检测器后，在当前嵌套三档密封预测动作空间中没有足够的 Avg-mAP/高 tIoU 联合开发集 headroom。
它不证明动态预算一般问题失败，不证明 Top-K 不可微是根因，不证明控制器过拟合，也不证明分类天然需要低预算、
边界定位天然需要高预算。

此前 Pro 已对这一旧动作空间裁决 `STOP`。Marginal-v1、cap-release、96-state 和 whole-video 分支继续只读，
不得重跑、扩张、改门、训练旧 controller、选择后补 bootstrap 或访问 official test。

## 用户提供的新 REVISE

当前建议的新科学问题是：保持现有嵌套 K256/K384/K512 位置构造、Scout、物理时间逆映射、检测器结构、损失、
Soft-NMS、数据、评价器、checkpoint 规则和实际成本口径不变，只比较：

1. 固定预算控制：训练只使用 K384；
2. 多预算适应：训练时使用 K256/K384/K512，初始名义概率 `0.25/0.50/0.25`，再按短窗口折叠后的实际
   observation 数校准，使平均训练成本尽量匹配固定 K384。

两臂必须从同一 H65 checkpoint 开始，匹配成功更新数、优化器、学习率日程、随机种子、可训练参数集合和最终
EMA 选择规则。第一轮明确不加入预算条件嵌入、蒸馏、Gumbel-Softmax、新 Scout/head/selector、DFT、Mamba、
Block Drop、CUDA/TensorRT 或跨数据集扩展。

同一用户消息中的较早附件另有“预算原生 H65 选点 + 多预算训练”的提案；这会同时改变选点和训练分布。当前
正文要求第一轮保留现有嵌套位置构造。请你独立裁决冲突；Codex 当前没有把两者合并。

建议中的输出与门槛为：

- 分别报告 K256/K384/K512 的 Avg-mAP、mAP@0.3--0.7、proposal recall、起点/终点误差、proposal 数、
  NMS 前后假阳性和短/中/长动作结果；
- K384 安全门相对同更新数固定控制：`ΔAvg-mAP >= -0.2` 个百分点且
  `ΔmAP@0.7 >= -0.2` 个百分点；
- 在未参与参数学习和规则选择的训练侧开发集上重算等成本 whole-video oracle：
  `ΔAvg-mAP >= +0.8`、`ΔmAP@0.7 >= +1.0` 个百分点，且实际 observation 成本不高于固定 K384；
- oracle 若失败，停止当前 K256/K384/K512 动态转移路线；若 Avg-mAP 恢复而高 tIoU 仍失败，才另行讨论
  K、相邻物理间隔和局部采样密度条件，不在本轮预埋。

## 必须由你独立冻结的未决项

当前建议仍缺少两个会改变科学解释的关键决定，Codex 不得自行选择：

1. **匹配训练日程。** 请在“从 H65 terminal checkpoint 进行两臂完全相同的短期继续训练，仅作机制诊断”与
   “从冻结起点进行匹配完整训练，形成论文级比较”之间作出唯一选择，并给出精确起点 checkpoint/state key、
   每臂成功更新数或轮数、训练阶段、优化器/学习率/EMA规则、随机种子、checkpoint 与中间验证用途。若你认为
   应采用另一种最小日程，请明确说明理由和完整数值。
2. **独立训练侧开发划分。** 旧 40-video controller holdout 已被用于多轮 oracle 和规则裁决，不能被静默称为
   未参与规则选择。请给出一个不会使用 official test、且未参与本次参数学习或规则选择的确切方案：最好给出
   可复现的 ID 清单生成规则、seed、视频数量以及 train/dev 的用途与封存边界。若现有 200 个训练视频不足以
   同时满足该要求，请明确裁决可接受的替代证据设计，而不是让 Codex自行重切数据。

## 你的返回合同

请输出一份可保存的科学裁决，包含：

1. `SESSION_ASSERTION`：原样回显 nonce、Project ID、H65 base `04c35a3b...` 和诊断提交 `33e4ed...`。
2. `SCIENTIFIC_DECISION`：只选 `CONTINUE / REVISE / PIVOT / STOP` 之一，并说明旧 STOP 与新问题的边界。
3. `CAUSAL_ISOLATION`：判断第一轮应保留嵌套选点还是改预算原生选点；只允许冻结一个干预变量。
4. `TRAINING_FREEZE`：给出无歧义的两臂起点、训练长度、优化、随机性、checkpoint/EMA 与中间验证规则。
5. `DEVELOPMENT_SPLIT_FREEZE`：给出无歧义、可复现、无 official-test 泄漏的训练侧开发划分和使用边界。
6. `EVALUATION_AND_GATES`：确认或修订每档诊断、K384 安全门、等成本 oracle 门、停止规则和论文证据边界。
7. `CODEX_TASK_ORDER`：只在实验值得执行时，依次给出一个最小 Builder、一个独立 Critic、一个独立 Evaluator
   的范围、禁止项、终态产物和失败返回条件。不要要求工作流平台、复杂合同代码、哈希体系或额外审计框架。
8. `NEXT_RETURN`：明确 Codex 必须把哪些实现与结果证据返回给你，才能进行下一轮科学裁决。

没有新代码、训练或结果已经产生。不要把本 Prompt、用户建议或文档状态当作模型有效性证据。若关键上下文不足，
请直接冻结阻断条件；不要猜测。请在本轮独立完成科学取舍，不把路线选择交回 Codex。


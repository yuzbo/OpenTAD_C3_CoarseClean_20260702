---
type: source
status: discussed
updated: 2026-08-31
project: DUCA
---

# Pro 多预算检测器适应冻结裁决 v001

## 会话身份

- Nonce：`DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`
- Project：`g-p-6a91061f789881918ccd8357ca3d6c92`
- Conversation：`6a9521de-d020-83e9-a0b9-19045c8d5390`
- 模型选择器：Pro，已由浏览器运行记录验证；没有单独暴露额外 effort 控件。
- H65 模型基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- whole-video 诊断功能来源：`33e4ed137c33eef07f0452b44506a6993bdf7535`
- 完整原始回答：
  `.cvpr-pro-lab/pro-reviews/runs/duca-multi-budget-detector-adaptation-freeze-v001/visible-report.md`

首次浏览器控制尝试在提交前因登录探测失败终止，没有创建科学对话。随后同一 nonce 的一次传输恢复成功提交，
Project、conversation URL、nonce 和 Pro 模型选择均通过终态记录核验；因此科学尝试计数仍为一次。

## Pro 的科学裁决

Pro 选择 `CONTINUE`，只批准比较固定 K384 训练和嵌套 K256/K384/K512 多预算训练。Scout、位置构造、检测器
结构、损失、物理时间映射、Soft-NMS 和评价器保持不变；预算条件嵌入、蒸馏、Gumbel-Softmax、新 Scout、
Mamba、Block Drop 和部署优化均不进入第一轮。

Pro 还冻结了以下执行选择：

- 两臂都从 H65 Stage-1 `epoch_29/state_dict_ema` 开始，重新初始化训练状态；
- 每臂恰好完成 6,000 次成功 optimizer update，使用相同优化器、日程、随机种子、可训练参数与 terminal EMA；
- 多预算概率按实际 observation 均值校准，保持 `p384=0.5` 且期望成本等于固定 K384；
- terminal EMA 才是唯一结果模型，中间指标不得用于 checkpoint 或规则选择；
- Builder、独立 Critic、独立 Evaluator 依次执行，旧冻结检测器的 704-state 路线继续只读。

## 与后到人类约束的冲突

该 Pro prompt 在人类新增正式数据要求之前已提交。回答冻结的是：从 200 个训练侧视频中重切 160 个训练视频和
40 个开发视频，先在该 40-video development 上完成门控，并且本轮不访问 official test。这不是人类要求的正式
可比实验：两臂必须使用完整训练集完成匹配训练，并在完整官方 held-out evaluation split 上进行一次冻结后的
最终比较。

因此本回答的机制选择、6,000-update 建议和代码边界可作为下一轮 Pro 的输入，但其 160/40 数据协议不是当前可
执行授权。Codex 不建立 Builder、不提交 PRE_RUN 或训练，也不把 40-video 结果提升为论文主结果。需要一个新的
独立 Pro turn 冻结完整训练与完整留出评测的精确 subset 名称、视频 ID、annotation、类别映射、评价器、一次性
使用边界，以及训练侧诊断如何与正式全量训练隔离。

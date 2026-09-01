---
type: experiment
status: designed
updated: 2026-08-31
project: DUCA
---

# DUCA 多预算检测器适应

## 科学问题

在保持 H65 的当前嵌套 K256/K384/K512 位置构造、Scout、检测器结构和评价协议不变时，仅让检测器在三种预算
分布上共同训练，能否恢复跨预算检测兼容性，并在相同、无标签、等价预算的 fixed mixed-budget workload 下，
相对只在 K384 上训练的匹配控制提高 Avg-mAP 与高时间交并比性能？

这是位于此前项目级 `STOP` 边界之外的新机制检验。旧结论仍成立：冻结 K384 检测器后进行三档预算转移没有
达到预登记联合门。新实验不重跑旧 oracle，也不把跨预算表示不匹配预先写成已证实根因。

## 单一干预

- 固定预算控制只在 K384 上训练。
- 候选在 K256/K384/K512 上共同训练；冻结 `p384=0.50`，其余两档概率根据完整 200-video 训练集上的冻结
  6,000-update sample occurrence 计划和短窗口折叠后的实际 observation 成本唯一校准。
- 两臂使用同一 H65 起点、相同成功更新数、优化器、学习率日程、随机种子、可训练参数集合和最终指数移动平均
  模型选择规则。

第一轮保留当前嵌套位置构造。不得同时改成预算原生 H65 采样；否则将同时改变选点和训练分布，无法把结果归因
于检测器预算适应。

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

## 当前唯一执行项

Pro 已冻结 H65/OpenTAD 的 `training → validation` 数据语义，但 literal ID 集和 211/212 差异仍须由只读事实核验
物化。当前唯一任务是基于 `04c35a3b...` 建立最小 split identity audit，核对 annotation、config、loader、物理
视频、evaluator、历史 211 IDs 和 ActionFormer 212 来源；不得读取 held-out 动作类别、时间边界、预测或 mAP。

身份核验的 Builder 提交须经独立 Critic；Critic 通过后，独立 Evaluator 在 N16R4 CPU 上运行一次。结果无论通过
或阻断，都先返回 Pro；在 Pro 数据身份准入前不建立模型 Builder、不提交 PRE_RUN 或训练。当前仍没有新模型代码、
正式作业、性能或成本结果。

## Pro 完整数据协议终态与当前处置

新的精确 Project Pro turn `DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831` 已验证完成并选择 `REVISE`。
它撤销了上一轮 160/40 正式协议和有标签训练侧 oracle，冻结完整 200-video `training`、完整 `validation` held-out、
两臂 6,000-update 日程、一次性评测、fixed mixed-budget 直接差异和配对区间。完整报告保存在
`.cvpr-pro-lab/pro-reviews/runs/duca-full-data-comparable-protocol-v001/visible-report.md`；规范化记录位于
`research-wiki/sources/2026-08-31-duca-full-data-comparable-protocol-v001.md`。

# DUCA Marginal-v1 cap-release 终态科学裁决

Nonce：`DUCA-MARGINAL-CAP-RELEASE-TERMINAL-ADJUDICATION-v001-20260831`

你是本课题的独立科学负责人、路线设计者与最终审查者。Codex 只负责忠实执行你冻结的任务。本轮请基于下面完整证据，独立决定当前 DUCA 研究下一步；不要迎合 Codex，也不要把运行成功、代码审查或训练侧诊断扩大为论文结论。

## 最新公开代码真值

- GitHub 仓库：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- 最新实现分支：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-falsifier-v1-20260831
- 精确提交：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f
- cap-release runner：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f/tools/bata/run_duca_marginal_frozen_h65_probe.py
- 动态预算分配器：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f/opentad/models/duca/dynamic_budget.py
- 聚焦测试：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f/tests/test_duca_marginal_budget.py

以上 GitHub 链接是本轮最新实现真值。提交 `d2fad7c0...` 只增加独立的 `max_changed_fraction=1.0` 只读汇总入口与测试；默认 `0.5` 路径、K256/K384/K512 producer、模型、数据、NMS、评估器和门槛均未改变。分支已推送且干净；N16R4 14 项聚焦测试通过，独立 Critic 返回 PASS。

## 冻结科学问题与前置裁决

当前诊断冻结 H65 Scout、VideoMAE-S、Adapter、ActionFormer、损失、NMS 与评估器，通过同一 H65 priority sequence 构造嵌套的 K256/K384/K512 真实 observation 集合。在同一视频内严格保持总实际 observation 预算 `sum_i min(V_i,384)`，使用训练侧 40-video utility holdout 的真实反事实效用，检验跨窗口重新分配重型计算的 oracle headroom。

此前 50% 改变窗口上限的结果是：

- Fixed-H65-384：Avg-mAP `88.131197%`，mAP@0.7 `76.270583%`；
- capped oracle：Avg-mAP `88.856786%`，mAP@0.7 `76.999587%`；
- 增益：`+0.725589/+0.729004` 个百分点；
- 分配 K256/K384/K512=`11/102/11`，总实际 observation=`47110`，预算误差为零。

该结果介于预注册强 headroom 门 `+0.8/+1.0` 和无 headroom 边界 `<+0.3/<+0.5` 之间。你上一轮裁决为 `REVISE`，唯一允许的后继是解除 50% 改变窗口上限，在相同密封产物上只读计算 `max_changed_fraction=1.0`。你同时冻结：若两项点门没有同时通过，停止当前 Marginal-v1 机制且不运行 bootstrap；只有两项都通过才执行 seed 3407、10,000 次整视频配对 bootstrap。

## 唯一 cap-release 终态

唯一 Evaluator Job `1262117` 于 `2026-08-31T05:53:33+08:00` 启动，`05:54:25+08:00` 以 `COMPLETED 0:0` 结束。它只在 CPU 上读取原密封产物；没有执行 detector/Scout forward、模型训练、utility-head 拟合或 official test。

终态结果：

- 原 Fixed-H65-384 与 capped oracle 的所有点值复现误差均为 `0.0` 个百分点；
- released oracle：Avg-mAP `88.558507%`，mAP@0.7 `76.720863%`；
- released oracle 相对 Fixed-H65-384：`+0.427310/+0.450280` 个百分点；
- released oracle 相对 capped oracle：`-0.298279/-0.278724` 个百分点；
- released 分配 K256/K384/K512=`17/90/17`；改变 11 个视频、34 个窗口；
- 总实际 observation=`47110`，预算误差为零；
- 两项强门均失败，`strong_gate_pass=false`；
- 按冻结规则 `paired_interval_required=false`，0 次 bootstrap；
- runner 终态：`CAP_RELEASE_POINT_GATE_FAILED_STOP_CURRENT_MECHANISM`。

原始终态 JSON（随本问询完整附带）：
`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-d2fad7c0-job1262117/oracle_cap_release_result.json`

SHA-256：

- terminal result：`fb3c122e233952a4165c2ca9a6ff3d2839b8e0d108c977443786714ec0cf6ed4`
- original probe：`8d6df7240c8b81b4d6d9aa8ff98bae530d6823ddd1d411bed47ce983ebd94925`
- K384 producer：`1d668d4e5eb4b5ef3c1057c97ec63cc2c1eed3c0e62297290520063b4e1ec38f`
- K256 producer：`6dc8893a41b5c8132b176f32133ffc2f48a5491146385c147b8227167608a309`
- K512 producer：`c7fa06258c07163d0906b512a78e367c27607c64fc41b28bce9fe51fbd0815d7`

Producer 来源仍为 `f87555f7da362fe1a20d4ca08f7a68c975ed8280`；原 capped summary 来源为 `f67d96fdf68a295eaa7f678f3dfc125530828889`；cap-release runner 与终态来源为 `d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`。配置、checkpoint、annotation、类别映射和 VideoMAE 预训练身份不变。

## 证据边界

本结果是 40 个训练侧 holdout 视频上的真实效用 oracle 机制诊断，不是 learned allocator、official validation/test、统计显著性、端到端成本或论文主结果。预注册规则已经停止当前 Marginal-v1 机制，但这不能自动外推为所有动态预算、coverage 或物理时间方法无效。Codex 没有选择后继路线，也没有授权 predictor、official test、重训或新实验。

## 你的唯一任务

请独立完成以下工作：

1. 先给出且只给出一个总裁决：`CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`。
2. 判断当前负结果最支持什么、仍不能支持什么；区分“当前三档 oracle 重分配机制缺乏足够 headroom”与更广泛的动态计算假设。
3. 解释为什么解除改变窗口上限后 oracle 反而变差。请判断这更可能反映组合分配约束、真实效用的非加性、预算档位过粗、窗口间干扰、评估聚合性质，还是当前问题本身缺少可利用空间；不要做没有证据的唯一因果断言。
4. 独立决定 DUCA 现在应停止整个方向、修订科学问题，还是转向一个新的可证伪机制。不要受 Codex 既有实现偏好影响。
5. 只下达一项当前任务。它必须直接检验你认为最关键的不确定性，并写清：科学问题、机制或分析对象、最小实现边界、对照、公平性、数据与 split、主要指标、最便宜 falsifier、继续/停止门、禁止项、Builder→独立 Critic→Evaluator 的职责，以及绝对完成时限。
6. 明确当前证据能否进入论文；若只能作为负结果或内部路线淘汰，也请给出准确表述。

不要要求 Codex先替你选线，不要同时下达多条探索，不要把流程工程、额外合同、哈希系统或重复审计当成科研任务。优先选择能最快减少论文核心不确定性的真实分析或实验。

回复必须包含本 nonce：`DUCA-MARGINAL-CAP-RELEASE-TERMINAL-ADJUDICATION-v001-20260831`。

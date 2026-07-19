---
type: experiment
node_id: exp:phystime-p0-fullprecision-nms-replay
title: "PhysTime P0 frozen full-precision NMS replay"
idea: idea:phystime-tad-2
status: tested
verdict: postprocessing_confound_closed_main_effect_preserved
confidence: full_remote_suite_passed_single_frozen_run
metrics: "EMA fullprecision selected/physical Avg-mAP 41.2830/57.6087%; delta +16.3257 pp. Rounding changes Avg-mAP by -0.0366 to +0.0338 pp; validity filtering changes 0."
provenance: "runtime c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c / tree 0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338; run root phystime_p0_fullprecision_c2cfcfa_20260720_025843_+0800"
added: 2026-07-20T00:00:00+08:00
---

# PhysTime P0 全精度 NMS 冻结重放

## 目的

本实验不训练新模型，也不改 Q、loss、采样器或网络结构。它只回答两个发布级问题：

1. 旧评估链在跨窗口 NMS 前把 segment/score 舍入到 2/4 位，是否改变抑制、排序和最终 mAP？
2. 是否存在非有限、非正时长，或由舍入诱发的零时长 proposal；显式过滤会造成多大影响？

冻结来源是 full60 commit
`0dc5851a8feb12b97d16bdb5ea8fc60e9273d132`、tree
`bddc9b9386604d00d213275a47ce7997b35d3f4c` 的 selected-axis 与
physical-metric epoch-59 checkpoint。旧 `41.28/57.57%` 结果保持原证据
等级，不因本实验尚未完成而撤销。

## 固定设计

- 两臂：selected-axis、physical-metric。
- 两套权重：online、EMA。
- 每套权重只做一次真实 raw-video 推理，保存跨窗口 NMS 前的全精度 proposal。
- 同一份冻结 proposal 重放四种模式：
  - `legacy_unfiltered`
  - `legacy_filtered`
  - `fullprecision_unfiltered`
  - `fullprecision_filtered`
- 四个 GPU 重放作业全部通过后，才运行一个独立 CPU 总验证器并生成
  `P0_SUITE_COMPLETE.json`。

## 实现合同

- `SingleStageDetector` 不再提前舍入科学输出。
- 跨窗口 class id 固定为 `torch.long`。
- 合法性分为三层：原始 proposal、舍入后实际 NMS 输入、NMS 后最终输出。
- filtered 模式在舍入后过滤；unfiltered 模式发现任一非法 proposal 时在 NMS
  前 fail-closed。
- direct result、metrics、pre-cross artifact 和 audit 都必须绑定 checkpoint
  epoch 59、runtime commit/tree 与文件 SHA256。
- EMA 的 `legacy_unfiltered` 必须逐预测、逐指标复现旧 full60 结果。
- 单臂 validator 独立固定四模式定义，并重算 evaluator、文件哈希、canonical
  prediction hash、proposal 数量与全部 delta。
- 总验证器独立绑定四份 `P0_COMPLETE.json`，重算 physical-minus-selected、
  EMA-minus-online、舍入/过滤主效应，并报告抑制/排序差异、边界位移和按
  GT 时长四分位分层的 proposal recall@0.5/0.7/0.9。

## 当前验证

- 纯部署、总验证器与旧 track validator：本地 `25 passed`。
- Python 语法、Bash `-n` 与 `git diff --check`：通过。
- 数值测试在本机因 Windows PyTorch `c10.dll` 初始化失败无法收集；该测试
  已是远端真实 gate 的硬依赖，不能被跳过。
- 第一轮独立 max 审查给出 HOLD，并发现舍入后过滤顺序、总聚合器、
  validator 独立性、direct epoch、坐标语义绑定和对抗测试缺口；这些问题
  已完成代码修复。
- 后续复审发现 source config canonical hash、artifact/run-dir 绑定、计数守恒、
  普通配置兼容性、source dataset 路径和边界位移诊断等缺口；均已修复。
- 最终限定复核结论为 `DEPLOY`，剩余 P0/P1 为零。提交器会从冻结 full60
  两臂共同绑定的 `g1a_gate.dataset_manifest` 自动恢复真实数据路径，不再
  使用另一个 raw-video 目录作为默认值。
- 首次提交预检被集群策略拒绝：该集群只接受 `--gpus=1`，不接受
  `--gres=gpu:1`，且没有独立 CPU partition。修复后 gate、四个 replay 与
  suite 均申请 1 卡；suite 只使用 CPU 验证逻辑，并在部署清单中显式记录
  `suite_validator_uses_cuda=false`。
- 重试期间实际生成的旧 gate `1174679` 在 focused tests 阶段 fail-closed：
  `31/33` 通过，两个失败分别是测试夹具不支持 ConfigDict 式赋值，以及
  DDP gather 对首个 rank 列表的原地扩展污染测试期望。生产合并现复制首个
  rank 列表，测试夹具改用真实 `ConfigDict`；旧依赖作业
  `1174680–1174683` 已取消，不属于实验结果。
- 最终运行锚点：commit `c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c`，
  tree `0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338`，clean snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_phystime_p0_c2cfcfa_20260720`。
- 正式 run root：
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_p0_fullprecision_c2cfcfa_20260720_025843_+0800`。
  gate `1174688` 正在运行且远端 focused tests 已 `33 passed`；四个 replay
  `1174689–1174692` 依赖 gate，suite `1174693` 依赖全部四臂。

## 状态边界

当前状态为 `tested`，不是新的训练结果，也不是 `paper_ready`。本实验只关闭
冻结 epoch-59 评估链中的舍入与 proposal 合法性混杂；它不提供多种子、第二
数据集、计算成本或新模型有效性证据。

## 终态结果

正式 DAG 全部正常完成：gate `1174688`、四条冻结回放
`1174689–1174692` 与独立 suite `1174693` 均为 `COMPLETED 0:0`。
远端 focused tests 为 `33 passed`；四份 `P0_COMPLETE.json` 均
`validation_pass=true`，最终 `P0_SUITE_COMPLETE.json` 也为
`validation_pass=true`。

| 冻结臂 | legacy Avg-mAP | fullprecision Avg-mAP | 全精度减 legacy |
| --- | ---: | ---: | ---: |
| selected-online | 41.293237 | 41.256604 | -0.036632 |
| selected-EMA | 41.283790 | 41.283021 | -0.000769 |
| physical-online | 57.568992 | 57.555581 | -0.013411 |
| physical-EMA | 57.574915 | 57.608685 | +0.033770 |

fullprecision-filtered 的原始 IoU 指标如下：

| 冻结臂 | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg-mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected-online | 64.5045 | 56.3893 | 42.6635 | 27.8208 | 14.9050 | 41.2566 |
| selected-EMA | 64.8555 | 56.3456 | 42.6152 | 27.7450 | 14.8538 | 41.2830 |
| physical-online | 77.0402 | 70.5574 | 62.0749 | 48.5937 | 29.5117 | 57.5556 |
| physical-EMA | 77.2122 | 70.4557 | 62.5761 | 49.0066 | 28.7927 | 57.6087 |

关键审计事实：

- 每条臂均有 `1,584,000` 个 NMS 前 proposal、`422,000` 个 NMS 后预测；
  非有限值、非正时长、非法标签和 malformed proposal 全为 `0`。
- filtered 与 unfiltered 的预测、指标和 proposal 级匹配完全相同，过滤效应
  严格为 `0`。
- legacy-EMA 逐指标复现冻结 full60 的 `41.283790/57.574915%`，证明
  checkpoint、数据、评估器和 replay 绑定正确。
- 舍入确实改变局部决策：全精度与 legacy 的 IoU>=0.5 一对一匹配率为
  `99.62%–99.78%`，每臂有 `947–1,605` 个预测无法配对；匹配预测的边界
  位移中位数约 `0.0025s`、P95 小于 `0.00482s`。但最终 Avg-mAP 绝对变化
  最大仅 `0.0367` 个百分点。
- physical-EMA 相对 selected-EMA 的 fullprecision Avg-mAP 仍为
  `+16.325664` 个百分点；legacy 下为 `+16.291125`。因此旧舍入最多只改变
  两臂差值约 `0.0345` 点，不能解释 physical-metric 的主效应。
- 在 fullprecision-filtered 最终预测上，physical-EMA 相对 selected-EMA 的
  proposal recall@0.7：全体 `+12.36` 点、短动作（<=1.7s）`+31.59` 点、
  中等动作 `+6.93` 点、长动作 `+3.75` 点。该诊断包含分类、排序与 NMS 的
  联合作用，只能定位后续机制分解重点，不能单独证明秒域 assignment 的因果性。

最终 suite SHA256：
`afb3e300424a57eb590a21129217e040677dc875fdede3be344352dc2bd268e7`。

## 裁决与下一步

P0 关闭了发布级后处理混杂：未来 PhysTime 主实验应显式使用全精度
cross-window NMS；历史 legacy 结果保留以便复现。当前结果清晰、验证器通过，
无需为 P0 再发起 Pro 讨论。

下一项决定性任务是冻结 decode cross-replay，然后在原 Q192 结构内做
UU/UP/PU/PP 因子化：首字母表示 decode/回归坐标轴，次字母表示 assignment
坐标轴。它必须继续冻结 checkpoint、候选和评估协议，先区分收益来自秒域
assignment 还是秒域 decode；不得同时加入 Q-lift、新 loss、采样器或新训练。

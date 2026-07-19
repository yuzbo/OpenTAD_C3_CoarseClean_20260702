---
type: experiment
node_id: exp:phystime-p0-fullprecision-nms-replay
title: "PhysTime P0 frozen full-precision NMS replay"
idea: idea:phystime-tad-2
status: experiment_running
verdict: real_gate_running_replays_dependency_blocked
confidence: remote_focused_tests_passed_full_gate_pending
metrics: "NA; this experiment does not train and has not produced a remote completion artifact."
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

当前状态为 `experiment_running`，不是 `empirically_supported` 或
`paper_ready`。远端 gate 未通过时，四个 replay
不得视为有效；任一 unfiltered 模式因非法 proposal 阻断时，必须先报告具体
视频、阶段与计数，再决定是否需要 Pro 讨论。

# PhysTime-TAD 论文级实验计划（现行版）

**更新日期**：2026-07-23  
**当前任务**：完全离线、固定无学习不规则采样下的稀疏 TAD 检测头。不是 DUCA、在线 TAD、选帧插件或 ChronoTransport。

## 研究问题与主张边界

给定同一组不规则 K 个原始视频观测、同一 VideoMAE/AdaTAD 载体和同一评测器，真实物理秒坐标能否改善时序动作定位，尤其是高 IoU 边界定位。

`K` 是原始观察槽位，`J` 是 VideoMAE 原生 tubelet token，`Q` 是检测候选；三者必须分别报告。GT、预测和评测始终使用原视频秒坐标。当前不得声称“首个 continuous-time TAD”、dense SOTA 或论文已完成。

当前已验证但有限的历史锚点是：K384/J192/Q378、THUMOS14、seed42、60 epoch 下，selected-axis/physical-metric 为 `41.28/57.57 Avg-mAP`。它是 `full60-single-seed-supported`，不是 paper-ready。

## 现行执行总则

1. 先证明生产推理、capture 与 replay 是同一个数值程序，再做任何新训练。
2. 当前修复只能改变 replay 传输/审计合同；不得改变模型、采样、Q、assignment、NMS、loss、scheduler 或 evaluator。
3. 每一阶段必须绑定 commit/tree、config、data、VideoMAE、checkpoint、state_dict、runtime、Slurm token 和原始结果哈希。
4. 任一 hard gate 失败即 fail-closed：下游训练不启动，正式 mAP 写 `NA`。

## Phase R00：冻结 decode-replay 合同修复

### 为什么先做

真实 gate `1175820` 已证明：生产 CPU top-k 使用 source `float16` 分类分数，capture 却将其上转为 `float32`。数值扩展无损，但大量同分候选的排序和 top-k 成员可改变。因此旧 replay 不是生产推理的合法副本。

### 实现 A-STRICT-SOURCE-DTYPE

- 只对 ordering-sensitive `cls_scores` 强制 source-dtype transport：当前失败条件应为 `torch.float16 -> numpy.float16 -> CPU torch.float16`。
- 生产的 `SingleStageDetector.post_processing`、Soft-NMS、evaluator、checkpoint 和训练完全不改。
- 逐张量记录 source/stored/replay/ordering dtype；其他浮点张量采用明确的逐张量合同，不能用 blanket float32 或 blanket source-dtype 假设替代验证。
- 新 schema 拒绝 `source=float16/stored=float32` 的旧 ordering-sensitive artifact。
- 增加 ordered hash、top-k candidate-ID ordered/set hash、tie-boundary、first-diff、symmetric-difference 和失败诊断 artifact。

### R00 DAG

| 阶段 | 内容 | 通过条件 | 失败动作 |
|---|---|---|---|
| R00.1 | 本地 focused tests | dtype roundtrip、tie/top-k、capture-invariance、failure artifact 全通过 | 修代码，不提交 Slurm |
| R00.2 | 目标集群 CPU preflight | source/stored dtype、runtime fingerprint、固定内容 hash 全通过 | 停止 |
| R00.3 | 四条件 capture micro-gate | capture off/on 的 direct 输出完全一致 | 停止 |
| R00.4 | 四条件 CUDA native exact gate | selected/physical x online/EMA 全部 ordered exact | 停止，mAP=NA |
| R00.5 | 四条正式冻结 replay | 每臂先 P0 direct re-anchor，再 U/P decode | 任一失败时 suite 无效 |
| R00.6 | suite | 独立重算、哈希、DAG 与日志全部通过 | 保持 `tested`，不训练 |

R00 通过仅说明冻结 decode-axis replay 可审计；不产生 paper-ready 主张。

## Phase R01：冻结 decode-axis 机制证据

固定 selected-axis/physical-metric 的 epoch-59 online/EMA checkpoint。对每个 checkpoint 用同一捕获张量分别执行 uniform-axis 与 physical-axis 解码，得到八个结果条件。

**要回答的问题**：不改变网络权重，只改变可离线重算的 decode/regression 几何时，结果如何变化？

**解释边界**：同一 checkpoint 的 `decode-P - decode-U` 是合法的冻结推理干预；不同 checkpoint 的差和差中之差仅为描述性诊断，不能冒充训练因果效应。

## Phase R02：固定 Q192 的训练因子化

仅在 R00/R01 完全通过后开始。固定 K384/J192/Q378、同一采样、数据、预训练权重、优化器、日程、seed、评测器与真实秒 GT。禁止 Q-lift、插值、新 support 模块或新采样器。

| 条件 | decode/regression/inside-GT 轴 | assignment eligibility 轴 |
|---|---|---|
| UU | uniform | uniform |
| UP | uniform | physical |
| PU | physical | uniform |
| PP | physical | physical |

主效应预注册为：

```text
Delta_decode = ((PU - UU) + (PP - UP)) / 2
Delta_assignment = ((UP - UU) + (PP - PU)) / 2
```

首轮仅部署严格 matched pilot；只有四臂 artifact、稳定性、高 IoU 方向和停止条件都明确，才批准 full60 与多 seed。

## Phase R03：无训练 Q-density 反事实

固定 R02 的 Q192 模型，在不增加原始观察、不训练模型的前提下测试候选密度的 subcell replay。只报告 pre-NMS/oracle 高 IoU 覆盖和候选数量。

**目的**：验证 Q 是否确为瓶颈。只有出现明确、可审计的高 IoU 候选覆盖收益，才允许讨论训练型 Q-lift；否则 Q384、插值、cross-attention 继续冻结。

## Phase R04：统计、强基线、成本与鲁棒性

只有 R02/R03 给出清晰机制后才启动：

- 相同机制臂的多 seed，报告视频级 bootstrap CI；
- dense AdaTAD 与旧 K384 random ActionFormer 的协议审计/必要时复现；旧 `63.61` 仅是历史锚点，不能直接作为当前因果对照；
- batch=1 端到端 latency、显存、FLOPs、VideoMAE/TIA/head/NMS 分解；
- K 预算和 uniform/random/bursty-gap 采样鲁棒性。

预算、采样模式和第二数据集不是第一轮部署内容。

## Phase R05：外部有效性

仅在多 seed、成本和鲁棒性通过后，先审计 ActivityNet raw-video/FPS/window/evaluator 合同，再决定是否部署 dense、UU/PP 或其他已证明有效的最小比较。

## 当前状态

当前只执行 **R00.1：A-STRICT-SOURCE-DTYPE 代码与 focused tests**。未提交任何新 Slurm 作业，未启动训练，也没有新的 mAP。

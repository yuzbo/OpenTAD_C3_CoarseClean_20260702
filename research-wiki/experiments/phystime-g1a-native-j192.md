---
type: experiment
node_id: exp:phystime-g1a-native-j192
title: "PhysTime G1a native-J192 matched temporal-metric control"
idea: idea:phystime-tad-2
status: tested
verdict: pending_fixed_snapshot_real_gate_and_pilot
confidence: pending
metrics: "No mAP yet. Any later raw values must be recorded only in docs/evaluation/results.md."
provenance: "docs/superpowers/specs/2026-07-13-phystime-p0-rebuild-design.md"
added: 2026-07-13T00:00:00+08:00
---

# PhysTime G1a Native J192

## Question

在相同 raw RGB K384、native VideoMAE J192、ActionFormer 基础网格 Q0=192、六层总候选 QΣ=378、参数、初始化、优化器、assignment 与后处理下，只把均匀 rank-derived 秒轴替换为原生 tubelet 物理秒轴，是否产生可解释差异？

## Implemented Contract

- K384 输入槽、J192 原生 token、Q0 与 QΣ 分开审计；
- 不做 J192→Q0=384 feature/query lift；
- 完整记录 observed 与 padding-repeat patch 输入槽，以及 chunk 内 padding 可能影响有效 token 的风险；
- 两臂 GT、候选、回归、decode、NMS 与评价均在绝对视频秒坐标；
- 两臂只允许 `uniform_rank_seconds` 与 `physical_time_seconds` 的坐标张量不同；
- static contract、G0、real THUMOS CUDA gate 与 pilot artifact 逐级绑定 commit/tree/config/checkpoint/dataset/hash；
- 三步 gate 必须在两臂上使用相同真实样本、相同像素张量、相同初始化与优化器，并执行正式滑窗 NMS/evaluator；
- pilot 固定为 6 epoch，只有真实预测、有限 IoU-wise mAP 与 `epoch_5.pth` 全部通过原子验收后才生成完成记录。

## 2026-07-13 诊断修复

- 原始 AdaTAD 的特征插值本身并不非法；G1a 暂时取消 lift，是为了把 `K=384` 原始观测、`J=192` tubelet token 和 `QΣ=378` 候选严格分开。若 G1a 存活，G1b 可在两臂同时恢复同一个中性 `J192 -> Q0=384` lift，但不得把插值点称为新增观测。
- 扩展回归发现并修复 `AnchorFreeHead` 中的真实候选 mask 错误：`selected_center` 原先是 `point[:, 0]` 的视图，写入物理中心后被同步改写，导致物理秒值错误地与 selected count 比较并删掉合法候选；现已显式 clone，mask 在 rank 域计算，回归在秒域计算。
- 修复已删除变量 `dense_valid_len` 的残留引用；debug 元数据现在明确记录 `domain_end - domain_start`。
- canonical FPS、end-exclusive window domain、上游采样 provenance、结构性 padding lineage 与 per-sample effective query count 均已 fail-closed。
- VideoMAE 严格尾部隔离已深入 patch/tubelet mask、每层 attention 的 K/V、每个 residual/MLP、每个 TIA 卷积与 final norm；反事实 padding 像素和无效输入梯度测试通过，同时 all-valid 路径与旧实现逐值一致。
- 真实 gate 的尾部样本与 evaluator 已统一改用 `dataset.test`，不再用 train/validation 样本冒充 test 推理闭环；gate 还要求 train/test 全量 MP4 的 decoder FPS、帧数和 annotation duration 使用同一 fail-closed 合同。
- 数据集指纹已从文件名/大小升级为逐文件完整 SHA256 与 Merkle root；pilot 验收会真实反序列化 checkpoint，检查 epoch/state/optimizer/scheduler/有限性，并从 manifest 配置重新运行 evaluator 比对 metrics JSON。
- 对 THUMOS14 全部 411 个 MP4 的预部署时间基准审计显示：annotation 与 decoder 的最大相对 FPS 偏差约 1.12%，最大帧数偏差为 0；G1a 因而把 FPS/时长容差固定为 1.25%，帧数相对容差固定为 0.01%，正式 gate 会逐视频复核而不是信任该离线结论。

## Evidence Level

当前为 `tested`：本地编译与 diff 检查通过；本机 PyTorch 受既知 `c10.dll` 初始化故障阻断，模型测试转移到远端 Linux/Torch 临时树执行。新旧 PhysTime/C3、padding isolation、timebase、gate、artifact 与部署回归合计 `116 passed`。这仍不是正式实验：代码尚未形成最终 clean commit/fixed snapshot，真实 THUMOS CUDA gate、六 epoch pilot 与 mAP 尚未完成，因此不能声称 `experiment_running`、`empirically_supported` 或有效。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

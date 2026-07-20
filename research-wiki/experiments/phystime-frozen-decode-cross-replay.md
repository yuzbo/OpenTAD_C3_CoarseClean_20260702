---
type: experiment
node_id: exp:phystime-frozen-decode-cross-replay
title: "PhysTime frozen dual-axis decode cross replay"
idea: idea:phystime-tad-2
status: implemented
verdict: awaiting_real_gate_and_remote_suite
confidence: software_contract_only
metrics: "NA; no formal remote replay has completed."
provenance: "full60 source 0dc5851/bddc9b9; P0 runtime c2cfcfa/0b78dd4; implementation commit pending clean snapshot"
added: 2026-07-20T00:00:00+08:00
---

# PhysTime 冻结双轴解码交叉回放

## 为什么做

full60 的 physical-metric 比 selected-axis 高约 16.29 个 Avg-mAP 点，P0 又证明
这个差异不是旧的 NMS 提前舍入造成的。但现有两臂同时改变了训练几何和推理解码
几何，仍不能回答“只把同一组网络输出放回物理时间轴，性能是否会改变”。

本实验不训练新模型。它固定 full60 epoch-59 的 selected/physical checkpoint，
分别读取 online/EMA 权重；每个条件只跑一次真实 raw-video 网络推理，捕获同一组
分类 logits、分类分数、非负回归距离、候选 mask、基础 point grid 和两套时间轴，
随后从同一 artifact 分别执行：

- `uniform_rank_seconds` 解码；
- `physical_time_seconds` 解码。

## 四个冻结条件

| 条件 | 训练来源 | 权重 |
| --- | --- | --- |
| selected-online | selected-axis | online |
| selected-EMA | selected-axis | EMA |
| physical-online | physical-metric | online |
| physical-EMA | physical-metric | EMA |

每个条件内部的 `decode-P - decode-U` 是冻结 checkpoint 下的推理解码轴干预。
不同 checkpoint 之间的差值和差分之差只作描述，不能称为训练因果效应。

## 已实现合同

- 捕获功能默认关闭，不改变普通训练/测试路径。
- 同一条件只产生一份 pickle-free NPZ；数组逐项记录 dtype、shape 与 canonical
  SHA256。
- 原始 AMP 张量 dtype 单独登记；存档统一上转 float32，但不把 float32
  `sigmoid(logits)` 复算冒充生产 AMP 真值。
- 捕获内存设置 8 GiB 估计峰值硬上限；Slurm 作业申请 32 GiB 内存。
- native proposals 只作为重建误差审计参照，禁止覆盖重建 proposals。
- native 轴必须逐预测复现 direct pre-cross、最终结果和 mAP，否则 fail-closed。
- 当前 capture-enabled direct 还必须逐条件精确复现已审计 P0 direct 的
  pre-cross、最终结果和 mAP，避免“新 direct 与新 replay 一起漂移”仍自证通过。
- 除新增 capture 开关外，推理语义哈希必须与 P0 config 完全一致。
- selected/physical × online/EMA 四个真实窗口 gate 全部通过后，才允许四个正式
  replay 作业运行。
- 四条件 gate 在下游提交前比较共享视频/窗口/采样序列、U/P axis、base
  point/mask、native mask、class map 和各自 online/EMA state hash。
- 数据集全文件内容、VideoMAE、full60 source、P0 suite/gate、checkpoint、
  state_dict、config、sbatch、Slurm ID/依赖和日志均用哈希或身份链绑定。
- 首次 `sbatch` 前由纯 CPU preflight 重新计算全部不可变内容哈希；作业使用唯一
  DAG token/comment，网络重试先查询并接管已接受作业，避免重复提交。
- 来源数值语义明确记录为：AMP 产生张量，存档上转 float32，CPU float32
  重算 decode；不是完整复刻 autocast 算术。
- suite 使用生产语义的重复后处理/evaluator 复算，不冒充外部独立 evaluator。
- 时长分层 recall 基于最终检测结果，明确不是 pre-NMS proposal recall。

## 审查中发现并修复

第一轮独立审查发现 native 解码重建后又被捕获的 `native_proposals` 覆盖，会让
原生等价验证形成自证。该旁路已经删除并加入反向测试。审查还要求补齐数据和权重
内容哈希、四条件 gate、state_dict 哈希、内存预算、Slurm DAG 身份、日志扫描及
非因果指标命名，均已进入当前实现。

第二轮审查发现提交前内容未重算、`sbatch` 重试不幂等、source report 漏
`return`、当前 direct 未锚定 P0、配置语义与共享观测契约 fail-late 等问题。
当前实现已补齐 preflight manifest、唯一 DAG token、真实 scontrol/sacct
快照、P0 direct/semantic 双重锚定、四条件 state/observation 早期门禁、原子
状态文件和致命日志扩展；这些修复尚待远端 fixed-conda gate 实证。

第三轮独立复审发现一个会让真实 gate 必然误失败的契约混淆：
`window_sequence_sha256` 包含 `native_coordinate_mode`，因此
selected-axis 与 physical-metric 按设计不同，不能作为四条件共享观测哈希。
当前修复把共享稀疏观测、两套时间轴、mask、base points 和计数保留在
`observation_contract`；把原生轴与窗口哈希放入 `axis_window_contract`，
只要求同一 arm 的 online/EMA 完全一致，并要求 selected/physical 的轴特有
窗口哈希不同。行为测试覆盖了跨 arm 合法差异和同 arm 非法漂移。

第四轮部署复审发现两个会污染 Slurm 证据链的缺口：临时文件锁在提交器退出后
不能阻止同一 DAG token 被另一个运行目录接管；`sbatch` 响应丢失时，作业若
延迟出现在 `squeue/sacct`，单次查询后重试可能重复提交。当前实现新增全局
owner manifest，永久绑定 `token/run_root/commit/tree`，并与运行目录内 owner
双向校验；同 token 换目录会在任何队列查询或取消前失败。失败或非数字
`sbatch` 响应后先做有界可见性轮询。进一步复审确认“轮询预算耗尽后重投”
仍不具备服务端幂等，因此最终实现会在 `sbatch` 前持久化提交意图；未找到
唯一作业时保留 ambiguous 状态并退出，后续恢复只查询、绝不自动重投。测试
覆盖永久 owner 冲突、立即接管、预算内延迟可见接管、超预算后持续阻断、
记账可见后的恢复接管和重复 comment 拒绝。
`resolved` 与 `fatal` 也是永久状态：resolved 作业暂时不可见时只允许等待
记录的 Job ID，fatal token 永久拒绝复用。suite 会核对 gate、四个 replay
和自身共六个 resolved marker 与 `jobs.tsv` 的 token/comment/Job ID，并拒绝
遗留 ambiguous/fatal 文件。

## 当前状态与停止条件

当前状态只能是 `implemented`，正式 mAP 为 `NA`。本地纯部署测试和语法检查通过；
本机 Windows PyTorch 因 `c10.dll` 初始化失败，数值测试必须由远端固定 conda gate
执行。最终独立部署复审已给出 `DEPLOY`，P0/P1 为 0；该裁决只授权真实 gate，
不等于 gate 已通过或正式 replay 已完成。

若四条件真实 gate 中任一 native replay 不能精确复现 direct 结果，立即停止正式
DAG并发起 Pro 讨论；不得用捕获 proposals 替代重建来“修复”等价。只有 gate 和四
份 completion、suite 全部通过，状态才可升为 `tested`。通过后也只决定是否进入
Q192 UU/UP/PU/PP 训练设计，不产生 `paper_ready` 结论。

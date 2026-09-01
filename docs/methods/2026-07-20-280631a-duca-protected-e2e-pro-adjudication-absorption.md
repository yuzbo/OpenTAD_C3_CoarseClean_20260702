# DUCA Protected-E2E Pro 裁决吸收与复核

## 来源

- 原始回复：
  `docs/methods/reviews/2026-07-20-280631a-duca-protected-e2e-pro-adjudication-raw.txt`
- 原始回复 SHA-256：
  `f91db53a83d79f56927b04d38b1e886d2e4260e4528e7882ddd49adbda97ccb0`
- 外部审计可见提交：
  `280631a27ffadad7d47eff4d379d6203427e013e`
- 外部裁决：`REVISE`
- 本地复核时 protected-E2E 分支 HEAD：
  `b3222af0895e23eca83113977c1bcfad75258c9e`
- 复核日期：2026-07-20

原始回复已逐字归档。本文件只记录吸收结论，不替代原文。

## 裁决核心

外部审查批准的唯一候选是面向离线 TAD 的受保护端到端固定预算选择器：

1. 低分辨率完整时间序列经官方 ASFormer 产生动作二分类 logits 与 hidden。
2. transition descriptor 只使用动作状态变化，不恢复直接 boundary head、absolute
   hidden 或 raw RGB descriptor。
3. selector 显式拆为等容量的 adapter 与 score head。
4. 主臂中检测损失只更新 selector adapter/head；ASFormer 与 action head 在该路径
   detach。`rho` 消融只以固定小系数开放最后一个 ASFormer temporal block。
5. hard 与 soft 必须共享同一个 physical exact-K DAG。hard 用 Viterbi，soft 用
   同一合法路径分布的 Gibbs slot marginals。
6. detector 前向严格消费真实 hard gather；soft assignment 只提供 backward
   Jacobian。
7. GT、target、decode、proposal、NMS 和 evaluator 全部使用 dense/native physical
   coordinate，禁止 selected-axis GT remap。
8. P0 协议、P1 实现、P2 分损失梯度所有权和 P3 真实 hard-swap 方向门禁全部通过
   后，才允许运行 exact-uniform、transition-no-bridge、protected-E2E 和
   protected-E2E-rho 四臂 official-60。

## 对当前实现的独立复核

外部审查只读取到 `280631a`，没有看到后续实现提交。因此以下陈述已经过时：

- “尚无 protected-E2E 配置/实现”已不再成立。后续提交 `0477c55`、
  `c2226dc`、`e7aa881` 和 `b3222af` 已实现配置、桥、P1/P2/P3 工具。
- “真实 official detector 到 selector 的梯度仍为零”已不再成立。
  Slurm Job `1176948` 在真实 THUMOS batch、完整 VideoMAE-S adapter、
  projection/neck/ActionFormerHead、FP16 与单卡 DDP 下，通过了主臂和 rho 臂的
  detector-only 梯度所有权检查。
- 旧 proof-only temporal-mean smoke 已被真实 full-model exact gate 取代。

但外部审查指出的下列结构性阻断仍然成立：

1. 当前 `protected_structured_transport` 是围绕 hard positions 的局部时序斜率
   代理，不是同一 global physical exact-K DAG 的 Gibbs marginals。
2. 当前规范约束仍是 candidate hole `G=2`，而不是由真实 source frame/fps
   建立的 per-sample seconds cap，也没有共同编码 source/sink physical edges。
3. 当前正式 route 仍使用 selected-axis detector semantics 与 GT remap；这会把
   不同物理间隔压成相邻 rank，污染非均匀选择的效应。
4. 当前主配置使用 detector bridge 终值 `0.25` 且带 warmup/ramp；新裁决冻结为
   `1.0` 且无 ramp。
5. 当前 rho 为 `0.05`；新裁决冻结为 `0.01`。
6. 当前 planned official-60 为 seed 0、100 step/epoch、6000 successful
   updates；新裁决提出 seed 3407、99 step/epoch、5940 updates。
7. 当前 P3 只有 4 个 full windows、每窗 8 个局部候选，并读取旧训练 checkpoint；
   新裁决要求初始化态、48 个分层训练窗口、576 个 preregistered physical swaps、
   至少 512 个有效样本及分层 cluster bootstrap。
8. Job `1176948` 的 P1/P2 exact gates 通过，但 P3 在读取旧 manifest 时因要求一个
   实际不存在的 `training_profile` 字段而失败。该失败是证据解析契约错误，不是
   hard-soft 数值结果；不过旧 P3 即使修复，也达不到新裁决的 P3 规格。
9. 当前分支在 `b3222af` 之后仍有未提交的 gate hardening，不能作为不可变证据。
10. 当前没有任何该新架构的 official-60 mAP，四臂训练保持禁止。

## 接受的建议

本项目接受以下内容作为新的设计边界：

- 当前任务是离线 TAD，不是 Online TAD。
- 不恢复 `structured_zero_forward`、`soft_to_hard_resample` 或其他旧局部 bridge。
- 使用同一 physical exact-K DAG 的 hard Viterbi 与 soft forward-backward
  marginals。
- 复用已有 Allocation-Ceiling physical axis/validator、acquisition assignment
  tensor bridge 和 physical-grid ActionFormer，不重复造检测器。
- 主臂保护 ASFormer/action head，只让 detector loss 更新 selector adapter/head。
- 关闭 counterfactual teacher、utility distillation、soft max-gap legality、
  post-hoc repair、learnable coverage floor、policy homotopy 和 detector-gradient
  ramp。
- P2 必须按 detector/action/transition 三种单独损失审计参数所有权，不能使用总
  `cost` 冒充归因。
- P3 必须比较 soft score gradient 与同一 physical feasible family 中真实 hard
  finite differences。
- selected-axis remap 必须 fail closed；physical target/decode/NMS/evaluator 必须
  同轴。
- 四臂 terminal EMA official mAP 是唯一主裁决，机制与成本指标只能解释。

## 有条件接受或保留异议

### 训练更新数

不直接接受未经运行时证据支持的 `99×60=5940`。当前基础训练集合曾明确打印
`200 videos`，batch size 为 2，历史正式协议实际使用 100 step/epoch。新 P0 必须从
精确数据 manifest、sampler、drop-last 和真实 loader 长度生成并哈希冻结更新数。
若 exact loader 仍为 100，则 official-60 应为 6000 successful updates；只有实际
loader 证明 99 时才能使用 5940。

### P3 初始化态

接受 P3 作为 surrogate fidelity 的训练前门禁，但必须明确它测试的是“随机检测头/正式
预训练 backbone 初始化处的局部方向”，不是已训练 detector 的性能效用。若采用该门禁，
不能把通过结果写成 mAP 支持；若失败，也只能否定该冻结 surrogate/初始化合同，不能
外推否定所有智能时序选择。

### physical-grid

selected-axis 对全局非均匀选择存在确定性语义问题，因此 physical-grid 是必要修复。
但历史 PhysTime/稀疏 head 诊断显示物理 anchor 可能降低短动作 positive assignment。
所以 physical-grid 不能因“已有实现”直接视为正确，必须通过 target/decode roundtrip、
positive support、micro-overfit 和 same-selection parity gate。

### 覆盖概率与固定系数

`lambda=0.10`、`tau_score=0.70`、`tau_path=1.00`、`rho=0.01` 与 detector bridge
coefficient `1.0` 可作为本次单一冻结实现的预注册值，但它们是设计选择，不是已经由
证据证明的最优超参数。四臂开始后不得调参。

## 当前路线裁决

- `idea:duca-protected-e2e` 仍为 `designed`。
- `b3222af` 路线是 `implemented_nonconforming_diagnostic`，不是新裁决的 P1。
- Job `1176948` 只证明旧候选的真实梯度连通与部分所有权；它不是 P3 PASS，也不授权
  official-60。
- 之前准备的 local-slope bridge official-60 四臂不得提交。
- 下一步必须先冻结新的 P0 manifest，再实现 physical exact-K DAG、Gibbs slot
  marginals、physical-grid dense/native coordinate 和严格 P1/P2/P3。
- 任一门禁失败即 STOP；不得降低阈值、恢复 selected-axis、换回旧 surrogate 或扩展
  X3D/MUST/MobileNet/其他 detector。

## 论文口径

在新的四臂 terminal EMA mAP 产生前，只能说：

> 已设计并正在审计一种受保护的 detector-to-selector 梯度机制；现有真实 full-model
> gate 证明了旧候选的梯度连通性，但同一物理可行域上的 hard-soft 对齐、physical
> coordinate 正确性和最终 mAP 均未得到证明。

不得说“最终模型已完成”“端到端梯度已证明有效”“已优于均匀采样”或“论文主方法已
成立”。

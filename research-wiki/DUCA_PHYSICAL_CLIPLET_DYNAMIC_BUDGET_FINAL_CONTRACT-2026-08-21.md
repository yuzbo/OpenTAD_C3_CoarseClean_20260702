---
type: research_contract
title: "DUCA 物理连续片段与动态预算最终合同"
status: fixed_m24_negative_terminal
canonical: true
updated: 2026-08-22
decision: STOP_FIXED_CONTIG
candidate: DUCA_PHYSICAL_CLIPLET_CONTIG_S0-v001
base: OpenTAD official AdaTAD 01c58b9f2370e914150cf94d392208a4e211c053
---

# DUCA 物理连续片段与动态预算最终合同

## 1. 科学问题与证据边界

DUCA 研究低成本语义侦察器能否预测逐时刻动作性、动作起点和动作终点，再由完全
确定性的获取规则选择物理时间连续的 16 帧片段，并按视频窗口分配动态片段数，从而
真实减少 VideoMAE 的输入计算且保护高时间交并比定位。

本合同是第二轮且最终的 Pro 裁决的规范化版本。官方 AdaTAD `01c58b9` 派生的干净
候选已完成实现、独立静态审查、S0 训练与训练总体语义测量。固定 `M=24` 的 FZ_CONTIG
与 JT_CONTIG 已通过目标环境 PRE_RUN，正在进行完整 60-epoch 训练，但尚无本路线的终态
检测性能或成本结果。旧 65.xx、UVT、Fovea、density、prefix-budget 和 U/O/R
只作为历史诊断，不能支持本合同的主张。

## 2. 不得漂移的因果链

```text
低分辨率 dense scout
  -> action / start / end logits
  -> 确定性的物理连续片段选择
  -> fixed M=24 基础门；后续 dynamic M
  -> 只将实际选择的 16-frame cliplets 输入 VideoMAE
  -> 按 int64 物理时间重建到官方 detector 网格
  -> 在 filtering、top-k、IoU、NMS、voting 前进入物理坐标
  -> 不变的 AdaTAD / ActionFormer detector、loss、NMS、evaluator
```

- Scout 不直接预测 frame index、片段数 M、proposal、类别或 NMS score。
- 不训练第四个 uncertainty head；不确定性只由 action/start/end 的 Bernoulli 熵导出。
- Query-Bridge 只能在 S0 通过后改善三类语义；cycle 只能在 SQ 通过后作为训练期
  detached 条件臂。语义蒸馏永久删除，只有 S0、SQ、SQC。
- fixed `M=24,K=384` 仅是控制、归因截面和故障回退；dynamic outer M 才是候选标题机制。
- `executed_K` 必须来自真正送入 temporal patch embedding 的 RGB 帧数；dense/Kmax
  padding、先跑 dense 再 mask、缓存 heavy feature 或只改 metadata 都不合格。

## 3. S0 标签、损失与定点语义质量

S0 输出 `action_logits[B,Ts]`、`start_logits[B,Ts]`、`end_logits[B,Ts]`，以及按
int64 表示、左闭右开的 frame/time support。bins 必须有序、互不重叠并完整覆盖有效窗口。
训练标签只来自 canonical training annotation：与动作区间有正长度交集为 action 正例；
每个 start/end 依左闭右开规则唯一落入一个 bin，窗口右边界终点归最后一个 bin。

三个通道使用在完整训练标签账本上冻结正例率的 balanced BCE，S0 损失为三者均值。
禁止 frame-index、K、proposal、teacher、distillation 或 detector-output 监督。

对 `q in {a,s,e}`：

```text
p = sigmoid(logit)
u = BernoulliEntropy(p) / log(2)
g = max(0, 2p - 1)
c = g * (1-u)
r = g * u
Q = 2^20
q_Q(x) = floor(Q*x + 0.5)
```

所有 `[0,1]` 量先以 float64 计算，再 half-up 量化为非负 int64。候选排序、
边际增益、阈值与预算判定只使用 int64；不得使用随机容差。

## 4. CONTIG 候选、证书与 nested 获取

有效帧数 `T>=16`。候选 `C_c={c,...,c+15}`，且源 frame index 严格连续、timestamp
严格递增。所选候选两两不重叠。端点上下文 `chi(i,C)` 使用候选内可获得的左右最多
4 帧上下文；窗口边缘只要求真实存在的上下文。

证书为七个归一化分量：start/end 覆盖 `B_s/B_e`、start/end 上下文 `X_s/X_e`、
按真实 frame time support 加权的 action coverage `A`、正边界证据上的 ambiguity risk
coverage `R`、以及最大物理时间空洞对应的 coverage `P`。任一分母为零时，该分量恒定
定义为 1，不参与边际选择。完整公式以最终 Pro 原文第 166--241 行为规范来源。

`J_Q(S)` 是七个分量 half-up 量化后的 int64 和。每一步只在加入后仍能用剩余 free
runs 填满目标序列的候选中，选择 `J_Q` 边际增益最大者；完全同分取更小源帧 start。
必须直接验证不重叠、源帧连续、nested prefix 和完成容量。没有随机 tie、GT、teacher、
detector prediction 或 direct-index policy。

固定阶段：`M_cap=floor(T/16)`，`M_fixed=min(24,M_cap)`，选择同一 nested 序列的
前 `M_fixed` 个 cliplet。

## 5. 动态预算的唯一整数定义

候选 support 只能是 `{16,20,24,28,32}`；每个值都必须在 training-side CAL 证明：
真实不同 heavy execution、无 Kmax padding、N16R4 可达、完整成本可测且
`executed_K=16*M`。任一值失败则整个 dynamic stage 阻断，不能事后缩小 support。
短窗口把每个候选截到 `M_cap` 后去重排序，并生成到最大值的同一 nested 序列。

对某个前缀 `S_m`，将证书七个已量化分量记为 `q_Q(V_j(S_m))`，定义：

```text
D_Q(m,x) = max_j (Q - q_Q(V_j(S_m)))          # int64, [0,Q]
```

阈值 `theta_Q` 只取 CAL 中实际出现的 `D_Q` 值及 `{0,Q}`。预算为最小满足
`D_Q(m,x) <= theta_Q` 的候选 `m`，没有满足项时取最大候选。CAL 使用逐行实测完整链路
成本选择与 fixed M24 平均成本最近的阈值；完全同分依次取更小阈值、更小平均 executed M、
更小候选序号。只有成本差落在预注册的 N16R4 重复性带内才能封存。

## 6. GAPPACK 的失败关闭控制

GAPPACK 不是主方法。它只在实例化后的官方 VideoMAE 模块逐项证明 temporal
kernel=stride=`tau`、无跨块 padding、`16 % tau == 0`、`L=16/tau>=2`、无 temporal
pool/merge/drop、每个输出 slot 与一个输入 atom 一一对应，且不改变官方 backbone/detector
语义时存在。否则状态必须是 `GAPPACK_DROPPED_FAIL_CLOSED`，CONTIG 路线继续。

若准入，对 M 个 cliplet、L 个 atoms 使用非恒等双射
`pi(m,r)=((m+r) mod M,r)`，packed block `(b,r)` 取原 atom `((b-r) mod M,r)`；heavy
输出在物理重建前执行精确逆排列。准入只比较 frame/atom/permutation/inverse 的直接数组
相等、双射和 round-trip；禁止新增 hash、checksum 或近似时间戳恢复。

官方 AdaTAD `01c58b9` 当前 VideoMAE-S 的 `tau=2,L=8`，但配置后处理包含 temporal
Reduce(mean)，因此现有模块不能证明 atom 输出一一对应。除非 Builder 在不改变官方语义的
前提下证明真实实例满足全部条件，本周期按 `GAPPACK_DROPPED_FAIL_CLOSED` 执行。

## 7. 不重复训练的分阶段实验

Dense、equidistant-uniform 和 random 的既有训练不得重跑。官方 dense 与 exact-uniform
只读绑定 immutable compatible receipt；没有兼容 receipt 时，可以做基础内部机制实验，
但论文非劣性、SQ、SQC 与 dynamic stage 全部阻断。

1. 训练一个 full canonical THUMOS training population 的 `S0_TERMINAL`，只用语义损失。
2. 第一 seed `3203700` 运行完整 60 epoch：`FZ_CONTIG`、条件成立时的
   `FZ_GAPPACK_ATOM`、`JT_CONTIG`。这一步只估计时间呈现 bundle 和 frozen-vs-joint
   scout drift，不估计 selector、Query 或 dynamic budget。
3. 只有 S0 相对 compatible uniform 通过冻结统计门，才增加 `SQ_CONTIG_M24`；SQ 通过后
   才增加 `SQC_CONTIG_M24`。不得增加 SQD。
4. 只有前述门通过，才复用已冻结的 fixed receipt，并新增 `SEMANTIC_DYNAMIC`、
   `K_SHUFFLE`、`ACTIONNESS_ONLY_DYNAMIC`。K-shuffle 保持 cost strata 内 M histogram；
   actionness-only 保持同一 histogram，但关闭 start/end 证据。
5. 扩展 seeds 固定为 `1677630095`、`1453526567`。不得以 subset 或本地 CPU 作为效能证据。

## 8. 公平性、统计、成本和恢复

- 数据为 canonical THUMOS14，train=`training`、official eval=`validation`；FIT/CAL/HOLD
  在训练 population 内视频不相交，official evaluation 在 PRE_RUN 前不可访问。
- 各臂共享 detector、assignment、loss、optimizer updates、LR、augmentation、NMS、
  evaluator、seed 与 final/final-EMA 选择。`final-EMA` 是唯一 primary，`final` 是预注册
  secondary diagnostic；不得按中间验证挑最佳。
- 统计使用 10,000 次 paired video cluster bootstrap；三 seed 后使用层次 bootstrap。
  practical-equivalence margin 只能由至少三个兼容 immutable uniform receipts 的训练侧
  CAL 差异确定；否则为 `UNDEFINED`，不得作论文非劣性结论，也不得进入 SQ/dynamic。
- 完整成本分解 scout、decode/materialization、CPU transform、H2D、patch embedding、
  VideoMAE、重建、detector、物理坐标转换、NMS/voting、serialization；CAL 上对最终 N16R4
  重复同一完整 workload 10 次冻结成本重复性带。
- 每 5 epoch 保存完整可恢复 `.pth`，保留 latest-3、20/40/60、final、final-EMA；恢复
  detector/S0/Query、optimizer、scheduler、AMP scaler、EMA、epoch/update、Python/NumPy/
  Torch/CUDA RNG、sampler/DataLoader、冻结选择数组、dynamic support/threshold/strata 与
  cost-ledger cursor。

## 9. 实现、审查与 PRE_RUN 交付

唯一候选从只读官方 AdaTAD `01c58b9f2370e914150cf94d392208a4e211c053` 派生到新的
DUCA-owned clean clone/worktree。禁止继承 SparseHead、`pc_ot_mras`、U/O/R、density、
prefix 或修改后的 detector。最小交付包括：S0、标签账本、CONTIG acquisition、真实 sparse
VideoMAE materialization、physical reconstruction/state machine、fixed FZ/JT configs、
条件 GAPPACK precondition、checkpoint/resume、full-stack ledger 和拒绝全部非空
`--cfg-options` 的 launcher。

独立 Critic 对同一 clean revision 只返回 `DUCA_FINAL_CONTRACT_STATIC_PASS` 或
`DUCA_FINAL_CONTRACT_BLOCKED`；GAPPACK 单独失败时记录 fail-closed，但不得扩大为 CONTIG
失败。Evaluator 只有在代码、config、canonical manifests、shared official receipt、资源元组、
resume 与 metric embargo 全部绑定后才可给 `PRE_RUN_READY`。通过后立即提交第一完整 seed；
否则报告一个客观 blocker，不得启动训练或读取 official metric。

## 10. 主张与反证

只有 selector、semantic、dynamic、physical-time、真实成本和三 seed 统计门全部通过，才可
主张：动作/边界语义形成的显式 coverage certificate 能驱动物理连续 cliplet 和 matched-cost
动态预算，在不改变官方 detector 与后处理的条件下减少重型 VideoMAE 计算并保护定位。

以下任一现象会删除相应或全部主张：selector 未超过 compatible uniform；动态预算未超过
fixed、K-shuffle 或 actionness-only；M 只由长度/总 actionness/decode cost 解释；存在 hidden
dense/Kmax compute；物理时间晚于 filtering/NMS；必须改变 detector/head/loss；Query/cycle
只改善辅助损失；完整链路成本没有下降。负结果必须保存，禁止静默改 support、阈值、loss、
seed、split 或 checkpoint 规则。

## 11. 原始裁决与当前下一动作

最终 Pro 原文：
`.cvpr-pro-lab/pro-reviews/runs/duca-physical-cliplet-contract-final-pro-20260820T180224Z-7fac8d4a/raw-response.md`；
Project conversation `6a874179-a388-83ea-8ebd-2ec53d2a2624`，裁决 `CONTINUE`。

## 12. 第一种子终态结果（2026-08-22）

固定 `M=24,K=384` 的完整 seed `3203700` 已结束。FZ_CONTIG 的 Avg-mAP 为 `49.89`，
tIoU `0.3/0.4/0.5/0.6/0.7` 为 `65.04/59.98/52.51/42.22/29.68`；JT_CONTIG 为
`47.24` 和 `63.51/57.89/49.69/38.57/26.52`。训练与官方 validation 均正常退出，执行账本确认
`executed_k=patch_embed_input_k=384`，因此失败不是 padding 假稀疏或未运行。

该结果满足本合同的停止条件：固定连续片段基础门不能保护定位，联合训练也没有恢复性能。
Query/cycle 与 dynamic-M 不再准入；不重复已有 dense/uniform/random。保留的结论仅是“真实稀疏已实现，
但当前连续片段获取造成严重定位损失”，不得形成正向论文主张。

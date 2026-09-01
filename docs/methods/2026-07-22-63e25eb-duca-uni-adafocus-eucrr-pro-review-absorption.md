# DUCA V8 × Uni-AdaFocus / EU-CRR Pro 审查吸收

## 记录信息

- 审查对象：`yuzbo/OpenTAD_C3_CoarseClean_20260702`
- 精确提交：`63e25eb17e523d369f73434ed4d9b6446608861a`
- 上游参照：`LeapLabTHU/Uni-AdaFocus@8846488310fdd4a18412608006030643e794c36e`
- 原文归档：
  `docs/methods/reviews/2026-07-22-63e25eb-duca-uni-adafocus-eucrr-pro-review-raw.txt`
- 原文大小：`65,069` bytes，`1,426` lines
- 原文与归档 SHA-256：
  `0678A31C17D3FCD983726CE9056E463CF09A0325DAF69C7C41947EEB57602DAA`
- 原审查裁决：`HOLD`
- 项目独立裁决：
  `SUBSTANTIAL_ACCEPT_DIAGNOSIS / CONDITIONAL_ACCEPT_DIAGNOSTIC / REJECT_AS_MAINLINE_REPLACEMENT`

## 一句话结论

我认可该回复对当前 V8 代码、Uni-AdaFocus 的可迁移边界、梯度语义和成本口径的主要诊断；也认可把零门控 coarse residual 作为一次严格单变量诊断。但我不认可把 EU-CRR 直接提升为最终模型或唯一下一步，也不认可 fusion/physical-grid 失败后直接否定整个 DUCA。当前冻结的主线仍是 Oracle-calibrated bilateral boundary burst；EU-CRR 只能登记为正交、条件性的表示复用实验。

## 已独立核验的代码事实

1. 本地干净审计副本 HEAD 精确为
   `63e25eb17e523d369f73434ed4d9b6446608861a`。
2. 当前 `duca_online_frame_selector.py` 从 online coarse probe 取得
   `[B,T,96]` hidden，并把它交给 acquisition/scorer；selector 返回给 detector 的
   主输出只有 hard selected RGB、mask、meta 和 `selector_outputs`。当前没有
   `selected_coarse_hidden` detector contract。
3. `ActionFormer.forward_train/forward_test` 在 selector 后直接执行
   `x = self.backbone(inputs)`，随后进入 token compressor/projection/neck/head；当前没有
   coarse/VideoMAE feature fusion。
4. VideoMAE wrapper 的 post-processing 输出是 `[B,384,K]`，不是原文示意图中的
   `[B,K,384]`。原文给出的 `LayerNorm + Conv1d` 模块可适配真实 `[B,C,K]`
   形状，但必须以真实 wrapper 输出做 full-model gate，不能只靠示意张量。
5. VideoMAE 输出 K 轴来自 tubelet token、chunk 重排和插值；它与 hard selected slot
   数量相同，但“同一 ordinal 就是同一原始帧语义”仍需显式时序对齐测试，shape
   相等不能证明语义相等。
6. hard Viterbi/LongTensor index 被 detach。`protected_structured_transport` 是
   hard-forward、surrogate-backward 的检测损失派生桥，不是穿过 hard index 的直接梯度。
   配置中的 `detector_gradient_is_direct=True` 必须更名或严格限定含义。
7. 当前数据通路在 selector 前已 decode、transform、materialize 并 H2D 传输 dense
   768 帧。因此当前可证明的成本主张只应是“减少 heavy VideoMAE 处理帧数”，完整
   端到端降本仍待测量。

## 我认可的建议

### 1. 对当前低于均匀采样原因的排序

认可优先审查以下问题，而不是先假设“缺少 coarse fusion”是根因：

1. hard deployment 与训练 surrogate 不同构；
2. irregular physical time 被压成 selected-rank 轴；
3. coarse 二分类/transition 证据不等价于类别化 TAD cls/reg utility；
4. K=384/G=2 可行族的自由度与收益上限；
5. learned positions 是否真实增加 detector utility；
6. 最后才是 coarse hidden 未复用。

### 2. 对 Uni-AdaFocus 的迁移边界

认可：Uni-AdaFocus 证明的是视频识别中的 cheap-global / expensive-local 协作，不是
TAD 边界定位；hard temporal index 本身不可微；policy 依赖独立 surrogate；global
feature reuse 有价值的结论不能直接外推为 THUMOS14 高 tIoU 增益。

### 3. EU-CRR 作为干净诊断

认可以下实验原则：

- 用同一 P0 checkpoint、同一 exact-uniform K=384 positions、同一 6000 updates 和
  同一 seed 对比 U0/U1；
- U0 仍运行 coarse probe，仅丢弃 hidden，从而保持计算与状态匹配；
- fusion 位于 VideoMAE 后、official projection 前；
- coarse hidden detach，coarse probe frozen/eval；
- channel-wise residual gate 全零初始化，gate=0 时逐值退化为 U0；
- 不同时修改 selector、DP、detector、loss 或训练长度；
- 检查 optimizer coverage、zero decay gate、EMA、AMP/DDP、泄漏和完整成本。

这能回答一个窄问题：**在固定均匀选帧下，cheap coarse representation 是否对 TAD
检测表征有额外价值。**

### 4. 论文与成本口径

完全认可：一旦 detector 消费 post-VideoMAE coarse residual，方法就不再是严格
pre-backbone-only plugin，而应称为 acquisition-and-fusion adapter 或
coarse-assisted sparse TAD；单一 AdaTAD backend 也不能证明 plug-and-play。

## 我不完全认可、必须修订的部分

### 1. EU-CRR 不能替代当前最终主线

U0/U1 都使用 exact-uniform positions。它们不会检验、更不会修复当前真正的选帧目标：
状态转变中心是否准确、是否在真实起止边界两侧形成 Oracle 式 3--5 帧微簇、是否对
多端点公平分配、是否在配额饱和后把剩余预算用于全局上下文。因此 EU-CRR 不能取代
G23 和 `exp:duca-oracle-calibrated-boundary-burst`。

### 2. “唯一下一步”顺序不成立

当前更直接且更便宜的第一步仍是 R0 train-split Oracle K/G reachability：先证明允许
边界微簇的可行族对 official detector 有 mAP headroom。只有这样才知道是目标/可行域
有问题，还是表示复用有问题。两臂完整 6000-update fusion 实验不应抢在这个无训练
上限诊断之前。

### 3. Gate S 的因果对比不完整

四臂应同时报告四个预注册 contrast：

- `U1-U0`：均匀位置下的 fusion 效果；
- `L1-L0`：学习位置下的 fusion 效果；
- `L0-U0`：无 fusion 时的 selection 效果；
- `L1-U1`：有 fusion 时的 selection 效果。

原文只用 `L1-U1` 裁决 Gate S，只能回答“fusion 存在时 learned selection 是否优于
uniform”，不能隔离 learned 场景中的 fusion 贡献；既然 L0 已运行，就必须使用
`L1-L0`，并检查 selection × fusion interaction。

### 4. 失败范围过宽

- U1 失败只能 KILL post-VideoMAE coarse residual reuse。
- physical-grid 对照失败只能否定该坐标合同在冻结设置下的收益。
- 二者都不能自动否定 Oracle-calibrated boundary-burst acquisition，因为后者是不同的
  选择目标与不同的可证伪问题。

### 5. `+0.50` 不是当前论文主 GO 线

`+0.50` 可作为 fusion 诊断的严格效果阈值，但其来源是恢复 V5 deficit 的三分之二，
不是统计或任务的自然阈值。当前主方法预注册线仍是 terminal-EMA Avg-mAP
`>=65.00`、相对 matched U `>=+0.20`、@0.6/@0.7 下降均不超过 `0.20`，且完整
端到端实测成本低于 dense。两套 gate 不得混写。

### 6. selected coarse hidden 不是完整 dense context

EU-CRR 只 gather K 个 hidden。ASFormer hidden 具有全窗口感受野，因此可能携带未选帧
上下文，但它并不等价于 Uni-AdaFocus 保留整条 global stream。论文必须准确称
“selected coarse states with full-window receptive field”，不能声称完整 dense coarse
sequence 已被 detector 消费。

## 修订后的路线位置

### 主线不变

```text
dense low-resolution coarse/official-ASFormer evidence
  -> indirect transition centers
  -> Oracle-calibrated bilateral capped boundary bursts
  -> overlap-aware saturation + residual global utility
  -> existing exact-K/max-hole DP
  -> hard original-time RGB observations
  -> official-derived AdaTAD/ActionFormer
```

状态保持 `designed_not_implemented`；没有 V9，没有修改正在运行的 V8 Job
`1178989`。

### EU-CRR 的正确定位

- 状态：`discussed_conditional_diagnostic_not_authorized`。
- 作用：诊断 frozen coarse representation 对 sparse detector 的附加价值。
- 不是：边界微簇 selector、最终 pre-backbone plugin、V9 或论文主方法。
- 运行前提：V8 终局已封存；R0 Oracle reachability 已完成；U0/U1 能绑定同一 P0、
  同一 commit、同一 terminal-EMA evaluator 和完整成本账本。
- 若 U1 GO：再决定是否把最终产物从 strict pre-backbone acquisition 扩展为
  acquisition-and-fusion adapter；不能静默改变论文主张。
- 若 U1 KILL：只永久停止 EU-CRR，不影响 G23/R0--R5 主线。

## 最终裁决

这份回复是高质量审查，最有价值的部分是：它没有把 Uni-AdaFocus 当作可直接照搬的
答案，准确指出了当前 coarse hidden 的真实流向、hard-index 梯度边界、selected-rank
时间语义和完整成本缺口，并给出一个可严格归因的 U0/U1 诊断。

但项目不能因此偏离已经冻结的设计初心：我们首先要解决的是**如何把粗动作状态变化
转成可部署的 Oracle 式边界聚集选帧**，而不是先让 detector 在均匀位置上再吃一份
coarse feature。故本轮结论是“主要诊断认可，实验条件性认可，主线替代不认可”。


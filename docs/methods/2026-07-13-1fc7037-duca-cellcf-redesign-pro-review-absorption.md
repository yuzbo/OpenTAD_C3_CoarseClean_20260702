# DUCA `1fc7037` CellCF Pro 审查吸收记录

## 来源与证据边界

- 审查固定仓库：`yuzbo/OpenTAD_C3_CoarseClean_20260702`。
- 审查固定提交：`1fc7037358e1141f7555ad87d1edd9128ce2e6a5`。
- 原附件：`eb295612-1df9-44b8-9c98-6fd813d0552c/pasted-text.txt`。
- 原文与本地归档 SHA-256：
  `DDBC15BC20BFDD503FAA2DA4832093325EB2D8997E4A685638DFF46F90CC780D`。
- 原文归档：
  `docs/methods/reviews/2026-07-13-1fc7037-duca-cellcf-redesign-pro-review-raw.txt`。
- 审查能够读取 GitHub 代码和已提交的选择质量诊断，但看不到多数远端原始日志、
  外部 ASFormer 源仓库和未提交本地状态。远端数字仍须按项目 provenance 单独核验。

## 总裁决

接受审查对当前路线的 `REDESIGN` 裁决：

1. 当前 `transition_only + global_structured_topk + G=15 +
   structured_zero_forward` 不应继续作为论文主方法直接扩展训练。
2. 这不是对“低成本状态变化可帮助离线 TAD 去冗余”科学问题的否定。
3. DUCA 只保留一次有界申诉：coverage-preserving local cell deformation，配合训练期
   hard counterfactual detector-utility distillation，即审查命名的 DUCA-CellCF。
4. CellCF 当前状态仅为 `discussed/design proposed`，不是 `implemented`、`tested` 或
   `empirically_supported`。

## 已由本地代码复核确认的关键问题

### P0

- `acquisition.py` 的 legacy/direct stable route 仍使用 midpoint targets；
  `0ea4e15` 只修复了 transition helper，没有完成 repo-wide exact-uniform 统一。
- `balanced_binary_actionness_loss()` 按 batch 动态计算 `pos_weight`。其 sigmoid 输出是
  cost-sensitive score，不能未经校准就解释为后验 `p_action`，尤其不能据此赋予 entropy/
  uncertainty 严格概率语义。
- `local_boundary_coverage_loss()` 对未归一化局部 occupancy sum 使用 `-log`；当局部和
  大于 1 时 loss 可为负并继续奖励聚集，属于确定性目标错误。
- 当前导出器把 `transition_score = abs_delta_p_action + uncertainty_peak` 写入
  `raw_transition_scores`。既有诊断只证明 learned scorer 弱于这个复合手工代理，不能
  证明其弱于纯 `abs(delta p_action)`。
- `structured_zero_forward` 的 hard-forward/soft-RGB-backward 只证明梯度连通，尚未用
  feasible hard one-swap 验证方向一致性。

### P1/P2

- 在 `T=768,K=384` 下，`G=15` 远松于 exact uniform 自然产生的 1--2 帧间隔；实测
  learned decoder 与 unconstrained utility top-k 平均重合 99.80%，行为上接近全局 top-k。
- transition scorer 主要学习 GT endpoint Gaussian representation proxy，不是 detector
  utility。
- 当前 coarse 模块是自定义两层低分辨率 spatial stem 加 official ASFormer temporal
  module，并非 official pretrained raw-video ASFormer；checkpoint/protocol 必须如实披露。
- standalone 和 joint action target 分别使用 `frame+0.5` inclusive 与整数位置
  start-inclusive/end-exclusive 语义，必须统一到一个 helper。
- detector 运行于 selected axis；真实 selected positions 参与 remap，但 detector 卷积并
  不显式理解不规则物理间隔。NMS 前 inverse map 合同存在，但内部几何风险仍未消除。
- detector gradient 当前被刻意限制到 scorer；这不是遗漏，但也不能称整个 coarse probe
  被下游检测任务协同优化。
- `B x K x T` soft RGB bridge 有明显训练成本，必须计入或删除。
- structured DP 递推本身暂未发现确定性错误；仍需小规模穷举与梯度/可行集测试。
- optimizer 参数覆盖、leaf loss 去重和主推理路径 GT/teacher 泄漏目前没有发现问题。

## 对 DUCA-CellCF 的吸收

### 核心可行集

- 由 `round(linspace(0,n-1,k))` 生成 exact-uniform anchors。
- 用相邻 anchor 中点把有效时间轴划分为互不重叠且保序的 cells。
- 每个 cell 必须且只能选择一帧。`T=768,K=384` 时 cell 宽度至多 3，最坏相邻未选
  hole 为 3；exact K、顺序和 coverage 由构造保证。
- scorer 采用 `gamma * abs(delta p_action) + zero-init residual`，推理只读取可部署的
  state-change descriptors，不读取 GT、teacher、dense detector predictions 或高分辨率
  RGB shortcut。

### 训练职责

- `L_state` 更新 spatial stem、ASFormer 和 action head。
- `L_transition` 更新 coarse path 与 scorer，但只能作为 transition representation prior。
- 训练期周期性构造少量 cell 内 hard alternatives，以 EMA detector 的真实
  `cls+reg` loss 形成 stop-gradient preference，`L_cf` 只更新 scorer。
- `L_detector` 只更新真实 AdaTAD-derived detector；不再通过 soft RGB surrogate 直接
  反传到 selector。
- 单次训练运行先以 exact-uniform hard policy warm up，再只在 cell 内连续过渡；具体
  10%/25% schedule 是待验证超参，不是既定科学事实。

## 不完全认可与保留意见

1. **固定 detector grid 不是已证明修复。** 把实际采集的 `s_j` 帧标记为固定 anchor
   `u_j` 会引入受控但真实的观测时间错位。它可能稳定 ActionFormer 几何，也可能伤害
   边界，必须做 same-selected-frames geometry gate；失败时应转 physical-time-aware
   backend，而不是强行采用固定 grid。
2. **cell-wise CF utility 不是全局集合效用。** 检测 loss 对多个 cell 的联合替换可能
   非加性；EMA teacher 也随训练变化。必须审计 hard alternative rank 稳定性、交互项和
   held-out sign/Spearman，不能仅凭 CF loss 下降宣称学到 detector utility。
3. **一格一帧主动限制了 Oracle 上限。** 该约束适合保护 uniform，但无法复制 GT Oracle
   跨 cell 集中多帧的特权策略。因此论文主张必须收缩为“学习 uniform 的局部残差”，
   不能声称普适最优采样或逼近完整 Oracle。
4. **概率校准有条件重要。** 若使用 entropy/uncertainty，固定 prior、unweighted BCE 或
   train-only calibration 是必要的；若最终只使用 logits ranking，则无需把所有 score
   强行解释为后验概率。
5. 审查给出的 AUROC、Spearman、mAP 和成本阈值适合作为预注册 stop rules，但不是
   已验证自然常数。正式报告必须给效应量和 video-cluster uncertainty。
6. “official AdaTAD”只能指核心 backend/head 配置和未激活分支语义接近官方；wrapper、
   selected-frame 输入、target remap 和 metadata 合同均是项目修改。

## 决定性顺序与停止条件

1. 先修四个确定性问题：统一 exact uniform、校准/重定义 coarse score、修 coverage
   loss 与 diagnostic key、统一 action target helper。
2. 在 full train 前完成：pure delta/compound/learned 分离诊断、hard one-swap alignment、
   cell coverage、same-selected-frames geometry 和 coarse standalone/joint matched gate。
3. 仅当机制门成立，运行 same-commit exact-uniform 与 CellCF fixed-384 一 seed pilot。
4. pilot 没有稳定正增益时停止另外 seeds；不得解锁 MUST、X3D/SlowFast、多 detector 或
   动态预算。
5. 只有 fixed-384 多 seed 优于 exact uniform、保护高 tIoU，且 full-stack p50/p95/
   energy 确有净节省，DUCA 才能升级为 `empirically_supported`。
6. 若 CellCF 仍不能超过 matched exact uniform，DUCA 退出主方法；可保留为负结果、
   coverage-preserving baseline 或 irregular sampling failure analysis。

## 最终判断

审查对当前 global selector 的否定和对 coverage-first、hard-counterfactual utility 的方向
判断基本可信；但 CellCF 仍包含固定-grid时间错位和局部 utility 非加性两项核心风险。
因此项目接受它作为 DUCA 的一次有界最终申诉，而不接受它已经是“完美最终模型”。


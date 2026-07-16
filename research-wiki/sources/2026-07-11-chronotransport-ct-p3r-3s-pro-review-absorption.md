---
type: source_record
title: "ChronoTransport CT-P3R-3S Pro review absorption"
source_sha256: "E7971A22044B384092B833A1137F8EC0B543B504D271078CBCB4198F96D35CAF"
verdict: revise_spec_before_code
updated: 2026-07-11
---

# ChronoTransport CT-P3R-3S Pro Review Absorption

## 1. Source and provenance

用户三次提供了内容相同的 Pro review 文本。三份附件均为 64,725 bytes、1,069 行，
SHA-256 均为：

E7971A22044B384092B833A1137F8EC0B543B504D271078CBCB4198F96D35CAF

最新附件路径：

C:\Users\skywalker\.codex\attachments\63d3e02e-06fe-4a40-bdb0-b8ff7f61de7d\pasted-text.txt

前两份重复附件 ID：

- 9fb7a806-4525-435c-a7f2-459d42ea7c07
- 2ca84118-d719-4f37-ad52-b87442c617f9

附件文本已完整、逐行归档到
[raw review](2026-07-11-chronotransport-ct-p3r-3s-pro-review-raw.md)。本页只负责结构化
吸收，不替代原文。

## 2. Executive verdict

Pro 总裁决为 REVISE_SPEC_BEFORE_CODE：

- b74101d 不能原样执行；
- 当前不必永久终止整个 ChronoTransport 假设族；
- 唯一可接受路线是 Route B：先做最小、预实验的统计/定义修订，再实施；
- 新协议必须命名为 CT-P3R-3S-r1，具有新 spec SHA；
- 在 r1 经用户复核并冻结前，不写争议实现、不跑 Gate 1、不做 GPU profiling、不训练
  新 seed，更不进入 Stage C/P5。

这不是新的实验结果，也不是 ChronoTransport 成功证据。它把路线从
bounded rescue designed 改为 spec revision required before code。

## 3. Evidence boundary

Reviewer 环境没有挂载本地 E:\... 仓库，也无法访问本地 commits b74101d、
fbf8f43、92029ea 或当前 ChronoTransport 源码。因此：

- 对数学/统计合同的审查可吸收；
- 对官方 OpenTAD/AdaTAD/VideoMAE 上游语义的固定版本核验可作为 reviewer-reported
  primary-source audit；
- 对本地 risk.py、cache、scheduler、profiler、optimizer、Stage C runner 和 tests 的
  P0/P1 判断只能登记为待本地核验风险，不能升级为 repository fact；
- LOCAL_VISIBILITY_BLOCKED 是 reviewer 环境限制，不是本地仓库缺失的证据。

Reviewer 报告的上游固定版本：

- OpenTAD：1aa8ca4ac5e846b1e8ff69298dd6607121a01589
- AdaTAD：25e06c720e450298ca5267fda6927f3591dcdfef
- VideoMAE：14ef8d856287c94ef1f985fe30f958eb4ec2c55d

这些 SHA 与 local-vs-upstream 差异需要本地代理重新核验后，才能成为当前仓库代码事实。

## 4. Absorbed protocol corrections

以下修订被吸收为 r1 的必审内容；它们不改变 risk head、seeds、candidate library、
训练预算、quantile、epsilon 或 Gate 1/2/4 数值门槛。

### 4.1 Gate 1 only proves oracle headroom

冻结 candidate library 的逐窗口 minimum 天然优于子集合 minimum。Gate 1 的 10% 和 CI
只能证明 equal-cost oracle headroom，不能单独证明 deploy-visible input dependence。
输入依赖与可预测性必须由 Gate 3 的窗口内 candidate-vector ranking 和实际选择来证明。

### 4.2 Cost envelope and candidate feasibility

B* = full-stack p50(periodic4_transport) 可以保留为预注册外部部署预算，但每个 HOLD
candidate 必须用自身完整端到端实测 total latency 判断是否 cost <= B*。不得使用
periodic4 action-count cost、线性 group cost或阶段 p50 代替。

### 4.3 Evaluation-best is diagnostic only

evaluation-best global static 只能是 label-using diagnostic upper comparator。它不得决定
deploy static policy、candidate library、threshold、scheduler、后续拟合或 pass/fail 的
可部署 comparator。主 comparator 必须在 fit/calibration 侧冻结。

### 4.4 Simultaneous conformal claim is marginal

每 calibration window 先对 16 candidates 取最大 residual，再用 30 个 window scores 做
finite-sample 0.9 quantile，是有效的 window-level simultaneous marginal guarantee。
它不能直接推出 scheduler selected non-dense 条件下的 coverage。

因此 actual-selected coverage 是评估量，不是由 simultaneous guarantee 自动推出的
条件保证。

### 4.5 Coverage becomes one-sided

原 [0.85, 0.95] 双边 hard gate 在少量 non-dense selections 下会产生显著纯采样假失败。
r1 应保留下界 coverage >= 0.85；coverage > 0.95 标记为 OVERCOVERED，但不直接 FAIL。
保守性由 pinball loss、non-dense selection rate、upper-bound sharpness 和 unique
window support 共同约束。

必须同时报告：

- seed-window selection count；
- unique selected window count；
- all-window coverage；
- selected non-dense coverage；
- cluster-aware uncertainty。

### 4.6 Ranking unit is a window candidate vector

禁止把 seed × window × candidate rows pooled 成一个 Spearman。r1 主统计为：

1. 每个 seed/window 内对完整 candidate vector 做 Spearman；
2. 每 seed 聚合 window correlations；
3. pooled 统计对 seed/window 等权；
4. bootstrap 以 unique window ID 为外层 cluster，candidate vector 整体移动；
5. 少于 3 个 distinct ranks 时 fail closed，不填 0、不静默删除。

### 4.7 Successful updates, not loop iterations

140 steps 必须定义为 140 次 successful optimizer updates。正式 artifact 必须记录：

- attempted iterations；
- successful optimizer updates；
- AMP skipped updates；
- per-schedule exposure；
- LR/EMA/scheduler update counts。

不得通过增加训练轮数救活路线。若当前 runner 以 loop iteration 停止，必须在 r1 实施前
修正或明确 formal run invalid。

### 4.8 Counterfactual replay must be order invariant

同一 window 的 dense/candidate regret 必须保持 augmentation、RNG、weights 与 mutable
loss state 一致。特别需要本地核验 ActionFormer loss_normalizer 是否在 replay 间更新。
必须 snapshot/restore mutable state，或使用冻结 evaluation semantics，并加入 candidate
order permutation regression test。

### 4.9 Dense fallback does not satisfy budget

dense upper risk 可以固定为 0，但：

- dense 必须从 non-dense coverage 分母排除；
- dense fallback 必须计入真实 selection rate 与 full-stack cost；
- dense 超过 B* 时记录 safety_override_budget_violation；
- missing cost/hash/calibration/checkpoint 不得被 dense fallback 掩盖成 cost success。

### 4.10 Full-stack percentiles come from total samples

禁止把 decode、patch、heavy、adapter、head、scheduler 等阶段的 p50/p95 相加。必须对
完整 forward 记录每次 total_ms，直接从 total samples 计算 p50/p95；阶段 samples 只做
诊断。lookup 必须绑定 exact schedule shape、hardware/software fingerprint 与 provenance。

### 4.11 Stage C exposure and secondary metrics

matched dense control 必须对齐 successful updates、window order、augmentation RNG、
LR trace、AMP skip vector、EMA 和 common-parameter exposure，不能用相同 epoch 替代。

shortest-duration quartile threshold 必须从 fit annotations 冻结。endpoint/high-IoU proxy
必须有精确公式、单位、聚合和 no-leak schema，只能作辅助诊断，不能替代真实 mAP@0.7。
NVML 10 Hz energy 只能作为长 timed-block secondary result，不能声称精确 single-inference
energy。

## 5. Conditions deliberately kept unchanged

以下严格条件继续保留：

- Gate 2 同 mask P2/P4/P8 TRANSPORT vs HOLD；
- detector-regret improvement CI lower > 0；
- feature-MSE improvement CI lower > 0；
- 任一 seed 均值反转则失败；
- seeds 仍为 3407/3408/3409；
- window quantile head 结构不变；
- Stage B 总预算仍为 140 successful updates；
- 不增加 target normalization、attention pooling、第二 risk head 或额外训练轮数；
- 任一 gate 失败即永久冻结 bounded appeal。

Pro 明确认为 feature MSE 双门虽然可能产生假阴性，但它是当前机制主张的预注册严格度，
不能在看到结果后放宽。

## 6. Local code risks requiring independent verification

以下是 reviewer 基于上游/公开 fork 推导的高优先级风险，不是已经确认的本地 bug：

1. risk/transport 如果挂在 backbone 内，可能被 paramwise optimizer 的 backbone lr=0
   静默冻结；必须生成逐参数 LR/WD/coverage 审计。
2. shared SingleStageDetector、AnchorFreeHead 与 post-processing 可能仍带 DUCA
   frame selector、physical-grid 或 selected-axis remap；CT 必须显式禁用并验证。
3. vit_adapter.py 可能同时含 packed-tubelet route；CT 必须是唯一 heavy routing path。
4. adapter 前必须恢复完整、按 clip/tubelet lexicographic order 的 dense tensor。
5. 140 loop iterations 可能因 AMP skip 少于 140 successful updates。
6. mutable loss normalizer 可能导致 candidate order-dependent regret。
7. checkpoint/sidecar 需要绑定 source/spec/library/split/cost/calibration hashes、seed、
   successful updates 与 claim flags。
8. Stage C runner、matched dense control 和 post-Stage-C recalibration 仍需逐文件核验。

这些风险必须在 r1 规格冻结后、正式代码修改前进行本地 source audit 和 TDD。不能把
reviewer 的不可见性误写成“本地代码不存在”。

## 7. Referenced but unavailable generated artifacts

原 review 引用了五个 sandbox artifacts，但本轮用户只提供了 review 文本；这些文件没有
附加到本地，不能登记为已获得、已审查或已集成代码：

| Artifact | Reviewer-reported SHA-256 | Local status |
|---|---|---|
| CT_P3R_3S_REQUIRED_SPEC_AMENDMENTS.md | 3bd771d98a157205f12906bec08b5a68bc30413b76f6f44e629353cd80ae37d0 | absent |
| chronotransport_protocol_primitives.py | f1284b3aac04097fa8720ae81629428198b3116237a81e6a3a7033819695b08e | absent |
| test_chronotransport_protocol_primitives.py | c5c4df5a4361ca81c70e2e39723a0aaa743944bc26be48620a43c7a1ed9d0a67 | absent |
| chronotransport_protocol_primitives.patch | 5bdcca14690c7c84da396076512bf6e112761d2957898abef3147b8058ff5408 | absent |
| CT_P3R_3S_GENERIC_PATCH_README.md | 3f35780c1e2fcd45e2520eb11f239f441971e0423c79261711beb5cee892fce4 | absent |

Reviewer 报告的 10 passed in 0.10s 仅适用于其 standalone generic primitives tests，
不能传播为本仓库 risk/cache/scheduler/profiler/Gate1-4/Stage C 通过。

## 8. Result-to-claim absorption

- Gate 1 PASS 只允许写：冻结 library 在 B* 下存在 oracle headroom。
- Gate 1 PASS 不允许写：input-dependent scheduler 已成立。
- Gate 2 PASS、Gate 3 FAIL 只允许 mechanism-level TRANSPORT claim。
- Gate 3 PASS、Gate 4 未做不允许写 full-stack speedup、高 IoU protection 或 paper-ready。
- latency PASS/quality FAIL 不能写有效 cost-quality trade-off。
- quality PASS/latency FAIL 不能写计算加速。
- 全 Gate PASS 也只能收缩到固定 THUMOS14/AdaTAD/VideoMAE-S/hardware/config 范围；
  不允许首次提出、普适 SOTA、Online TAD、跨 detector/generalization 或理论保证。
- 任一 hash/split/checkpoint mismatch 使 run invalid。
- local smoke/standalone test 只表示 engineering precheck。

## 9. Immediate project decision

吸收后的当前状态：

1. b74101d 保留为历史 design artifact，但标记 not executable as written。
2. ChronoTransport 没有被判死刑，也没有恢复成 active implementation。
3. 下一合法动作是写出、复核并冻结 CT-P3R-3S-r1 规格。
4. 在 r1 新 SHA 前，禁止实现争议代码、运行 profiler/Gate 1、训练 seed 或解锁 Stage C。
5. r1 通过后先做 local source audit/TDD，再按 Gate 1 早停；任一 gate FAIL 转 Route C，
   永久冻结为 baseline。

本页的状态是 review_absorbed / spec_revision_required，不是 implemented、tested、
experiment_running 或 empirically_supported。

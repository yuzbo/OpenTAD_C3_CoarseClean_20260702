# 1. Executive Verdict

## **REVISE_SPEC_BEFORE_CODE**

`b74101d` 不能原样执行。Gate 3 的双边 coverage 门、candidate-row Spearman、Gate 1 的 oracle→input-dependence 归因、evaluation-best comparator 隔离，以及 full-stack p50 组合语义存在阻断性数学问题。它们可在不改 head、seed、训练预算、candidate library、quantile、epsilon 和 Gate 1/2/4 数值门槛的前提下预先修订，因此尚不必立即永久终止整条假设族。

当前环境未挂载 `E:\...`，三个目标 commit 与全部 ChronoTransport 源码也不在公开 fork 可达对象中。因此本报告**不是本地实现 PASS/FAIL 证书**；只对书面合同、公开 fork 共享表面和官方上游语义作出裁决。完整集成 patch 必须停止；仅生成并验证了不依赖争议规格的通用协议原语。

---

# 2. Evidence Visibility Certificate

## 2.1 可见性状态

| 项目                                                        | 状态                                         | 证据与限制                                                                                                 |
| --------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| 本地仓库 `E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702`     | **LOCAL_VISIBILITY_BLOCKED**               | 当前 Linux 运行时未挂载 Windows `E:`；全局浅层目录搜索未发现仓库。                                                           |
| local HEAD / branch / status / staged / dirty / untracked | **UNVERIFIED**                             | 不能用公开 fork 替代本地事实。                                                                                    |
| local origin URL                                          | **UNVERIFIED**                             | 用户给出了预期 GitHub fork，但不能证明当前本地 origin 未被修改。                                                            |
| GitHub fork                                               | **VISIBLE**                                | `yuzbo/OpenTAD_C3_CoarseClean_20260702` 可访问。                                                          |
| 公开默认线可见 HEAD                                              | **VISIBLE**                                | 可见提交 `3554b6ffa3f56083baea9e7ea4399976fb572543`，提交信息为 `docs: record DUCA review gates`。               |
| `b74101d`                                                 | **NOT FOUND ON ACCESSIBLE GITHUB OBJECTS** | GitHub commit fetch 返回 422 `No commit found for SHA`。                                                 |
| `fbf8f43`                                                 | **NOT FOUND ON ACCESSIBLE GITHUB OBJECTS** | 同上。                                                                                                   |
| `92029ea`                                                 | **NOT FOUND ON ACCESSIBLE GITHUB OBJECTS** | 同上。                                                                                                   |
| 已知较晚 fork commit `88e50b1...`                             | **VISIBLE**                                | 完整 SHA 为 `88e50b17241cf4ef7d891e87b5455f45d72e345d`。                                                  |
| `opentad/models/chronotransport/risk.py`                  | **404**                                    | 在公开 HEAD 与 `88e50b1...` 均不存在。                                                                         |
| 2026-07-11 bounded-rescue spec                            | **404**                                    | `docs/superpowers/specs/2026-07-11-chronotransport-bounded-rescue-validation-design.md` 在公开 HEAD 不存在。 |
| `research-wiki/*chronotransport*`                         | **404 / search no result**                 | 公开仓库规则本身仍明确排除 `research-wiki`。                                                                        |
| File Library                                              | **NO SOURCE SNAPSHOT**                     | 仅检索到本轮 prompt 的文本副本，没有代码、diff、zip 或 repo bundle。                                                      |
| GitHub 官方上游                                               | **VISIBLE**                                | OpenTAD、AdaTAD 和 VideoMAE 固定到下述 SHA。                                                                  |
| Chrome 9223                                               | **NOT USED**                               | 因此没有获取或操作共享 Chrome target，也没有产生锁责任。                                                                   |

公开 fork 的 `AGENTS.md` 与 `RTK.md` 仍把仓库定义为 C3/`p_action`/ledger 路线，并明确要求不要加入历史 `research-wiki`。这与用户所述本地 ChronoTransport 工作树直接冲突，证明公开 HEAD 是**错误审查对象**，而不是证明 ChronoTransport 本地不存在。

## 2.2 固定的官方上游

| 上游                 | 固定版本                                       | 角色                                                                |
| ------------------ | ------------------------------------------ | ----------------------------------------------------------------- |
| `sming256/OpenTAD` | `1aa8ca4ac5e846b1e8ff69298dd6607121a01589` | detector、AdaTAD config、VideoMAE adapter、训练引擎、ActionFormer         |
| `sming256/AdaTAD`  | `25e06c720e450298ca5267fda6927f3591dcdfef` | 项目入口；README 明确说明正式实现与 checkpoints 已迁入 OpenTAD 的 `configs/adatad`。 |
| `MCG-NJU/VideoMAE` | `14ef8d856287c94ef1f985fe30f958eb4ec2c55d` | 原始 tubelet embedding、attention、MLP、ViT block 语义                   |

---

# 3. Fact Table

| 类型                  | 结论                                                                                                      | 证据等级                                           |
| ------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Repository fact** | 公开 fork 当前可见规则仍是 C3/`p_action`，不是 ChronoTransport。                                                      | 高；公开源码                                         |
| **Repository fact** | 三个目标短 SHA 与 ChronoTransport 源码/规格未在公开可达对象中定位。                                                           | 高；GitHub fetch/search                          |
| **Repository fact** | 公开 fork 的 `BackboneWrapper` 与官方 OpenTAD blob 相同。                                                        | 高；两边 blob SHA 均为 `c2ea79...`。                  |
| **Repository fact** | 公开 fork 的 `vit_adapter.py`、`SingleStageDetector`、`AnchorFreeHead`、post-processing 和 train engine 已偏离官方。 | 高；blob 与函数语义对照                                 |
| **Experiment fact** | seed 3407 旧 P3 为正式负结果，Spearman=-0.1914、row-level calibration 错误、Gate C 未解锁。                             | **用户提供；未独立打开 research-wiki/checkpoint/report** |
| **Experiment fact** | 当前 H1 unsupported、H2 partial、H3 no、H4 unverified。                                                       | **用户提供；本报告接受为冻结先验，不升格为独立核验**                   |
| **Inference**       | 本地工作树很可能包含未推送提交或位于不可见 ref；公开 fork 不能代表目标实现。                                                             | 高可信推断                                          |
| **Inference**       | 只要 local CT 复用了公开 fork 的共享 detector/backbone 表面，就必须显式关闭 DUCA/physical-grid/packed-route 行为。             | 条件性结论                                          |
| **Proposal**        | 先冻结 `CT-P3R-3S-r1` 规格，再读本地源、修实现、跑 Gate 1。                                                               | 本报告裁决                                          |
| **Proposal**        | 不更换 risk head、seed、library、训练步数、quantile 或 Gate 1/2/4 数值门。                                              | 保持一次有界上诉                                       |

---

# 4. Current Implementation Map

## 4.1 已核验的官方 AdaTAD/OpenTAD tensor flow

```text
输入
[B, 1, 3, 768, H, W]
        │
        │ pre_processing:
        │ b n c (t1 t) h w -> (b t1) n c t h w
        │ t1 = 48, t = 16
        ▼
[B*48, 1, 3, 16, H, W]
        │ BackboneWrapper flatten batch × num_seg
        ▼
[B*48, 3, 16, H, W]
        │ Conv3d patch/tubelet embedding
        │ tubelet_size = 2
        ▼
[B*48, 8*h*w, 384]
        │ 12 × transformer block
        │ attention/MLP 在单个 16-frame clip 行内执行
        │ 每个启用 block 后接 AdaTAD temporal adapter
        ▼
[B*48, 8*h*w, 384]
        │ adapter 临时重排整个窗口：
        │ [B, 384, h, w, C]
        │ dense temporal depth-wise Conv1d
        ▼
[B*48, C, 8, h, w]
        │ spatial mean + clip concat
        ▼
[B, C, 384]
        │ temporal interpolate
        ▼
[B, C, 768]
        │ Conv1DTransformerProj → FPN → ActionFormerHead
        │ dense points / loss / NMS / seconds conversion
        ▼
检测区间
```

官方 THUMOS config 固定 `window_size=768`、`chunk_num=48`、12 个 adapter block；post-processing 先得到 384 点特征，再插值回 768 点。

VideoMAE 原始 patch embedding 使用 kernel/stride 均为 `(tubelet_size, patch_size, patch_size)` 的 `Conv3d`；标准 block 是 attention 残差后接 MLP 残差。

AdaTAD adapter 在 down-projection 后，把所有 48×8=384 个 tubelet 时刻恢复为全窗口时间轴，做 dense temporal convolution，再 residual 回输入。

**由此可以确认：**

* ChronoTransport 的合法动态单元确实可以是 `48 clips × 3 layer groups`。
* heavy attention/MLP 的 clip-row gather 可以节省真实 heavy execution。
* 但每个 block 的输出在进入 adapter 前必须恢复完整、顺序正确的 dense token tensor。
* patch embedding、12 个 temporal adapter、384→768 interpolation、projection、head 和 NMS 均不因 CT 动作而稀疏。

## 4.2 目标 ChronoTransport flow：仅合同层可重建，未核验实现

```text
dense patch tokens
    │
group [0:4]
    ├─ RECOMPUTE: gather rows → heavy attention/MLP → scatter
    ├─ TRANSPORT: transport(latest cache, deploy-visible delta)
    └─ HOLD: exact latest cache
    │
dense adapter
    │
group [4:8]
    ├─ ...
    │
dense adapter
    │
group [8:12]
    ├─ ...
    │
dense adapter → dense 384 → dense 768 → detector
```

应有的 cache state：

```text
anchor_feature
latest_feature
anchor_time
latest_source_time
age
valid
action_history
group_id
clip_id
```

但以下均为 **UNVERIFIED**：

* HOLD 是否 bitwise latest；
* TRANSPORT 是否从 latest 链式递推，而不是从 anchor 重算；
* cache age/source-time 更新顺序；
* gather 是否真正绕过 heavy block；
* scatter 后 adapter 输入是否严格 dense；
* transported tensor 是否错误 detach；
* risk/transport 参数是否进入非零 LR optimizer group；
* forced-dense 是否与官方 dense path bitwise/数值等价；
* 144-cell sum head 是否仍存活；
* split、candidate order/hash、calibration、coverage 和 bootstrap 的真实实现。

## 4.3 已核验的公开 fork 共享表面偏差

公开 fork commit `88e50b1...` 中：

* `SingleStageDetector` 增加了 `frame_selector`、selector loss、meta 转发和 true-time remap。
* `AnchorFreeHead` 增加了 `physical_grid_actionformer`、selected positions 与 dense-axis GT 检查。
* post-processing 增加 selected-axis→dense-axis 插值。
* `vit_adapter.py` 增加 `TubeletTokenRedundancyAux` 和 `PackedTubeletRuntimeRoute`；后者在 heavy subpath 中 packed execution，再在 adapter 前 scatter。
* train engine 增加 AMP skipped-step 识别、`after_optimizer_step` hook 和 DUCA diagnostics。

因此“本地使用 ActionFormerHead/VideoMAE adapter 文件名”绝不等于“完整官方语义”。

---

# 5. Written-Spec Compliance Matrix

| 规格面                               | 状态                                           | 裁决                                                       |
| --------------------------------- | -------------------------------------------- | -------------------------------------------------------- |
| 权限与停止边界                           | **PASS**                                     | 未连接远端、未运行 GPU、未训练、未修改用户工作区、未 push/PR。                    |
| local HEAD/status/origin 证书       | **FAIL / VISIBILITY BLOCKED**                | 本地仓库未挂载。                                                 |
| 必读本地材料                            | **FAIL / VISIBILITY BLOCKED**                | 仅公开旧版 AGENTS/RTK 可读；目标 wiki/spec 不可见。                    |
| GitHub fork 固定提交可见性               | **FAIL**                                     | 三个目标 SHA 未找到。                                            |
| 官方 upstream 核验                    | **PARTIAL PASS**                             | 基础 AdaTAD/OpenTAD/VideoMAE 已固定并核验；local counterpart 未见。  |
| 768→48×16→384→768 几何              | **PASS for upstream / UNVERIFIED local CT**  | 官方语义明确。                                                  |
| 三层组 `[0:4],[4:8],[8:12]`          | **TARGET VALID / IMPLEMENTATION UNVERIFIED** | 切分数学上连续、无重叠、完整覆盖 12 blocks。                              |
| window quantile risk head         | **UNVERIFIED**                               | 目标架构可实现，但本地 `risk.py` 不可见。                               |
| seeds/split/140 optimizer steps   | **UNVERIFIED**                               | 不能确认是成功 optimizer update 还是 loop iteration。              |
| Gate 1                            | **FAIL AS WRITTEN**                          | claim 归因和 comparator/budget 定义需修订。                       |
| Gate 2                            | **PARTIAL**                                  | 机制门本身保守有效；实现、cluster bootstrap 与 mask matching 未核验。      |
| Gate 3                            | **FAIL AS WRITTEN**                          | coverage 双边门和 rank 单位存在阻断问题。                             |
| Gate 4                            | **PARTIAL / UNVERIFIED**                     | 门槛合理；full-stack percentile、matched exposure、proxy 定义未闭合。 |
| fail-closed artifacts/checkpoints | **UNVERIFIED**                               | 官方 checkpoint schema 不含 CT hash/claim flags。             |
| Stage C runner                    | **UNVERIFIED**                               | config 文件名来自 prompt，源码不可见。                               |
| local TDD                         | **NOT EXECUTED**                             | 没有目标 source tree。                                        |
| 通用协议 primitives                   | **PASS standalone only**                     | 10 tests 通过，不代表仓库集成。                                     |

---

# 6. Statistical Red-Team Audit

## 1. Gate 1 用 `periodic4_transport` 成本定义 HOLD-only 的 `B*`

**Verdict：REPAIRABLE**

把 `B*` 当成预注册的外部部署预算是允许的；跨 action 定义本身不必然不公平。真正条件是：

[
\text{feasible}(s,w)
====================

\mathbf 1{\operatorname{p50}(C^{\text{full}}_{s,w})\le B^*}
]

每个 HOLD candidate 必须使用其**自己的完整端到端实测成本**判断可行，不能套用 periodic4 的 action-count 成本。

* 影响：H1、H4。
* 假阳性：若用 periodic4 形状或线性 action count 代替 HOLD 实测，会错误放大 joint oracle 的可行集合。
* 假阴性：若把 transport overhead 重复加到 HOLD，会错误缩小集合。
* 最小修正：保留 `B*` 数值来源，明确它是 external envelope，并逐 candidate 实测。
* 仍属同一次上诉：**是**。

## 2. joint oracle 天然受 candidate-set size 影响

**Verdict：REPAIRABLE**

[
\min_{s\in A\cup B}R_{w,s}
\le
\min_{s\in A}R_{w,s}
]

因此 10% 改善和 paired CI 只能证明**冻结 candidate library 存在 oracle headroom**，不能单独证明 deploy-visible input dependence。候选集合变大不是 bug，而是 oracle headroom 的定义；错误是把它直接归因给 H1。

当前 prompt 没有给出 shuffle 的精确定义。若 shuffle 仍把 candidate rows 当独立样本，结论无效；若只是置换 schedule labels，oracle minimum 又可能不变。

* 影响：H1。
* 假阳性：把集合最小值优势写成输入可预测价值。
* 最小修正：Gate 1 只支持 “oracle headroom”；H1 的输入依赖归因全部交给 Gate 3。
* 仍属同一次上诉：**是**。

## 3. `evaluation-best global static` 读取 evaluation labels

**Verdict：REPAIRABLE**

它若只是一个更强的、label-using diagnostic oracle，通常是保守的，不会自动制造模型胜利。但它不能：

* 决定 deploy static policy；
* 选择 candidate/library/threshold；
* 进入 scheduler；
* 影响后续模型或 calibration；
* 被描述成可部署 comparator。

若所有 gate 和算法在打开 evaluation 前完全预注册，后续只是 futility stop，顺序执行本身不等于调参；真正危险是把 evaluation-selected identity 反馈进后续流程。

* 影响：H1、H3、H4。
* 假阳性：evaluation identity 被反馈到部署或后续拟合。
* 假阴性：作为 oracle comparator 会过强。
* 最小修正：`evaluation-best` 仅 diagnostic；主 comparator 从 fit split 冻结。
* 仍属同一次上诉：**是**。

## 4. 30 个 calibration windows、16 candidates、max residual

**Verdict：VALID，需限定 claim**

令每个窗口：

[
S_w=\max_{s\in\mathcal S_{\text{non-dense}}}
\left(R_{w,s}-\widehat q_{w,s}\right)
]

`n_cal=30`、`alpha=0.1` 时：

[
k=\left\lceil(30+1)\times0.9\right\rceil=28
]

取第 28 个 window score，有限样本下界为：

[
\frac{28}{31}\approx0.9032
]

因为先对 16 candidates 取 max，不需要再把 16 当独立样本或 Bonferroni。它给出的是**新窗口上全部候选同时覆盖的 marginal guarantee**。它不直接给出：

[
P(\text{covered}\mid \text{scheduler chose non-dense})
]

* 影响：H3。
* 假阳性：把 unconditional simultaneous guarantee 写成 conditional selective guarantee。
* 最小修正：保留算法，改 claim 与报告。
* 仍属同一次上诉：**是**。

## 5. non-dense coverage 的离散分辨率与重复窗口

**Verdict：BLOCKING**

三 seed×30 windows 是 90 个 seed-window outcome，但不是 90 个独立视频窗口。独立内容单位仍最多 30。

若 non-dense rate 恰为 20%，只有 18 个 seed-window selections：

* `16/18 = 0.8889`
* `17/18 = 0.9444`
* `18/18 = 1.0`

原 `[0.85,0.95]` 只允许前两种结果。其 Clopper–Pearson 95% 区间约为：

* `16/18`: `[0.653, 0.986]`
* `17/18`: `[0.727, 0.999]`

如果三 seed 都选同样的 6 个窗口，实际 unique-window support 可能只有 6。

* 影响：H3。
* 假阴性：一两个 outcome 就能翻转 gate。
* 最小修正：报告 seed-window count 与 unique-window count；bootstrap 以 window 为外层 cluster；取消 upper hard fail。
* 仍属同一次上诉：**是，但需新 spec SHA**。

## 6. coverage `>0.95` 直接失败

**Verdict：BLOCKING**

Conformal safety 的主要错误是 undercoverage。Overcoverage 表示保守或不够 sharp，应由 pinball 与 non-dense selection rate 约束，而不是奖励模型人为缩小上界。

即便真实 coverage 正好是 0.9：

* 18 个 outcome 时观察到 `18/18 >0.95` 的概率约 **15.0%**；
* 30 个 outcome 时观察到 `29/30` 或 `30/30 >0.95` 的概率约 **18.4%**。

这会造成显著的纯采样假失败。

* 影响：H3。
* 假阴性：有效 calibrator 被 overcoverage upper cap 杀死。
* 最小修正：coverage 只保留下界 `>=0.85`；`>0.95` 标记 `OVERCOVERED`；pinball 和选择率门不变。
* 仍属同一次上诉：**是**。

## 7. candidate-row pooled Spearman

**Verdict：BLOCKING**

把 `30×16` rows 直接做 Spearman 会混合：

* schedule 固定主效应；
* window 难度；
* seed 训练随机性；
* 同一窗口内强相关 candidate vector。

行级 CI 会严重低估不确定性。主点估计应是：

1. 每个 seed/window 内，对完整 16-candidate vector 做 Spearman；
2. 每 seed 对窗口 correlation 取 median；
3. pooled statistic 对 seed/window correlation 等权聚合；
4. bootstrap resample window IDs，seed 作为算法 replicate，候选向量整体保留。

如果预测或 target 少于 3 个不同 rank，该 seed-window 应 fail closed，不能返回伪造的 0 或跳过后继续。

* 影响：H1、H3。
* 假阳性：schedule main effect 造成高 row-pooled correlation。
* 假阴性：window difficulty 可能造成 Simpson reversal。
* 最小修正：改统计单位，不改模型。
* 仍属同一次上诉：**是**。

## 8. 140 steps 与 16 schedules 的 exposure

**Verdict：UNRESOLVED**

“平均 8–9 步/schedule”只在每 optimizer step 仅抽一个 schedule 时成立。若每个 window step 向量化计算全部 16 candidates，则每个 schedule 有 140 次 exposure。

必须从代码确认：

```text
optimizer step unit
candidate vectorization
sampler balance
actual successful AMP updates
skipped update count
```

公开 fork train engine 能识别 AMP skipped step，但 `max_train_iters` 仍限制 loop iterations；因此 140 iterations 不等于 140 successful optimizer updates。

* 影响：H3。
* 主要风险：假阴性。
* 最小修正：测试每 schedule exposure；任何 skipped update 使 formal run fail 或按预注册规则处理，不能静默少于 140 updates。
* “多训几轮”：**禁止**。
* 仍属同一次上诉：代码修复可以；修改 140 步不可以。

## 9. raw detector-loss regret 与 `epsilon=1.0`

**Verdict：REPAIRABLE**

反事实 regret 必须在同一窗口上严格 paired：

* 相同 augmentation；
* 相同 GT；
* 相同 detector/adapter weights；
* 相同 RNG 或关闭 dropout/drop-path；
* 不允许 candidate 顺序改变 loss state。

OpenTAD `AnchorFreeHead` 的 `loss_normalizer` 是可变 buffer，训练模式下会随 positive count 更新。若 dense/candidate replay 依次调用而不 snapshot/restore，target 会依赖 candidate ordering。

* 影响：H1、H3。
* 假阳性/阴性：candidate order 改变 regret。
* 最小修正：counterfactual evaluator 在 frozen/eval semantics 下运行，或逐 candidate 恢复 normalizer 与 RNG。
* target normalization：当前合同不允许新增；不能事后加入。
* 仍属同一次上诉：**是**。

## 10. dense upper risk=0 与 fail-closed

**Verdict：REPAIRABLE**

dense=0 是合理安全基准，但会形成定义漏洞：

* dense 必须从 coverage 分母排除，否则 coverage 被人为推高；

* dense 必须计入实际选择率和 full-stack cost；

* fallback dense 可能超过 `B*`，这是 safety override，不是 cost-feasible success；

* missing cost 不能因为 dense risk=0 而让 Gate 4 继续。

* 影响：H3、H4。

* 假阳性：风险安全但预算失败，被误记为整体成功。

* 最小修正：记录 `safety_override_budget_violation` 和 fallback reason。

* 仍属同一次上诉：**是**。

## 11. Gate 2 同时要求 detector 与 feature MSE CI>0

**Verdict：VALID，但保守**

feature MSE 与 detector utility 不必单调一致。一个 transport 可能改善 detector、却让 global feature MSE 变差。因此该条件可能产生假阴性。

但当前 bounded appeal 要验证的不只是任务偶然收益，还包括“transport 近似机制”。保留双门是合理的严格度，不会制造假阳性。为避免事后放宽，建议**保持原样**：

* detector PASS / feature FAIL：只能记录 task-level anomaly；

* 不允许宣称 H2；

* 整条上诉仍按合同停止。

* 影响：H2。

* 规格修改：**不需要**。

## 12. Stage C matched dense control 与 exposure

**Verdict：UNRESOLVED**

匹配单位必须是 successful optimizer updates，而不是 epoch：

* common parameter 初始化；
* fit-window order；
* augmentation RNG；
* batch/accumulation；
* AMP skip；
* LR scheduler；
* EMA；
* gradient clip；
* common parameter update count。

官方 OpenTAD upstream 每 iteration 都推进 scheduler/EMA；公开 fork 改成只在真实 optimizer step 后推进，这是实质语义差异。

* 影响：H4。
* 假阳性：dynamic 与 dense 获得不同 shared-parameter exposure。
* 最小修正：versioned exposure ledger，任何 mismatch hard FAIL。
* 仍属同一次上诉：**是**。

## 13. full-stack dynamic p50 与 static lookup

**Verdict：BLOCKING**

一般情况下：

[
Q_{0.5}(X_1+\cdots+X_k)
\neq
Q_{0.5}(X_1)+\cdots+Q_{0.5}(X_k)
]

因此不得把 patch、heavy、adapter、head、scheduler 等阶段的 p50 相加。正确方式：

* 对完整一次 forward 采集 `total_ms`；

* p50/p95 直接从 total samples 计算；

* stage samples 只做诊断；

* lookup 必须存 exact shape 的 empirical total sample distribution 或真实完整样本。

* 影响：H4。

* 假阳性：低估 tail/correlation；

* 假阴性：阶段抖动方向相反时也可高估。

* 最小修正：端到端 profiler schema 与 missing-shape fail closed。

* 仍属同一次上诉：**是**。

## 14. NVML 10 Hz energy

**Verdict：REPAIRABLE**

10 Hz 即约 100 ms 一个采样点，通常不足以解析单次 inference。它只有在以下条件下可作 block-level 次要证据：

* warmup 后同步；

* 连续 timed block 足够长；

* 记录原始时间戳和功率；

* 独立 idle baseline；

* trapezoidal integration；

* 多个独立 block 与 CI；

* 固定 clocks、温度、其他 GPU 负载。

* 影响：H4。

* 假阳性：把 under-resolved block 积分写成精确 per-inference energy。

* 最小修正：energy 降为 secondary；Gate 4 主要依赖 latency。

* 仍属同一次上诉：**是**。

## 15. endpoint/high-IoU 与 short-action proxy

**Verdict：BLOCKING**

必须预先定义：

* shortest-duration quartile 的阈值来源；
* endpoint proxy 的公式、单位、聚合方式；
* 是否只读 fit GT；
* 是否可能进入 scheduler signal；
* 与 detector regret/mAP 的关系。

建议：

* shortest quartile threshold 从 fit annotations 冻结；

* endpoint/high-IoU proxy 仅用于 fit-side辅助诊断；

* 永不进入 inference；

* 不能替代 Gate 4 的真实 `mAP@0.7` 和 shortest-quartile `mAP@0.7`。

* 影响：H4。

* 假阳性：用 train proxy 替代高 IoU 检测质量。

* 最小修正：补精确数学定义与 no-leak schema。

* 仍属同一次上诉：**是**。

---

# 7. Route Comparison

| Route              | 科学有效性   | 工程风险            | 结果可解释性                                                                     | 裁决                                   |
| ------------------ | ------- | --------------- | -------------------------------------------------------------------------- | ------------------------------------ |
| **A：严格执行 b74101d** | 不足      | 高               | Gate 3 可能因纯采样 overcoverage 假失败；row-Spearman 和 p50 可能产生错误证据                 | **拒绝**                               |
| **B：最小规格修订后实施**    | 可接受     | 仍高，但有 Gate 1 早停 | 能区分 oracle headroom、transport mechanism、risk calibration 与真实 speed-quality | **唯一选择**                             |
| **C：立即永久冻结**       | 安全但可能过早 | 低               | 会放弃一次仍可证伪的、有早停保护的 bounded appeal                                           | 当前不选；作为 Gate 1 或任一后续 gate FAIL 的强制终点 |

**选择：Route B。**

这不是对成功概率的乐观判断。旧 P3 的负结果、三 seed 约束、Gate 2 双重 CI 和 Gate 4 高 IoU 门意味着实证通过概率很可能不高；但 Gate 1 在任何新 risk/transport 训练前运行，能以较低成本决定是否立即进入 Route C。

---

# 8. GitHub Upstream Verification Matrix

## 8.1 永久链接

* VideoMAE block/patch embedding：
  `https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/modeling_finetune.py#L57-L158`
* OpenTAD AdaTAD adapter：
  `https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/backbones/vit_adapter.py`
* THUMOS VideoMAE-S adapter config：
  `https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`
* SingleStage detector：
  `https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/detectors/single_stage.py`
* AnchorFree/ActionFormer head：
  `https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/dense_heads/anchor_free_head.py`
* Train engine：
  `https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/cores/train_engine.py`
* Optimizer：
  `https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/cores/optimizer.py`
* Checkpoint：
  `https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/utils/checkpoint.py`

## 8.2 对照表

| 表面                       | 官方语义                                                   | 可见 fork `88e50b1...`                                            | 分类                                          | 科学影响                                      |
| ------------------------ | ------------------------------------------------------ | --------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------- |
| VideoMAE Conv3d tubelet  | tubelet=2；16 frames→8 time tokens                      | 基础形式沿用 mmcv PatchEmbed                                          | **lightly wrapped upstream**                | 支持 48×8=384 几何                            |
| ViT attention/MLP        | 标准残差 block                                             | 使用 OpenTAD SDPA，并加入 packed route 辅助类                            | **structurally modified**                   | CT 必须证明自己的 gather，不可复用未声明 R30 route       |
| AdaTAD adapter           | 每 block 后 dense temporal adapter                       | 基础 adapter保留，但同文件加入其他 routing surfaces                          | **modified file / base adapter compatible** | 进入 adapter 前必须恢复 dense                    |
| 384→768 config           | 48 clips，384 concat，interpolate 768                    | blob SHA 与官方 config 相同 `e0dd2a...`                              | **SAME config**                             | detector 外部时间格 dense                      |
| BackboneWrapper          | flatten clip batch、backbone、unflatten/postprocess、fp32 | blob 与官方同为 `c2ea79...`                                          | **SAME**                                    | wrapper 不是 CT 动态发生点                       |
| ActionFormer base config | projection/FPN/point strides/head loss                 | blob 与官方同为 `975a23...`                                          | **SAME config**                             | config 同不代表 inherited runtime 同           |
| `ActionFormerHead` 文件    | thin subclass                                          | 文件 blob相同                                                       | **SAME file**                               | 但继承 base 已改，不能判整体 SAME                    |
| `AnchorFreeHead`         | 标准 point assignment/loss                               | 增加 physical-grid 与 meta 路径；官方 blob `fe9c12...`，fork `8c8779...` | **MODIFIED**                                | dense CT config 必须显式禁用 physical-grid      |
| `SingleStageDetector`    | backbone→projection→neck→head                          | 增加 frame selector/remap；官方 `3ba729...`，fork `ad4e9e...`         | **MODIFIED**                                | 必须 assert `frame_selector is None`        |
| PointGenerator           | `0..T-1 × stride`                                      | blob与官方同为 `d13056...`                                           | **SAME file**                               | physical-grid base class仍可能替换 points      |
| post-processing          | grid→seconds                                           | fork增加 selected-axis remap；官方 `07978d...`，fork `d69e97...`      | **MODIFIED**                                | CT 不得携带 selected-axis metadata            |
| Optimizer grouping       | backbone rest/custom substring groups                  | blob与官方同为 `f06f75...`                                           | **SAME**                                    | CT params 名称决定是否被 lr=0 冻结                 |
| AMP/EMA/clip             | upstream每 iteration step/scheduler/EMA                 | fork识别 AMP skipped update                                       | **MODIFIED, generally safer**               | formal 140 steps须定义 successful updates    |
| checkpoint               | state/model/optimizer/scheduler/EMA                    | 公开 fork仍同一基础 schema                                             | **SAME, insufficient for CT**               | 不含 split/library/calibration/claim hashes |

官方 `BackboneWrapper` 的 checkpoint load 使用 `mmengine.load_checkpoint(..., map_location="cpu")`，随后由 config 决定预处理、backbone 和 post-processing。

官方基础 checkpoint 只保存 epoch、model、optimizer、scheduler 和可选 EMA，不包含 CT 所需 provenance。

---

# 9. Code Findings

## P0

### P0-1：目标源码不可见，任何 local implementation PASS/FAIL 都是不诚实的

* **Repository fact**：三个固定 SHA 未找到；ChronoTransport path 404；本地盘未挂载。
* **Violated contract**：必须先读 local HEAD、代码、wiki、spec。
* **影响**：无法核验 cache、risk、scheduler、profiler、Stage C 或 tests。
* **最小修复**：提供只读 repo snapshot、固定 zip/diff，或把固定 commit 置于外部可达 ref。
* **Regression test**：provenance validator 必须验证 full SHA、tree hash、dirty status、candidate/spec hash。

### P0-2：Gate 3 书面门会产生统计性假失败

* **位置**：`b74101d` 目标 spec，实际文件不可见；依据用户提供摘要。
* **违反**：coverage `[0.85,0.95]`、row-level pooled Spearman。
* **影响**：即使真实 coverage=0.9，也有约 15%–18% 几率仅因样本波动触发 upper fail。
* **修复**：采用第 10 节 amendment。
* **Regression test**：`18/18` 应报告 `OVERCOVERED`，不得 gate FAIL；cluster 数必须是 unique windows。

### P0-3：CT 参数可能被官方 optimizer 规则静默设为 LR=0

官方 THUMOS config 把 backbone rest 参数设 `lr=0`，只把名称包含 `"adapter"` 的参数放到 `2e-4` group。

optimizer 又按 substring 分类参数。

若 risk/transport/scheduler 模块挂在 backbone 内、名称不匹配专门 custom group，它们会被静默冻结。

* **Repository fact**：上游/公开 fork optimizer 规则明确。
* **Local status**：UNVERIFIED。
* **影响**：所谓 Stage B 训练可能只有 adapter 更新，risk/transport 不学习。
* **最小修复**：optimizer build 后逐参数生成审计表并 assert：

  * risk LR=`1e-4`；
  * transport LR=`1e-4`；
  * WD=`0`；
  * 无重复/遗漏参数。
* **Regression test**：按 exact parameter names 检查 group identity、LR、WD 和 `requires_grad`。

## P1

### P1-1：`ActionFormerHead` 文件相同，但有效 head 语义已修改

* fork `ActionFormerHead` 薄封装相同。
* 继承的 `AnchorFreeHead` 增加 physical-grid。
* `SingleStageDetector` 增加 selector/remap。

**修复**：CT validator 强制：

```text
frame_selector is None
physical_grid_actionformer.enabled is False
no irregular_selected_positions
no selected_axis segment remap
no GT selected-axis remap
```

**Regression test**：给 meta 注入上述任一 key，formal dense/CT config 都应 hard FAIL，而不是启动 modified path。

### P1-2：公开 `vit_adapter.py` 含其他 packed routing surface

该文件不是纯官方 adapter。CT 必须证明：

* `TubeletTokenRedundancyAux.enabled=False`
* `PackedTubeletRuntimeRoute.enabled=False`
* ChronoTransport 自己的 gather/scatter 是唯一 heavy routing path
* forced-dense 不经过任何 R28/R30 mask

**Regression test**：收集 block execution counters；dense 与 forced-dense 每 layer/clip 数完全一致。

### P1-3：adapter 前 dense reconstruction 是硬 shape 合同

Adapter 会把输入 reshape 到全局 `temporal_size=384`。任何缺行、乱序、重复 scatter 都可能：

* 直接 reshape 错；
* 更危险地，shape 正确但时间顺序错。

**Regression test**：

```text
clip IDs 0..47
tubelet IDs 0..7
before each adapter exact lexicographic order
forced-dense output ≈ official dense output
```

### P1-4：140 optimizer steps 可能被 AMP skip 偷减

公开 fork 识别 skipped step，但 `max_train_iters` 以 loop iteration 计数。

**修复**：formal runner 输出：

```text
attempted_iterations
successful_optimizer_updates
amp_skipped_updates
per-schedule exposure
```

任何 formal run `successful_optimizer_updates != 140` 必须 FAIL，不得用后续样本自动补齐，除非 r1 规格明确规定重放同一 batch。

### P1-5：counterfactual loss 可能受 mutable normalizer 与候选顺序污染

**修复**：

* snapshot/restore `loss_normalizer`；
* freeze RNG；
* candidate order permutation test；
* dense reference只算一次并复用 detached scalar。

**Regression test**：逆转 candidate order 后，每个 `(window,schedule)` regret 在严格容差内相同。

### P1-6：官方 checkpoint schema不足以支撑 CT claim

公开 checkpoint 只含训练状态。

CT checkpoint 必须另含或绑定不可变 manifest：

```text
schema_version
source_commit
spec_hash
candidate_library_hash
split_manifest_hash
risk_head_schema
cost_profile_hash
calibration_hash
seed
successful_optimizer_steps
claim_flags
```

任何不匹配 dense fallback，所有 claim flags false。

## P2

### P2-1：公开 post-processing 已支持 selected-axis remap

虽然 CT 应保持 dense 768 grid，但共享 post-processing 已修改。

**修复**：CT run artifact 明确记录：

```text
detector_axis = dense_768
selected_axis_remap_used = false
physical_grid_used = false
```

### P2-2：Stage C “同 epoch”不是 matched control

必须比较 successful updates、sampler/augmentation、EMA 和 common-parameter trajectory。测试应至少比较：

* common parameter name set；
* step count；
* LR trace hash；
* batch/window order hash；
* AMP skip vector；
* EMA update count。

### P2-3：NVML 10 Hz 不支持单次 inference 精确能耗

只允许 block-level secondary result；不足解析度必须标 `ENERGY_UNDERRESOLVED`。

## P3

### P3-1：公开 AGENTS/RTK 已过时于本地研究状态

这不是 ChronoTransport 方法 bug，而是 provenance/synchronization 缺陷。公开 fork 自身仍要求 C3/p_action 路线。

---

# 10. Required Spec Amendments

完整、可直接提交的新规格文本：

[下载 CT-P3R-3S-r1 必需规格修订](sandbox:/mnt/data/CT_P3R_3S_REQUIRED_SPEC_AMENDMENTS.md)

SHA-256：

```text
3bd771d98a157205f12906bec08b5a68bc30413b76f6f44e629353cd80ae37d0
```

必须修改的核心条款：

1. `evaluation-best global static` 只允许 diagnostic，不参与 pass/fail、配置、部署或后续拟合。
2. `B*` 保留 periodic4_transport 的完整端到端 p50 来源，但每个 HOLD candidate 用自身完整实测 p50 判可行。
3. Gate 1 只证明 oracle headroom，不宣称 input dependence；H1 归因移至 Gate 3。
4. conformal sample unit 固定为 window；每窗口先取 16-candidate max residual。
5. coverage 改为单边 `>=0.85`；`>0.95` 记 `OVERCOVERED`，不失败。
6. Spearman 改为 seed/window 内 candidate-vector rank，再以 unique-window cluster bootstrap。
7. constant quantile 只能在 fit split 拟合并冻结。
8. dense fallback 记录 safety override 与预算违规，不能自动算 cost success。
9. p50/p95 只能来自完整 end-to-end samples。
10. Stage C control 匹配 successful optimizer updates，不用 epoch 代替。
11. short-action 阈值从 fit annotations 冻结；NVML energy 降为 block-level secondary。

采纳后实验名必须改为 **`CT-P3R-3S-r1`**。它仍是同一个 bounded appeal，但不能把 `b74101d` 与 r1 结果混成同一 protocol。

---

# 11. Patch Architecture

| 文件/模块                        | 责任                                                                                                  | 当前状态                                   |
| ---------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `protocol_primitives.py`     | canonical JSON/hash、library identity、split manifest、conformal、coverage、bootstrap、profile、gate chain | **完整 proposed code，standalone tested** |
| `risk.py`                    | 精确 window quantile head、dense zero、schedule conditioning                                            | **STOPPED：需真实 D/schema/调用点**           |
| `replay.py`                  | 每 window 输出完整 candidate vector；禁止 row sample                                                        | **需本地 schema**                         |
| `formal_stage_b.py`          | Gate 1/2/3 report 与 stop chain                                                                      | **受 r1 规格影响，停止**                       |
| `profiler.py`                | 完整 end-to-end sample schema                                                                         | generic primitives 已给；integration 未做   |
| `cost_lookup.py`             | exact-shape empirical totals + provenance                                                           | 需真实 profiler/shape schema              |
| `training.py`                | successful update counter、exposure ledger                                                           | 需真实 runner                             |
| `checkpoint.py` 或 CT sidecar | CT hashes、claim flags、atomic artifact                                                               | generic atomic writer 已给               |
| `scheduler.py`               | dense fallback、actual-selected logging                                                              | 受本地接口影响                                |
| `tests/`                     | cluster、hash、coverage、profile、gate chain                                                            | generic 10 tests 已给                    |
| Stage C config/runner        | matched control、recalibration                                                                       | 完全未核验                                  |

## 必须保持的接口不变量

```text
candidate order is immutable
library hash includes ordered complete schedule payload
split manifest is shared across all three seeds
dense upper risk is exact zero
candidate rows never become calibration/bootstrap samples
all evaluation baselines are already frozen
all missing/nonfinite/hash mismatch paths choose dense
dense fallback does not erase a budget violation
end-to-end totals are measured samples
```

---

# 12. Complete Optimization Code

由于总裁决是 `REVISE_SPEC_BEFORE_CODE`，且目标 local source 不可见，没有生成伪造的 `risk.py`/runner 集成 diff。以下是**唯一允许生成的争议无关、add-only 通用补丁**：

* [下载 unified patch](sandbox:/mnt/data/chronotransport_protocol_primitives.patch)
* [下载完整 Python 模块](sandbox:/mnt/data/chronotransport_protocol_primitives.py)
* [下载 focused tests](sandbox:/mnt/data/test_chronotransport_protocol_primitives.py)
* [下载补丁边界说明](sandbox:/mnt/data/CT_P3R_3S_GENERIC_PATCH_README.md)

SHA-256：

```text
protocol module
f1284b3aac04097fa8720ae81629428198b3116237a81e6a3a7033819695b08e

tests
c5c4df5a4361ca81c70e2e39723a0aaa743944bc26be48620a43c7a1ed9d0a67

unified patch
5bdcca14690c7c84da396076512bf6e112761d2957898abef3147b8058ff5408

README
3f35780c1e2fcd45e2520eb11f239f441971e0423c79261711beb5cee892fce4
```

该模块完整实现：

* ordered candidate-library canonical hash；
* 已存在 140/30/30 shared split 的 fail-closed 校验；
* one-max-residual-per-window simultaneous calibration；
* actual-selected non-dense coverage；
* fit-only per-schedule constant quantile；
* unique-window clustered bootstrap；
* complete end-to-end p50/p95；
* recursive forbidden-key rejection；
* Gate 1–3 Stage-C lock；
* atomic canonical artifact writes。

它**没有**臆造：

* frozen 16 schedules；
* risk-head 输入维度；
* cache/runtime tensor schema；
* Gate 1 oracle runner；
* Stage B/Stage C CLI；
* local import/registry wiring。

---

# 13. Verification Evidence

| Exact command / operation                                                                                                              |  执行 | Exit/result                                   | 含义                                           |
| -------------------------------------------------------------------------------------------------------------------------------------- | --: | --------------------------------------------- | -------------------------------------------- |
| 搜索当前文件系统与 `/mnt` 中的目标 repo                                                                                                             | YES | exit 0；未发现                                    | `LOCAL_VISIBILITY_BLOCKED`                   |
| `git clone --filter=blob:none --no-checkout https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git /tmp/ct-public-clone`         | YES | exit 128；`Could not resolve host: github.com` | container Git 网络不可用；随后改用 GitHub connector，只读 |
| GitHub `fetch_commit(b74101d)`                                                                                                         | YES | 422 not found                                 | 目标 commit 不在可达 GitHub 对象                     |
| GitHub `fetch_commit(fbf8f43)`                                                                                                         | YES | 422 not found                                 | 同上                                           |
| GitHub `fetch_commit(92029ea)`                                                                                                         | YES | 422 not found                                 | 同上                                           |
| GitHub `fetch_commit(88e50b1)`                                                                                                         | YES | success                                       | 证明 fork 本身可访问且其他短 SHA fetch 正常               |
| GitHub fetch `risk.py` at public HEAD/88e50                                                                                            | YES | 404                                           | 只证明公开 ref 缺失，不证明本地缺失                         |
| `git apply --check /mnt/data/chronotransport_protocol_primitives.patch`，在隔离临时 Git repo                                                 | YES | exit 0                                        | add-only patch 语法可应用                         |
| `git apply ...`                                                                                                                        | YES | exit 0                                        | 隔离 temp repo 应用成功                            |
| `git diff --check`                                                                                                                     | YES | exit 0                                        | 无 whitespace/error                           |
| `PYTHONPYCACHEPREFIX=/tmp/... python -m py_compile chronotransport_protocol_primitives.py test_chronotransport_protocol_primitives.py` | YES | exit 0                                        | standalone 语法通过                              |
| `python -m pytest -p no:cacheprovider -q test_chronotransport_protocol_primitives.py`                                                  | YES | `10 passed in 0.10s`                          | generic primitives/tests 通过                  |
| 本地现有 ChronoTransport tests                                                                                                             |  NO | `NOT_EXECUTED_LOCAL_VISIBILITY_BOUNDARY`      | 无目标 source tree                              |
| C3 regression tests                                                                                                                    |  NO | 未触及实际 repo/shared imports                     | 不能声称通过                                       |
| GPU profiling / detector eval / training                                                                                               |  NO | 禁止边界                                          | 无科学实验结果                                      |
| remote PRECHECK/Slurm                                                                                                                  |  NO | 禁止边界                                          | 无远端证据                                        |

**测试结论边界：**“10 passed”只适用于下载补丁中的 standalone protocol primitives；绝不意味着 CT risk head、cache、scheduler、profiler、Gate 1–4 或 Stage C 已测试。

---

# 14. Unverified Items

以下项目仍是阻断项，不能从 prompt 或公开旧分支推断：

* local HEAD、branch、origin、dirty/staged/untracked；
* `b74101d` 完整规格原文及其精确定义；
* `fbf8f43` formal negative report；
* `92029ea` Stage-B 实现闭环；
* 全部 `research-wiki` 原始节点、source registry、query pack；
* 全部 `opentad/models/chronotransport/*`；
* 16 个 non-dense schedules 的动作矩阵、顺序、首 clip、canonical hash；
* split manifest 的真实 200 window IDs；
* 三 seed 是否共享同一 split；
* risk head D、schedule/action/age/offset conditioning；
* HOLD/TRANSPORT cache state；
* gather/scatter 是否真实节省 heavy compute；
* transport loss 与 detector loss gradient path；
* counterfactual loss 是否 order-invariant；
* actual successful optimizer-step count；
* checkpoint claim flags、resume、atomicity；
* measured periodic4_transport full-stack p50；
* static cost lookup 是否 nonlinear；
* profiler 是否测完整 pipeline；
* Gate 1 oracle/comparator/shuffle implementation；
* Gate 2 P2/P4/P8 exact mask matching；
* Gate 3 actual-selected coverage；
* Stage C runner 与 matched dense control；
* mAP@0.7、short quartile mAP、energy；
* 任何旧或新实验数字的原始 artifact。

---

# 15. Next-Step Plan

所有步骤均为计划，未执行。

| 步骤                            | 输入                             | 验证/命令入口                                                     | 输出                                    | 硬停止条件                                    | research-wiki 状态                   |
| ----------------------------- | ------------------------------ | ----------------------------------------------------------- | ------------------------------------- | ---------------------------------------- | ---------------------------------- |
| 1. 冻结 r1 规格                   | 本报告 amendment + b741 原文        | `git diff --check`; 人工逐条对照                                  | 新 spec SHA                            | 任一 amendment 被拒绝则不写代码                    | `BLOCKED_SPEC → SPEC_R1_FROZEN`    |
| 2. 建隔离 worktree               | approved local SHA             | `git worktree add --detach <temp> <sha>`                    | 不污染当前工作区的 temp tree                   | dirty/source SHA 不符                      | `SPEC_R1_FROZEN`                   |
| 3. 完整读取材料                     | 第 2 节全部文件                      | `git status --short`; `rg --files \| rg -i chronotransport` | visibility certificate                | 任一必读文件缺失                                 | `SOURCE_AUDIT`                     |
| 4. 对现有实现做逐文件 diff             | local source + upstream SHAs   | `git diff <upstream> -- <paths>`                            | local-vs-upstream matrix              | 无法固定 upstream counterpart                | `SOURCE_AUDIT`                     |
| 5. 先写 failing tests           | r1 spec                        | pytest focused files                                        | 失败证据                                  | 测试未覆盖 candidate clustering/fallback      | `TDD_RED`                          |
| 6. 应用经审查 patch                | 本地真实接口 + generic primitives    | `git apply --check`; `py_compile`; pytest no cache          | local patch evidence                  | 任何 existing C3 regression 失败             | `TDD_GREEN_LOCAL`                  |
| 7. 本地 PRECHECK                | validators/configs             | 先运行各 entrypoint `--help`，再按真实 CLI                           | manifest/hash/optimizer/freeze report | 非零 LR、hash、dense parity 任一失败             | `LOCAL_PRECHECK_PASS/FAIL`         |
| 8. 远端 `PRECHECK_ONLY`         | frozen bundle                  | 真实 launcher 中 `PRECHECK_ONLY=1`                             | 环境与 artifact schema                   | 不得启动训练；任一写界越界停止                          | `REMOTE_PRECHECK_PASS/FAIL`        |
| 9. GPU1 measured-cost profile | exact hardware/software/config | `profile_chronotransport_schedules.py` 的真实 CLI              | complete end-to-end samples           | stage-p50 相加、样本不足、provenance mismatch    | `COST_READY/FAIL`                  |
| 10. Gate 1 HOLD oracle        | cost profile + frozen replay   | Gate-1-only entrypoint                                      | oracle headroom report                | 任一门 FAIL → 永久冻结                          | `GATE1_PASS` 或 `FROZEN_GATE1_FAIL` |
| 11. 三 seed Stage B            | 仅 Gate1 PASS                   | seeds 3407/3408/3409；140 successful updates                 | three checkpoints + exposure ledgers  | 任何 seed protocol/hash/update mismatch    | `STAGE_B_COMPLETE/INVALID`         |
| 12. Gate 2                    | P2/P4/P8 exact masks           | matched mechanism evaluator                                 | detector+feature clustered CI         | 任一门/seed reversal FAIL → 冻结              | `GATE2_PASS` 或 `FROZEN_GATE2_FAIL` |
| 13. Gate 3                    | calibration + eval             | r1 selection-aware evaluator                                | coverage/rank/pinball/selection       | 任一门 FAIL → 冻结                            | `GATE3_PASS` 或 `FROZEN_GATE3_FAIL` |
| 14. Stage C/P5                | 仅 Gates1–3 PASS                | matched dynamic+dense runners                               | matched checkpoints + new calibration | exposure mismatch或eval leakage           | `STAGE_C_COMPLETE/INVALID`         |
| 15. Gate 4                    | full-stack profile + mAP       | final evaluator                                             | final claim artifact                  | 任一 speed/quality/overhead/seed gate FAIL | `GATE4_PASS` 或 `FROZEN_GATE4_FAIL` |

由于本地 CLI 不可见，当前不能诚实给出带 flags 的“精确运行命令”。下一名工程师首先应执行：

```bash
python tools/bata/validate_chronotransport_adatad.py --help
python tools/bata/run_chronotransport_stage_b_formal.py --help
python tools/bata/profile_chronotransport_schedules.py --help
python tools/bata/train_chronotransport_stage_b.py --help

rg -n "ArgumentParser|add_argument|PRECHECK_ONLY|if __name__" \
  tools/bata scripts opentad/models/chronotransport
```

不能凭文件名猜 CLI 后把命令写成已验证。

---

# 16. Result-to-Claim Matrix

| 最终状态                              | 允许表述                                                                                                             | 禁止表述                                                   |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| 目标 source 仍不可见                    | “公开审查因 provenance/visibility 阻断”                                                                                 | “代码通过/失败”“实现完整”                                        |
| Gate 1 FAIL                       | “冻结 library 在该预算下未显示足够 oracle headroom”                                                                          | H1–H4；继续换 library/head/seed                            |
| Gate 1 PASS                       | “存在预算可行的冻结-library oracle headroom”                                                                              | “input-dependent scheduler 已成立”                        |
| Gate 2 FAIL                       | 可报告 oracle headroom                                                                                              | TRANSPORT>HOLD；修改 transport 后复活同一 appeal               |
| Gate 2 PASS、Gate 3 FAIL           | “TRANSPORT mechanism 在冻结测试中通过”                                                                                   | calibrated risk、deploy scheduler、H3/H4                 |
| Gate 3 PASS、Gate 4 未做             | “冻结 risk/scheduler 在 evaluation protocol 上通过 calibration/rank 门”                                                 | full-stack speedup、高 IoU保护、paper-ready                 |
| Gate 4 latency PASS、quality FAIL  | 可报告系统加速但任务质量门失败                                                                                                  | “有效 cost-quality trade-off”                            |
| Gate 4 quality PASS、latency FAIL  | 可报告质量保留                                                                                                          | “计算加速/能效提升”                                            |
| Gate 1–4 全 PASS                   | “在固定 THUMOS/AdaTAD/VideoMAE-S/hardware/config 下，r1 bounded appeal 同时通过 oracle、transport、risk 和 full-stack gates” | 首次提出、普适 SOTA、Online TAD、跨 backbone/generalization、理论保证 |
| 任一 hash/split/checkpoint mismatch | “run invalid”                                                                                                    | 将结果纳入任何 gate 或论文表                                      |
| coverage=1.0                      | “overcovered；结合 pinball/selection rate 解释”                                                                       | 自动写成 calibration 成功或失败                                 |
| feature MSE 单独改善                  | “feature approximation signal”                                                                                   | 替代 detector regret 或 mAP                               |
| local smoke/standalone test PASS  | “工程 precheck”                                                                                                    | 科学 gate PASS                                           |

---

# 17. Final Kill Criteria

## 继续的必要条件

只有全部满足才允许进入下一阶段：

1. `CT-P3R-3S-r1` 在任何新训练/评估前固定 SHA；
2. local HEAD、spec、candidate library、split manifest、upstream commits 全部可见并哈希；
3. dense/forced-dense parity 通过；
4. HOLD bitwise latest；
5. TRANSPORT latest-based；
6. risk/transport 参数具有正确 LR/WD/梯度；
7. 140 次是 successful optimizer updates；
8. calibration/bootstrap sample unit 是 window；
9. profiler 使用完整 end-to-end samples；
10. 所有 inference payload no-GT/no-teacher/no-oracle/no-ledger；
11. 任一 missing/mismatch 都 dense fallback 并阻断 claim。

## 允许修代码、但不算科学复活

* Gate 前发现确定性代码 bug；
* local import/schema/serialization bug；
* profiler instrumentation bug；
* environment/CUDA/c10.dll 问题；
* artifact 未生成导致 run 根本无科学结果。

修复后必须从受影响 gate 起完整重跑，保留旧 run 为 `INVALID_IMPLEMENTATION`。

## 必须永久冻结 ChronoTransport bounded appeal

以下任一发生即转 Route C，不得换 head、loss、seed、library、训练长度或门槛：

* Gate 1 oracle improvement/CI/cost saving 任一失败；
* Gate 2 detector 或 feature CI 失败；
* Gate 2 任一 seed 均值反转；
* Gate 3 coverage、rank、pinball、selection rate 任一失败；
* 需要把 140 steps 改成更多 steps 才能“救活”；
* 需要 normalization/新 target/attention pooling/第二 risk head；
* Gate 4 saving、高 IoU、short-action、overhead、Pareto 或 seed consistency 任一失败；
* 任一 evaluation leakage；
* 任一 candidate/split/library hash 漂移；
* 任一结果后再修改 r1 统计定义；
* full-stack cost 只能靠 FLOPs、线性 action cost 或 stage-percentile sum 支撑。

**最终行动含义：先批准并冻结 `CT-P3R-3S-r1`，再允许代码审计和实现；在此之前，不运行 Gate 1、不训练新 seed、不做 GPU profiling，更不进入 Stage C。**

## 1. Executive Verdict

**`REVISE_SPEC_BEFORE_PLAN`**

`02199f8` 已正确冻结 pre-adapter cache、all-row TIA、age 47/embedding cap 8 与 requested/executed cost，但仍存在会改变样本单位、训练暴露、梯度、Gate 1、Gate 3、Gate 4 和 claim unlock 的 P0/P1 歧义。协议可通过一次最小修订闭合；当前禁止进入 `writing-plans`。

---

## 2. Evidence Visibility Certificate

| 项目                                                     | 状态                                                           | 证据分类与说明                                                                                                                                                              |
| ------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 固定审查 commit `1f5f7254a390f183121e6c4b7cebcebd2f2954d1` | **VISIBLE**                                                  | `REPOSITORY_FACT`：commit、固定树和全部下述文件均可读取。                                                                                                                             |
| 规格 commit `02199f8`                                    | **VISIBLE**                                                  | 完整 SHA 为 `02199f8eb34fc0e34e342afcc357269df457d091`。                                                                                                                 |
| 被替代规格 commit `b74101d`                                 | **VISIBLE**                                                  | 完整 SHA 为 `b74101d3c5d91a79a0a9a0f81f1dd87a755fb5cf`。                                                                                                                 |
| 历史 Stage-B/P3 commit `92029ea`                         | **VISIBLE**                                                  | 完整 SHA 为 `92029eaa60aabb79de4cc6bd48595e9cf6910f8f`。                                                                                                                 |
| r1 规格文件                                                | **VISIBLE**                                                  | `REPOSITORY_FACT`：Git blob SHA 为 `b72b0d2d17f575f8ac00cbac0f87063613b8a687`，共 536 行。                                                                                 |
| 规格 SHA-256                                             | **REGISTERED MATCH / BYTE HASH NOT INDEPENDENTLY CERTIFIED** | review index 与用户给出的登记值均为 `871420261BD1C19CC515218A6016A91ED7D553B73740AB41C2E02AA7F96609F9`；当前只读 GitHub connector 没有提供独立的原始字节 SHA-256 复算证书。因此不能把“登记值一致”夸大为“已独立重算一致”。 |
| ChronoTransport 源码                                     | **VISIBLE**                                                  | `actions/cache/transport/runtime/risk/scheduler/profiler/cost_lookup/losses/training/formal_stage_b/replay` 均可见。                                                     |
| tests、configs、runners、validators                       | **VISIBLE**                                                  | 包括核心测试、integration、formal Stage-B tests、Stage A/B/C configs、GPU1 launcher 与 profiler builder。                                                                        |
| review index 中五份材料                                     | **VISIBLE**                                                  | 原 Pro review、absorption、local source audit、两份独立审查均可读取；index 明确要求它们只能作为待核验论据。                                                                                         |
| OpenTAD 固定上游                                           | **VISIBLE**                                                  | commit `1aa8ca4ac5e846b1e8ff69298dd6607121a01589` 的模型、数据、optimizer、scheduler、train engine 可读。                                                                        |
| AdaTAD 固定上游                                            | **VISIBLE**                                                  | commit `25e06c720e450298ca5267fda6927f3591dcdfef` 可读；该仓库 README 指向 OpenTAD 中的正式 AdaTAD 实现。                                                                           |
| VideoMAE 固定上游                                          | **VISIBLE**                                                  | commit `14ef8d856287c94ef1f985fe30f958eb4ec2c55d` 可读。                                                                                                                |
| 原始实验 artifacts                                         | **NOT PRESENT IN REVIEW PACKAGE**                            | `REPOSITORY_FACT`：固定 package 明确不包含数据、checkpoint、GPU logs 或新行为结果。                                                                                                     |
| 独立 `EXPERIMENT_FACT`                                   | **NONE ESTABLISHED**                                         | `92029ea` 和 reviewer 文档中的数值是仓库记录或 reviewer 报告；本轮没有打开其原始 ledger/checkpoint/log，因此不升级为本轮独立核验的 `EXPERIMENT_FACT`。                                                       |

证据边界如下：

* `REPOSITORY_FACT`：固定 commit 中的实际源码、规格、测试、配置与固定上游源码。
* `EXPERIMENT_FACT`：本轮无。
* `REVIEWER_REPORT`：旧 P3 数值、H1–H4 历史状态以及既有 reviewer 的判断。
* `INFERENCE`：由公式、源码和统计定义推出的歧义、混淆或不成立性。
* `PROPOSAL`：第 14 节给出的唯一规格修订文本。

结论：**不触发 `GITHUB_VISIBILITY_BLOCKED`**。

---

## 3. Repository / Experiment / Reviewer / Inference / Proposal Fact Table

| 类型                | 本轮可接受的结论                                                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `REPOSITORY_FACT` | 02199f8 已把 cache 边界定义为 pre-adapter heavy output，并要求完整 48-row heavy surrogate 经原 AdaTAD adapter 写回全部 rows。                                                                       |
| `REPOSITORY_FACT` | 固定提交中的历史 runtime 仍只把 `adapted` 写回 `effective_flat_mask`，即 RECOMPUTE rows；这是 pre-r1 历史实现，不是 r1 行为结果。                                                                             |
| `REPOSITORY_FACT` | 历史 risk head 对 48×3 个非负 cell scalar 求和；r1 规格要求替换为 mean/max pooling 的窗口级 head。                                                                                                   |
| `REPOSITORY_FACT` | 官方训练使用带 GT 保留条件的 `random_trunc`，val/test 使用 `sliding_window`；因此 200 个 video ID 不能直接当作 200 个冻结 window。                                                                           |
| `REPOSITORY_FACT` | 官方 AdaTAD block 的顺序是 attention residual → MLP residual → full temporal adapter；adapter 在完整 384 点 temporal axis 上做无 causal mask 的 Conv1d。                                        |
| `EXPERIMENT_FACT` | **本轮无独立核验项。**                                                                                                                                                                   |
| `REVIEWER_REPORT` | 历史 P3 被记录为负结果，Stage C/P5 未解锁；旧数值不在本轮被重新执行或独立验证。                                                                                                                                 |
| `INFERENCE`       | 当前 Stage-B 候选公式产生 candidate/video `mod 4` 混淆；Gate 1 oracle-shuffle 条件近乎由逐窗口最小值定义保证；Stage-C loss ownership 与 retry 尚不可确定实现。                                                      |
| `PROPOSAL`        | 采用一窗一视频 Option A、block-rotated exposure、显式 `autograd.grad` 路由、可回滚 AMP retry、evaluation-only adjudicator、group-wise motion top-k、pre-Gate1 immutable registration 和聚类 Gate 4 推断。 |

---

## 4. Sealed Provisional Review

`REPOSITORY_FACT`：我先按 review index 的 Round 1 顺序读取 r1 规格、固定提交源码、tests/configs/runners 与固定上游；在读取任何 reviewer 结论之前，形成的 provisional verdict 已是：

> **`REVISE_SPEC_BEFORE_PLAN`**

封闭审查独立发现：

1. `INFERENCE`：§5 只冻结了 video IDs，却在 §9–§12 把其直接称为 window；由于官方 train crop 依赖 GT，conformal 的 `n=30`、bootstrap 外层单位和跨 seed 配对没有封闭定义。
2. `INFERENCE`：§7 的 `(update+offset) mod 16` 在 offsets 为 0/4/8 时始终保持 `candidate mod 4 = video-position mod 4`。
3. `INFERENCE`：§9 的“逐窗口 minimum oracle 对 shuffled assignment”在相同 feasible set 内近乎定义成立，不能构成科学 Gate。
4. `INFERENCE`：§13 只写了 loss ownership 目标，没有冻结可执行的多 loss AMP/autograd 算法、None-grad 语义和参数覆盖证明。
5. `INFERENCE`：Stage-C matched exposure 要求“AMP skip vector 一致”，但没有定义 overflow 后如何恢复 mutable state、如何保留 scaler backoff，以及独立 arms 是否必须 lockstep skip。
6. `INFERENCE`：Gate 3 缺最低 unique-window support 和 fit-only constant baseline 的完整 replay 定义。
7. `INFERENCE`：Gate 4 仍使用 point thresholds 和 ratio-of-medians；没有为 p50 saving、mAP noninferiority、overhead margin 与 static latency 冻结单侧聚类推断。
8. `INFERENCE`：motion 的单个全局 threshold 无法逐窗口精确匹配 periodic RECOMPUTE count。
9. `INFERENCE`：checkpoint sidecar 不等于“profile 前完成的不可变注册”；checkpoint/data/config identity 仍可在 profile 前漂移。
10. `INFERENCE`：Gate 3/4 的 claim 名称和 `deploy=false / paper=false` 的永久条件不够精确。

读取既有材料后，以上 provisional blockers 均没有被推翻；第二位独立 reviewer 反而确认了其中六项。

---

## 5. Spec Compliance Matrix

| 规格章节                           |                             状态 | 裁决                                                                                            |
| ------------------------------ | -----------------------------: | --------------------------------------------------------------------------------------------- |
| §1 裁决与证据边界                     |                        PARTIAL | 有界上诉和 stop chain 清楚；文件头仍写“设计已获用户批准”，与当前 `REVISE_SPEC_BEFORE_PLAN` 状态冲突。                       |
| §2 官方骨架                        |                           PASS | 768→48×16、384 内部 grid、12 blocks、三层组、离线任务边界均已冻结。                                               |
| §3 cache/adapter               |       PASS WITH TEST AMENDMENT | pre-adapter cache、all-row adapter、latest-chain、detach 方向正确；需补 live-current-row 与 parity 验收定义。 |
| §3.4 age                       |                           PASS | hard validity 47 与 embedding cap 8 分离正确。                                                      |
| §4 window risk head            |                        PARTIAL | mean/max window head 正确；action/group embedding 维数未显式冻结为 8/8。                                  |
| §5 split/checkpoint/randomness |                   **BLOCKING** | 冻结的是 video IDs，不是实际 temporal windows；checkpoint 也尚未进入独立 pre-Gate1 registration。               |
| §6 candidate library           |                        PARTIAL | 16 candidates 已冻结；motion threshold 不可保证逐窗口 exact count。                                       |
| §7 Stage B                     |                   **BLOCKING** | 候选与 canonical video position 存在 mod-4 混淆。                                                     |
| §8 cost/B*                     | PASS WITH IMPLEMENTATION TESTS | full-stack total、requested/executed、B* 和 repair invalidation 方向正确。                            |
| §9 Gate 1                      |                   **BLOCKING** | oracle-shuffle 近乎定义性必胜；evaluation-best 的角色与 bootstrap selection 尚未闭合。                         |
| §10 Gate 2                     |                        PARTIAL | 科学门合理且保守；需冻结一窗一视频和 window→seed hierarchical unit。                                             |
| §11 Gate 3                     |                   **BLOCKING** | support、constant baseline replay、selected-window coverage 与 cluster CI 仍未完整定义。                |
| §12 bootstrap/proxies          |                        PARTIAL | 5000 replicates、vector preservation 等需更精确；Gate 4 与 Gate 1–3 是不同 population。                   |
| §13 Stage C                    |                   **BLOCKING** | loss-specific gradient routing、AMP retry、successful schedule exposure 和 no-DDP 合同缺失。          |
| §13 Gate 4                     |                   **BLOCKING** | point estimates 不足；需要单侧 CI、video clustering、balanced crossover 和 paired margin。               |
| §14 provenance/claims          |                   **BLOCKING** | 缺独立 immutable registration；claim unlock 名称与后置要求不完整。                                           |
| §15 remote discipline          |                        PARTIAL | 历史 launcher 有 GPU1/root/precheck guards，但 r1 Stage C/P5 launcher 尚未按新 unlock chain 实现。        |
| §16 completion                 |                        PARTIAL | 应明确“写入新 spec commit + 独立 SHA + spec-only re-review”后才可进入 plans。                               |

---

## 6. Severity-Ranked Findings

### P0-1 — Video 与 window 统计单位没有冻结

**裁决：BLOCKING。**

规格定义 140/30/30 个 video IDs，却在 conformal、Spearman 和 bootstrap 中直接使用“30 windows”。官方 train path 的 `random_trunc` 会反复采样 crop，并使用 GT intersection 保证至少包含动作；它既不是一窗一视频，也不是 label-free。继续沿用它会使不同 seed、schedule 和 replay 看到不同 temporal content，破坏配对和 exchangeability。

固定行号：

* [https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1f5f7254a390f183121e6c4b7cebcebd2f2954d1/docs/superpowers/specs/2026-07-11-chronotransport-ct-p3r-3s-r1-design.md#L153-L179](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1f5f7254a390f183121e6c4b7cebcebd2f2954d1/docs/superpowers/specs/2026-07-11-chronotransport-ct-p3r-3s-r1-design.md#L153-L179)
* [https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/datasets/transforms/end_to_end.py#L201-L317](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/datasets/transforms/end_to_end.py#L201-L317)

### P0-2 — Candidate × video exposure confounding

当前公式等价于：

[
c=(j+\text{offset})\bmod 16
]

而 0、4、8 三个 offset 均为 0 modulo 4，所以三个 seed 都满足：

[
c\bmod4=j\bmod4.
]

每个 candidate 只训练在一个固定的 video-position 子群上，schedule effect 与 canonical video-order effect 不可区分。

固定行号：

* [https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1f5f7254a390f183121e6c4b7cebcebd2f2954d1/docs/superpowers/specs/2026-07-11-chronotransport-ct-p3r-3s-r1-design.md#L243-L253](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1f5f7254a390f183121e6c4b7cebcebd2f2954d1/docs/superpowers/specs/2026-07-11-chronotransport-ct-p3r-3s-r1-design.md#L243-L253)

### P0-3 — Stage-C loss ownership 不可直接实现

“LD 更新 A/T、LF 只更新 T、LR 只更新 R”不是一个足够的实现合同。若简单求 total loss，LF 会写入 adapters；若 detach adapter 输出，又会切断 LF→T 的 adapter Jacobian。规格必须冻结 object-identity sets、`autograd.grad` 顺序、同一 GradScaler scale、None-grad 语义、一次 unscale/clip/step 以及 DDP 禁令。

固定行号：

* [https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1f5f7254a390f183121e6c4b7cebcebd2f2954d1/docs/superpowers/specs/2026-07-11-chronotransport-ct-p3r-3s-r1-design.md#L457-L466](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1f5f7254a390f183121e6c4b7cebcebd2f2954d1/docs/superpowers/specs/2026-07-11-chronotransport-ct-p3r-3s-r1-design.md#L457-L466)

### P0-4 — AMP retry 与 matched exposure 未闭合

官方 OpenTAD 在每个 iteration 后推进 scheduler 和 EMA；r1 若采用 overflow retry，就必须显式改为“只在 successful update 后推进”。否则两个 arms 即使 epoch 相同，也可能拥有不同的 LR、EMA 和 common-parameter exposure。官方 scheduler 将 epoch 数乘以 dataloader length，并按 iteration step。

### P0-5 — Gate 1 oracle-shuffle 是近乎定义性 Gate

对每个窗口，joint oracle 已定义为 feasible schedule regret 的 minimum。若 shuffle 后仍给该窗口分配 feasible schedule，则：

[
R^{oracle}_w \le R^{shuffled}_w
]

逐窗口成立。该 test 主要验证“minimum 小于任意元素”，不是验证 input-dependent schedule structure。它应删除，而不是修补 permutation 细节。

### P0-6 — 缺 pre-Gate1 immutable registration

当前 checkpoint/sidecar 条款是在训练产物层绑定 identity；它没有要求在任何 profile、replay 或 evaluation 打开前，以独立 artifact 冻结 checkpoint、data、window、library、exposure、profiler order 与硬件环境。因而仍存在结果前后替换起点或数据 identity 的自由度。

### P1-1 — Motion baseline 名称与行为不一致

当前历史函数是单一 threshold；其 RECOMPUTE 数随窗口运动分布变化，无法保证逐窗口与 P2/P4/P8 数量匹配。

应采用唯一的 **group-wise per-window top-k**，并重命名为 `motion_topk_p{2,4,8}`。

### P1-2 — Gate 3 support 与 constant baseline 不充分

只要求 pooled 20% non-dense 可能由三个 seed 反复选择同六个窗口达成。还必须冻结：

* 每 seed 至少 6/30；
* pooled 至少 18 个 seed-window；
* 至少 10 个 distinct windows；
* all-16-schedules × all-140-fit-windows 的 no-grad replay constant baseline。

### P1-3 — Gate 4 需要聚类单侧推断

当前 point saving、point mAP drop 和 ratio-of-medians 可能将 timing noise、overlapping windows 和 seed 伪重复当作证据。应以 official video 为最外层 cluster，直接对 total samples 重算 p50，并对 mAP 重建完整 metric。

### P1-4 — Claim unlock 仍过宽

`calibrated_risk` 应缩为 `calibrated_risk_on_frozen_window_protocol`。Gate 4 只能打开固定 AdaTAD/THUMOS14/GPU1 的 metric/latency；`deploy` 和 `paper` 即使全 PASS 也必须保持 false。

### P1-5 — Forced-dense 与 head dimensions 缺确定性验收

当前规格只写“numerical parity”，未冻结 FP32 exact、AMP tolerance、gradient parity、adapter/non-adapter block 与 checkpoint parity。action/group embedding 维度也未显式固定；历史实现实际均使用 8。

### P2

1. “5000-window bootstrap”应统一改为“5000 bootstrap replicates”；实际 evaluation windows 仍为 30。
2. 规格登记 SHA-256 尚缺独立字节复算证书。
3. `APPROVE` 后仍应进行一次 mechanical spec-only diff review，避免粘贴修订时产生新冲突。

---

## 7. Architecture and Cache Semantics Audit

### 7.1 正确的精确 tensor/cache flow

```text
block input z_l
= previous block's complete post-adapter dense state
  [B, 48, N, C]

for each block l in group g:

    for each clip c:
        RECOMPUTE:
            u_live[c] = H_l(z_l[c])
            anchor_detached = latest_detached = u_live[c].detach()
            age = 0

        HOLD:
            u_live[c] = latest_detached
            age += 1

        TRANSPORT:
            u_live[c] =
                T_g(latest_detached,
                    current block input z_l[c],
                    clamp(age + 1, max=8))
            latest_detached = u_live[c].detach()
            age += 1

    U_l = ordered concatenation of all 48 u_live rows

    if adapter enabled:
        z_{l+1} = original AdaTAD Adapter_l(U_l)
                   applied to all 48 rows
    else:
        z_{l+1} = U_l

same-block cache boundary remains U_l, before Adapter_l
```

`REPOSITORY_FACT`：原 VideoMAE block 先做 attention residual，再做 MLP residual。

`REPOSITORY_FACT`：AdaTAD 在上述 heavy block 后调用 adapter；adapter 把 48 clips × 8 tubelets 恢复为完整 384 temporal axis，再做带对称 padding 的 Conv1d，没有 causal mask。

固定上游链接：

* [https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/modeling_finetune.py#L104-L133](https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/modeling_finetune.py#L104-L133)
* [https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/backbones/vit_adapter.py#L21-L77](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/backbones/vit_adapter.py#L21-L77)
* [https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/backbones/vit_adapter.py#L274-L296](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/backbones/vit_adapter.py#L274-L296)

### 7.2 十二项架构裁决

| 问题                                           | 裁决                                                                                                         |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| pre-adapter heavy cache 是否忠实恢复官方 composition | **是。** 它把 action 限定在 attention/MLP heavy subpath，随后恢复官方 all-row adapter。                                   |
| cache 能否放在 post-adapter                      | **不能。** post-adapter tensor 已混合完整窗口 temporal context；把它作为逐 clip rolling cache 会引入未来 clip 对缓存状态的影响，并产生循环语义。 |
| HOLD bitwise identity 边界                     | 仅限 **pre-adapter `latest` heavy state**。完整 block output 经非因果 adapter 后不应 bitwise 恒等。                       |
| `cache_detach=True` 的作用                      | 只截断跨 clip recurrent gradient；不能 detach 当前 row 写入 `U_l` 的 live tensor。                                      |
| 当前 TRANSPORT 是否仍能获得 adapter Jacobian         | **必须能。** `u_live` 先进入 `U_l` 与 adapter，另存 detached alias 进 cache。                                           |
| 当前历史 runtime 是否已满足                           | **否，但这只是 pre-r1 static fact。** 它只把 adapter 输出写回 RECOMPUTE rows。                                            |
| forced-dense 路径                              | 必须直接调用原 block forward，而不是经 action gather/scatter 重构。历史 runtime 已采用 direct dense path。                      |
| age 47                                       | clip 0 重算后，clips 1–47 的实际 age 为 1–47；均合法。                                                                  |
| age 48                                       | 仅在假设存在第 49 个 clip 时出现，必须 hard-invalid。                                                                     |
| embedding cap 8                              | age 1–8 对应各自 index；9–47 均映射 index 8。                                                                       |
| risk age                                     | 使用真实 age 的 `age/(1+age)`，不得 clamp 8。                                                                       |
| only schedules                               | 保留原名；不删除、不修复、不重命名。ages 9–47 在其训练 exposure 中真实出现，因此不是完全训练外分布，但 exposure 很少。                                 |

---

## 8. Exposure and Sample-Unit Audit

### 8.1 Block-rotated 公式独立复算

拟议公式：

[
b=\lfloor j/16\rfloor,\quad p=j\bmod16,\quad
c=(p+5b+\text{offset})\bmod16.
]

offset 为 0、4、8。

#### 每 seed

| Seed | 9 次 exposure 的 candidates | 8 次 exposure 的 candidates | tail 顺序，`j=128..139` |
| ---- | ------------------------- | ------------------------- | -------------------- |
| 3407 | 0–3、8–15                  | 4–7                       | 8–15、0–3             |
| 3408 | 0–7、12–15                 | 8–11                      | 12–15、0–7            |
| 3409 | 0–11                      | 12–15                     | 0–11                 |

#### 三 seed 汇总

* candidate 0–3：各 27 次；
* candidate 4–15：各 26 次；
* 总数：`4×27 + 12×26 = 420`；
* 每个 video 跨三 seed 获得 `c, c+4, c+8 mod16`，三者不同。

#### 完整 128-window 部分

* 每个 16-window block 都是 0–15 的 permutation；
* 每 candidate、每 seed 在四种 `p mod4` 中各出现 **2 次**；
* 三 seed 汇总后，每 candidate 在四种 `p mod4` 中各出现 **6 次**。

这纠正了第二 reviewer 文本中可能被理解为“每 seed 各 6 次”的表述；**6 次是三 seed 汇总值，不是单 seed 值**。其主要结论仍成立。

### 8.2 是否仍有结构混淆

`INFERENCE`：仍存在确定性的 block/position/candidate 结构，但它不再是固定的 `mod4` 分层：

* candidate 在 block 间以 5 旋转；
* 5 与 16 互素；
* 完整 blocks 内 exact permutation；
* tail 不平衡被完整冻结和登记。

在仅 140 updates、禁止增加 seed/steps 的约束下，不需要再引入 hash permutation 或更强 Latin square。更复杂的构造会增加隐藏设计自由度，且不能增加实际信息量。

### 8.3 140 updates 能否充分训练 risk head

`INFERENCE`：不能预先保证。每 candidate 每 seed 只有 8–9 个直接 optimizer exposures；整个共享 head 有 140 个训练窗口。该预算足以构成一次有界可证伪测试，但不支持“所有 transport/risk ideas 均失败”的广泛否定。

允许的负结论必须收缩为：

> 固定 window head、固定 16-candidate library、固定 block-rotated exposure 和固定 140-successful-update protocol 未获得所需证据。

### 8.4 唯一样本单位方案

**选择 Option A。拒绝 Option B。**

Gate 1–3：

* 200 个 train videos；
* 每 video 恰好一个 label-free、hash-frozen 768-point temporal window；
* fit/calibration/evaluation 分别是 140/30/30 个 unique windows；
* 一个 manifested window 严格一一对应一个 video；
* conformal 外层单位：30 个 calibration windows；
* Spearman 单位：每 `seed × evaluation window` 的完整 16-vector；
* coverage observation：selected seed-window，但 uncertainty 以 window cluster；
* bootstrap outer：unique manifested window；
* seed 为内层 algorithm replicate。

Gate 4：

* 使用不同的 official full-video/sliding-window population；
* overlapping windows 必须聚类到 official video；
* Gate 3 的 conformal guarantee **不转移**到 Gate 4；
* Gate 4 只验证固定 full-video population 上的 metric/latency。

Option A 会产生明确 population shift，但这是可接受的 bounded-appeal 最小修复；关键是收缩 claim，而不是假装两个 population 相同。

---

## 9. Statistical Red-Team Audit

### 9.1 Gate 1

#### Oracle-shuffle

**删除 hard gate。**

逐窗口 minimum 本来就不劣于同 feasible set 内的任意 assignment；shuffle 不能验证 input dependence。

#### Comparator taxonomy

| 类型                                      | 可使用 evaluation labels | 能否进 deployment/checkpoint/fit |                  是否可进 Gate 1 |
| --------------------------------------- | --------------------: | ----------------------------: | ---------------------------: |
| deployable comparator                   |                     否 |                             是 |                            是 |
| calibration-frozen comparator           |        只在 calibration |                可作为冻结 baseline |                            是 |
| evaluation-only adjudication comparator |                     是 |                      **绝对禁止** | **是，仅用于 oracle-headroom 裁决** |
| diagnostic oracle                       |                     是 |                             否 |          仅报告或保守 adjudication |

`evaluation-best global static` 应从“diagnostic-only、完全不参与 pass/fail”改为：

> **evaluation-only adjudication comparator**。

原因是 Gate 1 本身只判断 frozen library 的 oracle headroom，而不是部署性能。允许一个 evaluation-selected static 作为更强的保守 comparator 不会产生 scheduler leakage；它只增加假阴性风险。

要求：

* 每个 bootstrap replicate 内重新选择 evaluation-best static；
* 每个 replicate 内重新选 strongest comparator；
* time-only/layer-only/joint oracle 的 candidate-set size 必须报告；
* comparator identity 不得写入 checkpoint、Stage B config、threshold 或后续 fit。

#### Gate 1 hard conditions

1. full-sample mean relative reduction ≥10%；
2. 5000 次 paired unique-window bootstrap，absolute improvement 的 95% LCB >0；
3. `B*` 相对 dense full-stack p50 saving ≥20%；
4. diversity 只作 diagnostic。

若 comparator mean ≤`1e-12`，relative reduction 未定义，Gate 1 直接 FAIL；不得用加 epsilon 人为制造比例。

### 9.2 Gate 2

现有双门保留：

* detector absolute improvement CI lower >0；
* feature-MSE improvement CI lower >0；
* 每 seed 均值不得反转。

这是严格但有效的 mechanism gate。feature MSE 可能产生假阴性，但放宽它会改变预注册的 H2。

统计单位固定：

* outer：unique manifested evaluation window；
* inner：三 seeds；
* P2/P4/P8 vector 整体移动；
* 不把 3 periods 或 candidate rows 当独立窗口。

### 9.3 Gate 3

#### Conformal

每 calibration window：

[
S_i=\max_{s=1}^{16}\max(r_{i,s}-\hat q_{i,s},0).
]

`n=30, τ=.9`：

[
k=\lceil(30+1)\times .9\rceil=28.
]

这提供的是：

> 对一个新 frozen-window，16 candidates 同时覆盖的 marginal guarantee。

它不是：

[
P(\text{covered}\mid\text{scheduler selected non-dense}).
]

不需要对 16 candidates 做 Bonferroni，因为先对 candidate residual 取了 max。

#### Minimum support

保留并同时要求：

* 每 seed non-dense selection ≥6/30；
* pooled selected seed-window ≥18；
* 至少 10 个 distinct evaluation windows 被任一 seed 选择。

这些阈值不是自然常数，而是预注册的 anti-degeneracy floor；在禁止增加 evaluation sample 的条件下合理。

#### Coverage

* pooled point coverage ≥0.85；
* per-seed coverage 必报；
* window-all-selected coverage 必报；
* cluster-aware one-sided 95% LCB 必报；
* CI lower 不作为 0.85 hard gate；
* coverage >0.95 记 `OVERCOVERED`，不失败。

原因：在最低 support `n=18` 时，即使 `18/18` 全覆盖，one-sided exact 95% lower bound 约为 **0.8467**。要求 CI lower≥0.85 会使最低合法 support 下的完美结果仍失败，形成近乎不可能 Gate。

#### 防止“总 dense + 极保守 upper”伪成功

同时依赖：

* non-dense support；
* distinct-window support；
* vector Spearman；
* pooled rho CI；
* pinball improvement；
* sharpness；
* calibrated/uncalibrated/dense selection-rate 对照。

#### Constant baseline

不能只用每 schedule 8–9 个 optimizer exposure target。

唯一合法定义是：

* Stage B 训练完成后；
* 在全部 140 fit manifested windows 上；
* no-grad replay 全部 16 schedules；
* 每 schedule 使用 140 targets 的 finite-sample `τ=.9` order statistic，rank=`ceil(141×.9)=127`；
* baseline hash 在打开 calibration/evaluation 前冻结。

#### Bootstrap 与 multiplicity

* outer：window；
* inner：seed；
* candidate vector 整体移动；
* 不反向以 seed 为 outer；
* Gate 3 多条件是 intersection-union gate，全部必须通过，因此无需额外 multiplicity correction。

### 9.4 Gate 4

#### Saving 与 noninferiority

定义：

[
\Delta_{\text{lat}}=
\frac{p50_{dense}-p50_{CT}}{p50_{dense}},
]

要求 one-sided 95% LCB ≥0.15。15% 不是统计学必然值，而是预注册的系统意义阈值；应保留。

定义：

[
\Delta_{\text{map}}=mAP@0.7_{dense}-mAP@0.7_{CT}.
]

one-sided 95% UCB ≤1.5 正确表达 noninferiority。shortest-fit-Q1 同理。

#### Bootstrap 单位

* latency：official video → matched invocation block → seed；
* mAP：official video → seed；
* **不得把重复 timing invocations 当作新的 mAP label samples**；
* 每次 mAP replicate 必须重建该 replicate 的完整 predictions/GT 和 mAP，不能平均 per-video AP 代替；
* overlapping sliding windows 均留在对应 video cluster 内。

#### Overhead

使用逐 invocation paired margin：

[
m_i=0.40(dense_heavy_i-selected_heavy_i)-overhead_i.
]

要求：

* median heavy saving >0；
* bootstrap LCB of median(`m_i`) >0。

这优于当前 ratio-of-medians，因为它保留 heavy saving 与 overhead 的同 invocation 相关性。

#### Static comparator

* hard latency comparator：**calibration-frozen static**，身份固定，不在 evaluation 或 bootstrap 中重选；
* 要求 one-sided UCB of `p50_CT-p50_static` ≤0；
* 当前已有的 CT 对 calibration-frozen static 的 detector-regret improvement CI lower>0 应保留，以防“adaptive scheduler 不优于 static”仍解锁；
* evaluation-best/adaptive adjudicator 若作 diagnostic，则必须在每个 bootstrap replicate 内重选。

#### Multiplicity

所有 hard conditions 构成 intersection-union test；只有全部通过才解锁，不需要额外 multiplicity correction。

---

## 10. Stage-B / Stage-C Execution Audit

### 10.1 Stage B

`REPOSITORY_FACT`：02199f8 已冻结 Stage B 为 FP32、batch 1、world size 1、shuffle false、140 successful updates；历史 bespoke trainer 没有 autocast/GradScaler。

因此：

* Stage B 的 AMP skips 必须恒为 0；
* 非有限梯度或未执行 optimizer step 是 `INVALID_IMPLEMENTATION`；
* 不能通过补 epochs 达到 140；
* resume 只能恢复尚未完成的同一 140-update sequence；
* candidate×video matrix 必须逐 update 登记。

### 10.2 官方 Stage-C 语义核验

固定 OpenTAD config：

* train batch size=2；
* AMP=true；
* EMA=true；
* adapters LR=`2e-4`、WD=`0.05`；
* backbone rest LR=0；
* scheduler warmup epoch=5、max epoch=100；
* workflow end epoch=60。

官方 scheduler 将 epoch 数乘以 dataloader length，并按 iteration 调用 `step()`。

官方 config **没有冻结 world size**。r1 的 `world_size=1` 是本协议选择，不应伪称官方事实。

在 140 fit windows、batch 2 下：

* 70 successful updates/epoch；
* 60 epochs；
* 4200 successful updates；
* warmup steps=`5×70=350`；
* cosine max steps=`100×70=7000`；
* training 在 step 4200 停止；
* EMA decay 固定为 OpenTAD 默认 `0.999`。

### 10.3 唯一 Stage-C gradient algorithm

选择 **object-identity-disjoint `autograd.grad`**。拒绝：

* single total loss + gradient hooks；
* functional/frozen adapter 副本；
* name-substring ownership；
* DDP/no_sync。

参数集合：

* `A`：AdaTAD adapters；
* `T`：transport；
* `R`：canonical risk predictor。

要求：

[
A\cap T=A\cap R=T\cap R=\varnothing
]

且其 union 恰好覆盖全部 `requires_grad=True` parameters。

同一 counterfactual forward 产生 `LD/LF/LR`。使用同一个 GradScaler 当前 scale：

1. `gD = grad(scale(LD), A∪T, retain_graph=True)`；
2. `gF = grad(scale(0.1LF), T, retain_graph=True)`；
3. `gR = grad(scale(0.1LR), R, retain_graph=False)`；
4. 手工写入 scaled `.grad`；
5. 一次 `unscale_(optimizer)`；
6. finite/expected-unused audit；
7. global clip=1；
8. 一次 `scaler.step/update`。

这样：

* LF forward 仍穿过当前 trainable adapter 的 input Jacobian 到 T；
* 因为 A 不在该次 `autograd.grad` 的 inputs 中，LF 不写 A.grad；
* LR 的 signals 与 regret target 对 A/T detach；
* scheduler argmin 不反传。

Expected unused：

* A 的 detector-loss aggregate grad 必须 finite/nonzero；
* R 的 pinball aggregate grad 必须 finite/nonzero；
* executed action 中无 TRANSPORT 时，T grad 可全 None/0；
* 有 TRANSPORT 的 successful exposures 汇总后，T grad norm 必须 finite且>0。

### 10.4 AMP retry

两个 arms 可有不同 overflow 历史。要求相同的是：

* ordered successful batch hashes；
* augmentation hashes；
* 4200 common adapter updates；
* common-A LR trace；
* common-A EMA update count。

**不要求 lockstep mutual skip。** 强制一方因另一方 overflow 而跳过会人为耦合两个实验。

Overflow 时必须：

1. 保留 GradScaler backoff；
2. 不推进 batch/sampler/successful index/schedule/LR/optimizer/EMA；
3. 清空 grads；
4. 恢复 pre-forward RNG；
5. 恢复全部 forward-mutated model buffers 和 Python state；
6. retry 同一 materialized batch。

必须 bitwise 恢复的状态包括：

* BN running stats 与 counters；
* `AnchorFreeHead.loss_normalizer`；
* dropout/checkpoint RNG；
* CT `latest_*`、action history、cache、summary；
* profiler buffers/counters；
* scheduler Python state；
* optimizer state；
* EMA；
* append 到模型内部的 diagnostic lists。

唯一允许不恢复的是：

* GradScaler backoff；
* 模型外部 append-only retry audit log。

初始 attempt 后最多 3 次 retry；第 4 次仍 overflow 为 `INVALID_IMPLEMENTATION`。

---

## 11. Cost and Profiler Audit

### 11.1 历史实现状态

`REPOSITORY_FACT`：历史 `ChronoProfiler` 只聚合 stage samples；旧 builder 从外部 `latency_ms` 生成 p50/p95，没有证明这些样本来自完整 decode→postprocess 链。

这不表示 r1 已失败，只表示 implementation plan 必须实现新 profiler。

### 11.2 Full-stack total

每个 invocation 的唯一 primary timing：

```text
CUDA synchronize
start wall-clock
decode
preprocess
H2D
innovation
scheduler
heavy / transport / cache
all AdaTAD adapters
projection / neck / head
postprocess
CUDA synchronize
stop wall-clock
```

* p50/p95 直接从 `total_ms` samples 计算；
* 不得把 stage percentiles 相加；
* stage timings 只作诊断；
* decode 在每个 arm 内独立执行；
* 不允许 dense decode 后把 tensor 直接复用给 CT/static。

### 11.3 Balanced crossover

对三个 primary arms：

* `D`：matched dense；
* `C`：ChronoTransport；
* `S`：calibration-frozen static。

使用六序列循环：

```text
D C S
C S D
S D C
S C D
C D S
D S C
```

这使每个 arm 在三个位置各出现两次，并使每个有向 predecessor pair 各出现两次。

timing invocation 数冻结为：

* 每 arm/seed 至少 200；
* 最终数为 6 的整数倍；
* 所有 official invocation 至少出现一次；
* 不足或为完成六序列 block 的附加 repetitions 由预注册 hash 排序确定。

### 11.4 Requested/executed

02199f8 的方向正确：

* formal Gate 中任何 repair、transport NaN fallback 或 whole-window dense fallback 都使该 sample/run invalid；
* requested cost 不得继续用于 gate；
* executed action hash 与 actual total_ms 仅用于错误审计；
* dense fallback 不能掩盖 `B*` violation。

历史 runtime 确实在 repair 后仍保留原 selection 的 `estimated_cost`，证明该测试是必要的。

固定源码链接：

* [https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1f5f7254a390f183121e6c4b7cebcebd2f2954d1/opentad/models/chronotransport/runtime.py#L562-L589](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1f5f7254a390f183121e6c4b7cebcebd2f2954d1/opentad/models/chronotransport/runtime.py#L562-L589)
* [https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1f5f7254a390f183121e6c4b7cebcebd2f2954d1/opentad/models/chronotransport/runtime.py#L636-L692](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1f5f7254a390f183121e6c4b7cebcebd2f2954d1/opentad/models/chronotransport/runtime.py#L636-L692)

### 11.5 Energy

NVML 10 Hz 只能用于长 timed block 的梯形积分 secondary metric。不得声称单 inference energy。记录 raw timestamps、power、idle baseline、clock policy、temperature 与 throttle flags。

---

## 12. Leakage and Provenance Audit

### 12.1 禁止进入 inference payload

以下键及其递归别名必须被 schema 拒绝：

* GT segments/labels；
* teacher features/predictions；
* dense heavy reference；
* counterfactual regret；
* raw detector predictions；
* full-token state；
* replay ledger；
* evaluation oracle identity；
* calibration/evaluation labels。

允许：

* deploy-visible pooled signals；
* requested candidate identity；
* true deterministic age；
* registered cost table；
* calibrated offset；
* validity/hash metadata。

### 12.2 Window selection

window start 的 hash 输入不得包含：

* annotation bytes；
* GT count；
* action duration；
* class；
* detector output。

annotation SHA-256 可以进入 identity manifest，但不能进入 start-selection digest。

### 12.3 Oracle leakage

* evaluation-best static 可影响 Gate 1 adjudication；
* 不得写入 Stage B checkpoint；
* 不得影响 candidate library、B*、motion/random、epsilon、q_conf；
* 不得成为 deploy static；
* bootstrap 中必须重新选择，以反映其 evaluation selection。

### 12.4 Checkpoint/data identity

任何 profile/replay 前必须冻结：

* spec bytes hash；
* implementation commit；
* config；
* dense checkpoint；
* video/media；
* annotation；
* split/window manifests；
* library/action matrices；
* candidate exposure matrix；
* profiler order；
* upstream；
* software/hardware。

这应由独立 registration artifact 完成，而不是散落在后续 checkpoints 中。

---

## 13. Agreement Matrix

缩写：

* **Pro**：原 Pro review；
* **Abs**：原 review absorption；
* **Local**：local source audit；
* **Ind-1**：第一位空白上下文 reviewer；
* **Ind-2**：第二位规格 reviewer；
* **Oracle**：本轮独立裁决。

| 争议                                                  | Pro                     | Abs                     | Local                   | Ind-1                   | Ind-2                      | Oracle            |
| --------------------------------------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | -------------------------- | ----------------- |
| 旧 visibility 结论适用于当前固定 commit                       | `DISAGREE`              | `PARTIALLY_AGREE`       | `DISAGREE`              | `DISAGREE`              | `DISAGREE`                 | `DISAGREE`        |
| Gate 1 只证明 oracle headroom                          | `AGREE`                 | `AGREE`                 | `AGREE`                 | `AGREE`                 | `ALREADY_FIXED_IN_02199F8` | `AGREE`           |
| evaluation-best 必须永远 diagnostic-only                | `AGREE`                 | `AGREE`                 | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `DISAGREE`                 | `PARTIALLY_AGREE` |
| evaluation-best 可作 evaluation-only hard adjudicator | `DISAGREE`              | `DISAGREE`              | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `AGREE`                    | `AGREE`           |
| conformal 应先 per-window max                         | `AGREE`                 | `AGREE`                 | `AGREE`                 | `AGREE`                 | `ALREADY_FIXED_IN_02199F8` | `AGREE`           |
| coverage upper cap 应删除                              | `AGREE`                 | `AGREE`                 | `AGREE`                 | `AGREE`                 | `ALREADY_FIXED_IN_02199F8` | `AGREE`           |
| vector Spearman / window cluster                    | `AGREE`                 | `AGREE`                 | `AGREE`                 | `AGREE`                 | `ALREADY_FIXED_IN_02199F8` | `AGREE`           |
| pre-adapter cache + all-row TIA                     | `PARTIALLY_AGREE`       | `PARTIALLY_AGREE`       | `AGREE`                 | `AGREE`                 | `ALREADY_FIXED_IN_02199F8` | `AGREE`           |
| age 47 / embedding cap 8                            | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `PARTIALLY_AGREE`       | `AGREE`                 | `ALREADY_FIXED_IN_02199F8` | `AGREE`           |
| requested/executed cost 分离                          | `PARTIALLY_AGREE`       | `AGREE`                 | `AGREE`                 | `AGREE`                 | `ALREADY_FIXED_IN_02199F8` | `AGREE`           |
| candidate/video mod-4 混淆                            | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `PARTIALLY_AGREE`       | `AGREE`                    | `AGREE`           |
| `p mod4` 每 candidate 各 6 次的表述                       | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `PARTIALLY_AGREE`          | `PARTIALLY_AGREE` |
| video/window 单位缺口                                   | `PARTIALLY_AGREE`       | `PARTIALLY_AGREE`       | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `AGREE`                    | `AGREE`           |
| Stage-C loss-specific autograd                      | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `PARTIALLY_AGREE`       | `INSUFFICIENT_EVIDENCE` | `AGREE`                    | `AGREE`           |
| 独立 arms 可不同 overflow history                        | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `AGREE`                    | `AGREE`           |
| oracle-shuffle 近乎 tautology                         | `PARTIALLY_AGREE`       | `PARTIALLY_AGREE`       | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `AGREE`                    | `AGREE`           |
| pre-Gate1 immutable registration                    | `PARTIALLY_AGREE`       | `PARTIALLY_AGREE`       | `AGREE`                 | `PARTIALLY_AGREE`       | `AGREE`                    | `AGREE`           |
| Gate4 单侧 clustered CI                               | `PARTIALLY_AGREE`       | `PARTIALLY_AGREE`       | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `AGREE`                    | `AGREE`           |
| motion 应改为 exact top-k                              | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `PARTIALLY_AGREE`          | `AGREE`           |
| deploy/paper 始终 false                               | `PARTIALLY_AGREE`       | `PARTIALLY_AGREE`       | `PARTIALLY_AGREE`       | `PARTIALLY_AGREE`       | `AGREE`                    | `AGREE`           |

关键差异只有两项：

1. 本轮允许 `evaluation-best static` 作为 **evaluation-only adjudication comparator**，但绝不允许进入部署或后续拟合。
2. block-rotation 的 `p mod4` 平衡是每 seed 各 2 次、三 seed 汇总各 6 次。

---

## 14. Patch-Ready Amendment Set

以下是唯一建议修订。不得保留 A/B 选项，也不得在结果后改变。

### Amendment 1 — Cache live tensor、parity 与 embedding dimensions

**SECTION:** §3.3、在 §3.4 后新增 §3.5，并修订 §4 输入维数。

**PROBLEM:** 当前方向正确，但没有明确区分当前 row 的 live tensor 与 detached cache alias，也没有冻结 forced-dense 验收和 embedding dimensions。

**VERDICT:** `REVISE`.

**WHY:** 错误 detach 位置会让 LF/LD 无法回传到 transport；过弱 parity test 会允许 masked adapter 或其他 dense drift。

**FALSE_POSITIVE_RISK:** 一个改变官方 block 语义的实现被误判为 parity。

**FALSE_NEGATIVE_RISK:** 对 CUDA AMP 强求不必要的 bitwise equality。

**MINIMAL_AMENDMENT:** 明确 live/current-row、detached recurrent cache、CPU/FP32 exact 与 AMP tolerance；action/group embedding 均冻结为 8。

**DELETE/REPLACE:** 替换现有 §3.3；在 §3.4 后插入 §3.5；替换 §4 关于每 cell 输入的首段。

**EXACT NEW MARKDOWN:**

```markdown
### 3.3 截断梯度与当前 row 的 live-tensor 合同

`cache_detach=True` 只截断跨 clip 的 recurrent gradient，不得截断当前
RECOMPUTE/TRANSPORT row 到本 block 输出的梯度。

对每个当前 row，先产生 `u_live` 并把该 live tensor 写入完整 heavy-surrogate
tensor `U_l`：

- RECOMPUTE：`u_live = H_l(z)`；
- HOLD：`u_live = latest_detached`；
- TRANSPORT：
  `u_live = T_g(latest_detached, z, min(actual_age + 1, 8))`。

随后才更新 recurrent cache alias：

- RECOMPUTE：
  `anchor_detached = u_live.detach()`，
  `latest_detached = u_live.detach()`；
- TRANSPORT：
  `latest_detached = u_live.detach()`；
- HOLD：cache 不变。

因此，当前 RECOMPUTE/TRANSPORT row 仍可通过
`U_l -> Adapter_l -> downstream loss` 获得本 row 梯度；后续 clip 不得通过
`latest_detached/anchor_detached` 向此前 clip 回传 recurrent gradient。本协议不允许
full BPTT。

### 3.5 forced-dense parity acceptance

forced-dense 必须直接调用原始 AdaTAD/VideoMAE block forward，不得通过
RECOMPUTE gather/scatter 重构 dense 路径。

正式实现必须同时满足：

1. CPU deterministic tiny-block：输出与输入梯度 bitwise equal；
2. remote CUDA deterministic FP32：backbone 输出、detector loss、输入梯度和
   adapter parameter gradients bitwise equal；
3. remote CUDA AMP FP16：上述量满足 `atol=1e-6, rtol=1e-5`；
4. 分别覆盖 adapter-enabled block、adapter-disabled block、activation checkpoint
   on/off；
5. forced-dense 不得产生 schedule repair、fallback 或 requested/executed action
   mismatch；
6. legacy dense checkpoint strict load 后必须继续满足上述 parity。

任一失败属于 `INVALID_IMPLEMENTATION`，不是 science FAIL。

## 4. 唯一窗口级 quantile head

每个 cell 的输入维数固定为：

- deploy-visible signal：6；
- action embedding：8；
- group embedding：8；
- normalized true age：1。

因此 `D=23`。action embedding 与 group embedding 必须是两个独立的 8-dimensional
embedding tables，不得共享参数或在实现后改变维数。
```

**RATIONALE:** 该修订不改变模型结构，只关闭 detach 和 parity 的实现自由度。

**REQUIRED TEST:** TDD 1–8、15、18。

---

### Amendment 2 — One-window-per-video Option A

**SECTION:** 替换 §5.1，并在 §5.3 前新增 frozen-window protocol。

**PROBLEM:** video IDs 与 temporal windows 混用；官方 `random_trunc` 使用 GT，不能用来构造 label-free frozen windows。

**VERDICT:** `REVISE`.

**WHY:** 不冻结实际 temporal indices，就没有 paired replay、30-window conformal 或 window-cluster bootstrap。

**FALSE_POSITIVE_RISK:** schedule 恰好看到更有利的随机 crop，被误认为方法收益。

**FALSE_NEGATIVE_RISK:** 不同 schedule 看见不同困难 crop，抹去真实收益。

**MINIMAL_AMENDMENT:** 每个 train video 用唯一 label-free SHA-256 规则冻结一个 768-point window。

**DELETE/REPLACE:** 替换现有 §5.1；保留 split 比例，但把单位改为 manifested windows。

**EXACT NEW MARKDOWN:**

```markdown
### 5.1 共享 video split 与 one-window-per-video manifest

Gate 1–3 的 population 固定为 THUMOS14 train split 中的 200 个 unique videos。
首先以 split seed `3407` 对 video IDs 做 canonical SHA-256 排序，冻结：

- fit：140 videos；
- calibration：30 videos；
- evaluation：30 videos。

然后为每个 video 恰好生成一个 label-free 768-point temporal window。不得调用
official `random_trunc` 决定 window start，因为该 transform 使用 GT intersection
选择含动作的 crop。

对 video `v`：

1. 按固定 dataset config 的 `snippet_stride`、`scale_factor`、rounding 和 clipping
   规则构造 source sampled-index vector `I_v`；
2. 令 `n_v = len(I_v)`、`W = 768`；
3. 计算：
   `d_v = SHA256("CT-P3R-3S-r1-window-v1\0" + video_id + "\0" +
                  media_sha256 + "\0" + str(n_v))`；
4. 若 `n_v <= W`，`start_v=0`；
5. 否则：
   `start_v = uint64_big_endian(d_v[0:8]) mod (n_v-W+1)`；
6. 取 `I_v[start_v:start_v+W]`；
7. 少于 W 时使用 official edge-padding 语义补齐，并生成 exact valid mask。

annotation bytes、GT count、class、segment duration 和 detector output 均不得进入
`d_v` 或 `start_v`。annotation SHA-256 只作为数据 identity 记录。

one-window manifest 对每个 video 必须记录：

- video ID；
- media path/registry ID 与 media SHA-256；
- source total frames、fps、snippet stride、scale factor；
- source sampled-index vector length；
- window start；
- 全部 768 sampled frame indices；
- padding positions 与 valid mask；
- data/config/annotation SHA-256；
- canonical per-window payload SHA-256。

必须断言：

- 恰有 200 个 unique video IDs；
- 恰有 200 个 unique window IDs；
- video ID 与 window ID 一一对应；
- 三个 split 分别恰有 140/30/30 个 windows；
- 三个训练 seed、全部 schedules 和全部 paired branches 使用同一 temporal window。

Stage B 固定 `batch_size=1`、`world_size=1`、`shuffle=false`。fit 允许由 run seed
控制的空间/颜色 augmentation，但同一 paired dense/counterfactual forward 必须共享已
materialize 的像素 tensor。calibration/evaluation 使用确定性 spatial transform。

Gate 1–3 的 conformal、Spearman、coverage 和 bootstrap outer unit 均为上述 unique
manifested window。

Gate 4 是不同的 official full-video/sliding-window population。Gate 3 在 frozen-window
population 上的 conformal guarantee、coverage 或 safety statement不得自动转移到 Gate 4。
```

**RATIONALE:** Option B 会把 140-update协议扩张为所有 sliding windows，并迫使对 candidate×windows 取 max，不再是 bounded minimal appeal。

**REQUIRED TEST:** TDD 12–14、16、17、27。

---

### Amendment 3 — Motion 与 random controls

**SECTION:** 替换 §6 中 `motion_p{2,4,8}` 和 `random_p{2,4,8}` 定义。

**PROBLEM:** global threshold 不能逐窗口匹配 exact RECOMPUTE count。

**VERDICT:** `REVISE`.

**WHY:** requested count 与 executed count 不同会同时破坏成本公平性和 baseline 名称。

**FALSE_POSITIVE_RISK:** 高运动窗口比 periodic 使用更多 RECOMPUTE，取得不公平质量优势。

**FALSE_NEGATIVE_RISK:** 低运动窗口几乎不重算，baseline 被人为削弱。

**MINIMAL_AMENDMENT:** 使用 group-wise top-k，并改名。

**DELETE/REPLACE:** 删除 motion threshold/quantile calibration 文本。

**EXACT NEW MARKDOWN:**

```markdown
非学习 controls 固定为：

- `motion_topk_p{2,4,8}`：
  只使用 deploy-visible cosine-change signal。对每个 window、每个 layer group，
  从对应 frozen periodic action matrix 读取该 group 的 exact RECOMPUTE count `K_p`。
  clip 0 必须 RECOMPUTE；在 clips 1–47 中按 cosine change 从高到低排序，选择
  `K_p-1` 个 positions RECOMPUTE，其余全部 HOLD。finite ties 按较小 clip index
  优先。finite constant-motion window 是合法输入，并按该 tie rule 确定结果。
  该 baseline 不再称为 motion-threshold，也不拟合 calibration threshold。

- `random_p{2,4,8}`：
  对每个 `(window_id, seed, group, period, clip)` 计算 canonical SHA-256。clip 0
  强制 RECOMPUTE；clips 1–47 按 digest ascending 排序，选择 `K_p-1` 个 positions
  RECOMPUTE，其余 HOLD。

motion/random 的 requested action matrix 必须与 periodic comparator 在每个
`window × group` 上具有完全相同的 RECOMPUTE count。成本使用 requested/executed
action hash 对应的实测分布，不得借用 periodic 名称的 cost。

任何 non-finite motion signal 必须触发 dense safety fallback，并使该 formal comparator
sample/run 标记为 `INVALID_IMPLEMENTATION`；不得以 repaired motion schedule 进入 Gate 1。

`motion_topk` 与 `random` 都是 Gate 1 的 hard comparators，不是仅作图表的 diagnostics。
```

**RATIONALE:** top-k 是唯一同时满足 deploy-visible、逐窗口 exact count 和无 calibration label 的定义。

**REQUIRED TEST:** TDD 9、23、24。

---

### Amendment 4 — Block-rotated Stage-B exposure

**SECTION:** 替换 §7 的 schedule assignment 段落。

**PROBLEM:** 当前 offset cycle 与 video canonical position 有 mod-4 confounding。

**VERDICT:** `REVISE`.

**WHY:** 训练结果可能反映 video-order subgroup，而不是 candidate effect。

**FALSE_POSITIVE_RISK:** 某 candidate 恰好绑定到更容易的 video quartile。

**FALSE_NEGATIVE_RISK:** 某 candidate 恰好绑定到更困难的 quartile。

**MINIMAL_AMENDMENT:** 采用固定 `+5*b` block rotation，不增加 steps/seeds/candidates。

**DELETE/REPLACE:** 删除 `(update + seed_offset) mod 16` 段落。

**EXACT NEW MARKDOWN:**

```markdown
### 7.x Candidate × video exposure

令 `j` 为 canonical fit-window index，`j in [0,139]`：

- `b = floor(j / 16)`；
- `p = j mod 16`；
- `candidate = (p + 5*b + seed_offset) mod 16`。

seed offsets 固定：

- seed 3407：0；
- seed 3408：4；
- seed 3409：8。

该公式不可替换为 hash permutation、Latin-square 搜索或结果后重新排布。

正式 validator 必须断言：

1. 每个完整 16-window block 内，candidate 0–15 各出现一次；
2. 每 seed：12 个 candidates 暴露 9 次、4 个暴露 8 次；
3. seed 3407 的 8-exposure candidates 恰为 4–7；
4. seed 3408 的 8-exposure candidates 恰为 8–11；
5. seed 3409 的 8-exposure candidates 恰为 12–15；
6. 三 seed 汇总：candidate 0–3 各 27 次，candidate 4–15 各 26 次；
7. 在前 128 个完整-block windows 中，每 candidate、每 seed 在四种
   `p mod 4` 上各出现 2 次；三 seed 汇总后各出现 6 次；
8. 每个 fit window 跨三 seed 获得三个不同 candidates；
9. tail candidate 顺序必须精确为：
   - seed 3407：`[8,9,10,11,12,13,14,15,0,1,2,3]`；
   - seed 3408：`[12,13,14,15,0,1,2,3,4,5,6,7]`；
   - seed 3409：`[0,1,2,3,4,5,6,7,8,9,10,11]`。

artifact 必须保存 canonical fit-window order hash、每 seed 的完整 140-row exposure
matrix、三个 per-seed hashes 和 combined matrix SHA-256。任一 mismatch 为
`INVALID_IMPLEMENTATION`。

固定 140-update protocol 的负结果最多否定本 head/library/exposure/budget 组合，不得写成
“所有 transport 或 dynamic refresh ideas 均失败”。
```

**RATIONALE:** 它消除固定 mod-4 confounding，同时冻结 tail 不平衡，不创造额外训练量。

**REQUIRED TEST:** TDD 10、11、22、27。

---

### Amendment 5 — Replace Gate 1

**SECTION:** 整体替换 §9。

**PROBLEM:** oracle-shuffle 近乎定义成立；evaluation-best 的角色不一致；bootstrap 未重做 selection。

**VERDICT:** `REVISE`.

**WHY:** 当前 Gate 可能在没有任何输入依赖结构时通过。

**FALSE_POSITIVE_RISK:** 把逐窗口 minimum 的定义优势写成科学 headroom。

**FALSE_NEGATIVE_RISK:** evaluation-selected comparator 不重选时低估 selection uncertainty。

**MINIMAL_AMENDMENT:** 删除 shuffle，允许 evaluation-only adjudicator，并在 replicate 内重选。

**DELETE/REPLACE:** 删除现有 §9 全文。

**EXACT NEW MARKDOWN:**

```markdown
## 9. Gate 1：冻结库的 equal-cost oracle headroom

Gate 1 在任何新 Stage-B seed 训练前运行。它只支持
`oracle_headroom=true/false`，不支持 deploy-visible input dependence。

首先以每个 candidate 自身的 registered full-stack p50 判断
`p50(candidate) <= B*`，冻结 cost-feasible HOLD schedule set。不得借用 P4 cost、
action-count proxy 或 stage-percentile sum。

Comparator 分为：

1. `calibration-frozen global static`：
   在 calibration windows 上选择 mean regret 最低且 cost-feasible 的单一 HOLD
   schedule；identity 在 evaluation 前冻结；
2. `motion_topk` 与 `random`：
   使用第 6 节的 frozen exact-count controls；
3. `time-only oracle`：
   每 evaluation window 在 P2/P4/P8 HOLD 中取 minimum；
4. `layer-only oracle`：
   每 evaluation window在 early/late HOLD 中取 minimum；
5. `evaluation-best global static`：
   在 evaluation windows 上选择 mean regret 最低的单一 cost-feasible HOLD schedule；
   它是 evaluation-only adjudication comparator；
6. `joint oracle`：
   每 evaluation window 在全部 cost-feasible HOLD schedules 中取 minimum。

`evaluation-best global static` 可以参与 Gate 1 pass/fail，但不得进入 deployment、
checkpoint、threshold、candidate library、Stage B/C training、calibration 或后续配置。

删除全部 oracle-assignment shuffle hard conditions。schedule diversity、selection entropy
和被选 schedule 数量只作 diagnostics。

`strongest_comparator` 是上述 1–5 中 evaluation mean regret 最低者。若其 mean regret
小于等于 `1e-12`，relative-reduction criterion 未定义，Gate 1 直接 FAIL。

Gate 1 同时要求：

1. joint oracle 相对 strongest comparator 的 full-sample mean detector-regret
   relative reduction `>=10%`；
2. 5000 次 paired unique-window bootstrap 中，
   `regret_strongest_comparator - regret_joint_oracle`
   的 percentile 95% CI lower `>0`；
3. `B*` 相对 dense measured full-stack p50 saving `>=20%`。

每个 bootstrap replicate 必须：

1. resample complete evaluation windows；
2. 在 replicate 内重新选择 evaluation-best global static；
3. 在 replicate 内重新确定 strongest comparator；
4. 重新计算 per-window joint/time/layer oracles；
5. 保持每个 window 的完整 candidate vector。

candidate-set size、feasible names 与各 oracle 的 set size 必须报告。Gate 1 PASS
只允许打开 `oracle_headroom=true`；input dependence 只能由 Gate 3 裁决。
```

**RATIONALE:** 该方案允许最强保守 adjudicator，但把其物理隔离在 evaluation report 中。

**REQUIRED TEST:** Gate-1 adjudicator selection-resampling test；TDD 14、23、27、29。

---

### Amendment 6 — Gate 3 support、baseline 与统计单位

**SECTION:** 替换 §11.2–§11.4，并补充 §12.1。

**PROBLEM:** support 可集中于少数窗口；constant baseline 的 fit targets 数不明确；coverage uncertainty 容易伪独立。

**VERDICT:** `REVISE`.

**WHY:** 极保守、几乎总 dense 的系统可能满足 coverage，却没有可用选择行为。

**FALSE_POSITIVE_RISK:** 重复选择同六个窗口伪造 20% pooled support。

**FALSE_NEGATIVE_RISK:** 在 n=18 时用 CI lower≥0.85 杀死 18/18 覆盖。

**MINIMAL_AMENDMENT:** 加 unique support、all-fit replay baseline、window-outer bootstrap；point coverage 保留。

**DELETE/REPLACE:** 替换现有 Gate 3 ranking/support/baseline 条款。

**EXACT NEW MARKDOWN:**

```markdown
### 11.2 Fit-only schedule-conditioned constant baseline

每个 Stage-B seed 完成 140 successful updates 后，在打开 calibration/evaluation 前，
对全部 140 fit manifested windows no-grad replay 全部 16 non-dense schedules。

对每个 schedule `s`，使用其 140 个 fit regret targets 的 finite-sample tau=.9
order statistic：

`rank = ceil((140+1)*0.9) = 127`。

得到 schedule-conditioned constant prediction `q_const_s`。baseline payload、完整
fit replay key set 和 SHA-256 必须在 calibration 前冻结。不得只使用该 schedule 的
8–9 个 optimizer-exposure targets。

### 11.3 Ranking 与 bootstrap unit

对每个 `seed × evaluation window`，在完整 16-candidate vector 内计算 Spearman rho。
predicted vector 或 regret vector 少于 3 个 distinct ranks 时，该 seed-window
fail closed。

每 seed score 是其 30 个 window rhos 的 arithmetic mean。pooled score 对所有
`seed × window` 等权。

所有 Gate 2/3 bootstrap：

- outer resample unit：unique manifested window；
- inner resample unit：三个 seeds；
- candidate/period vector 必须随 window 整体移动；
- candidate rows 不得成为 bootstrap samples；
- bootstrap replicates 固定为 5000，seed 固定 `20260711`。

### 11.4 Selected support 与 coverage

evaluation 上同时要求：

1. 每 seed 至少 `6/30` windows 选择 non-dense；
2. pooled selected `seed × window` count 至少 18；
3. 至少 10 个 distinct evaluation windows 被任一 seed 选择；
4. pooled selected non-dense point coverage `>=0.85`；
5. 每 seed mean rho `>=0`；
6. 三个 seed means 的 median `>=0.2`；
7. pooled rho hierarchical-bootstrap 95% CI lower `>0`；
8. evaluation pinball loss 相对第 11.2 节 constant baseline 至少降低 10%。

coverage denominator 只包含实际 selected non-dense seed-window rows。dense fallback
不得进入 coverage numerator 或 denominator。

同时报告：

- per-seed selected coverage；
- pooled selected coverage；
- selected unique-window count；
- all-window simultaneous coverage；
- 对每个 selected window，
  `min_seed_selected(upper-regret)`；
- `window_all_selected_covered =
   1[min_seed_selected(upper-regret) >= 0]` 的比例；
- window-clustered one-sided 95% coverage LCB。

coverage CI lower 不作为 `>=0.85` hard gate。coverage `>0.95` 只标记
`OVERCOVERED`，不直接失败。

该 selected coverage 是 empirical diagnostic/gate，不得描述为 split-conformal
selected-conditional theoretical guarantee。

所有 Gate 3 hard conditions 构成 intersection-union gate，不作额外 multiplicity
correction。baseline pinball mean 小于等于 `1e-12` 时，10% relative improvement
未定义，Gate 3 FAIL。
```

**RATIONALE:** 不增加 evaluation windows、seeds、quantile 或 epsilon。

**REQUIRED TEST:** TDD 14–17、29。

---

### Amendment 7 — Stage-C gradient、exposure 与 AMP retry

**SECTION:** 替换 §13 中 Stage-C optimizer、training 与 matched-control 段落。

**PROBLEM:** loss ownership、batch-level candidate assignment、overflow state 和 successful exposure 没有可执行定义。

**VERDICT:** `REVISE`.

**WHY:** 常规 `.backward(total_loss)` 无法实现 LF→T 但 LF↛A；不同 overflow 会导致 LR/EMA drift。

**FALSE_POSITIVE_RISK:** auxiliary LF 实际更新 adapters，CT arm获得额外 objective。

**FALSE_NEGATIVE_RISK:** 过度 detach adapter，切断 LF→T 的有效 Jacobian。

**MINIMAL_AMENDMENT:** world1 plain module、object-identity `autograd.grad`、successful-update scheduler、retry rollback。

**DELETE/REPLACE:** 替换现有 Stage-C training 及 matched exposure 段落。

**EXACT NEW MARKDOWN:**

```markdown
### 13.1 Stage-C fixed execution semantics

Stage C 固定：

- `world_size=1`；
- plain single-process module，不使用 DDP、FSDP 或 `no_sync`；
- global batch size=2；
- no gradient accumulation；
- `drop_last=false`；
- 140 fit windows；
- 70 successful optimizer updates/epoch；
- 60 epochs；
- 4200 successful optimizer updates；
- AMP FP16；
- EMA decay=0.999；
- global clip grad norm=1.0。

LR scheduler 保持 fixed OpenTAD semantics：

- `LinearWarmupCosineAnnealingLR`；
- warmup steps=`5×70=350`；
- max scheduler steps=`100×70=7000`；
- training 在 successful update 4200 后停止；
- scheduler 与 EMA 只在 successful optimizer update 后各推进一次。

### 13.2 Stage-C per-window candidate exposure

令 successful update index `u in [0,4199]`，batch 内 canonical position
`r in {0,1}`，window-exposure ordinal：

`e = 2*u + r`。

令：

- `b = floor(e/16)`；
- `p = e mod 16`；
- `candidate = (p + 5*b + seed_offset) mod 16`。

三个 seed offsets 仍为 0/4/8。每 seed 恰有 8400 window exposures，因此每个
candidate 恰好出现 525 次。

CT arm 可在同一 batch 内为两个 examples 执行不同 action matrices。matched-dense
arm 保存完全相同的 shadow candidate assignment ledger，但实际执行 forced dense。
两个 arms 必须共享 ordered materialized batch/augmentation hashes。

### 13.3 Object-identity parameter ownership

以 Python object identity 构造、排序并冻结三个 parameter tuples：

- `A`：全部 AdaTAD adapter parameters；
- `T`：全部 ChronoTransport transport parameters；
- `R`：canonical risk predictor parameters。

必须断言：

- `A`、`T`、`R` 两两不交；
- 三者 union 恰好等于全部 `requires_grad=True` parameters；
- optimizer 中每个 parameter object 恰好出现一次；
- heavy VideoMAE、projection、head 和其他参数均 `requires_grad=False` 且不在 optimizer；
- `risk_predictor` 与 `scheduler.predictor` alias 只以 canonical object 进入 R 一次。

禁止 Stage C 使用 generic name-substring optimizer grouping。

### 13.4 Loss-specific AMP gradient algorithm

同一 counterfactual forward 产生：

- `LD`：detector task loss；
- `LF`：feature consistency loss；
- `LR`：pinball risk loss。

`LR` 的 deploy-visible signals 与 regret target 对 A/T detach；scheduler argmin 不反传。

每个 attempt 必须：

1. `optimizer.zero_grad(set_to_none=True)`；
2. 读取当前 GradScaler scale `S`；
3. 计算：
   - `gD = autograd.grad(scaler.scale(LD), A+T,
                          retain_graph=True, allow_unused=True)`；
   - `gF = autograd.grad(scaler.scale(0.1*LF), T,
                          retain_graph=True, allow_unused=True)`；
   - `gR = autograd.grad(scaler.scale(0.1*LR), R,
                          retain_graph=False, allow_unused=True)`；
4. 在 scaler update 前断言三次 `scale()` 使用的 scale 均为 S；
5. 写入 scaled gradients：
   - `A.grad = gD[A]`；
   - `T.grad = gD[T] + gF[T]`，None 按零处理；
   - `R.grad = gR[R]`；
6. 恰好调用一次 `scaler.unscale_(optimizer)`；
7. 进行 finite 与 expected-unused audit；
8. finite 时对 `A+T+R` 执行 global clip norm=1.0；
9. 恰好调用一次 `scaler.step(optimizer)` 和一次 `scaler.update()`。

expected-unused 规则：

- A 的 aggregate detector gradient 必须 finite 且 nonzero；
- R 的 aggregate pinball gradient 必须 finite 且 nonzero；
- executed batch 中没有 TRANSPORT cell 时，T gradient 可以全部 None/0；
- executed batch 中存在 TRANSPORT cell 时，该 successful exposure 的 T aggregate
  gradient 必须 finite；全部 TRANSPORT exposures 汇总 T norm 必须大于 0。

LF forward 可以使用当前 trainable adapter parameters 计算 input Jacobian，但由于 A 不在
该次 `autograd.grad` inputs 中，LF 不得写 A.grad。

### 13.5 AMP overflow retry

每个 batch 在首次 forward 前必须：

- materialize并 hash batch 与 augmentation；
- snapshot Python/NumPy/Torch/CUDA RNG；
- snapshot全部 forward-mutated model buffers、Python state、optimizer、EMA、scheduler、
  CT diagnostics 与 profiler state。

overflow 时：

1. 调用 `scaler.step(optimizer)`，并验证 optimizer 未改变；
2. 调用 `scaler.update()`，保留 scaler backoff；
3. 不推进 sampler、batch cursor、successful-update index、candidate exposure index、
   LR scheduler或 EMA；
4. 清空 gradients；
5. 恢复 pre-forward RNG；
6. bitwise 恢复除 GradScaler 外的全部 snapshot state；
7. retry 同一 materialized batch。

必须恢复的状态包括 BN running statistics、`AnchorFreeHead.loss_normalizer`、
checkpoint/dropout RNG、CT `latest_*`、cache/action history、profiler buffers、
Python lists/counters、optimizer、scheduler 和 EMA。模型内部 diagnostic state 不得豁免；
只有模型外部 append-only retry audit log 可以保留。

初始 attempt 后最多允许 3 次 retry。第 4 个 overflow attempt 仍失败时，run 标记
`INVALID_IMPLEMENTATION`，不是 science FAIL。

ChronoTransport arm 与 matched-dense arm 可以拥有不同 GradScaler/overflow histories。
matched exposure 的必要条件是：

- 两 arm successful batch hashes 完全相同且顺序相同；
- 两 arm各有 4200 次 common-adapter updates；
- common-A LR trace 与 EMA update count完全相同；
- dense arm和CT arm的 attempted/retry/scaler traces分别完整报告。

不得采用 lockstep mutual skip。
```

**RATIONALE:** 这是唯一无需 hooks、functional clone 或额外模型副本即可满足三种 loss ownership 的算法。

**REQUIRED TEST:** TDD 18–22、30。

---

### Amendment 8 — Gate 4 timing 与 inference

**SECTION:** 替换 §13 中 Gate 4/P5 条件。

**PROBLEM:** 当前 point thresholds、ratio-of-medians 和非聚类 mAP 不足以支持 metric/latency unlock。

**VERDICT:** `REVISE`.

**WHY:** overlapping sliding windows、重复 timing invocation 和三 seeds 均不是独立 label units。

**FALSE_POSITIVE_RISK:** 热漂移或伪重复使 p50 saving/NI CI 过窄。

**FALSE_NEGATIVE_RISK:** 不平衡顺序使 scheduler 总在更热的时段运行。

**MINIMAL_AMENDMENT:** balanced six-order crossover、video-cluster bootstrap、单侧 CI、paired margin。

**DELETE/REPLACE:** 替换现有 Gate 4 primary conditions 与 profiler order。

**EXACT NEW MARKDOWN:**

```markdown
### 13.x Gate-4 population and matched timing

Gate 4 使用 official full-video/sliding-window evaluation population。全部 overlapping
windows 归属于其 official video ID。

每个 arm/seed 的 timed invocation set 必须：

- 包含 official evaluation 中使用的全部 invocation IDs；
- 每 arm/seed 至少 200 个 invocations；
- 总数为 6 的整数倍；
- 若需附加 repetitions，只能按预注册 invocation hash 顺序补齐；
- exact invocation list、repetition IDs 与 order SHA-256 在 timing 前冻结。

primary timing arms：

- D：matched dense；
- C：calibrated ChronoTransport；
- S：calibration-frozen global static。

使用以下六序列循环：

1. D-C-S；
2. C-S-D；
3. S-D-C；
4. S-C-D；
5. C-D-S；
6. D-S-C。

每个 arm 独立执行 decode、preprocess、H2D、model 与 postprocess，不得共享已 decode
tensor或模型中间缓存。每次 invocation 在 timing 边界前后 CUDA synchronize，并保存完整
`total_ms`。stage durations 只作 diagnostics。

### 13.x Gate-4 bootstrap

latency bootstrap：

1. outer resample official video IDs；
2. 在每个 sampled video 内 resample complete matched invocation blocks；
3. inner resample三 seeds；
4. 每个 replicate 从 raw total samples 重新计算各 arm p50；
5. 不得 bootstrap stage percentiles 后相加。

mAP bootstrap：

1. outer resample official video IDs；
2. inner resample三 seeds；
3. 每个 duplicated video copy使用新的 synthetic ID，防止不同 bootstrap copies 之间
   错误 NMS/aggregation；
4. 使用该 replicate 的完整 predictions/GT 重新计算 mAP；
5. 重复 timing invocations不得成为额外 mAP samples。

bootstrap replicates=5000，seed=`20260711`。

### 13.x Gate-4 hard conditions

定义：

`latency_saving = (p50_dense - p50_CT) / p50_dense`。

Gate 4 同时要求：

1. latency_saving 的 one-sided 95% LCB `>=0.15`；
2. `mAP@0.7_dense - mAP@0.7_CT` 的 one-sided 95% UCB `<=1.5`；
3. fit-Q1 shortest-duration subset 上同一 mAP drop 的 one-sided 95% UCB `<=1.5`；
4. 对每个 matched invocation：
   - `heavy_saving_i = dense_heavy_i - selected_heavy_i`；
   - `overhead_i = innovation_i + scheduler_i + transport_i + cache_movement_i`；
   - `margin_i = 0.40*heavy_saving_i - overhead_i`；
   要求 full-sample median heavy_saving `>0`，且 median margin 的 bootstrap
   one-sided 95% LCB `>0`；
5. `p50_CT - p50_static` 的 one-sided 95% UCB `<=0`；
6. CT 相对 calibration-frozen static 的 evaluation detector-regret absolute improvement
   hierarchical-bootstrap 95% CI lower `>0`；
7. 每 seed 的 point latency saving、mAP@0.7 drop、shortest-Q1 drop、median margin 和
   CT-static latency difference 均不得越过对应失败阈值。

calibration-frozen static identity 在 evaluation 与 bootstrap 中保持固定。任何
evaluation-selected diagnostic comparator 必须在每个 bootstrap replicate 内重新选择，
但不得替换 hard static comparator。

上述 hard conditions 构成 intersection-union gate，不作额外 multiplicity correction。
p95、throughput、peak memory、stage breakdown、raw total distributions 和 block-level
NVML energy 同时报告，但不替代 primary gates。
```

**RATIONALE:** latency 和 mAP 使用不同的合法 sampling hierarchy；重复 timing 不得伪造检测样本量。

**REQUIRED TEST:** TDD 25、26；Gate4 synthetic bootstrap tests。

---

### Amendment 9 — Immutable pre-Gate1 registration

**SECTION:** 在 §14 前插入新 §14.1。

**PROBLEM:** checkpoint sidecar 太晚，不能阻止 profile/replay 前替换 identity。

**VERDICT:** `REVISE`.

**WHY:** checkpoint、data、window 或 config 可在看到 profile 后漂移。

**FALSE_POSITIVE_RISK:** 选择更有利的 checkpoint/profile 环境后仍声称预注册。

**FALSE_NEGATIVE_RISK:** 把普通 infrastructure resume 误当作协议漂移。

**MINIMAL_AMENDMENT:** 选择独立 registration artifact 方案 B。

**DELETE/REPLACE:** 不删除现有 checkpoint 条款；在其前插入以下文本。

**EXACT NEW MARKDOWN:**

```markdown
## 14.1 Immutable pre-Gate1 registration

选择独立 immutable registration artifact。不得仅把 identity 写入后续 checkpoint。

流程固定：

1. 先产生 clean implementation commit `I`，其中包含最终 spec、实现、tests、configs、
   launchers和 adjudicators，但不包含任何 profile/replay/evaluation result；
2. 在 detached clean worktree at `I` 生成 canonical
   `chronotransport_pre_gate1_registration.json`；
3. registration generator 不得接收或读取 profile、replay、calibration、evaluation、
   Gate report 或实验 output path；
4. 将该 JSON 作为唯一内容变化提交为 registration commit `R`；
5. formal launcher 的 Git HEAD 必须精确等于 `R`；
6. unlock chain 同时记录 `I`、`R` 和 registration file SHA-256。

registration 至少包含：

- protocol ID；
- spec commit、spec exact-byte SHA-256；
- implementation commit I；
- registration parent/tree identity；
- all source/test/config/launcher file hashes；
- OpenTAD/AdaTAD/VideoMAE upstream commits；
- dense checkpoint registry ID、authenticated URI/local content-addressed path、byte size、
  SHA-256；
- data root identity、annotation SHA-256、每个 media SHA-256；
- 200-video split manifest与 one-window manifest hashes；
- canonical 16-candidate library、全部 action hashes；
- Stage-B per-seed candidate×window matrices与 hashes；
- Stage-C candidate exposure formula；
- motion/random exact algorithms与 hashes；
- bootstrap seeds、profiler invocation/order hashes；
- all fixed Gate thresholds；
- expected GPU model/UUID、driver、CUDA、PyTorch、cuDNN、precision、environment/container
  hashes；
- output root；
- attestation that no profile/replay/evaluation result was read.

若 checkpoint 只存在远端，必须在 registration 前从 authenticated registry 读取一次，
计算完整文件 SHA-256，并复制到 content-addressed path；remote path/mtime 不能替代
content hash。

formal launch要求：

- `git status --porcelain` 为空；
- HEAD=`R`；
- source/spec/config/checkpoint/data/window/library/exposure hashes全部重新验证；
- symlink-resolved output root 在允许根目录内。

不强制 GPG/signature。公开 immutable Git commit R、registration exact-byte SHA-256 和
content-addressed inputs 是本协议的必要完整性证据；签名可附加但不得成为结果后新增门槛。

profile 完成后生成独立 profile artifact hash，并把它链接到 registration R；不得修改原
registration。任何 code/spec/data/checkpoint 变化均要求新的 I/R，旧结果不得迁移。

resume 必须重新验证 registration、profile input chain、checkpoint、optimizer、GradScaler、
EMA、scheduler、successful exposure ledger prefix 与 next cursor。任一 mismatch 为
`INVALID_IMPLEMENTATION`。
```

**RATIONALE:** 两 commit 结构避免 registration artifact 需要包含自身 commit SHA 的循环依赖。

**REQUIRED TEST:** TDD 27、30–32。

---

### Amendment 10 — Header、claim unlock 与 completion

**SECTION:** 文件头、§14 claim flags、§16 completion。

**PROBLEM:** 当前文件头与 review status 冲突；claim 名称没有编码 population/scope；deploy/paper 后置要求不完整。

**VERDICT:** `REVISE`.

**WHY:** 即使 Gate 4 PASS，也不能自动获得部署安全或论文新颖性结论。

**FALSE_POSITIVE_RISK:** 固定 GPU/detector 结果被扩张为通用 deployment 或论文 claim。

**FALSE_NEGATIVE_RISK:** 无；该修订只限制表述，不改变实验行为。

**MINIMAL_AMENDMENT:** 精确重命名 flags，并永久冻结 deploy/paper=false。

**DELETE/REPLACE:** 替换文件头的决策状态、§14 claim flags 段和 §16 下一步。

**EXACT NEW MARKDOWN:**

```markdown
决策状态：`REVISE_SPEC_BEFORE_PLAN`。只有本轮全部 patch-ready amendments 写入新的
spec commit、生成 exact-byte SHA-256，并通过一次 spec-only diff review 后，状态才可改为
`APPROVE_SPEC_FOR_PLAN`。

claim flags 初始全部 false：

- `oracle_headroom`；
- `mechanism`；
- `calibrated_risk_on_frozen_window_protocol`；
- `metric_adatad_thumos14_official_full_video`；
- `latency_gpu1_fixed_stack`；
- `deploy`；
- `paper`。

unlock chain 固定：

1. Gate 1 PASS 只允许 `oracle_headroom=true`；
2. Gate 2 PASS 只允许 `mechanism=true`；
3. Gate 3 PASS 只允许
   `calibrated_risk_on_frozen_window_protocol=true`；
4. Gate 4 全 PASS 后，只允许：
   - `metric_adatad_thumos14_official_full_video=true`；
   - `latency_gpu1_fixed_stack=true`。

即使 Gate 1–4 全 PASS：

- `deploy=false`；
- `paper=false`。

`deploy=true` 至少还需要：

- official full-video population 上独立的 calibration/safety protocol；
- 与本 bounded appeal 不共享 evaluation selection 的新验证；
- deployment hardware/traffic/failure-mode 审查。

`paper=true` 至少还需要：

- 外部 novelty/collision review；
- paper-level claims-evidence review；
- 若主张 detector-agnostic plugin 泛化，至少一个第二 detector 的独立验证。

不得把 Gate 4 PASS 扩张为首次提出 time×depth routing、三动作 cache、MoD、
conformal compute control、通用 TAD deployment 或 detector-agnostic generalization。

## 16. 完成定义

当前规格不得进入 writing-plans。唯一下一步是：

1. 将本轮全部 amendments 原样写入新的规格；
2. 生成新的 spec Git commit；
3. 独立复算 exact-byte SHA-256；
4. 执行一次只比较规格文本、不读取结果的 spec-only review；
5. 仅在该 review 输出 `APPROVE_SPEC_FOR_PLAN` 后，才允许调用 writing-plans。

上述步骤完成前，implementation、profiling、Gate 1、新 seed 与 Stage C/P5 全部锁定。
```

**RATIONALE:** 精确限制每个 Gate 能支持的最窄 claim。

**REQUIRED TEST:** TDD 29、30。

---

## 15. Post-Amendment Verdict

`PROPOSAL`：若第 14 节所有 amendments 被完整、无变体地写入一个新 spec commit，并满足：

* exact-byte SHA-256 独立复算；
* 新 commit 与修订文本逐段一致；
* 没有重新引入 choices、optional branches 或结果后自由度；
* spec-only reviewer 确认内部无冲突；

则新规格可以被裁决为：

> **`APPROVE_SPEC_FOR_PLAN`**

仍需要一次 spec-only 复核，但不需要重新进行完整文献或源代码审查。

以下项目必须永久冻结：

* pre-adapter cache、all-row TIA、detach 语义；
* age 47/cap 8；
* window head 与 8/8 embeddings；
* one-window manifest 规则；
* 16-candidate library；
* Stage-B/Stage-C exposure formula；
* seeds、140 Stage-B updates、4200 Stage-C updates；
* quantile、epsilon、losses；
* Gate thresholds；
* bootstrap units/seeds；
* timing order；
* registration chain；
* claim unlock；
* Gate FAIL 后不再上诉。

当前终点是：

> 完成 patch-ready spec amendments，等待作者写入新规格并生成新的 spec commit 与 SHA。

不得越过到 implementation plan。

---

## 16. Results-to-Claims / Kill Matrix

| 有效结果组合                                           | 允许表述                                                             | 禁止表述                                                              | 后续状态                                                     |
| ------------------------------------------------ | ---------------------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------- |
| 任一 identity/test/profiler/retry violation        | 无科学结论；只能写 `INVALID_IMPLEMENTATION`                               | PASS、FAIL、趋势、negative result                                      | 修复实现后从受影响 Gate 完整重跑；旧 run 永久 invalid                     |
| Gate 1 FAIL                                      | `oracle_headroom=false` under frozen library/B*                  | transport无效、risk无效、全部动态计算无效                                       | 永久冻结 bounded appeal；Gate 2–4 不运行                         |
| Gate 1 PASS，Gate 2 FAIL                          | frozen library 有 oracle headroom；`mechanism=false`               | TRANSPORT 优于 HOLD、calibrated scheduler、speed-quality claim        | 永久冻结                                                     |
| Gate 1–2 PASS，Gate 3 FAIL                        | oracle headroom；matched TRANSPORT mechanism evidence             | deploy-visible ranking、calibrated scheduler、metric/latency        | 永久冻结                                                     |
| Gate 1–3 PASS，Gate 4 FAIL                        | oracle headroom；mechanism；frozen-window calibrated-risk evidence | official full-video metric、GPU1 latency、deployment、paper          | 永久冻结                                                     |
| Gate 1–4 PASS                                    | 仅可打开五个固定 flags 中前五个；固定 AdaTAD/THUMOS14/GPU1 metric/latency       | deploy、paper、generalization、novelty、第二 detector、通用 safety         | 停止 bounded appeal；进入独立 deployment/novelty review，而不是继续调参 |
| Gate 3 frozen-window coverage PASS、Gate 4 未运行    | frozen-window simultaneous marginal calibration                  | official sliding-window coverage 或 selected-conditional guarantee | Gate 4 仍锁定或等待合法执行                                        |
| 单 seed PASS、pooled/CI FAIL                       | 单 seed diagnostic                                                | 多 seed scientific claim                                           | 按 Gate FAIL 处理                                           |
| point threshold PASS、CI FAIL                     | point estimate diagnostic                                        | unlock claim                                                      | Gate FAIL                                                |
| Gate metric FAIL 后更改 seed/head/library/threshold | 无                                                                | “修复后再试”                                                           | 明确违反 bounded appeal，整条 appeal rejected                   |

---

## 17. Strongest Objections

### 1. Frozen one-window population 过于人工

**反对意见：** Gate 1–3 可能只在每 video 一个 hash crop 上成立，不能代表 official sliding-window TAD。

**解除证据：**

* label-free immutable manifest；
* claim 严格写成 frozen-window protocol；
* Gate 4 使用 official population；
* deployment 前另做 independent full-video calibration/safety。

### 2. 140 updates 对窗口级 risk head 太少

**反对意见：** 每 candidate 仅 8–9 次直接 exposure，负结果可能只是 undertraining。

**解除证据：**

* block-rotated exposure integrity；
* shared cell/action representation；
* all-140×16 fit replay diagnostics；
* 将负结论严格限制在 fixed 140-update protocol。

本 bounded appeal 不允许通过增加训练量解除该反对意见。

### 3. Gate 1 仍受 candidate-set-size advantage 驱动

**反对意见：** joint oracle 比 time/layer subsets 强，可能只是因为集合更大。

**解除证据：**

* 明确只称 oracle headroom；
* strongest comparator 包含 time/layer per-window oracles 和 evaluation-best static；
* 报告 candidate-set sizes；
* input dependence 只交给 Gate 3。

### 4. Stage-C autograd/retry 极易悄然偏离

**反对意见：** 手写 scaled grads、alias、mutable buffers 和 retry 很脆弱。

**解除证据：**

* object-identity coverage；
* exact zero/nonzero ownership tests；
* overflow induced test；
* 除 scaler 外的 bitwise state restoration；
* 4200 successful batch hash ledger；
* matched A LR/EMA traces。

### 5. Gate 4 timing 可能主要测到 decode、热漂移或 cache order

**反对意见：** 15% saving 可能来自 arm order、文件缓存或 GPU thermal state。

**解除证据：**

* independent decode；
* six-order balanced crossover；
* exact matched invocation hash；
* raw total_ms；
* video-cluster bootstrap；
* registered clocks/UUID/software；
* throttle/concurrent-process invalidation；
* paired overhead margin。

---

## 18. Final Go/No-Go

`GO_TO_SPEC_REVISION_ONLY`

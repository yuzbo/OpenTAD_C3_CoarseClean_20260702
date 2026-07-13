# 1. Executive Verdict

**`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。** 固定提交可见，规范 Git blob 与批准提交一致；但当前代码不得生成 `I/R`。七项既有 blocker 全部成立，另发现两项注册级缺陷：r2 配置覆盖层级错误，以及 Gate-3 conformal 将 `window×candidate` 展平。当前没有 r2 实验事实，Gate 1、Stage B/C、Gate 4 均继续锁定。

# 2. GitHub Visibility Certificate

## 2.1 可见性结论

审查固定在：

* Repository：`yuzbo/OpenTAD_C3_CoarseClean_20260702`
* Commit：`4b07020acb2611c3f085488d2f678f3be037f1be`
* Commit message：`docs: record independent ChronoTransport implementation audit`

该 commit、必读文档、列出的 ChronoTransport 核心源码、配置、现有工具、脚本和测试均可通过 GitHub 只读接口打开。没有发生 `GITHUB_VISIBILITY_BLOCKED`。

## 2.2 规范身份

固定提交中的 r2 spec 与批准提交 `e4422f5…` 返回相同 Git blob：

```text
0f54c9392f512c29f7ef59eb0afda61fe8dfa5f2
```

因此二者是同一 Git 对象、同一字节内容。审查将 prompt 给出的 SHA-256

```text
87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8
```

作为项目提供的固定身份；由于 GitHub connector 未暴露可下载的原始字节流，本次没有独立执行 SHA-256 重算。这一限制不影响“批准提交与固定提交字节相同”的 Git blob 结论。

规范永久链接：

[CT-P3R-3S-r2 normative specification](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/docs/superpowers/specs/2026-07-12-chronotransport-ct-p3r-3s-r2-design.md)

## 2.3 在固定树中不存在，而不是“不可访问”的表面

直接读取返回不存在的包括：

```text
tools/bata/train_chronotransport_r2_stage_b.py
tools/bata/run_chronotransport_r2_gates23.py
tools/bata/train_chronotransport_r2_stage_c.py
tools/bata/train_chronotransport_r2_matched_dense.py
opentad/models/chronotransport/full_stack_profiler.py
tools/bata/profile_chronotransport_r2_full_stack.py
tools/bata/run_chronotransport_r2_gate4.py
tests/test_chronotransport_r2_gate1.py
```

实现计划也仍把 formal manifest、Stage B/C、matched dense、Gate 3/4、full-stack profiler、深注册和下游 launchers 列为后续任务，而不是已完成工作流。

## 2.4 上游可见性

以下固定上游均可访问：

* OpenTAD：`sming256/OpenTAD@1aa8ca4ac5e846b1e8ff69298dd6607121a01589`
* AdaTAD：`sming256/AdaTAD@25e06c720e450298ca5267fda6927f3591dcdfef`
* VideoMAE：`MCG-NJU/VideoMAE@14ef8d856287c94ef1f985fe30f958eb4ec2c55d`

AdaTAD 固定仓库主要提供论文、配置和指向 OpenTAD 的正式实现入口，因此具体 adapter 源码应以固定 OpenTAD commit 为主，不应把当前 fork 称为“official unmodified AdaTAD”。

---

# 3. Evidence Table

| 证据类别                         | 本次可用内容                                                                                        | 不允许推导的内容                           |
| ---------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------- |
| `REPOSITORY_FACT`            | 固定 SHA 中的源代码、配置、测试、wiki、计划；上游固定 commit                                                        | 代码能够在真实数据/GPU 上正确执行；Gate 已通过       |
| `PROJECT_REPORTED_TEST_FACT` | wiki 报告 110 tests / 84.58s；局部 runtime、risk、Gate 1/2 primitive、Stage-C gradient primitive 测试通过 | 不能当作独立复跑，也不能证明缺失 workflow 存在       |
| `EXPERIMENT_FACT`            | **r2 appeal 无任何可用 experiment fact**                                                           | 不得生成新三种子结论、成本、mAP、coverage、Gate 状态 |
| `INFERENCE_OR_PROPOSAL`      | 本报告的失效轨迹、统计反例、patch、测试计划                                                                      | 不得称为已执行或已验证                        |

项目 wiki 自身明确说明：110 个测试只验证现有子集，formal Stage B/C、full-stack timing、registration 和 Gate 1–4 没有被验证，也没有科学 claim 解锁。

历史 `92029ea` 记录的是旧 Stage-B/P3 负结果：risk-regret 排序为负、旧 cell-risk 聚合与窗口 target 尺度错配、P3 FAIL；该记录既不能验证 r2，也不能被 r2 文档审查抹去。

---

# 4. Independent-Audit Recheck

七项全部为 **`AGREE`**。没有一项被源码推翻。

## 4.1 Gate 3、Gate 4 adjudicator 和 launcher 缺失

**裁决：`AGREE`**

**源码证据。** [adjudication.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/adjudication.py#L60-L246) 只实现 `gate1_oracle_headroom` 与 `gate2_matched_transport`；没有 Gate 3 或 Gate 4。固定树中相应 runner 和 launcher 不存在。

**违反规范。** §11、§12、§13.6–13.8、§15、§16。

**错误风险。**

* False positive：极高。synthetic primitive 可能被误写成 Gate 3/4 PASS。
* False negative：低。缺失就是缺失，不依赖运行环境。

**最小修复。** 加入纯 Gate-3/Gate-4 adjudicator、不可变 artifact schema、upstream unlock 验证和 GPU1 launcher。

**精确回归测试。**

```text
test_gate3_requires_exact_3_seed_30_window_16_candidate_tensor
test_gate3_rejects_dense_fallback_in_coverage_denominator
test_gate4_requires_official_video_seed_vectors_and_six_order_timing
test_gate4_rejects_missing_post_stage_c_gate3_unlock
```

**是否改科学规格：** 否，只实现批准规格。

---

## 4.2 没有可执行 r2 Stage B、Stage C 或 matched-dense

**裁决：`AGREE`**

**源码证据。**

旧 [run_chronotransport_stage_b_formal.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/tools/bata/run_chronotransport_stage_b_formal.py#L43-L269)：

* 固定六个旧 schedule；
* 按 run seed 创建旧 split；
* 使用 `cfg.dataset.train`；
* `total_steps=len(fit)*epochs`；
* coverage、Spearman、bootstrap 都由 CLI 提供。

通用 [train_chronotransport_stage_b.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/tools/bata/train_chronotransport_stage_b.py) 允许任意 `steps`，没有 r2 formula、140 成功更新锁、LR-scheduler 成功计数或完整注册验证。

**违反规范。** §5、§7、§11.2、§13.1–13.5、§14–16。

**错误风险。**

* False positive：极高。旧六 schedule/旧 split 结果可能被贴上 r2 标签。
* False negative：高。错误 exposure、随机窗口和训练长度也可能错误否定 r2。

**最小修复。** 三个正式 runner：

```text
train_chronotransport_r2_stage_b.py
train_chronotransport_r2_stage_c.py
train_chronotransport_r2_matched_dense.py
```

并由 registration 派生所有常量。

**精确回归测试。**

```text
test_stage_b_runner_has_exactly_140_successful_updates_and_formula
test_stage_b_runner_rejects_epoch_or_schedule_cli_override
test_stage_c_runner_has_exactly_4200_successful_updates
test_matched_dense_and_ct_share_successful_batch_hash_sequence
```

**是否改科学规格：** 否。

---

## 4.3 Transactional AMP overflow retry 缺失

**裁决：`AGREE`**

**源码证据。** [stage_c.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/stage_c.py#L81-L157) 在 `unscale_` 后检测到非有限梯度便立即抛异常，尚未执行 `scaler.step()` 和 `scaler.update()`，因此没有 scaler backoff；文件没有 model/optimizer/EMA/scheduler/RNG/buffer snapshot、恢复、同 batch retry 或三次上限。

**违反规范。** §13.5。

**错误风险。**

* False positive：极高。scheduler/EMA/cursor 可能与成功 optimizer updates 失配。
* False negative：高。一次可恢复 overflow 会错误终止运行。

**最小修复。** 加入事务 snapshot、除 scaler 外 bitwise restore、same-materialized-batch retry、最多三次 retry、成功后才推进 scheduler/EMA/exposure cursor。

**精确回归测试。**

```text
test_stage_c_overflow_preserves_scaler_backoff_but_restores_all_other_state
test_stage_c_retry_reuses_identical_materialized_batch_and_rng
test_stage_c_success_advances_every_success_counter_once
test_stage_c_fourth_overflow_marks_invalid_implementation
```

**是否改科学规格：** 否。

---

## 4.4 Scheduler 未约束注册 `B*` 与 exact requested cost

**裁决：`AGREE`**

**源码证据。** [scheduler.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/scheduler.py#L98-L230) 的可行集只检查 finite risk、age、`risk<=epsilon`；没有 `candidate_cost<=B*`。exact lookup 可选，缺失时退回 action-cell additive estimate。

**违反规范。** §8.2、§8.4、§11.4。

**错误风险。**

* False positive：极高。风险低但超预算的 candidate 会被选中。
* False negative：中。错误 proxy 成本也可能排除真实可行 candidate。

**最小修复。** formal mode 强制：

```text
registered_budget_p50
registered_library_sha256
exact ScheduleCostLookup
requested action hash identity
cost_valid && requested_p50 <= B*
```

**精确回归测试。**

```text
test_scheduler_filters_candidate_above_registered_bstar
test_formal_scheduler_rejects_missing_exact_lookup
test_scheduler_fails_dense_when_all_non_dense_cost_keys_missing
test_scheduler_rejects_registered_library_hash_mismatch
```

**是否改科学规格：** 否。

---

## 4.5 Profiler 和 cost key 不足以产生 formal full-stack evidence

**裁决：`AGREE`**

**源码证据。**

[profiler.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/profiler.py#L15-L110) 只积累 stage timers；`fill_missing=True` 会为未测 stage 生成 `count=0,p50=0,p95=0`。它没有 invocation-level `total_ms`、50 warmups、200 measured invocations、六序列 crossover 或注册身份。

[cost_lookup.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/cost_lookup.py#L12-L59) 的 key 只有 hardware、precision、batch size、candidate name、RECOMPUTE rows；缺少 UUID、driver、CUDA、PyTorch、source/spec/config/checkpoint/library/action/environment hashes。

**违反规范。** §8、§13.6–13.8、§14。

**错误风险。**

* False positive：极高。stage percentile 和或占位零可能被冒充 full-stack latency。
* False negative：高。错误环境或旧 checkpoint cost 也会污染比较。

**最小修复。** invocation-level profiler，直接测 `total_ms`，并把完整 provenance 写入不可变 raw-row artifact。

**精确回归测试。**

```text
test_full_stack_profile_rejects_49_warmups
test_full_stack_profile_rejects_199_measured_invocations
test_full_stack_profile_rejects_stage_sum_as_total_ms
test_cost_key_changes_on_every_registered_provenance_field
```

**是否改科学规格：** 否。

---

## 4.6 Registration 接受 caller-authored identity

**裁决：`AGREE`**

**源码证据。**

[registration.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/registration.py#L12-L74) 只检查顶层字段、protocol ID、一个 attestation 和 registration 自哈希，不从文件系统、Git、checkpoint 或数据重新派生身份。

[register_chronotransport_r2.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/tools/bata/register_chronotransport_r2.py#L19-L28) 直接读取 `--identity` JSON 并封装。

测试甚至接受：

```python
"spec": {"commit": "e4422f5", "sha256": "87FA"}
"implementation_commit": "I" * 40
"window_manifest": {"sha256": "w" * 64, ...}
```

说明测试只验证自一致封装，不验证真实身份。

**违反规范。** §14.1–14.2、§15。

**错误风险。**

* False positive：极高。任意伪造 hash 可形成“有效 registration”。
* False negative：低。

**最小修复。** 删除 `--identity`；generator 只接收不可变输入路径，自己重算 Git、spec、source、config、checkpoint、media、manifest、library、exposure、environment 身份。

**精确回归测试。**

```text
test_registration_cli_has_no_identity_argument
test_registration_rejects_short_nonhex_or_placeholder_hash
test_registration_recomputes_every_identity_from_bytes
test_registration_rejects_i_to_r_extra_changed_file
```

**是否改科学规格：** 否。

---

## 4.7 Gate-1 数据和统计常量仍由 caller 控制

**裁决：`AGREE`**

**源码证据。**

[run_chronotransport_r2_gate1.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/tools/bata/run_chronotransport_r2_gate1.py#L19-L29) 将整个 caller JSON 直接 `gate1_oracle_headroom(**payload)`。

Gate-1 函数允许 caller 提供：

```text
calibration
evaluation
candidate_cost_p50
dense_cost_p50
budget
bootstrap_samples
bootstrap_seed
```

且候选集合由 mapping 名字后缀推断，而不是验证完整注册 library。

[GPU1 launcher](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/scripts/run_chronotransport_r2_gate1_gpu1.sh#L9-L36) 虽检查 clean tree、HEAD、GPU1、Slurm，但不加载 registration、不重算 content hashes、不限制 output root、不验证 upstream unlock。

**违反规范。** §9、§12、§14、§15。

**错误风险。**

* False positive：极高。可替换预算、候选、population、bootstrap seed 或输入 regret。
* False negative：高。错误输入同样可错误冻结路线。

**最小修复。** Gate-1 CLI 只接受 `--registration` 和注册内固定 artifact IDs；所有常量从 registration 派生，禁止 CLI override。

**精确回归测试。**

```text
test_gate1_cli_rejects_budget_bootstrap_or_population_override
test_gate1_requires_exact_30_calibration_and_30_evaluation_windows
test_gate1_rejects_extra_missing_or_reordered_library_candidates
test_gate1_launcher_recomputes_registration_and_output_root_identity
```

**是否改科学规格：** 否。

---

# 5. Additional P0/P1 Findings

## P0-A：r2 config overlay 写在错误嵌套层

[r2 Stage-B config](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/configs/adatad/thumos/c3_chronotransport_r2_stage_b.py) 写的是：

```python
model.backbone.chronotransport
```

但基础模型的结构是：

```text
model.backbone                    = mmaction.Recognizer3D
model.backbone.backbone           = VisionTransformerAdapter
model.backbone.backbone.chronotransport
```

基础配置中 `chronotransport` 正确放在第二层 `backbone.backbone`。

因此当前 overlay 不能可靠覆盖真正 runtime 的旧字段。可能结果只有两种：

1. MMAction 对额外 `chronotransport` 参数报构建错误；
2. wrapper 接受/忽略额外字段，而内层继续使用 Stage-A 的 legacy `max_cache_age=8`。

第二种情况下，`hold_only` 和 `transport_only` 会在 age 8 后被 repair，直接违反 r2 的 `hard_cache_validity_age=47`。

**Mandatory test：**

```text
test_r2_stage_b_resolved_config_places_chronotransport_only_under_inner_vit
test_r2_stage_c_resolved_config_has_hard_age47_embedding_cap8
test_r2_configs_build_detector_without_unknown_wrapper_parameter
```

---

## P0-B：Gate-3 conformal 使用错误独立单位

[risk.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/risk.py#L75-L92) 的 helper：

```python
residual = (target - prediction).flatten().clamp_min(0.0)
```

旧 [formal_stage_b.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/formal_stage_b.py#L114-L143) 同样把每个 schedule row 直接放进 conformal residual set。

规范要求：

```text
每个 calibration window：
score_i = max over 16 candidates max(regret - q_hat, 0)

然后对 30 个 score_i 取 rank 28。
```

而不是对 `30×16=480` rows 取 quantile。

**最小反例。**

* 30 个窗口；
* 27 个窗口全部 residual=0；
* 3 个窗口各有一个 candidate residual=100，其余为 0。

正确：

```text
window maxima = 27×0 + 3×100
rank 28 / 30 = 100
```

当前 flatten：

```text
477×0 + 3×100
rank ceil(481×0.9)=433 / 480 = 0
```

结果会把 `q_conf=100` 错算为 `0`，产生严重 undercover 和假 Gate-3 PASS。

---

## P1-C：没有完整 label-free one-window manifest workflow

`protocol.py` 已实现 NFC、200-video split、window digest、edge padding 和 Stage-B exposure primitive；这些是正确的局部实现。

但 `build_window_payload`：

* 允许任意 `width`，没有 formal `width==768` 锁；
* 不构造 source sampled-index vector；
* 不记录 fps、stride、scale factor、source frames、media path/registry、data/config/annotation hashes；
* 不构造 exact 200-window manifest、split membership、window IDs 和整体 manifest hash；
* 没有 executable builder/validator。

当前 factory 反而构造 `cfg.dataset.train`。基础训练 pipeline 使用 `method="random_trunc"`。

上游 `random_trunc` 明确根据 GT intersection 和 `trunc_thresh` 选择包含动作的 crop，因此不能用于 r2 formal manifest。

---

## P1-D：旧统计实现把不同窗口/候选合并后算一个 Spearman

旧 `summarize_stage_b_evaluation` 对全部 records 的 predicted risk 和 regret 直接算一个 pooled Spearman，并在零方差时返回 0；规范要求每个 `seed×window` 在完整 16-candidate vector 内计算 rho，少于 3 个 distinct ranks 必须 fail closed。

这不是仅仅“Gate 3 尚未实现”，而是现存旧 formal helper 与 r2 定义相冲突，必须禁止 r2 runner 调用。

---

## P1-E：Stage-C exposure 只有 generator，没有 validator、hash 或 runner 绑定

`stage_c_exposure_matrix()` 会生成 8,400 rows/seed，但文件在生成后结束，没有：

* 525/candidate validator；
* exact successful-update/batch-position 检查；
* per-seed/combined SHA-256；
* checkpoint resume prefix 验证；
* CT/matched-dense shadow ledger 一致性。

---

## P1-F：requested/executed 成本账本仍可能误标 proxy cost

Runtime 已记录 requested/executed action hashes，这是正确进展。但 forced schedule 分支先 repair schedule，再用 additive `cost_table.estimate(schedule.actions)`；summary 又把同一值同时写为 `estimated_cost` 和 `requested_estimated_cost`。这不是原始 requested candidate 的 exact measured cost。

formal repair 已使样本 invalid，因此不能用该值通过 Gate；但 ledger 仍必须保留真正 requested cost 与 executed diagnostic cost，不能复用一个 proxy。

---

## P1-G：profiler 的“完整字段”验证可接受未测量的零占位

`summary(fill_missing=True)` 为未出现的 stage 写入：

```json
{"count": 0, "total": 0, "p50": 0, "p95": 0}
```

`validate_summary` 只检查 key 是否存在，不检查 `count>0`。一个完全没测 `data_decode` 的运行仍可通过字段完整性检查。

---

# 6. End-to-End Implementation Map

## 6.1 Tensor 路径

以下 shape 是由固定配置和源码推导，不是实际运行观测。

| 阶段                             | Tensor / 状态                 | 实际语义                                                                    |
| ------------------------------ | --------------------------- | ----------------------------------------------------------------------- |
| Dataset/window                 | 768-point temporal window   | r2 要求一个 video 一个固定 label-free window；当前 formal factory 尚未实现             |
| Wrapper preprocessing          | 48 个 16-frame chunks        | `768/16=48`；不删帧                                                         |
| VisionTransformerAdapter input | 约 `[B×48,3,16,160,160]`     | 每个 chunk 独立进入 VideoMAE patch embedding                                  |
| Patch embedding                | `[B×48,800,384]`            | tubelet 2 → 8 temporal tubelets；160/16=10×10 spatial；8×10×10=800 tokens |
| Runtime reshape                | `[B,48,800,384]`            | 48 个 chunk rows                                                         |
| 每 block heavy path             | selected rows `[R,800,384]` | 只控制 attention+MLP                                                       |
| Heavy surrogate                | `[B,48,800,384]`            | RECOMPUTE/TRANSPORT/HOLD 重新组成全部 rows                                    |
| AdaTAD adapter                 | full surrogate              | adapter 对所有 48 rows 执行，结果进入下一 block                                     |
| Backbone output                | `[B×48,384,8,10,10]`        | spatial feature map                                                     |
| Postprocess                    | `[B,384,384]`               | 48 chunks×8 tubelet points                                              |
| Interpolate                    | `[B,384,768]`               | 恢复 detector 768 grid                                                    |
| Projection/head/loss           | ActionFormer path           | CT 不修改 projection/head/loss                                             |
| Postprocess/NMS                | official OpenTAD downstream | CT 不改变 NMS                                                              |

基础配置的 48-chunk reshape、spatial reduction、384→768 interpolation 可在固定配置中直接看到。

Patch embedding 会在 160×160 输入时把位置编码插值到实际 10×10 grid。

## 6.2 每种 action 的状态与梯度

| Action      | 当前 row 来源                                      | Cache mutation                                        | 当前梯度                                  | 下一 block 看到什么                              |
| ----------- | ---------------------------------------------- | ----------------------------------------------------- | ------------------------------------- | ------------------------------------------ |
| `RECOMPUTE` | `H_l(current)`                                 | `anchor=detach(u_live)`；`latest=detach(u_live)`；age=0 | 当前 heavy row 保持 live，经 adapter 到 loss | 全 rows adapter 输出                          |
| `HOLD`      | `latest_detached`                              | cache 不变；age+1                                        | 对历史 row 无 recurrent gradient          | HOLD surrogate 经全 rows adapter 后进入下一 block |
| `TRANSPORT` | `T(latest_detached,current,min(actual_age,8))` | `latest=detach(u_live)`；anchor 不变；age+1               | 当前 transport/current row 保持 live      | transport surrogate 经 adapter 后进入下一 block  |

源码确实先对 selected rows执行 heavy，再按 temporal order填充 HOLD/TRANSPORT，最后对完整 surrogate 执行 adapter，并将 adapter 输出写回全部 rows。

### 正确实现的局部语义

* cache 位于 pre-adapter heavy boundary；
* TRANSPORT 使用 latest，不是 anchor；
* current RECOMPUTE/TRANSPORT row 保持 live；
* recurrent cache detach；
* forced dense 直接调用原始 block path；
* dynamic heavy 和 adapter 都覆盖 activation checkpoint 分支。

### 尚未形成 formal evidence 的部分

* config 可能根本没有启用 r2 age 合同；
* exact requested cost 未绑定；
* repair/fallback 只会标记 runtime summary，尚无 formal artifact validator；
* forced-dense CPU/CUDA FP32/AMP parity 没有本审查者独立执行。

## 6.3 Stage-B loss/RNG 流

现有 primitive 的意图是：

```text
materialized batch
  ├─ restore RNG → dense no-grad branch
  └─ restore same RNG → counterfactual grad branch

regret = max(L_counterfactual - L_dense, 0)
LB = L_counterfactual
   + 0.1*MSE(Fcounterfactual, Fdense.detach())
   + 0.1*Pinball(q_hat, regret.detach())
```

这一局部 loss 关系基本符合规范。问题在于 executable workflow 仍使用 train `random_trunc`、循环旧 schedule、任意 steps，而不是 registered one-window manifest 和 exact exposure formula。

## 6.4 Stage-C A/T/R 流

已实现：

* object-identity A/T/R primitive；
* 三次 `autograd.grad`；
* A 只接 LD；
* T 接 LD+0.1LF；
* R 接 0.1LR；
* 一次 `unscale_`、一次 clip、一次 scaler step/update。

未实现：

```text
materialized batch hash
→ full mutable-state snapshot
→ attempt
→ overflow skip/backoff
→ restore except scaler
→ same-batch retry
→ success-only scheduler/EMA/cursor advance
→ resume-prefix validation
```

## 6.5 Scheduler 流

当前：

```text
16 non-dense + dense
→ q_hat
→ dense risk = 0
→ finite/risk/age filter
→ lowest estimated cost
→ no feasible → dense
```

正式要求但缺失：

```text
registered library hash
AND exact requested action hash cost key
AND requested_p50 <= registered B*
AND upper <= epsilon
AND finite
AND metadata/hash valid
```

## 6.6 Registration → Gate 流

当前实际链：

```text
caller identity JSON
→ shallow top-level checks
→ self-hash
→ caller Gate-1 JSON
→ pure Gate-1 function
```

规范要求：

```text
clean I
→ detached generator derives every identity from bytes
→ sole registration commit R
→ launcher verifies HEAD=R, clean tree, all hashes, GPU/Slurm/output root
→ registered profile/replay population
→ Gate1
→ Stage B
→ Gate2
→ Gate3
→ Stage C + matched dense
→ recalibration + Gate3 rerun
→ Gate4
```

两者目前不是同一个系统。

---

# 7. Sixteen-Section Spec Compliance Matrix

本表不授予任何 `PASS_IMPLEMENTED_AND_TESTED`，因为本审查没有执行测试。

| 规范节                                | 状态                                     | 结论                                                                                 |
| ---------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------- |
| §1 裁决、证据边界与目标                      | `IMPLEMENTED_NOT_INDEPENDENTLY_TESTED` | wiki/claim flags 基本诚实；没有把 tests 写成 science                                         |
| §2 不变任务与官方骨架                       | `PARTIAL`                              | 768/48/384、下游 detector 保留；r2 config overlay 错层                                     |
| §3 block/cache/adapter 语义          | `IMPLEMENTED_NOT_INDEPENDENTLY_TESTED` | runtime 源码大体匹配；CUDA parity 未独立验证                                                   |
| §4 唯一窗口 quantile head              | `IMPLEMENTED_NOT_INDEPENDENTLY_TESTED` | D=23、64-64、mean/max、Softplus 已实现                                                   |
| §5 split/checkpoint/RNG            | `CONTRADICTS_SPEC`                     | split/window primitives 存在，但 executable path 使用 GT-aware random_trunc；无完整 manifest |
| §6 冻结 candidate library            | `IMPLEMENTED_NOT_INDEPENDENTLY_TESTED` | 16 non-dense+dense、controls primitives 存在；formal hash binding 缺失                   |
| §7 Stage-B 140 更新                  | `CONTRADICTS_SPEC`                     | 现有 runner 任意 steps/旧 cyclic schedules，无 exact formula 和成功更新账本                      |
| §8 cost/B*/requested-executed      | `CONTRADICTS_SPEC`                     | 无 full-stack total artifact；cost key 不完整；scheduler 无 B*                            |
| §9 Gate 1                          | `PARTIAL`                              | adjudication 主要数学结构存在，包括 replicate-time reselection；输入链未注册绑定                       |
| §10 Gate 2                         | `PARTIAL`                              | pure hierarchical primitive 存在；不强制 30×3、seeds、action-mask/hash、registered rows     |
| §11 Gate 3                         | `CONTRADICTS_SPEC`                     | adjudicator缺失；现有 conformal/旧 Spearman helper 使用错误统计单位                              |
| §12 统计、proxy、泄漏边界                  | `PARTIAL`                              | 部分 bootstrap primitive；无完整 Gate3/4 units、proxy artifacts 或泄漏验证                     |
| §13 Stage C/matched dense/Gate4    | `PARTIAL`                              | ownership/one-attempt gradient primitive 存在；retry、runners、matched dense、Gate4 均缺失  |
| §14 provenance/registration/claims | `CONTRADICTS_SPEC`                     | claim flags正确；registration 由 caller 编写且不深验                                         |
| §15 远端执行/产物/停止纪律                   | `MISSING`                              | 只有 Gate-1 launcher skeleton；无完整 formal launcher/unlock/output-root chain           |
| §16 完成定义                           | `MISSING`                              | 当前尚不满足 `implemented`，更不满足 `tested` 或 `experiment_running`                          |

规范本身明确要求：`implemented` 必须包含 Gate 1–4 adjudicators、Stage B/C、matched control、profiler 和 GPU1 launchers；当前显然未达到。

---

# 8. Statistical Red-Team Audit

## 8.1 可确认的数学常量

```text
Gate-3 calibration:
ceil((30+1)*0.9) = ceil(27.9) = 28

Fit-only constant baseline:
ceil((140+1)*0.9) = ceil(126.9) = 127
```

这两个 rank 在现有一维 helper 测试中被正确断言，但一维 rank 正确不等于 30×16 simultaneous calibration 正确。

## 8.2 需要严格区分的覆盖率

1. **Simultaneous marginal coverage**

   每个 `seed×window` 的 16 candidates 是否全部被 upper cover。

2. **Selected empirical coverage**

   scheduler 实际选中的 non-dense row 是否 covered。

3. **All-candidate-covered**

   ```text
   min over 16 candidates (upper-regret) >= 0
   ```

4. **Window-all-selected-covered**

   对至少一个 seed 选择 non-dense 的窗口，所有这些 selected seed rows 是否都 covered。

当前没有 Gate-3 adjudicator，因此后两项不存在可执行定义。

## 8.3 数值反例

| 缺陷                         | 最小反例                                                                                                 | 正确结论                                 | 错误实现可能结论                                     |
| -------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------- |
| Conformal row flatten      | 30 windows，3 个 window 各有一个 residual=100                                                              | rank28=100                           | 480 rows 的 rank433=0                         |
| Dense fallback 混入 coverage | 1 个 selected non-dense 且 uncovered；29 个 dense fallback                                               | coverage=0/1=0                       | 错算 29/30=0.967                               |
| Spearman Simpson reversal  | window A: pred `[0,1,2]`, regret `[2,1,0]`；window B: pred `[100,101,102]`, regret `[1002,1001,1000]` | 两个 window rho 都是 -1，mean=-1          | pooled rho≈+0.5429                           |
| Spearman degeneracy        | 16 predictions 全相同                                                                                   | `INVALID_IMPLEMENTATION`/fail closed | 旧 helper 返回 0                                |
| Gate-2 population未锁        | 8 windows、1 seed、3 periods                                                                           | formal invalid                       | 当前函数可形成完整 inferred Cartesian set并 adjudicate |
| B* 未约束                     | `B*=6ms`；candidate cost=7.5ms，risk最低                                                                 | 不可行                                  | 当前 scheduler 可选中                             |
| Caller-controlled budget   | 注册 B*=6，Gate-1 JSON 写 budget=8                                                                       | 必须拒绝输入                               | 当前 Gate-1 接受                                 |
| Candidate-row伪独立           | 30 窗口内16 rows高度相关                                                                                    | n=30 clusters                        | row bootstrap 可伪装 n=480，CI 过窄                |
| Overflow backoff缺失         | scale=1024，出现 inf gradient                                                                           | step skip、scale下降、restore/retry      | 当前在 scaler update 前抛异常，scale仍1024            |
| Stage percentile相加         | 两个异步/重叠 stage 各 p50=10ms，真实 total p50=12ms                                                           | total=12ms                           | stage sum=20ms                               |
| Invocation伪独立              | 3 seeds、200 calls/seed                                                                               | video outer + seed inner hierarchy   | 把600 calls当独立会严重缩窄CI                         |

## 8.4 Gate-1 现有统计中正确的部分

`gate1_oracle_headroom` 在每个 bootstrap replicate 内重新选择 evaluation-best static 和 strongest comparator，并重新计算 time/layer/joint oracle；这一点符合规范。

它也在 strongest mean `<=1e-12` 时把 relative reduction 定义为无效，而不是加 epsilon。

问题不在这两个公式，而在 caller 能任意给 population、candidate set、budget、cost 和 bootstrap constants。

## 8.5 Gate-2 hierarchical bootstrap

现有 Gate-2 primitive 会 resample windows 和 seeds，并把 period vector留在 key 中，方向基本正确。

但它不强制：

```text
windows == exact registered 30 evaluation windows
seeds == (3407,3408,3409)
periods 对应相同 requested action masks
每行 action/library/profile hash 完整
bootstrap_samples == 5000
bootstrap_seed == 20260711
```

因此只是统计 primitive，不是 formal Gate 2。

## 8.6 Gate-4 confidence bounds

规范要求多处 one-sided LCB/UCB，包括：

* latency saving LCB；
* mAP drop UCB；
* shortest-Q1 drop UCB；
* overhead margin LCB；
* CT-static latency UCB。

当前没有 Gate-4 adjudicator，也没有 seed-level mAP vectors 或 official-video hierarchical bootstrap。不能用现有通用 two-sided percentile helper替代。

---

# 9. Official Upstream Verification Matrix

| 项目                               | 永久上游                                                                                                                                                                | Fork 分类                                                | 科学后果                                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| VideoMAE tubelet patch embedding | [VideoMAE PatchEmbed](https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/modeling_finetune.py#L136-L158)                             | `WRAPPED`                                              | Conv3d tubelet kernel/stride语义保留；fork增加位置编码空间插值和TAD wrapper                        |
| VideoMAE block order             | [VideoMAE Block](https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/modeling_finetune.py#L104-L133)                                  | dynamic 为 `STRUCTURALLY_MODIFIED`；forced dense为 `SAME` | 原顺序是 norm-attn residual，再 norm-MLP residual；dynamic路径对rows进行gather/cache/transport |
| AdaTAD adapter位置                 | [OpenTAD fixed implementation](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/backbones/vit_adapter.py#L192-L207) | `STRUCTURALLY_MODIFIED`                                | adapter仍位于attention+MLP之后，但输入由完整heavy surrogate重组；必须依靠full-row/parity测试            |
| AdaTAD temporal reshape          | [OpenTAD adapter source](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/backbones/vit_adapter.py)                 | `WRAPPED`                                              | 原dense temporal adapter保留；CT不能声称adapter计算被节省                                       |
| `random_trunc`                   | [OpenTAD LoadSnippetFrames](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/datasets/transforms/end_to_end.py#L56-L99)    | `SAME` primitive，但formal使用 `CONTRADICTS_SPEC`          | 使用GT intersection选择crop，不能构造label-free manifest                                    |
| short-vector edge padding        | [OpenTAD edge padding](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/datasets/transforms/end_to_end.py#L139-L159)       | `SAME`                                                 | `numpy.pad(mode="edge")`语义可用于r2 manifest                                           |
| ActionFormer head/loss           | [ActionFormerHead](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/dense_heads/actionformer_head.py)               | `WRAPPED`                                              | CT不修改head，但input feature已改变；不能称整套模型“unmodified official”                           |
| NMS                              | [OpenTAD 1D NMS](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/utils/post_processing/nms/nms.py#L103-L210)       | `SAME`                                                 | fork与上游文件 Git blob 均为 `c506ff35…`；NMS源码未改                                          |
| Activation checkpoint            | [OpenTAD Block checkpoint](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/models/backbones/vit_adapter.py#L194-L207)     | `STRUCTURALLY_MODIFIED`                                | dynamic heavy与adapter分别用non-reentrant checkpoint；必须单独验证梯度parity                    |
| Optimizer grouping               | [OpenTAD optimizer](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/cores/optimizer.py)                                   | formal Stage C 应为 `CUSTOM`，当前 workflow `MISSING`       | 标准name-based grouping不能替代A/T/R object identity                                     |
| AMP/scheduler/EMA                | [OpenTAD train engine](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/cores/train_engine.py#L39-L67)                     | formal Stage C `CUSTOM/MISSING`                        | 上游engine在每次scaler step后无条件推进scheduler/EMA，不满足overflow transaction合同                |
| LR scheduler                     | [LinearWarmupCosineAnnealingLR](https://github.com/sming256/OpenTAD/blob/1aa8ca4ac5e846b1e8ff69298dd6607121a01589/opentad/cores/scheduler.py#L10-L146)              | `SAME` primitive，formal runner `MISSING`               | r2必须只在successful update后推进；现无4200-update runner                                    |

**结论：** 当前 fork 是基于 OpenTAD/AdaTAD/VideoMAE 骨架的结构性修改版本，不是“official unmodified AdaTAD”。

---

# 10. Code Findings P0 → P3

## P0-01 — r2 配置未覆盖实际 runtime

* **类型：** wrong implementation
* **位置：** [r2 Stage-B config](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/configs/adatad/thumos/c3_chronotransport_r2_stage_b.py)
* **事实：** overlay 放在 `model.backbone.chronotransport`，实际 runtime 位于 `model.backbone.backbone.chronotransport`。
* **违反：** §3.4、§16。
* **失败轨迹：** 内层保留 legacy age8 → long HOLD/TRANSPORT被repair → action hash变化 → formal sample invalid。
* **影响：** 所有 r2 runtime、profile、Gate 数据不可识别。
* **补丁边界：** 两个 r2 config和resolved-config build tests。
* **测试：** `test_r2_stage_b_overrides_inner_vit_runtime_not_recognizer3d`。

## P0-02 — simultaneous conformal 被候选行伪重复

* **类型：** wrong statistical implementation
* **位置：** [risk.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/risk.py#L75-L92)、[formal_stage_b.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/formal_stage_b.py#L114-L143)
* **违反：** §11.1。
* **失败轨迹：** 30×16 flatten → rank433，而不是30 window maxima的rank28。
* **影响：** 低估 `q_conf`、undercoverage、假 Gate-3 PASS。
* **补丁边界：** 加 `simultaneous_conformal_offset`；一维 helper拒绝二维矩阵。
* **测试：** `test_simultaneous_gate3_calibration_takes_candidate_max_before_rank28`。

## P0-03 — caller 可伪造完整 registration

* **类型：** wrong implementation
* **位置：** [registration.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/registration.py#L50-L74)
* **违反：** §14。
* **失败轨迹：** caller填 `"87FA"`、`"w"*64` → self-hash仍合法。
* **影响：** 所有 provenance 和 stop-chain 失去意义。
* **补丁边界：** derived generator、deep validator、I→R diff verifier。
* **测试：** `test_registration_recomputes_identity_and_rejects_placeholders`。

## P0-04 — formal Stage B/C/Gate 3/Gate 4 不存在

* **类型：** missing surface
* **位置：** 固定树缺失相应文件。
* **违反：** §7、§11、§13、§15、§16。
* **失败轨迹：** primitive/config 被误当 executable workflow。
* **影响：** 不得创建 I/R，不得启动任何 formal GPU job。
* **补丁边界：** 完整 runners、adjudicators、launchers。
* **测试：** repository-contract test必须import/CLI-precheck所有正式入口。

## P0-05 — Stage-C overflow 无 backoff/restore/retry

* **类型：** wrong implementation
* **位置：** [stage_c.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/stage_c.py#L130-L144)
* **违反：** §13.5。
* **失败轨迹：** inf grad → 先抛异常 → scaler不backoff → batch/cursor无法按合同重试。
* **影响：** CT与matched dense successful exposure不可比。
* **补丁边界：** transaction wrapper。
* **测试：** overflow/backoff/retry/resume determinism suite。

## P0-06 — 成本可行性不是 formal exact cost

* **类型：** wrong implementation
* **位置：** [scheduler.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/scheduler.py#L136-L207)
* **违反：** §8、§11.4。
* **失败轨迹：** proxy cost或无B* → 超预算candidate进入argmin。
* **影响：** latency/quality Pareto不可解释。
* **补丁边界：** exact lookup+B*+hash validity。
* **测试：** budget and missing-key fail-closed tests。

## P1-01 — formal 数据路径仍使用 GT-aware random_trunc

* **类型：** wrong workflow
* **位置：** [factory](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/tools/bata/chronotransport_opentad_factory.py#L101-L126)
* **违反：** §5.1。
* **影响：** window population label-leaked且不同seed/schedule可能不固定。
* **测试：** formal dataset不得包含 `random_trunc`，每video frame indices须等于manifest。

## P1-02 — old formal Spearman/coverage/bootstrap units 不兼容 r2

* **类型：** wrong implementation
* **位置：** [formal_stage_b.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/formal_stage_b.py)
* **违反：** §11–12。
* **影响：** Simpson reversal、row pseudoreplication、degeneracy被填0。
* **测试：** per-window rho和cluster bootstrap反例。

## P1-03 — Gate-1 候选集合和 tie order 可由 mapping 顺序改变

* **类型：** weak schema
* **位置：** [adjudication.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/adjudication.py#L76-L100)
* **违反：** §6、§9。
* **影响：** 可加入 `cheat_hold`，或通过mapping顺序改变equal-mean static identity。
* **测试：** exact names/order/hash，不允许额外候选。

## P1-04 — Stage-C generic grouping仍存在且formal runner未绑定object identity

* **类型：** incomplete implementation
* **位置：** `training.py` 与缺失 runner。
* **违反：** §13.3。
* **影响：** A/T/R optimizer exact-once membership不能证明。
* **测试：** requires-grad union、alias uniqueness、optimizer membership exact once。

## P1-05 — full-stack profiler与raw timing population缺失

* **类型：** missing surface
* **位置：** [profiler.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/profiler.py)
* **违反：** §8、§13.6–13.8。
* **影响：** B*、latency saving、overhead margin均不可计算。
* **测试：** 50 warmups/≥200 raw totals/six-order crossover。

## P2-01 — transport age 参数被 in-place clamp

* **类型：** local state mutation
* **位置：** [transport.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/4b07020acb2611c3f085488d2f678f3be037f1be/opentad/models/chronotransport/transport.py#L57-L75)
* **事实：** `age.clamp_(0,self.max_age)` 可能修改 caller Tensor/view。
* **反例：** caller传 `tensor([47])`，调用后变成 `tensor([8])`。
* **影响：** 当前 runtime传临时Tensor，直接风险有限；未来audit/ledger复用会污染actual age。
* **补丁：** `age = age.clamp(0,self.max_age)`。
* **测试：** `test_transport_does_not_mutate_age_argument`。

## P2-02 — paired replay缺少candidate-order permutation与resume状态测试

* **类型：** weak test / unverified plausible path
* **影响：** 当前RNG primitive可能正确，但没有证明schedule order不影响regret，也没有绑定loader cursor和materialized pixels。
* **测试：** permutation invariance、resume prefix、pixel hash equality。

## P2-03 — dense fail-closed summary缺少与non-dense等价的完整身份字段

* **类型：** incomplete ledger
* **影响：** downstream artifact容易把dense safety fallback误算成non-dense success。
* **测试：** 所有fallback必须带requested/executed hashes、cost-validity与selection exclusion flag。

## P3-01 — canonical JSON的非ASCII字节形式未在规范中明确

* **类型：** specification ambiguity
* **事实：** `canonical_json_bytes` 使用 `ensure_ascii=True`。
* **影响：** 不影响内部自一致，但外部独立实现可能选择 raw UTF-8，从而得到不同 hash。
* **修复：** 不改当前哈希算法；在schema文档中明确 JSON escapes、sorting、separators、BOM和newline规则。
* **测试：** NFC非ASCII golden-byte vectors。

---

# 11. Route A/B/C Discussion and Mandatory Verdict

| 路线                                           | 科学有效性                      | False-positive risk |          工程成本 |          信息增益 | Kill criterion                                    |
| -------------------------------------------- | -------------------------- | ------------------: | ------------: | ------------: | ------------------------------------------------- |
| **A：接受当前提交并注册/Gate1**                        | 无效                         |                  极高 | 表面低、实际会污染正式记录 |           接近0 | 当前七项blocker或新增P0任一成立即kill；已经成立                    |
| **B：补全spec-preserving surfaces，再独立复审并创建I/R** | 唯一可保留bounded appeal可识别性的路线 |            修复后可降至受控 |             高 |  高；可真正裁决H1–H4 | 任一mandatory surface无法实现而不改规范，或第二次审查仍有P0，即转Route C |
| **C：现在永久冻结**                                 | 科学上安全                      |                  最低 |            最低 | 0；放弃r2 appeal | 团队不愿承担完整修复，或发现不可修复的核心语义/成本障碍                      |

## Mandatory verdict

# `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`

选择 **Route B**，但这不是“边修边跑”。在第二次独立 implementation review 输出批准前：

```text
禁止 I
禁止 R
禁止 profile
禁止 Gate 1
禁止新 Stage-B seeds
禁止 Stage C
禁止 Gate 4
```

---

# 12. Minimal Mandatory Patch Architecture

按依赖顺序，最小 patch set 是：

1. **Config与统计单位先修**

   * 修正 inner runtime config；
   * simultaneous window-level conformal；
   * 禁止旧 row-level r2 calibration。

2. **完整 manifest**

   * exact 200 videos；
   * shared 140/30/30；
   * one label-free 768-point window/video；
   * media/frame/config/annotation identities；
   * canonical manifest hash。

3. **Full-stack profiler与cost artifact**

   * 50 warmups；
   * ≥200 raw `total_ms`；
   * provenance-complete key；
   * exact requested/executed hashes；
   * B* derivation。

4. **Scheduler与Gate-1注册绑定**

   * exact cost+B* feasibility；
   * exact candidate library；
   * fixed 30/30 population；
   * fixed 5000/20260711；
   * no caller formal constants。

5. **Formal Stage B、Gate 2、Gate 3**

   * exact 140 successful updates；
   * formula exposure；
   * full 16-candidate fit replay；
   * rank-127 baseline；
   * per-window rho；
   * simultaneous rank-28 calibration；
   * hierarchical bootstrap。

6. **Transactional Stage C和matched dense**

   * object identity optimizer；
   * 4200 successful updates；
   * 8400/525 exposure；
   * overflow snapshot/restore/retry；
   * matched successful-batch hashes；
   * resume validation。

7. **Gate 4**

   * official full-video population；
   * three-seed prediction vectors；
   * six-order timing；
   * video/seed hierarchical bootstrap；
   * all hard one-sided bounds。

8. **Derived registration与formal launchers**

   * generator无 `--identity`；
   * clean detached I；
   * sole R；
   * deep revalidation；
   * output-root和unlock chain；
   * `PRECHECK_ONLY=1`。

9. **第二次独立实现审查**

   * 只在通过后创建 I/R；
   * 不提前读取 formal profile/replay/evaluation data。

---

# 13. Complete Unified Diffs / Complete Replacement Code

完整、源码约束内能够确定的 unified diffs 和 replacement code 已整理为：

[下载 ChronoTransport r2 registration-blocking patch proposal](sandbox:/mnt/data/chronotransport_r2_registration_blocking_patch_proposal.md)

其中包含：

| Patch                                                   | 状态                                                                     |
| ------------------------------------------------------- | ---------------------------------------------------------------------- |
| A：r2 config嵌套修复和resolved-config tests                   | 完整 diff                                                                |
| B：simultaneous conformal和pseudoreplication tests        | 完整 diff                                                                |
| C：registered `B*` / exact-cost scheduler feasibility    | 完整 diff                                                                |
| D：transactional Stage-C overflow snapshot/restore/retry | 完整 replacement code                                                    |
| E：full-stack profiler/cost artifact                     | `PATCH_BLOCKED_BY_MISSING_FACT`，给出完整interface与test contract            |
| F：derived immutable registration                        | `PATCH_BLOCKED_BY_MISSING_FACT`，给出完整CLI、derivation与validation contract |
| G：formal Stage B/C/matched dense/Gate3/Gate4            | `PATCH_BLOCKED_BY_MISSING_FACT`，给出完整workflow与hard assertions           |

被阻塞部分缺少的不是“编码耐心”，而是固定树中尚未定义的正式事实：

* canonical real-data manifest adapter；
* authenticated checkpoint registry schema；
* environment fingerprint schema；
* full-stack invocation factory；
* raw profile artifact schema；
* Stage-B/C/Gate3/Gate4 artifact schemas；
* official-video prediction/mAP vector schema。

在这些事实不存在时直接生成“完整生产 runner”只能靠发明接口，违反本次审查边界。

**全部补丁均为 `NOT_EXECUTED_BY_REVIEWER`。**

---

# 14. TDD and Verification Matrix

| Patch               | Red test                         | Green command               | 环境                | Passing证明什么                      | 不证明什么                  |
| ------------------- | -------------------------------- | --------------------------- | ----------------- | -------------------------------- | ---------------------- |
| Config              | resolved config仍在wrapper层        | focused config pytest       | CPU               | r2字段落到inner ViT                  | 不证明CUDA runtime        |
| Conformal           | 30×16反例返回0                       | risk pytest                 | CPU               | window max→rank28正确              | 不证明真实coverage          |
| Scheduler           | 超B* candidate被选                  | budget pytest               | CPU               | formal feasibility含exact cost+B* | 不证明cost测量真实            |
| Stage-C transaction | overflow后参数/RNG变化                | fake-scaler tests           | CPU               | snapshot/restore/counter语义       | 不证明PyTorch CUDA scaler |
| Stage-C CUDA        | real scaler overflow/retry       | remote CUDA suite           | GPU1              | backoff、step skip、retry行为        | 不证明4200-update结果       |
| Manifest            | random_trunc或非200/140/30/30可通过   | manifest tests              | CPU/data metadata | population/hash合同                | 不证明视频decode            |
| Profiler            | 49 warmups/199 rows可通过           | profiler schema tests       | CPU               | artifact fail-closed             | 不证明GPU latency         |
| Gate3/4             | wrong units/population可PASS      | adjudicator synthetic tests | CPU               | 数学与schema                        | 不证明science gate        |
| Launcher            | dirty tree/HEAD≠R/root escape可运行 | precheck tests              | CPU/remote shell  | launch guard                     | 不证明Slurm/GPU job       |

## 14.1 基础静态矩阵

```bash
# NOT_EXECUTED_BY_REVIEWER
git diff --check

# NOT_EXECUTED_BY_REVIEWER
python -m py_compile \
  opentad/models/chronotransport/risk.py \
  opentad/models/chronotransport/scheduler.py \
  opentad/models/chronotransport/runtime.py \
  opentad/models/chronotransport/stage_c.py \
  opentad/models/chronotransport/registration.py \
  tools/bata/train_chronotransport_r2_stage_b.py \
  tools/bata/train_chronotransport_r2_stage_c.py \
  tools/bata/train_chronotransport_r2_matched_dense.py \
  tools/bata/profile_chronotransport_r2_full_stack.py \
  tools/bata/run_chronotransport_r2_gates23.py \
  tools/bata/run_chronotransport_r2_gate4.py
```

## 14.2 Focused r2 tests

```bash
# NOT_EXECUTED_BY_REVIEWER
python -m pytest \
  tests/test_chronotransport_r2_configs.py \
  tests/test_chronotransport_r2_protocol.py \
  tests/test_chronotransport_r2_actions_cache.py \
  tests/test_chronotransport_r2_risk.py \
  tests/test_chronotransport_r2_scheduler_budget.py \
  tests/test_chronotransport_r2_runtime.py \
  tests/test_chronotransport_r2_adjudication.py \
  tests/test_chronotransport_r2_stage_c.py \
  tests/test_chronotransport_r2_registration.py \
  tests/test_chronotransport_r2_formal_workflows.py \
  tests/test_chronotransport_r2_profiler.py -q
```

## 14.3 现有回归

```bash
# NOT_EXECUTED_BY_REVIEWER
python -m pytest \
  tests/test_chronotransport_core.py \
  tests/test_chronotransport_integration.py \
  tests/test_chronotransport_opentad_integration.py \
  tests/test_chronotransport_repository_contract.py \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py -q
```

## 14.4 Remote CUDA behavior

```bash
# NOT_EXECUTED_BY_REVIEWER
python -m pytest \
  tests/test_chronotransport_r2_forced_dense_cuda_parity.py \
  tests/test_chronotransport_r2_amp_overflow_transaction.py \
  tests/test_chronotransport_r2_retry_resume_determinism.py -q
```

必须覆盖：

```text
forced dense FP32 output/loss/input-grad/adapter-grad bitwise parity
forced dense AMP atol=1e-6 rtol=1e-5
adapter enabled/disabled
checkpoint on/off
overflow on initial attempt and each retry
scaler backoff preserved
optimizer/scheduler/EMA/cursor exact counts
resume prefix and next exposure identity
```

## 14.5 Launcher precheck

```bash
# NOT_EXECUTED_BY_REVIEWER
PRECHECK_ONLY=1 \
CUDA_VISIBLE_DEVICES=1 \
scripts/run_chronotransport_r2_gate1_gpu1.sh
```

后续每个 Stage B/C/matched/profile/Gate launcher 都必须有独立 `PRECHECK_ONLY=1`，不能只复用 Gate-1 脚本存在性作为证明。

---

# 15. Registration Readiness Checklist

| 项目                                      | 状态                  |
| --------------------------------------- | ------------------- |
| 固定 commit 可见                            | ✅                   |
| spec Git blob与批准提交相同                    | ✅                   |
| exact SHA-256由本审查独立重算                   | ⚠️ 未重算；项目提供值        |
| 16-candidate/library primitive          | ⚠️ 已实现，未独立运行        |
| dual-age/cache/runtime primitive        | ⚠️ 已实现，但r2 config错层 |
| resolved r2 Stage-B/C configs           | ❌                   |
| exact 200-video manifest                | ❌                   |
| one-window-per-video artifact           | ❌                   |
| full-stack raw `total_ms` profiler      | ❌                   |
| provenance-complete exact cost lookup   | ❌                   |
| registered B* scheduler enforcement     | ❌                   |
| executable r2 Stage B                   | ❌                   |
| exact 140-update/exposure ledger        | ❌                   |
| full fit 16-candidate rank-127 baseline | ❌                   |
| Gate 2 registered workflow              | ❌                   |
| Gate 3 adjudicator/launcher             | ❌                   |
| transactional Stage-C retry             | ❌                   |
| executable Stage C/matched dense        | ❌                   |
| post-Stage-C recalibration/Gate3 rerun  | ❌                   |
| Gate 4 adjudicator/launcher             | ❌                   |
| derived registration generator          | ❌                   |
| deep registration validator             | ❌                   |
| clean implementation commit I           | ❌                   |
| registration commit R                   | ❌                   |
| Gate results                            | ❌                   |
| `deploy=false` / `paper=false`          | ✅                   |

## Registration readiness

# `NOT_READY`

---

# 16. Next-Step Plan

## Step 1 — 修复不依赖正式数据的P0

**Inputs**

```text
pinned source
approved spec
synthetic fixtures only
```

**Outputs**

```text
config nesting tests
simultaneous conformal tests
registered-B* scheduler tests
transactional Stage-C tests
```

**Stop conditions**

* resolved config不能构建；
* 30×16 counterexample仍失败；
* overflow后任何非scaler状态变化；
* exact cost缺失时非dense仍可被选。

**Wiki state**

```text
implementation_incomplete
→ implementation_repair_in_progress
```

## Step 2 — 定义并实现不可变输入/artifact schemas

**Inputs**

```text
dataset metadata contract
checkpoint registry contract
environment fingerprint contract
full-stack invocation boundary
```

**Outputs**

```text
canonical 200-video/window manifest
profile raw-row schema
Stage-B/C ledgers
Gate3/Gate4 input schemas
```

**Stop conditions**

* schema仍允许caller提供formal常量；
  -任何hash不能从bytes重算；
* formal generator能读取result/profile/evaluation路径。

## Step 3 — 完成formal workflows，但只用synthetic/fixture验证

**Outputs**

```text
r2 Stage-B runner
Gates 2/3
Stage-C runner
matched-dense runner
full-stack profiler
Gate4
all GPU1 launchers
```

**Stop conditions**

* 任意CLI仍可改seed、candidate、formula、threshold、bootstrap或population；
* 任意runner可多执行一次successful update；
* 任意repair/fallback可作为valid sample；
* Gate4可在缺少post-Stage-C Gate3时启动。

**Wiki state**

```text
implementation_repair_in_progress
→ implementation_candidate_unreviewed
```

## Step 4 — Remote controlled verification

只执行测试和 precheck，不打开正式 Gate 数据。

**Required outputs**

```text
CPU focused suite report
CUDA forced-dense parity report
overflow/retry/resume report
launcher PRECHECK reports
source/test/config hashes
```

**Stop conditions**

任一失败即返回 `implementation_repair_in_progress`，不得生成 I。

**Wiki state**

```text
implementation_candidate_unreviewed
→ tested_candidate_pending_independent_review
```

## Step 5 — 第二次独立 implementation audit

审查者必须从零读取最终源码，并验证：

```text
all mandatory surfaces exist
all formal constants registration-bound
no placeholder identities
no result-aware registration
no untested config overlay
no old runner reachable from r2 launcher
```

只有输出：

```text
APPROVE_IMPLEMENTATION_FOR_REGISTRATION
```

后才允许：

1. 创建 clean implementation commit `I`；
2. detached worktree at I 生成 registration；
3. 只提交 registration和批准的wiki状态为 `R`；
4. 再运行 formal precheck。

## Step 6 — 硬停止链

```text
Gate1 FAIL → permanent freeze
Gate1 PASS → Stage B
Gate2 FAIL → permanent freeze
Gate2 PASS → Gate3
Gate3 FAIL → permanent freeze
Gate3 PASS → Stage C + matched dense
Stage-C invalid implementation → invalidate run；code change要求新I/R
post-Stage-C Gate3 FAIL → permanent freeze
Gate4 FAIL → permanent freeze
Gate4 PASS → bounded evidence only
```

---

# 17. Result-to-Claim Matrix

| 当前/未来结果                  | 允许 claim                                                                          | 禁止 claim                                          |
| ------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------- |
| 当前固定 commit              | implementation subset存在；历史负结果仍有效                                                  | 任意r2 Gate、metric、latency、deploy、paper             |
| 110 reported tests       | 局部implementation tests项目报告通过                                                      | formal workflow完整、science supported               |
| Gate 1 PASS              | `oracle_headroom=true`                                                            | learned input dependence、mechanism、metric、latency |
| Gate 2 PASS              | `mechanism=true`                                                                  | calibrated scheduler、official metric、latency      |
| Gate 3 PASS              | `calibrated_risk_on_frozen_window_protocol=true`                                  | Gate3保证转移到official full-video                     |
| Gate 4 PASS              | `metric_adatad_thumos14_official_full_video=true`；`latency_gpu1_fixed_stack=true` | deploy、paper、detector-agnostic、通用TAD              |
| 任一 Gate FAIL             | 对冻结r2组合的负结果                                                                       | 通过改Gate、补seed、换预算、换head再次上诉                       |
| `INVALID_IMPLEMENTATION` | 仅说明运行无效                                                                           | 不得解释为science PASS或FAIL                            |

即使四个 Gate 全部通过：

```text
deploy = false
paper = false
```

这与规范和当前 `claim_flags` 实现一致。

---

# 18. Final Kill Criteria

以下任一项发生，都必须停止，不得通过“先跑实验再看”绕过。

## 18.1 Registration前 kill

* r2 config仍不能确定落到 inner VisionTransformerAdapter；
* formal path仍可调用 `random_trunc`；
* manifest不是exact 200/140/30/30；
* registration仍接受caller identity/formal constants；
* exact cost key缺少任何规范字段；
* scheduler可在没有B*或exact key时选择non-dense；
* Stage-C overflow不能bitwise恢复全部非scaler状态；
* Gate3/4或formal runner仍是缺失、fixture、plan、config而非workflow；
* remote CPU/CUDA验证未记录；
* 第二次独立审查仍有任一P0。

## 18.2 Formal run kill

* requested/executed action hash不同；
* 任意repair、fallback、missing cost或non-finite值；
* candidate/window/exposure/hash不完整；
* attempted/successful/EMA/LR-scheduler更新数不精确；
* resume prefix或next cursor不匹配；
* profile不是直接测量的full-stack `total_ms`；
* population、seed、candidate、threshold或bootstrap常量与registration不同。

这些属于：

```text
INVALID_IMPLEMENTATION
```

不是科学 FAIL。

## 18.3 科学 kill

* Gate 1 任一 hard condition失败；
* Gate 2 任一 hard condition失败；
* Gate 3 任一 support/coverage/ranking/pinball条件失败；
* post-Stage-C Gate 3 rerun失败；
* Gate 4 任一 latency、mAP、short-action、overhead、static或Pareto条件失败；
* 任一 seed越过规范规定的失败阈值。

这些结果必须转为：

```text
negative_gate / frozen_baseline
```

不得补 seed、换预算、改统计单位、调 epsilon、改 head、延长训练或再申请 bounded appeal。

---

# Final Decision

```text
Route A: REJECTED
Route B: SELECTED
Route C: retained as mandatory fallback if the minimal patch set cannot pass re-audit

MANDATORY VERDICT:
REVISE_IMPLEMENTATION_BEFORE_REGISTRATION
```

当前提交不能成为 implementation commit `I`，不能生成 registration commit `R`，不能启动 Gate 1。

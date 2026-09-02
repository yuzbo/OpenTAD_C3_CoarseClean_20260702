# DUCA 统一全矩阵：从代码基座到完整 Slurm 部署与终态分析的单次 Agent 命令

**任务标识：`DUCA-UNIFIED-FULLMATRIX-v001-20260902`**

你是本轮唯一的执行总控 Agent。你必须在同一次任务中完成：权威代码核对、全矩阵实现、配置生成、聚焦测试、独立 Critic 审查、独立 Evaluator 运行前核验、形成唯一精确提交、建立远端干净快照、一次性提交完整 Slurm DAG，以及返回真实 Job ID、结果根和证据边界。

本任务不是让你逐轮提出建议，也不是让你等待某个实验结果后再决定下一项实验。**所有实验臂、确认种子、成本测量和统计作业都已预先冻结，必须在代码审查和运行前核验通过后一次性部署。** 结果依赖只用于保证作业输入存在，不允许根据中间 mAP 动态增删实验臂。

不要把代码存在、测试通过、Slurm 接受、作业 PENDING、作业启动失败或单个点估计误写成科学结论。不要询问已经可以从仓库、环境变量或本任务说明中确定的信息。确定性工程错误做最小修复并继续同一科学任务；任何会改变科学问题、数据、指标、基线、更新数、机制定义或论文主张的修改必须停止部署并在最终报告中列为科学 blocker。

---

## 一、权威身份与证据边界

### 1. Repository

```text
https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
```

### 2. 实现整合基座

```text
branch:   codex/duca-total60-plugin-cvpr-20260727
revision: 95ca6eb4a7e0ba8259c5afd976cc30d0fea58865
```

该 revision 仅作为当前可达实现的整合基座，不是实验结果身份。开始修改前必须：

```bash
git fetch --all --tags --prune
git cat-file -e 95ca6eb4a7e0ba8259c5afd976cc30d0fea58865^{commit}
git status --porcelain=v1
git rev-parse HEAD
```

若工作区不干净，创建新的隔离 worktree；不得覆盖用户改动。新分支固定为：

```text
codex/duca-unified-fullmatrix-20260902
```

### 3. 必须只读核对的干净历史身份

```text
H65 30+60 clean reference:
04c35a3b76897e6c1569eeede41ed3aecaf7f854

PJST-D1 matched training:
c73e8418de31cdcb2a445ff58a1e33ab9ab6a508

PJST-D1 read-only terminal evaluation:
7bd120f0d342bf175c97c365fba7cbd359df055e

Official dense AdaTAD shared reference:
01c58b9f2370e914150cf94d392208a4e211c053
```

只读参考值：

- 共享未修改 dense AdaTAD：Avg-mAP `68.73`，不得重新训练。
- 当前干净 H65 30+60 参考：Avg-mAP `65.13`、mAP@0.7 `43.31`。
- 历史 `65.385724` 仅是 30+60 诊断锚点，不是严格总计 6000-update 的匹配控制。
- PJST-D1 OFF/ON 点估计为 `65.063283/64.590802`；由于预登记 bootstrap 为 `0/10000`，没有总体置信区间，不得写成统计显著负结论。

### 4. 本任务的正式匹配基线

历史值只作描述。本任务的因果控制必须在同一新提交、相同公共预训练起点、相同数据顺序、相同 6000 次成功 optimizer update、相同 terminal EMA 和相同 evaluator 下重新建立：

- `U0`：strict-S60 exact-uniform K384；
- `H0`：strict-S60 原 H65 retention/transition 机制。

任何 arm 不得在 update 0 之前加载经过 THUMOS14 监督训练的 Stage-1 checkpoint。允许所有 arm 共享同一个公开 VideoMAE-S 预训练权重；如果 ASFormer 需要初始化，则所有涉及 ASFormer 的 arm 必须共享完全相同、未使用当前 THUMOS14 held-out validation/test 监督的初始化，并将其身份记录在 run manifest 中。

---

## 二、已经冻结的科学裁决

### 1. 主论文问题

在离线时序动作检测中，在 `T=768` 原始窗口只允许重型 VideoMAE-S 处理 `K=384` 帧、总训练不超过 6000 次成功 detector update 的条件下，使用低成本 ASFormer 语义证据构建尺度稳定的动作中心、onset、offset 和覆盖场，并以有上下界的自适应预算进行 exact-K 选帧，是否能优于使用相同语义先验的旧分配策略，同时保持下游检测器、损失、NMS、数据划分和官方 evaluator 不变？

### 2. 主因果比较

```text
A11 − A10
```

- `A10`：ASFormer semantic prior + legacy dual-phase allocation；
- `A11`：同一 semantic prior + robust signed phase field + adaptive bounded allocation。

只有 `A11−A10` 可以作为“新 phase allocation 的净效应”。`A10−A00` 只回答从 motion prior 恢复 semantic prior 的效应，不得归到 phase mechanism 名下。

### 3. 论文贡献层级

1. **主精度方法：** semantic robust phase allocation + Dense VideoMAE-S。
2. **独立表示研究：** corrected physical-time propagation；不得与纯 pre-backbone 主张混写。
3. **监督增强：** detector-feature signed deletion Taylor ranking；H65 已有 grad×input 家族，因此不得声称首次引入 Taylor。
4. **效率扩展：** Mixture-of-Depths，最终容量 0.65；不得称为 MoE，不得把固定 Top-K 温度误写成会改变 hard route identity。
5. **课程训练：** successful-update 驱动的训练基础设施，不作为独立模型创新。

### 4. 禁止的叙述

- 不得声称五个机制可以线性叠加得到 68%。
- 不得把 `32.14@0.7` 归给当前 CT-Conv。
- 不得声称 motion prior 已被证明单独造成固定 4–5 mAP 损失。
- 不得称 `2×2^l` 为恒定物理感受野。
- 不得称 B-AMoD 为多专家 MoE。
- 不得称温度变化会改变固定 Top-K 的前向排序。
- 不得将 `65.13`、`65.385724` 或 `68.73` 当作新 strict-S60 arm 的匹配对照。

---

## 三、完整实验矩阵：全部实现、全部一次性部署

完整定义以同目录 `duca_unified_matrix_manifest.yaml` 为唯一机器可读清单。你必须解析并验证它，而不是自行重新设计矩阵。

### Panel 0：严格控制锚点

| Arm | 定义 |
|---|---|
| U0 | exact-uniform K384，strict total 6000 successful updates |
| H0 | 原 H65 semantic retention/transition，strict total 6000 successful updates |

### Panel 1：Prior × Allocation

| Arm | Prior | Allocation |
|---|---|---|
| A00 | motion | legacy dual-phase |
| A10 | semantic | legacy dual-phase |
| A01 | motion | robust phase, adaptive quota |
| A11 | semantic | robust phase, adaptive quota |

主要效应与交互：

```math
semantic\ recovery = A10-A00
```

```math
phase\ effect\ under\ semantic = A11-A10
```

```math
phase\ effect\ under\ motion = A01-A00
```

```math
interaction = A11-A10-A01+A00
```

### Panel 2：Curvature × Quota

| Arm | Curvature | Quota |
|---|---|---|
| B00 | OFF | fixed |
| B10 | ON | fixed |
| A11 | OFF | adaptive |
| B11 | ON | adaptive |

Curvature 没有独立帧预算，只能作为 onset/offset 候选邻域中的弱局部重排序项。

### Panel 3：Phase × Physical Time

| Arm | Allocation | Physical time |
|---|---|---|
| A10 | legacy | OFF |
| A11 | robust phase | OFF |
| C01 | legacy | ON |
| C11 | robust phase | ON |

### Panel 4：Physical Time × Mixture-of-Depths

| Arm | Physical time | MoD |
|---|---|---|
| A11 | OFF | OFF |
| C11 | ON | OFF |
| E01 | OFF | ON |
| E11 | ON | ON |

### Panel 5：Taylor × Mixture-of-Depths

| Arm | Taylor | MoD |
|---|---|---|
| A11 | OFF | OFF |
| D1 | ON | OFF |
| E01 | OFF | ON |
| F11 | ON | ON |

### Panel 6：Schedule × Selector

| Arm | Selector | Schedule |
|---|---|---|
| H0 | H65 | 20/20/20 |
| A11 | robust phase | 20/20/20 |
| G10 | H65 | 15/20/25 |
| G11 | robust phase | 15/20/25 |

### 运行规模

开发矩阵：

```text
17 unique arms × seed 3407 = 17 runs
```

预冻结确认矩阵：

```text
arms = U0,H0,A10,A11,C11,D1,E01,F11
seeds = 4407,5407,6407
8 × 3 = 24 runs
```

总训练/终态评测任务：

```text
17 + 24 = 41 GPU tasks
```

开发 seed `3407` 不进入最终 seed 统计。禁止等开发结果出来后再选择确认 arm；上述 8 个 arm 已经预登记并必须随同整个 DAG 一次提交。

---

## 四、实现原则与允许修改的代码面

### 1. 必须优先复用

在修改前使用 `git grep`、`git log -S`、`git show` 和历史 registry 查找现有实现。必须复用：

- `opentad/models/selectors/duca_online_frame_selector.py`
  - 现有 `DucaOnlineFrameSelector`；
- `opentad/models/duca/acquisition.py`
  - 现有 acquisition 和 `global_structured_topk` / exact-K / max-gap 路径；
- `opentad/models/bricks/scale_adaptive_conv1d.py`
  - 现有 `ContinuousTimeScaleAdaptiveConv1d`；
- `opentad/models/backbones/vit_adapter.py`
  - 现有 B-AMoD/MoD token-depth routing 表面；
- 现有 ActionFormer detector 接线、selected-axis 到 original-time 的 inverse map；
- 现有 successful optimizer update 计数、EMA、scheduler 和 checkpoint 恢复路径。

### 2. 禁止重新创建

不得新建同义：

- selector；
- exact-K decoder；
- max-gap repair；
- actionness source；
- detector wrapper；
- ActionFormer head；
- NMS；
- evaluator；
- 通用实验编排平台或大而全 schema 系统。

### 3. 允许的新文件

仅允许为清晰和测试新增最小文件：

```text
opentad/models/duca/phase_fields.py
opentad/models/duca/feature_attribution.py
configs/adatad/thumos/duca_unified_fullmatrix/*.py
tools/bata/generate_duca_unified_fullmatrix.py
tools/bata/aggregate_duca_unified_fullmatrix.py
tools/bata/bootstrap_duca_unified_fullmatrix.py
scripts/duca_unified_fullmatrix/preflight.sbatch
scripts/duca_unified_fullmatrix/train_eval_array.sbatch
scripts/duca_unified_fullmatrix/cost_array.sbatch
scripts/duca_unified_fullmatrix/bootstrap_array.sbatch
scripts/duca_unified_fullmatrix/finalize.sbatch
scripts/duca_unified_fullmatrix/audit_afterany.sbatch
scripts/duca_unified_fullmatrix/submit_all.sh
tests/test_duca_unified_phase.py
tests/test_duca_unified_physical_time.py
tests/test_duca_unified_attribution.py
tests/test_duca_unified_mod.py
tests/test_duca_unified_curriculum.py
docs/experiments/DUCA_UNIFIED_FULLMATRIX_FREEZE.md
```

纯 helper 文件不能注册第二套 selector 或 decoder；它们只提供数学函数供现有类调用。

### 4. 绝对禁止改变

- THUMOS14 train/validation 视频身份；
- category map；
- official tIoU `[0.3,0.4,0.5,0.6,0.7]` evaluator；
- ActionFormer 分类/回归目标；
- NMS 算法和参数；
- `T=768, K=384`；
- 每个 arm 的 6000 次成功 optimizer update；
- terminal epoch-59 `state_dict_ema` 选择规则；
- 验证集不得用于选 epoch、选超参数或补训；
- 推理时不得使用 GT、annotation、teacher、oracle 或 raw-prediction cache。

---

## 五、机制的精确实现合同

### A. Robust semantic phase field

在 Scout/ASFormer 的原生时间网格上、FP32 中计算。`semantic` arm 使用 pre-sigmoid action logit `z(t)`；`motion` arm 只用于 Prior × Allocation 因果诊断，必须使用当前 legacy dual-phase 的同一低分辨率 temporal-motion score 作为源信号，随后通过完全相同的平滑、正/负变化、配额和 exact-K 代码路径。不得让 motion arm 偷用 ASFormer 特征，也不得让 semantic arm混入 RGB motion priority。对 motion arm，onset/offset 只解释为源信号的 rising/falling transition，不作为动作起止语义主张。

优先公式：

```math
z_\sigma(t)=G_\sigma*z(t),\quad \sigma\in\{1.5,3.0\}
```

```math
onset_\sigma(t)=\operatorname{ReLU}\left(\sigma(G'_\sigma*z)(t)\right)
```

```math
offset_\sigma(t)=\operatorname{ReLU}\left(-\sigma(G'_\sigma*z)(t)\right)
```

```math
curvature_\sigma(t)=\sigma^2\left|(G''_\sigma*z)(t)\right|
```

```math
core(t)=\sigma_{sigmoid}(z_\sigma(t))
```

实现要求：

1. Gaussian、first derivative、second derivative kernel 在 FP32 构造；
2. 离散 `G'` 强制奇对称、零和；`G''` 强制零均值；
3. valid-mask-aware smoothing；短序列反射 padding 不合法时退化为 replicate；
4. 每个场在有效区域用 detached q90/q95 或 MAD 做鲁棒归一化；禁止单峰 min-max 主导全部预算；
5. 多尺度使用预定义聚合，不根据 validation mAP 搜索尺度；建议 median 或 capped max，一旦选择写入 freeze 文档；
6. onset 与 offset 保持符号区分；
7. curvature 只能局部重排边界 shoulder，不得独占 128 帧；
8. camera cut 风险至少记录 scene-change proxy 与 selected ratio，不得把它静默解释为动作边界。

#### Fixed quota

```text
scaffold 128 + onset 64 + offset 64 + core 128 = 384
```

#### Adaptive quota

```text
minimum:
scaffold 96
onset    32
offset   32
core     64

remaining = 160
```

剩余预算按 detached robust evidence mass 通过有上限的 deterministic allocation 和 largest-remainder rounding 分配。配额和必须位置不足时由全局综合分 deterministic refill。

#### Selection order

```text
1. scaffold
2. onset temporal-NMS, excluding selected
3. offset temporal-NMS, excluding selected
4. core temporal-NMS/priority, excluding selected
5. global refill
6. existing global exact-K/max-gap contract
7. sort and assert
```

最终逐样本断言：

```text
selected_count == min(K, valid_len)
unique == true
strictly_increasing == true
all indices valid
no duplicate original timestamps
```

不得依靠给重复 timestamp 加 epsilon 掩盖上游重复选择。

### B. Physical-time propagation

复用 `ContinuousTimeScaleAdaptiveConv1d`，不得另写第二套 CT Conv。

1. 为每个 ActionFormer feature pyramid level 显式传播 `tau_l` 和 `valid_l`；
2. `tau_{l+1}` 必须由实际下采样算子的有效输入坐标生成，不能只硬编码 `2^l`；
3. local branch 使用恒定 physical span；
4. context branch 使用 scale-covariant span；
5. context residual gate 零初始化，初始时严格退化为 local/standard path；
6. geometry 可用于 `searchsorted`，但 selector 梯度继续走现有结构化 bridge；不要把离散 bracket 选择当稳定 selector gradient；
7. 超出时间边界的 tap 必须输出有效性诊断，不得静默把大量 tap 变成 zero-padding；
8. uniform timestamp 下必须与 reference Conv1d 达到预注册数值容差；
9. 该因子单独标记为 detector-representation change，不得归入纯 pre-backbone主结果。

### C. Detector-feature signed Taylor ranking

H65 旧 teacher 保持原样作为 D0。D1/F11 采用：

```python
grad = torch.autograd.grad(
    detector_objective.float(),
    feature_tokens,
    retain_graph=True,
    create_graph=False,
    allow_unused=False,
)[0]

with torch.no_grad():
    target = -(
        grad.detach().float() * feature_tokens.detach().float()
    ).sum(dim=1)
    target = target.relu()
```

要求：

- feature tap 固定为 P0+P1，不允许事后选择最佳层；
- cls/reg attribution 分别记录，再按冻结权重合并；
- target 完全 detach；不得产生 Hessian/double backward；
- 每 4 个成功 update 计算一次，其余步骤使用 EMA target；
- 监督使用 pairwise 或 listwise ranking，不直接拟合不稳定的绝对幅度；
- 保留 uniform companion 或分层探索，避免“只有已选帧才有 attribution”的自确认闭环；
- 在固定 train-only 小样本上执行 real legal one-swap，报告与真实 `ΔL` 的 Spearman；
- 只有相对旧 teacher 的 Spearman 至少提高 0.05 且最终检测也提高，才允许写 teacher-quality 主张。
- 本任务要求一次性部署，所以 one-swap 阈值是**证据准入门槛**而不是结果依赖的 Job 创建门槛；D1/F11 仍随完整 DAG 预先提交，最终若 gate 不过则其检测结果只能作为负面或诊断证据。

### D. Mixture-of-Depths

在 `vit_adapter.py` 当前 token depth routing 上就地修复，不得另建 MoE。

1. 名称统一为 Mixture-of-Depths；
2. `capacity=1.0` 必须与 Dense 数值等价；
3. 每个样本使用 `k_b=round(valid_token_count_b*C)`；padding token 被选比例必须为 0；
4. hard route 由 router logits Top-K 决定；temperature 只作用于 soft surrogate/entropy/consistency，文档明确它不改变单调 hard Top-K 排序；
5. 最终容量为 0.65，不在主矩阵直接使用 0.5；
6. capacity 由成功 update 调度：

```text
0–1500:    1.00
1500–3500: cosine 1.00 → 0.75
3500–5000: cosine 0.75 → 0.65
5000–6000: 0.65
```

7. 每 8 个成功 update 运行一次 Dense companion，做 feature/prediction consistency；
8. 至少记录每层实际容量、route churn/Jaccard、padding selection、每 token 执行深度、Dense/Sparse feature drift；
9. MoD 的准入是实际性能—成本 Pareto，不以提高 mAP 为必要条件。

### E. Curriculum

课程只由 checkpoint-persisted `global_successful_update` 驱动：

```text
default 20/20/20:
0–2000     uniform warmup
2000–4000  transition
4000–6000  joint

alternate 15/20/25:
0–1500     uniform warmup
1500–3500  transition
3500–6000  joint
```

AMP overflow、nonfinite loss、skipped optimizer step 不推进：

- curriculum；
- scheduler；
- EMA；
- Taylor period；
- MoD capacity/temperature。

checkpoint 必须保存和恢复：

- model；
- `state_dict_ema`；
- optimizer；
- scheduler；
- AMP scaler；
- `global_successful_update`；
- RNG；
- sampler epoch；
- curriculum state。

---

## 六、配置与任务表的生成

从随本任务提供的 `duca_unified_matrix_manifest.yaml` 生成 17 个薄配置，不复制整份大 base config。每个 config 必须在顶层显式写入：

```python
experiment_id
arm_id
factor_levels
seed
max_successful_updates = 6000
terminal_state_key = "state_dict_ema"
terminal_epoch = 59
```

生成脚本必须输出并提交：

```text
configs/adatad/thumos/duca_unified_fullmatrix/*.py
scripts/duca_unified_fullmatrix/matrix.tsv
scripts/duca_unified_fullmatrix/matrix.json
```

`matrix.tsv` 固定 41 行任务，列至少包括：

```text
array_index
phase                 # development or confirmation
arm_id
seed
config_path
run_relpath
```

运行：

```bash
python -m tools.bata.generate_duca_unified_fullmatrix \
  --manifest docs/experiments/duca_unified_matrix_manifest.yaml \
  --output-config-dir configs/adatad/thumos/duca_unified_fullmatrix \
  --output-matrix-tsv scripts/duca_unified_fullmatrix/matrix.tsv \
  --output-matrix-json scripts/duca_unified_fullmatrix/matrix.json
```

如果仓库约定要求 manifest 放在其他路径，可以复制一份进入 `docs/experiments/`，但机器清单内容必须字节一致并记录 SHA-256。

---

## 七、必须完成的测试与真实后端门禁

### 1. Phase tests

```text
test_constant_logit_has_zero_derivatives
test_step_logit_produces_signed_onset_and_offset
test_curvature_is_shoulder_not_independent_quota
test_masked_smoothing_ignores_padding
test_phase_selector_exact_k_unique_sorted
test_phase_selector_no_action_falls_back_to_coverage
test_phase_selector_short_window_uses_min_k_valid
test_adaptive_quota_sum_and_bounds
test_no_duplicate_timestamp_epsilon_repair
test_no_inference_gt_or_annotation_access
```

### 2. Physical time tests

```text
test_uniform_grid_matches_reference_conv
test_timestamp_pyramid_matches_feature_lengths
test_physical_to_index_roundtrip
test_no_double_stride_scaling
test_endpoint_invalid_tap_accounting
test_duplicate_timestamp_rejected
test_affine_time_rescaling_contract
```

### 3. Attribution tests

```text
test_taylor_create_graph_false
test_taylor_target_detached
test_signed_deletion_matches_linear_finite_difference
test_no_parameter_grad_pollution
test_period_uses_successful_steps
test_uniform_companion_exploration_exists
test_one_swap_uses_train_only_and_legal_replacement
```

### 4. MoD tests

```text
test_capacity_one_matches_dense
test_padding_tokens_never_selected
test_per_sample_valid_capacity
test_temperature_changes_soft_surrogate_not_hard_rank
test_capacity_resume_exact
test_dense_companion_period
test_measured_route_capacity
```

### 5. Curriculum tests

```text
test_amp_skip_does_not_advance_successful_step
test_scheduler_and_ema_advance_only_on_success
test_resume_reproduces_all_schedules
test_default_20_20_20_boundaries
test_alternate_15_20_25_boundaries
test_terminal_update_exactly_6000
test_no_external_thumos_stage1_checkpoint
```

### 6. Config/fairness tests

对 17 个 arm 全部检查：

- T/K；
- split；
- detector/head/loss/NMS/evaluator；
- pretrain；
- update count；
- terminal checkpoint；
- only intended factor differences；
- seed binding；
- no validation-best selection。

### 7. 运行命令

```bash
python -m py_compile \
  opentad/models/selectors/duca_online_frame_selector.py \
  opentad/models/duca/acquisition.py \
  opentad/models/duca/phase_fields.py \
  opentad/models/duca/feature_attribution.py \
  opentad/models/bricks/scale_adaptive_conv1d.py \
  opentad/models/backbones/vit_adapter.py \
  tools/bata/generate_duca_unified_fullmatrix.py \
  tools/bata/aggregate_duca_unified_fullmatrix.py \
  tools/bata/bootstrap_duca_unified_fullmatrix.py

python -m pytest \
  tests/test_duca_unified_phase.py \
  tests/test_duca_unified_physical_time.py \
  tests/test_duca_unified_attribution.py \
  tests/test_duca_unified_mod.py \
  tests/test_duca_unified_curriculum.py \
  -q

git diff --check
bash -n scripts/duca_unified_fullmatrix/*.sh
bash -n scripts/duca_unified_fullmatrix/*.sbatch
```

然后使用 Slurm 提交一个代表性真实 CUDA preflight，不在登录节点运行 GPU：

```text
U0, H0, A11, C11, D1, E01, F11
```

每个代表 config 至少完成：

- real data loader；
- real model construction；
- public pretrain load；
- 2 个成功 forward/backward optimizer update；
- selector exact-K checks；
- detector loss finite；
- expected modules receive/non-receive gradients；
- one terminal-eval dry binding without full inference。

Preflight 通过后 train array 自动由 `afterok` dependency 释放；不要人工逐臂提交。

---

## 八、独立 Builder / Critic / Evaluator 执行边界

### Builder

当前主 Agent 作为 Builder 完成最小实现和测试。完成后形成一个候选 commit，但暂不提交正式训练。

### Independent Critic

必须启动全新无实现上下文、只读的独立审查。给它：

- 本任务科学冻结；
- exact candidate commit；
- `git diff <base>..<candidate>`；
- 17 configs 和 matrix manifest；
- 聚焦测试结果。

Critic 只审查会改变以下事项的缺陷：

- phase 数学；
- exact-K/时间坐标；
- 梯度归属；
- strict-S60 公平性；
- GT/teacher/cache 泄漏；
- detector/loss/NMS/evaluator；
- MoD 有效容量与 Dense parity；
- Taylor 是否构建二阶图；
- 实际可运行性。

不因代码风格、额外日志或通用完备性阻塞。若 Critic 返回确定性 blocker，集中最小修复一次，再由新的独立 Critic 复审一次；不得形成无限循环。若剩余问题改变科学定义，停止部署并报告。

建议命令：

```bash
codex exec --ephemeral --sandbox read-only \
  "Independently audit the fixed DUCA unified-matrix candidate described in the supplied brief and diff. Return PASS or BLOCK with exact file:line findings only for mechanism, gradient, fairness, leakage, evaluation, or executability defects." \
  < docs/experiments/DUCA_UNIFIED_FULLMATRIX_CRITIC_INPUT.md \
  > docs/experiments/DUCA_UNIFIED_FULLMATRIX_CRITIC_REPORT.md
```

### Independent Evaluator

Critic PASS 后，启动另一全新只读上下文检查：

- THUMOS14 200/211 identity；
- class map、annotation、video root；
- canonical pretrain；
- 41-task matrix；
- 6000 successful updates；
- terminal EMA；
- official evaluator/NMS；
- prediction save paths；
- cost口径；
- bootstrap pairing；
- Slurm array和dependency。

Evaluator 只返回 `ELIGIBLE` 或 `BLOCKED`，并绑定 exact commit。确定性路径问题集中修复后只需一次重新核验；不得修改模型或科学协议。

---

## 九、远端环境与一次性 Slurm DAG

### 1. N16R4 固定环境

```bash
BASE=/data/run01/sczc063/yuzibo
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"

export HOME="$BASE/tmp/home"
export XDG_CACHE_HOME="$BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$BASE/tmp/xdg_config"
export HF_HOME="$BASE/hf_cache"
```

数据：

```text
$BASE/thumos14/annotations/thumos_14_anno.json
$BASE/thumos14/annotations/category_idx.txt
$BASE/thumos14/raw_data/video
$BASE/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
```

Slurm shell 必须在 `set -u` 和 `module load` 之前：

```bash
source /etc/profile
```

不得固定物理 GPU index，不得覆盖 Slurm 的 `CUDA_VISIBLE_DEVICES`；单卡进程使用 `cuda:0`。

### 2. 干净远端快照

候选 commit 通过两项独立核验后：

1. commit 所有实现、配置、测试、freeze 文档和脚本；
2. `git status --porcelain` 必须为空；
3. push 到新分支；若既有安全机制阻止 push，但远端有授权写入边界，则生成 Git bundle，传到 N16R4 后以同一 SHA 导入；
4. 远端 checkout 必须验证 exact SHA 和 clean tree；
5. 正式训练只能从该只读快照运行。

远端路径：

```bash
FINAL_SHA=$(git rev-parse HEAD)
REMOTE_REPO="$BASE/projects/opentad_duca_unified_${FINAL_SHA:0:12}"
RUN_ROOT="$BASE/experiments/duca_unified_fullmatrix_${FINAL_SHA:0:12}_$(date +%Y%m%d_%H%M%S)"
```

### 3. 一次性 DAG 结构

`submit_all.sh` 必须一次执行并提交：

```text
PREFLIGHT_GPU
      │ afterok
      ▼
TRAIN_EVAL_ARRAY[0-40]%MAX_CONCURRENT
      ├────────────── afterok ──────────────┐
      ▼                                      ▼
COST_ARRAY[0-4]                         BOOTSTRAP_ARRAY[0-15]
      └──────────────── afterok ─────────────┘
                              ▼
                         FINALIZER

TRAIN/COST/BOOTSTRAP ── afterany ── AUDIT_AFTERANY
```

- `TRAIN_EVAL_ARRAY` 每个 task 在同一 GPU allocation 内顺序执行完整训练、terminal EMA 官方 211-video inference 和原始 prediction 保存，避免 element-wise dependency 错配。
- `COST_ARRAY` 只测 `U0,A11,C11,E01,F11` 的 seed `4407` terminal checkpoint；每臂 3 次重复。
- `BOOTSTRAP_ARRAY` 共 16 shards，合计 10,000 次；必须以 video 为 cluster，并在确认 seed 和视频两个层级做 hierarchical resampling。
- `FINALIZER` 只在所有必要产物完整时产生科学汇总。
- `AUDIT_AFTERANY` 无论成败都生成作业状态、首个失败、缺失产物和证据有效性报告；它不得用残缺输出给出方法结论。

### 4. 每个 train/eval task 必须写出的产物

```text
<run>/identity.json
<run>/config_resolved.py
<run>/environment.txt
<run>/git_identity.txt
<run>/train.log
<run>/successful_updates.json
<run>/checkpoint/epoch_59.pth
<run>/terminal_eval/metrics.json
<run>/terminal_eval/result_detection.json
<run>/terminal_eval/prediction_identity.json
<run>/selector_diagnostics.json
<run>/physical_time_diagnostics.json       # applicable or explicit disabled
<run>/attribution_diagnostics.json         # applicable or explicit disabled
<run>/mod_diagnostics.json                 # applicable or explicit disabled
<run>/resource_usage.json
```

每条 identity 至少包含：

- final commit；
- config SHA；
- arm/seed；
- dataset/annotation/category/pretrain SHA；
- exact video count；
- terminal checkpoint SHA；
- evaluator source SHA；
- NMS config；
- successful updates；
- start/end timestamps；
- Slurm Job/array task ID。

### 5. 提交入口

Agent 实现完成后必须实际执行：

```bash
bash scripts/duca_unified_fullmatrix/submit_all.sh \
  --repo-root "$REMOTE_REPO" \
  --revision "$FINAL_SHA" \
  --run-root "$RUN_ROOT" \
  --base "$BASE" \
  --account sczc063 \
  --partition gpu \
  --qos normal \
  --max-concurrent "${DUCA_MATRIX_MAX_CONCURRENT:-8}"
```

只有 `sbatch` 返回有效 Job ID 才能写“已部署”。`submit_all.sh` 必须原子写入：

```text
$RUN_ROOT/submission_manifest.json
$RUN_ROOT/matrix.tsv
$RUN_ROOT/scientific_freeze.md
$RUN_ROOT/source_manifest_sha256.txt
```

`submission_manifest.json` 必须含：

```text
preflight_job_id
train_eval_array_job_id
cost_array_job_id
bootstrap_array_job_id
finalizer_job_id
audit_afterany_job_id
final_commit
remote_repo
run_root
submission_argv
```

---

## 十、终态评测、统计与分析

### 1. 检测指标

每个 run 必须报告：

- mAP@0.3/0.4/0.5/0.6/0.7；
- Avg-mAP；
- duration quartile AP；
- onset MAE；
- offset MAE；
- normalized boundary error；
- proposal recall。

### 2. Selector 指标

- onset/offset BoundaryHit@±2/±4/±8；
- both-endpoint recall；
- action interior coverage；
- background budget ratio；
- maximum/p95 physical gap；
- phase requested and actual quota；
- duplicate/refill count；
- scene-cut selected ratio；
- semantic-vs-motion selection Jaccard。

### 3. CT 指标

- per-level physical span；
- offset mean/p95/max；
- invalid/clamped/zero-padding tap ratio；
- uniform-grid parity error；
- `tau→u→tau_hat` roundtrip error。

### 4. Taylor 指标

- old/new teacher 与 legal one-swap `ΔL` Spearman；
- cls/reg attribution；
- target temporal variation；
- selected/exploration distribution；
- VJP wall time and peak memory。

### 5. MoD/成本指标

- per-layer actual capacity；
- padding-selection ratio；
- route Jaccard/churn；
- token execution depth；
- Dense/Sparse feature cosine/KL；
- scout/selector/backbone/detector/total latency；
- peak memory；
- measured backbone work；
- end-to-end throughput。

### 6. 统计

开发 seed 只用于机制地图，不进入 final seed mean。

确认阶段：

- 每个确认 arm 使用 `4407,5407,6407`；
- 报告 mean ± SD；
- 对同 seed 的候选/控制按 video ID 配对；
- 10,000 次 hierarchical seed+video bootstrap；
- video identity 必须 211/211 且完全相同；
- 不把重叠窗口视作独立样本；
- primary contrast 为 `A11−A10` 的 Avg-mAP；
- mAP@0.7 和其他机制 contrasts 为 secondary；
- 不允许事后改 primary metric 或阈值。

### 7. Finalizer 输出

```text
$RUN_ROOT/final/results_all_runs.csv
$RUN_ROOT/final/terminal_metrics_by_arm_seed.csv
$RUN_ROOT/final/panel_factor_effects.csv
$RUN_ROOT/final/hierarchical_bootstrap.json
$RUN_ROOT/final/selector_mechanism_table.csv
$RUN_ROOT/final/physical_time_table.csv
$RUN_ROOT/final/attribution_alignment_table.csv
$RUN_ROOT/final/cost_pareto_table.csv
$RUN_ROOT/final/claim_adjudication.md
$RUN_ROOT/final/evidence_validity.md
$RUN_ROOT/final/job_and_artifact_audit.json
```

`claim_adjudication.md` 必须分别裁决：

1. primary phase claim；
2. semantic-prior recovery；
3. physical-time representation；
4. Taylor teacher refinement；
5. MoD efficiency Pareto；
6. schedule compression；
7. 哪些结果只能作为单 seed exploratory evidence。

---

## 十一、预登记成功、未决和反驳标准

### Primary phase support

同时满足：

```text
mean(A11 − A10) Avg-mAP >= +0.30 pp
three confirmation seeds all positive
hierarchical 95% CI lower bound > 0
mean mAP@0.7 drop no worse than -0.10 pp
```

否则：

- mean positive但 CI跨0：未决/有希望，不写稳定增益；
- mean≤0，或至少2/3 seed为负且无预登记高tIoU补偿：反驳当前 phase mechanism。

### Physical-time support

```text
C11 − A11 Avg-mAP >= +0.20 pp
```

或：短持续时间/高 tIoU 显著改善且总体 Avg-mAP 不低于 `-0.10 pp`。否则不进入主方法。

### Taylor support

同时满足：

```text
Spearman improvement over old teacher >= 0.05
D1 − A11 Avg-mAP >= +0.20 pp
confirmation seed directions consistent
```

### MoD efficiency support

同时满足：

```text
Avg-mAP loss <= 0.30 pp
measured backbone work reduction >= 25%
measured end-to-end latency improvement >= 10%
```

若仅理论 capacity 下降而实际吞吐未改善，不成立效率主张。

### 历史分数

`A11 >=65.13`、`>=65.385724` 或 `>=66` 只作描述。真正论文证据来自 strict-S60 matched contrasts，不能用跨协议历史阈值替代。

---

## 十二、失败分类与自动停止边界

### 可最小修复后继续

- shell bootstrap；
- module 未加载；
- Python module invocation；
- 路径多一层 `gpu1_id0/`；
- Slurm dependency/array index；
- manifest 写出；
- 无模型语义变化的 import/runtime compatibility。

上述错误不得登记新模型版本，也不得解释成方法失败。

### 必须停止部署并报告

- 无法保证 strict total 6000 updates；
- 需要隐藏 THUMOS Stage-1 checkpoint；
- split/video identities 不一致；
- validation/test GT、teacher 或 raw prediction 进入 selector；
- exact-K、original-time inverse map 或 NMS 顺序改变；
- detector/loss/evaluator 不匹配；
- capacity=1 无法与 Dense parity；
- Taylor 意外构建二阶图；
- 需要改变本任务已冻结 factor definition 才能运行。

### 科学结果不是工程失败

完整合法 run 得到负 mAP，必须保留并进入 final table；不得通过换 seed、挑 epoch、降低门槛、删除 arm 或补训来“修复”。

---

## 十三、你当前必须执行的完整命令序列

在本地/工作站：

```bash
set -euo pipefail
REPO="${DUCA_REPO:?set DUCA_REPO}"
cd "$REPO"

git fetch --all --tags --prune
git worktree add \
  "../OpenTAD_DUCA_UnifiedFullMatrix_20260902" \
  95ca6eb4a7e0ba8259c5afd976cc30d0fea58865
cd "../OpenTAD_DUCA_UnifiedFullMatrix_20260902"
git switch -c codex/duca-unified-fullmatrix-20260902

# Read exact history before editing.
git show --stat --oneline 04c35a3b76897e6c1569eeede41ed3aecaf7f854
git show --stat --oneline c73e8418de31cdcb2a445ff58a1e33ab9ab6a508
git show --stat --oneline 7bd120f0d342bf175c97c365fba7cbd359df055e

# Implement all factors/configs/scripts/tests described above.
# Copy the supplied manifest into docs/experiments byte-for-byte.

python -m tools.bata.generate_duca_unified_fullmatrix \
  --manifest docs/experiments/duca_unified_matrix_manifest.yaml \
  --output-config-dir configs/adatad/thumos/duca_unified_fullmatrix \
  --output-matrix-tsv scripts/duca_unified_fullmatrix/matrix.tsv \
  --output-matrix-json scripts/duca_unified_fullmatrix/matrix.json

python -m pytest \
  tests/test_duca_unified_phase.py \
  tests/test_duca_unified_physical_time.py \
  tests/test_duca_unified_attribution.py \
  tests/test_duca_unified_mod.py \
  tests/test_duca_unified_curriculum.py \
  -q

git diff --check
bash -n scripts/duca_unified_fullmatrix/*.sh
bash -n scripts/duca_unified_fullmatrix/*.sbatch

# Builder candidate commit.
git add opentad configs tools scripts tests docs
git commit -m "feat(duca): implement strict-s60 unified full experiment matrix"
CANDIDATE_SHA=$(git rev-parse HEAD)

# Create and execute independent Critic and Evaluator briefs.
# Apply at most one concentrated deterministic repair batch, then fresh re-review.

# Final frozen commit after PASS/ELIGIBLE.
test -z "$(git status --porcelain=v1)"
FINAL_SHA=$(git rev-parse HEAD)
git push -u origin codex/duca-unified-fullmatrix-20260902

# Build/verify authorized N16R4 clean snapshot, then submit the whole DAG once.
bash scripts/duca_unified_fullmatrix/submit_all.sh \
  --repo-root "/data/run01/sczc063/yuzibo/projects/opentad_duca_unified_${FINAL_SHA:0:12}" \
  --revision "$FINAL_SHA" \
  --run-root "/data/run01/sczc063/yuzibo/experiments/duca_unified_fullmatrix_${FINAL_SHA:0:12}_$(date +%Y%m%d_%H%M%S)" \
  --base "/data/run01/sczc063/yuzibo" \
  --account sczc063 \
  --partition gpu \
  --qos normal \
  --max-concurrent "${DUCA_MATRIX_MAX_CONCURRENT:-8}"
```

如果本机不具备 N16R4 shell，使用已授权的现有 SSH/remote execution 路径执行远端快照和最后一条 `submit_all.sh`；不要在代码或日志中写入凭据。若无授权连接，完成所有本地实现和审查后返回 exact blocker 与唯一待执行的远端命令，不得伪造 Job ID。

---

## 十四、最终 Agent 回复格式

最终回复必须包含，且只能使用实际证据填写：

1. `status`: `DEPLOYED`、`IMPLEMENTED_NOT_DEPLOYED` 或 `BLOCKED`；
2. integration base；
3. final branch 和 exact commit；
4. modified/added files；
5. matrix count：17 development、24 confirmation、41 total；
6. focused tests、Critic verdict、Evaluator verdict；
7. remote clean snapshot；
8. run root；
9. 六个真实 Slurm Job/Array ID；
10. exact submission argv；
11. expected terminal artifacts；
12. blockers；
13. 明确说明尚未产生的 mAP/统计结果不能被声称。

不要以“下一轮再实现某模块”结束；本任务要求所有矩阵代码、配置和部署入口在一个精确提交中完整形成。若真实权限允许，必须完成一次性 Slurm DAG 提交；若权限不允许，必须把唯一剩余动作压缩为一条可复制执行的远端命令。

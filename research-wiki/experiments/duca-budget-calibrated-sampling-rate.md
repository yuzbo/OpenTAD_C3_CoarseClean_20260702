---
title: DUCA budget-calibrated sampling-rate curriculum
status: experiment_running
updated: 2026-07-23
---

# Budget-Calibrated Sampling Rate

## Purpose

This is a single offline-TAD pre-backbone acquisition candidate, not a second
detector or a separate deployment pipeline. It predicts a bounded per-frame
retention rate, calibrates the rates to exact-K, and uses deterministic
cumulative thresholds to select the K original-time observations. The hard
decoder has no learned Top-K operation, no duplicate repair, and no mandatory
local-cell rule.

## Training ownership

The curriculum keeps exact-uniform observations during the early phase. The
coarse action BCE continues to train the low-resolution spatial stem, all
ASFormer encoder layers, and the action decoder/head. In the sampling phase,
uniform detector rows provide train-only classification and regression
contribution targets. These targets supervise the shared transition/utility
head, while the real TAD objective follows the hard-forward, soft-backward
sampling-rate bridge.

The variants are intentionally nested:

| Variant | Detector-contribution target | ASFormer route for TAD/utility adaptation |
| --- | --- | --- |
| rate-only | none | none |
| rate-cls / rate-reg / rate-both | cls / reg / both | none |
| rate-both-last | both | final ASFormer encoder layer |
| rate-both-full | both | all ASFormer encoder layers |

`rate-both-full` is the requested full dual-supervision variant. Detector and
contribution gradients reach every ASFormer encoder layer through a replayed
hidden-feature bridge. They do not enter the spatial stem. This is deliberate:
the full coarse action objective still trains the spatial stem, encoder,
decoder and action head, whereas detector feedback should adapt the temporal
representation used by the selector rather than overwrite image-level action
semantics. The action decoder does not receive detector gradients because it is
not on the selector's causal input path.

## Exact implementation identity

- branch: `codex/duca-density-transport-20260723`
- implementation base: `685ebe106302e20bed9e933fa6a01945b0b72cc4`
- current exact gate commit: `5357ff1b06d661ebfa276b91d60fac769b35d4f9`
- GitHub: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-density-transport-20260723`
- remote snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_duca_rate_685ebe1_20260723`
- frozen VideoMAE SHA256: `4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251`

## Gate evidence

The first remote wrapper attempt `1181228` never executed Python or a model:
Slurm used `/bin/sh` for `--wrap`, which rejects Bash-only `source` and
`pipefail`; its dependent `1181229` was cancelled. This is a zero-update
deployment error, not an optimization or model result. The corrected Bash
submissions are:

| Job | Role | Dependency | Status at record time |
| --- | --- | --- | --- |
| 1181249 | focused sampling-rate / official60 tests | none | completed: 10 passed |
| 1181250 | real AdaTAD full-ASFormer gradient gate | afterok:1181249 | zero-update data-environment failure |
| 1181254 | real AdaTAD full-ASFormer gradient gate retry | none | zero-update data-environment failure |
| 1181256 | real AdaTAD full-ASFormer gradient gate with canonical environment | none | code-path failure before optimizer update |

The intermediate `1181234/1181235` pair belongs to the first-code revision:
the focused suite exposed zero-initialized utility logits that blocked the
first ASFormer contribution gradient, and the subsequent gate exposed two
legacy constructor restrictions that assumed `global_structured_topk`. The
current commit uses a small neutral utility initialization while retaining a
zero-initialized policy fusion layer, and admits the sampling-rate exact-K
bridge to the uniform-teacher path. Neither failed pair performed a valid
model optimizer update or produced an mAP result.

No terminal official THUMOS mAP exists for this candidate. Formal multi-arm
official-60 training is blocked only on these real-model contracts and one
independent mechanism review, not on another redesign.

## 2026-07-23 Fixed-small-sample visualization diagnostic

Commit `5357ff1b06d661ebfa276b91d60fac769b35d4f9` adds a deliberately small,
post-checkpoint diagnostic rather than another training path. It is
`implemented/tested` locally for JSON schema and actual PNG/PDF rendering, and
`experiment_running` for the first CUDA replay.

- **Fixed training evidence:** one deterministic train loader batch,
  `batch-index=0, batch-size=4`, replayed with `model.train()` and a backward
  pass but **no optimizer update**. Per sample it exports actionness,
  transition score, rate/density, hard selected positions, actual selected-RGB
  classification/regression input-times-gradient, its explicitly marked dense
  interpolation, and the absolute detector-loss gradient on rate logits.
- **Fixed validation evidence:** one deterministic validation batch,
  `batch-size=4, limit-batches=1`, uses selector-only `forward_test` with no
  GT passed to the decision. It records only inference-time rate/density,
  selected positions and boundary-neighbourhood evaluation overlays.
- **Epoch comparison:** records from the same four windows at checkpoints
  `0/10/20/30/40/50/59` produce individual four-lane pages plus one
  same-window overlay across epochs. GT remains labelled as train-loss or
  evaluation overlay only, never an inference-time selector input.

The fixed-window choice is an efficiency contract: it verifies that the
visualization is causal and stable without turning training diagnostics into a
second full-dataset experiment. It cannot establish mAP or replace the final
official validation.

The first wrapper `1181409` exited at Slurm shell setup because `module` was
not available before `/etc/profile` was loaded. It made zero Python/model/
optimizer updates and is not a model result. Replacement Job `1181418` is the
first combined focused-test and real full-ASFormer/AdaTAD gate for this exact
commit. Its root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_5357ff1_gate_20260723_1350`.
Only after it passes may the first real checkpoint drive the four-training and
four-validation sample diagnostic.

## 2026-07-23 14:10 gate failure classification

Jobs `1181250` and `1181254` did not source the canonical THUMOS environment,
so the dataset path resolved to a non-existent default. They made zero official
updates. Job `1181256` correctly reached the real data loader, VideoMAE,
AdaTAD forward pass and contribution-distillation path, then raised
`NameError: F is not defined` in the distribution loss before the first
optimizer update. Commit `e8a2fea3a6034ae51960aca230ec0fb6efd1aff3` adds the
missing `torch.nn.functional` import only; it changes neither the sampling
policy nor the loss formula. The next gate must run from this exact commit with
the canonical environment. These failures are implementation-contract evidence,
not numerical or mAP evidence.

## Current resubmission

The remote source snapshot is detached at
`e8a2fea3a6034ae51960aca230ec0fb6efd1aff3`. A minimal two-job GPU chain was
submitted without an explicit memory request, because the N16R4 site rejects
manual single-GPU memory overrides. Its root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_e8a2fea_gate_20260723_1418`:

| Job | Role | Dependency | Status at submission |
| --- | --- | --- | --- |
| 1181288 | focused sampling-rate / official60 tests | none | zero-update profile-order failure |
| 1181289 | real AdaTAD full-ASFormer gradient gate | afterok:1181288 | dependency never satisfied |
| 1181295 | focused sampling-rate / official60 tests | none | submitted with profile loaded before `set -u` |
| 1181296 | real AdaTAD full-ASFormer gradient gate | afterok:1181295 | zero-update truncated-commit contract failure |
| 1181304 | real AdaTAD full-ASFormer gradient gate | afterok:1181295 | submitted with the full 40-character commit |

This is still gate evidence only. Formal official-60 training remains blocked
on a passing full-model artifact and the independent mechanism review.

## 2026-07-23 14:00 - Every-five-epoch mAP matrix submitted

Commit `aa439fb2e2f3d29e3225c73b49f733ad93a40906` changes the sampling-rate
official-60 contract so that the real THUMOS validation evaluator runs at
one-based epochs `5, 10, ..., 60`. These are explicitly **diagnostic learning
curve points**: the only primary result remains the sealed terminal
`epoch_59.pth` `state_dict_ema`, so no intermediate validation result can be
used to select a checkpoint.

The earlier Job `1181418` stopped before a model forward/update because its
then-current config still declared 132 epochs while the `official60` validator
required 60. It is a zero-update configuration-contract failure, not evidence
about sampling quality.

The remote snapshot
`/data/run01/sczc063/yuzibo/projects/opentad_duca_rate_685ebe1_20260723` is
clean and detached at `aa439fb2e2f3d29e3225c73b49f733ad93a40906`. The four
formal arms were submitted independently, with no inter-arm dependency:

| Job | Arm | Variant | Root |
| --- | --- | --- | --- |
| 1181495 | strict matched uniform control | `sampling_rate_exact_uniform` | `duca_rate_aa439fb_every5_20260723_1400_final/exact_uniform` |
| 1181496 | learnable sampling rate only | `sampling_rate_only` | `duca_rate_aa439fb_every5_20260723_1400_final/rate_only` |
| 1181497 | rate plus detector cls/reg contribution distillation | `sampling_rate_both` | `duca_rate_aa439fb_every5_20260723_1400_final/contribution_both` |
| 1181498 | contribution distillation plus full ASFormer temporal adaptation | `sampling_rate_both_asformer_full` | `duca_rate_aa439fb_every5_20260723_1400_final/contribution_both_asformer_full` |

Every arm first runs the same real AdaTAD gate inside its own Slurm allocation,
then trains for 60 epochs, retains checkpoints every five epochs and records
the official evaluator trajectory. The preceding `1181482--1181485`,
`1181486--1181489`, and `1181491--1181494` submissions exited before Python:
respectively Slurm `/bin/sh` rejected `source`, nested argument quoting dropped
the variant, and pre-creating `ARM_ROOT` violated the launcher contract. They
are zero-update deployment history only. The final `1181495--1181498` jobs
are simultaneously `RUNNING` on four different GPU nodes. Status is
`experiment_running`; no mAP has been produced by this new matrix yet.

## 2026-07-23 14:06 Remote state correction

- The intended final four-arm matrix did not remain concurrent: `1181495` exact-uniform,
  `1181497` contribution-both, and `1181498` contribution-both-ASFormer-full ended
  `FAILED/1:0` within about one minute. `1181496` rate-only is still `RUNNING` on `g0045`.
  No terminal evaluation, checkpoint comparison, or mAP has been produced by any arm.
- This leaves only one live arm and no matched control. It is not a valid four-arm performance
  comparison; retain the route as `experiment_running / incomplete matrix / no_terminal_map`
  until the failure causes are audited and a valid matched protocol exists.

## 2026-07-23 14:12 Remote completion correction

- A subsequent remote `sacct` check records all four final-matrix jobs as failed: `1181495`
  exact-uniform (`1:0`, 56 s), `1181496` rate-only (`1:0`, 1 m 55 s), `1181497`
  contribution-both (`1:0`, 1 m 03 s), and `1181498` full-ASFormer contribution-both
  (`1:0`, 57 s). No arm has a terminal evaluation artifact or mAP.
- The sampling-rate matrix is therefore not executing and has produced no test result. Its next
  state is `deployment_failed / failure-cause-audit-required / no_terminal_map`, not an active
  performance comparison.

## 2026-07-23 14:25 Every-five-epoch curve and gradient-contract repair

- The intended official-60 output is now the complete validation trajectory at
  epochs `5, 10, ..., 60`, not terminal mAP alone. Every value is retained for
  a learning-curve plot and the final report will include both the peak epoch
  and the sealed epoch-59 EMA result. Validation mAP must never alter a
  training gradient; terminal EMA remains the matched primary comparison.
- Jobs `1181495--1181498` on `ff8dfb3` all failed in the real full-model gate
  before any official optimizer update or checkpoint. The failures exposed
  model-contract issues: transition loss did not demonstrably reach ASFormer,
  rate-only still constructed the contribution path, and full adaptation did
  not demonstrate detector-to-ASFormer gradient flow. They are not mAP data.
- Commit `3d133a08fe6a90fd2cd1426a78b0a17a87d2b348` makes rate-only a true
  ablation by omitting the contribution-logit computation entirely when its
  components are `none`. It also records the action/transition gradient
  partitions and the policy-hidden `requires_grad` state in the existing real
  CUDA gate. Local focused validation is `5 passed`.
- Job `1181556` is the sole real-CUDA diagnosis on that commit, using the
  ordinary official ASFormer coarse probe, VideoMAE AdaTAD detector, THUMOS
  loader and K=384 sampling-rate route. It is a gate-only job; its outcome
  decides the minimal remaining gradient-path correction before long training
  is resubmitted.

## 2026-07-23 14:40 - periodic validation and first-step transition repair

- Commit `d2fd58d531f81af24b0c77a235d61cea9200df29` supersedes the earlier
  diagnostic-only validation wording. The official THUMOS evaluator now runs
  after epochs `5, 10, ..., 60`, writes one immutable metric JSON per EMA
  checkpoint, and updates a small `best_validation_ema.json` pointer by
  `average_mAP`. The terminal epoch-59 EMA remains reported separately, so
  the curve exposes convergence and possible overfitting instead of hiding it.
  Evaluation is read-only: it changes no gradient, optimizer, scheduler, EMA,
  early-stop, or training-data decision.
- The same commit removes the zero initialization of the transition scorer
  output only for `budget_calibrated_sampling_rate`. The uniform-start policy
  is already guaranteed by `policy_alpha=0`; zero output initialization was
  therefore unnecessary and blocked transition-supervision gradients from
  reaching ASFormer on the first real update.
- Real CUDA gate `1181576` completed successfully on exact commit `d2fd58d`
  with the official THUMOS loader, VideoMAE AdaTAD, AMP replay, DDP wrapper,
  and one successful optimizer update. Its gradient partition confirms that
  transition-only supervision now reaches the earlier ASFormer encoder layers
  (`72.78` absolute-gradient sum) and the last layer (`29.55`), while the
  protected rate-only detector route still leaves ASFormer at zero. This is a
  model-path gate result, not a mAP claim. Status: `implemented/tested`; the
  re-submitted official-60 arms remain required for performance evidence.

## 2026-07-23 14:45 - exact-commit seven-arm learning-curve matrix

All arms were independently submitted from the clean N16R4 snapshot at
`d2fd58d531f81af24b0c77a235d61cea9200df29`; there are no inter-arm
dependencies. Their common root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_d2fd58d_every5_20260723_144503`.

| Job | Variant | Question |
| --- | --- | --- |
| 1181580 | exact uniform | matched detector/control curve |
| 1181581 | sampling rate only | whether learned retention rates help without contribution distillation |
| 1181582 | rate + cls contribution | contribution-classification ablation |
| 1181583 | rate + reg contribution | contribution-regression ablation |
| 1181584 | rate + cls/reg contribution | combined-distillation effect |
| 1181585 | combined + last ASFormer layer | narrow detector-adaptation effect |
| 1181586 | combined + full ASFormer | full dual-supervision effect |

At submission all seven jobs were `PENDING(Priority)`, which is a GPU-scheduler
state rather than an error. Each arm first re-runs its own config and real-model
gate, then performs 60 epochs with full official THUMOS validation at every
five epochs. The run-level `jobs.tsv` is the deployment receipt; no mAP exists
yet and the matrix remains `experiment_running`.

## 2026-07-23 15:18 - formal-routing correction and active seven-arm matrix

- The first seven jobs (`1181580--1181586`) all stopped before their first
  optimizer update. The sampling-rate configs had not declared the existing
  selected-axis formal protocol, so `tools/train.py` routed them to the legacy
  P0 validator, which rejected the otherwise valid independent variant name.
  This is a launcher/config-routing failure, not a numerical failure and not
  a performance result; it produced neither checkpoints nor mAP.
- Commit `9491ba1b09209ac58588f9ee777038d374b73c41` adds the existing
  `duca_selected_axis_optimization_v1` protocol declaration and maps all
  seven variants into the existing selected-axis runtime binder. Focused tests
  pass locally (`7 passed`) and on the exact Linux snapshot (`7 passed`). No
  selector, decoder, detector, loss, or training schedule was changed by this
  correction.
- Fresh independent jobs `1181593--1181599` are the only valid sampling-rate
  learning-curve matrix. They use the same exact commit, VideoMAE pretrain
  SHA-256 `4b96b7f4...e0de251`, 60 epochs, and official THUMOS validation at
  epochs `5, 10, ..., 60`. Run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_9491ba1_every5_20260723_151800`.
  `jobs.tsv` binds every job, variant and arm path. At deployment all seven
  were accepted by Slurm and were pending resource allocation; no mAP exists.

## 2026-07-23 16:30 - trained small-sample mechanism diagnostic

- Exact implementation commit: `2a86d15f8eead82cc4945899ab52a5f2c3366cd9`
  on `codex/duca-density-transport-20260723`. It does not introduce a new
  selector or decoder. It reuses the full-ASFormer, cls+reg-contribution,
  budget-calibrated sampling-rate model and only compresses its existing
  curriculum for a fast mechanism check.
- The diagnostic performs **real optimization**: 10 epochs x 4 actual train
  batches x batch size 1 = 40 optimizer updates. At one-based epochs 1, 5,
  and 10 it replays the same deterministic training window to export detector
  input-times-gradient, sampling-rate-logit gradients, predicted contribution,
  rate density, and selected positions. It also runs teacher-free selector
  inference on the same two validation windows for each checkpoint.
- GT is used only for the real training loss and for post-hoc overlays. It is
  never passed to the inference selector. The run produces figure-ready
  individual and across-epoch panels, but reports no mAP and cannot be used as
  official performance evidence.
- The attribution exporter now strips a uniform DDP `module.` checkpoint
  prefix before strict full-model loading, while rejecting mixed namespaces.
  This fixes a real diagnostic-only checkpoint-loading defect without changing
  the DUCA model or decision rule. Local and exact remote focused checks both
  pass (`15 passed`).
- Slurm Job `1181615` was submitted with one GPU. Source snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_mini_visual_2a86d15_20260723`.
  Evidence root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_2a86d15_mini_visual_20260723_1630`.
  Initial status is `PENDING(Priority)`, so no actual update or figure exists
  yet. Status: `implemented/tested/experiment_running`.

## 2026-07-23 16:41 - variable-window sampling-rate contract repair

- The initial trained mini-visual Job `1181615` and formal sampling-rate jobs
  `1181593--1181599` exposed a shared decoder-shape error before a usable
  optimizer update or mAP. Randomly truncated valid windows can be shorter
  than the padded 768-frame batch. Hard occupancy was padded, but soft
  occupancy retained valid-window length, so straight-through occupancy failed
  on tensors such as 768 versus 417 frames.
- Commit `dea59d228bee7bc8c78d29187a1cfd9bdcb987b2` pads soft occupancy on the
  same temporal axis as hard occupancy and adds a mixed-valid-length gradient
  test. Valid positions preserve their rate behavior; padded positions are
  exactly zero and receive no gradient. This is a decoder-contract correction,
  not a sampling-policy or loss redesign.
- These failed runs are implementation failures, not mAP or model-performance
  evidence. Exact Linux Torch verification and a fresh real-update mini-visual
  submission are pending.

## 2026-07-23 16:05 - repaired trained mini-visual submission

- Exact Linux verification of `dea59d2` passed (`14 passed in 42.78s`) across
  the rate decoder, mini-visual config, and official-60 contract tests.
- Fresh Slurm Job `1181648` was submitted independently from clean snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_rate_dea59d2_20260723` at
  exact commit `dea59d228bee7bc8c78d29187a1cfd9bdcb987b2`. Its evidence root
  is `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_dea59d2_mini_visual_20260723_1540`.
  It remains the same 40-update trained mechanism diagnostic, with fixed
  checkpoint visualizations at one-based epochs 1, 5, and 10; it does not
  produce official mAP.

## 2026-07-23 16:47 - uint8 contribution-teacher repair and resubmission

- Real THUMOS RGB observations are stored as `uint8`. Job `1181648` reached
  the first real training batch but failed before any optimizer update because
  the training-only contribution teacher requested `requires_grad_(True)` on
  an integer observation tensor. This is an Autograd dtype error, not a
  detector, selector, loss, or mAP result.
- Commit `596d9829dff4999d1e5d3e9028f4ab9310ea2bfc` promotes only the
  identity-valued contribution-teacher view to `float32`, retaining identical
  numeric RGB values and the straight-through route to the sampling policy.
  It never changes stored video frames and is absent at inference. The exact
  remote focused mini-visual suite passed (`3 passed in 37.31s`), including a
  regression proving finite contribution gradients on a `uint8` observation.
- Fresh Job `1181671` is independently queued from the same clean snapshot at
  exact commit `596d9829dff4999d1e5d3e9028f4ab9310ea2bfc`; evidence root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_596d982_mini_visual_20260723_154700`.
  It remains a 40-update trained mechanism diagnostic with fixed-window plots
  at epochs 1/5/10, not an official TAD mAP experiment.

## 2026-07-23 16:52 - trained checkpoints retained; visualization recovery

- Job `1181671` completed all forty real optimizer updates and wrote the
  required epoch 1/5/10 checkpoints. Its post-training attribution exporter
  then failed because it reconstructed the detector from the relative
  VideoMAE path in the config rather than the absolute path used at training.
  This is a diagnostic-launcher defect after training, not a failed update,
  non-finite collapse, or mAP result.
- Commit `5029691ee0e95d6adeb2dc165325de1e335084c9` adds one explicit,
  narrow `--backbone-pretrain` option to the full-detector attribution exporter.
  The selector-only inference exporter is unchanged because it does not build
  VideoMAE. Job `1181683` failed before reading a checkpoint because Slurm
  `--wrap` invoked `/bin/sh` while the recovery command used Bash `source`.
  It is a zero-work shell error. Job `1181684` is the corrected Bash read-only
  recovery from the three existing checkpoints into
  `.../duca_rate_596d982_mini_visual_20260723_154700/postprocess_5029691`.
  It exports contribution/gradient, sampling-rate, selected-position and
  boundary-neighborhood plots without rerunning optimization.
- Job `1181684` was another zero-work Slurm quoting failure. Active Job
  `1181685` is the same read-only recovery expressed as a single Bash command;
  it is running from the exact `5029691` snapshot. It is the only recovery job
  whose artifact state may become evidence.

## 2026-07-23 训练中贡献/梯度可视化 v2

- 精确实现提交为
  `codex/duca-density-transport-20260723@0b7e07584779651196f1560a90c4dc40744ed22c`。
  这是现有 sampling-rate 训练路径的轻量诊断升级，状态为
  `implemented / remote_focused_test_pending`，不是新的训练框架、不是 official mAP
  试验，也不改变 selector、decoder 或检测头决策。
- 诊断固定两条训练窗口和两条验证窗口：小样本实训 30 epoch、每 epoch 4 个真实 update，
  在 10/20/30 epoch 导出同一批训练窗口和验证窗口。训练记录真实 AdaTAD 检测头输入的
  多尺度时序特征 `|F*dL/dF|`、`|dL/dF|` 与采样率 logit 的 `|dL/dz|`；验证只运行
  teacher-free selector inference，GT 仅在图中作事后边界叠加。
- 原先的 selected-RGB `|x*dL/dx|` 被明确降为辅助像素敏感性，而非主任务贡献：最后一层
  参数梯度本身没有逐时间点归因能力，主图必须使用每个时序特征激活与检测损失梯度的乘积。
  图形导出同时生成单 epoch 页面和同一窗口跨 10/20/30 epoch 的可叠加曲线；验证图输出采样率、
  选中位置 raster 与到最近边界的距离分布。
- 现有 40-update Job `1181671` 的 checkpoint 只能作为旧 v1 的恢复来源；v2 的 120-update
  小诊断须在该精确提交通过远端 focused checks 后单独 Slurm 提交。它只用于解释训练机制，
  不能用中间图挑选 official checkpoint 或主张 mAP 增益。

## 2026-07-23 30-epoch 图形诊断已提交

- `0b7e075` 的远端 focused 检查为 `12 passed in 104.84s`，并完成 py_compile 与 Bash 语法检查。
  Windows 本机 PyTorch 因 `c10.dll` 初始化失败无法收集测试，不作为代码失败证据。
- Slurm Job `1181894`（`duca-ratevis30`）已在 `n16r4` 提交，精确 commit 为
  `0b7e07584779651196f1560a90c4dc40744ed22c`，快照为
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_attribution_d844c18_20260723`，
  产物根为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_0b7e075_mini_visual_20260723_1635`。
  它不依赖 R 系列作业、没有 official mAP 输出；预期保存 10/20/30 epoch 的两条训练窗口
  贡献/梯度图与两条验证窗口采样图。状态：`experiment_running`。

## 2026-07-23 16:38 图形诊断运行健康检查

- `1181894` 已实际获得 GPU 并推进到 epoch 7；训练日志有有限的 `cls_loss/reg_loss`，显存约
  `5.7GB`，没有 Traceback、OOM、ValueError、non-finite collapse 或 AMP replay 耗尽。
  epoch 3 的一个 batch 发生两次 AMP replay 后成功完成，属于已恢复的孤立数值事件。
- 当前仍在课程前段：actionness 与 transition 监督占主导，采样率混合权已从约 `0.0011` 升至
  `0.2222`，检测梯度权重刚开始从零升起（约 `0.0024`）。日志中的聚合字段
  `detector_loss=0.0000` 不代表 `cls_loss/reg_loss` 为零；后两项已为有限非零值。此时尚不应
  解读贡献图或判断模型优劣，首个可比可视化产物在 epoch 10 后生成。

## 2026-07-23 16:43 - attribution v2 checkpoints and joint-stage health

- Lightweight diagnostic Job `1181894` at exact commit
  `0b7e07584779651196f1560a90c4dc40744ed22c` has written
  `epoch_9.pth` and `epoch_19.pth`, which are the planned epoch-10 and
  epoch-20 visualization checkpoints. It is now continuing through epoch 26
  toward the epoch-30 checkpoint.
- The current schedule is `joint_transition_detection`: policy mixing alpha is
  `1.0`, and the detector-to-sampling bridge weight is `0.25`. `cls_loss`,
  `reg_loss`, actionness, and transition losses remain finite and nonzero;
  GPU memory is stable at about 5.8 GB. One AMP replay at epoch 23 recovered
  within the allowed retry budget. No traceback, OOM, or non-finite collapse
  was found.
- Postprocessing deliberately occurs after the small training run completes,
  then exports the same two training windows and two teacher-free validation
  windows for epochs 10/20/30. This avoids an additional training framework
  and guarantees directly comparable axes. `duca_effective_budget_mean < 384`
  on a few padded short windows denotes `min(K, valid_length)`, not a global
  sampling-rate budget collapse.

## 2026-07-23 16:48 - first visualization audit and read-only repair

- Job `1181894` completed all 120 real optimizer updates and retained all
  three checkpoints (`epoch_9/19/29.pth`). Its first two training attribution
  JSONL files and epoch-10 teacher-free selection JSONL are valid. The job
  then stopped in the obsolete `analyze_duca_selection_quality` summary step:
  a short window without valid boundaries reached a legacy `None - None`
  statistic. This is a post-training visualization-launch error, not a model,
  numerical, or mAP failure.
- A direct rendering of the retained epoch-10 records confirms that primary
  detector-head feature contributions are localized in the action-region
  neighborhoods, so the attribution plot itself is functional. However, the
  first two validation windows have `valid_len=256 < K=384`; therefore every
  valid point is selected and the sampling-rate plot is identically one. They
  are invalid witnesses for non-uniform sampling and must not be used in a
  paper figure.
- Commit `5459275ddb3e231467f7769da2737574fc3a8973` removes the unused legacy
  summary call and makes fixed validation evidence explicit: collect exactly
  two windows with `valid_len >= 384` and at least one valid GT boundary.
  Read-only recovery Job `1182026` uses the existing three checkpoints and
  writes a separate `postprocess_5459275` root. It repeats no optimizer update
  and remains `experiment_running`; no visual mechanism claim is sealed until
  the three-epoch overlay is available.

## 2026-07-23 17:10 - density-transport visualization correction

- Recovery `1182026` failed before Python or checkpoint access: its strict
  shell mode conflicted with an unset `XDG_DATA_DIRS` referenced by the site
  `/etc/profile`. It is a zero-work launcher error, not model evidence.
- Commit `4253639f692daa24ac07aeefc9b3b21070c819e9` corrects the diagnostic
  itself. For density transport, detector loss reaches acquisition through
  `density_probabilities` and the soft `[K,T]` slot assignment, not necessarily
  through the hard decoder's `decode_policy_logits`. The exporter now records
  `|dL/d density|`, slot-summed `|dL/d assignment|`, and `|dL/d center_score|`;
  legacy logit gradients are retained only for compatibility. The remote
  PyTorch focused suite completed successfully on this exact commit.
- Fresh read-only recovery Job `1182079` reuses only `1181894` checkpoints and
  writes `postprocess_4253639`. It filters validation examples to
  `valid_len >= 384` with valid GT boundaries. It is `experiment_running` and
  must still produce finite nonzero detector-to-density gradients plus all
  epoch-10/20/30 figures before the diagnostic can be called successful.

## 2026-07-23 17:18 - coarse actionness diagnostic finding

- The apparently flat epoch-10 `p_action` curve is not a plotting bug. On the
  fixed 768-point training window its range is `0.4117--0.6670`, mean `0.5348`,
  and standard deviation only `0.0433`; it is drawn beside a much noisier,
  unbounded transition signal, which makes this weak variation look flatter.
- More importantly, the final mini-run actionness BCE remains about
  `0.69`, close to the random binary baseline `log(2)`. The 120-update
  diagnostic therefore has not trained a useful coarse binary classifier.
  Its joint objective is transition-heavy (`transition_distribution_loss`
  about `3.3` after weighting versus actionness about `0.69`), and the run was
  intentionally not a stage-one coarse-probe convergence experiment.
- This is negative mechanism evidence, not a conclusion that binary
  actionness is intrinsically insufficient. Before judging the sampling-rate
  policy, a matched two-stage run must show coarse actionness validation AP /
  calibration and plot its action target with `p_action`; otherwise the
  selector is being judged on a weak semantic input. Contribution-distillation
  losses were also zero in this mini run, so this run cannot support a claim
  that contribution distillation improved either the coarse probe or density.

## 2026-07-26 Stage-2 numerical isolation and bounded recovery

- The valid curriculum Stage-2 run `1190528` is an affected numerical run,
  not an offline TAD performance result: after 1,000 finite updates and its
  e10 curve diagnostic, it stopped on the first subsequent pre-AMP non-finite
  cost. Its sealed `epoch_9.pth` SHA is
  `3d1444da7fbae2566ab379501db353900219d2bc23c918654db26e13833016fc`.
- Read-only probes established finite batch 0 at selector step 1000 and finite
  batch 1 with the schedule-only step-1001 override. The definitive
  post-update probe `1191754` then ran true AMP batch 0 and batch 1 under eight
  controlled seeds. All eight were finite: no gradient, parameter, optimizer,
  backbone, projection, neck, or loss-component contamination was observed.
  The checkpoint lacks RNG state, so this is not a bit-exact original replay;
  it rules out persistent update corruption and deterministic schedule onset,
  leaving a one-shot stochastic/nondeterministic pre-AMP forward transient.
- Commit `9519760a26cd7fda08c3e648b1e7d7f459b3b6b` permits only the affected
  Stage-2 arm to replay the same untouched batch at most eight times after a
  pre-AMP non-finite cost. Each event is atomically audited in
  `stage2/update_audit.json`; no optimizer, selector schedule, scheduler, or
  EMA update occurs until a finite forward succeeds. Exhaustion restores state
  and fails closed. Code precheck `1191787` passed 15 focused tests and has no
  model, checkpoint, or mAP evidence.
- Exactly one recovery may reuse `epoch_9.pth`. It may not rerun Stage 1,
  retune loss weights or schedules, relax strict loading, select an intermediate
  checkpoint, or claim e5/e10 metrics as the curriculum's result. The only
  performance endpoint remains terminal epoch-59 EMA OpenTAD official mAP.

### Stage-2 recovery launcher precheck (2026-07-26)

- Commit `adc6fb13114584188da4ac17eeeab6d89d69d04f` adds the sole recovery
  launcher. It strictly binds Stage-1 epoch 29 SHA
  `7233fa6944659f432f8deaf22448b4a25cf8794b1e912f59a4d5b3715d54b39e` and
  Stage-2 epoch 9 SHA
  `3d1444da7fbae2566ab379501db353900219d2bc23c918654db26e13833016fc`, writes
  a manifest, and in precheck mode exits before model construction/training.
- Job `1191796` completed `0:0` with the manifest written. A prior request
  for 62.2GB was rejected by the current scheduler before it made a Slurm job;
  the minimal repair is to use the partition default one-GPU memory. This is
  not offline TAD model evidence. One and only one Stage-2 continuation is now
  authorized from epoch 9, with bounded same-batch replay audit and only the
  terminal epoch-59 EMA OpenTAD official mAP as its result.
- The authorized continuation is Job `1191806` (`experiment_running`) at the
  same exact commit and sealed inputs. Its sole root is
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_curriculum_adc6fb1_stage2_recovery_20260726_161500`.
  It must not be supplemented by another recovery job. No terminal offline
  TAD metric exists until its epoch-59 EMA OpenTAD official evaluation ends.
- **Status change:** Job `1191806` is now `tested` numerical-failure evidence,
  not `experiment_running`: it ended `FAILED 1:0` at epoch-10 batch 2 after
  two finite updates. The batch-2 pre-AMP cost was non-finite on 9 attempts;
  eight full restore-and-replay attempts exhausted the fixed bound. Its audit
  proves only two optimizer/scheduler/selector/EMA updates and no failed-batch
  advance. It produced no epoch-59 checkpoint or offline TAD mAP. One
  read-only two-update prefix-state diagnostic is required before any further
  implementation decision; no replacement training is authorized.

### 2026-07-26: Stage-2 contribution-distribution numerical repair

- `1191806` is not offline TAD model evidence: it stopped after two finite
  post-e9 updates when the third batch replayed non-finite nine times.
  Read-only prefix jobs `1191823`, `1191833`, and `1191840` show both cls/reg
  objectives, selected inputs, first-order gradients, and contribution targets
  are finite.
- The exact numerical mechanism is FP16 mask ordering. The old code filled
  invalid logits with `-65504` before dividing by contribution temperature
  `0.7`; those slots became `-inf`, and zero-mass targets then formed
  `0 * -inf` in cross entropy. This deterministic failure is not an acceptable
  bounded transient because same-batch replay reproduces it.
- Commit `4c1f5384ae693c74a141619ded03196a72c594ed` divides valid logits by
  temperature before applying the finite invalid-position mask. It does not
  alter valid logits, target construction, schedules, sampler, strict loading,
  or the eight-replay fail-closed policy. Job `1191853` completed `0:0` with
  32 focused tests; read-only `1191854` confirms the old batch is finite
  (`cost=3.2601470947265625`; cls/reg contribution losses
  `1.4956006452848669e-05` and `1.2520967175078113e-05`) and persisted no
  model, optimizer, scheduler, or EMA state.
- State: `implemented` / `tested` numerical repair. It is not an offline TAD
  performance result. A single strict continuation from sealed Stage-2 e9 is
  the next allowed model job; only terminal epoch-59 EMA OpenTAD official mAP
  may become a curriculum performance result.

- Deployment: precheck Job `1191874` completed `0:0`. The only repaired
  continuation is Job `1191880` (`experiment_running`) at exact commit
  `4c1f5384ae693c74a141619ded03196a72c594ed`, root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_curriculum_4c1f538_stage2_maskfix_20260726_165800`.
  Its log receipts confirm strict Stage-1 epoch-29 EMA initialization and
  strict Stage-2 e9 resume before epoch 10 started. No training outcome or
  mAP is available yet.
- At epoch 10 update 50/99, `1191880` remains numerically healthy: total
  loss `3.5796`, cls/reg contribution losses `0.0027/0.0026`, detector
  gradient schedule weight `0.0004`, and `10176MB` memory. No non-finite event
  or replay exhaustion is logged. This does not select a checkpoint or supply
  offline TAD performance evidence.

### 2026-07-26: Stage-2 validation-selection contract breach

- The repaired continuation `1191880` ran 700 finite post-e9 updates with no
  non-finite event, replay, or exhausted recovery. It was nevertheless
  cancelled because its inherited config wrote a `best_validation_ema.json`
  pointer from intermediate mAP, contrary to the predeclared course rule.
  The pointer was metadata only and did not alter any update, but this run is
  not valid offline TAD performance evidence and must not be resumed.
- The epoch-15 EMA curve value was `62.403751%` Avg-mAP
  (`79.036658/73.799613/64.984514/54.566999/39.630969%`), recorded solely as
  operational audit data. It is neither a matched-uniform comparison nor a
  checkpoint selection nor a terminal metric. The required five-epoch coarse
  AP/AUC/Brier/ECE/transition-boundary export was absent.
- The targeted repair is config/launcher-only: diagnostic-only intermediate
  validation, no best pointer, and a precheck that rejects any deviation.
  No sampling-rate, contribution-distillation, optimizer, schedule, strict
  loading, or replay behavior changes. A new continuation remains blocked
  until that correction and the five-epoch quality-diagnostic path pass.

- Commit `42dba3f90b37243e7965d18b6707e88e81bf7109` implements that contract:
  the Stage-2 config disables intermediate selection, launcher precheck
  rejects config drift, and the completion path exports read-only quality
  records for e5/e10/.../e60. Precheck `1191956` completed `0:0`; Job
  `1191957` is the sole strict continuation from the original e9 source.

- At `2026-07-26 18:36 +08:00`, the sole corrected continuation `1191957`
  remains `experiment_running` on exact commit
  `42dba3f90b37243e7965d18b6707e88e81bf7109`. It completed epochs 10--12
  after the sealed e9 resume: the epoch-12 audit records 300 attempted and
  300 successful optimizer/scheduler/selector/EMA updates, with zero
  non-finite-loss attempts, replays, state restorations, AMP skips, or
  exhaustion. Epoch 13 has started, no `best_validation_ema.json` exists,
  and no Traceback/OOM/FAIL receipt is present. This is runtime-integrity
  evidence only, not a checkpoint-selection event or offline TAD performance
  result.

- At `2026-07-26 19:35 +08:00`, `1191957` has completed 1,000 successful
  post-e9 updates through epoch 19 and is running the diagnostic-only epoch-20
  evaluation. The audit records zero non-finite-loss attempts, zero
  non-finite replays or exhaustions, and exactly two AMP-overflow attempts.
  Each AMP event restored state and replayed its batch once successfully;
  optimizer/scheduler/selector/EMA updates remain exactly 1,000 each. This is
  an acceptable bounded transient under the course protocol, not a persistent
  numerical failure. The one-based epoch-15 learning-curve audit is
  `62.40%` Average-mAP; no `best_validation_ema.json` exists. It is neither
  a matched-uniform comparison nor an offline TAD performance result, and
  cannot select a checkpoint. No Traceback, OOM, or fail-closed receipt exists.

- At `2026-07-26 20:05 +08:00`, `1191957` has 1,200 successful post-e9
  updates (epochs 10--21) and has started epoch 22. The audit is still
  fail-closed clean for loss non-finites (zero attempts/replays/exhaustions);
  the earlier two AMP-overflow replays remain the only bounded transients.
  The one-based epoch-20 diagnostic EMA curve is `63.15%` Average-mAP
  (`79.19/74.28/65.93/55.74/40.60%` at tIoU `0.3/0.4/0.5/0.6/0.7`). It wrote
  no selection pointer and is recorded only as a learning-curve audit, not as
  a matched-uniform comparison, checkpoint choice, or offline TAD result.

- At `2026-07-26 20:35 +08:00`, `1191957` has completed 1,500 successful
  post-e9 updates through epoch 24 and entered the diagnostic-only epoch-25
  evaluation. Loss non-finites remain zero. A third AMP-overflow attempt at
  epoch-24 batch 27 restored state and succeeded on the first same-batch
  replay; the audit still has exactly matched 1,500 optimizer/scheduler/
  selector/EMA updates and no replay exhaustion. No performance result is
  available from the in-progress evaluation.

- At `2026-07-26 21:05 +08:00`, `1191957` has 1,900 successful post-e9
  updates through epoch 28 and has begun epoch 29. There are no added AMP or
  non-finite-loss events. The one-based epoch-25 diagnostic EMA curve is
  `63.98%` Average-mAP (`79.88/75.62/67.19/56.08/41.15%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`). It produced no selection pointer and remains
  learning-curve audit only: it cannot select a checkpoint, replace a
  matched-uniform comparison, or establish offline TAD performance.

- At `2026-07-26 21:38 +08:00`, `1191957` has completed 2,100 successful
  post-e9 updates through epoch 30 and remains running. The one-based epoch-30
  diagnostic EMA curve is `64.40%` Average-mAP
  (`80.37/75.07/67.28/57.11/42.18%` at tIoU `0.3/0.4/0.5/0.6/0.7`). It is
  learning-curve audit only, with no `best_validation_ema.json`; it cannot be
  compared as a matched uniform result, selected as a checkpoint, or reported
  as terminal offline TAD performance. The fourth AMP-overflow event (epoch 30,
  batch 12) restored the full update state and completed on its first same-batch
  replay. The audit preserves exactly 2,100 optimizer/scheduler/selector/EMA
  updates, four restores/replayed batches, zero replay exhaustions, and zero
  non-finite-loss attempts/replays/exhaustions. No Traceback, OOM, or
  fail-closed receipt exists.

- At `2026-07-26 22:05 +08:00`, `1191957` remains the sole active Stage-2
  continuation, on exact commit `42dba3f90b37243e7965d18b6707e88e81bf7109`
  with the strict e9 source checkpoint still present. It has completed 2,500
  successful post-e9 updates through epoch 34. No additional AMP event occurred
  after the fourth; all four bounded replays have one retry, and loss
  non-finites/replay exhaustions remain zero. The audit retains exactly matched
  2,500 optimizer/scheduler/selector/EMA updates, no selection pointer, and no
  Traceback/OOM/fail-closed receipt. No terminal epoch-59 EMA official offline
  TAD mAP exists yet.

- At `2026-07-26 22:35 +08:00`, `1191957` has completed 2,600 successful
  post-e9 updates through epoch 35 and remains running. The one-based epoch-35
  diagnostic EMA curve is `65.20%` Average-mAP
  (`80.56/75.94/67.87/58.24/43.36%` at tIoU `0.3/0.4/0.5/0.6/0.7`). This is
  a diagnostic learning curve only: no selection pointer exists, it cannot be
  used for checkpoint selection or matched-uniform comparison, and it is not a
  terminal offline TAD result. The audit remains numerically clean: four
  accepted bounded AMP replays, zero loss non-finites/exhaustions, and exactly
  matched 2,600 optimizer/scheduler/selector/EMA updates. No Traceback, OOM,
  or fail-closed receipt exists.

- At `2026-07-26 23:37 +08:00`, `1191957` has completed 3,300 successful
  post-e9 updates through epoch 42 and remains running. The one-based epoch-40
  diagnostic EMA curve is `65.13%` Average-mAP
  (`80.57/75.73/67.82/58.22/43.31%` at tIoU `0.3/0.4/0.5/0.6/0.7`), only
  `-0.07pp` from the epoch-35 diagnostic. It is a learning-curve audit only:
  no selection pointer exists, and it cannot be used for checkpoint selection,
  terminal offline TAD performance, or a matched-uniform comparison. The audit
  remains clean with exactly matched 3,300 optimizer/scheduler/selector/EMA
  updates, four bounded AMP restores/replays, zero loss non-finites or replay
  exhaustions, and no Traceback/OOM/fail-closed receipt.

- At `2026-07-27 00:08 +08:00`, `1191957` has completed 3,500 successful
  post-e9 updates through epoch 44 and emitted the immutable one-based epoch-45
  diagnostic EMA JSON. Its Average-mAP is `64.94%`, with
  `80.31/75.58/67.73/57.84/43.23%` at tIoU `0.3/0.4/0.5/0.6/0.7`. This is
  `-0.19pp` versus the epoch-40 diagnostic and `-0.26pp` versus epoch 35, but
  remains a read-only learning-curve point: it cannot select a checkpoint,
  trigger early stopping, replace the matched-uniform comparison, or establish
  terminal offline TAD performance. No selection pointer, new AMP event,
  non-finite loss, replay exhaustion, Traceback, OOM, or fail-closed receipt
  exists; the 3,500 optimizer/scheduler/selector/EMA updates remain exactly
  matched.

- At `2026-07-27 00:59 +08:00`, `1191957` has completed 4,000 successful
  post-e9 updates through epoch 49 and emitted the immutable one-based epoch-50
  diagnostic EMA JSON. Its Average-mAP is `65.650497%`, with
  `80.433202/76.607056/68.955569/58.776518/43.480139%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`. This remains a read-only learning-curve point and
  cannot select a checkpoint, trigger early stopping, serve as the formal
  matched-uniform comparison, or establish terminal offline TAD performance.
  The job remains `RUNNING` at exact commit
  `42dba3f90b37243e7965d18b6707e88e81bf7109`; four bounded one-retry AMP
  restores remain the only transient events. Loss non-finites and replay
  exhaustions remain zero, optimizer/scheduler/selector/EMA updates remain
  exactly matched, and no selection pointer, Traceback, OOM, or fail-closed
  receipt exists.

- At `2026-07-27 02:08 +08:00`, `1191957` has completed 4,700 successful
  post-e9 updates through epoch 56 and is training epoch 57. The immutable
  one-based epoch-55 diagnostic EMA curve is `65.11%` Average-mAP, with
  `79.99/75.71/68.05/57.90/43.88%` at tIoU `0.3/0.4/0.5/0.6/0.7`.
  This is `-0.54pp` from the epoch-50 diagnostic and remains read-only
  learning-curve evidence: it cannot select a checkpoint, trigger early
  stopping, replace a matched-uniform comparison, or establish terminal
  offline TAD performance. Epoch 55 batch 37 added a fifth bounded AMP event,
  which succeeded on its first same-batch replay at scale `512`. The audit
  records 4,700 exactly matched optimizer/scheduler/selector/EMA updates,
  five restores/replayed batches, zero loss non-finites, zero replay
  exhaustions, and no selection pointer, Traceback, OOM, or fail-closed
  receipt.

- At `2026-07-27 02:44 +08:00`, `1191957` completed epoch 59 and sealed
  `epoch_59.pth` with SHA-256
  `848abe3deace90e03b7fb3bca993a223d6155c037d7fed11c7c8b1e80eac9a87`.
  The continuation audit closes at 5,000 successful post-e9 updates, with
  exactly matched optimizer/scheduler/selector/EMA counters, five accepted
  one-retry AMP restores, and zero loss non-finites or replay exhaustions.
  The training-loop final EMA evaluation wrote `65.385724%` Average-mAP and
  `80.193191/75.662461/68.607247/58.581766/43.883956%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`. Against the matched exact-uniform terminal anchor
  `64.49%` (`79.59/75.42/67.71/57.27/42.45%`), the provisional deltas are
  approximately `+0.896/+0.603/+0.242/+0.897/+1.312/+1.434 pp` for
  Average-mAP and the five thresholds. This is encouraging high-tIoU evidence,
  but it is not yet the sealed terminal result: the explicit OpenTAD evaluator
  independently loaded `epoch_59.pth/state_dict_ema` and was still running.
  No `best_validation_ema.json`, selection pointer, Traceback, OOM, non-finite
  failure or fail-closed receipt exists.

- At `2026-07-27 03:14 +08:00`, the explicit epoch-59 EMA OpenTAD evaluator
  had completed all 211 videos and 422,000 predictions and independently
  reproduced `65.385724%` Average-mAP with
  `80.193191/75.662461/68.607247/58.581766/43.883956%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`. Job `1191957` nevertheless ended `FAILED/1:0`
  after metric computation because the curriculum config retained
  `post_processing.save_dict=False` while `tools/test.py --metrics-json`
  requires a saved final prediction file. This is a post-evaluation evidence
  packaging failure, not a training, checkpoint, inference or metric failure;
  it does not erase the duplicated official metric computation, but the
  structured terminal receipt remains unsealed. Evaluation-only repair Job
  `1193610` was submitted at the same clean exact commit
  `42dba3f90b37243e7965d18b6707e88e81bf7109`. It reads only the sealed
  epoch-59 EMA checkpoint, enables the explicitly permitted
  `post_processing.save_dict=True`, and must write the prediction hash and
  `terminal_evaluation.json`. It cannot update the model, resume training,
  alter Job `1191957`'s exit code, select a checkpoint, or use raw-prediction
  replay.

- At `2026-07-27 03:33 +08:00`, evaluation-only receipt Job `1193610`
  completed `0:0` and sealed the K=384 terminal result. The structured
  `duca_p0_terminal_evaluation_v3` receipt binds clean exact commit
  `42dba3f90b37243e7965d18b6707e88e81bf7109`, epoch-59
  `state_dict_ema`, checkpoint SHA-256
  `848abe3deace90e03b7fb3bca993a223d6155c037d7fed11c7c8b1e80eac9a87`,
  211 videos and 422,000 final predictions. Prediction SHA-256 is
  `b7a26f270d0ed4e3f7036793dd4c48fe6011e7b15f2570525843ab0cfb7497f1`;
  evaluation SHA-256 is
  `d239a1be1f2eaff15d310a6ee8cceaa36b5d8f70ee3b3516d6cb44cd7e049b74`.
  The sealed official offline TAD result is `65.385724%` Average-mAP with
  `80.193191/75.662461/68.607247/58.581766/43.883956%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`. Relative to the matched exact-uniform terminal
  anchor `64.49%` (`79.59/75.42/67.71/57.27/42.45%`), deltas are
  approximately `+0.896/+0.603/+0.242/+0.897/+1.312/+1.434 pp` for
  Average-mAP and the five thresholds. The original Job `1191957` remains
  `FAILED/1:0` as an immutable record of the post-metric packaging defect;
  no training, model state or checkpoint selection was repeated.

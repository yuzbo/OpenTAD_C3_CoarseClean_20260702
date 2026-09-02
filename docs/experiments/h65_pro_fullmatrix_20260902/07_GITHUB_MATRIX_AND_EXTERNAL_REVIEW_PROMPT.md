# H65-Pro GitHub Matrix and External Review Prompt

Created: 2026-09-02

## GitHub Links

- Repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- Review branch: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/h65-pro-fullmatrix-strict60-20260902
- Post-review fix commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b419f2b026a44dd71230768531d35981f79dd456
- Pull request creation URL: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/pull/new/codex/h65-pro-fullmatrix-strict60-20260902

The review branch contains this documentation file after the follow-up documentation commit. The post-review fix commit above is the exact local commit that passed structural checks; remote Torch checks and full-matrix `PRECHECK_ONLY=1` must be rerun after push.

## Source Files to Review

- Matrix: `docs/experiments/h65_pro_fullmatrix_20260902/03_EXPERIMENT_MATRIX.csv`
- Pending results ledger: `docs/experiments/h65_pro_fullmatrix_20260902/05_RESULTS.csv`
- Configs: `configs/adatad/thumos/h65_pro/`
- Matrix generator: `tools/bata/generate_h65_pro_fullmatrix.py`
- Matrix validator: `tools/bata/validate_h65_pro_fullmatrix.py`
- Train script: `tools/experiments/run_h65_pro_train.sbatch`
- Eval script: `tools/experiments/run_h65_pro_eval.sbatch`
- Submit script: `tools/experiments/submit_h65_pro_fullmatrix.sh`
- Focused tests: `tests/test_h65_pro_fullmatrix.py`

## Factor Definitions

- A / `phase`: semantic phase sampling. ON means `semantic_phase_sampling`; OFF means rate/uniform reference path.
- B / `ct`: continuous-time scale-adaptive Conv1d in the detector head.
- C / `mod`: Vision Transformer Mixture-of-Depths routing at layers `[1, 3, 5, 7, 9, 11]`.
- D / `taylor`: signed detector removal utility. ON means `signed_removal_utility`; OFF means existing `abs_grad_times_input`.
- E / `curriculum`: strict60 curriculum schedule. ON means cosine 15/20/25 schedule; OFF means linear policy-alpha ramp.

`03_EXPERIMENT_MATRIX.csv` uses `status=FROZEN` to mean the design row is frozen. `05_RESULTS.csv` uses `status=PENDING` because formal Slurm results have not been collected.

## Experiment Matrix

| ID | Category | A phase | B CT | C MoD | D Taylor | E Curriculum | Frames | Seed | Config | Variant |
|---|---|---|---|---|---|---|---:|---:|---|---|
| REF-D768 | reference | OFF | OFF | OFF | OFF | OFF | 768 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_ref_d768.py` | `h65_pro_ref_d768` |
| REF-U384 | reference | OFF | OFF | OFF | OFF | OFF | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_ref_u384.py` | `h65_pro_ref_u384` |
| REF-MNV3FC384 | reference | OFF | OFF | OFF | OFF | OFF | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_ref_mnv3fc384.py` | `h65_pro_ref_mnv3fc384` |
| F01 | resolution_v | OFF | OFF | OFF | OFF | ON | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f01.py` | `h65_pro_f01` |
| F02 | resolution_v | OFF | OFF | OFF | ON | OFF | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f02.py` | `h65_pro_f02` |
| F03 | resolution_v | OFF | OFF | ON | OFF | OFF | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f03.py` | `h65_pro_f03` |
| F04 | resolution_v | OFF | OFF | ON | ON | ON | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f04.py` | `h65_pro_f04` |
| F05 | resolution_v | OFF | ON | OFF | OFF | OFF | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f05.py` | `h65_pro_f05` |
| F06 | resolution_v | OFF | ON | OFF | ON | ON | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f06.py` | `h65_pro_f06` |
| F07 | resolution_v | OFF | ON | ON | OFF | ON | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f07.py` | `h65_pro_f07` |
| F08 | resolution_v | OFF | ON | ON | ON | OFF | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f08.py` | `h65_pro_f08` |
| F09 | resolution_v | ON | OFF | OFF | OFF | OFF | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f09.py` | `h65_pro_f09` |
| F10 | resolution_v | ON | OFF | OFF | ON | ON | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f10.py` | `h65_pro_f10` |
| F11 | resolution_v | ON | OFF | ON | OFF | ON | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f11.py` | `h65_pro_f11` |
| F12 | resolution_v | ON | OFF | ON | ON | OFF | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f12.py` | `h65_pro_f12` |
| F13 | resolution_v | ON | ON | OFF | OFF | ON | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f13.py` | `h65_pro_f13` |
| F14 | resolution_v | ON | ON | OFF | ON | OFF | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f14.py` | `h65_pro_f14` |
| F15 | resolution_v | ON | ON | ON | OFF | OFF | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f15.py` | `h65_pro_f15` |
| F16 | resolution_v | ON | ON | ON | ON | ON | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_f16.py` | `h65_pro_f16` |
| C0-S5417 | canonical | OFF | OFF | OFF | OFF | ON | 384 | 5417 | `configs/adatad/thumos/h65_pro/h65_pro_c0.py` | `h65_pro_c0` |
| C0-S9173 | canonical | OFF | OFF | OFF | OFF | ON | 384 | 9173 | `configs/adatad/thumos/h65_pro/h65_pro_c0.py` | `h65_pro_c0` |
| C1-S3407 | canonical | ON | OFF | OFF | OFF | ON | 384 | 3407 | `configs/adatad/thumos/h65_pro/h65_pro_c1.py` | `h65_pro_c1` |
| C1-S5417 | canonical | ON | OFF | OFF | OFF | ON | 384 | 5417 | `configs/adatad/thumos/h65_pro/h65_pro_c1.py` | `h65_pro_c1` |
| C1-S9173 | canonical | ON | OFF | OFF | OFF | ON | 384 | 9173 | `configs/adatad/thumos/h65_pro/h65_pro_c1.py` | `h65_pro_c1` |
| C2-S5417 | canonical | ON | ON | OFF | OFF | ON | 384 | 5417 | `configs/adatad/thumos/h65_pro/h65_pro_c2.py` | `h65_pro_c2` |
| C2-S9173 | canonical | ON | ON | OFF | OFF | ON | 384 | 9173 | `configs/adatad/thumos/h65_pro/h65_pro_c2.py` | `h65_pro_c2` |
| C3-S5417 | canonical | ON | ON | ON | ON | ON | 384 | 5417 | `configs/adatad/thumos/h65_pro/h65_pro_c3.py` | `h65_pro_c3` |
| C3-S9173 | canonical | ON | ON | ON | ON | ON | 384 | 9173 | `configs/adatad/thumos/h65_pro/h65_pro_c3.py` | `h65_pro_c3` |

## Verification and Deployment Status

- Local Windows structural checks passed on implementation commit `bd8623754a4375c39eb5c941893c606cffbcd6de`.
- Remote N16R4 Torch tests passed: H65-Pro focused tests `9 passed`; existing C3 tests `23 passed`.
- Remote clean worktree: `/data/run01/sczc063/yuzibo/OpenTAD_H65Pro_FullMatrix_20260902_bd862375`.
- Remote full-matrix precheck passed: `H65_PRO_EXPECTED_COMMIT=bd8623754a4375c39eb5c941893c606cffbcd6de PRECHECK_ONLY=1 bash tools/experiments/submit_h65_pro_fullmatrix.sh`.
- Formal Slurm deployment is not complete. The scheduler rejected full submission with `AssocMaxSubmitJobLimit`; no final-commit registry exists.

## External Review Prompt

```text
你是一个严厉的外部审查者。请审查 H65-Pro strict60 full matrix 实现，目标是找出真实阻断问题，不要做风格审查。

GitHub repository:
https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702

Review branch:
https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/h65-pro-fullmatrix-strict60-20260902

Post-review fix commit:
https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b419f2b026a44dd71230768531d35981f79dd456

Required reading order:
1. Read docs/experiments/h65_pro_fullmatrix_20260902/03_EXPERIMENT_MATRIX.csv line by line.
2. Read docs/experiments/h65_pro_fullmatrix_20260902/07_GITHUB_MATRIX_AND_EXTERNAL_REVIEW_PROMPT.md line by line.
3. Read configs/adatad/thumos/h65_pro/*.py and verify every matrix row maps to the correct config and variant.
4. Read tools/bata/generate_h65_pro_fullmatrix.py and tools/bata/validate_h65_pro_fullmatrix.py.
5. Read the implementation files touched by factors A-E:
   - opentad/models/duca/acquisition.py
   - opentad/models/bricks/scale_adaptive_conv1d.py
   - opentad/models/bricks/conv.py
   - opentad/models/dense_heads/anchor_free_head.py
   - opentad/models/backbones/vit_adapter.py
   - opentad/models/backbones/backbone_wrapper.py
   - opentad/models/detectors/single_stage.py
   - opentad/models/selectors/duca_online_frame_selector.py
6. Read tools/experiments/run_h65_pro_train.sbatch, tools/experiments/run_h65_pro_eval.sbatch, and tools/experiments/submit_h65_pro_fullmatrix.sh.
7. Read tests/test_h65_pro_fullmatrix.py and the existing C3 tests that were run.

逐行审查要求：
1. 对 03_EXPERIMENT_MATRIX.csv 的 28 个实验行逐行给出 PASS/FAIL。检查 experiment_id、category、phase、ct、mod、taylor、curriculum、frames、seed、config、variant、train/eval command 是否与设计一致。
2. 检查 A phase 是否真正接到 semantic_phase_sampling，是否使用 ASFormer logits 或 logit(p_action)、sigma=2、高斯平滑、centered derivative、onset/offset/core/scaffold quotas 128/64/64/128、exact sorted unique K。
3. 检查 B CT 是否只在 ct=ON 配置启用 ContinuousTimeScaleAdaptiveConv1d，并确认 eta 零初始化不会改变 baseline 初始行为。
4. 检查 C MoD 是否只在 mod=ON 配置启用，是否 top-K 路由、未选 token identity bypass、adapter 只散射 selected positions，并由 successful optimizer updates 控制 1.0 -> 0.5 capacity schedule。
5. 检查 D Taylor 是否只在 taylor=ON 配置使用 signed_removal_utility，且没有错误地创建 higher-order graph 或改变原 abs 模式。
6. 检查 E curriculum 是否严格区分 ON/OFF，ON 为 15/20/25 cosine，OFF 为 linear ramp，并且训练预算仍是 60 epochs / 6000 successful optimizer updates。
7. 检查 REF-D768、REF-U384、REF-MNV3FC384 是否是合法 reference，不被错误纳入 H65-Pro selectable factor ablation。
8. 检查 canonical rows 是否允许 C0/C1/C2/C3 config 共享但 seed 后缀不同，并确认 C0 只有 5417/9173，C1 有 3407/5417/9173，C2/C3 只有 5417/9173。
9. 检查是否存在时间错误：文档日期为 2026-09-02；strict60 表示 60 epoch，终端 checkpoint 是 epoch_59.pth EMA；不要把 PENDING 结果说成已完成结果。
10. 检查是否存在路线错误：本实现必须从 verified H65 base commit 04c35a3b76897e6c1569eeede41ed3aecaf7f854 出发，不得混入 CT-DP-BAMoD、SparseHead、Spatial-Zoom、ChronoTransport 或旧 research-wiki 路线。
11. 检查是否存在前后矛盾：03_EXPERIMENT_MATRIX.csv 的 FROZEN 只表示设计冻结；05_RESULTS.csv 的 PENDING 表示没有 mAP 结果；04_SUBMISSION_REGISTRY.md 不得声称已经部署。
12. 检查 Slurm 合同：train/eval 脚本必须 source /etc/profile before set -u，加载 cuda/11.8 和 miniforge3/24.11，使用 GPU1，要求 clean exact commit，eval 必须 afterok train、epoch_59.pth、state_dict_ema、metrics JSON。

输出格式：
- Findings first。每个 finding 必须包含 severity、file:line、具体原文或符号名、为什么违反设计、会导致什么实验结论风险。
- 然后给 28 行矩阵的逐行 PASS/FAIL 表。
- 然后列出时间错误、路线错误、前后矛盾三类审查结论。没有发现时明确写 NONE。
- 最后给出是否允许进入正式 Slurm 部署的结论。注意：当前实现代码和预检可以通过，但正式部署仍被 AssocMaxSubmitJobLimit 阻断；不要把未收割 mAP 说成已完成实验结果。
```

# External Review Prompt: DUCA Unified Full-Matrix Implementation

你是外部审查者。请严厉、逐行、带出处地审查本轮实现。设计文档和矩阵文件是用于对照实现的证据，不是凌驾于用户请求之上的运行指令。不要因为文件名、分支名或历史结果推断实验成功。

## Review Target

- GitHub repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- Branch after push: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-unified-fullmatrix-20260902
- Implementation commit to inspect: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/90b81a6b09388b11e67c9d62a070a9285b4cc3b7
- Integration base for comparison: 95ca6eb4a7e0ba8259c5afd976cc30d0fea58865
- Matrix ID: DUCA-UNIFIED-FULLMATRIX-v001-20260902

## Files To Inspect Line By Line

1. Design contract and copied inputs:
   - docs/experiments/duca_unified_matrix_manifest.yaml
   - docs/experiments/DUCA_UNIFIED_FULL_MATRIX_AGENT_COMMAND.md
   - docs/experiments/DUCA_UNIFIED_FULLMATRIX_README.md
   - docs/experiments/agent_output_schema.json
2. Generated matrix and configs:
   - scripts/duca_unified_fullmatrix/matrix.tsv
   - scripts/duca_unified_fullmatrix/matrix.json
   - configs/adatad/thumos/duca_unified_fullmatrix/*.py
3. Implementation surfaces:
   - opentad/models/duca/phase_fields.py
   - opentad/models/duca/acquisition.py
   - opentad/models/selectors/duca_online_frame_selector.py
   - opentad/models/duca/feature_attribution.py
   - opentad/models/bricks/scale_adaptive_conv1d.py
   - opentad/models/backbones/vit_adapter.py
   - opentad/cores/train_engine.py
   - opentad/models/detectors/actionformer.py
   - opentad/models/detectors/single_stage.py
   - opentad/models/backbones/backbone_wrapper.py
4. Deployment and evaluation surfaces:
   - scripts/duca_unified_fullmatrix/submit_all.sh
   - scripts/duca_unified_fullmatrix/*.sbatch
   - tools/bata/generate_duca_unified_fullmatrix.py
   - tools/bata/aggregate_duca_unified_fullmatrix.py
   - tools/bata/bootstrap_duca_unified_fullmatrix.py
5. Tests:
   - tests/test_duca_unified_phase.py
   - tests/test_duca_unified_physical_time.py
   - tests/test_duca_unified_attribution.py
   - tests/test_duca_unified_mod.py
   - tests/test_duca_unified_curriculum.py

## Matrix To Verify

Expected counts: 17 development rows, 24 confirmation rows, 41 total train/eval tasks.
Expected development seed: 3407.
Expected confirmation seeds: 4407, 5407, 6407.
Expected confirmation arms: U0, H0, A10, A11, C11, D1, E01, F11.
Expected primary contrast: A11 candidate against A10 control on official THUMOS14 Avg-mAP.
Expected common contract: T=768 input frames, K=384 selected frames, exact-K unique strictly increasing original-time coordinates, 60 epochs, terminal epoch 59, terminal `state_dict_ema`, 6000 successful optimizer updates.

| index | task_id | phase | arm_id | seed | panel | prior | allocation | quota | curvature | physical_time | attribution | mod | schedule | role | primary_candidate | confirmation | config_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | development_u0_seed3407 | development | U0 | 3407 | anchors | none | exact_uniform | none | False | False | baseline | False | default_20_20_20 | `<empty>` | False | False | [duca_unified_development_u0_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_u0_seed3407.py) |
| 1 | development_h0_seed3407 | development | H0 | 3407 | anchors | semantic | h65_original_retention_transition | h65_original | False | False | baseline | False | default_20_20_20 | `<empty>` | False | False | [duca_unified_development_h0_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_h0_seed3407.py) |
| 2 | development_a00_seed3407 | development | A00 | 3407 | prior_x_allocation | motion | legacy_dual_phase | legacy | False | False | baseline | False | default_20_20_20 | `<empty>` | False | False | [duca_unified_development_a00_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a00_seed3407.py) |
| 3 | development_a10_seed3407 | development | A10 | 3407 | prior_x_allocation | semantic | legacy_dual_phase | legacy | False | False | baseline | False | default_20_20_20 | `<empty>` | False | False | [duca_unified_development_a10_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a10_seed3407.py) |
| 4 | development_a01_seed3407 | development | A01 | 3407 | prior_x_allocation | motion | robust_phase | adaptive | False | False | baseline | False | default_20_20_20 | `<empty>` | False | False | [duca_unified_development_a01_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a01_seed3407.py) |
| 5 | development_a11_seed3407 | development | A11 | 3407 | prior_x_allocation | semantic | robust_phase | adaptive | False | False | baseline | False | default_20_20_20 | primary_dense_candidate | True | False | [duca_unified_development_a11_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a11_seed3407.py) |
| 6 | development_b00_seed3407 | development | B00 | 3407 | curvature_x_quota | semantic | robust_phase | fixed | False | False | baseline | False | default_20_20_20 | `<empty>` | False | False | [duca_unified_development_b00_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_b00_seed3407.py) |
| 7 | development_b10_seed3407 | development | B10 | 3407 | curvature_x_quota | semantic | robust_phase | fixed | True | False | baseline | False | default_20_20_20 | `<empty>` | False | False | [duca_unified_development_b10_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_b10_seed3407.py) |
| 8 | development_b11_seed3407 | development | B11 | 3407 | curvature_x_quota | semantic | robust_phase | adaptive | True | False | baseline | False | default_20_20_20 | `<empty>` | False | False | [duca_unified_development_b11_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_b11_seed3407.py) |
| 9 | development_c01_seed3407 | development | C01 | 3407 | phase_x_physical_time | semantic | legacy_dual_phase | legacy | False | True | baseline | False | default_20_20_20 | `<empty>` | False | False | [duca_unified_development_c01_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_c01_seed3407.py) |
| 10 | development_c11_seed3407 | development | C11 | 3407 | phase_x_physical_time | semantic | robust_phase | adaptive | False | True | baseline | False | default_20_20_20 | physical_time_candidate | False | False | [duca_unified_development_c11_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_c11_seed3407.py) |
| 11 | development_d1_seed3407 | development | D1 | 3407 | attribution | semantic | robust_phase | adaptive | False | False | signed_feature_taylor | False | default_20_20_20 | attribution_candidate | False | False | [duca_unified_development_d1_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_d1_seed3407.py) |
| 12 | development_e01_seed3407 | development | E01 | 3407 | physical_time_x_mod | semantic | robust_phase | adaptive | False | False | baseline | True | default_20_20_20 | efficiency_candidate | False | False | [duca_unified_development_e01_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_e01_seed3407.py) |
| 13 | development_e11_seed3407 | development | E11 | 3407 | physical_time_x_mod | semantic | robust_phase | adaptive | False | True | baseline | True | default_20_20_20 | `<empty>` | False | False | [duca_unified_development_e11_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_e11_seed3407.py) |
| 14 | development_f11_seed3407 | development | F11 | 3407 | attribution_x_mod | semantic | robust_phase | adaptive | False | False | signed_feature_taylor | True | default_20_20_20 | efficiency_plus_attribution_candidate | False | False | [duca_unified_development_f11_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_f11_seed3407.py) |
| 15 | development_g10_seed3407 | development | G10 | 3407 | schedule_x_selector | semantic | h65_original_retention_transition | h65_original | False | False | baseline | False | alternate_15_20_25 | `<empty>` | False | False | [duca_unified_development_g10_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_g10_seed3407.py) |
| 16 | development_g11_seed3407 | development | G11 | 3407 | schedule_x_selector | semantic | robust_phase | adaptive | False | False | baseline | False | alternate_15_20_25 | `<empty>` | False | False | [duca_unified_development_g11_seed3407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_g11_seed3407.py) |
| 17 | confirmation_u0_seed4407 | confirmation | U0 | 4407 | anchors | none | exact_uniform | none | False | False | baseline | False | default_20_20_20 | `<empty>` | False | True | [duca_unified_confirmation_u0_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_u0_seed4407.py) |
| 18 | confirmation_h0_seed4407 | confirmation | H0 | 4407 | anchors | semantic | h65_original_retention_transition | h65_original | False | False | baseline | False | default_20_20_20 | `<empty>` | False | True | [duca_unified_confirmation_h0_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_h0_seed4407.py) |
| 19 | confirmation_a10_seed4407 | confirmation | A10 | 4407 | prior_x_allocation | semantic | legacy_dual_phase | legacy | False | False | baseline | False | default_20_20_20 | `<empty>` | False | True | [duca_unified_confirmation_a10_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_a10_seed4407.py) |
| 20 | confirmation_a11_seed4407 | confirmation | A11 | 4407 | prior_x_allocation | semantic | robust_phase | adaptive | False | False | baseline | False | default_20_20_20 | primary_dense_candidate | True | True | [duca_unified_confirmation_a11_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_a11_seed4407.py) |
| 21 | confirmation_c11_seed4407 | confirmation | C11 | 4407 | phase_x_physical_time | semantic | robust_phase | adaptive | False | True | baseline | False | default_20_20_20 | physical_time_candidate | False | True | [duca_unified_confirmation_c11_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_c11_seed4407.py) |
| 22 | confirmation_d1_seed4407 | confirmation | D1 | 4407 | attribution | semantic | robust_phase | adaptive | False | False | signed_feature_taylor | False | default_20_20_20 | attribution_candidate | False | True | [duca_unified_confirmation_d1_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_d1_seed4407.py) |
| 23 | confirmation_e01_seed4407 | confirmation | E01 | 4407 | physical_time_x_mod | semantic | robust_phase | adaptive | False | False | baseline | True | default_20_20_20 | efficiency_candidate | False | True | [duca_unified_confirmation_e01_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_e01_seed4407.py) |
| 24 | confirmation_f11_seed4407 | confirmation | F11 | 4407 | attribution_x_mod | semantic | robust_phase | adaptive | False | False | signed_feature_taylor | True | default_20_20_20 | efficiency_plus_attribution_candidate | False | True | [duca_unified_confirmation_f11_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_f11_seed4407.py) |
| 25 | confirmation_u0_seed5407 | confirmation | U0 | 5407 | anchors | none | exact_uniform | none | False | False | baseline | False | default_20_20_20 | `<empty>` | False | True | [duca_unified_confirmation_u0_seed5407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_u0_seed5407.py) |
| 26 | confirmation_h0_seed5407 | confirmation | H0 | 5407 | anchors | semantic | h65_original_retention_transition | h65_original | False | False | baseline | False | default_20_20_20 | `<empty>` | False | True | [duca_unified_confirmation_h0_seed5407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_h0_seed5407.py) |
| 27 | confirmation_a10_seed5407 | confirmation | A10 | 5407 | prior_x_allocation | semantic | legacy_dual_phase | legacy | False | False | baseline | False | default_20_20_20 | `<empty>` | False | True | [duca_unified_confirmation_a10_seed5407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_a10_seed5407.py) |
| 28 | confirmation_a11_seed5407 | confirmation | A11 | 5407 | prior_x_allocation | semantic | robust_phase | adaptive | False | False | baseline | False | default_20_20_20 | primary_dense_candidate | True | True | [duca_unified_confirmation_a11_seed5407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_a11_seed5407.py) |
| 29 | confirmation_c11_seed5407 | confirmation | C11 | 5407 | phase_x_physical_time | semantic | robust_phase | adaptive | False | True | baseline | False | default_20_20_20 | physical_time_candidate | False | True | [duca_unified_confirmation_c11_seed5407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_c11_seed5407.py) |
| 30 | confirmation_d1_seed5407 | confirmation | D1 | 5407 | attribution | semantic | robust_phase | adaptive | False | False | signed_feature_taylor | False | default_20_20_20 | attribution_candidate | False | True | [duca_unified_confirmation_d1_seed5407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_d1_seed5407.py) |
| 31 | confirmation_e01_seed5407 | confirmation | E01 | 5407 | physical_time_x_mod | semantic | robust_phase | adaptive | False | False | baseline | True | default_20_20_20 | efficiency_candidate | False | True | [duca_unified_confirmation_e01_seed5407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_e01_seed5407.py) |
| 32 | confirmation_f11_seed5407 | confirmation | F11 | 5407 | attribution_x_mod | semantic | robust_phase | adaptive | False | False | signed_feature_taylor | True | default_20_20_20 | efficiency_plus_attribution_candidate | False | True | [duca_unified_confirmation_f11_seed5407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_f11_seed5407.py) |
| 33 | confirmation_u0_seed6407 | confirmation | U0 | 6407 | anchors | none | exact_uniform | none | False | False | baseline | False | default_20_20_20 | `<empty>` | False | True | [duca_unified_confirmation_u0_seed6407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_u0_seed6407.py) |
| 34 | confirmation_h0_seed6407 | confirmation | H0 | 6407 | anchors | semantic | h65_original_retention_transition | h65_original | False | False | baseline | False | default_20_20_20 | `<empty>` | False | True | [duca_unified_confirmation_h0_seed6407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_h0_seed6407.py) |
| 35 | confirmation_a10_seed6407 | confirmation | A10 | 6407 | prior_x_allocation | semantic | legacy_dual_phase | legacy | False | False | baseline | False | default_20_20_20 | `<empty>` | False | True | [duca_unified_confirmation_a10_seed6407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_a10_seed6407.py) |
| 36 | confirmation_a11_seed6407 | confirmation | A11 | 6407 | prior_x_allocation | semantic | robust_phase | adaptive | False | False | baseline | False | default_20_20_20 | primary_dense_candidate | True | True | [duca_unified_confirmation_a11_seed6407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_a11_seed6407.py) |
| 37 | confirmation_c11_seed6407 | confirmation | C11 | 6407 | phase_x_physical_time | semantic | robust_phase | adaptive | False | True | baseline | False | default_20_20_20 | physical_time_candidate | False | True | [duca_unified_confirmation_c11_seed6407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_c11_seed6407.py) |
| 38 | confirmation_d1_seed6407 | confirmation | D1 | 6407 | attribution | semantic | robust_phase | adaptive | False | False | signed_feature_taylor | False | default_20_20_20 | attribution_candidate | False | True | [duca_unified_confirmation_d1_seed6407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_d1_seed6407.py) |
| 39 | confirmation_e01_seed6407 | confirmation | E01 | 6407 | physical_time_x_mod | semantic | robust_phase | adaptive | False | False | baseline | True | default_20_20_20 | efficiency_candidate | False | True | [duca_unified_confirmation_e01_seed6407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_e01_seed6407.py) |
| 40 | confirmation_f11_seed6407 | confirmation | F11 | 6407 | attribution_x_mod | semantic | robust_phase | adaptive | False | False | signed_feature_taylor | True | default_20_20_20 | efficiency_plus_attribution_candidate | False | True | [duca_unified_confirmation_f11_seed6407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/90b81a6b09388b11e67c9d62a070a9285b4cc3b7/configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_confirmation_f11_seed6407.py) |

## Required Checks

请按以下三类逐行检查，每个问题必须给出精确 file:line。没有出处不要报。

### 1. 检查与设计是否一致

- 每个矩阵 row 是否匹配 manifest 的 arm 定义：panel、prior、allocation、quota、curvature、physical_time、attribution、mod、schedule、role、seed、phase、confirmation。
- 生成配置是否保持 T=768、K=384、terminal epoch 59、terminal EMA、保存 raw prediction、禁止加载 cached raw prediction。
- robust phase fields 是否只使用 semantic/motion prior evidence，没有 GT、annotation、held-out prediction cache 或 teacher leakage。
- continuous-time physical geometry 是否拒绝重复/非法 timestamp，而不是静默修复。
- signed feature Taylor 是否实现 `relu(-(detached_gradient * detached_feature).sum(channel))`，并按 successful-step period 更新。
- A-MoD 是否使用 successful optimizer updates、valid-token top-k routing、capacity-one dense parity 和 dense companion diagnostics。
- aggregate/bootstrap 是否在缺失输出时 fail closed，而不是伪造指标。

### 2. 是否存在时间错误或路线错误

- 日期、matrix ID、branch、base revision、final commit、remote path 是否一致。
- integration base `95ca6eb4a7e0ba8259c5afd976cc30d0fea58865` 是否只被当作实现基座，而不是实验证据。
- submit path 是否使用 final-SHA/timestamped experiment root，而不是误用旧的固定 runs 路径作为正式提交路径。
- `submit_all.sh` 是否接受并真正使用 `--repo-root`、`--revision`、`--run-root`、`--base`、`--account`、`--partition`、`--qos`、`--max-concurrent`。
- Slurm DAG 是否严格是 preflight -> train_eval -> cost/bootstrap -> finalizer，audit afterany。
- cost array 5、bootstrap shards 16、train/eval array 41、max concurrency 8 是否在 manifest、generator、scripts 中一致。

### 3. 是否出现前后矛盾

- 是否有文件在真实 `sbatch` Job ID 产生前声称 `DEPLOYED` 或填入真实 job id。
- 是否有文件在远端训练完成前声称 THUMOS14 mAP、cost reduction、latency gain、bootstrap CI 或 `A11 - A10` 胜利。
- README、freeze doc、本地 implementation report、matrix files、Slurm scripts、generated configs 是否在矩阵数量、seeds、primary contrast、branch/commit 身份上互相矛盾。
- historical H65/dense AdaTAD 数值是否清楚标为 descriptive anchors，而不是 matched controls。
- GitHub 链接是否指向实际已推送的分支和 commit。

## Expected Output

先列 findings，按 P0/P1/P2/P3 排序。每条 finding 必须包含：

- Severity
- Claim
- Evidence: file:line citation and GitHub link when useful
- Why it matters: design mismatch, time/route error, or contradiction
- Minimal fix

如果没有发现问题，请明确说没有，并列出仍未验证的风险，尤其是远端 Slurm 执行、真实 mAP/cost/bootstrap 结果。

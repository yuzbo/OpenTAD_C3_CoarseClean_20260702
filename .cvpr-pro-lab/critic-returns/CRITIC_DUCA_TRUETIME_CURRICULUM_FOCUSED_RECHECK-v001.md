# CRITIC_DUCA_TRUETIME_CURRICULUM_FOCUSED_RECHECK-v001

- verdict: `DYNAMIC_TRUETIME_FOCUSED_STATIC_BLOCKED`
- admission: `BLOCKED_PRE_RUN`
- evidence_class: `STATIC_IMPLEMENTATION_ONLY / NO_DATA / NO_GPU / NO_METRIC`
- scientific_question: 在历史 ASFormer 间接逐帧选取、固定 `K=384` 和总计 60 epoch 的课程训练中，若从重型 VideoMAE 的第一次时间混合开始持续使用真实物理时间，能否恢复高 IoU 定位性能？
- frozen_design: `E:/DeskTop/TAD/OpenTAD_DUCA_TrueTimeCurriculum_20260821/docs/superpowers/specs/2026-08-21-duca-truetime-indirect-curriculum-design.md`
- worktree: `E:/DeskTop/TAD/OpenTAD_DUCA_TrueTimeCurriculum_20260821`
- branch: `codex/duca-truetime-curriculum-20260821`
- historical_parent: `42dba3f90b37243e7965d18b6707e88e81bf7109`
- design_commit: `d712df7f582c8a35af9f0f2e1b2ff4c81f90c61d`
- initial_candidate: `e6708ef8d8669cb239dafe65a18dafaa3be67743`
- final_candidate: `60816c9ba359c17491f48bae76b84513d532a6a0`
- binding: final candidate worktree is clean at the exact revision above.

## Terminal findings

1. `PhysicalTubeletPatchEmbed` is defined in `opentad/models/backbones/vit_adapter.py:19-53`, but the production `VisionTransformerAdapter` still constructs the ordinary Conv3D `PatchEmbed` at lines 813-821 and unconditionally calls `self.patch_embed(x)[0]` at lines 879-895. No production config instantiates the new operator, so source positions are not consumed before the first temporal convolution.
2. `BackboneWrapper` conditionally forwards `source_positions` only when the backbone exposes `physical_time`; the production VideoMAE forward accepts only `x`. Its `TypeError` fallback therefore executes the unchanged rank-packed path. The branch is not an active true-time computation path.
3. The metadata reshape keeps a length-384 selected-position row while `PhysicalTubeletPatchEmbed` requires one `[B,16]` position row per 16-frame clip. Even if the branch were forced, the current tensor contract is inconsistent.
4. The candidate does not consume returned physical midpoints/supports in the detector path, does not build a physical detector grid, and does not remove the selected-axis time warp. It therefore cannot establish input-to-detection physical-time consistency.
5. The declared 20/20/20 curriculum is configuration text only. No verified training consumer binds 2,000 warmup updates, 2,000 cosine-transition updates and 2,000 joint updates to the existing optimizer-step schedule.
6. Both launcher prechecks still read `c.requested_k`, although the value is nested under `c.experiment_scope.requested_k`; the launcher cannot admit either arm. The recovery lifecycle and the optional `RANKPACK_J192` diagnostic are also not implemented.

## Scientific boundary

This verdict does **not** refute the historical indirect-selection hypothesis or the physical-time hypothesis. No Evaluator PRE_RUN, dataset traversal, remote job, optimizer update, detection metric or cost measurement occurred. It only establishes that the frozen candidate does not execute the claimed mechanism and is therefore inadmissible for experiment.

## Terminal disposition

- correction_budget: exhausted after the bounded implementation cycle; no fourth focused correction is permitted.
- evaluator: dormant; structural intake was not dispatched.
- remote_experiment: not launched.
- next_owner: `Coordinator terminal hold`.
- next_action: do not submit this candidate. Continuation requires a genuinely new, separately frozen implementation cycle that integrates physical-time tubelet formation into the actual VideoMAE forward and detector coordinate path, or an explicit scientific simplification to a post-backbone PhysTime diagnostic. The latter changes the tested claim and cannot be silently substituted.
- dependency: a new accepted implementation disposition; current candidate `60816c9b` remains immutable failure evidence.
- expected_return_at: no automatic return while terminal hold remains.
- single_recovery: `none`.

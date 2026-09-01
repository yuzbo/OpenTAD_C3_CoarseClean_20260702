---
artifact_id: CRITIC_DUCA_DENSITY_UOR_IMPLEMENTATION_INDEPENDENT_PRE_RUN_REVIEW-v001
role: Critic
kind: UOR_TRACKED_PACKAGE_INDEPENDENT_REVIEW_V001
status: REACHABILITY_PRE_RUN_BLOCKED
finding_classification: IMPLEMENTATION_CORRECTION
scientific_ambiguity: NONE
queue_message: msg-20260814T003017Z-2d29f401c190
parent_artifacts:
  - PRO_DUCA_DENSITY_UOR_IMPLEMENTATION-v001
  - DUCA_DENSITY_REACHABILITY_PROTOCOL-v001
  - BUILDER_DUCA_DENSITY_UOR_IMPLEMENTATION-v001
reviewed_revision: 7f07e4545fafda5ca9b86ead14a089b3515a06d0
review_binding: CLEAN_PASS
review_cwd: C:/Users/skywalker/.codex/worktrees/24ef/OpenTAD_C3_CoarseClean_20260702-critic-7f07e454
execution_state: NOT_EXECUTED
scientific_evidence_status: BLOCKED_PRE_RESULT
next_owner: Builder
---

# Verdict

`REACHABILITY_PRE_RUN_BLOCKED`.

Three deterministic implementation findings remain. They do not require a scientific choice and must not be routed to Pro. The smallest dependency chain is: Builder correction on a new tracked revision, then one focused independent Critic recheck, then Coordinator/Evaluator PRE_RUN assembly and admission. This receipt grants no training, inference, evaluator, metric, remote, GPU, Slurm, or scientific authority.

# Findings

## DUCA-REACH-IMP-001 — official-validation/data firewall is not executable

Classification: `IMPLEMENTATION_CORRECTION`.

The detector-FIT config overrides only `dataset.train.ann_file` (`configs/adatad/thumos/duca_density_detector_fit_n16r4.py:101-103`). Its inherited `dataset.val` and `dataset.test` remain validation-subset loaders (`configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py:41-72`) and are further bound to the inherited validation-video root `/root/autodl-tmp/test` (`configs/adatad/thumos/input_random_fixed_50pct_c3_physical_grid_actionformer_precheck.py:77-107`). `tools/train.py` constructs both loaders unconditionally before model construction, even when the launcher supplies `--not_eval` (`tools/train.py:140-169`). Reader FIT inherits the same behavior.

The CAL config replaces the annotation and subset for `val`/`test` but does not replace their inherited `data_path` (`configs/adatad/thumos/duca_density_cal_uor_n16r4.py:78-82`), so CAL training-population videos are still resolved through the inherited validation-video root. Under the frozen validation-deny seal this either accesses a forbidden root or fails because that root is absent. It cannot satisfy the exact FIT/CAL derivative firewall and official-validation deny.

Smallest owner/action: Builder must make FIT entrypoints construct only FIT-bound loaders (or bind every unavoidably constructed loader to the sealed FIT annotation and allowed training-video root), and bind CAL `val`/`test` explicitly to the sealed CAL annotation and allowed training-video root. Add a focused resolved-config/entrypoint test proving that no FIT, reader-FIT, or CAL process can construct an official-validation loader. Dependency: corrected tracked source/config revision before Critic recheck.

## DUCA-REACH-IMP-002 — future argv/artifact/gate wiring is not fail-closed

Classification: `IMPLEMENTATION_CORRECTION`.

The frozen launcher invokes `python tools/train.py` and `python tools/test.py` directly (`scripts/run_duca_density_reachability_n16r4.sh:45-56,70-91`), while both entrypoints require `LOCAL_RANK`, `WORLD_SIZE`, and `RANK` before dataset/model access (`tools/train.py:109-119`; `tools/test.py:133-143`). Checking only `SLURM_JOB_ID` does not establish those variables or the required distributed launch context.

The launcher also seals paths that OpenTAD does not produce. `update_workdir` appends `gpu{N}_id{ID}` (`opentad/utils/misc.py:25-27`), checkpoints are written below `work_dir/checkpoint/epoch_{epoch}.pth` (`opentad/utils/checkpoint.py:22-49`), and the 60-epoch loop saves zero-based epoch `59` (`tools/train.py:263-264,304-313`). The launcher instead expects `detector_fit/epoch_60.pth` and `reader_fit/epoch_60.pth`, and similarly expects CAL predictions directly below `cal_U`, `cal_O`, and `cal_R` (`scripts/run_duca_density_reachability_n16r4.sh:46-62,72-95`). The first phase cannot reach its declared completion seal.

Separately, the entrypoint gate explicitly disables the resolved-config digest (`configs/adatad/thumos/duca_density_detector_fit_n16r4.py:39-62`) and the launcher exports it as empty (`scripts/run_duca_density_reachability_n16r4.sh:19-27`). The strict gate payload binds phase labels but not the actual FIT/CAL annotation paths, checkpoint paths, work directories, active arm, exact argv/cwd, or Slurm launch identity. Those values are read directly from mutable environment variables (`duca_density_detector_fit_n16r4.py:11,113-116`; `duca_density_reader_fit_n16r4.py:9-12,64,73-76`; `duca_density_cal_uor_n16r4.py:9-14,75,125`). A direct entrypoint invocation can therefore reuse a valid phase gate with a different runtime binding.

Smallest owner/action: Builder must use the exact supported single-process distributed argv, align every declared/sealed artifact with the actual OpenTAD work-directory and checkpoint/result layout, and require a nonempty resolved-config identity plus equality checks for every phase-critical argv/environment/path/arm/cwd/Slurm binding before loader, checkpoint, or output access. Dependency: corrected launcher, configs, guard, and focused fail-closed tests before Critic recheck.

## DUCA-REACH-IMP-003 — arm/receipt/final sealing does not bind evaluator inputs

Classification: `IMPLEMENTATION_CORRECTION`.

`build_phase_completion_receipt` marks a receipt `COMPLETE_SEALED` and chmods only the single artifact (`tools/bata/duca_density_reachability.py:232-294`). The receipt is written afterward by `_write_json_exclusive`, which performs exclusive file publication but leaves the receipt writable (`:556-577,667-678`); no arm root is sealed non-writable. Predecessor and embargo validation then trust the mutable receipt's declared state/identity (`:246-262,205-229`).

The evaluator accepts three arbitrary prediction paths on its CLI (`:597-606`) and reads them independently of the receipt `artifact` fields (`:649-665`). It never proves that the prediction objects being evaluated are the objects sealed by the U/O/R receipts. Finally, it writes one exclusive result file but neither seals that receipt nor atomically publishes a dedicated final result root (`:556-577,665`). This violates the frozen requirement that all three arm roots and immutable phase receipts be sealed before metrics, that the metric process receive those exact sealed roots simultaneously, and that the sealed final receipt be atomically published as the final result root.

Smallest owner/action: Builder must seal each complete arm root and its completion receipt; bind each evaluator input to the exact artifact/root named by its independently verified receipt; reject writable, mismatched, missing, or substituted inputs before evaluator import; and seal plus atomically publish a dedicated final-result root. Focused tests must cover a substituted prediction path, writable arm/receipt, mutable predecessor receipt, and unsealed final publication. Dependency: corrected tracked tooling/launcher revision before Critic recheck.

# Independently accepted static surfaces

- The reviewed HEAD is exactly `7f07e4545fafda5ca9b86ead14a089b3515a06d0`; the named worktree porcelain was empty, and the implementation has no untracked prototype dependency.
- The manifest builder/validator deterministically partitions unique whole-video IDs, preserves source order, enforces exact disjoint union/counts, records derivative inheritance, denies official validation declaratively, and verifies its canonical digest (`tools/bata/duca_density_reachability.py:56-146`). Runtime admission remains blocked by DUCA-REACH-IMP-001/002.
- U/O/R use the same selector path and differ in config only by the density-logit source (`configs/adatad/thumos/duca_density_cal_uor_n16r4.py:67-73,97-121`). GT handoff is source-gated to O (`opentad/models/detectors/base.py:23-34`; `opentad/models/detectors/actionformer.py:215-232`). U/R reject forbidden GT/teacher/cache/prediction/result/metric payloads, and R consumes the learned browser-memory density logits (`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:1914-1996`). Raw-prediction cache loading/saving is denied on the live path (`opentad/models/detectors/base.py:36-61`).
- Constant logits return the canonical uniform integer tensor directly, and the nonconstant route uses the current production fixed-point projector and verifies the exact objective certificate (`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:450-533`). The focused no-data suite passed independently in the named `open_mmlab` environment: `16 passed, 1 warning`.
- The paired whole-video bootstrap uses the exact frozen prefix/nonce SHA-256 seed derivation, PCG64, 10,000 paired cluster draws, percentage-point differences, and nearest ranks 500/9500 (`tools/bata/duca_density_reachability.py:14-18,297-299,311-401`). No metric or evaluator was executed in this review.

# Review boundary

This review used only bounded repository/control-plane reads, frozen Git identity/status inspection, and the focused CPU synthetic/no-data test module. It did not access or list data roots, official validation, models, checkpoints, real videos, predictions, metrics, browser/Sources/Pro, remote hosts, GPU/CUDA, Slurm, training, inference, or evaluator execution. No repository file was patched, deleted, cleaned, or otherwise changed; this durable Critic return is the sole write.

CRITIC_DECISION: REACHABILITY_PRE_RUN_BLOCKED.

# DUCA `043be401` Pro Audit Absorption

## Source

- Attachment: `48c9c615-e001-40cb-8207-951cb504198f/pasted-text.txt`
- Exact archive: `docs/methods/reviews/2026-07-15-043be401-duca-exact-commit-pro-audit-review-raw.txt`
- SHA-256: `1D395844396D644295BF83BF08753C14B2E638295B8C37D15048924B0F415FC9`
- Audit target: `043be401ba2b694342dc395f263e9a9858628d69`
- Reviewer visibility: `VISIBLE_WITH_EXTERNAL_LIMITS`; repository code was read, but remote Slurm artifacts, the external ASFormer bytes, the VideoMAE checkpoint, and current mAP were not independently read.

## Reviewer Verdict

The reviewer returned `GO` for completing the four exact-commit seed-0 jobs and `HOLD` for paper claims. It found no confirmed P0 invalidating the running suite. It reported two P1 findings:

1. THUMOS test data is used by repeated intermediate evaluations while the post-run contract still permits `best_or_final`, so the primary checkpoint is not sealed against test-set model selection.
2. Counterfactual distillation ranks candidate swaps only. Without a no-op class at utility/score zero, it does not preserve whether every candidate is beneficial or harmful relative to retaining the baseline selection.

## Local Recheck

The following findings were independently confirmed against the exact local commit:

- `counterfactual_utility_distillation_loss` applies cross-entropy only across valid candidates. It is a relative ranking objective and has no no-op anchor.
- Formal transition arms set direct detector-gradient weight to zero. The counterfactual arm uses detached detector-derived one-swap ranking distillation; it is not direct detector-loss backpropagation through selection.
- The actionness head is binary, but the shared ASFormer hidden route and transition scorer receive train-only endpoint-derived transition-density supervision. This matches the intended separation between coarse state learning and boundary-oriented selector supervision, but it forbids a whole-model `binary-only` claim.
- Hard and relaxed structured paths share each sample's valid prefix, effective K and max-hole feasible family. Exact uniform uses rounded endpoint linspace rather than the invalid legacy midpoint tie-break.
- The detector consumes hard-gathered RGB frames and performs assignment on the selected axis before inverse true-time remapping. The mapping is internally consistent, while nonuniform selected-axis receptive-field geometry remains a genuine mechanism confound.
- With `val_start_epoch=47`, anchor `47`, interval `5`, the outer zero-based guard and inner one-based schedule produce the first evaluation after one-based epoch 52, not epoch 47.
- The so-called DDP pilot is a one-rank DDP-wrapper pilot. This is sufficient for the current one-GPU jobs but is not evidence about multi-rank synchronization.
- The runtime source SHA is bound consistently across arms, but the actual normalized-LF ASFormer hash is not fail-closed against the expected config constant.

## Project Assessment

Verdict: `PARTIAL_ACCEPT / AGREE_WITH_CORE_VERDICT_NOT_ALL_RECOMMENDATIONS`.

### Subsequent Evidence Supersession

The review did not inspect remote Slurm artifacts. A later live audit found that
all four formal arms had missed successful optimizer updates relative to the
declared schedule. That evidence supersedes only the review's run-qualification
judgment: Jobs `1164700-1164703` may continue as diagnostic runs, but they are
not valid formal matched evidence. The code/claim findings below remain useful.

### Accepted

- At the time of this code-only review, no confirmed code-level P0 justified
  cancellation. Later remote evidence limits Jobs `1164700-1164703` to
  diagnostic use; their completion cannot prove C3/C4.
- Keep C3/C4 `unproven` until matched results exist.
- Seal a primary checkpoint policy before the first test evaluation. For this running suite, the cleanest non-restart rule is final one-based epoch 132, `state_dict_ema`; every intermediate test mAP is diagnostic only.
- Describe the current C4 arm as `relative hard one-swap ranking distillation`, not signed utility learning and not direct detector-gradient learning.
- Treat the selected-axis geometry, full-stack cost, simple non-learned transition proxy, short-action/high-tIoU behavior, and endpoint coverage as required post-run diagnostics.
- Report matched-uniform training control separately from a bare exact-uniform deployment/cost baseline.
- Treat direct-a5 as a system-level explicit-boundary baseline, not a single-factor causal ablation.
- Keep dynamic budget, X3D/SlowFast, MUST and extra selector complexity frozen until fixed-384 C3/C4 are decided.

### Not Fully Accepted

- Adding a no-op class to one softmax is a good minimal signed-anchor candidate, but it is not uniquely optimal. A no-op-anchored categorical loss couples calibration to candidate count; an independent baseline-vs-swap logistic/Bradley-Terry target or calibrated utility regression is also plausible. Only one bounded follow-up should be chosen after the current C4 result and gradient/utility diagnostics.
- Disabling every intermediate evaluation is unnecessary for the running suite. The integrity requirement is to predeclare and hash-bind the primary result; diagnostic evaluations may continue if they cannot select the checkpoint.
- A global rewrite of `train_schedule.py` should not be accepted from the review patch without regression checks across non-DUCA configs. The immediate correction is documentation/monitoring plus a DUCA-scoped future protocol field.
- A two-rank DDP pilot is not required for the current one-GPU formal protocol. It becomes mandatory only before multi-GPU execution or a multi-rank robustness claim.
- A physical-time-aware head is not the default next implementation. Existing PhysTime diagnostics already show that naive physical-coordinate assignment can reduce eligible positives. First run a fixed-selection geometry diagnostic; redesign the head only if selector quality improves while TAD remains limited and the residual correlates with local sampling gaps.
- Endpoint-derived supervision on the selector does not violate the project's original intention. The intended claim is binary action-state supervision for the coarse head plus boundary-oriented train-only supervision for indirect selection.
- A second detector is required only for a detector-agnostic/plugin-generality claim. The paper may instead scope evidence to the AdaTAD/ActionFormer backend.
- Fixed final EMA is the correct sealed primary for this seed-0 screening, but it is not automatically the only defensible final-paper protocol. A future paper protocol may use a genuinely train-derived held-out validation split or another pre-registered fixed checkpoint rule.

## Evidence and Claim Impact

- Implementation status remains `tested`.
- Formal matched-evidence status is invalidated; the jobs remain
  `experiment_running` only as diagnostics.
- C3 (`transition beta0 > matched exact-uniform`) remains `unproven`.
- C4 (`counterfactual > beta0`) remains `unproven`; the current experiment tests relative ranking distillation only.
- Signed counterfactual utility is `not implemented`.
- End-to-end cost saving, selected-axis attribution, second-detector generality, and paper readiness remain `unproven`.
- The review is design/code-audit evidence, not an experiment result.

## Bounded Next Actions

1. Do not use the exact-commit jobs as formal matched evidence. They may finish
   unchanged for diagnosis, but any replacement formal run requires a newly
   audited exact commit and successful-update gate.
2. Before the first evaluation, hash-bind a study-level primary-result declaration: final one-based epoch 132, checkpoint file `epoch_131.pth`, `state_dict_ema`, intermediate test mAP diagnostic only. This was completed at 2026-07-15T12:16:50Z with SHA-256 `AAC0FCA8671AE6F58CF4C9B5D4D40282BE714AA354028246E86504FD39C89B48`; the root and four variant copies are byte-identical and read-only.
3. Correct monitoring language from `first evaluation at epoch 47` to `first evaluation after one-based epoch 52` for this commit.
4. Finish the four arms and decide C3/C4 from the sealed final checkpoint before implementing a new counterfactual loss.
5. If C4 is non-positive or the utility-sign diagnostics fail, run exactly one signed-anchor follow-up chosen after comparing no-op categorical and baseline-vs-swap logistic formulations. Do not sweep weights, candidate counts or temperatures first.
6. If learned selector quality improves but TAD does not, run the selected-axis geometry attribution diagnostic before any physical-time head implementation.

---
updated: 2026-07-07
status: active
scope: Absorbed review explaining GAS-VT early plateau and PAction learned fixed_384 advantage
out-of-scope: Treating the explanation as final causal proof, changing running experiments, or making paper claims
---

# GAS-VT Plateau And PAction Advantage Review Absorption

Raw record: `docs/methods/reviews/2026-07-07-gasvt-plateau-paction-advantage-review-raw.txt`

Related evidence note: `docs/methods/2026-07-07-gasvt-paction-diagnosis-evidence.md`

## Absorption Verdict

I accept this response as a strong working diagnosis, not as complete proof.

It directly answers the question that was still under-specified in the previous review: why GAS-VT can reach around 40 mAP very early, why it then plateaus, and why PAction learned fixed_384 can be stronger. The answer is consistent with our current log evidence and ledger summaries, but it still requires controlled matched reruns and ablations before we can treat it as causal.

The key conclusion to absorb is:

- GAS-VT early progress is likely mostly due to half-density fixed_384 input plus useful p_action coarse action signal.
- GAS-VT later plateau is likely due to missing detector-aware boundary utility, possible train/apply budget-conditioned feature mismatch, non-sequential decoding, and over-regularizing gap/coverage/repair constraints.
- PAction learned fixed_384 is likely stronger because it is simpler, more direct, less constrained, and preserves boundary-relevant p_action-derived evidence better.
- This does not make PAction the final method. It makes PAction learned fixed_384 the current strongest Stage1 baseline that Stage2/Stage3 must beat.

## What This Review Adds

Compared with the previous HOLD review, this one provides a clearer mechanistic explanation:

1. Fixed_384 is not extremely sparse.
   - Selecting 384 out of 768 already exposes half the temporal grid.
   - This is enough for AdaTAD to learn coarse action presence and low-IoU localization quickly.
   - Therefore an early 40 mAP result is not proof of the GAS-VT mechanism.

2. Current GAS-VT is not faithful sequential value transport.
   - The implementation has state-like features such as remaining budget, gap urgency, and distance to previous selection.
   - The actual decode path still behaves like score ranking plus constrained top-k/repair.
   - Without autoregressive selection and state update, the method should be described as an engineered gap-aware constrained selector, not as full sequential value transport.

3. GAS-VT may have a train/apply distribution shift.
   - Training constructs features with a target budget.
   - Apply-time checkpoint scoring can construct features without the same concrete target budget.
   - This changes remaining_budget, budget_pressure, gap_urgency, and related feature semantics.
   - This is a P0 implementation risk for both Stage1 GAS-VT and any Stage2 detector-aware selector that reuses the same feature machinery.

4. GAS-VT regularization can fight detector utility.
   - Value BCE, boundary coverage, boundary bracket, action interior, CVaR max-hole, budget, and p_action dependence can push selection toward coverage and uniform-like behavior.
   - High-IoU TAD needs boundary-discriminative evidence and coordinate geometry, not just small holes and broad coverage.
   - This can explain why GAS-VT improves coarse mAP early but does not keep improving high-IoU mAP.

5. Hard repair can create uniform-like outputs without explicit uniform fill.
   - `uses_uniform_fill=False` and `uses_uniform_scaffold=False` are not enough.
   - A max-hole repair pass can still force selected positions into a nearly uniform lattice.
   - Therefore `uniform_similarity` and repair statistics must be measured after final hard decode, not inferred from metadata flags.

6. PAction learned may win for conservative reasons.
   - It is closer to p_action ranking, deltas, and boundary-related positions.
   - It has fewer competing constraints.
   - Fixed_384 is high budget, so strong hole and boundary statistics can be obtained without a truly detector-aware policy.
   - Its advantage should not be overclaimed as solving final sparse TAD acquisition.

## Alignment With Current Evidence

This review matches our current evidence note in several ways:

- GAS-VT fixed_384 rose quickly from early low mAP to around 40 mAP, then stayed near the mid-40s.
- PAction learned fixed_384 reached a much stronger high-IoU regime in the available run.
- PAction validation summaries showed much better boundary support and dramatically smaller p95 unselected hole than GAS-VT, despite similar selected-count means.
- PAction's advantage was not explained by simple p_action top-k imitation because measured top-k Jaccard was not higher than GAS-VT.

The review also explains why the p95 hole and uniform-like observations are not contradictory:

- A method can report no uniform fill/scaffold and still become uniform-like after hard repair.
- A method can show acceptable mean selected count while still having harmful per-row geometry.
- A method can improve coarse mAP while failing to improve boundary geometry needed for mAP@0.6 and mAP@0.7.

## Claim Boundary

This response tightens the claim boundary:

| Claim | Current status | Reason |
|---|---:|---|
| GAS-VT is the main publishable method | FAIL for now | It is not yet true sequential VT and may be over-regularized |
| GAS-VT early 40 mAP proves its mechanism | FAIL | Fixed_384 density plus p_action can explain it |
| PAction learned fixed_384 is the best current Stage1 baseline | WARN/HOLD | Supported by current logs, but not yet matched same-commit proof |
| PAction learned solves detector-aware acquisition | FAIL | It remains p_action/GT-surrogate supervised Stage1 selection |
| Stage2 teacher utility is the right next scientific step | PASS as plan | It directly tests detector utility beyond p_action |
| Stage3 joint selector-detector training is required for end-to-end claim | PASS as plan | It is the first honest end-to-end route |

## Required Evidence Before Final Diagnosis

The review says the current explanation is plausible, but incomplete. We must collect:

1. Matched mAP curves.
   - GAS-VT and PAction under the same commit, same data, same pretrain, same AdaTAD config, same eval epochs.
   - Report Avg mAP plus mAP@0.3/0.5/0.6/0.7, not only Average-mAP.

2. Matched ledger statistics.
   - selected-count histograms;
   - boundary support;
   - max/p95/p99 unselected hole;
   - gap CV;
   - final selected-position uniformity;
   - overlap with p_action top-k, delta top-k, and boundary-score top-k.

3. GAS-VT implementation ablations.
   - target-budget apply fix vs old apply;
   - true sequential decode vs current constrained top-k;
   - with and without CVaR max-hole;
   - with and without hard repair;
   - with and without boundary bracket/action interior losses.

4. PAction learned ablations.
   - PAction learned vs pure p_action top-k at fixed_384;
   - PAction learned with and without boundary loss;
   - PAction learned with and without the same max-hole repair;
   - PAction fixed_384 vs fixed_768 vs dynamic with matched selected-count reporting.

5. Stage2/Stage3 evidence.
   - Dense AdaTAD teacher utility exported from train split only.
   - Stage2 detector-aware selector must beat PAction under matched budgets.
   - Stage3 must prove detector loss gradients reach selector parameters and improve mAP/high-IoU without collapse.

## Concrete Work Queue

P0:

- Fix apply-time target-budget conditioning for GAS-VT.
- Fix any inherited apply-time budget conditioning in Stage2 detector-aware policy.
- Add post-decode hard-repair metrics and richer uniformity metrics.
- Add top-k overlap and selected-count histograms to validation summaries.
- Preserve the current PAction learned run as a strong Stage1 baseline, not as the final method.

P1:

- Run matched Stage1 matrix after the fixes:
  - pure p_action top-k fixed_384;
  - PAction learned fixed_384;
  - PAction learned fixed_384 without boundary loss;
  - PAction learned fixed_384 with and without repair;
  - GAS-VT fixed_384 old/fixed budget-conditioning;
  - GAS-VT no CVaR/no hard repair;
  - uniform/random fixed_384 anchors.
- Report all results with mAP@0.6 and mAP@0.7 highlighted.
- Continue dense teacher full training and use it to enable Stage2.
- Start Stage3 full only after current GPU1 Stage1 baseline releases or after a clean allocation is available.

P2:

- Rename or demote GAS-VT in the paper story unless a true sequential implementation is added and shown to help.
- Treat PAction as the empirical Stage1 baseline that motivates detector-aware acquisition, not as the end contribution.
- Frame the paper around pre-backbone detector-utility-aware temporal acquisition rather than handcrafted gap regularization.

## Paper Story Update

The paper story should not be:

> We designed GAS-VT and it outperforms PAction.

The evidence and review now push the story toward:

> PAction and GAS-style selectors reveal that p_action-supervised sparse acquisition can preserve enough coarse signal for AdaTAD, but handcrafted gap/coverage objectives saturate and can damage boundary geometry. The actual research contribution must be detector-utility-aware temporal acquisition: first distilling dense AdaTAD teacher utility into a selector, then jointly optimizing selector and detector with true-time coordinate preservation and anti-collapse constraints.

This keeps the original long-term goal intact while demoting a weaker intermediate mechanism.

## Absorption Status

The review has been recorded and absorbed as a claim-locking diagnosis. It strengthens the decision to:

- keep GAS-VT as Stage1 baseline/safety anchor;
- use PAction learned fixed_384 as the current strong Stage1 comparator;
- prioritize Stage2 dense AdaTAD teacher utility and Stage3 joint training;
- require matched ablations before explaining PAction's advantage as causal.

No running experiment was changed by this record.

# DUCA Physical Cliplet Dynamic Budget — fresh Project Pro adversarial review

Fresh nonce: `DUCA-PHYSICAL-CLIPLET-CONTRACT-20260820T173128Z-82c432fb0919137d1a4a5a99`

You are the independent Scientific First-Author Agent for this DUCA review turn. Work simultaneously from two demanding perspectives:

1. the harshest technically competent CVPR reviewer, actively trying to reject the work for weak causality, unfair comparison, hidden compute, leakage, temporal-geometry errors, redundant ablations, unoriginality, or an unconvincing paper story;
2. an excellent temporal action detection researcher with strong scientific taste, responsible for turning a viable kernel into the smallest elegant, falsifiable and publishable study.

Do not merely summarize or agree. Return exactly one overall disposition: `CONTINUE`, `REVISE`, `PIVOT`, or `STOP`. Make the scientific choice yourself; do not return it to the human or Codex. Write the complete answer in clear Chinese suitable for an expert committee.

## Exact Project and evidence identity

- Exact ChatGPT Project ID: `g-p-6a796fef9a00819194024cf1de3bd697` (DUCA).
- Canonical public repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- Public historical branch: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-density-transport-20260723
- Public historical exact commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/42dba3f90b37243e7965d18b6707e88e81bf7109
- The current local coordinator root is dirty and its HEAD `a6bdc084...` belongs to a contaminated SparseHead history. It is not an admissible DUCA implementation identity.
- The attached frozen contract is `designed_not_implemented_not_tested`. There is no clean implementation revision, PRE_RUN admission, current-route training, current-route mAP, or current-route cost result.
- UVT and Fovea code commits are locally retained but not visible on the public remote: UVT deployment `df544c78ce515d925dc7019f106fce09a53c09f8`; Fovea training `4ae5067100c4490c7110c00a1ad406230ba603cd`. Treat their reported metrics as supplied experiment receipts, not as independently GitHub-verified code facts.

## Human-frozen scientific boundary

The headline must remain indirect semantic acquisition: a cheap dense scout predicts binary actionness plus start/end boundary importance; a deterministic acquisition rule derives frame/cliplet value and a per-video or per-window dynamic outer budget. Fixed K is only a control or fallback. A small model directly predicting indices cannot be the headline. Test-time GT, teacher, cycle target, detector prediction cache, or EMA target cannot choose samples. Physical timestamps must exist before thresholding/top-k/proposal filtering/IoU/NMS. Executed compute must come from actual frames sent through VideoMAE, not requested K or metadata.

The contract proposes a primary input of physically consecutive 16-frame cliplets selected by endpoint coverage, plus a strong `GAPPACK` control that reuses the exact same RGB-frame set, K, order, detector and physical output timestamps but repacks selected-rank, non-uniform frames into 16-frame blocks. It also proposes a fixed-budget semantic gate at `M=24, K=384`:

- `S0`: action/background + start/end + uncertainty scout;
- `SQ`: class-agnostic Query-Bridge helps those semantic predictions but cannot output indices, K or proposals;
- `SQC`: SQ plus train-only detached post-heavy cycle consistency;
- `SQD`: SQC plus train-only detached semantic distillation, absent at inference.

Only after this semantic gate and the CONTIG/GAPPACK attribution would the study test dynamic M against fixed M=24, K-shuffle preserving the M histogram/cost strata, and actionness-only dynamic. The exact dynamic support and thresholds must be calibrated on video-disjoint training-population FIT/CAL and sealed before evaluation. Dense, equidistant-uniform and random baselines have already been trained repeatedly in the project; do not prescribe another duplicate training run. Use their immutable receipts as background and demand only a genuinely new mechanism-changing experiment.

## Experiment facts that must constrain the review

1. Historical DUCA selected-rank/non-uniform input reached `65.385724%` at public commit `42dba3f...`, but consumed a 30-epoch warm-up plus a 60-epoch curriculum and was protocol/compute unmatched. Another physical-grid exploration reached `65.696%` and a uniform sparse anchor about `64.352%`; neither is admissible as a clean current claim. They nevertheless show that irregular non-uniform RGB input was not intrinsically catastrophic.
2. A later old-code three-seed matrix (`1244133`, commit `7529fba6...`) reported mean Avg-mAP: dense 68.25, uniform K384 64.23, dynamic K 56.25, dynamic-no-risk 55.94, learned K384 49.44. This snapshot is not matched to UVT/Fovea.
3. UVT job `1244840`, seed 3407, 60 epochs: legacy off 57.35; GT-geometry V(t) 55.93; geometry+self-EMA 55.92. V(t) simultaneously changed selection score and budget evidence, so the negative result is confounded. Foveated decoder and portal detector feedback were off.
4. Fovea/Query-Bridge job `1244851`, seed 3407, 60 epochs: query_cycle 54.67; query_gt_mask 49.16; query_only 45.26; query_fovea 43.77; baseline_fused 42.94. Two planned arms were absent; same-commit exact matched controls were absent. Query-cycle is the best member of that failed family, especially at high IoU, but it does not establish efficacy.
5. Published/released official AdaTAD anchor is approximately 69.03 Avg-mAP, 48.27 at tIoU 0.7; the project-wide released-checkpoint/shared official receipt is still the authority and must not be replaced by 65/66.xx historical numbers.

## Questions you must adjudicate

1. Is the contract's central mechanism scientifically necessary and sufficiently TAD-specific, or is it merely endpoint-biased sampling plus ordinary adaptive compute?
2. Does requiring physical continuity within each 16-frame VideoMAE cliplet impose an unjustified prior? Can `CONTIG vs GAPPACK` with the identical frame set actually isolate temporal presentation, and what residual confounds remain?
3. Are S0/SQ/SQC/SQD causally distinct and minimal? Which query/cycle/distillation mechanism is worth retaining from UVT/Fovea, and which should be deleted before implementation?
4. Does endpoint coverage risk missing action interiors, long actions, multiple events, overlapping boundaries, or background context? Specify the deterministic rule or sufficient statistics that should be frozen before code.
5. Can dynamic M be learned or calibrated without evaluation leakage, hidden dense heavy compute, or post-hoc threshold tuning? Define the cleanest falsifier against fixed M and K-shuffle at matched realized end-to-end cost.
6. What is the most likely explanation for the 10–15 point gap between UVT/Fovea and dense, and which single new experiment best discriminates among input temporal presentation, selector quality, optimization interference, dynamic-budget error, and detector/backbone mismatch?
7. Is the proposed P1→P2→P3 ordering economical, or does it waste full runs? Redesign it if needed, while respecting the prohibition on duplicate dense/uniform/random training.
8. What novelty/prior-art attack would most likely kill a CVPR paper, and what one surprising prediction would make an expert care?

## Required response structure

1. `SESSION_ASSERTION`: echo exact Project ID and nonce; state this is a fresh conversation.
2. `EVIDENCE_BOUNDARY`: distinguish GitHub-verifiable code, supplied receipts, frozen design, and absent evidence.
3. `REVIEWER_REJECTION_CASE`: strongest rejection in no more than eight ranked findings.
4. `SCIENTIFIC_COLLEAGUE_CASE`: what is genuinely valuable, interesting, and worth preserving.
5. `DECISION`: exactly one of CONTINUE/REVISE/PIVOT/STOP, with a single recommended route.
6. `CONTRACT_CORRECTIONS`: exact textual/algorithmic changes to the frozen contract; say plainly which clauses are already correct.
7. `MINIMUM_NONREDUNDANT_EXPERIMENT`: the smallest new real-video experiment that changes the route decision, with arms, frozen variables, seeds, official evaluator, success/failure thresholds, high-IoU/short-action diagnostics and full-stack cost. Do not rerun already completed dense/uniform/random training.
8. `IMPLEMENTATION_CONTRACT`: inputs/outputs, gradients, detach points, physical-time mapping, actual heavy-compute accounting and fail-closed invariants sufficient for a clean Builder plan.
9. `PUBLICATION_JUDGMENT`: paper thesis, closest novelty invalidators, likely reviewer objections, and the evidence needed before any claim.
10. `NEXT_ACTION`: one bounded next owner/action/dependency/return artifact; no generic audit or additional theory round.

Do not authorize a remote experiment or claim that any current-route implementation/result already exists. This turn is scientific adjudication and contract review only.

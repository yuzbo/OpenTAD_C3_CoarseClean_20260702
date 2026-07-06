# Stage-2 Detector-Aware Offline Selector Route Packet

Route label: `DIVERGENT_INNOVATION_DETECTOR_AWARE_UTILITY_DO_NOT_MERGE_WITH_C3`.

This route answers one 4-week question: can dense AdaTAD teacher utility train an acquisition policy better than p_action-only under matched fixed_384, fixed_768, and dynamic budgets?

Claim boundary:
- This is offline selector pretraining plus strict value-transport ledger validation.
- This is not end-to-end and does not pass teacher utility, GT, value targets, or raw prediction caches into val/test deploy selection or `forward_test`.
- Dense teacher utility is train-only. The intended integration point is around AdaTAD dense-head train/test artifacts such as `AnchorFreeHead.forward_train`, `AnchorFreeHead.forward_test`, and `get_refined_proposals`; exported teacher utility is an artifact consumed by the selector trainer, not by detector inference.
- mAP, runtime, deploy, or paper claims remain locked until the AdaTAD full train/eval variants produce detector mAP.

Decision hooks:
- detector-aware selector metrics: `detector_utility_coverage`, `detector_utility_ndcg`
- strict ledger metrics: selected count, boundary support/bracket, action interior coverage, max hole, p95 hole, uniform similarity
- matched baselines: p_action-only and GAS-VT with the same fixed_384, fixed_768, and dynamic budgets
- full-run metric: later AdaTAD mAP for each detector-aware and baseline variant

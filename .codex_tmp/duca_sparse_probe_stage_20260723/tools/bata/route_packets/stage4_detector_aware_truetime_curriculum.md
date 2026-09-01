# Stage4 Detector-Aware TrueTime Curriculum Route Packet

Route:
`DIVERGENT_INNOVATION_DETECTOR_AWARE_TRUETIME_CURRICULUM_DO_NOT_MERGE_WITH_C3`

Purpose:
connect Stage2 train-only AdaTAD teacher utility, Stage2 learned detector-aware
acquisition policy, and Stage3 TrueTime ST detector-gradient smoke into a
claim-locked curriculum/bilevel evidence gate.

Required phases:

1. `dense_teacher_utility_export`
2. `detector_aware_selector_pretrain`
3. `offline_sparse_detector_warmup`
4. `truetime_st_joint_smoke`
5. `bilevel_fulltrain_candidate`

Required evidence:

- Stage2 teacher export decision is `C3_DETECTOR_TEACHER_UTILITY_EVIDENCE_PASS`
- Stage2 teacher is train split only and has checkpoint/config SHA evidence
- Stage2 policy decision is `C3_DETECTOR_AWARE_POLICY_TRAIN_READY`
- Stage2 policy source is `learned_detector_aware_policy_checkpoint` when present
- Stage2 detector-aware ledger summaries cover `detector_aware_fixed_384`,
  `detector_aware_fixed_768`, and `detector_aware_dynamic`
- Dynamic ledger selected-count distribution is non-collapsed
- Ledger summaries report selected count, boundary support, action coverage,
  max/p95 gap, max/p95 unselected hole, and uniform-similarity diagnostics
- Stage3 proof has non-zero selected-input selector gradient
- Stage3 proof has non-zero detector-loss selector gradient
- Stage3 proof has non-zero ActionFormer forward-train selector gradient
- geometry roundtrip and selected-axis inverse mapping both pass
- TrueTime ActionFormer configs require `physical_grid_actionformer`, so
  selected-axis detector assignment is evaluated on true-time dense positions

Forbidden until full detector mAP evidence exists:

- end-to-end paper claim
- mAP improvement claim
- runtime/FLOPs claim
- deployment claim
- val/test teacher or GT selection
- raw prediction cache or evaluator output as selector input
- uniform fill or uniform scaffold rescue

Validator:

```bash
python -m tools.bata.validate_stage4_detector_aware_truetime_curriculum \
  --stage2-teacher-summary-json ${TEACHER_SUMMARY_JSON} \
  --stage2-teacher-output-jsonl ${TEACHER_UTILITY_JSONL} \
  --stage2-policy-summary-json ${POLICY_SUMMARY_JSON} \
  --stage2-ledger-summary-json ${LEDGER_SUMMARY_FIXED384_JSON} \
                               ${LEDGER_SUMMARY_FIXED768_JSON} \
                               ${LEDGER_SUMMARY_DYNAMIC_JSON} \
  --stage3-proof-json ${TRUETIME_SELECTOR_GRAD_PROOF_JSON} \
  --write-evidence-json ${STAGE4_EVIDENCE_JSON}
```

Pass token:
`C3_STAGE4_CURRICULUM_EVIDENCE_PASS`

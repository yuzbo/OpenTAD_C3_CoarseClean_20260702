# DUCA round-2 minimal change plan

1. `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`: change
   `_semantic_budget_from_predictions` to emit immutable per-video receipt records
   (requested/effective/executed K plus policy factors), and attach them without
   breaking tensor consumers. Add B>1 focused static test.
2. `opentad/models/backbones/actionformer.py` and DUCA selector/backbone boundary:
   make dynamic execution fail closed unless variable-K/bucketed execution is
   actually used; expose an instrumented temporal-work counter for static tests.
3. `opentad/models/detectors/single_stage.py`: map selected irregular coordinates
   to physical seconds before threshold/top-k/NMS exactly once; add ordering test.
4. `configs/adatad/thumos/*duca*`: replace placeholder arm configs with concrete
   official-derived shared-base configs for dense, native uniform, actionness,
   actionness+boundary, dynamic outer-K, and direct-selector ablation, including
   FIT/CAL/HOLD firewall and fixed data/optimizer/evaluator bindings.
5. `scripts/` (N16R4): add literal future launcher/PRE_RUN and recovery contract
   documenting checkpoint state and retention; do not execute it.

Focused verification: py_compile touched Python; selector B>1 receipt test;
backbone work-unit static test; post-processing ordering test; config import/
instantiation smoke. No training, inference, remote, or efficacy claims.

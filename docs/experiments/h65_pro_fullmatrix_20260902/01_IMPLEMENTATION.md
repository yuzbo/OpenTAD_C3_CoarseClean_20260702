# Implementation Notes

- A Phase: `semantic_phase_sampling` in `DucaAcquisitionAdapter` uses ASFormer logits or logit(p_action), masked Gaussian smoothing sigma=2, centered derivative, ReLU onset/offset, sigmoid core, q0.9 per-component scaling, and fixed scaffold/onset/offset/core quotas 128/64/64/128 with deterministic backfill to exact K.
- B CT: `ContinuousTimeScaleAdaptiveConv1d` provides standard, local delta_tau=2, and context delta_tau=2*2^level branches with `y = y_std + eta_l * (y_ct - y_std)` and eta initialized to zero.
- C MoD: `VisionTransformerAdapter` supports Mixture-of-Depths top-K execution at layers [1,3,5,7,9,11], unselected identity bypass, and successful-update capacity schedule 1.0 -> 0.5 over updates 1500-3500.
- D Taylor: detector contribution distillation keeps the existing abs(x*grad) mode and adds signed removal utility `relu(-x*grad)` at the same autograd site, without higher-order graph creation.
- E Curriculum: configs freeze either linear strict60 policy-alpha ramp or 15/20/25 cosine curriculum. MoD capacity is driven by successful optimizer updates.
- Slurm: `tools/experiments/run_h65_pro_train.sbatch`, `tools/experiments/run_h65_pro_eval.sbatch`, and `tools/experiments/submit_h65_pro_fullmatrix.sh` enforce official60, clean exact commit, GPU1, epoch-59 EMA, and out-of-repo submission registries.
- Diagnostic: `tools/bata/h65_pro_hard_one_swap_diagnostic.py` is an offline hard one-swap alignment summary tool and is not called from the training path.

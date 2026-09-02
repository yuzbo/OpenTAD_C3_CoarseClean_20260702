# H65-Pro Full Matrix Base Identity

- Branch: `codex/h65-pro-fullmatrix-strict60-20260902`
- Verified H65 base commit: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- Base route: real H65 selected-axis official60 stack, not CT-DP-BAMoD.
- H65 lineage files retained: `duca_h65_first_singleclock_cycle2.py`, `duca_h65_first_singleclock_cycle3.py`, `duca_h65_first_singleclock_cycle4.py`.
- Fixed dataset: THUMOS14.
- Fixed dense window: T=768.
- Fixed sparse detector budget for H65-Pro and K384 references: K=384.
- Fixed training budget: 60 epochs, 6000 successful optimizer updates.
- Fixed pretrain: public VideoMAE-S Kinetics-400 checkpoint already used by the H65/AdaTAD configs.

CT-DP-BAMoD branches were inspected only as prior implementation references. This branch starts from the verified H65 commit above.

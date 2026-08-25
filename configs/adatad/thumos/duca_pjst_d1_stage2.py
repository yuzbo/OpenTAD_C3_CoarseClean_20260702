"""Frozen-selector PJST-D1 causal mechanism configuration."""

_base_ = ["./duca_h65_60_stage2_transition20_joint20.py"]

pjst_derivative_only = True
single_clock_admission = False
pjst_selector_fixed = True
paper_claim_allowed = False


# 2026-07-09 Final DUCA Complete Model Review Absorption

Source raw review:

- `docs/methods/reviews/2026-07-09-544eca6-final-duca-complete-model-review-raw.txt`

## Final Model Requirements Absorbed

- The paper method must be a final online DUCA frame-selection plugin, not a ledger-only or proof-only experiment.
- The main detector backend must remain the official AdaTAD/ActionFormer path with `ActionFormerHead`; `DucaOnlinePrecheckHead` is allowed only for smoke/precheck tests.
- The coarse probe is supervised by binary actionness labels, but the selector must be transition/boundary/utility-first. Actionness can be an auxiliary signal, not the dominant selection objective.
- The selector must observe the online coarse probe hidden features, not only `p_action` curves.
- Detector loss must backpropagate through the detector-consumed sparse input into the selector and online coarse probe.
- The final selector must include both a differentiable soft max-gap loss and a hard fail-closed max-gap repair mechanism.
- `detector_utility_target` based on GT boundaries must be named honestly as a boundary-utility proxy, with the old name kept only as a deprecated compatibility alias.
- X3D/SlowFast-style dense frozen priors are baselines or upper-bound diagnostics, not the low-cost main pre-backbone method.

## Code Changes Landed In This Pass

- `C3CoarseProbeActionnessSource` now requests and validates online hidden features from the C3/official action segmentation probe.
- `DucaAcquisitionAdapter` now fuses dense observations, transition/actionness features, and coarse hidden features.
- `budgeted_center_radius_decode` now supports hard `max_unselected_hole` repair with fail-closed infeasibility checks.
- `duca_losses` now exposes `boundary_utility_proxy_distribution_loss` and `temporal_max_gap_hole_loss`.
- Official fixed-384 and DUCA-MUST configs enable hidden-feature fusion, boundary proxy supervision, hard max-gap repair, and scheduled max-gap loss.
- Validators enforce that official configs use the final model contract.
- `run_duca_official_adatad_one_step_grad_proof.py` checks an official `ActionFormerHead` one-step backward path and optimizer coverage.
- Focused tests cover max-gap loss/repair, hidden fusion, config contracts, and optimizer coverage.

## Remaining Evidence Boundary

- Local Windows checks can compile and run focused tests, but the official one-step gradient proof must be executed in the remote Linux OpenTAD environment because local Torch has `c10.dll` instability and the alternate local env lacks `mmaction.registry`.
- Full paper claims still require remote full-run mAP evidence for fixed-384 and dynamic DUCA-MUST after this final-contract code is pushed and deployed.

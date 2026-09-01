# Critic Return: PJST-D1 Cycle 2

verdict: BLOCKED_PRE_RUN
role: independent Critic
candidate: 987f48113784295d80e8edc2bd91ff69ec895756
parent: b2ccfccab5b4912b59954afcc9b0364955327f7c
next_owner: Coordinator
next_action: one focused Builder correction/recheck, then rerun this Critic review

## Deterministic blockers

1. `IMPLEMENTATION_CORRECTION` — the matched OFF/ON config switch is not wired. Both configs define `pjst_derivative_only` (`configs/adatad/thumos/duca_pjst_d1_matched_off.py:2`, `...matched_on.py:2`), but no runtime code reads this symbol (repository search finds only those two definitions). `BackboneWrapper.forward` unconditionally computes and passes `pair_scale/pair_valid` whenever irregular positions exist (`opentad/models/backbones/backbone_wrapper.py:98-112,134-163`), so OFF is not raw PatchEmbed and the two causal arms are not distinct. Smallest fix: thread the config flag to the wrapper/backbone and pass PJST metadata only when ON; OFF must pass `None`.

2. `IMPLEMENTATION_CORRECTION` — mixed-batch uniform identity is violated. `VisionTransformerAdapter.forward` enters the transform for the whole batch when any pair scale differs (`opentad/models/backbones/vit_adapter.py:891-904`); `torch.allclose` is global (`:897`), then computes float32 `m/v`, casts back, and `torch.where` only protects invalid pairs, not uniform rows (`:898-903`). A batch containing one irregular row and one exact-uniform row therefore numerically rewrites the uniform row, contrary to the contract's byte identity. Smallest fix: select only rows with at least one valid non-unit scale for arithmetic, leave exact-uniform rows as the original `x` without cast/division, while still using one PatchEmbed call.

3. `IMPLEMENTATION_CORRECTION` — required selected-to-physical remap ordering is absent. The frozen code still filters and top-k selects proposals before remapping (`opentad/models/detectors/single_stage.py:176-196`); `_remap_selector_segments_for_post_processing` is called only after score filtering/top-k. The accepted contract requires remap immediately after raw segment extraction and before filtering/top-k/IoU/NMS. The diff does not modify `single_stage.py` or `two_stage.py`. Smallest fix: apply the existing remap once to raw per-video segments before confidence filtering/top-k, then carry the selected physical segments through NMS; preserve mapping/NMS/thresholds/output.

4. `EVIDENCE_GAP` — no focused tests or executable launcher/validator were added in this candidate. Diff stat contains only two configs and three model files; no `tests/`, `tools/bata/`, or `scripts/` changes. Thus the contract-required shape/layout, mixed-batch uniform byte identity, algebraic reference, padding, K384, no-new-parameter, finite-gradient, production trace, physical-decode-once, and matched OFF/ON identity checks have no candidate-local focused receipts/path. Smallest fix: add the minimal focused tests and existing H65 launcher/validator wiring required by the contract, without adding a new framework or changing claims.

## Checks performed

Read-only static review of the full accepted Pro contract and frozen diff; searched all PJST symbols and remap call sites. No data, GPU, Slurm, training, evaluation, browser, or metric execution.

The candidate cannot enter Evaluator structural PRE_RUN until the four blockers are corrected and independently rechecked.

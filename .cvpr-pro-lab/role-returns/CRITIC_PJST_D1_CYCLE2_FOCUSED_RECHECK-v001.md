# Critic Return: PJST-D1 Cycle 2 Focused Recheck

verdict: BLOCKED_PRE_RUN
role: independent Critic
candidate: c8faf96be69cc8302ea0f5d1e38dc089ce70c429
parent: 987f48113784295d80e8edc2bd91ff69ec895756
next_owner: Coordinator
next_action: route the remaining deterministic corrections to Builder, then obtain one fresh focused Critic recheck

## Findings

1. `IMPLEMENTATION_CORRECTION` — OFF still constructs PJST metadata. `BackboneWrapper.forward` unconditionally calls `global_rank_clip_coordinates` when `irregular_selected_positions` is present (`opentad/models/backbones/backbone_wrapper.py:97-118`), including OFF. That constructs `pair_scale/pair_valid`; checkpointing also receives them (`:125-133`, `:147-155`) even though adapter kwargs are omitted. The explicit contract requires OFF never constructs/passes/acts on PJST metadata, including checkpointed paths.

2. `IMPLEMENTATION_CORRECTION` — selected-to-physical remap remains after score filtering/top-k. `single_stage.py:176-196` filters/sorts/selects proposals and only then calls `_remap_selector_segments_for_post_processing` at line 196. The required ordering is exactly once immediately after raw segment extraction and before filtering/top-k/NMS; candidate does not modify this path.

3. `EVIDENCE_GAP` — focused tests are insufficient for the mandated coverage. `tests/test_pjst_d1_cycle2_corrections.py:5-19` has only signature and packed-metadata assertions; it does not exercise transform arithmetic/gradient, B>1 identity, checkpoint slicing/rejection, OFF/ON state-key/config identity, or remap order. The pytest run is additionally blocked by local Torch DLL initialization (`WinError 1114`, c10.dll).

4. `EVIDENCE_GAP` — launcher is a precheck/contract echo, not an executable matched-arm launcher/validator. `scripts/duca_pjst_d1_h65_30plus60_precheck.sh:18-28` only asserts environment strings for `PRECHECK_ONLY=1`; non-precheck path prints that training/evaluation requires another launcher and exits 2. It cannot express or launch the shared 30-epoch Stage-1 checkpoint plus matched 60-epoch/6000-update OFF/ON runs.

## Closed items

- Config switch reaches `BackboneWrapper.custom.pjst_derivative_only` (`configs/adatad/thumos/duca_pjst_d1_matched_off.py:2-3`, `...matched_on.py:2-3`) and non-checkpoint adapter calls conditionally pass kwargs (`backbone_wrapper.py:135-165`).
- Mixed-batch transform selects nonuniform rows and preserves untouched rows in the direct adapter path (`vit_adapter.py:898-918`), but required focused runtime evidence is absent.

## Checks

- `python -m py_compile ...`: PASS.
- `PRECHECK_ONLY=1 DUCA_STAGE1_CHECKPOINT=/tmp/stage1.pth DUCA_STAGE1_CHECKPOINT_SHA256=test bash scripts/duca_pjst_d1_h65_30plus60_precheck.sh`: PASS (`PJST_D1_H65_30PLUS60_PRECHECK_PASS`).
- `python -m pytest tests/test_pjst_d1_cycle2_corrections.py -q`: BLOCKED during Torch import by `WinError 1114` loading `c10.dll`; no data/GPU/Slurm/training/evaluation/metrics used.

No browser, data, GPU, Slurm, training, evaluation, metrics, or claims.

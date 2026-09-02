# ZoomToken Continuous-RoI S2-v3 full-data compute-Pareto Builder plan

## Authoritative assignment

- Pro request: `PRO_CONTINUOUS_ROI_S2_V3_FULL_DATA_SINGLE_AXIS_OBJECTIVE_ADJUDICATION-v002`
- Exact Project conversation: `6a9641aa-a8f8-83ea-a5e7-b64da4daffc2`
- Decision: `REVISE`; role contract: `KEEP`
- Task: `ZOOMTOKEN-CONTINUOUS-ROI-S2-V3-FULL200-COMPUTE-PARETO-3X3-v001`
- Base: `10aed28659a08fa703def278fc0f5f1422dcad89`
- Branch: `codex/zoomtoken-continuous-roi-s2-v3-full200-compute-v001`
- Primary plane: detection performance versus executed inference computation.
- Latency, throughput, peak memory and energy are disclosure-only diagnostics and have no admission threshold.

The earlier `fit160/gate40`, 129-window, 4,800-update and joint latency/resource-gate plan is superseded. It produced no candidate commit, formal job or scientific result and must not be revived.

## Frozen scientific comparison

The matrix is `D160`, `G96`, and `U128-A0` at paired seeds `4407`, `4408`, and `4409`.

- `D160`: complete source frame letterboxed to `160×160`.
- `G96`: complete source frame letterboxed to `96×96`.
- `U128-A0`: shared-weight `Global96` plus a source-native fixed-center `128×128` crop. For canonical `180×320`, the crop is `[96,26,224,154]`.
- U128-A0 has no learned selector, annotation/metric access, trajectory fitting, candidate search or second backbone copy.
- Builder inspection of base `10aed286` found that the historical Continuous-RoI learned fusion adds 609,449 parameters and is therefore ineligible. The frozen A0 arm instead reuses `NativeCropBackboneWrapper` with its parameter-free `fixed_mean` fusion and one shared VideoMAE object. Static config-surface validation confirms that D160, G96 and U128-A0 retain the same parameter-bearing model and optimizer configuration; N16R4 must still confirm the runtime named-parameter identity/count before PRE_RUN.

## Complete-data and schedule contract

- Training: every annotation identity in `subset_name="training"`, all 200 videos exactly once per epoch, no drop or duplicate.
- Evaluation: all 211 validation videos and the complete ordered 792-window loader with `window_size=768` and `overlap_ratio=0.5`.
- The repository's val/test loaders map to the same physical validation population, so there is one campaign-level metric-bearing opening, not a fabricated independent third split.
- Training uses a training-only annotation. Held-out inference uses a label-free 211/792 manifest. The final evaluator is the only GT consumer and starts only after 9/9 final checkpoints and 9/9 predictions are sealed.
- Each cell uses 2 GPUs, global/local batch `2/1`, 60 epochs, 100 successful updates per epoch, and 6,000 successful updates total.
- The only primary checkpoint is `epoch_59 state_dict_ema` after successful update 6,000. There is no best checkpoint, best seed, early stopping or metric-bearing intermediate validation.

Before candidate commit, Builder generates immutable SHA256 manifests for the canonical annotation, class map, and all 411 realpath media files. Formal start fails closed on any later mismatch.

## Computation contract

`C_exec` begins at the first arm-dependent decoded-RGB crop/resize/normalization operation and ends at pre-NMS raw detections. It includes view construction, patch embedding, every VideoMAE block, Adapter, fusion, detector and candidate-specific control work at actual runtime tensor shapes over the complete 792-window population. FMA counts as 2 FLOPs. Unsupported operators require a frozen explicit cost rule or conservative upper bound; incomplete coverage is a protocol blocker.

The sole resource gate is aggregate `C_exec(U128-A0) / C_exec(D160) <= 0.90` across all three seeds and all 792 windows. Token count and attention-only FLOPs are not substitutes.

## Minimal implementation surface

Only these task surfaces are allowed:

1. nine configs `configs/adatad/thumos/continuous_roi_s2_v3_{d160,g96,u128_a0}_seed{4407,4408,4409}.py`;
2. `docs/methods/continuous_roi_s2_v3_full200_compute_protocol.json`;
3. task-local manifest, training-receipt, label-free inference, one-shot evaluator and compute-profiler tools under `tools/bata/`;
4. `scripts/run_zoomtoken_continuous_roi_s2_v3_full200_compute_n16r4.sh`;
5. focused tests;
6. `opentad/datasets/transforms/native_crop.py` or `opentad/models/backbones/vit_adapter.py` only if the existing A0/U128 path cannot satisfy the frozen semantics.

Do not merge the uncommitted WIP wholesale. Reapply only necessary files to a clean descendant of the exact base. The official evaluator, Soft-NMS, detector loss, class map, annotations and pretrained checkpoint remain unchanged.

## Builder verification

Focused tests must establish:

- 200 identities/epoch, 100 successful updates/epoch and 6,000/cell;
- the exact 211-video/792-window ordered population;
- identical temporal windows, evaluator, postprocess and NMS across arms;
- absence of `fit160`, `gate40`, 129-window and sampled-loader formal routes;
- fixed A0 crop, label-free construction, fixed global/local order and actual shared backbone parameters;
- D160/G96 closed-path parity with the base;
- complete recovery state and final-EMA-only selection;
- full operator coverage and actual-shape logging for `C_exec`;
- a physical one-shot barrier that refuses to evaluate before 9/9 predictions are sealed;
- physical separation of diagnostic latency/memory/energy fields from admission state;
- no live, partial or intermediate performance output.

## Review, resources and terminal states

- Critic reviews only exact SHA, mechanism fidelity, parameter fairness, complete data/schedule, leakage barriers, final checkpoint, compute coverage, evaluator identity, preregistered statistics and absence of hidden system gates or wholesale WIP.
- Result-blind Evaluator checks the clean SHA, nine configs, data manifests, pretrained weight, job graph/resources, one-shot barrier, compute ledger, empty formal namespace and scheduler test-only admission.
- After both pass, submit one 9-leaf training graph, with at most 3 cells/6 GPUs concurrently, followed by one terminal prediction/evaluation/diagnostic graph on 1 GPU.
- U128-A0 passes only if all preregistered performance/stability gates and the computation gate pass and G96 does not strictly dominate it.
- If G96 strictly dominates an otherwise passing U128-A0, return `REVISE_TO_G96_CONTROL_ONLY`.
- Any U128-A0 hard gate failure returns `STOP_S2_V3_A0_EXACT_ROUTE`.
- Any incomplete population/schedule, leakage, checkpoint selection, second GT opening or incomplete compute ledger returns `NO_SCIENTIFIC_DECISION_OBJECTIVE_BLOCKER` without partial interpretation.
- Any complete terminal or objective blocker immediately freezes new experiments and triggers one fresh exact-Project Pro review.

## Beijing deadlines

- Builder plan and role-rule sync: `2026-09-01 14:00`
- Clean candidate: `2026-09-02 20:00`
- Critic: `2026-09-03 08:00`
- Evaluator: `2026-09-03 14:00`
- Formal start after admission: `2026-09-03 18:00`
- Terminal evidence or objective queue blocker: `2026-09-08 23:00`
- Scientific return: `2026-09-09 02:00`

The role contract is `KEEP`; only this task-level single-axis rule is synchronized.

# GPT/Pro Review Prompt: DUCA Online Plugin with Official AdaTAD Backend

请你进行一次严厉、逐行、以 CVPR 论文主方法为标准的代码与实验设计审查。不要把 proof-of-concept、toy precheck、offline ledger pipeline 或历史 Stage2/lattice 路线当成最终方法。请直接判断当前实现是否真的朝最终论文主方法闭环前进，并指出所有必须修复的问题。

## 0. Visibility / Fixed Revision

Repository:

- GitHub: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- Branch: `codex/gas-vt-stage23-detector-aware-20260706`
- Review base commit: `bcb9882`
- Review head commit: `84e95d63694a88657048f08b7c94a98821edb671`
- Head short: `84e95d6 Add DUCA official AdaTAD backend main config`

Please review:

```bash
git diff --stat bcb9882..84e95d63694a88657048f08b7c94a98821edb671
git diff bcb9882..84e95d63694a88657048f08b7c94a98821edb671
```

The latest commit adds a main DUCA experiment intended to use an as-original-as-possible AdaTAD backend:

- `configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py`
- `tools/bata/validate_duca_official_adatad_backend.py`
- `scripts/run_duca_online_official_adatad_backend_gpu1.sh`

It also includes supporting DUCA online selector/precheck changes:

- `configs/adatad/thumos/duca_online_actionformer_no_physical_grid_precheck.py`
- `configs/adatad/thumos/duca_online_actionformer_physical_grid_precheck.py`
- `opentad/models/selectors/duca_online_frame_selector.py`
- `opentad/models/duca/acquisition.py`
- `opentad/models/detectors/actionformer.py`
- `opentad/models/detectors/base.py`
- `opentad/models/detectors/single_stage.py`
- `tools/bata/validate_duca_online_adatad_precheck.py`
- `tests/test_duca_online_precheck_config.py`
- `tests/test_duca_online_frame_selector_contracts.py`

## 1. Intended Final Paper Direction

The final paper should not claim to introduce a new TAD detector. The intended contribution is an online temporal acquisition plugin / adapter placed before an existing detector:

```text
video / temporal observations
  -> low-cost or frozen actionness source
  -> DUCA online acquisition adapter
  -> hard selected_positions in original dense-time, <=384
  -> sparse gather + SparseTemporalGrid / remap metadata
  -> original AdaTAD / ActionFormer-style detector forward
  -> TAD predictions
```

The final method is not:

- C3 as the main contribution;
- p_action top-k;
- GAS-VT;
- lattice / frame-moving heuristics;
- offline ledger training;
- a modified detector head disguised as a plugin.

Those can only be baselines, ablations, diagnostics, initialization sources, or auxiliary inputs.

## 2. What This Round Implemented

This round tries to close a key gap raised by prior reviews: if the paper claims a pre-backbone plugin, the downstream detector should be as close as possible to the official AdaTAD baseline.

The main config `duca_online_official_adatad_backend_full_train.py` inherits:

- `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`

and adds:

- `model.frame_selector = dict(type="DucaOnlineFrameSelector", ...)`;
- DUCA budget `384`;
- dense candidate window `768`;
- detector-consumed selected length `384`;
- VideoMAE selected input adjustment:
  - `model.backbone.backbone.total_frames = 384`;
  - `model.backbone.backbone.tubelet_size = 16`;
  - `model.projection.max_seq_len = 384`;
  - `model.projection.chunk_num = 24`;
- no physical-grid ActionFormer mode in the main config;
- no offline ledger decision path;
- zero-shot / motion-style actionness source with fail-closed provenance metadata.

The validator `validate_duca_official_adatad_backend.py` checks the static contract:

- `official_adatad_backend=True`;
- `model.type == "ActionFormer"`;
- `frame_selector.type == "DucaOnlineFrameSelector"`;
- `rpn_head` config matches the official base config;
- `physical_grid_actionformer` is disabled;
- no `bata_value_transport` / no ledger path;
- budget <= 384 and dense window 768;
- selected positions are original-time indices;
- selected-axis prediction remap metadata is expected;
- dataset still uses online `random_trunc` / `sliding_window`, not ledger sampling;
- actionness provenance is fail-closed.

Remote verification already done:

- Remote clean snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_stage23_84e95d6_20260708_191738`
- Remote validator: PASS
- Remote launcher `PRECHECK_ONLY=1`: PASS
- Remote focused test: `tests/test_duca_online_precheck_config.py` -> `8 passed`

Submitted jobs:

- Official clean AdaTAD baseline from official OpenTAD clone:
  - Slurm job: `1150761`
  - Official repo: `https://github.com/sming256/OpenTAD.git`
  - Official HEAD: `1aa8ca4ac5e846b1e8ff69298dd6607121a01589`
  - Config: `e2e_thumos_videomae_s_768x1_160_adapter.py`
- DUCA online official-backend main candidate:
  - Slurm job: `1150764`
  - Snapshot commit: `84e95d63694a88657048f08b7c94a98821edb671`
  - Backend contract: `official_AdaTAD_ActionFormerHead_no_physical_grid_no_ledger`
  - Budget: `384`
  - Dense window: `768`

## 3. Review Questions: Be Severe

Please answer these directly and with file:line references.

### A. Does this truly preserve the official AdaTAD backend?

The main config inherits the official AdaTAD config and keeps the official `rpn_head` config. However, the repository still contains modifications in:

- `opentad/models/detectors/actionformer.py`
- `opentad/models/detectors/single_stage.py`
- `opentad/models/detectors/base.py`

Questions:

1. Does this invalidate the claim that DUCA is paired with an "original AdaTAD backend"?
2. Is checking that `rpn_head` config matches the official base config enough?
3. Should the final implementation instead wrap an unmodified official detector object, avoiding detector source edits entirely?
4. Are the current hook changes acceptable if they are pure `frame_selector` plumbing and disabled unless configured?

### B. Is adapting VideoMAE input length to 384 legitimate?

The plugin selects <=384 observations before the detector. The config changes `total_frames`, `chunk_num`, and `max_seq_len` to match the selected length.

Questions:

1. Is this a valid pre-backbone sparse-input adaptation, or does it modify the detector baseline too much?
2. Should we instead keep official `768` internals and pass masks/sparse indices differently?
3. How should the paper describe this so reviewers do not see it as a detector reimplementation?

### C. Is the online plugin contract actually satisfied?

Please check whether the code ensures:

- online `adapter.acquire()` / `frame_selector` path at train and test;
- hard selected positions, not soft dense forward;
- selected positions are original dense-time indices;
- detector actually consumes only selected observations;
- `detector_input_length == selected_count <= 384`;
- ledger/audit is not involved in decisions;
- val/test/inference do not use GT, teacher utility, cached predictions, raw predictions, oracle boundary, or ledger decisions.

Find any violations or ambiguities.

### D. Is selected-axis to original-time localization closed?

Sparse detection creates a coordinate risk: the detector may predict proposal times on selected-axis positions, while mAP requires original-time coordinates.

Please inspect:

- whether selected positions and inverse/remap metadata are attached to `metas`;
- whether GT used for training is remapped correctly if the detector trains on selected-axis;
- whether predictions/NMS/eval are remapped to original-time before scoring;
- whether high-IoU mAP can be trusted.

If any part is missing, mark it Critical.

### E. Does detector loss truly train the selector?

Prior reviews required that detector loss, not just auxiliary DUCA losses, influence selector parameters through hard-forward ST/surrogate paths.

Please inspect:

- whether real AdaTAD/ActionFormerHead losses backpropagate into selector parameters;
- whether tests prove this using a real registry-built detector, not only fake toy heads;
- whether `selected_mask_st` aligns with actual consumed selected positions;
- whether gradients are meaningful enough for the paper claim.

### F. Is zero-shot/motion actionness a real contribution or a weak heuristic?

The current main config uses a low-cost actionness source. Please judge:

1. Is motion/feature-energy actionness sufficient for a DUCA-Frozen branch?
2. Should the main paper use X-CLIP / ActionCLIP / SlowFast / VideoMAE / InternVideo actionness instead?
3. Which source is fair as a low-cost selector and which is only an upper-bound teacher?
4. Does provenance fail closed for manual/precomputed p_action?
5. What cost accounting must be added so reviewers do not say we hid a huge foundation model in the sampler?

### G. Does this have CVPR-level potential?

Please do not be polite. Decide:

1. Is the current method a paper-worthy online temporal acquisition method or still an engineering pipeline?
2. Is the novelty enough if it is "actionness + budgeted hard selection + sparse AdaTAD"?
3. What model-design element is still missing to make it elegant and non-heuristic?
4. What minimum mAP/high-IoU/latency evidence would make the claim credible?

## 4. Required Reviewer Attacks and Defenses

For each attack, state whether it currently succeeds, partially succeeds, or fails. Then say exactly what code/experiment/figure would defend against it.

- This is not original AdaTAD; you changed the detector.
- This is just actionness top-k.
- The plugin is actually an offline ledger pipeline.
- The physical-grid ActionFormer modification is the real trick.
- Zero-shot/motion actionness is a heuristic, not learned acquisition.
- Training and inference mismatch.
- Detector still sees dense inputs somewhere.
- Selected positions are metadata only, not actually consumed.
- Selected-axis time breaks localization.
- High-IoU mAP does not improve.
- Only AdaTAD works; ActionFormer generality is unproven.
- Dynamic K / fixed K comparisons are unfair.
- Teacher utility leaks into validation or inference.
- C3/p_action is THUMOS-trained and cannot be called frozen/no-label.
- Cost accounting ignores selector/backbone/gather/NMS/zero-shot source.
- Raw-frame compute is not actually reduced.
- The method contract is unclear.

## 5. What to Inspect Line by Line

Please review these files line by line:

- `configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py`
- `tools/bata/validate_duca_official_adatad_backend.py`
- `scripts/run_duca_online_official_adatad_backend_gpu1.sh`
- `configs/adatad/thumos/duca_online_actionformer_no_physical_grid_precheck.py`
- `configs/adatad/thumos/duca_online_actionformer_physical_grid_precheck.py`
- `opentad/models/selectors/duca_online_frame_selector.py`
- `opentad/models/duca/acquisition.py`
- `opentad/models/detectors/actionformer.py`
- `opentad/models/detectors/single_stage.py`
- `opentad/models/detectors/base.py`
- `tools/bata/validate_duca_online_adatad_precheck.py`
- `tests/test_duca_online_precheck_config.py`
- `tests/test_duca_online_frame_selector_contracts.py`

Also compare against the official clean OpenTAD AdaTAD config:

- official repo: `https://github.com/sming256/OpenTAD.git`
- official config: `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`

## 6. Output Format

Please return:

### Verdict

- `PASS`, `HOLD`, or `FAIL`.
- One paragraph explaining whether current code is deployable for the main mAP experiment.

### Critical Issues

Each issue must include:

- file:line reference;
- exact failure mode;
- why it threatens the final paper claim;
- concrete fix or patch sketch.

### Important Issues

Same format, but for non-blocking gaps.

### Minor Issues

Style, naming, documentation, test polish.

### CVPR-Level Assessment

Answer:

- Is the current direction CVPR-level if results are strong?
- If not, what exact model idea would make it stronger?
- What claims should be avoided until evidence exists?

### Smarter Selector Proposal

Give a concrete, implementable alternative to the current actionness/radius/gap design. It must preserve:

- online plugin interface;
- hard budget <=384;
- original-time selected positions;
- no GT/teacher/cache in validation/inference;
- compatibility with AdaTAD and ActionFormer;
- detector loss can influence selector.

Please include PyTorch-style pseudocode or patch-level design, not just conceptual prose.

### Required Experiments / Figures

List the minimum experiment matrix and paper figures needed to make the paper logically self-contained:

- dense official AdaTAD;
- dense ActionFormer;
- uniform/random 384;
- actionness top-k;
- zero-shot/frozen actionness;
- C3/PAction/GAS-VT/lattice baselines as failure/diagnostic baselines;
- DUCA-Frozen;
- DUCA-Adapted;
- DUCA with and without hard-forward ST joint fine-tuning;
- physical-grid diagnostic only;
- high-IoU mAP and remap correctness;
- latency/cost accounting;
- geometry/coverage/collapse diagnostics.

### Final Decision

State whether the next step should be:

1. run full mAP now;
2. fix critical code gaps first;
3. redesign selector before spending GPU;
4. split engineering baseline from final paper method.

Be strict. Do not treat a passing precheck as proof of paper readiness.

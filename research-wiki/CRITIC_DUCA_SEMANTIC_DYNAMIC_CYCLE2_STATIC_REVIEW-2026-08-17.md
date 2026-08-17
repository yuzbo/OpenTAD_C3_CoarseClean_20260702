# DUCA semantic dynamic cycle2 static critic receipt — 2026-08-17

Scope: frozen candidate `5863b2f1fa1812c6c39ed275e1639c3dd78d4468`, parent `6125654b946cc30c614428ce1141f1903b015867`. Read-only static review.

## Verdict

**BLOCKED** (`DYNAMIC_ROUTE_STATIC_PASS` is not supportable).

## Contract findings

1. **Dynamic outer-K / heavy path: BLOCKED — IMPLEMENTATION_CORRECTION.** The selector computes and records `requested_k`, `effective_k`, and `executed_k` (`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:1298-1307`, `:1600-1613`), and emits a mask based on `selected_output_valid_lengths` (`:1365-1366`). However `_apply_sparse_transport` always allocates and returns `[B, self.target_len, ...]` (`:3369-3379`); the detector therefore receives padded fixed `target_len` tensors, with only a mask/metadata indication of shorter K. No ActionFormer/VideoMAE bucket/variable-length heavy invocation or measured executed-K work is wired here. Minimal fix/test: pass per-sample packed/bucketed selected tensors into the actual backbone/detector and assert heavy input token/frame count equals executed K (not target_len), including mixed-K batch.

2. **Six arms / shared stack: PARTIAL, not executable evidence — IMPLEMENTATION_CORRECTION.** Config declares six names (`configs/adatad/thumos/duca_semantic_indirect_six_arm_n16r4.py:4-10`) and shared detector/loss/NMS/evaluator/seed fields (`:3-12`), while validator only imports/compares names and recovery strings (`tools/bata/validate_duca_semantic_indirect_n16r4.py:11-15`). No launcher or train.py arm dispatch is present in this candidate evidence; the config is explicitly “no execution” (`duca_semantic_indirect_six_arm_n16r4.py:1`). Minimal fix/test: instantiate every arm through actual `tools/train.py` parser/model builder and assert isolated selector policy plus identical detector/loss/NMS/evaluator/update/seed signatures.

3. **Physical timestamps: PARTIAL / SCIENTIFIC_AMBIGUITY.** Selector preserves original dense positions in metadata (`...frame_selector.py:3435-3445`) and post-processing contains selected-axis→dense interpolation before final output (`opentad/models/utils/post_processing/utils.py:74-103`). Training GT is remapped to selected axis by default (`...frame_selector.py:3988-4009`, `:4047-4059`). Static code does not demonstrate that threshold/top-k/IoU/NMS all happen after inverse mapping in the actual ActionFormer path; the selector itself still advertises “unchanged ... detector” (`:979-983`). Minimal test: synthetic nonuniform positions with known segment, verify threshold, IoU and NMS inputs/outputs are dense physical coordinates and no selected-axis metric occurs.

4. **FIT/CAL/HOLD: BLOCKED — IMPLEMENTATION_CORRECTION.** Search found no FIT/CAL/HOLD manifest, disjointness validator, or actual data-entry binding in candidate files. The plan merely asserts “explicit disjoint manifests” (`research-wiki/DUCA_SEMANTIC_DYNAMIC_CYCLE2_MINIMAL_CHANGE_PLAN-2026-08-17.md:18-21`). Selector metadata flags forbid deploy-time GT/teacher/cache payloads (`...frame_selector.py:3127-3131`, `:3580-3587`), which is useful but does not establish split binding or held-out calibration. Minimal fix/test: add manifests and validator proving pairwise disjoint sample IDs and bind each arm's FIT/CAL/HOLD loaders; reject GT/teacher/raw-cache at deploy and test with an integration fixture.

5. **Checkpoint/recovery: BLOCKED — IMPLEMENTATION_CORRECTION.** `tools/train.py` resumes model, optimizer, scheduler and EMA only (`tools/train.py:233-245`), while `save_checkpoint` stores epoch/model/optimizer/scheduler/EMA (`opentad/utils/checkpoint.py:22-44`) and writes `epoch_<epoch>.pth` (`:49`). It stores neither scaler nor Python/NumPy/Torch/CUDA RNG nor DataLoader state; no latest3/milestone/final/final-EMA retention logic exists in helper (`:46-105`). The config's recovery tuple is declarative only (`duca_semantic_indirect_six_arm_n16r4.py:12`), and validator checks strings, not runtime behavior. Minimal fix/test: checkpoint and restore all named states, implement retention/final artifacts, then simulate resume and compare next batch/update/RNG plus validator invoking the actual parser.

## Verification

`python -m py_compile` on selector, post-processing, `tools/train.py`, and DUCA validator passed. `PRECHECK_ONLY=1 python tools/bata/validate_duca_semantic_indirect_n16r4.py` returned `{'status': 'precheck_only', 'data_access': False, 'execution': False}`. This is only a declarative precheck, not six-arm execution or scientific validation. No torch runtime test was claimed.

## Fairness / leakage

Static selector flags explicitly set `uses_gt=False`, `uses_teacher=False`, `uses_raw_prediction_cache=False` in deploy metadata (`...frame_selector.py:3127-3131`, `:3580-3587`), but absence of FIT/CAL/HOLD bindings prevents a positive split-fairness conclusion. No evidence found that validation/test GT or teacher enters dynamic budget; no evidence found that the required held-out calibration protocol is implemented.

## Next owner

DUCA implementation owner: wire true variable/bucketed heavy execution, executable six-arm dispatch, physical-time postprocess integration tests, FIT/CAL/HOLD manifests/loaders, and complete checkpoint state/retention. Do not change the scientific route or promote the dynamic arm until these gates pass.

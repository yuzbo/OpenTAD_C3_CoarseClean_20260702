# Evaluator — PJST-D1 c73e8418 terminal prediction and paired-bootstrap finalizer

This is a bounded result-finalization continuation under the user's current explicit authority. Reuse the already registered evaluation-only identity `codex-evaluator-pjst-d1-cycle4-slurm-submit-v005` and its clean workspace. Do not register or create another Evaluator.

## Frozen experiment identity

- revision: `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`
- remote checkout: `/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_cycle4_c73e8418_20260826`
- OFF job/checkpoint: `1256372`, `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/off/gpu1_id0/checkpoint/epoch_59.pth`
- ON job/checkpoint: `1256373`, `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/on/gpu1_id0/checkpoint/epoch_59.pth`
- checkpoint state: `state_dict_ema` only for the primary comparison
- seed: `3407`; fixed `K=384`; complete canonical THUMOS14 validation population; unchanged official evaluator/NMS
- annotation: `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json`
- class map: `/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt`

The two training jobs are terminal `COMPLETED 0:0`. Their in-memory official point estimates are:

- OFF Avg-mAP `0.6506328323`; mAP@0.3/0.4/0.5/0.6/0.7 `0.8004698811/0.7556871469/0.6802175108/0.5803293531/0.4364602698`
- ON Avg-mAP `0.6459080197`; mAP@0.3/0.4/0.5/0.6/0.7 `0.7925176714/0.7431627020/0.6787476664/0.5774244005/0.4376876583`

These scalars are not a terminal causal verdict. The run roots contain no saved per-video prediction, finalizer, bootstrap CI, cost ledger, or gate artifact.

## Authorized action

1. Reopen both exact epoch-59 checkpoints and perform one frozen, evaluation-only inference per arm. Save the complete per-video prediction artifacts. Do not train, resume, update parameters, change checkpoint selection, inspect another epoch, or change data/model/NMS/evaluator semantics. An execution-only `save_dict`/output-path override is allowed only if it changes serialization and nothing scientific.
2. Recompute the official point metrics from the saved predictions and require exact agreement with the terminal log values at the frozen numerical tolerance. Require 211 evaluated videos and the canonical annotation/class-map/evaluator identity.
3. Execute exactly 10,000 paired whole-video cluster-bootstrap draws. Every draw must resample the same video clusters for OFF and ON and rerun the pooled official AP evaluator. Use PCG64 with a fixed recorded nonce/namespace. The two-sided 95% interval uses nearest-rank, no interpolation: sorted one-based ranks 250 and 9750.
4. Report `ON-OFF` for Avg-mAP, mAP@0.6 and mAP@0.7, including point estimate, LCB95 and UCB95. Run the already frozen short/adjacent and physical-gap strata only when their input definitions and artifacts are already available; never invent or tune a bin after seeing outcomes.
5. Reopen the existing identity seals: same selected K384 positions/RGB/masks and exposure order, derivative-only ON as the sole mechanism difference, pre-filter/pre-top-k/pre-IoU/pre-NMS physical decode, same final-EMA checkpoint policy, evaluator and data. If a seal cannot be established, report it as an objective blocker rather than weakening the contract.
6. Record Slurm accounting for jobs 1256372/1256373 and the evaluation/finalizer jobs. Do not claim full-stack efficiency unless a matched frozen full-stack cost artifact actually exists.
7. Apply the accepted gate exactly:
   - `MECHANISM_PASS` only if every identity seal passes, `LCB95(delta Avg-mAP)>0`, point `delta mAP@0.6>=0`, point `delta mAP@0.7>=0`, no frozen short/adjacent stratum has `UCB95<0`, and the high-gap point improvement exceeds the low-gap point improvement.
   - `MECHANISM_KILL` if `UCB95(delta Avg-mAP)<=0`, or `UCB95(delta mAP@0.7)<0`, or a mandatory identity/coordinate/mechanism seal fails.
   - otherwise `INCONCLUSIVE_STOP_EXPANSION`: no claim, tuning, second seed, support-weighted variant, dynamic-K expansion, or end-to-end selector expansion.

Use the existing evaluator engine where applicable: `tools/bata/bootstrap_duca_h65_official_map.py` and `tools/bata/merge_duca_h65_bootstrap_shards.py`. Do not edit model code. If the existing engine cannot consume the newly serialized PJST OFF/ON predictions through its CLI without a production-code change, stop with `NEEDS_ATTENTION` and name the exact missing wiring.

Return one durable receipt with exact commands, Slurm job IDs/dependencies, artifact paths, hashes/identities, recomputed point metrics, bootstrap intervals, gate verdict, evidence class, and the following handoff fields:

- `current_scientific_question`: whether derivative-only PJST-D1 improves H65 first-mixing representation under frozen matched selection
- `next_owner`: DUCA Coordinator
- `next_action`: ingest the frozen finalizer and route a material terminal result to scientific adjudication only if the gate artifact is complete
- `dependency`: saved per-video OFF/ON predictions plus completed 10,000-draw bootstrap and identity seals
- `expected_return_at`: `2026-08-27T18:00:00+08:00`
- `single_recovery`: one same-task infrastructure recovery before an unknown output state; never duplicate a completed inference or bootstrap


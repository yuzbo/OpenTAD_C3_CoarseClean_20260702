---
id: exp:duca-two-stage-curriculum-official60
type: experiment
status: experiment_running
updated: 2026-07-21
---

# DUCA two-stage curriculum official-60

## Question

Does a training-only coarse/transition frontend pretraining stage, followed by
uniform AdaTAD warmup and delayed protected joint training, outperform both
same-commit exact-uniform K=384 and scratch joint training on terminal mAP?

## Exact implementation

- Branch: `codex/duca-two-stage-curriculum-20260721`
- GitHub:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-two-stage-curriculum-20260721`
- Exact commit: `6f2ed48d8cb31fe984b8a20223a0624fcf95d4b9`
- Remote snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_twostage_7431543_20260721`
- Clean remote tests: `83 passed in 50.12s`

## P0 and selection gate

- P0 variants: action/transition/boundary weights `1/0.05/8`, `1/0.10/16`,
  and `1/0.20/32`.
- Duration: 20 frontend-only epochs; checkpoint every five epochs.
- Split: deterministic THUMOS training-only 80/20 split, seed 3407.
- Split manifest SHA-256:
  `c4bb87e9c2f824079faa8b4f500a4053939c5fde834a6131a2da4e07446feee9`.
- Fail-closed gates include action AUROC, transition AUROC against pure delta,
  boundary recall versus uniform, endpoint distance, short-action both-endpoint
  recall, exact K and max-hole.
- Validation/test TAD mAP does not select the P0 checkpoint.

## Formal four arms

1. `two_stage_exact_uniform`
2. `two_stage_scratch`
3. `two_stage_pretrained_joint`
4. `two_stage_pretrained_frozen`

Every arm uses official-60, 6000 successful updates, terminal epoch-59 EMA,
exact K=384 and the same official AdaTAD/ActionFormer backend. The first 1000
updates are exact-uniform detector warmup and do not train the frontend.

## Deployment history

- The first serial Job `1178480` failed before any optimizer update because
  the frontend-only config inherited an AdaTAD `backbone` optimizer subconfig
  after the detector backbone had been frozen. PyTorch AdamW rejected that
  leaked keyword. This is deployment-contract evidence, not a model result.
- Commit `6f2ed48` deletes the inherited backbone group for P0, adds config
  regression assertions and makes the submit precheck fail closed on future
  optimizer leakage.
- Replacement Job `1178487` entered epoch 0, built the
  optimizer and started training without Traceback/OOM/non-finite loss. At
  step 20 it reported finite total loss `0.8680`, raw action/transition/
  boundary losses `0.7413/6.3456/0.0073`, exact effective K=384, zero detector
  loss/bridge weight and `duca_detector_path=skipped`. The generic phase label
  still says `joint_transition_detection`; the explicit detector-path and loss
  audit fields are authoritative for P0. No P0 completion artifact or mAP
  exists at registration time.
- At 12:31 +08:00 the shared `/data` filesystem reported 100% use. Accounting
  later sealed Job `1178487` as `FAILED/1:0` at 12:33:31 after 24m02s. Its P0
  log stopped during epoch 1 and no checkpoint was written. This is an
  infrastructure failure with finite early frontend losses, not P0 convergence
  evidence and not a model-quality result.
- After checkpoint cleanup restored about 310 GiB free space, the full
  dependency-graph precheck passed at
  `duca_twostage_6f2ed48_precheck_20260721_133354`. Atomic parallel submission
  was rejected by Slurm with `AssocMaxSubmitJobLimit`; its transactional
  cleanup left no admissible partially submitted experiment.
- The repository's protocol-equivalent serial executor passed its own
  precheck and was submitted as Job `1178591`. Fresh run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_twostage_6f2ed48_serial_20260721_133422`;
  split-manifest SHA-256:
  `be84b85a38b9ec9176d80418a3a866e143a1e7073ab70f80805ebf570b82118a`.
- `1178591` entered RUNNING on `g0063`. Inside one allocation it will run
  the three P0 candidates, holdout-only selection, the real two-stage gate and
  the four frozen official-60 arms in order. The model/config/seed contracts
  are identical to the rejected parallel DAG; only scheduling is serialized.
- First P0 candidate `a1_t005_b8` reached epoch 8 and at least step 700 with
  finite total/frontend losses, requested fixed K=384, detector loss/bridge
  weight zero and explicit `duca_detector_path=skipped`. The batch-mean
  effective count can fall below 384 for shorter valid windows and remains a
  mechanism-gate diagnostic rather than a budget change. This is execution
  evidence only, not a selected frontend or mAP result.

## Claim boundary

Status is `experiment_running` after clean redeployment. The experiment
supports no C3/C4/C7 claim until the P0 gate passes and all four terminal EMA
checkpoints are evaluated.

## Exact-commit Pro audit and coordinator adjudication

Raw review SHA-256:
`0b265d08b811b821b1014cf7c52b579a759ee79e637710260a48cfc284367379`.
The external review statically audited exact commit `6f2ed48`; coordinator
verification used the clean remote snapshot at the same commit.

Confirmed blockers are:

1. P0's declared three-loss config inherits nonzero Python default losses,
   and every returned selector loss is summed into `cost`.
2. Transition-distribution supervision reaches ASFormer hidden features.
3. The official warmup uses shared AdamW/global clipping and does not prove
   frontend parameter, optimizer-state, buffer or EMA invariance.
4. The warmup/ramp boundary uses an inclusive comparison rather than the
   intended half-open successful-update intervals.
5. The current hard-forward/soft-backward bridge is connected but does not
   represent the true legal hard-swap detector-loss change after GT remapping.

Therefore Job `1178591` is reclassified as a running protocol-invalidated
diagnostic. Its P0 logs remain useful for understanding action, transition and
selection behavior, but it must not automatically authorize the downstream
official-60 matrix or a paper claim.

The leading repair is an exact-uniform/pure-delta anchored bounded residual
selector with detached coarse features, explicit loss ownership, isolated
optimizer transactions and detached legal hard-swap utility distillation.
This is `designed`, not `implemented` or empirically supported. The proposed
local radius and GO/KILL numbers require reachability, variance, cost and
matched terminal-mAP evidence before freezing.

## Second exact-commit audit and comparison

The second static review audited the same exact commit and independently
reached the same implementation verdict: `HOLD_CURRENT_CODE / NO_GO_AS_IS`.
Its raw SHA-256 is
`bca69084bfb1c09f5fe92d49aa10362b18fecf69ff8d2fa754c1d53335734703`.

It confirms the same hidden-loss, coarse-gradient, fake-freeze, global-policy
and bridge-alignment defects, and adds these exact-code risks: entropy loss can
push `p_action` toward 0.5; the named balanced BCE defaults to `pos_weight=1`;
radius-four training is misaligned with the radius-one gate; 192 of 197
descriptor dimensions are hidden differences; padded frames enter two
train-mode BatchNorm layers before masking.

The two reviews do not recommend the same detector feedback. The first uses a
detached hard-swap utility teacher and disables a direct bridge; the second
uses a local-cell hard-forward/soft-RGB-backward bridge after a real hard-swap
alignment gate. The coordinator therefore records neither as final. A common
alignment harness must decide the feedback route before any repaired
official-60 job is allowed.

The immediate sequence is: local-family reachability audit, P0 contract
repair, real-model feedback alignment, then matched `U/D/R0/R1` terminal-mAP
experiments. Job `1178591` remains diagnostic and cannot unlock this sequence.

## 2026-07-21 local reachability/P0 repair execution

- Holdout export Job `1178738`: `COMPLETED/0:0`, 120 records, source checkpoint
  SHA-256 `f604d4bdbb90058516d5aa90c9329550d2c79284597a5a6ed41fd0fc0920d94a`.
- Export decision contract: detector backbone false, GT/teacher/raw prediction
  false, selector-only inference true, training holdout only.
- Five-record pilot: local and global GT oracles both have r1 boundary recall
  and both-endpoint coverage 1.0; current checkpoint has r1 recall 0.8679 and
  both-endpoint coverage 0.7357. This is only a pilot.
- Eight-record stable pilot: local and global GT oracles again match at r1
  recall/coverage 1.0; current checkpoint is 0.8575 / 0.7150. Exact uniform is
  already 1.0 / 1.0 at r1, so r1 geometry alone cannot predict detector mAP.
- The first full run failed closed on record 7 because a 20-bit MILP tie-break
  returned variables outside [0,1]. The same row passed with 8-bit blocks; the
  full 120-record 8-bit run is in progress.
- Repaired P0 code/config status: `tested_local_contract`, `23 passed` for
  pure config and exact-solver suites. Exact commit, Linux Torch tests, CUDA
  one-step gate, candidate training and holdout winner do not yet exist.

## Full local reachability result and P0 repair checkpoint

The exact 120-record audit completed on 40 training-holdout videos. Raw
actionness quality is AUROC `0.6160999979`, AUPRC `0.3749576676` and Brier
`0.2042125679`. Exact-uniform, pure-delta, invalidated current checkpoint,
local GT oracle and global GT oracle mean endpoint distances are respectively
`0.4774574`, `0.5219517`, `0.5649486`, `0.2484311` and `0.2462081`.

Local and global GT oracles match on every reported boundary-recall and
both-endpoint-coverage radius. At radius zero their boundary recall is both
`0.2506608`, versus `0.1429119` for uniform; at radius one uniform is already
`0.9997746`. This proves local geometric reachability for the reported
boundary metrics, not detector-mAP optimality.

The repaired P0 implementation has `25 passed` pure tests plus Python and shell
syntax checks. CUDA status is `gate_pending`. The admissible next run is
frontend-only: one real THUMOS/AdaTAD-object one-step gate, then three
sequential P0 candidates and a training-holdout decision. The older official-
60 continuation remains locked.

Exact repair commit `5d17dcbe564efd1e69194dd5faddf34266e39f86`
passed `96` Linux focused tests with `2` skips. Clean snapshot:
`/data/run01/sczc063/yuzibo/projects/opentad_duca_local_5d17dcb_20260721`.
Single frontend-only Slurm Job `1178774` is pending from run root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_p0_5d17dcb_20260721_1640`.
Split SHA-256 is
`1a946a7890318ece2b2f500d84cccc2b2785e08f5780bbb5239ca208e9483be1`.
Status remains `experiment_running/gate_pending`; no candidate or mAP result
may be claimed.

## P0 real-gate classifier correction

Job `1178774` failed before candidate training because the gate grouped the
executed parameter path `probe_module.spatial_stem.*` using the obsolete name
`spatial_encoder`. The actionness loss did reach the coarse probe; the gate
misclassified the spatial-stem gradient as `coarse_other`. This is a gate
evidence bug, not evidence that the spatial frontend is disconnected.

The one-line classifier correction and regression test are sealed in exact
commit `9442b9487f871efd02c85dceeed26574c641369d`. Its clean Linux snapshot
passed `74` focused tests with `3` skips. Replacement frontend-only Job
`1178809` uses the unchanged split SHA-256
`1a946a7890318ece2b2f500d84cccc2b2785e08f5780bbb5239ca208e9483be1`
and the same simple sequence: one real gate, three sequential P0 candidates,
aggregate, stop. No old official-60 arm is authorized.

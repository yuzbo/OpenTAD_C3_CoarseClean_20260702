# DUCA selected-axis optimization official-60

## Question

Can a protected detector-gradient bridge improve fixed-K=384 offline TAD over
the same-commit exact-uniform control when detector geometry is represented on
the selected axis rather than the rejected sparse physical head?

## Frozen comparison

- Control: exact-uniform K=384.
- Arm A: learned policy from step zero; detector bridge uses the same delayed
  0-to-0.25 schedule as the homotopy arms.
- Arm B: canonical uniform warmup and continuous uniform-to-learned homotopy;
  detector bridge follows the shared delayed 0-to-0.25 schedule.
- Arm C: Arm B plus a training-only 50% exact-uniform companion within the same
  detector forward, inspired by Uni-AdaFocus input diversity.
- All arms: offline TAD, 60 epochs, 6000 successful updates, seed 3407,
  checkpoint every five epochs, terminal epoch-59 EMA only.

## Geometry and gradient contract

- Hard selected RGB frames are the only AdaTAD backbone input.
- GT segments are remapped by actual selected positions to selected-axis slots.
- Predictions are inverse-mapped to true time for evaluation.
- Official ActionFormerHead config/loss/NMS are unchanged; physical-grid mode is
  disabled.
- Action and transition supervision train the coarse ASFormer path.
- Detector loss trains the detector and transition scorer. It cannot update
  the ASFormer action head or temporal trunk in these three main variants.
- Companion rows are hard exact-uniform and block detector-to-selector gradient;
  learned rows preserve the protected structured transport bridge.

## Current evidence state

- Pushed implementation `1678d13` passed 34 clean-Linux focused tests and all
  four static config contracts.
- CUDA gate Job `1177721` reached the real THUMOS/AdaTAD graph, then failed its
  ownership-only backward because the old gate multiplied the transition
  objective by a fresh default GradScaler before accepting any replay. It did
  not start formal training and produced no mAP.
- Read-only diagnostic Job `1177724` split distribution, boundary-mass and
  combined transition objectives. At scales 1 and 65536 every gradient was
  finite; structured occupancy mass was exactly 384 for both rows, with range
  0.002256--0.997759. This does not support rewriting the selector DP.
- Zero-runtime diagnostics `1177722` and `1177723` exposed only temporary
  wrapper/PYTHONPATH mistakes and are not model evidence.
- The local successor fixes a comparison confound: all three learned arms now
  share one detector-gradient schedule, while only policy initialization/
  homotopy and the Uni companion differ.
- The successor also fixes gate semantics: seed 3407 is frozen before model
  construction; ownership losses use finite unscaled backward; exact-uniform
  is checked against canonical endpoint positions; and every arm must execute
  the production `train_one_epoch` AMP replay with a forced overflow, one
  successful optimizer update, and exactly one selector/scheduler/EMA update.
- Commit `c2de186f8edae3b3d19e799cff4792b44b827159` passed 35 focused
  Linux tests, but gate `1177732` required a selector-parameter delta while
  the proof step still had the formal warmup learning rate of zero. The
  production step itself ran; the failure is an audit-positioning defect, not
  a numerical model failure, training result or mAP.
- Exact replacement commit/tree is
  `1af6ff84f2cc5c4348710807bd960cea5d1741c0` /
  `95043c2eb7aed0247ed6eb53c7c72a4f61406047`; it changes only the
  gate proof setup by placing the scheduler at the first nonzero successful
  step. Formal training continues to start from step zero. Its clean remote
  snapshot passed 35 focused Linux tests.
- Replacement real CUDA gate Job `1177733` completed successfully in 4m23s.
  Gate suite SHA-256 is
  `38d5e185b36dd1ffc0adba979ce00623ed202b42d604eee811cf8f9c35d80c09`;
  all four full-model arms passed real-loader gradient ownership, actual hard
  positions, max-gap, forced AMP overflow/replay, optimizer, scheduler and EMA
  checks. The suite explicitly sets `formal_training_unlocked=true`.
- The first exact-commit official-60 Jobs `1177734-1177737` all failed in
  33--41 seconds before model construction or any optimizer update. The
  shared cause was protocol routing: `tools/train.py` applied the legacy
  terminal-epoch-131 checkpoint criterion to these terminal-epoch-59 configs.
  This is a deployment-contract failure, not four model or mAP failures.
- Exact pushed correction `cb89586a92b8b0a8349ecc9551bc50aa97982360`
  introduces `duca_selected_axis_optimization_v1`, freezes 60 epochs/6000
  updates/terminal epoch-59 EMA, limits CLI overrides, and verifies the exact
  gate/config/pretrain hashes at the production entrypoint. Its clean Linux
  snapshot passed 38 focused tests plus 23 C3/ASFormer regressions.
- Replacement four-arm real CUDA gate `1177776` completed successfully in
  4m26s. Gate-suite SHA-256 is
  `76628abd88f0e34f52ae1281c2beeae334798b33151cc45ba1d1ed706b970a27`.
  A separate read-only real-path runtime-binding preflight passed all four
  variants and was sealed at SHA-256
  `0844c0302ab1c456734c52a1a1fb4c844f61f96f19847b9753c9d4f5d0329504`.
- Replacement official-60 Jobs `1177779` exact-uniform, `1177780`
  direct-0.25, `1177781` homotopy-0.25 and `1177782`
  homotopy+uniform-companion were submitted and entered epoch 0 on four Slurm
  GPUs. All manifests bind commit `cb89586`, seed 3407, the exact config hash,
  common gate-suite hash and terminal epoch-59 EMA.
- Successor state: `experiment_running` at formal-training scope.
- Early runtime evidence: all four arms reached `duca_schedule_step=50` in
  epoch 0 with finite total loss 5.5197--5.5264, K=384 and 8596--8597 MB peak
  memory. Each arm replayed isolated AMP skips at batches 17 and 47 while
  reducing scale from 32768 to 16384; every event remained at replay 1/8 with
  no exhaustion. This proves real optimizer/schedule progress, not terminal
  stability or mAP.
- Epoch-0 closure: all four per-arm audit JSONs are present and hash sealed.
  Each records exactly 100 attempted batches, 100 successful optimizer,
  scheduler, EMA and selector-schedule updates, 102 optimizer attempts, two
  replayed AMP skips, `max_amp_retries_observed=1`, and zero replay
  exhaustion. All four entered epoch 1. This proves the formal update contract
  for one epoch only; terminal stability and mAP remain unproven.
- Runtime transition checkpoint: all four arms completed three epochs and
  exactly 300 successful optimizer/scheduler/EMA/selector-schedule updates.
  The first homotopy log after that boundary, step 350, reports
  `duca_phase=continuous_policy_homotopy` with finite loss and no fatal
  anomaly. Under the frozen 300-step warmup plus 1800-step cosine schedule,
  its expected policy alpha is approximately `0.0019`. The emitted
  `duca_schedule_progress=1.0000` is not policy alpha and must not be used as
  evidence that learned hard selection is already fully active.
- No hard selected-position overlap/geometry has yet been exported from a
  trained checkpoint. Entering the homotopy phase proves schedule execution,
  not policy deformation or boundary-quality gain.
- All four epoch-4 EMA checkpoints were written. First read-only diagnostic
  Job `1177987` failed before model construction because its launcher invoked
  a repository tool by file path without exposing the repository as a Python
  package (`ModuleNotFoundError: tools`). It is immutable launcher history,
  not selector evidence. Corrected Job `1178004` uses module entry points and
  is running on commit `cb89586`; it freezes the first 32 validation batches
  per arm, executes only the coarse probe and selector, and exports
  actionness, transition, uniform-overlap, boundary-distance, boundary-recall
  and max-hole diagnostics. It does not run the AdaTAD heavy backbone, mutate
  training state, evaluate intermediate mAP or select a checkpoint.
- At 09:40 +08:00 all four arms had completed at least seven epochs / 700
  successful optimizer, scheduler, EMA and selector-schedule updates;
  exact-uniform had completed epoch index 7 and entered epoch 8. All remained
  RUNNING on separate Slurm GPUs. No Traceback, OOM, ValueError, non-finite
  loss or AMP replay exhaustion was present. Each arm had three isolated AMP
  skips, all recovered at replay 1/8; the apparent `FAIL` grep hits were only
  fail-closed configuration field names.
- Corrected read-only diagnostic Job `1178004` completed successfully in
  8m23s and hash-sealed all four epoch-4 summaries. It covered 64 windows from
  40 validation videos and did not execute the heavy AdaTAD backbone.
- Coarse actionness at epoch 4 remained weak across arms: pooled AUROC
  `0.4625--0.4689`, AUPRC `0.2550--0.2571`, and F1@0.5
  `0.00044--0.00102`.
- Independent coordinate audit passed. For exact-uniform, all `64/64` exported
  GT windows and valid lengths reproduced from original THUMOS metadata, and
  formal training action targets equaled diagnostic labels at
  `46,527/46,527` positions. Recomputed AUROC/AUPRC was
  `0.463353/0.255005`; action mean p_action `0.458222` was slightly below
  background `0.459672`. One-candidate shifts changed AUROC by under 0.002.
  Thus the weak epoch-4 coarse classifier is not a coordinate or label bug.
- Radius-1 transition AUROC from the learned policy was only
  `0.5267--0.5300`, whereas pure `abs(delta p_action)` reached
  `0.6110--0.6175`. This localizes the early bottleneck to representation/
  policy scoring rather than exact-K or max-gap feasibility.
- Relative to uniform, direct, homotopy and companion improved exact boundary
  recall r0 by `+0.013899`, `+0.011332` and `+0.009066`, but reduced radius-1
  recall by `-0.092398`, `-0.098419` and `-0.092625`. Their mean max hole was
  `1.90625` versus uniform `1.796875`, still within the hard G=2 contract.
- This diagnostic is early mechanism evidence only. At 700--800 updates the
  protected detector-to-selector bridge is still zero by design and starts at
  step 2100. It cannot establish the value of end-to-end detector feedback or
  select an intermediate checkpoint.
- Independent read-only audit of exact commit `cb89586` found no P0 defect that
  invalidates or requires stopping the four arms. It did narrow the method
  language: interpolation is continuous in logits and soft occupancy, while
  the Viterbi hard path is piecewise constant and may change abruptly at score
  crossings. The current companion is an AdaFocusV2-inspired batchwise
  uniform-policy companion, not the paper's same-video second random-crop
  forward or an exact reproduction of its diversity augmentation.
- Under T=768/K=384/G=2 the feasible set still permits substantial frame
  relocation and local clustering, but radius-2/radius-4 boundary recall is
  nearly saturated by construction. Primary geometry evidence must therefore
  emphasize r0/r1, endpoint distance, gap distribution and short-window
  freedom rather than broad-radius recall.
- A direct hard-set audit excluded the three of 64 windows with
  `valid_len<=K`, where selection is necessarily all-frame. Across the 61 free
  windows, direct, homotopy and companion retained only `0.5143`, `0.5166` and
  `0.5184` of exact-uniform positions, replacing `186.5`, `185.6` and `185.0`
  of 384 frames on average. Their adjacent-selection rates were
  `0.3522--0.3553` versus uniform `0.0441`, and longest adjacent runs averaged
  about eight frames versus uniform `1.18`.
- The epoch-4 export is an inference-endpoint probe: `eval()/forward_test`
  forces learned arms to alpha=1, and every record confirms that value. It is
  not the alpha about 0.03 hard path used during epoch-4 training.
- Exact trajectory replay keeps every currently completed sample uniform for
  alpha 0 through 0.1. First observed changes begin at alpha 0.3, with mean
  first-change alpha about 0.34. At alpha=1 the policy still has ample
  clustering freedom and its endpoint clusters remain poorly boundary-aligned,
  but the previous small-alpha jump interpretation is invalidated.
- A separate pure-JSON decomposition was sealed under
  `diagnostics/epoch4_decomposition_v1` with manifest SHA-256
  `dd8a8603787d97747df75551cd7bb008130ac6d84c2ad9f663057b2a34053827`.
  Its evaluation-only GT-informed heuristic raises mean r0 boundary recall
  from `0.134192` to `0.247182` and lowers endpoint distance from `0.483369`
  to `0.241629`, but its mean max hole is `11.359375`, violating G=2. It is
  neither a feasible oracle nor deployable evidence and cannot be compared as
  a method arm.
- A dedicated read-only trajectory audit was added on branch
  `codex/duca-selected-axis-diagnostics-20260721` at commit
  `87cfd20938dd9411d8f1dc82091eaf9ec93c7f1d`. It accepts hash-bound exported
  scorer records, scans alpha through the exact soft/hard DP, and reports hard
  swaps/Jaccard, soft occupancy L1, plateaus, gap histograms, adjacent runs and
  short-window freedom. It constructs no detector or optimizer.
- Clean remote verification for that diagnostic commit is `47 passed, 2
  skipped`. GPU Job `1178357` was submitted to scan the three epoch-4 learned
  arms over 20 alpha values. The v1 submission attempt was rejected before a
  Job ID because N16R4 disallows an explicit nondefault memory request; v2
  script SHA-256 is `1ac8f120...de138` and is the only submitted run.
- Checkpoints: epoch-4 EMA exists for every arm; terminal epoch-59 EMA does not.
- At that runtime checkpoint terminal mAP did not yet exist; this statement is
  superseded by the partial terminal evidence section below.
- Claim `Avg-mAP > 65`: unproven.

## Runtime checkpoint at 2026-07-21 10:23 +08:00

- `1177779` exact-uniform: completed epoch index 14 / update 1499 and entered
  epoch 15; latest logged loss 4.3831.
- `1177780` direct-0.25: epoch index 14 / update 1450; latest logged loss
  4.3632.
- `1177781` homotopy-0.25: completed epoch index 13 / update 1399 and entered
  epoch 14; latest logged loss 4.4130.
- `1177782` homotopy+uniform companion: completed epoch index 14 / update 1499
  and entered epoch 15; latest logged loss 4.3829.
- All four logs show fixed effective K=384, finite component losses and about
  8.6 GB memory. Three isolated AMP skips per arm were recovered at replay
  1/8. No executed fatal anomaly exists.
- Detector gradient weight is still `0.0` by protocol before update 2100. The
  first planned post-bridge mechanism audit remains epoch index 24; no
  intermediate checkpoint will be selected by mAP.
- Trajectory diagnostic Job `1178357` is RUNNING from read-only commit
  `87cfd20`. It changes neither the method commit nor formal training state.

## Bounded residual diagnosis at 2026-07-21 10:40 +08:00

- Endpoint learned-score RMS is about 44 times the raw-delta RMS at epoch 4,
  so an unnormalized direct addition cannot test whether raw transitions help.
- Commit `7f9ad10` adds a read-only standardized residual sweep with
  `gamma=0,0.05,0.1,0.25,0.5,0.75,1,1.5,2,4`. It preserves gamma-zero hard
  selection and applies the same K=384/G=2 decoder at every point.
- Clean remote tests: `9 passed in 37.10s`.
- Job `1178384` is pending on `afterok:1178357`. Its output may diagnose a
  bounded successor but cannot select a checkpoint, tune on GT or establish
  detection mAP.

## Infrastructure termination at 2026-07-21 12:31 +08:00

- Jobs `1177779-1177782` are all `FAILED/1:0` after about 2h46m--2h51m.
- Every arm's `train.out` contains `OSError: [Errno 122] Disk quota exceeded`.
- The arms had reached approximately epoch 26--27 with finite losses before
  the shared write failure. This is not a numerical collapse or a terminal
  matched comparison.
- No arm produced terminal epoch-59 EMA mAP. C3/C4 and greater-than-65 remain
  unproven; any replacement must first restore and audit storage headroom.

## Recoverable checkpoint audit at 2026-07-21 13:22 +08:00

- Every arm retained its newest complete `epoch_24.pth`; earlier epochs
  4/9/14/19 were removed under the global one-valid-record retention policy.
- All four retained files passed full ZIP CRC and `data.pkl` validation, then
  loaded successfully with `torch.load(map_location="cpu")`.
- Each checkpoint declares epoch 24 and contains `state_dict`,
  `state_dict_ema`, `optimizer`, `scheduler`, `grad_scaler`, `rng_state` and
  `experiment_metadata`. They are genuine same-commit resume states, not only
  inference weights.
- Resume is admissible only with immutable method commit `cb89586`, the same
  resolved arm config and an explicit continuation receipt. It cannot convert
  this interrupted suite into terminal evidence without finishing the frozen
  60-epoch/6000-update protocol.
- Per-arm paths and SHA-256 hashes are sealed in
  `/data/run01/sczc063/yuzibo/cleanup_manifests/checkpoint_cleanup_consolidated_20260721_132157.json`.

## Exact continuation deployment at 2026-07-21 13:34 +08:00

- Continuation root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_selected_axis_cb89586_resume_e24_20260721_133337`.
- Resume gate Job `1178581` completed `0:0` in 8m55s. Gate-suite SHA-256 is
  `ef9ab397...6974`; resume-checkpoint gate SHA-256 is `614871a1...e125`.
  It reopened all four real-model configs on immutable commit `cb89586`, then
  verified storage and fully loaded each parent checkpoint.
- First continuation Jobs `1178582-1178585` failed before model construction
  or any optimizer update because the generated wrapper did not export
  `DUCA_EXPECTED_COMMIT` into `tools/train.py`. This is one launcher defect,
  not four resume/checkpoint/model failures.
- v2 Jobs `1178614-1178617` failed in zero to one second because the generated
  batch wrapper omitted the canonical `BASE` export. v3 Jobs
  `1178633-1178636` then failed before checkpoint restore because the wrapper
  did not pass `DUCA_SELECTED_OPT_GATE_SUITE` into `tools/train.py`. Both are
  launcher-contract failures with zero optimizer updates, not model results.
- The admissible v4 root is
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_selected_axis_cb89586_resume_e24_v4_20260721_135701`.
  All four absolute-path launcher preflights passed. A separate runtime
  preflight recomputed every checkpoint binding under the original official-60
  environment and original work directory; all four matched exactly. Its
  SHA-256 is
  `dae2775878465da16417faf33e20236bf8658f5d2df317b1ec9e5dda72d009d1`.
- Slurm Job `1178642` is RUNNING on `g0066` as a two-GPU/two-wave allocation,
  avoiding the account's submitted-job limit without serializing all four
  arms. Wave one is exact-uniform plus direct-0.25; wave two is homotopy-0.25
  plus homotopy+uniform-companion. Exact-uniform and direct-0.25 both restored
  epoch 24, completed all 100 updates of epoch 25 and entered epoch 26. Their
  selector schedule advanced continuously from 2500 to 2599; exact-uniform
  retained zero detector bridge, while direct-0.25 reached bridge weight
  `0.0623`. One direct-arm AMP skip replayed successfully under the frozen
  bounded-replay contract. The second wave remains ordered behind successful
  completion of the first wave inside the same allocation.
- Every arm resumes from its hash-bound `epoch_24.pth`, so the next loop epoch
  is 25. Model, EMA, optimizer, scheduler, GradScaler, RNG and formal audit
  state are restored; seed 3407, config hashes, checkpoint interval five,
  terminal epoch-59 EMA and official-60 evaluation remain unchanged.
- Parent checkpoint SHA-256 values are respectively `069916ba...8e59`,
  `612df5fb...bea6`, `ad760257...3eb` and `5cd37d6b...d61e`.
- Deployment-manifest SHA-256 is
  `c1c57afd23e3d44e08f02ca800845e086c79109f25a1a361a423a26157d6f273`;
  the two-wave batch script SHA-256 is
  `d0a129517dcc5cff27f7861584827d59bd9ac0a14f54d83e280879b02c21f182`.
- Cross-route runtime validation sealed the two resumed epoch-25 completions,
  the active P0 epoch-8 evidence, clean error scan and storage headroom at
  `runtime_validation.json`, SHA-256
  `3af133daa84e8d31de2c8cb5b08ca30b440a0e381461030e4007e82c9466c0b5`.
- Status remains `experiment_running`. Submission and resumability are not
  terminal mAP evidence; C3/C4 and Avg-mAP greater than 65 remain unproven.

## Partial terminal evidence at 2026-07-21 19:00 +08:00

- Job `1178642` remains `RUNNING`: wave one completed and wave two
  (`homotopy025`, `homotopy+uniform-companion`) is still training.
- Exact-uniform terminal epoch-59 EMA Avg-mAP is `64.4579977`; IoU-wise mAP at
  0.3/0.4/0.5/0.6/0.7 is
  `79.7557/75.5604/67.5863/56.7664/42.6212`.
- Direct-0.25 terminal epoch-59 EMA Avg-mAP is `63.7101546`; IoU-wise mAP is
  `79.4111/74.5551/66.6419/55.4667/42.4759`.
- Direct-0.25 is `0.7478431` Avg-mAP below the same-commit exact-uniform arm.
  The loss is present at every reported IoU threshold. This is terminal model
  evidence against the old direct-from-scratch bridge, not a launcher or
  numerical-failure diagnosis.
- Exact-uniform terminal artifact SHA-256:
  `a58fb9b76090d6da955c23563150f5fe24b6c9a1eddad9330b88dd5ba7d1ae1e`.
- Direct-0.25 terminal artifact SHA-256:
  `4f3f26ea17311212d8c86f2a7fba2687189a16557a0db24d5346e92d88f4747b`.
- The result does not adjudicate the repaired global-curriculum G1 arm:
  `4c777a6` first trains P0, freezes the coarse action branch, and confines
  detector feedback to the transition scorer. It must still be judged by its
  own matched U/G0/G1/G2 terminal table.
- C3, C4 and Avg-mAP greater than 65 remain unproven until the remaining V5
  arms and the repaired global-curriculum matrix finish.

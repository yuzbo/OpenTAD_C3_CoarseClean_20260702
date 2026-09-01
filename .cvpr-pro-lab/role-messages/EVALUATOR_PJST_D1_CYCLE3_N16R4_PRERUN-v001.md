# EVALUATOR_PJST_D1_CYCLE3_N16R4_PRERUN-v001

## Role and binding

- You are the sole Evaluator for the Critic-passed PJST-D1 Cycle-3 snapshot.
- Exact GitHub repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Exact branch: `codex/duca-pjst-cycle3-builder-20260826`.
- Exact commit: `cbefa51563adce5c512403695259f2fcb3da16fa`.
- Local clean binding: `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826`.
- Read the accepted contract and terminal reviews:
  - `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/pro-reviews/runs/duca-pjst-derivative-causal-freeze-v002/PRO_DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002.md`
  - `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/CRITIC_PJST_D1_CYCLE3_FOCUSED_REVIEW-v001.md`
  - `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/CRITIC_PJST_D1_CYCLE3_CHECKPOINT_BINDING_RECHECK-v001.md`
- Evaluation-only access. Do not edit or commit code, change scientific/config values, browse, call Pro/Sources, open official held-out data, or claim efficacy.

## N16R4 identity

Use the verified login, without printing credential material:

```text
ssh -o IdentitiesOnly=yes -o PubkeyAcceptedAlgorithms=+ssh-rsa -o HostkeyAlgorithms=+ssh-rsa -i /c/Users/skywalker/.ssh/id_rsa -p 22 -l 'sczc063@BSCC-N16R4' ssh.cn-zhongwei-1.paracloud.com
```

Canonical base/environment:

- `/data/run01/sczc063/yuzibo`
- `module load cuda/11.8`
- `module load miniforge3/24.11`
- `source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate`

Create/use one isolated evaluation checkout under `/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_cycle3_cbefa515_20260826`, fetch the exact pushed commit, and prove HEAD/tree cleanliness. Do not alter any shared dataset/checkpoint.

## Frozen resources

Read-only verify:

- Stage-1 epoch-29 checkpoint:
  `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
- required SHA-256:
  `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`
- THUMOS14 canonical videos: `/data/run01/sczc063/yuzibo/thumos14/raw_data/video` (411 valid canonical symlinks, 200 training + 211 validation; never add the two noncanonical physical files)
- annotation: `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json`
- category map: `/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt`
- VideoMAE-S pretrain: `/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`

## PRE_RUN gates

Perform only checks that change admission. Save raw stdout/stderr and exact commands.

1. **Linux focused tests:** run `tests/test_duca_pjst_d1_derivative_only.py` in the canonical environment. Zero skipped tests are required. Run `py_compile`, validator with the exact real Stage-1 path/hash/epoch, launcher `bash -n`, and `git diff --check`.
2. **Checkpoint identity/state:** prove readable file and exact SHA; `state_dict_ema` exists and can load into the frozen Stage-2 model with the expected key policy. Verify the Stage-2 initialization path actually consumes the resolved `workflow.model_initialization`, not merely records it.
3. **Model/config instantiation:** instantiate both resolved configs at the exact commit. OFF/ON parameter counts and state-dict keys must be identical; the only resolved experiment differences are output root and PJST flag. ON must reach the transform through the real detector/backbone path; OFF must not construct PJST metadata.
4. **Selector/exposure identity:** on training-population input only, perform the smallest non-optimizing OFF/ON structural smoke needed to prove the frozen selector produces exactly identical K384 positions, selected RGB, masks, ordering, and metadata under the same seed/input. Selector Jaccard must equal 1.0 and actual positions must be semantically nonuniform (not the exact-uniform fallback). This is identity evidence, never efficacy.
5. **Physical-time/PJST runtime:** prove finite OFF/ON forward and finite ON input gradient; exact-uniform and invalid-pair identity; exactly one PatchEmbed; selected-to-physical remap exactly once before filter/top-k/NMS; no selected-axis leakage at post-processing output.
6. **Distributed/launcher binding:** run the launcher with `PRECHECK_ONLY=1` using the real checkpoint/path/hash and exact source root in the target environment. Verify Slurm-safe `LOCAL_RANK/RANK/WORLD_SIZE`, rendezvous, no physical GPU pinning, official path binding, seed 3407, and distinct fresh output roots.
7. **Recovery contract:** establish that formal Stage-2 will save resumable `.pth` every 5 epochs, retain latest 3 plus milestone/final, and restore model, optimizer, scheduler, AMP scaler, epoch/update and required RNG/DataLoader behavior. Final epoch-59 and final-EMA are pre-registered; no intermediate cherry-picking. If the actual framework cannot save/restore the required state, return `PRE_RUN_BLOCKED` rather than relying on config prose.
8. **Resource/quota:** read-only verify sufficient storage and Slurm availability for the matched two-arm Stage-2 60-epoch/6000-update experiment. Do not submit formal training unless every gate above passes.

The structural smoke may use synthetic tensors or a training-population batch but is not performance evidence. Do not evaluate validation/test mAP during PRE_RUN.

## Admission and launch disposition

- If any gate fails, do not submit training. Return `PRE_RUN_BLOCKED` with the exact objective blocker and smallest owner/action.
- If every gate passes, return `PRE_RUN_READY` and, under the user's standing experiment authorization, submit exactly the matched `STAGE2_OFF` and `STAGE2_ON` formal jobs from the same Stage-1 epoch-29 checkpoint, seed 3407, official THUMOS14 training/validation protocol, 60 epochs/6000 successful updates, every-5-epoch recovery, fixed final/final-EMA, official evaluator, distinct immutable result roots. Do not add dense/uniform/random controls, dynamic K, support bridge, SingleClock, Query, or any other arm.
- A submission receipt must contain both Slurm job IDs, exact commit/tree, launcher/config, checkpoint path/hash, data paths, seed/update/epoch/checkpoint policy, resources, result roots, and stop rule. No result may be inferred while jobs run.

Write the durable receipt to `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/EVALUATOR_PJST_D1_CYCLE3_N16R4_PRERUN-v001.md` with `current_scientific_question`, `next_owner`, `next_action`, `dependency`, `expected_return_at`, and `single_recovery`. Update no model code.

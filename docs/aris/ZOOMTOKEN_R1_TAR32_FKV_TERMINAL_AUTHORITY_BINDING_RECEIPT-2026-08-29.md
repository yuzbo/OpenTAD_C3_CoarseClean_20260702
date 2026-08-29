# ZoomToken R1-TAR32-FKV terminal authority binding receipt

## Purpose and timing

This receipt freezes the authority, execution identity, and preregistered accuracy
decision rule for `ZT-CPTC-TAR32-TERMINAL-001` **before any terminal metric from
Slurm job `1260166` is read**. It does not report or imply a terminal result.

## Bound authority chain

1. Original user-confirmed Pro response:
   - path: `C:/Users/skywalker/.codex/attachments/80d2cc2c-87fe-403e-960f-c7a27467c77f/pasted-text.txt`
   - SHA-256: `50389340a26a16b658db9292e39865113528025b2d0a4fd3b857863e93c39322`
   - task: `ZT-CODEX-COMP-PROBE-001`
   - experiment: `ZT-R1-TAR32-FKV-S42-E60-v001`
2. Project-distilled assignment:
   - path: `.cvpr-pro-lab/reviews/PRO_R1_TAR32_FKV_COMPOSITE_PROBE_ASSIGNMENT-v001.md`
   - SHA-256: `a226328333a0f11aa1f6b227ce85569169800c8fab79ef4e82241dcd43ee7983`
3. Formal training start receipt:
   - path: `docs/aris/ZOOMTOKEN_R1_TAR32_FKV_FORMAL_TRAINING_START_RECEIPT-2026-08-29.md`
   - SHA-256: `9649b7d6f7a379eb4ca9d656dec8b90de77127adcd5bf92abb3a56a31bc3c196`
4. BPNS-R1 v004 terminal Pro review, which orders TAR32 terminal validation and
   a conditional matched cost closure:
   - path: `docs/aris/ZOOMTOKEN_BPNS_R1_V004_TERMINAL_PRO_REVIEW_RECEIPT-2026-08-29.md`
   - unique next task: `ZOOMTOKEN-R1-TAR32-FKV-TERMINAL-VALIDATION-AND-K100-MATCHED-FULL-STACK-COST-CLOSURE-v001`
5. User-supplied final route revision:
   - path: `C:/Users/skywalker/.codex/attachments/218e5755-062c-4fbf-97d6-e7b44c1bb82c/pasted-text.txt`
   - SHA-256: `ca5a82226307d602d2dae0aea7b555332335743175c21893bb4fc4b9a36de4c3`
   - decision: `REVISE_AND_CONTINUE`
   - route: `CPTC-vFinal-20260829` / Context-Preserved Transformation Compression
   - current single task: `ZT-CPTC-TAR32-TERMINAL-001`

The original user-confirmed Pro response contains no `request_id` or `nonce`.
Those fields are therefore recorded as **not present in the source** and are not
invented. Authority remains uniquely identifiable through the immutable source
path and SHA-256, the task and experiment identifiers, the exact candidate and
formal job, and the user's explicit confirmation of that response as the Pro
instruction.

## Frozen execution identity

- exact base: `2d945e64bdccd09ae2e2916524562e3f388c5a2a`
- branch: `codex/zoomtoken-r1-tar32-fkv-v001`
- clean/pushed candidate: `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`
- formal job / JobName: `1260166` / `zt-r1-tar32-fkv-s42-e60-v001`
- seed: `42`
- resources: two GPUs, local/global batch `1/2`
- training: `60` actual epochs with scheduler horizon `100`, AMP and EMA enabled
- primary result: epoch-59 `state_dict_ema`
- expected route: `[64,32,64,32,64,32,64,32,64,32,64,32]`
- spatial support: current-only contiguous R1 `8x8/K64`
- context and output semantics: all K64 remain current K/V, all K64 pass through
  the existing Adapter, non-selected odd-block tokens use identity residual bypass,
  and the detector receives the complete dense temporal representation
- prohibited additions: cache, old state, new parameter, new loss, dynamic K,
  fallback, rescue, resume, second seed, auxiliary arm
- remote source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_src_b0a1ca11`
- result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_v001_seed42_20260830`
- log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_logs_b0a1ca11_20260830`

## Frozen preregistered TAR32 accuracy gate

The original Pro response freezes the R1/FULL64 reference vector as
Avg-mAP / mAP@0.6 / mAP@0.7 = `69.07 / 61.14 / 46.57`. Admission requires all
of the following:

- Avg-mAP `>= 68.07`
- mAP@0.6 `>= 60.14`
- mAP@0.7 `>= 45.82`
- short-action mAP decrease `<= 1.50 pp`
- start-boundary median absolute error worsening `<= 10%`
- end-boundary median absolute error worsening `<= 10%`

These values are frozen before terminal metrics. They must not be replaced by
BPNS-v004 values, rounded terminal observations, or a post-hoc wider gate.

## Conditional next action

First perform one immutable terminal audit, one independent Critic, and one
result Evaluator. If training/model/checkpoint identity is valid and the original
accuracy gate passes, authorize exactly one same-job `R1/FULL64` versus
`R1-TAR32-FKV` matched full-stack cost measurement. The user-supplied final route
is the most recent authority for the arm identity and supersedes the earlier v004
receipt's `K100` label for this current TAR32 closure; `K100` is reserved for the
frozen successor residual-probe task. The matched order is
`R1/FULL64,TAR32,TAR32,R1/FULL64,TAR32,R1/FULL64,R1/FULL64,TAR32`, with 792
ordered items, 50 warmup windows per pass, decode-to-Soft-NMS scope, pass-local
20 ms power sampling, coverage `1.0`, and local maximum gap `<=100 ms`.

If the protocol is valid but accuracy fails, the frozen classification is
`STOP_R1_TAR32_FKV_EXACT_COMPOSITION`. If the checkpoint is valid but formal EMA
validation is absent, only one evaluation-only completion is allowed. If the
training/checkpoint identity is invalid, return an engineering/protocol blocker and
do not infer a scientific direction. `ZT-CPTC-RP-K100-v001` remains frozen and
unauthorized until a fresh Project Pro adjudication after this terminal closure.

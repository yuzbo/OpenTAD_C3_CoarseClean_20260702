# CPTC TAR32 evaluation-only pre-model blocker — fresh Pro adjudication request

## Immutable request identity

- request_id: `PRO_CPTC_TAR32_EVAL_ONLY_PRE_MODEL_BLOCKER_ADJUDICATION-v001`
- nonce: `ZOOMTOKEN-CPTC-TAR32-EVAL-BLOCKER-PRO-v001-20260829T203500+0800`
- exact Project: `g-p-6a79701398bc8191a9ef61db6302b24b`
- route authority: `CPTC-vFinal-20260829`
- current task: `ZT-CPTC-TAR32-TERMINAL-001`
- transport: attachment-only; `browserInlineFiles=false`
- requested response: one complete Chinese answer, no follow-up

## Your role

You are the ZoomToken project's scientific head, overall research planner, method
designer, process maintainer, and final reviewer. Codex is the executor and reports
the complete context below without proposing the scientific answer. Independently
review the framing, reject it if wrong, and consider alternatives not listed by Codex.
The priority is honest, fast paper evidence and model innovation, not an elaborate
engineering contract.

## Frozen scientific context

The final route treats context-preserved transformation compression as distinct from
state reuse: keep dense current token identity, full current K/V context, all-K64
Adapter execution, and a dense detector input; compress only odd-block nonlinear
transformation updates. The current TAR32 arm alternates K64/K32 Query-output-MLP
updates across twelve blocks, uses the immediately preceding dense block's attention
column mean for exact K32 selection, and uses identity residual bypass for unselected
tokens. There is no cache, old state, new parameter, new loss, dynamic K, or fallback.

The only training job, `1260166`, bound to candidate
`b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`, completed `0:0`. The epoch-59
checkpoint is structurally complete; its `state_dict_ema` SHA-256 is
`fc70557ef00788f8e788d59464d8c392943638c446d949d586fefc68c6d9390b`.
Training emitted no official final-EMA prediction/evaluator output. The single
independent terminal Critic therefore returned `PASS_WITH_BLOCKER`: mechanism,
route, context, gradients, Adapter and checkpoint identity passed; exactly one
evaluation-only completion was needed before any accuracy or cost conclusion.

The original preregistered R1/FULL64 reference is
Avg-mAP/mAP@0.6/mAP@0.7 `69.07/61.14/46.57`. Admission requires all of:

- Avg-mAP `>=68.07`;
- mAP@0.6 `>=60.14`;
- mAP@0.7 `>=45.82`;
- short-action mAP decrease `<=1.50 pp`;
- start-boundary median absolute error worsening `<=10%`;
- end-boundary median absolute error worsening `<=10%`.

These gates remain frozen. No terminal TAR32 metric has been observed or used to
alter them. If later admitted, the final route's current cost comparison is
`R1/FULL64` versus `R1-TAR32-FKV`; K100 belongs to the frozen successor residual
probe, not this TAR32 closure.

## Exact blocker

Codex built and pushed the evaluation-only launcher at docs-branch commit
`3ade5500`. It requested the same candidate/config, only epoch-59 `state_dict_ema`,
no training/resume/update, canonical validation, official evaluator and Soft-NMS,
and a fresh result root. It then made exactly one Slurm submission:

- job / JobName: `1261121` / `zt-r1-tar32-eval-b0a1`;
- state / exit: `FAILED` / `2:0`;
- submit/start/end: `20:26:04 / 20:26:08 / 20:26:12 +08:00`;
- elapsed/node/resources: `4 s`, `g0067`, two GPUs/eight CPUs;
- stderr: `canonical video inventory is not 411 MP4 files`;
- result root: not created;
- model/checkpoint/loader/evaluator: not reached;
- prediction/metric: absent;
- training/resume/parameter update: absent;
- formal submission count: `1`;
- scientific evaluation execution count: `0`.

Root cause is exact and result-blind. The launcher counted top-level regular MP4
files. The canonical video root contains 411 recursively nested MP4 symbolic links:
top-level regular=`0`, recursive symlink=`411`, recursive follow-links regular=`411`.
The minimal correction is one external-launcher line:

```bash
find -L "$VIDEO_ROOT" -type f -name '*.mp4'
```

It does not alter the candidate, config, checkpoint, dataset, evaluator, Soft-NMS,
accuracy gate, or scientific mechanism. The correction and complete blocker receipt
are pushed at `e311804f`. Codex did not submit a second job because the final route
states “一次提交”. No partial result is being interpreted.

## Independent adjudication requested

In one complete Chinese answer:

1. Distinguish engineering, protocol, and scientific evidence. State whether job
   `1261121` is only a pre-model operational blocker or has any scientific meaning.
2. Independently decide whether the one-submit rule makes the evaluation-only action
   irreversibly exhausted, or whether one replacement evaluation-only submission is
   scientifically permissible because the first submission executed zero model/data/
   evaluator work. Do not defer this decision to Codex.
3. If you authorize a replacement, freeze its exact scope and anti-drift constraints;
   if you do not, close TAR32 honestly and determine the only permissible next action.
   You may reject both framings and give another scientifically justified disposition.
4. Preserve or revise the original accuracy gate and the current-arm cost baseline only
   if scientifically necessary, explaining any change without using unseen terminal
   results. Do not use BPNS-v004 measurements to create a post-hoc TAR32 gate.
5. Decide whether `ZT-CPTC-RP-K100-v001` remains frozen or becomes the unique next task;
   do not authorize concurrent routes, rescue sweeps, second seeds, or silent retries.
6. Adjudicate the Pro/Codex role contract as `KEEP` or `REVISE`, with only the minimal
   process-rule change needed to prevent this kind of pre-science submission waste.
7. Assign exactly one next paper-experiment-development task to Codex, with explicit
   Beijing deadlines for Builder, Critic, Evaluator/formal action, terminal return, and
   mandatory fresh post-result Pro review. Explain how Codex should proceed continuously
   without foreground polling or repeated permission requests.

Start the answer by echoing the nonce, request_id, exact Project ID, the attachment
filenames actually used, and the browser-visible model route. Do not request a
follow-up and do not assume Codex's proposed correction must be accepted.

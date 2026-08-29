# ZoomToken CPTC TAR32 evaluation blocker Pro review receipt

## Turn identity

- request: `PRO_CPTC_TAR32_EVAL_ONLY_PRE_MODEL_BLOCKER_ADJUDICATION-v001`
- nonce: `ZOOMTOKEN-CPTC-TAR32-EVAL-BLOCKER-PRO-v001-20260829T203500+0800`
- exact Project: `g-p-6a79701398bc8191a9ef61db6302b24b`
- browser-visible model: `GPT-5.6 Pro`
- conversation: `6a92d5cb-ac50-83ea-8e0f-3ca229ce9ba7`
- URL: `https://chatgpt.com/g/g-p-6a79701398bc8191a9ef61db6302b24b-zoomtoken/c/6a92d5cb-ac50-83ea-8e0f-3ca229ce9ba7`
- submitted / completed: `2026-08-29T20:51:07+08:00` / `2026-08-29T21:03:57+08:00`
- transport: nine attachment-only files; `browserInlineFiles=false`
- scientific submission count: `1`; follow-up count: `0`

The response, Oracle transcript and metadata are complete. Their SHA-256 values are
respectively `faa8a4c719f63bc11983aeb5558c2d4b27e9c3b7ae718d99ff8b2a965798c979`,
`eec189d79e1586693ee77f6ccba10c12421d0cb96a1a9abf9c7c8ac388cd08b2`, and
`9ce8338c1d6c2d19a0732cd34885479e78157d0fc9f1f9e8f66ef687795c47d8`.

## Scientific adjudication

Pro returned `REVISE_AND_CONTINUE` and authorized exactly one replacement
evaluation-only completion for `ZT-CPTC-TAR32-TERMINAL-001`. Job `1261121` is
classified as `PRE_MODEL_EXTERNAL_LAUNCHER_DEFECT /
PROTOCOL_INCOMPLETE_BEFORE_SCIENTIFIC_EXECUTION / NO_TAR32_SCIENTIFIC_EVIDENCE`.
It remains the first Slurm submission but did not consume the one scientific
evaluation attempt because model, checkpoint, canonical loader, evaluator,
prediction, metric and parameter update were never reached.

The replacement must retain candidate
`b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`, epoch-59 `state_dict_ema` SHA
`fc70557ef00788f8e788d59464d8c392943638c446d949d586fefc68c6d9390b`, seed 42,
the frozen config/data/evaluator/Soft-NMS and two-GPU/eight-CPU resources. The only
runtime change is `find -L "$VIDEO_ROOT" -type f -name '*.mp4'`; the frozen
launcher SHA is `5b157901598782aeb62a95803ff4f8955c8402bfdcfcd2d1d6f9acf89b46e34e` at
commit `e311804f`.

Scheduler submission ordinal after replacement is `2`; scientific evaluation
attempt ordinal is `1`. A third submission, retry, resume, cost run, second seed,
auxiliary arm and `ZT-CPTC-RP-K100-v001` are forbidden. A second failure returns
directly to a fresh Pro review.

The original accuracy, short-action and boundary gates remain unchanged. Even on
an accuracy pass, the terminal classification is
`ACCURACY_ADMITTED_PENDING_FRESH_PRO`; matched `R1/FULL64` versus
`R1-TAR32-FKV` cost is not started in this task. Any valid accuracy failure is
`STOP_R1_TAR32_FKV_EXACT_COMPOSITION`; incomplete identity or diagnostics are
`ENGINEERING_OR_PROTOCOL_BLOCKER`.

## Role-contract revision

Pro returned `REVISE`: the repository role file and `RTK.md` now distinguish a
scientific attempt from a scheduler call for deterministic, result-blind,
pre-science launcher failures. One minimally reviewed replacement is permitted;
the old scheduler submission remains recorded and the replacement never creates
automatic authority for a third submission.

## Deadlines (Beijing)

- role sync / Builder plan: `2026-08-29 21:45`
- Builder evidence: `2026-08-29 22:15`
- change-surface Critic: `2026-08-29 22:45`
- result-blind Evaluator: `2026-08-29 23:15`
- replacement formal action: `2026-08-29 23:30`
- terminal return: `2026-08-30 12:00`
- mandatory fresh post-result Project Pro: `2026-08-30 13:00`

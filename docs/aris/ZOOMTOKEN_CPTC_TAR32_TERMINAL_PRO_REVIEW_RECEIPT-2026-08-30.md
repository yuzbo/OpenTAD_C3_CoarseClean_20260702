# ZoomToken CPTC TAR32 terminal Pro review receipt

## Turn identity

- Request: `PRO_CPTC_TAR32_TERMINAL_RESULT_ADJUDICATION-v001`
- Nonce: `ZOOMTOKEN-CPTC-TAR32-TERMINAL-PRO-v001-20260829T224000+0800`
- Exact Project: `g-p-6a79701398bc8191a9ef61db6302b24b`
- Profile/CDP: `61` / `127.0.0.1:15359`
- Browser route: `GPT-5.6 Sol / Power=Pro (5 of 5)`
- Conversation: `6a930db4-fb90-83ea-ae8b-16e5028b6a45`
- URL: `https://chatgpt.com/g/g-p-6a79701398bc8191a9ef61db6302b24b-zoomtoken/c/6a930db4-fb90-83ea-ae8b-16e5028b6a45`
- Started/completed: `2026-08-29T16:49:40.729Z` / `2026-08-29T17:05:07.051Z`
- Transport: eight attachment-only files; `browserInlineFiles=false`
- Actual scientific submissions: `1`; follow-ups: `0`; Project Sources mutated: `false`

The first automation attempt stopped before attachment upload, prompt submission or conversation creation because the updated browser UI separated the model from reasoning power. It consumed zero scientific submissions. The completed attempt used the same request and nonce after independently verifying `GPT-5.6 Sol` and maximum `Pro (5 of 5)` reasoning.

## Preserved evidence

- Request: `.cvpr-pro-lab/reviews/PRO_CPTC_TAR32_TERMINAL_RESULT_ADJUDICATION_REQUEST-v001.md`
- Response: `.cvpr-pro-lab/reviews/PRO_CPTC_TAR32_TERMINAL_RESULT_ADJUDICATION_RESPONSE-v001.md`
- Streaming receipt: `.cvpr-pro-lab/reviews/PRO_CPTC_TAR32_TERMINAL_RESULT_ADJUDICATION_STREAMING_RECEIPT-ATTEMPT2-v001.json`
- Terminal receipt: `.cvpr-pro-lab/reviews/PRO_CPTC_TAR32_TERMINAL_RESULT_ADJUDICATION_TERMINAL_RECEIPT-ATTEMPT2-v001.json`
- Oracle transcript: `.cvpr-pro-lab/reviews/runs/zoomtoken-cptc-tar32-terminal-pro-attempt2-20260830t002500/oracle-home/sessions/zoomtoken-cptc-tar32-terminal-pro/artifacts/transcript.md`, SHA-256 `6fab173e97ce288fb188ecb12e8bffbb48d2b658d523a5f9e903db4223599a23`
- Oracle meta: the same session's `meta.json`, SHA-256 `6acfc93a20b47824f1e415f417e5ff95c64de92c512d403fa828044ec62c9d47`
- Saved response SHA-256: `c27abb72748824e9799b40aa9c974a4d884eb962a6b516a84506898d77061c93`

## Scientific adjudication

Pro judged the terminal package valid and complete for the frozen single-seed accuracy admission. Engineering evidence is `PASS_STRONG`; protocol is `VALID_WITH_DISCLOSED_ROUTE_AUDIT_LIMITATION`. The exact composition is stopped as `STOP_R1_TAR32_FKV_EXACT_COMPOSITION`, while the broader CPTC family is not globally rejected.

- Project route: `PIVOT`.
- TAR32 cost, third evaluation, retraining, added seeds and post-hoc rescue remain forbidden.
- `ZT-CPTC-RP-K100-v001` remains frozen.
- Role contract: `KEEP`; no role-file or `RTK.md` revision is required.

The negative result applies to the exact R1/K64 plus `[K64,K32]x6` attention-column selector and identity-bypass composition. It does not prove that fixed half transformation necessarily fails on the native K100 grid. This is the only remaining high-information interaction ambiguity.

## Unique next task

`ZT-CPTC-K100-TAR50-INTERACTION-FALSIFIER-001`

Build a minimal clean descendant of `2d945e64bdccd09ae2e2916524562e3f388c5a2a` using native K100 and `[K100,K50]x6` (`800/400` flattened tokens), full K/V, full Adapter, dense detector input and exact identity bypass. Use the same immediately preceding dense attention-column mean selector, with no cache, new state, parameters, loss, distillation, dynamic K, fallback or residual predictor.

Bind strict A-MoD capacity=1 job `1254040` before submission. Its frozen route-matched reference is `68.73/61.59/47.20` for Average mAP/mAP@0.6/mAP@0.7. Run one seed-42, 60-epoch, two-GPU official THUMOS14 training-to-canonical-validation Slurm job and evaluate only epoch-59 `state_dict_ema`. The formal `sbatch` hard cap is one; failure returns to Pro without retry, resume or replacement.

All six gates must pass: Average mAP `>=67.73`, mAP@0.6 `>=60.59`, mAP@0.7 `>=46.45`, short-action drop `<=1.50 pp`, start boundary ratio `<=1.10`, and end boundary ratio `<=1.10`. Passing yields `K100_TAR50_ACCURACY_ADMITTED_PENDING_FRESH_PRO`; any valid failure yields `STOP_FIXED_HALF_UPDATE_ATTENTION_COLUMN_IDENTITY_BYPASS_FAMILY`; incomplete identity/protocol yields `ENGINEERING_OR_PROTOCOL_BLOCKER`. No cost or successor is authorized before the mandatory fresh Pro review.

Frozen Beijing deadlines: candidate `2026-08-30 09:30`; Critic `11:00`; Evaluator `12:30`; single formal submission `13:00`; terminal/blocker return `2026-08-31 12:00`; fresh Pro `2026-08-31 13:00`. The Builder-plan deadline was overtaken by the serialized browser turn and is disclosed rather than silently rewritten.

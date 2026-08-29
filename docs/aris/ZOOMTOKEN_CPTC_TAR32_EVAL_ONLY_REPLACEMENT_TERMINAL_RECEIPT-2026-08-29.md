# ZoomToken CPTC TAR32 replacement evaluation terminal receipt

## Terminal identity

- Task: `ZT-CPTC-TAR32-TERMINAL-001`
- Slurm job / JobName: `1261142` / `zt-r1-tar32-eval-b0a1`
- State / exit: `COMPLETED` / `0:0`
- Node / elapsed: `g0067` / `00:14:58`
- Start / end: `2026-08-29T21:11:38+08:00` / `2026-08-29T21:26:36+08:00`
- Candidate: `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`
- Checkpoint: epoch-59 `state_dict_ema`
- Execution: evaluation-only; no training, resume, optimizer step or parameter update
- Population: canonical validation, 211 videos / 792 ordered windows / 411 MP4
- Evaluator: official OpenTAD evaluator plus configured Soft-NMS
- Route contract: `R1-TAR32-FKV [64,32]x6`; all K64 remain K/V context and execute the existing Adapter; no fallback

The terminal receipt reports that every official evaluation forward completed under the
candidate's in-forward route assertions. The result root contains exactly the launch
receipt, terminal receipt, official evaluation log, official prediction and rank log,
plus the subsequently reconstructed diagnostic JSON. There is no standalone per-window
route-ledger artifact, cost profile or power trace.

## Immutable artifacts

| Artifact | SHA-256 |
|---|---|
| `launch_receipt.tsv` | `20b83360bd4b76aaf1f9994d94ef42c92b2a8d423022a894b1871013c9027604` |
| `terminal_receipt.tsv` | `b07b9644c6b7d99e2c611243a9be8136b2941109f7584d007ca48e16698f73d3` |
| `evaluation.log` | `1ecc94904017c0d8f0b7763c50c41dd73fcf86649aa5c2570d0440e6524f0381` |
| `gpu2_id0/result_detection.json` | `a70bae55e05219ed1c03b918fb57a4e560c19f0c8c6770908628b9b08ea767df` |
| `gpu2_id0/log.json` | `ec212a0e013dbc573a2221486aa2d63396323fa922bcc6bede6aa14078db8f41` |
| `terminal_diagnostics.json` | `2d2c3e6b9e41df22cc61d5442619299ea35d8dc8a9dfdd0bc6f6fc0bbfed1b96` |

The config, checkpoint, annotation, class map and pretrained identities match the launch
receipt. Their SHA-256 values are respectively
`b372d759c402bd82dbc758faa4b69e89351d757e57c8f76d1369f5fee7edc8ec`,
`fc70557ef00788f8e788d59464d8c392943638c446d949d586fefc68c6d9390b`,
`ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad`,
`a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31`, and
`4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251`.

## Official accuracy evidence

The terminal log directly records the two-decimal official values. Offline re-evaluation
of the frozen official prediction with the same OpenTAD AP primitive reconstructs the
unrounded vector below.

| Metric | Unrounded percent | Terminal-log percent |
|---|---:|---:|
| Average mAP | 64.98114078028013 | 64.98 |
| mAP@0.3 | 80.06313890018572 | 80.06 |
| mAP@0.4 | 75.25400276430542 | 75.25 |
| mAP@0.5 | 68.54872153389365 | 68.55 |
| mAP@0.6 | 57.37073781627205 | 57.37 |
| mAP@0.7 | 43.66910288674386 | 43.67 |

The official evaluator consumed 3,325 ground-truth instances and 422,000 predictions.
No traceback, OOM or non-finite failure occurred.

## Reconstructed short-action and boundary evidence

The result-blind tool `tools/bata/evaluate_zoomtoken_tar32_terminal_diagnostics.py`
reconstructed both arms from the frozen R1 and TAR32 predictions plus the canonical
annotation. These are reconstructed diagnostics, not official full-accuracy output and
not cost evidence.

| Diagnostic | Frozen R1 | TAR32 | Guard result |
|---|---:|---:|---|
| Short-action average mAP (`0 < duration <= 5 s`) | 41.15554711506739% | 37.83795524307160% | decrease 3.31759187199580 pp; **FAIL** (`<=1.50 pp`) |
| Median normalized start error | 0.09152542372880687 | 0.10000000000000014 | ratio 1.092592592592674; **PASS** (`<=1.10`) |
| Median normalized end error | 0.08292682926829344 | 0.08440972222222115 | ratio 1.017881944444422; **PASS** (`<=1.10`) |

## Frozen gate and disposition

The preregistered R1 reference is `69.07/61.14/46.57` for Average mAP / mAP@0.6 /
mAP@0.7. TAR32 misses the three main thresholds by 3.08885921971987,
2.76926218372795 and 2.15089711325614 percentage points respectively, and also fails
the short-action guard. The two boundary guards pass. Because admission is conjunctive,
the valid terminal classification is:

`STOP_R1_TAR32_FKV_EXACT_COMPOSITION`

This is a valid single-seed negative accuracy result for the exact composition. It does
not establish efficiency, novelty, multi-seed generalization, route-level causal behavior
or a general failure of context-preserved transformation compression. Cost measurement,
a third evaluation submission and the residual successor remain frozen. The mandatory
next action is exactly one fresh Project Pro adjudication of this complete package.

---
updated: 2026-07-08
status: active
scope: Absorbed severe review of the DUCA online plugin implementation, X3D train-free actionness path, and current experiment queue
out-of-scope: Claiming detector mAP results, treating prechecks as evidence, or endorsing offline JSONL/ledger pipelines as the final method
---

# DUCA Code, Experiment, and Train-Free X3D Severe Review Absorption

Raw record:

- `docs/methods/reviews/2026-07-08-duca-online-plugin-code-experiment-trainfree-x3d-severe-review-raw.txt`

Raw record SHA256:

```text
7F5551348235D1CDF2A854510F8C5413E41BB618B51D3CD997931AAE1EFF15BA
```

## Core Verdict

The current DUCA online plugin has a real detector-front selector scaffold:
`SingleStageDetector` can call a `frame_selector` before the backbone/projection
and the selector can gather selected temporal observations before handing the
shorter tensor to the detector. This is a genuine engineering milestone, not
only a README claim.

However, the current version is not yet a paper-ready DUCA main method. The
official AdaTAD full-train config still uses `ZeroShotMotionActionnessSource`,
not frozen X3D/Kinetics. The config itself marks deploy, metric, and paper
claims as disallowed. Remote evidence contains prechecks and smoke runs, but no
official detector full mAP result.

Cold conclusion from the review:

```text
The current version can be described as "DUCA online plugin scaffold passes
precheck", not as "DUCA-TAD works".
```

## Method Position

The final method must remain an online detector-front acquisition plugin:

```text
deploy-visible source -> DUCA acquisition -> original-time selected_positions
-> selected-only detector forward -> TAD predictions -> audit-only ledger
```

The review accepts that the skeleton partly satisfies this target, but it also
flags that the X3D train-free path is still outside the plugin. It currently
lives as an exporter plus JSONL plus geometry evaluation path, while the official
detector config still uses motion actionness.

The offline JSONL path may be kept only as a reproducibility cache or audit
trace. It cannot be the method story unless an equivalent online source path is
implemented and proven.

## P0 Blockers

- No official DUCA full mAP exists. `1150790 duca_off_adatad` passed precheck
  and then stopped because `FULLTRAIN_CANDIDATE=1` was not set.
- Frozen X3D/Kinetics actionness is not wired into the official selector forward
  path. It is still an external exporter/selection-eval route.
- Inference-side external `p_action` or actionness score injection is not
  provenance-bound enough. The review requires an `ActionnessPacket`-style
  contract instead of naked score tensors.
- Selected-axis training remap may distort high-IoU localization. True-time to
  selected-axis clamp/interpolation must be tested and justified.
- The queued validation X3D provider/interval grid risks hyperparameter fishing
  if its best cell is used in the main table.
- Efficiency claims are not closed. X3D decode and classifier cost must be
  included in any end-to-end cost table.

## P1 Required Fixes

- Add `FrozenKineticsActionnessSource` or an equivalent strict online/replay
  source integrated with `DucaOnlineFrameSelector`.
- Introduce a provenance-bound `ActionnessPacket` containing scores, source
  hash, clip parameters, cost ledger, and online-equivalence status.
- Make inference fail closed for naked or untrusted `p_action` payloads.
- Make CUDA requests fail closed; avoid silent CPU fallback in formal X3D runs.
- Replace ambiguous `manual` baseline naming with explicit names such as
  `external-actionness-duca` or `x3d-score-duca`.
- Ensure official selection eval cannot silently fall back to top-k when native
  DUCA decode or validation is unavailable.
- Add tests for real or faithful mock X3D tensor layout, selected-position to
  consumed-tensor exactness, GT remap round trips, and official ActionFormer
  mini-forward behavior.

## Train-Free X3D Verdict

Frozen Kinetics X3D actionness is conditionally reasonable as a generic
deploy-visible action prior. It is not yet enough as a main innovation.

The review is explicit that classifier confidence is not automatically temporal
actionness. `max_prob`, `inverse_entropy`, and `entropy_mix` are useful
diagnostics, but the final protocol must pre-register the score mode or treat
the alternatives as sensitivity analysis. A validation-set grid over provider
and interval cannot be used to pick the main result without a no-leak protocol.

Acceptable protocols include:

- a priori fixed provider and interval chosen before looking at THUMOS metrics;
- train-only tuning followed by one official validation/test evaluation;
- nested train split tuning;
- full sensitivity reporting without best-cell selection.

## Experiment Status Absorbed

Remote facts recorded by the review:

- `1150891 duca_x3d_grid` was pending as an X3D train-free interval grid.
- `1150809 duca_pre_gpu0seq` completed online AdaTAD and zero-shot motion
  prechecks, but remains precheck evidence only.
- `1150790 duca_off_adatad` failed after precheck because full-train candidate
  gating was not enabled.
- `1150818 duca_trainfree` was cancelled after the weak train-free route was
  rejected.
- `1150791 duca_uniform384` was cancelled and therefore does not provide a
  same-backend control.
- The X3D smoke output with one video, dense size 4, and four rows is only a
  chain smoke, not method evidence.

## Required Main Experiment Matrix

The main table must be same-backend, same-schedule, same-budget, and same-eval.
Minimum rows:

- full AdaTAD 768 baseline;
- uniform384, stride384, and random384 controls;
- motion384 top-k/radius as a weak train-free baseline;
- frozen X3D top-k384 with a pre-registered source setting;
- frozen X3D DUCA-radius384 with the same source;
- DUCA-TaskAdapted384 with teacher-free, GT-free, ledger-free inference.

Required supporting experiments:

- budget curve, at least 256/384/512 before expanding to 192/256/384/512/768;
- ablations for radius, redundancy, hole/gap, boundary utility, ST detector loss,
  and X3D prior;
- plug-and-play evidence on ActionFormer variants or another dataset;
- end-to-end efficiency table including raw decode, source forward, detector
  forward, wall-clock time, memory, and cache/no-cache accounting.

## Downgrade To Diagnostic Or Appendix

The following must not be treated as main evidence:

- C3, PAction, GAS-VT, lattice, and other offline ledger routes;
- oracle-actionness except as a diagnostic upper bound;
- precheck jobs and one-video smoke runs;
- validation X3D interval grid results if used to choose a best cell;
- adapter bridge/densepass/gridaware jobs that do not satisfy the final DUCA
  online plugin contract;
- old OnTAD/P1 jobs unrelated to the official DUCA AdaTAD plugin.

## Next Implementation Gate

The next code milestone is not another smoke run. It is a strict DUCA-X3D
integration:

```text
FrozenKineticsActionnessSource or ActionnessPacket replay
-> DucaOnlineFrameSelector
-> selected-only official AdaTAD/ActionFormer forward
-> no-leak manifest and cost ledger
-> official full mAP run
```

Until that exists, train-free X3D remains a diagnostic source study rather than
the final online plugin method.

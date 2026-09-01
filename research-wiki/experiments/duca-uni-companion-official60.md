# DUCA Uni-Companion Official-60

## Status

`deployment_failed_before_model_runtime`

The exact gate was submitted but failed before model runtime because the
generated non-login Slurm shell did not initialize the environment-modules
function. No optimizer step or mAP was produced.

## Immutable binding

- Branch: `codex/duca-uni-companion-inputfix-20260721`
- Commit: `4d84acda4d073fb6aac956c21386df8ed5d4d2f5`
- Tree: `b15a064784f25d888cc66df01c39781422403195`
- Snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_inputfix_4d84acd_20260721`
- Exact-snapshot Linux focused tests: `67 passed in 58.33s`
- Required legacy C3/ASFormer tests: `23 passed in 20.32s`
- P0:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_inputfix_4d84acd_p0_20260721_0430/protocol_manifest.json`
- P0 file SHA-256:
  `eabc6da8c3cc4308b70a8c8d6bbecc6c6e4b408cb17d2ee6041ed83f24a4eb3f`
- P0 content SHA-256:
  `e40252750c2fa680178fc45f38d014cb32a0e0acb34beb4477c5a6f0c8f02b93`
- Frozen training contract: batch size 2, 100 loader steps/epoch, 60 epochs,
  6000 successful updates/arm, seed 3407, terminal
  `epoch_59.pth:state_dict_ema`.

## Queue

| Role | Job | Dependency | Initial state |
| --- | ---: | --- | --- |
| P1/P2/P3 gate | `1177696` | none | `FAILED/127 at 00:00:00` |
| direct bridge 1.0 | `1177697` | `afterok:1177696` | `CANCELLED; never ran` |
| bridge 0.25 | `1177698` | `afterok:1177696` | `CANCELLED; never ran` |
| bridge 0.25 + companion | `1177699` | `afterok:1177696` | `CANCELLED; never ran` |
| rho=0.01 last-ASFormer adaptation | watcher `883230` | gate + direct completion | `EXITED; no Job ID` |
| transition without detector bridge | watcher `933605` | gate + rho completion | `EXITED; no Job ID` |

Official run root:
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_inputfix_4d84acd_official60_20260721_0435`.

Exact-uniform is implemented and P0-frozen but not queued because the account
submission limit is four Jobs. A bounded fail-closed watcher (PID `808310`,
script SHA-256
`98958f4c6dc0d57dd5412034c1eafc78bf087a803b35a4663486e561bb642cae`)
will submit exact-uniform only after gate `1177696` completes successfully and
the authorization artifact exists.

A second fail-closed watcher (PID `883230`, SHA-256
`c9b15b5f3a0cf369349a66548686db509a4dd90bf521ae4d418c5557860b1902`)
will submit the already P0-frozen `protected_e2e_rho001` arm only after both
the gate and direct arm complete successfully. This is the bounded test of
whether detector feedback scaled by `0.01` into only the final official
ASFormer encoder layer improves over selector-only feedback. It has no Job ID
and is not running yet.

A third fail-closed watcher (PID `933605`, script SHA-256
`940ddb5797d998850be2477ab47cc0a2fbaf840e76da12b62109c1ba4eaed136`)
will submit the P0-bound `transition_no_bridge` arm only after the rho watcher
has produced a valid receipt and that rho Job completes successfully. Its
launcher SHA-256 is
`ed406cec21c625cf09b4db9e94101c1fca776aa03de853795a8be5ef805121cf`.
Before submission it reopens P0, verifies the exact config source hash and
checks the original-four-arm authorization scope. This is deployment intent,
not a Slurm Job or result.

Deployment manifest SHA-256 is
`13f2b13c906f6605b8bbca6d06ad24201bcae83a027477dd42385b242807f6f4`;
`jobs.tsv` SHA-256 is
`9c9805019916a2547a1813c16d10072449cfc4199bbc91be8a8378badd995c4b`.

## Superseded zero-runtime submission

The earlier `d748684` jobs `1177687/1177690/1177691/1177692` were cancelled
with elapsed `00:00:00`. A real-loader audit showed that THUMOS full-train
windows are `torch.uint8`, while that exact selector revision rejected every
non-floating input before soft resampling. Those jobs would have failed before
meaningful CUDA or optimizer work and are not experimental evidence.

Commit `4d84acd` repairs only this production input contract: hard gather keeps
the exact raw values, while the differentiable soft-resampling/bridge branch is
promoted to FP32. It also makes the full-model gate compare the actual promoted
detector input. No AdaTAD/ActionFormer, SparseHead, Spatial-Zoom,
ChronoTransport, GAS-VT or spatial-crop route was changed.

## Claims

- Three learned versions implemented: `true`
- Three learned versions submitted: `true, but never started`
- Real-loader uint8 contract repaired and exact-snapshot tested: `true`
- Gate passed: `false`
- Terminal official mAP available: `false`
- Strictly above 65: `unproven`
- Paper ready: `false`

## Zero-runtime launcher failure

At `2026-07-21 05:49 +0800`, `sacct` recorded gate `1177696` as
`FAILED/127:0` with zero elapsed runtime. Its stderr is exactly:
`module: command not found`. The generated sbatch used `#!/usr/bin/env bash`
with `set -euo pipefail` but omitted `source /etc/profile` before `module
load`. This invalidates deployment only; it neither supports nor refutes the
model. The dependent jobs and all watchers were cleaned up to avoid stale
queue state. A successor may not be submitted until generated-sbatch tests
assert environment initialization.

## Registered diagnostic risk

The learned arms do not currently have a proven exact-uniform initialization.
`DucaProtectedTransitionScorer` is constructed with its default randomly
initialized output head. Separately, the global physical exact-K Viterbi
solver resolves tied paths lexicographically rather than to the canonical
exact-uniform positions. This is a hypothesis for early detector disruption,
not a diagnosed failure and not permission to alter the sealed run.

Required evidence before acting:

1. record initial and early-epoch selected-position overlap with exact uniform;
2. report boundary distance, max interval and per-cell displacement;
3. compare the sealed terminal mAP first;
4. if warranted, freeze a separate successor whose hard initial policy is
   exactly uniform and whose later deviation is learned, without reusing the
   current Job IDs as evidence.

Independent gradient-ownership review found a separate companion confound.
At the frozen batch size two, one row is uniform and has a constant hard
assignment, while only the learned row receives the `0.25` detector bridge.
Under comparable row statistics its aggregate detector-to-selector exposure
is therefore about half that of the plain bridge-0.25 arm. The submitted
companion remains a valid mixed-input diagnostic, but it does not isolate
input diversity at matched aggregate bridge strength. Any normalized
companion or uniform-to-learned homotopy must use a new exact commit/P0.

Lightweight solver-only diagnostics sharpen
the risk but are not dataset/model evidence. At `T=768,K=384`, the actual
exact-uniform physical cap is 3 frames. All-zero scores under the global
physical Viterbi tie-break have only `0.5000` set overlap with exact uniform
and `96.9974` frame rank-aligned mean absolute error. A synthetic seed-3407
random scorer/descriptors example has `0.5208` overlap, `7.4974` frame mean
rank error and 13-frame maximum rank error while still satisfying max gap 3.
By contrast, `exact_uniform_reference_scores` followed by the current
coverage-floor transform and physical Viterbi recovers the exact-uniform path
bit-for-bit. The random-descriptor number is illustrative only and is
superseded for initialization diagnosis by the bounded real-loader audit below.

A bounded real-loader audit after the input repair covered eight full training
windows. At random initialization it measured mean exact-uniform overlap
`0.502604`, mean rank-aligned error `4.038737` frames and maximum rank error
18. Mean nearest true-boundary distance was `0.528646` for the learned initial
path versus `0.555556` for exact uniform. This proves that initialization is
nonuniform and unstable, but it does not show an aggregate boundary-quality
deficit and cannot justify silently adding a homotopy to the submitted
protocol.

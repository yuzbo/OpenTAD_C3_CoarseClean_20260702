---
type: experiment
node_id: exp:georoute-estimator-representation-pilot-v1
title: "GeoRoute estimator/representation exploratory pilot v1"
idea: idea:geo-route-adatad
stage: experiment_running
status: fresh_c822add3_multiple_amp_hard_fail_waiting_closeout
verdict: incomplete_no_performance_inference_pending_closeout
confidence: medium
commit: c822add335c38a9f6c63e609237c4bfa9b9f468d
jobs: 1204301-1204314
updated: 2026-07-29
---

# GeoRoute estimator/representation exploratory pilot v1

## Question

On one frozen development-only seed, which effect explains detector utility:
PL versus ST credit assignment, geometry representation at fixed support,
geometry representation at learned ROI support, or ROI versus residual token
support under matched PL and representation-off settings?

## Frozen protocol

Six arms use exact `K=64`, seed `3407`, 20 epochs, absolute pretrained
position enabled, uniform-selected pooling, final-only checkpoints, and the
same immutable inputs. Four contrasts are preregistered. The Slurm DAG is six
parallel P0 leaves, one afterany P0 finalizer, six afterany fail-closed wrappers
that require the sealed PASS P0 suite before cell creation, and one afterany
non-promoting finalizer over all terminal leaves.

## Current state

The first runtime `02b6efe7` passed remote Linux tests `108/108`, but its six P0
Jobs `1203380`--`1203385` all failed mechanically before model reports and no
training ran. Jobs `1203386`--`1203392` were canceled after becoming
impossible; finalizer `1203393` exposed JSON-key-order validation and wrote no
finalization. The namespace is immutable failure evidence. A v2 repair is
implemented: module-mode P0 and early root bootstrap; Slurm-job-scoped
`127/8`, kernel-assigned ports, 120-second readiness and hashed failure
diagnostics with whole-process-group cleanup and same-leaf node revalidation;
arm-set normalization independent of JSON key order; fail-safe P0/final
control receipts; and the all-terminal fail-closed DAG above. N16R4 probes
`1203460/1203461` rejected `--resv-ports=2`, so it is not used.

Fresh runtime `cbe0a08218a2f4550960f7c832f88c8cf77757c1` was synced through
the academic proxy with exact HEAD/origin/clean-tree parity and passed remote
Linux tests `118/118`. Same-node gate Jobs `1203689/1203690` completed `0:0`
concurrently on `g0005` with distinct job-scoped hosts and four distinct
runtime TCPStore ports. Under the fresh run root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_cbe0a082_20260729_1849`,
P0 Jobs `1203707`--`1203712` and finalizer `1203713` completed `0:0`.
Sealed suite
`00b7c0e3251f3d384df91cf900267694918d1245b4a5803150e8e2e1465210d2`
is `PASS_MECHANICAL_ONLY`. Six training/evaluation leaves
`1203714`--`1203719` started in parallel. Job `1203715`
(`residual_pl_rep_off`) then hard-failed on real batch 0 after eight AMP
retries (scale `32768` down to `256`) and produced no checkpoint or metric.
The other five leaves completed `0:0` only to preserve terminal provenance.
Afterany closeout `1203720` completed `0:0` and sealed
`INCOMPLETE_EXPLORATORY_PILOT /
PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, with
`all_six_arms_passed=false`, an empty contrast set, and selector/P2/P3/official
test/paper claim all false. Its canonical self-hash and file SHA-256 are
`738e9875de2e9e08408263fd7d359e60f5ba1ca1912d0fbb9062a462c58cbf3a`
and
`63f73a353e356bc77a7a701972f22f62620b35e46b0c8f3eba0fc3c9816db0cc`.
Root cause is the FP16 production-horizon PL
temporal sum overflowing before GradScaler can recover, a case missed by the
float64 `T=1` KAT and synthetic P0. An FP32 likelihood/reduction repair plus
mandatory `T=384/N=220/K=64` AMP P0 KAT is implemented locally for a future
fresh source and full restart. Repair commit
`30f9ca6fff1572e2eabc6c1b6636c4cc23595a62` is now proxy-synced in an exact
clean snapshot and passed the complete remote Linux suite `120/120`.
Standalone CUDA KAT Job `1203873` completed `0:0`: half source, FP32
likelihood/loss, objective magnitude `128637.0234375 > 65504`, and finite
scaled gradients. Receipt internal/file SHA-256 values are
`7d0ccc346b95180d02a5ddcf4253ac0278e83f39a6f7e434357c86067e3c8e84`
and
`75ef280473f5032fd734fb86f1f58207702c1999d34c5c7132d40ff5017ae4a4`.
This is `tested` numerical-correctness evidence only. The current experimental
node still has a hard-failed arm and is not `empirically_supported` or
`paper_ready`.

A fresh history-free agent independently audited the raw Pro review and its
absorption, frozen six-arm contract, no-leak paths, PL AMP repair, P0/finalizer
logic, and all-or-none 14-job deployer. It returned
`DEPLOY_AFTER_OLD_CLOSEOUT_AND_CAPACITY`. This closes the need for another Pro
discussion before the minimal pilot, but it does not bypass old closeout,
submit capacity, fresh per-arm P0, or all-six completion. The audit is archived
at
`docs/methods/reviews/2026-07-29-georoute-estimator-pilot-independent-agent-audit.md`.

The old closeout and all-at-once capacity conditions are now satisfied. With
two unrelated active jobs, the deployer admitted its complete 14-job DAG at the
account limit `2 + 14 = MaxSubmitJobs 16`; it did not split or cancel work.
Exact clean runtime `30f9ca6fff1572e2eabc6c1b6636c4cc23595a62` created
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_30f9ca6f_20260729_2023`.
Deployment internal/file SHA-256 values are
`09e837c5d1815ce4a60e58eb06b91ca366edcf71043362dd7a04ad0e986d1ca1`
and
`6bdd27425ab95eace6bfe93db361299be8966fc97b70ae6d30958964378d0b7c`.
P0 Jobs `1204015`--`1204020` and finalizer `1204021` completed `0:0`.
Suite internal/file SHA-256 values are
`2aea448be4c8d72957b3c904bb22c5ae39689cb0010c3b18a4914bd71f5265ec`
and
`6d832d164fcce9889733bcf8ef5bcb6506d787b79147b56640645ec9a02a9a37`;
status is `PASS_MECHANICAL_ONLY`. Critical score-function P0 `1204016` is
schema v4 and passed the real `180x320`, `11x20/N=220`, `T=384/K=64` AMP
horizon with FP16 source, FP32 likelihood/loss, objective
`128637.0234375 > 65504`, and finite scaled gradients. Its report internal/file
SHA-256 values are
`ab0b1304c62168dc6c257a18b6efc06eddbe12af8a62d0a5bedab63f5a2ce747`
and
`85eaea59cfb56ab40bd7b30c8815b46a959904b0c9b0c05e1b68a8e918589358`.
Six stage Jobs `1204022`--`1204027` now run in parallel; afterany closeout
`1204028` remains dependency-held. This is `experiment_running`, not a result.

Residual-PL stage `1204023` then failed on its first real batch after all eight
AMP retries (`32768` down to the attempted floor scale `256`) without a
checkpoint or metric. Failure internal/file SHA-256 values are
`f70b0a541cbfbbcf6595e8dfe7d7ef46ce16426d09a5ff7bc9fc921273c9eb81`
and
`e36556f20a5cdbf138779fd46efc2f31462aaec19dc49296bb23fa94180edb5e`;
traceback SHA-256 is
`405b761dcf158daf952752095df3d01e9c815147b294d3816899809a73400fc7`.
The other five jobs continue only for terminal provenance. One new
history-free independent agent returned `HOLD -> REPAIR`: P0 wrapped neither
its real model forward nor backward in autocast/GradScaler, then ran the
production-horizon AMP assertion only on an isolated logits leaf. Thus the
suite did not test a scaled optimizer update across scout/detector/adapter/
backbone. The frozen six-arm questions remain valid, but this namespace cannot
emit any contrast and must close INCOMPLETE.

The next source was implemented without changing the experiment. GeoRoute
backbone schema v4 disables autocast for the complete FP32 scout/route graph,
and P0 schema v5 requires a real full-model FP16-autocast, GradScaler-256
backward, unscaled required-gradient audit, and successful zero-learning-rate
optimizer step. The isolated production-horizon KAT remains but can no longer
authorize stages alone. Exact source
`c822add335c38a9f6c63e609237c4bfa9b9f468d` passed the exact clean remote
Linux suite `121/121`. Standalone CUDA P0 Job `1204087` completed `0:0`.
Its schema-v5 report internal/file SHA-256 values are
`4a9cd451e59417b6e606e841bcda47ebd5dd9b8b4c45ee3cfd42c0e0922d88aa`
and
`6dee7330c9e77c2bb78281e8410fbde4a768fedf88d841c422e2cb9458e40293`.
It binds Slurm ID and same-leaf rendezvous, `180x320 -> 11x20/N=220`,
`T=384/K=64`, FP16 source, FP32 likelihood/loss, objective
`128637.0234375 > 65504`, full-graph scale `256 -> 256`, finite required
gradients, FP32 scout execution and a successful optimizer update. It created
zero checkpoints and no metric/test/claim. State is `tested` numerical
correctness only; old run `1204028` must seal INCOMPLETE before a wholly fresh
six-arm restart.

That closeout condition is now satisfied. The five surviving jobs completed
only for provenance, and Job `1204028` completed `0:0`. Finalization status is
`INCOMPLETE_EXPLORATORY_PILOT`, decision is
`PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, `all_six_arms_passed=false`, and
the contrast map is empty. Self/file SHA-256 values are
`60c9dab575e65830b7b849437963de2c7f789743caedb130b499c142c49c76ab`
and
`6ad32b7822042685b353f378c6eb9ea14be061e7f22d7db7288a129cbe080f06`.
No five-arm metric is interpreted.

The complete fresh restart is now `experiment_running`. Capacity preflight
recorded `active=2`, `additional=14`, `MaxSubmitJobs=16`, with zero remaining
headroom. Exact clean source `c822add335c38a9f6c63e609237c4bfa9b9f468d`
created
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_c822add3_20260729_2149`.
Deployment self/file SHA-256 values are
`7f445af550cf359b3d13d174f1199fa9b45192223461291f73f66a001c97b202`
and
`48f19fd88741361e9290ae6f444ccbd38f07030c82bc0c2726abad514de1158b`.
P0 jobs are `1204301`--`1204306`, P0 finalizer is `1204307`, stages are
`1204308`--`1204313`, and afterany closeout is `1204314`. Arms, seed `3407`,
`K=64`, 20 epochs, inputs, preexperiment parent and four contrasts are unchanged.

All six P0 Jobs and finalizer `1204307` completed `0:0`. The sealed P0 suite is
self-hash-valid `PASS_MECHANICAL_ONLY`; its internal/file SHA-256 values are
`f6f423670c9c2417aadfca97c67d794427ee337c359ba2d2509faee53a5ccdb6`
and
`114cef2be09cc674429d6d732991048a73345c0a8d3583f56310a5a05344bba0`.
This did not predict the real batch. Residual-PL Job `1204309` started epoch 0,
retried the first batch at GradScaler values
`32768,16384,8192,4096,2048,1024,512,256`, and then raised
`FloatingPointError: S1 AMP could not produce a successful optimizer update
after 8 retries`. Slurm accounting is `FAILED 1:0`, with no OOM or rendezvous
error. Failure receipt internal/file SHA-256 values are
`5e55619291285a36a8410be9582a79f293b33cd135053b4c6a3967e9d8beb5c8`
and
`77c9eff76c5881f8daa45ef68dd0d0d71bec2f5e171f31e2a8a97fc734b9f3c5`;
traceback SHA-256 is
`17ec9adb5a41b48a16d0a76221248a4bfe4f123e99ec03e16f6397b0919649ad`.
The failed cell inventory contains only its hashed failure receipt, storage
preflight, bound config, `train.out`, and `log.json`: no checkpoint, prediction,
metric, or stage result exists. The other five stages run only to terminal
provenance and may not be interpreted. No arm is resumed or replaced. Closeout
`1204314` must seal `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE` with empty
contrasts and all promotion guards false.

ROI-PL representation-on Job `1204313` subsequently crossed the preregistered
recoverable-skip boundary while still running for terminal provenance. Its
eleventh `AMP skipped batch` event was logged at
`2026-07-29 22:47:59 CST`, batch 111, retry 1/8, scale `64`. At the observation
time, Slurm still reported `RUNNING`, the logged loss/cost remained finite, and
there was no Traceback or OOM. Nevertheless, `>10` skips is a protocol hard
failure; later process completion or a final checkpoint cannot make this arm or
the namespace performance evidence. ROI-PL representation-off `1204312`
remained at exactly ten skips at the same audit point and was not independently
reclassified.

The evidence narrows the execution diagnosis: schema-v5 synthetic full-graph
AMP is mechanically valid but insufficient as a real-batch stability gate. It
does not establish that PL is scientifically inferior to ST, nor does it
support another estimator repair without a new cause analysis and experiment
decision.

## Claim boundary

For the sealed `cbe0a082` namespace, finalizer `1203720` emitted
`PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE` with an empty contrast set; none of
the five completed-arm outputs may be interpreted. The `30f9ca6f` namespace is
also sealed under the same decision, and the current `c822add3` namespace has
already hard-failed one arm and must do likewise when `1204314` runs. A future
fresh namespace may emit descriptive single-seed contrasts only if all six arms
pass. No finalizer can emit a winner, reuse the historical selector, promote
P2/P3, open official test, or authorize an efficiency, Geometry Zoom, or paper
claim.

## Provenance

Protocol:
`docs/methods/2026-07-29-georoute-estimator-pilot.md`.

## Connections

Maintained in `research-wiki/graph/edges.jsonl`.

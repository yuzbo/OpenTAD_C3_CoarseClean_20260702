---
type: source_registry
updated: 2026-07-27
---

## 2026-08-23 EAST 50Salads official RGB reproduction

- Official EAST repository and immutable upstream revision:
  `https://github.com/tqosu/EAST` at
  `a3233c2e6a6e3bbe36f9663e18180bdc5c126556`.
- Local read-only official clone:
  `E:/DeskTop/TAD/external/EAST_DUCA_20260822`.
- Clean execution candidate and remote checkout:
  `37c0d080a2bce948dc73643578f05b2229934d2c` at
  `/data/run01/sczc063/yuzibo/projects/duca_east_baseline_37c0d08_20260822`.
- Canonical raw RGB/protocol root:
  `/data/run01/sczc063/yuzibo/datasets/TAS/east_50salads_160x160_2fps`;
  50 videos, `121,662,019` bytes, `160x160 @ 2 fps`, five 40/10 folds.
- ViT-G pretrain:
  `/data/run01/sczc063/yuzibo/pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth`;
  ViT-S pretrain:
  `/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`.
- Evidence boundary: ViT-G failed before its first optimizer update on 24 GiB GPUs,
  so it has no metric. Official ViT-S two-epoch train/evaluator admission `1249796`
  completed. Five-fold full training `1249797_1…5` then completed fixed epoch-199
  EMA testing with Avg-mAP `81.68/83.59/86.08/84.62/82.97`; the five-fold mean is
  `83.79`, with mean tIoU 0.1/0.25/0.5/0.6/0.7 mAP
  `89.50/88.21/84.42/80.78/76.03`. ViT-S cannot substitute for the published
  ViT-G anchor, for which the official repository provides the paper result but
  no public S-model result.
- Official released detector checkpoint source:
  `https://oregonstate.box.com/s/c14851yhp3pibqefkfbcrqukrw0eatis`. On
  2026-08-23 the official share required Oregon State University Box login in
  anonymous HTTP and both available browser sessions; no detector checkpoint was
  present locally or in the bounded remote inventory. The 2.03 GB
  `vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth` is backbone pretrain,
  not the released EAST detector. Evidence status:
  `CHECKPOINT_ACCESS_REQUIRED / NO_EVALUATION_SUBMITTED`.

## 2026-08-21 DUCA temporal-action-segmentation migration sources

- MS-TCN++ official repository: `https://github.com/sj-li/MS-TCN2`.
- Current upstream master inspected: `f423a9e65f4ccb1cd7322eb9f94946a19e787993`;
  it is syntactically broken at `model.py:14` and was not altered.
- Executable official historical anchor:
  `9d31fb3c23467b9ce3030d43b6d33a96869b6422`.
- Local clean clone: `E:/DeskTop/TAD/external/MS-TCN2_DUCA_20260821`.
- Remote clean clone:
  `/data/run01/sczc063/yuzibo/external_official_action_segmentation_repos/DUCA_MS-TCN2_9d31fb3`.
- Official data record: `https://doi.org/10.5281/zenodo.3625992`, `data.zip`
  `30,210,005,282` bytes, CC BY 4.0. Remote direct download was terminated after
  observed throughput around 15 KB/s; the partial file is not admissible data.
- FineGym official sources: `https://sdolivia.github.io/FineGym/` and
  `https://github.com/SDOlivia/FineGym` (CVPR 2020; official action-instance
  recognition/localization protocols, not a native MS-TCN++ TAS protocol).
- FineDiving official source: `https://github.com/xujinglin/FineDiving` (CVPR
  2022; procedure-aware action quality assessment, not a standard TAS benchmark).
- Evidence class: official-source/code identity and remote execution-preparation
  evidence only. No TAS training, metric, efficiency, or paper claim exists.

## 2026-07-20 DUCA physical allocation-family Pro review

- Repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Exact audited evidence commit:
  `4ce69c852bdbd902046b47bc6019ae11e850dbe4`.
- Immutable trained-model ancestor:
  `1642f265e48391418a7c8a4a087e33e2b7bf6899`.
- Original attachment:
  `C:/Users/skywalker/.codex/attachments/6190aa05-a97e-4c41-82a7-7c74bed997ad/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-20-4ce69c8-duca-physical-allocation-family-pro-review-raw.txt`.
- Raw/archive size: `75,867` bytes, `2,001` lines.
- Raw/archive SHA-256:
  `E40A69BD2DA9EBE32B41B45A136C2AA1A9FB8109A4875A16E2E3ABB7AF8FCC14`.
- Structured absorption:
  `docs/methods/2026-07-20-4ce69c8-duca-physical-allocation-family-pro-review-absorption.md`.
- Reviewer verdict: `HOLD_AND_REVISE_FAMILY`; replace fixed `192+192`
  scaffold-first CARA with a global exact-K physical-gap ceiling before any
  new model training.
- Project absorption:
  `SUBSTANTIAL_ACCEPT / NOT_FULL_ACCEPT`. Independent checks confirmed the
  stride-4 physical conversion, uniform 12-frame maximum interval and
  physical-15 scaffold minima `255/382`. The value/unit of the cap, proposed
  solver code, paired statistics and non-independent val/test protocol still
  require correction. No implementation, experiment or claim was promoted.

## 2026-07-19 DUCA-CARA feasible-set ceiling Pro prompt

- Prompt:
  `docs/methods/prompts/2026-07-19-duca-cara-feasible-set-ceiling-pro-audit-prompt.md`.
- Repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Exact review tree:
  `4ce69c852bdbd902046b47bc6019ae11e850dbe4`; immutable model ancestor
  `1642f265e48391418a7c8a4a087e33e2b7bf6899`.
- Scope: one bounded Pro review of allocation-family mathematics, physical
  coordinates, exact ceiling optimization, implementation blueprint, tests
  and GO/HOLD/KILL criteria.
- Boundary: this prompt is a discussion artifact. It authorizes no model
  implementation, training, experiment result or claim transition.

## 2026-07-19 CellCF KILL / CARA redesign Pro review

- Repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Audited model commit:
  `1642f265e48391418a7c8a4a087e33e2b7bf6899`.
- Audited cost/evidence commit:
  `4ce69c852bdbd902046b47bc6019ae11e850dbe4`.
- Original attachment:
  `C:/Users/skywalker/.codex/attachments/d5c1b391-e4d6-45bb-b835-7ec95a146b0c/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-19-1642f26-4ce69c8-cellcf-kill-cara-redesign-pro-review-raw.txt`.
- Raw/archive size: `68,339` bytes.
- Raw/archive SHA-256:
  `3FB06655193E7CF665BB37CF0701C2708139B15DF40AC2114742C23B19E292E7`.
- Structured absorption:
  `docs/methods/2026-07-19-1642f26-4ce69c8-cellcf-kill-cara-redesign-pro-review-absorption.md`.
- Reviewer verdict: current CellCF `KILL`, broader DUCA `REDESIGN`, dynamic
  MUST frozen.
- Project absorption: `PARTIAL_ACCEPT_STRONG_DIAGNOSIS / HOLD_NEW_METHOD`.
  Independent source/math checks confirmed one-per-cell quota rigidity,
  actual-position/fixed-anchor coordinate substitution, detached rather than
  direct detector gradients, the non-upper-bound ceiling tooling and the
  existing physical-grid head. The proposed CARA architecture, `G=3`,
  `192+192`, loss weights, cadence, thresholds and schedules remain
  unimplemented hypotheses.

## 2026-07-18 DUCA-CellCF cost-recovery evidence

- Evidence repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Branch: `codex/duca-cellcf-evidence-20260717`.
- Current exact evidence commit:
  `e153c96bfa0f37b9d4b82046e05b1bbce70dfe50`.
- Clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_evidence_e153c96_20260718`.
- Exact Linux gate: `230 passed`; Python compile, Bash syntax, exact HEAD and
  clean-tree checks passed. Independent review: `GO`, P0=0, P1=0.
- Immutable trained commit remains
  `1642f265e48391418a7c8a4a087e33e2b7bf6899`; no model retraining occurred.
- Original terminal jobs: `1167485 FAILED/1:0`, `1167486 CANCELLED`.
- Failed recovery diagnostics: `1170338` and `1170354`, both cancelled with
  zero runtime. Their roots are retained and are not reused.
- Current recovery root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200/cost_recovery_e153c96_v1`.
- Recovery outcome: Job `1170366 FAILED/1:0` after 1,357 seconds because
  `build_profile_summary()` rejected seven profiler-produced
  `*_cpu_enqueue_ms` fields as unsupported. Dependent Job `1170367` was
  cancelled with zero runtime.
- Recovery manifest SHA-256:
  `e595768d3ddfeccb47d32d5fd0e1a476cbb81b9587ba66165d5e7ef66e8d6c4a`.
- Recovery ledger SHA-256:
  `96e8f27ad9e6f47f31f30d340e3cee8a3389a173f675113a52eb34ccef00d2b2`.
- Boundary: status returns to `implemented + tested, deployment_failed`; no
  successful measured cost, final-suite artifact or external seal exists.

## 2026-07-17 DUCA-CellCF terminal raw metrics

- Formal commit:
  `1642f265e48391418a7c8a4a087e33e2b7bf6899`.
- Root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200`.
- Jobs `1167481/1167482/1167483`: all `COMPLETED/0:0`.
- Avg-mAP: uniform `0.6385937484867475`, transition-beta0
  `0.6427548755668553`, CellCF `0.6406099697875949`.
- IoU 0.3/0.4/0.5/0.6/0.7: uniform
  `0.788009/0.734968/0.665040/0.568974/0.435978`; transition
  `0.789614/0.744893/0.672996/0.574936/0.431336`; CellCF
  `0.788992/0.746776/0.666185/0.562856/0.438241`.
- Boundary: aggregate `1167484` is complete, but cost `1167485`, completion
  `1167486` and the repaired external seal are not. These are raw one-seed
  metrics, not final claims.

## 2026-07-17 DUCA-CellCF post-run evidence tooling

- Repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Branch: `codex/duca-cellcf-evidence-20260717`.
- Exact commit: `2a0f848f7dbf17b7bcb40aa7a996954e8f87c4de`.
- Clean Linux snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_evidence_8327c2f_20260717`.
- Exact remote verification: `303 passed, 3 skipped`; clean tree before and
  after; `REMOTE_EXACT_VALIDATION_PASS`.
- Scope: profile-aware prepare/submit contracts for `exposure132` and
  `official60`, convergence inspection, raw full-stack timing, raw Slurm
  allocation replay, exclusive evidence publication and break-even inputs.
- Evidence boundary: implementation/test source only. It does not replace the
  formal model commit `1642f26` and contains no terminal mAP or measured paper
  cost result.

## 2026-07-16 DUCA-CellCF evidence-DAG replacement

- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Branch: `codex/duca-cellcf-20260716`.
- Exact commit: `3a0f5ae54d1dbd23ff170cda8a4706f5ed0d38d3`.
- Independent reviewer task:
  `019f6af9-7f66-7ea2-9bd8-38cfb75b92c8`, read-only `gpt-5.6-sol/max`.
- Verdict sequence: `HOLD` for unreachable completed-job accounting fallback,
  then `GO_TO_EXACT_COMMIT_GATE` after target-cluster reproduction and repair.
- Evidence scope: local implementation/contracts only; no mAP, latency,
  energy, memory or paper claim.

## 2026-07-16 DUCA-CellCF local implementation

- Worktree: `E:/DeskTop/TAD/OpenTAD_DUCA_TransitionOnly_20260711`.
- Branch: `codex/duca-cellcf-20260716`.
- Base: `7525efb2e07214615a59c482443246174a6adaf1`.
- Exact implementation commit:
  `475634e1be4a77ad1d9bc6bcf5f4bed04c3d6f31` (pushed).
- Permanent URL:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/475634e1be4a77ad1d9bc6bcf5f4bed04c3d6f31`.
- Local checks: CellCF contracts `27 passed, 3 Windows-only skips`; required
  C3 regressions `23 passed`; Python compile and CellCF shell syntax passed.
- Independent audit agent: `019f6ab3-516f-7171-9a32-fd5dc9a1748c`.
- This entry is implementation provenance only; no CUDA, mAP, cost or claim
  evidence is registered yet.

## 2026-07-16 DUCA-CellCF remote gate evidence

- Clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_475634e_20260716`.
- Clean Linux focused checks: `62 passed in 54.78s`.
- Synthetic gate:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_475634e_synth_debug_20260716_201424/synthetic_gate.json`.
- Synthetic gate SHA-256:
  `ada3a32faaa496924a867ee616309ef06c5c3b653135b828f03107ac9ec7519c`.
- Synthetic evidence: clean exact commit, L=1..768 geometry, K=384/T=768
  one-per-cell contract, step-zero exact uniform, distinct-cell incidence
  `AA^T=2I`, signed descent direction and connected zero utility all passed.
- First real-loader submission: Job `1167135`, failed before Python in one
  second because `/etc/profile.d/Z97-byobu.sh` read unset `LC_BYOBU` under
  nounset. This is deployment-bootstrap evidence, not a model/gate result.
- First environment repair Job `1167140` failed closed on the stale fallback
  THUMOS path because canonical dataset variables were not exported.
- Passed real THUMOS loader CUDA gate: Job `1167145`, run root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_475634e_real_gate_envfix_20260716_202103`.
- Real gate artifact SHA-256:
  `e0f762fb1387fc823ca1b8ab5b2c291052897b24a75f712d1b6ba9e810b6d7f3`.
- Gate facts: `full/mixed/all_short` all executed from the real loader;
  local utility had three positive and one negative value; forced overflow
  replayed the same batch; optimizer, EMA, scheduler and DUCA schedule each
  advanced exactly once; all six trainable parameter groups saw gradients;
  maximum CUDA allocation was about 8296 MiB.
- Passed three-arm DDP pilot: Job `1167146`, run root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_475634e_ddp_pilot_20260716_202319`.
- Pilot artifact SHA-256:
  `1c180572683e5dafea00cea7364253b1a5fcc7a24b1916d34642c831de7929c0`.
- Pilot facts: all three arms completed ten successful updates and one forced
  overflow replay; CellCF had informative utility in nine steps and an allowed
  zero-candidate short-window step.
- Independent deployment audit agent:
  `019f6ade-319b-79a3-ae17-754c0795fcdc`, verdict `HOLD` with five P1
  handoff findings (seed/job hash, terminal reopening, checkpoint-bound cost,
  mandatory cost DAG and idempotent receipts).
- Current scope: no full training was submitted. There is no mAP, cost or
  paper-readiness result; the repaired commit must repeat the evidence DAG.

## 2026-07-16 DUCA two-round Pro review prompts

- Audit target: GitHub commit
  `7525efb2e07214615a59c482443246174a6adaf1` on
  `codex/duca-transition-only-20260711`.
- Permanent commit URL:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/7525efb2e07214615a59c482443246174a6adaf1`.
- Round 1 code/math/training-contract audit:
  `docs/methods/prompts/2026-07-16-7525efb-duca-pro-round1-code-math-audit.md`.
- Round 2 final-method/experiment/publication verdict:
  `docs/methods/prompts/2026-07-16-7525efb-duca-pro-round2-method-paper-verdict.md`.
- Round 2 is invalid without the exact Round 1 `HANDOFF_PACKET`; the split is
  a reasoning-window control and creates no new empirical evidence.

## 2026-07-16 DUCA `7525efb` Round-1 Pro review

- Source attachment:
  `C:/Users/skywalker/.codex/attachments/52c99bc3-7775-40d9-91e1-df9f6c819e2b/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-16-7525efb-duca-round1-code-math-pro-review-raw.txt`.
- Structured absorption:
  `docs/methods/2026-07-16-7525efb-duca-round1-code-math-pro-review-absorption.md`.
- SHA-256:
  `DA4201C2D947C81EE6A799EF8B4572AD3D9C11DF047E29F8B70A9462B475F4C1`.
- Audit target:
  `7525efb2e07214615a59c482443246174a6adaf1`.
- Reviewer verdict: `GO_TO_REAL_GATE`; no static P0 model blocker, while
  real-loader CUDA, AMP/DDP, pilot, mAP and cost evidence remain absent.
- Project absorption: `ACCEPT_WITH_SCOPE / GO_TO_REAL_GATE_ONLY`. This source
  changes no empirical claim status.

## 2026-07-16 DUCA `7525efb` Round-2 method/paper Pro verdict

- Source attachment:
  `C:/Users/skywalker/.codex/attachments/d256ace7-867b-4ce0-a999-ab9bb3bae56e/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-16-7525efb-duca-round2-method-paper-pro-verdict-raw.txt`.
- Structured absorption:
  `docs/methods/2026-07-16-7525efb-duca-round2-method-paper-pro-verdict-absorption.md`.
- SHA-256:
  `B4415ABA4B7B779257DF0F0D4E107586181C4DCBCBFF9F8B38BADB156A191E0B`.
- Reviewer verdict: `REDESIGN`; sole route is exact-uniform-anchored Local-cell
  DUCA with detached local hard-flip utility.
- Project verdict: `PARTIAL_ACCEPT / ACCEPT_CORE_LOCAL_CELL_REDESIGN`.
  CellCF is `designed`; no implementation or empirical status changes.
- Literature correction: TAPS is Dinai et al., ACCV 2024 temporal-attention
  pruning/scaling; the attachment's TAPOS citation is a different paper.

## 2026-07-15 DUCA signed score-space repair sources

- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Branch: `codex/duca-transition-only-20260711`.
- Exact commit:
  `7525efb2e07214615a59c482443246174a6adaf1`.
- Permanent commit URL:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/7525efb2e07214615a59c482443246174a6adaf1`.
- Clean remote focused verification: `160 passed, 7 skipped` in scratch
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_noop_diag_20260715_2219`.
- Independent read-only auditor:
  `019f6620-3469-79d1-84b3-7b8bbd42b2d1`; mathematical direction passed,
  pre-fix verdict `HOLD` on AMP/exact-commit/real-loader and audit gaps.
- Exact gate Job `1165646` on predecessor `a6903ae` failed the utility-direction
  gate. Jobs `1165650` (shell failure) and `1165654` (canceled) are diagnostic
  deployment artifacts, not experiment evidence.
- No CUDA gate, real THUMOS sample gate, pilot, mAP or cost result exists for
  `7525efb`; this registry entry is implementation/test provenance only.
- Exact-commit Pro discussion prompt:
  `docs/methods/prompts/2026-07-15-7525efb-duca-signed-utility-exact-commit-pro-audit-prompt.md`.

## 2026-07-15 DUCA successful-update implementation sources

- Branch: `codex/duca-transition-only-20260711`.
- Exact commit: `a6903ae036d7b4bfd0c25752c51f020b20427fff`.
- Local verification: `80 passed, 3 skipped`; compilation, shell syntax, and
  diff checks passed.
- Read-only auditors: `019f631d-1917-7f13-b982-6b433b2b3924` and
  `019f6603-d9b2-7e20-8a26-57739fa78561`; final narrow verdict `GO`.
- This is implementation evidence only, not CUDA, mAP, cost, or claim evidence.

## S1 / DUCA exact-commit Pro audit (2026-07-15)

- Original attachment:
  `C:/Users/skywalker/.codex/attachments/69a2a56a-019c-43d1-9063-a2333ce34faa/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-15-35204f5-043be401-s1-duca-pro-audit-raw.txt`.
- Raw/archive SHA-256:
  `AC54C5B633DC9FD0CD801B2B12B2C4E44114E16B7569C220B77528674E2D04E2`.
- Structured absorption:
  `docs/methods/2026-07-15-35204f5-043be401-s1-duca-pro-audit-absorption.md`.
- Audited commits: Spatial Zoom S1 `35204f58`; DUCA `043be401`.
- Reviewer verdict: `STOP_AND_FIX`. Independent code inspection confirmed the
  major findings. A same-turn remote audit further showed that DUCA formal Jobs
  `1164700-1164703` already lag the declared successful-update schedule and
  that all nine running S1 cells emit nondeterministic CUDA interpolation
  warnings. The review's patches and numerical thresholds remain proposed
  specifications, not implemented or empirically supported facts.

## DUCA `043be401` exact-commit Pro audit (2026-07-15)

- Original attachment:
  `C:/Users/skywalker/.codex/attachments/48c9c615-e001-40cb-8207-951cb504198f/pasted-text.txt`.
- Exact raw archive:
  `docs/methods/reviews/2026-07-15-043be401-duca-exact-commit-pro-audit-review-raw.txt`.
- Structured absorption:
  `docs/methods/2026-07-15-043be401-duca-exact-commit-pro-audit-review-absorption.md`.
- SHA-256:
  `1D395844396D644295BF83BF08753C14B2E638295B8C37D15048924B0F415FC9`.
- Audit target: `043be401ba2b694342dc395f263e9a9858628d69`.
- Reviewer visibility: `VISIBLE_WITH_EXTERNAL_LIMITS`; it read repository code
  but not remote Slurm artifacts, the external ASFormer bytes, the VideoMAE
  checkpoint, or current mAP.
- Reviewer verdict: continue the four seed-0 jobs (`GO`) while holding paper
  claims. It found no confirmed P0, but reported an unsealed test-set
  checkpoint policy and the absence of a no-op anchor in counterfactual
  ranking.
- Project absorption: `PARTIAL_ACCEPT`. The code facts and evidence discipline
  are accepted; a no-op categorical loss is not treated as the unique solution,
  two-rank DDP is not required for the current one-GPU protocol, and a
  physical-time head remains conditional on a fixed-selection geometry
  diagnostic. Its GO-to-complete judgment was later superseded by live evidence
  of missing successful optimizer updates in all four formal arms; the jobs are
  now diagnostic-only and this review changes no empirical claim status.
- Primary-result protocol artifact:
  `docs/methods/2026-07-15-duca-043be401-primary-result-protocol.json`, SHA-256
  `AAC0FCA8671AE6F58CF4C9B5D4D40282BE714AA354028246E86504FD39C89B48`.
  It was declared before any evaluation and copied read-only to the formal run
  root and all four variant log directories. It fixes final one-based epoch 132
  `state_dict_ema` as the only primary result; intermediate test mAP is
  diagnostic only.

## Spatial Zoom S1 independent Max review (2026-07-13)

- Reviewer agent: `019f5b3c-9b7f-73f2-8eea-00157a60a119`.
- Model/effort: `gpt-5.6-sol` / `max`.
- Review mode: independent, read-only code and protocol audit.
- Local archive:
  `docs/methods/reviews/2026-07-13-spatial-zoom-s1-independent-max-review.md`.
- Round-1 verdict: `FAIL_BEFORE_REMOTE_TRAINING`; remediation implemented.
- Round-2 verdict: `FAIL_BEFORE_REMOTE_TRAINING`; five P1 findings remediated.
- Round-3 verdict: `FAIL_BEFORE_REMOTE_TRAINING`; five provenance P1 findings
  remediated.
- Later adversarial rounds found checkpoint-writer schema drift, mutable
  experiment-level test locking, unsealed final reports, and unenforced profile
  order. Each was remediated and regression-tested.
- Final verdict after stable study locking and pre-test identity checks:
  `PASS_BEFORE_REMOTE_TRAINING`; P0/P1/P2 all empty.
- Evidence boundary: local `46` focused / `26` S1 tests only. Formal CUDA,
  full train, sealed test, cost, and S1 GO remain absent.

## Dense-Time Spatial Zoom Pro review (2026-07-13)

- Original attachment:
  `C:/Users/skywalker/.codex/attachments/5f8506a8-5538-4099-9074-a633872a238b/pasted-text.txt`.
- Exact raw archive:
  `docs/methods/reviews/2026-07-13-dense-time-spatial-zoom-pro-review-raw.txt`.
- Structured absorption:
  `docs/methods/2026-07-13-dense-time-spatial-zoom-pro-review-absorption.md`.
- SHA-256:
  `667A319CA2ABB0601EE0D6A76DF9D8D139D1F116A7BD93D55B48CFA2DC655650`.
- Archive is byte-identical to the 84,533-byte attachment.
- Reviewer verdict: `HOLD`; only S1/S2 falsification is recommended before a
  learned model. Project absorption is `PARTIAL_ACCEPT / HOLD_FULL_MODEL`:
  S1 infrastructure is authorized, S2 and DART-Zoom remain gated.

## DUCA Oracle-gap and reachability audit (2026-07-13)

- Audit node: `experiments/duca-oracle-gap-reachability-audit.md`.
- Current method-code target:
  `0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d`; documentation/diagnostic HEAD
  `84bcb2b62684688315365066c566a7a6a8b695fc`.
- Oracle Job `1001959`, completed exit 0; raw log:
  `/data/home/sczc063/run/yuzibo/OpenTAD_Back_check/logs/oracle_boundary_adapter_repro_20260604_1001959.out`.
- Oracle implementation source:
  `E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/opentad/datasets/transforms/end_to_end.py`,
  especially `_build_frame_oracle_groups`, `_select_oracle_positions`, and
  `_remap_gt_to_selected_axis`.
- Historical full-frame record:
  `E:/DeskTop/TAD/temrefuse-tad/evaluation_data.csv`, epoch 55, Avg-mAP 68.97,
  IoU-wise 83.42/79.41/72.20/62.39/47.46.
- Primary literature used for reachability context:
  BasicTAD (`https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_An_Empirical_Study_of_End-to-End_Temporal_Action_Detection_CVPR_2022_paper.pdf`),
  KTS adaptive tokenizer (`https://openaccess.thecvf.com/content/ICCV2023W/RCV/papers/Afham_Revisiting_Kernel_Temporal_Segmentation_as_an_Adaptive_Tokenizer_for_Long-form_ICCVW_2023_paper.pdf`),
  ResidualViT (`https://openaccess.thecvf.com/content/ICCV2025/papers/Soldan_ResidualViT_for_Efficient_Temporally_Dense_Video_Encoding_ICCV_2025_paper.pdf`),
  and Progressive Block Drop (`https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Temporal_Action_Detection_Model_Compression_by_Progressive_Block_Drop_CVPR_2025_paper.pdf`).

## DUCA selection-quality diagnostic evidence (2026-07-13)

- Remote Job: `1161079`, completed exit 0.
- Model commit/checkpoint: `8bfc0e549434591b9bf1a9cd5563deb0da388f92`,
  EMA epoch 89, checkpoint SHA-256
  `8ea8e5c7a53ba159285ec244ecd34b7e79f86cec89acca965351f7a4bd869749`.
- Remote run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_selection_quality_8bfc0e5_epoch89_20260713`.
- Local artifact root:
  `E:/DeskTop/TAD/analysis_outputs/duca_selection_quality_20260713/output`.
- Structured record: `experiments/duca-selection-quality-epoch89.md`.
- Evidence status: `tested/diagnostic`; legacy homotopy invalidated, no paper
  claim and no transfer to corrected commit `0ea4e15`.

## DUCA `0ea4e15` Pro audit prompt (2026-07-12)

- GitHub branch head: `b38080d` (documentation-only final diagnostic update).
- Method-code audit target: `0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d`.
- The head commit adds documentation only; it does not change model code or
  experiment protocol.
- Prompt artifact:
  `docs/methods/prompts/2026-07-12-duca-transition-only-0ea4e15-pro-audit-prompt.md`.
- GitHub prompt URL:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/b38080d/docs/methods/prompts/2026-07-12-duca-transition-only-0ea4e15-pro-audit-prompt.md`.
- Scope: exact-commit code visibility, uniform/homotopy invalidation,
  protected-gradient ownership, structured surrogate alignment, selected-axis
  detector geometry, total-cost accounting, publication claims, three
  mutually exclusive redesign routes, and implementation-level final verdict.

## DUCA exact-uniform audit evidence (2026-07-12)

- Defective implementation commit: `8bfc0e549434591b9bf1a9cd5563deb0da388f92`.
- Corrective implementation commit: `0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d`.
- Local DUCA worktree:
  `E:/DeskTop/TAD/OpenTAD_DUCA_TransitionOnly_20260711`.
- Remote focused verification copy:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_uniform_fix_test_20260712`;
  result `26 passed, 2 skipped`.
- Invalid control: Job `1159414`, run root listed below. Its best 55.67 is a
  DP tie-break diagnostic, not exact-uniform.
- Historical native stride-2 uniform source: Job `1150701`, log
  `/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702/logs/slurm_adapter_matched_diag/adapter_stride2_uniform_dense65-1150701.out`;
  best Avg-mAP 64.352, IoU-wise 79.40/74.57/67.98/57.17/42.64.
- Historical grid-aware uniform source: Job `1150842`, log
  `/data/run01/sczc063/yuzibo/OpenTAD_SparseHeadClean_20260702/logs/slurm_adapter_matched_diag/adapter_uniform_gridaware_densehead-1150842.out`;
  best Avg-mAP 65.696, IoU-wise 80.88/76.62/68.50/58.43/44.05.
- The two historical anchors are auditable but protocol-unmatched to the current
  P0 matrix; they must not be copied into its main table.

## DUCA transition-only P0 remote evidence (2026-07-12)

- Commit: `8bfc0e549434591b9bf1a9cd5563deb0da388f92`
- Formal gate: Job `1159395`
- P0 jobs: invalid alpha0 control `1159414`, direct-a5 `1159415`, transition beta=0
  `1159416`, transition beta=0.25 `1159417`
- Run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_transition_8bfc0e5_p0_20260711_2317`
- Evaluation sources: each variant's `train.out`; asynchronous evaluations were
  captured through completion on 2026-07-13 01:08 +0800. All jobs completed
  with exit code 0; the matrix remains protocol-invalidated by the exact-uniform
  audit above.
- Result-to-claim trace:
  `.aris/traces/result-to-claim/2026-07-13_run01/`. Reviewer delegation was
  unavailable, so the local `C3=no`, `C4=no`, high-confidence verdict is marked
  pending external review.

# 来源注册表

## Dense-Time Spatial Zoom Pro discussion prompt (2026-07-13)

- Prompt artifact:
  `docs/methods/prompts/2026-07-13-dense-time-spatial-zoom-pro-discussion-prompt.md`.
- Public repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Public code anchor embedded in the prompt:
  `1f5f7254a390f183121e6c4b7cebcebd2f2954d1`.
- Scope: skill-loading certificate, primary-source novelty audit, repository
  visibility, dense-resolution and oracle-ROI kill gates, four mutually
  exclusive spatial-compute routes, one implementation-ready conditional
  design, strict full-stack cost, result-to-claim matrix, and mock top-tier
  review. This is a discussion artifact, not an implementation or experiment.

## Codex Tasks

| 来源 | 覆盖范围 | 本地归档 |
|---|---|---|
| `019f49d2-a7ef-7273-b420-8732fae46bf8` | DUCA 主讨论，191 轮，158 条用户消息 | [完整用户记录](sources/thread-019f49d2-user-record.md) |
| `019f20d8-5e8d-72d3-a2dc-898b75ce03ea` | 目标、实现、部署代理 | [近期记录](sources/delegated-thread-recent-record.md) |
| `019f3cd2-30cd-7452-a210-1ef9fd53fd14` | 论文写作代理 | [近期记录](sources/delegated-thread-recent-record.md) |
| `019f4066-8bd9-73f0-9af5-30dc9da45cce` | 早期目标梳理 | [近期记录](sources/delegated-thread-recent-record.md) |
| `019f4ae5-93dd-7381-8203-42360125b41b` | ChronoTransport 提出、MoD/粒度/在线纠错、实现部署、正式 Stage-B 与本轮查新 | [讨论与决策记录](sources/thread-019f4ae5-decision-record.md) |

## 代码来源

- C3 clean repo：`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`
- DUCA/GASVT 审计 worktree：`E:/DeskTop/TAD/OpenTAD_GASVT_CostAudit_20260710`
- GitHub：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- DUCA branch：`codex/gas-vt-stage23-detector-aware-20260706`
- 当前审计 commit：`a5e1774b9941312569ca645341da1abad339db61`
- 当前正式训练 commit：`70aa069b895322c2307ffbb13dfdef9fac0d1305`

## 关键评审与吸收记录

以下记录是主要二级来源；原始 raw review 仍保存在各 repo 的
`docs/methods/reviews/`：

- `gas-vt-stage01-53124a2-review-absorption.md`
- `2026-07-07-c69c1a0-paction-gasvt-hold-review-absorption.md`
- `2026-07-07-46cacc1-pro-final-route-review-absorption.md`
- `2026-07-08-fbea37b-learned-context-radius-hold-review-absorption.md`
- `duca-plugin-final-method-review-absorption.md`
- `duca-online-plugin-final-design-review-absorption.md`
- `duca-online-plugin-603ed02-real-detector-review-absorption.md`
- `2026-07-09-544eca6-duca-transition-first-critical-review-absorption.md`
- `2026-07-09-7bea4fc-duca-hold-paper-claim-review-absorption.md`
- `2026-07-10-88e50b1-duca-final-method-audit-review-absorption.md`
- `2026-07-10-70aa069-researchclaw-duca-divergent-audit-absorption.md`
- `2026-07-10-duca-official-adatad-structural-audit.md`
- `2026-07-11-a5e1774-duca-transition-only-pro-review-absorption.md`

评审建议不是实验事实。wiki 中只有带 run root、Job ID、日志或 result artifact 的内容
才可标记为实验事实。

### ChronoTransport CT-P3R-3S Pro review

- 原始 review：
  [完整原文](sources/2026-07-11-chronotransport-ct-p3r-3s-pro-review-raw.md)
- 结构化吸收：
  [吸收记录](sources/2026-07-11-chronotransport-ct-p3r-3s-pro-review-absorption.md)
- 原附件 SHA-256：
  `E7971A22044B384092B833A1137F8EC0B543B504D271078CBCB4198F96D35CAF`
- 用户提供的三个附件 ID `9fb7a806...`、`2ca84118...`、`63d3e02e...` 字节完全相同，
  已按 SHA 去重归档。
- review 裁决为 `REVISE_SPEC_BEFORE_CODE`，属于规格/代码审计，不是实验结果。
- review 中引用的 amendments、generic module、tests、patch、README 仅有 sandbox 链接
  与 SHA；文件未随附件提供，不能登记为已获得、已运行或已集成代码。
- reviewer 没有访问本地 `E:\...` 工作树；其本地代码风险必须重新核验。
- 本地源码复核：
  [CT-P3R-3S-r1 local source audit](sources/2026-07-11-chronotransport-r1-local-source-audit.md)
  （commit `375094d` 的静态审计；不是 GPU 行为或实验结果）。
- 独立 agent 复核：
  [ChronoTransport r1 independent review](sources/2026-07-11-chronotransport-r1-independent-agent-review.md)
  （空白对话 fork，先 sealed provisional review、再与本地审计对照；无 GPU 实验）。
- 用户批准后的 r1 书面规格：
  `docs/superpowers/specs/2026-07-11-chronotransport-ct-p3r-3s-r1-design.md`，commit
  `02199f8`，SHA-256
  `871420261BD1C19CC515218A6016A91ED7D553B73740AB41C2E02AA7F96609F9`；当前待用户最终复核。
- 第二个空白上下文 reviewer 对书面规格的独立审查：
  [02199f8 independent written-spec review](sources/2026-07-12-chronotransport-r1-spec-independent-agent-review.md)，
  verdict=`REVISE_SPEC_BEFORE_PLAN`；无 GPU 实验、未修改规格。

## 远端实验来源

- PhysTime-AdaTAD matched K=384 jobs: `1159491-1159495`, commit
  `3ac93a12c299012db64513567d5bdedf0c6d5f71`.
- PhysTime-AdaTAD run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_adatad_3ac93a1_k384_final_20260712_023243_+0800`.
- Independent result-to-claim trace:
  `.aris/traces/result-to-claim/2026-07-12_run01/trace.md`.

- DUCA 70aa fixed-384 Job：`1154971`
- Run root：`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_70aa069_final_20260710_1544`
- a5e cost smoke Job：`1156079`
- Cost smoke root：`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cost_profile_smoke_20260710_1652`

## 外部文献来源

以下来源用于 2026-07-11 的新颖性与前沿路线复核，均为论文官方开放页面：

- AdaFrame, CVPR 2019：自适应逐视频选帧与 utility/policy-gradient 训练。
  `https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html`
- Action Sensitivity Learning, ICCV 2023：学习帧价值并重加权 TAL 子任务梯度。
  `https://openaccess.thecvf.com/content/ICCV2023/html/Shao_Action_Sensitivity_Learning_for_Temporal_Action_Localization_ICCV_2023_paper.html`
- AdaTAD, CVPR 2024：长视频端到端 TAD 与 temporal-informative adapter。
  `https://openaccess.thecvf.com/content/CVPR2024/html/Liu_End-to-End_Temporal_Action_Detection_with_1B_Parameters_Across_1000_Frames_CVPR_2024_paper.html`
- TE-TAD, CVPR 2024：物理时间对齐坐标和随视频长度变化的 query 数量。
  `https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html`
- TAPS, ACCV 2024：利用全时间上下文进行逐层动态 filter pruning/scaling。
  `https://openaccess.thecvf.com/content/ACCV2024/html/Dinai_TAPS_Temporal_Attention-based_Pruning_and_Scaling_for_Efficient_Video_Action_ACCV_2024_paper.html`
- Progressive Block Drop, CVPR 2025：面向 TAD 的硬件友好 block dropping 与性能恢复。
  `https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Temporal_Action_Detection_Model_Compression_by_Progressive_Block_Drop_CVPR_2025_paper.html`
- Mixture-of-Depths, arXiv 2024：token×layer 固定容量动态计算。
  `https://arxiv.org/abs/2404.02258`
- Eventful Transformers, arXiv 2023：只重算随时间显著变化的 video tokens。
  `https://arxiv.org/abs/2308.13494`
- ResidualViT, ICCV 2025：dense temporal encoding 的 residual path 与 token reduction。
  `https://arxiv.org/abs/2509.13255`
- Adaptive Temporal Refinement, arXiv 2025：TAD 边界驱动的 continuous depth allocation。
  `https://arxiv.org/abs/2511.03943`
- SCOPE, arXiv 2026：`cache/predict/recompute` 三模视频计算和稳定控制。
  `https://arxiv.org/abs/2604.02979`
- Conformal Thinking, ICML 2026：风险约束下的自适应推理预算。
  `https://arxiv.org/abs/2602.03814`
- Uni-AdaFocus, TPAMI/arXiv 2024：低分辨率全局观察、动态 patch 定位与高容量局部分支，
  并扩展时间和样本级动态计算。
  `https://arxiv.org/abs/2412.11228`
  `https://github.com/LeapLabTHU/Uni-AdaFocus`
- AdaSpot, CVPR 2026：面向 Precise Event Spotting 的低分辨率全局特征、training-free
  task-aware saliency ROI 与时空平滑高分辨率局部分支。
  `https://openaccess.thecvf.com/content/CVPR2026/html/Xarles_AdaSpot_Spend_Resolution_Where_It_Matters_for_Precise_Event_Spotting_CVPR_2026_paper.html`
  `https://arxiv.org/abs/2602.22073`
  `https://github.com/arturxe2/AdaSpot`

本轮 ChronoTransport 查新的检索 claims、独立 reviewer route 与最终裁决保存在
`.aris/traces/novelty-check/2026-07-11_run01/trace.md`。新颖性评分属于审计判断，
不是论文事实。

## 外部附件

主任务显式引用的关键附件包括：`5f9a0d62...`、`86b473c6...`、
`d0087ae1...`、`1705e957...`、`60cb3e7a...`、`0ce290f9...`、
`a885a659...`、`391f061f...`、`c2008dfb...`、`c8a36eba...`、
`1fc36774...`、`d8b9f9fc...`。其中最后一份 ResearchClaw 审查已原样归档，
SHA256 为 `E4344DAED297F02E23CE355A4B0BBA1845F2C05393820CE04843374AAB6A59AC`。

2026-07-11 的两个附件 `258e6bbc...` 与 `bca8f4a3...` 字节完全一致，均为 DUCA
joint-ASFormer/transition-only Pro 审查。原文归档为
`docs/methods/reviews/2026-07-11-a5e1774-duca-transition-only-pro-review-raw.txt`，SHA256 为
`011EBB67CC52D943248D18E4638E2220763DED44329BEF8EB78DBD77973BE863`；吸收记录为
`docs/methods/2026-07-11-a5e1774-duca-transition-only-pro-review-absorption.md`。

2026-07-13 附件 `6065c548...` 是固定到代码提交 `0ea4e15` 的 DUCA exact-commit Pro
审计。原附件 SHA256 为
`60D4D9414F3F2D90EC9A0CE0F2D704D2184D8EEED9CE2FBB5315932997CEE957`；原文归档为
`docs/methods/reviews/2026-07-13-0ea4e15-duca-fsu-pro-audit-review-raw.txt`，吸收记录为
`docs/methods/2026-07-13-0ea4e15-duca-fsu-pro-audit-review-absorption.md`。reviewer verdict
为 `HOLD`；DUCA-FSU、数值门槛和代码片段均为建议，不是已实现或实验事实。

2026-07-13 附件 `eb295612-1df9-44b8-9c98-6fd813d0552c` 是固定到 GitHub 提交
`1fc7037358e1141f7555ad87d1edd9128ce2e6a5` 的 DUCA selection-quality/architecture
Pro 审查。原附件与归档 SHA-256 为
`DDBC15BC20BFDD503FAA2DA4832093325EB2D8997E4A685638DFF46F90CC780D`；原文归档为
`docs/methods/reviews/2026-07-13-1fc7037-duca-cellcf-redesign-pro-review-raw.txt`，吸收记录为
`docs/methods/2026-07-13-1fc7037-duca-cellcf-redesign-pro-review-absorption.md`。reviewer
verdict=`REDESIGN`，推荐 DUCA-CellCF；其数学目标、patch、schedule、阈值和 gates 均为
设计建议，不是已运行代码或实验事实。审查还纠正既有 diagnostic：所谓 raw transition
实际为 `abs_delta_p_action + uncertainty_peak` compound proxy。

## 2026-07-15 DUCA exact-gate / pilot repair evidence

- GitHub branch: `codex/duca-transition-only-20260711`.
- Current exact commit: `043be401ba2b694342dc395f263e9a9858628d69`.
- Clean remote snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_043be40_20260715`.
- Clean Linux focused result: `122 passed, 5 skipped`.
- Stale successful gate for AMP-cache fix: Job `1164279`, root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_18dc1cd_exact_gate2_20260715_142105_+0800`.
- Invalidated pilot diagnostic: Job `1164286`, root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_18dc1cd_ddp_pilot_20260715_143003_+0800`.
- Exact gate: Job `1164318` (`COMPLETED/0`), root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_043be40_exact_gate_20260715_1500`.
- Replacement four-arm pilot: Job `1164319` (`COMPLETED/0`), root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_043be40_ddp_pilot_20260715_1500`.
- Formal matched seed-0 suite: Jobs `1164700-1164703`, root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_043be40_p0_formal_seed0_20260715_1647`.
- Independent reviewer task `019f631d-1917-7f13-b982-6b433b2b3924` audited
  exact commit `043be401` read-only and returned GO with P0=0/P1=0. This is
  implementation-review evidence, not empirical performance evidence.

## 2026-07-16 PhysTime G1b SDPQ revise-before-full-train review

- Source attachment:
  `C:\Users\skywalker\.codex\attachments\3a385d8f-5028-4961-a3f2-e8e228a25b21\pasted-text.txt`.
- Archived raw review:
  `docs/methods/reviews/2026-07-16-372fcbf-phystime-g1b-sdpq-revise-before-full-train-raw.txt`.
- SHA-256:
  `E3389D57F179BB4FFD6C1F25AC24FF1321C7865E1EBEF80BC02EF2A4E59368AF`.
- Scope: GitHub commit `372fcbf58d1b2eb895b724f6f040458bde4d636e`,
  PhysTime G1b SDPQ head, native J192/K384 metadata, gate/pilot evidence.
- Verdict: `REVISE-BEFORE-FULL-TRAIN`; engineering runnable/gate-passed, not
  empirically supported and not paper-ready.

## 2026-07-16 DUCA-CellCF replacement gate evidence

- GitHub branch: `codex/duca-cellcf-20260716`.
- Exact commit: `3a0f5ae54d1dbd23ff170cda8a4706f5ed0d38d3`.
- Clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_3a0f5ae_20260716`.
- Synthetic gate:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_3a0f5ae_synth_20260716_2115/synthetic_gate.json`,
  SHA-256 `1d8234e9c186e1726f08fa59cd892c63f02fd9a47d498d8afe304f05fed9adad`.
- Valid real-loader CUDA gate: Job `1167222`, artifact
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_3a0f5ae_real_gate_envfix_20260716_2126/real_loader_gate.json`,
  SHA-256 `b128f58755dcf6bd924fad60d8a84c02a1f149ec0fa70119b836fb61be0c4334`.
- Immutable failed deployment diagnostics: Jobs `1167220` (interpreter not at
  byte zero) and `1167221` (canonical dataset root not sourced).
- Exact pilot: Job `1167227`, run root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_3a0f5ae_ddp_pilot_20260716_2130`.
- Pilot result: `COMPLETED/0`, artifact SHA-256
  `f199f4dc14aeef8c03ad91838e31281f93bd551ab4a45c01284e48d4aa3d8085`.
- Invalid formal root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_3a0f5ae_formal_seed0_20260716_2138`.
- Invalid/cancelled arm Jobs: `1167234`, `1167235`, `1167236`; no aggregate,
  cost or completion Slurm job was created. Null dependent receipts are
  deployment-failure artifacts, not evidence.
- Exact transaction-fix commit:
  `b8cd29f621d410b720f12380b3095dd39574e01f`.
- Independent final review task:
  `019f6b35-56d5-7043-a9d0-3854a4f6d018`, verdict `GO`, P0/P1=0.
- Clean accelerated snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_b8cd29f_20260716`;
  Linux result 155 passed/3 skipped plus 23 C3 regressions.
## 2026-07-16 DUCA-CellCF `b8cd29f` exact gates

- Synthetic gate artifact:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_b8cd29f_synth_20260716_2250/synthetic_gate.json`,
  SHA-256
  `9606f6325e05767e7b748b85e73352cdc52a439b382541a4dd5ef66ca855a76f`.
- Wrapper diagnostic Job `1167338`: `FAILED/127:0` before Python because the
  batch shell did not expose `module`; no model evidence.
- Valid real-loader CUDA gate Job `1167345`: `COMPLETED/0:0`, artifact
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_b8cd29f_real_gate_envfix_20260716_2335/real_loader_gate.json`,
  SHA-256
  `c4f6b5ce7d2bb830236ee51cef6d2b5ac5965bd4b84811a12cb2e86eb039b673`.
- Gate-bound DDP pilot Job `1167348` later completed; see the outcome entry
  below for the frozen artifact and SHA-256.

## 2026-07-16 DUCA-CellCF `b8cd29f` pilot/formal submission outcome

- DDP pilot Job `1167348`: `COMPLETED/0:0`, 4:38; artifact
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_b8cd29f_ddp_pilot_20260716_2345/duca_cellcf_ddp_pilot_suite.json`,
  SHA-256
  `572e47440c54da558f6320148549de8fd62204d0f524b410f53400fe02249270`.
- Invalid formal root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_b8cd29f_formal_seed0_20260716_2355`.
  Pending Jobs `1167359/1167360` had zero runtime and were cancelled after the
  receipt validator rejected an empty pending `sacct Comment`; exact tokens
  remained visible in `squeue Comment` and `sacct SubmitLine`. No downstream
  DAG jobs existed and no result is admissible.

## 2026-07-17 DUCA-CellCF current exact evidence

- Branch: `codex/duca-cellcf-20260716`; exact commit
  `1642f265e48391418a7c8a4a087e33e2b7bf6899`.
- Clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_1642f26_20260717`;
  Linux 212 passed/3 skipped plus 23 required C3 regressions.
- Synthetic gate:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_1642f26_synth_20260717_0145/synthetic_gate.json`,
  SHA-256 `3dd4750cc97d0287b647125264a5495626cb87df6aec6b099b4aed48a523e5cd`.
- Real-loader CUDA gate Job `1167479`, artifact
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_1642f26_real_gate_20260717_0150/real_loader_gate.json`,
  SHA-256 `3d630a323e79c694f663c31151c070fd46943296937ceafdd5f9bcacfcbd7cde`.
- DDP pilot Job `1167480`, artifact
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_1642f26_ddp_pilot_20260717_0150/duca_cellcf_ddp_pilot_suite.json`,
  SHA-256 `8e6a59e92f12b15ec1e7c3671104959c0533c9ba9b68dd36550c0294c8b48cd3`.
- Formal root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200`;
  Jobs `1167481-1167486` as recorded in the experiment node. Status:
  `experiment_running`; no terminal metric/cost artifact yet. Mutable progress
  evidence is the three `work_dirs/<variant>/gpu1_id0/duca_cellcf_training_audit.json`
  files plus checkpoint sidecars. At 12:45 CST these sealed 9,100/9,000/8,300
  successful updates through 91/90/83 completed epochs; they are progress
  evidence only and cannot replace terminal epoch-131 EMA aggregation.
- Invalid zero-runtime diagnostics: `4bf6485` Jobs `1167469-1167471` and
  `522925e` Jobs `1167475-1167478`; never use as result evidence.
- Independent final live-dependency review task
  `019f6ba6-d502-77c3-b585-7311fae38fe0`: GO, P0/P1=0.

## 2026-07-17 CellCF training-budget code evidence

- Exact code object:
  `1642f265e48391418a7c8a4a087e33e2b7bf6899`.
- `scripts/duca_cellcf_canonical_env.sh` binds
  `DUCA_OFFICIAL_ADATAD_END_EPOCH=132` and
  `DUCA_LOSS_SCHEDULE_TOTAL_STEPS=13200`.
- `scripts/run_duca_cellcf_variant.sh` binds the terminal evidence to
  `epoch_131.pth/state_dict_ema` and 13,200 successful updates.
- The inherited repository AdaTAD base
  `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py` has
  `workflow.end_epoch=60`. This establishes the protocol-length difference;
  it does not yet establish final measured GPU-hour overhead.
- Decision provenance: user training-cost concern plus local exact-commit code
  audit on 2026-07-17. Live throughput estimates are mutable diagnostics; the
  mandatory trained-checkpoint cost job remains the final cost source.

## 2026-07-17 CellCF post-run mutation-monitor gate

- Evidence code commit:
  `9e96967a158534b014aacde57c1b78bd1591e71a`.
- Superseded evidence commit: `787569e11b8944cf58e15e0e8643a82fde6e2991`;
  its target Linux suite failed two mutation tests and must not deploy the
  post-run DAG.
- Clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_evidence_9e96967_20260717`.
- Target-mount gate root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_evidence_9e96967_linux_gate_20260717_2050`.
- Filesystem evidence: `filesystem=fuseblk`.
- Finalizer gate: `14 passed`; output SHA-256
  `ef6449bff689529ececfaec4f63a562b21c7a6972c19ca20d4394ae4406d70e6`.
- Broad evidence gate: `253 passed`; output SHA-256
  `79deb192f7ef292f6b91f594700218a653ef245a7e05d8e204b0c7eb4f85b2e9`.
- Independent reviewer task `019f7009-e7fe-75c0-987d-f4ef9a03335e`
  returned code-level GO with no P0/P1 after two review rounds.

## 2026-07-18 CellCF CPU-enqueue schema repair

- Evidence branch: `codex/duca-cellcf-evidence-20260717`.
- Exact pushed commit:
  `4ce69c852bdbd902046b47bc6019ae11e850dbe4`.
- Clean remote snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_evidence_4ce69c8_20260718`.
- Local broad result: `259 passed, 10 skipped`.
- Exact Linux result: `279 passed in 113.22s`, plus py_compile, Bash syntax,
  clean-tree and exact-HEAD checks.
- Immutable wrapper diagnostic Job `1170932`: `FAILED/1:0`; profiler artifacts
  passed independent strict reconstruction, but the temporary verifier had a
  quote-stripped Python `SyntaxError`.
- Passing real GPU schema gate Job `1170940`: `COMPLETED/0:0`, 43 seconds.
- Gate root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_schema_gate_4ce69c8_20260718_110645_+0800`.
- Gate receipt `schema_gate.json` SHA-256:
  `f69bc872993fc778b2ceaf6b1a179721861aa57c176d0e12b5869a3913e14758`.
- Samples JSONL SHA-256:
  `ae3f3ac474f95349c620813618a983ed2937c95627569716521cc033f08fbfb3`.
- Summary JSON SHA-256:
  `e39fe74155bde9a28854110e270accb6697f813a2f2ec316a80853a53e56617b`.
- Raw-sample ordered/multiset fingerprints:
  `d6ed9e550224ee9621e5e94e5d3927e389d1d5a9890feb9f08b45ce85c31b142` /
  `dbc53206721c9d88e2d782499353ea9aa364107b4801a26f9031edf750d0a14d`.
- Evidence scope: producer/consumer schema compatibility only. No replacement
  formal 500-sample cost pair or C7 support exists.

## 2026-07-20 DUCA Allocation-Ceiling implementation and formal DAG

- GitHub branch:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-allocation-ceiling-20260720`.
- Exact current commit:
  `1d51379d5feb32c8dfb11ec9a2ef238f4c3f7bbe`.
- Clean remote snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_allocation_1d51379_20260720`.
- Verification: local `49 passed`; clean Linux `105 passed`; py_compile and
  Bash syntax pass.
- Full training-video metadata audit: 200 videos, decoded-minus-annotation
  frame-count error set `{0}`, maximum absolute error `0`; FPS-clock drift
  median about `3.00` frames and maximum `3.69` frames.
- Invalid first root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_allocation_b18dd8f_training_20260720_035519`;
  Job `1174706` `FAILED/1:0` on the superseded FPS-drift contract and
  `1174707-1174710` were cancelled without runtime.
- Current formal root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_allocation_1d51379_training_20260720_041247`.
- Current Jobs: `1174711` gate, `1174712` export, `1174713` diagnostics,
  `1174714` candidate detector loss, `1174715` completion.
- Gate `1174711`: `COMPLETED/0:0` in `00:02:18`; gate artifact SHA-256
  `34246ef45d6e4835e32e0e720dfec0017743928b4aaf098eb2ed6d3bd0e482d0`.
  Its exact solver replay, candidate-loss validation, solver-cost validation
  and scheduler validation all report `validation_passed=true`.
- After the gate, export Job `1174712` started on `g0048`; later jobs remain
  dependency-pending.
- Suite manifest SHA-256:
  `5cfa0112c3fe715618f845ea1732540803d1b322a240421cda079bd16f6b8d96`.
- Submission intent/token SHA-256:
  `7e499ad4045ca05b085dc4d6b0283323bab9aa0e23721e28d423a3c6402c5d20`.
- Submission JSON SHA-256:
  `36689e17fad7bf5c046babddb17cb0165960b5f81fd13877c36e29676d1deae0`.
- Pre-release scheduler snapshot SHA-256:
  `1b3a4f0157181173aad15a70222d53603c9d2180f0cf93ce354f188c20b4c87a`.
- Post-release scheduler receipt SHA-256:
  `3239f5e43e54c9b8e016ba7fb6c9fc9c86753ba4b8aff5b186a2d104cbefd2e3`.
- Evidence scope: `experiment_running`, training-side necessary-condition
  diagnostic only. Validation/test and selector training are not authorized.

## 2026-07-20 DUCA Allocation-Ceiling numerical-certificate repair

- Superseded chain outcome: Job `1174712` `COMPLETED/0:0` in `00:25:55`;
  Job `1174713` `FAILED/1:0` in `00:12:43`; Jobs `1174714/1174715`
  `CANCELLED` with zero runtime.
- Failure site: first GT32 sample `video_validation_0000983|1016`,
  `lex_block_0210_0240`.
- Recoverability output SHA-256:
  `e23fdf6ee72341c74944592abea57970de65a63b50975b6c77cdb2d7d9365968`.
- Recoverability summary SHA-256:
  `999b38bf7413b849d75a49c853bf933b3f995dfc669fc48409941c6863af848f`.
- GT32 input SHA-256:
  `3a1eb27ed20e91124aa876bfe65c14ae7baf4689033b79c4270d2887bcecbd78`.
- Replacement exact commit:
  `8ebdd2a11ea5cc0644979324872a3b1cae5a2170`.
- GitHub:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-allocation-ceiling-20260720`.
- Local verification: `55 passed`; independent audit `P0=0/P1=0/GO`.
- Clean Linux snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_allocation_8ebdd2a_20260720`;
  relevant verification `111 passed`.
- Valid old-failure replay Job: `1175393`, `COMPLETED/0:0`, `00:01:48`.
- Replay output/summary/validation SHA-256:
  `b877bfccf079c628cbdabe28012f1ca2b2f6be9d157ccdca4029ea20dce64b97`,
  `22425605c49e1b7cd9bc229a2c9437f3ee13bb342ba345c11a050799d5ad7294`,
  `19aeae245f4c926f212cf65858a8fbe19e5229b7a8ff63a9322b8fd34b81b0eb`.
- Replacement formal root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_allocation_8ebdd2a_training_20260720_1320`.
- Replacement Jobs: `1175395` gate, `1175396` export, `1175397`
  diagnostics, `1175398` candidate, `1175399` completion.
- Submission JSON SHA-256:
  `82237c255ec9e7e46941fdf634f1e556f968d256e437512499a82031be591491`.
- Scheduler receipt SHA-256:
  `25bb30d11e2a725bb393462eb8bfb9be819610bc9bbfeb7e18c50a2212f38514`.
- Replacement gate `1175395`: `COMPLETED/0:0` in `00:04:02`.
- Replacement gate artifact SHA-256:
  `6030d9fb7110aa7c73b2df244eff50136d1342c5e2e90bd86db485d38faafc61`.
- Gate contract: exact solver replay, candidate-loss, solver-cost, submission
  and scheduler validation all report `validation_passed=true`; export
  `1175396` is released and priority-pending.
- Evidence scope: `experiment_running`; complete hash-bound DAG evidence is
  still required.

## 2026-07-20 DUCA Allocation-Ceiling sealed negative result

- All replacement Jobs `1175395-1175399`: `COMPLETED/0:0`; all stderr files
  are empty.
- Final evidence SHA-256:
  `8232f2f0889bc5e0579abcf82d42ab4009397366c5c4b0e6bfd71d0c658ad6d6`.
- Full recoverability summary SHA-256:
  `1f7d61a96c4dd50116cdf432aa37ae9c0ba780545a8c77ae2572b0d20e52d37e`.
- GT32 ceiling summary SHA-256:
  `b887c3d6e114845506aa3654907d2c6eeab06a6f29be7e274fe55b50a2af1413`.
- Candidate detector-loss summary SHA-256:
  `09dd62d1b14b0f760900ce90cd27e3113c039625800035c1b8b75e8136a1609a`.
- Solver-cost summary SHA-256:
  `b8e604487e951796c53e125ab8ea7fb29b1682183747b0267eb9ee42a1c35e55`.
- Evidence scope: training-side necessary-condition diagnostic only. No
  validation/test, detector mAP, selector training or full-stack cost claim.

## 2026-07-20 DUCA detector-utility Pro audit prompt

- Prompt:
  `docs/methods/prompts/2026-07-20-duca-allocation-negative-result-detector-utility-pro-audit-prompt.md`.
- Documentation commit:
  initial `706e23b`; final-mAP correction `db11aee`.
- GitHub branch:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-allocation-ceiling-20260720`.
- Immutable implementation audit target:
  `8ebdd2a11ea5cc0644979324872a3b1cae5a2170`.
- Scope: physical-grid/frozen-detector evaluator validity, causal explanation
  of the sealed loss contradiction, detector-compatible utility definition
  and one bounded next experiment. The corrected prompt requires final mAP for
  route judgment and does not treat frozen loss as a KILL signal. It does not
  authorize selector training, other detectors or a paper claim.

## 2026-07-20 DUCA Protected-E2E design-adjudication prompt

- Prompt:
  `docs/methods/prompts/2026-07-20-duca-protected-e2e-design-adjudication-pro-prompt.md`.
- GitHub repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Clean branch:
  `codex/duca-protected-e2e-20260720`, created from `db11aee`.
- Code baselines:
  `db11aee`, `8ebdd2a`, `4ce69c8`, and immutable trained model `1642f26`.
- Scope: one protected direct detector-gradient architecture, P0-P3 gates and
  one four-arm official-60 preregistration. It explicitly forbids new routes
  and creates no implementation or empirical evidence.

## 2026-07-20 DUCA Protected-E2E Pro adjudication response

- Raw response:
  `docs/methods/reviews/2026-07-20-280631a-duca-protected-e2e-pro-adjudication-raw.txt`.
- Raw response SHA-256:
  `f91db53a83d79f56927b04d38b1e886d2e4260e4528e7882ddd49adbda97ccb0`.
- Absorption:
  `docs/methods/2026-07-20-280631a-duca-protected-e2e-pro-adjudication-absorption.md`.
- External visibility scope: repository and prompt commit
  `280631a27ffadad7d47eff4d379d6203427e013e`; it did not inspect later
  protected implementation commits.
- External verdict: `REVISE`.
- Local current-code recheck target:
  `b3222af0895e23eca83113977c1bcfad75258c9e`.
- Diagnostic runtime: Job `1176948` passed main/rho exact full-model P1/P2
  gates and stopped before P3 statistics on a stale manifest-field
  expectation.
- Accepted scope: physical exact-K hard/soft DAG, protected gradient
  ownership, physical-grid coordinate contract, strict P0-P3 and one
  four-arm terminal-EMA mAP matrix.
- Qualification: the response's 5940-update count remains unverified until
  exact-loader P0 evidence; current branch is nonconforming and no
  official-60 is authorized.

## 2026-07-20 DUCA pre-backbone design and paper-readiness review

- Structured local review:
  `docs/methods/reviews/2026-07-20-duca-prebackbone-design-paper-readiness-review.md`.
- Local implementation scope:
  `.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720`, an uncommitted
  construction tree based on `b3222af`; it is not an empirical evidence
  commit.
- Registered local evidence: physical-DAG `9 passed`, integrated selector
  `14 passed`, detector-contract focused gate `24 passed`, CellCF matched
  seed-0 terminal values and Allocation-Ceiling sealed diagnostics.
- Literature boundary sources:
  - Yeung et al., CVPR 2016, frame-glimpse action detection:
    `https://openaccess.thecvf.com/content_cvpr_2016/html/Yeung_End-To-End_Learning_of_CVPR_2016_paper.html`.
  - ETAD, CVPR Workshops 2023, snippet/proposal sampling for efficient
    end-to-end TAD training:
    `https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Liu_ETAD_Training_Action_Detection_End_to_End_on_a_Laptop_CVPRW_2023_paper.html`.
  - TE-TAD, CVPR 2024, time-aligned coordinate expression:
    `https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html`.
  - Progressive Block Drop, CVPR 2025, TAD model compression:
    `https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Temporal_Action_Detection_Model_Compression_by_Progressive_Block_Drop_CVPR_2025_paper.html`.
  - Search-Map-Search, CVPR 2023, task-aware frame selection for action
    recognition:
    `https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Search-Map-Search_A_Frame_Selection_Paradigm_for_Action_Recognition_CVPR_2023_paper.html`.
  - TAPS, ACCV 2024, temporal attention pruning/scaling for video action
    recognition:
    `https://openaccess.thecvf.com/content/ACCV2024/html/Dinai_TAPS_Temporal_Attention-based_Pruning_and_Scaling_for_Efficient_Video_Action_ACCV_2024_paper.html`.
- Review outcome: `HOLD_AND_REVISE`, with no claim-status promotion. The
  frozen four-arm route is retained, while P1 gains an explicit backbone
  irregular-time semantic gate.
## Protected-E2E focused pre-submission evidence (2026-07-21)

- Active isolated draft:
  `.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720`,
  branch `codex/duca-physical-protected-e2e-20260720`, still uncommitted at
  evidence time.
- Disposable N16R4 test copy:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_protected_selector_draft_20260720_01`.
- Focused Linux/Torch result: `55 passed in 45.68s`.
- Required legacy C3 result: `23 passed in 19.85s`.
- Remote queue audit at `2026-07-21 00:52:55 +0800`: no DUCA jobs.
- Evidence meaning: component/protocol `tested_focused` only; no P0 freeze,
  CUDA gate, P3 result, official-60 training, terminal EMA mAP or `>65`
  support.

## Protected-E2E exact P0 and low-quota deployment (2026-07-21)

- GitHub branch:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-physical-protected-e2e-20260720`.
- Exact commit:
  `ee05f610133fc37f8f1ee67b7225bb38ae917cc5`.
- Exact tree:
  `a190e399bb1fdfdac230c0a4305c4b08946a8ec1`.
- Clean remote snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_protected_physical_ee05f61_20260721`.
- Exact Linux suite: `84 passed in 60.51s`.
- P0 manifest:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_protected_physical_ee05f61_p0_20260721_021454/protocol_manifest.json`.
- P0 file SHA-256:
  `a02b6e690804d574d7929a408c17b396cc3cca4887a352be6c55270846e46a7e`.
- P0 content SHA-256:
  `aa10ae4b06b903806ffee1331ee64b0c14846d5ca649170b95c49643733a77d9`.
- Single-job precheck:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_protected_physical_ee05f61_gate_single_precheck_20260721_021605`.
- Formal single-allocation gate root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_protected_physical_ee05f61_gate_single_20260721_021705`.
- Formal Slurm Job: `1177681` (`dp_all_ee05f61`), submitted at
  2026-07-21 02:23 +0800; initial state `PENDING (AssocGrpGRES)`.
- `jobs.tsv` SHA-256:
  `29c69d51c2bc99199e8ddfa4dafa1f56fac66c856452a5a9854d3d5405f8f8b7`.
- `execution_plan.tsv` SHA-256:
  `d91e2ae701328e3914b4274fa54280b50dfc5b98271c39ac9e6b2bace50d6fb0`.
- Generated sbatch SHA-256:
  `cff10183d2e7d04f7d55def0f031d1d42265bcfc6fc1097bfa8245bdc0e2aff6`.
- Evidence meaning: gate-stage status is `experiment_running`; no CUDA/P3
  artifact, authorization, official-60 training, terminal mAP, or claim exists
  until Job `1177681` completes successfully.
## Uni-AdaFocus official implementation (2026-07-21)

- Repository: `https://github.com/LeapLabTHU/Uni-AdaFocus`.
- Exact audited commit:
  `8846488310fdd4a18412608006030643e794c36e`.
- Relevant files:
  `Uni-AdaFocus with Experiments on ActivityNet, FCVID and Mini-Kinetics/models/uni_adafocus.py`
  and the matching `main.py`.
- Verified facts: the training graph includes learned and random local input
  branches, a soft temporal policy surrogate and policy-specific optimization;
  hard temporal indices used by the heavy local path are detached.
- Deeper code audit: `policy_sample_indices` performs inverse-CDF quantile
  sampling and collision repair, guaranteeing K ordered unique indices over
  probability mass but no physical-time max-gap or TAD-boundary coverage.
  `MCSampleFeature` supplies the differentiable temporal-policy task surrogate.
  The final classifier reuses the cheap global features together with local
  heavy features; the paper reports roughly 0.8--2.8 classification points from
  this reuse across settings.
- Metric boundary: ActivityNet results in this paper are video-level
  multi-label classification mAP, not temporal action detection mAP, and must
  not be compared numerically with THUMOS14 TAD Avg-mAP.
- Project use: inspiration for a training-only exact-uniform companion and
  detector-bridge gradient scaling. It is not copied as proof that hard
  temporal sampling receives direct downstream gradients.

## DUCA Uni-Companion exact deployment (2026-07-21)

- GitHub branch:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-uni-companion-20260721`.
- Exact commit:
  `d748684bc6a3da5b5cbbb0b78a64b71ef1cdd1dc`.
- Exact tree:
  `50e43e7a91dc529b11d660f21e6fef46e4340601`.
- Remote Linux focused result: `66 passed in 58.34s`.
- Required legacy C3/ASFormer result: `23 passed in 21.13s`.
- P0 manifest file SHA-256:
  `e4fc629305fbb41ca5915ad71f866041340cf7e5b1c61d960ea647c74f6d2538`.
- Deployment manifest:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_uni_d748684_official60_20260721_0330/deployment_manifest.json`.
- Deployment manifest SHA-256:
  `97df66fa5c6e05907d8cdae513476e0aaaa501f5ebc9a12e135b643011eb000e`.
- Superseded Job IDs: gate `1177687`; direct `1177690`; bridge-0.25
  `1177691`; Uni-companion `1177692`. All were later cancelled at zero runtime
  after the real-loader uint8 incompatibility was found.
- Bounded exact-uniform/aggregate watcher script SHA-256:
  `f69c8387bda43a06e9373d8bcd05d5a98d9c5992ede1883b9bb965fb5faa338a`;
  initial remote PID `485235`.

## DUCA Uni-Companion real-loader input repair and replacement deployment (2026-07-21)

- GitHub branch:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-uni-companion-inputfix-20260721`.
- Exact commit/tree:
  `4d84acda4d073fb6aac956c21386df8ed5d4d2f5` /
  `b15a064784f25d888cc66df01c39781422403195`.
- Clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_inputfix_4d84acd_20260721`.
- Exact Linux verification: `67` focused plus `23` required legacy tests,
  no failure.
- P0 manifest/file SHA-256/content SHA-256:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_inputfix_4d84acd_p0_20260721_0430/protocol_manifest.json`,
  `eabc6da8c3cc4308b70a8c8d6bbecc6c6e4b408cb17d2ee6041ed83f24a4eb3f`,
  `e40252750c2fa680178fc45f38d014cb32a0e0acb34beb4477c5a6f0c8f02b93`.
- Replacement gate/learned Jobs:
  `1177696`, `1177697`, `1177698`, `1177699`.
- Official run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_inputfix_4d84acd_official60_20260721_0435`.
- Deployment manifest/jobs TSV SHA-256:
  `13f2b13c906f6605b8bbca6d06ad24201bcae83a027477dd42385b242807f6f4`,
  `9c9805019916a2547a1813c16d10072449cfc4199bbc91be8a8378badd995c4b`.
- Exact-uniform watcher script SHA-256/PID:
  `98958f4c6dc0d57dd5412034c1eafc78bf087a803b35a4663486e561bb642cae`,
  `808310`.
- Rho=0.01 sequential wrapper/watcher SHA-256/PID:
  `20e39c47014a2a3e51a3277507b751a4a994f5c4e400518d117a86bfdc3362b4`,
  `c9b15b5f3a0cf369349a66548686db509a4dd90bf521ae4d418c5557860b1902`,
  PID `883230`.
- Superseded Jobs `1177687/1177690-1177692` were cancelled with zero
  runtime after the real uint8-loader contract failure was found. They are
  not scientific evidence.

## DUCA 4d84acd zero-runtime Slurm failure (2026-07-21)

- Gate Job: `1177696`, `FAILED/127:0`, elapsed `00:00:00`.
- Stderr:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_inputfix_4d84acd_gate_20260721_0430/logs/all_gates-1177696.err`.
- Exact error: `module: command not found` before any model command.
- Dependent Jobs `1177697-1177699` were never started and were cancelled.
- This source records a deployment-wrapper defect only; it contains no model,
  optimizer, checkpoint, cost or mAP evidence.

## DUCA homotopy exact gate deployment (2026-07-21)

- GitHub branch:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-uniform-homotopy-20260721`.
- Superseded exact commit/tree:
  `be18ba53fb34c6d68d60b7b63edf1a7380d55c93` /
  `3050759426596db68e2d1bf247ac63150c1861ac`.
- Bundle SHA-256:
  `ddbd50e94ae2bafa74bcf10d1894c56a58da520de2b98b39f0be9f3a4fea6b88`.
- Remote snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_homotopy_be18ba5_20260721`.
- Exact Linux verification: `152 passed in 64.20s`.
- Gate/P0 root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_homotopy_be18ba5_gate_20260721_062249`.
- P0, jobs TSV, execution plan and generated sbatch SHA-256:
  `8385b9fb74c90c0faf7fad4761d85864450460bc43c69e6a606a5a0f1dfb8414`,
  `0bf80f77180700e74c5f3a48b91e314503e4bcebb6a39d2bd39abd15bab19c7a`,
  `a2b20b26fe28193c83d3b0bc7daf8054fe0b210305e48bba2a2fb187a6f93618`,
  `dad287cb963cfde3f6ff2b2da0c38bd11551cea9866848fb19c21e4e7391a90b`.
- Formal Slurm gate `1177713`: `FAILED/1:0`, elapsed `00:02:33`. It exposed
  physical-cap dtype narrowing and produced no authorization or mAP.
- Replacement commit/tree:
  `bc503fc3aa5c21487ca0c3679648f3c3085af82d` /
  `eb6cf2f3ff7ede7e7593ea44a8597f2f79d8cc87`.
- Replacement bundle SHA-256:
  `a34d38b619e7f9e768e3da461f13745665f1a5bc3897c46169d36d7a82d15044`.
- Replacement snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_homotopy_bc503fc_20260721`.
- Replacement Linux verification: `154 passed in 63.67s`.
- Replacement gate/P0 root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_homotopy_bc503fc_gate_20260721_063354`.
- Replacement P0, jobs TSV, execution plan and sbatch SHA-256:
  `7b5820fea25ae7866952341b9983c23f6d3a4891d4cf2aaf047175cb6ad96483`,
  `d22772525141bb060ac737122807ae87101450592f9028b8b3181b137d13e77c`,
  `a2b20b26fe28193c83d3b0bc7daf8054fe0b210305e48bba2a2fb187a6f93618`,
  `e781d4d6fc627b7871a403513a4eee1c334fa760292759a1f6a2fdd0488e1357`.
- Replacement Slurm gate Job `1177714`: `FAILED/1:0`, elapsed `00:02:33`.
  The full-model path crossed the physical-cap check, then the gate-only
  perturbation helper raised `normal_kernel_cuda not implemented for Byte` on
  real-loader uint8 RGB. It produced no authorization, checkpoint or mAP.
- Current exact commit/tree:
  `b987c8c6bd2b9f83027354adaaf6f338a205798a` /
  `d33d91941578e16cbf5a8cdc67b8b58471a29411`.
- Current bundle SHA-256:
  `e0ad6f46f861a51edec569e512244e6349f1f31d50aec3b89cba77f6d3196cfb`.
- Current snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_homotopy_b987c8c_20260721`.
- Current Linux verification: `155 passed in 63.67s`.
- Current gate/P0 root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_homotopy_b987c8c_gate_20260721_064211`.
- Current P0 SHA-256:
  `a246dc8c3fbc6f6e4a65a3a706a1259e54421f93a4707a922c567db1c92f9b99`.
- Current Slurm gate Job `1177715` entered `RUNNING` on `g0003` at
  2026-07-21 06:46 +0800. No CUDA/P3 authorization, official-60 checkpoint
  or mAP exists at registration time.
- Current Slurm gate Job `1177715`: `FAILED/1:0`, elapsed `00:02:40`.
  Fail-closed reason: exact-uniform physical and selected-axis detector losses
  disagree. No authorization file or training Job was produced.
- Read-only parity diagnostic Job `1177719`: `FAILED/1:0` by design after
  printing the blocked assertion inputs. Physical cls/reg/objective were
  `0.040603362/0.031404633/0.072007999`; selected-axis values were
  `0.054389104/0.040477306/0.094866410`. Absolute objective difference was
  `0.022858411` (24.10% relative to selected-axis). Uniform gap histogram was
  `{2: 382, 3: 1}`. The diagnostic changed no repository file or checkpoint.

## DUCA selected-axis epoch-4 optimization diagnostics (2026-07-21)

- Method commit remains immutable `cb89586a92b8b0a8349ecc9551bc50aa97982360`.
- Diagnostic branch:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-selected-axis-diagnostics-20260721`.
- Hard-trajectory commit/job: `87cfd20938dd9411d8f1dc82091eaf9ec93c7f1d` /
  `1178357`.
- Normalized delta-residual commit/tree:
  `7f9ad10ac35cb61fa68a17003f2bc1c488dd9c10` /
  `c397d073d9b1c396c06babf61f4ee0b3aa22ced3`.
- Residual bundle SHA-256 and clean snapshot:
  `651bdd634652b058ac7fab1847b091be12dd5ea9c706f493a2e121a2f31f8383`,
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_diag_7f9ad10_20260721`.
- Clean residual/trajectory tests: `9 passed in 37.10s`.
- Residual script SHA-256/job/dependency:
  `f797af69b0ee29f2bd9b3ef38d373caccc8ab6c9a786f80835e38fe3bc3d997b`,
  `1178384`, `afterok:1178357`.
- Both jobs are selector-only diagnostics and produce no detector mAP.

## DUCA two-stage curriculum (2026-07-21)

- GitHub branch:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-two-stage-curriculum-20260721`.
- Exact commit: `6f2ed48d8cb31fe984b8a20223a0624fcf95d4b9`.
- Remote snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_twostage_7431543_20260721`.
- Clean remote focused/regression evidence: `83 passed in 50.12s`.
- Superseded storage-failed root/job:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_twostage_6f2ed48_serial_20260721_1220` / `1178487`.
- Superseded Job `1178480` failed before optimizer construction due leaked
  frontend optimizer config and contains no model/mAP evidence.
- At 12:31 +08:00, `df -h /data/run01/sczc063/yuzibo` reported the shared
  JuiceFS `/data` mount at 100% use with zero available space. Job `1178487`
  remained RUNNING but its P0 log had stopped near epoch 1; status is
  infrastructure-at-risk.
- Selected-axis Jobs `1177779-1177782` all ended `FAILED/1:0`; every arm log
  records `OSError: [Errno 122] Disk quota exceeded`. They contain no terminal
  checkpoint or mAP evidence.

## Remote checkpoint retention audit (2026-07-21)

- Scope: `/data/run01/sczc063/yuzibo`, excluding datasets, pretrained weights,
  environments, caches, packages and Git metadata.
- Consolidated manifest:
  `/data/run01/sczc063/yuzibo/cleanup_manifests/checkpoint_cleanup_consolidated_20260721_132157.json`.
- Manifest SHA-256:
  `a06d3062a1fc2f8ec9d1ef336271f688368dfe2c788fea6933d8cc9e1a04b60a`.
- Retained TSV SHA-256:
  `b0c1cd7c78e9d9825094e06ea7e45002bd4bfd4fcf2d215aac3bb829fe91f006`.
- Final invariant: 220 numeric files in 220 independent groups, all validated;
  689 files/334,791,638,367 bytes deleted; `/data` 87% used with about 310 GiB
  available.
- Old selected-axis four-arm recovery root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_selected_axis_cb89586_suite_20260721_0845`.
- Retained per-arm record: `epoch_24.pth`; CPU load verified model/EMA,
  optimizer, scheduler, GradScaler and RNG state.
- Two-stage Job `1178487`: `FAILED/1:0`, elapsed 00:24:02, no checkpoint.

## DUCA storage-recovery redeployment (2026-07-21 13:34 +08:00)

- Selected-axis exact continuation root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_selected_axis_cb89586_resume_e24_20260721_133337`.
- Immutable method snapshot/commit:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_selected_cb89586_20260721` /
  `cb89586a92b8b0a8349ecc9551bc50aa97982360`.
- Resume gate/arms: `1178581`; `1178582`, `1178583`, `1178584`, `1178585`.
  The four arm dependencies are `afterok:1178581` and their parent hashes are
  recorded in the continuation root's `jobs.tsv` and
  `deployment_manifest.json`.
- Gate `1178581` completed `0:0`; gate-suite/resume-gate SHA-256 values are
  `ef9ab397642455af8c8c6580198df855049c7db268430b2b568124ec2a336974`
  and `614871a17437b906aa7cc2daaae5a32f658baa0bd91bd0e8fcf338013647e125`.
- Jobs `1178582-1178585`, `1178614-1178617` and `1178633-1178636` are
  pre-restore launcher failures from missing commit, canonical `BASE`, and
  gate-suite exports. They made zero optimizer updates.
- Admissible v4 root/job:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_selected_axis_cb89586_resume_e24_v4_20260721_135701` / `1178642`.
  This is one two-GPU/two-wave allocation. Runtime-resume preflight SHA-256 is
  `dae2775878465da16417faf33e20236bf8658f5d2df317b1ec9e5dda72d009d1`;
  deployment-manifest SHA-256 is
  `c1c57afd23e3d44e08f02ca800845e086c79109f25a1a361a423a26157d6f273`.
  Runtime-validation SHA-256 after both first-wave arms completed resumed
  epoch 25 is
  `3af133daa84e8d31de2c8cb5b08ca30b440a0e381461030e4007e82c9466c0b5`.
  Exact-uniform and direct-0.25 restored epoch 24, completed epoch 25 and
  entered epoch 26; both homotopy arms are ordered as wave two.
- Two-stage immutable snapshot/commit:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_twostage_7431543_20260721` /
  `6f2ed48d8cb31fe984b8a20223a0624fcf95d4b9`.
- Parallel precheck root:
  `duca_twostage_6f2ed48_precheck_20260721_133354`; precheck passed, but Slurm
  rejected DAG submission with `AssocMaxSubmitJobLimit` and the transaction
  rolled back all held jobs.
- Admissible serial root/job:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_twostage_6f2ed48_serial_20260721_133422` /
  `1178591`. Split-manifest SHA-256 is
  `be84b85a38b9ec9176d80418a3a866e143a1e7073ab70f80805ebf570b82118a`.

## DUCA two-stage exact-commit Pro audit (2026-07-21)

- Reviewed commit:
  `6f2ed48d8cb31fe984b8a20223a0624fcf95d4b9`.
- Raw review:
  `docs/methods/reviews/2026-07-21-6f2ed48-duca-two-stage-curriculum-pro-audit-raw.txt`.
- Raw review SHA-256:
  `0b265d08b811b821b1014cf7c52b579a759ee79e637710260a48cfc284367379`.
- Coordinator absorption:
  `docs/methods/2026-07-21-6f2ed48-duca-two-stage-curriculum-pro-audit-absorption.md`.
- External evidence mode: read-only GitHub static audit; no CUDA, Slurm,
  training, profiling or terminal evaluation was executed by the reviewer.
- Independent verification target: clean remote snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_twostage_7431543_20260721`,
  exact HEAD `6f2ed48`.
- Confirmed facts: hidden nonzero loss defaults enter selector `cost`;
  transition supervision reaches ASFormer hidden; official warmup lacks
  optimizer/clip/EMA isolation; schedule boundaries are inclusive; the direct
  bridge is not the true legal hard-swap utility.
- Adjudication: accept implementation HOLD and route continuation after repair.
  Treat the bounded-residual architecture and fixed numeric thresholds as
  proposals pending reachability, cost and matched terminal-mAP evidence.

## DUCA two-stage exact-commit route audit V2 (2026-07-21)

- Reviewed commit:
  `6f2ed48d8cb31fe984b8a20223a0624fcf95d4b9`.
- Raw review:
  `docs/methods/reviews/2026-07-21-6f2ed48-duca-two-stage-curriculum-route-audit-v2-raw.txt`.
- Raw review SHA-256:
  `bca69084bfb1c09f5fe92d49aa10362b18fecf69ff8d2fa754c1d53335734703`.
- Cross-audit absorption:
  `docs/methods/2026-07-21-6f2ed48-duca-two-stage-audit-v1-v2-comparison-absorption.md`.
- External evidence mode: read-only GitHub static audit; no CUDA, Slurm,
  training, profiling or terminal evaluation was executed by the reviewer.
- Independent verification target: clean remote snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_twostage_7431543_20260721`,
  exact HEAD `6f2ed48`.
- Additional confirmed facts: negative entropy can oppose binary separation;
  the named balanced BCE defaults to `pos_weight=1`; transition radii default
  to four; hidden differences dominate the descriptor; padded frames enter
  train-mode BatchNorm before masking.
- Cross-audit adjudication: shared HOLD and bounded-residual direction are
  accepted. V1 hard-swap distillation and V2 local soft-RGB backward are
  separate candidates pending one real hard-swap alignment harness.

## DUCA full local-reachability audit (2026-07-21)

- Input records:
  `E:/DeskTop/TAD/duca_local_reachability_artifacts_20260721/holdout_records.jsonl`;
  SHA-256 `867c245c33d8efa60b9a92ce691a208c0277357771bb5d3dac31c1844dddfe2a`.
- Training-holdout split manifest SHA-256:
  `be84b85a38b9ec9176d80418a3a866e143a1e7073ab70f80805ebf570b82118a`.
- Full semantic records:
  `E:/DeskTop/TAD/duca_local_reachability_artifacts_20260721/full120_semantic.records.jsonl`;
  SHA-256 `362aa4b22a5fa56e4a393bdbdba025f2ea47afa094d34ec43bda92a7e459b2e2`.
- Summary:
  `E:/DeskTop/TAD/duca_local_reachability_artifacts_20260721/full120_semantic.summary.json`.
- Scope: 120 samples, 40 training-holdout videos, exact K=384, matched physical
  cap, no test subset and no detector mAP. Privileged local/global GT oracles
  are evaluation-only and non-deployable.
- Key result: local and global oracles match on every reported coverage radius;
  local/global mean endpoint distance `0.2484311/0.2462081`. Coarse
  AUROC/AUPRC/Brier `0.6161000/0.3749577/0.2042126`.

## DUCA repaired frontend P0 deployment (2026-07-21)

- Repository branch: `codex/duca-local-residual-20260721`.
- Exact commit: `5d17dcbe564efd1e69194dd5faddf34266e39f86`.
- Clean remote snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_local_5d17dcb_20260721`.
- Linux focused verification: `96 passed, 2 skipped`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_p0_5d17dcb_20260721_1640`.
- Split manifest SHA-256:
  `1a946a7890318ece2b2f500d84cccc2b2785e08f5780bbb5239ca208e9483be1`;
  160 train videos, 40 training-holdout videos, zero overlap, test unused.
- Slurm Job: `1178774`; one GPU, one real gate followed by three sequential
  frontend candidates, then stop. Status at registration: `PENDING`; no gate
  verdict or model result yet.
- Gate diagnostic: Job `1178774`, `FAILED/1:0`, no candidate update. The gate
  searched for `spatial_encoder` while the executed path was `spatial_stem`.
- Corrective exact commit:
  `9442b9487f871efd02c85dceeed26574c641369d`.
- Corrective clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_local_9442b94_20260721`.
- Corrective Linux verification: `74 passed, 3 skipped`.
- Replacement Job `1178809`; run root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_p0_9442b94_20260721_165719`.

## DUCA global-curriculum exact implementation and submission (2026-07-21)

- Repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Branch: `codex/duca-global-curriculum-20260721`.
- Exact commit:
  `4c777a691d65fe484dfe537ac3e33f82b5bbe5a8`.
- Local isolated clone:
  `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721`.
- Remote clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_4c777a6_20260721`.
- Remote focused evidence: `74 passed, 2 skipped`; local Python and shell
  syntax checks also passed.
- Formal run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_4c777a6_serial_20260721_1849`.
- Slurm Job `1178911`; observed at 2026-07-21 18:51 +08:00 as
  `PENDING (Priority)`.
- Final scheduler state: `CANCELLED`, elapsed `00:09:01`; cancellation was
  deliberate after the P0 protocol audit and all partial logs remain history.
- Immutable submission receipt:
  `duca_global_4c777a6_serial_20260721_1849/submission/receipt.json`;
  schema `duca_two_stage_serial_submission_v1`; split-manifest SHA-256
  `3e98c3fff0e24fe50003e6af3cad7f88e02b32fed8161dfb470a445cb875059a`.
- Evidence scope: source/config/gate identity and focused contracts only.
  CUDA full-model authorization, terminal official-60 mAP and cost results do
  not yet exist.

## DUCA selected-axis partial terminal evidence (2026-07-21)

- Exact method commit: `cb89586a92b8b0a8349ecc9551bc50aa97982360`.
- Continuation Job/root: `1178642` /
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_selected_axis_cb89586_resume_e24_v4_20260721_135701`.
- Exact-uniform terminal JSON:
  `runs/exact_uniform/terminal_evaluation.json`; SHA-256
  `a58fb9b76090d6da955c23563150f5fe24b6c9a1eddad9330b88dd5ba7d1ae1e`;
  checkpoint SHA-256
  `17d7461ec48eb70d1efa0ac85be28319a60cb0ea601141bcc0ccb44949bdb5a2`.
- Direct-0.25 terminal JSON:
  `runs/direct025/terminal_evaluation.json`; SHA-256
  `4f3f26ea17311212d8c86f2a7fba2687189a16557a0db24d5346e92d88f4747b`;
  checkpoint SHA-256
  `5cd0ca6b0b3b23a7c30fca9e6a74f0c60336b1a72e892180cf92f97af8fccfd5`.
- Raw terminal Avg-mAP: exact-uniform `0.6445799769441278`;
  direct-0.25 `0.6371015461666212`.
- Evidence boundary: wave two remains running; this source is a partial
  terminal comparison and not a completed four-arm or greater-than-65 result.

## DUCA P0 evidence-contract replacement (2026-07-21)

- Repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Branch: `codex/duca-global-curriculum-20260721`.
- Exact commit:
  `e0397ec0bcb917593664ce36efd8105e31d0a302`.
- Local isolated clone:
  `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721`.
- Remote clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_e0397ec_20260721_v2`.
- Transfer provenance: verified 10,140-byte incremental Git bundle requiring
  parent `4c777a6`; remote checkout and clean-tree hash were verified after
  the remote GitHub TLS connection failed. No third-party mirror was used.
- Remote verification: directly affected suite `27 passed`; complete DUCA
  regression suite `158 passed, 3 skipped`; Python compile and shell syntax
  checks passed.
- Active run root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_e0397ec_serial_20260721_1939`.
- Slurm Job: `1178927`; initial state `PENDING (Priority)`.
- Submission receipt SHA-256:
  `ee921b6bfcfdfdec4ab9360bf68c07e0b48714406b4933ad6e3d6f8b482a6834`.
- Split manifest SHA-256:
  `8819e531440a2385f3c44fd939e293602c253ea988bf9862e217b204cbad3290`.
- Evidence boundary: exact code/tests/submission only. P0 holdout winner, CUDA
  full-model gates, terminal official-60 mAP and greater-than-65 remain absent.

## DUCA quality-export launcher correction (2026-07-21)

- Exact commit: `2c403a853d55057ae772e1b8dcc0c4ebb8cbc0f5` on the same
  `codex/duca-global-curriculum-20260721` branch.
- Remote clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_2c403a8_20260721`.
- Exact verification: `28 passed`; both
  `python -m tools.bata.export_duca_selection_quality --help` and
  `python -m tools.bata.analyze_duca_selection_quality --help` succeeded;
  shell syntax and clean-tree checks passed.
- Failure source: local-cell Job `1178863` ended after the first 20-epoch P0
  candidate with `ModuleNotFoundError: No module named 'tools'` during
  post-training quality export. This classifies it as launcher failure.
- Zero-runtime intermediate Job `1178927`: `CANCELLED` before execution.
- Then-active Job/root at this superseded checkpoint: `1178933` /
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_2c403a8_serial_20260721_1949`.
- Submission receipt SHA-256:
  `d25a81a8142fe98d4e3c08acf4aa44f8f58f4f07ac4de19e7684acd06c1e3f0d`.
- Split manifest SHA-256:
  `12dba835cb2c881f90ed14849f7077088d07b619ed9ad1099f3abbee72e68a12`.
- Initial scheduler state: `PENDING (Priority)`; no model-result claim.

## DUCA complete-entry correction and ASFormer compatibility audit (2026-07-21)

- Exact commit: `6b6363e93674652706a15214a0ffbdc299d706dc` on
  `codex/duca-global-curriculum-20260721`.
- Remote clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_6b6363e_20260721`.
- Exact remote verification: `18 passed`; both aggregation modules resolve;
  `py_compile`, serial-launcher `bash -n`, exact HEAD and empty porcelain pass.
- Superseded Job `1178933`: `FAILED/1:0`, elapsed `00:00:03`; raw error is
  `[DUCA_TWO_STAGE_SERIAL][FAIL] clean tree required`. It has no model update.
- Then-active Job/root at this superseded checkpoint: `1178947` /
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_6b6363e_serial_20260721_2022`.
- Submission receipt SHA-256:
  `af788b5447eb21152fb8ab6feaf06098d7786a738515eed97eecccb5dfbeafd5`.
- Split manifest SHA-256:
  `c0fb8172a576eea980fab370e9174f1d26d5e8819d7c411ac222c5feb1a0f3fc`.
- Existing standalone official-ASFormer checkpoint:
  `.../official_action_seg_official_asformer_64/probe_reader.pth`, SHA-256
  `34e4d510441dc711bfc12599ae772f05c372a89d8988529abfbe6b3405f3bbba`.
  Its summary reports AP `0.4348660938`, AUROC `0.6315410299`; strict loading
  passes for BatchNorm and fails for current GroupNorm only on six BN running-
  statistic/counter keys. This is compatibility evidence, not an authorized
  warm-start experiment or a terminal TAD result.

## DUCA P0 optimizer-gate classifier correction (2026-07-21)

- Exact commit: `91381568637f6358bdec67e3d8400d70869f1dd6` on
  `codex/duca-global-curriculum-20260721`.
- GitHub:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-global-curriculum-20260721`.
- Remote clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_9138156_20260721`.
- Incremental transfer bundle SHA-256:
  `19c629e3f59c10e16a6c1020a9c8c526a980b806264481797fd2100c1c8b0c66`;
  prerequisite commit `6b6363e93674652706a15214a0ffbdc299d706dc`. The temporary local and
  remote bundle files were removed after the exact snapshot passed HEAD and
  clean-tree verification; the immutable hash remains the transfer receipt.
- Exact verification: targeted optimizer/gate tests `15 passed`, Python compile,
  launcher syntax, exact HEAD and empty porcelain passed.
- Superseded Job `1178947`: `FAILED/1:0`, elapsed `00:01:00`; gate artifact says
  `an optimizer group mixes declared component learning rates`. Source audit
  showed this was an over-broad gate classifier for official-ASFormer internal
  attention `conv_out`, not an actual optimizer partition failure. No optimizer
  update occurred.
- Active Job/root: `1178975` /
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_9138156_serial_20260721_2042`.
- Submission receipt SHA-256:
  `9726d8829e5bb488726c4180f1418dd384cfc00f9338fb4244be833c323de394`.
- Training-only split-manifest SHA-256:
  `54ddff7e548389ae79f27c3e7d53344348f4f7a94ed95f03ef7a806165865243`.
- Evidence boundary: model identity remains V8; no P0 winner or terminal mAP.

## DUCA P0 EMA group-audit correction (2026-07-21)

- Exact commit: `63e25eb17e523d369f73434ed4d9b6446608861a` on
  `codex/duca-global-curriculum-20260721`.
- GitHub:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-global-curriculum-20260721`.
- Remote clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_63e25eb_20260721`.
- Superseded Job `1178975`: `FAILED/1:0`, elapsed `00:00:56`. It completed one
  finite P0 optimizer step with detector skipped; failure artifact says
  `EMA did not update both P0 branches` because the gate inspected one
  representative tensor per group.
- Exact change: group-wide EMA parameter-change evidence plus one focused test;
  model, selector, decoder, losses, optimizer, schedule and four-arm protocol
  are unchanged. Remote affected regression: `21 passed`, plus `py_compile`
  and diff check.
- Active Job/root: `1178989` /
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_63e25eb_serial_20260721_2120`.
- Submission receipt SHA-256:
  `b1b7892b282b265a77efc7df42a034bb2d1c7fde415471f06b6179fa07d42c85`.
- Training-only split-manifest SHA-256:
  `7b381a38d4a0d66f5746c768df2a9c2ab7f27e6c93e83a99e707eaa6300217a7`.
- Evidence boundary: same V8 model; no P0 winner or terminal mAP.

## DUCA V8 unique-endpoint Pro review (2026-07-21)

- Repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Exact audited commit:
  `63e25eb17e523d369f73434ed4d9b6446608861a` on
  `codex/duca-global-curriculum-20260721`.
- Original attachment:
  `C:/Users/skywalker/.codex/attachments/e62d0b32-a5d2-4b44-bc30-6c43ec3f8d0c/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-21-63e25eb-duca-v8-unique-endpoint-pro-review-raw.txt`.
- Raw/archive size: `46,310` bytes, `1,126` lines.
- Raw/archive SHA-256:
  `DF19960D0B3158CE7F31E0FE4A92F8CD22C7B2AAFD5FB78D13E91DDACEA8EC70`.
- Structured absorption:
  `docs/methods/2026-07-21-63e25eb-duca-v8-unique-endpoint-pro-review-absorption.md`.
- Reviewer verdict: `HOLD`; replace Gaussian/mass coverage with a unique
  endpoint event objective and require a real hard-swap alignment gate.
- Project absorption:
  `SUBSTANTIAL_ACCEPT_DIAGNOSIS / REVISE_OBJECTIVE_BEFORE_IMPLEMENTATION`.
  Static code facts were confirmed, but the proposed radius-one exact event is
  tautological under max-hole two and cannot be implemented unchanged.

## DUCA V8 × Uni-AdaFocus / EU-CRR Pro review (2026-07-22)

- DUCA repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Exact audited DUCA commit:
  `63e25eb17e523d369f73434ed4d9b6446608861a`.
- Audited Uni-AdaFocus upstream commit:
  `8846488310fdd4a18412608006030643e794c36e`.
- Original attachment:
  `C:/Users/skywalker/.codex/attachments/0df7b420-40f9-459e-9e5e-07de47d4905a/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-22-63e25eb-duca-uni-adafocus-eucrr-pro-review-raw.txt`.
- Raw/archive size: `65,069` bytes, `1,426` lines.
- Raw/archive SHA-256:
  `0678A31C17D3FCD983726CE9056E463CF09A0325DAF69C7C41947EEB57602DAA`.
- Structured absorption:
  `docs/methods/2026-07-22-63e25eb-duca-uni-adafocus-eucrr-pro-review-absorption.md`.
- Reviewer verdict: `HOLD`; only an exact-uniform zero-gated residual-reuse
  diagnostic is proposed before any learned-selection/fusion combination.
- Project verdict:
  `SUBSTANTIAL_ACCEPT_DIAGNOSIS / CONDITIONAL_ACCEPT_DIAGNOSTIC / REJECT_AS_MAINLINE_REPLACEMENT`.
  G23/R0--R5 remains the canonical acquisition route; EU-CRR is an orthogonal,
  not-authorized representation diagnostic.

## DUCA corrected R0 exact-quota evidence (2026-07-22)

- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Branch/commit:
  `codex/duca-boundary-burst-20260722@22555a4e830ce24f9bb516897b1bb7f44b70c188`.
- Superseded immutable R0 Job: `1179392`; failure occurred before detector
  evaluation and produced no mAP.
- Failed-run input JSONL SHA-256:
  `4fceead48e87210e4b7ef8bb42ab0696e4b9057fde2912e5c34a5133f235501b`.
- Corrected real-sample Oracle replay SHA-256:
  `168c6f21f869d802e8e3a11fdfcedc2ddc7968fe6fb5b6909776fdb8f84e76ce`.
- Remote clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_22555a4_20260722`.
- Verification: solver/Oracle 22, P0 summary 9, runtime/gate/aggregate 54,
  mandatory C3 23 tests passed; pycompile/bash/exact HEAD/clean tree passed.
- Independent exact-commit reviewer:
  `019f8743-aed1-7a80-a7d6-552b08491019` (pending at registration time).
- Evidence boundary: exact-quota implementation and one real failure replay;
  no corrected R0 detector mAP, P0 winner, official-60 result or paper claim.

## DUCA e49ef696 R0--R5 Pro audit (2026-07-22)

- Repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Exact audited model commit:
  `e49ef69605e1f98a7217957483f93a8a64bfc348`.
- Current evidence/runtime successor:
  `8d85929ea04dc40f1eb0c3cc806061ce3b071d3f`; no selector, decoder,
  detector, loss or training-schedule change relative to the audited model.
- Original attachment:
  `C:/Users/skywalker/.codex/attachments/966f12f4-7208-41b9-88bb-a2ad8fb71d5b/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-22-e49ef696-duca-r0-r5-pro-audit-raw.txt`.
- Raw/archive size: `53,578` bytes, `1,404` lines.
- Raw/archive SHA-256:
  `1D0F9909D2C3DF3966DED0B9F71BFA0A73F9CA2B8D7C68DF15F64265EC8AD636`.
- Structured absorption:
  `docs/methods/2026-07-22-e49ef696-duca-r0-r5-pro-audit-absorption.md`.
- Reviewer verdict: `HOLD_FIX_REQUIRED` for paper evidence and system claims;
  continue the then-current e49 DAG.
- Project verdict:
  `SUBSTANTIAL_ACCEPT_CODE_DIAGNOSIS / PARTIAL_ACCEPT_RECOMMENDED_PLAN /
  REJECT_STALE_SERIAL_DAG_AND_UNCONDITIONAL_MATRIX_EXPANSION`.
- Verification: H1--H5 confirmed by current source; relevant remote focused
  suite passed `96 passed, 1 warning`. Passing tests do not implement the
  missing H1/H3/H4/H5 contracts.
- Evidence boundary: current independent four-arm official-validation suite is
  running; no terminal epoch-59 EMA mAP or paper-ready claim exists.

## Official coarse temporal backends for DUCA (2026-07-23)

- MS-TCN2: `https://github.com/sj-li/MS-TCN2`, fixed commit `f423a9e65f4ccb1cd7322eb9f94946a19e787993`.
- ASFormer: `https://github.com/ChinaYi/ASFormer`, fixed local official-source commit `e1bbe4f3ed083748f91467c51a63ac2a8b9277ad`.
- FACT: `https://github.com/ZijiaLewisLu/CVPR2024-FACT`, fixed commit `7bd81bda2b84618a3e23876a2637a82f24881122`.
- Video-Mamba suite: `https://github.com/OpenGVLab/video-mamba-suite`, fixed commit `ec9108b72d5db59f6d634c94cd0e008228a7b918`; DUCA uses its temporal-action-segmentation model source.
- Remote source root: `/data/run01/sczc063/yuzibo/projects/external_official_action_segmentation_repos_20260702`.

## DUCA a00498e selected-axis / TTDI Pro review (2026-07-23)

- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Audited commit: `a00498e15d69294f78d0abeadfb47bc456db0b0e`.
- Current model-equivalent execution successor:
  `9f97f2c7f081b10fbf1f63d0602a621c6b43a780`; diff changes runner scripts/tests only.
- Current diagnostic-only child: `4f81299f826a4d33b18f21af8436ec1bd8cc4f51`; adds coarse-backend P0 launcher/tests only.
- Original attachment:
  `C:/Users/skywalker/.codex/attachments/61034c80-3c84-4534-a575-3024e7e7a651/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-23-a00498e-duca-selected-axis-ttdi-pro-review-raw.txt`.
- Raw/archive size: `76,688` bytes; physical lines: `1,705`.
- Raw/archive SHA-256:
  `36523b2f1a7456f8d4a4314ea445971f8066eec59611f9632d7bc1d33e31a884`.
- Structured absorption:
  `docs/methods/2026-07-23-a00498e-duca-selected-axis-ttdi-pro-review-absorption.md`.
- Project verdict:
  `SUBSTANTIAL_ACCEPT_MODEL_DIAGNOSIS / PARTIAL_ACCEPT_TTDI_REMEDY /
  REJECT_UNCONDITIONAL_FULL_TTDI_AS_FINAL_MODEL`.
- Evidence boundary: no terminal `9f97f2c` official mAP; TTDI is not implemented or empirically supported.

## Frozen and target-train-free temporal evidence sources (2026-07-23)

- Uni-AdaFocus, arXiv 2024: `https://arxiv.org/abs/2412.11228`. Relevant boundary: lightweight global
  sequence encoding, interpolation of policy outputs/features and joint task-loss optimization; it is not a
  strict target-train-free selector.
- T3AL, CVPR 2024: `https://openaccess.thecvf.com/content/CVPR2024/papers/Liberatori_Test-Time_Zero-Shot_Temporal_Action_Localization_CVPR_2024_paper.pdf`.
  Relevant boundary: no labeled target training, but per-video test-time projector adaptation; it therefore
  does not satisfy DUCA's strict zero-optimization contract.
- CLIP official description: `https://openai.com/index/clip/`. Relevant boundary: externally pretrained
  vision-language representations permit zero-shot text-conditioned transfer, but do not prove generic
  action/background prompts are calibrated for TAD.
- Memory Matters, CVPR 2026: `https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Memory_Matters_Boosting_Training-Free_Zero-Shot_Temporal_Action_Localization_with_a_CVPR_2026_paper.html`.
  Relevant boundary: current literature may call a method training-free while still updating test-time memory
  or residual state; DUCA must state its stricter no-gradient/no-target-tuning definition explicitly.
- Microsoft VLM Video Action Localization: `https://microsoft.github.io/VLM-Video-Action-Localization/`.
  Relevant boundary: learning-free coarse-to-fine VLM temporal search is possible, but repeated VLM queries are
  a cost-heavy query-conditioned reference rather than a cheap generic pre-backbone probe.

## Frozen encoder checkpoints for target-train-free DUCA (2026-07-23)

- Torchvision MobileNetV3-Small ImageNet-1K checkpoint:
  `https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth`; size `10,306,551` bytes;
  SHA-256 `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.
- PyTorchVideo SlowFast-R50 Kinetics-400 checkpoint filename `SLOWFAST_8x8_R50.pyth`; the experiment
  executes only its Fast pathway. Local sealed size `277,138,115` bytes; SHA-256
  `454f39e1c1f985df2bee2aa27887ed53ff56e74ed8b8cca11203a1a1264d7cc2`.
- Academic mirror used for the sealed SlowFast file:
  `https://hf-mirror.com/AkaneTendo25/ayase-models/resolve/main/rqvqa/SLOWFAST_8x8_R50.pyth`.

## 2026-07-26 DUCA multi-round joint review synthesis

- Report title: `OpenTAD_C3 / DUCA 多轮联合审阅综合报告`.
- Original attachment:
  `C:/Users/skywalker/.codex/attachments/4672e0d4-479e-4a82-a819-7266a000e06c/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-26-duca-multiround-joint-review-raw.txt`.
- Raw/archive size: `18,959` bytes; physical lines: `141`.
- Raw/archive SHA-256:
  `67409BC9B140275BFC6804DD65FACBBEB568719304768A322FCF3A3F54576484`.
- Structured absorption:
  `docs/methods/2026-07-27-duca-multiround-joint-review-absorption.md`.
- Reviewer verdict: implementation review incomplete; require G0 source surface,
  bounded D0/D1 mechanism gates, matched/equal-cost baselines, multi-seed/budget
  evidence, complete cost accounting, and `paper_claim_allowed=False` until
  terminal evidence.
- Project verdict:
  `SUBSTANTIAL_ACCEPT_GOVERNANCE_AND_EXPERIMENT_DESIGN /
  ACCEPT_WITH_CURRENT_FACT_CORRECTIONS /
  REJECT_STALE_STATUS_AS_CURRENT_CONTRACT /
  HOLD_REVIEWER_PROPOSED_THRESHOLDS_UNTIL_RATIFIED`.
- Current corrections: Draft PR #2 has closed the read-surface portion of G0
  at exact public commit `42dba3f90b37243e7965d18b6707e88e81bf7109`;
  independent line-by-line adjudication remains open. Public base and DUCA
  branches still contain plaintext proxy authentication in `README.md`, so
  redaction, credential rotation, and history treatment remain separate urgent
  actions. Stage-2 numerical cause has since been isolated and the sole valid
  Job `1191957` has long-run finite-update evidence, but no terminal epoch-59
  EMA offline TAD result yet.
- Evidence boundary: the report is review/design evidence. Its proposed D1
  thresholds, Path A/B publication lines, exact ablation matrix, seed/budget
  counts, and ChronoTransport parking decision are not automatically
  preregistered project contracts.

## 2026-07-27 DUCA pure-plugin and official-baseline recovery

- Review corpus: all locally preserved DUCA raw discussion files under
  `docs/methods/reviews/`, with method identity, curriculum contracts,
  terminal verdicts and evidence boundaries rechecked against current Wiki
  nodes and exact configurations.
- Canonical synthesis:
  `research-wiki/duca_prebackbone_plugin_and_baseline_recovery_contract.md`.
- Original curriculum sources:
  `research-wiki/ideas/duca-two-stage-curriculum.md`,
  `research-wiki/experiments/duca-two-stage-curriculum-official60.md`,
  `docs/methods/reviews/2026-07-21-6f2ed48-duca-two-stage-curriculum-pro-audit-raw.txt`
  and its second-round audit.
- Official implementation reference:
  `sming256/OpenTAD@1aa8ca4ac5e846b1e8ff69298dd6607121a01589`.
  Local `configs/adatad/README.md` records VideoMAE-S `768/160` THUMOS
  Average-mAP `69.03%`
  (`83.90/79.01/72.38/61.57/48.27%` at tIoU `0.3--0.7`).
- Sparse parity audit:
  `research-wiki/duca_model_version_registry.md` records that the base config,
  nominal head configuration, projection, cls/reg objectives and NMS remain
  official-derived, while exact commit `42dba3f9` extends both the active
  `ActionFormer` and `AnchorFreeHead` source files. The DUCA detector wrapper,
  sparse selection, target/prediction mapping, temporal shapes and several
  runtime/training settings deliberately differ; upstream execution parity
  has not been established.
- Train-free sources:
  `research-wiki/ideas/duca-target-train-free-transition-prior.md` and
  `research-wiki/experiments/duca-t1-and-target-trainfree-official60.md`.
- Evidence boundary: this entry establishes design provenance and baseline
  mismatches. It does not create a clean native K=384/K=192 uniform mAP,
  prove the cause of the local dense gap, or provide a new model result.

## 2026-07-27 DUCA total-60 pre-backbone Pro major review

- Original attachment:
  `C:/Users/skywalker/.codex/attachments/975d262f-00ea-4639-a85c-a9c45aa03f9a/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-27-duca-total60-prebackbone-pro-review-raw.txt`.
- Raw/archive size: `52,824` bytes; physical lines: `886`.
- Raw/archive SHA-256:
  `D493FD3497D412B3B873940447F1C743F3A1A50418EBCFC20B9FCE16945A4E11`.
- Structured project adjudication:
  `docs/methods/2026-07-27-duca-total60-prebackbone-pro-review-absorption.md`.
- Reviewer verdict: `major revision / re-review`; do not start A3/A4 long
  training before clean A0, unique decoder/coordinate contract and the
  applicable hard-utility gate.
- Project verdict:
  `ACCEPT_MAJOR_REVISION / ACCEPT_SCIENTIFIC_CORE /
  HOLD_UNVERIFIED_FORMULAS_AND_THRESHOLDS / CORRECT_STALE_EVIDENCE_STATUS`.
- Current correction: K=192 now has terminal official `57.967272%` evidence,
  so the review's intermediate-only grade is stale. It remains a 90-epoch
  over-budget diagnostic without clean native K=192 uniform.
- Evidence boundary: the review is design and scientific-adjudication
  evidence. Its density constants, exact RDD target, publication thresholds,
  cost ratios and exact arm timing are proposals until frozen before formal
  results.

External sources independently rechecked for this adjudication:

- MGSampler, ICCV 2021:
  `https://openaccess.thecvf.com/content/ICCV2021/html/Zhi_MGSampler_An_Explainable_Sampling_Strategy_for_Video_Action_Recognition_ICCV_2021_paper.html`.
  It uses cumulative motion distribution for explainable video sampling.
- AdaFrame, CVPR 2019:
  `https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html`.
  It learns adaptive frame selection and future utility for recognition.
- Benchmarking TAD Robustness Against Temporal Corruptions, CVPR 2024:
  `https://openaccess.thecvf.com/content/CVPR2024/html/Zeng_Benchmarking_the_Robustness_of_Temporal_Action_Detection_Models_Against_Temporal_CVPR_2024_paper.html`.
  It reports that even sparse temporal corruption can damage TAD, with
  localization a main failure source.
- Hartley, JASA 1966, systematic unequal-probability sampling:
  `https://www.tandfonline.com/doi/abs/10.1080/01621459.1966.10480902`.
  It is a classical cumulative-size systematic-sampling near neighbor.

Additional primary sources checked for the reviewer-defense theory closure:

- Adaptive Keyframe Sampling for Long Video Understanding, CVPR 2025:
  `https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.html`.
  It explicitly combines relevance with coverage, reinforcing that relevance
  scoring alone is not a sufficient novelty or geometry argument.
- Wavelet-based Frame Selection by Detecting Semantic Boundary for Long
  Video Understanding, CVPR 2026:
  `https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Wavelet-based_Frame_Selection_by_Detecting_Semantic_Boundary_for_Long_Video_CVPR_2026_paper.html`.
  It is a training-free semantic-boundary selector and is a required
  contemporary comparison for the frozen-detector route.
- AdapTok, CVPR 2026:
  `https://openaccess.thecvf.com/content/CVPR2026/html/Li_AdapTok_Learning_Adaptive_and_Temporally_Causal_Video_Tokenization_in_a_CVPR_2026_paper.html`.
  It formulates sample-wise dynamic token allocation under a budget, making
  average-budget matching essential for any dynamic-K claim.
- AdapTok official supplement:
  `https://openaccess.thecvf.com/content/CVPR2026/supplemental/Li_AdapTok_Learning_Adaptive_CVPR_2026_supplemental.pdf`.
- AdapTok official code snapshot:
  `https://github.com/VisionXLab/AdapTok/tree/a72076cf6474f930a181aa78971de70d65289b49`.
  Locally inspected primary paths include
  `models/mask_generator.py` (`MaskSampler`, `solve_ilp_min`),
  `models/adaptok.py` (`decode_scorer`, `encode_eval`),
  `models/transformer.py` (`TransformeScorer`),
  `trainers/adaptok_get_scores.py` (multi-budget quality labels),
  `trainers/adaptok_trainer.py` (offline/online scorer training),
  `datasets/video_dataset.py` (score normalization), and
  `eval/rfvd_evaluator.py` (actual-token accounting). The code confirms
  that AdapTok's adaptive allocation is a compound mechanism: nested
  block-prefix masking, score-curve supervision, a learned scorer and a
  batch/global integer budget solver, not an ILP-only trick.

## 2026-07-27 DUCA dynamic-K / AdapTok research-takeover response

- Original attachment:
  `C:/Users/skywalker/.codex/attachments/370a2c39-6571-4a98-af52-1445fd6fc21e/pasted-text.txt`.
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-research-takeover-raw.txt`.
- Raw/archive size: `152,867` bytes; physical lines: `4,589`.
- Raw/archive SHA-256:
  `5ae7850662d726d91c4b3dc7f362ad223d33c35e3cbad9bb87771e939e07e031`.
- Structured independent audit:
  `docs/methods/2026-07-27-duca-dynamic-k-adaptok-research-takeover-absorption.md`.
- Reviewer verdict: `MAJOR REDESIGN — continue`, with dynamic K as a candidate
  part of a hierarchical pre-backbone model, hard marginal TAD utility,
  paired-boundary/high-IoU risk, physical-time exact-K selection and a
  train-calibrated per-video dual policy.
- Project verdict:
  `SUBSTANTIAL_ACCEPT_RESEARCH_DIRECTION /
  MAJOR_CORRECTION_BEFORE_DESIGN_FREEZE /
  DYNAMIC_K_REQUIRED_CANDIDATE_NOT_EMPIRICAL_FACT`.
- Material conflict: lines `203--221` and `779--790` reject strict nested
  physical-frame sets, while lines `2481--2491`, `2738--2798`,
  `2940--2973` and the final method require them. The response therefore
  cannot serve as a frozen mathematical or implementation contract.
- Evidence correction: its initial K=192 `PARTNER_CLAIM` label is stale
  relative to the later post-snapshot project record. The terminal value and
  mechanism summaries are documented 90-epoch diagnostics, still lacking a
  clean native K=192 uniform and still invalid as fair paper support.
- Evidence boundary: the response supplies literature analysis and design
  proposals only. It creates no bounded-density decoder, dynamic Oracle,
  nested-regret result, hard utility gate, RIME checkpoint, fair total-60 mAP,
  second-detector evidence or end-to-end cost result.

Primary near-neighbor sources additionally checked during project audit:

- AdaFocus, ICCV 2021:
  `https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Adaptive_Focus_for_Efficient_Video_Recognition_ICCV_2021_paper.html`.
  A lightweight network reads the full video and guides a high-capacity network
  to task-relevant regions; the paper also discusses temporal skipping.
- AdaFocusV3, ECCV 2022:
  `https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4120_ECCV_2022_paper.php`.
  It jointly allocates spatial-temporal heavy computation and dynamically
  configures per-video cube counts.
- SMART Frame Selection, AAAI 2021:
  `https://ojs.aaai.org/index.php/AAAI/article/view/16235`.
  It jointly selects distributed frames for action recognition, so joint frame
  subset selection is not a DUCA novelty.
- GAP / Post-Processing Temporal Action Detection:
  `https://arxiv.org/abs/2211.14924`.
  It directly identifies fixed-length snippet downsampling and temporal
  quantization as a TAD boundary-resolution problem, making physical-time and
  high-IoU motivation a close neighbor rather than an uncontested first.
- ETAD, Search-Map-Search, TE-TAD and Uni-AdaFocus were already registered
  above and remain required boundary comparisons.

## 2026-07-27 DUCA dynamic-K / AdapTok dual takeover replies

### Source A: DUCA-METER / METER-TAD response

- Source ID: `source:duca-meter-takeover-a-20260727`.
- Original attachment:
  `C:/Users/skywalker/.codex/attachments/8dd661a0-1596-4394-ba09-e293fb3c9169/pasted-text.txt`.
- Byte-identical archive:
  `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-takeover-response-a-2032fca-raw.txt`.
- Size/physical lines: `96,650` bytes / `2,694`.
- SHA-256:
  `2032fcaeddbd4f758ac1be024dd3f867e8dbc6baacd9955de40241ce35595127`.
- Role: proposes DUCA-METER/METER-TAD with strict nested marginal evidence,
  paired-boundary risk, exact-K physical transport and a frozen per-video dual.
  It is a design/reviewer source, not implementation or experimental evidence.

### Source B: MERTAD response

- Source ID: `source:mertad-takeover-b-20260727`.
- Original attachment:
  `C:/Users/skywalker/.codex/attachments/38deddb7-5b11-45e5-9f30-e8ecfe25a557/pasted-text.txt`.
- Byte-identical archive:
  `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-takeover-response-b-e2231c0-raw.txt`.
- Size/physical lines: `122,113` bytes / `2,667`.
- SHA-256:
  `e2231c0928c7dd345a4c7a0cf8b55afe4de95270b710b95602ddd6b5c3fb4bf5`.
- Role: proposes MERTAD/MERTAD-Lite, a paired-branch total-60 contract,
  cross-fitted hard ranking, risk calibration, fixed dual inference and
  explicit fixed-K fallback. It is a design/reviewer source, not implementation
  or experimental evidence.

### Project comparison

- Structured comparison:
  `docs/methods/2026-07-27-duca-dynamic-k-adaptok-dual-response-comparison.md`.
- Verdict:
  `top_level_highly_aligned / executable_spec_not_aligned /
  accept_scientific_core_with_major_corrections`.
- Neither source upgrades `idea:duca-rime` beyond `discussed` or
  `exp:duca-dynamic-k-rime-oracle` beyond `discussed_proposal`.

## 2026-07-28 SparseHead route consolidation sources

### `source:sparsehead-archive-dce2c66`

- Local archive: `E:\DeskTop\TAD\OpenTAD_SparseHeadClean_20260702`.
- Branch/HEAD: `codex/sparse-head-clean-20260702` /
  `dce2c66d1053d53dfcc40b051399cd4c2ecde9ad`.
- Relationship to current repository: no Git merge base; source is inspected
  file-by-file, never whole-tree merged or cherry-picked.
- Seal-time local state: 3 tracked modifications and 22 untracked paths
  (16 configs and 6 `remote_runs` launchers). Binary dirty diff digest:
  `f3e4d66044cc64c634c45d408c412c8a55ea0d6a`.
- Absorbed dirty bridge SHA-256:
  `fb05c10491ddfa6c85ca5183878eee80b33a77b57a27ee74fbac9328e2222a2e`.
  Old audit SHA-256:
  `6d14121989f1dbbad0108b13aea916b633db5f620c9c1189d7df42eb9d47d076`;
  the current copy changes only its repository-root parent depth after moving
  from `tools/` to `tools/bata/`.
- Evidence boundary: the repair configs and assignment fallback were
  uncommitted and unverified. They support diagnostic code preservation only,
  not dense equivalence, mAP, training authorization or a paper claim.

### `source:phystime-e05f6231`

- Source branch/tip:
  `codex/phystime-performance-diagnosis-20260712@e05f623133128c9a4cd56be4656c8fb5099426ac`.
- Core SDPQ provenance: `e6221955` (initial head), `d72948d` (null-evidence
  gate), `996f9288` (native-J metadata), `698ee4be` (evidence-aware
  assignment), `372fcbf` (query-count gate).
- Absorbed surface: PhysTime raw/feature geometry, detector, projection,
  physical and SDPQ heads, native temporal geometry, matched configs,
  launchers/gate, focused tests and canonical experiment records.
- Evidence boundary: tip `e05f6231` contains a source-dtype repair for
  decode-cross replay but no fresh gate, Slurm job or metric. It is an
  implementation source, not proof that decode-cross confounding is closed.

### `source:phystime-g1-full60-0dc5851`

- Commit/tree: `0dc5851a8feb12b97d16bdb5ea8fc60e9273d132` /
  `bddc9b9386604d00d213275a47ce7997b35d3f4c`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1_matched_full60_0dc5851_20260718_112053_+0800`.
- Jobs `1170945/1170946/1170947` completed `0:0`.
- Epoch-59 EMA: selected-axis Avg-mAP `41.28`; physical-metric `57.57`.
- Evidence boundary: matched THUMOS single seed, no SDPQ arm, no complete
  cost/multi-dataset/statistical package. Status is
  `empirically_supported_single_seed`, not `paper_ready`.

### `source:phystime-p0-fullprecision-c2cfcfa`

- Commit/tree: `c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c` /
  `0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_p0_fullprecision_c2cfcfa_20260720_025843_+0800`.
- Jobs `1174688--1174693` completed. Full-precision replay gives selected EMA
  `41.283021`, physical EMA `57.608685`, delta `+16.325664 pp`.
- Evidence boundary: closes the rounding/full-precision NMS post-processing
  confound only; it does not establish SDPQ superiority or paper readiness.

### `source:phystime-decode-cross-approach-a-v4-20260728`

- Clean runtime snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260728_v4`;
  branch `codex/sparsehead-evidence-deploy-20260728`, commit/tree
  `8e31b9e3c08b0a8d320e031b04dfd63e19eb08df` /
  `aae5503424aa3925ef99bba851d600a03e3c3377`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260728_v4`.
- Full-content preflight manifest:
  `preflight_manifest.json`, SHA-256
  `3551816b8e056b9afea4fc9ee8575f525e78ffba64ff087915130b2e10e54712`,
  `validation_pass=true`; reproduced dataset manifest
  `1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2`.
- Immutable model inputs: selected epoch-59 checkpoint SHA-256
  `6fd0781b53e094bb30f0664e006a657fa7c7ef5b3be2de558856c8d23b6bb417`;
  physical epoch-59 checkpoint SHA-256
  `c83a3463155c0a9926a4fc8d62f4d0ee7540c1a58293fb4c3cc9bad8ce9237ed`;
  VideoMAE SHA-256
  `4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251`.
- Submitted Slurm Job `1201048` (`ptdc-a1-full`) on `n16r4/gpu`, one generic
  GPU, six CPUs, six-hour limit. It runs gate → four replays → suite serially
  and fail closed. Terminal state is `FAILED 1:0` after `1m54s` on `g0043`.
- Failure signature:
  `actionformer_native_temporal_geometry_constructor_contract_v1`. Gate
  pre-tests were `39 passed`, then the first model build rejected the missing
  ActionFormer constructor contract. No gate artifact, formal replay, suite or
  new mAP exists. `1201047` remains only a test-only number.

### `source:phystime-decode-cross-approach-a-v5-recovery-20260729`

- Clean recovery runtime:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260728_v5`;
  branch `codex/sparsehead-evidence-recovery-20260728-v5`, commit/tree
  `0338f4777bd02fb327573ef716f54fec76d4af0e` /
  `cb98c64c17d2983c22181d4908c4f31024a82a2f`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260728_v5`.
- Recovery contract: first recorded automatic repair for
  `actionformer_native_temporal_geometry_constructor_contract_v1`; restores
  historical native-J192 ActionFormer alignment without changing experiment
  protocol or immutable inputs. Linux focused suite: `74 passed in 79.28s`.
  This is not a retry-count cap: later confirmed engineering failures must be
  repaired through new immutable roots until final performance is available.
- Full-content preflight manifest SHA-256:
  `77b9918aa3173b73fc71d821defa8c14b3165de1b35f0ae4c0382eeb5d21b43d`,
  `validation_pass=true`; dataset manifest remains
  `1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2`.
- Submitted Slurm Job `1201317` (`ptdc-a1-r1`) on `n16r4/gpu`; `1201316` is
  test-only. The job passed `41` gate pre-tests. `decode_cross_gate.json` reports
  `gate_pass=true`, `all_native_direct_exact_equivalence=true`, all four
  conditions present, and all four raw tensor sets immutable.
- Terminal state is `FAILED 1:0` after `25m29s`. Selected-online direct
  inference displayed Avg-mAP `41.26`, mAP@0.3--0.7
  `64.50/56.39/42.66/27.82/14.90`, `3325` GT and `422000` predictions; the
  producer then failed to provide `pre_cross_window_detections.json.gz`.
  Failure signature:
  `direct_postprocessing_artifact_producer_contract_missing_v1`.
- Evidence boundary: no formal completion exists; the other three replays and
  suite did not start. The partial direct metrics are diagnostic-only and not
  an experiment result.

### `source:phystime-decode-cross-approach-a-v6-recovery-20260729`

- Clean recovery runtime:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260729_v6`;
  branch `codex/sparsehead-evidence-recovery-20260729-v6`, commit/tree
  `ac326ffdc97652433b55ccc596e734b112f51806` /
  `0c58027756997995bda0de6fdd8ec0deb49966d3`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v6`.
- Recovery contract: first repair for
  `direct_postprocessing_artifact_producer_contract_missing_v1`; restores the
  promised pre-cross gzip artifact, post-processing audit and evaluation
  metrics producer without changing experimental protocol or immutable model
  inputs. Linux focused suite: `75 passed in 76.43s`; test-log SHA-256
  `7bf34814a236f56eec3892b829b0eacabbe0cce8ee85c1dc52ef0fa688d9e56f`.
- Full-content preflight manifest SHA-256:
  `97fe5af28b2647396c052c9bdf956997d98e264af74432b57e0fc983b071fb91`,
  `validation_pass=true`; dataset manifest remains
  `1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2`.
- Deployment identity SHA-256:
  `958e273d87228eeb262084b83df2e3cb860b73b8ab781bd57b442d6b7d2faf92`.
  The submission receipt binds Job `1201469` (`ptdc-a1-r2`) on
  `n16r4/gpu`, one GPU, six CPUs, six-hour limit, no dependency; `1201468`
  is test-only.
- Latest verified state: `RUNNING` on `g0030`; gate focused suite
  `42 passed`. The v6 four-condition CUDA gate artifact has SHA-256
  `775e1f2dae70b7863324fd9d235712195dca4d0846968b3bd5e55b754e7b3ea4`
  and reports `gate_pass=true`, `all_native_direct_exact_equivalence=true`,
  all four conditions present and all raw tensors immutable.
- `selected_online` full direct inference completed. Exact metrics JSON:
  Avg-mAP `0.4125660433077075`; mAP@0.3--0.7
  `0.6450446628552113 / 0.5638932489689005 / 0.4266348135535575 /
  0.27820781407261164 / 0.14904967708825695`.
- The repaired real-run producer artifacts are present: pre-cross schema
  `opentad_pre_cross_window_detections_v1` (`211` videos), audit schema
  `opentad_post_processing_audit_v1`, evaluation metrics and epoch-59 result,
  all carrying commit/tree `ac326ffd/0c580277`. Audit-recorded pre-cross
  SHA-256:
  `31e70dc728aff9061f2c56266e3e6d32ef892b227a5c16b15da85e81f731b50e`.
- Job `1201469` is terminal `FAILED 1:0` after `32m32s`. Selected-online
  replay producer completion SHA-256 is
  `0283620a7c5308275c45d03ab1cf639cb8b889d385122d9907fa3e373ef74062`;
  uniform/native Avg-mAP is `0.4125660433077075`, physical-time cross-decode
  Avg-mAP is `0.5015355102106833`. Validator completion assembly then failed
  on unbound `numeric_precision`, signature
  `decode_cross_validator_numeric_precision_scope_v1`.
- Evidence boundary: no formal `DECODE_CROSS_COMPLETE.json`, other three
  replays or suite exists. The selected-online dual-axis values are
  diagnostic-only, not a route verdict.

### `source:phystime-decode-cross-approach-a-v7-presubmit-20260729`

- Clean runtime:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260729_v7`;
  branch `codex/sparsehead-evidence-recovery-20260729-v7`, commit/tree
  `1631d0b60f6552a6f5eb0378d74e766850f34ffd` /
  `f485c8708e22bbbf9a73063d5293a20bc4aa658f`.
- The v6 exact recovery surface plus numeric-precision regression passed
  `76 tests`; full-content preflight passed.
- Deployment failed before `sbatch --test-only` because the expected focused
  test-log SHA-256 omitted its final digit. Failure signature:
  `deployment_expected_sha256_truncation_v1`; failure receipt SHA-256
  `a9c12078c0340cd7f862183d9dae9f828ec2cd3807b0ec8bea290a37168649db`.
  No Slurm job/model forward exists; this source is diagnostic-only.

### `source:phystime-decode-cross-approach-a-v8-recovery-20260729`

- Fresh clean runtime:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260729_v8`;
  branch `codex/sparsehead-evidence-recovery-20260729-v8`, commit/tree
  `1631d0b60f6552a6f5eb0378d74e766850f34ffd` /
  `f485c8708e22bbbf9a73063d5293a20bc4aa658f`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v8`.
- Linux exact recovery suite: `76 passed`; test-log SHA-256
  `16f53fc3cf8a9c5010bce3fd1ed98c4e347add284ba4b2443c00b49b5e107390`.
  Full-content preflight `validation_pass=true`, SHA-256
  `e9f36c221156e5411dad5e3bfe43508b4aa59310539fdbe24da985fc99a27d53`.
- Deployment identity SHA-256:
  `abb8aefc41c24a7d94de5ec0938c42f4ebd17b84f3eff368cd7badbd61d87f22`;
  submission receipt SHA-256:
  `ae43e2744ece2898ca46dba2ad26d943a7524f932df95c71725f380a0b59cac4`.
- `1201494` is test-only. Job `1201495` (`ptdc-a1-r4`) is the only formal
  successor and is `RUNNING` on `g0024`/RTX4090. It retains the single
  allocation serial gate -> four replays -> explicit suite chain.
- Evidence boundary: protocol and immutable model inputs are unchanged.
  Status remains `experiment_running`; no final four-condition result or suite
  verdict exists yet.
- Gate focused tests passed (`43 passed in 29.08s`). Four-condition real-CUDA
  gate artifact SHA-256 is
  `5e323e5ccdedd7dd39d70148aed7108beca94bb5952125a124ad20accfd634f6`;
  it reports `gate_pass=true`, exact native/direct equivalence and immutable
  raw tensors for all four conditions. The job is running selected-online
  direct inference; no v8 formal completion or suite verdict exists yet.
- Selected-online direct inference completed. Exact Avg-mAP is
  `0.4125660433077075`; mAP@0.3--0.7 is
  `0.6450446628552113 / 0.5638932489689005 / 0.4266348135535575 /
  0.27820781407261164 / 0.14904967708825695`. The producer emitted all
  declared artifacts; pre-cross schema covers `211` videos, is bound to exact
  v8 commit/tree, and has SHA-256
  `b4adcf545655424d2b2dfdfce0d107109c5010850143fadf925706fb3de60322`.
- Uniform-rank replay artifacts are complete and physical-time replay is
  running. No formal completion/suite verdict or hard failure exists yet;
  status remains `experiment_running`.

- Terminal update: Job `1201495` ended `FAILED 1:0` after `02:00:24`. Four
  `DECODE_CROSS_COMPLETE.json` artifacts and four producer completions exist,
  but the explicit suite completion does not. Completion/producer SHA-256:
  selected-online `e5a2c64c...b62f3` / `b75620ef...9161`,
  selected-EMA `5a76e12e...273d` / `da6165e9...038a`,
  physical-online `053cdbc2...dd97` / `55938bee...8ff`,
  physical-EMA `81da7980...0f0c` / `877ab9b1...738b`.
- Root cause is JSON container-type mismatch for clean
  `fatal_log_findings`: producer `{}`, consumer `[]`. Failure signature:
  `decode_cross_completion_fatal_log_findings_container_type_v1`. Suite log
  SHA-256:
  `558c78694ae18b9827e4b3cc27f731f3e684faa7eb9a08a1670584c154102919`;
  failure receipt SHA-256:
  `22739defebe8261f61e1fff9910d6d74592d6de4621f7147b07138154ae94d13`.
- Evidence boundary: all v8 metrics remain diagnostic-only because the explicit
  suite did not pass. This source is an engineering-failure root, not a model
  performance verdict.

### `source:phystime-decode-cross-approach-a-v9-presubmit-20260729`

- Partial runtime/run root:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260729_v9`
  and
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v9`.
- Failure signature `runtime_git_author_identity_missing_v1`: the clone lacked
  repository-local Git author name/email, so `git commit` stopped before an
  exact runtime commit, preflight, CUDA or Slurm submission.
- Pre-submission failure receipt SHA-256:
  `ca7f75bc72e85fd466331012775cff72ca14fd685b1db4cc52c8212450c994d2`.
  Slurm jobs created: zero. This source is diagnostic-only and immutable.

### `source:phystime-decode-cross-approach-a-v10-recovery-20260729`

- Clean runtime:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260729_v10`;
  branch `codex/sparsehead-evidence-recovery-20260729-v10`, commit/tree
  `c878fbe3a5e960671f03d93fff8367ed3414f5c5` /
  `8d3e73bb26544d1bcf7bfb61154d0b003f2658e0`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v10`.
- Exact Linux recovery suite `77 passed`; test-log SHA-256
  `7f1787308250a6c9bd62e452f6e16357f5d6bf44cdbcfc6fedd61b7cc63c6936`.
  Full-content preflight SHA-256:
  `f46f6299f7fccc899140ad8fdf001052772ef550dd34cdb68c17d5ba5fc59a8f`.
- Fix scope: clean fatal findings serialize as JSON array, explicit PhysTime
  error markers enter replay log scanning, and the producer/consumer container
  type has a focused regression. Model/config/checkpoint/seed/data/evaluator
  are unchanged.
- Deployment identity SHA-256:
  `1ece7c71b3fc9c396f49401460e5474e3dcaa7ba7f6cf009b987c2b3909a2246`;
  submission receipt SHA-256:
  `9ec33e550d72f69847bcb2a5b2457fad03aa15df54d843e233b2020b5ef5724f`.
- `1203046` is test-only. The sole formal Job is `1203047`
  (`ptdc-a1-r5`), `RUNNING` on `g0050`/RTX4090. Gate focused tests passed
  `44 passed in 29.68s`. The four-condition real-CUDA gate then passed;
  artifact SHA-256 is
  `e5516af02289d15dd1465f5387471bb1a3c357873980d22645c08acbf6aa141c`,
  with exact native/direct equivalence and all four raw-tensor sets immutable.
- Selected-online fresh replay completion SHA-256:
  `a4e727cf094127be7b91a4a13b140463ad9dc3e0c8c1bcfa3acb9887b5ff6dda`;
  producer-completion SHA-256:
  `8a2d38db8a2130a8b617940361a8637dfdc0bff3b6947b0f35d75167a809bfa6`.
  Validator passed with `fatal_log_findings=[]`, same frozen raw tensors,
  native/direct exact equivalence and reviewed-P0 parity. Uniform / physical
  Avg-mAP is `0.4125660433077075 / 0.5015355102106833`; exact threshold rows
  are recorded in the experiment node. This component is `tested`.
- Selected-EMA completion / producer-completion SHA-256:
  `0c6f87617b1cbd6a5bc4a6be6e9a5a2174f8a5a568c2f24db7253c15a315b8dc` /
  `ddddd42174eb987cdeb723ae4422df8105e773bd7af74d31e67760dba20d74ff`.
  Validator passed with `fatal_log_findings=[]`, same frozen raw tensors,
  native/direct exact equivalence and reviewed-P0 parity. Uniform / physical
  Avg-mAP is `0.41283020792762315 / 0.5009785403306161`; exact threshold rows
  are in the experiment node. This component is `tested`.
- Physical-online completion / producer-completion SHA-256:
  `02384da2c71c93bdcd6ce003cd59451510c9d095e222653202f09f38b73b153f` /
  `b9ba401a92e0d828aeabe48cb8972df74a64720a12f160d939daa355856aaf58`.
  Validator passed with `fatal_log_findings=[]`, same frozen raw tensors,
  native/direct exact equivalence and reviewed-P0 parity. Uniform / physical
  Avg-mAP is `0.40107677185286417 / 0.5755558109390063`; exact threshold rows
  are in the experiment node. This component is `tested`.
- Job `1203047` has advanced to physical-EMA direct inference. One component
  completion plus explicit suite remain pending. Its native direct inference
  has completed with Avg-mAP/mAP@0.3–0.7
  `0.5760868491267752 /
  0.7721224901972557/0.7045574192938243/0.6257613932435541/
  0.4900660583199814/0.28792688457926047`; direct-metrics SHA-256 is
  `43c33d551c19f4f3ab90108af30b13c103aa3a875fd87f00a4f50c7e5a83ecac`.
  Replay/validator completion is absent, so this row is `diagnostic_only`.
  Overall status is `experiment_running`; no primary metric claim is allowed.

### `source:phystime-decode-cross-approach-a-v10-terminal-20260729`

- Job `1203047` is terminal `FAILED 1:0`. Physical-EMA completion / producer
  SHA-256 are
  `a5c0c5248bf196d17f1cbf4f11a61d01459cb2ff3cfbf37541046fdb508b7ad1` /
  `8433bd22b620cd60300d94289cf991b69c1f64bcd5eacea557fbc463d7981086`;
  uniform / physical Avg-mAP is
  `0.40296498031949024 / 0.5760868491267752`. Together with the prior three,
  all four replay components are `tested`.
- Explicit suite failure signature:
  `decode_cross_suite_checkpoint_binding_schema_shape_mismatch_v1`. Suite log
  SHA-256:
  `68b7b3d34e587392bdac2df1eb2a36d971009d4c07165ef2a18157449ccb931f`;
  failure receipt SHA-256:
  `42c394f11153a862819876b3915c34ca2ef0a68b6b62ed78a121d65db4269cec`.
- Evidence boundary: same resolved checkpoint path/file SHA; only record metadata
  shapes differ. This source is an engineering-consumer failure, not a route
  performance verdict.

### `source:phystime-decode-cross-approach-a-v11-v15-presubmit-20260729`

- v11 `runtime_profile_source_under_nounset_and_mode_preservation_v1`,
  receipt SHA-256
  `2a95ca48464564d4979754525129414124c769a8a97852a9fad404087bc08545`.
- v12 `recovery_exact_suite_invocation_scope_drift_v1`, receipt SHA-256
  `387d61f33eb3dc055c182a8df23c721378ac4191ad170646311df021fc67e259`.
- v13 `preflight_repo_import_path_unbound_v1`, receipt SHA-256
  `a6d0ccf593e5cb01b9f6a90dee1a47d0042f8ab8a4201be68033b6868fb19858`.
- v14 `deployment_finalizer_base_relative_template_token_mismatch_v1`, receipt
  SHA-256
  `8a85f361fdfa90a6a753c5c3446a43617359cd18ee2a5ea541eac4f6ac00d387`.
- v15 `ssh_transport_interruption_during_exact_recovery_launch_v1`, receipt
  SHA-256
  `f7b7402cc1c565a69a01d057c0e50d7ea63632c3a6a8613be30adc866401630e`.
- All five roots created zero Slurm jobs and are immutable diagnostic sources.

### `source:phystime-decode-cross-approach-a-v16-recovery-20260729`

- Clean runtime/run root:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260729_v16`
  and
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v16`.
  Branch/commit/tree:
  `codex/sparsehead-evidence-recovery-20260729-v16` /
  `54e7f9abeaabf710a505f0a0f595a4eb3bb47f98` /
  `f8490f9c25c2e0e6958c406e19c83cc3d5a40535`.
- Exact recovery `78 passed`, test-log SHA-256
  `d81ca79bd9af216c106fb9718e7b171dd47c9aff3ddecb9787d8e0203c88d0fc`;
  full-content preflight `validation_pass=true`, SHA-256
  `ccc7a83e27b8d18ad0892b644e7338667b72d8eba3e3feedbc387dc4ac1d9a0d`.
- Preparation/env/sbatch/finalizer SHA-256:
  `6c703fcc87abbc5fd03fc87d5028ec3d3335ebe8f822ea07f3961ab19cbb441c` /
  `f8894bb25dbbf62382fda75eaa1e47c9598ecd2de5128526d1ba58c6ce93a5e3` /
  `fa8a223f220d7020202f461984b67aeac43a6facb522c7efd66f562c15769eaa` /
  `768722061cb102a1bbfff3f2b3937d2a5118d2e5e504000280b137e11691fac4`.
- Deployment identity / submission receipt SHA-256:
  `6f22152938b2ad3949a19672e622e97d861a7604f8ff9b5408d59e21bcfcf6d4` /
  `65c325fbd53b3c8386ce459e557f7d8e09f768eb38d77057d8e442b680393ad7`.
- `1203916` is test-only. Formal Job `1203917` (`ptdc-a1-r11`) is `RUNNING`
  on `g0045`/RTX4090. Gate focused tests passed (`45 passed`). Four-condition
  real-CUDA gate artifact SHA-256:
  `0d2153effee84a0e1aa6410125bb291eb4ef4d41e4b40604f49d9e5868e0ada9`;
  `gate_pass=true`, all native/direct exact equivalence, and all four raw-tensor
  sets are immutable. Selected-online direct inference is running; hard-failure
  scan is empty and no completion/suite exists. Source state:
  `experiment_running`.
- Selected-online direct and dual-axis replay producer subsequently completed.
  Direct/uniform evaluation-metrics SHA-256:
  `8860bdcaf3b998e6cddb1187c564d0bb0693496552439b104efad7145a6bd34c`;
  physical-time evaluation-metrics SHA-256:
  `7a032eaf8e4fc776ae0d670d572e02f74c23b82ef55bc29185e796e5be2f0f8b`;
  producer completion SHA-256:
  `97410d9855a3f6db859e36213bf6b201e10c96941a164b5588af02cdfba4ee20`.
  Uniform/native and physical Avg-mAP are
  `0.4125660433077075 / 0.5015355102106833`; producer
  `validation_pass=true`. Formal component completion and suite remain
  pending, so this source is still `experiment_running`.

- v16 formal selected-online component:
  `selected_online/DECODE_CROSS_COMPLETE.json` SHA-256
  `6937fc6b7b050fd7009ee967ceef446aebaa8b3daa695c7959106ff87048c038`.
  The receipt binds commit/tree
  `54e7f9abeaabf710a505f0a0f595a4eb3bb47f98` /
  `f8490f9c25c2e0e6958c406e19c83cc3d5a40535`, records
  `status=tested`, `validation_pass=true`, `fatal_log_findings=[]`,
  frozen-raw/native-direct/reviewed-P0 parity and `new_training=false`.
  Job `1203917` then entered `selected_ema`; three formal components and the
  explicit suite remain pending, so route state remains `experiment_running`.

- v16 formal selected-EMA component:
  direct/uniform metrics SHA-256
  `ed3750a61a27dc70ac570f29ccefff8eef8d4dc10ea29802743b403807b82a34`;
  physical-time metrics SHA-256
  `742b9a810f52dfe9bd12c29987148bf3c95e99c58aefb5774f2f8b3d18d30c1b`;
  producer/formal completion SHA-256
  `43c737fe3c5a9a534c565bf63e419fa152ee35b3be796ddf3f601c954fa52877` /
  `4a1b405b7849f396e1b649da8895070e6176023c4a959c6d7fd9148f2bd8afe0`.
  Uniform/native and physical Avg-mAP are
  `0.41283020792762315 / 0.5009785403306161`.
  The formal receipt records `status=tested`, `validation_pass=true`,
  `fatal_log_findings=[]`, frozen-raw/native-direct/reviewed-P0 parity and
  `new_training=false`. Job `1203917` then entered `physical_online`; two
  formal components plus the explicit suite remain pending, so route state is
  still `experiment_running`.

- v16 physical-online producer evidence:
  direct/physical-time metrics SHA-256
  `b68f2ad1393b59c40d58f7cfa1e450a52f84d8acbc80ad785a2d3a31352d6009`;
  uniform-rank cross-decode metrics SHA-256
  `0c258e563fe7b9886e6d56c9c3370b6536e187b521526318622b07ffcf1e4a4b`;
  producer completion SHA-256
  `d61d8fbf8b977b59b65eb87d55227904b2a5a2e6994e584226bda19a265b26eb`.
  Physical/native and uniform Avg-mAP are
  `0.5755558109390063 / 0.40107677185286417`. Producer validation passes, but
  formal `physical_online/DECODE_CROSS_COMPLETE.json` is absent. This source is
  `diagnostic_only` until the formal validator writes its receipt; route state
  remains `experiment_running`.

- v16 formal physical-online component:
  `physical_online/DECODE_CROSS_COMPLETE.json` SHA-256
  `fd18348e6ae6ecf4bdc4390ca4620a109616582f7f77138ed137085e0df6c260`.
  The receipt binds the v16 commit/tree and records `status=tested`,
  `validation_pass=true`, `fatal_log_findings=[]`,
  frozen-raw/native-direct/reviewed-P0 parity and `new_training=false`.
  Job `1203917` then entered `physical_ema`; the fourth formal component and
  explicit suite remain pending, so route state remains `experiment_running`.

- v16 formal physical-EMA component:
  `physical_ema/DECODE_CROSS_COMPLETE.json` SHA-256
  `cd6da2f827524e0b9eb2b46c6cbbcc5b6e89243aa9cd8d7e45efafcb4cb6b565`;
  producer completion SHA-256
  `aa6356a509898b94a38f2b9e0548c5f647cc6498655697b37fd39ea8982fc733`.
  Uniform/physical metrics SHA-256 are
  `5058f789de9fd74544427fd8201d7b32cc83f18524409ee9e8f3b96fe32292dc` /
  `43c33d551c19f4f3ab90108af30b13c103aa3a875fd87f00a4f50c7e5a83ecac`;
  Avg-mAP are `0.40296498031949024 / 0.5760868491267752`. The receipt binds
  v16 commit/tree and records `status=tested`, `validation_pass=true`,
  `fatal_log_findings=[]`, frozen-raw/native-direct/reviewed-P0 parity and
  `new_training=false`. All four formal components are now `tested`, but the
  explicit suite artifact and terminal Job `1203917` state remain pending, so
  the route remains `experiment_running`.

- v16 terminal Slurm record: Job `1203917` (`ptdc-a1-r11`) is
  `COMPLETED 0:0`, elapsed `02:34:30`, node `g0045`. Runtime branch/commit/tree
  and clean worktree match the registered deployment identity; hard-failure
  scan is empty.
- Explicit evidence suite completion:
  `DECODE_CROSS_EVIDENCE_SUITE_COMPLETE.json`, schema
  `phystime_decode_cross_evidence_suite_completion_v1`, SHA-256
  `ed2770c35cf9a3acd5fa80465eda1c34b3541ba3dea404c75388aaeffefbdc31`;
  `status=tested`, `validation_pass=true`, `new_training=false`,
  `fatal_findings=[]`, 13 logs scanned.
- Independent suite marker:
  `DECODE_CROSS_EVIDENCE_SUITE_VALIDATED.json`, SHA-256
  `f2da143127b3a01aef7bda451e2351c494f72552f3810f604f895f4c0a7767d3`,
  `validation_pass=true`.
- The suite binds source/runtime commit and tree, full-content preflight, CUDA
  gate, P0 gate/suite, all four formal completion SHA-256 records and checkpoint
  state-dict identities. This source is now `tested`; it is not a multi-seed,
  independent-evaluator, cost or paper-ready source.
- Pro-analysis interpretation is registered in
  `exp:phystime-frozen-decode-cross-replay`: the within-checkpoint decode-axis
  intervention is causal because the raw tensors are sealed, while
  cross-checkpoint selected/physical differences remain descriptive. Missing
  class/calibration/NMS/failure-sample/assignment-support/cost fields are
  explicitly not supplied by this source.

### `source:actionformer-official-pin-61ea7eb-20260729`

- Repository: <https://github.com/happyharrycn/actionformer_release>.
- Exact commit/tree:
  `61ea7eb9308a568b0cf45e3804830836e30061de` /
  `7b06c5261ba244788c942a0d73e304581bc35154`.
- `configs/thumos_i3d.yaml` SHA-256:
  `73f8aeaf7deef93aba57259badd4c454990ec1e0ce6eaa7c3434db44baaeeaf0`.
- `README.md` SHA-256:
  `bdee4eb088a74e190935097742c7dbfaf254eb912f79729dccd73b9b36b33db8`.
- Official THUMOS archive MD5:
  `375f76ffbf7447af1035e694971ec9b2`.
- Pinned evaluator file hashes include `eval.py`
  `525d859ff0ae9dfcee3c91b3fd96227cbd67d0774f4ed062f196a1b888fafcc4`,
  `libs/utils/metrics.py`
  `b937a20f8ee06d43669eef57d12a01708d68eed6937bfd9074fd764a7551a535`,
  `libs/utils/nms.py`
  `b3234a72126cf82ace87b1653d85438a425d6c5a3947f7f68f9ad61e7c83ba42`
  and `libs/utils/train_utils.py`
  `a05ddd1c9493f3190833e0b12fb673688cecebc61beb46c9d1aa643364e61e1e`.
- The release documents the current checkpoint result
  `82.13/77.80/70.95/59.40/43.87`, average `66.83`; it is an official anchor,
  not a matched delta against raw-VideoMAE SparseHead.

### `source:sparsehead-diagnostic-closure-57917e7-20260729`

- Isolated branch/commit/tree:
  `codex/sparsehead-diagnostic-closure-20260729` /
  `57917e7bf2b991478b4f6fc4ce1db5ca5878b68d` /
  `aaf7c82bd837078bb7276baf6c0a504da0684194`.
- Exact base:
  `54e7f9abeaabf710a505f0a0f595a4eb3bb47f98` /
  `f8490f9c25c2e0e6958c406e19c83cc3d5a40535`.
- Added independent frozen-artifact evaluator, 64-window SDPQ support audit,
  official ActionFormer record builder, fail-closed comparability classifier,
  design and focused tests.
- Local verification: Python compilation, staged diff check and four focused
  suites, `35 passed`. This is implementation/tool evidence only; no remote
  completion, official reproduction or new model metric exists.

### `source:sparsehead-diagnostic-closure-6d74ad7-20260729`

- Branch/commit/tree:
  `codex/sparsehead-diagnostic-closure-20260729` /
  `6d74ad7b7c7736bbff48976a626b951512a54e96` /
  `80cd2431ebf9809f03ab1216b84b45380d51f33b`.
- Local/Linux focused verification:
  `46 passed, 1 skipped` / `58 passed, 1 skipped`.
- Added fail-closed valid-prefix/NaN-padding handling, explicit OpenTAD logical
  test-to-annotation-validation binding with 211-video/3,325-GT/20-class
  checks, and explicit expected checkpoint epoch/state-key binding for the SDPQ
  audit.
- Independent failed roots:
  `.../runs/sparsehead_diagnostic_closure_20260729_v1` and `_v2`; active clean
  successor `_v3`.
- SDPQ exact diagnostic source: clean config repo commit/tree
  `4a57577193c07cc90ac0867176aa79c76f637c36` /
  `2d9ae007b7d9cea179a9ec5e08a82bf01ef4cf4c`, config SHA-256
  `21ba537b007f416050db41bf9cb19a07a145f10d8f4a6e8623d8f405b67ca26e`,
  epoch-19 online checkpoint SHA-256
  `40fccfd854a88903aaf795c04b94068af4007663c5d63064201990d70b2c3fc7`.

### `source:actionformer-official-resources-20260729-v3`

- THUMOS release archive:
  `/data/run01/sczc063/yuzibo/datasets/actionformer_official_61ea7eb_20260729_v3/thumos.tar.gz`,
  MD5 `375f76ffbf7447af1035e694971ec9b2`.
- Released checkpoint/log ZIP (download filename retains `.tar.gz`):
  SHA-256
  `e028f7e487713d0c68f0515ba9bdafda0ed05fc1271b9999ea995652b034c929`;
  it contains `thumos_i3d_reproduce/epoch_034.pth.tar`,
  `thumos_reproduce_log.txt` and `thumos_reproduce_results.txt`.
- Upstream runtime remains pinned to commit/tree
  `61ea7eb9308a568b0cf45e3804830836e30061de` /
  `7b06c5261ba244788c942a0d73e304581bc35154`.
- Resource integrity alone is not a benchmark result. Official save-only raw
  predictions, official evaluation, independent recomputation and strict
  comparability verdict remain required.

### `source:sparsehead-diagnostic-closure-2b07484-20260729`

- Branch/commit/tree:
  `codex/sparsehead-diagnostic-closure-20260729` /
  `2b074845497f6ada3314cb895f0d4ab2f4ce3eca` /
  `7779862c5422dc8e527b304bf881a760b0c90625`.
- Bundle SHA-256:
  `a97dc6d61e4e6a4e4fb6734fd7d3c724ada5ae8f88ea4d7ef29903fc80037686`.
- Exact Linux runtime:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260729_v8`;
  focused result `95 passed, 1 skipped`, log SHA-256
  `265046cd7fc3b1e847e87880e061a5a76092c4b194d1d4e727ca706f5b8884b6`.
- It seals official class/split/feature/raw-prediction/evaluator evidence,
  base-anchored matched-source attestation and the SDPQ padded-query mask
  regression. Status: tools `tested`; no model result.

### `source:actionformer-official-repro-job1205131-diagnostic-20260729`

- Preserved run:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_repro_20260729_v4`.
- Diagnostic official-evaluator output:
  mAP@0.3–0.7 `82.13/77.81/70.95/59.40/43.87`, Avg `66.83`;
  raw prediction SHA-256
  `1333df9202eec7ae217542b6bd2b15b597c1a004ebb3634de54a7a37adb6d7fe`.
- Raw file contains 42,400 predictions over 212 videos; exact ID-set SHA-256
  `7543da7a293c941bf19c388ecb92b7bd2520904cbfd704e60275acb53691490d`.
- Job exited nonzero only because the superseded record builder required a
  literal nominal split shape; failure signature
  `official_annotation_split_schema_contract_v1`. This source is
  `diagnostic_only`; successor Job `1205178` must pass the strict verdict before
  any paper-main-table claim.

### `source:sdpq-support-job1205132-failure-20260729`

- Preserved run:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sdpq_support_observability_20260729_v6`.
- Failure signature:
  `sdpq_support_overlap_query_padding_mask_omission_v1`.
- Exact source remains config commit `4a57577193c07cc90ac0867176aa79c76f637c36`,
  epoch-19 online checkpoint SHA-256
  `40fccfd854a88903aaf795c04b94068af4007663c5d63064201990d70b2c3fc7`,
  seed 42 and 64 sealed windows.
- This records an implementation-correctness failure, not model performance.
  Unique repaired successor is Job `1205179`.

### `source:actionformer-official-eval-job1205206-unsealed-20260729`

- Released official-checkpoint evaluation produced mAP@0.3–0.7
  `82.133988/77.805571/70.953608/59.401673/43.872118`, Avg
  `66.833392`.
- Raw prediction SHA-256:
  `1333df9202eec7ae217542b6bd2b15b597c1a004ebb3634de54a7a37adb6d7fe`;
  exact 212-video ID-set SHA-256:
  `7543da7a293c941bf19c388ecb92b7bd2520904cbfd704e60275acb53691490d`.
- The old receipt recorded seed `0`; the official config and train log specify
  `1234567891`. Status: `tested` numerical source, not main-table eligible
  until a fresh strict reseal.

### `source:sdpq-support-job1205240-20260729`

- Completion SHA-256:
  `abf28cf420f0e2e06b3d727e9da92c98f55fba626f334cd73c6b4c4cb3ee1167`.
- All 647 GT matches across 64 sealed windows have assignment, support and
  domain evidence; missing/collision/uncovered counters are zero and maximum
  offset error is `3.0517578125e-05`.
- Status: diagnostic observability `tested`; no performance claim.

### `source:independent-recompute-job1205243-failure-20260729`

- Preserved run:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260730_v8`.
- Failure signature: `independent_recompute_semantic_match_drift_v1`.
- Raw scores/masks and proposal geometry were exact and all metric-delta signs
  matched. The independent validator differed in stable/float64 sort and
  Soft-NMS semantics from PyTorch `2.0.1` CPU unstable sort and scalar
  float32 C++ Soft-NMS. Status: engineering failure, not model evidence.

### `source:sparsehead-diagnostic-closure-e2a0d74-20260729`

- Branch/commit/tree:
  `codex/sparsehead-diagnostic-closure-20260729` /
  `e2a0d74f561b158c531d4909e72ecee69b153c16` /
  `0b6cb7996ee90f3209a78b78bbf7a55525e3badd`.
- Exact Linux runtime:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v16`.
- Exact suite: `127 passed, 2 skipped`; log SHA-256
  `115dd497a3a662b3fc0f19ae9104257d245cbadbb7fd4001f3eb3ea71432534c`.
- Implements hardened official source/effective-config gates and independent
  pinned sort plus scalar float32 Soft-NMS/`expf` semantics. Status: tools
  `tested`; no model result.

### `source:independent-recompute-job1205388-failure-20260729`

- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260730_v9`.
- Submission receipt SHA-256:
  `bf27a72af865c7db4148912df9b3fbdc75530fba01b0aae006953f844054fbcb`.
- Uses exact audit commit/tree `e2a0d74f561b158c531d4909e72ecee69b153c16` /
  `0b6cb7996ee90f3209a78b78bbf7a55525e3badd`; test-only ID `1205384`
  is not a job.
- Terminal state `FAILED`, exit `127:0`, elapsed `00:00:01`. Failure signature
  `slurm_module_function_unavailable_v1`: the non-login allocation lacked the
  shell `module` function, and neither the model nor validator started.
- Failure receipt SHA-256:
  `f4f2b305be639575310dc290accbc88d381812902b3edd090a9137438f7a0359`.
  Status: preserved engineering failure, not model evidence.

### `source:independent-recompute-job1205400-running-20260729`

- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260730_v10`.
- Uses exact runtime commit/tree
  `e2a0d74f561b158c531d4909e72ecee69b153c16` /
  `0b6cb7996ee90f3209a78b78bbf7a55525e3badd` and direct activation of
  the pinned Conda environment after an empty-environment probe.
- Sbatch/submission receipt SHA-256:
  `21372695291fdf8089f93920665a9ac844f4ee21ca9da07dcb5a6c95df9dd506` /
  `da679424ad5a3dbfd3cc0b6e28fd74b638d2bc7873098a4d1c46a2e80c14bea2`.
  Test-only ID `1205398` is not a job.
- Status: `experiment_running`; no result before terminal state and formal
  completion validation.

### `source:actionformer-official-anchor-job1205409-failure-20260729`

- Fresh run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_anchor_reseal_20260730_v1`.
- Official source commit/tree:
  `61ea7eb9308a568b0cf45e3804830836e30061de` /
  `7b06c5261ba244788c942a0d73e304581bc35154`; audit commit/tree:
  `e2a0d74f561b158c531d4909e72ecee69b153c16` /
  `0b6cb7996ee90f3209a78b78bbf7a55525e3badd`.
- Sbatch/submission receipt SHA-256:
  `4a377d0d580c6baae2b50a277a2c9f04ce0d5470719bcf32354005775e14cfa0` /
  `87f901b944fe5a1054cd9ae168336c024d7d8142407fc322b25956163ec1b68d`.
  Test-only ID `1205408` is not a job.
- Terminal state `FAILED`, exit `1:0`, elapsed `00:00:36`. Signature
  `official_environment_probe_nms_import_order_v1`: NMS build/ABI passed, but
  the probe imported the extension before `torch` loaded `libc10.so`; official
  inference never started.
- Failure receipt SHA-256:
  `2d4df6637af61f39f0d516eeba519da9fcaff73289e0b73ff55f2bbf2c841af6`.
  Status: preserved engineering failure, no metric.

### `source:actionformer-official-anchor-job1205419-failure-20260729`

- Fresh run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_anchor_reseal_20260730_v2`.
- Official/audit commit and tree identities, checkpoint, train log, seed
  `1234567891`, effective config, 15 receipts and strict main-table gate are
  unchanged from the predecessor. The only change restores the proven
  `torch`-before-NMS-extension probe order.
- Sbatch/submission receipt SHA-256:
  `bced7838ee244613222c236fd9393baf40eb09aa8e6a0d7021042ebb95597777` /
  `3a70fd84de376f377ce34d0085ea11772ca0fefaa8223b4d2d495edf0a69f03a`.
  Test-only ID `1205418` is not a job.
- Terminal state `FAILED`, exit `1:0`, elapsed `00:01:42`. Both official
  evaluation passes reproduced Avg-mAP `66.83`, then the record builder exposed
  `official_released_train_log_default_serialization_omission_v1`: the released
  log omits the exact upstream default `model.fpn_start_level=0`.
- Raw predictions SHA-256:
  `1333df9202eec7ae217542b6bd2b15b597c1a004ebb3634de54a7a37adb6d7fe`;
  failure receipt SHA-256:
  `079818253bc87a78ed67ce41dbd092aa64f0e54b5a61972f2313adeb7d10fa4a`.
  Status: preserved engineering provenance failure; numerical output is not
  main-table eligible.

### `source:sparsehead-diagnostic-closure-8b80c98-20260729`

- Branch/commit/tree:
  `codex/sparsehead-diagnostic-closure-20260729` /
  `8b80c98ee2af65561bf305b4fdc2ef16e460da73` /
  `148a93eac4ff1b6a3be46fdca72c705aa17294a6`.
- Pins released raw train-log effective-config SHA-256
  `ad426e1a25be48423e21f854bbc6d815c6063388811350ad5fada5ac8933d3a7`,
  permits exactly one documented missing upstream default
  `model.fpn_start_level=0`, and hashes both raw and normalized identities.
- Local focused result: `44 passed, 1 skipped`. Remote v17 GitHub clone failed
  with a TLS transport termination; clean exact runtime v18 was frozen from
  complete Git-bundle SHA-256
  `2a8ab74ff5f7b6c5ebeb6bb97fccc7030d901a71cc8cc1c24210a6560fcbe2e1`.
  Linux full preflight passed `131 passed, 2 skipped`, log SHA-256
  `6899bf6126d1ce9b3d880d348cdf5c1f152235d3b2e6f6de028b5fc807fb34fb`.
  Status: audit tooling `tested`; no model result.

### `source:actionformer-official-anchor-job1205455-20260729`

- Fresh run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_anchor_reseal_20260730_v3`.
- Official commit/tree remain `61ea7eb9308a568b0cf45e3804830836e30061de` /
  `7b06c5261ba244788c942a0d73e304581bc35154`; audit commit/tree are
  `8b80c98ee2af65561bf305b4fdc2ef16e460da73` /
  `148a93eac4ff1b6a3be46fdca72c705aa17294a6`.
- Sbatch/submission receipt SHA-256:
  `76fc3df0c1faadfc9f62fb2982a8aee6013fc4a6a502cd32dbdf3e27fd7ec0a7` /
  `8f26fbed8284f83d6d099779f88c76d61ee0181323d208f355aff10dbb426744`.
  Test-only ID `1205454` is not a job.
- Model/protocol settings are unchanged; the only recovery is the exact
  upstream-default normalization attestation.
- Slurm terminal state is `COMPLETED 0:0`. Official independent mAP@0.3–0.7 is
  `0.821339880697554/0.7780557086995361/0.7095360789567791/`
  `0.5940167327663141/0.4387211844309326`, Avg
  `0.6683339171102232`, from 42,400 predictions.
- All 15 receipts pass. Completion/protocol/verdict SHA-256 values are
  `90c8bae14fcb20cc2434cea37f47065704766e38ff9663eac6e70c0d338b9e94` /
  `808199b54b0ebcfebda403419873cc5fd46c36a4d404d3d8ce31838ce3b5bd95` /
  `0706247ef978bf339f9a9cb4adaef07500e8d991129c6d0862118088b917a2ec`.
  Verdict: `official_actionformer_protocol_match=true`,
  `main_table_eligible=true`; status `empirically_supported`.

### `source:phystime-decode-cross-job1203917-complete-20260729`

- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v16`.
- Job `1203917` completed `0:0`; all four formal completion artifacts and the
  explicit evidence suite validate.
- Suite completion/validation/deployment SHA-256:
  `ed2770c35cf9a3acd5fa80465eda1c34b3541ba3dea404c75388aaeffefbdc31` /
  `f2da143127b3a01aef7bda451e2351c494f72552f3810f604f895f4c0a7767d3` /
  `bc825f08445e4c8fe8f3ab5dd768b6f9cdf3ec7fdd40dc02438428237c004b2e`.
- Status: `tested`; frozen single-seed inference decode-axis evidence only.

### `source:actionformer-native-k384-candidate-55763a9-20260729`

- Candidate branch/commit/tree:
  `codex/actionformer-sparsehead-official-matched-20260730` /
  `55763a9ef7ce18a51827fe48040081c4fe2b84d4` /
  `c489a54aa501b39421cddb5df98385b3889ed479`.
- Clean remote runtime:
  `/data/run01/sczc063/yuzibo/projects/actionformer_sparsehead_official_matched_20260730_v2`;
  Linux focused suite passed `11`.
- Full official I3D input is retained. The explicit method contract is
  `model.query_budget=384`, deterministic stratified-uniform native-grid
  selection, original physical FPN geometry and
  `training.loss_support=selected_native_grid_queries`.
- This is not unchanged-loss/execution-only. Released-checkpoint evaluation is
  diagnostic; matched official dense/sparse retraining is required for paper
  eligibility. Status: implementation `tested`, performance untested.

### `source:sparsehead-k384-audit-aab72e4-20260729`

- Audit branch commit/tree:
  `aab72e484538931a565930b99d1beb71f47b9ceb` /
  `25e7e0eb3b8cd5edfb48eac594eda6b89edffa36`;
  clean remote runtime:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v21`.
- Local audit/launcher focused suites passed `40` / `5`. Remote candidate,
  audit and source-diff SHA-256 receipts are
  `1174f5b4036458a598d20e06913fef9bf2561ded1b2c9f8c675e627304be6b3b` /
  `5890bde1cbdc8ad72edabcfe48ffff36d4d1028c81eab67cd1db11b3e2a25b39` /
  `409ffd3035a0c957d3b250db24fe017c5c09efda526d746ace0d54f00c695abc`.
- The validator pins official source/config/checkpoint identities and measures
  selected-output equivalence, unselected zeros, immutable tensors/masks and
  isolated head-path cost. `wall_clock_claim_allowed=false` by design.

### `source:actionformer-native-k384-cuda-job1205541-running-20260729`

- Unique formal Slurm Job `1205541`; test-only ID `1205539` is not a job.
- Immutable run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_native_grid_k384_cuda_gate_20260730_v1`.
- Deployment/submission receipt SHA-256:
  `b37a08c2031bb7b043038ea6baf022830bda4ca1203abbff41619401537e8a8e` /
  `471022b2e726cf923e5a445aef8c21ca5f17c9e59b7e586ed8fb3ed4bbc49665`.
- Last observed status at registration: `PENDING (Priority)`. This is an
  engineering correctness/isolated-cost gate with no model metric claim.

### `source:actionformer-native-k384-cuda-job1205541-failure-20260729`

- Job `1205541` terminal state: `FAILED 2:0`, elapsed `00:00:30`; immutable
  run root is the v1 root registered above.
- Correctness passed with maximum error `4.0531158447265625e-06`, immutable
  tensors/masks and zero unselected outputs. Cost failed at dense/sparse-
  preselected/sparse-with-selector means
  `6.191821/19.650751/20.504609 ms`, speedup `0.3009129x`.
- Failure signature:
  `native_grid_sparse_head_microkernel_launch_and_scatter_slowdown_v1`.
  CUDA gate/failure-analysis SHA-256:
  `8aeb2cdbf02da0f8ad675b2f5a33d3ef6d89198ac7e216511ffde45d66f505a3` /
  `ef6b462d79316e2c3f80bf125eb8704b30c0c3e229568048b67095a172152b7d`.
- Status: engineering failure; no model or paper metric.

### `source:actionformer-native-k384-candidate-d64e66d-20260729`

- Candidate commit/tree:
  `d64e66dfd7fc9881552b342f5523926cc78c0848` /
  `16265c70b235034acb52521b00c259ec6d8b59e1`; clean remote runtime:
  `/data/run01/sczc063/yuzibo/projects/actionformer_sparsehead_official_matched_20260730_v3`.
- The implementation packs samples and FPN levels into one convolution per
  head layer while retaining exact physical/mask/autograd semantics. Candidate
  Linux suite passed `12`.
- Audit/launcher suites passed `40` / `1`; source-diff SHA-256 is
  `5aea817bf1fd1b2c0e36193b9d99ee71bde3dfd00c05673ece5dc4f6da9304d4`.
- Status: implementation `tested`; cost and model performance unresolved.

### `source:actionformer-native-k384-cuda-job1205567-running-20260729`

- Unique formal Job `1205567`; `1205566` is test-only.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_native_grid_k384_cuda_gate_20260730_v2`.
- Candidate/audit identities are `d64e66d...` / `aab72e4...`; deployment and
  submission SHA-256 values are
  `c2890c1b37e22810fdc8284b80ca6292e7bf5cc1c38820fb74e8d68d96647b52` /
  `f71c394c09f5d5a65bdf37036739294d553098ea3ecfa79b8ebf10c8486b3798`.
- Last observed state: `PENDING (Priority)`. Claim boundary:
  engineering gate only, no model metric.

### `source:actionformer-native-k384-cuda-job1205567-failure-20260729`

- Job `1205567` terminal state `FAILED 2:0`, elapsed `00:00:28`; immutable run
  root is the registered v2 root.
- Correctness passed; dense/sparse-preselected/sparse-with-selector means were
  `6.240900/12.765710/13.618182 ms`, selector-inclusive speedup `0.4590397x`.
- Signature:
  `native_grid_sparse_head_packed_patch_materialization_and_microconv_slowdown_v1`.
  CUDA-gate/failure-analysis SHA-256:
  `f4a0479b48c434832c45d84e9eccc6ebc9e56be88a03d8e8eff4fca525981113` /
  `fe2f6d62272ad558be18e068ca1796808d516b105b2ed41202eb5a7e0e1fb6d6`.
- Status: engineering failure; no model or paper metric.

### `source:actionformer-native-k384-candidate-31e6112-20260729`

- Candidate commit/tree:
  `31e6112ea28747098cfe5412c097d737731bfaa1` /
  `d2619cd075c4e7192ca060f34d811ac3fe5768f8`; flattened GEMM is algebraically
  equivalent to the packed length-three Conv1d. Linux suite: `12 passed`.
- Exact clean runtime:
  `/data/run01/sczc063/yuzibo/projects/actionformer_sparsehead_official_matched_20260730_v6`.
  Bundle SHA-256:
  `c50bea0b79e242bb4c96cf11fb35a3ef095a8b9c3bc4a13fc56abca02be4ec49`.
- Preserved setup signatures:
  `github_https_clone_tls_termination_v1` and
  `bundle_clone_remote_head_unset_v1`.
- Status: implementation `tested`; cost failed and model performance is
  unresolved.

### `source:actionformer-native-k384-source-diff-local-31e6112-20260729`

- Remote live-ref lookup failed as
  `github_remote_ref_dns_timeout_during_source_diff_v1`.
- Clean local live-ref source-diff attestation/provenance SHA-256:
  `3ef485f82678453538aef6f58ba81d548149394ef93356a811593e67cdf22e9d` /
  `780c0aa5a8a00ba9180974d4bee001782e83d747d492589a2a2da4b5bc40e2d6`.
- Claim boundary: engineering gate only,
  `paper_main_table_seal_allowed=false`; live remote recomputation remains
  mandatory for a paper row.

### `source:actionformer-native-k384-cuda-job1205569-failure-20260729`

- Job `1205569` terminal state `FAILED 2:0`, elapsed `00:00:28`; run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_native_grid_k384_cuda_gate_20260730_v3`.
- Correctness passed at maximum selected error `4.5299530029296875e-06`,
  immutable inputs/masks and zero unselected outputs. Dense/sparse-preselected/
  sparse-with-selector means were `6.193413/12.764605/13.546889 ms`; speedup
  `0.4576558x`.
- Failure signature:
  `native_grid_sparse_head_packed_gather_scatter_overhead_v1`.
  CUDA-gate/stdout/stderr/failure-analysis SHA-256:
  `7e91345babcce40bb9a157d2b29fbc718fe7f0e2a059bdc02e2edff386709197` /
  `fb3abb66cf7690fd7965165d409039ea1701ac3ab0c4027e4f94652863373afa` /
  `4c6d87aa6b85dbbe173a1eae119bac562aa43f55c6fa847b256c9c05d25c79e0` /
  `8b49859031a48ef2a4367a156f452761c66a1e75c1a5e6a87a8fb242766f3a50`.
- Status: engineering failure; no model or paper metric. The next global
  packed-state prototype is only `designed`.

### `source:actionformer-native-k384-global-packed-d86a4ac-20260729`

- Candidate commit/tree:
  `d86a4acda21e35a1609f19f1a46bc470ee18b7e1` /
  `327c032a1ab3c14d0e34d6339df36f8a33ec6907`; exact clean remote v7.
- Audit commit/tree:
  `14bd14f9b6a087dc2ec623fc4238c89e0cb86960` /
  `b782404ddef9a65f19fea70fbf993a3a9d6e0420`; exact clean remote v22.
- Candidate/audit focused suites pass `14/18`. Source-diff attestation/
  provenance SHA-256:
  `68d2cf726cc8523094847337eb5ebe604ca5ad46cbe99d2a5fba2b78f45e67db` /
  `e57fc3d618f86faacbd79cb77121796a18fed5fafdc0fe80506e40f9aba6237c`.
- Status: implementation `tested`; provenance permits engineering gate only,
  not a paper row.

### `source:actionformer-native-k384-cuda-job1205571-complete-20260729`

- Unique formal Job `1205571`; `1205570` is test-only. Terminal state:
  `COMPLETED 0:0`, elapsed `00:00:30`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_native_grid_k384_cuda_gate_20260730_v4`.
- Full-content preflight/deployment/submission SHA-256:
  `08b05123edbaccd10d5b43031a43ebac11a3616ceb454bfbd588d4d7395a6a95` /
  `f070f46f023be6152faf1818342633a8d6f713fb55e37fa5c79fc2a43434f140` /
  `04d206c3ad220155f8f63a1b6a086c6c3c6c5beaeac13a7a001334f2d0fef4c7`.
- Dense / sparse-preselected / sparse-with-selector median latency:
  `6.240573 / 3.129646 / 3.970906 ms`; selector-inclusive median speedup
  `1.571574x`, with all synchronized rounds passing.
- Gate/completion/runtime-log SHA-256:
  `cddfb80af237a41d3c3e1121e39cbc5114ad8abc472c56f6daf519a50cf95988` /
  `ceec00f799eb40a1dd56c1949576783e06599205d63f1d1909a598787d99fd85` /
  `f3f4b13be3433d2307ce10a8370ab168d8af00368060e61229441e27131cb0f5`.
- Status: `tested`. Claim boundary: isolated-head engineering gate only;
  `paper_metric_claim_allowed=false` and
  `end_to_end_wall_clock_claim_allowed=false`.

### `source:actionformer-native-k384-remote-live-source-diff-20260729`

- Candidate repository/commit/tree:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git` /
  `d86a4acda21e35a1609f19f1a46bc470ee18b7e1` /
  `327c032a1ab3c14d0e34d6339df36f8a33ec6907`.
- Official repository/commit/tree:
  `https://github.com/happyharrycn/actionformer_release` /
  `61ea7eb9308a568b0cf45e3804830836e30061de` /
  `7b06c5261ba244788c942a0d73e304581bc35154`.
- Remote live-ref attestation and independent live-validation log SHA-256:
  `a07d038d87632d1f8cc984ba24af44ca7ce9a9902e30e501f5de80a32265d46b` /
  `80c261c0e417d73c979a8a4de2f55b0176fc8cd7e7952190a2b47247280f7b47`.
- Status: `tested` paper-record provenance component; it does not contain a
  model metric.

### `source:actionformer-official-data-live-revalidation-20260729`

- Fresh v2 recomputation hashes all 413 official I3D files and exactly matches
  sealed feature-manifest SHA-256
  `cda269dace393b9af1f6fcb87a9a531beed69e3c71279ba3ca2cee76e198d59a`.
- Annotation SHA-256:
  `3b025685a07fb98fc58d2399fb5fa9493c2168632d8ae1a8c3f4689897d2fbb2`;
  validation log SHA-256:
  `59045c8e77a071b1e8003eec0c6a8941a986220ee9a95f0006e04d4a865b1395`.
- Preserved failed v1 signatures:
  `official_data_live_revalidation_import_scope_v1` and
  `preflight_failure_receipt_python_environment_unloaded_v1`; receipt SHA-256
  `2cd20095d49566761ed8feb16af7989d96cbe57d2b5441f10e12fa2504ababde`.
- Status: successful v2 is a `tested` data-provenance component; v1 contains no
  model result.

### `source:actionformer-official-matched-pair-job1205573-running-20260729`

- `1205572` is test-only; unique formal Job `1205573` was last observed
  `PENDING`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_matched_pair_k384_seed1234567891_20260730_v1`.
- Candidate commit/tree:
  `d86a4acda21e35a1609f19f1a46bc470ee18b7e1` /
  `327c032a1ab3c14d0e34d6339df36f8a33ec6907`; audit commit/tree:
  `643c42e8cfe4018fb891202f7ffdae554acc2e4a` /
  `25fa3eda9fc62960c69c2952c957ebab39e71c27`.
- Full-content preflight/deployment/submission SHA-256:
  `3b827cfe10b3267d013373f89a9c3b90b2eb6f450b0aa4b7d1e5082615a0ac4e` /
  `65cb544960c619f4243c7829a41950719d2591493c05fbad70a07f1b9a037da2` /
  `ead6f35af71e2de9308d6ed0aad642dc27845e68169f1cee8ca32e3d157a3e77`.
- Protocol: same candidate commit, seed `1234567891`, official data, 5+30
  epochs, explicit epoch-35 EMA, identical evaluator, independent raw
  prediction recomputation, no resume.
- Status: `experiment_running`, single-seed screening;
  `paper_main_table_eligible=false`.

### `source:actionformer-official-matched-pair-job1205584-failure-20260729`

- Terminal state/elapsed: `FAILED 1:0`, `00:09:48`.
- Stage: `dense_saveonly_eval`; dense completed all 35 epochs, sparse did not
  start and no metric was produced.
- Dense terminal checkpoint SHA-256:
  `ea3c16fcf17fd6fb8cec57829804e96736a8ab231b07d820e5939fd5db3cba00`;
  status `diagnostic_only_not_reusable_by_successor`.
- Signature:
  `official_actionformer_softnms_extension_abi_shadowed_by_opentad_v9arg_v1`.
- Conflicting OpenTAD extension SHA-256:
  `4ccea1d7bae60a3edb735280c564928f18e89bd01e160a1c9fa200625a660450`.
- Failure-analysis/saveonly/runtime SHA-256:
  `99f83a03715fa935a422451f9fe842aeaae867546d37c9af39cda8869958f852` /
  `496468bf5c327ae0a31a3a581cc086fd7cfb69dd5d2b249b088acc6e8aee7338` /
  `b3f8cca479ad22a674a433badcabb9d928b012af7b60221ec11f8e54e5bf6cc5`.
- Status: engineering failure only; `model_result_claim_allowed=false`.

### `source:actionformer-sparsehead-official-main-table-prereg-20260729`

- Node:
  `research-wiki/experiments/actionformer-sparsehead-official-main-table-prereg-20260729.md`.
- Frozen before Job `1205584` metrics after five independent reviews.
- Paired seeds:
  `1234567891/1423812477/737690612/1788897292/1322022747`; canonical JSON
  SHA-256
  `a4038a752aa46b97e5854c20574d65ece078bad6124e4778cc4269e75747c7c6`.
- S0 GO bounds: Avg delta `>=-1.00 pp`; @0.6 and @0.7 deltas each
  `>=-1.50 pp`.
- Main accuracy-preserving bounds: five valid paired seeds; Avg 95%-CI lower
  bound `>=-0.20 pp`; @0.6/@0.7 lower bounds each `>=-0.50 pp`.
- Cost bound: official precomputed-feature detector pipeline, median speedup
  `>=1.05x`, lower CI `>1.00x`, no duration stratum crossing unity.
- Required attribution: full/selected training support x dense/K384
  evaluation-query 2x2 cross.
- Status: `designed`; not deployed and not empirical evidence.

### `source:actionformer-official-matched-pair-job1205580-failure-20260729`

- Terminal `FAILED 1:0` after 26 seconds; no optimizer step ran.
- Failure signature:
  `official_declared_tensorboard_dependency_missing_v1`.
- Failure receipt SHA-256:
  `a959ef415f383d5368edf806b1166cca9cd25e91e49ea4398853775059e35385`.
- The source/offline validation and focused tests passed before official
  `train.py` raised `ModuleNotFoundError: tensorboard`.
- Status: engineering environment failure only; no model metric.

### `source:actionformer-tensorboard-environment-recovery-a3d9879-20260729`

- Audit commit/tree:
  `a3d987961c0e6ac0166194cfc30ca0d375765ef1` /
  `51c53773d266e614d6c1054a1e6127fe73c69f38`.
- Runtime:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v26`;
  bundle SHA-256
  `e8812a84489bb55aea419b1b637778574539a44b0c7399b18a04d346430ce419`.
- Environment:
  `/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_tensorboard_2_20_0_20260730_v1`;
  receipt SHA-256
  `acc5909360970cfad1f390a4f5ab046a3876ac9378448b2f94da26ffb312ece2`.
- Versions: Python `3.10.20`, torch `2.0.1`, CUDA `11.8`, NumPy `1.23.5`,
  TensorBoard `2.20.0`; SummaryWriter RNG-immutability probe passed.
- Remote focused suite: `19/19`, log SHA-256
  `f18a52300731975c81d0fffa1cd4c8e5787ccc83b07abba212d4d2a1f6fcbb7c`.
- Status: `tested` engineering recovery; official scientific protocol
  unchanged.

### `source:actionformer-official-matched-pair-job1205584-running-20260729`

- `1205583` is test-only; unique formal Job `1205584` was observed `RUNNING`
  on g0024.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_matched_pair_k384_seed1234567891_20260730_v3`.
- Candidate commit/tree remain
  `d86a4acda21e35a1609f19f1a46bc470ee18b7e1` /
  `327c032a1ab3c14d0e34d6339df36f8a33ec6907`.
- Full-content preflight/deployment/submission SHA-256:
  `9ff27367e10717b012d0f06a85b980f54c9b91a6fe45be9e8f87c00cac90d47b` /
  `00736c6b07fff77e0a6ca92ad24744eab0e2c089a22b350f9f2537054891b4f4` /
  `f4512010b2d675611f97e61a929ee4edda421b7f29506969d49028b3a7ac041a`.
- Candidate `14/14` and audit `4/4` focused tests passed before dense training.
- Status: `experiment_running`, single-seed screening;
  `paper_main_table_eligible=false`.

### `source:actionformer-official-matched-pair-job1205573-failure-20260729`

- Terminal state `FAILED 1:0`, elapsed `00:00:31`; training did not start.
- Failure signature:
  `compute_node_github_dns_during_redundant_live_source_diff_revalidation_v1`.
- Failure/runtime/stdout/stderr SHA-256:
  `f0bf8fe6258260d55fffe88d35dfb75d647340adccb06dc2efae1c5e419c64d9` /
  `8bc85a66f37bc98eec780ec76ef5fab1978bd45195917780c269267dc5b2a057` /
  `8bc85a66f37bc98eec780ec76ef5fab1978bd45195917780c269267dc5b2a057` /
  `fef8ce4b812cf04882328f4f12a5ddcac8c61077a1f8107c19b63e142808d74b`.
- Status: engineering failure only; no model metric.

### `source:actionformer-official-matched-pair-offline-recovery-debbde4-20260729`

- Audit commit/tree:
  `debbde469f938e09e4debfe7831e64755ae665f5` /
  `3721612aae55eecb07e9f4183a53e1d8156e143b`.
- Exact clean v25 bundle SHA-256:
  `6c59f1d568017d8ee82e32d3132b595b73c3d469a2cb91976968d330cd789104`;
  partial v24 is preserved after
  `github_https_clone_tls_termination_during_audit_runtime_freeze_v1`.
- Recovery tests/offline snapshot validation SHA-256:
  `d3d76af3095d792b6af0a8709a7e83addca17aa8e1d5e4d36a13b9cc8d9856f7` /
  `f409abc67b630fbc6c1b30db7ba5e614ecb8925a2d5c7aa6b0e9d7746581067b`.
- Status: `tested` engineering recovery; scientific protocol unchanged.

### `source:actionformer-official-matched-pair-job1205580-running-20260729`

- `1205579` is test-only; unique Job `1205580` was last observed `PENDING
  (Priority)`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_matched_pair_k384_seed1234567891_20260730_v2`.
- Candidate commit/tree remain `d86a4ac...` / `327c032a...`; audit
  commit/tree are `debbde4...` / `3721612...`.
- Recovery preflight/deployment/submission SHA-256:
  `45e60ba0f68132b8cfa11ec036ed71789e83d718dc300df62f0cdf19f1375e8a` /
  `a151cf03c67395771eb386c6fe48687e867b40df1d8f7a562be6d1df459125a0` /
  `fca38a1cad01222ef8bda967116993742319bdc94b2d8e9582a783abe21c479f`.
- Status: `experiment_running`, single-seed screening;
  `paper_main_table_eligible=false`.

### `source:actionformer-official-nms-runtime-recovery-71f955a-20260730`

- Audit commit/tree:
  `71f955a7301f07875a35e0be366241e548e5c775` /
  `d328093644e040741e16dbdd8bc93b6b0d608a10`.
- Exact clean v27 bundle SHA-256:
  `a9ee267333c9371d087e806fe61cef19c14122b18fee1a4e6c75fa4c58846ad6`.
- Runtime/environment receipt/NMS extension SHA-256:
  `/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_official_runtime_20260730_v2` /
  `13d57c1161905f059204f7101f26029503a03da7f5eb44b81c418a0b97999f24` /
  `b67e0e41f9f55cd69e8b90cfc75a1947214365857d851a510047838ad49ed98d`.
- Remote candidate/audit test-log SHA-256:
  `f0ae7ae24f8439ba95aed967799db6d50acc8d50f401ae3d2a60480aa6693936` /
  `42aa27fe2cd2eed1543ebbf0ad635ab3a3941254ddff5778d3f2f599f3e8a16d`
  (`14+5` passed).
- Live 413-feature rehash receipt SHA-256:
  `73a2f714c100f541306d7d7f9c32e36481574d2ac6c5e78925ee4ee1dcca96b3`;
  official/candidate dense-config SHA-256:
  `c0ac0df560cd564941b56cd9391ad0bd5cea386d2e4b6cf9fc8ffcab821955cd`.
- Status: `tested` engineering/protocol recovery; scientific conditions
  unchanged.

### `source:actionformer-official-matched-pair-job1205594-running-20260730`

- `1205593` is test-only. Unique formal Job `1205594` was submitted and last
  observed `PENDING (Priority)`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_matched_pair_k384_seed1234567891_20260730_v4`.
- Full-content preflight/deployment/submission SHA-256:
  `d9e1f897de51e46aac52cb450f72daa8bc19a64bf999b01112013489038d4a55` /
  `b8d4079c9ddc8faa7a0a575dbe63f700c2448409df5dbccf972101cc0e4a282b` /
  `2a31a1d01056f39159d17d99fb9047f5bd6946b68475c1eae31008659df07a08`.
- Both arms restart from scratch at official seed `1234567891`, official
  `validation`/`test` split, 5+30 schedule and terminal epoch-35 EMA.
- Status: `experiment_running`, official-comparable single-seed screening;
  `paper_main_table_eligible=false`.

### `source:actionformer-official-matched-pair-job1205594-failure-20260730`

- Terminal `FAILED 1:0` after four seconds at `python_environment`; no test,
  optimizer step or metric ran.
- Failure signature:
  `official_environment_probe_nms_import_order_v1`.
- Failure/failure-analysis/runtime/stderr SHA-256:
  `68d2ec8ddd1d2a69c1181d532325368975c95a306a2c7d368226905044ee321f` /
  `06bbc29e5f57b3b9a12f421f5ddd814487bf01733d0f0e5bbcc4c0551c877a41` /
  `5988ed65e4ebbd8dde6a334ffe7f2ae3c8825fbd5e61d1c6198388c0284443fb` /
  `4c6d87aa6b85dbbe173a1eae119bac562aa43f55c6fa847b256c9c05d25c79e0`.
- Status: engineering import-order failure only; no model result.

### `source:actionformer-official-nms-import-order-recovery-98f5b87-20260730`

- Audit commit/tree:
  `98f5b875315b4a2b5c6829f5d74ccce68f478e47` /
  `2e6b4bba6868c323d70c97140f7cbed044eb1a7b`.
- Clean v28 runtime:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v28`;
  bundle SHA-256
  `713a1d839e8e8ea50f141df9dba1feb44dc43c91dffbd4dd85bf8910bbdf9e24`.
- The launcher now loads `torch` before the official NMS extension; the focused
  regression binds this dynamic-loader precondition in addition to path/hash
  and seven-argument ABI.
- Remote ordered-import/seven-argument probe log SHA-256:
  `7d79381ed64b27059aa6f4204bbfce3f606fc1e81e0a7962e4e1d1c7413a0488`.
  Candidate/audit focused suites pass `14+5`.
- Status: `tested`; scientific conditions unchanged.

### `source:actionformer-official-matched-pair-job1205599-running-20260730`

- `1205598` is test-only. Unique formal Job `1205599` was last observed
  `RUNNING` on g0030.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_matched_pair_k384_seed1234567891_20260730_v5`.
- Preflight/deployment/submission SHA-256:
  `19230f06e0eda57c34607db250dba9ebc1f0d6365e5ab33c339dffe0468ddd86` /
  `250068a1de36c00fabe37596e302dc9e3fd22249be09b267fc4e9762e6f4ce46` /
  `0549ff04a30bb4efea176a484a6f51d652b8bdd023227564b0fc2fdfe492cabf`.
- Candidate/audit commit/tree:
  `d86a4acda21e35a1609f19f1a46bc470ee18b7e1` /
  `327c032a1ab3c14d0e34d6339df36f8a33ec6907` and
  `98f5b875315b4a2b5c6829f5d74ccce68f478e47` /
  `2e6b4bba6868c323d70c97140f7cbed044eb1a7b`.
- In-allocation environment/source and `14+5` focused gates passed; dense
  training completed and sparse training started.
- Dense ARM/independent-attestation SHA-256:
  `a15b0526ef9a75a0fe32c0798b609c738781ab5c063c53df165ace6cbcdf138a` /
  `59a9d037faf0418e226f184b87c66d484c7a64b81a911692ba65d44c1cc195d7`.
  Exact 212-video/42,400-prediction Avg-mAP is `0.6658301251307708`, with
  mAP@0.3–0.7
  `0.8190849486121916/0.7795203466370499/0.7128549836803181/0.5825550463357125/0.43513530038858167`.
- Status: `experiment_running`, official-comparable single-seed screening;
  `paper_main_table_eligible=false`.

### `source:actionformer-official-matched-pair-job1205599-complete-negative-20260730`

- Slurm Job `1205599` (`af-k384-pair-r5`) completed `0:0` on g0030 in
  `00:19:21`. Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_matched_pair_k384_seed1234567891_20260730_v5`.
- Pair completion SHA-256:
  `545e420aa1d437aedeffd15cb30390ceb0cfe4d6565d7eb35c53a8bf17ac76fd`;
  `validation_pass=true`, issues empty and all comparability flags true.
- Dense/sparse ARM SHA-256:
  `a15b0526ef9a75a0fe32c0798b609c738781ab5c063c53df165ace6cbcdf138a` /
  `fc682cfb01b9ed6639f821938922051edc2afa55490f504170eb7e3a6fd49037`.
  Independent metric SHA-256:
  `59a9d037faf0418e226f184b87c66d484c7a64b81a911692ba65d44c1cc195d7` /
  `499b0e7f34b854ca6c9915a95d0c522106dda859079a50a7c4f28bbd6ede65ba`.
- Dense/sparse terminal checkpoint SHA-256:
  `ea3c16fcf17fd6fb8cec57829804e96736a8ab231b07d820e5939fd5db3cba00` /
  `511bc98b8a5cf2649ef1ad1808b750f5402352a637cd7a9b8c75386e578cf6cd`.
  Raw prediction SHA-256:
  `a9cb1e2a42bc966126120f76f8227b8b5ba4114b0db362026cc4095fc6ae39c8` /
  `27962ac4f169c6a6f6c30f710be843c3071922c4ce8fa039151b3d2f0aa72cae`.
- Both arms use epoch-35 EMA, no resume, seed `1234567891`, exact official
  validation/test data, 212 evaluated videos and 42,400 predictions.
- Dense/sparse Avg-mAP is `0.6658301251307708/0.4391969933812866`;
  mAP@0.3–0.7 is
  `0.8190849486121916/0.7795203466370499/0.7128549836803181/0.5825550463357125/0.43513530038858167`
  versus
  `0.6492524848028804/0.5664284467904844/0.4595264064199946/0.32783176623331417/0.19294586265975905`.
- Verdict: legal official-comparable S0 negative;
  `KILL_CURRENT_K384_SELECTED_LOSS_INTERVENTION`.
  Experiment status `tested`; frozen intervention-level rejection
  `empirically_supported`; `paper_main_table_eligible=false`.

### `source:actionformer-sparsehead-s0-negative-analysis-20260730`

- Dedicated analysis:
  `research-wiki/experiments/actionformer-sparsehead-s0-negative-analysis-20260730.md`.
- Six independent reviews considered structural query coverage, selected-loss
  optimization/normalizer drift, calibration/NMS and implementation/evaluator
  explanations. Combined coverage plus supervision damage is the current
  leading explanation.
- Next authorized work is no-retraining 2x2 cross-evaluation and frozen raw
  prediction/assignment diagnostics. No new training, rescue tuning,
  multiseed or cost claim is authorized.
- Status: `designed` diagnostic follow-up over an `empirically_supported`
  negative S0 result.

### `source:actionformer-sparsehead-s0-attribution-job1205701-complete-20260730`

- Slurm Job `1205701` completed `0:0`. Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_sparsehead_s0_negative_attribution_20260730_v1`.
- Attribution/negative-diagnostics/suite SHA-256:
  `d0bffe87cfb582b1b0649da3833e9fe0147db5a0a78500b6700fb78019323afb` /
  `a6b7fa0c4a41aac75ae2fb4cb4fcfbe68cf48bc7d2c813b37485b35998838791` /
  `e71721cb07334f1b6abb09347a7b609e51d6da1ed4be864c190ed60433a197d6`.
- Full×dense/full×K384/selected×dense/selected×K384 Avg-mAP:
  `0.6658301251307708/0.4578433218148983/0.6453734282343033/0.4391969933812866`.
- K384 execution/selected-loss/interaction Avg effects:
  `-20.7082/-1.9552/+0.1810 pp`.
- Status: `tested`; no new training or seed;
  `paper_main_table_eligible=false`.

### `source:actionformer-sparsehead-s0-assignment-audit-465b2bc-20260730`

- Audit implementation commit/tree:
  `465b2bc284d5c3b62ec9e21023052b5eabddf260` /
  `da1e515398017345deb4c39d98751ade0a8aa8db`.
- Clean remote runtime:
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v30`.
- Local/remote focused suites and full-content preflight passed. Preflight
  SHA-256:
  `ac6b8e265a54d06d49e0e0461242f8a194751928b8c9b093621041a99087056d`.
- Status: `implemented` before formal execution.

### `source:actionformer-sparsehead-s0-assignment-job1205799-complete-20260730`

- Slurm Job `1205799` (`af-k384-asg-r1`) completed `0:0` on g0063 in
  `00:00:41`. Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_sparsehead_s0_assignment_support_20260730_v1`.
- Suite/producer/rows/sample-seal SHA-256:
  `475b61ddad4b0b56a86b2e2616ef2584b252c3169b4ad1268223f21d6e118567` /
  `ca7e97a4124e49eb2ac30e949bcd50d4407998e8518eb72c8c6c8c8bb3f86e8b` /
  `a73b6f69c8655fed584774d131388ebf4974cf001f3efd9f492a952251e96b7f` /
  `d02f1de5fe9320cea47011b4af253001db77ecb7aadff83b8185a3350c7c55f4`.
- Exactly 64 official `validation` training windows, 804 GT, no test GT,
  training or model selection. Dense/K384 positives are `2721/461`;
  K384 has `395` GT without candidates and `427` without assignments.
- Status: `tested`; diagnostic-only; `paper_main_table_eligible=false`.

### `source:actionformer-sparsehead-s0-integrity-audit-20260730`

- Source and receipt audit:
  `research-wiki/experiments/actionformer-sparsehead-s0-integrity-audit-20260730.md`
  and machine-readable JSON companion.
- Real dataset GT, official ActionFormer evaluator/config, raw-score AP,
  frozen artifacts and actual launcher invocation pass.
- Overall verdict: `WARN`, because the official result is one paired seed and
  configured external cross-model file reviewers were unavailable. No metric
  integrity failure was found.

### `design:actionformer-sparsehead-dcsr-official-prereg-20260730`

- Frozen design/protocol:
  `research-wiki/experiments/actionformer-sparsehead-dcsr-official-prereg-20260730.md`.
- DCSR keeps a cheap dense native-grid proposal/support scaffold and applies
  sparse expensive residual refinement. Unselected queries keep scaffold
  outputs.
- The final official study requires five fixed paired seeds, same-run dense
  controls, paired uncertainty and synchronized complete
  feature-to-final-detection cost. Status: `designed`; no result exists.

### `implementation:actionformer-sparsehead-dcsr-g0-g1-bf0df83-20260730`

- Branch/commit/tree:
  `codex/actionformer-dcsr-g0-g1-20260730` /
  `bf0df83d7400c89fc61f38d169d68085420a2263` /
  `2f9346fcfd2bfb7fc5a76a86ef65545030a67469`.
- Clean N16R4 runtime:
  `/data/run01/sczc063/yuzibo/projects/actionformer_dcsr_g0_g1_20260730_v5`.
- Linux focused/module-entry suite: `31 passed`.
- Implements official-identity G0 and cheap-dense-scaffold plus sparse signed
  residual G1. Status: `implemented/tested`; no performance result.

### `source:actionformer-dcsr-g0-job1206168-complete-20260730`

- Slurm Job `1206168` completed `0:0` on RTX 4090.
- Receipt path:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_g0_gate_20260730_v5/seed_2026073001/DCSR_G0_EQUIVALENCE.json`.
- Receipt SHA-256:
  `b87fc59ec6529e83e99f7bf5fbfb7f3bff5ec637060c62057da07a669a8c1ff4`.
- All state-key, point, full-mask, pre-decode and final official
  Soft-NMS/timestamp exact checks pass. No test GT/predictions or
  metric/efficiency claim. Status: `tested`.

### `source:actionformer-dcsr-g1-job1206273-running-20260730`

- Formal Slurm array: `1206273_[0-2]`; `1206266` is test-only.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_g0_g1_internal_20260730_v5`.
- Validation-only 160/40 manifest SHA-256:
  `ba683bc5ddbb1fe219fab0545e9d808808d9b25fc9b32e7c5c0b6339b68b9bbb`.
- Frozen dev seeds:
  `2026073001/2026073002/2026073003`.
- Status: `experiment_running`; internal architecture gate only;
  `paper_main_table_eligible=false`.

### `source:actionformer-dcsr-g1-job1206273-complete-negative-20260730`

- Formal array `1206273_[0-2]` completed for all three frozen seeds.
- Aggregate SHA-256:
  `b98d59468ef39aa6fe6de387adfd6f872c848ab8f63b26c3bf1bf6161f5f7939`.
- Pair SHA-256 values:
  `c924ed997a438f14e3d4660906635e2cca90b34b8ac2d2dc7c4170df2a4a5867`,
  `a2ecb27e8485c10fe97a5319b930c1e7d49d5f918f02c64269ec6281d87f88da`,
  `9b85c1f38f5ecb4ba7fbb9e60c39c9e6005dc8201d828aa79cce0a365cdfcd40`.
- Mean DCSR-minus-dense:
  `-7.556202 pp` Avg and `-11.043134/-11.019821 pp` at 0.6/0.7.
- Status: `tested`; exact G1 rejection `empirically_supported`; validation-only;
  `paper_main_table_eligible=false`.

### `implementation:actionformer-dcsr-negative-diagnostics-8d6f6e5-20260730`

- Branch/commit/tree:
  `codex/actionformer-dcsr-g0-g1-20260730` /
  `8d6f6e5e7fcf8c27b6aa46870bc4c0b242f6314b` /
  `1ac5a68c6b8d0b1c9028ea3154765ae20e87622a`.
- Clean N16R4 runtime:
  `/data/run01/sczc063/yuzibo/projects/actionformer_dcsr_negative_diagnostics_20260730_v3`.
- Linux suite: `38 passed`; source runtime clean and exact.
- Implements no-training scaffold-only/all-query replay, prediction diagnostics
  and checkpoint dynamics. Status: `implemented/tested`.

### `source:actionformer-dcsr-negative-diagnostics-job1207441-complete-20260730`

- Three counterfactual tasks and aggregate Job `1207441` completed `0:0`.
- Run root:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_negative_diagnostics_20260730_v3`.
- Completion/prediction/checkpoint SHA-256:
  `954d7944428fcf0d26dd917ff9562a9c3e7a53de71c09e9a382aaf49f5bd4a53` /
  `47dcca7e179544e348966bf92cf92cddeff19a1fdc8cfea100150dc1bc580a36` /
  `c596bc942d2617e3824d21c96d0289316be4ee1ad465f23dc507b2d90466e006`.
- Scaffold-only/all-query/K384 minus dense Avg is
  `-7.418076/-6.316665/-7.556202 pp`; K384 support penalty is
  `-1.239537 pp`.
- Status: `tested`; no training/test subset/paper row/efficiency authority.

### `source:actionformer-dcsr-g1-negative-analysis-20260730`

- Analysis:
  `research-wiki/experiments/actionformer-sparsehead-dcsr-g1-negative-analysis-20260730.md`.
- Ranks weak scaffold/decomposition first, residual support second and
  optimization/zero-init third, with support, counterevidence, falsifiable
  predictions and minimal new-route experiments.
- Status: exact G1 rejection `empirically_supported`; universal sparse-head
  rejection unsupported.

### `source:actionformer-dcsr-g1-integrity-audit-20260730`

- Audit Markdown and machine-readable JSON:
  `research-wiki/experiments/actionformer-sparsehead-dcsr-g1-integrity-audit-20260730.md`
  and `.json`.
- Scientific integrity: `PASS`; official paper comparability: `FAIL`; overall:
  `WARN`.
- External Claude/Gemini/GPT-4o/MiniMax reviewer routes were unavailable due
  credentials. Local independent evidence/code reviews completed.

### `design:actionformer-odfcr-internal-factorial-20260731`

- Preregistration:
  `research-wiki/experiments/actionformer-odfcr-internal-factorial-prereg-20260731.md`.
- Candidate design branch/commit:
  `codex/actionformer-densefloor-factorial-20260731` /
  `77244d5`.
- Full implementation specification:
  `docs/superpowers/specs/2026-07-31-actionformer-official-dense-floor-factorial-design.md`
  in the isolated ActionFormer candidate repository.
- Frozen internal matrix: scaffold depth `1/3` × residual
  `off/all_valid`, three new paired training seeds and a new holdout-v2 drawn
  only from the prior train-160. K384 is frozen replay only.
- Two independent reviews covered model identifiability and split leakage.
  Design status is frozen at `77244d5`.

### `implementation:actionformer-odfcr-01cdb78-20260731`

- Branch/commit/tree:
  `codex/actionformer-densefloor-factorial-20260731` /
  `01cdb78d2b7668098b6b13a1e49433d48fbc1a8d` /
  `e70d2956a197b1204e721239178e76152efe282b`.
- Linux focused suite: `71 passed`.
- Holdout-v2 SHA-256:
  `b8cac555f3d31e02468dbca3b3b0ada2d30b05bf046c10eb16304abb92499d1a`.
- Deployment receipt SHA-256:
  `ee5ae82d7f7bfbf6aa3e67615136e99c207bb643d0b23908ed4e5b596ea5ac5d`.
- Status: `tested` for static/Linux and real-CUDA G0 implementation contracts;
  no trained-arm metric or paper authority.

### `source:actionformer-odfcr-job1209259-running-20260731`

- Formal factorial array: `1209259_[0-2]`; unique G2 successor:
  `1209267` with `afterok:1209259`.
- Three real-CUDA G0 SHA-256:
  `212835e56ddbd7538ec14173f52e6fa1323adef82b11ff446429b3b314cbbfc7`,
  `7b6aa7a00583d59000c9eb3e4cdaac1145b9cd92643e66bc72c8a3fa85978aa6`,
  `25ac7539f8b9e86bf66b5eace8b0905f5420a763364afb0857f2ba152b5a68f8`.
- Each has `gate_pass=true` and all 14 exact checks true. All three tasks entered
  `d1_off` training.
- Status: `experiment_running`; internal validation only, no arm metric,
  official test, paper row or efficiency claim.

### `source:actionformer-odfcr-g2-complete-negative-20260801`

- Formal tasks `1209259/1209260/1209261` and G2 Job `1209267` completed `0:0`.
- G2 aggregate path:
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_odfcr_internal_20260731_v3/aggregate_g2/ODFCR_G2_AGGREGATE_COMPLETE.json`.
- Aggregate SHA-256:
  `9172eddcbf5f9a4943b303e20b57f4492f0a44b18c39f892d5829b1f0a79ddec`.
- Matrix receipt SHA-256 for seeds `01/02/03`:
  `c4c7bcf3c63314e3c436bc1d9bb903b05350f7dfafb1e1c0e0cb8aabfc0f409f`,
  `397ce608487a27afd3e9a94c773717a720bedef07e80b94a93016e53af393549`,
  `46bca957890f44e2cd03d7b132c5087bf58713b210cbca38d4c686352e0ac698`.
- Mean `d3_all-d3_off=-0.180645 pp`; per-seed deltas
  `-0.696973/+1.507030/-1.351994 pp`; @0.6/@0.7 deltas
  `-2.746761/+0.596011 pp`. `residual_utility_gate_pass=false`.
- No K384/G3 job or completion artifact exists. Status: `tested`, exact G2
  rejection `empirically_supported`; internal validation only.

### `source:actionformer-odfcr-terminal-analysis-20260801`

- Analysis is recorded in
  `research-wiki/experiments/actionformer-odfcr-internal-factorial-prereg-20260731.md`
  and result-to-claim trace `.aris/traces/result-to-claim/2026-08-01_run01/`.
- Inputs: twelve attested post-NMS prediction files, 40 holdout videos, 454 GT
  instances, train logs/TensorBoard scalars and the validated G2 aggregate.
- Findings: depth-three floor improves Avg `+7.5600 pp` and high-IoU recall;
  all-valid residual on that floor is null/negative with heterogeneous class,
  duration and video effects despite lower late training loss.
- Limitations: no pre-NMS suppressed proposals, probability calibration,
  residual/gate/activation/gradient telemetry, independent holdout, test data or
  synchronized cost. No paper/official/efficiency claim is authorized.

### `review:duca-query-bridge-pro-20260820`

- Exact Project-scoped external review session: `duca-project-query-review`;
  requested/resolved model: Pro; status: `completed`.
- Browser transcript:
  `C:/Users/skywalker/.oracle/sessions/duca-project-query-review/artifacts/transcript.md`,
  SHA-256 `e293399465904e2b9151ace704c6286f41bcde17e29d2d884ccd4245ba807d04`.
- User-provided review copy:
  `C:/Users/skywalker/.codex/attachments/69e5ae9f-e9a2-4afb-afd2-7c7e9eb9bb63/pasted-text.txt`,
  SHA-256 `f1de2f65a6b5f8bb1c2ba70bb462f1e1787c5eda22afb8567495b6816d8b40df`.
- Structured project absorption and local source checks:
  `research-wiki/DUCA_PRO_REVIEW_ABSORPTION-2026-08-20.md`.
- Verdict: `SUBSTANTIAL_ACCEPT / NOT_FULL_ACCEPT`. The semantic-indirect,
  fixed-K-first diagnosis is accepted; proposed block geometry, loss weights,
  update schedule, K set and numerical gates remain hypotheses. The review's
  statement that no new training existed is superseded by verified UVT/Fovea
  development-job records; those results remain non-matched diagnostics.

### `review:duca-semantic-indirect-external-text-20260820`

- User-provided external review text:
  `C:/Users/skywalker/.codex/attachments/cd13a5e4-b2db-4934-9d02-8f6a75b4decb/pasted-text.txt`,
  SHA-256 `fe19d0e776dc59d526f96f00ed62fad5609afd876518e4563820728ee58d475b`.
- No model/session/raw transcript accompanied the pasted text; it is therefore
  recorded as an external proposal, not an independently authenticated model
  verdict.
- Comparison, code checks, state correction and conditional absorption:
  `research-wiki/DUCA_SECOND_REVIEW_COMPARISON_AND_ABSORPTION-2026-08-20.md`.
- Verdict: `SUBSTANTIAL_ACCEPT / NOT_FULL_ACCEPT`. It agrees with the prior
  review on semantic-indirect, fixed-K-first, physical-time and variable-compute
  constraints. Its clean-cliplet runtime, semantic label, update schedule, K
  set, numerical gates and ordering differences remain unverified design choices.

### `source:actionformer-odfcr-monitor-retired-20260801`

- Heartbeat automation: `sparsehead-official-matched-monitor`.
- Retirement followed validated matrix, G2, attribution and claim-trace
  completion. The app self-delete RPC timed out repeatedly from its own active
  heartbeat, so the exact configuration was moved recoverably to
  `C:/Users/skywalker/.codex/automation-archive/sparsehead-official-matched-monitor_20260801_terminal/automation.toml`.
- The active automation path no longer exists. No Slurm job was inspected,
  submitted or cancelled by this fallback.

### `source:duca-tas-ms-tcn2-finegym-finediving-20260821`

- MS-TCN++ official repository: `https://github.com/sj-li/MS-TCN2.git`.
  The DUCA isolated clone is
  `E:/DeskTop/TAD/external/MS-TCN2_DUCA_20260821`; upstream master
  `f423a9e65f4ccb1cd7322eb9f94946a19e787993` contains a syntax error at
  `model.py:14`, while official historical revision
  `9d31fb3c23467b9ce3030d43b6d33a96869b6422` has a parseable MS-TCN++ path.
- FineGym official sources: `https://sdolivia.github.io/FineGym/`,
  `https://github.com/SDOlivia/FineGym`, CVPR 2020 paper
  `https://openaccess.thecvf.com/content_CVPR_2020/html/Shao_FineGym_A_Hierarchical_Video_Dataset_for_Fine-Grained_Action_Understanding_CVPR_2020_paper.html`.
  It provides hierarchical temporal annotations but not an official MS-TCN++ TAS protocol.
- FineDiving official sources: `https://github.com/xujinglin/FineDiving`, CVPR 2022 paper
  `https://openaccess.thecvf.com/content/CVPR2022/html/Xu_FineDiving_A_Fine-Grained_Dataset_for_Procedure-Aware_Action_Quality_Assessment_CVPR_2022_paper.html`.
  Its official task is procedure-aware action quality assessment; data access requires the official agreement.
- N16R4 read-only inventory on 2026-08-21 found no authoritative binding for FineGym,
  FineDiving, 50Salads, GTEA or Breakfast at common shared-root paths. Recursive search timed out,
  so the evidence class is `UNVERIFIED/MISSING_AT_COMMON_PATHS`, not authoritative absence.

### `source:duca-tas-east-50salads-rgb-20260822`

- EAST official repository: `https://github.com/tqosu/EAST`; paper:
  `https://openaccess.thecvf.com/content/ICCV2025W/SVU/papers/Wang_End-to-End_Action_Segmentation_Transformer_ICCVW_2025_paper.pdf`.
- EAST 50Salads preparation specifies `data/50salads/raw_data/video_fps2/`; the author-shared
  videos are RGB `160x160 @ 2 FPS`. The Dundee source describes the original acquisition as
  `640x480 @ 30 Hz` DivX AVI under CC BY-NC-SA 4.0.
- Dundee dataset DOI: `10.15132/10000120`; official landing page:
  `https://discovery.dundee.ac.uk/en/datasets/50-salads/`.
- Facebook Research AVT annotation archive:
  `https://dl.fbaipublicfiles.com/avt/datasets/50salads/annotations.zip`; remote durable copy:
  `/data/run01/sczc063/yuzibo/datasets/TAS/annotations/50Salads/avt_50salads_annotations.zip`.
- Acquisition evidence on 2026-08-22: the Oregon State Box share was verified as the EAST
  `video_fps2` release with 50 RGB MP4 files. Per-file transfer completed at
  `/data/run01/sczc063/yuzibo/datasets/TAS/east_50salads_160x160_2fps/data/50salads/raw_data/video_fps2`:
  50/50 sizes match the Box inventory, total size is `121,662,019` bytes, and every file passed
  ffprobe plus full ffmpeg decode. The earlier monolithic `video_fps2.zip.part` is incomplete/corrupt
  transport evidence and is not used by the dataset.
- The Box `annotations` archive is complete at the same root as `east_annotations_bundle.zip.part`
  (1,417,273 bytes); ZIP integrity passed and the archive contains the category map plus all five
  `50salads.fps2` split JSON and `.swp.json` pairs. The ordinary five split JSONs were schema-checked:
  each has 40 training and 10 validation videos, all 50 filenames resolve, and every video is held out
  exactly once across folds. Dundee 30 Hz AVI remains unavailable and is not substituted for the EAST
  2 FPS protocol. Evidence class: `RGB_AND_PROTOCOL_READY / FULL_DECODE_VALIDATED`.
- Official EAST code identity: `https://github.com/tqosu/EAST@a3233c2e6a6e3bbe36f9663e18180bdc5c126556`.
  The launch candidate `94b24753588ff60be986b35fefcca3f43d9c3fe6` adds complete epoch-boundary
  resume state and removes two undefined names from the official backbone export list; it does not alter
  the ViT-G model or protocol. The official VideoMAEv2-G K710 pretrain is bound at
  `/data/run01/sczc063/yuzibo/pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth`.
  The released EAST checkpoint link currently requires Oregon State login and remains unavailable.

### `source:duca-h65c-singleclock-pro-v002-20260822`

- Exact DUCA Project ID: `g-p-6a796fef9a00819194024cf1de3bd697`.
- Routing nonce: `DUCA-H65-TRUETIME-UVT-FOVEA-PRO-v002-20260822`.
- Oracle final: `C:/Users/skywalker/.codex/oracle/duca-h65-truetime-pro-20260822-v002/final.md`;
  SHA-256 `55954295c8224a476c1119bd6509b4c9d9b1b938bfc31a9d41dc8b26b60f3740`.
- Raw Oracle log: `C:/Users/skywalker/.fastctx/jobs/j-y27w4i/output.log`;
  SHA-256 `ec5d9e3e67ef3d20615750406213f26064d45caefa1d9739fbd6d8ddcc332964`.
- User attachment: `C:/Users/skywalker/.codex/attachments/4466c474-6623-4cdb-b89e-0fa9c8b0bc52/pasted-text.txt`;
  raw SHA-256 `899089c813ac995ab191ad5273271dfffe63434f5a365fdb4fbffa868918e46a`.
  Its normalized-LF SHA equals the Oracle final; the raw hash difference is only CRLF line endings.
- Model evidence: requested `gpt-5.5-pro`, resolved browser `Pro`, picker verified; one completed fresh turn.
- Coordinator verification and implementation outcome:
  `research-wiki/DUCA_H65C_SINGLECLOCK_PRO_VERIFICATION_AND_IMPLEMENTATION-2026-08-22.md`.
- Evidence boundary: external scientific/code review plus local static implementation audit; no Unit-1 PRE_RUN,
  model execution, GPU/Slurm training, mAP, cost, or paper claim.

### `source:duca-h65-60-lr-schedule-reassessment-v002-20260824`

- Exact DUCA Project ID: `g-p-6a796fef9a00819194024cf1de3bd697`.
- Routing nonce: `DUCA-H65-60-LR-SCHEDULE-REASSESS-v002-20260824`.
- Oracle session/conversation: `duca-h65-lr-reassessme-v2` /
  `6a8c0302-b5e4-83ea-b87e-3bcaa8130dde`.
- Prompt:
  `.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_H65_60_LR_SCHEDULE_REASSESSMENT-v002.md`;
  SHA-256 `fb4e80a00d9afa6b48d8cf0e6336294f0d4b9cc07c4b31ead8714e467490792d`.
- Visible final report:
  `.cvpr-pro-lab/pro-reviews/runs/duca-h65-60-lr-reassessment-v002/PRO_DUCA_H65_60_LR_SCHEDULE_REASSESSMENT-v002.md`;
  SHA-256 `ea3911d2d633a759b2d08aa0b20345f25edc4938287c0f535f7d9d87dd530b04`.
- Raw browser transcript:
  `.cvpr-pro-lab/pro-reviews/runs/duca-h65-60-lr-reassessment-v002/oracle-home/sessions/duca-h65-lr-reassessme-v2/artifacts/transcript.md`;
  SHA-256 `8aabef6e7b3e9bbc13786b2066a1f883208b710963eac780dafecebf652d3f39`.
- Model evidence: requested `gpt-5-pro`, resolved browser `Pro`, picker verified; one completed fresh exact-Project turn.
- Decision: `CONTINUE / HOLD_NEW_TUNING_UNTIL_TERMINAL`; only the already-running Jobs
  `1252979/1252980` may complete before terminal identity/result adjudication.
- Evidence boundary: external training-dynamics adjudication and frozen result thresholds; no new model,
  completed schedule result, mAP recovery, efficiency, or paper claim.

### `source:duca-h65-60-compression-diagnosis-v003-20260824`

- Exact DUCA Project ID: `g-p-6a796fef9a00819194024cf1de3bd697`.
- Routing nonce: `DUCA-H65-60-COMPRESSION-DIAGNOSIS-v003-20260824`.
- Oracle session/conversation: `duca-h65-compressio-v3` /
  `6a8c09ca-0844-83ea-9d6e-ad5fe5f73a50`.
- Prompt: `.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_H65_60_COMPRESSION_DIAGNOSIS-v003.md`;
  SHA-256 `1de5e5391c6f88cc56f39bceb0a846a3760dba5952e29deac1097603cea21d46`.
- Visible final report:
  `.cvpr-pro-lab/pro-reviews/runs/duca-h65-60-compression-diagnosis-v003/PRO_DUCA_H65_60_COMPRESSION_DIAGNOSIS-v003.md`;
  SHA-256 `a09866e0bc6ea8055c075f2c0dbf504c1f3632cce00624cadff668afae1dc1dc`.
- Raw browser transcript:
  `.cvpr-pro-lab/pro-reviews/runs/duca-h65-60-compression-diagnosis-v003/oracle-home/sessions/duca-h65-compressio-v3/artifacts/transcript.md`;
  SHA-256 `0bc17a0ca248f1ff0d9c629fd3f36980644cc9828f75612702894a8d4a9e1195`.
- Model evidence: requested `gpt-5-pro`, resolved browser `Pro`, picker verified at 5/5; one completed fresh exact-Project turn.
- Decision: `CONTINUE`; complete Jobs `1252979/1252980`, then apply the frozen recovery/gray/fail and terminal-slope branches. No third 60-epoch scheduler is authorized.
- Evidence boundary: external training-dynamics adjudication and a conditional one-extension decision tree; no new terminal A/B result, mAP recovery, implementation, efficiency, or paper claim.

### `source:duca-h65-60-compression-terminal-v001-20260824`

- Exact DUCA Project ID: `g-p-6a796fef9a00819194024cf1de3bd697`.
- Routing nonce: `DUCA-H65-60-COMPRESSION-TERMINAL-v001-20260824`.
- Oracle session/conversation: `duca-h65-terminal-v1` / `6a8c1c88-5680-83ea-8170-910401f870af`.
- Prompt: `.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_H65_60_COMPRESSION_TERMINAL_ADJUDICATION-v001.md`; SHA-256 `c1e410c10e42e784d157f91709c84aae45f0830e1c58c94e5ff3b5334e77e726`.
- Visible final report: `.cvpr-pro-lab/pro-reviews/runs/duca-h65-60-compression-terminal-v001/PRO_DUCA_H65_60_COMPRESSION_TERMINAL_ADJUDICATION-v001.md`; SHA-256 `a8ac84c1c6f1670b9c0509d715a693a69f7db8031306ee006cfd31392095db76`.
- Raw browser transcript: `.cvpr-pro-lab/pro-reviews/runs/duca-h65-60-compression-terminal-v001/oracle-home/sessions/duca-h65-terminal-v1/artifacts/transcript.md`; SHA-256 `8af5d8d52fd8f16f800295cae7b6c2b4f176086e0bcd69261f85d713d0b39cf8`.
- Model evidence: requested `gpt-5-pro`, resolved browser `Pro`, picker verified at 5/5; one completed fresh exact-Project turn.
- Decision: `STOP_60_EPOCH_COMPRESSION`; retain historical 30+60, perform only a read-only matched-update postmortem, and do not run a third scheduler or the conditional extension.
- Evidence boundary: single-seed terminal schedule attribution. It does not falsify H65 semantic indirect selection, prove theoretical impossibility of 60 epochs, establish multi-seed stability, or support an efficiency claim.

### `source:duca-all-matrix-same-budget-v001-20260825`

- Exact DUCA Project ID: `g-p-6a796fef9a00819194024cf1de3bd697`.
- Routing nonce: `DUCA-ALL-MATRIX-SAME-BUDGET-v001-20260824`.
- Oracle session/conversation: `duca-all-matrix-budget-v1` / `6a8c67bb-5ea4-83ea-a99d-590b3a0a744c`.
- Prompt: `.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_ALL_MATRIX_SAME_BUDGET_ADJUDICATION-v001.md`;
  SHA-256 `7de7f43fc67f16b26a00652189d6dde630d5cd6c5f6cc8626ce1080232a5b9c5`.
- Visible final report: `.cvpr-pro-lab/pro-reviews/runs/duca-all-matrix-same-budget-v001/PRO_DUCA_ALL_MATRIX_SAME_BUDGET_ADJUDICATION-v001.md`;
  SHA-256 `e936d2453516020508779ab947e070371e25c7ec22c3b2ad55cae0677df3f224`.
- Raw browser transcript: `.cvpr-pro-lab/pro-reviews/runs/duca-all-matrix-same-budget-v001/oracle-home/sessions/duca-all-matrix-budget-v1/artifacts/transcript.md`;
  SHA-256 `62458f883bdb58cab626383111b7cd3e1bfefb44597689ba1db60cd27fd3d12d`.
- Model evidence: requested `gpt-5-pro`, resolved browser `Pro`, picker verified; one completed fresh exact-Project turn.
- Decision: narrow `CONTINUE`; retain H65 30+60 terminal-EMA as the sole equal-budget base and perform only the existing-artifact SingleClock Gate-v2 admission closure.
- Evidence boundary: external all-matrix design/implementation/result adjudication. It authorizes no retraining, new inference, Query, Bridge, dynamic-K, efficiency claim, SingleClock PASS/KILL, or paper claim.

### `source:duca-pjst-pro-response-user-supplied-v001-20260825`

- User-supplied visible review title: `DUCA 稀疏 Token 物理时间表示终态裁决`.
- Self-reported exact Project ID: `g-p-6a796fef9a00819194024cf1de3bd697`.
- Self-reported nonce: `DUCA-H65-SPARSE-PHYSICAL-TOKEN-TRANSFER-v001-20260825`.
- Target DUCA revision: `b2ccfccab5b4912b59954afcc9b0364955327f7c`.
- Exact raw archive:
  `docs/methods/reviews/2026-08-25-b2ccfcca-duca-pjst-pro-response-user-supplied-raw.md`;
  SHA-256 `d1dce144eeff2b2bc474154df948b20b82536252df1f24cc40e4d84b62a02160`.
- Coordinator verification and absorption:
  `docs/methods/2026-08-25-b2ccfcca-duca-pjst-pro-review-absorption.md`.
- Proposed decision: `REVISE` to `DUCA_PHYSICAL_JACOBIAN_SUPPORT_TUBELET-v001`.
- Routing caveat: the formal local browser receipt for the related invocation was not accepted as an exact-DUCA
  completed turn; this source is therefore classified as a user-supplied independent review, not a repaired browser receipt.
- Evidence boundary: design proposal and code-fit analysis only; no PJST implementation, PRE_RUN, training, mAP,
  cost result, novelty closure, or execution authorization.

### `source:duca-marginal-cap-release-neighborhood-terminal-v001-20260831`

- Exact DUCA Project ID: `g-p-6a91061f789881918ccd8357ca3d6c92`.
- Routing nonce: `DUCA-MARGINAL-CAP-RELEASE-NEIGHBORHOOD-TERMINAL-ADJUDICATION-v001-20260831`.
- Oracle session/conversation: `duca-marginal-cap-neighborhood-terminal-v002` /
  `6a94bbaf-b2f8-83ea-81bc-5c0b6b23bdb5`.
- Latest public implementation:
  `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831@46812facc8773d9b4a9c21833cbe397c8aaa5a2d`.
- Link correction: the preserved transport manifest named the parent branch without `neighborhood`; the exact commit URL was
  correct, and the authoritative remote branch containing `46812fac...` is the branch recorded above.
- Prompt: `.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_MARGINAL_CAP_RELEASE_NEIGHBORHOOD_TERMINAL_ADJUDICATION-v001.md`;
  SHA-256 `9b82223a7b28ee67ae09dd67166a64dd01e1d0381c6569ed09cb4497a1386444`.
- Visible final report:
  `.cvpr-pro-lab/pro-reviews/runs/duca-marginal-cap-release-neighborhood-terminal-compact-v002/visible-report.md`;
  SHA-256 `9523b942ab88b755ad0b34a81b915b7be82dc19b56f0e270709d831bf4e3dffa`.
- Terminal experiment JSON:
  `.cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-neighborhood-46812fac-job1262121/oracle_cap_release_neighborhood_result.json`;
  independently verified SHA-256 `a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`.
- Model evidence: requested `gpt-5-pro`, resolved browser `Pro`, picker verified; one completed fresh exact-Project turn.
- Decision: `STOP`; close the existing additive Marginal-v1 and its capped→released joint-neighborhood repair. No further
  implementation, review, evaluation, bootstrap, utility-head training or official-test action is authorized.
- Evidence boundary: same-holdout metric-oracle mechanism diagnosis. It does not reject all dynamic computation, the H65
  priority sequence or the three budget values, and it provides no deployable policy, official validation/test result,
  uncertainty interval or end-to-end efficiency claim.

### `source:duca-project-level-after-marginal-stop-v001-20260831`

- Exact DUCA Project ID: `g-p-6a91061f789881918ccd8357ca3d6c92`.
- Conversation: `6a94c0ae-5388-83e9-afd3-6b8f1e596e1e`.
- Nonce: `DUCA-PROJECT-LEVEL-AFTER-MARGINAL-STOP-v001-20260831`.
- Latest public implementation:
  `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831@46812facc8773d9b4a9c21833cbe397c8aaa5a2d`.
- Prompt: `.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_PROJECT_LEVEL_AFTER_MARGINAL_STOP-v001.md`;
  SHA-256 `2d9a87a2c425d920ca1f9c8cb3dc4469ce7a4aa5f4a76b3bf2f3e4397a9527d1`.
- Visible report: `.cvpr-pro-lab/pro-reviews/runs/duca-project-level-after-marginal-stop-v001/visible-report.md`;
  SHA-256 `f2fcef731e3e6545ae06759bc6eae9f7f900f9807cf97c2583aa4c4d65f350ce`.
- Model evidence: requested `gpt-5-pro`, resolved browser `Pro`, picker verified; exact Project, nonce and latest GitHub
  repository/actual remote branch/commit/key-file URLs were bound.
- Decision: `PIVOT` to exactly one whole-video consistent-budget, cross-video donor-recipient oracle falsifier. If no legal
  candidate passes the simultaneous `+0.8/+1.0` gate at actual cost `<=47110`, the frozen consequence is project-level
  STOP within the current THUMOS14/H65/three-tier/resource boundary.
- Evidence boundary: development-set privileged oracle only. No deployable controller, official validation/test result,
  uncertainty interval, model forward, training or end-to-end cost claim is authorized by this decision.

### `source:duca-whole-video-consistent-budget-implementation-c27d77aa-20260831`

- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>.
- Actual remote branch:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>.
- Exact commit:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/c27d77aafd4aa514def033b03f2dfc2d6c24771e>.
- Runner:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/c27d77aafd4aa514def033b03f2dfc2d6c24771e/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>.
- Focused test:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/c27d77aafd4aa514def033b03f2dfc2d6c24771e/tests/test_duca_whole_video_consistent_budget_falsifier.py>.
- Independent Critic: PASS on the exact clean commit. Local focused result: 4 passed; legacy marginal test skipped on
  Windows by its existing module guard. No PRE_RUN or scientific metric result is attributed to this source.
- N16R4 snapshot: `/data/run01/sczc063/yuzibo/duca_whole_video_c27d77aa_20260831`; PRE_RUN Job `1262147`;
  launcher SHA-256 `ba0540563af8b4e876945befac109954791bbb8b51a73eb1c76646fd882f62e9`. Submission is infrastructure
  evidence only until the terminal receipt is read.

### `source:duca-whole-video-consistent-budget-implementation-33e4ed13-20260831`

- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>.
- Actual remote branch:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>.
- Exact commit:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>.
- Runner:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>.
- Focused test:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py>.
- Unchanged three-tier allocator:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>.
- PRE_RUN Job `1262147` exposed a deterministic proposal-row-order replay defect before candidate performance. Commit
  `33e4ed...` preserves the sealed producer order and adds its regression test; 28 focused tests and an independent Critic
  passed on the exact clean commit.
- Clean N16R4 snapshot: `/data/run01/sczc063/yuzibo/duca_whole_video_33e4ed13_20260831`. Corrected PRE_RUN Job
  `1262161` passed on 40 videos, 124 windows, fixed cost `47110`, 1560 ordered pairs, 704 legal candidates, 1330 actual
  interventions and `0.0 pp` fixed/capped/released anchor error. Receipt SHA-256:
  `734b178bfb7bdaa05879edfeb8e129263c9e2c4cf80867415eec6d41df3c12a3`; manifest SHA-256:
  `c4a02c47be1ab7e73dc81c18b32635d3347ece2f0d26b0d96de3ec4af053f69a`.
- Formal Evaluator Job `1262162` ended `NODE_FAIL` after `500/704` candidates because `g0022` went down. It produced no
  terminal result or runner failure receipt and therefore no performance verdict. Exact same-task recovery Job `1262190`
  reuses the same clean snapshot, script, manifest, sealed predictions, evaluator and output directory; no third job is
  authorized.

### `source:duca-whole-video-consistent-budget-terminal-result-20260831`

- Slurm Job: `1262190`, `COMPLETED 0:0`, start `2026-08-31T10:22:38+08:00`, end
  `2026-08-31T11:59:29+08:00`, node `g0024`.
- Public implementation:
  `feature/duca-whole-video-consistent-budget-falsifier-v1-20260831@33e4ed137c33eef07f0452b44506a6993bdf7535`.
- Terminal JSON:
  `/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_consistent_budget_result.json`;
  SHA-256 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`.
- Candidate manifest SHA-256: `c4a02c47be1ab7e73dc81c18b32635d3347ece2f0d26b0d96de3ec4af053f69a`.
  PRE_RUN receipt SHA-256: `734b178bfb7bdaa05879edfeb8e129263c9e2c4cf80867415eec6d41df3c12a3`.
- Evidence: 40 holdout videos, 124 windows, 1560 ordered pairs, 704 legal candidates, 705 evaluator calls, fixed actual
  observation cost `47110`, all fixed metric reproduction errors `0.0 pp`, and zero candidates passing the simultaneous
  `+0.8 pp` Avg-mAP / `+1.0 pp` mAP@0.7 gate.
- Best Avg-mAP delta: `+0.694215/-0.043632 pp` for Avg-mAP/mAP@0.7 at cost `46982`. Best mAP@0.7 delta:
  `-0.235922/+0.496998 pp` at cost `46854`. Best joint-gate delta: `+0.147383/+0.489786 pp` at cost `45830`.
- Evidence boundary: training-side development-holdout privileged oracle; no detector/Scout forward, training, gradient,
  bootstrap, official validation/test, uncertainty interval, paper-performance claim or deployable-policy claim.

### `source:duca-whole-video-terminal-pro-adjudication-v001-20260831`

- Exact DUCA Project ID: `g-p-6a91061f789881918ccd8357ca3d6c92`.
- Conversation: <https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a9501ec-3cc4-83ea-ba60-b8302e6e2632>.
- Nonce: `DUCA-WHOLE-VIDEO-TERMINAL-ADJUDICATION-v001-20260831`.
- Latest public repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>.
- Actual remote branch:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>.
- Exact commit:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>.
- Runner:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>.
- Focused test:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py>.
- Unchanged allocator:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>.
- Prompt: `.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_WHOLE_VIDEO_TERMINAL_ADJUDICATION-v001.md`;
  SHA-256 `128f49e6dd43e3835057e9d8cc2379cba92b319425aff0683d1069babbff2f46`.
- Visible report: `.cvpr-pro-lab/pro-reviews/runs/duca-whole-video-terminal-adjudication-v001/visible-report.md`;
  SHA-256 `4ed9e00834d9980bf44fc703d559de50abdd8f9b9e48d1764679f7c9e007359c`.
- Terminal experiment JSON:
  `/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_consistent_budget_result.json`;
  SHA-256 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`.
- Model evidence: requested `gpt-5-pro`, resolved browser `Pro`, model picker verified; exact Project, nonce, current
  conversation and all latest GitHub bindings agreed at terminal.
- Decision: `STOP` for the current THUMOS14 training-side holdout, frozen H65 detector/priority sequence,
  K256/K384/K512 sealed-prediction observation-transfer action space and resource boundary. No new Builder, Critic,
  Evaluator, Slurm, bootstrap, official-validation or official-test task remains.
- Evidence boundary: this closes the present action space, not all dynamic computation, Scouts, budget-conditioned training,
  token/layer conditional compute, budget spaces, detectors or datasets. The strongest cross-budget representation-mismatch
  explanation remains untested.

### `source:duca-gemini-dynamic-budget-review-user-paste-20260831`

- Source type: user-pasted external model analysis in the current Codex conversation.
- Model identity boundary: the text is available, but the exact Gemini model tier, browser session and cited literature were not
  independently verified by this record.
- Normalized audit:
  `research-wiki/sources/2026-08-31-duca-gemini-dynamic-budget-review-audit.md`.
- Bound code identity:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>.
- Disposition: retained as an external hypothesis and design-question source. It does not supersede the accepted Pro `STOP`,
  does not prove the cross-budget representation-mismatch mechanism, and creates no implementation or experiment task.

### `source:duca-multi-budget-detector-adaptation-revise-user-paste-20260831`

- Source type: user-provided scientific adjudication in the current Codex conversation.
- Source identity boundary: accepted under the user's instruction as the current scientific decision; the external generation
  session and model identity were not independently verified in this record.
- Normalized decision:
  `research-wiki/sources/2026-08-31-duca-multi-budget-detector-adaptation-revise.md`.
- Experiment design:
  `research-wiki/experiments/duca-multi-budget-detector-adaptation.md`.
- Model base: clean H65 commit
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>.
- Diagnostic functionality source only:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>.
- Decision: `REVISE`. The previous `STOP` remains for frozen-detector observation transfer; one distinct detector-adaptation
  hypothesis is designed. The current text supersedes the attached budget-native-selection proposal for the first test.
- Execution boundary: no implementation or experiment starts until Pro freezes a matched update schedule and exact independent
  train-side development video IDs.

### `constraint:duca-full-train-official-heldout-evaluation-20260831`

- Source type: explicit human experimental constraint in the current Codex conversation.
- Normalized record:
  `research-wiki/sources/2026-08-31-duca-full-train-official-test-human-constraint.md`.
- Requirement: after design freeze, both matched arms use the complete frozen THUMOS14 training split and the complete official
  held-out evaluation split. Development subsets and pilots are diagnostic-only and cannot replace formal evidence.
- Held-out boundary: evaluation only; no training, checkpoint/threshold/rule selection, route selection or iterative peeking.
- Unresolved identity conflict: OpenTAD/DUCA records use a 211-video `training/validation` convention, while ActionFormer official
  records use a 212-video `validation/test` convention. Exact subset names and complete video-ID sets require Pro adjudication.
- The in-flight Pro turn predates the constraint and remains untouched. No Builder, PRE_RUN or training is authorized until the
  terminal plan is checked and any conflict is resolved in a separate fresh Pro turn.

### `source:duca-multi-budget-pro-freeze-v001-20260831`

- Source type: verified Pro scientific adjudication in the exact DUCA Project.
- Nonce: `DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`.
- Conversation: <https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a9521de-d020-83e9-a0b9-19045c8d5390>.
- Full captured response:
  `.cvpr-pro-lab/pro-reviews/runs/duca-multi-budget-detector-adaptation-freeze-v001/visible-report.md`.
- Normalized source record:
  `research-wiki/sources/2026-08-31-duca-multi-budget-pro-freeze-v001.md`.
- Identity evidence: recovery manifest records exact Project, nonce and verified Pro model selection; the first control-plane
  attempt failed before submission and created no scientific turn.
- Decision: `CONTINUE` for one nested-K multi-budget training experiment with 6,000 successful updates per matched arm.
- Current disposition: its 160-train/40-development and no-official-test protocol conflicts with the later human full-data
  constraint. It is not executable until a new independent Pro turn freezes the complete formal data protocol.

### `source:duca-full-data-comparable-protocol-v001-20260831`

- Source type: verified Pro scientific and experimental-protocol adjudication in the exact DUCA Project.
- Nonce: `DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831`.
- Conversation: <https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a952a19-9294-83ea-b09f-5524e7825316>.
- Full captured response:
  `.cvpr-pro-lab/pro-reviews/runs/duca-full-data-comparable-protocol-v001/visible-report.md`.
- Terminal identity manifest:
  `.cvpr-pro-lab/pro-reviews/runs/duca-full-data-comparable-protocol-v001/manifest.json`.
- Normalized source record:
  `research-wiki/sources/2026-08-31-duca-full-data-comparable-protocol-v001.md`.
- Decision: `REVISE`; preserve nested-K detector adaptation but revoke 160/40, labeled training-side gates and the old oracle.
  Formal arms use all 200 `training` videos and a once-only complete `validation` held-out evaluation.
- Current authorization: only a read-only data-identity audit on base `04c35a3b...`; model implementation and compute remain
  forbidden until literal 211/212 evidence passes independent review and returns to Pro.
- Same-turn external summary boundary: a user-referenced `research_project_analysis.md` was not found in the repository. Its
  progressive-unfreezing, five-budget and ActivityNet suggestions remain unverified and do not alter the Pro-frozen task.

### `source:duca-irregular-temporal-sampling-external-proposal-20260831`

- Source type: user-supplied external research proposal in the current Codex conversation.
- Source identity boundary: the complete text is available, but its external model identity, literature claims and predicted
  gains were not independently verified.
- Complete raw archive:
  `docs/methods/reviews/2026-08-31-duca-irregular-temporal-sampling-external-proposal-raw.txt`.
- Structured absorption:
  `research-wiki/sources/2026-08-31-duca-irregular-temporal-sampling-external-proposal.md`.
- Retained hypothesis families: native consecutive tubelet acquisition, explicit physical-time encoding, sparse-to-dense
  reconstruction and end-to-end optimization stability.
- Proposed but unverified bundle: 144/48 dual-stream allocation, continuous-time rotary position encoding, adaptive Gaussian
  temporal splatting, Gumbel annealing, H65 feature distillation and a five-budget curve.
- Disposition: retained for future Pro-led mechanism design. It does not alter the current 211/212 data-identity audit, authorize
  model work, or supersede the frozen full-data nested-K detector-adaptation experiment.

### `source:agy-gemini-duca-comprehensive-wiki-code-review-v001-20260831`

- Source type: independently executed read-only external model review before the next Pro scientific adjudication.
- Transport identity: `agy` CLI, model `gemini-3.7-flash-high`, `effort=high`, plan/sandbox mode; the run completed with
  `GEMINI_DUCA_ADVISORY_READY` and did not edit files, submit compute, read held-out labels or operate a browser.
- Complete report:
  `research-wiki/sources/2026-08-31-agy-gemini-comprehensive-wiki-code-review-v001.md`.
- Review scope: complete current Wiki history plus the cited H65, TrueTime and whole-video diagnostic implementation snapshots.
- Disposition: evidence-index and independent advisory only. Its cross-budget mismatch explanation, old end-to-end failure
  attribution, non-contiguous tubelet claim, exact split choice, thresholds and milestones are recommendations or unisolated
  hypotheses unless separately supported by the Wiki and exact code. They do not authorize a route or an experiment.
- Required downstream use: the next Pro turn must read the full public Wiki tree and exact GitHub versions, reconcile Gemini's
  recommendations against original evidence, and independently choose the scientific route and unique next task.

### `source:pro-duca-github-wiki-comprehensive-review-v002-20260831`

- Source type: fresh exact-Project Pro scientific adjudication over the complete public Wiki and exact historical GitHub revisions.
- Identity: Project `g-p-6a91061f789881918ccd8357ca3d6c92`, conversation
  `6a954e5a-9c9c-83ea-95a8-2e1345c9178a`, nonce
  `DUCA-GITHUB-WIKI-COMPREHENSIVE-REVIEW-v002-20260831`, terminal marker
  `DUCA_GITHUB_WIKI_COMPREHENSIVE_REVIEW_READY`.
- Public evidence base: Wiki revision `8935e97219431b006fb04bbfc12c1005ebd81a05`, H65 scientific base
  `04c35a3b76897e6c1569eeede41ed3aecaf7f854`, whole-video diagnostic revision
  `33e4ed137c33eef07f0452b44506a6993bdf7535`, plus the Gemini full-history review.
- Complete report: `research-wiki/sources/2026-08-31-pro-github-wiki-comprehensive-review-v002.md`.
- Decision: `REVISE`. The current frozen-detector three-budget transfer route remains stopped. The only present task is a
  read-only full-data identity audit; model implementation is not authorized until its PASS/BLOCK return is independently reviewed
  and admitted by Pro.
- Conditional experiment: after data admission only, compare fixed K384 training with matched K256/K384/K512 training exposure on
  the complete 200-video training population, three seeds, 6,000 successful updates, sealed predictions, one complete held-out
  evaluation and paired whole-video uncertainty analysis.

### `source:pro-duca-comprehensive-route-integration-v001-20260831`

- Source type: newer exact-Project Pro comprehensive scientific, code and publication-route adjudication supplied by the user.
- Identity: Project `g-p-6a91061f789881918ccd8357ca3d6c92`, nonce
  `DUCA-COMPREHENSIVE-ROUTE-INTEGRATION-v001-20260831`.
- Complete report: `research-wiki/sources/2026-08-31-pro-duca-comprehensive-route-integration-v001.md`.
- Decision: `REVISE`. The only current task remains the complete train/held-out identity audit. Conditional on later data
  admission, test H65 system multi-budget exposure adaptation under unchanged nested K256/K384/K512 positions; do not call the
  estimand detector-only because the Stage-2 trainable system includes Scout/selector-related and feedback paths.
- Downstream conflict retained: this report uses seed 3407 as a mechanism gate and launches 3408/3409 only after it passes, whereas
  `source:pro-duca-github-wiki-comprehensive-review-v002-20260831` describes the formal comparison as three seeds. The conflict is
  irrelevant to the identity audit and must be resolved by Pro before model execution.

### `source:duca-full-data-identity-audit-fdd2bcdd-20260831`

- Source type: read-only full-data identity evidence produced by the authorized Builder, independent Critic and N16R4 CPU
  Evaluator chain.
- Code identity: branch `feature/duca-full-data-identity-audit-v1-20260831`, commit
  `fdd2bcdddf3f23f3546244adf90c4427ed022837`, parent
  `04c35a3b76897e6c1569eeede41ed3aecaf7f854`, clean tree; only audit tool and focused test differ.
- GitHub commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdd2bcdddf3f23f3546244adf90c4427ed022837>.
- Full report and literal manifests:
  `research-wiki/sources/2026-08-31-duca-full-data-identity-audit-fdd2bcdd/`.
- Effective report SHA-256: `d7251c11935644cf8661e6bfdcfb857e29d2357cb894b7de9d8b2bd7eaf6f1ab`.
- Result: 200/200/200 training identity; 211/211/211/211/211 OpenTAD held-out identity; 212 ActionFormer annotation IDs with
  sole additional ID `video_test_0000270`, explained by OpenTAD source line 11 as removed for wrong annotations. All 411 expected
  canonical videos decode; no intersection, missing media, duplicate, unassigned ID or unexplained difference.
- Isolation: no held-out label/segment decoding, checkpoint/model/GPU/mAP, prediction payload or per-video utility access.
- Verdict: `DATA_IDENTITY_PASS_211`. This verdict must return to Pro and does not independently authorize model work.

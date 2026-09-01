---
doc_id: MODEL_EXPERIMENT_HISTORY
version: v001
date: 2026-08-10
stage: DRAFT
status: active
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: none
---

# Model and Experiment History

## Research evolution

1. Early C3/PAction/GAS-VT/lattice work established attribution tools and exposed that apparent sparse gains are not meaningful without exact heavy-frame accounting, official detector/evaluator parity, and physical-time coordinate correctness.
2. Several direct DUCA selection variants failed to establish a fair advantage over uniform sampling. Local-cell selection, naive actionness top-k, selected-rank decoding, hard K384 query deletion/SparseHead, DCSR-G1, and ODF-CR-G2 were rejected or negative within their tested scopes.
3. The key semantic correction was to treat selected tokens as samples at physical times. Detector proposals must be mapped from selected coordinate q to physical coordinate t before unchanged official NMS/evaluation. Selected rank is not physical time.
4. The research question then moved from a fixed selector toward joint acquisition: a low-cost full-video scout, exact unique monotone positions, and eventually a per-video discrete budget under an average realized-cost constraint. AdaTok/AdapTok-style adaptive token ideas motivated competition checks, but do not by themselves resolve TAD boundary localization, physical acquisition, or full-stack cost.
5. Repeated recovery runs fixed engineering contracts including runtime data bindings, mask propagation, short-window effective-K semantics, dynamic temporal axes, and immutable receipts. These are infrastructure evidence only and do not prove the method.
6. A formal Stage-A design converged on four cells across registered seeds: dense T768, exact-uniform fixed K384, uniform mixed-K training with exact-uniform K384 evaluation, and learned fixed-K384 placement. Full official training/validation splits, terminal checkpoints, identical evaluator rules, and sealed multi-seed analysis are mandatory.

## Evidence corrections

- Intermediate, subset, train-domain, single-seed, failed-root, over-budget, synthetic, Oracle, proxy, or upstream metrics were explicitly demoted and cannot support paper claims.
- The historical AdaTAD/uniform 0.5 sampling value near mAP 65 is a baseline to reproduce under the locked protocol, not proof of current superiority.
- Engineering gates, unit tests, validator receipts, and successful scheduling are not model-quality evidence.
- Dynamic-budget, learned-position, and full-stack-efficiency claims remain unsupported until a complete official matrix is sealed and analyzed under preregistered rules.

## Current unresolved questions for Pro

The new Pro takeover must decide whether the pinned implementation faithfully tests the paper hypothesis; whether fixed-K learned placement is sufficiently novel and isolated; whether the downstream detector is genuinely unchanged; how to enforce batch-independent, training-calibrated per-video budgets; which minimal independent-versus-nested allocation comparison is decisive; and which literature, fairness, leakage, cost, and stop-rule gaps must be closed before any authorized run.

## Current terminal state

There is no paper-ready result and no authorized next experiment. The project is waiting for a fresh twelve-source Pro decision. All earlier coordinator route choices without a matching versioned Pro decision are advisory/quarantined and must not be executed.

---
doc_id: CLAIM_MAP
version: v001
date: 2026-08-10
stage: DRAFT
status: active
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: none
---

# Claim Map

| Candidate claim | Required evidence | Current state |
|---|---|---|
| C1: learned physical-time acquisition beats exact-uniform sampling at matched K | Full official split, multi-seed fixed-K contrast, identical detector/training exposure, terminal checkpoints | designed; not empirically supported |
| C2: per-video dynamic budget improves the accuracy-cost frontier | Dynamic DUCA versus identical K sequence with uniform positions, shuffled K histogram, fixed-K controls, realized cost | discussed/designed; not implemented as a paper-ready route |
| C3: DUCA protects high-IoU localization and short actions | mAP@0.6/0.7, short-action strata, boundary-error analysis | planned; no formal result |
| C4: savings are real end-to-end savings | Full-stack latency/FLOPs/memory/throughput including scout and transport | planned; no formal result |
| C5: allocation is deploy-valid and leakage-free | Input provenance audit; test-time payload denylist; video-disjoint calibration; immutable evaluator | partially implemented; not independently certified |
| C6: the method is novel and CVPR-relevant | Verified literature audit, precise differentiation from AdaTAD, AdapTok, token pruning, and adaptive computation | blocked pending literature verification |

No candidate claim is currently paper-ready. Infrastructure tests may advance implementation confidence but cannot advance these claims without formal model evidence.

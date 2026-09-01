---
doc_id: LITERATURE_AND_GAP
version: v001
date: 2026-08-10
stage: DRAFT
status: active
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: none
---

# Literature and Gap

## Relevant method families

- AdaTAD/OpenTAD supplies the downstream offline temporal detection setting and the uniform temporal-reduction reference used by this project.
- Adaptive token/frame selection methods motivate input-dependent allocation, but many optimize recognition, reconstruction, or latent-token objectives rather than high-IoU temporal localization.
- AdapTok is a key recent competitor and design reference for adaptive token allocation. Its allocation principles may transfer, but generative/reconstruction objectives, latent-token semantics, and any online assumptions must not be treated as directly equivalent to offline TAD.
- Dynamic inference and adaptive-computation literature motivates per-sample budgets, yet fairness requires comparing under the same realized full-stack cost rather than only nominal token counts.

## Research gap

The unresolved gap is not merely selecting salient frames. Offline TAD needs physical-time coverage, boundary-sensitive evidence, correct coordinate transport, and robust localization of short actions. Existing uniform downsampling does not allocate compute according to video difficulty; naive top-k policies can collapse coverage and destroy temporal geometry.

## Novelty question

The publishable novelty must come from a TAD-specific joint allocation formulation and evidence that it dominates strong cost-matched controls. A generic importance scorer plus top-k is insufficient.

## Required verification

Precise bibliographic metadata, contemporary competitors, and claims of novelty require a dedicated literature audit before paper writing. This v001 document deliberately contains no unverified publication numbers or priority claims.

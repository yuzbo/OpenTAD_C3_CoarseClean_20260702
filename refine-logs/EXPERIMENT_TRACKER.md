# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| ODFCR-M0 | M0 | freeze data and contracts | holdout-v2 + focused tests | validation 160/40 v2 | schema/class/leak checks | MUST | IMPLEMENTED_LOCAL | builder/consumer/evidence contracts and pure tests pass; real assets + Linux pending |
| ODFCR-G0 | M1 | exact official floor identity | official dense vs d3_off | validation probe | tensor/output exact | MUST | TODO | blocks training |
| ODFCR-S1-D1O | M2 | depth/residual factorial | d1_off, seed 2026073101 | holdout-v2 | Avg, 0.3--0.7 | MUST | TODO | paired |
| ODFCR-S1-D1A | M2 | depth/residual factorial | d1_all, seed 2026073101 | holdout-v2 | Avg, 0.3--0.7 | MUST | TODO | paired |
| ODFCR-S1-D3O | M2 | depth/residual factorial | d3_off, seed 2026073101 | holdout-v2 | Avg, 0.3--0.7 | MUST | TODO | official reference |
| ODFCR-S1-D3A | M2 | depth/residual factorial | d3_all, seed 2026073101 | holdout-v2 | Avg, 0.3--0.7 | MUST | TODO | utility candidate |
| ODFCR-S2 | M2 | repeat factorial | four arms, seed 2026073102 | holdout-v2 | paired metrics | MUST | TODO | paired |
| ODFCR-S3 | M2 | repeat factorial | four arms, seed 2026073103 | holdout-v2 | paired metrics | MUST | TODO | paired |
| ODFCR-K384 | M3 | isolate support penalty | frozen d1_all/d3_all K384 replay | holdout-v2 | paired all-vs-K384 | MUST | TODO | no training |
| ODFCR-AUDIT | M4 | final claim boundary | aggregate + diagnostics | holdout-v2 | audit verdict | MUST | TODO | paper row false |

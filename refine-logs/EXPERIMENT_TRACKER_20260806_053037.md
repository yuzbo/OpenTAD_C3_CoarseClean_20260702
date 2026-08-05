# 实验跟踪表

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0/M1/M2 | fresh matched control | `none_control` / seed3407 | Fit -> Gate | integrity, duplicate SHA, mAP@0.6/0.7, Avg-mAP | MUST | RUNNING | Job 1223819; P0 PASS; fresh epoch 0 started |
| R002 | M0/M1/M2 | test offset-only repair | `residual_window_center` / seed3407 | Fit -> Gate | integrity, role reachability, duplicate SHA, mAP@0.6/0.7, Avg-mAP | MUST | RUNNING | Job 1223820; P0 PASS; sole changed factor is calibration mode |
| R003 | M3 | fail-closed matched decision | after-any finalizer | Gate | centered-minus-none signs | MUST | RUNNING | Job 1223821 pending exact after-any predecessors; invalid pair => empty contrasts |
| R004 | M4 | full-stack Pareto confirmation | none/center ABBA+BAAB | Gate | p50/p95, memory, energy, attention pairs | CONDITIONAL | BLOCKED | Opens only if R003 PASS; requires new preregistration |
| R005 | M5 | disjoint-seed confirmation | none/center seeds3408/3409 | Fit -> Gate | paired accuracy and cost | CONDITIONAL | BLOCKED | Opens only after accuracy+cost Pareto pass |

状态词严格区分：`TODO`、`RUNNING`、`COMPLETE`、`FAILED`、`BLOCKED`；实验
wiki 另行使用 `designed`、`implemented`、`tested`、`experiment_running`、
`empirically_supported`、`paper_ready`。

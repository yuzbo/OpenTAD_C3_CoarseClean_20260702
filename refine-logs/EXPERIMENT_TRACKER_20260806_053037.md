# 实验跟踪表

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0/M1/M2 | fresh matched control | `none_control` / seed3407 | Fit -> Gate | integrity, duplicate SHA, mAP@0.6/0.7, Avg-mAP | MUST | COMPLETE | Job 1223819 completed 0:0; 60 epochs/9,600 updates; exact duplicate replay; Avg/mAP@.6/.7 = 10.52/8.90/6.98 |
| R002 | M0/M1/M2 | test offset-only repair | `residual_window_center` / seed3407 | Fit -> Gate | integrity, role reachability, duplicate SHA, mAP@0.6/0.7, Avg-mAP | MUST | COMPLETE | Job 1223820 completed 0:0; reachability PASS; Avg/mAP@.6/.7 = 12.57/11.04/8.14 |
| R003 | M3 | fail-closed matched decision | after-any finalizer | Gate | centered-minus-none signs | MUST | COMPLETE | Job 1223821 completed 0:0; finalization 2a9351a3; all three signs PASS; paired cost authorized only |
| R004 | M4 | full-stack cost non-inferiority | one-job none/center ABBA+BAAB | Gate | decode-to-NMS p50/p95, memory, energy, K_t/attention pairs | CONDITIONAL | RUNNING | Exact execution 2eca86cf remote 65/65 + precheck PASS; immutable deployment 3e12809c binds Job 1233097; currently PENDING (Priority) |
| R005 | M5 | disjoint-seed confirmation | none/center seeds3408/3409 | Fit -> Gate | paired accuracy and cost | CONDITIONAL | BLOCKED | Opens only after accuracy+cost Pareto pass |

状态词严格区分：`TODO`、`RUNNING`、`COMPLETE`、`FAILED`、`BLOCKED`；实验
wiki 另行使用 `designed`、`implemented`、`tested`、`experiment_running`、
`empirically_supported`、`paper_ready`。

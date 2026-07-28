# Research Log

## 2026-07-27 — DUCA-RIME four-stage implementation

- Recorded user approval for direct four-stage execution.
- Marked the earlier total-60 bounded-density plan as superseded by the
  dynamic-budget RIME adjudication.
- Implemented Phase 1 exact-K/geometry/cost evidence, Phase 2 `U-mixed-K` and
  causal gates, Phase 3 candidate/ablation matrix, and Phase 4 formal matrix.
- Corrected the mixed-K schedule to a per-video stateless 60-entry exposure
  with exact mean K=384.
- Corrected K192 to `fixed_floor_budget_position_only`; it cannot support a
  dynamic-budget claim.
- Corrected full-stack cost matching to exact `U-same-K` replay.
- Corrected the profiler to consume `effective_k`.
- Added an explicit Phase-4 authorization → Phase-2 receipt → budget protocol
  path/hash → checkpoint audit → terminal identity binding.
- Independent MAX audit found and corrected a deployment-blocking dense
  evidence SHA variable-name mismatch between the Phase-3 controller and
  Phase-3/4 submitters.
- Current state: `implemented/tested`; remote code gate and Slurm deployment
  remain pending. No empirical or paper-ready claim has been made.

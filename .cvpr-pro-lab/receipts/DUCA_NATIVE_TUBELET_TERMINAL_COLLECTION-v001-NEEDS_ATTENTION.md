# DUCA native-tubelet terminal collection receipt

- Recorded at: 2026-08-29T07:38:03+08:00
- Scientific experiment: fixed-budget native-tubelet uniform Job `1260184` and task-state coreset Job `1260185`
- Frozen implementation: `b33391126eac05e3353d322b973dda91741f0732`
- Attempted operational action: validate a minimal Slurm `afterany` terminal collector that would inspect the two existing jobs and their frozen terminal artifacts without polling
- Validation result: the collector script passed shell syntax validation, but Slurm test-only admission rejected the CPU-only request because this cluster partition requires a GPU allocation
- Submitted finalizer job: none
- Effect on the two training jobs: none; neither job was changed, restarted, cancelled, duplicated, or reconfigured
- Scientific evidence: none; this is an operational collection limitation, not a model or experiment result
- Disposition: do not allocate a GPU merely to wait and collect. Consume the existing jobs only after an external terminal event or a user-triggered read-only status check, then verify each job exit, `epoch_59.pth`, `state_dict_ema`, 6,000 successful updates, and `metrics_epoch59_ema.json`.
- Recovery: none while the formal pair is active

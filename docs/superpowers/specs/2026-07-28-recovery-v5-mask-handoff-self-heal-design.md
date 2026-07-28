# DUCA-RIME Recovery-v5 mask handoff and bounded self-healing design

## Decision

Recovery-v4 is engineering-failed-closed because the exact-uniform
ActionFormer path enabled `dynamic_temporal_bucket` but did not pass the
aligned `[B,K]` mask to `BackboneWrapper`. Recovery-v5 repairs the shared
detector-to-backbone contract for ActionFormer and TriDet. It does not change
the scientific protocol, model objective, data split, budget schedule,
checkpoint, seed, or paper evidence boundary.

## Code contract

`SingleStageDetector._forward_backbone_with_temporal_mask` is the single
handoff point:

- dynamic temporal backbones always receive the exact detector mask;
- ordinary backbones retain the legacy mask-free invocation;
- a `duca_rime_physical` selector with a non-dynamic backbone fails closed.

Both train and test paths in ActionFormer and TriDet use this helper. The
Slurm code gate runs a focused runtime test for both detector classes, and
the Phase-1 uniform precheck requires the frozen dynamic-bucket configuration.

## Deployment contract

The failed recovery-v4 root remains immutable. Recovery-v5 requires a new
clean Git commit, new commit-bound physical and salvage manifests, a new
submission manifest, and a fresh transaction root. The exact failed path
must pass the code gate before Phase 1 can start. Phase 4 remains disabled
and official-final remains sealed.

## Scheduled self-healing boundary

The monitor may automatically repair and redeploy at most once per unique
`commit + failure-signature` when all of the following are true:

1. the failure is deterministic and has an exact code/log cause;
2. the repair is a bounded engineering correction that does not change a
   scientific protocol, model design choice, hyperparameter, split,
   checkpoint, metric, or evidence claim;
3. focused tests, a Slurm code gate, clean-source identity, and fresh
   manifest/root checks all pass.

Ambiguous failures, repeated signatures, numerical/model-quality failures,
data integrity or leakage risks, protocol changes, and scientific-gate
failures must stop closed and notify the user. The monitor must never open
Phase 4, consume official-final, report intermediate metrics, reuse a failed
root, or retry indefinitely.

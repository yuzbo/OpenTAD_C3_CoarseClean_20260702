# DUCA Stage-A split historical negative-control adjudication absorption

## Source identity

- Source ID: `U-PRO-STAGEA-SPLIT-LEGACY-NC-1`
- Local source: `C:/Users/skywalker/.codex/attachments/504d0755-71a8-4640-9676-b2513b93923d/pasted-text.txt`
- SHA-256: `4f900ba49d0d7884a0a2beecbaf809542bb80e19bb42fee65c8bd003cfd2e082`
- Source length: 1,497 lines
- Reviewed repository head: `f9339422ef3a0c5d8f297f2ee0f11e90565b8cee`
- Runtime experiment parent: `4b766457b5abd8247f7e054d64bf6eb725183493`
- External verdict: `GO_SPLIT_HISTORICAL_NEGATIVE_CONTROL`

The source was read in full. It is retained as an external adjudication, while
all repository identities, implementation claims and executable details remain
subject to direct inspection and authoritative Linux/Slurm validation.

## Accepted scientific decision

The central verdict is accepted.

The retired raw FP32 alpha/beta row-mass statistic is representation- and
gauge-dependent. After the solver repair introduced score centering and carried
message scales, asking a repaired production trajectory to reproduce the old
raw-message failure is not a structural correctness predicate. Its absence in
job `1223308` is therefore not evidence that DUCA is numerically invalid.

The gate is split into two logically separate objects:

1. A deterministic historical code-regression negative control proves that the
   frozen old helper fails on a fixed tensor while the current normalized solver
   satisfies its structural identities on the same tensor.
2. The production numeric gate captures the first eligible `T=768, K=384`
   solver tensor from a real successful optimizer update and judges only the
   current structural oracles.

The legacy statistic may be recorded on the production capture only with
`role=diagnostic_only` and `admission_effect=none`. It cannot change capture,
owner selection, search horizon, seed, PASS/FAIL or release authorization.

## Rejected routes

- Keeping the double requirement “current oracles pass and the repaired real
  trajectory reproduces the retired guard” is rejected.
- Extending the 100-update search, changing the seed, lowering the legacy
  envelope or tuning a fixture after observing production behavior is rejected.
- Deleting all historical regression evidence is rejected; it would lose a
  useful solver-regression test.
- Treating any gate result as accuracy, convergence or paper-method evidence is
  rejected.

## Implementation contract

### Deterministic historical negative control

The frozen fixture is:

- `T=96`, `K=48`, `temperature=1`;
- physical seconds `0..95` in FP64;
- base scores `linspace(-3, 3, 96)` in FP32;
- historical stress shift `+2000`;
- current additive-shift oracle `+37`;
- the maximum physical gap is derived once by the repository's exact-uniform
  helper and then passed to both old and current computations.

It must establish all of the following:

- the retired raw-message guard triggers on the stressed fixed tensor;
- the current solver remains finite and satisfies slot-row, column-occupancy and
  ordered-slot-expectation identities on that tensor;
- `+37` preserves current soft slots, gradients and the hard path;
- FP64 log-partition displacement equals `K * shift / temperature` within the
  existing dual-logZ tolerance.

This exact fixture passed a focused Linux/PyTorch Slurm precheck in job
`1233456` with exit `0:0`. Jobs `1233451` and `1233452` failed before Python
because non-interactive Slurm did not provide `module/source`; they are launcher
failures and do not bear on the fixture. The successful precheck used the
repository's canonical interpreter directly.

### Production numeric gate

The first global successful update containing an eligible solver call is
selected. Every rank participates in the same collectives; the minimum rank
with a candidate owns the frozen tensor. The successful-update index is checked
for rank agreement. `NO_TARGET_CAPTURE` is a terminal failure only if no
eligible successful update appears in the unchanged 100-update bound.

The current solver admission retains the pre-registered thresholds and adds
explicit checks already implied by the solver contract:

- FP32/FP64 slots and gradients;
- slot-row mass and total occupancy;
- dual log-partition reconstruction;
- edge-flow conservation;
- finite gradients;
- exact FP32/FP64 hard path;
- maximum column occupancy;
- strictly ordered slot expectations;
- additive-shift invariance of slots, gradients, hard path and FP64 logZ.

No model, selector, loss, budget, threshold, seed, data split, checkpoint rule,
evaluator or Stage-A cell configuration is changed by this gate repair.

### Receipt and release contract

Expected terminal failure classes are:

- `NO_TARGET_CAPTURE`;
- `AMP_REPLAY_EXHAUSTED`;
- `CURRENT_STRUCTURAL_ORACLE_FAILED`;
- `DDP_STEP_OUTCOME_DIVERGENCE`;
- `WATCHDOG_OR_COLLECTIVE_FAILURE`;
- `INTERNAL_ERROR`.

Python writes best-effort per-rank and rank-zero numeric failure receipts. The
outer Slurm launcher consolidates timeout/rank-death failures and writes an
aggregate release failure receipt without opening metrics. Passed aggregate
release receipts are self-hashed and revalidated against all four child gates.

## Agreement boundaries

The scientific route is fully accepted. Individual code snippets from the
external source are not accepted verbatim. In particular:

- the fixed fixture had to be executed in the exact Linux/PyTorch environment
  before being frozen;
- legacy non-finite output must be representable without serializing JSON NaN or
  Infinity;
- synchronized capture ownership and successful-update identity must be explicit;
- timeout/rank-death evidence requires shell-level consolidation;
- receipt counter identities are validated only against the actual AMP replay
  semantics.

These are implementation corrections within the accepted route, not changes to
the scientific decision.

## Next-stage authorization

The route is sufficiently specified to execute without another Pro discussion.
The required order is:

1. create a clean code commit and push it;
2. create a new immutable remote checkout and transaction root;
3. pass the clean Linux/PyTorch code gate, including the fixed negative control;
4. pass the real natural-short heavy-backbone gate;
5. pass the two-rank production numeric gate;
6. pass the exact-211 physical identity gate and validated aggregate release;
7. only then create a new 12-cell Stage-A manifest and submit the three-seed
   full-200 training/exact-211 evaluation matrix;
8. read no partial metric; analyze only the sealed complete matrix.

Until the final seal exists, every output is `ENGINEERING_STATUS`. The gates do
not show that DUCA converges, learns useful positions, beats matched uniform
sampling, exceeds AdaTAD mAP 65, improves high-IoU localization or supports
Stage B.

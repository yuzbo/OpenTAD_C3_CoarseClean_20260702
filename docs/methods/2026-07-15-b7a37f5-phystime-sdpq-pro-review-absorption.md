# b7a37f5 PhysTime sparse-head Pro review absorption

## Scope

Review object: `b7a37f584ba7477159dd90ba08c14728c65fb19e`, message
`Add PhysTime rank-assignment diagnostic`.

Active task: sparse adaptation of the downstream TAD detection head. This is
not a DUCA selector task and must not be mixed with DUCA matched experiments.

## Absorbed verdict

I accept the main verdict:

- Kill the current observation-timestamp-coupled physical-anchor ActionFormer
  route as a method candidate.
- Kill `physical_time + rank_assignment` as a method candidate; keep it only as
  a negative geometry diagnostic.
- Do not kill physical time itself.
- The next candidate must decouple complete physical query anchors from sparse
  observation support.

## Why the current route fails

The current physical-metric head uses selected observation timestamps as point
centers and regresses nonnegative left/right distances. If a short GT action has
no selected observation-derived center inside it, the head cannot represent that
GT segment. Deleting the physical-inside guard is not valid because it creates
negative distance targets.

Rank assignment does not solve this. It intersects physical center constraints,
rank-center sampling, and rank range constraints, which worsens short-action
coverage in the diagnostic.

The current encoder/FPN is still rank-topological. Seconds are mainly injected
as head point geometry, so the system is not yet a physical-time-native
detector.

## Accepted next route

Candidate name: support-decoupled physical query sparse TAD head.

Core requirements:

- Use a complete uniform physical query grid for detection anchors.
- Treat sparse observations as support evidence, not as the query centers.
- Add atom-overlap pooling and physical-relative cross-attention over sparse
  support.
- Add a learned null-evidence token for unobserved or weakly observed query
  regions.
- Replace nonnegative left/right regression with signed center/width
  parameterization:
  `c_hat = c_q + qscale_q * delta_c`,
  `w_hat = qscale_q * exp(delta_log_w)`.
- Assign by duration level and normalized center/width cost, with short-action
  priority and at least one reserved match per GT when query capacity permits.

## Gates before training claims

- Runtime-vs-diagnostic parity for points, masks, assigned GT ids, and
  normalized targets.
- Rank-assignment config must be diagnostic-only and blocked from training.
- Report no-eligible and final no-assigned GT, including `<1s` cases.
- Report support observability and boundary observability.
- Compare matched controls: current uniform-rank seconds, current physical
  anchor, interpolation control, timestamp-only control, and the new
  support-decoupled candidate.
- Pass real THUMOS geometry, one-step gradient, micro-overfit, short pilot, and
  multi-seed gates before any paper-level claim.

## Caveats

The audit's direction is accepted, but its numeric thresholds are treated as
proposed gates rather than empirical facts. The rank-assignment diagnostic
proves the hybrid fix is bad; it does not alone prove the full causal
decomposition of G1a's mAP drop. A support-decoupled query head improves
representability, but still needs observability and cost evidence.

# Builder plan brief — P0 production interface freeze

Authority: accepted `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001`. Frozen revision:
`63a726a4aaf48ecbf6780bb196de43a890c6b4df`.

Write only `BUILDER_DUCA_P0_IDENTITY_PRODUCTION_PLAN-v001.md` in your allowed
worktree/return area. Map the already closed C-PROJ-001 production path to the
gate’s canonical input and output interface:

- exact production symbol and call boundary receiving canonical `(T,K,u,a,Q)`;
- exact position sequence, feasibility certificate, candidate-order and typed
  failure fields the gate must observe;
- the smallest interface exposure needed for one later bounded comparison;
- confirmation that no clipping, deduplication, fallback, second decoder,
  float tolerance or heuristic changes are proposed.

Include a minimal change plan only; do not edit production code, run a test,
execute Python, access data, invoke CPU/GPU/Slurm, use the browser, or start the
gate. State `NOT_EXECUTED` and identify any deterministic interface mismatch as
an IMPLEMENTATION_CORRECTION, not a scientific conclusion.

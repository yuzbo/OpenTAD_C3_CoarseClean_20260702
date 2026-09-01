# Evaluator authoring brief — P0 identity/optimality gate

Authority: accepted `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001`, section
“Required ultimate return”. Frozen revision:
`63a726a4aaf48ecbf6780bb196de43a890c6b4df`.

Create only the following evaluation-artifact package in your allowed evaluation
area, then return a durable authoring receipt:

1. `DUCA_P0_PROJECTOR_NORMATIVE_SPEC-v001.json`: canonical integer contract for
   `T,K,u,a,Q=2^20`, effective-K rule, endpoint rules, stride/disp constraints,
   exact lexicographic candidate key, typed failures and forbidden operations.
2. `DUCA_P0_IDENTITY_FIXTURE_MATRIX-v001.json`: the closed positive, negative
   and mutation fixture identifiers and definitions from the accepted decision;
   do not substitute or add fixtures.
3. `DUCA_P0_REFERENCE_PROJECTOR-v001.py`: an independent, non-importing
   reference implementation source. It must contain no import or invocation of
   production projector/selector/helper code.
4. `EVALUATOR_DUCA_P0_IDENTITY_AUTHORING-v001.md`: list paths, frozen revision,
   independence declaration and a clear `NOT_EXECUTED` receipt.

This is authoring/freeze only: do not run Python, pytest, production code,
fixtures, data access, CPU/GPU work, Slurm, browser or any experiment. Do not
change the mechanism, claim, split, metric, budget, detector, loss, NMS or the
P0 policy. Exact expected execution receipts are produced only after the
package has been independently frozen and the Coordinator authorizes the single
bounded gate execution.

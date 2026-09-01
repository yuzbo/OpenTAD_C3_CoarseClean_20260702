# Third focused Builder correction — DUCA H65 SingleClock physical range

- **Parent commit:** `f3e89b059c038837c7a2b1fbe10b22fe4cb77e7f`
- **Clean correction commit:** `b2ccfccab5b4912b59954afcc9b0364955327f7c`
- **Evidence class:** deterministic input/terminal-contract validation; no efficacy evidence.

The SingleClock metadata entry point and terminal identity finalizer now require
physical positions to be strictly increasing and inside
`[0, dense_valid_len)`.  Tests distinguish negative, repeated, descending and
upper-bound-invalid inputs from the valid final physical index.

Changes are limited to the identity entry point, terminal finalizer and their two
focused test files.  `py_compile` passed.  The local combined PyTorch test could
not collect because Windows failed to initialize `c10.dll` (`WinError 1114`), so
the exact focused suite remains mandatory in the N16R4 environment.  No valid
input computation, selection, checkpoint, dataset, evaluator, configuration,
launcher, threshold, training, inference, metric or claim changed.

- **next_owner:** same independent Critic
- **next_action:** terminal focused recheck at `b2ccfcca...`
- **dependency:** clean detached Critic binding
- **expected_return:** PASS to Evaluator or terminal cycle blocker
- **single_recovery:** all three focused correction allowances consumed

# Second focused Builder correction — DUCA H65 SingleClock identity records

- **Parent commit:** `27e593a1368ca273847a942a9af31297e275df51`
- **Clean correction commit:** `f3e89b059c038837c7a2b1fbe10b22fe4cb77e7f`
- **Evidence class:** deterministic terminal-evidence validation; no efficacy evidence.

The terminal finalizer now requires a nonempty deterministic record list whose
length equals `sample_count`; complete typed physical identity fields; unique,
sorted `(video,start)` IDs; valid selected-position lengths/order; and duplicate
accounting that refers only to recorded physical windows.  Malformed paired
payloads fail closed.

Only the terminal finalizer and its focused test file changed.  `py_compile`,
`git diff --check`, and `13` focused tests passed.  Model, recorder, dataset,
evaluator, configuration, launcher, scientific thresholds and checkpoint are
unchanged.

- **next_owner:** same independent Critic
- **next_action:** final focused recheck of `f3e89b05...`
- **dependency:** clean detached Critic binding
- **expected_return:** PASS to Evaluator or terminal implementation blocker
- **single_recovery:** second focused correction consumed

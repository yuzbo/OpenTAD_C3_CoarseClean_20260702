# H-RIME v1 Implementation Plan

**Design source**:
`docs/superpowers/specs/2026-07-28-hrime-v1-budget-conserving-design.md`

**Approved scope**: implement the corrected Stage-0 foundation and H-RIME
software/oracle surfaces. Large learned-H-RIME training remains gated by the
same-total-cost oracle; official-final evaluation remains sealed.

## Task 1 — Freeze research state

Files:

- `research-wiki/query_pack.md`
- `research-wiki/anti_repetition.md`
- `research-wiki/source_registry.md`
- `research-wiki/ideas/duca-rime-graph.md`
- `research-wiki/experiments/duca-dynamic-k-rime-oracle.md`
- `research-wiki/graph/edges.jsonl`
- `research-wiki/log.md`

Actions:

1. Register the full Pro report as an external review.
2. Record conditional acceptance and every correction in the design.
3. Promote H-RIME from `awaiting_user_approval` to
   `user_approved/designed/implementation_started`.
4. Record verified raw checkpoint paths/hashes and unavailable energy evidence.
5. Commit design/wiki state before model code.

Verification:

```bash
git diff --check
git status --short
```

## Task 2 — Repair true short-window execution

Files:

- `opentad/models/detectors/duca_protected_e2e_frame_selector.py`
- exact-K/ledger focused tests under `tests/`

Actions:

1. Add one canonical quantum-aligned effective-K mapping.
2. Deduplicate nominal budgets that map to the same effective K.
3. Make the heavy backbone input width equal to effective K.
4. Remove duplicated inactive tail execution.
5. Bind ledger equality:
   `backbone_input_k == padded_k == unique_k == effective_k`.
6. Add explicit `valid_len=231` coverage:
   `(192,256,384,512) -> (192,224,224,224)`.
7. Fail closed when no positive quantum-aligned option exists.

Verification:

```bash
python -m pytest tests -q -k "duca and (short_window or exact_k or ledger)"
```

## Task 3 — Repair compaction and create immutable salvage

Files:

- `tools/bata/compact_duca_rime_checkpoint.py`
- dense ActionFormer/TriDet launchers under `scripts/`
- new `tools/bata/salvage_duca_rime_dense_checkpoint.py`
- new salvage launcher under `scripts/`
- launcher/receipt focused tests

Actions:

1. Invoke the compactor as
   `python -m tools.bata.compact_duca_rime_checkpoint`.
2. Add a clean-cwd smoke test using the exact launcher command.
3. Add a manifest-driven salvage tool that verifies source path, size, SHA-256,
   source job ID, raw checkpoint schema and external provenance.
4. Write outputs only into a new immutable transaction root.
5. Produce terminal EMA, compaction audit, checkpoint binding and salvage
   receipt without altering the failed source root.
6. Mark missing embedded commit/variant/seed metadata explicitly instead of
   inferring it.

Verification:

```bash
python -m py_compile tools/bata/compact_duca_rime_checkpoint.py
python -m py_compile tools/bata/salvage_duca_rime_dense_checkpoint.py
python -m pytest tests/test_duca_rime_launchers.py -q
```

### Task 3.1 — Close recovery-v3 evaluator contract gaps

Design source:
`docs/superpowers/specs/2026-07-28-stage0-recovery-v3-contract-repair-design.md`

Files:

- `scripts/run_duca_rime_phase1_uniform_eval.sh`
- `tools/bata/duca_rime_training.py`
- `tools/test.py`
- focused launcher/training-contract tests

Actions:

1. Bind the exact absolute VideoMAE path/SHA in uniform precheck, actual
   inference, and its receipt.
2. Add a dedicated dense-reference protocol predicate and contract validator;
   do not add dense protocols to ordinary trainable-RIME formal routing.
3. Route dense development evaluation through its registered `training` subset
   and a dedicated engineering-only terminal schema.
4. Require a new clean commit, remote Torch checks, affected launcher prechecks,
   independent deployment audit, new manifests, and a fresh transaction root.
5. Keep Phase 4 disabled and official-final sealed.

Verification:

```bash
python -m py_compile tools/test.py tools/bata/duca_rime_training.py
python -m pytest tests/test_duca_rime_launchers.py \
  tests/test_duca_rime_training_contract.py -q
bash -n scripts/run_duca_rime_phase1_uniform_eval.sh
```

## Task 4 — Seal Phase-4 release

Files:

- `scripts/run_duca_rime_phase3_submit_controller.sh`
- controller/launcher focused tests

Actions:

1. Default Phase-4 submission to disabled.
2. Require explicit opt-in plus a verified development authorization receipt.
3. Bind receipt path/hash, candidate checkpoint hash, protocol hash and code
   commit before submission.
4. Keep the official-final data path unopened in precheck and failed states.

Verification:

```bash
python -m pytest tests/test_duca_rime_launchers.py -q
```

## Task 5 — Implement deterministic H-RIME core

Files:

- new `opentad/models/duca/__init__.py`
- new `opentad/models/duca/hrime.py`
- new `tests/test_hrime_budget_allocator.py`

Actions:

1. Implement canonical feasible effective-K mapping.
2. Implement reachable-total enumeration/projection.
3. Implement exact-equality MCKP dynamic programming.
4. Freeze integer score quantization and deterministic tie-break.
5. Emit solver input and assignment hashes.
6. Implement planner/window-option typed contracts without GT-dependent input.
7. Test infeasible raw caps, short windows, duplicate options, exact totals,
   deterministic ties and brute-force agreement.

Verification:

```bash
python -m pytest tests/test_hrime_budget_allocator.py -q
```

## Task 6 — Implement video grouping and two-pass dispatch contracts

Files:

- focused dataset/collate/sampler adapters under `opentad/datasets/`
- two-pass adapter under `opentad/models/duca/`
- integration tests

Actions:

1. Group flat windows by video without changing physical window metadata.
2. Guarantee stable `(video_id, window_start_frame)` ordering and group hash.
3. Add one shared cheap-scan result protocol for video/window summaries.
4. Convert solver assignments into the existing per-window replay schema.
5. Stable-bucket windows by effective K and restore original order.
6. Reuse `DucaRimeFrameSelector`; do not create another physical selector.
7. Ledger actual scan count, heavy count, overlap duplication and fallback.

Verification:

```bash
python -m pytest tests -q -k "hrime or video_group or homogeneous_k"
```

## Task 7 — Implement held-out same-total-cost oracle

Files:

- new oracle tool under `tools/bata/`
- new validator/launcher under `scripts/`
- oracle tests

Actions:

1. Require video-disjoint train/calibration/development manifests.
2. Compare uniform, independent-window and joint-video allocation at identical
   per-video realized effective-K totals.
3. Generate hash-bound replay maps.
4. Re-run full official prediction merge/NMS/evaluator for selected allocations.
5. Audit additive-surrogate rank correlation, sign agreement and worst error.
6. Require a pre-committed primary endpoint/multiplicity/guardrail manifest.
7. Emit a fail-closed oracle receipt that alone can authorize Task 8 training.

Verification:

```bash
python -m pytest tests -q -k "hrime and oracle"
```

## Task 8 — Remote precheck and gated experiment deployment

Actions:

1. Run all registered local focused checks.
2. Push the exact clean implementation commit.
3. Run corresponding `PRECHECK_ONLY=1` launchers through Slurm.
4. Submit a new immutable Stage-0 salvage/Phase-1 closure DAG.
5. Submit the Stage-1 oracle only after Stage-0 receipts pass.
6. Do not submit learned H-RIME training unless the oracle receipt authorizes it.
7. Do not submit official-final Phase 4 from this plan.

Required handoff:

- exact commit and protocol hashes;
- Slurm job IDs and immutable transaction roots;
- per-stage receipt status;
- explicit evidence class (`ENGINEERING_STATUS` or
  `COMPLETE_DEVELOPMENT_EXPERIMENT`);
- no paper performance claim from incomplete jobs.

## Task 9 — Final verification and memory update

Verification:

```bash
python -m py_compile tools/train.py tools/bata/train_lowres_action_probe.py
python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q
git diff --check
git status --short
```

Update the research wiki in the same turn with implementation state, test
results, transaction identities, negative findings and next gate. State
transitions must remain literal; do not jump from `tested` to
`empirically_supported`.

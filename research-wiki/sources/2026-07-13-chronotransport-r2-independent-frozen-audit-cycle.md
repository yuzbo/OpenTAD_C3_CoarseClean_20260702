---
type: source_record
source: independent_agents
date: 2026-07-13
status: absorbed_under_repair
---

# ChronoTransport r2 frozen-slice audit cycle

This record preserves the independent source-level review conclusions for the current uncommitted
implementation based on committed base `797a2df`. These are implementation-integrity findings, not
experiment results. No formal profile, training seed, Gate result, registration `I/R`, or GPU job was
created.

## Gate 1 verdict: `REVISE_GATE1_FROZEN`

The reviewer matched all 11 supplied frozen file SHA-256 values, then found that typed wrappers did not
close every acceptance path:

1. test-only raw timing rows could still be rebuilt into the formal profile schema and accepted;
2. paired replay still accepted a caller detector/batches/motion source, while raw rows could create a
   formally accepted regret artifact;
3. direct adjudication/unlock APIs and the CLI could bypass clean detached `HEAD=R`, deep source
   validation, and the unresolved random-control lock;
4. `PRECHECK_ONLY` exited before a protected Slurm/physical-GPU check and
   `CUDA_VISIBLE_DEVICES=1` did not prove physical GPU1;
5. source hashes did not verify every required regular Git blob/mode against `I:path`;
6. result and terminal-marker paths were contained but not fixed/distinct, so `SUCCESS` could overwrite
   the result artifact.

Correct but insufficient subcontracts were the 23-arm order, same-RNG reverse-order probe, media
pre-verification, canonical output base, atomic four-state marker, core registration context, and random
fail-closed behavior on the canonical launcher path.

## Stage B repair and verification state

The next repair bound the baseline to the exact 140 batch/exposure window order and the registered action
hash for every 140x16 row; required the independent EMA map to equal every `state_dict_ema` alias
bitwise; hashed the logical predictor once; and confined four writable artifacts to pairwise-distinct
direct children of `FORMAL_OUTPUT_BASE/<R>/<seed>/`, outside the repository and distinct from inputs.

Remote isolated verification at
`/data/run01/sczc063/yuzibo/ct_stageb_p1_20260713_a61f09` passed five new adversarial tests, then the
complete Stage-B file passed `46 passed in 130.27s`. One intermediate failure was an obsolete duplicate-
alias hash in the test fixture; after using the canonical helper, the isolated failure and complete file
rerun passed. Independent exact-SHA source review returned `APPROVE_STAGEB_FROZEN`; the slice is
approved as code, not as a trained seed or science result.

## Stage C verdict: `REVISE_STAGEC_FROZEN`

All five supplied frozen hashes matched. Batch-2 exposure, success-state coherence,
`use_reentrant=False`, and the broad rollback chain were confirmed, but two P1 paths remained:

1. a forward using dense/wrong actions could return the expected caller `action_payload`, so losses and
   gradients were not bound to runtime-executed actions and repair/fallback state;
2. an unordered Parameter identity set could miss swapping two isomorphic depth-12 ViT blocks, changing
   `path -> object`, module order, and retry topology.

Generic shared mutable-container aliases could also split during Python-state restore. Required repairs
are runtime-owned action installation/readback, exact ordered topology/alias graphs, and memoized object-
graph rollback.

The follow-up repair reproduced five targeted failures, then passed 5/5 targeted, 46/46 focused with
one protected-CUDA skip, and 88/88 compatibility checks with the same skip. It is frozen at
`/data/run01/sczc063/yuzibo/ct_stagec_final2_20260713_6c117f`. A new agent that did not implement the
repair is performing the exact-SHA/line-by-line audit; until its verdict, status is
`tested_under_review`, not approved.

The independent audit has already reproduced two additional P1s on those exact bytes: loss tensors can
be disconnected from the single audited runtime forward, and `.data` mutation can change a frozen heavy
parameter without changing its Tensor version. Both probes returned `SUCCESS`; therefore the frozen
slice cannot be approved even before the reviewer finishes checking for further blockers.

Final verdict: `REVISE_STAGEC_FROZEN`. A third P1 accepted a runtime summary that omitted the formal
forced-dense/fallback/evidence-valid fields because missing `dict.get` values were treated as safe; the
runtime was also recognized by duck typing rather than production class/source identity. The real-CUDA
GradScaler test remains skipped without protected physical GPU1. P2 findings additionally require
success-path Python-state constraints, exact scaler/autocast evidence, complete direct-dependency source
hashes, and formal-runner binding of materialized batches to canonical fit windows/order.

## Still-unimplemented plan surfaces

The planned formal Gate-2/3 runner/tests, Stage-C and matched-dense runners/tests, Gate-4 runner/tests,
the complete precheck/Stage-B/Gates-23/Stage-C/Gate-4 launchers, and launcher tests are absent. Existing
pure primitives do not satisfy those tasks; the repository cannot yet be labelled `implemented`.

Subsequent progress: the Gate-4 pure adjudicator and synthetic tests now exist and passed 12/12 remotely,
including report recomputation from raw evidence and tamper rejection.
This narrows but does not close the recorded gap: the formal Gate-4 CLI/launcher, registered source chain,
Stage-C/matched-dense completion inputs, matched profiling producer and independent audit remain missing.

## External specification blockers

- Unsuffixed `random_p{2,4,8}` names conflict with a seed-dependent digest. The recommended minimal
  amendment is fixed seed `3407`; code remains fail closed pending explicit approval and review.
- Approved `CUDA_VISIBLE_DEVICES=1` conflicts with observed Slurm cgroup remapping to local ordinal 0.
  The recommended amendment verifies physical GPU1 from Slurm IDs and one local ordinal `0`. No eligible
  allocation currently exists.

## Gates 2/3 verdict: `REVISE_GATES23_FROZEN`

All three frozen SHA values matched. Independent remote audit found three P0s: caller-written 180-row
replay/no-leak booleans could become formal replay evidence; phase validation accepted a text checkpoint,
self-reported predictor identity, and arbitrary 140-row ledger; and public adjudication/report/unlock APIs
did not independently validate clean detached `R`, the random lock, or the full Gate-1 unlock. P1s were
the globally clustered seed bootstrap being incorrectly redrawn per window (with a reproduced PASS/FAIL
CI reversal), an unbound `registration_commit`, and a CLI call incompatible with the hardened Gate-1
unlock builder. Symlink leaves and missing adversarial tests were P2. Audit scratch:
`/data/run01/sczc063/yuzibo/tmp/audits/r2_gates23_independent_audit_f7f8340_20260713a`.

## Gate 1 hardened verdict: `REVISE_GATE1_HARDENED`

All 12 hardened SHA values matched. No P0 was assigned, but four P1 classes remained: formal public
functions permitted a contextless repository branch; importable internal issuers/raw-materialized runners
could still construct formal objects from caller-provided execution inputs; the in-memory registration was
not compared with exact canonical bytes and regular-blob mode at `R:path`; and result/terminal creation
used overwrite-capable `os.replace`/`mv` without an exclusive run lock or atomic no-clobber. Existing
random/GPU fail-closed behavior and required-source blob checks were confirmed correct but insufficient.

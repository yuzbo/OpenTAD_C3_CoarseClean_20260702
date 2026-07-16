---
type: source
id: source:chronotransport-r2-predeployment-integrity-audit-20260715
title: "ChronoTransport r2 pre-deployment integrity audit"
date: 2026-07-15
status: revise_before_registration
---

# ChronoTransport r2 pre-deployment integrity audit

This is a read-only independent audit of the current dirty implementation bytes on base
`797a2df8d00560c8f7a7f66c13e95bb5b0d836ee`. It did not run an experiment, mint a Gate artifact,
approve registration, or independently rerun the historical remote test counts.

Verdict: `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`.

## Registration and reachability blockers

- `REQUIRED_REGISTRATION_SOURCE_PATHS` was exact-set enforced but omitted the Gate-1 hardening test,
  the Gate-4 adjudicator/test, and every still-missing Stage-C/matched-dense/Gate-4 runner/launcher.
- the formal profile contains the three required random controls, while
  `validate_formal_random_control_lock` intentionally rejects every random plan because the approved
  spec never fixed the unsuffixed control seed. Formal Gate 1 is therefore correctly locked, not a
  scientific FAIL.
- `chronotransport_r2_stage_b_factory.py` read obsolete flat registration keys
  `window_manifest.manifest_sha256` and `exposures.stage_b_sha256`; the validated schema embeds these
  as `window_manifest.artifact.manifest_sha256` and
  `exposures.stage_b.artifact_sha256`. The Stage-B CLI also read outer
  `window_manifest.split_hashes` instead of `window_manifest.artifact.split_hashes`. The real Stage-B
  chain was statically unreachable.

## Stage C and Gate 4 blockers

- the Stage-C success path requires every registered buffer to remain bitwise unchanged. A real
  train-mode ActionFormer forward updates registered `rpn_head.loss_normalizer`, so the current
  primitive rejects a normal successful update as `INVALID_IMPLEMENTATION`.
- the audited Stage-C hook requires the top-level model forward to return one differentiable Tensor,
  while the real ActionFormer training forward returns a loss dictionary. The current loss container
  and toy tests also do not prove two per-window regret targets from that same batch-two forward.
- there is no formal A-only matched-dense primitive or 4,200-success runner. The present Stage-C
  primitive owns non-empty A/T/R groups and the repository only tests toy per-update behavior.
- no formal Gate-4 CLI/runner/launcher binds official invocation order, Stage-C checkpoints,
  calibration-frozen static policy, live full-stack timing, predictions/GT, regret and clean detached
  registration R. The existing `gate4.py` is a pure adjudicator only.

Registration and Stage-B publication still use replace-style overwrites without one exclusive formal
writer/no-clobber transaction. Gates23 protects individual writes but can leave an unrecoverable
replay/report half-state if the process stops before terminal publication.

## Leakage review

The reviewed Gate-1 replay obtains GT detector loss only in evaluation-only adjudication; motion
controls are derived from deploy-visible runtime signals. Stage-B GT regret is training supervision.
No reviewed path showed GT, teacher, raw prediction cache or counterfactual ledger entering the
runtime scheduling decision. This is a bounded code finding, not proof for the still-missing formal
Stage-C/Gate-4 chain.

## Governance conflict

The approved r2 text and existing launch surfaces require physical GPU1 and exact
`CUDA_VISIBLE_DEVICES=1`. The current repository instruction forbids fixing a physical GPU index or
overwriting Slurm's visibility and requires the process to address its assigned single GPU as
`cuda:0`. A reviewed protocol amendment is required before any formal Slurm launch; code must not
silently choose one contract.

## Repair initiated in the same round

- added a regression and nested-schema validator for the Stage-B factory;
- added the already-existing Gate-1 hardening and Gate-4 adjudicator/test paths to the current
  registration source vector.

These repairs have only passed local `py_compile`/static checks so far. The registration remains
`NOT_READY`; Stage C, matched dense, Gate 4, I/R and all formal jobs remain locked.

After the audit, the seven new targeted regression cases were copied to the isolated remote audit
worktree
`/data/run01/sczc063/yuzibo/tmp/audits/ct_r2_integrity_fix_20260715_a` and run in the fixed OpenTAD
environment. Result: `7 passed in 57.28s`. The cases cover nested Stage-B manifest/exposure/split
identity, registration and Stage-B completion no-clobber, checkpoint/ledger no-clobber, and the
exclusive Stage-B writer lock. This is focused implementation evidence only, not full Stage-B
approval or a Gate result.

The complete affected Stage-B and registration suites were then run in the same isolated worktree.
Result: `89 passed, 1 xfailed in 310.86s`. The expected xfail remained protected; there were no test
failures. This verifies the current repaired implementation surfaces only. It does not approve the
frozen protocol, create implementation commit I or registration R, unlock any formal job, or resolve
the Stage-C/Gate-4 and governance blockers above.

A compatibility matrix covering Gate-1 hardening, Gates 2/3 and the Gate-4 pure adjudicator also
passed in that worktree: `43 passed in 295.25s`. This confirms that the bounded Stage-B/registration
repairs did not break those currently implemented test surfaces. It did not invoke a formal Gate or
exercise the still-missing Gate-4 evidence producer.

The audit's unrecoverable Gates-2/3 half-publication finding was reproduced with a RED regression.
The runner now permits an interrupted immutable R to continue from an existing replay or report only
after exact canonical-byte recomputation; a mismatched/non-regular artifact or any existing terminal
still fails closed. The targeted GREEN passed `1/1` in 37.95 seconds and the Gates-2/3 plus
registration matrix passed `59 passed, 1 xfailed in 206.70s`. These exact bytes are
independently approved as `APPROVE_GATES23_RECOVERY`. The reviewer matched runner SHA-256
`4CED5459B1785855F46FE0A22748229D77D885245D44B8A84160C3B814616885` and test SHA-256
`10D134573FEB8029DCE02F4A01E0CE2D40006DAD199FA0E7AAA32129BF65AFB9`, and confirmed `_run_locked`
integration, exact-byte reuse, terminal refusal, regular-file/symlink checks, hard-link no-clobber and
exclusive locking. This is bounded code approval, not formal Gate evidence or registration approval.

A follow-on Stage-B path audit reproduced two additional formal-integrity failures: a symlink parent
was accepted for the exclusive run lock, and resolving output arguments before inspection allowed a
symlink alias to the canonical R/seed root to be laundered into an accepted lexical path. RED was
`2 failed`. The bounded repair now checks every existing component with `lstat` before following any
path, uses `O_EXCL` plus `O_NOFOLLOW` for the lock, and removes the lock only if its device/inode still
matches the opened descriptor. Targeted GREEN passed `5/5 in 44.70s`; the complete affected Stage-B
plus registration matrix passed `91 passed, 1 xfailed in 291.82s`. Independent exact-byte review
returned `APPROVE_STAGEB_PATH_LOCK_HARDENING`, matching runner SHA-256
`64B4A5AAE70FEE358DEE3F639B8E9063E72DB6B8813509D9FB9BB16423053B3D` and test SHA-256
`1E7BB88394521E8441FF92FC30AE8CC99A9F1099B531F85A9A77E7CB137094CA`. The approval is bounded to
lexical path, lock identity and no-clobber behavior; no formal Stage-B run or Gate artifact was created.

A Gate-1 precheck follow-up reproduced parent-symlink laundering through the registration path,
output root and fixed Gate-1 artifact paths. The repair removes pre-validation `resolve()` calls,
checks every existing lexical component with `lstat`, and independently reconstructs the registered
`R/shared/gate1` output before accepting the resolver result. Remote evidence was: focused regression
`1 passed in 48.56s`, complete registration/precheck suite `38 passed, 1 xfailed in 199.52s`, and
Gate-1 hardening/cost compatibility `25 passed in 235.45s`. Independent exact-byte review returned
`APPROVE_GATE1_PRECHECK_PATH_HARDENING`, matching precheck SHA-256
`0BE0EA8BA3FCAD46387611E3140E116381FD7EE50344291F02E6D724FCF76808` and registration-test SHA-256
`55916FBD5182EB2D6024BA5EA9A16B66117F51E545BEF704C3584B862B1C10BA`. This is bounded code approval;
Gate 1 was not run and the full implementation remains `NOT_READY`.

Stage-B interruption recovery was then tested from two real publication cuts: a periodic/final ledger
existing before its checkpoint, and a complete checkpoint+ledger existing before baseline/phase-marker
publication. Exact existing regular files may now be reused without changing inode only when their bytes
match the recomputation; mismatches and impossible marker-without-baseline states remain fail closed.
The first exact candidate passed `98 passed, 1 xfailed` plus a 44-case Gate compatibility matrix, but
independent exact-byte review returned `REJECT_STAGEB_PARTIAL_PUBLICATION_RECOVERY`: the public phase
builder still followed a dense-checkpoint parent alias, and final-pair detection/read occurred through
`Path.exists()`/pathname `torch.load()` after the pre-lock path check. That rejected candidate must not
be reused.

The replacement reads dense and trained checkpoints through `O_NOFOLLOW` descriptors, binds
`lstat`/`fstat` device+inode before and after one byte read, and uses those same bytes for SHA-256 and
`torch.load`. CLI state detection now performs regular-file checks inside the seed lock; phase
checkpoint/ledger/baseline readers independently reject parent aliases. The expanded dense-alias RED
failed as expected, then targeted recovery/path tests passed `5/5 in 47.25s`; final Stage-B plus
registration passed `98 passed, 1 xfailed in 295.42s`; final Gate compatibility passed `44 passed in
285.70s`. Independent review returned `APPROVE_STAGEB_PARTIAL_PUBLICATION_RECOVERY`, matching runner
SHA-256 `50F4469D82C4F2530741DB7E3D7B88C5517B73A91D6F087AF6342E04146F4F84`, core SHA-256
`47342FFE2BC83481F76D004840C76D9FF72F79BC8BF8D7DAEF4B5ABA818A7670`, and test SHA-256
`9BB46DE26A8C9A38F7AA97F31F3D5F1546F189B1F8AD64872340F992C03E378D`. This remains bounded code
approval; no Stage-B seed, Gate, registration or paper result was run or minted.

## Gate-4 caller-evidence formal lock

A new RED established that the pure Gate-4 statistic API accepted caller-supplied dictionaries with
`formal=True`; the only extra restriction was the bootstrap count/seed, so this surface could mint a
formally named report without any registered producer or provenance. The candidate repair makes valid
`formal=True` calls stop before evidence parsing and changes every synthetic/non-formal result to the
explicit `chronotransport-r2-gate4-test-only-v1` schema with `formal_evidence=false`. Remote evidence is
focused `1 passed`, forged-payload-with-recomputed-hash targeted `1 passed in 105.28s`, full Gate-4
`13 passed in 242.40s`, and registration source coverage `1 passed`.
Exact SHAs are `A581D71338B130C2FF0ECB2B833B29F1B7B1FD5A8F5C36E7A24BC7B954B1A75F` and
`5C0FFAF398EC45958045C46CE714BE391E987197532F60409D840F6AAAB4506E`. Independent exact-byte
review returned `APPROVE_GATE4_CALLER_EVIDENCE_LOCK_FINAL`, so this bounded test/formal boundary is
`tested_and_bounded_code_approved`; no formal Gate-4 workflow or result was created.

## Stage-C measured-cost flag lock

The broad reachability audit also found that formal Stage-C accepted `cost_is_measured=False`: the field
was present in the exact schema but only required to have Python `bool` type. A RED case that ran the
production runtime and then changed that summary field to `False` failed with `DID NOT RAISE`. The
production validator now requires the field to be exactly `True`.

The first complete test rerun after that change produced `39 failed, 32 passed, 1 skipped in 63.44s`,
because the existing formal toy and ViT fixtures both constructed the runtime with `measured_cost=None`
and therefore correctly advertised proxy cost. This intermediate failure is retained as integrity
evidence. The fixtures now bind an explicit test-only measured-cost table, without claiming a registered
profile. Final remote evidence is focused `1 passed, 71 deselected in 45.26s` and complete Stage-C
`71 passed, 1 skipped in 76.60s`. Exact SHAs are
`5BDC1862AD90F1D0A6134ADD778D5978A536848EEAD63EDE973444CBCA5577C4` for `stage_c.py` and
`C92FED397F69E03BE6F0189483250F8132579DD844C521CE3E17BEF0B3A262D7` for its test. Independent review
returned `APPROVE_STAGEC_MEASURED_COST_FLAG_LOCK`, making only this false-boolean path
`tested_and_bounded_code_approved`. A runtime boolean is not immutable cost-profile provenance, and no
formal runner currently binds it to registration or Gate-1 cost bytes.

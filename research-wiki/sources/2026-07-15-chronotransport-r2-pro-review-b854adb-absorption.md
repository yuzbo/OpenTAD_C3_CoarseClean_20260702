---
type: external_code_audit_absorption
route: ChronoTransport CT-P3R-3S-r2
date: 2026-07-15
review_sha: b854adb4f4c9235580b5e58c3f3255db6e9adbc0
source_sha256: 1A7B9D5AEA47302AC7BCB29DB9EF54DAD97CF3D45DF1536691CB9B536EC4C376
status: discussed_and_recorded
experiment_fact: false
---

# Pro review absorption for immutable review snapshot `b854adb`

## Source identity and evidence boundary

- User-supplied source: Codex attachment `3800e20d-86c5-4b3c-9cfb-66c1c6d2b408/pasted-text.txt`.
- Verbatim repository archive:
  `research-wiki/sources/2026-07-15-chronotransport-r2-pro-review-b854adb-verbatim.txt`.
- Source and archive are both 73,605 bytes / 1,430 lines and have identical SHA-256
  `1A7B9D5AEA47302AC7BCB29DB9EF54DAD97CF3D45DF1536691CB9B536EC4C376`.
- Reviewed GitHub snapshot: `b854adb4f4c9235580b5e58c3f3255db6e9adbc0` on
  `codex/chronotransport-r2-implementation`.
- Frozen science specification identity cited by the review:
  `87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8`.
- The reviewer did not run tests, CUDA or Slurm, and its coverage certificate explicitly does not claim
  complete line coverage of every repository file. This record is an implementation audit, not a Gate,
  experiment result, I/R approval, or scientific claim.

## Direct answer

The overall verdict and stop condition are accepted: the snapshot must remain
`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION` / `NOT_READY`, and no formal execution may start.
The concrete patch suggestions are not accepted verbatim. The correct position is:

> conclusion-level agreement, implementation-level qualified agreement.

The principal blockers are factual and mandatory. Two proposed mechanisms—the broad test glob and the
draft Stage-C evidence interface—need refinement before implementation. No Pro suggestion alone changes
the approved specification or supplies authority for A1--A4.

## Finding-by-finding disposition

| Review item | Disposition | Repository evidence / qualification |
|---|---|---|
| Snapshot gate passes for fresh SHA `b854adb` | accept | The branch now resolves to the review-only descendant snapshot, not forbidden `797a2df`. |
| Overall `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION` | accept | Required formal workflows and protocol decisions remain absent. |
| The snapshot is neither implementation commit I nor registration commit R | accept | Commit role is explicitly review-only. This label is useful provenance, but substantive readiness—not the commit message alone—is the real barrier. |
| Missing formal Stage-C, matched-dense and Gate-4 producer/runner chain | accept | The named tools and Stage-C launcher are absent. No 4,200-update workflow or official-population producer is reachable. |
| Current Stage-C primitive is not connected to the real ActionFormer training contract | accept | `ActionFormer.forward_train` returns a loss dictionary, while the Stage-C hook demands one differentiable canonical detector Tensor. |
| Train-mode `loss_normalizer` conflicts with the current success-state invariant | accept | `AnchorFreeHead.losses` advances the EMA buffer during training. A3 must uniquely define successful-state semantics and matched-arm alignment before a production bridge is written. |
| A1--A4 remain unapproved and formal paths must stay locked | accept | The amendment is only `proposed_unapproved`; it cannot silently modify the frozen specification. |
| Registration source vector omits the two changed integration tests | accept | `tests/test_chronotransport_pipeline.py` and `tests/test_chronotransport_vit_adapter_integration.py` are in `b854adb` but absent from `REQUIRED_REGISTRATION_SOURCE_PATHS`. Under this project’s exact-source contract this is a registration-blocking P1 integrity defect, although it does not by itself mint fake evidence. |
| Slurm/physical-GPU1 contract conflict | accept | Governing repository rules prohibit overriding scheduler visibility and require local `cuda:0`; A2 must be approved before any formal launcher is frozen. |
| `cost_is_measured=True` is not immutable provenance | accept | The Boolean lock is necessary but insufficient; formal Stage C still needs exact profile artifact, producer, environment and requested/executed-cost identities bound through registration. |
| Stage-C toy tests are insufficient for production reachability | accept with wording correction | They are valid primitive hardening tests, not evidence that real ActionFormer Stage C is executable. Calling the whole suite “false positive” would overstate the defect. |
| Pure Gate-4 adjudicator slice can remain bounded-approved | accept | The current formal lock/test-only schema is useful, but it is not a formal Gate-4 workflow or result. |
| No P0 was confirmed in reviewed surface | accept as bounded statement | This is not evidence that no P0 exists outside the reviewer’s read surface. |
| Proposed dependency order A1--A4 → real Stage-C contract → runners → Gate-4 producer → registration closure → precheck → I/R → stop-chain | accept with one restriction | Any pre-registration exercise must be synthetic/test-only; official evaluation population and formal artifacts remain untouched until a valid I/R and stop-chain unlock. |

## Concrete suggestions that must not be copied verbatim

### 1. Do not bind registration to a raw `test_chronotransport*.py` glob

Read-only inventory at `b854adb` found 21 matching test files, of which 14 are registered. A raw glob
would additionally demand five old/general/non-r2 files besides the two confirmed omissions:

- `tests/test_chronotransport_core.py`
- `tests/test_chronotransport_opentad_replay.py`
- `tests/test_chronotransport_repository_contract.py`
- `tests/test_chronotransport_stage_a_smoke.py`
- `tests/test_chronotransport_stage_b_formal.py`

Those files may or may not belong to the formal r2 vector; filename prefix alone is not a protocol
classification. The robust implementation is:

1. explicitly add the two confirmed changed integration tests;
2. maintain a canonical classification/manifest for every `test_chronotransport*.py` file as
   `formal_r2_source`, `legacy`, `general_nonformal`, or another frozen category;
3. fail when a newly discovered ChronoTransport test is unclassified;
4. require every `formal_r2_source` entry—and only the approved formal entries—in the exact registration
   source vector.

This preserves completeness without letting an unrelated legacy test silently alter R.

### 2. Treat `FormalStageCDetectorEvidence` as a design sketch, not a frozen API

The proposed fields identify real needs—canonical differentiable loss, per-window regret, normalizer
before/after, forward order, batch and augmentation identity—but the exact interface is not yet approved.
In particular:

- `canonical_detector_tensor` is ambiguous when the official detector owns a loss dictionary;
- separate dense/candidate loss fields must not accidentally create extra or unmatched forwards;
- normalizer updates and rollback must follow the eventual A3 decision;
- dense reference and differentiable counterfactual order/count must follow A4;
- batch-two per-window loss semantics must come from the exact official head, not an invented reduction.

The interface may be implemented only after A3/A4 are approved and independently frozen.

### 3. Do not preserve `_gpu1.sh` naming or physical-index semantics by inertia

The review correctly flags A2, but one proposed missing path is
`scripts/run_chronotransport_r2_stage_c_gpu1.sh`. A future launcher should use an A2-consistent neutral
name and Slurm-assigned single-device semantics unless the approved amendment explicitly selects another
contract. It must not set `CUDA_VISIBLE_DEVICES=1`.

### 4. Do not use an official-population “small GPU precheck” before registration

Before I/R, verification may compile, run unit/integration tests and use synthetic fixtures. It may not
touch the official evaluation population in a way that produces reusable Gate-4 evidence. Formal profiling,
checkpoint evaluation and official-population production belong after registration and the preceding
stop-chain unlocks.

## Absorbed implementation requirements

The next implementation candidate must therefore satisfy all of the following before a new registration
review is requested:

1. Obtain explicit approval and independent spec-only freeze for A1--A4; until then keep every formal
   random, Slurm, Stage-C and registration path fail closed.
2. Define and test a production ActionFormer Stage-C bridge whose official loss dictionary, per-window
   regret, forward count/order and `loss_normalizer` transitions are exact and matched.
3. Implement repository-owned 4,200-successful-update candidate and A-only matched-dense workflows with
   shared batch, augmentation, LR and EMA evidence.
4. Implement a repository-owned Gate-4 official-population producer/runner with post-Stage-C Gate-3
   unlock, candidate/static/checkpoint/profile identities, no-clobber publication and raw-prediction AP
   reconstruction.
5. Repair source-vector completeness using explicit formal classification, including the two confirmed
   integration tests and all future formal runners/validators/launchers/tests.
6. Bind measured cost to immutable profile provenance rather than a Boolean assertion.
7. Re-run focused and compatibility checks, obtain an independent complete implementation review, and
   only then consider immutable I followed by registration-only R.

## Persistent status after absorption

- A1--A4: `proposed_unapproved`.
- Overall implementation: `implemented_partial` / `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`.
- Registration: `NOT_READY`; I and R do not exist.
- Gate 1, Gates 2/3, Stage C and Gate 4: not formally run.
- ChronoTransport training Job: none.
- Scientific result / paper number: none.
- Historical `92029ea` negative result remains historical evidence and is neither erased nor overturned.
- No experiment was started or modified while recording this review.

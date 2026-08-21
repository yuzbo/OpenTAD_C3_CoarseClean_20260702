# DUCA True-Time Indirect Sampling Curriculum — Frozen Design

**Date:** 2026-08-21  
**Status:** human approved for implementation and formal experiment preparation  
**Parent revision:** `42dba3f90b37243e7965d18b6707e88e81bf7109`  
**Branch:** `codex/duca-truetime-curriculum-20260821`

## Scientific question

Does preserving physical time throughout the historical indirect, non-uniform,
pre-backbone frame-selection path recover high-IoU localization that is lost when
selected frames are packed and interpreted by the heavy encoder as an equally
spaced sequence?

The learned task remains framewise binary actionness plus boundary importance.
Frame indices are produced only by the existing deterministic exact-K decoder;
the scout must not directly learn indices. This isolated experiment keeps
`K=384`. Dynamic outer-K remains the eventual headline route but is not enabled
until the physical-time hypothesis is decided.

## Paired arms

1. `RANKPACK_K384`: historical indirect selector and selected-rank temporal
   interpretation, retrained under the same 60-epoch schedule as the treatment.
2. `TRUETIME_K384`: identical semantic scout, exact-K positions, losses,
   optimizer, updates, detector, NMS, evaluator and seed, but every selected
   observation retains its original physical coordinate from acquisition through
   heavy feature formation, target assignment, regression decode and pre-NMS
   proposal mapping.

Existing dense, exact-uniform and random results are references only. This
experiment must not repeat those matrices.

## True-time heavy path

`TRUETIME_K384` must not merely attach timestamps after a normal VideoMAE pass.
It must prevent non-contiguous observations from being treated as unit-spaced
motion. The minimal accepted design is:

- preserve strictly increasing source positions alongside gathered RGB;
- use per-observation spatial patch embedding before any temporal mixing that
  assumes adjacency;
- condition temporal mixing on physical coordinate differences (`delta_t`) or
  an equivalent physical relative-position operator;
- preserve validity/observation masks through the sparse feature path;
- perform detector location construction, assignment and regression in physical
  coordinates, or reconstruct onto the physical detector grid before those
  operations;
- decode proposals to physical seconds before unchanged official NMS;
- record requested/effective/executed K at the actual heavy call boundary.

The implementation must reuse existing OpenTAD/AdaTAD modules and the historical
DUCA selector. It must not introduce a new generic training framework.

## 60-epoch curriculum

All comparisons use 6,000 successful detector updates and the same checkpoint
selection rule.

### Epochs 1–20: decoupled semantic warm-up

- detector receives exact-uniform K=384 through the arm's own temporal path;
- ASFormer semantic scout learns actionness and boundary objectives;
- scout outputs do not control hard acquisition;
- detector loss does not update the scout through the selection bridge.

### Epochs 21–40: cosine homotopy

Let `p=(epoch-20)/20` and `alpha=0.5*(1-cos(pi*p))`. Mix calibrated retention
rates, not integer frame indices:

`rate = (1-alpha) * uniform_rate + alpha * semantic_rate`.

The existing deterministic decoder then produces sorted unique exact-K positions.
The detector-to-scout bridge is ramped with the same `alpha` up to its frozen
weight.

### Epochs 41–60: joint training

- semantic acquisition is fully enabled;
- actionness and boundary supervision remain active to preserve the indirect
  semantic meaning;
- detector feedback may use only the existing bounded differentiable bridge;
- hard positions remain deterministic and exact-K;
- final/final-EMA is preregistered; intermediate validation cannot select an
  epoch.

This is a curriculum/continuation method: the input distribution moves
continuously from stable uniform sampling to learned semantic sampling.

## Data, evaluation and recovery

- THUMOS14 canonical training/validation mapping and official evaluator;
- identical detector, losses, NMS, augmentation, optimizer-update count and seed
  across the paired arms;
- no validation/test GT, teacher, prediction cache or checkpoint selection in
  acquisition;
- resumable `.pth` every 5 epochs, retaining latest 3 plus milestones and final;
- resume state includes model, optimizer, scheduler, AMP scaler, epoch/update,
  RNG/DataLoader state and DUCA curriculum state.

## Decision metrics and stop rule

Primary: Average-mAP and mAP at tIoU 0.6/0.7.  
Diagnostics: short-action mAP, bilateral endpoint coverage, maximum physical
sampling hole, actual heavy executed K, throughput, memory and full-path cost.

The physical-time hypothesis passes only if `TRUETIME_K384` improves high-IoU
localization without a mismatched update, data, evaluator or compute budget. A
negative or null paired result stops this physical-time implementation from being
promoted; it does not erase the historical indirect-selection result.

## Execution order

Builder minimal plan and implementation -> independent Critic review -> Evaluator
PRE_RUN -> two full 60-epoch N16R4 jobs. No benchmark job may start before the
clean frozen revision, formal configs, commands, result roots and PRE_RUN record
are bound.

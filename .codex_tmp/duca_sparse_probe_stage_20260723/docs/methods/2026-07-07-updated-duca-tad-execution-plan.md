---
updated: 2026-07-07
status: active
scope: Consolidated execution plan after the latest external GPT reviews on GAS-VT, PAction learned, detector utility, and CVPR-grade sparse acquisition.
out-of-scope: Reporting final experiment numbers, claiming CVPR readiness, or replacing controlled experiments with static review conclusions.
---

# Updated DUCA-TAD Execution Plan

## North Star

The project goal is not to make GAS-VT more complicated. The goal is to build a
deployable pre-backbone sparse temporal acquisition module for TAD that selects
at most 384 positions and improves detector mAP, especially high-IoU
localization, under matched AdaTAD settings.

The paper story should center on detector-utility-calibrated acquisition:

> Sparse TAD is not solved by action coverage. A selector must acquire boundary
> and localization evidence that the detector actually uses.

## Current Route Lock

- GAS-VT is demoted to an engineered Stage1 diagnostic baseline.
- PAction learned fixed_384 is the current strongest Stage1 p_action-supervised
  baseline and should be preserved as the main comparator.
- PAction score-only lattice replacement is only a diagnostic probe for
  detector-compatible decoding geometry. It is not the final method.
- Stage2 dense-teacher utility selector is the next scientific main route.
- Stage3 selector-detector joint training is the first route that can support an
  honest end-to-end claim.

## Claim Gates

A method claim is not allowed unless all of the following are satisfied:

1. selected positions are 384 or fewer;
2. selection uses train-only supervision and no val/test GT, teacher prediction,
   oracle boundary, or cached detector prediction;
3. ledger validation reports selected-count, boundary support, hole/gap, repair
   or replacement counts, and utility provenance;
4. comparisons use the same commit, source, AdaTAD config, pretrain, eval
   epochs, and detector settings;
5. high-IoU mAP does not collapse;
6. Stage3 end-to-end claims show detector-loss gradient or parameter movement in
   the selector.

## Immediate Execution Order

### Step 0: Finish Current Running Evidence

Let the currently running PAction and dense teacher jobs finish unless they
crash. Do not interrupt them to launch speculative variants.

Outputs required:

- PAction learned fixed/dynamic detector results and ledgers;
- dense AdaTAD teacher checkpoint;
- dense teacher train-only utility export evidence.

### Step 1: Build Matched Stage1 Diagnosis

Purpose: explain why PAction learned beats GAS-VT and lock the baseline.

Required comparisons:

- uniform_384;
- raw p_action top-k fixed_384;
- PAction learned fixed_384;
- PAction learned no-repair or no-hole variant if available;
- GAS-VT fixed_384 with corrected apply-time budget conditioning;
- GAS-VT no-CVaR/no-repair diagnostic variants only if prechecks pass.

Primary outputs:

- mAP curves by eval epoch;
- mAP at high IoU;
- selected-count histogram;
- boundary r1/r2/r4;
- start/end boundary distances separately;
- max/p95/p99 holes;
- p_action/delta/boundary-score top-k overlap;
- repair or replacement counts.

### Step 2: Detector-Utility Selector

Purpose: test whether dense AdaTAD teacher utility trains a better acquisition
policy than p_action-only supervision.

Implementation requirements:

- export train-only utility from dense AdaTAD teacher;
- record utility source per row: proposal score, point responsibility, cls/reg
  loss, saliency, or counterfactual utility;
- separate utility heads where possible: classification, start boundary, end
  boundary, uncertainty/context, false-positive risk;
- train a selector under fixed_384 budget;
- run AdaTAD with the same detector settings as baselines.

Main claim test:

> Stage2 must beat PAction learned fixed_384 under matched settings before it
> can become the paper's main method.

### Step 3: Joint Selector + AdaTAD

Purpose: prove genuine end-to-end sparse TAD optimization.

Minimum gates:

- straight-through or differentiable hard selector in the same graph as AdaTAD;
- detector loss backward produces non-zero selector gradient;
- selector parameters or selected-position distribution change due to detector
  loss;
- anti-collapse checks for count, duplicate positions, max hole, entropy, and
  boundary support;
- sparse detector mAP improves or at least does not collapse versus Stage2.

### Step 4: True-Time Geometry

Purpose: resolve selected-axis remapping as a high-IoU confounder.

Experiment:

- compare selected-axis AdaTAD against true-time or hybrid sparse geometry for
  the strongest 384 selector;
- feed selected time/cell width/valid mask to the detector path;
- report mAP@0.6 and mAP@0.7 separately.

## What Not To Do

- Do not use 640/768 as the main sparse claim.
- Do not claim GAS-VT is the main method unless it becomes true sequential VT
  and wins.
- Do not describe repair or lattice replacement as pure learned intelligence.
- Do not treat engineering no-leak checks as a method contribution by
  themselves.
- Do not start large ablation matrices before one 384-or-less route beats the
  strong baselines.

## Decision Tree

If PAction learned remains best:

- treat it as the empirical Stage1 baseline;
- use it to motivate detector utility, not as the final method.

If PAction lattice improves:

- conclude geometry-compatible decoding helps;
- keep it as a diagnostic bridge, then replace hand constraints with learned
  geometry or detector utility.

If Stage2 improves over PAction:

- promote detector-utility-calibrated acquisition as the main method.

If Stage2 fails:

- inspect utility export quality, true-time mismatch, and whether teacher utility
  is only proposal score rather than localization utility.

If Stage3 improves and selector gradients are verified:

- promote end-to-end selector-detector optimization as the final model claim.

## Updated Paper Claim

The target paper claim should be:

> We identify that actionness-driven sparse acquisition is insufficient for
> high-IoU TAD localization, and propose detector-utility-calibrated
> pre-backbone acquisition that learns what temporal evidence the detector needs
> under a strict 384-or-less observation budget.


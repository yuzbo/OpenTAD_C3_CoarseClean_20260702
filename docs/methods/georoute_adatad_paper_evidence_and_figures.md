# GeoRoute-AdaTAD Paper Evidence and Figure Plan

**Status:** `designed`
**Inputs:** only validated structured result records; no committed plot output,
server log, checkpoint, or prediction file.

## Paper Claim Ladder

The manuscript must make only the strongest claim supported by the completed
stage.  The initial paper story is a hypothesis, not a conclusion:

> Native, structured spatial routing can allocate a fixed heavy-backbone token
> budget more effectively than unstructured routing for high-precision offline
> temporal localization, while reducing measured end-to-end cost.

The story has three necessary links.  Missing any link makes the claim smaller.

1. **Mechanism:** a single heavy VideoMAE path consumes native tubelets under
   an exact `K` route; P0 verifies shape, count, gradient-estimator semantics,
   and component cost observability.
2. **Decision:** at the same `K`, ROI-only and ROI-plus-residual must be
   compared with uniform, random, and free TokenSelect.  If free TokenSelect
   dominates the high-IoU/cost Pareto, the ROI is not the paper's primary
   mechanism, even if ROIs look intuitive.
3. **External consequence:** only a surviving method may be promoted through
   three seeds, budget curves, a second detector/dataset, and one sealed
   official-test evaluation.

This avoids the likely reviewer criticism that the method is merely a pile of
ROI, token-selection, and adapter engineering.  The novelty is not the word
"ROI".  It is the testable joint hypothesis that a continuous geometric prior
plus residual allocation is a better fixed-cost approximation family for
native VideoMAE tubelets in offline high-IoU TAD than free token routing.

## Required Matched Controls

The main paper cannot omit these controls.

| Control | What it rules out |
| --- | --- |
| Dense-native upper-cost reference | a gain caused only by degrading the detector or changing its temporal interface |
| Fixed lattice/uniform exact-K | a gain caused only by fewer tokens rather than adaptive allocation |
| Fixed lattice + learned geometry side-channel | a gain caused by feeding the detector extra scout geometry rather than using geometry to select native tokens |
| Random exact-K | an accidental benefit from a particular count or padding pattern |
| Free TokenSelect exact-K | the hypothesis that arbitrary token utility is enough and ROI structure adds nothing |
| ROI-only exact-K | the contribution of residual free evidence |
| ROI-plus-residual exact-K | the proposed spatial allocation family |
| Stop-gradient policy | a performance change caused by detector-driven policy learning rather than an extra route head |
| Equal scout and cost accounting | an uncharged low-cost observer or hidden compute advantage |

Dense-native is an upper-compute reference, not a same-budget competitor.  The
fair same-budget comparison set is fixed-lattice, fixed-lattice plus the same
learned geometry side-channel, random, free, ROI-only, and ROI-plus-residual
at identical `K`, source grid, scout protocol, detector, updates, and seeds.

## Figure Set

The plotting tool reads `georoute-paper-result-v1` records and emits files only
to a caller-provided output directory.  The repository stores neither figures
nor results.

| ID | Figure/table | Question answered | Minimum evidence | Prohibited interpretation |
| --- | --- | --- | --- | --- |
| F1 | Architecture schematic | Where geometry, residual routing, and detector feedback enter one heavy path | architecture FigureSpec | Do not depict a second heavy local encoder or resized crop |
| F2 | Accuracy-cost Pareto | Does a matched structured route lie on the measured end-to-end Pareto frontier? | per-seed Avg-mAP, p50/p95, energy, memory, full cost scope | FLOPs alone or an uncharged scout cannot support this claim |
| F3 | High-IoU comparison | Is localization retained at tIoU 0.6/0.7 rather than only Avg-mAP? | per-seed mAP at 0.3--0.7 and short/boundary metrics where defined | Avg-mAP-only superiority |
| F4 | Budget curve | Does the relative ranking persist across `K`? | matched `K` values for all primary controls | selecting the best `K` separately per method without disclosure |
| F5 | Mechanism ablation table | Is every component necessary? | fixed geometry-side-channel, free, ROI-only, hybrid, no residual, no coordinates, no detector policy gradient | claiming a component works from a visually plausible ROI |
| F6 | Spatial/temporal structure diagnostics | Does geometry avoid collapse and how often does residual allocation matter? | ROI area, centre velocity, selected-region fraction, residual fraction, coverage and time-bin summaries | semantic actor tracking without annotation evidence |
| F7 | Estimator/stability panels | Does the selected estimator train stably and obey the P0 known-answer test? | finite-gradient KAT, gradient variance, loss/update/retry logs | treating a nonzero gradient as estimator correctness |
| T1 | Main raw table | Are the P2/P3 claims reproducible across seeds? | raw seed rows before pooling; external `render_georoute_paper_tables.py` output | pooled mean without per-seed rows |
| T2 | Generalization table | Is the claim detector/data specific? | second detector or second dataset under a frozen policy/config protocol | calling one AdaTAD-derived result generality |

## Claim Ladder and Current Risk

The paper has to earn its narrative in order.  A more elaborate diagram does
not move a result to a higher rung.

| Rung | Minimum evidence | Allowed wording | Still forbidden |
| --- | --- | --- | --- |
| M0 | P0 CUDA gate | "The native packed path, exact-K accounting, and labelled estimator paths execute." | any accuracy, efficiency, or novelty claim |
| M1 | P1 one-seed development comparison | "The predeclared ROI hypothesis was screened against matched controls." | paper conclusion, generalization, or official-test wording |
| M2 | P2/P3 three-seed development matrix with full-cost protocol | "Under the stated AdaTAD/THUMOS development protocol, structured routing improves the matched high-IoU/cost trade-off." | dataset-wide or detector-agnostic claim |
| M3 | frozen second detector/data and sealed one-time official test | "GeoRoute improves the stated offline TAD trade-off under the reported evaluation scope." | universal superiority or a theorem about mAP |

At the current pre-result stage, reviewers would correctly attack three
things: the proposal could be a collection of familiar modules; the apparent
saving could exclude scout/gather/dense-adapter work; and an appealing ROI
trajectory could be a post-hoc visualization.  The P1 free-TokenSelect stop
rule, P2 paired three-seed rule, the scope-complete cost record, and all-video
diagnostics are not optional polish: they are the evidence that turns those
attacks into falsifiable tests.

## Captions as Claim Guards

Every caption should state the comparison set, data split, whether costs are
end-to-end, and the statistic.  Examples of allowed caption language:

- "Development-set accuracy against measured per-window end-to-end latency at
  a shared native-token budget; points show individual seeds."
- "High-tIoU mAP under matched exact-K routing.  Dense-native is an
  upper-compute reference and is not part of the equal-budget comparison."
- "Route diagnostics describe allocation statistics; they are not spatial
  ground-truth accuracy or evidence of actor tracking."

Do not use "efficient" or "superior" in a caption before both the matched
accuracy and total-cost evidence have passed the predeclared decision rule.

## Reviewer Attack Map

| Likely attack | Required response in evidence, not prose |
| --- | --- |
| "ROI is unnecessary; free tokens can do this." | F2/F3/F4 compare hybrid and ROI-only directly to free TokenSelect at equal K |
| "Your ROI gain is just an extra geometry feature." | F5 includes fixed-lattice plus the same learned geometry-side-channel, while its selected native token lattice remains deterministic |
| "This is only heuristic engineering." | Theory package labels assumptions; F5 isolates geometry, residual, coordinates, and detector policy feedback |
| "Savings ignore local/global overhead." | F2 records decode through NMS p50/p95, memory, and energy with a machine-readable scope |
| "It wins only loose localization." | F3 reports tIoU 0.6/0.7, short-action and boundary diagnostics where valid |
| "The hard route has invalid gradients." | F7 reports score-function KAT and labels ST as biased |
| "ROIs collapse or are cherry-picked." | F6 aggregates all development videos and seeds, never selected qualitative examples alone |
| "It is AdaTAD with arbitrary changes." | F1/T1 retain `[B,384,768]`, detector loss/head/NMS, and enumerate backbone-side changes |

## Result-Blind Flow

1. Before reading metrics, freeze variants, budgets, seeds, update counts,
   split roles, cost scope, selection rule, and schema version.
2. Each run emits one structured record with hashes and raw measurements.
3. `analyze_georoute_results.py --development-only` validates that no official
   test record is mixed into development selection, rejects duplicate run
   identities, and produces an analysis JSON without changing any result.
4. `plot_georoute_paper.py` reads that JSON and the validated records.  It
   cannot invent values and writes figures outside the repository.
5. `render_georoute_paper_tables.py` binds the same record hash to a raw-seed
   CSV, a descriptive summary table, a LaTeX table, and a matched-control
   coverage audit. It cannot decide a paper claim or silently omit seeds.
6. Only after P3 decisions are frozen may a separate sealed official-test
   evidence package be analyzed with an explicit override and reported as
   confirmatory evidence.

## Architecture Figure Source

`georoute_adatad_architecture_spec.json` is an editable FigureSpec source for
F1.  It is intentionally a source specification rather than a checked-in SVG.
Render it outside the repository with:

```bash
python tools/bata/plot_georoute_architecture.py \
  --spec docs/methods/georoute_adatad_architecture_spec.json \
  --output /outside/repository/georoute_architecture.pdf
```

The rendered figure must be reviewed against the runtime audit before use in a
paper.  In particular, its red dashed feedback arrows are labelled either
score-function feedback or biased ST feedback; they must never be redrawn as a
generic exact pathwise gradient.

## Non-Claims

- A Pareto plot does not establish causality without matched controls.
- A lower selected-token count does not establish lower total cost.
- A spatial trajectory plot does not show ground-truth localization.
- A P0 numerical gate or P1 single seed is not paper evidence.
- The theory document does not prove mAP, latency, or novelty.

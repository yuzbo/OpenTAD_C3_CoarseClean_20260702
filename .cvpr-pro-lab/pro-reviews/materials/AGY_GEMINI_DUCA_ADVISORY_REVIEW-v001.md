# AGY Gemini DUCA advisory provenance and mandatory corrections

## Invocation identity

- CLI: `agy`
- Model: `gemini-3.7-flash-high` (`Gemini 3.7 Flash (High)`)
- Requested effort: `high`
- Execution mode: read-only `plan` with sandboxed terminal access
- Source prompt:
  `.cvpr-pro-lab/pro-reviews/prompts/GEMINI_DUCA_COMPREHENSIVE_ANALYSIS-v001.md`
- Complete raw response:
  `C:/Users/skywalker/.fastctx/jobs/j-02swjq/output.log`
- Terminal marker: `GEMINI_DUCA_ADVISORY_READY` verified
- No repository edit, training, remote job, held-out metric access, or browser action was performed by Gemini.

Gemini is an independent technical consultant, not the scientific decision maker. Pro must use the report as a concise
second opinion, inspect its evidence, and make an independent route decision.

## Mandatory factual corrections before scientific use

1. The raw Gemini report calls cross-budget representation mismatch a **proven root cause** of the 0/704 result. This is too
   strong. The terminal Pro/Wiki state says it is the strongest untested explanation. The experiment proved only that the
   frozen-K384 detector, current nested K256/K384/K512 predictions, and frozen transfer space lacked the preregistered joint
   headroom. Multi-budget detector adaptation is a falsifiable hypothesis, not an established repair.
2. For the old `58.39/34.53` end-to-end run, loader/update exposure mismatch is established. Random coarse-stem
   initialization and surrogate-gradient drift are plausible additional explanations but were not isolated as joint causal
   roots. Pro must not treat that multi-cause sentence as a completed causal decomposition.
3. The raw report describes non-contiguous native-tubelet packing as a mechanism defect. Existing evidence has not isolated
   pairing continuity while holding the RGB set and downstream model fixed. It remains a strong representation hypothesis,
   not an observed causal defect.
4. The raw report names the OpenTAD `validation` set in its proposed terminal evaluation. That identity is not yet admitted.
   The exact held-out set remains blocked on the literal 211-versus-212 ID, loader, physical-file, annotation, class-map, and
   evaluator audit. Pro must freeze the complete admitted set before any model work or evaluation.
5. Gemini's probability formula, optimizer values, thresholds, baselines, ablation list, deadlines, and project-level STOP
   rule are recommendations. They are not frozen project facts and must be independently accepted, revised, or rejected by
   Pro. In particular, a failed multi-budget experiment closes that tested mechanism; it does not automatically prove every
   possible DUCA research question false unless Pro explicitly makes and bounds that decision.

## Correct use in the Pro turn

Pro should compare the complete raw report with the neutral evidence ledger, exact GitHub commits, current Wiki history, and
the full-data protocol. The report's route recommendation may be accepted only after the above overclaims are corrected and
the current data-identity blocker is kept explicit.


# ZoomToken R1-TAR32-FKV terminal Critic receipt

## Scope

- task: `ZT-CPTC-TAR32-TERMINAL-001`
- role: the single independent terminal Critic allowed by the frozen route
- candidate: `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`
- formal job: `1260166`
- review mode: result-blind code, mechanism, checkpoint, and protocol audit
- result: `PASS_WITH_BLOCKER`

## Passed checks

- The frozen route alternates dense blocks `0/2/4/6/8/10` and TAR32 blocks
  `1/3/5/7/9/11` on exact current-only K64 support.
- Each odd block's stable per-tubelet K32 mask is derived only from the immediately
  preceding dense block's attention-column mean.
- Odd-block Query/output/MLP execute on K32 while all current K64 remain Key/Value
  context; unselected tokens take identity residual bypass rather than deletion.
- The existing Adapter still executes on all K64 in all twelve blocks, and the
  detector receives a complete dense temporal representation.
- The parameter surface matches the dense carrier; there is no cache, old state,
  new trainable parameter, new loss, dynamic K, or fallback.
- Focused tests cover stable exact-K32 selection, non-selected-token gradient
  dependence through full K/V, all-Adapter gradients, route ledgers, statelessness,
  and real-shape CUDA AMP finite forward/backward.
- Slurm reports `COMPLETED 0:0`; the source and receipts bind the exact candidate;
  epoch-59 exists and contains both `state_dict` and `state_dict_ema`; the launcher
  terminal receipt records `torchrun_exit_code=0`; no hard-error marker was found.

## Decisive blocker and disposition

The formal training job produced no official final-EMA validation, prediction, or
evaluator output. Therefore no accuracy conclusion and no cost admission can yet
be made. Under the preregistered route this is not a model failure: the checkpoint
is valid, so exactly one evaluation-only completion is authorized, using the same
candidate and config, epoch-59 `state_dict_ema`, canonical 211-video/792-item
validation population, official evaluator and Soft-NMS, and a fresh result root.

The Critic does not authorize training, resume, parameter update, a second seed,
cost measurement before accuracy admission, or concurrent Residual Probe work.

# ZoomToken CPTC TAR32 replacement result-blind Evaluator receipt

## Verdict

`PRE_RUN_READY_REPLACEMENT`

## Frozen execution identity

- task/action: `ZT-CPTC-TAR32-TERMINAL-001` / `RPL1_EVALUATION_ONLY_COMPLETION`
- replacement for: Slurm job `1261121`
- candidate: `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`
- checkpoint: epoch-59 `state_dict_ema`
- checkpoint SHA: `fc70557ef00788f8e788d59464d8c392943638c446d949d586fefc68c6d9390b`
- population: canonical validation, 211 videos / 792 ordered items / 411 MP4
- evaluator/postprocess: official evaluator / configured Soft-NMS
- resources: two GPUs / eight CPUs
- training/resume/parameter update: false / false / false

## Result-blind checks

The source is clean and equals the pushed candidate; all frozen checkpoint,
config, annotation, class-map and pretrained identities match. The fresh result
root was absent before submission. The only scientific execution change is the
external inventory check from a top-level regular-file count to recursive
follow-symlink `find -L`; it does not change the model, data population,
checkpoint, evaluator, postprocess or resource contract. The launcher exposes
only `tools/test.py` and no training, resume, optimizer or parameter-update path.

The authorized counters are scheduler ordinal `2` and scientific-attempt ordinal
`1`. A third submission, cost measurement, retry/resume and successor task are
forbidden before fresh terminal Pro adjudication. No result or performance value
was read to produce this verdict.

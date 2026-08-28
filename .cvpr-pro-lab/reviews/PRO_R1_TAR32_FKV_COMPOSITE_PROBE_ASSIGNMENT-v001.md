# ZoomToken R1-TAR32-FKV composite probe assignment

- source: user-pasted Pro response, confirmed by the user as the latest Pro instruction
- decision: `CONTINUE_COMPOSITE_PROBE`
- task: `ZT-CODEX-COMP-PROBE-001`
- experiment: `ZT-R1-TAR32-FKV-S42-E60-v001`
- candidate: `ZoomToken-R1-TAR32-FKV`
- scientific status: composition-first research probe; method novelty not established
- exact base: `2d945e64bdccd09ae2e2916524562e3f388c5a2a`
- branch: `codex/zoomtoken-r1-tar32-fkv-v001`

## Frozen mechanism

Retain the current-only contiguous R1 `8x8/K64` support. VideoMAE blocks
`0/2/4/6/8/10` update all K64 tokens. Each following odd block consumes only the
immediately preceding dense block's attention column mean, selects exact K32 inside each
tubelet with stable native-index tie breaking, computes Query/attention output/MLP only
for those K32 tokens, and uses all K64 tokens as non-detached Key/Value context. The
existing coordinate-lineage Adapter continues unchanged on all K64 tokens. There is no
cache, temporal state, new trainable parameter, new loss, dynamic cardinality or fallback.

## Allowed training-candidate paths

- `opentad/models/backbones/vit_adapter.py`
- `configs/adatad/thumos/georoute_official_r1_tar32_fkv_prebackbone_seed42_v001.py`
- `scripts/run_zoomtoken_r1_tar32_fkv_n16r4.sh`
- `tests/test_zoomtoken_r1_tar32_fkv.py`
- `tools/train.py` only if one literal route allowlist addition is necessary

## Frozen formal treatment

Seed 42; two GPUs; local/global batch 1/2; 60 epochs; the R1 optimizer, scheduler,
AMP, EMA, checkpoint, data, evaluator, Soft-NMS, detector and Adapter contracts remain
unchanged. Only one training cell is allowed. No pilot, auxiliary arm, sweep, second seed,
cache, teacher, distillation, rescue, correction or automatic successor is allowed.

## Decision boundary

The heavy-block proxy (`79.06%` of R1) is architecture accounting only. Accuracy and
probe-grade matched full-stack cost are required before any efficacy or efficiency
interpretation. Every terminal state returns to a fresh Pro review. This task is isolated
from the already submitted v004 cost job `1260095`; neither task consumes the other's
partial evidence or changes the other's frozen protocol.

## Beijing deadlines from Pro

- Builder plan: `2026-08-29T09:30:00+08:00`
- clean candidate: `2026-08-29T20:00:00+08:00`
- Critic: `2026-08-30T00:30:00+08:00`
- Evaluator: `2026-08-30T03:00:00+08:00`
- formal training submit: `2026-08-30T04:00:00+08:00`
- accuracy return bound: `2026-08-30T20:00:00+08:00`
- cost submit: `2026-08-30T23:00:00+08:00`
- post-result Pro: `2026-08-31T12:00:00+08:00`

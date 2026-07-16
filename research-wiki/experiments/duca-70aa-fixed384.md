---
type: experiment
node_id: exp:duca-70aa-fixed384
title: "DUCA 70aa069 fixed-384 official-derived AdaTAD full train"
idea: idea:duca-offline-full-window
verdict: running
confidence: high
commit: "70aa069b895322c2307ffbb13dfdef9fac0d1305"
jobs: "1154971"
updated: 2026-07-11
---

# DUCA 70aa069 fixed-384 official-derived AdaTAD full train

## Raw metrics / observations

截至 2026-07-11 01:02 +08:00，训练到 epoch 41/60。评估 Avg-mAP 依次为 3.84、17.03、24.05、34.02、44.94、49.12、52.06、53.84。最新 tIoU 0.3/0.4/0.5/0.6/0.7 为 73.19/65.89/56.16/44.33/29.65。

## Interpretation

当前曲线持续上升，说明修复后的 fixed-384 可训练；尚无 matched uniform/dense，因此不能判断论文增益。

## Limitations

不是最新 a5e1774；requested K=384 时历史 batch 的 effective K 偶尔低于 384，需审计；未完成 60 epoch。

## Provenance

/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_70aa069_final_20260710_1544

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。

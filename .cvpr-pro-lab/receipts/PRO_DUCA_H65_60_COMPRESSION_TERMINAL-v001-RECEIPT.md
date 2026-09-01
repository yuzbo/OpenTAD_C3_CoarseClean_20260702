# PRO_DUCA_H65_60_COMPRESSION_TERMINAL-v001 — FINAL RECEIPT

- status: `COMPLETED / ACCEPTED_TERMINAL_ADJUDICATION`
- exact_project: `DUCA` / `g-p-6a796fef9a00819194024cf1de3bd697`
- nonce: `DUCA-H65-60-COMPRESSION-TERMINAL-v001-20260824`
- oracle_session: `duca-h65-terminal-v1`
- conversation_id: `6a8c1c88-5680-83ea-8170-910401f870af`
- conversation_url: `https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697-duca/c/6a8c1c88-5680-83ea-8170-910401f870af`
- runtime: `profile61`, CDP `127.0.0.1:14106`, target `79470F3BA0A3857CA4A44D8E1F710FA7`
- model: requested `gpt-5-pro`; resolved browser label `Pro`; picker verified; effort retained at `Pro 5/5`
- submitted_at: `2026-08-24T18:27:02+08:00`
- completed_at: `2026-08-24T18:37:25+08:00`
- actual_attempt_count: `1`
- decision: `STOP_60_EPOCH_COMPRESSION`
- raw_oracle_job: `j-wyl52d` / `C:/Users/skywalker/.fastctx/jobs/j-wyl52d/output.log`
- prompt: `.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_H65_60_COMPRESSION_TERMINAL_ADJUDICATION-v001.md`
- prompt_sha256: `c1e410c10e42e784d157f91709c84aae45f0830e1c58c94e5ff3b5334e77e726`
- final_visible_report: `.cvpr-pro-lab/pro-reviews/runs/duca-h65-60-compression-terminal-v001/PRO_DUCA_H65_60_COMPRESSION_TERMINAL_ADJUDICATION-v001.md`
- final_visible_report_sha256: `a8ac84c1c6f1670b9c0509d715a693a69f7db8031306ee006cfd31392095db76`
- transcript: `.cvpr-pro-lab/pro-reviews/runs/duca-h65-60-compression-terminal-v001/oracle-home/sessions/duca-h65-terminal-v1/artifacts/transcript.md`
- transcript_sha256: `8af5d8d52fd8f16f800295cae7b6c2b4f176086e0bcd69261f85d713d0b39cf8`

## Scientific disposition

Both mature-Stage-1 `30+30` LR-tail arms are clear terminal failures. AM-RPCH25 reaches Avg-mAP/mAP@0.7 `63.22/41.25`; LongCosine-H6000 reaches `63.56/41.01`, versus the historical `30+60` reference `65.1257/43.3137`. LongCosine retains about 18.33% more cumulative relative LR area and a 0.571157 terminal factor, but its improvement is limited to aggregate low/mid-IoU performance and does not recover the high-IoU gate.

Both arms also fail the preregistered rising-tail condition because epoch-29 Avg-mAP is below epoch 24. The accepted Pro adjudication therefore applies the frozen flat/falling branch: stop all 60-epoch schedule tuning, do not run a third scheduler or the conditional `+1000` extension, and retain historical `30+60` as the current H65 training reference.

The allowed next action is a read-only matched-successful-update postmortem over existing checkpoints and logs: actual per-group LR and parameter displacement, curriculum/feedback clocks, selector dynamics, terminal online versus EMA, and high-IoU boundary error. Missing telemetry must be recorded as `NOT_MEASURED`; no new training is authorized by this receipt.

- next_owner: `Independent DUCA Evaluator / Training-Dynamics Analyst`
- next_action: seal `STOP_60_EPOCH_COMPRESSION` and produce `H65_COMPRESSION_POSTMORTEM-v001` from existing artifacts only
- dependency: existing historical 30+60 and two 30+30 checkpoints, logs, LR/update traces, available selector telemetry and predictions
- expected_return_at: when the read-only postmortem is durably recorded
- single_recovery: `none`; any future single-mechanism experiment requires a new evidence-triggered preregistration

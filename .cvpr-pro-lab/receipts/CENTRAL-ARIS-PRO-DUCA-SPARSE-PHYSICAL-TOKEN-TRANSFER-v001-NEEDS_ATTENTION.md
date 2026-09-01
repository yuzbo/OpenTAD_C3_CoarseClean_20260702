# NEEDS_ATTENTION RECEIPT

- dispatch_id: `CENTRAL-ARIS-PRO-DUCA-SPARSE-PHYSICAL-TOKEN-TRANSFER-v001`
- request_id: `PRO_REQUEST_DUCA_SPARSE_TOKEN_PHYSICAL_TIME_TRANSFER-v001`
- invocation_job: `j-ftr1d9` (`EXITED_0`; Oracle scientific capture incomplete)
- oracle_session: `duca-sparse-physical-time-v1`
- intended_project_id: `g-p-6a796fef9a00819194024cf1de3bd697`
- intended_project_url: `https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697/project`
- actual_terminal_project_url: `https://chatgpt.com/g/g-p-6a88618e65748191896e4fa06602a20c-ke-ti/c/6a8d2fed-ee5c-83e9-9e31-24dac31ba1ab`
- actual_terminal_project_title: `课题`
- actual_terminal_conversation_id: `6a8d2fed-ee5c-83e9-9e31-24dac31ba1ab`
- nonce: `DUCA-H65-SPARSE-PHYSICAL-TOKEN-TRANSFER-v001-20260825`
- browser_profile: `61`
- runtime_cdp: `127.0.0.1:14106`
- requested_model: `gpt-5.6-sol`
- resolved_model: `GPT-5.6 Sol` (picker verification in Oracle metadata)
- effort_evidence: `MAX_EFFORT_NOT_SEPARATELY_EXPOSED`
- submitted_at: `2026-08-25T13:24:22+08:00`
- oracle_completed_at: `2026-08-25T14:04:31+08:00`
- receipt_completed_at: `2026-08-25T14:06:08+08:00`
- hard_timeout_at: `2026-08-25T15:24:22+08:00`
- actual_attempt_count: `1`
- recovery: `one read-only same-session liveness inspection consumed; no resubmission`
- terminal_status: `NEEDS_ATTENTION / INVALID_PROJECT_ROUTING_AND_INCOMPLETE_CAPTURE`
- scientific_decision: `NONE`
- selected_route: `NONE`
- implementation_authority: `NONE`
- experiment_authority: `NONE`

## Evidence reconciliation

The streaming receipt recorded the intended DUCA Project and a provisional fresh-conversation identity. Terminal Oracle evidence supersedes that provisional routing claim. The saved `meta.json` and timeout DOM both identify a different Project (`课题`) and a different conversation. The timeout DOM contains a different user request (`Pasted markdown(2).md`) and an incomplete assistant research preamble; it does not contain the DUCA nonce or the requested DUCA adjudication. Oracle terminated its capture after 40 minutes with `incomplete-capture`.

Therefore this invocation is not an admissible exact-DUCA-Project Pro turn and its visible content must not be used for scientific adjudication, implementation, PRE_RUN, or experiment dispatch.

## Durable evidence

- prompt: `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_SPARSE_TOKEN_PHYSICAL_TIME_TRANSFER-v001.md`
- raw log: `C:/Users/skywalker/.fastctx/jobs/j-ftr1d9/output.log`
- Oracle metadata: `C:/Users/skywalker/.codex/oracle/duca-sparse-token-physical-time-v001/sessions/duca-sparse-physical-time-v1/meta.json`
- terminal DOM: `C:/Users/skywalker/.codex/oracle/duca-sparse-token-physical-time-v001/sessions/duca-sparse-physical-time-v1/artifacts/assistant-timeout-2026-08-25T06-04-31-255Z.dom.json`
- terminal screenshot: `C:/Users/skywalker/.codex/oracle/duca-sparse-token-physical-time-v001/sessions/duca-sparse-physical-time-v1/artifacts/assistant-timeout-2026-08-25T06-04-31-255Z.png`
- final visible DUCA report: `ABSENT`

- next_owner: `DUCA Coordinator / central browser scheduler`
- next_action: `terminally ingest this routing failure; a future exact-DUCA Pro attempt requires a new explicit dispatch and fresh nonce`
- dependency: `new browser authority with verified single-owner profile/runtime routing`
- expected_return_at: `not scheduled`
- single_recovery: `consumed; none remaining for this dispatch`


# DUCA parallel Pro routing stress audit

- evidence_class: `EXPERIMENTAL_PARALLEL_ROUTING_EVIDENCE`
- quarantine: `true`
- scientific_decision_eligible: `false`
- role_dispatch_allowed: `false`
- local_verdict: `FAIL_BACKEND_CONCURRENCY`
- joint_round_stop_reason: `FAIL_PROJECT_CROSSWIRE observed in another project`

## Expected identity

- coordinator_thread_id: `019fa3db-42bf-7f30-a0ab-2b8171ab33ed`
- project_id: `g-p-6a796fef9a00819194024cf1de3bd697`
- project_url: `https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697/project`
- github_commit: `63a726a4aaf48ecbf6780bb196de43a890c6b4df`
- current_state_source: `CURRENT_RESEARCH_STATE-v001.md`
- history_source: `MODEL_EXPERIMENT_HISTORY-v001.md`
- confirmed_source_count: `12`
- turn_nonce: `bed5d343023bcce60a35a9233e18f7aa`
- marker: `PARALLEL-PRO-STRESS-20260811::019fa3db-42bf-7f30-a0ab-2b8171ab33ed::g-p-6a796fef9a00819194024cf1de3bd697::bed5d343023bcce60a35a9233e18f7aa`
- oracle_home_dir: `C:/Users/skywalker/.codex/oracle-projects/duca-g-p-6a796fef9a00819194024cf1de3bd697-parallel-stress-20260811`

## Browser and model route

- iXBrowser profile: `61`
- CDP endpoint used: `127.0.0.1:59064`
- browser probe: `Chrome/148.0.7778.167`, Protocol `1.3`
- requested Oracle alias: `gpt-5.5-pro`
- observed web selector: `Pro, 5 of 5`
- observed advanced UI: `Model GPT-5.6 Sol / Effort Pro`
- actual DUCA CDP target: `F9D1D42ECC2077A939E7749798119D92`
- actual conversation URL: `https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697-duca/c/6a7a0bde-9cf8-83ea-ac31-ee2cffc15fcf`
- actual conversation ID: `6a7a0bde-9cf8-83ea-ac31-ee2cffc15fcf`

The Oracle log confirms navigation to the configured DUCA Project, the Pro
picker, prompt submission, and the DUCA conversation URL. The target and
Project route remained stable until the joint fail-close stop. No local
target/tab/Project crosswire was observed before abort.

## Timing and overlap

- preflight recorded: `2026-08-10T17:32:54Z`
- submit observed by: `2026-08-10T17:35:39Z`
- first visible streaming state observed no later than: `2026-08-10T17:37:33Z`
- stop/close initiated: approximately `2026-08-10T17:49:31Z`
- exact DUCA target close confirmed: `2026-08-10T17:49:44Z`

The first-visible-streaming timestamp is a browser observation, not an exact
backend first-token timestamp.

True overlap was observed with at least these distinct Project conversations:

- OnlineTAD target `7F9AD5B3DC0EF218BBC2E43F183DC8A3`, conversation `6a7a0b86-ddb0-83ea-82a2-6f871d1eaf7e`
- SparseTAD target `15C7E91172B0D0E8A55EFA59A8EA2B9A`, conversation `6a7a0b5d-1058-83ea-a3ca-6857a1d1a4e1`

Observed overlap was at least 6 minutes 14 seconds, so this was not merely a
backend queue. The combined test was stopped after another project produced an
explicit `FAIL_PROJECT_CROSSWIRE` signal.

## Response and auditability

The DUCA browser stayed in a streaming/stale state for about fourteen minutes.
The detached run `j-dtcmr5` was stopped and exited `1`. No terminal assistant
response was captured. Post-close readback contained the submitted user prompt
but no terminal assistant message. Therefore the required assistant echo of
Project ID, marker, nonce, commit, and Source versions cannot be verified.

No other-project marker or scientific material was observed in a captured DUCA
assistant response because no assistant response was captured. This is not
evidence that unseen partial generation was isolated.

Pre-submit attempts retained for completeness:

- `j-6roo74`: rejected locally because Oracle did not recognize `--debug`; no browser submit.
- `j-y9knuo`: failed closed before submit because Oracle v0.17.1 expected the obsolete `Pro Extended` option; the diagnostic exposed the current unified `Pro, 5 of 5` and `GPT-5.6 Sol / Effort Pro` UI.

Raw local logs:

- `C:/Users/skywalker/.fastctx/jobs/j-dtcmr5/output.log`
- `C:/Users/skywalker/.fastctx/jobs/j-6roo74/output.log`
- `C:/Users/skywalker/.fastctx/jobs/j-y9knuo/output.log`

## Disposition

This DUCA attempt is classified `FAIL_BACKEND_CONCURRENCY`, not
`PASS_ISOLATED`, because the joint round was force-aborted before a terminal
identity-bearing response could be audited. The joint round root failure is
`FAIL_PROJECT_CROSSWIRE` from another project. The response is quarantined;
nothing from this attempt may enter ChatGPT Project Sources, scientific state,
route decisions, claims, or Builder/Critic/Evaluator queues.

Default policy is restored: long Pro/Project discussions sharing one iXBrowser
profile are serialized across Projects; Source mutations remain mutually
exclusive; every future scientific discussion must use a fresh conversation,
new nonce, exact Project identity, and independently verified model/effort.


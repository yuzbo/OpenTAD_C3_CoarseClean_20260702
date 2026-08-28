# ZoomToken BPNS-R1 v002 terminal Pro review receipt

## Browser and capture identity

- Exact Project: `g-p-6a79701398bc8191a9ef61db6302b24b`
- Conversation: `6a90d60e-e71c-83ea-8f84-8d48d111c251`
- URL: `https://chatgpt.com/g/g-p-6a79701398bc8191a9ef61db6302b24b-zoomtoken/c/6a90d60e-e71c-83ea-8f84-8d48d111c251`
- Oracle session: `zoomtoken-bpns-v002-244b776d-pro`
- Nonce: `ZOOMTOKEN-BPNS-R1-V002-TERMINAL-PRO::g-p-6a79701398bc8191a9ef61db6302b24b::244b776d14f1db2bbc509ab3457bd77c`
- Observed browser model label: `GPT-5.6 Pro`
- Submitted: `2026-08-28T00:27:41Z`
- Completed: `2026-08-28T00:45:18Z`
- Transport: attachment-only; `browserInlineFiles=false`; six requested attachments uploaded
- Actual scientific submissions: `1`; follow-ups: `0`

The valid invocation completed with return code 0. The prior pre-contact metadata error,
logged-out attempt and user-interrupted marker-only conversation were not scientific
submissions and are preserved only as execution receipts.

## Durable evidence

- Full visible Chinese response:
  `.cvpr-pro-lab/reviews/PRO_BPNS_R1_V002_TERMINAL_PARITY_FAILURE_REVIEW_RESPONSE-v001.md`
- Streaming receipt:
  `.cvpr-pro-lab/reviews/PRO_BPNS_R1_V002_TERMINAL_PARITY_FAILURE_REVIEW_STREAMING_RECEIPT-ATTEMPT4-v001.json`
- Terminal receipt:
  `.cvpr-pro-lab/reviews/PRO_BPNS_R1_V002_TERMINAL_PARITY_FAILURE_REVIEW_TERMINAL_RECEIPT-ATTEMPT4-v001.json`
- Raw Oracle run directory:
  `.cvpr-pro-lab/reviews/runs/zoomtoken-bpns-r1-v002-terminal-pro-attempt4-244b776d14f1db2bbc509ab3457bd77c`
- Transcript SHA-256 recorded by Oracle:
  `33cd3817b4b00f8dc910b9ffea41118de89cad68e26936a06406f26a34e60727`

The terminal receipt's broad `targets_after` snapshot still includes the earlier malformed
tab. The authoritative session metadata, transcript, response and source URL all bind the
completed answer to the exact conversation above; this is a capture-side target-list
limitation and does not alter the scientific response.

## Pro scientific adjudication

- Decision: `REVISE`
- Role contract: `KEEP`
- v002 evidence status: `PERMANENTLY_CLOSED_AS_EFFICIENCY_EVIDENCE`
- Terminal classification:
  `VALID_FROZEN_CONTRACT_EXECUTION__INVALID_SCIENTIFIC_ADMISSION__NO_MODEL_OR_COST_RESULT`
- BPNS-R1 status: single-seed accuracy feasible; efficiency unknown
- Paper status: not publishable as an efficiency result

The implementation correctly enforced its frozen contract. The scientific admission rule
was not identifiable because reported-2dp `61.14` represents `[61.135,61.145)`. The
observed `61.0869609029443100` is `0.0480390970556900 pp` from the nearest compatible
raw value, so the historical comparison is `indeterminate`; the point-center difference
`0.0530390970556900 pp` does not establish a raw-to-raw mismatch greater than `0.05 pp`.
The observation must not replace the historical reference.

## Only next task and deadlines

Task: `ZOOMTOKEN-BPNS-R1-IDENTITY-GATED-FULL-STACK-REPLAY-v003`.

Hard gates cover execution identity and measurement completeness. Historical accuracy is
a nonblocking interval-aware `compatible/incompatible/indeterminate` diagnosis. The run
must preserve `K100,R1,R1,K100,R1,K100,K100,R1`, persist per-pass predictions,
evaluator vectors and identities, and compute primary p50 and gross-energy ratios from
the median of four complete pass estimates per arm. Identity or measurement failure is
protocol-invalid only and does not authorize automatic resubmission.

Beijing deadlines:

- Builder plan: `2026-08-28 10:00`
- Builder candidate: `2026-08-28 14:30`
- Critic: `2026-08-28 16:00`
- Evaluator: `2026-08-28 17:00`
- Formal action: `2026-08-28 17:30`
- Queue check: `2026-08-29 00:00`
- Queue blocker return: `2026-08-29 00:15`
- Scientific return: `2026-08-29 12:00`

Git push, remote write and Slurm/GPU execution require explicit human authorization.

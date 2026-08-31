# ZoomToken R1 depth/sparsity four-arm terminal Pro review receipt

## 1. Invocation identity

- Request: `PRO_R1_DEPTH_SPARSITY_FOUR_ARM_FULLSTACK_PARETO_TERMINAL_ADJUDICATION-v001`.
- Nonce: `ZOOMTOKEN-R1-DEPTH-SPARSITY-FOUR-ARM-TERMINAL-PRO-v001-20260831T220837+0800`.
- Exact Project: `g-p-6a79701398bc8191a9ef61db6302b24b`.
- Conversation: `6a958d01-9768-83ea-b163-9b481bb64856`.
- URL: <https://chatgpt.com/g/g-p-6a79701398bc8191a9ef61db6302b24b-zoomtoken/c/6a958d01-9768-83ea-b163-9b481bb64856>.
- Browser-visible route: `GPT-5.6 Pro` / `Pro`.
- Transport: attachment-only, `browserInlineFiles=false`; 12 logical attachments were delivered in one browser bundle.
- Actual scientific submissions: `1`; follow-ups: `0`.
- Started `2026-08-31T22:17:13+08:00`; completed `2026-08-31T22:53:43+08:00`; elapsed `36m29s`.
- Transcript SHA256: `b6b18cf5651d620f627663e4eefa191753b0a1b0121044ad14a97380739a9bd0`.

The first Oracle invocation contacted the browser but timed out while uploading the same attachment bundle before conversation creation or prompt submission. It had `actual_submission_count=0`, `conversation_snapshot=[]`, and consumed no scientific turn. The second invocation changed only the attachment wait budget to five minutes and completed the sole scientific submission. Oracle's one internal page reload recovered the same conversation and did not send a follow-up.

## 2. Bound sources and implementation

Pro explicitly reported reading Project Sources `PROJECT_CHARTER-v002.md`, `CURRENT_RESEARCH_STATE-v018.md`, and `MODEL_EXPERIMENT_HISTORY-v013.md`. Oracle's immutable manifest confirms delivery of 12 logical attachments. The response correctly enumerated the first 11 but named `.cvpr-pro-lab/state.json` as item 12, whereas the actual manifest item 12 was `AGENTS.md`. This is a nonfatal attachment-name reporting discrepancy: all terminal science authorities, raw profile/terminal, role file and GitHub identities were correctly bound, but the response's attachment list is not treated as the transport authority. It bound:

- repository <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>;
- implementation branch <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-r1-depth-pareto-v001>;
- exact implementation <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b82441c1aa2663069033d394794298d5c723bbb6>;
- research-memory branch <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-cvpr2027>;
- terminal-doc commit <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/51455628>.

It found no scientific provenance blocker. Its comment that `.cvpr-pro-lab/state.json` was an older coordination snapshot is retained only as a source statement because that file was not in the Oracle attachment manifest. The later request, raw profile/terminal, durable terminal receipt and GitHub terminal-doc commit were consistent and authoritative.

## 3. Independent adjudication

```text
decision=PIVOT
engineering_decision=PASS_STRONG
protocol_decision=VALID_FOR_NARROW_CLAIM_GRADE_SYSTEMS_NEGATIVE
scientific_decision=VALID_HIGH_CONFIDENCE_NEGATIVE_FOR_B_C_D_FIXED_POINTS
paper_claim_decision=NEGATIVE_ABLATION_ELIGIBLE_NOT_POSITIVE_HEADLINE
terminal_disposition=STOP_R1_FIXED_DEPTH_SPARSITY_FOURPOINT_AS_CURRENT_EFFICIENCY_ROUTE
role_contract_decision=KEEP
provenance_blocker=NONE
```

Pro confirmed that job `1262120` permanently stops the three frozen B/DSR6-KV, C/MOD32-KV and D/DROP32 points as the current efficiency route, and stops local sweeps of K, split depth, fixed refresh ratio or equivalent identity-bypass variants on the same execution base. The valid paper statement is a narrow negative ablation: on one RTX 4090, complete THUMOS14 validation and a 16-pass matched full-stack replay, all three fixed points worsened p50 by about 10–11% and gross GPU energy by about 6–10%; D's memory reduction does not make it a Pareto survivor.

It explicitly rejected official-test, matched-training, multi-seed, cross-hardware/dataset/detector, Online TAD, boundary-protection, universal depth-sparsity failure and positive efficiency-headline claims.

## 4. Why the route pivots

The terminal profile exposes a dominant serial input path: descriptive pooled p50 values are about `2867.53 ms` full-stack, `2688.44 ms` input pipeline, `85.43 ms` model-forward CUDA and `51.36 ms` heavy-backbone CUDA. These pooled components are not additive, but they show that repeated serial sliding-window decode dominates the current full-stack execution. Pro therefore refused a fifth token operator or fixed sparsity point. It froze one last full-stack viability closure that removes deterministic repeated decode without changing either model's output.

## 5. Unique next task

`ZOOMTOKEN-ORDERED-VIDEO-DECODE-REUSE-R1-K100-FULLSTACK-VIABILITY-CLOSURE-v001`

Scientific question: under a completely symmetric ordered-video rolling decode-reuse execution path, can R1/K64 achieve at least 5% lower complete-pass wall time and gross GPU energy than K100 while preserving exact input/prediction/evaluator identity and not increasing allocated/reserved memory by more than 5%?

Execution base and branch:

```text
execution_base=b82441c1aa2663069033d394794298d5c723bbb6
new_branch=codex/zoomtoken-ordered-video-reuse-r1-k100-v001
```

Only these new files are allowed:

```text
tools/bata/profile_zoomtoken_ordered_video_reuse_r1_k100_cost.py
scripts/run_zoomtoken_ordered_video_reuse_r1_k100_cost_n16r4.sh
tests/test_zoomtoken_ordered_video_reuse_r1_k100_cost.py
```

No dataset/transform/model/backbone/Adapter/detector/evaluator/NMS/checkpoint/training-config modification is allowed. No new training, resume, seed, B/C/D, alternate arm, async prefetch, worker sweep, GPU/model/prediction/hidden cache, official test, threshold change, retry or replacement is allowed.

The iterator must process canonical windows in order; retain only a bounded CPU uint8 rolling buffer for the current/next overlapping window; decode a source frame at most once per video/arm/pass; reset on video, discontinuity, arm and pass; and produce exactly the same pre-H2D tensors, labels and metadata as the legacy loader. Decode remains inside the timed full-stack boundary.

Formal population and order:

```text
training_population=NONE
evaluation_population=THUMOS14 complete canonical validation, 211 videos / 792 ordered windows
official_test_opened=false
formal_pass_order=K100,R1,R1,K100,R1,K100,K100,R1
warmup_windows_per_pass=50
formal_submission_limit=1/1
resource_limit=1 Slurm-visible RTX 4090, 6 CPUs, 16h
```

Acceptance requires exact tensor/prediction/evaluator/population/power/artifact parity and all four gates: R1/K100 median-four complete-pass wall-time `<=0.95`, gross-energy `<=0.95`, allocated memory `<=1.05`, reserved memory `<=1.05`. A valid cost failure produces `STOP_R1_CURRENT_CONTIGUOUS_SUPPORT_AS_SINGLE_GPU_EFFICIENCY_ROUTE`; an objective protocol blocker produces no scientific decision and has no replacement. Either outcome returns to a fresh Pro within one hour.

## 6. Beijing deadlines

```text
role_rules_sync_due_at=2026-09-01T00:30:00+08:00
builder_plan_due_at=2026-09-01T02:00:00+08:00
builder_candidate_due_at=2026-09-01T18:00:00+08:00
critic_due_at=2026-09-01T21:00:00+08:00
evaluator_due_at=2026-09-02T00:30:00+08:00
precheck_due_at=2026-09-02T02:00:00+08:00
formal_action_due_at=2026-09-02T03:00:00+08:00
queue_state_check_due_at=2026-09-02T18:00:00+08:00
queue_blocker_return_due_at=2026-09-02T18:15:00+08:00
scientific_return_due_at=2026-09-04T12:00:00+08:00
expected_return_bound=PT16H_AFTER_ALLOCATION_START_PLUS_PT1H_TERMINAL_PACKAGING
```

Role contract remains `KEEP`; only this concrete STOP/PIVOT/task is synchronized into the current research state. Mandatory fresh post-result Pro review remains frozen.

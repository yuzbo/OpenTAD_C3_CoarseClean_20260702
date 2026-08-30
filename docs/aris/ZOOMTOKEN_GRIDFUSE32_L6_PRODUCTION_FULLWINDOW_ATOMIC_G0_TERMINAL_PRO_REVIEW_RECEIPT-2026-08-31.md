# ZoomToken GridFuse32-L6 production-full-window atomic G0 fresh Pro review receipt

## Browser and transport audit

- Request ID: `PRO_GRIDFUSE32_L6_PRODUCTION_FULLWINDOW_ATOMIC_G0_TERMINAL_NEGATIVE_ADJUDICATION-v001`
- Nonce: `ZOOMTOKEN-GRIDFUSE32-L6-PRODUCTION-FULLWINDOW-G0-TERMINAL-PRO-v001-20260831T055129+0800`
- Exact Project: `g-p-6a79701398bc8191a9ef61db6302b24b`
- Conversation: `6a94a6c7-e0a4-83e9-b4cc-e5dd883cf6b6`
- URL: https://chatgpt.com/g/g-p-6a79701398bc8191a9ef61db6302b24b-zoomtoken/c/6a94a6c7-e0a4-83e9-b4cc-e5dd883cf6b6
- Browser-visible model: `GPT-5.6 Pro`; picker/effort route: `Pro`
- Transport: attachment-only; `browserInlineFiles=false`
- Attachments: 9, all named and acknowledged in the response
- Actual submissions: 1; follow-ups: 0
- Oracle status: completed
- Response: `.cvpr-pro-lab/reviews/PRO_GRIDFUSE32_L6_PRODUCTION_FULLWINDOW_ATOMIC_G0_TERMINAL_NEGATIVE_ADJUDICATION_RESPONSE-v001.md`
- Transcript: `.cvpr-pro-lab/reviews/runs/zoomtoken-gridfuse32-l6-production-fullwindow-g0-terminal-pro-v001/oracle-home/sessions/zoomtoken-gridfuse32-l6-production-fullwindow/artifacts/transcript.md`

The submitted prompt and the visible response both bound the latest implementation using all three links:

- Repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- Branch: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001
- Exact reviewed commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/dde46aee17f10bf793e5407055fc7b3416d93205

Pro explicitly treated the exact commit object, not the moving branch head or local worktree, as the code identity.

## Adjudication

- Decision: `PIVOT`
- Current route: `STOP_GRIDFUSE32_L6_EXACT_ROUTE_VALID_G0_NEGATIVE`
- Engineering: `PASS_STRONG`
- Protocol: `PROTOCOL_COMPLETE_VALID_G0_NEGATIVE`
- Science: `VALID_NEGATIVE_FOR_EXACT_GRIDFUSE32_L6_EFFICIENCY_FEASIBILITY`
- Evidence grade: `DECISION_GRADE_WITH_NONFATAL_PROVENANCE_METADATA_BLOCKER`
- Role contract: `KEEP`; no replacement text

The nonfatal provenance blocker is that the exact config retains old GATED task/failure metadata and bucket-level token labels. The launcher, profiler, profile and terminal receipt bind the actual full-window task, ordinals, shape and gates, so the mismatch prevents claim-grade self-containment but does not invalidate the decision-grade negative.

GridFuse32-L6 is permanently closed as the exact route. No repair, rerun, replacement, scheduler ordinal 3, G1, G2, training, pair/depth/orientation sweep, compile/Triton/CUDA rescue or renamed replay is authorized.

## Unique next task

`ZOOMTOKEN-R1-DEPTH-SPARSITY-READONLY-FOUR-ARM-FULLSTACK-PARETO-CLOSURE-v001`

Scientific question: on the same physical GPU and complete decode-to-Soft-NMS path, do any of the already-trained frozen checkpoints `R1/FULL64`, `DSR6-KV`, `MOD32-KV` or `DROP32` form a real latency-energy-memory Pareto survivor?

Frozen arms:

- A: R1/FULL64, job `1249099`, epoch-59 EMA
- B: DSR6-KV, job `1252527`, epoch-59 EMA
- C: MOD32-KV, job `1252180`, epoch-59 EMA
- D: DROP32, job `1252179`, epoch-59 EMA

Allowed files only:

- `tools/bata/profile_zoomtoken_r1_depth_pareto_cost.py`
- `scripts/run_zoomtoken_r1_depth_pareto_cost_n16r4.sh`
- `tests/test_zoomtoken_r1_depth_pareto_cost.py`

Formal protocol: 211 videos / 792 ordered loader items, one GPU, one continuous NVML sidecar, 50 warmup windows per pass, and Williams order `A B D C / B C A D / C D B A / D A C B`. Primary estimates are medians of four complete pass-local p50 and total-joule estimates per arm; pooled-window estimates are forbidden.

For any candidate relative to FULL64, all must hold: p50 ratio `<=0.95`, gross-energy ratio `<=0.95`, allocated ratio `<=1.05`, reserved ratio `<=1.05`. A valid complete terminal with at least one survivor is `PARETO_SYSTEMS_SURVIVOR_PENDING_FRESH_PRO`; no survivor is `STOP_R1_FIXED_DEPTH_SPARSITY_FOURPOINT_AS_CURRENT_EFFICIENCY_ROUTE`; incomplete construction/identity/artifact/power/OOM/walltime evidence is `BLOCKER_NO_SCIENTIFIC_COST_DECISION`.

Exactly one formal job is allowed: 1 GPU, 5 CPUs, 16 hours, formal submission `1/1`, no replacement. No new model, training, seed, arm, GridFuse/RACER rescue or CPTC successor is authorized.

## Beijing deadlines

- Builder plan: `2026-08-31T07:00:00+08:00`
- clean/pushed candidate and focused tests: `2026-08-31T12:00:00+08:00`
- independent Critic: `2026-08-31T13:15:00+08:00`
- result-blind Evaluator/construction witness: `2026-08-31T14:45:00+08:00`
- formal action: `2026-08-31T15:15:00+08:00`
- latest start or scheduler blocker: `2026-08-31T18:00:00+08:00`
- terminal evidence to fresh Pro: `2026-09-01T11:00:00+08:00`


# DUCA DSH Cycle 2 ARM/Split Recheck Receipt

- date: 2026-08-18
- scope: snapshot `cf7207301d0f29204fd64d704f90a5cac6f305c3`; constructor/split contract only
- runnercommandshape: `node E:/DeskTop/TAD/健身/external/dsh-anchored-standard/verify/run-verify.mjs --preset anchored-standard --task "..." --cwd "E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702"`
- required header: `anchored-standard`, `deepseek-official`, `deepseek-v4-pro`, `max effort`, `256000 tokens`; persona exact `You are a helpful software engineer assistant.`
- required first-request tools: `bash + str_replace_editor`; no Iris/history/stop-after-first
- cwd: `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`
- sessionstate: no session created; no raw `session.jsonl.zstd`; no visible report; no first reasoning line
- error: trusted harness launcher `apps/cli/lib/bin.js` was not found at the documented/default candidate roots checked (`D:/Software/deepseek-harness`, `E:/Software/deepseek-harness`, and the canonical external runner tree). A bounded filesystem probe for `C:/Users/skywalker` timed out without output.
- credentials: not read, printed, or modified
- verdict: NEEDS_ATTENTION; no model conclusion


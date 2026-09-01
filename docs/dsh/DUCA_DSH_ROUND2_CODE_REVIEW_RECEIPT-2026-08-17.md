# DUCA round-2 DSH code-review receipt

- Frozen revision: `90748a1e46efbed760401edf80cec7c5816af0b4` (base `6125654b946cc30c614428ce1141f1903b015867`)
- DSH package/profile: `@deepseek-ai/dsh` `0.1.0-rc.6`; `anchored-standard`
- Provider/model/effort/max tokens: `deepseek-official` / `deepseek-v4-pro` / `max` / `256000`
- Persona: `You are a helpful software engineer assistant.`
- Initial tools: `bash`, `str_replace_editor`
- Session: `session-ae79794b-5e51-4a69-b065-6d09d9ec4cb5`
- Job: FastCtx `j-bkf74e`, completed 2026-08-17
- Unique fingerprint: `/^We need\b/`
- firstReasoningLine: `We need respond in Chinese. Need conduct read-only code review of repo at path. Need inspect commit 90748a1e vs base 6125654b perhaps diff. Need no modification. Need run commands only read. Need examine code, configs, launcher, checkpoints, focused tests. Need verify specifics:`
- weNeedFingerprint: `true`
- turn/end.kind: `completed`
- Acceptance: `ACCEPTED`
- Visible verdict: `FAIL` (implementation and protocol review; no scientific-route or efficacy conclusion)

## Durable artifacts

- Raw session: `DUCA_DSH_ROUND2_CODE_REVIEW.session.jsonl.zstd`
- Complete visible Chinese report and stdout/stderr: `DUCA_DSH_ROUND2_CODE_REVIEW.stdout_stderr.log`

## Material disposition

The accepted reviewer independently confirmed that the frozen candidate does not reduce the actual VideoMAE backbone work, does not instantiate or isolate the six arms, and lacks an executable FIT/CAL/HOLD, checkpoint-recovery and launcher contract. It found the pre-NMS physical-time mapping structurally ordered correctly. This is static-review evidence only: there is no PRE_RUN, data access, GPU/remote execution, metric, cost, or efficacy result.

Next owner: Builder, for the single focused correction permitted in this review round; then the same internal Critic and a new DSH focused recheck.

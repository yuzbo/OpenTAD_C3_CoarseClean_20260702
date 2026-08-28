# ZoomToken BPNS-R1 v003 身份门控完整栈回放启动回执

- 日期：2026-08-28
- 状态：`FORMAL_REPLAY_V003_RUNNING`
- 科学目的：在不新增训练、不改变 K100/R1 模型和测量定义的条件下，完成同硬件八 pass 回放，检验 K64 连续原生支持是否带来真实 decode→Soft-NMS 延迟、显存与 gross energy 改善，并保护高 tIoU、短动作和动作边界。

## 固定候选与独立准入

- base：`e9323448f6cd78b99bb3de53fd9ffb55f3676d65`
- clean/pushed candidate：`8a59d655005b9030d8ea5dc17ee2620844cb587b`
- branch：`codex/zoomtoken-bpns-r1-identity-replay-v003`
- 修改面仅为 `tools/bata/profile_zoomtoken_bpns_r1_cost.py`、`tests/test_zoomtoken_bpns_r1_cost.py`、`scripts/run_zoomtoken_bpns_r1_cost_n16r4.sh`
- local/remote focused verification：Python compile、Shell syntax、`git diff --check` 通过；pytest `21 passed`
- fresh independent Critic：`PASS`
- fresh result-blind Evaluator：`PRE_RUN_READY`
- 远端 source：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_cost_src_8a59d655_v003`；HEAD、remote-tracking ref 与候选 SHA 一致，worktree clean

## 结果盲动态预检

- Slurm job：`1258524`（`zt-bpns-v003-pre-8a59d655`）
- 终态：`COMPLETED 0:0`
- 节点与时间：`g0063`，`2026-08-28 11:57:39–11:58:20 +08:00`
- 回执：`PRECHECK_READY`
- 已核验：exact pushed revision、411 canonical MP4、211 validation videos/792 ordered items、两份 epoch-59 EMA checkpoint、config/data/evaluator/NMS identity、新 result root、单 GPU/5 CPU 合同
- 预检明确 `reads_validation_metrics=false`、`trains_or_resumes=false`

## 唯一正式执行

- Slurm job：`1258526`
- JobName：`zt-bpns-v003-8a59d655`
- 提交次数：`1`
- 开始：`2026-08-28 11:58:39 +08:00`
- 节点：`g0063`
- 资源：`gpu` partition，1 GPU、5 CPU、8 小时；不固定物理 GPU，不覆盖 Slurm 的 `CUDA_VISIBLE_DEVICES`
- result root：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_r1_identity_v003_8a59d655_seed42_20260828`
- log root：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_v003_logs_8a59d655_20260828`
- 顺序：`K100,R1,R1,K100,R1,K100,K100,R1`；每臂四个完整 pass，每 pass 50-window warmup
- 冻结输入：K100 job `1248835` 与 R1 job `1249099` 的 epoch-59 `state_dict_ema`；canonical THUMOS14 411/211/792；既有 evaluator 与 Soft-NMS

## 证据与解释边界

运行期间不读取、汇总或解释 live/partial accuracy、cost、power、prediction、short-action 或 boundary 数值。当前只有实现、独立准入、动态身份预检和运行状态证据，没有新的性能或效率结论。仅在 job 终态后核验八 pass、逐 pass prediction/evaluator/hash、cost rows、raw power coverage、pass-level p50/p95/throughput/memory/gross energy、短动作/边界、`profile.json` 与成功或受控失败 `terminal_receipt.json`。完整终态证据随后必须返回 fresh ZoomToken Project Pro 独立裁决，在其回复前不追加实验。

# ZoomToken BPNS-R1 v004 解耦诊断与成本闭环启动回执

- 日期：2026-08-29
- 状态：`FORMAL_REPLAY_V004_QUEUED`
- 唯一任务：`ZOOMTOKEN-BPNS-R1-DECOUPLED-DIAGNOSTIC-AND-COST-CLOSURE-v004`
- 科学边界：本轮只检验已冻结 K100/R1 单种子候选的真实同硬件成本与既有定位诊断；不训练、不改模型、checkpoint、数据、测量定义、估计器、阈值或八次顺序。

## 固定候选与独立准入

- base：`8a59d655005b9030d8ea5dc17ee2620844cb587b`
- clean/pushed candidate：`a4694019fd4cbbdc74885e160163e23d947dc05f`
- branch：`codex/zoomtoken-bpns-r1-decoupled-cost-v004`
- 修改面仅为 `tools/bata/profile_zoomtoken_bpns_r1_cost.py`、`tests/test_zoomtoken_bpns_r1_cost.py`、`scripts/run_zoomtoken_bpns_r1_cost_n16r4.sh`
- 本地 focused verification：`26 passed, 1 skipped`；跳过项是 Windows Torch DLL 限制下的真实 evaluator factory 测试
- N16R4 focused verification：`27 passed`，包含上述真实 factory 测试
- fresh independent Critic：`PASS`
- fresh result-blind Evaluator：`PRE_RUN_READY`
- 远端 source：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_cost_src_a4694019_v004`；HEAD、remote-tracking ref 与候选 SHA 一致，worktree clean

## 结果盲动态预检

- 首次预检 job `1260082`：`FAILED 1:0`。原因是部署只生成 `FETCH_HEAD`、未生成身份门要求的 remote-tracking ref；未开始科学计算、未产生结果。补齐同一已推送 GitHub 分支的远端引用后，候选代码没有变化。
- 有效预检 job：`1260092`（`zt-bpns-v004-precheck2-a4694019`）
- 终态：`COMPLETED 0:0`，节点 `g0015`
- 回执：`PRECHECK_READY`
- 已核验：exact pushed revision、411 canonical MP4、211 validation videos/792 ordered items、两份 epoch-59 EMA checkpoint、config/data/evaluator/NMS identity、production evaluator factory 的全 1 synthetic known-answer、raw writer known-answer、八次冻结顺序与新 result root
- 预检明确 `reads_validation_metrics=false`、`trains_or_resumes=false`

## 唯一正式执行

- Slurm job：`1260095`
- JobName：`zt-bpns-v004-a4694019`
- 提交次数：`1`
- 提交时间：`2026-08-29 00:37:10 +08:00`
- 提交时状态：`PENDING (Priority)`；开始时间和节点尚未分配
- 资源：`gpu` partition，1 GPU、5 CPU、8 小时；不固定物理 GPU，不覆盖 Slurm 的 `CUDA_VISIBLE_DEVICES`
- result root：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_r1_decoupled_v004_a4694019_seed42_20260829`
- log root：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_v004_logs_a4694019_20260829`
- 顺序：`K100,R1,R1,K100,R1,K100,K100,R1`；每臂四个完整 pass
- 每 pass 原子保存 prediction/evaluator、冻结 SHA 核验、raw cost、power coverage 与 pass receipt；八 pass 完成后才运行 short-action/boundary diagnostics
- 主判据：`median4(R1 pass p50) / median4(K100 pass p50) <= 0.95` 且 `median4(R1 complete-pass joules) / median4(K100 complete-pass joules) <= 0.95`

## 静默终态等待与解释边界

FastCtx background job `j-sjvtib` 以 300 秒间隔只读取 Slurm 终态，不读取或输出运行中性能。运行期间不读取、汇总或解释 live/partial accuracy、cost、power、prediction、short-action 或 boundary 数值。当前只有实现、独立准入、结果盲预检与正式提交事实，没有新的性能或效率结论。无论正式作业成功、受控失败或 Slurm 硬失败，终态证据都必须完整摄取，随后向 exact ZoomToken Project 发起恰好一次全新 post-result Pro 独立裁决；裁决前不追加实验。

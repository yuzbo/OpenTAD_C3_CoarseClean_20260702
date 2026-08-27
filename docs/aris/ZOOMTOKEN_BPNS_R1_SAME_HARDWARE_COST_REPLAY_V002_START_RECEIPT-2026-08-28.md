# ZoomToken BPNS-R1 同硬件成本回放 v002 启动回执

- 日期：2026-08-28
- 状态：`FORMAL_REPLAY_V002_RUNNING`
- 科学目的：不新增训练；用既有 K100/R1 epoch-59 EMA 在同一张 Slurm GPU 上完成完整 ABBA+BAAB 回放，判断 36% 原生空间输入减少是否转化为真实 decode→Soft-NMS 延迟、显存和 gross energy 改善，并复核短动作与边界质量。

## 固定候选与准入

- 基线：`b7357817d81127ab2d713b5471d008ea893efd35`。
- clean/pushed candidate：`e9323448f6cd78b99bb3de53fd9ffb55f3676d65`，分支 `codex/zoomtoken-bpns-r1-parity-v002`。
- 远端 source：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_cost_src_e9323448`；HEAD、remote-tracking ref 与候选完整 SHA 一致，worktree clean。
- 变更仅限 `tools/bata/profile_zoomtoken_bpns_r1_cost.py` 与 `tests/test_zoomtoken_bpns_r1_cost.py`。模型、forward、训练配置、数据、checkpoint、population、顺序、warmup、evaluator/NMS、成本仪器和硬件合同未修改。
- parity：六项指标必须完整且 finite；以未舍入 percentage points 对冻结 reported-2dp reference 做 inclusive `0.05 pp` 比较；HALF_UP 两位展示与准入分离。
- 验证：Python compile、Shell syntax、`git diff --check` 通过；focused pytest `13 passed`。
- fresh independent Critic：`PASS`。
- fresh result-blind Evaluator：`PRE_RUN_READY`；已核验 411 MP4/0 断链、211 validation 视频/792 ordered items、两份 epoch-59 EMA、新 result/log roots 与 JobName、旧 job `1257281` 封存及 `sbatch --test-only`。

## 正式执行

- Slurm Job ID：`1258299`
- JobName：`zt-bpns-r1-pv2-e9323448`
- 首次提交次数：`1`
- 提交/开始：`2026-08-28 00:20:23/00:20:24 +08:00`
- 当前节点：`g0048`
- 资源：`gpu` partition，1 GPU、5 CPU、8 小时；不固定物理 GPU，不覆盖 Slurm 的 `CUDA_VISIBLE_DEVICES`。
- source root：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_cost_src_e9323448`
- result root：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_r1_cost_parity_e9323448_seed42_20260828`
- log root：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_parity_logs_e9323448_20260828`
- runner：`scripts/run_zoomtoken_bpns_r1_cost_n16r4.sh`，`PRECHECK_ONLY=0`。
- 顺序：`K100,R1,R1,K100,R1,K100,K100,R1`；每臂四个完整 pass，每 pass warmup 50 windows。
- 冻结输入：K100 job `1248835` 与 R1 job `1249099` 的 epoch-59 `state_dict_ema`；canonical THUMOS14 411/211/792；既有 evaluator、Soft-NMS、成本与边界诊断。

## 证据边界

当前只有实现、独立准入和运行状态证据，没有新的性能或成本结论。运行期间不解释 live/partial 数值。只有 job 终态且八个 pass、`profile.json`、`terminal_receipt.json`、预测、evaluator vectors、功耗轨迹、显存、延迟、短动作和边界产物完整后才应用冻结门槛；随后立即进入新的 ZoomToken Project Pro 复盘，在其裁决前不追加实验。

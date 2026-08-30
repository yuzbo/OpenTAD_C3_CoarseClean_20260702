# ZoomToken-RACER24 Iteration-0 微基准启动回执

## 边界

- 用户于 `2026-08-31T01:23:36+08:00` 确认将手工转交的 RACER24 裁决作为 Iteration-0 实现与冻结微基准的执行权威。
- 该材料仍属于 `USER_MANUAL_TRANSFER_NOT_BROWSER_AUDITED`；本回执不补造 Project conversation、nonce、附件或 Oracle provenance。
- 本轮只执行 block/model-path 微基准，不运行数据集、训练、正式 full-stack cost、FARM24 或 PairLatent32。

## 冻结候选与审查

- base：`2d945e64bdccd09ae2e2916524562e3f388c5a2a`
- branch：`codex/zoomtoken-racer24-v001`
- candidate：`5ebaa74f611bb3a43c3042700a78b92a9e5e74fb`
- Builder commits：`e767c7ac`、`9b7ea8a3`、`243082cf`、`5ebaa74f`
- final Critic：`PASS`
- N16R4 focused checks：`16 passed in 48.99s`
- result-blind Evaluator：`PRE_RUN_READY`
- GitHub HTTPS push 在本机因 `SSL_ERROR_SYSCALL` 失败；候选通过增量 Git bundle 部署到 N16R4 clean detached checkout，未声称已推送。
- remote source：`/data/run01/sczc063/yuzibo/projects/zoomtoken_racer24_src_5ebaa74f`

## 冻结协议

- 形状：`B=1`、8 tubelets、每 tubelet 64 token、每 tubelet selected 24、总 `Q=192`、`K/V=512`。
- matched control 与 candidate 使用相同权重、输入、dtype、device、warmup 与同步。
- warmup `50`，timed repetitions `200`；记录 p50、p95、peak allocated 与 peak reserved。
- 通过门：p50 speedup `>=1.08x`，allocated 与 reserved memory ratio 均 `<=1.05`。
- 任一真实门失败即停止 Iteration-0；不得调 K、改 blocks、训练或开启后继候选。

## 作业

首次 scheduler job `1262067` 在 `2026-08-31T02:52:40+08:00` 以 `FAILED 127:0`、elapsed `00:00:01` 终止。Slurm `--wrap` 被 `/bin/sh` 解释，`source`、`module` 与 `python` 均不可用；stdout、profile、terminal receipt 和模型/CUDA执行均为空。独立 Evaluator 将其定性为 `PRE_EXECUTION_OPERATIONAL_BLOCKER`，科学尝试序号未消耗，并准入一次仅把 wrapper 改为显式 `/bin/bash -lc` 的 replacement。

- 唯一 scientific-attempt job：`1262068`
- JobName：`zt-racer24-i0-rpl1-5ebaa74f`
- 资源：`gpu` partition，1 GPU，4 CPU，10 分钟
- result root：`/data/run01/sczc063/yuzibo/results/zoomtoken_racer24_i0_5ebaa74f_20260831`
- log root：`/data/run01/sczc063/yuzibo/logs/zoomtoken_racer24_i0_5ebaa74f_20260831`
- terminal waiter：FastCtx `j-add984`，每 60 秒只输出终态
- replacement、retry、resume、第二 seed 或训练：禁止

当前没有性能结果；运行中不读取或解释 partial timing/memory。

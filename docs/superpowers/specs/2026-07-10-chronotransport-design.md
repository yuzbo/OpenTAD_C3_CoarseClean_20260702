# ChronoTransport 完整模型设计

日期：2026-07-10

## 1. 目标与边界

在 AdaTAD 的 VideoMAE-S backbone 内实现面向 TAD 的动态特征刷新模型。模型保持 768 点 dense physical-time 输出，不做 pre-backbone 删帧；对每个时间 chunk 和可控层组选择 `RECOMPUTE`、`TRANSPORT` 或 `HOLD`，以降低真实重计算成本，并保护高 tIoU 与短动作定位。

现有 C3/DUCA 路线不删除、不重写，作为固定 384、边界导向采样和完整成本对照。第一版不修改 decoder，不宣称节省 decode 成本。

## 2. 可证伪假设

主假设：VideoMAE 的 time × layer 重算价值高度不均匀，且可由 deploy-visible 的 cache age、低成本输入变化和 feature drift 预测；在相同实测延迟下，风险调度优于 periodic refresh、motion threshold、HOLD-only 和现有 DUCA。

停止条件：

1. 完整 p50/p95 延迟节省低于 15%；
2. periodic refresh 在三种子 95% CI 内与学习调度持平；
3. mAP@0.7 或最短 duration quartile 下降超过 1.5 absolute points；
4. scheduler、transport 与 cache movement 吃掉重算收益的 40% 以上；
5. predicted risk 与真实 counterfactual detector regret 相关性接近零。

## 3. 架构

### 3.1 执行粒度

输入 768 帧沿用 AdaTAD 的 16-frame VideoMAE chunk，形成 48 个时间块。12 个 ViT block 分成可配置层组，默认 `[0:4]`、`[4:8]`、`[8:12]`。调度单位为 `chunk × layer_group`，避免第一版产生 48×12 的高开销细粒度控制。

### 3.2 三种动作

- `RECOMPUTE`：使用当前 chunk 的真实 token 执行该层组，并刷新缓存；
- `TRANSPORT`：用低成本 temporal transport adapter 将最近缓存状态变换到当前 chunk，再做残差修正；
- `HOLD`：直接复用最近缓存状态，仅允许在风险上界和最大 cache age 同时满足时使用。

每个 chunk 都产生 feature，post-processing 仍输出原始 768 时间轴。ActionFormer projection、head、GT 与后处理保持不变。

### 3.3 组件边界

新增 `opentad/models/chronotransport/`：

- `actions.py`：动作枚举、schedule schema 与严格验证；
- `cache.py`：按层组管理 feature、age、来源时间与 detach 策略；
- `transport.py`：temporal feature transport 和 residual correction；
- `risk.py`：schedule-conditioned quantile regret predictor；
- `scheduler.py`：在有限 schedule library 中选择满足风险阈值的最低成本方案；
- `losses.py`：transport consistency 与 quantile pinball loss；
- `profiler.py`：阶段耗时、显存与动作统计。

`VisionTransformerAdapter` 增加一个默认关闭的 ChronoTransport 执行入口。关闭时必须与原 dense forward 数值兼容；开启时由独立 runtime wrapper 控制层组，不把调度逻辑塞入通用 Block。

## 4. 数据流

### 4.1 Dense reference

训练时对同一窗口执行 dense reference，得到各层组 reference feature 和 detector loss。Reference 仅用于训练 target，不进入 val/test 决策。

### 4.2 Counterfactual schedule

从有限 schedule library 采样：dense、periodic-2/4/8、hold-only、transport-only、motion threshold、time-only、layer-only 和 joint time×layer。执行 schedule 后计算相对 dense reference 的 detector regret。

第一阶段使用 task loss difference 与 feature consistency；具备稳定 detector prediction 对齐后再加入 endpoint/high-IoU/short-action tail 分量。禁止使用 test GT、test teacher 或 raw-prediction cache 参与决策。

### 4.3 推理

每个窗口重置 cache，首 chunk 强制 `RECOMPUTE`。后续 chunk 从 deploy-visible signal 预测各候选 schedule 的 upper-risk 与 measured cost，选择最便宜且 `upper_risk <= epsilon` 的方案。无可行方案、非有限数值、cache 失效或 OOD guard 触发时 fail closed 到 `RECOMPUTE`。

## 5. 训练阶段

### 阶段 A：Counterfactual replay

冻结 VideoMAE 与 detector，在 dense feature replay 上验证 HOLD/TRANSPORT/periodic 的 regret map。产出可重复的 counterfactual ledger，但该 ledger只作训练/分析数据，不是部署决策缓存。

### 阶段 B：Transport 与 risk 学习

冻结主 backbone，训练 transport adapter、risk predictor 和 scheduler。目标为：

`L = L_task + lambda_transport * L_transport + lambda_risk * L_pinball`

### 阶段 C：AdaTAD adapter 联调

保持 VideoMAE 主干冻结，联合微调现有 AdaTAD adapters、transport adapter 和 risk predictor。三种子运行，报告 mean、std 和 bootstrap/seed CI。

## 6. 对照与消融

必须包含：dense 768、exact-uniform 384、当前 DUCA fixed384、periodic-2/4/8、motion threshold、HOLD-only、TRANSPORT-only、learned risk without calibration、learned risk with calibration，以及 oracle schedule（仅诊断）。

消融分别移除 transport correction、cache age、feature drift、risk calibration、layer action 和 time action。每次只改变一个因素。

## 7. 成本与指标

分项测量 data/decode、preprocess、H2D、innovation、scheduler、recompute、transport、cache movement、neck/head 和 postprocess。报告 p50/p95 latency、throughput、peak GPU memory、CPU memory、GPU utilization 与可用时的 energy。

任务指标报告 THUMOS14 mAP@0.3:0.7、mAP@0.7、duration quartile、start/end boundary error 和 short-action tail。第一部署目标是 THUMOS14；跨数据集扩展在机制通过 kill gate 后进行。

## 8. 配置与部署

新增最小 C3 命名配置、validator、focused tests 和 `scripts/run_chronotransport_adatad_gpu1.sh`。启动器默认 `PRECHECK_ONLY=1`，要求物理 GPU1；正式运行必须处于 Slurm allocation/step 或显式授权保护分配。输出仅写入 `/data/run01/sczc063/yuzibo` 范围，不同步 checkpoint、数据集或日志回仓库。

部署顺序：本地静态测试 → N16R4 validator/precheck → GPU1 smoke replay → GPU1 三种子 Stage A/B → kill-gate 审核 → Stage C。未通过前一 gate 不自动进入下一阶段。

## 9. 测试策略

严格 TDD：每项生产行为先写失败测试并确认预期失败，再写最小实现。

重点覆盖：动作 schema、首块强制重算、cache age、层组隔离、transport shape、HOLD 不变性、非法 schedule fail closed、dense-off 数值兼容、无 GT/teacher inference、风险阈值选择、deterministic seed、GPU1 与 Slurm 启动保护、validator 和成本字段完整性。

## 10. 完成定义

“实现完成”要求模型组件、配置、训练/评测工具、validator、启动器和 focused tests 均存在且通过；“部署完成”要求远端 precheck 通过并在 GPU1 的合法分配中启动至少一个可观察的 smoke/Stage-A 作业。科学主张只有在三种子结果通过预注册 kill criteria 后才能成立。

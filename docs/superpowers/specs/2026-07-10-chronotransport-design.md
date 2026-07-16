# ChronoTransport 完整模型设计规格（复核校正版）

日期：2026-07-10

## 1. 目标与任务定义

ChronoTransport 是离线、全窗口的 AdaTAD 条件计算模型。它不删除输入帧，不改变 decoder、ActionFormer head 或 NMS；它在保持现有 detector 768 点输出网格的前提下，减少 VideoMAE heavy attention/MLP 的不均匀重复计算。

Scheduler 可观察完整测试窗口中的部署可见输入，但不得访问 GT、teacher、dense heavy reference、detector raw-prediction cache 或 counterfactual ledger。窗口内按时间维护 cache 是计算实现，不构成 online/causal TAD 声明；v1 每窗口重置 cache，不跨 sliding window 复用。

旧 C3/DUCA 只作为外部 baseline、协议检查和完整成本对照，不向 ChronoTransport 主路径提供信号。

## 2. 外部与内部时间几何

当前输入为 B×768 帧，并重排为 (B×48)×3×16×H×W。VideoMAE tubelet_size=2，每个 16-frame clip 产生 8 个 temporal tubelet，因此内部 backbone 时间格为 48 clips × 8 tubelets = 384 internal temporal points。

现有 post-processing 将 384 点插值回 detector 的 768 点网格。v1 调度键为 [B,48,G]：48 个 clip × G 个 layer group。不能声称 backbone 产生 768 个独立 dense feature。

默认 G=3，层组为 [0:4] / [4:8] / [8:12]；层组必须连续、无重叠并覆盖全部 12 层。更细 tubelet routing 仅作为后续消融，不属于 v1 主合同。

## 3. v1 的真实动态计算边界

始终 dense 执行：

- decode、preprocess 与 patch embedding；
- deploy-visible innovation signal；
- 每个原 block 后的 AdaTAD temporal adapter；
- neck、head 与 postprocess。

动态执行：

- VideoMAE heavy attention/MLP；
- ChronoTransport transport correction；
- scheduler 与 cache movement。

AdaTAD adapter 会将 48×8 tubelet 拼为长度 384 的序列并做 dense temporal convolution，因此不能随某个 clip row 一起跳过。mixed schedule 中，RECOMPUTE row 的 attention/MLP 真实执行，但 adapter 会看到 RECOMPUTE/TRANSPORT/HOLD 混合上下文；只有 forced-dense 原 block loop 是 dense-reference 数值等价锚点。

## 4. 三种动作与缓存语义

- RECOMPUTE：当前 clip 的真实 group input 执行该组 heavy attention/MLP；每个 block 后仍执行 dense AdaTAD adapter；anchor=latest=current，age=0。
- TRANSPORT：跳过该组 heavy attention/MLP，从 latest cache 与当前 deploy-visible group input 生成 correction；更新 latest，保留 anchor，age+=1。
- HOLD：输出逐位等于 latest；不改变 anchor/latest，age+=1。

anchor 是最近真实 RECOMPUTE 状态，latest 是最近 RECOMPUTE 或 TRANSPORT 状态。TRANSPORT 必须从 latest 链式递推，不能每次重新从 anchor 生成。

首 clip、cache invalid、age 超限、非法 action、非有限 signal/transport/output、OOD、无 measured cost、无可行 calibrated-risk candidate、risk/checkpoint 未 ready 时 fail closed 到 RECOMPUTE。

## 5. 部署可见风险信号

只允许当前 patch/group input 的能量和低成本变化、previous/latest cache 的代理 drift、cache age、chunk position、layer-group identity、候选 schedule action，以及 OOD、有限值和 cache-validity 标志。

明确禁止当前 chunk 的 dense heavy reference feature、test GT、test teacher、raw detector prediction 和 Stage-A ledger 查询。任何 observation 都必须单独计时；成本接近完整 backbone 的 observation 不得进入主路径。

## 6. 风险、成本与调度

同一 batch、同一增广和同一 RNG 下执行 dense reference no-grad 与 counterfactual schedule 双前向。单侧 regret target 为 max(L_counterfactual-L_dense,0)。

风险 predictor 使用 schedule-conditioned quantile regression；fit、calibration 与 evaluation split 必须隔离。推理在具备 calibration 和专用 ChronoTransport checkpoint 时，选择满足 upper-risk 阈值的最低实测成本 schedule；否则 dense fail closed。

线性 group cost 只允许 TDD、precheck 或 debug。正式 cost lookup 键至少包含 hardware、precision、batch_size、candidate_schedule、selected_rows_per_group 和 statistic={p50,p95}，因为 GPU gather 延迟受 occupancy、row 数和 schedule 形状影响，并非按层数线性相加。

## 7. 模块边界

新增 opentad/models/chronotransport/，包含 actions.py、cache.py、transport.py、risk.py、scheduler.py、losses.py、profiler.py 和 runtime.py。

VisionTransformerAdapter 只增加默认关闭的 chronotransport runtime 入口。关闭或 forced-dense 时走原 block loop；现有 packed-tubelet route 与 ChronoTransport 显式互斥。

## 8. 实施阶段

### P0：核心执行合同

完成 action/schema、cache、latest-based transport、risk、measured-cost scheduler、fail-closed、forced-dense、mixed gathered heavy rows、dense adapter innovation 与 profiler focused tests。

### P1：生产 backbone 与 Stage-A smoke

接入 VisionTransformerAdapter，增加 Stage-A config、validator 和 GPU1/Slurm launcher。旧 dense checkpoint 只允许 forced baseline；learned mode 必须加载包含 ChronoTransport 参数和校准状态的 checkpoint。

### P2：Detector-level paired replay

新增独立 runner：同 batch/RNG 的 dense no-grad 与 counterfactual 双前向，记录 task、endpoint/high-IoU、short-action regret。持久化 ledger 只含 compact signal、schedule、cost 和 regret label，不保存 raw predictions 或 full-token state，也不参与 inference。

### P3：Transport 与 risk 训练

冻结 VideoMAE、projection/head 参数，但 detector counterfactual branch 对输入保留梯度。训练目标为 L_task_cf + lambda_transport×L_transport + lambda_risk×L_pinball。Scheduler 初期使用有限 library target，不通过不可微 argmin 反传。

### P4：AdaTAD adapter 联调

解冻现有 AdaTAD adapters、transport 和 risk；dense reference branch no-grad；禁止重复计入同一 detector loss。至少三种子。

### P5：真实成本与 kill gate

完整计 decode/data、preprocess、H2D、innovation、scheduler、heavy recompute、transport、cache movement、dense adapter、neck/head 和 postprocess。第一版不宣称 decode saving。

## 9. 基线与停止条件

基线包括 dense 768、exact-uniform 384、DUCA fixed384、periodic-2/4/8+TRANSPORT、periodic-2+HOLD、motion threshold、HOLD-only、TRANSPORT-only、layer-only、joint time×layer、uncalibrated/calibrated risk 和 diagnostic-only oracle。

满足任一条件即停止：

- p50 latency saving <15%；
- periodic baseline 在三 seed 95% CI 内持平；
- mAP@0.7 或 shortest-duration quartile 下降 >1.5 absolute；
- scheduler+transport+cache overhead 超过重算收益 40%；
- calibrated risk 与 counterfactual regret 相关性接近零。

## 10. 仓库与部署合同

允许新增 ChronoTransport package、配置、validator、launcher、focused tests 和方法文档，不删除或改写 C3/DUCA。远端写入仍限定 /data/run01/sczc063/yuzibo；GPU1 launcher 默认 PRECHECK_ONLY=1，正式任务必须位于 Slurm allocation/step 或明确授权的保护分配。

所有 deploy、metric、latency 和 paper claim 默认 false，只有 P5 三种子 kill gate 通过后才允许解锁。

## 11. 完成定义

工程完成：P0–P4 代码、测试、配置、runner、validator 和 launcher 全部落地并通过本地及远端 focused checks。

部署完成：N16R4 validator、真实 ViT integration tests 和 GPU1 Stage-A/P2 smoke 可观察运行。

科学完成：三种子结果通过全部停止条件。未达到时只能报告机制失败或负结果，不能宣称有效、加速或可部署。

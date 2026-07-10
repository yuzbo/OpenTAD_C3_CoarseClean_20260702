# ChronoTransport：规格复核、实施计划与首轮 TDD 状态

- 日期：2026-07-10
- 固定审查基线：`c26b349ee27b6e427fa5cbff8c011778c2684b17`
- 目标后端：THUMOS14 / AdaTAD / VideoMAE-S / ActionFormerHead
- 当前实现级别：核心 runtime 垂直切片 + Stage-A 配置/validator/启动保护
- 尚未完成：detector-level paired counterfactual replay、Stage-B loss 接线、Stage-C 联调、GPU 实测与三种子科学结论

## 1. 裁决

规格**有条件确认**。研究问题是成立且可证伪的：不删除输入帧，而是在已解码的 dense 视频上减少 VideoMAE 重型 attention/MLP 的重复计算，保留原 detector 时间轴，并用 detector regret 风险约束调度。这比把所有成本都转移到一个 pre-backbone scout 更接近“真正减少 backbone 重算”的问题。

但原规格必须采用以下校正后才可实现和审计：

1. **外部 768，内部不是 768 个独立 backbone 点。** 当前配置把 768 帧重排为 48 个 16-frame clip；VideoMAE tubelet size 为 2，因此每 clip 输出 8 个 tubelet，内部时间格是 `48×8=384`，现有 post-processing 再插值为 detector 的 768 点输出。调度键是 `48 chunks × layer groups`。
2. **第一版只门控重型 attention/MLP。** 当前 AdaTAD Adapter 在每个 block 后把 48×8 个 tubelet 重新拼成 384 长序列做 dense temporal convolution。为了不改官方 adapter 位置和 decoder，patch embedding 与 Adapter innovation 保持 dense，并单独计入成本。
3. **TRANSPORT 从 latest cache 递推。** `anchor` 是最近一次真实 RECOMPUTE 的参照；`latest` 是最近一次 RECOMPUTE/TRANSPORT 的可复用状态。TRANSPORT 从 `latest` 生成当前状态，age 仍按距真实 RECOMPUTE 的距离增长。
4. **风险信号必须 deploy-visible。** 允许 patch/group 输入能量、低成本变化、与 latest/anchor 的代理漂移、cache age、schedule action；禁止使用当前 dense heavy feature、test teacher、test GT 或 detector raw-prediction cache。
5. **Stage-A ledger 不保存全量空间 token cache。** 持久化内容仅包括窗口标识、schedule、deploy-visible signal、池化组特征、动作统计、分项成本和 regret label。全 token teacher state 仅在训练 step 内短暂存在。
6. **regret 使用同随机源、单侧目标。** dense 与 counterfactual 前向必须共享窗口、增广和随机状态；主风险 target 为 `max(L_cf-L_dense,0)`。校准 split 与 predictor fit split 分离。
7. **无实测 cost 或无校准 risk checkpoint 时必须 dense fail-closed。** 代理 FLOPs 或理论层数不能解锁 deployment scheduler。
8. **现有 packed-tubelet route 与 ChronoTransport 互斥。** 两者都接管 transformer block 执行，不能叠加后再归因。
9. **每窗口重置 cache。** v1 不声称跨 sliding-window 或流式视频复用。
10. **旧 dense checkpoint 只用于 Stage-A 强制 schedule。** learned mode 必须加载包含 ChronoTransport 参数和校准状态的专用 checkpoint。

## 2. v1 执行语义

### 2.1 几何

```text
input: B × 768 frames
  -> pre-processing: (B×48) × 3 × 16 × H × W
  -> patch embed: (B×48) × (8·h·w) × C
  -> ChronoTransport: action[B, 48, 3]
  -> VideoMAE output: B × C × 384 × h × w (conceptually)
  -> existing pooling/rearrange/interpolate
  -> detector feature grid: B × C × 768
```

默认层组：`[0:4] / [4:8] / [8:12]`。

### 2.2 动作

- `RECOMPUTE`：当前 clip 的真实 group input 执行该组全部 heavy subpath；在每个原 block 位置继续执行 dense AdaTAD adapter；刷新 anchor/latest，age=0。注意，在 mixed schedule 下，attention/MLP 的该 row 是真实重算，但其 AdaTAD temporal adapter 会看到由 RECOMPUTE/TRANSPORT/HOLD 混合组成的 dense 上下文，因此不能把该 row 称为完整 dense-reference 等价。
- `TRANSPORT`：跳过该组的 heavy subpath，以 latest cache 和当前 deploy-visible group input 经低秩 transport correction 得到状态；latest 更新，anchor 不变，age+1。
- `HOLD`：逐位复用 latest；latest/anchor 不变，age+1。

因为原 adapter 位于每个 block 后，group action 会扩展到组内各 block；每个 block 内部维护滚动状态，三个动作在整个组内一致。transport 模块按 layer group 共享，这是 v1 的低开销选择，后续可消融为 per-block transport。

### 2.3 fail-closed

以下任一条件触发当前 cell 或整个窗口 RECOMPUTE：首 chunk、cache invalid、age 超限、非法 action、transport 非有限、最终输出非有限、signal OOD、无候选满足 calibrated risk、无 measured cost、risk 未 ready、learned mode 未加载 ChronoTransport checkpoint。

## 3. 代码面

```text
opentad/models/chronotransport/
  actions.py       action enum、layer group、schedule schema
  cache.py         anchor/latest/age/source-time 与 detach contract
  transport.py     latest-cache conditioned low-rank correction
  risk.py          schedule-conditioned quantile predictor + split conformal offset
  scheduler.py     finite library、measured cost、risk-constrained selection、motion baseline
  losses.py        one-sided regret、pinball、transport consistency
  profiler.py      required stage latency/action/memory schema
  runtime.py       gathered heavy execution、dense adapter innovation、fail-closed
```

`VisionTransformerAdapter` 只增加默认关闭的 runtime 入口，不把调度逻辑塞入通用 `Block`。关闭或 forced-dense 时走原 block loop；packed route 与 ChronoTransport 有显式互斥检查。

## 4. 实施阶段与 gate

### P0：契约与核心 TDD（本轮已完成）

- action/schema、首块强制重算、层组完整覆盖；
- cache anchor/latest/age 与 detach；
- latest-based transport、zero-init=HOLD；
- risk quantile、candidate age、conformal offset；
- measured-cost scheduler、OOD/nonfinite/no-feasible fail-closed；
- forced dense 数值等价；
- mixed schedule 实际减少 heavy rows；
- dense adapter 接收真实 `h,w`；
- profiler 字段完整；
- 无 GT/teacher inference API。

Gate：focused tests 全绿、`compileall` 通过、默认 claim flags 全 false。

### P1：生产 backbone 接入与 Stage-A smoke（补丁已提供，GPU 未执行）

- `VisionTransformerAdapter(chronotransport=...)`；
- legacy dense checkpoint 仅用于 forced baseline；
- Stage-A 配置、validator、GPU1/Slurm launcher；
- forced dense / periodic / hold / transport / layer / joint schedule 单作业 smoke；
- p50/p95 分项 profiler schema落盘。

Gate：dense-off 与 forced-dense 对原 checkpoint 输出误差为 0（或 bitwise 相等）；至少一个合法 GPU1 smoke 可观察；packed route 互斥测试通过。

### P2：Stage-A paired counterfactual replay（未实现）

新增独立 runner，而不是复用 raw-prediction shortcut：

1. 同一 batch 固定 RNG，先 dense no-grad reference；
2. 采样/枚举 schedule，执行 counterfactual forward；
3. 计算 detector loss、预测分布、边界和短动作分层 regret；
4. ledger 只写 compact label/signal/cost，不写可部署 raw prediction cache；
5. oracle schedule 只在 train/diagnostic split 生成。

Gate：相同 dense schedule 的 regret≈0；重复运行 ledger hash 一致；val/test inference 决策不读取 ledger。

### P3：Stage-B transport + risk（未实现）

- 冻结 VideoMAE 原参数、projection/head；
- detector 冻结参数但保留 input gradient；
- transport 训练用 group reference feature；
- risk target 为 one-sided detector regret；
- predictor fit / calibration / evaluation 三段隔离；
- scheduler 初期只用离线 library target，不通过不可微 argmin 反传。

总损失：

```text
L = L_task_cf
  + λ_transport · L_transport
  + λ_risk · L_pinball
```

Gate：transport 比 HOLD-only 显著降低 feature/detector regret；risk–regret rank correlation 明显为正；coverage 在 calibration held-out 上达到目标。

### P4：Stage-C AdaTAD adapter 联调（未实现）

- 解冻现有 AdaTAD adapters + transport + risk；
- dense backbone 原权重继续冻结；
- paired reference 的 dense branch no-grad；
- 防止同一 detector loss 被重复计入；
- 三 seed，报告 seed CI 与 bootstrap sample CI。

Gate：通过预注册停止条件后才解锁 `metric/latency/paper_claim_allowed`。

### P5：真实成本与科学 kill gate（未执行）

完整计费：decode/data、preprocess、H2D、patch innovation、scheduler、heavy recompute、transport、cache movement、AdaTAD adapter、neck/head、postprocess。第一版不宣称 decode saving。正式 scheduler 不能只使用“层数×单 cell 成本”的线性估计；GPU gather 的成本随选中 row 数、batch occupancy 和 schedule 形状非线性变化，必须优先使用按候选 schedule、batch size 与 p50/p95 实测得到的 cost lookup，线性 group cost 只允许 precheck/debug。

硬停止：

- p50 latency saving <15%；
- periodic baseline 在三 seed 95% CI 内持平；
- mAP@0.7 或 shortest-duration quartile 下降 >1.5 absolute；
- scheduler+transport+cache overhead >重算收益的40%；
- calibrated risk 与 counterfactual regret 相关性接近0。

## 5. 对照矩阵

- dense 768；
- exact-uniform 384；
- DUCA fixed384；
- periodic-2/4/8 + TRANSPORT；
- periodic-2 + HOLD；
- motion threshold；
- HOLD-only；
- TRANSPORT-only；
- layer-only early/late recompute；
- joint progressive time×layer；
- learned risk uncalibrated；
- learned risk calibrated；
- oracle schedule（train/diagnostic only）。

每次消融只改变一个因素：transport correction、cache age、proxy drift、calibration、time action、layer action、adapter dense innovation。

## 6. 当前证据边界

本轮只能证明核心执行契约在 CPU/fake-block focused tests 上成立，不能证明：

- AdaTAD GPU latency 已下降；
- transport 已学到 detector-useful correction；
- calibrated risk 能保护 mAP@0.7/短动作；
- Stage-B/C 已闭环；
- 三种子 kill gate 已通过。

因此所有 deploy/metric/latency/paper claim 默认保持 `False`。

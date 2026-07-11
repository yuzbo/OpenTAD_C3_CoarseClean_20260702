# Research Wiki Query Pack

更新时间：2026-07-11。长度必须保持在 8000 字符以内。

## 当前方向

唯一主线是 `PhysTime-AdaTAD 1.0`：THUMOS14 raw RGB，逻辑 768 时间位置，用相同、确定性、无学习、无 GT 的策略选 K=384，`DecordDecode` 和 VideoMAE-S 只消费选中帧。在完全相同 backbone、checkpoint、增强、schedule、seed、NMS 和 selected indices 下比较：selected-axis ActionFormer、physical-grid ActionFormer、`PhysTimeMeasureProjection + PhysTimeHead`。长期目标是一个独立离线 TAD detector，输入任意不规则观测及真实时间戳/支持区间，直接在秒坐标上分类和定位。

当前事实：PhysTime-TAD 2.0 的 feature-geometry 核心代码已实现。PhysTime-AdaTAD 1.0 已完成 raw-video transform、三份 matched K384 配置、same-index validator、one-step 梯度证明、真实 gate 工具及 gate-dependent 三头启动器。四次 gate 均在模型构建前关闭具体基础设施/确定性缺口，依赖训练均未启动；最新逐 transform 诊断 `1158614` 已证明三头从 decode 到 ColorJitter/FormatShape 的像素 hash 一致。仍需新 commit 通过完整 detector gate，不得声称已有 mAP。

## Top gaps

1. `gap:G9`：raw-video 集成代码已测试，但真实 THUMOS CUDA gate 尚未通过。
2. `gap:G4`：三头 matched K384 合同已建立，full-run 结果尚缺。
3. `gap:G3`：需区分 mTAN、TE-TAD、FrameDrop/TRC、LiquidTAD；continuous time 本身不新。
4. `gap:G5`：缺 mAP@0.6/0.7、boundary error、短动作证据。
5. `gap:G7`：缺 raw decode 到 NMS 的全栈成本账本。

## 失败/降级路线，禁止遗忘

- PAction 是强 Stage1 baseline，不是最终 detector-aware 方法。
- GAS-VT 非真正 sequential，存在 budget-conditioned train/apply shift；coverage、CVaR hole 与 hard repair 可能对 detector boundary utility 有害。
- lattice/move/radius 从 uniform scaffold 局部替换，只是几何诊断；聚集偏移可能由二分类标签过粗、probe 时延和 hard/soft 不同构造成。
- DUCA 完成了 no-leak、original-time、official head gradient、optimizer、exact-K/max-gap 等大量工程修复，但核心仍接近复杂 score+top-k+scaffold，GT boundary proxy 不是 detector utility，旧 full runs 不支持最终主张。
- MUST 的 expected/hard/unique/padded/backbone K 不统一，padded cap 不是真 variable compute；动态预算降级 appendix。
- X3D 和 SlowFast Fast 是 frozen prior appendix，dense inference 过慢且有 Kinetics prior/类别重叠风险，不能作为低成本主 probe。
- ChronoTransport/DCRT 虽被 Pro 推荐，但用户否决为当前主线：接近 MoD/feature reuse，层级策略僵硬、系统工程与归因风险高。
- ChronoTransport 不是纯 idea：本地 `codex/c3-coarse-clean-20260702` 已在 `92029ea` 完成正式单种子 Stage-B 闭环，但 P3 science gate 因 risk-regret 负相关、risk target 尺度错配和 feature transport 优势不稳定而失败；origin 落后 15 commits，P5 未解锁，所以状态是“工程闭环存在、科学 gate 失败、当前暂停”。
- CoDeR 依赖 codec/硬件；ACTAL 是 streaming 新任务；均偏离离线 TAD。
- PhysTime-TAL 1.0 的 normalized time、support width、固定 M、hazard 和双视图一致性定义不严，被 PhysTime-TAD 2.0 取代。
- I3D feature-token PhysTime jobs 已取消；只保留算子测试，不是 raw-video 论文证据。

## 活跃机制链

不规则 raw frames -> 原始帧号/FPS 生成秒时间戳与不扩张 support cells -> overlap mass 定义 query evidence -> 每层直接从原始不规则观测投影到 global seconds query grid -> PhysTimeHead 在秒上 assignment、回归、endpoint、NMS -> 可按 `round(t*fps)` 导出原视频帧号。

## 必须遵守

- GT 和预测不能映射到 selected-rank；原视频帧号导出允许。
- sparse gap 不能被 Voronoi/support expansion 填满。
- primary comparison 不加 learned selector、actionness、teacher、ledger、dynamic K 或 paired consistency。
- 所有头逐样本 selected-index checksum 一致。
- smoke、one-step、full mAP、claim evidence 分级，不可混淆。
- 实验数字只写 `docs/evaluation/results.md` 或正式 artifact。

## 近邻文献簇

- 不规则时间：mTAN 证明 continuous-time embedding+attention 已存在。
- TAD 实际时间：TE-TAD 已用 actual timeline coordinate 和长度相关 query。
- 缺帧鲁棒性：Temporal Robustness Benchmark 表明退化主要来自 localization，并提出 FrameDrop/TRC。
- 连续动力学 TAD：LiquidTAD 已用 parallel liquid-inspired temporal relaxation，不能宽泛声称首个 continuous-time TAD。
- detector 基础：ActionFormer 和 AdaTAD 是当前 matched backbone/head 参照。

## 开放未知

1. PhysTime 能否同时胜 selected-axis 与 physical-grid baseline？
2. 改善是否集中在 mAP@0.7、短动作和 contiguous gaps？
3. support-integrated operator 是否明显胜 timestamp embedding、interpolation 和 mTAN-like projection？
4. K=384 raw-video 节省在完整 decode/VideoMAE/head latency 中是否真实？
5. 第二数据集和 held-out sampling family 是否复现？

完整历史细节必须读取 `research-wiki/routes/`，不能只凭本 query pack 恢复 DUCA 或 ChronoTransport。

# Research Wiki Query Pack

当前 G1a 部署状态（2026-07-13）：独立 Max 复审已完成两轮，第二轮 4 个 P1 已按测试先行修复；gate/artifact `65 passed`，PhysTime/shared physical-grid `240 passed`。第三轮必须达到零 P0/P1 后才允许创建 clean snapshot 和运行 real gate；目前仍是 `tested`，不是 `experiment_running`，没有新 mAP。完整审查见 `research-wiki/reviews/2026-07-13-phystime-g1a-max-code-review.md`。

更新时间：2026-07-13。长度必须保持在 8000 字符以内。

## 当前方向

长期唯一主线仍是独立离线 physical-time TAD detector。`PhysTime-AdaTAD 1.0` 的 THUMOS14 raw-RGB/K384 三头 full run 已完成并冻结为负基线。当前执行阶段是 `HOLD AND REBUILD`：先建立 native tubelet feature-support provenance，以及 capacity/context/candidate/assignment-matched 的 selected-time 与 physical-time controls；随后才决定是否实现 `idea:sm-ptaf`。独立核验要求分开 `K=384` raw observations、`J=192` native tubelet tokens、基础候选网格 `Q0` 与多尺度总候选 `QΣ`：G1a 使用 `Q0=J=192`、官方六层金字塔 `QΣ=378`，不做 J192→Q0=384 lift；G1b 才给双侧共享 `Q0=384` 中性 lift，最后才允许 mass residual。候选坐标、GT、回归、decode、NMS 与评测始终使用秒，不能回到 selected rank。

当前事实：最终修复 commit 为 `3ac93a1`，诊断锚点为 `d900c7c`。真实 gate、稳定性 gate、三头 full run 与最佳 checkpoint 复算均已完成。性能下降不是 NaN、evaluator、重复坐标换算或 checkpoint 读取错误。诊断与 2026-07-13 Pro 审查共同定位：原生 tubelet 轴被插值后错误绑定 raw-frame supports、检测容量/上下文不公平、raw absolute seconds 主导 query、粗层 attention 有效聚合坍缩、候选/短动作监督不足、assignment 不同构。`SM-PTAF` 只有 `designed` 状态，回复中的代码不是实现。原始数字只见 `docs/evaluation/results.md`。
G1a native-J192 matched control 已实现并通过远端新旧相关回归 `142 passed`：Q0=192、QΣ=378，不做 J192→Q0=384 lift；双臂共享官方 ActionFormer，仅改变统一秒轴。已修复 candidate mask 污染，并加入逐层 padding isolation、真实 test evaluator、完整内容指纹、checkpoint 反序列化与 evaluator 重算。正式 dataset 消费 411 个视频，test 根目录额外 2 个未引用 MP4 被显式登记。gate `1161304` 因旧审计范围失败，`1161353` 因 scalar state byte-view 失败；`1161378` 又在 selected-axis 首个样本因旧逐步回归梯度合同 fail-closed，三轮 pilot 均未启动。旧 artifact 不足以证明 ReLU 根因；v3 gate 现使用正式 batch=2 DataLoader、warmup scheduler、EMA 与生产更新顺序，并记录 assignment/pre-ReLU/梯度/LR/optimizer state 和真实参数 delta，独立 validator 从逐步证据重算。首轮独立审查的 4 个 P1/3 个 P2 已修复，正在复审；证据保持 `tested`。AdaTAD interpolation 只允许在 G1b 作为双臂共享中性 lift，不能计作新增观测。

## Top gaps

1. `gap:G4`：三头 full run 已完成，但不是坐标表示的等容量隔离；必须先构建 capacity/context/candidate/assignment-matched control。
2. `gap:G2`：算子级 support 已有，但 native VideoMAE tubelet feature 与多原子 support 的 provenance 尚未关闭，禁止继续使用 `192 -> 384` 长度相等冒充语义对齐。一个 tubelet 已融合两帧，multi-atom 只能先称 set-valued anchor，不能自动声称 feature measure 可加。
3. `gap:G5`：边界与短动作诊断已完成；修复后的 coordinate-only、mass residual、bounded content 与 assignment 因果消融尚缺。
4. `gap:G3`：需区分 mTAN、TE-TAD、FrameDrop/TRC、LiquidTAD；continuous time 本身不新。
5. `gap:G7`：缺 raw decode 到 NMS 的全栈成本账本；泛化仍只有 THUMOS14 单协议、单种子证据。

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
- PhysTime-AdaTAD 1.0 的首个 full run 是高价值负结果：不要通过继续训练、增加 endpoint 权重或单独调 NMS 掩盖结构混杂；必须先做等容量、同上下文的因果对照。
- 不把 Pro 给出的 SM-PTAF 公式、伪代码和 patch map 误写成已实现；不在 feature provenance、candidate parity 和 assignment parity 前启动新 full train。

## 活跃机制链

不规则 raw frames -> 原始帧号/FPS 生成秒时间戳与不扩张 support cells -> native tubelet token 绑定 multi-atom support provenance -> overlap mass 保底路径与有界 correction -> candidate-matched physical query encoder -> ActionFormer-equivalent assignment/head 在秒上回归、NMS -> 可按 `round(t*fps)` 导出原视频帧号。

## 必须遵守

- GT 和预测不能映射到 selected-rank；原视频帧号导出允许。
- sparse gap 不能被 Voronoi/support expansion 填满。
- primary comparison 不加 learned selector、actionness、teacher、ledger、dynamic K 或 paired consistency。
- 所有头逐样本 selected-index checksum 一致。
- query 坐标与 candidate cardinality 分离：K 只可用于公平匹配候选数量，不能成为物理坐标或 rank stride。
- zero-coverage cell 可以保留为检测候选，但不得被表述为已经观测或插值重建的 feature。
- smoke、one-step、full mAP、claim evidence 分级，不可混淆。
- 实验数字只写 `docs/evaluation/results.md` 或正式 artifact。

## 近邻文献簇

- 不规则时间：mTAN 证明 continuous-time embedding+attention 已存在；RCL 已做连续锚表示。
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
6. 在相同 ActionFormer 上下文与候选密度下，只改变物理时间表示后是否仍有收益？
7. 在 tubelet Conv3d 与 TIA 已发生 selected-rank mixing 的前提下，multi-atom anchor 是否足够；还是必须使用 frame-separable tokenizer/physical-gap-conditioned stem？

完整历史细节必须读取 `research-wiki/routes/`，不能只凭本 query pack 恢复 DUCA 或 ChronoTransport。

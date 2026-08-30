---
type: decision_history
updated: 2026-08-29
---

# 路线演化与选择理由

## 1. C3 粗分类出发点

最初洞察：低成本模型即使不能直接精确选帧，也可能可靠地区分动作/背景并暴露
状态变化。选择“粗分类 → 间接选择”，而不是让小模型直接承担 TAD。

保留原因：动作性提供低成本候选证据。限制：动作内部高分不等于边界有用。

## 2. PAction 与严格 ledger

建立 train-time p_action 监督、strict budget、no-leak ledger 和 AdaTAD 消费链路。
它证明工程路径可行，并成为固定预算安全锚点。

未选为最终方法：多阶段、detector-unaware、硬 gap decoder，创新性与联合优化不足。

## 3. GAS-VT

尝试学习 frame value、边界覆盖、预算与 gap，再生成 value-transport ledger。

降级理由：apply-time 不是忠实 sequential value transport；训练/应用特征存在风险；
多重覆盖损失和 repair 可能推向均匀；PAction 的简单策略反而更强。保留为诊断 baseline。

## 4. Detector-aware teacher 与 TrueTime

为了让选择服务 TAD，引入 train-only dense detector utility、selected-axis 到原时间映射
和 detector-loss gradient proof。

未作为最终三阶段方案：teacher 预提取仍然多阶段；toy gradient 和 wrapper 不足以证明
联合检测收益；selected-axis 的不等物理间隔语义尚未解决。

## 5. Lattice、move25/move50 与 learned radius

为了避免大 gap，用 uniform scaffold/local replacement 和边界附近膨胀保护覆盖。

降级理由：它是工程启发式；repair/膨胀消耗预算并可掩盖分数偏移；move 分析显示聚集
存在但位置仍可能偏离边界。只保留为几何诊断和对照。

## 6. DUCA 插件化

从 ledger 转向 detector 前 forward 内生成选择：coarse probe、selector、hard selected
positions、official detector backend、teacher-free inference。固定 K 是归因锚点，MUST
尝试动态预算。

选择理由：统一模型接口，允许 detector 梯度反馈，避免测试时 ledger/cache。

## 7. X3D / SlowFast frozen prior

尝试 train-free 视频动作先验，验证视频预训练特征是否比图像 MobileNet 更懂动作。

降级理由：密集视频推理过慢，可能吞掉 heavy backbone 节省；Kinetics 预训练还带类别
先验和 overlap 风险。仅作为 frozen-prior diagnostic/upper baseline。

## 8. Transition/boundary-first 修正

move 分析和早期 DUCA 低性能表明 actionness coverage 再次主导。于是 selector 改为读取
`delta_p_action`、绝对变化、不确定性和粗分类隐藏特征，以 transition/boundary/utility
优先，actionness 降为小权重辅助。

这是当前必须保留的设计初心。

## 9. Progressive joint 与 structured bridge

为协调 probe、selector、detector，采用 progressive joint schedule：先稳定 coarse/boundary
目标，再逐步打开 detector gradient；detector loss 始终训练 backend。structured
zero-forward bridge 保持 hard forward，同时向 selector 传递近似梯度。

未闭环问题：梯度非零不等于 hard utility 对齐，必须做 finite-difference one-swap 审计。

## 10. Offline full-window 语义纠正

严厉审查曾把目标误认为 causal online acquisition，提出 prefix invariance/CFPA。用户明确
纠正：项目从未要求流式因果，完整窗口可见是允许的。CFPA 只保留给未来 streaming 版本。

## 11. 70aa069 冻结与 a5e1774 审计

`70aa069` 修复 full-window structured joint training 并启动 fixed-384 full run。
ResearchClaw 审查指出总成本、hard/soft utility 与 selected-axis geometry 是决定性门槛。
结论不是立即停止，也不是继续堆模块，而是冻结候选并裁决。

`a5e1774` 修复完整成本统计和官方 AdaTAD 表述：base/head 配置一致，但 detector 源码、
输入长度、GT 坐标与后映射并非完全官方不变。

## 12. 当前选择

主科学问题保持“任务感知时序去冗余并保护高 tIoU”。当前先完成 DUCA fixed-384 的
决定性证据；MUST、physical-grid、X3D 主方法和更多 loss 暂停。ChronoTransport 与
PhysTime 是并行新假设，必须独立记账，不能用来改写 DUCA 结果。

## 13. ZoomToken 的连续 ROI 因果比较与运行时公平门

当前主张不再把残差校准当成贡献，而是直接检验连续 ROI 是否在自由原生 token 选择
之上带来高 IoU 定位收益。已冻结的后续比较是 F（完整连续 ROI SCNR-Core）、N（关闭
ROI 的匹配对照）、Q（仅共享基础分数的自由 token 对照）和 D（匹配 dense 参考）。

选择理由：F-N 隔离 ROI，F-Q 排除普通自由 token 选择的解释，F-D 检验效率–精度
Pareto；三者共同避免用单一分支诊断代替论文主张。残差分支校准保留为防止选择退化的
内部诊断，不能单独成文。

在 N16R4 上，普通 `gres/gpu` 不能在调度前固定 GPU SKU，因此不能直接作为公平证据。
已接受的 `PRO_RUNTIME_FAIRNESS_AND_CAUSAL_STUDY_DECISION-v001` 选择了可证伪的替代：
在实际分配和容器启动后、模型与数据访问前，自动记录实际 GPU 与软件环境；所有预检与
正式 pass 的运行时类别必须完全一致，否则整套比较停止。该协议尚待静态实现和独立检查，
所以当前没有可用于论文的性能或效率结果。

## 14. CPR 提案：把论文终点定回“减少空间重计算”（2026-08-14）

按人类约束重述论文终点：**端到端离线 TAD 中减少冗余空间重计算并保护高 IoU
定位**；连续 ROI 与 F/N/Q/D 是机制/消融工具，不是论文主问题。三条不等价路线
（均为提案，最终由全新 Project ChatGPT Pro 裁决）：

1. **TAD-aware 动态空间计算路由**（推荐）：逐 tubelet 空间显著度 + 动态 `K_t`，
   重骨干只算被选原生 token；ROI 是可选先验，残差是防塌缩校准。
2. **原始 SCNR exact-B 全修饰子族**（拒绝）：把 base+ROI+residual 作为同一
   headline 的方法；机制捆绑、历史 M2 角色塌缩 `0/0/3,342,336`、过度工程。
3. **更激进的时空联合预算**（拒绝）：内容依赖的窗口级 `B_t` 混淆“选哪些”与
   “算多少”，破坏 exact-B 匹配纪律与运行时公平，且被 query_pack 分阶段规则排除。

路由层 `AUDIT_PASS` / 干净候选 `cd6463df…` 只是静态接线证据；`b157433d…` 是已关闭
的检测器 fixture 负例。回放与完整训练在全新 Pro 裁决加全部 PRE_RUN/运行时/数据/资源
门禁通过前保持阻塞。

## 15. P1 `{DN,U,R,Q}` 包：Pro 已接受 Q-core `REVISE`（2026-08-14）

全新 Project ChatGPT Pro 咨询已返回 `PRO_DYNAMIC_SPATIAL_ROUTING_DECISION-v001
= REVISE`，**接受 Q-core 路线**：论文终点定为“端到端离线 TAD 中减少冗余 VideoMAE
空间重计算并保护高 IoU/边界定位”；保留核心是 `Q`（检测器监督的 base 空间效用、一次
全局唯一原生 token exact-`B` 选择、诱导动态 `K_t`、一次无 padding ragged 重前向、
masked-zero 载体）。连续 ROI 与校准残差降为**事后因果对照**（`G/N/F`），不是 headline。

首个真实视频 P1 屏是 `{DN, U, R, Q}` @ seed 3407（`DO` 官方 dense 复现为锚点）；
仅当 `Q` 同时通过精度/边界与全栈成本门后，`G/N/F` 才打开 ROI/残差因果对照。冻结项：
`B=24576`（384 tubelet × 64）、动态 token 身份/位置、seed 3407 先导 + 3408/3409 仅
准入后、官方 THUMOS14/AdaTAD 匹配 detector/loss/NMS/split/9600 updates/AMP/EMA、
高 IoU 与边界 estimand、selector-inclusive decode-to-NMS p50/p95、memory/energy
普查，以及 Pro 的成败/停止规则（`Q` 必须胜过 uniform+random 且成本 ≤0.85× dense；
F/D Pareto 下限 ≥−0.50 pp 且成本上界 ≤0.85；三种子 F/N/Q 阈值见 packet）。

明确状态：先前 seed-3407 残差居中信号、冷态配对成本屏、F/N/Q/D 静态接线
`AUDIT_PASS`、`cd6463df…` 均只是**诊断或静态证据**；**不存在任何实际 P1 / 真实视频
/ GPU 结果**。完整 P1 包落盘于
`docs/aris/ARIS_P1_DNURQ_PACKET-2026-08-14.md`；回放与完整训练在所有 PRE_RUN /
运行时 / 数据 / 资源门禁通过前保持阻塞。

## 16. P1 G5 评估定义待冻结（2026-08-14）

独立 Evaluator 对 P1 静态候选给出 `PRE_RUN_NOT_READY`：DO/DN/U/R/Q 的路由、预算、
动态 `K_t`、ragged/masked-zero 与无泄漏静态契约可供后续实现，但不构成运行时、性能或
成本证据。Builder 证明现有 deployer/launcher/finalizer 可以在六个既有文件、一个入口内
实现五臂部署、运行时 attestation 和全栈 raw-video 成本闭环；它没有编辑代码。

停止原因是科学定义缺失，而非工程困难：短动作阈值/分层、起止边界误差的匹配与归一化、
high-IoU 错误分解、以及 Q `<=0.85x` 成本门的 dense 对照尚未冻结。它们交由一次受限
Pro 裁决；裁决不得重开 Q-core 路线、B=24576、DO/DN/U/R/Q@3407、条件 G/N/F、官方
THUMOS14/AdaTAD 合法性或远端 P1。裁决后才允许 Builder 依原六文件计划落地，随后由
Critic 和 Evaluator 再次关门。此记录没有新增实验或论文结果。

## 17. P1 G5 语义裁决：继续 Q-core（2026-08-14）

`PRO_G5_FOUR_DEFINITION_DECISION-v001 = CONTINUE` 没有重开路线；它只把 P1
最终器此前不能由实现者决定的四项评估语义冻结。短动作为官方源视频 GT 的
`0 < d <= 5.0` 秒（恰为 5 秒归短），按去重后的 GT 身份统计且仅报告；边界诊断
以同类、分数排序、贪心一对一、含等号 `tIoU >= 0.50` 配对，时长归一化起止误差，
未配对单列且不填补，也仅报告。官方 `mAP@0.7` 仍是高 IoU 门；七个 GT 域和三个
未配对预测域错误桶只解释失败来源。成本门唯一用匹配 native-source dense `DN` 作
分母：Q/DN 的端到端 p50 和平均 gross energy 两个单侧 95% 上界都必须 `<=0.85`；
`DO` 保留为必报官方锚点但不替代分母。

首个可改变路线的证据仍是完整 seed-3407 `DO/DN/U/R/Q` 矩阵：Q 必须同时通过对 U、R
的官方 `mAP@0.7` 门及两个 Q/DN 成本门。有效失败为 `STOP_Q_CORE_P1`，关闭 G/N/F、
种子扩展和完整训练；任何不完整或污染矩阵为 `NO_SURVIVOR_INVALID_P1`，不允许从
幸存单臂推论。随后仅落地既有六文件 G1/G2/G5 计划并经独立 Critic、结果盲 Evaluator
复核。此项是科学定义和协议证据，未新增真实视频、GPU、性能或成本结果。

## 18. 75c8f6e8 F0 准入与 P1 提交（2026-08-15）

`75c8f6e8c2f433c85ed8b8d488f3c867e5652d6b` 的 F0 只通过
`PASS_OFFICIAL_COMPARABLE_PREFLIGHT_ONLY / FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED`：
PL/ST/world2/finalizer `1238554–1238557` 都是 `COMPLETED 0:0`，world-two FP32
DDP KAT 通过，official test 仍关闭。它不是准确率、延迟、能耗、成本或论文证据。

它仅准入一个全新的 seed-3407 P1 `{DO,DN,U,R,Q}` 原子矩阵：runtime `1238573`、
五个 accuracy leaves `1238574–1238578`、八个 cost leaves `1238579–1238586` 和
afterany finalizer `1238587`。P1 继承 Q-core 的 exact `B=24576`、动态 `K_t`、
ragged/no-padding、官方 THUMOS14/AdaTAD 和闭合 official-test 合同；Q/DN 的 p50 和
gross-energy 单侧 95% 上界 `<=0.85` 是成本门，DO 只报告。提交本身不改变路线、主张或
证据等级：只有完整、身份匹配的 finalizer 才能产生 `STOP_Q_CORE_P1` 或后续准入判断。

## 19. 5491 P1 终态：协议准入失败，不改变 Q-core（2026-08-15）

权威终结器将唯一的 `5491c580…` real-video P1 root
`zoomtoken_p1_dnurq_5491c580_seed3407_20260815_0650` 封存为
`INVALID_P1_MATRIX / NO_SURVIVOR_INVALID_P1`：完成 accuracy cells 为 `0/5`、
cost leaves 为 `0/8`，official test、paper claim 与 partial-arm conclusion 均为 false。
DN/U/R/Q 的调度作业虽完成 `0:0`，仍因 `formal artifact changed: config_path` 被最终器
拒绝；DO 在 epoch-59 checkpoint 后的 official dense test 因未发出 GeoRoute window
telemetry 而失败；成本叶则改变冻结的 136-window/40-video population。因此这些是实施/
协议准入缺陷，不是 Q-core efficacy/cost 证据，也不是 `STOP_Q_CORE_P1`。

决定：封存该 root，不读取或推广指标，不从任何单臂推断，不重试、重排、续跑、补臂或
开启 official test。任何后续 distinct epoch 必须有新的明确授权，并先通过针对上述
确定性缺陷的独立修正审查；在此之前路线、Q-core 合同和证据边界保持不变。

## 20. 完整官方矩阵产生首批可比基线（2026-08-16）

用户明确要求以完整官方远端实验取代子集和反复协议讨论。新 epoch 不复用 `5491c580…` 的失效终结器，而是对官方 AdaTAD 完整计算、同源完整计算、均匀选择、随机选择和 Q 动态空间路由执行三种子、60 轮训练及官方 THUMOS14 validation。确定性 evaluator 标注路径和 checkpoint 目录错误修正后，completion array `1239607` 已完成两个同源完整计算对照：seed 3407 Avg-mAP 66.42%、mAP@0.7 45.19%；seed 3408 Avg-mAP 67.14%、mAP@0.7 45.84%。

这两个结果建立了 Q 直接比较所需的公平完整计算分母，但尚不回答 Q 是否有效。随后 seed 3408 的均匀选择与随机选择也完成 official validation：Avg-mAP 60.05%/61.53%，mAP@0.7 40.17%/41.80%，均低于同种子 dense 67.14%/45.84%。这说明朴素稀疏化会损害检测质量；只有 Q 完成后，才能判断任务相关路由是否找回精度。其余方法/种子仍在训练或排队，最终方法判断必须等待五组×三种子的完整准确率和全栈成本结果。ROI 与 residual 继续作为 Q 通过后的因果对照，不提前升级为主故事。

## 21. Q-core 的首个完整正式负结果（2026-08-16）

seed 3408 的 Q 内容动态路由完成同一 60 轮 checkpoint 的 official THUMOS14 validation：Avg-mAP 57.84%，mAP@0.7 36.93%。同种子 matched dense/uniform/random 分别为 67.14/60.05/61.53 Avg-mAP 和 45.84/40.17/41.80 mAP@0.7。Q 因此比 dense 低 9.30 Avg-mAP 点，也比 uniform/random 低 2.21/3.69 点。

这是方法级、种子级负结果：内容分数没有恢复稀疏计算损失。它不允许被工程失败或流程措辞隐藏。其余种子继续运行，用于判断失败是否稳定；在出现相反证据前，ROI/residual 第二阶段不自动开放，下一科学动作应分析 Q 对边界与上下文覆盖的选择错误。

## 22. 三种子完整计算基线稳定；Q 负结果需要选择行为诊断（2026-08-16）

seed 3409 matched dense 完成 official validation：Avg-mAP 65.99%，mAP@0.7 45.02%。因此 matched dense 三种子为 66.42/67.14/65.99，跨种子差异小于 seed-3408 Q 相对 dense 的 9.30 点缺口，Q 负结果不能解释为基线种子异常。

当前 seed-3408 Q 配置关闭了 diagnostic telemetry；现有正式结果证明精度失败，但不能证明失败来自哪些 token、哪些时间窗口或哪类边界。决定保持剩余完整训练，同时准备使用同一 epoch-59 checkpoint 的官方 validation replay 打开只读路由遥测；该诊断不重训、不改变分数，只用于确定下一次机制校正应针对选择覆盖、动态预算分布还是边界信息缺失。

## 23. 诊断复用已训练模型，不再增加方法假设（2026-08-16）

uniform seed3409 的 60.55 Avg-mAP / 40.32 mAP@0.7 与 seed3408 的 60.05/40.17 接近。为解释 Q seed3408 的正式负结果，`27181b0f…` 只增加一个 checkpoint-only telemetry replay：不改变模型、checkpoint、官方 validation 或 mAP，只记录每个 tubelet 的 K、零预算、几何饱和和角色覆盖。远端作业 `1239655` 已提交。其结果将直接决定后续是修正动态 K 分配、空间覆盖，还是终止内容路由；在该证据前不开放 ROI/residual 第二阶段。

## 24. Q 内容路由在第二个独立种子上再次失败（2026-08-16）

seed 3409 的随机选择与 Q 分别取得 61.41/53.81 Avg-mAP、41.95/33.40 mAP@0.7；同种子的均匀选择和完整计算为 60.55/65.99 Avg-mAP。Q 再次低于随机、均匀和完整计算，且比 seed 3408 的 Q 结果更差。这使“内容分数本身能找回被稀疏化丢失的信息”成为跨两个种子的重复负结论。

决定：不以措辞挽救 Q，也不直接跳到 ROI/residual 新故事。先完成 seed 3407 与官方公开配置，并读取已经排队的同 checkpoint 路由遥测，区分动态预算坍缩、空间覆盖不足和边界信息丢失。只有该分析指出可检验的单一根因，才允许形成下一版模型；否则终止当前 Q-core。

## 25. Baseline-first 身份更正（2026-08-17）

全量 Wiki、官方源码和资源审计发现：决策 20–24 中的 `66.42/67.14/65.99` 及其
Q/U/R 对照来自 matched-source full-compute family，虽使用 THUMOS14 validation evaluator，
却不是 clean AdaTAD release `01c58b9f…` 的未经修改官方复现。因此它们不得再称
“official AdaTAD baseline/result”，也不能对 published AdaTAD table `Avg=69.03,
mAP@0.7=48.27` 作质量结论。

决定：优先以 release checkpoint 在 clean official config/evaluator 上复核 published anchor；
再逐项 diff data preprocessing、config inheritance、pretrain、optimizer/schedule、EMA/
checkpoint selection、evaluator/NMS 与 runtime。当前 Q 的同源负向输出保留为 source-family
诊断，既不被删除，也不升级为 final method/paper STOP。完整数据资源事实与未来 checkpoint
规则见 `WIKI_MEMORY_AUDIT-2026-08-17.md`。

## 26. AdaTAD 官方基线改为跨项目唯一、方法准备并行（2026-08-17）

决定将原始 AdaTAD official baseline 的执行权集中于 ZoomToken。唯一可执行 packet 是
`docs/aris/ADATAD_SHARED_OFFICIAL_BASELINE_PACKET-2026-08-17.md`；其最终 receipt 必须同时绑定
clean release、未改 config、canonical THUMOS14 411、checkpoint 或 pretrain、seed、evaluator/NMS、
EMA/final、运行时和唯一结果根。先评测 released checkpoint；仅在该 artifact 确实不可得且同一
负责人确认必须时，才运行一次 clean untouched official training。所有其他项目只读消费 receipt，
禁止第二次复现。

此决定不把 ZoomToken 整体置为等待状态：未绑定的 shared dense 数字阻止性能/论文解释，但不阻止
Q 正式矩阵入口的确定性修复与审查、ROI/残差条件对照的协议准备、5-epoch 可恢复 checkpoint
机制或 PRE_RUN。任何新远端方法矩阵仍需独立明确授权；`66.xx` 永远只保留为 matched-source
provenance。

## 27. DSH 外部审查完成，但只作为实现审查输入（2026-08-17）

fresh `deepseek-v4-pro` DSH 会话的身份、首请求与唯一 `/^We need\b/` 指纹均由
`docs/aris/DSH_FORMAL_REVIEW_RECEIPT-2026-08-17.md` 持久化，终态为 `completed`；可见中文报告在
`docs/aris/DSH_FORMAL_REVIEW_REPORT-2026-08-17.md`。这是一次只读外部审查，不是 Pro 科学裁决、
Critic PASS、Evaluator PRE_RUN 或新实验结果。

审查保持 Q-core 主问题不变，并把 5491 的 `NO_SURVIVOR_INVALID_P1` 重新确认成 implementation/
protocol admission failure。它提出三个应由既有 Builder→Critic→Evaluator 链独立复核的最小实现
问题：官方 DO report-only cost leaf 不得要求 GeoRoute audit、five-epoch recovery 必须有可达且
unsealed 的 resume 入口、shared official receipt 必须实际校验 69.03/48.27 anchor 与 clean
official provenance。审查不授权扩大路线、打开 G/N/F、重试封存 root 或发起任何远端运行。

## 28. b798 独立 Critic 确认准入修正，不改变科学路线（2026-08-18）

新的独立只读 Critic 在干净 `b798e9f…` 完成审查，结论为
`IMPLEMENTATION_CORRECTION`。它确认 Q 的 global exact-B、动态 `K_t`、ragged/no-padding、
masked-zero、Q/DN 成本分母、条件 G/N/F 以及静态 no-leak 边界没有漂移；这不是实际运行或
性能证据。

两个独立确认的阻断点是：五 epoch recovery 虽保存完整状态，却被现有 `work_dir` 与 cell-root
拒绝逻辑阻断，因而没有可达的 unsealed resume；stage/cost/finalizer 虽相互检查 136 window/40
video，却未与冻结 manifest 的有序物理身份比较。与已接受外部审查合并后，限定下一次实现为：DO
只能消费带身份的共享 official receipt 而不能运行 GeoRoute cost leaf；receipt 必须区分 released
checkpoint evaluation 与 untouched official reproduction，防止 `66.xx` 进入官方槽位；同时实现
真正可达且可测试的恢复入口，并把 population manifest 作为 admission binding。以上均不改变
Q-core、数据划分、指标、阈值或路线；修复后必须再走独立 Critic 和新 external review，PRE_RUN
仍然关闭。

## 29. ROI-first 60-epoch 官方可比合同待裁决（2026-08-20）

用户要求以 ROI 为优先对象，完成一次可与原始官方 AdaTAD 比较的 60-epoch 真实 THUMOS14
实验。历史项目记录显示 ROI-only 只有 20-epoch 开发诊断（13.18 Avg-mAP、8.95 mAP@0.7）、
一次评测前视频解码失败的 20-epoch 训练，以及没有检测/成本闭环的 continuous-ROI 60-epoch
training-only 矩阵；这些都不能称为官方可比证据。远端 BATA 目录名检索亦未发现一个遗漏的
ZoomToken ROI 正式终结回执。

当前已接受 Q-first 顺序把 G/N/F 置于 Q gate 之后，而 matched-source Q 在两个 seed 有负向
观察。故“ROI-first 60 epoch”是路线顺序和实验合同的科学问题，不能由 Builder 擅自决定。唯一
待裁决材料 `docs/aris/ROI_OFFICIAL_COMPARABLE_60EPOCH_MATERIAL-2026-08-20.md` 要求精确冻结
ROI-only 与 ROI+residual 的因果 arm、DO official receipt 与 DN 的角色、强对照、预算、seeds、
官方评测、全栈成本、停止规则与论文边界；未裁决前不实现或提交 ROI 矩阵。

## 30. ROI60 用户提供外部建议：部分采纳并修正（2026-08-20）

用户提供的 ROI60 执行裁决建议收敛为 seed 3407、60 epoch 的 ROI-modifier-only `G` 对同源
dense `DN`。项目核验后采纳其主方向：`G` 是 `q_base + ROI modifier`、residual 关闭的 dynamic
exact-B 原生 token 路由，不是硬 ROI crop；旧 ROI 数字、Q 负向观察、`5491c580...` 失效矩阵和
`66.xx` 均不能越级成官方或 ROI 结论。

但不接受其执行合同的三个部分：报告的 `7f0d0eb...` 不是指定 GitHub 分支的最新提交
（`2e99ce0...` 是其后继）；未证明身份等价的旧 DN 不得默认复用；不得在 epoch-60 的 raw-final
和 EMA 间按看到的 validation 数字择高，必须事先统一地固定一个选择规则。另外，单种子的“升级”
不能由单独的 latency 或容忍性 mAP@0.6 触发；它需要通过主要准确率与端到端时延的联合条件。

具体实现范围仍需 Builder 在 `2e99ce0...` 后继干净候选上提出最小变更计划，并经 Critic 和
Evaluator 独立核查。该决定没有启动训练或产生性能结果。

## 31. 共享官方 AdaTAD 基线进入唯一复现（2026-08-20）

官方 released checkpoint 尚无可验证副本，故不以 matched-source `66.xx` 代替。用户授权后，项目
在 clean `01c58b9...` release、未修改官方 config、seed 42 与 canonical THUMOS14 上启动唯一
60-epoch fallback reproduction。它只回答上游 `69.03/48.27` 是否可在本地环境复核；不包含
ZoomToken、ROI、residual、Q 或官方 test。结果完成前仅为 execution evidence。

## 32. ROI-only 60 轮配对实验进入真实执行（2026-08-21）

在历史 Q-core 两个 seed 重复低于 matched dense/uniform/random 后，用户批准以最小机制校正直接
检验 ROI。当前 clean official-base revision `321f1f76…` 只打开 ROI modifier，关闭 residual，保留
全局唯一 exact `B=24576`、动态 `K_t`、ragged/no-padding 和 no-leak 约束。其数据变换缺失被 fresh
external review 发现后，只恢复了历史已审的 `GeoRouteSourceViews`，并经独立 Critic PASS；没有改变
模型目的、数据划分、指标或训练配方。

决定执行同源 DN 与 ROI-only G 的 seed-3407、60-epoch配对训练。第一对 jobs
`1245897/1245898` 因优化器参数组别名在有效训练前终止；G 的后续
`1245908/1245909/1245910` 又依次暴露并定位了辅助损失在 DDP 与实际
`ActionFormer.forward_train` 路径中的所有权问题。这些均为无性能输出的实现诊断。

最终入场运行是 DN `1245907`（clean `d2b5de05…`）与 ROI-only G `1245924`
（clean `59960255…`）。两臂从同一 VideoMAE-S 预训练开始，使用 canonical THUMOS14
training/validation、同一优化/EMA、官方 validation evaluator/NMS、同一单卡资源以及每 5 epoch
可恢复 checkpoint；DN 路径未受后续 G-only 计算图修正影响。DN 已进入 epoch 7 并保存首个 recovery，
G 已完成至少 50 个有限损失的真实优化更新。共享 clean AdaTAD `01c58b9…` reproduction `1245842`
保持独立，只提供官方锚点；DN 不得冒充它。训练运行中不作性能推断，终态先裁决 G 相对 DN 的准确率，
只有同硬件完整成本测量成立后才解释成本，再决定是否扩展 residual 或多种子。

## 33. ROI-only 首个可恢复检查点形成（2026-08-21）

ROI-only G job `1245924` 完成 epoch 4、进入 epoch 5，并在冻结结果根写出约 628 MB 的
`recovery_epoch_4.pth`。这证明 5-epoch 周期恢复机制已经在真实 THUMOS14 训练中落地；它仍只是
运行与可恢复性证据，不包含 validation 性能或成本结论。同一时刻，DN `1245907` 已进入 epoch 13，
具有 epoch 4/9 两个恢复点；共享官方 AdaTAD `1245842` 已进入 epoch 36并保存 `epoch_35.pth`。
三项作业仍在运行，继续等待预注册的 60-epoch final/final-EMA 评测，不追加或重排实验。

截至 03:54，DN 已进入 epoch 15 并形成第三个周期恢复点 `recovery_epoch_14.pth`；共享官方
AdaTAD 已进入 epoch 38并形成 `epoch_37.pth`，G 已进入 epoch 6。该进展不改变路线与终态
裁决规则，只继续确认既有训练与恢复策略按预注册节奏运行。

## 34. 三项 60 轮训练完成，但正式评测尚未形成（2026-08-21）

只读 Slurm 摄取确认：共享官方 AdaTAD `1245842`、DN `1245907` 与 ROI-only G `1245924`
均为 `COMPLETED 0:0`，均完成 epoch 59 的最后训练迭代；日志未见硬故障。DN/G 各保留
epoch 44/49/54 三个恢复点。终态结果根未发现 final validation、stage result 或 finalization
收据，因此该事件只升级为“完整训练完成”，不升级为“官方复现”或“ROI 性能结果”。下一步
只允许从既有终态 checkpoint 按预注册 final/final-EMA 规则完成官方 evaluator 评测，不重复训练。

## 35. 原始终态日志补充核验：ROI-only G 低于 DN（2026-08-21）

上一节仅检索结构化结果文件，遗漏了训练日志末尾由官方 evaluator 直接打印的 validation 指标。
补充只读核验确认：clean official AdaTAD reproduction 的 Avg-mAP/mAP@0.6/mAP@0.7 为
`68.73/61.58/47.24`；matched-source DN 为 `64.73/56.14/43.26`；ROI-only G 为
`61.49/53.42/39.99`。G 相对 DN 分别低 `3.24/2.72/3.27` 个百分点，且在 tIoU 0.3–0.7
全部阈值均更低。因此当前 ROI-only G 被判为准确率负结果，不扩展为 residual 或多种子结论。

官方路径按未修改配置在后 20 轮每 2 轮评测一次；DN/G 启动脚本明确只在第 60 轮后评测一次，
因此缺少中途性能曲线。这是执行合同选择，不是 evaluator 故障。三项仍无完整端到端成本证据，
不能据此判断效率或形成论文主张。

随后对 DN/G 的 epoch 44/49/54 EMA recovery checkpoint 进行了 validation-only 补充评测。六项
作业全部正常完成，DN Avg-mAP 为 `65.50/65.06/64.84`，G 为 `62.42/62.00/61.80`，差值
`-3.08/-3.05/-3.04`；终态为 `-3.24`。因此拒绝“G 只是在最后几轮偶然退化”的解释，下一科学
动作改为分析 ROI 覆盖与边界信息损失，而不是延长训练或重复提交同一配置。

## 36. 用严格三臂归因区分 adapter 与 ROI（2026-08-21）

首轮 DN/G 与官方作业在 seed、GPU 数和执行路径上并非严格的一变量三臂比较，因此不能回答性能
下降究竟来自 sparse adapter 还是 ROI。新的冻结设计将 A 固定为已完成的未修改官方 AdaTAD
job `1245842`；B 在同一官方 dense 主干输出上以 sparse adapter 聚合全部 token；C 与 B 共享
同一主干和 adapter，仅把聚合支持改为后主干 ROI `K=64`。这使 A→B 估计 adapter 聚合影响，
B→C 估计 ROI 支持选择的增量影响，同时明确放弃把该后主干实验包装为计算节省证据。

实现 revision `1a18565b…` 已通过 21 项静态测试和独立 Critic；结果盲 PRE_RUN 通过后提交 B/C
jobs `1247290/1247291`，同为 seed 42、双卡 local batch 1/global batch 2、60 轮官方训练/评估
合同。两项当前排队，尚无性能结论；A 没有重复训练。

## 37. 三臂归因矩阵进入真实训练（2026-08-22）

B/C 两项作业已在 2026-08-21 23:46 同时取得双卡资源并开始训练。只读核验时，B 已进入 epoch 37，
C 已进入 epoch 34；二者按未修改官方节奏每两轮保存恢复 checkpoint，且未出现 Traceback、显存溢出
或非有限损失。该事件只把证据状态从“已排队”提升为“实验运行中”，不产生 adapter 或 ROI 的性能
结论。继续保持 A 不重复训练，并等待 B/C 的 60 轮 final/final-EMA 官方 validation 后进行 A→B、
B→C 两个预注册差分。

## 38. 首个同阶段中间验证分离出 adapter 与 ROI 信号（2026-08-22）

第 42 轮训练开始前的同阶段官方 validation 中，A/B/C 的 Avg-mAP 分别为 `67.88/67.06/67.86`，
mAP@0.7 为 `46.19/45.72/46.14`。单个中间节点显示 B 相对 A 下降，而 C 相对 B 恢复约 `0.80`
Avg-mAP；C 暂时接近 A。该结果支持继续完成预注册 60 轮，以检验这一差异是否稳定；它不授权挑选
中间 checkpoint，不足以形成最终准确率结论，也没有任何重骨干计算节省含义。

## 39. 旧 ROI-G 与当前 ROI-C 的代码身份回验（2026-08-22）

旧 G 与当前 C 不能解释为同一 ROI 方法的两次重复。旧 G 在 VideoMAE 之前按全局
`B=24576` 从 38,400 个原生空间 token 中删除 36%，采用动态 `K_t`、真实 ragged 重主干和
masked-zero 载体；当前 C 先完整执行官方稠密 VideoMAE，再从每个 tubelet 已完成的 100 个
稠密特征中固定选择 64 个供同一 adapter 聚合。旧 DN/G 还使用 seed 3407、单卡和 GeoRoute
source/loss 路径；当前 A/B/C 使用 seed 42、双卡和官方 dense 主干路径。

官方 A 从 epoch-41 中间验证到 epoch-59 终态只增加约 `0.85` Avg-mAP，不能解释旧 G
`61.49` 与当前 C 中间值 `67.86` 的差距。主要归因是主干前空间支持删除及旧 GeoRoute
训练图，而非最后若干训练轮。当前 C 可隔离稠密特征上的 ROI 聚合准确率影响，但不构成
VideoMAE 重计算减少或真实成本下降的证据。

用户进一步明确最终方法边界：ROI 必须在 VideoMAE 重主干之前选择原生 token，重主干不得先
执行完整稠密计算。因而当前 C 只作为诊断矩阵保留，不进入最终方法身份；下一正式实现必须在
同一官方 AdaTAD 配方下比较 full-token 与 pre-backbone ROI，并同时报告准确率和完整端到端成本。

## 40. 主干前固定 ROI 的严格一变量因果实验（2026-08-22）

为直接回答 `69→64` 与 `64→60` 的来源，冻结三臂为：A 只读复用未修改官方 AdaTAD；B 让
全部 100 个原生空间 token 通过同一 native/ragged VideoMAE 与 sparse adapter；C 只把重主干
之前的支持集合改为 ROI fixed `K=64`，其余路径与 B 相同。该设计拒绝后主干 ROI、动态 `K_t`、
residual 和额外辅助损失，因此 A→B 估计 adapter/接入路径损失，B→C 估计 ROI 删除增量。

clean revision `70dcbe10…` 经过 Builder、独立 Critic 和 Evaluator。Critic 先发现 fixed-support
分支遗漏训练端 regularization 生命周期，修复后又由真实 dataloader 暴露 job-global batch 被误写
为 1；两项均为 claim-preserving 实现缺陷，而非科学失败。最终双卡无指标前向验证 B/C 分别执行
38,400/24,576 个原生 token、单次重主干、零 padding，并输出相同形状。正式 B/C jobs
`1248835/1248834` 已进入 epoch 0；在 final/final-EMA validation 前不作性能归因。

## 41. 主干前固定 ROI 严格三臂终态裁决（2026-08-22）

B job `1248835` 与 C job `1248834` 均完成 60 轮训练并以 `COMPLETED 0:0` 终止，各保存
30 个周期 checkpoint，日志未见 Traceback、OOM 或非有限数值。官方 validation 的
Avg-mAP/mAP@0.6/mAP@0.7 为：A `68.73/61.58/47.24`，B `68.51/61.19/46.27`，
C `68.22/61.01/45.35`。

因此 A→B 为 `-0.22/-0.39/-0.97`，B→C 为 `-0.29/-0.18/-0.92`，A→C 为
`-0.51/-0.57/-1.89` 个百分点。结论是：当前 native-ragged/sparse-adapter 接入没有造成旧
matched-source 实验中的约 4 点 Avg-mAP 下降；固定 ROI `K=64` 删除 36% 空间 token 的平均
准确率增量代价也只有 0.29 点，但两阶段在高 tIoU 上分别损失约 1 点。这是 seed 42 的正式
准确率归因证据，尚未包含端到端延迟、能耗和显存，故不能晋级为效率 claim。下一动作是固定
上述终态模型身份，在同硬件上测量完整端到端成本，而不是重复旧训练或扩展 ROI/residual 路线。

## 42. `69 / 64 / 60` 路径身份与主干前边界裁决（2026-08-22）

完整代码回验确认，发表锚点 `69.03`（本地未修改官方复现 `68.73`）、旧 matched-source DN
`64.73`、旧 ROI-only G `61.49` 不是逐项只加入 adapter、再加入 ROI 的同配方消融。旧 DN/G
改用原生分辨率 `GeoRouteSourceViews`、native-packed/sparse-adapter、seed 3407、单卡 batch 1、
warmup 2/max 60；G 进一步使用全窗动态 exact-`B=24576`、可为零的 `K_t` 和 auxiliary/proxy
损失。故 `69→64` 是混合实现/训练效应，`64→60` 是旧动态强稀疏路径的联合负结果。

当前正式 revision `70dcbe10…` 明确继承官方增强/训练配方，并在原始 tubelet 上先选 support、
后且仅后执行一次 `forward_native_ragged`。B/C 只差每 tubelet 的 `100` 与 ROI `64` 支持，
patch embedding 和全部 VideoMAE blocks 只接收选中 token；目标双卡检查与 60 轮 jobs 进一步
确认 `38,400/24,576` 个 heavy token、单次 heavy、零 padding。最终裁决：当前 C 是主干前
ROI，不是后主干聚合；其相对本地官方 Avg-mAP 代价为 0.51 点，尚需完整端到端成本裁决。

## 43. Pro 裁决：先以严格矩形 R1 检验高 tIoU 损失（2026-08-22）

全新 exact-Project Pro 会话对 `70dcbe10…` 的代码、正式 A/B/C 结果和严格矩形设计作出
`REVISE` 裁决，而非 `PIVOT` 或 `STOP`。它确认当前 C 的确在 VideoMAE patch embedding 之前
删除原生 token，并且只执行一次 ragged 重主干；但当前支持由椭圆/高斯分数 Top-64 得到，不能
称为无孔洞矩形。C 相对 B 的 Avg-mAP 只低 0.29 点，而 mAP@0.7 低 0.92 点，因此下一问题被
收敛为：固定同样 `K=64` 时，完整矩形支持拓扑是否能恢复高 tIoU 边界定位。

唯一立即路线 R1 在 `10×10` 原生网格上从九个合法 `8×8` 完整矩形中稳定选择一个，执行且仅
执行其 64 个 token；不加入框内 Token Select、动态面积、框外 free token、residual 或新的成本
平台。首个 falsifier 是现有 C 对新 R1 的 seed-42、60-epoch 比较；R1 至少恢复 0.50 个百分点
mAP@0.7 且 Avg-mAP 不低于 0.20 点后，才进入同硬件端到端成本比较。R2/R3/R4 只冻结为后置
候选，不能同时实现或用于挽救 R1 失败。原始裁决与项目吸收见
`docs/aris/ZOOMTOKEN_STRICT_RECTANGLE_ROI_PRO_RESULT-v002-2026-08-22.md` 和
`docs/aris/PRO_STRICT_RECTANGLE_ROI_DECISION-v001-2026-08-22.md`。

## 44. 严格矩形 R1 实现冻结与正式训练启动（2026-08-22）

R1 在 `70dcbe10…` 的主干前 C 路径上作最小实现，最终 clean revision 为
`9e25c6d38de8c993948025629181470b858682b4`。实现只把椭圆/高斯 Top-64 支持替换为九选一
完整 `8×8` 矩形，仍固定 K64；raw native tubelet 在 patch embedding 前 gather，只执行一次
true-ragged VideoMAE，复用相同 sparse adapter、ActionFormer 与官方训练/评测配方。

独立 Critic 确认路由主体与配方无漂移；两次聚焦修正仅闭合 full-state recovery、单一原子 final
以及真实 capture/restore 测试，没有改变科学机制。N16R4 目标环境 9 项无数据 Torch 检查通过，
Evaluator 判定 `PRE_RUN_READY`。历史 C job `1248834` 的 seed、checkpoint endpoint、数据和 evaluator
身份匹配，可直接作为对照。因此只提交一个新的 R1 seed-42、60-epoch job `1249099`，不重训 C、
不启动成本或 R2/R3/R4。R1 结果必须先通过预注册准确率门，才允许后续成本测量。

## 45. 多分支严格矩形矩阵进入真实训练（2026-08-22）

用户进一步批准同时检验矩形内部选择、矩形外补充和动态面积，不再把 R2/R3/R4 全部
推迟到 R1 终态。为避免“多臂但无法归因”，每个方法只配一个能直接破坏其机制的对照：
R2 与框内分数乱序、全局 Top-48 比较；R3 与时间错位的面积轨迹比较；R4 与框外分数乱序、
全局 Top-64 比较。R4 固定为 7×7 core49 + 框外 Top15，总 K=64；旧 6×8+16 草案不执行。

clean revision `b1d9fa7b…` 只改 production routing 和 focused tests，保持选择发生在
VideoMAE patch embedding 前、单次 true-ragged heavy forward、零 padding、同一 sparse
adapter、ActionFormer、官方训练/评估配方和 no-leak 边界。独立 Critic PASS，目标环境
PRE_RUN_READY 后，八个 seed-42、60-epoch 单元 `1249125–1249132` 已一次性释放；六个进入
epoch 0，两个仅等待 GPU 并发额度。决定边界：在完整 validation 和同硬件成本形成前，不从
启动状态、token 数或单个幸存单元作方法优劣推断，也不改变正在运行的 R1 `1249099`。

## 46. 首个严格矩形恢复点进入运行证据（2026-08-22）

R1 job `1249099` 在不干预训练的只读核验中已进入 epoch 21，并保存 epoch 9/14/19 三个
full-state recovery；R2/R3/R4 六个已调度单元进入 epoch 2，两个对照仍等待账户 GPU 并发额度。
恢复点只证明中断恢复合同实际生效，不允许据此选择 checkpoint 或推断性能。当前仍以预注册的
final-EMA 为主、raw final 为次；在正式 validation 出现前，路线与裁决阈值均保持不变。

## 47. R2/R3/R4 矩阵首个恢复点（2026-08-22）

R2-SHUF48 在 epoch 4 后发布本矩阵首个 full-state recovery，其余五个运行单元同期进入 epoch 4，
两个对照仍等待账户 GPU 配额。该事件只验证恢复机制的运行可达性，不改变矩阵、性能门或模型
选择规则；所有分支仍必须等待完整官方 validation，并以 final-EMA 为主结果。

## 48. 严格矩形六个运行单元均形成首个恢复点（2026-08-22）

R1 已进入 epoch 26，并按最新三个恢复点策略保留 epoch 14/19/24；R2、R2-SHUF48、
Q48-GLOBAL、R3、R3-AREA-SHIFT 与 R4 均进入 epoch 7，且全部形成 epoch 4 恢复点。
R4-SHUF15 与 Q64-GLOBAL 继续等待账户 GPU 并发额度。该进展只证明训练与恢复链持续可用；
在正式 validation 与 final/final-EMA 形成前，不改变科学路线、阈值或模型选择规则。

## 49. R2/R3/R4 六个运行单元形成第二个恢复点（2026-08-22）

R2、R2-SHUF48、Q48-GLOBAL、R3、R3-AREA-SHIFT 与 R4 均进入 epoch 10，并各自形成
epoch 4/9 两个 full-state recovery；两个未调度对照继续等待账户 GPU 配额。R1 同期进入
epoch 29，尚未形成新的第 5 轮间隔恢复点。该事件不改变既定比较、停止规则或 final-EMA
模型选择，只确认六条真实训练链持续可恢复。

## 50. 严格矩形路线首批正式中间验证（2026-08-22）

R1 的最新中间 Avg-mAP/mAP@0.6/mAP@0.7 为 `68.63/60.84/46.60`，相对 C 的
`+0.41/-0.17/+1.25` 暂时满足预注册准确率门。R2 对 R2-SHUF48 的高 tIoU 优势为
`+1.09`，但相对 Q48-GLOBAL 为 `-0.04`；R3 相对面积轨迹错位在 mAP@0.6 为 `+0.92`、
在 mAP@0.7 为 `-0.15`。因此当前最合理的读法是：完整矩形对 R1 的高 tIoU 恢复具有正向
信号，R2 的框内内容排序也有正向信号，但矩形 eligibility 和动态面积的独立优势尚未成立。
该判断只决定继续原计划等待终态，不授权 checkpoint 选择、成本作业或路线晋级。

## 51. 严格矩形训练端点前信号更新（2026-08-22）

R1 在 epoch 59 前的最新中间 Avg-mAP/mAP@0.6/mAP@0.7 为 `68.75/60.95/46.55`，相对 C
为 `+0.53/-0.06/+1.20`；训练端点 checkpoint 已发布，但 final-EMA 与终态仍缺。R2 相对
R2-SHUF48 和 Q48-GLOBAL 的最新三指标差值分别为 `+0.36/+0.61/+1.12` 与
`+0.49/+0.63/+0.65`，使严格矩形 eligibility 与框内排序在当前节点同时呈正向信号；R3
相对面积轨迹错位为 `+0.30/+0.71/-0.06`，尚不能主张高 tIoU 改善。决定保持不变：等待
预注册 final/final-EMA，不从中间 checkpoint 选模，不提前启动成本或补充实验。

## 52. R1 完整矩形通过 seed-42 准确率门（2026-08-22）

R1 job `1249099` 完成 60 轮，终态 EMA 的 Avg-mAP/mAP@0.6/mAP@0.7 为
`69.07/61.14/46.57`，相对 C 为 `+0.85/+0.13/+1.22` 个百分点，三项预注册条件全部通过。
因此当前可以接受“在相同 K64、adapter 与训练配方下，完整 8×8 矩形支持比 C 的不规则 Top-64
支持更有利于 seed-42 准确率，尤其是高 tIoU 定位”；不能接受“已证明端到端效率”或“已经
多 seed 稳定”的扩张表述。R2/R3/R4 矩阵继续原计划完成，R1 的配对成本只在后续独立授权与
完整准确率摄取后执行；本轮不追加实验。

## 53. R2 框内排序信号保留，矩形 eligibility 仍待终态（2026-08-22）

epoch 54 前后的 R2/R2-SHUF48/Q48-GLOBAL 中间结果为
`66.28/58.76/44.75`、`65.90/58.24/43.80`、`65.77/58.51/44.74`。R2 相对乱序
对照的三指标差值 `+0.38/+0.52/+0.95` 继续支持框内内容排序；相对全局 Top-48 的
`+0.51/+0.25/+0.01` 则显示高 tIoU 几乎持平。因此当前不把 R2 解读为矩形 eligibility 已经
优于全局选择，也不据此改模型或挑 checkpoint；继续等待预注册终态及完整对照矩阵。

## 54. R3 对时间错位面积轨迹形成一致的中间优势（2026-08-22）

epoch 54 前后的 R3/R3-AREA-SHIFT 中间 Avg-mAP/mAP@0.6/mAP@0.7 为
`67.64/59.89/45.95` 与 `67.27/59.67/45.14`，差值为 `+0.36/+0.22/+0.81`。这使连续
动态矩形相对时间错位面积轨迹在当前节点的平均与高 tIoU 指标上同时为正，但仍不能替代
预注册终态。R4 同期为 `68.01/60.45/46.23`，由于 R4-SHUF15 尚处训练早期且 Q64-GLOBAL
尚未调度，不作 R4 机制结论。决定保持矩阵、阈值与 final-EMA 选择规则不变。

## 55. R2 的矩形 eligibility 与框内排序信号在后期中间节点同时为正（2026-08-22）

epoch 56 前后的 R2/R2-SHUF48/Q48-GLOBAL 中间 Avg-mAP/mAP@0.6/mAP@0.7 为
`66.44/59.08/44.93`、`66.03/58.37/44.53` 与 `65.64/58.48/44.41`。R2 相对乱序
对照为 `+0.41/+0.71/+0.40`，相对全局 Top-48 为 `+0.80/+0.60/+0.52`。这支持继续保留
两项机制假设，但此前节点的差值曾接近零，故不把单次中间验证升级为机制结论；继续等待
预注册 final-EMA，不改变训练、阈值或 checkpoint 选择规则。

## 56. 连续动态矩形的对齐优势在更晚中间节点扩大（2026-08-22）

epoch 56 前后的 R3/R3-AREA-SHIFT 中间 Avg-mAP/mAP@0.6/mAP@0.7 更新为
`67.89/60.17/46.31` 与 `67.42/59.95/45.00`，差值为 `+0.46/+0.22/+1.31`。这延续并扩大
了连续动态矩形相对时间错位面积轨迹的高 tIoU 优势，但仍只是过程性 validation，不替代
预注册终态。R4 同期为 `68.03/60.41/46.11`；其 R4-SHUF15 对照仅进入 epoch 11，Q64-GLOBAL
仍未调度，因此继续不作 R4 机制结论，也不改变模型、训练或 checkpoint 选择规则。

## 57. R2 两项对照接近训练端点，机制判断仍冻结至终态（2026-08-22）

R2-SHUF48 与 Q48-GLOBAL 在进入 epoch 59 前的最新官方中间 Avg-mAP/mAP@0.6/mAP@0.7
分别为 `66.03/58.26/44.39` 与 `65.88/58.75/44.73`；R2 的最新可见值仍为
`66.44/59.08/44.93`。按异步快照计算，R2 对两项对照仍为正，但本轮没有取得 R2 的同阶段更新，
故不把差值升级为机制结论。R2-SHUF48 已形成 `epoch_59.pth`，这只证明训练到达端点，不能替代
final/final-EMA。决定继续等待终态，不选择中间 checkpoint，不改变训练或补充实验。

## 58. 训练端点前信号与最后一项对照入场（2026-08-22）

R2/R2-SHUF48/Q48-GLOBAL 的最新同阶段中间结果为 `66.44/58.81/45.10`、
`66.03/58.26/44.39`、`65.88/58.75/44.73`，R2 对两项对照仍为正，但对 Q48 的
mAP@0.6 仅 `+0.06`，故矩形 eligibility 尚不能提前判定。R3/R3-AREA-SHIFT 最新为
`67.83/60.00/46.56` 与 `67.27/59.76/44.60`，对齐轨迹在三个指标上的差值扩大到
`+0.56/+0.24/+1.96`，仍须终态确认。Q64-GLOBAL 已从资源等待转为运行，R4 的机制对照链
终于齐备。决定保持所有配置和 final-EMA 规则不变，等待终态，不追加或重排实验。

## 59. 首批后续矩阵单元完成，但机制裁决等待完整终态（2026-08-22）

R2、R2-SHUF48、Q48-GLOBAL 与 R4 已完成 60 轮并以 `COMPLETED 0:0` 退出；R4 训练日志末次
官方 validation 为 `68.02/60.32/46.26`（Avg-mAP/mAP@0.6/mAP@0.7）。然而本次单次只读
检查尚未取得四项结构化 final-EMA 收据，且 R3、R3-AREA-SHIFT、R4-SHUF15、Q64-GLOBAL
仍在运行。因此不以训练端点 checkpoint 或日志末值替代预注册终态，不裁决框内排序、矩形
eligibility、动态矩形或框外 free-token 机制，也不启动成本或补充实验。下一动作仅为摄取既有
终态收据并等待剩余作业自然完成。

## 60. 终态支持框内排序与动态矩形时间对齐；R4 继续等待配对对照（2026-08-22）

六个完成单元的终态官方 validation 已从原始训练日志摄取。R2 相对 R2-SHUF48 的
Avg-mAP/mAP@0.6/mAP@0.7 增量为 `+0.39/+0.53/+0.70`，相对 Q48-GLOBAL 为
`+0.78/+0.44/+0.43`，因此在 seed 42 上保留“矩形内内容排序有效且矩形 eligibility 本身有益”
的机制解释。R3 相对时间错位面积轨迹为 `+0.38/+0.06/+1.32`，保留“时间对齐的动态矩形
主要保护高 tIoU 定位”的解释；其中 @0.6 增量接近零，不能夸大为全阈值一致提升。

R4 的终态为 `68.02/60.32/46.26`，但其乱序框外对照和 Q64 全局对照仍在训练，因此框外
free-token 的增量继续冻结。下一动作仍是等待既有两项对照自然完成；不据过程 checkpoint 选模，
不补臂、不启动成本，也不把单 seed 准确率升级为论文级效率结论。

## 61. R4 机制裁决继续等待完整配对终态（2026-08-22）

R4-SHUF15 已运行至 epoch 25并形成 epoch-24 恢复点，Q64-GLOBAL 已运行至 epoch 9；两项
均无新的正式 validation 或硬故障。恢复点只证明可恢复训练链正常，不提供模型选择或机制证据。
因此维持上一决策：不从 R4 单臂终态推断框外 free-token 作用，等待两项对照完成后再比较。

## 62. Q64 全局对照恢复链形成，R4 裁决保持冻结（2026-08-22）

Q64-GLOBAL 已运行至 epoch 12并形成 epoch 4/9 两个恢复点；R4-SHUF15 已运行至 epoch 28，
最近恢复点仍为 epoch 14/19/24。两项均无新的正式 validation 或硬故障。新增恢复点只支持训练
连续性，不能替代终态模型选择或机制比较。因此继续等待两项既有作业自然完成，不改变配置，
不启动成本、补充实验或恢复操作。

## 63. 两项 R4 对照的三点恢复链均已形成，机制裁决不提前（2026-08-22）

R4-SHUF15 已运行至 epoch 31并形成 epoch 19/24/29 三个最近恢复点；Q64-GLOBAL 已运行至
epoch 15并形成 epoch 4/9/14 三个恢复点。两项均无新的正式 validation 或硬故障。恢复链完整
只说明训练可持续恢复，不提供框外 free-token 的性能归因。因此仍等待两项自然完成，以冻结的
final-EMA 为主、raw final 为辅进行配对比较，不启动成本、补臂或恢复操作。

## 64. 两项 R4 对照继续稳定训练，维持终态优先（2026-08-23）

R4-SHUF15 与 Q64-GLOBAL 分别新增 epoch-34 与 epoch-19 的完整恢复点，最近三份恢复状态更新为
epoch 24/29/34 与 epoch 9/14/19；两项仍在运行，本轮没有新的正式 validation、终态或硬错误
回执。该进展只证明训练链继续可恢复，不改变机制判断。继续等待两项自然完成，并仅以冻结的
final-EMA 为主、raw final 为辅比较 R4、R4-SHUF15 与 Q64-GLOBAL；不启动成本、补臂或恢复操作。

## 65. R4 乱序对照进入后 20 轮，仍不提前解释机制（2026-08-23）

R4-SHUF15 已进入 epoch 40并形成 epoch-39 恢复点，Q64-GLOBAL 已进入 epoch 23；两项均无
新的正式 validation、终态或硬错误。进入后 20 轮只改变可观测训练阶段，不改变裁决规则。
继续等待冻结的 final-EMA 终态，以 R4、R4-SHUF15 和 Q64-GLOBAL 的完整配对结果判断框外
free-token 作用；不据恢复点或异步进度选择模型，也不启动成本、补臂或恢复操作。

## 66. Q64 全局对照形成后半程恢复链，R4 裁决继续冻结（2026-08-23）

Q64-GLOBAL 新发布 epoch-24 完整恢复点，最近三份恢复状态更新为 epoch 14/19/24；
R4-SHUF15 最近三份恢复状态仍为 epoch 29/34/39。两项均保持运行且尚无新终态，因此恢复链
只用于证明训练可继续，不改变科学比较。继续等待 R4、R4-SHUF15 与 Q64-GLOBAL 的冻结
final-EMA 终态，不据恢复点选模，也不启动成本、补臂或恢复操作。

## 67. R4 乱序对照出现首个中间性能，终局归因仍冻结（2026-08-23）

R4-SHUF15 在 epoch 42 后的官方中间 validation 给出 Avg-mAP `65.87`，低于 R4 的终态
Avg-mAP `68.02`，但本次快照没有同时取得 mAP@0.6/mAP@0.7，且对照尚未完成 60 轮。
Q64-GLOBAL 已进入 epoch 31并形成 epoch 19/24/29 三个最近恢复点，尚无正式 validation。
因此该观察只支持继续完成既定对照，不能选择中间 checkpoint、裁决框外 free-token、启动成本或
补充实验；最终判断仍以 R4、R4-SHUF15、Q64-GLOBAL 的冻结 final-EMA 为准。

## 68. R4 乱序对照的完整中间三指标仍低于 R4，继续等待终态（2026-08-23）

R4-SHUF15 在 epoch 43 后的最新官方中间 validation 为 `66.27/59.02/44.59`，相对 R4
终态 `68.02/60.32/46.26` 暂低 `1.75/1.30/1.67` 个百分点。该一致负向差值加强了“框外
内容排序可能有用”的工作假设，但不能构成结论，因为比较节点并非同一训练阶段，SHUF15 尚未
完成 60 轮，Q64-GLOBAL 也尚无正式 validation。维持既定决定：不据中间 checkpoint 选模，
不启动成本或补臂，只等待两项冻结 final-EMA 完成 R4 机制归因。

## 69. 两项尾部对照新增可恢复检查点，不改变科学裁决（2026-08-23）

01:06 CST 只读核验确认，R4-SHUF15 与 Q64-GLOBAL 仍为 `RUNNING`，分别新增
`recovery_epoch_44.pth` 与 `recovery_epoch_34.pth`；最近三个恢复点更新为 34/39/44 与
24/29/34。R4-SHUF15 最新中间 validation 仍为 `66.27/59.02/44.59`，Q64-GLOBAL 仍无
正式 validation，且未发现 Traceback、显存溢出或非有限数值。该事件只证明训练连续性，不提供
新的性能或机制证据；继续等待冻结 final/final-EMA，不据恢复点选模，也不启动成本或补臂。

## 70. token-selection 主图采用各方法自身 checkpoint，并隔离未终态行（2026-08-23）

为回答“同一样本、不同帧上实际选择了哪些 native token”，主图冻结为 on-policy 形式：每行使用
该方法自己的 checkpoint，并在同一个 THUMOS14 validation 滑窗上运行 production selector。
R4-SHUF15 与 Q64-GLOBAL 尚未完成 60 轮，因此只绑定当时最新的
`recovery_epoch_44.pth/recovery_epoch_39.pth`，在行标签和图注中明确为“仅作定性观察”。该图用于
检查矩形连通性、框内/框外选择和时间变化，不把不同 checkpoint 的差异解释为严格因果效应，也
不产生准确率或成本结论。若以后需要机制因果图，应另用同一公共 checkpoint 做反事实重放，而不
改写本主图的证据类型。

## 71. 两项尾部对照继续形成恢复链，R4 机制裁决保持冻结（2026-08-23）

02:12 CST 的一次只读核验确认，R4-SHUF15 job `1249131` 与 Q64-GLOBAL job `1249132`
仍为 `RUNNING`，分别新增 `recovery_epoch_49.pth` 与 `recovery_epoch_44.pth`；最近三个恢复点
更新为 epoch 39/44/49 与 epoch 34/39/44。本次没有新的正式 validation、终态或硬错误回执，
因此这些 checkpoint 只证明训练连续性。R4 的框外 free-token 归因继续冻结至 R4、R4-SHUF15
与 Q64-GLOBAL 的 final/final-EMA 完整配对终态；不据恢复点选模，也不启动成本、补臂或恢复。

## 72. Pro 冻结“保留 K64 上下文、只深刷新 K32”的下一路线（2026-08-23）

ZoomToken Project 的 fresh Pro 裁决为 `REVISE` 后继续。它拒绝把 learned task router、vanilla
MoD、ChronoTransport 与 cache 同时塞入首个实验，也拒绝直接把 K64 空间支持降成 K24/K18。
冻结方法 `RC32-KV` 保留 R1 严格矩形 K64；同一个确定性 K32 refresh mask 在 12 个
VideoMAE block 中只执行 query/output attention 与 MLP，另外 K32 继续作为 K/V。cache 仅取
前一 tubelet 同物理位置的 block-input activation，窗口内有效并截断梯度；每 block 只增加一个
标量混合参数。最小矩阵为 FULL64、DROP32、MOD32-KV、RC32-KV，后三臂共享同一 K32 mask。
只有 seed-42 的 final-EMA 准确率、边界和 selector-inclusive 实测成本通过，才开放额外 seed；
K24/K18 暂不实现。该裁决是设计冻结，不是实现、效率或性能证据。

## 73. RC32-KV 冻结实现通过独立审查并进入 seed-42 完整训练（2026-08-23）

在 Pro 冻结的四臂设计下，clean revision
`836f2ce4beafa8cbab513604dfa74be01a977a3c` 完成最小实现并推送到
`codex/zoomtoken-rc32-kv-v001`。实现保留 R1 K64 严格矩形支持；DROP32 只构造 K32，
MOD32-KV 与 RC32-KV 均保留 K64 K/V 上下文、只让确定性 K32 执行 query/output 与 MLP，
RC 再加入上一 tubelet 同物理位置、窗口内、detached block-input carry。目标 N16R4 环境的 8 项
focused Torch 测试通过。独立 Critic 在核对 Pro 对 `cache_valid` 的原定义后无条件 PASS：有效性
表示上一 tubelet 的 K64 同位置 lineage 存在，而非“上一帧是否属于 K32 refresh”；这保证未刷新
token 也能以浅路径连续传播，而不引入未来、标签、teacher 或 prediction 侧信道。结果盲 Evaluator
随后给出 `PRE_RUN_READY/PASS`。

FULL64 不重复训练，复用已完成的 R1 job `1249099` 及其 final-EMA。三个新增 seed-42、60-epoch
单元 DROP32/MOD32-KV/RC32-KV 已作为 jobs `1250604/1250605/1250606` 提交到共同根目录
`/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_refresh_rc32_836f2ce4_seed42_20260823T0355`。
这一状态为 `experiment_running`：尚无新增性能或成本结果，不开放 K24/K18、多 seed 或论文效率
主张；先等待冻结 final-EMA 与 selector-inclusive 实测成本。

## 74. RC32-KV 首个部署因 recovery route-schema 准入缺口终止（2026-08-23）

只读终态核验确认 jobs `1250604/1250605/1250606` 分别运行 34/34/30 秒后全部
`FAILED 1:0`。三个 rank-0/rank-1 traceback 完全同源：`tools/train.py` 在模型、数据和 checkpoint
访问前调用 `_zoomtoken_recovery_contract(cfg)`，该函数仍只接受历史冻结 route surface，因而对
DROP32/MOD32-KV/RC32-KV 的新 config schema 抛出
`ValueError: ZoomToken recovery is restricted to the frozen route surfaces`。该失败没有进入首个
训练 batch，没有 checkpoint、validation、成本或模型结果；不能解释为 RC32-KV、K32 refresh 或
carry 的效果。原 namespace 不恢复、不重排、不重复。若另行授权 distinct epoch，唯一合理前置是
对该入口 allowlist/validator 做最小 claim-preserving 修正，并用实际三 config 覆盖 recovery-contract
入口后再独立复核。既有 Q64-GLOBAL job `1249132` 保持 RUNNING，未被干预。

## 75. Q64-GLOBAL 继续训练并形成 epoch 54 可恢复点（2026-08-23）

单次只读核验确认 Q64-GLOBAL job `1249132` 继续 `RUNNING`，累计运行约 5 小时 25 分；新增
`recovery_epoch_49.pth` 与 `recovery_epoch_54.pth`，最近三个完整恢复点为 epoch 44/49/54。
当前没有正式 validation、终态、Traceback、显存溢出或非有限数值，也没有成本结果。该变化只证明
训练连续性，不用于 checkpoint 选择、性能比较或机制裁决。RC32-KV 三项训练入口失败终态保持不变。

## 76. Q64-GLOBAL 完成训练，机制结论等待终态指标摄取（2026-08-23）

只读 Slurm 核验确认 Q64-GLOBAL job `1249132` 于 04:56:25 以 `COMPLETED 0:0`
结束，累计运行 `06:06:57`，最近三个恢复点仍为 epoch 44/49/54。本次单次终态检查没有
取得正式 validation 行或结构化终结文件，因此只把状态从训练中升级为训练完成，不以恢复点、
作业退出码或缺失的终结文件推断准确率。R4 的框外 free-token 机制归因仍须摄取既有
R4-SHUF15 与 Q64-GLOBAL 冻结 final/final-EMA 数值后才可完成；本次不启动成本、补臂、
重复训练或新路线。RC32-KV 三项训练入口失败终态保持不变。

## 77. Q64-GLOBAL 终态准确率已摄取，R4 完整归因仍需配对终态（2026-08-23）

同一冻结日志在 `Training Over` 前给出 Q64-GLOBAL 的终态官方 validation：Avg-mAP、
mAP@0.6、mAP@0.7 为 `67.84/60.66/45.39`。与 R4 的 `68.02/60.32/46.26` 相比，
R4−Q64-GLOBAL 为 `+0.18/-0.34/+0.87` 个百分点。决定将其登记为真实单种子准确率证据：
它支持“连续矩形 core 可能保护高 tIoU”的有限观察，但三项指标并非同向，且 R4-SHUF15
冻结终态尚未完成摄取，因此不宣称框外 token 排序机制已被证明。保持不启动成本、多种子、
K24/K18、补臂或新路线；RC32-KV 首次部署的训练前失败分类不变。

## 78. 接受 RC32-KV recovery 入口的最小实现修复（2026-08-23）

决定接受 clean candidate `813012620dca991ff90121d0d9faf688f303d1ef`。该修复只把冻结设计
已有的 `R1-DROP32`、`R1-MOD32-KV`、`R1-RC32-KV` 加入训练入口的 recovery route 集合，并
增加实际三配置调用 `_zoomtoken_recovery_contract` 的正例，以及未知 route 继续失败的负例。
目标 N16R4 环境 `10 passed`，独立审查 PASS；原有 5-epoch 恢复、完整状态、canonical source
与 update-index 约束均保持不变。因此这是 claim-preserving 的入口闭合，不是模型、路由机制或
实验协议变化。旧 `836f2ce4…` namespace 保持终态且不恢复；本决定不授权新的 Slurm 作业，未来
distinct epoch 仍须以该 clean candidate 重新完成运行前核验后另行启动。

## 79. RC32-KV 修复版通过 PRE_RUN 并进入新的独立完整训练（2026-08-23）

以 `813012620dca991ff90121d0d9faf688f303d1ef` 为唯一执行 revision 完成结果盲 PRE_RUN。
实际三配置 recovery 测试、精确 clean source、canonical THUMOS14 411、官方 annotation/class map/
VideoMAE-S pretrain、空结果根、容量与三个 `sbatch --test-only` 均通过。决定执行原冻结四臂中的
三个新增臂：DROP32 `1252179`、MOD32-KV `1252180`、RC32-KV `1252181`；FULL64 仍只读复用
`1249099`。三项保持 seed 42、60 epochs、双卡 global/local batch `2/1`、相同增强/优化器/
scheduler/AMP/EMA/evaluator/NMS、每 5 epoch 完整恢复和 final-EMA 主结果。旧 `836f2ce4` namespace
仍封存，不 resume、不补臂。只读日志核验已确认三项均进入第 0 轮，旧 recovery 入口故障没有复现。
下一裁决事件仅来自首个 recovery、正式 validation、硬失败或终态；当前不启动成本、多 seed、
K24/K18，也不从运行状态推断模型结果。

## 80. RC32-KV recovery-49 中间验证只作过程诊断（2026-08-24）

只读核验确认 jobs `1252179/1252180/1252181` 均在第 51 轮运行，三臂各保存
`recovery_epoch_39/44/49.pth`，无 Traceback、OOM 或非有限损失。第 49 轮后官方 validation 的
Avg-mAP/mAP@0.6/mAP@0.7 分别为 DROP32 `65.48/57.13/43.50`、MOD32-KV
`65.50/57.65/43.90`、RC32-KV `64.20/56.21/42.18`。当前 RC32-KV 相对 MOD32-KV 的描述性
差值为 `-1.30/-1.44/-1.72` 个百分点，没有显示 temporal carry 增量；但冻结合同只允许
final-EMA 做正式裁决，因此决定不提前停止、不按中间 checkpoint 选模，也不开放成本、多 seed、
K24/K18 或新机制。等待三臂终态后再按 Pro 预注册门裁决。

## 81. RC32-KV 与 MOD32-KV 在 seed 42 终态准确率门失败，停止该深度稀疏路线（2026-08-24）

修复版 clean revision `813012620dca991ff90121d0d9faf688f303d1ef` 的 DROP32、MOD32-KV、
RC32-KV jobs `1252179/1252180/1252181` 均完成 60 轮、`COMPLETED 0:0`，每个 cell 均存在
`checkpoint/epoch_59.pth`，日志无 Traceback、OOM 或非有限数值。冻结 EMA 的
Avg-mAP/mAP@0.6/mAP@0.7 分别为 `66.11/57.83/44.88`、`66.50/59.24/45.21`、
`64.73/57.34/42.91`；只读 FULL64 为 `69.07/61.14/46.57`。

RC32-KV 相对 FULL64 为 `-4.34/-3.80/-3.66`，同时低于 DROP32
`-1.38/-0.49/-1.97` 和 MOD32-KV `-1.77/-1.90/-2.30`；明确违反 Pro 冻结的 D−A、D−B、
D−C 准确率门。MOD32-KV 相对 FULL64 也为 `-2.57/-1.90/-1.36`，不能作为 cache 删除后的
简化论文方法。决定按预注册停止分支关闭 RC32-KV/MOD32-KV 当前深度稀疏路线，不启动成本、
多 seed、K24/K18、learned gate、蒸馏或其他补救。理论重块 FLOPs 代理只作为失败点的位置描述，
不得宣称真实延迟或能耗收益。严格矩形 K64/R1 的准确率正证据保持有效；下一科学讨论应把
“保留 K64 支持的前提下如何避免固定 K32 深刷新造成的表征损失”作为新问题，而不是复活当前路线。

## 82. Pro 选择 DSR6-KV 作为唯一下一候选（2026-08-24）

fresh exact-Project Pro 在读取 `CURRENT_RESEARCH_STATE-v011` 与
`MODEL_EXPERIMENT_HISTORY-v006` 后裁决 `REVISE`。停止在全部 12 层使用 K32 query/MLP 的
MOD32-KV，也不再使用 RC32 的前 tubelet hidden carry。唯一下一候选为 `DSR6-KV`：严格
K64 空间支持和既有 refresh 排序保持不变，blocks 0–5 完整更新 K64，blocks 6–11 对同一个
K32 子集更新 query/output/MLP，同时保留全部 K64 为不 detach 的 K/V context，并让既有 Adapter
继续作用于 K64。该候选不增加参数、loss、浅层 transport、hidden cache 或逐层动态预算。

只授权一个 seed42、双卡、60-epoch THUMOS14 development cell；FULL64、DROP32、MOD32-KV
全部只读复用。final-EMA 的 Avg-mAP/mAP@0.6/mAP@0.7 必须同时达到
`68.57/60.64/46.07`，任一失败即停止整个 temporal/depth route，不追加切分点、种子、MoD、
K24/K18、teacher 或蒸馏，也不进入成本测量。准确率通过后才允许完整 decode-to-NMS 延迟、
能耗与显存核算。当前决定仅把路线推进到 `designed`；实现必须先经过 Builder、独立 Critic 和
结果盲 Evaluator/PRE_RUN。

## 83. DSR6-KV 最小实现与 PRE_RUN 通过，等待唯一训练授权（2026-08-24）

在 clean `4e940b…` 上形成单一后继
`3260cd39154069138c6b1757326372cc3b73754e`，并推送到
`codex/zoomtoken-dsr6-kv-v001`。实现没有新增网络模块或参数：前六层直接复用 FULL64 路径，
后六层直接复用 MOD32-KV 路径；同一个 K32 mask 在后六层复用，K64 K/V 不 detach，既有
Adapter 仍作用于全部 K64。配置、训练 allowlist、现有双卡 launcher、5-epoch recovery 与执行
账本同步更新。静态/编译/Shell 与小型无数据 Torch 前后向通过；本机完整 pytest 因 DLL/
OpenMMLab 环境组合无法收集，未伪报为 PASS。独立只读 Critic 给出 `AUDIT_PASS`。

结果盲 PRE_RUN 随后确认：GitHub ref 精确解析该 SHA；canonical THUMOS14 为 411 个 MP4、0
断链；注释、类别映射、VideoMAE-S 预训练与 OpenTAD 环境存在；拟定唯一 run root 不存在；
`/data` 容量充足。决定将候选推进到 `PRE_RUN_READY`，但本轮没有 GPU/Slurm/训练授权，故
仍为 0 个新 job、0 个 DSR6 性能或成本结果。下一动作仅是请求一次明确授权，之后原子部署
clean source 并提交一个 seed42、双卡、60-epoch cell；其余臂全部只读复用。

## 84. 首次 DSR6-KV 单次调度在 sbatch 前停止（2026-08-24）

中央授权了 exact SHA `3260cd39154069138c6b1757326372cc3b73754e` 的唯一 seed42、双卡、
60-epoch DSR6-KV 单元。提交前只读门确认 GitHub ref、canonical THUMOS14 411/0、依赖文件、
空 source/run root、无同名作业与容量均满足。随后通过规定学术代理建立 immutable source
`/data/run01/sczc063/yuzibo/projects/zoomtoken_dsr6_src_3260cd39`，并再次确认
HEAD、remote-tracking ref 与 clean 状态一致。

目标环境 focused 检查在正式 `sbatch` 前因非登录 SSH shell 尚未 source `/etc/profile`、
`module` 命令不可见而退出。按单次调度的 fail-closed 条款没有改命令后重试：
`actual_attempt_count=0`，结果根不存在，Slurm/sacct 中无同名作业。该事件是提交前环境初始化
顺序问题，不是 DSR6 模型、训练或准确率证据。恢复条件是新的单次调度明确允许先 source
`/etc/profile`，再加载 CUDA/miniforge、运行同一 focused precheck 与 `sbatch --test-only`，最后
只提交同一个冻结单元；不得改变 SHA、config、seed、数据、资源或准确率门。

机械重调度 `CENTRAL-RUN-ZOOMTOKEN-DSR6-KV-SEED42-v002` 严格执行上述唯一修正，但
`source /etc/profile` 在当前 `set -u` shell 中进入 `/etc/profile.d/apps-bin-path.sh` 时引用未定义的
`XDG_DATA_DIRS`，再次在 focused test 与 `sbatch --test-only` 前退出。复核仍为 source exact/clean、
result root absent、job absent、`actual_attempt_count=0`。下一机械恢复只能在 source profile 的局部
范围暂时关闭 nounset，完成后立即恢复 `set -u`；不得借此改变训练入口或科学合同。

## 85. DSR6-KV 唯一 seed42 完整训练已提交（2026-08-24）

最终机械调度 `CENTRAL-RUN-ZOOMTOKEN-DSR6-KV-SEED42-v003` 仅在 source `/etc/profile`
期间执行 `set +u`，随后恢复 `set -u`。目标 N16R4 环境 focused suite 为 `11 passed in 41.46s`，
launcher `bash -n` 与 `sbatch --test-only` 通过；checkout 在测试后仍为 exact SHA/ref 且 clean。
随后仅提交一次正式 job `1252521`，初始状态 `PENDING`，申请 1 node、2 GPUs、8 CPUs、8 hours，
未指定物理 GPU 或覆盖 `CUDA_VISIBLE_DEVICES`。

运行固定 config `georoute_official_r1_dsr6_kv_prebackbone_seed42_v001.py`、seed42、global/local
batch `2/1`、60 epochs、canonical THUMOS14 training→validation。每 5 epoch 保存 full-state
recovery、保留 latest3+final，epoch59 EMA 为唯一主结果。当前是 `experiment_running` 启动证据，
不是准确率或成本结果；只监控 job `1252521`。final-EMA 三项门任一失败即
`STOP_DEPTH_ROUTE`，不追加成本、seed 或结构补救。

## 86. DSR6-KV job 1252521 在模型前因 launcher profile 初始化失败（2026-08-24）

job `1252521` 终态为 `FAILED 1:0`，elapsed `00:00:02`。stdout 为空，stderr 仅为
`/etc/profile.d/Z97-byobu.sh: line 24: LC_BYOBU: unbound variable`；目标 cell 目录不存在。
因此未发生数据读取、模型构建、checkpoint、validation、准确率或成本计算。

根因是训练 launcher 自身在 Slurm 非登录 shell 中启用 `set -u` 后 source `/etc/profile`；
提交端的局部 `set +u` 只保护 precheck shell，不能传播进 job 的新 shell。决定把 `1252521`
封存为 pre-data infrastructure failure，禁止 resume/requeue/in-place retry。下一步不是科学改线，
而是对现有 launcher 做最小 claim-preserving 修正：只在 source `/etc/profile` 的局部关闭
nounset、之后立即恢复，并加入可区分测试；经独立 Critic 与结果盲 PRE_RUN 后才可请求 distinct
epoch。准确率门与 DSR6 科学合同完全不变。

## 87. launcher-only candidate 通过目标测试但独立 Critic 要求显式回归收束（2026-08-24）

clean/pushed candidate `4eb40fe3eb67ea3511a16d26e38d6bdca3ca5c93`（父
`3260cd39154069138c6b1757326372cc3b73754e`）仅修改 launcher 和 focused test。launcher 在
`source /etc/profile` 前后加入 `set +u`/`set -u`；N16R4 exact checkout 的 focused suite
`12 passed in 39.91s`，并保持 clean。没有模型、config、数据、seed、资源或阈值变化。

fresh 独立 Critic 确认 launcher 修正与两文件范围正确、source failure 仍 fail-closed、无 science
drift，但对 regression 返回 `NEEDS_ATTENTION`：嵌入式 shell probe 在 nounset 验证 `if` 后没有
显式 `exit 0`，Critic 认为成功状态不够明确。虽然目标 N16R4 的真实 pytest 与独立 shell probe
均返回 PASS，按独立门规则仍不由主代理覆盖该结论；Evaluator/PRE_RUN 未启动。下一步等待
Coordinator 对“只增加显式成功退出”的 focused test-only 修正给出范围处置。

## 88. DSR6 launcher 聚焦修正通过双重门并恢复 PRE_RUN_READY（2026-08-24）

中央允许在 `4eb40fe3…` 上仅修回归 probe 的成功终止。clean/pushed successor
`c6327a891809aa30370b3b2d9bedab0dcfe0d326` 只在
`tests/test_zoomtoken_r1_refresh_carry_k32.py` 增加一行显式 `exit 0`；launcher、模型、config、
数据、seed、资源、恢复、EMA 和阈值均未改变。N16R4 同一 focused suite 为
`12 passed in 40.20s`，fresh 独立 Critic 返回 `AUDIT_PASS`。

fresh 结果盲 Evaluator 随后以真实双卡 Slurm job-shell witness `1252525` 执行 exact launcher
初始化路径；job `COMPLETED 0:0`，并证明 `/etc/profile` 在局部关闭 nounset 后可加载且 nounset
立即恢复。该 witness 没有训练数据、模型、checkpoint、validation 或 official-test 访问。
canonical 411/0、依赖与 exact clean ref 均通过，拟定 distinct run root 和同名 job 均不存在，故
状态恢复为 `PRE_RUN_READY`。当前仍无正式 c632 训练 job；下一动作只允许请求一个新的 exact
seed42/双卡/60-epoch 单元，旧 `1252521` 与旧根永久封存。

## 89. DSR6-KV distinct seed42 完整训练已提交（2026-08-24）

首次 formal dispatch 在 `sbatch` 前发现 PRE_RUN witness `1252525` 使用了原拟定 job name，故以
actual attempt count 0 停止；结果根仍不存在。中央随后只替换正式 job name 为
`zt-dsr6-train-s42-c6327a89`，其余 SHA、config、数据、seed、资源、root、恢复、EMA 和准确率门
全部不变。

最小复核确认 source HEAD/ref 均为 clean
`c6327a891809aa30370b3b2d9bedab0dcfe0d326`，新 exact name 与 distinct root 不存在。随后唯一
正式 job `1252527` 于 `2026-08-24T04:46:31+08:00` 提交，初始 `PENDING`，申请 1 node、2 GPUs、
8 CPUs、8 hours。该事件只证明 frozen DSR6-KV seed42/60-epoch cell 已入队；尚无训练进度、
准确率或成本结果。接下来只做事件驱动只读监控，epoch59 EMA 三项门任一失败即停止深度路线。

## 90. DSR6-KV 正式单元进入真实训练（2026-08-24）

job `1252527` 于 `04:46:36 CST` 从排队转为 `RUNNING`，分配节点 `g0041`。启动日志确认 CUDA 与
Miniforge 模块加载成功、双进程 torchrun 初始化完成，随后完成模型参数登记并打印
`Training Starts`、`Epoch 0 started`。当前没有 Traceback、OOM 或非有限 loss，也尚未形成首个
5-epoch recovery。该节点只证明真实模型/数据训练链已进入 epoch 0，不提供准确率、效率或论文
结论；继续按原合同只读等待恢复点、硬故障或 epoch59 EMA 终态。

## 91. DSR6-KV 首个五轮完整恢复点形成（2026-08-24）

job `1252527` 在 `g0041` 保持 `RUNNING`。训练日志依次开始 epoch 0–5，并于
`2026-08-24 05:06:03 CST` 发布
`recovery_epoch_4.pth`（`627,950,731` bytes），证明冻结的每 5 epoch full-state recovery 路径
已在真实训练中首次落盘。同期日志没有 Traceback、CUDA OOM 或非有限 loss。

该恢复点仅用于中断恢复和运行审计，不参与 checkpoint 选择，也不是中间性能、最终准确率、
真实延迟或能耗证据。维持 epoch59 final EMA 为唯一主结果；不启动第二单元、成本或额外 seed，
继续事件驱动等待后续恢复点、硬故障或终态。

## 92. DSR6-KV 五轮恢复节奏得到第二次运行确认（2026-08-24）

job `1252527` 在 `05:24:54 CST` 开始 epoch 10，并发布第二份完整恢复点
`recovery_epoch_9.pth`（`627,950,731` bytes）；先前的 `recovery_epoch_4.pth` 仍存在。
调度状态保持 `RUNNING`，日志没有 Traceback、CUDA OOM 或非有限 loss。

该事件确认恢复机制按冻结的五轮间隔持续工作，但仍不读取或解释中间 checkpoint 性能。维持
epoch59 final EMA 唯一主结果及三项准确率门；不增加作业、种子、成本或结构变体。

## 93. DSR6-KV 已形成三份有效恢复点（2026-08-24）

job `1252527` 于 `05:43:47 CST` 发布 `recovery_epoch_14.pth`（`627,952,846` bytes），随后进入
epoch 15。此前的 epoch 4 和 epoch 9 恢复点仍存在，因此当前恰有三份有效恢复文件；日志没有
Traceback、CUDA OOM 或非有限 loss。

这一状态验证“保留最近三份”的容量上限已经达到，但尚未验证第四次发布时的淘汰行为。继续只读
监控下一恢复事件；恢复文件仍不用于选模或性能解释，最终准入只看 epoch59 EMA。

## 94. DSR6-KV latest-3 恢复轮换在真实训练中闭合（2026-08-24）

job `1252527` 发布第四个恢复点 `recovery_epoch_19.pth` 后进入 epoch 20。checkpoint 目录恰好保留
epochs `9/14/19`，最旧 epoch 4 文件已被移除；因此“每 5 epoch 保存、仅保留最近三份”的恢复
与轮换合同均得到运行验证。日志仍无 Traceback、CUDA OOM 或非有限 loss。

该验证仅关闭恢复机制的工程问题，不形成任何模型质量或效率结论。后续不再因常规恢复轮换改变
科学处置；除非出现硬故障或终态，继续等待 epoch59 final EMA。

后续运行证据与上述决定一致：`recovery_epoch_24.pth` 发布后，job 进入 epoch 25，最近三份恢复点
轮换为 epochs `14/19/24`，且没有硬故障。该常规事件不产生新的科学决定。

`recovery_epoch_29.pth` 发布后，job 进一步进入 epoch 30，最近三份恢复点轮换为 epochs
`19/24/29`，仍无硬故障；这同样不产生新的科学决定。

`recovery_epoch_34.pth` 发布后，job 进入 epoch 35，最近三份恢复点为 epochs `24/29/34`，
仍无硬故障；科学处置和 final-EMA 判定保持不变。

`recovery_epoch_39.pth` 发布后，job 进入 epoch 40，最近三份恢复点为 epochs `29/34/39`，
仍无硬故障；科学处置、成本关闭和 final-EMA 判定继续保持不变。

`recovery_epoch_44.pth` 发布后，job 进入 epoch 45，最近三份恢复点为 epochs `34/39/44`，
仍无硬故障；该常规恢复事件不产生新的科学决定，成本继续关闭并等待 epoch59 final-EMA。

## 95. 严格 A-MoD 参照通过审查，帧间特征保存主线继续（2026-08-24）

决定把“逐层计算分配”和“跨帧特征复用”拆成两个可独立验证的问题。严格 A-MoD 参照已在
clean revision `a41714e9f9271906a2eb4505e3fedc590c838055` 完成：VideoMAE 的偶数块完整更新，
奇数块使用紧邻前一完整块的注意力概率列均值选择 400/800 token，只更新选中 token 的
Attention 与 MLP；未选 token 恒等旁路，现有 Adapter 仍处理全部 token。N16R4 无数据测试
`8 passed`，独立 Critic 为 `AUDIT_PASS`。该决定只确认参照实现正确，不构成性能或成本结论。

用户明确要求帧时序特征保存和映射不能中断，因此跨帧方向保持 active。旧 RC32 的同位置
detached carry 负结果只否定该具体实现，不否定对齐良好的状态复用。下一项科学动作是通过新的
Project Pro 讨论冻结：保存层级、空间/时间对应、当前帧变化评分、刷新与失效/场景切换、梯度边界、
无未来帧泄漏和最小对照矩阵。冻结前不把临时 cache 写进 A-MoD，也不提交新的正式训练。
现有 DSR6 job `1252527` 独立继续，不能被取消、复制或解释为帧间复用证据。

后续只读监控显示 `recovery_epoch_49.pth` 发布后，job 已进入 epoch 51，最近三份恢复点为
epochs `39/44/49`。日志仍无 Traceback、CUDA OOM 或非有限 loss；过程性 validation 只用于运行
诊断，不用于选择 checkpoint 或改变三项终态门。

严格 A-MoD 的两项非阻塞测试缺口随后在 test-only successor
`31e4b1e61a23c4f1b319249684c8f05da6734235` 闭合：逐块 marker 测试证明每个 A-MoD block 只接收
紧邻前一 Dense block 的 score；官方 16-frame/160×160 几何的 hook 测试证明 12 个 Adapter 均接收
完整 `[1,800,C]` token grid。N16R4 CPU-only suite 为 `10 passed`。模型、配置、launcher 与科学语义
均未改变，因此无需重开科学裁决或把后继视为新方法。

## 96. Pro 冻结 APM32-CTX64，最小实现通过代码审查但 PRE_RUN 缺两项机械证据（2026-08-24）

新的 Project Pro 裁决为 `REVISE / APM32-CTX64`。方法固定使用严格 R1 K64 空间支持，仅保存前一
tubelet 的 detached、未加位置编码 patch embedding；在半径 2 的局部邻域内，以 FP32 标准化相似度
做确定性 mutual-nearest 对齐，阈值为 `0.80`。有效匹配不少于 32 时保留最高相似度 K32，并对其余
K32 做当前 tubelet 的完整重计算；所有 K64 仍作 K/V 上下文。匹配不足、首 tubelet 或非法数值时
精确回退 K64。matched control `CUR32-CTX64` 共享同一 mask/fallback，只把保留载体换成当前 embedding。
该路线没有新参数、loss、隐藏态/KV cache，也不与严格 A-MoD、DSR6 或 ChronoTransport 合并。

实现从 `31e4b1e6…` 起步；初始七文件候选为 `435ab8dd…`。fresh Critic 发现批内不同样本可能因
匹配质量不同而产生不同 K64 fallback 总数，但旧执行入口仍要求批内刷新总数相等。聚焦后继
`d985dfb8b0cba4f70c28770643145ee44cb451d2` 删除该过时假设，按每行真实 Query 数分桶，并增加
`64/96` 不同刷新总数的批内 known-answer 测试。N16R4 标准 OpenTAD 环境 `13 passed`，fresh 独立
Critic 为 `AUDIT_PASS`。

fresh 结果盲 Evaluator 返回 `PRE_RUN_NOT_READY`，单一准入缺口是尚未形成两项绑定该 SHA 的
结果盲运行收据：一次有限 loss、精确账本且不做指标评测的 Fit/train 单批前向/反向；以及一次覆盖
model/optimizer/scheduler/scaler/EMA/RNG 的完整 save/resume fixture，并证明 temporal memory 不进入
checkpoint。当前不提交训练、不读取 official test，也不把 Pro/测试/审查写成准确率或效率证据。

## 97. APM/CUR 生产单批与完整恢复入口已实现，剩余双卡见证（2026-08-24）

对 Evaluator 指出的准入缺口做执行面审计后确认：`d985dfb8…` 的 APM/CUR 配置虽然继承恢复
合同，但 `tools/train.py` 的恢复臂白名单尚未包含二者，因此完整状态恢复在生产入口不可达。该
问题属于结果盲执行闭合，不改变模型机制、数据、优化、评测或停止规则。

clean/pushed successor `e92df6a4737a10955722c6aedc2f079e0d285a18`（父提交 `d985dfb8…`）
把 APM/CUR 纳入既有恢复合同，并增加仅限这两个臂的单批 preflight：复用 production train loader、
loss、optimizer、scheduler、AMP、EMA 和 checkpoint 路径，消费恰好一个训练 batch；验证有限 loss、
原生 ragged 账本、model/EMA/optimizer/scheduler/scaler/训练计数/sampler/RNG 的完整保存—恢复，并
拒绝序列化 temporal memory。该入口不构建 validation/test loader、不评测指标、不接受 resume。

N16R4 CPU-only focused suite 为 `19 passed`，fresh 独立 Critic 为 `AUDIT_PASS`。fresh 结果盲
Evaluator 仍返回 `PRE_RUN_NOT_READY`，但唯一剩余项已经收敛为：在实际双卡 Slurm allocation 中
分别执行 APM 与 CUR 的一个生产 batch，并保存 metric-free 机械回执。该见证未获资源授权，故当前
没有提交训练或见证；DSR6 job `1252527` 未被读取、取消、恢复或重新解释。

## 98. DSR6-KV 发布 epoch-59 checkpoint，保持终态结果盲（2026-08-24）

唯一正式 DSR6-KV job `1252527` 已发布五轮恢复点 `recovery_epoch_54.pth` 并按 latest-three
保留 `44/49/54`，随后发布预注册 `epoch_59.pth`。监控时作业仍为 `RUNNING`，final validation
尚未完成，且未见 Traceback、OOM 或 non-finite loss。因此不提前应用准确率门、不读取 live metric、
不据中间/刚发布 checkpoint 选模。只有 scheduler 终态与 epoch-59 final-EMA 官方 validation 同时
存在后，才按 `68.57/60.64/46.07` 三项 all-of 门裁决；此前成本和追加实验保持关闭。

## 99. DSR6-KV 训练执行完成，但终态准确率收据缺失（2026-08-24）

job `1252527` 于 `10:53:53 CST` 在 `g0041` 以 `COMPLETED 0:0` 终止，完整运行
`06:07:17`。预注册 `epoch_59.pth` 与最近三份恢复点 `44/49/54` 均存在；日志硬故障扫描未见
Traceback、CUDA OOM 或非有限 loss。这证明唯一 seed-42、60-epoch DSR6-KV 单元完成了训练执行。

然而精确结果根没有独立 result/finalization/metric JSON，本次有界终态快照也没有取得可审计的
final-EMA Avg-mAP/mAP@0.6/mAP@0.7 三元组。因此当前既不能判定三项门 PASS，也不能在缺少数值
时触发 `STOP_DEPTH_ROUTE`。成本、额外 seed 与结构补救继续关闭；下一动作仅是对不可变终态
stdout/checkpoint 做一次另行授权的只读结果摄取，不重训、不恢复、不补臂。

## 100. DSR6-KV final-EMA 三项门失败，停止深度稀疏路线（2026-08-24）

对 job `1252527` 的不可变终态 stdout 进行只读摄取后，确认 epoch 59 训练结束后的最终官方
validation 使用 `model_ema`：Avg-mAP/mAP@0.6/mAP@0.7 为 `67.38/59.34/46.01`。相对预注册
all-of 阈值 `68.57/60.64/46.07` 分别低 `1.19/1.30/0.06` 个百分点，三项均失败；相对只读
FULL64 `69.07/61.14/46.57` 分别低 `1.69/1.80/0.56` 点。

因此裁决为 `STOP_DEPTH_ROUTE`。不运行 selector-inclusive cost，不增加 seed、K24/K18、第二个
切分点或浅层补救，也不把 `79.055%` 理论 block-FLOPs 代理写成速度或能耗结果。该作业是完整、
有效的模型级负结果，不是 launcher、数据或训练基础设施失败。

## 101. 将帧间载体与 K32 深度裁剪解耦，执行 R1-APM-C32/FULL64（2026-08-25）

完整矩阵科学复核认为，旧 `APM32-CTX64` 同时引入帧间表征替换与 K32 深度更新，无法清楚判断
收益或损失来自哪一项。决定先执行性能优先的单变量实验 `R1-APM-C32/FULL64`：保留既有严格
8×8 K64 支持、前一 tubelet detached memory、半径 2 双向一致匹配、阈值 0.80、clip reset 与
K64 fallback；仅让 32 个可靠匹配位置使用“前一表征 + 当前残差”的输入载体，随后全部 64 个
token 在 12 个 VideoMAE block 和 Adapter 中完整更新。该决定不恢复已停止的 K32 深度路线，
也不产生计算节省主张。

clean revision `bffff43dad28ca1042602ad3a01ba2990b953c13` 在 N16R4 focused suite 达到
`22 passed`，fresh 独立 Critic 为 `AUDIT_PASS`，结果盲 Evaluator 为 `PRE_RUN_READY`。唯一
seed-42、双卡、60-epoch THUMOS14 training→validation job `1254008` 已提交，结果根为
`/data/run01/sczc063/yuzibo/projects/zoomtoken_apm_full64_bffff43d_seed42_20260825`。只使用
epoch-59 final EMA；在终态前不提升中间指标。若不能同时达到 `68.73/61.58/47.24` 且至少一项
严格改善，则停止该载体，不启动成本或额外种子。

## 102. APM-C32/FULL64 三项门失败，停止当前帧间载体（2026-08-25）

job `1254008` 已 `COMPLETED 0:0`。预注册 epoch-59 final EMA 的完整 mAP@0.3/0.4/0.5/0.6/0.7
为 `83.74/78.93/72.41/60.43/45.60`，Avg-mAP 为 `68.22`。相对冻结门
`68.73/61.58/47.24`，Avg-mAP/mAP@0.6/mAP@0.7 分别低 `0.51/1.15/1.64` 个百分点，且没有
任何一项严格改善。

因此裁决为 `STOP_APM_MEMORY`：不启动成本、额外 seed 或结构补救。该结论否定的是“当前
one-tubelet detached matching carrier 在全部 K64、全部 12 层重计算时可保持或提高性能”这一
具体假设。它不构成计算节省结果，也不能外推为所有时序特征复用方法无效；严格 A-MoD-50 仍是
独立且尚未训练的深度分配参照。

## 103. R4-SHUF15 终态未通过框外内容排序门（2026-08-25）

对遗漏的 job `1249131` 终态进行只读摄取后，确认其 60-epoch epoch-59 final EMA 的
Avg-mAP/mAP@0.6/mAP@0.7 为 `67.19/60.17/46.20`。相对同配方 R4
`68.02/60.32/46.26`，R4−R4-SHUF15 为 `+0.83/+0.15/+0.06`。预注册要求
R4−R4-SHUF15 的 mAP@0.7 至少 `+0.30`，实际仅 `+0.06`，因此框外 learned token ranking
没有通过因果门。

Q64-GLOBAL 为 `67.84/60.66/45.39`，R4−Q64-GLOBAL 是
`+0.18/-0.34/+0.87` 的交叉结果；它说明连续 core 对高 tIoU 可能有利，但不能替代
R4-SHUF15 的内容排序对照。决定不为 R4 框外排序启动成本、多 seed 或重复训练；R1 K64 仍是
现有单种子准确率最强的主干前空间路线。

## 104. 将深度稀疏裁决从单一准确率门修订为准确率—计算 Pareto 裁决（2026-08-25）

用户明确要求：若计算量显著下降，小幅准确率损失可以接受，不能仅凭近无损准确率门全面停止
效率候选。原预注册门的历史事实不修改：DSR6-KV、MOD32-KV、DROP32 都未达到“近乎无损”
标准；但科学结论修订为分层 Pareto 判断。相对 FULL64，DSR6-KV 的 VideoMAE 重块算量代理
减少 `20.94%`，Avg-mAP/mAP@0.7 下降 `1.69/0.56`；MOD32-KV 减少 `41.89%`，下降
`2.57/1.36`；DROP32 减少 `50.68%`，下降 `2.96/1.69`。三者分别保留为保守、中等与激进
预算候选。

RC32-KV 与 MOD32-KV 的代理成本同为 `58.11%`，但三项准确率均更低，因此仍被严格支配并停止。
APM-C32/FULL64 没有减少主干重块计算且低于 matched R1/FULL64，因此仍停止。下一项最小且能改变
结论的证据不是重训或增加结构，而是复用现有 final checkpoints，在同一硬件、同一完整 validation
population 上测量 FULL64/DSR6/MOD32/DROP32 的 decode→NMS p50/p95、吞吐、峰值显存和能耗。
在该测量前，现有百分比只能称 block-FLOPs proxy，不能称真实加速。

## 105. Pro 提议 ACR16，但因遗漏 Eventful Transformers 暂不进入实现（2026-08-25）

全新 ZoomToken Project Pro 会话完成视觉、视频、扩散模型和 LLM 的 token 迁移/变换讨论，终态为
`PIVOT`，提议 `R1-ACR16-Δ1-FKV`：blocks 0–3 密集；blocks 4–9 对每个非首 tubelet 最多 16 个
稳定 token 仅做一次低秩当前差分更新并旁路 Attention+MLP，其余 token 刷新；全部当前 K64 仍生成
K/V；blocks 10–11 重新密集。该设计避免旧 hidden/KV 复用，保持当前 K64 身份与 AdaTAD 接口。

独立一手来源核验发现报告遗漏了最近邻 Eventful Transformers（ICCV 2023）。该工作已经维护 token
reference/buffer、选择时间上变化的 token，并对 token-wise、QK 与 attention-value 运算做稀疏或
增量更新。因此不接受“变化 token 深算、稳定 token 复用”本身的新颖性，也不按 Pro 的
`next_owner=Builder` 自动实施。当前裁决为 `DISCUSSED / NEEDS_PRIOR_ART_CORRECTION`：先完成
Eventful-style 直接基线差异与保守全栈节省上界，再决定 ACR16 是否具有足够差异和实际收益空间。
本轮未修改代码、未提交训练，也没有新增准确率或效率证据。

## 106. ACR16/Eventful 直接迁移在实现前停止（2026-08-25）

在把 Eventful Transformers 正式论文、作者实现和保守全栈算量作为强制输入后，fresh ZoomToken
Project Pro 单轮复核返回 `STOP`。逐算子核验确认，ACR16 不复用旧 hidden、Q/K/V、attention
乘积或 MLP 输出；它从当前 K64 重算全部 K/V，让稳定 token 旁路中间深度残差分支，并额外加入
一次低秩输入差分残差。因此其剩余贡献是 Eventful 式变化证据与条件深度跳过的应用组合，而不是
新的时序复用原理。

独立重算确认，ACR16 对 12 个 VideoMAE 主块的理论节省上限为 `9.446%`；计入全 K/V、全
Adapter、patch embedding、匹配和已知必要计算后，骨干算术上限约为 `8.80%`。这一余量不足以
可信地支持完整 `decode→Soft-NMS` 链路的 p50 延迟和 gross energy 同时改善至少 `5%`。据此将
该路线记为 `STOP_BEFORE_IMPLEMENTATION`：新增训练单元为 0，不派 Builder，不进入 PRE_RUN，
不启动 ID/Delta/SHUF 或 Eventful 直接迁移对照。本裁决只停止 ACR16/Eventful-transfer，不否定
全部时序冗余研究；现有正式性能与成本证据不因本次讨论发生变化。

## 107. 三类动态选择性重计算候选在实现前停止（2026-08-25）

按用户授权在 exact ZoomToken Project 发起一轮新的附件式单轮 Pro 讨论，固定
`bffff43dad28ca1042602ad3a01ba2990b953c13` 与 A-MoD 参照
`a41714e9f9271906a2eb4505e3fedc590c838055`。Pro 比较 clip 内动态重计算与
rank-32 当前差分更新 `IC-DRU`、重叠滑窗精确依赖锥缓存 `OW-ECR`、current proxy 驱动的
嵌套深度更新 `PCD-DRU`，终态返回 `STOP_BEFORE_IMPLEMENTATION`。

理想、偏向候选的已知骨干算术上界为 IC-DRU 约节省 `50.12%`、PCD-DRU 约节省
`60.16%`，但二者要求平均刷新率约 25%，首 tubelet full refresh 后其余 tubelet 仅约
14.29%，缺少高 tIoU 准确率先验；变化证据、稳定 token 轻量残差与逐 token 深度分配也可被
Eventful、视频缓存、ResidualViT 与 MoD/A-MoD 等已有工作分解覆盖。OW-ECR 因 Adapter
依赖传播，理想已知骨干节省仅约 `7.66%`，且十二层 overlap hidden 约增加 `54 MiB/样本`。

项目接受这一 STOP 对三个精确定义候选的约束：不创建 Builder、PRE_RUN 或新训练单元，不通过
seed、K、rank、threshold、额外 gate、teacher 或蒸馏营救。但不把它外推为动态 20%–30%
刷新或全部时序复用的经验失败；三类机制都没有训练。本轮新增准确率与成本证据均为 0。任何重开
必须先提出超出现有 cache/change-routing/depth-routing 组件组合的独立误差控制或执行原理，并
给出保持双向 VideoMAE 语义的合法状态合同与保守全链路收益余量。

## 108. 用户提供 R-PADT-v0 完整报告；部分吸收但原样路线需修订（2026-08-26）

用户补充提供了此前浏览器会话未完整摄取的 R-PADT-v0 下载报告与粘贴文本。两份材料均提出：
在两个稠密 VideoMAE 前缀块后，以周期 K64 锚点、非锚点同格点 delta top-16 和四个摘要
压缩后缀序列，再在 neck 前将未保留位置复制为锚点同格点输出。其 Cell 0/1/2、clip-local
状态、同预算压缩反事实和完整链路成本思想具有可证伪性，作为用户提供的候选设计予以记录。

但项目终态回执仍为 `TERMINAL_INCOMPLETE_NO_SCIENTIFIC_DECISION`，当前无法证明下载报告是同一
completed Pro 会话的权威完整输出；报告与粘贴文本对 nonce 的陈述也矛盾。独立代码核验确认
T=384、K64、D=384、12 blocks 与稳定网格 lineage，同时确认每层仍有要求完整时空格点的
Adapter，报告的短序列示意没有闭合该成本。R1 还会按 tubelet 独立地从九个合法 8×8 框中选
一个，因此相邻 K64 集合不保证相同，报告要求的全 K64 锚点一一映射可能在第零门失败。科学核验
还发现：该机制是后缀深度压缩而非前一帧完整表征复用；直接复制锚点后缀输出存在当前时间身份
陈旧风险；Q=4 摘要混入第二项机制；仅匹配 N' 不能形成严格公平的 ToMe 对照。

相关工作复核补入被报告遗漏的 STA 与 PVC，并确认粘贴版引用存在多处错配。当前可守新颖性只到
“离线 TAD 的严格 ROI 来源映射、VideoMAE 后缀压缩、检测前 T×K 恢复与定位敏感成本门”，不构成
新的时序复用原理。项目裁决为 `PARTIAL_ACCEPT_REVISE_BEFORE_G4 / discussed`：不接受外部
`CONTINUE` 直接升级路线，不冻结 `L_p=2/R=4/m=16/Q=4`，不派 Builder、不进入 PRE_RUN、
不启动实验。完整摄取与核验见
`research-wiki/sources/2026-08-26-r-padt-v0-user-report-intake-audit.md`。

## 109. 当前官方结构下的完整表征时序复用在实现前停止（2026-08-26）

在用户完成 profile 61 登录后，项目以两个附件在精确 ZoomToken Project 中发起一次全新、单轮
Pro 科学裁决，固定 revision `bffff43dad28ca1042602ad3a01ba2990b953c13`。会话正常
`completed`，顶层结论为 `STOP`，适用范围仅为“主干前一次确定 stable/changed，changed token
完整执行 12 层，stable token 复用上一时间单位完整表征”的精确定义路线。

裁决的核心不是 MoD 深度路由，而是状态依赖：16 帧 clip 是双向 attention bucket，却不是完整
backbone state；每个 block 后的 Adapter 沿全局 384-tubelet lineage 传播。只缓存 block-11 输出
无法为 changed token 提供逐层 K/V 和 residual 上下文；逐层缓存虽能写出近似合同，但固定 mask
不对 attention 与 Adapter 的失效传播闭合，且操作族与 Eventful Transformers、STC-Cacher 高度
重叠。按 clip 顺序执行还可能损失 48-clip attention batching 的系统优势。

项目独立核验基本接受该结构判断，同时保留两项限定：完整 backbone 本就含跨 clip Adapter，故
顺序化的真实延迟损失必须实测；与已有工作操作重合是新颖性警告，不能单独证明任何 TAD 应用研究
均无价值。结合当前准确率负证据和缺少可信全栈收益下界，停止本精确路线仍是合理决定。

本轮没有代码、配置、Builder 实现、PRE_RUN、GPU/Slurm 或训练单元，也没有新性能或成本证据。
完整 Pro 报告保存于
`.cvpr-pro-lab/reviews/PRO_FULL_REPRESENTATION_TEMPORAL_REUSE_ARCHITECTURE_ADJUDICATION_RESPONSE-v001.md`。
该停止不影响独立 A-MoD，也不停止整个 ZoomToken；下一项现成、论文相关的证据仍是对 FULL64、
DSR6-KV、MOD32-KV、DROP32 已有 final checkpoint 做同硬件完整端到端成本测量。

## 110. 从历史表征复用转向当前原生支持 BPNS-R1（2026-08-26）

按用户要求，项目在同一 exact ZoomToken Project 中以三份附件和三份长期 Source 发起全新单轮
Pro 前进路线裁决。初始本地捕获器在回答完成前超时，随后只重新连接同一 conversation，未重投、
未 follow-up；终态为 `completed / PIVOT`。完整报告选择
`ZoomToken-BPNS-R1`：每个当前 tubelet 在 `10x10` 原生网格上保留一个连续无孔洞的 `8x8/K64`
支持，全部 K64 继续执行 12 层 VideoMAE-S 和既有 Adapter，不保存或复用历史 hidden/KV/cache。

这一转向由现有真实结果支撑：同源 K100 final-EMA 为 `68.51/61.19/46.27`，R1 为
`69.07/61.14/46.57`，即 R1 少输入 36% 原生空间 token，但 Avg-mAP/mAP@0.7 分别高
`0.56/0.30` 点，mAP@0.6 仅低 `0.05` 点。该结果支持继续检查严格原生支持，不等于已经证明
加速、显存或能耗收益。

项目接受科学 PIVOT，并将新增 60-epoch 单元冻结为 0。唯一优先动作是用既有 K100 job `1248835`
与 R1 job `1249099` 的 final-EMA 在同一硬件完成数值重放和完整 decode-to-NMS p50/p95、峰值显存、
gross energy、短动作与边界误差测量；DSR6-KV、MOD32-KV、DROP32 只作为随后复用同一工具的辅助
Pareto 点。项目不照搬报告中的额外 checksum/协议层：用精确 job/config/path/EMA 和数值 parity
即可。若 R1 不能形成真实全栈收益并保护边界，则停止 ZoomToken 效率论文主张。本轮没有代码、
GPU/Slurm、训练或新性能/成本证据。完整回答与终态回执分别为
`.cvpr-pro-lab/reviews/PRO_POST_STOP_TEMPORAL_REDUNDANCY_FORWARD_ROUTE_ADJUDICATION_RESPONSE-v001.md`
和
`.cvpr-pro-lab/reviews/PRO_POST_STOP_TEMPORAL_REDUNDANCY_FORWARD_ROUTE_ADJUDICATION_TERMINAL_RECEIPT-v001.md`。

## 111. BPNS-R1 同硬件 final-EMA 成本回放进入正式执行（2026-08-27）

项目没有新增 60-epoch 训练，而是在 clean candidate
`b7357817d81127ab2d713b5471d008ea893efd35` 上闭合 K100 job `1248835` 与严格矩形 R1 job
`1249099` 的同硬件 epoch-59 EMA 重放。实现只新增 profiler、launcher 和 focused test 三个执行
表面；正式 population 使用官方 validation loader 的 211 个视频、792 个有序样本项，不沿用旧
P1 开发矩阵的 40-video/136-window population。K100/R1 在同一 Slurm GPU 内按 ABBA+BAAB 各执行
四次完整 pass，测量 decode→NMS 的阶段及全链路 p50/p95、吞吐率、峰值显存、gross energy，
并复核官方 evaluator、短动作和边界质量。

目标环境检查发现并纠正旧 population 假设与功耗 sidecar CPU affinity 两个确定性缺陷；最终
focused pytest `6 passed`，fresh independent Critic 为 `AUDIT_PASS`。真实结果盲 job-shell 见证
`1257250` 已 `COMPLETED 0:0`，fresh Evaluator 为 `PRE_RUN_READY`。唯一正式 replay job
`1257281`（`zt-bpns-formal-b7357817`）已在节点 `g0003` 进入 `RUNNING`，结果根为
`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_r1_cost_b7357817_seed42_20260827`。

当前裁决只把状态提升为 `experiment_running`，没有提升为 `empirically_supported`：尚未摄取完整
profile、终态数值或成本结论。只有 job 终态、数值 parity 和全部测量产物齐备后，才判断 R1 的
36% 原生空间输入削减是否形成真实全栈收益；live/intermediate 数值不进入论文结论。

## 112. BPNS-R1 正式成本回放因 K100 数值绑定不一致而停止（2026-08-27）

唯一正式 replay job `1257281` 在节点 `g0003` 运行 `00:38:04` 后以 `FAILED 1:0` 终止。它完成
首个 K100 validation pass 后，结果一致性门观测到 `mAP@0.7=46.246663`，与 profiler 预填的历史
值 `46.27` 不一致，因而主动抛出 `RuntimeError`。结果根没有 `profile.json` 或
`terminal_receipt.json`，R1 与其余 counterbalanced passes 未完成。

裁决为 `INCOMPLETE_REPLAY_NUMERICAL_BINDING_FAILURE`：这不是 K100/R1 的性能或效率失败，不能
解释任何局部延迟、显存或能耗数据，也不改变已冻结的训练准确率。该 job/namespace 保留为失败
执行证据，不自动恢复或重提。下一步只允许先明确历史精度的原始数值、舍入和容差合同，并让最小
修正重新经过独立 Critic 与结果盲 Evaluator；没有新的运行授权前不创建替代作业。

## 113. BPNS-R1 数值绑定最小修正通过并启动唯一替代回放（2026-08-28）

针对 job `1257281` 的准入失败，clean/pushed revision
`e9323448f6cd78b99bb3de53fd9ffb55f3676d65` 只修改成本 profiler 的 accuracy-parity 合同和
focused tests，不修改模型、forward、训练配置、数据、checkpoint、population、pass 顺序、
warmup、evaluator/NMS、成本仪器或硬件合同。六项 evaluator raw fraction 先转换为未舍入百分点，
再与冻结 reported-2dp reference 按 inclusive `0.05 pp` 比较；HALF_UP 两位展示与准入判定分离。

Focused tests 为 `13 passed`，fresh independent Critic 为 `PASS`，fresh result-blind Evaluator 为
`PRE_RUN_READY`。在明确运行授权后只创建一个替代 job `1258299`
（`zt-bpns-r1-pv2-e9323448`），按原八 pass K100/R1 同 GPU 协议运行。当前状态仍只是
`experiment_running`：完整 `profile.json`、`terminal_receipt.json`、预测、evaluator vectors、
功耗、显存、延迟、短动作和边界产物尚未形成，因此没有新的准确率、效率或论文结论。终态前
不得读取或解释 partial 数值、重复提交、追加种子或替换科学问题。

## 114. BPNS-R1 唯一替代回放在冻结一致性门处终止（2026-08-28）

唯一替代 job `1258299` 在节点 `g0048` 运行 `01:13:06` 后以 `FAILED 1:0` 终止。首个 R1
pass 的未舍入 `mAP@0.6=61.0869609029443100 pp` 与冻结 reference `61.14 pp` 相差
`0.0530390970556900 pp`，超过 inclusive `0.05 pp`。候选实现只在差值严格大于门槛时失败，
且 focused test 明确覆盖边界，因此该终止是冻结合同的正确执行，而不是单位、舍入展示或比较方向
的工程错误。

裁决记录为：该轮是 replay admission/protocol failure，不能成为 K100/R1 准确率、延迟、显存、
能耗、短动作或边界的科学结果。八个 pass 未完成，result root 为空，完整 profile、终结收据、
预测、cost samples 与功耗轨迹均不存在。Codex 不自行判断两位小数 reference 与 `0.05 pp` 合同
是否应修订，也不放宽门槛、补造产物、恢复或重提。按照既有 Pro 任务的 stop/post-result 规则，
下一步是一次全新的中性 Pro 科学复盘，由 Pro 独立决定停止、修订协议或其他唯一下一任务。

## 115. Pro 修订 replay admission 并冻结唯一 v003 任务（2026-08-28）

exact ZoomToken Project 的唯一有效全新 Pro 复盘在 verified Pro 路由下完成，裁决为 `REVISE`，
角色合同为 `KEEP`。Pro 将 v002 分类为
`VALID_FROZEN_CONTRACT_EXECUTION__INVALID_SCIENTIFIC_ADMISSION__NO_MODEL_OR_COST_RESULT`：
实现正确执行了冻结的 point-distance 合同，但 reported-2dp `61.14` 只标识舍入区间
`[61.135,61.145)`。观测 `61.0869609029443100` 到该区间的最小距离为
`0.0480390970556900 pp`，因此不能从点值差 `0.0530390970556900 pp` 识别真实 raw-to-raw
差是否超过 `0.05 pp`。该历史比较只能判为 `indeterminate`。v002 永久关闭为效率证据，且不得
把当前观测改写成新的历史 reference；BPNS-R1 只保留单种子准确率可行、效率未知的窄主张。

唯一下一任务冻结为 `ZOOMTOKEN-BPNS-R1-IDENTITY-GATED-FULL-STACK-REPLAY-v003`。候选必须是
`e9323448…` 的最小 clean descendant；硬门只覆盖 execution identity 与 measurement completeness，
历史两位小数准确率改为非阻塞的 `compatible/incompatible/indeterminate` 诊断。完整回放顺序仍为
`K100,R1,R1,K100,R1,K100,K100,R1`，每个 pass 持久化预测、evaluator 向量和身份，延迟与
gross energy 的主估计先按 pass 计算再取每臂四次的中位数。身份或测量不完整只形成协议无效终态，
不解释性能也不自动重提。Git push、远端写入和 Slurm/GPU 动作依赖人工授权。

北京时间节点冻结为：Builder plan `2026-08-28 10:00`，candidate `14:30`，Critic `16:00`，
Evaluator `17:00`，formal action `17:30`，queue check `2026-08-29 00:00`，blocker return
`00:15`，scientific return `12:00`。完整结果的主接受门为 p50 与 gross energy 各至少下降 `5%`，
且 worst-case `min(R1)-max(K100)` 的 Avg-mAP 与 mAP@0.7 均不低于 `-0.30 pp`；未取得完整
终态前不形成效率结论。

## 116. v003 协议失败后只允许一次解耦式成本闭环（2026-08-28）

fresh exact-Project post-result Pro 将 v003 裁为 `CONTINUE_ONCE_WITH_DECOUPLED_COST_CLOSURE`，
角色合同 `KEEP`。v003 的八个 prediction/SHA/vector 是执行确定性和固定种子准确率诊断；R1 相对
K100 的 Avg-mAP 为 `+0.5353 pp`，mAP@0.6 为 `-0.1042 pp`，属于混合且接近的观察。由于没有
成本、功耗、显存和边界产物，它既不能支持也不能否定效率主张。

唯一 successor 是 `ZOOMTOKEN-BPNS-R1-DECOUPLED-DIAGNOSTIC-AND-COST-CLOSURE-v004`。
成本采集、prediction identity 与离线诊断必须解耦：每 pass 先原子保存 raw cost、power coverage、
prediction SHA 和 pass receipt，八 pass 后才运行非计时 diagnostics。主判据冻结为每臂四 pass
p50 中位数比与完整 pass gross-energy 中位数比均 `<=0.95`。任一失败即停止当前效率 headline；
若在 raw acquisition 完成前再次协议失败，则不授权 v005 或更多 BPNS 重放。任何终态均先交 fresh
Pro，不自动添加种子、辅助臂、阈值修订、resume 或 rerun。

## 117. v004 完整成本否定 BPNS-R1 效率 headline（2026-08-29）

唯一 v004 job `1260095` 完成冻结的八 pass、raw cost、power、prediction identity、offline
short-action/boundary diagnostics、profile 与 terminal receipt。按每臂四个完整 pass 重算中位数后，
R1/K100 的 p50 比为 `0.9849289616`，gross-energy 比为 `0.9350002508`。能耗下降 6.50%，
但 p50 只下降 1.51%，没有满足两项均不高于 `0.95` 的预注册联合门。

因此科学裁决按冻结规则直接记录为 `STOP_BPNS_R1_EFFICIENCY_HEADLINE`：连续 K64 支持仍可保留为
单种子准确率与机制归因观察，但不能作为当前论文的真实端到端效率主张。显存下降与能耗通过只可
作为单硬件固定条件测量事实，不覆盖延迟失败。短动作与边界指标方向混合，不建立边界保护主张。
K100 pass 3 的约 `2.805 s` 功耗采样 gap 作为能耗不确定性披露；它不影响延迟判据，也不授权
重放。禁止 v005、retry、resume、阈值调整、额外 seed 或辅助臂；下一步仅为一次 fresh exact-Project
Pro 独立裁决，Codex 不自行选择替代路线。

## 118. v004 Pro 冻结 BPNS 并转向 TAR32 终态与匹配成本闭环（2026-08-29）

fresh exact-Project `GPT-5.6 Pro` 复盘返回 `PIVOT`，角色合同 `KEEP`。v004 工程证据为
`PASS_STRONG`，协议为 `VALID_WITH_DISCLOSED_POWER_UNCERTAINTY`，科学证据为
`VALID_NEGATIVE_FOR_STANDALONE_FULL_STACK_LATENCY_HEADLINE`。因此
`STOP_BPNS_R1_EFFICIENCY_HEADLINE` 维持不变；BPNS-R1 只冻结为空间支持可行性、局部 GPU/显存/
能耗归因以及“结构计算下降没有转化为 5% 串行全栈 p50 收益”的负系统证据。2.805 s 功耗 gap
只作能耗不确定性披露；下一成本工具必须按 pass 区间重算局部 gap。没有 v005、BPNS replay、额外
seed、阈值修订或边界保护主张。

唯一下一任务是
`ZOOMTOKEN-R1-TAR32-FKV-TERMINAL-VALIDATION-AND-K100-MATCHED-FULL-STACK-COST-CLOSURE-v001`。
先在读取指标前绑定原始 TAR32 授权、准确率门和成本规则的路径与 SHA，只读核验 job `1260166` 的
exact revision、60 epoch、epoch-59 EMA、official evaluator/prediction、`[64,32]x6` ledger 与无
fallback/新参数/新 loss。身份或终态产物无效即 `STOP_BEFORE_COST`。有效时仅允许从
`b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7` 建立模型零修改的成本后继，经过一次 Critic、一次
结果盲 Evaluator 后提交一个 K100/TAR32 同 GPU 八-pass 作业；p50 与 gross energy 分别按每 pass
计算、每臂四-pass 中位数形成比值，均须 `<=0.95`，功耗 coverage 为 `1.0` 且 pass-local gap
`<=100 ms`。任何终态后先做 fresh Pro，不自动 retry、resume、第二 seed 或第三臂。

## 119. pre-model blocker 只授权一次 TAR32 replacement evaluation（2026-08-29）

fresh exact-Project `GPT-5.6 Pro` 对 job `1261121` 的四秒失败裁决为
`REVISE_AND_CONTINUE`，角色合同 `REVISE`。该 job 在模型、checkpoint、loader 和 evaluator 前退出，
没有 prediction、metric、训练、resume 或参数更新，因此 scheduler submission 为 1，但 scientific
attempt 为 0；它不是 TAR32 正负结果。Pro 只授权 `RPL1_EVALUATION_ONLY_COMPLETION`：修正外部
launcher 的递归软链接清单检查，冻结 candidate、checkpoint、数据、evaluator、资源和无训练语义，
提交 scheduler ordinal 2 / scientific-attempt ordinal 1。第三提交、retry、resume 均禁止。

Builder、独立 Critic 和结果盲 Evaluator 已分别完成身份检查、`PASS` 与
`PRE_RUN_READY_REPLACEMENT`；唯一 replacement job `1261142` 已启动。终态前不消费 partial 指标。
若原冻结准确率门通过，只记 `ACCURACY_ADMITTED_PENDING_FRESH_PRO`；若失败，记
`STOP_R1_TAR32_FKV_EXACT_COMPOSITION`；身份或产物不完整则记
`ENGINEERING_OR_PROTOCOL_BLOCKER`。当前任务不授权成本，`ZT-CPTC-RP-K100-v001` 仍冻结，任何终态
均先进入一次新的 Project Pro 裁决。

## 120. TAR32 有效负结果停止精确组合并补齐 K100-TAR50 交互单元（2026-08-30）

fresh exact-Project `GPT-5.6 Sol / Power=Pro (5 of 5)` 裁决为 `PIVOT`，角色合同 `KEEP`。
job `1261142` 及冻结诊断足以形成单种子、准确率层面的有效负结果：三个官方主指标与短动作门均
失败，因此 `STOP_R1_TAR32_FKV_EXACT_COMPOSITION`。TAR32 成本、第三次评测、重训、附加 seed、
预算/层序/selector sweep、原地 residual rescue 和 `ZT-CPTC-RP-K100-v001` 全部冻结。该裁决只
否定 R1/K64 空间压缩与 `[K64,K32]x6` 固定半变换叠加的精确组合，不否定所有 CPTC。

唯一下一任务是 `ZT-CPTC-K100-TAR50-INTERACTION-FALSIFIER-001`，用于补齐 K100+半变换这一
2x2 机制矩阵缺口，区分“固定半更新本身失败”与“它只在叠加 R1/K64 后失败”。执行必须是
`2d945e64...` 的最小 clean descendant、native K100、`[K100,K50]x6`、full K/V、full Adapter、
恒等旁路、seed 42、60 epoch、epoch-59 EMA、一个正式 Slurm 提交。先绑定 capacity=1 job
`1254040` 及其 checkpoint/prediction/config SHA 和官方向量；正式失败不重提。六项准确率、短动作和
边界门全通过才记 `K100_TAR50_ACCURACY_ADMITTED_PENDING_FRESH_PRO`；任一有效失败即停止固定半更新
attention-column/identity-bypass family。无论结果如何，成本和后继仍需 fresh Pro 才能解冻。

## 121. RACER24 用户转交裁决按 provenance-warning 进入 Builder 规划（2026-08-31）

用户转交的 Pro 风格材料给出 `PIVOT`、角色合同 `KEEP`，并提出唯一候选 `ZoomToken-RACER24`：
保留 BPNS K64 连续原生支持，在 blocks `{4,6,8,10}` 进行 per-tubelet 24/64 selected-Q/full-KV，
以无参数 residual completion 恢复未选 token，随后维持 dense carrier 和全 token Adapter。该结构、
Iteration-0/1 顺序和停止门按 proposal 保存。

但材料没有 exact Project/conversation/nonce/model/附件/提交/terminal receipt，且包含两项已核验错误：
job `1258299` 已终态而非仍在运行；当前 base 没有现成 selected-Q/full-KV helper。因此本轮不把它
登记为浏览器审计完成的 Pro turn，也不把 RACER24 写成已实现或已授权训练的项目事实。唯一决策是允许
同一 Builder 返回 Iteration-0 `MINIMAL_CHANGE_PLAN`；在 clean candidate、parity、真实形状
microbenchmark、fresh Critic 与 result-blind Evaluator 前，不进行数据/GPU/Slurm/训练/成本，也不解冻
FARM24 或 PairLatent32。材料未给出新的北京时间 deadline，旧任务 deadline 不迁移。

## 122. RACER24 Iteration-0 未通过最低工程效率门并停止（2026-08-31）

用户另行确认了 provenance-warning 下的 Iteration-0 实现与冻结微基准权威。clean candidate
`5ebaa74f611bb3a43c3042700a78b92a9e5e74fb` 通过 focused tests、fresh Critic 与 result-blind
Evaluator。唯一有效微基准 job `1262068` 在固定 `B=1/T8/K64/Q192/KV512`、warmup 50、每臂 200 次
的 matched block path 上测得 p50 speedup `0.24964x`；peak allocated/reserved memory ratio 为
`1.98884/1.84615x`。因此预注册 `>=1.08x` speed 与 `<=1.05x` memory 门全部失败。

决定：`STOP_RACER24_ITERATION0_AND_RETURN_TO_PRO`。不运行训练、full-stack cost、K/block 调整、
FARM24、PairLatent32 或其他 rescue。该结果只否定当前 exact RACER24 实现的最低 block-path 工程效率
可行性；不外推为全部 selected-Q/full-KV、completion 或 CPTC 失败。下一科学任务必须由 fresh Pro
在完整负结果与 provenance 边界上独立下达。

## 123. fresh Pro 停止 RACER24 并冻结 GridFuse32-L6 门控任务（2026-08-31）

exact ZoomToken Project conversation `6a94842b-1370-83ea-a13c-2cc492170597` 在浏览器可验证 Pro
路由下完成一次提交、零 follow-up。裁决为 `PIVOT`、`STOP_RACER24_ITERATION0`、角色合同 `KEEP`。
job `1262068` 保留为 `DECISION_GRADE_VALID_NEGATIVE_NOT_CLAIM_GRADE`：运行时 candidate 尚未推送，
所以同时性 provenance 降级，但 local、remote 与后来 GitHub 上的 exact SHA 一致，足以支持内部停止，
不授权重跑。它不支持 accuracy、full-stack、energy、跨硬件或家族级外推。

浏览器日志实际上传 6 个文件；Pro 回答声称读取 7 个 attachment-only 文件并额外列出旧 BPNS 回执。
该冲突作为传输审计差异保存，不用第二次提交或 follow-up 修补。

唯一下一任务为 `ZOOMTOKEN-GRIDFUSE32-L6-GATED-v001`：从 clean base `2d945e64...` 开始，保持 R1
K64、全部 temporal tubelet 与 dense Adapter；blocks 6–11 用固定相邻 pair 将完整 Q/K/V/MLP 的物理
序列缩到 N256，再把 block residual 广播回 N512。G0 先要求六-block segment p50 speedup `>=1.35x`
且 allocated/reserved ratios 均 `<=1.05`；仅通过才开启单 seed G1，G1 通过才开启 matched full-stack G2。
任一门失败即终态并返回 fresh Pro，不做 rescue、sweep、第二候选或额外 seed。

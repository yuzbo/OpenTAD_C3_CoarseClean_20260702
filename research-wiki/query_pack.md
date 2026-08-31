---

## 2026-08-17 DUCA official-baseline 资源校正

中央的 N16R4 只读核验已纠正先前将 `yuzibo` 当 SSH hostname 的错误：共享根
`/data/run01/sczc063/yuzibo` 中 THUMOS14 raw video（411 个有效 MP4 symlink：200 training、211 validation）、
annotation/category map 与 VideoMAE-S pretrain 均为 **COMPLETE**，`/data` 约 3.1T free。此事实只解除
raw-video AdaTAD-S baseline 的数据/存储阻塞；它不构成任何 DUCA 效能、成本或官方 mAP 结果。OpenTAD I3D/
InternVideo2 feature 预期目录仍 absent；空 alternate I3D 下载、SigLIP2 特征及 native MATR pickles 都不能代替
official raw-video identity。现存第一实验仅等待：released AdaTAD-S checkpoint 的精确可读绑定，以及 clean
official code/config/evaluator/runtime 的 project-owned PRE_RUN；不得下载、混用资产或直接训练。详见
`DUCA_WIKI_MEMORY_AUDIT-2026-08-17.md`。

远端视频地图补充：THUMOS14 physical store 的 413 个文件中只有 canonical 411（200 training +
211 validation）可绑定 annotation；两项 extra test files 永不采用。TOC-Bench（1951 MP4）和
Charades（9848 MP4）存在完整 raw trees，但尚没有 DUCA protocol；MultiSports 只有 archives/18 extracted，
ActivityNet 只有 3 readable MP4，FineAction/HACS/EPIC-Kitchens/Ego4D 没有实际视频树。它们不改变当前
THUMOS baseline，也不授权解压、下载或跨数据集训练。

## 2026-07-28 SparseHead 唯一路线已恢复并合并

当前仓库/工作区成为 SparseHead/PhysTime 的唯一可写研究面；
`OpenTAD_SparseHeadClean_20260702@dce2c66` 进入只读封存。两仓无共同 Git
祖先，禁止整树 merge/cherry-pick；本轮只吸收旧仓 irregular bridge/point
generator/native-axis assignment audit 的有价值实现，并把它们锁成
`diagnostic_only / primary_result_forbidden / slurm_forbidden`。唯一允许继续裁决的结构
候选是 SDPQ：完整 physical query grid 与 sparse observation support 解耦；但当前最强
经验基线仍是 native-J192 physical-metric ActionFormer。

证据边界：matched 20-epoch 的 selected/physical/SDPQ Avg-mAP 为
`30.42/44.88/30.88`，所以当前 SDPQ 不优于 physical-metric；full60 仅比较
selected 与 physical，P0 full-precision replay 为 `41.283021/57.608685`
（`+16.325664 pp`），关闭 rounding/NMS 舍入混淆，但仍只是 THUMOS 单种子且不包含
SDPQ。decode-cross source-dtype evidence chain 已实现，并在 N16R4 隔离包
`e4814e...ef388` 通过配置/导入/编译/launcher syntax 与 `59 passed` Linux CPU
focused tests；远端测试还关闭了 physical config 未消费显式 seconds-axis 字段的
生产契约缺口。它仍没有新 CUDA gate/job/result 或 mAP。下一步必须冻结精确提交并
运行 selected/physical × online/EMA 四条件 real CUDA gate。详见
`experiments/sparsehead-route-consolidation-20260728.md`。

## 2026-07-23 免目标域训练粗分类/选帧候选与 T1 裁决

`ideas/duca-target-train-free-transition-prior.md` 已升级为
`implemented/tested/experiment_running`。唯一分支为
`codex/duca-t1-trainfree-20260723@4c5604b4a0abde9e59f625d519934e855bfe1519`，Linux focused
`29 passed`。免训练只指 pre-backbone 前端：冻结外部预训练视觉先验、不使用目标标签/微调/
测试时梯度，并用无参数状态变化证据驱动现有 exact-K/G 或 R2Q3 分配；AdaTAD detector 仍在
THUMOS 训练。MobileNetV3 是低成本候选；SlowFast 只执行 Fast pathway，Slow 与 lateral fusion
均关闭，是高成本强先验诊断。Fast Job `1180653` 已完成 epoch 0 并进入 epoch 1，MobileNet 最终
重试 `1180654` 已提交并等待 GPU 配额。当前无 terminal mAP/成本结论，详见
`experiments/duca-t1-and-target-trainfree-official60.md`。

PhysTime 负结果被重新限定为 v1 多变量失败。下一次只允许相同硬选帧和 selected-axis detector 下
测试零初始化 T1 true-time residual、baseline 与打乱/常量时间码；T1 无益时不得直接做 T2。
五类论文图已映射到当前实验：没有 terminal mAP/完整成本的运行中或 coarse-only 实验禁止预画
性能结论。

## 2026-07-23 DUCA 论文叙事、理论与图表合同

新增唯一论文证据规划 `research-wiki/duca_paper_story_theory_figure_contract.md`。当前投稿
裁决为 `HOLD`：代码主线是 offline TAD 的 transition-calibrated boundary-burst
pre-backbone acquisition，但 `paper/` 仍描述旧 zero-shot/teacher-utility/Top-K/
window-online 方法，且 C1/C3/C4/C7 尚无 terminal official mAP、完整总成本和多种子闭环。
R0--R5 只能作为内部证据流程，论文必须改写为“可达性、可学习性、有效性、归因性、效率与
泛化”五个科学问题。主图固定为方法总览、同视频机制图、官方准确率-总成本 Pareto、五预算
mAP 曲线和边界分布/失败分析；内部 93--94 holdout mAP 禁止进入论文主表。

## 2026-07-23 01:16 稀疏粗扫描 hidden-linear 已实现并部署

本轮统一总账见 `research-wiki/experiments/duca-20260723-nightly-implementation-and-deployment.md`。

唯一实现身份为 `codex/duca-sparse-probe-interpolation-20260723@dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45`。模型只在 d=1/2/3/4 的规则时间 anchor 上运行低分辨率空间 stem 与官方 ASFormer，将 action logits、多维 temporal encoder hidden 和 policy hidden 线性插值回完整候选网格，再交给现有 R2Q3/K384/G2 selector；不输入 anchor mask 或 anchor 距离。VideoMAE、official-derived AdaTAD/ActionFormer、损失与 official-60 协议均保持一致。

新 focused `4 passed`；真实 CUDA Gate `1180556` 已 `COMPLETED/0:0`，证明四档完整长度、有限数值、空间/时序梯度非零和 MACs 单调下降。正式四卡套件 `1180557` 已 RUNNING，四臂都进入 P0 epoch 0；运行根为 `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329`。当前没有 terminal epoch-59 EMA mAP，不能宣称稀疏粗扫保持检测性能。

同晚四种官方粗分类器 P0 Jobs `1180502--1180505` 已全部完成。Video-Mamba-ASFormer 的 Action AP/`delta_p_action` 边界支持@1 为 `0.4161/0.8302`，ASFormer ROC-AUC/F1 为 `0.6237/0.5158`；四者最佳边界策略均是 `delta_p_action`。这些只是粗分类/间接边界诊断，不是 TAD mAP，也不能在统一 temporal-hidden 完整 TAD 前直接决定后端。

## 2026-07-23 a00498e selected-axis / TTDI Pro 审查已复核

原文已按 SHA-256 `36523b2f...a884` 逐字归档。远端 diff 证明 `a00498e -> 9f97f2c`
只改执行脚本，因此 selected-rank 时间扭曲、mandatory group union 无接纳前 completeability、
以及 local-RGB-slope detector surrogate 三项模型风险仍存在；五预算成本 parser 也仍只认
K384/K256。当前五预算已统一到 `9f97f2c`，所以旧“跨提交合并”不是本轮直接性能风险。
项目只部分接受 TTDI：先等待 matched terminal mAP 与 selection-quality 分叉；若选帧质量更好
但高 tIoU mAP 不升，只做 zero-init true-time feature residual 三臂，physical-coordinate head
必须作为第二变量。禁止把 TTDI、具体 loss 权重或 2000/6000 更新配方提前写成最终事实。

性能诊断顺序已进一步冻结：learned hard selection 若不优于 uniform，先修 group-aware
mandatory/completeability 与 scorer；selection 已优而 mAP 不升，才测试 T1 true-time feature
residual；若 G1/G2 低于 G0 或 legal hard-swap 对齐失败，关闭当前 local-RGB-slope bridge 并比较
真实 hard-forward/soft-backward selected-RGB 传输或 legal-swap detector utility。成本 parser、
G2 命名和 official-derived 后端表述影响证据可信度，不应冒充 mAP 模型修复。

## 2026-07-23 四种官方粗分类后端已并行启动

当前代码分支新增精确提交 `4f81299`，只加入极简四后端粗分类实验入口，不改变正在运行的 `9f97f2c` R 系列模型。MS-TCN2、ASFormer、FACT、Video-Mamba-ASFormer 四个官方源码适配已通过 Linux focused `11 passed` 与真实单卡 CUDA 小前向，独立 Jobs `1180502--1180505` 已同时运行。协议固定为 64x64、768 点窗口、seed 3407、20 epochs、终轮评估、无 early stopping。

稀疏粗扫合同已按最新裁决修正：线性插值的是时序隐藏特征而非单一 `p_action`，插值结果直接作为 selector 证据；不额外提供真实观测掩码或距 anchor 距离。当前四后端 P0 只产生粗分类/边界诊断，不是 TAD mAP。完整 TAD 公平比较前必须让四后端统一输出 temporal hidden，禁止用 ASFormer temporal hidden 对比其他模型的 spatial-stem hidden。

## 2026-07-22 模型算法优先决策

仓库级优先级已经冻结：目标是提出性能更好、创新性更强且可由实验裁决的模型算法，不是建设更复杂的工程项目。默认关键路径为模型假设、最小端到端实现、matched baseline、官方 mAP/高 tIoU、真实成本和机制消融；工程只保留防止模型行为错误、数据泄漏、指标失真或任务无法运行的最小闭环。

对于当前空间路线，主问题是让类似 Uni-AdaFocus 的全局观察与连续区域策略通过真实 AdaTAD 检测损失学习可变中心和宽高。手工候选、GT 特权裁剪和 crop-sufficiency 枚举只能作为诊断上界，不得替代或阻塞可学习模型的直接实验。状态：`decision_frozen / implementation_priority`。

## 2026-07-22 16:23 e49 R0 三组精确点估计

精确提交 `e49ef69605e1f98a7217957483f93a8a64bfc348` 的冻结 official-AdaTAD
training-internal holdout 重放已得到三组精确 Avg-mAP：exact-uniform
`93.587070`、projected R2Q3 `94.190497`、projected R4Q5 `93.999241`。
R2Q3/R4Q5 相对 uniform 分别为 `+0.603427/+0.412170 pp`；R2Q3 仍是当前
原始点估计最好的 projected family。R4Q5 在 tIoU 0.7 略高于 R2Q3
（`85.9983` 对 `85.6068`），但总体更密、更宽的 burst 没有带来更高平均收益。
unrestricted Oracle 也已复现 `93.970057`，仅比 uniform 高 `+0.382987 pp`，并低于
R2Q3 `-0.220440 pp`。四族点估计全部完成后，R0 已进入 1000 次 paired-video
bootstrap；CI 决策与 P0 解锁仍在运行，因此这仍是 Oracle 几何可达性证据，不是
learned selector 或论文测试集结果。

## 2026-07-22 16:05 R0-R5 精确版本已全面部署

唯一正式候选为 `codex/duca-boundary-burst-20260722@e49ef69605e1f98a7217957483f93a8a64bfc348`，GitHub 精确对象为
`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/e49ef69605e1f98a7217957483f93a8a64bfc348`。干净 N16R4 快照
`/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_e49ef69_20260722`
通过 R0-R5 focused `192 passed`、强制 C3/official-ASFormer `23 passed`、
pycompile/bash/clean-tree。部署前独立 MAX
`019f88bf-272f-7373-b702-5b66b142cbdc` 给出 `GO_TO_SLURM`，确认终端证据、
完整成本、K256 burst 自由度和 R5 聚合四项旧 blocker 已关闭。

正式根为
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037`。
R0--R3 Job 为 `1179795/1179796/1179797/1179798/1179799/1179825`，R4 为
`1179826`，R5 TemporalMaxer 真后端门禁为 `1179827`。N16R4 的
`AssocMaxSubmitJobLimit` 阻止 24+9 个单元逐 Job 展开，因此六个尚未运行的重复单元
`1179828--1179833` 已取消；等价地提交四个训练批次
`1179861--1179864`，每批顺序执行 6 个原始哈希绑定 sbatch，后处理 Job `1179865`
顺序执行 9 个成本 profile 和最终 aggregate。实验内容没有减少或改变。

部署收据 SHA-256 为
`ed217ee233f43505ce025c0979451dadf21508715ab5b2534598e1f662eaa1bd`，R5 bundle
manifest SHA-256 为
`e6ca0acfa38e0c0929adfe7cb2f56f6af5b60941b09bd7504f712a2de45656d8`。
16:05 健康检查无 Traceback/OOM/non-finite/FAIL；`1179795` 正在运行，其余均为合法
dependency pending。状态是 `experiment_running`，不是 `empirically_supported` 或
`paper_ready`；R5 仍须等待 R0 family、P0、U/G0、R4 alignment 后才能实际启动。

## 2026-07-22 12:25 独立 MAX 前移为正式部署许可门

R0-R5 全部生产代码与真实后端门禁完成后，必须先启动一次全新、无实现上下文的独立 MAX 审阅；只有其从模型设计、机理、梯度归属、真实硬选帧和训练/推理一致性角度明确给出 GO，才允许提交新的正式 Slurm 实验 DAG。不得把该审阅拖到实验结果收割之后，也不得把普通工程洁癖升级为方法否决项。当前已运行的 R0/P0 仅作为此前阶段证据，不能替代完整 R0-R5 候选的部署前审阅。

## 2026-07-22 12:31 R1/R3 生产执行合同完成

Canonical branch 新提交 `523b45f62eb3a3d0c2856f33161dc541932c564a` 将主 DAG 固定为 `R0 -> P0 -> gate -> {matched U, runtime-selected G0} -> aggregate`。仅 R0 封存的唯一 projected family 可成为 G0；Gaussian 与未选 family 不生成正式作业，也不能阻断主结论。远端临时副本 focused suite 为 `74 passed`，真实资产 `PRECHECK_ONLY=1` 生成六个 sbatch；按新的审阅时点合同尚未执行 `sbatch`。

## 2026-07-22 12:33 模型实现优先于启动工程

R4/R5 立即停止扩张通用 orchestrator、复杂 journal/schema/router。只保留直接运行、依赖和结果路径所需的最小配置与 sbatch 入口；实现与审核重心固定在粗分类证据、Oracle 式边界微簇、真实 hard swap、检测梯度归属、DUCA->VideoMAE->真实第二 detector 和最终 mAP。工程问题只有在改变模型行为、证据真实性或可复现性时才阻断。

## 2026-07-22 12:47 R4 legal hard-swap 实现完成

Canonical commit `e253bba52dce1814f4ef356adcad286bc0884457` 实现真实 selected-RGB 一进一出 hard swap、selected-axis GT 重映射、冻结 G0 epoch-59 EMA official AdaTAD 的 `base_loss-candidate_loss` utility、48 windows/576 swaps 的 signed alignment 与逐视频 bootstrap。G1 只允许 detector bridge 更新 transition scorer；G2 加入 50% train-only exact-uniform companion，推理仍只走 learned policy。R4 focused `47 passed`、基础回归 `23 passed`；仅保留单一直接 Slurm 入口，尚未部署，等待合并后的真实 CUDA gate 与部署前独立 MAX。

## 2026-07-22 12:49 R5 真实第二后端实现完成

Canonical commit `163be81fb640376f0d6e1b09c86eb011f3402242` 接通 live `DUCA selected RGB -> VideoMAE -> TemporalMaxer`，补齐 selected-axis GT/推理逆映射、selector 参数 optimizer 覆盖，以及真实 dataset/config/model 的一步 detector-loss backward/update 门禁。R5 矩阵固定为 ActionFormer/TemporalMaxer × U/learned × K384/K256 × seeds 3407/5801/8123，共 24 个显式配置；此前过度复杂的通用 orchestrator 已删除，保留最小生成器与直接 sbatch。Windows focused `6 passed, 2 skipped`，两项 CUDA/PyTorch 机制门禁须在 N16R4 补齐；未部署正式作业。

## 2026-07-22 12:59 最终合并候选 Linux 门禁通过

唯一合并 HEAD 为 `44c7227b575b22c666b2f309c69b1dcfdc4102c8`，远端干净快照为 `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_44c7227_20260722`。第一次 `01d1ed0` 核验暴露 R5 测试 fake selector 漏写 selected-axis 输出坐标字段；生产 selector 与 fail-closed 模型检查均正确，仅测试夹具在 `44c7227` 补齐。最终远端集中结果为 R0-R5 focused `90 passed`、强制 C3/ASFormer `23 passed`、py_compile/bash 通过。尚缺真实 CUDA one-step gate 与部署前独立 MAX；不得部署正式 DAG。

## 2026-07-22 13:05 最终合并候选真实模型门禁已排队

Slurm Job `1179602` (`duca44c_model_gate`) 已按 `afterok:1179533` 排队，当前为 `PENDING (Dependency)`。它不是论文正式训练，而是部署前真实后端门禁：从 d9 P0 的封存 winner 加载 selector，在精确 `44c7227` 快照上分别执行 R0-selected family 的 G1/G2 official AdaTAD exact full-model forward/backward/hard-gather gate，再执行 live DUCA->VideoMAE->TemporalMaxer 一步 optimizer gate。输出根为 `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_44c7227_model_gate_20260722_1305`。该 Job 通过后才允许启动最终独立 MAX；MAX GO 前仍禁止正式实验 DAG。

## 2026-07-22 11:46 corrected R0 evaluating; P0 dependency queued

Exact-d9 R0 Job `1179517` remains `RUNNING` on `g0048`. The 124-window input
export and all four selection-family construction completed; official frozen
AdaTAD replay is now evaluating `R2Q3`. stderr and the error scan are empty and
no mAP is sealed yet.

To remove avoidable serial queue delay, P0 Job `1179533` was atomically appended
to the existing journal with dependency `afterok:1179517`. The journal SHA-256
is `cbd7f59a94eb472daf5d94df5728822d59eed6890e0e9140d97774952148a750`.
The P0 consumer reopens and validates the sealed R0 summary; if R0 returns the
pre-registered KILL/nonzero status, P0 cannot start. No gate or official-60 job
has been submitted.

## 2026-07-22 11:24 corrected R0-only submitted

The authorized exact-d9 corrected R0 was submitted alone as Slurm Job
`1179517` (`burst_r0_d9fb398`). Run root:
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357`.
At 11:26 +08:00 it is `RUNNING` on `g0048`; export has begun and stderr is
empty. The atomic journal contains only
`r0_holdout_map -> 1179517` with no dependency; no P0, gate, official arm or
aggregate job was submitted.

The exact manifest/r0-sbatch/jobs hashes are respectively
`66bb5f5d...74e15`, `4a6ace6c...ab8c`, and `25d90d86...07f4a`; the sealed
split manifest hash is `88309edf...708f`. Current stage is
`experiment_running_r0_only`. The next decision depends only on frozen-detector
paired-bootstrap R2Q3/R4Q5/unrestricted Oracle headroom over exact-uniform.

## 2026-07-22 11:24 independent MAX grants R0-only GO

Fresh no-context MAX `019f87bb-d767-7713-825e-92b893e49a98` audited the full
`22555a4..d9fb398` change set and granted **R0-only GO** with no remaining R0
model, math, determinism or Linux blocker. It separately kept P0, full-model
gate and official-60 on HOLD until the corrected R0 proves positive paired
headroom and seals one selected projected family.

The review also found two future official-only evidence defects: the current
submitter still allocates four sentinel GPU jobs so diagnostics can block the
main DAG, and terminal aggregation checks cross-arm pretrain equality without
rebinding it to the P0/full-gate pretrain path identity. Neither affects the
R0-only evaluator. They must be repaired and independently re-audited before
official training, not before R0. The only authorized next action is a fresh
exact-d9 corrected R0 job with no P0 or long-run dependency.

## 2026-07-22 11:06 exact d9 candidate and deterministic real replay

The bounded solver and downstream evidence fixes are now unified, pushed and
clean at `codex/duca-boundary-burst-20260722@d9fb398578716d278e818745677a92976bcedf2c`.
The clean Linux snapshot is
`/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_d9fb398_20260722`.
It passed the focused DUCA suite (`88 passed`), mandatory C3 regressions
(`23 passed`), pycompile and shell syntax checks. The selected weakest R0
projected family is now the sole mandatory learned P0/gate/R3 family; P0
rehashes the actual official-ASFormer training consumer, and full-model gates
reopen artifact content rather than trusting path identity.

The old 124-window real R0 input was solved twice from the exact d9 snapshot.
Both runs produced byte-identical family JSONL with SHA-256
`b49e03c2f4222512cf7752bd3c89bad714868ae69e7c9d05980f9e9f47edd6d7`.
The summary hashes differ only because each summary records its own output
path; this is not a selection difference. Thus the strict deterministic-tie
R0 blocker is closed on real data. Fresh no-context MAX
`019f87bb-d767-7713-825e-92b893e49a98` is still auditing d9 and must return
separate R0/P0/full-model/official decisions. No corrected R0, P0 or
official-60 job has been submitted, and no new detector mAP exists.

## 2026-07-22 10:08 R0 deterministic solver integrated

The MAX-only R0 blocker was minimally fixed and cherry-picked into the
canonical branch as `c418a951a9b9b7f7f19df785ead8642a4205c804` (source worker
commit `e267e1f9562c91fc0ad9a60382eb829d82d41acd`). The exact-quota MILP now
first pins optimum exact-uniform overlap, then optimum position sum, then
applies the existing 20-position-block lexicographic pin. No deployable model,
K/G, quota, loss, decoder or detector changed.

Evidence is focused solver/GT `24 passed`, mandatory C3 `23 passed`, local
canonical tied-instance/repeat test `7 passed`, pycompile and clean source
worker tree. The old real 124-window production replay and remote Linux suite
remain required after the downstream evidence patch is merged into one new
exact commit. Current HEAD is not yet deployment-authorized or pushed as the
next audited version.

## 2026-07-22 09:37 independent MAX HOLD on 22555a4

Replacement no-context MAX `019f875f-1668-7e51-bf97-1f565b25e106`
returned separate HOLD decisions for R0-only, P0, full-model gate and
official-60. It verified the exact-K/center/joint-quota/bilateral/projected
physical-path/max-gap/unrestricted semantics, zero MIP gap, R0 mAP/bootstrap
recomputation, sealed P0 record reanalysis, and atomic self-sealed aggregate.
The only corrected-R0 blocker is deterministic tie resolution: maximizing
uniform overlap and then minimizing the sum of positions does not uniquely
order every equal optimum. This affects reproducibility, not feasible-space
mathematics or optimality.

The only allowed R0 repair is to preserve the first two objective optima and
then apply a strict per-position lexicographic pin, with tied-instance,
repeat-solve and real-failure replay tests. Parallel worker
`019f877c-4595-7431-96a0-edff1f7b8251` owns only this solver/test surface.
Downstream worker `019f8766-c4db-7e30-8fc8-265d85d83b07` continues the bounded
selected-family propagation, P0 official-ASFormer source revalidation and
full-model gate artifact-consumption fixes. Exact commit `22555a4` is now
explicitly HOLD and must not be submitted.

## 2026-07-22 09:12 corrected R0 audit restart and bounded parallel work

Beijing 09:00 passed without a corrected R0 mAP result. Exact commit
`22555a4e830ce24f9bb516897b1bb7f44b70c188` remains the only candidate and
the canonical tree is clean. The first fresh reviewer
`019f8743-aed1-7a80-a7d6-552b08491019` stayed running without returning a
verdict and was shut down; it produced no deployment authorization. A new
independent no-context MAX reviewer `019f875f-1668-7e51-bf97-1f565b25e106`
is now required to return separate R0/P0/full-model/official-60 GO/HOLD
decisions.

In parallel, a bounded implementation worker
`019f8766-c4db-7e30-8fc8-265d85d83b07` is closing only rule 300: the sealed R0
`selected_weakest_projected_family` must be the sole mandatory learned family
for P0, the real-model gate and matched R3 U/G0. Gaussian and the unselected
burst family may remain optional diagnostics but cannot veto or force extra
official-60 arms. This work does not alter the coarse model, scorer, decoder,
loss, K/G or official AdaTAD. No corrected R0/P0/official-60 job is currently
authorized or submitted, and there is still no new detector mAP.

## 2026-07-22 08:40 corrected R0 exact-quota candidate

R0-only Job `1179392` from `f90595d` is immutable `FAILED` and is not mAP
evidence. It exported all 124 holdout windows correctly, then failed before
detector replay because the privileged Oracle incorrectly converted
"choose at least Q positions within radius R" into a fixed nearest-Q required
set. On the first failing full window (`video_validation_0000206|0`, K=384,
G=2), the fixed R4Q5 union needed a 429-position physical path, while a true
joint quota MILP has an exact 384-position, G2, bilateral, zero-gap witness.

The bounded repair is exact commit
`22555a4e830ce24f9bb516897b1bb7f44b70c188` on
`codex/duca-boundary-burst-20260722`. It keeps the deployable selector,
K/G and official AdaTAD path unchanged; only the privileged R0 Oracle now
jointly enforces endpoint center, within-radius quota, bilateral support,
exact-K and the physical path cap. The same commit closes the previously
authorized P0/aggregate evidence gaps. Remote clean-snapshot evidence is
Oracle/solver `22 passed`, P0 summary `9 passed`, runtime/gate/aggregate
`54 passed`, required C3 `23 passed`, plus pycompile/bash/HEAD/clean. Replaying
the exact failed real sample now builds U/R2Q3/R4Q5/unrestricted with
`ok=true` and output SHA-256
`168c6f21f869d802e8e3a11fdfcedc2ddc7968fe6fb5b6909776fdb8f84e76ce`.

Current paper stage is still **R0 geometric/headroom calibration**, not P0 or
main official-60 training. A fresh independent MAX audit
`019f8743-aed1-7a80-a7d6-552b08491019` is reviewing the exact commit. Only a
corrected R0 rerun may follow its R0 GO; P0/full-model/official-60 remain held.
No boundary-burst terminal mAP, >65, V9 or paper-ready claim exists. No
open-ended Pro redesign is needed; only this bounded exact-commit audit and the
pre-registered R0 statistical adjudication are allowed.

## 2026-07-22 06:21 86f7663 independent audit HOLD

No-context MAX `019f86a6-fe1b-7921-b576-8d9cd3d4c8ac` returned
`HOLD_FIX_REQUIRED`. The boundary-burst architecture and official AdaTAD path were not killed,
but no CUDA/R0/P0/official-60 job may run yet. Required repairs are bounded: complete R0 with
uniform/projected/unrestricted Oracle, official-evaluator per-video bootstrap and full artifact
identity recomputation; place simple `abs(delta p_action)` in the same global exact-K/max-hole DP
and enforce its stop rule; close crop-valid, atomic submission-journal and no-mock integration
evidence gaps. Current stage is R0-before-R1 contract repair, not main-experiment training. Do not
start another open-ended Pro redesign; after repair run only a new exact-commit review.

## 2026-07-22 04:45 f629ad7 exact re-audit candidate

The two remaining aa3352e audit blockers were minimally closed and pushed as
`f629ad79461941f405bc2028f087034abd17a840` on
`codex/duca-boundary-burst-20260722`. Submit-time AdaTAD pretrain path/SHA is
now injected into every DAG job and independently reopened by P0, every
frontend arm and the full-model gate; behavioral tests reject path/content
drift. Production runtime binding now unconditionally rejects G1/G2 before
any gate or config consumption until real legal hard-swap alignment exists.
No scorer, burst objective, K/G, DP or official detector changed. The clean
remote snapshot passed affected DUCA `63 passed`, required C3 `23 passed`,
py_compile/bash/HEAD/clean. A broader historical set produced 171 pass/3 skip
and six old transition-only protocol failures; these are disclosed to review
and are not being repaired in this route. Fresh no-context MAX
`019f866a-6879-75a0-99f4-3c9524ebd076` is auditing the exact commit. No CUDA,
R0, P0 or official-60 job is yet authorized or queued.

## 2026-07-22 04:26 aa3352e independent audit HOLD

Independent MAX review of exact commit
`aa3352ecf803c81d007a62ed5398667d9551684b` returned
`HOLD_FIX_REQUIRED`, not deployment GO. The boundary-burst objective, global
exact-K/max-hole DP, selected-axis mapping, no-leak contract and official
detector were not rejected. Two P0 evidence-contract blockers remain: (1) the
AdaTAD pretrained-weight SHA must be frozen at submission and reverified by
P0/frontend/gate consumers; (2) G1/G2 must be rejected by the production
runtime binder until real legal hard-swap alignment passes. Remote DUCA
`139 passed, 3 skipped` and C3 `23 passed` remain static evidence only. Job
`1178989` is terminal `FAILED/2:0` by the old V8 mechanism HOLD, and no
boundary-burst CUDA/R0/P0/official-60 job is queued. Current paper stage is
still R0-before-R1 contract repair, not main-experiment training.

## 2026-07-22 04:05 aa3352e exact candidate

上一轮独立 MAX 对 `899630a` 指出的 runtime binder、pooled validity 与 provenance/hash
blocker 已在同一 branch 修复并推送为
`aa3352ecf803c81d007a62ed5398667d9551684b`。四臂 production binding、boundary gate
schema/config-stem artifacts、crop-valid pooled metrics、split/checkpoint/pretrain 和上游
SHA seals 已闭环。干净远端快照通过 DUCA `139 passed, 3 skipped`、C3 `23 passed`、
compile/bash/HEAD/clean。全新独立 MAX `019f8647-ad93-70f3-a763-218f7552ac95` 正只读
审查精确提交。当前阶段为 R0 前 R1 复审；CUDA gate、R0/P0/official60 均未提交，仍无
headroom、winner、terminal mAP、V9 或 >65 证据。

## 2026-07-22 04:00 boundary-burst audit HOLD and paper position

当前唯一候选仍来自
`codex/duca-boundary-burst-20260722@899630a5ef4927e78ef4ca6b8cc51fdf754056da`，
但独立 MAX `019f8614-53e8-79e2-8daa-d52f7be04623` 已裁决
`HOLD_FIX_REQUIRED`。模型主体、全局 DP、quota burst、selected-axis/no-leak/official
detector 未被否定；阻断是四臂 runtime binder/gate mapping、pooled crop validity 与
R0/P0/aggregate provenance/hash 闭环。pooled validity 已完成 focused `20 passed` 的最小
修复，其余修复仍在同一分支进行。当前论文阶段是 V8 负向封存之后、R0 运行之前的 R1
部署前证据合同修复；没有 R0 headroom、P0 winner、terminal mAP、V9 或 >65 结论。
不需要开放式 Pro 发散，只需要修复后 exact commit 的独立逐行复审；随后才允许 CUDA
gate 与 R0->P0->U/Gaussian/R2Q3/R4Q5 DAG。

## 2026-07-22 03:27 boundary-burst exact candidate

Canonical DUCA successor 当前是
`codex/duca-boundary-burst-20260722@899630a5ef4927e78ef4ca6b8cc51fdf754056da`，
不是旧 `fdf25f5`、V8/local-cell 或新 selector。它在既有 V8 scorer/全局 exact-K/G DP
中实现预测 transition center 的 R2Q3/R4Q5 双侧配额微簇，并关闭逐样本 K/G、
`floor(start)/ceil(end)-1`、crop validity、earliest-pass P0、R0 `afterok` 阻断、split
hash reopen 和 U/learned 制品语义。最终干净 Linux 快照通过 `136 passed, 3 skipped`
DUCA 回归、`23 passed` C3 回归、compile/shell/clean-tree。独立 MAX
`019f8614-53e8-79e2-8daa-d52f7be04623` 后续已给出 HOLD，因此真实 CUDA gate 和正式
R0->P0->gate->U/Gaussian/R2Q3/R4Q5 DAG 尚未提交。旧 V8 Job `1178989` 已
`FAILED/2:0`，只待封存失败阶段；它不能再解锁论文主张。

## 2026-07-22 02:33 boundary-burst implementation candidate

首个 canonical successor 候选已固定并推送：
`codex/duca-boundary-burst-20260722@fdf25f5d08bc0bf9b550e059228ce1d6ac587499`。
它在现有 V8 scorer/全局 exact-K max-hole DP 上实现 Oracle 式左右边界微簇，冻结
Gaussian/R2Q3/R4Q5 P0 与 U/Gaussian/R2Q3/R4Q5 official-60 四臂；G1/G2 仍禁止。
R0 selected-axis 回放已修复背景窗口、training evaluator subset 与 blocked-video 真值范围，
但因冻结 detector 见过 training 视频，R0 只解释 Oracle-U 相对 headroom，绝对 mAP 不进主表。
远端 Linux focused 为 `84 passed, 2 skipped`，必要回归为 `23 passed`，提交 DAG 预检通过。
独立 MAX 对 `fdf25f5` 的裁决为 `HOLD_FIX_REQUIRED`：核心模型的 offset 梯度、全局
exact-K/G DP、selected-axis 映射、无 GT 泄漏和官方 detector 未发现阻断，但 P0 逐样本
K/G 自证、端点评估坐标、earliest-pass、R0 DAG 阻断、裁剪端点 validity、R0 双侧配额
Oracle、U arm 制品语义与 split 哈希仍需修复。三个合同切片已并行修复；主线程把 Q 改成
预测中心的 quota-limited offset support。尚无新训练 Job、terminal mAP、V9 或论文结论。

## 2026-07-22 DUCA 论文闭环总表冻结

唯一最终合同仍是 `duca_final_model_contract.md`：固定预算 offline-TAD pre-backbone
插件，低成本粗动作二分类与 official ASFormer 提供状态证据，boundary-burst selector
学习 Oracle 式端点居中、左右 3--5 帧微簇、配额饱和与端点公平，现有 global exact-K/
max-hole DP 负责全局剩余覆盖，后接 official-derived AdaTAD/ActionFormer。当前 Job
`1178989` 仅处于 V8/P0 负诊断封存阶段；第一候选粗动作 macro AUROC 达 `0.624512`，
但旧 policy transition AUROC `0.521321` 低于 simple delta `0.553237`，且边界覆盖/距离
不胜均匀采样，第二候选仍在运行。后续固定顺序为 R0 Oracle 可达性、R1 实现门禁、
R2 P0 三臂机制、R3 U/G0、R4 alignment 后 G1/G2、R5 三种子/K384-K256/第二 backend/
完整成本。当前不再发起开放式 Pro 发散；只在 R0 参数冻结、R1 exact-commit 长训前和
R4 hard-swap alignment 三处做有界审查。
type: query_pack
updated: 2026-07-22
max_chars: 8000
---

## 2026-07-21 DUCA two-stage curriculum

The bounded successor is branch `codex/duca-two-stage-curriculum-20260721` at
exact commit `6f2ed48d8cb31fe984b8a20223a0624fcf95d4b9`. P0 genuinely skips the
entire VideoMAE/AdaTAD path and trains only the low-cost RGB/official-ASFormer
coarse branch plus transition/boundary selector for 20 epochs on a sealed
training-only 80/20 split. The official-60 stage then spends its first 1000 of
6000 successful updates warming AdaTAD on canonical exact-uniform K=384 while
all frontend and detector-bridge weights are zero; this adds no detector
epochs. It compares exact uniform, scratch joint, P0-pretrained joint and
P0-pretrained frozen-coarse arms.

Remote evidence is `83 passed`. Initial Job `1178480` failed before training
because a frozen AdaTAD backbone optimizer field leaked into AdamW. Exact
commit `6f2ed48` fixed and regression-guarded that config contract. Replacement
Job `1178487` built the optimizer and entered P0 epoch 0. Step 20 had finite loss 0.8680, exact K=384, zero
detector loss/bridge and an explicit skipped-detector audit. At 12:31 +08:00
the shared `/data` filesystem reported 100% use and the P0 log had stopped
advancing after epoch 1. Accounting sealed it `FAILED/1:0` at 12:33:31 with no
checkpoint, so it is an infrastructure failure rather than model evidence.
There is no selected frontend checkpoint, terminal
mAP, greater-than-65 evidence or paper-ready claim.

After storage recovery, the exact `6f2ed48` parallel-DAG precheck passed but
Slurm rejected atomic submission with `AssocMaxSubmitJobLimit`. The submitter
rolled back its held jobs. Protocol-equivalent serial Job `1178591` is now
running from fresh root
`duca_twostage_6f2ed48_serial_20260721_133422`; its split-manifest SHA-256 is
`be84b85a...2118a`. It restarts P0 because `1178487` left no checkpoint. The
first candidate reached epoch 8/step 700 with finite frontend losses, requested
K=384, zero detector loss/bridge and the detector path explicitly skipped.

An exact-commit static Pro audit, independently checked against the clean
remote `6f2ed48` snapshot, places this implementation on protocol HOLD. P0
inherits hidden nonzero loss defaults; transition supervision reaches ASFormer
hidden features; the official warmup does not isolate frontend AdamW/clip/EMA
state; and the direct bridge is not the true discrete hard-swap utility. Job
`1178591` is therefore diagnostic-only and cannot unlock paper-grade
official-60. The leading repaired proposal anchors exact uniform and robust
pure delta with a zero-init bounded local residual, isolated optimizers and a
detached legal hard-swap teacher. Its radius and numeric kill thresholds are
not yet frozen or empirically supported.

A second exact-commit audit agrees on HOLD and the bounded local-residual
family, but proposes a local-cell soft-RGB backward bridge instead of the first
audit's detached hard-swap teacher. They are not identical recommendations.
The next approved sequence is a no-training local reachability/oracle audit,
strict P0 loss/gradient/padding repair, and a real hard-swap alignment gate
that chooses one detector-feedback mechanism. Official training keeps the
coarse model frozen and compares matched exact-uniform, pure-delta, residual
without feedback and residual with the approved feedback. Terminal mAP and
decode-to-output cost remain decisive.

The local-residual repair is now implemented on branch
`codex/duca-local-residual-20260721` at exact commit
`6c56e11a5dd90196bf09bd0673b548fd06bd321e`. It adds matched `U/D/R0/R1/R1+UC`
official-60 configs: exact uniform, detached `abs(delta p_action)` pure delta,
bounded local residual without detector feedback, bounded local residual with
`local_cell_straight_through`, and a training-only 50% uniform companion arm.
The serial P0 launcher now reopens train/holdout block paths from the split
manifest and propagates `DUCA_FRONTEND_ONLY=1` into sbatch. Remote academic-
accelerated shallow clone succeeded; Linux focused evidence is `87 passed/2
skipped` at `56c2683` and `19 passed` plus submit precheck at `6c56e11`. P0
frontend-only Job `1178863` is pending under
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_local_residual_6c56e11_p0_20260721_175500`.
Status is `experiment_running` only; no P0 winner, official-60 mAP, or
greater-than-65 claim exists.

## 2026-07-21 remote checkpoint retention

The full `yuzibo` tree was reduced from 909 numeric training checkpoints to
220 CRC-validated records, exactly one per independent directory. A total of
689 files/334,791,638,367 bytes was removed; `/data` recovered from 100% use
to 87% use with about 310 GiB free. Datasets, pretrained weights, environments,
logs, configs and result artifacts were not deleted. The consolidated manifest
SHA-256 is `a06d3062...04b60a`.

The interrupted selected-axis four arms are recoverable from their retained
`epoch_24.pth` files. All four load on CPU and include model/EMA, optimizer,
scheduler, scaler and RNG states. This proves resumability only, not terminal
mAP or the greater-than-65 claim.

## 2026-07-21 DUCA selected-axis official-60 live evidence

The exact pushed implementation is commit
`cb89586a92b8b0a8349ecc9551bc50aa97982360`. It freezes offline-TAD
selected-axis K=384 training at 60 epochs/6000 successful updates/seed 3407,
terminal epoch-59 EMA only, and reopens the exact gate/config/pretrain/data
hashes at runtime. Clean Linux evidence is 38 focused tests plus 23 required
C3/ASFormer regressions. Real full-model gate Job `1177776` completed and
sealed suite SHA-256 `76628abd...0a27`; gradient ownership, hard positions,
max-gap, forced AMP replay, optimizer, scheduler and EMA all passed.

The admissible same-commit arms were `1177779` exact-uniform, `1177780`
direct-0.25, `1177781` uniform-to-learned homotopy-0.25 and `1177782`
homotopy plus one-pass 50% uniform companion. All four were interrupted around
epoch 26--27 by `Disk quota exceeded` after `/data` reached 100%. This is a
shared infrastructure failure, not a matched terminal result; no terminal EMA
or mAP exists. Before the storage failure, logs showed finite cls/reg losses,
bounded AMP replay and the bridge ramp beginning after step 2100.
All four newest complete records are retained at epoch 24 and are true resume
states; epochs 4/9/14/19 were removed after validation. Any continuation must
use exact commit `cb89586` and the frozen arm configs.

Resume gate `1178581` completed and sealed gate/resume SHA-256 values
`ef9ab397...6974` / `614871a1...e125`. Jobs `1178582-1178585`,
`1178614-1178617` and `1178633-1178636` are immutable pre-restore launcher
failures caused respectively by missing commit export, missing canonical
`BASE`, and missing gate-suite environment. None made an optimizer update.
The admissible current root is
`duca_selected_axis_cb89586_resume_e24_v4_20260721_135701`; runtime-binding
preflight SHA-256 is `dae27758...09d1`. Two-GPU/two-wave Job `1178642` is
RUNNING. Exact-uniform and direct-0.25 restored epoch 24, completed epoch 25
and entered epoch 26 with selector step 2599; the two homotopy arms are the
ordered second wave. The only admissible endpoint remains epoch-59 EMA and no
terminal mAP exists yet.

Epoch-4 selector-quality diagnostic Job `1178004` completed successfully over
64 matched validation windows without executing the heavy AdaTAD backbone or
selecting a checkpoint. Coarse actionness remained weak (pooled AUROC
0.463--0.469; AUPRC 0.255--0.257). Learned policy transition AUROC at radius 1
was 0.527--0.530, below pure `abs(delta p_action)` at 0.611--0.618. Relative to
exact uniform, the three learned policies improved exact boundary recall r0 by
0.0091--0.0139 but reduced r1 recall by 0.0924--0.0984 and slightly increased
mean endpoint distance/max hole. This is early negative mechanism evidence,
not terminal mAP and not a checkpoint-selection rule. Terminal Avg-mAP and the
greater-than-65 claim remain unproven.

An independent coordinate audit reproduced exact-uniform AUROC/AUPRC
`0.463353/0.255005` from all 64 records. All 64 GT windows and valid lengths
matched original THUMOS annotations, and formal action targets matched the
diagnostic labels at all 46,527 candidate positions. Action mean p_action was
0.458222 versus background 0.459672; +/-1-candidate label shifts changed AUROC
by less than 0.002. The weak coarse score is therefore a real early
optimization result, not a stride/window/GT-coordinate bug.

A follow-up hard-set audit excluded the three windows with `valid_len<=384`.
On the remaining 61 windows, epoch-4 direct/homotopy/companion policies retained
only 51.4%/51.7%/51.8% of exact-uniform positions and replaced about
186.5/185.6/185.0 of 384 frames. Adjacent-selection rate jumped from uniform
4.4% to 35.2%--35.5%. These are inference-endpoint `alpha=1` positions, not
the epoch-4 training hard inputs: the exporter calls `eval()/forward_test`,
and every learned record explicitly stores `policy_mix_alpha=1.0`. Exact
replay shows all audited samples remain uniform through alpha 0.1; the first
observed changes begin at alpha 0.3 (mean first-change alpha about 0.34). Thus
the endpoint scorer can cluster strongly but is not yet boundary-aligned;
the earlier claim that alpha about 0.03 had already replaced half the path is
invalidated.

Read-only hard-trajectory tooling is now pushed on branch
`codex/duca-selected-axis-diagnostics-20260721` at exact commit `87cfd20`.
It scans frozen scorer logits over alpha, compares soft occupancy with hard DP
swaps/Jaccard, and reports short-window/gap freedom without constructing
AdaTAD. Clean remote focused regression is 47 passed/2 skipped. Epoch-4
three-arm trajectory Job `1178357` is submitted; it is diagnostic only and
cannot select a checkpoint or mutate the four formal runs.

## 2026-07-21 DUCA Uni-companion submission status

The active optimization implementation is the isolated branch
`codex/duca-uni-companion-inputfix-20260721` at exact commit
`4d84acda4d073fb6aac956c21386df8ed5d4d2f5` and tree
`b15a064784f25d888cc66df01c39781422403195`. The clean N16R4 snapshot
passes 67 focused tests plus 23 required C3/ASFormer regressions. Exact P0 is
frozen at file SHA-256
`eabc6da8c3cc4308b70a8c8d6bbecc6c6e4b408cb17d2ee6041ed83f24a4eb3f`:
batch size 2, 100 loader steps, 60 epochs, 6000 successful updates per arm,
and 48 P3 windows/576 legal swaps.

Status is `deployment_failed_before_model_runtime`. Gate Job `1177696` failed
at elapsed `00:00:00` with exit `127` because its non-login Slurm shell called
`module load` before initializing `/etc/profile`. This is a launcher defect,
not a CUDA/model result. Dependent Jobs `1177697-1177699` were
`DependencyNeverSatisfied` and then cancelled without training. Watchers
`808310/883230/933605` exited and produced no Job IDs. Therefore commit
`4d84acd` still has no optimizer update, checkpoint or mAP evidence.

The isolated successor branch `codex/duca-uniform-homotopy-20260721` now
implements a hard-forward exact-uniform warmup followed by a successful-step
cosine transition to the learned physical exact-K policy. It also fixes Slurm
environment initialization and adds a same-commit four-arm suite:
exact-uniform, direct bridge 1.0, bridge 0.25 and homotopy bridge 0.25. Its
status is `implemented_draft/tested_static`; it has no exact commit, P0,
CUDA gate, training Job or mAP until the new evidence chain is completed.

The old `d748684` Jobs `1177687/1177690/1177691/1177692` were cancelled at
zero runtime after a real-loader audit proved an input-contract blocker:
THUMOS emits uint8 RGB while the old selector rejected non-floating input.
Commit `4d84acd` preserves exact hard gather and promotes only the
differentiable soft bridge to FP32. It changes no official detector or
unrelated route. The old jobs are not evidence.

No CUDA/P3 authorization, optimizer progress, checkpoint or mAP exists yet.
Greater than 65 Avg-mAP remains the GO criterion, never a promised result.
One implementation risk is now explicitly registered for diagnosis: learned
arms instantiate `DucaProtectedTransitionScorer` without zero-initializing its
output head, while the physical global exact-K Viterbi tie-break is
lexicographic rather than exact-uniform. Therefore the learned hard policy
does not have a proven exact-uniform initialization. An eight-window
real-loader audit found mean uniform overlap `0.502604`, mean rank error
`4.038737` frames, and no aggregate boundary-distance deficit versus uniform.
Do not alter the sealed `4d84acd` suite post hoc; measure its trained geometry
and terminal mAP, then decide whether a separately frozen
uniform-initialized-successor experiment is justified.

## 2026-07-20 pre-backbone 设计与论文准备度裁决

Protected-E2E 是合理但未证实的研究假设；当前正确的是 P0-P3 加四臂
official-60 的证伪顺序，不是“方法已正确”。组件已有 `24 passed` focused
evidence，但 full-model P0-P3、终点 mAP、重复种子和 full-stack cost 均未闭环，
所以 C1/C3/C4/C7 不变，论文仍 `NOT PAPER READY`。

新增最高优先级风险是 backbone 时间语义：非均匀 hard 帧按 selected rank
重新打包进 VideoMAE 16-frame clip/tubelet，physical ActionFormer head 只能修正
backbone 后的 proposal 坐标，不能自动修正已经按规则时间位置产生的特征。P1
必须加入正式 384-frame chunk/feature/mask build、uniform-384 end-to-end parity、
短动作 support、时间间隔反事实和 raw-gather-to-head roundtrip；失败即先修订
representation contract，不得启动长训练。

same-DAG 与 hard-forward equality 仍不能证明 soft backward 对齐真实 hard-swap
detector utility；已冻结的 P3 是 direct-gradient claim 的必要门。固定 K=384
只能称内容自适应位置分配，不是 dynamic budget。成本必须含 dense decode、
preprocess、H2D、全 768 帧 coarse probe、selector、backbone、head、NMS、能耗与
显存。完整评审见
`docs/methods/reviews/2026-07-20-duca-prebackbone-design-paper-readiness-review.md`。

## 2026-07-20 worktree isolation

There are 37 top-level local `OpenTAD_*` directories; adding the isolated
nested DUCA construction clone gives 38 relevant Git trees in the audit.
There are 19 registered worktrees, including one detached Codex inspection
tree outside `E:\DeskTop\TAD`. Nested historical copies are deliberately
excluded. Independent cross-tree audits found no already-complete frozen DUCA
implementation.
The Spatial-Zoom primary tree is dirty and is not the DUCA implementation
surface. Protected-E2E work is isolated in
`.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720` on
`codex/duca-physical-protected-e2e-20260720` from `b3222af`. Only the
Allocation-Ceiling physical solver, Protected-E2E ASFormer/gate machinery,
PhysTime physical-head tests and TrueTime gradient-proof patterns are eligible
for narrow reuse. SparseHead, Spatial-Zoom, ChronoTransport and historical
routes are read-only. See `worktree_inventory.md`.

## 2026-07-20 Protected-E2E single-route freeze

The bounded sequence remains P0 protocol, P1 implementation, P2 loss-source
gradient ownership, P3 soft-gradient versus real hard-swap alignment, then
only one four-arm official-60 matrix: exact-uniform,
transition-no-bridge, protected-E2E and protected-E2E-rho. No X3D, SlowFast,
MobileNet, MUST, dynamic budget, new detector or new dataset is admissible.

The Pro adjudication (`f91db53a...97ccb0`) freezes a stricter architecture:
selector adapter/head; main detector gradient stopped before ASFormer/action
head; rho 0.01 only into the last ASFormer block; hard Viterbi and soft Gibbs
slot marginals from the same physical exact-K DAG; native physical
target/decode/NMS/evaluator and no selected-axis remap. Counterfactual teacher,
utility distillation, local bridge, soft legality, post-hoc repair, homotopy
and detector-gradient ramp are disabled.

The reviewer read only `280631a`, so its claim that no implementation or real
detector gradient existed is stale. Commits through `b3222af` and Job
`1176948` passed real full-model main/rho P1/P2 connectivity, but that
candidate is nonconforming: local slope surrogate, candidate-hole `G=2`,
selected-axis GT, rho 0.05, bridge 0.25 ramp and a 4x8 trained-checkpoint P3.
P3 stopped before statistics on a stale manifest-field requirement. No mAP or
official-60 authorization exists.

The review's seed 3407 and 48-window/576-swap P3 are accepted as the new
bounded gate. Its claimed 99 steps/epoch and 5940 updates are not yet facts:
P0 must derive and hash exact loader length; historical 200-video/batch-2
runs used 100 steps. Protected-E2E remains `designed`.

The isolated branch has now advanced one component from `implemented` to
standalone `tested`: the shared physical exact-K hard-Viterbi/soft-Gibbs graph
passed a remote exhaustive focused suite (`9 passed`) on 2026-07-20. This does
not advance the full method beyond `designed/partially_implemented`; selector
integration, native-time detector semantics and P0-P3 remain unproven.

An uncommitted protected-selector integration draft now exists in the same
isolated tree. It adds the explicit 197-to-64 adapter/head, fixed coverage
floor and four arms, keeps dense GT unchanged, and implements exact-hard
forward with soft-Gibbs backward. Local compile passed; the integrated remote
tests were not run before interruption. ActionFormer RNG/optimizer contracts,
strict physical-head validation, configs and P0-P3 remain missing. Status is
`implemented_draft`, not `tested`, and no experiment is authorized.

The interrupted integrated test was subsequently completed on the disposable
remote copy: after correcting one test-only fake-ASFormer indexing typo, the
physical-DAG plus four-arm selector suite passed `14/14`. Selector status is
now `tested_focused`, not full-model tested. Independent P0-P3 audit remains
`HOLD`: official configs/validator are still selected-axis, train constructs
val/test loaders, loader exposure is not hash-derived, protected optimizer/RNG
contracts are incomplete and P3 is stale.

## 2026-07-20 Allocation-Ceiling solver repair and replacement boundary

The bounded training-side Allocation-Ceiling diagnostic is implemented and
pushed at exact solver-repair commit
`8ebdd2a11ea5cc0644979324872a3b1cae5a2170` on
`codex/duca-allocation-ceiling-20260720`. It is not selector training and not a
final DUCA method. It tests whether global exact-K physical-gap allocation has
privileged boundary headroom, whether frozen deploy-visible transition scores
recover that headroom, whether the frozen official AdaTAD/ActionFormer
physical-grid path benefits in detector loss, and whether the exact family-D
solver is computationally plausible.

The first exact `b18dd8f` gate Job `1174706` failed because its timeline
contract incorrectly treated decoder FPS and THUMOS annotation FPS as the same
clock. Descendants `1174707-1174710` were cancelled without runtime. Auditing
all 200 training videos established exact decoded/annotation frame-count
agreement for every video, while FPS-clock drift had median about 3.00 and
maximum 3.69 frames. Commit `1d51379` now uses decoded frame index as the
canonical axis, fails closed on frame-count mismatch, and records FPS-clock
drift only as an audit diagnostic.

The superseded `1d51379` formal root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_allocation_1d51379_training_20260720_041247`
where gate `1174711` and full export `1174712` completed. The 670-window
deploy-visible recoverability artifact validates, but GT diagnostic Job
`1174713` failed on the first GT32 sample because a roughly `1e-12` binary
residual was multiplied by a `2^29` lexicographic weight. Jobs `1174714` and
`1174715` were cancelled at zero runtime. This is an implementation failure,
not negative method evidence.

Commit `8ebdd2a` canonicalizes integer variables, requires every subproblem to
report `OPTIMAL`, numeric zero MIP gap, finite objective and dual bound, derives
all scientific objective values exactly from selected positions, and rechecks
all pinned objectives at termination. Tiny-residual, material-residual,
nonzero-gap, terminal-swap, multi-block and upper-envelope attacks are covered;
local focused contracts are `55 passed` and independent review is
`P0=0/P1=0/GO`; the clean Linux snapshot passed `111` relevant tests. Exact
precheck passed, and real failed-sample replay Job `1175393` completed
`0:0` with both privileged solvers and independent artifact replay valid. The
replacement formal root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_allocation_8ebdd2a_training_20260720_1320`
with Jobs `1175395-1175399`. Exact gate `1175395` completed `0:0` in 4:02;
its artifact SHA-256 is
`6030d9fb7110aa7c73b2df244eff50136d1342c5e2e90bd86db485d38faafc61`.
Solver replay, candidate-loss, solver-cost, submission and scheduler
validation all report `validation_passed=true`. All five Jobs
`1175395-1175399` subsequently completed `0:0`; final evidence SHA-256 is
`8232f2f0889bc5e0579abcf82d42ab4009397366c5c4b0e6bfd71d0c658ad6d6`.

The result is a registered warning, not a runtime success or route KILL. On 670
matched training windows, deploy-score endpoint distance is `0.508843` versus
uniform `0.464399`, and radius-1 both-endpoint recall is worse by `0.151970`.
Transition-policy AP/F1/AUC is only
`0.219662/0.304830/0.579192`. On the exact 32-window subset, privileged GT
does expose modest geometry headroom (`0.446381 -> 0.238764` mean endpoint
distance), but frozen detector loss is worse for deploy score, constrained GT
and unrestricted GT: `0.275252/0.318532/0.487739` versus uniform `0.224627`.
These facts show proxy and loss misalignment, but frozen training loss cannot
replace decoded/NMS final mAP. The current
transition-score/global-allocation/frozen-detector contract is HOLD pending one
hash-bound official mAP replay of exact uniform versus deploy-visible
transition selection under the same checkpoint and post-processing.
Privileged GT must not enter the deployable replay. C3/C4/C7 and paper
readiness remain unproven.

A bounded detector-utility Pro audit prompt was first published at
documentation commit `706e23b` and corrected at `db11aee` on the same branch.
Its immutable code-review target remains implementation commit `8ebdd2a`; the
documentation commits are not model changes. The corrected prompt makes
frozen loss diagnostic-only and specifies one hash-bound deployable mAP replay
as the next task.

## 2026-07-20 CellCF KILL and physical allocation-family revision

Exact model commit `1642f26` and evidence commit `4ce69c8` remain the audited
facts. One-per-uniform-cell CellCF cannot transfer quota across regions,
gathers actual content under uniform detector geometry and uses detached
counterfactual supervision. With seed-0 CellCF `64.0610 < 64.2755`
transition-beta0, it remains `tested_diagnostic`, not a boundary-adaptive main
method.

The fixed `G=3,192 scaffold + 192 residual` CARA proposal is withdrawn. If
"15 frames" means original decoded frames, the formal stride-4 grid has
effective cap 12: exact uniform is feasible, but a fixed scaffold needs at
least 255 arbitrary points or 382 exact-uniform-subset points. The revised
primary ceiling is global exact-K under a frozen physical maximum interval;
scaffold/residual is post-hoc decomposition only.

The broader offline DUCA route remains `REDESIGN`. Physical unit/value is
still unfrozen; the supplied solver code is unintegrated; paired statistical
gates need correction; and current val/test share THUMOS validation. Run no
long training before the exact family, deploy-visible recoverability,
physical-grid detector and dense full-stack cost gates. Dynamic MUST remains
frozen; C3/C4/C7 and paper readiness remain unproven.

## 2026-07-18 DUCA-CellCF current evidence closure

The immutable `1642f26` exposure-132 arms finished `COMPLETED/0:0`.
Terminal-EMA Avg-mAP is `63.8594` for exact-uniform, `64.2755` for
transition-beta0 and `64.0610` for CellCF. Transition-beta0 is `+0.4161`
percentage points over uniform; CellCF is `-0.2145` below transition-beta0.
These are one-seed raw facts, not sealed claims. Aggregate Job `1167484`
completed. Original cost Job `1167485` is immutably `FAILED/1:0`; original
completion Job `1167486` is immutably cancelled and must not be reconstructed
as successful.

Evidence-only commit `e153c96bfa0f37b9d4b82046e05b1bbce70dfe50`
passed 230 exact Linux CellCF tests plus compile, Bash and clean-tree gates.
Recovery root `cost_recovery_e153c96_v1` is now also failed diagnostic
evidence. Job `1170366` failed before publishing cost because the profiler
emitted seven `*_cpu_enqueue_ms` fields that the strict summary schema rejected
as unsupported; dependent Job `1170367` was cancelled without runtime. Failed roots
`cost_recovery_5ab3042_v1` and `cost_recovery_67a8a0a_v1` remain immutable
diagnostics. C3/C4/C7, dense full-stack savings and paper readiness remain
unproven. Do not claim current CellCF improves transition-only selection.

## 2026-07-17 post-run evidence hardening gate

Post-run tooling now advances to exact commit
`9e96967a158534b014aacde57c1b78bd1591e71a` on
`codex/duca-cellcf-evidence-20260717`. It supersedes `787569e`, whose Linux
tests exposed a real transient swap/restore gap in terminal stat-only
validation. The replacement establishes Linux inotify monitoring before each
evidence read/hash, binds the watched directory inode through an
`O_DIRECTORY|O_NOFOLLOW` descriptor, revalidates identities, and closes the
monitor on every exit path. Independent max review returned code-level GO.
The clean snapshot
`/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_evidence_9e96967_20260717`
passed `14` finalizer tests and `253` broad evidence tests with pytest
temporary files located on the actual `fuseblk` `/data/run01` mount. This
includes an independent-process transient replacement test. The repaired
post-run DAG is not yet submitted: formal cost Job `1167485` is still running,
and completion Job `1167486` remains dependency-pending.

## 2026-07-17 DUCA-CellCF evidence tooling continuation

The immutable formal training remains on model commit `1642f26`; it is not
relabelled as a newer model run. Post-run protocol and cost tooling is now
implemented and tested on branch `codex/duca-cellcf-evidence-20260717` at exact
commit `2a0f848f7dbf17b7bcb40aa7a996954e8f87c4de`. The clean Linux snapshot
`/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_evidence_8327c2f_20260717`
passed 303 tests with three skips. This includes explicit `exposure132` and
`official60` profiles, profile-aware six-job names and manifests, prepared
suite reopening without scheduler calls, path/symlink fail-closed checks,
epoch-59/89/131 convergence inspection, raw full-stack samples, raw Slurm
allocation replay, non-overwrite evidence publication, and training/inference
break-even inputs. These tools are `implemented + tested`; no new convergence,
allocation-cost, official-60 training or paper result has run yet.

## 2026-07-17 DUCA-CellCF current exact state

This section supersedes the older CellCF deployment notes below. The only
current candidate is exact commit
`1642f265e48391418a7c8a4a087e33e2b7bf6899` on
`codex/duca-cellcf-20260716`, clean snapshot
`/data/run01/sczc063/yuzibo/projects/opentad_duca_cellcf_1642f26_20260717`.
Linux verification is 212 passed/3 skipped plus 23 required C3 regressions.
The synthetic gate SHA-256 is
`3dd4750cc97d0287b647125264a5495626cb87df6aec6b099b4aed48a523e5cd`.
Real-loader CUDA gate Job `1167479` completed `0:0`; artifact SHA-256 is
`3d630a323e79c694f663c31151c070fd46943296937ceafdd5f9bcacfcbd7cde`.
DDP pilot Job `1167480` completed `0:0`; independently reopened artifact
SHA-256 is
`8e6a59e92f12b15ec1e7c3671104959c0533c9ba9b68dd36550c0294c8b48cd3`.
All three arms reached 10/10 successful optimizer/EMA/scheduler/selector
updates with full/mixed/all-short coverage and no unseen gradient group;
CellCF had nonzero distinct-cell utility on 9/10 steps.

Formal root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200`
is `experiment_running`. Jobs are `1167481` exact-uniform, `1167482`
transition-beta0, `1167483` CellCF, `1167484` aggregate, `1167485` cost and
`1167486` completion. The dependency DAG and all six scheduler-stored script
hashes were independently reopened. The three arms are progressing normally.
At the 16:45 CST audit, exact-uniform, transition-beta0 and CellCF had started
epochs 125/123/114. Checkpoints exist through epochs 124/119/109 at the
unchanged five-epoch interval.
AMP replay counts are 5/6/5, each isolated at replay 1/8
with no exhaustion. Requested K remains 384; logged effective-K means span
190--384 because short valid windows cap K. CellCF's logged counterfactual
distillation loss remains finite and nonzero (latest 0.1664), which proves only
that the objective is active. Slurm stderr is empty, every logged loss is finite
and there is no Traceback/OOM/ValueError/non-finite-loss/FAIL hit. No terminal
mAP or cost result exists. C3/C4/C7 and paper readiness remain unproven.

Training-budget interpretation is frozen as follows. The current
`epoch_131.pth` means 132 epochs and exactly 13,200 successful updates. It was
chosen to match the historical separated-training exposure near 13,080 updates
and remove the old 5,940-update undertraining confound. It is therefore a
fully-trained matched diagnostic, not automatically the paper's primary
training recipe. The repository AdaTAD base ends at 60 epochs, so a paper
efficiency claim also requires a same-commit 60-epoch schedule for
exact-uniform, transition-beta0 and CellCF, plus training GPU-hours, peak
memory, inference savings and break-even analysis. Existing epoch 59/89/131
checkpoints may be evaluated only as a predeclared convergence trajectory;
epoch 59 inherits the 132-epoch LR schedule and must not be called an official
60-epoch run or used for checkpoint selection. The live 132-epoch suite must
finish unchanged. If CellCF gains only at 132 epochs and not under the
60-epoch/equal-compute contract, the efficient-training claim is unsupported.

The `522925e` formal root and Jobs `1167475-1167478` are permanently invalid:
all were cancelled with zero runtime after live Slurm rendered canonical
`afterok:a:b:c` as repeated comma-separated
`afterok:id(unfulfilled)` tokens. Commit `1642f26` accepts only that strict live
rendering while preserving canonical SubmitLine and predecessor accounting
proof. Earlier `4bf6485` formal Jobs `1167469-1167471` are also invalid and
zero-result diagnostics.

## 2026-07-16 DUCA-CellCF implementation state

The current admissible Round-2 CellCF implementation is exact commit
`b8cd29f621d410b720f12380b3095dd39574e01f` on
`codex/duca-cellcf-20260716`. It implements one-frame-per-exact-uniform-cell
acquisition, transition-first local deformation, separate acquisition and
fixed detector coordinates, detached distinct-cell hard-flip utility, three
matched fixed-K arms, terminal-EMA evidence, and a mandatory trained-checkpoint
frontend cost/completion DAG.

The old exact commit `3a0f5ae` passed real-loader gate `1167222` and DDP pilot
`1167227`, but its first formal submission exposed a P0 transaction defect:
nested Bash substitutions wrote null dependent receipts. Only arm Jobs
`1167234-1167236` existed; they were cancelled after 95 seconds, and no
aggregate/cost/completion job existed. That suite is permanently inadmissible.

Commit `b8cd29f` closes that defect. Receipts require canonical positive
`jobid;cluster`, exact scheduler-side dependency evidence, durable file and
parent-directory fsync, and predecessor success/time-order proof when Slurm
removes satisfied `afterok` IDs. Independent max review converged to `GO` with
P0/P1=0 after five adversarial rounds. Local checks are 127 passed/5 skipped
plus 23 C3 regressions. The clean accelerated snapshot
`opentad_duca_cellcf_b8cd29f_20260716` passed 155/3 skipped plus 23 C3
regressions. Its exact synthetic gate passed with SHA-256
`9606f6325e05767e7b748b85e73352cdc52a439b382541a4dd5ef66ca855a76f`.
Real-loader CUDA gate Job `1167345` then completed `0:0`; the independently
reopened artifact SHA-256 is
`c4f6b5ce7d2bb830236ee51cef6d2b5ac5965bd4b84811a12cb2e86eb039b673`.
The first wrapper-only retry `1167338` failed before Python because its
non-login shebang could not resolve `module`; it is deployment-diagnostic only.
Gate-bound DDP pilot `1167348` completed `0:0`: all three arms reached 10/10
successful optimizer/EMA/scheduler/selector updates, replayed one forced AMP
overflow, covered full/mixed/all-short batches and all parameter groups; CellCF
had nonzero distinct-cell utility on 9/10 steps. Pilot SHA-256 is
`572e47440c54da558f6320148549de8fd62204d0f524b410f53400fe02249270`.
The first formal submission then fail-closed on the second arm because this
cluster exposes an empty pending `sacct Comment` despite preserving the exact
token in live `squeue Comment` and `sacct SubmitLine`. Jobs `1167359/1167360`
never ran and were cancelled; the suite is invalid. A fail-closed SubmitLine
fallback is under exact-commit repair. No 132-epoch result, terminal mAP or
formal cost result exists, so C3/C4/C7 remain unproven.

## 2026-07-16 DUCA Round-2 REDESIGN verdict

The method/paper reviewer selected one route: exact-uniform-anchored,
one-frame-per-cell local deformation with deploy-visible transition evidence
and detached hard local-flip detector utility. The project accepts the core
coverage-preserving redesign and advances DUCA-CellCF to `designed`, not
implemented or supported. It does not fully accept the response: TAPS was
incorrectly replaced by TAPOS; detached utility cannot reuse the direct-
gradient C4 claim; local-cell only supports uniform residual learning, not
global budget allocation; fixed weights/schedules/numerical publication
thresholds are proposals; and one seed can kill the registered configuration,
not scientifically refute every DUCA hypothesis. No training is authorized
before local-cell implementation/tests, an exact-commit synthetic gate, a real
THUMOS loader/DDP/AMP-replay gate and a mixed-batch pilot. C3/C4/C7 remain
unproven.

## 2026-07-16 `7525efb` Round-1 Pro audit

The exact-commit code/math reviewer returned `GO_TO_REAL_GATE`, with no static
P0 model blocker. This is not permission for pilot/full training and creates no
performance evidence. The review confirms that the current route uses
detached hard-swap selection-policy utility rather than direct detector
gradient, reuses official AdaTAD components through a materially extended
wrapper, and has a locally correct signed score-space proximal direction. Its
utility includes selected-axis geometry and reassignment, not pure RGB-frame
content. The exact commit still has no real-loader CUDA gate and its AMP/DDP
contract remains unresolved. Before any pilot, fix the all-zero/single-value
utility alignment false-positive, freeze the utility provenance/name, and run
a fail-closed real THUMOS loader gate through actual DDP, `train_one_epoch`,
AMP replay, EMA and schedule. Status remains `tested`; C3/C4/C7 remain
unproven. Raw review and absorption are registered in `source_registry.md`.

## 2026-07-15 current DUCA signed-utility candidate

The current exact GitHub implementation is
`7525efb2e07214615a59c482443246174a6adaf1` on
`codex/duca-transition-only-20260711`. Gate `1165646` invalidated the old
candidate-relative counterfactual loss. The replacement uses a Gram-whitened
signed score-space proximal objective so hard-swap detector utility
`L_baseline - L_swap` determines the local update of the actual selector center
scores even when swaps share a removed frame. It also closes known AMP,
invalid-index, mixed-batch, all-short, RNG and dirty-tree/hash audit defects.
Clean remote focused tests: `160 passed, 7 skipped`. No exact-commit CUDA gate,
real THUMOS loader gate, forced-overflow pilot, replacement four-arm train or
mAP exists. Status is `tested`; C3/C4 remain unproven.

## 2026-07-15 successful-update replacement suite

The only admissible DUCA P0 implementation is commit
`a6903ae036d7b4bfd0c25752c51f020b20427fff`. It implements exact AMP
same-batch replay/state rollback, 13,200 successful optimizer/LR/EMA/selector
updates, five-epoch checkpoints, terminal epoch-131 `state_dict_ema`, official
OpenTAD mAP recomputation from the prediction JSON, and idempotent
cluster-bound Slurm receipts. Local verification is `80 passed, 3 skipped` and
final independent blocking review is `GO`. Status is only
`implemented + tested`: fresh exact-commit CUDA gate and forced-overflow
four-arm pilot must pass before long training. Jobs `1164700-1164703` remain
protocol-invalid diagnostics; C3/C4 remain unproven.

## 2026-07-15 Active-task correction: Spatial Zoom only

Superseded for the current turn by the user request on PhysTime sparse
downstream detection heads. Spatial Zoom remains a separate route and must not
be mixed with the current PhysTime SDPQ implementation/deployment.

## 2026-07-15 Current turn: PhysTime SDPQ sparse detector head

The active task is to refactor the downstream detection head so **physical
queries** and **sparse observation support** are fully decoupled. This is a
PhysTime sparse-head route, not DUCA selector work and not Spatial Zoom.

Implemented and pushed on
`codex/phystime-performance-diagnosis-20260712` at commit
`372fcbf58d1b2eb895b724f6f040458bde4d636e`:

- `SupportDecoupledPhysicalQueryHead`: complete physical query grid, sparse
  support evidence, signed center/width regression, assignment without requiring
  query centers to lie inside GT.
- `PhysTimeMeasureProjection` support for keeping uncovered physical queries
  with learned null evidence.
- Raw-video G1b native-J192 config
  `configs/adatad/thumos/phystime_g1b_sdpq_pool_native_j192.py`, explicitly
  avoiding post-backbone temporal interpolation.
- Real THUMOS gate and pilot Slurm scripts for `K=384`, `J=192`, SDPQ head.

Remote clean snapshot:
`/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1b_sdpq_372fcbf_20260715`.
Remote focused verification passed: `41 passed in 49.38s`.

Current pilot result:
run root
`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1b_sdpq_372fcbf_20260715_205352_+0800`;
gate Job `1165340` completed with `gate_pass=true`; pilot Job `1165341`
completed successfully after 6 epochs. The gate proves real
THUMOS one-step execution, no interpolation, finite predictions, zero missing
GT assignment, and nonzero finite gradients through adapter/projection/null
evidence/classification/regression/endpoint. Pilot final mAP from `train.out`
is Avg 10.17 / mAP@0.3 23.72 / 0.4 15.22 / 0.5 7.65 / 0.6 3.26 / 0.7 1.01.
`epoch_5.pth` exists, `PILOT_COMPLETE.json` exists and reports
`validation_pass=true`, but its parser failed to capture mAP
(`best_average_mAP=null`, empty observed list). Compared with G1a matched
pilots, SDPQ is slightly below selected-axis Avg 10.26 and physical-metric Avg
10.56, while close enough to be considered `gate-passed` and weakly
`pilot-supported` as a runnable decoupled geometry. It is not superiority
evidence and not paper-ready; next work must run same-commit matched controls,
fix result parsing, and diagnose assignment/support/query-scale/NMS before full
train.

External review archived on 2026-07-16 as
`docs/methods/reviews/2026-07-16-372fcbf-phystime-g1b-sdpq-revise-before-full-train-raw.txt`
with SHA-256
`E3389D57F179BB4FFD6C1F25AC24FF1321C7865E1EBEF80BC02EF2A4E59368AF`.
Verdict: `REVISE-BEFORE-FULL-TRAIN`. Accepted interpretation: SDPQ fixes the
G1a anchor representability defect, but support-query decoupling is incomplete.
The current implementation keeps uncovered domain-valid queries alive with
position-poor/null evidence, assignment does not distinguish domain validity
from evidence validity, support overlap mass and query geometry are weakly
represented in output features, and the pilot only provides engineering
stability. Full train is blocked until same-commit matched controls and P0
fixes/diagnostics address evidence masks, coverage-aware assignment, query
geometry residuals, support context, result attestation, and high-IoU
localization/ranking failure decomposition.

The active implementation route is **Dense-Time Spatial Zoom for offline
TAD**, not DUCA: retain the complete temporal observation grid, establish
whether higher spatial resolution improves high-IoU/short-action localization,
then test oracle ROI/crop sufficiency before implementing a learnable spatial
zoom policy connected to the official AdaTAD detector. DUCA remains an
independent historical/parallel worktree and is outside this task.

The clean continuation worktree is
`E:/DeskTop/TAD/OpenTAD_SpatialZoom_S1_AuditFix_20260715`, branch
`codex/spatial-zoom-s1-audit-fix-20260715`, base `35204f5`. No actual ROI,
scout, crop policy, or fusion module exists yet; current code is the S1
dense160/224/256 falsification infrastructure only.

The agent incorrectly cancelled S1 Jobs `1164291/1164307-1164314` while
combining a DUCA/S1 audit. Their partial logs, 222 checkpoints, and 222 metadata
sidecars are preserved,
but all cells are diagnostic-only because the deterministic interpolation
contract was already invalidated. Last train epochs were 56/55/55 (160),
47/47/47 (224), and 44/43/43 (256). No final mAP, sealed test, cost table, or
S1 GO/KILL exists. The next sequence is deterministic/protocol repair -> exact-
commit CUDA gate -> fresh S1 3x3 -> cost-aware S1 decision -> S2 oracle crop
only if S1 is GO.

## 2026-07-15 S1 / DUCA STOP_AND_FIX audit

The exact-commit Pro audit is archived at
`docs/methods/reviews/2026-07-15-35204f5-043be401-s1-duca-pro-audit-raw.txt`;
the independently checked decision record is
`docs/methods/2026-07-15-35204f5-043be401-s1-duca-pro-audit-absorption.md`.

DUCA Jobs `1164700-1164703` are already protocol-invalidated for formal matched
evidence. Around epochs 24-25 their schedule steps lag theoretical exposed
batches by 3-4 successful updates; a fixed 132-epoch loop cannot reach the
declared 13,200 updates. Any terminal mAP is diagnostic only. The repair should
prefer reusing S1's state-exact same-batch AMP replay contract, with complete
mutable-state restoration and replay-exhaustion failure, rather than merely
crashing on the first skipped update. Checkpoint selection and all downstream
artifacts must be frozen and hash-bound before a replacement run. C3/C4 remain
unproven.

Spatial Zoom S1 Jobs `1164291/1164307-1164314` remain running without fatal
collapse and have recovered 31 AMP retries, but all nine cells emit the CUDA
`upsample_linear1d_backward_out_cuda` nondeterminism warning (221 occurrences
at audit). Current artifacts are diagnostic candidates, not strict formal
deterministic evidence; sealed test remains closed. Do not simply set
`warn_only=False`: first replace the interpolation with a verified
deterministic equivalent, then rerun 3x3. The class-support-conditioned
bootstrap must also be replaced and preregistered; Bayesian bootstrap is a
candidate, not an automatically accepted answer.

## 2026-07-15 PhysTime / sparse detector-head current task

The current user-directed task is sparse adaptation of the downstream TAD
detection head, not a DUCA selector/gate task. DUCA remains a separate running
experiment track and must not be mixed into PhysTime sparse-head claims,
protocols, or implementation decisions.

Completed G1a native-J192 matched pilot at commit `623a376`:
gate `1162048`, selected-axis pilot `1162049`, physical-metric pilot `1162050`
all completed successfully. Raw six-epoch results are selected-axis Avg-mAP
10.26 / mAP@0.7 1.09 and physical-metric Avg-mAP 10.56 / mAP@0.7 1.04.
This is only a weak low-IoU signal and not paper evidence. It proves final-only
checkpoint and native-J192 matched pipeline stability, not high-IoU benefit.
Follow-up geometry diagnostics at commit `f2725f5` show a concrete failure
mode: physical-time seconds reduces ActionFormer positives relative to
uniform-rank seconds. On validation, positives are 10765 vs 11030 and `<1s`
GT no-eligible fraction rises from 2.87% to 7.79%; on train sampled windows,
positives are 20437 vs 21043 and `<1s` no-eligible rises from 3.62% to 11.49%.
A rank-assignment diagnostic that kept physical decode centers but used
uniform-rank seconds for assignment was negative: validation positives fell to
7745 with 18.96% GT no-eligible and `<1s` no-eligible 65.16%; train-s5 fell to
14201 positives with 25.13% GT no-eligible and `<1s` no-eligible 72.02%. This
means assignment-axis swapping alone is not a solution. The next work must
redesign the sparse head so query anchors and sparse observation support are
separated, rather than resume DUCA selector work or long-train the current
physical-metric/rank-assignment variant.

External Pro audit on commit `b7a37f584ba7477159dd90ba08c14728c65fb19e`
confirms the same direction with a stricter verdict: KILL the current
observation-timestamp-coupled physical-anchor ActionFormer route and KILL
`physical_time + rank_assignment` as a candidate method, but do not kill
physical time itself. The replacement candidate is a support-decoupled physical
query sparse head: complete uniform physical query grid, sparse observation
support as evidence, atom-overlap pooling plus physical-relative attention,
learned null-evidence token, and signed center/width regression instead of
nonnegative left/right distances from observation centers. Required next gates
include runtime-vs-diagnostic parity, diagnostic-only lock for the rank config,
`assigned_count == 0` / support-observability diagnostics, C0-C4 matched
controls, and real THUMOS geometry / one-step / micro-overfit / pilot gates
before any paper claim.

## 2026-07-15 DUCA P0 deployment hardening status

Current exact implementation is commit
`043be401ba2b694342dc395f263e9a9858628d69` on
`codex/duca-transition-only-20260711`. Commit `18dc1cd` first isolated the
counterfactual teacher from CUDA autocast's detached weight cache; exact gate
`1164279` then passed real AdaTAD/ActionFormer detector gradients, AMP update,
K=384 and max-hole checks. That gate became stale after the next code fix.

Four-arm pilot `1164286` completed the uniform arm but failed on direct-a5's
first real mixed-length batch. The old training surrogate always ran a
T=768/K=384 soft DP and then masked/renormalized short samples, while the hard
path used each sample's real `valid_count/effective_k`. Commit `043be401`
removes that false feasible family: hard and soft paths now share the same
valid prefix, effective K and max-hole contract, followed only by zero padding.
Clean Linux focused verification is `122 passed, 5 skipped`; independent
exact-commit review reports GO with no P0/P1. Exact CUDA gate `1164318` and
four-arm 10-step DDP pilot `1164319` both completed with exit code 0. Every arm
completed 10 optimizer updates, covered full/mixed/all-short batches, and left
no trainable parameter group unseen. Formal 132-epoch seed-0 Jobs
`1164700/1164701/1164702/1164703` are running for exact-uniform, direct-a5,
transition beta=0, and transition counterfactual. A later successful-update
audit supersedes their initial deployment qualification: all four have already
missed optimizer updates and are diagnostic-only. C3/C4 remain unproven.

## 2026-07-15 DUCA `043be401` Pro audit absorption

The earlier exact-commit Pro audit returned GO for completing Jobs
`1164700-1164703` and HOLD for paper claims, but it did not inspect remote
Slurm artifacts. Its run-qualification judgment is superseded by the later
observed successful-update deficits. Its remaining code/claim analysis records
that the current
counterfactual arm is detached relative one-swap ranking, not signed utility
and not direct detector-gradient learning; its candidate-only softmax has no
no-op anchor. It also found that the current schedule first evaluates after
one-based epoch 52, not epoch 47, and that the THUMOS test split must not select
the checkpoint. For this running suite the primary result must be predeclared
as final one-based epoch 132 `state_dict_ema`; intermediate mAP is diagnostic.
The final-EMA primary protocol was sealed before any evaluation under SHA-256
`AAC0FCA8671AE6F58CF4C9B5D4D40282BE714AA354028246E86504FD39C89B48`.
The project does not accept the proposed no-op softmax as uniquely optimal,
does not require a two-rank pilot for the current one-GPU protocol, and will not
implement a physical-time head before a fixed-selection geometry diagnostic.
The jobs remain `experiment_running` only as diagnostics; C3/C4 and all cost
claims remain unproven.

## 2026-07-13 S1 infrastructure status

S1 infrastructure is `tested` (46 focused / 26 S1 tests). Formal configs bind
the frozen manifest, full-CUDA precheck, clean commit, canonical workdir,
gate evidence, 3x3 selections, one study-level test lock, ordered matched
profiling, and deterministically rebuilt results. Independent
`gpt-5.6-sol`/`max` review ended `PASS_BEFORE_REMOTE_TRAINING` with no
P0/P1/P2. CUDA full-window, training, mAP, and cost remain unrun: this is not
S1 GO or empirical support. S2/ROI/policy stay locked.

## 2026-07-15 S1 formal Slurm deployment

Spatial Zoom S1 is now `experiment_running` at commit
`35204f58fd3e91d7cf8f5888928a41e9bf6c2e72`. Fixed physical-GPU-1 binding was
removed from the S1 contract and launchers: jobs request `--gres=gpu:1` and use
the Slurm allocation as logical `cuda:0`. Full precheck Job `1164289` passed on
Slurm-selected physical GPU 4. The `9298c0e` suite is invalid: pilot `1164261`
failed its first checkpoint when legal runtime config mutations were mistaken
for protocol drift, and Jobs `1164267-1164274` were cancelled. Current pilot
`1164291` completed two epochs and produced a hash-validated checkpoint and
sidecar after two AMP replays. The remaining 3x3 jobs are `1164307-1164314`;
all nine current cells entered RUNNING without a fatal startup pattern. This is
deployment evidence only: no S1 mAP, cost, GO/KILL result, or permission to
start S2 exists yet.

## 2026-07-13 selection-quality diagnostic

Legacy beta=0 EMA epoch 89 was evaluated on 211 validation videos / 487
windows with GT used only after selection. Coarse action-state quality is only
moderate (pooled AUROC/AUPRC 0.621/0.411, ECE 0.171). Learned transition scores
are worse than the audited `abs_delta + uncertainty_peak` compound proxy at
every boundary radius; pure `abs(delta p_action)` was not separately analyzed. Against exact
uniform at matched K, learned selection gains +1.53 points exact r0 recall but
loses 15.55 points r1 recall and increases endpoint distance by 0.195 frames;
it is worse in 308/487 windows. This is `tested` diagnostic evidence for the
invalidated `8bfc0e5` checkpoint, not evidence about corrected `0ea4e15` and
not paper-ready. Full details: `experiments/duca-selection-quality-epoch89.md`.

# DUCA Query Pack

## 2026-07-16 CellCF formal handoff status

The current deployable candidate is exact commit
`3a0f5ae54d1dbd23ff170cda8a4706f5ed0d38d3` on
`codex/duca-cellcf-20260716`. It preserves offline TAD, T=768/K=384, the three
matched arms, 132 epochs, 13,200 successful updates, checkpoint every five,
and terminal `epoch_131.pth/state_dict_ema`. Its formal evidence DAG now makes
trained-checkpoint cost mandatory and distinguishes
`runs_complete_cost_pending` from final `complete`. Local checks and an
independent max review passed, but status remains `tested`: exact-commit Linux,
synthetic, real CUDA and DDP pilot evidence have not yet been regenerated.
Old commit `475634e` Jobs `1167145/1167146` are diagnostic only.

## 2026-07-15 dynamic-DDP corrected P0 status

Real-batch diagnostics invalidated the old `with_cp=True/static_graph=True`
protocol. The current four arms retain `with_cp=False`, `static_graph=False`,
`find_unused_parameters=True`, 132 epochs, and checkpoint interval 5. Commit
`043be401` additionally aligns the relaxed structured distribution to each
sample's real hard feasible family. Gate `1164318` and replacement pilot
`1164319` passed and authorize only the hash-bound formal Jobs
`1164700-1164703`; stale gates and pilot `1164286` remain invalid. The corrected
P0 suite is `experiment_running`, but C3/C4 remain unproven until matched mAP
and post-run evidence exist.

## 2026-07-13 corrected transition-only P0 status

Commit `40eb86ee69e19b3105f9ddd6a977fb7693f724ad` is a superseded diagnostic
transition-only implementation. Formal CUDA gate `1161590` passed exact-uniform,
finite-update, optimizer-coverage, and all-short counterfactual graph checks.
Independent exact-commit audit and a multi-iteration DDP pilot remain required
before matched full train. Status is `tested`, not empirical support.

Do not reuse Jobs `1159414-1159417` or `1161482-1161548`: they exposed invalid
uniform/protocol, Slurm GPU remap, FP16, calibration, short-window, or DDP graph
failures and remain diagnostics only. Exact reasons stay in experiment nodes
and `log.md`.

## 当前裁决

本项目研究的是离线 TAD 的任务感知时序去冗余，不是流式或因果 Online
TAD。目标是在完整离线窗口中，用低成本粗动作状态证据分配有限帧/片段计算，
在降低真实全流程成本的同时保护高 tIoU 定位性能。

当前 global Structured-TopK Transition-Only 不再是允许直接扩展的最终候选。
DUCA 只保留一次有界申诉：DUCA-CellCF。粗分类器仍只学二分类动作状态，selector
只读 deploy-visible state changes；exact-uniform anchors 划分保序 cells，每格恰选一帧；
训练期用少量 hard alternatives 的 detector loss 蒸馏 cell 内 preference，删除
`G=15` global top-k 和未经 one-swap 验证的 soft RGB bridge。CellCF 当前仅为
`designed`，不是已实现、已测试或最终模型。

## 2026-07-13 空间 Zoom gate-level 设计裁决

`Dense-Time Spatial Zoom` 升为 `designed` 仅指 S1/S2 gate protocol 已形成，尚未替代
DUCA，也没有模型或实验结果。当前只允许实现 S1：从 dense160 派生 matched dense224/
256 configs、validator、manifest、统计和 full-stack profiler，先验证高 tIoU/短动作
spatial headroom。S1 未 GO 不实现 S2；S2 未 GO 不实现 DART-Zoom。

接受 dense-time regular grid、AdaSpot strongest-neighbor、native 96/112 crop 和 strict
decode-to-output cost 原则；不接受 reviewer 给出的 scout/J/K/loss/teacher cadence/15%
阈值为已批准规格。S2 所谓 label-free oracle 是 privileged dense-teacher reference，只能
在冻结 gate split 做 headroom，official test 在路线冻结前封存。任一 gate 失败即停止，
不能增加 policy/loss/candidates 延长路线。

## 2026-07-13 `1fc7037` CellCF REDESIGN 吸收

审查与本地复核共同确认四项确定性缺陷：legacy/direct midpoint uniform 残留、
batch-varying `pos_weight` 破坏 `p_action` 概率解释、coverage loss 可为负并奖励聚集、
selection diagnostic 把 compound proxy 误标为 pure delta。optimizer coverage、leaf loss
去重和主推理路径 GT/teacher 泄漏未发现问题。

项目基本接受 coverage-preserving local cells + hard counterfactual utility 的方向，但不
接受它已被证明。固定 detector anchor 可能造成采集帧与时间标签错位；cell-wise utility
可能忽略跨 cell 交互。先过 pure/compound/learned、coarse matched、one-swap、coverage、
geometry 五类机制门，再允许 fixed-384 pilot。pilot 不优于 same-commit exact uniform
即停止 DUCA 主方法，不开放 MUST、X3D/SlowFast、多 detector 或新 loss。

## 失效 DUCA 协议记忆

`8bfc0e5` 的 midpoint uniform 在 `T=768,K=384` 产生全零 logits 与退化
Viterbi tie-break：和 rounded-endpoint uniform 仅重合 47.135%，rank MAE
179.695 帧。因此 Jobs `1159414-1159417` 不是 matched uniform/homotopy 证据；
55.67 也不是均匀基线。`0ea4e15` 统一 rounded endpoints，但后续审计又发现
legacy/direct 残留、raw-pixel bridge 与 hard one-swap utility 未对齐。所有路径必须
逐点审计 selected positions；旧 gate、旧 mAP 与文档 HEAD 均不得冒充修复后证据。

## 可审计实验背景

- DUCA 70aa fixed-384 Job `1154971`：best Avg-mAP 58.39，@0.7 为 34.53；
  loader exposure 与历史实验不匹配，不能直接裁决联合训练。
- 旧 `8bfc0e5` P0 已完成，best 55.67/57.71/64.34/63.55
  分别对应错误 uniform/direct/beta0/beta0.25，只保留为诊断数字。
- 历史强均匀采样来源已定位：Job `1150701` 的 native stride-2 + adapter +
  ActionFormer best Avg-mAP 64.352；Job `1150842` 的 grid-aware detector best
  Avg-mAP 65.696。两者证明“uniform 约 65”有真实日志来源，但 detector/geometry
  与当前 P0 不完全相同，不能直接填入 matched 主表。
- 历史分离 lattice best 63.18、PAction best 61.02；其 loader 为 218
  step/epoch，70aa 为 99 step/epoch，必须按 optimizer steps、effective K 和 LR
  progress 对齐。
- PhysTime v1 已被否定：selected-axis/physical-grid/PhysTime best Avg-mAP
  63.61/59.14/57.21。不得启动 Phase 2 或宣称它改善 TAD。
- X3D/SlowFast 仅允许作为 frozen-prior 或 appendix diagnostic；密集推理成本可能
  吞掉省下的 heavy-backbone 成本，不能作为默认主方法。
- Dynamic MUST、MobileNet、新 detector head 和新 selector loss 在 corrected
  fixed-384 决定性门槛前保持冻结。

## 当前 claim 状态

- C1，低成本粗动作状态可支持有用稀疏选择：
  `unproven/implementation_tested_matched_protocol_invalidated`。
  transition-only 已实现，但 corrected matched P0 尚未运行。
- C3，DUCA fixed-K 优于 matched uniform/random：`unproven`。旧 55.67
  baseline 无效，历史 64.35/65.70 又不是同协议。
- C4，detector gradient 改善 selector：`unproven`。失效协议下 beta=0.25
  比 beta=0 低 0.79 best Avg-mAP，且仍缺 corrected 对照与 one-swap alignment。
- 成本 claim：`unproven`。必须统计 decode、预处理、H2D、probe、selector、
  backbone、head、后处理、显存和能耗，不能只报理论 FLOPs。

## DUCA-FSU / CellCF 仍受限

审计建议以 train-only feasible hard-swap detector gain 蒸馏 cell 内 preference，删除
soft RGB bridge；这属于 counterfactual utility distillation，不等于 detector gradient
直接穿过 hard selection。FSU/CellCF 均仅为 `discussed/designed`，必须先过 uniform、
coarse quality、learned-vs-noise、one-swap、geometry 与成本门，不能写成已实现方法。

## 决定性实验顺序

1. 先统一所有 exact-uniform route、action target helper、coarse score语义、coverage
   loss 与 diagnostic keys，并运行 focused/exhaustive tests。
2. 运行 pure delta/compound/learned、matched coarse、hard one-swap、cell coverage 与
   same-selected-frames geometry gates；未通过不得 full train。
3. 机制门通过后，同一 commit 运行 exact-uniform 与 CellCF fixed-384 一 seed pilot；
   没有稳定正增益即停止，不补另外 seeds。
4. pilot 通过才做三 seed、trained-checkpoint full-stack cost；fixed-384 成立后才讨论
   第二 detector。dynamic budget 继续冻结。

## 不得重走

- 不得再以 config 名称、alpha=0 或 selected_count=K 代替选中位置审计。
- 不得把 Job `1159414` 的最终 best 55.67 写作 exact-uniform，也不得把 64.352/65.696
  直接冒充当前 matched baseline。
- 不得复用 gate `1159395` 启动修复后的实验；它没有验证 uniform 位置合同。
- 不得把现有 beta0/beta0.25 曲线写成正确 homotopy 的证据。
- 不得用 actionness top-k、GT 边界推理、硬膨胀或 post-hoc repair 掩盖 selector
  学习失败。max-gap 必须属于同一可行集合，而不是训练外补丁。
- 不得把三阶段独立训练包装成最终联合模型，也不得称当前任务为 Online TAD。
- smoke、precheck、nonzero gradient、单 seed 和旧 commit mAP 都不是论文主证据。
- 不得再把 `transition_score=abs_delta+uncertainty_peak` 称为 pure raw delta。
- 不得把 CellCF review 代码片段称为已实现或把其阈值写成实验事实。

状态必须严格区分：`discussed -> designed -> implemented -> tested ->
experiment_running -> empirically_supported -> paper_ready`，不得跨级。

## 2026-07-18 CellCF 成本数据契约状态

`1170366` 的失败已定位为成本 profiler 生产端与严格 summary schema
之间的字段合同遗漏，不是模型训练失败。精确修复提交为
`4ce69c852bdbd902046b47bc6019ae11e850dbe4`：七个注册的
`*_cpu_enqueue_ms` 字段只保留在 raw sample 中，不进入正式 stage
p50/p95 或总延时；未知字段、负值和非有限值继续 fail-closed，并在每个
sample 写入前立即验证。

本地 broad suite 为 `259 passed, 10 skipped`，远端 clean Linux 为
`279 passed`。Job `1170932` 的真实 profiler 产物通过独立严格重建，但
该作业因临时 heredoc 校验器引号损坏而 `FAILED/1:0`，只能作为包装器
诊断。唯一通过的真实 GPU 短门禁是 Job `1170940`：
`COMPLETED/0:0`，收据 SHA-256
`f69bc872993fc778b2ceaf6b1a179721861aa57c176d0e12b5869a3913e14758`。

当前状态只推进到“数据合同 `tested`”。两样本门禁不提供正式成本数值，
新的 500-sample CellCF/bare-uniform cost pair 尚未提交；C7、dense
full-stack 节省、break-even 与论文准备度仍为 `unproven`。
## 2026-07-21 Uni-AdaFocus-inspired bounded optimization

- Official source audited at `LeapLabTHU/Uni-AdaFocus@8846488`.
- The transferable mechanisms are training input diversity and policy-specific
  gradient/LR control. Its detached hard temporal indices are not a substitute
  for DUCA's detector-to-selector bridge.
- An isolated branch now implements three matched learned variants:
  direct bridge `1.0`, scaled bridge `0.25`, and scaled bridge `0.25` plus a
  one-pass exact-uniform training companion.
- The companion changes training rows only. Inference remains one learned hard
  physical exact-K path with no companion and no extra inference cost.
- Current status is `implemented_local_static`; Windows Torch is unavailable,
  so remote exact-snapshot tests and CUDA gates are still required. No mAP or
  greater-than-65 claim exists.

## 2026-07-21 DUCA Uni-Companion replacement deployment status

- Current exact branch: `codex/duca-uni-companion-inputfix-20260721`.
- Current exact commit/tree:
  `4d84acda4d073fb6aac956c21386df8ed5d4d2f5` /
  `b15a064784f25d888cc66df01c39781422403195`.
- Clean Linux suites: `67` focused plus `23` required legacy tests.
- Exact P0 file SHA-256:
  `eabc6da8c3cc4308b70a8c8d6bbecc6c6e4b408cb17d2ee6041ed83f24a4eb3f`.
  It freezes batch size 2, 60 epochs and 6000 successful updates per arm.
- Current gate Job `1177696` is `PENDING (AssocGrpGRES)`.
- Three learned official-60 Jobs use `afterok:1177696`: direct `1177697`,
  bridge-0.25 `1177698`, and Uni-companion `1177699`.
- Exact-uniform is P0-frozen and watcher PID `808310` will submit it only
  after successful gate authorization and a freed association slot.
- The same P0/gate also freezes a rho=0.01 arm that exposes only the final
  official ASFormer encoder block to detector gradients. Watcher PID `883230`
  submits it only after the gate and direct arm both complete successfully;
  it currently has no Job ID.
- Superseded `d748684` Jobs `1177687/1177690-1177692` were cancelled at zero
  runtime after the real uint8-loader incompatibility was found.
- Status is `experiment_running` at the gate/dependency level. CUDA/P3
  authorization, all training outputs, exact-uniform matched mAP and any
  greater-than-65 claim remain unproven.

## 2026-07-21 DUCA uniform-policy homotopy exact gate

- Current exact method commit/tree: `b987c8c6...5798a` /
  `d33d9194...29411`; clean Linux verification is `155 passed`.
- P0 freezes exact-uniform alpha-zero hard input, 300 successful-update
  warmup steps, 1800 homotopy steps, 3900 learned-policy steps, and 6000 total
  updates per arm. Inference always uses the learned endpoint.
- Gate `1177713` is a failed dtype-contract diagnostic: the float64 physical
  cap was narrowed before validation. Gate `1177714` is also a failed
  gate-tool diagnostic: its perturbation helper called floating-point
  `randn_like` on real-loader `uint8` RGB. Neither gate reached training or
  produced mAP. The deterministic uint8-safe correction is isolated in
  `b987c8c`; fresh P0 SHA-256 is
  `a246dc8c3fbc6f6e4a65a3a706a1259e54421f93a4707a922c567db1c92f9b99`.
- Replacement Job `1177715` is running from the clean snapshot and fresh P0.
  It includes a forced
  full-model AMP-overflow replay check and must produce a hash-bound
  authorization before any four-arm official-60 submission.
- Gate `1177715` failed closed at elapsed `00:02:40` on the full-window
  exact-uniform physical-vs-selected-axis detector-loss parity check. A
  read-only diagnostic on the identical commit/P0 measured physical objective
  `0.072007999` versus selected-axis objective `0.094866410` (24.10% relative
  difference); classification and regression differed by 25.35% and 22.41%.
  The exact-uniform path contains 382 gaps of 2 and one gap of 3, so its
  rounded endpoint map is not globally affine. This is not numerical noise
  and must not be bypassed by loosening tolerance.
- Status is `gate_failed/stop_and_revise_representation`; no official-60 arm,
  optimizer update, checkpoint or mAP exists for this successor.

## 2026-07-21 selected-axis optimization successor

- The successor explicitly chooses the established selected-axis detector
  semantics: train GT is remapped through the actual hard selected positions,
  terminal proposals are inverse-mapped to true time, and the official
  `ActionFormerHead` config remains unchanged with no physical-grid extension.
- Three learned fixed-K=384 variants are implemented on one protocol:
  direct bridge-0.25, uniform-to-learned homotopy bridge-0.25, and homotopy
  plus a one-pass 50% exact-uniform training companion. Exact-uniform is the
  matched control.
- The companion is training-only. At batch size two, one row is canonical
  endpoint-uniform and one row is learned; detector gradient reaches the
  transition scorer only through the learned row. Inference always uses one
  learned hard path and has no companion cost.
- Status is `implemented_local_static`. Config contracts pass locally, but the
  exact Linux/CUDA gate and formal Slurm jobs are not yet complete. No new mAP
  and no greater-than-65 evidence exists.

## 2026-07-21 10:23 selected-axis formal runtime checkpoint

- Exact formal evidence remains method commit `cb89586`, gate Job `1177776`
  and four-arm Jobs `1177779-1177782`; all four are RUNNING on separate GPUs.
- Exact-uniform and companion completed epoch index 14 / 1500 updates and
  entered epoch 15; direct and homotopy were within epoch index 14. Losses are
  finite, K is exactly 384 and reported memory is about 8.6 GB.
- Each arm has three isolated AMP replay events, all recovered at replay 1/8.
  There is no Traceback, OOM, NaN/Inf, replay exhaustion or executed failure.
- `duca_detector_grad_w` is still zero because the protected bridge starts at
  successful update 2100. Current selector diagnostics therefore test coarse
  and transition pretraining, not detector-to-selector feedback.
- Read-only trajectory Job `1178357` is RUNNING from diagnostic commit
  `87cfd20`; it does not alter the formal model or checkpoints.
- Terminal epoch-59 EMA and official mAP still do not exist. Greater than 65
  remains the decision threshold, not an achieved result.

## 2026-07-21 10:40 normalized delta-residual diagnostic

- Early endpoint scorer logits have mean per-window RMS about `0.116`, while
  `abs_delta_p_action` has RMS about `0.0026`; a raw unscaled residual would be
  numerically negligible.
- Diagnostic commit `7f9ad10` implements the read-only formula
  `z(transition_policy_scores) + gamma*z(abs_delta_p_action)` and decodes every
  gamma through the same hard exact-K=384/G=2 solver. Gamma zero is required
  to preserve the learned endpoint hard path.
- Clean Linux verification is `9 passed`; exact commit/tree are
  `7f9ad10ac35cb61fa68a17003f2bc1c488dd9c10` /
  `c397d073d9b1c396c06babf61f4ee0b3aa22ced3`.
- Slurm Job `1178384` is dependency-pending after successful trajectory Job
  `1178357`. It is evaluation-only and cannot authorize a model replacement
  or choose gamma on validation GT. No formal training arm was changed.

## 2026-07-21 local reachability and repaired P0 checkpoint

- Job `1178738` completed a selector-only export of 120 records from the
  training-only holdout. Its source is the epoch-19 EMA left by diagnostic Job
  `1178591`; no detector, GT, teacher or prediction cache entered selection.
- The exact audit compares matched `U` exact uniform, `D` pure delta with one
  point per uniform cell, `C` current checkpoint, privileged local GT oracle
  `L`, and privileged global GT oracle `G`, all at exact K=384 and the same
  uniform-reference gap cap. These are geometry diagnostics, not detector-mAP
  oracles.
- The full 120-record exact run completed. Local and global privileged GT
  oracles have identical boundary recall and both-endpoint coverage at every
  reported radius. At radius zero, uniform/local/global boundary recall is
  `0.1429/0.2507/0.2507`; mean endpoint distance is
  `0.4775/0.2484/0.2462`. Learning and supervision, not one-per-cell
  reachability, are the present bottleneck.
- Coarse actionness remains weak (`AUROC=0.6161`, `AUPRC=0.3750`,
  `Brier=0.2042`). The invalidated current checkpoint trails exact uniform and
  pure delta at radius one. Uniform already reaches `0.9998` r1 recall, so r1
  coverage cannot stand in for terminal TAD mAP.
- The accepted exact solver pins every semantic objective at zero gap and
  disables only the numerically unstable final position tie-break. It does not
  round invalid variables or replace the non-additive oracle with additive DP.
- P0 repair is implemented in isolated branch
  `codex/duca-local-residual-20260721`: all 19 weights explicit, inactive losses
  graph-free, class-balanced action BCE, transition supervision detached from
  coarse representation, GroupNorm, exact optimizer coverage and no global
  clipping. Local pure tests are `25 passed`; Linux/CUDA remains an exact-
  commit Slurm gate. Deployment is one real one-step gate followed by three
  sequential P0 candidates, with `DUCA_FRONTEND_ONLY=1` preventing automatic
  entry into unrepaired old official-60 arms.
- Exact implementation commit is
  `5d17dcbe564efd1e69194dd5faddf34266e39f86`; clean remote Linux verification
  is `96 passed, 2 skipped`. Single fail-closed Slurm Job `1178774` is pending
  from root `duca_p0_5d17dcb_20260721_1640`; no CUDA-gate verdict or trained
  candidate exists yet.

## 2026-07-21 16:57 P0 replacement gate

- Job `1178774` failed before training because the evidence classifier looked
  for obsolete `spatial_encoder` parameter names instead of the executed
  `spatial_stem` path. This is not a model-gradient result.
- Exact corrective commit is
  `9442b9487f871efd02c85dceeed26574c641369d`; clean Linux verification is
  `74 passed, 3 skipped`.
- Replacement frontend-only Job `1178809` uses the same training-only split
  and the same one-gate/three-candidate protocol. No official-60 run is
  authorized until P0 produces a valid holdout decision.

## 2026-07-21 local-cell route裁决

The current local-residual branch `codex/duca-local-residual-20260721` at
`6c56e11` is no longer considered a final-method direction. Its one-frame-per-
exact-uniform-cell contract protects coverage but violates the original DUCA
motivation: coarse actionness/state-transition evidence should indirectly
allocate observations toward semantic boundaries, not merely decide which of
two neighboring frames is kept inside every uniform bin. The running P0 Job
`1178863` remains useful only as a diagnostic of coarse evidence and local
delta scoring. It must not automatically unlock official-60 paper-grade main
experiments, and any result should be reported as a conservative local-uniform
ablation.

The next admissible main-method family must contain exact uniform as a
recoverable baseline while allowing cross-region quota transfer under an
audited hard coverage/max-gap constraint. Candidate language: protected
coverage scaffold plus transition/boundary-driven residual budget allocation.
The selector objective should prioritize state-transition peaks, uncertainty
peaks and GT boundary coverage; actionness itself remains binary coarse
supervision and a weak auxiliary signal, not the primary selection score.

## 2026-07-21 DUCA model-lineage freeze

The required global family is not missing. Commit `cb89586` already uses
`global_structured_topk` to solve full-window exact-K/max-gap selection, so it
allows cross-region quota transfer without one-frame-per-cell slots. The next
implementation must reuse that selected-axis policy and combine only existing
repairs: P0 numerical/gradient contracts from `5d17dcb/6c56e11`, curriculum
structure from `6f2ed48`, and protected selector-gradient components from
`cb89586/ee05f61`. Local-cell decoding remains diagnostic and must not replace
the global feasible set. The authoritative inventory and reuse map are in
`duca_model_version_registry.md`; creating another actionness source, exact-K
solver, detector wrapper, or profiler is forbidden unless a concrete missing
contract is first recorded there.

## 2026-07-21 global-curriculum exact implementation

The non-local-cell successor is frozen at exact commit
`4c777a691d65fe484dfe537ac3e33f82b5bbe5a8` on branch
`codex/duca-global-curriculum-20260721`. It reuses the existing selected-axis
`global_structured_topk` policy and official AdaTAD/ActionFormer backend; it
does not introduce a new selector or detector wrapper. The matched arms are U
exact-uniform, G0 global without detector feedback, G1 protected feedback to
the transition scorer only, and G2 G1 plus a training-only exact-uniform
companion with learned-row gradient-exposure normalization. Clean remote
focused evidence is `74 passed, 2 skipped`. Serial Job `1178911` was submitted
from the exact clean snapshot and was `PENDING (Priority)` at 18:51 +08:00.
Status is `experiment_running`; CUDA full-model gates and terminal official-60
mAP do not yet exist, so greater-than-65 and paper-readiness remain unproven.

The P0 audit superseded the executable identity without creating a new model.
Exact commit `63e25eb17e523d369f73434ed4d9b6446608861a` keeps V8's global selector
and U/G0/G1/G2 matrix, fixes the saturated radius-one checkpoint rule, and
tests three component-LR profiles under one fixed auxiliary-loss definition.
Parent regression evidence is `158 passed, 3 skipped`; the final complete-entry
revision passes `18` affected Linux tests and resolves export, analysis and both
aggregators as modules. Jobs `1178911/1178927/1178933` are immutable failed or
cancelled history with no admissible model result. Job `1178947` then failed
before any update because the real P0 gate misclassified attention-layer
`conv_out` projections as action heads. The optimizer itself was correct.
Commit `9138156` aligned the gate with the existing ActionFormer grouping.
Job `1178975` then executed one valid P0 optimizer/EMA call but failed because
the gate inferred branch-level EMA activity from one FP32 representative
parameter. Commit `63e25eb` checks all parameters in each existing group and
changes no model, loss or schedule; affected remote contracts are `21 passed`.
Active Job `1178989` runs from `duca_global_63e25eb_serial_20260721_2120`. No
terminal mAP exists yet.

At 21:27 +08:00, a read-only recheck found `1178989` still pending for
priority with zero runtime. Predecessor `1178642` remained finite with its two
remaining arms at epochs 58 and 57 of 60, but no new terminal JSON. This is a
monitoring update only: V8 stays frozen and no new selector/model/worktree is
admissible.

Job `1178989` then started and its exact real-data P0 gate passed with
`ok=true`: complete AdaTAD construction but zero detector calls, byte-invariant
detector, exact optimizer coverage, disjoint coarse/scorer gradient ownership,
one finite AMP/scheduler/EMA update, exact K=384 and max-hole 2. The first
control-LR P0 candidate entered epoch 0 with finite losses. This advances V8
from queued to real-gate-tested/experiment-running, but supplies no P0 winner
or terminal mAP.

## 2026-07-21 selected-axis partial terminal verdict

Predecessor Job `1178642` completed all four terminal epoch-59 EMA arms on
exact commit `cb89586`: exact-uniform `64.4580`, direct-0.25 `63.7102`,
homotopy-0.25 `63.0601`, and homotopy plus uniform companion `63.6931` Avg-mAP.
Every learned V5 arm is below uniform; companion recovers about 0.6330 over
homotopy but remains 0.7649 below uniform. Do not rerun or rename these old
arms. This negative result does not adjudicate V8 `63e25eb`, whose repaired P0
is currently training, whose official-60 coarse branch is frozen and whose
protected detector gradient updates only the transition scorer. Greater-than-
65 remains unproven.

## 2026-07-21 Uni-AdaFocus transfer boundary

Official commit `8846488` uses inverse-CDF quantiles to obtain K ordered unique
heavy-frame indices, but provides no physical-time max-gap or TAD-boundary
coverage; hard indices are detached. Its transferable ingredients are
component-specific learning rates, auxiliary task supervision, a cheap global
stream and final reuse of global features. Therefore it supports V8's current
P0 learning-rate and protected-gradient design, while the conditional post-V8
hypothesis is dense cheap coarse context plus sparse heavy refinement. Do not
change running Job `1178989` or compare its ActivityNet classification mAP with
THUMOS TAD Avg-mAP.

The corresponding post-V8 fusion contract is now bounded: VideoMAE remains the
primary detector representation; the full coarse sequence is projected by a
separate adapter and enters selected VideoMAE tokens only as timestamp-aware,
zero-gated context. TAD loss must not freely update the coarse action head or
trunk. This context-fusion test is distinct from a later canonical-grid
coarse-fallback test that might justify relaxing max-hole. Neither is part of
running Job `1178989`.

## 2026-07-21 V8 Pro supervision-objective audit

Exact-commit static review confirms that V8 reuses the radius-4 Gaussian
transition target in an `exp(-neighborhood_mass)` coverage loss, which can keep
rewarding repeated occupancy around one endpoint. It also confirms that the
G1/G2 bridge is surrogate transport and that the serial gate does not execute
the preregistered real hard-swap alignment before declaring formal training
unlocked. These are real evidence-contract defects, not terminal mAP results.

The review's proposed `radius=1` exact event replacement is itself invalid under
the frozen `max-hole=2` family: every internal three-position event is already
guaranteed to be hit, yielding probability one and zero gradient. The project
therefore records `SUBSTANTIAL_ACCEPT_DIAGNOSIS / REVISE_OBJECTIVE_BEFORE_IMPLEMENTATION`.
Any successor must first demonstrate nontrivial set-level headroom, with rounded
`radius=0` unique endpoints as the bounded candidate, and must separately pass
the real hard-swap alignment gate before G1/G2. Job `1178989` remains unchanged,
`experiment_running`, and diagnostic; no V8 winner or terminal mAP exists.

## 2026-07-21 first V8 P0 candidate verdict

The first component-LR profile completed 20 epochs without runtime failure, but
its `ok=true` receipt is only a training-integrity result. At epoch 20 the
coarse branch reached macro AUROC `0.624512`, while the learned policy's
radius-zero transition AUROC was `0.521321`, below simple
`abs(delta p_action)` at `0.553237`. Learned radius-one endpoint recall was
`0.883237` versus exact-uniform `0.999775`, and mean endpoint distance was
`0.538120` versus `0.477457`. Thus the coarse classifier is improving but the
transition scorer is not turning it into superior boundary placement. This
candidate is not a winner and does not support C3/C4; the other two profiles,
holdout selection and U/G0/G1/G2 terminal mAP remain pending in Job `1178989`.

## 2026-07-21 Oracle-like boundary clustering correction

The project objective is not one observation per boundary. The historical GT
Oracle first selects every endpoint center plus/minus radius two and then fills
the remaining budget uniformly, so useful local clustering is part of the
target behavior. The defect in V8 is uncontrolled broad-band mass accumulation:
it does not guarantee that every start/end is centered, receives bilateral
support, or stops receiving reward after a useful local quota is reached.
Accordingly, a radius-zero unique event can only be an anchor term. A bounded
successor objective must combine exact endpoint anchoring, oracle-calibrated
capped multi-frame boundary bursts and residual global coverage, with deduped
overlapping endpoints. No running V8 code or Job `1178989` is changed by this
clarification.

## 2026-07-21 center/bilateral-count implementation audit

Registered V0--V8 and inherited PAction/GAS-VT/lattice code contain all of the
following separately: legacy center-plus-symmetric-radius decoding, learned
context radius and score dilation, a binary GT boundary bracket loss, global
exact-K/max-hole selection, and left/right evaluation metrics. They do not
contain one deployable model that jointly predicts transition centers,
allocates capped left/center/right multi-frame bursts per endpoint, deduplicates
overlapping endpoints, globally spends the residual budget and trains through
the current official AdaTAD graph. Oracle is the only implementation with an
explicit center-plus/minus-two pattern, but it consumes GT at evaluation and is
not deployable. The missing contract is now G23 and must be implemented, if
authorized after V8, by extending the existing V8 scorer/DP rather than creating
another selector family. Status: `designed_not_implemented`; Job `1178989`
remains unchanged.

## 2026-07-22 canonical DUCA final product and roadmap

The final paper artifact is one offline-TAD pre-backbone acquisition plugin,
not a new detector, Online TAD system or three independently deployed models.
Its canonical graph is dense cheap low-resolution coarse/official-ASFormer
evidence -> indirect transition center -> Oracle-calibrated bilateral capped
boundary burst -> overlap-aware saturation and residual global utility -> the
existing global exact-K/max-hole DP -> hard observations with original-time
metadata -> official-derived AdaTAD/ActionFormer.

The model is trained with a two-stage curriculum. P0 skips the heavy detector:
binary action loss updates the coarse branch, while center/bilateral/quota/
context objectives update scorer/burst parameters. The official stage starts
with exact-uniform detector warmup inside the same update budget, then freezes
coarse semantics and permits TAD loss to reach only scorer/burst parameters via
a real-hard-swap-aligned protected bridge. Radius-zero is only a center anchor;
the target includes 3--5-frame bilateral bursts and residual global coverage.

The fixed execution order is Oracle K/G reachability, mathematical/gradient
gates, a single-variable P0 objective comparison, terminal-EMA U versus G0,
hard-swap alignment before G1/G2, then three seeds, budget curve, a second
detector and complete measured cost. If feasible Oracle has no mAP headroom,
G0 does not beat U, bridge alignment fails, or full-stack cost has no net
saving, stop the corresponding claim. Status remains
`designed_not_implemented`; no V9 is created by this plan.

For the current THUMOS/AdaTAD anchor, GO additionally requires terminal-EMA
Avg-mAP >=65.00, at least +0.20 over matched U, high-tIoU loss no worse than
0.20, and lower measured end-to-end cost than dense. Fixed K=384 is primary,
K=256 is the first efficiency extension; dynamic budget is not a main claim.

## 2026-07-22 Uni-AdaFocus / EU-CRR Pro review absorption

The `HOLD` review was archived byte-for-byte with SHA-256
`0678A31C17D3FCD983726CE9056E463CF09A0325DAF69C7C41947EEB57602DAA`.
Exact `63e25eb` code review confirms that coarse hidden is consumed by the
selector only; ActionFormer sends hard selected RGB directly through VideoMAE
and has no detector-side coarse fusion. The protected bridge is a detector-
loss-derived surrogate, not direct differentiation through hard indices, and
the full cost claim must include dense decode/transform/H2D/coarse/DP.

Project verdict:
`SUBSTANTIAL_ACCEPT_DIAGNOSIS / CONDITIONAL_ACCEPT_DIAGNOSTIC / REJECT_AS_MAINLINE_REPLACEMENT`.
One matched exact-uniform U0/U1 zero-gated post-VideoMAE coarse residual
diagnostic is scientifically clean, but remains
`discussed_conditional_diagnostic_not_authorized`. It cannot test transition
centers, bilateral 3--5-frame bursts, endpoint fairness, quota saturation or
residual global coverage, so it cannot replace G23/R0--R5 or create V9.

The fixed order remains: seal V8 Job `1178989`, run R0 train-split Oracle K/G
reachability, then implement G23 R1--R3. If EU-CRR is later authorized, report
`U1-U0`, `L1-L0`, `L0-U0` and `L1-U1`. U1 failure kills fusion only; U1
success changes the honest artifact name to acquisition-and-fusion adapter,
not strict pre-backbone-only plugin.

## 2026-07-22 canonical final-model single source of truth

`research-wiki/duca_final_model_contract.md` is now authoritative. The paper
artifact is one fixed-budget offline-TAD pre-backbone acquisition plugin:
low-resolution binary coarse action state -> indirect transition centers ->
Oracle-calibrated bilateral 3--5-frame boundary bursts -> saturating overlap
deduplication plus residual global context -> existing exact-K/max-hole DP ->
hard original-time RGB -> unchanged official-derived AdaTAD/ActionFormer head.

Coarse action supervision updates only the coarse stem/ASFormer/action head;
endpoint anchor/bilateral/quota/fairness losses update scorer/burst only. P0
skips the detector. Official-60 spends updates 0--999 on exact-uniform detector
warmup, ramps the learned policy during 1000--2499, freezes coarse throughout,
and permits TAD-derived surrogate updates to scorer/burst only after real legal
hard-swap alignment. Main K is 384 and first efficiency extension is 256.

Evidence order is immutable: seal V8 -> R0 Oracle K/G reachability -> R1
math/full-model contracts -> R2 P0 objective isolation -> R3 matched U/G0 ->
R4 alignment and G1/G2 -> R5 three seeds, budget curve, second detector and
deployment-mode full cost. Final GO requires Avg-mAP >=65.00, >=+0.20 over
matched U, high-tIoU loss <=0.20, positive three-seed mean and measured total
cost below dense. Status remains `designed_not_implemented`; no V9 exists.

## 2026-07-22 05:45 exact evidence-contract re-audit

The current unique candidate is
`codex/duca-boundary-burst-20260722@86f7663a94d628eace316d17e31db7043f731f75`.
The preceding `7b9ad0b` terminal-evidence repair passed Linux tests, but a
real historical selected-axis checkpoint sidecar proved that the production
audit builder omitted `formal_protocol` and `training_profile` while the
terminal validator required them. Unit fixtures had hand-injected those keys.
`86f7663` minimally fixes that producer/consumer mismatch, makes the fixtures
use the real production builder, and makes final aggregation independently
check the protocol identity. The exact clean remote snapshot passed DUCA
`64 passed` and C3/update-evidence `29 passed` plus compile/bash/HEAD/clean.
A fresh no-context MAX review is running. There is still no CUDA gate, R0
headroom, P0 winner, terminal mAP, V9 or paper-ready claim.

## 2026-07-22 06:54 4ec3e07 bounded R0 repair candidate

The exact pushed candidate is now
`codex/duca-boundary-burst-20260722@4ec3e078a3aad834ffe504d74d414bf7e2b6fad3`.
It closes only the bounded `86f7663` audit findings: R0 now evaluates exact-uniform,
R2Q3/R4Q5 projected Oracles and an exact-K unrestricted boundary-burst Oracle; performs
paired per-video bootstrap by rerunning the official OpenTAD evaluator; and makes the P0
consumer reopen and recompute the split, blocked set, annotation, class map, prediction,
checkpoint, config, family artifact and bootstrap decision. Simple `abs(delta p_action)` now
uses the same global exact-K/max-hole DP and candidate eligibility stops unless the learned
selector strictly Pareto-improves the same-feasible baseline. Crop-valid metrics, a no-mock
producer-to-evaluator test, and a crash-safe per-job Slurm journal are included.

Local evidence is `22 passed` for R0/oracle/evaluator contracts plus `8 passed` for the
submission journal, with py_compile, shell syntax and diff checks passing. PyTorch-dependent
tests remain pending on Linux because local Windows Torch cannot load `c10.dll`. The current
stage is `implemented_local_tested_pending_linux_and_independent_max`; no CUDA, R0, P0 or
official-60 job is queued, and there is still no headroom, mAP, V9 or paper-ready result.

Remote exact snapshot `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_4ec3e07_20260722`
is clean at the verified 40-character commit. Linux evidence is `109 passed` for the complete
affected DUCA surface and `23 passed` for the mandatory C3/official-ASFormer regressions, plus
py_compile, bash syntax, exact HEAD and clean-tree checks. This authorizes only a fresh bounded
independent MAX audit; CUDA/R0/P0/official-60 remain blocked until that audit explicitly returns
`GO_TO_REAL_CUDA_GATE`.

At 07:15 +08:00 a launch-policy consistency audit found that R0 records a unique
`selected_weakest_projected_family`, while the current P0/gate/DAG still requires Gaussian,
R2Q3 and R4Q5 all to pass and queues all four official-60 arms. This can let a diagnostic family
wrongly veto the R0-selected main candidate and does not match the frozen R3 order of U versus
one corrected G0 first. The issue is `designed_fix_pending_independent_max`, not a model result;
no job is queued and no architecture change is authorized.

The fresh MAX audit then returned `HOLD_FIX_REQUIRED` for `4ec3e07`: the split files are
consumer-oriented exclusion lists, so `frontend_holdout_block_list` contains train videos and is
the correct source for a holdout evaluator's blocked set. The launcher had used
`frontend_train_block_list`, which contains holdout videos; the finalizer would reject it after the
GPU run, but the run would be wasted. Exact successor
`f90595d8620e42e8e3d74722f2ab48126c6b65f2` changes only that source and adds a semantic test that
blocked IDs equal `train_videos` and evaluator targets equal `holdout_videos`. Clean remote evidence
is `168 passed, 2 skipped` plus mandatory C3 `23 passed`; no-submit manifest SHA is
`14f345dc53b246b036ba1c80c993454c0d83a1173aae8481b54ac0f8647c8a2c`. A new independent MAX
review is running; no CUDA/R0/P0/official-60 job exists.

The second independent MAX returned `HOLD_FIX_REQUIRED` with explicit stage authorization:
R0 is allowed, while P0/CUDA downstream unlock/official-60 remain blocked. It verified the repaired
split/evaluator direction, Oracle families, bootstrap, same-feasible simple delta, selected-axis,
no-leak, official-derived detector and Slurm DAG. Remaining P1 evidence gaps are records→summary
recomputation for P0 winners, binding `p0_real_gate.json`, cross-arm matched evaluator/input identity,
and production verification of the frozen official-ASFormer hash. Exact R0-only Job `1179392` was
submitted from `f90595d` with no downstream jobs; run root is
`duca_boundary_f90595d_r0_formal_20260722_0753`, and the job is RUNNING on `g0006`. P0 fixes proceed
in parallel without modifying the running R0 code or evidence.

## 2026-07-22 corrected R0 raw headroom

At exact commit `d9fb398`, frozen official-AdaTAD replay on the sealed 40-video
training-internal holdout gives Avg-mAP `93.587070` for exact-uniform,
`94.190497` for projected R2Q3, `93.999241` for projected R4Q5 and `93.970057`
for the unrestricted GT Oracle. R2Q3 has the best raw gain (`+0.603427` pp),
showing bounded boundary bursts have nonzero detector headroom, while the weaker
unrestricted Oracle warns against spending too much budget at boundaries. The
1000-resample paired-video bootstrap and unique family decision are still
pending; these internal-holdout numbers are not paper test mAP and do not prove
the learned selector.

## 2026-07-22 model-first bootstrap consumer correction

The R0 producer still runs the preregistered 1000 paired video-cluster samples
through the official evaluator. Downstream consumers no longer repeat those
4,000 evaluator calls on every validation. They reopen and hash-check all
predictions/evaluations, recompute the four official point estimates, then
recompute bootstrap deltas and confidence intervals from the sealed producer
samples. This preserves the statistical protocol while removing hours of
non-model orchestration. Legacy P0 Job `1179533` is held before execution while
the corrected exact-commit consumer is tested.

## 2026-07-22 R0-R5 full deployment and bootstrap execution fix

The complete production R0--R5 implementation was sealed at exact commit
`e49ef69605e1f98a7217957483f93a8a64bfc348`. Its clean Linux snapshot passed
R0--R5 focused `192 passed` and mandatory C3 `23 passed`; a single independent
MAX gave `GO_TO_SLURM`. The formal DAG is fully submitted: R0 `1179795`, P0
`1179796`, real full-model gate `1179797`, matched U/G0 `1179798/1179799`, R3
aggregate `1179825`, legal hard-swap R4 `1179826`, real TemporalMaxer gate
`1179827`, four R5 bundles `1179861--1179864` covering all 24
backend/policy/budget/seed terminal cells, and final nine-profile cost/aggregate
`1179865`. Dependencies are fail-closed; only R0 is currently running.

Exact successor `9ed10139317c4196072d471ced883eb1dfc31703` changes only the sealed
R0 bootstrap executor from serial to process-parallel while precomputing the
same parent RNG draw sequence and preserving ordered outputs. Serial and
parallel samples are exactly equal; the clean remote snapshot passed relevant
R0 tests `35 passed`, mandatory C3 `23 passed`, compile/bash/HEAD/clean. Real
prediction benchmark Job `1179956` is running with 100 samples, four families
and eight workers. It is not a model revision and does not supersede the
already submitted e49 formal DAG unless the benchmark passes and the still
running serial R0 has not completed. R0 raw point estimates remain U
`93.587070`, R2Q3 `94.190497`, R4Q5 `93.999241`, unrestricted `93.970057` on
the sealed 40-video train-internal holdout; paired-bootstrap CI and family
decision are pending.

## 2026-07-22 R0 protocol correction: 94 is not comparable with official 65

The R0 40-video split was created for frontend-family selection after the reused
`transition_beta0/epoch_131.pth` detector had already been trained under the full
THUMOS `training` subset protocol. The R0 split does not prove that those 40
videos were excluded from detector training; the inherited training config uses
`subset_name="training"` with no R0 holdout block list. Therefore the 93--94
Avg-mAP values must be treated as detector-seen, training-internal replay and
not as held-out generalization. The exact-uniform value itself rises from the
historical official-validation 64--65 range to 93.59 under R0, proving the scale
change comes from protocol/provenance rather than a DUCA gain. R0 is now only a
paired mechanism diagnostic; its absolute mAP and bootstrap must not enter the
paper main table or be compared with official validation/test baselines.

The point estimates are computed with OpenTAD's official mAP implementation,
but the 40-video train-internal split and the custom 1000-resample paired
bootstrap are not the official THUMOS benchmark protocol. Main claims must use
one standard full validation/test evaluation per terminal checkpoint, matched
commits/configs/seeds, and multi-seed mean/std when uncertainty is needed. A
clean R0 headroom claim requires either a detector trained without the 40
frontend-holdout videos or a clearly privileged official-validation diagnostic
that is never used for test-set model selection.

## 2026-07-22 independent official-mAP correction

Exact commit `2bc6ca6fcf34f3e980437b5b830cabeef0de63c0` adds a protocol audit and a
self-contained four-arm official-60 runner without changing the DUCA model,
selector, decoder or detector. R0 is train-internal diagnostic only; P0,
one-step gates, hard-swap alignment and cost profiles are not mAP. A formal row
must use the complete THUMOS validation subset, OpenTAD `mAP`, tIoU
`[0.3,0.4,0.5,0.6,0.7]`, and `epoch_59.pth/state_dict_ema`. Unfinished R5 cells
are only protocol-eligible and cannot be pre-certified.

Independent MAX review found no P0/P1 blocker after the R5 wording correction.
Local and remote focused evidence is `49 passed`; the no-submit manifest records
`inter_job_dependencies=false`. The old serial DAG Jobs `1179795--1179865` were
cancelled. Four same-commit jobs now run independently with Slurm
`Dependency=(null)`: `1180075` exact-uniform, `1180076` Gaussian G0, `1180077`
R2Q3 G0 and `1180078` R4Q5 G0. Each learned arm internally performs fixed P0
epoch 19 on training data, a real full-model gate, official-60 training and full
validation evaluation. Its extra P0 training cost must be disclosed.

## 2026-07-22 official-mAP gate correction and final independent queue

The first independent submissions exposed three audit-gate assumptions before
any official-60 optimizer update: production ASFormer provenance lives at
`raw_actionness_source.probe.official_source`; real DDP may rebuild an equal
`gt_boundary_validity` container; and the compared tensors may reside on CPU
and CUDA. These were evidence-gate defects, not model, loss, detector or mAP
failures. Exact successor `8d85929ea04dc40f1eb0c3cc806061ce3b071d3f`
fixes only those checks, passes the same local and remote focused suite
(`28 passed`), and leaves the model/protocol unchanged.

The current formal root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_independent_8d85929_formal_20260722_1820`.
Jobs U/Gaussian/R2Q3/R4Q5 are `1180111/1180112/1180113/1180114`; every Slurm
record has `Dependency=(null)`. At registration time all four were independently
eligible and pending only for scheduler priority. Jobs `1180075`, `1180097` and
`1180106` are immutable pre-training gate failures with no mAP; their learned
companions were canceled before reaching the same known-bad gate and are not
experimental results.

## 2026-07-22 e49ef696 Pro audit absorption

The byte-identical Pro review is archived at
`docs/methods/reviews/2026-07-22-e49ef696-duca-r0-r5-pro-audit-raw.txt`
(SHA-256 `1D0F9909D2C3DF3966DED0B9F71BFA0A73F9CA2B8D7C68DF15F64265EC8AD636`),
with structured verification at
`docs/methods/2026-07-22-e49ef696-duca-r0-r5-pro-audit-absorption.md`.

Project verdict is `SUBSTANTIAL_ACCEPT_CODE_DIAGNOSIS / PARTIAL_ACCEPT_PLAN`.
Source inspection confirms: R2Q3/R4Q5 are soft bilateral burst objectives rather
than hard bilateral decoders; dense 160x160 decode/H2D precedes selection; the R5
aggregate does not independently rerun raw predictions; cost pairing is not
fail-closed; and dense commit identity is not historically pinned. Remote focused
verification is `96 passed, 1 warning`; these tests prove current contracts remain
stable, not that the missing contracts exist.

The review's stale execution advice is rejected. The old e49 serial DAG and
R0-authorized family selection have been superseded by four independent complete
validation jobs `1180111--1180114`. The current priority is their terminal
epoch-59 EMA mAP. H3/H4/H5 may be repaired evidence-only in parallel; H1 is
diagnosed before any hard decoder is added. Only one model change may follow a
decisive failure localization, and the 24-cell matrix remains conditional.

## 2026-07-22 current unique R0-R5 execution

The unique current candidate is `cd68d89dcc0854baa3c0107607086e801509b552` with formal root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_cd68d89_parallel_20260722_205506`.
Jobs `1180336--1180340` independently cover R0/R1, R2/R3 core, R2/R3 adapted, R4 and R5;
`1180341` is the R5 aggregate. The 24-cell R5 matrix is ActionFormer/TemporalMaxer x
uniform/learned x K384/K256 x three seeds. Formal paired cost is restricted to four
seed-3407 ActionFormer cells against the same-backend dense AdaTAD baseline. Superseded
`2645e68` Jobs `1180326--1180331` failed before mAP from a PyTorch 2.0 CUDA row-reset issue and
must not be reused. Current status is `experiment_running`; terminal mAP and costs are absent.

At `2026-07-22 21:04 +08:00`, all five cd68 model bundles `1180336--1180340` were running
concurrently on separate nodes and aggregate `1180341` was dependency-pending on R5 only. The
scoped log scan was clean; two earliest consumers entered epoch 0, but no successful optimizer-step
record, terminal artifact, official-validation mAP, or cost result existed. Do not upgrade the claim.
## 2026-07-22 21:45 DUCA 五点预算曲线已完整部署

当前预算实验由旧 `cd68d89` 的 K384/K256 24 cells 与新 `a00498e15d69294f78d0abeadfb47bc456db0b0e` 的 K320/K192/K128 36 cells 组成，共 60 个完整 official-validation terminal cells。新 Jobs 为 `1180356`（36 cells）、`1180357`（新矩阵聚合）和 `1180358`（五点官方 mAP + 选帧分布）；精确路径和口径见 `research-wiki/experiments/duca-five-point-budget-curve.md`。K/G 固定为 384/2、320/2、256/3、192/4、128/6。当前只有部署与门禁证据，没有五点 mAP。

## 2026-07-22 23:38 R0-R5 调度错误已倒带

旧 `cd68d89/a00498e` 作业存在两项已证实执行错误：R4/R5 各自重复完整 bootstrap；多 GPU bundle 的首个 `srun` step 继承全部 GPU，其他子臂实际串行。旧 Jobs `1180336--1180341,1180356--1180358` 已取消并保留日志/checkpoint。唯一当前身份改为 `codex/duca-boundary-burst-20260722@9f97f2c7f081b10fbf1f63d0602a621c6b43a780`，正式根为 `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343`，Jobs 为 `1180490--1180496`。R2/R3 四个子臂已在真实 Slurm 上同时 RUNNING，且每个 step 精确绑定一张 GPU。R4/R5 只消费一次共享前置 receipt，不再重复训练。完整账本见 `research-wiki/experiments/duca-r0-r5-9f97f2c-shared-bootstrap.md`；当前仍无 terminal mAP。

## 2026-07-23 06:54 R0 预注册停止条件生效

`1180493` 已完整生成 1000 次 paired bootstrap 与 `r0_summary.json`，随后按预注册规则以
`KILL_PROJECTED_FEASIBLE_SET`/exit `2` 终止；这不是代码异常。内部 training-holdout replay 中，
exact-uniform/R2Q3/R4Q5/unrestricted Oracle 的 Avg-mAP 为
`93.5871/94.1905/93.9992/93.9701`，但相对均匀采样的 95% CI 下界分别为
R2Q3 `-0.5300`、R4Q5 `-1.3154`、Oracle `-2.0322` 个百分点，均未严格超过预注册
`+0.20` 个百分点门槛。因此没有合法 selected projected family，R4/R5 原依赖作业
`1180494/1180495` 为 `DependencyNeverSatisfied`。这些 90+ 数值仅为冻结 detector 的内部
holdout replay，不是完整 validation terminal OpenTAD mAP；正在独立训练的正式臂继续用于最终
mAP 诊断，但当前不允许宣称边界微簇几何已获统计支持。

## 2026-07-23 09:02 Fast-only 终点官方 mAP

冻结 SlowFast Fast pathway + 固定 transition fusion R2Q3 的 Job `1180653` 已完成全部
`6000` 次优化更新和 `epoch_59.pth/state_dict_ema` 完整 THUMOS validation 评估。官方
Avg-mAP 为 `63.5297%`，mAP@0.3/0.4/0.5/0.6/0.7 为
`79.9106/74.5241/66.3665/54.7535/42.0937%`。该值没有达到项目希望守住的约 65% 水平，
但同协议 exact-uniform 尚在完成终点评估，因此暂时只能裁决为“Fast-only 冻结先验未证明优于
均匀采样”，不能精确报告 matched delta。R0 又已显示 R2Q3 几何没有可靠正 headroom，故当前
优先核验 matched uniform、非均匀 selected-axis 时间扭曲与 Fast motion 先验的边界偏移，禁止
无诊断地重跑同一 Fast-only 配置。

## 2026-07-23 09:18 R0 停止条件解释

R0 的 40-video paired bootstrap 中，R2Q3 差值为正的比例是 `85.2%`、超过 `+0.2 pp` 的比例是
`72.3%`，但 2.5% 分位仍为 `-0.5300 pp`，所以没有通过“95% CI 下界严格高于 +0.20 pp”的
预注册门槛。该结果来自 detector-seen 的 `training_internal_holdout` 冻结重放，schema 也明确标记
`diagnostic_only=true`。正确结论是“不能据此稳定选择 family”，不是“R2Q3 已被官方 mAP 否定”。

## 2026-07-23 09:11 matched exact-uniform 评估中

Job `1180637` 已正常完成 60 epoch 和 6000 次训练更新，正用 `epoch_59.pth/state_dict_ema`
执行 211-video 官方完整验证；09:11 为 `134/396` batches。mAP 尚未落盘，Fast-only
`63.5297%` 暂无合法 matched delta。

## 2026-07-23 09:26 matched exact-uniform 终点

matched exact-uniform 已完成官方完整 THUMOS validation：Avg-mAP `64.49%`，
mAP@0.3--0.7 为 `79.59/75.42/67.71/57.27/42.45%`。Fast-only `63.5297%`
相对其为 `-0.9603 pp`，且在 tIoU 0.4--0.7 全部更差，其中 tIoU=0.6
为 `-2.5165 pp`。因此当前冻结 Fast 运动先验 + R2Q3 是配对负结果，不得作为
主方法；下一步应诊断运动峰与真实动作边界的偏移、R2Q3 预算挤占和
selected-axis 时间扭曲，而不是原样重跑。

## 2026-07-23 09:28 稀疏粗扫尚无 mAP

稀疏 hidden-linear 四档已完成并封存 P0，但恢复 Job `1180696` 在首个
official optimizer update 前因 `sparse_probe_hidden_linear_d1...d4` 未登记到
selected-axis runtime 映射而 fail-closed。此为工程合同错误，不是方法性能结果；当前
不得声称 stride 4/8/12/16 中任何一档有下游 TAD mAP。

## 2026-07-23 09:38 R 系列必须收割 official mAP

R2Q3 `1180717`、R4Q5 `1180674`、soft-detached `1180685`、hard-detached
`1180686`、soft-adapted `1180687` 已明确与 R0 停止条件解耦，五臂均正常
训练。必须继续到 epoch-59 EMA 完整 THUMOS validation，并与 matched exact-uniform
`64.49%` 比较；在此之前不得根据 R0、P0 或中间 loss 对这些方法作性能裁决。

## 2026-07-23 09:44 均匀基线和中间 mAP 纠正

历史 `65.696` 来自改造后的 physical-grid/grid-aware ActionFormer，不是本项目标准
AdaTAD 均匀基线。历史 native stride-2 selected-axis 终点为 `64.31%`，与当前
exact-uniform `64.49%` 一致。R2Q3/R4Q5/soft/hard/adapted 协议不运行中间
validation，当前没有可报告的中间 mAP；训练 loss 不得代替性能。Fast-only
是参数无关的冻结特征转变 + R2Q3，不是 THUMOS 动作二分类器。

## 2026-07-23 10:02 checkpoint 曲线与更宽 selector 方向

六臂 checkpoint-EMA 完整验证诊断已由 Jobs `1180868/1180869` 启动，覆盖 one-based
epoch 10/20/30/40/50；终点 60 复用正式评估。中间结果只用于分析收敛，不得选择 checkpoint。

当前关键提交和历史 DUCA 树没有连续采样密度/逆 CDF 传输实现。新方向登记为
`ideas/duca-continuous-density-transport.md`，状态仅为 `designed`：粗模型输出平滑采样密度，
用 K 个固定分位点经过逆 CDF 得到连续位置，再作无参数唯一整数投影。它允许把最大间隔放宽至
10/15，并把背景预算全局迁移到多个边界微簇；后续是否实现须以六臂终点和 checkpoint 曲线为依据。

## 2026-07-26 curriculum Stage-1 evidence and Stage-2 recovery

The full-data uniform Stage-1 completed 30 epochs and produced the sealed
terminal EMA checkpoint `epoch_29.pth` (SHA-256
`7233fa6944659f432f8deaf22448b4a25cf8794b1e912f59a4d5b3715d54b39e`). Its
epoch-30 diagnostic validation Avg-mAP is `60.39`, which is not comparable
with the 60-epoch matched uniform baseline `64.49`. The original Stage-2
process never reached a model or optimizer update because an inherited legacy
P0 contract attempted to bind an empty variant. Commit `b554f04` removes that
incorrect P0 route. Its first Stage-2-only submission, Job `1190439`, ended at
zero runtime while the outer Slurm wrapper sourced `/etc/profile` under
nounset mode; no Python or optimizer update occurred. Job `1190528` is the
only replacement, with that wrapper-only repair and the same SHA-bound Stage-1
EMA. It must be judged solely by its final 60-epoch EMA evaluation.

At 2026-07-26 05:35 +08:00, Stage 2 has finite updates through epoch 8 and its
scheduled one-based epoch-5 validation is `60.52` Avg-mAP. This is a
non-selecting learning-curve diagnostic, not a 60-epoch comparison. The
separate e5 actionness/calibration/boundary diagnostic found an evaluator-entry
serialization bug before inference; `a1bf61d` fixes only deterministic protocol
serialization, passed 9 focused checks on the clean remote checkout, and Job
`1190633` then failed before loading the Stage-2 checkpoint because the wrapper
omitted the existing absolute pretrain override. Job `1190637` added that
override, strictly loaded the same SHA-bound EMA, and began official testing,
but was cancelled after `10/396` windows because it omitted the required fixed
64-window actionness/calibration/boundary export. It has no final mAP or quality
artifact and is not model evidence. The unique replacement must preserve the
same SHA-bound EMA and add that export before it starts. Job `1190643` is that
unique read-only replacement; it uses `selection_rule=none`, full official
validation, and exactly 64 fixed quality-export windows.

## 2026-07-26 e5 completed diagnostic and Stage-2 fail-closed update

Read-only Job `1190643` completed (`0:0`) on the exact Stage-2 epoch-5 EMA
SHA `4016ca083dd56286b6a32edce92f085945ce415c0b237c1fc43b0cb7ad1cc2a3`.
Full official THUMOS validation over 211 videos/396 windows reports Avg-mAP
`60.521318%` (`76.917970/71.979260/63.698867/52.930814/37.079679%` at tIoU
`0.3/0.4/0.5/0.6/0.7`), independently confirming the scheduled e5 curve
point. It is still a non-selecting early learning-curve observation, not a
terminal 60-epoch comparison. Its fixed 64-window/40-video selector-only
quality export contains no detector, GT, raw-prediction, or teacher input to
the selector: coarse actionness AUPRC/AUROC/Brier/ECE are
`0.343840/0.577626/0.202376/0.012158`; learned action enrichment is only
`1.003583` (CI `0.998122--1.008924`); learned r0 boundary recall is
`0.537405` (CI `0.482378--0.610760`) and its delta from matched uniform is
`-0.017789` (CI `-0.075717--0.046480`). Pure-delta same-feasible-DP has
stronger action enrichment `1.070008` and R2Q3/R4Q5 both-endpoint coverage
`0.506823/0.509369` versus learned `0.331581/0.356900`; this is mechanism
diagnosis only, not TAD validation evidence.

Stage-2 Job `1190528` made 1,000 finite updates and sealed `epoch_9.pth`,
then completed its scheduled one-based epoch-10 validation at diagnostic
Avg-mAP `61.62%` (`77.94/72.84/64.61/54.34/38.39%`). Immediately after
`Epoch 10 started`, its first next training forward triggered the formal
fail-closed pre-AMP non-finite-loss guard. It has no terminal EMA result and
is an affected-run numerical failure, not a negative offline TAD model result.
The timing coincides with the detector-gradient/contribution warmup boundary,
but causality is unproven. Do not resume or submit a replacement until a
read-only single-batch replay from `epoch_9.pth` identifies the non-finite
component and schedule weights; no Stage-1 rerun, `strict=False`, checkpoint
selection, or healthy-arm resubmission is permitted.

At 2026-07-26 06:50 +08:00, unique Job `1190683` was submitted for that
prerequisite and is pending Slurm priority. It is a read-only single-GPU
epoch-10 batch-0 replay under exact commit
`3a87132d60b0a328ccbe9d153e795a7ce3987911`, with a clean checkout, strict
Stage-1 initialization, training-state `epoch_9.pth/state_dict` SHA
`3d1444da7fbae2566ab379501db353900219d2bc23c918654db26e13833016fc`, and
the original absolute pretrain path. Its manifest explicitly prohibits
backward, optimizer, scheduler, EMA construction, checkpoint mutation, and
selection. Its result is component-local numerical diagnosis only; it cannot
be treated as an offline TAD performance result or authorize recovery until it
completes and is inspected.

Job `1190683` completed `0:0` at 06:51 +08:00 and preserves its input
checkpoint SHA before/after. Its exact epoch-10 batch-0 read-only forward has
finite cost `3.402128` and every recorded component finite at selector step
`1000`; detector-gradient, detector-contribution, and detector-utility
weights are all exactly zero. The contemporaneous original training log
records an `upsample_linear1d_backward_out_cuda` warning between `Epoch 10
started` and the pre-AMP failure, so batch 0 did complete backward and a
successful optimizer step advanced the curriculum to 1001 before the failing
forward. Thus `1190683` rules out batch-0/step-1000 and a zero-weight
contribution path, but does not establish the precise failing term. The sole
next diagnostic may inspect batch 1 with the selector step overridden only in
memory to `1001`; it must still avoid backward, optimizer/scheduler/EMA, or
checkpoint mutation.

Job `1190699` completed `0:0` at 06:56 +08:00 under exact commit
`45198e45af141605db3bda31ccc54a7ac58e4c8c`, with its input checkpoint SHA
unchanged. It reads epoch-10 batch 1 with only an in-memory selector clock of
1001, no backward, optimizer, scheduler, EMA, parameter, or checkpoint
mutation. Cost `4.324651` and every loss are finite. The first nonzero
detector-gradient/contribution weights (`1.542125e-7`/`6.168501e-7`) and
cls/reg contribution losses (`4.095707e-6`/`4.101499e-6`) are finite. The
schedule opening alone therefore does not reproduce the failure; recovery
remains blocked pending a separate explanation of the ephemeral post-batch-0
state or nondeterministic input/kernel interaction.

The post-update-state prerequisite is now complete. File-mode Job `1191745`
failed before model construction and is not numerical evidence. Corrected
module-mode Job `1191754` completed `0:0` under
`65a4cfb31716f84c153af881a71fe05069637848`: it restored the sealed epoch-9
model, optimizer, scheduler, EMA, and GradScaler in memory, executed real AMP
batch 0, advanced the selector/scheduler to 1001, and evaluated batch 1 under
controlled seeds 3407--3414. All eight batch-1 outcomes were finite. In every
trial, batch-0 gradients, 49,914,588 updated parameter values, and 56,069,713
optimizer-state values were finite, and the AMP scale remained 8192. This
rejects persistent post-update contamination and a deterministic boundary
failure. The checkpoint contains no serialized RNG state, so no exact replay
of the original stochastic draw is possible; the remaining supported cause
class is a one-shot pre-AMP stochastic/nondeterministic forward transient.

Commit `9519760a26cd7fda08c3e648b1e7d7f459b3b6b` implements the bounded,
fail-closed recovery only for the Stage-2 curriculum. It records every
pre-AMP non-finite attempt to `stage2/update_audit.json`, restores RNG,
buffers, and custom state, and replays the same batch up to eight times. No
optimizer, selector, scheduler, or EMA transition occurs until a finite
forward succeeds; exhaustion restores state and raises. Slurm precheck
`1191787` completed `0:0` with 15 focused tests passing and no model update,
checkpoint, or mAP. Exactly one recovery may now reuse the sealed epoch-9
source; it remains an offline TAD run and must use only terminal epoch-59 EMA
OpenTAD official mAP as its result.

The isolated continuation launcher is now prechecked: Job `1191796` completed
`0:0` at commit `adc6fb13114584188da4ac17eeeab6d89d69d04f` and created only a
hash-bound recovery manifest. It executes no training in precheck mode. The
previous `--mem=62200M` request was rejected before a job was created by the
current 55GB-per-GPU scheduler limit; the repaired request uses the partition
default memory. Exactly one Stage-2 continuation from the sealed epoch-9
source is authorized, with terminal epoch-59 EMA OpenTAD official mAP and
`stage2/update_audit.json` as its only performance and numerical endpoints.

That sole continuation, Job `1191806`, failed closed at epoch-10 batch 2:
two preceding updates were finite, but the pre-AMP cost remained non-finite
through eight state-restored same-batch replays. Its audit records 9 attempts,
8 replays, 9 restorations, one exhaustion, and only two optimizer/scheduler/
selector/EMA updates. Thus the condition is reproducible in the sealed
prefix-state path, not an acceptable isolated transient. It is not an offline
TAD performance result and has no terminal mAP. Do not repeat the continuation;
isolate it with one read-only two-update prefix diagnostic first.

### Stage-2 numerical resolution (2026-07-26 16:53 +08:00)

- `1191806` remains `tested` numerical-failure evidence with no terminal
  offline TAD mAP. At the reproducible batch-2 state, all contribution inputs
  and first-order gradients are finite; the sole defect is masking FP16 logits
  before temperature division, producing `-inf` invalid slots and `0 * -inf`
  in the contribution cross entropy.
- Commit `4c1f5384ae693c74a141619ded03196a72c594ed` scales logits before the
  invalid-position mask. It preserves valid logits and leaves the course
  schedule unchanged. Focused verification `1191853` passed 32 tests;
  read-only `1191854` makes the former batch finite (`cost=3.2601470947265625`,
  cls/reg contribution losses `1.4956006452848669e-05` and
  `1.2520967175078113e-05`) without state persistence or evaluation.
- Status: `implemented` and `tested` numerical repair, not
  `empirically_supported`. The only admissible next model action is one fresh
  strict Stage-2 continuation from sealed e9 at this commit; no Stage-1 rerun,
  intermediate checkpoint selection, or duplicate diagnostic is permitted.

- Deployment: launcher precheck `1191874` completed `0:0`. Job `1191880` is
  the sole `experiment_running` repaired continuation at commit
  `4c1f5384ae693c74a141619ded03196a72c594ed`, strict Stage-1 EMA/e9 bound,
  with the epoch-59 EMA OpenTAD official mAP as its sole performance endpoint.
  Its epoch-10 update 50/99 receipt is finite (total `3.5796`, contribution
  losses `0.0027/0.0026`, memory `10176MB`); this is health evidence only.

### Stage-2 course protocol correction (2026-07-26 18:07 +08:00)

- Job `1191880` was stopped by the experiment controller at `CANCELLED 0:0`
  after 700 finite post-e9 updates. Its audit records zero non-finite attempts,
  zero replays, and a one-to-one match among optimizer, selector, scheduler,
  and EMA updates. Numerically it is healthy, but it is not a valid offline
  TAD performance run.
- The resolved Stage-2 config inherited
  `intermediate_validation_selects_checkpoint=True` and wrote
  `best_validation_ema.json`, contrary to the course rule that intermediate
  mAP never selects a checkpoint. This pointer did not feed back into any
  completed optimizer/EMA update or early stopping, but its existence is a
  protocol violation; do not use the run's checkpoints or curve points as
  model evidence.
- Its one-based epoch-15 EMA validation was `62.403751%`
  (`79.036658/73.799613/64.984514/54.566999/39.630969%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`). It is retained only as operational audit data, not
  as a comparator, checkpoint choice, or terminal offline TAD mAP. The
  mandated AP/AUC/Brier/ECE/state-transition-boundary quality export was not
  produced at this checkpoint.
- The smallest correction overrides the Stage-2 config to
  `intermediate_validation_role=learning_curve_only` and
  `intermediate_validation_selects_checkpoint=False`, and makes the recovery
  precheck reject drift. A fresh strict continuation may be considered only
  after this corrected contract and the missing five-epoch read-only quality
  diagnostic path are prechecked; it must reuse the same sealed e9 source and
  retain epoch-59 EMA official mAP as its sole performance endpoint.

- Correction precheck `1191956` completed `0:0` at commit
  `42dba3f90b37243e7965d18b6707e88e81bf7109`, producing only a strict recovery
  manifest. It verifies the same sealed Stage-1/e9 hashes, diagnostic-only
  intermediate validation, no checkpoint selection, and the bounded replay
  policy. Job `1191957` is the sole new `experiment_running` continuation;
  its five-epoch quality exports will be read-only and post-training only.

- At `2026-07-27 00:59 +08:00`, `1191957` remains healthy at the same exact
  commit after 4,000 successful post-e9 updates through epoch 49. Its
  diagnostic-only epoch-50 EMA curve is `65.650497%` Average-mAP
  (`80.433202/76.607056/68.955569/58.776518/43.480139%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`). It has no selection pointer and is not a terminal
  offline TAD result or a formal matched-uniform comparison. The audit keeps
  exactly matched optimizer/scheduler/selector/EMA updates, four accepted
  one-retry AMP restores, zero loss non-finites, and zero replay exhaustion;
  terminal epoch-59 EMA official mAP is still absent.

## 2026-07-27 multi-round joint review absorption

The 2026-07-26 joint review is archived byte-identically at
`docs/methods/reviews/2026-07-26-duca-multiround-joint-review-raw.txt`
(SHA-256
`67409BC9B140275BFC6804DD65FACBBEB568719304768A322FCF3A3F54576484`);
the corrected project adjudication is
`docs/methods/2026-07-27-duca-multiround-joint-review-absorption.md`.
Its evidence-first governance, matched/equal-cost baselines, teacher-free
inference, teacher-fidelity checks, zero-training cross-pair, temporal-zoom
diagnostic, systematic-resampling novelty positioning and complete-cost
requirements are accepted.

The review is a dated snapshot, not the current execution contract. Draft PR
[#2](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/pull/2) is now
open and exposes immutable blobs at
`42dba3f90b37243e7965d18b6707e88e81bf7109`, so
`G0-read-surface=PASS`; independent source adjudication remains open. Both the
public base and DUCA branches still expose plaintext proxy authentication in
`README.md`, traceable to initial commit `7eb8a413`; current-surface redaction,
credential rotation and history treatment are three separate urgent actions.
Do not reproduce the secret in Wiki or reports.

The review's claim that repaired Stage-2 had not progressed beyond D0 is also
superseded. The FP16 mask-order cause is tested, and sole Job `1191957` has
4,000 successful post-e9 updates through epoch 49 with four bounded one-retry
AMP restores, zero loss non-finites/exhaustions and matched optimizer/
scheduler/selector/EMA updates. It was still `RUNNING` at
`2026-07-27 01:05 +08:00`; no terminal epoch-59 EMA offline TAD result exists.
Thus numerical recovery has strong runtime support while complete performance
and `paper_claim_allowed` remain unresolved/false.

The report's exact D1 thresholds, Path A/B mAP bands, eight-arm D2 matrix,
seed/budget counts and ChronoTransport parking are recorded only as
`designed_reviewer_proposal`. They do not override the canonical contract or
the repository's explicit ChronoTransport parallel-route rule without a new
project decision.

## DUCA 25% matched-rate curriculum (2026-07-27)

- The new experiment changes the complete course from `K=384/768` (50%) to
  `K=192/768` (25%). It is not a rate-only arm: Stage 2 retains `cls+reg`
  contribution distillation, `density_transport_st` detector gradient and full
  ASFormer adaptation. Stage 1 remains the matched 30-epoch exact-uniform
  warmup at the candidate budget.
- Exact implementation commit:
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2`, branch
  `codex/duca-rate25-curriculum-20260727`. Linux checks passed 18
  curriculum/config tests and 23 C3 regression tests.
- Precheck `1193418` is a zero-update launcher-order failure and not model
  evidence. Corrected GPU precheck `1193433` completed `0:0`.
- Formal Job `1193437` is the only `experiment_running` K=192 course. At
  `2026-07-27 01:31 +08:00` it entered Stage-1 epoch 0 on `g0015`, with an
  exact manifest and no Traceback/OOM/non-finite/fail-closed receipt.
- Its sole performance endpoint is terminal Stage-2 epoch-59 EMA OpenTAD
  official mAP. Intermediate mAP cannot select a checkpoint. Cross-budget
  comparison measures the 50%-to-25% shift; it does not replace a matched
  K=192 uniform control for attributing gains to learned sampling.
- At `2026-07-27 01:37 +08:00`, K=384 Job `1191957` had completed 4,500
  successful post-e9 updates through epoch 54 with four bounded AMP restores,
  zero loss non-finites/exhaustions and matched optimizer/scheduler/selector/
  EMA counters. K=192 Job `1193437` completed Stage-1 epoch 0 and entered
  epoch 1; batch 17 required two bounded AMP restores and then completed.
  Its final epoch-0 losses and `K=192` requested/effective budgets are finite
  and exact. Neither job has a new terminal offline TAD result.
- At `2026-07-27 02:08 +08:00`, K=384 Job `1191957` had completed 4,700
  successful post-e9 updates through epoch 56 and was training epoch 57. Its
  diagnostic-only epoch-55 EMA curve is `65.11%` Average-mAP
  (`79.99/75.71/68.05/57.90/43.88%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`), `-0.54pp` from epoch 50. A fifth bounded
  one-retry AMP restore completed successfully; loss non-finites and replay
  exhaustions remain zero, and all 4,700 update counters match.
- K=192 Job `1193437` completed Stage-1 epoch 4 with finite losses and exact
  `192/192` budget, sealed `epoch_4.pth`, and entered its scheduled one-based
  epoch-5 diagnostic. It has three accepted AMP replay attempts across two
  batches, no failure signature, and no Stage-2 or terminal result. Neither
  intermediate diagnostic may select a checkpoint or establish offline TAD
  performance.
- At `2026-07-27 02:44 +08:00`, K=384 Job `1191957` completed epoch 59,
  sealed `epoch_59.pth` (SHA-256
  `848abe3deace90e03b7fb3bca993a223d6155c037d7fed11c7c8b1e80eac9a87`),
  and closed its continuation audit at 5,000 successful post-e9 updates with
  matched optimizer/scheduler/selector/EMA counters. Its training-loop final
  EMA diagnostic is `65.385724%` Average-mAP
  (`80.193191/75.662461/68.607247/58.581766/43.883956%`), provisionally
  about `+0.896pp` over matched exact-uniform `64.49%`, with larger gains at
  tIoU `0.6/0.7` (`+1.312/+1.434pp`). The explicit terminal OpenTAD evaluator
  loaded the same `epoch_59.pth/state_dict_ema` and remains running, so the
  terminal result is not sealed yet.
- K=192 Job `1193437` reached Stage-1 epoch 8 after its one-based epoch-5 EMA
  diagnostic: `8.730398%` Average-mAP
  (`19.988334/12.780593/6.844830/3.013880/1.024351%`). This is an early
  exact-uniform warmup curve, not a terminal or checkpoint-selection result.
  Its separate AP/AUC/Brier/ECE and transition-boundary quality export is
  contractually deferred until the Stage-1 checkpoint set is sealed. Both
  source snapshots remain clean at their exact commits, and neither run has a
  selection pointer or failure signature.
- At `2026-07-27 03:14 +08:00`, the explicit K=384 epoch-59 EMA OpenTAD
  evaluator completed all 211 videos and 422,000 predictions and independently
  reproduced `65.385724%` Average-mAP
  (`80.193191/75.662461/68.607247/58.581766/43.883956%`). Job `1191957`
  then ended `FAILED/1:0` only because the config had
  `post_processing.save_dict=False` while structured `--metrics-json`
  evidence requires a saved final prediction file. This is a post-metric
  evidence-packaging failure, not model or metric failure. Evaluation-only
  repair Job `1193610` is pending at the same clean exact commit and sealed
  epoch-59 EMA; it may only enable final prediction saving and write the
  prediction hash plus `terminal_evaluation.json`. It does not resume
  training, change Job `1191957`, select a checkpoint, or permit raw-prediction
  replay. Until that receipt seals, report the duplicated official metric and
  the packaging failure together.
- K=192 Job `1193437` remains healthy and completed its one-based epoch-10 EMA
  diagnostic at `27.82%` Average-mAP
  (`50.03/39.49/28.41/15.44/5.75%`), then entered Stage-1 epoch 10. This is
  exact-uniform warmup learning-curve evidence only; it cannot select a
  checkpoint or establish learned-selector or terminal K=192 performance.
- At `2026-07-27 03:33 +08:00`, evaluation-only Job `1193610` completed
  `0:0` and sealed the K=384 terminal epoch-59 EMA OpenTAD receipt. It binds
  clean exact commit `42dba3f90b37243e7965d18b6707e88e81bf7109`,
  checkpoint SHA-256
  `848abe3deace90e03b7fb3bca993a223d6155c037d7fed11c7c8b1e80eac9a87`,
  211 videos and 422,000 predictions. Prediction SHA-256 is
  `b7a26f270d0ed4e3f7036793dd4c48fe6011e7b15f2570525843ab0cfb7497f1`
  and evaluation SHA-256 is
  `d239a1be1f2eaff15d310a6ee8cceaa36b5d8f70ee3b3516d6cb44cd7e049b74`.
  The sealed official result is `65.385724%` Average-mAP
  (`80.193191/75.662461/68.607247/58.581766/43.883956%`), approximately
  `+0.896pp` over matched exact-uniform `64.49%`, with
  `+1.312/+1.434pp` at tIoU `0.6/0.7`. Job `1191957` remains
  `FAILED/1:0` only as the immutable post-metric packaging-failure record;
  no training or checkpoint selection was rerun.
- At `2026-07-27 03:37 +08:00`, K=192 Job `1193437` remains healthy and
  entered Stage-1 epoch 14. Its latest sealed checkpoint is epoch 9; the
  epoch-15 diagnostic and post-Stage-1 AP/AUC/Brier/ECE/boundary-quality
  exports are still pending. No Stage-2 initialization, selection pointer,
  Traceback, OOM, non-finite-loss failure or fail-closed receipt exists.
- At `2026-07-27 04:07 +08:00`, K=192 Job `1193437` sealed its one-based
  epoch-15 EMA diagnostic from `epoch_14.pth/state_dict_ema` and entered
  Stage-1 epoch 16. The exact-uniform warmup curve is `35.616501%`
  Average-mAP
  (`59.176347/48.839082/36.417352/22.762136/10.887588%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions. It remains
  learning-curve evidence only and cannot select a checkpoint or establish
  terminal or learned-selector K=192 performance. The source is still clean
  at exact commit `ed0d4900bffe3546997ea1f00ae806d82cad55f2`; no Stage-2
  initialization, selection pointer, Traceback, OOM, non-finite-loss failure
  or fail-closed receipt exists.
- At `2026-07-27 05:07 +08:00`, K=192 Job `1193437` sealed its one-based
  epoch-20 EMA diagnostic from `epoch_19.pth/state_dict_ema` and entered
  Stage-1 epoch 23. The exact-uniform warmup curve is `43.132056%`
  Average-mAP
  (`65.227910/55.799303/44.794554/32.149065/17.689446%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions. It remains
  diagnostic-only and cannot select a checkpoint or establish learned-selector
  or terminal K=192 performance. The exact source remains clean; no Stage-2
  initialization, selection pointer, Traceback, OOM, non-finite-loss failure
  or fail-closed receipt exists.
- At `2026-07-27 06:07 +08:00`, K=192 Job `1193437` sealed its one-based
  epoch-25 EMA diagnostic at `49.091036%` Average-mAP
  (`69.158801/61.778057/51.280304/38.465483/24.772537%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions. It remains
  exact-uniform warmup learning-curve evidence only. All 30 Stage-1 training
  epochs then completed and sealed `epoch_29.pth` with SHA-256
  `141e4c1f3ce7b1b11a477fecf59478694055b8897102180137f007a825fe2595`.
  The epoch-30 EMA diagnostic and AP/AUC/Brier/ECE/boundary-quality exports
  are pending, and strict Stage-2 initialization has not begun. No selection
  pointer or failure signature exists.
- At `2026-07-27 06:37 +08:00`, the K=192 Stage-1 epoch-30 EMA diagnostic
  from sealed `epoch_29.pth/state_dict_ema` completed at `51.954148%`
  Average-mAP
  (`70.747919/64.400867/54.530413/41.835973/28.255569%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions. This closes
  the exact-uniform warmup learning curve but is not the terminal full-course
  K=192 offline TAD endpoint. The launcher is now exporting the one-based
  epoch-5 selection-quality records; AP/AUC/Brier/ECE/boundary-support
  summaries and strict Stage-2 initialization remain pending. No checkpoint
  selection or failure signature exists.
- At `2026-07-27 07:10 +08:00`, K=192 Job `1193437` remains healthy at its
  clean exact commit. Stage-1 selection-quality summaries for epochs
  5/10/15 are sealed. Coarse macro AP rises
  `0.332623 -> 0.378387 -> 0.406062`, macro AUC rises
  `0.485973 -> 0.542316 -> 0.575196`, pooled Brier falls
  `0.233573 -> 0.222322 -> 0.217861`, and pooled ECE falls
  `0.118097 -> 0.060039 -> 0.024695`. Transition macro policy AUPRC at
  `r0/r1/r2/r4/r8` reaches
  `0.031783/0.071428/0.110247/0.177410/0.287755` at epoch 15. Because
  Stage 1 is exact uniform, action enrichment (`0.996621`), boundary recall
  (`0.276691/0.759128/1/1/1`), bilateral endpoint coverage
  (`0.089063/0.584968/1/1/1`), endpoint distance (`0.964181`) and maximum
  hole (`3.735113`) are invariant, and every learned-minus-uniform delta is
  exactly zero by construction. This validates the matched control; it is
  not learned-selector evidence. Epoch-20 quality export is running, while
  Stage-2 initialization and terminal K=192 offline TAD performance remain
  pending. No failure signature or checkpoint-selection pointer exists.
- At `2026-07-27 07:37 +08:00`, the K=192 Stage-1 epoch-20/25 quality
  summaries completed. Epoch-25 macro AP/AUC is `0.410307/0.579620`, pooled
  AP/AUC is `0.367067/0.571312`, Brier is `0.215715`, and ECE is `0.021642`;
  transition macro policy AUPRC at `r0/r8` is `0.031584/0.292148`. The small
  epoch-15-to-25 macro AP/AUC change (`+0.004245/+0.004424`) indicates that
  Stage-1 coarse evidence is approaching a plateau. Exact-uniform selection
  geometry and all paired learned-minus-uniform deltas remain unchanged by
  construction. Epoch-30 quality export is in progress; Stage-2 has not
  initialized. Job `1193437` remains healthy with null dependency and clean
  independently verified source HEAD
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2`; no failure or checkpoint-
  selection evidence exists.
- At `2026-07-27 08:07 +08:00`, K=192 Stage-1 sealed all six quality
  checkpoints. Epoch-30 macro AP/AUC is `0.418318/0.584329`, pooled AP/AUC is
  `0.372005/0.574455`, Brier/ECE is `0.215742/0.021818`, and transition macro
  policy AUPRC at `r0/r8` is `0.032568/0.294064`. Exact-uniform geometry and
  every learned-minus-uniform delta remain invariant, so this is the complete
  warmup diagnostic curve, not learned-selector or terminal evidence.
  Stage-2 then strictly initialized the full model from
  `epoch_29.pth/state_dict_ema` at the matching SHA
  `141e4c1f3ce7b1b11a477fecf59478694055b8897102180137f007a825fe2595`,
  resetting only the approved schedule step. Through epoch 2 it completed 300
  successful optimizer/EMA/scheduler/schedule updates. One AMP attempt was
  restored and replayed successfully once; non-finite-loss attempts and
  replay exhaustions remain zero. Epoch 3 is running with finite losses and
  exact budget 192. The detector-gradient and teacher-utility weights are
  still zero only because the registered bridge is delayed to step 2,100 and
  then ramped for 1,500 steps, not because those plugins are disabled. No
  checkpoint selection or hard failure exists.
- At `2026-07-27 08:37 +08:00`, K=192 Stage-2 sealed its one-based epoch-5
  EMA diagnostic at `53.426779%` Average-mAP
  (`71.631253/64.833503/56.501860/44.442697/29.724581%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`). This is `+1.472631pp` over the Stage-1 epoch-30
  diagnostic, with the largest change at tIoU 0.6 (`+2.606724pp`), but it is
  an intermediate learning-curve point and cannot select a checkpoint or
  isolate learned selection from uniform. Epoch-4 audit evidence contains
  500 successful optimizer/EMA/scheduler/schedule updates; two AMP attempts
  each recovered on bounded retry 1/8, with zero non-finite-loss attempts or
  replay exhaustion. The run entered epoch 5. Detector-gradient and
  contribution-teacher weights remain in their registered pre-bridge delay,
  so the intermediate gain cannot be assigned to those later terms. No hard
  failure or checkpoint-selection pointer exists.
- At `2026-07-27 09:37 +08:00`, K=192 Stage-2 epoch-10 EMA reached
  `54.515667%` Average-mAP
  (`72.941354/65.978947/57.882904/45.013669/30.761460%`), which is
  `+1.088889pp` over epoch 5 and `+2.561519pp` over the Stage-1 endpoint.
  This remains non-selecting intermediate evidence. At successful step 1,099,
  cls/reg contribution-distillation losses became finite and nonzero
  (`0.0115/0.0113`) and `duca_detector_grad_w` became `0.0015`, proving that
  contribution distillation and the `density_transport_st` path have entered
  their scheduled training phase; the separate detector-utility weight is
  still zero. The audit has 1,100 successful updates, with the same two
  recovered AMP skips and zero non-finite-loss attempts or replay exhaustion.
  Epoch 11 is running with no hard failure or checkpoint-selection pointer.
- At `2026-07-27 10:16 +08:00`, K=192 Stage-2 epoch-15 EMA reached
  `55.403415%` Average-mAP
  (`73.319852/67.646858/58.673394/45.968481/31.408491%`), which is
  `+0.887748pp` over epoch 10 and `+3.449267pp` over the Stage-1 endpoint.
  This is still a non-selecting intermediate diagnostic. The epoch-14 audit
  has 1,500 successful optimizer/EMA/scheduler/schedule updates. A third
  isolated AMP skip recovered on retry 1/8; non-finite-loss attempts and
  replay exhaustions remain zero. At step 1,499, finite cls/reg contribution
  losses (`0.7043/0.6853`) and `duca_detector_grad_w=0.0365` confirm continued
  plugin engagement with exact requested/effective budget `192/192`; the
  separate detector-utility weight remains zero. Epoch 15 is running with no
  hard failure or checkpoint-selection pointer.
- At `2026-07-27 11:07 +08:00`, K=192 Stage-2 epoch-20 EMA reached
  `56.050489%` Average-mAP
  (`73.483220/67.033731/58.709834/47.921247/33.104414%`), which is
  `+0.647074pp` over epoch 15 and `+4.096341pp` over the Stage-1 endpoint.
  The five-epoch change is concentrated at high tIoU 0.6/0.7
  (`+1.952766/+1.695923pp`), while tIoU 0.4 changes by `-0.613127pp`.
  This remains non-selecting intermediate evidence. The audit has 2,000
  successful optimizer/EMA/scheduler/schedule updates; four isolated AMP
  skips recovered on retry 1/8, with zero non-finite-loss attempts or replay
  exhaustion. At step 1,999, finite cls/reg contribution losses
  (`2.5477/2.5470`) and `duca_detector_grad_w=0.1248` confirm continued
  plugin engagement; the separate detector-utility weight remains zero.
  Epoch 20 is running with no hard failure or checkpoint-selection pointer.
- At `2026-07-27 12:19 +08:00`, K=192 formal Job `1193437` is terminal
  `FAILED/1:0`, but the failure boundary is isolated after epoch-24 training
  and checkpoint save: the one-based epoch-25 non-selecting EMA evaluation
  failed at 147/396 batches because a Decord worker exhausted 10,240 EOF
  retries. There is no epoch-25 mAP receipt and no update after the sealed
  `epoch_24.pth` (SHA-256
  `d37cad6e1fcbf9078f9e186c0735f291461332572df67ef6df16ab05db3c00f6`).
  The source audit (SHA-256
  `ec1536dfd68d16144e242c8ca7ee10828b5de05d5fea4f26b968e21b7a1dcf9d`)
  contains 2,500 successful updates, six bounded AMP skips, maximum retry
  2/8, and zero non-finite-loss attempts or replay exhaustion. This is a data
  decoder/evaluation failure, not OOM, non-finite training, model failure or
  terminal K=192 performance.
- Minimal affected-arm recovery precheck Job `1194469` completed `0:0`.
  Formal recovery Job `1194471` is `RUNNING` at the same clean exact commit
  with null dependency. Its only runtime change is bounded
  `DECORD_EOF_RETRY_MAX=20480`; before any new training update it must
  reproduce the missing epoch-25 EMA evaluation, then resume epochs 25--59
  with strict state loading and a separately audited 3,500-update
  continuation. Source plus continuation must total 6,000 successful
  updates, with no intermediate checkpoint selection. The source checkpoint
  lacks global RNG state, so model/EMA/optimizer/scheduler/GradScaler/DUCA
  schedule restoration is exact but random-stream continuation is not
  bit-exact; this limitation is sealed in the recovery manifest and must
  accompany terminal evidence.
- At `2026-07-27 12:26:54 +08:00`, recovery Job `1194471` crossed the
  original decoder failure position, reaching 169/396 epoch-25 EMA evaluation
  batches without Decord error, Traceback, OOM or non-finite evidence. No
  continuation training or new update audit has started, preserving the
  evaluate-before-update gate. This supports the bounded decoder repair but
  is not an mAP result or full-evaluation completion.
- At `2026-07-27 12:41 +08:00`, recovery Job `1194471` completed the missing
  one-based epoch-25 EMA diagnostic from sealed
  `epoch_24.pth/state_dict_ema`: official OpenTAD Average-mAP is
  `56.646995%`
  (`73.566280/67.812407/59.759908/48.284381/33.811996%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions.
  Prediction SHA-256 is
  `17c9fce0f909eed7d08e82ba3cd133c68bf681c75635b3e2edeb946c1674d422`
  and receipt SHA-256 is
  `e9069dc8d621268014f32761759f3a590f2ff9d85ac822a9ec09425423894638`.
  The point is `+0.596506pp` over epoch 20 and `+4.692847pp` over the
  Stage-1 endpoint, with the largest five-epoch gain at tIoU 0.5
  (`+1.050074pp`). It is still non-selecting intermediate evidence and
  cannot establish terminal K=192 performance or learned-selector gain over
  a missing matched K=192 uniform endpoint.
- The recovered evaluation finished before any continuation update. Only
  after the receipt sealed did Job `1194471` strictly restore the epoch-24
  model/EMA, optimizer, scheduler, GradScaler and DUCA schedule and enter
  epoch 25 at `12:38:28`. At `12:41:24` it remains running with null
  dependency and no Decord error, Traceback, OOM, non-finite evidence or
  checkpoint-selection pointer. The recorded lack of source global RNG state
  still prevents a bit-exact random-stream continuity claim.
- At `2026-07-27 12:45 +08:00`, the K=192 continuation completed epoch 25
  and entered epoch 26. Its first 100 resumed updates are finite, advancing
  the restored selector schedule through step 2,599. Final epoch-25 total,
  cls/reg and cls/reg contribution-distillation losses are respectively
  `11.6996`, `0.2304/0.2485` and `5.1453/5.0894`;
  `duca_detector_grad_w=0.2260` and requested/effective budget remains
  exactly `192/192`. No hard error or replay exhaustion exists. The large but
  finite contribution terms remain under trend monitoring rather than being
  classified as a non-finite failure.
- At `2026-07-27 13:07 +08:00`, recovery Job `1194471` completed epochs
  25--29 and sealed `epoch_29.pth` (SHA-256
  `4ec40e031af6087ff4db509df333e62da3514440d55d3615907cdd8ec2acd2dc`)
  at scheduler and selector step 3,000. Continuation audit SHA-256
  `fc9860d56fc80980f1bdedd050ebf390f15644daf5f16f7b4fbb29576f1f81f4`
  records exactly 500 successful optimizer/EMA/scheduler/schedule updates,
  zero AMP skips, zero non-finite losses and zero replay exhaustion. The
  homotopy reached schedule progress `1.0000` and detector-gradient weight
  `0.2500` with finite cls/reg and contribution losses.
- Logged effective-budget means `128.5` and `153.0` are accepted
  short-valid-sequence cases, not budget drift: exact code uses
  `effective_k=min(K, valid_len)`, the source run contains the same pattern,
  requested budget stays 192, and full-length batches remain `192/192`.
  The non-selecting epoch-30 EMA evaluation is running from sealed
  `epoch_29.pth`; no new mAP, best-checkpoint pointer or hard failure exists.
- At `2026-07-27 13:38 +08:00`, recovery Job `1194471` completed the
  non-selecting one-based epoch-30 EMA evaluation from sealed
  `epoch_29.pth/state_dict_ema`: official OpenTAD Average-mAP is
  `57.464558%`
  (`74.527791/69.177787/60.361168/49.020347/34.235695%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions.
  Intermediate receipt SHA-256 is
  `30fdb7579505e04619f932a0673701ed6733152540ad7092733b4c225750ea39`.
  This is `+0.817563pp` over epoch 25 and `+5.510410pp` over the Stage-1
  endpoint. Its apparent `-7.921166pp` gap to the sealed K=384 terminal
  endpoint is still intermediate-to-terminal; the point cannot select a
  checkpoint, establish terminal K=192 offline TAD performance, or prove
  learned-selector gain without a matched K=192 uniform endpoint.
- Continuation epochs 30--31 then completed and epoch 32 started. The
  `13:38` audit snapshot records 700 successful updates from 700 attempts,
  zero AMP skips, non-finite losses or replay exhaustion. Final epoch-31
  total, cls/reg and cls/reg contribution losses are finite at `12.2577`,
  `0.2188/0.2354` and `5.4511/5.5401`; detector-gradient weight and schedule
  progress are `0.2500/1.0000`, with budget `192/192`. Job `1194471` remains
  running with null dependency, no hard failure or best-checkpoint pointer,
  and the sealed global-RNG continuity limitation remains in force.
- At `2026-07-27 14:12 +08:00`, recovery Job `1194471` completed the
  non-selecting one-based epoch-35 EMA evaluation from sealed
  `epoch_34.pth/state_dict_ema`. Official OpenTAD Average-mAP is
  `57.921948%`
  (`74.668088/69.304388/61.149872/49.416606/35.070788%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions.
  Checkpoint SHA-256 is
  `2da7de83c0e9feb1f3b267deed7a41593680305b2491d47d4352a81254ac4a02`;
  intermediate receipt SHA-256 is
  `00d22c18c188738d63b9f065a912ba08dfb74cf5d78df7e24d2fabe48d1784cc`.
  This is `+0.457390pp` over epoch 30 and `+5.967800pp` over the Stage-1
  endpoint; tIoU 0.5/0.7 improve by `+0.788703/+0.835092pp`. Its apparent
  `-7.463776pp` gap to sealed K=384 terminal performance remains
  intermediate-to-terminal and cannot select a checkpoint, establish
  terminal K=192 offline TAD performance, or isolate learned selection
  without a matched K=192 uniform endpoint.
- The epoch-34 continuation audit records 1,000 successful updates. A single
  epoch-33 batch incurred three bounded AMP-overflow skips, restored exact
  state after each attempt, reduced scale from 512 to 128 and then completed.
  There are zero non-finite-loss attempts or replay exhaustion, so this is an
  accepted finite transient under the bounded AMP contract, not model
  failure. Job `1194471` entered epoch 35 at the clean exact commit with null
  dependency, no hard failure or best-checkpoint pointer, and the global-RNG
  continuity limitation remains in force.
- At `2026-07-27 15:08 +08:00`, recovery Job `1194471` completed the
  non-selecting one-based epoch-40 EMA evaluation from sealed
  `epoch_39.pth/state_dict_ema`. Official OpenTAD Average-mAP is
  `58.116412%`
  (`73.838536/69.371290/61.536349/50.339880/35.496002%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions.
  Checkpoint SHA-256 is
  `c1a9c6393920f1189000800362564239d5b9ee38ef94c00373f6fe6551b1f445`;
  intermediate receipt SHA-256 is
  `5ef548dd4ab4cb8f7bdadc5938ce3670d1c075620a35279df39b99cc953f547e`.
  This is `+0.194464pp` over epoch 35 and `+6.162264pp` over the Stage-1
  endpoint. Relative to epoch 35, tIoU 0.3 changes by `-0.829552pp`, while
  tIoU 0.4--0.7 change by
  `+0.066902/+0.386477/+0.923274/+0.425214pp`. Its apparent
  `-7.269312pp` gap to sealed K=384 terminal performance remains
  intermediate-to-terminal; it cannot select a checkpoint, establish
  terminal K=192 offline TAD performance, or isolate learned selection
  without a matched K=192 uniform endpoint.
- The epoch-40 continuation audit SHA-256 is
  `8bb3290894c0d3642a5326149313272c1357e862445468dcce3dd344a2cd21ee`.
  It records 1,600 successful updates from 1,600 attempted batches and
  1,603 optimizer attempts. The three AMP skips/restorations remain confined
  to the previously sealed single epoch-33 batch; non-finite-loss attempts
  and all replay exhaustions remain zero. By `15:11 +08:00`, Job `1194471`
  completed epoch 41 and entered epoch 42 from the clean exact commit with
  null dependency, no hard failure or
  best-checkpoint pointer, and the global-RNG continuity limitation remains
  in force.
- At `2026-07-27 15:37 +08:00`, recovery Job `1194471` completed epochs
  40--44 and sealed `epoch_44.pth` (SHA-256
  `fabe373abd4e7f2f982bf6a6fad26e7d022930ea4da8789d400f613200c8c9ea`).
  Audit SHA-256
  `6991f412001928853a3976baa07dc5c2b4bb46c288cc9788e3ad2c4bb2219413`
  records exactly 2,000 successful updates, with the same three bounded AMP
  restores from one epoch-33 batch, zero non-finite-loss attempts and zero
  replay exhaustion. The non-selecting epoch-45 EMA evaluation is in progress
  and has no metric JSON yet. The job remains running from the clean exact
  commit with null dependency, no hard failure or best-checkpoint pointer;
  epoch 40 remains the latest completed mAP point.
- At `2026-07-27 15:44 +08:00`, the non-selecting one-based epoch-45 EMA
  evaluation completed from sealed `epoch_44.pth/state_dict_ema`. Official
  OpenTAD Average-mAP is `57.877041%`
  (`74.416763/68.892371/61.103833/49.558953/35.413287%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions.
  Intermediate receipt SHA-256 is
  `4e13c794f1cf5ad709081e227833abf422be245f102ce97feb7a9f2f2b902670`.
  This is `-0.239371pp` versus epoch 40 and `+5.922893pp` over Stage-1.
  The decline is concentrated at tIoU 0.4--0.6, while tIoU 0.3 improves
  `+0.578227pp`. The point remains non-selecting and cannot establish
  terminal K=192 offline TAD performance or learned-selector gain without a
  matched K=192 uniform endpoint. Job `1194471` entered epoch 45 and remains
  running at exact commit
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2` with null dependency; no
  hard-failure or checkpoint-selection evidence exists.

## 2026-07-27 training-budget correction and model-first priority

- The sealed K=384 result `65.385724%` is produced by 30 epochs of full-model
  exact-uniform training followed by 60 epochs of full-model joint training.
  Its comparable optimization budget is therefore 90 epochs, not the official
  60-epoch budget. It remains a valid over-budget curriculum candidate and
  diagnostic, but is no longer treated as the decisive fair-budget DUCA paper
  result against the 60-epoch `64.49%` uniform anchor.
- Stage-2 epoch 50 reached the best observed diagnostic result
  `65.650497%`, but the total model budget at that point is still 80 epochs.
  Any best-checkpoint result must use the same held-out selection rule and the
  same maximum update budget for every compared arm.
- The dominant model hypothesis from the original multi-round source reviews
  is a physical-time mismatch: nonuniformly selected observations are consumed
  by the detector as an equally spaced sequence. Post-hoc inverse mapping
  cannot repair features, assignment or regression targets already computed
  under the selected-rank metric.
- The next paper-facing model target is a budget-conserving transport on
  physical time: exact-K monotone density transport, explicit timestamp/gap
  conditioning, and assignment/regression in physical coordinates. Uniform
  sampling is retained as a special case. Raw contribution-magnitude matching
  should be tested against normalized ranking or transport supervision because
  its current terms dominate the K=192 Stage-2 loss.
- The immediate decisive experiments are scientific controls, not a broad
  engineering matrix: total-60 joint-from-scratch, total-60 short-warmup
  curriculum, a 90-epoch uniform control, and selected-axis versus
  physical-time detector geometry at K=384 and K=192. Independent seeds follow
  only after a fair-budget model change is positive.

## 2026-07-27 pure pre-backbone recovery superseding physical-head priority

- The canonical upstream AdaTAD VideoMAE-S `768/160` THUMOS result is
  `69.03%` Average-mAP
  (`83.90/79.01/72.38/61.57/48.27%` at tIoU `0.3--0.7`), as recorded in
  `configs/adatad/README.md`. The local approximately `68.29%` dense result
  is not yet an audited reproduction of the same released weight,
  checkpoint-selection rule and training schedule.
- The current `64.49%` K=384 exact-uniform result is repeatable evidence for
  the DUCA selected-axis sparse wrapper, but it is not a clean native
  official AdaTAD 1/2-downsampling baseline. The wrapper extends the detector,
  remaps targets and predictions, changes the 768-to-384 temporal surface and
  changes several runtime/training settings. At exact commit `42dba3f9`, both
  the active `ActionFormer` and `AnchorFreeHead` source files are extended and
  are not byte-identical to upstream; nominal head configuration, projection,
  cls/reg objectives and NMS remain official-derived, but execution parity has
  not been established. A clean upstream native-uniform K=384 result and its
  K=192 counterpart are still missing.
- The original curriculum contract did not prescribe 30 detector-training
  epochs followed by 60 more detector-training epochs. It allowed a
  detector-free frontend P0, then exactly 6,000 detector updates in one
  official-60 course: exact-uniform warmup followed by staged release of the
  selector, normalized contribution supervision and detector-to-selector
  gradient. Current K=384 and K=192 30+60 courses are over-budget diagnostics.
- The paper-facing method is again restricted to a detector-agnostic
  pre-backbone acquisition plugin. It may perform exact-K monotone selection
  and a generic inverse coordinate map, but the downstream backbone,
  projection, assignment, regression, head, losses and NMS must remain
  unchanged in the main claim. Physical-time injection into the detector is
  retained only as a diagnostic or explicitly labeled enhanced integration,
  not as the pure-plugin result.
- The theory target is bounded task-aware temporal reparameterization:
  exact constant budget, uniform identity path, monotone inverse-CDF sampling,
  bounded local dilation/compression, and normalized cls/reg utility transport
  with verified hard-swap alignment. This supersedes an unconstrained
  collection of sampling plugins as the main CVPR model story.
- Target-domain train-free selector experiments exist, but no frozen prior has
  beaten the K=384 wrapper-uniform control. Fast-only SlowFast reaches
  `63.5297%`; MobileNet feature-change reaches about `63.27%`; fixed fusion
  reaches about `64.33%`. These experiments still train the target detector.
  A truly plug-and-play experiment with both selector and released detector
  frozen has not been completed.
- The next decisive model loop uses one total 60-epoch detector budget and one
  shared best-checkpoint rule: clean native uniform, bounded density from
  scratch, short uniform warmup plus release, normalized contribution
  ranking/transport, then detector-gradient feedback only after alignment.
  Additional seeds and detector families follow a clear first-seed gain.
- Canonical synthesis and evidence boundaries:
  `research-wiki/duca_prebackbone_plugin_and_baseline_recovery_contract.md`.

## 2026-07-27 approved total-60 pure-plugin experiment

- The user approved the clean official baseline recovery, pure pre-backbone
  bounded monotone density model, fair five-arm total-60 comparison, temporal
  warp/contribution ablations, multi-seed and second-detector follow-up, and a
  separate frozen-detector plug-and-play mode.
- A one-frame remove/add swap is retained only as the smallest local
  finite-difference test. It is not sufficient evidence by itself. The
  detector-gradient gate now also requires dispersed 1%/5%/10% multi-frame
  swaps, contiguous-block swaps, and full hard re-decoding after
  0.25/0.5/1.0 density steps.
- Direct detector gradient may enter the final model only when local and
  multi-frame utilities both align with the true frozen-detector loss.
  Otherwise the final arm uses detached normalized rank/transport supervision.
- Canonical written design:
  `docs/superpowers/specs/2026-07-27-duca-total60-prebackbone-plugin-cvpr-design.md`.
- At `2026-07-27 17:21 +08:00`, K=192 recovery Job `1194471` completed the
  non-selecting one-based epoch-50 and epoch-55 EMA diagnostics. Their official
  OpenTAD Average-mAP values are respectively `58.383005%` and `58.082562%`;
  epoch 55 is `-0.300443pp` versus epoch 50 and `+6.128414pp` over the
  Stage-1 endpoint. The epoch-54 audit records 3,000 continuation updates,
  giving 5,500/6,000 combined Stage-2 updates. The job entered epoch 55 and
  remains healthy at exact commit
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2`, with null dependency, no hard
  failure and no best-checkpoint pointer. These points cannot select a model;
  only the predeclared epoch-59 EMA evaluation is terminal K=192 evidence.
- At `2026-07-27 17:49 +08:00`, recovery Job `1194471` completed Stage-2
  epoch 59 and sealed `epoch_59.pth` with SHA-256
  `4a5389506263b8fd76ca3de6ce3475dee64cc0d9ed1ca73c896692c8db288455`.
  Its continuation audit SHA-256
  `36fc64d4542ce671b4c891f8b8270a51b629ad4b514165f80fd66b439a7451f0`
  records 3,500/3,500 successful continuation updates; together with the
  source audit this is exactly 6,000/6,000 Stage-2 updates. Three bounded AMP
  restores remain confined to one epoch-33 batch, with zero non-finite-loss
  attempts or replay exhaustion. The terminal epoch-59 EMA OpenTAD evaluation
  is running and had reached 126/396 batches; no terminal mAP or metric receipt
  exists yet. Job `1194471` remains healthy at exact commit `ed0d4900...`,
  with null dependency, no hard failure and no best-checkpoint pointer.
- At `2026-07-27 18:20 +08:00`, the explicit terminal evaluator loaded
  `epoch_59.pth/state_dict_ema` and exactly reproduced the automatic final
  diagnostic. The sealed K=192 30+60 course reaches official OpenTAD
  Average-mAP `57.967272%`, with
  `73.907179/68.926135/61.194230/49.841145/35.967670%` at tIoU
  `0.3--0.7`, over 211 videos and 422,000 predictions. Terminal receipt,
  prediction and combined-update-audit SHA-256 values are respectively
  `febc59d463476bcf6a1a0d77f237a54f12c59ce3028ce623fba9844c07fada04`,
  `719ba43b0f76f5647b2394a23b622aff0bac0c17a54d15613c4d9dbdb57d02d0`
  and `f13ef3f8650c4fe75f795ddf255b32a377ea4b7c31d4bdc007f0121114ba97a1`.
  This is `+6.013124pp` over the Stage-1 endpoint and `-7.418452pp` versus
  the K=384 30+60 course, but both comparisons are confounded: this run used
  90 full-model epochs, has no matched native K=192 uniform endpoint, and its
  recovery cannot reproduce the source job's global RNG bit-for-bit. It is
  terminal only for the historical over-budget diagnostic course, not a fair
  total-60 result or proof that learned selection beats uniform at 25%.
  Job `1194471` remained running after the terminal receipt solely for
  selector-quality export, with empty formal stderr, no hard failure and no
  best-checkpoint pointer.
- At `2026-07-27 18:46 +08:00`, post-terminal Stage-2 epoch-5/10
  selection-quality summaries completed over 487 windows and 211 videos, with
  SHA-256
  `fcbeb7b77f3dc574639261399848baaf6e0172809a3d924d443d390529c01864`
  and
  `c6677d757340c8fc721fa289743f638087e2291ad3b812eca5c9c92e6c7d910d`.
  Coarse macro AUPRC/AUROC changes only from `0.421082/0.585920` to
  `0.425132/0.590137`; pooled Brier/ECE changes from
  `0.215769/0.027737` to `0.216125/0.035281`, so slight discrimination
  improvement accompanies worse calibration. Learned versus matched uniform
  remains mixed: epoch-10 action enrichment is only `+0.014851`, boundary
  recall at radii 0/1 is `-0.006059/+0.013539`, R2Q3 bilateral endpoint
  recall is `+0.014461`, and R4Q5 bilateral endpoint recall is
  `-0.026231`. Mean endpoint distance improves only `0.007479`, while mean
  maximum hole worsens `0.030801`. This does not establish a clear learned
  geometry advantage and cannot replace a full-model K=192 matched-uniform
  terminal result. Job `1194471` continues post-terminal epoch-15 quality
  export with empty stderr and no hard failure.
- At `2026-07-27 19:10 +08:00`, Stage-2 epoch-15/20 quality summaries
  completed with SHA-256
  `a8b0917ad46cac2921b2534318cec1eb89a2d62a1f9ce4d5bd299a44d1fb824e`
  and
  `3de85f7579650c007d23af1d22b73e461ab03226478fdd4c315a10c45eb7f757`.
  By epoch 20, coarse macro AUPRC/AUROC improves to
  `0.437216/0.599782`, but ECE worsens to `0.050933`. Learned versus
  matched uniform has only a significant radius-1 boundary-recall gain
  (`+0.017566`, 95% CI `[0.001394, 0.034104]`); exact-boundary recall,
  R2Q3 paired support and endpoint distance have zero-crossing intervals.
  Wider R4Q5 bilateral support is significantly worse by `-0.026594`
  (`[-0.033483, -0.021020]`), and mean maximum hole worsens `0.121150`.
  The selector therefore exhibits a narrow local gain but no general paired
  boundary advantage, consistent with weak high-tIoU behavior. This remains
  explanatory, not a causal attribution or substitute for the missing K=192
  matched-uniform terminal. Job `1194471` continues epoch-25 quality export
  with empty stderr.
- At `2026-07-27 19:46 +08:00`, Stage-2 epoch-25/30/35 quality summaries
  completed with SHA-256
  `cfab3759813577fe1187d9f2a6c8340642c991f92577523eff8409d3ac5e8af6`,
  `719da69f91bcab1decfac2dfe47600fa7d5f014151f4d51a505e4f59e0589f0a`,
  and
  `284f91c9dde2c01b8782a9ea6d3581ea07aa61eed10fc34556bc7ce0b0934ef6`.
  By epoch 35, coarse macro AUPRC/AUROC reaches
  `0.441664/0.605025` and action enrichment is `+0.024621`. Radius-1
  boundary recall (`+0.031290`, CI `[0.013792, 0.053063]`) and mean
  endpoint distance (`+0.047062`, CI `[0.013641, 0.078358]`) improve, but
  exact-radius recall and R2Q3 bilateral support remain inconclusive. R4Q5
  bilateral support degrades by `-0.045869`
  (`[-0.053979, -0.037741]`) and maximum hole worsens `0.501027`.
  The learned policy is increasingly action-enriched and locally
  boundary-near, yet progressively worse at paired wide-boundary protection.
  This is a plausible mechanism for weak high-tIoU localization, not a causal
  conclusion without the registered objective/geometry ablations. Job
  `1194471` remains in post-terminal quality export with empty stderr and no
  hard failure.

## 2026-07-27 total-60 Pro major-review absorption

- The 886-line review is archived byte-identically at
  `docs/methods/reviews/2026-07-27-duca-total60-prebackbone-pro-review-raw.txt`
  with SHA-256
  `D493FD3497D412B3B873940447F1C743F3A1A50418EBCFC20B9FCE16945A4E11`.
  Project adjudication is
  `docs/methods/2026-07-27-duca-total60-prebackbone-pro-review-absorption.md`.
- Verdict is `major_revision_accepted_with_corrections`, not blanket
  acceptance. Fully accepted scientific requirements are clean official
  dense/native K384/K192 uniform and wrapper parity, one unique
  `e -> p -> F -> y -> S` exact-K contract, symmetric warped-time I/O,
  separate `G_rank` and `G_direct`, video-level statistics, a development
  seed excluded from final statistics, and separate task-adapted/train-free
  claims.
- Pure-plugin output semantics are now explicit: inverse-map raw detector
  proposals from warped time q to physical time t before running unchanged
  official NMS. Nonlinear time warps do not preserve segment IoU, so NMS in q
  followed by inverse mapping is not accepted as the strict-plugin path.
- Reviewer constants (`p_i` bounds, `4/K` CDF shift, DP gap/anchor bounds),
  the exact linear-rank RDD formula, `+1.0pp/40% dense-gap`, second-detector
  and cost thresholds remain `designed_reviewer_proposal`. They must be
  frozen after clean-baseline/reachability/power analysis and before formal
  results, not treated as established theory.
- `G_rank` failure kills the current continuous-gradient contribution
  teacher and its RDD arms. It does not logically prohibit one explicit
  redesign around hard counterfactual utility; repeated loss substitution is
  forbidden. `G_direct` failure kills A4 only.
- Final checkpoint policy is conditional: use 6,000-update terminal EMA when
  there is no independent training-side model-selection set; otherwise all
  arms may use the same preregistered held-out selection rule. Official test
  cannot select epochs. Any A4 short fork is development-only; a formal A4
  must still train within 6,000 total updates.
- The review's K192 evidence grade is stale: terminal official
  `57.967272%` now exists with sealed receipts. Scientific status is still
  `terminal_over_budget_diagnostic` because it used 90 full-model epochs,
  lacks clean K192 uniform, and has a documented non-bit-exact RNG recovery.
- PR #3 remains an open draft with 133 changed files and 27,627 additions
  against its current base. Cleaning that review surface is required for
  truthful public review but is nonblocking for model science.
- The fair total-60 bounded model remains
  `designed_major_revision / implementation_not_started`. Immediate work is
  parallel P1 clean baselines/parity, P2 mathematical decoder/coordinate
  contract, and video-level `G_rank/G_direct` tooling. No A1-A4 long training
  is authorized yet.

## 2026-07-27 K192 post-terminal quality epochs 40 and 45

- Stage-2 epoch-40/45 selection-quality summaries completed with SHA-256
  `a9d361a4f0c7d9f56095bfc19210ce979e3652db9262a23c2a3ebd26619dfa50`
  and
  `224f87458dceab37276142d08c3f92dae36674bb2f466f7e1d0810af86c17994`.
  Each covers 487 windows, 211 videos and 355,592 frame observations.
- Coarse macro AUPRC/AUROC reaches `0.443028/0.605677` and
  `0.443816/0.606281`; pooled Brier/ECE is
  `0.215309/0.048988` and `0.215244/0.049396`. Action enrichment versus
  matched uniform remains small at `+0.025338/+0.025265`.
- Radius-1 boundary recall remains significantly positive at
  `+0.039993` (`[0.022085, 0.060272]`) and
  `+0.034523` (`[0.016535, 0.051169]`), while endpoint distance improves
  `0.036928/0.054350` with positive intervals. Exact-radius and R2Q3
  bilateral changes remain inconclusive.
- R4Q5 bilateral endpoint recall remains significantly worse by
  `-0.046030` (`[-0.053938, -0.037155]`) and
  `-0.046575` (`[-0.055881, -0.038003]`); maximum hole worsens
  `0.542094/0.570842`. The late curve therefore confirms narrow local
  support at the cost of broad paired-boundary protection, not a general
  learned-selector advantage.
- At `20:21 +08:00`, Job `1194471` remained `RUNNING` only for
  post-terminal epoch-50 quality export, with null dependency, exact clean
  commit `ed0d4900bffe3546997ea1f00ae806d82cad55f2`, empty formal stderr,
  no hard failure and no best-checkpoint pointer.
- At `20:24 +08:00`, epoch-50 quality summary completed with SHA-256
  `cdb63e8babcd239967b68dd95818c2c4e3fc0d4d865d340c0bc25afcfcc1c2a4`.
  Coarse macro AUPRC/AUROC is `0.444491/0.606776`, pooled Brier/ECE is
  `0.215020/0.047492`, and action enrichment is `+0.024000`. Radius-1
  boundary recall remains positive (`+0.040996`,
  `[0.020502, 0.058475]`) and endpoint distance improves `0.049390`,
  but exact-radius and R2Q3 bilateral support remain inconclusive. R4Q5
  bilateral support is still worse by `-0.043515`
  (`[-0.052543, -0.036561]`) and maximum hole worsens `0.579055`.
  This continues the same local-gain/wide-paired-support-loss diagnosis.

## 2026-07-27 reviewer-defense theory closure discussion

- Status: `discussed / not_frozen / no_model_implementation_started`.
- The publishable problem is not generic inverse-CDF frame sampling. The
  candidate scientific core is localization-preserving temporal measure
  transport for offline TAD: an exact-K bounded transport, hard
  counterfactual-benefit alignment, and explicit protection of paired
  boundaries and high-tIoU localization.
- The plugin claim must include an honest coordinate-adapter boundary.
  Training-side `t -> q` GT mapping is an invertible I/O transform, but it is
  still a training-path change and must not be described as "only deleting
  frames." Detector architecture, detector losses and NMS algorithm can
  remain unchanged only when the adapter is external and auditable.
- Current code has selected-axis interpolation in
  `opentad/models/utils/post_processing/utils.py`, but the generic AdaTAD
  path in `opentad/models/detectors/single_stage.py` applies NMS before that
  inverse mapping. The required canonical contract is raw proposals in
  selected coordinates -> inverse map to physical time -> unchanged official
  NMS. That contract is designed but not yet implemented as the total-60
  plugin path.
- `G_rank` and `G_direct` remain separate evidence gates. Classification and
  regression contribution weights must be normalized and selected only on a
  training-side calibration split using multi-scale hard swaps and matched
  random controls; test mAP must not tune their ratio.
- Geometry bounds are stability constraints, not an mAP theorem. Density
  lower/upper bounds control local dilation/compression; cumulative-drift,
  anchor-distance and maximum-gap limits control coverage. Their constants
  still require derivation from localization tolerance and pre-result
  freezing.
- Recommended paper route is fixed-K bounded transport as the main result,
  with risk-controlled dynamic K and a truly frozen-detector train-free mode
  as extensions. A new Pro review is useful only as a focused theory-closure
  adjudication after these unresolved choices are written down; another
  broad job/version audit would be repetitive.

## 2026-07-27 K192 post-terminal quality completion and Slurm closure

- Job `1194471` reached final Slurm state `COMPLETED`, exit `0:0`, at
  `20:47:16 +08:00` after `08:28:13`. It is absent from `squeue`; exact
  commit `ed0d4900bffe3546997ea1f00ae806d82cad55f2`, sealed terminal
  `57.967272%` official Avg-mAP and all terminal hashes remain unchanged.
- Epoch-55/60 quality-summary SHA-256 values are
  `f74cb521dd3c16af3bb6fc42a476f9ead8e69deee9fcac31999149ce5877e19f`
  and
  `fb53b13243235be945e30fd9b2b9bede7cfb2f3558b80c763d83f84c574fc2e3`.
- Terminal epoch-60 coarse macro AUPRC/AUROC is
  `0.445240/0.607663`, pooled Brier/ECE is `0.214704/0.044528`, and
  learned-minus-matched-uniform action enrichment is `+0.024107`.
  Radius-1 boundary recall remains positive at `+0.036524`
  (`[0.018434, 0.054814]`) and endpoint distance improves `0.040021`
  (`[0.009562, 0.067923]`), while exact-radius and R2Q3 bilateral intervals
  cross zero. R4Q5 bilateral support remains significantly worse by
  `-0.041925` (`[-0.049793, -0.033066]`) and maximum hole worsens
  `0.587269`.
- All 12 scheduled quality checkpoints, one-based epochs
  `5,10,...,55,60`, now exist. Final scans show no Traceback, OOM, Decord
  failure, hard `FAIL` or best-checkpoint pointer. The complete curve closes
  the same diagnosis: stable narrow local support plus progressively worse
  broad paired-boundary coverage, not a general learned-selector advantage.
- This closes monitoring of the K192 over-budget recovery itself. It remains
  diagnostic-only because the full curriculum used 90 model-training epochs,
  lacks a clean native K192 matched-uniform terminal and resumed without
  bit-exact global RNG.

## 2026-07-27 dynamic-K and AdapTok Pro adjudication request

- User decision: reject the prior default that fixed K must be the paper main
  result. Dynamic K is now a required candidate core innovation, subject to
  an explicit Pro scientific adjudication rather than automatic acceptance.
- Status:
  `user_priority_recorded / pro_adjudication_prompt_ready /
  mathematical_model_not_frozen / no_training_authorized`.
- The adjudication must compare three paper structures: fixed-K main with
  dynamic extension; dynamic-K main with fixed-K as a special case; and a
  unified hierarchical model whose outer allocator chooses per-video K under
  an average budget and whose inner decoder performs localization-preserving
  exact-K transport.
- AdapTok, CVPR 2026, is a mandatory primary comparison. The review must read
  its block-wise variable-length training, causal quality scorer, marginal
  score curves, Fixed/BiThr/BiDelta/ILP allocation ablations and code. It must
  separate transferable adaptive-budget principles from non-transferable
  reconstruction, generation, latent-token and online-causality assumptions.
- The prompt is preserved at
  `docs/methods/prompts/2026-07-27-duca-dynamic-k-adaptok-pro-adjudication-prompt.md`.
  It pins the public DUCA repository to commit
  `63a726a4aaf48ecbf6780bb196de43a890c6b4df` and asks for an equation-level
  unified model, fair total-60 experiment matrix, AdapTok novelty boundary,
  reviewer attacks and kill criteria.
- The initial three-route adjudication prompt was judged too conservative.
  It has been superseded in place by a first-author research-takeover prompt.
  The new prompt does not constrain Pro to route A/B/C: it requires at least
  ten scientifically distinct model families, three rounds of novelty,
  mechanism and evidence-cost elimination, and a binding winner/fallback/
  high-risk selection.
- AdapTok review is now source-complete by contract. Pro must read every
  paper and supplement section, enumerate the complete repository tree at
  commit `a72076cf6474f930a181aa78971de70d65289b49`, and ground each mechanism
  claim in a concrete path, class, function and line. Mandatory execution
  paths include block-prefix masking, multi-budget score-label export,
  `TransformeScorer`, offline/online scorer training, `AdapTok.encode_eval`
  and `solve_ilp_min`.
- The expanded adjudication centers the marginal value of temporal evidence
  under localization risk. It asks Pro to derive dynamic-K dual allocation,
  batch-composition-invariant budget calibration, paired-boundary/max-hole
  risk, nested anytime frame sets, multi-scale hard counterfactual utility,
  a total-60 mixed-budget curriculum, and a genuinely frozen train-free
  mode. These remain `DESIGN_PROPOSAL / UNRESOLVED`; no implementation or
  training is authorized by the prompt revision itself. The 707-line prompt
  SHA-256 is
  `01c496a94bd9f349d5b0d4ca9ca073568805b890b63ea73cf98379c83e788548`.

## 2026-07-27 dynamic-K / AdapTok takeover response absorption

- The 4,589-line response is byte-identically archived at
  `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-research-takeover-raw.txt`
  (152,867 bytes, SHA-256
  `5ae7850662d726d91c4b3dc7f362ad223d33c35e3cbad9bb87771e939e07e031`).
  The structured independent audit is
  `docs/methods/2026-07-27-duca-dynamic-k-adaptok-research-takeover-absorption.md`.
- Project verdict:
  `substantial_accept_research_direction /
  major_correction_before_design_freeze /
  dynamic_k_required_candidate_not_empirical_fact /
  no_model_code_or_long_training_authorized`.
- Accepted scientific center: low-cost pre-backbone evidence should estimate
  train-only hard budget-conditional cls/reg/high-IoU and paired-endpoint
  value; a frozen per-video dual policy should allocate realized heavy-frame
  cost; each selected set must obey exact-K physical-time geometry; raw
  proposals must map `q -> t` before unchanged official NMS.
- The response is not a consistent implementation spec. Its compact part
  explicitly rejects `S(K1) subset S(K2)` and defines independent-per-K
  budget-policy value, while the expanded takeover report later requires a
  strict nested ladder and group-add utility. Strict nested, weak-overlap and
  independent decoding must first be compared by a train-only Oracle regret
  gate; one family is frozen before any formal run.
- Reviewer constants and thresholds remain proposals. The response contains
  conflicting Oracle `+0.75/+1.0pp`, 20/40% gap-recovery and 25/30% cost
  targets; density feasibility, discrete inverse, gap indexing, cost scaling,
  clustered budget certification and endpoint units also require correction.
- Dynamic experiments require controls absent from the proposed minimum
  matrix: clean uniform at each K or an identical per-video K sequence with
  uniform positions, identical mixed-K exposure, a K-histogram shuffle,
  random/shuffled/reversed utility nulls, video-ID-disjoint fit/calibration/
  certification splits, and explicit hard-label GPU cost.
- Current root still implements the old online/selected-axis DUCA
  (`C3CoarseProbeActionnessSource`, `DucaAcquisitionAdapter`,
  `PrefixMarginalUtilityBudgetController`,
  `budgeted_center_radius_decode`, `DucaOnlineFrameSelector`). It does not
  implement bounded-density exact-K transport, nested/independent Oracle,
  hard RIME utility/risk, pre-NMS physical inverse mapping, K-bucket execution
  or total-60 RIME configs.
- Recommended decision sequence:
  clean dense/native uniform/wrapper parity and physical-time-before-NMS;
  model-independent exact-K/coordinate property tests; dynamic headroom,
  decoder-family regret, `G_rank` and pair-risk on train-only frozen-detector
  hard evidence; one 6,000-update development seed; only then three fresh
  seeds, second detector and full-stack cost.
- New nodes:
  `research-wiki/ideas/duca-rime.md` is `discussed/pending`, and
  `research-wiki/experiments/duca-dynamic-k-rime-oracle.md` is
  `discussed_proposal/no_training_authorized`. The old negative MUST record
  remains negative and is not overwritten by this new candidate.

## 2026-07-27 dual dynamic-K / AdapTok takeover-response adjudication

- Two additional first-author takeover replies were fully read and archived
  byte-identically:
  `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-takeover-response-a-2032fca-raw.txt`
  (`96,650` bytes, `2,694` physical lines, SHA-256
  `2032fcaeddbd4f758ac1be024dd3f867e8dbc6baacd9955de40241ce35595127`)
  and
  `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-takeover-response-b-e2231c0-raw.txt`
  (`122,113` bytes, `2,667` physical lines, SHA-256
  `e2231c0928c7dd345a4c7a0cf8b55afe4de95270b710b95602ddd6b5c3fb4bf5`).
  The project comparison is
  `docs/methods/2026-07-27-duca-dynamic-k-adaptok-dual-response-comparison.md`.
- Their top-level direction is highly aligned: stop density/inverse-CDF as the
  paper center; treat dynamic K as a required decisive candidate; use
  train-only hard cls/reg/high-IoU and paired-endpoint value, exact-K physical
  acquisition, a train-calibrated per-video cost price, pre-NMS physical-time
  restoration, equal-update total-60 controls and full-stack cost.
- They are not one executable specification. A proposes DUCA-METER/METER-TAD,
  a dense K grid, mostly one-main-forward training with low-frequency
  alternatives, hard risk feasibility and explicit Kmax fallback. B proposes
  MERTAD, a different K grid and five-stage schedule, matched half-batch paired
  forwards, decomposed rank/delta/risk/calibration losses and a different
  hard/soft risk treatment. Their proposed numerical gates also differ.
- Project verdict:
  `accept_scientific_core_with_major_corrections /
  retain_duca_rime_as_unfrozen_internal_name /
  dynamic_k_required_candidate_not_empirical_fact /
  strict_nesting_requires_train_only_oracle_regret /
  thresholds_and_training_contract_not_frozen /
  no_full_rime_implementation_or_long_training_authorized`.
- The prior 2026-07-22 fixed-K canonical recommendation does not override the
  later explicit user decision at lines `3144--3148`: dynamic K must be
  decisively adjudicated. This does not make dynamic K positive evidence.
- Current tracked root remains commit
  `63a726a4aaf48ecbf6780bb196de43a890c6b4df`. Its generic
  `opentad/models/detectors/single_stage.py` still invokes NMS before
  `convert_to_seconds`; experimental files in dirty/untracked workspaces do
  not prove the canonical `raw q -> physical t -> NMS` contract.
- Next authorized design sequence remains Phase 0 measurement/split/power,
  Phase 1 geometry and coordinate property tests, Phase 2 O1 dynamic headroom,
  O2 decoder regret, O3 video-cluster `G_rank` and O4 pair-risk calibration.
  A single development-seed total-60 contract is frozen only after those gates
  and explicit design approval.

## 2026-07-28 SparseHead Approach A full-chain deployment

- Job `1201048` is terminal `FAILED 1:0` after `1m54s`. Its gate pre-tests
  passed (`39 passed`), but the first model construction hit failure signature
  `actionformer_native_temporal_geometry_constructor_contract_v1`. No gate
  JSON, replay or suite was produced. Preserve its v4 runtime/run root.
- The v5 SparseHead decode-cross recovery deployment was Job
  `1201317`, run root
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260728_v5`.
  Its exact clean runtime commit/tree is
  `0338f4777bd02fb327573ef716f54fec76d4af0e` /
  `cb98c64c17d2983c22181d4908c4f31024a82a2f`.
- The protocol-preserving repair restored the historical native-J192
  ActionFormer alignment contract; it did not switch to `PhysTimeTAD`, change
  model/checkpoints/config/seed/data/evaluator, or alter decision gates.
  Recovery Linux tests are `74 passed`.
- v5 full-content preflight passed. Manifest SHA-256 is
  `77b9918aa3173b73fc71d821defa8c14b3165de1b35f0ae4c0382eeb5d21b43d`;
  the frozen dataset manifest remains
  `1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2`.
- One allocation executes the four-condition real-CUDA gate, then four full
  direct-inference/cross-decode replays, then the explicit evidence suite.
  Every transition is fail closed. This is scheduling compression only.
- Its four-condition gate pre-tests were
  `41 passed`. The four-condition real-CUDA gate has `gate_pass=true`,
  `all_native_direct_exact_equivalence=true`, and all four
  `raw_tensors_immutable=true`. `selected_online` direct inference then
  displayed Avg-mAP `41.26` and mAP@0.3--0.7
  `64.50/56.39/42.66/27.82/14.90`, but the producer failed to create
  `pre_cross_window_detections.json.gz`. Job `1201317` is terminal
  `FAILED 1:0`; the other three replays and suite never started.
- User-corrected recovery policy: confirmed non-model engineering failures have
  no fixed automatic-repair count. Preserve each failed root, reproduce and
  diagnose the cause, add a focused regression, rerun checks/full-content
  preflight, and deploy a new clean commit/root until final performance is
  obtained. Repeated signatures require deeper diagnosis and a materially
  evidenced fix, never an unchanged resubmission.
- If the complete model result is negative, preserve it as scientific evidence
  and immediately perform a Pro-level attribution rather than relabeling it as
  infrastructure failure. Compare all raw metrics against matched controls,
  full60 and P0; decompose high-tIoU, class/duration/boundary errors, proposal
  recall/calibration/NMS, assignment/support/native geometry, decoder regret,
  online/EMA and cost; test contradictions across all four conditions; produce
  at least two competing explanations with counterevidence, falsifiable
  predictions and minimal decisive experiments. Analysis precedes any core
  algorithm change or retraining.
- Deployment v1--v3 produced no Slurm jobs and are diagnostics only:
  PowerShell stderr handling, missing repository-root `PYTHONPATH`, and wrong
  generic raw-video paths respectively. The frozen P0 dataset paths are
  `/data/run01/sczc063/yuzibo/thumos14/train` and
  `/data/run01/sczc063/yuzibo/thumos14/test`.

## 2026-07-29 SparseHead Approach A v5 artifact failure -> v6 recovery

- v5 failure signature is
  `direct_postprocessing_artifact_producer_contract_missing_v1`. It is a
  producer/consumer implementation-contract failure, not a model negative.
  Preserve the complete v5 root and logs.
- The v5 `selected_online` direct metrics above are `diagnostic_only`:
  `primary_result_allowed=False`, `metric_claim_allowed=False`. They cannot be
  treated as a replay completion, matched four-condition result, suite verdict,
  or paper evidence.
- The protocol-preserving repair restores the declared pre-cross gzip artifact,
  post-processing audit and evaluation metrics producer, with an end-to-end
  regression. It does not alter model/config/checkpoint/seed/data/evaluator.
- The only active deployment is v6 Job `1201469` (`ptdc-a1-r2`), run root
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v6`.
  Clean runtime commit/tree is
  `ac326ffdc97652433b55ccc596e734b112f51806` /
  `0c58027756997995bda0de6fdd8ec0deb49966d3`; Linux focused suite is
  `75 passed`; preflight SHA-256 is
  `97fe5af28b2647396c052c9bdf956997d98e264af74432b57e0fc983b071fb91`.
- Latest verified state is `RUNNING` on `g0030`; gate focused tests are
  `42 passed`. The v6 four-condition real-CUDA gate also passed with
  `gate_pass=true`, `all_native_direct_exact_equivalence=true` and four
  immutable raw-tensor sets; gate artifact SHA-256 is
  `775e1f2dae70b7863324fd9d235712195dca4d0846968b3bd5e55b754e7b3ea4`.
  `selected_online` full direct inference completed with exact Avg-mAP
  `0.4125660433077075` and mAP@0.3--0.7
  `0.6450446628552113/0.5638932489689005/0.4266348135535575/
  0.27820781407261164/0.14904967708825695`.
- The repaired producer contract is observed in the real run: pre-cross gzip
  schema `opentad_pre_cross_window_detections_v1` covers `211` videos; audit
  schema is `opentad_post_processing_audit_v1`; metrics/result artifacts carry
  epoch 59 and exact v6 Git identity. Audit-recorded pre-cross SHA-256 is
  `31e70dc728aff9061f2c56266e3e6d32ef892b227a5c16b15da85e81f731b50e`.
- `selected_online` dual-axis replay is running and has emitted uniform-rank
  intermediate artifacts, but no `DECODE_CROSS_COMPLETE.json`, suite verdict
  or final cross-decode result exists yet. Do not submit a duplicate.
- Status remains `experiment_running` until all four full replays and the
  explicit suite close with attested terminal metrics. No
  `empirically_supported` or `paper_ready` claim is authorized.

## 2026-07-29 SparseHead Approach A v6 validator failure -> v8 recovery

- Job `1201469` is terminal `FAILED 1:0` after `32m32s`. Its
  selected-online direct inference and dual-axis replay producer completed,
  but `validate_run()` referenced unbound local `numeric_precision` while
  assembling the formal completion. Failure signature:
  `decode_cross_validator_numeric_precision_scope_v1`.
- The producer completion passed its contracts and has SHA-256
  `0283620a7c5308275c45d03ab1cf639cb8b889d385122d9907fa3e373ef74062`.
  Uniform/native direct Avg-mAP is `0.4125660433077075`; physical-time
  cross-decode Avg-mAP is `0.5015355102106833` (`+8.89694669029758 pp`).
  These are `diagnostic_only`, `primary_result_allowed=False`,
  `metric_claim_allowed=False`: no formal `DECODE_CROSS_COMPLETE.json`, other
  three conditions or suite exists in v6.
- The protocol-preserving fix binds, validates and propagates a copied
  producer numeric-precision contract and adds a focused non-mutation
  regression. Exact runtime commit/tree is
  `1631d0b60f6552a6f5eb0378d74e766850f34ffd` /
  `f485c8708e22bbbf9a73063d5293a20bc4aa658f`.
- v7 passed the v6 test surface plus the new regression (`76 passed`) and
  full-content preflight, but a deployment metadata hash omitted its final
  hexadecimal digit. Signature `deployment_expected_sha256_truncation_v1`
  was caught before `sbatch --test-only`; zero Slurm jobs and zero model
  forwards were created. Preserve the v7 root as a pre-submission diagnostic.
- The only formal successor is v8 Job `1201495` (`ptdc-a1-r4`), run root
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v8`.
  Clean branch is `codex/sparsehead-evidence-recovery-20260729-v8` at the exact
  commit/tree above. Linux exact recovery tests are `76 passed`, log SHA-256
  `16f53fc3cf8a9c5010bce3fd1ed98c4e347add284ba4b2443c00b49b5e107390`;
  preflight SHA-256 is
  `e9f36c221156e5411dad5e3bfe43508b4aa59310539fdbe24da985fc99a27d53`.
  `1201494` is test-only, not a job.
- Latest verified state is `RUNNING` on `g0024`/RTX4090. The unchanged
  single-allocation chain remains gate -> selected-online -> selected-EMA ->
  physical-online -> physical-EMA -> explicit suite. Status is
  `experiment_running`; do not promote it until all four formal completions,
  suite verdict and terminal metrics exist.
- v8 gate focused tests passed (`43 passed in 29.08s`). Four-condition CUDA
  gate SHA-256 is
  `5e323e5ccdedd7dd39d70148aed7108beca94bb5952125a124ad20accfd634f6`;
  `gate_pass=true`, all native/direct comparisons are exact and all four raw
  tensor sets are immutable. Selected-online direct inference is now running;
  there is still no v8 formal completion or suite verdict.
- v8 selected-online direct inference then completed. Exact Avg-mAP is
  `0.4125660433077075`; mAP@0.3--0.7 is
  `0.6450446628552113/0.5638932489689005/0.4266348135535575/
  0.27820781407261164/0.14904967708825695`. Its 211-video pre-cross artifact
  is bound to the v8 commit/tree and has SHA-256
  `b4adcf545655424d2b2dfdfce0d107109c5010850143fadf925706fb3de60322`.
  Dual-axis replay has completed uniform-rank artifacts and is producing the
  physical-time mode. No formal completion/suite exists; no hard failure was
  observed.

## 2026-07-29 SparseHead Approach A v8 suite failure -> v10 recovery

- Job `1201495` is terminal `FAILED 1:0` after `02:00:24`. All four replay
  completions exist and independently passed frozen-tensor, native/direct and
  reviewed-P0 parity, but no evidence-suite completion exists.
- Failure signature:
  `decode_cross_completion_fatal_log_findings_container_type_v1`. Each replay
  completion serialized zero fatal findings as JSON object `{}` while the suite
  contract requires JSON array `[]`. Suite validator log SHA-256 is
  `558c78694ae18b9827e4b3cc27f731f3e684faa7eb9a08a1670584c154102919`;
  failure receipt SHA-256 is
  `22739defebe8261f61e1fff9910d6d74592d6de4621f7147b07138154ae94d13`.
  This is an evidence-container implementation failure, not a model negative.
- v8 diagnostic Avg-mAP pairs, written as uniform-decode -> physical-decode,
  are selected-online `0.4125660433 -> 0.5015355102`, selected-EMA
  `0.4128302079 -> 0.5009785403`, physical-online
  `0.4010767719 -> 0.5755558109`, and physical-EMA
  `0.4029649803 -> 0.5760868491`. Exact threshold rows are preserved in
  `experiments/phystime-frozen-decode-cross-replay.md`. Until a complete suite
  exists these values are `diagnostic_only`, with no metric or primary-result
  claim allowed.
- v9 failed before commit/preflight/Slurm because repository-local Git author
  identity was absent. Signature `runtime_git_author_identity_missing_v1`;
  zero jobs; receipt SHA-256
  `ca7f75bc72e85fd466331012775cff72ca14fd685b1db4cc52c8212450c994d2`.
- The sole formal successor is v10 runtime branch
  `codex/sparsehead-evidence-recovery-20260729-v10`, exact commit/tree
  `c878fbe3a5e960671f03d93fff8367ed3414f5c5` /
  `8d3e73bb26544d1bcf7bfb61154d0b003f2658e0`. Exact Linux recovery suite is
  `77 passed` (SHA-256
  `7f1787308250a6c9bd62e452f6e16357f5d6bf44cdbcfc6fedd61b7cc63c6936`);
  preflight SHA-256 is
  `f46f6299f7fccc899140ad8fdf001052772ef550dd34cdb68c17d5ba5fc59a8f`.
- `1203046` is test-only. Formal Job `1203047` (`ptdc-a1-r5`) is `RUNNING`
  on `g0050`/RTX4090; gate focused tests passed (`44 passed in 29.68s`) and
  the four-condition real-CUDA gate passed with SHA-256
  `e5516af02289d15dd1465f5387471bb1a3c357873980d22645c08acbf6aa141c`,
  exact native/direct equivalence and four immutable raw-tensor sets.
- The first fresh v10 component, selected-online, is now complete. Completion /
  producer-completion SHA-256 are
  `a4e727cf094127be7b91a4a13b140463ad9dc3e0c8c1bcfa3acb9887b5ff6dda` /
  `8a2d38db8a2130a8b617940361a8637dfdc0bff3b6947b0f35d75167a809bfa6`.
  It has `validation_pass=true`, `fatal_log_findings=[]`, frozen-raw equality,
  native/direct exact equivalence and reviewed-P0 parity. Uniform-rank decode
  Avg-mAP/mAP@0.3–0.7 is `0.4125660433 /
  0.6450446629/0.5638932490/0.4266348136/0.2782078141/0.1490496771`;
  physical-time decode is `0.5015355102 /
  0.7227308253/0.6467256017/0.5253267642/0.3903281011/0.2225662588`.
  Physical decode improves Avg by `+8.8969466903` pp and mAP@0.6 by
  `+11.2120287043` pp on the same frozen raw tensors.
- The second fresh component, selected-EMA, also completed. Completion /
  producer-completion SHA-256 are
  `0c6f87617b1cbd6a5bc4a6be6e9a5a2174f8a5a568c2f24db7253c15a315b8dc` /
  `ddddd42174eb987cdeb723ae4422df8105e773bd7af74d31e67760dba20d74ff`;
  validation, clean fatal-finding array, frozen-raw/native-direct and
  reviewed-P0 parity all pass. Uniform-rank decode Avg-mAP/mAP@0.3–0.7 is
  `0.4128302079 /
  0.6485551725/0.5634561176/0.4261516171/0.2774504499/0.1485376826`;
  physical-time decode is `0.5009785403 /
  0.7226575722/0.6462475078/0.5282674811/0.3876402291/0.2200799115`.
  Physical decode improves Avg by `+8.8148332403` pp and mAP@0.6 by
  `+11.0189779124` pp on the same frozen raw tensors.
- Physical-online is now the third valid `tested` component. Completion /
  producer SHA-256 are
  `02384da2c71c93bdcd6ce003cd59451510c9d095e222653202f09f38b73b153f` /
  `b9ba401a92e0d828aeabe48cb8972df74a64720a12f160d939daa355856aaf58`;
  all validation, clean fatal-array, frozen-raw/native-direct and reviewed-P0
  checks pass. Uniform decode Avg-mAP/mAP@0.3–0.7 is `0.4010767719 /
  0.6148821307/0.5285472772/0.4134826475/0.2861537181/0.1623180857`;
  physical decode is `0.5755558109 /
  0.7704022473/0.7055742485/0.6207490393/0.4859365795/0.2951169400`.
  Physical-minus-uniform is `+17.4479039086` pp Avg and
  `+19.9782861383` pp at mAP@0.6.
- Three components are `tested`, not a final route result. The unchanged
  allocation has completed physical-EMA native direct inference with
  Avg-mAP/mAP@0.3–0.7 `0.5760868491 /
  0.7721224902/0.7045574193/0.6257613932/0.4900660583/0.2879268846`;
  direct-metrics SHA-256 is
  `43c33d551c19f4f3ab90108af30b13c103aa3a875fd87f00a4f50c7e5a83ecac`.
  Its final dual-axis replay is running and no completion exists, so this row
  is `diagnostic_only`, not the fourth component.
- Overall status remains `experiment_running`; no Pro model-result attribution
  starts before the final completion and explicit suite verdict.

## 2026-07-29 SparseHead Approach A v10 terminal -> v16 recovery

- v10 physical-EMA is no longer direct-only. Completion / producer SHA-256 are
  `a5c0c5248bf196d17f1cbf4f11a61d01459cb2ff3cfbf37541046fdb508b7ad1` /
  `8433bd22b620cd60300d94289cf991b69c1f64bcd5eacea557fbc463d7981086`;
  all replay validation, empty-fatal-array, frozen-raw/native-direct and P0
  parity checks pass. Uniform / physical Avg-mAP is
  `0.40296498031949024 / 0.5760868491267752`; physical decode improves Avg by
  `+17.312186880728497` pp and mAP@0.6 by `+20.198173771581317` pp.
- Job `1203047` nevertheless ended `FAILED 1:0` at the explicit suite. The
  checkpoint path and file SHA match between preflight and gate, but the
  consumer compared their differently enriched dictionaries for raw equality.
  Signature:
  `decode_cross_suite_checkpoint_binding_schema_shape_mismatch_v1`; suite log /
  failure receipt SHA-256:
  `68b7b3d34e587392bdac2df1eb2a36d971009d4c07165ef2a18157449ccb931f` /
  `42c394f11153a862819876b3915c34ca2ef0a68b6b62ed78a121d65db4269cec`.
  Four v10 components are `tested`; v10 is not a complete suite or model verdict.
- The protocol-preserving repair validates each record, compares canonical
  resolved path + SHA, and retains online/EMA state-dict checks. Its focused
  regression accepts metadata-schema enrichment for the same artifact and
  rejects a different path even when bytes match. Model/config/checkpoints/
  seed/data/evaluator/thresholds are unchanged.
- Immutable zero-job roots and receipt SHA-256 are: v11
  `runtime_profile_source_under_nounset_and_mode_preservation_v1` /
  `2a95ca48464564d4979754525129414124c769a8a97852a9fad404087bc08545`;
  v12 `recovery_exact_suite_invocation_scope_drift_v1` /
  `387d61f33eb3dc055c182a8df23c721378ac4191ad170646311df021fc67e259`;
  v13 `preflight_repo_import_path_unbound_v1` /
  `a6d0ccf593e5cb01b9f6a90dee1a47d0042f8ab8a4201be68033b6868fb19858`;
  v14 `deployment_finalizer_base_relative_template_token_mismatch_v1` /
  `8a85f361fdfa90a6a753c5c3446a43617359cd18ee2a5ea541eac4f6ac00d387`;
  v15 `ssh_transport_interruption_during_exact_recovery_launch_v1` /
  `f7b7402cc1c565a69a01d057c0e50d7ea63632c3a6a8613be30adc866401630e`.
- The sole formal successor is v16 branch
  `codex/sparsehead-evidence-recovery-20260729-v16`, commit/tree
  `54e7f9abeaabf710a505f0a0f595a4eb3bb47f98` /
  `f8490f9c25c2e0e6958c406e19c83cc3d5a40535`. Exact Linux recovery is
  `78 passed` (SHA-256
  `d81ca79bd9af216c106fb9718e7b171dd47c9aff3ddecb9787d8e0203c88d0fc`);
  full-content preflight passed with SHA-256
  `ccc7a83e27b8d18ad0892b644e7338667b72d8eba3e3feedbc387dc4ac1d9a0d`.
  Deployment identity / submission receipt SHA-256 are
  `6f22152938b2ad3949a19672e622e97d861a7604f8ff9b5408d59e21bcfcf6d4` /
  `65c325fbd53b3c8386ce459e557f7d8e09f768eb38d77057d8e442b680393ad7`.
  `1203916` is test-only. Formal Job `1203917` (`ptdc-a1-r11`) is `RUNNING`
  on `g0045`/RTX4090. Gate focused tests passed (`45 passed`); the four-condition
  real-CUDA gate artifact SHA-256 is
  `0d2153effee84a0e1aa6410125bb291eb4ef4d41e4b40604f49d9e5868e0ada9`.
  It reports `gate_pass=true`, all native/direct exact equivalence, and all four
  raw-tensor sets immutable. Selected-online direct and dual-axis replay
  producer have now completed. Uniform/native Avg-mAP is
  `0.4125660433077075`; physical-time Avg-mAP is
  `0.5015355102106833` (`+8.89694669029758 pp`). Direct/uniform and physical
  metrics SHA-256 are
  `8860bdcaf3b998e6cddb1187c564d0bb0693496552439b104efad7145a6bd34c` /
  `7a032eaf8e4fc776ae0d670d572e02f74c23b82ef55bc29185e796e5be2f0f8b`;
  producer completion SHA-256 is
  `97410d9855a3f6db859e36213bf6b201e10c96941a164b5588af02cdfba4ee20`
  and formal `selected_online/DECODE_CROSS_COMPLETE.json` SHA-256 is
  `6937fc6b7b050fd7009ee967ceef446aebaa8b3daa695c7959106ff87048c038`.
  The formal receipt has `status=tested`, `validation_pass=true`,
  `fatal_log_findings=[]`, frozen-raw/native-direct/reviewed-P0 parity all
  passed and `new_training=false`.
- `selected_ema` is now the second formal `tested` component. Its
  uniform/native and physical-time Avg-mAP are
  `0.41283020792762315 / 0.5009785403306161`
  (`+8.814833240299292 pp`); producer/formal completion SHA-256 are
  `43c737fe3c5a9a534c565bf63e419fa152ee35b3be796ddf3f601c954fa52877` /
  `4a1b405b7849f396e1b649da8895070e6176023c4a959c6d7fd9148f2bd8afe0`.
  `validation_pass=true`, `fatal_log_findings=[]`, and frozen-raw/
  native-direct/reviewed-P0 parity all pass. Job `1203917` has entered
  `physical_online`.
- Current route state remains `experiment_running`: two formal components and
  the explicit suite are still pending. The close online/EMA values are
  descriptive stability evidence only.
- `physical_online` direct and replay producer have now completed. Physical/
  native Avg-mAP is `0.5755558109390063`; uniform-rank cross-decode is
  `0.40107677185286417` (`+17.447903908614215 pp` for physical-minus-uniform).
  Direct/physical, uniform and producer SHA-256 are
  `b68f2ad1393b59c40d58f7cfa1e450a52f84d8acbc80ad785a2d3a31352d6009` /
  `0c258e563fe7b9886e6d56c9c3370b6536e187b521526318622b07ffcf1e4a4b` /
  `d61d8fbf8b977b59b65eb87d55227904b2a5a2e6994e584226bda19a265b26eb`.
  Formal completion SHA-256 is
  `fd18348e6ae6ecf4bdc4390ca4620a109616582f7f77138ed137085e0df6c260`;
  all validator contracts pass, making this the third `tested` component.
  Job `1203917` then entered `physical_ema`.
- `physical_ema` is now the fourth formal `tested` component. Uniform-rank and
  physical/native Avg-mAP are
  `0.40296498031949024 / 0.5760868491267752`
  (`+17.312186880728497 pp`); their mAP@0.3--0.7 are
  `0.622154649489393 / 0.5316588686305871 / 0.4113769771975965 /
  0.2880843206041682 / 0.16155008567570622` and
  `0.7721224901972557 / 0.7045574192938243 / 0.6257613932435541 /
  0.4900660583199814 / 0.28792688457926047`. Uniform/physical metrics,
  producer and formal completion SHA-256 are
  `5058f789de9fd74544427fd8201d7b32cc83f18524409ee9e8f3b96fe32292dc` /
  `43c33d551c19f4f3ab90108af30b13c103aa3a875fd87f00a4f50c7e5a83ecac` /
  `aa6356a509898b94a38f2b9e0548c5f647cc6498655697b37fd39ea8982fc733` /
  `cd6da2f827524e0b9eb2b46c6cbbcc5b6e89243aa9cd8d7e45efafcb4cb6b565`.
  All validator contracts pass. All four components are now `tested`, but the
  explicit suite artifact and terminal Job `1203917` state remain mandatory
  before final route or model attribution.

## 2026-07-29 SparseHead/PhysTime Approach A terminal answer

- Job `1203917` is terminal `COMPLETED 0:0` after `02:34:30`; runtime
  commit/tree and cleanliness match the submission identity. Hard-failure scan
  is empty.
- Explicit suite completion / validation-marker SHA-256 are
  `ed2770c35cf9a3acd5fa80465eda1c34b3541ba3dea404c75388aaeffefbdc31` /
  `f2da143127b3a01aef7bda451e2351c494f72552f3810f604f895f4c0a7767d3`.
  Both report `validation_pass=true`; completion reports `status=tested`,
  `new_training=false`, `fatal_findings=[]` and binds preflight, CUDA gate,
  P0 evidence, four formal completions and checkpoint-state identity.
- Four physical-minus-uniform Avg-mAP deltas are
  `+8.8969466903/+8.8148332403/+17.4479039086/+17.3121868807 pp`
  for selected-online/selected-EMA/physical-online/physical-EMA. All signs
  agree; online/EMA differences are tiny. At high tIoU and on short actions,
  physical-time decode also improves proposal recall and localization.
- P0 is exactly reproduced by selected-EMA uniform (`41.283021`) and
  physical-EMA physical (`57.608685`). This closes the decode-axis,
  rounding and evidence-chain confounds. It does not establish training
  assignment causality.
- Interpretation: this is a strong `tested` positive result for
  physical-time-before-NMS decoding and a negative result for the claim that
  selected-rank decode is harmless. It is not a positive SparseHead/SDPQ
  verdict: matched 20-epoch selected/physical/SDPQ remains
  `30.42/44.88/30.88`, and a fixed-physical-decode cross-checkpoint gap of
  about `7.40--7.51 pp` remains descriptive and unexplained.
- Ranked causes: (1) temporal coordinate/decode geometry mismatch,
  high confidence; (2) training assignment/support and representation coupling,
  medium confidence; (3) ranking/NMS amplification, medium-low confidence;
  (4) single-seed chance, low confidence as the sole cause but still an
  external-validity limit.
- Missing evidence is explicit: no independent evaluator, class-wise AP,
  calibration curves, independent NMS counts, failure IDs, assignment/support
  observability, multi-seed estimate or end-to-end cost.
- Next decision sequence, with no silent retraining: independent sealed-artifact
  mapper/NMS/GT evaluation -> 64-window assignment/support audit -> existing
  artifact class/calibration/failure decomposition -> native K parity -> SDPQ
  micro-overfit gate -> multi-seed/bootstrap -> full cost ledger.
- Claim boundary: one frozen single-seed THUMOS replay establishes a causal
  within-checkpoint decode-axis effect because raw tensors are fixed. It does
  not establish cross-checkpoint training causality, SparseHead superiority,
  compute savings, robustness, cross-backend generalization or a paper-ready
  claim. Current experiment/route state is `tested`.

## 2026-07-29 SparseHead diagnostic closure and official-comparability gate

- The approved next-stage implementation is frozen in an isolated clean branch
  `codex/sparsehead-diagnostic-closure-20260729` at commit/tree
  `57917e7bf2b991478b4f6fc4ce1db5ca5878b68d` /
  `aaf7c82bd837078bb7276baf6c0a504da0684194`, based exactly on v16
  `54e7f9abeaabf710a505f0a0f595a4eb3bb47f98`. Local compilation and the four
  focused suites pass (`35 passed`). This makes the tools `tested`; no new
  remote diagnostic, training or benchmark result exists yet.
- The no-training closure now contains an independent NumPy/float64
  mapper/stable-ranking/Gaussian-Soft-NMS/THUMOS AP implementation and a sealed
  64-window SDPQ assignment/support audit. The first tool is forbidden from
  importing production decode/NMS/evaluator helpers; the second runs only
  backbone/projection/query construction/target assignment under
  `eval()/no_grad()` and checks production targets against an independent NumPy
  reconstruction. Their future outputs remain `diagnostic_only`.
- Paper comparability is now a hard gate, not a naming convention. The pinned
  official ActionFormer source is
  `happyharrycn/actionformer_release@61ea7eb9308a568b0cf45e3804830836e30061de`,
  tree `7b06c5261ba244788c942a0d73e304581bc35154`, official config SHA-256
  `73f8aeaf7deef93aba57259badd4c454990ec1e0ce6eaa7c3434db44baaeeaf0`,
  README SHA-256
  `bdee4eb088a74e190935097742c7dbfaf254eb912f79729dccd73b9b36b33db8`
  and THUMOS archive MD5 `375f76ffbf7447af1035e694971ec9b2`.
- The official-record builder hashes every I3D feature, validates raw
  `eval_results.pkl`, parses the official evaluation log, invokes the pinned
  official `ANETdetection` independently, and requires logged/recomputed
  metrics to agree before issuing `paper_main_table`. There is no skip-hash
  mode. Matched rows must be anchored to a verified official reproduction and
  declare exactly one predefined intervention:
  `selection_budget`, `head_projection` or `coordinate_geometry`.
- Therefore the v16 VideoMAE/K384 results and historical `63.61` remain
  non-main-table evidence. The former is a frozen mechanistic diagnostic; the
  latter lacks an exact current seed/checkpoint/evaluator/raw-prediction
  receipt and is `external_reference_only`. Neither may define the paper delta
  against official ActionFormer `66.83`.
- Next execution order is fixed: remote exact-commit independent v16
  recomputation; sealed SDPQ support audit only if an exact SDPQ checkpoint is
  available; exact official ActionFormer I3D resource/preflight check and
  official reproduction; then a same-protocol one-variable I3D matched matrix.
  No structural retraining is authorized before these evidence gates and the
  bounded Pro decision.

## 2026-07-29 active official-comparable execution

- Current clean diagnostic commit/tree:
  `6d74ad7b7c7736bbff48976a626b951512a54e96` /
  `80cd2431ebf9809f03ab1216b84b45380d51f33b`; local focused result
  `46 passed, 1 skipped`, Linux result `58 passed, 1 skipped`.
- Independent replay failure roots are preserved and classified as
  `independent_recompute_padded_axis_nonfinite_scope_v1` and
  `independent_recompute_annotation_subset_contract_v1`. The first came from
  legal NaN padding outside valid prefixes; the second from using logical
  `test` against an annotation whose exact evaluation subset is `validation`
  (211 videos, 3,325 GT, 20 classes). The fresh v3 recomputation is running;
  no independent metric claim exists yet.
- The exact official ActionFormer THUMOS archive is now locally verified with
  MD5 `375f76ffbf7447af1035e694971ec9b2`; released checkpoint/log package
  SHA-256 is
  `e028f7e487713d0c68f0515ba9bdafda0ed05fc1271b9999ea995652b034c929`.
  Extraction/preflight is running in the pinned upstream runtime. Existing
  OpenTAD pickle features were deliberately not converted or called official.
- SDPQ support observability is bound to clean config commit `4a57577`, exact
  epoch-19 online checkpoint SHA-256 `40fccfd...b2c3fc7`, seed 42 and exactly
  64 sealed training windows. Job `1204961` failed in `/bin/sh` before Python;
  Job `1204981` is the only successor and is pending with explicit Bash.
  This artifact is always `diagnostic_only_no_training_support_observability`.
- Hard decision boundary: do not launch a matched method row until the official
  released anchor has raw `eval_results.pkl`, official log, independent
  recomputation and `paper_main_table` comparability verdict. Only then may one
  predeclared intervention be tested under the same I3D/data/schedule/evaluator
  protocol.

## 2026-07-29 sealed official-protocol execution update

- The current clean diagnostic closure is commit/tree
  `2b074845497f6ada3314cb895f0d4ab2f4ce3eca` /
  `7779862c5422dc8e527b304bf881a760b0c90625`. Its exact Linux runtime passed
  `95 passed, 1 skipped`; preflight log SHA-256 is
  `265046cd7fc3b1e847e87880e061a5a76092c4b194d1d4e727ca706f5b8884b6`.
  The tools are `tested`; this commit contains no model training or new method
  metric.
- Official ActionFormer Job `1205131` completed inference/evaluation but failed
  the older record contract. Its diagnostic metrics are
  `82.13/77.81/70.95/59.40/43.87` at tIoU `0.3:0.7` (Avg `66.83`), raw
  prediction SHA-256
  `1333df9202eec7ae217542b6bd2b15b597c1a004ebb3634de54a7a37adb6d7fe`.
  The failure signature is `official_annotation_split_schema_contract_v1`,
  not a model failure: the nominal THUMOS protocol has 200 validation and 213
  test videos, while the pinned official annotation database has 200
  `Validation` and 212 `Test` entries. The official raw file has 42,400
  predictions over exactly those 212 evaluated videos (ID-set SHA-256
  `7543da7a293c941bf19c388ecb92b7bd2520904cbfd704e60275acb53691490d`);
  the 413-file feature inventory additionally contains unannotated
  `video_test_0001292`.
- The record builder now pins the canonical 20-class ID/name map, hashes and
  reloads every finite nonempty `T×2048` I3D array, reparses the raw prediction
  pickle, and requires exact prediction/evaluation video-set equality. Pinned
  upstream Git blob SHA-256 values are config
  `c0ac0df560cd564941b56cd9391ad0bd5cea386d2e4b6cf9fc8ffcab821955cd`
  and README
  `f0431584b4df0702fa08f961fb0038e1277f41c12b7df47b7d2bfed47e59af23`.
  Job `1205178` is the unique clean official rerun. Until its
  `STRICT_COMPARABILITY_VERDICT.json` says `main_table_eligible=true`, `66.83`
  is only an official-reproduction candidate, not a paper-main-table result.
- SDPQ support Job `1205132` failed with
  `sdpq_support_overlap_query_padding_mask_omission_v1`: the support-overlap
  branch clamped padded zero widths to epsilon and omitted the final
  `query_mask`, causing tiny positive coverage at the first ownership
  interval. Production now masks support mass after both geometry branches;
  valid queries are numerically unchanged. The exact Linux regression passes,
  and unique diagnostic successor Job `1205179` is running. This is an
  implementation-correctness repair, not an SDPQ performance result.
- Independent four-condition recomputation Job `1205133` remains the sole
  active replay. Matched method rows remain fail-closed until a live,
  base-anchored source-diff attestation proves that the candidate is an exact
  official-stack descendant with one declared intervention. Consequently no
  current OpenTAD `30.42/44.88/30.88`, v16 decode-cross number, or historical
  `63.61` may be placed beside the official anchor as a paper-comparable method
  result.

## 2026-07-29 current official-comparability closure

- Job `1205206` evaluated the released official checkpoint at mAP@0.3–0.7
  `82.133988/77.805571/70.953608/59.401673/43.872118`, Avg-mAP
  `66.833392`, with raw prediction SHA-256
  `1333df9202eec7ae217542b6bd2b15b597c1a004ebb3634de54a7a37adb6d7fe`.
  The output is numerically valid but is not yet paper-eligible: its old
  receipt falsely recorded seed `0`, whereas the official config and training
  log use `1234567891`. A fresh fifteen-receipt reseal is required.
- Official source/config identity is commit/tree
  `61ea7eb9308a568b0cf45e3804830836e30061de` /
  `7b06c5261ba244788c942a0d73e304581bc35154`, and the full official
  effective-config canonical SHA-256 is
  `835cf30fbcfd27bd6af8885fff002813c8596e2948fce3adf29e3716f316dde4`.
- The hardened matched-comparison gate is commit/tree
  `e2a0d74f561b158c531d4909e72ecee69b153c16` /
  `0b6cb7996ee90f3209a78b78bbf7a55525e3badd`; the exact Linux suite is
  `127 passed, 2 skipped`, log SHA-256
  `115dd497a3a662b3fc0f19ae9104257d245cbadbb7fd4001f3eb3ea71432534c`.
  It requires a clean live official-base descendant, exact allowed A/M source
  paths, no rename/copy/delete, expanded effective-config equality for all
  protected fields and exactly one declared method intervention.
- SDPQ support successor Job `1205240` passed all 647 assignment/support
  matches with zero missing evidence, collisions or uncovered positives.
  This closes observability only; it does not reverse the matched SDPQ
  performance result.
- Independent recomputation Job `1205243` failed with
  `independent_recompute_semantic_match_drift_v1`. The failure is pinned to
  stable NumPy/float64 versus PyTorch `2.0.1` unstable CPU sort and scalar
  float32 C++ Soft-NMS/`expf` semantics. Raw tensors, proposal geometry and
  all delta signs matched, so it is not model evidence. Job `1205388` then
  failed before validator startup with
  `slurm_module_function_unavailable_v1`; the non-login allocation did not
  export `module`. Unique Job `1205400` directly activates the pinned Conda
  environment and tests the same exact semantics without tolerance widening.
- A paper-comparable K384 candidate must operate on the original full-video
  feature grid with deterministic max-K384 mask/scatter and measured skipped
  computation. Rank-coordinate remapping, fixed-window cropping, GT-dependent
  selection or zero-fill-only “savings” are excluded from the main table.
  Implementation starts only after the official dense anchor is resealed.
- Official anchor Job `1205409` failed before inference with
  `official_environment_probe_nms_import_order_v1`. Job `1205419` repaired the
  probe and completed both official evaluations at Avg-mAP `66.83`, but the
  builder then failed closed with
  `official_released_train_log_default_serialization_omission_v1`. The sole
  parsed difference is missing `model.fpn_start_level` in the released log
  versus exact integer default `0` injected by the pinned official loader.
  Failure receipt SHA-256 is
  `079818253bc87a78ed67ce41dbd092aa64f0e54b5a61972f2313adeb7d10fa4a`;
  this remains a diagnostic numerical reproduction, not a paper row.
- Audit commit/tree `8b80c98ee2af65561bf305b4fdc2ef16e460da73` /
  `148a93eac4ff1b6a3be46fdca72c705aa17294a6` pins the released raw train-log
  effective-config SHA-256
  `ad426e1a25be48423e21f854bbc6d815c6063388811350ad5fada5ac8933d3a7`,
  attests only the upstream `fpn_start_level=0` default, hashes raw and
  normalized configs, and still requires exact normalized equality with
  source-expanded SHA-256 `835cf30f...dde4`. Focused tests are
  `44 passed, 1 skipped`; clean Linux v18 passed `131 passed, 2 skipped`, log
  SHA-256 `6899bf6126d1ce9b3d880d348cdf5c1f152235d3b2e6f6de028b5fc807fb34fb`.
  Official anchor Job `1205455` completed `0:0` (`1205454` is test-only).
  All 15 receipts pass and the strict verdict is
  `main_table_eligible=true`, `official_actionformer_protocol_match=true`,
  with mAP@0.3–0.7
  `82.133988/77.805571/70.953608/59.401673/43.872118`, Avg
  `66.833392`. Completion/protocol/verdict SHA-256 values are
  `90c8bae14fcb20cc2434cea37f47065704766e38ff9663eac6e70c0d338b9e94` /
  `808199b54b0ebcfebda403419873cc5fd46c36a4d404d3d8ce31838ce3b5bd95` /
  `0706247ef978bf339f9a9cb4adaef07500e8d991129c6d0862118088b917a2ec`.
  This `66.83` row, not historical `63.61`, is the locked paper comparator.
- K384 implementation is now authorized only as one official matched-method
  intervention on the original full-video/FPN physical grid. The primary
  control is deterministic stratified-uniform native-grid support; a
  video-hash-fixed random mask is secondary robustness evidence. The head must
  truly skip unselected-query computation, scatter selected outputs back to
  original indices and preserve points, GT mapping, loss, decoder, NMS,
  schedule, seed, EMA checkpoint and evaluator. Any diagnostic pilot or
  zero-fill-only result remains excluded from the paper main table.

## 2026-07-29 current official-native K384 execution state

- Frozen decode-cross Job `1203917` completed `0:0`; all four formal
  components and the explicit suite validate. This closes the old
  `selected/physical × online/EMA` decoder-axis diagnostic at status `tested`;
  it is not an official ActionFormer-comparable paper row.
- Official anchor Job `1205455` is the locked comparator: mAP@0.3–0.7
  `82.133988/77.805571/70.953608/59.401673/43.872118`, Avg `66.833392`,
  `main_table_eligible=true`. Historical `63.61` is forbidden as the
  comparator for the new route.
- The exact official-native candidate is branch
  `codex/actionformer-sparsehead-official-matched-20260730`, commit/tree
  `55763a9ef7ce18a51827fe48040081c4fe2b84d4` /
  `c489a54aa501b39421cddb5df98385b3889ed479`. Its preregistered intervention
  is: full official I3D observation input, model query budget `384`,
  deterministic stratified-uniform native-grid selection, original physical
  FPN geometry, and training loss support
  `selected_native_grid_queries`.
- The loss-support change is explicit and material: unselected positives and
  negatives do not enter the sparse model's training loss or its normalizer.
  Therefore K384 is not an execution-only/unchanged-loss intervention.
  Released-checkpoint K384 evaluation is `diagnostic_only`; a paper comparison
  requires an official dense retrain and sparse retrain under the same
  5-warmup + 30-epoch schedule, terminal epoch-35 EMA evaluation, seed,
  manifests and evaluator.
- Candidate Linux tests passed `11`; audit focused tests passed `40`, launcher
  tests passed `5`. The exact source-diff receipt SHA-256 is
  `409ffd3035a0c957d3b250db24fe017c5c09efda526d746ace0d54f00c695abc`;
  the pinned official effective-config SHA-256 is
  `835cf30fcdce07ad25c0d173be90c908d8c13486c132487f8e2e8e49c80dde4`.
- Job `1205541` is the unique real-CUDA correctness and isolated-head-cost
  gate under run root
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_native_grid_k384_cuda_gate_20260730_v1`;
  `1205539` is test-only. At submission it was `PENDING (Priority)`, so no
  duplicate is allowed. Deployment/submission receipt SHA-256 values are
  `b37a08c2031bb7b043038ea6baf022830bda4ca1203abbff41619401537e8a8e` /
  `471022b2e726cf923e5a445aef8c21ca5f17c9e59b7e586ed8fb3ed4bbc49665`.
- Preserved preflight failure signatures are
  `audit_preflight_test_path_drift_v1`,
  `slurm_module_function_unavailable_v1`,
  `runtime_profile_source_under_nounset_v1`,
  `github_http2_remote_ref_transport_v1`,
  `github_remote_ref_live_check_transport_hang_v1` and
  `readonly_bundle_push_target_v1`. None is a model result.
- Required order is gate → released-checkpoint inference diagnostic → matched
  official dense/sparse retraining → Pro-level result analysis. A gate failure
  authorizes only engineering correction and a fresh immutable successor, not
  training or a performance claim.

## 2026-07-29 current K384 CUDA recovery state

- Job `1205541` is terminal `FAILED 2:0`, not a model failure. Numerical
  equivalence, raw/mask immutability, exact K384 and zero unselected outputs
  all passed. Cost failed: dense/sparse-preselected/sparse-with-selector means
  were `6.1918/19.6508/20.5046 ms`; overall speedup `0.3009x` and all three
  rounds about `0.30x`. Exact signature:
  `native_grid_sparse_head_microkernel_launch_and_scatter_slowdown_v1`.
- Candidate commit/tree `d64e66dfd7fc9881552b342f5523926cc78c0848` /
  `16265c70b235034acb52521b00c259ec6d8b59e1` replaces per-sample/FPN
  micro-kernels with one packed convolution per head layer. The method contract
  and official protected config remain unchanged. Linux candidate/audit/
  launcher suites pass `12/40/1`; source-diff SHA-256 is
  `5aea817bf1fd1b2c0e36193b9d99ee71bde3dfd00c05673ece5dc4f6da9304d4`.
- Job `1205567` is the only formal successor under
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_native_grid_k384_cuda_gate_20260730_v2`;
  `1205566` is test-only. Current state is `PENDING (Priority)`.
  Deployment/submission receipt SHA-256:
  `c2890c1b37e22810fdc8284b80ca6292e7bf5cc1c38820fb74e8d68d96647b52` /
  `f71c394c09f5d5a65bdf37036739294d553098ea3ecfa79b8ebf10c8486b3798`.
- A one-off SCP close occurred before submission while copying the
  pre-submission hash file. Remote inspection showed `MISSING`, the retry was
  hash-verified, and the job was submitted only afterward. Signature:
  `ssh_transport_interruption_during_pre_submission_receipt_copy_v1`.
- Do not relax the timing gate. A pass is still isolated-head engineering
  evidence only; released-checkpoint inference remains `diagnostic_only`, and
  paper eligibility still requires official matched dense/sparse training.

## 2026-07-29 current K384 global-packed recovery state

- Job `1205567` is terminal `FAILED 2:0`. Numerical equivalence passed, but
  dense/sparse-preselected/sparse-with-selector means were
  `6.2409/12.7657/13.6182 ms` and selector-inclusive speedup was `0.4590x`.
  Signature:
  `native_grid_sparse_head_packed_patch_materialization_and_microconv_slowdown_v1`.
- Commit/tree `31e6112ea28747098cfe5412c097d737731bfaa1` /
  `d2619cd075c4e7192ca060f34d811ac3fe5768f8` replaced packed Conv1d with exact
  flattened GEMM. Job `1205569` is also terminal `FAILED 2:0`: correctness
  passed, but `6.1934/12.7646/13.5469 ms`, speedup `0.4577x`, all rounds near
  `0.458x`. Signature:
  `native_grid_sparse_head_packed_gather_scatter_overhead_v1`. CUDA gate and
  failure-analysis SHA-256 values are
  `7e91345babcce40bb9a157d2b29fbc718fe7f0e2a059bdc02e2edff386709197` /
  `8b49859031a48ef2a4367a156f452761c66a1e75c1a5e6a87a8fb242766f3a50`.
- Exact remote v6 was built from bundle SHA-256
  `c50bea0b79e242bb4c96cf11fb35a3ef095a8b9c3bc4a13fc56abca02be4ec49`
  after preserved failures `github_https_clone_tls_termination_v1` and
  `bundle_clone_remote_head_unset_v1`.
- Remote live-ref source diff failed as
  `github_remote_ref_dns_timeout_during_source_diff_v1`. Local live-ref
  attestation/provenance SHA-256 values are
  `3ef485f82678453538aef6f58ba81d548149394ef93356a811593e67cdf22e9d` /
  `780c0aa5a8a00ba9180974d4bee001782e83d747d492589a2a2da4b5bc40e2d6`;
  the latter explicitly has `paper_main_table_seal_allowed=false`.
- Next is one bounded `designed` global packed-state implementation: share the
  cls/reg plan, retain raw-hole semantics at layer one, use exact sparse
  physical lookup afterward, and scatter only final outputs. Below `1.0x`
  stops this execution route; `>=1.05x` is still required to unlock matched
  training.
- Never present the sealed released official `66.833392` anchor as a matched
  causal comparator. A paper result requires same-commit dense and K384 sparse
  training under the official 5-warmup + 30-epoch loop, terminal epoch-35 EMA,
  identical seed/data/evaluator, independent receipts, multiple preregistered
  seeds and synchronized end-to-end cost.
- Global packed-state candidate/audit commits are
  `d86a4acda21e35a1609f19f1a46bc470ee18b7e1` /
  `14bd14f9b6a087dc2ec623fc4238c89e0cb86960`; exact Linux suites pass
  `14/18`. Full-content preflight/deployment/submission SHA-256 values:
  `08b05123edbaccd10d5b43031a43ebac11a3616ceb454bfbd588d4d7395a6a95` /
  `f070f46f023be6152faf1818342633a8d6f713fb55e37fa5c79fc2a43434f140` /
  `04d206c3ad220155f8f63a1b6a086c6c3c6c5beaeac13a7a001334f2d0fef4c7`.
- Job `1205571` is the sole formal v4 CUDA gate and reached `COMPLETED 0:0` in
  `00:00:30`; `1205570` is test-only. Dense / sparse-preselected /
  sparse-with-selector median latency was `6.240573 / 3.129646 / 3.970906 ms`,
  and selector-inclusive median speedup was `1.571574x`; all synchronized
  rounds passed the unchanged threshold at
  `1.573072/1.570395/1.568621x`. Gate/completion SHA-256:
  `cddfb80af237a41d3c3e1121e39cbc5114ad8abc472c56f6daf519a50cf95988` /
  `ceec00f799eb40a1dd56c1949576783e06599205d63f1d1909a598787d99fd85`.
- This is `tested` isolated-head CUDA evidence only. It explicitly has
  `paper_metric_claim_allowed=false` and
  `end_to_end_wall_clock_claim_allowed=false`; do not put it in a main result
  table. The next authorized evidence is same-candidate-commit official
  dense/K384 retraining, both evaluated from terminal epoch-35 EMA with the
  exact official data/evaluator. The first paired seed is screening; paper
  statistics require preregistered independent seeds after the structure is
  frozen.

## 2026-07-29 current official matched K384 training state

- The source proof is now remotely and live recomputed against both published
  GitHub refs, SHA-256
  `a07d038d87632d1f8cc984ba24af44ca7ce9a9902e30e501f5de80a32265d46b`.
  Unlike the earlier local provenance receipt, this is eligible to participate
  in a later paper record.
- The official data was revalidated live: all 413 `T x 2048` I3D files produce
  the exact sealed feature-manifest SHA-256
  `cda269dace393b9af1f6fcb87a9a531beed69e3c71279ba3ca2cee76e198d59a`;
  `thumos14.json` SHA-256 is
  `3b025685a07fb98fc58d2399fb5fa9493c2168632d8ae1a8c3f4689897d2fbb2`.
- Audit commit/tree `643c42e8cfe4018fb891202f7ffdae554acc2e4a` /
  `25fa3eda9fc62960c69c2952c957ebab39e71c27` adds the formal paired launcher
  and independent official raw-prediction evaluator. N16R4 focused tests pass
  `18/18`; log SHA-256
  `f15e5d2c6b8cfeba5a31489b318f3e784233ddc8880fd09966e98e6ff63fcded`.
- The pair uses one candidate commit for dense and sparse, official seed
  `1234567891`, no resume, the official 5-warmup + 30-epoch loop, terminal
  epoch-35 EMA, identical I3D inputs/post-processing/evaluator, and an
  independent metric recomputation. K384 plus
  `selected_native_grid_queries` is a method intervention; mAP differences
  must not be called a pure kernel effect.
- `1205572` is test-only. Unique Job `1205573` is `PENDING` at
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_matched_pair_k384_seed1234567891_20260730_v1`.
  Preflight/deployment/submission SHA-256:
  `3b827cfe10b3267d013373f89a9c3b90b2eb6f450b0aa4b7d1e5082615a0ac4e` /
  `65cb544960c619f4243c7829a41950719d2591493c05fbad70a07f1b9a037da2` /
  `ead6f35af71e2de9308d6ed0aad642dc27845e68169f1cee8ca32e3d157a3e77`.
- This job is single-seed screening and its completion schema hard-codes
  `paper_main_table_eligible=false`. Do not report its delta as the final
  paper result. If it passes the preregistered accuracy and engineering
  criteria, freeze the method and run paired independent seeds plus
  synchronized end-to-end cost before any main-table claim.
- Preserve pre-Slurm signatures
  `official_data_live_revalidation_import_scope_v1` and
  `preflight_failure_receipt_python_environment_unloaded_v1`. The failed v1
  root produced no training/model result; v2 data revalidation passed.

### Matched-pair engineering recovery

- Job `1205573` is terminal `FAILED 1:0` after 31 seconds, before tests/training.
  The compute node lacked GitHub DNS while the launcher redundantly repeated
  the already sealed `ls-remote`. Signature:
  `compute_node_github_dns_during_redundant_live_source_diff_revalidation_v1`.
  Failure/runtime/stderr SHA-256:
  `f0bf8fe6258260d55fffe88d35dfb75d647340adccb06dc2efae1c5e419c64d9` /
  `8bc85a66f37bc98eec780ec76ef5fab1978bd45195917780c269267dc5b2a057` /
  `fef8ce4b812cf04882328f4f12a5ddcac8c61077a1f8107c19b63e142808d74b`.
- Audit commit/tree `debbde469f938e09e4debfe7831e64755ae665f5` /
  `3721612aae55eecb07e9f4183a53e1d8156e143b` keeps the live-ref attestation
  fixed but revalidates every local Git/config/diff/allowlist field offline in
  the allocation. Remote tests pass `19/19`; test/offline-validation SHA-256:
  `d3d76af3095d792b6af0a8709a7e83addca17aa8e1d5e4d36a13b9cc8d9856f7` /
  `f409abc67b630fbc6c1b30db7ba5e614ecb8925a2d5c7aa6b0e9d7746581067b`.
- GitHub cloning v24 failed with TLS termination. It remains preserved; exact
  clean v25 came from verified bundle SHA-256
  `6c59f1d568017d8ee82e32d3132b595b73c3d469a2cb91976968d330cd789104`.
  A bundle verification launched outside a Git repository and one SSH
  interruption were transport/invocation failures, not source/model failures;
  verification then passed from the exact v23 repository.
- `1205579` is test-only; unique successor Job `1205580` is `PENDING
  (Priority)` at
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_matched_pair_k384_seed1234567891_20260730_v2`.
  Preflight/deployment/submission SHA-256:
  `45e60ba0f68132b8cfa11ec036ed71789e83d718dc300df62f0cdf19f1375e8a` /
  `a151cf03c67395771eb386c6fe48687e867b40df1d8f7a562be6d1df459125a0` /
  `fca38a1cad01222ef8bda967116993742319bdc94b2d8e9582a783abe21c479f`.
  The same-commit/seed/data/epoch-35 EMA/evaluator contract is unchanged.

### Declared-dependency recovery and current unique successor

- Job `1205580` failed `1:0` after 26 seconds, after offline source validation
  and focused tests but before any optimizer step. Official `train.py` could
  not import TensorBoard. Preserve
  `official_declared_tensorboard_dependency_missing_v1`; failure receipt
  SHA-256 is
  `a959ef415f383d5368edf806b1166cca9cd25e91e49ea4398853775059e35385`.
  This is an official-declared environment dependency failure, not a model
  result.
- The repair uses isolated venv
  `/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_tensorboard_2_20_0_20260730_v1`.
  It adds TensorBoard `2.20.0` while preserving Python `3.10.20`, PyTorch
  `2.0.1`, CUDA `11.8` and NumPy `1.23.5`; receipt SHA-256 is
  `acc5909360970cfad1f390a4f5ab046a3876ac9378448b2f94da26ffb312ece2`.
  The SummaryWriter probe leaves RNG state unchanged.
- Audit commit/tree
  `a3d987961c0e6ac0166194cfc30ca0d375765ef1` /
  `51c53773d266e614d6c1054a1e6127fe73c69f38` is frozen in exact clean v26
  from bundle SHA-256
  `e8812a84489bb55aea419b1b637778574539a44b0c7399b18a04d346430ce419`.
  Remote focused tests pass `19/19` (SHA-256
  `f18a52300731975c81d0fffa1cd4c8e5787ccc83b07abba212d4d2a1f6fcbb7c`).
- Fresh full-content preflight/deployment/submission SHA-256 values are
  `9ff27367e10717b012d0f06a85b980f54c9b91a6fe45be9e8f87c00cac90d47b` /
  `00736c6b07fff77e0a6ca92ad24744eab0e2c089a22b350f9f2537054891b4f4` /
  `f4512010b2d675611f97e61a929ee4edda421b7f29506969d49028b3a7ac041a`.
  `1205583` is test-only. Unique Job `1205584` was observed running on g0024
  under fresh immutable v3 root; environment/source probes, candidate `14/14`
  tests and audit `4/4` tests passed, and dense training had just begun.
- Scientific conditions did not change. The job is still
  `experiment_running`, single-seed screening and
  `paper_main_table_eligible=false`. A main-table conclusion remains forbidden
  until frozen paired independent seeds, uncertainty and synchronized
  end-to-end cost are complete.

### Paper-stage protocol frozen before S0 metrics

- New node:
  `experiments/actionformer-sparsehead-official-main-table-prereg-20260729.md`,
  status `designed`.
- Five paired seeds are fixed as
  `[1234567891,1423812477,737690612,1788897292,1322022747]`; canonical
  seed-set SHA-256 is
  `a4038a752aa46b97e5854c20574d65ece078bad6124e4778cc4269e75747c7c6`.
  Four extra seeds are derived from SHA-256 of a fixed namespace and cannot be
  replaced or performance-selected.
- S0 GO requires complete validated dense/sparse results with
  `Delta Avg>=-1.00 pp` and `Delta mAP@0.6,@0.7>=-1.50 pp` each. A legal
  result below the bound stops automatic multiseed deployment and opens the
  frozen negative-result analysis; it does not authorize tuning.
- Paper accuracy-preserving efficiency requires all five pairs, seed-level
  paired 95% intervals with Avg lower bound `>=-0.20 pp`, mAP@0.6/@0.7 lower
  bounds `>=-0.50 pp`, plus precomputed-feature detector-pipeline median
  speedup `>=1.05x`, lower CI `>1.00x` and no duration-stratum regression.
- A 2x2 cross of full/selected training support and dense/K384 evaluation
  queries is preregistered to separate execution/representation loss from
  selected-loss optimization without extra checkpoint selection.
- Before the paper study, strengthen the launcher from a 413-file count to a
  live per-ID/content/shape/dtype feature rehash and add effective-config,
  command, split and terminal optimizer/scheduler/EMA receipts. Job `1205584`
  remains useful screening but cannot satisfy those main-table requirements.

### Current S0 engineering failure

- Job `1205584` is terminal `FAILED 1:0`, elapsed `00:09:48`.
- Dense completed exactly 35 epochs and wrote checkpoint SHA-256
  `ea3c16fcf17fd6fb8cec57829804e96736a8ab231b07d820e5939fd5db3cba00`,
  but save-only EMA evaluation failed before any metric; sparse never started.
- Signature:
  `official_actionformer_softnms_extension_abi_shadowed_by_opentad_v9arg_v1`.
  The candidate's official caller supplies seven Soft-NMS arguments, while
  Python resolved the unrelated OpenTAD site-packages extension that requires
  nine. Loaded extension SHA-256 is
  `4ccea1d7bae60a3edb735280c564928f18e89bd01e160a1c9fa200625a660450`.
- Failure-analysis/saveonly/runtime SHA-256 values:
  `99f83a03715fa935a422451f9fe842aeaae867546d37c9af39cda8869958f852` /
  `496468bf5c327ae0a31a3a581cc086fd7cfb69dd5d2b249b088acc6e8aee7338` /
  `b3f8cca479ad22a674a433badcabb9d928b012af7b60221ec11f8e54e5bf6cc5`.
- This is a non-model ABI/provenance failure. Recovery must build and
  receipt-bind the exact official extension in a new isolated environment,
  assert the seven-argument call at preflight, freeze a new clean audit runtime
  and retrain both arms from scratch under a fresh run root.

### Official-comparable S0 recovery now running

- Audit commit/tree
  `71f955a7301f07875a35e0be366241e548e5c775` /
  `d328093644e040741e16dbdd8bc93b6b0d608a10`, exact v27 bundle SHA-256
  `a9ee267333c9371d087e806fe61cef19c14122b18fee1a4e6c75fa4c58846ad6`,
  changes only NMS module provenance. It keeps the official seven-argument
  caller and rejects the unrelated OpenTAD nine-argument ABI.
- Isolated runtime/environment receipt/NMS extension SHA-256:
  `/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_official_runtime_20260730_v2` /
  `13d57c1161905f059204f7101f26029503a03da7f5eb44b81c418a0b97999f24` /
  `b67e0e41f9f55cd69e8b90cfc75a1947214365857d851a510047838ad49ed98d`.
  The actual seven-argument probe passes; remote focused tests are `14+5`.
- Official and candidate dense configs are byte-identical at SHA-256
  `c0ac0df560cd564941b56cd9391ad0bd5cea386d2e4b6cf9fc8ffcab821955cd`.
  Official THUMOS uses `validation` training and `test` evaluation; do not
  replace this with generic default splits.
- A live full rehash of all 413 I3D files exactly matches sealed manifest
  `cda269dace393b9af1f6fcb87a9a531beed69e3c71279ba3ca2cee76e198d59a`
  for IDs, content, shape and dtype. Receipt SHA-256 is
  `73a2f714c100f541306d7d7f9c32e36481574d2ac6c5e78925ee4ee1dcca96b3`.
- Full-content preflight/deployment/submission SHA-256:
  `d9e1f897de51e46aac52cb450f72daa8bc19a64bf999b01112013489038d4a55` /
  `b8d4079c9ddc8faa7a0a575dbe63f700c2448409df5dbccf972101cc0e4a282b` /
  `2a31a1d01056f39159d17d99fb9047f5bd6946b68475c1eae31008659df07a08`.
  `1205593` is test-only; unique Job `1205594` is `PENDING (Priority)` at
  fresh v4 root. It restarts both arms and remains
  `paper_main_table_eligible=false` single-seed screening.

### Job 1205594 import-order failure and exact recovery

- Job `1205594` is terminal `FAILED 1:0` after `00:00:04`. It stopped at
  `python_environment` before focused tests or training.
- Signature is the known
  `official_environment_probe_nms_import_order_v1`: `nms_1d_cpu` was imported
  before `torch`, so the dynamic loader could not find `libc10.so`.
- Failure/failure-analysis/runtime/stderr SHA-256:
  `68d2ec8ddd1d2a69c1181d532325368975c95a306a2c7d368226905044ee321f` /
  `06bbc29e5f57b3b9a12f421f5ddd814487bf01733d0f0e5bbcc4c0551c877a41` /
  `5988ed65e4ebbd8dde6a334ffe7f2ae3c8825fbd5e61d1c6198388c0284443fb` /
  `4c6d87aa6b85dbbe173a1eae119bac562aa43f55c6fa847b256c9c05d25c79e0`.
- The recurrence exposes a deeper regression gap: path/hash/arity were tested,
  but `torch-before-extension` import order was not. Commit/tree
  `98f5b875315b4a2b5c6829f5d74ccce68f478e47` /
  `2e6b4bba6868c323d70c97140f7cbed044eb1a7b` adds that exact assertion.
  Clean v28 bundle SHA-256 is
  `713a1d839e8e8ea50f141df9dba1feb44dc43c91dffbd4dd85bf8910bbdf9e24`.
  Remote exact recovery is running; no successor is yet authorized until it
  passes.

- Remote torch-before-NMS/seven-argument probe passed (log SHA-256
  `7d79381ed64b27059aa6f4204bbfce3f606fc1e81e0a7962e4e1d1c7413a0488`);
  candidate/audit exact suites pass `14+5`.
- Fresh preflight/deployment/submission SHA-256:
  `19230f06e0eda57c34607db250dba9ebc1f0d6365e5ab33c339dffe0468ddd86` /
  `250068a1de36c00fabe37596e302dc9e3fd22249be09b267fc4e9762e6f4ce46` /
  `0549ff04a30bb4efea176a484a6f51d652b8bdd023227564b0fc2fdfe492cabf`.
  `1205598` is test-only; unique Job `1205599` is `PENDING (Priority)` under
  fresh v5 root. It is the only active S0 attempt.

- Job `1205599` then entered `RUNNING` on g0030. Environment/source and
  `14+5` focused gates passed inside the allocation; dense training reached
  epoch 24 with finite loss. No metric exists yet.

- Dense is now a validated `tested` component. Epoch-35 EMA, no-resume,
  exact 212-video coverage and independent recomputation all pass. Avg-mAP is
  `0.6658301251307708`; mAP@0.3–0.7 is
  `0.8190849486121916/0.7795203466370499/0.7128549836803181/0.5825550463357125/0.43513530038858167`.
  ARM completion SHA-256 is
  `a15b0526ef9a75a0fe32c0798b609c738781ab5c063c53df165ace6cbcdf138a`.
  Sparse is training, so the S0 delta remains undefined.

### Official-comparable S0 is complete and negative

- Job `1205599` completed `0:0`; `MATCHED_PAIR_COMPLETE.json` SHA-256 is
  `545e420aa1d437aedeffd15cb30390ceb0cfe4d6565d7eb35c53a8bf17ac76fd`.
  Both arms are fresh epoch-35 EMA runs with exact same commit, official data,
  seed, schedule, environment, Soft-NMS and evaluator. Each independent
  recomputation covers 212 official test videos and 42,400 predictions.
- Dense Avg/mAP@0.3–0.7:
  `66.583013 / 81.908495/77.952035/71.285498/58.255505/43.513530`.
  Sparse:
  `43.919699 / 64.925248/56.642845/45.952641/32.783177/19.294586`.
  Sparse-minus-dense:
  `-22.663313 / -16.983246/-21.309190/-25.332858/-25.472328/-24.218944 pp`.
- The frozen S0 bounds fail decisively. Stop the five-seed and cost study for
  this exact K384 + selected-loss intervention. Status is `tested`; the
  intervention-level negative conclusion is `empirically_supported`;
  `paper_main_table_eligible=false`.
- Do not compare the sparse arm causally against the released `66.833392`
  anchor or historical PhysTime/OpenTAD values. The matched causal control is
  the same-run dense `66.583013`.
- Primary mechanism hypothesis is combined inference coverage loss and
  selected-query supervision/normalizer damage. Alternatives are
  calibration/Soft-NMS density mismatch and a lower-probability implementation
  contract defect. The immediate plan is the preregistered no-retraining 2x2
  checkpoint cross-eval, per-class/duration/boundary/retained-recall diagnostics
  and assignment/support observability. No silent rescue tuning or retraining
  is authorized.

### S0 attribution is closed; DCSR is the only continuing SparseHead route

- Job `1205701` completed the frozen no-retraining 2x2:
  full×dense `66.583013`, full×K384 `45.784332`,
  selected×dense `64.537343`, selected×K384 `43.919699`.
  K384 execution main effect is `-20.7082 pp`, selected-loss training
  `-1.9552 pp`, interaction `+0.1810 pp`. Hard proposal/query removal is the
  dominant observed cause; selected-loss is secondary and disproportionately
  hurts high IoU.
- Post-NMS class-aware/class-agnostic recall@0.7 falls from
  `76.50/80.85%` to `42.41/44.55%`; topK gaps widen and all 20 classes are
  negative. Score compression exists but cannot explain absent segments.
- Job `1205799` completed the 64-window official-training-split
  assignment/support audit. K384 retains `461/2721 = 16.9423%` positives,
  leaves `395/804` GT with no candidate and `427/804` with no assignment.
  Suite SHA-256 is
  `475b61ddad4b0b56a86b2e2616ef2584b252c3169b4ad1268223f21d6e118567`.
  It uses no test GT or training and is diagnostic-only.
- There is no contradiction with earlier physical decode gains: decode changes
  can improve retained frozen predictions but cannot recover proposals removed
  before decoding.
- The hard K384 selected-loss method is closed. The only continuing route is
  DCSR (`designed`): keep a cheap dense proposal/support scaffold on every
  native query and sparsify only expensive residual refinement. No DCSR result
  exists yet.
- The next official study is preregistered for five paired seeds and complete
  feature-to-detection cost. Internal architecture/budget selection is
  validation-only. A positive paper claim requires both non-inferior Avg/high
  IoU and synchronized detector-pipeline speedup; no S0/2x2/assignment row may
  be promoted to the positive main table.

### DCSR G0 passed; validation-only G1 is running

- DCSR is now `implemented/tested/experiment_running` at exact commit/tree
  `bf0df83d7400c89fc61f38d169d68085420a2263` /
  `2f9346fcfd2bfb7fc5a76a86ef65545030a67469`.
- G0 uses the complete official dense head with residual disabled; it is an
  identity gate, not a cheap-head result. Real-CUDA Job `1206168` completed
  `0:0`; receipt SHA-256 is
  `b87fc59ec6529e83e99f7bf5fbfb7f3bff5ec637060c62057da07a669a8c1ff4`.
  State keys, points, full masks, pre-decode logits/offsets and final official
  Soft-NMS/timestamps are exact.
- G1 is one-layer dense scaffold everywhere plus uniform K384 three-layer
  signed residual refinement, with full-grid supervision/normalizer. Formal
  array `1206273_[0-2]` uses three frozen dev seeds and a validation-only
  160/40 manifest SHA-256
  `ba683bc5ddbb1fe219fab0545e9d808808d9b25fc9b32e7c5c0b6339b68b9bbb`.
- `1206266` is test-only. Jobs `1206160/1206166` are preserved zero-metric
  launcher/import engineering failures, not model results.
- G1 may only choose or kill the architecture. Its numbers are forbidden from
  the paper table. Official paper eligibility still requires G0--G4 freeze,
  five paired full-validation-to-test seeds and complete synchronized cost.

### DCSR G1 failed; the current SparseHead route is closed

- Formal G1 array `1206273_[0-2]` completed for all three frozen seeds.
  Aggregate SHA-256 is
  `b98d59468ef39aa6fe6de387adfd6f872c848ab8f63b26c3bf1bf6161f5f7939`.
- Dense/DCSR mean Avg-mAP is `0.5680730871/0.4925110665`; DCSR-minus-dense is
  `-7.556202 pp`. Mean mAP@0.3--0.7 deltas are
  `-5.719942/-3.507626/-6.490487/-11.043134/-11.019821 pp`. Every seed and
  threshold is negative; Avg-delta SD is only `0.3139 pp`.
- No-training counterfactuals isolate scaffold-only at `-7.418076 pp`,
  all-query residual at `-6.316665 pp`, all-query residual value at
  `+1.101411 pp`, and the K384 support penalty at `-1.239537 pp`
  (`-2.4102/-2.3511 pp` at 0.6/0.7).
- Leading attribution: the one-layer scaffold/decomposition is weak relative
  to the official dense head. K384 residual support is a smaller,
  high-IoU-sensitive factor. Residual final projections are nonzero by epoch 5
  and continue updating, so persistent zero-init death is not the primary
  explanation; representation versus optimization is not uniquely identified.
- At recall@200/tIoU 0.7, dense/all-query/K384 class-aware recall is
  `62.25/53.03/49.86%`; K384 median normalized start/end boundary errors are
  `0.1803/0.1549` versus dense `0.1371/0.1156`. Losses concentrate in 2--8 s
  and 16--32 s actions and several high-IoU classes.
- The exact G1 hypothesis is rejected with `empirically_supported` negative
  evidence. The preregistered route stops: no G2--G4, five official seeds or
  cost run. A possible official-quality dense proposal floor with gated
  residual compute is only `discussed` and requires a new preregistration.
- G1 is a valid internal method-kill result, not an official benchmark row.
  Never compare its absolute values with historical `63.xx`, released
  `66.833392`, or official S0 `66.583013`.
- Diagnostic completion/prediction/checkpoint SHA-256 values are
  `954d7944428fcf0d26dd917ff9562a9c3e7a53de71c09e9a382aaf49f5bd4a53` /
  `47dcca7e179544e348966bf92cf92cddeff19a1fdc8cfea100150dc1bc580a36` /
  `c596bc942d2617e3824d21c96d0289316be4ee1ad465f23dc507b2d90466e006`.
  They record no test subset, no new training and no paper/cost authority.

### ODF-CR launch preregistration (historical state)

- At formal launch ODF-CR was `experiment_running`, not a continuation or relabeling of
  DCSR G1: official-quality dense proposal floor plus independent conditional
  residual compute.
- Its frozen internal training matrix is depth `1/3` × residual
  `off/all_valid`: `d1_off/d1_all/d3_off/d3_all`. Residual-off controls isolate
  scaffold depth and residual value; K384 is frozen replay, not a trained arm.
- A new holdout-v2 must be selected exclusively from the old train-160 and be
  disjoint from the already observed old holdout-40. Seeds
  `2026073101/02/03` are paired training repeats on that one split, not
  independent validations.
- Real-CUDA zero-tolerance `d3_off == official dense` is a blocking engineering
  gate. The depth-3 residual utility gate is `>= +0.25 pp` Avg, positive in
  `2/3` seeds and nonnegative at 0.6/0.7. Passing it unlocks only frozen
  `stratified_uniform` K384 replay with the preregistered support bounds.
- This experiment is validation-only method selection:
  `paper_main_table_eligible=false`, `official_test_authorized=false`. Absolute
  values cannot be compared with historical `63.xx` or official `66.xx`.
- Design commit is `77244d5`; the frozen formal implementation is
  `codex/actionformer-densefloor-factorial-20260731@01cdb78d2b7668098b6b13a1e49433d48fbc1a8d`
  with tree `e70d2956a197b1204e721239178e76152efe282b`. Linux focused
  preflight is `71 passed`; holdout-v2 SHA-256 is
  `b8cac555f3d31e02468dbca3b3b0ada2d30b05bf046c10eb16304abb92499d1a`.
- Formal array `1209259_[0-2]` began all three frozen seeds. All three
  real-CUDA G0 receipts pass the 14 exact official-floor/zero-residual/paired
  initialization checks; tasks have entered `d1_off` epoch 0. Unique G2
  successor Job `1209267` initially waited on `afterok:1209259`.
- At launch there were no arm metrics or model conclusions. Internal validation
  results were forbidden from the paper main table and from absolute
  comparison with historical `63.xx` or official `66.xx`.

### ODF-CR completed with a legal G2 negative

- Array tasks `1209259/1209260/1209261` and G2 Job `1209267` completed `0:0`.
  Runtime commit/tree remained
  `01cdb78d2b7668098b6b13a1e49433d48fbc1a8d` /
  `e70d2956a197b1204e721239178e76152efe282b`, clean and without a hard-failure
  signature. G2 aggregate SHA-256 is
  `9172eddcbf5f9a4943b303e20b57f4492f0a44b18c39f892d5829b1f0a79ddec`.
- Mean Avg for `d1_off/d1_all/d3_off/d3_all` is
  `61.2262/65.0952/68.7863/68.6056%`. `d3_all-d3_off` is `-0.1806 pp`, with
  paired seeds `-0.6970/+1.5070/-1.3520 pp`, only `1/3` positive, and threshold
  deltas `+0.8466/+0.7666/-0.3656/-2.7468/+0.5960 pp`.
- G2 fails mean Avg, positive-seed count and @0.6. No K384/G3 job or receipt was
  created. Do not tune K or reinterpret this legal negative as engineering
  failure.
- `d3_off-d1_off=+7.5600 pp` Avg and `+11.4261/+14.6071 pp` at @0.6/@0.7;
  class-aware recall@200 gains `+5.87/+13.00 pp`. The official deep floor is the
  supported internal prerequisite.
- `d1_all-d1_off=+3.8689 pp`, but the depth-by-residual interaction is
  `-4.0496 pp`. The all-valid residual helps the weak one-layer floor but has no
  reliable incremental value atop the deep floor.
- Twelve raw-prediction diagnostics show mixed classes, durations and videos.
  `d3_all` slightly improves best-match boundary/recall statistics yet loses at
  @0.6; late training loss is lower than `d3_off`. Saturation/overfit or ranking
  interference lead, while calibration/NMS and gradient conflict remain
  unproven because pre-NMS and gradient/gate telemetry were not recorded.
- Status is `tested`; the G2 residual-utility rejection is
  `empirically_supported`. Sparse conditional execution was not tested and is
  not universally rejected. No main-table, official-test, efficiency,
  significance, equivalence or absolute `63.xx/66.xx` claim is allowed.
- Terminal attribution and claim tracing are complete. The heartbeat monitor
  was retired after its self-delete RPC timed out; its configuration is kept in
  a recoverable archive outside the active automation directory. No new job was
  created, inspected or cancelled during retirement.

### 2026-08-11 — accepted Pro P0 semantic-repair decision

- Fresh Project Pro decision `PRO_P0_BLOCKER_DECISION-v001` (turn
  `duca-p0-blocker-51c88fd75537120ce96a417beb7e81dd`) is the active scientific
  authority and returns `REVISE`, preserving only
  `DUCA_FIXEDK_BOUNDED_MONOTONE_DENSITY_ACQUISITION-v001` as a candidate route.
  Evidence status is `BLOCKED_PRE_RESULT`: no model-quality, cost, data,
  validation/test, metric, CPU, GPU, or Slurm evidence was consumed or created.
- The P0 repair is correctness-only. It fixes the disagreement between the
  data-path uniform endpoint (766) and selector endpoint (767) at T=768/K=384
  through one integer-half-up canonical generator, and moves selected-to-
  physical raw-proposal transport to before every per-sample detector NMS.
  The detector architecture, assignment, head, loss, NMS configuration, and
  evaluator remain unchanged.
- Only three bounded no-execution queues are admitted: Builder prepares a clean
  exact-commit static patch; Evaluator writes a remote-P1 protocol amendment;
  Critic performs a read-only closure review only after Builder returns its
  complete diff. P1, P2, dataset traversal, remote CPU/GPU execution, metrics,
  Git push, and claim promotion remain prohibited pending a future fresh Pro
  decision. Do not reuse the historical 766-endpoint implementation or any
  post-NMS coordinate mapping.

### 2026-08-11 — P0 density decoder is an unresolved scientific definition

- The subsequent Builder plan and independent Critic review agree that a shared
  canonical-uniform generator and pre-NMS coordinate transport are deterministic
  corrections, but the frozen revision has no named positive per-time density,
  inverse-CDF hard decoder, or constant-density specialization. No patch or
  execution occurred.
- Do not reinterpret existing slot allocation, rank top-k, or transport symbols
  as the density route. That choice would define a new model mechanism. The
  active state is `SCIENTIFIC_AMBIGUITY / BLOCKED_PRE_RESULT` pending one fresh,
  serial Pro adjudication that names the density input and exact decoder API, or
  explicitly revises the route. No model result, metric, cost, or claim exists.

### 2026-08-11 — Pro P0 v002 resolves the definition, not the evidence gate

- Fresh Project Pro decision `PRO_P0_ROUTE_ADJUDICATION-v002` supersedes the
  preceding ambiguity. The only designed candidate is now
  `DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002`: a dedicated
  `duca_density_logits[b,t]` reader over dense `browser_memory`, with
  `selection_unit=1` and the identity physical candidate grid. Legacy slots,
  actionness/boundary signals, rank/top-k, quota, allocation and soft transport
  are explicitly not density aliases.
- Its named hard API is
  `decode_duca_density_positions_v001(density_logits_valid, requested_k=384)`:
  positive trapezoidal mass, endpoint-inclusive inverse-CDF quantiles and one
  constrained integer projection. Exact constant valid-prefix logits must call
  the canonical integer-half-up uniform generator; near-constant inputs do not
  receive a tolerance shortcut. The allowed deterministic corrections remain
  canonical uniform generation and exactly-once selected-q to physical-dense
  transport before per-sample filtering, top-k, IoU and unchanged NMS.
- This is `designed / BLOCKED_PRE_RESULT`, not implemented or tested. It grants
  no data, CPU/GPU/Slurm, metric, checkpoint, Git push, result or paper-claim
  activity. Builder's only current work is the durable minimal-change plan;
  Critic starts after a complete diff and Evaluator only prepares a no-execution
  protocol amendment. The accepted decision plus current state/history Sources
  are prepared locally and await a centrally leased Project-Source sync.

### 2026-08-12 — P0 projection objective is frozen; evidence gate is not

- Accepted fresh Project Pro decision `PRO_P0_PROJECTION_POLICY-v001` resolves the remaining nonconstant integer-projection ambiguity without changing the DUCA route: it fixes exact binary64-to-`Q=2^20` half-up target conversion, the endpoint/stride/displacement feasible set, exact lexicographic objective `(E2,E_inf,E1,U1,position-vector)`, ascending candidate order and typed fail-closed behavior. Do not substitute a weighted objective, tolerance, greedy/clip/dedup repair, legacy selector or uniform fallback.
- The mandated evidence level is `CROSS_IMPLEMENTATION_IDENTITY_REQUIRED` for equal serialized `(T,K,u,a)` inputs. It is explicitly not a result: no test, data access, CPU/GPU/Slurm work, metric, cost, training, baseline comparison or paper claim is authorized. Builder next supplies only a file/symbol plan; Critic and Evaluator remain authored-not-run.

### 2026-08-17 — shared official baseline; restored semantic-indirect DUCA contract

- The unmodified official AdaTAD THUMOS14 run is one shared baseline owned by the ZoomToken
  lead. DUCA must not duplicate released-checkpoint evaluation or source training. Until that
  durable receipt arrives, the official dense number is blank; existing 66.xx and all historical
  DUCA numbers are not substitutes.
- DUCA may continue non-duplicate implementation and PRE_RUN preparation. Its current method
  contract is a scout trained for framewise binary actionness and boundary importance; deterministic
  acquisition derives importance, physical positions, and per-video/window dynamic outer-K from
  these semantic predictions. A direct index policy is an ablation only; fixed K is a
  baseline/control/fallback only. Any future complete run retains the official interval if more
  frequent than five epochs, otherwise saves resumable `.pth` every five epochs without changing
  final/final-EMA selection.

### 2026-08-25 — PJST-D1 contract is frozen; implementation is starting, not evidence

- Target code identity is clean DUCA revision `b2ccfccab5b4912b59954afcc9b0364955327f7c`, where
  Conv3D PatchEmbed mixes each selected-rank frame pair before SingleClock reaches block 0. The dirty
  root `a6bdc084...` is not a production identity.
- A later fresh exact-DUCA completed Pro turn is now authoritative and freezes the derivative-only
  PJST-D1 mechanism plus the fixed-selector representation-attribution estimand. The earlier
  quarantined cross-Project turn remains non-authoritative and must not be rehabilitated.
- The frozen minimal revision keeps the ordinary pair mean and rescales only the
  frame difference by canonical-gap/physical-gap, preserve canonical-uniform byte identity and use
  support intervals as audit metadata only. This isolates the first-mixing physical-Jacobian claim.
- The first causal gate freezes/replays the H65 selector so OFF and PJST receive identical K384
  positions/RGB. End-to-end selector mediation is a later total-effect question, not the same claim.
  A 10,000-sample two-sided 95% percentile interval uses frozen 2.5%/97.5% quantiles; 500/9500 is not
  a 95% central interval. The first implementation package and its one focused correction are now closed
  after a second equivalent deterministic integration defect. Status is
  `designed_frozen / implementation_package_closed / PRE_RUN_blocked / no result`; the science was not
  tested or falsified.

### 2026-08-29 — native-tubelet fixed-budget attribution precedes true dynamic budgeting

- The current attribution gate keeps `K=384` RGB fixed by selecting 192 VideoMAE-native two-frame
  tubelets from the 384-tubelet grid. It compares deterministic uniform tubelet selection against
  a frozen actionness/boundary/temporal-novelty coreset while matching context recycling, physical-time
  reconstruction, detector, optimizer schedule, NMS, split and evaluator.
- This gate tests whether the complete temporal-coreset assembly is viable; it does not redefine
  fixed K as the DUCA paper method. Implementation revision is clean
  `b33391126eac05e3353d322b973dda91741f0732`; focused N16R4 tests, independent static review and both
  execution-and-resume PRE_RUN jobs passed. The matched uniform and coreset arms both completed 60 epochs
  and official evaluation on all 211 validation videos. Their log point estimates are Avg-mAP
  `64.13%/62.81%` and mAP@0.7 `42.45%/40.56%`; the coreset-minus-uniform differences are `-1.32` and
  `-1.89` percentage points, respectively.
- Both jobs then failed at evidence packaging because `post_processing.save_dict=False` left no saved
  predictions for the structured metric writer. No `metrics_epoch59_ema.json`, paired interval or measured
  cost artifact exists. This is a diagnostic negative result for the exact task-state coreset, especially at
  high tIoU, not a training failure, a population-level conclusion or an efficiency result.
- The neutral terminal evidence must return to Pro. Pro independently decides whether an evaluation-only
  sealing pass would change the scientific decision, whether the selector should be revised, whether a true
  dynamic-budget phase remains justified, or whether this candidate should stop. Any dynamic candidate must
  change actually executed heavy clips and match or reduce mean realized VideoMAE compute; padding-only
  nominal K variation is not dynamic computation.

### 2026-08-29 — Pro freezes the window-level real-variable-compute successor

- Fresh exact-DUCA Project Pro returned `PIVOT`: stop the task-state fine-grained coreset, with no score,
  endpoint, maximum-gap or packing adjustment, no retraining, and no isolated rerun of the old control.
- The only successor reallocates compute across windows of the same video. Freeze the H65 Stage-1 epoch-29
  EMA scout; rank windows using mean actionness, the 90th percentile of boundary evidence and the 90th
  percentile of novelty, with the completed coreset's fixed component weights. Sort by demand with earlier
  physical start time as the tie-break: the lowest and highest `floor(W/2)` windows execute 16 and 24 clips;
  an odd video's single median window executes 20 clips.
- Within each budget use deterministic uniform native-tubelet selection. The per-video mean is exactly
  20 clips, 16.67% below the fixed 24-clip control. The three budgets must execute as three real heavy
  paths; padding every row to 24 clips and recording nominal K is not dynamic computation.
- Keep the fixed detector grid, physical-time reconstruction, VideoMAE-S, Adapter, ActionFormer, losses,
  NMS, official evaluator, seed 3407 and 60 epochs/6000 successful updates unchanged. Train one dynamic
  arm only and reuse Job `1260184` epoch-59 EMA as the fixed control.
- The minimal implementation is now frozen on clean branch
  `codex/duca-dynamic-native-tubelet-budget-20260829` at
  `d127c2b2ceea7ff8a6932aa4a1925e1ff86cf610`. It builds split-local per-video window tables from the
  frozen Stage-1 scout, selects 128/160/192 uniform native tubelets, and groups windows so VideoMAE
  actually executes 16/20/24 clips without pre-backbone padding. Existing selected-rank packing and
  post-backbone physical-time residual/reconstruction are preserved; physical coordinates are not newly
  injected into VideoMAE because that would change the matched control representation.
- Short windows can no longer silently reduce a declared budget: table construction and runtime selection
  fail if the valid physical tubelet grid cannot support the assigned clips. Python compilation, launcher
  syntax and pure launcher tests passed; a fresh independent static review passed the exact commit. N16R4
  PRE_RUN Job `1261074` has been submitted and must decide actual split coverage, CUDA behavior and
  checkpoint recovery. No full training job or new performance/cost result exists yet.

### 2026-08-29 — Coverage-v1 is the current single-variable allocation gate

- The latest Pro decision narrows the next attribution to H65 frame allocation. Freeze the clean H65
  `04c35a3b...` priority scores, selected count `K=384`, VideoMAE-S/Adapter/ActionFormer, training schedule,
  NMS, split and official evaluator. Replace only H65 Top-K with a deterministic temporal facility-location
  selector over 96 physical-time anchors. Boundary gradients, feature similarity, feature merging, attention
  bias and dynamic K are excluded.
- The clean implementation is `feature/duca-coverage-only-v1-20260829@a8a0514b00c3528fcf201e6a042b6056429346e1`.
  Focused tests, strict Stage-1 loading, Bash syntax and independent read-only reviews passed. The exact
  snapshot is deployed at `/data/run01/sczc063/yuzibo/duca_coverage_v1_a8a0514b_20260829`.
- The real-training-data unlabeled replay gate must demonstrate actual set intervention, improved anchor
  coverage and smaller maximum physical gaps while retaining at least 90% of H65 normalized priority.
  Passing implementation review is not efficacy evidence.
- The first PRE_RUN submission was rejected before job creation by Slurm `AssocMaxSubmitJobLimit`; there is
  no job ID, training, mAP, interval or cost result. Do not cancel unrelated shared jobs or change the
  experiment to evade the limit. Submit the same frozen PRE_RUN when capacity becomes available.

### 2026-08-30 — Coverage-v1 PRE_RUN exposed and fixed one validation-contract mismatch

- PRE_RUN Job `1261660` was eventually created but stopped before training. The inherited Stage-2 contract
  explicitly uses five-epoch validation only as a learning curve and fixes the scientific result to the
  terminal epoch-59 EMA. The generic formal validator incorrectly accepted only the separate mode that
  records a curve-best validation checkpoint.
- The minimal fix is `feature/duca-coverage-only-v1-20260829@048143124e2a36a76575200ae17d6f42ec79ea3a`.
  Coverage now states `learning_curve_only` and no intermediate checkpoint selection directly in its config;
  the generic validator accepts that consistent pair while preserving the existing official60
  `full_curve_and_best_validation_checkpoint` mode. Model, selector, data, schedule, terminal EMA and official
  evaluation are unchanged. A fresh independent Critic passed the exact clean snapshot.
- The first corrected submission attempt was rejected before job creation by the same Slurm user submit limit.
  At the user's request, one self-terminating process on N16R4 now retries only this exact PRE_RUN after
  60-second real waits and exits after one successful submission. No Coverage training, mAP, interval or cost
  evidence exists yet.

### 2026-08-30 — Coverage-v1 real-data intervention gate did not pass

- The remote submitter created PRE_RUN Job `1261679` for clean commit `048143124e2a36a76575200ae17d6f42ec79ea3a`
  and then exited. The job ran 27 focused tests successfully and replayed 200 unlabeled training samples from
  the frozen Stage-1 epoch-29 EMA. It stopped before smoke or training because the preregistered intervention
  gate returned `passed=false`.
- Valid/unique selection and priority identity passed. Median set change was `0.4805 < 0.80`; relative anchor
  coverage gain was `0.0332 < 0.10`; max-gap p95 changed from control `2` to candidate `8`, yielding a reduction
  statistic of `-3.0 < 0.20`; retained normalized H65 priority ratio was `1.2097 >= 0.90`. Selection used no GT,
  teacher or prediction cache. There is no Coverage mAP, paired interval or cost result.
- Code inspection exposes a scientific-description mismatch: the matched control calls H65
  `budget_calibrated_sampling_rate` systematic sampling, not literal Top-K, and only the scout/priority path is
  frozen while the matched detector backend remains trainable. The facility-location anchors use normalized
  dense candidate indices rather than seconds. These facts must be returned to Pro before changing the selector,
  thresholds, baseline or formal experiment.

### 2026-08-30 — Pro pivots from Coverage-v1 to marginal heavy-compute value

- Pro accepted the Coverage gate and control-identity evidence and returned `PIVOT`. Stop Coverage-v1 matched
  60-epoch training. The concrete 96-anchor facility-location rule failed its own intermediate mechanism gate;
  this does not establish that all coverage, diversity or dynamic-budget methods are ineffective.
- The new question is whether a frozen low-cost H65 scout can predict the TAD marginal loss change from moving a
  window between K256, K384 and K512, and use that prediction to reallocate observations across windows while
  preserving exactly `sum(min(V_i,384))` actual observations per video.
- The only current experiment is a frozen-H65 counterfactual probe on clean base `04c35a3b...`. K384 must reproduce
  the existing H65 epoch-59 EMA checkpoint; K256/K384/K512 must be nested non-contiguous H65-ranked observations
  and execute real different VideoMAE shapes without K512 padding. Detector, scout, VideoMAE, head, loss, NMS,
  split and evaluator remain frozen.
- Use a deterministic 160/40 controller fit/holdout split inside the 200-video training side. Train-side GT may
  generate detached marginal-utility labels and the holdout oracle only; official test GT is never visible to the
  learned allocator. The 40 videos are held out from utility-head fitting, not from the already trained H65 detector.
- Official test is permitted only after H65 parity, oracle-headroom and predictability gates all pass. Intermediate
  threshold cases remain unresolved and return to Pro; Codex may not relax gates or add a new route.
- Full Pro report: `.cvpr-pro-lab/pro-reviews/runs/duca-marginal-v1-pivot-user-supplied/visible-report.md`.

### 2026-08-31 — Marginal short-window contract implemented and under PRE_RUN

- Pro returned `REVISE`, not a route change. A requested tier uses `min(V_i,K)` actual observations; a nonbaseline
  tier with the same actual cost as K384 collapses to K384 and aliases its loss, prediction, positions and execution
  without another detector forward. Historical K384 always executes 384 slots; a distinct nonbaseline tier executes
  only `16 * ceil(actual/16)` slots with less than one packet of trailing padding.
- The public implementation is branch `feature/duca-marginal-budget-v1-20260830`. The independently reviewed model
  implementation is `be5bb803...`; the latest commit is
  `f87555f7da362fe1a20d4ca08f7a68c975ed8280`:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f87555f7da362fe1a20d4ca08f7a68c975ed8280`.
  The successor changes only one focused-test argument from `1.0` to the Pro-frozen changed-window bound `0.5`.
- Job `1262073` was a pre-model `/bin/sh` launch failure. Job `1262075` then exposed the inconsistent test bound; it
  did not run the counterfactual probe. The same PRE_RUN on clean commit `f87555f7...` is Job `1262076`. Dependent Job
  `1262077` verifies that exact output root's `PRE_RUN_PASS` before running the four already authorized stages.
- Job `1262076` completed `PRE_RUN_PASS` over all 200 training-side videos and 720 windows. It verified the 160/40
  split, short-window inclusion, 47 collapsed aliases, exact per-video K384 target, complete K384 tensor parity,
  observed packetized execution classes, frozen detector/Scout gradients, and absence of training/evaluator/test use.
  This is implementation eligibility only. Until Job `1262077` completes, there is no admitted headroom,
  predictability, mAP or compute result.

### 2026-08-31 — Marginal oracle diagnostic lands in the preregistered gray zone

- Job `1262077` sealed K384, K256 and K512 producer artifacts, then failed only in `summarize` because a newline-text
  holdout block list was passed to a JSON-reading evaluator. It produced no `probe_result.json`; do not treat that
  failed job as science evidence and do not rerun the completed producers.
- The summary-only fix is public commit `f67d96fdf68a295eaa7f678f3dfc125530828889` on
  `feature/duca-marginal-budget-v1-20260830`:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f67d96fdf68a295eaa7f678f3dfc125530828889`.
  Job `1262098` reran only the current-commit PRE_RUN identity check and summary, reusing the immutable producer
  artifacts. Producer source remains `f87555f7...`; summary/pre-run source is `f67d96fd...`; config, checkpoint,
  annotation, class map, pretrain and their hashes match. Preserve this provenance in every downstream review.
- On the 40-video/124-window training-side utility holdout, Fixed-H65-384 obtained Avg-mAP `88.131197%` and
  mAP@0.7 `76.270583%`; the true-utility equal-budget oracle obtained `88.856786%` and `76.999587%`. The gain is
  `+0.725589/+0.729004` percentage points, with 102/11/11 windows at K384/K256/K512 and zero budget error.
- This is neither the preregistered strong headroom pass (`+0.8/+1.0`) nor the no-headroom region (`<+0.3/<+0.5`).
  The runner correctly stopped before utility-head fitting. No predictability gate, learned policy, official test,
  uncertainty interval or end-to-end cost result exists. Return the exact gray-zone evidence to Pro; Codex must not
  relax thresholds, infer a route, or launch the next experiment.

### 2026-08-31 — Pro freezes one read-only cap-release falsifier

- Pro returned `REVISE` after receiving the latest repository, branch, exact `f67d96fd...` commit, key-file GitHub
  links and all raw artifacts. It admitted the gray-zone point estimate only as a training-side holdout mechanism
  diagnostic and accepted the explicit `f87555f7` producer / `f67d96fd` summary provenance.
- The only allowed intervention is `max_changed_fraction: 0.5 -> 1.0` on the same sealed K256/K384/K512 artifacts,
  with the same true counterfactual utilities, exact per-video actual-observation budget, tie-break, NMS and evaluator.
  No new producer forward, budget tier, model training, utility-head fitting or official-test access is allowed.
- The public implementation is
  `feature/duca-marginal-cap-release-falsifier-v1-20260831@d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`.
  Linux focused tests report `14 passed`; an independent Critic returned PASS. Evaluator Job `1262117` is the sole
  read-only execution. Until its terminal result, do not infer cap-release headroom or start predictor work.

### 2026-08-31 — Cap release reduces rather than enlarges oracle headroom

- Evaluator Job `1262117` completed successfully and ran only the read-only CPU summary over the same sealed
  K256/K384/K512 artifacts. It executed no detector/Scout forward, training, utility-head fitting or official test.
- It reproduced the fixed and 50%-capped results exactly. Releasing the cap changed the allocation from
  K256/K384/K512 `11/102/11` to `17/90/17`, changed 11 videos and 34 windows, and preserved the exact total actual
  observation cost `47110`.
- The released oracle obtained Avg-mAP `88.558507%` and mAP@0.7 `76.720863%`: only
  `+0.427310/+0.450280` percentage points over fixed K384 and `-0.298279/-0.278724` points below the capped oracle.
  Both preregistered `+0.8/+1.0` point gates failed, so no paired bootstrap was run and the frozen rule stops the
  current Marginal-v1 mechanism.
- This is a training-side 40-video holdout mechanism diagnostic, not official validation/test evidence or a general
  rejection of dynamic budgeting. Return the terminal result to Pro with the latest public implementation
  `feature/duca-marginal-cap-release-falsifier-v1-20260831@d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`;
  Codex must not choose or launch a replacement route.

### 2026-08-31 — Pro pivots from additive window loss to one joint-mAP neighborhood falsifier

- The exact-Project Pro turn was bound to the latest public branch, commit `d2fad7c0...`, permanent runner/allocator/test
  URLs, terminal JSON and sealed producer artifacts. It returned `PIVOT` and stopped the current three-tier additive
  Marginal-v1 mechanism, while explicitly leaving broader task-aware dynamic computation unresolved.
- The key supported failure is objective mismatch: releasing the cap enlarges the feasible set and improves the
  allocator's additive window-loss objective, yet worsens final mAP after overlapping-window merge, Soft-NMS,
  confidence ranking and AP aggregation. Coarse tiers and paired exact-budget moves may amplify this mismatch; small
  intrinsic headroom remains an alternative explanation rather than a settled cause.
- The sole task is a read-only enumeration over the capped-to-released difference neighborhood. Derive the 12 changed
  windows and every balanced per-video subset from data, then enumerate the Cartesian product. Current sealed inputs
  must yield 96 unique exact-budget states without hard-coding arbitrary pairings, including for
  `video_validation_0000419`.
- Reuse the same 40-video holdout predictions, NMS and evaluator. Reproduce fixed/capped/released within `1e-6 pp`,
  preserve each video's actual observation target and global cost `47110`, and execute no model forward, training,
  utility-head fitting, official test or bootstrap.
- Continue only if at least one state simultaneously reaches `+0.8 pp` Avg-mAP and `+1.0 pp` mAP@0.7 over fixed.
  Even then, return the result to Pro before any predictor. If none passes, stop the joint-utility repair of this
  difference neighborhood; Codex must not choose a broader successor.

### 2026-08-31 — Joint-mAP neighborhood fails the frozen continue gate

- The latest public implementation is
  `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831@46812facc8773d9b4a9c21833cbe397c8aaa5a2d`:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d`.
  The runner and focused test changed; `opentad/models/duca/dynamic_budget.py` is byte-identical to its parent.
- Independent review passed. Slurm Job `1262121` completed all 96 exact-budget CPU evaluator states and reproduced
  fixed/capped/released within `0.0 pp`. The scheduler required one GPU allocation, but the job hid CUDA and ran the
  evaluator on CPU; no detector/Scout forward, training, gradient, utility-head fitting, official test or bootstrap ran.
- No state passed the joint `+0.8 pp` Avg-mAP / `+1.0 pp` mAP@0.7 gate. The best joint-margin state was
  `+0.553972/+0.933234`; the Avg-best state was `+0.732990/+0.479291`; the @0.7-best state was
  `+0.548669/+0.933539` relative to fixed K384.
- None of the eight minimal legal transfers improved both gate metrics. Under the frozen classification this supports
  single-item misranking as the primary failure within this neighborhood, not a window-interaction reversal.
- Stop the joint-utility repair of this cap-release difference neighborhood. The result is a same-holdout metric-oracle
  diagnostic, not a deployable policy or paper-ready performance result. Return the raw result, provenance and latest
  permanent GitHub URLs to Pro; Codex must not choose a successor route.

### 2026-08-31 — Pro STOP closes the existing additive Marginal-v1

- The exact DUCA Project turn completed with verified `Pro`, nonce and latest public
  `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831@46812facc8773d9b4a9c21833cbe397c8aaa5a2d`
  GitHub bindings. The visible report is
  `.cvpr-pro-lab/pro-reviews/runs/duca-marginal-cap-release-neighborhood-terminal-compact-v002/visible-report.md`.
- Pro returned `STOP`. The supported paper-language explanation is that additive window-level counterfactual detector
  loss is not a sufficient ranking statistic for video-level joint detection utility in this neighborhood. Failure is
  already present at the minimal equal-cost transfer level; no minimal transfer improves both Avg-mAP and mAP@0.7.
- Stop the complete existing Marginal-v1 mechanism: the same H65 priority sequence, K256/K384/K512 tiers,
  per-window counterfactual detector-loss utility, exact per-video additive allocation, and any cap, pairing, tie-break
  or same-neighborhood state repair. There is no further Builder, Critic, PRE_RUN, evaluator, bootstrap, utility-head or
  official-test task for this mechanism.
- Do not generalize the stop to the three budget values themselves, the H65 priority sequence, every three-tier
  allocation, or task-aware dynamic computation in general. Those directions are merely unresolved and are not
  automatically authorized. Any future work requires a new Pro hypothesis and task, not a Marginal-v1 recovery.
- The terminal JSON SHA-256 was independently reverified as
  `a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`. The implementation branch is now read-only
  negative evidence; preserve the result for failure-mechanism analysis or supplementary material.

### 2026-08-31 — Project-level Pro PIVOT freezes one whole-video action-space falsifier

- A fresh exact DUCA Project turn used verified browser `Pro`, nonce
  `DUCA-PROJECT-LEVEL-AFTER-MARGINAL-STOP-v001-20260831`, and the latest public
  `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831@46812facc8773d9b4a9c21833cbe397c8aaa5a2d`
  repository, branch, commit, runner, allocator and test URLs. Pro returned `PIVOT` rather than project-level stop.
- The sole remaining hypothesis is that per-window mixed budgets are misaligned with video-level merged proposals. Give
  every window of one donor video K256, every window of one recipient video K512, keep all other videos K384, and move
  compute only between videos. This does not call or revise the stopped Marginal allocator.
- Generate all ordered donor-recipient candidates before reading labels or metrics. Both changed videos need an actual
  non-baseline window; retain only candidates with total actual observation cost `<=47110`, using
  `min(valid_observations,K)` rather than requested K or padding.
- Reuse the same sealed K256/K384/K512 predictions, H65 priority sequence, coordinate mapping, Soft-NMS and evaluator.
  Run no model forward, training, gradient, bootstrap or official test.
- Continue only if at least one legal candidate simultaneously reaches `+0.8 pp` Avg-mAP and `+1.0 pp` mAP@0.7 over
  fixed K384. If `passing_candidate_count=0`, the frozen consequence is project-level STOP within current THUMOS14,
  H65 priority-sequence, three-tier real-observation action space and resource boundary. Do not extend the search.
- Builder may add only an independent runner, focused test and, if required, a thin CPU Slurm entry on new branch
  `feature/duca-whole-video-consistent-budget-falsifier-v1-20260831`; all existing model and evidence files stay unchanged.

### 2026-08-31 — Whole-video falsifier implementation is public and independently reviewed

- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>. Actual remote branch:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>.
  Exact implementation commit:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>.
- The public implementation is located by the permanent
  [runner](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py),
  [focused test](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py),
  and unchanged [three-tier allocator](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py).
  Candidate generation uses only sample/video identity and sealed actual-cost accounting, writes the complete ordered
  manifest, and only then loads terminal metrics, annotation and the evaluator.
- The first PRE_RUN, Job `1262147`, stopped before candidate performance because the runner sorted sealed proposal rows
  lexicographically and therefore changed score-tie input order for the unchanged Soft-NMS. This was a deterministic
  evidence-replay defect, not evidence about the mechanism. Commit `33e4ed137c33eef07f0452b44506a6993bdf7535`
  changes only row-order preservation and its regression test. Twenty-eight focused tests and a fresh independent Critic
  passed on the exact clean commit.
- Corrected PRE_RUN Job `1262161` passed on 40 videos and 124 windows: fixed cost `47110`, 1560 ordered pairs, 704 legal
  candidates, 1330 actual interventions, and `0.0 pp` reproduction error for fixed, capped and released anchors. Receipt
  SHA-256 is `734b178bfb7bdaa05879edfeb8e129263c9e2c4cf80867415eec6d41df3c12a3`; candidate-manifest SHA-256 is
  `c4a02c47be1ab7e73dc81c18b32635d3347ece2f0d26b0d96de3ec4af053f69a`.
- Formal Job `1262162` ended `NODE_FAIL` after `500/704` candidates because node `g0022` went down. It wrote neither the
  terminal `whole_video_consistent_budget_result.json` nor a runner failure receipt, so it carries no performance verdict.
  Exact same-task infrastructure recovery Job `1262190` reuses the same clean snapshot, script, candidate manifest, sealed
  predictions, evaluator, gates and output directory. It is the sole active recovery; do not submit a third job.

### 2026-08-31 — Whole-video falsifier completes with no passing candidate

- Exact same-task recovery Job `1262190` completed all `704/704` legal candidates with Slurm `COMPLETED 0:0` on the clean
  public implementation
  `feature/duca-whole-video-consistent-budget-falsifier-v1-20260831@33e4ed137c33eef07f0452b44506a6993bdf7535`.
  No runner failure receipt exists. The terminal result is
  `/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_consistent_budget_result.json`,
  SHA-256 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`.
- The fixed K384 anchor is `88.1312%` Avg-mAP and `76.2706%` mAP@0.7 at actual observation cost `47110`; all six metric
  reproduction errors are `0.0 pp`. The evaluator was called 705 times: once for fixed K384 and once per legal candidate.
- The best Avg-mAP candidate improves Avg-mAP by `+0.6942 pp` but changes mAP@0.7 by `-0.0436 pp` at cost `46982`.
  The best mAP@0.7 candidate changes Avg-mAP/mAP@0.7 by `-0.2359/+0.4970 pp` at cost `46854`. The best joint-gate
  candidate changes them by `+0.1474/+0.4898 pp` at cost `45830`; its joint margin is still `-0.6526 pp`.
- No candidate simultaneously reaches the frozen `+0.8 pp` Avg-mAP and `+1.0 pp` mAP@0.7 gate;
  `passing_candidate_count=0`. This triggers the previously frozen stop boundary for the present THUMOS14 controller
  holdout, H65 priority sequence, K256/K384/K512 real-observation action space and resource scope. Do not add videos,
  combine transfers, alter tiers, lower the gate, train a controller, bootstrap after selection or access official test.
- This is a training-side development-holdout privileged-oracle falsifier, not a deployable method or paper performance
  result. It ran no detector/Scout forward, training, gradient, bootstrap, official validation or official test and has no
  uncertainty interval. The neutral terminal evidence and latest permanent GitHub links were then returned to Pro; the
  completed independent decision is recorded below. Codex did not preselect a successor.

### 2026-08-31 — Pro STOP closes the current three-tier observation-transfer route

- The completed exact-Project Pro turn was bound to the latest pushed repository, actual branch, exact commit
  `33e4ed137c33eef07f0452b44506a6993bdf7535`, permanent runner/test/allocator URLs, terminal JSON path and SHA-256.
- Pro returned `STOP`. Close Marginal-v1, cap-release, the 96-state neighborhood and the whole-video donor-recipient
  extension as read-only negative evidence. Do not enlarge the search, alter K256/K384/K512 or the gate, train a
  utility/controller, add bootstrap after oracle selection, or access official validation/test for this route.
- The supported statement is scoped: redistributing observations from the same frozen H65 nested priority sequence across
  windows or videos did not expose preregistered joint Avg-mAP/high-tIoU development headroom. Avg-mAP and mAP@0.7 optima
  were different states, and consistent whole-video budgets did not rescue the action space.
- The strongest untested explanation is cross-budget representation mismatch between the H65 priority sequence and a
  detector trained only at K384. It remains a hypothesis; testing it would require a new training mechanism or action
  space and is not a repair task.
- There is no current Builder, Critic, Evaluator or experiment task. Reopening requires a clearly distinct mechanism outside
  this boundary to first show preregistered joint oracle headroom on an independent training-side development split at
  matched real compute, without post-hoc candidate or threshold changes.

### 2026-08-31 — REVISE opens one detector-adaptation hypothesis outside the stopped route

- A user-supplied current adjudication keeps the prior `STOP` for frozen-detector observation transfer, but defines one new
  falsifiable question: does training the same detector jointly at nested K256/K384/K512 restore per-budget quality and
  matched-cost whole-video oracle headroom?
- The first test preserves the existing nested K positions. It compares K384-only training with multi-budget adaptation and
  changes no selector, Scout, detector architecture, losses, physical-time mapping, NMS, evaluator or cost definition.
  An earlier attached proposal to change to budget-native per-K sampling is superseded for this first test.
- Base the model on clean H65 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`. Reuse from `33e4ed...` only verified
  variable-length execution, packet alignment, actual-observation accounting, K384 parity, whole-video evaluation and
  preservation of producer order; do not use the diagnostic branch as the model base.
- The multi-budget arm starts from nominal K probabilities `0.25/0.50/0.25`, calibrated against actual observations after
  short-window collapse. Both arms must match start checkpoint, successful updates, optimizer, LR schedule, seed, trainable
  parameter set and terminal EMA rule.
- Required gates are K384 safety (`>=-0.2 pp` Avg-mAP and mAP@0.7 versus matched control) and a disjoint train-side
  matched-cost oracle (`>=+0.8 pp` Avg-mAP, `>=+1.0 pp` mAP@0.7, no higher actual cost). Per-budget proposal and boundary
  diagnostics are mandatory.
- Exact matched training duration/update count and the train-side development video IDs not used for learning or rule
  selection remain unspecified. Until Pro freezes both, status is `designed`: no Builder branch, PRE_RUN, Slurm or result.

### 2026-08-31 — Formal comparison requires full training and complete held-out evaluation

- A new human constraint requires both matched arms to train on the complete frozen THUMOS14 training split. Training subsets,
  the prior 40-video controller holdout and other pilots may support pre-formal diagnosis, but cannot become the paper's main
  comparison or replace a new matched full training run after the design is frozen.
- Final comparable evidence must use the complete official held-out evaluation split with identical annotation semantics,
  class mapping, evaluator and Soft-NMS. Held-out evaluation is evaluation-only: it cannot select checkpoints, thresholds,
  rules, routes or hyperparameters, and repeated result peeking cannot drive revisions.
- The repository records two incompatible naming/video-count conventions: OpenTAD/DUCA `training/validation` with 211 evaluated
  videos, and ActionFormer official `validation/test` with historical records of 212 evaluated videos. Pro must freeze exact
  config subset names and complete video-ID sets; Codex must not silently equate them.
- The current in-flight Pro turn predates this constraint. Do not interrupt, follow up or resubmit it. After its terminal result,
  audit whether it separates development diagnosis from full training and complete held-out evaluation. If not, request a new
  fresh Pro adjudication before Builder work.

### 2026-08-31 — Pro v001 completed, but its 160/40 protocol is not executable

- The exact-Project Pro turn completed with verified nonce, conversation, Project and Pro model selection. It chose `CONTINUE`,
  preserved nested K256/K384/K512 positions, and froze matched Stage-2 training from the H65 Stage-1 EMA for exactly 6,000
  successful updates per arm, with actual-observation-calibrated budget probabilities and terminal EMA evaluation.
- The report was prompted before the human full-data constraint. It partitions the 200 training-side videos into 160 training and
  40 development videos and explicitly avoids official test. That protocol may support a mechanism-development experiment but
  cannot satisfy the required formal comparison.
- No Builder, PRE_RUN or training is authorized from this report. A new independent Pro turn must freeze the complete training
  identity, complete official held-out identity, 211/212 discrepancy, one-shot evaluation boundary and any diagnostic-to-formal
  sequence. Do not append a follow-up to the completed conversation.

### 2026-08-31 — Verified Pro full-data protocol replaces 160/40 and freezes identity audit first

- The exact-Project Pro turn `DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831` completed with verified Project,
  conversation, nonce and Pro model selection. It chose `REVISE`, retained the single-variable nested-K detector-adaptation
  question, and revoked the 160/40 formal split, labeled training-side mAP gates and the old 40-video oracle.
- Formal training uses every annotation ID with subset `training`, expected to be exactly 200 with identical loader and physical
  coverage. Formal held-out evaluation uses subset `validation`, but 211 versus 212 remains a factual identity question; Codex
  must compare annotation, loader, physical files, evaluator, historical 211 IDs and a source-backed ActionFormer 212 manifest.
- The only current task is the read-only split identity audit on base `04c35a3b...`. Before it passes and returns to Pro, do not
  create the model Builder branch, load a checkpoint, submit PRE_RUN/GPU/training, generate held-out predictions or compute mAP.
- After a future Pro data-admission decision, both arms use the full 200-video training domain for 6,000 successful updates from
  the same Stage-1 EMA. Predictions are sealed before one held-out evaluation; the comparison uses the same unlabeled fixed
  mixed-budget manifest and a 10,000-replicate paired whole-video bootstrap.
- The contemporaneous user summary referring to `research_project_analysis.md` is not backed by a file in the current repository.
  Its progressive-unfreezing, five-budget and ActivityNet proposals are not authorized by this task and do not supersede Pro.

### 2026-08-31 — External irregular-time sampling proposal recorded without route authorization

- A user-supplied proposal has been preserved verbatim and normalized into four separable hypothesis families: native consecutive
  tubelet acquisition, explicit physical-time encoding, sparse-to-dense temporal reconstruction and end-to-end optimization
  stability. Its proposed combined system additionally uses a 144/48 dual-stream allocation, continuous-time rotary position
  encoding, adaptive Gaussian temporal splatting and a three-stage Gumbel/distillation curriculum.
- Existing evidence makes the tubelet/time/reconstruction questions scientifically relevant, but does not establish any of them
  as the root cause of the H65--end-to-end gap. Native-tubelet uniform/coreset results were 64.13/62.81 Avg-mAP; they do not
  validate the proposed dual-stream allocation. The proposal's +0.8--1.5 pp, >=64.5 and five-budget expectations are targets.
- Historical anti-duplication audit: native consecutive tubelets and CONTIG bundles have already been implemented and formally
  trained, but no matched experiment has isolated pairing continuity alone; physical-time representations have been tested
  repeatedly, with mixed results and a partially positive RankPack/TrueTime comparison (+0.6208 Avg-mAP); hidden-linear
  sparse-to-dense reconstruction has implementation/CUDA evidence but no formal kernel-versus-boundary comparison; curricula,
  homotopy and CellCF distillation have been attempted, but the apparent 30+60 gain is confounded by 90 versus 60 total epochs.
- Do not rerun those historical bundles. The genuinely unresolved tests are respectively a pairing-only control, a clearly novel
  time encoding compared with TrueTime, a frozen-input reconstruction-kernel comparison, and a total-update-matched curriculum
  or distillation comparison. None is authorized before the current data-identity and Pro-admission sequence completes.
- Do not implement the combined proposal or let it supersede the verified Pro sequence. The only current task remains the
  read-only 211/212 split identity audit. After data admission, the first frozen model experiment remains the single-variable
  K384-only versus nested K256/K384/K512 detector-adaptation comparison.
- Structured evidence audit:
  `research-wiki/sources/2026-08-31-duca-irregular-temporal-sampling-external-proposal.md`.
- Complete raw text:
  `docs/methods/reviews/2026-08-31-duca-irregular-temporal-sampling-external-proposal-raw.txt`.

### 2026-08-31 — Comprehensive GitHub/Wiki Pro review reaffirms identity audit before model work

- A fresh exact-Project Pro turn read the complete public Wiki, Gemini review and the key exact GitHub implementation lineage. It
  returned `REVISE` with the required nonce and terminal marker. Full text:
  `research-wiki/sources/2026-08-31-pro-github-wiki-comprehensive-review-v002.md`.
- H65 `04c35a3b...` remains the only clean model mainline. The Wiki synchronization revision is documentation-only, and
  `33e4ed...` remains a read-only negative diagnostic rather than a new-model parent.
- The only current task is `feature/duca-full-data-identity-audit-v1-20260831`: materialize literal annotation, physical-media,
  loader, evaluator, historical 211-prediction and ActionFormer 212-source ID sets; explain every set difference; return exactly
  `DATA_IDENTITY_PASS` or `DATA_IDENTITY_BLOCKED` after independent Critic and CPU Evaluator review.
- No model, checkpoint, PRE_RUN, GPU, training, held-out prediction or mAP is authorized before that result returns to Pro.
- Conditional on data admission, the sole model experiment is fixed K384 versus matched K256/K384/K512 training exposure using
  all 200 training videos, seeds 3407/3408/3409, 6,000 successful updates, sealed predictions, one complete held-out evaluation and
  10,000 paired whole-video bootstrap replicates. It does not add budget embeddings, distillation, a new selector or another route.

### 2026-08-31 — Full-data identity audit returns `DATA_IDENTITY_PASS_211`

- A newer exact-Project Pro report, nonce `DUCA-COMPREHENSIVE-ROUTE-INTEGRATION-v001-20260831`, independently reviewed the
  historical code and evidence, renamed the conditional experiment to H65 system multi-budget exposure adaptation, and kept the
  read-only full-data identity audit as the sole current task. Complete text:
  `research-wiki/sources/2026-08-31-pro-duca-comprehensive-route-integration-v001.md`.
- Builder implementation is the clean direct child `fdd2bcdddf3f23f3546244adf90c4427ed022837` of H65
  `04c35a3b76897e6c1569eeede41ed3aecaf7f854`, on branch
  `feature/duca-full-data-identity-audit-v1-20260831`. Only the audit tool and one focused test were added. Local checks produced
  29 passes; the N16R4 focused run produced 6 passes. An independent read-only Critic returned `PASS`.
- The N16R4 CPU Evaluator used no GPU, model, checkpoint, held-out label/segment, prediction payload or metric. It verified all
  411 expected canonical videos with `ffprobe`: no missing file, broken symlink, duplicate ID, unassigned ID or decode failure.
- Training annotation, formal loader replay and canonical physical media are the same 200 IDs; all three manifests have SHA-256
  `5b11e290eb24c93c79f23cb1aecc8b85be4c13b47d7cf3b35e30601c1663f4c0`. Train-held-out intersection is empty.
- OpenTAD held-out annotation, loader, physical media, evaluator and the historical PJST prediction-key set are the same 211 IDs;
  all five manifests have SHA-256 `5f9adf639fbcff869075ac78f6aa26d9da14986199a7d5b2181127769600746e`.
- The original ActionFormer annotation has 212 literal `Test` IDs. Its only right-only ID versus OpenTAD is
  `video_test_0000270`. OpenTAD `tools/prepare_data/thumos/README.md:11` states that this video is removed for wrong annotations
  and `video_test_0001292` is removed for empty annotations. The latter is not in ActionFormer's 212 annotation set; it is an
  extra physical/feature file, not an evaluated video.
- The effective report is under remote `result_v2` and has SHA-256
  `d7251c11935644cf8661e6bfdcfb857e29d2357cb894b7de9d8b2bd7eaf6f1ab`. Full report and literal manifests are preserved at
  `research-wiki/sources/2026-08-31-duca-full-data-identity-audit-fdd2bcdd/`.
- The first evaluator invocation passed lowercase `test` instead of the source's case-sensitive literal `Test`, yielding an empty
  ActionFormer set and an invocation-layer `BLOCK`. That output is preserved remotely. A single argument-only correction produced
  the effective `PASS`; code, data and source files were unchanged. Do not report the first result as a dataset conflict.
- Current conclusion is `DATA_IDENTITY_PASS_211`, but model work remains blocked until Pro explicitly admits this evidence. The two
  latest Pro reports differ only on the later seed launch order: immediate three-seed execution versus seed 3407 followed by
  3408/3409 only after all gates pass. Codex does not resolve that downstream difference.

### 2026-08-31 — Pro admits 200/211 and unlocks one H65 system-level experiment

- Exact Project Pro conversation `6a956592-da38-83e9-b50c-fd3906c0ec41`, nonce
  `DUCA-FULL-DATA-IDENTITY-ADMISSION-v001-20260831`, returned `CONTINUE` and formally admitted the complete 200-video
  `training` set plus the complete 211-video OpenTAD `validation` held-out set. Full report:
  `research-wiki/sources/2026-08-31-pro-duca-full-data-identity-admission-v001.md`.
- ActionFormer 212 remains source-tracing only; `video_test_0000270` is excluded for wrong annotations and
  `video_test_0001292` is not an annotation evaluation member. The 211 held-out set cannot be used for checkpoint, threshold,
  budget mixture, rule, arm, seed or route selection.
- The seed conflict is resolved as blind sequential execution of all three seeds `3407 → 3408 → 3409`. No held-out metric may be
  read until all six Control/Candidate training units and all complete prediction views are sealed.
- The only current task is the Builder for `feature/duca-h65-system-multibudget-exposure-v1-20260831`, directly from H65
  `04c35a3b...`. The sole variable is Stage-2 budget exposure: Control always K384; Candidate uses the existing nested
  K256/K384/K512 positions with actual-observation-matched probabilities. The full H65 Stage-2 trainable set remains unchanged.
- No model code, PRE_RUN, training, held-out prediction, mAP or cost result existed at this admission point. Builder must return one
  exact clean revision to an independent Critic before any Evaluator execution.

### 2026-08-31 — Gemini 3.7 Flash High independently reviews the admitted Builder task

- A fresh `agy` CLI read-only consultation inspected the exact Wiki commit `a6c246a6...`, H65 base `04c35a3b...`, data audit
  `fdd2bcdd...`, frozen-detector falsifier `33e4ed...`, and the still-unimplemented public Builder branch. Complete response:
  `research-wiki/sources/2026-08-31-agy-gemini-duca-post-admission-optimization-v001.md`.
- It supports the current single-variable experiment and highlights budget-homogeneous logical updates, an independent budget RNG,
  short-window actual-cost collapse, true variable-length packet execution, K384 parity, and successful-update/EMA clock integrity as
  the highest-value Builder checks.
- It has advisory authority only. Its proposed `0.1%` tolerance, exact changed-file list, later controller ablations and spatial-route
  fallback are not current tasks. Its claims that PASS proves a particular mechanism or FAIL proves an intrinsic discontinuity and
  closes the project are stronger than the evidence and must not be adopted. Pro remains the only scientific decision maker.
- There is still no multi-budget model commit, PRE_RUN, training, held-out prediction, performance or cost result.

### 2026-08-31 — H65 system multi-budget Builder is implemented and awaits independent Critic

- The exact public implementation is
  `feature/duca-h65-system-multibudget-exposure-v1-20260831@0d67d49c2fc4a5f50aa784f7809c0dd936492109`,
  whose sole parent is H65 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`.
- Control retains the exact historical K384 path. Candidate constructs nested K256/K384/K512 exposure from the same H65 priority
  sequence, uses one homogeneous budget per successful optimizer update without consuming the data RNG, executes actual
  VideoMAE packets, removes padded terminal-packet features, and restores the frozen 384-point detector grid.
- The 6,000-update occurrence plan is `1454/3000/1546`; full-training actual-cost calibration gives probabilities
  `0.24235161911751213/0.5/0.25764838088248787`. These are label-free preparation facts, not performance evidence.
- Held-out inference uses a 211-video annotation stripped of action labels and temporal segments. Nine prediction/cost views must be
  sealed before the one-time evaluator parses the complete held-out annotation once and applies shared 10,000-replicate whole-video
  bootstrap indices.
- Cost sealing now distinguishes actual observations and packet slots, realized data-consumer wait, model/window processing,
  per-video Soft-NMS, full-population wall time, Scout/VideoMAE/detector time and GPU peak memory. It does not infer savings from K
  or padding.
- Local diff/compile/launcher checks and N16R4 focused tests pass (`25 passed`). This advances the task only to implemented and
  locally tested. No PRE_RUN, full training, prediction, mAP, interval or efficiency result exists. A fresh independent Critic must
  review the exact commit before Evaluator work.

### 2026-08-31 — Exact multi-budget implementation passes independent Critic

- A fresh, read-only Critic reviewed exact clean commit `0d67d49c2fc4a5f50aa784f7809c0dd936492109` and returned `PASS`.
  It confirmed the frozen single variable, 6,000-successful-update clock, K384 legacy execution, nested budget sets, real packet
  execution and padding trim, complete 200/211 boundary, nine sealed views, one-time annotation parse, official evaluator, shared
  paired bootstrap and measured-cost boundary.
- The Critic could not import local Windows PyTorch because of the existing `c10.dll` initialization failure; this does not negate
  the N16R4 Linux `25 passed` result. The review gate is now closed, so the next action is PRE_RUN on the same exact commit.
- This is code-admission evidence only. There is still no full training, held-out prediction, mAP, interval or efficiency result.

### 2026-08-31 — Exact PRE_RUN submitted

- Exact commit `0d67d49c2fc4a5f50aa784f7809c0dd936492109` is deployed as the clean N16R4 snapshot
  `/data/run01/sczc063/yuzibo/duca_h65_multibudget_0d67d49c_20260831`.
- The sole PRE_RUN is Slurm Job `1262690`, with immutable output root
  `/data/run01/sczc063/yuzibo/duca_h65_multibudget_prerun_0d67d49c_20260831`.
- It may only establish runtime eligibility through preparation, focused tests, compilation, four successful updates and strict
  checkpoint/probe validation. It does not access held-out metrics and is not a performance or efficiency result.

### 2026-08-31 — First PRE_RUN exposes short-window true-time metadata defect; one minimal recovery submitted

- Job `1262690` failed during the first real training forward, before any successful optimizer update, because a short-window K384
  row carried inactive `-1` detector padding into `selected_axis_to_true_time_dense_index`. `TrueTimeMap` correctly rejected those
  positions as outside `valid_len`. No checkpoint, complete training, held-out prediction, mAP, interval or cost result was produced.
- Exact correction `409f370a7ed14e7077bc87138196ab6abe459f99` has sole parent `0d67d49c...` and changes only selector metadata plus one
  focused regression. It filters true-time/remap metadata with the existing detector mask; the 384-point execution tensor and mask,
  acquisition sets, model/config/data/loss/NMS/evaluator/statistic surfaces remain unchanged.
- Exact N16R4 snapshot `/data/run01/sczc063/yuzibo/duca_h65_multibudget_409f370a_20260831` returns `26 passed`; a fresh
  independent Critic returned `PASS`.
- The only corrected PRE_RUN is Job `1262693`, output root
  `/data/run01/sczc063/yuzibo/duca_h65_multibudget_prerun_409f370a_20260831`. Six complete training units remain blocked until
  this same job passes all four successful updates and checkpoint/probe validation.

### 2026-08-31 — Corrected PRE_RUN passes; six complete training units are deployed blind

- Job `1262693` completed `0:0`: four attempted batches produced four finite-loss, finite-gradient and successful optimizer updates;
  optimizer, scheduler, EMA and DUCA schedule counters all equal four; the smoke checkpoint exists and the strict validator emitted
  `PRE_RUN_PASS`.
- Full-training jobs are Control/Candidate `1262696/1262697` for seed 3407, `1262698/1262699` for seed 3408, and
  `1262700/1262701` for seed 3409. The latter pairs use strict `afterok` dependencies on the preceding seed pair.
- All jobs bind exact commit `409f370a...`, the admitted complete 200-video training population, the same Stage-1 epoch-29 EMA,
  6,000 successful updates and terminal epoch-59 EMA rule. No held-out label, metric or intermediate performance has been read.
- The next admissible event is a training terminal. Predictions remain blocked until all six training units complete successfully.

### 2026-08-31 — Legacy training binder mismatch is removed; authoritative training DAG is replaced

- Seed-3407 Jobs `1262696/1262697` exited before model/data training because an empty formal protocol was incorrectly routed into
  the unrelated legacy `duca_p0_training` binder. They produced no successful update, checkpoint, prediction or held-out metric.
  Dependent Jobs `1262698`–`1262701` were cancelled without starting. This is execution-binding evidence, not a scientific result.
- Exact recovery `2b3b3243066a89e5a4be5acdb178c318fbeceac0` disables only that inapplicable legacy binder while preserving the 6,000
  successful-update budget, update audit, terminal EMA and held-out sealing. N16R4 returns `26 passed`; a fresh independent Critic
  returns `PASS`.
- PRE_RUN `1262715` completed `0:0`: all four updates have finite loss and gradient; optimizer, scheduler, EMA and DUCA counters
  are four; the smoke checkpoint exists. This restores runtime admission only.
- The current blind full-training chain is Control/Candidate `1262719/1262720` for seed 3407, `1262721/1262722` for seed 3408,
  and `1262723/1262724` for seed 3409, with strict pairwise `afterok` seed order. All bind exact commit `2b3b3243...` and the same
  frozen calibration. No held-out metric may be read before all training and prediction seals are complete.

### 2026-09-01 — Node-level launch abort is transport-only; replacement DAG is authoritative

- Root Jobs `1262719/1262720` were both placed on `g0030` and aborted with signal 53 before their batch scripts started. They have
  no stdout, model process, successful update, run root or checkpoint. Dependent Jobs `1262721`–`1262724` were cancelled unstarted.
- No code, model, configuration, calibration or scientific protocol changed. The sole transport correction excludes `g0030` and
  binds the Slurm working directory to the same exact clean snapshot.
- The authoritative blind chain is now `1262743/1262744` (seed 3407), `1262745/1262746` (seed 3408), and
  `1262747/1262748` (seed 3409), with the same strict `afterok` order. This is not a new scientific attempt or extra seed.

### 2026-09-01 — Seed 3407 matched training is complete; held-out remains sealed

- Control `1262743` and Candidate `1262744` both completed `0:0`. Each terminal audit records 6,000 successful optimizer,
  scheduler, EMA and DUCA schedule updates, and each run has an epoch-59 checkpoint.
- The dependency chain automatically started seed-3408 Jobs `1262745/1262746`; seed-3409 Jobs `1262747/1262748` remain
  dependency-blocked.
- No intermediate or held-out metric has been read. Predictions and one-time evaluation remain blocked until all three seeds finish
  and every terminal identity is validated.

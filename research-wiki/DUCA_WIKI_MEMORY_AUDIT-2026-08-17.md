---
type: memory_audit
node_id: audit:duca-wiki-memory-2026-08-17
date: 2026-08-17
status: audited_no_execution
scope: DUCA only; no model-code, data, remote, or experiment mutation
---

# DUCA 全量研究记忆与资源审计（2026-08-17）

## 结论先行

DUCA 最初且仍应恢复的科学对象不是“小模型直接学习帧索引”。它是一个**间接采集**模型：低成本 scout 在训练期学习逐帧的 `0/1` 动作/背景语义及边界重要性；随后由确定性、可审计的采集器把 `p_action`、状态变化、边界证据和全局上下文转成真实物理时间上的选帧位置与预算。重型 VideoMAE/AdaTAD 只消费选中的原始帧，输出仍以真实时间坐标经过不变的官方解码/NMS。

用户此次校正后，唯一正确的待验证路线是：**间接动作性+边界重要性预测为学习目标，确定性 acquisition 为选择机制，动态 outer-K 为论文核心，fixed-K 只作公平基线、归因控制或失败回退。** 这不是已证实结果：现有历史结果未提供这一路线的干净、同预算、官方 AdaTAD 对照；当前根目录的 `a6bdc084...` 是 SparseHead 污染身份，不能充当 DUCA 证据或训练基线。

本审计只读取资料与资源，不执行代码、训练、评估、数据访问或远端操作。

## 审计材料与证据分级

已复核项目规则、`RTK.md`、`PAPER_PROGRESS.md`、Wiki 入口/查询包/反重复约束/决策史/日志/来源登记/DUCA 版本册、全部以 `duca` 命名的 idea 与 experiment 节点，以及这些节点指向的吸收记录、当前代码和官方参考树。较早过程记录保留为历史证据，不因本次路线校正而删除。

术语严格如下：

- **历史正式结果**：具备明确 job、配置、commit 和官方 evaluator，但仍可能因协议不匹配而不能进主表；
- **诊断结果**：内部 holdout、选择质量、成本 smoke 或单组件验证，不能替代 TAD 主结果；
- **设计/静态实现**：代码、计划或静态测试存在，但无真实视频效果证据；
- **当前资源状态**：本次实际可只读观察到的状态，而非历史日志的转述。

## 1. 原始想法（朴素语言）

先用便宜的低分辨率视频流判断“这里是否像动作、是否像状态转折/边界”，再把这些**语义预测**转换为有限预算的真实帧。不是让一个索引策略网络直接报出第几个帧，而是：

```text
cheap dense video
  -> binary p_action + coarse hidden state
  -> delta/entropy/hidden-change 等间接 transition 描述
  -> transition-centre 与双侧 boundary-burst 重要性
  -> 受限的确定性 exact-K / physical-time acquisition
  -> chronological RGB gather + 原始 timestamp
  -> 不变的官方 VideoMAE/AdaTAD detector、decode 与 NMS
```

原始合同明确 `L_action` 仅学习动作/背景语义，边界/双侧/配额损失学习 scorer 与 burst；只有已证明的合法 hard-swap bridge 才能让 TAD 损失影响 selector，默认不得反写粗 action trunk。训练/验证测试的 GT 边界不应进入推理决策路径。[`ideas/duca-offline-full-window.md:197-229`](ideas/duca-offline-full-window.md:197) [`ideas/duca-offline-full-window.md:241-276`](ideas/duca-offline-full-window.md:241)

这也是本次“0/1 actionness + boundary importance、确定性选择”的直接来源。把它简写成 actionness top-K，或把小 controller 说成论文主方法，都会丢失原始因果结构。

## 2. 时间线、实现和结果的逐项重建

| 阶段 | 实现/任务 | 可确认结果 | 正确解释 |
|---|---|---|---|
| 早期 fixed-K/ledger | PAction、GAS-VT、lattice 与 7e3 budget suite | MUST 出现 `64/384` 跳变、低 mAP；MUST256 节点失败 | 是动态预算的负证据和归因工具，不是当前成功 dynamic-K。 [`ideas/duca-must.md:13-50`](ideas/duca-must.md:13) [`experiments/duca-7e3-budget-suite.md:17`](experiments/duca-7e3-budget-suite.md:17) |
| 70aa fixed-384 | `70aa069...`, job `1154971`, 60 epoch | Avg-mAP `58.39`；effective K 平均 `360.55`、最低 `214` | 不是严格每样本 fixed-384，也无同提交 uniform，不能归因给 learned selection。 [`experiments/duca-70aa-fixed384.md:17`](experiments/duca-70aa-fixed384.md:17) |
| 物理网格 | native stride-2/adaptor ActionFormer 与 grid-aware physical-grid ActionFormer | jobs `1150701/1150842` 的 `64.352/65.696` | `65.696` 来自改变 detector/geometry 的 physical-grid ActionFormer，绝非标准 AdaTAD uniform anchor。 [`query_pack.md:1355`](query_pack.md:1355) [`query_pack.md:2147`](query_pack.md:2147) |
| CellCF | `1642f26...`，jobs `1167481–83` | uniform `63.8594`，transition beta0 `64.2755`，CellCF `64.0610` | CellCF 不能跨 cell 转移配额，已被杀死为 adaptive-allocation 主方法；只保留局部消融。 [`ideas/duca-cellcf.md:15`](ideas/duca-cellcf.md:15) [`duca_model_version_registry.md:225`](duca_model_version_registry.md:225) |
| direct/selected-axis V8 | `cb89586...` 的 direct、homotopy、companion | uniform `64.4580`；direct `63.7102`、homotopy `63.0601`、companion `63.6931` | 这是直接 selector/controller 与 selected-axis bridge 的负结果；不能替代或否定正确的间接 semantic-prediction route。 [`query_pack.md:1692`](query_pack.md:1692) [`duca_model_version_registry.md:242`](duca_model_version_registry.md:242) |
| 80/90 epoch 课程 | K384 与 K192 后续课程 | K384 `65.385724`，K192 `57.967272` | K384 为 90 epoch、epoch-50 为 80 epoch；K192 无 matched native uniform。都是超预算诊断，不能写成 official-60 或 dynamic gain。 [`ideas/duca-two-stage-curriculum.md:170`](ideas/duca-two-stage-curriculum.md:170) [`experiments/duca-dynamic-k-rime-oracle.md:24`](experiments/duca-dynamic-k-rime-oracle.md:24) |
| 选择质量诊断 | epoch-89 转换分数检查 | learned 相对 uniform r0 微升但 r1/endpoint/max-hole 变差 | `abs(delta p_action)+uncertainty` compound proxy 不能叫 pure delta；说明 ranking/coverage trade-off，非 max-gap 漏检。 [`experiments/duca-selection-quality-epoch89.md:70`](experiments/duca-selection-quality-epoch89.md:70) [`query_pack.md:1412`](query_pack.md:1412) |
| RIME/outer-K | `duca-rime` 和 Oracle protocol | 尚无 dynamic Oracle、nested regret、RIME implementation 或同成本 TAD 结果 | RIME 是 discussed candidate，旧 MUST 不能改名复活。 [`ideas/duca-rime.md:109`](ideas/duca-rime.md:109) [`experiments/duca-dynamic-k-rime-oracle.md:24`](experiments/duca-dynamic-k-rime-oracle.md:24) |
| 2026-08 静态 B | dynamic outer-K + physical transport package | 两次静态 locality defect；第二次等价缺陷终止 correction loop | 仅 `STATIC_PLAN_ONLY/PRE_RUN_NOT_READY`，没有 efficacy、成本或论文证据。 [`log.md:5631`](log.md:5631) [`log.md:5653`](log.md:5653) |
| 2026-08-16 日志中的矩阵 | dynamic/full matrix 的后续日志 | 有 job/epoch/loss 的历史记录，但基于 dirty、非 DUCA `a6bdc084...` 根身份；本次远端不可访问 | 不能作为本审计中可复核的 clean official baseline 或动态方法结论。 [`log.md:5657`](log.md:5657) [`PAPER_PROGRESS.md:3`](../PAPER_PROGRESS.md:3) |

## 3. 代码谱系：哪些真的实现了，哪些没有

### 3.1 当前根目录的 DUCA 表面

- `opentad/models/duca/acquisition.py` 中的 `ZeroShotActionnessSource` 可用 motion、feature MLP、manual 或 video-text 来源；默认 motion 是无 THUMOS label 的 fallback，而非本审计要求的受监督 `0/1` scout。 [`acquisition.py:333-444`](../opentad/models/duca/acquisition.py:333)
- `DucaAcquisitionAdapter` 已有 `fixed` 与 `dynamic_must` 两种 budget mode；后者借由 controller 输出 hard K。 [`acquisition.py:917-1053`](../opentad/models/duca/acquisition.py:917)
- `forward_scores` 已显式组合 `p_action`、`transition_score`、uncertainty、boundary 与 utility，但现有网络表面还不是已复核的“二值 scout 后确定性间接 acquisition”的干净官方实验。 [`acquisition.py:1317-1398`](../opentad/models/duca/acquisition.py:1317)
- `PrefixMarginalUtilityBudgetController` 是对**排序后的选择特征**作 block-wise marginal continuation 的小 controller；它不是逐帧 `0/1` 动作分类器。 [`dynamic_budget.py:68-227`](../opentad/models/duca/dynamic_budget.py:68)
- `density_decode.py` 是另一条已声明为 authoritative 的固定密度/投影 decoder；常数 logits 退化 endpoint half-up uniform，但它并不能证明 scout actionness/boundary 路线已经完成或跑过。 [`density_decode.py:1-42`](../opentad/models/duca/density_decode.py:1) [`density_decode.py:253-283`](../opentad/models/duca/density_decode.py:253)

`opentad/models/selectors/duca_online_frame_selector.py` 的当前 dirty 表面确实由 GT 生成 `action_target`、`boundary_target` 与 proxy target，并对 selector 计损失。它表明要恢复的监督语义在代码中留有接口；但该文件处于污染根的未跟踪/未冻结状态，本审计不能把它宣称为 clean official 实现或公平实验结果。 [`duca_online_frame_selector.py:396-451`](../opentad/models/selectors/duca_online_frame_selector.py:396)

### 3.2 被遗忘的间接 scout 证据

历史 diagnostic worktree 的 `tools/bata/train_lowres_action_probe.py` 曾明确用 GT segment center 产生 `0/1` action target，并从 `p_action` 导出 entropy、change、uncertainty/boundary score；随后用确定性 top-K 比较多种间接策略。它自己标为 `diagnostic_only`、`not_connected_to_detector`、`no_detector_training/eval`。因此正确结论是“成功地保留了监督/诊断谱系”，而不是“已有成功 detector method 被遗忘”。 [`train_lowres_action_probe.py:62-88`](../.codex_tmp/OpenTAD_SparseHead_DiagnosticClosure_20260729/tools/bata/train_lowres_action_probe.py:62) [`train_lowres_action_probe.py:493-577`](../.codex_tmp/OpenTAD_SparseHead_DiagnosticClosure_20260729/tools/bata/train_lowres_action_probe.py:493) [`train_lowres_action_probe.py:1010-1037`](../.codex_tmp/OpenTAD_SparseHead_DiagnosticClosure_20260729/tools/bata/train_lowres_action_probe.py:1010) [`train_lowres_action_probe.py:3948-3970`](../.codex_tmp/OpenTAD_SparseHead_DiagnosticClosure_20260729/tools/bata/train_lowres_action_probe.py:3948)

PAction 的 learned fixed/dynamic policy 也停留在该历史 prototype/ledger 表面，不能作为新主线的实现或 mAP 证据。 [`paction_acquisition_policy.py:138-219`](../.codex_tmp/OpenTAD_SparseHead_DiagnosticClosure_20260729/tools/bata/paction_acquisition_policy.py:138) [`paction_acquisition_policy.py:410-429`](../.codex_tmp/OpenTAD_SparseHead_DiagnosticClosure_20260729/tools/bata/paction_acquisition_policy.py:410)

## 4. 被遗漏、扭曲或必须保留的发现

1. **不得说“原始 DUCA 是 actionness top-K”。** 原始合同是 binary actionness + hidden state，经 state-change/transition/boundary 描述后确定性 exact-K；top-K/direct policy 只能是 ablation。
2. **`65.696` 不是 AdaTAD dense/uniform。** 它混入了物理网格 ActionFormer detector/geometry，不能解释目前 `66.42/67.14/65.99` 与官方 `69.03` 的差距。
3. **`65.385724`、`65.650497`、`57.967272` 均不是公平 60 epoch 论文结果。** 前两者分别来自 90/80 total epochs，后者缺 K192 uniform。
4. **CellCF、旧 MUST、direct bridge 都有明确负证据。** 它们可作为基线/归因或 fallback，但不能借改名重回 headline。
5. **动态 K 尚无经验支持。** RIME 的 O1--O4、Oracle regret、risk calibration、same-mean-cost controls均未由干净路线闭环。 [`experiments/duca-dynamic-k-rime-oracle.md:68-171`](experiments/duca-dynamic-k-rime-oracle.md:68)
6. **当前 `PAPER_PROGRESS` 内部有状态混写。** 它一处称 dynamic 训练 active，一处仍称 static block；这反映旧总述、dirty root 和后续运行日志没有同一 clean identity，不能据此提升 claim。 [`PAPER_PROGRESS.md:3-10`](../PAPER_PROGRESS.md:3) [`PAPER_PROGRESS.md:85-122`](../PAPER_PROGRESS.md:85)
7. **日志中 2026-08-16 的 runtime 只可作待核对线索。** 远端无法在本审计中解析，且根 HEAD 是 SparseHead；在 clean identity、官方 checkpoint eval 和数据可见性重新确认前，任何它产生的 loss/epoch 都不转化为 DUCA efficacy 证据。

## 5. 现行方法纠正与最小后续实现合同

在 official baseline identity 通过后，Builder 的最小恢复应保持 detector、loss、NMS 与 evaluator 不变，且先实现/验证下列因果链：

1. train-only scout：逐帧 binary action/background target；独立的 boundary-importance target（含 start/end 及双侧覆盖定义）；
2. deterministic acquisition：仅从 scout 输出及冻结规则派生 frame importance、physical positions 和 outer K；严格记录 requested/effective/unique/backbone K；
3. dynamic outer K：K 由聚合的 actionness/boundary evidence 与预注册风险/成本策略决定，不能由 padding 把实际执行偷偷回填到 Kmax；
4. physical-time：每个选帧始终保留原 timestamp；detector decode、IoU、NMS、序列化前均使用正确原始时间，不把稀疏序号伪装为均匀时间；
5. direct selector：只作为对照，不得承担 headline；fixed K：只作 uniform/fixed learned attribution 与 fallback。

首次正式同合同矩阵固定为：official dense、native uniform fixed-K、间接 actionness-only fixed-K、间接 actionness+boundary fixed-K、同一间接预测器的 dynamic outer-K、direct-selection ablation。六臂共享官方数据/视频级拆分、detector/loss/NMS/evaluator、updates、seed 与 full-stack cost 定义；最终只报告 matched realised mean cost 下的差异。

## 6. baseline-first 执行顺序（尚未启动）

1. 在新的 DUCA-side clean execution worktree 中，从官方 AdaTAD 参考 `01c58b9f2370e914150cf94d392208a4e211c053` 固定原始发布配方；不修改中央官方参照和本 dirty 根。
2. 先评估发布 checkpoint：配置 `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`，同一 `mAP` evaluator、THUMOS `validation` subset、tIoU `0.3:0.7`。发布锚点是 Avg-mAP **69.03**、mAP@0.7 **48.27**。 [`configs/adatad/README.md:63-81`](../../OpenTAD_OFFICIAL_BASELINE_AUDIT_20260817/configs/adatad/README.md:63) [`e2e_thumos_videomae_s_768x1_160_adapter.py:138-159`](../../OpenTAD_OFFICIAL_BASELINE_AUDIT_20260817/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py:138)
3. 只有 checkpoint evaluation 重现且 PRE_RUN 资源通过，才做完全官方训练复现；之后才能将 current matched-dense 的 `66.42/67.14/65.99` 与 official 比较。
4. 该语义差异必须逐项从有效配置和输出比对得出：dataset pipeline/config inheritance、raw video 路径与 split、backbone/预训练权重、optimizer/scheduler、EMA/终点 checkpoint、Soft-NMS/evaluator、runtime/PyTorch/CUDA。不得把 2--3 pp 差距归咎于任一项而不先留存 exact diff。
5. baseline 身份通过后，才恢复上节的间接 scout，并按六臂矩阵先做一个 preregistered pilot；pilot 通过 stop gates 后才启动 full training。

官方 binding 已由只读源码确定：annotation `data/thumos-14/annotations/thumos_14_anno.json`、category map `data/thumos-14/annotations/category_idx.txt`、raw video `data/thumos-14/raw_data/video`；train=`training`，validation/test=`validation`，evaluation=`mAP` with thresholds 0.3--0.7。 [`e2e_train_trunc_test_sw_256x224x224.py:1-4`](../../OpenTAD_OFFICIAL_BASELINE_AUDIT_20260817/configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py:1) [`e2e_train_trunc_test_sw_256x224x224.py:41-100`](../../OpenTAD_OFFICIAL_BASELINE_AUDIT_20260817/configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py:41)

## 7. 数据与资源只读审计

| 必需物 | 观察到的绑定/路径 | 本地 | 远端 | 阻塞的精确实验 | 最小合法下一步 |
|---|---|---|---|---|---|
| THUMOS raw 视频与 official split | remote `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`；train `training`，eval `validation` | **MISSING**：官方参照树无 `data/` | **COMPLETE（中央 2026-08-17 只读核验）**：411 个 MP4 symlink、0 broken、目标约 33G；200 training + 211 validation | 无数据阻塞；仍阻塞于 release checkpoint 与 clean baseline binding | 只读绑定该共享根，不复制数据；symlink 计数须跟随 target，不能用 `find -type f` 误判。 |
| annotation、类别/UID 映射、official evaluator | remote `.../thumos14/annotations/thumos_14_anno.json`、`category_idx.txt`；OpenTAD `mAP` | **PARTIAL**：配方/源码 complete，真实文件 missing | **COMPLETE（中央只读核验）**：annotation 与 map 在共享根；官方 `mAP`/validation/tIoU 0.3--0.7 binding 已由源码固定 | 无数据阻塞 | clean baseline worktree 的 PRE_RUN 只需将官方配置明确绑定到该只读根；不得替换 split/evaluator。 |
| VideoMAE-S pretrain | remote `/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth` | **MISSING**：无 `pretrained/` | **COMPLETE（中央只读核验）** | 无 pretrain 数据阻塞 | 仅在 clean official binding 中引用此共享只读路径。 |
| 发布 AdaTAD THUMOS checkpoint | README 的 VideoMAE-S release link；config 如上 | **MISSING**：无 `exps/` 或 checkpoint | **UNVERIFIED**：本次核验未报告可读 released AdaTAD checkpoint；native MATR checkpoint 目录为空且只有不完整 `.part`，不能替代 | **baseline checkpoint evaluation（第一实验）** | 一次只读确认官方 AdaTAD-S released checkpoint 的准确路径、文件完整性和 config compatibility；若不存在，需有下载权限者按官方 release 准备。 |
| 预处理 feature/cache/track/proposal/manifest | raw-video AdaTAD-S baseline 不需要 I3D/InternVideo2 tensor feature | **MISSING/不适用** | **PARTIAL**：OpenTAD I3D 与 InternVideo2 预期目录 absent；alternate I3D 为空且仅不完整 `.part`；SigLIP2 823 files/~477M、MATR val/test pickles ~3.33/~3.69G 均为不兼容身份 | 不阻塞 raw-video AdaTAD-S checkpoint eval；会阻塞任何错误地把这些异构资产代入 OpenTAD feature 配方的实验 | 不下载、不重链接、不混用。DUCA 的 raw-video official baseline 后，才由 Builder 按冻结实现的真实输入契约准备项目资产。 |
| licence/access receipt | AdaTAD README 说明 pretrain 来源；VideoMAEv2-g 明确不得重分发，需 OpenGVLab request form | **PARTIAL**：政策可读，实际 access 未验证 | **PARTIAL**：共享 THUMOS/VideoMAE-S 已可读；release AdaTAD-S checkpoint access 仍未证实 | release checkpoint eval/任何受限 giant weight 运行 | S baseline 只需确认 release checkpoint；若用 VideoMAEv2-g，必须由用户/机构完成 request form，代理不得代办。 |
| N16R4 storage/quota/Slurm runtime | shared `/data/run01/sczc063/yuzibo`；正确 endpoint 为 `ssh.cn-zhongwei-1.paracloud.com` + `sczc063@BSCC-N16R4` | **UNVERIFIED** | **PARTIAL（中央只读核验）**：`/data` 5.3T total、3.1T free；共享数据根可读。Slurm quota/clean project binding 未在本轮确认 | official train/PRE_RUN（不是 checkpoint eval 的数据根） | 用中央给出的正确 SSH FQDN 做一次项目级只读 PRE_RUN resource/binding inspection；旧 `ssh ... yuzibo` alias 失败是 transport 名称错误，不是数据缺失。 |

### 本轮资源结论

**已更正：共享 THUMOS14 raw/annotation/category-map 与 VideoMAE-S pretrain 的远端资源足以支撑 raw-video AdaTAD-S baseline 的数据侧。** 先前的 `UNVERIFIED` 仅源于把未定义的本地 alias `yuzibo` 当作 SSH host；它不是远端数据不存在的证据。I3D/InternVideo2 的缺失、空 alternate download、SigLIP2 和 native MATR assets 都不得被混入该 official raw-video baseline。

**完整官方训练仍不可开跑**，但剩下的是两个明确的 baseline identity/PRE_RUN blocker，而不是数据或存储 blocker： (1) released AdaTAD VideoMAE-S checkpoint 的可读、完整、与官方 config 匹配的精确绑定；(2) 一个 clean DUCA-owned execution worktree 对 official commit/config/evaluator/runtime 的冻结与 Slurm quota PRE_RUN。checkpoint evaluation 可在 (1) 与其 clean binding 完成后进行；只有 checkpoint evaluation 通过，才有资格开始 untouched official training。若 release checkpoint 不在共享根，需要有下载权限者按官方链接准备；VideoMAEv2-g request form 仍是机构/用户权限事项。

### 7.1 远端视频资源地图（中央 2026-08-17 只读核验）

| 数据集 | 可见资源与状态 | DUCA 使用约束 |
|---|---|---|
| THUMOS14 | **COMPLETE（canonical only）**：`thumos14/raw_data/video` 有 411 个有效 MP4 symlink（200 training、211 validation、0 broken）。物理存储共 413 个视频约 33G；其中 `video_test_0000270.mp4` 与 `video_test_0001292.mp4` 是非 canonical extra，不能加入。 | 当前 official AdaTAD-S baseline 只绑定 canonical 411 和 annotation video-ID mapping；不复制视频，也不得把 physical store 的 extra test files 纳入训练或评估。 |
| MultiSports | **PARTIAL / archived**：`projects/stad-paper/data/r0b02/archives/{aerobic_gymnastics,basketball,football,volleyball}.tar` 合计 43,820,810,240 B，内含 2129 videos；只有 18 个 MP4 被抽取。annotation 位于 `projects/stad-paper/data/multisports_01600...`，`MultiSports_box.zip` proposal 存在。 | 不是可直接运行的完整视频树；不得为 DUCA baseline 解压、复制或用 18 个样本替代完整数据集。 |
| TOC-Bench | **COMPLETE（raw-video tree）**：`tstep_v0_phase0/datasets/toc_bench_full/videos` 有 1951 MP4、15,235,916,868 B、无零字节文件；另有 75 个小型 logic/diagnostic videos。 | 仅记录为可用的将来跨数据集候选；没有冻结 split/evaluator/route 前，不进入 DUCA baseline 或训练。 |
| Charades | **COMPLETE（raw-video tree）**：`datasets/charades/raw_data/Charades_v1_480` 有 9848 MP4、16,588,858,990 B，原 zip 亦存在。 | 未冻结 TAD protocol/official evaluator，不能作为当前 THUMOS14 baseline 的替代。 |
| ActivityNet | **PARTIAL / unusable**：仅 3 个直接可读 MP4（约 71.7MB）；v1-2/v1-3 archives 与 missing-files zips 未组装、未验证为视频树。 | 当前不可提交为完整 ActivityNet 实验；不得从 archives 推定可用性。 |
| FineAction、HACS、EPIC-Kitchens、Ego4D | **MISSING**：仅 scripts/readmes/小型 source archives，FineAction probe 为空。 | 不构成实验数据；若未来需要，必须走独立合法获取与 protocol freeze。 |
| EventMATR / feature assets | **PARTIAL but incompatible**：无重复 raw video；native MATR val/test feature pickles 存在，MATR checkpoint 不完整；OpenTAD I3D/InternVideo2 tensors 仍 absent。 | 不可替代 raw-video AdaTAD 或本项目的 OpenTAD feature identity，不能混用作 baseline。 |

这个地图只描述资源可见性，不提供跨数据集结果、许可证推断、性能、成本或训练授权；所有可能的后续 config/manifest 或 symlink 变更必须由 Builder 在冻结的数据协议内完成。

## 8. 周期性 checkpoint 与恢复规则（用户修正）

- 后续任何完整 official training：默认每 **5 epochs** 写一次可恢复 `.pth`；若完全未改的官方 recipe 更频繁，保留官方频率。AdaTAD S 配方已有 `checkpoint_interval=2`，所以该 baseline 及其严格复现应保持 **每 2 epochs**，不降为 5。 [`e2e_thumos_videomae_s_768x1_160_adapter.py:150-156`](../../OpenTAD_OFFICIAL_BASELINE_AUDIT_20260817/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py:150)
- periodic checkpoint 只用于恢复/诊断；模型选择仍严格遵守 preregistered final 或 final-EMA，不得查看中间 validation 后 cherry-pick。
- 每个未来 work dir 必须保留最近 **3** 个有效恢复点及预注册 milestone/final。要在 PRE_RUN packet 写明 interval、retention、resume argv、输出目录和估计存储。
- 官方保存代码包含 model、EMA、optimizer、scheduler、epoch。 [`checkpoint.py:5-22`](../../OpenTAD_OFFICIAL_ADATAD_01c58b9/opentad/utils/checkpoint.py:5) 但它**没有**保存 AMP scaler 或 RNG state，且 `save_best_checkpoint` 不含 optimizer/scheduler。任何需要中断后严格可复现恢复的 DUCA full-run，须在 PRE_RUN 前由 Builder 做最小项目局部 recovery patch，补齐 scaler、epoch/update 与 Python/NumPy/PyTorch/CUDA RNG restore，并经 Critic/Evaluator 验证；不得把该补丁用于改变模型选择规则。

## 当前交接

- **当前科学问题**：动作/边界语义驱动的确定性间接 acquisition，能否在 matched realised mean cost 下以 dynamic outer-K 保护 high-IoU，而非 direct index policy？
- **下一动作**：由 Builder 建立 clean official AdaTAD execution binding，并只读确认 released AdaTAD-S checkpoint；在 checkpoint eval 前冻结 config/evaluator/runtime 与 Slurm PRE_RUN packet。
- **next_owner**：DUCA owner/Builder（baseline binding；不修改中央 official reference 或 dirty root）。
- **dependency**：released AdaTAD-S checkpoint 的可读完整 binding + clean official code/config/evaluator/runtime/Slurm PRE_RUN；THUMOS raw/annotation/VideoMAE-S 与 storage 已非 blocker。
- **expected_return**：baseline PRE_RUN receipt，随后才是 released-checkpoint evaluation receipt；没有 checkpoint binding 则 `NEEDS_ATTENTION`。
- **single_recovery**：一次使用正确 N16R4 FQDN 的只读 checkpoint/binding inspection；不得下载、重传、猜测或提交训练。

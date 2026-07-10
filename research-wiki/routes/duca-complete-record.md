# DUCA 完整讨论、实现与实验档案

更新时间：2026-07-11

## 0. 档案范围

本页整合 C3/PAction/GAS-VT 到 DUCA、DUCA-JCT、DUCA-MUST、CFPA、X3D/SlowFast、全栈成本审计和最终 pivot 的全部关键讨论。原始长评审仍由 `source_map.md` 固定路径、commit 与 SHA；本页负责把它们组织为一条不互相矛盾的研究线。

## 1. 最初问题与设计初心

最初问题不是“对 AdaTAD 再加一个模块”，而是：

> 能否先用显著低于 VideoMAE 的粗粒度视觉模型识别动作/背景与状态变化，再间接选择对 TAD 边界定位真正有用的帧，使重 backbone 只处理有限观测？

初心中的模块职责一直是：

1. **粗分类器**：以动作/背景二分类 GT 监督，产生 deploy-visible `p_action` 和隐藏特征。
2. **间接选择器**：首要保护状态转换、起点、终点和边界上下文，而不是动作内部覆盖。
3. **TAD detector**：使用真实检测损失决定哪些观测有任务效用，并反向影响前两者。
4. **预算/覆盖合同**：固定或动态预算必须真的改变 detector 消费；最大间隔用于安全，不是论文创新。

## 2. 路线演化

### 2.1 C3 / PAction

低成本 probe 学习 `p_action`，PAction 用概率、变化量和少量边界/coverage 代理进行固定预算选择。

得到的正面认识：

- 二分类 actionness 能提供有用稀疏输入信号；
- 简单、直接的选择器可能比复杂先验更强；
- fixed384 是重要归因锚点。

得到的限制：

- `p_action` 高分主要落在动作内部；
- 二分类监督不提供精确 start/end；
- PAction 仍是 actionness/GT surrogate，不是 detector-aware utility。

### 2.2 GAS-VT

GAS-VT 引入 remaining budget/time、budget pressure、gap urgency、boundary bracket、action interior、CVaR hole 和 hard repair，希望形成 value transport。

最终诊断：

- 实际路径是全局打分 -> top-k -> repair，不是逐点选择、更新状态再选择的 sequential VT；
- train/apply 对 target budget 的条件可能不一致；
- fixed384 本身是半密度输入，早期 mAP 不能归因于 GAS；
- coverage/gap/action interior 正则可能与高-IoU boundary utility 冲突；
- hard repair 可生成接近 uniform 的最终格点，即使 metadata 标记无 uniform fill。

因此 GAS-VT 被降级为 engineered coverage baseline。

### 2.3 Detector utility / Stage2

主线一度转为用 dense AdaTAD train-only teacher 估计 observation utility，再蒸馏给 deployable selector。

理想 utility 应包含：

- per-point positive assignment responsibility；
- classification/regression/quality loss sensitivity；
- start/end boundary responsibility；
- high-IoU matching gain；
- false-positive/background-suppression risk；
- signed positive gain 与 negative risk；
- 可选 group/span counterfactual utility。

现实实现多次只达到 proposal score spreading、GT endpoint proxy 或 boundary-utility proxy，因此不能称为真实 detector utility。

### 2.4 DUCA plugin

目标被重构为 detector 前的 temporal acquisition adapter，而非新 detector：

```text
full-window raw observations
  -> trainable coarse probe
  -> p_action + hidden features
  -> transition/start/end/context/utility-first selector
  -> exact hard budget and max-gap feasibility
  -> selected observations + original positions
  -> official-derived AdaTAD/ActionFormer backend
```

Ledger 只允许做事后 audit，不得成为最终 inference 决策源。

### 2.5 “online”术语纠错

DUCA 不是 Online TAD、streaming 或 causal acquisition。历史代码名中的 `online` 只表示：

- actionness 在同一 model forward 内产生；
- 不依赖预导出 JSONL、ledger、selector checkpoint 或 prediction cache；
- probe/selector 可以观察完整离线窗口后再选择。

最终正确术语是：`offline_full_window`、`runtime_generated`、`cache_free`、`in_forward`、`jointly_trained`。

## 3. 最终尝试过的 DUCA 模型合同

### 3.1 Coarse probe

- 主配置要求 official ASFormer code path；明确禁止 `asformer_lite`。
- “official code”不等于加载 official pretrained weights，必须单独记录 checkpoint/provenance。
- probe 使用 binary action target；selector 可读取 `p_action`、logits 和 hidden features。
- 冻结 probe 时，父模型 `train()` 不得把它切回 train mode。

### 3.2 Selector observable inputs

讨论和最终代码覆盖过：

- dense observation descriptors；
- `p_action`、actionness logits；
- `delta_p_action`、`abs_delta`；
- entropy/uncertainty peaks；
- coarse hidden features；
- current position、valid mask、budget state；
- transition/start/end/context/utility heads。

最终要求 actionness 权重显著小于 transition/boundary 权重，actionness 只做辅助校准。

### 3.3 Hard selection

早期：center/radius decode、top-k 和 hard max-gap repair。

关键修正：hard max-gap repair 必须先建立满足最大 hole 的可行骨架，再用剩余预算做边界优先填充；否则 full train 会在验证阶段 fail closed。

最终 CFPA/structured route：

- 同一 exact-K/max-gap feasible-state machine；
- hard Viterbi 负责真实 forward/inference；
- entropy-regularized forward-backward 是同一路径的 relaxation；
- 不再使用另一套 soft-resample policy 近似 hard policy。

### 3.4 Detector gradient bridge

目标是 detector loss 对真实 hard selection 有意义地求导。

先后出现过：

- ST selected mask weighting；
- soft context gradient path；
- slotwise soft-to-hard resampling；
- structured zero-forward bridge。

最终合同要求：

```text
hard_features = true hard gather consumed by detector
soft_features = relaxation of the same structured policy
bridge_output = hard_features + alpha * (soft_features - stopgrad(soft_features))
```

数值 forward 必须等于 hard path；eval 时 `alpha=0` 且不执行 soft branch。

但 nonzero gradient 只证明连通。仍需 hard one-swap finite difference 验证梯度方向是否与真实 detector loss change 对齐。

### 3.5 Detector backend

准确口径不是“完全未修改的官方 AdaTAD”。结构审计表明：

- official AdaTAD base config 与 `ActionFormerHead` 可保持字节/配置一致；
- `anchor_free_head.py`、`actionformer.py`、`single_stage.py`、`vit_adapter.py` 已扩展；
- detector 输入长度从 768 改为 K；
- selected-axis baseline 将 GT 映到 0..K-1，head 在等间隔 rank 上做 assignment/回归；
- postprocess 再映回 original time。

安全表述是：

> official OpenTAD/AdaTAD-derived detector components with a DUCA pre-backbone selector and selected-axis coordinate adapter.

selected-axis 的内部物理时间错误是 DUCA 未解决风险，也是后来转向 PhysTime 的直接动机。

## 4. 训练方式争论与最终尝试

### 4.1 为什么 GPT 多次建议三阶段

建议的 teacher utility pretrain -> sparse detector warmup -> joint fine-tune，主要是为了解决离散选择、随机初始化和 detector/selector 同时漂移的稳定性问题。它作为验证路线有合理性，但用户正确指出：若最终论文仍依赖三个独立模型/checkpoint，就不够优雅，也无法声称单模型协同学习。

### 4.2 DUCA-JCT 单作业 curriculum

最终尝试改成同一训练作业：

1. detector 从开始就训练；
2. 早期以稳定 selection/reference 输入校准 probe；
3. 按真实 optimizer steps 渐进打开 learned structured policy 和 detector-to-selector bridge；
4. actionness 权重下降，transition/endpoint/proxy 与 detector gradient 上升；
5. MUST 的 dual update 也只在 optimizer step 后执行。

### 4.3 必须避免的训练错误

- leaf losses、alias 和 `total_loss` 被重复聚合；
- detector loss 在 wrapper 和 engine 两处重复计入；
- utility alias 与同一 loss 重复；
- warmup 与 joint phase 改变 DDP 参数使用集合；
- uint8 raw window 直接进入 einsum；
- `get_optim_groups()` 漏掉 Conv2d/BatchNorm/Embedding/probe/selector/controller 参数；
- schedule 按 forward count 而不是 optimizer step 推进；
- detector loss 很晚才进入，且 soft direction 不对应 hard inference。

这些问题可以解释联合训练一度低于分离训练：输入分布先被扰动，随后被错误 proxy 主导，detector gradient 到达太晚且方向错配。

## 5. Fixed 与 dynamic budget

### 5.1 Fixed

- fixed384 是主要安全锚点；
- fixed256/fixed128 用于 accuracy-budget curve；
- 选择数必须是 detector 实际消费数，而非 ledger 行数或 mask cap。

### 5.2 MUST

MUST 尝试 prefix marginal utility stopping、离散 K buckets、Lagrangian/dual constraint。

暴露的 K 语义：

- soft expected K；
- hard requested K；
- effective K；
- unique K；
- padded/backbone K；
- ST K。

当 detector 总是 pad 到 budget max 时，K(x) 不产生真实 variable compute。64/384 跳变来自离散 bucket、温度、dual update 和 padded execution 的不一致。最终配置已标记 `diagnostic_only=True`、`dynamic_compute_realized=False`。

## 6. Max-gap、边界和 move 系列

用户要求最大未选间隔放宽到 10 或 15 帧/位置，并延续 move 系列骨架与 soft interval constraint。

重要结论：

- max-gap 是 safety/observability constraint，不是主 novelty；
- hard repair 必须记录 pre/post hole 与插入/替换次数；
- soft hole loss 不保证 hard geometry；
- move25/move50 聚集偏移不能只归咎于 coarse probe；
- 更可能是 binary target 粗、probe stride/smoothing delay、score/decoder mismatch 和 repair 改写共同造成。

## 7. Train-free prior 讨论

### 7.1 X3D

- frozen Kinetics X3D 的 max probability/entropy 被用作 train-free action prior；
- 它不是无预训练，也不天然 THUMOS-free；
- formal interval grid/export 运行时间过长；
- JSONL 会把主方法重新变成 offline precompute；
- dense scout 成本可能淹没 heavy backbone savings。

结论：停止密集 X3D 主实验，只保留 appendix baseline。

### 7.2 SlowFast Fast

- Fast pathway 与 Slow pathway 联合预训练，不是独立官方模型；
- Fast-only 更偏 motion/transition，但易受 camera motion 干扰；
- fused/Slow/Fast 可作 prior 消融；
- 必须记录 Kinetics checkpoint 与 class overlap。

结论：appendix/upper-bound diagnostic，不替代低成本主 probe。

## 8. 当前代码库存

当前 PhysTime 分支仍包含 DUCA 主要代码：

- `opentad/models/duca/acquisition.py`
  - `SparseTemporalGrid`
  - `ZeroShotActionnessSource`
  - `C3CoarseProbeActionnessSource`
  - `DucaAcquisitionAdapter`
  - `budgeted_center_radius_decode`
  - `gather_selected_observations`
  - `DucaOnlineSparseDetectorWrapper`
  - `duca_losses/duca_forward_train/duca_forward_test`
- `opentad/models/duca/structured_selection.py`
  - hard Viterbi、soft forward-backward、`global_structured_topk`
- `opentad/models/duca/dynamic_budget.py`
  - `DynamicBudgetDecision`
  - `PrefixMarginalUtilityBudgetController`
- `opentad/models/selectors/duca_online_frame_selector.py`
  - target generation、schedule、bridge、external actionness rejection、selected-axis mapping
- official/precheck configs、validators、one-step proof、suite monitor、paper evidence collector 和 focused tests。

## 9. 关键提交谱系

| Commit | 作用 | 当前证据口径 |
| --- | --- | --- |
| `603ed02` 附近 | 首版 DUCA online plugin core | skeleton/smoke |
| `c799c48` | detector-aware teacher utility adapter | 早期独立 worktree |
| `edaf589` | signed detector utility evidence | teacher/utility 诊断分支 |
| `3ce6bae` / `679f194` / `b15c278` | Stage2-4 runners、precheck 与 evidence gates | 分工 worktrees，不是单一最终模型 |
| `36c92d4` / `05baa48` | TrueTime E2E precheck 与 selector gradient proof | one-step/contract evidence |
| `84e95d6` | official AdaTAD backend config | backend integration candidate |
| `32507c6` | MUST dynamic budget | diagnostic candidate |
| `ed3d703` | runtime C3 coarse probe | joint probe path |
| `986c83c` | X3D actionness 接 official backend | appendix baseline path |
| `308088c` 至 `cbde70d` | JCT progressive schedule | single-job joint training |
| `544eca6` | transition-first | selector priority correction |
| `41bc7c9` | final selector contract | hidden/boundary/gap contracts |
| `7bea4fc` | official proof loss config | loss/gate repair |
| `7e3a508` | max-gap scaffold repair | old full-run anchor，非最终证据 |
| `f705dda` / `88e50b1` | SlowFast Fast diagnostic | appendix only |
| `1684f6b` / `c26b349` / `70aa069` | structured joint repair | latest DUCA code inherited by PhysTime branch |
| `a5e1774` | 全栈成本与官方结构审计 | divergent DUCA audit branch，非当前 PhysTime HEAD |

全部本地 worktree、branch 与审计 HEAD 见 `../worktree_inventory.md`。这些分支曾按 stage/owner 拆分，不能把各自通过的局部 gate 拼成一个已经验证的最终 DUCA 模型。

## 10. 实验作业谱系与证据分类

历史作业包含：

- `1151091/1151092`：旧 official fixed/MUST，因缺预训练 checkpoint 失败；
- `1151093`、`1151305` 等：X3D grid/export，长时间运行且被降级；
- `1151072`：move50 diagnostic，旧 permission failure；
- `1151863/1151864`、`1151927` 至 `1151955`：不同旧 commit 的 fixed/dynamic/budget diagnostics，后来取消释放队列；
- `1152332` 至 `1152338`：7bea4fc suite，被 7e3a508 supersede；
- `1152687` 至 `1152693`：7e3a508 gate + fixed384/256/128 + MUST384/320/256。

这些结果来自不同修复状态，不得拼成 final table。Wiki 不复制 mAP 数字；需要时回到对应 run artifact 和权威 result record。

## 11. ResearchClaw 第二轮 24 个候选

另一轮对 `70aa069` 的发散审查产生 24 个候选，和后续 23 候选不是同一集合：

- Codec/acquisition：Deterministic Codec Skeleton、CoDeTAD、Boundary-Triggered Decode Refinement。
- Time x layer：CVCR-TAD、Packed Time-Layer MoD、Monotone Anytime Depth Ladder。
- Reuse/transport：BCFT、Codec-MV Residual Transport、Cross-Video Prototype Cache。
- Physical time：Continuous-Time Physical Head、Gap-Aware Temporal Convolution、Time-Warp Consistency。
- Spatiotemporal allocation：Joint Allocation、Boundary Zoom、Short-Action Insurance。
- Utility/rate-distortion：One-Swap Utility Distillation、High-tIoU Risk-Rate、Compute Attribution Audit。
- Cross-task：Unified Controller、Procedure-State Allocation、Dense Physical Compute API。
- Systems/risk：Full Compute Ledger、Distribution-Shift Risk、Offline Global Budget Allocation。

该轮 Top-5 为 CVCR-TAD、CoDeTAD、BCFT、Continuous-Time Physical Head 和 End-to-End Compute Ledger。吸收记录没有直接接受“立即停止 DUCA、CVCR 必须取代”的裁决，而是要求先做三项最便宜实验：full-stack trace、one-swap utility alignment、same-selected-frames geometry comparison。

## 12. 全栈成本与结构审计分支

`a5e1774` 额外实现/记录：

- input/decode/preprocess/H2D/probe/selector/backbone/projection/neck/head/postprocess 分段计时；
- p50/p95、显存、energy 与成本守恒；
- 正式 bridge 静态 FLOPs 漏记修正；
- official AdaTAD source/blob 对照；
- 明确 selected-axis 改变 assignment 坐标语义；
- one-swap finite-difference audit 仍是待完成科学 gate。

该分支当前是 `codex/gas-vt-stage23-detector-aware-20260706` 的 `a5e1774`，不在当前 PhysTime HEAD 的 ancestry 中。

## 13. 为什么最终 pivot

DUCA 不是“代码没实现”而被放弃。相反，它已经积累了大量工程正确性资产。最终 pivot 的原因是研究中心仍面临：

1. 决策变量过窄：只选 frame subset，没有控制 layer/cache/decode 等资源；
2. 目标不纯：actionness、GT boundary proxy、gap、entropy 与 detector bridge 混合；
3. selected-axis 内部时间几何未解决；
4. 与 adaptive sampling/top-k/scaffold 近邻碰撞强；
5. 全栈 savings 与跨 detector/dataset 证据不足；
6. 动态预算没有真实 variable execution。

用户随后选择把“selected-axis 几何错误”提升为新的独立 TAD 问题，即 PhysTime。

## 14. 保留资产

- no-GT/no-teacher/no-cache recursive audit；
- original-position 与 seconds round-trip tests；
- official backend parity/optimizer coverage；
- hard/soft structured discrete policy tests；
- full-stack cost profiler 思想；
- selector geometry、boundary error、high-IoU evidence schema；
- old baselines 和 failure diagnostics。

## 15. 禁止重复

- 不再增加 frozen actionness prior；
- 不再靠调 actionness/boundary/uncertainty 权重挽救主线；
- 不再增加另一套 differentiable top-k；
- 不再把 max-gap/radius repair 包装成核心创新；
- 不再把 nonzero gradient 当 hard utility evidence；
- 不再把 padded MUST 写成 dynamic compute；
- 不再把 selected-axis post-hoc remap 写成内部物理时间已解决；
- 不再把不同 commit 的 mAP 混为主结果。

## 16. 当前裁决

DUCA 是**完整记录、停止扩展的历史主线**。其代码、合同、成本审计与失败分析继续服务 PhysTime baseline 和工程质量，但当前论文方向不再以选帧插件为中心。

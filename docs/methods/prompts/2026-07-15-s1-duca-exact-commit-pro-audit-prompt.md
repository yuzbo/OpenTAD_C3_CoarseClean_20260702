# Spatial Zoom S1 与 corrected DUCA P0 exact-commit Pro 审核 Prompt

请以最高推理强度完成一次只读、证据优先、逐行代码审核。你的任务不是替现有路线辩护，
也不是根据本 Prompt 的叙述补全一个看似合理的故事。你必须从公开 GitHub 代码中独立重建
实现、训练协议、推理协议、证据链和下一步计划，然后严厉判断它们是否正确、是否足以回答
研究问题、是否还隐藏确定性错误。

本轮只做审核、批判与可执行修复设计。不要修改仓库，不要提交代码，不要启动训练，不要
把未完成实验推测成结果。

## 0. 强制工作方式

开始前发现并加载当前环境中与本任务相关的技能。至少检查：

- `gpt-5-pro`、`research-review`、`experiment-audit`、`paper-claim-audit`；
- `oss-audit`、`pytorch-training`、`mixed-precision`、`distributed-training`；
- `experimental-design`、`result-to-claim`、`ccf-a-editorial-review`、`kill-argument`；
- `research-wiki`、`analyze-results`、`system-profile`。

在回复开头逐项报告 `loaded / unavailable / emulated`。不得虚构已经加载的技能。若支持并行
独立 reviewer，请至少安排四个互不共享结论的角色：

1. PyTorch/DDP/AMP 代码审计；
2. OpenTAD/AdaTAD/TAD 方法审计；
3. 实验完整性、统计与数据泄漏审计；
4. 系统成本与可部署性审计。

最后由一个 adversarial senior reviewer 统一裁决。任何 reviewer 的结论都必须回到代码、
配置、测试或真实 artifact；不得用多数投票代替证据。

## 1. 固定审核对象与可见性证书

仓库：

`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

### 1.1 Spatial Zoom S1 主审核对象

- 分支：`codex/spatial-zoom-s1-formal-20260715`
- exact commit：`35204f58fd3e91d7cf8f5888928a41e9bf6c2e72`
- commit URL：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/35204f58fd3e91d7cf8f5888928a41e9bf6c2e72`
- exact tree：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/35204f58fd3e91d7cf8f5888928a41e9bf6c2e72`

### 1.2 corrected DUCA P0 独立审核对象

- 分支：`codex/duca-transition-only-20260711`
- exact commit：`043be401ba2b694342dc395f263e9a9858628d69`
- commit URL：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/043be401ba2b694342dc395f263e9a9858628d69`
- exact tree：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/043be401ba2b694342dc395f263e9a9858628d69`

这两个 commit 属于两条独立实验线。不得把一个分支的代码、测试、结果或结论移植成另一分支
的证据。

首先输出 Repository Visibility Certificate，至少包括：

- 实际读取的仓库、分支和完整 commit SHA；
- 实际打开并完整阅读的文件清单；
- 无法访问、被截断或只看到摘要的文件；
- 是否能看到测试、配置、训练入口、推理入口和证据生成代码；
- 是否存在 Prompt 所述 commit 与实际 tree 不一致的情况。

如果不能读取 exact commit，停止给出代码裁决。不得用 commit message、文件名、Prompt 摘要
或历史 review 冒充源码证据。所有代码判断必须给出可点击的 GitHub `file#Lx-Ly` 链接。

## 2. 不得改写的研究边界

1. 任务是完整窗口可见的离线 TAD，不是 Online/Streaming/Causal TAD。
2. 最终目标是在真实全流程成本下降时保护高 tIoU 定位，不是单纯减少理论 token 或 FLOPs。
3. S1 只是空间分辨率 headroom 的 falsification gate，不是 Zoom 模型。S1 不应包含 ROI、
   scout、teacher、routing policy、时序 selector 或新 detector。
4. S1 只有在 dense224/256 相对 matched dense160 显示可信的高 tIoU/短动作收益，并经过
   完整成本分析后，才可能授权 S2。S1 失败必须 KILL 空间 Zoom 路线，不能靠增加 policy/loss
   延长路线。
5. DUCA P0 是独立的时序去冗余裁决。它只测试 corrected fixed-K transition-only 假设及
   counterfactual detector supervision，不得与 S1 混成一个方法。
6. DUCA 的粗模型应学习二分类动作状态；间接定位来自 deploy-visible 状态变化和不确定性。
   不得把 GT start/end 直接变成粗模型推理输入，也不得把显式 boundary predictor 偷换成
   “粗分类间接边界定位”。
7. validation/test GT、teacher、dense prediction cache、oracle 或 ledger 不得参与部署决策。

如果代码与这些边界冲突，必须报告冲突，而不是自动修改研究目标来迁就代码。

## 3. 必须先读的仓库记忆与合同

两个 exact commit 都应先读：

- `AGENTS.md`
- `RTK.md`
- `research-wiki/query_pack.md`
- `research-wiki/anti_repetition.md`

S1 commit 重点读：

- `docs/methods/spatial_zoom_s1_contract.md`
- `research-wiki/experiments/spatial-zoom-s1-infrastructure.md`
- `configs/adatad/thumos/s1_dense160_videomae_s_768x1_adapter.py`
- `configs/adatad/thumos/s1_dense224_videomae_s_768x1_adapter.py`
- `configs/adatad/thumos/s1_dense256_videomae_s_768x1_adapter.py`
- `tools/train.py`、`tools/test.py`
- `opentad/cores/train_engine.py`、`opentad/cores/test_engine.py`
- `tools/bata/spatial_zoom_s1_contract.py`
- `tools/bata/validate_spatial_zoom_s1.py`
- `tools/bata/run_spatial_zoom_s1_precheck.py`
- `tools/bata/spatial_zoom_s1_training.py`
- `tools/bata/spatial_zoom_s1_evidence.py`
- `tools/bata/spatial_zoom_s1_test_open.py`
- `tools/bata/select_spatial_zoom_s1_checkpoint.py`
- `tools/bata/profile_spatial_zoom_s1.py`
- `tools/bata/spatial_zoom_s1_cost.py`
- `tools/bata/analyze_spatial_zoom_s1_results.py`
- `tools/bata/build_spatial_zoom_s1_run_descriptor.py`
- `scripts/run_spatial_zoom_s1_precheck_slurm.sh`
- `scripts/run_spatial_zoom_s1_train_slurm.sh`
- `scripts/run_spatial_zoom_s1_test_profile_slurm.sh`
- `tests/test_spatial_zoom_s1_infrastructure.py`
- `tests/test_train_engine_max_train_iters.py`

DUCA commit 重点读：

- `docs/methods/duca_transition_only_contract.md`
- `opentad/models/duca/transition_only.py`
- `opentad/models/duca/structured_selection.py`
- `opentad/models/duca/counterfactual_utility.py`
- `opentad/models/duca/acquisition.py`
- `opentad/models/selectors/duca_online_frame_selector.py`
- `opentad/models/detectors/actionformer.py`
- `opentad/models/dense_heads/actionformer_head.py`
- `configs/adatad/thumos/duca_exact_uniform_fixed384_official_adatad_backend_full_train.py`
- `configs/adatad/thumos/duca_direct_boundary_fixed384_13200_official_adatad_backend_full_train.py`
- `configs/adatad/thumos/duca_transition_only_fixed384_no_detector_bridge_official_adatad_backend_full_train.py`
- `configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py`
- `scripts/duca_transition_only_p0_canonical_env.sh`
- `scripts/prepare_duca_transition_only_p0_suite.sh`
- `scripts/prepare_duca_transition_only_p0_ddp_pilot.sh`
- `scripts/run_duca_transition_only_p0_ddp_pilot.sh`
- `tools/bata/run_duca_transition_only_formal_full_model_gate.py`
- `tools/bata/validate_duca_transition_only_p0_variant.py`
- `tools/bata/validate_duca_transition_only_p0_suite.py`
- `tools/bata/validate_duca_transition_only_p0_ddp_pilot.py`
- 所有 `tests/test_duca_transition_only*`、optimizer coverage、detector-gradient、
  counterfactual utility、structured selection 和 mixed-length 测试。

不得只抽查测试而不读生产调用点。不得只读生产函数而不追踪配置如何实际调用它。

## 4. 当前外部运行快照，只能作为待核验线索

以下内容来自 2026-07-15 16:53 +0800 的远端 Slurm 状态，不在 GitHub tree 内。若你无法
读取远端 artifact，必须标记为 `OPERATIONAL CLAIM / NOT INDEPENDENTLY VERIFIED`，不能把它
当作代码正确性或性能证据。

### 4.1 S1

- full precheck Job `1164289` 被报告为完成；Slurm 分配物理 GPU 4，进程内使用逻辑
  `cuda:0`。
- 当前 3x3 训练 Jobs：`1164291`、`1164307-1164314`，对应
  dense160/224/256 × seeds 3407/3408/3409。
- 当时九个任务均为 `RUNNING`，累计发现 85 个 `epoch_*.pth`，未扫描到 Traceback、OOM 或
  non-finite loss/cost。
- pilot `1164291` 被报告已写出并校验 epoch-1 checkpoint/sidecar。
- 这些只是运行状态，不是 mAP、cost、S1 GO 或论文证据。

### 4.2 DUCA corrected P0

- exact CUDA gate Job `1164318` 和四臂 10-step DDP pilot Job `1164319` 被报告为成功。
- formal seed-0 Jobs `1164700-1164703` 对应 exact-uniform、direct-a5、transition beta=0、
  transition counterfactual；当时均为 `RUNNING`。
- 当前 corrected P0 没有 mAP，不得宣称 C3/C4 成立。

### 4.3 明确无效或不可匹配的历史现象

- 历史 uniform Jobs `1150701/1150842` 的 Avg-mAP 64.352/65.696 有日志来源，但协议、
  detector/geometry 或 loader 与 corrected P0 不完全匹配，不能填入当前 matched 表。
- 旧分离 lattice best 63.18、PAction best 61.02 与部分 DUCA run 的 optimizer exposure 不同，
  不能直接归因为“联合训练更差”。
- `1159414-1159417` 的 uniform/homotopy 协议已失效；其数字只能作为失败诊断。
- S1 的 `911448a`、`7d1e9cc`、`9298c0e` 前序部署分别暴露 AMP fail-fast/端口冲突、
  replay 未恢复 forward-mutated buffers、合法 runtime config mutation 被误判为漂移。
  它们不是 S1 结果。
- corrected S1 和 corrected DUCA 仍在运行。禁止预测它们最终会优于或劣于基线。

请首先判断这些“现象”中哪些能由 GitHub 代码复核，哪些必须等待远端 artifact，并明确指出
当前能回答与不能回答的问题。

## 5. S1 必须逐行审核的问题

### 5.1 matched matrix 与官方模型语义

1. 展开三份 config 的完整继承链，机器式比较最终配置。是否真的只有 spatial resize/crop 和
   `work_dir` 不同？训练步数、augmentation、temporal grid、VideoMAE-S、adapter、
   ActionFormer projection/head、NMS 和 evaluator 是否完全匹配？
2. dense160 是否真正对应当前 official-derived AdaTAD 本地基线，而不是另一个经过隐式修改的
   模型？列出相对官方 AdaTAD/OpenTAD 的所有结构与行为差异。
3. 160/224/256 是否产生预期 10x10/14x14/16x16 spatial token grid？位置编码插值、
   checkpoint load 和 detector 输入 `[B,384,768]` 是否一致？
4. 是否存在 resolution 改变后 batch size、有效更新数、显存失败、数据增强面积或优化难度
   不匹配，从而破坏“只比较空间分辨率”的问题？

### 5.2 manifest、split 与 test sealing

1. fit/gate/sealed-test 是否由 annotation 可确定地重建，是否可能通过路径、顺序、缓存、
   result 文件或重复 commit 绕过？
2. checkpoint 选择是否只使用 gate split，test 是否只能打开一次？global marker 的原子性、
   并发安全、跨 namespace/commit 等价重跑是否严格？
3. 是否有 validation/test GT、raw prediction、手填 epoch 或旧 artifact 泄漏到训练/选择？
4. evidence hash 是否绑定实际字节、语义 identity、code commit、config、checkpoint、prediction、
   evaluator、profile 和 marker？是否存在 TOCTOU、symlink、相对路径、先验文件覆盖漏洞？

### 5.3 训练正确性、AMP replay 与 DDP

1. 逐行追踪一次正常 update、一次 AMP overflow/retry、一次 checkpoint save。列出所有会在
   forward 中变化的 RNG、buffer、loss normalizer、optimizer、scheduler、EMA、GradScaler 和
   dataloader exposure 状态。
2. 当前 replay 是否状态精确：失败 attempt 后恢复哪些状态、保留哪些 scaler backoff、成功
   update 如何只推进一次？是否可能重复消费增强样本或改变 DDP collective 次序？
3. `copy.deepcopy` optimizer/scheduler/inference/post-processing config 的修复是否覆盖全部合法
   mutation callsite？是否掩盖了本应被识别的协议漂移？
4. c10d `127.0.0.1:0`、单进程 DDP、Slurm GPU 映射是否在并发 Job 下可靠？是否仍有固定物理
   GPU、覆盖 `CUDA_VISIBLE_DEVICES`、登录节点训练或跨 Job rendezvous 风险？
5. 训练合同写“skipped AMP update fails”而实现采用 same-batch replay 时，二者语义是否一致？
   必须区分 skipped attempt、successful optimizer update 和 skipped sample exposure。

### 5.4 checkpoint、评估与统计

1. sidecar/checkpoint 是否原子写入且互相绑定？异常终止能否留下看似合法的半成品？
2. eligible checkpoints 是否完整生成 gate raw predictions？selector 是否从 raw predictions 使用
   官方 evaluator 重算 `(mAP@0.6+mAP@0.7)/2`，并正确处理 exact tie earliest epoch？
3. 官方 evaluator parity 是否真的逐 class/IoU 一致，而非只比较聚合数字？
4. short-action、boundary error、paired video-cluster 与跨 training-seed hierarchical bootstrap 的
   统计单位是否正确？max-T 是不是单侧 simultaneous lower bound，比较方向和 pivot 是否正确？
5. cost 是否只用于 Pareto/resolution freeze，而不会污染 accuracy GO/KILL？代码和文档是否一致？

### 5.5 full-stack cost

逐行确认 profiler 是否真实测量：decode、preprocessing、H2D、完整 backbone、adapter、head、
跨窗口聚合/NMS、峰值显存、GPU energy。必须检查：

- p50/p95 是否来自每次完整 forward 的 total sample，而不是阶段 percentile 相加；
- warmup、同步、batch size、workers、硬件/软件 fingerprint 和 profile order 是否冻结；
- power sampler 的采样持续时间、边界、积分和 idle baseline 是否合理；
- 三个分辨率是否使用同一节点/硬件条件，失败重试是否会破坏可比性；
- source decode 与常驻高分辨率帧成本是否真的进入总成本。

## 6. corrected DUCA P0 必须逐行审核的问题

1. 独立重建四臂的唯一差异。exact-uniform、direct-a5、transition beta=0、counterfactual 是否
   共享相同 detector、loader、optimizer updates、LR progress、effective K、max-hole 和评估？
2. 对每个样本，hard 与 soft structured selection 是否共享相同的
   `valid_count/effective_k/max-hole` feasible family，短样本是否只做零填充而非先在 T=768 求解
   后掩码重归一？
3. exact-uniform 是否精确使用预注册 positions，而不是 all-equal logits 加 Viterbi tie-break？
   请给出 T=768/K=384 和 mixed-length 的位置证明。
4. counterfactual teacher 是否 train-only、no-grad、无推理泄漏？在 CUDA autocast 中是否禁用
   detached cast cache 污染主 detector 梯度？
5. detector loss 对 selector 的梯度是否只是 backward connectivity，还是与真实 hard one-swap
   utility 方向一致？代码是否错误地把 nonzero gradient 称为 detector utility supervision？
6. 粗分类二分类 actionness、状态变化、uncertainty 和 selector score 的实际数据流是什么？
   selector 是否看到了合理的粗特征，还是又退化为显式 GT boundary predictor/actionness top-k？
7. official ActionFormerHead/AdaTAD 的训练、assignment、decode 和 NMS 是否保持正确？selected-axis
   与 physical-time 是否造成 label/query geometry 错位？
8. `with_cp=False/static_graph=False/find_unused_parameters=True` 是否是当前动态参数使用图所需的
  最小正确协议？四臂 pilot 是否真正覆盖 full/mixed/all-short batch、schedule transition 和全部
   trainable parameter groups？
9. 当前 seed-0 formal suite 即使成功，能证明什么、不能证明什么？何种结果才允许追加 seeds、
   cost 或第二 detector；何种结果必须停止 DUCA？

## 7. 对当前现象的严厉解释要求

不得先给原因再找证据。请按以下顺序：

1. 列出已经被代码或 artifact 证实的现象；
2. 列出只有历史跨协议数字支持的现象；
3. 列出当前完全未知、必须等待 corrected run 的问题；
4. 对每个可能原因建立“机制 -> 可观测预测 -> 最小诊断 -> 反证条件”表；
5. 删除不能由现有代码或最小诊断区分的故事性解释。

重点回答：

- 为什么旧 learned sampling 约 63 与历史 uniform 约 65 不能直接证明联合训练失败？
- corrected P0 仍可能失败的最危险机制是什么？
- S1 的 dense224/256 即使 mAP 更高，什么情况下仍不能支持 Zoom 路线？
- S1 的 dense224/256 若无增益，是否应无条件停止 S2？是否存在代码/协议错误会造成假阴性？
- 当前两个 experiment-running 矩阵是否真的回答各自问题，还是只完成了工程闭环？

## 8. 审核下一步计划，而不是默认接受

当前拟议顺序是：

1. 完成 S1 3x3 训练并核验完整 evidence；
2. gate-only checkpoint selection；
3. 九个 selection 全部有效后一次性打开 sealed test；
4. 按冻结顺序完成测试和 full-stack profile；
5. 运行 paired/hierarchical statistics 与 cost-aware resolution freeze；
6. S1 GO 才实现 S2 oracle ROI sufficiency，S1 KILL 则停止空间 Zoom；
7. 独立完成 DUCA corrected P0 四臂裁决，未胜 matched exact-uniform 则停止扩展。

请逐项判断此计划是 `KEEP / MODIFY / DELETE`，并说明代码证据。尤其检查：

- 是否应在现有训练结束前停止某些 Job；
- 是否缺少会使全部结果无效的 P0 gate；
- test-open 与 profile 顺序是否会引入选择偏差；
- seed 数与统计模型是否匹配；
- S1 GO/KILL 是否真正由 accuracy headroom 决定，成本只负责冻结部署分辨率；
- DUCA seed-0 是否只能做决定性 pilot，而不能直接形成论文主表；
- 两条路线并行是否造成 GPU 资源浪费或研究叙事污染。

最后给出你认可的最小执行 DAG。每个节点写清输入 artifact、输出 artifact、pass/fail 条件、
失败后的唯一动作和预计 GPU/时间成本。不要提出“再调一下”“多试几个 loss”之类开放式路线。

## 9. 必须给出具体实现，不得只提概念

对每个确认的 P0/P1 问题，必须提供：

1. 根因和可触发的最小反例；
2. 精确 `file:line`；
3. 最小 patch，优先 unified diff；若无法给 diff，给完整函数级代码；
4. 为什么该 patch 不改变预注册科学问题；
5. focused unit test、integration test 和远端 precheck；
6. 旧 checkpoint/gate/Job 是否失效，哪些必须取消或重跑；
7. 修复后的 evidence schema 是否需要升级。

禁止重新造轮子。若 OpenTAD、AdaTAD、PyTorch、SciPy、官方 evaluator 或现有仓库 helper 已有
正确实现，优先复用并给出来源。不要为了“更优雅”重写官方 detector、NMS、AP evaluator、
DDP 或 GradScaler。

若没有发现 P0/P1，不要虚构修改。明确写 `NO CODE CHANGE REQUIRED`，然后只给 evidence
closure 计划和剩余 P2 风险。

## 10. 强制输出格式

按以下顺序输出，全文使用清晰中文，代码标识和公式可保留英文：

1. Skill Loading Certificate；
2. Repository Visibility Certificate；
3. 审核范围、已核验事实、未核验运行声明；
4. 一句话总裁决：`GO / HOLD / STOP_AND_FIX / KILL`；
5. P0/P1/P2 findings，按严重性排序，每项带 GitHub `file:line`；
6. S1 实际架构、训练、评估、test sealing 和 cost pipeline 重建图；
7. DUCA 四臂实际数据流、梯度流和 hard/soft feasible-set 重建图；
8. official AdaTAD/OpenTAD 一致性差异表；
9. 实验完整性审计：GT provenance、split、selection、result existence、dead code、scope；
10. 当前现象的证据分层与机制诊断表；
11. 对现有下一步计划逐项 `KEEP / MODIFY / DELETE`；
12. P0/P1 的 unified diff、测试和重跑影响；
13. 最小执行 DAG 与 GPU 成本优先级；
14. result-to-claim matrix：S1 和 DUCA 各种结果下允许、降级或禁止的 claim；
15. 模拟 CVPR/ICCV 审稿：Summary、Strengths、Weaknesses、Questions、Score、Confidence；
16. 最后用不超过 12 条给出确定执行顺序。

## 11. 最终红线

- 不得把 `tested`、gate pass 或短 pilot 写成 `empirically_supported`。
- 不得把正在运行写成完成，不得预测未出现的 mAP。
- 不得把历史 64.352/65.696 填入 corrected matched 表。
- 不得把旧失效 Job 当作修复后代码的证据。
- 不得把 S1 称为完整 Zoom 模型，也不得在 S1 GO 前设计成既成事实的 S2 policy。
- 不得把 DUCA 称为 Online TAD。
- 不得用 validation/test GT、teacher、oracle、cache 或 ledger 参与部署决策。
- 不得只看 selected count、config 名称或 nonzero gradient 就宣布 contract 成立。
- 不得在未验证真实完整成本时宣称高效。
- 不得给无文件路径、无测试、无失效影响分析的抽象建议。

你可以彻底否定任一或两条路线。真正需要的是可证伪、可执行、不会重复历史错误的裁决，
而不是让现有工作显得合理。

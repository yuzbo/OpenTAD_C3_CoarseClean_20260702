# DUCA `7525efb` 精确提交 Pro 严厉审核与最终路线裁决 Prompt

你现在不是项目顾问，而是 **CCF-A/CVPR 级方法审稿人、PyTorch 训练系统审计员、离线 TAD 专家和反方证明者**。请直接读取下面的 GitHub 精确提交，逐行核验实际代码，不要接受项目方的概括，不要因测试通过而默认方法正确，也不要先迎合“DUCA 应该成立”的结论。

## 0. 唯一审计对象与可见性纪律

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 分支（会继续移动，仅用于导航）：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-transition-only-20260711>
- **唯一有效代码快照**：`7525efb2e07214615a59c482443246174a6adaf1`
- 永久提交页：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/7525efb2e07214615a59c482443246174a6adaf1>
- 上游 AdaTAD/OpenTAD 对照：`sming256/OpenTAD@1aa8ca4ac5e846b1e8ff69298dd6607121a01589`
- 上游 ASFormer 对照：`ChinaYi/ASFormer@e1bbe4f3ed083748f91467c51a63ac2a8b9277ad`

先输出“可见性证书”：列出你实际打开的 commit、文件和行号。若无法读取精确提交，立即输出 `VISIBILITY_BLOCKED`，不得根据以下说明猜测代码。所有结论必须标记为以下四类之一：`CODE_FACT`、`REPO_ARTIFACT`、`USER_REPORTED_UNVERIFIED`、`PROPOSAL`。

## 1. 研究初心与当前目标

任务是 **离线 Temporal Action Detection**，不是 Online TAD、流式 TAD 或因果在线推理。模型在做选择前允许观察完整视频窗口，但应以低成本低分辨率模块为重型 backbone 分配计算。

最初思想不是让小模型直接精确回归动作边界，而是：

1. 用低成本二分类粗模型学习每个时间点的动作/背景状态；
2. 从动作概率、二分类不确定性和粗模型隐藏特征的状态变化中，间接推断语义转换与潜在边界；
3. 在严格预算和最大空洞约束下选择原始 RGB 帧；
4. 只让重型 TAD backbone 与检测头消费选中帧；
5. 训练时允许下游检测任务校准“哪些选择真正有用”，推理时禁止 GT、teacher、oracle、缓存预测和 ledger 决策。

请首先判断当前实现是否忠实于此初心，尤其检查：共享 ASFormer encoder 同时接收动作二分类、transition/endpoint 和下游 utility 相关梯度后，它还是“粗分类后间接边界定位”，还是已悄然变成一个弱边界网络；若存在矛盾，请给出最小且可训练的梯度隔离或参数共享方案，不要只建议“调 loss 权重”。

## 2. 需要你从代码独立核验的当前结构

项目方声称当前候选大致为：完整 `T=768` 离线窗口 -> 64x64 低分辨率 spatial stem -> 官方 ASFormer 二分类时序模块 -> `p_action` 与 encoder hidden -> `delta logit / abs delta / entropy delta / hidden delta / cosine change` -> transition-first scorer -> exact-`K=384`、`max_unselected_hole=15` 的共享 hard/soft structured DP -> 采集原始 RGB -> VideoMAE-S Adapter -> ActionFormerHead。

训练时，GT segment 生成 binary actionness 与 endpoint/transition target。另有 train-only hard one-swap teacher：对当前 hard selection 和最多 4 个可行 swap 分别运行官方 `cls_loss + reg_loss`，定义 `u = L_baseline - L_swap`；teacher 在 `no_grad` 中运行，推理时不存在。当前实现 **不是直接 detector-gradient estimator**，而是 detached detector-utility surrogate。请用实际调用图证实或否定以上每一项。

## 3. 必须逐行读取的文件

至少完整审查：

- `opentad/models/duca/counterfactual_utility.py`
- `opentad/models/selectors/duca_online_frame_selector.py`
- `opentad/models/detectors/actionformer.py`
- `opentad/models/duca/acquisition.py`
- `opentad/models/duca/structured_selection.py`
- `opentad/models/duca/transition_only.py`
- `opentad/cores/optimizer.py`
- `opentad/cores/train_engine.py`
- `configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py`
- `tools/bata/run_duca_transition_only_formal_full_model_gate.py`
- `tools/bata/run_duca_transition_only_p0_ddp_pilot.py` 及实际 suite/launcher
- 所有 `test_duca_counterfactual*`、`test_duca_online_coarse_probe*`、structured-selection、optimizer、AMP、DDP 和 formal-gate tests。

同时与上游 AdaTAD/OpenTAD 的 detector、VideoMAE Adapter、ActionFormerHead、loss normalizer、point generation、assignment、decode/NMS、base config 逐项比较。严禁用“官方 AdaTAD 完全未修改”这种笼统表述；请准确区分官方组件复用、wrapper 扩展、坐标适配、输入长度变化和源代码差异。

## 4. 对新 signed score-space proximal 的数学审判

当前代码宣称：令每个 swap 的 incidence 行为 `a_m = e_add - e_remove`，组成 `A`；中心分数为 `s`，pair score 为 `d=As`；将 detached utility 经每样本 mean-abs scale 与 `tanh` 得到 `u_tilde`；令 `G=A A^T`，`v=G^{-1}u_tilde`，`d*=stopgrad(d)+eta v`，并优化

`L_b = (1 / (2 M_b)) ||d-d*||^2`。

请从代码和微分两侧独立证明或构造反例：

1. `A(-grad_s L_b)` 是否严格与每个 signed utility 同号；batch mean、loss weight、GradScaler、AMP cast、DDP 后比例是否仍为正；
2. shared-remove、不同候选数、all-positive、all-negative、mixed-sign、zero utility、all-short、masked candidate、重复 add/remove、奇异/病态 Gram 是否正确；
3. 候选生成是否确实保证 unique add、合法 remove 与 `cond(G)<=5`，还是 gate 只在特定随机样本中自证；
4. `score_space_utility_alignment()` 是否独立审计真实梯度，还是使用同一公式制造必然通过的循环证明；
5. `target=stopgrad(current)+shift` 本质上是否只是自定义局部梯度。它能否学习稳定的跨样本策略，还是每步都追随噪声很大的局部 policy improvement；
6. mean-abs + tanh 是否不恰当地抹去 utility 量级与跨视频可比性；temperature 和 step size 的语义是否可识别；
7. 只从当前 scorer 的 top unselected 与 low-score removal 构造最多 4 个候选，是否造成严重自确认与探索盲区；
8. 若该目标不够优雅或不稳定，请在 baseline-anchored logistic、signed regression、listwise/no-op ranking、policy gradient、implicit/influence approximation、local-cell counterfactual 中进行严格比较，并只选择一个最终推荐，不要罗列概念。

## 5. 监督、梯度与坐标语义审计

请输出“模块 x loss”的梯度矩阵，至少包含 spatial stem、ASFormer encoder/decoder/action head、transition scorer、structured selector、VideoMAE Adapter、projection、ActionFormerHead。每格标明：直接梯度、ST/代理梯度、detached teacher 监督、明确 stop-grad 或无路径，并给出 `file:line`。

重点回答：

- 粗分类器是否真的主要由 binary actionness 学习，transition/endpoint 是否污染了“间接定位”设定；
- selector 是否看到了粗模型的有效隐藏语义，而不只是 `p_action` 曲线；这些 hidden 是否绝对特征泄漏或只使用状态差分；
- 主 detector loss 是否能影响 hard selection 决策；若只能通过 detached swap utility 影响，请直说，不得称“下游梯度直接反传”；
- baseline 与 swap 改变 selected-axis 后，GT remap、point assignment 与 loss normalizer 是否完全可比；utility 是否混入了坐标扭曲或 target-remap 伪差异；
- hard train、soft surrogate、inference decoder 是否属于同一 exact-K/max-gap 可行族；short/mixed window 是否一致；
- `max_unselected_hole=15` 是真正的硬结构约束还是事后 repair；是否会吞掉 scorer 的自由度并使 learned policy 退化为近似均匀；
- 推理是否真正 teacher-free、GT-free、cache-free；完整窗口可见性只能称 offline full-window acquisition。

## 6. AMP、DDP、状态恢复和 gate 完整性

逐行审查 FP32 proximal、外层 autocast、`autograd.grad` inside forward、graph retention、显存、DDP hook 与 unused-parameter 行为。验证 teacher 是否无副作用地恢复 Python/NumPy/Torch CPU/CUDA RNG、所有非 selector buffers、训练模式和 ActionFormer loss normalizer。

当前 gate 已要求 clean tree 并记录核心文件 SHA，但它仍明确写入 `input_provenance=deterministic_synthetic_contract_probe` 与 `real_dataset_loader_executed=False`。请严厉判断：随机 RGB + 人工 GT 的 gate 最多能证明什么，绝不能证明什么；给出一个使用真实 THUMOS train loader、真实 GT、full/mixed/all-short batch、outer AMP、实际 optimizer/scaler/EMA/schedule replay 的最小 exact-commit CUDA gate 实现。不得通过降低阈值、关闭 fail-closed 或删掉异常样本来让 gate 通过。

同时检查 `with_cp=False + static_graph=False + find_unused_parameters=True + world_size=1` 与官方 AdaTAD 的差异、必要性和成本。确认 132 epoch 是否等价于精确 13,200 次 **成功** optimizer/LR/EMA/selector-schedule 更新；AMP skip 是否同 batch、同 RNG、同 mutable state 重放；checkpoint/resume 是否保持该合同。

## 7. 当前证据边界（不得混淆）

- `CODE_FACT/REPO_ARTIFACT`：精确提交为 `7525efb...`，代码声明 paper/deployment claim 不允许。
- `USER_REPORTED_UNVERIFIED`：clean remote focused tests 为 `160 passed, 7 skipped`；CUDA-only 测试在非 GPU 登录节点跳过。
- `USER_REPORTED_UNVERIFIED`：前一提交 `a6903ae` 的 exact gate Job `1165646` 因 utility-direction gate 失败；`1165650` 是 shell 失败，`1165654` 已取消，均不是性能证据。
- 当前提交没有 exact-commit CUDA gate、真实 loader gate、pilot、full train、mAP 或成本结果。
- 历史 learned transition 约 64.34、uniform 64.352/65.696、oracle 约 78 只能作为协议不匹配或 privileged ceiling 背景；不得放进当前 matched 主表，不得推导当前方法已接近或超过 uniform。
- 旧 Jobs `1164700-1164703` 因成功更新数不足已失效，只能作为诊断。

## 8. 请裁决最终模型，而不是继续堆补丁

在完成代码审计后，请明确选择：

1. 保留当前“binary state -> transition evidence -> structured fixed-K -> signed hard-swap utility”的主线并给出最终简化版；
2. 保留 transition-only，但删除 counterfactual teacher；
3. 改为局部 cell deformation/coverage-preserving policy；
4. 改为另一种可证明且更简单的 detector-aware 监督；
5. 若根本假设不成立，直接 `KILL DUCA`，说明智能选帧为何无法从 uniform 65 走向 practical dense/oracle ceiling。

最终推荐必须同时满足：离线全窗口语义正确；训练/推理决策同构；粗模型低成本；边界覆盖优先；exact budget/max-gap；detector-aware 但无推理泄漏；总推理成本可审计；对 uniform 的增益可被同协议实验归因。不要用 X3D、SlowFast、MUST、动态预算、第二检测头或 physical-grid 转移 fixed-384 核心问题。

## 9. 最小且可证伪的实验闭环

请先判断以下四臂是否足以回答当前 claim，并修正不合理处：同 commit/seed/data/order/成功更新数/EMA/evaluator 的 exact-uniform、transition beta=0、transition + signed utility，以及 direct-boundary 归因臂。另设一个不运行 probe 的 bare-uniform 只用于真实成本基线，不能与 matched attribution control 混为一谈。

主 claim 仅允许：`C3: transition policy > matched exact-uniform`；`C4: signed detector utility > transition beta0`；`C7: probe+selector+稀疏 heavy stack 的端到端推理成本低于 dense stack`。为每个 claim 给出最小 seeds、主指标、high-tIoU/short-action分层、边界 recall/distance、max-hole、candidate utility sign/rank、训练和推理成本、失败阈值与 preregistered GO/HOLD/KILL。最终 checkpoint 必须预先固定，禁止用 THUMOS test 中间 mAP 选 epoch。

只有 fixed-384 在真实 gate、pilot 和 matched seed-0 上成立后，才允许扩展更多 seed、预算或检测头。若它不超过 exact-uniform，应给出停止规则，而不是继续调参。

## 10. 新颖性与可发表性

请检索并引用可核验的一手论文/官方代码，至少与 AdaFrame、Action Sensitivity Learning、AdaTAD、TE-TAD、TAPS、Progressive Block Drop 及近期 task-aware/adaptive video computation 比较。回答贡献究竟是新的学习问题、结构化 acquisition、detector-utility 学习，还是多个已有组件的工程组合。指出最强 novelty attack、最强 methodology attack、最强 efficiency attack，并判断在何种实证门槛下才可能达到 CVPR/CCF-A，达不到时应降为什么论文主张。

## 11. 强制输出格式

按以下顺序输出，不得省略：

1. **可见性证书与证据分级**；
2. **一句话总裁决：GO / HOLD / REDESIGN / KILL**；
3. **真实 forward/train/inference 调用图**；
4. **模块-loss-梯度矩阵**；
5. **P0/P1/P2/P3 逐行发现表**：每项含 `file:line`、复现条件、影响、最小修复；
6. **signed proximal 完整数学推导或最小反例**；
7. **官方 AdaTAD 差异表与坐标/GT-remap 审计**；
8. **当前最终架构裁决**：保留/删除/替换哪些模块及唯一推荐理由；
9. **可直接落地的核心代码 patch**：不是伪代码，附对应单测与 CUDA/real-loader gate；
10. **严格实验 DAG 与 preregistered result-to-claim 表**；
11. **计算成本与新颖性裁决**；
12. **最终执行合同**：列出再次训练前必须全部满足的检查项。

任何“可能、建议、看起来”都必须紧跟可证伪测试。不要把 proposal 写成 implemented，不要把 test 写成 experiment，不要把 diagnostic 写成 evidence。若你认可当前 proximal，也必须先给出它最强的失败反例；若你否定它，也必须提供一个更简单且能落到现有代码调用图中的替代实现。

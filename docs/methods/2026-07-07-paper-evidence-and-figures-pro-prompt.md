# Pro Review Prompt: Paper Evidence, Figures, and Missing Analyses

请你作为 CVPR/ICCV/NeurIPS 级别的严格审稿人和研究合作者，基于公开 GitHub 仓库进行逐行代码审查、实验逻辑审查和论文证据链设计。

仓库：

- GitHub: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- 分支: `codex/gas-vt-stage23-detector-aware-20260706`

背景目标：

我们研究的是 temporal action detection (TAD) 的 pre-backbone sparse frame acquisition/selection。核心问题是：在预算不超过 384 帧的条件下，能否通过一个学习式、任务感知的 selector 选择少量关键帧，再接 AdaTAD/OpenTAD detector，取得优于 uniform sparse sampling 的 detector mAP，同时保护高 IoU 定位能力。当前路线包含：

1. `PAction learned`：低成本粗分类/间接信号训练 acquisition policy，再生成 strict ledger 接 AdaTAD。
2. `GAS-VT`：gap-aware sequential state、CVaR max-hole、boundary bracket、action interior 等规则化 selector。
3. `PAction lattice / replacement`：用同一 PAction 分数源，在 384 预算内约束/重分配选帧，作为诊断性 decoder 实验，不应被过度包装为最终智能方法。
4. `Stage2 detector-aware selector`：dense AdaTAD teacher 导出 train-only point/responsibility/utility，用 detector utility 训练 selector。
5. `Stage3 TrueTime joint selector + AdaTAD`：ST/hard selector 与 AdaTAD 在同一训练图中联合优化。
6. `Stage4 curriculum/bilevel`：dense teacher -> selector pretrain -> sparse detector -> joint fine-tune，防止 selector collapse 和高 IoU 崩溃。

请重点审查这些新增/关键文件：

- `tools/bata/analyze_selector_geometry.py`
- `tools/bata/plot_selector_geometry.py`
- `tools/bata/plot_selector_timeline.py`
- `tools/bata/plot_selector_dashboard.py`
- `tools/bata/export_selector_paper_tables.py`
- `tools/bata/validate_selector_geometry_metrics.py`
- `tools/bata/generate_selector_failure_gallery.py`
- `tools/bata/export_dense_adatad_teacher_points.py`
- `tools/bata/export_adatad_responsibility_utility.py`
- `tools/bata/train_detector_aware_acquisition_policy.py`
- `tools/bata/run_truetime_joint_selector_precheck.py`
- `tools/bata/run_paction_lattice_replacement_ledger_pipeline.py`
- `tools/bata/generate_uniform_sparse_ledger.py`
- `configs/adatad/thumos/*c3*`
- `tests/test_analyze_selector_geometry.py`
- `tests/test_plot_selector_geometry.py`
- `tests/test_selector_geometry_outputs.py`
- `tests/test_adatad_responsibility_utility.py`
- `tests/test_paction_training_loss_terms.py`

请回答以下问题，要求严厉、具体、可执行：

1. 当前实现是否足以支撑一篇论证完整、逻辑自洽的论文？如果不能，缺的是代码、实验、分析还是叙事？
2. 当前 paper claim 应该如何收敛？哪些 claim 可以说，哪些必须 HOLD？
3. 为证明 sparse pre-backbone selector 真有用，至少需要哪些主结果表？请列出每个表的行/列、数据来源、判断阈值。
4. 为解释为什么 `PAction learned` 比 `GAS-VT` 更好，需要哪些结论图？请覆盖：
   - mAP 曲线分解：Avg / tIoU 0.3 / 0.5 / 0.7
   - selected frame region share：background / boundary band / action interior
   - boundary distance CDF
   - per-action instance boundary recall
   - max/p95 unselected hole by region
   - selected-frame 与 PAction score / teacher utility / detector responsibility 的校准曲线
   - failure gallery：GAS-VT 失败、PAction 成功、PAction 失败的典型样本
5. 当前 `analyze_selector_geometry.py` 的坐标约定是否正确？请检查 dense index、half-open GT segment `[start,end)`、end boundary `end-1`、frame-center distance、selected index 映射、sample_id/video_id/grouping 是否存在潜在 off-by-one 或泄漏。
6. 如何避免把 action-positive coverage 误当成 boundary coverage？还需要加入哪些统计字段和测试？
7. 如果目标是超过 uniform sparse 384 的约 65 mAP 锚点，最短、最可靠的实验路径是什么？请给出排序后的实验队列，而不是泛泛建议。
8. 对 Stage2 detector-aware utility，teacher utility 应该如何定义才最合理？请比较 point responsibility、cls/reg loss、saliency、counterfactual deletion utility，并给出推荐实现。
9. 对 Stage3 joint training，如何让 detector loss 真正回传到 selector，同时避免 selector collapse、duplicate selection、large holes 和高 IoU mAP 崩溃？请给出关键 PyTorch 代码框架。
10. 当前 `PAction lattice / replacement` 是否只是工程性 post-processing？它在论文中最多能作为什么角色？如果不能作为主贡献，应如何改造成可学习的 spacing regularizer 或 differentiable decoder？
11. 请给出完整实验计划：baseline、ours、ablation、negative control、seed、budget、GPU/epoch 设置、precheck、fail gate、claim gate。
12. 请给出关键代码补丁或伪代码：数据导出、metric 聚合、plot 生成、selector loss、ST top-k/relaxed top-k、gap-aware differentiable penalty、detector-aware utility loss、joint training step。

输出格式要求：

- 第一部分：总体判定 `PASS/WARN/HOLD/FAIL`。
- 第二部分：必须补齐的分析和结论图清单，每项包含目的、数据源、脚本、期望结论、失败解释。
- 第三部分：代码审查发现，按 P0/P1/P2 排序，必须引用具体文件和逻辑。
- 第四部分：实验路线图，分 immediate / 1-week / 4-week / paper-ready。
- 第五部分：关键实现代码，尽量给出可直接落地的 Python/PyTorch 片段。
- 第六部分：如果最终目标是 CVPR 论文，请明确论文故事：问题、洞察、方法、贡献、风险、最小可发表证据。


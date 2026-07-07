---
updated: 2026-07-08
status: active
scope: 记录并吸收 Pro/GPT 对 commit 52ab63b selector geometry visualization suite 的 HOLD 审查
out-of-scope: 不声明当前分析图已支撑论文 claim；不把 Stage2、Stage3、lattice 视为 completed evidence
---

# 52ab63b 可视化审查吸收记录

## 原文归档

- 原始审查文本已归档到 `docs/methods/reviews/2026-07-08-52ab63b-visualization-hold-review-raw.txt`。
- 审查固定对象为 commit `52ab63bfb5ca8187cf5e4e913e9e17153802ea06`。
- 可见性结论：固定 commit 可见；但 GitHub branch commits 页面曾仍显示旧 HEAD，因此后续讨论必须锁定固定 SHA 或确认 branch HEAD。

## 总体吸收

本轮结论为 **HOLD**。阻塞原因不再是“缺少分析脚本”，而是当前图表和导出链路仍有 schema、统计口径、证据链与 claim 边界问题。现有结果最多能说明 PAction learned 在若干几何统计与 mAP 上优于 GAS-VT fixed384；还不能证明“覆盖几何导致 proposal quality 改善并最终提升 high-IoU TAD”。

最关键的吸收是：论文级证据必须从 `coverage geometry -> proposal quality -> high-IoU mAP` 建立实例级链路，而不是只展示 Avg-mAP 与少量 whole-video 几何统计的相关性。

## 必须修正的问题

1. `analyze_selector_geometry.py` 需要补强核心 contract：`valid_len == 768`、`selected_count <= 384`、`target_len == 384`、annotation unit、一致的 method/budget/grid 元数据，以及 action-local hole 统计。
2. `plot_selector_timeline.py` 和 `plot_selector_dashboard.py` 必须识别 analyzer 的真实 selected 字段，如 `selected_position`，避免生成空但看似成功的图。
3. `plot_selector_geometry.py` 需要修正字段别名和数据源：边界距离、hole boxplot、region stacked bar 均要对齐 analyzer 输出 schema。
4. `plot_selector_paper_summary.py` 不能用 `or` 处理 coverage fallback，避免把合法 `0.0` 当成缺失；缺失 mAP 不能画成 0-height bar，也不能按最差 score 着色；lattice diagnostic 必须和正式方法分开。
5. `export_selector_paper_tables.py` 需要按 radius pivot boundary recall，修正 `holes_by_region.csv` 的 schema 匹配，并避免把 in-progress Stage2/Stage3 当成 evidence table。
6. `generate_selector_failure_gallery.py` 目前仍偏索引生成器，不是论文级 failure gallery；需要接入真实实例、proposal/detector 结果，并按 completed/diagnostic/in-progress 过滤。
7. `validate_selector_geometry_metrics.py` 必须检查 schema 兼容性、method category、missing mAP policy、grid/budget/radius 口径、action-local hole，不应在缺少 selected 字段时默认 selected=True。
8. 当前测试过于 toy schema 化，必须增加从 analyzer 输出到 table/plot/validator 的端到端 schema 测试。

## 后续证据链

下一轮实现应新增或修正以下 paper-facing artifact：

- `method_registry.json`：统一标记 completed、diagnostic、in-progress、baseline、budget、grid、commit、checkpoint、结果可用性。
- `selector_geometry_per_instance.csv`：每个 GT instance 的边界支撑、interior 支撑、action-local hole、selected density。
- `detector_match_per_instance.csv`：每个 GT instance 的 best proposal tIoU、boundary error、matched score、是否命中高 IoU。
- `map_tiou_decomposition.csv`：按 tIoU threshold 展开 mAP，而不是只看 Avg-mAP。
- `normalized_action_time_density.csv`：把动作归一化到 0..1，显示 selected density 是否集中在 boundary、interior 或 background。
- `compute_quality_summary.csv`：把 FLOPs/observation count/latency 或 proxy cost 与 mAP 形成 Pareto 证据。

对应图应优先包括：mAP@tIoU decomposition、boundary error CDF、best proposal tIoU CDF、normalized action-time density、compute-quality Pareto、deterministic representative gallery。

## Claim 边界

目前可以说：

- 当前 commit 已有 analyzer、table exporter、基础图、timeline/dashboard/failure-index、validator 与 focused tests。
- PAction learned fixed384 在已知实验中优于 GAS-VT fixed384，并且 dense AdaTAD 是上界锚点。
- 当前系统更准确定位为 AdaTAD/OpenTAD 768 temporal grid 上的 sparse temporal acquisition / pre-detector selection。

目前不能说：

- 不能声称已经证明几何覆盖因果提升 detector quality。
- 不能声称这是严格 raw-frame pre-backbone 方法。
- 不能把 Stage2、Stage3、lattice 当作 completed leaderboard evidence。
- 不能只用 Avg-mAP 支撑 high-IoU 定位改善。
- 不能把缺失 mAP 画成 0 或按最差颜色处理。
- 不能把 whole-video p95 hole 当作 action-local localization 证据。
- 在 uniform、random、p_action top-k 等 matched sparse baselines 跑完之前，不能声称超过强 sparse baseline。

## 当前状态

本轮审查已完成本地归档与吸收。代码层修复尚未完成；论文 claim 维持 HOLD。下一步应先修正 schema/validator/plot/table 的一致性，再补 per-instance detector matching 和 high-IoU 证据链。

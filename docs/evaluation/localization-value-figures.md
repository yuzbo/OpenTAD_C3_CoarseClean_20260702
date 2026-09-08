---
updated: 2026-09-08
status: specified-data-and-some-producers-missing
scope: 四张论文证据图的数据要求和解释范围
out_of_scope: 虚构曲线、attention 替代因果收益、声称已实现诊断
---

# 四张定位价值证据图

当前已有选择、预算、曲线及 Pareto 绘图框架；以下新增干预数据的生产器尚未实现。这里只交付规格，不生成带科学数值的示意结果。原 117 个诊断不能因本文件而记为完成。

所有图使用可追溯训练 ID、checkpoint 身份、协议、视频/窗口、种子及实际成本；缺失项显示“未测”，不填0。收集分析干预使用训练视频或预先固定的诊断窗口，不能由测试集反复调方法。全量测试用于既定最终评价；其事后分析清楚标注用途。定性视频在查看方法成绩前冻结选例规则。

## 图一：分类收益与定位收益

每行数据对应同一基准计划的一次合法干预：

    train_id, checkpoint_identity, video_id, window_id, intervention_id,
    base_plan, alternative_plan, action_kind, donor_id, candidate_id,
    affected_layers, actual_support, cls_loss_before, cls_loss_after,
    loc_loss_before, loc_loss_after, delta_heavy_macs, model_mode,
    rng_policy, annotation_region

横轴 ΔL_cls，纵轴 ΔL_loc；保留正负象限、密度及样本数。acquisition 与 swap 分面，避免把预算增加与等成本替换混为一组。各模型同时导出共同的标准检测 loss 分解；新 interval risk 另列，不能因归约尺度不同制造视觉差异。

动作核心/边界邻域/外部上下文标签依赖真实 GT 与预定距离规则；无相应标注为 NA。需要直接报告路由对“分类收益小、定位收益大”候选的覆盖，才能讨论系统性遗漏。未出现稳定分歧时收回这一动机，不挑选少数好看点。

## 图二：计算位置到受益位置

行是干预 a，列是原始物理时间上的检测位置 q，颜色为有符号 U_(a→q)。多尺度 head 的 q 包含 level/native coordinate；不同尺度先分别展示，再按明确规则聚合，不能直接混成一个 rank 轴。

    intervention_id, action_kind, affected_layers, source_support,
    output_level, output_native_id, output_time_s, output_valid,
    unreduced_cls_before, unreduced_cls_after,
    unreduced_loc_before, unreduced_loc_after, reduction_weight

采用同 checkpoint、相同基准计划及确定性干预状态；记录未匹配 GT 和背景。验证加权逐位置差分能重建对应标量 loss 差，才能用这张图解释传播。图上可叠加原始选择位置、parent 边界及 GT 区间，不能用 selected rank 重新排列物理时间。

这反映当前模型在指定计划下的收益传播，不能单凭远距离亮区断言全由 TIA 引起：attention、检测头也会传播影响。若要归因 TIA，需要独立的受控传播实验。attention 权重不替代此差分，也不要求推理阶段计算整张矩阵。

## 图三：定位质量与计算预算分布

预算曲线分面展示 Avg-mAP、高 tIoU AP、短动作召回、起止误差与漏检率。横轴首先为实际 Heavy MAC，另给总延迟版本；标注 A0/A1/A2/M0–M3、fixed/dynamic、目标预算与实际分布。不同 checkpoint 的点不得连成同一运行的动态预算曲线。

    train_id, checkpoint_identity, design_id, selection_rule,
    tiou_thresholds, ap_per_tiou, avg_map, short_definition,
    short_recall, start_error, end_error, miss_rate,
    error_matching_rule, video_count, window_count,
    per_window_heavy_macs, per_layer_parent_token_counts

同时画每窗口预算直方图或 ECDF、parent 内计算集中度、全粗/局部零执行比例；按动作时长/实例数分组展示成本与错误。窗口级 loss、recall 可与窗口成本关联，数据集 AP 不能假称“每窗口 mAP”。

没有匹配预测的实例不能从报告中消失：条件边界误差与漏检率并列，任何合并惩罚事先定义。短动作阈值从训练数据或已冻结协议获得。seed0 不生成种子方差；视频配对 bootstrap 如采用，仅说明给定权重的数据采样不确定性。

## 图四：总延迟 Pareto 与开销分解

一张硬件面板对应同 GPU 类型、测量边界、batch、窗口集合和实现设置。坐标为完整测量边界的实际 p50 延迟与同 checkpoint 的正式精度；附 p95、吞吐、VRAM 和成本分布。不同 GPU 分面，不给4090与A100混合排名。

    train_id, checkpoint_identity, model_source, resolved_config,
    gpu_uuid, gpu_model, precision, batch, window_ids, timing_boundary,
    latency_samples_ms, p50_ms, p95_ms, windows_per_second, peak_vram,
    router_scout_time, heavy_time, tia_time, packing_time,
    transfer_time, detector_postprocess_time, decode_transform_time

分解优先使用真实 profiler；存在并行重叠时组件时间不能简单相加冒充端到端时间。device model、decoded tensor→output、encoded video→output 分开命名，未测解码不得称包含解码。每窗口 batch 延迟不冒充整段长视频全窗口及最终合并时间。

固定预算同 parent swap 的理论等 MAC 仍须实测 latency。动态预算同时给 p95 和 packing/padding 分布，以判断前置规划是否有系统价值。没有具体权重配对的 hardware receipt 不进入 Pareto。

## 与现有选择可视化的衔接

已有 native 选择图用于回答“选了什么”。在其旁边显示实际源时间支持、原图 ROI/变换合法性、GT 和预测、每 parent/每层计算数，再链接上述收益图，才能区分“看起来合理”与“对检测确有贡献”。B 的 anchor support 不等于完整编码感受野；旋转/剪切图须使用正确多边形映射，未恢复几何标为 NA。

新图的数据生产器、真实数据及成图状态分别登记；交付本规格只完成设计，不表示已准备好所有论文实证图。

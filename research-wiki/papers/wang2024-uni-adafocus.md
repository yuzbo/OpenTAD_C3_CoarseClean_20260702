---
type: paper
node_id: paper:wang2024-uni-adafocus
title: "Uni-AdaFocus: Spatial-temporal Dynamic Computation for Video Recognition"
authors: ["Yulin Wang", "Haoji Zhang", "Yang Yue", "Shiji Song", "Chao Deng", "Junlan Feng", "Gao Huang"]
year: 2024
venue: "IEEE TPAMI / arXiv"
external_ids: {arxiv: "2412.11228", doi: null, s2: null}
tags: ["video-recognition", "spatial-cropping", "dynamic-compute"]
added: 2026-07-13
---

# Uni-AdaFocus

## One-line thesis

先用轻量全局编码器观察完整视频，再由 policy 定位任务相关 patch，交给高容量局部网络；
并统一扩展到时间和样本级动态计算。

## Reusable Ingredients

低分辨率全局观察、局部高分辨率 heavy branch、全局特征复用、时间平滑 patch 轨迹、
feature-space policy optimization 与现成 TSM/X3D backbone 兼容性。

## Limitations for This Project

主要监督与证据来自视频分类，不直接保护 TAD 边界、高 tIoU 或多人并发事件。其动态
丢帧与 early exit 也不属于 dense-time spatial zoom 的目标。原样搬移只能作为 baseline。

## 2026-07-21 official-code audit

- The official ActivityNet/FCVID/Mini-Kinetics code uses ImageNet-pretrained
  MobileNetV2 as the global CNN and explicit optimizer multipliers: temporal
  policy `0.2`, spatial policy `0.2`, global CNN `0.5`, classifier `20`, local
  CNN `1`.
- Hard temporal indices are sampled from `weights_T.detach()`. The heavy local
  branch therefore does not directly differentiate through discrete frame
  indices. The temporal policy instead receives an auxiliary Monte-Carlo task
  classification loss.
- Training also pairs learned spatial crops with random crops through the same
  local CNN. This provides task-supervised input diversity, but it is not an
  exact reproduction of DUCA's one-pass temporal uniform companion.
- The transferable lesson for DUCA is component-specific learning speed and an
  auxiliary task objective for the policy. It does not justify claiming direct
  heavy-branch gradient through hard sampling, nor does it require MobileNet in
  the main method before a cost-matched P0 comparison.

## 2026-07-21 deeper method and coverage audit

- 时间策略不是普通 top-k。官方 `policy_sample_indices` 先把归一化权重累积为
  CDF，再用 K 个分位点做逆 CDF 采样；训练时分位点带随机扰动，测试时固定在
  每个概率质量区间的中点。碰撞修复只保证索引严格递增、互不重复。
- 因而它保证的是“在学习到的概率质量上恰好取 K 个有序样本”，不是物理时间上的
  均匀覆盖、最大空洞约束或边界召回保证。权重高度集中时，重分支帧仍可聚集并留下
  长时间空洞；权重均匀时，该采样才近似均匀时间采样。
- 论文系统真正的覆盖安全网是两部分：轻量全局网络先看均匀抽取的全局帧；最终分类
  同时复用全局特征和局部重特征。官方消融报告全局特征复用带来约 0.8--2.8 个点的
  分类收益。它不是只把全局网络当一次性 selector。
- `MCSampleFeature` 用无放回蒙特卡洛近似“按策略采样后的期望分类特征/损失”，给
  temporal policy 提供可微任务监督。真实重分支使用的硬时间索引仍来自
  `weights_T.detach()`，所以“联合训练”不等于检测/分类损失穿过离散索引。
- 官方 `PoolingClassifier` 也不直接相加 global/local raw features：两支分别经过
  独立 MLP 与累积 max pooling，local classifier 再拼接 pooled local 与 pooled
  global 表示。协调来自最终分类损失和各分支辅助损失，而不是强迫两种表征同空间。
- 空间策略同样避免直接依赖像素裁剪梯度：实际 crop action 被 detach，策略主要用
  深层特征插值的辅助目标训练。可迁移原则是稳定的任务代理与明确梯度归属，而不是
  宣称硬决策端到端可微。
- 论文 ActivityNet 指标是视频级多标签分类 mAP，不是 TAD temporal mAP；其数值不得
  与 THUMOS14 AdaTAD Avg-mAP 横向比较。其跨 TSM/X3D 和多数据集结果证明的是视频
  识别适配性，也不是 detector-agnostic TAD 插件证据。

## Relevance to current DUCA V8

- 支持当前 V8 的 P0 组件学习率裁决：官方实现也让 temporal/spatial policy 使用
  `0.2` 倍学习率、global CNN 使用 `0.5` 倍、local CNN 使用 `1` 倍，并让最终分类器
  更快学习。它直接说明 coarse evidence 与 selector 不应默认同速竞争。
- 支持“受保护梯度归属”：动作二分类和状态表征先由显式监督学稳；下游任务反馈通过
  专用策略代理或受限 scorer 更新，而不是无约束改写粗分类 trunk。
- 不支持把其 inverse-CDF sampler 当作 DUCA 的边界覆盖机制。TAD 仍需状态转变、
  边界和短动作端点监督，以及可审计的物理时间覆盖合同。
- 最有价值但尚未实现的后续启发，是让低成本 dense coarse sequence 也进入最终 TAD
  融合，用它保持全时序上下文，再让稀疏 VideoMAE/重 backbone 特征承担边界附近的
  高成本细化。这样才可能放宽重分支 max-hole，而不把 detector 的全部信息押在稀疏
  selected-axis 上。
- 该融合属于 V8 终局后才可裁决的有界后继，不得在 Job `1178989` 运行期间改写为
  新 selector、decoder 或模型版本。

## Source

- Paper: https://arxiv.org/abs/2412.11228
- Code: https://github.com/LeapLabTHU/Uni-AdaFocus

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。

## 2026-07-22 粗扫描粒度复核

- Uni-AdaFocus 并不把原视频的每一帧都送入网络。官方常用设置先从整段视频均匀形成
  48 个候选位置，只用其中 16 个位置运行轻量全局网络，再把策略权重插值到 48 个
  候选位置，并用逆累积分布抽取 16 个位置运行重网络。
- 当前 DUCA 也不是逐原始帧扫描。THUMOS 配置中的候选网格相邻点约隔 4 个源视频帧，
  一个训练窗口共有 768 个低分辨率候选点；粗扫描在张量中批量运行，而不是 Python
  循环逐帧调用。但当前数据管线仍先解码、缩放并搬运全部 768 个候选点，因此只能
  严格声称减少了重 VideoMAE backbone 的处理帧数，不能声称解码或总输入成本按 K 缩减。
- Uni-AdaFocus 的逆累积分布采样保证固定数量、时间有序并覆盖所学概率质量，但不保证
  TAD 所需的物理最大间隔、短动作覆盖或起止边界召回，不能直接替换 DUCA 的边界合同。
- V8/R 系列终局之后的有界成本实验应保持重分支预算 K、selector 和 detector 不变，
  单独比较：现有密集候选粗扫、降低候选频率、降低频率后在状态转变/不确定性峰值附近
  局部补扫。主指标必须是官方 THUMOS mAP 与完整端到端成本，边界召回只能作为诊断。
- 该实验不得回退到 local-cell 或每格一帧。最终重帧分配仍应允许跨区域转移预算，并在
  动作起止边界附近形成 Oracle 式多帧微簇。

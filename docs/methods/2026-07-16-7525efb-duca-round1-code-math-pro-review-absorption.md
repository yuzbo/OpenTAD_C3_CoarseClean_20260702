# DUCA `7525efb` 第一轮代码/数学 Pro 审计吸收记录

## 来源与完整性

- 原附件：
  `C:/Users/skywalker/.codex/attachments/52c99bc3-7775-40d9-91e1-df9f6c819e2b/pasted-text.txt`
- 字节一致归档：
  `docs/methods/reviews/2026-07-16-7525efb-duca-round1-code-math-pro-review-raw.txt`
- 原文与归档长度：`16034` bytes
- SHA-256：
  `DA4201C2D947C81EE6A799EF8B4572AD3D9C11DF047E29F8B70A9462B475F4C1`
- 审计对象：
  `7525efb2e07214615a59c482443246174a6adaf1`
- 永久提交页：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/7525efb2e07214615a59c482443246174a6adaf1`
- Reviewer visibility：`VISIBLE`。

## Reviewer 裁决

Reviewer 返回 `GO_TO_REAL_GATE`，不是 `GO_TO_PILOT`、`GO_TO_FULL_TRAIN`
或论文方法通过。其含义严格限定为：没有发现必须先改模型目标、梯度路由或训练事务的
P0 静态实现错误；该提交已经成为一个可由真实 THUMOS/CUDA gate 证伪的 fixed-384
候选。

`HANDOFF_PACKET` 的关键事实为：

- `direct_detector_gradient=false`；
- `train_inference_isomorphic=conditional`；
- `official_adatad_fidelity=official_components_with_wrapper`；
- `signed_proximal_math=pass`；
- `coordinate_utility_validity=pass`，但只指当前策略效用在自身坐标合同内可比较；
- `amp_ddp_contract=unresolved`；
- `real_loader_gate_exists=false`。

## 已核实的方法真相

1. 当前方法是离线全窗口 pre-backbone 选择，而不是 Online TAD。完整低分辨率窗口
   经过 spatial stem 和官方 ASFormer 路径，生成 actionness、hidden 差分和
   transition evidence，再进行 exact-`K=384`、最大未选空洞不超过 15 的结构化选择。
2. 二分类 actionness head 学习动作/背景状态；transition/endpoint 目标会更新共享
   encoder 和 scorer。因此不能把整个低成本模块描述为“只受二分类监督”。
3. selector 使用 ASFormer hidden 的状态差分；actionness logits 在 transition
   descriptor 中 detach，counterfactual proximal 只更新 scorer。
4. 主 detector `cls_loss + reg_loss` 不会穿过 hard gather 直接更新 hard selection。
   train-only teacher 在 `no_grad` 下评估 baseline 和 hard swaps，并把 detached
   `L_baseline-L_swap` 作为策略监督。它不是 direct detector-gradient estimator。
5. 推理路径不读取 GT、teacher、ledger、外部 actionness JSONL 或 prediction cache。
6. AdaTAD 的 VideoMAE Adapter、ActionFormerHead、assignment/loss 和 NMS 组件被复用，
   但输入长度、wrapper、selected-axis geometry、`with_cp` 和 DDP 图合同已有实质扩展；
   只能称 `official components with wrapper`，不能称 source-identical。

## Signed Proximal 数学吸收

对有效 swap incidence `A`、中心分数 `s`、`d=As`、Gram
`G=AA^T` 和归一化 detached utility `u`，当前目标使用
`v=G^{-1}u`、`d*=stopgrad(d)+eta*v` 与
`L=(1/(2m))||d-d*||^2`。在当前点：

```text
A(-grad_s L) = (eta / m) u
```

正 loss weight、active-batch mean、GradScaler 和 DDP averaging 只引入正尺度，因此
不改变该 loss 分量相对于显式 score tensor 的符号。shared-remove builder 下 Gram
由 `I + 11^T` 块构成，最多四个候选时 `cond(G) <= 5` 是当前候选构造的可证明性质。

必须保留两项限制：

- 该推导不保证共享参数经过其他 losses、AdamW 预条件、动量和 weight decay 后，
  每个 hard swap 在真实 optimizer step 中仍逐项改善。
- `score_space_utility_alignment()` 使用同一 loss、incidence 与 utility 做代数自检，
  能发现 wiring/mask/scale 错误，但不能独立证明 teacher、总梯度或最终 mAP 正确。

## 独立代码复核

本轮针对精确提交复核了 reviewer 的关键静态判断：

- `counterfactual_utility.py` 的 score-space audit 对 all-zero utility 返回
  `sign=1`、`spearman=1` 和 `error=0`，确有无信息样本形成“完美 alignment”的风险。
- formal gate 明确记录
  `input_provenance=deterministic_synthetic_contract_probe` 和
  `real_dataset_loader_executed=False`；它没有证明真实 loader、真实 batch strata、
  DDP reducer、AMP replay、EMA 或 selector schedule。
- 最终 config 将 `detector_gradient_final_weight=0.0`，并明确标记
  `detector_utility_is_direct_gradient=False`；把当前路线称为直接检测梯度反传是错误的。
- config 使用 `with_cp=False`、`static_graph=False`、
  `find_unused_parameters=True`。这是 DUCA 动态参数使用图的项目扩展，不是官方 dense
  AdaTAD 的原始执行合同。

因此，本项目对该 review 的裁决为：
`ACCEPT_WITH_SCOPE / GO_TO_REAL_GATE_ONLY`。

## 必须吸收的 P1/P2

| 级别 | 问题 | 项目处理 |
|---|---|---|
| P1 | synthetic gate 未走真实 loader/DDP/train engine/replay/EMA/schedule | 在任何 pilot/full train 前新增 fail-closed 真实 THUMOS loader CUDA gate |
| P1 | all-zero 或单值 utility 可形成无信息 alignment 假阳性 | 记录 informative/nonzero/distinct counts；无信息样本不能通过方向 gate |
| P1 | `L_baseline-L_swap` 混合 RGB 内容、selected-axis geometry 与重新 assignment | 统一称 `selection-policy utility including selected-axis geometry`，禁止称纯帧内容边际效用 |
| P1 | exact-commit 的 world-size=1 DDP reducer 未运行验证 | real gate 必须实际 DDP-wrap 并调用正式 `train_one_epoch` |
| P2 | 每步 alignment 额外反向、retain graph 和 condition-number 同步增加成本 | 正式训练只低频采样该诊断，gate/debug 保留完整检查 |
| P2 | ASFormer sliding-window mask 行为来自固定上游 | 固定上游 source hash并披露，不在本轮另造 ASFormer 变体 |

## 证据与 Claim 影响

- 实现状态保持 `tested`，没有升级为 `experiment_running`、
  `empirically_supported` 或 `paper_ready`。
- C3 仍为 `unproven`：没有 same-commit matched exact-uniform terminal-EMA mAP。
- C4 的原始“直接 detector 梯度改善 selector”仍为 `unproven`，并且当前 detached
  hard-swap candidate并不直接检验该表述。未来若保留该机制，应使用独立的
  “detached selection-policy utility improves selector” claim。
- C7 仍为 `unproven`：没有 trained-checkpoint full-stack cost、p50/p95、energy 或
  accuracy-cost Pareto 证据。
- `signed_proximal_math=pass` 只是局部数学/连接合同证据，不是效用、训练稳定性或性能证据。
- 第二轮最终方法裁决仍未执行；本审计不把 DUCA 宣布为最终论文方法。

## 有界下一步

1. 修复无信息 utility 的 gate 假阳性，并冻结策略效用命名与 provenance 字段。
2. 新增 exact-commit real-loader CUDA gate：真实 THUMOS GT，覆盖 full、mixed 和
   all-short；实际 world-size=1 DDP、optimizer、scheduler、GradScaler、EMA 和
   selector schedule；强制一次 AMP overflow 并验证同 batch/state replay。
3. Gate 必须验证成功更新计数一致、exact-K/max-hole、真实 GT remap/inverse-map、
   finite loss/grad，以及 proximal、coarse、adapter/projection/head 的梯度所有权。
4. 仅当该 gate 通过后，才允许 forced-overflow/mixed-batch pilot；pilot 通过后才允许
   matched seed-0 full train。
5. 在进入性能实验前，把本记录中的 `HANDOFF_PACKET` 交给第二轮 Pro 裁决，确定唯一
   最终方法和 claim，避免把“静态可 gate”误当成“路线已经成立”。

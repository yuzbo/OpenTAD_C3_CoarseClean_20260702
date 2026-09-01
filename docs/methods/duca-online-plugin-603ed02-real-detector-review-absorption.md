---
updated: 2026-07-08
status: active
scope: 吸收 commit 603ed02 DUCA online plugin 外部严厉审查，记录论文级方法约束、P0/P1 缺口与后续实现门槛
out-of-scope: 不记录新实验结果；不把 smoke/precheck 结果升级为 detector mAP 证据
---

# DUCA Online Plugin 603ed02 Review Absorption

原始审查全文已完整归档：

- `docs/methods/reviews/2026-07-08-603ed02-duca-online-plugin-real-detector-review-raw.txt`

## 总判定

这轮 review 对 `603ed0203ef2ebf631523ca42d9061b5ee877de9` 的裁决必须被完全吸收：

- `603ed02` 不是 CVPR-ready。
- `603ed02` 不是可信的 full experimental prototype。
- 它是有价值的 engineering scaffold / integration precheck。
- 真实进展是存在的：`DucaOnlineFrameSelector` 已能通过 OpenTAD registry / `frame_selector` seam 接入 `SingleStageDetector`，并让 selected inputs/masks 进入下游 detector path。
- 但它尚未解决论文主问题：真实 AdaTAD/ActionFormer sparse-time 训练评估、严格 no-leak provenance、正确 sparse temporal geometry、以及真正 detector-loss-aware selector learning。

因此，后续任何文档、prompt、论文叙事、实验汇报都不能把 `DucaOnlinePrecheckHead`、toy precheck config、fake AdaTAD wrapper、geometry-only metrics、或 smoke gradient 当作 DUCA 主方法证据。

## 必须降级的 Claim

以下 claim 当前只能说 partial / unsupported，不能作为论文结论：

- real AdaTAD / ActionFormer mAP：blocked，当前 precheck head 是 toy head，不是真 detector head。
- detector-loss-aware acquisition：blocked，现有 ST 只证明 graph connectivity，不证明 detector loss 会给未选位置提供有效 credit。
- high-IoU localization：blocked，GT / prediction / selected-axis / original-time 坐标契约尚未闭合。
- pre-backbone raw-frame compute saving：unsupported，当前 original-time 更准确说是 dense feature-grid / temporal observation index。
- no-target-label zero-shot actionness：blocked，manual / p_action / actionness provenance 还不够 fail-closed。
- strict `<=384` fairness：partial，selector 内部还需要硬 cap 阻止 runtime budget override。
- ActionFormer transfer：unsupported，缺少 sparse true-time point generator / target assigner / decode proof。

## P0 吸收项

1. 真实 detector 证据缺失
   - `duca_online_adatad_precheck.py` 和 `duca_online_zeroshot_actionness_precheck.py` 只能作为 smoke/precheck。
   - `DucaOnlinePrecheckHead` 不是 AdaTAD / ActionFormer。
   - 下一步必须新增真实 AdaTAD/OpenTAD config，跑真实 THUMOS train/eval，且 metric run 必须 fail if smoke head is used。

2. GT coordinate contract 不闭合
   - 现在 sparse selected-axis inputs 进入 detector，但 `gt_segments` 可能仍是 dense/original coordinates。
   - 如果 detector head 在 selected-axis 上做 assignment/regression，GT 必须 remap 到 selected-axis。
   - 更优雅的最终路线是 irregular true-time sparse head：head 直接在 selected true-time positions 上生成 points、assign GT、回归 true-time offsets。

3. Prediction remap 不足以证明 high-IoU
   - selected-rank segment 线性插值回 original-time 只是临时桥。
   - 如果 selected observations 跨过 boundary，remap 无法恢复缺失边界信息。
   - 高 IoU 主张必须依赖 true-time sparse head 或严格 selected-axis GT/prediction remap 验证。

4. Original-time unit 语义必须明确
   - 当前 `original_time_index` 实际多半是 detector dense temporal observation / feature-grid index。
   - 论文必须区分 `raw_frame_index`、`video_time_sec`、`snippet_index`、`adatad_feature_index`、`selected_axis_index`。
   - 若选择发生在 feature extraction 后，就不能声称 backbone FLOPs saving。

5. ST gradient estimator 不足
   - 当前 hard gather indices 不可微；ST mask 主要只影响已选位置的 amplitude。
   - 需要 Gumbel-TopK / perturb-and-MAP / SoftSort / Sinkhorn / REINFORCE / soft all-candidate backward path 等更有意义的 budgeted selection estimator。
   - 测试必须证明 detector loss 能改变 unselected-position logits，而不只是 selected-slot weights。

6. Provenance / no-leak 不够 fail-closed
   - 每个 `p_action` / `actionness_logits` 必须携带 `ActionnessSourceProvenance`。
   - provenance 缺失、unknown、THUMOS-trained、uses teacher/cache/oracle 时，selector 应 fail-closed 或标记为 non-deployable baseline。
   - zero-shot 工具的 leakage scanner 不能只停留在 tool-local，必须成为 selector train/test path 的统一约束。

7. Budget cap 必须在 selector 内 enforce
   - 外部 `budget` override 不能突破 hard cap 384。
   - 需要记录 requested/effective budget，并测试 `budget=385` 必须 fail。

8. Metadata validator 必须验证一致性
   - 不只检查 metadata key 是否存在。
   - 必须重算 `selected_inputs == gather(dense_inputs, selected_positions)`、mask sum、positions sorted/unique/in-range、remap inverse consistency、padded slot 不影响 loss/proposal。
   - reserved metadata keys 应 overwrite，而不是 `setdefault` 保留陈旧值。

## P1 吸收项

- `selected_mask_st` forward 对齐了 consumed positions，但作为 selector learning estimator 仍弱。
- `remap_gt_to_selected_axis=True` 目前必须要么真正实现，要么移除该 claim。
- center-radius residual fill 需要报告 fill ratio，否则可能退化为 score top-k。
- metadata 中需暴露 padded slot masks，避免 dynamic budget 下 detector 消费无效 slot。
- manual/precomputed zero-shot provenance 默认不能标成 no-THUMOS。
- `video_text_mock` 只能作为接口测试，不能作为 zero-shot foundation-model evidence。
- geometry metrics 只能说明 selection geometry，不能证明 detector mAP。
- ActionFormer 需要真实 irregular-grid implementation，不能只靠 generic hook claim。

## 最终主方法应吸收为 DUCA-STG

review 推荐的最终主方法是 DUCA-STG：

```text
hard-forward sparse TopK selection
+ differentiable soft-backward budgeted relaxation
+ learned radius/context
+ detector-responsibility or detector-loss feedback
+ true-time irregular sparse grid head
+ fail-closed no-leak provenance
+ strict K <= 384
```

关键原则：

- detector 必须只消费 selected observations。
- train forward 与 test forward 都必须 hard select。
- selector 的训练必须来自真实 detector loss，不能只来自 fake/smoke loss。
- inference 不能使用 teacher、GT、cache、offline decision ledger。
- ledger 只能是 audit/reproducibility artifact。

## 可执行实现门槛

后续实现至少要完成：

1. hard budget cap and provenance validator。
2. metadata consistency validator，重算 gather/mask/remap。
3. interim AdaTAD selected-axis GT remap，或直接 true-time sparse head。
4. real AdaTAD sparse config and full THUMOS eval。
5. metric runs 禁止使用 `DucaOnlinePrecheckHead`。
6. ST TopK / Gumbel-TopK selector，并验证 unselected-logit gradients。
7. ActionFormer sparse true-time point generator。
8. latency/cost logging，包括 selector/gather/backbone/detector/NMS/total。

## 必须的实验矩阵

论文级实验至少覆盖：

- Dense AdaTAD / Dense ActionFormer。
- Uniform fixed 384。
- Random fixed 384，至少 5 seeds。
- `p_action` top-K。
- zero-shot X-CLIP / ActionCLIP / SlowFast / VideoMAE actionness top-K。
- C3 / GAS-VT / lattice baselines。
- DUCA-Frozen。
- DUCA-Adapted without teacher warm-up。
- DUCA-Adapted with teacher warm-up。
- DUCA online plugin without joint fine-tune。
- DUCA online plugin with hard-forward ST fine-tune。
- AdaTAD and ActionFormer versions。
- dynamic-K and matched-average fixed-K。

必须报告的指标：

- Average mAP。
- mAP@0.5 / 0.6 / 0.7。
- high-IoU drop/gain vs uniform。
- selected count / budget violation rate。
- selector / gather / backbone / detector / NMS / total latency。
- training cost / teacher warm-up cost / zero-shot source cost。
- boundary support / short-action recall。
- action-local max hole / p95 hole。
- uniform similarity。
- actionness-interior over-selection。
- selector entropy / collapse diagnostics。
- no-leak audit pass/fail。

## 当前哪些只能作为 Scaffold

以下只能作为工程 gate，不能作为主方法结果：

- `DucaOnlinePrecheckHead` train/test。
- toy precheck configs。
- fake AdaTAD sparse wrapper。
- `real_detector_loss_selector_grad_nonzero` from smoke/fake detector。
- motion/feature-energy fallback 作为 foundation-model zero-shot evidence。
- geometry-only selection metrics 作为 detector mAP evidence。

## 最终吸收结论

`603ed02` 的价值是它建立了正确的架构缝隙：OpenTAD-buildable online selector 可以位于 detector 前并向下游传递 selected tensors。

但真正的论文方法尚未实现。下一步决定性里程碑不是更多 smoke tests，而是：

```text
真实 AdaTAD/ActionFormer sparse-time training/evaluation
+ 正确 temporal geometry
+ detector-loss-aware selection
+ strict no-leak provenance
+ fair cost accounting
+ 超过 uniform fixed 384 的真实 sparse detector mAP
```

只有达到这些条件，DUCA 才能从“工程 scaffold”升级为“CVPR 级方法候选”。

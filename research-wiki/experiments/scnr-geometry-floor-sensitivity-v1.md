---
type: experiment
node_id: exp:scnr-geometry-floor-sensitivity-v1
title: "SCNR-TAD native-cell ROI floor sensitivity v1"
stage: experiment_running
status: m2_arms_complete_cost_schema_failure_recovery_tested_local
outcome: pending
added: 2026-08-02
updated: 2026-08-05
---

# SCNR-TAD native-cell ROI floor sensitivity v1

## 问题与主张

本实验只回答一个问题：在其余动态 Stage-1 路由协议完全匹配时，连续 ROI 的
最小尺度应采用一个还是两个原生空间 token cell。它不用于重新比较 Free-first、
PL/ST、固定 `8/28/28`，也不单独产生 ROI+TokenSelect 有效性或效率主张。

主设置使用运行时网格推导的独立轴向下限：

`w_min = 1 / W_grid, h_min = 1 / H_grid`。

唯一的尺度敏感性消融使用：

`w_min = 2 / W_grid, h_min = 2 / H_grid`。

两者都使用同一个有界中心参数化，且不加入 Uni-AdaFocus Eq. 15 的
full-frame size penalty，也不加入 area、coverage 或 smoothness loss。

## Claim map

| Claim | 最小可信证据 | 反主张 |
| --- | --- | --- |
| G1：一个 native cell 是合理的主 floor | 1x1/2x2 在相同动态 exact-B 路径、初始化、数据顺序和训练协议下完成，并报告高 tIoU、ROI 尺度分布与真实成本 | 结果仅来自隐藏的较大窗口、额外计算或面积正则 |
| G2：方法不依赖精确的单一 floor 超参 | 1x1 不发生 sub-token collapse，且相对 2x2 不出现跨种子不稳定的灾难性退化 | 单个开发 seed 的偶然优势被当作 floor 最优性证明 |

## 冻结两臂

| Arm | `georoute_roi_extent_floor_mode` | `georoute_roi_extent_floor_cells` | 角色 |
| --- | --- | ---: | --- |
| G1 `native_1cell_main` | `native_cells` | 1 | 主设置 |
| G2 `native_2cell_sensitivity` | `native_cells` | 2 | 匹配尺度敏感性消融 |

必须逐项匹配：同一 source/scout 输入、VideoMAE 初始化、动态全窗口 exact-`B`、
允许 `K_t=0` 的 masked-zero carrier、Scheme-A utility、hard top-B/ST、早期 proxy
schedule、训练 split、样本顺序、seed、成功 optimizer update 数、AMP、EMA、detector、
evaluator/NMS 与成本测量。两臂的 `geometry_smoothness_weight`、
`area_prior_weight` 和任何 full-frame size penalty 都必须为零。

## 指标与诊断

- 决策指标：Avg-mAP、mAP@0.6、mAP@0.7，以及同预算下的实测成本 Pareto；
- 机制指标：逐 tubelet 的 `w/h/area`，floor/ceiling 饱和率，ROI token 占用，
  `K_t` 与角色计数分布；
- 计算账本：全窗口唯一 selected/executed `B`，每个 native clip 的 `b_c`，
  `sum_c b_c^2`，实际 patch embedding、attention、MLP、Adapter 调用，端到端
  p50/p95、峰值显存和 energy；
- 失败诊断：若 1x1 长期贴 floor 且定位下降，而 2x2 稳定改善，说明一格下限过松；
  若二者近似等价，保留更少先验的 1x1；若两者都失败，不能继续调 floor 来掩盖
  allocator、proxy 或 ragged execution 的问题。

## 执行顺序

| Milestone | 内容 | Gate | 状态 |
| --- | --- | --- | --- |
| M0 | 公式、独立宽高、11x20 的 1x1/2x2 known-answer tests | 精确得到 `(1/20,1/11)` 与 `(2/20,2/11)`，in-bounds 且梯度有限 | tested at exact source `4be71844`: focused `36/36`; complete GeoRoute+C3 `194 passed, 1 skipped` |
| M1 | 真实 180x320 → 11x20 P0，验证配置、审计字段和零正则 | 不产生 metric/checkpoint；任一 hidden clamp/penalty/静态 0.20 即失败 | tested: G1 Job `1215358`; G2 Job `1215364` |
| M2 | 匹配 G1/G2 development 训练、完整 accuracy/telemetry 回放、独立全栈成本回放 | 两臂全完成、population/hash/cost ledger 一致后才读结果 | G1/G2 `1216180/1216181` complete; four failed cost attempts are preserved; final-call repair `42923d9f` passed local/remote checks; replacement `1222889/1222890` is `experiment_running`, no result yet |
| M3 | 仅在动态主方法通过总 gate 后进入 disjoint-seed confirmation | 不从单 seed 宣称 floor 最优 | blocked |

## 当前边界

M0 与 M1 已通过。G1 的 corrected P0 Job `1215358` 使用一格 floor，G2 Job
`1215364` 使用两格 floor；两者都在动态全窗口 exact-`B`、masked-zero、真 ragged
路径上完成真实模型 forward/backward，且没有 metric/checkpoint。G2 配置在 source
`8aa8e2a3` 的契约测试中与 G1 逐字段匹配，唯一方法差异是
`georoute_roi_extent_floor_cells: 1 -> 2`。提交 `7e5775e8` 已补齐逐 tubelet
ROI/`K_t`/角色/真 ragged 成本遥测并在 N16R4 Linux 回归中通过 `35/35`。提交
`ec8de9f51f85fc81031d82b79e30019d57a381b4` 已实现并冻结单 seed-3407、60 epoch
两臂 runner、最终 epoch-59 checkpoint/成功更新/EMA 原子 sidecar、完整 Gate
accuracy/telemetry 回放、同一 GPU 上 `G1 -> G2 -> G2 -> G1` 的独立全栈成本回放，
以及 after-any fail-closed finalizer。成本 validator 会从原始 NVML trace 与单调时间窗
重新积分 energy，并绑定完整 population、配置、checkpoint 和 stage-result 谱系。
本地编译、Bash、whitespace 与聚焦契约回归通过（`29 passed`）。干净远端 runtime
`9d6641a6` 进一步通过 Linux/Torch `76/76` 与两臂、成本、终结器四项 precheck；
存储要求 `47,244,640,256` bytes、可用 `225,293,430,784` bytes，提交额度也通过。
但正式部署在任何 Job 创建前被 N16R4 submit Lua 拒绝：CPU-only finalizer 不满足该站点
必须申请 GPU 的规则。旧 root
`/data/run01/sczc063/yuzibo/scnr_dynamic_floor_m2_9d6641a6_s3407_20260804_0507`
只含 storage preflight，永久不复用。最小修复
`bad14693daa1fe414e56bf697c617e76f96eed48` 将 finalizer 改为 1 GPU/1 CPU 调度
（不进行模型或成本计算，并在 deployment receipt 中披露为 scheduling overhead），
本地 `13/13` 通过。新源的远端回归、新 precheck 与新 namespace 尚待执行；训练、
checkpoint、metric、latency、energy 仍全部未产生。

替换 runtime `6ee97336775a09611f10423e07cafcea375e191a` 已通过相同远端
Linux/Torch `76/76` 与全新四项 precheck，并在新 root
`/data/run01/sczc063/yuzibo/scnr_dynamic_floor_m2_6ee97336_s3407_20260804_0525`
原子提交和释放 G1 `1216180`、G2 `1216181`、paired-cost `1216182`、finalizer
`1216183`。部署 self-hash 为
`a0504e45179957f20580b901e6ef7723d63c7b0ed445d8b3c35c3b5aaa02b89a`，
deployment 文件 SHA-256 为
`188da9dbf8cabffc1ab59cd90822e117adcd1457729c5a08806c916516ac8284`。
G1/G2 新 P0 均为 `PASS_NO_PERFORMANCE_P0`（文件 SHA-256
`90d8cc10b9cb13853d90a8ff2cf4d92115ee3f99ffdcca2c12a4318cdfc08010` /
`30517e05372d4307aa2d40301031d42fdc33b4cd5d205aaa4d14f7234eceb0c1`），
随后两臂完成全部 60 epochs、accuracy 与 telemetry 回放并以 `0:0` 终止；stage-result
文件 SHA-256 分别为
`bc78df2304560b399e6d29fcf04027d888ae377feb6cbac16df25e97abfb0572` /
`eb0b677c1b47b4b66d30b9a9cbfa0aabc1e31e90fc8cbea4813f76768dcfeb8a`。
paired-cost Job `1216182` 在首个 timed-forward 审计处失败：当前 native-ragged
executor 输出 `packed.attention_pairs_per_window`，profiler 却读取不存在的
`packed.attention_pairs`。这发生在成本工件完整发布前，不是模型、训练、mAP 或成本
结果。Finalizer `1216183` 完成 `0:0` 并正确封口为
`FAIL_INCOMPLETE_NO_FLOOR_INFERENCE / INCOMPLETE_NO_FLOOR_INFERENCE`，
`paired_cost_present=false`、`official_test_opened=false`、
`paper_claim_allowed=false`。最小恢复实现改为校验当前 per-window ragged ledger，并将
冻结模型 runtime `6ee97336` 与 clean execution-repair commit 分开记录；本地 focused
tests `15/15`。恢复必须保留原失败工件，复用两臂而不重训，完整重跑
`G1 -> G2 -> G2 -> G1` 与新 finalizer。当前仍是 `experiment_running`，不得读取
两臂局部指标或形成 floor 结论。

恢复执行 commit `c67e13e84e47d17fb48ab416c35fa0786c16f2f3` 已在独立 clean
checkout 通过远端 focused `50/50` 与 cost precheck；它没有修改 `opentad/` 或
`configs/`。重新核验 `1216180/1216181` 的 `COMPLETED 0:0` 与上述 stage-result
哈希后，完整 cost-only Job `1222672` 和 fresh finalizer Job `1222673` 已先 hold、
写入并验证不可变 recovery receipt、再释放。由于旧完成作业已不能作为 live-controller
依赖目标，真实 scheduler DAG 为 cost 无依赖、finalizer `afterany:1222672`；回执中的
顶层依赖仅保留 fail-closed finalizer 所需的冻结科学 DAG，并在 recovery 字段明确披露
这一差异。回执 self/file SHA-256 为
`12cbbb3f609adaa57ca9b29bf930bd124cd35c5f33aaa966fc6a9529c3d1de89` /
`f67703bf4dc5d066f64a1bafa36d49be133a7b6701507ef8ef31d651a7d2fba7`。
Job `1222672` 在首个 timed sample 因 `sparse_adapter_ms=0` 失败：动态路径直接调用
`sparse_adapter.forward_ragged`，而旧 profiler 的 module hook 只在
`nn.Module.__call__` 时触发。Job `1222673` 正确封口为 incomplete；两者工件已归档，
未读取任何中间数值。最小 execution-only 修复
`6341927f099bd59e0be6aff9b4b1062b4f76150e` 改为包裹真实 ragged 方法并列出无效
stage，本地 `16/16`、远端 `51/51` 与 cost precheck 通过，且 `opentad/`、
`configs/` 不变。重新核验两臂后，cost `1222700` 与 finalizer `1222701` 已通过
self-hashed recovery-v2 receipt 原子释放；receipt self/file SHA-256 为
`9370e5908718e0c6fef857c3b29ddeefbe5701bbc6a1c221f7ad7f828dac99e7` /
`4968593b4172df4bbe7feeec9bad623ae188ffeeadedbd2b87c48a0bfa811fa3`。
Job `1222700` 完成四个 timed pass 后只在最终 profile validator 失败：producer 在
构建并哈希 cost config 前强制设置了 `post_processing.sliding_window=True`，validator
却使用未包含该变更的独立重建配置，因此
`pass_receipt.cost_config_sha256` 不一致。Finalizer `1222701` 再次正确封口为
incomplete。四个 raw cost 文件和 finalization 均已归档；由于 pass receipts 与完整
profile-level provenance 只存在内存中，协议禁止手工拼装或复用这些 raw 数值。精确
execution-only 修复 `011d2943c698bb8a3727de9163034a7153779b64` 让 producer、
validator、fixture 共用同一 cost-config builder（含 `sliding_window=True`），本地
`16/16`、远端 `51/51` 与 precheck 通过，模型、配置、训练和两臂工件均未变化。
held cost/finalizer `1222869/1222870` 已绑定 recovery-v3 receipt 后释放；receipt
self/file SHA-256 为
`4cc8b7649d82cfd89530453df6be02609af5a30334f0f9f44a3efa447bf584e2` /
`623c2f958d0a86fdba5ffd81f14c413e652c4d788b2fc089151b48bbf9ce81fe`，
真实 scheduler DAG 为 cost 无依赖、finalizer `afterany:1222869`。状态仍是
`experiment_running`，尚无可解释的成本或 floor 结果。Job `1222869` 随后在创建
`cost/` 前失败：population preflight 仍调用重构时已删除的 `_cost_config`，触发
`NameError`；Job `1222870` 正确封口为 incomplete，且不存在 pass sample 或 cost
profile。精确修复 `42923d9f7aaddb14368f82aacda5c77e1f857a24` 将最后一个调用点
改为共享 builder，并用 AST 回归禁止加载 legacy 名称；本地 `16/16`、远端 `51/51`
和 precheck 通过。held cost/finalizer `1222889/1222890` 已绑定 recovery-v4 后
释放，receipt self/file SHA-256 为
`5fe63bce1811abddadb5dda60bc67385b07693f7642c4de016616cd8756c1e1c` /
`5bd504a60668eeb204035d25e4853d601c67fc5097b474987e718562c7b51226`；
真实 scheduler DAG 为 cost 无依赖、finalizer `afterany:1222889`。当前仍无可解释的
成本或 floor 结果。

P0 中三角色计数、`K_t` 范围、loss 或梯度只能证明路径非退化且可训练，不能用于
判断 1x1/2x2 谁更好。只有完整匹配的 development 训练、相同 population/hash、
决策指标与实测成本全部封存后，才允许给出 floor 敏感性结论；仍不得把旧固定
K64/`8/28/28` checkpoint 换 floor 后混入本实验。

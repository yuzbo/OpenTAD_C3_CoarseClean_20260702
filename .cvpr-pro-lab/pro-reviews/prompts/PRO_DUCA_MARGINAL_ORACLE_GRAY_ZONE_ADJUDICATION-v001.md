# DUCA-Marginal 灰区终态科学裁决

**Nonce：`DUCA-MARGINAL-ORACLE-GRAY-ZONE-ADJUDICATION-v001-20260831`**

你是本课题持续任职的第一科学负责人、路线设计者和最终科研审查者。Codex 只负责在你冻结的单一任务内完成最小代码实现、独立代码审查、正式实验执行和证据回传。请依据下面的公开代码与原始终态材料独立判断，不沿用 Codex 的路线偏好，也不要为了延续已有投入而默认继续。

本轮只要求一次科学裁决。你必须给出且只给出一个 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`，并冻结唯一下一项可证伪任务；如果现有证据已经足以终止当前机制，请直接停止，不要为了“还有可试参数”追加搜索。如果应继续，请给出最小、直接、论文优先的实现与实验，不建设新的工作流框架、证明系统或复杂合同代码。

## 一、最新公开实现（必须以这些 GitHub 链接为代码真相）

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 当前分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-budget-v1-20260830>
- 最新精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f67d96fdf68a295eaa7f678f3dfc125530828889>
- 动态预算分配：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f67d96fdf68a295eaa7f678f3dfc125530828889/opentad/models/duca/dynamic_budget.py>
- 反事实效用头：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f67d96fdf68a295eaa7f678f3dfc125530828889/opentad/models/duca/counterfactual_utility.py>
- 冻结 H65 探针与汇总入口：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f67d96fdf68a295eaa7f678f3dfc125530828889/tools/bata/run_duca_marginal_frozen_h65_probe.py>
- 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f67d96fdf68a295eaa7f678f3dfc125530828889/tests/test_duca_marginal_budget.py>

三个重型 producer 产物实际生成于前一公开提交：

- <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f87555f7da362fe1a20d4ca08f7a68c975ed8280>

`f87555f7... → f67d96fd...` 的生产代码差异只有：把换行文本 holdout block-list 确定性序列化成 OpenTAD mAP evaluator 接受的 JSON 数组，并为这个适配增加回归测试。没有修改模型 forward、K256/K384/K512 选择、反事实 loss/prediction、分配器、配置、checkpoint、数据、NMS、指标或预注册科学门槛。

## 二、当前科学问题与冻结合同

此前固定预算原生 tubelet coreset 在 THUMOS14 上低于匹配均匀选择：Avg-mAP `62.81%` 对 `64.13%`，该细粒度选帧路线已经停止。随后 96-anchor temporal facility-location coverage 没有通过预注册中间机制门，也未进入完整训练。

你上一轮将问题修订为：冻结 H65 Scout 与 epoch-59 EMA detector，测量一个窗口从 K384 降到 K256 的损失代价、从 K384 升到 K512 的收益，并在同一视频严格不变的实际 observation 总预算下跨窗口重分配计算。

冻结合同为：

- `c_i(K)=min(V_i,K)`；每视频目标是 `sum_i min(V_i,384)`。
- 若非基线请求与 K384 暴露相同有效 observation，则完全别名 K384 的位置、loss、prediction 和执行，不做第二次 detector forward。
- 历史 K384 保持 384 个执行 slots；实际不同的非基线只执行 `16*ceil(actual/16)` 个 slots，末包 padding 少于 16。
- K256/K384/K512 是同一非连续 H65 priority sequence 的嵌套集合。
- 每视频最多改变 `floor(0.5N)` 个窗口；优化顺序为最大真实效用、较少改变、确定性 tier 字典序。
- 200 个训练侧视频按 seed 3407 固定划分为 160 个 utility-fit 视频与 40 个 utility-holdout 视频。训练侧 GT 只产生 detached 反事实效用和 oracle 诊断；official test 不可见。
- oracle 强 headroom 门：`ΔAvg-mAP ≥ +0.8` 且 `ΔmAP@0.7 ≥ +1.0` 个百分点。
- 无 headroom 边界：`ΔAvg-mAP < +0.3` 且 `ΔmAP@0.7 < +0.5`。
- 只有强 headroom 通过才训练 utility head；灰区必须返回你裁决。

## 三、运行与来源事实

1. Job `1262076` 在 `f87555f7...` 上完成 PRE_RUN：200 个训练侧视频、720 个窗口、160/40 划分、所有短窗口、47 个相同成本别名、完整 K384 输入张量逐窗口相同、冻结 detector/Scout、无训练和无 official test 均通过。
2. Job `1262077` 完成 `select-k384`、`counterfactual-k256` 和 `counterfactual-k512`，随后在 `summarize` 中因换行文本 block-list 被 JSON evaluator 读取而失败。它没有生成 `probe_result.json`；该失败不是科学结果。
3. 最小修复 `f67d96fd...` 经独立代码审查和 N16R4 11 项聚焦测试通过。Job `1262098` 只在该 clean 提交下重新执行 PRE_RUN 身份绑定与 `summarize`，复用已密封的三个 producer 产物；没有重跑 producer、训练 detector/Scout/utility head 或访问 official test。它于 2026-08-31 05:09:34 +08:00 以 `COMPLETED 0:0` 结束。
4. 当前 PRE_RUN 与最终 `probe_result.json` 绑定 `f67d96fd...`；三个 producer receipts 保留 `f87555f7...` 来源。两者的 config SHA、epoch-59 `state_dict_ema` checkpoint SHA、annotation、类别映射和 VideoMAE 预训练权重哈希一致。
5. 对此来源边界出现两种独立审查意见：一项审查认为 summary-only 修复且模型/数据哈希相同，复用产物可准入；另一项严格审查认为 producer 与 summary commit 不完全一致，应标为 BLOCKED。请你明确裁决这是否影响科学证据准入；不得忽略或静默改写这一分歧。

## 四、原始终态结果

训练侧 utility holdout：40 个视频、124 个窗口。

- `Fixed-H65-384`：Avg-mAP `88.131197%`；mAP@0.7 `76.270583%`。
- 使用真实反事实效用的 `Oracle-Reallocate-384`：Avg-mAP `88.856786%`；mAP@0.7 `76.999587%`。
- 增益：Avg-mAP `+0.725589` 个百分点；mAP@0.7 `+0.729004` 个百分点。
- oracle 分配：K384 `102` 个窗口、K256 `11` 个窗口、K512 `11` 个窗口。
- 所有视频预算误差为零；124 个窗口实际 observation 总成本 `47110`。
- 实现门：K384 selection 与 prediction 对所有窗口精确一致；detector/Scout frozen；utility targets detached；非基线观察到的真实执行 slots 为 256/400/448/464/480/496/512，没有统一 padding 到上界。
- 结果低于强 headroom 门，但高于无 headroom 边界，runner 正确写为灰区并停止。
- utility head 没有训练；没有 predictability、learned allocation、official test、配对区间或端到端 latency/FLOPs 结果。
- secondary K320 没有运行，因为奇数窗口视频仅用 K256/K384 不能保证精确均值，且没有冻结额外规则。
- 这些 `88.xx%` 是训练侧 utility holdout 诊断，不得与官方 validation/test 结果直接比较，也不构成论文主结果。

随本提示词提供的原始文件：

- `pre_run_receipt.json`
- `probe_result.json`
- `selection_k384_receipt.json`
- `counterfactual_k256_receipt.json`
- `counterfactual_k512_receipt.json`
- `job1262077_failure_all.json`

## 五、你必须完成的裁决

请独立回答：

1. 当前双提交来源是否足以准入这份灰区诊断？若不足，说明会改变科学判断的最小必要修复；不得要求无关的哈希系统、框架或重算。
2. `+0.726/+0.729` 的真实效用 oracle 灰区意味着什么？它更支持“存在有限但不足的跨窗口预算空间”、当前三档/改变比例约束过强、统计波动未量化，还是该核心假设缺乏足够价值？请区分观察、推断与未知。
3. 是否允许训练已实现的 utility head，或是否应停止/修订/转向？Codex 不预设答案。
4. 冻结且只冻结一个当前任务。若继续或修订，请给出：允许修改的最小文件/符号、保持不变的机制与数据边界、最便宜的 falsifier、正式实验或只读诊断、指标/阈值/停止条件、Builder→独立 Critic→Evaluator 的最小交接，以及绝对截止时间。若停止或转向，请说明现有方向的失败根因、可保留的论文负结果和下一条唯一科学问题。
5. 明确当前可以写入论文的事实、不可写入的主张，以及该轮是否已经形成可投稿贡献。

输出必须先给出唯一裁决，然后给出证据准入、科学解释、唯一任务、实现边界、评估协议、论文主张边界，以及 `next_owner / next_action / dependency / expected_return_at`。不要给并列路线菜单，不要要求 Codex 自行选线，不要把浏览器、队列或工程流程当作科学工作。

请在回复中原样保留 Nonce：`DUCA-MARGINAL-ORACLE-GRAY-ZONE-ADJUDICATION-v001-20260831`。

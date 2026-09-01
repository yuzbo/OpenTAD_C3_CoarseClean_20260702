# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a949f18-f568-83ea-9871-70c012a75b2b

## Prompt

Read every attached file completely and answer the authoritative prompt as one independent scientific adjudication. Treat the GitHub repository, branch, exact commit f67d96fdf68a295eaa7f678f3dfc125530828889, and key-file URLs in the prompt as the latest code truth. Preserve nonce DUCA-MARGINAL-ORACLE-GRAY-ZONE-ADJUDICATION-v001-20260831 verbatim in the response.

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_MARGINAL_ORACLE_GRAY_ZONE_ADJUDICATION-v001.md
Lines: 1-87
```md
 1 | # DUCA-Marginal 灰区终态科学裁决
 2 | 
 3 | **Nonce：`DUCA-MARGINAL-ORACLE-GRAY-ZONE-ADJUDICATION-v001-20260831`**
 4 | 
 5 | 你是本课题持续任职的第一科学负责人、路线设计者和最终科研审查者。Codex 只负责在你冻结的单一任务内完成最小代码实现、独立代码审查、正式实验执行和证据回传。请依据下面的公开代码与原始终态材料独立判断，不沿用 Codex 的路线偏好，也不要为了延续已有投入而默认继续。
 6 | 
 7 | 本轮只要求一次科学裁决。你必须给出且只给出一个 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`，并冻结唯一下一项可证伪任务；如果现有证据已经足以终止当前机制，请直接停止，不要为了“还有可试参数”追加搜索。如果应继续，请给出最小、直接、论文优先的实现与实验，不建设新的工作流框架、证明系统或复杂合同代码。
 8 | 
 9 | ## 一、最新公开实现（必须以这些 GitHub 链接为代码真相）
10 | 
11 | - 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
12 | - 当前分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-budget-v1-20260830>
13 | - 最新精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f67d96fdf68a295eaa7f678f3dfc125530828889>
14 | - 动态预算分配：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f67d96fdf68a295eaa7f678f3dfc125530828889/opentad/models/duca/dynamic_budget.py>
15 | - 反事实效用头：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f67d96fdf68a295eaa7f678f3dfc125530828889/opentad/models/duca/counterfactual_utility.py>
16 | - 冻结 H65 探针与汇总入口：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f67d96fdf68a295eaa7f678f3dfc125530828889/tools/bata/run_duca_marginal_frozen_h65_probe.py>
17 | - 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f67d96fdf68a295eaa7f678f3dfc125530828889/tests/test_duca_marginal_budget.py>
18 | 
19 | 三个重型 producer 产物实际生成于前一公开提交：
20 | 
21 | - <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f87555f7da362fe1a20d4ca08f7a68c975ed8280>
22 | 
23 | `f87555f7... → f67d96fd...` 的生产代码差异只有：把换行文本 holdout block-list 确定性序列化成 OpenTAD mAP evaluator 接受的 JSON 数组，并为这个适配增加回归测试。没有修改模型 forward、K256/K384/K512 选择、反事实 loss/prediction、分配器、配置、checkpoint、数据、NMS、指标或预注册科学门槛。
24 | 
25 | ## 二、当前科学问题与冻结合同
26 | 
27 | 此前固定预算原生 tubelet coreset 在 THUMOS14 上低于匹配均匀选择：Avg-mAP `62.81%` 对 `64.13%`，该细粒度选帧路线已经停止。随后 96-anchor temporal facility-location coverage 没有通过预注册中间机制门，也未进入完整训练。
28 | 
29 | 你上一轮将问题修订为：冻结 H65 Scout 与 epoch-59 EMA detector，测量一个窗口从 K384 降到 K256 的损失代价、从 K384 升到 K512 的收益，并在同一视频严格不变的实际 observation 总预算下跨窗口重分配计算。
30 | 
31 | 冻结合同为：
32 | 
33 | - `c_i(K)=min(V_i,K)`；每视频目标是 `sum_i min(V_i,384)`。
34 | - 若非基线请求与 K384 暴露相同有效 observation，则完全别名 K384 的位置、loss、prediction 和执行，不做第二次 detector forward。
35 | - 历史 K384 保持 384 个执行 slots；实际不同的非基线只执行 `16*ceil(actual/16)` 个 slots，末包 padding 少于 16。
36 | - K256/K384/K512 是同一非连续 H65 priority sequence 的嵌套集合。
37 | - 每视频最多改变 `floor(0.5N)` 个窗口；优化顺序为最大真实效用、较少改变、确定性 tier 字典序。
38 | - 200 个训练侧视频按 seed 3407 固定划分为 160 个 utility-fit 视频与 40 个 utility-holdout 视频。训练侧 GT 只产生 detached 反事实效用和 oracle 诊断；official test 不可见。
39 | - oracle 强 headroom 门：`ΔAvg-mAP ≥ +0.8` 且 `ΔmAP@0.7 ≥ +1.0` 个百分点。
40 | - 无 headroom 边界：`ΔAvg-mAP < +0.3` 且 `ΔmAP@0.7 < +0.5`。
41 | - 只有强 headroom 通过才训练 utility head；灰区必须返回你裁决。
42 | 
43 | ## 三、运行与来源事实
44 | 
45 | 1. Job `1262076` 在 `f87555f7...` 上完成 PRE_RUN：200 个训练侧视频、720 个窗口、160/40 划分、所有短窗口、47 个相同成本别名、完整 K384 输入张量逐窗口相同、冻结 detector/Scout、无训练和无 official test 均通过。
46 | 2. Job `1262077` 完成 `select-k384`、`counterfactual-k256` 和 `counterfactual-k512`，随后在 `summarize` 中因换行文本 block-list 被 JSON evaluator 读取而失败。它没有生成 `probe_result.json`；该失败不是科学结果。
47 | 3. 最小修复 `f67d96fd...` 经独立代码审查和 N16R4 11 项聚焦测试通过。Job `1262098` 只在该 clean 提交下重新执行 PRE_RUN 身份绑定与 `summarize`，复用已密封的三个 producer 产物；没有重跑 producer、训练 detector/Scout/utility head 或访问 official test。它于 2026-08-31 05:09:34 +08:00 以 `COMPLETED 0:0` 结束。
48 | 4. 当前 PRE_RUN 与最终 `probe_result.json` 绑定 `f67d96fd...`；三个 producer receipts 保留 `f87555f7...` 来源。两者的 config SHA、epoch-59 `state_dict_ema` checkpoint SHA、annotation、类别映射和 VideoMAE 预训练权重哈希一致。
49 | 5. 对此来源边界出现两种独立审查意见：一项审查认为 summary-only 修复且模型/数据哈希相同，复用产物可准入；另一项严格审查认为 producer 与 summary commit 不完全一致，应标为 BLOCKED。请你明确裁决这是否影响科学证据准入；不得忽略或静默改写这一分歧。
50 | 
51 | ## 四、原始终态结果
52 | 
53 | 训练侧 utility holdout：40 个视频、124 个窗口。
54 | 
55 | - `Fixed-H65-384`：Avg-mAP `88.131197%`；mAP@0.7 `76.270583%`。
56 | - 使用真实反事实效用的 `Oracle-Reallocate-384`：Avg-mAP `88.856786%`；mAP@0.7 `76.999587%`。
57 | - 增益：Avg-mAP `+0.725589` 个百分点；mAP@0.7 `+0.729004` 个百分点。
58 | - oracle 分配：K384 `102` 个窗口、K256 `11` 个窗口、K512 `11` 个窗口。
59 | - 所有视频预算误差为零；124 个窗口实际 observation 总成本 `47110`。
60 | - 实现门：K384 selection 与 prediction 对所有窗口精确一致；detector/Scout frozen；utility targets detached；非基线观察到的真实执行 slots 为 256/400/448/464/480/496/512，没有统一 padding 到上界。
61 | - 结果低于强 headroom 门，但高于无 headroom 边界，runner 正确写为灰区并停止。
62 | - utility head 没有训练；没有 predictability、learned allocation、official test、配对区间或端到端 latency/FLOPs 结果。
63 | - secondary K320 没有运行，因为奇数窗口视频仅用 K256/K384 不能保证精确均值，且没有冻结额外规则。
64 | - 这些 `88.xx%` 是训练侧 utility holdout 诊断，不得与官方 validation/test 结果直接比较，也不构成论文主结果。
65 | 
66 | 随本提示词提供的原始文件：
67 | 
68 | - `pre_run_receipt.json`
69 | - `probe_result.json`
70 | - `selection_k384_receipt.json`
71 | - `counterfactual_k256_receipt.json`
72 | - `counterfactual_k512_receipt.json`
73 | - `job1262077_failure_all.json`
74 | 
75 | ## 五、你必须完成的裁决
76 | 
77 | 请独立回答：
78 | 
79 | 1. 当前双提交来源是否足以准入这份灰区诊断？若不足，说明会改变科学判断的最小必要修复；不得要求无关的哈希系统、框架或重算。
80 | 2. `+0.726/+0.729` 的真实效用 oracle 灰区意味着什么？它更支持“存在有限但不足的跨窗口预算空间”、当前三档/改变比例约束过强、统计波动未量化，还是该核心假设缺乏足够价值？请区分观察、推断与未知。
81 | 3. 是否允许训练已实现的 utility head，或是否应停止/修订/转向？Codex 不预设答案。
82 | 4. 冻结且只冻结一个当前任务。若继续或修订，请给出：允许修改的最小文件/符号、保持不变的机制与数据边界、最便宜的 falsifier、正式实验或只读诊断、指标/阈值/停止条件、Builder→独立 Critic→Evaluator 的最小交接，以及绝对截止时间。若停止或转向，请说明现有方向的失败根因、可保留的论文负结果和下一条唯一科学问题。
83 | 5. 明确当前可以写入论文的事实、不可写入的主张，以及该轮是否已经形成可投稿贡献。
84 | 
85 | 输出必须先给出唯一裁决，然后给出证据准入、科学解释、唯一任务、实现边界、评估协议、论文主张边界，以及 `next_owner / next_action / dependency / expected_return_at`。不要给并列路线菜单，不要要求 Codex 自行选线，不要把浏览器、队列或工程流程当作科学工作。
86 | 
87 | 请在回复中原样保留 Nonce：`DUCA-MARGINAL-ORACLE-GRAY-ZONE-ADJUDICATION-v001-20260831`。
```

### File: .cvpr-pro-lab/pro-reviews/runs/duca-marginal-oracle-gray-zone-v001/materials/pre_run_receipt.json
Lines: 1-455
```json
  1 | {
  2 |   "all_k384_video_actual_cost": {
  3 |     "video_validation_0000051": 1152,
  4 |     "video_validation_0000052": 1152,
  5 |     "video_validation_0000053": 1152,
  6 |     "video_validation_0000054": 768,
  7 |     "video_validation_0000055": 1152,
  8 |     "video_validation_0000056": 768,
  9 |     "video_validation_0000057": 1536,
 10 |     "video_validation_0000058": 768,
 11 |     "video_validation_0000059": 1536,
 12 |     "video_validation_0000060": 1152,
 13 |     "video_validation_0000151": 251,
 14 |     "video_validation_0000152": 1152,
 15 |     "video_validation_0000153": 1152,
 16 |     "video_validation_0000154": 384,
 17 |     "video_validation_0000155": 384,
 18 |     "video_validation_0000156": 1536,
 19 |     "video_validation_0000157": 378,
 20 |     "video_validation_0000158": 1920,
 21 |     "video_validation_0000159": 3072,
 22 |     "video_validation_0000160": 2688,
 23 |     "video_validation_0000161": 384,
 24 |     "video_validation_0000162": 1152,
 25 |     "video_validation_0000163": 1536,
 26 |     "video_validation_0000164": 1152,
 27 |     "video_validation_0000165": 768,
 28 |     "video_validation_0000166": 768,
 29 |     "video_validation_0000167": 1536,
 30 |     "video_validation_0000168": 768,
 31 |     "video_validation_0000169": 1152,
 32 |     "video_validation_0000170": 1536,
 33 |     "video_validation_0000171": 768,
 34 |     "video_validation_0000172": 768,
 35 |     "video_validation_0000173": 1536,
 36 |     "video_validation_0000174": 768,
 37 |     "video_validation_0000175": 384,
 38 |     "video_validation_0000176": 768,
 39 |     "video_validation_0000177": 768,
 40 |     "video_validation_0000178": 768,
 41 |     "video_validation_0000179": 1152,
 42 |     "video_validation_0000180": 1536,
 43 |     "video_validation_0000181": 768,
 44 |     "video_validation_0000182": 384,
 45 |     "video_validation_0000183": 384,
 46 |     "video_validation_0000184": 1152,
 47 |     "video_validation_0000185": 384,
 48 |     "video_validation_0000186": 384,
 49 |     "video_validation_0000187": 384,
 50 |     "video_validation_0000188": 1152,
 51 |     "video_validation_0000189": 768,
 52 |     "video_validation_0000190": 67,
 53 |     "video_validation_0000201": 331,
 54 |     "video_validation_0000202": 1152,
 55 |     "video_validation_0000203": 1152,
 56 |     "video_validation_0000204": 1920,
 57 |     "video_validation_0000205": 3840,
 58 |     "video_validation_0000206": 1152,
 59 |     "video_validation_0000207": 1152,
 60 |     "video_validation_0000208": 768,
 61 |     "video_validation_0000209": 4224,
 62 |     "video_validation_0000210": 768,
 63 |     "video_validation_0000261": 175,
 64 |     "video_validation_0000262": 244,
 65 |     "video_validation_0000263": 356,
 66 |     "video_validation_0000264": 1920,
 67 |     "video_validation_0000265": 384,
 68 |     "video_validation_0000266": 1152,
 69 |     "video_validation_0000267": 2688,
 70 |     "video_validation_0000268": 1920,
 71 |     "video_validation_0000269": 164,
 72 |     "video_validation_0000270": 384,
 73 |     "video_validation_0000281": 1536,
 74 |     "video_validation_0000282": 288,
 75 |     "video_validation_0000283": 384,
 76 |     "video_validation_0000284": 384,
 77 |     "video_validation_0000285": 768,
 78 |     "video_validation_0000286": 1536,
 79 |     "video_validation_0000287": 768,
 80 |     "video_validation_0000288": 768,
 81 |     "video_validation_0000289": 768,
 82 |     "video_validation_0000290": 384,
 83 |     "video_validation_0000311": 4224,
 84 |     "video_validation_0000312": 384,
 85 |     "video_validation_0000313": 768,
 86 |     "video_validation_0000314": 6528,
 87 |     "video_validation_0000315": 1152,
 88 |     "video_validation_0000316": 1152,
 89 |     "video_validation_0000317": 1536,
 90 |     "video_validation_0000318": 1152,
 91 |     "video_validation_0000319": 1920,
 92 |     "video_validation_0000320": 3840,
 93 |     "video_validation_0000361": 4992,
 94 |     "video_validation_0000362": 3456,
 95 |     "video_validation_0000363": 5376,
 96 |     "video_validation_0000364": 768,
 97 |     "video_validation_0000365": 1920,
 98 |     "video_validation_0000366": 241,
 99 |     "video_validation_0000367": 768,
100 |     "video_validation_0000368": 1920,
101 |     "video_validation_0000369": 8448,
102 |     "video_validation_0000370": 5376,
103 |     "video_validation_0000411": 4608,
104 |     "video_validation_0000412": 2688,
105 |     "video_validation_0000413": 4224,
106 |     "video_validation_0000414": 1920,
107 |     "video_validation_0000415": 1920,
108 |     "video_validation_0000416": 3840,
109 |     "video_validation_0000417": 2304,
110 |     "video_validation_0000418": 1536,
111 |     "video_validation_0000419": 5376,
112 |     "video_validation_0000420": 5376,
113 |     "video_validation_0000481": 3072,
114 |     "video_validation_0000482": 384,
115 |     "video_validation_0000483": 768,
116 |     "video_validation_0000484": 5760,
117 |     "video_validation_0000485": 1920,
118 |     "video_validation_0000486": 1152,
119 |     "video_validation_0000487": 2304,
120 |     "video_validation_0000488": 768,
121 |     "video_validation_0000489": 1152,
122 |     "video_validation_0000490": 2304,
123 |     "video_validation_0000661": 1152,
124 |     "video_validation_0000662": 384,
125 |     "video_validation_0000663": 1536,
126 |     "video_validation_0000664": 1536,
127 |     "video_validation_0000665": 3840,
128 |     "video_validation_0000666": 8448,
129 |     "video_validation_0000667": 2304,
130 |     "video_validation_0000668": 2688,
131 |     "video_validation_0000669": 1536,
132 |     "video_validation_0000670": 768,
133 |     "video_validation_0000681": 384,
134 |     "video_validation_0000682": 1152,
135 |     "video_validation_0000683": 285,
136 |     "video_validation_0000684": 384,
137 |     "video_validation_0000685": 768,
138 |     "video_validation_0000686": 231,
139 |     "video_validation_0000687": 245,
140 |     "video_validation_0000688": 768,
141 |     "video_validation_0000689": 153,
142 |     "video_validation_0000690": 1920,
143 |     "video_validation_0000781": 768,
144 |     "video_validation_0000782": 768,
145 |     "video_validation_0000783": 1152,
146 |     "video_validation_0000784": 384,
147 |     "video_validation_0000785": 768,
148 |     "video_validation_0000786": 768,
149 |     "video_validation_0000787": 296,
150 |     "video_validation_0000788": 384,
151 |     "video_validation_0000789": 1152,
152 |     "video_validation_0000790": 384,
153 |     "video_validation_0000851": 233,
154 |     "video_validation_0000852": 1536,
155 |     "video_validation_0000853": 384,
156 |     "video_validation_0000854": 127,
157 |     "video_validation_0000855": 1152,
158 |     "video_validation_0000856": 384,
159 |     "video_validation_0000857": 369,
160 |     "video_validation_0000858": 768,
161 |     "video_validation_0000859": 384,
162 |     "video_validation_0000860": 234,
163 |     "video_validation_0000901": 768,
164 |     "video_validation_0000902": 4224,
165 |     "video_validation_0000903": 3840,
166 |     "video_validation_0000904": 1536,
167 |     "video_validation_0000905": 1152,
168 |     "video_validation_0000906": 3072,
169 |     "video_validation_0000907": 260,
170 |     "video_validation_0000908": 1920,
171 |     "video_validation_0000909": 1152,
172 |     "video_validation_0000910": 1152,
173 |     "video_validation_0000931": 384,
174 |     "video_validation_0000932": 768,
175 |     "video_validation_0000933": 1152,
176 |     "video_validation_0000934": 384,
177 |     "video_validation_0000935": 384,
178 |     "video_validation_0000936": 319,
179 |     "video_validation_0000937": 384,
180 |     "video_validation_0000938": 260,
181 |     "video_validation_0000939": 1152,
182 |     "video_validation_0000940": 269,
183 |     "video_validation_0000941": 1536,
184 |     "video_validation_0000942": 384,
185 |     "video_validation_0000943": 768,
186 |     "video_validation_0000944": 1152,
187 |     "video_validation_0000945": 1152,
188 |     "video_validation_0000946": 384,
189 |     "video_validation_0000947": 384,
190 |     "video_validation_0000948": 226,
191 |     "video_validation_0000949": 152,
192 |     "video_validation_0000950": 151,
193 |     "video_validation_0000981": 768,
194 |     "video_validation_0000982": 160,
195 |     "video_validation_0000983": 768,
196 |     "video_validation_0000984": 348,
197 |     "video_validation_0000985": 1152,
198 |     "video_validation_0000986": 151,
199 |     "video_validation_0000987": 768,
200 |     "video_validation_0000988": 768,
201 |     "video_validation_0000989": 278,
202 |     "video_validation_0000990": 768
203 |   },
204 |   "all_k384_video_expected_target": {
205 |     "video_validation_0000051": 1152,
206 |     "video_validation_0000052": 1152,
207 |     "video_validation_0000053": 1152,
208 |     "video_validation_0000054": 768,
209 |     "video_validation_0000055": 1152,
210 |     "video_validation_0000056": 768,
211 |     "video_validation_0000057": 1536,
212 |     "video_validation_0000058": 768,
213 |     "video_validation_0000059": 1536,
214 |     "video_validation_0000060": 1152,
215 |     "video_validation_0000151": 251,
216 |     "video_validation_0000152": 1152,
217 |     "video_validation_0000153": 1152,
218 |     "video_validation_0000154": 384,
219 |     "video_validation_0000155": 384,
220 |     "video_validation_0000156": 1536,
221 |     "video_validation_0000157": 378,
222 |     "video_validation_0000158": 1920,
223 |     "video_validation_0000159": 3072,
224 |     "video_validation_0000160": 2688,
225 |     "video_validation_0000161": 384,
226 |     "video_validation_0000162": 1152,
227 |     "video_validation_0000163": 1536,
228 |     "video_validation_0000164": 1152,
229 |     "video_validation_0000165": 768,
230 |     "video_validation_0000166": 768,
231 |     "video_validation_0000167": 1536,
232 |     "video_validation_0000168": 768,
233 |     "video_validation_0000169": 1152,
234 |     "video_validation_0000170": 1536,
235 |     "video_validation_0000171": 768,
236 |     "video_validation_0000172": 768,
237 |     "video_validation_0000173": 1536,
238 |     "video_validation_0000174": 768,
239 |     "video_validation_0000175": 384,
240 |     "video_validation_0000176": 768,
241 |     "video_validation_0000177": 768,
242 |     "video_validation_0000178": 768,
243 |     "video_validation_0000179": 1152,
244 |     "video_validation_0000180": 1536,
245 |     "video_validation_0000181": 768,
246 |     "video_validation_0000182": 384,
247 |     "video_validation_0000183": 384,
248 |     "video_validation_0000184": 1152,
249 |     "video_validation_0000185": 384,
250 |     "video_validation_0000186": 384,
251 |     "video_validation_0000187": 384,
252 |     "video_validation_0000188": 1152,
253 |     "video_validation_0000189": 768,
254 |     "video_validation_0000190": 67,
255 |     "video_validation_0000201": 331,
256 |     "video_validation_0000202": 1152,
257 |     "video_validation_0000203": 1152,
258 |     "video_validation_0000204": 1920,
259 |     "video_validation_0000205": 3840,
260 |     "video_validation_0000206": 1152,
261 |     "video_validation_0000207": 1152,
262 |     "video_validation_0000208": 768,
263 |     "video_validation_0000209": 4224,
264 |     "video_validation_0000210": 768,
265 |     "video_validation_0000261": 175,
266 |     "video_validation_0000262": 244,
267 |     "video_validation_0000263": 356,
268 |     "video_validation_0000264": 1920,
269 |     "video_validation_0000265": 384,
270 |     "video_validation_0000266": 1152,
271 |     "video_validation_0000267": 2688,
272 |     "video_validation_0000268": 1920,
273 |     "video_validation_0000269": 164,
274 |     "video_validation_0000270": 384,
275 |     "video_validation_0000281": 1536,
276 |     "video_validation_0000282": 288,
277 |     "video_validation_0000283": 384,
278 |     "video_validation_0000284": 384,
279 |     "video_validation_0000285": 768,
280 |     "video_validation_0000286": 1536,
281 |     "video_validation_0000287": 768,
282 |     "video_validation_0000288": 768,
283 |     "video_validation_0000289": 768,
284 |     "video_validation_0000290": 384,
285 |     "video_validation_0000311": 4224,
286 |     "video_validation_0000312": 384,
287 |     "video_validation_0000313": 768,
288 |     "video_validation_0000314": 6528,
289 |     "video_validation_0000315": 1152,
290 |     "video_validation_0000316": 1152,
291 |     "video_validation_0000317": 1536,
292 |     "video_validation_0000318": 1152,
293 |     "video_validation_0000319": 1920,
294 |     "video_validation_0000320": 3840,
295 |     "video_validation_0000361": 4992,
296 |     "video_validation_0000362": 3456,
297 |     "video_validation_0000363": 5376,
298 |     "video_validation_0000364": 768,
299 |     "video_validation_0000365": 1920,
300 |     "video_validation_0000366": 241,
301 |     "video_validation_0000367": 768,
302 |     "video_validation_0000368": 1920,
303 |     "video_validation_0000369": 8448,
304 |     "video_validation_0000370": 5376,
305 |     "video_validation_0000411": 4608,
306 |     "video_validation_0000412": 2688,
307 |     "video_validation_0000413": 4224,
308 |     "video_validation_0000414": 1920,
309 |     "video_validation_0000415": 1920,
310 |     "video_validation_0000416": 3840,
311 |     "video_validation_0000417": 2304,
312 |     "video_validation_0000418": 1536,
313 |     "video_validation_0000419": 5376,
314 |     "video_validation_0000420": 5376,
315 |     "video_validation_0000481": 3072,
316 |     "video_validation_0000482": 384,
317 |     "video_validation_0000483": 768,
318 |     "video_validation_0000484": 5760,
319 |     "video_validation_0000485": 1920,
320 |     "video_validation_0000486": 1152,
321 |     "video_validation_0000487": 2304,
322 |     "video_validation_0000488": 768,
323 |     "video_validation_0000489": 1152,
324 |     "video_validation_0000490": 2304,
325 |     "video_validation_0000661": 1152,
326 |     "video_validation_0000662": 384,
327 |     "video_validation_0000663": 1536,
328 |     "video_validation_0000664": 1536,
329 |     "video_validation_0000665": 3840,
330 |     "video_validation_0000666": 8448,
331 |     "video_validation_0000667": 2304,
332 |     "video_validation_0000668": 2688,
333 |     "video_validation_0000669": 1536,
334 |     "video_validation_0000670": 768,
335 |     "video_validation_0000681": 384,
336 |     "video_validation_0000682": 1152,
337 |     "video_validation_0000683": 285,
338 |     "video_validation_0000684": 384,
339 |     "video_validation_0000685": 768,
340 |     "video_validation_0000686": 231,
341 |     "video_validation_0000687": 245,
342 |     "video_validation_0000688": 768,
343 |     "video_validation_0000689": 153,
344 |     "video_validation_0000690": 1920,
345 |     "video_validation_0000781": 768,
346 |     "video_validation_0000782": 768,
347 |     "video_validation_0000783": 1152,
348 |     "video_validation_0000784": 384,
349 |     "video_validation_0000785": 768,
350 |     "video_validation_0000786": 768,
351 |     "video_validation_0000787": 296,
352 |     "video_validation_0000788": 384,
353 |     "video_validation_0000789": 1152,
354 |     "video_validation_0000790": 384,
355 |     "video_validation_0000851": 233,
356 |     "video_validation_0000852": 1536,
357 |     "video_validation_0000853": 384,
358 |     "video_validation_0000854": 127,
359 |     "video_validation_0000855": 1152,
360 |     "video_validation_0000856": 384,
361 |     "video_validation_0000857": 369,
362 |     "video_validation_0000858": 768,
363 |     "video_validation_0000859": 384,
364 |     "video_validation_0000860": 234,
365 |     "video_validation_0000901": 768,
366 |     "video_validation_0000902": 4224,
367 |     "video_validation_0000903": 3840,
368 |     "video_validation_0000904": 1536,
369 |     "video_validation_0000905": 1152,
370 |     "video_validation_0000906": 3072,
371 |     "video_validation_0000907": 260,
372 |     "video_validation_0000908": 1920,
373 |     "video_validation_0000909": 1152,
374 |     "video_validation_0000910": 1152,
375 |     "video_validation_0000931": 384,
376 |     "video_validation_0000932": 768,
377 |     "video_validation_0000933": 1152,
378 |     "video_validation_0000934": 384,
379 |     "video_validation_0000935": 384,
380 |     "video_validation_0000936": 319,
381 |     "video_validation_0000937": 384,
382 |     "video_validation_0000938": 260,
383 |     "video_validation_0000939": 1152,
384 |     "video_validation_0000940": 269,
385 |     "video_validation_0000941": 1536,
386 |     "video_validation_0000942": 384,
387 |     "video_validation_0000943": 768,
388 |     "video_validation_0000944": 1152,
389 |     "video_validation_0000945": 1152,
390 |     "video_validation_0000946": 384,
391 |     "video_validation_0000947": 384,
392 |     "video_validation_0000948": 226,
393 |     "video_validation_0000949": 152,
394 |     "video_validation_0000950": 151,
395 |     "video_validation_0000981": 768,
396 |     "video_validation_0000982": 160,
397 |     "video_validation_0000983": 768,
398 |     "video_validation_0000984": 348,
399 |     "video_validation_0000985": 1152,
400 |     "video_validation_0000986": 151,
401 |     "video_validation_0000987": 768,
402 |     "video_validation_0000988": 768,
403 |     "video_validation_0000989": 278,
404 |     "video_validation_0000990": 768
405 |   },
406 |   "all_k384_video_target_exact": true,
407 |   "checkpoint_payload_epoch": 59,
408 |   "checkpoint_state_key": "state_dict_ema",
409 |   "collapsed_alias_count": 47,
410 |   "detector_or_scout_gradients_created": false,
411 |   "detector_training_performed": false,
412 |   "fit_video_count": 160,
413 |   "frozen_loss_normalizer": 45.0,
414 |   "holdout_video_count": 40,
415 |   "k384_full_tensor_equal_all_windows": true,
416 |   "official_evaluator_called": false,
417 |   "official_test_consumed": false,
418 |   "real_forward_execution_classes": [
419 |     "historical_k384_short",
420 |     "explicit_k384_full",
421 |     "k256_exec256",
422 |     "k512_exec400",
423 |     "k512_exec448",
424 |     "k512_exec464",
425 |     "k512_exec480",
426 |     "k512_exec496",
427 |     "k512_exec512"
428 |   ],
429 |   "short_windows_included": true,
430 |   "source": {
431 |     "annotation": "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json",
432 |     "annotation_sha256": "ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad",
433 |     "checkpoint": "/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth",
434 |     "checkpoint_epoch": 59,
435 |     "checkpoint_sha256": "dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c",
436 |     "checkpoint_state_key": "state_dict_ema",
437 |     "class_map": "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt",
438 |     "class_map_sha256": "a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31",
439 |     "config": "/data/run01/sczc063/yuzibo/duca_marginal_f67d96fd_20260831/configs/adatad/thumos/duca_marginal_frozen_h65_probe.py",
440 |     "config_sha256": "02e091995124496c5c5e0011923ac12de6b8dea29679ec821e64e6e5b6271ca6",
441 |     "git": {
442 |       "branch": "feature/duca-marginal-budget-v1-20260830",
443 |       "dirty": false,
444 |       "head": "f67d96fdf68a295eaa7f678f3dfc125530828889"
445 |     },
446 |     "train_data": "/data/run01/sczc063/yuzibo/thumos14/raw_data/video",
447 |     "videomae_pretrain": "/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
448 |     "videomae_pretrain_sha256": "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
449 |   },
450 |   "stage": "pre-run",
451 |   "status": "PRE_RUN_PASS",
452 |   "training_video_count": 200,
453 |   "training_window_count": 720,
454 |   "utility_head_fit_performed": false
455 | }
```

### File: .cvpr-pro-lab/pro-reviews/runs/duca-marginal-oracle-gray-zone-v001/materials/probe_result.json
Lines: 1-1556
```json
   1 | {
   2 |   "fit_video_count": 160,
   3 |   "fit_window_count": 596,
   4 |   "fixed_arm_name": "Fixed-H65-384",
   5 |   "fixed_h65_384": {
   6 |     "average_mAP": 0.8813119705666871,
   7 |     "mAP@0.3": 0.9583379279516155,
   8 |     "mAP@0.4": 0.936845289659517,
   9 |     "mAP@0.5": 0.8958122338381976,
  10 |     "mAP@0.6": 0.8528585697798222,
  11 |     "mAP@0.7": 0.7627058316042832
  12 |   },
  13 |   "holdout_video_count": 40,
  14 |   "holdout_window_count": 124,
  15 |   "implementation_gate": {
  16 |     "detector_frozen": true,
  17 |     "k384_prediction_exact_all_windows": true,
  18 |     "k384_selection_bit_exact_all_windows": true,
  19 |     "observed_distinct_execution_slot_classes": [
  20 |       256,
  21 |       400,
  22 |       448,
  23 |       464,
  24 |       480,
  25 |       496,
  26 |       512
  27 |     ],
  28 |     "padded_to_upper_budget": false,
  29 |     "scout_frozen": true,
  30 |     "utility_targets_detached": true
  31 |   },
  32 |   "method": "DUCA-Marginal-v1",
  33 |   "official_test_consumed": false,
  34 |   "oracle_allocation": {
  35 |     "video_validation_0000055": {
  36 |       "actual_budget_error": 0,
  37 |       "actual_cost": [
  38 |         384,
  39 |         384,
  40 |         384
  41 |       ],
  42 |       "budgets": [
  43 |         384,
  44 |         384,
  45 |         384
  46 |       ],
  47 |       "collapsed_to_k384": [
  48 |         false,
  49 |         false,
  50 |         false
  51 |       ],
  52 |       "execution_slots": [
  53 |         384,
  54 |         384,
  55 |         384
  56 |       ],
  57 |       "padding_slots": [
  58 |         0,
  59 |         0,
  60 |         0
  61 |       ],
  62 |       "predicted_total_utility": 0.0,
  63 |       "requested_budgets": [
  64 |         384,
  65 |         384,
  66 |         384
  67 |       ],
  68 |       "target_actual_cost": 1152,
  69 |       "window_count": 3
  70 |     },
  71 |     "video_validation_0000059": {
  72 |       "actual_budget_error": 0,
  73 |       "actual_cost": [
  74 |         384,
  75 |         256,
  76 |         512,
  77 |         384
  78 |       ],
  79 |       "budgets": [
  80 |         384,
  81 |         256,
  82 |         512,
  83 |         384
  84 |       ],
  85 |       "collapsed_to_k384": [
  86 |         false,
  87 |         false,
  88 |         false,
  89 |         false
  90 |       ],
  91 |       "execution_slots": [
  92 |         384,
  93 |         256,
  94 |         512,
  95 |         384
  96 |       ],
  97 |       "padding_slots": [
  98 |         0,
  99 |         0,
 100 |         0,
 101 |         0
 102 |       ],
 103 |       "predicted_total_utility": 0.1640600860118866,
 104 |       "requested_budgets": [
 105 |         384,
 106 |         256,
 107 |         512,
 108 |         384
 109 |       ],
 110 |       "target_actual_cost": 1536,
 111 |       "window_count": 4
 112 |     },
 113 |     "video_validation_0000158": {
 114 |       "actual_budget_error": 0,
 115 |       "actual_cost": [
 116 |         512,
 117 |         256,
 118 |         384,
 119 |         384,
 120 |         384
 121 |       ],
 122 |       "budgets": [
 123 |         512,
 124 |         256,
 125 |         384,
 126 |         384,
 127 |         384
 128 |       ],
 129 |       "collapsed_to_k384": [
 130 |         false,
 131 |         false,
 132 |         false,
 133 |         false,
 134 |         false
 135 |       ],
 136 |       "execution_slots": [
 137 |         512,
 138 |         256,
 139 |         384,
 140 |         384,
 141 |         384
 142 |       ],
 143 |       "padding_slots": [
 144 |         0,
 145 |         0,
 146 |         0,
 147 |         0,
 148 |         0
 149 |       ],
 150 |       "predicted_total_utility": 0.06998297572135925,
 151 |       "requested_budgets": [
 152 |         512,
 153 |         256,
 154 |         384,
 155 |         384,
 156 |         384
 157 |       ],
 158 |       "target_actual_cost": 1920,
 159 |       "window_count": 5
 160 |     },
 161 |     "video_validation_0000164": {
 162 |       "actual_budget_error": 0,
 163 |       "actual_cost": [
 164 |         384,
 165 |         384,
 166 |         384
 167 |       ],
 168 |       "budgets": [
 169 |         384,
 170 |         384,
 171 |         384
 172 |       ],
 173 |       "collapsed_to_k384": [
 174 |         false,
 175 |         false,
 176 |         false
 177 |       ],
 178 |       "execution_slots": [
 179 |         384,
 180 |         384,
 181 |         384
 182 |       ],
 183 |       "padding_slots": [
 184 |         0,
 185 |         0,
 186 |         0
 187 |       ],
 188 |       "predicted_total_utility": 0.0,
 189 |       "requested_budgets": [
 190 |         384,
 191 |         384,
 192 |         384
 193 |       ],
 194 |       "target_actual_cost": 1152,
 195 |       "window_count": 3
 196 |     },
 197 |     "video_validation_0000172": {
 198 |       "actual_budget_error": 0,
 199 |       "actual_cost": [
 200 |         384,
 201 |         384
 202 |       ],
 203 |       "budgets": [
 204 |         384,
 205 |         384
 206 |       ],
 207 |       "collapsed_to_k384": [
 208 |         false,
 209 |         false
 210 |       ],
 211 |       "execution_slots": [
 212 |         384,
 213 |         384
 214 |       ],
 215 |       "padding_slots": [
 216 |         0,
 217 |         0
 218 |       ],
 219 |       "predicted_total_utility": 0.0,
 220 |       "requested_budgets": [
 221 |         384,
 222 |         384
 223 |       ],
 224 |       "target_actual_cost": 768,
 225 |       "window_count": 2
 226 |     },
 227 |     "video_validation_0000173": {
 228 |       "actual_budget_error": 0,
 229 |       "actual_cost": [
 230 |         384,
 231 |         384,
 232 |         256,
 233 |         512
 234 |       ],
 235 |       "budgets": [
 236 |         384,
 237 |         384,
 238 |         256,
 239 |         512
 240 |       ],
 241 |       "collapsed_to_k384": [
 242 |         false,
 243 |         false,
 244 |         false,
 245 |         false
 246 |       ],
 247 |       "execution_slots": [
 248 |         384,
 249 |         384,
 250 |         256,
 251 |         512
 252 |       ],
 253 |       "padding_slots": [
 254 |         0,
 255 |         0,
 256 |         0,
 257 |         0
 258 |       ],
 259 |       "predicted_total_utility": 0.1362585425376892,
 260 |       "requested_budgets": [
 261 |         384,
 262 |         384,
 263 |         256,
 264 |         512
 265 |       ],
 266 |       "target_actual_cost": 1536,
 267 |       "window_count": 4
 268 |     },
 269 |     "video_validation_0000181": {
 270 |       "actual_budget_error": 0,
 271 |       "actual_cost": [
 272 |         384,
 273 |         384
 274 |       ],
 275 |       "budgets": [
 276 |         384,
 277 |         384
 278 |       ],
 279 |       "collapsed_to_k384": [
 280 |         false,
 281 |         false
 282 |       ],
 283 |       "execution_slots": [
 284 |         384,
 285 |         384
 286 |       ],
 287 |       "padding_slots": [
 288 |         0,
 289 |         0
 290 |       ],
 291 |       "predicted_total_utility": 0.0,
 292 |       "requested_budgets": [
 293 |         384,
 294 |         384
 295 |       ],
 296 |       "target_actual_cost": 768,
 297 |       "window_count": 2
 298 |     },
 299 |     "video_validation_0000186": {
 300 |       "actual_budget_error": 0,
 301 |       "actual_cost": [
 302 |         384
 303 |       ],
 304 |       "budgets": [
 305 |         384
 306 |       ],
 307 |       "collapsed_to_k384": [
 308 |         false
 309 |       ],
 310 |       "execution_slots": [
 311 |         384
 312 |       ],
 313 |       "padding_slots": [
 314 |         0
 315 |       ],
 316 |       "predicted_total_utility": 0.0,
 317 |       "requested_budgets": [
 318 |         384
 319 |       ],
 320 |       "target_actual_cost": 384,
 321 |       "window_count": 1
 322 |     },
 323 |     "video_validation_0000206": {
 324 |       "actual_budget_error": 0,
 325 |       "actual_cost": [
 326 |         384,
 327 |         384,
 328 |         384
 329 |       ],
 330 |       "budgets": [
 331 |         384,
 332 |         384,
 333 |         384
 334 |       ],
 335 |       "collapsed_to_k384": [
 336 |         false,
 337 |         false,
 338 |         false
 339 |       ],
 340 |       "execution_slots": [
 341 |         384,
 342 |         384,
 343 |         384
 344 |       ],
 345 |       "padding_slots": [
 346 |         0,
 347 |         0,
 348 |         0
 349 |       ],
 350 |       "predicted_total_utility": 0.0,
 351 |       "requested_budgets": [
 352 |         384,
 353 |         384,
 354 |         384
 355 |       ],
 356 |       "target_actual_cost": 1152,
 357 |       "window_count": 3
 358 |     },
 359 |     "video_validation_0000207": {
 360 |       "actual_budget_error": 0,
 361 |       "actual_cost": [
 362 |         384,
 363 |         384,
 364 |         384
 365 |       ],
 366 |       "budgets": [
 367 |         384,
 368 |         384,
 369 |         384
 370 |       ],
 371 |       "collapsed_to_k384": [
 372 |         false,
 373 |         false,
 374 |         false
 375 |       ],
 376 |       "execution_slots": [
 377 |         384,
 378 |         384,
 379 |         384
 380 |       ],
 381 |       "padding_slots": [
 382 |         0,
 383 |         0,
 384 |         0
 385 |       ],
 386 |       "predicted_total_utility": 0.0,
 387 |       "requested_budgets": [
 388 |         384,
 389 |         384,
 390 |         384
 391 |       ],
 392 |       "target_actual_cost": 1152,
 393 |       "window_count": 3
 394 |     },
 395 |     "video_validation_0000266": {
 396 |       "actual_budget_error": 0,
 397 |       "actual_cost": [
 398 |         384,
 399 |         384,
 400 |         384
 401 |       ],
 402 |       "budgets": [
 403 |         384,
 404 |         384,
 405 |         384
 406 |       ],
 407 |       "collapsed_to_k384": [
 408 |         false,
 409 |         false,
 410 |         false
 411 |       ],
 412 |       "execution_slots": [
 413 |         384,
 414 |         384,
 415 |         384
 416 |       ],
 417 |       "padding_slots": [
 418 |         0,
 419 |         0,
 420 |         0
 421 |       ],
 422 |       "predicted_total_utility": 0.0,
 423 |       "requested_budgets": [
 424 |         384,
 425 |         384,
 426 |         384
 427 |       ],
 428 |       "target_actual_cost": 1152,
 429 |       "window_count": 3
 430 |     },
 431 |     "video_validation_0000267": {
 432 |       "actual_budget_error": 0,
 433 |       "actual_cost": [
 434 |         512,
 435 |         384,
 436 |         256,
 437 |         384,
 438 |         384,
 439 |         384,
 440 |         384
 441 |       ],
 442 |       "budgets": [
 443 |         512,
 444 |         384,
 445 |         256,
 446 |         384,
 447 |         384,
 448 |         384,
 449 |         384
 450 |       ],
 451 |       "collapsed_to_k384": [
 452 |         false,
 453 |         false,
 454 |         false,
 455 |         false,
 456 |         false,
 457 |         false,
 458 |         false
 459 |       ],
 460 |       "execution_slots": [
 461 |         512,
 462 |         384,
 463 |         256,
 464 |         384,
 465 |         384,
 466 |         384,
 467 |         384
 468 |       ],
 469 |       "padding_slots": [
 470 |         0,
 471 |         0,
 472 |         0,
 473 |         0,
 474 |         0,
 475 |         0,
 476 |         0
 477 |       ],
 478 |       "predicted_total_utility": 0.10147061944007874,
 479 |       "requested_budgets": [
 480 |         512,
 481 |         384,
 482 |         256,
 483 |         384,
 484 |         384,
 485 |         384,
 486 |         384
 487 |       ],
 488 |       "target_actual_cost": 2688,
 489 |       "window_count": 7
 490 |     },
 491 |     "video_validation_0000282": {
 492 |       "actual_budget_error": 0,
 493 |       "actual_cost": [
 494 |         288
 495 |       ],
 496 |       "budgets": [
 497 |         384
 498 |       ],
 499 |       "collapsed_to_k384": [
 500 |         false
 501 |       ],
 502 |       "execution_slots": [
 503 |         384
 504 |       ],
 505 |       "padding_slots": [
 506 |         96
 507 |       ],
 508 |       "predicted_total_utility": 0.0,
 509 |       "requested_budgets": [
 510 |         384
 511 |       ],
 512 |       "target_actual_cost": 288,
 513 |       "window_count": 1
 514 |     },
 515 |     "video_validation_0000283": {
 516 |       "actual_budget_error": 0,
 517 |       "actual_cost": [
 518 |         384
 519 |       ],
 520 |       "budgets": [
 521 |         384
 522 |       ],
 523 |       "collapsed_to_k384": [
 524 |         false
 525 |       ],
 526 |       "execution_slots": [
 527 |         384
 528 |       ],
 529 |       "padding_slots": [
 530 |         0
 531 |       ],
 532 |       "predicted_total_utility": 0.0,
 533 |       "requested_budgets": [
 534 |         384
 535 |       ],
 536 |       "target_actual_cost": 384,
 537 |       "window_count": 1
 538 |     },
 539 |     "video_validation_0000285": {
 540 |       "actual_budget_error": 0,
 541 |       "actual_cost": [
 542 |         384,
 543 |         384
 544 |       ],
 545 |       "budgets": [
 546 |         384,
 547 |         384
 548 |       ],
 549 |       "collapsed_to_k384": [
 550 |         false,
 551 |         false
 552 |       ],
 553 |       "execution_slots": [
 554 |         384,
 555 |         384
 556 |       ],
 557 |       "padding_slots": [
 558 |         0,
 559 |         0
 560 |       ],
 561 |       "predicted_total_utility": 0.0,
 562 |       "requested_budgets": [
 563 |         384,
 564 |         384
 565 |       ],
 566 |       "target_actual_cost": 768,
 567 |       "window_count": 2
 568 |     },
 569 |     "video_validation_0000367": {
 570 |       "actual_budget_error": 0,
 571 |       "actual_cost": [
 572 |         384,
 573 |         384
 574 |       ],
 575 |       "budgets": [
 576 |         384,
 577 |         384
 578 |       ],
 579 |       "collapsed_to_k384": [
 580 |         false,
 581 |         false
 582 |       ],
 583 |       "execution_slots": [
 584 |         384,
 585 |         384
 586 |       ],
 587 |       "padding_slots": [
 588 |         0,
 589 |         0
 590 |       ],
 591 |       "predicted_total_utility": 0.0,
 592 |       "requested_budgets": [
 593 |         384,
 594 |         384
 595 |       ],
 596 |       "target_actual_cost": 768,
 597 |       "window_count": 2
 598 |     },
 599 |     "video_validation_0000417": {
 600 |       "actual_budget_error": 0,
 601 |       "actual_cost": [
 602 |         256,
 603 |         512,
 604 |         384,
 605 |         384,
 606 |         384,
 607 |         384
 608 |       ],
 609 |       "budgets": [
 610 |         256,
 611 |         512,
 612 |         384,
 613 |         384,
 614 |         384,
 615 |         384
 616 |       ],
 617 |       "collapsed_to_k384": [
 618 |         false,
 619 |         false,
 620 |         false,
 621 |         false,
 622 |         false,
 623 |         false
 624 |       ],
 625 |       "execution_slots": [
 626 |         256,
 627 |         512,
 628 |         384,
 629 |         384,
 630 |         384,
 631 |         384
 632 |       ],
 633 |       "padding_slots": [
 634 |         0,
 635 |         0,
 636 |         0,
 637 |         0,
 638 |         0,
 639 |         0
 640 |       ],
 641 |       "predicted_total_utility": 0.019536063075065613,
 642 |       "requested_budgets": [
 643 |         256,
 644 |         512,
 645 |         384,
 646 |         384,
 647 |         384,
 648 |         384
 649 |       ],
 650 |       "target_actual_cost": 2304,
 651 |       "window_count": 6
 652 |     },
 653 |     "video_validation_0000419": {
 654 |       "actual_budget_error": 0,
 655 |       "actual_cost": [
 656 |         384,
 657 |         384,
 658 |         384,
 659 |         256,
 660 |         384,
 661 |         384,
 662 |         384,
 663 |         384,
 664 |         384,
 665 |         512,
 666 |         256,
 667 |         256,
 668 |         512,
 669 |         512
 670 |       ],
 671 |       "budgets": [
 672 |         384,
 673 |         384,
 674 |         384,
 675 |         256,
 676 |         384,
 677 |         384,
 678 |         384,
 679 |         384,
 680 |         384,
 681 |         512,
 682 |         256,
 683 |         256,
 684 |         512,
 685 |         512
 686 |       ],
 687 |       "collapsed_to_k384": [
 688 |         false,
 689 |         false,
 690 |         false,
 691 |         false,
 692 |         false,
 693 |         false,
 694 |         false,
 695 |         false,
 696 |         false,
 697 |         false,
 698 |         false,
 699 |         false,
 700 |         false,
 701 |         false
 702 |       ],
 703 |       "execution_slots": [
 704 |         384,
 705 |         384,
 706 |         384,
 707 |         256,
 708 |         384,
 709 |         384,
 710 |         384,
 711 |         384,
 712 |         384,
 713 |         512,
 714 |         256,
 715 |         256,
 716 |         512,
 717 |         512
 718 |       ],
 719 |       "padding_slots": [
 720 |         0,
 721 |         0,
 722 |         0,
 723 |         0,
 724 |         0,
 725 |         0,
 726 |         0,
 727 |         0,
 728 |         0,
 729 |         0,
 730 |         0,
 731 |         0,
 732 |         0,
 733 |         0
 734 |       ],
 735 |       "predicted_total_utility": 0.10009123384952545,
 736 |       "requested_budgets": [
 737 |         384,
 738 |         384,
 739 |         384,
 740 |         256,
 741 |         384,
 742 |         384,
 743 |         384,
 744 |         384,
 745 |         384,
 746 |         512,
 747 |         256,
 748 |         256,
 749 |         512,
 750 |         512
 751 |       ],
 752 |       "target_actual_cost": 5376,
 753 |       "window_count": 14
 754 |     },
 755 |     "video_validation_0000483": {
 756 |       "actual_budget_error": 0,
 757 |       "actual_cost": [
 758 |         384,
 759 |         384
 760 |       ],
 761 |       "budgets": [
 762 |         384,
 763 |         384
 764 |       ],
 765 |       "collapsed_to_k384": [
 766 |         false,
 767 |         false
 768 |       ],
 769 |       "execution_slots": [
 770 |         384,
 771 |         384
 772 |       ],
 773 |       "padding_slots": [
 774 |         0,
 775 |         0
 776 |       ],
 777 |       "predicted_total_utility": 0.0,
 778 |       "requested_budgets": [
 779 |         384,
 780 |         384
 781 |       ],
 782 |       "target_actual_cost": 768,
 783 |       "window_count": 2
 784 |     },
 785 |     "video_validation_0000489": {
 786 |       "actual_budget_error": 0,
 787 |       "actual_cost": [
 788 |         384,
 789 |         384,
 790 |         384
 791 |       ],
 792 |       "budgets": [
 793 |         384,
 794 |         384,
 795 |         384
 796 |       ],
 797 |       "collapsed_to_k384": [
 798 |         false,
 799 |         false,
 800 |         false
 801 |       ],
 802 |       "execution_slots": [
 803 |         384,
 804 |         384,
 805 |         384
 806 |       ],
 807 |       "padding_slots": [
 808 |         0,
 809 |         0,
 810 |         0
 811 |       ],
 812 |       "predicted_total_utility": 0.0,
 813 |       "requested_budgets": [
 814 |         384,
 815 |         384,
 816 |         384
 817 |       ],
 818 |       "target_actual_cost": 1152,
 819 |       "window_count": 3
 820 |     },
 821 |     "video_validation_0000490": {
 822 |       "actual_budget_error": 0,
 823 |       "actual_cost": [
 824 |         384,
 825 |         384,
 826 |         512,
 827 |         256,
 828 |         384,
 829 |         384
 830 |       ],
 831 |       "budgets": [
 832 |         384,
 833 |         384,
 834 |         512,
 835 |         256,
 836 |         384,
 837 |         384
 838 |       ],
 839 |       "collapsed_to_k384": [
 840 |         false,
 841 |         false,
 842 |         false,
 843 |         false,
 844 |         false,
 845 |         false
 846 |       ],
 847 |       "execution_slots": [
 848 |         384,
 849 |         384,
 850 |         512,
 851 |         256,
 852 |         384,
 853 |         384
 854 |       ],
 855 |       "padding_slots": [
 856 |         0,
 857 |         0,
 858 |         0,
 859 |         0,
 860 |         0,
 861 |         0
 862 |       ],
 863 |       "predicted_total_utility": 0.13576892018318176,
 864 |       "requested_budgets": [
 865 |         384,
 866 |         384,
 867 |         512,
 868 |         256,
 869 |         384,
 870 |         384
 871 |       ],
 872 |       "target_actual_cost": 2304,
 873 |       "window_count": 6
 874 |     },
 875 |     "video_validation_0000664": {
 876 |       "actual_budget_error": 0,
 877 |       "actual_cost": [
 878 |         256,
 879 |         384,
 880 |         384,
 881 |         512
 882 |       ],
 883 |       "budgets": [
 884 |         256,
 885 |         384,
 886 |         384,
 887 |         512
 888 |       ],
 889 |       "collapsed_to_k384": [
 890 |         false,
 891 |         false,
 892 |         false,
 893 |         false
 894 |       ],
 895 |       "execution_slots": [
 896 |         256,
 897 |         384,
 898 |         384,
 899 |         512
 900 |       ],
 901 |       "padding_slots": [
 902 |         0,
 903 |         0,
 904 |         0,
 905 |         0
 906 |       ],
 907 |       "predicted_total_utility": 0.009859195910394192,
 908 |       "requested_budgets": [
 909 |         256,
 910 |         384,
 911 |         384,
 912 |         512
 913 |       ],
 914 |       "target_actual_cost": 1536,
 915 |       "window_count": 4
 916 |     },
 917 |     "video_validation_0000681": {
 918 |       "actual_budget_error": 0,
 919 |       "actual_cost": [
 920 |         384
 921 |       ],
 922 |       "budgets": [
 923 |         384
 924 |       ],
 925 |       "collapsed_to_k384": [
 926 |         false
 927 |       ],
 928 |       "execution_slots": [
 929 |         384
 930 |       ],
 931 |       "padding_slots": [
 932 |         0
 933 |       ],
 934 |       "predicted_total_utility": 0.0,
 935 |       "requested_budgets": [
 936 |         384
 937 |       ],
 938 |       "target_actual_cost": 384,
 939 |       "window_count": 1
 940 |     },
 941 |     "video_validation_0000683": {
 942 |       "actual_budget_error": 0,
 943 |       "actual_cost": [
 944 |         285
 945 |       ],
 946 |       "budgets": [
 947 |         384
 948 |       ],
 949 |       "collapsed_to_k384": [
 950 |         false
 951 |       ],
 952 |       "execution_slots": [
 953 |         384
 954 |       ],
 955 |       "padding_slots": [
 956 |         99
 957 |       ],
 958 |       "predicted_total_utility": 0.0,
 959 |       "requested_budgets": [
 960 |         384
 961 |       ],
 962 |       "target_actual_cost": 285,
 963 |       "window_count": 1
 964 |     },
 965 |     "video_validation_0000690": {
 966 |       "actual_budget_error": 0,
 967 |       "actual_cost": [
 968 |         384,
 969 |         256,
 970 |         384,
 971 |         512,
 972 |         384
 973 |       ],
 974 |       "budgets": [
 975 |         384,
 976 |         256,
 977 |         384,
 978 |         512,
 979 |         384
 980 |       ],
 981 |       "collapsed_to_k384": [
 982 |         false,
 983 |         false,
 984 |         false,
 985 |         false,
 986 |         false
 987 |       ],
 988 |       "execution_slots": [
 989 |         384,
 990 |         256,
 991 |         384,
 992 |         512,
 993 |         384
 994 |       ],
 995 |       "padding_slots": [
 996 |         0,
 997 |         0,
 998 |         0,
 999 |         0,
1000 |         0
1001 |       ],
1002 |       "predicted_total_utility": 0.004800617229193449,
1003 |       "requested_budgets": [
1004 |         384,
1005 |         256,
1006 |         384,
1007 |         512,
1008 |         384
1009 |       ],
1010 |       "target_actual_cost": 1920,
1011 |       "window_count": 5
1012 |     },
1013 |     "video_validation_0000783": {
1014 |       "actual_budget_error": 0,
1015 |       "actual_cost": [
1016 |         384,
1017 |         384,
1018 |         384
1019 |       ],
1020 |       "budgets": [
1021 |         384,
1022 |         384,
1023 |         384
1024 |       ],
1025 |       "collapsed_to_k384": [
1026 |         false,
1027 |         false,
1028 |         false
1029 |       ],
1030 |       "execution_slots": [
1031 |         384,
1032 |         384,
1033 |         384
1034 |       ],
1035 |       "padding_slots": [
1036 |         0,
1037 |         0,
1038 |         0
1039 |       ],
1040 |       "predicted_total_utility": 0.0,
1041 |       "requested_budgets": [
1042 |         384,
1043 |         384,
1044 |         384
1045 |       ],
1046 |       "target_actual_cost": 1152,
1047 |       "window_count": 3
1048 |     },
1049 |     "video_validation_0000851": {
1050 |       "actual_budget_error": 0,
1051 |       "actual_cost": [
1052 |         233
1053 |       ],
1054 |       "budgets": [
1055 |         384
1056 |       ],
1057 |       "collapsed_to_k384": [
1058 |         false
1059 |       ],
1060 |       "execution_slots": [
1061 |         384
1062 |       ],
1063 |       "padding_slots": [
1064 |         151
1065 |       ],
1066 |       "predicted_total_utility": 0.0,
1067 |       "requested_budgets": [
1068 |         384
1069 |       ],
1070 |       "target_actual_cost": 233,
1071 |       "window_count": 1
1072 |     },
1073 |     "video_validation_0000852": {
1074 |       "actual_budget_error": 0,
1075 |       "actual_cost": [
1076 |         384,
1077 |         384,
1078 |         384,
1079 |         384
1080 |       ],
1081 |       "budgets": [
1082 |         384,
1083 |         384,
1084 |         384,
1085 |         384
1086 |       ],
1087 |       "collapsed_to_k384": [
1088 |         false,
1089 |         false,
1090 |         false,
1091 |         false
1092 |       ],
1093 |       "execution_slots": [
1094 |         384,
1095 |         384,
1096 |         384,
1097 |         384
1098 |       ],
1099 |       "padding_slots": [
1100 |         0,
1101 |         0,
1102 |         0,
1103 |         0
1104 |       ],
1105 |       "predicted_total_utility": 0.0,
1106 |       "requested_budgets": [
1107 |         384,
1108 |         384,
1109 |         384,
1110 |         384
1111 |       ],
1112 |       "target_actual_cost": 1536,
1113 |       "window_count": 4
1114 |     },
1115 |     "video_validation_0000858": {
1116 |       "actual_budget_error": 0,
1117 |       "actual_cost": [
1118 |         384,
1119 |         384
1120 |       ],
1121 |       "budgets": [
1122 |         384,
1123 |         384
1124 |       ],
1125 |       "collapsed_to_k384": [
1126 |         false,
1127 |         false
1128 |       ],
1129 |       "execution_slots": [
1130 |         384,
1131 |         384
1132 |       ],
1133 |       "padding_slots": [
1134 |         0,
1135 |         0
1136 |       ],
1137 |       "predicted_total_utility": 0.0,
1138 |       "requested_budgets": [
1139 |         384,
1140 |         384
1141 |       ],
1142 |       "target_actual_cost": 768,
1143 |       "window_count": 2
1144 |     },
1145 |     "video_validation_0000905": {
1146 |       "actual_budget_error": 0,
1147 |       "actual_cost": [
1148 |         384,
1149 |         384,
1150 |         384
1151 |       ],
1152 |       "budgets": [
1153 |         384,
1154 |         384,
1155 |         384
1156 |       ],
1157 |       "collapsed_to_k384": [
1158 |         false,
1159 |         false,
1160 |         false
1161 |       ],
1162 |       "execution_slots": [
1163 |         384,
1164 |         384,
1165 |         384
1166 |       ],
1167 |       "padding_slots": [
1168 |         0,
1169 |         0,
1170 |         0
1171 |       ],
1172 |       "predicted_total_utility": 0.0,
1173 |       "requested_budgets": [
1174 |         384,
1175 |         384,
1176 |         384
1177 |       ],
1178 |       "target_actual_cost": 1152,
1179 |       "window_count": 3
1180 |     },
1181 |     "video_validation_0000908": {
1182 |       "actual_budget_error": 0,
1183 |       "actual_cost": [
1184 |         384,
1185 |         384,
1186 |         384,
1187 |         384,
1188 |         384
1189 |       ],
1190 |       "budgets": [
1191 |         384,
1192 |         384,
1193 |         384,
1194 |         384,
1195 |         384
1196 |       ],
1197 |       "collapsed_to_k384": [
1198 |         false,
1199 |         false,
1200 |         false,
1201 |         false,
1202 |         false
1203 |       ],
1204 |       "execution_slots": [
1205 |         384,
1206 |         384,
1207 |         384,
1208 |         384,
1209 |         384
1210 |       ],
1211 |       "padding_slots": [
1212 |         0,
1213 |         0,
1214 |         0,
1215 |         0,
1216 |         0
1217 |       ],
1218 |       "predicted_total_utility": 0.0,
1219 |       "requested_budgets": [
1220 |         384,
1221 |         384,
1222 |         384,
1223 |         384,
1224 |         384
1225 |       ],
1226 |       "target_actual_cost": 1920,
1227 |       "window_count": 5
1228 |     },
1229 |     "video_validation_0000909": {
1230 |       "actual_budget_error": 0,
1231 |       "actual_cost": [
1232 |         384,
1233 |         384,
1234 |         384
1235 |       ],
1236 |       "budgets": [
1237 |         384,
1238 |         384,
1239 |         384
1240 |       ],
1241 |       "collapsed_to_k384": [
1242 |         false,
1243 |         false,
1244 |         false
1245 |       ],
1246 |       "execution_slots": [
1247 |         384,
1248 |         384,
1249 |         384
1250 |       ],
1251 |       "padding_slots": [
1252 |         0,
1253 |         0,
1254 |         0
1255 |       ],
1256 |       "predicted_total_utility": 0.0,
1257 |       "requested_budgets": [
1258 |         384,
1259 |         384,
1260 |         384
1261 |       ],
1262 |       "target_actual_cost": 1152,
1263 |       "window_count": 3
1264 |     },
1265 |     "video_validation_0000937": {
1266 |       "actual_budget_error": 0,
1267 |       "actual_cost": [
1268 |         384
1269 |       ],
1270 |       "budgets": [
1271 |         384
1272 |       ],
1273 |       "collapsed_to_k384": [
1274 |         false
1275 |       ],
1276 |       "execution_slots": [
1277 |         384
1278 |       ],
1279 |       "padding_slots": [
1280 |         0
1281 |       ],
1282 |       "predicted_total_utility": 0.0,
1283 |       "requested_budgets": [
1284 |         384
1285 |       ],
1286 |       "target_actual_cost": 384,
1287 |       "window_count": 1
1288 |     },
1289 |     "video_validation_0000938": {
1290 |       "actual_budget_error": 0,
1291 |       "actual_cost": [
1292 |         260
1293 |       ],
1294 |       "budgets": [
1295 |         384
1296 |       ],
1297 |       "collapsed_to_k384": [
1298 |         false
1299 |       ],
1300 |       "execution_slots": [
1301 |         384
1302 |       ],
1303 |       "padding_slots": [
1304 |         124
1305 |       ],
1306 |       "predicted_total_utility": 0.0,
1307 |       "requested_budgets": [
1308 |         384
1309 |       ],
1310 |       "target_actual_cost": 260,
1311 |       "window_count": 1
1312 |     },
1313 |     "video_validation_0000943": {
1314 |       "actual_budget_error": 0,
1315 |       "actual_cost": [
1316 |         384,
1317 |         384
1318 |       ],
1319 |       "budgets": [
1320 |         384,
1321 |         384
1322 |       ],
1323 |       "collapsed_to_k384": [
1324 |         false,
1325 |         false
1326 |       ],
1327 |       "execution_slots": [
1328 |         384,
1329 |         384
1330 |       ],
1331 |       "padding_slots": [
1332 |         0,
1333 |         0
1334 |       ],
1335 |       "predicted_total_utility": 0.0,
1336 |       "requested_budgets": [
1337 |         384,
1338 |         384
1339 |       ],
1340 |       "target_actual_cost": 768,
1341 |       "window_count": 2
1342 |     },
1343 |     "video_validation_0000944": {
1344 |       "actual_budget_error": 0,
1345 |       "actual_cost": [
1346 |         384,
1347 |         384,
1348 |         384
1349 |       ],
1350 |       "budgets": [
1351 |         384,
1352 |         384,
1353 |         384
1354 |       ],
1355 |       "collapsed_to_k384": [
1356 |         false,
1357 |         false,
1358 |         false
1359 |       ],
1360 |       "execution_slots": [
1361 |         384,
1362 |         384,
1363 |         384
1364 |       ],
1365 |       "padding_slots": [
1366 |         0,
1367 |         0,
1368 |         0
1369 |       ],
1370 |       "predicted_total_utility": 0.0,
1371 |       "requested_budgets": [
1372 |         384,
1373 |         384,
1374 |         384
1375 |       ],
1376 |       "target_actual_cost": 1152,
1377 |       "window_count": 3
1378 |     },
1379 |     "video_validation_0000945": {
1380 |       "actual_budget_error": 0,
1381 |       "actual_cost": [
1382 |         384,
1383 |         384,
1384 |         384
1385 |       ],
1386 |       "budgets": [
1387 |         384,
1388 |         384,
1389 |         384
1390 |       ],
1391 |       "collapsed_to_k384": [
1392 |         false,
1393 |         false,
1394 |         false
1395 |       ],
1396 |       "execution_slots": [
1397 |         384,
1398 |         384,
1399 |         384
1400 |       ],
1401 |       "padding_slots": [
1402 |         0,
1403 |         0,
1404 |         0
1405 |       ],
1406 |       "predicted_total_utility": 0.0,
1407 |       "requested_budgets": [
1408 |         384,
1409 |         384,
1410 |         384
1411 |       ],
1412 |       "target_actual_cost": 1152,
1413 |       "window_count": 3
1414 |     },
1415 |     "video_validation_0000984": {
1416 |       "actual_budget_error": 0,
1417 |       "actual_cost": [
1418 |         348
1419 |       ],
1420 |       "budgets": [
1421 |         384
1422 |       ],
1423 |       "collapsed_to_k384": [
1424 |         false
1425 |       ],
1426 |       "execution_slots": [
1427 |         384
1428 |       ],
1429 |       "padding_slots": [
1430 |         36
1431 |       ],
1432 |       "predicted_total_utility": 0.0,
1433 |       "requested_budgets": [
1434 |         384
1435 |       ],
1436 |       "target_actual_cost": 348,
1437 |       "window_count": 1
1438 |     },
1439 |     "video_validation_0000988": {
1440 |       "actual_budget_error": 0,
1441 |       "actual_cost": [
1442 |         384,
1443 |         384
1444 |       ],
1445 |       "budgets": [
1446 |         384,
1447 |         384
1448 |       ],
1449 |       "collapsed_to_k384": [
1450 |         false,
1451 |         false
1452 |       ],
1453 |       "execution_slots": [
1454 |         384,
1455 |         384
1456 |       ],
1457 |       "padding_slots": [
1458 |         0,
1459 |         0
1460 |       ],
1461 |       "predicted_total_utility": 0.0,
1462 |       "requested_budgets": [
1463 |         384,
1464 |         384
1465 |       ],
1466 |       "target_actual_cost": 768,
1467 |       "window_count": 2
1468 |     },
1469 |     "video_validation_0000990": {
1470 |       "actual_budget_error": 0,
1471 |       "actual_cost": [
1472 |         384,
1473 |         384
1474 |       ],
1475 |       "budgets": [
1476 |         384,
1477 |         384
1478 |       ],
1479 |       "collapsed_to_k384": [
1480 |         false,
1481 |         false
1482 |       ],
1483 |       "execution_slots": [
1484 |         384,
1485 |         384
1486 |       ],
1487 |       "padding_slots": [
1488 |         0,
1489 |         0
1490 |       ],
1491 |       "predicted_total_utility": 0.0,
1492 |       "requested_budgets": [
1493 |         384,
1494 |         384
1495 |       ],
1496 |       "target_actual_cost": 768,
1497 |       "window_count": 2
1498 |     }
1499 |   },
1500 |   "oracle_headroom": {
1501 |     "delta_average_mAP_pp": 0.725589312687891,
1502 |     "delta_mAP_at_0.7_pp": 0.7290037014890505,
1503 |     "gray_zone_requires_pro": true,
1504 |     "no_headroom_boundary": false,
1505 |     "strong_gate_pass": false
1506 |   },
1507 |   "oracle_reallocate_384": {
1508 |     "average_mAP": 0.888567863693566,
1509 |     "mAP@0.3": 0.960349418437806,
1510 |     "mAP@0.4": 0.9399008234073724,
1511 |     "mAP@0.5": 0.9104458544593358,
1512 |     "mAP@0.6": 0.8621473535441421,
1513 |     "mAP@0.7": 0.7699958686191737
1514 |   },
1515 |   "paper_claim_allowed": false,
1516 |   "schema": "duca_marginal_frozen_h65_probe_result_v1",
1517 |   "secondary_mean_k320": {
1518 |     "reason": "K in {256,384} cannot guarantee exact mean K=320 for videos with an odd number of windows without an additional frozen rule",
1519 |     "status": "NOT_RUN_UNRESOLVED_EXACT_VIDEO_BUDGET"
1520 |   },
1521 |   "source": {
1522 |     "annotation": "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json",
1523 |     "annotation_sha256": "ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad",
1524 |     "checkpoint": "/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth",
1525 |     "checkpoint_epoch": 59,
1526 |     "checkpoint_sha256": "dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c",
1527 |     "checkpoint_state_key": "state_dict_ema",
1528 |     "class_map": "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt",
1529 |     "class_map_sha256": "a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31",
1530 |     "config": "/data/run01/sczc063/yuzibo/duca_marginal_f67d96fd_20260831/configs/adatad/thumos/duca_marginal_frozen_h65_probe.py",
1531 |     "config_sha256": "02e091995124496c5c5e0011923ac12de6b8dea29679ec821e64e6e5b6271ca6",
1532 |     "git": {
1533 |       "branch": "feature/duca-marginal-budget-v1-20260830",
1534 |       "dirty": false,
1535 |       "head": "f67d96fdf68a295eaa7f678f3dfc125530828889"
1536 |     },
1537 |     "train_data": "/data/run01/sczc063/yuzibo/thumos14/raw_data/video",
1538 |     "videomae_pretrain": "/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
1539 |     "videomae_pretrain_sha256": "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
1540 |   },
1541 |   "stage_artifacts": {
1542 |     "k256": {
1543 |       "path": "/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/counterfactual_k256.jsonl.gz",
1544 |       "sha256": "6dc8893a41b5c8132b176f32133ffc2f48a5491146385c147b8227167608a309"
1545 |     },
1546 |     "k512": {
1547 |       "path": "/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/counterfactual_k512.jsonl.gz",
1548 |       "sha256": "c7fa06258c07163d0906b512a78e367c27607c64fc41b28bce9fe51fbd0815d7"
1549 |     },
1550 |     "selection": {
1551 |       "path": "/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/selection_k384.jsonl.gz",
1552 |       "sha256": "1d668d4e5eb4b5ef3c1057c97ec63cc2c1eed3c0e62297290520063b4e1ec38f"
1553 |     }
1554 |   },
1555 |   "status": "ORACLE_HEADROOM_GRAY_ZONE_RETURN_TO_PRO"
1556 | }
```

### File: .cvpr-pro-lab/pro-reviews/runs/duca-marginal-oracle-gray-zone-v001/materials/selection_k384_receipt.json
Lines: 1-36
```json
 1 | {
 2 |   "artifact": "/data/run01/sczc063/yuzibo/duca_marginal_prerun_f87555f7_20260831/selection_k384.jsonl.gz",
 3 |   "artifact_sha256": "1d668d4e5eb4b5ef3c1057c97ec63cc2c1eed3c0e62297290520063b4e1ec38f",
 4 |   "checkpoint_payload_epoch": 59,
 5 |   "detector_frozen": true,
 6 |   "fit_video_count": 160,
 7 |   "frozen_loss_normalizer": 45.0,
 8 |   "holdout_video_count": 40,
 9 |   "k384_prediction_exact_all_windows": true,
10 |   "k384_selection_bit_exact_all_windows": true,
11 |   "scout_frozen": true,
12 |   "source": {
13 |     "annotation": "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json",
14 |     "annotation_sha256": "ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad",
15 |     "checkpoint": "/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth",
16 |     "checkpoint_epoch": 59,
17 |     "checkpoint_sha256": "dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c",
18 |     "checkpoint_state_key": "state_dict_ema",
19 |     "class_map": "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt",
20 |     "class_map_sha256": "a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31",
21 |     "config": "/data/run01/sczc063/yuzibo/duca_marginal_f87555f7_20260831/configs/adatad/thumos/duca_marginal_frozen_h65_probe.py",
22 |     "config_sha256": "02e091995124496c5c5e0011923ac12de6b8dea29679ec821e64e6e5b6271ca6",
23 |     "git": {
24 |       "branch": "HEAD",
25 |       "dirty": false,
26 |       "head": "f87555f7da362fe1a20d4ca08f7a68c975ed8280"
27 |     },
28 |     "train_data": "/data/run01/sczc063/yuzibo/thumos14/raw_data/video",
29 |     "videomae_pretrain": "/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
30 |     "videomae_pretrain_sha256": "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
31 |   },
32 |   "stage": "select-k384",
33 |   "status": "SELECTION_K384_COMPLETE",
34 |   "video_count": 200,
35 |   "window_count": 720
36 | }
```

### File: .cvpr-pro-lab/pro-reviews/runs/duca-marginal-oracle-gray-zone-v001/materials/counterfactual_k256_receipt.json
Lines: 1-40
```json
 1 | {
 2 |   "artifact": "/data/run01/sczc063/yuzibo/duca_marginal_prerun_f87555f7_20260831/counterfactual_k256.jsonl.gz",
 3 |   "artifact_sha256": "6dc8893a41b5c8132b176f32133ffc2f48a5491146385c147b8227167608a309",
 4 |   "checkpoint_payload_epoch": 59,
 5 |   "collapsed_alias_count": 17,
 6 |   "detector_frozen": true,
 7 |   "distinct_forward_count": 703,
 8 |   "frozen_loss_normalizers": [
 9 |     45.0
10 |   ],
11 |   "observed_execution_slot_classes": [
12 |     256
13 |   ],
14 |   "padded_to_k512": false,
15 |   "requested_budget": 256,
16 |   "source": {
17 |     "annotation": "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json",
18 |     "annotation_sha256": "ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad",
19 |     "checkpoint": "/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth",
20 |     "checkpoint_epoch": 59,
21 |     "checkpoint_sha256": "dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c",
22 |     "checkpoint_state_key": "state_dict_ema",
23 |     "class_map": "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt",
24 |     "class_map_sha256": "a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31",
25 |     "config": "/data/run01/sczc063/yuzibo/duca_marginal_f87555f7_20260831/configs/adatad/thumos/duca_marginal_frozen_h65_probe.py",
26 |     "config_sha256": "02e091995124496c5c5e0011923ac12de6b8dea29679ec821e64e6e5b6271ca6",
27 |     "git": {
28 |       "branch": "HEAD",
29 |       "dirty": false,
30 |       "head": "f87555f7da362fe1a20d4ca08f7a68c975ed8280"
31 |     },
32 |     "train_data": "/data/run01/sczc063/yuzibo/thumos14/raw_data/video",
33 |     "videomae_pretrain": "/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
34 |     "videomae_pretrain_sha256": "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
35 |   },
36 |   "stage": "counterfactual-k256",
37 |   "status": "COUNTERFACTUAL_K256_COMPLETE",
38 |   "video_count": 200,
39 |   "window_count": 720
40 | }
```

### File: .cvpr-pro-lab/pro-reviews/runs/duca-marginal-oracle-gray-zone-v001/materials/counterfactual_k512_receipt.json
Lines: 1-45
```json
 1 | {
 2 |   "artifact": "/data/run01/sczc063/yuzibo/duca_marginal_prerun_f87555f7_20260831/counterfactual_k512.jsonl.gz",
 3 |   "artifact_sha256": "c7fa06258c07163d0906b512a78e367c27607c64fc41b28bce9fe51fbd0815d7",
 4 |   "checkpoint_payload_epoch": 59,
 5 |   "collapsed_alias_count": 30,
 6 |   "detector_frozen": true,
 7 |   "distinct_forward_count": 690,
 8 |   "frozen_loss_normalizers": [
 9 |     45.0
10 |   ],
11 |   "observed_execution_slot_classes": [
12 |     400,
13 |     448,
14 |     464,
15 |     480,
16 |     496,
17 |     512
18 |   ],
19 |   "padded_to_k512": false,
20 |   "requested_budget": 512,
21 |   "source": {
22 |     "annotation": "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json",
23 |     "annotation_sha256": "ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad",
24 |     "checkpoint": "/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth",
25 |     "checkpoint_epoch": 59,
26 |     "checkpoint_sha256": "dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c",
27 |     "checkpoint_state_key": "state_dict_ema",
28 |     "class_map": "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt",
29 |     "class_map_sha256": "a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31",
30 |     "config": "/data/run01/sczc063/yuzibo/duca_marginal_f87555f7_20260831/configs/adatad/thumos/duca_marginal_frozen_h65_probe.py",
31 |     "config_sha256": "02e091995124496c5c5e0011923ac12de6b8dea29679ec821e64e6e5b6271ca6",
32 |     "git": {
33 |       "branch": "HEAD",
34 |       "dirty": false,
35 |       "head": "f87555f7da362fe1a20d4ca08f7a68c975ed8280"
36 |     },
37 |     "train_data": "/data/run01/sczc063/yuzibo/thumos14/raw_data/video",
38 |     "videomae_pretrain": "/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
39 |     "videomae_pretrain_sha256": "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
40 |   },
41 |   "stage": "counterfactual-k512",
42 |   "status": "COUNTERFACTUAL_K512_COMPLETE",
43 |   "video_count": 200,
44 |   "window_count": 720
45 | }
```

### File: .cvpr-pro-lab/pro-reviews/runs/duca-marginal-oracle-gray-zone-v001/materials/job1262077_failure_all.json
Lines: 1-6
```json
1 | {
2 |   "error": "Expecting value: line 1 column 1 (char 0)",
3 |   "error_type": "JSONDecodeError",
4 |   "stage": "all",
5 |   "status": "DUCA_MARGINAL_PROBE_FAILED"
6 | }
```

## Answer

# 唯一裁决：**REVISE**

**Nonce：`DUCA-MARGINAL-ORACLE-GRAY-ZONE-ADJUDICATION-v001-20260831`**

当前灰区诊断**准入**，但只准入为训练侧留出集合上的机制诊断；**现在不允许训练 utility head**。唯一下一任务是：不增加预算档位、不运行新的 detector forward、不训练任何模型，只解除“每视频最多改变 `floor(0.5N)` 个窗口”这一项约束，使用现有密封 K256/K384/K512 产物执行一次只读 oracle 证伪。

---

## 一、证据准入裁决

### 1. 双提交来源足以准入，但必须按双阶段来源表述

当前分支确实指向精确提交 `f67d96fdf68a295eaa7f678f3dfc125530828889`，其直接父提交是 producer 使用的 `f87555f7da362fe1a20d4ca08f7a68c975ed8280`。

从 `f87555f7...` 到 `f67d96fd...` 只改动：

* `tools/bata/run_duca_marginal_frozen_h65_probe.py`：把换行文本 block-list 确定性转换成 evaluator 接受的 JSON 数组；
* `tests/test_duca_marginal_budget.py`：增加对应回归测试。

提交没有修改模型前向、三档选择、反事实损失、分配器、配置、检查点、数据、NMS、评估指标或科学阈值。

因此，严格审查提出的“producer commit 与 summary commit 不完全一致”是一个**必须显式披露的来源边界**，但不是足以否定结果的科学 blocker。正确表述是：

> K384/K256/K512 producer 产物生成于 `f87555f7...`；`f67d96fd...` 在模型、配置、数据和 producer 语义不变的条件下，只重新执行身份绑定和汇总评估。

不得把三个 producer 重新标成由 `f67d96fd...` 生成，也不得把整条计算链简写为“全部运行于 f67d96fd”。

### 2. 不需要重跑 producer

最小必要修复本应是：

1. 在 `f67d96fd...` 上重新通过 PRE_RUN；
2. 读取原密封 producer 产物；
3. 用修复后的 JSON block-list 重新执行 `summarize`；
4. 保留 producer receipts 的 `f87555f7...` 来源和产物哈希。

Job `1262098` 已经完成这四项。重新执行三个重型 producer 不会增加科学信息，反而只会引入无意义的重复计算。因此不要求重算、不建设额外来源框架，也不增加新的哈希系统。

### 3. 准入级别

这份结果的证据级别固定为：

* **实现与来源：准入；**
* **训练侧留出集合机制诊断：准入；**
* **正式论文主结果：不准入；**
* **官方 validation/test 性能结论：不准入；**
* **显著性或总体效应结论：不准入。**

附件中的代码库存边界、角色规则、Evaluator 原始证据、项目规则、论文优先流程、旧问询、材料说明、旧研究状态和历史版本表只用于约束证据层级与执行方式；它们不覆盖本轮 `f67d96fd...` 和原始 JSON 的最新事实。        

---

## 二、灰区结果的科学解释

### 1. 已观察到的事实

在固定的 40 个训练侧 utility-holdout 视频、124 个窗口上：

* `Fixed-H65-384`：Avg-mAP `88.131197%`，mAP@0.7 `76.270583%`；
* `Oracle-Reallocate-384`：Avg-mAP `88.856786%`，mAP@0.7 `76.999587%`；
* 增益为 Avg-mAP `+0.725589` 个百分点、mAP@0.7 `+0.729004` 个百分点；
* 相比强门槛，分别还差 `0.074411` 和 `0.270996` 个百分点；
* 结果同时高于无 headroom 边界，因此既不是强通过，也不是近零结果；
* 124 个窗口中，102 个保持 K384，11 个降为 K256，11 个升为 K512；实际改变 22 个窗口，占 `17.74%`；
* 发生非基线分配的 9 个视频中，有 5 个视频恰好达到其 `floor(0.5N)` 改变窗口上限；
* 每视频实际 observation 总预算精确不变，整体实际成本为 `47110`；
* detector、Scout、utility targets 与执行长度合同均通过。

分配器当前确实按每视频精确实际成本运行，并把 `max_changed_fraction` 纳入动态规划；汇总入口则把该参数硬编码为 `0.5`。

### 2. 可以作出的推断

最有证据支持的解释是：

> **冻结 H65 优先序列内部存在局部、有限的跨窗口预算异质性，但在当前三档预算和 50% 改变比例约束下，机制特定 oracle 的收益尚不足以支持学习阶段。**

具体而言：

* 结果明显高于无 headroom 边界，因而不能说“不同窗口完全没有不同的边际计算价值”。
* 只有 22 个窗口发生改变，说明有效信号是**稀疏和局部的**，不是大多数窗口普遍需要动态预算。
* 五个发生重分配的视频达到改变比例上限，使“50% 上限截断了一部分可用转移”成为有原始证据依据的解释。
* 但“达到上限”不等于证明解除上限一定增加 mAP；在无约束重算前，它仍只是一个待证伪解释。
* 当前 oracle 使用真实冻结 detector loss 构造边际效用，而不是直接全局优化 mAP。因此它是当前 utility-head 机制的**操作性上界诊断**，不是所有可能动态预算方法的数学 mAP 上界。
* 由于已实现 utility head 的目标就是近似这组 loss-based utility，一个尚未通过强门槛的真实效用 oracle 不足以支持继续承担预测误差、分配误差和有限样本误差。

### 3. 当前仍未知的事项

现有结果不能判断：

* `+0.726/+0.729` 是否由少量视频驱动，或其整视频配对区间是否跨零；
* 解除 50% 改变上限后是否能够通过原强门槛；
* K256/K384/K512 三档量化本身是否是主要限制；
* Scout 特征能否预测 downgrade penalty 和 upgrade gain；
* 学习分配能恢复多少 oracle 增益；
* official validation/test 上是否存在收益；
* 实际端到端延迟、吞吐量或 FLOPs 是否改善。

特别是，**当前数据不支持把三档预算称为失败根因**。验证更多档位需要新的 producer forward，并会演化成事后预算搜索，因此本轮禁止执行。

---

## 三、utility head 裁决

**不允许训练。**

当前代码只在 Avg-mAP 同时达到 `+0.8`、mAP@0.7 同时达到 `+1.0` 后进入 `_fit_utility_head`；灰区正确地停在训练之前。

现在直接训练 utility head 会产生三个问题：

1. 违背冻结的强 headroom 前置门槛；
2. 在机制特定 oracle 尚不足时，用预测器误差进一步压缩本已有限的收益；
3. 把“是否存在足够预算空间”和“预算空间是否可预测”两个问题混为一次实验。

即使下一项只读证伪通过，也只返回 Pro 再次裁决；本任务内仍不得训练 utility head。

---

## 四、唯一冻结任务

### 任务名称

**解除 50% 改变窗口上限的只读 oracle 证伪**

### 科学问题

当前灰区主要来自跨窗口边际效用本身不足，还是来自每视频最多改变 `floor(0.5N)` 个窗口这一人为约束截断了有价值的预算转移？

### 唯一干预

把 oracle 分配的：

```text
max_changed_fraction = 0.5
```

改为：

```text
max_changed_fraction = 1.0
```

除此之外任何科学变量均不得变化。

### 可证伪预测

若 50% 上限确实是主要限制，那么在：

* 相同 K256/K384/K512 产物；
* 相同真实反事实 utility；
* 相同每视频实际 observation 总预算；
* 相同 tie-break；
* 相同 NMS 和 evaluator；

条件下，解除上限应使 oracle 同时达到原冻结强门槛：

* `ΔAvg-mAP ≥ +0.8` 个百分点；
* `ΔmAP@0.7 ≥ +1.0` 个百分点。

### 相反解释

若解除上限后仍未同时通过两项强门槛，则当前收益上限主要不是由 50% 约束造成，而是来自：

* 冻结 H65 priority sequence 中可交换预算的真实价值有限；
* 三档 loss-based utility 与最终 mAP 的对齐有限；
* 或当前三档机制本身的表达能力有限。

此时终止 `DUCA-Marginal-v1` 当前机制，不追加预算档位、utility-head 训练或比例搜索。

### 任务性质

这是**只读诊断**，不是正式论文实验：

* 不训练任何模型；
* 不访问 official test；
* 不运行新的 detector/Scout forward；
* 不生成 K320 或其他预算；
* 不需要 GPU；
* 只消费已密封的三个 producer 产物。

---

## 五、实现边界

### 权威基座

* Repository：`yuzbo/OpenTAD_C3_CoarseClean_20260702`
* Base revision：`f67d96fdf68a295eaa7f678f3dfc125530828889`
* Base branch：`feature/duca-marginal-budget-v1-20260830`
* 新分支：`feature/duca-marginal-cap-release-falsifier-v1-20260831`

### 允许修改的文件与符号

只允许修改两个文件：

1. `tools/bata/run_duca_marginal_frozen_h65_probe.py`

   * 给 `_allocate_rows_by_video` 增加显式 `max_changed_fraction` 参数，默认值必须保持 `0.5`；
   * 原 `run_summary_stage` 的现有结果路径必须保持不变；
   * 增加一个独立只读诊断入口，使用 `max_changed_fraction=1.0`；
   * 输出单独的 `oracle_cap_release_result.json`，不得覆盖原始 `probe_result.json`；
   * 复用 `_official_holdout_metrics`、现有 NMS 和现有 sealed predictions。

2. `tests/test_duca_marginal_budget.py`

   * 验证默认 `0.5` 路径不变；
   * 验证 `1.0` 在合成四窗口样例中允许两组正效用预算转移；
   * 验证实际成本严格等于 all-K384 目标；
   * 验证 tie-break 仍是“最大 utility → 较少改变 → 固定档位字典序”；
   * 验证原 `probe_result.json` 不被写回。

现有 allocator 已经支持任意合法 `max_changed_fraction`，因此不得修改 `allocate_video_budgets_exact` 的数学实现。现有聚焦测试也已经覆盖短窗口别名、真实 observation 成本、部分末包 padding、detached targets 和 evaluator block-list 修复。

### 明确禁止修改

不得修改：

* `opentad/models/duca/dynamic_budget.py`；
* `opentad/models/duca/counterfactual_utility.py`；
* H65 selector、priority sequence 或嵌套集合；
* detector、Scout、loss normalizer、VideoMAE、projection、head；
* detector grid 与物理时间映射；
* K256/K384/K512 三档定义；
* 160/40 split、seed 3407；
* annotation、class map、checkpoint、预训练权重；
* NMS、官方 evaluator、指标；
* 强 headroom 与无 headroom 数值门槛；
* producer artifacts 及其 receipts；
* utility-head 结构、训练轮数或优化器。

不得新建通用 bootstrap 框架、调度系统、来源系统、配置族或兼容层。

---

## 六、评估协议与停止条件

### 阶段 1：原结果复现门

用原 sealed artifacts 重新执行 `max_changed_fraction=0.5` 的汇总，必须满足：

* Fixed 与 capped-oracle 所有 mAP 值相对原 `probe_result.json` 误差不超过 `1e-6` 个百分点；
* K256/K384/K512 计数保持 `11/102/11`；
* 所有 artifact SHA 与原 receipts 一致；
* 每视频预算误差为零。

不满足时属于实现或产物绑定错误，只做一次最小修复，不产生科学解释。

### 阶段 2：解除上限的点估计

在 `max_changed_fraction=1.0` 下报告：

* Fixed 与 cap-release oracle 的 mAP@0.3–0.7；
* Avg-mAP；
* 相对 Fixed 的两项主增益；
* 相对原 capped oracle 的增量；
* K256/K384/K512 窗口数；
* 改变窗口数、改变视频数；
* capped 方案中达到上限的视频数；
* cap-release 后的每视频实际成本误差；
* 原始与新分配的确定性摘要。

### 阶段 3：条件式配对区间

只有 cap-release **点估计同时通过原强门槛**时，才执行：

* 40 个视频的整视频配对 bootstrap；
* 10,000 次；
* seed `3407`；
* Fixed 与 cap-release oracle 使用完全相同的重采样索引；
* 每次重新计算 Avg-mAP 和 mAP@0.7；
* 报告 2.5%/97.5% percentile 区间。

该区间不修改原强门槛，只作为继续投入的证据准入条件。

### 唯一判定规则

| 结果                          | 科学处理                                                           |
| --------------------------- | -------------------------------------------------------------- |
| cap-release 任一主点指标未通过原强门槛   | 终止当前 `DUCA-Marginal-v1`；不运行 bootstrap，不训练 utility head，不测试更多档位 |
| 两项点指标通过，但任一 95% 配对区间下界不高于 0 | 终止当前机制；证据不足以承担 predictor 误差                                    |
| 两项点指标通过，且两项区间下界均高于 0        | 返回 Pro 进行新的 utility-head 任务冻结；本任务仍不得训练                         |
| 实际成本、数据身份、原结果复现或实现门失败       | 仅视为实现失败；最小修复后重做同一只读诊断                                          |

---

## 七、Builder → 独立 Critic → Evaluator 交接

### Builder

交付：

* 从 `f67d96fd...` 派生的单一 clean commit；
* 只包含上述两个文件；
* 原 capped 路径不变；
* 新 cap-release 入口及独立结果文件；
* 聚焦测试通过。

截止：**2026-08-31 18:00:00 +08:00**

### 独立 Critic

只审查：

* 是否真的只改变 `max_changed_fraction`；
* 原 `0.5` 路径是否保持默认及结果兼容；
* 是否没有新 detector forward、utility-head 训练或 official-test 访问；
* 每视频实际成本、短窗口 alias、tie-break 是否保持；
* 是否没有覆盖或重标 producer artifacts。

不得因代码风格、额外日志或通用完备性制造修复循环。

截止：**2026-08-31 22:00:00 +08:00**

### Evaluator

在 Critic 通过的同一 clean commit 上：

1. 验证 sealed artifact SHA；
2. 精确复现 capped 结果；
3. 执行 cap-release 只读汇总；
4. 仅在点门通过时运行配对 bootstrap；
5. 返回原始 JSON、指标、分配和科学判定，不做 utility-head 训练。

截止：**2026-09-01 12:00:00 +08:00**

---

## 八、论文主张边界

### 当前可以写入论文或研究记录的事实

可以写：

1. 在冻结 H65 Scout、epoch-59 EMA detector 和训练侧 40-video 留出集合上，三档真实反事实效用重分配在相同实际 observation 总预算下获得：

   * Avg-mAP `+0.725589` 个百分点；
   * mAP@0.7 `+0.729004` 个百分点。
2. 该重分配只改变 22/124 个窗口，说明边际预算价值集中在少数窗口。
3. 实现保持 K384 selection/prediction 一致、detector/Scout frozen、targets detached，并满足精确实际成本。
4. 结果处于预注册灰区，没有触发 utility-head 训练。
5. producer 来源为 `f87555f7...`，汇总修复和最终结果绑定 `f67d96fd...`。
6. 当前 50% 改变比例约束在五个发生重分配的视频上达到上限，因此值得进行一次有界的约束释放心证。

### 当前不可写入的主张

不可声称：

* 动态预算已在 THUMOS14 官方 validation/test 上提高性能；
* `+0.726/+0.729` 具有统计显著性或稳定总体效应；
* utility head 可以预测真实边际效用；
* learned allocation 有效；
* 50% 改变上限已被证明是收益不足的根因；
* 三档预算已被证明过粗；
* 当前结果是所有动态预算方法的上界；
* 已获得端到端 FLOPs、延迟、吞吐量或能耗收益；
* DUCA-Marginal 已形成可投稿主方法；
* 训练侧 `88.xx%` 可以与官方 validation/test 指标直接比较。

### 当前贡献状态

**本轮尚未形成可投稿贡献。**

它形成的是一个实现严格、来源可解释、但收益不足且统计未闭合的机制诊断。其价值在于约束后续研究：当前 frozen-H65 三档边际重分配并非完全没有空间，但在原 50% 改变比例下尚不足以承担学习与部署阶段。只有唯一 cap-release 证伪通过后，才有理由重新讨论 predictor；否则该机制应作为受控负结果保存。

---

```text
next_owner: Codex Builder
next_action: 在 f67d96fdf68a295eaa7f678f3dfc125530828889 上实现并执行 max_changed_fraction=1.0 的只读 oracle 约束释放心证；原 0.5 路径必须精确复现；禁止 utility-head 训练和新 producer forward
dependency: 三个已密封 producer artifacts、对应 f87555f7 receipts、f67d96fd PRE_RUN/source bindings、原 160/40 split、冻结 evaluator/NMS/checkpoint/data
expected_return_at: 2026-09-01T12:00:00+08:00
```

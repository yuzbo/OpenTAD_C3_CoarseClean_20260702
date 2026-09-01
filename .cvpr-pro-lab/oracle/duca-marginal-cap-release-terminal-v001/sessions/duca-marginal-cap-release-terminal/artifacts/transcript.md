# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a94abb7-bd48-83e9-9516-c650c982dd57

## Prompt

Read every attached file completely and answer the authoritative prompt as one independent scientific adjudication. Treat the GitHub repository, branch, exact commit d2fad7c0dfc4a5efe98b10b9eee4723c6805699f, runner, allocator and test permalinks in the prompt as the latest code truth. Preserve nonce DUCA-MARGINAL-CAP-RELEASE-TERMINAL-ADJUDICATION-v001-20260831 verbatim in the response.

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_MARGINAL_CAP_RELEASE_TERMINAL_ADJUDICATION-v001.md
Lines: 1-77
```md
 1 | # DUCA Marginal-v1 cap-release 终态科学裁决
 2 | 
 3 | Nonce：`DUCA-MARGINAL-CAP-RELEASE-TERMINAL-ADJUDICATION-v001-20260831`
 4 | 
 5 | 你是本课题的独立科学负责人、路线设计者与最终审查者。Codex 只负责忠实执行你冻结的任务。本轮请基于下面完整证据，独立决定当前 DUCA 研究下一步；不要迎合 Codex，也不要把运行成功、代码审查或训练侧诊断扩大为论文结论。
 6 | 
 7 | ## 最新公开代码真值
 8 | 
 9 | - GitHub 仓库：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
10 | - 最新实现分支：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-falsifier-v1-20260831
11 | - 精确提交：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f
12 | - cap-release runner：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f/tools/bata/run_duca_marginal_frozen_h65_probe.py
13 | - 动态预算分配器：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f/opentad/models/duca/dynamic_budget.py
14 | - 聚焦测试：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f/tests/test_duca_marginal_budget.py
15 | 
16 | 以上 GitHub 链接是本轮最新实现真值。提交 `d2fad7c0...` 只增加独立的 `max_changed_fraction=1.0` 只读汇总入口与测试；默认 `0.5` 路径、K256/K384/K512 producer、模型、数据、NMS、评估器和门槛均未改变。分支已推送且干净；N16R4 14 项聚焦测试通过，独立 Critic 返回 PASS。
17 | 
18 | ## 冻结科学问题与前置裁决
19 | 
20 | 当前诊断冻结 H65 Scout、VideoMAE-S、Adapter、ActionFormer、损失、NMS 与评估器，通过同一 H65 priority sequence 构造嵌套的 K256/K384/K512 真实 observation 集合。在同一视频内严格保持总实际 observation 预算 `sum_i min(V_i,384)`，使用训练侧 40-video utility holdout 的真实反事实效用，检验跨窗口重新分配重型计算的 oracle headroom。
21 | 
22 | 此前 50% 改变窗口上限的结果是：
23 | 
24 | - Fixed-H65-384：Avg-mAP `88.131197%`，mAP@0.7 `76.270583%`；
25 | - capped oracle：Avg-mAP `88.856786%`，mAP@0.7 `76.999587%`；
26 | - 增益：`+0.725589/+0.729004` 个百分点；
27 | - 分配 K256/K384/K512=`11/102/11`，总实际 observation=`47110`，预算误差为零。
28 | 
29 | 该结果介于预注册强 headroom 门 `+0.8/+1.0` 和无 headroom 边界 `<+0.3/<+0.5` 之间。你上一轮裁决为 `REVISE`，唯一允许的后继是解除 50% 改变窗口上限，在相同密封产物上只读计算 `max_changed_fraction=1.0`。你同时冻结：若两项点门没有同时通过，停止当前 Marginal-v1 机制且不运行 bootstrap；只有两项都通过才执行 seed 3407、10,000 次整视频配对 bootstrap。
30 | 
31 | ## 唯一 cap-release 终态
32 | 
33 | 唯一 Evaluator Job `1262117` 于 `2026-08-31T05:53:33+08:00` 启动，`05:54:25+08:00` 以 `COMPLETED 0:0` 结束。它只在 CPU 上读取原密封产物；没有执行 detector/Scout forward、模型训练、utility-head 拟合或 official test。
34 | 
35 | 终态结果：
36 | 
37 | - 原 Fixed-H65-384 与 capped oracle 的所有点值复现误差均为 `0.0` 个百分点；
38 | - released oracle：Avg-mAP `88.558507%`，mAP@0.7 `76.720863%`；
39 | - released oracle 相对 Fixed-H65-384：`+0.427310/+0.450280` 个百分点；
40 | - released oracle 相对 capped oracle：`-0.298279/-0.278724` 个百分点；
41 | - released 分配 K256/K384/K512=`17/90/17`；改变 11 个视频、34 个窗口；
42 | - 总实际 observation=`47110`，预算误差为零；
43 | - 两项强门均失败，`strong_gate_pass=false`；
44 | - 按冻结规则 `paired_interval_required=false`，0 次 bootstrap；
45 | - runner 终态：`CAP_RELEASE_POINT_GATE_FAILED_STOP_CURRENT_MECHANISM`。
46 | 
47 | 原始终态 JSON（随本问询完整附带）：
48 | `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-d2fad7c0-job1262117/oracle_cap_release_result.json`
49 | 
50 | SHA-256：
51 | 
52 | - terminal result：`fb3c122e233952a4165c2ca9a6ff3d2839b8e0d108c977443786714ec0cf6ed4`
53 | - original probe：`8d6df7240c8b81b4d6d9aa8ff98bae530d6823ddd1d411bed47ce983ebd94925`
54 | - K384 producer：`1d668d4e5eb4b5ef3c1057c97ec63cc2c1eed3c0e62297290520063b4e1ec38f`
55 | - K256 producer：`6dc8893a41b5c8132b176f32133ffc2f48a5491146385c147b8227167608a309`
56 | - K512 producer：`c7fa06258c07163d0906b512a78e367c27607c64fc41b28bce9fe51fbd0815d7`
57 | 
58 | Producer 来源仍为 `f87555f7da362fe1a20d4ca08f7a68c975ed8280`；原 capped summary 来源为 `f67d96fdf68a295eaa7f678f3dfc125530828889`；cap-release runner 与终态来源为 `d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`。配置、checkpoint、annotation、类别映射和 VideoMAE 预训练身份不变。
59 | 
60 | ## 证据边界
61 | 
62 | 本结果是 40 个训练侧 holdout 视频上的真实效用 oracle 机制诊断，不是 learned allocator、official validation/test、统计显著性、端到端成本或论文主结果。预注册规则已经停止当前 Marginal-v1 机制，但这不能自动外推为所有动态预算、coverage 或物理时间方法无效。Codex 没有选择后继路线，也没有授权 predictor、official test、重训或新实验。
63 | 
64 | ## 你的唯一任务
65 | 
66 | 请独立完成以下工作：
67 | 
68 | 1. 先给出且只给出一个总裁决：`CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`。
69 | 2. 判断当前负结果最支持什么、仍不能支持什么；区分“当前三档 oracle 重分配机制缺乏足够 headroom”与更广泛的动态计算假设。
70 | 3. 解释为什么解除改变窗口上限后 oracle 反而变差。请判断这更可能反映组合分配约束、真实效用的非加性、预算档位过粗、窗口间干扰、评估聚合性质，还是当前问题本身缺少可利用空间；不要做没有证据的唯一因果断言。
71 | 4. 独立决定 DUCA 现在应停止整个方向、修订科学问题，还是转向一个新的可证伪机制。不要受 Codex 既有实现偏好影响。
72 | 5. 只下达一项当前任务。它必须直接检验你认为最关键的不确定性，并写清：科学问题、机制或分析对象、最小实现边界、对照、公平性、数据与 split、主要指标、最便宜 falsifier、继续/停止门、禁止项、Builder→独立 Critic→Evaluator 的职责，以及绝对完成时限。
73 | 6. 明确当前证据能否进入论文；若只能作为负结果或内部路线淘汰，也请给出准确表述。
74 | 
75 | 不要要求 Codex先替你选线，不要同时下达多条探索，不要把流程工程、额外合同、哈希系统或重复审计当成科研任务。优先选择能最快减少论文核心不确定性的真实分析或实验。
76 | 
77 | 回复必须包含本 nonce：`DUCA-MARGINAL-CAP-RELEASE-TERMINAL-ADJUDICATION-v001-20260831`。
```

### File: .cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-d2fad7c0-job1262117/oracle_cap_release_result.json
Lines: 1-3076
```json
   1 | {
   2 |   "cap_release_allocation": {
   3 |     "video_validation_0000055": {
   4 |       "actual_budget_error": 0,
   5 |       "actual_cost": [
   6 |         384,
   7 |         384,
   8 |         384
   9 |       ],
  10 |       "budgets": [
  11 |         384,
  12 |         384,
  13 |         384
  14 |       ],
  15 |       "collapsed_to_k384": [
  16 |         false,
  17 |         false,
  18 |         false
  19 |       ],
  20 |       "execution_slots": [
  21 |         384,
  22 |         384,
  23 |         384
  24 |       ],
  25 |       "padding_slots": [
  26 |         0,
  27 |         0,
  28 |         0
  29 |       ],
  30 |       "predicted_total_utility": 0.0,
  31 |       "requested_budgets": [
  32 |         384,
  33 |         384,
  34 |         384
  35 |       ],
  36 |       "target_actual_cost": 1152,
  37 |       "window_count": 3
  38 |     },
  39 |     "video_validation_0000059": {
  40 |       "actual_budget_error": 0,
  41 |       "actual_cost": [
  42 |         512,
  43 |         256,
  44 |         512,
  45 |         256
  46 |       ],
  47 |       "budgets": [
  48 |         512,
  49 |         256,
  50 |         512,
  51 |         256
  52 |       ],
  53 |       "collapsed_to_k384": [
  54 |         false,
  55 |         false,
  56 |         false,
  57 |         false
  58 |       ],
  59 |       "execution_slots": [
  60 |         512,
  61 |         256,
  62 |         512,
  63 |         256
  64 |       ],
  65 |       "padding_slots": [
  66 |         0,
  67 |         0,
  68 |         0,
  69 |         0
  70 |       ],
  71 |       "predicted_total_utility": 0.27181801199913025,
  72 |       "requested_budgets": [
  73 |         512,
  74 |         256,
  75 |         512,
  76 |         256
  77 |       ],
  78 |       "target_actual_cost": 1536,
  79 |       "window_count": 4
  80 |     },
  81 |     "video_validation_0000158": {
  82 |       "actual_budget_error": 0,
  83 |       "actual_cost": [
  84 |         512,
  85 |         256,
  86 |         384,
  87 |         384,
  88 |         384
  89 |       ],
  90 |       "budgets": [
  91 |         512,
  92 |         256,
  93 |         384,
  94 |         384,
  95 |         384
  96 |       ],
  97 |       "collapsed_to_k384": [
  98 |         false,
  99 |         false,
 100 |         false,
 101 |         false,
 102 |         false
 103 |       ],
 104 |       "execution_slots": [
 105 |         512,
 106 |         256,
 107 |         384,
 108 |         384,
 109 |         384
 110 |       ],
 111 |       "padding_slots": [
 112 |         0,
 113 |         0,
 114 |         0,
 115 |         0,
 116 |         0
 117 |       ],
 118 |       "predicted_total_utility": 0.06998297572135925,
 119 |       "requested_budgets": [
 120 |         512,
 121 |         256,
 122 |         384,
 123 |         384,
 124 |         384
 125 |       ],
 126 |       "target_actual_cost": 1920,
 127 |       "window_count": 5
 128 |     },
 129 |     "video_validation_0000164": {
 130 |       "actual_budget_error": 0,
 131 |       "actual_cost": [
 132 |         384,
 133 |         384,
 134 |         384
 135 |       ],
 136 |       "budgets": [
 137 |         384,
 138 |         384,
 139 |         384
 140 |       ],
 141 |       "collapsed_to_k384": [
 142 |         false,
 143 |         false,
 144 |         false
 145 |       ],
 146 |       "execution_slots": [
 147 |         384,
 148 |         384,
 149 |         384
 150 |       ],
 151 |       "padding_slots": [
 152 |         0,
 153 |         0,
 154 |         0
 155 |       ],
 156 |       "predicted_total_utility": 0.0,
 157 |       "requested_budgets": [
 158 |         384,
 159 |         384,
 160 |         384
 161 |       ],
 162 |       "target_actual_cost": 1152,
 163 |       "window_count": 3
 164 |     },
 165 |     "video_validation_0000172": {
 166 |       "actual_budget_error": 0,
 167 |       "actual_cost": [
 168 |         384,
 169 |         384
 170 |       ],
 171 |       "budgets": [
 172 |         384,
 173 |         384
 174 |       ],
 175 |       "collapsed_to_k384": [
 176 |         false,
 177 |         false
 178 |       ],
 179 |       "execution_slots": [
 180 |         384,
 181 |         384
 182 |       ],
 183 |       "padding_slots": [
 184 |         0,
 185 |         0
 186 |       ],
 187 |       "predicted_total_utility": 0.0,
 188 |       "requested_budgets": [
 189 |         384,
 190 |         384
 191 |       ],
 192 |       "target_actual_cost": 768,
 193 |       "window_count": 2
 194 |     },
 195 |     "video_validation_0000173": {
 196 |       "actual_budget_error": 0,
 197 |       "actual_cost": [
 198 |         384,
 199 |         384,
 200 |         256,
 201 |         512
 202 |       ],
 203 |       "budgets": [
 204 |         384,
 205 |         384,
 206 |         256,
 207 |         512
 208 |       ],
 209 |       "collapsed_to_k384": [
 210 |         false,
 211 |         false,
 212 |         false,
 213 |         false
 214 |       ],
 215 |       "execution_slots": [
 216 |         384,
 217 |         384,
 218 |         256,
 219 |         512
 220 |       ],
 221 |       "padding_slots": [
 222 |         0,
 223 |         0,
 224 |         0,
 225 |         0
 226 |       ],
 227 |       "predicted_total_utility": 0.1362585425376892,
 228 |       "requested_budgets": [
 229 |         384,
 230 |         384,
 231 |         256,
 232 |         512
 233 |       ],
 234 |       "target_actual_cost": 1536,
 235 |       "window_count": 4
 236 |     },
 237 |     "video_validation_0000181": {
 238 |       "actual_budget_error": 0,
 239 |       "actual_cost": [
 240 |         384,
 241 |         384
 242 |       ],
 243 |       "budgets": [
 244 |         384,
 245 |         384
 246 |       ],
 247 |       "collapsed_to_k384": [
 248 |         false,
 249 |         false
 250 |       ],
 251 |       "execution_slots": [
 252 |         384,
 253 |         384
 254 |       ],
 255 |       "padding_slots": [
 256 |         0,
 257 |         0
 258 |       ],
 259 |       "predicted_total_utility": 0.0,
 260 |       "requested_budgets": [
 261 |         384,
 262 |         384
 263 |       ],
 264 |       "target_actual_cost": 768,
 265 |       "window_count": 2
 266 |     },
 267 |     "video_validation_0000186": {
 268 |       "actual_budget_error": 0,
 269 |       "actual_cost": [
 270 |         384
 271 |       ],
 272 |       "budgets": [
 273 |         384
 274 |       ],
 275 |       "collapsed_to_k384": [
 276 |         false
 277 |       ],
 278 |       "execution_slots": [
 279 |         384
 280 |       ],
 281 |       "padding_slots": [
 282 |         0
 283 |       ],
 284 |       "predicted_total_utility": 0.0,
 285 |       "requested_budgets": [
 286 |         384
 287 |       ],
 288 |       "target_actual_cost": 384,
 289 |       "window_count": 1
 290 |     },
 291 |     "video_validation_0000206": {
 292 |       "actual_budget_error": 0,
 293 |       "actual_cost": [
 294 |         512,
 295 |         384,
 296 |         256
 297 |       ],
 298 |       "budgets": [
 299 |         512,
 300 |         384,
 301 |         256
 302 |       ],
 303 |       "collapsed_to_k384": [
 304 |         false,
 305 |         false,
 306 |         false
 307 |       ],
 308 |       "execution_slots": [
 309 |         512,
 310 |         384,
 311 |         256
 312 |       ],
 313 |       "padding_slots": [
 314 |         0,
 315 |         0,
 316 |         0
 317 |       ],
 318 |       "predicted_total_utility": 0.08902537822723389,
 319 |       "requested_budgets": [
 320 |         512,
 321 |         384,
 322 |         256
 323 |       ],
 324 |       "target_actual_cost": 1152,
 325 |       "window_count": 3
 326 |     },
 327 |     "video_validation_0000207": {
 328 |       "actual_budget_error": 0,
 329 |       "actual_cost": [
 330 |         384,
 331 |         384,
 332 |         384
 333 |       ],
 334 |       "budgets": [
 335 |         384,
 336 |         384,
 337 |         384
 338 |       ],
 339 |       "collapsed_to_k384": [
 340 |         false,
 341 |         false,
 342 |         false
 343 |       ],
 344 |       "execution_slots": [
 345 |         384,
 346 |         384,
 347 |         384
 348 |       ],
 349 |       "padding_slots": [
 350 |         0,
 351 |         0,
 352 |         0
 353 |       ],
 354 |       "predicted_total_utility": 0.0,
 355 |       "requested_budgets": [
 356 |         384,
 357 |         384,
 358 |         384
 359 |       ],
 360 |       "target_actual_cost": 1152,
 361 |       "window_count": 3
 362 |     },
 363 |     "video_validation_0000266": {
 364 |       "actual_budget_error": 0,
 365 |       "actual_cost": [
 366 |         384,
 367 |         384,
 368 |         384
 369 |       ],
 370 |       "budgets": [
 371 |         384,
 372 |         384,
 373 |         384
 374 |       ],
 375 |       "collapsed_to_k384": [
 376 |         false,
 377 |         false,
 378 |         false
 379 |       ],
 380 |       "execution_slots": [
 381 |         384,
 382 |         384,
 383 |         384
 384 |       ],
 385 |       "padding_slots": [
 386 |         0,
 387 |         0,
 388 |         0
 389 |       ],
 390 |       "predicted_total_utility": 0.0,
 391 |       "requested_budgets": [
 392 |         384,
 393 |         384,
 394 |         384
 395 |       ],
 396 |       "target_actual_cost": 1152,
 397 |       "window_count": 3
 398 |     },
 399 |     "video_validation_0000267": {
 400 |       "actual_budget_error": 0,
 401 |       "actual_cost": [
 402 |         512,
 403 |         384,
 404 |         256,
 405 |         384,
 406 |         512,
 407 |         256,
 408 |         384
 409 |       ],
 410 |       "budgets": [
 411 |         512,
 412 |         384,
 413 |         256,
 414 |         384,
 415 |         512,
 416 |         256,
 417 |         384
 418 |       ],
 419 |       "collapsed_to_k384": [
 420 |         false,
 421 |         false,
 422 |         false,
 423 |         false,
 424 |         false,
 425 |         false,
 426 |         false
 427 |       ],
 428 |       "execution_slots": [
 429 |         512,
 430 |         384,
 431 |         256,
 432 |         384,
 433 |         512,
 434 |         256,
 435 |         384
 436 |       ],
 437 |       "padding_slots": [
 438 |         0,
 439 |         0,
 440 |         0,
 441 |         0,
 442 |         0,
 443 |         0,
 444 |         0
 445 |       ],
 446 |       "predicted_total_utility": 0.11238357424736023,
 447 |       "requested_budgets": [
 448 |         512,
 449 |         384,
 450 |         256,
 451 |         384,
 452 |         512,
 453 |         256,
 454 |         384
 455 |       ],
 456 |       "target_actual_cost": 2688,
 457 |       "window_count": 7
 458 |     },
 459 |     "video_validation_0000282": {
 460 |       "actual_budget_error": 0,
 461 |       "actual_cost": [
 462 |         288
 463 |       ],
 464 |       "budgets": [
 465 |         384
 466 |       ],
 467 |       "collapsed_to_k384": [
 468 |         false
 469 |       ],
 470 |       "execution_slots": [
 471 |         384
 472 |       ],
 473 |       "padding_slots": [
 474 |         96
 475 |       ],
 476 |       "predicted_total_utility": 0.0,
 477 |       "requested_budgets": [
 478 |         384
 479 |       ],
 480 |       "target_actual_cost": 288,
 481 |       "window_count": 1
 482 |     },
 483 |     "video_validation_0000283": {
 484 |       "actual_budget_error": 0,
 485 |       "actual_cost": [
 486 |         384
 487 |       ],
 488 |       "budgets": [
 489 |         384
 490 |       ],
 491 |       "collapsed_to_k384": [
 492 |         false
 493 |       ],
 494 |       "execution_slots": [
 495 |         384
 496 |       ],
 497 |       "padding_slots": [
 498 |         0
 499 |       ],
 500 |       "predicted_total_utility": 0.0,
 501 |       "requested_budgets": [
 502 |         384
 503 |       ],
 504 |       "target_actual_cost": 384,
 505 |       "window_count": 1
 506 |     },
 507 |     "video_validation_0000285": {
 508 |       "actual_budget_error": 0,
 509 |       "actual_cost": [
 510 |         384,
 511 |         384
 512 |       ],
 513 |       "budgets": [
 514 |         384,
 515 |         384
 516 |       ],
 517 |       "collapsed_to_k384": [
 518 |         false,
 519 |         false
 520 |       ],
 521 |       "execution_slots": [
 522 |         384,
 523 |         384
 524 |       ],
 525 |       "padding_slots": [
 526 |         0,
 527 |         0
 528 |       ],
 529 |       "predicted_total_utility": 0.0,
 530 |       "requested_budgets": [
 531 |         384,
 532 |         384
 533 |       ],
 534 |       "target_actual_cost": 768,
 535 |       "window_count": 2
 536 |     },
 537 |     "video_validation_0000367": {
 538 |       "actual_budget_error": 0,
 539 |       "actual_cost": [
 540 |         384,
 541 |         384
 542 |       ],
 543 |       "budgets": [
 544 |         384,
 545 |         384
 546 |       ],
 547 |       "collapsed_to_k384": [
 548 |         false,
 549 |         false
 550 |       ],
 551 |       "execution_slots": [
 552 |         384,
 553 |         384
 554 |       ],
 555 |       "padding_slots": [
 556 |         0,
 557 |         0
 558 |       ],
 559 |       "predicted_total_utility": 0.0,
 560 |       "requested_budgets": [
 561 |         384,
 562 |         384
 563 |       ],
 564 |       "target_actual_cost": 768,
 565 |       "window_count": 2
 566 |     },
 567 |     "video_validation_0000417": {
 568 |       "actual_budget_error": 0,
 569 |       "actual_cost": [
 570 |         256,
 571 |         512,
 572 |         384,
 573 |         384,
 574 |         384,
 575 |         384
 576 |       ],
 577 |       "budgets": [
 578 |         256,
 579 |         512,
 580 |         384,
 581 |         384,
 582 |         384,
 583 |         384
 584 |       ],
 585 |       "collapsed_to_k384": [
 586 |         false,
 587 |         false,
 588 |         false,
 589 |         false,
 590 |         false,
 591 |         false
 592 |       ],
 593 |       "execution_slots": [
 594 |         256,
 595 |         512,
 596 |         384,
 597 |         384,
 598 |         384,
 599 |         384
 600 |       ],
 601 |       "padding_slots": [
 602 |         0,
 603 |         0,
 604 |         0,
 605 |         0,
 606 |         0,
 607 |         0
 608 |       ],
 609 |       "predicted_total_utility": 0.019536063075065613,
 610 |       "requested_budgets": [
 611 |         256,
 612 |         512,
 613 |         384,
 614 |         384,
 615 |         384,
 616 |         384
 617 |       ],
 618 |       "target_actual_cost": 2304,
 619 |       "window_count": 6
 620 |     },
 621 |     "video_validation_0000419": {
 622 |       "actual_budget_error": 0,
 623 |       "actual_cost": [
 624 |         384,
 625 |         256,
 626 |         512,
 627 |         256,
 628 |         384,
 629 |         384,
 630 |         512,
 631 |         256,
 632 |         384,
 633 |         512,
 634 |         256,
 635 |         256,
 636 |         512,
 637 |         512
 638 |       ],
 639 |       "budgets": [
 640 |         384,
 641 |         256,
 642 |         512,
 643 |         256,
 644 |         384,
 645 |         384,
 646 |         512,
 647 |         256,
 648 |         384,
 649 |         512,
 650 |         256,
 651 |         256,
 652 |         512,
 653 |         512
 654 |       ],
 655 |       "collapsed_to_k384": [
 656 |         false,
 657 |         false,
 658 |         false,
 659 |         false,
 660 |         false,
 661 |         false,
 662 |         false,
 663 |         false,
 664 |         false,
 665 |         false,
 666 |         false,
 667 |         false,
 668 |         false,
 669 |         false
 670 |       ],
 671 |       "execution_slots": [
 672 |         384,
 673 |         256,
 674 |         512,
 675 |         256,
 676 |         384,
 677 |         384,
 678 |         512,
 679 |         256,
 680 |         384,
 681 |         512,
 682 |         256,
 683 |         256,
 684 |         512,
 685 |         512
 686 |       ],
 687 |       "padding_slots": [
 688 |         0,
 689 |         0,
 690 |         0,
 691 |         0,
 692 |         0,
 693 |         0,
 694 |         0,
 695 |         0,
 696 |         0,
 697 |         0,
 698 |         0,
 699 |         0,
 700 |         0,
 701 |         0
 702 |       ],
 703 |       "predicted_total_utility": 0.11090625077486038,
 704 |       "requested_budgets": [
 705 |         384,
 706 |         256,
 707 |         512,
 708 |         256,
 709 |         384,
 710 |         384,
 711 |         512,
 712 |         256,
 713 |         384,
 714 |         512,
 715 |         256,
 716 |         256,
 717 |         512,
 718 |         512
 719 |       ],
 720 |       "target_actual_cost": 5376,
 721 |       "window_count": 14
 722 |     },
 723 |     "video_validation_0000483": {
 724 |       "actual_budget_error": 0,
 725 |       "actual_cost": [
 726 |         384,
 727 |         384
 728 |       ],
 729 |       "budgets": [
 730 |         384,
 731 |         384
 732 |       ],
 733 |       "collapsed_to_k384": [
 734 |         false,
 735 |         false
 736 |       ],
 737 |       "execution_slots": [
 738 |         384,
 739 |         384
 740 |       ],
 741 |       "padding_slots": [
 742 |         0,
 743 |         0
 744 |       ],
 745 |       "predicted_total_utility": 0.0,
 746 |       "requested_budgets": [
 747 |         384,
 748 |         384
 749 |       ],
 750 |       "target_actual_cost": 768,
 751 |       "window_count": 2
 752 |     },
 753 |     "video_validation_0000489": {
 754 |       "actual_budget_error": 0,
 755 |       "actual_cost": [
 756 |         384,
 757 |         384,
 758 |         384
 759 |       ],
 760 |       "budgets": [
 761 |         384,
 762 |         384,
 763 |         384
 764 |       ],
 765 |       "collapsed_to_k384": [
 766 |         false,
 767 |         false,
 768 |         false
 769 |       ],
 770 |       "execution_slots": [
 771 |         384,
 772 |         384,
 773 |         384
 774 |       ],
 775 |       "padding_slots": [
 776 |         0,
 777 |         0,
 778 |         0
 779 |       ],
 780 |       "predicted_total_utility": 0.0,
 781 |       "requested_budgets": [
 782 |         384,
 783 |         384,
 784 |         384
 785 |       ],
 786 |       "target_actual_cost": 1152,
 787 |       "window_count": 3
 788 |     },
 789 |     "video_validation_0000490": {
 790 |       "actual_budget_error": 0,
 791 |       "actual_cost": [
 792 |         384,
 793 |         384,
 794 |         512,
 795 |         256,
 796 |         384,
 797 |         384
 798 |       ],
 799 |       "budgets": [
 800 |         384,
 801 |         384,
 802 |         512,
 803 |         256,
 804 |         384,
 805 |         384
 806 |       ],
 807 |       "collapsed_to_k384": [
 808 |         false,
 809 |         false,
 810 |         false,
 811 |         false,
 812 |         false,
 813 |         false
 814 |       ],
 815 |       "execution_slots": [
 816 |         384,
 817 |         384,
 818 |         512,
 819 |         256,
 820 |         384,
 821 |         384
 822 |       ],
 823 |       "padding_slots": [
 824 |         0,
 825 |         0,
 826 |         0,
 827 |         0,
 828 |         0,
 829 |         0
 830 |       ],
 831 |       "predicted_total_utility": 0.13576892018318176,
 832 |       "requested_budgets": [
 833 |         384,
 834 |         384,
 835 |         512,
 836 |         256,
 837 |         384,
 838 |         384
 839 |       ],
 840 |       "target_actual_cost": 2304,
 841 |       "window_count": 6
 842 |     },
 843 |     "video_validation_0000664": {
 844 |       "actual_budget_error": 0,
 845 |       "actual_cost": [
 846 |         256,
 847 |         384,
 848 |         384,
 849 |         512
 850 |       ],
 851 |       "budgets": [
 852 |         256,
 853 |         384,
 854 |         384,
 855 |         512
 856 |       ],
 857 |       "collapsed_to_k384": [
 858 |         false,
 859 |         false,
 860 |         false,
 861 |         false
 862 |       ],
 863 |       "execution_slots": [
 864 |         256,
 865 |         384,
 866 |         384,
 867 |         512
 868 |       ],
 869 |       "padding_slots": [
 870 |         0,
 871 |         0,
 872 |         0,
 873 |         0
 874 |       ],
 875 |       "predicted_total_utility": 0.009859195910394192,
 876 |       "requested_budgets": [
 877 |         256,
 878 |         384,
 879 |         384,
 880 |         512
 881 |       ],
 882 |       "target_actual_cost": 1536,
 883 |       "window_count": 4
 884 |     },
 885 |     "video_validation_0000681": {
 886 |       "actual_budget_error": 0,
 887 |       "actual_cost": [
 888 |         384
 889 |       ],
 890 |       "budgets": [
 891 |         384
 892 |       ],
 893 |       "collapsed_to_k384": [
 894 |         false
 895 |       ],
 896 |       "execution_slots": [
 897 |         384
 898 |       ],
 899 |       "padding_slots": [
 900 |         0
 901 |       ],
 902 |       "predicted_total_utility": 0.0,
 903 |       "requested_budgets": [
 904 |         384
 905 |       ],
 906 |       "target_actual_cost": 384,
 907 |       "window_count": 1
 908 |     },
 909 |     "video_validation_0000683": {
 910 |       "actual_budget_error": 0,
 911 |       "actual_cost": [
 912 |         285
 913 |       ],
 914 |       "budgets": [
 915 |         384
 916 |       ],
 917 |       "collapsed_to_k384": [
 918 |         false
 919 |       ],
 920 |       "execution_slots": [
 921 |         384
 922 |       ],
 923 |       "padding_slots": [
 924 |         99
 925 |       ],
 926 |       "predicted_total_utility": 0.0,
 927 |       "requested_budgets": [
 928 |         384
 929 |       ],
 930 |       "target_actual_cost": 285,
 931 |       "window_count": 1
 932 |     },
 933 |     "video_validation_0000690": {
 934 |       "actual_budget_error": 0,
 935 |       "actual_cost": [
 936 |         384,
 937 |         256,
 938 |         384,
 939 |         512,
 940 |         384
 941 |       ],
 942 |       "budgets": [
 943 |         384,
 944 |         256,
 945 |         384,
 946 |         512,
 947 |         384
 948 |       ],
 949 |       "collapsed_to_k384": [
 950 |         false,
 951 |         false,
 952 |         false,
 953 |         false,
 954 |         false
 955 |       ],
 956 |       "execution_slots": [
 957 |         384,
 958 |         256,
 959 |         384,
 960 |         512,
 961 |         384
 962 |       ],
 963 |       "padding_slots": [
 964 |         0,
 965 |         0,
 966 |         0,
 967 |         0,
 968 |         0
 969 |       ],
 970 |       "predicted_total_utility": 0.004800617229193449,
 971 |       "requested_budgets": [
 972 |         384,
 973 |         256,
 974 |         384,
 975 |         512,
 976 |         384
 977 |       ],
 978 |       "target_actual_cost": 1920,
 979 |       "window_count": 5
 980 |     },
 981 |     "video_validation_0000783": {
 982 |       "actual_budget_error": 0,
 983 |       "actual_cost": [
 984 |         384,
 985 |         384,
 986 |         384
 987 |       ],
 988 |       "budgets": [
 989 |         384,
 990 |         384,
 991 |         384
 992 |       ],
 993 |       "collapsed_to_k384": [
 994 |         false,
 995 |         false,
 996 |         false
 997 |       ],
 998 |       "execution_slots": [
 999 |         384,
1000 |         384,
1001 |         384
1002 |       ],
1003 |       "padding_slots": [
1004 |         0,
1005 |         0,
1006 |         0
1007 |       ],
1008 |       "predicted_total_utility": 0.0,
1009 |       "requested_budgets": [
1010 |         384,
1011 |         384,
1012 |         384
1013 |       ],
1014 |       "target_actual_cost": 1152,
1015 |       "window_count": 3
1016 |     },
1017 |     "video_validation_0000851": {
1018 |       "actual_budget_error": 0,
1019 |       "actual_cost": [
1020 |         233
1021 |       ],
1022 |       "budgets": [
1023 |         384
1024 |       ],
1025 |       "collapsed_to_k384": [
1026 |         false
1027 |       ],
1028 |       "execution_slots": [
1029 |         384
1030 |       ],
1031 |       "padding_slots": [
1032 |         151
1033 |       ],
1034 |       "predicted_total_utility": 0.0,
1035 |       "requested_budgets": [
1036 |         384
1037 |       ],
1038 |       "target_actual_cost": 233,
1039 |       "window_count": 1
1040 |     },
1041 |     "video_validation_0000852": {
1042 |       "actual_budget_error": 0,
1043 |       "actual_cost": [
1044 |         384,
1045 |         384,
1046 |         384,
1047 |         384
1048 |       ],
1049 |       "budgets": [
1050 |         384,
1051 |         384,
1052 |         384,
1053 |         384
1054 |       ],
1055 |       "collapsed_to_k384": [
1056 |         false,
1057 |         false,
1058 |         false,
1059 |         false
1060 |       ],
1061 |       "execution_slots": [
1062 |         384,
1063 |         384,
1064 |         384,
1065 |         384
1066 |       ],
1067 |       "padding_slots": [
1068 |         0,
1069 |         0,
1070 |         0,
1071 |         0
1072 |       ],
1073 |       "predicted_total_utility": 0.0,
1074 |       "requested_budgets": [
1075 |         384,
1076 |         384,
1077 |         384,
1078 |         384
1079 |       ],
1080 |       "target_actual_cost": 1536,
1081 |       "window_count": 4
1082 |     },
1083 |     "video_validation_0000858": {
1084 |       "actual_budget_error": 0,
1085 |       "actual_cost": [
1086 |         384,
1087 |         384
1088 |       ],
1089 |       "budgets": [
1090 |         384,
1091 |         384
1092 |       ],
1093 |       "collapsed_to_k384": [
1094 |         false,
1095 |         false
1096 |       ],
1097 |       "execution_slots": [
1098 |         384,
1099 |         384
1100 |       ],
1101 |       "padding_slots": [
1102 |         0,
1103 |         0
1104 |       ],
1105 |       "predicted_total_utility": 0.0,
1106 |       "requested_budgets": [
1107 |         384,
1108 |         384
1109 |       ],
1110 |       "target_actual_cost": 768,
1111 |       "window_count": 2
1112 |     },
1113 |     "video_validation_0000905": {
1114 |       "actual_budget_error": 0,
1115 |       "actual_cost": [
1116 |         384,
1117 |         256,
1118 |         512
1119 |       ],
1120 |       "budgets": [
1121 |         384,
1122 |         256,
1123 |         512
1124 |       ],
1125 |       "collapsed_to_k384": [
1126 |         false,
1127 |         false,
1128 |         false
1129 |       ],
1130 |       "execution_slots": [
1131 |         384,
1132 |         256,
1133 |         512
1134 |       ],
1135 |       "padding_slots": [
1136 |         0,
1137 |         0,
1138 |         0
1139 |       ],
1140 |       "predicted_total_utility": 0.025414079427719116,
1141 |       "requested_budgets": [
1142 |         384,
1143 |         256,
1144 |         512
1145 |       ],
1146 |       "target_actual_cost": 1152,
1147 |       "window_count": 3
1148 |     },
1149 |     "video_validation_0000908": {
1150 |       "actual_budget_error": 0,
1151 |       "actual_cost": [
1152 |         384,
1153 |         384,
1154 |         384,
1155 |         384,
1156 |         384
1157 |       ],
1158 |       "budgets": [
1159 |         384,
1160 |         384,
1161 |         384,
1162 |         384,
1163 |         384
1164 |       ],
1165 |       "collapsed_to_k384": [
1166 |         false,
1167 |         false,
1168 |         false,
1169 |         false,
1170 |         false
1171 |       ],
1172 |       "execution_slots": [
1173 |         384,
1174 |         384,
1175 |         384,
1176 |         384,
1177 |         384
1178 |       ],
1179 |       "padding_slots": [
1180 |         0,
1181 |         0,
1182 |         0,
1183 |         0,
1184 |         0
1185 |       ],
1186 |       "predicted_total_utility": 0.0,
1187 |       "requested_budgets": [
1188 |         384,
1189 |         384,
1190 |         384,
1191 |         384,
1192 |         384
1193 |       ],
1194 |       "target_actual_cost": 1920,
1195 |       "window_count": 5
1196 |     },
1197 |     "video_validation_0000909": {
1198 |       "actual_budget_error": 0,
1199 |       "actual_cost": [
1200 |         384,
1201 |         384,
1202 |         384
1203 |       ],
1204 |       "budgets": [
1205 |         384,
1206 |         384,
1207 |         384
1208 |       ],
1209 |       "collapsed_to_k384": [
1210 |         false,
1211 |         false,
1212 |         false
1213 |       ],
1214 |       "execution_slots": [
1215 |         384,
1216 |         384,
1217 |         384
1218 |       ],
1219 |       "padding_slots": [
1220 |         0,
1221 |         0,
1222 |         0
1223 |       ],
1224 |       "predicted_total_utility": 0.0,
1225 |       "requested_budgets": [
1226 |         384,
1227 |         384,
1228 |         384
1229 |       ],
1230 |       "target_actual_cost": 1152,
1231 |       "window_count": 3
1232 |     },
1233 |     "video_validation_0000937": {
1234 |       "actual_budget_error": 0,
1235 |       "actual_cost": [
1236 |         384
1237 |       ],
1238 |       "budgets": [
1239 |         384
1240 |       ],
1241 |       "collapsed_to_k384": [
1242 |         false
1243 |       ],
1244 |       "execution_slots": [
1245 |         384
1246 |       ],
1247 |       "padding_slots": [
1248 |         0
1249 |       ],
1250 |       "predicted_total_utility": 0.0,
1251 |       "requested_budgets": [
1252 |         384
1253 |       ],
1254 |       "target_actual_cost": 384,
1255 |       "window_count": 1
1256 |     },
1257 |     "video_validation_0000938": {
1258 |       "actual_budget_error": 0,
1259 |       "actual_cost": [
1260 |         260
1261 |       ],
1262 |       "budgets": [
1263 |         384
1264 |       ],
1265 |       "collapsed_to_k384": [
1266 |         false
1267 |       ],
1268 |       "execution_slots": [
1269 |         384
1270 |       ],
1271 |       "padding_slots": [
1272 |         124
1273 |       ],
1274 |       "predicted_total_utility": 0.0,
1275 |       "requested_budgets": [
1276 |         384
1277 |       ],
1278 |       "target_actual_cost": 260,
1279 |       "window_count": 1
1280 |     },
1281 |     "video_validation_0000943": {
1282 |       "actual_budget_error": 0,
1283 |       "actual_cost": [
1284 |         384,
1285 |         384
1286 |       ],
1287 |       "budgets": [
1288 |         384,
1289 |         384
1290 |       ],
1291 |       "collapsed_to_k384": [
1292 |         false,
1293 |         false
1294 |       ],
1295 |       "execution_slots": [
1296 |         384,
1297 |         384
1298 |       ],
1299 |       "padding_slots": [
1300 |         0,
1301 |         0
1302 |       ],
1303 |       "predicted_total_utility": 0.0,
1304 |       "requested_budgets": [
1305 |         384,
1306 |         384
1307 |       ],
1308 |       "target_actual_cost": 768,
1309 |       "window_count": 2
1310 |     },
1311 |     "video_validation_0000944": {
1312 |       "actual_budget_error": 0,
1313 |       "actual_cost": [
1314 |         384,
1315 |         384,
1316 |         384
1317 |       ],
1318 |       "budgets": [
1319 |         384,
1320 |         384,
1321 |         384
1322 |       ],
1323 |       "collapsed_to_k384": [
1324 |         false,
1325 |         false,
1326 |         false
1327 |       ],
1328 |       "execution_slots": [
1329 |         384,
1330 |         384,
1331 |         384
1332 |       ],
1333 |       "padding_slots": [
1334 |         0,
1335 |         0,
1336 |         0
1337 |       ],
1338 |       "predicted_total_utility": 0.0,
1339 |       "requested_budgets": [
1340 |         384,
1341 |         384,
1342 |         384
1343 |       ],
1344 |       "target_actual_cost": 1152,
1345 |       "window_count": 3
1346 |     },
1347 |     "video_validation_0000945": {
1348 |       "actual_budget_error": 0,
1349 |       "actual_cost": [
1350 |         384,
1351 |         384,
1352 |         384
1353 |       ],
1354 |       "budgets": [
1355 |         384,
1356 |         384,
1357 |         384
1358 |       ],
1359 |       "collapsed_to_k384": [
1360 |         false,
1361 |         false,
1362 |         false
1363 |       ],
1364 |       "execution_slots": [
1365 |         384,
1366 |         384,
1367 |         384
1368 |       ],
1369 |       "padding_slots": [
1370 |         0,
1371 |         0,
1372 |         0
1373 |       ],
1374 |       "predicted_total_utility": 0.0,
1375 |       "requested_budgets": [
1376 |         384,
1377 |         384,
1378 |         384
1379 |       ],
1380 |       "target_actual_cost": 1152,
1381 |       "window_count": 3
1382 |     },
1383 |     "video_validation_0000984": {
1384 |       "actual_budget_error": 0,
1385 |       "actual_cost": [
1386 |         348
1387 |       ],
1388 |       "budgets": [
1389 |         384
1390 |       ],
1391 |       "collapsed_to_k384": [
1392 |         false
1393 |       ],
1394 |       "execution_slots": [
1395 |         384
1396 |       ],
1397 |       "padding_slots": [
1398 |         36
1399 |       ],
1400 |       "predicted_total_utility": 0.0,
1401 |       "requested_budgets": [
1402 |         384
1403 |       ],
1404 |       "target_actual_cost": 348,
1405 |       "window_count": 1
1406 |     },
1407 |     "video_validation_0000988": {
1408 |       "actual_budget_error": 0,
1409 |       "actual_cost": [
1410 |         384,
1411 |         384
1412 |       ],
1413 |       "budgets": [
1414 |         384,
1415 |         384
1416 |       ],
1417 |       "collapsed_to_k384": [
1418 |         false,
1419 |         false
1420 |       ],
1421 |       "execution_slots": [
1422 |         384,
1423 |         384
1424 |       ],
1425 |       "padding_slots": [
1426 |         0,
1427 |         0
1428 |       ],
1429 |       "predicted_total_utility": 0.0,
1430 |       "requested_budgets": [
1431 |         384,
1432 |         384
1433 |       ],
1434 |       "target_actual_cost": 768,
1435 |       "window_count": 2
1436 |     },
1437 |     "video_validation_0000990": {
1438 |       "actual_budget_error": 0,
1439 |       "actual_cost": [
1440 |         384,
1441 |         384
1442 |       ],
1443 |       "budgets": [
1444 |         384,
1445 |         384
1446 |       ],
1447 |       "collapsed_to_k384": [
1448 |         false,
1449 |         false
1450 |       ],
1451 |       "execution_slots": [
1452 |         384,
1453 |         384
1454 |       ],
1455 |       "padding_slots": [
1456 |         0,
1457 |         0
1458 |       ],
1459 |       "predicted_total_utility": 0.0,
1460 |       "requested_budgets": [
1461 |         384,
1462 |         384
1463 |       ],
1464 |       "target_actual_cost": 768,
1465 |       "window_count": 2
1466 |     }
1467 |   },
1468 |   "cap_release_allocation_summary": {
1469 |     "actual_budget_error": 0,
1470 |     "actual_observation_cost": 47110,
1471 |     "budget_counts": {
1472 |       "256": 17,
1473 |       "384": 90,
1474 |       "512": 17
1475 |     },
1476 |     "capped_allocation_limit_hit_video_count": 4,
1477 |     "changed_video_count": 11,
1478 |     "changed_window_count": 34,
1479 |     "target_observation_cost": 47110
1480 |   },
1481 |   "cap_release_headroom": {
1482 |     "delta_average_mAP_pp": 0.42731030121248015,
1483 |     "delta_mAP_at_0.7_pp": 0.45027996133246706,
1484 |     "delta_over_capped_average_mAP_pp": -0.2982790114754108,
1485 |     "delta_over_capped_mAP_at_0.7_pp": -0.2787237401565834,
1486 |     "paired_interval_pass": null,
1487 |     "paired_interval_required": false,
1488 |     "strong_gate_pass": false
1489 |   },
1490 |   "cap_release_oracle_384": {
1491 |     "average_mAP": 0.8855850735788119,
1492 |     "mAP@0.3": 0.9547129241523014,
1493 |     "mAP@0.4": 0.9371032082987869,
1494 |     "mAP@0.5": 0.908550967086231,
1495 |     "mAP@0.6": 0.860349637139132,
1496 |     "mAP@0.7": 0.7672086312176079
1497 |   },
1498 |   "capped_allocation": {
1499 |     "video_validation_0000055": {
1500 |       "actual_budget_error": 0,
1501 |       "actual_cost": [
1502 |         384,
1503 |         384,
1504 |         384
1505 |       ],
1506 |       "budgets": [
1507 |         384,
1508 |         384,
1509 |         384
1510 |       ],
1511 |       "collapsed_to_k384": [
1512 |         false,
1513 |         false,
1514 |         false
1515 |       ],
1516 |       "execution_slots": [
1517 |         384,
1518 |         384,
1519 |         384
1520 |       ],
1521 |       "padding_slots": [
1522 |         0,
1523 |         0,
1524 |         0
1525 |       ],
1526 |       "predicted_total_utility": 0.0,
1527 |       "requested_budgets": [
1528 |         384,
1529 |         384,
1530 |         384
1531 |       ],
1532 |       "target_actual_cost": 1152,
1533 |       "window_count": 3
1534 |     },
1535 |     "video_validation_0000059": {
1536 |       "actual_budget_error": 0,
1537 |       "actual_cost": [
1538 |         384,
1539 |         256,
1540 |         512,
1541 |         384
1542 |       ],
1543 |       "budgets": [
1544 |         384,
1545 |         256,
1546 |         512,
1547 |         384
1548 |       ],
1549 |       "collapsed_to_k384": [
1550 |         false,
1551 |         false,
1552 |         false,
1553 |         false
1554 |       ],
1555 |       "execution_slots": [
1556 |         384,
1557 |         256,
1558 |         512,
1559 |         384
1560 |       ],
1561 |       "padding_slots": [
1562 |         0,
1563 |         0,
1564 |         0,
1565 |         0
1566 |       ],
1567 |       "predicted_total_utility": 0.1640600860118866,
1568 |       "requested_budgets": [
1569 |         384,
1570 |         256,
1571 |         512,
1572 |         384
1573 |       ],
1574 |       "target_actual_cost": 1536,
1575 |       "window_count": 4
1576 |     },
1577 |     "video_validation_0000158": {
1578 |       "actual_budget_error": 0,
1579 |       "actual_cost": [
1580 |         512,
1581 |         256,
1582 |         384,
1583 |         384,
1584 |         384
1585 |       ],
1586 |       "budgets": [
1587 |         512,
1588 |         256,
1589 |         384,
1590 |         384,
1591 |         384
1592 |       ],
1593 |       "collapsed_to_k384": [
1594 |         false,
1595 |         false,
1596 |         false,
1597 |         false,
1598 |         false
1599 |       ],
1600 |       "execution_slots": [
1601 |         512,
1602 |         256,
1603 |         384,
1604 |         384,
1605 |         384
1606 |       ],
1607 |       "padding_slots": [
1608 |         0,
1609 |         0,
1610 |         0,
1611 |         0,
1612 |         0
1613 |       ],
1614 |       "predicted_total_utility": 0.06998297572135925,
1615 |       "requested_budgets": [
1616 |         512,
1617 |         256,
1618 |         384,
1619 |         384,
1620 |         384
1621 |       ],
1622 |       "target_actual_cost": 1920,
1623 |       "window_count": 5
1624 |     },
1625 |     "video_validation_0000164": {
1626 |       "actual_budget_error": 0,
1627 |       "actual_cost": [
1628 |         384,
1629 |         384,
1630 |         384
1631 |       ],
1632 |       "budgets": [
1633 |         384,
1634 |         384,
1635 |         384
1636 |       ],
1637 |       "collapsed_to_k384": [
1638 |         false,
1639 |         false,
1640 |         false
1641 |       ],
1642 |       "execution_slots": [
1643 |         384,
1644 |         384,
1645 |         384
1646 |       ],
1647 |       "padding_slots": [
1648 |         0,
1649 |         0,
1650 |         0
1651 |       ],
1652 |       "predicted_total_utility": 0.0,
1653 |       "requested_budgets": [
1654 |         384,
1655 |         384,
1656 |         384
1657 |       ],
1658 |       "target_actual_cost": 1152,
1659 |       "window_count": 3
1660 |     },
1661 |     "video_validation_0000172": {
1662 |       "actual_budget_error": 0,
1663 |       "actual_cost": [
1664 |         384,
1665 |         384
1666 |       ],
1667 |       "budgets": [
1668 |         384,
1669 |         384
1670 |       ],
1671 |       "collapsed_to_k384": [
1672 |         false,
1673 |         false
1674 |       ],
1675 |       "execution_slots": [
1676 |         384,
1677 |         384
1678 |       ],
1679 |       "padding_slots": [
1680 |         0,
1681 |         0
1682 |       ],
1683 |       "predicted_total_utility": 0.0,
1684 |       "requested_budgets": [
1685 |         384,
1686 |         384
1687 |       ],
1688 |       "target_actual_cost": 768,
1689 |       "window_count": 2
1690 |     },
1691 |     "video_validation_0000173": {
1692 |       "actual_budget_error": 0,
1693 |       "actual_cost": [
1694 |         384,
1695 |         384,
1696 |         256,
1697 |         512
1698 |       ],
1699 |       "budgets": [
1700 |         384,
1701 |         384,
1702 |         256,
1703 |         512
1704 |       ],
1705 |       "collapsed_to_k384": [
1706 |         false,
1707 |         false,
1708 |         false,
1709 |         false
1710 |       ],
1711 |       "execution_slots": [
1712 |         384,
1713 |         384,
1714 |         256,
1715 |         512
1716 |       ],
1717 |       "padding_slots": [
1718 |         0,
1719 |         0,
1720 |         0,
1721 |         0
1722 |       ],
1723 |       "predicted_total_utility": 0.1362585425376892,
1724 |       "requested_budgets": [
1725 |         384,
1726 |         384,
1727 |         256,
1728 |         512
1729 |       ],
1730 |       "target_actual_cost": 1536,
1731 |       "window_count": 4
1732 |     },
1733 |     "video_validation_0000181": {
1734 |       "actual_budget_error": 0,
1735 |       "actual_cost": [
1736 |         384,
1737 |         384
1738 |       ],
1739 |       "budgets": [
1740 |         384,
1741 |         384
1742 |       ],
1743 |       "collapsed_to_k384": [
1744 |         false,
1745 |         false
1746 |       ],
1747 |       "execution_slots": [
1748 |         384,
1749 |         384
1750 |       ],
1751 |       "padding_slots": [
1752 |         0,
1753 |         0
1754 |       ],
1755 |       "predicted_total_utility": 0.0,
1756 |       "requested_budgets": [
1757 |         384,
1758 |         384
1759 |       ],
1760 |       "target_actual_cost": 768,
1761 |       "window_count": 2
1762 |     },
1763 |     "video_validation_0000186": {
1764 |       "actual_budget_error": 0,
1765 |       "actual_cost": [
1766 |         384
1767 |       ],
1768 |       "budgets": [
1769 |         384
1770 |       ],
1771 |       "collapsed_to_k384": [
1772 |         false
1773 |       ],
1774 |       "execution_slots": [
1775 |         384
1776 |       ],
1777 |       "padding_slots": [
1778 |         0
1779 |       ],
1780 |       "predicted_total_utility": 0.0,
1781 |       "requested_budgets": [
1782 |         384
1783 |       ],
1784 |       "target_actual_cost": 384,
1785 |       "window_count": 1
1786 |     },
1787 |     "video_validation_0000206": {
1788 |       "actual_budget_error": 0,
1789 |       "actual_cost": [
1790 |         384,
1791 |         384,
1792 |         384
1793 |       ],
1794 |       "budgets": [
1795 |         384,
1796 |         384,
1797 |         384
1798 |       ],
1799 |       "collapsed_to_k384": [
1800 |         false,
1801 |         false,
1802 |         false
1803 |       ],
1804 |       "execution_slots": [
1805 |         384,
1806 |         384,
1807 |         384
1808 |       ],
1809 |       "padding_slots": [
1810 |         0,
1811 |         0,
1812 |         0
1813 |       ],
1814 |       "predicted_total_utility": 0.0,
1815 |       "requested_budgets": [
1816 |         384,
1817 |         384,
1818 |         384
1819 |       ],
1820 |       "target_actual_cost": 1152,
1821 |       "window_count": 3
1822 |     },
1823 |     "video_validation_0000207": {
1824 |       "actual_budget_error": 0,
1825 |       "actual_cost": [
1826 |         384,
1827 |         384,
1828 |         384
1829 |       ],
1830 |       "budgets": [
1831 |         384,
1832 |         384,
1833 |         384
1834 |       ],
1835 |       "collapsed_to_k384": [
1836 |         false,
1837 |         false,
1838 |         false
1839 |       ],
1840 |       "execution_slots": [
1841 |         384,
1842 |         384,
1843 |         384
1844 |       ],
1845 |       "padding_slots": [
1846 |         0,
1847 |         0,
1848 |         0
1849 |       ],
1850 |       "predicted_total_utility": 0.0,
1851 |       "requested_budgets": [
1852 |         384,
1853 |         384,
1854 |         384
1855 |       ],
1856 |       "target_actual_cost": 1152,
1857 |       "window_count": 3
1858 |     },
1859 |     "video_validation_0000266": {
1860 |       "actual_budget_error": 0,
1861 |       "actual_cost": [
1862 |         384,
1863 |         384,
1864 |         384
1865 |       ],
1866 |       "budgets": [
1867 |         384,
1868 |         384,
1869 |         384
1870 |       ],
1871 |       "collapsed_to_k384": [
1872 |         false,
1873 |         false,
1874 |         false
1875 |       ],
1876 |       "execution_slots": [
1877 |         384,
1878 |         384,
1879 |         384
1880 |       ],
1881 |       "padding_slots": [
1882 |         0,
1883 |         0,
1884 |         0
1885 |       ],
1886 |       "predicted_total_utility": 0.0,
1887 |       "requested_budgets": [
1888 |         384,
1889 |         384,
1890 |         384
1891 |       ],
1892 |       "target_actual_cost": 1152,
1893 |       "window_count": 3
1894 |     },
1895 |     "video_validation_0000267": {
1896 |       "actual_budget_error": 0,
1897 |       "actual_cost": [
1898 |         512,
1899 |         384,
1900 |         256,
1901 |         384,
1902 |         384,
1903 |         384,
1904 |         384
1905 |       ],
1906 |       "budgets": [
1907 |         512,
1908 |         384,
1909 |         256,
1910 |         384,
1911 |         384,
1912 |         384,
1913 |         384
1914 |       ],
1915 |       "collapsed_to_k384": [
1916 |         false,
1917 |         false,
1918 |         false,
1919 |         false,
1920 |         false,
1921 |         false,
1922 |         false
1923 |       ],
1924 |       "execution_slots": [
1925 |         512,
1926 |         384,
1927 |         256,
1928 |         384,
1929 |         384,
1930 |         384,
1931 |         384
1932 |       ],
1933 |       "padding_slots": [
1934 |         0,
1935 |         0,
1936 |         0,
1937 |         0,
1938 |         0,
1939 |         0,
1940 |         0
1941 |       ],
1942 |       "predicted_total_utility": 0.10147061944007874,
1943 |       "requested_budgets": [
1944 |         512,
1945 |         384,
1946 |         256,
1947 |         384,
1948 |         384,
1949 |         384,
1950 |         384
1951 |       ],
1952 |       "target_actual_cost": 2688,
1953 |       "window_count": 7
1954 |     },
1955 |     "video_validation_0000282": {
1956 |       "actual_budget_error": 0,
1957 |       "actual_cost": [
1958 |         288
1959 |       ],
1960 |       "budgets": [
1961 |         384
1962 |       ],
1963 |       "collapsed_to_k384": [
1964 |         false
1965 |       ],
1966 |       "execution_slots": [
1967 |         384
1968 |       ],
1969 |       "padding_slots": [
1970 |         96
1971 |       ],
1972 |       "predicted_total_utility": 0.0,
1973 |       "requested_budgets": [
1974 |         384
1975 |       ],
1976 |       "target_actual_cost": 288,
1977 |       "window_count": 1
1978 |     },
1979 |     "video_validation_0000283": {
1980 |       "actual_budget_error": 0,
1981 |       "actual_cost": [
1982 |         384
1983 |       ],
1984 |       "budgets": [
1985 |         384
1986 |       ],
1987 |       "collapsed_to_k384": [
1988 |         false
1989 |       ],
1990 |       "execution_slots": [
1991 |         384
1992 |       ],
1993 |       "padding_slots": [
1994 |         0
1995 |       ],
1996 |       "predicted_total_utility": 0.0,
1997 |       "requested_budgets": [
1998 |         384
1999 |       ],
2000 |       "target_actual_cost": 384,
2001 |       "window_count": 1
2002 |     },
2003 |     "video_validation_0000285": {
2004 |       "actual_budget_error": 0,
2005 |       "actual_cost": [
2006 |         384,
2007 |         384
2008 |       ],
2009 |       "budgets": [
2010 |         384,
2011 |         384
2012 |       ],
2013 |       "collapsed_to_k384": [
2014 |         false,
2015 |         false
2016 |       ],
2017 |       "execution_slots": [
2018 |         384,
2019 |         384
2020 |       ],
2021 |       "padding_slots": [
2022 |         0,
2023 |         0
2024 |       ],
2025 |       "predicted_total_utility": 0.0,
2026 |       "requested_budgets": [
2027 |         384,
2028 |         384
2029 |       ],
2030 |       "target_actual_cost": 768,
2031 |       "window_count": 2
2032 |     },
2033 |     "video_validation_0000367": {
2034 |       "actual_budget_error": 0,
2035 |       "actual_cost": [
2036 |         384,
2037 |         384
2038 |       ],
2039 |       "budgets": [
2040 |         384,
2041 |         384
2042 |       ],
2043 |       "collapsed_to_k384": [
2044 |         false,
2045 |         false
2046 |       ],
2047 |       "execution_slots": [
2048 |         384,
2049 |         384
2050 |       ],
2051 |       "padding_slots": [
2052 |         0,
2053 |         0
2054 |       ],
2055 |       "predicted_total_utility": 0.0,
2056 |       "requested_budgets": [
2057 |         384,
2058 |         384
2059 |       ],
2060 |       "target_actual_cost": 768,
2061 |       "window_count": 2
2062 |     },
2063 |     "video_validation_0000417": {
2064 |       "actual_budget_error": 0,
2065 |       "actual_cost": [
2066 |         256,
2067 |         512,
2068 |         384,
2069 |         384,
2070 |         384,
2071 |         384
2072 |       ],
2073 |       "budgets": [
2074 |         256,
2075 |         512,
2076 |         384,
2077 |         384,
2078 |         384,
2079 |         384
2080 |       ],
2081 |       "collapsed_to_k384": [
2082 |         false,
2083 |         false,
2084 |         false,
2085 |         false,
2086 |         false,
2087 |         false
2088 |       ],
2089 |       "execution_slots": [
2090 |         256,
2091 |         512,
2092 |         384,
2093 |         384,
2094 |         384,
2095 |         384
2096 |       ],
2097 |       "padding_slots": [
2098 |         0,
2099 |         0,
2100 |         0,
2101 |         0,
2102 |         0,
2103 |         0
2104 |       ],
2105 |       "predicted_total_utility": 0.019536063075065613,
2106 |       "requested_budgets": [
2107 |         256,
2108 |         512,
2109 |         384,
2110 |         384,
2111 |         384,
2112 |         384
2113 |       ],
2114 |       "target_actual_cost": 2304,
2115 |       "window_count": 6
2116 |     },
2117 |     "video_validation_0000419": {
2118 |       "actual_budget_error": 0,
2119 |       "actual_cost": [
2120 |         384,
2121 |         384,
2122 |         384,
2123 |         256,
2124 |         384,
2125 |         384,
2126 |         384,
2127 |         384,
2128 |         384,
2129 |         512,
2130 |         256,
2131 |         256,
2132 |         512,
2133 |         512
2134 |       ],
2135 |       "budgets": [
2136 |         384,
2137 |         384,
2138 |         384,
2139 |         256,
2140 |         384,
2141 |         384,
2142 |         384,
2143 |         384,
2144 |         384,
2145 |         512,
2146 |         256,
2147 |         256,
2148 |         512,
2149 |         512
2150 |       ],
2151 |       "collapsed_to_k384": [
2152 |         false,
2153 |         false,
2154 |         false,
2155 |         false,
2156 |         false,
2157 |         false,
2158 |         false,
2159 |         false,
2160 |         false,
2161 |         false,
2162 |         false,
2163 |         false,
2164 |         false,
2165 |         false
2166 |       ],
2167 |       "execution_slots": [
2168 |         384,
2169 |         384,
2170 |         384,
2171 |         256,
2172 |         384,
2173 |         384,
2174 |         384,
2175 |         384,
2176 |         384,
2177 |         512,
2178 |         256,
2179 |         256,
2180 |         512,
2181 |         512
2182 |       ],
2183 |       "padding_slots": [
2184 |         0,
2185 |         0,
2186 |         0,
2187 |         0,
2188 |         0,
2189 |         0,
2190 |         0,
2191 |         0,
2192 |         0,
2193 |         0,
2194 |         0,
2195 |         0,
2196 |         0,
2197 |         0
2198 |       ],
2199 |       "predicted_total_utility": 0.10009123384952545,
2200 |       "requested_budgets": [
2201 |         384,
2202 |         384,
2203 |         384,
2204 |         256,
2205 |         384,
2206 |         384,
2207 |         384,
2208 |         384,
2209 |         384,
2210 |         512,
2211 |         256,
2212 |         256,
2213 |         512,
2214 |         512
2215 |       ],
2216 |       "target_actual_cost": 5376,
2217 |       "window_count": 14
2218 |     },
2219 |     "video_validation_0000483": {
2220 |       "actual_budget_error": 0,
2221 |       "actual_cost": [
2222 |         384,
2223 |         384
2224 |       ],
2225 |       "budgets": [
2226 |         384,
2227 |         384
2228 |       ],
2229 |       "collapsed_to_k384": [
2230 |         false,
2231 |         false
2232 |       ],
2233 |       "execution_slots": [
2234 |         384,
2235 |         384
2236 |       ],
2237 |       "padding_slots": [
2238 |         0,
2239 |         0
2240 |       ],
2241 |       "predicted_total_utility": 0.0,
2242 |       "requested_budgets": [
2243 |         384,
2244 |         384
2245 |       ],
2246 |       "target_actual_cost": 768,
2247 |       "window_count": 2
2248 |     },
2249 |     "video_validation_0000489": {
2250 |       "actual_budget_error": 0,
2251 |       "actual_cost": [
2252 |         384,
2253 |         384,
2254 |         384
2255 |       ],
2256 |       "budgets": [
2257 |         384,
2258 |         384,
2259 |         384
2260 |       ],
2261 |       "collapsed_to_k384": [
2262 |         false,
2263 |         false,
2264 |         false
2265 |       ],
2266 |       "execution_slots": [
2267 |         384,
2268 |         384,
2269 |         384
2270 |       ],
2271 |       "padding_slots": [
2272 |         0,
2273 |         0,
2274 |         0
2275 |       ],
2276 |       "predicted_total_utility": 0.0,
2277 |       "requested_budgets": [
2278 |         384,
2279 |         384,
2280 |         384
2281 |       ],
2282 |       "target_actual_cost": 1152,
2283 |       "window_count": 3
2284 |     },
2285 |     "video_validation_0000490": {
2286 |       "actual_budget_error": 0,
2287 |       "actual_cost": [
2288 |         384,
2289 |         384,
2290 |         512,
2291 |         256,
2292 |         384,
2293 |         384
2294 |       ],
2295 |       "budgets": [
2296 |         384,
2297 |         384,
2298 |         512,
2299 |         256,
2300 |         384,
2301 |         384
2302 |       ],
2303 |       "collapsed_to_k384": [
2304 |         false,
2305 |         false,
2306 |         false,
2307 |         false,
2308 |         false,
2309 |         false
2310 |       ],
2311 |       "execution_slots": [
2312 |         384,
2313 |         384,
2314 |         512,
2315 |         256,
2316 |         384,
2317 |         384
2318 |       ],
2319 |       "padding_slots": [
2320 |         0,
2321 |         0,
2322 |         0,
2323 |         0,
2324 |         0,
2325 |         0
2326 |       ],
2327 |       "predicted_total_utility": 0.13576892018318176,
2328 |       "requested_budgets": [
2329 |         384,
2330 |         384,
2331 |         512,
2332 |         256,
2333 |         384,
2334 |         384
2335 |       ],
2336 |       "target_actual_cost": 2304,
2337 |       "window_count": 6
2338 |     },
2339 |     "video_validation_0000664": {
2340 |       "actual_budget_error": 0,
2341 |       "actual_cost": [
2342 |         256,
2343 |         384,
2344 |         384,
2345 |         512
2346 |       ],
2347 |       "budgets": [
2348 |         256,
2349 |         384,
2350 |         384,
2351 |         512
2352 |       ],
2353 |       "collapsed_to_k384": [
2354 |         false,
2355 |         false,
2356 |         false,
2357 |         false
2358 |       ],
2359 |       "execution_slots": [
2360 |         256,
2361 |         384,
2362 |         384,
2363 |         512
2364 |       ],
2365 |       "padding_slots": [
2366 |         0,
2367 |         0,
2368 |         0,
2369 |         0
2370 |       ],
2371 |       "predicted_total_utility": 0.009859195910394192,
2372 |       "requested_budgets": [
2373 |         256,
2374 |         384,
2375 |         384,
2376 |         512
2377 |       ],
2378 |       "target_actual_cost": 1536,
2379 |       "window_count": 4
2380 |     },
2381 |     "video_validation_0000681": {
2382 |       "actual_budget_error": 0,
2383 |       "actual_cost": [
2384 |         384
2385 |       ],
2386 |       "budgets": [
2387 |         384
2388 |       ],
2389 |       "collapsed_to_k384": [
2390 |         false
2391 |       ],
2392 |       "execution_slots": [
2393 |         384
2394 |       ],
2395 |       "padding_slots": [
2396 |         0
2397 |       ],
2398 |       "predicted_total_utility": 0.0,
2399 |       "requested_budgets": [
2400 |         384
2401 |       ],
2402 |       "target_actual_cost": 384,
2403 |       "window_count": 1
2404 |     },
2405 |     "video_validation_0000683": {
2406 |       "actual_budget_error": 0,
2407 |       "actual_cost": [
2408 |         285
2409 |       ],
2410 |       "budgets": [
2411 |         384
2412 |       ],
2413 |       "collapsed_to_k384": [
2414 |         false
2415 |       ],
2416 |       "execution_slots": [
2417 |         384
2418 |       ],
2419 |       "padding_slots": [
2420 |         99
2421 |       ],
2422 |       "predicted_total_utility": 0.0,
2423 |       "requested_budgets": [
2424 |         384
2425 |       ],
2426 |       "target_actual_cost": 285,
2427 |       "window_count": 1
2428 |     },
2429 |     "video_validation_0000690": {
2430 |       "actual_budget_error": 0,
2431 |       "actual_cost": [
2432 |         384,
2433 |         256,
2434 |         384,
2435 |         512,
2436 |         384
2437 |       ],
2438 |       "budgets": [
2439 |         384,
2440 |         256,
2441 |         384,
2442 |         512,
2443 |         384
2444 |       ],
2445 |       "collapsed_to_k384": [
2446 |         false,
2447 |         false,
2448 |         false,
2449 |         false,
2450 |         false
2451 |       ],
2452 |       "execution_slots": [
2453 |         384,
2454 |         256,
2455 |         384,
2456 |         512,
2457 |         384
2458 |       ],
2459 |       "padding_slots": [
2460 |         0,
2461 |         0,
2462 |         0,
2463 |         0,
2464 |         0
2465 |       ],
2466 |       "predicted_total_utility": 0.004800617229193449,
2467 |       "requested_budgets": [
2468 |         384,
2469 |         256,
2470 |         384,
2471 |         512,
2472 |         384
2473 |       ],
2474 |       "target_actual_cost": 1920,
2475 |       "window_count": 5
2476 |     },
2477 |     "video_validation_0000783": {
2478 |       "actual_budget_error": 0,
2479 |       "actual_cost": [
2480 |         384,
2481 |         384,
2482 |         384
2483 |       ],
2484 |       "budgets": [
2485 |         384,
2486 |         384,
2487 |         384
2488 |       ],
2489 |       "collapsed_to_k384": [
2490 |         false,
2491 |         false,
2492 |         false
2493 |       ],
2494 |       "execution_slots": [
2495 |         384,
2496 |         384,
2497 |         384
2498 |       ],
2499 |       "padding_slots": [
2500 |         0,
2501 |         0,
2502 |         0
2503 |       ],
2504 |       "predicted_total_utility": 0.0,
2505 |       "requested_budgets": [
2506 |         384,
2507 |         384,
2508 |         384
2509 |       ],
2510 |       "target_actual_cost": 1152,
2511 |       "window_count": 3
2512 |     },
2513 |     "video_validation_0000851": {
2514 |       "actual_budget_error": 0,
2515 |       "actual_cost": [
2516 |         233
2517 |       ],
2518 |       "budgets": [
2519 |         384
2520 |       ],
2521 |       "collapsed_to_k384": [
2522 |         false
2523 |       ],
2524 |       "execution_slots": [
2525 |         384
2526 |       ],
2527 |       "padding_slots": [
2528 |         151
2529 |       ],
2530 |       "predicted_total_utility": 0.0,
2531 |       "requested_budgets": [
2532 |         384
2533 |       ],
2534 |       "target_actual_cost": 233,
2535 |       "window_count": 1
2536 |     },
2537 |     "video_validation_0000852": {
2538 |       "actual_budget_error": 0,
2539 |       "actual_cost": [
2540 |         384,
2541 |         384,
2542 |         384,
2543 |         384
2544 |       ],
2545 |       "budgets": [
2546 |         384,
2547 |         384,
2548 |         384,
2549 |         384
2550 |       ],
2551 |       "collapsed_to_k384": [
2552 |         false,
2553 |         false,
2554 |         false,
2555 |         false
2556 |       ],
2557 |       "execution_slots": [
2558 |         384,
2559 |         384,
2560 |         384,
2561 |         384
2562 |       ],
2563 |       "padding_slots": [
2564 |         0,
2565 |         0,
2566 |         0,
2567 |         0
2568 |       ],
2569 |       "predicted_total_utility": 0.0,
2570 |       "requested_budgets": [
2571 |         384,
2572 |         384,
2573 |         384,
2574 |         384
2575 |       ],
2576 |       "target_actual_cost": 1536,
2577 |       "window_count": 4
2578 |     },
2579 |     "video_validation_0000858": {
2580 |       "actual_budget_error": 0,
2581 |       "actual_cost": [
2582 |         384,
2583 |         384
2584 |       ],
2585 |       "budgets": [
2586 |         384,
2587 |         384
2588 |       ],
2589 |       "collapsed_to_k384": [
2590 |         false,
2591 |         false
2592 |       ],
2593 |       "execution_slots": [
2594 |         384,
2595 |         384
2596 |       ],
2597 |       "padding_slots": [
2598 |         0,
2599 |         0
2600 |       ],
2601 |       "predicted_total_utility": 0.0,
2602 |       "requested_budgets": [
2603 |         384,
2604 |         384
2605 |       ],
2606 |       "target_actual_cost": 768,
2607 |       "window_count": 2
2608 |     },
2609 |     "video_validation_0000905": {
2610 |       "actual_budget_error": 0,
2611 |       "actual_cost": [
2612 |         384,
2613 |         384,
2614 |         384
2615 |       ],
2616 |       "budgets": [
2617 |         384,
2618 |         384,
2619 |         384
2620 |       ],
2621 |       "collapsed_to_k384": [
2622 |         false,
2623 |         false,
2624 |         false
2625 |       ],
2626 |       "execution_slots": [
2627 |         384,
2628 |         384,
2629 |         384
2630 |       ],
2631 |       "padding_slots": [
2632 |         0,
2633 |         0,
2634 |         0
2635 |       ],
2636 |       "predicted_total_utility": 0.0,
2637 |       "requested_budgets": [
2638 |         384,
2639 |         384,
2640 |         384
2641 |       ],
2642 |       "target_actual_cost": 1152,
2643 |       "window_count": 3
2644 |     },
2645 |     "video_validation_0000908": {
2646 |       "actual_budget_error": 0,
2647 |       "actual_cost": [
2648 |         384,
2649 |         384,
2650 |         384,
2651 |         384,
2652 |         384
2653 |       ],
2654 |       "budgets": [
2655 |         384,
2656 |         384,
2657 |         384,
2658 |         384,
2659 |         384
2660 |       ],
2661 |       "collapsed_to_k384": [
2662 |         false,
2663 |         false,
2664 |         false,
2665 |         false,
2666 |         false
2667 |       ],
2668 |       "execution_slots": [
2669 |         384,
2670 |         384,
2671 |         384,
2672 |         384,
2673 |         384
2674 |       ],
2675 |       "padding_slots": [
2676 |         0,
2677 |         0,
2678 |         0,
2679 |         0,
2680 |         0
2681 |       ],
2682 |       "predicted_total_utility": 0.0,
2683 |       "requested_budgets": [
2684 |         384,
2685 |         384,
2686 |         384,
2687 |         384,
2688 |         384
2689 |       ],
2690 |       "target_actual_cost": 1920,
2691 |       "window_count": 5
2692 |     },
2693 |     "video_validation_0000909": {
2694 |       "actual_budget_error": 0,
2695 |       "actual_cost": [
2696 |         384,
2697 |         384,
2698 |         384
2699 |       ],
2700 |       "budgets": [
2701 |         384,
2702 |         384,
2703 |         384
2704 |       ],
2705 |       "collapsed_to_k384": [
2706 |         false,
2707 |         false,
2708 |         false
2709 |       ],
2710 |       "execution_slots": [
2711 |         384,
2712 |         384,
2713 |         384
2714 |       ],
2715 |       "padding_slots": [
2716 |         0,
2717 |         0,
2718 |         0
2719 |       ],
2720 |       "predicted_total_utility": 0.0,
2721 |       "requested_budgets": [
2722 |         384,
2723 |         384,
2724 |         384
2725 |       ],
2726 |       "target_actual_cost": 1152,
2727 |       "window_count": 3
2728 |     },
2729 |     "video_validation_0000937": {
2730 |       "actual_budget_error": 0,
2731 |       "actual_cost": [
2732 |         384
2733 |       ],
2734 |       "budgets": [
2735 |         384
2736 |       ],
2737 |       "collapsed_to_k384": [
2738 |         false
2739 |       ],
2740 |       "execution_slots": [
2741 |         384
2742 |       ],
2743 |       "padding_slots": [
2744 |         0
2745 |       ],
2746 |       "predicted_total_utility": 0.0,
2747 |       "requested_budgets": [
2748 |         384
2749 |       ],
2750 |       "target_actual_cost": 384,
2751 |       "window_count": 1
2752 |     },
2753 |     "video_validation_0000938": {
2754 |       "actual_budget_error": 0,
2755 |       "actual_cost": [
2756 |         260
2757 |       ],
2758 |       "budgets": [
2759 |         384
2760 |       ],
2761 |       "collapsed_to_k384": [
2762 |         false
2763 |       ],
2764 |       "execution_slots": [
2765 |         384
2766 |       ],
2767 |       "padding_slots": [
2768 |         124
2769 |       ],
2770 |       "predicted_total_utility": 0.0,
2771 |       "requested_budgets": [
2772 |         384
2773 |       ],
2774 |       "target_actual_cost": 260,
2775 |       "window_count": 1
2776 |     },
2777 |     "video_validation_0000943": {
2778 |       "actual_budget_error": 0,
2779 |       "actual_cost": [
2780 |         384,
2781 |         384
2782 |       ],
2783 |       "budgets": [
2784 |         384,
2785 |         384
2786 |       ],
2787 |       "collapsed_to_k384": [
2788 |         false,
2789 |         false
2790 |       ],
2791 |       "execution_slots": [
2792 |         384,
2793 |         384
2794 |       ],
2795 |       "padding_slots": [
2796 |         0,
2797 |         0
2798 |       ],
2799 |       "predicted_total_utility": 0.0,
2800 |       "requested_budgets": [
2801 |         384,
2802 |         384
2803 |       ],
2804 |       "target_actual_cost": 768,
2805 |       "window_count": 2
2806 |     },
2807 |     "video_validation_0000944": {
2808 |       "actual_budget_error": 0,
2809 |       "actual_cost": [
2810 |         384,
2811 |         384,
2812 |         384
2813 |       ],
2814 |       "budgets": [
2815 |         384,
2816 |         384,
2817 |         384
2818 |       ],
2819 |       "collapsed_to_k384": [
2820 |         false,
2821 |         false,
2822 |         false
2823 |       ],
2824 |       "execution_slots": [
2825 |         384,
2826 |         384,
2827 |         384
2828 |       ],
2829 |       "padding_slots": [
2830 |         0,
2831 |         0,
2832 |         0
2833 |       ],
2834 |       "predicted_total_utility": 0.0,
2835 |       "requested_budgets": [
2836 |         384,
2837 |         384,
2838 |         384
2839 |       ],
2840 |       "target_actual_cost": 1152,
2841 |       "window_count": 3
2842 |     },
2843 |     "video_validation_0000945": {
2844 |       "actual_budget_error": 0,
2845 |       "actual_cost": [
2846 |         384,
2847 |         384,
2848 |         384
2849 |       ],
2850 |       "budgets": [
2851 |         384,
2852 |         384,
2853 |         384
2854 |       ],
2855 |       "collapsed_to_k384": [
2856 |         false,
2857 |         false,
2858 |         false
2859 |       ],
2860 |       "execution_slots": [
2861 |         384,
2862 |         384,
2863 |         384
2864 |       ],
2865 |       "padding_slots": [
2866 |         0,
2867 |         0,
2868 |         0
2869 |       ],
2870 |       "predicted_total_utility": 0.0,
2871 |       "requested_budgets": [
2872 |         384,
2873 |         384,
2874 |         384
2875 |       ],
2876 |       "target_actual_cost": 1152,
2877 |       "window_count": 3
2878 |     },
2879 |     "video_validation_0000984": {
2880 |       "actual_budget_error": 0,
2881 |       "actual_cost": [
2882 |         348
2883 |       ],
2884 |       "budgets": [
2885 |         384
2886 |       ],
2887 |       "collapsed_to_k384": [
2888 |         false
2889 |       ],
2890 |       "execution_slots": [
2891 |         384
2892 |       ],
2893 |       "padding_slots": [
2894 |         36
2895 |       ],
2896 |       "predicted_total_utility": 0.0,
2897 |       "requested_budgets": [
2898 |         384
2899 |       ],
2900 |       "target_actual_cost": 348,
2901 |       "window_count": 1
2902 |     },
2903 |     "video_validation_0000988": {
2904 |       "actual_budget_error": 0,
2905 |       "actual_cost": [
2906 |         384,
2907 |         384
2908 |       ],
2909 |       "budgets": [
2910 |         384,
2911 |         384
2912 |       ],
2913 |       "collapsed_to_k384": [
2914 |         false,
2915 |         false
2916 |       ],
2917 |       "execution_slots": [
2918 |         384,
2919 |         384
2920 |       ],
2921 |       "padding_slots": [
2922 |         0,
2923 |         0
2924 |       ],
2925 |       "predicted_total_utility": 0.0,
2926 |       "requested_budgets": [
2927 |         384,
2928 |         384
2929 |       ],
2930 |       "target_actual_cost": 768,
2931 |       "window_count": 2
2932 |     },
2933 |     "video_validation_0000990": {
2934 |       "actual_budget_error": 0,
2935 |       "actual_cost": [
2936 |         384,
2937 |         384
2938 |       ],
2939 |       "budgets": [
2940 |         384,
2941 |         384
2942 |       ],
2943 |       "collapsed_to_k384": [
2944 |         false,
2945 |         false
2946 |       ],
2947 |       "execution_slots": [
2948 |         384,
2949 |         384
2950 |       ],
2951 |       "padding_slots": [
2952 |         0,
2953 |         0
2954 |       ],
2955 |       "predicted_total_utility": 0.0,
2956 |       "requested_budgets": [
2957 |         384,
2958 |         384
2959 |       ],
2960 |       "target_actual_cost": 768,
2961 |       "window_count": 2
2962 |     }
2963 |   },
2964 |   "capped_allocation_summary": {
2965 |     "actual_budget_error": 0,
2966 |     "actual_observation_cost": 47110,
2967 |     "budget_counts": {
2968 |       "256": 11,
2969 |       "384": 102,
2970 |       "512": 11
2971 |     },
2972 |     "capped_allocation_limit_hit_video_count": 5,
2973 |     "changed_video_count": 9,
2974 |     "changed_window_count": 22,
2975 |     "target_observation_cost": 47110
2976 |   },
2977 |   "capped_oracle_384": {
2978 |     "average_mAP": 0.888567863693566,
2979 |     "mAP@0.3": 0.960349418437806,
2980 |     "mAP@0.4": 0.9399008234073724,
2981 |     "mAP@0.5": 0.9104458544593358,
2982 |     "mAP@0.6": 0.8621473535441421,
2983 |     "mAP@0.7": 0.7699958686191737
2984 |   },
2985 |   "detector_forward_executed": false,
2986 |   "fixed_h65_384": {
2987 |     "average_mAP": 0.8813119705666871,
2988 |     "mAP@0.3": 0.9583379279516155,
2989 |     "mAP@0.4": 0.936845289659517,
2990 |     "mAP@0.5": 0.8958122338381976,
2991 |     "mAP@0.6": 0.8528585697798222,
2992 |     "mAP@0.7": 0.7627058316042832
2993 |   },
2994 |   "holdout_video_count": 40,
2995 |   "holdout_window_count": 124,
2996 |   "intervention": {
2997 |     "capped_value": 0.5,
2998 |     "only_changed_variable": "max_changed_fraction",
2999 |     "released_value": 1.0
3000 |   },
3001 |   "method": "DUCA-Marginal-v1",
3002 |   "official_test_consumed": false,
3003 |   "original_result": {
3004 |     "path": "/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/probe_result.json",
3005 |     "reproduction_error_pp": {
3006 |       "capped_oracle": {
3007 |         "average_mAP": 0.0,
3008 |         "mAP@0.3": 0.0,
3009 |         "mAP@0.4": 0.0,
3010 |         "mAP@0.5": 0.0,
3011 |         "mAP@0.6": 0.0,
3012 |         "mAP@0.7": 0.0
3013 |       },
3014 |       "fixed_h65_384": {
3015 |         "average_mAP": 0.0,
3016 |         "mAP@0.3": 0.0,
3017 |         "mAP@0.4": 0.0,
3018 |         "mAP@0.5": 0.0,
3019 |         "mAP@0.6": 0.0,
3020 |         "mAP@0.7": 0.0
3021 |       }
3022 |     },
3023 |     "sha256": "8d6df7240c8b81b4d6d9aa8ff98bae530d6823ddd1d411bed47ce983ebd94925"
3024 |   },
3025 |   "paired_whole_video_bootstrap": null,
3026 |   "paper_claim_allowed": false,
3027 |   "provenance": {
3028 |     "base_revision": "f67d96fdf68a295eaa7f678f3dfc125530828889",
3029 |     "current_git": {
3030 |       "branch": "HEAD",
3031 |       "dirty": false,
3032 |       "head": "d2fad7c0dfc4a5efe98b10b9eee4723c6805699f"
3033 |     },
3034 |     "sealed_pre_run": "/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/pre_run_receipt.json",
3035 |     "stage_artifacts": {
3036 |       "k256": {
3037 |         "path": "/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/counterfactual_k256.jsonl.gz",
3038 |         "producer_revision": "f87555f7da362fe1a20d4ca08f7a68c975ed8280",
3039 |         "sha256": "6dc8893a41b5c8132b176f32133ffc2f48a5491146385c147b8227167608a309"
3040 |       },
3041 |       "k512": {
3042 |         "path": "/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/counterfactual_k512.jsonl.gz",
3043 |         "producer_revision": "f87555f7da362fe1a20d4ca08f7a68c975ed8280",
3044 |         "sha256": "c7fa06258c07163d0906b512a78e367c27607c64fc41b28bce9fe51fbd0815d7"
3045 |       },
3046 |       "selection": {
3047 |         "path": "/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/selection_k384.jsonl.gz",
3048 |         "producer_revision": "f87555f7da362fe1a20d4ca08f7a68c975ed8280",
3049 |         "sha256": "1d668d4e5eb4b5ef3c1057c97ec63cc2c1eed3c0e62297290520063b4e1ec38f"
3050 |       }
3051 |     }
3052 |   },
3053 |   "schema": "duca_marginal_oracle_cap_release_result_v1",
3054 |   "source": {
3055 |     "annotation": "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json",
3056 |     "annotation_sha256": "ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad",
3057 |     "checkpoint": "/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth",
3058 |     "checkpoint_epoch": 59,
3059 |     "checkpoint_sha256": "dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c",
3060 |     "checkpoint_state_key": "state_dict_ema",
3061 |     "class_map": "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt",
3062 |     "class_map_sha256": "a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31",
3063 |     "config": "/data/run01/sczc063/yuzibo/duca_marginal_cap_release_d2fad7c0_20260831/configs/adatad/thumos/duca_marginal_frozen_h65_probe.py",
3064 |     "config_sha256": "02e091995124496c5c5e0011923ac12de6b8dea29679ec821e64e6e5b6271ca6",
3065 |     "git": {
3066 |       "branch": "HEAD",
3067 |       "dirty": false,
3068 |       "head": "d2fad7c0dfc4a5efe98b10b9eee4723c6805699f"
3069 |     },
3070 |     "train_data": "/data/run01/sczc063/yuzibo/thumos14/raw_data/video",
3071 |     "videomae_pretrain": "/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
3072 |     "videomae_pretrain_sha256": "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
3073 |   },
3074 |   "status": "CAP_RELEASE_POINT_GATE_FAILED_STOP_CURRENT_MECHANISM",
3075 |   "utility_head_training_performed": false
3076 | }
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

## Answer

# PIVOT

**Nonce：`DUCA-MARGINAL-CAP-RELEASE-TERMINAL-ADJUDICATION-v001-20260831`**

**终止当前 DUCA-Marginal-v1 的“三档预算、逐窗口加性反事实损失、逐视频等总 observation 重分配”机制；不终止 DUCA 的任务感知动态计算方向。论文问题应转向：在重叠滑窗时序动作检测中，动态预算是否必须优化视频级联合预测集，而不能把窗口级检测损失边际简单相加。**

---

## 1. 终态证据核验

GitHub 分支确实指向精确提交 `d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`，其直接父提交是 `f67d96fdf68a295eaa7f678f3dfc125530828889`。该提交只修改 cap-release runner 和聚焦测试，没有改变 `dynamic_budget.py`、H65 producer、模型、数据、损失、NMS 或评估器。

代码审查与终态 JSON 一致：

* runner 要求干净提交、核对原 capped 结果及密封 K256/K384/K512 产物，重新计算 Fixed-H65、capped 和 released 三个结果，并且不能覆盖原 `probe_result.json`；
* 强门严格为 Avg-mAP 增益至少 `+0.8` 个百分点且 mAP@0.7 增益至少 `+1.0` 个百分点；只有两者同时通过才运行 10,000 次整视频配对 bootstrap；否则终止当前机制；
* allocator 在每个视频内最大化三个预算档位的**加性窗口效用**，严格保持实际 observation 总数等于全 K384 的目标，并使用确定性 tie-break；
* 新测试验证了解除上限能引入更多等成本 K256/K512 转移，同时不改变默认 0.5 路径和原结果文件。

终态比较如下：

| 分配                       |    Avg-mAP |    mAP@0.7 |         相对 Fixed-H65-384 |
| ------------------------ | ---------: | ---------: | -----------------------: |
| Fixed-H65-384            | 88.131197% | 76.270583% |                        — |
| 50% capped loss-oracle   | 88.856786% | 76.999587% | +0.725589 / +0.729004 pp |
| cap-released loss-oracle | 88.558507% | 76.720863% | +0.427310 / +0.450280 pp |

解除上限后：

* 相对 capped allocation 下降 `-0.298279/-0.278724` 个百分点；
* 分配从 `11/102/11` 变成 `17/90/17`；
* 只新增了 6 组等成本转移，即 12 个变化窗口，集中在 5 个视频；
* 总实际 observation 始终为 `47110`，预算误差为零；
* Fixed 与 capped 的全部复现误差为 `0.0`；
* 两项强门均失败，因此不运行 bootstrap 是冻结规则的正确执行，而不是遗漏。

附件中的旧研究状态、库存边界和历史注册表只承担背景、谱系和规则说明，不能覆盖本轮 `d2fad7c0...` 与终态 JSON。PJST-D1 的统计缺口也与本轮 Marginal-v1 终态无关，不重新开启。      项目与角色规则要求把真实负结果、实现成功和论文主张分别处理，并只下达一项能改变科学结论的任务。  

---

## 2. 当前负结果最支持什么

### 直接支持

**第一，当前 Marginal-v1 没有达到预先要求的强 oracle headroom。**

这里的“oracle”只意味着分配器看到了真实的逐窗口 K256/K384/K512 反事实检测损失。即使给予这种不可部署的特权信息，解除改变窗口上限后仍只有 `+0.427/+0.450` 个百分点，明显低于冻结强门。因此，不应再训练一个 predictor 去逼近同一个不足的目标。

**第二，逐窗口反事实检测损失不是最终 mAP 的充分加性效用。**

cap release 扩大了分配可行集。allocator 在扩大的可行集中获得了更高的内部加性效用，却得到更低的最终 mAP。因而被直接否定的不是“优化没找到解”，而是以下隐含假设：

> 在保持逐视频 observation 总成本不变时，独立窗口检测损失的改善可以相加，并能单调转化为视频级 TAD mAP 改善。

**第三，50% cap 更像一个偶然有效的 trust region，而不是得到验证的科学机制。**

它限制了弱效用信号被过度应用，因而在这批数据上优于 unrestricted allocation。但不能据此把 `0.5` 写成有普适意义的最优比例，也不能继续做 0.4、0.6、0.7 等 cap 搜索。

### 仍不能支持

本结果不能推出：

* 所有动态预算或动态计算方法均无效；
* K256/K512 档位本身无用；
* 当前三档可行空间完全没有任何 metric-level headroom；
* coverage、物理时间、边界保护或语义选帧假设无效；
* learned allocator 必然失败；
* 当前差异具有统计显著性；
* 在 official validation/test 上会出现相同结果；
* 已获得任何端到端效率结论。

尤其不能把训练侧 40-video holdout 上的 `88.xx%` 与 official validation 的 `65.xx/68.xx%` 直接比较。

---

## 3. 为什么解除上限后反而变差

### 最可能的主因：效用目标与最终评估的非加性错配

Marginal-v1 对每个窗口使用：

* K256 相对效用：`-(loss256 - loss384)`；
* K384 相对效用：`0`；
* K512 相对效用：`loss384 - loss512`。

随后把这些量在视频内相加并求精确预算最优解。最终 mAP 却是在所有窗口预测合并、滑窗去重、soft-NMS、类别内全局置信度排序和 AP 聚合之后得到。窗口损失和最终 mAP 之间不存在可保证的线性关系。

因此，以下现象完全可能同时发生：

1. 某个 K512 窗口的局部训练损失下降；
2. 与其重叠窗口产生更多重复或竞争 proposal；
3. soft-NMS 改变保留分数与排序；
4. 对应的 K256 窗口丢失另一段动作或背景抑制信息；
5. 最终 AP 下降。

released allocation 在 mAP@0.3 上甚至低于 Fixed-H65-384，而在 mAP@0.5–0.7 上仍保持小幅正值。这更符合 proposal 覆盖、分类排序或窗口去重受到干扰，而不是单纯的高 tIoU 边界问题。

### 次要放大因素：三档预算和精确成本约束过粗

K384 到 K256/K512 是一次 `±128` observation 的离散跃迁。严格等总成本通常要求一个 downgrade 与一个 upgrade 成对出现，而不是允许连续微调。

此外，`int(window_count × max_changed_fraction)` 与成对转移存在奇偶阶跃：

* 3 个窗口、50% cap 时最多允许改变 1 个窗口，但等成本转移至少需要改变 2 个，因此实际上不允许任何转移；
* cap release 后会突然允许一整对 K256/K512 改变。

这解释了为什么解除上限会离散地新增整组转移，但它本身不能解释为什么 mAP 下降；下降仍需要效用误排序或窗口间交互。

### 仍然存活的替代解释：可利用空间本来就很小

capped 和 released 都只显示亚百分点级开发集增益，因此当前固定 H65 priority sequence 加三档预算的真实可用空间可能有限。40-video holdout 也可能放大偶然波动。

但现有实验没有直接优化最终 mAP，也没有搜索联合预测集的 metric-level allocation，所以尚不能把“空间不足”定为唯一原因。

**综合判断：最强证据指向“窗口级加性效用错配”，其中重叠窗口干扰、NMS 和 AP 聚合是主要表现；粗预算档位与等成本组合约束是放大器；固有 headroom 较小是仍未排除的替代解释。**

---

## 4. 修订后的科学问题与路线

新的核心问题应为：

> 在固定逐视频重型 observation 总成本下，滑窗时序动作检测的预算分配是否必须依据多个重叠窗口联合产生的预测集效用，而不是独立窗口的检测损失边际？

这一路线的潜在机制不是再增加一个 utility head，而是把分配单位从“独立窗口”提升为“视频内相互重叠的窗口集合”，显式处理：

* 不同窗口之间的 proposal 重复与互补；
* 边界和动作区域的跨窗口覆盖；
* NMS 前后的置信度竞争；
* downgrade 与 upgrade 的联合后果。

这一问题具有论文价值，因为当前终态揭示了一个反直觉现象：**一个拥有真实窗口反事实损失、内部目标更优且预算完全守恒的 oracle，可能因忽略联合预测集结构而得到更差的最终 mAP。**

但现阶段不授权实现视频级 predictor、图网络、集合模型或新的训练损失。首先必须证明现有 sealed predictions 中确实存在可被联合选择恢复的 metric-level 空间。

---

## 5. 唯一当前任务单

### 任务名称

**cap-release 差分邻域的联合 mAP 穷举诊断**

### 科学问题

解除上限新增的 6 组等成本转移之所以降低 mAP，究竟是因为：

1. 每一组新增转移本身就被窗口损失误排序；
2. 某些转移单独有利，但组合后因窗口重叠、NMS 或 AP 聚合发生负交互；
3. 两者兼有；
4. 即使联合选择，局部 action space 也没有足够强的 headroom。

### 权威代码与允许修改面

* 基座：`d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`
* 新分支：`feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831`
* 只允许修改：

  * `tools/bata/run_duca_marginal_frozen_h65_probe.py`
  * `tests/test_duca_marginal_budget.py`
* `opentad/models/duca/dynamic_budget.py` 必须保持逐字不变。
* 不新增模型类、配置族、启动框架、通用搜索器或 provenance 系统。

### 最小实现

在现有 runner 中增加一个只读分析 stage：

1. 读取并验证原 capped result、cap-release terminal result 和三个密封 producer；
2. 从 capped 与 released allocation 的差异自动导出新增变化：

   * 当前预期为 5 个视频；
   * 12 个新增变化窗口；
   * 6 组净等成本转移；
3. 不得为 `video_validation_0000419` 人为指定两组配对。该视频有两个新增 downgrade 和两个新增 upgrade，必须枚举所有满足实际 observation 成本守恒的平衡子集；
4. 在每个视频内只允许使用 capped→released 差分窗口，并按真实 `actual_cost` 检查成本，不得假设所有窗口都正好是 ±128；
5. 对各视频合法状态做笛卡尔积。按当前终态应得到 **96 个唯一、确定性、逐视频等成本的联合状态**；实现必须由数据推导状态数，再断言当前输入确实为 96，不能只硬编码 96；
6. 对全部 96 个状态使用同一 40-video holdout、同一密封预测、同一 sliding-window NMS 和同一评估器计算：

   * Avg-mAP；
   * mAP@0.3/0.4/0.5/0.6/0.7；
   * 相对 Fixed、capped、released 的差值；
7. 单独报告：

   * 每个最小合法单转移的指标变化；
   * 所有可组合单转移的联合变化；
   * 联合变化减去单项变化之和的交互残差；
   * 96 个状态中的最佳点；
8. 写入独立结果文件，不得覆盖 `probe_result.json` 或 `oracle_cap_release_result.json`。

### 公平性与必须复现的控制

正式枚举前必须再次满足：

* Fixed-H65-384 全部指标复现误差不超过 `1e-6 pp`；
* capped oracle 全部指标复现误差不超过 `1e-6 pp`；
* released oracle 全部指标复现误差不超过 `1e-6 pp`；
* 每个枚举状态逐视频实际 observation 成本均与 K384 目标完全一致；
* 全局实际 observation 始终为 `47110`；
* 只允许 12 个 capped→released 差分窗口变化；
* detector/Scout forward 次数为零；
* 模型训练、utility-head 拟合、梯度计算和 official test 消耗均为零。

这是纯预测重组分析，因此不需要也不允许增加模型形状、梯度或物理坐标新功能。

### 主要指标与最便宜 falsifier

主要指标继续使用冻结的：

* 相对 Fixed-H65-384 的 Avg-mAP 增益；
* 相对 Fixed-H65-384 的 mAP@0.7 增益。

最便宜 falsifier 就是 96 个 CPU-only 官方评估器调用。无需 GPU、无需前向、无需训练。

### 继续与停止门

**联合效用问题获得继续研究资格，当且仅当：**

* 96 个状态中至少一个同时达到：

  * Avg-mAP 相对 Fixed-H65-384 `≥ +0.8 pp`；
  * mAP@0.7 相对 Fixed-H65-384 `≥ +1.0 pp`。

这只能证明当前局部 action space 存在 metric-level 开发集 headroom，并证明加性损失排序遗漏了它。结果必须返回 Pro；不得自动训练任何 predictor。

**若没有任何状态同时通过两项强门：**

* 终止“用视频级联合效用修复本次 cap-release 差分”的路线；
* 不再对当前 H65 priority、K256/K384/K512 和等逐视频成本合同做 cap sweep、档位搜索或 utility predictor 训练；
* broader DUCA、coverage 或其他动态计算问题保持未决，但必须另行重新定义科学问题，不能由 Codex自动选择。

### 根因分类规则

* **单项误排序为主：** 所有最小合法新增转移都不能同时改善 capped 的两项主要指标；
* **窗口交互为主：** 至少两个最小转移单独都改善两项主要指标，但联合后至少一项收益发生符号反转或低于两者中的较好单项；
* **混合原因：** 其余模式。

这只是确定性机制诊断，不是统计总体结论。

### 禁止项

本任务禁止：

* 任何 detector、Scout 或 utility predictor 训练；
* detector/Scout forward；
* official validation/test；
* bootstrap；
* 修改 H65 priority sequence；
* 改变预算档位、split、checkpoint、annotation、类别映射、NMS 或 evaluator；
* 根据 96 个状态事后设计新的阈值；
* 扩展到任意全空间搜索；
* 新增视频级网络、图模型、集合模型或可学习交互项；
* 把最佳 96-state 点写成可部署或可发表性能。

之所以不运行 bootstrap，是因为最佳状态将在同一 40-video development holdout 的 96 个候选中被选择，选择后的区间不能被冒充为预登记确认性统计。

### Builder → Critic → Evaluator

**Builder**

* 在上述两个文件内完成最小实现和聚焦测试；
* 证明状态枚举唯一、完整、确定性且逐视频成本严格守恒；
* 截止：`2026-08-31T10:30:00+08:00`。

**独立 Critic**

只审查：

* 96-state 枚举是否完整；
* `video_validation_0000419` 是否避免任意配对；
* 是否只改变 12 个差分窗口；
* 是否保持逐视频实际成本、NMS、评估器与数据边界；
* 是否存在任何 forward、训练或 official-test 路径；
* 是否把同集 metric oracle 错写成论文证据。

通过即停止，不审查代码风格或扩展性。

截止：`2026-08-31T12:00:00+08:00`。

**独立 Evaluator**

* 在 Critic 通过的唯一干净提交上运行一个 CPU Slurm 作业；
* 先复现 Fixed/capped/released，再运行全部 96 个状态；
* 返回完整状态表、最佳点、单转移效应、交互残差和上述冻结门的机械判定；
* 结果根使用
  `/data/run01/sczc063/yuzibo/duca_marginal_cap_release_neighborhood_<implementation-short-sha>_20260831`；
* 最终截止：`2026-08-31T18:00:00+08:00`。

唯一客观 blocker 是任一密封输入缺失或 SHA 不一致。出现该 blocker 时停止，不重新生成 producer、不重跑 detector，也不建立第二条工程修复链。

---

## 6. 论文可写入边界

### 当前可以准确记录

本结果可以作为**开发阶段负结果、机制淘汰证据或相关论文的附录消融**：

> 在冻结 H65、同一 40-video training-side holdout 和严格相同的逐视频实际 observation 总成本下，基于真实逐窗口反事实检测损失的 K256/K384/K512 重分配在 50% 改变上限时相对固定 K384 提高 Avg-mAP/mAP@0.7 `+0.726/+0.729` 个百分点；解除上限后增益下降为 `+0.427/+0.450` 个百分点，并比 capped allocation 低 `0.298/0.279` 个百分点。该诊断否定了“扩大当前加性窗口损失分配的自由度会单调改善最终 TAD mAP”的假设，因此该 allocator 被停止。

若后续论文采用视频级联合效用机制，这一结果可以作为其问题动机；否则只作为内部路线淘汰记录。

### 当前仍不可声称

不得声称：

* DUCA-Marginal-v1 是有效方法；
* capped allocation 的增益具有统计显著性；
* released allocation 证明动态预算无效；
* 当前结果来自 official validation/test；
* 当前结果能推广到其他 seed、数据集、检测器或预算档位；
* learned allocator 能达到任何 oracle 数值；
* 已证明窗口交互是唯一因果来源；
* 已获得性能—成本 Pareto 改善；
* coverage、物理时间或整个 DUCA 方向已被否定。

**本轮仅授权上述一项联合 mAP 穷举诊断。当前 Marginal-v1 机制已达到终态，不再允许通过训练 predictor、调整 cap 或增加预算档位进行恢复。**

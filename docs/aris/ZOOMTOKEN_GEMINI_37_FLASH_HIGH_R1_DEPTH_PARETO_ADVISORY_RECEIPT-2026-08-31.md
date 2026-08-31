# ZoomToken R1 depth/sparsity Pareto：Gemini 3.7 Flash High 独立咨询回执

## 1. 调用身份

- 时间：2026-08-31T16:03:31+08:00 至 2026-08-31T16:05:18+08:00。
- CLI：`agy 1.1.22`。
- 模型参数：`--model gemini-3.7-flash-high`；运行日志解析为浏览器标签 `Gemini 3.7 Flash (High)`。
- 推理参数：`--effort high --mode plan`。
- conversation：`cebd7927-ed62-45f3-8418-f594a7908c96`。
- 终态：`SUCCESS`。
- 性质：独立咨询审查，不是 ZoomToken Project Pro 裁决，不更改冻结任务或实验。

本轮使用的完整请求保存在
`.cvpr-pro-lab/reviews/GEMINI_37_FLASH_HIGH_R1_DEPTH_PARETO_REVIEW_REQUEST-v001.md`，
CLI 运行日志保存在
`.cvpr-pro-lab/reviews/GEMINI_37_FLASH_HIGH_R1_DEPTH_PARETO_AGY-v004.log`；
FastCtx 原始 JSON 回答保存在后台作业 `j-dqizrn` 的
`C:/Users/skywalker/.fastctx/jobs/j-dqizrn/output.log`。

为无头模式临时加入的 `read_file(*)` 权限仅覆盖本轮本地只读审查，回答回取后已从
`C:/Users/skywalker/.gemini/antigravity-cli/settings.json` 删除并恢复原设置。本轮没有操作浏览器、远端、Slurm、Project Source、仓库代码或运行中结果。

## 2. 审查对象

- 当前冻结任务：`ZOOMTOKEN-R1-DEPTH-SPARSITY-READONLY-FOUR-ARM-FULLSTACK-PARETO-CLOSURE-v001`。
- GitHub：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-r1-depth-pareto-v001>
- candidate：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b82441c1aa2663069033d394794298d5c723bbb6>
- formal job：`1262120`，只读成本回放，四臂、16 pass、完整 THUMOS14 validation population（211 videos / 792 ordered windows）。

## 3. Gemini 总评

Gemini 给出的咨询结论是 `SOUND_WITH_LIMITS`：当前四臂 Williams-balanced matched full-stack 回放在限定范围内科学问题清晰、执行身份充分、估计量和拒绝规则明确，可以回答固定深度/固定稀疏度候选在当前单卡全栈实现上是否具有系统 Pareto 价值；它不能回答多种子泛化、official-test 准确率或动态机制优劣。

本结论是在没有读取任何 live/partial 指标的前提下给出的，因此只审查协议和实现，不能预告 job `1262120` 的终态分类。

## 4. 主代理抽查后可采纳的实现结论

1. `PROFILE_ORDER` 确为 `A B D C / B C A D / C D B A / D A C B`，每臂四次；完整人口常量为 211/792，warmup 为每 pass 50 windows（profiler 36-47）。
2. 每 pass 加载 `epoch_59` 的 `state_dict_ema`，重建模型并保持 loader identity/order 检查（profiler 893-902、945-952）。
3. 主估计量确为每臂四个完整 pass 的中位数，候选相对 A 的硬门为 p50 和 gross energy 均 `<=0.95`、allocated/reserved peak memory 均 `<=1.05`（profiler 1500-1535、1578-1611）。
4. 功率侧车只采集 `power.draw`；温度和 clocks 未采集。这不使当前门禁无效，但限制了对热节流原因的归因（profiler 1242-1250）。
5. loader 使用 `num_workers=0`，因此结果代表冻结的严格串行 full-stack 通路，而不是多 worker/prefetch 的吞吐上限（profiler 889）。这应作为披露边界，不是当前作业缺陷。
6. Final video Soft-NMS 在 pass 末执行并按 792 windows 均摊回单行；pass 总成本与主裁决保持一致，但单个 row 的 NMS 数值不能解释成对应视频的独立实测（profiler 996-1006）。
7. 历史 reported-2dp accuracy 只允许作为非阻塞兼容性诊断；本轮不生成新的准确率主张。

## 5. 仅作建议、不得升级为路线裁决的内容

Gemini 建议在固定削减路线终态后考虑内容感知的动态 tubelet 预算，并建议论文主张最终使用完整训练集、多种子和冻结 official-test opening。前者是未经 Project Pro 裁决的新提案，其中 `K_t=0`/Masked-Zero 的描述还可能与当前项目要求的稠密时间网格和 AdaTAD 耦合发生冲突，不能直接实现；后者与仓库的 Formal Dataset Completeness 规则方向一致，但具体 seed 数和 test-opening 协议仍须由后续冻结任务决定。

因此本轮不修改 `research-wiki/query_pack.md`、`research-wiki/anti_repetition.md` 或当前实验门禁，不创建新候选，不提交新实验。job `1262120` 终态证据摄取后，仍按既有流程交由 fresh exact-Project Pro 独立裁决。

## 6. 本轮失败尝试说明

前三次 `agy` headless 尝试均在本地 `read_file` 权限请求处返回空回答，没有产生模型审查、没有修改项目状态，也没有消费任何科研提交。第四次在显式只读许可下成功完成；正式咨询结果仅计一次。

---
updated: 2026-09-08
status: ready-prompt-not-consulted
scope: 后续 Pro 研究实现审查 Prompt
out_of_scope: 自动外发、已获得 Pro 意见、用审查替代实验
---

# 定位价值与 MoD＋TIA 的严格审查 Prompt

请独立、严厉且以证据为基础地审查 GeoSparse-TAD 的研究定位与实现差距。不要为了严厉制造错误，也不要因缺最终结果就断言路线无效。

先读 docs/README.md、docs/project/paper-plan.md、docs/methods/localization-value-routing.md、docs/evaluation/mod-tia-comparison-plan.md、docs/evaluation/mod-tia-proposals.json 和 docs/evaluation/localization-value-figures.md。实际模型 M 固定为 b70ae056c495b43ca3f305fe438926b97b2723b5；审查分支后续文档提交不是新训练模型。对照实际 geosparse_ext 和官方源代码，不能把旧 340a3541 当作当前实现。

研究已明确接受 MoD＋完整原生 TIA；A 不声称首先发明 dense state、gather/scatter、identity skip 或 dense-light/sparse-heavy。请检验以下问题：

1. 对 MoD、CoDA、VideoLLM-MoD、ASL、AdaTAD 与动态特征选择的归因是否准确？“定位价值学习”相对已有任务梯度/反事实获取是否仍只是换名？指出最接近的原始论文与具体机制，不用无来源的相似性印象。
2. 当前 A 的一次计划、全窗口 TIA、稀疏 Heavy 与 signed acquisition 实际做了什么？哪些拟议内容尚无代码？尤其不能把 loss 自带定位项说成已实现 interval risk/swap。
3. swap 是否以 donor、candidate、集合及作用层为条件？同 parent 等有效 token 是否保持被计数 Heavy MAC？跨 parent、逐层重新路由和 padding 是否改变成本？相同 MAC 是否被误说成相同延迟？
4. 新风险如何归约到实例、背景、空 GT 和漏检？M2/M3/A2 是否得到完全相同目标？A1→A2 混合改动是否足以归因，缺哪一个最小控制？
5. M0 是否保留原风格梯度及 alternating dense/routed 机会？统一缩放时，硬 Top-K Router 是否仍有可训练梯度？不能以删除 MoD 的 TIA、梯度或有利布局制造弱基线。
6. A 跨层动作与 MoD 单层动作的标签、后续重路由、训练额外成本是否可解释？配对 forward 的状态、loss normalizer 与随机性是否足以支持差分？不得把简单重置 RNG 直接当作异形张量的逐 token 随机掩码一致。
7. 四张图的数据生产器是否真的存在？逐位置损失之和能否重建标量风险？是否用 attention 冒充计算收益，或把窗口 loss 冒充窗口 mAP？
8. 哪些已有 A1、B/C、official/dense 与补充训练能直接复用？哪些改变训练目标后确实需要新实验？不能要求无差别重训，也不能把旧权重改名为 A2。

用户协议是全200训练、全211测试/792窗口、768×160、seed0、60轮；Geo 每5轮全测试 EMA best 已明确授权。保留此规则并披露选择偏差，另报固定60和共同50/60节点；不要擅自恢复holdout。官方原版、released权重复测、统一dense control 分开。A/B/C主方法动态与固定.5、两服务器并行优先，实验不设成绩门槛，不重复既有任务。

请给出：已确认正确处；实际错误与file:line/可达条件；尚未冻结的关键方法选择；区分机制所需的最小对照；每种可能结果对应的允许主张。对于实施建议说明改变哪些代码/目标、能复用什么、需什么验证，不以继续堆模块作为默认答案。

本 Prompt 已准备但未发送。若触发用户已授权的性能讨论条件，操作员必须通过 Computer Use 使用 ixBrowser 中实际的 ChatGPT Pro，记录材料和结论；不得静默改用 API 或其他浏览器，也不得称现在已经咨询。

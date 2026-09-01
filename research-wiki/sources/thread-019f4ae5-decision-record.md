---
type: source_record
thread_id: 019f4ae5-93dd-7381-8203-42360125b41b
title: "ChronoTransport discussion and decision record"
updated: 2026-07-11
---

# ChronoTransport discussion and decision record

本页保存当前任务中对路线有约束力的讨论。完整逐项消息仍由 Codex task ID 保留；
这里不复制工具输出、服务器日志或长监控过程，只记录用户问题、纠正和形成的正式决策。

## 1. 外部 ResearchClaw 审查

用户要求完整记录并吸收固定 commit 的外部方法论审查。审查裁决是：保留 C3/DUCA
作为 baseline 与诊断工具，把研究问题从“选哪些帧”转向“哪些计算值得重新执行”，
并要求 problem-first、claim-evidence-falsifier、完整成本和失败 pivot。

## 2. 为什么提出 ChronoTransport

用户追问外部报告的建议及是否认可。形成的判断是：pre-backbone frame dropping 存在
scout 成本、selected-axis 几何、高 tIoU/短动作风险；新路线保留 dense 时间轴，在
backbone 内对时间与深度分配重算、传递或复用。但该路线当时只是可证伪假设，不是
已经成立的最终方法。

## 3. 直白定义

用户要求直白报告。固定表述为：

> 旧路线研究“少看哪些帧”；ChronoTransport 研究“哪些昂贵计算必须重新做，哪些
> 可以沿用”，同时每个 detector 时间点仍保留输出。

## 4. 与 MoD 的关系

用户先后询问“这是否是 MoD”“与原始 MoD 有何区别”。形成的边界是：

- C3 的 `p_action → Δp_action → 边界导向选帧` 不是 MoD。
- ChronoTransport 是 MoD-like，但原始 MoD 是 token×layer 的 `COMPUTE/SKIP` 和
  固定 top-k 容量；ChronoTransport 是有状态的时间×层 `RECOMPUTE/TRANSPORT/HOLD`。
- 如果最终只剩 token 打分、top-k 计算和 residual bypass，就会退化为 Video MoD，
  创新不足。

## 5. C3/pre-backbone 路线的准确语义

用户明确纠正：当前 pre-backbone 路线不是选择动作帧，而是用动作/背景二分类产生
连续 `p_action`，通过其变化间接定位状态转换和潜在边界。这个表述更合理；actionness
只是构造边界证据的工具，不能再写成 action-frame top-k。

## 6. “重算”的含义

用户询问重算是什么、为什么需要。固定定义为：对当前新时刻/clip 的真实 group input
执行 VideoMAE heavy attention/MLP，得到 fresh state。重算不是重新训练，也不是重复
跑整段视频；它用于建立 anchor、阻止 HOLD/TRANSPORT 漂移，并在 cache invalid、
age/OOD/nonfinite/risk fail 时 fail closed。

## 7. 16 帧、帧、tubelet 与 token 粒度

用户质疑固定 16 帧×4 层是否过于死板。结论是：

- 768 帧被 AdaTAD 包装为 48 个 16-frame clips；16 帧是执行容器。
- VideoMAE `tubelet_size=2`，内部有 384 个时间 tubelets；2 帧 tubelet 是更自然的
  原生时间单位。
- v1 为控制真实 GPU 开销，最终合同仍采用 `48 chunks × 3 layer groups`；默认
  `[0:4]/[4:8]/[8:12]` 可配置，不是理论限制。
- tubelet/token 细粒度 routing 属于后续消融，不能冒充 v1 已实现能力。

## 8. 旧 p_action 污染

用户发现解释中出现 `p_action/Δp_action` 并追问原因。确认这是路线污染：这些信号只
属于 C3/DUCA baseline，不得进入 ChronoTransport 主路径。ChronoTransport 只允许
deploy-visible input/group energy、cheap change、proxy drift、cache age、position、
group identity、OOD/finite/cache-validity 等信号。

## 9. 在线任务误判

用户先询问是否在线，随后明确“没有在线需求”。之前把 cache/stateful execution 与
online TAD 混淆是错误任务漂移。ChronoTransport 的正式语义是离线全窗口 TAD；
scheduler 可以观察完整部署可见窗口，每窗口重置 cache。ACTAL/CFPA 等 causal
streaming idea 被标记为 out-of-scope。

## 10. 复核校正版规格

用户提供 TDD 包和“有条件确认”复核。吸收后的关键合同：外部 768 detector grid，
内部 384 tubelet grid；dense patch embedding、AdaTAD temporal adapter、head、NMS；
只动态执行 heavy attention/MLP、transport 和 scheduler；TRANSPORT 从 latest cache
链式递推；无 measured cost、calibrated risk 或专用 checkpoint 时回退 dense；GT、
teacher、raw prediction cache、counterfactual ledger 禁止参与推理。

## 11. 远端实现与验证

用户要求所有行为验证推迟远端，并指定读取 RTK、连接 N16R4、使用物理 GPU1。
ChronoTransport P0–P4 工程合同、Stage-A smoke、paired replay 和 Stage-B 单步训练完成；
这些只证明工程可运行，不证明科学有效。

## 12. 正式 Stage-B seed 3407

用户两次明确要求完整 fit/calibration/evaluation 闭环并部署首个单 seed。commit
`92029ea` 完成 140/30/30 split、140 fit steps、各 180 条 calibration/evaluation ledger、
真实 EMA、恢复与严格重载。P3 总 gate=`FAIL`：risk-regret Spearman=`-0.1914`；
feature improvement CI 跨 0；TRANSPORT 相对 HOLD 只有微弱 detector-regret 正信号。
Stage C/P5 因而锁定。

## 13. Research-wiki 与防遗忘要求

用户指出反复遗忘会导致原地打转，要求完整记录讨论、idea、否定和选择理由、当前方向
与最终目标。当前 wiki 被确认为单一研究记忆；failed ideas、用户纠正和负实验不得删除。

## 14. 2026-07-11 新颖性追问

用户询问当前路线是否新提出、是否存在更前沿方案。查新裁决：ChronoTransport 是
项目内 2026-07-10 新提出的具体路线，但不是文献中的全新范式，新颖性暂评 `4.5/10`。
强近邻包括 MoD、Eventful Transformers、ResidualViT、Progressive Block Drop、
Adaptive Temporal Refinement、SCOPE 和 Conformal Thinking。路线维持暂停，只允许
一次有界 P3 修复；首选新候选是 Boundary-Adaptive Temporal Multigrid。

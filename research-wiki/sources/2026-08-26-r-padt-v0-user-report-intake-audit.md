---
type: source_intake_audit
title: "R-PADT-v0 user-provided report intake and independent audit"
updated: 2026-08-26
status: discussed
external_artifact_status: designed
project_verdict: PARTIAL_ACCEPT_REVISE_BEFORE_G4
paper_claim_admissible: false
implementation_authority: none
experiment_authority: none
---

# R-PADT-v0 用户报告摄取与独立核验

## 1. 来源与证据身份

用户提供了两份内容高度同源的材料：

- 完整下载报告：`E:/下载/ZoomToken_开放式调研与科学裁决报告_R-PADT-v0.md`；
- 对话粘贴文本：
  `C:/Users/skywalker/.codex/attachments/c7fc56b9-8e69-4459-a729-95494dce258f/pasted-text.txt`。

两者共同给出 `CONTINUE`、固定代码 `bffff43dad28ca1042602ad3a01ba2990b953c13`、
隔离的 A-MoD 参考 `a41714e9f9271906a2eb4505e3fedc590c838055`、候选
`R-PADT-v0` 和 Cell 0/1/2。完整报告明确不授权代码、训练或提交作业，且
`paper_claim_admissible=false`。

这些科学内容目前只能登记为**用户提供的外部候选报告**，不能登记为已验收的同会话 Pro
裁决。对应正式请求确实绑定 exact Project、nonce 和上述 revisions，但项目内终态回执仍是
`TERMINAL_INCOMPLETE_NO_SCIENTIFIC_DECISION`：浏览器/Oracle 只摄取到一句开场说明，明确禁止
将其视为 Pro 许可、路线裁决、实现规格或实验授权。另有两项来源矛盾：正式请求含 nonce
`ZOOMTOKEN-OPEN-TEMPORAL-TOKEN-REUSE-v001-20260825`，下载版却写“未检出 nonce”，而粘贴版
又声称下载版包含 nonce；下载版列出的附件 bundle 当前也没有随材料提供，无法核验。

因此本页吸收其科学建议，但保留来源隔离：`external_artifact_status=designed`，项目内仅为
`discussed / PARTIAL_ACCEPT_REVISE_BEFORE_G4`。

## 2. 报告实际提出的机制

报告不是让后一帧直接复用前一帧的完整 VideoMAE 表征。其真实机制是：

1. 前两个 VideoMAE block 对 K64 全部 token 做稠密计算；
2. 每四个 tubelet 设一个锚点，锚点保留全部 K64；
3. 非锚点依据前缀表示与锚点同格点表示的 cosine delta，仅保留 top-16，并加入四个
   2×2 分区均值摘要；
4. 在更短序列上执行其余后缀 block；
5. 进入 neck 前，把被选 token 散射回原位置，把未选位置直接复制为锚点同格点的后缀输出。

所以更准确的名称是 **prefix-conditioned suffix token compression with anchor-based
approximate restoration**（前缀条件化的后缀 token 压缩与锚点近似恢复）。它不是 LLM 式
KV cache，不是精确跨帧复用，也不是前一帧完整 hidden state 的直接复用。

## 3. 固定代码核验

对 revision `bffff43...` 的只读核验确认：

- VideoMAE-S 为 12 个 block、D=384、patch size 16，12 层均接有 Adapter；
- 官方输入为 768 帧，tubelet size 2，因此 T=384；160×160 输入形成 10×10 原生空间网格；
- dense 路径为 K100，严格 R1 为完整 8×8、K64；
- native 排列是 tubelet-major、空间 row-major，并已携带 tubelet/spatial lineage；
- 现有 packed 路径有 gather/scatter，能够作为实现原语，但不能自动证明 R-PADT 的语义正确。

严格 R1 的 8×8 框也不是跨 tubelet 固定不动。选择器接收 `[B,T,4]` geometry logits，并对每个
`[样本,tubelet]` 独立地从 10×10 网格上的九个合法 8×8 block 中选一个。因此代码能审计绝对
patch ID，但相邻 tubelet 的两个 K64 集合不保证相同；只有交集位置才能按绝对 patch ID 直接
对应。报告要求所有非锚点 K64 与锚点建立一一格点映射且不允许补充对齐，原样路线因而可能在
自己的 Builder 第零门直接 STOP。

关键限制是：当前 Adapter 使用时序卷积并要求完整时空格点；普通 packed block 会先 scatter 回
dense carrier 再运行 Adapter。故报告示意的后缀 token 减半没有计入每层 dense Adapter，不能
直接解释为主干或端到端节省。若未来实现，只能复用已经保留 coordinate lineage 的 packed/ragged
Adapter 路径，并实测完整成本；不得假设 Adapter 随短序列自然缩短。

即使是支持交集中的相同空间索引，也只证明相同网格坐标，不证明相邻 tubelet 中是同一物体或
同一语义内容。报告的直接锚点复制仍可能在运动、遮挡和动作边界处陈旧。

## 4. 相关工作核验

报告关于 Eventful、ToMe、STTS、扩散模型缓存和 MoD 的大方向基本正确，但引用与覆盖范围不合格：

- [Eventful Transformers](https://arxiv.org/abs/2308.13494) 是直接的时间变化检测、reference/
  buffer 与增量 Transformer 近邻；
- [STA](https://openaccess.thecvf.com/content/ICCV2023/html/Ding_Prune_Spatio-temporal_Tokens_by_Semantic-aware_Temporal_Accumulation_ICCV_2023_paper.html)
  利用连续帧 token 相似性累积时间冗余并剪除视频 Transformer token，是报告遗漏的最近邻；
- [PVC](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_PVC_Progressive_Visual_Token_Compression_for_Unified_Image_and_Video_CVPR_2025_paper.html)
  逐帧补充此前未编码的信息并进行渐进视觉 token 压缩，也是必须讨论的跨帧压缩先例；
- [ToMe](https://openreview.net/forum?id=JroZRaRw7Eu) 已覆盖无需重新训练的通用 token merge 与视频
  加速；
- DOI `10.1145/3633781` 实际对应视频动作识别的 Spatial-temporal Token Merger，而不是粘贴位置
  所暗示的扩散缓存工作；`VideoZip` 未在本轮找到可唯一核验的正式论文/作者实现，应删除或澄清。

粘贴版引用 `[1]–[6]` 多数与相邻正文不匹配，不能作为论文引用表。R-PADT 可辩护的新颖性最多是：
**离线 TAD + 严格 ROI 格点 provenance + 少量 dense prefix 后的 VideoMAE suffix 压缩 + 检测前
T×K 恢复 + 定位敏感的完整链路成本门**。不得宣称首次视频 token 复用、首次时间冗余加速、首次
锚点缓存、首次 token pruning/merge 或首次周期特征广播。

## 5. 科学核验与项目裁决

### 可以吸收

- 保留严格 K64、官方 detector/evaluator 和固定网格接口；
- 状态限定在单次 clip forward 内，不跨样本、batch、video 或 checkpoint；
- 先做 identity/parity 检查，再做压缩探针；
- 必须有同预算压缩反事实，并以完整 decode-to-NMS p50/p95、吞吐、显存与能耗裁决；
- 只作窄的任务/接口级主张，不能冒充时间复用原理创新。

### 不能原样接受

1. **问题错位。** 该机制是后缀深度压缩，不是用户目标中的“后一帧稳定 token 直接复用前一帧
   完整表征”。顶层 `CONTINUE` 只能理解为值得做结构探针，不能理解为目标已经闭合。
2. **锚点复制的时间身份不成立。** 前缀 cosine 相近不保证经过双向全局 self-attention 后的后缀
   输出相近；把锚点最终输出复制给当前 tubelet 会把当前时间语义替换为旧时间语义。
3. **全 K64 一一映射不成立。** R1 的 8×8 框按 tubelet 独立选择，两个框可能平移；绝对来源
   只能可靠映射交集，无法为全部 64 个位置自动构造报告要求的一一锚点绑定。
4. **摘要引入第二个机制。** Q=4 均值摘要混合多个位置，并在后缀 attention 中改变所有保留
   token；主格同时改变 delta 选择、锚点复制和 summary，失败时无法归因。
5. **训练分布偏移。** 既有 FULL64 checkpoint 从未见过压缩序列、摘要 token 和复制恢复。无训练
   评估适合作为最便宜的结构探针，但失败不能单独证明该机制经适配后无效。
6. **成本门未被历史预注册。** 报告的 20% analytical FLOPs、15% p50 和质量阈值是新建议，
   不是项目既有合同。示例 `N'=248/512` 也不是当前完整 T=384 运行的成本收据。
7. **Cell 2 只匹配 N' 不足以称公平 ToMe。** 必须同时披露 merge/score、位置处理、provenance、
   restore、显存访问与完整链路成本；否则只能叫“同 token 预算压缩反事实”。

### 最小修订边界

- 把候选明确改称“前缀条件化后缀压缩与近似恢复”，不再称完整特征复用或 KV 复用；
- 第零门写实 T=384、K=64、D=384、12 blocks、absolute-position 处理、Adapter 位置与 neck 接口；
- 首个主探针先取消 summary（Q=0），避免把 anchor/delta 与摘要混成一个干预；
- 在接受任何直接锚点复制前，必须证明当前 tubelet 的时间/位置身份得到保留；否则该恢复规则需重新
  科学裁决，不能由 Builder自行补一个变换器；
- 同预算 merge 仅作压缩反事实，不作为已经公平的 ToMe reproduction；
- 推理期探针、后续训练和真实成本严格分层；不得用 token 数或理论 FLOPs 写成速度/能耗收益。

## 6. 当前状态与下一合法动作

项目结论是 **不完全认可，部分吸收，原样路线需修订**。R-PADT-v0 保持 `discussed`，没有升级为
当前主路线或 project-designed 合同；`L_p=2/R=4/m=16/Q=4` 也没有被项目冻结。当前没有 Builder、
Critic、Evaluator、PRE_RUN、GPU、Slurm 或训练授权，没有新准确率或效率证据。

若用户决定继续，下一步不是立即写完三格或启动训练，而是先形成一份经过上述更正的最小机制
规格：澄清它究竟验证“后缀压缩”还是“完整前帧特征复用”，并关闭跨 tubelet K64 映射、当前
时间身份、Adapter 成本和 STA/PVC 最近邻四项 P0。只有用户再次明确接受该修订规格，才进入
最小 Builder 周期。

# DUCA PJST 最终机制与因果合同裁决

Nonce: `DUCA-PJST-DERIVATIVE-CAUSAL-FREEZE-v002-20260825`

Exact Project: `g-p-6a796fef9a00819194024cf1de3bd697`（DUCA）  
GitHub repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`  
Frozen code/evidence revision: `b2ccfccab5b4912b59954afcc9b0364955327f7c`

你是本轮 Scientific First-Author Agent 和最严厉的独立审稿人。请直接核验公开仓库的冻结 revision，
给出唯一 `CONTINUE / REVISE / PIVOT / STOP`。不能把机制或实验选择退回给人类或 Coordinator。

## 已冻结背景

DUCA/H65 使用低成本 ASFormer scout 学习动作性与边界语义，再通过确定性 transport 间接选出有序、
非均匀的真实 RGB 帧；小模型不直接学习帧索引。本轮固定 `K=384` 只用于表示归因，未来 dynamic outer-K
仍是论文主线。不得改变 selector/ASFormer、VideoMAE-S/Adapter、ActionFormer、loss、NMS、THUMOS14
官方 split/evaluator、seed 3407 或 H65 的 30+60 训练合同。

代码核验已确认：`vit_adapter.py:889` 在任何真实物理时间残差进入第 0 个 Transformer block 之前执行
temporal-kernel=2 Conv3D PatchEmbed。因此，相邻 selected rank 被当作等间隔帧进行首次重型混合；
post-PatchEmbed SingleClock 和 pre-NMS physical decode 均不能撤销这次混合。

历史 H65 terminal-EMA `65.1257` 是单 seed、30+60 的诊断锚点。SingleClock 终结作业为
`TIMEOUT / EVIDENCE_ADMISSION_BLOCKED`，没有合法 PASS/KILL；不得重训 SingleClock。RankPack/TrueTime 的
单 seed `+0.6208 Avg` 是相关机制信号，不是 PJST 证据。当前没有 PJST 实现、PRE_RUN、训练或结果。

此前 PJST 浏览器调用因 authoritative terminal metadata 落入其他 Project 而被隔离；用户提供的完整可见
回复只能作为独立审查，不能授权实施。本轮是新的 exact-DUCA 科学冻结请求。

## 必须裁决的机制分叉

### A. 原 support-aware PJST

```text
m = (q_i*x_i + q_{i+1}*x_{i+1}) / (q_i + q_{i+1})
v = (canonical_delta / actual_delta) * (x_{i+1} - x_i) / 2
z = (W^- + W^+) * m + (W^+ - W^-) * v + b
```

它同时改变零阶外观 quadrature 与一阶变化率。support/Voronoi 在全局 K384 上形成，再切为 24×16。

### B. derivative-only PJST（Coordinator 核验后的最小提案）

```text
m = (x_i + x_{i+1}) / 2
v = (canonical_delta / actual_delta) * (x_{i+1} - x_i) / 2
z = (W^- + W^+) * m + (W^+ - W^-) * v + b
```

它保持原零阶外观平均，只校正真实物理间隔下的一阶变化率；support 仅作审计元数据，不进入 forward。
其目的不是声称完整连续时间建模，而是最小地证伪“首次 tubelet mixing 的错误时间尺度造成损失”。

两者都必须：零新增参数/optimizer group；canonical uniform 在任何浮点时间计算前直接旁路原 PatchEmbed；
保持 selected RGB、rank、K384、heavy token 数；有效位置严格递增且无 duplicate；physical decode 在
filtering/top-k/IoU/NMS 前恰好一次；不允许 post-processing 二次映射。

请在 A、B 或一个严格更小且不改变 H65 selector/detector 的替代中冻结唯一机制。若 support-weighted
zero-order term 不能由当前因果问题唯一推出，应明确删除或后置为独立消融，而不是为了复杂度保留。

## 必须裁决的因果口径

完整 Stage-2 联合训练会让表示梯度改变 selector，不能同时声称“端到端训练”和“两个实验臂 selected RGB
完全相同”。请冻结唯一首个正式 falsifier：

1. `fixed/replayed-selector representation attribution`：OFF 与 PJST 从同一 Stage-1 起点出发，训练期间
   接收相同逐窗口 positions/RGB/mask/K384；若既有 OFF 不满足该合同，说明 matched frozen-selector OFF
   是否必须新增。只允许声称首次表示效应。
2. `end-to-end system total effect`：只新增 PJST ON，允许 selector 漂移；报告 Jaccard、边界覆盖、gap
   分布与 selector 位移。只允许声称完整系统总效应，不能声称同一 RGB。

选择能最便宜、最干净地回答首次混合因果问题的一项。不要把两个 estimand 合并进同一结论。

## 统计修正与实验边界

- paired whole-video bootstrap 必须重跑 pooled official evaluator，不能先算 per-video AP 再平均。
- 10,000 个样本的双侧 95% percentile interval 冻结为 2.5%/97.5% quantiles，并明确索引/插值规则；
  第 500/9500 个顺序统计量约为中心 90%，不得称为 95%。
- 单 seed 视频 bootstrap 不是训练 seed 稳健性。
- `+0.50 pp`、时延/显存 `1.02x`、short-action `-0.50 pp` 等门若保留，必须给出历史方差、最小可检测
  效应或资源预算依据；否则冻结更合适的决策规则。
- 不重复 dense/uniform/random、RankPack/TrueTime、SingleClock、H65 60-epoch compression、UVT、Fovea、
  Query/Bridge、continuous cliplet 或 dynamic-K 矩阵。
- 不用 subset、synthetic、本地 CPU 或早期 checkpoint 形成效能结论；不接触 held-out/test 反馈。

## 必须返回的终稿

第一行仅给出 `CONTINUE / REVISE / PIVOT / STOP`，随后冻结：

1. 唯一 PJST 公式、机制名、论文直觉、falsifiable prediction、anti-claim；
2. 唯一 causal estimand，以及干预、结果与中介变量；
3. 精确 tensor/shape/dtype/device、global-K384 pair metadata、padding/short-video/mixed-batch、identity、
   numerical-stability 与 exactly-once physical decode 合同；
4. 最小 Builder 修改面，优先只允许 `temporal_grid.py`、`backbone_wrapper.py`、`vit_adapter.py`、一份配置、
   focused tests 和既有 launcher；列明禁止修改面；
5. shape/layout、uniform byte identity、显式代数参考、constant-pair invariance、gap scaling、全局坐标、
   padding、K384、无新参数、有限梯度、production trace 与 pre-NMS 单次映射测试；
6. 同 checkpoint、同输入只读兼容检查能排除什么、不能证明什么、停止条件；
7. 唯一真实 THUMOS14/N16R4 实验：是否需要 matched OFF、60 epoch/6000 successful updates、每 5 epoch
   完整恢复 checkpoint、latest-3+milestone+final、terminal final/final-EMA、seed、成本、结果根；
8. Avg-mAP、mAP@0.6/@0.7、short/adjacent/gap strata、bootstrap 与预先冻结的通过/停止规则；
9. 最窄新颖性 claim 及 TDN、TAdaConv、Run-Length Tokenization、ToMe、TE-TAD 的 invalidator；
10. `next_owner / next_action / dependency / expected_return_at / single_recovery`。默认链为 clean Builder
    最小实现 → 独立 Critic → Evaluator PRE_RUN → 仅在准入后立即运行唯一正式实验。

不能声称 PJST 已实现、已 PRE_RUN、有效、提高 mAP、降低成本或 paper-ready。


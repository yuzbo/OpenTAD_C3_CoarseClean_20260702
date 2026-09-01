# DUCA 第二份独立审查的对照、核验与吸收（2026-08-20）

## 证据身份与结论边界

- 本记录比较两份外部审查文本。第一份为已核验的 Project 内 Pro 会话 `duca-project-query-review`，其转录及已核验吸收见 `DUCA_PRO_REVIEW_ABSORPTION-2026-08-20.md`。
- 第二份为用户提供的文本 `C:/Users/skywalker/.codex/attachments/cd13a5e4-b2db-4934-9d02-8f6a75b4decb/pasted-text.txt`，SHA-256 为 `fe19d0e776dc59d526f96f00ed62fad5609afd876518e4563820728ee58d475b`。未提供该文本的 session、模型或逐字原始转录，故只把它视为可复核的外部建议，而不把其身份或任一参数建议视为权威决定。
- 本记录进行了只读代码与既有实验记录核验；没有改动模型代码、访问数据、启动训练或产生新的性能/成本结论。

**总判断：`SUBSTANTIAL_ACCEPT / NOT_FULL_ACCEPT`。** 两份审查对诊断和方法边界高度同向；应吸收其共同的因果与实现约束，但不能把第二份给出的结构、训练日程、K 集合或数值门槛直接冻结为主路线。

## 两份审查一致且应保留的内核

1. DUCA 的主张应是：低成本 scout 预测动作性、起点和终点边界语义；确定性规则由这些预测产生位置价值；动态 K 只由冻结语义的独立聚合给出。小模型不应直接学习帧索引，固定 K 只能是公平对照或回退。
2. 当前 UVT/Fovea 包不应继续补丁化为论文主方法。UVT 把 V(t) 的位置排序、几何/EMA 监督和 K 证据混在一起；Fovea 的现有 arm 也不是单变量对照。
3. 必须先在同一可运行 runtime 内得到 dense、exact-uniform、seeded-random、actionness-only、actionness+boundary 的 fixed-K 因果矩阵，再开放 dynamic K。所有 arm 必须共享 detector、训练更新、NMS、评估器、数据划分、终止 checkpoint 规则和成本边界。
4. 非连续物理帧不能在未处理其真实时间间隔的情况下按 selected-rank 直接组成 VideoMAE temporal clip；NMS 前的 proposal 必须已回到原始物理时间。是否采用连续 cliplet 是待实现验证的候选，不是已取得的性能事实。
5. semantic scout 与 detector 的梯度和优化状态应隔离：GT 只用于训练期语义损失，部署选择不能读取 GT、teacher、原始 detector 预测缓存或通过 hard index 接受 detector 反向梯度。
6. 历史 `65.385724` 只说明曾存在值得重建的非均匀语义采样信号；它是 30+60 epoch、多组件课程，不能充当公平 official-60 或 selector 独立增益证据。

## 已由本地代码和记录核验的事实

| 审查陈述 | 核验 | 证据与限定 |
| --- | --- | --- |
| UVT 的 `off/geo/geo_ema` 不是单变量归因 | 成立 | `dynamic_B`、straight-through detector loss、256--512 dynamic budget 和五项辅助损失同时存在；K 仅用 actionness，而 `value_mode` 同时变更 alpha、geometry 与 EMA（`C:/Users/skywalker/.codex/worktrees/duca-uvt-official-v2/configs/adatad/thumos/duca_uvt_value_portal_n16r4.py:36-86`）。 |
| 当前 UVT 可能向 VideoMAE 输入伪连续时间 | 成立为**实现风险**，实际性能因果未证实 | selected frames 先 reshape 为连续 16-frame clips 并进入 backbone，physical positions 只在 backbone 后用于插值（`C:/Users/skywalker/.codex/worktrees/duca-uvt-official-v2/opentad/models/backbones/backbone_wrapper.py:177-212`）。这足以要求 runtime 对照，不能由静态阅读估算 mAP 损失。 |
| Fovea arm 不是严格单变量链，且所有 arm 均有动态预算 | 成立 | `query_gt_mask`、`query_cycle`、`query_fovea` 除分数外还分别改变 GT mask、cycle、quota/MMR；config 写为 `dynamic_budget=True`，selector 也硬编码该值（`C:/Users/skywalker/.codex/worktrees/duca-full-official/configs/adatad/thumos/duca_fovea_qb_thumos.py:22-52,71-101`; `.../fovea_query_bridge_selector.py:222-234`）。 |
| Fovea 当前损失没有独立边界/几何语义 | 成立 | endpoint=2 与 interior=1 最终均被 `(gt_mask>0.5)` 二值化；center 回归到零、width 均值被最小化；所谓 diversity 只有正号 entropy（`C:/Users/skywalker/.codex/worktrees/duca-full-official/opentad/models/losses/fovea_losses.py:43-47,87-112`）。因此现有 arm 不能检验“边界语义改善”。 |
| `64.352` 与 `65.696` 不能改称为间接非均匀采样收益 | 成立，但需更精确措辞 | 两者均为 uniform 历史锚点；前者是 native stride-2/adaptor ActionFormer，后者是改 detector/geometry 的 grid-aware ActionFormer，均与当前协议不匹配（`research-wiki/source_registry.md:430-437`; `research-wiki/query_pack.md:1378-1381`）。`65.696` 不应笼统称为普通 uniform control。 |
| 历史 `65.385724` 不是 matched official-60 | 成立 | 它是 30 epoch uniform full-detector 加 60 epoch learned/full detector，并混入 distillation、transport gradient 与 ASFormer adaptation（`research-wiki/ideas/duca-two-stage-curriculum.md:170-186`; `research-wiki/experiments/duca-rate25-sampling-rate-curriculum.md:17-31`）。 |

## 第二份审查需要纠正或保留条件的地方

1. **“没有新训练/当前仍只是旧 PRE_RUN 包”不准确。** `BLOCKED_PRE_RUN / CYCLE2_IMPLEMENTATION_PACKAGE_CLOSED` 是旧 clean cycle2 包 `d80022...` 的终态（`research-wiki/log.md:5792-5800`）。其后已经有真实、单 seed、60-epoch 的开发性 THUMOS14 训练：UVT Job `1244840` 与 Fovea Job `1244851`。它们缺同提交 dense/uniform/random、部分 arm 或完整成本，因而不是论文级 matched evidence；但不能写成“未训练”。详见 `DUCA_PRO_REVIEW_ABSORPTION-2026-08-20.md:23,33`。
2. **连续 16-frame cliplet 不是已验证的唯一修复。** 它合理地避免 temporal tubelet 跨不连续帧，却会改变 VideoMAE 的批处理、绝对位置编码、block 间建模和有效采样几何。只有实现 regular-grid parity、真实 timestamp metamorphic、pre-NMS trace、token/latency census 后，才能决定它是否比其他物理时间 adapter 更忠实。
3. **第二份提案中 exact K 的可行集合尚未自洽。** 它一处要求每个 canonical cell 选一个 cliplet，另一处没有冻结是否强制首末 block；第一份则显式强制首末 block。两者对 coverage、short-window 与 action 位于首尾时的行为不同，不能同时作为一个实现合同。
4. **边界标签尚未冻结。** 第一份建议起/终点 radius-8 的软标签，第二份给出单点 0/1 target。两种目标对短动作、类别不均衡与 cliplet 聚合不同，不能把任一写成“正确答案”；需要只用 FIT/CAL 的语义指标和预注册 fixed-K 矩阵选择。
5. **训练日程和 dynamic-K 时机相互冲突。** 第一份建议 1000 update uniform 后再 500 update 平滑释放，并在三 seed fixed-K/Query 归因后才做 dynamic K；第二份建议 1000 update 后直接切换，且在单 seed Stage 1 后即进入 dynamic K。前者更强于归因，后者更快但更容易把单 seed 波动带入 K 主张。两者的数值日程均是建议，不是当前决定。
6. **预算集合、均值和阈值不能直接合并。** 第一份使用 `{256,320,384}`、目标均值 320；第二份使用 `{256,320,384,448,512}`、目标均值 384，并给出不同的 mAP/高 IoU 门槛。它们不是文本小差异，而会改变成本匹配、统计功效和论文主张；必须由正式 protocol 单独预注册，不能从两份审查中拼接。
7. **一 seed paired bootstrap 不能替代多 seed。** 视频级 bootstrap 可以刻画同一 seed 内的视频采样不确定性，不能估计优化随机性。它可作为早期 kill gate，不能单独支持 dynamic-K 或 Query-Bridge 的论文结论。

## 当前吸收后的科研立场

接受的是**方向性约束**：semantic-indirect、物理时间原生、fixed-K-first、真实 variable heavy compute、完整恢复与成本核验。没有接受为事实或现成设计的是 `DUCA-SI-QB-vNext`/`DUCA-SQB-Block-DK-v1` 名称、Query 数与层数、16-frame 单元、标签半径、optimizer/课程、K 集合、动态阈值、bootstrap 次数、mAP 门槛或延迟百分比。

优先核验顺序保持不变：先完成同一 model factory 的 runnable dense/uniform/random runtime 与 official binding；再以 runtime hook 证明 selected clip 的连续性、实际 token 和 `executed_k`；随后验证语义标签/梯度所有权、split firewall、checkpoint interruption-resume 和 full-stack cost。fixed-K 的 actionness+boundary 必须先超过 uniform 与 random，才能把动态 K 作为论文核心的实证问题打开。

状态：`REVISE / designed_for_clean_reimplementation`。本记录不增加任何效能、成本或论文 claim。

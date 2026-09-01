# DUCA IPEC-K 第四份外部审查：对照与核验（2026-08-21）

## 来源与证据边界

- 本文记录用户提供的《DUCA 对抗性科学审查》副本（860 行）。它推荐 `PIVOT` 并提出
  `DUCA-IPEC-K`，但没有可复核的审查 session、header 或原始转录；故具体公式、模块、
  阈值和实验包均是**外部 proposal**，而不是已采纳的科学决定。
- 审查所称公开 GitHub 不能解析 Fovea `4ae50671` / `46c7142` 和 UVT `df544c78` /
  `59f27d59` 的边界成立。本地对象库仍保有前两个训练 commit；公开不可核验不等于本地
  代码或 Job 不存在。UVT/Fovea 的数值只可作为项目材料包中的真实、单 seed 开发事实，
  不能在本轮升级为独立外部代码审计证据。
- 本页只保存审查、历史实现与已有 receipt 的核验。没有新增实现、数据、训练、成本、性能
  或论文 claim。

## 第四份审查中得到复核的事实

1. `42dba3f` 的历史实现不是单一 non-uniform selector：它含 30-epoch uniform warm start、
   60-epoch joint course、`density_transport_st`、detector contribution/gradient bridge、
   uniform companion、full ASFormer adaptation 与 terminal-EMA 规则。`65.385724` 因而是
   多变量联合课程的历史正信号，不能写成 selector 单因素增益。
2. 该 selector 在 backbone 前 gather 原始非均匀 RGB；随后按 selected rank 把 384 帧划为
   24 个 16-frame VideoMAE clip。`BackboneWrapper` 不接收 timestamp、PTS、frame index 或
   frame interval；NMS 前 true-time 回映修正 proposal 几何，不能重建表征层的真实时间。
3. 因此历史的“非均匀选择”与“selected-rank 伪连续时间压缩”在 VideoMAE 输入内彼此耦合。
   不能把历史 65 直接解释为 non-uniform selection 收益，也不能仅凭静态代码断定伪连续
   时间造成了多少性能损失；它可能同时压缩背景、也可能扭曲边界运动。
4. 当前 UVT `1244840` 的 `off/geo/geo_ema` 和 Fovea `1244851` 的各 arm 是复合干预的
   单 seed 开发训练。UVT 的负向 bundle 与 Fovea `query_cycle` 的相对较高数值，都不能分别
   归因给 value、geometry、EMA、Query、cycle、teacher 或 dynamic K。
5. 逻辑 K、mask 或 metadata 不是实际 VideoMAE 工作量。尚无已运行的 backbone-near
   `executed_k` 收据；padding 到 Kmax、inactive cliplet 进入 backbone 或只减少输出 mask
   均不支持效率主张。scout/teacher 的训练与推理成本也必须单列。

## 四份审查的一致内核：条件接受

1. 主线保持为“低成本 semantic scout 的 action/start/end 预测 → 确定性物理采样 →
   per-video/window dynamic K”；fixed-K 仅用于对照、归因和回退。
2. Query、cycle 和 teacher 只能作为 semantic scout 的单变量机制，不能直接输出 index、K、
   utility、detector proposal 或读取 detector feedback；部署期不允许 teacher、GT 或 cache
   驱动采样。
3. 真实连续 physical cliplet、从 VideoMAE 调用点审计实际工作量、pre-NMS physical-time
   proposal path，以及同运行时的 fixed-K 与内容—预算错配对照，是任何主张的必要前提。
4. 当前没有 dynamic-K efficacy、Query efficacy、端到端加速或论文级性能结论；历史 65 和
   UVT/Fovea 只能保留为诊断材料。

## 第四份相对前三份的新增价值

- `S0/SQ/SQC/SQD` 把 Query residual、reverse semantic cycle 与 detached teacher 分开，
  比此前只用 `S0/SQ/SQD` 更能追查 Fovea `query_cycle` 的来源。它符合用户保留
  Query 前后协同与知识传递的要求，但四臂的语义 supervision、参数/FLOPs、更新数和数据顺序
  必须严格匹配。
- IPEC 把动态 K 假设具体化为“成对 start/end 后验覆盖所需的最小重型证据量”，比泛化的
  actionness/难度预算更直接指向短动作和高 IoU 边界。这是一个可证伪的候选科学机制，而非
  已证明的 endpoint posterior 或 IoU 保证。
- 它明确区分 heavy-only、model-path、raw-video end-to-end 三层成本，并要求 ragged cliplet
  的真实执行审计；这是对前三份共同问题的可操作收束。

## 与前三份不一致或仍未闭合的部分

| 维度 | RiskClip | BSC-DK | SCOPE-DK | IPEC-K（第四份） | 结论 |
| --- | --- | --- | --- | --- | --- |
| dynamic-K 命题 | 残余风险阈值的最小预算 | 固定总预算重分配 | 语义 coverage 的最小 J | 成对端点 coverage 的最小 M | 四者不能共用同一动态收益或成本 claim。 |
| TAD 训练合同 | 倾向冻结 detector 插件诊断 | 所有臂等更新 full training | 先 sealed 初始化，再训练 variable-support detector | P0 语义预训练后训练 S0/SQC 的 TAD 臂 | 实现前必须预注册一种合同，不能事后混用。 |
| physical 重建 | scatter/interpolate | masked-zero scatter | splat + support/evidence 通道 | physical-grid reconstruction | 不可假定等价；任何新增 support/evidence 通道都可能是 detector 输入混杂。 |
| physical-time 对照 | 未专设 | 未专设 | `GAPPACK` vs `CONTIG`、同一 RGB 集 | P1 只有 S0-F24 vs SQC-F24 | IPEC 当前遗漏了 SCOPE 最关键的“同一帧集、仅组织方式不同”连续性对照。 |
| dynamic 强控制 | K-shuffle | 缺 K-shuffle | 明确 K-shuffle | P2 未把 K-shuffle 列为独立 arm | IPEC 继续前必须补内容—预算错配控制，否则 K histogram 仍可解释差异。 |

## 对 IPEC-K 的判断

### 可保留为下一次设计输入的部分

1. 四臂 semantic P0（S0/SQ/SQC/SQD）和独立 action/start/end 监督，可以区分 Query、cycle
   与 teacher；其中 SQD 只应是单列成本的附录性 teacher ablation。
2. `GAPPACK`/`CONTIG` 应从 SCOPE 方案保留：固定完全相同的 source RGB frame set、K、
   detector 初始化和训练合同，只改变伪连续组织与真实连续 cliplet，才可测试 physical-time
   表示是否有高-IoU 贡献。
3. selector 应接收 detached semantic posterior，VideoMAE 只运行 ragged、真实连续的有效
   cliplet；在实际调用点断言无 Kmax padding、无 dummy/inactive cliplet、并记录 full cost。
4. dynamic 阶段需把每视频 K 的内容关联与 actionness/有效长度分开检验，并加入 K-shuffle；
   否则不能说明预算分配源自端点不确定性。

### 不能直接接受为科学事实或冻结设计的部分

1. IPEC 的 `q(s,e)` 是由 start/end/action 概率组合出的**启发式分数**；在没有联合校准和
   独立验证前，不能称为概率意义上的“成对端点 posterior”。它也隐含 start/end 条件独立，
   对多实例重叠动作可能失真。
2. `rho_tau` 的 IoU 近似仅在端点误差模型成立时才有解释力。cliplet 覆盖不是 detector
   端点误差上界，更不是 AP 或 IoU 的保证；它必须先由 train-only coverage—高-IoU 关联实证。
3. 48×16、`16≤M≤32`、平均 `M≈24`、greedy rule、P0/P1/P2 阈值、三 seed 与 `+0.5 pp`
   等均属 proposal，不能由历史 K=384 或审查文字自动确定。
4. P1 仅比较 S0-IPEC-F24 与 SQC-IPEC-F24，能检验 cycle/Query 的传递，不能检验 physical
   cliplet 相对 selected-rank runtime 的效应；必须保留上表的 `GAPPACK`/`CONTIG` 对照。
5. 若 SQC 失败，应停止 Query/cycle claim；若 SQD 失败，应停止 teacher claim。不能据此
   自动停止 S0 的语义间接 dynamic-K 核心。反之，若 IPEC dynamic K 未超过相同平均 K 的
   fixed-K 曲线并且 K-shuffle 不恶化，才是核心机制的强反证。
6. 其 cited prior-art 只提示新颖性风险，不是覆盖性检索；不得写成“首次”或已完成 novelty
   审核。

## 本次吸收后的事实边界

第四份与前三份共同加强了对当前 bundle 的否定性诊断，却没有产生新的有效性证据。项目不采用
`DUCA-IPEC-K` 名称、公式或训练阈值作为现行实现决定；它们与 RiskClip/BSC/SCOPE 的冲突需在
下一份可执行合同中先行消解。当前可保留的最小内核仍是“语义预测 → 确定性物理采样 → dynamic K”，
并附带四个不可省略的核验：真实连续 cliplet、backbone-near executed-K、同一帧集的物理时间对照、
内容—预算 K-shuffle 对照。

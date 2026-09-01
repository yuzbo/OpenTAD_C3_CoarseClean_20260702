# DUCA 对抗性 RiskClip 审查：记录与核验（2026-08-21）

## 来源与证据边界

- 输入是用户提供的《DUCA 对抗性科学审查》文本（731 行）；它给出 `PIVOT` 与
  `DUCA-RiskClip` 建议。该副本未附可复核的会话标识、原始 header 或完整会话转录，
  因而本页将它视为**外部审查建议**，而不是自动生效的科学裁决。
- 审查者可以从 GitHub 独立读取的历史父版本是
  `42dba3f90b37243e7965d18b6707e88e81bf7109`。本地 git 也确认
  `4ae5067100c4490c7110c00a1ad406230ba603cd` 与
  `df544c78ce515d925dc7019f106fce09a53c09f8` 是真实提交；但它们所属的
  2026-08-19 分支没有出现在 origin 的远端 refs。因此“该次 GitHub 审查无法逐行
  读取 UVT/Fovea”的边界成立，不能推论为“这些本地训练代码不存在”。
- 项目实际状态不是笼统的 `BLOCKED_PRE_RESULT`：UVT `1244840` 与 Fovea
  `1244851` 都已完成真实、单 seed 的 60-epoch 开发训练。它们仍没有形成同提交、
  可归因、可写入论文主表的比较证据；详见 `PAPER_PROGRESS.md`。

## 本地复核后接受的事实

1. 历史 `42dba3f` 的 `DucaOnlineFrameSelector` 在重型 backbone 前用
   `selected_positions` gather RGB；`TwoStageDetector` 用 selector 输出替换输入，
   所以 `65.385724` 是 pre-backbone 的非均匀 RGB 输入，而非 uniform 或 backbone
   后 token pruning。
2. 该版本在 `batched_nms` 前将 selected-axis segment 回映为真实物理时间；这保证
   几何 NMS 的坐标顺序，但不保证 VideoMAE 内部时间建模正确。
3. 已按物理位置排序但不连续的帧被按 selected rank 重排成 16-frame clip；VideoMAE
   没有获知真实间隔。因此“输出坐标回映正确”和“骨干内部存在伪连续时间”可同时成立。
4. `65.385724` 是 30-epoch uniform warm start 加 60-epoch learned-sampling 的课程，
   同时包含 `density_transport_st`、detector contribution/gradient bridge、full
   ASFormer adaptation 和 uniform companion exposure；它是非均匀输入值得追踪的历史
   信号，绝不是 selector 单因素的公平增益。
5. UVT 的负结果与 Fovea `query_cycle` 的较高开发值都只是复合干预的诊断。它们不能
   单独证明 V(t)、Query、cycle、teacher、位置或动态预算有效。

## 条件接受的方向性建议

### 语义间接性与动态预算

接受以下方法身份：低成本 scout 只预测 0/1 动作性、start/end 边界语义；确定性规则
从这些语义导出位置价值与逐视频/窗口动态 K。Query 只能产生三个语义 logit 的上下文
残差；teacher 只能在 FIT 产生 detached 语义软目标，不能参与部署、位置/K 决策、
detector feedback 或额外 detector 更新。fixed-K 继续只是归因和回退。

这是对现有用户科学约束的收缩与澄清，而不是对 Query 或知识传递有效性的经验确认。
`SQ` 相对 `S0`，以及 `SQD` 相对 `SQ` 必须继续是独立、预注册的因果比较。

### 连续物理 cliplet

接受“禁止把两个物理不连续区间拼入同一个 16-frame VideoMAE clip”作为待实现的运行时
合同候选。每个 cliplet 必须保留原始 frame index、timestamp、fps/timebase，并在任何
几何 IoU/NMS 前保持物理坐标。

但不接受它是已证实的免费修复：独立运行 16-frame cliplet、再 scatter/reconstruct 到
检测网格，会改变 VideoMAE 的跨 clip 时序上下文、位置编码分布和 detector 输入 mask。
它必须先以同一新 runtime 的 fixed-K physical-uniform 对照检验；不能拿旧的
dense/uniform/random receipt 直接填补这一差异。

### 不重复历史基线

接受不因本次审查重跑既有 official dense、strict uniform 或 seeded-random 训练。
这些 VC receipt 保持只读。若新 physical-cliplet runtime 与旧运行时不同构，旧数字只能
标作跨运行时诊断锚点。新 runtime 所必需的一次 physical-uniform fixed-K 控制不是
“重复旧基线”，但必须明确它只服务输入合同归因，不能伪装为新的 official baseline。

## 不作为既定决定接受的部分

1. `DUCA-RiskClip` 名称、48 个 cliplet、16-frame 单元、K 集合、风险公式、动态规划、
   阈值、每项百分点门槛和停线规则是审查者的候选设计，尚无实测或本地实现核验，不能
   直接冻结。
2. “VideoMAE、projection、head 和 detector 一律冻结”不能成为当前项目的普遍定理。
   对 P0 纯语义机制，冻结 detector 是合理隔离；对之后的新 physical runtime TAD
   比较，必须在实现前二选一并预注册：使用可绑定的冻结 released detector，或让所有
   新 runtime arms 接受完全相同的 full-training 更新数、优化器和终止 checkpoint。
   不能只冻结主臂或只训练某个臂。
3. 以 masked zero/scatter 取代未观察特征并不自动与 AdaTAD detector 兼容；它需要
   真实的 shape、mask、timestamp metamorphic、pre-NMS trace、无 padding 和
   `executed_k` 运行时测试。否则连续 cliplet 可能用丢失跨块上下文而非修复时间语义。
4. 对 AdaFrame、TE-TAD 和其他 prior art 的存在性引用说明新颖性风险真实，但本轮未能
   独立抓取其官方页面，也未完成覆盖性文献检索；不能据此宣布 novelty 已通过或已失败。
5. `PIVOT` 可以被理解为“停止原样延续 UVT/Fovea 的直接选择/复合训练合同”，但不应
   被理解为删除 UVT/Fovea。它们的 Query 上下文与 semantic-only knowledge transfer
   仍保留为严格隔离的候选机制；其单 seed 结果继续作为诊断性证据。

## 本次吸收后的唯一事实边界

当前只接受“语义预测 -> 确定性间接采样 -> dynamic K”这条科学主线，且 Query/KD
受限于语义 scout。未接受任何 RiskClip/BSC 的具体实现或超参数。没有新增代码、数据
访问、训练、性能、成本或论文 claim。下一次实现前必须先冻结可执行的 physical-time
runtime 合同和新 runtime 的最小、非重复归因控制；现有开发训练不能作为其性能依据。


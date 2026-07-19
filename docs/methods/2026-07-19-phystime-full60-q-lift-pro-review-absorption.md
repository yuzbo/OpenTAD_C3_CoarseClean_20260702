# PhysTime Full60 / Q-Lift Pro 审查吸收记录

日期：2026-07-19

## 1. 来源与冻结证据

- 外部审查原文：
  `docs/methods/reviews/2026-07-19-phystime-full60-q-lift-pro-review-raw.md`
- 原文 SHA256：
  `BBD48B6BCE5E4AC612A395561D2EABCBB1F6DB5880B329EF21CAC6808CFBD5E0`
- 被审训练代码：
  commit `0dc5851a8feb12b97d16bdb5ea8fc60e9273d132`
- 被审 Git tree：
  `bddc9b9386604d00d213275a47ce7997b35d3f4c`
- 已验证结果：
  selected-axis `41.28%`、physical-metric `57.57%` Avg-mAP，
  physical-metric 的 mAP@0.7 为 `28.64%`。
- 历史外部锚点：
  旧随机采样 ActionFormer `63.61%`，dense AdaTAD `68.29%`。

现有 `57.57%` 继续记为可信的
`full60-single-seed-supported` 证据，不因本轮审查被撤销。

## 2. 独立总裁决

**分级认可，不完全认可。**

外部审查的核心裁决正确：

1. 当前代码正确实现了检测头后段的物理时间度量，但没有让
   VideoMAE、TIA 或 ActionFormer projection 看见时间戳。
2. 同 commit 的 selected-axis 与 physical-metric 两臂是公平对照；
   与旧 `63.61%`、dense `68.29%` 不是严格公平的因果对照。
3. `K=384` 原始观测、`J=192` 原生 tubelet token、基础 query 数和
   多尺度候选总数必须分开登记。
4. 旧随机采样分支的 `J192 -> Q384` feature interpolation 同时改变
   feature、感受野、projection 状态、assignment 机会和候选数，不能被
   解释为中性的 query 数量提升。
5. 当前 physical-metric 从检测头 point construction 开始，确实覆盖
   秒域 GT、assignment、regression range、center sampling、decode、
   domain clamp、NMS 和 evaluator。
6. 当前滑窗路径在跨窗口 NMS 前把 segment 舍入到 `0.01s`；应改成
   内部全精度，最终序列化时再格式化。
7. “无 GT 采样”只能描述已接受窗口内部的固定不规则子采样；
   训练阶段标准 `random_trunc` crop 本身使用 GT 保证动作相交。
8. G1b 的负结果不能证明所有 support-query 解耦结构都无效，因为
   G1b 同时改变了过多结构。
9. 当前结果不是 paper-ready；仍缺机制隔离、多 seed、成本、扰动族和
   第二数据集。

## 3. 不照单全收的部分

### 3.1 交叉注意力不是已经证明的“唯一主方案”

稀疏 support 到 deterministic query 的 masked cross-attention 是目前
最值得优先验证的候选，但它还没有实现和实验。它增加了参数、上下文与
表示能力，也可能被审稿人视为 mTAN/DETR 式重网格化。当前只能登记为
`designed candidate`，不能称“唯一正确结构”。

下一轮必须保留一个更简单的共享 query-lift 辅助对照，例如共享
feature interpolation 或显式的非学习 query copy。该辅助对照不作主方法，
但用于判断收益来自物理坐标、query 数量、额外容量还是 cross-attention。

### 3.2 数值准入阈值是建议，不是统计事实

`+6pp`、`+4pp`、`+1.5pp`、pre-NMS recall `+5pp`、成本 `1.40x` 和显存
`1.35x` 等阈值没有由当前方差、功效分析或系统预算推导。它们可作为首轮
资源管理参考，但在取得 seed 方差和真实成本前不得写成科学结论或机械
终止规则。

正式 gate 应先预注册主要指标、最小有意义效应和成本预算；多 seed 后
同时报告 paired effect、视频级嵌套 bootstrap 与失败模式，而不是用
三个 seed 的区间下界机械决定路线生死。

### 3.3 时间戳反事实必须保持合法时间轴

不能直接随机打乱 timestamp，因为这会破坏严格单调、support domain 和
现有 fail-closed 合同。可接受的反事实包括：

- 把实际不规则时间轴均匀化；
- 打乱 gap 序列后重新累积为严格递增时间轴；
- 在保持端点、观测顺序和 gap 边际分布的条件下做置换。

反事实只能破坏物理 gap 信息，不能制造无效时间轴。

### 3.4 第二数据集不能未经论证直接固定

ActivityNet-v1.3 是候选，不是既定答案。应根据 raw-video 可获得性、
动作持续时间分布、边界密度、窗口协议和总算力，在
ActivityNet-v1.3、HACS、FineAction 等候选中做协议审计后决定。

### 3.5 新四臂不能直接复用旧两臂数值

只要加入新的 support-to-query bridge，即使 `Q=J`，其 feature state
也不再与当前 ActionFormer 完全相同。因此 A/B/C/D 四臂必须在同一新
commit 下全部重跑；旧 `41.28/57.57` 只能作为历史外部锚点。

## 4. 吸收后的研究决策

### P0：先关闭评估与命名问题

1. 跨窗口 NMS 内部保持全精度 segment 和 score，只在最终 JSON 展示层
   格式化。
2. 明确区分：
   `uniform-rank-seconds`、`physical-seconds` 和旧
   `selected-rank-remap`。
3. sampler 文案改为“已接受窗口内无 GT 固定不规则子采样”。
4. 补 K/J/Q、query provenance、参数与成本 artifact validator。

### P1：同一新架构做 2x2 因子实验

- A：规则 rank 秒轴，Q192；
- B：物理秒轴，Q192；
- C：规则 rank 秒轴，Q384；
- D：物理秒轴，Q384。

四臂必须共享 K384/J192 原始输入、support encoder、参数形状、训练数据、
seed、优化器、schedule、loss、decode、NMS 和 evaluator。Q384 的 query
是检测状态，不是新增 RGB、frame、tubelet 或 observation。

### P2：必须同时报告机制诊断

- 每个 GT 的 eligible/positive query 数与 zero-positive 比例；
- pre-NMS class-agnostic/class-aware recall；
- oracle score、oracle boundary 和固定 top-N replay；
- start/end 到最近真实观测的距离；
- support gap 分位、短动作和高 IoU 分层；
- query coverage、分类排序、边界 MAE；
- 训练/推理延时、峰值显存与候选总数；
- 合法的 timestamp uniformization / gap counterfactual。

### P3：升级和停止原则

- 若物理轴在 Q192 与 Q384 都稳定胜规则轴，才支持 physical mechanism。
- 若 Q384 的增益在两种轴上近似相同，降级为通用 query-density 工程。
- 若物理 Q384 不改善高 IoU/pre-NMS recall，或明显破坏 Avg-mAP，则停止
  把该 Q-lift 包装成 PhysTime 主方法。
- 只有首轮 20-epoch 因子实验通过后，才进入多 seed、完整训练和第二
  数据集；不得直接提交新的 60-epoch 大矩阵。

## 5. 论文表述边界

当前可以说：

> 在固定 K384/J192、不做 feature interpolation 的 THUMOS14 单种子完整
> 训练中，真实物理时间度量相对规则 rank 秒轴获得显著收益。

当前不能说：

- 已达到随机采样 ActionFormer 或 dense AdaTAD 的公平 SOTA；
- 剩余性能差距已经由 query 数量解释；
- cross-attention Q-lift 已经有效；
- 当前方法是首个 continuous-time TAD；
- 当前路线已经 paper-ready。

潜在的新颖性只能收缩为：

> 固定稀疏原始观测下，显式区分 observation support、native feature
> token 和 detection query，不把 query 计作观测，不做 dense RGB
> evidence imputation，并在物理时间上构造 query、assignment 和 decode。

该表述仍需因子实验、多 seed、成本与跨数据集证据后才能升级为论文主张。

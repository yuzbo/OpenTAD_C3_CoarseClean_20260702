# PhysTime 冻结解码交叉回放设计

## 1. 目的

在不训练新模型的前提下，使用 full60 的 selected-axis 与
physical-metric epoch-59 冻结 checkpoint，回答 physical 收益是否能仅由
推理阶段的回归解码时间轴触发。

四个核心条件是：

| 训练来源 | 解码轴 |
| --- | --- |
| train-U | decode-U |
| train-U | decode-P |
| train-P | decode-U |
| train-P | decode-P |

该实验不能隔离训练期 assignment，但可以判断冻结模型是否能够跨时间轴
解码，以及收益是否必须依赖 physical-axis 训练。

## 2. 固定项与唯一干预

固定：

- full60 commit `0dc5851` 的 epoch-59 checkpoint；
- online 与 EMA 两套权重，二者同时报告；
- THUMOS14 原始视频、K384/J192、确定性测试采样与窗口；
- backbone、projection、neck、分类与回归卷积；
- Q0=192、六层候选总数 QΣ=378、有效 mask；
- 分类 logits、非负左右回归量；
- per-window top-k、外部分级标签、全精度跨窗口 NMS 与 evaluator。

唯一干预是把同一份解码前原量分别放到：

- `uniform_rank_seconds`；
- `physical_time_seconds`。

不得加入训练、Q-lift、插值、新损失、新采样器、新 NMS 或新阈值。

## 3. 方案裁决

### 方案 A：四个配置重跑完整推理

优点是代码少；缺点是无法用一个 artifact 证明两次运行的 logits、回归量和
mask 完全相同，因此不作为正式方案。

### 方案 B：每个冻结权重只推理一次，再离线双解码

每个 `checkpoint × weights_source` 只运行一次真实 raw-video 推理，保存
解码前原量与两套可审计时间轴。独立 CPU 回放器从同一 artifact 生成 U/P
结果。该方案能直接证明分类、回归和候选不变，采用此方案。

### 方案 C：从 P0 已解码 proposal 反推回归量

一个区间不能唯一确定其候选中心、stride 和左右归一化距离；该逆问题不可
辨识，禁止使用。

## 4. 解码前 artifact

每个冻结权重生成一份 tensor artifact 和一份 JSON manifest。

Tensor artifact 至少包含：

- `cls_logits[N,Q,C]`，float32；
- `reg_distances[N,Q,2]`，ReLU 与 level scale 后的 float32 原量；
- `valid_mask[N,Q]`；
- `base_points[Q,4]`，列为 rank center、regression range 下/上界、nominal
  stride；
- `level_offsets[L+1]`；
- `uniform_axis_sec[N,J]` 与 `physical_axis_sec[N,J]`；
- `native_valid_count[N]`；
- `domain_sec[N,2]`。

JSON manifest 至少包含：

- runtime/source commit、tree、config 与 checkpoint 哈希；
- train axis、weights source、epoch；
- 每个窗口的 `video_name`、`duration`、窗口标识和采样 checksum；
- class map、K/J/Q/C、dtype/shape；
- tensor artifact SHA256 与各核心张量 canonical hash。

artifact 只在显式 replay 配置开启时生成。普通训练与普通测试路径不产生
额外状态或文件。

## 5. 离线双解码

对每个窗口和 FPN level：

1. 以原始 rank center、nominal stride 构造基础 point；
2. 用端点 `[-0.5, J-0.5] -> [domain_start, domain_end]` 和 J 个轴中心做
   分段线性映射；
3. 分别映射 center、`center-0.5*stride`、`center+0.5*stride`；
4. physical stride 为左右映射之差，regression range 同比例缩放；
5. 使用同一 `reg_distances` 解码
   `start=center-left*stride`、`end=center+right*stride`；
6. 按同一 valid mask 取候选并 clamp 到窗口物理域；
7. 使用生产 `SingleStageDetector.post_processing`、全精度跨窗口 NMS 和
   evaluator。

U/P 回放共享同一个 tensor artifact，禁止在两个分支中重新运行网络。

## 6. 等价校验与 fail-closed

原生路径必须逐预测复现：

- train-U 的 decode-U 等于该次真实推理 direct result；
- train-P 的 decode-P 等于该次真实推理 direct result。
- 每个当前 capture-enabled direct 还必须等于对应已审计 P0 direct 的
  pre-cross、最终结果和 mAP。

比较内容包括：

- pre-cross-window canonical prediction hash；
- fullprecision-filtered 最终 prediction hash；
- 五个 IoU mAP 与 Avg-mAP，容差 `1e-12`；
- proposal 数、非法 proposal 数、NMS 输入/输出计数；
- evaluation epoch、class map、checkpoint/config/data 绑定。

以下任一情况立即失败：

- 缺 logits、回归量、mask、base point、任一时间轴或窗口绑定；
- 轴非有限、非严格递增、越过物理域，或 valid count 不一致；
- U/P 分支的 logits、回归量、mask 或候选拓扑 hash 不同；
- 原生回放不能逐预测复现 direct result；
- 当前 direct 不能逐预测复现已审计 P0 direct；
- 除 capture 开关外的推理语义 hash 与 P0 不一致；
- checkpoint 不是 epoch 59，或 source/runtime/data hash 不匹配；
- 出现非法 proposal、NaN/OOM/Traceback 或 completion 缺失。

禁止用最终 proposal 反推缺失原量，也禁止用近似等价替代原生逐预测等价。

## 7. 实验 DAG

1. 首次 `sbatch` 前运行纯 CPU preflight：
   - 重算 THUMOS 全文件内容、VideoMAE、两个 checkpoint、source
     manifest/completion、P0 suite/gate/四 condition completion 的哈希；
   - 原子写入 preflight manifest，并把其 SHA 注入全部作业。
2. 一个真实 gate：
   - 静态配置与 checkpoint/data/hash 检查；
   - focused tests；
   - selected/physical × online/EMA 四个真实 THUMOS window 的 artifact
     capture、U/P decode、native/P0 equivalence；
   - 四条件共享观测/时间轴契约与各自 state hash 早期校验。
3. 四个冻结任务：
   - selected-online；
   - selected-EMA；
   - physical-online；
   - physical-EMA。
4. 一个 suite validator，依赖四个任务全部成功。

每个冻结任务只做一次 GPU 网络推理，U/P 解码和评价在同一任务内由 CPU
artifact replay 完成。suite 只聚合已验证 completion，不重新训练或改写
预测。每个作业使用唯一 DAG token/comment；全局 owner manifest 必须永久
绑定 `token/run_root/commit/tree`，同 token 换运行目录必须在查询或取消
任何作业前失败。调用 `sbatch` 前先持久化提交意图；响应丢失、非数字响应或
已接受作业在有界轮询内仍不可见时，必须保留 ambiguous 状态并退出，禁止同一
调用自动重投。后续恢复只能查询并接管唯一 exact-comment 作业；可靠 accounting
或人工核验前仍不可见时继续失败闭合。提交后保存 scontrol 依赖快照，suite
保存 sacct 终态与 ExitCode。`resolved` marker 永久绑定已确认 Job ID，
暂时不可见时只能等待该 ID；`fatal` marker 永久禁用该 token。suite 必须
逐项证明六个 resolved marker 与 `jobs.tsv` 的 token/comment/Job ID 一致，
并拒绝任何 ambiguous/fatal marker。

## 8. 报告与裁决

先报告八行原始结果：

`train axis × weights source × decode axis` 的五个 IoU mAP 与 Avg-mAP。

随后报告：

- 同一 checkpoint 的 `decode-P - decode-U`；
- 固定 decode 轴下 `train-P - train-U`；
- 仅作描述的跨 checkpoint 差分之差，不解释为训练因果效应；
- online/EMA 一致性；
- 基于最终检测结果的短/中/长动作 oracle recall@0.5/0.7/0.9，不称为 pre-NMS proposal recall；
- NMS 决策与边界位移。

状态最高为 `tested`。单数据集、单 seed 的冻结回放不能产生
`paper_ready` claim。

若 artifact 可辨识且原生等价通过，则继续正式回放；若原生等价失败或冻结
原量无法唯一绑定，则停止部署并发起 Pro 讨论，不进入 UU/UP/PU/PP 训练。

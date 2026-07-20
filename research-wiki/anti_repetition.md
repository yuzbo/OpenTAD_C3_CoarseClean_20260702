# Anti-Repetition Contract

## 2026-07-20 冻结解码交叉回放禁区

- 本轮只做冻结 epoch-59 张量的离线重解码，不得加入新训练、Q-lift、
  interpolation、新 sampler、新 loss、新 assignment 或新 NMS。
- 捕获的 native proposal 仅用于数值审计，禁止把它覆盖回重建结果；
  否则 native exactness 是自证循环，整个实验无效。
- 不得把强制 float32 sigmoid 当作原 AMP 生产路径真值；必须保留并记录
  原始 dtype。准确表述是“AMP 产生的张量数值以 float32 存储，再由 CPU
  float32 重算 decode”，并用 native direct 与 P0 direct 双重等价门禁。
- 真实门禁必须覆盖 selected-axis / physical-metric 乘 online / EMA
  四个条件。缺任一条件、共享观测/时间轴契约、内容哈希或 checkpoint
  state 哈希，都不得提交正式 replay。
- 除新增 capture 开关外，模型、数据流水线、后处理、评估与 test solver
  的推理语义必须与已审计 P0 完全一致。
- 同 checkpoint 内 U/P 差异才是固定张量的解码轴干预；跨 checkpoint
  差异与差中之差只允许作描述性诊断，不得宣称训练轴因果效应。
- 时长分组指标来自最终检测的 oracle recall，不能命名或解释成 pre-NMS
  proposal recall。
- native direct 或 P0 direct 等价门禁失败时，必须 fail-closed 并发起
  Pro 讨论；禁止绕过门禁、降低阈值或先跑 Q192 训练。
- suite 通过最多把状态提升为 `tested`，不能直接提升为
  `empirically_supported` 或 `paper_ready`。

## 2026-07-20 P0 实现禁区

- 不得把 P0 写成训练实验；它只使用冻结 epoch-59 online/EMA 权重。
- 不得先过滤全精度输入、后做 legacy 舍入。必须分别审计 raw validity 与
  effective NMS-input validity；舍入诱发零时长属于第二层。
- unfiltered 不等于“把 NaN/零时长送入 NMS”。发现非法值必须在 NMS 前
  fail-closed，并保存审计。
- 四个单臂完成不等于 suite 完成；缺 `P0_SUITE_COMPLETE.json` 时不得形成
  physical-minus-selected 或 online/EMA 结论。
- 单臂 validator 不得导入 producer 的模式常量或信任 producer delta；
  suite validator 不得只信任四份 completion 中的汇总数字。
- 不得用 P0 同时修改 Q、插值、loss、采样、assignment、网络结构或训练
  schedule；这些变量继续冻结。
- P0 若显示舍入/过滤影响很小，只能关闭后处理混杂，不能自动证明模型结构
  新颖或 paper-ready；若显示影响很大，则先修正绝对指标并重新审视机制。

## 2026-07-20 冻结解码回放契约

- `observation_sequence_sha256`、U/P 两套轴数组、mask、base points 和计数
  才是 selected/physical 四条件应共享的观测契约。
- `window_sequence_sha256` 含 `native_coordinate_mode`，属于轴特有契约；
  selected-axis 与 physical-metric 按设计不同，只能要求同一 arm 的
  online/EMA 一致。禁止再次把它并入四条件全等检查。
- 临时 `flock` 不能替代永久所有权。每个 DAG token 必须由全局 owner
  manifest 永久绑定唯一 `run_root/commit/tree`；同 token 换目录必须在
  查询或取消作业前失败。
- `sbatch` 响应丢失后不得只查一次就重投。必须先做有界可见性轮询，并在
  调用前持久化提交意图。轮询预算耗尽仍不可见时必须保留 ambiguous 状态并
  退出；恢复流程只能查询 exact comment，不得自动再次 `sbatch`。延迟可见的
  唯一作业必须被接管，可靠记账或人工核验前不得清除 ambiguous 状态。
- `resolved` marker 不是日志装饰：作业暂时不可见时仍只能等待其记录的
  Job ID，禁止重投。`fatal` marker 永久禁用整个 token；suite 必须验证六个
  resolved marker 与 `jobs.tsv` 一致并拒绝任何 ambiguous/fatal 状态。
- Slurm 的多作业 `afterok:a:b:c:d` 在 `scontrol` 中可显示为
  `afterok:a(unfulfilled),afterok:b(unfulfilled),...`。禁止再做删除括号后的
  字面字符串比较；必须解析依赖类型与数字 Job ID，按集合核对且拒绝缺失、
  额外、错误类型、错误 Job ID 或重复项。
- capture focused test 的最小 `Config` 可以不含 `solver`；读取审计字段时
  必须安全处理可选段。该兼容性只影响清单记录，禁止借机改动生产配置或 AMP
  推理语义。

## 2026-07-20 STOP-Q-LIFT 禁区

- 不把 `STOP-Q-LIFT` 写成“Q 已被证明无用”。准确含义是：Q bottleneck
  未被证明，因此训练型 Q-lift 当前 fail-closed 不获授权。
- 当前唯一任务是 `P0-FULLPRECISION-NMS-REPLAY`；不得并行加入 Q384、
  interpolation、copy、cross-attention、gap projection、新 loss 或训练。
- P0 必须分别记录 rounding 开关与 proposal validity filter 开关的影响；
  不把两个评估修复混成一个因果解释。
- `CODE_CORRECT=false` 不得用于撤销现有 `57.57%`；正确边界是核心
  physical geometry/结果有效，发布级 evaluator 与 tail-mask 合同未闭环。
- UU/UP/PU/PP 首字母是 decode/回归轴，次字母是 assignment 轴。正确
  主效应为：
  `Δdecode=((PU-UU)+(PP-UP))/2`，
  `Δassignment=((UP-UU)+(PP-PU))/2`。
- 当前 strict inside-GT 使用 decode center。四臂只分解
  regression/decode/inside-GT 轴与 center-sampling/range-eligibility 轴；
  禁止宣称 assignment 与 decode 已完全纯净解耦。
- FPN tail、零时长 proposal、`random_trunc` fallback 是待验证代码风险，
  不是已经证明的 `57.57` 性能来源。
- 只有无训练 Q-density replay 的 oracle/pre-NMS 高 IoU 指标显示明确收益，
  才允许恢复训练型 Q-lift 讨论。

## 2026-07-19 Full60 / Q-Lift 审查禁区

- 不推翻现有 `57.57%`：它是可信的单种子 full60 matched 结果；NMS 提前
  round 是共享协议问题，不使 `57.57 vs 41.28` 差值失效。
- 不再把旧随机 `63.61%` 或 dense `68.29%` 当当前方法的公平因果对照。
  旧随机包含 `J192 -> Q384` feature interpolation、不同候选数和
  selected-rank GT；dense 又改变观测与计算。
- 不把剩余 mAP@0.7 缺口直接归因于 Q。Q、表示、assignment、排序、回归、
  观测缺失和 NMS 都必须由四臂与 proposal replay 分解。
- 不把 support-to-query cross-attention 称为已证明的唯一方案。它只是
  `designed candidate`，必须与共享简单 lift 辅助对照一起审计容量贡献。
- 新 bridge 的 Q192 也不等于当前旧 Q192 feature state；四臂必须在同一
  新 commit 下全部重跑，旧 `41.28/57.57` 只能作历史锚点。
- 不把 query 计作 frame、RGB、tubelet、observation 或新增 evidence；
  Q384 只能表示更多检测状态。
- 不做破坏单调性的字面 timestamp shuffle。反事实必须保持观测顺序、
  合法 domain 和严格递增时间轴，可用 uniformization 或 gap 置换后累积。
- 不把 `+6pp/+4pp/+1.5pp`、`1.40x/1.35x` 等外部建议阈值写成已证明合同；
  先由 seed 方差、功效与成本预算预注册。
- 不未经 raw-video 协议审计就把 ActivityNet-v1.3 固定为第二数据集。
- 不直接启动新 Q-lift 60-epoch/full matrix。该旧四臂 Q×coordinate 计划
  已被 2026-07-20 裁决取代；顺序改为 P0 replay、冻结 decode
  cross-replay、Q192 轴因子化、无训练 Q-density replay，再决定是否恢复。
- “无 GT 采样”只允许描述已接受窗口内的固定不规则子采样；训练
  `random_trunc` crop 使用 GT，必须明确披露。

## 2026-07-18 Full60 Guardrail

- The user has explicitly authorized the matched 60-epoch survivor run; do not
  cancel it because older wiki text says full60 was awaiting authorization.
- Only `selected_axis` and `physical_metric` belong to this run. Do not add G1b,
  DUCA, a learned selector, interpolation, a new sampler, or a dynamic budget.
- `scheduler.max_epoch` and `workflow.end_epoch` must both be 60. Stopping a
  100-epoch cosine schedule at epoch 60 is not an admissible full60 result.
- A Slurm completion code is insufficient. Both arms require independently
  recomputed epoch-59 mAP and replayable finite online/EMA final checkpoints.
- All matched validations through epoch 59 favor physical-metric. Final
  selected-axis/physical-metric Avg-mAP is `41.28/57.57`, and both completion
  artifacts pass. The admissible status is now `full60-single-seed-supported`.
- Do not promote this single-seed THUMOS result to `paper_ready`; multi-seed,
  mechanism, cost, robustness, and cross-dataset evidence remain missing.

## 2026-07-16 PhysTime GT/window boundary guardrail

- 不再允许 `BuildPhysTimeRawFrameGeometry` 因 GT segment 落在 end-exclusive window 外一点点而在训练中随机抛错。必须先定位 `video_name`、dense crop window、selected raw frame window、秒坐标 domain，再执行可审计的 clamp/filter。
- clamp/filter 只能发生在 train-time GT 转秒坐标之后：segment 与 `[domain_start_sec, domain_end_sec]` 相交则 clamp 到窗口内；clamp 后长度不大于 eps 的 segment 必须过滤；`gt_labels` 必须同步过滤。
- 每次修复都必须写入 `phystime_gt_boundary_audit`，至少包含原始/保留/过滤/裁剪数量、越界幅度、filtered/clamped indices、视频名和窗口元数据。不能把这类问题伪装成数据加载偶发错误，也不能静默吞掉。

- 不得把 focused tests 或独立审查前两轮修复称为部署完成。G1a 必须先取得第三轮零 P0/P1，再绑定 clean commit/tree；真实 gate 未通过时不得提交 pilot，pilot 未产生原始 mAP 时不得写方法 claim。
- gate 的 optimizer 证据必须绑定固定参数名称集合，并逐步满足 state count 完整、min=max=当前 step；不得用最大 step、动态 `requires_grad` hash 或 buffer 变化替代真实参数更新。
- production `drop_last` 必须来自实际 DataLoader 属性；CPU batch 不得被 gate 原地搬到 GPU 并跨步骤持有；正式训练与 gate 必须显式绑定同源 sampler/generator seed。

开始方法修改、实验部署、论文改写或外部讨论前，必须先读本文件与 `query_pack.md`。

## 禁止回退

- 不把 DUCA、X3D、SlowFast、ChronoTransport 或 feature-token pilot 恢复为当前论文主线，除非有新的 superseding decision 和新证据。
- 不把 selected-rank 当物理时间，不把 GT 或预测边界映射到 selected-rank；只允许从秒坐标导出原视频帧号。
- 不用 Voronoi/support 扩张填满真实缺失区，不用 learned selector、actionness、teacher、oracle、ledger 或动态 K 污染 K384 三头主比较。
- 不混用不同 commit、采样、增强、checkpoint、schedule、seed、NMS 或 selected indices 的结果。
- 不把 smoke、one-step、gradient proof、进程存活或 epoch 0 loss 当成 mAP 与论文 claim。

## PhysTime 数值教训

- masked softmax 必须先把未覆盖 logits 置为 `-inf` 再求指数；禁止先 `exp` 后乘零，否则 AMP 下会出现 `inf * 0 -> NaN`。
- 单视频 one-step gate 只能证明局部合同，不能覆盖批间时长、support、mask 与 logit 极值；正式训练至少要越过首个 logging window，并扫描每个 leaf loss 的非有限值。
- gate 通过后 formal 仍可能揭示实现错误；此时必须将 gate 与 full-run 证据分级记录，旧作业降为 diagnostic，并以同一修复 commit 重跑全部 matched heads。
- 只越过 epoch 0 或首个 logging window 仍不足以证明稳定；`0bbf0e9` 的 PhysTime 在 epoch 1 end 才首次记录全 NaN，后续 gate 必须执行多 optimizer step 并 fail-closed。
- 不要把 AMP 缩放后的纯 Inf 与模型 NaN 混为一谈。先在 `unscale_` 后按参数记录 NaN/Inf，再决定：纯 Inf 且 scale 正常下降可在严格次数预算内恢复；任何 NaN、参数污染或跳步超限立即失败。绝不能对 Inf 梯度继续做 `clip_grad_norm_`。
- 单 GPU 不启用 FP16 DDP bucket compression；它没有通信收益，还会放大 scaled-gradient 溢出风险。PhysTime matched 协议固定 `amp_init_scale=1024`。
- 正式 gate 必须实际构建 evaluator 并验证 annotation/class-map 解析；训练配置能读数据不等于 evaluator 的独立相对路径可用。

## PhysTime 性能诊断教训

- 不把当前三头称为“仅检测头/仅坐标表示隔离”：PhysTime 同时删除了 ActionFormer temporal projection、跨 query 上下文并显著缩小可训练容量。
- 不把 raw absolute seconds 直接当 content embedding；秒坐标用于几何、assignment、decode 和 evaluation，表征输入必须做窗口/域归一化并单独审计尺度贡献。
- 不以“query 覆盖若干观测”证明 support integration 有效；必须同时报告 effective observation count、content/relative logit span 和层级坍缩。
- 不用延长训练、调 NMS 或增大 endpoint loss 处理短动作崩溃；先匹配候选密度、target assignment、容量和时序上下文。
- 不把 PhysTime 1.0 的负结果外推为 physical-time TAD 无效；当前实验首先证明的是实现与对照存在架构混杂。
- 不在 PhysTime 1.0 上继续调 endpoint、NMS、训练长度或单独 attention weight；该实现已经冻结为负基线。
- 不再把原生 192 tubelet feature 插值为 384 后与 384 raw supports 一一绑定，并把长度相等称为 feature provenance。
- 不把一个已融合两帧的 tubelet token 当成两个可独立相加的 feature values；multi-atom 首先只是 set-valued anchor provenance。
- 不在同一个“coordinate-only”实验里同时引入 `J192 -> Q384` lift、候选恢复和 support-mass operator；`K`、`J`、`Q` 必须分别审计。
- 不把 zero-coverage query 直接删除，也不把 gap token/跨 query 推断描述成已观测 feature 或 dense imputation。
- 不把 `SM-PTAF` 的外部公式、伪代码或 patch map 写成 `implemented`、`tested` 或已有 mAP。
- 不用参数总数接近替代容量公平；projection 深度、跨 query context、candidate topology、assignment 与训练更新必须同时对齐。
- 不把“原始 AdaTAD 使用 interpolation”误解为 G1a 也必须立即恢复 interpolation。插值可作为两臂共享的中性 query-grid lift，但必须单独归因，且永远不能把插值位置计作新增原始观测。
- 物理点写入和候选 mask 必须使用不同张量：rank/slot center 在写入物理中心前必须 clone。禁止把已被物理秒坐标原地改写的 view 与 selected count 比较，否则会静默删除合法候选并伪造性能下降。
- static precheck、真实 CUDA gate 与 pilot 完成是三种不同证据；只有 commit/tree/config/data/checkpoint 哈希一致、真实三步更新和正式 evaluator 通过，才允许启动依赖 pilot。
- 不能只在 VideoMAE 输入前把重复 padding 像素置零：无效 token 还会经 attention K/V、残差、MLP、TIA 卷积和 norm 回流污染有效 token；严格隔离必须逐层传播 mask，并以 padding 反事实和无效输入零梯度验证。
- gate 的推理尾样本和 evaluator 必须来自真实 test split；使用 validation/train 样本即使能跑 NMS 也不能证明测试闭环。
- 数据集 provenance 不能只哈希文件名与大小；必须使用完整文件内容摘要。checkpoint 不能只检查文件存在或字节非空；必须真实反序列化并从 manifest 独立重算 evaluator。
- FPS 容差不能凭经验拍定或默认为零；先全量审计 decoder FPS、annotation duration 与 frame count，再把保守阈值写入 train/test 同一合同，并由正式 gate 重算。
- 全量 timebase 审计的范围必须来自正式 dataset `data_list`，不能直接把数据根目录每个 MP4 都假设为 evaluator 样本；目录中的未引用文件必须显式披露并纳入完整 inventory 哈希，被 dataset 消费但缺失的文件则必须 fail-closed。
- 模型 state-dict 摘要必须覆盖 0 维标量 buffer；不同元素大小的 dtype byte-view 前先 reshape 为一维，不能假设所有参数/缓冲区至少一维。
- 多步真实 gate 不能伪装成“每个单样本所有参数族梯度都必须非零”。ActionFormer 回归头末端 ReLU 可在某个有效样本上让参数梯度为零，即使该步有正 assignment 和正回归损失；正确合同是每步正 assignment、正回归监督与全部有限，关键通路逐步非零，而回归参数族必须在固定三步聚合中至少一次非零。不能删除回归梯度证明，也不能仅凭正 loss 判定梯度已连通。

## 当前唯一主线

PhysTime 1.0 的 THUMOS14 raw-RGB/K384 三头实验、native-J192 matched full60
与 `P0-FULLPRECISION-NMS-REPLAY` 均已完成并冻结。当前唯一主线是
`exp:phystime-frozen-decode-cross-replay`，但真实门禁已经失败并进入 Pro
严审阻断态；`SM-PTAF` 仍为 designed candidate，训练型 Q-lift 与
Q192 UU/UP/PU/PP 均暂停。当前状态以 `query_pack.md`、
`experiments/phystime-frozen-decode-cross-replay.md` 和
`docs/evaluation/results.md` 为准。

## 2026-07-20 Decode-Cross Gate-Failure Guardrail

- `06a6734` 的真实 gate `1175820` 已失败，禁止继续写成
  `experiment_running`；状态是 `tested`、verdict 是 gate failed、mAP 是 NA。
- selected-online 与 selected-EMA 的单窗口 native 等价通过；
  physical-online 失败，physical-EMA 未执行。禁止把未执行条件写成失败或通过。
- 失败不是 physical point/proposal 重建误差。边界裁剪后 native proposal 与
  重建 proposal 逐元素相同；审计后处理结果也逐行相同。
- 已复现的根因是生产 `float16` 分类分数被存档统一上转 `float32`，导致并列
  排序和 top-2000 成员变化。禁止把“数值无损上转”等同于“排序语义不变”。
- 禁止降低哈希/逐行等价要求、增加容差、事后舍入、忽略 top-k 集合差异，或用
  捕获的 native proposal 覆盖重建结果。
- 禁止复用 run root、DAG token 或直接重排 `1175820–1175825`。下一次部署必须
  来自 Pro 审核后的新 commit/tree、clean snapshot、新 token 和四条件真实 gate。
- 在 Pro 明确裁决“保留源 score dtype”与“生产/重放共同稳定并列排序”的边界前，
  不修改正式实现、不启动 replay，不进入 Q192 UU/UP/PU/PP。

## 2026-07-17 G1b Medium-Run Guardrail

- G1b SDPQ 的 20-epoch medium run 只能证明该实现稳定训练且持续学习，不能与旧 G1a 六轮 pilot 直接比较。
- 禁止把 `30.88%` 与旧 selected-axis `10.26%`、physical-metric `10.56%` 的差值解释为结构收益；commit、训练轮数和训练阶段均不匹配。
- 下一项决定性实验固定为同一 commit、同 K384/J192、同采样、同 seed、同 20 epochs 的 selected-axis / physical-metric / G1b SDPQ 三臂比较。
- 三臂训练必须共享 gate、dataset manifest、预训练权重、优化器、scheduler、验证周期与 evaluator；任一差异必须在 manifest 中显式列出并解释。
- 若正式评价使用 EMA，最终轻量 checkpoint 必须保留 `state_dict_ema`；只保存 online 权重再声称 checkpoint 可复现评价属于证据断裂。
- 三臂结果完成前，不启动 60-epoch full train，不创建 paper claim，也不把 `medium_run_supported` 写成 `paper_ready`。

## 2026-07-17 Matched-Medium Result Guardrail

- 三臂结果已经完成；禁止继续写成 `experiment_running` 或 mAP=NA。
- matched 结果支持的是 physical-time metric：physical-metric `44.88%` 对 selected-axis `30.42%` Avg-mAP。
- 当前结果不支持 G1b SDPQ 结构优势：G1b `30.88%`，只在高 IoU 有小幅改善，同时低 IoU 覆盖下降。
- 禁止把 physical-metric 的收益归到 SDPQ、support-measure operator 或“continuous-time TAD”整体概念。
- 单 seed、20 epochs、单 THUMOS 只能写 `matched-medium-supported`；多 seed、完整 schedule、成本和跨数据集证据前不得写 `paper_ready`。
- 不因 medium survivor 自动提交 60-epoch full train；先明确复现矩阵、机制拆分和停止条件。

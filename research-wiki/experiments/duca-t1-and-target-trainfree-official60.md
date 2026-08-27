# DUCA T1 与目标域免训练前端 official-60

## 状态

`implemented / tested / experiment_running / no_terminal_map`

任务始终是 offline TAD。这里“target-train-free”只指 pre-backbone 选择前端；检测器仍用
THUMOS training split 训练。所有论文准确率只能来自完整 validation、terminal epoch-59 EMA、
OpenTAD tIoU 0.3--0.7 mAP。

## 科学问题

1. T1：相同 R2Q3/K384/G2 硬选择下，给 selected-frame feature 加零初始化真实时间残差，
   能否修复 selected-axis 把非均匀帧误当等间隔的问题？
2. T1 control：反转时间描述符若没有同向收益，可区分真实时间信息与单纯新增参数。
3. target-train-free：冻结外部先验、不用目标标签或目标校准时，状态变化/语义/不确定性证据
   能否优于 exact-uniform？
4. Fast-only：更强视频运动先验是否提高边界优先间接选帧；其完整成本是否抵消重 backbone 节省？

## 精确代码

- GitHub branch：`codex/duca-t1-trainfree-20260723`
- 初始模型提交：`f81aef436f57a7ed4aa23ae36e13fabfbcbf14f3`
- official-60 合同修复：`4c5604b4a0abde9e59f625d519934e855bfe1519`
- GitHub：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/4c5604b4a0abde9e59f625d519934e855bfe1519`
- 干净快照：`/data/run01/sczc063/yuzibo/projects/opentad_duca_t1_trainfree_4c5604b_20260723`
- Linux 回归：`29 passed in 43.19s`；pycompile、bash syntax、clean-tree 均通过。

## 实验矩阵

| 臂 | 前端证据 | 分配 | 目的 |
|---|---|---|---|
| exact-uniform | 无 | exact K384/G2 | 匹配基线 |
| trainable R2Q3 G0 | 可训练 ASFormer 转变 | R2Q3 | 可训练主候选锚点 |
| T1 actual | 同 G0 + 真实时间残差 | R2Q3 | 检验 selected-axis 时间扭曲 |
| T1 reversed | 同 G0 + 反转时间残差 | R2Q3 | 参数量匹配控制 |
| MobileNet change | 冻结特征变化 | 全局 exact-K/G | 低成本无参数变化先验 |
| MobileNet semantic | 冻结 ImageNet 分类语义 | 全局 exact-K/G | 语义先验消融 |
| MobileNet fusion | 变化/语义/不确定性固定融合 | R2Q3 | 低成本 target-train-free 候选 |
| SlowFast Fast fusion | 冻结 Fast pathway 固定融合 | R2Q3 | 高成本强视频先验诊断 |

SlowFast 臂只执行 `multipathway_blocks[1]`，不执行 Slow pathway 和 lateral fusion。它不是
低成本主候选，必须单列 encoder FLOPs/latency/energy。

## 权重封存

- MobileNetV3-Small ImageNet-1K：
  `/data/run01/sczc063/yuzibo/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth`，
  size `10,306,551`，SHA-256
  `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`。
- SlowFast-R50 Kinetics-400：
  `/data/run01/sczc063/yuzibo/.cache/torch/hub/checkpoints/SLOWFAST_8x8_R50.pyth`，
  size `277,138,115`，SHA-256
  `454f39e1c1f985df2bee2aa27887ed53ff56e74ed8b8cca11203a1a1264d7cc2`。

## 作业与证据

### 初始 T1 根

`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_trainfree_f81aef4_formal_20260723_1048`

- `1180637`：exact-uniform、trainable R2Q3 G0、T1 actual；RUNNING。
- `1180638`：T1 reversed 继续 RUNNING；同 bundle 的两个 MobileNet 臂因旧合同零步退出。
- `1180639`：首次 Fast 权重下载和 MobileNet 旧合同失败；无 optimizer update。
- `1180644`：Fast-only CUDA preflight 通过后在旧 checkpoint criterion 退出；无 optimizer update。

### 合同修复根

`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_4c5604b_retry_20260723_1107`

- `1180652`：三 MobileNet 臂已越过旧合同并构建真实 AdaTAD，随后因计算节点无法联网下载
  torchvision 权重零步退出；不是模型结果。
- `1180653`：Fast-only，无依赖；CUDA preflight 再次通过，epoch 0 已完成并进入 epoch 1。
  batch 99 的有限损失为 `Loss=1.4430 / cls=0.9721 / reg=0.4709`，K384，显存 6299 MB；
  两次孤立 AMP replay 均未耗尽重试，不构成训练崩溃。

### MobileNet 离线缓存重试根

`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_mobilenet_4c5604b_retry2_20260723_1115`

- `1180654`：三 MobileNet 臂，3-GPU exact/exclusive 并行，无依赖；已提交，当前
  `PENDING(AssocGrpGRES)`，属于账户 GPU 配额等待。

## 当前裁决

- T1、MobileNet 与 Fast-only 都尚无 terminal mAP，状态不得超过 `experiment_running`。
- Fast-only 已证明实现确实只执行 Fast 分支，但尚未证明性能收益，也尚未形成完整成本。
- MobileNet 是低成本候选；Fast-only 是强视频先验/成本上界诊断。二者不能互换叙事。
- 只有 target-train-free 前端同 K/G 超过 exact-uniform，且计入全部前端成本后仍有 Pareto
  优势，才允许升级为即插即用贡献。

## 2026-07-23 04:20 MobileNet 语义先验修复

- Job `1180654` 三臂均在构建 selector、optimizer 前退出：冻结循环对尚未首次前向的
  `nn.LazyLinear` 调用 `requires_grad_(False)`。没有 optimizer update 或 mAP，不能作为
  MobileNet 先验的性能结论。
- 逐行复核发现更重要的模型语义问题：`preserve_pretrained_classifier=True` 时旧实现仍把
  ImageNet 1000 类 logits 送入一个随机、未训练的 `LazyLinear(1000->1)`；这不仅触发异常，
  也破坏了“冻结 ImageNet 类分布作为免训练语义证据”的定义。
- 修复分支为 `codex/duca-t1-trainfree-lazyfix-20260723`，精确提交
  `e30db0f3987128798da6bc8ff446065b818b1a7f`。冻结语义模式现在直接保留预训练多类 logits，
  只有需要学习二分类且无法替换既有分类头时才允许惰性输出层。新增真实多类输出、有限置信度和
  全参数冻结回归测试；远端快照
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_t1_e30db0f_20260723` 上为
  `8 passed in 40.73s`，pycompile 与 clean-tree 通过。
- 唯一有效 MobileNet 重试为 Job `1180697`，在一个三卡 allocation 内并行
  feature-change、semantic 与 fixed-fusion R2Q3 三臂。run root 为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_mobilenet_e30db0f_retry_20260723_042025`，
  deployment manifest SHA-256 为
  `006a6db30cc63a8de37e162c639fb7e50276e0bad97b1ec308de9247d033fb2b`。当前
  `PENDING(AssocGrpGRES)`，不是失败。
- T1 `1180637/1180638` 与 Fast-only `1180653` 持续健康运行，不因该修复取消或重提。
  当前所有 train-free 臂仍无 terminal mAP，状态保持 `experiment_running`。

## 2026-07-23 04:25 自动巡检

- T1 core/controls `1180637/1180638` 已推进到约 epoch 15，Fast-only `1180653` 已进入
  official-60 epoch 13；活动日志没有新的 Traceback、OOM、non-finite collapse 或 FAIL。
- 修复后的 MobileNet 三臂 `1180697@e30db0f` 仍为 `PENDING(AssocGrpGRES)`，尚未开始模型
  构建或 optimizer update；旧 `1180654` 的 LazyLinear 堆栈不得再次计为新失败。
- 当前仍无 terminal mAP。Fast-only 或 T1 的有限训练损失只用于健康检查，不能替代最终
  OpenTAD Avg-mAP、逐 tIoU 指标与完整前端成本。

## 2026-07-23 05:05 T1 门禁失败与封存 P0 恢复

- `1180638` 已失败；`1180637` 顶层仍运行只是因为 `two_stage_exact_uniform` 对照继续训练。
  逐 step 核验确认 `boundary_burst_r2q3_g0`、`t1_true_time_residual_g0` 与
  `t1_reversed_time_residual_g0` 三个学习臂均已完成 20-epoch P0，随后在整模门禁命中此前已知的
  六个 BatchNorm 非参数 buffer 缺失。没有 official-60 optimizer update 或 mAP，因此不是 T1
  性能失败。
- 三个 epoch-19 P0 checkpoint 均为 `407,346,459` bytes，SHA-256 均为
  `acb6e30673e811f34ce84d710442581bec8a74ca68e9187eb71e005e01536c9b`。三者相同是因为 T1
  residual 在 P0 阶段尚未启用；它们没有被错误覆盖。
- 修复分支 `codex/duca-t1-gatefix-20260723` 的精确提交为
  `26ce86d7810e8f7c0568dc045bb1db7240c66de2`。它只移植严格参数/显式 buffer 初始化与哈希绑定
  P0 恢复，不改 T1 residual、selector、K/G 或 detector。远端不可变快照
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_t1_26ce86d_20260723` 通过 `15 passed`、
  pycompile、bash syntax 与 clean-tree。
- 三个无依赖恢复 Jobs 为 `1180717` R2Q3、`1180718` true-time residual、`1180719`
  reversed-time residual；run root 为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_26ce86d_recovery_20260723_050516`。
  deployment manifest 与 jobs ledger SHA-256 分别为
  `3f7b8c99775e67e077ce739c9238ac74404ea4e9b19c20ed7794813327395003` 和
  `d64a2e9331eeca88a991d1648a7eb674a028406736d398e0357420d27e077117`。`1180717/1180718`
  已获得 GPU，`1180719` 为 `PENDING(AssocGrpGRES)`。
- MobileNet 修复 Job `1180697` 的三臂均已通过真实整模门禁并进入 official-60 epoch 1；这证明
  `e30db0f` 已越过原 LazyLinear 构建故障，但仍无 terminal mAP。
- 05:07 复核：`1180717` 已通过原失败门禁并进入 official-60 epoch 0；`1180718` 正在执行恢复
  门禁，`1180719` 等待 GRES。当前新恢复根没有新增错误。

## 2026-07-23 05:35 T1 运行时枚举修复

- `1180718/1180719` 已在复用 P0 并通过整模门禁后、official-60 第一个训练 batch 前失败。根因是
  `run_duca_independent_official60_gpu1.sh` 正确选择了两份 T1 config，却把实验标签
  `t1_{true,reversed}_time_residual_g0` 传入 selected-axis runtime binder；旧
  `VARIANT_CONFIGS` 没有登记这两个合法实验/config 对，因此 fail-closed 抛出
  `invalid selected-axis variant`。没有 optimizer update 或 mAP，不是 T1 性能负证据。
- 最小修复精确提交为
  `codex/duca-t1-gatefix-20260723@919aa555d1aa36191ee318477409dfbfdfb0e807`：只登记
  T1 actual/reversed 与各自 official-60 config 的一一映射，不改 residual、selector、K/G、
  detector、损失或训练日程。GitHub 已推送；远端干净快照
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_t1_919aa55_20260723` 通过
  `16 passed in 49.55s`、pycompile、bash syntax 与 clean-tree。
- 只恢复两个受影响臂，不重跑 P0，也不重复健康 R2Q3：新 Jobs 为 `1180731` true-time residual
  与 `1180732` reversed-time residual；run root 为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_919aa55_recovery_20260723_053508`。
  deployment manifest 与 jobs ledger SHA-256 分别为
  `8fc94b4eb02994c42d137fd65ee6286b574a875fd488e7d7cb4a152c0432df26` 和
  `623b575ad7e8f93c45692c45f4e7e21264753048edb0e98e9c450680d6389dd2`。
- `1180717@26ce86d` 仍是健康 R2Q3 对照并继续训练；`1180718/1180719` 只保留为零更新合同失败
  历史。05:37 复核确认 `1180731` 已通过真实整模门禁并进入 official-60 epoch 0；`1180732`
  为 `PENDING(AssocGrpGRES)`。当前仍无 T1 terminal OpenTAD mAP。

## 2026-07-23 05:58 自动巡检

- T1 actual `1180731@919aa55` 已进入 official-60 epoch 2，batch 50 总损失 `0.9366`，
  `cls/reg=0.5039/0.3368`，K384 且更新有限，证明运行时枚举修复已越过原失败点。
- matched R2Q3 `1180717` 已到 epoch 6；exact-uniform 与 Fast-only 已到 epoch 30；MobileNet
  feature-change/semantic/fusion 三臂约 epoch 8。活动日志没有新错误。
- T1 reversed `1180732` 与 sparse recovery `1180696` 仍为 `PENDING(AssocGrpGRES)`，仅是资源等待。
  所有这些实验仍无 terminal epoch-59 EMA OpenTAD mAP。

## 2026-07-23 07:27 自动巡检

- T1 reversed `1180732@919aa55` 已获得 GPU、通过真实整模门禁并形成有限 official-60 更新；
  当前到 epoch 3 batch 50，总损失 `0.8838`，`cls/reg=0.4636/0.3281`。启动时一次 AMP
  skip 已由 replay `1/8` 恢复，之后持续有限更新，不构成数值失败。
- T1 actual `1180731@919aa55` 到 epoch 14 batch 50，总损失 `1.0453`；matched R2Q3
  `1180717` 已进入 epoch 18。三臂均未出现 Traceback、OOM、non-finite collapse、ValueError
  或 replay 耗尽。
- exact-uniform 已进入 epoch 44，Fast-only 到 epoch 46，MobileNet 三臂约 epoch 20--21；
  sparse recovery `1180696` 仍为 `PENDING(AssocGrpGRES)`。当前仍无 terminal epoch-59 EMA
  OpenTAD mAP，以上损失和轮次只用于训练健康检查。

## 2026-07-23 09:02 Fast-only 终点结果

- Job `1180653` 状态 `COMPLETED/0:0`，完成 `6000/6000` 次 optimizer、scheduler 与 EMA 更新。
  主 checkpoint 为 `epoch_59.pth/state_dict_ema`，SHA-256
  `d75aaabf5b85257e242ea239cd54f70ad2063c6a58b7cfb61d6112cf6bd8eef0`；terminal evaluation
  SHA-256 为 `f4870208f9c2ae8b779861b3d8877da2b0ae8ec3948aa613b9720f881cb5b7fd`。
- 完整 THUMOS validation、OpenTAD 官方 evaluator、211 videos、3325 GT instances 的结果为：

| 方法 | Avg-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen SlowFast Fast + transition fusion R2Q3, K384 | 63.5297 | 79.9106 | 74.5241 | 66.3665 | 54.7535 | 42.0937 |

- 这是首个可比较的 train-free terminal mAP，但 matched exact-uniform 仍在训练/评估，当前不得引用
  历史 64--65 作为精确 paired baseline。现阶段结论仅为：Fast-only 没有达到约 65% 目标，尚未
  证明冻结视频运动先验优于均匀采样。
- 初步根因候选按证据强弱排序：R0 已否定 R2Q3 几何存在稳定正 headroom；Fast pathway 对运动和
  相机变化敏感但不直接校准动作起止边界；非均匀 R2Q3 位置仍由 selected-axis detector 当作等间隔
  输入。必须等待 matched uniform，并补做硬位置到 GT 边界距离/微簇与左右分布诊断后才能区分
  “粗先验偏移”“选择几何不足”和“时间坐标扭曲”。
- 该 run 尚未形成配对前端 latency/FLOPs/energy 与终点选帧质量产物，因此当前不能主张 Pareto
  优势；Fast-only 继续定位为强冻结先验诊断，而不是最终低成本插件。

## 2026-07-23 09:11 exact-uniform 终点评估启动

- Job `1180637` 的 matched exact-uniform 已完成 60 epoch、`6000/6000` 次训练更新，训练日志以
  `Training Over` 正常结束；没有 OOM、非有限损失或 replay 耗尽。
- 官方终点评估已严格加载 `epoch_59.pth/state_dict_ema`，验证集为 211 videos / 792 windows；
  09:11 时进度约为 `134/396` batches。尚未生成最终 mAP，因此当前仍不得计算 Fast-only 的
  matched delta，也不得用历史 64--65 结果代替本次精确对照。

## 2026-07-23 09:26 exact-uniform 与 Fast-only 配对结果

- matched exact-uniform 已用 `epoch_59.pth/state_dict_ema` 完成 211-video THUMOS validation
  官方 OpenTAD 评估：Avg-mAP `64.49%`，mAP@0.3/0.4/0.5/0.6/0.7 为
  `79.59/75.42/67.71/57.27/42.45%`。
- Fast-only 的 Avg-mAP `63.5297%`，相对同协议 exact-uniform 为 `-0.9603 pp`。
  分 tIoU 差值为 `+0.3206/-0.8959/-1.3435/-2.5165/-0.3563 pp`：只在 tIoU=0.3
  有微小正差，在 0.4--0.7 均更差，尤其 tIoU=0.6 退化明显。
- 这是当前可比较的官方终点证据：冻结 SlowFast Fast 运动先验 + R2Q3
  没有优于均匀采样，也没有守住约 65% 目标。它继续作为 train-free 强先验
  负结果/诊断对照，不升级为主方法。

## 2026-07-23 09:44 历史 65.696 身份纠正

- 历史 Job `1150842` 的 epoch-59 终点确实为 Avg-mAP `65.69%`，但它使用
  `physical_grid_actionformer.enabled=True` 的 grid-aware ActionFormer，GT 与 prior/assignment/decode
  保持原始物理时间坐标。这是改造后的检测几何诊断，不是标准 AdaTAD 均匀基线。
- 更相当的历史 native stride-2/selected-axis Job `1150701` 终点为 `64.31%`，
  best 为 `64.35%`。当前 exact-uniform `64.49%` 与它一致，所以不存在明显的
  均匀训练回归；表面上相对 65.696 的约 1.2 pp 差距主要来自检测时间几何不同。
- Fast-only 并不是“冻结动作二分类器”。它是冻结 Kinetics SlowFast Fast 特征，
  用固定 `0.75 feature-change + 0.20 semantic + 0.05 uncertainty` 证据和参数无关
  R2Q3 分配；selector 不在 THUMOS 上训练，也不接收 detector gradient。

## 2026-07-23 14:04 Remote T1 and MobileNet terminal update

- T1 actual `1180731@919aa55` completed 6000 updates and produced a hash-bound terminal EMA
  evaluation: Avg-mAP `64.0200%`, with @0.3/0.4/0.5/0.6/0.7 of
  `79.1470/74.4357/67.1649/56.9444/42.4083%`. It is `-0.4700 pp` versus the matched
  exact-uniform `64.49%`, so actual-time residual has no positive support yet.
- T1 reversed `1180732@919aa55` was still RUNNING at 14:04 after training and during terminal
  evaluation; it had no `Average-mAP` line or terminal JSON at that check.
- MobileNet single-seed terminals were feature-change `63.27%`, semantic `62.78%`, and fixed
  fusion R2Q3 `64.33%` Avg-mAP. Fusion is best among these three but still about `-0.16 pp`
  below matched uniform; its @0.7 `42.73%` alone does not establish a Pareto advantage. No
  trained full frontend-cost pair exists.
- Fast-only, T1 actual, and MobileNet remain diagnostic/negative results rather than a main
  method or a training-free end-to-end claim.

## 2026-07-23 14:45 R2Q3 versus MobileNet fixed-fusion selector replay

Slurm Job `1181557` is an `experiment_running` visualization-only replay. It
uses the terminal `epoch_59.pth/state_dict_ema` checkpoints for the official
R2Q3 hard-detached arm (`38580fa...e375`) and MobileNet fixed-fusion arm
(`89345f...4474`), respectively. It runs exactly one deterministic validation
batch (four samples) through the existing selector-quality exporter, with no
optimizer update and no checkpoint mutation, then reuses the existing geometry
analyzer, timeline, dashboard, and geometry-suite plotters. GT appears only in
the post-hoc evaluation overlay; it is not passed to either selector.

The output root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_selection_visualization_20260723_1445`.
This produces mechanism visualizations only; it does not alter the terminal
mAP comparison, choose a checkpoint, or support a cost/Pareto claim.

The first wrapper `1181557` failed at zero runtime because `/etc/profile`
read an unset `XDG_DATA_DIRS` under strict-unset mode; no Python, GPU replay,
or optimizer step ran. Replacement `1181559` disables unset-variable checking
only while sourcing the profile, then restores it before module/environment
setup. It is the single valid visualization replay submission.

`1181559` then reached the exporter but failed before constructing the model:
the R2Q3 config correctly requires the hash-bound epoch-19 frontend-init
environment variables even when replaying the terminal checkpoint. It created
no selector record and made no optimizer update. Replacement `1181560` passes
that exact recorded path, SHA-256, and epoch (`19`) as config prerequisites;
it is the only live replay job.

`1181560` loaded the validation videos and reached the R2Q3 exporter, then
correctly exposed that the official validation pipeline did not collect
`gt_boundary_validity` for a post-hoc overlay. It emitted no complete record
and performed zero optimizer updates. The derived diagnostic configs under the
visualization run root keep the original validation videos, windows, decoding,
and selector model unchanged, adding only the existing endpoint-validity field
to `LoadFrames`/`Collect`. The selector `forward_test` still receives only
inputs, masks, and metadata. A malformed path-only submission `1181563` was
cancelled immediately before replay; `1181564` is the sole corrected live job.

`1181564` completed the R2Q3 export, but MobileNet construction attempted to
download public torchvision initialization weights and the compute node had no
DNS route. This is an external-cache failure before MobileNet selector output,
not an experimental result. The terminal checkpoint already contains the
MobileNet parameters, so the derived selector-only config used by `1181572`
disables constructor-time download and strict-loads the hash-bound terminal
state immediately afterward. No pretrained weight, selector parameter, or
decision rule is substituted.

The first derived override was attached at the selector root rather than the
nested `actionness_source_cfg`; `1181572` was cancelled after static config
inspection, before it could produce a valid MobileNet record. `1181575` is the
sole replacement. Its CPU precheck resolves
`actionness_source_cfg.mobilenet_pretrained is False`; terminal checkpoint
strict-loading remains unchanged.

`1181575` completed successfully. It exported four identical validation
windows per arm and rendered the timeline/dashboard artifacts. Both arms use
K=384 and max-hole=2. Their selected-set Jaccard overlap is 1.0000 on the
253-frame short window (both necessarily select every valid frame), 0.6879 on
the 503-frame window, and 0.3813/0.3497 on the two 768-frame windows of
`video_test_0000007`. The full-window `|6912` timeline is the meaningful
decision visualization; its dense rugs show different R2Q3 and MobileNet
fixed-fusion positions even under the identical K/G contract. These are
`tested` mechanism diagnostics only, not performance or cost evidence.

The follow-up figure reads `gt_segments` directly from the original exported
R2Q3 record for `video_test_0000007|6912`, rather than from the reduced
geometry-analysis CSV. It overlays the four GT action intervals and boundary
lines (strictly post-hoc) and plots `transition_policy_scores` as the R2Q3
curve; the flat `p_action` trace is deliberately omitted. R2Q3 and MobileNet
fixed-fusion selected-position rugs remain in separate aligned lanes. This is
`tested` visualization repair only: the GT is not supplied to selector
inference and no checkpoint, decision rule, or experiment output changed.

For that single full window, its 94 raw-GT grid positions quantify the visual
impression: hard-detached R2Q3 selects 53 inside GT (56.4% GT coverage; 13.8%
of its K=384 selections), whereas MobileNet fixed fusion selects 79 (84.0%;
20.6%). This supports a selector-quality explanation only. It must not be
read as a detection-performance ranking: the single-seed terminal Avg-mAP is
64.00% for hard-detached R2Q3 versus 64.33% for fixed fusion, while fixed
fusion remains about 0.16 pp below matched exact-uniform and has no matched
full frontend-cost pair.

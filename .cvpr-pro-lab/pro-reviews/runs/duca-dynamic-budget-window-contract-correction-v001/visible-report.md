# 唯一裁决：**REVISE**

原来的动态预算科学问题值得继续，但原训练窗口合同必须修订。直接把“同一视频内窗口排序”写进现有 H65 forward 会因训练时每个视频每轮只有一个随机截断而退化为 `n=1 → K=384`，不能产生动态预算证据；这不是路线失败，而是实验估计量没有被正确实例化。

**修订后的一句话科学问题：**在保持 H65“每个视频每个 epoch 只训练一个 768 个时序位置的随机截断”、Stage-2 语义学习目标和下游 AdaTAD/ActionFormer 不变的条件下，能否用固定的 Stage-1 语义侦察证据，在同一视频跨 60 个预先冻结的训练截断以及固定验证滑窗之间重新分配 `K∈{256,384,512}`，使每视频平均重型预算仍为 384，并在真实 variable-K VideoMAE 执行下优于内容无关的同预算置换，同时相对同训练合同的固定 `K=384` 对照下降不超过 `0.30` 个 Avg-mAP 百分点？

## 1. 代码核验后的关键更正

提示词指出的三个冲突均成立，而且还存在一个必须一起修复的 VideoMAE 输入合同。

H65 的正式训练并不是滑动窗口训练。训练 pipeline 是 `LoadFrames(method="random_trunc", trunc_len=768, trunc_thresh=0.75, crop_ratio=[0.9,1.0])`；验证和测试才使用滑动窗口。训练集本身仍是一视频一个 dataset item，batch size 为 2，dataloader 是普通 `DistributedSampler`。因此当前一次训练 forward 确实没有“同视频窗口总体”。

H65 Stage-2 的 ASFormer 动作性路径和 transition scorer 也确实继续更新；Stage-2 仍保留动作性、transition 和边界监督，并按 6000 次成功 optimizer update 推进。因此“用固定预算清单”与“让产生该清单的同一个侦察器继续变化”不能同时成立。

当前 `acquisition.py` 虽然已经能够得到不同的 requested/effective K，但记录的 `backbone_input_k` 仍被写成最大预算，并明确标记 `dynamic_compute_realized=False`。

还有一个先前合同没有写全的实现事实：H65 的 VideoMAE wrapper 本身固定为 `24 × 16 = 384` 个输入位置，post-processing 又固定插值到 384，`projection.max_seq_len=384`；`ActionFormer.pad_data()` 也会把短输入补到 `max_seq_len`。所以只改 acquisition 层仍会制造伪 variable-K。真正的 `256/384/512` 必须同时让 VideoMAE wrapper 和 ActionFormer detector path 接收真实长度。

这也是我选择 **REVISE 而不是 PIVOT** 的理由：动态预算问题仍然可证伪，只是原来的训练窗口总体定义错误。

---

# 2. 冻结的数据与窗口合同

## 2.1 Training：保留 H65 的随机截断分布，不改成滑窗训练

**不把训练改成固定滑动窗口。**

每个 THUMOS14 training video 在 Stage-2 的每个 epoch 仍恰好贡献一个 temporal crop，因此：

* 60 epochs；
* 每视频共形成 `n_v=60` 个**训练窗口 occurrence**；
* 每个 epoch 仍只有这个视频的一个训练样本；
* 数据集长度、每 epoch batch 数和 6000 次成功 optimizer update 不变。

唯一变化是：原本在 worker 内临时产生的 60 次随机截断，改为**正式训练前一次性确定并冻结**，三条实验臂完全共用。

对于视频 `v` 和 epoch `e∈[0,59]`：

1. 使用独立确定性随机源 `hash(nonce, seed=3407, video_name, epoch)`；
2. 完整复现当前 H65 `random_trunc` 的 temporal sampling law，包括：

   * dense window length `768`；
   * `trunc_thresh=0.75`；
   * 最多 200 次候选尝试；
   * `crop_ratio=[0.9,1.0]` 的短视频行为；
   * 当前 GT 截断与 `gt_boundary_validity` 语义；
3. 只冻结 temporal start、valid length 和 epoch identity；**不冻结 RGB augmentation**。RandomResizedCrop、Flip、ImgAug、ColorJitter 等仍按现有训练 pipeline 工作。

当前 `feature_stride=4, sample_stride=1`，所以这里的 768 是 **768 个 temporal observations**，相邻 observation 对应原视频约 4 raw-frame indices；训练窗口的 raw-frame 起点由 `4 × start_snippet` 得到。

训练窗口之间**没有人为规定重叠比例**；它们仍是 H65 random-trunc 分布产生的自然重叠。重复 start 也允许，window occurrence 用 `(video_name, epoch, start, valid_len)` 唯一标识。

这保留了 H65“一视频一 epoch 一个随机 crop”的统计单位，而不是用大量滑窗把训练样本分布重写。

## 2.2 Validation / official evaluation：完全保留现有滑窗

不修改 `ThumosSlidingDataset` 的窗口定义：

* training-time validation：`window_size=768`，overlap `0.25`，stride `576`；
* official final evaluation：`window_size=768`，overlap `0.50`，stride `384`；
* 长视频的最后一个窗口按现有实现向前 back-shift，使其结束于视频尾部，而不是产生一个任意短的尾窗；
* 整个视频短于 768 时只有一个窗口，并按当前 mask/padding 规则处理。

### 短视频的硬条件

为了同时满足：

1. `K=512` 必须代表 512 个真实、唯一的 VideoMAE 输入 observation；
2. 不允许用 padding 冒充计算；
3. 每视频严格保持平均预算 384，

正式协议要求所有进入预算重新分配的窗口满足：

**`valid_len ≥ 512`。**

PRE_RUN 必须对完整 THUMOS14 training、validation 和 official-evaluation 窗口实际核验这一事实。

若真实数据中存在 `valid_len<512` 的正式窗口，**协议直接判为无效，不得把 padding 计入 K，也不得临时改比例或 K 集合**。这是数学上的预算合同不兼容，不是允许 Builder 自行修补的工程细节。

---

# 3. 冻结的侦察器与预算证据

这里必须明确区分两个科学角色，解决“冻结”和“继续训练”的表面矛盾。

### 预算排名用侦察器：永久冻结

唯一身份：

* checkpoint：`/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
* expected epoch：`29`
* state key：`state_dict_ema`
* SHA-256：`bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`

该身份已经在现有正式证据中核验。

它只负责**一次性产生窗口预算证据**：

* `eval()`；
* `requires_grad=False`；
* `torch.no_grad()`；
* training 60-window population、validation windows 和 final-evaluation windows 各预计算一次；
* Stage-2 期间绝不重算；
* 不从 Stage-2 checkpoint 更新；
* 不访问 validation/test GT、类别标签、detector prediction 或 raw-prediction cache。

### H65 Stage-2 内部的选帧侦察器：继续训练

正式三臂里的 H65 frame selector 仍从同一个 Stage-1 checkpoint 初始化，然后**完全按照现有 H65 Stage-2 合同继续学习**：

* actionness objective 不变；
* transition objective 不变；
* transition-boundary objective 不变；
* counterfactual detector-utility training 不变；
* learning-rate schedule 不变。

也就是说：

> **冻结的是“预算排名的只读副本”，不是 H65 Stage-2 正在训练的 selector。**

两者没有共享可变参数。预算生成工具运行结束后，只留下窗口→K 的结果；正式训练模型不能改写它。

这样既不要求“不断变化的 scout 产生固定清单”，也不为了预算实验把 H65 原有 Stage-2 语义学习冻结掉。

---

# 4. 语义预算和内容无关控制的唯一数学定义

对一个视频的窗口总体 `W_v={w_1,…,w_n}`，冻结预算侦察器为每个窗口产生现有 H65 路径中的：

* `transition_center_scores`；
* `p_action`。

窗口的两个统计量固定为：

$$
B_w=\operatorname{mean}_{t\in valid}\left[\text{transition\_center\_scores}_t\right]
$$

$$
U_w=\operatorname{mean}_{t\in valid}\left[1-|2p_{\text{action},t}-1|\right]
$$

其中第二项就是当前 acquisition 代码中已有的 actionness uncertainty 定义，不新增预测头。

在**每个视频内部**分别对 `B_w`、`U_w` 做 deterministic mid-rank percentile，然后：

$$
S_w=\operatorname{rank}(B_w)+\operatorname{rank}(U_w)
$$

不跨视频归一化或排名。

令：

$$
q_v=\lfloor n_v/3\rfloor
$$

则语义臂：

* `S_w` 最低的 `q_v` 个窗口：`K=256`
* 最高的 `q_v` 个：`K=512`
* 其余：`K=384`

所以恒有：

$$
256q_v+512q_v+384(n_v-2q_v)=384n_v.
$$

训练时 `n_v=60`，因此**每视频严格为 20 个 K256、20 个 K384、20 个 K512**。

内容无关控制臂不用任何 RGB、scout score、GT 或 detector 输出；它只根据

`SHA256(nonce || video_name || window_identity || "content-independent-control")`

获得窗口顺序，然后套用**同一个 `{256,384,512}` 多重集**。

`n_v<3` 时 `q_v=0`，所有窗口自然为 K384；不得为了“产生动态性”事后修改规则。

预算相等是**逐视频**成立，不只是全数据集平均成立。

---

# 5. Training / validation / cost 的匹配原则

三个正式实验臂为：

1. **semantic budget allocation**
2. **content-independent matched allocation**
3. **same-contract fixed K=384 companion**

三臂共享：

* 完全相同的 training temporal window schedule；
* 相同 Stage-1 初始化；
* seed `3407`；
* 相同训练增广；
* 相同 6000 次 optimizer update；
* 相同 detector、loss、NMS、evaluation；
* 相同 checkpoint 规则。

语义臂和内容无关臂在每一个视频上必须有**完全相同的 K 多重集**，因此 VideoMAE 输入 observation 总数和 16-frame clip-row 数量完全相同。

fixed-K companion 的每视频总 K 也为 `384n_v`，只是每个窗口都是 K384。

对于 end-to-end cost，冻结预算侦察 prepass 也要对三臂都执行并计时；control 和 fixed companion 即使最终不使用这些 semantic scores，也不得通过跳过这次低成本计算获得人工成本优势。

---

# 6. H65 参考身份的修订

**支持门中的“相对 H65 下降不超过 0.30 点”不再绑定历史 `65.1257`。**

它必须绑定本实验中新跑的 **same-contract fixed K=384 companion**。

理由不是否定 `65.1257`，而是当前正式训练已经把在线随机截断 realization 改为三臂共享的预冻结 60-window realization，并使用新的 variable-K-capable detector/backbone path。历史 H65 仍然是重要只读参考，但不是严格同训练数据合同的因果对照。

因此：

* `65.1257`：historical H65 reference；
* 新 fixed K384 companion：**当前支持门的唯一 H65 comparator**。

不重复训练：

* official dense AdaTAD `68.73`；
* 旧 uniform；
* random-frame；
* PJST；
* UVT；
* Fovea。

共享 official dense 仍只是论文全局参考。

---

# 7. 最小允许代码表面

权威父提交固定为：

`04c35a3b76897e6c1569eeede41ed3aecaf7f854`

实现分支固定为：

`codex/duca-semantic-budget-matched-20260828`

Builder 只允许触碰以下表面：

1. `opentad/datasets/thumos.py`

   * `ThumosPaddingDataset`：增加可选的 epoch-window schedule 和 `set_epoch()`；
   * 不修改普通 H65 无 schedule 时的行为。

2. `opentad/datasets/transforms/end_to_end.py`

   * `LoadFrames`：增加复用现有 random-trunc 几何的 **scheduled random trunc** 路径；
   * 增加极小的窗口预算 lookup/injection；
   * 不复制一套新的 loader。

3. `opentad/models/selectors/duca_online_frame_selector.py`

   * `DucaOnlineFrameSelector` 的 budget resolution；
   * 从 sample metadata 读取冻结的 `K∈{256,384,512}`；
   * hard capacity 为 512；
   * **不启用新的 learned budget controller**；
   * `global_structured_topk`、actionness/transition 学习目标保持原样。

4. `opentad/models/duca/acquisition.py`

   * 只修正 scheduled K 和真实 compute metadata；
   * 禁止新增 selector 或 decoder。

5. `opentad/models/backbones/backbone_wrapper.py`

   * 增加 opt-in variable temporal chunk path；
   * 对 K256/K384/K512 分别真实形成 `16/24/32` 个 16-frame VideoMAE chunks；
   * 替代当前固定 `t1=24`/固定 post-interpolate-384 行为。

6. `opentad/models/detectors/actionformer.py`

   * variable-K bucket execution；
   * detector core helper；
   * counterfactual-teacher 的相同 bucket 语义；
   * variable-length `pad_data`；
   * train/test sample-order restoration。

7. 新增三个极薄配置文件，均继承现有 Stage-2：

   * semantic
   * content-independent control
   * fixed384 companion

8. 一个最小预算生成入口：
   `tools/bata/build_duca_dynamic_budget_window_schedule.py`

9. 两个 focused test 文件即可：

   * window/budget contract；
   * true variable-K detector execution。

**明确禁止修改**：

* `VisionTransformerAdapter` block/adapter 数学；
* ActionFormerHead；
* classification/regression loss；
* NMS；
* THUMOS14 split；
* official evaluator；
* VideoMAE pretrained weights；
* Stage-1 checkpoint；
* H65 actionness/transition learning objective；
* 60-epoch scheduler；
* PJST/UVT/Fovea/连续片段代码。

现有 VideoMAE 本体已经按 16-frame clip 工作，其 adapter temporal dimension 是 clip 内的 8 tubelets，因此没有理由修改 ViT block 数学；变化只应发生在 wrapper 如何组成 `16/24/32` 个 clips。

---

# 8. 真实 K 分桶合同

一次逻辑训练 batch 仍然是 **2 个样本**。

frame selector 先对完整 B=2 batch 运行，得到每个样本自己的 K 和原时间坐标选帧结果；随后 detector 仅按 K 分桶：

`K256 → K384 → K512`

每个非空 bucket：

* 只保留该样本实际的 K 个 RGB observations；
* **在进入 VideoMAE 之前**裁成 K；
* 绝不 pad 到 512 后再送入 backbone；
* VideoMAE clip rows 分别为 `16/24/32`；
* 同一套 VideoMAE 参数、projection、ActionFormerHead 被重复调用。

对于 detector loss，若 bucket 内样本数为 `b_k`，原逻辑 batch 为 `B=2`：

$$
L_{\mathrm{detector}}=\sum_k \frac{b_k}{B}L_k
$$

selector 自己在完整 B=2 上计算的语义损失只加入一次。

整个 `model(...)` 最终仍返回一个 scalar `cost`。现有 train engine 只执行：

* 一次 `optimizer.zero_grad()`；
* 一次总 loss backward；
* 一次 optimizer step；
* 一次 scheduler step；
* 一次 EMA update。

不得把三个 K bucket 变成三次 optimizer update。现有训练引擎本身已经是这个“一 forward / 一逻辑 update”的结构。

Inference 时，bucket 完成后必须按原 batch index 恢复：

* proposals；
* scores；
* metas；

再进入现有 post-processing/evaluator。

### 物理坐标

K bucketing 不改变 H65 原有坐标合同：

* `selected_positions` 始终是原 dense window 中的 original-time indices；
* 一个样本内部的 selected positions 必须严格有序、唯一；
* train GT 只通过现有 selected-axis mapping 转换；
* inference proposals 在 NMS **之前**逆映射回原时间；
* 不能把第 `0…K-1` 个 packed slot 当作均匀物理时间。

### 成本记录

每个真实样本至少记录：

* scheduled K；
* effective K；
* actual `backbone_input_k`；
* VideoMAE 16-frame clip rows；
* bucket；
* `dynamic_compute_realized=True`；
* `padded_to_max_k=False`；
* synchronized VideoMAE CUDA elapsed time；
* full detector elapsed time。

形式上出现 K256、但 VideoMAE 实际收到 512，即直接判协议无效。

---

# 9. 最小可区分测试与 PRE_RUN

PRE_RUN 必须一次性杀死以下错误，不建立新的审计框架。

**窗口总体退化：**一个正式 training video 必须有 60 个 epoch-window identities，预算为精确 `20×256 + 20×384 + 20×512`；不得从当前 minibatch 推导 `n`。这直接杀死 `n=1` 退化。

**跨视频误排名：**改变视频 A 的全部 semantic scores，不得改变视频 B 的任何 K。

**内容泄漏：**冻结窗口 identity 后，semantic budget scorer 的输入不得包含 GT、label、detector prediction 或 cache；控制臂在改变 RGB/scout scores 后 K 必须逐项不变。training GT 只允许用于复现原 H65 random-trunc crop eligibility，不得进入 budget ranking。

**侦察器漂移：**预算生成必须核对 epoch-29、`state_dict_ema` 和上述 SHA；任意改变 Stage-2 scout 参数后，预生成 K schedule 必须 byte-for-byte 不变。

**最大 K padding：**用 spy backbone 分别执行 K256/K384/K512，实际输入必须恰为 256/384/512、clip rows 为 16/24/32；任何 max-K padding 失败。

**K384 companion parity：**同一输入、同一 frozen weights 下，新 variable-capable path 的 K384 forward 必须与原 H65 K384 detector path 做数值 parity；差异只能来自已明确允许的训练 window realization，不能来自 backbone 数学漂移。

**预算匹配：**逐视频检查 semantic/control 的 K multiset 完全一致，三臂总 K 均为 `384n_v`。

**窗口身份漂移：**train runtime `(video,epoch,start,valid_len)` 与 schedule 精确相符；validation/final test `(video,window_start,valid_len)` 与当前 SlidingDataset 精确相符。

**batch/order/梯度：**一个人为构造的 `[K256,K512]` batch 必须验证：

* 输出恢复原样本顺序；
* weighted detector loss 等于两个独立 forward 的 `1/2 + 1/2` 参考；
* optimizer、scheduler、EMA 均只更新一次。

**physical remap：**至少对三种 K 各做一次 selected-axis → original-time proposal round trip，确认 inverse mapping 在 NMS 前完成。

**短视频：**synthetic `valid_len<512` 必须 fail closed；PRE_RUN 同时确认正式 THUMOS14 没有这种窗口。

**统计路径：**在正式训练前用 dummy predictions 跑通 paired-bootstrap 的实际输出目录解析，避免重演 PJST 的 `work/result_detection.json` / `work/gpu1_id0/result_detection.json` 路径错误。PJST 的失败本身只是工程证据，不改变本实验科学结论。

---

# 10. 唯一正式实验

正式实验只有这一个三臂 matched study，不增加第四种模型候选。

| 项目                   | 冻结值                                                                    |
| -------------------- | ---------------------------------------------------------------------- |
| Dataset              | 完整 THUMOS14 official training / validation                             |
| Seed                 | `3407`                                                                 |
| Stage-1 init         | epoch-29 `state_dict_ema`，SHA 如上                                       |
| Stage-2              | 60 epochs                                                              |
| Optimizer updates    | 精确 6000 次成功逻辑 update                                                   |
| Batch                | 2，单 GPU formal training                                                |
| K                    | `{256,384,512}`；fixed companion=`384`                                  |
| Checkpoints          | 至少每 5 epochs                                                           |
| Final model          | epoch-59 terminal `state_dict_ema`                                     |
| Checkpoint selection | 禁止 intermediate best selection                                         |
| Evaluator            | 现有 official THUMOS14 evaluator                                         |
| tIoU                 | `0.3,0.4,0.5,0.6,0.7`                                                  |
| NMS                  | 完全沿用 H65                                                               |
| Result root          | `/data/run01/sczc063/yuzibo/duca_dynamic_budget_window_v001_20260828/` |

主要输出：

* 五个 tIoU 下 mAP；
* Avg-mAP；
* mAP@0.7 单独报告；
* semantic − control；
* semantic − fixed384 companion；
* 10,000 次**整视频配对 bootstrap**，两组比较使用相同 resampling identities；
* 95% paired interval；
* 每视频 K 总量；
* actual VideoMAE input K / clip rows；
* VideoMAE CUDA time；
* end-to-end time；
* peak GPU memory。

## 支持、反驳和协议无效门

**支持当前科学假设**必须同时满足：

1. `semantic − content-independent control` 的 Avg-mAP 95% paired interval **下界 > 0**；
2. `semantic − same-contract fixed384` Avg-mAP **≥ −0.30 pp**；
3. semantic/control 的真实 VideoMAE 工作量逐视频匹配；
4. K256/K384/K512 均真实执行，没有 max-K padding。

**反驳当前预算分配假设**，满足任一即可：

* `semantic − control` paired interval **上界 ≤ 0**；或
* semantic 相对 same-contract fixed384 **低于 −0.30 pp**。

如果 semantic-control interval 跨 0，而 fixed384 retention 通过，则结论为**未决**；不允许因此修改 K、三分之一比例、semantic score、checkpoint 或 validation threshold 后重跑同一正式实验。

以下情况属于**协议无效而不是科学负结果**：

* 预算清单或 scout identity 漂移；
* formal data 出现无法实现 K512 的窗口；
* semantic/control K multiset 不匹配；
* GT/teacher/prediction leakage；
* VideoMAE 实际 padding 到最大 K；
* 不是 6000 次成功 update；
* final checkpoint 不是 epoch-59 EMA；
* official evaluation 或 paired interval 未完整产生。

启动脚本、路径、Slurm 或结果目录等确定性故障不属于上面这些科学失败；Builder/Evaluator 在**同一任务、同一科学合同、同一模型 commit**上最小修正并继续，禁止再次上升为路线讨论。

一个重要解释边界是：即使这个实验成功，三臂的平均 K 都是 384，所以**它不能声称相对固定 K384 减少总 VideoMAE 工作量**。它能证明的是“在相同总重型预算下，语义驱动的跨窗口预算重分配是否比内容无关重分配更有效”，以及 variable-K 是否真实执行。降低平均预算并形成性能—成本优势属于后续问题，不得提前写成当前结果。

---

# `CURRENT_TASK_ORDER`

**当前唯一任务：实现并完成上述“冻结 60 个 H65 random-trunc occurrence + Stage-1 冻结预算排名 + semantic/control/fixed384 三臂真实 variable-K”完整实验。不得同时实现其他动态预算定义。**

**Builder → 独立 Critic → Evaluator，连续执行：**

**Builder** 从 `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 的 `codex/duca-semantic-budget-matched-20260828` 工作，严格限定于上列代码表面。返回 exact commit、三个配置、预算/窗口生成入口、focused tests、K384 parity 和真实 K bucket smoke evidence。截止：**2026-08-29 18:00 北京时间（UTC+8）**。

**独立 Critic** 只审查固定 Builder commit 是否忠实实现：窗口总体、无跨视频排名、无数据泄漏、冻结预算侦察器、H65 live scout 继续训练、真实 variable-K、loss weighting、一次 optimizer update、physical-time remap 和 matched workload。普通代码风格不得成为 blocker。确定性的 claim-preserving 缺陷由 Builder 在本任务内集中最小修复后一次复核。截止：**2026-08-29 23:00 北京时间**。

**Evaluator** 在 Critic 通过的唯一 exact commit 上执行完整 PRE_RUN；通过后直接提交三臂 full THUMOS14 Stage-2，不再发起路线确认。三臂可并行，每臂一张正常 Slurm GPU；随后运行 terminal EMA official evaluation、两组 10,000-draw whole-video paired bootstrap 和实际成本测量。PRE_RUN 与正式提交截止：**2026-08-30 08:00 北京时间**；完整终态返回截止：**2026-09-02 23:00 北京时间**。

依赖仅包括现有 THUMOS14、canonical VideoMAE-S pretrain、上述 Stage-1 checkpoint 和正常 N16R4 Slurm 资源；不需要新数据、新 teacher、新 detector 或新基线。

最终返回物只有科学上必要的内容：**exact implementation commit、三臂 Job/run roots、完整 terminal mAP 表、两组 paired intervals、逐视频 K/workload equality、实际 VideoMAE/end-to-end cost，以及“支持 / 反驳 / 未决 / 协议无效”中的唯一证据结论。**

当前可写入研究记录的只是：**原动态预算训练合同会退化，现已修订为保持 H65 一视频一轮训练分布的跨 60-epoch 窗口总体，并且真正的 variable-K 需要同时修正 VideoMAE wrapper 与 ActionFormer 执行路径。** 当前仍**没有**任何动态预算 mAP 增益、计算节省、优于 official dense `68.73`、多种子稳定性或性能—成本联合优势可以声称。 

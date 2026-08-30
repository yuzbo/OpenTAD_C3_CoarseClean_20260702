ZOOMTOKEN-GRIDFUSE32-L6-G0-CW-BLOCKER-PRO-v001-20260831T050330+0800::g-p-6a79701398bc8191a9ef61db6302b24b

`request_id=PRO_GRIDFUSE32_L6_G0_CONSTRUCTION_WITNESS_BLOCKER_ADJUDICATION-v001`
`exact Project ID=g-p-6a79701398bc8191a9ef61db6302b24b`

**既有 Project Sources：** `PROJECT_CHARTER-v002.md`、`CURRENT_RESEARCH_STATE-v018.md`、`MODEL_EXPERIMENT_HISTORY-v013.md`。其中 Charter 继续约束论文目标与证据标准；后两份 Source 停留在 RACER24 终态，GridFuse 当前事实由本轮请求、终态回执和 pinned GitHub 实现更新。

**实际读取的八个 attachment-only 文件：**

1. `PRO_GRIDFUSE32_L6_G0_CONSTRUCTION_WITNESS_BLOCKER_ADJUDICATION_REQUEST-v001.md`
2. `PAPER_PROGRESS.md`
3. `query_pack.md`
4. `ZOOMTOKEN_GRIDFUSE32_L6_G0_CONSTRUCTION_WITNESS_RPL1_MINIMAL_CHANGE_PLAN-2026-08-31.md`
5. `ZOOMTOKEN_GRIDFUSE32_L6_G0_TERMINAL_PRO_REVIEW_RECEIPT-2026-08-31.md`
6. `ZOOMTOKEN_GRIDFUSE32_L6_G0_CONSTRUCTION_WITNESS_RPL1_TERMINAL_RECEIPT-2026-08-31.md`
7. `ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md`
8. `state.json`

**最新 GitHub 权威入口：**

* repository: [https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702)
* branch: [https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001)
* exact latest implementation commit: [https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b5993faaaa59be318557ca314697e38c4b39b6a1](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b5993faaaa59be318557ca314697e38c4b39b6a1)

GitHub branch 当前确实指向 `b5993faaaa59be318557ca314697e38c4b39b6a1`，其直接父提交为冻结执行基线 `0b734ab839973b2c945b012f066db8222d235bb9`。

**浏览器可见模型与 effort 路由：** `GPT-5.6 Pro`；`effort=Pro`。

---

# 一、总裁决

## `REVISE`

具体处置为：

> **`STOP_CURRENT_8-TUBELET_SEGMENT_G0_PROTOCOL`**
> **`CONTINUE_GRIDFUSE32_L6_ONCE_WITH_PRODUCTION-FULL-WINDOW_ATOMIC_G0`**

角色合同：

> **`REVISE`**

GridFuse32-L6 **不因本次错误而被判为科学失败，也不因“尚无科学结果”而自动获得继续修补权**。当前单 clip、八 tubelet 的 G0 协议永久关闭；允许且只允许一次重新定义后的、真实 384-tubelet Adapter 语义下的最终原子 G0。该最终动作不得修改 GridFuse 模型、Adapter、配置、checkpoint、门槛或训练条件。

若该最终动作出现任何 construction blocker、执行完整性失败或有效 G0 门失败，**精确定义的 GridFuse32-L6 路线永久停止，不再有第四轮修复、replacement、参数调整或 G1/G2**。只有完整 G0 通过时，结果才返回 fresh Pro 决定是否值得训练；G1 不自动开启。

---

# 二、工程、协议与科学证据必须分开

## 1. 工程证据：`PARTIAL_PASS_WITH_DECISIVE_TEST_GAP`

已经成立的工程事实是：

* `b5993faa…` 相对 `0b734ab8…` 只修改了 segment profiler、focused test 和 launcher 三个文件，没有修改模型或配置；
* canonical `opentad.datasets` registry 初始化成功；
* 真实 config 解析成功；
* detector 构造成功；
* epoch-59 `state_dict_ema` 严格加载成功；
* 12 个 VideoMAE blocks 与 final-six Adapter 成功绑定；
* N16R4 focused suites 为 `12/12/8 passed`。

但工程准入并非完整通过。测试套件没有真正覆盖生产 Adapter 的时间轴：

* 关键 GridFuse 单元测试大量使用 `use_adapter=False`；
* backbone ledger 测试显式构造 `total_frames=16`，于是测试 Adapter 的 temporal size 恰好是 `8`；
* “真实 detector”新增测试只做到构造 detector，没有执行生产 Adapter forward；
* launcher 测试只检查是否包含 witness 参数字符串。

因此，`12 passed` 证明了 GridFuse 配对、广播、梯度和小形状 ledger，但**没有证明冻结的生产 config 下 full-window Adapter 能运行**。把这些测试写成“production construction witness 已被测试覆盖”是不成立的。

## 2. 协议证据：`INVALID_CURRENT_G0_SHAPE_PROTOCOL`

当前 profiler 构造的是：

* 一个 attention bucket；
* `8` 个 tubelet；
* 每 tubelet `64` token；
* 总计 `512` token；
* `total_tubelets=8`；
* 然后直接执行 final-six blocks 的 Adapter。

但继承的正式 R1/AdaTAD 配置是：

* `window_size=768` 帧；
* tubelet size 为 `2`；
* 一个正式窗口含 `384` 个 tubelet；
* 16-frame VideoMAE attention clip 只是 `8` 个 tubelet 的 attention bucket；
* 一个窗口共有 `48` 个 attention buckets；
* Adapter 沿整个 `384`-tubelet lineage 工作。

生产 Adapter 的 `forward_native_ragged` 明确要求：

```text
total_tubelets == self.temporal_size
```

不成立时即抛出：

```text
ragged Adapter temporal axis differs from pretrained Adapter
```

因此，旧 G0 把“一个 16-frame attention bucket 的真实形状”错误等同于“包括 dense Adapter 在内的完整生产执行形状”。已有研究记录此前也明确指出：16-frame clip 只是 attention bucket，而 Adapter 耦合全 384-tubelet 时间轴。

这不是一个可以通过把 `8` 改成 `384`、同时仍只提供 512 个 token 来合法修复的小错误；那会把其余 376 个 tubelet 隐式当作不存在或零邻居，仍然不是生产语义。合法协议必须真正构造全部 384 个 tubelet、48 个 attention buckets 和完整 Adapter lineage。

## 3. 科学证据：`NO_NEW_GRIDFUSE_SCIENTIFIC_EVIDENCE`

job `1262099` 在第一条 **dense arm** dry ledger 中终止：

* GridFuse candidate arm 没有开始；
* warmup 没有开始；
* timing、memory 没有开始；
* prediction、metric、gate 没有开始；
* training、resume、parameter update 均没有开始。

所以本轮不能支持以下任一结论：

* GridFuse 快或慢；
* GridFuse 显存好或坏；
* GridFuse 与 R1/Adapter 不兼容；
* GridFuse 会保持或损害准确率；
* GridFuse 应训练或不应训练。

`state.json` 中 `execution_state` 和 GridFuse 子对象正确记录了 blocker，但顶层仍残留 `GRIDFUSE32_L6_G0_REPLACEMENT_BUILDER` 等旧状态字段；这是状态同步不一致，不改变上述证据等级。

---

# 三、第二个 construction blocker 的根本归属

## 主归属：冻结 G0 witness/shape 协议冲突

## 次归属：entry harness 与测试覆盖缺陷

## 不归属：GridFuse 机制与 R1/Adapter config 已被证明不兼容

理由如下。

第一，错误发生在 dense arm，GridFuse pairing、mean fusion、N256 QKV/MLP 和 residual broadcast 都未执行，因此不能归因于 GridFuse 机制。

第二，正式 config 本身没有要求 Adapter 在八 tubelet 上独立运行。GridFuse config 保持每 clip 512 个恢复后的 native token，并要求 dense Adapter；问题来自 profiler 将一个 attention bucket 单独交给全窗口 Adapter。GridFuse config 的“每 clip 512/256”是 bucket 内 Transformer 数量合同，不是 Adapter 的全局 temporal-size 合同。

第三，这也不能简单定性为“普通 entry-harness 一行修复”。从单 bucket 改为 full-window 会改变：

* 输入总 token 数；
* attention bucket 数；
* Adapter 实际工作量；
* 峰值显存；
* kernel launch 和循环行为；
* 最终 p50 测量对象。

也就是说，旧 G0 所测对象本身定义错了。必须由 fresh Pro 重写 G0 协议，而不能由 Codex 在旧 repair 授权下自行调整。

最准确的分类是：

> **`PROTOCOL-CONSTRUCTION_MISMATCH_EMBODIED_AS_HARNESS_AND_TEST_DEFECT`**

---

# 四、GridFuse32-L6 是否继续

## 裁决：继续一次，但不是继续修补旧 G0

GridFuse 值得保留最后一次机会，依据不是“已经投入了很多”，而是：

1. 当前失败完全发生在 dense control，尚未触及候选机制。
2. RACER24 的负结果主要来自 selected-Q/full-KV、completion 和 gather/scatter 路径的高开销；GridFuse 使用固定相邻 pairing、N256 全 QKV/MLP 和直接 residual broadcast，执行结构明显更规整，仍存在通过工程门的合理可能。
3. 一个 full-window final-six G0 所需 GPU 成本远小于 60-epoch 训练，却能直接否定或保留该路线。
4. 不允许任何模型、配置或门槛修改，因此不会演变成结构救援或参数搜索。

但是，连续两个独立 construction blocker 已经耗尽普通修复容忍度。因此下一任务必须是 **最终一次、单提交、无 replacement 的原子动作**。它不能再拆成“GPU witness 失败—修复—Critic—Evaluator—replacement”。

---

# 五、角色合同裁决：`REVISE`

在 `ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md` 的“可执行准入见证”之后，直接增加以下条款：

```markdown
### 连续两个独立 construction blocker 的强制收束

对同一精确定义候选、同一科学门和同一正式执行链：

1. 首个结果盲、预科学 construction blocker 至多获得一次由 fresh Pro
   明确授权的最小修复任务。该任务不得改变候选机制、模型前向、正式
   config、数据、checkpoint、资源语义、测量范围、聚合、门槛或论文主张。

2. 上述最小修复的 exact candidate 若在科学测量开始前又出现第二个独立
   construction blocker，原 repair task 立即永久关闭。Codex 不得继续修改、
   再次请求 Critic/Evaluator、另建 construction witness、提交 replacement，
   或把不同异常拆成无限多个准入循环。

3. fresh Pro 此时只能作出以下二选一裁决：
   - STOP 该精确定义候选；或
   - 当不可变证据证明 blocker 主要属于实验协议而非候选机制时，重新定义
     恰好一个 final atomic task。该任务必须保持模型、正式 config、数据、
     checkpoint 和原科学门不变，只删除已被证明错误的协议假设。

4. final atomic task 只允许一个 scheduler submission，不设 replacement。
   同一作业必须先执行不计时、不记录显存、不产生 prediction/metric 的
   fail-closed production construction phase；只有该 phase 完整通过后，才在
   同一进程和同一作业中进入冻结测量。任何 construction、OOM、完整性、
   walltime 或 gate failure 都终止该精确定义候选，不再修复或重提。

5. final atomic task 通过只意味着返回 fresh Pro 复盘资格，不自动开启训练、
   下一 gate、额外 seed、full-stack replay 或论文主张。
```

同步写入 `RTK.md` 的简短原则：

```text
同一候选连续两个独立 construction blocker 后退出普通 repair 模式：
fresh Pro 只能 STOP，或授权一次无 replacement 的 production-faithful
atomic witness+measurement；该动作任一失败即终止精确定义候选。
```

---

# 六、唯一下一任务

## `ZOOMTOKEN-GRIDFUSE32-L6-PRODUCTION-FULLWINDOW-ATOMIC-G0-v001`

### 科学目的

在不训练、不修改模型、不读取性能结果调参的前提下，判断精确定义 GridFuse32-L6 在 **真实 384-tubelet Adapter 时间轴、48 个 VideoMAE attention buckets** 下，是否具备进入准确率训练所需的最低工程 Pareto 潜力。

### 唯一可证伪预测

与同一模型、同一 checkpoint、同一 full-window lineage 的 dense final-six blocks 相比，GridFuse final-six blocks 必须同时满足：

* `dense_p50 / gridfuse_p50 >= 1.35`
* `gridfuse_peak_allocated / dense_peak_allocated <= 1.05`
* `gridfuse_peak_reserved / dense_peak_reserved <= 1.05`

`p95` 只报告，不作门。

### 正确的 full-window shape

必须实际构造：

* batch size：`1`
* tubelets：`384`
* tokens per tubelet：`64`
* full native carrier：`24,576` tokens
* attention buckets：`48`
* dense bucket size：`512`
* candidate merged bucket size：`256`
* Adapter temporal size：`384`
* blocks：`6–11`
* dtype：现有 fp16 autocast
* dense 与 candidate 共用完全相同的 input tensor 和 lineage

禁止只提供 8 个 tubelet 后谎报 `total_tubelets=384`；禁止补零、dummy、padding 或缺失 tubelet。384 个 tubelet 必须全部具有合法、严格递增的 native lineage。支持矩形可使用确定性循环覆盖九个合法 8×8 位置，但两个 arm 必须完全相同。

### dry-ledger 必须精确成立

Dense final six：

```text
attention_bucket_calls = 6 × 48 = 288
attention_tokens       = 6 × 48 × 512 = 147456
kv_tokens              = 147456
attention_pairs        = 6 × 48 × 512 × 512 = 75497472
mlp_tokens             = 147456
adapter_tokens         = 6 × 384 × 64 = 147456
```

GridFuse final six：

```text
gridfuse_bucket_calls  = 6 × 48 = 288
attention_tokens       = 6 × 48 × 256 = 73728
kv_tokens              = 73728
attention_pairs        = 6 × 48 × 256 × 256 = 18874368
mlp_tokens             = 73728
adapter_tokens         = 147456
restored_native_tokens = 24576 per block
```

任一 ledger 不一致即 fail closed，不能计时。

### 允许修改的科学代码文件

仅允许：

1. `tools/bata/profile_zoomtoken_gridfuse32_l6_segment.py`
2. `tests/test_zoomtoken_gridfuse32_l6.py`
3. `scripts/run_zoomtoken_gridfuse32_l6_gated_n16r4.sh`

角色条款同步可另行只修改：

* `docs/aris/ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md`
* `RTK.md`

### 禁止事项

不得修改：

* `opentad/models/backbones/vit_adapter.py`
* GridFuse config 或 R1 base config
* Adapter `temporal_size`
* checkpoint 或加载语义
* pairing、fusion、broadcast completion
* blocks `0–5` / `6–11` 划分
* fp16、GPU/CPU 资源、门槛
* dataset、detector、evaluator、NMS
* G1/G2、训练、prediction、metric、energy 或 full-stack 代码
* compile、CUDA graph、candidate-only optimization
* warmup `100` 或 timed iterations `500`
* 任何 K、层数、pairing orientation 或阈值 sweep

### Builder 必须增加的测试

1. 生产 config 的 Adapter temporal size 明确为 `384`。
2. 旧 `total_tubelets=8` 生产 Adapter 调用必须稳定复现拒绝。
3. 小 embed-dim、但 `total_frames=768` 的 full-window 384-tubelet Adapter+GridFuse forward 必须通过。
4. 48 个 bucket 完整覆盖 24,576 个 token，无重叠、无遗漏、无 padding。
5. dense/candidate full-window ledger 精确等于上述值。
6. 既有 GridFuse、R1 regression、strict-rectangle suites 全部继续通过。
7. launcher 的正式路径只能提交一个 atomic witness+measurement job。

### Critic 范围

Critic 只判断：

* 是否真正执行 384-tubelet Adapter；
* 是否偷偷用八 tubelet 加伪 `total_tubelets=384`；
* 两 arm 输入、lineage、dtype、block、Adapter 是否完全匹配；
* 是否修改了模型/config/checkpoint/gate；
* ledger 和计时边界是否正确；
* 是否存在 candidate-only compile 或隐藏预热不公平。

只有 `PASS` 才进入 Evaluator。

### Evaluator 范围

Evaluator保持结果盲，只核验：

* exact clean/pushed commit；
* branch/commit 可由 GitHub API 验证；
* exact config 与 epoch-59 R1 `state_dict_ema`；
* 唯一 result root、job name 和 formal ordinal；
* 1 GPU、4 CPU、2 小时；
* atomic 作业先 witness、后 measurement；
* 无独立 GPU witness job、无 replacement；
* 终态无论成功或失败都写 exclusive terminal receipt。

唯一准入字符串：

```text
PRE_RUN_READY_ATOMIC_FULLWINDOW_G0
```

### 正式动作

* `1262090` 永久保留为 formal G0 scheduler ordinal 1。
* `1262099` 永久保留为 construction-witness blocker，不是性能 measurement attempt。
* 新任务只允许 **formal G0 scheduler ordinal 2 / scientific measurement attempt ordinal 1**。
* 该 ordinal 2 是 GridFuse 的最终 formal submission。
* 不存在 ordinal 3，也不存在 wrapper replacement、resume、requeue 或 rerun。

单个作业内部顺序必须是：

```text
Git/config/checkpoint identity
→ detector construction
→ strict EMA load
→ 384-tubelet full-window dry ledger
→ 100 warmups per arm
→ 500 alternating timed measurements per arm
→ peak memory
→ gate
→ exclusive profile/terminal receipt
```

---

# 七、接受、停止与 blocker-return 判据

## 接受

只有以下全部成立才记为：

```text
GRIDFUSE32_L6_PRODUCTION_FULLWINDOW_G0_PASS_PENDING_FRESH_PRO
```

条件：

* construction phase 完整通过；
* full-window ledger 完整通过；
* 100/500 协议完整；
* profile 与 terminal receipt 完整；
* speedup `>=1.35x`；
* allocated/reserved 两项均 `<=1.05`。

接受后仍然：

* 不训练；
* 不启动 G1；
* 不提交 G2；
* 只在 `PT1H` 内进入 fresh exact-Project Pro。

## 有效门失败

若 measurement 完整但任一性能门失败：

```text
STOP_GRIDFUSE32_L6_EXACT_ROUTE_VALID_G0_NEGATIVE
```

永久禁止：

* G1/G2；
* pairing/K/层数/门槛 sweep；
* 第三次 G0；
* residual 或 learned fusion 救援；
* 额外 microbenchmark。

这只否定精确定义 GridFuse32-L6，不外推为全部 token merging 或固定空间聚合失败。

## construction 或执行 blocker

若发生 shape、OOM、CUDA、checkpoint、ledger、artifact、walltime 等 blocker：

```text
STOP_GRIDFUSE32_L6_EXACT_ROUTE_FINAL_EXECUTION_BLOCKER
```

这是工程/协议终止，不冒充科学性能负结果；但由于已经是最终 atomic task，仍永久关闭该精确定义路线，不再修复。

## 队列 blocker

若正式 ordinal 2 已提交但截至冻结期限仍未启动，立即返回 fresh Pro 作为客观 scheduler blocker；不得取消后另投新 job，也不得创建其他实验。

---

# 八、精确北京时间期限

```text
role_rules_sync_due_at=2026-08-31T06:00:00+08:00
builder_plan_due_at=2026-08-31T06:15:00+08:00
builder_candidate_due_at=2026-08-31T09:00:00+08:00
critic_due_at=2026-08-31T10:00:00+08:00
evaluator_due_at=2026-08-31T10:45:00+08:00
formal_action_due_at=2026-08-31T11:30:00+08:00
queue_start_or_blocker_due_at=2026-09-01T08:00:00+08:00
queue_blocker_return_due_at=2026-09-01T08:30:00+08:00
scientific_return_due_at=within PT1H of terminal, no later than 2026-09-01T12:00:00+08:00
```

---

# 可直接摄取的执行块

```text
decision=REVISE
disposition=STOP_CURRENT_8_TUBELET_SEGMENT_G0_PROTOCOL__CONTINUE_ONCE_WITH_PRODUCTION_FULLWINDOW_ATOMIC_G0
selected_task=ZOOMTOKEN-GRIDFUSE32-L6-PRODUCTION-FULLWINDOW-ATOMIC-G0-v001
scientific_purpose=Test exact GridFuse32-L6 minimum engineering Pareto under the production 384-tubelet Adapter temporal axis and 48 real VideoMAE attention buckets before any training.
blocker_root=PROTOCOL_CONSTRUCTION_MISMATCH_EMBODIED_AS_HARNESS_AND_TEST_DEFECT
gridfuse_mechanism_scientifically_tested=false
role_contract_decision=REVISE
execution_base=b5993faaaa59be318557ca314697e38c4b39b6a1
branch=codex/zoomtoken-gridfuse32-l6-v001
allowed_scientific_paths=tools/bata/profile_zoomtoken_gridfuse32_l6_segment.py;tests/test_zoomtoken_gridfuse32_l6.py;scripts/run_zoomtoken_gridfuse32_l6_gated_n16r4.sh
allowed_rule_paths=docs/aris/ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md;RTK.md
forbidden_changes=vit_adapter.py;all model/config/data/checkpoint/gate changes;Adapter temporal_size changes;fake 384-tubelet metadata;dummy/padding;G1;G2;training;prediction;metric;energy;full-stack;compile;K/layer/pairing/threshold sweeps
production_shape=B1,T384,K64,N24576,48_buckets_x_512
candidate_shape=48_buckets_x_256_with_N24576_restored_before_each_dense_Adapter
warmup_per_arm=100
timed_iterations_per_arm=500
g0_p50_speedup_min=1.35
g0_peak_allocated_ratio_max=1.05
g0_peak_reserved_ratio_max=1.05
critic_required_verdict=PASS
evaluator_required_verdict=PRE_RUN_READY_ATOMIC_FULLWINDOW_G0
formal_scheduler_ordinal=2
scientific_measurement_attempt_ordinal=1
formal_submission_limit=1
replacement_allowed=false
g1_allowed=false
g2_allowed=false
acceptance_rule=Complete production-full-window construction and ledger, complete 100/500 measurement, p50 speedup>=1.35, allocated<=1.05, reserved<=1.05.
stop_rule=Any construction/execution/artifact blocker or any valid G0 gate failure permanently stops exact GridFuse32-L6; no repair, retry, sweep, training, G1 or G2.
post_result_pro_trigger=Every terminal outcome or objective blocker, before any other experiment.
role_rules_sync_due_at=2026-08-31T06:00:00+08:00
builder_plan_due_at=2026-08-31T06:15:00+08:00
builder_candidate_due_at=2026-08-31T09:00:00+08:00
critic_due_at=2026-08-31T10:00:00+08:00
evaluator_due_at=2026-08-31T10:45:00+08:00
formal_action_due_at=2026-08-31T11:30:00+08:00
queue_start_or_blocker_due_at=2026-09-01T08:00:00+08:00
queue_blocker_return_due_at=2026-09-01T08:30:00+08:00
scientific_return_due_at=PT1H_AFTER_TERMINAL_AND_NOT_LATER_THAN_2026-09-01T12:00:00+08:00
next_owner=Codex Builder
```

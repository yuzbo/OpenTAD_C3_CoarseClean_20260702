---
type: route_consolidation
node_id: route:sparsehead-canonical
title: "SparseHead 唯一路线合并与旧仓封存"
status: tested
verification: v16_job_completed_suite_validated_pro_analysis_completed_claim_limited
canonical_repository: "OpenTAD_C3_CoarseClean_20260702"
canonical_branch_at_merge: "codex/duca-total60-plugin-cvpr-20260727"
consolidation_base: "63a726a4aaf48ecbf6780bb196de43a890c6b4df"
archived_repository: "OpenTAD_SparseHeadClean_20260702"
archived_head: "dce2c66d1053d53dfcc40b051399cd4c2ecde9ad"
updated: 2026-07-29
---

# SparseHead 唯一路线合并与旧仓封存

## 路线所有权

自 2026-07-28 起，当前仓库是 SparseHead/PhysTime 稀疏检测头的唯一可写研究面。
`E:\DeskTop\TAD\OpenTAD_SparseHeadClean_20260702` 被定义为只读历史档案；不得再在该目录
新增实现、配置、实验或 claim。它与当前仓没有共同 Git 祖先，因此本轮没有做 merge 或
cherry-pick，而是按文件语义和证据边界选择性吸收。当前工作区已有大量 DUCA/wiki 用户改动，
本轮没有整树覆盖、提交或重写这些改动。

路线科学身份必须分成两层：

1. native-J192 `physical-metric` ActionFormer 是当前最强、已经完成 full60 的经验基线；
2. `SupportDecoupledPhysicalQueryHead`（SDPQ）是唯一允许继续裁决的稀疏头结构候选：
   完整物理 query 网格与稀疏 observation support 解耦，使用 support pooling /
   relative attention / null evidence 与 signed center-width regression。

“唯一可写路线”只表示代码与研究记忆的所有权，不表示 SDPQ 已经超过
physical-metric，也不表示任何 SparseHead 版本已经 `paper_ready`。

## 两个来源的比较与吸收

| 来源 | 可用价值 | 本轮处理 | 不得继承的结论 |
| --- | --- | --- | --- |
| 旧 SparseHead 仓 `dce2c66` 加本地脏改动 | irregular projection/FPN、point generator、bridge head、native-axis 合同、hard GT coverage/balance fallback、assignment audit | 核心实现选择性保留；audit 迁入 `tools/bata/` 并只调整仓库根路径；建立两个 fail-closed 诊断配置与 consolidation test | 旧 bridge 不是 dense-equivalent，也不是论文主方法；未提交 repair 配置没有训练证据 |
| PhysTime 历史分支 `codex/phystime-performance-diagnosis-20260712@e05f6231` | raw/native geometry、PhysTime detector/projection/head、SDPQ、严格 padding isolation、matched 配置/门禁/实验记录 | 恢复主实现、注册面、配置、gate/launcher、focused tests 和关键 experiment nodes | branch tip 的 decode-cross dtype 修复没有新结果；不能把它写成 confound 已关闭 |

旧仓在封存时有 3 个 tracked 修改和 22 个 untracked 文件；其 binary diff digest 为
`f3e4d66044cc64c634c45d408c412c8a55ea0d6a`。有价值的 dirty bridge 文件已经按
SHA-256 `fb05c10491ddfa6c85ca5183878eee80b33a77b57a27ee74fbac9328e2222a2e`
吸收；旧 audit 仅因从 `tools/` 移到 `tools/bata/` 而修改 `ROOT` 层级。17 个历史
repair/retrain config 与 6 个 `remote_runs/` launcher 不迁入：它们重复、未验证，并会误导
未来直接启动旧路线。其 16 个 config、6 个 launcher 文件名和 dirty digest 保留在本节点与
source registry 中。

## 当前代码面

主候选：

- `opentad/models/dense_heads/support_decoupled_physical_query_head.py`
- `opentad/models/detectors/phystime_tad.py`
- `opentad/models/projections/phystime_projection.py`
- `opentad/models/utils/phystime_geometry.py`
- `opentad/models/utils/native_temporal_geometry.py`
- `opentad/datasets/transforms/phystime.py`
- `opentad/datasets/transforms/phystime_raw.py`
- `configs/adatad/thumos/phystime_g1b_sdpq_pool_native_j192.py`
- `tools/bata/run_phystime_g1b_sdpq_real_gate.py`

matched controls 与诊断面：

- `selected_axis_adatad_sparse_k384.py`
- `physical_grid_adatad_sparse_k384.py`
- `phystime_adatad_sparse_k384.py`
- `phystime_tad_i3d_feature_gate0b.py`
- `phystime_sdpq_i3d_feature_gate0b.py`
- `sparsehead_irregular_bridge_k384_baseline.py`
- `sparsehead_irregular_bridge_k384_balanced_repair.py`
- `tools/bata/audit_sparse_head_assignment.py`

两个 irregular-bridge 配置都显式设置
`diagnostic_only=True`、`primary_result_allowed=False`、
`metric_claim_allowed=False` 和 `slurm_allowed=False`。balanced repair 只能验证
assignment coverage 假设，不能直接升级为训练臂。

VideoMAE strict temporal padding isolation 已合入现有 adapter/backbone wrapper。它与
ChronoTransport 或 packed routing 同时开启时 fail closed，避免把 ChronoTransport
动态刷新语义误当作 SparseHead padding mask。

## 本轮验证

- SparseHead consolidation contracts：`3 passed, 1 skipped`；Windows 跳过项是
  真实 tensor balanced-assignment 检查。
- 仓库要求的 C3 focused checks：`20 passed`。
- 所有新增/修改 SparseHead、PhysTime、backbone、config、gate Python 文件通过
  `py_compile`；三个 Slurm shell 通过 `bash -n`。
- selected/physical/PhysTime/SDPQ 五个配置可由 `mmengine.Config` 完整解析；
  两个 legacy diagnostic config 的 fail-closed 字段再次验证通过。
- graph `edges.jsonl` 的 227 行均为合法 JSON 且包含关系必需字段。
- 六个 Torch-dependent PhysTime suites 在收集期统一被本机
  `torch/lib/c10.dll` 的 `WinError 1114` 阻断，没有运行任何模型断言。这是
  `RTK.md` 已登记的本机环境限制；它们现已在 N16R4 Linux OpenTAD 环境补跑。
- evidence-first 合并后的最终隔离包
  `e4814e3544784b3608c007a11946464b4f597e0fbf9a23a5910e3b0171bef388`
  在 N16R4 通过 3 个配置解析、核心导入闭包、全树 `compileall`、3 个启动器
  `bash -n` 和 `59 passed` focused tests。这里没有 GPU/CUDA forward、正式
  replay 或新指标，因此验证状态仍低于 `empirically_supported`。

## 可继承的实验事实

| 实验 | 结果 | 正确裁决 |
| --- | --- | --- |
| raw-video K384 v1 full60 | selected-axis 63.61；physical-grid 59.14；PhysTime 57.21 | v1 同时改变 head/geometry/objective，PhysTime 整体主张被否定 |
| native-J192 G1a 6 epochs | selected 10.26；physical-metric 10.56 | 只证明可运行；物理 anchor 对短动作 assignment 不友好 |
| SDPQ 6 epochs | 10.17 | gate/pilot 通过，不是性能证据 |
| native-J192 matched 20 epochs | selected 30.42；physical-metric 44.88；SDPQ 30.88 | physical-metric `+14.46`；当前 SDPQ 不优于 matched survivor |
| native-J192 matched full60 | selected EMA 41.28；physical-metric EMA 57.57 | physical-metric 单种子经验支持；该 full60 没有 SDPQ 臂 |
| P0 full-precision NMS replay | selected 41.283021；physical 57.608685；差值 `+16.325664 pp` | rounding/NMS 影响只有约 `-0.0366..+0.0338 pp`，主差异不是后处理舍入造成 |

full60 来源为 commit `0dc5851a8feb12b97d16bdb5ea8fc60e9273d132` /
tree `bddc9b9386604d00d213275a47ce7997b35d3f4c`，Jobs
`1170945/1170946/1170947`。P0 来源为 commit
`c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c`，Jobs
`1174688--1174693`。两者都是 THUMOS 单数据集、单 seed，不支持论文最终主张。

decode-cross replay 仍是 `implemented`：旧 gate 捕获了 selected online/EMA 与
physical online，physical EMA 未执行；validator 又发现源 FP16 score 在 CPU
sort/top-k 前被转为 FP32。`e05f6231` 的 source-dtype 修复没有新 gate/job/result，
所以不得写成“跨解码器混淆已关闭”。

## 下一次唯一有效的推进顺序

1. 把当前已通过 Linux CPU focused tests 的合并工作区冻结成唯一精确提交，再运行
   selected/physical × online/EMA 四条件真实 CUDA gate；CPU 测试不能替代真实
   raw-video forward、native parity 和 THUMOS evaluator。
2. 先复现 physical-metric survivor，并把它设为所有 SDPQ 改动的 matched control。
3. SDPQ 必须报告 all-GT/short-GT assignment、support/boundary observability、null-evidence
   使用、high-tIoU 和端到端真实成本；不能只看 training loss。
4. 只有 SDPQ 在相同 K384/J192、相同 seed/更新预算/评价器下超过 physical-metric
   matched medium gate，才允许 full60；否则保留为负结构消融。
5. full60 优势仍需未参与开发的多种子、第二 detector/数据集与完整成本才能升级为
   `paper_ready`。

## Approach A 完整链部署

2026-07-28 已把 evidence-first decode-cross 链冻结为远端独立干净快照
`8e31b9e3c08b0a8d320e031b04dfd63e19eb08df` /
`aae5503424aa3925ef99bba851d600a03e3c3377`，运行根为
`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260728_v4`。
full-content preflight 已通过，manifest SHA-256 为
`3551816b8e056b9afea4fc9ee8575f525e78ffba64ff087915130b2e10e54712`。

正式 Slurm Job `1201048` 在一个 GPU allocation 内按 gate → 四条件 replay →
suite 串行 fail-closed 执行，但在第一次模型构造时暴露 consolidation 漏恢复的
ActionFormer native-geometry constructor 合同，以 `FAILED 1:0` 终止。它没有
产生 gate/replay/suite 或 mAP 结果。

按同一实验协议恢复历史 native-J192 ActionFormer 对齐，并新增 constructor/config
与 runtime 消费回归测试；Linux focused suite `74 passed`。新 clean runtime
commit/tree 为 `0338f4777bd02fb327573ef716f54fec76d4af0e` /
`cb98c64c17d2983c22181d4908c4f31024a82a2f`，preflight SHA-256
`77b9918aa3173b73fc71d821defa8c14b3165de1b35f0ae4c0382eeb5d21b43d`。
唯一替代 Job `1201317` 已开始运行，gate 前置 `41 passed`；四条件 real-CUDA
gate 的 `gate_pass`、native direct exact equivalence 与 raw tensor immutability
全部通过。`selected_online` direct inference 完成后显示 Avg-mAP `41.26`，
mAP@0.3--0.7 为 `64.50 / 56.39 / 42.66 / 27.82 / 14.90`，但 launcher
随后因未生成 `pre_cross_window_detections.json.gz` 而 fail closed。失败签名为
`direct_postprocessing_artifact_producer_contract_missing_v1`；其余三条件与
suite 未开始，因此这些 direct 数值只能作诊断，不能称正式 replay 或新结果。

2026-07-29 已恢复历史 direct post-processing artifact producer contract，并加入
端到端回归；模型、配置、epoch-59 权重、seed、数据和评价器均未改变。新 clean
runtime commit/tree 为 `ac326ffdc97652433b55ccc596e734b112f51806` /
`0c58027756997995bda0de6fdd8ec0deb49966d3`，Linux focused suite
`75 passed`，preflight SHA-256 为
`97fe5af28b2647396c052c9bdf956997d98e264af74432b57e0fc983b071fb91`。
唯一后继 Job `1201469`（`ptdc-a1-r2`）正在运行；路线保持
`experiment_running`，不是 `empirically_supported`。v6 四条件 real-CUDA
gate 也已通过：`gate_pass=true`、native direct exact equivalence 全通过、
四项 raw tensor immutable；gate artifact SHA-256 为
`775e1f2dae70b7863324fd9d235712195dca4d0846968b3bd5e55b754e7b3ea4`。
`selected_online` 全量 direct inference 已完成，精确 Avg-mAP
`0.4125660433077075`，mAP@0.3--0.7 为
`0.6450446628552113 / 0.5638932489689005 / 0.4266348135535575 /
0.27820781407261164 / 0.14904967708825695`。v6 已实际生成并核验 pre-cross、
post-processing audit、evaluation metrics 与 result artifact，证明 producer 修复
越过 v5 失败点。当前执行 `selected_online` 双轴 replay，尚无 completion/suite/
final cross-decode verdict。

Job `1201469` 后续在 selected-online replay producer 完成后由 validator
fail closed。uniform-rank Avg-mAP 为 `0.4125660433077075`，physical-time
cross-decode Avg-mAP 为 `0.5015355102106833`，但 completion 组装引用未绑定
`numeric_precision`，失败签名为
`decode_cross_validator_numeric_precision_scope_v1`。另三条件和 suite 未启动，
因此 `+8.89694669029758 pp` 只是一条件诊断，不是正式路线结论。

修复后的 commit/tree 为 `1631d0b60f6552a6f5eb0378d74e766850f34ffd` /
`f485c8708e22bbbf9a73063d5293a20bc4aa658f`。v7 通过 `76` 项精确恢复测试和
preflight，但部署元数据把 focused-log 哈希漏掉最后一位，因而在
`sbatch --test-only` 前停止；没有产生 Slurm 作业，v7 根已封存。全新的 v8
runtime/run root 复跑 `76 passed` 与 full-content preflight，preflight SHA-256
为 `e9f36c221156e5411dad5e3bfe43508b4aa59310539fdbe24da985fc99a27d53`。
`1201494` 仅为 test-only；唯一正式 Job `1201495`（`ptdc-a1-r4`）正在
`g0024`/RTX4090 运行。路线状态仍为 `experiment_running`。

v8 gate focused tests 为 `43 passed`；四条件 real-CUDA gate 也已通过，
artifact SHA-256
`5e323e5ccdedd7dd39d70148aed7108beca94bb5952125a124ad20accfd634f6`，
并再次确认 native/direct exact equivalence 与四组 raw tensor immutable。当前
执行 `selected_online` direct inference，尚无 v8 formal completion/suite。

v8 selected-online direct inference 已完成并复现精确 Avg-mAP
`0.4125660433077075`；pre-cross/audit/metrics/result 四类 producer 合同均存在，
pre-cross SHA-256
`b4adcf545655424d2b2dfdfce0d107109c5010850143fadf925706fb3de60322`。
双轴 replay 当前由 uniform-rank 转入 physical-time mode，尚不能升级为 formal
completion 或路线结果。

v8 随后完成了四个双轴 replay，但 Job `1201495` 在最终 suite 以
`FAILED 1:0` 终止。四 completion 的 frozen tensor、native/direct 和 reviewed-P0
parity 均为真；唯一失败是 producer 把零 `fatal_log_findings` 序列化为 `{}`，
consumer 要求 `[]`。失败签名为
`decode_cross_completion_fatal_log_findings_container_type_v1`。四条件八行指标
已保存在 decode-cross experiment 节点，但在 suite 未通过前全部保持
`diagnostic_only`，不能作为路线正/负结论。

v9 因 repository-local Git author 缺失在 commit/preflight/Slurm 前停止，签名
`runtime_git_author_identity_missing_v1`，零作业，根已封存。唯一正式后继为
v10 commit/tree
`c878fbe3a5e960671f03d93fff8367ed3414f5c5` /
`8d3e73bb26544d1bcf7bfb61154d0b003f2658e0`；Linux `77 passed`，preflight
SHA-256 `f46f6299f7fccc899140ad8fdf001052772ef550dd34cdb68c17d5ba5fc59a8f`。
`1203046` 仅为 test-only，唯一 Job `1203047`（`ptdc-a1-r5`）在
`g0050`/RTX4090 `RUNNING`，gate focused tests 已 `44 passed`，四条件
real-CUDA gate 也已通过（SHA-256
`e5516af02289d15dd1465f5387471bb1a3c357873980d22645c08acbf6aa141c`）。
selected-online 已形成首个 v10 正式 component completion（SHA-256
`a4e727cf094127be7b91a4a13b140463ad9dc3e0c8c1bcfa3acb9887b5ff6dda`），
producer completion SHA-256 为
`8a2d38db8a2130a8b617940361a8637dfdc0bff3b6947b0f35d75167a809bfa6`。
validator 通过，`fatal_log_findings=[]`，frozen-raw/native-direct/P0 parity
均通过。uniform / physical decode Avg-mAP 分别为
`0.4125660433077075 / 0.5015355102106833`。

selected-EMA 第二个 component 也已通过，completion / producer SHA-256 为
`0c6f87617b1cbd6a5bc4a6be6e9a5a2174f8a5a568c2f24db7253c15a315b8dc` /
`ddddd42174eb987cdeb723ae4422df8105e773bd7af74d31e67760dba20d74ff`；
契约字段同样全真且 `fatal_log_findings=[]`。其 uniform / physical decode
Avg-mAP 为 `0.41283020792762315 / 0.5009785403306161`。两个 selected-axis
component 均只到 `tested`，尚不是路线结论。physical-online 随后也形成
第三个正式 component：completion / producer SHA-256 为
`02384da2c71c93bdcd6ce003cd59451510c9d095e222653202f09f38b73b153f` /
`b9ba401a92e0d828aeabe48cb8972df74a64720a12f160d939daa355856aaf58`，
全部契约字段通过且 `fatal_log_findings=[]`。其 uniform / physical decode
Avg-mAP 为 `0.40107677185286417 / 0.5755558109390063`，physical decode
增益为 `+17.447903908614215` pp。Job 已串行进入 physical-EMA direct
inference。此次只修复证据 container 类型及显式错误标记扫描，SparseHead
唯一路线的模型协议未改变；总状态仍为 `experiment_running`。

physical-EMA direct/native physical-time inference 随后完成，Avg-mAP
`0.5760868491267752`，mAP@0.3–0.7 为
`0.7721224901972557/0.7045574192938243/0.6257613932435541/
0.4900660583199814/0.28792688457926047`。最后一次双轴 replay/validator
仍在运行，尚无 completion，所以该行保持 `diagnostic_only`，路线状态仍为
`experiment_running`。

### v10 terminal -> v16 unique formal successor

上述 physical-EMA 中间状态已被正式 completion 取代：completion / producer
SHA-256 为
`a5c0c5248bf196d17f1cbf4f11a61d01459cb2ff3cfbf37541046fdb508b7ad1` /
`8433bd22b620cd60300d94289cf991b69c1f64bcd5eacea557fbc463d7981086`，
uniform / physical Avg-mAP 为
`0.40296498031949024 / 0.5760868491267752`，全部 replay 合同通过。

但 v10 Job `1203047` 最终 `FAILED 1:0`：suite 把同一 checkpoint 的两种
合法 artifact-record schema 做整字典比较，误报 binding mismatch。签名
`decode_cross_suite_checkpoint_binding_schema_shape_mismatch_v1`，suite log /
failure receipt SHA-256 为
`68b7b3d34e587392bdac2df1eb2a36d971009d4c07165ef2a18157449ccb931f` /
`42c394f11153a862819876b3915c34ca2ef0a68b6b62ed78a121d65db4269cec`。
这仍是工程证据 consumer 失败，不是 SparseHead 模型负结果。

v11–v15 分别因 profile+nounset/mode、精确测试调用范围、preflight
`PYTHONPATH`、finalizer `$BASE` token、SSH transport interruption 在 Slurm 前
停止；五个根均为零作业并已封存。唯一正式后继 v16 commit/tree 为
`54e7f9abeaabf710a505f0a0f595a4eb3bb47f98` /
`f8490f9c25c2e0e6958c406e19c83cc3d5a40535`，Linux `78 passed`，preflight
SHA-256
`ccc7a83e27b8d18ad0892b644e7338667b72d8eba3e3feedbc387dc4ac1d9a0d`。
`1203916` 仅为 test-only；Job `1203917`（`ptdc-a1-r11`）在
`g0045`/RTX4090 `RUNNING`。v16 gate focused tests `45 passed`，四条件
real-CUDA gate artifact SHA-256
`0d2153effee84a0e1aa6410125bb291eb4ef4d41e4b40604f49d9e5868e0ada9`；
gate、native/direct exact equivalence 与四项 raw-tensor immutability 均通过。
当前进入 selected-online direct inference，hard-failure scan 为空，尚无
completion/suite。SparseHead 仍是本仓库唯一稀疏头路线，状态为
`experiment_running`，不能提前声称 `empirically_supported`。

v16 `selected_online` 随后完成 direct、双轴 replay producer 和正式 validator。
uniform/native
Avg-mAP 为 `0.4125660433077075`，physical-time Avg-mAP 为
`0.5015355102106833`，差 `+8.89694669029758 pp`；producer completion
SHA-256 为
`97410d9855a3f6db859e36213bf6b201e10c96941a164b5588af02cdfba4ee20`，
正式 `DECODE_CROSS_COMPLETE.json` SHA-256 为
`6937fc6b7b050fd7009ee967ceef446aebaa8b3daa695c7959106ff87048c038`；
`validation_pass=true`、`fatal_log_findings=[]`、frozen-raw/native-direct/
reviewed-P0 parity 均通过。当前已进入 `selected_ema`；另三条件与 suite
未完成，所以路线仍为 `experiment_running`，单个 `tested` component 不得升级为
最终正负结论。

v16 `selected_ema` 随后也完成 direct、双轴 replay producer 与正式 validator。
uniform/native 与 physical-time Avg-mAP 为
`0.41283020792762315 / 0.5009785403306161`，差
`+8.814833240299292 pp`；producer / formal completion SHA-256 为
`43c737fe3c5a9a534c565bf63e419fa152ee35b3be796ddf3f601c954fa52877` /
`4a1b405b7849f396e1b649da8895070e6176023c4a959c6d7fd9148f2bd8afe0`，
同样通过全部 frozen-raw/native-direct/reviewed-P0/fatal-log 合同。
Job `1203917` 已进入 `physical_online`。当前只有两个正式 `tested`
component，另两条件与 explicit suite 未完成，路线继续保持
`experiment_running`。

`physical_online` 随后完成 direct 与双轴 replay producer。physical/native 与
uniform-rank Avg-mAP 为
`0.5755558109390063 / 0.40107677185286417`，差
`+17.447903908614215 pp`；direct/physical、uniform 与 producer completion
SHA-256 分别为
`b68f2ad1393b59c40d58f7cfa1e450a52f84d8acbc80ad785a2d3a31352d6009` /
`0c258e563fe7b9886e6d56c9c3370b6536e187b521526318622b07ffcf1e4a4b` /
`d61d8fbf8b977b59b65eb87d55227904b2a5a2e6994e584226bda19a265b26eb`。
正式 validator receipt 仍未写，因此这不是第三个 `tested` component，只能是
`diagnostic_only` producer evidence；路线仍为 `experiment_running`。

随后正式 `physical_online/DECODE_CROSS_COMPLETE.json` 已生成，SHA-256
`fd18348e6ae6ecf4bdc4390ca4620a109616582f7f77138ed137085e0df6c260`；
`status=tested`、`validation_pass=true`、`fatal_log_findings=[]` 与
frozen-raw/native-direct/reviewed-P0 parity 均通过。它现在是第三个正式
component，Job `1203917` 已进入 `physical_ema`。第四条件与 explicit suite
仍未完成，路线继续保持 `experiment_running`。

`physical_ema/DECODE_CROSS_COMPLETE.json` 现已生成，SHA-256
`cd6da2f827524e0b9eb2b46c6cbbcc5b6e89243aa9cd8d7e45efafcb4cb6b565`，
producer completion SHA-256
`aa6356a509898b94a38f2b9e0548c5f647cc6498655697b37fd39ea8982fc733`。
uniform/physical Avg-mAP 为
`0.40296498031949024 / 0.5760868491267752`，差
`+17.312186880728497 pp`；两轴 mAP@0.3--0.7 分别为
`0.622154649489393 / 0.5316588686305871 / 0.4113769771975965 /
0.2880843206041682 / 0.16155008567570622` 与
`0.7721224901972557 / 0.7045574192938243 / 0.6257613932435541 /
0.4900660583199814 / 0.28792688457926047`。全部 validator 合同通过，
这是第四个正式 `tested` component。Job `1203917` 仍在运行且 explicit suite
尚未落盘，故唯一路线继续为 `experiment_running`，不得提前作最终模型归因。

### v16 终态、路线裁决与后续唯一方向

Job `1203917` 已 `COMPLETED 0:0`（`02:34:30`）。suite completion /
validated SHA-256 为
`ed2770c35cf9a3acd5fa80465eda1c34b3541ba3dea404c75388aaeffefbdc31` /
`f2da143127b3a01aef7bda451e2351c494f72552f3810f604f895f4c0a7767d3`；
两者均 `validation_pass=true`，completion 为 `status=tested`、
`fatal_findings=[]`，完整绑定 preflight、gate、P0、四个 completion 与
checkpoint state identity。SparseHead 唯一路线因此从
`experiment_running` 升级为 `tested`，但不是 `empirically_supported` 或
`paper_ready`。

四个冻结条件的 physical-minus-uniform Avg 增益为
`+8.8969/+8.8148/+17.4479/+17.3122 pp`，online/EMA 近乎重合，高 tIoU 与
short-action proposal recall 同向改善。不存在四条件内部矛盾；它们共同说明
selected-rank decode 是严重 confound，physical-time-before-NMS 应成为后续
SparseHead 的固定几何基线。

这项结果不等于 SparseHead/SDPQ 获得正结论。matched 20-epoch
selected/physical/SDPQ `30.42/44.88/30.88` 仍显示 SDPQ 未恢复；固定 physical
decode 时 physical checkpoint 比 selected checkpoint 仍高约
`7.40--7.51 pp`，而跨 checkpoint 差只能描述，不能称训练因果。当前最大缺口是
independent evaluator、assignment/support observability、class/calibration/
NMS/failure-sample 分解、多 seed 与完整成本。

唯一路线的下一步固定为：先做 sealed-artifact 独立 mapper/NMS/GT evaluator，
再做 64-window assignment/support audit；随后才允许 SDPQ micro-overfit
机制门禁、多 seed 和成本 ledger。负结果分析本身不授权静默改核心算法或重训。
旧仓 `OpenTAD_SparseHeadClean_20260702` 继续封存，本仓仍是唯一维护面。

## 2026-07-29 paper-comparability closure

The sole SparseHead route now has a tested diagnostic-closure implementation
at commit/tree
`57917e7bf2b991478b4f6fc4ce1db5ca5878b68d` /
`aaf7c82bd837078bb7276baf6c0a504da0684194`; four focused suites pass
(`35 passed`). It adds:

- independent v16 NumPy/float64 decode/NMS/AP recomputation;
- a sealed 64-window SDPQ support/assignment audit;
- a fail-closed official ActionFormer record builder and comparability
  classifier.

The route's benchmark boundary is now explicit. v16 raw-VideoMAE/K384 is
`diagnostic_only`; historical `63.61` is `external_reference_only`. A
paper-main-table anchor must be regenerated from pinned official ActionFormer
commit `61ea7eb9308a568b0cf45e3804830836e30061de`, released I3D/THUMOS bytes,
official config/evaluator, raw predictions and independent metric
recomputation. Matched method results must use the same official protocol and
exactly one declared intervention. No new model result or retraining is
claimed by the local implementation.

## 2026-07-29 official-comparability execution status

The diagnostic branch is now clean and pushed at commit/tree
`6d74ad7b7c7736bbff48976a626b951512a54e96` /
`80cd2431ebf9809f03ab1216b84b45380d51f33b`; Linux focused verification is
`58 passed, 1 skipped`. Two independent-recompute failures were implementation
contracts, not model outcomes: valid-prefix versus NaN-padding scope, then
logical `test` versus OpenTAD annotation subset `validation`. Both old roots
are preserved; the fresh v3 independent recomputation is running.

Official ActionFormer resources were not synthesized from existing pickles.
The exact released THUMOS archive was downloaded and verified at MD5
`375f76ffbf7447af1035e694971ec9b2`; the released checkpoint/log package
SHA-256 is
`e028f7e487713d0c68f0515ba9bdafda0ed05fc1271b9999ea995652b034c929`.
The first download root records missing proxy export and the second records
home-cache quota exhaustion; the complete resources live in immutable v3.
Extraction into the gitignored data/pretrained surface of the clean upstream
runtime is running. No official `paper_main_table` record exists until
save-only raw predictions, official evaluation, independent evaluator replay
and the strict record builder all pass.

The exact SDPQ epoch-19 online support audit is diagnostic-only. Its first
formal Job `1204961` failed in the POSIX wrapper before touching CUDA; successor
Job `1204981` is pending with an explicit Bash launcher. This does not upgrade
SDPQ and does not authorize training. The route remains `tested`.

## 2026-07-29 official-paper gate sealed

Commit/tree `2b074845497f6ada3314cb895f0d4ab2f4ce3eca` /
`7779862c5422dc8e527b304bf881a760b0c90625` passed the exact Linux focused
suite (`95 passed, 1 skipped`). It converts official comparability from a
dataset-name convention into a content-verified contract: pinned upstream Git
bytes, canonical classes, nominal/evaluated split distinction, every I3D file,
exact raw prediction video set, official log, independent evaluator and
strict receipts. Candidate method rows additionally require a live
official-base source-diff attestation with exactly one declared intervention.

Job `1205131` supplied a strong diagnostic official anchor at Avg-mAP `66.83`
but failed only the superseded nominal-versus-evaluated split schema. It is not
yet paper-eligible. Unique clean Job `1205178` is running the complete official
chain; only its explicit `main_table_eligible=true` verdict may promote the
anchor.

Job `1205132` exposed a padded-query masking omission in SDPQ support
observability. The fix is padding-only, tested, and deployed as unique
diagnostic successor Job `1205179`; it does not rescue the matched SDPQ score
or authorize retraining. Independent replay Job `1205133` remains active.
Accordingly the canonical route stays `tested`/`experiment_running`, not
`empirically_supported` or `paper_ready`. Existing VideoMAE/K384 and historical
numbers remain attribution context, never official main-table comparisons.

## 2026-07-29 official-anchor reseal and exact independent replay

The released official ActionFormer checkpoint was evaluated by Job `1205206`
with the pinned official source and evaluator. It produced mAP@0.3–0.7
`82.133988/77.805571/70.953608/59.401673/43.872118` and Avg-mAP
`66.833392`. These numerical outputs are real, but the old record asserted
seed `0` while the official config and training log use seed `1234567891`.
Therefore this is an unsealed reproduction source, not yet a paper row. A
fresh record must bind the true seed, full effective config, raw predictions
and all fifteen strict receipts before promotion.

The official-comparability gate is now hardened at commit/tree
`e2a0d74f561b158c531d4909e72ecee69b153c16` /
`0b6cb7996ee90f3209a78b78bbf7a55525e3badd`. Its Linux exact suite passed
`127 passed, 2 skipped` (log SHA-256
`115dd497a3a662b3fc0f19ae9104257d245cbadbb7fd4001f3eb3ea71432534c`).
It validates a live official-base diff, forbids rename/copy/delete and
undeclared source edits, expands both effective configs and requires every
protected protocol field to remain equal.

Independent recomputation Job `1205243` failed the exact closure with
`independent_recompute_semantic_match_drift_v1`. Exact raw scores, masks,
proposal geometry and every metric-delta sign were preserved; the mismatch
came from the validator using stable NumPy/float64 ordering and exponential
semantics instead of production PyTorch `2.0.1` unstable CPU sort plus scalar
float32 C++ Soft-NMS. This is an engineering semantics failure, not negative
model evidence. Commit `e2a0d74` ports those pinned semantics independently,
including the `expf` bit probe, without importing production decode/NMS or
evaluator code. Job `1205388` failed before validator startup because its
non-login allocation lacked the shell `module` function; signature
`slurm_module_function_unavailable_v1` is engineering-only. Unique clean
successor Job `1205400` directly activates the pinned environment and is
`experiment_running`; exact detection and `1e-4` aggregate thresholds are
unchanged.

The next official candidate is deliberately not a rank-coordinate gather or a
fixed 768-window crop. The paper-comparable K384 design must retain the
official full-video temporal grid, GT mapping, point generator, decoder,
30-epoch schedule, seed, loss, NMS and evaluator, and change only a
deterministic original-grid max-K384 compute mask/scatter intervention.
Actual stage-level compute must be measured; zero filling without skipped
computation is not a valid efficiency claim. The route remains
`experiment_running`, not `empirically_supported` or `paper_ready`.

Official-anchor Job `1205409` failed before inference because its environment
probe imported the NMS extension before `torch` loaded `libc10.so`; signature
`official_environment_probe_nms_import_order_v1` is engineering-only.
Successor Job `1205419` completed official inference and reproduced Avg-mAP
`66.83`, then failed the record gate because the released train log omits the
official loader default `model.fpn_start_level=0`. Signature
`official_released_train_log_default_serialization_omission_v1` and failure
receipt SHA-256
`079818253bc87a78ed67ce41dbd092aa64f0e54b5a61972f2313adeb7d10fa4a`
are preserved; the output is not a paper row.

Commit/tree `8b80c98ee2af65561bf305b4fdc2ef16e460da73` /
`148a93eac4ff1b6a3be46fdca72c705aa17294a6` now attests exactly that one
upstream-default omission while pinning the raw released-log config hash and
requiring normalized equality with the full source-expanded config. It does
not relax any model, schedule, seed, data, NMS, evaluator or receipt field.
The v18 clean Linux preflight passed `131 passed, 2 skipped` (log SHA-256
`6899bf6126d1ce9b3d880d348cdf5c1f152235d3b2e6f6de028b5fc807fb34fb`).
Unique fresh anchor Job `1205455` completed `0:0` under v3 root; `1205454` is
test-only. All 15 receipts pass and its strict verdict is
`official_actionformer_protocol_match=true`, `main_table_eligible=true`.
Independent mAP@0.3–0.7 is
`82.133988/77.805571/70.953608/59.401673/43.872118`, Avg `66.833392`;
completion/protocol/verdict SHA-256 values are `90c8bae1...b9e94` /
`808199b5...5bd95` / `0706247e...a2ec`. The official dense comparator is now
`empirically_supported` and replaces historical `63.61` for all paper claims.

K384 implementation is unblocked but remains `designed`, not tested. Its
primary matched control is deterministic stratified-uniform support over the
original per-level physical FPN grid; video-hash-fixed random support is only a
secondary robustness control. Selected centers use the exact radius-3 halo
needed by the three kernel-3 classification/regression head layers; outputs
must be scattered to original indices before unchanged loss, decoder and NMS.
Only actually skipped query-head computation counts as savings, and the
official 30-epoch seed-`1234567891` EMA/evaluator contract remains protected.
The overall route remains `experiment_running`.

## 2026-07-29 official-native implementation and comparability correction

The earlier phrase “unchanged loss” is superseded for the executable K384
candidate. Sparse execution returns logits only on selected physical queries,
so the formal training objective necessarily uses those same selected queries
as its classification/regression support. The method is therefore registered
as one compound but reviewable intervention:
`native-grid K384 head-query policy + selected-query loss support`. It must
never be described as an execution-only change or as input-frame sampling.
The backbone still observes every official I3D feature, hence
`input.observation_budget=null`; the separate model field is
`model.query_budget=384`.

Candidate commit/tree
`55763a9ef7ce18a51827fe48040081c4fe2b84d4` /
`c489a54aa501b39421cddb5df98385b3889ed479` implements true three-layer
head skipping with original-index scatter. It strict-loads the official state
dict and leaves all official model parameters/state keys unchanged. The
primary policy is deterministic stratified-uniform; video-hash random remains
secondary. Linux candidate tests pass `11/11`.

Audit commit/tree
`aab72e484538931a565930b99d1beb71f47b9ceb` /
`25e7e0eb3b8cd5edfb48eac594eda6b89edffa36` now separates input observation
from head-query budget, requires the selected-query loss contract, pins the
official config/config-loader/checkpoint identities, checks prefix-invalid
masks and uses three interleaved CUDA timing rounds. Linux audit and launcher
tests pass `40/40` and `5/5`; live source-diff SHA-256 is
`409ffd3035a0c957d3b250db24fe017c5c09efda526d746ace0d54f00c695abc`.

Several immutable preflight failures are engineering-only: a first invocation
named a stale test path (`audit_preflight_test_path_drift_v1`); non-login
profile attempts exposed missing `module` and profile-before-`nounset`
ordering; one GitHub HTTP/2 remote-ref fetch failed and one HTTPS live check
hung until terminated. HTTP/1.1 live `ls-remote` then verified exact official
and candidate refs, and fresh v5/v6 preflights passed. None produced model
metrics.

Job `1205541` is the only formal real-CUDA gate (`1205539` is test-only).
It is currently pending, not failed. Even a pass authorizes only the next
inference-only diagnostic and matched dense/sparse training pair. A paper row
still requires the exact official 35-loop implementation semantics, explicit
terminal checkpoint selection, EMA evaluation, full result receipts and a
strict `main_table_eligible=true` verdict. Route status remains
`experiment_running`.

## 2026-07-29 first CUDA cost verdict and exact recovery

The prior pending statement is superseded: Job `1205541` failed only the
isolated cost criterion. All numerical/native-grid contracts passed, but
sparse-with-selector required `20.5046 ms` versus dense `6.1918 ms`
(`0.3009x`, every round about `0.30x`) despite a theoretical head-MAC fraction
of `0.3359`. This proves that the first Python gather/scatter implementation
was not an efficient sparse execution kernel. It says nothing about model
accuracy. Preserved signature:
`native_grid_sparse_head_microkernel_launch_and_scatter_slowdown_v1`.

The exact recovery at candidate commit/tree
`d64e66dfd7fc9881552b342f5523926cc78c0848` /
`16265c70b235034acb52521b00c259ec6d8b59e1` batches samples and FPN levels so
each three-layer classification/regression head launches one Conv1d per layer.
It does not change K384, policy, physical grid, selected-query loss support,
official config/checkpoint, evaluator or evidence boundary. Linux tests pass
`12/12`; audit tests pass `40/40`; launcher test passes `1/1`; live source-diff
SHA-256 is
`5aea817bf1fd1b2c0e36193b9d99ee71bde3dfd00c05673ece5dc4f6da9304d4`.

Job `1205567` is the unique successor real-CUDA gate (`1205566` is test-only),
currently `PENDING (Priority)`, run root
`.../runs/actionformer_native_grid_k384_cuda_gate_20260730_v2`. No training or
paper metric is authorized until correctness and the unchanged `>=1.05x`
selector-inclusive three-round threshold both pass. Route status:
`experiment_running`.

## 2026-07-29 packed-path convergence and final execution hypothesis

The pending statement above is superseded. Job `1205567` completed
`FAILED 2:0`: all correctness contracts passed, while dense/sparse-preselected/
sparse-with-selector means were `6.2409/12.7657/13.6182 ms` and speedup was
`0.4590x`. Its preserved signature is
`native_grid_sparse_head_packed_patch_materialization_and_microconv_slowdown_v1`.

Commit/tree `31e6112ea28747098cfe5412c097d737731bfaa1` /
`d2619cd075c4e7192ca060f34d811ac3fe5768f8` substituted an exactly equivalent
flattened GEMM for the packed Conv1d. Job `1205569` also completed
`FAILED 2:0`: correctness passed, but means were
`6.1934/12.7646/13.5469 ms`, speedup `0.4577x`, and every round remained near
`0.458x`. Failure signature:
`native_grid_sparse_head_packed_gather_scatter_overhead_v1`. This convergence
shows that convolution arithmetic is not the dominant latency; repeated
planning, patch materialization and intermediate dense scatter are.

The exact candidate is clean at remote v6 from SHA-verified bundle
`c50bea0b79e242bb4c96cf11fb35a3ef095a8b9c3bc4a13fc56abca02be4ec49`.
Preserved transport/runtime signatures are
`github_https_clone_tls_termination_v1` and
`bundle_clone_remote_head_unset_v1`. Remote GitHub ref DNS also failed as
`github_remote_ref_dns_timeout_during_source_diff_v1`; the local live-ref
attestation/provenance
`3ef485f82678453538aef6f58ba81d548149394ef93356a811593e67cdf22e9d` /
`780c0aa5a8a00ba9180974d4bee001782e83d747d492589a2a2da4b5bc40e2d6`
is valid only for engineering and explicitly cannot seal a paper row.

The sole remaining execution hypothesis is a shared global packed sparse
state with no intermediate dense scatter. It is `designed`, not yet `tested`.
If its selector-inclusive isolated-head speed remains below `1.0x`, further
Python/PyTorch sparse-head micro-optimization stops and the implementation is
rejected. If it passes the unchanged `>=1.05x` formal gate, the next evidence
is an official same-commit matched dense/sparse retraining pair; released-
checkpoint inference remains `diagnostic_only`. Main-table eligibility still
requires terminal epoch-35 EMA, identical official schedule/data/evaluator,
independent source/evaluator receipts, multiple preregistered seeds and
end-to-end cost.

The bounded prototype is now `tested` at candidate `d86a4ac...` and audit
`14bd14f...`. It uses one zero-guarded global physical axis, one shared
dependency/index plan and only final dense scatter. Candidate/audit focused
suites pass `14/18`; exact clean runtimes are v7/v22. Full-content preflight
SHA-256 is
`08b05123edbaccd10d5b43031a43ebac11a3616ceb454bfbd588d4d7395a6a95`.
Unique Job `1205571` (`1205570` test-only) completed `0:0` under the fresh v4
gate root. Exact selected-output equivalence passed and selector-inclusive
median latency improved from dense `6.240573 ms` to `3.970906 ms`, a
`1.571574x` speedup; the three synchronized-round speedups were
`1.573072/1.570395/1.568621x`. Gate/completion SHA-256 values are
`cddfb80af237a41d3c3e1121e39cbc5114ad8abc472c56f6daf519a50cf95988` /
`ceec00f799eb40a1dd56c1949576783e06599205d63f1d1909a598787d99fd85`.
This advances only the implementation to `tested`: it is an isolated-head
CUDA gate with no model-metric or end-to-end-cost claim authority. Official
same-commit dense/sparse training is now authorized; main-table eligibility
still requires the terminal epoch-35 EMA matched pair, sealed evaluator/data
receipts, preregistered independent seeds and synchronized end-to-end cost.

The first official matched pair is now `experiment_running`. Remote live
source-diff SHA-256
`a07d038d87632d1f8cc984ba24af44ca7ce9a9902e30e501f5de80a32265d46b`
binds the published official/candidate refs, and the live 413-file I3D
manifest exactly reproduces official SHA-256
`cda269dace393b9af1f6fcb87a9a531beed69e3c71279ba3ca2cee76e198d59a`.
Audit commit `643c42e...` runs dense then K384 sparse in one allocation, with
the same candidate commit, official seed `1234567891`, official 5+30 schedule,
terminal epoch-35 EMA, data, NMS and evaluator; raw predictions from each arm
are independently re-evaluated by the pinned official source.

Unique Job `1205573` (`1205572` test-only) is pending at immutable root
`.../actionformer_official_matched_pair_k384_seed1234567891_20260730_v1`.
Preflight/deployment/submission SHA-256:
`3b827cfe10b3267d013373f89a9c3b90b2eb6f450b0aa4b7d1e5082615a0ac4e` /
`65cb544960c619f4243c7829a41950719d2591493c05fbad70a07f1b9a037da2` /
`ead6f35af71e2de9308d6ed0aad642dc27845e68169f1cee8ca32e3d157a3e77`.
This is single-seed screening, never a final paper row. Sparse training changes
the selected-query loss support and is explicitly a method intervention, not
an execution-only comparison. Preserved preflight-only signatures:
`official_data_live_revalidation_import_scope_v1` and
`preflight_failure_receipt_python_environment_unloaded_v1`; neither started
training.

Parent matched Job `1205573` failed in source identity before training because
the compute node could not resolve GitHub during a redundant live-ref lookup;
signature
`compute_node_github_dns_during_redundant_live_source_diff_revalidation_v1`.
Its failure/runtime SHA-256 values are
`f0bf8fe6258260d55fffe88d35dfb75d647340adccb06dc2efae1c5e419c64d9` /
`8bc85a66f37bc98eec780ec76ef5fab1978bd45195917780c269267dc5b2a057`;
no model iteration ran.

Audit successor `debbde4...` separates the proof correctly: the live published
refs stay sealed in the preflight receipt, while Slurm recomputes all local
source/config/diff/allowlist content offline. Remote recovery tests pass
`19/19`, and actual offline validation passes. A GitHub TLS failure left v24
preserved; v25 was produced from verified bundle SHA-256
`6c59f1d568017d8ee82e32d3132b595b73c3d469a2cb91976968d330cd789104`.
`1205579` is test-only; unique Job `1205580` is pending under fresh v2 root.
Recovery preflight/deployment/submission SHA-256:
`45e60ba0f68132b8cfa11ec036ed71789e83d718dc300df62f0cdf19f1375e8a` /
`a151cf03c67395771eb386c6fe48687e867b40df1d8f7a562be6d1df459125a0` /
`fca38a1cad01222ef8bda967116993742319bdc94b2d8e9582a783abe21c479f`.
No scientific condition changed.

Job `1205580` subsequently failed before training because the official
launcher could not import its declared TensorBoard dependency. Preserve
signature `official_declared_tensorboard_dependency_missing_v1` and failure
receipt SHA-256
`a959ef415f383d5368edf806b1166cca9cd25e91e49ea4398853775059e35385`;
this is not a negative model result. The repair creates a sealed isolated
TensorBoard `2.20.0` venv while retaining Python `3.10.20`, PyTorch `2.0.1`,
CUDA `11.8` and NumPy `1.23.5`. Environment receipt SHA-256 is
`acc5909360970cfad1f390a4f5ab046a3876ac9378448b2f94da26ffb312ece2`.

Audit commit/tree `a3d9879...` / `51c5377...`, exact v26 bundle
`e8812a84489bb55aea419b1b637778574539a44b0c7399b18a04d346430ce419`
and remote `19/19` tests seal the dependency recovery. Fresh
preflight/deployment/submission SHA-256 values are
`9ff27367e10717b012d0f06a85b980f54c9b91a6fe45be9e8f87c00cac90d47b` /
`00736c6b07fff77e0a6ca92ad24744eab0e2c089a22b350f9f2537054891b4f4` /
`f4512010b2d675611f97e61a929ee4edda421b7f29506969d49028b3a7ac041a`.
`1205583` is test-only; unique Job `1205584` is running on g0024 at the fresh
v3 root. Source/environment probes and `14+4` focused tests passed before the
dense arm began. This remains official-protocol single-seed screening, not a
paper row; multiseed paired accuracy and synchronized end-to-end cost remain
mandatory.

The main-table protocol was frozen before the screening metric. It fixes five
paired seeds (seed-set SHA-256
`a4038a752aa46b97e5854c20574d65ece078bad6124e4778cc4269e75747c7c6`),
terminal epoch-35 EMA, seed-level paired uncertainty, Avg/high-IoU
non-inferiority, a preregistered full-vs-selected training support x
dense-vs-K384 evaluation-query attribution, and synchronized detector-pipeline
cost. The cost boundary begins at official precomputed I3D features and ends
at serialized detections; it must not be called raw-video end-to-end.

S0 continuation bounds are `Delta Avg>=-1.00 pp` and both high-IoU deltas
`>=-1.50 pp`. A paper accuracy-preserving efficiency claim requires five
complete pairs, Avg CI lower bound `>=-0.20 pp`, mAP@0.6 and @0.7 lower bounds
`>=-0.50 pp`, and detector-pipeline median speedup `>=1.05x` with CI lower
bound `>1.00x` in aggregate and no duration stratum crossing unity. Status of
this protocol is `designed`; Job `1205584` remains only
`experiment_running` screening.

Job `1205584` then failed after dense completed all 35 epochs but before the
first EMA metric. The clean candidate had no local official NMS extension and
loaded OpenTAD's same-named nine-argument module, while official ActionFormer
calls seven arguments. Signature:
`official_actionformer_softnms_extension_abi_shadowed_by_opentad_v9arg_v1`;
failure-analysis SHA-256
`99f83a03715fa935a422451f9fe842aeaae867546d37c9af39cda8869958f852`.
Sparse never started, so there is no matched/model result. The dense epoch-35
checkpoint remains immutable diagnostic evidence and is forbidden as a
successor resume source.

## 2026-07-30 official seven-argument NMS recovery and S0 relaunch

The ABI failure is repaired at audit commit/tree
`71f955a7301f07875a35e0be366241e548e5c775` /
`d328093644e040741e16dbdd8bc93b6b0d608a10`. Exact v27 bundle SHA-256 is
`a9ee267333c9371d087e806fe61cef19c14122b18fee1a4e6c75fa4c58846ad6`.
The isolated runtime/environment receipt/official NMS extension SHA-256 values
are
`/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_official_runtime_20260730_v2` /
`13d57c1161905f059204f7101f26029503a03da7f5eb44b81c418a0b97999f24` /
`b67e0e41f9f55cd69e8b90cfc75a1947214365857d851a510047838ad49ed98d`.
Candidate/audit remote tests pass `14+5`.

Before deployment, all 413 official I3D files were reloaded and rehashed;
IDs, bytes, shapes and dtypes exactly reproduce manifest SHA-256
`cda269dace393b9af1f6fcb87a9a531beed69e3c71279ba3ca2cee76e198d59a`.
The candidate dense config is byte-identical to upstream official commit
`61ea7eb...` at SHA-256
`c0ac0df560cd564941b56cd9391ad0bd5cea386d2e4b6cf9fc8ffcab821955cd`;
upstream itself uses THUMOS `validation` for training and `test` for
evaluation. Generic loader defaults are not the official experiment config.

Preflight/deployment/submission SHA-256 values are
`d9e1f897de51e46aac52cb450f72daa8bc19a64bf999b01112013489038d4a55` /
`b8d4079c9ddc8faa7a0a575dbe63f700c2448409df5dbccf972101cc0e4a282b` /
`2a31a1d01056f39159d17d99fb9047f5bd6946b68475c1eae31008659df07a08`.
`1205593` is test-only. Unique Job `1205594` is pending under fresh v4 root;
normal priority wait is not a failure. Its status is `experiment_running` and
`paper_main_table_eligible=false`. It retrains dense and K384 from scratch
under the same official seed/schedule/EMA/evaluator and provides only the
predeclared S0 decision for the five-seed study.

Job `1205594` failed in four seconds before tests/training because its Python
probe imported the official extension before PyTorch loaded `libc10.so`.
This is the recurring engineering signature
`official_environment_probe_nms_import_order_v1`, not a model result. Failure
analysis receipt SHA-256 is
`06bbc29e5f57b3b9a12f421f5ddd814487bf01733d0f0e5bbcc4c0551c877a41`.
The deeper missing contract was import order, which the earlier ABI/path test
did not assert.

Audit recovery commit/tree
`98f5b875315b4a2b5c6829f5d74ccce68f478e47` /
`2e6b4bba6868c323d70c97140f7cbed044eb1a7b` adds that focused regression
and loads PyTorch first. Clean v28 bundle SHA-256 is
`713a1d839e8e8ea50f141df9dba1feb44dc43c91dffbd4dd85bf8910bbdf9e24`.
No model/config/protocol field changed; remote exact tests must pass before
one fresh successor is submitted.

The ordered import probe and official seven-argument call now pass remotely
(SHA-256
`7d79381ed64b27059aa6f4204bbfce3f606fc1e81e0a7962e4e1d1c7413a0488`);
candidate/audit suites pass `14+5`. New preflight/deployment/submission
SHA-256 values are
`19230f06e0eda57c34607db250dba9ebc1f0d6365e5ab33c339dffe0468ddd86` /
`250068a1de36c00fabe37596e302dc9e3fd22249be09b267fc4e9762e6f4ce46` /
`0549ff04a30bb4efea176a484a6f51d652b8bdd023227564b0fc2fdfe492cabf`.
`1205598` is test-only; Job `1205599` is the sole v5 formal successor and is
pending. It remains official-comparable single-seed screening, not a paper
row.

Job `1205599` is now `RUNNING` on g0030. Its in-allocation gates passed and
dense training reached epoch 24 with finite loss; no metric or completed arm
exists yet.

Dense is now a formal `tested` component: exact 212-video/42,400-prediction
independent Avg-mAP is `66.583013`, with mAP@0.3–0.7
`81.908495/77.952035/71.285498/58.255505/43.513530`. ARM completion SHA-256
is `a15b0526ef9a75a0fe32c0798b609c738781ab5c063c53df165ace6cbcdf138a`.
Sparse training has begun; this dense row alone is not a model verdict or
paper row.

## 2026-07-30 official S0 verdict: current K384 intervention rejected

Job `1205599` is terminal `COMPLETED 0:0`. The matched pair, both arm
completions, terminal EMA checkpoints, raw predictions and pinned-official
independent recomputations all validate. Pair SHA-256 is
`545e420aa1d437aedeffd15cb30390ceb0cfe4d6565d7eb35c53a8bf17ac76fd`.

The same-commit dense/sparse Avg-mAP values are `66.583013/43.919699`; their
mAP@0.3–0.7 values are
`81.908495/77.952035/71.285498/58.255505/43.513530` and
`64.925248/56.642845/45.952641/32.783177/19.294586`. The complete intervention
loses `22.663313 pp` Avg-mAP, `25.472328 pp` at 0.6 and `24.218944 pp` at 0.7,
far outside the preregistered `-1.00/-1.50/-1.50 pp` continuation bounds.

Consequently, `KILL_CURRENT_K384_SELECTED_LOSS_INTERVENTION` is frozen. This is
an `empirically_supported` rejection of that exact method intervention, not an
engineering failure and not a rejection of every sparse-head design.
Five-seed/cost expansion is forbidden. Because S0 was explicitly
`paper_main_table_eligible=false`, these numbers can only be reported as a
fully receipted negative screening/ablation result. The route now enters
no-retraining 2x2 attribution and negative diagnostics before any redesigned
method is authorized.

## 2026-07-30 route closure: hard K384 killed, DCSR designed

The no-retraining 2x2 Job `1205701` completed and isolates the failure:

| training support | dense eval | K384 eval |
|---|---:|---:|
| full | 66.583013 | 45.784332 |
| selected | 64.537343 | 43.919699 |

K384 execution contributes `-20.7082 pp` Avg-mAP on average, selected-loss
training `-1.9552 pp`, with only `+0.1810 pp` interaction. The selected-trained
checkpoint is therefore largely recoverable under dense proposal coverage,
while the full-trained checkpoint still fails under K384.

Terminal assignment/support audit Job `1205799` further shows only
`16.9423%` positive retention, `395/804` sampled official-training GT with no
candidate and `427/804` with no assignment. Post-NMS class-agnostic and
fixed-topK recall also collapse. Together these results make structural
proposal/support deletion the dominant explanation; selected-loss and score
calibration are secondary. All diagnostic artifacts are
`paper_main_table_eligible=false`.

The unique SparseHead route is now DCSR: dense cheap native-grid scaffold plus
sparse expensive residual refinement. Unselected queries retain scaffold
outputs; dense proposal and supervision support are never zeroed. Per-FPN
coverage floors and a length-adaptive residual budget are mandatory design
elements. DCSR is `designed`, not implemented/tested.

Its official-comparable preregistration is
`research-wiki/experiments/actionformer-sparsehead-dcsr-official-prereg-20260730.md`.
It requires internal validation-only method selection, five fixed paired
official seeds, same-run dense controls, paired uncertainty, and synchronized
feature-to-final-detection cost including scaffold, selector, refinement,
scatter, decoder and NMS. Only simultaneous accuracy and cost-gate passage can
authorize a positive main-table efficiency claim.

The archived `OpenTAD_SparseHeadClean_20260702` route must not be revived as an
independent line. Its useful evidence is absorbed here; the exact hard-K384
failure remains immutable negative memory.

## 2026-07-30 DCSR implementation and G1 launch

DCSR is now implemented and tested on the unique SparseHead branch. Exact
commit/tree is
`bf0df83d7400c89fc61f38d169d68085420a2263` /
`2f9346fcfd2bfb7fc5a76a86ef65545030a67469`. G0 uses an official-dense
identity scaffold only to prove routing equivalence; G1 uses the actual cheap
one-layer dense scaffold plus uniform K384 three-layer residual refinement.
This distinction prevents an identity test from being misreported as a cost
or performance result.

Real-CUDA G0 Job `1206168` completed `0:0` with all native-grid,
pre-decode and final official Soft-NMS/timestamp values exact. Formal
validation-only G1 array `1206273_[0-2]` is running over three frozen
development seeds. The 160/40 holdout is derived only from official
`validation`; G1 remains forbidden from paper performance tables.

The next decision is mechanical: aggregate the three same-seed dense/DCSR
pairs against the preregistered Avg/@0.6/@0.7 bounds. Failure triggers deep
model analysis and route termination/redesign discussion; passage unlocks
G2--G4 internal design, not official claims. Five paired official seeds and
complete synchronized cost remain mandatory after architecture freeze.

## 2026-07-30 DCSR G1 verdict and SparseHead route termination

Formal G1 array `1206273_[0-2]` completed for all three frozen development
seeds. Aggregate receipt SHA-256 is
`b98d59468ef39aa6fe6de387adfd6f872c848ab8f63b26c3bf1bf6161f5f7939`.
Dense/DCSR mean Avg-mAP is `0.5680730871/0.4925110665`; paired deltas are
`-7.556202 pp` Avg and `-11.043134/-11.019821 pp` at 0.6/0.7. All seeds and
all thresholds are negative, so the frozen G1 non-inferiority gate is false.

No-training counterfactual diagnostics completed under exact diagnostic
commit/tree
`8d6f6e5e7fcf8c27b6aa46870bc4c0b242f6314b` /
`1ac5a68c6b8d0b1c9028ea3154765ae20e87622a`. They show:

- scaffold-only minus dense: `-7.418076 pp`;
- all-query residual minus dense: `-6.316665 pp`;
- all-query residual value over scaffold: `+1.101411 pp`;
- K384 support penalty versus all-query residual: `-1.239537 pp`, including
  `-2.4102/-2.3511 pp` at 0.6/0.7.

The leading observed failure is therefore the weak one-layer
scaffold/decomposition, with fixed K384 residual support a smaller
high-IoU-sensitive factor. Checkpoint dynamics show nonzero residual finals by
epoch 5 and settled late updates, so a residual branch that remained dead from
zero initialization is not a sufficient explanation. Representation capacity
versus optimization is not uniquely identified.

Under the preregistered kill rule, the current DCSR route and therefore the
current unique SparseHead route terminate at G1. G2--G4, five official seeds
and cost expansion are forbidden. This exact rejection is
`empirically_supported`; it is not a universal rejection of all sparse heads.
An official-quality dense proposal floor with selectively gated residual
compute is only `discussed` as a possible separately named successor. It
requires a new preregistration before any implementation or training.

G1 uses an internal 160/40 official-validation split and is not an official
THUMOS test row. Its absolute values cannot be compared with historical
`63.xx`, released `66.833392`, or the official S0 same-run dense `66.583013`.
The complete analysis and integrity audit are:

- `research-wiki/experiments/actionformer-sparsehead-dcsr-g1-negative-analysis-20260730.md`;
- `research-wiki/experiments/actionformer-sparsehead-dcsr-g1-integrity-audit-20260730.md`.

## 2026-07-31 ODF-CR successor preregistration

The user authorized direct execution of the recommended successor after the
completed G1 Pro analysis. ODF-CR is not a reopening of terminated DCSR G1. It
keeps an official-quality dense proposal floor and asks whether a separately
parameterized residual supplies useful incremental computation.

The frozen internal 2x2 is scaffold depth `1/3` × residual
`off/all_valid`. `d3_off` must be bitwise identical to official dense
ActionFormer; `d3_all` adds the independent residual. `d1_off/d1_all` diagnose
the shallow-floor failure seen in G1. K384 is a no-training replay after
all-query training, so support is not confounded with optimization.

The decision uses a new 160/40 validation-only holdout whose 40 decision videos
are selected only from the previous train-160 and are disjoint from the old
holdout. Three new seeds are paired training replicates on this fixed split.
This remains internal method selection and cannot be compared to `63.xx` or
official `66.xx`, placed in a paper table, or used for an efficiency claim.
The design commit is `77244d5`. The exact implemented/running candidate is
`01cdb78d2b7668098b6b13a1e49433d48fbc1a8d` with tree
`e70d2956a197b1204e721239178e76152efe282b`; Linux focused preflight is
`71 passed`. The immutable holdout-v2 SHA-256 is
`b8cac555f3d31e02468dbca3b3b0ada2d30b05bf046c10eb16304abb92499d1a`.

At formal launch, array `1209259_[0-2]` began the three frozen seeds. All three
real-CUDA G0 receipts pass official `d3_off` state/geometry/pre-decode/final
identity, `d3_all` floor/zero-residual and paired d1 initialization. The unique
G2 Job `1209267` was dependency-pending on the complete array. No arm metric or
route conclusion existed then; status was `experiment_running`.

## 2026-08-01 ODF-CR G2 negative and route narrowing

All three factorial tasks and dependent G2 Job `1209267` completed `0:0` with
clean frozen runtime identity and no hard-failure signature. Aggregate SHA-256
is `9172eddcbf5f9a4943b303e20b57f4492f0a44b18c39f892d5829b1f0a79ddec`.

The depth-three all-valid residual changes Avg by `-0.1806 pp`, with paired
seed deltas `-0.6970/+1.5070/-1.3520 pp`, only one positive seed, and
threshold deltas `+0.8466/+0.7666/-0.3656/-2.7468/+0.5960 pp`. G2 therefore
fails the frozen mean, sign-count and @0.6 checks. K384/G3 were not submitted.

The matrix nonetheless confirms the leading DCSR attribution internally:
`d3_off-d1_off` is `+7.5600 pp` Avg and `+11.4261/+14.6071 pp` at @0.6/@0.7.
Raw-prediction diagnostics show corresponding class-aware recall@200 gains of
`+5.87/+13.00 pp` at @0.6/@0.7 and smaller boundary errors. Conversely,
`d1_all-d1_off` is `+3.8689 pp` while `d3_all-d3_off` is null/negative, yielding
a `-4.0496 pp` depth-by-residual interaction. The all-valid residual helps a
weak floor but does not add reliable utility atop the official deep floor.

Training evidence further narrows the mechanism: `d3_all` reduces late
training loss below `d3_off` but does not improve holdout Avg. Class, duration,
video and score-bin effects are mixed. This is compatible with saturation,
overfit or ranking interference, not a dead residual or gross engineering
failure. Calibration/NMS and gradient-conflict causality remain unidentifiable
because only post-NMS predictions and ordinary loss scalars were recorded.

The ODF-CR all-valid residual route terminates at G2 with `empirically_supported`
negative evidence. The official-quality dense floor remains a supported design
constraint; conditional sparse execution itself was never run and is not
universally rejected. Any modified residual/gating study requires a new name,
fresh preregistration and fresh disjoint validation evidence. This internal
result remains ineligible for paper tables, official test, absolute comparison
to `63.xx/66.xx`, or efficiency claims.

The terminal ODF-CR heartbeat monitor was then retired. Its self-delete RPC
timed out, so the exact configuration was recoverably archived outside the
active automation directory; no cluster action accompanied this closure.

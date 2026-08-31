---
type: experiment
node_id: exp:phystime-frozen-decode-cross-replay
title: "PhysTime frozen dual-axis decode cross replay"
idea: idea:phystime-tad-2
status: tested
verdict: v16_suite_passed_decode_axis_effect_tested_route_claim_limited
confidence: v16_job_1203917_completed_0_0; explicit_suite_and_marker_validation_pass_true; four_formal_components_frozen_raw_native_direct_and_reviewed_p0_parity_passed; strong_single_seed_decode_axis_evidence_but_no_assignment_causality_multiseed_cost_or_paper_ready_claim
metrics: "v16 selected-online uniform/physical Avg-mAP=0.4125660433077075/0.5015355102106833 (+8.89694669029758 pp), completion/producer=6937fc6b7b050fd7009ee967ceef446aebaa8b3daa695c7959106ff87048c038/97410d9855a3f6db859e36213bf6b201e10c96941a164b5588af02cdfba4ee20. Selected-EMA uniform/physical=0.41283020792762315/0.5009785403306161 (+8.814833240299292 pp), completion/producer=4a1b405b7849f396e1b649da8895070e6176023c4a959c6d7fd9148f2bd8afe0/43c737fe3c5a9a534c565bf63e419fa152ee35b3be796ddf3f601c954fa52877. Physical-online uniform/physical=0.40107677185286417/0.5755558109390063 (+17.447903908614215 pp), completion/producer=fd18348e6ae6ecf4bdc4390ca4620a109616582f7f77138ed137085e0df6c260/d61d8fbf8b977b59b65eb87d55227904b2a5a2e6994e584226bda19a265b26eb. Physical-EMA uniform/physical=0.40296498031949024/0.5760868491267752 (+17.312186880728497 pp), completion/producer=cd6da2f827524e0b9eb2b46c6cbbcc5b6e89243aa9cd8d7e45efafcb4cb6b565/aa6356a509898b94a38f2b9e0548c5f647cc6498655697b37fd39ea8982fc733. Suite completion/validated SHA-256=ed2770c35cf9a3acd5fa80465eda1c34b3541ba3dea404c75388aaeffefbdc31/f2da143127b3a01aef7bda451e2351c494f72552f3810f604f895f4c0a7767d3."
provenance: "full60 0dc5851/bddc9b9; P0 c2cfcfa/0b78dd4; v10 tested components c878fbe3/8d3e73bb but suite engineering failure; v16 54e7f9abeaabf710a505f0a0f595a4eb3bb47f98/f8490f9c25c2e0e6958c406e19c83cc3d5a40535; preflight ccc7a83e27b8d18ad0892b644e7338667b72d8eba3e3feedbc387dc4ac1d9a0d; gate 0d2153effee84a0e1aa6410125bb291eb4ef4d41e4b40604f49d9e5868e0ada9; Slurm 1203917"
added: 2026-07-20T00:00:00+08:00
---

# PhysTime 冻结双轴解码交叉回放

## 为什么做

full60 的 physical-metric 比 selected-axis 高约 16.29 个 Avg-mAP 点，P0 又证明
这个差异不是旧的 NMS 提前舍入造成的。但现有两臂同时改变了训练几何和推理解码
几何，仍不能回答“只把同一组网络输出放回物理时间轴，性能是否会改变”。

本实验不训练新模型。它固定 full60 epoch-59 的 selected/physical checkpoint，
分别读取 online/EMA 权重；每个条件只跑一次真实 raw-video 网络推理，捕获同一组
分类 logits、分类分数、非负回归距离、候选 mask、基础 point grid 和两套时间轴，
随后从同一 artifact 分别执行：

- `uniform_rank_seconds` 解码；
- `physical_time_seconds` 解码。

## 四个冻结条件

| 条件 | 训练来源 | 权重 |
| --- | --- | --- |
| selected-online | selected-axis | online |
| selected-EMA | selected-axis | EMA |
| physical-online | physical-metric | online |
| physical-EMA | physical-metric | EMA |

每个条件内部的 `decode-P - decode-U` 是冻结 checkpoint 下的推理解码轴干预。
不同 checkpoint 之间的差值和差分之差只作描述，不能称为训练因果效应。

## 已实现合同

- 捕获功能默认关闭，不改变普通训练/测试路径。
- 同一条件只产生一份 pickle-free NPZ；数组逐项记录 dtype、shape 与 canonical
  SHA256。
- 原实现把原始 AMP 张量 dtype 单独登记、把存档统一上转 float32。真实门禁已经
  证明：在本次 `physical_online` 单窗口 native 轴上，point/proposal 重建经相同
  clamp 后没有差异；这不能外推为全数据集回归合同。分类分数上转会改变并列排序
  和 top-k 截断行为，因此统一上转策略不能继续作为已接受合同。
- 捕获内存设置 8 GiB 估计峰值硬上限；Slurm 作业申请 1 卡，由 N16R4
  按已验证的每卡默认 55 GiB 合同分配内存，不显式覆盖 `--mem`。
- native proposals 只作为重建误差审计参照，禁止覆盖重建 proposals。
- native 轴必须逐预测复现 direct pre-cross、最终结果和 mAP，否则 fail-closed。
- 当前 capture-enabled direct 还必须逐条件精确复现已审计 P0 direct 的
  pre-cross、最终结果和 mAP，避免“新 direct 与新 replay 一起漂移”仍自证通过。
- 除新增 capture 开关外，推理语义哈希必须与 P0 config 完全一致。
- selected/physical × online/EMA 四个真实窗口 gate 全部通过后，才允许四个正式
  replay 作业运行。
- 四条件 gate 在下游提交前比较共享视频/窗口/采样序列、U/P axis、base
  point/mask、native mask、class map 和各自 online/EMA state hash。
- 数据集全文件内容、VideoMAE、full60 source、P0 suite/gate、checkpoint、
  state_dict、config、sbatch、Slurm ID/依赖和日志均用哈希或身份链绑定。
- 首次 `sbatch` 前由纯 CPU preflight 重新计算全部不可变内容哈希；作业使用唯一
  DAG token/comment，网络重试先查询并接管已接受作业，避免重复提交。
- 本次失败实现的数值语义为：AMP 产生张量，存档统一上转 float32，CPU float32
  重算 decode。该语义并不完整复刻生产后处理，现已被真实门禁否决。
- suite 使用生产语义的重复后处理/evaluator 复算，不冒充外部独立 evaluator。
- 时长分层 recall 基于最终检测结果，明确不是 pre-NMS proposal recall。

## 审查中发现并修复

第一轮独立审查发现 native 解码重建后又被捕获的 `native_proposals` 覆盖，会让
原生等价验证形成自证。该旁路已经删除并加入反向测试。审查还要求补齐数据和权重
内容哈希、四条件 gate、state_dict 哈希、内存预算、Slurm DAG 身份、日志扫描及
非因果指标命名，均已进入当前实现。

第二轮审查发现提交前内容未重算、`sbatch` 重试不幂等、source report 漏
`return`、当前 direct 未锚定 P0、配置语义与共享观测契约 fail-late 等问题。
当前实现已补齐 preflight manifest、唯一 DAG token、真实 scontrol/sacct
快照、P0 direct/semantic 双重锚定、四条件 state/observation 早期门禁、原子
状态文件和致命日志扩展；这些修复尚待远端 fixed-conda gate 实证。

第三轮独立复审发现一个会让真实 gate 必然误失败的契约混淆：
`window_sequence_sha256` 包含 `native_coordinate_mode`，因此
selected-axis 与 physical-metric 按设计不同，不能作为四条件共享观测哈希。
当前修复把共享稀疏观测、两套时间轴、mask、base points 和计数保留在
`observation_contract`；把原生轴与窗口哈希放入 `axis_window_contract`，
只要求同一 arm 的 online/EMA 完全一致，并要求 selected/physical 的轴特有
窗口哈希不同。行为测试覆盖了跨 arm 合法差异和同 arm 非法漂移。

第四轮部署复审发现两个会污染 Slurm 证据链的缺口：临时文件锁在提交器退出后
不能阻止同一 DAG token 被另一个运行目录接管；`sbatch` 响应丢失时，作业若
延迟出现在 `squeue/sacct`，单次查询后重试可能重复提交。当前实现新增全局
owner manifest，永久绑定 `token/run_root/commit/tree`，并与运行目录内 owner
双向校验；同 token 换目录会在任何队列查询或取消前失败。失败或非数字
`sbatch` 响应后先做有界可见性轮询。进一步复审确认“轮询预算耗尽后重投”
仍不具备服务端幂等，因此最终实现会在 `sbatch` 前持久化提交意图；未找到
唯一作业时保留 ambiguous 状态并退出，后续恢复只查询、绝不自动重投。测试
覆盖永久 owner 冲突、立即接管、预算内延迟可见接管、超预算后持续阻断、
记账可见后的恢复接管和重复 comment 拒绝。
`resolved` 与 `fatal` 也是永久状态：resolved 作业暂时不可见时只允许等待
记录的 Job ID，fatal token 永久拒绝复用。suite 会核对 gate、四个 replay
和自身共六个 resolved marker 与 `jobs.tsv` 的 token/comment/Job ID，并拒绝
遗留 ambiguous/fatal 文件。

## 2026-07-23 Pro verdict intake

External Pro review attachment SHA256
`28C6D00404B7530D5A85E27538FA3EFBE07021ACDC54AEA713E0E4222EA79CC1`
was recorded in `docs/methods/reviews/2026-07-23-pro-decode-replay-verdict-intake.md`.
The accepted verdict is `REVISE_BEFORE_REQUEUE / HOLD`, not abandon. Local
source review confirms that the capture layer rewrites source `cls_scores` to
float32 and both replay validators enforce that rewrite, whereas direct
post-processing sorts the source score after CPU transfer.

The next implementation is A-STRICT-SOURCE-DTYPE: preserve ordering-sensitive
scores at source dtype and use that dtype for replay CPU pre-NMS operations.
It must not change production post-processing, NMS, evaluator, checkpoint, or
training. Stable total ordering is explicitly deferred as a new inference
semantic requiring a separate re-anchor. Existing remote hashes/counts remain
forensic records pending independent raw-artifact recomputation.

Required before requeue: schema-v2 dtype provenance, v1 source-fp16/stored-fp32
rejection, source-score roundtrip/tie-boundary tests, ordered/top-k candidate
diagnostics, failure artifacts, runtime fingerprint, P0 direct anchoring, then
the four-condition CUDA gate. Training remains frozen.

## 当前状态与停止条件

当前状态是 `tested`，裁决为真实门禁失败，正式 mAP 为 `NA`。本地部署测试、远端
Linux focused suite 和 Slurm 身份链均通过，但真实 CUDA 门禁在
`physical_online` 的 native direct/replay 精确等价检查处失败。此前独立部署复审
给出的 `DEPLOY` 只授权执行真实 gate，不等于实现语义已正确。

若四条件真实 gate 中任一 native replay 不能精确复现 direct 结果，立即停止正式
DAG并发起 Pro 讨论；不得用捕获 proposals 替代重建来“修复”等价。只有 gate 和四
份 completion、suite 全部通过，才允许裁决该回放是否获得实验支持。失败门禁本身
可记录为 `tested / real_gate_failed`，但不得写成 replay 已完成。通过后也只决定
是否进入 Q192 UU/UP/PU/PP 训练设计，不产生 `paper_ready` 结论。

## 2026-07-20 `9bbc6ea` 无效部署审计

运行时 commit `9bbc6eadf85dd65364223da719d13dd5b3789dda`、tree
`68b5cc3f68ec1dfedbba82ac1421bf89d88b88d8` 的 CPU 全内容 preflight
通过，且无显式 `--mem` 的 N16R4 一卡资源合同通过 `sbatch --test-only`。
随后用全新 DAG token
`ptdc_9bbc6ea_6f75b261e21d4626a7399a248afd6aee` 创建了六个作业：

| 角色 | Job ID | 最终状态 |
| --- | ---: | --- |
| gate | 1175739 | `FAILED 1:0` |
| selected-online | 1175740 | `CANCELLED`，未启动 |
| selected-EMA | 1175741 | `CANCELLED`，未启动 |
| physical-online | 1175742 | `CANCELLED`，未启动 |
| physical-EMA | 1175743 | `CANCELLED`，未启动 |
| suite | 1175744 | `CANCELLED`，未启动 |

精确 run root 为
`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_9bbc6ea_20260720_153600_0800_6f75b261e21d4626a7399a248afd6aee`。
gate 只运行到 focused tests，结果为 `61 passed / 1 failed`；失败发生在
`DecodeReplayCollector.finalize()` 读取最小测试配置中不存在的
`cfg.solver`，真实四条件 CUDA 推理、native direct 等价和 P0 direct 锚定均
未执行，gate JSON 也未产生。因此这不是科学门禁失败，不触发方法否定或 mAP
讨论。

同一提交的 submission scheduler capture 还发现 Slurm 会把
`afterok:a:b:c:d` 展开为
`afterok:a(unfulfilled),afterok:b(unfulfilled),...`。旧比较仅删除括号，
没有把两种表示解析成同一依赖集合，因而在六个作业身份均正确时误报 suite
dependency mismatch。提交失败路径和人工复核只定向清理了该 token 的六个
作业；旧 token、run root 与 resolved markers 永不复用。

修复边界严格限定为：

- 可选 `solver` 段缺失时只把清单中的 `source_amp_enabled` 记为 `false`；
  生产完整配置的 AMP 值仍按原配置读取。
- 调度依赖按“依赖类型 + 数字 Job ID”结构解析、去状态注释、顺序无关比较；
  缺失、额外、错误类型、错误 Job ID 和重复依赖继续 fail-closed。
- scheduler snapshot 升级为 v2，同时保存原始依赖、预期依赖和两者的规范化
  记录；最终 suite 必须再次核对六个作业身份和规范化依赖。

修复不改变模型、输入张量、checkpoint、decode、NMS、evaluator 或任何指标。
新部署必须使用新 commit/tree、clean snapshot、run root 和 DAG token。

修复代码已冻结为 commit
`06a6734449024875031cc3d1e0d08520824d2e67`、tree
`c11dc39670254c90ad21f3e26581e4f654f25c59`。本地完整 focused suite 为
`46 passed / 6 skipped`，Python 编译、Bash 语法和 `git diff --check`
通过。同一独立部署审查代理在 P2 测试缺口补齐后复核，最终裁决为
`DEPLOY`，P0/P1 为 0。该裁决只授权新 clean snapshot 和真实 gate；
experiment 状态仍为 `implemented`，mAP 仍为 `NA`。

## 2026-07-20 `06a6734` 正式 DAG

clean snapshot：
`/data/run01/sczc063/yuzibo/projects/opentad_phystime_decode_cross_06a6734_20260720`；
正式 run root：
`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_06a6734_20260720_161200_0800_9c608d9ee647451a91ec438c93ecc2f1`；
DAG token：
`ptdc_06a6734_9c608d9ee647451a91ec438c93ecc2f1`。

| 角色 | Job ID | 最终状态 |
| --- | ---: | --- |
| gate | 1175820 | `FAILED 1:0` |
| selected-online | 1175821 | `CANCELLED`，未启动 |
| selected-EMA | 1175822 | `CANCELLED`，未启动 |
| physical-online | 1175823 | `CANCELLED`，未启动 |
| physical-EMA | 1175824 | `CANCELLED`，未启动 |
| suite | 1175825 | `CANCELLED`，未启动 |

提交前全内容 preflight 为 `validation_pass=true`，重算并复现：

- dataset manifest：
  `1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2`；
- VideoMAE：
  `4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251`；
- selected checkpoint：
  `6fd0781b53e094bb30f0664e006a657fa7c7ef5b3be2de558856c8d23b6bb417`；
- physical checkpoint：
  `c83a3463155c0a9926a4fc8d62f4d0ee7540c1a58293fb4c3cc9bad8ce9237ed`。

真实 scheduler snapshot v2 已核对六个 Job ID、name、comment、stdout/stderr
与依赖。suite 的 Slurm 原始依赖为逗号分隔并带 `(unfulfilled)` 的四个
`afterok` 子句，规范化后与预期 `afterok:1175821:1175822:1175823:1175824`
完全相同，证明本轮依赖修复在真实调度器上生效。

gate 内与生产一致的 Linux focused suite 为 `73 passed`，随后进入真实 CUDA
门禁。`selected_online` 与 `selected_ema` 的单窗口 native direct/replay 精确
等价通过；`physical_online` 在同一检查失败，`physical_ema` 未执行。由于失败
发生在最终 gate JSON 写出前，没有 `decode_cross_gate.json`。五个下游作业均按
同一 DAG token 定向取消且从未执行，正式 mAP 为 `NA`。

## 真实门禁失败取证

失败窗口为 `video_test_0000004`，物理时间域 `[0.0, 33.826]` 秒；原始观测
`K=384`、选中原始帧 `253`、native token `J=192`、有效 token `127`、候选
`Q=378`。三份已生成 artifact 的共享观测序列哈希均为
`502cfeb5e2bf5eb0e0cc0f40fe43c5b22682ece3e711b4fe74d1b6ae158dc1b6`。

对 `physical_online` artifact 的只读取证得到：

- native point 重建最大绝对误差为 `0.0`；
- native proposal 重建最大绝对误差为 `0.0`；
- 裁剪前有 7 个坐标值不同，最大差 `5.807575225830078`，均来自生产路径尚未
  执行的边界裁剪；执行同一 `[0, 33.826]` 裁剪后，251 行 proposal 全部逐元素
  相同；
- 用捕获的 native proposal 作审计参照与从原始回归张量重建的结果，后处理后均
  为 2000 条，逐行完全相同，哈希均为
  `08022c244c88a5446e1a3f66dc569c29d055ee66fb13d8079a6bc3ba97715f51`；
- 因此失败不是 physical point/proposal 重建误差。

进一步只读模拟复现了直接路径与存档路径的真正差异。生产 AMP 的
`cls_scores` 是 `float16`，捕获器在
`AnchorFreeHead._capture_decode_replay_state()` 中把它统一转成 `float32`；
数值虽然无损扩展，但 `SingleStageDetector.post_processing()` 的 CPU
`sort(descending=True)` 在大量并列分数上产生不同顺序，并在
`pre_nms_topk=2000` 截断边界改变候选集合。

| 条件 | 分数元素 | 同值冗余计数 `sum(n-1)` | float16 / float32 结果 |
| --- | ---: | ---: | --- |
| selected-online | 5020 | 3812 | 逐行、哈希完全相同 |
| selected-EMA | 5020 | 3810 | 逐行、哈希完全相同 |
| physical-online | 5020 | 3830 | 零基索引 147（第 148 条）分叉；top-k 集合各有 2 条独有结果 |

`physical_online` 的 float16 审计结果哈希为
`fd07d8b7c6f366e0996cd97c4ee04d7d51a02e03213a508abcd624f9d3b5ceb3`，
float32 重放哈希为
`08022c244c88a5446e1a3f66dc569c29d055ee66fb13d8079a6bc3ba97715f51`。
其中 4613 个分数元素属于至少含两个元素的同值组，3830 是每个同值组扣除首项后
的冗余计数，二者不得混写。
这说明“统一上转 float32 不改变推理语义”是错误合同。不能降低精确等价门禁、
不能对结果事后舍入、不能用捕获 proposal 覆盖重建 proposal，也不能直接重排
原 DAG。下一步必须先由 Pro 严审决定：保留源分数 dtype，还是为生产与重放共同
定义可审计的稳定并列排序；通过审查与新回归后，才允许新 commit/tree/token
重新走单窗口四条件 gate。

独立终态复审给出 `REVIEW_VERDICT=REVISE_BEFORE_REQUEUE`、
`EXECUTION_VERDICT=HOLD`，确认当前根因链足以解释 exact gate 失败，但不等于
PhysTime 性能下降根因，也不使既有 full60 数值失效。复审同时指出：正式 gate
没有记录 CPU 型号、实际 `torch.get_num_threads()` 或线程环境，现有证据只能绑定
当前固定 PyTorch/NumPy/节点架构，尚未证明跨 PyTorch/CPU 的并列排序可移植性。

## 2026-07-28 Approach A 实现与远端 CPU 验证

按已批准的 evidence-first 方案恢复并整理了严格 source-dtype 回放链。分类排序分数
不再在捕获时统一扩展为 float32；schema-v2 同时记录源 Torch dtype 与存储 NumPy
dtype，并拒绝 ordering-sensitive widening。捕获仍默认关闭，未消费上一批状态时
fail closed。suite 改为只接受显式 preflight、gate、P0、四个 completion 和日志
路径，不再恢复 owner、`jobs.tsv`、scheduler snapshot 或 submission-state 框架。

远端收集同时暴露了一个真实生产契约缺口：PhysTime physical config 显式声明
`positions_key=phystime_g1a_axis_positions_sec`、native count 和秒域起止字段，但当前
`AnchorFreeHead` 没有消费这些配置，会回退到 dense index。该缺口不是测试夹具偶然
失败；它会破坏“预测已经位于 seconds 轴”的合同以及 native direct/replay parity。
当前实现只在显式配置这些字段时恢复历史已验证的 rank-to-seconds 插值、局部物理
stride、严格递增/域检查与最终 domain clamp；未配置的现有 C3 irregular grid 路径
保持原行为。P0 所需的 full-precision cross-window NMS helper、非法 proposal 审计和
class-index long dtype 也按精确历史来源恢复；普通配置仍默认先舍入，只有 P0 配置关闭。

隔离验证过程保留了负证据：

- `_05`：静态关卡通过，`44 passed / 5 failed`；五项同根失败定位到上述未消费的
  explicit physical-axis config；
- `_06`：运行语义通过到 `58 passed / 1 failed`；唯一失败是既有静态测试要求调用
  保持单行源码字符串，属于等价格式差异；
- `_07`：package SHA-256
  `e4814e3544784b3608c007a11946464b4f597e0fbf9a23a5910e3b0171bef388`，
  `CONFIG_RESOLVE_OK 3`、`IMPORT_CLOSURE_OK`、`PY_COMPILE_OK`、
  `BASH_SYNTAX_OK`，focused tests `59 passed in 64.69s`。

最终目录是
`/data/run01/sczc063/yuzibo/sparsehead_remote_cpu_a6bdc084_20260728_07`。
截至该 CPU 验证节点，没有申请 GPU、没有提交 Slurm 任务、没有运行四条件 CUDA
gate，也没有产生新 mAP；当时状态只能升级为 `tested`。

## 2026-07-28 v4 完整实验部署

Approach A 的完整证据链已冻结并提交到 N16R4。独立、干净的远端 Git 快照位于
`/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260728_v4`，
分支为 `codex/sparsehead-evidence-deploy-20260728`，精确 commit/tree 为
`8e31b9e3c08b0a8d320e031b04dfd63e19eb08df` /
`aae5503424aa3925ef99bba851d600a03e3c3377`。该快照建立在当前路线 base
`a6bdc084cc145c80b6b2c68d0a38f0deea3e8518` 上，并只冻结本次证据链所需内容；
没有把本地其他脏工作区状态当成运行身份。

运行根目录是
`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260728_v4`。
CPU full-content preflight 的 manifest 为 `preflight_manifest.json`，SHA-256
`3551816b8e056b9afea4fc9ee8575f525e78ffba64ff087915130b2e10e54712`，
`validation_pass=true`。它复核了冻结 THUMOS train/test 内容、VideoMAE 权重、
selected epoch-59 checkpoint 与 physical epoch-59 checkpoint；数据 manifest
复现为 `1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2`。

由于账号当前存在 GPU 组配额等待，部署使用单个 Slurm 分配串行执行严格
fail-closed 链：

1. selected/physical × online/EMA 四条件、单窗口 real-CUDA gate；
2. gate 全部通过后才顺序运行四个完整 direct-inference + 双 decode-axis replay；
3. 四个 completion 全部存在后才运行显式 evidence suite。

这只压缩调度，不改变模型、配置、checkpoint、seed、evaluator 或判定门槛。正式
作业是 `1201048`（`ptdc-a1-full`，`gpu` partition，1 GPU，6 CPU，6 小时）；
`sbatch --test-only` 打印的 `1201047` 只是测试编号，不是作业。

前三次部署尝试均为零作业诊断：v1 被 Windows PowerShell 对远端正常 stderr 的
处理截断；v2 未设置仓库根 `PYTHONPATH`；v3 使用了通用 raw 路径而非 P0 冻结的
`/thumos14/train` 与 `/thumos14/test`，并被 preflight 正确拒绝。三者都没有提交
Slurm、没有模型 forward、没有实验结果；不得与 v4 或科学证据合并。

Job `1201048` 后续在 `g0043` 实际运行 `1m54s` 并以 `FAILED 1:0` 终止。gate
前置 `39 passed`，随后第一次真实模型构造在
`build_detector(cfg.model)` 失败：
`ActionFormer.__init__() got an unexpected keyword argument
'native_temporal_geometry'`。没有生成 gate JSON，四个 replay 和 suite 均未启动。

## 2026-07-29 v5 工程恢复

失败签名登记为
`actionformer_native_temporal_geometry_constructor_contract_v1`。根因不是模型、
checkpoint 或数据，而是路线合并时漏恢复 commit `8e2b8322` 中 native-J192
ActionFormer 的 K384→J192 对齐合同。G1a full60/P0 权重仍对应
`ActionFormer + Conv1DTransformerProj + FPNIdentity + ActionFormerHead`；把配置
改成 `PhysTimeTAD` 会同时破坏 constructor、projection/head API 与 checkpoint
命名空间，因此被拒绝。

协议不变修复恢复了 `ActionFormer` 对
`native_temporal_geometry` 的显式归一化、strict-padding-aware backbone 调用、
native mask/meta 对齐和 query audit；未配置该字段的其他 ActionFormer 路径继续
使用原调用。新增两个回归关卡分别绑定 resolved G1a config 与 constructor signature，
并验证 ActionFormer 在不更换 detector family、不插值的情况下消费 native geometry。

新的独立干净 runtime 位于
`/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260728_v5`，
branch `codex/sparsehead-evidence-recovery-20260728-v5`，commit/tree 为
`0338f4777bd02fb327573ef716f54fec76d4af0e` /
`cb98c64c17d2983c22181d4908c4f31024a82a2f`。Linux recovery focused suite
`74 passed in 79.28s`；full-content preflight 继续复现同一 dataset/checkpoint/
VideoMAE 哈希并通过，manifest SHA-256 为
`77b9918aa3173b73fc71d821defa8c14b3165de1b35f0ae4c0382eeb5d21b43d`。

唯一替代作业是 `1201317`（`ptdc-a1-r1`）；`1201316` 仅为 test-only 编号。
它在 `g0053`/RTX 4090 上通过 gate 的 `41` 项前置测试和四条件真实 CUDA gate：
`gate_pass=true`、`all_native_direct_exact_equivalence=true`，且
selected/physical × online/EMA 四项均 `raw_tensors_immutable=true`。

## 2026-07-29 v5 Job1201317 工程失败

`selected_online` direct inference 随后完整跑完，并在 evaluator stdout 中显示
Avg-mAP `41.26`，mAP@0.3--0.7 依次为 `64.50 / 56.39 / 42.66 / 27.82 /
14.90`，输入为 `3325` 个 GT 和 `422000` 条 prediction。紧接着 launcher
找不到已声明的
`direct_work/gpu1_id0/pre_cross_window_detections.json.gz`，Job `1201317`
以 `FAILED 1:0` 终止；其余 `selected_ema / physical_online /
physical_ema` replay 与 suite 均未开始。

失败签名登记为
`direct_postprocessing_artifact_producer_contract_missing_v1`。根因是路线整理时
consumer/validator/launcher 的 direct-artifact 合同被保留，但
`eval_one_epoch()` 中对应 producer 被遗漏；这不是路径拼写、模型性能或评价器
失败。上面的 `41.26` 仅是失败根中已完成 direct inference 的诊断性输出，
`primary_result_allowed=False`、`metric_claim_allowed=False`，不得冒充正式
decode-cross completion、suite verdict 或新方法结果。

## 2026-07-29 v6 producer-contract 修复与重部署

协议不变修复恢复
`opentad_pre_cross_window_detections_v1`、`opentad_post_processing_audit_v1`
及 `evaluation_metrics.json` 的原子生产，并新增从一轮 direct inference 到四个
声明产物的端到端回归。新 clean runtime 位于
`/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260729_v6`，
branch `codex/sparsehead-evidence-recovery-20260729-v6`，commit/tree 为
`ac326ffdc97652433b55ccc596e734b112f51806` /
`0c58027756997995bda0de6fdd8ec0deb49966d3`。Linux focused suite
`75 passed in 76.43s`；full-content preflight `validation_pass=true`，manifest
SHA-256 为
`97fe5af28b2647396c052c9bdf956997d98e264af74432b57e0fc983b071fb91`。

唯一后继作业是 `1201469`（`ptdc-a1-r2`）；`1201468` 仅为 test-only
编号。它在 `g0030`/RTX 4090 上通过 gate focused tests `42 passed`，并已产生
SHA-256
`775e1f2dae70b7863324fd9d235712195dca4d0846968b3bd5e55b754e7b3ea4`
的 v6 real-CUDA gate artifact：`gate_pass=true`、
`all_native_direct_exact_equivalence=true`，selected/physical × online/EMA
四项均 `raw_tensors_immutable=true`。`selected_online` 全量 direct inference
随后完成，精确 metrics JSON 为 Avg-mAP `0.4125660433077075`，mAP@0.3--0.7
依次为 `0.6450446628552113 / 0.5638932489689005 /
0.4266348135535575 / 0.27820781407261164 / 0.14904967708825695`。

本轮已越过 v5 的精确失败点：direct workdir 中存在 schema
`opentad_pre_cross_window_detections_v1` 的 `211`-video gzip artifact、schema
`opentad_post_processing_audit_v1` 的 audit、`evaluation_metrics.json` 和
epoch-59 `result_detection.json`，均绑定 v6 commit/tree；audit 登记的 pre-cross
SHA-256 为
`31e70dc728aff9061f2c56266e3e6d32ef892b227a5c16b15da85e81f731b50e`。
Job 仍为 `RUNNING`，已进入 `selected_online` 双轴 replay，并写出
uniform-rank decoded candidates/pre-cross 中间产物；尚无
`DECODE_CROSS_COMPLETE.json`、suite verdict 或最终 cross-decode 结论。
状态仍为 `experiment_running`，不能升级为
`empirically_supported`，也不能声称 cross-decoder confound 已关闭。

## 2026-07-29 监控与负结果处理策略修正

用户明确取消“每个工程失败签名最多自动修复一次”的限制。对已确认的非模型工程
故障，后续监控必须持续执行“保全失败根 → 复现与根因定位 → focused regression →
静态/CPU/full-content preflight → 新 clean commit 与不可覆盖 run root → Slurm”
闭环，直到获得完整最终性能。相同签名重复时必须深化根因和修复，不允许无变化盲重提。

若最终模型结果合法但为负，则不自动把它解释成部署失败，也不静默调参使结果转正。
必须先开展深入技术讨论：列全原始结果并相对 matched controls、full60、P0 与四条件
交叉回放比较，分解高 tIoU、类别/时长/边界、proposal recall、校准/NMS、assignment、
support observability、native geometry、decoder-axis regret、online/EMA 与成本；
检查内部矛盾和与历史证据的冲突；至少提出两种竞争解释及其反证、可证伪预测和最小
决定性实验。核心算法修改与重训须建立在这份分析之后。

## 2026-07-29 v6 validator-scope 失败

Job `1201469` 最终在 `g0030` 运行 `32m32s` 后以 `FAILED 1:0` 终止。
失败前，`selected_online` direct inference 与双轴 replay producer 均完整结束。
producer completion（SHA-256
`0283620a7c5308275c45d03ab1cf639cb8b889d385122d9907fa3e373ef74062`）
报告 `validation_pass=true`、相同冻结 raw tensors、native/direct exact
equivalence。uniform-rank/native direct 的 Avg-mAP 为
`0.4125660433077075`，mAP@0.3--0.7 为
`0.6450446628552113 / 0.5638932489689005 / 0.4266348135535575 /
0.27820781407261164 / 0.14904967708825695`；physical-time cross-decode
的 Avg-mAP 为 `0.5015355102106833`，mAP@0.3--0.7 为
`0.7227308252635063 / 0.6467256016774477 / 0.5253267642276429 /
0.39032810111601973 / 0.22256625876880035`，同一 frozen
`selected_online` 输出上的 decoder-axis 差为 `+8.89694669029758 pp`。

随后 post-replay validator 在组装 completion 时执行未绑定的局部变量
`numeric_precision`，触发 `NameError`。失败签名登记为
`decode_cross_validator_numeric_precision_scope_v1`：helper 曾校验临时
`producer.get("numeric_precision", {})`，但 `validate_run()` 没有把它绑定后传入
completion。这是确定性的 validator 实现故障，不是模型负结果。正式
`DECODE_CROSS_COMPLETE.json` 未生成，另三条件和 suite 未开始；上述双轴数值只能
标记 `diagnostic_only`、`primary_result_allowed=False`、
`metric_claim_allowed=False`。

## 2026-07-29 v7/v8 validator 修复与唯一后继

修复把 producer numeric-precision 合同复制到局部变量、先校验再写入 completion，
并新增“不修改 producer 且 completion 保留已校验 precision”的回归。v7 clean
runtime commit/tree 为 `1631d0b60f6552a6f5eb0378d74e766850f34ffd` /
`f485c8708e22bbbf9a73063d5293a20bc4aa658f`；与 v6 精确测试面加新回归为
`76 passed`，full-content preflight 也通过。

v7 在 `sbatch --test-only` 之前被部署哈希门禁拒绝：预期 focused-log SHA-256
漏写末位 `6`。失败签名为 `deployment_expected_sha256_truncation_v1`，没有创建
Slurm 作业、没有 CUDA/model forward；v7 root 只作为不可覆盖的部署诊断根保留。

唯一正式后继迁移到新的 v8 clean runtime
`/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260729_v8`，
branch `codex/sparsehead-evidence-recovery-20260729-v8`，沿用同一修复
commit/tree。新 run root 为
`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v8`；
精确 Linux recovery suite `76 passed`，日志 SHA-256
`16f53fc3cf8a9c5010bce3fd1ed98c4e347add284ba4b2443c00b49b5e107390`；
preflight `validation_pass=true`，SHA-256
`e9f36c221156e5411dad5e3bfe43508b4aa59310539fdbe24da985fc99a27d53`。
`1201494` 只是 test-only 编号；唯一正式 Job `1201495`
（`ptdc-a1-r4`）已在 `g0024`/RTX 4090 上 `RUNNING`。它仍按一个 allocation
内 gate → 四 replay → explicit suite 串行 fail-closed；模型、配置、epoch-59
checkpoint、seed42、数据 manifest、evaluator、门槛和证据边界均未改变。状态保持
`experiment_running`。

v8 随后通过 gate focused suite（`43 passed in 29.08s`）和四条件 real-CUDA
gate。gate artifact SHA-256 为
`5e323e5ccdedd7dd39d70148aed7108beca94bb5952125a124ad20accfd634f6`；
`gate_pass=true`、`all_native_direct_exact_equivalence=true`，且
selected/physical × online/EMA 四项均 `raw_tensors_immutable=true`。Job
已进入 `selected_online` 全量 direct inference，尚无 v8 completion 或 suite。

v8 `selected_online` direct inference 随后完整结束。精确 Avg-mAP 为
`0.4125660433077075`，mAP@0.3--0.7 为
`0.6450446628552113 / 0.5638932489689005 / 0.4266348135535575 /
0.27820781407261164 / 0.14904967708825695`。schema
`opentad_pre_cross_window_detections_v1` 覆盖 `211` videos 并绑定 v8
commit/tree；pre-cross SHA-256 为
`b4adcf545655424d2b2dfdfce0d107109c5010850143fadf925706fb3de60322`，
audit 中的登记值一致。当前双轴 replay 已完成 uniform-rank mode 产物并在生成
physical-time mode；尚无 formal completion 或 suite，未发现 hard-failure
signature。

## 2026-07-29 v8 四回放完成后的 suite 类型合同失败

Job `1201495` 最终在 `g0024` 运行 `02:00:24` 后以 `FAILED 1:0`
终止。四个 `DECODE_CROSS_COMPLETE.json` 和四个 producer completion 均已生成；
每个 replay 都报告 `validation_pass=true`、`status=tested`、
`same_frozen_raw_tensors_for_both_axes=true`、
`native_direct_exact_equivalence=true` 和
`reviewed_p0_direct_exact_equivalence=true`。四条件 gate 仍为通过，模型与原始
tensor 没有发生变化。

v8 的全部双轴原始指标如下。每行顺序均为
`Avg-mAP / mAP@0.3 / @0.4 / @0.5 / @0.6 / @0.7`：

- selected-online，uniform-rank：
  `0.4125660433077075 / 0.6450446628552113 / 0.5638932489689005 /
  0.4266348135535575 / 0.27820781407261164 / 0.14904967708825695`；
  physical-time：
  `0.5015355102106833 / 0.7227308252635063 / 0.6467256016774477 /
  0.5253267642276429 / 0.39032810111601973 / 0.22256625876880035`。
- selected-EMA，uniform-rank：
  `0.41283020792762315 / 0.648555172457041 / 0.5634561176089795 /
  0.42615161707187166 / 0.2774504499366236 / 0.14853768256360006`；
  physical-time：
  `0.5009785403306161 / 0.7226575722209517 / 0.6462475077859726 /
  0.5282674811180088 / 0.3876402290602233 / 0.22007991146792394`。
- physical-online，uniform-rank：
  `0.40107677185286417 / 0.6148821307005312 / 0.5285472772022218 /
  0.41348264750812447 / 0.28615371812507695 / 0.16231808572836598`；
  physical-time：
  `0.5755558109390063 / 0.7704022473065874 / 0.7055742485050899 /
  0.6207490393477052 / 0.48593657950784275 / 0.29511694002780653`。
- physical-EMA，uniform-rank：
  `0.40296498031949024 / 0.622154649489393 / 0.5316588686305871 /
  0.4113769771975965 / 0.2880843206041682 / 0.16155008567570622`；
  physical-time：
  `0.5760868491267752 / 0.7721224901972557 / 0.7045574192938243 /
  0.6257613932435541 / 0.4900660583199814 / 0.28792688457926047`。

对应 completion / producer SHA-256 分别为：
selected-online `e5a2c64c...b62f3` / `b75620ef...9161`，
selected-EMA `5a76e12e...273d` / `da6165e9...038a`，
physical-online `053cdbc2...dd97` / `55938bee...8ff`，
physical-EMA `81da7980...0f0c` / `877ab9b1...738b`。

最终 explicit suite 在读取第一个 completion 时 fail closed。四个 completion
都把 `fatal_log_findings` 写成空 JSON object `{}`，而 suite consumer 明确要求
空 array `[]`。因此失败签名登记为
`decode_cross_completion_fatal_log_findings_container_type_v1`。suite validator
log SHA-256 为
`558c78694ae18b9827e4b3cc27f731f3e684faa7eb9a08a1670584c154102919`，
v8 failure receipt SHA-256 为
`22739defebe8261f61e1fff9910d6d74592d6de4621f7147b07138154ae94d13`。
这是一项 producer/consumer 序列化类型合同错误，不是合法的模型负结果；由于
suite completion 缺失，上述八行仍是 `diagnostic_only`，
`primary_result_allowed=False`、`metric_claim_allowed=False`。

## 2026-07-29 v9 预提交失败与 v10 唯一后继

第一次新根 v9 在形成修复 commit 前被 Git author 门禁拒绝：远端克隆没有
repository-local `user.name/user.email`。失败签名为
`runtime_git_author_identity_missing_v1`；没有 commit、preflight、CUDA forward
或 Slurm 作业。v9 pre-submission failure receipt SHA-256 为
`ca7f75bc72e85fd466331012775cff72ca14fd685b1db4cc52c8212450c994d2`，
该半初始化根永久保留且不复用。

唯一正式后继是 v10：

- runtime：
  `/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_approach_a_20260729_v10`；
  branch `codex/sparsehead-evidence-recovery-20260729-v10`；
  commit/tree
  `c878fbe3a5e960671f03d93fff8367ed3414f5c5` /
  `8d3e73bb26544d1bcf7bfb61154d0b003f2658e0`。
- run root：
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v10`。
- Linux exact recovery suite 为 `77 passed`，日志 SHA-256
  `7f1787308250a6c9bd62e452f6e16357f5d6bf44cdbcfc6fedd61b7cc63c6936`；
  full-content preflight SHA-256
  `f46f6299f7fccc899140ad8fdf001052772ef550dd34cdb68c17d5ba5fc59a8f`。
- 修复只把零 fatal finding 统一序列化为 array，并新增显式
  `[PhysTime ... ERROR:]` 扫描和 producer/consumer 类型回归；不改变模型、配置、
  checkpoint、seed42、数据、评价器或门槛。
- `1203046` 只是 `sbatch --test-only` 编号。唯一正式 Job `1203047`
  （`ptdc-a1-r5`）已在 `g0050`/RTX4090 `RUNNING`；deployment identity /
  submission receipt SHA-256 分别为
  `1ece7c71b3fc9c396f49401460e5474e3dcaa7ba7f6cf009b987c2b3909a2246` /
  `9ec33e550d72f69847bcb2a5b2457fad03aa15df54d843e233b2020b5ef5724f`。
  gate focused tests 已通过 `44 passed in 29.68s`；四条件 real-CUDA gate
  也已通过，artifact SHA-256
  `e5516af02289d15dd1465f5387471bb1a3c357873980d22645c08acbf6aa141c`，
  `gate_pass=true`、全部 native/direct exact equivalence 且四组 raw tensor
  immutable。

`selected_online` 已完成新鲜 direct inference、双轴 replay 和 validator，并写出
正式 component completion：

- `DECODE_CROSS_COMPLETE.json` SHA-256
  `a4e727cf094127be7b91a4a13b140463ad9dc3e0c8c1bcfa3acb9887b5ff6dda`；
  producer completion SHA-256
  `8a2d38db8a2130a8b617940361a8637dfdc0bff3b6947b0f35d75167a809bfa6`。
- `validation_pass=true`、`fatal_log_findings=[]`、same frozen raw tensors
  为真、native/direct exact equivalence 为真、reviewed-P0 direct exact
  equivalence 为真；runtime commit/tree 与 v10 冻结身份完全一致，且
  `new_training=false`。
- uniform-rank decode：
  Avg-mAP `0.4125660433077075`；mAP@0.3–0.7 为
  `0.6450446628552113 / 0.5638932489689005 / 0.4266348135535575 /
  0.27820781407261164 / 0.14904967708825695`。
- physical-time decode：
  Avg-mAP `0.5015355102106833`；mAP@0.3–0.7 为
  `0.7227308252635063 / 0.6467256016774477 / 0.5253267642276429 /
  0.39032810111601973 / 0.22256625876880035`。
- physical-minus-uniform 为 Avg `+8.89694669029758` pp，按阈值分别
  `+7.768616240829496 / +8.283235270854728 / +9.869195067408542 /
  +11.21202870434081 / +7.35165816805434` pp；两种 decode 均覆盖
  211 videos、1,584,000 个 NMS 输入和 422,000 个输出 prediction，invalid /
  filtered / rounding-induced-invalid 均为零。

`selected_ema` 随后也完成 direct inference、双轴 replay 和 validator：

- `DECODE_CROSS_COMPLETE.json` SHA-256
  `0c6f87617b1cbd6a5bc4a6be6e9a5a2174f8a5a568c2f24db7253c15a315b8dc`；
  producer completion SHA-256
  `ddddd42174eb987cdeb723ae4422df8105e773bd7af74d31e67760dba20d74ff`。
- `validation_pass=true`、`fatal_log_findings=[]`、same frozen raw tensors、
  native/direct exact equivalence、reviewed-P0 direct exact equivalence 均为真；
  runtime commit/tree 与 v10 冻结身份一致，`new_training=false`。
- uniform-rank decode：
  Avg-mAP `0.41283020792762315`；mAP@0.3–0.7 为
  `0.648555172457041 / 0.5634561176089795 / 0.42615161707187166 /
  0.2774504499366236 / 0.14853768256360006`。
- physical-time decode：
  Avg-mAP `0.5009785403306161`；mAP@0.3–0.7 为
  `0.7226575722209517 / 0.6462475077859726 / 0.5282674811180088 /
  0.3876402290602233 / 0.22007991146792394`。
- physical-minus-uniform 为 Avg `+8.814833240299292` pp，按阈值分别
  `+7.410239976391075 / +8.279139017699311 / +10.211586404613715 /
  +11.018977912359967 / +7.154222890432388` pp；两种 decode 同样覆盖
  211 videos、1,584,000 个 NMS 输入和 422,000 个输出 prediction，invalid /
  filtered / rounding-induced-invalid 均为零。

`physical_online` 随后完成双轴 replay 和 validator，形成第三个正式
component completion：

- `DECODE_CROSS_COMPLETE.json` SHA-256
  `02384da2c71c93bdcd6ce003cd59451510c9d095e222653202f09f38b73b153f`；
  producer completion SHA-256
  `b9ba401a92e0d828aeabe48cb8972df74a64720a12f160d939daa355856aaf58`。
- `validation_pass=true`、`fatal_log_findings=[]`、same frozen raw tensors、
  native/direct exact equivalence、reviewed-P0 direct exact equivalence 均为真；
  runtime commit/tree 与 v10 冻结身份一致，`new_training=false`。
- uniform-rank decode：
  Avg-mAP `0.40107677185286417`；mAP@0.3–0.7 为
  `0.6148821307005312 / 0.5285472772022218 / 0.41348264750812447 /
  0.28615371812507695 / 0.16231808572836598`。
- physical-time decode：
  Avg-mAP `0.5755558109390063`；mAP@0.3–0.7 为
  `0.7704022473065874 / 0.7055742485050899 / 0.6207490393477052 /
  0.48593657950784275 / 0.29511694002780653`。
- physical-minus-uniform 为 Avg `+17.447903908614215` pp，按阈值分别
  `+15.552011660605613 / +17.702697130286804 / +20.726639183958074 /
  +19.97828613827658 / +13.279885429944056` pp；两种 decode 同样覆盖
  211 videos、1,584,000 个 NMS 输入和 422,000 个输出 prediction，invalid /
  filtered / rounding-induced-invalid 均为零。

当前三个 component 均为 `tested`，仍不是四条件路线终态。
`physical_ema` direct/native physical-time inference 已完成，Avg-mAP
`0.5760868491267752`，mAP@0.3–0.7 为
`0.7721224901972557 / 0.7045574192938243 / 0.6257613932435541 /
0.4900660583199814 / 0.28792688457926047`；direct metrics SHA-256 为
`43c33d551c19f4f3ab90108af30b13c103aa3a875fd87f00a4f50c7e5a83ecac`。
最后一次双轴 replay 正在运行，producer / `DECODE_CROSS_COMPLETE.json`
尚未写出，因此上述行保持 `diagnostic_only`，不是第四个 `tested` component。

Job `1203047` 仍在同一 allocation 内健康运行，当前未发现 hard failure。
路线总状态保持 `experiment_running`；必须等待最后一个 replay/validator
completion 与 explicit suite，才能形成正式性能/正负结论或升级
`empirically_supported`。

## 2026-07-29 v10 terminal suite failure and v16 exact recovery

此前“physical-EMA replay 尚未完成”的记录已被后续证据取代。v10
`physical_ema` 已形成第四个正式 `tested` component：

- completion / producer-completion SHA-256 为
  `a5c0c5248bf196d17f1cbf4f11a61d01459cb2ff3cfbf37541046fdb508b7ad1` /
  `8433bd22b620cd60300d94289cf991b69c1f64bcd5eacea557fbc463d7981086`；
- `validation_pass=true`、`fatal_log_findings=[]`、frozen raw tensors、
  native/direct exact equivalence、reviewed-P0 parity 均通过；
- uniform-rank Avg-mAP / mAP@0.3–0.7 为
  `0.40296498031949024 /
  0.622154649489393/0.5316588686305871/0.4113769771975965/
  0.2880843206041682/0.16155008567570622`；
- physical-time Avg-mAP / mAP@0.3–0.7 为
  `0.5760868491267752 /
  0.7721224901972557/0.7045574192938243/0.6257613932435541/
  0.4900660583199814/0.28792688457926047`；physical-minus-uniform Avg
  `+17.312186880728497` pp，mAP@0.6 `+20.198173771581317` pp。

Job `1203047` 随后在 explicit suite 以 `FAILED 1:0` 终止。失败不是 checkpoint
内容不一致：preflight 与 gate 的 resolved path、文件 SHA-256 完全相同；consumer
却把 preflight `{path, sha256, size_bytes}` 与 gate
`{path, sha256, epoch, state hashes, ...}` 做整字典相等比较。失败签名为
`decode_cross_suite_checkpoint_binding_schema_shape_mismatch_v1`，suite log
SHA-256 为
`68b7b3d34e587392bdac2df1eb2a36d971009d4c07165ef2a18157449ccb931f`，
failure receipt SHA-256 为
`42c394f11153a862819876b3915c34ca2ef0a68b6b62ed78a121d65db4269cec`。
因此四个 component 保持 `tested`，但 v10 不是完整 suite 结果，不能升级为
路线正/负结论。

协议不变的修复只做三件事：分别验证两种 checkpoint artifact record；比较
canonical resolved path + SHA-256；继续独立核验 online/EMA state-dict hash。
新增回归同时证明“同 artifact、不同 metadata schema”应接受，“同 bytes、不同
path”应拒绝。模型、配置、epoch59 checkpoints、seed42、数据、评价器和门槛均未
改变。

v11–v15 均为不可覆盖的零作业 pre-submission 根：

- v11 `runtime_profile_source_under_nounset_and_mode_preservation_v1`；
- v12 `recovery_exact_suite_invocation_scope_drift_v1`；
- v13 `preflight_repo_import_path_unbound_v1`；
- v14 `deployment_finalizer_base_relative_template_token_mismatch_v1`；
- v15 `ssh_transport_interruption_during_exact_recovery_launch_v1`。

唯一正式后继为 v16 runtime branch
`codex/sparsehead-evidence-recovery-20260729-v16`，commit/tree
`54e7f9abeaabf710a505f0a0f595a4eb3bb47f98` /
`f8490f9c25c2e0e6958c406e19c83cc3d5a40535`。Linux exact suite
`78 passed`，log SHA-256
`d81ca79bd9af216c106fb9718e7b171dd47c9aff3ddecb9787d8e0203c88d0fc`；
full-content preflight `validation_pass=true`，SHA-256
`ccc7a83e27b8d18ad0892b644e7338667b72d8eba3e3feedbc387dc4ac1d9a0d`。
`1203916` 只是 test-only；唯一正式 Job `1203917`（`ptdc-a1-r11`）已在
`g0045`/RTX4090 `RUNNING`。v16 gate focused tests 为 `45 passed`；四条件
real-CUDA gate 已通过，artifact SHA-256 为
`0d2153effee84a0e1aa6410125bb291eb4ef4d41e4b40604f49d9e5868e0ada9`，
`gate_pass=true`、`all_native_direct_exact_equivalence=true`，且
selected/physical × online/EMA 四项 `raw_tensors_immutable=true`。当前执行
selected-online 全量 direct inference，hard-failure scan 为空，尚无 v16
completion/suite。路线状态保持 `experiment_running`。

## 2026-07-29 v16 selected-online formal component

Job `1203917` 继续健康运行。`selected_online` 与 `selected_ema` 已完成全量
direct inference、两种解码轴 replay producer 和正式 component validator；
随后已进入 `physical_online` runtime preflight/direct 链。后续两条件和
explicit suite 仍未完成。

- direct 与 uniform-rank replay 的 evaluation-metrics SHA-256 均为
  `8860bdcaf3b998e6cddb1187c564d0bb0693496552439b104efad7145a6bd34c`，
  Avg-mAP / mAP@0.3–0.7 为
  `0.4125660433077075 /
  0.6450446628552113/0.5638932489689005/0.4266348135535575/
  0.27820781407261164/0.14904967708825695`；
- physical-time replay metrics SHA-256 为
  `7a032eaf8e4fc776ae0d670d572e02f74c23b82ef55bc29185e796e5be2f0f8b`，
  Avg-mAP / mAP@0.3–0.7 为
  `0.5015355102106833 /
  0.7227308252635063/0.6467256016774477/0.5253267642276429/
  0.39032810111601973/0.22256625876880035`；
- producer completion `DECODE_CROSS_REPLAY_COMPLETE.json` SHA-256 为
  `97410d9855a3f6db859e36213bf6b201e10c96941a164b5588af02cdfba4ee20`，
  `validation_pass=true`、same frozen raw tensors、native/direct exact
  equivalence、`new_training=false`。
- 正式 `DECODE_CROSS_COMPLETE.json` SHA-256 为
  `6937fc6b7b050fd7009ee967ceef446aebaa8b3daa695c7959106ff87048c038`；
  `status=tested`、`validation_pass=true`、`fatal_log_findings=[]`、
  reviewed-P0 direct exact equivalence、211 videos、每轴 1,584,000 个 NMS
  输入和 422,000 个输出均通过，invalid/filtered/rounding-induced-invalid 为零。

同一冻结输出上 physical-minus-uniform Avg 为 `+8.89694669029758 pp`。
这是 v16 第一个正式 `tested` component；在另三项 completion 与最终 suite
之前仍不能形成路线正负结论。

### v16 selected-EMA formal component

第二个正式 component 也完成全部合同校验：

- direct 与 uniform-rank replay 的 evaluation-metrics SHA-256 均为
  `ed3750a61a27dc70ac570f29ccefff8eef8d4dc10ea29802743b403807b82a34`，
  Avg-mAP / mAP@0.3–0.7 为
  `0.41283020792762315 /
  0.648555172457041/0.5634561176089795/0.42615161707187166/
  0.2774504499366236/0.14853768256360006`；
- physical-time replay metrics SHA-256 为
  `742b9a810f52dfe9bd12c29987148bf3c95e99c58aefb5774f2f8b3d18d30c1b`，
  Avg-mAP / mAP@0.3–0.7 为
  `0.5009785403306161 /
  0.7226575722209517/0.6462475077859726/0.5282674811180088/
  0.3876402290602233/0.22007991146792394`；
- producer / formal completion SHA-256 分别为
  `43c737fe3c5a9a534c565bf63e419fa152ee35b3be796ddf3f601c954fa52877` /
  `4a1b405b7849f396e1b649da8895070e6176023c4a959c6d7fd9148f2bd8afe0`；
  `status=tested`、`validation_pass=true`、`fatal_log_findings=[]`、
  frozen-raw/native-direct/reviewed-P0 parity、`new_training=false` 均通过。

同一冻结输出上 physical-minus-uniform Avg 为 `+8.814833240299292 pp`。
online 与 EMA 的 uniform Avg 只差 `+0.026416461991563 pp`，physical Avg
只差 `-0.055696988006725956 pp`；这是权重源近似稳定的描述性证据，不是
最终路线结论。当前有两个正式 `tested` component，`physical_online` 已启动；
另两项 completion 与最终 suite 仍是 `experiment_running` 的必要条件。

### v16 physical-online formal component

`physical_online` 已完成 direct inference、两种解码 replay producer 与正式
validator，成为第三个 `tested` component：

- direct/physical-time metrics SHA-256 均为
  `b68f2ad1393b59c40d58f7cfa1e450a52f84d8acbc80ad785a2d3a31352d6009`，
  Avg-mAP / mAP@0.3–0.7 为
  `0.5755558109390063 /
  0.7704022473065874/0.7055742485050899/0.6207490393477052/
  0.48593657950784275/0.29511694002780653`；
- uniform-rank cross-decode metrics SHA-256 为
  `0c258e563fe7b9886e6d56c9c3370b6536e187b521526318622b07ffcf1e4a4b`，
  Avg-mAP / mAP@0.3–0.7 为
  `0.40107677185286417 /
  0.6148821307005312/0.5285472772022218/0.41348264750812447/
  0.28615371812507695/0.16231808572836598`；
- producer completion SHA-256 为
  `d61d8fbf8b977b59b65eb87d55227904b2a5a2e6994e584226bda19a265b26eb`；
  正式 `DECODE_CROSS_COMPLETE.json` SHA-256 为
  `fd18348e6ae6ecf4bdc4390ca4620a109616582f7f77138ed137085e0df6c260`；
  `status=tested`、`validation_pass=true`、`fatal_log_findings=[]`、
  frozen-raw/native-direct/reviewed-P0 parity、`new_training=false` 均通过，
  hard-failure scan 为空。

physical-minus-uniform Avg 为 `+17.447903908614215 pp`，表明这一冻结
physical-axis 模型对 decode axis 极敏感。Job `1203917` 已进入
`physical_ema`；在第四条件与 suite 完成前，三个 `tested` component 仍不能
升级为路线终态或启动最终负结果归因。

### v16 physical-EMA formal component

第四个正式条件已完成并通过 validator：

- uniform-rank Avg-mAP 为 `0.40296498031949024`，mAP@0.3--0.7 为
  `0.622154649489393 / 0.5316588686305871 / 0.4113769771975965 /
  0.2880843206041682 / 0.16155008567570622`；
- physical/native Avg-mAP 为 `0.5760868491267752`，mAP@0.3--0.7 为
  `0.7721224901972557 / 0.7045574192938243 / 0.6257613932435541 /
  0.4900660583199814 / 0.28792688457926047`；
- uniform 与 physical metrics SHA-256 分别为
  `5058f789de9fd74544427fd8201d7b32cc83f18524409ee9e8f3b96fe32292dc` /
  `43c33d551c19f4f3ab90108af30b13c103aa3a875fd87f00a4f50c7e5a83ecac`；
- producer completion 与正式 `DECODE_CROSS_COMPLETE.json` SHA-256 分别为
  `aa6356a509898b94a38f2b9e0548c5f647cc6498655697b37fd39ea8982fc733` /
  `cd6da2f827524e0b9eb2b46c6cbbcc5b6e89243aa9cd8d7e45efafcb4cb6b565`；
- `status=tested`、`validation_pass=true`、`fatal_log_findings=[]`、
  frozen-raw/native-direct/reviewed-P0 parity、`new_training=false` 均通过，
  hard-failure scan 为空。

physical-minus-uniform Avg 为 `+17.312186880728497 pp`。至此四个 formal
component 均为 `tested`；但 Job `1203917` 仍在运行，explicit evidence suite
尚未落盘，因此路线继续保持 `experiment_running`，尚不启动最终模型归因。

## 2026-07-29 v16 terminal suite 与 Pro 级归因

### 终态与证据封签

Slurm Job `1203917` 已以 `COMPLETED 0:0` 终止，elapsed `02:34:30`，
node `g0045`。运行时 branch/commit/tree 与提交身份一致，仓库 clean，
hard-failure scan 为空。explicit evidence suite 已完成：

- `DECODE_CROSS_EVIDENCE_SUITE_COMPLETE.json` SHA-256
  `ed2770c35cf9a3acd5fa80465eda1c34b3541ba3dea404c75388aaeffefbdc31`，
  schema `phystime_decode_cross_evidence_suite_completion_v1`，
  `status=tested`、`validation_pass=true`、`new_training=false`、
  `fatal_findings=[]`，共扫描 13 份日志；
- `DECODE_CROSS_EVIDENCE_SUITE_VALIDATED.json` SHA-256
  `f2da143127b3a01aef7bda451e2351c494f72552f3810f604f895f4c0a7767d3`，
  `validation_pass=true`；
- suite 同时绑定 v16 commit/tree、preflight、CUDA gate、P0 gate/suite、
  四份 formal completion 与 checkpoint state-dict identity。

因此 v16 从 `experiment_running` 升级为 `tested`。它不是
`empirically_supported` 或 `paper_ready`：当前仍是一个 frozen、
single-seed、无新训练的机制实验。

### 四条件全部原始数值

| checkpoint | weight | uniform-rank Avg / mAP@0.3--0.7 | physical-time Avg / mAP@0.3--0.7 | P-U |
|---|---|---|---|---:|
| selected | online | 0.4125660433 / 0.6450446629, 0.5638932490, 0.4266348136, 0.2782078141, 0.1490496771 | 0.5015355102 / 0.7227308253, 0.6467256017, 0.5253267642, 0.3903281011, 0.2225662588 | +8.8969466903 pp |
| selected | EMA | 0.4128302079 / 0.6485551725, 0.5634561176, 0.4261516171, 0.2774504499, 0.1485376826 | 0.5009785403 / 0.7226575722, 0.6462475078, 0.5282674811, 0.3876402291, 0.2200799115 | +8.8148332403 pp |
| physical | online | 0.4010767719 / 0.6148821307, 0.5285472772, 0.4134826475, 0.2861537181, 0.1623180857 | 0.5755558109 / 0.7704022473, 0.7055742485, 0.6207490393, 0.4859365795, 0.2951169400 | +17.4479039086 pp |
| physical | EMA | 0.4029649803 / 0.6221546495, 0.5316588686, 0.4113769772, 0.2880843206, 0.1615500857 | 0.5760868491 / 0.7721224902, 0.7045574193, 0.6257613932, 0.4900660583, 0.2879268846 | +17.3121868807 pp |

四个同 checkpoint 的 P-U 对比同号且幅度大，@0.6/@0.7 的提升尤其明显：
selected-online 为 `+11.2120/+7.3517 pp`，physical-online 为
`+19.9783/+13.2799 pp`。online/EMA 的 Avg 差均不超过 `0.1888 pp`，
不足以解释轴差。

### 与 controls 的合法比较

- full60 旧结果约为 selected `41.28`、physical `57.57` Avg-mAP；
  P0 full-precision 为 selected-EMA `41.283021`、physical-EMA
  `57.608685`。v16 的 selected-EMA uniform 与 physical-EMA physical
  精确复现 P0 数值，说明本次链路可复现，并关闭 rounding/full-precision/
  NMS-precision 混淆。
- 固定 physical decode 时，physical checkpoint 相对 selected checkpoint
  online/EMA 仍高 `+7.4020/+7.5108 pp`；这是跨 checkpoint 的描述性差异，
  不能归因为训练因果。
- 同样，physical checkpoint 相对 selected checkpoint 的 decoder-axis gain
  多 `+8.5510/+8.4974 pp`，提示训练坐标与解码坐标存在耦合，但仍需
  assignment/support 观测才能给出训练机制因果。
- matched 20-epoch 历史 selected/physical/SDPQ 为
  `30.42/44.88/30.88`；SDPQ 没有随 physical control 一起恢复。因此 v16
  支持物理时间解码轴，不支持“当前 SDPQ 已被救活”。

### 诊断分解

suite 的 shared observation 是 792 windows、192 native tokens、378
candidates、20 classes、3325 GT。按动作时长分层（short `<=1.7s`，839 GT；
middle 1660；long `>6.1s`，826）：

- physical-online checkpoint 在 physical decode 下，全体 mean-best-IoU
  `0.8003`，oracle proposal recall@0.5/0.7/0.9 为
  `0.9660/0.8523/0.1967`；uniform decode 为
  `0.7608` 与 `0.9450/0.7218/0.1504`。
- short 动作的差异最大：physical decode mean-best-IoU `0.7153`、
  recall `0.8725/0.6210/0.1132`；uniform 为 `0.6593`、
  `0.8081/0.4577/0.0632`。这与高 tIoU mAP 改善一致。
- selected-online 的 physical decode 也把全体 proposal recall 提升到
  `0.9359/0.7666/0.1528`；其 selected-uniform short @0.7/@0.9 recall
  仅 `0.2932/0.0250`，显示 selected-rank 几何对短动作边界最不友好。
- within-checkpoint 决策并非简单逐项等价：physical-online 两轴仅
  `78.41%` 的 422k 输出匹配，边界偏移 p95 约 `1.79/1.76s`，
  rank delta median/p95 `217/1355`。这足以通过 NMS/ranking 放大，但
  oracle recall 同时改善，故不能把全部收益归因于 score calibration。

当前 suite **没有** class-wise AP、ECE/Brier/校准曲线、独立 NMS 决策计数、
failure-sample IDs、assignment/support observability 或端到端成本。上述缺口
必须写作未测，而不能由 aggregate mAP 推断。

### 竞争解释、反证与可证伪预测

1. **首要解释：非均匀观测被 uniform-rank 解码，造成真实时间坐标失配。**
   支持证据是四个冻结 checkpoint/weight 条件全部产生
   `+8.81--+17.45 pp` P-U 增益，高 tIoU 与 short recall 同步改善。
   反证边界是 selected checkpoint 即使用 physical decode 仍只有约 `50.1`，
   未达到 physical checkpoint 的约 `57.6`。可证伪预测：独立实现的
   float64 `q->t` mapper、stable-sort/NMS 与官方 evaluator 应复现四行方向和
   近似幅度。最小决定性实验是不训练、只对封签 raw tensors 运行一套独立
   mapper/NMS/GT evaluator。
2. **次要解释：physical training 的 assignment/support 改变了表示，形成与
   physical decode 配套的 checkpoint。** 历史证据显示 physical assignment
   的 positive/no-eligible 分布，尤其短动作，与 rank assignment 不同；当前
   2x2 也显示 physical checkpoint 只在匹配 physical decode 时显著领先。
   反证是当前 suite 没有直接记录 assignment/support，且物理解码本身已解释
   大量收益。可证伪预测：残余 selected-vs-physical 差应集中于 short、
   no-support、zero-assigned queries。最小实验是在 64 个固定 train-side
   windows 上导出 positives、no-eligible、assigned-count、support mass，
   不更新模型。
3. **竞争解释：score ranking/NMS 放大边界映射差异。** 输出匹配率、rank
   delta 与秒级尾部边界偏移支持此解释；但 pre-NMS oracle recall 本身也明显
   改善，反对“纯后处理假象”。可证伪预测：若 NMS/校准主导，NMS-off/fixed
   top-K 会显著缩小 AP 差而 oracle recall 基本不变。最小实验是封签预测上的
   independent stable sort、NMS-off、fixed-topK 与仅用 training split 拟合的
   temperature replay。
4. **单 seed/checkpoint 偶然性不是首要解释。** online/EMA 与 P0 数值高度复现，
   反对纯随机噪声；但没有多 seed，因此不允许鲁棒性 claim。最终需 paired
   video bootstrap 与多 seed 验证符号一致性。

实现/评价错误已被 exact gate、raw immutability、checkpoint/state hashes、
native-direct exact parity、生产 evaluator 重算、零 invalid counters 与空
fatal log 大幅排除。剩余审计风险是 mapper/NMS/evaluator 仍共享生产实现，
GT parser 没有独立重建，以及数值容差和 single-seed；这些由解释 1 的独立
CPU replay 直接裁决。

### 排序后的判断与路线裁决

1. decoder/native temporal geometry mismatch：高置信；
2. assignment/support 与表示耦合：中等置信，缺直接 observability；
3. ranking/NMS amplification：中低置信，可能放大但不是全部原因；
4. 单 seed 偶然性：低置信作为唯一原因，但限制外推。

这不是 Approach A 的整体负结果：physical-time decode 假设得到强而一致的
`tested` 支持。负结果落在两个更窄的主张上：selected-rank decode 并不无害；
当前 SDPQ/SparseHead 也没有因解码修复而自动成为优于 physical control 的方法。
路线建议是 **继续但修改**：以后把 physical-time-before-NMS 作为必须匹配的
几何基线，先做独立 evaluator 与 assignment/support 审计，再决定是否修改
SparseHead 核心训练。不得从本分析静默启动重训。

按信息增益排序的下一步设计：

1. 封签 artifact 的独立 mapper/NMS/GT evaluator（CPU，无训练）；
2. 64-window assignment/support observability audit（无训练）；
3. 从现有 artifact 补 class、calibration/NMS、failure-sample 分解；
4. upstream-native K=192/K=384 parity；Avg 差 `<=0.5 pp`、@0.7
   `<=1 pp` 才视为几何闭合；
5. 通过以上门禁后才做 SDPQ micro-overfit/one-step support-null gate；
6. 机制通过后做多 seed 与 paired-video bootstrap；
7. 最后补 decode->NMS 全栈 latency/memory/energy ledger。

### Claim boundary

本实验只证明：在一个 frozen single-seed THUMOS replay 中，同一 raw tensor
artifact 采用 physical-time-before-NMS 解码，稳定优于 uniform-rank 解码。
它不证明 assignment 的训练因果、多 seed 鲁棒性、SparseHead/SDPQ 优于
matched physical control、计算节省、跨 backend 泛化或 paper-ready claim。

## 2026-07-29 independent-closure implementation

The next no-training closure is now implemented and locally `tested` at
`codex/sparsehead-diagnostic-closure-20260729@57917e7bf2b991478b4f6fc4ce1db5ca5878b68d`
(tree `aaf7c82bd837078bb7276baf6c0a504da0684194`), with `35 passed`.

The independent replay reconstructs rank-to-seconds geometry, stable top-k,
Gaussian Soft-NMS, duplicate-GT semantics and VOC2011 AP in NumPy/float64. It
does not import OpenTAD decode, NMS or evaluator helpers. It fails on
non-finite/invalid tensors, source artifact drift, native-point error,
proposal/result mismatch, metric mismatch or sign reversal. The frozen v16
metrics remain unchanged until the remote four-condition run produces formal
independent completion artifacts.

A separate 64-window audit seals sample/input/mask/GT/support hashes before
model observation, rebuilds the same loader, and compares production
classification/offset/segment/endpoint targets with an independent assignment
implementation. Its duration buckets are `<1s`, `[1s,4s)`, `[4s,16s)` and
`>=16s`. It is always `diagnostic_only` and requires an exact SDPQ checkpoint;
it cannot be run on the v16 G1a ActionFormer checkpoints.

This implementation does not upgrade the experiment above `tested`. The
independent remote replay, SDPQ support artifact, class/calibration/failure
decomposition, official ActionFormer anchor and matched controls are still
missing.

## 2026-07-29 remote independent-closure execution

The closure branch advanced through three fail-closed contract repairs and is
now frozen at
`6d74ad7b7c7736bbff48976a626b951512a54e96` /
`80cd2431ebf9809f03ab1216b84b45380d51f33b`. The repairs do not change any
v16 model, checkpoint, raw tensor, NMS setting or metric:

- `independent_recompute_padded_axis_nonfinite_scope_v1`: the first validator
  incorrectly required padded axis arrays to be globally finite. The real
  792-window capture has zero non-finite/non-increasing values in every valid
  prefix and exactly 1,443 NaNs in each contractual padding tail. The repaired
  validator accepts only-NaN padding while still rejecting non-finite or
  non-increasing valid prefixes and finite padding.
- `independent_recompute_annotation_subset_contract_v1`: the second validator
  used the logical evaluation name `test` directly against the OpenTAD
  annotation. The frozen annotation actually contains 200 `training` and 211
  `validation` videos, with 3,003/3,325 GT instances. Policy now explicitly
  binds logical `test` to annotation subset `validation` and requires
  211 videos, 3,325 GT and 20 classes.

Local focused verification is `46 passed, 1 skipped`; the clean Linux v5
runtime passed `58 passed, 1 skipped`. The first two failed run roots remain
preserved. The fresh four-condition independent recomputation is running at
`.../runs/sparsehead_diagnostic_closure_20260729_v3`; until its formal JSON
passes, the original v16 result remains only `tested`.

The separate SDPQ audit is bound to the exact clean historical config repo
`4a57577193c07cc90ac0867176aa79c76f637c36`, epoch-19 online checkpoint
SHA-256 `40fccfd854a88903aaf795c04b94068af4007663c5d63064201990d70b2c3fc7`
and VideoMAE SHA-256
`4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251`.
The audit tool now requires the expected checkpoint epoch explicitly and never
substitutes online for absent EMA. Job `1204961` failed before Python because
Slurm `--wrap` invoked `/bin/sh` with unsupported `pipefail`; the preserved
successor Job `1204981` uses an explicit `/bin/bash -lc` wrapper and is pending.
Neither job is a performance experiment.

## 2026-07-29 publishability-gate closure and active evidence jobs

The clean evidence implementation is now
`2b074845497f6ada3314cb895f0d4ab2f4ce3eca` /
`7779862c5422dc8e527b304bf881a760b0c90625`; its exact Linux runtime passed
`95 passed, 1 skipped`. This remains tool-level `tested` evidence and does not
alter the frozen v16 tensors or metrics.

Official ActionFormer Job `1205131` generated raw predictions and official
evaluation at Avg-mAP `66.83`
(`82.13/77.81/70.95/59.40/43.87`), but the old record builder failed with
`official_annotation_split_schema_contract_v1`. The mismatch was between
nominal THUMOS test cardinality 213 and the pinned annotation database's 212
case-normalized `Test` entries, not a model error. The raw artifact contains
42,400 predictions over exactly those 212 evaluated videos. The hardened audit
also validates the canonical 20 classes, all 413 `T×2048` feature files and the
single feature-only unannotated video. Unique successor Job `1205178` must
produce `main_table_eligible=true`; until then `66.83` is not a paper result.

SDPQ support Job `1205132` failed on
`sdpq_support_overlap_query_padding_mask_omission_v1`. The support-overlap
branch had not zeroed padded queries after epsilon width clamping. The repaired
production path multiplies mass by `query_mask` after both branches, with
focused proof that valid queries and their gradients are unchanged. Unique
successor Job `1205179` is diagnostic-only. Independent replay Job `1205133`
continues without a duplicate.

The experiment state remains `experiment_running`: the official anchor,
independent replay and support artifact are pending. Even if all pass, the
decode-cross evidence can support only the causal claim that physical-time
decoding improves fixed frozen outputs. It cannot make the v16 VideoMAE/K384
numbers, historical `63.61`, or matched `30.42/44.88/30.88` officially
ActionFormer-comparable. Any future matched row must first pass a live
base-anchored source-diff attestation and then use the identical official I3D,
annotation, schedule, checkpoint rule, NMS and evaluator.

## 2026-07-29 official gate hardening and independent-sort closure

Official released-checkpoint evaluation Job `1205206` completed with exact
mAP@0.3--0.7
`0.8213398807/0.7780557087/0.7095360790/0.5940167328/0.4387211844`
and Avg-mAP `0.6683339171`. Raw prediction SHA-256 is
`1333df9202eec7ae217542b6bd2b15b597c1a004ebb3634de54a7a37adb6d7fe`.
The values are real, but the old record falsely labels the seed as `0`; the
official config and released train log use `1234567891`. The row is not yet
paper-main-table eligible and must be re-signed with full effective-config
SHA-256 `835cf30fbcfd27bd6af8885fff002813c8596e2948fce3adf29e3716f316dde4`
and all 15 live receipts.

The source-diff/effective-config gate is clean at
`e2a0d74f561b158c531d4909e72ecee69b153c16` /
`0b6cb7996ee90f3209a78b78bbf7a55525e3badd`. It verifies the official base,
remote refs, regular non-executable blobs, exact allowed paths, protected
config loader, full expanded base/candidate configs and protected optimizer,
schedule, backbone, loss, NMS and evaluator fields. Linux v14 passed
`125 passed, 2 skipped` (log SHA-256
`41548be21c3f3aad6246a4782ff5feac9e9ad00d20f8a8d5616152399028ba5c`);
the current numeric closure passed `127 passed, 2 skipped` in v16 (log SHA-256
`115dd497a3a662b3fc0f19ae9104257d245cbadbb7fd4001f3eb3ea71432534c`).

Independent recomputation Job `1205243` failed with signature
`independent_recompute_semantic_match_drift_v1`, not a model-negative result.
Across all eight arm/axis reports, candidate masks and scores are bit-exact,
proposal error is zero and every physical-minus-uniform sign is preserved.
The largest metric drift is `0.00185658`. The old closure used stable
NumPy/float64 ranking and Soft-NMS while production uses PyTorch 2.0.1 CPU
unstable sorting and float32 C++ Soft-NMS. The revised v3 closure pins the exact
PyTorch sort identity, independently ports the C++ scalar float32 algorithm,
and seals `libm.so.6/expf` with a float32 bit probe.

Job `1205388` (`1205384` is test-only) failed in one second with exit `127:0`
before the validator started. Failure signature
`slurm_module_function_unavailable_v1` records that the non-login Slurm
allocation did not export the shell `module` function. The immutable v9 root,
logs and receipt are preserved. A clean-shell probe proved that the pinned
Conda activation script resolves the exact Python and PyTorch `2.0.1` without
modules. The only successor is Job `1205400` (`1205398` is test-only), using
`.../runs/sparsehead_diagnostic_closure_20260730_v10`; submission receipt
SHA-256 is
`da679424ad5a3dbfd3cc0b6e28fd74b638d2bc7873098a4d1c46a2e80c14bea2`.
Its finalizer still requires exact pre-cross and post-NMS detection equality
for all eight reports, the existing metric bound and unchanged delta signs.
Until it passes, the route remains `experiment_running`.

Separate SDPQ support Job `1205240` completed its diagnostic-only 64-window
audit: 647/647 reserved GT matches, zero missing assignment/evidence/domain
rows, zero ownership collisions or uncovered positives, and maximum offset
error `3.0517578125e-05`. This supports observability on the sealed sample only;
it does not rescue the negative SDPQ performance result.

Official anchor Job `1205409` failed before inference with
`official_environment_probe_nms_import_order_v1`: its new probe imported the
NMS extension before `torch` had loaded `libc10.so`. NMS build and ABI smoke
passed; no metric was produced. Job `1205419` then completed both official
evaluation passes and reproduced Avg-mAP `66.83`, but the record builder failed
closed with
`official_released_train_log_default_serialization_omission_v1`. Exact
comparison found one difference only: the released training log omits
`model.fpn_start_level`, while the pinned official loader injects the exact
integer default `0`. Failure receipt SHA-256 is
`079818253bc87a78ed67ce41dbd092aa64f0e54b5a61972f2313adeb7d10fa4a`.
Thus the metric remains diagnostic and not paper-main-table eligible.

Audit commit/tree
`8b80c98ee2af65561bf305b4fdc2ef16e460da73` /
`148a93eac4ff1b6a3be46fdca72c705aa17294a6` implements a fail-closed
normalization attestation: it pins the released raw-config SHA-256
`ad426e1a25be48423e21f854bbc6d815c6063388811350ad5fada5ac8933d3a7`,
permits only the documented missing upstream default, hashes both raw and
normalized configs, and still requires exact equality with source-expanded
config SHA-256
`835cf30fbcfd27bd6af8885fff002813c8596e2948fce3adf29e3716f316dde4`.
Focused tests are `44 passed, 1 skipped`; the new clean Linux runtime is v18
after preserving a v17 GitHub TLS clone failure. Full Linux preflight passed
`131 passed, 2 skipped`, log SHA-256
`6899bf6126d1ce9b3d880d348cdf5c1f152235d3b2e6f6de028b5fc807fb34fb`.
Unique formal anchor Job `1205455` (`1205454` is test-only) completed `0:0`
under fresh v3 root; sbatch/submission receipt SHA-256 values are
`76fc3df0c1faadfc9f62fb2982a8aee6013fc4a6a502cd32dbdf3e27fd7ec0a7` /
`8f26fbed8284f83d6d099779f88c76d61ee0181323d208f355aff10dbb426744`.
All 15 receipts pass. Independent mAP@0.3–0.7 is
`0.821339880697554/0.7780557086995361/0.7095360789567791/`
`0.5940167327663141/0.4387211844309326`, Avg `0.6683339171102232`.
Completion/protocol/verdict SHA-256 values are
`90c8bae14fcb20cc2434cea37f47065704766e38ff9663eac6e70c0d338b9e94` /
`808199b54b0ebcfebda403419873cc5fd46c36a4d404d3d8ce31838ce3b5bd95` /
`0706247ef978bf339f9a9cb4adaef07500e8d991129c6d0862118088b917a2ec`.
Verdict: `official_actionformer_protocol_match=true`,
`main_table_eligible=true`. The official anchor is
`empirically_supported`; K384 is now authorized only as an exact matched
method intervention and the overall SparseHead route remains
`experiment_running`.

## 2026-07-29 v16 terminal suite and official-native K384 handoff

Job `1203917` is now terminal `COMPLETED 0:0` after `02:34:30`.
All four formal decode-cross completions and the explicit evidence suite pass.
Suite completion/validation/deployment SHA-256 values are
`ed2770c35cf9a3acd5fa80465eda1c34b3541ba3dea404c75388aaeffefbdc31` /
`f2da143127b3a01aef7bda451e2351c494f72552f3810f604f895f4c0a7767d3` /
`bc825f08445e4c8fe8f3ab5dd768b6f9cdf3ec7fdd40dc02438428237c004b2e`.
The four metric rows remain exactly:

- selected-online: uniform/native `41.256604`, physical-time `50.153551`;
- selected-EMA: uniform/native `41.283021`, physical-time `50.097854`;
- physical-online: uniform-rank `40.107677`, physical/native `57.555581`;
- physical-EMA: uniform-rank `40.296498`, physical/native `57.608685`.

This closes the frozen single-seed inference-axis replay at status `tested`.
It establishes that physical-time decoding materially improves the same frozen
raw tensors. It does not make the VideoMAE/K384 checkpoint comparable with the
official ActionFormer `66.833392`, and it does not establish assignment
causality, multi-seed robustness or compute savings. Independent exact
recomputation Job `1205400` remains the sole active closure and is not
duplicated.

The official-native successor is candidate commit/tree
`55763a9ef7ce18a51827fe48040081c4fe2b84d4` /
`c489a54aa501b39421cddb5df98385b3889ed479` on
`codex/actionformer-sparsehead-official-matched-20260730`. It preserves full
official I3D observation, backbone/FPN, physical coordinates, point generator,
optimizer/schedule/seed/EMA and evaluator. Its one declared method
intervention is deterministic K384 FPN-head query execution plus the explicit
`selected_native_grid_queries` training-loss support. This is not described as
an execution-only or unchanged-loss intervention.

Linux candidate tests passed `11 passed`; audit/source-diff tests passed
`40 passed`; launcher tests passed `5 passed`. The exact preflight hashes are
candidate `1174f5b4036458a598d20e06913fef9bf2561ded1b2c9f8c675e627304be6b3b`,
audit `5890bde1cbdc8ad72edabcfe48ffff36d4d1028c81eab67cd1db11b3e2a25b39`,
launcher `1fe0a5d05103e725adb9b8255b68b09f479d93a75d94d241e9f791b04b716537`
and live source-diff
`409ffd3035a0c957d3b250db24fe017c5c09efda526d746ace0d54f00c695abc`.
Audit/launcher commit/tree are
`aab72e484538931a565930b99d1beb71f47b9ceb` /
`25e7e0eb3b8cd5edfb48eac594eda6b89edffa36`.

Unique real-CUDA engineering gate Job `1205541` is pending under immutable root
`.../runs/actionformer_native_grid_k384_cuda_gate_20260730_v1`; `1205539` is
test-only. Submission receipt SHA-256 is
`471022b2e726cf923e5a445aef8c21ca5f17c9e59b7e586ed8fb3ed4bbc49665`.
The gate requires official-checkpoint selected-output equivalence, raw tensor
immutability, exact K384, theoretical head-MAC reduction and at least `1.05x`
isolated-head median speedup in every one of three interleaved CUDA timing
rounds. It explicitly forbids a paper metric or end-to-end wall-clock claim.
The route remains `experiment_running`.

## 2026-07-29 CUDA cost failure and packed-kernel successor

Job `1205541` reached `FAILED 2:0` after `00:00:30`. Correctness passed:
selected outputs matched the official checkpoint within maximum absolute error
`4.0531158447265625e-06`, raw tensors and selected masks were immutable, and
unselected outputs were exactly zero. The theoretical head-MAC fraction was
`0.3359251591135649`. The isolated cost gate failed decisively: dense mean
`6.191821147998174 ms`, sparse preselected mean `19.65075126952595 ms`,
sparse-with-selector mean `20.504609247048695 ms`, overall speedup
`0.30091287378297016`; all three rounds were approximately `0.30x`.
Failure signature:
`native_grid_sparse_head_microkernel_launch_and_scatter_slowdown_v1`.
CUDA gate/failure-analysis SHA-256 values are
`8aeb2cdbf02da0f8ad675b2f5a33d3ef6d89198ac7e216511ffde45d66f505a3` /
`ef6b462d79316e2c3f80bf125eb8704b30c0c3e229568048b67095a172152b7d`.
This is an engineering negative with no model metric.

The root cause is the per-sample/per-FPN sequence of tiny Conv1d launches plus
repeated physical scatter allocation. Candidate commit/tree
`d64e66dfd7fc9881552b342f5523926cc78c0848` /
`16265c70b235034acb52521b00c259ec6d8b59e1` packs all samples and levels into
one Conv1d launch per head layer while preserving physical neighborhoods,
mask-hole semantics, zero boundaries and autograd. A focused regression locks
the three-layer head to exactly three Conv1d calls. Clean Linux tests passed
`12/12`; source-diff SHA-256 is
`5aea817bf1fd1b2c0e36193b9d99ee71bde3dfd00c05673ece5dc4f6da9304d4`.

Unique successor Job `1205567` is pending under immutable run root
`.../runs/actionformer_native_grid_k384_cuda_gate_20260730_v2`; `1205566` is
test-only. Deployment/submission receipt SHA-256 values are
`c2890c1b37e22810fdc8284b80ca6292e7bf5cc1c38820fb74e8d68d96647b52` /
`f71c394c09f5d5a65bdf37036739294d553098ea3ecfa79b8ebf10c8486b3798`.
One pre-submission SCP interruption left no partial file and was recovered by
hash-verified retry; signature
`ssh_transport_interruption_during_pre_submission_receipt_copy_v1`.
Status remains `experiment_running`; training is still blocked by the gate.

## 2026-07-29 packed GEMM negative and bounded global-state recovery

Job `1205567` reached `FAILED 2:0` after `00:00:28`. Its numerical contract
passed (maximum selected error `2.86102294921875e-06`, immutable inputs/masks
and exactly zero unselected outputs), but dense/sparse-preselected/sparse-with-
selector means were `6.240900/12.765710/13.618182 ms`; selector-inclusive
speedup was only `0.4590397x`, with all three rounds near `0.459x`.
Failure signature:
`native_grid_sparse_head_packed_patch_materialization_and_microconv_slowdown_v1`.
CUDA-gate/failure-analysis SHA-256 values are
`f4a0479b48c434832c45d84e9eccc6ebc9e56be88a03d8e8eff4fca525981113` /
`fe2f6d62272ad558be18e068ca1796808d516b105b2ed41202eb5a7e0e1fb6d6`.
This is an engineering cost failure, not a model metric.

Candidate commit/tree `31e6112ea28747098cfe5412c097d737731bfaa1` /
`d2619cd075c4e7192ca060f34d811ac3fe5768f8` replaced the length-three packed
Conv1d call with algebraically equivalent flattened `F.linear`; Linux focused
tests passed `12/12`. GitHub HTTPS clone v4 failed with
`github_https_clone_tls_termination_v1`; a first bundle clone v5 had no remote
HEAD and failed closed as `bundle_clone_remote_head_unset_v1`. The SHA-verified
bundle
`actionformer_sparsehead_31e6112_c50bea0b.bundle`
(`c50bea0b79e242bb4c96cf11fb35a3ef095a8b9c3bc4a13fc56abca02be4ec49`)
then produced exact clean runtime
`/data/run01/sczc063/yuzibo/projects/actionformer_sparsehead_official_matched_20260730_v6`.

Remote live source-diff ref resolution failed before the engineering gate as
`github_remote_ref_dns_timeout_during_source_diff_v1`. A clean local
recomputation against live official/candidate GitHub refs generated source-
diff attestation/provenance SHA-256 values
`3ef485f82678453538aef6f58ba81d548149394ef93356a811593e67cdf22e9d` /
`780c0aa5a8a00ba9180974d4bee001782e83d747d492589a2a2da4b5bc40e2d6`.
The provenance explicitly sets `paper_main_table_seal_allowed=false`; it was
used only to unblock an engineering gate and must be recomputed live on a
paper-sealing host.

Job `1205569` then reached `FAILED 2:0` after `00:00:28`. Correctness still
passed at maximum error `4.5299530029296875e-06`, but dense/sparse-preselected/
sparse-with-selector means were `6.193413/12.764605/13.546889 ms`, overall
speedup `0.4576558x`, and per-round speedups
`0.4572781/0.4579003/0.4572336`. The unchanged latency after replacing
convolution arithmetic isolates the remaining bottleneck to packed gather,
materialization, concatenation, repeated hidden-state dense scatter and
duplicated cls/reg planning. Failure signature:
`native_grid_sparse_head_packed_gather_scatter_overhead_v1`.
CUDA-gate/stdout/stderr/failure-analysis SHA-256 values are
`7e91345babcce40bb9a157d2b29fbc718fe7f0e2a059bdc02e2edff386709197` /
`fb3abb66cf7690fd7965165d409039ea1701ac3ab0c4027e4f94652863373afa` /
`4c6d87aa6b85dbbe173a1eae119bac562aa43f55c6fa847b256c9c05d25c79e0` /
`8b49859031a48ef2a4367a156f452761c66a1e75c1a5e6a87a8fb242766f3a50`.

One final global packed-state implementation is now `designed`: cls/reg share
one physical-position plan, the first layer gathers raw official FPN values
with exact mask-hole semantics, later layers carry sorted sparse states and
resolve neighborhoods by exact physical-index lookup, and only final selected
outputs scatter to dense API tensors. Its unchanged real-CUDA gate has a hard
stop: below `1.0x` it is execution-inviable and no matched training is
authorized; `>=1.05x` remains the formal gate. Even a pass is engineering-only.
The paper row still requires same-commit official dense and K384 sparse
terminal epoch-35 EMA retraining, the same 212-video split/evaluator, sealed
receipts, independent seeds and end-to-end synchronized cost. The released
official `66.833392` checkpoint is an anchor, not a causal matched delta.

The global packed-state design is now `tested` in candidate commit/tree
`d86a4acda21e35a1609f19f1a46bc470ee18b7e1` /
`327c032a1ab3c14d0e34d6339df36f8a33ec6907`. It inserts zero guards between
jobs, builds one global physical dependency plan, shares precomputed gather
indices and first-layer patches across cls/reg, keeps hidden state sparse and
scatters only final outputs. A radius-two cross-job contamination regression
and the full focused candidate suite pass `14/14`. Audit commit/tree
`14bd14f9b6a087dc2ec623fc4238c89e0cb86960` /
`b782404ddef9a65f19fea70fbf993a3a9d6e0420` passes `18/18` exact focused
tests. Full-content preflight/deployment/submission SHA-256 values are
`08b05123edbaccd10d5b43031a43ebac11a3616ceb454bfbd588d4d7395a6a95` /
`f070f46f023be6152faf1818342633a8d6f713fb55e37fa5c79fc2a43434f140` /
`04d206c3ad220155f8f63a1b6a086c6c3c6c5beaeac13a7a001334f2d0fef4c7`.

Unique CUDA gate Job `1205571` reached `COMPLETED 0:0` in `00:00:30` under
immutable root `.../runs/actionformer_native_grid_k384_cuda_gate_20260730_v4`;
`1205570` is only the `sbatch --test-only` identifier. The formal gate passed:
dense / sparse-preselected / sparse-with-selector median latency was
`6.240573 / 3.129646 / 3.970906 ms`, giving selector-inclusive median speedup
`1.571574x`; all three synchronized rounds passed at
`1.573072 / 1.570395 / 1.568621x`. Maximum selected-output error was
`4.529953e-6`, prefix error `3.814697e-6`, unselected nonzeros were zero, and
raw/selected masks were immutable. Gate/completion/runtime-log SHA-256 values
are `cddfb80af237a41d3c3e1121e39cbc5114ad8abc472c56f6daf519a50cf95988` /
`ceec00f799eb40a1dd56c1949576783e06599205d63f1d1909a598787d99fd85` /
`f3f4b13be3433d2307ce10a8370ab168d8af00368060e61229441e27131cb0f5`.
Status is `tested` isolated-head CUDA engineering evidence only:
`paper_metric_claim_allowed=false` and
`end_to_end_wall_clock_claim_allowed=false`. It unlocks, but does not replace,
official same-commit dense/sparse retraining.

## 2026-07-29 official matched ActionFormer screening deployment

The paper-comparability substrate is now independently sealed before training.
Remote live GitHub-ref source-diff recomputation binds official
`61ea7eb9308a568b0cf45e3804830836e30061de` to candidate
`d86a4acda21e35a1609f19f1a46bc470ee18b7e1`; its attestation SHA-256 is
`a07d038d87632d1f8cc984ba24af44ca7ce9a9902e30e501f5de80a32265d46b`.
A live rehash of all 413 official I3D files reproduced the sealed feature
manifest exactly at
`cda269dace393b9af1f6fcb87a9a531beed69e3c71279ba3ca2cee76e198d59a`;
the annotation remains
`3b025685a07fb98fc58d2399fb5fa9493c2168632d8ae1a8c3f4689897d2fbb2`.

Audit commit/tree `643c42e8cfe4018fb891202f7ffdae554acc2e4a` /
`25fa3eda9fc62960c69c2952c957ebab39e71c27` adds a same-candidate-commit
dense/K384 pair launcher and an independent pinned-official evaluator of raw
predictions. Exact N16R4 focused tests pass `18/18` (log SHA-256
`f15e5d2c6b8cfeba5a31489b318f3e784233ddc8880fd09966e98e6ff63fcded`).
The launcher fixes seed `1234567891`, official 5-warmup + 30-epoch execution,
terminal `epoch_035.pth.tar`, EMA evaluation, the official dataset and
evaluator, and forbids resume.

One pre-Slurm data-revalidation invocation failed before model execution due
to repository import scope; the first attempt to write its receipt also used
the unloaded system Python. Preserved signatures are
`official_data_live_revalidation_import_scope_v1` and
`preflight_failure_receipt_python_environment_unloaded_v1`; failure receipt
SHA-256 is
`2cd20095d49566761ed8feb16af7989d96cbe57d2b5441f10e12fa2504ababde`.
The new v2 root passed after activating the pinned environment and binding the
audit root on `sys.path`; no training or model result existed in the failed
root.

Full-content preflight SHA-256 is
`3b827cfe10b3267d013373f89a9c3b90b2eb6f450b0aa4b7d1e5082615a0ac4e`.
Deployment/submission SHA-256 values are
`65cb544960c619f4243c7829a41950719d2591493c05fbad70a07f1b9a037da2` /
`ead6f35af71e2de9308d6ed0aad642dc27845e68169f1cee8ca32e3d157a3e77`.
`1205572` is test-only; unique formal Job `1205573` is `PENDING` under
`.../runs/actionformer_official_matched_pair_k384_seed1234567891_20260730_v1`.
The route remains `experiment_running`. This pair is preregistered
single-seed screening and is forced to
`paper_main_table_eligible=false`; a paper result still requires the
multiseed paired suite plus synchronized end-to-end cost.

Job `1205573` then terminated `FAILED 1:0` in `00:00:31`, before focused tests
or either training arm. The pre-submission live-ref attestation had already
passed, but the launcher redundantly called `git ls-remote` inside the compute
allocation and g0024 could not resolve GitHub. Failure signature:
`compute_node_github_dns_during_redundant_live_source_diff_revalidation_v1`.
Failure/runtime/stderr SHA-256:
`f0bf8fe6258260d55fffe88d35dfb75d647340adccb06dc2efae1c5e419c64d9` /
`8bc85a66f37bc98eec780ec76ef5fab1978bd45195917780c269267dc5b2a057` /
`fef8ce4b812cf04882328f4f12a5ddcac8c61077a1f8107c19b63e142808d74b`.
This is engineering-only and contains no model result.

The recovery retains the live-sealed attestation but recomputes its local
commit/tree, cleanliness, remote URLs, config blobs/expansion, binary/name
diff, exact allowlist and policy without a network request inside Slurm.
Audit commit/tree `debbde469f938e09e4debfe7831e64755ae665f5` /
`3721612aae55eecb07e9f4183a53e1d8156e143b` passes `19/19` focused tests
and an actual remote offline-snapshot validation. Test/validation SHA-256:
`d3d76af3095d792b6af0a8709a7e83addca17aa8e1d5e4d36a13b9cc8d9856f7` /
`f409abc67b630fbc6c1b30db7ba5e614ecb8925a2d5c7aa6b0e9d7746581067b`.
The first v24 GitHub clone failed with a TLS termination and remains
preserved; exact clean v25 was frozen from verified bundle SHA-256
`6c59f1d568017d8ee82e32d3132b595b73c3d469a2cb91976968d330cd789104`.

Recovery preflight/deployment/submission SHA-256:
`45e60ba0f68132b8cfa11ec036ed71789e83d718dc300df62f0cdf19f1375e8a` /
`a151cf03c67395771eb386c6fe48687e867b40df1d8f7a562be6d1df459125a0` /
`fca38a1cad01222ef8bda967116993742319bdc94b2d8e9582a783abe21c479f`.
`1205579` is test-only. Unique successor Job `1205580` is `PENDING
(Priority)` at fresh immutable v2 run root. The route remains
`experiment_running`; the complete matched protocol and all claim boundaries
are unchanged.

### Official matched-pair environment recovery and Job 1205584

Job `1205580` reached `FAILED 1:0` after 26 seconds. Offline source
validation and the candidate/audit focused suites had passed, but importing
the official `train.py` failed before any optimizer step with
`ModuleNotFoundError: tensorboard`. The preserved failure signature is
`official_declared_tensorboard_dependency_missing_v1`; failure receipt
SHA-256 is
`a959ef415f383d5368edf806b1166cca9cd25e91e49ea4398853775059e35385`.
Runtime/stdout/stderr SHA-256 values are
`24a2afdb56d776084b796aeb221bb9579b2985a6f45a6eec56ec3bd8d291ced9` /
`24a2afdb56d776084b796aeb221bb9579b2985a6f45a6eec56ec3bd8d291ced9` /
`fef8ce4b812cf04882328f4f12a5ddcac8c61077a1f8107c19b63e142808d74b`.
Candidate/audit test-log SHA-256 values are
`ca6eefaf05d226c6cbaa36b31e2c86a85abc36d0e7f5702edccd75d9e987fc3e` /
`8afafe9ad1b44247f833653a0d44ed1c6c99c905c5f43aa417a182803eb96c89`;
the empty pre-training dense-log SHA-256 is
`0ec2347eb0afdce06e5dec81d9afc9086f8a64f7b3bea59bb1d9b68789832fc2`.
This is an engineering dependency failure and contains no model result.

The official ActionFormer `INSTALL.md` declares TensorBoard. Recovery uses a
dedicated venv at
`/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_tensorboard_2_20_0_20260730_v1`
with TensorBoard `2.20.0`, while retaining Python `3.10.20`, PyTorch `2.0.1`,
CUDA `11.8` and NumPy `1.23.5`. Its environment receipt SHA-256 is
`acc5909360970cfad1f390a4f5ab046a3876ac9378448b2f94da26ffb312ece2`;
the import/SummaryWriter probe also verified that construction did not mutate
the RNG state.

Audit commit/tree
`a3d987961c0e6ac0166194cfc30ca0d375765ef1` /
`51c53773d266e614d6c1054a1e6127fe73c69f38` binds this exact environment
receipt and fails closed on version drift. Exact clean v26 was frozen from
bundle SHA-256
`e8812a84489bb55aea419b1b637778574539a44b0c7399b18a04d346430ce419`;
remote focused tests pass `19/19` with log SHA-256
`f18a52300731975c81d0fffa1cd4c8e5787ccc83b07abba212d4d2a1f6fcbb7c`.

Fresh full-content preflight/deployment/submission SHA-256 values are
`9ff27367e10717b012d0f06a85b980f54c9b91a6fe45be9e8f87c00cac90d47b` /
`00736c6b07fff77e0a6ca92ad24744eab0e2c089a22b350f9f2537054891b4f4` /
`f4512010b2d675611f97e61a929ee4edda421b7f29506969d49028b3a7ac041a`.
`1205583` is test-only. Unique formal successor Job `1205584` was observed
`RUNNING` on g0024 under immutable v3 root. The environment/source probes,
candidate `14/14` tests and audit `4/4` tests passed; the dense arm had just
started and no metric existed. Status remains `experiment_running` and
`paper_main_table_eligible=false`. The model, K384 intervention, seed,
official data, 5+30 schedule, terminal epoch-35 EMA and evaluator are
unchanged.

Before Job `1205584` emitted any metric, five independent reviews froze the
paper-stage protocol in
`experiments/actionformer-sparsehead-official-main-table-prereg-20260729.md`.
Status is `designed`, not deployed. The paired seed set is
`1234567891/1423812477/737690612/1788897292/1322022747`, canonical SHA-256
`a4038a752aa46b97e5854c20574d65ece078bad6124e4778cc4269e75747c7c6`.
S0 unlocks the five-seed study only when the complete matched result has
`Delta Avg>=-1.00 pp` and both `Delta mAP@0.6/@0.7>=-1.50 pp`.

The final accuracy-preserving claim requires five complete paired seeds,
95%-CI lower bounds of at least `-0.20 pp` for Avg and `-0.50 pp` for each of
mAP@0.6/0.7, plus detector-pipeline median speedup `>=1.05x` with CI lower
bound `>1.00x` and no duration-stratum regression. The predeclared 2x2
train-support x evaluation-query cross distinguishes execution/representation
from selected-loss optimization. Main-study runtime must additionally rehash
all 413 feature IDs/content/shape/dtype and emit effective-config/CLI/split
receipts; the current count-only runtime check is adequate for screening but
not the paper row.

Job `1205584` is terminal `FAILED 1:0` after `00:09:48`. Dense training
completed all 35 epochs and wrote terminal checkpoint SHA-256
`ea3c16fcf17fd6fb8cec57829804e96736a8ab231b07d820e5939fd5db3cba00`,
but the first save-only EMA evaluation failed before a metric; sparse training
never started. The checkpoint is preserved `diagnostic_only` and may not seed
or resume the successor.

Failure signature is
`official_actionformer_softnms_extension_abi_shadowed_by_opentad_v9arg_v1`.
The clean candidate contained no in-place NMS extension, so absolute import
`nms_1d_cpu` resolved the unrelated OpenTAD site-packages module
(SHA-256 `4ccea1d7bae60a3edb735280c564928f18e89bd01e160a1c9fa200625a660450`)
whose nine-argument Soft-NMS ABI conflicts with the pinned official
seven-argument caller. Failure-analysis receipt SHA-256 is
`99f83a03715fa935a422451f9fe842aeaae867546d37c9af39cda8869958f852`;
save-only eval/runtime log SHA-256 values are
`496468bf5c327ae0a31a3a581cc086fd7cfb69dd5d2b249b088acc6e8aee7338` /
`b3f8cca479ad22a674a433badcabb9d928b012af7b60221ec11f8e54e5bf6cc5`.
This is an engineering ABI-shadowing failure with no model result.

## 2026-07-30 official NMS provenance recovery and unique S0 successor

Status remains `experiment_running`. Audit commit/tree
`71f955a7301f07875a35e0be366241e548e5c775` /
`d328093644e040741e16dbdd8bc93b6b0d608a10` repairs only the Python
module provenance for the official seven-argument ActionFormer Soft-NMS ABI.
It does not add OpenTAD-only `t1/t2` arguments or change model, config, data,
seed, schedule, checkpoint rule, NMS parameters or evaluator. Exact clean
audit v27 was frozen from bundle SHA-256
`a9ee267333c9371d087e806fe61cef19c14122b18fee1a4e6c75fa4c58846ad6`.

The isolated official runtime is
`/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_official_runtime_20260730_v2`.
Its receipt SHA-256 is
`13d57c1161905f059204f7101f26029503a03da7f5eb44b81c418a0b97999f24`;
the pinned extension SHA-256 is
`b67e0e41f9f55cd69e8b90cfc75a1947214365857d851a510047838ad49ed98d`.
The launcher asserts the resolved module path/hash and executes a real
seven-argument call before training. Remote candidate/audit focused tests
pass `14/14` and `5/5`, with log SHA-256 values
`f0ae7ae24f8439ba95aed967799db6d50acc8d50f401ae3d2a60480aa6693936` /
`42aa27fe2cd2eed1543ebbf0ad635ab3a3941254ddff5778d3f2f599f3e8a16d`.

Official comparability was rechecked before submission. Candidate and pinned
official `configs/thumos_i3d.yaml` are byte-identical at SHA-256
`c0ac0df560cd564941b56cd9391ad0bd5cea386d2e4b6cf9fc8ffcab821955cd`.
The apparent `validation`-train/`test`-eval split is the pinned upstream
THUMOS protocol, not a drift from generic defaults. A live rehash of all 413
I3D files reproduced the sealed per-ID/content/shape/dtype feature manifest
SHA-256
`cda269dace393b9af1f6fcb87a9a531beed69e3c71279ba3ca2cee76e198d59a`;
the live-rehash receipt SHA-256 is
`73a2f714c100f541306d7d7f9c32e36481574d2ac6c5e78925ee4ee1dcca96b3`.

Full-content preflight/deployment/submission SHA-256 values are
`d9e1f897de51e46aac52cb450f72daa8bc19a64bf999b01112013489038d4a55` /
`b8d4079c9ddc8faa7a0a575dbe63f700c2448409df5dbccf972101cc0e4a282b` /
`2a31a1d01056f39159d17d99fb9047f5bd6946b68475c1eae31008659df07a08`.
`1205593` is only `sbatch --test-only`. Unique formal Job `1205594` is
`PENDING (Priority)` under fresh immutable v4 root
`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_matched_pair_k384_seed1234567891_20260730_v4`.
Both arms must retrain from scratch; no Job `1205584` checkpoint may be used.
Even a successful result is official-comparable single-seed S0 screening,
not a paper main-table row.

Job `1205594` then reached `FAILED 1:0` after four seconds at
`python_environment`, before tests or training. Failure signature is the
known recurring `official_environment_probe_nms_import_order_v1`: the new
launcher imported `nms_1d_cpu` before `torch`, so `libc10.so` was not yet
loaded. The deeper regression gap is that the previous focused test bound the
extension path/hash and seven-argument call but did not bind import order.
Failure/failure-analysis/runtime/stderr SHA-256 values are
`68d2ec8ddd1d2a69c1181d532325368975c95a306a2c7d368226905044ee321f` /
`06bbc29e5f57b3b9a12f421f5ddd814487bf01733d0f0e5bbcc4c0551c877a41` /
`5988ed65e4ebbd8dde6a334ffe7f2ae3c8825fbd5e61d1c6198388c0284443fb` /
`4c6d87aa6b85dbbe173a1eae119bac562aa43f55c6fa847b256c9c05d25c79e0`. No model iteration
or metric exists.

Recovery commit/tree
`98f5b875315b4a2b5c6829f5d74ccce68f478e47` /
`2e6b4bba6868c323d70c97140f7cbed044eb1a7b` imports PyTorch first and adds
an explicit ordering regression. Local `5/5` tests pass. Exact clean v28 is
frozen from bundle SHA-256
`713a1d839e8e8ea50f141df9dba1feb44dc43c91dffbd4dd85bf8910bbdf9e24`;
remote exact validation is in progress. Scientific conditions remain
unchanged.

Remote ordered-import/seven-argument probe passed with log SHA-256
`7d79381ed64b27059aa6f4204bbfce3f606fc1e81e0a7962e4e1d1c7413a0488`;
candidate/audit tests again pass `14+5`, with SHA-256 values
`1c2508394c210adca15c54acded1d470a1328253b4f9377d5255614e36dd40f4` /
`d14f97df2e52d4704acddb6a2f896d90d868e9d77ccc27e24b2ed3ccb7e349cc`.
Fresh preflight/deployment/submission SHA-256 values are
`19230f06e0eda57c34607db250dba9ebc1f0d6365e5ab33c339dffe0468ddd86` /
`250068a1de36c00fabe37596e302dc9e3fd22249be09b267fc4e9762e6f4ce46` /
`0549ff04a30bb4efea176a484a6f51d652b8bdd023227564b0fc2fdfe492cabf`.
`1205598` is test-only. Unique formal Job `1205599` is
`PENDING (Priority)` at fresh v5 root. Status returns to
`experiment_running`; all official-comparability and S0-only claim boundaries
remain unchanged.

Job `1205599` subsequently entered `RUNNING` on g0030. Environment/source
gates and candidate/audit focused tests passed inside the allocation; dense
training reached epoch 24 with finite loss. There is no arm completion or
metric yet.

The dense arm is now a validated `tested` component. Its ARM/independent
attestation/checkpoint SHA-256 values are
`a15b0526ef9a75a0fe32c0798b609c738781ab5c063c53df165ace6cbcdf138a` /
`59a9d037faf0418e226f184b87c66d484c7a64b81a911692ba65d44c1cc195d7` /
`48abfd3480159e367bc710df158b53186494d05fdf5e9c37cfa751873a0e8f4d`.
Independent recomputation over the exact 212-video set and 42,400 predictions
gives Avg-mAP `0.6658301251307708` and mAP@0.3–0.7
`0.8190849486121916/0.7795203466370499/0.7128549836803181/0.5825550463357125/0.43513530038858167`.
It is a newly trained epoch-35 EMA same-commit control, not the released
`66.83` anchor. Sparse training has started; no matched delta exists yet.

## 2026-07-30 official-comparable S0 terminal negative result

Job `1205599` completed `0:0` in `00:19:21`. Pair completion SHA-256 is
`545e420aa1d437aedeffd15cb30390ceb0cfe4d6565d7eb35c53a8bf17ac76fd`;
its validation issues are empty and all same-commit/seed/data/schedule/EMA/
evaluator/environment comparability contracts pass. Sparse ARM, independent
metric and checkpoint attestation SHA-256 values are
`fc682cfb01b9ed6639f821938922051edc2afa55490f504170eb7e3a6fd49037` /
`499b0e7f34b854ca6c9915a95d0c522106dda859079a50a7c4f28bbd6ede65ba` /
`e010f72ef19e84c4b28bf8f7b6c0118eaa8d4bf518e289ecea3d69adb4d311ec`.
The independent evaluator covers exactly 212 test videos and 42,400
predictions.

Dense versus sparse Avg-mAP is `66.583013` versus `43.919699`; mAP@0.3--0.7
is respectively
`81.908495/77.952035/71.285498/58.255505/43.513530` and
`64.925248/56.642845/45.952641/32.783177/19.294586`.
Sparse-minus-dense deltas are
`-22.663313/-16.983246/-21.309190/-25.332858/-25.472328/-24.218944 pp`
for Avg/@0.3/@0.4/@0.5/@0.6/@0.7.

This legal model result catastrophically fails the frozen S0 continuation
bounds. The current K384 + `selected_native_grid_queries` intervention is
`tested` and its S0 rejection is `empirically_supported`; five-seed and cost
deployment are stopped. `paper_main_table_eligible=false` remains binding.
This does not reject all SparseHead or PhysTime variants. The frozen Pro-level
analysis ranks combined query-coverage and selected-loss supervision damage
above calibration/NMS and implementation defects. The next authorized
experiment is the preregistered no-retraining 2x2 cross-evaluation plus raw
prediction/assignment diagnostics.

## 2026-07-30 official S0 negative-attribution closure

The authorized no-retraining attribution is complete. Slurm Job `1205701`
completed `0:0`; attribution/negative-diagnostics/suite SHA-256 values are
`d0bffe87cfb582b1b0649da3833e9fe0147db5a0a78500b6700fb78019323afb` /
`a6b7fa0c4a41aac75ae2fb4cb4fcfbe68cf48bc7d2c813b37485b35998838791` /
`e71721cb07334f1b6abb09347a7b609e51d6da1ed4be864c190ed60433a197d6`.
The exact frozen-checkpoint 2x2 Avg-mAP matrix is:

| training support | dense eval | K384 eval |
|---|---:|---:|
| full native grid | 66.583013 | 45.784332 |
| selected native grid | 64.537343 | 43.919699 |

The K384 execution main effect is `-20.7082 pp`; selected-loss training is
`-1.9552 pp`; interaction is `+0.1810 pp`. Therefore hard query/proposal
removal, not catastrophic checkpoint optimization, is the dominant observed
factor. Post-NMS class-aware/class-agnostic recall@0.7 falls
`76.50/80.85% -> 42.41/44.55%`, and fixed-topK recall gaps widen with K.
Calibration compression is secondary and cannot restore absent segments.

Assignment/support audit commit/tree
`465b2bc284d5c3b62ec9e21023052b5eabddf260` /
`da1e515398017345deb4c39d98751ade0a8aa8db` was tested by terminal Slurm Job
`1205799` (`COMPLETED 0:0`). Suite/producer SHA-256 values are
`475b61ddad4b0b56a86b2e2616ef2584b252c3169b4ad1268223f21d6e118567` /
`ca7e97a4124e49eb2ac30e949bcd50d4407998e8518eb72c8c6c8c8bb3f86e8b`.
Across exactly 64 deterministic official `validation` training windows,
K384 retains `461/2721 = 16.9423%` dense positives, leaves `395/804` GT
without a candidate and `427/804` without assignment. It uses no test GT,
loss, backward, optimizer, training or model selection and is diagnostic-only.

No contradiction exists with the earlier decode-cross evidence: physical-time
decoding improved frozen predictions within its own conditions but cannot
recover queries removed before proposal generation. The hard K384 ActionFormer
formulation is now `empirically_supported` as rejected. The route may continue
only through the separately `designed` DCSR formulation, which keeps a dense
cheap support/proposal scaffold and sparsifies only expensive residual
refinement. DCSR has no metric yet.

Official-paper boundary: Job `1205599` is a valid official-comparable negative
appendix result, not a released-number reproduction or positive main-table
row. Jobs `1205701` and `1205799` are mechanism diagnostics only. No current
speed, robustness or general SparseHead claim is allowed.

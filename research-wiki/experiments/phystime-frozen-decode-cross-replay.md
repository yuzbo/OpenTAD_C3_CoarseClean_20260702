---
type: experiment
node_id: exp:phystime-frozen-decode-cross-replay
title: "PhysTime frozen dual-axis decode cross replay"
idea: idea:phystime-tad-2
status: implemented
verdict: r00_gate_passed_but_formal_replay_validator_failed
confidence: source_dtype_gate_passed; formal_suite_not_completed
metrics: "NA; only two selected-axis replay producers completed, then their final validator raised NameError; physical jobs and suite were not submitted after Slurm submit-limit rejection."
provenance: "full60 0dc5851/bddc9b9; P0 c2cfcfa/0b78dd4; source-dtype runtime 6230082/f7366c6; run phystime_decode_cross_6230082_r00gate_20260723_014500_0800"
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

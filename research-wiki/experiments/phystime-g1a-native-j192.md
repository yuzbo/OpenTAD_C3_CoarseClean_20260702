# 2026-07-13 d1747d6 queued gate status

Status: `queued_for_gate`, not `experiment_running`.

- Independent Max verdict: `GREEN_FOR_CLEAN_SNAPSHOT_AND_REAL_GATE`, P0/P1 none.
- Clean snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1a_d1747d6_20260713_gate`.
- Commit/tree: `d1747d6657e185495b4db9eb491fd135d4b90360` / `2651bd30eda5b0e0960518da4060ccfc628b7a58`.
- Run root: `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_d1747d6_gatepilot_20260713_192727_+0800`.
- Jobs: `1161476` real gate, `1161477` selected-axis pilot afterok, `1161478` physical-metric pilot afterok.
- Submission check: gate was `PENDING (Priority)` and pilots were `PENDING (Dependency)`. No gate artifact and no mAP yet.
- Follow-up: gate `1161476` failed before training/pilot because the gate still conflated K384 decoded slots with 384 valid observations per real production training window. This is a gate-contract bug, not a method result. Fix status: implemented and remote focused tests passed (`30 passed` gate contract, `243 passed` PhysTime/C3 physical-grid), awaiting renewed Max green before requeue.
- Final requeue after renewed Max green: commit `56c7e98e54ba83eb32b84dbdbeb74c3b5698eca2`, clean snapshot `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1a_56c7e98_20260713_gate`, run root `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_56c7e98_gatepilot_20260713_194728_+0800`. Jobs: `1161486` gate, `1161487` selected-axis afterok, `1161488` physical-metric afterok. First check: gate `PENDING (Priority)`, pilots `PENDING (Dependency)`. No gate pass or mAP yet.

---
type: experiment
node_id: exp:phystime-g1a-native-j192
title: "PhysTime G1a native-J192 matched temporal-metric control"
idea: idea:phystime-tad-2
status: tested
verdict: pending_independent_review_then_fixed_snapshot_real_gate_and_pilot
confidence: pending
metrics: "No mAP yet. Any later raw values must be recorded only in docs/evaluation/results.md."
provenance: "docs/superpowers/specs/2026-07-13-phystime-p0-rebuild-design.md"
added: 2026-07-13T00:00:00+08:00
---

# PhysTime G1a Native J192

## 2026-07-13 独立部署审查状态

当前仍为 `tested`，不是 `experiment_running`。第二轮 Max 审查的 4 个 P1 已修复：严格 assignment 数量关系、固定 optimizer 参数集合与逐步 state 覆盖、真实 DataLoader `drop_last`、pilot 对 Git/config/data/contract/G0/checkpoint 的独立重算。远端 gate/artifact tests 为 `65 passed`，PhysTime/shared physical-grid tests 为 `240 passed`。第三轮零 P0/P1 是 clean snapshot 与 real gate 的必要前置条件；尚无新 pilot 或 mAP。

## Question

在相同 raw RGB K384、native VideoMAE J192、ActionFormer 基础网格 Q0=192、六层总候选 QΣ=378、参数、初始化、优化器、assignment 与后处理下，只把均匀 rank-derived 秒轴替换为原生 tubelet 物理秒轴，是否产生可解释差异？

## Implemented Contract

- K384 输入槽、J192 原生 token、Q0 与 QΣ 分开审计；
- 不做 J192→Q0=384 feature/query lift；
- 完整记录 observed 与 padding-repeat patch 输入槽，以及 chunk 内 padding 可能影响有效 token 的风险；
- 两臂 GT、候选、回归、decode、NMS 与评价均在绝对视频秒坐标；
- 两臂只允许 `uniform_rank_seconds` 与 `physical_time_seconds` 的坐标张量不同；
- static contract、G0、real THUMOS CUDA gate 与 pilot artifact 逐级绑定 commit/tree/config/checkpoint/dataset/hash；
- 三步 gate 必须在两臂上使用相同真实样本、相同像素张量、相同初始化与优化器，并执行正式滑窗 NMS/evaluator；
- pilot 固定为 6 epoch，只有真实预测、有限 IoU-wise mAP 与 `epoch_5.pth` 全部通过原子验收后才生成完成记录。

## 2026-07-13 诊断修复

- 原始 AdaTAD 的特征插值本身并不非法；G1a 暂时取消 lift，是为了把 `K=384` 原始观测、`J=192` tubelet token 和 `QΣ=378` 候选严格分开。若 G1a 存活，G1b 可在两臂同时恢复同一个中性 `J192 -> Q0=384` lift，但不得把插值点称为新增观测。
- 扩展回归发现并修复 `AnchorFreeHead` 中的真实候选 mask 错误：`selected_center` 原先是 `point[:, 0]` 的视图，写入物理中心后被同步改写，导致物理秒值错误地与 selected count 比较并删掉合法候选；现已显式 clone，mask 在 rank 域计算，回归在秒域计算。
- 修复已删除变量 `dense_valid_len` 的残留引用；debug 元数据现在明确记录 `domain_end - domain_start`。
- canonical FPS、end-exclusive window domain、上游采样 provenance、结构性 padding lineage 与 per-sample effective query count 均已 fail-closed。
- VideoMAE 严格尾部隔离已深入 patch/tubelet mask、每层 attention 的 K/V、每个 residual/MLP、每个 TIA 卷积与 final norm；反事实 padding 像素和无效输入梯度测试通过，同时 all-valid 路径与旧实现逐值一致。
- 真实 gate 的尾部样本与 evaluator 已统一改用 `dataset.test`，不再用 train/validation 样本冒充 test 推理闭环；gate 还要求 train/test 全量 MP4 的 decoder FPS、帧数和 annotation duration 使用同一 fail-closed 合同。
- 数据集指纹已从文件名/大小升级为逐文件完整 SHA256 与 Merkle root；pilot 验收会真实反序列化 checkpoint，检查 epoch/state/optimizer/scheduler/有限性，并从 manifest 配置重新运行 evaluator 比对 metrics JSON。
- 对 THUMOS14 全部 411 个 MP4 的预部署时间基准审计显示：annotation 与 decoder 的最大相对 FPS 偏差约 1.12%，最大帧数偏差为 0；G1a 因而把 FPS/时长容差固定为 1.25%，帧数相对容差固定为 0.01%，正式 gate 会逐视频复核而不是信任该离线结论。

## Evidence Level

当前为 `tested`：本地编译、diff 与 launcher shell 检查通过；本机 PyTorch 受既知 `c10.dll` 初始化故障阻断，模型测试转移到远端 Linux/Torch 临时树执行。新旧 PhysTime/C3、padding isolation、timebase、gate、artifact 与部署回归合计 `142 passed`。首轮独立最高强度逐行审查发现 4 个 P1 与 3 个 P2，修复后正在由同一代理复审；代码尚未形成最终 clean commit/fixed snapshot，真实 THUMOS CUDA gate、六 epoch pilot 与 mAP 尚未完成，因此不能声称 `experiment_running`、`empirically_supported` 或有效。

## Formal Deployment Attempts

- 首个 clean snapshot commit `8e2b832` 的 gate `1161304` 正确 fail-closed，依赖 pilot `1161305/1161306` 未启动并已取消。失败发生在模型前：test 目录有 213 个 MP4，但 annotation 与正式 OpenTAD test `data_list` 只消费 211 个；原 gate 错把两个未引用文件也要求具有 annotation timebase。
- 修复后 timebase 范围由正式 `build_dataset(...).data_list` 决定：实际审计 200 个 train 与 211 个 test 视频；`video_test_0000270`、`video_test_0001292` 作为未引用 inventory 显式登记并继续受完整目录 Merkle 指纹约束。任何被 dataset 消费但目录缺失的视频仍立即失败。
- 修复后的真实目录范围 precheck 与远端 `116 passed` 已完成；新的 clean snapshot real gate 与 pilots 待重新提交。当前状态仍为 `tested`，不能把失败 gate 或 pending dependency 写成实验结果。
- 范围修复 commit `e598bd7` 的第二次 gate `1161353` 已越过 timebase 范围，但在首个模型状态摘要处暴露远端 PyTorch 对 0 维 LongTensor 不允许跨元素大小 `view(torch.uint8)` 的兼容性错误；依赖 pilot `1161354/1161355` 未启动并取消。摘要现改为先展平再按字节 view，并加入 scalar integer buffer 回归测试；这仍是工程 gate 失败，不是方法结果。
- 标量摘要修复 commit `d193417` 的第三次 gate `1161378` 越过数据、checkpoint、evaluator 与模型构建，在 selected-axis 的第一个真实训练样本上因 `regression_gradient=0` fail-closed；依赖 pilot `1161379/1161380` 未启动并取消。该现象与 ActionFormer 末端 ReLU dead zone 一致，但旧失败 artifact 没有记录正 assignment、正 `reg_loss` 或 pre-ReLU 激活，因此不能把机制可能性写成已证明根因。v3 gate 现使用正式 batch=2 DataLoader、warmup scheduler、EMA 和生产更新顺序；每步保存 assignment、pre-ReLU 激活、所有梯度、LR、clip、optimizer state，三步内要求回归参数至少一次非零，并用 trainable-only hash 与数值 delta 证明真实参数更新。独立 validator 会从逐步证据重算合同。该修改已通过远端 `142 passed`，仍须独立复审和新 clean gate 才能部署 pilot。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

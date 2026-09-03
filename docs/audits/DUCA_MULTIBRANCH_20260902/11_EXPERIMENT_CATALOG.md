# DUCA/ZoomToken 全部代码实验目录

最后更新时间（UTC）：`2026-09-03T08:56:37+00:00`

本表用完整中文描述实验目的；括号中的内部 ID 仅用于与 Slurm/manifest 对照。每一行都是独立代码身份，结果不能跨 SHA 转移。

## 当前实验与修正路线

| 实验名称（面向外部读者） | 本地目录 | GitHub 提交 | 部署状态 | 结果状态与最终结果 | 下一步 |
|---|---|---|---|---|---|
| H65-Pro 严格 60 轮全矩阵：物理时间坐标与高质量动作定位（`H65_PRO`） | `E:/DeskTop/TAD/_duca_audit_worktrees/h65_pro` | [`cfb7041d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/cfb7041d876f6e38e9ef6ce77cef7cee04b79659) | 已完成精确 SHA CUDA focused admission；P0 admission 失败，正式矩阵未提交 | 无最终结果：15 个 focused CUDA 测试通过；更深 P0 检查 14 通过、1 失败，暴露 x-only backbone 收到 masks 的签名错误 | 在独立修正 SHA 完成签名路由复验，再重新冻结 H65 SHA |
| DUCA 统一全矩阵：Taylor 归因、H65 保留机制与真实成本（`DUCA_UNIFIED`） | `E:/DeskTop/TAD/_duca_audit_worktrees/duca_unified` | [`89b9ea3e`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/89b9ea3e8e018b41034917ee14de7f409354a7e9) | 生成器 fail-closed；Taylor P0/P1、原始 H65 retention/transition、真实 cost 未实现，未提交训练 | 无最终结果：无合法 mAP、速度或成本结果；41 个 cell 保持关闭 | 完成三个真实机制后重新运行 generator、preflight 和 exact-head admission |
| DUCA 证据恢复：历史 H65 证据链与 8261 单种子数值复现（`EVIDENCE`） | `E:/DeskTop/TAD/_duca_audit_worktrees/evidence` | [`08d425a2`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/08d425a259fc468dde7c496e77b4c43e953d8d0c) | 精确 SHA CUDA focused admission 和 seed 8261 precheck 已通过；C0 parity 尚未完成，正式训练未提交 | 无最终结果：35 个 focused CUDA/证据测试通过；尚无 terminal EMA、官方评测或 mAP | 完成 indices、physical positions、features、logits、loss、decode、predictions 的 C0 精确 parity |
| DUCA CT-DP-BAMoD：CT-Tubelet 物理时间差归一化与 B-AMoD 稀疏层路由（`CT_DP_BAMOD`） | `E:/DeskTop/TAD/_duca_audit_worktrees/ct_dp_bamod` | [`2b7f8180`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/2b7f81808006c6cb09a4d21a7f6fdc8ed3f6babc) | 精确 SHA geometry focused admission 已通过；冻结 SHA 的 G0/G1 因子化与声明冲突，正式矩阵未提交 | 无最终结果：7 个 focused CUDA/几何测试通过；不能据此宣称 CT-DP 机制有效 | 采用独立修正分支完成 geometry、有限差分 gradient、batch/DDP 后重新冻结 SHA |
| ZoomToken BAFDR：48 分块全局低清、K16 局部高清路由与 D160 教师蒸馏（`BAFDR`） | `E:/DeskTop/TAD/_duca_audit_worktrees/bafdr` | [`fdeaeb98`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdeaeb98340bf7070201a02feb8093f50486aeaa) | 静态协议 admission 已通过；精确 SHA 五臂 screen 尚未通过，21-cell 矩阵关闭 | 无最终结果：11 个静态协议测试通过；缺少同种子 D160 epoch 59 EMA Teacher 和 selection-screen PASS | 提供并核验 terminal Teacher，再运行不依赖 held-out 的五臂 screen |
| ZoomToken ET-TRC：Transformer 内部 Anchor 全计算与非 Anchor 局部 Taylor/JVP 修正（`ET_TRC`） | `E:/DeskTop/TAD/_duca_audit_worktrees/et_trc` | [`59eab0c6`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/59eab0c6aaacf5039d2ae20969a6dd5772bcb80f) | 静态 launcher/pretrain 协议测试已通过；真实 checkpoint coverage、单卡加载和双 GPU DDP 尚未完成 | 无最终结果：10 个协议测试通过；无合法 OFF/ON terminal EMA 或评测结果 | 核验 VideoMAE checkpoint 覆盖，再执行真实 global-batch=2 双 GPU OFF/ON DDP 和 resume |
| H65-Pro 当前正式单种子矩阵：384 帧四相预算与物理时间定位（`H65_PRO_ACTIVE`） | `E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission` | [`e553a5a4`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/e553a5a4a1063a755900d3dfa4bf8909bf97d466) | 远端 exact-SHA admission 1267684 已通过；REF-D768、REF-U384、REF-MNV3FC384 正式训练 1267709/1267711/1267737 正在运行，依赖评估 1267710/1267712/1267738 等待 | 正式训练进行中，尚无终态性能：已修复短窗口 padding 后 K=383 与输入 K=384 不一致；本地 32 passed、远端 admission 成功，当前无 epoch-59 EMA 或官方 mAP | 持续核验成功更新、有限 loss、终态 EMA；训练成功后自动运行官方 THUMOS14 评估 |
| CT-DP 当前正式四臂：基础嵌入、CT-Tubelet、B-AMoD 及二者组合（`CT_DP_BAMOD_ACTIVE`） | `E:/DeskTop/TAD/OpenTAD_CTDP_FormalRepair_20260903` | [`c0fae67a`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/c0fae67a1236f2c47e6c2935d217659cd1f8fb9d) | 远端 admission 1267221 已通过；G0、G1、G2、G3 正式训练 1267229-1267232 正在运行 | 四臂正式训练进行中，尚无终态性能：远端 26 项测试与 CUDA 几何/梯度门禁通过；当前训练 loss 有限，尚无 epoch-59 EMA 或官方 mAP | 等待四臂终态 checkpoint，随后运行同一 evaluator 并做配对比较 |
| DUCA-Unified 当前 41 单元正交消融控制台（`DUCA_UNIFIED_ACTIVE`） | `E:/DeskTop/TAD/OpenTAD_DUCA_Unified_FormalGates_20260903` | [`793c4f9c`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/793c4f9cdf7dac4f224bc73012aff8bc93949f87) | 提交器 fail-closed，Taylor P0/P1、H65 retention/transition 与真实 cost 未落地前禁止提交相关单元 | BLOCKED_UNIMPLEMENTED，无正式性能：manifest/生成器/准入规则可验证，但不能把缺失机制的占位配置当实验 | 逐项实现并测试缺失机制后重新生成 41-cell manifest，再分阶段释放正式矩阵 |
| BAFDR 当前 seed 4407 正式流水线：D160 教师与五臂 K16 筛选（`BAFDR_ACTIVE`） | `E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_admission` | [`539287fa`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/539287fa8a035765afd7e79863ce77278bef83f2) | 真实 CUDA/focused 门禁 1267855 已通过；D160 seed 4407 教师 1267884 正在运行，成功后按分钟提交 G96、U16、LATE、NOKD、FULL | 教师正式训练进行中，学生五臂尚未提交，无终态性能：保留 1267818 的 /bin/sh wrap 失败；539287fa 已改为 bash -lc 并重跑门禁；efe69d2e checkpoint 由独立任务按旧 SHA 另行评测，不能归入本提交 | 完成同 SHA、同 seed 的 D160 epoch-59 EMA 教师后开放五臂 screen |
| ET-TRC 当前双臂：完整 Transformer 与局部 Taylor/JVP 近似（`ET_TRC_ACTIVE`） | `E:/DeskTop/TAD/zoomtoken_ettrc_correction_20260902` | [`74473c27`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/74473c2775caebf0da9d368ce8009d78e2942098) | 真实 VideoMAE 覆盖与 2-GPU/global-batch=2 admission 已通过；OFF 1267218 与 ON 1267219 正在运行 | 双臂正式训练进行中，尚无终态性能：当前训练 loss 有限；ET-TRC 保持 100% dense temporal states，只减少 Transformer 块内完整计算次数 | 等待 OFF/ON epoch-59 EMA 与官方评估，比较定位性能和执行算子计数 |
| BAFDR 历史五臂三种子终态 checkpoint 独立评测（`BAFDR_EFE69D2E_EVAL`） | `E:/DeskTop/TAD/NO_LOCAL_WORKTREE_FOR_EFE69D2E` | [`efe69d2e`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/efe69d2ea10accd01d0129dfe99cba4d1d5773cb) | 1266410-1266414 已产生 15 个 epoch-59 checkpoint；独立任务正在运行 prediction eval 1267920，postprocess 1267921 afterok 等待 | 历史 exact-SHA 评测进行中，尚无可报告终态指标：只按 efe69d2e 的真实训练/teacher/数据/evaluator receipts 报告，绝不迁移为当前 539287fa 结果 | 等待 792-window prediction seal、C_exec、官方 mAP opening 与 strict completeness receipt |
| Evidence-Recovery 当前 C0：轻量侦察器不确定性与最大空洞补漏（`EVIDENCE_ACTIVE`） | `E:/DeskTop/TAD/OpenTAD_Evidence_FormalRepair_20260903` | [`77c8d173`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/77c8d173c95aef153c04fd1355a0e75a63ff22c9) | 修复版 admission/C0 每分钟流水线 PID 2400024 正在等待 QOS 提交名额；队列降到限制后自动先跑 actual-ledger admission，再提交 C0 seed 8261 与 afterok 评估 | 等待重提，尚无终态性能：保留 1267747 的 219-vs-100 loader 合同失败和 1267857 的 legacy ledger policy 字段错配；77c8d173 已修复两者并把真实三份 ledger schema 纳入 admission | 等待每分钟提交器取得名额并通过 admission；C0 首个 100-update epoch 成功后再逐步释放其余七臂 |

## 监督器与动态状态

远端 N16R4 监督器：`/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_20260902`，每 60 秒轮询；本地 heartbeat 每 30 分钟刷新本表。当前远端监督器状态：`NOT_QUERIED`，dispatcher：`未知`。

## 明确排除的旧远端作业

这些作业可以继续作为诊断材料，但不属于当前冻结实验，不能写入最终结果：

| 作业号 | 远端目录 | source HEAD | 排除原因 |
|---|---|---|---|
| `1266325-1266330` | `/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_6ae16954` | `6ae16954` | 旧 BAFDR checkout，不属于当前冻结 SHA；1266328-1266330 stderr 已诊断为 LoadFrames.__init__ 不接受 window_size，修复已移植到 BAFDR_ADMISSION_FIX，旧作业不重标结果 |
| `1266185-1266186` | `/data/run01/sczc063/yuzibo/projects/zoomtoken_et_trc_correction_20260902_59eab0c6` | `be330c07` | 晚于冻结 ET-TRC SHA；1266185 在 S1 batch 17 出现 cls_loss/reg_loss/cost 非有限，1266186 随后取消，未产生合法 checkpoint |
| `1265704-1265705` | `/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902` | `679b7121` | 旧 CT-DP checkout；stderr 显示启动器引用不存在的 opentad_ct_dp_revised_20260902 路径，属于提交协议错误 |
| `1266218-1266219` | `/data/run01/sczc063/yuzibo/projects/zoomtoken_et_trc_correction_20260902_59eab0c6` | `be330c07` | 晚于冻结 ET-TRC SHA；Slurm COMPLETED 但仅有 log/config，没有 terminal checkpoint 或 receipt，不纳入当前结果 |
| `1266401-1266420` | `/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_5dba75c7` | `efe69d2e` | 不是当前 539287fa BAFDR 身份；1266401/1266402 的 LoadFrames 失败保留，1266410-1266414 的 15 个终态 checkpoint 由独立任务按 efe69d2e 身份评测，满足 receipt 后可单列历史结果，但不得迁移到当前提交 |
| `1266475,1266479,1266480,1267819,1267820,1267822` | `/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_5dba75c7` | `efe69d2e` | efe69d2e 的旧评测/cexec/summary 链；与当前 539287fa 分栏，后续评测由任务 01a0660c-d75e-7f92-8921-d902ce792561 独立负责，本监督任务不得再取消或重提 |
| `1267747,1267748` | `/data/run01/sczc063/yuzibo/projects/duca_evidence_formal_21d1d229` | `21d1d229` | Evidence C0 首次正式启动因 DataLoader 219 batches 与 100-update 合同实现冲突而失败；0d1abf6d 修复了更新暴露合同 |
| `1267857,1267858` | `/data/run01/sczc063/yuzibo/projects/duca_evidence_formal_0d1abf6d` | `0d1abf6d` | Evidence C0 第二次启动读取真实 legacy ledger 时发现 policy_source/config-hash 契约与行 schema 错配；77c8d173 已改为绑定 policy=c3_lowres_probe_delta_p_action 并在 admission 扫描三份 ledger |
| `1267818` | `/data/run01/sczc063/yuzibo/projects/zoomtoken_bafdr_formal_52c940f2` | `52c940f2` | BAFDR D160 教师提交脚本由 /bin/sh 执行 source/pipefail，启动即失败；539287fa 已改为 bash -lc 并重新运行 CUDA/focused 门禁 |
| `1265777-1265778` | `/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902` | `679b7121` | 旧 CT-DP checkout；G0/G1 已完成并留下 epoch_59.pth 与 Average-mAP 63.95% 日志，但无当前 exact SHA 或 audit-owned terminal receipt，mAP 不纳入结果 |
| `1265779-1265780` | `/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902` | `679b7121` | 旧 CT-DP checkout；G2/G3 仍在运行，不能迁移为当前冻结或修正路线结果 |
| `1265077_[0-2,3-7]` | `/data/run01/sczc063/yuzibo/projects/opentad_duca_evidence_recovery` | `647151fa` | dirty 且旧 Evidence checkout；失败/完成作业均不纳入当前结果，缺少 exact SHA 与终态 receipt |

结果规则：没有 exact SHA、clean-tree、terminal EMA、官方 evaluator 和合法 aggregation receipt，不得报告为最终科学结果。当前结果账本仍为 `NO_VALID_RESULTS`，不得从 admission 测试推导 mAP、speedup、bootstrap 或 cost。

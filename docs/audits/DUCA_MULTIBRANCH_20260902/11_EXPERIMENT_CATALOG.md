# DUCA/ZoomToken 全部代码实验目录

最后更新时间（UTC）：`2026-09-02T15:42:52+00:00`

本表用完整中文描述实验目的；括号中的内部 ID 仅用于与 Slurm/manifest 对照。每一行都是独立代码身份，结果不能跨 SHA 转移。

## 当前实验与修正路线

| 实验名称（面向外部读者） | 本地目录 | GitHub 提交 | 部署状态 | 结果状态与最终结果 | 下一步 |
|---|---|---|---|---|---|
| H65-Pro 严格 60 轮全矩阵：物理时间坐标与高质量动作定位（`H65_PRO`） | `E:/DeskTop/TAD/_duca_audit_worktrees/h65_pro` | [`cfb7041d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/cfb7041d876f6e38e9ef6ce77cef7cee04b79659) | 已完成精确 SHA CUDA focused admission；P0 admission 失败，正式矩阵未提交 | 无最终结果：15 个 focused CUDA 测试通过；更深 P0 检查 14 通过、1 失败，暴露 x-only backbone 收到 masks 的签名错误 | 在独立修正 SHA 完成签名路由复验，再重新冻结 H65 SHA |
| DUCA 统一全矩阵：Taylor 归因、H65 保留机制与真实成本（`DUCA_UNIFIED`） | `E:/DeskTop/TAD/_duca_audit_worktrees/duca_unified` | [`89b9ea3e`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/89b9ea3e8e018b41034917ee14de7f409354a7e9) | 生成器 fail-closed；Taylor P0/P1、原始 H65 retention/transition、真实 cost 未实现，未提交训练 | 无最终结果：无合法 mAP、速度或成本结果；41 个 cell 保持关闭 | 完成三个真实机制后重新运行 generator、preflight 和 exact-head admission |
| DUCA 证据恢复：历史 H65 证据链与 8261 单种子数值复现（`EVIDENCE`） | `E:/DeskTop/TAD/_duca_audit_worktrees/evidence` | [`08d425a2`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/08d425a259fc468dde7c496e77b4c43e953d8d0c) | 精确 SHA CUDA focused admission 和 seed 8261 precheck 已通过；C0 parity 尚未完成，正式训练未提交 | 无最终结果：35 个 focused CUDA/证据测试通过；尚无 terminal EMA、官方评测或 mAP | 完成 indices、physical positions、features、logits、loss、decode、predictions 的 C0 精确 parity |
| DUCA CT-DP-BAMoD：双阶段时空几何、CT-Conv 与 B-AMoD 机制（`CT_DP_BAMOD`） | `E:/DeskTop/TAD/_duca_audit_worktrees/ct_dp_bamod` | [`2b7f8180`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/2b7f81808006c6cb09a4d21a7f6fdc8ed3f6babc) | 精确 SHA geometry focused admission 已通过；冻结 SHA 的 G0/G1 因子化与声明冲突，正式矩阵未提交 | 无最终结果：7 个 focused CUDA/几何测试通过；不能据此宣称 CT-DP 机制有效 | 采用独立修正分支完成 geometry、有限差分 gradient、batch/DDP 后重新冻结 SHA |
| ZoomToken BAFDR：基于梯度的 16 帧动态帧率与五臂筛选（`BAFDR`） | `E:/DeskTop/TAD/_duca_audit_worktrees/bafdr` | [`fdeaeb98`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdeaeb98340bf7070201a02feb8093f50486aeaa) | 静态协议 admission 已通过；精确 SHA 五臂 screen 尚未通过，21-cell 矩阵关闭 | 无最终结果：11 个静态协议测试通过；缺少同种子 D160 epoch 59 EMA Teacher 和 selection-screen PASS | 提供并核验 terminal Teacher，再运行不依赖 held-out 的五臂 screen |
| ZoomToken ET-TRC：固定步长 Taylor 时序保留与双 GPU 对照训练（`ET_TRC`） | `E:/DeskTop/TAD/_duca_audit_worktrees/et_trc` | [`59eab0c6`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/59eab0c6aaacf5039d2ae20969a6dd5772bcb80f) | 静态 launcher/pretrain 协议测试已通过；真实 checkpoint coverage、单卡加载和双 GPU DDP 尚未完成 | 无最终结果：10 个协议测试通过；无合法 OFF/ON terminal EMA 或评测结果 | 核验 VideoMAE checkpoint 覆盖，再执行真实 global-batch=2 双 GPU OFF/ON DDP 和 resume |
| H65 backbone 参数签名修正实验（不替代冻结 H65）（`H65_ADMISSION_FIX`） | `E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission` | [`78cde6aa`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/78cde6aa5335b2e399e597ce9229d8657e6760a5) | 修正已提交并应用到远端验证 worktree；复验受 Slurm AssocMaxSubmitJobLimit 阻塞 | 修正尚未获得远端终态结果：本地 official optimizer coverage 1 passed、2 skipped；不能晋级冻结 H65 结果 | 资源释放后重跑 H65 P0 admission，并据此决定新的冻结 SHA |
| CT-DP G0/G1 因子化修正实验（不替代冻结 CT-DP）（`CTDP_ADMISSION_FIX`） | `E:/DeskTop/TAD/_duca_fix_worktrees/ctdp` | [`d62cab76`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d62cab763c8e0478e73c6c47a4c185db45164dda) | 恢复 G0/G1 正交机制定义并通过本地因子化测试；尚未完成远端 geometry/gradient/batch admission | 无最终结果：因子化测试 1 passed；不是 CT-DP 训练或性能结果 | 远端完成 geometry、gradient、batch/DDP 后重新冻结 CT-DP SHA |
| DUCA Unified fail-closed 准入修正实验（不替代冻结 Unified）（`UNIFIED_ADMISSION_GATES`） | `E:/DeskTop/TAD/_duca_fix_worktrees/unified` | [`98d559ee`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/98d559ee414504caaa480294ce4d066276cdebe6) | 已将 D1/F11/H0/G10/G11 和 cost 标为 BLOCKED_UNIMPLEMENTED，submitter fail-closed；未提交训练 | 无最终结果：admission gate test、generator check、Python 编译通过；机制仍未实现 | 实现 Taylor/H65 retention/cost 后再生成可提交的 41-cell manifest |

## 监督器与动态状态

远端 N16R4 监督器：`/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_20260902`，每 60 秒轮询；本地 heartbeat 每 30 分钟刷新本表。当前远端监督器状态：`ACTIVE`，dispatcher：`BLOCKED`。

## 明确排除的旧远端作业

这些作业可以继续作为诊断材料，但不属于当前冻结实验，不能写入最终结果：

| 作业号 | 远端目录 | source HEAD | 排除原因 |
|---|---|---|---|
| `1266325-1266330` | `/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_6ae16954` | `6ae16954` | 旧 BAFDR checkout，不属于当前冻结 SHA |
| `1266218-1266219` | `/data/run01/sczc063/yuzibo/projects/zoomtoken_et_trc_correction_20260902_59eab0c6` | `be330c07` | 晚于冻结 ET-TRC SHA，不纳入当前结果 |
| `1265777-1265780` | `/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902` | `679b7121` | 旧 CT-DP checkout，不纳入当前结果 |
| `1265077_[3-6]` | `/data/run01/sczc063/yuzibo/projects/opentad_duca_evidence_recovery` | `647151fa` | dirty 且旧 Evidence checkout，不纳入当前结果 |

结果规则：没有 exact SHA、clean-tree、terminal EMA、官方 evaluator 和合法 aggregation receipt，不得报告为最终科学结果。当前结果账本仍为 `NO_VALID_RESULTS`，不得从 admission 测试推导 mAP、speedup、bootstrap 或 cost。

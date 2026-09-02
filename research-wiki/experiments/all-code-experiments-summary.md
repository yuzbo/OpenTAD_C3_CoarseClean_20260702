# ZoomToken / DUCA 代码实验总表

> 更新时间：2026-09-02（本地汇总快照）<br>
> 远程仓库：[OpenTAD_C3_CoarseClean_20260702](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702)<br>
> 当前工作分支：`codex/zoomtoken-ba-fdr-k16-fullmatrix-v001`；本地 HEAD：`5dba75c799852b65bca07d020215bf6702303252`

## 阅读规则

- **正式结果**：有完整数据集、冻结训练/评测协议和终态 receipt/metrics；才可用于论文性能结论。
- **工程证据**：预检、单步门禁、单元测试、成本回放或短跑；只能证明实现/管线性质，不能替代正式 mAP。
- **运行中/排队**：已提交但尚无最终指标；不得提前填写性能结论。
- **失败/停止**：有明确失败门槛、协议阻塞或负结果；保留原始结果，不改写成成功。
- 表格按“实验族”归并，但每一行列出该族的主要子实验、作业号或结果，避免只留下内部缩写。

## 全部实验清单

| 面向外部读者的实验名称 | 本地实现、配置和证据位置 | GitHub 链接 | 部署/结果状态 | 最终结果或当前结论 |
|---|---|---|---|---|
| **官方 AdaTAD 密集时序基线**：完整 160 点输入、既有 VideoMAE 主干、Adapter、检测头和 NMS | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\configs\adatad\thumos\` 中的官方/最小 base 配置；`E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\opentad\` | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 已有官方复现和匹配源基线终态 | 官方复现作业 1245842：`68.73 / 61.58 / 47.24`（Avg-mAP / mAP@0.6 / mAP@0.7）；匹配源作业 1245907：`64.73 / 56.14 / 43.26`。这是准确率参照，不是低成本方法。 |
| **低分辨率动作性探针**：用低成本全局视图预测动作性，再分析哪些时间片值得保留 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\tools\bata\train_lowres_action_probe.py`、`c3_coarse_classifier_model_matrix.py`；相关 `configs/adatad/thumos/*c3*` | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 历史诊断/归因实验，已停止 | 证明动作性信号可用于归因，但不是当前论文主方法；不能用短跑指标宣称完整 TAD 性能。 |
| **PAction 学习式采样策略**：从动作性和边界线索学习采样决策 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\tools\bata\train_paction_acquisition_policy.py`、`apply_paction_acquisition_policy.py`、PAction 配置 | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 历史归因路线，已停止 | 用于解释动态采样收益/失败原因；没有当前冻结协议下的最终官方测试证据。 |
| **GAS-VT 与 Lattice 几何采样对照**：比较规则网格、几何约束和动作性选择 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\tools\bata\` 中 GAS-VT/Lattice 工具、配置及对应研究档案 | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 历史基线/对照，已停止 | 属于方法归因工具，不是现行主线；未形成可替代 BPNS-R1 的端到端效率证据。 |
| **DUCA 固定预算套件**：在固定候选数量下比较多时钟/多分支特征恢复策略（固定 384/256/128、MUST 384/320/256） | 当前树中的历史档案：`E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\experiments\duca-7e3-budget-suite.md`；DUCA 模型源文件在当前 HEAD 不存在 | [DUCA 历史提交记录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/7e3a5081f58958fc924accf43088b24e2bf3093a) | 作业 1152688–1152693 已结束；一个 MUST256 为 `NODE_FAIL`；整族已停止 | 这是历史预算扫描。没有完整、同协议的最终 mAP/端到端成本证据，不能作为当前主方法。 |
| **DUCA 固定 384 全训练**：把固定高预算恢复作为 DUCA 的强对照 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\experiments\duca-70aa-fixed384.md`；远端 worktree 记录见该档案 | [DUCA 提交](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/70aa069b895322c2307ffbb13dfdef9fac0d1305) | 作业 1154971 曾运行至约 41/60 epoch，非完整终态；路线已停止 | 中途记录 Avg-mAP 约 `53.84`，mAP@0.3–0.7 为 `73.19 / 65.89 / 56.16 / 44.33 / 29.65`；该数值不是正式最终结果。 |
| **DUCA 成本与可运行性审计**：检查多时钟插件、成本计数和随机初始化管线 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\experiments\duca-a5e-cost.md`；当前树无 `opentad/models/backbones/duca_*` | [DUCA 提交](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/a5e1774b9941312569ca645341da1abad339db61) | 40 个本地测试、141 个远程测试和 21 项成本预检完成；无正式训练 checkpoint | 只能证明工程接口/成本审计，不提供论文性能或能耗结论；当前树不应标记为“已实现 DUCA”。 |
| **ChronoTransport 动态特征刷新**：在 384 点内部 tubelet 网格上复用/刷新时序特征，缺少成本与风险证据时回退密集计算 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\opentad\models\chronotransport\runtime.py`；`configs/adatad/thumos/c3_chronotransport_*stage_{a,b,c}.py`；`tools/bata/run_chronotransport_stage_b_formal.py`、`train_chronotransport_stage_b.py` | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | Stage-B 正式作业已结束；Stage-C 未进入论文主结果 | P3 总门禁失败：风险-回报、尺度匹配和特征误差累积不满足要求。工程上可运行，但科学上回退 dense，路线停止。 |
| **GeoRoute/SCNR 动态路由与残差校准系列**：测试几何中心、残差居中、角色校准、混合因果和成本配对 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\opentad\models\backbones\georoute_wrapper.py`；`E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\experiments\georoute-*.md`、`scnr-*.md` | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 多个开发/训练/成本子实验已完成，整族停止 | 最清晰的匹配训练结果：Control `10.52 / 8.90 / 6.98`，Centered `12.57 / 11.04 / 8.14`，增益 `+2.05 / +2.14 / +1.16` pp；但来源/成本改变，不能作为无偏主结论。 |
| **原生坐标裁剪 S1 垂直切片**：从 CPU uint8 原图按原生坐标裁出局部视图，与全局视图共用一个 VideoMAE | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\opentad\models\backbones\native_crop_wrapper.py`；`E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\experiments\native-crop-s1-vertical-slice.md`；配置/工具在 `tools/bata/` 和 `configs/adatad/thumos/` | [S1 提交](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/0bf59be877eeb6879166893641c12bc4e60a2b53) | 作业 1174671 门禁完成；开发证据 | 通过结构与单步门禁，证明 `NativeCropSourceViews` 和共享 VideoMAE 通路可运行；没有官方 test mAP 或正式效率结果。 |
| **Continuous-RoI S2-v3 空间连续变焦矩阵**：比较密集 160、全局 96、全量局部 128 和连续局部候选 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\opentad\models\backbones\native_crop_wrapper.py`；`configs/adatad/thumos/continuous_roi_s2_v3_*.py`；`tools/bata/continuous_roi_s2_v3_full200_compute.py`、`continuous_roi_s2_v3_full200_compute_train.py`；验证器 `tools/bata/validate_continuous_roi_s2_implementation.py` | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 历史 9-cell 训练作业 1177668–1177676 已结束；当前 v3 协议结果保持审查中 | 训练 receipt 不等于完整 mAP/cost 证据；当前验证器仍指向旧 base 配置，和 v3 U128-A0 期望不一致，属于协议/实现对齐 blocker。 |
| **BPNS-R1 准确率归因矩阵**：在完整主干前保留连续无孔洞空间支持，比较 K100、随机/全局、R1/R2/R3/R4 等候选 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\configs\adatad\thumos\` 中 BPNS/R1 配置；`E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\tools\bata\` 中 probe、ledger、validator；研究档案 `bpns-r1-*.md` | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 多个完整训练/评测作业已完成 | 代表性结果：K100 `68.51 / 61.19 / 46.27`；严格连续 8×8、K64 的 R1 `69.07 / 61.14 / 46.57`；R2 `66.56 / 59.06 / 45.17`；R3 `67.88 / 60.32 / 46.41`；R4 `68.02 / 60.32 / 46.26`。这些是准确率证据，不自动证明真实端到端省时。 |
| **BPNS-R1 完整端到端成本闭环**：在相同评测人口上测真实延迟、能耗和内存，而非只看 FLOPs | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\experiments\bpns-r1-v004-fullstack-cost.md`；成本工具和 receipt 在 `tools/bata/` | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 作业 1260095 已有完整终态；冻结停止 | p50 延迟比 `0.984929`（约 1.51% 改善，未达到 5% 门槛）；能耗比 `0.935000`（约 6.50% 改善）；峰值/工作集内存比 `0.75130 / 0.68966`。论文效率 headline 按协议停止，不宣称满足延迟门槛。 |
| **深度/时间稀疏四臂 Pareto 对照**：DROP32、MOD32-KV、DSR6-KV 及完整参考臂 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\experiments\r1-depth-sparsity-four-arm-pareto-closure.md`；对应 `configs/adatad/thumos/`、`tools/bata/` | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 成本回放作业 1262120 已结束 | 参考臂 p50 2848.74 ms；DSR6/MOD32/DROP32 延迟比分别 `1.1123 / 1.1102 / 1.1028`，均变慢，无 Pareto survivor。 |
| **跨帧缓存与压缩对照**：RC32-KV、A-MoD-50、TAR32-FKV、K100-TAR50 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\experiments\r1-tar32-k100-tar50-transform-compression.md`；相关配置/工具在 `configs/adatad/thumos/`、`tools/bata/` | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 多数作业已终态；K100-TAR50 在首个 update 前 hook mismatch 失败 | RC32 相对 MOD32 约下降 `1.77` pp；TAR32-FKV `64.981 / 57.371 / 43.669`，短动作下降约 `3.318` pp；A-MoD 仅 parity 证据；整族停止。 |
| **RACER24 与 GridFuse32-L6 成本微基准**：验证极低预算稀疏调度是否真的减少端到端时间 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\experiments\racer24-gridfuse32-microbenchmarks.md`；`tools/bata/` 成本工具 | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 作业 1262068、1262108 已结束 | RACER24 p50 speedup `0.24964×`、内存比 `1.988`；GridFuse32-L6 p50 speedup `0.56687×`；两者均未形成可接受 Pareto 点，停止。 |
| **有序视频解码复用闭环**：在同一完整窗口人口上测 K100 与 R1 的解码/特征复用收益 | `E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\experiments\ordered-video-decode-reuse-viability-closure.md`；`tools/bata/` 对应 replay/ledger 工具 | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 作业 1262753 已终态 | K100 p50 1310.20 ms，R1 1285.30 ms，延迟比 `0.980999`（约 1.90% 改善，仍未过 5% 门槛）；能耗比 `0.921086`，内存比 `0.751260 / 0.689655`；停止为论文效率 headline。 |
| **BA-FDR-K16 边界感知固定容量双分辨率刷新完整矩阵**：全局 96 载体预测边界/动作性，只对固定 16 个 chunk 做原生局部刷新，并比较 7 个 arm × 3 个 seed | 核心：`E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\opentad\models\backbones\bafdr_wrapper.py`、`opentad\models\projections\bafdr_asymmetric_proj.py`、`opentad\datasets\transforms\bafdr.py`；21 配置：`E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\configs\adatad\thumos\bafdr_k16_*.py`；工具/测试：`tools/bata/bafdr_k16_fullmatrix.py`、`bafdr_k16_fullmatrix_train.py`、`tests/test_bafdr_k16_architecture.py`；协议档案同目录 `bafdr-k16-fullmatrix-protocol.md` | [BA-FDR 分支](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-ba-fdr-k16-fullmatrix-v001) | 预检作业 1266316 已 PASS；截至本快照 1266325 D160、1266326 G96、1266327 U128-ALL48-A0 RUNNING；1266328 U16-UNIFORM-A0、1266329 LATE、1266330 NOKD PENDING；FULL 训练 1266351 已提交并处于 dependency pending；下游评测尚未产生终态。 | 静态账本 `543.48 GFLOPs`、`C_exec=0.494276` 只是代理量；目前没有可填入的正式 BA-FDR mAP/延迟/能耗最终结果。当前代码和测试通过静态/预检不等于科学矩阵完成。 |
| **D2S-TAD 动态双速变焦**：根据动作性在低速全局与高速局部之间切换，计划比较 D160/G96/D2S-U128-B128 | 当前工作区未发现 `d2s_videomae_wrapper.py`、对应配置或测试；历史设计只在用户/远端记录中出现 | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 未实现/未找到可核验本地提交；不能据“计划作业号”标记为运行或完成 | 当前没有本地代码、receipt 或最终指标；应视为提案，需先补齐实现、注册、测试和 PRECHECK。 |
| **PA-TAD 不对称时序金字塔**：只在低层注入局部残差，高层保持全局上下文，计划比较 PATAD-U128-B128 | 当前工作区未发现 `pyramid_aware_asymmetric_proj.py`、PA-TAD 配置或测试；历史提交存在注册/前向语义问题记录 | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 历史实现无效/未在当前树部署；不能把排队作业当成成功 | 已知问题包括 wrapper 注册不完整、forward 未正确使用 burst mask/split/gate；应先修复并通过 focused tests，再决定是否重提。 |
| **ET-TRC Taylor 残差流形主干**：用 Taylor attention、低秩 JVP 和残差块诊断特征刷新误差 | 当前工作区未发现 `et_trc_videomae.py` 或 ET-TRC 配置；历史记录在 `research-wiki/` 外部/远端作业中 | [仓库根目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702) | 当前本地树无实现；不占 GPU，等待明确诊断需求 | 没有当前可复现实验和最终指标；不能作为已实现路线或论文证据。 |

## 当前运行与更新机制

截至本快照，BA-FDR 远程 minute watcher 位于：

- 本地脚本：`E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\scripts\watch_zoomtoken_bafdr_fullmatrix_submit.sh`
- 远端 watcher worktree：`/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_5dba75c7`
- 共享计算目录：`/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_6ae16954_compute`
- watcher 日志：`/data/run01/sczc063/yuzibo/slurm_logs/bafdr_fullmatrix_submit_watch.log`

watcher 每 60 秒检查一次缺失 arm，并在 Slurm 资源允许时提交 FULL 及下游任务；遇到 `MaxSubmitJobLimit` 或 GPU 配额时只记录并等待，不重复提交。另有 Codex 自动化 `zoomtoken-experiment-monitor` 每 30 分钟检查 ZoomToken、DUCA、ET-TRC、ChronoTransport、PA-TAD 和 BA-FDR 的作业、receipt、metrics 与失败日志，并更新本文件；更新只接受可核验证据，不把预检、短跑或静态 FLOPs 记为正式结果。

## 证据边界

主登记册：`E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\experiments\zoomtoken-all-implemented-experiments-registry.md`。<br>
当前研究决策：`E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\query_pack.md`、`E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027\research-wiki\anti_repetition.md`。<br>
任何实验只有在完整官方训练/评测人口、冻结协议和终态 receipt 同时存在时，才可升级为正式科学结果；其余状态均保留为工程证据、失败证据或待验证提案。

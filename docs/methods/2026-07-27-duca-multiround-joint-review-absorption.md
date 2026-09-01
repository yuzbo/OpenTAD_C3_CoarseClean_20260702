# DUCA 多轮联合审阅综合报告吸收与时点校正

## 原始记录

- 报告日期：`2026-07-26`
- 原始附件：
  `C:/Users/skywalker/.codex/attachments/4672e0d4-479e-4a82-a819-7266a000e06c/pasted-text.txt`
- 字节一致归档：
  `docs/methods/reviews/2026-07-26-duca-multiround-joint-review-raw.txt`
- 原文/归档大小：`18,959` bytes
- 原文/归档物理行数：`141`
- 原文/归档 SHA-256：
  `67409BC9B140275BFC6804DD65FACBBEB568719304768A322FCF3A3F54576484`
- 字节一致性：`true`

## 项目裁决

```text
SUBSTANTIAL_ACCEPT_GOVERNANCE_AND_EXPERIMENT_DESIGN
ACCEPT_WITH_CURRENT_FACT_CORRECTIONS
REJECT_STALE_STATUS_AS_CURRENT_CONTRACT
HOLD_REVIEWER_PROPOSED_THRESHOLDS_UNTIL_RATIFIED
```

我认同报告的主要科学治理观点，但不认同把它的全部状态判断原样当成当前事实。该报告
是一份有价值的 `2026-07-26` 审阅快照；PR、代码、数值诊断和运行证据在成稿后继续变化。
因此吸收方式必须是“保留原文、逐项校正、只升级已核验事实”，不能让旧快照覆盖后续证据。

## 2026-07-27 事实复查

### G0 可读治理门

- Draft PR
  [`#2`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/pull/2)
  已于 `2026-07-26T16:16:12Z` 开出，标题为
  `DUCA source audit surface (read-surface only, do not merge)`。
- PR head 为
  `codex/duca-density-transport-20260723@42dba3f90b37243e7965d18b6707e88e81bf7109`，
  base 为 `codex/c3-coarse-clean-20260702`，状态为 `OPEN/DRAFT`。
- PR 描述已提供采样率解码、fixed-budget backward、ST bridge、贡献教师、official-60
  配置、课程恢复、hard-soft gate、TrueTimeFeatureResidual 和几何导出工具的 immutable
  blob permalinks。
- 因此 `G0-read-surface=PASS`。但“入口可读”不等于“独立逐文件源码复核已经完成”；
  `G0-source-adjudication` 仍为 `open`，不能把待复核的 `PARTNER_CLAIM` 自动升级成
  `CODE_FACT`。

### README 凭据治理

- 当前公开 base 分支与 DUCA PR head 分支的 `README.md` 均仍含明文代理认证信息。
- 该内容可追溯到初始提交
  `7eb8a413085b822d6143c19e97fecd09393bd835`，不是仅存在于本地未提交文件。
- 因此报告的“凭据清理未闭环”成立，而且应拆成三个独立动作：
  `redact-current-surface`、`rotate-exposed-credential`、`purge-or-neutralize-history`。
  删除当前 README 文本不能替代凭据轮换，也不能消除历史 blob。
- 本记录不复制用户名、口令或认证 URL，避免二次扩散。

### Stage-2 数值与运行状态

- 报告只观察到旧 Job `1190528` 的 1,000 次有限更新后失败，因此它关于“修复后尚未通过
  D0、完整 Stage-2 仍阻塞”的描述已被后续证据取代。
- 后续诊断把可重复的贡献分布失败隔离为 FP16 mask/temperature 顺序：
  旧实现先填 `-65504` 再除以 `0.7`，无效位变成 `-inf`，随后出现 `0 * -inf`。
  `4c1f5384ae693c74a141619ded03196a72c594ed` 只调整无效位 mask 顺序；focused tests
  和旧失败 batch 的只读复现均为有限。
- 当前唯一有效 continuation 为 Job `1191957`，使用公开精确提交
  `42dba3f90b37243e7965d18b6707e88e81bf7109`。截至
  `2026-07-27 00:59 +08:00` 已完成 4,000 次 post-e9 成功更新并越过 epoch 49；
  optimizer/scheduler/selector/EMA 更新数严格一致，只有四次可恢复 AMP overflow，
  每次一次同 batch replay 成功，loss non-finite 与 replay exhaustion 均为零。
- `2026-07-27 01:05 +08:00` 作业仍为 `RUNNING`。尚无 `epoch_59.pth`、terminal EMA
  OpenTAD official mAP 或完整质量产物。因此数值恢复已获得强运行证据，但完整 offline
  TAD 性能仍未形成，`paper_claim_allowed=False` 保持不变。
- 报告定义的每个 D0 子项没有被逐字照表执行；不能事后声称“报告版 D0 全部通过”。
  正确表述是：原数值原因已隔离并经旧失败 batch 与长程真实训练验证，报告中
  “不得继续 Stage-2”的旧执行阻塞已失效。

## 完全吸收的原则

1. 主证据必须是同协议 terminal EMA official offline TAD mAP 与完整成本，内部
   holdout、coarse AP/AUC、边界 proxy 和中间曲线不能替代。
2. 强基线必须包括 matched exact-uniform、stratified random、低成本 heuristic 与
   `U-Curriculum-Matched`；若实际成本变化，还必须加入 equal-cost `uniform@K_eq`。
3. 推理期必须 teacher-free、GT-free；检测梯度与贡献教师只允许训练期使用。
4. 不使用 official validation/test 结果做 checkpoint、阈值、family 或模块选择。
5. hard one-swap 教师必须补 teacher fidelity；soft surrogate 只能与真实 hard swap
   做 Spearman、top-q overlap 和 sign agreement 对齐后作为训练代理。
6. 固定预算采样率、systematic resampling / probability-proportional-to-size 与 Madow
   类方法必须诚实查新和定位；新颖性只能来自 TAD 任务条件化、可部署梯度桥、课程和
   高 tIoU/成本实证，不能来自“首次系统抽样”。
7. 机制诊断至少覆盖 headroom、teacher fidelity、zero-training cross-pair、temporal
   zoom/FPN assignment、selected-axis 时间扭曲和高 tIoU 边界误差。
8. 低性能必须先对 matched/equal-cost uniform 定位，再检查粗证据、边界距离、微簇、
   soft/hard、检测 cls/reg 与时间轴；禁止直接叠加模块或事后调权重。
9. 完整效率主张必须计入 cheap proxy、selector、decode/random access、H2D、重 backbone、
   detector、peak memory 和 energy；仅报告重 backbone FLOPs 不足以支持系统效率。

## 只作为设计提案吸收的内容

以下内容有价值，但报告本身不能把它们升级为预注册项目合同：

- D1-H 的具体数值阈值与 “oracle gain `<0.5 pp` 即停止学习式 density”；
- D1-A/B/C 的全部具体门槛；
- Path A 的 `+0.7--1.0 pp` 与 Path B 的 `+0.2--0.5 pp` 发表阈值；
- D2-I 八臂矩阵、D2-II 的精确 seed/预算数量和 D2-III 的跨数据集规模；
- 将 ChronoTransport 直接置为 `parked`。

这些条目状态为 `designed_reviewer_proposal`。只有与当前 canonical contract 对齐并由项目
显式批准后，才可成为冻结阈值。尤其 ChronoTransport 在当前 `AGENTS.md/RTK.md` 中是独立
并行路线，不能由本报告单方面取消。

## 当前治理状态

| 项目 | 状态 | 解释 |
|---|---|---|
| 原始报告归档 | `completed` | 字节一致、哈希登记 |
| G0 read surface | `passed` | PR #2 + immutable blob links |
| 独立源码逐项裁决 | `open` | 可读不等于已审完 |
| README 当前面清理 | `open_urgent` | 公开分支仍含明文认证信息 |
| 凭据轮换 | `owner_action_required` | 仅删除文本不能完成 |
| Git 历史处置 | `open_urgent` | 初始提交起存在历史暴露 |
| Stage-2 数值原因 | `tested` | 已隔离并通过旧 batch/长程训练验证 |
| Stage-2 完整性能 | `experiment_running` | 无 terminal epoch-59 EMA |
| 报告阈值/矩阵 | `designed_reviewer_proposal` | 未自动冻结 |
| 论文主张 | `hold` | `paper_claim_allowed=False` |


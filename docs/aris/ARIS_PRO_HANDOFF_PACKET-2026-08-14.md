# ARIS Pro Handoff Packet — DUCA 密度采集路线独立批评（2026-08-14）

> **等待中央 profile61 自动串行槽位，不是等待人工。** 本包是固定 revision + 材料路径 +
> raw evidence 的独立 Pro 批评输入；本 Executor 不操作浏览器、不自行调用 Pro。

## 1. 固定身份

- 会话根：`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`
- pinned revision：`a6bdc084cc145c80b6b2c68d0a38f0deea3e8518`
- branch：`codex/duca-total60-plugin-cvpr-20260727`（worktree dirty，保留，不回滚）
- Pro 冻结路由：`DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002`
- 冻结投影策略：`PRO_P0_PROJECTION_POLICY-v001` + `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001`
- 证据状态：`BLOCKED_PRE_RESULT`

## 2. 请求的批评对象（一次独立 Pro 攻击）

请独立攻击以下**科学命题与 falsifier 设计**，只给 `HOLD` / `REVISE` / `GO` 之一 + 一个
最小反例或盲区：

**命题**：在正确 physical-time decode（raw proposal 于 NMS 前从 selected-q 映回
physical-dense）下，GT-boundary oracle 密度选择相对 exact-uniform 的 mAP 增益是可学习密度
有效性的**上界**；若该上界 ≤ 0，则任何 learned 密度都不能超过 uniform，路线应被证伪。

**Falsifier**：无训练 frozen-detector，比较 `O_gt`（GT 边界 ±radius 集中密度，经冻结
`decode_duca_density_positions_v001` 投影）与 `U`（exact-uniform），官方 THUMOS validation
mAP，逐视频 paired CI。KILL 条件 = CI 下界 ≤ 0；GO 条件 = 点估计 ≥ 门槛且 CI 下界 > 0。

请特别检查：
1. oracle 密度“是 learned 密度的上界”这一论证是否成立（GT 特权 + 单调投影是否真构成
   learned 可达到集的上界）。
2. falsifier 是否仍被坐标 confound / detector-seen / split 泄漏污染。
3. 停止规则（CI 下界 ≤ 0 → KILL）是否过强或过弱。
4. 有无更便宜的、仍然 route-changing 的 falsifier 被我遗漏。

## 3. 材料路径（本地）

- CPR 计划：`docs/aris/ARIS_CPR_PLAN-2026-08-14.md`
- 决策日志：`docs/aris/ARIS_DECISION_LOG-2026-08-14.md`
- 最小实现：`opentad/models/duca/density_decode.py`
- 测试：`tests/test_duca_density_decode.py`
- Pro 决策原文：`.cvpr-pro-lab/pro-reviews/runs/duca-projection-policy-v001/raw-response.md`
- 治理记忆：`research-wiki/query_pack.md`、`research-wiki/anti_repetition.md`

## 4. Raw evidence（关键数字，均来自已封存收据）

- decode-cross（冻结 raw tensors，physical-time decode 恢复）：selected-online
  `0.4126→0.5015`，selected-EMA `0.4128→0.5010`，physical-online
  `0.4011→0.5756`，physical-EMA `0.4030→0.5761`（Avg-mAP）。
- 官方 ActionFormer S0：dense `66.5830`，full×K384 uniform `45.7843`（−20.71pp），
  selected-loss 训练再 −1.96pp。
- DUCA exact-uniform K=384 selected-rank：`0.64458`（checkpoint SHA `17d7461e...`，
  config SHA `9edb24fa...`）。
- 历史全部 learned 选择臂 ≤ exact-uniform（boundary-burst/CellCF/local-cell/protected-E2E）。

## 5. 本轮已验证的最小实现回执

- 冻结 fixture 全通过（本机 CPU + N16R4 双端）：768/384 canonical uniform 止于 767；
  G16-U/G17-E2/EINF/E1/U1/PLEX/G31-U/G32-U/F768-U/G767-U/G385-X；负例 code 全对；
  非恒定 768 解码 ~40-78ms、stride≤4/位移≤16/端点合法。

## 6. 等待与不阻塞声明

- 本包等待中央 profile61 串行 Pro 槽位自动处理；Source 上传失败不阻塞本地科学与代码。
- 在 Pro 返回前，本 Executor 持续推进：oracle 密度选择接入 decode-cross 冻结评估的
  坐标一致性验证与 launch-ready Slurm 收据。

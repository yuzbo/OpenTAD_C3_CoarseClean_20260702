---
type: gap_map
updated: 2026-07-22
---

# Gap Map

| ID | 缺口 | 当前状态 | 关闭条件 |
|---|---|---|---|
| G1 | 同提交强基线缺失 | open | dense/exact-uniform/periodic/random/actionness 与 DUCA matched matrix |
| G2 | ST surrogate 与 hard utility 未对齐 | open | one-swap finite-difference 正相关且优于简单信号 |
| G3 | selected-axis 不等物理间隔 | open | same-selected-frames 几何对照；或新 head 明确改善 |
| G4 | 真实总成本未知 | partial | trained checkpoint 下分别完成 dense-materialization 复现模式与 low-resolution proxy + selected-high-resolution materialization 部署模式的 decode/CPU/H2D/coarse/DP/VideoMAE/head/remap/NMS p50/p95、throughput、memory、energy 矩阵；部署总成本低于 dense 才关闭 |
| G5 | coarse probe 与 selector 错误难分 | open | probe actionness/transition 质量与 selector 条件性能分解 |
| G6 | requested/effective/unique K 不一致 | open | 每样本严格 ledger 与 backbone 实际消费核验 |
| G7 | dynamic MUST 不稳定 | open | 多 target、matched mean-K、真实 K/latency 与稳定 mAP |
| G8 | 高 tIoU 与短动作保护 | open | mAP@0.6/0.7、边界距离、短动作分层不退化 |
| G9 | 插件泛化未证明 | open | 第二 detector；最好第二数据集 |
| G10 | no-leak/provenance | partial | 递归 fail-closed + artifact hashes + clean manifests |
| G11 | 官方 AdaTAD 语义边界 | partial | 论文和 validator 使用 official-derived 诚实口径 |
| G12 | 论文创新性与停止条件 | open | 结果达到 claim threshold；否则降级/转向 |
| G13 | ChronoTransport 近邻碰撞 | open | 相对 SCOPE/Eventful/ResidualViT/PBD/ATR 给出不可还原的 TAD-specific delta |
| G14 | 窗口级风险尺度、排序与选择后校准失败 | open | window 内完整 candidate-vector rank、unique-window cluster CI、actual-selected coverage≥0.85；overcoverage 由 pinball/sharpness/selection rate 约束 |
| G15 | transport feature 优势不稳定 | open | 相对 HOLD 的 feature improvement bootstrap CI 全为正 |
| G16 | cache/skip 之外的新表示 | open | multigrid、compute-value 或 spectral candidate 通过最小 oracle falsification |
| G17 | 输入相关调度价值未证明 | open | Gate 1 先证明 equal-cost frozen-library oracle headroom；Gate 3 再证明 deploy-visible window-vector ranking 与实际动态选择优于冻结 comparator |
| G18 | coarse hidden 语义错误与 direct-boundary bypass | open | 暴露真实 ASFormer encoder state；删除 absolute/raw/direct heads；logits-equivalence 与 transition-only tests 通过 |
| G19 | full-model optimizer exact coverage 未闭环 | open | 实例化 DDP full model 后每个 requires-grad 参数恰好进入一个 optimizer group；exclude 参数真正冻结 |
| G20 | external official ASFormer 不可复现 | open | external source 固定 upstream commit、文件 SHA256、checkpoint/config provenance |
| G21 | TAD 空间分辨率 headroom 与少量 ROI 充分性未知 | open | dense 160/224/256 matched test；oracle ROI/fixed ROI/multi-ROI 在等 heavy-token 和完整成本下比较 |
| G22 | V8 的 `max_unselected_hole=2` 可能过强地逼近均匀覆盖 | discussed | 固定 K=384，对同一保存分数做 G=2/3/5/7 的无训练可达性与 oracle 诊断；仅当更大 G 明显提高终点/短动作 headroom 且不引发错误聚集时，才允许同一 V8 内做单变量 mAP 消融 |
| G23 | Oracle 式边界微簇没有形成完整可部署合同 | open | 先用 train-split Oracle 冻结有 headroom 的 K/G、半径与 3--5 帧局部配额；随后在同一 V8 scorer/DP 中由 deploy-visible 状态转变证据定位 center，完成双侧多帧分配、配额饱和、公平分配、重叠去重与剩余全局预算。val/test GT 不得进入决策；matched terminal-EMA G0 必须同时超过 same-commit U 与旧 Gaussian-mass G0，且完整成本仍有净节省 |
| G24 | frozen coarse hidden 对 sparse TAD detector 的独立表示价值未知 | discussed_conditional | 在 V8 终局与 R0 Oracle reachability 封存后，用同一 commit/P0/exact-uniform K384/6000 updates/seed/terminal-EMA 对比 U0 discard 与 U1 zero-gated post-VideoMAE residual。U1 失败只 KILL fusion；成功也只能支持 acquisition-and-fusion adapter，不得替代 G23 或声称 strict pre-backbone-only |

## 优先级

P0：G1、G2、G4、G6、G8、G13、G14、G15、G17、G18、G19、G20、G21、G22、G23。
P1：G3、G5、G9、G10、G16。
P2：G7、G11、G12、G24。

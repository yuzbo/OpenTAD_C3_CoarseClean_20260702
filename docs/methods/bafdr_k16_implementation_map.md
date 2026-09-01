# BA-FDR K16 Full-Matrix Implementation Map

## 1. 核心架构与机制对齐

- **全局时间载体 (G96)**:
  - 48 个 16-frame chunks 全量经过 VideoMAE-S (96x96 letterbox, 6x6 patches, 288 tokens/chunk)
  - 产生全局 tubelet 特征 $G \in \mathbb{R}^{B \times 384 \times 384}$ 及 chunk 描述子 $G^{\text{chunk}} \in \mathbb{R}^{B \times 48 \times 384}$
- **边界感知固定容量路由 (BA-FDR Router)**:
  - 结构: $\text{LayerNorm}(384) \rightarrow \text{Conv1d}(384, 128, k=3, p=1) \rightarrow \text{GELU} \rightarrow \text{Conv1d}(128, 4, k=1)$
  - 变化率: 对称 Cosine Distance $d_t = 1.0 - \text{sim}(G^{\text{chunk}}_t, G^{\text{chunk}}_{t-1})$
  - 路由分数: $s_t = 0.40 \cdot \text{rank01}(d_t) + 0.30 \cdot \sigma(b_t^{\text{start}}) + 0.30 \cdot \sigma(b_t^{\text{end}})$
  - 规则 Top-K: Peak-first 稳定排序，严格选取固定 $K=16$ 个 chunks
- **真实物理跳过 (True Physical Skip)**:
  - 从 CPU/uint8 源视频中仅针对选中的 16 个 chunks 提取中心裁剪 $128 \times 128$ 区域 ($[96, 26, 224, 154]$)
  - 仅这 16 个 chunks 执行归一化、H2D、Patch Embedding (8x8 patches, 512 tokens/chunk) 和 12 层主干计算
  - 未选中的 32 个 chunks 严格零局部计算
- **稠密残差回填 (Dense Residual Scatter)**:
  - $R_{\text{sel}} = \gamma \cdot \sigma(g_{\text{sel}}) \cdot (P_L(L_{\text{sel}}) - P_G(G_{\text{sel}}))$
  - 散射回全零密集张量 $R \in \mathbb{R}^{B \times 384 \times 384}$，稠密特征 $Z = G + R$（未选位置 $Z_t = G_t$ 严格恒等）
- **非对称金字塔投影 (BAFDRAsymmetricProjection)**:
  - $L_0 = P_0(G) + Q_0(R)$
  - $L_1 = P_1(G) + Q_1(R)$ (Stride-2 Conv1D)
  - $L_2 \sim L_5 = P_2(G) \sim P_5(G)$ (对局部残差 $R$ 严格 Bitwise 恒定)
- **D160 Teacher 边界蒸馏 (Arm BAFDR-K16-FULL)**:
  - 同 Seed D160 epoch-59 EMA 作为冻结教师
  - $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{TAD}} + 0.50 \mathcal{L}_{\text{router}} + 0.20 \mathcal{L}_{\text{feature}} + 0.20 \mathcal{L}_{\text{cls-KD}} + 0.10 \mathcal{L}_{\text{reg-KD}}$

---

## 2. 21-Cell 实验矩阵映射

| Arm ID | 配置文件路径 | 作用 |
|---|---|---|
| `D160` | `configs/adatad/thumos/bafdr_k16_d160_seed{4407,4408,4409}.py` | Dense 上限与同 Seed Teacher |
| `G96` | `configs/adatad/thumos/bafdr_k16_g96_seed{4407,4408,4409}.py` | 纯便宜载体下限 |
| `U128-ALL48-A0` | `configs/adatad/thumos/bafdr_k16_u128_all48_a0_seed{4407,4408,4409}.py` | 全量局部计算/准确率天花板 |
| `U16-UNIFORM-A0` | `configs/adatad/thumos/bafdr_k16_u16_uniform_a0_seed{4407,4408,4409}.py` | 归因 1: 验证自适应路由价值 |
| `BAFDR-K16-LATE` | `configs/adatad/thumos/bafdr_k16_late_seed{4407,4408,4409}.py` | 归因 2: 验证非对称注入价值 |
| `BAFDR-K16-NOKD` | `configs/adatad/thumos/bafdr_k16_nokd_seed{4407,4408,4409}.py` | 归因 3: 隔离蒸馏收益 |
| `BAFDR-K16-FULL` | `configs/adatad/thumos/bafdr_k16_full_seed{4407,4408,4409}.py` | 预注册主候选模型 |

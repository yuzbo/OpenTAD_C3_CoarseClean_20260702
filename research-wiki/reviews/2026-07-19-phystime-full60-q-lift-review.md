# 2026-07-19 PhysTime Full60 / Q-Lift Pro 审查

## 来源

- 原文：
  `docs/methods/reviews/2026-07-19-phystime-full60-q-lift-pro-review-raw.md`
- SHA256：
  `BBD48B6BCE5E4AC612A395561D2EABCBB1F6DB5880B329EF21CAC6808CFBD5E0`
- 完整吸收：
  `docs/methods/2026-07-19-phystime-full60-q-lift-pro-review-absorption.md`
- 审查锚点：commit `0dc5851`，tree `bddc9b9`。

## 独立裁决

结论为**分级认可，不完全认可**。

认可：

- 当前 `57.57%` 是可信的单种子完整训练结果；
- physical-metric 的正确作用域是检测头后段，不是时间感知 backbone；
- 当前两臂公平，旧 `63.61%` 与 dense `68.29%` 只能作外部锚点；
- K/J/Q 必须解耦，旧 feature interpolation 不是中性 Q-lift；
- 应修复全精度跨窗口 NMS，并收窄“无 GT 采样”措辞；
- 下一轮应使用同 commit 的 Q 与 coordinate 2x2 因子设计；
- 当前不是 paper-ready。

保留：

- cross-attention 是优先候选，不是已证明的唯一结构；
- 审查给出的固定收益/成本阈值缺少方差与预算推导；
- 字面 timestamp shuffle 会制造无效非单调时间轴；
- ActivityNet-v1.3 只是第二数据集候选；
- 三 seed 区间下界不能单独承担论文裁决。

## 状态

- `exp:phystime-g1-matched-full60`：
  保持 `full60-single-seed-supported`。
- `idea:phystime-tad-2`：
  物理时间度量获得单种子 full60 支持，独立方法仍未成立。
- `idea:sm-ptaf`：
  吸收 support-preserving physical query lift，状态仍是 `designed`，
  尚未实现、测试或获得 mAP。
- 新 full train：
  未解锁。

## 下一决定性步骤

先完成全精度 NMS、K/J/Q provenance、合法 timestamp counterfactual 和
support-preserving query bridge 的静态/单元/真实 CUDA gate，再运行
Q192/Q384 × 规则轴/物理轴四臂 20-epoch matched 因子实验。旧
`41.28/57.57` 仅作历史锚点，不能混入新架构主表。

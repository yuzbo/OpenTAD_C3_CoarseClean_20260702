# PhysTime 性能下降 Pro 审查吸收记录

日期：2026-07-13

## 1. 来源与完整性

- 原始回复：`docs/methods/reviews/2026-07-13-phystime-performance-drop-pro-audit-response-raw.md`
- 原始附件 SHA256：`651C4CA673073D7E4C05746138C82EBBE2E6174C459516FB40B3EFDCA47305AB`
- 审查分支：`codex/phystime-performance-diagnosis-20260712`
- 正式训练实现锚点：`3ac93a12c299012db64513567d5bdedf0c6d5f71`
- 诊断与审计锚点：`d900c7ce12081de3b7932fab5af8cabe4278abbd`

原文已逐字归档，字节数、行数与 SHA256 均和附件一致。本文件只记录吸收裁决，不替代原文，也不复制权威实验数字；实验数字仍只以 `docs/evaluation/results.md` 为准。

## 2. 状态分级

| 对象 | 状态 | 不能越级声称的内容 |
| --- | --- | --- |
| PhysTime-AdaTAD 1.0 | `tested / negative_current_implementation` | 不能称 paper-ready，也不能外推为 physical-time TAD 无效 |
| 性能下降诊断 | `empirically_supported` | 机制诊断不能自动证明下一架构有效 |
| Pro 审查回复 | `recorded / externally_reviewed` | 外部判断仍需本地逐项 gate 验证 |
| capacity-matched physical-time control | `designed` | 尚未实现、未训练、无 mAP |
| SM-PTAF | `designed` | 回复中的 PyTorch 片段不是已落库代码，也不是通过测试的实现 |

## 3. 总裁决

接受审查的总裁决：**HOLD AND REBUILD**。

这意味着：

1. 冻结 PhysTime 1.0 为可复现负基线，不再通过训练更久、调 NMS、加 endpoint 权重或局部修 attention 掩盖结构混杂。
2. 不 pivot 掉 physical-time TAD 科学问题，因为现有比较尚未公平隔离物理时间建模。
3. 下一步先完成等容量、同上下文、同候选、同 assignment 的 coordinate-only control。
4. 只有 P0 公平性与 feature provenance gate 通过后，才实现并训练 SM-PTAF。

## 4. 已吸收的代码与科学问题

### P0 阻断

1. **Feature-support provenance 未成立。** VideoMAE 原生 tubelet 特征轴与 raw-frame support 轴不是天然一一对应；现有 `192 -> 384` 插值只保证长度相等，不保证语义锚点相等。
2. **所谓三头隔离同时更换了架构与容量。** PhysTime 1.0 删除了 ActionFormer 同等级 temporal projection 和跨 query context，并显著改变可训练容量。
3. **候选拓扑与有效候选 mask 不公平。** 不能把“cell 无直接 observation”直接等价为“该位置没有合法检测候选”；这会系统性伤害 gap 与短动作。
4. **Assignment 不同构。** classification 必须恢复 ActionFormer tied-shortest multi-label 语义，regression 再选择一个 shortest GT。
5. **未归一化绝对秒与无界 content dot product 不可辨识。** absolute seconds 应服务几何、assignment、decode 与 evaluation；表征路径只能使用受控相对时间和有界物理尺度信息。

### P1 高风险

1. PhysTime 缺少与 ActionFormer 等级相当的跨 query 时序上下文。
2. physical-grid 是 selected-rank feature geometry 与 physical assignment 的后置拼接，只能作为有意构造的 geometry-mismatch control。
3. dropout 不能作用于已经归一化的 mass base path，否则训练态不再满足测度守恒。
4. endpoint 不增加候选、不进入 score 或 decode，只能通过共享塔间接作用；它必须排在结构修复之后做因果消融。
5. “无 GT 采样”必须精确表述为 **K384 子采样无 GT**；训练时上游 AdaTAD crop 仍可能使用 annotation-aware truncation。
6. 仓库中的 hash 登记不能替代可访问的 checkpoint、prediction、manifest 与原始日志归档。

### P2 次要但需修复

1. 滑窗末窗去重需要 manifest regression test。
2. global-zero cell phase 与尾部 clipped cell 的不对称需要作为次级几何消融，而不是当前主因。
3. `AGENTS.md` 与路线档案中“尚无 mAP/尚未实现”的过期状态必须修正。

## 5. 唯一推荐路线的正确理解

Pro 推荐的 `SM-PTAF` 不是“把三个新模块全部堆起来”，其最小不可分核心是：

> 原生 tubelet feature-support provenance，加上不插值、不跨 gap 填补的 measure-preserving set-to-physical-query pyramid。

其中：

- capacity-matched Physical-Time ActionFormer 是必须先做的科学 control，不是自动成立的论文贡献；
- mass residual 是保护 support measure 不被 learned content logits 覆盖的核心算子；
- ActionFormer 等级 encoder、候选数对齐和多标签 assignment 属于公平性条件；
- endpoint 暂不进入主模型；
- 秒坐标继续用于 GT、候选几何、回归、decode、NMS 与 evaluation；K 只允许决定 matched comparison 的候选基数，不能定义坐标或 rank stride。

## 6. 对 Pro 代码片段的本地保留意见

这些片段是设计参考，不能直接复制后宣称完成：

1. `decode_seconds()` 示例只按 `[0, duration]` clamp；正式窗口使用 absolute-video seconds 时必须显式处理 `domain_start_sec/domain_end_sec` 与 full-video duration，避免丢失 window offset。
2. `build_tubelet_support_atoms()` 只覆盖 audited `tubelet_size=2`。它给出 anchor provenance，不代表 TIA/ViT 的完整神经感受野；必须追加 native-token Jacobian 与 rank-mixing 诊断。
3. `Q_l=ceil(K/2^l)` 用于候选公平，不意味着 native token 数等于 query 数。K384、J192 与 Q0=384 三个概念必须分别审计。
4. zero-coverage query 使用 gap token 和跨 query context 是“保留检测候选”，不是重建缺失 feature；论文必须避免把它写成无条件 no-imputation。
5. all-masked attention 必须使用项目已验证的 fail-closed masked-softmax 合同；不能只依赖 `finfo.min` 后 softmax 再清零而缺少 AMP 回归测试。
6. 参数量差异小于某个比例只是必要条件，不是功能容量等价的充分条件。深度、宽度、local window、候选拓扑、assignment 和训练更新必须一起匹配，禁止靠无效参数或任意 FFN 膨胀“配平”。
7. ActivityNet-v1.3 作为第二数据集只是候选；必须先核验 raw-video 许可、成本、标注时间合同和可复现性，再进入正式矩阵。

## 7. 固定执行顺序

1. **G0 feature provenance**：移除主路线中的 `192 -> 384` feature interpolation，建立 native tubelet multi-atom support，并做真实 CUDA/Jacobian/provenance gate。
2. **G1 coordinate-only control**：相同 ActionFormer 容量、候选、assignment、head 与更新，只切换 selected-coordinate 和 physical-coordinate。
3. 修复 raw absolute center、candidate parity 与 tied multi-label assignment。
4. 在 survivor 上比较 mass residual、bounded/off content correction。
5. 只有上述机制合同通过，才实现完整 SM-PTAF 并进入单 seed pilot。
6. endpoint 必须最后消融；多 seed、第二数据集、sampling families 与完整成本账本在 pilot 通过后解锁。

## 8. 论文主张边界

当前唯一可写入论文的结论仍是：PhysTime 1.0 在当前 THUMOS14/K384 matched run 中失败；由于 detector capacity、候选、assignment 与 feature provenance 混杂，该负结果尚未裁决 physical-time-native TAD 的科学假设。

`SM-PTAF` 的一句贡献表述只能作为**目标主张**，不能写成已得到实验支持的摘要结论。只有 coordinate-only control、SM-PTAF 因果增量、多 seed、第二数据集、固定无 GT sampling families、置信区间和全栈成本闭环后，才可能升级为 `paper_ready`。

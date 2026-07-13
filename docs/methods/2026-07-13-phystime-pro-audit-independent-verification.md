# PhysTime Pro 审查独立核验与路线裁决

日期：2026-07-13

审查对象：`codex/phystime-performance-diagnosis-20260712`，当前文档提交 `bb23fe7`；正式训练实现 `3ac93a1`。

## 1. 总裁决

**不完全认同，但认同 `HOLD AND REBUILD`。**

- 对 PhysTime-AdaTAD 1.0 的结果、结构混杂和停止继续调参的判断，核验后成立。
- 对 physical-time TAD 研究问题仍可继续，核验后成立；当前负结果只否定 1.0 实现。
- 对 `SM-PTAF` 作为值得验证的候选方向，原则上认同。
- 不认同把 `SM-PTAF` 直接写成“唯一最终模型”。它仍是 `designed` 假设，Pro 伪代码还存在 token/query 混杂、tubelet 可加性、窗口解码边界和 no-imputation 表述风险。

当前正确状态是：**实验可信，当前方法失败，研究问题未被否定，下一版必须先做因果隔离。**

## 2. 结果复核

远端 2026-07-13 复核：真实 gate `1159491`、稳定性 gate `1159492`、三头正式训练 `1159493/1159494/1159495`、最佳 checkpoint 复算 `1159819/1159820/1159821` 均为 `COMPLETED 0:0`。正式快照关键合同测试为 `69 passed`。

| Head | Best epoch | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected-axis | 59 | 79.87 | 74.15 | 66.12 | 56.02 | 41.87 | 63.61 |
| physical-grid | 57 | 77.09 | 71.80 | 63.74 | 50.74 | 32.34 | 59.14 |
| PhysTime 1.0 | 59 | 72.70 | 68.38 | 60.94 | 49.06 | 34.96 | 57.21 |

PhysTime 1.0 相对 selected-axis 的 Avg 低 6.40，mAP@0.7 低 6.91；相对 physical-grid 的 Avg 低 1.93，但 mAP@0.7 高 2.62。它没有胜过任一 sparse baseline，也没有达到 dense AdaTAD 68.29 的本地参考锚点。

预测分解进一步表明，失败主要发生在候选覆盖、类别排序和短动作，而不是所有已命中候选的边界回归：PhysTime 的 all class-aware R@0.7 为 79.70，selected-axis 为 89.95；`<1 s` class-aware R@0.7 为 7.08，对照为 50.00；但在 IoU>=0.7 的成功匹配中，PhysTime start/end MAE 为 0.361/0.333 秒，优于 physical-grid 的 0.402/0.406 秒。

## 3. 已确认的代码问题

1. 配置将 384 个稀疏帧分成 24 个 16-frame chunk；VideoMAE tubelet size 为 2，原生时间长度为 192，随后 `Interpolate(size=384)`。长度相等不等于 feature-support provenance 成立。
2. PhysTime 同时替换 projection、跨 query context、head、候选密度和 assignment。其 trainable adapter+detector 只有 ActionFormer 对照的 26.58%，三头不是纯坐标隔离。
3. PhysTime 测试候选均值 397.52，对照为 748.86；短动作正样本显著不足。
4. `output_mask = query_mask & coverage > eps` 把“当前 cell 无直接观测”误当成“没有合法检测候选”。
5. query embedding 直接输入 raw `center_sec/width_sec`；诊断确认 absolute seconds 主导 content query，粗层 learned content logits 压过 relative-time 项。
6. 归一化 attention weights 后再 dropout，训练态不再严格保持质量和为 1。
7. PhysTime classification 只保留一个 `min_index` 标签；ActionFormer 对 tied-shortest GT 保留多标签，assignment 不同构。
8. endpoint 分支不进入推理 score 或 decode，只通过共享 regression tower 间接影响结果，不能补回候选覆盖。
9. K384 子采样本身无 GT，但训练 dense window 先经过 annotation-aware `random_trunc`；只能表述为“K 子采样无 GT”。

## 4. 对 Pro 重建方案的独立保留意见

### 4.1 两原子 support 仍不是严格可加测度

一个 tubelet token 先由同一个 Conv3d tubelet kernel 联合编码两个 selected frames，随后又经过非线性网络。如果两帧跨越很大物理 gap，把该 token 绑定为两个离散 atoms 可以记录 anchor provenance，却不能把 token value 严格拆成两个可加证据。query 只覆盖其中一个 atom 时，使用的 value 仍包含另一个 frame。

因此 2.0 最多可以先声称“set-valued anchor support”和“support-weighted aggregation”，不能直接声称 feature 本身 measure-preserving。若最终坚持严格测度语义，需要额外验证 frame-separable tokenizer、tubelet size 1，或显式 physical-gap-conditioned tubelet stem；三者都会改变预训练和计算，必须独立对照。

### 4.2 `K=384`、`J=192`、`Q=384` 必须分开

删除 `192 -> 384` interpolation 后，native observations 是 `J=192`；把候选恢复到 `Q0=384` 必然新增 set-to-query lift。若在所谓 coordinate-only gate 同时引入这个 lift，就再次把坐标、投影算子和候选恢复混在一起。

第一轮必须先做 `Q=J` 的无 lift temporal-metric control，或让 selected/physical 两侧使用完全相同的中性 query lift。随后才能单独测试候选恢复和 support-mass lift。

### 4.3 K 决定候选数只适用于公平控制

`Q_l=ceil(K/2^l)` 能匹配 detector capacity，但不能成为“任意不规则观测检测器”的唯一自然定义。最终系统需分别报告 observation budget、视频持续时间、candidate density 与 observability；K 不能重新变成隐式 rank stride。

### 4.4 gap token 是潜变量推断，不是严格 no-imputation

保留 zero-coverage query 是正确的；但 gap token 加跨 query encoder 会推断未观测区的候选。准确表述应是“无 feature interpolation、无伪造 observation support”，不能笼统称“无插补”。

### 4.5 参数量接近不足以证明公平

参数差异小于 1% 只是必要条件。projection 深度、跨 query context、候选拓扑、assignment、训练更新、MACs 和有效 receptive field 也必须同构；不能靠扩大无关 FFN 把参数数目填平。

### 4.6 Pro 伪代码的窗口 clamp 不完整

`decode_seconds()` 把起点 clamp 到 0、终点 clamp 到 duration；对于绝对秒窗口 `[a,b]`，应 clamp 到 `domain_start_sec/domain_end_sec`，并明确 full-video seconds 与 window-local seconds 的唯一转换点。

## 5. 可行性分级

| 对象 | 裁决 | 理由 |
| --- | --- | --- |
| PhysTime-AdaTAD 1.0 | 不可继续作为主方法 | 稳定但明确负结果，且存在多重结构混杂 |
| physical-time TAD 科学问题 | 条件可行 | 不规则观测下 selected-rank 几何确有真实问题，但尚无正向 detector 证据 |
| capacity-matched physical-time ActionFormer | 高可行、必要对照 | 最接近现有强骨架，能先裁决 temporal metric 的价值 |
| SM-PTAF | 中等可行、高风险 | 机制有潜力，但尚未实现；易被质疑为 mTAN-style regridding + ActionFormer |
| 当前 CCF-A 论文主张 | HOLD | 单数据集、单 seed、负结果、无公平 control、无完整成本闭环 |

新颖性不能写成“首个 continuous-time TAD”或“首个 actual-time TAD”。可防守的目标只能是：**面向原生不规则视频证据，显式维护非扩张 observation supports，并在不伪造缺失观测的前提下建立物理时间候选与定位。**

## 6. 固定下一步

### G0：provenance gate，不训练

1. 删除实验分支中的 feature interpolation，暴露 `J=192` native tubelet tokens。
2. 记录 raw frame index -> tubelet token -> TIA 后 token 的 lineage、mask 和 hash。
3. 用扰动/Jacobian 审计两帧 tubelet 融合与 TIA rank mixing；不得把 atom anchor 写成完整神经感受野。
4. 对跨大 gap 的 frame pair 单独测试，决定是否接受 set-valued token，还是进入 frame-separable tokenizer 分支。
5. validator 分开登记 `K`、`J`、`Q`、候选数、参数、MACs 和有效 mask。

### G1a：无 lift 的 matched temporal-metric pilot

在相同 native `J=192` tokens、相同宽度/深度/head/assignment/更新下，令 `Q=J`，只比较 selected-rank metric 与 physical-time metric。这里应称“matched temporal-metric control”，不能夸大成只改一个标量坐标。

若 physical 版本在 Avg、mAP@0.7 和短动作 recall 上同时明确劣化，则停止堆叠 mass/endpoint，先裁决 temporal operator 本身。

### G1b：共享候选恢复

只有 G1a 存活后，给 selected/physical 两侧使用完全相同的中性 `J192 -> Q384` query lift，验证候选密度恢复是否解释短动作差距。不得在这一 gate 引入 support-mass residual。

### G2：support-mass 增量

在 G1b survivor 上，仅把中性 lift 替换为 mass base + 小幅 bounded correction。mass path 不 dropout，absolute seconds 不进入 content embedding，zero-coverage query 保留但明确标记未观测。endpoint 保持关闭。

### G3：正式证据

单 seed pilot 通过后才运行三 seeds；THUMOS14 至少比较 selected-metric、physical-metric、SM-PTAF 三者，并报告高 tIoU、短动作、gap 分组、paired-video bootstrap 和全栈成本。第二数据集与多 sampling family 只在 THUMOS 因果链成立后解锁。

## 7. 当前允许写入论文的唯一结论

在 THUMOS14 raw-video K384 单种子协议下，PhysTime-AdaTAD 1.0 稳定完成但弱于两个 sparse controls。由于 feature provenance、容量、上下文、候选和 assignment 同时变化，该负结果不能外推为 physical-time TAD 无效；SM-PTAF 仍是未实现、未测试的重建假设。

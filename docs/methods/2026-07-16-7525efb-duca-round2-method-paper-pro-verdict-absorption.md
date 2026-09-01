# DUCA `7525efb` 第二轮方法/论文 Pro 裁决吸收记录

## 来源与完整性

- 原附件：
  `C:/Users/skywalker/.codex/attachments/d256ace7-867b-4ce0-a999-ab9bb3bae56e/pasted-text.txt`
- 字节一致归档：
  `docs/methods/reviews/2026-07-16-7525efb-duca-round2-method-paper-pro-verdict-raw.txt`
- 原文与归档长度：`11877` bytes
- SHA-256：
  `B4415ABA4B7B779257DF0F0D4E107586181C4DCBCBFF9F8B38BADB156A191E0B`
- 代码对象：
  `7525efb2e07214615a59c482443246174a6adaf1`
- 输入交接包：第一轮 `VISIBLE / GO_TO_REAL_GATE`，无静态 P0，真实 loader、
  AMP/DDP、mAP 和成本证据缺失。

## Reviewer 总裁决

Reviewer 给出 `REDESIGN`，唯一推荐是 coverage-preserving local-cell
deformation：以 exact-uniform anchors 划分互不重叠 cells，每格恰选一帧；保留
binary-state -> transition evidence 与 train-only detached detector-derived utility；
删除全局自由选帧、全局 homotopy 和为 shared-remove 冲突设计的 Gram-whitened
proximal。

项目吸收裁决为：

```text
PARTIAL_ACCEPT / ACCEPT_CORE_LOCAL_CELL_REDESIGN
```

这不是对全部细节的完全认可，也不是把 Local-cell DUCA 宣布为最终论文模型。其状态从
`discussed` 升为 `designed`，仍未 `implemented`、`tested`、
`experiment_running`、`empirically_supported` 或 `paper_ready`。

## 完全认可的核心部分

1. **失败反例成立。** 全局 exact-K/max-hole 可以在长动作强边界附近堆积采样，却遗漏
   被低分辨率 probe 平滑掉的短动作；selected-axis reassignment 还可能让 detached
   teacher 合理地偏爱这种错误策略。数学方向正确不等于任务效用正确。
2. **Coverage 应成为可行集不变量。** one-per-cell 使 exact-uniform 位于可行集中心，
   不再依赖弱 coverage loss 或 post-hoc max-gap repair 保底。
3. **Hard/soft 必须使用同一 local-cell family。** 每格 hard argmax 与同格 softmax
   是清晰、可穷举和可审计的结构合同；零初始化并显式 anchor tie-break 后，step 0
   可以严格等于 exact-uniform，无需全局 alpha homotopy。
4. **保留间接边界初心。** action head 仍学 binary state；selector 读取 delta
   logit/entropy/hidden、绝对变化和 cosine change，不允许 unrestricted absolute hidden、
   raw-RGB mean、位置编码或直接 endpoint head 绕过机制。
5. **Detector feedback 应与真实 hard action 对齐。** train-only baseline 与少量不同
   cell flips 的官方 `cls+reg` 差值，比 raw-pixel ST bridge 更诚实；推理删除 teacher。
6. **完整成本必须实测。** 50% heavy frames 不等于 50% wall-clock 节省；decode、H2D、
   probe、selector、gather、heavy stack、head 和 NMS 都必须进入 C7。
7. **实验顺序正确。** redesign/tests -> synthetic contract gate -> real-loader CUDA
   gate -> forced-overflow DDP pilot -> matched seed-0 -> result-to-claim -> additional
   seeds -> cost，之前不得解锁动态预算、第二检测头或 physical-grid 救火。

## Local-cell 数学与结构复核

对互不重叠 cell 中的不同 flips，若每个 incidence row 仅含该 cell 的
`+e_add-e_remove`，各行支撑集不相交，因此 `AA^T=2I`。Reviewer 给出的 weighted
logistic：

```text
u_m = L_det(S) - L_det(S_m)
r_m = s(add_m) - s(remove_m)
L_u = sum |u_tilde_m| softplus(-sign(u_tilde_m) r_m / tau)
      / (sum |u_tilde_m| + eps)
```

在 `u_m>0` 时推动 `r_m` 增大，在 `u_m<0` 时推动 `r_m` 减小，局部符号正确；在
候选 cells 不重复、add/remove 合法且非同一位置的前提下，不需要 Gram solve 或 forward
内 `autograd.grad`。

本轮独立枚举 `T=768,K=384` 的 nearest-anchor cells：384 个 anchors 唯一；cell
宽度分布为 1 个宽度 1、382 个宽度 2、1 个宽度 3；每格任取一帧时最大可能
`max_unselected_hole=3`。因此该特例的 coverage 结论成立，但实现必须显式固定 midpoint
归属、anchor tie-break、short-window `effective_k` 和 padding 规则，不能依赖框架默认
`argmax` tie-break。

## 不完全认可与必须纠正之处

### 1. TAPS 被错误替换为 TAPOS

原回复称 TAPS 无唯一展开并改用 TAPOS。项目语境中的 TAPS 是 ACCV 2024
`Temporal Attention-based Pruning and Scaling for Efficient Video Action Recognition`，
使用跨时间上下文的 per-layer/per-filter 动态剪枝与缩放：
`https://openaccess.thecvf.com/content/ACCV2024/html/Dinai_TAPS_Temporal_Attention-based_Pruning_and_Scaling_for_Efficient_Video_Action_ACCV_2024_paper.html`。
附件引用的 `arXiv:2005.10229` 是 Temporal Action Parsing/TAPOS，不是同一工作。
Related-work 表必须恢复 TAPS；TAPOS 可另列，但不能替代。

### 2. 不能继续把 detached utility 叫 C4 直接梯度

现有 `claim:C4` 是“下游 detector 梯度改善 selector”。Local-cell teacher 在
`no_grad` 下提供 detached policy utility，仍不直接检验 C4。未来应由正式 claim
审计另立或改写为“detached detector-derived local policy utility improves selector”；
在此之前 C4 保持 `unproven`，不能借用编号制造机制连续性。

### 3. Local-cell 是最终候选，不是已成立的最终模型

one-per-cell 把可跨格的全局预算重分配收缩成 uniform residual learning。它提高 coverage
和几何稳定性，却也大幅缩小表达能力，无法逼近可跨 cell 聚集的 GT Oracle。合法主张必须
改为“语义引导的局部 uniform deformation”，不能继续声称广义动态预算分配已经实现。

### 4. 数值权重、schedule 与阈值没有证据地位

`1.0/0.5/0.25` losses、utility weight 0.25、20% warmup、10% ramp、teacher 每四步、
EMA 0.999、`+0.30/+0.20` mAP 门槛和固定百分比成本门槛都是可执行 proposal，不是由
当前代码或实验推导出的唯一值。可以预注册为首个 bounded configuration，但不能在未测前
写成“正确训练方案”或 CVPR 普适门槛。

### 5. 单个 seed-0 不应成为科学上的永久否定

严格 seed-0 screen 可停止当前 Local-cell 配置和后续无界调参；但 `Delta Avg<=0` 的
单次结果不足以从科学上永久否定所有 DUCA 假设。若差值接近零，应结合 paired-video
bootstrap、boundary/short-action 机制指标和预注册 practical-equivalence margin 判断。
仍然禁止以 MUST、X3D、第二头或 physical-grid 无界续命。

### 6. direct-boundary attribution 只能延期，不能永久删除

它不是 seed-0 C3/C7 最小矩阵的必需项，可以暂缓；但如果论文最终主张“由粗状态变化
间接定位优于直接小模型边界预测”，direct-boundary attribution 是必要消融。只有在论文
主动删除该机制主张时，才可从最终实验中删除。

### 7. selected-axis 风险只是被限制，没有被证明解决

Local-cell 把间隔波动限制在很小邻域，因此风险应下降；这不等于 geometry 已正确。
仍需 same-selected-frames coordinate attribution、GT remap/inverse-map round-trip 和
high-tIoU/short-action 分解。若 local-cell 仍不能保护 @0.7，应停止该候选，不在本轮引入
physical-grid。

## 冻结后的设计合同

当前允许实现的唯一 DUCA 设计为：

```text
完整离线低分辨率窗口
-> 64x64 spatial stem + 官方 ASFormer binary-state probe
-> deploy-visible transition descriptors
-> shared transition scorer
-> exact-uniform anchored one-per-cell selector, K=384
-> 原始 RGB hard gather
-> official AdaTAD components with wrapper
-> ActionFormerHead
```

训练仍是一次联合训练循环，但 hard acquisition 不可微：action BCE 更新 probe；transition
target 更新 encoder/scorer；coverage 与 detached local-flip utility 更新 scorer；官方
detector `cls/reg` 只更新 heavy stack。不得把它称为 detector loss 直接穿过 hard selector
的端到端梯度学习。

尚未冻结的内容包括具体 loss 权重、teacher cadence、warmup/ramp、EMA decay、GO/KILL
数值阈值和第二数据集选择。它们必须在实现前登记为首个、唯一的 bounded protocol，或由
mechanism gate 给出修改理由。

## Claim 与实验影响

- `7525efb` 保持 `tested`，但不再是允许直接 full-train 的最终方法；它是 Local-cell
  redesign 的审计前驱。
- DUCA-CellCF 升为 `designed`，尚未实现或测试。
- C3 保持 `unproven`；下一合法比较是 same-commit matched exact-uniform、local-cell
  transition beta=0 和 local-cell detached utility。
- C4 保持 `unproven`，且当前 local utility 不直接检验现有 C4 文本。
- C7 保持 `unproven`；bare-uniform 仅作为成本下界，不能替代 matched trained control。
- 任何历史 64/65/oracle 78 数字仍是协议不匹配背景或 privileged ceiling。

## 有界下一步

1. 先实现 local-cell hard/soft family、显式 anchor tie-break、different-cell flip builder、
   weighted logistic 和完整 provenance；不启动训练。
2. 增加 exhaustive cell、uniform identity、short/full/mixed、flip sign、no-information
   utility、coordinate round-trip、optimizer ownership 和 inference-no-leak tests。
3. 通过 exact-commit synthetic gate 后，再运行真实 THUMOS loader、world-size=1 DDP、
   official `train_one_epoch`、forced AMP replay、EMA/schedule 的 CUDA gate。
4. Pilot 通过后只跑 matched seed-0 三臂；结果明显不优于 exact-uniform时停止当前 DUCA
   论文路线，不引入此前冻结的扩展。
5. 只有 C3 与 detached-utility mechanism 均得到支持，才增加 seeds、成本和必要的
   direct-boundary attribution；C7 通过后才讨论第二数据集或 detector generality。

# DUCA `7525efb` Pro 第一轮：代码、数学与训练合同审计

你现在是 **PyTorch 训练系统审计员、离线 TAD 代码审稿人和反方证明者**。本轮只回答一件事：**GitHub 精确提交 `7525efb` 是否实现了一个数学方向正确、训练/推理合同一致、可进入真实 CUDA gate 的 DUCA fixed-384 候选？**

不要在本轮讨论论文创新性、替代研究路线、多检测头、动态预算或大规模实验规划。发现缺陷时只给最小修复，不扩展新方法。

## 1. 唯一审计对象

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 永久提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/7525efb2e07214615a59c482443246174a6adaf1>
- SHA：`7525efb2e07214615a59c482443246174a6adaf1`
- 导航分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-transition-only-20260711>
- 上游 AdaTAD/OpenTAD：`sming256/OpenTAD@1aa8ca4ac5e846b1e8ff69298dd6607121a01589`
- 上游 ASFormer：`ChinaYi/ASFormer@e1bbe4f3ed083748f91467c51a63ac2a8b9277ad`

先输出可见性证书，列出实际打开的 commit、文件和行号。无法读取精确提交时只输出 `VISIBILITY_BLOCKED`。所有判断标记为 `CODE_FACT`、`REPO_ARTIFACT`、`USER_REPORTED_UNVERIFIED` 或 `PROPOSAL`。

## 2. 最小方法背景

任务是 **离线 TAD**，不是 Online TAD。项目方声称实际路径为：完整 `T=768` 窗口 -> 64x64 低成本 spatial stem + 官方 ASFormer 二分类时序模块 -> `p_action`、不确定性与 hidden 的状态差分 -> transition-first scorer -> exact-`K=384`、`max_unselected_hole=15` structured selection -> 采集原始 RGB -> VideoMAE-S Adapter -> ActionFormerHead。

训练时使用 binary actionness 与 transition/endpoint supervision。另有 train-only hard one-swap teacher：用官方 `cls_loss + reg_loss` 评估 baseline 与最多 4 个可行 swap，定义 `u=L_baseline-L_swap`。该 teacher 位于 `no_grad`，所以这是 detached detector-utility supervision，不是直接 detector gradient estimator。

## 3. 必读文件

逐行检查以下核心文件即可，不要扩张到无关路线：

- `opentad/models/duca/counterfactual_utility.py`
- `opentad/models/selectors/duca_online_frame_selector.py`
- `opentad/models/detectors/actionformer.py`
- `opentad/models/duca/structured_selection.py`
- `opentad/models/duca/transition_only.py`
- `opentad/models/duca/acquisition.py` 中 fixed-K/max-gap 解码与 gather 合同
- `opentad/cores/train_engine.py` 中 AMP replay/successful-update 合同
- `opentad/cores/optimizer.py` 中参数覆盖
- `configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py`
- `tools/bata/run_duca_transition_only_formal_full_model_gate.py`
- 对应 counterfactual、FP32、all-short、structured-selection、optimizer 和 formal-gate tests。

## 4. 四项强制审计

### A. 真实调用图与梯度所有权

给出 train/inference 调用图及“模块 x loss”矩阵。至少覆盖 spatial stem、ASFormer encoder/decoder/action head、transition scorer、structured selector、VideoMAE Adapter、projection、ActionFormerHead。每格标明直接梯度、代理梯度、detached teacher、stop-grad 或无路径，并给出 `file:line`。

明确回答：粗分类器是否仍主要学习 binary action state；transition/endpoint 是否进入共享 encoder；selector 是否真的使用 ASFormer hidden 差分；主 detector loss 能否影响 hard selection；推理是否 GT/teacher/cache-free。不得把 detached one-swap utility 称为直接下游梯度反传。

### B. Signed score-space proximal 数学

代码声称：swap incidence 为 `A`，中心分数 `s`，`d=As`；detached utility 归一化为 `u_tilde`；`G=AA^T`，`v=G^-1 u_tilde`，`d*=stopgrad(d)+eta v`，`L=(1/(2M))||d-d*||^2`。

独立推导 `A(-grad_s L)`，并检查 batch mean、外部 loss weight、AMP、GradScaler 与 DDP 后符号是否仍正确。必须覆盖 shared-remove、不同候选数、all-positive/all-negative/mixed/zero、all-short、masked/非法索引、重复 swap、奇异或病态 Gram。判断 `cond(G)<=5` 是候选结构的可证明结论还是 gate 的偶然现象。

严查 `score_space_utility_alignment()` 是否循环自证；`autograd.grad` inside forward 是否影响图、显存或 DDP；FP32 禁用 autocast 是否完整。若失败，给出最小真实代码 patch 和对应测试；不要在本轮另造新目标。

### C. AdaTAD 与坐标可比性

与指定上游逐项比较 detector、VideoMAE Adapter、ActionFormerHead、loss normalizer、point assignment、decode/NMS 与 base config。准确区分官方组件复用、wrapper 扩展、输入长度变化、selected-axis、GT remap 和源代码差异。

重点核验 baseline 与 swap 的 detector loss 是否可比：更换 selected positions 后，GT remap、selected-axis 几何、point assignment 和 frozen normalizer 会不会让 `L_baseline-L_swap` 混入坐标扭曲而非真实帧效用。

### D. AMP/DDP/gate 合同

检查 teacher 是否恢复 Python/NumPy/Torch CPU/CUDA RNG、非 selector buffers、训练模式和 loss normalizer。检查 `with_cp=False + static_graph=False + find_unused_parameters=True + world_size=1` 的必要性及与官方设置的差异。

确认 132 epoch 是否被严格约束为 13,200 次成功 optimizer/LR/EMA/selector-schedule 更新；AMP skip 是否同 batch、同 RNG、同 mutable state 重放；checkpoint/resume 是否保持合同。

当前 formal gate 虽拒绝 dirty tree并记录文件 SHA，但仍声明 `input_provenance=deterministic_synthetic_contract_probe`、`real_dataset_loader_executed=False`。请限定它能证明什么，并给出最小真实 THUMOS loader CUDA gate patch：真实 GT、full/mixed/all-short、outer AMP、实际 optimizer/scaler/EMA/schedule。不得降低阈值或过滤坏样本。

## 5. 当前证据边界

- 精确提交没有 CUDA gate、真实 loader gate、pilot、full train、mAP 或成本结果。
- 项目方自报 clean Linux focused tests 为 `160 passed, 7 skipped`，其中 CUDA-only 测试未执行；这是 `USER_REPORTED_UNVERIFIED`。
- 前一提交 `a6903ae` 的 Job `1165646` 因 utility-direction gate 失败；`1165650` shell 失败、`1165654` 取消，均不是性能证据。
- 历史 64.34、64.352、65.696 和 oracle 约 78 均协议不匹配，本轮不得用于方法优劣判断。

## 6. 输出限制与交接包

正文控制在 **7000 个中文字符以内**，按顺序输出：

1. 可见性证书；
2. `GO_TO_REAL_GATE / HOLD_FOR_PATCH / KILL_IMPLEMENTATION`；
3. 调用图与梯度矩阵；
4. signed proximal 推导或反例；
5. P0/P1/P2 表，每项必须有 `file:line`、影响和最小修复；
6. AdaTAD/坐标差异表；
7. CUDA real-loader gate 的最小 patch/test 清单；
8. 最后单独输出以下机器可读交接包，不要在其中写长解释：

```yaml
HANDOFF_PACKET:
  audit_commit: 7525efb2e07214615a59c482443246174a6adaf1
  visibility: VISIBLE | BLOCKED
  implementation_verdict: GO_TO_REAL_GATE | HOLD_FOR_PATCH | KILL_IMPLEMENTATION
  actual_method_one_sentence: "..."
  direct_detector_gradient: true | false
  train_inference_isomorphic: true | false | conditional
  official_adatad_fidelity: source_identical | official_components_with_wrapper | materially_changed
  signed_proximal_math: pass | fail | conditional
  coordinate_utility_validity: pass | fail | unresolved
  amp_ddp_contract: pass | fail | unresolved
  real_loader_gate_exists: true | false
  p0_blockers: ["file:line - ..."]
  p1_risks: ["file:line - ..."]
  required_minimal_patches: ["..."]
  forbidden_claims: ["..."]
  allowed_next_step: "..."
```

不得在本轮进行文献综述、CVPR 新颖性评分、动态预算/X3D/SlowFast/MUST/第二检测头讨论，也不得给大规模实验矩阵。

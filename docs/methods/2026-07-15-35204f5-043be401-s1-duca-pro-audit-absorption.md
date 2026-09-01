# S1 / DUCA Pro 审查吸收裁决

## 1. 审查对象与来源

- 审查原文附件：
  `C:/Users/skywalker/.codex/attachments/69a2a56a-019c-43d1-9063-a2333ce34faa/pasted-text.txt`
- 仓库内字节级归档：
  `docs/methods/reviews/2026-07-15-35204f5-043be401-s1-duca-pro-audit-raw.txt`
- 原文与归档 SHA-256：
  `AC54C5B633DC9FD0CD801B2B12B2C4E44114E16B7569C220B77528674E2D04E2`
- Spatial Zoom S1 审查提交：`35204f58fd3e91d7cf8f5888928a41e9bf6c2e72`
- DUCA 审查提交：`043be401ba2b694342dc395f263e9a9858628d69`
- 审查总裁决：`STOP_AND_FIX`。

本记录区分审查者陈述、独立复核事实和后续设计选择。审查文本中的
patch、阈值和统计替代方案均不是已经实现或经过实验支持的事实。

## 2. 独立代码复核

以下主要发现与精确提交中的代码一致：

1. DUCA 在 GradScaler overflow 时跳过 optimizer update；scheduler、EMA 和
   DUCA schedule 只在成功更新后前进，但固定 epoch 循环不会补足跳过的更新。
2. DUCA 协议声明 `100 * 132 = 13200`，却没有强制训练结束时恰好完成
   13,200 次成功 optimizer update，也没有逐 batch 的同批重放闭环。
3. DUCA post-run validator 接受模糊的
   `same_workflow_best_or_final_declared`，没有把 checkpoint、原始预测、评估结果
   和 evaluator 的字节身份绑定成唯一证据链。
4. `balanced_binary_actionness_loss` 默认 prior=0.5，实际 `pos_weight=1`；当前生产
   调用未提供其他 prior，因此名称会误导读者，数值行为本质是普通 BCE。
5. counterfactual formal gate 只要求 sign agreement 和 Spearman 有限，没有要求
   正方向或预注册下置信界。
6. DUCA runner 使用固定 `MASTER_PORT=30471`，并发任务存在冲突风险。
7. S1 bootstrap 会拒绝缺少 overall/short 支持的重采样，实际抽样分布与文档中
   声明的普通 paired video-cluster bootstrap 不一致。
8. S1 使用 `torch.use_deterministic_algorithms(..., warn_only=True)`；full precheck
   没有包含 `test_train_engine_max_train_iters.py` 的 AMP 重放测试。
9. S1 完成 marker 与 sidecar 不是事务式提交；成本口径是 warm、serial、gross GPU
   energy，不能写成 cold-start、video-level 或 incremental deployment cost。
10. selected-axis remap 的代码内部一致，但 detector 的感受野和回归 ranges 并未
    因此自动获得 physical-time invariance。

## 3. 远端实时证据对审查的强化

核验时间：2026-07-15 20:10 +0800。

### 3.1 DUCA

正式 seed-0 四臂 Jobs `1164700-1164703` 均仍在运行且未发现 OOM、Traceback 或
non-finite loss collapse，但 schedule 日志已经证明成功更新数落后于 batch 暴露数：

| Arm | 已完成 epoch 末 | 理论更新数 | 实际 schedule step | 缺口 |
| --- | ---: | ---: | ---: | ---: |
| exact-uniform | 25 | 2600 | 2596 | 4 |
| direct-a5 | 25 | 2600 | 2597 | 3 |
| transition beta=0 | 25 | 2600 | 2596 | 4 |
| transition counterfactual | 24 | 2500 | 2496 | 4 |

因此这不是“终点可能不满足 13,200”的理论风险，而是已发生的协议失配。按当前固定
132 epoch 循环，这四个任务不可能恰好得到 13,200 次成功更新，不能作为 matched
formal table、C3 或 C4 的证据；即使最终 mAP 可读取，也只能作为诊断结果。

### 3.2 Spatial Zoom S1

Jobs `1164291`、`1164307-1164314` 均在运行，当前未见 fatal/OOM/non-finite/replay
exhaustion；已产生 213 个 checkpoint，并有 31 次 AMP 同批重放全部恢复成功。

但是九个 cell 全部出现
`upsample_linear1d_backward_out_cuda` 无确定性实现的 warning，共 221 次。由此：

- 直接把 `warn_only=False` 会使现有实现立即失败，不是完整修复；
- 当前九个任务可继续形成候选/诊断 artifact，但不能称严格确定性的 formal S1；
- sealed test 必须继续关闭；在替换或证明等价的确定性插值算子后，应重新运行 3x3。

## 4. 吸收裁决

### 4.1 完全认可的部分

- 认可 `STOP_AND_FIX`，并认可当前结果不能支持论文主张。
- 认可 DUCA 必须按成功 optimizer updates 审计，而不是按 epoch 或 loop 次数审计。
- 认可 checkpoint、prediction、result、evaluator 必须形成唯一且可哈希复核的证据链。
- 认可 S1 当前 bootstrap 口径不诚实、确定性封闭不完整、full precheck 漏测 AMP
  replay；这些都必须在开封 test 前修复。
- 认可 actionness loss 应按真实数学行为改名。
- 认可 counterfactual alignment 必须有正方向 claim gate，而不是只检查 finite。
- 认可固定端口、非事务式完成标记和成本口径属于需要修复或澄清的问题。
- 认可 S1 与 DUCA 必须作为两条独立研究路线叙述和裁决。

### 4.2 有条件认可或需要修改的部分

1. **DUCA AMP 修复不能只加 fail-on-skip。** 仓库的 S1 train engine 已有经过测试的
   state-exact same-batch replay 思路。优先复用同一机制：恢复 RNG 与所有 forward
   mutable state，保留 GradScaler backoff，只让 scheduler/EMA/schedule 前进一次，并在
   replay exhaustion 时 fail closed。若无法证明状态完整，才使用 fail-on-first-skip。
2. **terminal EMA 不是唯一合法 checkpoint 规则。** 预注册且冻结的 gate-best 规则也可
   审计；不过在当前 falsification pilot 中，terminal EMA 更保守。必须在重跑前只选一种，
   不能事后依据结果切换。
3. **Bayesian cluster bootstrap 只是候选方案。** 它通过正权重避免类别支持消失，却改变
   了推断目标，并可能弱化稀有短动作的支持不确定性。必须先做 synthetic coverage、
   parity 和 rare-class sensitivity 检查，再决定采用 Bayesian、诚实分层 cluster
   bootstrap，或支持感知的逐类区间。
4. **`warn_only=False` 不是 S1 修复本身。** 必须先替换为语义等价且确定性的 temporal
   interpolation，例如经数值等价测试的固定插值矩阵/matmul，再开启 strict mode。
5. **counterfactual LCB 阈值不是自然常数。** 正方向门槛是必要的，但具体 bootstrap、
   LCB 和阈值必须预注册并经机制测试校准，不能照抄审查文本后写成实验事实。

因此，对审查的回答是：**不完全照单全收，但高度认可其核心诊断和停止正式 claim 的
裁决；对若干具体补丁需要采用更严格、复用现有基础设施且先验证统计语义的实现。**

## 5. 状态与下一步

### DUCA

- 当前状态：`experiment_running`，但正式 seed-0 suite 已
  `protocol_invalidated_by_successful_update_deficit`。
- 当前 Jobs `1164700-1164703` 的任何结果只能进入 diagnostic appendix，不能进入主表。
- 下一次正式提交前必须完成：state-exact AMP replay、loader/update count assertion、唯一
  checkpoint rule、artifact SHA chain、positive CF gate、动态 c10d 端口，以及完整 exact-
  commit CUDA gate 和多 batch pilot。

### Spatial Zoom S1

- 当前状态：`experiment_running`，但严格 deterministic formal 资格已被 warning 破坏。
- 当前 3x3 可用于排查训练、成本和效应方向；sealed test 继续关闭。
- 下一次 formal 3x3 前必须完成：确定性插值替代、strict deterministic gate、full precheck
  纳入 AMP replay tests、bootstrap 口径冻结与 synthetic audit、事务式 artifact finalize，
  以及准确的 warm serial gross-cost 表述。

在上述修复和重跑完成前，DUCA 的 C3/C4 与 Spatial Zoom S1 的 GO/KILL 都保持
`unproven`，不得提升为 `empirically_supported` 或 `paper_ready`。

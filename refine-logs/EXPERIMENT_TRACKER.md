# PhysTime-TAD 实验执行台账（现行版）

| Run ID | 阶段 | 目的 | 变体 | 固定条件 | 成功证据 | 当前状态 |
|---|---|---|---|---|---|---|
| R00.1 | replay repair | 修 source-score dtype 传输合同 | frozen decode replay | 不训练；不改生产后处理 | focused tests、diff/compile | 进行中 |
| R00.2 | CPU preflight | 验证新 artifact 合同与环境 | 同 R00.1 | 新 commit/tree/snapshot | dtype、hash、fingerprint | 未启动 |
| R00.3 | CUDA micro-gate | 验证 capture 不改变 direct | S/P x online/EMA | 单真实窗口 | capture off/on ordered exact | 未启动 |
| R00.4 | CUDA native gate | 验证 replay 复刻 direct | S/P x online/EMA | source score dtype | 四条件 ordered exact | 未启动 |
| R00.5 | frozen replay | decode-axis 干预 | 4 checkpoint x U/P | P0 direct re-anchor 后 | completion、mAP | 未启动 |
| R00.6 | replay suite | 完整性与独立复算 | R00.5 全部 | 所有哈希/DAG 完整 | suite pass | 未启动 |
| R01 | decode-axis 分析 | 解释同 checkpoint U/P 差异 | R00.5 结果 | 不训练 | effect table | 未解锁 |
| R02 | Q192 factorization | 拆 decode 与 assignment | UU/UP/PU/PP | K384/J192/Q378 | matched pilot | 未解锁 |
| R03 | Q-density replay | 判断 Q 是否瓶颈 | no-training subcell | 不增观察 | oracle/pre-NMS coverage | 未解锁 |
| R04 | 统计与成本 | 复现和效率证据 | multi-seed/baselines/cost | 机制固定后 | CI、latency、memory | 未解锁 |
| R05 | 外部有效性 | 第二数据集 | protocol-audited minimum arms | raw-video contract | independent result | 未解锁 |

## 永久禁区

- 旧 `06a6734` token、snapshot 与 run root 不复用。
- R00 通过前不训练，不启动 Q-lift、Q384、插值、G1b、DUCA、动态采样、loss/NMS/schedule 改动。
- 任一 gate 失败时，正式 mAP 为 `NA`，下游不启动。

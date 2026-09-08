# 每五轮全测试集选模与调参：协议修订和部署

核查时间：2026-09-08 17:27:59 CST。本次是用户交互任务，不是补记心跳。

17:33:38 补充：五项正式作业仍为 RUNNING，H65 三组均已进入 epoch 1。
五份真实 protocol.json 的精确 SHA、seed、五轮日程、测试集选模标记和自哈希
均核对通过；两条路线均覆盖 211 视频、792 窗口。公共未分配 GPU 1/200，
账户 9 RUNNING / 2 PENDING，磁盘剩余 168.4 GiB。未生成新全测试集成绩。

17:39:58 补充：五项仍 RUNNING、未发现新增报错。ET-TRC ON/OFF 分别在
17:37:04/17:37:11 完成第 5 轮后实际触发了完整测试，尚未返回完整 mAP；
H65 三组均到 epoch 2（第 3 轮）。五轮评测已经实际执行，不是只有配置。

## 用户修订

用户明确要求“选最佳 checkpoint 且按测试集调参”。新实验采用 `TEST_GUIDED_EXPLORATORY_EVAL5`：

- 完成第 5、10、15、20、25、30、35、40、45、50、55、60 轮后，对完整 THUMOS14 测试集运行官方 evaluator。
- 按未四舍五入的 Avg-mAP 选择最佳 EMA，同分保留较早轮次；保存最佳 EMA 文件、指标与所有中间结果。
- 允许用测试集成绩决定后续超参数，逐次保存修改内容、SHA/配置、全部尝试及成绩。
- 独立保留第 60 轮（epoch 59）EMA 和其终态分数，继续核对实际 6000 次成功更新。
- 明确披露测试集参与调参/选模；不称独立未见测试集泛化或原论文协议公平比较。模型推理仍不能接收测试 GT。
- 不追溯修改旧实验，也不解除尚未实现机制的限制。此次尚无新测试成绩，未进行或宣称已完成结果驱动的超参数调整。

## 本轮正式作业

以下五项均已由 Slurm 返回有效 job，并在本次核查时处于 RUNNING，不只是 PRECHECK。

| 实验（完整名称） | Job | 种子 / GPU | 精确代码 | 状态 |
|---|---|---|---|---|
| Transformer 低秩残差近似开启：修复锚点后重训 | 1280117 | 4407 / 2 | [9a346f0d](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/9a346f0d71e8870ad499bb0a58b9f1824b8904c0) | RUNNING，尚无新全测试集分数 |
| 完整 Transformer 对照：相同双 GPU 与 FP32 训练 | 1280118 | 4407 / 2 | [9a346f0d](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/9a346f0d71e8870ad499bb0a58b9f1824b8904c0) | RUNNING，尚无新全测试集分数 |
| 均匀抽取 384 帧：整窗时间适配器匹配对照 | 1280127 | 3407 / 1 | [629162cd](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/629162cdddbe490d896899a067e9803e237df467) | RUNNING，尚无新全测试集分数 |
| 动态选帧但关闭四相分配：整窗时间适配器对照 | 1280128 | 3407 / 1 | [629162cd](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/629162cdddbe490d896899a067e9803e237df467) | RUNNING，尚无新全测试集分数 |
| 四相预算 128+64+64+128：整窗时间适配器实验 | 1280129 | 3407 / 1 | [629162cd](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/629162cdddbe490d896899a067e9803e237df467) | RUNNING，尚无新全测试集分数 |

## 代码与路径

### H65-Pro

本地：`E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission`  
分支：[codex/h65-pro-tia-eval5-20260908](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/h65-pro-tia-eval5-20260908)

- 均匀抽取 384 帧：整窗时间适配器匹配对照：`configs/adatad/thumos/h65_pro/h65_pro_eval5_uniform.py`；输出 `/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/TEST-UNIFORM_seed3407/gpu1_id0`；日志 `/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/logs/uniform_1280127.out/.err`。
- 动态选帧但关闭四相分配：整窗时间适配器对照：`configs/adatad/thumos/h65_pro/h65_pro_eval5_phaseoff.py`；输出 `/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/TEST-PHASEOFF_seed3407/gpu1_id0`；日志 `/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/logs/phaseoff_1280128.out/.err`。
- 四相预算 128+64+64+128：整窗时间适配器实验：`configs/adatad/thumos/h65_pro/h65_pro_eval5_phaseon.py`；输出 `/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/TEST-PHASEON_seed3407/gpu1_id0`；日志 `/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/logs/phaseon_1280129.out/.err`。
### ET-TRC

本地：`E:/DeskTop/TAD/_duca_fix_worktrees/ettrc_terminal_eval`  
分支：[codex/zoomtoken-ettrc-anchor-eval5-20260908](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-ettrc-anchor-eval5-20260908)

- Transformer 低秩残差近似开启：修复锚点后重训：`configs/adatad/thumos/ettrc_test_guided_on_seed4407.py`；输出 `/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/on/gpu2_id0`；日志 `/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/logs/on_1280117.out/.err`。
- 完整 Transformer 对照：相同双 GPU 与 FP32 训练：`configs/adatad/thumos/ettrc_test_guided_off_seed4407.py`；输出 `/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/off/gpu2_id0`；日志 `/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/logs/off_1280118.out/.err`。

## 已修复与验证

H65：恢复每视频整窗 TIA 聚合；选 384 帧的实际时间轴为 192 tubelets，全 768 帧为 384 tubelets。CUDA 测试确认跨小片段传输输出/梯度、不会跨视频污染。相位开关两组只改变 allocation，不再拿训练日程不同的 F01/F09 作单因素比较。均匀组关闭学习选帧监督，但保持检测器和优化器配置匹配；它仍不是官方原版 AdaTAD 的复现替代品。本地 36 passed / 12 Windows Torch skipped；远端精确干净 SHA 的 51 项测试、三组配置 PRECHECK 由 1280125 全部通过。

ET-TRC：在 TIA 前屏蔽 anchor 位置的低秩补偿，防止非中心时间卷积 tap 污染精确 anchor 残差；仍是 fixed-stride 的共享可学习低秩代理，并非精确 Jacobian 或真实事件门。远端 37 项测试通过；1280077/1280078 完成真实 VideoMAE 预训练加载、两 GPU 训练与推理。实际 full-test loader 覆盖 211 视频、792 窗口。新 OFF/ON 均 global batch 2、FP32；目前已进入 epoch 1。首轮全测试集指标尚未生成。

评测保存/恢复训练 RNG 和各模块模式；每次使用当前 EMA 重新推理，不读 raw-prediction cache。最优指标为 `test_guided/best_test_metrics.json`，最优 EMA 为 `test_guided/best_test.pth`，每轮分数为 `test_guided/epoch_XX/metrics.json`。真实训练终态另存 `checkpoint/epoch_59.pth`。

## 仍未完成

- CT-DP：已完成 G0/G1 结果不重训；后续新矩阵尚需接入本次五轮测试协议。
- BAFDR：padding 路由、真实终态 screen 和专用训练器的五轮评测接入尚未完成。
- Evidence-Recovery：utility 学习路径、真实 robust/cycle 仍待实现；不能以 refiner 初始化修复代表整体完成。
- DUCA-Unified：真实 Taylor P0/P1、合法 one-swap、H65 retention/transition 仍是实现阻塞。

未取消或修改历史 BAFDR 1267920/1267921。五项新作业不代表六路线全矩阵已完成。每 30 分钟心跳已更新为收集新协议下的完整曲线、最佳与终态分数，以及调参尝试；失败继续先读日志、独立修复和新身份重提。

结构化记录：[29_TEST_GUIDED_EVAL5_DEPLOYMENT_20260908.json](29_TEST_GUIDED_EVAL5_DEPLOYMENT_20260908.json)。

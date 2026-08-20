---
type: wiki_log
append_only: true
---

# Research Wiki Log

- 2026-07-11：初始化 C3/DUCA research-wiki。
- 2026-07-11：逐轮读取主任务 191 轮，归档 158 条用户侧原始消息。
- 2026-07-11：登记实现代理、论文代理和早期目标任务的近期记录。
- 2026-07-11：登记 C3、PAction、GAS-VT、lattice、detector-aware、TrueTime、
  DUCA、MUST、X3D/SlowFast、physical-grid、CFPA、CVCR、ChronoTransport、
  PhysTime 路线。
- 2026-07-11：冻结当前裁决：70aa069 是待裁决 DUCA baseline，a5e1774 是最新
  审计代码；正式论文 claim 尚未闭环。
- 2026-07-11：wiki lint 通过：16 ideas、7 experiments、10 claims、47 edges、
  0 orphan nodes、0 curated broken links；query pack 2825 chars。
- 2026-07-11：纠正 ChronoTransport 过期状态：`92029ea` formal Stage-B P3 science gate 为负，Stage C/P5 未解锁；新增独立 negative experiment 节点，路线暂停。
- 2026-07-12：为无法读取本地工作区的 Pro reviewer 建立 GitHub 固定提交审查入口；仅同步
  ChronoTransport r1 规格、实现表面、原 Pro 记录、两轮独立复核与本地源码审计，明确排除
  数据、checkpoint、GPU 日志和新行为结果。审查仍止于 `REVISE_SPEC_BEFORE_PLAN`，不得借
  GitHub 同步越过到实现、profiling、Gate 1、新 seed 或 Stage C。

- 2026-08-19：批准 DUCA-FoveaSampler / Query-Bridge 最终实现契约（`docs/superpowers/specs/2026-08-19-duca-foveasampler-query-bridge-design.md`，commit `9affb525`）；保留手工三分支，不再删除或合并。
- 2026-08-19：在 `codex/duca-fovea-query-bridge-20260819` 实现 FoveaScout、QueryBank/LightQueryDecoder、FoveaHeads（saliency/boundary/uncertainty 三分支）、FoveatedSampler（Gumbel-TopK 训练 / 贪心 MMR 推理、exact-K）、fovea losses、FoveaQueryBridgeFrameSelector、ActionFormer 后置 cycle hook；提交 `0975aac3`..`a32a374`。
- 2026-08-19：远端 CPU focused tests `tests/test_fovea_query_bridge.py` 11/11 通过；完整模型在远端 CPU 可构建（pretrained 路径由 launcher 注入）。CPU 端到端 forward 的临时探针被中断且无 durable 记录，因此不计为通过。
- 2026-08-19：GPU one-step gate `1244839` 已提交（pending，30min）；正式 Fovea-QB 开发矩阵尚未提交，不能声称 Fovea-QB 实验已在完整训练。
- 2026-08-19：旧矩阵 `1244133`（7529fba6 wv1）仍在完整训练：`_0`..`_7` 已 COMPLETED（exit 0:0），`_8`..`_14` RUNNING；远程 N16R4 GPU 分配未受影响。

- 2026-08-19：GPU one-step gate 经过 6D dense 输入、Byte 插值、Gumbel 索引排序等修复后通过：`1244850` COMPLETED exit 0，单步 Loss=1.1862，selector 五类 loss 与 cls/reg 均有限。
- 2026-08-19：Fovea-QB 开发矩阵第一波 `1244851`（arms 0-4：baseline_fused / query_only / query_gt_mask / query_cycle / query_fovea，seed 3407）已提交并 RUNNING，run root `duca_fovea_qb_4ae50671_dev_20260820T014200Z`；受 MaxSubmit=16 限制，query_fovea_dpp 与 full 两臂待旧矩阵释放槽位后提交。

- 2026-08-20：Fovea-QB 第一波 `1244851` 全部 5/5 完成（train 60 epoch + epoch_59 test，exit 0:0，单 cell 约 9.9–10.7h）。THUMOS14 val（seed 3407）：query_cycle 54.67 > query_gt_mask 49.16 > query_only 45.26 > query_fovea 43.77 > baseline_fused 42.94；query_cycle 高 IoU 也最好（0.6: 46.33 / 0.7: 31.63）。旧矩阵 `1244133` 15/15 完成。
- 2026-08-20：Fovea-QB 尚无同提交 exact-uniform/random/dense 对照，且 `full` / `query_fovea_dpp` 未跑，不能据此宣布 Fovea-QB 路线优于或劣于 DUCA 旧路线；仅记录为 development-matrix 原始结果。

- 2026-08-20：接收独立 Pro 科学审查，唯一建议 REVISE：终止当前 UVT/Fovea 直接选择实现为主路线，新建 `DUCA-SQB-Block-DK-v1`（语义 Query-Bridge + 连续 16-frame 物理块 + 确定性 fixed-K 归因 + 通过后 dynamic-K）。我方初步结论：认可核心方向；UVT 已由实际负结果封存，Fovea 仅保留诊断与可复用组件，不再作为主方法；最终是否启动新 clean cycle 待用户裁决。

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
- 2026-07-15：纠正活动任务为 Spatial Zoom-only。S1 保持完整时间轴，只比较 matched
  dense160/224/256；DUCA 与时序选帧不属于此执行线。旧 `35204f5` 3x3 任务因每臂均出现
  `upsample_linear1d_backward_out_cuda` nondeterminism warning 而失效并取消，222 个
  checkpoint/sidecar 仅保留为诊断，无 sealed-test、最终 mAP、成本或 GO/KILL。
- 2026-07-15：在 `codex/spatial-zoom-s1-audit-fix-20260715` 实现 exact-2x deterministic
  temporal interpolation、formal strict determinism、真实 full-model AMP backward precheck、
  paired Bayesian video-cluster bootstrap、test-open crash recovery、原子证据发布和成本口径
  限定。本地 S1/train-iteration tests 为 `40 passed, 1 skipped`，required C3 regression 为
  `20 passed`；
  远端 replacement CUDA gate 与 3x3 matrix 尚未产生结果。

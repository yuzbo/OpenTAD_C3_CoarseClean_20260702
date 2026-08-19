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
- 2026-08-19：DUCA-UVT 规格已获用户批准并完成首版实现，分支
  `codex/duca-uvt-utility-value-20260819@9459c515`；新增单头 signed V(t) 残差、
  GT geometry target、value-head self-EMA、四 token Query Cross-Attention portal（门控关闭）、
  boundary-foveated exact-K decoder + greedy MMR；`V_off` 分数级与解码器级 legacy 等价测试通过。
- 2026-08-19：本地聚焦测试 5/5 通过；远程干净运行时
  `/data/run01/sczc063/yuzibo/runtime/duca-uvt-9459c515/source` 已部署，
  远程 CPU 容器测试 5/5 通过，既有 `test_duca_dynamic_variable_compute.py` 8/8 通过；
  `bash -n`、`git diff --check`、三种 value_mode 配置解析均通过。
- 2026-08-19：发展种子数组（off/geo/geo_ema, seed=3407）准备提交；`sbatch --test-only`
  当前被 AssocMaxSubmitJobLimit 阻止，原因是不修改的现役矩阵 `1244133`（15 task）仍在占满
  `MaxSubmitJobs=16` 配额；待其任一 task 完成后重试。现役 `1244133` 未做任何修改。
- 2026-08-19：DUCA-UVT 四层有限差分 portal gate 与 CUDA one-step gate 尚未独立产证；
  portal 保持关闭，首轮三臂仅作为诊断性发展种子，不形成 mAP/efficiency/paper claim。
- 2026-08-19：为不触碰现役 `1244133`，已注册远程 tmux 轮询器
  `duca_uvt_submit_44adb` 与锁文件；当用户待处理作业数为 0 且总作业数 ≤13 时自动
  `sbatch --parsable` 提交 DUCA-UVT 发展种子数组（commit `44adb917`），并写
  `duca_uvt_official_44adb917.submit_manifest.json`。提交前会再次校验远端 source
  干净且 HEAD 匹配。

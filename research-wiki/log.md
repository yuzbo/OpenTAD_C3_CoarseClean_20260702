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
- 2026-07-12：完整归档 GitHub-visible Pro 终审，附件 SHA-256
  `07A5B4B519E64A39D7F84CE862F0E56117BFF2DB62206B6AE24BDD66768B19FE`，裁决
  `REVISE_SPEC_BEFORE_PLAN` / `GO_TO_SPEC_REVISION_ONLY`。本地独立复算确认 exposure、
  conformal rank、coverage 下界和 Stage-C 525/candidate；官方上游确认 GT-aware
  `random_trunc`、edge padding 与 all-time adapter。不同意原样照抄三处未闭合文本：split
  digest 字节协议、Gate-3 coverage margin 和 Gate-4 seed-level mAP bootstrap 必须在 r2
  中精确定义。当前状态为 `spec_revision_in_progress`，尚未实现或部署。
- 2026-07-12：完成 ChronoTransport r2 单文件书面规格 commit `d825520`，853 行，
  committed-blob 与 worktree 双重 SHA-256 均为
  `2551DC68F2FE94A204BAF722E8FC60143FD0D77B6024979F32EBC65BE4F69912`。完整吸收十项
  amendments，并额外冻结 split digest 字节协议、Gate-3 coverage margin 与 Gate-4
  seed-level mAP bootstrap。状态为 `written_spec_pending_spec_only_review`；尚未授权实现。
- 2026-07-12：空白上下文独立 agent 首轮 spec-only review 发现唯一 P1：Gate-4 detector-regret
  bootstrap 未定义 official-video sample unit。按 exact replacement 修复为单文件 commit
  `e4422f5`，870 行、47,546 bytes，SHA-256
  `87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8`；复审最终返回
  `APPROVE_SPEC_FOR_PLAN`。状态升级为 `spec_approved`，只解锁 writing-plans，尚未实现。
- 2026-07-12: Created the executable CT-P3R-3S-r2 implementation plan after the independent
  `APPROVE_SPEC_FOR_PLAN` verdict. The plan preserves remote-only behavioral verification,
  implementation/registration commit separation, and the Gate-1-first hard stop chain. Status is
  `implementation_planned`; no new behavior is yet claimed implemented or tested by this entry.
- 2026-07-12: ChronoTransport r2 implementation batch 1 reached partial `tested` status on the remote
  CPU environment: protocol tests 7/7 and candidate/control/cache plus legacy-core tests 36/36. This
  does not unlock a Gate or scientific claim; runtime and later protocol surfaces remain in progress.
- 2026-07-12: ChronoTransport r2 runtime and window-risk implementation reached partial `tested`:
  runtime/integration 35/35 and risk/core 30/30 on the remote CPU environment. The implementation now
  uses all-row AdaTAD adapter writeback, detached historical cache with live current rows, distinct
  requested/executed ledgers, and the fixed D=23 mean/max window head. No Gate has run.
- 2026-07-12: Added and remotely tested Gate-1/Gate-2 pure adjudication (4/4) and Stage-C
  object-identity/loss-specific AMP primitives (4/4). These are synthetic implementation checks;
  no formal data was opened and no Gate result exists.
- 2026-07-12: Added registration/claim-chain primitives, Gate-1 CLI, r2 config overlays and a launcher
  requiring exact GPU1 plus Slurm allocation. Remote registration tests 4/4 and `bash -n` passed. The
  current SSH session has no active allocation, so no formal GPU experiment was started.

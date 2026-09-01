# ARIS Raw Session — 2026-08-14

本文件保存本会话的固定身份、精确 prompt 摘要、pinned revision、命令回执与 raw evidence
路径，供复现与 Pro 批评核验。

## 1. 固定身份

- 会话根：`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`
- pinned revision：`a6bdc084cc145c80b6b2c68d0a38f0deea3e8518`
- branch：`codex/duca-total60-plugin-cvpr-20260727`
- 本机：Windows，Python 3.11.7（C:\ProgramData\anaconda3\python.exe），numpy 1.23.5；
  torch 因 `c10.dll` WinError 1114 无法加载（已知，RTK.md 已记录）。
- 远端：`ssh N16R4`（ssh.cn-zhongwei-1.paracloud.com，user sczc063@BSCC-N16R4），
  OpenTAD env `/data/run01/sczc063/yuzibo/conda_envs/opentad`，`/usr/bin/sbatch` 存在，
  THUMOS annotations 齐备。

## 2. 精确 prompt（用户指令要点）

> 你是 DUCA 项目唯一 ARIS Code 主进程 / Executor / First Author / 证据裁决执行者。继续
> 会话根与已读材料。固定基线 revision a6bdc084cc145c80b6b2c68d0a38f0deea3e8518，保留 dirty
> worktree。遵循 CPR：C 锁可证伪主路线，P 落地最小实现+精确命令，R 只跑一次最便宜批准
> falsifier；旧 P0 identity/optimality gate 按原始授权与 PRE_RUN 边界处理，不可执行则继续
> 不依赖它的本地准备。讨论预算：一次 DeepSeek 提案、一次独立 Pro 攻击、最多一次 DeepSeek
> 修订；一次路线型 probe 后 full train；第二次等价科学失败终止并采用已记录 fallback。
> 持久化 docs/aris/ARIS_CPR_PLAN-2026-08-14.md 与 ARIS_DECISION_LOG-2026-08-14.md；保存
> 原始 transcript/prompt/pinned revision/命令回执；同步 wiki 与 log；每 completion 输出
> ARIS_STATUS 单行 JSON。

（完整原文以本会话 system/user message 为准，此处为压缩要点。）

## 3. 命令回执（关键）

- `git rev-parse HEAD` → `a6bdc084cc145c80b6b2c68d0a38f0deea3e8518`
- `git rev-parse --abbrev-ref HEAD` → `codex/duca-total60-plugin-cvpr-20260727`
- 本机 decoder 自检（importlib 直载，全部冻结 fixture 通过）：
  `768/384 u[-1]=767`、`G16-U/G17-E2/EINF/E1/U1/PLEX/G31-U/G32-U/F768-U/G767-U/G385-X 全 True`、
  负例 code（U_LENGTH_MISMATCH / A_LENGTH_MISMATCH / A_ENDPOINT_MISMATCH /
  U_CANONICAL_MISMATCH / A_ORDER_MISMATCH / INFEASIBLE）全对、
  `nonconstant 768 decode len 384 time_ms 40.0 geom True`。
- `python -m py_compile opentad/models/duca/density_decode.py tests/test_duca_density_decode.py opentad/models/duca/__init__.py` → `PY_COMPILE_OK`
- 远端（scp 模块 + OpenTAD python）：`remote 768/384 u[-1]=767 / G17-E2 True / nonconstant len 384 time_ms 78.0 geom True`
- 环境探针：`ssh N16R4` → `SSH_OK sczc063 ln01`；`which sbatch` → `/usr/bin/sbatch`；
  `ls thumos14/annotations` → category_idx.txt / thumos_14_anno.json / tad_{train,test}_videos.txt 等。

## 4. 本轮落地文件（workspace-write，未触碰用户既有改动）

- `opentad/models/duca/density_decode.py`（新增，纯 Python 整数解码器/投影）
- `tests/test_duca_density_decode.py`（新增，冻结 fixture + brute-force reference）
- `opentad/models/duca/__init__.py`（追加导出新符号，其余不动）
- `docs/aris/ARIS_CPR_PLAN-2026-08-14.md`（新增）
- `docs/aris/ARIS_DECISION_LOG-2026-08-14.md`（新增）
- `docs/aris/ARIS_PRO_HANDOFF_PACKET-2026-08-14.md`（新增）
- `tmp/dd_probe_runner.py`（临时探针，可删）

## 5. Raw evidence 路径（远端，未改动）

- 冻结 uniform checkpoint 收据：
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_selected_axis_cb89586_resume_e24_v4_20260721_135701/runs/exact_uniform/terminal_evaluation.json`
  （checkpoint_sha256 `17d7461e...`，config_sha256 `9edb24fa...`，average_mAP `0.6445799769`）
- decode-cross 四条件：见 `research-wiki/experiments/phystime-frozen-decode-cross-replay.md`
- S0 负结果：`research-wiki/experiments/actionformer-sparsehead-official-main-table-prereg-20260729.md`

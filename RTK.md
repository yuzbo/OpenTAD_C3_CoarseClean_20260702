# RTK Project Rules

本文件是当前纯净仓库的简短上下文锚点。它只记录当前 C3 粗分类 + OpenTAD 路线所需规则，不承载历史 wiki、旧日志或旧实验追踪。

## 当前目标

最终研究目标是任务感知动态时序采集：根据视频、窗口、动作区域和难度动态分配帧/片段/Token，减少持续时序冗余，把更有用的信息送入 TAD 检测器，并保护高 IoU 定位性能。

当前仓库的阶段性目标是固定预算 C3 控制锚点：

- 用低成本粗分类模型估计动作/背景概率 `p_action`；
- 将 `p_action` 转换为严格的 value-transport 帧选择 ledger；
- 将 384/768 选择输入接入 OpenTAD/AdaTAD；
- 验证输入侧粗分类采集是否能在无测试 GT、无测试 teacher、无 raw-prediction shortcut 的协议下支撑检测。

固定 384/768 或 50% 输入只是归因、安全门和失败诊断锚点，不是最终动态采集目标。

## 仓库范围

本仓库只应包含：

- `opentad/` OpenTAD 主库与当前 C3 接入代码；
- `configs/` 中当前 C3 路线所需配置与最小 base config；
- `tools/bata/` 中当前 coarse probe、model matrix、ledger conversion、validator；
- `scripts/` 中当前 N16R4 GPU1 启动器/ watcher；
- focused `tests/`；
- `README.md`、`AGENTS.md`、本 `RTK.md`。

不要加入历史 `research-wiki/`、旧 tracker、旧 server logs、生成图、检查点、数据集、压缩包、bundle、临时 worktree 或旧路线报告。

## 协议规则

- 不允许 validation/test GT 参与测试时选择。
- 不允许 validation/test teacher leakage。
- 不允许 hidden raw-prediction cache shortcut。
- ledger 若用于 deployable selection，必须记录 no-GT/no-teacher/no-oracle/no-raw-prediction/no-checkpoint flags。
- C3 主线优化默认使用物理 GPU1；GPU0 保留给发散创新实验，除非用户同轮明确覆盖。
- 不在 N16R4 登录节点直接训练；正式训练使用 Slurm 或已授权保护分配。
- 发起 Pro 讨论时，默认共享本机 Chrome `--remote-debugging-port=9223` 实例；Chrome 只开一个，例如 `chrome.exe --remote-debugging-port=9223 --user-data-dir=<shared-profile>`。共享端口的核心是共享 DevTools 入口，但不共享同时控制权。
- 共享 Chrome 9223 时必须加全局调度锁，默认锁文件为当前仓库 `.codex/chrome-9223.lock`；同一时间只允许一个 agent 操作页面。
- agent 操作前先抢锁；拿到锁后调用 `http://127.0.0.1:9223/json/list` 找到或新建自己的页面，并在锁内容中记录 `owner`、`pid`、`targetId`、`pageId`、`webSocketDebuggerUrl`、`url`、`startedAt`、`expiresAt`。
- agent 只操作锁中绑定的 Chrome DevTools `targetId`/`pageId`，不要临时切到“当前激活页”或其他未绑定页面。
- 操作完成后必须释放锁；若 agent 崩溃，锁必须有 TTL/heartbeat，过期后允许下一个 agent 清理并重新抢锁。
- 共享端口与共享 profile 仍不如独立端口和独立 profile 稳定；需要并行、高可靠或隔离状态时，优先为每个 agent 使用独立 Chrome 端口和独立 `--user-data-dir`。

## N16R4 环境

### SSH 身份与入口

N16R4 的三个标识不可混用：

- 公网 SSH 网关：`ssh.cn-zhongwei-1.paracloud.com`；
- ParaCloud 登录身份：`sczc063@BSCC-N16R4`；
- 登录后的 Slurm `ClusterName`：`n16r4`。

本机 `C:/Users/skywalker/.ssh/config` 登记 `N16R4` 与 `BSCC-N16R4`
别名后，标准入口为 `ssh N16R4`。若别名缺失，等价的显式只读探针为：

```text
ssh -p 22 -l sczc063@BSCC-N16R4 ssh.cn-zhongwei-1.paracloud.com \
  "hostname; whoami; command -v sbatch; scontrol show config"
```

不得把 `N16R4`/`n16r4` 直接当作可由公网 DNS 解析的服务器名，也不得把
`BSCC-N16R4` 误写成 Slurm 集群名。连接诊断顺序固定为：先用 `ssh -G
N16R4` 核验本地展开，再做只读身份探针；只有远端返回预期用户、`sbatch`
路径和 `ClusterName=n16r4` 后，才允许执行后续 Slurm 操作。
Windows OpenSSH 还要求该配置文件使用收紧的 ACL；权限不得宽于现有
`id_rsa` 的 `Administrators/SYSTEM/skywalker` 范围。若出现 `Bad owner or
permissions`，先修复本机 ACL，再判断网络或远端状态。

远端写入边界：

```bash
~/run/yuzibo
/data/run01/sczc063/yuzibo
```

默认环境：

```bash
BASE=/data/run01/sczc063/yuzibo
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"

export HOME="$BASE/tmp/home"
export XDG_CACHE_HOME="$BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$BASE/tmp/xdg_config"
export HF_HOME="$BASE/hf_cache"
```

### Slurm shell bootstrap

- A standalone Slurm script must execute `source /etc/profile` **before**
  `set -u` and before every `module load`. Slurm batch shells are
  non-interactive and do not otherwise define `module`.
- A failure at this point is a zero-update launch-environment error. Repair
  the one wrapper line and resubmit the exact same model commit; do not create
  a new model revision, new gate, or a performance conclusion.

需要下载外部资源时使用登录节点代理：

```bash
export http_proxy='http://u-MtfrT7:vH5orjDV@10.244.6.36:3128'
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
```

THUMOS14 默认路径：

```bash
$BASE/thumos14/annotations/thumos_14_anno.json
$BASE/thumos14/annotations/category_idx.txt
$BASE/raw/Validation Data/validation
$BASE/raw/Test Data/TH14_test_set_mp4
```

## 常用检查

本地轻量检查：

```bash
python -m py_compile tools/train.py tools/test.py tools/bata/train_lowres_action_probe.py tools/bata/c3_coarse_classifier_model_matrix.py
python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q
```

远端启动前：

```bash
PRECHECK_ONLY=1 bash scripts/run_c3_asformer_delta_ledger_adatad_full_train_gpu1.sh
```

当前本机 Windows Python 的 user-site `torch` 可能加载 `c10.dll` 失败；完整 Torch 相关测试优先在 N16R4 OpenTAD 环境中验证。

## R0-R5 生产交付合同（2026-07-22）

本轮唯一验收目标是：R0-R5 生产代码全部完成，真实后端门禁通过，所有正式实验具有可运行配置并实际提交 Slurm，同时完整记录 Job、依赖、精确提交、run root、manifest/hash 与预期产物路径。缺少任一项都不得称为“完整实现”或“完整部署”。

- 性能实验优先。实现、测试和成本统计可并行准备；上游证据仅通过 `afterok` 和 fail-closed consumer 约束下游实际运行。
- 模型实现优先于启动工程。启动器只保留直接运行、依赖与结果路径所需的最小逻辑；禁止扩张通用编排、复杂 schema/journal/router。时间优先投入粗分类证据、边界微簇、真实硬换帧、检测梯度回传与真实检测后端。
- R0 必须给出同 K/G 的 exact-uniform、R2Q3、R4Q5、unrestricted Oracle 冻结 official detector mAP、逐视频 paired bootstrap、sealed summary 和唯一 family 裁决。
- R1 必须落实真实数据/模型 identity、无泄漏、原子提交 journal、哈希封存和 fail-closed 消费链。
- R2 必须使用 R0 唯一选中 family 完成真实 coarse/transition/burst P0 训练与 holdout 裁决。
- R3 必须在同一精确协议下部署 matched U 与 G0 的 official TAD 正式训练。
- R4 必须实现真实合法 hard-swap、冻结 official detector signed utility/alignment 门禁，并部署真实 G1/G2。
- R5 必须覆盖三种子、K384/K256、official AdaTAD、仓库中真实可训练的第二 TAD backend，以及完整端到端成本测量。
- mock、sentinel、占位 backend、仅 `PRECHECK_ONLY`、TODO、未接真实 loader/model/CUDA/backend 的入口均是半成品。
- “已部署”仅在 `sbatch` 返回有效 Job ID，并记录 dependency、exact commit、run root、manifest/hash 和预期终端产物后成立。
- 并行智能体的文字结论不算交付；代码必须合入唯一分支、通过 focused/contract/Linux/真实 CUDA 门禁，形成一个精确提交后才可部署。
- 独立 MAX 审计只在完整关键合并版本执行一次；禁止把每个小补丁扩成新一轮门禁。Wiki 只在提交、Job、错误、mAP、成本或裁决发生变化时更新。
- 全部 R0-R5 生产实现和真实后端门禁完成后、任何新的正式 Slurm 实验 DAG 部署前，启动一个全新、无实现上下文的独立 MAX 审阅智能体；只有审阅通过才允许部署，禁止把审阅拖到结果收割之后。审阅重心是模型设计与机理是否忠于最初目标，真实梯度归属、硬选帧行为与训练推理合同是否正确；工程问题只在改变模型行为、实验真实性或可复现性时阻断，不得纠缠无关细节。
- 禁止修改或借用 SparseHead、Spatial-Zoom、ChronoTransport 路线；禁止重新创建同义 selector、decoder、worktree。

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

## ChronoTransport 动态特征刷新并行路线

ChronoTransport 与 C3/DUCA 并行存在，不做 pre-backbone 删帧。v1 在 48 个 16-frame clip × layer-group 上调度 VideoMAE heavy attention/MLP，保持 patch embedding、AdaTAD temporal adapter、384→768 后处理和 detector head dense。TRANSPORT 必须从 latest cache 递推；正式 learned scheduler 必须使用按硬件、精度、batch、schedule 形状与 selected rows 实测的 p50/p95 cost lookup。

Stage A/B 的 dense reference 与 counterfactual branch 必须同 batch、同增广、同 RNG；ledger 只能保存 compact signal、schedule、cost 与 regret label，不能在推理时查询。所有 deploy、metric、latency 与 paper claim 默认关闭，直到三种子 kill gate 通过。

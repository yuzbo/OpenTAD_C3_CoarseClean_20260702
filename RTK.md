# RTK Project Rules

## Research priority

The first obligation is to improve the scientific model: a stronger, more
novel, and empirically defensible offline-TAD algorithm. Engineering rigor is
required only to the extent that it makes the model, comparison, and claim
auditable. Do not substitute a more elaborate framework, provenance layer, or
visualization for a model improvement; do not spend material research time on
infrastructure that cannot change a scientific decision. Conversely, do not make
an efficiency, novelty, or paper claim without the minimum matched evidence
that can falsify it.

## 论文实验推进优先

ZoomToken 由 Pro 冻结唯一科学任务与裁决门，Codex 只做最小忠实实现和授权实验，Critic/Evaluator 独立且有限范围。论文实验推进优先，基础设施失败只做最小修复。每个正式终态或客观 blocker 后必须进入新的 Project Pro 复盘；失败路线先做有界、证据化根因分析，复盘同轮给出唯一下一任务，不得停在纯 STOP 或无限审计。具体角色边界见 `docs/aris/ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md`。

当证据形成真实科学选择、缺失信息或来源冲突时，Codex 可主动提交充分权威上下文并请求 Pro 独立建议、裁决或下一任务；不得预置偏好路线、默认方案、期望裁决或穷尽候选，必须允许 Pro 拒绝当前 framing、提出未列替代并独立决定方向。

本文件是当前仓库的简短上下文锚点。详细研究记忆以 `research-wiki/` 为单一事实源。

## 当前目标

最终研究目标是任务感知动态时序采集：根据视频、窗口、动作区域和难度动态分配帧/片段/Token，减少持续时序冗余，把更有用的信息送入 TAD 检测器，并保护高 IoU 定位性能。

当前科学状态：

- C3/PAction/GAS-VT/lattice 保留为固定预算、no-leak、归因与失败诊断基线；
- 当前主路线 BPNS-R1 只用当前观测，在 VideoMAE 前选择连续 `8×8/K64` 原生支持，并让 K64 完整执行全部主干和 Adapter；它不使用 cache、carry 或深度跳过；
- 同源 K100 与 R1 的 seed-42 final-EMA 为 `68.51/61.19/46.27` 与 `69.07/61.14/46.57`（Avg-mAP/mAP@0.6/mAP@0.7）。这支持单种子准确率可行性，不证明泛化或真实效率；
- R1 减少 36% 原生空间输入是结构事实。旧同硬件回放 job `1257281` 因 K100 原始精度值与舍入口径未正确绑定而在首个 pass 后终止；最小数值合同修正 `e9323448…` 已通过独立准入，唯一替代 job `1258299` 正在执行完整八 pass 回放。终态产物形成前仍没有 R1 延迟、显存或能耗结果；
- DUCA、RC32 carry、当前 APM 载体和若干直接缓存/深度路线保留为历史候选与负证据，不得复活为当前主方法；
- ChronoTransport 与 PhysTime 是独立并行假设，不得与 BPNS-R1 或历史 DUCA 证据混用。

固定 384/768 或 50% 输入只是归因、安全门和失败诊断锚点，不是最终动态采集目标。

## 仓库范围

本仓库只应包含：

- `opentad/` OpenTAD 主库与当前 C3 接入代码；
- `configs/` 中当前 C3 路线所需配置与最小 base config；
- `tools/bata/` 中当前 coarse probe、model matrix、ledger conversion、validator；
- `scripts/` 中当前 N16R4 Slurm GPU 启动器/ watcher；
- focused `tests/`；
- `research-wiki/` 当前研究记忆；
- `README.md`、`AGENTS.md`、本 `RTK.md`。

不要加入旧 tracker、旧 server logs、生成图、检查点、数据集、压缩包、bundle、临时 worktree 或旧路线报告。

## 科学证据与运行边界

- 开始工作前必须读 `research-wiki/query_pack.md` 与 `research-wiki/anti_repetition.md`。
- 科学决策、重要负结果、有效实验结果和论文主张变化写入相应 wiki 节点；原始历史只追加，不以文档润色改变既有证据。长期记忆保持精简，不保存浏览器、队列或一般协调流量。
- 如使用 Project Sources，只保留当前目标/边界、当前研究状态和重要实验历史的最新聚合材料；当轮代码差异、专用报告与临时审查作为本轮输入，不制造版本堆积。
- 不允许 validation/test GT 参与测试时选择。
- 不允许 validation/test teacher leakage。
- 不允许 hidden raw-prediction cache shortcut。
- ledger 若用于 deployable selection，必须记录 no-GT/no-teacher/no-oracle/no-raw-prediction/no-checkpoint flags。
- GPU 任务必须使用 Slurm 正常分配的设备；不得固定物理索引或覆盖 Slurm 的 `CUDA_VISIBLE_DEVICES`。单卡任务在进程内使用 `cuda:0`。
- 历史文件名中残留的 `gpu0`/`gpu1` 只代表旧协议，不得直接复用；再次运行前必须改成正常 Slurm 映射并重新门禁。
- 不在 N16R4 登录节点直接训练；正式训练使用 Slurm 或已授权保护分配。

## 共享 AdaTAD 官方基线（跨项目唯一）

- ZoomToken 是原始 AdaTAD 官方基线的**唯一执行负责人**。所有相关 TAD 项目只能只读引用
  `docs/aris/ADATAD_SHARED_OFFICIAL_BASELINE_PACKET-2026-08-17.md` 及其最终 durable receipt，
  不得各自重复 released-checkpoint evaluation 或从头训练。
- 共享运行必须固定 clean official revision、未改原始 config、canonical THUMOS14 411、预训练或
  released checkpoint、seed、evaluator/NMS、EMA/final 选择、运行时身份和唯一结果根；receipt
  缺任一绑定即不构成共享 baseline。
- 先且只先评测可验证的 released checkpoint；仅在 checkpoint 确实不可得且同一负责人确认需要
  reproduction 时，才执行一次 clean untouched official training。`66.42/67.14/65.99` 是
  matched-source dense，永远不得冒充官方复现。
- 等待共享 dense 数字不能让各项目全面停工：ZoomToken 可继续已接受方法的最小实现、独立审查、
  launcher、checkpoint 恢复与 PRE_RUN 准备；共享数值在 receipt 到位前仅是待绑定输入，不得
  触发方法质量或论文 claim。

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

远端 GitHub 同步的固定规则：

- 每次在 N16R4 执行 `git clone`、`git fetch`、`git pull`、`git
  ls-remote` 或下载 GitHub release，必须先设置上面的学术加速代理四个
  环境变量；不得先尝试公网直连，也不得在直连失败后才临时切换代理。
- 同步必须先用代理解析远端 branch/commit，再通过同一代理 clone/fetch；
  完成后验证 `git rev-parse HEAD` 等于预期完整 SHA、对应 remote-tracking ref
  指向同一 SHA，且 `git status --porcelain` 为空。
- 学术加速节点不可用时同步 fail closed，保留诊断并重试代理；不得用未提交的
  rsync/scp 源码覆盖来伪造 GitHub 同步。Git bundle 只能传递已经推送且可由
  commit/ref 验证的对象，并在最终代理同步成功后删除。

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

ChronoTransport 与 BPNS-R1 相互独立，不做 pre-backbone 删帧。v1 在 48 个 16-frame clip × layer-group 上调度 VideoMAE heavy attention/MLP，保持 patch embedding、AdaTAD temporal adapter、384→768 后处理和 detector head dense。TRANSPORT 必须从 latest cache 递推；正式 learned scheduler 必须使用按硬件、精度、batch、schedule 形状与 selected rows 实测的 p50/p95 cost lookup。

Stage A/B 的 dense reference 与 counterfactual branch 必须同 batch、同增广、同 RNG；ledger 只能保存 compact signal、schedule、cost 与 regret label，不能在推理时查询。所有 deploy、metric、latency 与 paper claim 默认关闭，直到三种子 kill gate 通过。

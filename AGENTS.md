@RTK.md

## User checkpoint cleanup authorization (2026-09-09)

After the stop order, the user authorized deleting historical checkpoints and
retaining only the last usable checkpoint per independent experiment/run/seed.
This overrides checkpoint-retention requirements only for the explicitly owned
roots in 50_CHECKPOINT_CLEANUP_SCOPE_20260909.json. CPU deserialization and
finite-parameter checks, file cleanup, and maintenance records are authorized;
training, GPU evaluation, retries and supervision remain stopped. Keep the
retained checkpoint intact with its EMA/optimizer state and required metadata.
Keep logs, configurations and metric receipts. Earlier best-test weight copies
may be removed under this authorization; disclose their removal without
rewriting their historical measured scores. Other tasks, protected historical
BAFDR runs and external pretrained models remain outside the cleanup scope.

## User stop order (2026-09-09, effective immediately)

The user has terminated all experiments owned by this six-route supervisory
task. Do not launch, resume, retry, evaluate, diagnose on GPU, or restart its
remote supervisor unless the user explicitly authorizes resumption after this
order. This supersedes earlier deployment, automatic repair and heartbeat
instructions for this task. Preserve all jobs, logs, checkpoints and results.
Do not cancel or adopt jobs owned by other tasks, including historical BAFDR
1267920/1267921. See 49_USER_STOP_20260909.md/JSON in the audit directory.

Remote supervisor PID 1914589 was terminated at 2026-09-09 13:52:44 CST and
verified absent after 70 seconds. No owned Slurm job was active or pending.
The local heartbeat opentad-c3-duca still has ACTIVE configuration; its
management tool is unavailable in this session, so disabling the automation
has not been confirmed. If a stale heartbeat fires, honor this stop order,
do not contact the cluster or restart experiments, and disable the heartbeat
using automation_update when that tool becomes available.

## OpenTAD six-route protocol amendment (2026-09-08)

The user's latest instruction explicitly authorizes selecting the best checkpoint
and tuning hyperparameters using the THUMOS14 test set. For new six-route runs,
evaluate the full test set after completed epochs 5, 10, ..., 60; select EMA by
unrounded official Avg-mAP, keeping the earliest epoch on ties. Label these runs
TEST_GUIDED_EXPLORATORY_EVAL5. Preserve epoch-59 EMA and its separate terminal
score, the full learning curve, and every tuning attempt with its changed
parameters, source/config identity and observed test scores. Never recast a
test-selected score as unseen-test generalization or a fair official-paper
comparison. Keep test GT out of the model's inference inputs. This supersedes
older no-test-tuning/no-best-selection rules only for these newly disclosed
six-route experiments; historical sealed runs retain their original protocol.
The successful-update, failure-repair, exact-source and artifact-retention rules
remain in force. Do not use this amendment to unlock unimplemented mechanisms.

# Repository Instructions

这是当前 C3 粗分类路线的纯净 OpenTAD 仓库。保持仓库小而可运行：不要加入历史 `research-wiki/`、旧 tracker、服务器日志、生成图、检查点、数据集、压缩包或旧路线报告。

## Objective

当前目标是用低成本粗分类模型产生可部署 `p_action` 信号，构造严格的 value-transport 帧选择 ledger，并把 384/768 选择输入接到 OpenTAD/AdaTAD 检测器。固定预算阶段是安全与归因锚点；最终目标仍是动态、任务感知的时序采集，并保护高 IoU TAD 定位。

## Scope

允许维护的主要表面：

- `opentad/`
- `configs/adatad/thumos/*c3*` 及其最小 base config
- `tools/bata/` 中的 C3 probe、ledger、validator 工具
- `scripts/` 中的 C3/N16R4 启动器
- focused `tests/`

新增内容应服务当前粗分类/ledger/AdaTAD 接入路线。不要把协调根里的历史 worktree、wiki、log 或临时产物搬进来。

## Remote Rules

远端写入边界是 `~/run/yuzibo` / `/data/run01/sczc063/yuzibo`。默认环境：

```bash
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
```

C3 主线优化默认使用物理 GPU1；GPU1 子启动器必须在 `CUDA_VISIBLE_DEVICES=1` 时才继续。不要在登录节点直接训练。

## Verification

改动后至少跑 focused checks：

```bash
python -m py_compile tools/train.py tools/bata/train_lowres_action_probe.py
python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q
```

远端训练前先跑对应 `PRECHECK_ONLY=1` 启动器或 validator。

## OpenTAD 六路线：性能不达标时的 Pro 讨论授权

2026-09-08 用户明确授权：后续本任务负责的 OpenTAD 实验性能不达标时，可以使用 ixBrowser 中的 ChatGPT Pro 讨论路线、实现和实验，分析结果并定位失败原因，无需为该用途重复请求许可。范围包括 H65-Pro、CT-DP、DUCA-Unified、BAFDR、ET-TRC、Evidence-Recovery，以及对应的基线、消融与修复实验；不缩减已有其他路线的授权。

- 必须通过 Computer Use 操作 ixBrowser 中的 ChatGPT Pro，确认实际浏览器和所选 Pro 模型。此用途覆盖 RTK.md 的默认 Chrome 讨论入口；不得静默改用 API、直接 HTTP 请求、其他浏览器或其他模型。
- 以已确认的科研目标、有效评测和协议匹配的基线判断是否不达标；不得擅自新增分数门槛，不把中期低分、排队或未完成评测当作最终性能失败。
- 讨论应提供相关精确 SHA、代码与配置、官方基线身份、训练更新数、真实 stdout/stderr、评测指标及收据，明确哪些是已证实错误、未实现机制或待验证假设。仅发送必要材料，不上传 SSH 私钥、令牌、代理凭证或无关私密资料。
- 记录讨论时间、页面或对话链接（可获得时）、关键结论、定位依据、采纳或拒绝建议的理由，以及后续修改和验证结果，关联回对应实验目录记录。
- Pro 建议不等于验收。采纳前核对实际源码和原设计；修改继续遵守独立 codex/ 修复分支、本地 focused tests、远端 exact-SHA clean tests、必要 CUDA 与对应 PRECHECK、新命名空间重提及保留旧产物的规则。不得为追分偷换官方基线、数据划分或训练预算；新六路线实验按上方用户修订允许测试集调参和最佳 EMA 选择，必须如实披露。
- 如果 Computer Use 无法操作 ixBrowser、账户未登录或 Pro 不可用，报告并记录具体阻塞，继续可独立推进的工作；不得声称已经咨询，也不得绕过指定入口。

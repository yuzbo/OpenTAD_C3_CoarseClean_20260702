# DUCA 统一全矩阵执行包

任务标识：`DUCA-UNIFIED-FULLMATRIX-v001-20260902`

这个包把三份 H65-Pro 路线裁决整合为一个不可再分叉的实验合同：

1. 首先以 `Prior × Allocation` 分离“恢复 ASFormer 语义先验”和“新相位分配机制”的贡献；
2. 在同一矩阵中测试 curvature、fixed/adaptive quota、physical-time、feature Taylor、Mixture-of-Depths 和课程调度；
3. 所有开发臂和预登记确认种子一次性实现并提交，不根据中间结果动态生成新臂；
4. 正式论文主比较使用 strict total 6000 successful updates，历史 H65 与 dense AdaTAD 只读引用；
5. 主精度方法保持 Dense VideoMAE-S，physical-time、Taylor、MoD 分别作为表示、监督和效率扩展。

## 包内文件

- `DUCA_UNIFIED_FULL_MATRIX_AGENT_COMMAND.md`：完整 Agent 科学与执行命令；
- `duca_unified_matrix_manifest.yaml`：17 个唯一开发 arm、确认 arm、种子、机制和判据；
- `run_duca_unified_matrix_agent.sh`：单命令调用 Codex CLI；
- `agent_output_schema.json`：要求 Agent 返回 commit、审查、Job ID、run root 和 blocker；
- `SHA256SUMS`：文件完整性。

## 一条命令启动完整实现与部署

在 Linux、WSL 或能够访问仓库和既有 N16R4 授权连接的受控终端中：

```bash
unzip duca_unified_matrix_bundle.zip
cd duca_unified_matrix_bundle

bash run_duca_unified_matrix_agent.sh \
  --repo /absolute/path/to/OpenTAD_C3_CoarseClean_20260702 \
  --remote-base /data/run01/sczc063/yuzibo \
  --max-concurrent 8 \
  --submit
```

也可以使用 tar 包：

```bash
tar -xzf duca_unified_matrix_bundle.tar.gz
cd duca_unified_matrix_bundle
bash run_duca_unified_matrix_agent.sh \
  --repo /absolute/path/to/OpenTAD_C3_CoarseClean_20260702 \
  --remote-base /data/run01/sczc063/yuzibo \
  --max-concurrent 8 \
  --submit
```

`--submit` 是显式部署授权：Agent 会在代码、独立 Critic 和独立 Evaluator 通过后，调用其新实现的 `scripts/duca_unified_fullmatrix/submit_all.sh` 一次性提交完整 DAG。它不表示可以绕过现有凭据、Slurm 配额或仓库安全规则。

不加 `--submit` 时，Agent 仍须完成所有实现、配置、测试与独立核验，并把唯一剩余远端命令写入结构化结果。

## 可选模型选择

脚本默认使用 Codex CLI 当前配置的模型。只有确实需要覆盖时使用：

```bash
bash run_duca_unified_matrix_agent.sh \
  --repo /absolute/path/to/repo \
  --model '<available-codex-model>' \
  --submit
```

不要在脚本、仓库或日志中写入 API key。脚本复用既有 Codex CLI 登录；远端连接复用既有授权配置。

## 固定矩阵规模

- 开发：17 arms × seed 3407；
- 确认：8 arms × seeds 4407/5407/6407；
- 完整 train+terminal-eval：41 GPU array tasks；
- 成本：5 GPU array tasks；
- bootstrap：16 CPU array tasks，合计 10,000 hierarchical seed+video draws；
- 一个 GPU preflight；
- 一个 afterok finalizer；
- 一个 afterany audit。

## 科学主比较

```text
A11 − A10
```

A10 和 A11 使用同一 ASFormer semantic prior；唯一主变量是 legacy dual-phase allocation 与 robust signed phase + bounded adaptive allocation。这样不会把从 motion prior 恢复 semantic prior 的收益误写成 phase field 收益。

## 结果何时算存在

- Agent 返回真实 `sbatch` Job ID，才算已部署；
- 41 个任务形成 terminal epoch-59 `state_dict_ema`、211/211 predictions 和官方指标，才算有检测点估计；
- 10,000 次统计完整合并，才有预登记区间；
- 负结果必须保留；PENDING、launch error、路径错误和测试通过都不是方法结果。

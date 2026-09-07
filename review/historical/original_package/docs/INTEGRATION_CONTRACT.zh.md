# 仓库接入、不可变执行与adapter契约

交付包提供的是论文/模型/实验规范、可运行的矩阵编译器与本地队列，以及参考原语。完整VideoMAE/TIA/TAD训练实现由agents在用户真实仓库完成。`geosparse_ext.entry`是要求实现的新入口，不是声称用户仓库已有此命令。

## 1. 开始时环境清单

记录repo路径与commit、branch dirty状态、CUDA/PyTorch/driver、GPU型号与显存、数据raw-video/frames格式、PTS是否存在、官方split、预训练权重SHA、现有DUCA代码与配置、TIA full-time/clip-local作用域。

没有读取到真实资产不得标ready。禁止自动删除/覆盖用户数据，禁止自动下载未经授权的数据或租用付费GPU。

## 2. 每个agent隔离开发

使用不同worktree/branch，避免共享文件互相覆盖；共有contracts由Agent00拥有。样例（需真实git仓库）：

```bash
git worktree add ../gs-route-a -b geosparse/route-a
git worktree add ../gs-route-b -b geosparse/route-b
git worktree add ../gs-route-c -b geosparse/route-c
```

不要用`git reset --hard`、强推、覆盖其他agent分支。

各agent通过固定contracts开发，可使用明确mock做单测，但不得从mock生成科学result。合入integration时运行contract tests；capability receipt包含commit、tests路径与所支持variant集合。

## 3. 新的统一入口

必须实现：

```bash
python -m geosparse_ext.entry --job /ABS/run/job.json \
  --output /ABS/run --bindings /ABS/configs/bindings.local.json
```

该入口根据kind dispatch train/evaluate/benchmark/diagnostic，读取job_id而不是人为重新输入一堆超参。它必须检测不支持的variant并exit78；不能默认退化成主配置完成任务。资源不足exit非零并记录OOM。

模板stub位于tools/adapter_stub.py，只会拒绝运行，不生成假结果。

## 4. 不可变代码与配置

job启动时读取capability_receipts中的commit，将其和common/router/route源码锁定到只读snapshot/worktree。已开始run不得因为其他agentmerge而更换code。entry模块作为bootstrap必须保证其实际训练imports来自该snapshot，记录snapshot SHA。protocol文件、训练split、weights、实际参数全部hash。

bindings可更新新就绪资产路径，但改变数据split或数学配置必须protocol amendment、新work_root和manifest版本。不同snapshot包含同语义优化时也保留receipt，不覆盖旧运行。

## 5. capability示例

在capability_dir分别写common.json、route_A.json、router.json等（每agent只写自己文件，原子rename）：

```json
{"ready": true,
 "commit": "REAL_COMMIT_SHA",
 "test_receipt": "/ABS/tests/A_correctness.json",
 "supported_variants": ["READ_THE_ACTUAL_MATRIX"],
 "protocol_version": "geosparse-v1.0"}
```

如果仅支持部分variant，入口必须在执行前check所需variant，不支持exit78，修复后使用`--retry-blocked`重新入队。ready不能由mAP值决定。

## 6. result receipt

真实train完成后：

```json
{
  "job_id": "tr-REAL_MANIFEST_ID",
  "status": "completed",
  "is_mock": false,
  "source_commit": "REAL_COMMIT",
  "resolved_config_sha256": "REAL_SHA",
  "split_sha256": "REAL_SHA",
  "completed_epochs": 60,
  "checkpoint_path": "/ABS/real_checkpoint.pth",
  "metrics_path": "/ABS/metrics.json"
}
```

evaluate/benchmark/diagnostic使用同一provenance基础字段并附真实结果路径。没有encoded-video计时等子测量可以measurement_status=NA，但不能将其标数值0。异常不写completed。

## 7. runner范围与限制

`tools/dispatch.py`可在单机明确GPU slots上执行DAG；默认仅plan；必须`--execute`才运行。`--watch`只关注资源/能力/自身checkpoint就绪，不关注精度。

该脚本不提供真实agent spawning，不连接远程GPU集群，不替代Slurm/Kubernetes。多节点由平台agent实现cluster adapter并保持相同manifest/receipt。基准隔离只约束该dispatcher拥有的任务；外部用户进程仍需硬件agent检测并标测量污染，禁止杀死非本项目进程。

worker运行中dispatcher中断只终止自己启动的进程组；下次以自身checkpoint恢复。留下RUNNING状态而进程所有权不明时拒绝自动重复发起，先审计PID/输出。

## 8. 小型公共测试先决条件

测试可以在所有路线并行开发过程中进行：layout、pairing、PTSunion、PEgather、actor sign、PL概率、parent isolation、dense-limit、packed等价、null证据、空GT/尾部mask、冻层梯度、无重复写回。

这些测试属于运行正确性，不是“先做Oracle看有效才继续”的科研门槛。

## 9. capability所有权

common：00/01；route_DENSE/COARSE/DUCA/DEPTH/BCR：01；route_A/B/C：02/03/04；router：05；roi/acquisition_round2：07；evaluation/diagnostics：08；benchmark/hardware_lut：09。一个capability应指向经过集成并锁定的真实源码提交，而非相互不兼容的多个工作树。LUT是无标签算子/shape微测量产物，不能依赖其他模型达到精度目标。

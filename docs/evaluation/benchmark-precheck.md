# 独立硬件路径预检（2026-09-08）

冻结模型 M=b70ae056 的训练 GPU 预检只认证模型及评价能力，没有生产 `benchmark` 能力收据的入口。因此，正式硬件任务即使等到训练完成，仍会停在 `BLOCKED_CAPABILITY`。

`review/launchers/benchmark_capability_precheck.py` 补充这一入口。它在独立 Slurm allocation 中加载已完成60轮、全部选点完整的同一 EMA best，使用原生产代码核验 checkpoint identity、CUDA/NVML UUID、GPU隔离、reference/bucket同计划输出，并执行 batch1 的 device、decoded、encoded 三种边界。每种实现/边界只做1次预热、2次执行，结果标为非科学结果；六条路径全部通过后才为该训练 ID 写入 `benchmark.json`。

这个能力只证明硬件路径能够执行，不证明 batch8/32 能装入显存，也不形成可发表的延迟。正式登记的 batch1/8/32、50次预热、200次重复、三边界/两实现完全不变，真实 OOM/失败仍必须保留。同卡训练期间不得计时。预检使用训练保存的 `resolved_opentad.py`，并与 JSON provenance 逐项核对；不能直接用 JSON 重建配置，因为它会把 Resize 必须使用的 tuple 变成 list。

部署 helper 是外部执行目录的运维工具，依赖该目录现有 `deploy_corrected.command`（仅SSH传输函数）、`audit_deployment.latest.json` 和真实 bindings。先将这三个新脚本复制到该目录；不要从审查仓库直接运行旧部署器的 main：

```bash
python -m unittest test_benchmark_capability_precheck -v
python deploy_benchmark_precheck.py --cluster N16R4 --train-id tr-0afe8e4fc09b
python deploy_benchmark_precheck.py --cluster N16R4 --train-id tr-0afe8e4fc09b --execute
```

重复调用报告已记录的提交，不重复 sbatch。训练已完成时以其真实完成/选点凭证作为产物依赖；不再依赖可能已从 Slurm 控制器清除的旧完成作业 ID。N16继续保留物理GPU1→CUDA0检查及已有节点UUID约束。新脚本与运维状态都在冻结模型 checkout 外，运行源码不变。

本地5项收据及配置回归测试已通过。首次真实GPU预检在正确的4090 UUID上发现上述tuple读取问题，失败证据保留在远端 `control/benchmark_capability_v1_20260908`；修订脚本独立保存在 `benchmark_capability_v2_20260908`。v2的完整CUDA结果在实际通过前保持PENDING，不以CPU测试或已提交状态代替。正式科学图只消费原 benchmark 的完整硬件结果，不消费预检的 sanity durations。

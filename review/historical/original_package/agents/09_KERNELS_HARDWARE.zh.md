# 09_KERNELS_HARDWARE｜真实稀疏执行与硬件

你拥有geosparse_ext/ops与benchmark。
为A/B/C做selected QKV/MLP、parent分组、bucket/varlen、有效padding、gather/scatter实现；先reference数值/梯度一致，再标优化可用。不能把mask或dense MLP算成token比例省费。
所有benchmark包含batch1/8/32、50warmup/200repeats、100分层固定真实视频、p50/p95、throughput、peak allocated/reserved。device-only/decoded-input/encoded-video三种路径分开，解码不存在则NA。
测量时排除同卡外部负载；发现其他用户进程不杀，只记污染/排队。此隔离是物理必要条件，不是串行科研门槛。
记录PE/TIA/Scout/route/crop/packing/heavy/head/postprocess成本，实际形状与MAC口径；同精度同kernel对待baseline。训练probe/反事实额外成本另表。
C coarse成本和A dense TIA不能遗漏。对所有seed0注册benchmark运行完整batch菜单；OOM标OOM不偷偷改resolution。
交付optimized ops、equivalence receipt、硬件原始sample、profiler traces和cost LUT（版本固定）。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。

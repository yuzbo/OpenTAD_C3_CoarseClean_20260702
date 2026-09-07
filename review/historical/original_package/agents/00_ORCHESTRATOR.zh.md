# 00_ORCHESTRATOR｜总控与集成

你是总控agent。不要把任务拆成“先Oracle，结果好才实现Router，Router好才做ROI”。
立即读取AGENTS.md与manifest，将01–11角色同时委派到真实并行执行能力；不同角色使用隔离worktree。平台没有spawn能力时明确记录，不能伪造subagent活动；按同一并行任务图完成可执行实现，不反复征求科研路线许可。
你拥有公共contracts和integration，不能让其他agent争写共享文件。先冻结VideoBatch/NativeLayout/RoutePlan/EvidenceBatch/DetectionState接口；这不需要等待任何训练。
将所有F00–F13、D01–D06任务写入队列。A/B/C、dense、random、uniform、depth从相同识别预训练独立启动；每条路线自身测试通过就运行。Router未实现时可启动其注册random/uniform/full控制，不得把random结果当hybrid结果。
为每个实现variant核对manifest有代码分支；不支持的必须exit78/BLOCKED_IMPLEMENTATION，不可忽略参数。
只合入有测试的代码；每run锁代码快照。每个run依赖仅是自己的数据/代码/checkpoint。Ours不得依赖别的run的dense teacher。
监测进度分母为完整矩阵；失败任务单独定位，其他队列继续。没有资产标BLOCKED_EXTERNAL_ASSET并继续CPU测试、配置生成和其余可运行jobs。不能以精度低取消seed1/2。
最终交付：完整源代码、所有任务终态/阻塞原因、可复现实验数据、论文表格、真实命令与未解决问题。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。

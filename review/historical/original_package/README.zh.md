# GeoSparse-TAD：论文、实现与全矩阵并行执行包

版本1.0，2026-09-06。

## 交付边界

这是供coding/research agents接入真实仓库的完整研究规格与执行基础设施。提供了：论文框架、A/B/C实现契约、12份agent指令、去重的全量实验manifest、本地多GPU调度器、参考采样/支持原语和单元测试。

**不是已经训练好的TAD模型，也不是声称已接入你的私有代码/GPU。** 本轮仅实际执行工具级单元测试、manifest编译和plan-only检查。完整`geosparse_ext.entry`与各模型由agents依规范在真实仓库实现。工具级测试不能代替模型正确性测试。

## 阅读顺序

1. `docs/PAPER_FRAMEWORK.zh.md`：论文标题、无虚构结果摘要、假设/贡献/方法/图表。
2. `docs/MODEL_SPEC.zh.md`：模型数学含义、接口、默认参数、训练估计器。
3. `docs/EXPERIMENT_PROTOCOL.zh.md`：F00–F13与D01–D06、评价/Oracle/硬件/统计。
4. `AGENTS.md` 与 `agents/00_ORCHESTRATOR.zh.md`：总任务。
5. `agents/01_...` 至 `11_...`：并行角色任务。
6. `manifests/experiments.jsonl`：实际任务ID、配置、seed、kind、技术依赖。
7. `docs/INTEGRATION_CONTRACT.zh.md`：真实仓库入口与输出receipt。

## 执行命令

在当前包根目录：

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
python tools/dispatch.py \
  --manifest manifests/experiments.jsonl \
  --bindings configs/bindings.example.json
```

最后一条默认plan-only，输出`launched: 0`。example中的路径是占位符，不是发现的实际资产。

由环境/总控agent自动盘点用户已授权仓库、数据、识别预训练和GPU，生成真实`bindings.local.json`，并让实现agent提交真实capabilities、实现`geosparse_ext.entry`后：

```bash
python tools/dispatch.py \
  --manifest manifests/experiments.jsonl \
  --bindings configs/bindings.local.json \
  --execute --watch
```

修复entry返回78的不支持variant后，可停止本dispatcher并以`--retry-blocked`重新运行；失败checkpoint恢复由adapter实现，需`--retry-failed`显式入队。

该脚本单机本地调度；不会真实spawn其他agents。将12份任务指令交给当前平台**真实**的并行agent能力。远程集群需平台真实cluster adapter，不能用虚构API。

## 完整矩阵

去重后的精确数量见`manifests/summary.json`。每个配置3seed、60epochs。所有训练没有其他实验结果依赖。eval/benchmark/diagnostic只等待自己的checkpoint；资源不足保持排队或显式blocked。

当前注册：609次train、609次evaluate、203组hardware benchmark、117组diagnostic；共1538个任务。其中203个配置×dataset组合各3seed训练，累计36540个训练epoch（这不是GPU-hours估计）。硬件以seed0、每组batch1/8/32、多个计时路径测试。缺资源条目仍是全量矩阵的一部分。

这个规模是全矩阵设计，不隐含你拥有足够GPU立即同时启动。调度只使用明确声明的已有资源；不得自动租赁算力。不得为了更快完成而静默减少epoch、seed或实验族。

## 论文主线与并行原则

A主路线：native dense carrier+规则TIA+稀疏重型更新。
B挑战：dense cheap query+sparse packet/ROI evidence+物理支持receiver。
C挑战：mixed-scale tokens+原坐标回写+规则TIA。

三路线同时落实。禁止“先验证A/Oracle成功再开始B/C/Router”的科学门槛；保留实现正确性、真实资源和自身产物的必要依赖。公开失败、负结果、OOM、缺少slice标注和不可用资产，不编造结果。

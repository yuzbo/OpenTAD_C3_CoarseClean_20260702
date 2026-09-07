# 06_GEOMETRY_TESTS｜时间几何与正确性审计

你拥有geosparse_ext/geometry、tests/invariants与support generator。
实现PTS秒坐标、native配对、original PE gather、parent mask、support union、ROI原图映射和有效时间mask。观察包络和实际支持必须分开。
实现reference-vs-optimized、K0/Kall、梯度穿冻层、parent隔离、permutation、tail、emptyGT、VFR等测试；测试真实模型而非只对toy模型通过就标route ready。
为D02生成共享selection manifests，uniform/cluster/gap不同非均匀程度配对；rank-only/centroid/hull作为明确负对照。retiming同步变换视频与标签，metadata corruption单独命名。
测试不依赖mAP阈值；发现bug通知对应agent并让其他路线继续。不得全局阻塞训练。
交付每路线test receipt、可复现失败样例、坐标单位与grid说明、geometry variant配置。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。

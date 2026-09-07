# 01_DATA_BASELINES｜数据与基线

你拥有geosparse_ext/data、baselines和base-protocol审计。
读取实际AdaTAD/DUCA/OpenTAD源码、训练配置与预训练来源。记录原parent clip、native tubelet、TIA placement/full-time或clip-local、PE与384→768映射。不得根据论文图直接重写原模型。
实现F00 dense/cheap、F02 uniform/random/CDF-native/whole-clip/legacy DUCA、F03静态4/6/8/10层和明确标记modified的teacher-free PBD-inspired、BCR前缀0/2/4。没有legacy DUCA文件就标BLOCKED，不用CDF-native冒充。
数据loader保存源PTS、目标规则时间、source ids、尾部mask、原图ROI映射。完成internal_dev/hash split；官方held-out只最终评估。
基线完整60epochs，三seed。数据预处理与source解码缓存规则对所有方法一致。缺FineAction/large weights时单列阻塞，不影响THUMOS/ActivityNet。
交付：base audit JSON、split manifests/hashes、真实source adapter、baseline configs/tests和运行receipt。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。

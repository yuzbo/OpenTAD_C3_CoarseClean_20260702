# 07_ROI_BUDGET｜ROI、动态预算与二轮扩展

你拥有geosparse_ext/acquisition与budget扩展，使用A/B/C公开接口，不直接改它们私有文件。
实现F06、F09、F11：固定quota/global/K0、controller/threshold、hardware LUT cost；原图ROI112/160/224和1/2ROI、tubelet内固定、跨tubelet平滑、fallback与低清放大负对照；一次/两次总预算相同。
ROI来自原分辨率源像素并保存变换；crop224并不比full224少tokens。成本按所有ROI、重叠去重、rounds、decode、router逐项计算。
Budget阈值只internal_dev校准；不归一化掉跨video绝对收益。histogram打乱只改变预算，位置同score重选。
二轮proposal来自当前cheap detector，不借GT；pair两端同packet不可重复计费。实现cache合法性接口，不先验承诺cache。
这些变体与主模型并行开发/训练，不等待A论文成立。交付configs、合法性测试、cost规则和完整manifest coverage。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。

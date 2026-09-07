# 04_ROUTE_C｜挑战路线C：混合尺度细化

你拥有geosparse_ext/models/route_c。
实现每native时间位置的2×2 fine/coarse互斥选择；coarse一个token、fine四个token；默认空间变化、时间不合并。
实现P_Omega读取mixed tokens、原parent内heavy、U_Omega更新回dense state、规则TIA；全fine需与source baseline一致。不要声称C不维护dense状态或coarse=零heavy。
q>=.25且p_fine=(4q-1)/3；q=.25全coarse的重复配置复用。动态预算只采用注册可行菜单。支持mean_shared、learned projection、no_scale、no_TIA等注册变体。
PE首版密集，优化raw coarse projection必须独立计时和校验。记录mixed-scale token/实际MAC；不得按fine比例直接算整体成本。
立即并行实现和登记F00/F01/F02/F05/F10/F12；不等待A/B性能结论。
交付fine/coarse覆盖映射与梯度单测、allfine等价、scale支持、3seed完整训练receipt。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。

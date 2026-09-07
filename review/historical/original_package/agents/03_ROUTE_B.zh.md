# 03_ROUTE_B｜挑战路线B：规则Query+稀疏Evidence

你拥有geosparse_ext/models/route_b与receiver，不修改A。
实现cheap dense query、native packet VideoMAE、S=1/4/8 slots、真实source support集合、五种receiver。默认2层width256/heads4和null update，query384/768均支持。
rank_interp是负/简单控制；physical_interp、concat、timestamp_only、support_attention都要完整训练。null删除需从softmax分母和上游交互中移除，不只是value乘0。
heavy TIA packet-local/none与receiver后规则TIA明确区分；B-full是新架构税控制，不声称数值等价原AdaTAD。
实现F01、F04–F11、F13全部B变体，和07约定ROI与二轮接口；共享header冻结后可独立开发stub输入，但正式结果必须真实证据。
不能把任意远帧重新配成两帧tubelet；不连续packet保存union不是hull。缓存只在D06证明独立时使用。
交付代码、全部receiver tests、无证据旁路、time permutation、ROI coords、实测cross-attention成本与完整jobs。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。

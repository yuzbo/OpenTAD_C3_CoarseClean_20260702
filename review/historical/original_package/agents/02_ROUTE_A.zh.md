# 02_ROUTE_A｜主路线A：规则状态+稀疏重更新

你拥有geosparse_ext/models/route_a。
按MODEL_SPEC实现native dense token底座、真实selected MHSA/MLP更新和原scope的TIA。逐源码计算边界hook，保持LN/residual/drop-path顺序，不擅自把原block改成ViT后接TIA。
selection atom默认native tubelet×2×2 spatial patches；T-only选择整幅空间。不同parent clips不能合成全局attention。route在各层共享，另支持注册each-layer变体。
K_j=0旁路重层但保留状态；K=all等价原模型；frozen heavy仍允许梯度传回adapter。
支持F01、F04–F08、F13全部A变体，包括故意rank-TIA负对照（必须单独配置、不能污染主路）、局部quota和zero规则。
使用固定接口和reference selector立即开发；不等待Router科研有效性。自己的dense-limit/packing/empty tests通过后，A-full/random/uniform可先进入已注册队列，hybrid在router能力就绪后独立运行。
交付逐层shape trace、dense-limit output+gradient测试、K0/Kall、parent isolation、native pairing、compiled variants列表。性能低也完成三seed矩阵。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。

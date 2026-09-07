# 05_ROUTER_TRAINING｜任务价值路由和联合训练

你拥有geosparse_ext/routing、losses和training hooks。
实现MODEL_SPEC的轻Scout、gain head、预算head、可执行hard采样PG，以及PG+每32batch真实acquisition校准。借助reference/routing.py核对PL有序log_prob，但不能假定参考文件就是完整trainer。
采样顺序与原生执行顺序分离；global budget先选K再选位置的joint logprob必须正确。critic baseline不能依赖本次随机action。cost作为需要最小化的量，actor loss符号有解析单测。
对随机探索分支不写假的learned-policy logprob。epoch0–5随机receiver训练，6–20比例.5→.1，21–59保持.1；完整60epoch一次训练，无独立dense teacher。
实现PG-only、retention-gradient、zero-gate probe、ST有偏估计器、探索和probe间隔全部F08变体；记额外forward和训练成本。所有TAD loss normalizer与基线相同。
A/C反事实必须重跑依赖路径；B缓存需合法。把acquisition、retention、swap分头输出/标识，不混成“真utility”。
同时给A/B/C相同契约，必要时用小tensor做单测；不等待其他agent的高精度checkpoint。交付gradient tests、采样概率验证、预算可行性和variant支持表。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。

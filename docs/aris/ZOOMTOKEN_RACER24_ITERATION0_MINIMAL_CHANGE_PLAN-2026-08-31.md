# ZoomToken-RACER24 Iteration-0 Minimal Change Plan

## 1. 绑定与边界

- 计划生成时间：`2026-08-31T01:11:01+08:00`
- 用户转交裁决摄取：`docs/aris/ZOOMTOKEN_RACER24_USER_SUPPLIED_PRO_DECISION_INTAKE-2026-08-31.md`
- proposed base：`2d945e64bdccd09ae2e2916524562e3f388c5a2a`
- proposed branch：`codex/zoomtoken-racer24-v001`
- 当前协调分支：`codex/zoomtoken-cvpr2027`，HEAD `1ea16b77c4535100610331907d9851335f97ae46`

当前协调分支不是 proposed base 的 descendant，且含研究记忆修改与用户历史未跟踪文档。实现前必须使用独立、干净、精确绑定 `2d945e64...` 的 Builder worktree；不得在协调分支上混写模型代码，也不得触碰无关未跟踪材料。

本计划只覆盖 Iteration 0：最小实现、focused parity/no-new-parameter tests 和一个 matched real-shape block microbenchmark。它不授权数据集运行、Slurm、60-epoch训练、正式成本回放、FARM24 或 PairLatent32。

## 2. 现有代码事实

1. `opentad/models/backbones/vit_adapter.py:749-829` 的 `Attention.forward(self, x)` 从同一输入生成 Q/K/V，没有 selected-Q/full-KV 接口。
2. `Block._packed_attention_mlp_forward`（约 `:914-939`）和 `_ragged_attention_mlp_forward`（约 `:941-990`）均在 selected tensor 上执行 self-attention 与 MLP；其 K/V 不是 full context。
3. `Block.forward_native_ragged`（约 `:992-1031`）的 Adapter 只处理 ragged selected carrier，不满足 RACER 恢复 dense 512 carrier 后让既有 Adapter处理全部 token 的合同。
4. 普通 dense block loop 位于 `VisionTransformerAdapter.forward`（约 `:1933-1943`）；`Block` 已有的 dense Adapter 语义位于约 `:1058-1064`。
5. 仓库中没有 RACER router、parameter-free residual completion 或 selected-Q/full-KV helper。它们必须作为 task-specific 最小机制新增，不能用改名冒充复用。

## 3. 精确最小文件面

### 3.1 修改 `opentad/models/backbones/vit_adapter.py`

在 `Attention` 增加唯一新接口：

```python
forward_selected_query_full_kv(query_x, kv_x)
```

接口分别从 `query_x` 生成 Q、从 `kv_x` 生成 K/V，输出只含 selected-Q 行；现有 `forward(x)` 完全不变。它不得注册新 module 或 parameter。

在 `Block` 增加 task-specific RACER forward：

- 输入始终是 dense `[B, 512, C]` carrier；
- 每个 tubelet 在 native 64 spatial token 中精确选择 24，八个 tubelet共 192 query；
- attention 为 `Q=192` 对 `K/V=512`；MLP 只更新 192 selected token；
- router 只读紧邻 dense block 的 pre-Adapter residual relative magnitude 与 adjacent-tubelet residual surprise；route score/index stop-gradient，并以 native spatial index 稳定打破并列；
- 未选 token 以当前 selected residual、当前 key 与前一 dense residual做 parameter-free completion；不得新增 projection、MLP、learnable scale 或 cache；
- selected exact residual与 completion residual scatter 回原生位置，显式恢复 dense `[B,512,C]`；随后调用现有 Adapter处理全部 512 token；
- 不读取或保存跨 clip/window/video state。

在 `VisionTransformerAdapter.forward` 只按 config 对 blocks `{4,6,8,10}` 调 RACER forward；其余 blocks 保持现有 dense forward。保存完成 router 所需的紧邻 dense pre-Adapter residual只限当前 clip、本次 forward，不能成为持久 cache。

### 3.2 新增 `configs/adatad/thumos/georoute_official_r1_racer24_prebackbone_seed42_v001.py`

只声明：

- BPNS continuous native `8x8/K64`、dense carrier 512；
- `racer_blocks=(4,6,8,10)`；
- `select_per_tubelet=24`、`spatial_tokens_per_tubelet=64`；
- `selected_query_tokens=192`、`full_kv_tokens=512`；
- `completion=parameter_free`；
- no auxiliary loss/teacher/cache/cross-clip state/new parameters。

不得加入第二 budget、动态 K、contingency selector 或训练 recipe 漂移。

### 3.3 新增 `tools/bata/profile_zoomtoken_racer24_block.py`

实现单一 real-shape matched microbenchmark：同一 checkpoint、输入、dtype/device 和 RACER block位置，control 执行 dense Q/K/V=512 + dense MLP + dense Adapter，candidate 执行 RACER24。只测 block/model path，不读取 validation/test GT，也不把结果称为 full-stack TAD efficiency。

### 3.4 新增 `scripts/run_zoomtoken_racer24_iteration0_n16r4.sh`

只运行 focused static/parity suite 与上述 microbenchmark；不调用训练 launcher、不提交 Slurm、不创建正式 run root。

### 3.5 新增 `tests/test_zoomtoken_racer24.py`

集中覆盖本计划全部机械合同，避免建立通用测试框架。

`tools/train.py` 不在 Iteration-0 修改面。只有 MCL 之后另行授权 Iteration 1，且已证明通用 successful-update hook 能支持该 backbone 时，才评估最小 allowlist；本轮不预先改训练逻辑。

## 4. Focused checks

1. selected-Q/full-KV 接口 shape 与数值：`Q=192`、`K/V=512`，现有 dense `Attention.forward` 输出保持不变。
2. 每个 tubelet 精确 24/64；禁止 flattened 512 上的 global top-192；并列按 native index确定性处理。
3. block schedule 精确为 `{4,6,8,10}`，其他 blocks 与 dense control 数值等价。
4. router score/index stop-gradient；无 router parameter；输入只来自指定的 preceding dense pre-Adapter residual与 adjacent-tubelet surprise。
5. completion 保持 `[B,512,C]`，未选 token 不删除；下一 Adapter 确实接收全部 512 token。
6. candidate 与 control 的 `named_parameters()` 名称、数量和 trainability 完全相同；diff 不得新增 `nn.Parameter`、loss、teacher、cache或 state field。
7. 无跨 clip/window/video state；连续两次独立 forward 不共享 route/completion 状态。
8. `python -m py_compile`、focused pytest、`bash -n` 与 `git diff --check` 全部通过后才允许 microbenchmark。

## 5. Matched real-shape microbenchmark

- 形状：`B=1`、8 tubelets、每 tubelet 64 native spatial tokens、dense carrier 512、每 tubelet selected 24、总 Q=192、KV=512。
- control/candidate 使用同一权重、输入、dtype、device、warmup、同步与 block 集合。
- warmup 后至少 200 次 timed repetitions；记录同步后的 p50/p95、peak allocated 与 peak reserved memory。
- 通过门：candidate p50 相对 matched R1 dense RACER-block control 至少 `1.08x`；peak memory 不高于 control `5%`。
- 任一门失败即记录 Iteration-0 benchmark failure并停止；不得调 K、改 blocks、训练或开启 FARM24/PairLatent32。

## 6. 明确拒绝

- 不运行数据集、CUDA/GPU、Slurm、训练、正式 validation、正式 cost replay或第二 seed。
- 不实现 FARM24、PairLatent32、第二 selector/capacity sweep或 rescue。
- 不复用现有 selected-only ragged attention冒充 full-KV。
- 不修改 K100-TAR50 terminal evidence，不重试 job `1261680`。
- 不补造 Project Pro provenance或北京时间 deadline。

## 7. 计划结论

MCL 机械可行，但它要求真正新增 selected-Q/full-KV attention 与 dense completion 机制；不是纯配置改动。用户已于 `2026-08-31T01:23:36+08:00` 明确授权 RACER24 Iteration-0 实施。下一责任人是独立 Builder：绑定 proposed base 后实现最小候选并返回 clean revision 与 focused checks；60-epoch训练与后继仍冻结。

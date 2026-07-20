# PhysTime 冻结双轴解码真实门禁失败 Pro 严审提示词

请作为最严厉的资深 TAD / PyTorch / 数值可复现实验审稿人与代码审计者，
直接读取并逐行检查下列 GitHub 仓库和固定提交。不要接受本文对根因的预设，
请从代码、张量 dtype、候选顺序、后处理与实验合同独立重建事实。

## 固定审计对象

- 仓库：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- 分支：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/phystime-performance-diagnosis-20260712
- 真实门禁运行时代码 commit：
  `06a6734449024875031cc3d1e0d08520824d2e67`
- tree：
  `c11dc39670254c90ad21f3e26581e4f654f25c59`
- 固定 full60 模型来源 commit/tree：
  `0dc5851a8feb12b97d16bdb5ea8fc60e9273d132` /
  `bddc9b9386604d00d213275a47ce7997b35d3f4c`
- 固定 P0 全精度后处理来源 commit/tree：
  `c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2` /
  `0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338`

任务是完全离线的稀疏 TAD 检测头机制诊断，不是 DUCA、选帧插件或 Online TAD。
本轮不训练模型，只固定 epoch-59 checkpoint，在同一冻结分类/回归张量上交换
uniform-rank 与 physical-time 解码轴。

## 必须逐行审查的代码

1. `opentad/models/dense_heads/anchor_free_head.py`
   - `enable_decode_replay_capture`
   - `consume_decode_replay_state`
   - `_capture_decode_replay_state`
   - `forward_test`
   - `get_valid_proposals_scores`
   - `_clamp_physical_proposals_to_domain`
2. `opentad/cores/phystime_decode_replay_capture.py`
   - 张量收集、dtype 转换、NPZ 写出、manifest 与内存合同
3. `tools/bata/replay_phystime_decode_cross.py`
   - `map_selected_axis`
   - `build_axis_points`
   - `decode_axis`
   - native 重建误差门禁、后处理、哈希与 evaluator
4. `tools/bata/run_phystime_decode_cross_gate.py`
   - `run_real_window`
   - direct 推理、collector、U/P replay、精确等价、四条件合同
5. `opentad/models/detectors/base.py`
6. `opentad/models/detectors/single_stage.py`
   - 尤其是 `post_processing` 的 threshold、flatten、sort、top-k、NMS 前行为
7. 两个 replay 配置：
   - `configs/adatad/thumos/phystime_g1a_selected_axis_native_j192_decode_replay.py`
   - `configs/adatad/thumos/phystime_g1a_physical_metric_native_j192_decode_replay.py`
8. 部署与验证：
   - `scripts/run_phystime_decode_cross_gate_slurm.sh`
   - `scripts/submit_phystime_decode_cross_replay.sh`
   - `tools/bata/capture_phystime_decode_cross_scheduler.py`
   - `tools/bata/validate_phystime_decode_cross_suite.py`
   - 相关 focused tests

## 已发生的正式部署

- clean snapshot：
  `/data/run01/sczc063/yuzibo/projects/opentad_phystime_decode_cross_06a6734_20260720`
- run root：
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_06a6734_20260720_161200_0800_9c608d9ee647451a91ec438c93ecc2f1`
- DAG token：
  `ptdc_06a6734_9c608d9ee647451a91ec438c93ecc2f1`
- Job：
  - gate `1175820`：`FAILED 1:0`
  - selected-online `1175821`：取消，未启动
  - selected-EMA `1175822`：取消，未启动
  - physical-online `1175823`：取消，未启动
  - physical-EMA `1175824`：取消，未启动
  - suite `1175825`：取消，未启动

全内容 preflight、checkpoint/data/VideoMAE 哈希、scheduler snapshot v2 均通过；
gate 内 Linux focused suite 为 `73 passed`。失败发生在：

```text
ValueError: real gate native replay differs from direct inference
tools/bata/run_phystime_decode_cross_gate.py:555
```

门禁按顺序完成 selected-online、selected-EMA、physical-online。
前两者 native 精确等价通过；physical-online 失败，physical-EMA 未执行。
最终 `decode_cross_gate.json` 未生成，所以正式 mAP 必须写 `NA`。

## 固定失败窗口与 artifact

- 视频：`video_test_0000004`
- 时间域：`[0.0, 33.826]` 秒
- 原始观测：`K=384`
- 选中原始帧：`253`
- native token：`J=192`
- 有效 native token：`127`
- 总候选：`Q=378`
- 共享观测序列 SHA256：
  `502cfeb5e2bf5eb0e0cc0f40fe43c5b22682ece3e711b4fe74d1b6ae158dc1b6`
- physical window SHA256：
  `62f6485ac2ad12cc32ea8bc9f35e502ae9086573eed68cf9d8f05c52313208ba`

artifact：

```text
<run_root>/gate/real_window/selected_online/decode_replay_inputs.npz
<run_root>/gate/real_window/selected_online/decode_replay_manifest.json
<run_root>/gate/real_window/selected_ema/decode_replay_inputs.npz
<run_root>/gate/real_window/selected_ema/decode_replay_manifest.json
<run_root>/gate/real_window/physical_online/decode_replay_inputs.npz
<run_root>/gate/real_window/physical_online/decode_replay_manifest.json
```

physical-EMA artifact 不存在，因为门禁在此前已 fail-closed。

文件级证据合同：

| 条件 | NPZ SHA256 / 字节 | manifest SHA256 / 字节 |
| --- | --- | --- |
| selected-online | `b6afd0afcf169e20dac43e62641cafb65e8c0ce8a49d0bee19db5ed7b1800803` / 29855 | `ba4ad835b8fb4ade18140b50437af6630766aebf78953b8c68c2b2c4ac48cd12` / 5968 |
| selected-EMA | `e741de538e2a13b7d7ac28ad25709583d695f8f4d32a6e7f26a190fe1023597e` / 29905 | `2dd361c5639fc0dc0dbc8cea15e92e60651592d5e0705514bdf2554194574548` / 5962 |
| physical-online | `1d2e8b066c4b99ab94a82ff10e3a25bdbbf06e5102022ea06fa1f18f4270854a` / 30105 | `8c6e8d05728eb13e62826489863e0634a3535bd09f6f097cdb86220ed7a1de49` / 5973 |

这些 artifact 按仓库规则不提交到 GitHub。只有 GitHub 访问权的审查者可以独立
核验代码执行链、哈希合同与下述复算程序，但不能独立读取私有远端文件；请把
“代码可核验事实”和“给定哈希绑定的远端取证”分开裁决，不要伪称已读取 artifact。

运行环境：

- gate 节点：`g0045`，`x86_64`，Linux `5.15.0-78-generic`，Slurm
  `23.11.10`；
- 分配：6 CPU、1 GPU、93300 MiB 内存；
- 固定 conda 环境：Python `3.10.20`、PyTorch `2.0.1`、NumPy `1.23.5`；
- 正式 gate 没有持久化 CPU 具体型号、实际 `torch.get_num_threads()`、
  `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 或其他线程环境。登录节点探针不能代替
  正式作业环境证据，因此当前尚未证明 tie ordering 跨 PyTorch/CPU 可移植。

源张量 dtype：

```text
cls_logits       torch.float16
cls_scores       torch.float16
reg_distances    torch.float16
base_points      torch.float32
native_points    torch.float32
native_proposals torch.float32
```

当前捕获器把 `cls_logits`、`cls_scores`、`reg_distances` 等统一转成 CPU
`float32` 后写 NPZ。

## 只读取证原始事实

physical-online：

```text
native_point_reconstruction_max_abs_error    = 0.0
native_proposal_reconstruction_max_abs_error = 0.0
proposal_coordinate_count                    = 502
proposal_rows_exact_before_clamp             = 244 / 251
nonzero_coordinate_differences_before_clamp  = 7
max_abs_before_clamp                         = 5.807575225830078
proposal_rows_exact_after_clamp              = 251 / 251
nonzero_coordinate_differences_after_clamp   = 0
native_result_count                          = 2000
replay_result_count                          = 2000
rowwise_exact_count                          = 2000
native/replay audit SHA256                   =
08022c244c88a5446e1a3f66dc569c29d055ee66fb13d8079a6bc3ba97715f51
```

这组结果只用捕获的 native proposal 作审计参照，没有把它替换进正式 replay。
它表明 point/proposal 重建与生产边界裁剪本身可精确闭合。

将同一捕获分数按源 dtype `float16` 与存档 dtype `float32` 分别送入完全相同的
`SingleStageDetector.post_processing()`，得到：

| 条件 | 分数元素 | 同值冗余计数 `sum(n-1)` | 结果 |
| --- | ---: | ---: | --- |
| selected-online | 5020 | 3812 | fp16/fp32 逐行及哈希相同 |
| selected-EMA | 5020 | 3810 | fp16/fp32 逐行及哈希相同 |
| physical-online | 5020 | 3830 | 零基索引 147（第 148 条）分叉；top-2000 各有 2 条独有 |

physical-online：

```text
float16 postprocess SHA256 =
fd07d8b7c6f366e0996cd97c4ee04d7d51a02e03213a508abcd624f9d3b5ceb3

float32 postprocess SHA256 =
08022c244c88a5446e1a3f66dc569c29d055ee66fb13d8079a6bc3ba97715f51
```

首个顺序差异的分数都为 `0.06805419921875`，但候选分别为
`CricketBowling [27.9617577, 28.7025948]` 和
`JavelinThrow [19.3885345, 21.8515282]`。由于 top-k 截断，差异不只是列表
重排，而是最终 2000 条集合各有 2 条不同。physical-online 中 4613 个元素属于
至少含两个元素的同值组；3830 是每个同值组扣除首项后的冗余计数。

在固定 snapshot 与 run root 可见的环境中，以下只读程序复算 score dtype
干预，不写 artifact、不使用捕获 proposal 覆盖重建 proposal：

```python
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
from mmengine.config import Config

from opentad.models.detectors.single_stage import SingleStageDetector
from tools.bata.replay_phystime_decode_cross import (
    build_metas,
    canonical_sha256,
    decode_axis,
)

run_root = Path(
    "/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/"
    "phystime_decode_cross_06a6734_20260720_161200_0800_"
    "9c608d9ee647451a91ec438c93ecc2f1"
)
case_root = run_root / "gate" / "real_window" / "physical_online"
manifest = json.loads(
    (case_root / "decode_replay_manifest.json").read_text()
)
with np.load(
    case_root / "decode_replay_inputs.npz",
    allow_pickle=False,
) as archive:
    arrays = {name: archive[name] for name in archive.files}

cfg = Config.fromfile(
    "configs/adatad/thumos/"
    "phystime_g1a_physical_metric_native_j192_decode_replay.py"
)
post_cfg = copy.deepcopy(cfg.post_processing)
post_cfg.sliding_window = True
detector = SingleStageDetector()
outputs = {}
for dtype in (np.float16, np.float32):
    replay_arrays = dict(arrays)
    replay_arrays["cls_scores"] = arrays["cls_scores"].astype(
        dtype,
        copy=True,
    )
    decoded = decode_axis(
        replay_arrays,
        "physical_time_seconds",
        "physical_time_seconds",
    )
    outputs[np.dtype(dtype).name] = detector.post_processing(
        (decoded["proposals"], decoded["scores"]),
        build_metas(manifest),
        post_cfg,
        manifest["class_map"],
    )

video = manifest["windows"][0]["video_name"]
valid_scores = arrays["cls_scores"][
    arrays["native_mask"]
].astype(np.float16).reshape(-1)
counts = Counter(valid_scores.tolist())
print("score_elements", valid_scores.size)
print(
    "duplicate_excess_count",
    sum(value - 1 for value in counts.values() if value > 1),
)
for name, result in outputs.items():
    print(name, len(result[video]), canonical_sha256(result))

left = outputs["float16"][video]
right = outputs["float32"][video]
first = next(
    (
        index
        for index, pair in enumerate(zip(left, right))
        if pair[0] != pair[1]
    ),
    None,
)
print("first_diff_zero_based", first)
print("first_diff_one_based", None if first is None else first + 1)
encode = lambda item: json.dumps(
    item,
    sort_keys=True,
    separators=(",", ":"),
)
left_set = {encode(item) for item in left}
right_set = {encode(item) for item in right}
print("float16_only_rows", len(left_set - right_set))
print("float32_only_rows", len(right_set - left_set))
```

固定输出应为：

```text
score_elements 5020
duplicate_excess_count 3830
float16 2000 fd07d8b7c6f366e0996cd97c4ee04d7d51a02e03213a508abcd624f9d3b5ceb3
float32 2000 08022c244c88a5446e1a3f66dc569c29d055ee66fb13d8079a6bc3ba97715f51
first_diff_zero_based 147
first_diff_one_based 148
float16_only_rows 2
float32_only_rows 2
```

## 请独立裁决的问题

1. 上述取证是否足以证明根因是 score dtype 改写触发的并列排序/top-k 语义变化？
   还缺哪些能推翻该解释的证据？
2. native exact canonical equality 是否仍应保留为硬门禁？如果不保留，哪一种更
   严格且不会掩盖候选集合变化的等价合同才合理？
3. 最小正确修复是否应为：
   - 在 artifact 中原样保留 `cls_scores=float16`；
   - 回归距离可继续显式转为 float32 做 CPU 解码；
   - U/P 两个解码都共享同一源 dtype 分数；
   - native replay 必须继续逐行、逐哈希等于 direct？
4. 或者是否必须给生产与重放共同引入显式、稳定、可审计的 total-order
   tie-break（分数、候选索引、类别索引）？这会不会改变冻结 P0/source 的生产
   推理语义，因而不再是合法的无训练反事实？
5. `cls_logits` 和 `reg_distances` 应保留源 dtype 还是分别规定存储 dtype /
   计算 dtype / 排序 dtype？请逐张量给出合同。
6. 当前 `canonical_sha256` 对列表顺序敏感是否正确？是否还应同时记录
   ordered hash、multiset hash、top-k candidate-id hash 和首个差异，但仍以
   ordered exact 为 pass 条件？
7. gate 是否应在 raise 前原子写出失败诊断 artifact，包括 direct/replay
   hashes、首个差异、集合差异、source/stored dtype，而不能只留下 traceback？
8. selected 两条件恰好通过是否会掩盖测试缺口？应增加哪些专门构造
   “top-k 截断处大量同分”的反例测试？
9. 修复后是否只需重跑单窗口四条件 gate，还是必须重新锚定 P0 direct、
   pre-cross、整套 full-dataset evaluator 与所有哈希？
10. 在新 gate 通过前，是否应继续冻结 Q192 UU/UP/PU/PP 训练？请明确
    `REQUEUE / REVISE / ABANDON` 裁决。
11. 新合同只需复现冻结运行时的生产语义，还是必须定义跨 PyTorch 版本、CPU
    型号和线程数均稳定的 total order？如果要求可移植稳定排序，如何避免把
    source/P0 推理语义改变后仍称为“同一冻结模型的无训练反事实”？

## 禁止接受的“修复”

- 降低哈希或逐行等价要求；
- 用数值容差忽略最终候选集合变化；
- 对 score/segment 事后舍入以制造相等；
- 把结果排序后只比较集合，掩盖 top-k 成员变化；
- 使用捕获的 native proposal 覆盖从原始回归张量重建的 proposal；
- 删除 physical-online 失败条件；
- 复用旧 run root、DAG token 或 Job ID；
- 在 corrected gate 通过前启动正式 replay 或任何新训练；
- 用 selected 两条件通过推断 physical 条件也正确。

## 要求输出格式

请用中文输出，代码标识可保留英文。先给结论，后给证据：

1. `RESEARCH_VERDICT=REQUEUE_AFTER_FIX / REVISE_BEFORE_REQUEUE / ABANDON`
2. P0/P1/P2 问题表，必须包含文件、函数、行号、触发条件、影响
3. 对根因的独立证明链与可证伪点
4. 逐张量 dtype/设备/计算/排序/存档合同
5. 最小代码修改清单，给出接近可落地 patch 的伪代码
6. 必须新增的单元、集成、真实 CUDA gate 测试
7. 新实验 DAG：preflight → 四条件 gate → 四 replay → suite
8. 明确停止条件与不可声称的论文结论

不要泛泛建议“尝试更多实验”。任何建议都必须说明它修复哪一条已观察失败、
是否改变冻结 source/P0 推理语义，以及如何用自动化门禁证明没有引入新的自证。

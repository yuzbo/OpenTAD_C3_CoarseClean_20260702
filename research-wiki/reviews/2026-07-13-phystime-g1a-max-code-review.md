# PhysTime G1a Max Code Review Ledger

状态：`tested`，尚未部署新 gate 或 pilot，尚无新 mAP。

## 审查门槛

用户要求独立 Max agent 对当前实现逐行审查；只有零 P0/P1、相关回归通过、clean snapshot 绑定完成后才允许部署真实 gate。gate 通过后也只能解锁 6-epoch pilot，不能直接声称方法有效。

## 第一轮发现与修复

- 顶层聚合字段可伪造：validator 现从逐步 assignment、梯度、LR、AMP 与 optimizer state 证据重算。
- state-dict buffer 可冒充参数更新：改为参数集合 hash 与实际参数 delta。
- gate 不是正式训练轨迹：改用 batch=2 production DataLoader、正式 warmup scheduler、EMA 和生产更新顺序。
- `rpn_head.scale` 漏出回归梯度族：已纳入。
- pilot schema、反序列化和 artifact 绑定不足：升级为 v3 并重算 evaluator。
- 对 `1161378` 的 ReLU 根因表述过强：旧 artifact 缺 assignment、reg_loss 与 pre-ReLU 证据，只能称“行为一致”，不能称已证明。

## 第二轮发现与修复

- assignment 计数可接受不可能数据：现在要求 batch=2 时 `valid_count=756`、`gt_count>0`，且 `active<=raw_positive<=2*active`。
- optimizer 仅检查最大 step：现在绑定固定 optimizer 参数名称集合；每一步要求 state count 完整、min=max=step，并在第三步复核。
- `build_dataloader` 只把 `drop_last` 传给 sampler：现在也传给真实 DataLoader，并以实际 loader 属性记录 gate contract。
- pilot artifact 校验仍有传递信任：完成时独立重查 clean Git commit/tree、canonical config、dataset Merkle、contract/G0 schema/pass/hash、EMA、optimizer 与 scheduler。
- VideoMAE 可能在 forward 后改变 `requires_grad`：hash 改为固定 optimizer 参数集合，不再依赖动态 trainable 集合。
- gate 原地把缓存 batch 搬到 GPU：改为每步构造短生命周期 device copy，CPU batch 保持不变；推理另建 device copy。
- gate/pilot worker 随机顺序可能不一致：DataLoader 接收显式 seed，DistributedSampler 与 generator 同源；正式 `tools/train.py` 同样传 seed。

## 验证

- 新增契约测试先得到 `6 failed, 22 passed, 31 errors`，证明缺口真实存在。
- 修复后 gate/artifact focused tests：`65 passed`。
- PhysTime 与 shared physical-grid 全部相关 tests：`240 passed`。
- 全仓回归：`783 passed, 7 failed`；其中 shared physical-grid 的旧期望已确认与映射公式矛盾并修正为 stride=5、target=0.1。其余 6 个失败属于已排除的 DUCA、外部 action-seg 仓缺失和旧 PC-OT fake-Torch 路线，不作为 G1a 通过证据，也不在本轮修改。

## 当前结论

第三轮独立 Max 复审尚未完成，所以状态仍是 `tested`，禁止部署。只有第三轮零 P0/P1 后才创建 clean commit/snapshot，并运行静态 contract -> G0 -> real gate；real gate 成功后再解锁 matched pilots。

## 第三轮等待期间的执行端自查

- 发现正式 launcher 的 `manifest.checkpoint` 表示 VideoMAE 预训练权重，而不是 pilot 产出的 `epoch_5.pth`；旧测试夹具错误把两者合并，导致 validator 增加了不可能满足的路径相等约束。
- 已拆分 pretrained checkpoint 与 pilot epoch checkpoint：manifest/gate 继续绑定前者，completion 独立定位并反序列化后者，检查 EMA、optimizer 与 scheduler。focused tests 仍为 `65 passed`。
- 此修复属于部署前 P1，已通知独立 Max agent 必须重读最新 diff 后再裁决；旧 diff 的任何 GREEN 均无效。

## 第三轮 Max 复审 P1 与修复

- 第三轮独立 Max 复审 verdict 为 `HOLD_P1_FIX_REQUIRED`，阻塞 clean snapshot 和 real gate。
- P1-1：逐步 gate artifact 只记录 optimizer state count/min/max，未保存并验证 optimizer state parameter names SHA256；修复为每个 optimizer step 写入 `optimizer_state_parameter_names_sha256_after`，并在 validator 中要求它等于固定 optimizer 参数集合 hash。新增篡改单步 hash 的拒绝测试。
- P1-2：validator 直接信任顶层 `parameter_schema_match`、`initial_state_match`、`optimizer_schema_match`，未从两臂 `variants` 独立重算；修复为从 `selected_axis` 与 `physical_metric` artifact 重算三项一致性，并要求顶层布尔和重算结果同时为真。新增篡改单臂 parameter schema、initial state、optimizer schema 且保留顶层布尔为真的拒绝测试。
- 修复后本地状态仍为 `tested`，不是 `experiment_running`；必须等待 focused 回归和同一 Max agent 复审变为 `GREEN_FOR_CLEAN_SNAPSHOT_AND_REAL_GATE` 后，才允许提交、推送、创建 clean snapshot 与运行 real gate。

## 第三轮复审结论

- 同一独立 Max agent 复审修复后给出 `GREEN_FOR_CLEAN_SNAPSHOT_AND_REAL_GATE`。
- P0：无。P1：无。
- 绿色前验证：远端 `tests/test_phystime_g1a_real_gate_contract.py -q` 为 `30 passed`；远端 `tests/test_phystime*.py tests/test_c3_physical_grid*.py -q` 为 `243 passed`；AGENTS focused 远端为 `23 passed`；本地 `py_compile` 与 `git diff --check` 通过。
- 状态含义：允许创建 clean commit/snapshot 并运行真实 gate；仍不得在 gate/pilot 产生 mAP 前声明 `empirically_supported` 或 `paper_ready`。

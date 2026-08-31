---
type: experiment
status: needs_attention
updated: 2026-08-23
---

# EAST ViT-G 官方发布权重原配置评估

## 科学目的

在不训练、不修改网络结构的条件下，用 EAST 作者发布的 ViT-G detector checkpoint、原始
50Salads 配置、官方五折标注和官方 mAP evaluator，复核发布模型的低帧率动作检测结果。该评估
只建立 EAST 官方锚点，不构成 DUCA 方法结果，也不使用正在训练的 ViT-S 权重替代 ViT-G。

## 冻结身份

- 官方代码：`https://github.com/tqosu/EAST`，revision
  `a3233c2e6a6e3bbe36f9663e18180bdc5c126556`。
- 原始配置：
  `configs/adatad/50salads/e2e_50salads_videomaev2_g_768x1_160_fps2_adapter.py`。
- 官方发布页给出的低帧率 mAP 锚点：tIoU 0.3/0.4/0.5/0.6/0.7 为
  `92.6334/91.1523/89.6745/87.5004/83.3251`，平均 `88.8571`。
- 规范数据：
  `/data/run01/sczc063/yuzibo/datasets/TAS/east_50salads_160x160_2fps`；50 个 RGB 视频，
  `160x160 @ 2 fps`，五个互斥的 40/10 training/validation folds。
- 远端执行候选：
  `/data/run01/sczc063/yuzibo/projects/duca_east_baseline_37c0d08_20260822`，revision
  `37c0d080a2bce948dc73643578f05b2229934d2c`。候选只补齐恢复/保留合同和无效导出修复，
  不改变 ViT-G、数据、损失或 evaluator。

## 评估合同

作者命令入口为 `tools/test.py`，每个 checkpoint 必须显式绑定对应 fold：

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 tools/test.py \
  configs/adatad/50salads/e2e_50salads_videomaev2_g_768x1_160_fps2_adapter.py \
  --cfg-options split=splitN \
  annotation_path=data/annotations/50salads.fps2.splitN.json \
  feature_stride=2 \
  model.backbone.backbone.type=VisionTransformerAdapter3 \
  model.backbone.backbone.adapter_drop_rate=0.5 \
  work_dir=exps/50salads/adatad/e2e_actionformer_videomaev2_g_768x1_160_fps2_adapter3_p0.5 \
  --checkpoint /absolute/path/to/official_released_checkpoint.pth
```

配置启用 EMA，`tools/test.py` 因而必须从发布 checkpoint 严格加载
`state_dict_ema`。评估使用 annotation subset `validation` 和 tIoU
`[0.3,0.4,0.5,0.6,0.7]`。发布包取得后，必须先读取其文件清单与 fold 绑定；若作者只发布单个
fold checkpoint，则只报告该 fold，不把它伪装成五折均值。

## 当前资源与 blocker

- 已有文件
  `/data/run01/sczc063/yuzibo/pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth`
  是 2,025,314,665-byte VideoMAEv2 ViT-G backbone 预训练权重，不是 EAST detector checkpoint。
- 对相关远端目录和本机下载目录的定向清点均未发现 EAST ViT-G detector checkpoint。
- 官方模型链接
  `https://oregonstate.box.com/s/c14851yhp3pibqefkfbcrqukrw0eatis` 在匿名 HTTP、应用内浏览器和
  Edge 会话中都重定向到 Oregon State University Box 登录；页面不公开文件名、大小、校验信息
  或直接下载地址。
- 因此当前为 `NEEDS_ATTENTION / CHECKPOINT_ACCESS_REQUIRED`。没有提交 Slurm 评估，也没有
  产生指标。最小合法恢复是由具备该 Box share 访问权的用户会话下载作者原始发布包，或由作者
  提供新的公开官方链接；文件取得后先核验包内 fold/EMA 身份，再立即提交原配置评估。

## 用户提供归档的身份核验

2026-08-23 对 `D:/chrome_download/50salads.tar.gz` 作了只读清单核验。归档大小为
`56,052,488` bytes，SHA-256 为
`b604fc78c3b7fc142da1a2c4b5c350444c28dba62de7ad178298a51bc9e067f7`，包含五折
`l_5uniform/split_N/{best_eva_acc,best_eva_FEA}.{model,opt}` 和 100-epoch 训练日志。每个
`.model` 仅约 `2.06 MB`；命名、训练长度和保存逻辑与官方 `ms-tcn-master2/model.py` 的 EAST
Stage-2 高帧率聚合/细化网络完全对应。它既不含 `.pth`，也不含 ViT-G、adapter、ActionFormer
或 `state_dict_ema`，因此不能用于本节点的 Stage-1 ViT-G 原配置评估。该包保留为 Stage-2
候选材料，未解压、未部署、未产生新指标；ViT-G detector checkpoint 的访问 blocker 不变。

# 08_EVAL_ORACLES｜评估、风险切片和Oracle诊断

你拥有geosparse_ext/evaluation和diagnostics。
使用真实官方评估器并pin commit；统一mAP、high-tIoU、train分位数短动作、边界匹配率与漏检、长/重叠/长背景/gap/tail等切片。
实现D01–D06全部诊断，不等主结果出来再决定做哪些。D01固定128个内部诊断窗口、8原子菜单、K4的70组合，名称finite-menu loss-best；不称mAP理论上界。cache有依赖时要求重跑。
D02同selection manifests比较receiver；D03预算histogram与长度分层打乱；D04连续loss/两端误差交互；D05mask+content+switch；D06缓存前后真实输出校验。
测试GT只用于评估或明确Oracle，不输入部署router。小物体无标注、VFR无PTS则该slice为NA，不发明标签。不要只汇报成功匹配的边界。
每train完成自身epoch59即运行其eval，不等待其他所有训练；需要配对汇总时保存单边结果后继续其他任务。
交付raw predictions、slice definitions、bootstrap输入、每任务metrics/NA原因和官方code hash。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。

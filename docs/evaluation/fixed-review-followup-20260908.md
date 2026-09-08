# 固定审查 R4541c641 的修复与补测

本轮按用户提供的611行独立审查实施。R=`4541c641fc61333d6f2015a9f7e72c8b2a31c088`；生产M=`b70ae056c495b43ca3f305fe438926b97b2723b5`。修订从SCI3 `a4fa50b5a37184b24705ae261dc94b9188e3bc5c`建立独立 `codex/geosparse-review-fixes-20260908` 分支。官方核心、配置、原入口不改；不把新代码接到旧任务ID。

| 问题 | 处置及证据边界 |
| --- | --- |
| N01 推理预算退化 | 保留原精度与真实成本；C20复用245503，A E40固定q=.5复用1280030，不重提。新增A60与B20只回放冻结EMA Scout和原始计划，无Heavy、optimizer或新训练。没有完整账本前不推断新epoch预算。 |
| N02 多tubelet acquisition尾部 | 新增原子只写入有效native成员，重新生成执行索引；spatial_slots在聚合前交集layout.valid。768输入/755有效、query768是合法尾窗，不是query755。默认atom1主方法不包含混合有效/无效tubelet原子，不能由此宣称其精度受污染。 |
| N03 不完整MAC形成前沿 | 保留下界散点，以“>”标示；某数据集任一model count不完整时整条模型成本前沿不绘制。Heavy公式与实测latency仍可独立报告，不因缺失模型算子计数删除准确率。 |
| N04 风险—成本身份 | join复用require_same_checkpoint；同epoch/commit而不同权重的记录明确拒绝。测量修复仅需重新导出/绘图。 |
| N05 B交换非总成本匹配 | API改为equal_heavy_cost_swap，pair参数require_equal_heavy_cost、输出equal_heavy_cost。B每个真实forward临时观察receiver输入、各Linear/attention形状和MAC下界；总成本匹配保持NOT_ESTABLISHED。不因额外槽改变而否定已有Heavy等成本性质。 |
| N06 uniform控制 | 当前A/B uniform为T，主方法为ST；C coarse/fine uniform为匹配ST。保留已有任务，不将A/B该差值仅归因selector。新增匹配ST对照需明确独立任务，不能改名替代运行中的T对照。 |

另外，训练曲线按(epoch,step)保留最后一条日志，排除超时恢复的重复行；CSV保留成功更新标记，不能将overflow或废弃尝试算成新增训练。

## 验证范围

focused检查包括实际小宽度VideoMAE/TIA/ActionFormer的768尾窗acquisition前向/反向（atom1/4/8）、独立invalid evidence输入、真实B的同parent同tubelet 2x2交换但receiver槽1→2、同计划/反向pair及状态恢复、不完整MAC下界与前沿、仅权重SHA不同的risk join、恢复日志去重。

```bash
python -m pytest tests/test_geosparse_review_tail.py tests/test_geosparse_review_figures.py tests/test_sci3_interventions.py tests/test_geosparse_figures.py -q
```

这些是CPU合成输入正确性检查；通过情况由精确提交的测试收据记录，不是实际视频精度、CUDA训练准入或机制成立证明。真实诊断运行应从独立工具源码加载冻结M，不能默默改用本分支的模型修复版本。

## 检查点和研究结论

六个fixed/dynamic主任务、B-full、统一dense继续按M及原身份推进。官方独立训练best/E60保留；没有证据要求重训。缺失某epoch验证时补已有检查点推理，不重训60轮。

现有训练梯度分量工具在R之后已完成于SCI3 18281bac（10项CPU检查），因此不是仍无实现，但尚没有真实训练视频/普通训练检查点的梯度分量数据。没有数据前，不从actor loss标量推断梯度主导，不把B低分唯一归因为receiver或压缩。

同checkpoint强制固定预算只测计划敏感性，不等同于该预算完整训练成绩；未来改变预算训练/部署规则、B结构或C更新仍需新实验身份。所有测试集驱动开发照原授权如实标注，不把此类结果称为未见测试集泛化。

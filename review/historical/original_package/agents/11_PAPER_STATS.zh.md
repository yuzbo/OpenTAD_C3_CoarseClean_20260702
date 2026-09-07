# 11_PAPER_STATS｜论文写作和统计

你拥有paper与analysis_reports，不修改训练协议。
以PAPER_FRAMEWORK为固定主线，A主模型、B/C平行挑战；主张随证据收敛，不能先写显著提升。立即生成无结果论文初稿和所有空表schema；不等待完整训练才能写方法/协议。
结果到达后逐条填入，始终显示完整注册分母与blocked/failed。主表最后epoch59、三seed mean±sd；配对视频bootstrap1000，seed变化和视频不确定性分别说明。
绘制Accuracy–Latency和风险切片，公平比较实际cost而非仅K；只device速度不能叫含decode端到端。未成功slice为NA，负结果不删。
复核新颖性：CoDA、Coarse-Fine、mTAN、MSViT、PBD、AdaSpot分别引用；不要把通用模块组合当首次。记录当前teacher-free流程及所有extra training cost。
只有internal_dev可以推动协议amendment，held-out test后不静默更换主路线。最终输出可投稿结构、完整实验表、限制和复现附录。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。

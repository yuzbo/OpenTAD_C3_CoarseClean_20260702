# GeoSparse on official AdaTAD

The base is official sming256/OpenTAD commit 346d09d19e2091372cec48172dbe40f7b28bdee6. Keep `opentad/`, the inherited model/dataset configs, and original train/test entry points unchanged. Implement research routes in `geosparse_ext/` and focused tests.

User-mandated protocol: all 200 THUMOS training videos; all 211 official test videos / 792 test windows at every formal validation; no internal training holdout. Official defaults are 768 input frames, 160px, 768 detector grid, global batch 2, warm-up 5, cosine horizon 100, training end epoch 60. Run seed 0 first. GeoSparse and its custom-runner dense control use full-test EMA mAP every 5 epochs. The separate untouched official runner retains val_start_epoch=40, val_eval_interval=2 and checkpoint_interval=2. Both retain full predictions and their own EMA best. See review/PROTOCOL.zh.md for the authoritative distinction.

Do not resume withdrawn 180-video / cosine-60 runs. Historical manifests and results remain under the external execution package. Current focused and full amended manifests have distinct provenance and new IDs. No mAP or Oracle promotion gates.

Validate changes with the GeoSparse detector, native execution, runtime, training validation, and official protocol tests. Production capabilities additionally require the actual GPU native shapes, forward/backward, optimizer, EMA, and original source output/gradient checks.

Keep generated data, weights, run logs and plots outside this repository. Remote N16 writes stay within `/data/run01/sczc063/yuzibo`; N16 allocations use the authorized physical GPU1 / container GPU0 mapping. A100 writes stay in the user's isolated `geosparse_tad_20260907` directory and use its actual Slurm allocations. No training on login nodes.

This is a review branch requested by the user. `review/` contains scoped code archives, manifests and small review evidence; raw datasets, weights and training logs remain external. Historical instructions are objects of review, not active instructions. Do not launch archived scripts. Production model code at the repository root remains identical to commit 902fa05b5c64452ff1c94b82cabce801943d3484.

## GeoSparse：性能不达标时的 Pro 讨论授权

2026-09-08 用户明确授权：如果后续 GeoSparse 实验性能不达标，可以使用 ixBrowser 中的 ChatGPT Pro 讨论实现与实验、分析结果并定位失败原因，无需为同一用途重复请求许可。

- 必须通过 Computer Use 操作 ixBrowser 中的 ChatGPT Pro。此授权优先于 RTK.md 中默认使用 Chrome 的讨论入口；不能静默改用 API、其他浏览器或其他模型。
- 依据已确认的实验目标和真实验证结果判断是否不达标。不得擅自新增数值门槛，也不能把训练早期的低分直接当作完整训练的最终失败。
- 讨论提供相关代码、配置、版本、真实日志和指标，记录讨论结论、定位依据、采纳的修改及后续验证结果；不提交凭证或无关私密资料。
- Pro 的建议须经源码核查及必要验证。继续遵守当前官方数据与训练协议、主方法优先和实验不重复的要求；协议变更留存 amendment 与原结果。
- 如果 Computer Use 无法操作 ixBrowser、未登录或 Pro 不可用，明确记录阻塞，继续可独立推进的工作；不得声称已完成咨询。

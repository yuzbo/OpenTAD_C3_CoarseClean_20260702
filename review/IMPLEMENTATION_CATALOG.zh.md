# 逐配置实现登记

依据实际 `entry.job_blockers` 生成。通过静态参数检查不等于所有字段生效、GPU 就绪、实验完成或科学结论成立。
表中只展示 seed 0；全部种子、子任务及其拒绝原因见同目录 JSON。真实运行状态另见 `STATUS.zh.md`。

共 204 个配置与数据集组合，612 个训练注册项；静态入口接受其中 375 个训练注册项。

|seed 0 训练 ID|路线|数据集|配置名|静态检查|明确拒绝原因|
|---|---|---|---|---|---|
|tr-bacdbb058eca|DENSE|thumos14|geosparse_dense_control|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-32b5f5690706|DENSE|activitynet13|geosparse_dense_control|阻塞|activitynet13 raw-video protocol integration pending|
|tr-d790cedc3c3d|A|thumos14|A_full|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-0f38bed9ed41|A|activitynet13|A_full|阻塞|activitynet13 raw-video protocol integration pending|
|tr-c2d93c4c3979|B|thumos14|B_full|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-2c0b3c52e714|B|activitynet13|B_full|阻塞|activitynet13 raw-video protocol integration pending|
|tr-fb2b9b2359f8|C|thumos14|C_all_fine|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-672e335d258c|C|activitynet13|C_all_fine|阻塞|activitynet13 raw-video protocol integration pending|
|tr-43c9bfd942ed|COARSE|thumos14|cheap_only|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-16ea9a7df1cc|COARSE|activitynet13|cheap_only|阻塞|activitynet13 raw-video protocol integration pending|
|tr-a06d75fa95ab|A|thumos14|A_T_fixed_0.25|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-743a24a870fd|A|activitynet13|A_T_fixed_0.25|阻塞|activitynet13 raw-video protocol integration pending|
|tr-088ffe542ea8|A|thumos14|A_T_fixed_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-711aaa19bd57|A|activitynet13|A_T_fixed_0.5|阻塞|activitynet13 raw-video protocol integration pending|
|tr-27b999666771|A|thumos14|A_T_fixed_0.75|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-a10dc78bf59f|A|activitynet13|A_T_fixed_0.75|阻塞|activitynet13 raw-video protocol integration pending|
|tr-1e5e1b08fd53|A|thumos14|A_ST_fixed_0.25|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-f5ccbcad58f5|A|activitynet13|A_ST_fixed_0.25|阻塞|activitynet13 raw-video protocol integration pending|
|tr-c43d3e9cad58|A|thumos14|A_ST_fixed_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-96dba1dca8fc|A|activitynet13|A_ST_fixed_0.5|阻塞|activitynet13 raw-video protocol integration pending|
|tr-da137a0d2afc|A|thumos14|A_ST_fixed_0.75|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-a7c9b266acc1|A|activitynet13|A_ST_fixed_0.75|阻塞|activitynet13 raw-video protocol integration pending|
|tr-27cd6aee018e|A|thumos14|A_ST_dynamic_0.25|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-f694c3d3c6b3|A|activitynet13|A_ST_dynamic_0.25|阻塞|activitynet13 raw-video protocol integration pending|
|tr-32754eef72d1|A|thumos14|A_ST_dynamic_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-b381631f360c|A|activitynet13|A_ST_dynamic_0.5|阻塞|activitynet13 raw-video protocol integration pending|
|tr-b5e7578192e8|A|thumos14|A_ST_dynamic_0.75|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-9a4e838cf739|A|activitynet13|A_ST_dynamic_0.75|阻塞|activitynet13 raw-video protocol integration pending|
|tr-dfa40292845c|B|thumos14|B_T_fixed_0.25|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-c5fc8345399d|B|activitynet13|B_T_fixed_0.25|阻塞|activitynet13 raw-video protocol integration pending|
|tr-9ff2e1a742bd|B|thumos14|B_T_fixed_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-b8a8f1640fb7|B|activitynet13|B_T_fixed_0.5|阻塞|activitynet13 raw-video protocol integration pending|
|tr-150a69402b7a|B|thumos14|B_T_fixed_0.75|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-59cc9d3eac14|B|activitynet13|B_T_fixed_0.75|阻塞|activitynet13 raw-video protocol integration pending|
|tr-e00ad1e1116e|B|thumos14|B_ST_fixed_0.25|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-80774a9b7424|B|activitynet13|B_ST_fixed_0.25|阻塞|activitynet13 raw-video protocol integration pending|
|tr-be89cd6cb2ca|B|thumos14|B_ST_fixed_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-cbf905e9afe1|B|activitynet13|B_ST_fixed_0.5|阻塞|activitynet13 raw-video protocol integration pending|
|tr-e07941bc0730|B|thumos14|B_ST_fixed_0.75|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-f040a270726f|B|activitynet13|B_ST_fixed_0.75|阻塞|activitynet13 raw-video protocol integration pending|
|tr-cb1067c04f77|B|thumos14|B_ST_dynamic_0.25|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-39f8f04aed8a|B|activitynet13|B_ST_dynamic_0.25|阻塞|activitynet13 raw-video protocol integration pending|
|tr-8a922c4e6ca7|B|thumos14|B_ST_dynamic_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-897ae22f3c4c|B|activitynet13|B_ST_dynamic_0.5|阻塞|activitynet13 raw-video protocol integration pending|
|tr-4c353ec8d5c4|B|thumos14|B_ST_dynamic_0.75|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-6c64203ba911|B|activitynet13|B_ST_dynamic_0.75|阻塞|activitynet13 raw-video protocol integration pending|
|tr-5e74e2c1f122|C|thumos14|C_fixed_0.25|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-8b7f3283f211|C|activitynet13|C_fixed_0.25|阻塞|activitynet13 raw-video protocol integration pending|
|tr-a45686a40afa|C|thumos14|C_fixed_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-a8835a6c636b|C|activitynet13|C_fixed_0.5|阻塞|activitynet13 raw-video protocol integration pending|
|tr-2d7efeb32f89|C|thumos14|C_fixed_0.75|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-bca79ac1107f|C|activitynet13|C_fixed_0.75|阻塞|activitynet13 raw-video protocol integration pending|
|tr-72ba800e6f08|C|thumos14|C_dynamic_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-50239701ca26|C|activitynet13|C_dynamic_0.5|阻塞|activitynet13 raw-video protocol integration pending|
|tr-573527561f1c|C|thumos14|C_dynamic_0.75|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-4b317c0ba5a9|C|activitynet13|C_dynamic_0.75|阻塞|activitynet13 raw-video protocol integration pending|
|tr-f90ea0ae2ce4|A|thumos14|A_uniform|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-ff0d0e144b73|A|thumos14|A_random|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-c14dbe9b3709|A|thumos14|A_motion|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-896e28fef7de|A|thumos14|A_actionness|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-a2206cdebb45|A|thumos14|A_uncertainty|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-f1c633aca1d5|A|thumos14|A_cdf_native|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-dd00d28afcf8|A|thumos14|A_pg_only|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-6f26bfd4ff96|B|thumos14|B_uniform|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-2790b615f225|B|thumos14|B_random|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-cf2a5a9218bc|B|thumos14|B_motion|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-a8e8b7a36ead|B|thumos14|B_actionness|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-3fa03e702934|B|thumos14|B_uncertainty|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-e4abfa6daf11|B|thumos14|B_cdf_native|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-88e517bd9aae|B|thumos14|B_pg_only|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-8029fd7636a9|DUCA|thumos14|legacy_DUCA_0.25|阻塞|route='DUCA': implementation pending; selector='legacy': implementation pending; estimator='legacy': implementation pending|
|tr-53acaaa7f835|DUCA|activitynet13|legacy_DUCA_0.25|阻塞|route='DUCA': implementation pending; selector='legacy': implementation pending; estimator='legacy': implementation pending; activitynet13 raw-video protocol integration pending|
|tr-3094c92c3a6d|B|thumos14|wholeclip_uniform_0.25|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-1f5ff4c5bbaa|B|thumos14|wholeclip_hybrid_0.25|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-59e8d8f3ccb0|DUCA|thumos14|legacy_DUCA_0.5|阻塞|route='DUCA': implementation pending; selector='legacy': implementation pending; estimator='legacy': implementation pending|
|tr-5e27bdc35bfb|DUCA|activitynet13|legacy_DUCA_0.5|阻塞|route='DUCA': implementation pending; selector='legacy': implementation pending; estimator='legacy': implementation pending; activitynet13 raw-video protocol integration pending|
|tr-2bb36224e9fc|B|thumos14|wholeclip_uniform_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-b1685c37155c|B|thumos14|wholeclip_hybrid_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-239c4ae522b3|DUCA|thumos14|legacy_DUCA_0.75|阻塞|route='DUCA': implementation pending; selector='legacy': implementation pending; estimator='legacy': implementation pending|
|tr-ea3f14b8edc8|DUCA|activitynet13|legacy_DUCA_0.75|阻塞|route='DUCA': implementation pending; selector='legacy': implementation pending; estimator='legacy': implementation pending; activitynet13 raw-video protocol integration pending|
|tr-32bf8049c8b9|B|thumos14|wholeclip_uniform_0.75|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-300889e239f4|B|thumos14|wholeclip_hybrid_0.75|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-067daa13036b|C|thumos14|C_uniform|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-872934ec5cea|C|thumos14|C_random|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-f2913a6c0fba|DEPTH|thumos14|static_depth_4|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-d552707cd9f5|DEPTH|activitynet13|static_depth_4|阻塞|activitynet13 raw-video protocol integration pending|
|tr-4eba2105b039|DEPTH|thumos14|static_depth_6|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-127c20e9e3f3|DEPTH|activitynet13|static_depth_6|阻塞|activitynet13 raw-video protocol integration pending|
|tr-e2cafffbce5f|DEPTH|thumos14|static_depth_8|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-04810097ea1e|DEPTH|activitynet13|static_depth_8|阻塞|activitynet13 raw-video protocol integration pending|
|tr-cca67ca4737f|DEPTH|thumos14|static_depth_10|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-bc6a528461a8|DEPTH|activitynet13|static_depth_10|阻塞|activitynet13 raw-video protocol integration pending|
|tr-41db2380fdb0|DEPTH|thumos14|pbd_inspired_tf|阻塞|selector='pbd_inspired_tf': implementation pending; estimator='current_model_ablation': implementation pending; active_depth: registered extension pending|
|tr-841f2613a5db|DEPTH|activitynet13|pbd_inspired_tf|阻塞|selector='pbd_inspired_tf': implementation pending; estimator='current_model_ablation': implementation pending; active_depth: registered extension pending; activitynet13 raw-video protocol integration pending|
|tr-6e39e019b9d0|BCR|thumos14|BCR_prefix0|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-5f1105bb8d31|BCR|thumos14|BCR_prefix2|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-8b11cb94d570|BCR|thumos14|BCR_prefix4|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-ef2e3fdb32ae|A|thumos14|A_atom2|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-0c92ffd5dcca|A|thumos14|A_atom4|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-21faaf9316dc|A|thumos14|A_atom8|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-1c31a90a1056|B|thumos14|B_atom2|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-fa8e7f4508cb|B|thumos14|B_atom4|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-2db759758b8e|A|thumos14|A_spatialgroup1|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-a96edc439b2f|A|thumos14|A_spatialgroup7|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-b359acaa08c3|B|thumos14|B_rank_interp|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-7f5b2737d0bb|B|thumos14|B_physical_interp|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-a510e36e0524|B|thumos14|B_concat_scatter|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-21bb004fe441|B|thumos14|B_timestamp_attention|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-0058d7382058|A|thumos14|A_no_position|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-ca670215e501|A|thumos14|A_native_position_only|阻塞|geometry='native_position_only': implementation pending|
|tr-30d2d72e2562|A|thumos14|A_physical_bias|阻塞|geometry='physical_bias': implementation pending|
|tr-f87a8f86f5ea|A|thumos14|A_rank_tia_negative_control|阻塞|geometry='rank_tia_negative_control': implementation pending|
|tr-3aa9198ed215|C|thumos14|C_centroid_only|阻塞|geometry='centroid_only': implementation pending|
|tr-cf75b2f06f23|C|thumos14|C_support_no_scale|阻塞|geometry='support_no_scale': implementation pending|
|tr-b09dc6ff6f30|A|thumos14|A_per_clip_equal_quota|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-01727b18d792|A|thumos14|A_global_nonzero_per_clip|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-9c13abe83ced|A|thumos14|A_dynamic_threshold|阻塞|selector='threshold': implementation pending|
|tr-333678e65280|A|thumos14|A_dynamic_pg_only|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-78ac23996337|A|thumos14|A_hardware_cost|阻塞|cost_target='hardware_lut': implementation pending|
|tr-e1b0cde378f6|A|thumos14|A_budget_nozero|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-08dea71e336a|B|thumos14|B_per_clip_equal_quota|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-eb9229c5bc41|B|thumos14|B_global_nonzero_per_clip|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-ffba6492cde6|B|thumos14|B_dynamic_threshold|阻塞|selector='threshold': implementation pending|
|tr-761574de7d6a|B|thumos14|B_dynamic_pg_only|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-d8f778e15311|B|thumos14|B_hardware_cost|阻塞|cost_target='hardware_lut': implementation pending|
|tr-3fe44e64353a|B|thumos14|B_budget_nozero|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-0fb36019d1ad|A|thumos14|A_scout_res80|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-58c6a1877f58|A|thumos14|A_scout_res160|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-3969665d9d21|A|thumos14|A_scout_width64|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-fb64e7aaba04|A|thumos14|A_scout_width256|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-43d97b89a4c3|A|thumos14|A_scout_tstride2|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-cf81c51b1d9b|A|thumos14|A_scout_tstride4|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-e29130dc1162|A|thumos14|A_detail_probe|阻塞|detail_probe_fraction=0.03125: implementation pending|
|tr-5cbb39075fe6|A|thumos14|A_actionness_aux|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-4f9e8f0b10f6|B|thumos14|B_scout_res80|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-59a83f557060|B|thumos14|B_scout_res160|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-7b5a3fb1c38b|B|thumos14|B_scout_width64|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-d326bbe4c45c|B|thumos14|B_scout_width256|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-25b5ff01a4a4|B|thumos14|B_scout_tstride2|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-bc5ea4f5d61a|B|thumos14|B_scout_tstride4|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-11630296f619|B|thumos14|B_detail_probe|阻塞|detail_probe_fraction=0.03125: implementation pending|
|tr-2e46a3bd354f|B|thumos14|B_actionness_aux|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-2483d959753a|A|thumos14|A_est_pg|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-c0e8e69e046d|A|thumos14|A_est_retention_gradient|阻塞|estimator='retention_gradient': implementation pending|
|tr-7d0e1a8bea28|A|thumos14|A_est_zero_gate_probe|阻塞|estimator='zero_gate_probe': implementation pending|
|tr-5d3080485e5e|A|thumos14|A_est_straight_through|阻塞|estimator='straight_through': implementation pending|
|tr-4b9fffb5fdf4|A|thumos14|A_explore_none|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-7390bb08e63e|A|thumos14|A_explore_constant_0.1|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-19a2998312e0|A|thumos14|A_explore_constant_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-43d0f3a777b0|A|thumos14|A_probe_every8|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-614dfea94b1e|A|thumos14|A_probe_every128|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-29090a5df7b5|A|thumos14|A_warmup0|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-232963ad0f43|B|thumos14|B_est_pg|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-d95f228289f5|B|thumos14|B_est_retention_gradient|阻塞|estimator='retention_gradient': implementation pending|
|tr-0fffcd1de428|B|thumos14|B_est_zero_gate_probe|阻塞|estimator='zero_gate_probe': implementation pending|
|tr-3c016a83115d|B|thumos14|B_est_straight_through|阻塞|estimator='straight_through': implementation pending|
|tr-1debef2b8756|B|thumos14|B_explore_none|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-3c08d2e3284d|B|thumos14|B_explore_constant_0.1|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-0948bac79ee9|B|thumos14|B_explore_constant_0.5|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-55a5edc637e6|B|thumos14|B_probe_every8|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-7fc642613e23|B|thumos14|B_probe_every128|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-22a6dc013651|B|thumos14|B_warmup0|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-c2c578494c9d|B|thumos14|B_factor_T1.0_S224|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-1922175fc118|B|thumos14|B_factor_T0.5_S224|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-55fb601ec01e|B|thumos14|B_factor_T1.0_S112|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-5ceed0119bc7|B|thumos14|B_factor_T0.5_S112|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-7f11939e463c|B|thumos14|B_roi1_size112|阻塞|roi_mode='source_crop': implementation pending; roi_count: registered extension pending; roi_size: registered extension pending; roi_trajectory: registered extension pending|
|tr-4179ae90fb1f|B|thumos14|B_roi2_size112|阻塞|roi_mode='source_crop': implementation pending; roi_count: registered extension pending; roi_size: registered extension pending; roi_trajectory: registered extension pending|
|tr-c77329211441|B|thumos14|B_roi1_size160|阻塞|roi_mode='source_crop': implementation pending; roi_count: registered extension pending; roi_size: registered extension pending; roi_trajectory: registered extension pending|
|tr-1908281a0782|B|thumos14|B_roi2_size160|阻塞|roi_mode='source_crop': implementation pending; roi_count: registered extension pending; roi_size: registered extension pending; roi_trajectory: registered extension pending|
|tr-7a5520b4d378|B|thumos14|B_roi1_size224|阻塞|roi_mode='source_crop': implementation pending; roi_count: registered extension pending; roi_size: registered extension pending; roi_trajectory: registered extension pending|
|tr-30f22afe9dc4|B|thumos14|B_roi2_size224|阻塞|roi_mode='source_crop': implementation pending; roi_count: registered extension pending; roi_size: registered extension pending; roi_trajectory: registered extension pending|
|tr-b278e81461dc|B|thumos14|B_roi_fullframe_fallback|阻塞|roi_mode='source_crop': implementation pending; roi_count: registered extension pending; roi_option: registered extension pending; roi_size: registered extension pending|
|tr-3cefe031da95|B|thumos14|B_roi_smooth_trajectory|阻塞|roi_mode='source_crop': implementation pending; roi_count: registered extension pending; roi_option: registered extension pending; roi_size: registered extension pending|
|tr-7a8341178286|B|thumos14|B_roi_resize_lowres_negative_control|阻塞|roi_mode='source_crop': implementation pending; roi_count: registered extension pending; roi_option: registered extension pending; roi_size: registered extension pending|
|tr-d9e6c6de9026|B|thumos14|B_slots1|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-f581918405f7|B|thumos14|B_slots8|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-8c3ac91cf6f1|B|thumos14|B_receiverlayers1|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-7fc5bc501131|B|thumos14|B_receiverlayers4|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-3d9066c3932a|B|thumos14|B_no_null|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-c4d364a4578c|B|thumos14|B_feature_l2|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-cc1fc95f60e2|B|thumos14|B_coarse_overwrite|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-05b459aa6b12|C|thumos14|C_learned_projection|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-f54d79036011|C|thumos14|C_no_scale_embedding|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-9e206b50bb02|C|thumos14|C_coarse_update_no_tia|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-55dbbdefba01|B|thumos14|B_2round_random|阻塞|rounds=2: implementation pending; budget_includes_both_rounds: registered extension pending; second_round_proposal: registered extension pending|
|tr-ecfe8ee10d6e|B|thumos14|B_2round_cheap_boundary_single|阻塞|rounds=2: implementation pending; budget_includes_both_rounds: registered extension pending; second_round_proposal: registered extension pending|
|tr-590192611788|B|thumos14|B_2round_cheap_boundary_pair|阻塞|rounds=2: implementation pending; budget_includes_both_rounds: registered extension pending; second_round_proposal: registered extension pending|
|tr-290efe0eedc1|B|thumos14|B_context_local4_tubelets|阻塞|heavy_context='local4_tubelets': implementation pending|
|tr-73e8aa3c202a|B|thumos14|B_context_cross_parent_selected|阻塞|heavy_context='cross_parent_selected': implementation pending|
|tr-b6c5e2b45fbc|A|thumos14|A_refresh_each_layer|阻塞|route_refresh='each_layer': implementation pending|
|tr-fd4070c04f29|DENSE|fineaction|DENSE_fineaction|阻塞|fineaction raw-video protocol integration pending|
|tr-def36109d345|DENSE|thumos14|DENSE_large|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-b5e2c70a2e87|A|fineaction|A_fineaction|阻塞|fineaction raw-video protocol integration pending|
|tr-ff9d91ad642f|A|thumos14|A_large|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-c2fa75e028b4|B|fineaction|B_fineaction|阻塞|fineaction raw-video protocol integration pending|
|tr-83477ed01edb|B|thumos14|B_large|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-2f010bd28de1|C|fineaction|C_fineaction|阻塞|fineaction raw-video protocol integration pending|
|tr-2412677e174e|C|thumos14|C_large|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-31295879d61c|DENSE|thumos14|DENSE_tridet|阻塞|head='tridet': implementation pending|
|tr-31911470071c|A|thumos14|A_tridet|阻塞|head='tridet': implementation pending|
|tr-111ec00f8cef|B|thumos14|B_tridet|阻塞|head='tridet': implementation pending|
|tr-4d88ef3c2573|A|thumos14|A_query384|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|
|tr-33da9eb9ec7f|B|thumos14|B_query384|接受参数|仍需语义检查、真实资产及本配置 GPU 凭证|

import ast
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_dn_g_configs_preserve_the_frozen_route_contract():
    dn = (ROOT / "configs/adatad/thumos/georoute_p1_dn_seed3407_v001.py").read_text()
    g = (ROOT / "configs/adatad/thumos/georoute_p1_g_seed3407_v001.py").read_text()
    assert 'arm_surface="DN"' in dn and 'route_mode="dense"' in dn
    assert 'arm_surface="G"' in g and 'route_mode="dynamic_scnr"' in g
    assert "seed=3407" in dn and "seed=3407" in g
    assert "official_test_open_allowed=False" in dn + g


def test_georoute_source_views_restores_only_the_frozen_dn_g_input_contract():
    config = (
        ROOT / "configs/adatad/thumos/georoute_adatad_development_base.py"
    ).read_text()
    package = (ROOT / "opentad/datasets/transforms/__init__.py").read_text()
    transform = (ROOT / "opentad/datasets/transforms/georoute.py").read_text()
    tree = ast.parse(transform)

    assert config.count('dict(type="GeoRouteSourceViews"') == 3
    assert "from .georoute import GeoRouteSourceViews" in package
    assert '"GeoRouteSourceViews"' in package
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "GeoRouteSourceViews"
        for node in tree.body
    )
    assert "@PIPELINES.register_module()" in transform
    assert 'GEOROUTE_INPUT_SCHEMA = "georoute_native_source_scout_v1"' in transform
    assert "source_frames.append(np.ascontiguousarray(image))" in transform
    assert 'Image.Resampling.BILINEAR' in transform
    assert 'results[self.output_key] = {"source": source_tensor, "scout": scout_tensor}' in transform
    assert '"source_resized_before_native_patch_gather": False' in transform
    assert '"uses_gt": False' in transform
    assert '"uses_teacher": False' in transform
    assert '"uses_oracle": False' in transform
    assert '"uses_test_evidence": False' in transform
    for retired in (
        "NativeCropSourceViews",
        "ContinuousRoiSourceViews",
        "FullFrameLetterboxView",
    ):
        assert retired not in transform
        assert retired not in package
    assert "torch" not in sys.modules


def test_g_is_true_roi_only_global_dynamic_ragged():
    g = (ROOT / "configs/adatad/thumos/georoute_p1_g_seed3407_v001.py").read_text()
    assert "matched_window_budget = tubelets * 64" in g
    assert "georoute_window_token_budget=matched_window_budget" in g
    assert "georoute_dynamic_roi_modifier_enabled=True" in g
    assert "georoute_dynamic_residual_modifier_enabled=False" in g
    assert 'georoute_branch_calibration_mode="none"' in g
    assert "georoute_absolute_coordinates_enabled=False" in g
    assert "georoute_roi_relative_coordinates_enabled=False" in g
    assert "georoute_geometry_projection_enabled=False" in g
    assert "georoute_geometry_side_channel=False" in g
    assert "window_budget_is_global=True" in g
    assert "unique_physical_selection=True" in g
    assert "g_dynamic_k_t=True" in g and "k_t_zero_allowed=True" in g
    assert 'ragged_execution="true_clip_buckets_without_padding_or_dummy_tokens"' in g


def test_true_ragged_and_no_leakage_contract_is_present():
    source = (ROOT / "opentad/models/backbones/georoute_wrapper.py").read_text()
    adapter = (ROOT / "opentad/models/backbones/vit_adapter.py").read_text()
    assert "true_clip_ragged_no_padding" in source + adapter
    assert "masked-zero carrier" in source
    assert "forward_native_ragged" in source and "forward_native_ragged" in adapter
    for config in ("georoute_p1_dn_seed3407_v001.py", "georoute_p1_g_seed3407_v001.py"):
        text = (ROOT / "configs/adatad/thumos" / config).read_text()
        assert "gt_for_route_allowed=False" in text
        assert "teacher_for_route_allowed=False" in text
        assert "oracle_for_route_allowed=False" in text
        assert "raw_prediction_cache_allowed=False" in text


def test_checkpoint_policy_keeps_recovery_state_and_latest_three():
    checkpoint = (ROOT / "opentad/utils/checkpoint.py").read_text()
    train = (ROOT / "tools/train.py").read_text()
    configs = "".join(
        (ROOT / "configs/adatad/thumos" / name).read_text()
        for name in (
            "georoute_p1_dn_seed3407_v001.py",
            "georoute_p1_g_seed3407_v001.py",
        )
    )
    assert 'pattern = re.compile(r"^recovery_epoch_(\\d+)\\.pth$")' in checkpoint
    assert 'checkpoint_role == "recovery"' in checkpoint
    assert '"checkpoint_role": checkpoint_role or "final"' in checkpoint
    assert "_prune_recovery_checkpoints(save_dir, recovery_keep_latest)" in checkpoint
    assert 'checkpoint_role = "final" if is_final else "recovery"' in train
    assert 'recovery_keep_latest = None if is_final else recovery_contract["keep_latest"]' in train
    assert '"state_dict": model.state_dict()' in checkpoint
    assert '"optimizer": optimizer.state_dict()' in checkpoint
    assert '"scheduler": scheduler.state_dict()' in checkpoint
    assert '"scaler": scaler.state_dict()' in checkpoint
    assert '"training_state": dict(training_state)' in checkpoint
    for field in (
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "sampler_epoch",
        "completed_epoch",
        "next_successful_update_index",
    ):
        assert field in train
    assert 'ZOOMTOKEN_RECOVERY_ARMS = {"DN", "G"}' in train
    assert "same cell" in train
    assert "sealed ZoomToken cells are not resumable" in train
    assert "work_dir must be explicitly bound by --work-dir" in train
    assert 'parser.add_argument(\n        "--work-dir"' in train
    assert "cfg.work_dir = args.work_dir" in train
    assert "requires the frozen AMP/EMA recipe" in train
    assert 'args.seed != recovery_contract["seed"]' in train
    assert 'checkpoint.get("checkpoint_role") != "recovery"' in train
    assert 'scaler.load_state_dict(checkpoint["scaler"])' in train
    assert "bytes(torch.get_rng_state().cpu().tolist())" in train
    assert 'list(local_rng_state["torch_cpu_rng_state"])' in train
    assert configs.count("interval_epochs=5") == 2
    assert configs.count("keep_latest=3") == 2
    assert configs.count("save_final=True") == 2
    assert configs.count('checkpoint_policy="recovery_latest3_plus_final"') == 2


def test_g_auxiliary_losses_follow_the_successful_update_order():
    engine = (ROOT / "opentad/cores/train_engine.py").read_text()
    detector = (
        ROOT / "opentad/models/detectors/actionformer.py"
    ).read_text()
    base_detector = (
        ROOT / "opentad/models/detectors/single_stage.py"
    ).read_text()
    setter = "georoute_backbone.set_successful_update_index(successful_update_index)"
    forward = "losses = model(**data_dict, return_loss=True)"
    rpn_forward = "loc_losses = self.rpn_head.forward_train("
    base_cost = 'losses["cost"] = sum('
    consumer = "auxiliary_losses = auxiliary_consumer("
    preserve_input_masks = "input_masks = masks"
    pad_masks = "x, masks = self.pad_data("
    consume_input_masks = "masks=input_masks,"
    add_to_cost = 'losses["cost"] = losses["cost"] + auxiliary_cost'
    detector_return = "return losses"
    backward = 'scaler.scale(losses["cost"]).backward()'
    assert engine.index(setter) < engine.index(forward) < engine.index(backward)
    assert "consume_training_auxiliary_losses(" not in engine
    assert "consume_training_auxiliary_losses" not in base_detector
    assert detector.index(rpn_forward) < detector.index(base_cost) < detector.index(consumer)
    assert detector.index(consumer) < detector.index(add_to_cost) < detector.index(detector_return)
    assert detector.index(preserve_input_masks) < detector.index(pad_masks)
    assert detector.index(pad_masks) < detector.index("x, masks = self.projection(")
    assert consume_input_masks in detector
    assert "colliding_loss_keys = set(losses).intersection(auxiliary_losses)" in detector
    assert "losses.update(auxiliary_losses)" in detector
    assert "optimizer_update_succeeded = _amp_optimizer_step_was_run(" in engine
    assert "if successful_update_index is not None and optimizer_update_succeeded:" in engine
    assert "successful_update_index += 1" in engine
    assert "return successful_update_index" in engine


def test_dn_and_g_auxiliary_consumption_is_capability_bound():
    engine = (ROOT / "opentad/cores/train_engine.py").read_text()
    detector = (
        ROOT / "opentad/models/detectors/actionformer.py"
    ).read_text()
    base_detector = (
        ROOT / "opentad/models/detectors/single_stage.py"
    ).read_text()
    capability_binding = (
        'getattr(candidate_backbone, "consume_training_auxiliary_losses", None)'
    )
    index_gate = "if successful_update_index is not None:"
    setter = "georoute_backbone.set_successful_update_index(successful_update_index)"
    forward = "losses = model(**data_dict, return_loss=True)"
    assert capability_binding in engine
    assert engine.index(index_gate) < engine.index(setter)
    assert engine.index(setter) < engine.index(forward)
    assert "if georoute_backbone is None:" in engine
    assert "consume_training_auxiliary_losses(" not in engine
    assert (
        'getattr(self.backbone, "consume_training_auxiliary_losses", None)'
        in detector
    )
    assert "if self.training and self.with_backbone" in detector
    assert "if callable(auxiliary_consumer):" in detector
    assert detector.count("consume_training_auxiliary_losses") == 1
    assert "consume_training_auxiliary_losses" not in base_detector


def test_dn_g_retry_skips_no_success_state_and_fails_after_eight_retries():
    engine = (ROOT / "opentad/cores/train_engine.py").read_text()
    train = (ROOT / "tools/train.py").read_text()
    development_base = (
        ROOT / "configs/adatad/thumos/georoute_adatad_development_base.py"
    ).read_text()
    dynamic_base = (
        ROOT / "configs/adatad/thumos/georoute_dynamic_scnr_stage1_base.py"
    ).read_text()
    dn = (ROOT / "configs/adatad/thumos/georoute_p1_dn_seed3407_v001.py").read_text()
    g = (ROOT / "configs/adatad/thumos/georoute_p1_g_seed3407_v001.py").read_text()
    official = (
        ROOT / "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
    ).read_text()

    assert "max_amp_retries_per_batch=None" in engine
    assert "retry_skipped_updates = max_amp_retries_per_batch is not None" in engine
    assert "max_attempts = 1 + max_amp_retries_per_batch" in engine
    assert "for attempt_idx in range(max_attempts):" in engine
    assert engine.index("optimizer.zero_grad()") < engine.index(
        "georoute_backbone.set_successful_update_index("
    )
    assert engine.index("scaler.step(optimizer)") < engine.index("scaler.update()")
    assert "scale_before = scaler.get_scale()" not in engine
    assert "scaler.get_scale() >= scale_before" not in engine
    assert "optimizer.register_step_post_hook(mark_optimizer_step)" in engine
    assert "step_post_hook.remove()" in engine
    assert engine.index("optimizer.register_step_post_hook(mark_optimizer_step)") < engine.index(
        "scaler.step(optimizer)"
    )
    assert engine.index("scaler.step(optimizer)") < engine.index(
        "step_post_hook.remove()"
    )
    retry_exit = "if optimizer_update_succeeded or not retry_skipped_updates:"
    terminal = "if retry_skipped_updates and not optimizer_update_succeeded:"
    assert engine.index(retry_exit) < engine.index(terminal)
    assert engine.index(terminal) < engine.index("successful_update_index += 1")
    assert engine.index(terminal) < engine.index("scheduler.step()")
    assert engine.index(terminal) < engine.index("model_ema.update(model)")
    assert "AMP optimizer update failed after " in engine
    assert '"max_amp_retries_per_batch": 8' in train
    assert '"schedule_and_ema_on_success_only": True' in train
    assert '"fail_on_skipped_update": True' in train
    assert 'recovery_contract["max_amp_retries_per_batch"]' in train
    for field in (
        "require_successful_update_hook=True",
        "schedule_and_ema_on_success_only=True",
        "max_amp_retries_per_batch=8",
        "fail_on_skipped_update=True",
    ):
        assert field in development_base
    assert '_base_ = ["./georoute_adatad_development_base.py"]' in dn
    assert '_base_ = ["./georoute_adatad_development_base.py"]' in dynamic_base
    assert '_base_ = ["./georoute_dynamic_scnr_stage1_base.py"]' in g
    assert "max_amp_retries_per_batch" not in official


def test_backbone_exports_keep_only_the_supported_georoute_wrapper():
    package = (ROOT / "opentad/models/backbones/__init__.py").read_text()
    continuous_roi = ROOT / "opentad/models/backbones/continuous_roi_wrapper.py"
    native_crop = ROOT / "opentad/models/backbones/native_crop_wrapper.py"
    vit_ladder = ROOT / "opentad/models/backbones/vit_ladder.py"
    georoute = ROOT / "opentad/models/backbones/georoute_wrapper.py"

    assert not continuous_roi.exists()
    assert "continuous_roi_wrapper" not in package
    assert "ContinuousRoiBackboneWrapper" not in package
    assert native_crop.is_file()
    assert "NativeCropBackboneWrapper" not in package
    assert not vit_ladder.exists()
    assert "VisionTransformerLadder" not in package
    assert georoute.is_file()
    assert "GeoRouteBackboneWrapper," in package
    assert '"GeoRouteBackboneWrapper"' in package
    assert "GeoRoutePostBackboneAggregationWrapper," in package
    assert '"GeoRoutePostBackboneAggregationWrapper"' in package

    if os.environ.get("ZOOMTOKEN_AMP_RUNTIME_CHECK") == "1":
        from opentad.models.backbones import GeoRouteBackboneWrapper

        assert GeoRouteBackboneWrapper.__module__ == (
            "opentad.models.backbones.georoute_wrapper"
        )


def test_dn_g_builder_dispatches_only_the_supported_georoute_wrapper():
    builder = (ROOT / "opentad/models/builder.py").read_text()
    assert "GeoRouteBackboneWrapper," in builder
    assert "GeoRoutePostBackboneAggregationWrapper," in builder
    assert 'wrapper_type == "georoute_native_packed_v1"' in builder
    assert "return GeoRouteBackboneWrapper(cfg)" in builder
    assert 'wrapper_type == "georoute_postbackbone_sparse_aggregation_v1"' in builder
    assert "return GeoRoutePostBackboneAggregationWrapper(cfg)" in builder
    assert "unsupported backbone custom.wrapper_type" in builder
    assert "NativeCropBackboneWrapper" not in builder
    assert "ContinuousRoiBackboneWrapper" not in builder


def test_official_bc_wrapper_keeps_the_heavy_forward_dense_and_selects_afterward():
    source = (ROOT / "opentad/models/backbones/georoute_wrapper.py").read_text()
    tree = ast.parse(source)
    wrapper_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GeoRoutePostBackboneAggregationWrapper"
    )
    wrapper = ast.get_source_segment(source, wrapper_node)
    assert wrapper is not None
    assert any(
        isinstance(base, ast.Name) and base.id == "BackboneWrapper"
        for base in wrapper_node.bases
    )
    assert 'WRAPPER_TYPE = "georoute_postbackbone_sparse_aggregation_v1"' in wrapper
    assert "return super().forward(frames, masks)" in wrapper
    assert "def unflatten_and_pool_features(self, features, batches, num_segs):" in wrapper
    assert wrapper.index("return super().forward(frames, masks)") < wrapper.index(
        "def unflatten_and_pool_features"
    )
    assert "dense_features = features.reshape(" in wrapper
    assert "selected_features = dense_features.gather(" in wrapper
    assert "self.sparse_adapter = GeoRouteSparseTemporalAdapter" in wrapper
    assert wrapper.index("self.sparse_adapter = GeoRouteSparseTemporalAdapter") < wrapper.index(
        "self.scout = GeoRouteScout"
    )
    assert 'mode="roi"' in wrapper
    assert "self.roi_tokens != 64" in wrapper
    assert "self.scout._encode(scout_input)" in wrapper
    assert "self.scout.geometry_head(" in wrapper
    assert "self.scout.residual_head(" not in wrapper
    assert "use_absolute_coordinates=False" in wrapper
    assert "use_roi_relative_coordinates=False" in wrapper
    assert "use_geometry_projection=False" in wrapper
    for packed_surface in (
        "forward_native_packed",
        "forward_native_ragged",
        "extract_native_tubelets",
        "deterministic_linear_2x",
        "consume_training_auxiliary_losses",
        "dynamic_scnr",
    ):
        assert packed_surface not in wrapper
    assert "efficiency claim" in wrapper
    assert "torch" not in sys.modules


def test_official_bc_configs_differ_only_in_postbackbone_support():
    config_root = ROOT / "configs/adatad/thumos"
    common = (
        config_root / "georoute_official_postbackbone_bc_common_seed42_v001.py"
    ).read_text()
    arm_b = (
        config_root / "georoute_official_b_alltoken_postbackbone_seed42_v001.py"
    ).read_text()
    arm_c = (
        config_root / "georoute_official_c_roi_postbackbone_seed42_v001.py"
    ).read_text()

    assert '_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]' in common
    assignment_names = {
        target.id
        for node in ast.parse(common).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert assignment_names == {
        "_base_",
        "model",
        "optimizer",
        "official_bc_contract",
    }
    assert "dataset =" not in common
    assert "solver =" not in common
    assert "scheduler =" not in common
    assert "workflow =" not in common
    assert "post_processing =" not in common
    assert 'heavy_backbone_execution="untouched_official_dense_videomae_forward"' in common
    assert 'selection_application="post_backbone_pre_aggregation"' in common
    assert 'sparse_adapter="GeoRouteSparseTemporalAdapter_uniform_selected"' in common
    assert "residual_enabled=False" in common
    assert "auxiliary_or_proxy_loss_enabled=False" in common
    assert "adapter_side_channel_enabled=False" in common
    assert "efficiency_claim_allowed=False" in common
    assert common.index('dict(name="sparse_adapter"') < common.index(
        'dict(name="adapter"'
    )
    assert common.count("lr=2e-4, weight_decay=0.05") == 4

    normalized_b = (
        arm_b.replace('official_bc_arm = "B"', 'official_bc_arm = "ARM"')
        .replace('georoute_postbackbone_selection="all"', 'georoute_postbackbone_selection="SUPPORT"')
        .replace(
            'work_dir = "exps/thumos/adatad/georoute_official_b_alltoken_postbackbone_seed42_v001"',
            'work_dir = "WORK_DIR"',
        )
    )
    normalized_c = (
        arm_c.replace('official_bc_arm = "C"', 'official_bc_arm = "ARM"')
        .replace('georoute_postbackbone_selection="roi"', 'georoute_postbackbone_selection="SUPPORT"')
        .replace(
            'work_dir = "exps/thumos/adatad/georoute_official_c_roi_postbackbone_seed42_v001"',
            'work_dir = "WORK_DIR"',
        )
    )
    assert normalized_b == normalized_c
    assert 'georoute_postbackbone_selection="all"' in arm_b
    assert 'georoute_postbackbone_selection="roi"' in arm_c
    assert "torch" not in sys.modules


def test_official_bc_inherits_the_complete_official_training_and_evaluation_recipe():
    official = (
        ROOT / "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
    ).read_text()
    for binding in (
        'dict(type="mmaction.RandomResizedCrop")',
        'dict(type="mmaction.Flip", flip_ratio=0.5)',
        'dict(type="mmaction.ImgAug", transforms="default")',
        'dict(type="mmaction.ColorJitter")',
        "with_cp=True",
        "batch_size=2",
        "amp=True",
        "fp16_compress=True",
        "static_graph=True",
        "ema=True",
        'type="AdamW"',
        "lr=1e-4",
        "weight_decay=0.05",
        'type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=100',
        "use_soft_nms=True",
        "sigma=0.7",
        "max_seg_num=2000",
        "multiclass=True",
        "voting_thresh=0.7",
        "checkpoint_interval=2",
        "val_eval_interval=2",
        "val_start_epoch=40",
        "end_epoch=60",
    ):
        assert binding in official
    train = (ROOT / "tools/train.py").read_text()
    assert "is_final = epoch == max_epoch - 1" in train
    assert "save_checkpoint(" in train
    assert "model_ema=model_ema" in train
    assert "torch" not in sys.modules


def test_official_bc_launcher_submits_only_b_and_c_on_the_frozen_a_identity():
    packet = (
        ROOT / "scripts/run_zoomtoken_official_postbackbone_bc_n16r4.sh"
    ).read_text()
    assert packet.startswith("#!/usr/bin/env bash\n")
    assert 'LINEAGE_BASE="01c58b9f2370e914150cf94d392208a4e211c053"' in packet
    assert 'case "${ARM}" in\n  B)' in packet
    assert "georoute_official_b_alltoken_postbackbone_seed42_v001.py" in packet
    assert "georoute_official_c_roi_postbackbone_seed42_v001.py" in packet
    assert "permits only B or C; A is completed job 1245842" in packet
    assert "e2e_thumos_videomae_s_768x1_160_adapter.py" not in packet
    assert "--nproc_per_node=2" in packet
    assert "tools/train.py \"${CONFIG}\" --seed 42 --id 0" in packet
    assert '"${#visible_gpus[@]}" -eq 2' in packet
    assert "batch_size=" not in packet
    assert "--resume" not in packet
    assert "PRE_RUN" not in packet
    for identity in (
        "thumos14/raw_data/video",
        "thumos14/annotations/thumos_14_anno.json",
        "thumos14/annotations/category_idx.txt",
        "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
    ):
        assert identity in packet
    assert "torch" not in sys.modules


def test_official_prebackbone_bc_materializes_native_support_before_one_heavy_call():
    source = (ROOT / "opentad/models/backbones/georoute_wrapper.py").read_text()
    tree = ast.parse(source)
    wrapper_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GeoRouteBackboneWrapper"
    )
    method_node = next(
        node
        for node in wrapper_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_forward_official_fixed_support"
    )
    method = ast.get_source_segment(source, method_node)
    wrapper = ast.get_source_segment(source, wrapper_node)
    assert method is not None and wrapper is not None
    ordered = (
        "extract_native_tubelets(",
        "self._official_fixed_support_route(",
        "self._gather_selected_native_physical(",
        "self.model.backbone.forward_native_ragged(",
        "self.sparse_adapter.forward_ragged(",
        "deterministic_linear_2x(",
    )
    positions = [method.index(binding) for binding in ordered]
    assert positions == sorted(positions)
    assert method.count("self.model.backbone.forward_native_ragged(") == 1
    assert "forward_native_packed" not in method
    assert "super().forward" not in method
    assert "select_dynamic_global_exact_budget" not in method
    assert '"native_materialization_before_heavy": True' in method
    assert '"residual_enabled": False' in method
    assert '"uses_gt_for_route": False' in method
    assert '"uses_teacher": False' in method
    assert '"uses_oracle": False' in method
    assert '"uses_raw_prediction": False' in method
    assert 'self.official_support == "all_native"' in wrapper
    assert "self.official_roi_tokens" in wrapper
    assert "residual_logits=torch.zeros_like(roi_logits)" in wrapper
    assert "use_absolute_coordinates=False" in method
    assert "use_roi_relative_coordinates=False" in method
    assert "use_geometry_projection=False" in method
    assert "torch" not in sys.modules


def test_official_prebackbone_fixed_support_publishes_one_zero_consumer_contract():
    source = (ROOT / "opentad/models/backbones/georoute_wrapper.py").read_text()
    tree = ast.parse(source)
    wrapper_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GeoRouteBackboneWrapper"
    )
    methods = {
        node.name: ast.get_source_segment(source, node)
        for node in wrapper_node.body
        if isinstance(node, ast.FunctionDef)
    }
    fixed = methods["_forward_official_fixed_support"]
    consumer = methods["consume_training_auxiliary_losses"]
    assert fixed is not None and consumer is not None
    guard = "self._pending_regularization is not None"
    publish = 'self._pending_regularization = {"geometry": output.new_zeros(())}'
    clear = "self._pending_regularization = None"
    consume = 'regularization = self._pending_regularization.pop("geometry")'
    assert fixed.index(guard) < fixed.index("extract_native_tubelets(")
    assert fixed.index("output = deterministic_linear_2x(") < fixed.index(publish)
    assert fixed.index(publish) < fixed.index("return output.to(torch.float32)")
    assert fixed.count(publish) == 1
    assert clear in fixed
    assert '"geometry_regularization_enabled": False' in fixed
    assert consume in consumer
    assert consumer.index(consume) < consumer.index(
        'return {"georoute_geometry_regularization_loss": regularization}'
    )
    assert "torch" not in sys.modules


def test_official_prebackbone_ragged_path_preserves_official_checkpointing():
    source = (ROOT / "opentad/models/backbones/vit_adapter.py").read_text()
    tree = ast.parse(source)
    block_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Block"
    )
    method_node = next(
        node
        for node in block_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_native_ragged"
    )
    method = ast.get_source_segment(source, method_node)
    assert method is not None
    assert "checkpoint_active = bool(self.with_cp and x.requires_grad)" in method
    assert "checkpoint_active and torch.is_grad_enabled()" in method
    assert "cp.checkpoint(_inner_forward, x, use_reentrant=True)" in method
    assert "native ragged execution requires with_cp=False" not in source
    assert method.index("self._ragged_attention_mlp_forward(") < method.index(
        "self.adapter.forward_native_ragged("
    )
    assert "torch" not in sys.modules


def test_official_prebackbone_configs_differ_only_in_physical_support():
    config_root = ROOT / "configs/adatad/thumos"
    common = (
        config_root / "georoute_official_prebackbone_bc_common_seed42_v001.py"
    ).read_text()
    arm_b = (
        config_root / "georoute_official_b_alltoken_prebackbone_seed42_v001.py"
    ).read_text()
    arm_c = (
        config_root / "georoute_official_c_roi_k64_prebackbone_seed42_v001.py"
    ).read_text()

    assert '_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]' in common
    assert 'wrapper_type="georoute_native_packed_v1"' in common
    assert "georoute_official_roi_tokens=64" in common
    assert 'georoute_route_mode="roi"' in common
    assert "georoute_dynamic_roi_modifier_enabled=False" in common
    assert "georoute_dynamic_residual_modifier_enabled=False" in common
    assert "georoute_absolute_coordinates_enabled=False" in common
    assert "georoute_roi_relative_coordinates_enabled=False" in common
    assert "georoute_geometry_projection_enabled=False" in common
    assert "georoute_geometry_side_channel=False" in common
    assert "train=dict(batch_size=1)" in common
    assert "val=dict(batch_size=1)" in common
    assert "test=dict(batch_size=1)" in common
    assert "local_batch_size=1" in common and "global_batch_size=2" in common
    assert "arm_b_tokens_per_tubelet=100" in common
    assert "arm_c_tokens_per_tubelet=64" in common
    assert 'native_materialization="before_any_videomae_heavy_block"' in common
    assert "support_is_only_scientific_difference=True" in common
    assert "dynamic_k_t_enabled=False" in common
    assert "auxiliary_or_proxy_loss_enabled=False" in common
    assert "gt_for_route_allowed=False" in common
    assert "teacher_for_route_allowed=False" in common
    assert "oracle_for_route_allowed=False" in common
    assert "raw_prediction_cache_allowed=False" in common
    assert "dataset =" not in common
    assert "scheduler =" not in common
    assert "workflow =" not in common
    assert "post_processing =" not in common
    assert common.index('dict(name="sparse_adapter"') < common.index(
        'dict(name="adapter"'
    )

    normalized_b = (
        arm_b.replace('official_bc_arm = "B"', 'official_bc_arm = "ARM"')
        .replace(
            'georoute_official_support="all_native"',
            'georoute_official_support="SUPPORT"',
        )
        .replace(
            'work_dir = "exps/thumos/adatad/georoute_official_b_alltoken_prebackbone_seed42_v001"',
            'work_dir = "WORK_DIR"',
        )
    )
    normalized_c = (
        arm_c.replace('official_bc_arm = "C"', 'official_bc_arm = "ARM"')
        .replace(
            'georoute_official_support="roi_k64"',
            'georoute_official_support="SUPPORT"',
        )
        .replace(
            'work_dir = "exps/thumos/adatad/georoute_official_c_roi_k64_prebackbone_seed42_v001"',
            'work_dir = "WORK_DIR"',
        )
    )
    assert normalized_b == normalized_c
    assert "torch" not in sys.modules


def test_official_prebackbone_launcher_freezes_identity_without_recipe_overrides():
    packet = (
        ROOT / "scripts/run_zoomtoken_official_prebackbone_bc_n16r4.sh"
    ).read_text()
    assert packet.startswith("#!/usr/bin/env bash\n")
    assert 'LINEAGE_BASE="01c58b9f2370e914150cf94d392208a4e211c053"' in packet
    assert 'case "${ARM}" in\n  B)' in packet
    assert "georoute_official_b_alltoken_prebackbone_seed42_v001.py" in packet
    assert "georoute_official_c_roi_k64_prebackbone_seed42_v001.py" in packet
    assert "permits only B or C; A is completed job 1245842" in packet
    assert "--nproc_per_node=2" in packet
    assert "tools/train.py \"${CONFIG}\" --seed 42 --id 0" in packet
    assert '"${#visible_gpus[@]}" -eq 2' in packet
    assert "batch_size=" not in packet
    assert "optimizer." not in packet
    assert "scheduler." not in packet
    assert "solver." not in packet
    assert "post_processing." not in packet
    assert "workflow." not in packet
    assert "--resume" not in packet
    assert "PRE_RUN" not in packet
    assert "torch" not in sys.modules


def test_dn_g_import_time_local_module_closure_is_complete_without_torch():
    models_root = ROOT / "opentad/models"
    pending = [models_root / "__init__.py"]
    visited = set()
    missing = []

    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(ROOT).with_suffix("")
        package = list(relative.parts[:-1])

        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                parent_hops = node.level - 1
                module_parts = package[
                    : len(package) - parent_hops if parent_hops else len(package)
                ]
                if node.module:
                    module_parts += node.module.split(".")
            elif node.module and (
                node.module == "opentad.models"
                or node.module.startswith("opentad.models.")
            ):
                module_parts = node.module.split(".")
            else:
                continue

            candidate = ROOT.joinpath(*module_parts)
            module_file = candidate.with_suffix(".py")
            package_file = candidate / "__init__.py"
            if module_file.is_file():
                pending.append(module_file)
            elif package_file.is_file():
                pending.append(package_file)
            else:
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.module}")

    helper_tree = ast.parse(
        (models_root / "backbones/native_crop_wrapper.py").read_text()
    )
    helper_source = (models_root / "backbones/native_crop_wrapper.py").read_text()
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "deterministic_linear_2x"
        for node in helper_tree.body
    )
    assert "0.75 * left + 0.25 * right" in helper_source
    assert "0.25 * left + 0.75 * right" in helper_source
    assert "torch.cat((value[..., :1], between, value[..., -1:]), dim=-1)" in helper_source
    assert missing == []
    assert "torch" not in sys.modules


def test_amp_update_detection_runtime_contract():
    if os.environ.get("ZOOMTOKEN_AMP_RUNTIME_CHECK") != "1":
        return

    import torch

    from opentad.cores.train_engine import _amp_optimizer_step_was_run

    if not torch.cuda.is_available():
        raise RuntimeError("ZoomToken AMP runtime check requires allocated CUDA")

    parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda:0"))
    optimizer = torch.optim.SGD([parameter], lr=0.25)
    scaler = torch.cuda.amp.GradScaler()

    optimizer.zero_grad()
    finite_before = parameter.detach().clone()
    scaler.scale(parameter.square().sum()).backward()
    finite_step_ran = _amp_optimizer_step_was_run(scaler, optimizer)
    scaler.update()
    assert finite_step_ran
    assert not torch.equal(parameter.detach(), finite_before)

    optimizer.zero_grad()
    scaler.scale(parameter.square().sum()).backward()
    parameter.grad.fill_(float("inf"))
    skipped_before = parameter.detach().clone()
    skipped_step_ran = _amp_optimizer_step_was_run(scaler, optimizer)
    scaler.update()
    assert not skipped_step_ran
    assert torch.equal(parameter.detach(), skipped_before)


def test_g_update_index_is_checkpointed_and_restored_without_importing_torch():
    train = (ROOT / "tools/train.py").read_text()
    assert 'recovery_contract["arm_surface"] == "G"' in train
    assert '"next_successful_update_index": next_successful_update_index' in train
    assert "next_successful_update_index = _restore_zoomtoken_training_state(" in train
    assert "next_successful_update_index = train_one_epoch(" in train
    assert "successful_update_index=next_successful_update_index" in train
    assert "next_successful_update_index," in train
    assert "torch" not in sys.modules


def test_original_official_adatad_configs_are_not_recovery_opted_in():
    official = (
        ROOT / "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
    ).read_text()
    assert "zoomtoken_recovery" not in official
    assert "recovery_latest3_plus_final" not in official


def test_shared_official_reproduction_packet_uses_the_untouched_supported_cli():
    packet = (
        ROOT / "scripts/run_adatad_shared_official_reproduction_n16r4.sh"
    ).read_text()
    assert packet.startswith("#!/usr/bin/env bash\n")
    assert 'if [ -z "${BASH_VERSION:-}" ]; then' in packet
    assert "invoke this packet with bash, never sh" in packet
    assert 'EXPECTED_COMMIT="01c58b9f2370e914150cf94d392208a4e211c053"' in packet
    assert "official_adatad_reproduction_01c58b9" in packet
    assert "e2e_thumos_videomae_s_768x1_160_adapter.py" in packet
    assert "tools/train.py \"${CONFIG}\" --seed 42 --id 0" in packet
    assert '"work_dir=${RUN_ROOT}"' in packet
    assert "--work-dir" not in packet
    assert '[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]]' in packet
    assert '"${#visible_gpus[@]}" -eq 2' in packet
    assert "exactly two Slurm-visible GPUs" in packet
    assert "--nproc_per_node=2" in packet
    assert "--nproc_per_node=1" not in packet
    assert "--gpus" not in packet and "CUDA_VISIBLE_DEVICES=" not in packet
    for identity in (
        "thumos14/raw_data/video",
        "thumos14/annotations/thumos_14_anno.json",
        "thumos14/annotations/category_idx.txt",
        "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
    ):
        assert identity in packet


def test_roi60_packet_binds_only_dn_g_to_official_validation_and_recovery():
    packet = (ROOT / "scripts/run_zoomtoken_roi60_dn_g_n16r4.sh").read_text()
    assert packet.startswith("#!/usr/bin/env bash\n")
    assert 'if [ -z "${BASH_VERSION:-}" ]; then' in packet
    assert 'LINEAGE_BASE="bae6462754a3f1bc52da572c3c97444abd96e092"' in packet
    assert "ZOOMTOKEN_ROI60_EXPECTED_COMMIT:?set ZOOMTOKEN_ROI60_EXPECTED_COMMIT" in packet
    assert 'merge-base --is-ancestor "${LINEAGE_BASE}" "${EXPECTED_COMMIT}"' in packet
    assert 'case "${ARM}" in\n  DN)' in packet
    assert "georoute_p1_dn_seed3407_v001.py" in packet
    assert "georoute_p1_g_seed3407_v001.py" in packet
    assert "tools/train.py \"${CONFIG}\" --seed 3407 --id 0 --work-dir" in packet
    assert "--nproc_per_node=1" in packet
    assert '"dataset.train.subset_name=training"' in packet
    assert packet.count('"dataset.val.subset_name=validation"') == 1
    assert packet.count('"dataset.test.subset_name=validation"') == 1
    assert '"evaluation.subset=validation"' in packet
    assert '"evaluation.tiou_thresholds=[0.3,0.4,0.5,0.6,0.7]"' in packet
    for binding in (
        '"post_processing.nms.use_soft_nms=True"',
        '"post_processing.nms.sigma=0.7"',
        '"post_processing.nms.max_seg_num=2000"',
        '"post_processing.nms.multiclass=True"',
        '"post_processing.nms.voting_thresh=0.7"',
        '"workflow.end_epoch=60"',
        '"workflow.checkpoint_interval=5"',
        '"workflow.checkpoint_policy=recovery_latest3_plus_final"',
        '"workflow.val_eval_interval=60"',
        '"workflow.val_start_epoch=59"',
        'resume_args=(--resume "${RESUME}")',
    ):
        assert binding in packet
    assert "recovery_epoch_<N>.pth" in packet
    assert '"zoomtoken_p1_config.gt_for_route_allowed=False"' in packet
    assert '"zoomtoken_p1_config.teacher_for_route_allowed=False"' in packet
    assert '"zoomtoken_p1_config.oracle_for_route_allowed=False"' in packet
    assert "georoute_p1_q_" not in packet
    assert "georoute_p1_f_" not in packet
    assert "georoute_p1_n_" not in packet
    assert "torch" not in sys.modules


def test_g_dynamic_auxiliary_graph_is_owned_by_ddp_forward():
    """Opt-in CPU/Gloo check for the exact shared-head DDP failure shape."""
    if os.environ.get("ZOOMTOKEN_DDP_RUNTIME_CHECK") != "1":
        return

    import tempfile

    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel

    from opentad.models.detectors.actionformer import ActionFormer

    class ToyGeoRouteBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.dynamic_aux_head = nn.Linear(4, 3, bias=False)
            self.hard_output = None
            self.pending = None
            self.consumer_masks = None
            self.consume_count = 0

        def forward(self, inputs):
            if self.pending is not None:
                raise RuntimeError("pending auxiliary losses were not consumed")
            hard_output = inputs.clone()
            self.hard_output = hard_output.detach().clone()
            auxiliary_logits = self.dynamic_aux_head(inputs)
            proxy_logits = self.dynamic_aux_head(inputs * 0.5)
            self.pending = (auxiliary_logits, proxy_logits)
            return hard_output

        def consume_training_auxiliary_losses(self, masks, gt_segments, gt_labels):
            del gt_segments, gt_labels
            self.consumer_masks = masks
            auxiliary_logits, proxy_logits = self.pending
            self.pending = None
            self.consume_count += 1
            return {
                "georoute_geometry_regularization_loss": auxiliary_logits.sum() * 0.0,
                "georoute_dynamic_auxiliary_loss": auxiliary_logits.square().mean(),
                "georoute_dynamic_soft_proxy_loss": proxy_logits.square().mean(),
            }

    class ToyProjection(nn.Module):
        def forward(self, x, masks):
            return x, (masks, masks[:, ::2])

    class ToyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.received_masks = None

        def forward_train(self, x, masks, **kwargs):
            del kwargs
            self.received_masks = masks
            return {
                "cls_loss": x.square().mean(),
                "reg_loss": x.mean().square(),
            }

    detector = ActionFormer.__new__(ActionFormer)
    nn.Module.__init__(detector)
    detector.backbone = ToyGeoRouteBackbone()
    detector.projection = ToyProjection()
    detector.rpn_head = ToyHead()
    detector.max_seq_len = 4
    detector.max_div_factor = 1
    detector.train()
    inputs = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-1.5, 0.25, 2.0, -0.75]],
        dtype=torch.float32,
    )
    masks = torch.ones_like(inputs, dtype=torch.bool)

    with tempfile.TemporaryDirectory() as temporary_directory:
        rendezvous = (Path(temporary_directory) / "ddp_init").resolve().as_uri()
        dist.init_process_group(
            "gloo", init_method=rendezvous, rank=0, world_size=1
        )
        try:
            model = DistributedDataParallel(detector, find_unused_parameters=True)
            for expected_consume_count in (1, 2):
                model.zero_grad(set_to_none=True)
                losses = model(
                    inputs=inputs,
                    masks=masks,
                    metas=[{}, {}],
                    gt_segments=[torch.empty((0, 2)), torch.empty((0, 2))],
                    gt_labels=[
                        torch.empty((0,), dtype=torch.long),
                        torch.empty((0,), dtype=torch.long),
                    ],
                    return_loss=True,
                )
                assert torch.equal(model.module.backbone.hard_output, inputs)
                assert isinstance(model.module.rpn_head.received_masks, tuple)
                assert len(model.module.rpn_head.received_masks) == 2
                assert model.module.backbone.consumer_masks is masks
                assert torch.equal(model.module.backbone.consumer_masks, masks)
                assert model.module.backbone.pending is None
                assert model.module.backbone.consume_count == expected_consume_count
                losses["cost"].backward()
                gradient = model.module.backbone.dynamic_aux_head.weight.grad
                assert gradient is not None
                assert torch.isfinite(gradient).all()
                assert torch.count_nonzero(gradient).item() > 0
                assert {
                    "georoute_geometry_regularization_loss",
                    "georoute_dynamic_auxiliary_loss",
                    "georoute_dynamic_soft_proxy_loss",
                }.issubset(losses)
        finally:
            dist.destroy_process_group()

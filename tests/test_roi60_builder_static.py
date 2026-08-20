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
    setter = "georoute_backbone.set_successful_update_index(successful_update_index)"
    forward = "losses = model(**data_dict, return_loss=True)"
    consumer = "auxiliary_losses = georoute_backbone.consume_training_auxiliary_losses("
    add_to_cost = 'losses["cost"] = losses["cost"] + auxiliary_cost'
    backward = 'scaler.scale(losses["cost"]).backward()'
    assert engine.index(setter) < engine.index(forward)
    assert engine.index(forward) < engine.index(consumer)
    assert engine.index(consumer) < engine.index(add_to_cost)
    assert engine.index(add_to_cost) < engine.index(backward)
    assert engine.count("consume_training_auxiliary_losses(") == 1
    assert "colliding_loss_keys = set(losses).intersection(auxiliary_losses)" in engine
    assert "losses.update(auxiliary_losses)" in engine
    assert "scale_before = scaler.get_scale()" in engine
    assert "optimizer_update_succeeded = scaler.get_scale() >= scale_before" in engine
    assert "if georoute_backbone is not None and optimizer_update_succeeded:" in engine
    assert "successful_update_index += 1" in engine
    assert "return successful_update_index" in engine


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
    assert engine.index("scaler.update()") < engine.index(
        "optimizer_update_succeeded = scaler.get_scale() >= scale_before"
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

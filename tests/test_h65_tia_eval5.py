from pathlib import Path
import ast
import math

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/adatad/thumos/h65_pro"


def test_phase_pair_differs_only_in_allocation():
    on = Config.fromfile(str(CONFIGS / "h65_pro_eval5_phaseon.py"))
    off = Config.fromfile(str(CONFIGS / "h65_pro_eval5_phaseoff.py"))
    on_model, off_model = on.model.to_dict(), off.model.to_dict()
    assert on_model["frame_selector"].pop("acquisition_policy") == "semantic_phase_sampling"
    assert off_model["frame_selector"].pop("acquisition_policy") == "budget_calibrated_sampling_rate"
    assert on_model == off_model
    assert on.solver == off.solver and on.optimizer == off.optimizer
    assert on.scheduler == off.scheduler and on.workflow == off.workflow


def test_all_new_configs_keep_budget_full_test_and_distinct_identity():
    from tools.bata.duca_selected_axis_training import formal_training_contract, H65_PRO_VARIANT_CONFIGS
    for arm in ("uniform", "phaseoff", "phaseon"):
        filename = f"h65_pro_eval5_{arm}.py"
        cfg = Config.fromfile(str(CONFIGS / filename))
        contract = formal_training_contract(cfg)
        assert contract["expected_successful_optimizer_updates"] == 6000
        assert cfg.test_guided_exploratory
        assert cfg.workflow.val_eval_epochs == list(range(5, 61, 5))
        assert cfg.workflow.primary_checkpoint_epoch == 59
        assert cfg.workflow.primary_checkpoint_state_key == "state_dict_ema"
        assert cfg.model.backbone.backbone.total_frames == 384
        assert cfg.model.frame_selector.budget == 384
        assert cfg.dataset.test.test_mode
        assert cfg.dataset.test.block_list is None
        assert H65_PRO_VARIANT_CONFIGS[f"h65_pro_eval5_{arm}"] == filename
    dense = Config.fromfile(str(CONFIGS / "h65_pro_ref_d768.py"))
    assert dense.model.backbone.backbone.total_frames == 768


def test_best_selection_uses_full_precision_and_earliest_tie():
    source = ast.parse((ROOT / "opentad/utils/test_guided_eval.py").read_text())
    code = ast.Module(body=[node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in {"improves", "evaluation_due"}], type_ignores=[])
    namespace = {"math": math}
    exec(compile(code, "selection", "exec"), namespace)
    assert [epoch + 1 for epoch in range(60) if namespace["evaluation_due"](epoch)] == list(range(5, 61, 5))
    best = {"metrics": {"average_mAP": 0.600001}}
    assert namespace["improves"]({"average_mAP": 0.600002}, best)
    assert not namespace["improves"]({"average_mAP": 0.600001}, best)

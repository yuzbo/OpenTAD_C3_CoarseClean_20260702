import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "opentad/utils/test_guided_eval.py"


def pure_functions():
    module = ast.parse(SOURCE.read_text())
    selected = [node for node in module.body if isinstance(node, ast.FunctionDef)
                and node.name in {"evaluation_due", "improves"}]
    import math
    namespace = {"math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def test_full_sixty_epoch_schedule_and_best_score():
    functions = pure_functions()
    assert [epoch + 1 for epoch in range(60) if functions["evaluation_due"](epoch)] == list(range(5, 61, 5))
    improves = functions["improves"]
    assert improves({"average_mAP": 0.5}, None)
    assert improves({"average_mAP": 0.6}, {"metrics": {"average_mAP": 0.5}})
    assert not improves({"average_mAP": 0.5}, {"metrics": {"average_mAP": 0.5}})
    for value in [float("nan"), float("inf"), 65.0]:
        with pytest.raises(ValueError):
            improves({"average_mAP": value}, None)


def test_on_off_configs_are_matched_except_mechanism_and_path():
    from mmengine.config import Config
    configs = [Config.fromfile(str(ROOT / f"configs/adatad/thumos/ettrc_test_guided_{arm}_seed4407.py")).to_dict()
               for arm in ["on", "off"]]
    for cfg, enabled in zip(configs, [True, False]):
        assert cfg["model"]["backbone"]["backbone"].pop("enable_taylor") is enabled
        cfg.pop("work_dir")
        assert cfg["workflow"]["val_eval_interval"] == 5
        assert cfg["workflow"]["val_start_epoch"] == 4
        assert cfg["solver"]["ema"] and not cfg["solver"]["amp"]
    assert configs[0] == configs[1]


def test_periodic_inference_has_no_batch_limit_or_prediction_cache():
    text = SOURCE.read_text()
    assert "not_eval=False, max_batches=None" in text
    assert 'cfg.inference.save_raw_prediction = False' in text
    assert '"model_selection_uses_test": True' in text
    assert '"unseen_test_claim_allowed": False' in text
    assert 'state_dict_ema' in text and 'best_test.pth' in text

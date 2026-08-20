import ast
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_group_builder():
    path = ROOT / "opentad/cores/optimizer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_backbone_optim_groups"
    )
    namespace = {}
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["get_backbone_optim_groups"]


class _Backbone:
    def __init__(self):
        self.named = {
            "sparse_adapter.weight": object(),
            "scout.weight": object(),
            "adapter.weight": object(),
            "stem.weight": object(),
        }

    def named_parameters(self):
        return iter(self.named.items())

    def parameters(self):
        return iter(self.named.values())


class _Logger:
    def info(self, _message):
        return None


def _build(backbone, custom=None):
    cfg = {"lr": 1e-5, "weight_decay": 1e-4}
    if custom is not None:
        cfg["custom"] = [
            {"name": name, "lr": lr, "weight_decay": 0.0}
            for name, lr in custom
        ]
    model = SimpleNamespace(module=SimpleNamespace(backbone=backbone))
    return _load_group_builder()(cfg, model, _Logger())


def _ids(group):
    return {id(parameter) for parameter in group["params"]}


def test_custom_optimizer_groups_are_independent_unique_and_ordered():
    backbone = _Backbone()
    parameter = {name: id(value) for name, value in backbone.named.items()}
    groups = _build(
        backbone,
        [("sparse_adapter", 2e-5), ("scout", 3e-5), ("adapter", 4e-5)],
    )

    assert len(groups) == 4
    assert len({id(group["params"]) for group in groups}) == 4
    grouped_ids = [id(value) for group in groups for value in group["params"]]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(parameter.values())
    rest, sparse_adapter, scout, adapter = groups
    assert _ids(rest) == {parameter["stem.weight"]}
    assert _ids(sparse_adapter) == {parameter["sparse_adapter.weight"]}
    assert _ids(scout) == {parameter["scout.weight"]}
    assert _ids(adapter) == {parameter["adapter.weight"]}
    assert parameter["sparse_adapter.weight"] not in _ids(adapter)

    single = _build(backbone, [("scout", 3e-5)])
    assert len(single) == 2
    assert _ids(single[1]) == {parameter["scout.weight"]}
    assert _ids(single[0]) == set(parameter.values()) - {parameter["scout.weight"]}

    no_custom = _build(backbone)
    assert len(no_custom) == 1
    assert _ids(no_custom[0]) == set(parameter.values())
    assert no_custom[0]["lr"] == 1e-5
    assert no_custom[0]["weight_decay"] == 1e-4
    assert "torch" not in sys.modules

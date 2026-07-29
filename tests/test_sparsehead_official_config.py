import copy

import pytest
import torch

from libs.core import load_config
from libs.modeling import make_meta_arch


@pytest.mark.parametrize(
    "config_path,policy",
    [
        (
            "configs/thumos_i3d_sparsehead_k384_uniform.yaml",
            "stratified_uniform",
        ),
        (
            "configs/thumos_i3d_sparsehead_k384_hash.yaml",
            "video_hash_random",
        ),
    ],
)
def test_sparsehead_config_changes_only_declared_intervention(
    config_path, policy
):
    official = load_config("configs/thumos_i3d.yaml")
    sparse = load_config(config_path)
    intervention = copy.deepcopy(sparse["model"].pop("sparse_head"))
    assert sparse == official
    assert intervention == {
        "enabled": True,
        "budget": 384,
        "policy": policy,
        "hash_seed": 1234567891,
        "training_loss_support": "selected_native_grid_queries",
    }


def test_enabled_sparsehead_requires_explicit_training_loss_support():
    cfg = load_config("configs/thumos_i3d_sparsehead_k384_uniform.yaml")
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["sparse_head"].pop("training_loss_support")
    with pytest.raises(
        ValueError,
        match="training_loss_support=selected_native_grid_queries",
    ):
        make_meta_arch(cfg["model_name"], **model_cfg)


def test_sparse_model_strict_loads_official_state_and_masks_loss_support():
    official_cfg = load_config("configs/thumos_i3d.yaml")
    sparse_cfg = load_config(
        "configs/thumos_i3d_sparsehead_k384_uniform.yaml"
    )
    official = make_meta_arch(
        official_cfg["model_name"], **official_cfg["model"]
    )
    sparse = make_meta_arch(
        sparse_cfg["model_name"], **sparse_cfg["model"]
    )
    assert set(official.state_dict()) == set(sparse.state_dict())
    sparse.load_state_dict(official.state_dict(), strict=True)
    assert (
        sparse.sparse_training_loss_support
        == "selected_native_grid_queries"
    )

    time_size = 4
    num_classes = sparse.num_classes
    selected_mask = torch.tensor([[True, False, False, False]])
    offsets = [torch.zeros(1, time_size, 2)]
    labels = [torch.zeros(time_size, num_classes)]
    target_offsets = [torch.zeros(time_size, 2)]
    logits = torch.zeros(1, time_size, num_classes)

    sparse.loss_normalizer = 100.0
    reference = sparse.losses(
        [selected_mask],
        [logits.clone()],
        offsets,
        labels,
        target_offsets,
    )["cls_loss"]
    unselected_changed = logits.clone()
    unselected_changed[:, 1, :] = 100.0
    sparse.loss_normalizer = 100.0
    ignored = sparse.losses(
        [selected_mask],
        [unselected_changed],
        offsets,
        labels,
        target_offsets,
    )["cls_loss"]
    selected_changed = logits.clone()
    selected_changed[:, 0, :] = 10.0
    sparse.loss_normalizer = 100.0
    observed = sparse.losses(
        [selected_mask],
        [selected_changed],
        offsets,
        labels,
        target_offsets,
    )["cls_loss"]

    torch.testing.assert_close(ignored, reference)
    assert not torch.isclose(observed, reference)

import copy

import pytest

from libs.core import load_config


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
    }

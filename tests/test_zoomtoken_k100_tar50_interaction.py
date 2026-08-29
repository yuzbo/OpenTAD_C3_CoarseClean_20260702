from pathlib import Path

from mmengine import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "georoute_official_amod50_prebackbone_seed42_v001.py"
)
LAUNCHER_PATH = ROOT / "scripts" / "run_zoomtoken_k100_tar50_interaction_n16r4.sh"


def test_k100_tar50_reuses_the_frozen_parameter_free_amod50_route():
    config = Config.fromfile(CONFIG_PATH)
    amod = config.model.backbone.backbone.amod
    contract = config.official_amod_contract

    assert config.official_bc_arm == "AMOD50"
    assert amod.capacity == 0.5
    assert tuple(amod.dense_block_indices) == (0, 2, 4, 6, 8, 10)
    assert tuple(amod.amod_block_indices) == (1, 3, 5, 7, 9, 11)
    assert amod.routing_score == "preceding_dense_attention_column_mean"
    assert amod.unselected_update == "identity_bypass"
    assert config.zoomtoken_p1_config.support == "full_800_token_videomae_grid"
    assert config.zoomtoken_p1_config.selected_tokens_per_amod_block == 400
    assert contract.total_tokens == 800
    assert contract.adapter_execution == "dense_all_tokens"
    assert contract.new_trainable_router is False
    assert contract.auxiliary_loss is False
    assert contract.temporal_cache is False
    assert config.workflow.end_epoch == 60


def test_formal_launcher_binds_reference_and_one_final_ema_evaluation():
    source = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "PRECHECK_READY" in source
    assert "1254040" in source
    assert "3aca10bc3593e301b7d7e77271419b8bb557d8f8b29bead195fa2aa350e34ddd" in source
    assert "0d09e3fec839449923db1158a18ead631e813b9d00cdab051328cb2b407485f3" in source
    assert "81c805838502639d4fb0e6fcdd0848c53ccbd8eeccf7d1501562af2e84d9ac87" in source
    assert "epoch_59.pth" in source
    assert 'checkpoint.get("state_dict_ema")' in source
    assert source.count("tools/train.py") == 1
    assert source.count("tools/test.py") == 1
    assert "--resume" not in source
    assert "retry_resume_replacement\\tfalse" in source
    assert "cost_measurement\\tfalse" in source

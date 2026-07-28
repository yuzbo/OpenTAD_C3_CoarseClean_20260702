from __future__ import annotations

from pathlib import Path

from mmengine.config import Config


def test_phase1_cost_configs_freeze_probe_execution_and_uniform_selection(
    tmp_path,
    monkeypatch,
):
    root = Path(__file__).resolve().parents[1]
    block = tmp_path / "block.txt"
    block.write_text("video_validation_0001\n", encoding="utf-8")
    monkeypatch.setenv("DUCA_RIME_PHASE1_EVAL_BLOCK_LIST", str(block))

    no_probe = Config.fromfile(
        str(
            root
            / "configs/adatad/thumos/duca_rime_no_probe_uniform_phase1_cost.py"
        )
    )
    probe = Config.fromfile(
        str(
            root
            / "configs/adatad/thumos/duca_rime_probe_uniform_phase1_cost.py"
        )
    )

    assert no_probe.model.frame_selector.arm == "exact_uniform"
    assert no_probe.duca_rime_phase1_cost_contract.coarse_probe_executed is False
    assert (
        tuple(no_probe.duca_rime_phase1_cost_contract.checkpoint_drop_prefixes)
        == (
            "frame_selector._loss_weight_schedule_step",
            "frame_selector.adapter.transition_scorer.",
            "frame_selector.raw_actionness_source.",
        )
    )
    assert probe.model.frame_selector.arm == "probe_uniform"
    assert probe.duca_rime_phase1_cost_contract.coarse_probe_executed is True
    assert (
        probe.duca_rime_phase1_cost_contract.probe_output_used_for_selection
        is False
    )
    for cfg in (no_probe, probe):
        assert cfg.duca_rime_phase1_cost_contract.selection_policy == "exact_uniform"
        assert (
            cfg.duca_rime_phase1_cost_contract.paired_checkpoint_identity_required
            is True
        )
        assert cfg.duca_rime_phase1_cost_contract.accuracy_claim_allowed is False
        assert cfg.solver.test.num_workers == 0

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_convergence_runner_is_fixed_to_predeclared_diagnostic_epochs() -> None:
    text = (ROOT / "scripts/run_duca_cellcf_convergence_variant.sh").read_text(
        encoding="utf-8"
    )

    assert "for epoch in 59 89" in text
    assert "DUCA_CELLCF_TRAINING_PROFILE=exposure132" in text
    assert "--expected-checkpoint-epoch \"${epoch}\"" in text
    assert "--checkpoint-state-key state_dict_ema" in text
    assert "best" not in text.lower()


def test_convergence_summary_uses_sealed_terminal_evaluations() -> None:
    text = (ROOT / "scripts/summarize_duca_cellcf_convergence.sh").read_text(
        encoding="utf-8"
    )

    for variant in ("uniform", "transition_beta0", "cellcf"):
        assert (
            f"--evaluation \"{variant}:131=${{RUN_ROOT}}/logs/{variant}/terminal_evaluation.json\""
            in text
        )
    assert "--suite-aggregate \"${AGGREGATE_EVIDENCE}\"" in text
    assert "DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256" in text
    assert text.count("--variant-receipt") == 3
    assert "refusing to overwrite fixed trajectory JSON" in text
    assert "refusing to overwrite fixed trajectory TSV" in text

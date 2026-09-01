from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_duca_coarse_backend_ablation_gpu1.sh"


def test_coarse_backend_launcher_is_a_matched_four_backend_ablation():
    text = LAUNCHER.read_text(encoding="utf-8")

    for backend in (
        "official_ms_tcn2",
        "official_asformer",
        "official_fact",
        "official_video_mamba_asformer",
    ):
        assert backend in text

    assert '--epochs "${EPOCHS}"' in text
    assert 'EPOCHS="${EPOCHS:-20}"' in text
    assert 'SEED="${SEED:-3407}"' in text
    assert 'PROBE_WINDOW_SIZE="${PROBE_WINDOW_SIZE:-768}"' in text
    assert 'PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"' in text
    assert '--val-every-epochs "${EPOCHS}"' in text
    assert '--early-stop-patience 0' in text
    assert '--official-action-seg-backends "${BACKEND}"' in text
    assert "asformer_lite" not in text
    assert "temporal_mamba_lite" not in text


def test_coarse_backend_launcher_respects_slurm_gpu_mapping():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in text
    assert "torch.cuda.device_count() != 1" in text
    assert "export CUDA_VISIBLE_DEVICES" not in text
    assert "CUDA_VISIBLE_DEVICES=" not in text

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generated_sbatch_scripts_propagate_training_failures():
    for relative_path in (
        "scripts/submit_single_seed_matrix.sh",
        "scripts/queue_auto_dispatcher.sh",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert source.count("set -euo pipefail") >= 2
        generated_body = source[source.index("cat <<SBATCH_EOF") :]
        assert "\nsource /etc/profile\nset -euo pipefail\nmodule load" in generated_body


def test_queue_dispatcher_is_exact_checkout_and_restart_safe():
    source = (ROOT / "scripts/queue_auto_dispatcher.sh").read_text(encoding="utf-8")
    assert 'status --porcelain' in source
    assert 'QUEUE_FILE="${QUEUE_FILE:-}"' in source
    assert 'Already submitted: JobID=' in source
    assert "printf '%s\\t%s\\n' \"$ITEM\" \"$JOB_ID\" >> \"$STATE_FILE\"" in source


def test_full_window_temporal_auxiliaries_align_with_flattened_clips():
    import torch

    from opentad.models.backbones.backbone_wrapper import _align_temporal_auxiliary_to_clips

    boundary_prior = torch.arange(2 * 384, dtype=torch.float32).reshape(2, 384)
    aligned_prior = _align_temporal_auxiliary_to_clips(
        boundary_prior,
        flattened_batch=48,
        clip_len=16,
        tubelet_size=2,
        name="boundary_prior",
        reduce_frame_values=True,
    )
    expected_prior = boundary_prior.reshape(2, 24, 8, 2).amax(dim=-1).reshape(48, 8)
    assert torch.equal(aligned_prior, expected_prior)

    tubelet_delta_t = torch.arange(2 * 192, dtype=torch.float32).reshape(2, 192)
    aligned_delta_t = _align_temporal_auxiliary_to_clips(
        tubelet_delta_t,
        flattened_batch=48,
        clip_len=16,
        tubelet_size=2,
        name="delta_t",
    )
    assert torch.equal(aligned_delta_t, tubelet_delta_t.reshape(48, 8))

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
        assert "\nset -euo pipefail\nsource /etc/profile" in generated_body


def test_queue_dispatcher_is_exact_checkout_and_restart_safe():
    source = (ROOT / "scripts/queue_auto_dispatcher.sh").read_text(encoding="utf-8")
    assert 'status --porcelain' in source
    assert 'QUEUE_FILE="${QUEUE_FILE:-}"' in source
    assert 'Already submitted: JobID=' in source
    assert "printf '%s\\t%s\\n' \"$ITEM\" \"$JOB_ID\" >> \"$STATE_FILE\"" in source

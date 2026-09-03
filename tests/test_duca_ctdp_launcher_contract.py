from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "run_duca_ct_dp_revised_thumos_gpu.sh"
SUBMITTER = ROOT / "scripts" / "submit_duca_ctdp_corrected_campaign_n16r4.sh"


def test_ctdp_outputs_are_kept_outside_the_clean_checkout():
    text = RUNNER.read_text(encoding="utf-8")
    assert 'WORK_DIR="${RUN_ROOT}/${RUN_NAME}"' in text
    assert '--cfg-options work_dir="${WORK_DIR}"' in text
    assert '"${REPO_ROOT}/logs"' not in text


def test_ctdp_submission_is_throttled_and_persistently_recorded():
    text = SUBMITTER.read_text(encoding="utf-8")
    assert "retrying in 60 seconds" in text
    assert "submission_registry.tsv" in text
    assert '[[ ! -s "${REGISTRY}" ]]' in text
    assert '--account="${ACCOUNT}"' in text
    assert 'CTDP_RUN_ROOT="${RUN_ROOT}"' in text
    assert '--wrap="bash -lc' in text

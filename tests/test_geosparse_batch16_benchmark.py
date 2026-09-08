import copy
from pathlib import Path

import pytest

from geosparse_ext.benchmark import validate_benchmark_protocol
from geosparse_research.batch16_benchmark import amended_job, load_measurement_module


def original_job():
    return dict(job_id="benchmark-original", kind="benchmark", batches=[1, 8, 32],
                exclusive=True, warmup=50, repeats=200, seed=0, video_subset_size=100,
                source_train_id="train-original", depends_on=["train-original"],
                modes=["device_model", "decoded_tensor_to_output", "encoded_video_to_output"],
                implementation_pairs=["reference", "optimized"])


def test_supplement_changes_only_batch_and_explicit_identity():
    original = original_job()
    before = copy.deepcopy(original)
    amended = amended_job(original)
    assert original == before and validate_benchmark_protocol(original) == 1
    assert validate_benchmark_protocol(amended) == 16
    assert amended["batches"] == [16] and amended["job_id"] != original["job_id"]
    assert amended["amended_from_job"] == original["job_id"]
    for key in set(original) - {"job_id", "batches"}:
        assert amended[key] == original[key]


@pytest.mark.parametrize("change", [dict(batches=[16]), dict(warmup=5), dict(repeats=20),
                                   dict(exclusive=False), dict(implementation_pairs=["reference"]),
                                   dict(modes=["device_model"]), dict(benchmark_amendment="unknown")])
def test_unregistered_measurement_changes_are_rejected(change):
    job = original_job()
    job.update(change)
    with pytest.raises(ValueError):
        validate_benchmark_protocol(job)


def test_supplement_cannot_rerun_completed_small_batches():
    job = amended_job(original_job())
    job["batches"] = [1, 8, 16]
    with pytest.raises(ValueError):
        validate_benchmark_protocol(job)


def test_measurement_file_reuses_loaded_model_runtime_and_data():
    import geosparse_ext.runtime as runtime
    import geosparse_ext.data as data
    module = load_measurement_module(Path(__file__).parents[1] / "geosparse_ext/benchmark.py")
    assert module.load_trained_model is runtime.load_trained_model
    assert module.require_gpu is runtime.require_gpu
    assert module.video_batch is data.video_batch
    assert module.validate_benchmark_protocol(amended_job(original_job())) == 16

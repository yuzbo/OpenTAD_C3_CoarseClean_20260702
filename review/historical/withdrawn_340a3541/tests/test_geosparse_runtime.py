import json
import pytest
import torch
import numpy as np
from geosparse_ext.matrix import compile_all
from geosparse_ext.entry import register
from geosparse_ext.protocol import implementation_blockers
from geosparse_ext.records import prepare_split
from geosparse_ext.metrics import detection_risks


def test_registry_inherits_child_configuration_and_retains_every_job(tmp_path):
    jobs, _ = compile_all()
    manifest = tmp_path / "experiments.jsonl"
    manifest.write_text("\n".join(json.dumps(j) for j in jobs))
    result = register(manifest, {}, tmp_path)
    assert result["counts"] == dict(train=609, evaluate=609, benchmark=203, diagnostic=117)
    assert len(result["jobs"]) == 1538
    assert all(state["status"] != "DONE" for state in result["jobs"].values())
    baseline = next(j for j in jobs if j["label"] == "A_full" and j["dataset"] == "thumos14")
    assert result["jobs"][baseline["job_id"]]["implementation"] == []
    assert all(not j["depends_on"] for j in jobs if j["kind"] == "train")


def test_registered_extensions_are_rejected_until_implemented():
    jobs, _ = compile_all()
    pending = [j for j in jobs if j["kind"] == "train" and j["model"].get("rounds") == 2]
    assert pending and all(implementation_blockers(j["model"]) for j in pending)


def test_split_is_disjoint_and_rejects_changed_annotations(tmp_path):
    database = {f"v{i:03}": dict(subset="training", duration=i + 10,
                annotations=[dict(label="a", segment=[1., 2.])]) for i in range(200)}
    database["heldout"] = dict(subset="validation", duration=10., annotations=[])
    annotations = tmp_path / "source.json"
    annotations.write_text(json.dumps(dict(database=database)))
    split = prepare_split(annotations, tmp_path / "protocol")
    groups = [set(split[key]) for key in ("training", "internal_dev", "internal_diagnostic", "validation")]
    assert [len(g) for g in groups] == [180, 12, 8, 1]
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert prepare_split(annotations, tmp_path / "protocol") == split
    database["v000"]["annotations"][0]["segment"][1] = 3.
    annotations.write_text(json.dumps(dict(database=database)))
    with pytest.raises(ValueError, match="changed"):
        prepare_split(annotations, tmp_path / "protocol")


def test_missed_actions_remain_in_boundary_risk_denominator():
    annotation = dict(database={"v": dict(annotations=[dict(label="a", segment=[0., 1.]), dict(label="a", segment=[3., 4.])])})
    result = detection_risks(annotation, dict(results={"v": [dict(label="a", segment=[0., 1.], score=.9)]}), ["v"], 1.)
    assert result["short_recall"] == .5 and result["misses"] == 1
    assert result["gt_count"] == 2 and result["start_mae_seconds"] == 0.


def test_actual_cpu_nms_extension_is_executable():
    from opentad.models.utils.post_processing import batched_nms
    segments, scores, labels = batched_nms(torch.tensor([[0., 1.], [.01, 1.01]]), torch.tensor([.9, .8]),
                                         torch.tensor([0, 0]), use_soft_nms=True, sigma=.7, multiclass=True)
    assert len(segments) and torch.isfinite(scores).all()


def test_vfr_sampling_uses_source_pts_before_native_pairing():
    from geosparse_ext.data import nearest_physical_frames
    source = np.array([0., .04, .08, .3, .34, .38])
    selected = nearest_physical_frames(source, np.array([.08, .12, .28, .33]))
    assert selected.tolist() == [2, 2, 3, 4]
    # Repeated source observations remain explicit; no interval hull is invented.
    assert source[selected].tolist() == [.08, .08, .3, .34]


def test_trainable_checkpoint_reconstructs_exact_model_and_ema(tmp_path):
    import copy
    from geosparse_ext.runtime import TrainableEMA, save_checkpoint, restore_mutable_state
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model[0].requires_grad_(False)
    model.register_buffer("updates", torch.tensor(0))
    model.register_buffer("derived_cache", torch.ones(2), persistent=False)
    base = copy.deepcopy(model)
    model.minibatch = 7
    ema = TrainableEMA(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    x = torch.randn(2, 3)
    model(x).square().sum().backward()
    optimizer.step()
    scheduler.step()
    model.updates.add_(1)
    ema.update(model)
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(path, model, ema, optimizer, scheduler, scaler, 5, {"weights_sha256": "test-base"}, 1)
    saved = torch.load(path)
    assert "0.weight" not in saved["state_dict"] and saved["completed_epochs"] == 6
    restore_mutable_state(base, saved, "state_dict")
    torch.testing.assert_close(base(x), model(x), atol=0, rtol=0)
    assert base.updates.item() == 1
    restored_optimizer = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=.01)
    restored_optimizer.load_state_dict(saved["optimizer"])
    # A resumed update must follow exactly the same Adam moments.
    for current, opt in [(model, optimizer), (base, restored_optimizer)]:
        opt.zero_grad()
        current(x).sum().backward()
        opt.step()
    torch.testing.assert_close(base(x), model(x), atol=0, rtol=0)
    restore_mutable_state(base, saved, "state_dict_ema")
    for name, value in ema.shadow.items():
        torch.testing.assert_close(base.state_dict()[name], value, atol=0, rtol=0)
    saved["state_dict"].pop("1.bias")
    with pytest.raises(ValueError, match="missing"):
        restore_mutable_state(base, saved, "state_dict")


def test_latency_units_and_round_robin_video_coverage():
    from geosparse_ext.benchmark import timing_summary, video_round_robin
    summary = timing_summary([10., 20., 30.], 8)
    assert summary["p50_ms"] == 20. and summary["throughput_windows_per_second"] == 400.
    rows = [["long"]] * 10 + [["short"], ["medium"], ["medium"]]
    names, indices = video_round_robin(rows, 3, 0)
    assert {rows[i][0] for i in indices[:3]} == set(names)
    assert len(indices) == len(rows)
    with pytest.raises(ValueError, match="needs"):
        video_round_robin(rows, 100, 0)


def test_gpu_guard_checks_physical_allocation_and_container_mapping(monkeypatch):
    from geosparse_ext.runtime import require_gpu
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu")
    monkeypatch.setenv("SLURM_JOB_GPUS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    require_gpu()
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    with pytest.raises(RuntimeError, match="physical GPU1"):
        require_gpu()


def test_a100_uses_slurm_allocation_without_hardcoding_physical_gpu1(monkeypatch):
    from geosparse_ext.runtime import require_gpu
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "a100x")
    monkeypatch.setenv("SLURM_JOB_GPUS", "5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    require_gpu()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    with pytest.raises(RuntimeError, match="one Slurm-allocated GPU"):
        require_gpu()


def test_capability_covers_same_configuration_seeds_without_unlocking_other_arms():
    from geosparse_ext.capabilities import supported_training_jobs
    jobs, _ = compile_all()
    checked = next(j for j in jobs if j["kind"] == "train" and j["route"] == "B" and j["family"] == "F01"
                   and j["dataset"] == "thumos14" and j["seed"] == 0)
    precheck = dict(is_mock=False, completed_epochs=0, routes={
        "A": dict(status="FAIL", reason="unit-test failure"),
        "B": dict(status="PASS", job_id=checked["job_id"], optimizer_updates=1, ema_update="PASS")})
    supported = supported_training_jobs(jobs, precheck)
    assert len(supported) == 3 and {j["seed"] for j in supported} == {0, 1, 2}
    assert all(j["model"] == checked["model"] and j["route"] == "B" for j in supported)
    precheck["routes"]["B"].pop("optimizer_updates")
    assert supported_training_jobs(jobs, precheck) == []


def test_slurm_gpu1_resource_filter_and_fair_route_interleaving():
    from geosparse_ext.slurm_queue import free_gpu1_nodes, fair_order
    text = """NodeName=g1 CPUAlloc=48 CPUTot=64 GresUsed=gpu:6(IDX:0,2-4,6-7) State=MIXED rest
NodeName=g2 CPUAlloc=56 CPUTot=64 GresUsed=gpu:7(IDX:0-6) State=MIXED rest
NodeName=g3 CPUAlloc=0 CPUTot=64 GresUsed=gpu:0(IDX:N/A) State=IDLE rest
"""
    assert free_gpu1_nodes(text) == ["g1"]
    jobs = [dict(family="F01", route=route, kind="train", seed=seed) for route in ["A", "B", "C"] for seed in [0, 1, 2]]
    ordered = fair_order(jobs)
    assert [j["route"] for j in ordered[:3]] == ["A", "B", "C"]


def test_single_seed_phase_defers_repeats_before_capability_checks(monkeypatch, tmp_path):
    from geosparse_ext import slurm_queue
    jobs, _ = compile_all()
    calls = []
    monkeypatch.setattr(slurm_queue, "job_blockers", lambda job, bindings: (calls.append(job["job_id"]) or [], []))
    monkeypatch.setattr(slurm_queue, "capabilities_for", lambda job, bindings: ({}, []))
    by_id = {job["job_id"]: job for job in jobs}
    states = {job["job_id"]: dict(status="DONE") for job in jobs}
    for job in jobs:
        status, _ = slurm_queue.readiness(job, by_id, {"active_seeds": [0]}, tmp_path, states)
        assert status == ("QUEUED_RESOURCE" if job["seed"] == 0 else "DEFERRED_SEED")
    assert len(calls) == sum(job["seed"] == 0 for job in jobs)
    repeated = next(job for job in jobs if job["kind"] == "train" and job["seed"] == 1)
    status, _ = slurm_queue.readiness(repeated, by_id, {}, tmp_path, states)
    assert status == "QUEUED_RESOURCE"  # Restoring the full phase does not lose the task.


def test_host_assignment_applies_to_training_and_its_artifacts(tmp_path):
    from geosparse_ext.slurm_queue import readiness, worker_script
    jobs, _ = compile_all()
    by_id = {job["job_id"]: job for job in jobs}
    excluded = next(job for job in jobs if job["kind"] == "train" and job["seed"] == 0)
    children = [job for job in jobs if job.get("source_train_id") == excluded["job_id"]]
    for job in [excluded, *children]:
        assert readiness(job, by_id, {"active_seeds": [0], "assigned_training_ids": []}, tmp_path, {})[0] == "DEFERRED_HOST"
    script = worker_script(tmp_path/'job.json', tmp_path, tmp_path/'bindings.json',
                           dict(cluster="A100", repo_root="/isolated/repo", entrypoint=["/isolated/env/bin/python"]))
    assert "--partition=a100x" in script and "--gres=gpu:a100:1" in script
    assert script.index("#SBATCH --mem=64G") < script.index("source /etc/profile")
    assert "CUDA_VISIBLE_DEVICES=" not in script and "--nodelist" not in script
    # conda-pack activation reads an initially unset CONDA_PREFIX on A100.
    assert script.index("source /isolated/env/bin/activate") < script.index("set -u")
    assert "set -euo" not in script and "PYTHONNOUSERSITE=1" in script


def test_per_configuration_certification_and_failed_rerun_withdrawal(tmp_path):
    from geosparse_ext.capabilities import certify
    jobs, _ = compile_all()
    checked = next(j for j in jobs if j["kind"] == "train" and j["route"] == "B" and j["family"] == "F01" and j["seed"] == 0 and j["dataset"] == "thumos14")
    row = dict(status="PASS", job_id=checked["job_id"], optimizer_updates=1, ema_update="PASS", internal_dev_validation=dict(status="PASS"))
    path = tmp_path / "precheck.json"
    precheck = dict(source_commit="unit-fixture", is_mock=False, completed_epochs=0, routes={checked["job_id"]: row})
    path.write_text(json.dumps(precheck))
    bindings = dict(source_commit="unit-fixture", capability_dir=str(tmp_path / "capabilities"))
    certify(jobs, path, bindings)
    capability = json.loads((tmp_path / "capabilities/evaluation.json").read_text())
    assert capability["ready"] and len(capability["supported_variants"]) == 3
    row["status"] = "FAIL"
    path.write_text(json.dumps(precheck))
    certify(jobs, path, bindings)
    capability = json.loads((tmp_path / "capabilities/evaluation.json").read_text())
    assert not capability["ready"] and capability["supported_variants"] == []


def test_benchmark_queries_the_allocated_visible_gpu(monkeypatch):
    import os
    from geosparse_ext import benchmark
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        return str(os.getpid()) if "--query-compute-apps=pid" in command else "GPU-unit, A100, driver, memory, clock, power"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setattr(benchmark.subprocess, "check_output", query)
    benchmark.isolated_gpu()
    assert all(command[command.index("-i") + 1] == "5" for command in calls)

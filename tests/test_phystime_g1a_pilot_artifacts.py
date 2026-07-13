import ast
import hashlib
import importlib.util
import json
import math
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

import tools.bata.validate_phystime_g1a_pilot_artifacts as validator
from mmengine.config import Config
from tools.bata.run_phystime_g1a_real_gate import _canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METRICS = {
    "average_mAP": 1.0,
    "mAP@0.3": 1.0,
    "mAP@0.4": 1.0,
    "mAP@0.5": 1.0,
    "mAP@0.6": 1.0,
    "mAP@0.7": 1.0,
}
VALID_CHECKPOINT_BYTES = b"fake-valid-torch-checkpoint"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class _FakeTensor:
    def __init__(self, values):
        self.values = tuple(values)


class _FakeFiniteMask:
    def __init__(self, finite):
        self.finite = finite

    def all(self):
        return self

    def item(self):
        return self.finite


class _FakeTorch(types.ModuleType):
    def __init__(self, checkpoint):
        super().__init__("torch")
        self.checkpoint = checkpoint
        self.load_calls = []

    def load(self, path, map_location=None):
        path = Path(path)
        self.load_calls.append((path, map_location))
        if path.read_bytes() != VALID_CHECKPOINT_BYTES:
            raise RuntimeError("invalid torch checkpoint")
        return self.checkpoint

    @staticmethod
    def is_tensor(value):
        return isinstance(value, _FakeTensor)

    @staticmethod
    def isfinite(value):
        return _FakeFiniteMask(all(math.isfinite(item) for item in value.values))


@pytest.fixture
def pilot_artifacts(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    work_dir = run_dir / "work_dir" / "gpu1_id0"
    ground_truth_path = run_dir / "formal_ground_truth.json"
    _write_json(
        ground_truth_path,
        {
            "database": {
                "video_test_0000001": {
                    "subset": "validation",
                    "annotations": [{"segment": [1.0, 2.0], "label": "A"}],
                }
            }
        },
    )
    config_path = run_dir / "formal_config.py"
    config_path.write_text(
        "evaluation = dict(\n"
        "    type='mAP',\n"
        "    subset='validation',\n"
        "    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],\n"
        f"    ground_truth_filename={str(ground_truth_path)!r},\n"
        "    thread=1,\n"
        ")\n",
        encoding="utf-8",
    )
    config_sha256 = _canonical_sha256(
        Config.fromfile(config_path, lazy_import=False).to_dict()
    )
    gate_path = run_dir / "real_gate.json"
    contract_path = run_dir / "contract.json"
    static_g0_path = run_dir / "static_g0.json"
    bound_config_sha256 = {
        "selected_axis": config_sha256,
        "physical_metric": "e" * 64,
    }
    _write_json(
        contract_path,
        {
            "schema_version": "phystime_g1a_track_contract_v3",
            "contract_pass": True,
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "config_sha256": bound_config_sha256,
        },
    )
    _write_json(
        static_g0_path,
        {
            "schema_version": "phystime_g0_native_geometry_static_precheck_v2",
            "static_precheck_pass": True,
            "gate_pass": False,
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "config_sha256": bound_config_sha256,
        },
    )
    pretrained_checkpoint_path = run_dir / "videomae_pretrain.pth"
    pretrained_checkpoint_path.write_bytes(b"fake-videomae-pretrain")
    pretrained_checkpoint_sha256 = hashlib.sha256(
        pretrained_checkpoint_path.read_bytes()
    ).hexdigest()
    checkpoint_path = work_dir / "checkpoint" / "epoch_5.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(VALID_CHECKPOINT_BYTES)
    contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    static_g0_sha256 = hashlib.sha256(static_g0_path.read_bytes()).hexdigest()
    dataset_manifest_sha256 = "d" * 64
    gate_payload = {
        "schema_version": "phystime_g1a_real_gate_v3",
        "gate_pass": True,
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "evaluation_ground_truth_filename": str(ground_truth_path),
        "checkpoint_sha256": pretrained_checkpoint_sha256,
        "contract_sha256": contract_sha256,
        "static_g0_sha256": static_g0_sha256,
        "variants": {
            "selected_axis": {"canonical_config_sha256": config_sha256},
            "physical_metric": {"canonical_config_sha256": "e" * 64},
        },
    }
    _write_json(gate_path, gate_payload)
    manifest = {
        "schema_version": "phystime_g1a_pilot_manifest_v3",
        "pilot_epochs": 6,
        "warmup_epochs": 5,
        "started_at_unix": 0.0,
        "commit": "a" * 40,
        "git_tree": "b" * 40,
        "runtime_root": str(run_dir),
        "variant": "selected_axis",
        "config": str(config_path),
        "config_sha256": config_sha256,
        "checkpoint": str(pretrained_checkpoint_path),
        "checkpoint_sha256": pretrained_checkpoint_sha256,
        "gate": str(gate_path),
        "gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        "contract": str(contract_path),
        "contract_sha256": contract_sha256,
        "static_g0": str(static_g0_path),
        "static_g0_sha256": static_g0_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "ground_truth_filename": str(run_dir / "manifest_must_not_choose_gt.json"),
        "metrics": {key: 0.0 for key in REQUIRED_METRICS},
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    predictions = {
        "evaluation_epoch": 5,
        "results": {
            "video_test_0000001": [{"segment": [1.0, 2.0], "label": "A", "score": 0.5}]
        },
    }
    _write_json(work_dir / "result_detection.json", predictions)
    _write_json(work_dir / "evaluation_metrics.json", {"evaluation_epoch": 5, **REQUIRED_METRICS})

    checkpoint = {
        "epoch": 5,
        "state_dict": {"backbone.weight": _FakeTensor([1.0, -1.0])},
        "state_dict_ema": {"backbone.weight": _FakeTensor([1.0, -1.0])},
        "optimizer": {
            "state": {0: {"step": 6, "exp_avg": _FakeTensor([0.0])}},
            "param_groups": [{"lr": 1.0e-4, "params": [0]}],
        },
        "scheduler": {"last_epoch": 6, "_last_lr": [1.0e-4]},
    }
    fake_torch = _FakeTorch(checkpoint)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    evaluator_calls = []

    class _Evaluator:
        def evaluate(self):
            return dict(REQUIRED_METRICS)

    def fake_build_evaluator(cfg):
        evaluator_calls.append(cfg)
        return _Evaluator()

    monkeypatch.setattr(validator, "build_evaluator", fake_build_evaluator, raising=False)
    gate_validation_calls = []

    def fake_validate_gate_report(payload):
        gate_validation_calls.append(payload)
        return True

    monkeypatch.setattr(
        validator, "validate_gate_report", fake_validate_gate_report, raising=False
    )
    monkeypatch.setattr(
        validator,
        "build_dataset_manifest",
        lambda cfg, ground_truth: ({"database": "bound"}, dataset_manifest_sha256),
        raising=False,
    )

    def fake_git_output(command, cwd=None, text=None):
        if command[-2:] == ["status", "--porcelain"]:
            return ""
        if command[-1] == "HEAD":
            return "a" * 40 + "\n"
        if command[-1] == "HEAD^{tree}":
            return "b" * 40 + "\n"
        raise AssertionError(command)

    monkeypatch.setattr(validator.subprocess, "check_output", fake_git_output)
    return types.SimpleNamespace(
        run_dir=run_dir,
        work_dir=work_dir,
        manifest=manifest,
        config_path=config_path,
        ground_truth_path=ground_truth_path,
        predictions=predictions,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        gate_path=gate_path,
        gate_payload=gate_payload,
        contract_path=contract_path,
        static_g0_path=static_g0_path,
        fake_torch=fake_torch,
        evaluator_calls=evaluator_calls,
        gate_validation_calls=gate_validation_calls,
    )


def test_pilot_artifact_validator_accepts_recomputed_formal_artifacts(pilot_artifacts):
    output = pilot_artifacts.run_dir / "PILOT_COMPLETE.json"

    completion = validator.validate_pilot_artifacts(pilot_artifacts.run_dir, output=output)

    assert completion["validation_pass"] is True
    assert completion["prediction_count"] == 1
    assert completion["metrics"] == pytest.approx(REQUIRED_METRICS)
    assert completion["artifacts"]["checkpoint"]["sha256"]
    assert output.is_file()
    assert pilot_artifacts.fake_torch.load_calls == [(pilot_artifacts.checkpoint_path, "cpu")]
    assert completion["manifest_bindings"]["checkpoint"] == str(
        pilot_artifacts.pretrained_checkpoint_path
    )
    assert completion["artifacts"]["checkpoint"]["path"] == str(
        pilot_artifacts.checkpoint_path
    )
    assert (
        completion["manifest_bindings"]["checkpoint"]
        != completion["artifacts"]["checkpoint"]["path"]
    )
    assert len(pilot_artifacts.evaluator_calls) == 1
    assert pilot_artifacts.gate_validation_calls == [pilot_artifacts.gate_payload]
    evaluator_cfg = pilot_artifacts.evaluator_calls[0]
    assert evaluator_cfg["ground_truth_filename"] == str(pilot_artifacts.ground_truth_path)
    assert evaluator_cfg["prediction_filename"] == {"results": pilot_artifacts.predictions["results"]}


@pytest.mark.parametrize("artifact_name", ["result_detection.json", "evaluation_metrics.json"])
@pytest.mark.parametrize("bad_epoch", [None, 4, 6])
def test_pilot_artifact_validator_requires_epoch_five_evaluation_artifacts(
    pilot_artifacts, artifact_name, bad_epoch
):
    path = pilot_artifacts.work_dir / artifact_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluation_epoch"] = bad_epoch
    _write_json(path, payload)

    with pytest.raises(RuntimeError, match="evaluation_epoch"):
        validator.validate_pilot_artifacts(pilot_artifacts.run_dir)


def test_pilot_artifact_validator_rejects_arbitrary_checkpoint_bytes(pilot_artifacts):
    pilot_artifacts.checkpoint_path.write_bytes(b"this is not a torch checkpoint")

    with pytest.raises(RuntimeError, match="checkpoint"):
        validator.validate_pilot_artifacts(pilot_artifacts.run_dir)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda artifacts: artifacts.config_path.write_text("evaluation = dict()\n"), "config"),
        (lambda artifacts: artifacts.gate_path.write_text("{}\n"), "gate"),
        (
            lambda artifacts: artifacts.manifest.update(dataset_manifest_sha256="f" * 64),
            "dataset",
        ),
        (lambda artifacts: artifacts.manifest.update(commit="c" * 40), "commit"),
    ],
)
def test_pilot_artifact_validator_recomputes_all_manifest_bindings(
    pilot_artifacts, mutation, message
):
    mutation(pilot_artifacts)
    _write_json(pilot_artifacts.run_dir / "run_manifest.json", pilot_artifacts.manifest)

    with pytest.raises(RuntimeError, match=message):
        validator.validate_pilot_artifacts(pilot_artifacts.run_dir)


def test_pilot_artifact_validator_recomputes_dataset_manifest(pilot_artifacts, monkeypatch):
    monkeypatch.setattr(
        validator,
        "build_dataset_manifest",
        lambda cfg, ground_truth: ({"database": "changed"}, "f" * 64),
    )

    with pytest.raises(RuntimeError, match="dataset"):
        validator.validate_pilot_artifacts(pilot_artifacts.run_dir)


def test_pilot_artifact_validator_rejects_a_dirty_runtime_tree(pilot_artifacts, monkeypatch):
    def dirty_git_output(command, cwd=None, text=None):
        if command[-2:] == ["status", "--porcelain"]:
            return " M tools/train.py\n"
        if command[-1] == "HEAD":
            return "a" * 40 + "\n"
        if command[-1] == "HEAD^{tree}":
            return "b" * 40 + "\n"
        raise AssertionError(command)

    monkeypatch.setattr(validator.subprocess, "check_output", dirty_git_output)

    with pytest.raises(RuntimeError, match="not clean"):
        validator.validate_pilot_artifacts(pilot_artifacts.run_dir)


@pytest.mark.parametrize(
    "path_attribute,digest_key,mutate,expected_message",
    [
        (
            "contract_path",
            "contract_sha256",
            lambda payload: payload.update(contract_pass=False),
            "static contract",
        ),
        (
            "static_g0_path",
            "static_g0_sha256",
            lambda payload: payload.update(gate_pass=True),
            "static G0",
        ),
    ],
)
def test_pilot_artifact_validator_parses_bound_artifacts_after_hash_rebinding(
    pilot_artifacts, path_attribute, digest_key, mutate, expected_message
):
    artifact_path = getattr(pilot_artifacts, path_attribute)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(artifact_path, payload)
    rebound_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    pilot_artifacts.gate_payload[digest_key] = rebound_digest
    _write_json(pilot_artifacts.gate_path, pilot_artifacts.gate_payload)
    pilot_artifacts.manifest[digest_key] = rebound_digest
    pilot_artifacts.manifest["gate_sha256"] = hashlib.sha256(
        pilot_artifacts.gate_path.read_bytes()
    ).hexdigest()
    _write_json(pilot_artifacts.run_dir / "run_manifest.json", pilot_artifacts.manifest)

    with pytest.raises(RuntimeError, match=expected_message):
        validator.validate_pilot_artifacts(pilot_artifacts.run_dir)


@pytest.mark.parametrize("bad_epoch", [None, 4, 6])
def test_pilot_artifact_validator_rejects_wrong_checkpoint_epoch(pilot_artifacts, bad_epoch):
    pilot_artifacts.checkpoint["epoch"] = bad_epoch

    with pytest.raises(RuntimeError, match="epoch"):
        validator.validate_pilot_artifacts(pilot_artifacts.run_dir)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda checkpoint: checkpoint.pop("state_dict_ema"), "EMA"),
        (lambda checkpoint: checkpoint["optimizer"]["state"].update({1: {"step": 5}}), "optimizer"),
        (lambda checkpoint: checkpoint["scheduler"].update(last_epoch=5), "scheduler"),
    ],
)
def test_pilot_artifact_validator_requires_ema_and_consistent_optimizer_scheduler(
    pilot_artifacts, mutation, message
):
    mutation(pilot_artifacts.checkpoint)

    with pytest.raises(RuntimeError, match=message):
        validator.validate_pilot_artifacts(pilot_artifacts.run_dir)


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_state_dict",
        "empty_optimizer_state",
        "missing_optimizer_state",
        "missing_optimizer_param_groups",
        "empty_scheduler",
        "missing_scheduler",
        "nonfinite_state_tensor",
        "nonfinite_mapping_key",
        "nonfinite_optimizer_number",
    ],
)
def test_pilot_artifact_validator_rejects_incomplete_or_nonfinite_checkpoint(
    pilot_artifacts, mutation
):
    checkpoint = pilot_artifacts.checkpoint
    if mutation == "empty_state_dict":
        checkpoint["state_dict"] = {}
    elif mutation == "empty_optimizer_state":
        checkpoint["optimizer"]["state"] = {}
    elif mutation == "missing_optimizer_state":
        checkpoint["optimizer"].pop("state")
    elif mutation == "missing_optimizer_param_groups":
        checkpoint["optimizer"].pop("param_groups")
    elif mutation == "empty_scheduler":
        checkpoint["scheduler"] = {}
    elif mutation == "missing_scheduler":
        checkpoint.pop("scheduler")
    elif mutation == "nonfinite_state_tensor":
        checkpoint["state_dict"]["backbone.weight"] = _FakeTensor([float("nan")])
    elif mutation == "nonfinite_mapping_key":
        checkpoint["scheduler"][float("inf")] = "invalid"
    elif mutation == "nonfinite_optimizer_number":
        checkpoint["optimizer"]["param_groups"][0]["lr"] = float("inf")

    with pytest.raises(RuntimeError):
        validator.validate_pilot_artifacts(pilot_artifacts.run_dir)


def test_pilot_artifact_validator_rejects_tampered_metrics(pilot_artifacts):
    metrics_path = pilot_artifacts.work_dir / "evaluation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["mAP@0.7"] = 0.25
    _write_json(metrics_path, metrics)

    with pytest.raises(RuntimeError, match="mAP@0.7"):
        validator.validate_pilot_artifacts(pilot_artifacts.run_dir)


@pytest.mark.parametrize("mutation", ["nan_metric", "empty_prediction", "missing_checkpoint"])
def test_pilot_artifact_validator_fails_closed(pilot_artifacts, mutation):
    if mutation == "nan_metric":
        metrics_path = pilot_artifacts.work_dir / "evaluation_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["average_mAP"] = float("nan")
        _write_json(metrics_path, metrics)
    elif mutation == "empty_prediction":
        result_path = pilot_artifacts.work_dir / "result_detection.json"
        _write_json(result_path, {"evaluation_epoch": 5, "results": {"video": []}})
    elif mutation == "missing_checkpoint":
        pilot_artifacts.checkpoint_path.unlink()

    with pytest.raises((RuntimeError, ValueError)):
        validator.validate_pilot_artifacts(pilot_artifacts.run_dir)


class _AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _load_test_engine_with_stubs(monkeypatch, evaluator_metrics):
    fake_dist = types.ModuleType("torch.distributed")

    def all_gather_object(output, value):
        output[0] = value

    fake_dist.all_gather_object = all_gather_object
    fake_torch = types.ModuleType("torch")
    fake_torch.distributed = fake_dist
    fake_torch.float16 = "float16"
    fake_torch.no_grad = nullcontext
    fake_torch.cuda = types.SimpleNamespace(
        amp=types.SimpleNamespace(autocast=lambda **_kwargs: nullcontext())
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, disable=False: iterable
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)

    fake_utils = types.ModuleType("opentad.utils")
    fake_utils.create_folder = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(sys.modules, "opentad.utils", fake_utils)

    fake_post = types.ModuleType("opentad.models.utils.post_processing")
    fake_post.build_classifier = lambda cfg: cfg
    fake_post.batched_nms = lambda segments, scores, labels, **_kwargs: (segments, scores, labels)
    monkeypatch.setitem(sys.modules, "opentad.models.utils.post_processing", fake_post)

    evaluator_calls = []

    class _Evaluator:
        def evaluate(self):
            return dict(evaluator_metrics)

        def logging(self, _logger):
            return None

    fake_evaluations = types.ModuleType("opentad.evaluations")

    def build_evaluator(cfg):
        evaluator_calls.append(cfg)
        return _Evaluator()

    fake_evaluations.build_evaluator = build_evaluator
    monkeypatch.setitem(sys.modules, "opentad.evaluations", fake_evaluations)

    fake_dataset = types.ModuleType("opentad.datasets.base")
    fake_dataset.SlidingWindowDataset = type("SlidingWindowDataset", (), {})
    monkeypatch.setitem(sys.modules, "opentad.datasets.base", fake_dataset)

    module_name = "_phystime_test_engine_under_test"
    module_path = ROOT / "opentad" / "cores" / "test_engine.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evaluator_calls


@pytest.mark.parametrize("evaluation_epoch", [5, None])
def test_eval_one_epoch_writes_evaluation_epoch_to_both_artifacts(
    tmp_path, monkeypatch, evaluation_epoch
):
    module, evaluator_calls = _load_test_engine_with_stubs(monkeypatch, REQUIRED_METRICS)

    class _Dataset:
        class_map = ["A"]

    class _Loader(list):
        dataset = _Dataset()

    class _Model:
        def eval(self):
            return None

        def __call__(self, **_kwargs):
            return {
                "video_test_0000001": [
                    {"segment": [1.0, 2.0], "label": "A", "score": 0.5}
                ]
            }

    cfg = types.SimpleNamespace(
        work_dir=str(tmp_path),
        inference=_AttrDict(save_raw_prediction=False),
        post_processing=_AttrDict(save_dict=True, nms=None),
        evaluation=_AttrDict(),
    )
    logger = types.SimpleNamespace(info=lambda _message: None)

    module.eval_one_epoch(
        _Loader([{}]),
        _Model(),
        cfg,
        logger,
        rank=0,
        world_size=1,
        evaluation_epoch=evaluation_epoch,
    )

    result_payload = json.loads((tmp_path / "result_detection.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((tmp_path / "evaluation_metrics.json").read_text(encoding="utf-8"))
    assert result_payload["evaluation_epoch"] == evaluation_epoch
    assert metrics_payload["evaluation_epoch"] == evaluation_epoch
    assert metrics_payload["average_mAP"] == pytest.approx(1.0)
    assert evaluator_calls[0]["prediction_filename"] == result_payload


def test_training_passes_current_epoch_to_evaluation():
    tree = ast.parse((ROOT / "tools" / "train.py").read_text(encoding="utf-8"))
    eval_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval_one_epoch"
    ]
    assert len(eval_calls) == 1
    epoch_keywords = [keyword.value for keyword in eval_calls[0].keywords if keyword.arg == "evaluation_epoch"]
    assert len(epoch_keywords) == 1
    assert isinstance(epoch_keywords[0], ast.Name)
    assert epoch_keywords[0].id == "epoch"

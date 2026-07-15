import importlib
import hashlib
import inspect

import pytest
import torch

from opentad.models.chronotransport.controls import motion_topk_actions
from opentad.models.chronotransport.protocol import canonical_sha256


def test_production_profile_entrypoints_have_no_backend_injection_surface():
    from tools.bata.chronotransport_r2_profile_factory import (
        build_registered_profile_session,
    )
    from tools.bata.profile_chronotransport_r2_full_stack import profile_request

    assert tuple(inspect.signature(profile_request).parameters) == (
        "payload",
        "repository_root",
        "registration_commit",
        "registration_relpath",
    )
    assert tuple(inspect.signature(build_registered_profile_session).parameters) == (
        "registration",
    )


def test_all_media_are_preverified_once_and_lookup_never_rehashes(tmp_path, monkeypatch):
    from tools.bata import chronotransport_r2_opentad_profile_backend as backend_module

    windows = []
    for index in range(200):
        path = tmp_path / f"video-{index:03d}.mp4"
        payload = f"registered-media-{index}".encode()
        path.write_bytes(payload)
        windows.append(
            {
                "window_id": f"window-{index:03d}",
                "media_path": path.name,
                "media_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    registration = {"data": {"root_path": str(tmp_path)}}
    original = backend_module._file_sha256
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(backend_module, "_file_sha256", counted)
    verified = backend_module.preverify_registered_media(registration, windows)
    assert len(verified) == 200
    assert len(calls) == 200

    backend = backend_module.OpenTADRegisteredProfileBackend.__new__(
        backend_module.OpenTADRegisteredProfileBackend
    )
    backend.registration = registration
    backend._verified_media = verified
    for window in reversed(windows):
        assert backend._verify_media(window) == verified[window["window_id"]]
    assert len(calls) == 200


def test_repo_owned_opentad_profile_backend_module_is_present_and_fixed():
    try:
        module = importlib.import_module(
            "tools.bata.chronotransport_r2_opentad_profile_backend"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"fixed OpenTAD profile backend is missing: {exc}")
    backend = getattr(module, "OpenTADRegisteredProfileBackend", None)
    assert backend is not None
    assert backend.__module__ == (
        "tools.bata.chronotransport_r2_opentad_profile_backend"
    )


def test_audited_dense_checkpoint_load_is_strict_about_unexpected_keys(tmp_path):
    from tools.bata.chronotransport_r2_opentad_profile_backend import (
        audit_load_dense_checkpoint,
    )

    model = torch.nn.Linear(2, 2)
    checkpoint = tmp_path / "dense.pth"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    report = audit_load_dense_checkpoint(
        model,
        checkpoint,
        use_ema=False,
        allow_chronotransport_missing=False,
    )
    assert report["status"] == "PASS"
    assert report["state_key"] == "state_dict"

    stable_bytes = checkpoint.read_bytes()
    stable_sha256 = hashlib.sha256(stable_bytes).hexdigest()
    checkpoint.write_bytes(b"mutated-after-stable-read")
    stable_report = audit_load_dense_checkpoint(
        model,
        stable_bytes,
        use_ema=False,
        allow_chronotransport_missing=False,
        expected_sha256=stable_sha256,
        expected_bytes=len(stable_bytes),
    )
    assert stable_report["checkpoint_sha256"] == stable_sha256

    torch.save({"state_dict": model.state_dict()}, checkpoint)

    bad_state = dict(model.state_dict())
    bad_state["unregistered.weight"] = torch.ones(1)
    torch.save({"state_dict": bad_state}, checkpoint)
    with pytest.raises(RuntimeError, match="unexpected|incompatible"):
        audit_load_dense_checkpoint(
            model,
            checkpoint,
            use_ema=False,
            allow_chronotransport_missing=False,
        )


def test_backend_uses_official_forward_test_and_postprocess_with_exact_objects():
    from tools.bata.chronotransport_r2_opentad_profile_backend import (
        OpenTADRegisteredProfileBackend,
    )

    calls = []

    class Model:
        def forward_test(self, inputs, masks, metas=None, infer_cfg=None):
            calls.append(("forward_test", inputs, masks, metas, infer_cfg))
            return "official-predictions"

        def post_processing(self, predictions, metas, post_cfg, external_classes):
            calls.append(
                (
                    "post_processing",
                    predictions,
                    metas,
                    post_cfg,
                    external_classes,
                )
            )
            return "official-nms-output"

    backend = OpenTADRegisteredProfileBackend.__new__(OpenTADRegisteredProfileBackend)
    backend.model = Model()
    backend.post_cfg = object()
    backend.external_classes = ["class-0"]
    backend._timed = lambda callback, sync_cuda: (callback(), 1.25)
    inputs = object()
    masks = object()
    metas = [{"video_name": "registered", "valid_mask": [True, False]}]
    predictions, forward_ms = backend._model_forward(
        {"inputs": inputs, "masks": masks, "metas": metas}
    )
    nms_output, post_ms = backend._official_postprocess(predictions, metas)
    assert predictions == "official-predictions" and forward_ms == 1.25
    assert nms_output == "official-nms-output" and post_ms == 1.25
    assert calls == [
        ("forward_test", inputs, masks, metas, None),
        (
            "post_processing",
            predictions,
            metas,
            backend.post_cfg,
            backend.external_classes,
        ),
    ]


def _control_registration(candidate_name, action_payload):
    action_hash = canonical_sha256(action_payload)
    return {
        "candidate_library": {"candidates": []},
        "controls": {
            "motion_topk": {"sha256": hashlib.sha256(b"motion").hexdigest()},
            "random": {"sha256": hashlib.sha256(b"random").hexdigest()},
        },
        "profiler": {
            "invocation_ids": ["window-0"],
            "candidate_plan": [
                {
                    "candidate_name": candidate_name,
                    "factory_config": {
                        "candidate_name": candidate_name,
                        "mode": "registered_full_stack",
                    },
                    "requested_action_sha256_by_invocation": [action_hash],
                }
            ],
        },
    }


def test_motion_control_uses_deploy_visible_signal_and_rejects_fake_signal():
    from tools.bata.chronotransport_r2_opentad_profile_backend import (
        resolve_registered_action_payload,
    )

    signal = torch.arange(48, dtype=torch.float32).view(48, 1).expand(48, 3)
    expected = motion_topk_actions(signal.unsqueeze(0), period=4)[0].tolist()
    registration = _control_registration("motion_topk_p4", expected)
    actual = resolve_registered_action_payload(
        registration,
        window_id="window-0",
        candidate_name="motion_topk_p4",
        deploy_visible_motion=signal,
    )
    assert actual == expected

    fake_signal = torch.flip(signal, dims=(0,))
    with pytest.raises(RuntimeError, match="registered action hash"):
        resolve_registered_action_payload(
            registration,
            window_id="window-0",
            candidate_name="motion_topk_p4",
            deploy_visible_motion=fake_signal,
        )


def test_random_control_fails_closed_until_seed_is_frozen_in_factory_config():
    from tools.bata.chronotransport_r2_opentad_profile_backend import (
        resolve_registered_action_payload,
    )

    registration = _control_registration(
        "random_p4", [[0, 0, 0]] + [[2, 2, 2] for _ in range(47)]
    )
    with pytest.raises(RuntimeError, match="control_seed.*frozen"):
        resolve_registered_action_payload(
            registration,
            window_id="window-0",
            candidate_name="random_p4",
        )

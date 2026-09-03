from __future__ import annotations

from collections.abc import Mapping


def _move_tensors(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    return value


def prepare_zoomtoken_batch(batch, device, *, source_key: str = "source"):
    """Move model inputs to one rank while retaining source-native video on CPU."""
    import torch

    if not isinstance(batch, Mapping):
        raise TypeError("ZoomToken batches must be mappings")
    prepared = dict(batch)
    inputs = prepared.get("inputs")
    if isinstance(inputs, Mapping):
        moved_inputs = {}
        for key, value in inputs.items():
            if key == source_key:
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                    raise ValueError("source-native uint8 video must remain a CPU tensor")
                moved_inputs[key] = value
            else:
                moved_inputs[key] = _move_tensors(value, device)
        prepared["inputs"] = moved_inputs
    elif inputs is not None:
        prepared["inputs"] = _move_tensors(inputs, device)

    for key in ("masks", "gt_segments", "gt_labels"):
        if key in prepared:
            prepared[key] = _move_tensors(prepared[key], device)
    return prepared


class ZoomTokenDeviceLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield prepare_zoomtoken_batch(batch, self.device)

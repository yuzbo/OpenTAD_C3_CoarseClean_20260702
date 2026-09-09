"""Separate route/loss changes from ordinary backward repeatability.

Source and strict modes must run in separate processes. This diagnostic does
not decompose losses, update parameters, or certify the old attribution tool.
"""
import argparse
from contextlib import contextmanager
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import sys
import traceback


def backend_settings():
    import torch
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_enabled=torch.backends.cudnn.enabled,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


@contextmanager
def algorithm_mode(mode):
    import torch
    if mode not in {"source", "strict"}:
        raise ValueError("mode must be source or strict")
    original = backend_settings()
    try:
        if mode == "strict":
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        yield backend_settings()
    finally:
        torch.use_deterministic_algorithms(original["deterministic_algorithms"],
                                          warn_only=original["deterministic_warn_only"])
        torch.backends.cudnn.deterministic = original["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = original["cudnn_benchmark"]


def plan_tensors(plan):
    """Store actual values, including PL order; no digest or token-count proxy."""
    result = {}
    for field in fields(plan):
        value = getattr(plan, field.name)
        if isinstance(value, list):
            result.update({f"{field.name}.{i}": item.detach().cpu().clone()
                           for i, item in enumerate(value)})
        else:
            result[field.name] = value.detach().cpu().clone()
    return result


def ordinary_trial(model, batch, output, *, microbatch, amp, loss_scale):
    import torch
    from geosparse_ext.runtime import subset_batch
    from geosparse_research.training_gradients import _training_measurement_state

    size = len(batch["inputs"])
    if size == 0 or microbatch < 1 or size % microbatch:
        raise ValueError("microbatch must divide the effective batch")
    if amp and not next(model.parameters()).is_cuda:
        raise ValueError("AMP requires CUDA")
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("loss scale must be finite and positive")
    output.mkdir(parents=True, exist_ok=False)
    with _training_measurement_state(model):
        for parameter in model.parameters():
            parameter.grad = None
        for begin in range(0, size, microbatch):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                losses = model.forward_train(**subset_batch(batch, begin, begin + microbatch))
            # Persist before backward, including when strict mode rejects an op.
            # This CPU observation/synchronization is shared by every repeat.
            forward = dict(losses={key: value.detach().cpu().clone() for key, value in losses.items()},
                           plan=plan_tensors(model.latest_route_plan),
                           acquisition=model.latest_acquisition)
            torch.save(forward, output / f"forward-{begin // microbatch:03d}.pt")
            (losses["cost"] * (microbatch / size) * loss_scale).backward()
        vectors = {name: parameter.grad.detach().float().cpu().clone() / loss_scale
                   for name, parameter in model.named_parameters() if parameter.grad is not None}
    torch.save(vectors, output / "gradients.pt")
    return vectors


def compare_forwards(left, right):
    import torch
    plan_keys = set(left["plan"]) | set(right["plan"])
    different = [key for key in sorted(plan_keys)
                 if key not in left["plan"] or key not in right["plan"]
                 or not torch.equal(left["plan"][key], right["plan"][key])]
    # Gain/log-prob are floating router outputs, distinct from executed choices.
    choices = [key for key in different if key not in {"predicted_gain", "log_prob"}]
    losses = {}
    for key in sorted(set(left["losses"]) | set(right["losses"])):
        a, b = left["losses"].get(key), right["losses"].get(key)
        losses[key] = dict(present_both=a is not None and b is not None,
                          exact=a is not None and b is not None and torch.equal(a, b),
                          maximum_absolute=None if a is None or b is None else float((a.double() - b.double()).abs().max()))
    return dict(executed_choices_exact=not choices, differing_plan_fields=different,
                losses=losses, all_losses_exact=all(row["exact"] for row in losses.values()),
                acquisition_exact=left["acquisition"] == right["acquisition"])


def measure_repeats(model, batch, output, *, microbatch, amp, loss_scale, mode):
    import torch
    from geosparse_research.training_gradient_parity import compare_vectors

    output.mkdir(parents=True, exist_ok=False)
    vectors, comparisons = {}, {}
    with algorithm_mode(mode) as settings:
        for index in range(3):
            folder = output / f"repeat-{index}"
            vectors[index] = ordinary_trial(model, batch, folder, microbatch=microbatch,
                                            amp=amp, loss_scale=loss_scale)
            if index:
                forward = [compare_forwards(torch.load(path, map_location="cpu"),
                           torch.load(output / "repeat-0" / path.name, map_location="cpu"))
                           for path in sorted(folder.glob("forward-*.pt"))]
                comparisons[str(index)] = dict(forwards=forward,
                    gradients=compare_vectors(vectors[index], vectors[0]))
    result = dict(status="completed_ordinary_repeat_measurement", mode=mode, amp=amp,
                  loss_scale=loss_scale, backend=settings, comparisons=comparisons,
                  state_restored=True, optimizer_step_performed=False,
                  loss_component_attribution_result=False)
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(args):
    # Set before prepare_probe imports Torch or creates any CUDA handle.
    if args.mode == "strict":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from geosparse_research.training_gradient_diagnostic import prepare_probe
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(measurement="ordinary_repeat_routes_losses_gradients", mode=args.mode)
    try:
        model, batch, loaded, _ = prepare_probe(args)
        identity.update(loaded)
        identity["measurement"] = "ordinary_repeat_routes_losses_gradients"
        identity["original_backend"] = backend_settings()
        (args.output / "measurement.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
        modes = [("source_amp", True, identity["loss_scale"])] if identity["amp"] else []
        modes.append(("fp32", False, 1.))
        results = {}
        for precision, amp, scale in modes:
            results[precision] = measure_repeats(model, batch, args.output / precision,
                microbatch=identity["microbatch"], amp=amp, loss_scale=scale, mode=args.mode)
        (args.output / "result.json").write_text(json.dumps(dict(identity, results=results,
            status="completed_ordinary_repeat_measurement", loss_component_attribution_result=False,
            cross_mode_plan_equivalence_claimed=False), indent=2), encoding="utf-8")
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps(dict(identity, status="failed_measurement",
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()), indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--completed-epochs", type=int, required=True, choices=range(1, 60))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["source", "strict"], required=True)
    run(parser.parse_args())

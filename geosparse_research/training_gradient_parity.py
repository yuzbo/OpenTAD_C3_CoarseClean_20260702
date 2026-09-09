"""Persist ordinary/decomposed gradient vectors, including failed equivalence.

This is an isolated measurement of the existing trainer, not a new training
algorithm or a relaxed admission gate for loss-component attribution.
"""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys


def compare_vectors(measured, ordinary, *, rtol=1e-4, atol=1e-5):
    import torch
    from geosparse_research.training_gradients import _norm

    rows, difference = [], {}
    all_close = set(measured) == set(ordinary)
    for name in sorted(measured.keys() | ordinary.keys()):
        left, right = measured.get(name), ordinary.get(name)
        connected_both = left is not None and right is not None
        if left is None:
            left = torch.zeros_like(right)
        if right is None:
            right = torch.zeros_like(left)
        delta = left.double() - right.double()
        difference[name] = delta
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        close = connected_both and finite and torch.allclose(left, right, rtol=rtol, atol=atol)
        all_close = all_close and close
        rows.append(dict(name=name, connected_both=connected_both, finite=finite,
                         measured_norm=_norm({name: left}), ordinary_norm=_norm({name: right}),
                         difference_l2=_norm({name: delta}),
                         maximum_absolute=float(delta.abs().max()) if finite else None,
                         within_tolerance=bool(close)))
    norm_left, norm_right, error = _norm(measured), _norm(ordinary), _norm(difference)
    scalar_close = (norm_left is not None and norm_right is not None
                    and abs(norm_left - norm_right) <= atol + rtol * abs(norm_right))
    return dict(rtol=rtol, atol=atol, vector_close=bool(all_close), scalar_norm_close=bool(scalar_close),
                measured_norm=norm_left, ordinary_norm=norm_right, difference_l2=error,
                relative_l2=error / max(norm_right, 1e-12) if error is not None and norm_right is not None else None,
                parameters=rows)


def ordinary_vectors(model, batch, microbatch, amp, loss_scale):
    import torch
    from geosparse_ext.runtime import subset_batch
    from geosparse_research.training_gradients import _training_measurement_state

    size = len(batch["inputs"])
    if size == 0 or microbatch < 1 or size % microbatch:
        raise ValueError("microbatch must divide the effective batch")
    if amp and not next(model.parameters()).is_cuda:
        raise ValueError("AMP parity requires CUDA")
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("loss scale must be finite and positive")
    with _training_measurement_state(model):
        for parameter in model.parameters():
            parameter.grad = None
        for begin in range(0, size, microbatch):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                losses = model.forward_train(**subset_batch(batch, begin, begin + microbatch))
            (losses["cost"] * (microbatch / size) * loss_scale).backward()
        vectors = {name: parameter.grad.detach().float().cpu().clone() / loss_scale
                   for name, parameter in model.named_parameters() if parameter.grad is not None}
    return vectors


def measure_parity(model, batch, output, *, microbatch, amp, loss_scale, clip_norm):
    import torch
    from geosparse_ext.records import save_json
    from geosparse_research.training_gradients import measure_training_gradients

    output.mkdir(parents=True, exist_ok=False)
    vectors = {}
    for label in ("ordinary_before", "ordinary_repeat"):
        vectors[label] = ordinary_vectors(model, batch, microbatch, amp, loss_scale)
        torch.save(vectors[label], output / (label + ".pt"))
    result = measure_training_gradients(model, batch, microbatch_size=microbatch, amp=amp,
        loss_scale=loss_scale, clip_norm=clip_norm,
        total_gradient_sink=lambda total: vectors.update(decomposed_total=total))
    # Persist before comparing: a parity failure must not discard its evidence.
    torch.save(vectors["decomposed_total"], output / "decomposed_total.pt")
    save_json(output / "decomposition.unverified.json", result)
    vectors["ordinary_after"] = ordinary_vectors(model, batch, microbatch, amp, loss_scale)
    torch.save(vectors["ordinary_after"], output / "ordinary_after.pt")
    comparisons = {name: compare_vectors(vectors[name], vectors["ordinary_before"])
                   for name in ("ordinary_repeat", "decomposed_total", "ordinary_after")}
    receipt = dict(status="completed_gradient_parity_measurement", amp=amp, loss_scale=loss_scale,
        comparisons=comparisons, optimizer_step_performed=False, state_restored=True,
        attribution_equivalence_passed=all(row["vector_close"] and row["scalar_norm_close"]
                                          for row in comparisons.values()),
        interpretation="One next-epoch batch; parity evidence, not a population-wide loss attribution.")
    save_json(output / "parity.json", receipt)
    return receipt


def run(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from geosparse_research.training_gradient_diagnostic import prepare_probe
    model, batch, identity, cfg = prepare_probe(args)
    from geosparse_ext.records import save_json

    args.output.mkdir(parents=True, exist_ok=False)
    identity["measurement"] = "ordinary_checkpoint_next_training_batch_gradient_parity"
    save_json(args.output / "measurement.json", identity)
    try:
        modes = [("source_amp", True, identity["loss_scale"])] if identity["amp"] else []
        modes.append(("fp32", False, 1.))
        results = {}
        for label, amp, scale in modes:
            results[label] = measure_parity(model, batch, args.output / label,
                microbatch=identity["microbatch"], amp=amp, loss_scale=scale,
                clip_norm=float(cfg.solver.clip_grad_norm))
        save_json(args.output / "result.json", dict(identity,
            status="completed_gradient_parity_measurement", modes=results,
            loss_component_attribution_result=False,
            cross_precision="Each mode restarts the saved model/RNG. Precision may change plans; no AMP/FP32 same-plan claim.",
            completed_at_utc=datetime.now(timezone.utc).isoformat()))
    except Exception as error:
        save_json(args.output / "failure.json", dict(identity, status="failed",
            error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--completed-epochs", type=int, required=True, choices=range(1, 60))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())

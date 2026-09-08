"""Isolate observer-induced differences; never produces detection performance."""
from contextlib import ExitStack
import copy


def tensor_difference(reference, measured):
    import torch
    result = dict(reference_shape=list(reference.shape), measured_shape=list(measured.shape),
                  reference_dtype=str(reference.dtype), measured_dtype=str(measured.dtype))
    if reference.shape != measured.shape:
        return dict(result, exact=False, reason="shape_changed")
    left, right = reference.double(), measured.double()
    finite = torch.isfinite(left) & torch.isfinite(right)
    differences = (left - right).abs()
    nonzero = finite & (left != 0)
    return dict(result, exact=reference.dtype == measured.dtype and torch.equal(reference, measured),
                elements=reference.numel(), different_elements=int((reference != measured).sum()),
                nonfinite_elements=int((~finite).sum()),
                max_abs=float(differences[finite].max()) if finite.any() else None,
                max_rel_nonzero_reference=float((differences[nonzero] / left[nonzero].abs()).max())
                if nonzero.any() else None)


def compare_observers(model, batch, *, amp):
    """Nine same-state forwards: repeated controls plus separated instrumentation.

    Raw plans are compared before interpreting proposal/score differences. No
    tolerance is relaxed, no prediction is postprocessed, and no training occurs.
    The caller must supply the original loaded model and its actual test batch.
    """
    import torch
    from geosparse_ext.analysis_capture import OperationMacCounter, SelectionObserver
    from geosparse_research.interventions import _measurement_state

    if model.training or model.geosparse.config["route"] != "A":
        raise ValueError("requires an eval-mode Route A detector")
    if amp and not next(model.parameters()).is_cuda:
        raise ValueError("the production AMP comparison requires CUDA")
    original_plan = model.latest_route_plan
    cases = [(name, repeat) for name in ("plain", "selection", "counter", "both")
             for repeat in (0, 1)] + [("plain_after", 0)]
    outputs, rows = {}, []
    try:
        with _measurement_state(model) as reset:
            initial_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
            initial_rng = getattr(model.geosparse, "execution_rng", None)
            initial_trace = model.geosparse.encoder.trace
            for name, repeat in cases:
                reset()
                model.latest_route_plan = original_plan
                model.geosparse.execution_rng = initial_rng
                model.geosparse.encoder.trace = initial_trace
                key = f"{name}_{repeat}"
                with ExitStack() as contexts:
                    if name in {"selection", "both"}:
                        observer = SelectionObserver(model)
                        contexts.callback(observer.close)
                    counter = contexts.enter_context(OperationMacCounter(model)) if name in {"counter", "both"} else None
                    with torch.cuda.amp.autocast(enabled=amp):
                        proposals, scores = model.forward_test(**batch)
                plan = model.latest_route_plan
                outputs[key] = dict(proposals=[x.detach().cpu().clone() for x in proposals],
                                    scores=[x.detach().cpu().clone() for x in scores],
                                    selected_native=plan.selected_native.detach().cpu().clone(),
                                    selected_atoms=plan.selected_atoms.detach().cpu().clone(),
                                    requested_budget=plan.requested_budget.detach().cpu().clone(),
                                    trace=copy.deepcopy(model.geosparse.encoder.trace))
                changed = [buffer_name for buffer_name, value in model.named_buffers()
                           if buffer_name not in initial_buffers or
                           not torch.equal(initial_buffers[buffer_name], value.detach().cpu()) or
                           initial_buffers[buffer_name].dtype != value.dtype]
                baseline, current = outputs["plain_0"], outputs[key]
                comparisons = {field: tensor_difference(baseline[field], current[field])
                               for field in ("selected_native", "selected_atoms", "requested_budget")}
                for field in ("proposals", "scores"):
                    if len(baseline[field]) != len(current[field]):
                        raise ValueError("prediction batch length changed between observations")
                    comparisons[field] = [tensor_difference(a, b) for a, b in zip(baseline[field], current[field])]
                rows.append(dict(case=key, comparison_to="plain_0", comparisons=comparisons,
                                 buffers_changed=changed,
                                 trace_equal=current["trace"] == baseline["trace"],
                                 operation_counter=counter.report() if counter else None))
    finally:
        model.latest_route_plan = original_plan
    exact = all(all(row["comparisons"][key]["exact"] for key in ("selected_native", "selected_atoms", "requested_budget"))
                and all(item["exact"] for key in ("proposals", "scores") for item in row["comparisons"][key])
                and not row["buffers_changed"] and row["trace_equal"] for row in rows)
    return dict(status="EXACT_IN_THIS_PROBE" if exact else "OBSERVED_DIFFERENCES", amp=amp,
                scientific_result=False, optimizer_updates=0, forward_calls=len(rows),
                state_reset_before_each_forward=True, comparisons=rows), outputs

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tools.bata.continuous_roi_s2_v3_full200_compute import (
    atomic_publish_json,
    canonical_sha256,
    require_clean_commit,
    sha256_file,
)
from tools.bata.continuous_roi_s2_v3_full200_compute_profile import (
    FullOperatorLedger,
    compare_c_exec_receipts,
    validate_c_exec_receipt,
)
from tools.bata.continuous_roi_s2_v3_full200_compute_train import (
    validate_full_data_manifest,
)
from tools.bata.zoomtoken_full200_matrix_spec import (
    get_matrix_spec,
    validate_matrix_cell,
)


BOUNDARY_TRACE = {
    "start": "first_arm_dependent_decoded_rgb_transform",
    "end": "pre_nms_raw_detections",
    "nms_called": False,
    "evaluator_called": False,
}

AUTOMATIC_EVENT_NAMES = {
    "aten::add": "aten.add",
    "aten::addmm": "aten.addmm",
    "aten::bmm": "aten.bmm",
    "aten::matmul": "aten.matmul",
    "aten::mm": "aten.mm",
    "aten::mul": "aten.mul",
}

# These dispatcher entries either wrap a separately counted kernel/module or only
# reshape, allocate, copy, index, mask, or expose scalar metadata. They perform no
# floating-point arithmetic that is missing from the ledger.
ZERO_ARITHMETIC_OR_WRAPPER_EVENTS = {
    "aten::__and__",
    "aten::__or__",
    "aten::_conv_depthwise2d",
    "aten::_convolution",
    "aten::_flash_attention_forward",
    "aten::_index_put_impl_",
    "aten::_local_scalar_dense",
    "aten::_reshape_alias",
    "aten::_to_copy",
    "aten::_unsafe_view",
    "aten::alias",
    "aten::arange",
    "aten::argsort",
    "aten::as_strided",
    "aten::as_strided_",
    "aten::cat",
    "aten::clone",
    "aten::contiguous",
    "aten::conv1d",
    "aten::conv3d",
    "aten::convolution",
    "aten::copy_",
    "aten::cudnn_convolution",
    "aten::detach",
    "aten::dropout",
    "aten::dropout_",
    "aten::empty",
    "aten::empty_like",
    "aten::empty_strided",
    "aten::expand",
    "aten::expand_as",
    "aten::fill_",
    "aten::flatten",
    "aten::full",
    "aten::gather",
    "aten::index",
    "aten::index_put_",
    "aten::index_select",
    "aten::is_nonzero",
    "aten::item",
    "aten::layer_norm",
    "aten::lift_fresh",
    "aten::linear",
    "aten::linspace",
    "aten::masked_fill",
    "aten::matmul",
    "aten::max_pool1d",
    "aten::max_pool1d_with_indices",
    "aten::max_pool2d_with_indices",
    "aten::narrow",
    "aten::ones_like",
    "aten::permute",
    "aten::repeat",
    "aten::reshape",
    "aten::reshape_as",
    "aten::resize_",
    "aten::resolve_conj",
    "aten::resolve_neg",
    "aten::result_type",
    "aten::rsub",
    "aten::scaled_dot_product_attention",
    "aten::scatter",
    "aten::scatter_",
    "aten::select",
    "aten::set_",
    "aten::slice",
    "aten::softmax",
    "aten::square",
    "aten::squeeze",
    "aten::squeeze_",
    "aten::stack",
    "aten::t",
    "aten::to",
    "aten::transpose",
    "aten::unbind",
    "aten::unflatten",
    "aten::unfold",
    "aten::unsqueeze",
    "aten::unsqueeze_",
    "aten::upsample_bicubic2d",
    "aten::upsample_linear1d",
    "aten::upsample_nearest1d",
    "aten::view",
    "aten::view_as",
    "aten::zero_",
    "aten::zeros",
    "aten::zeros_like",
}


def _shape_numel(shape: Sequence[int]) -> int:
    if not shape:
        return 0
    return math.prod(int(value) for value in shape)


def _event_tensor_numel(event: Any) -> int:
    return max(
        (_shape_numel(shape) for shape in event.input_shapes if shape),
        default=0,
    )


def _sort_comparison_upper_bound(shape: Sequence[int], count: int) -> int:
    if not shape:
        return 0
    width = int(shape[-1])
    rows = _shape_numel(shape[:-1]) if len(shape) > 1 else 1
    return int(count) * rows * width * max(1, math.ceil(math.log2(max(2, width))))


def _manual_profiler_rule(event: Any) -> tuple[str, int, str, bool] | None:
    key = str(event.key)
    count = int(event.count)
    numel = _event_tensor_numel(event) * count
    if key == "aten::_scaled_dot_product_flash_attention":
        q, k, value = event.input_shapes[:3]
        if len(q) != 4 or len(k) != 4 or len(value) != 4:
            raise ValueError("flash-attention profiler shape changed")
        batch, heads, queries, q_dim = map(int, q)
        keys = int(k[-2])
        value_dim = int(value[-1])
        matmuls = 2 * batch * heads * queries * keys * (q_dim + value_dim)
        softmax = 5 * batch * heads * queries * keys
        operations = count * (matmuls + softmax)
        return (
            "flash_attention_runtime_shape_upper_bound",
            operations,
            "count*(2*B*H*Q*K*(Dq+Dv)+5*B*H*Q*K)",
            True,
        )
    if key == "aten::native_layer_norm":
        return "layer_norm_upper_bound", 7 * numel, "7*input_elements", True
    if key == "aten::_softmax":
        return "softmax_upper_bound", 5 * numel, "5*input_elements", True
    if key == "aten::gelu":
        return "gelu_upper_bound", 8 * numel, "8*input_elements", True
    if key in {"aten::mean", "aten::sum"}:
        return (
            "mean_reduction_upper_bound",
            2 * numel,
            "2*input_elements",
            True,
        )
    if key == "aten::linalg_vector_norm":
        return (
            "normalization_upper_bound",
            5 * numel,
            "5*input_elements",
            True,
        )
    if key in {"aten::sort", "aten::_unique2"}:
        shape = next((shape for shape in event.input_shapes if shape), ())
        return (
            "sort_comparison_upper_bound",
            _sort_comparison_upper_bound(shape, count),
            "rows*N*ceil(log2(N))",
            True,
        )
    if key in {
        "aten::add_",
        "aten::all",
        "aten::any",
        "aten::bitwise_and",
        "aten::bitwise_or",
        "aten::ceil",
        "aten::clamp",
        "aten::clamp_min",
        "aten::clamp_min_",
        "aten::cumsum",
        "aten::div",
        "aten::eq",
        "aten::ge",
        "aten::gt",
        "aten::logical_not",
        "aten::lt",
        "aten::masked_fill_",
        "aten::max",
        "aten::nonzero",
        "aten::pow",
        "aten::relu",
        "aten::relu_",
        "aten::remainder",
        "aten::scatter_add_",
        "aten::sigmoid",
        "aten::sub",
    }:
        multiplier = 4 if key in {"aten::pow", "aten::sigmoid"} else 1
        return (
            "elementwise_upper_bound",
            multiplier * numel,
            f"{multiplier}*broadcast_output_elements_upper_bound",
            True,
        )
    return None


@contextmanager
def _capture_interpolation(torch_module: Any):
    functional = torch_module.nn.functional
    original = functional.interpolate
    records: list[tuple[str, int, int]] = []

    def traced(input_tensor: Any, *args: Any, **kwargs: Any) -> Any:
        output = original(input_tensor, *args, **kwargs)
        mode = str(kwargs.get("mode", "nearest"))
        records.append((mode, int(input_tensor.numel()), int(output.numel())))
        return output

    functional.interpolate = traced
    try:
        yield records
    finally:
        functional.interpolate = original


def _register_module_shape_hooks(model: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    import torch

    records: list[dict[str, Any]] = []
    handles = []
    invocations: dict[str, int] = {}
    convolution_types = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    pooling_types = (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)

    for name, module in model.named_modules():
        if not isinstance(module, convolution_types + pooling_types):
            continue

        def hook(current: Any, inputs: tuple[Any, ...], output: Any, *, path: str = name) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                raise TypeError(f"shape hook for {path} did not receive a tensor")
            invocation = invocations.get(path, 0)
            invocations[path] = invocation + 1
            if isinstance(current, convolution_types):
                kernel_elements = math.prod(int(value) for value in current.kernel_size)
                macs = (
                    int(tensor.numel())
                    * (int(current.in_channels) // int(current.groups))
                    * kernel_elements
                )
                bias = int(tensor.numel()) if current.bias is not None else 0
                records.append(
                    {
                        "event_id": f"module/{path}/conv/{invocation}",
                        "operator": "convolution_runtime_shape_fma2",
                        "integer_operations": 2 * macs + bias,
                        "formula": "2*output_elements*(in_channels/groups)*kernel_elements+bias",
                        "conservative_upper_bound": False,
                    }
                )
            else:
                kernel = current.kernel_size
                kernel_elements = (
                    math.prod(int(value) for value in kernel)
                    if isinstance(kernel, tuple)
                    else int(kernel)
                )
                records.append(
                    {
                        "event_id": f"module/{path}/pool/{invocation}",
                        "operator": "max_pool_upper_bound",
                        "integer_operations": int(tensor.numel()) * kernel_elements,
                        "formula": "output_elements*kernel_elements",
                        "conservative_upper_bound": True,
                    }
                )

        handles.append(module.register_forward_hook(hook))
    return handles, records


def _synthetic_batch(arm: str, candidate_arm: str) -> dict[str, Any]:
    import torch

    if arm == candidate_arm:
        return {
            "inputs": {
                "global": torch.zeros(1, 1, 3, 768, 96, 96, dtype=torch.uint8),
                "source": torch.zeros(1, 1, 3, 768, 180, 320, dtype=torch.uint8),
            },
            "masks": torch.ones(1, 768, dtype=torch.bool),
        }
    size = 160 if arm == "D160" else 96
    return {
        "inputs": torch.zeros(1, 1, 3, 768, size, size, dtype=torch.uint8),
        "masks": torch.ones(1, 768, dtype=torch.bool),
    }


def _profile_arm(args: argparse.Namespace) -> dict[str, Any]:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("full-operator profiling requires a Slurm allocation")
    import torch
    from mmengine.config import Config
    from torch.profiler import ProfilerActivity, profile

    import opentad.datasets  # noqa: F401
    from opentad.models import build_detector
    from tools.bata.zoomtoken_batch_device import prepare_zoomtoken_batch

    spec = get_matrix_spec()
    if spec.key != args.matrix_kind:
        raise ValueError("matrix-kind differs from ZOOMTOKEN_MATRIX_KIND")
    if args.arm not in spec.arms:
        raise ValueError(f"arm {args.arm!r} is outside matrix {spec.key}")
    if not torch.cuda.is_available():
        raise RuntimeError("full-operator profiling requires a Slurm GPU")
    if not args.pretrained.is_file() or not args.protocol_doc.is_file():
        raise FileNotFoundError("pretrained checkpoint or protocol document is missing")
    require_clean_commit(args.expected_commit, root_dir)
    protocol = json.loads(args.protocol_doc.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != spec.protocol_id:
        raise ValueError("protocol document does not match the selected matrix")
    validate_matrix_cell(args.config, arm=args.arm, seed=4407, spec=spec)

    cfg = Config.fromfile(args.config)
    cfg.model.backbone.custom.pretrain = str(args.pretrained.resolve())
    model = build_detector(cfg.model).to("cuda:0").eval()
    prepared = prepare_zoomtoken_batch(
        _synthetic_batch(args.arm, spec.candidate_arm), torch.device("cuda", 0)
    )
    handles, module_records = _register_module_shape_hooks(model)
    try:
        with (
            torch.inference_mode(),
            torch.cuda.amp.autocast(dtype=torch.float16, enabled=bool(cfg.solver.amp)),
            _capture_interpolation(torch) as interpolation_records,
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
            ) as execution_profile,
        ):
            model.forward_test(prepared["inputs"], prepared["masks"], metas=[{}])
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    ordered_windows = int(args.ordered_window_count)
    manifest_sha = str(args.evaluation_manifest_sha256)
    if args.manifest is not None:
        manifest = validate_full_data_manifest(args.manifest)
        ordered_windows = int(manifest["evaluation"]["ordered_window_count"])
        manifest_sha = str(manifest["manifest_sha256"])
    if ordered_windows <= 0 or len(manifest_sha) != 64:
        raise ValueError("profiling requires a positive window count and 64-char manifest SHA")

    ledger = FullOperatorLedger(arm=args.arm)
    event_ordinal = 0
    for event in execution_profile.key_averages(group_by_input_shape=True):
        key = str(event.key)
        if not key.startswith("aten::"):
            continue
        flops = int(event.flops or 0)
        if flops > 0:
            operator = AUTOMATIC_EVENT_NAMES.get(key)
            if operator is None:
                ledger.mark_unsupported(key)
            else:
                ledger.add_automatic(
                    event_id=f"trace/automatic/{event_ordinal}/{key}",
                    operator=operator,
                    integer_operations=flops * ordered_windows,
                )
                event_ordinal += 1
            if key == "aten::addmm":
                shapes = [shape for shape in event.input_shapes if shape]
                if len(shapes) >= 2:
                    output_elements = _shape_numel(shapes[1][:-1]) * int(shapes[-1][-1])
                    ledger.add_manual(
                        event_id=f"trace/manual/{event_ordinal}/addmm_bias",
                        operator="linear_bias_add_upper_bound",
                        integer_operations=int(event.count) * output_elements * ordered_windows,
                        formula="count*output_elements*ordered_window_count",
                        conservative_upper_bound=True,
                    )
                    event_ordinal += 1
            continue
        manual = _manual_profiler_rule(event)
        if manual is not None:
            operator, operations, formula, upper = manual
            if operations:
                ledger.add_manual(
                    event_id=f"trace/manual/{event_ordinal}/{key}",
                    operator=operator,
                    integer_operations=operations * ordered_windows,
                    formula=formula + "*ordered_window_count",
                    conservative_upper_bound=upper,
                )
                event_ordinal += 1
        elif key not in ZERO_ARITHMETIC_OR_WRAPPER_EVENTS:
            ledger.mark_unsupported(key)

    for record in module_records:
        record = dict(record)
        record["integer_operations"] *= ordered_windows
        record["formula"] += "*ordered_window_count"
        ledger.add_manual(**record)

    interpolation_multipliers = {
        "nearest": 1,
        "linear": 4,
        "bilinear": 8,
        "bicubic": 32,
    }
    for index, (mode, _input_elements, output_elements) in enumerate(
        interpolation_records
    ):
        if mode not in interpolation_multipliers:
            ledger.mark_unsupported(f"interpolate:{mode}")
            continue
        multiplier = interpolation_multipliers[mode]
        operator = (
            "temporal_linear_interpolation_upper_bound"
            if mode == "linear"
            else "bilinear_resize_upper_bound"
            if mode in {"bilinear", "bicubic"}
            else "elementwise_upper_bound"
        )
        ledger.add_manual(
            event_id=f"transform/interpolate/{index}/{mode}",
            operator=operator,
            integer_operations=multiplier * output_elements * ordered_windows,
            formula=f"{multiplier}*output_elements*ordered_window_count",
            conservative_upper_bound=True,
        )

    global_size = 160 if args.arm == "D160" else 96
    ledger.add_manual(
        event_id="transform/decoded_rgb_letterbox",
        operator="bilinear_resize_upper_bound",
        integer_operations=8 * 768 * 3 * global_size * global_size * ordered_windows,
        formula="8*frames*channels*output_height*output_width*ordered_window_count",
        conservative_upper_bound=True,
    )
    identity = {
        "candidate_commit": args.expected_commit,
        "protocol_sha256": sha256_file(args.protocol_doc),
        "evaluation_manifest_sha256": manifest_sha,
        "checkpoint_policy": "epoch_59_state_dict_ema_update_6000_shape_invariant",
        "dtype": "float16",
        "batch_size": 1,
        "ordered_window_count": ordered_windows,
        "config_sha256": sha256_file(args.config),
        "trace_kind": "single_fixed_shape_window_times_complete_population",
        "weights_affect_operator_count": False,
    }
    receipt = ledger.receipt(
        execution_identity=identity,
        boundary_trace=BOUNDARY_TRACE,
    )
    receipt_without_digest = dict(receipt)
    receipt_without_digest.pop("ledger_sha256")
    receipt_without_digest["full_operator_c_exec_per_window"] = (
        int(receipt["full_operator_c_exec"]) // ordered_windows
    )
    receipt_without_digest["full_operator_gflops_per_window"] = (
        receipt_without_digest["full_operator_c_exec_per_window"] / 1.0e9
    )
    receipt = receipt_without_digest
    receipt["ledger_sha256"] = canonical_sha256(receipt)
    validate_c_exec_receipt(receipt)
    atomic_publish_json(args.output, receipt)
    print(
        json.dumps(
            {
                "status": "PASS",
                "arm": args.arm,
                "full_operator_c_exec": receipt["full_operator_c_exec"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return receipt


def _compare(args: argparse.Namespace) -> dict[str, Any]:
    spec = get_matrix_spec()
    if spec.key != args.matrix_kind:
        raise ValueError("matrix-kind differs from ZOOMTOKEN_MATRIX_KIND")
    receipts: dict[str, Mapping[str, Any]] = {}
    for path in args.receipts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = str(payload.get("arm"))
        if arm in receipts:
            raise ValueError(f"duplicate C_exec receipt for {arm}")
        receipts[arm] = payload
    comparison = compare_c_exec_receipts(receipts)
    atomic_publish_json(args.output, comparison)
    print(json.dumps({"status": "PASS", **comparison}, sort_keys=True))
    return comparison


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace D2S/PA-TAD full-operator C_exec")
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile_arm = subparsers.add_parser("profile-arm")
    profile_arm.add_argument("--matrix-kind", choices=("d2s", "patad"), required=True)
    profile_arm.add_argument("--arm", required=True)
    profile_arm.add_argument("--config", type=Path, required=True)
    profile_arm.add_argument("--pretrained", type=Path, required=True)
    profile_arm.add_argument("--expected-commit", required=True)
    profile_arm.add_argument("--protocol-doc", type=Path, required=True)
    profile_arm.add_argument("--manifest", type=Path)
    profile_arm.add_argument("--evaluation-manifest-sha256", default="0" * 64)
    profile_arm.add_argument("--ordered-window-count", type=int, default=792)
    profile_arm.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--matrix-kind", choices=("d2s", "patad"), required=True)
    compare.add_argument("--receipts", type=Path, nargs=3, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "profile-arm":
        _profile_arm(args)
    else:
        _compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTOMATIC_EVENT_NAMES",
    "BOUNDARY_TRACE",
    "ZERO_ARITHMETIC_OR_WRAPPER_EVENTS",
    "_manual_profiler_rule",
    "_sort_comparison_upper_bound",
]

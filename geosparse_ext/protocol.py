"""Resolve every registered field explicitly; unfinished arms stay registered."""
from pathlib import Path
from .matrix import base


def resolved_model_protocol(model, input_shape):
    geo = model.geosparse
    source = None if geo.encoder is None else geo.encoder.source
    return dict(route=geo.route, input_shape=list(input_shape), input_positions=input_shape[-3],
                query_length=geo.query_length, native_tubelet=2 if source is None else geo.encoder.tubelet_size,
                parent_input_positions=geo.parent_frames, parent_count=input_shape[-3] // geo.parent_frames,
                native_spatial_patch=16 if source is None else source.patch_size,
                heavy_input_resolution=geo.config["source_resolution"],
                heavy_spatial_grid=[geo.config["source_resolution"] // 16] * 2,
                quota=geo.config["quota"],
                source_class=None if source is None else type(source).__name__,
                source_width=None if source is None else source.embed_dims,
                source_layers=0 if source is None else len(source.blocks),
                active_source_layers=[] if geo.encoder is None else list(geo.encoder.active_layers),
                dense_prefix_layers=0 if geo.encoder is None else geo.encoder.prefix_depth,
                source_tia_temporal_sizes=[] if source is None else [block.adapter.temporal_size for block in source.blocks if block.use_adapter],
                regular_tia_temporal_size=getattr(getattr(geo, "regular_tia", None), "temporal_size", None),
                detector_mask_rule="original frame validity" if geo.query_length == input_shape[-3] else "any valid frame per native pair",
                source_block_order=[] if source is None else ["LN1", "MHSA", "residual", "LN2", "MLP", "residual"] + (["TIA"] if any(block.use_adapter for block in source.blocks) else []),
                pe=None if source is None else "disabled control" if geo.config["geometry"] == "no_position" else "source native sinusoid; selected indices retain original PE",
                gt_to_query_scale=geo.query_length / input_shape[-3],
                heavy_cost_scope="QKV, attention and MLP only; 1 MAC = 2 FLOPs",
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def implementation_blockers(config):
    known = set(base("A")) | {"active_depth", "prefix_depth", "roi_size", "roi_count", "roi_trajectory", "roi_option",
                              "second_round_proposal", "budget_includes_both_rounds"}
    problems = [f"unknown field {key}" for key in sorted(set(config) - known)]
    required = set(base("A"))
    problems += [f"missing field {key}" for key in sorted(required - set(config))]
    if problems:
        return problems
    allowed = {
        "route": {"A", "B", "C", "DENSE", "COARSE", "DEPTH", "BCR"},
        "backbone": {"videomae_b", "videomae_l"}, "head": {"actionformer"},
        "selector": {"none", "uniform", "random", "motion", "actionness", "uncertainty", "cdf_native", "hybrid", "pg_only", "static"},
        "geometry": {"native_support", "no_position"},
        "estimator": {"none", "pg", "pg_acquisition", "task_aux_or_none"},
        "quota": {"global_zero_allowed", "per_clip_equal_quota", "global_nonzero_per_clip"}, "roi_mode": {"none"}, "source_resolution": {112, 224},
        "heavy_context": {"parent16"}, "route_refresh": {"once"}, "rounds": {1},
        "tia_scope": {"source_native"}, "cost_target": {"traced_macs"}, "detail_probe_fraction": {0.},
        "receiver": {"support_attention", "timestamp_attention", "rank_interp", "physical_interp", "concat_scatter"},
        "receiver_layers": {1, 2, 4}, "evidence_slots": {1, 4, 8},
        "coarse_variant": {"mean_shared", "learned_projection", "no_scale_embedding", "coarse_update_no_tia"},
        "fusion_variant": {"residual", "no_null", "coarse_overwrite"},
        "query_length": {384, 768}, "axis": {"T", "ST"}, "budget_mode": {"fixed", "dynamic"},
        "temporal_atom_tubelets": {1, 2, 4, 8}, "spatial_group": {1, 2, 7},
        "scout_resolution": {80, 112, 160}, "scout_width": {64, 128, 256}, "scout_temporal_stride": {1, 2, 4},
        "exploration": {"default", "none", "constant_0.1", "constant_0.5"}, "warmup_epochs": {0, 6},
        "probe_every": {0, 8, 32, 128}, "budget": {0., .25, .5, .75, 1.},
        "allow_global_zero": {True, False}, "actionness_aux": {True, False},
    }
    for key, values in allowed.items():
        if key == "source_resolution":
            values = values | {160}
        if config[key] not in values:
            problems.append(f"{key}={config[key]!r}: implementation pending")
    handled = set()
    if config["route"] == "DEPTH" and config["selector"] == "static" and config.get("active_depth") in {4, 6, 8, 10}:
        handled.add("active_depth")
    if config["route"] == "BCR" and config.get("prefix_depth") in {0, 2, 4}:
        handled.add("prefix_depth")
    for key in sorted(set(config) - set(base("A")) - handled):
        problems.append(f"{key}: registered extension pending")
    if config["route"] == "C" and (config["spatial_group"] != 2 or config["temporal_atom_tubelets"] != 1 or config["axis"] != "ST"):
        problems.append("C requires native time and mutually exclusive 2x2 coarse/fine groups")
    if config["source_resolution"] == 112 and (config["route"] != "B" or config["axis"] != "T"):
        problems.append("112 full-frame fidelity is registered only for B temporal selection")
    if config["axis"] == "ST" and (config["source_resolution"] // 16) % config["spatial_group"]:
        problems.append("spatial_group must divide the actual source-resolution patch grid")
    if config["fusion_variant"] == "feature_l2":
        problems.append("registered feature L2 alignment loss is not implemented; normalization is not alignment")
    return problems


def resolve_opentad_config(job, bindings, annotation_file):
    from mmengine.config import Config
    root = Path(__file__).resolve().parents[1]
    if job["dataset"] != "thumos14":
        raise NotImplementedError("ActivityNet/FineAction dataset protocol adapters are pending")
    problems = implementation_blockers(job["model"])
    if problems:
        raise NotImplementedError("; ".join(problems))
    cfg = Config.fromfile(str(root / "configs/adatad/thumos/e2e_thumos_videomae_b_768x1_160_adapter.py"))
    model = job["model"]
    cfg.model.pop("backbone")
    cfg.model.type = "GeoSparseDetector"
    cfg.model.geosparse = model
    cfg.model.recognition_checkpoint = bindings["checkpoints"][model["backbone"]]
    cfg.model.projection.in_channels = 256 if model["route"] in {"B", "COARSE"} else (1024 if model["backbone"] == "videomae_l" else 768)
    cfg.model.projection.max_seq_len = model["query_length"]
    for split, data in cfg.dataset.items():
        data.ann_file = str(annotation_file)
        data.class_map = bindings["class_map"]
        data.data_path = bindings["video_roots"]["training" if split == "train" else "validation"]
        pipeline = []
        for transform in data.pipeline:
            transform = dict(transform)
            resolution = 160 if model["source_resolution"] == 112 else model["source_resolution"]
            if transform["type"] == "mmaction.Resize":
                transform["scale"] = (resolution, resolution) if transform.get("keep_ratio") is False else (-1, round(resolution * 182 / 160) if split == "train" else resolution)
            elif transform["type"] == "mmaction.CenterCrop":
                transform["crop_size"] = resolution
            elif transform["type"] == "Collect":
                transform["type"] = "GeoSparseCollect"
            elif transform["type"] == "mmaction.ImgAug":
                transform["type"] = "GeoSparseImgAug"
            pipeline.append(transform)
            if transform["type"] == "LoadFrames":
                pipeline.append(dict(type="CaptureGeoSparseSupport"))
        data.pipeline = pipeline
    assert cfg.scheduler.max_epoch == 100 and cfg.scheduler.warmup_epoch == 5
    cfg.workflow.end_epoch = 60
    cfg.workflow.val_eval_interval = 5
    cfg.workflow.val_start_epoch = 4
    # The pinned upstream VideoMAE-B recipe uses 1e-4 for its adapters as well.
    cfg.optimizer.backbone.custom[0].lr = 1e-4
    return cfg

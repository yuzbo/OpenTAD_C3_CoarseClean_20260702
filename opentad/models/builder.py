from mmengine.registry import Registry
from .backbones import (
    BackboneWrapper,
    GeoRouteBackboneWrapper,
    GeoRoutePostBackboneAggregationWrapper,
)

MODELS = Registry("models")

PROJECTIONS = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
PRIOR_GENERATORS = MODELS
PROPOSAL_GENERATORS = MODELS
HEADS = MODELS
TRANSFORMERS = MODELS
LOSSES = MODELS
DETECTORS = MODELS
MATCHERS = MODELS


def build_detector(cfg):
    """Build detector."""
    return DETECTORS.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    custom_cfg = getattr(cfg, "custom", None)
    wrapper_type = (
        getattr(custom_cfg, "wrapper_type", None)
        if custom_cfg is not None
        else None
    )
    if wrapper_type == "georoute_native_packed_v1":
        return GeoRouteBackboneWrapper(cfg)
    if wrapper_type == "georoute_postbackbone_sparse_aggregation_v1":
        return GeoRoutePostBackboneAggregationWrapper(cfg)
    if wrapper_type is not None:
        raise ValueError(f"unsupported backbone custom.wrapper_type={wrapper_type!r}")
    return BackboneWrapper(cfg)


def build_projection(cfg):
    """Build projection."""
    return PROJECTIONS.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_prior_generator(cfg):
    """Build prior generator."""
    return PRIOR_GENERATORS.build(cfg)


def build_proposal_generator(cfg):
    """Build proposal generator."""
    return PROPOSAL_GENERATORS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_transformer(cfg):
    """Build transformer in DETR-like model."""
    return TRANSFORMERS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_matcher(cfg):
    """Build external classifier."""
    return MATCHERS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

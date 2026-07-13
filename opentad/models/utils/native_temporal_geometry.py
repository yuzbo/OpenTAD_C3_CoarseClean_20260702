import torch


_PATCH_LINEAGE_PROVENANCE = "raw_atoms_exact_at_patch_embed_input"
_FINAL_FEATURE_LINEAGE = "structural_upper_bound_chunk_attention_global_temporal_adapter"


def _prefix_count(mask, *, sample_idx):
    count = int(mask.sum().item())
    expected = torch.arange(mask.numel(), device=mask.device) < count
    if count <= 0 or not torch.equal(mask, expected):
        raise ValueError(f"native temporal raw mask for sample {sample_idx} must be a non-empty valid prefix")
    return count


def align_native_tubelet_geometry(
    features,
    raw_masks,
    metas,
    *,
    tubelet_size,
    expected_raw_count=None,
    expected_token_count=None,
    expected_transformer_depth=None,
    expected_adapter_indices=None,
    expected_adapter_kernel_size=None,
    expected_adapter_dilation=None,
):
    """Align raw-frame masks and metadata to an unchanged native token tensor."""
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("native temporal features must have shape [B, C, J]")
    if not isinstance(raw_masks, torch.Tensor) or raw_masks.ndim != 2:
        raise ValueError("native temporal raw masks must have shape [B, K]")
    if raw_masks.shape[0] != features.shape[0]:
        raise ValueError("native temporal feature/mask batch sizes differ")
    if not isinstance(metas, (list, tuple)) or len(metas) != features.shape[0]:
        raise ValueError("native temporal geometry requires one metadata dictionary per sample")

    tubelet_size = int(tubelet_size)
    if tubelet_size <= 0:
        raise ValueError("native temporal tubelet_size must be positive")
    raw_count = int(raw_masks.shape[1])
    token_count = int(features.shape[2])
    if raw_count != token_count * tubelet_size:
        raise ValueError(
            f"native temporal K/J contract failed: K={raw_count}, J={token_count}, tubelet_size={tubelet_size}"
        )
    if expected_raw_count is not None and raw_count != int(expected_raw_count):
        raise ValueError(f"native temporal raw count mismatch: expected {expected_raw_count}, got {raw_count}")
    if expected_token_count is not None and token_count != int(expected_token_count):
        raise ValueError(f"native temporal token count mismatch: expected {expected_token_count}, got {token_count}")

    raw_masks = raw_masks.to(dtype=torch.bool, device=features.device)
    semantic_anchor_masks = raw_masks.reshape(raw_masks.shape[0], token_count, tubelet_size).any(dim=-1)
    raw_valid_counts = []
    native_valid_counts = []
    for sample_idx, meta in enumerate(metas):
        if not isinstance(meta, dict):
            raise ValueError("native temporal metadata entries must be dictionaries")
        if meta.get("phystime_patch_embed_lineage_provenance") != _PATCH_LINEAGE_PROVENANCE:
            raise ValueError("native temporal geometry received unaudited patch-input provenance")
        if meta.get("phystime_native_final_feature_lineage") != _FINAL_FEATURE_LINEAGE:
            raise ValueError("native temporal geometry is missing the final-feature receptive-field audit")
        if meta.get("phystime_native_final_feature_support_is_exact") is not False:
            raise ValueError("native temporal geometry must not claim exact two-atom final-feature support")
        if meta.get("phystime_subsample_uses_gt") is not False:
            raise ValueError("native temporal geometry requires GT-independent within-window subsampling")
        raw_valid = _prefix_count(raw_masks[sample_idx], sample_idx=sample_idx)
        native_valid = _prefix_count(semantic_anchor_masks[sample_idx], sample_idx=sample_idx)
        expected_native_valid = (raw_valid + tubelet_size - 1) // tubelet_size
        if native_valid != expected_native_valid:
            raise ValueError("native temporal reduced mask count is inconsistent with raw valid atoms")
        semantic_value = meta.get("phystime_patch_embed_semantic_atom_mask")
        compute_value = meta.get("phystime_patch_embed_compute_atom_mask")
        if semantic_value is None or compute_value is None:
            raise ValueError("native temporal patch-input atom audit metadata is missing")
        semantic_atoms = torch.as_tensor(semantic_value, dtype=torch.bool)
        compute_atoms = torch.as_tensor(compute_value, dtype=torch.bool)
        expected_atom_shape = (token_count, tubelet_size)
        if tuple(semantic_atoms.shape) != expected_atom_shape or tuple(compute_atoms.shape) != expected_atom_shape:
            raise ValueError("native temporal patch-input atom audit must cover every J token and tubelet slot")
        if int(semantic_atoms.sum().item()) != raw_valid or not bool(compute_atoms.all().item()):
            raise ValueError("native temporal patch-input atom audit disagrees with raw valid/compute slots")
        if int(meta.get("phystime_patch_embed_padding_repeat_count", -1)) != raw_count - raw_valid:
            raise ValueError("native temporal padding-repeat audit disagrees with K valid slots")

        structural_ranges = torch.as_tensor(
            meta.get("phystime_native_final_feature_raw_slot_ranges_exclusive"),
            dtype=torch.long,
        )
        if tuple(structural_ranges.shape) != (token_count, 2):
            raise ValueError("native temporal final-feature lineage must cover every J token")
        if bool((structural_ranges[:, 0] < 0).any()) or bool((structural_ranges[:, 1] > raw_count).any()):
            raise ValueError("native temporal final-feature lineage exceeds the K raw-slot domain")
        if bool((structural_ranges[:, 1] <= structural_ranges[:, 0]).any()):
            raise ValueError("native temporal final-feature lineage ranges must be non-empty")
        padding_dependency = torch.as_tensor(
            meta.get("phystime_native_final_feature_padding_dependency_upper_bound_mask"),
            dtype=torch.bool,
        )
        if tuple(padding_dependency.shape) != (token_count,):
            raise ValueError("native temporal padding-dependency audit must cover every J token")
        expected_padding_dependency = structural_ranges[:, 1] > raw_valid
        if not torch.equal(padding_dependency, expected_padding_dependency):
            raise ValueError("native temporal padding-dependency audit disagrees with structural lineage")

        expected_structural_values = {
            "phystime_native_transformer_depth": expected_transformer_depth,
            "phystime_native_adapter_kernel_size": expected_adapter_kernel_size,
            "phystime_native_adapter_dilation": expected_adapter_dilation,
        }
        for key, expected in expected_structural_values.items():
            if expected is not None and int(meta.get(key, -1)) != int(expected):
                raise ValueError(f"native temporal structural lineage mismatch for {key}")
        if expected_adapter_indices is not None:
            actual_indices = tuple(int(value) for value in meta.get("phystime_native_adapter_indices", ()))
            if actual_indices != tuple(int(value) for value in expected_adapter_indices):
                raise ValueError("native temporal structural lineage mismatch for adapter indices")

        expected_values = {
            "phystime_raw_observation_count": raw_count,
            "phystime_raw_valid_count": raw_valid,
            "phystime_native_token_count": token_count,
            "phystime_native_valid_count": native_valid,
            "phystime_native_tubelet_size": tubelet_size,
        }
        for key, expected in expected_values.items():
            if int(meta.get(key, -1)) != int(expected):
                raise ValueError(f"native temporal metadata mismatch for {key}: expected {expected}")
        positions = meta.get("phystime_g1a_axis_positions_sec")
        if positions is None or len(positions) != native_valid:
            raise ValueError("native temporal seconds-axis positions must match the valid native token count")
        raw_positions = meta.get("selected_dense_indices")
        if raw_positions is None or len(raw_positions) != raw_valid:
            raise ValueError("K raw selected positions must remain separate from J native token positions")
        raw_valid_counts.append(raw_valid)
        native_valid_counts.append(native_valid)

    candidate_masks = semantic_anchor_masks.clone()
    aligned_features = features * semantic_anchor_masks.unsqueeze(1).to(dtype=features.dtype)
    audit = {
        "schema_version": "phystime_native_temporal_geometry_v1",
        "raw_observation_count": raw_count,
        "native_token_count": token_count,
        "tubelet_size": tubelet_size,
        "raw_valid_counts": raw_valid_counts,
        "native_valid_counts": native_valid_counts,
        "semantic_anchor_mask_reduction": "any_valid_patch_atom",
        "candidate_mask_policy": "semantic_anchor_prefix",
        "base_candidate_tensor_count": token_count,
        "base_candidate_valid_counts": native_valid_counts,
        "feature_interpolation": False,
        "patch_lineage_provenance": _PATCH_LINEAGE_PROVENANCE,
        "final_feature_lineage": _FINAL_FEATURE_LINEAGE,
        "invalid_native_features_zeroed": True,
        "padding_repeat_counts": [
            int(meta["phystime_patch_embed_padding_repeat_count"]) for meta in metas
        ],
        "valid_tokens_may_depend_on_padding_repeats": [
            bool(meta.get("phystime_native_final_feature_valid_tokens_may_depend_on_padding_repeats", False))
            for meta in metas
        ],
        "lineage_evidence_level": "exact_patch_inputs_plus_structural_receptive_field_upper_bound",
    }
    return aligned_features, candidate_masks, metas, audit

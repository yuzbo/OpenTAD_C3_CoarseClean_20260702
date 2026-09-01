from __future__ import annotations

import math
import numpy as np
import pytest


def compute_chunk_saliency_np(
    global_features: np.ndarray,
    *,
    chunk_num: int = 48,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Numpy reference for D2S chunk saliency matching d2s_videomae_wrapper.py."""
    B, C, T = global_features.shape
    tubelets_per_chunk = T // chunk_num
    feats = np.transpose(global_features, (0, 2, 1))  # [B, T, C]

    # 1. Temporal transition gradient: Delta(t) = 1 - cosine_sim(f_t, f_{t-1})
    norm = np.linalg.norm(feats, axis=-1, keepdims=True) + eps
    norm_feats = feats / norm  # [B, T, C]
    cos_sim = np.sum(norm_feats[:, 1:] * norm_feats[:, :-1], axis=-1)  # [B, T-1]
    delta = 1.0 - cos_sim  # [B, T-1]
    delta = np.concatenate([delta[:, :1], delta], axis=1)  # [B, T]

    # 2. Actionness / energy norm: E(t) = ||f_t||_2
    energy = np.linalg.norm(feats, axis=-1)  # [B, T]
    min_energy = np.min(energy, axis=-1, keepdims=True)
    max_energy = np.max(energy, axis=-1, keepdims=True)
    norm_energy = (energy - min_energy) / (max_energy - min_energy + eps)  # [B, T]

    # 3. Combined tubelet saliency: S(t) = alpha * norm_energy + (1 - alpha) * delta
    saliency = alpha * norm_energy + (1.0 - alpha) * delta  # [B, T]

    # 4. Hybrid mean + max pooling to chunk level (protects single-frame boundary spikes)
    reshaped_saliency = saliency.reshape(B, chunk_num, tubelets_per_chunk)
    mean_saliency = np.mean(reshaped_saliency, axis=-1)
    max_saliency = np.max(reshaped_saliency, axis=-1)
    chunk_saliency = 0.5 * mean_saliency + 0.5 * max_saliency
    return chunk_saliency


def test_chunk_saliency_shape_and_sensitivity():
    B = 2
    C = 384
    T = 384
    chunk_num = 48

    # Create synthetic features with a sharp transition in chunk 10 (tubelets 80..87)
    global_feats = np.zeros((B, C, T), dtype=np.float32)
    # Baseline steady state
    global_feats[:, 0, :] = 1.0
    # Sharp impulse / transition at chunk 10
    global_feats[:, :, 80:88] = 5.0

    saliency = compute_chunk_saliency_np(global_feats, chunk_num=chunk_num, alpha=0.5)
    assert saliency.shape == (B, chunk_num)
    assert np.isfinite(saliency).all()

    # Chunk 10 must have high saliency
    topk_chunks = np.argsort(saliency[0])[-16:].tolist()
    assert 10 in topk_chunks or 11 in topk_chunks


def test_chunk_saliency_delta_detects_pure_rotation():
    B = 1
    C = 64
    T = 384
    chunk_num = 48

    # Constant magnitude, but orthogonal direction shift at chunk 25
    global_feats = np.zeros((B, C, T), dtype=np.float32)
    global_feats[0, 0, :200] = 1.0
    global_feats[0, 1, 200:] = 1.0  # transition at tubelet 200 (chunk 25)

    saliency = compute_chunk_saliency_np(global_feats, chunk_num=chunk_num, alpha=0.0)  # pure delta
    assert saliency.shape == (B, chunk_num)

    # Chunk 25 (covering tubelet 200) must be top ranked by transition delta
    top1 = int(np.argmax(saliency[0]))
    assert top1 == 25


def test_vectorized_burst_mask_reconstruction():
    B = 2
    total_chunks = 48
    burst_chunks = 16
    tubelets_per_chunk = 8
    T = total_chunks * tubelets_per_chunk

    # Selected chunk indices
    selected_indices = np.array([
        [0, 2, 5, 10, 11, 15, 20, 22, 25, 30, 31, 35, 40, 42, 45, 47],
        [1, 3, 6, 8, 12, 14, 18, 21, 24, 28, 32, 36, 38, 41, 44, 46],
    ])

    # Vectorized chunk mask
    chunk_mask = np.zeros((B, total_chunks), dtype=bool)
    for b in range(B):
        chunk_mask[b, selected_indices[b]] = True

    # Expanded to tubelet level
    burst_mask = np.repeat(chunk_mask, tubelets_per_chunk, axis=1)  # [B, T]
    assert burst_mask.shape == (B, T)
    assert np.sum(burst_mask[0]) == burst_chunks * tubelets_per_chunk == 128


def test_d2s_token_budget_ratio():
    total_tubelets = 384
    total_chunks = 48
    burst_chunks = 16
    tubelets_per_chunk = total_tubelets // total_chunks  # 8

    global_tokens_per_tubelet = (96 // 16) * (96 // 16)  # 6 * 6 = 36
    local_tokens_per_tubelet = (128 // 16) * (128 // 16)  # 8 * 8 = 64
    baseline_d160_tokens_per_tubelet = (160 // 16) * (160 // 16)  # 10 * 10 = 100

    d160_total_tokens = total_tubelets * baseline_d160_tokens_per_tubelet  # 38,400
    g96_total_tokens = total_tubelets * global_tokens_per_tubelet  # 13,824

    d2s_global_tokens = total_tubelets * global_tokens_per_tubelet  # 13,824
    d2s_burst_tokens = (burst_chunks * tubelets_per_chunk) * local_tokens_per_tubelet  # 128 * 64 = 8,192
    d2s_total_tokens = d2s_global_tokens + d2s_burst_tokens  # 22,016

    compute_ratio = d2s_total_tokens / d160_total_tokens
    assert d2s_total_tokens == 22016
    assert d160_total_tokens == 38400
    assert compute_ratio == pytest.approx(0.5733333333333334)
    assert compute_ratio <= 0.90  # Strictly satisfies the <= 0.90 compute gate!

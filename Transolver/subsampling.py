"""
Node subsampling strategies: uniform, biased, recon.

Key fix from original:
  The three subsample_mode branches were written as sequential `if` blocks,
  making the `recon` compute path unreachable (the `uniform` and `biased`
  branches both `return` unconditionally before it was reached).
  Rewritten as if / elif / elif / else.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree

from preprocessing import get_distance_column, reindex_airfoil_after_mask, sample_cache_key

_MEMORY_SUBSET_CACHE: Dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _variance_score_from_fx(sample: Dict) -> np.ndarray:
    v = sample["fx"][:, :15].reshape(-1, 5, 3)
    score = np.var(v, axis=1).sum(axis=1).astype(np.float32)
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0)


def _safe_normalize_positive(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0)
    m   = arr.mean()
    if not np.isfinite(m) or m <= eps:
        return np.ones_like(arr)
    return arr / (m + eps)


def _biased_weights(sample: Dict, cfg) -> np.ndarray:
    d = np.asarray(
        get_distance_column(
            sample,
            dist_feature_index=cfg.crop_dist_feature_index,
            recompute_if_missing=cfg.recompute_distance_if_missing,
        ),
        dtype=np.float64,
    )
    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf).clip(0.0)

    near_wall = _safe_normalize_positive(1.0 / (d + 1e-6))
    near_wall = near_wall.clip(1e-12) ** float(cfg.biased_sampling_near_wall_power)

    var_norm  = _safe_normalize_positive(_variance_score_from_fx(sample).astype(np.float64))
    var_norm  = var_norm.clip(1e-12) ** float(cfg.biased_sampling_variance_power)

    mix = float(np.clip(cfg.biased_sampling_mix_near_wall, 0.0, 1.0))
    w   = mix * near_wall + (1.0 - mix) * var_norm
    w   = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0)
    if not np.any(w > 0):
        w = np.ones_like(w)
    return w


# ---------------------------------------------------------------------------
# Index selection
# ---------------------------------------------------------------------------

def _select_indices_uniform(
    sample: Dict, n_points: int, rng: np.random.Generator,
    keep_all_airfoil: bool = True,
) -> np.ndarray:
    n = sample["x"].shape[0]
    if n_points >= n:
        return np.arange(n, dtype=np.int32)
    air = (np.unique(sample["idcs_airfoil"].astype(np.int32))
           if keep_all_airfoil else np.array([], dtype=np.int32))
    remaining = np.setdiff1d(np.arange(n, dtype=np.int32), air)
    need      = max(0, n_points - air.size)
    chosen    = rng.choice(remaining, size=need, replace=False)
    return np.sort(np.concatenate([air, chosen])).astype(np.int32)


def _select_indices_biased(
    sample: Dict, n_points: int, rng: np.random.Generator, cfg
) -> np.ndarray:
    n = sample["x"].shape[0]
    if n_points >= n:
        return np.arange(n, dtype=np.int32)
    air = (np.unique(sample["idcs_airfoil"].astype(np.int32))
           if cfg.keep_all_airfoil else np.array([], dtype=np.int32))
    remaining = np.setdiff1d(np.arange(n, dtype=np.int32), air)
    need      = max(0, n_points - air.size)
    if need <= 0:
        return np.sort(air).astype(np.int32)

    w = _biased_weights(sample, cfg)[remaining]
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0)
    s = w.sum()
    p = (w / s) if (np.isfinite(s) and s > 0 and np.all(np.isfinite(w / s))) else None

    chosen = rng.choice(remaining, size=need, replace=False, p=p)
    return np.sort(np.concatenate([air, chosen])).astype(np.int32)


# ---------------------------------------------------------------------------
# Apply indices
# ---------------------------------------------------------------------------

def _apply_indices(sample: Dict, idx: np.ndarray) -> Dict:
    mask = np.zeros(sample["x"].shape[0], dtype=bool)
    mask[idx] = True
    return {
        "x":           sample["x" ][idx].astype(np.float32),
        "fx":          sample["fx"][idx].astype(np.float32),
        "y":           sample["y" ][idx].astype(np.float32),
        "idcs_airfoil": reindex_airfoil_after_mask(mask, sample["idcs_airfoil"]),
    }


# ---------------------------------------------------------------------------
# Recon helpers
# ---------------------------------------------------------------------------

def _reconstruct_from_subset(
    source_sample: Dict, subset_sample: Dict, k: int = 4
) -> np.ndarray:
    x_src = source_sample["x"]
    x_sub = subset_sample["x"]
    y_sub = subset_sample["y"].reshape(-1, 5, 3)
    k_eff = min(k, x_sub.shape[0])
    tree  = cKDTree(x_sub)
    nn_d, nn = tree.query(x_src, k=k_eff)
    if k_eff == 1:
        nn, nn_d = nn[:, None], nn_d[:, None]
    w    = 1.0 / (nn_d ** 2 + 1e-16)
    w   /= w.sum(axis=1, keepdims=True) + 1e-12
    recon = (w[:, :, None, None] * y_sub[nn]).sum(axis=1)
    return recon.reshape(x_src.shape[0], 15).astype(np.float32)


def _recon_frames_to_check(mode: str):
    if mode in ("all", "outputs_only"):
        return [0, 1, 2, 3, 4]
    return [0, 2, 4]   # "subset"


def _relative_recon_error(
    y_true_flat: np.ndarray, y_recon_flat: np.ndarray, frames_to_check
) -> float:
    yt = y_true_flat.reshape(-1, 5, 3)
    yr = y_recon_flat.reshape(-1, 5, 3)
    errs = [
        float(np.sqrt(
            np.sum((yr[:, t] - yt[:, t]) ** 2) /
            (np.sum(yt[:, t] ** 2) + 1e-12)
        ))
        for t in frames_to_check
    ]
    return max(errs)


def _disk_cache_path(cache_dir: Path, key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"subset_{key}.pkl"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def subsample_sample(
    sample:    Dict,
    n_points:  int,
    rng:       np.random.Generator,
    cfg,
    cache_dir: Path | None = None,
    cache_tag: str = "",
) -> Tuple[Dict, Dict]:
    """
    Returns (subsampled_sample, stats_dict).
    subsample_mode must be one of: uniform | biased | recon
    """
    mode = cfg.subsample_mode

    # ------------------------------------------------------------------ recon
    # Check caches first (before any branching on mode).
    if mode == "recon":
        key = sample_cache_key(
            sample,
            suffix=(
                f"{mode}_{n_points}_{cache_tag}"
                f"_{cfg.recon_rel_threshold}_{cfg.recon_check_frames_mode}"
            ),
        )
        if cfg.recon_cache_memory and key in _MEMORY_SUBSET_CACHE:
            idx = _MEMORY_SUBSET_CACHE[key]
            return _apply_indices(sample, idx), {
                "mode": "recon", "accepted": True,
                "attempts": 0, "from_cache": True,
            }
        if cfg.recon_cache_disk and cache_dir is not None:
            p = _disk_cache_path(cache_dir, key)
            if p.exists():
                with open(p, "rb") as f:
                    idx = pickle.load(f)
                if cfg.recon_cache_memory:
                    _MEMORY_SUBSET_CACHE[key] = idx
                return _apply_indices(sample, idx), {
                    "mode": "recon", "accepted": True,
                    "attempts": 0, "from_cache": True,
                }

    # ----------------------------------------------------------------- uniform
    if mode == "uniform":
        idx = _select_indices_uniform(
            sample, n_points, rng,
            keep_all_airfoil=cfg.keep_all_airfoil,
        )
        return _apply_indices(sample, idx), {
            "mode": "uniform", "accepted": True,
            "attempts": 1, "from_cache": False,
        }

    # ------------------------------------------------------------------ biased
    elif mode == "biased":
        idx = _select_indices_biased(sample, n_points, rng, cfg)
        return _apply_indices(sample, idx), {
            "mode": "biased", "accepted": True,
            "attempts": 1, "from_cache": False,
        }

    # ------------------------------------------------------------------- recon
    elif mode == "recon":
        frames = _recon_frames_to_check(cfg.recon_check_frames_mode)
        best_idx, best_err = None, float("inf")
        accepted_idx = None

        for attempt in range(cfg.recon_max_attempts):
            idx  = _select_indices_biased(sample, n_points, rng, cfg)
            sub  = _apply_indices(sample, idx)
            recon = _reconstruct_from_subset(sample, sub, k=cfg.recon_k)
            err  = _relative_recon_error(sample["y"], recon, frames)
            if err < best_err:
                best_err, best_idx = err, idx
            if err <= cfg.recon_rel_threshold:
                accepted_idx = idx
                break

        final_idx = accepted_idx if accepted_idx is not None else best_idx

        if cfg.recon_cache_memory:
            _MEMORY_SUBSET_CACHE[key] = final_idx
        if cfg.recon_cache_disk and cache_dir is not None:
            with open(_disk_cache_path(cache_dir, key), "wb") as f:
                pickle.dump(final_idx, f)

        return _apply_indices(sample, final_idx), {
            "mode":     "recon",
            "accepted": best_err <= cfg.recon_rel_threshold,
            "attempts": attempt + 1,
            "best_err": float(best_err),
            "from_cache": False,
        }

    else:
        raise ValueError(f"Unknown subsample_mode='{mode}'")
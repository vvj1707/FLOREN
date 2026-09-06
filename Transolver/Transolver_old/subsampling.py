from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from preprocessing import get_distance_column, reindex_airfoil_after_mask, sample_cache_key

_MEMORY_SUBSET_CACHE = {}


def _variance_score_from_fx(sample: Dict) -> np.ndarray:
    v_flat = sample["fx"][:, :15]
    v = v_flat.reshape(v_flat.shape[0], 5, 3)
    var_score = np.var(v, axis=1).sum(axis=1).astype(np.float32)
    var_score = np.nan_to_num(var_score, nan=0.0, posinf=0.0, neginf=0.0)
    var_score = np.clip(var_score, 0.0, None)
    return var_score


def _safe_normalize_positive(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    m = arr.mean()
    if not np.isfinite(m) or m <= eps:
        return np.ones_like(arr, dtype=np.float64)
    return arr / (m + eps)


def _biased_weights(sample: Dict, cfg) -> np.ndarray:
    d = get_distance_column(
        sample,
        dist_feature_index=cfg.crop_dist_feature_index,
        recompute_if_missing=cfg.recompute_distance_if_missing,
    )
    d = np.asarray(d, dtype=np.float64)
    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf)
    d = np.clip(d, 0.0, None)

    var_score = _variance_score_from_fx(sample).astype(np.float64)

    near_wall = 1.0 / (d + 1e-6)
    near_wall = _safe_normalize_positive(near_wall)
    near_wall = np.clip(near_wall, 1e-12, None)
    near_wall = near_wall ** float(cfg.biased_sampling_near_wall_power)

    var_norm = _safe_normalize_positive(var_score)
    var_norm = np.clip(var_norm, 1e-12, None)
    var_norm = var_norm ** float(cfg.biased_sampling_variance_power)

    mix = float(cfg.biased_sampling_mix_near_wall)
    mix = min(max(mix, 0.0), 1.0)

    w = mix * near_wall + (1.0 - mix) * var_norm
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)

    if not np.any(w > 0):
        w = np.ones_like(w, dtype=np.float64)

    return w.astype(np.float64)


def _select_indices_uniform(sample: Dict, n_points: int, rng: np.random.Generator, keep_all_airfoil: bool = True) -> np.ndarray:
    n = sample["x"].shape[0]
    if n_points >= n:
        return np.arange(n, dtype=np.int32)
    air = np.unique(sample["idcs_airfoil"].astype(np.int32)) if keep_all_airfoil else np.array([], dtype=np.int32)
    remaining = np.setdiff1d(np.arange(n, dtype=np.int32), air, assume_unique=False)
    need = max(0, n_points - air.size)
    chosen = rng.choice(remaining, size=need, replace=False)
    out = np.concatenate([air, chosen])
    out.sort()
    return out.astype(np.int32)


def _select_indices_biased(sample: Dict, n_points: int, rng: np.random.Generator, cfg) -> np.ndarray:
    n = sample["x"].shape[0]
    if n_points >= n:
        return np.arange(n, dtype=np.int32)

    air = np.unique(sample["idcs_airfoil"].astype(np.int32)) if cfg.keep_all_airfoil else np.array([], dtype=np.int32)
    remaining = np.setdiff1d(np.arange(n, dtype=np.int32), air, assume_unique=False)
    need = max(0, n_points - air.size)

    if need <= 0:
        out = air.copy()
        out.sort()
        return out.astype(np.int32)

    w = _biased_weights(sample, cfg)[remaining]
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)

    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        p = None
    else:
        p = w / s
        if not np.all(np.isfinite(p)) or p.sum() <= 0:
            p = None

    chosen = rng.choice(remaining, size=need, replace=False, p=p)
    out = np.concatenate([air, chosen])
    out.sort()
    return out.astype(np.int32)


def _apply_indices(sample: Dict, idx: np.ndarray) -> Dict:
    mask = np.zeros(sample["x"].shape[0], dtype=bool)
    mask[idx] = True
    return {
        "x": sample["x"][idx].astype(np.float32),
        "fx": sample["fx"][idx].astype(np.float32),
        "y": sample["y"][idx].astype(np.float32),
        "idcs_airfoil": reindex_airfoil_after_mask(mask, sample["idcs_airfoil"]),
    }


from scipy.spatial import cKDTree

def _reconstruct_from_subset(source_sample, subset_sample, k=4):
    x_src = source_sample["x"]
    x_sub = subset_sample["x"]
    y_sub = subset_sample["y"].reshape(subset_sample["y"].shape[0], 5, 3)

    k_eff = min(k, x_sub.shape[0])
    tree = cKDTree(x_sub)
    nn_d, nn = tree.query(x_src, k=k_eff)  # (N_src, k_eff) each

    if k_eff == 1:
        nn = nn[:, None]
        nn_d = nn_d[:, None]

    w = 1.0 / (nn_d ** 2 + 1e-16)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    y_nn = y_sub[nn]  # (N_src, k_eff, 5, 3)
    recon = (w[:, :, None, None] * y_nn).sum(axis=1)
    return recon.reshape(x_src.shape[0], 15).astype(np.float32)


def _recon_frames_to_check(mode: str):
    if mode == "all":
        return [0, 1, 2, 3, 4]
    if mode == "outputs_only":
        return [0, 1, 2, 3, 4]
    return [0, 2, 4]


def _relative_recon_error(y_true_flat: np.ndarray, y_recon_flat: np.ndarray, frames_to_check) -> float:
    yt = y_true_flat.reshape(y_true_flat.shape[0], 5, 3)
    yr = y_recon_flat.reshape(y_recon_flat.shape[0], 5, 3)
    errs = []
    for t in frames_to_check:
        num = np.sum((yr[:, t, :] - yt[:, t, :]) ** 2)
        den = np.sum(yt[:, t, :] ** 2) + 1e-12
        errs.append(float(np.sqrt(num / den)))
    return max(errs)


def _disk_cache_path(cache_dir: Path, key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"subset_{key}.pkl"


def subsample_sample(sample: Dict, n_points: int, rng: np.random.Generator, cfg, cache_dir: Path | None = None, cache_tag: str = "") -> Tuple[Dict, Dict]:
    key = sample_cache_key(
        sample,
        suffix=f"{cfg.subsample_mode}_{n_points}_{cache_tag}_{cfg.recon_rel_threshold}_{cfg.recon_check_frames_mode}",
    )

    if cfg.subsample_mode == "recon":
        if cfg.recon_cache_memory and key in _MEMORY_SUBSET_CACHE:
            idx = _MEMORY_SUBSET_CACHE[key]
            sub = _apply_indices(sample, idx)
            return sub, {"mode": "recon", "accepted": True, "attempts": 0, "from_cache": True}
        if cfg.recon_cache_disk and cache_dir is not None:
            p = _disk_cache_path(cache_dir, key)
            if p.exists():
                with open(p, "rb") as f:
                    idx = pickle.load(f)
                if cfg.recon_cache_memory:
                    _MEMORY_SUBSET_CACHE[key] = idx
                sub = _apply_indices(sample, idx)
                return sub, {"mode": "recon", "accepted": True, "attempts": 0, "from_cache": True}

    if cfg.subsample_mode == "uniform":
        idx = _select_indices_uniform(sample, n_points, rng, keep_all_airfoil=cfg.keep_all_airfoil)
        return _apply_indices(sample, idx), {"mode": "uniform", "accepted": True, "attempts": 1, "from_cache": False}

    if cfg.subsample_mode == "biased":
        idx = _select_indices_biased(sample, n_points, rng, cfg)
        return _apply_indices(sample, idx), {"mode": "biased", "accepted": True, "attempts": 1, "from_cache": False}

    if cfg.subsample_mode == "recon":
        frames_to_check = _recon_frames_to_check(cfg.recon_check_frames_mode)
        accepted_idx = None
        best_idx = None
        best_err = float("inf")

        for attempt in range(cfg.recon_max_attempts):
            idx = _select_indices_biased(sample, n_points, rng, cfg)
            sub = _apply_indices(sample, idx)
            recon = _reconstruct_from_subset(sample, sub, k=cfg.recon_k)
            err = _relative_recon_error(sample["y"], recon, frames_to_check)

            if err < best_err:
                best_err = err
                best_idx = idx

            if err <= cfg.recon_rel_threshold:
                accepted_idx = idx
                break

        if accepted_idx is None:
            accepted_idx = best_idx

        if cfg.recon_cache_memory:
            _MEMORY_SUBSET_CACHE[key] = accepted_idx
        if cfg.recon_cache_disk and cache_dir is not None:
            with open(_disk_cache_path(cache_dir, key), "wb") as f:
                pickle.dump(accepted_idx, f)

        sub = _apply_indices(sample, accepted_idx)
        return sub, {
            "mode": "recon",
            "accepted": best_err <= cfg.recon_rel_threshold,
            "attempts": attempt + 1,
            "best_err": float(best_err),
            "from_cache": False,
        }

    raise ValueError(f"Unknown subsample_mode={cfg.subsample_mode}")

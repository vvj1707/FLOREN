"""
Mesh preprocessing: distance-based cropping and related utilities.
Unchanged in logic from original; cleaned up and type-annotated.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sample_cache_key(sample: Dict, suffix: str = "") -> str:
    h = hashlib.sha1()
    h.update(np.asarray(sample["x"],           dtype=np.float32).tobytes())
    h.update(np.asarray(sample["idcs_airfoil"], dtype=np.int32 ).tobytes())
    if suffix:
        h.update(suffix.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _compute_distance_to_airfoil(
    x: np.ndarray, idcs_airfoil: np.ndarray
) -> np.ndarray:
    if idcs_airfoil.size == 0:
        return np.full((x.shape[0],), np.inf, dtype=np.float32)
    surf = x[idcs_airfoil]
    d2   = ((x[:, None, :] - surf[None, :, :]) ** 2).sum(axis=-1)
    return np.sqrt(np.min(d2, axis=1)).astype(np.float32)


def get_distance_column(
    sample: Dict,
    dist_feature_index: int = 15,
    recompute_if_missing: bool = True,
) -> np.ndarray:
    fx = sample["fx"]
    if fx.shape[1] > dist_feature_index:
        return fx[:, dist_feature_index].astype(np.float32)
    if not recompute_if_missing:
        raise ValueError("Distance feature missing and recompute_if_missing=False")
    return _compute_distance_to_airfoil(sample["x"], sample["idcs_airfoil"])


# ---------------------------------------------------------------------------
# Mask / index helpers
# ---------------------------------------------------------------------------

def reindex_airfoil_after_mask(
    mask: np.ndarray, idcs_airfoil: np.ndarray
) -> np.ndarray:
    old_to_new = -np.ones(mask.shape[0], dtype=np.int32)
    old_to_new[np.where(mask)[0]] = np.arange(mask.sum(), dtype=np.int32)
    kept = idcs_airfoil[mask[idcs_airfoil]]
    return old_to_new[kept].astype(np.int32)


def apply_mask_to_sample(sample: Dict, mask: np.ndarray) -> Dict:
    return {
        "x":           sample["x" ][mask].astype(np.float32),
        "fx":          sample["fx"][mask].astype(np.float32),
        "y":           sample["y" ][mask].astype(np.float32),
        "idcs_airfoil": reindex_airfoil_after_mask(mask, sample["idcs_airfoil"]),
    }


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def crop_sample_by_distance(
    sample: Dict,
    max_dist: float = 0.4,
    dist_feature_index: int = 15,
    recompute_if_missing: bool = True,
) -> Tuple[Dict, Dict]:
    d       = get_distance_column(sample, dist_feature_index, recompute_if_missing)
    mask    = d < max_dist
    cropped = apply_mask_to_sample(sample, mask)
    stats   = {
        "n_before":      int(sample["x"].shape[0]),
        "n_after":       int(cropped["x"].shape[0]),
        "retained_frac": float(mask.mean()),
        "airfoil_before": int(sample["idcs_airfoil"].size),
        "airfoil_after":  int(cropped["idcs_airfoil"].size),
        "crop_max_dist":  float(max_dist),
    }
    return cropped, stats


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def maybe_cache_preprocessed_samples(
    samples: List[Dict], cache_dir: Path, tag: str
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{tag}.pkl"
    with open(path, "wb") as f:
        pickle.dump(samples, f)
    return path


def load_preprocessed_cache(path: Path) -> List[Dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def preprocess_samples(
    samples:    List[Dict],
    cfg,
    split_name: str,
    cache_dir:  Path | None = None,
) -> Tuple[List[Dict], List[Dict], Path | None]:
    processed, stats = [], []
    for s in samples:
        s_out   = s
        s_stats = {
            "n_before": int(s["x"].shape[0]),
            "n_after":  int(s["x"].shape[0]),
            "retained_frac": 1.0,
        }
        if cfg.crop_enabled:
            s_out, s_stats = crop_sample_by_distance(
                s_out,
                max_dist=cfg.crop_max_dist,
                dist_feature_index=cfg.crop_dist_feature_index,
                recompute_if_missing=cfg.recompute_distance_if_missing,
            )
        processed.append(s_out)
        stats.append(s_stats)

    cache_path = None
    if cache_dir is not None and cfg.preprocess_cache_to_disk:
        cache_path = maybe_cache_preprocessed_samples(
            processed, cache_dir, f"{split_name}_preprocessed"
        )
    return processed, stats, cache_path
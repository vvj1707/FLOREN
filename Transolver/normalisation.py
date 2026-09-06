"""
Normalisation utilities.

Key fixes vs original:
  - pos (spatial coordinates) are now normalised separately and consistently
    between JAX training and the PyTorch submission wrapper.
  - velocity and distance columns in fx are normalised independently so that
    their very different scales do not interfere.
  - All statistics are derived from training data only and returned so they
    can be saved and reloaded deterministically.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


NormStats = Dict[str, np.ndarray]   # {"mu": ..., "std": ...}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _fit(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) over axis-0, with a floor on std."""
    mu  = data.mean(axis=0).astype(np.float32)
    std = (data.std(axis=0) + 1e-8).astype(np.float32)
    return mu, std


def _apply(arr: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((arr - mu) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_pos_stats(train_samples: List[Dict]) -> NormStats:
    """Fit normalisation statistics for mesh coordinates."""
    pos_all = np.concatenate([s["x"] for s in train_samples], axis=0)
    mu, std = _fit(pos_all)
    return {"mu": mu, "std": std}


def fit_vel_stats(train_samples: List[Dict]) -> NormStats:
    """
    Fit normalisation for velocity columns only (first 15 cols of fx).
    Kept separate from the distance column.
    """
    vel_all = np.concatenate([s["fx"][:, :15] for s in train_samples], axis=0)
    mu, std = _fit(vel_all)
    return {"mu": mu, "std": std}


def fit_dist_stats(train_samples: List[Dict]) -> NormStats:
    """Fit normalisation for the distance-to-airfoil column (col 15 of fx)."""
    dist_all = np.concatenate([s["fx"][:, 15:16] for s in train_samples], axis=0)
    mu, std  = _fit(dist_all)
    return {"mu": mu, "std": std}


def fit_y_stats(train_samples: List[Dict]) -> NormStats:
    """Fit normalisation for the rollout target (y)."""
    y_all = np.concatenate([s["y"] for s in train_samples], axis=0)
    mu, std = _fit(y_all)
    return {"mu": mu, "std": std}


def apply_normalisation(
    samples:    List[Dict],
    pos_stats:  NormStats,
    vel_stats:  NormStats,
    dist_stats: NormStats,
    y_stats:    NormStats,
) -> List[Dict]:
    """
    Return new sample list with all fields normalised.
    fx is reconstructed from separately-normalised velocity and distance
    columns so neither dominates the other.
    """
    out = []
    for s in samples:
        vel_norm  = _apply(s["fx"][:, :15], vel_stats["mu"],  vel_stats["std"])
        dist_norm = _apply(s["fx"][:, 15:16], dist_stats["mu"], dist_stats["std"])
        out.append({
            "x":           _apply(s["x"], pos_stats["mu"], pos_stats["std"]),
            "fx":          np.concatenate([vel_norm, dist_norm], axis=1),
            "y":           _apply(s["y"], y_stats["mu"],   y_stats["std"]),
            "idcs_airfoil": s["idcs_airfoil"].astype(np.int32),
        })
    return out


def fit_and_apply(
    train_samples: List[Dict],
    test_samples:  List[Dict],
) -> Tuple[List[Dict], List[Dict], Dict[str, NormStats]]:
    pos_stats  = fit_pos_stats(train_samples)
    vel_stats  = fit_vel_stats(train_samples)
    dist_stats = fit_dist_stats(train_samples)
    y_stats    = fit_y_stats(train_samples)

    train_norm = apply_normalisation(
        train_samples, pos_stats, vel_stats, dist_stats, y_stats
    )
    test_norm = apply_normalisation(
        test_samples, pos_stats, vel_stats, dist_stats, y_stats
    )

    return train_norm, test_norm, {
        "pos":  pos_stats,
        "vel":  vel_stats,
        "dist": dist_stats,
        "y":    y_stats,
    }


def denormalise_y(y_norm: np.ndarray, y_stats: NormStats) -> np.ndarray:
    return (y_norm * y_stats["std"] + y_stats["mu"]).astype(np.float32)
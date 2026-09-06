"""
Evaluation utilities.

Simplified vs previous version:
  - Primary metrics are train_loss and test_loss, both computed every epoch
    on subsampled meshes with no teacher forcing.
  - Component breakdown (ux, uy, uz, |u|) computed alongside both.
  - Full-mesh eval retained as an optional periodic diagnostic but no
    longer the primary logged metric.
  - nRMSE retained for paper reporting but not the primary W&B metric.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from config import OUT_FRAMES, VEL_DIM
from normalisation import NormStats, denormalise_y


# ---------------------------------------------------------------------------
# Core scalar metrics
# ---------------------------------------------------------------------------

def relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.sum((y_pred - y_true) ** 2)
    den = np.sum(y_true ** 2) + 1e-12
    return float(np.sqrt(num / den))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse  = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    scale = float(y_true.max() - y_true.min()) + 1e-12
    return rmse / scale


# ---------------------------------------------------------------------------
# Per-component breakdown
# ---------------------------------------------------------------------------

def component_losses_single(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
) -> Dict[str, float]:
    """
    Compute per-component losses for one sample.

    y_true_flat / y_pred_flat : (N, 15)  — five frames of (ux, uy, uz)

    Returns dict with keys:
      ux_rel_l2, uy_rel_l2, uz_rel_l2, speed_rel_l2,
      ux_mae,    uy_mae,    uz_mae,    speed_mae,
      total_rel_l2
    """
    yt = np.asarray(y_true_flat, dtype=np.float32).reshape(-1, 5, 3)
    yp = np.asarray(y_pred_flat, dtype=np.float32).reshape(-1, 5, 3)

    def _rl2(a, b):
        num = np.sum((b - a) ** 2)
        den = np.sum(a ** 2) + 1e-12
        return float(np.sqrt(num / den))

    def _mae(a, b):
        return float(np.mean(np.abs(b - a)))

    gt_spd = np.linalg.norm(yt, axis=-1)
    pr_spd = np.linalg.norm(yp, axis=-1)

    return {
        "ux_rel_l2":    _rl2(yt[..., 0], yp[..., 0]),
        "uy_rel_l2":    _rl2(yt[..., 1], yp[..., 1]),
        "uz_rel_l2":    _rl2(yt[..., 2], yp[..., 2]),
        "speed_rel_l2": _rl2(gt_spd,     pr_spd),
        "ux_mae":       _mae(yt[..., 0], yp[..., 0]),
        "uy_mae":       _mae(yt[..., 1], yp[..., 1]),
        "uz_mae":       _mae(yt[..., 2], yp[..., 2]),
        "speed_mae":    _mae(gt_spd,     pr_spd),
        "total_rel_l2": _rl2(
            y_true_flat.reshape(-1),
            y_pred_flat.reshape(-1),
        ),
    }


def dataset_component_losses(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute per-component losses for every sample.
    Returns dict of arrays each of shape (B,).
    """
    rows = []
    keys = None
    for yt, yp in zip(y_true_list, y_pred_list):
        row = component_losses_single(yt, yp)
        if keys is None:
            keys = list(row.keys())
        rows.append([row[k] for k in keys])
    arr = np.array(rows, dtype=np.float32)
    return {k: arr[:, i] for i, k in enumerate(keys)}


def dataset_metrics(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
) -> Dict:
    """
    Compute aggregate metrics across all samples.
    Returns relative_L2_mean/std and nRMSE_mean/std/per_sample.
    """
    rel_l2s = np.array(
        [relative_l2(yt, yp) for yt, yp in zip(y_true_list, y_pred_list)],
        dtype=np.float32,
    )
    nrmses = np.array(
        [nrmse(yt, yp) for yt, yp in zip(y_true_list, y_pred_list)],
        dtype=np.float32,
    )
    return {
        "relative_L2_mean":       float(rel_l2s.mean()),
        "relative_L2_std":        float(rel_l2s.std()),
        "relative_L2_per_sample": rel_l2s,
        "nRMSE_mean":             float(nrmses.mean()),
        "nRMSE_std":              float(nrmses.std()),
        "nRMSE_per_sample":       nrmses,
    }


# ---------------------------------------------------------------------------
# Autoregressive inference helpers
# ---------------------------------------------------------------------------

def _split_fx(fx: jnp.ndarray, in_frames: int = 5, vel_dim: int = 3):
    v_flat = fx[:, : in_frames * vel_dim]
    dist   = fx[:, in_frames * vel_dim : in_frames * vel_dim + 1]
    v_in   = v_flat.reshape(fx.shape[0], in_frames, vel_dim)
    return v_in, dist


def _apply_noslip(v: jnp.ndarray, idcs: jnp.ndarray) -> jnp.ndarray:
    return v.at[idcs].set(0.0)


def predict_autoregressive(
    model,
    params,
    sample:  Dict,
    y_stats: NormStats,
    cfg,
    t_start: int = 0,
) -> np.ndarray:
    """
    Autoregressive rollout on a single sample (any mesh size).
    Returns denormalised predictions: (N, OUT_FRAMES * VEL_DIM).
    """
    x            = jnp.asarray(sample["x"])
    fx           = jnp.asarray(sample["fx"])
    idcs_airfoil = jnp.asarray(sample["idcs_airfoil"], dtype=jnp.int32)

    v_in, _ = _split_fx(fx)
    window  = v_in

    preds = []
    for step in range(OUT_FRAMES):
        t = (jnp.array(t_start + step, dtype=jnp.int32)
             if cfg.time_input else None)
        pred_next = model.apply(
            params, pos=x, v_window=window, t=t, training=False,
            method=lambda m, **kw: m.predict_step(**kw),
        )
        if cfg.hard_noslip:
            pred_next = _apply_noslip(pred_next, idcs_airfoil)
        preds.append(np.asarray(pred_next))
        window = jnp.concatenate(
            [window[:, 1:, :], pred_next[:, None, :]], axis=1
        )

    pred_roll  = np.stack(preds, axis=1)              # (N, 5, 3)
    pred_flat  = pred_roll.reshape(pred_roll.shape[0], -1)
    pred_denorm = denormalise_y(pred_flat, y_stats)

    if cfg.hard_noslip and sample["idcs_airfoil"].size > 0:
        yp = pred_denorm.reshape(-1, OUT_FRAMES, VEL_DIM)
        yp[sample["idcs_airfoil"]] = 0.0
        pred_denorm = yp.reshape(-1, OUT_FRAMES * VEL_DIM)

    return pred_denorm.astype(np.float32)


def predict_dataset(
    model,
    params,
    samples:  List[Dict],
    y_stats:  NormStats,
    cfg,
) -> List[np.ndarray]:
    """
    Evaluate all samples, grouping by node count to minimise JAX recompiles.
    """
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s["x"].shape[0]].append(i)

    preds = [None] * len(samples)
    for _, idxs in groups.items():
        for i in idxs:
            preds[i] = predict_autoregressive(
                model, params, samples[i], y_stats, cfg
            )
    return preds


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_per_sample_metrics(
    metrics_per_sample: np.ndarray,
    path: Path,
    names: Optional[List[str]] = None,
    extra_cols: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["idx", "name", "rel_l2"]
    if extra_cols:
        fieldnames += list(extra_cols.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, v in enumerate(metrics_per_sample):
            row = {
                "idx":    i,
                "name":   names[i] if names else "",
                "rel_l2": float(v),
            }
            if extra_cols:
                for k, arr in extra_cols.items():
                    row[k] = float(arr[i])
            writer.writerow(row)
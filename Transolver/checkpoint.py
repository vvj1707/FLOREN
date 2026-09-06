"""
Mid-training checkpoint saving and loading.

Fixes vs original:
  - Checkpoints are saved every N epochs (save_checkpoint_every in config),
    not only at the end of training.
  - Loading restores both params and opt_state so training can be resumed.
  - test_samples is passed explicitly — no free-variable capture.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import jax
import numpy as np


def _ckpt_path(ckpt_dir: Path, epoch: int) -> Path:
    return ckpt_dir / f"ckpt_epoch_{epoch:05d}.pkl"


def save_checkpoint(
    ckpt_dir:  Path,
    epoch:     int,
    params,
    opt_state,
    history:   dict,
    cfg,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":     epoch,
        "params":    jax.tree_util.tree_map(np.asarray, params),
        "opt_state": jax.tree_util.tree_map(np.asarray, opt_state),
        "history":   history,
    }
    path = _ckpt_path(ckpt_dir, epoch)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    # Keep a "latest" symlink for convenience
    latest = ckpt_dir / "latest.pkl"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    return path


def load_checkpoint(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_final_artifacts(
    run_dir:      Path,
    params,
    opt_state,
    history:      dict,
    norm_stats:   dict,
    cfg,
    test_samples: list,
    y_true:       list,
    y_pred:       list,
    metrics:      dict,
) -> None:
    """
    Save everything needed to reproduce inference and inspect results.
    test_samples is passed explicitly (no free-variable capture).
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # Model weights
    params_np = jax.tree_util.tree_map(np.asarray, params)
    with open(run_dir / "params.pkl", "wb") as f:
        pickle.dump(params_np, f)

    # Norm stats
    np.savez(run_dir / "norm_stats.npz", **{
        f"{split}_{key}": arr
        for split, d in norm_stats.items()
        for key, arr in d.items()
    })

    # Training history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Config
    from dataclasses import asdict
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Predictions (ragged — variable node counts)
    def _ragged(arrays):
        obj = np.empty(len(arrays), dtype=object)
        for i, a in enumerate(arrays):
            obj[i] = a
        return obj

    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)

    np.save(results_dir / "y_true.npy",  _ragged(y_true),  allow_pickle=True)
    np.save(results_dir / "y_pred.npy",  _ragged(y_pred),  allow_pickle=True)

    # Mesh coordinates and airfoil indices per test sample
    x_obj    = _ragged([s["x"].astype(np.float32)           for s in test_samples])
    idcs_obj = _ragged([s["idcs_airfoil"].astype(np.int32)  for s in test_samples])
    np.save(results_dir / "x_points.npy",     x_obj,    allow_pickle=True)
    np.save(results_dir / "idcs_airfoil.npy", idcs_obj, allow_pickle=True)

    # Scalar metrics
    np.save(results_dir / "rel_l2_per_sample.npy",
            metrics["relative_L2_per_sample"])
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "relative_L2_mean": float(metrics["relative_L2_mean"]),
                "relative_L2_std":  float(metrics["relative_L2_std"]),
            },
            f, indent=2,
        )
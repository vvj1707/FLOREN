from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np


def relative_l2_rollout(pred_rollout: jnp.ndarray, target_rollout: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    if mask is not None:
        pred_rollout = pred_rollout * mask[:, None, None]
        target_rollout = target_rollout * mask[:, None, None]
    num = jnp.sum((pred_rollout - target_rollout) ** 2)
    den = jnp.sum(target_rollout ** 2) + 1e-12
    return jnp.sqrt(num / den)


def mae_rollout(pred_rollout: jnp.ndarray, target_rollout: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    if mask is not None:
        pred_rollout = pred_rollout * mask[:, None, None]
        target_rollout = target_rollout * mask[:, None, None]
    return jnp.mean(jnp.abs(pred_rollout - target_rollout))


def relative_l2_dataset(y_true, y_pred):
    num = np.sum((y_pred - y_true) ** 2, axis=(1, 2))
    den = np.sum(y_true ** 2, axis=(1, 2)) + 1e-12
    per_sample = np.sqrt(num / den)
    return float(per_sample.mean()), per_sample


def compute_noslip_error(y_pred: np.ndarray, samples: List[Dict]) -> float:
    errs = []
    for i, s in enumerate(samples):
        air = s["idcs_airfoil"]
        if air.size == 0:
            continue
        yp = y_pred[i].reshape(y_pred.shape[1], 5, 3) if y_pred.ndim == 4 else y_pred[i].reshape(y_pred[i].shape[0], 5, 3)
        errs.append(float(np.mean(np.abs(yp[air]))))
    return float(np.mean(errs)) if errs else 0.0


def save_per_sample_metrics(per_sample: np.ndarray, out_path: Path, names: List[str] | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("sample_idx,file,rel_l2\n")
        for i, v in enumerate(per_sample):
            name = names[i] if names is not None else f"sample_{i}"
            f.write(f"{i},{name},{float(v):.8f}\n")


def save_json(data: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
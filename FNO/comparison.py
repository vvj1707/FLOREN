"""
comparison.py

Baseline comparisons:
- Persistence baseline (repeat last observed frame for 5 outputs)
"""

import numpy as np
from metrics import relative_l2_dataset, rmse_dataset, mae_dataset


def persistence_baseline(samples):
    """
    sample["fx"] = (N,16), first 15 channels are input vel history flattened:
      [v_t0(3), v_t1(3), v_t2(3), v_t3(3), v_t4(3)]
    baseline predicts outputs all equal to v_t4.
    returns: (B,N,15)
    """
    preds = []
    for s in samples:
        fx = s["fx"][:, :15]  # (N,15)
        last = fx[:, 12:15]   # last observed frame (N,3)
        pred = np.concatenate([last, last, last, last, last], axis=1)  # (N,15)
        preds.append(pred.astype(np.float32))
    return np.stack(preds, axis=0)


def evaluate_baseline(samples):
    y_true = np.stack([s["y"] for s in samples], axis=0).astype(np.float32)
    y_pred = persistence_baseline(samples)

    rel_mean, rel_per = relative_l2_dataset(y_true, y_pred)
    return {
        "relative_L2_mean": float(rel_mean),
        "relative_L2_std": float(rel_per.std()),
        "RMSE": rmse_dataset(y_true, y_pred),
        "MAE": mae_dataset(y_true, y_pred),
        "relative_L2_per_sample": rel_per,
    }
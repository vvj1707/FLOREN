"""
metrics.py

Point-level metrics aligned with competition evaluation style.
"""

import numpy as np


def relative_l2_dataset(y_true, y_pred):
    """
    y_true, y_pred: (B,N,15)
    """
    num = np.sum((y_pred - y_true) ** 2, axis=(1, 2))
    den = np.sum(y_true ** 2, axis=(1, 2)) + 1e-12
    per_sample = np.sqrt(num / den)
    return float(per_sample.mean()), per_sample


def rmse_dataset(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae_dataset(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))
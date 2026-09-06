"""
loss.py

Losses for grid training and point-level evaluation consistency.
"""

import jax.numpy as jnp


def relative_l2_grid(pred, target, mask=None):
    """
    pred/target: (B,C,gx,gy,gz)
    mask: (B,1,gx,gy,gz) or None
    """
    if mask is not None:
        pred = pred * mask
        target = target * mask
    num = jnp.sum((pred - target) ** 2)
    den = jnp.sum(target ** 2) + 1e-12
    return jnp.sqrt(num / den)


def mae_grid(pred, target, mask=None):
    if mask is not None:
        pred = pred * mask
        target = target * mask
    return jnp.mean(jnp.abs(pred - target))
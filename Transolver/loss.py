"""
Loss functions for autoregressive rollout training.

Fixes vs previous version:
  - relative_l2_step: NaN/Inf guard — returns 0 (zero gradient) instead
    of propagating NaN through the parameter tree.
  - rollout_loss: per-step losses individually guarded before weighting.
  - build_step_weights: unchanged.
  - divergence_penalty: unchanged.
"""
from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Step weights
# ---------------------------------------------------------------------------

def build_step_weights(
    n_steps:    int,
    gamma:      float = 1.0,
    supervised: Sequence[int] = (),
) -> jnp.ndarray:
    """
    Returns (n_steps,) weight vector.
    w_t = gamma^t, zeroed for steps not in `supervised` (pushforward).
    Empty supervised → all steps used.
    """
    w = jnp.array([gamma ** t for t in range(n_steps)], dtype=jnp.float32)
    if supervised:
        mask = jnp.zeros(n_steps, dtype=jnp.float32)
        for t in supervised:
            mask = mask.at[t].set(1.0)
        w = w * mask
    total = w.sum()
    w = jnp.where(total > 0, w / total, jnp.ones_like(w) / n_steps)
    return w


# ---------------------------------------------------------------------------
# Per-step relative L2
# ---------------------------------------------------------------------------

def relative_l2_step(
    pred:   jnp.ndarray,
    target: jnp.ndarray,
    mask:   Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Relative L2 for a single timestep.

    NaN/Inf guard: if pred contains non-finite values (e.g. from attention
    overflow), returns 0.0 so the gradient contribution is zero rather than
    corrupting the entire parameter tree.
    """
    diff = pred - target
    if mask is not None:
        diff   = diff   * mask[:, None]
        target = target * mask[:, None]

    num  = jnp.sum(diff ** 2)
    den  = jnp.sum(target ** 2) + 1e-12
    loss = jnp.sqrt(num / den)

    # If loss is non-finite, return 0 — zero gradient, not NaN gradient
    return jnp.where(jnp.isfinite(loss), loss, jnp.zeros_like(loss))


# ---------------------------------------------------------------------------
# Rollout loss
# ---------------------------------------------------------------------------

def rollout_loss(
    preds:        jnp.ndarray,
    targets:      jnp.ndarray,
    step_weights: jnp.ndarray,
    mask:         Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Weighted sum of per-step relative L2 losses.
    Each step is individually guarded so one bad step cannot NaN the whole loss.

    preds   : (n_steps, N, VEL_DIM)
    targets : (n_steps, N, VEL_DIM)
    """
    n_steps = preds.shape[0]
    per_step = jnp.stack([
        relative_l2_step(preds[t], targets[t], mask)
        for t in range(n_steps)
    ])                                                          # (n_steps,)

    # Guard per-step losses individually
    per_step = jnp.where(
        jnp.isfinite(per_step), per_step, jnp.zeros_like(per_step)
    )

    return jnp.sum(per_step * step_weights)


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

def mae_rollout(
    preds:   jnp.ndarray,
    targets: jnp.ndarray,
    mask:    Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    diff = jnp.abs(preds - targets)
    if mask is not None:
        diff = diff * mask[None, :, None]
    result = jnp.mean(diff)
    return jnp.where(jnp.isfinite(result), result, jnp.zeros_like(result))


# ---------------------------------------------------------------------------
# Divergence regularisation
# ---------------------------------------------------------------------------

def divergence_penalty(
    v:   jnp.ndarray,
    pos: jnp.ndarray,
    k:   int = 6,
) -> jnp.ndarray:
    dx  = pos[None, :, :] - pos[:, None, :]
    dv  = v  [None, :, :] - v  [:, None, :]
    d2  = jnp.sum(dx ** 2, axis=-1) + 1e-12
    neg_d2 = -d2
    _, nn_idx = jax.lax.top_k(neg_d2, k)
    dx_nn = dx[jnp.arange(pos.shape[0])[:, None], nn_idx]
    dv_nn = dv[jnp.arange(v.shape[0])  [:, None], nn_idx]
    d2_nn = d2[jnp.arange(d2.shape[0]) [:, None], nn_idx]
    div_contrib = jnp.sum(dv_nn * dx_nn, axis=-1) / d2_nn
    div_i       = div_contrib.sum(axis=1)
    result      = jnp.mean(div_i ** 2)
    return jnp.where(jnp.isfinite(result), result, jnp.zeros_like(result))
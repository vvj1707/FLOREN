"""
Slice-based physics attention for unstructured meshes.
O(N·G) complexity where G = slice_num << N.

Fixes vs previous version:
  - temperature shape corrected from (H,1,1) to (1,H,1).
  - Attention denominator floor raised from 1e-5 to 1e-2.
  - Slice weights clamped and renormalised after softmax to prevent
    near-zero slice occupancy from causing division explosion.
  - Output clamped before projection to catch any residual overflow.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


class PhysicsAttention(nn.Module):
    hidden_dim: int
    num_heads:  int
    slice_num:  int
    dropout:    float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        x : (N, C)   — single sample, no batch dim
        returns (N, C)
        """
        N, C = x.shape
        H    = self.num_heads
        D    = self.hidden_dim // H
        G    = self.slice_num

        # ── project to per-head streams ──────────────────────────────────────
        fx = nn.Dense(self.hidden_dim)(x).reshape(N, H, D)   # (N, H, D)
        xm = nn.Dense(self.hidden_dim)(x).reshape(N, H, D)   # (N, H, D)

        # ── temperature: (1, H, 1) broadcasts against (N, H, G) ─────────────
        temperature = self.param(
            "temperature",
            lambda rng, shape: jnp.ones(shape) * 0.5,
            (1, H, 1),
        )
        # Clamp temperature so it cannot collapse to zero
        temperature = jnp.clip(temperature, 0.01, 10.0)

        # ── slice assignment weights ─────────────────────────────────────────
        raw_slice = nn.Dense(G, use_bias=False)(xm)           # (N, H, G)
        sw = jax.nn.softmax(raw_slice / temperature, axis=-1) # (N, H, G)

        # Clamp: prevent any slice weight being exactly 0 or 1
        # This stops near-empty slices from causing sn → 0 division explosion
        sw = jnp.clip(sw, 1e-6, 1.0)
        # Renormalise rows after clamping so they still sum to 1
        sw = sw / (sw.sum(axis=-1, keepdims=True) + 1e-8)     # (N, H, G)

        # ── aggregate node features into slices ──────────────────────────────
        sn = sw.sum(axis=0)                                    # (H, G)
        st = jnp.einsum("nhd,nhg->hgd", fx, sw)               # (H, G, D)

        # Raised floor: 1e-2 instead of 1e-5
        # Prevents amplification when a slice has very few assigned nodes
        st = st / (sn[:, :, None] + 1e-2)                     # (H, G, D)

        # Clamp aggregated slice tokens to prevent overflow into attention
        st = jnp.clip(st, -1e4, 1e4)

        # ── slice-level self-attention ───────────────────────────────────────
        to_q = nn.Dense(D, use_bias=False)
        to_k = nn.Dense(D, use_bias=False)
        to_v = nn.Dense(D, use_bias=False)

        q = to_q(st)   # (H, G, D)
        k = to_k(st)
        v = to_v(st)

        scale = D ** -0.5
        attn  = jax.nn.softmax(
            jnp.einsum("hqd,hkd->hqk", q, k) * scale, axis=-1
        )                                                       # (H, G, G)

        if self.dropout > 0.0 and training:
            attn = nn.Dropout(self.dropout)(attn, deterministic=False)

        out_slice = jnp.einsum("hqk,hkd->hqd", attn, v)       # (H, G, D)

        # ── scatter back to nodes ────────────────────────────────────────────
        out_nodes = jnp.einsum("hgd,nhg->nhd", out_slice, sw)  # (N, H, D)
        out_nodes = out_nodes.reshape(N, self.hidden_dim)       # (N, C)

        # Clamp before final projection
        out_nodes = jnp.clip(out_nodes, -1e4, 1e4)

        return nn.Dense(self.hidden_dim)(out_nodes)
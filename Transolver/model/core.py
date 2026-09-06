"""
TransolverAR backbone — JAX/Flax implementation.

Fixes vs previous version:
  - NaN guards after every block: jnp.where(isfinite, h, zeros).
  - Activation clipping inside TransolverBlock residuals.
  - TemporalConv replaced with learned (W,W) mixing matrix —
    removes N from static conv shape, eliminating cuDNN recompilation.
  - Geometry-conditioned bias MLP on pos replaces scalar placeholder.
  - Timestep sinusoidal + learned embedding when cfg.time_input=True.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from .attention import PhysicsAttention


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    hidden_dim: int
    out_dim:    int
    n_layers:   int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        for _ in range(self.n_layers):
            residual = x
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
            x = x + residual
        return nn.Dense(self.out_dim)(x)


# ---------------------------------------------------------------------------
# Temporal mixing
# ---------------------------------------------------------------------------

class TemporalConv(nn.Module):
    """
    Learned (W, W) mixing matrix over the window axis.
    Replaces depthwise Conv1D which baked N into feature_group_count,
    causing cuDNN recompilation for every mesh size.
    Input/output: (N, W, VEL_DIM) — shape preserved, residual connection.
    """
    window_size: int
    vel_dim:     int = 3

    @nn.compact
    def __call__(self, v_window: jnp.ndarray) -> jnp.ndarray:
        W   = self.window_size
        mix = self.param(
            "temporal_mix",
            nn.initializers.orthogonal(),
            (W, W),
        )
        mix = jax.nn.softmax(mix, axis=-1)                     # (W, W)
        out = jnp.einsum("nwd,tw->ntd", v_window, mix)         # (N, W, D)
        return v_window + out


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    half  = dim // 2
    freqs = jnp.exp(
        -math.log(10_000) *
        jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1)
    )
    args = t.astype(jnp.float32) * freqs
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class TimestepEmbedding(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        emb = sinusoidal_embedding(t, self.hidden_dim)
        emb = nn.Dense(self.hidden_dim)(emb)
        emb = nn.gelu(emb)
        emb = nn.Dense(self.hidden_dim)(emb)
        return emb


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransolverBlock(nn.Module):
    hidden_dim: int
    num_heads:  int
    slice_num:  int
    mlp_ratio:  int   = 2
    dropout:    float = 0.0
    out_dim:    int   = 3
    last_layer: bool  = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        # Clip activations entering the block to prevent explosion
        # propagating through residual connections
        x = jnp.clip(x, -1e4, 1e4)

        # Attention residual
        attn_out = PhysicsAttention(
            hidden_dim = self.hidden_dim,
            num_heads  = self.num_heads,
            slice_num  = self.slice_num,
            dropout    = self.dropout,
        )(nn.LayerNorm()(x), training=training)

        # Guard attention output before adding to residual
        attn_out = jnp.where(jnp.isfinite(attn_out), attn_out, jnp.zeros_like(attn_out))
        x = x + attn_out

        # MLP residual
        mlp_out = MLP(
            hidden_dim = self.hidden_dim * self.mlp_ratio,
            out_dim    = self.hidden_dim,
            n_layers   = 0,
        )(nn.LayerNorm()(x))

        # Guard MLP output before adding to residual
        mlp_out = jnp.where(jnp.isfinite(mlp_out), mlp_out, jnp.zeros_like(mlp_out))
        x = x + mlp_out

        if self.last_layer:
            x = nn.Dense(self.out_dim)(nn.LayerNorm()(x))

        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TransolverCoreConfig:
    space_dim:   int   = 3
    window_size: int   = 5
    vel_dim:     int   = 3
    hidden_dim:  int   = 256
    n_layers:    int   = 6
    n_heads:     int   = 8
    slice_num:   int   = 64
    mlp_ratio:   int   = 2
    dropout:     float = 0.0
    time_input:  bool  = True
    n_rollout:   int   = 5


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class TransolverCore(nn.Module):
    cfg: TransolverCoreConfig

    def setup(self):
        c = self.cfg
        self.temporal_conv = TemporalConv(
            window_size=c.window_size, vel_dim=c.vel_dim
        )
        c_in = c.space_dim + c.window_size * c.vel_dim
        self.encoder = MLP(
            hidden_dim=c.hidden_dim * 2, out_dim=c.hidden_dim, n_layers=0
        )
        self.geo_bias_mlp = MLP(
            hidden_dim=c.hidden_dim, out_dim=c.hidden_dim, n_layers=0
        )
        if c.time_input:
            self.time_emb = TimestepEmbedding(hidden_dim=c.hidden_dim)

        self.blocks = [
            TransolverBlock(
                hidden_dim  = c.hidden_dim,
                num_heads   = c.n_heads,
                slice_num   = c.slice_num,
                mlp_ratio   = c.mlp_ratio,
                dropout     = c.dropout,
                out_dim     = c.vel_dim,
                last_layer  = (i == c.n_layers - 1),
            )
            for i in range(c.n_layers)
        ]

    def encode(
        self,
        pos:      jnp.ndarray,
        v_window: jnp.ndarray,
        t:        Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        N = pos.shape[0]

        v_w  = self.temporal_conv(v_window)
        feat = jnp.concatenate([pos, v_w.reshape(N, -1)], axis=-1)
        h    = self.encoder(feat)

        # Geometry-conditioned bias
        geo  = self.geo_bias_mlp(pos)
        geo  = jnp.where(jnp.isfinite(geo), geo, jnp.zeros_like(geo))
        h    = h + geo

        # Timestep conditioning
        if self.cfg.time_input and t is not None:
            t_emb = self.time_emb(t)
            t_emb = jnp.where(jnp.isfinite(t_emb), t_emb, jnp.zeros_like(t_emb))
            h = h + t_emb[None, :]

        # Guard encoder output
        h = jnp.where(jnp.isfinite(h), h, jnp.zeros_like(h))
        return h

    def predict_step(
        self,
        pos:      jnp.ndarray,
        v_window: jnp.ndarray,
        t:        Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        h = self.encode(pos, v_window, t=t, training=training)

        for block in self.blocks:
            h = block(h, training=training)
            # Per-block NaN guard: if a block produces NaN/Inf,
            # replace with zeros so subsequent blocks see finite input.
            # The loss will be 1.0 (zero prediction) for this sample,
            # which produces a zero gradient — bad sample is skipped
            # cleanly rather than corrupting the parameter tree.
            h = jnp.where(jnp.isfinite(h), h, jnp.zeros_like(h))

        return h

    def rollout(
        self,
        pos:     jnp.ndarray,
        v_init:  jnp.ndarray,
        n_steps: int,
        t_start: int = 0,
        training: bool = False,
    ) -> jnp.ndarray:
        window = v_init
        preds  = []
        for step in range(n_steps):
            t = (jnp.array(t_start + step, dtype=jnp.int32)
                 if self.cfg.time_input else None)
            v_next = self.predict_step(pos, window, t=t, training=training)
            preds.append(v_next)
            window = jnp.concatenate(
                [window[:, 1:, :], v_next[:, None, :]], axis=1
            )
        return jnp.stack(preds, axis=0)
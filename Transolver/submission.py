"""
GRaM @ ICLR 2026 submission wrapper — PyTorch.

Consistent with JAX training:
  - mlp_ratio = 2  (matches training config)
  - pos is normalised before the model (matches normalisation.py)
  - vel and dist normalised separately (matches normalisation.py)
  - Timestep conditioning active
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import math

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_

from einops import rearrange


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int, n_layers: int = 1):
        super().__init__()
        self.pre  = nn.Sequential(nn.Linear(n_in, n_hidden), nn.GELU())
        self.body = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU())
            for _ in range(n_layers)
        ])
        self.post = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = self.pre(x)
        for layer in self.body:
            x = layer(x) + x
        return self.post(x)


class _PhysicsAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int,
                 dropout: float, slice_num: int):
        super().__init__()
        inner        = dim_head * heads
        self.heads   = heads
        self.dh      = dim_head
        self.scale   = dim_head ** -0.5
        self.temp    = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.proj_fx = nn.Linear(dim, inner)
        self.proj_xm = nn.Linear(dim, inner)
        self.proj_sl = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.proj_sl.weight)
        self.to_q    = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k    = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v    = nn.Linear(dim_head, dim_head, bias=False)
        self.out     = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.drop    = nn.Dropout(dropout)
        self.sm      = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, _ = x.shape
        H, D    = self.heads, self.dh
        fx = self.proj_fx(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
        xm = self.proj_xm(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
        sw = self.sm(self.proj_sl(xm) / self.temp)          # B H N G
        sn = sw.sum(2)
        st = torch.einsum("bhnc,bhng->bhgc", fx, sw)
        st = st / (sn + 1e-5).unsqueeze(-1)
        q, k, v = self.to_q(st), self.to_k(st), self.to_v(st)
        attn = self.drop(self.sm(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        out  = torch.matmul(attn, v)
        out  = rearrange(
            torch.einsum("bhgc,bhng->bhnc", out, sw), "b h n d -> b n (h d)"
        )
        return self.out(out)


class _TemporalConv(nn.Module):
    """Depthwise 1-D conv over the window axis — same as JAX training."""
    def __init__(self, window_size: int, vel_dim: int = 3):
        super().__init__()
        self.W = window_size
        self.D = vel_dim
        self.conv = nn.Conv1d(
            in_channels=vel_dim, out_channels=vel_dim,
            kernel_size=min(3, window_size), padding="same",
            groups=vel_dim, bias=False,
        )

    def forward(self, v_window: torch.Tensor) -> torch.Tensor:
        # v_window: (B, N, W, D)
        B, N, W, D = v_window.shape
        x = v_window.reshape(B * N, D, W)
        x = self.conv(x).reshape(B, N, W, D)
        return v_window + x


def _sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10_000) *
        torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


class _TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.proj(_sinusoidal_emb(t, self.dim))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float,
                 mlp_ratio: int, slice_num: int,
                 last: bool = False, out_dim: int = 3):
        super().__init__()
        self.last = last
        self.ln1  = nn.LayerNorm(dim)
        self.attn = _PhysicsAttention(
            dim, heads, dim // heads, dropout, slice_num)
        self.ln2  = nn.LayerNorm(dim)
        self.mlp  = _MLP(dim, dim * mlp_ratio, dim, n_layers=0)
        if last:
            self.ln3  = nn.LayerNorm(dim)
            self.proj = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.attn(self.ln1(x)) + x
        x = self.mlp (self.ln2(x)) + x
        if self.last:
            x = self.proj(self.ln3(x))
        return x


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

@dataclass
class _Cfg:
    window_size: int   = 5
    hidden_dim:  int   = 256
    n_layers:    int   = 6
    n_heads:     int   = 8
    slice_num:   int   = 64
    mlp_ratio:   int   = 2      # consistent with JAX training
    dropout:     float = 0.0
    n_rollout:   int   = 5
    time_input:  bool  = True


class _Core(nn.Module):
    def __init__(self, cfg: _Cfg):
        super().__init__()
        self.cfg      = cfg
        c_in          = 3 + cfg.window_size * 3
        self.temp_conv = _TemporalConv(cfg.window_size)
        self.encoder  = _MLP(c_in, cfg.hidden_dim * 2, cfg.hidden_dim, n_layers=0)
        self.geo_bias = _MLP(3, cfg.hidden_dim, cfg.hidden_dim, n_layers=0)
        if cfg.time_input:
            self.time_emb = _TimestepEmbedding(cfg.hidden_dim)
        self.blocks   = nn.ModuleList([
            _Block(cfg.hidden_dim, cfg.n_heads, cfg.dropout,
                   cfg.mlp_ratio, cfg.slice_num,
                   last=(i == cfg.n_layers - 1), out_dim=3)
            for i in range(cfg.n_layers)
        ])
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def predict_step(self, pos, v_window, t=None):
        """pos (B,N,3), v_window (B,N,W,3), t scalar → (B,N,3)"""
        B, N = pos.shape[:2]
        vw   = self.temp_conv(v_window)
        feat = torch.cat([pos, vw.reshape(B, N, -1)], dim=-1)
        h    = self.encoder(feat) + self.geo_bias(pos)
        if self.cfg.time_input and t is not None:
            t_emb = self.time_emb(t.expand(B))   # (B, hidden)
            h     = h + t_emb.unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return h

    def rollout(self, pos, v_init, n_steps=None, t_start=0):
        n_steps = n_steps or self.cfg.n_rollout
        window  = v_init.clone()
        preds   = []
        for step in range(n_steps):
            t      = torch.tensor(t_start + step, device=pos.device) \
                     if self.cfg.time_input else None
            v_next = self.predict_step(pos, window, t)
            preds.append(v_next)
            window = torch.cat([window[:, :, 1:], v_next.unsqueeze(2)], dim=2)
        return torch.stack(preds, dim=1)   # (B, n_steps, N, 3)


# ---------------------------------------------------------------------------
# Public submission wrapper
# ---------------------------------------------------------------------------

class TransolverAR(nn.Module):
    """
    GRaM @ ICLR 2026 submission wrapper.

    forward(t, pos, idcs_airfoil, velocity_in) → velocity_out
      t             (batch, 10)      — unused (kept for interface compatibility)
      pos           (batch, N, 3)    — raw mesh coordinates
      idcs_airfoil  list[Tensor]     — unused
      velocity_in   (batch, 5, N, 3) — raw input velocity window
      returns       (batch, 5, N, 3) — predicted next 5 steps (raw)
    """

    def __init__(self):
        super().__init__()
        cfg         = _Cfg()
        self._inner = _Core(cfg)

        root = Path(__file__).parent
        sd   = torch.load(root / "state_dict.pt",
                          map_location="cpu", weights_only=True)
        self._inner.load_state_dict(sd)

        # Norm stats saved by normalisation.py → norm_stats.npz
        ns = np.load(root / "norm_stats.npz")
        self.register_buffer("pos_mean",  torch.from_numpy(ns["pos_mu"]))
        self.register_buffer("pos_std",   torch.from_numpy(ns["pos_std"]))
        self.register_buffer("vel_mean",  torch.from_numpy(ns["vel_mu"]))
        self.register_buffer("vel_std",   torch.from_numpy(ns["vel_std"]))

        self.eval()

    @torch.no_grad()
    def forward(self, t, pos, idcs_airfoil, velocity_in):
        # pos:          (B, N, 3)
        # velocity_in:  (B, 5, N, 3)
        pos_n = (pos - self.pos_mean) / self.pos_std          # (B, N, 3)
        v_n   = ((velocity_in.permute(0, 2, 1, 3)             # (B, N, 5, 3)
                  - self.vel_mean) / self.vel_std)
        preds_n = self._inner.rollout(pos_n, v_n, n_steps=5)  # (B, 5, N, 3)
        return preds_n * self.vel_std + self.vel_mean
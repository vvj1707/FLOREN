"""
config.py

Configuration + model + data conversion helpers for Warped-IFW FNO training.
Custom JAX training loop using scirex-zenteiq FNO:
    from operators.models.fno_jax import FNO
"""

from pathlib import Path
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from operators.models.fno_jax import FNO

# -------------------------
# Paths
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "trained_model"
RESULTS_DIR = SCRIPT_DIR / "results"

# -------------------------
# Dataset / split
# -------------------------
TEST_FRAC = 0.15
RANDOM_SEED = 42

# -------------------------
# Grid setup
# -------------------------
GRID_X = 64
GRID_Y = 32
GRID_Z = 32

# channels
IN_CH = 17    # 15 vel_in + 1 distance + 1 occupancy
OUT_CH = 15   # 5 future frames * 3 velocity components

# -------------------------
# FNO model config
# -------------------------
FNO_N_MODES = (16, 12, 12)        # must be <= half-ish resolution per axis
FNO_HIDDEN = 48
FNO_LAYERS = 4
FNO_USE_GRID_EMBED = False         # we already provide coords via channels optionally
FNO_DROPOUT = 0.0                  # not directly in FNO ctor; kept for consistency

# -------------------------
# Optimization
# -------------------------
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-6
WARMUP_EPOCHS = 5
LR_FLOOR = 1e-6
GRAD_CLIP = 1.0

DEVICE = jax.default_backend()


def seed_all(seed=RANDOM_SEED):
    np.random.seed(seed)


def make_schedule(
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    warmup_epochs=WARMUP_EPOCHS,
    steps_per_epoch=100,
    lr_floor=LR_FLOOR,
):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    decay_steps = max(1, (epochs - warmup_epochs) * steps_per_epoch)

    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    cosine = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=decay_steps,
        alpha=lr_floor / learning_rate,
    )
    return optax.join_schedules([warmup, cosine], [warmup_steps])


def build_fno():
    positional_embedding = None if not FNO_USE_GRID_EMBED else "grid"
    return FNO(
        n_modes=FNO_N_MODES,
        in_channels=IN_CH,
        out_channels=OUT_CH,
        hidden_channels=FNO_HIDDEN,
        n_layers=FNO_LAYERS,
        positional_embedding=positional_embedding,
        use_channel_mlp=True,
        domain_padding=None,
    )


# -------------------------
# Geometry/grid helpers
# -------------------------
def build_grid_spec(samples, gx=GRID_X, gy=GRID_Y, gz=GRID_Z):
    all_pos = np.concatenate([s["x"] for s in samples], axis=0)
    xyz_min = all_pos.min(axis=0).astype(np.float32)
    xyz_max = all_pos.max(axis=0).astype(np.float32)
    return {
        "xyz_min": xyz_min,
        "xyz_max": xyz_max,
        "gx": int(gx),
        "gy": int(gy),
        "gz": int(gz),
    }


def _normalize_pos(pos, xyz_min, xyz_max):
    return (pos - xyz_min[None, :]) / (xyz_max[None, :] - xyz_min[None, :] + 1e-12)


def points_to_grid(sample, grid_spec):
    """
    sample:
      x   (N,3)
      fx  (N,16) = 15 vel + 1 dist
      y   (N,15)

    returns:
      Xg  (IN_CH,gx,gy,gz)
      Yg  (OUT_CH,gx,gy,gz)
      Mg  (1,gx,gy,gz) occupancy mask
    """
    pos = sample["x"].astype(np.float32)
    fx = sample["fx"].astype(np.float32)
    y = sample["y"].astype(np.float32)

    gx, gy, gz = grid_spec["gx"], grid_spec["gy"], grid_spec["gz"]
    xyz_min, xyz_max = grid_spec["xyz_min"], grid_spec["xyz_max"]

    p = _normalize_pos(pos, xyz_min, xyz_max)
    ix = np.clip((p[:, 0] * (gx - 1)).astype(np.int32), 0, gx - 1)
    iy = np.clip((p[:, 1] * (gy - 1)).astype(np.int32), 0, gy - 1)
    iz = np.clip((p[:, 2] * (gz - 1)).astype(np.int32), 0, gz - 1)

    Xsum = np.zeros((IN_CH, gx, gy, gz), dtype=np.float32)
    Ysum = np.zeros((OUT_CH, gx, gy, gz), dtype=np.float32)
    Cnt = np.zeros((1, gx, gy, gz), dtype=np.float32)

    for i in range(pos.shape[0]):
        a, b, c = ix[i], iy[i], iz[i]
        Xsum[:16, a, b, c] += fx[i]
        Xsum[16, a, b, c] += 1.0  # occupancy
        Ysum[:, a, b, c] += y[i]
        Cnt[0, a, b, c] += 1.0

    mask = (Cnt > 0).astype(np.float32)
    valid = Cnt[0] > 0
    Xg = Xsum.copy()
    Yg = Ysum.copy()
    Xg[:, valid] /= Cnt[0, valid]
    Yg[:, valid] /= Cnt[0, valid]

    return Xg, Yg, mask


def trilinear_grid_to_points(grid, points, grid_spec):
    """
    grid: (C,gx,gy,gz)
    points: (N,3)
    returns: (N,C)
    """
    C, gx, gy, gz = grid.shape
    xyz_min, xyz_max = grid_spec["xyz_min"], grid_spec["xyz_max"]
    p = _normalize_pos(points, xyz_min, xyz_max)

    x = np.clip(p[:, 0] * (gx - 1), 0, gx - 1)
    y = np.clip(p[:, 1] * (gy - 1), 0, gy - 1)
    z = np.clip(p[:, 2] * (gz - 1), 0, gz - 1)

    x0 = np.floor(x).astype(np.int32); x1 = np.clip(x0 + 1, 0, gx - 1)
    y0 = np.floor(y).astype(np.int32); y1 = np.clip(y0 + 1, 0, gy - 1)
    z0 = np.floor(z).astype(np.int32); z1 = np.clip(z0 + 1, 0, gz - 1)

    xd = (x - x0).astype(np.float32)
    yd = (y - y0).astype(np.float32)
    zd = (z - z0).astype(np.float32)

    out = np.zeros((points.shape[0], C), dtype=np.float32)
    for i in range(points.shape[0]):
        c000 = grid[:, x0[i], y0[i], z0[i]]
        c001 = grid[:, x0[i], y0[i], z1[i]]
        c010 = grid[:, x0[i], y1[i], z0[i]]
        c011 = grid[:, x0[i], y1[i], z1[i]]
        c100 = grid[:, x1[i], y0[i], z0[i]]
        c101 = grid[:, x1[i], y0[i], z1[i]]
        c110 = grid[:, x1[i], y1[i], z0[i]]
        c111 = grid[:, x1[i], y1[i], z1[i]]

        c00 = c000 * (1 - xd[i]) + c100 * xd[i]
        c01 = c001 * (1 - xd[i]) + c101 * xd[i]
        c10 = c010 * (1 - xd[i]) + c110 * xd[i]
        c11 = c011 * (1 - xd[i]) + c111 * xd[i]

        c0 = c00 * (1 - yd[i]) + c10 * yd[i]
        c1 = c01 * (1 - yd[i]) + c11 * yd[i]

        out[i] = c0 * (1 - zd[i]) + c1 * zd[i]
    return out


def normalize_grid_inputs(X_train, X_all):
    """
    normalize channels 0..15; keep occupancy channel 16 untouched
    X shape: (B, C, gx, gy, gz)
    """
    mu = X_train[:, :16].mean(axis=(0, 2, 3, 4), keepdims=True).astype(np.float32)
    std = (X_train[:, :16].std(axis=(0, 2, 3, 4), keepdims=True) + 1e-8).astype(np.float32)

    Xn = X_all.copy()
    Xn[:, :16] = (Xn[:, :16] - mu) / std
    return Xn.astype(np.float32), mu.squeeze(), std.squeeze()


def normalize_grid_targets(Y_train, Y_all):
    mu = Y_train.mean(axis=(0, 2, 3, 4), keepdims=True).astype(np.float32)
    std = (Y_train.std(axis=(0, 2, 3, 4), keepdims=True) + 1e-8).astype(np.float32)
    Yn = (Y_all - mu) / std
    return Yn.astype(np.float32), mu.squeeze(), std.squeeze()


def save_artifacts(
    trained,
    metrics,
    y_true_points,
    y_pred_points,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    params_np = jax.tree_util.tree_map(np.asarray, trained["params"])
    with open(MODEL_DIR / "fno_params.pkl", "wb") as f:
        pickle.dump(params_np, f)

    np.save(MODEL_DIR / "x_mu.npy", trained["x_mu"])
    np.save(MODEL_DIR / "x_std.npy", trained["x_std"])
    np.save(MODEL_DIR / "y_mu.npy", trained["y_mu"])
    np.save(MODEL_DIR / "y_std.npy", trained["y_std"])

    with open(MODEL_DIR / "grid_spec.json", "w") as f:
        json.dump(
            {
                "xyz_min": trained["grid_spec"]["xyz_min"].tolist(),
                "xyz_max": trained["grid_spec"]["xyz_max"].tolist(),
                "gx": trained["grid_spec"]["gx"],
                "gy": trained["grid_spec"]["gy"],
                "gz": trained["grid_spec"]["gz"],
            },
            f,
            indent=2,
        )

    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(trained["history"], f, indent=2)

    np.save(RESULTS_DIR / "y_true_points.npy", y_true_points)
    np.save(RESULTS_DIR / "y_pred_points.npy", y_pred_points)
    np.save(RESULTS_DIR / "relative_l2_per_sample.npy", metrics["relative_L2_per_sample"])

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(
            {
                "relative_L2_mean": float(metrics["relative_L2_mean"]),
                "relative_L2_std": float(metrics["relative_L2_std"]),
            },
            f,
            indent=2,
        )
"""
visualise.py

Unified visualization pipeline for Warped-IFW results.

Outputs in one run:
1) 2D static contour+scatter panels (GT/Pred/Error), per-field relative L2 in titles
2) 2D static scatter-only panels (GT/Pred/Error), per-field relative L2 in titles
3) 3D wing-surface panel
4) 2D contour animation for chosen samples, all fields (|u|, ux, uy, uz)
5) 2D scatter animation for chosen samples, all fields (|u|, ux, uy, uz)
6) 4x3 Ux comparison summary for chosen samples
7) MATLAB export bundle (.mat if scipy.io available, else .npz)
8) batch report image + CSV

Supports both:
- dense arrays of shape (B, N, 15)
- ragged object arrays where each sample is shape (Ni, 15)

Directory structure:
- THIS = visualise.py
- TRANSOLVER_DIR = THIS.parent
- PROJECT_DIR = TRANSOLVER_DIR.parent
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay

# -----------------------------
# Project paths
# -----------------------------
THIS = Path(__file__).resolve()
TRANSOLVER_DIR = THIS.parent
PROJECT_DIR = TRANSOLVER_DIR.parent

DATA_DIR = PROJECT_DIR / "data" / "warped-ifw"
RESULTS_DIR = TRANSOLVER_DIR / "results"
OUT_DIR = TRANSOLVER_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Split config
# -----------------------------
TEST_FRAC = 0.15
SEED = 42

# -----------------------------
# Plot config
# -----------------------------
FIG_DPI = 140
ANIM_DPI = 130
ANIM_FPS = 2

CMAP_MAIN = "turbo"
CMAP_ERR = "magma"
CMAP_PRESS = "coolwarm"

SECTION_AXIS = "y"
SECTION_QUANTILE = 0.5
SECTION_BAND_FRAC = 0.015
GRID_NX = 320
GRID_NY = 190
GAUSS_SIGMA = 1.0

N_REPORT_SAMPLES = 10
SINGLE_SAMPLE_INDICES = [0, 10, 20, 30]
ANIM_SAMPLE_INDICES = [0, 10, 20, 30]

FIELD_NAMES = ["|u|", "ux", "uy", "uz"]

from load_data import load_raw_sample, load_split  # noqa: E402


# =============================
# Utilities
# =============================
def flat15_to_t53(arr):
    return arr.reshape(*arr.shape[:-1], 5, 3)


def speed_mag(u):
    return np.linalg.norm(u, axis=-1)


def rel_l2_sample(y_true_flat, y_pred_flat):
    num = np.sum((y_pred_flat - y_true_flat) ** 2)
    den = np.sum(y_true_flat ** 2) + 1e-12
    return float(np.sqrt(num / den))


def field_rel_l2(gt_vals, pr_vals):
    """Relative L2 for a single scalar field (e.g. ux for one frame), not the whole 15-channel sample."""
    num = np.sum((pr_vals - gt_vals) ** 2)
    den = np.sum(gt_vals ** 2) + 1e-12
    return float(np.sqrt(num / den))


def frame_fields(u):
    """u: (N,3) velocity for one frame. Returns dict name -> (N,) array, in FIELD_NAMES order."""
    return {
        "|u|": speed_mag(u),
        "ux": u[:, 0],
        "uy": u[:, 1],
        "uz": u[:, 2],
    }


def symmetric_error_limits(values, q=0.99, eps=1e-8):
    vmax = float(np.quantile(np.abs(values), q))
    vmax = max(vmax, eps)
    return -vmax, vmax


def get_sample(y, idx):
    return y[idx]


def load_once():
    y_true = np.load(RESULTS_DIR / "y_true.npy", allow_pickle=True)
    y_pred = np.load(RESULTS_DIR / "y_pred.npy", allow_pickle=True)

    if y_true.dtype == object:
        y_true = [np.asarray(a, dtype=np.float32) for a in y_true.tolist()]
    else:
        y_true = y_true.astype(np.float32)

    if y_pred.dtype == object:
        y_pred = [np.asarray(a, dtype=np.float32) for a in y_pred.tolist()]
    else:
        y_pred = y_pred.astype(np.float32)

    _, test_samples, _, test_files = load_split(
        data_dir=DATA_DIR, test_frac=TEST_FRAC, seed=SEED
    )

    if isinstance(y_true, list) != isinstance(y_pred, list):
        raise ValueError("y_true and y_pred must both be dense arrays or both be ragged object arrays")

    if isinstance(y_true, list):
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            if yt.shape != yp.shape:
                raise ValueError(f"Shape mismatch for sample {i}: y_true={yt.shape}, y_pred={yp.shape}")
            if yt.ndim != 2 or yt.shape[-1] != 15:
                raise ValueError(f"Expected ragged sample shape (N,15), got {yt.shape} for sample {i}")
        B = len(y_true)
    else:
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.ndim != 3 or y_true.shape[-1] != 15:
            raise ValueError(f"Expected prediction shape (B,N,15), got {y_true.shape}")
        B = y_true.shape[0]

    if B > len(test_samples):
        raise ValueError(
            f"Predictions have {B} samples but split only has {len(test_samples)} test samples."
        )

    test_samples = test_samples[:B]
    test_files = test_files[:B]

    x_path = RESULTS_DIR / "x_points.npy"
    idcs_path = RESULTS_DIR / "idcs_airfoil.npy"
    if x_path.exists() and idcs_path.exists():
        x_saved = np.load(x_path, allow_pickle=True)
        idcs_saved = np.load(idcs_path, allow_pickle=True)
        for i in range(len(test_samples)):
            test_samples[i] = dict(test_samples[i])
            test_samples[i]["x"] = np.asarray(x_saved[i], dtype=np.float32)
            test_samples[i]["idcs_airfoil"] = np.asarray(idcs_saved[i], dtype=np.int32)
    else:
        print(
            "WARNING: x_points.npy/idcs_airfoil.npy not found. Falling back to "
            "uncropped split positions; this will error if crop_enabled=True was used."
        )

    raw_samples = [load_raw_sample(fp) for fp in test_files]

    return y_true, y_pred, test_samples, test_files, raw_samples


def choose_section_mask(pos, axis="y", quantile=0.5, band_frac=0.015):
    axis_map = {"x": 0, "y": 1, "z": 2}
    k = axis_map[axis]
    vals = pos[:, k]
    lo, hi = vals.min(), vals.max()
    target = np.quantile(vals, quantile)
    half_band = max((hi - lo) * band_frac, 1e-8)

    mask = np.abs(vals - target) <= half_band
    if mask.sum() < 300:
        half_band *= 2.0
        mask = np.abs(vals - target) <= half_band
    if mask.sum() < 100:
        half_band *= 2.0
        mask = np.abs(vals - target) <= half_band
    return mask, target, half_band


def projected_axes(axis):
    if axis == "x":
        return 1, 2, ("y", "z")
    if axis == "y":
        return 0, 2, ("x", "z")
    if axis == "z":
        return 0, 1, ("x", "y")
    raise ValueError("axis must be x|y|z")


def interpolate_masked(px, py, values, nx=320, ny=190, sigma=1.0):
    xi = np.linspace(px.min(), px.max(), nx)
    yi = np.linspace(py.min(), py.max(), ny)
    XI, YI = np.meshgrid(xi, yi)

    pts = np.column_stack([px, py])

    z_lin = griddata(pts, values, (XI, YI), method="linear")
    z_near = griddata(pts, values, (XI, YI), method="nearest")
    Z = np.where(np.isnan(z_lin), z_near, z_lin)

    tri = Delaunay(pts)
    inside = tri.find_simplex(np.column_stack([XI.ravel(), YI.ravel()])) >= 0
    inside = inside.reshape(XI.shape)

    if sigma and sigma > 0:
        Zs = gaussian_filter(Z, sigma=sigma)
        Z = np.where(inside, Zs, np.nan)
    else:
        Z = np.where(inside, Z, np.nan)

    return XI, YI, Z


def pressure_out_frame(raw_sample, frame_idx):
    p = raw_sample["pressure"].astype(np.float32)
    return p[frame_idx + 5]


def align_sample_to_positions(field_flat, pos, raw_sample):
    if field_flat.shape[0] == pos.shape[0]:
        return field_flat

    n_field = field_flat.shape[0]
    n_pos = pos.shape[0]

    for key in ("fx", "x"):
        if key in raw_sample and raw_sample[key].shape[0] == n_field:
            return field_flat

    if abs(n_field - n_pos) <= 1:
        n = min(n_field, n_pos)
        return field_flat[:n]

    raise ValueError(
        f"Prediction/sample point mismatch: field has {n_field} points but test sample has {n_pos}. "
        "This usually means predictions were saved on cropped/reindexed points but visualisation is "
        "reading uncropped split metadata."
    )


def build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples):
    pos = test_samples[sample_idx]["x"]
    yt_s = np.asarray(get_sample(y_true, sample_idx), dtype=np.float32)
    yp_s = np.asarray(get_sample(y_pred, sample_idx), dtype=np.float32)

    raw = raw_samples[sample_idx]
    yt_s = align_sample_to_positions(yt_s, pos, raw)
    yp_s = align_sample_to_positions(yp_s, pos, raw)

    n = min(pos.shape[0], yt_s.shape[0], yp_s.shape[0])
    pos = pos[:n]
    yt_s = yt_s[:n]
    yp_s = yp_s[:n]

    return pos, yt_s, yp_s


# =============================
# 2D static plots
# =============================
def plot_2d_contour_scatter_panel(sample_idx, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir):
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples)
    yt = flat15_to_t53(yt_s)[:, frame_idx, :]
    yp = flat15_to_t53(yp_s)[:, frame_idx, :]

    gt_fields_full = frame_fields(yt)
    pr_fields_full = frame_fields(yp)

    mask, target, hw = choose_section_mask(
        pos, axis=SECTION_AXIS, quantile=SECTION_QUANTILE, band_frac=SECTION_BAND_FRAC
    )
    i, j, labels = projected_axes(SECTION_AXIS)
    px = pos[mask, i]
    py = pos[mask, j]

    fig, axs = plt.subplots(3, 4, figsize=(20, 13), dpi=FIG_DPI)

    for c, name in enumerate(FIELD_NAMES):
        gt_full = gt_fields_full[name]
        pr_full = pr_fields_full[name]
        rel = field_rel_l2(gt_full, pr_full)

        gt_vals = gt_full[mask]
        pr_vals = pr_full[mask]
        er_vals = gt_vals - pr_vals

        XI, YI, Zg = interpolate_masked(px, py, gt_vals, nx=GRID_NX, ny=GRID_NY, sigma=GAUSS_SIGMA)
        _, _, Zp = interpolate_masked(px, py, pr_vals, nx=GRID_NX, ny=GRID_NY, sigma=GAUSS_SIGMA)
        _, _, Ze = interpolate_masked(px, py, er_vals, nx=GRID_NX, ny=GRID_NY, sigma=GAUSS_SIGMA)

        vmin = np.nanmin([Zg, Zp])
        vmax = np.nanmax([Zg, Zp])
        levels = np.linspace(vmin, vmax, 50)
        emin, emax = symmetric_error_limits(np.nan_to_num(Ze[~np.isnan(Ze)], nan=0.0))
        elevels = np.linspace(emin, emax, 50)

        cf0 = axs[0, c].contourf(XI, YI, Zg, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
        axs[0, c].scatter(px, py, c=gt_vals, s=2, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax, alpha=0.30)
        axs[0, c].set_title(f"GT {name}")
        fig.colorbar(cf0, ax=axs[0, c], shrink=0.82)

        cf1 = axs[1, c].contourf(XI, YI, Zp, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
        axs[1, c].scatter(px, py, c=pr_vals, s=2, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax, alpha=0.30)
        axs[1, c].set_title(f"Pred {name}")
        fig.colorbar(cf1, ax=axs[1, c], shrink=0.82)

        cf2 = axs[2, c].contourf(XI, YI, Ze, levels=elevels, cmap=CMAP_ERR, alpha=0.95, extend="both")
        axs[2, c].scatter(px, py, c=er_vals, s=2, cmap=CMAP_ERR, vmin=emin, vmax=emax, alpha=0.30)
        axs[2, c].set_title(f"Error {name} (GT - Pred) | relL2={rel:.4f}")
        fig.colorbar(cf2, ax=axs[2, c], shrink=0.82)

        for r in [0, 1, 2]:
            axs[r, c].set_xlabel(labels[0])
            axs[r, c].set_ylabel(labels[1])
            axs[r, c].set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Sample {sample_idx} frame {frame_idx} | contour+scatter | "
        f"section @{SECTION_AXIS}={target:.4f}±{hw:.4f}"
    )
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_2d_contour_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_2d_scatter_only_panel(sample_idx, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir):
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples)
    yt = flat15_to_t53(yt_s)[:, frame_idx, :]
    yp = flat15_to_t53(yp_s)[:, frame_idx, :]

    gt_fields_full = frame_fields(yt)
    pr_fields_full = frame_fields(yp)

    mask, target, hw = choose_section_mask(
        pos, axis=SECTION_AXIS, quantile=SECTION_QUANTILE, band_frac=SECTION_BAND_FRAC
    )
    i, j, labels = projected_axes(SECTION_AXIS)
    px = pos[mask, i]
    py = pos[mask, j]

    fig, axs = plt.subplots(3, 4, figsize=(20, 13), dpi=FIG_DPI)

    for c, name in enumerate(FIELD_NAMES):
        gt_full = gt_fields_full[name]
        pr_full = pr_fields_full[name]
        rel = field_rel_l2(gt_full, pr_full)

        gt_vals = gt_full[mask]
        pr_vals = pr_full[mask]
        er_vals = gt_vals - pr_vals

        vmin = min(gt_vals.min(), pr_vals.min())
        vmax = max(gt_vals.max(), pr_vals.max())
        emin, emax = symmetric_error_limits(er_vals)

        sc0 = axs[0, c].scatter(px, py, c=gt_vals, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
        axs[0, c].set_title(f"GT {name}")
        fig.colorbar(sc0, ax=axs[0, c], shrink=0.82)

        sc1 = axs[1, c].scatter(px, py, c=pr_vals, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
        axs[1, c].set_title(f"Pred {name}")
        fig.colorbar(sc1, ax=axs[1, c], shrink=0.82)

        sc2 = axs[2, c].scatter(px, py, c=er_vals, s=4, cmap=CMAP_ERR, vmin=emin, vmax=emax)
        axs[2, c].set_title(f"Error {name} (GT - Pred) | relL2={rel:.4f}")
        fig.colorbar(sc2, ax=axs[2, c], shrink=0.82)

        for r in [0, 1, 2]:
            axs[r, c].set_xlabel(labels[0])
            axs[r, c].set_ylabel(labels[1])
            axs[r, c].set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Sample {sample_idx} frame {frame_idx} | scatter-only | "
        f"section @{SECTION_AXIS}={target:.4f}±{hw:.4f}"
    )
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_2d_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ux_summary_4x3(sample_indices, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir):
    rows = len(sample_indices)
    fig, axs = plt.subplots(rows, 3, figsize=(16, 4.0 * rows), dpi=FIG_DPI)
    if rows == 1:
        axs = axs[None, :]

    for r, sample_idx in enumerate(sample_indices):
        pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples)
        yt = flat15_to_t53(yt_s)[:, frame_idx, :]
        yp = flat15_to_t53(yp_s)[:, frame_idx, :]
        ux_rel = field_rel_l2(yt[:, 0], yp[:, 0])
        mask, target, hw = choose_section_mask(pos, axis=SECTION_AXIS, quantile=SECTION_QUANTILE, band_frac=SECTION_BAND_FRAC)
        i, j, labels = projected_axes(SECTION_AXIS)
        px = pos[mask, i]
        py = pos[mask, j]

        gt_vals = yt[mask, 0]
        pr_vals = yp[mask, 0]
        er_vals = gt_vals - pr_vals
        XI, YI, Zg = interpolate_masked(px, py, gt_vals, GRID_NX, GRID_NY, GAUSS_SIGMA)
        _, _, Zp = interpolate_masked(px, py, pr_vals, GRID_NX, GRID_NY, GAUSS_SIGMA)
        _, _, Ze = interpolate_masked(px, py, er_vals, GRID_NX, GRID_NY, GAUSS_SIGMA)

        vmin = np.nanmin([Zg, Zp])
        vmax = np.nanmax([Zg, Zp])
        levels = np.linspace(vmin, vmax, 50)
        emin, emax = symmetric_error_limits(np.nan_to_num(Ze[~np.isnan(Ze)], nan=0.0))
        elevels = np.linspace(emin, emax, 50)

        c0 = axs[r, 0].contourf(XI, YI, Zg, levels=levels, cmap=CMAP_MAIN, extend="both")
        axs[r, 0].set_title(f"Sample {sample_idx:03d} GT ux")
        fig.colorbar(c0, ax=axs[r, 0], shrink=0.82)

        c1 = axs[r, 1].contourf(XI, YI, Zp, levels=levels, cmap=CMAP_MAIN, extend="both")
        axs[r, 1].set_title(f"Sample {sample_idx:03d} Pred ux")
        fig.colorbar(c1, ax=axs[r, 1], shrink=0.82)

        c2 = axs[r, 2].contourf(XI, YI, Ze, levels=elevels, cmap=CMAP_ERR, extend="both")
        axs[r, 2].set_title(f"Sample {sample_idx:03d} Error ux (GT - Pred) | relL2={ux_rel:.4f}")
        fig.colorbar(c2, ax=axs[r, 2], shrink=0.82)

        for c in range(3):
            axs[r, c].set_xlabel(labels[0])
            axs[r, c].set_ylabel(labels[1])
            axs[r, c].set_aspect("equal", adjustable="box")

    fig.suptitle(f"Ux comparison summary | frame {frame_idx} | section axis {SECTION_AXIS}")
    out_path = out_dir / f"ux_summary_frame{frame_idx}_4x3.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================
# 3D static plots
# =============================
def plot_3d_wing_surface(sample_idx, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir):
    s = test_samples[sample_idx]
    raw = raw_samples[sample_idx]

    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples)
    idcs_air = s["idcs_airfoil"]
    yt = flat15_to_t53(yt_s)[:, frame_idx, :]
    yp = flat15_to_t53(yp_s)[:, frame_idx, :]
    rel = rel_l2_sample(yt_s, yp_s)

    p_full = pressure_out_frame(raw, frame_idx)
    p_surf = p_full[idcs_air] if idcs_air.size > 0 and idcs_air.max() < p_full.shape[0] else np.zeros_like(idcs_air, dtype=np.float32)

    surf_pts = s["x"][idcs_air]
    if surf_pts.shape[0] < 3:
        return None
    tri2 = Triangulation(surf_pts[:, 0], surf_pts[:, 2])

    pmin, pmax = np.min(p_surf), np.max(p_surf)

    fig = plt.figure(figsize=(14, 6), dpi=FIG_DPI)
    ax0 = fig.add_axes([0.02, 0.08, 0.42, 0.85], projection="3d")
    ax1 = fig.add_axes([0.53, 0.08, 0.42, 0.85], projection="3d")
    cax = fig.add_axes([0.465, 0.15, 0.015, 0.70])

    for ax, u, ttl in [(ax0, yt, "GT"), (ax1, yp, "Pred")]:
        surf = ax.plot_trisurf(
            surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2],
            triangles=tri2.triangles,
            cmap=CMAP_PRESS,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            alpha=0.98,
        )
        tri_p = p_surf[tri2.triangles].mean(axis=1)
        surf.set_array(tri_p.astype(np.float32))
        surf.set_clim(pmin, pmax)

        y = pos[:, 1]
        y0 = np.quantile(y, 0.5)
        dy = max((y.max() - y.min()) * 0.015, 1e-6)
        m = np.abs(y - y0) <= dy
        ux = u[m, 0]
        uxmin, uxmax = np.quantile(ux, [0.01, 0.99])

        ax.scatter(
            pos[m, 0], pos[m, 1], pos[m, 2],
            c=ux, s=1.2, cmap=CMAP_MAIN, vmin=uxmin, vmax=uxmax, alpha=0.45
        )

        ax.set_title(f"{ttl}: triangulated wing + section u_x")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-60)

    sm_p = plt.cm.ScalarMappable(cmap=CMAP_PRESS, norm=plt.Normalize(vmin=pmin, vmax=pmax))
    sm_p.set_array([])
    cb = fig.colorbar(sm_p, cax=cax)
    cb.set_label("Pressure (Pa) on wing")

    fig.suptitle(f"Sample {sample_idx} frame {frame_idx} | 3D wing surface | relL2={rel:.4f}")
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_3d_wing_surface.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================
# 2D animations (all fields)
# =============================
def make_contour_animation(sample_idx, y_true, y_pred, test_samples, raw_samples, out_dir):
    out_path = out_dir / f"sample{sample_idx:03d}_2d_contour_animation.mp4"

    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples)
    yt = flat15_to_t53(yt_s)
    yp = flat15_to_t53(yp_s)

    m, target, hw = choose_section_mask(pos, axis=SECTION_AXIS, quantile=SECTION_QUANTILE, band_frac=SECTION_BAND_FRAC)
    i, j, labels = projected_axes(SECTION_AXIS)
    px = pos[m, i]
    py = pos[m, j]

    # cache[t][name] = (XI, YI, Zg, Zp, Ze, rel)
    cache = []
    limits_main = {name: [np.inf, -np.inf] for name in FIELD_NAMES}
    limits_err = {name: [np.inf, -np.inf] for name in FIELD_NAMES}

    for t in range(5):
        gt_full = frame_fields(yt[:, t, :])
        pr_full = frame_fields(yp[:, t, :])
        frame_cache = {}
        for name in FIELD_NAMES:
            rel = field_rel_l2(gt_full[name], pr_full[name])
            gv = gt_full[name][m]
            pv = pr_full[name][m]
            ev = gv - pv

            XI, YI, Zg = interpolate_masked(px, py, gv, GRID_NX, GRID_NY, GAUSS_SIGMA)
            _, _, Zp = interpolate_masked(px, py, pv, GRID_NX, GRID_NY, GAUSS_SIGMA)
            _, _, Ze = interpolate_masked(px, py, ev, GRID_NX, GRID_NY, GAUSS_SIGMA)

            limits_main[name][0] = min(limits_main[name][0], np.nanmin([Zg, Zp]))
            limits_main[name][1] = max(limits_main[name][1], np.nanmax([Zg, Zp]))
            finite_e = Ze[~np.isnan(Ze)]
            if finite_e.size:
                limits_err[name][0] = min(limits_err[name][0], np.nanmin(finite_e))
                limits_err[name][1] = max(limits_err[name][1], np.nanmax(finite_e))

            frame_cache[name] = (XI, YI, Zg, Zp, Ze, rel)
        cache.append(frame_cache)

    err_sym = {}
    for name in FIELD_NAMES:
        lo, hi = limits_err[name]
        err_sym[name] = symmetric_error_limits(np.array([lo, hi], dtype=np.float32))

    fig, axs = plt.subplots(3, 4, figsize=(20, 13), dpi=ANIM_DPI)
    title = fig.suptitle("")

    def draw(t):
        for r in range(3):
            for c in range(4):
                axs[r, c].clear()

        frame_cache = cache[t]
        for c, name in enumerate(FIELD_NAMES):
            XI, YI, Zg, Zp, Ze, rel = frame_cache[name]
            vmin, vmax = limits_main[name]
            levels = np.linspace(vmin, vmax, 50)
            emin, emax = err_sym[name]
            elevels = np.linspace(emin, emax, 50)

            axs[0, c].contourf(XI, YI, Zg, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
            axs[1, c].contourf(XI, YI, Zp, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
            axs[2, c].contourf(XI, YI, Ze, levels=elevels, cmap=CMAP_ERR, alpha=0.95, extend="both")

            axs[0, c].set_title(f"GT {name}")
            axs[1, c].set_title(f"Pred {name}")
            axs[2, c].set_title(f"Error {name} (GT - Pred) | relL2={rel:.4f}")

            for r in [0, 1, 2]:
                axs[r, c].set_xlabel(labels[0])
                axs[r, c].set_ylabel(labels[1])
                axs[r, c].set_aspect("equal", adjustable="box")

        title.set_text(
            f"Sample {sample_idx} | frame {t} (chronological) | contour | "
            f"section @{SECTION_AXIS}={target:.4f}±{hw:.4f}"
        )
        return []

    anim = FuncAnimation(fig, draw, frames=5, interval=750, blit=False, repeat=True)
    try:
        anim.save(out_path, writer=FFMpegWriter(fps=ANIM_FPS))
    except Exception:
        out_path = out_path.with_suffix(".gif")
        anim.save(out_path, writer=PillowWriter(fps=ANIM_FPS))
    plt.close(fig)
    return out_path


def make_scatter_animation(sample_idx, y_true, y_pred, test_samples, raw_samples, out_dir):
    out_path = out_dir / f"sample{sample_idx:03d}_2d_scatter_animation.mp4"

    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples)
    yt = flat15_to_t53(yt_s)
    yp = flat15_to_t53(yp_s)

    m, target, hw = choose_section_mask(pos, axis=SECTION_AXIS, quantile=SECTION_QUANTILE, band_frac=SECTION_BAND_FRAC)
    i, j, labels = projected_axes(SECTION_AXIS)
    px = pos[m, i]
    py = pos[m, j]

    # cache[t][name] = (gv, pv, ev, rel)
    cache = []
    limits_main = {name: [np.inf, -np.inf] for name in FIELD_NAMES}
    limits_err = {name: [np.inf, -np.inf] for name in FIELD_NAMES}

    for t in range(5):
        gt_full = frame_fields(yt[:, t, :])
        pr_full = frame_fields(yp[:, t, :])
        frame_cache = {}
        for name in FIELD_NAMES:
            rel = field_rel_l2(gt_full[name], pr_full[name])
            gv = gt_full[name][m]
            pv = pr_full[name][m]
            ev = gv - pv

            limits_main[name][0] = min(limits_main[name][0], gv.min(), pv.min())
            limits_main[name][1] = max(limits_main[name][1], gv.max(), pv.max())
            limits_err[name][0] = min(limits_err[name][0], ev.min())
            limits_err[name][1] = max(limits_err[name][1], ev.max())

            frame_cache[name] = (gv, pv, ev, rel)
        cache.append(frame_cache)

    err_sym = {}
    for name in FIELD_NAMES:
        lo, hi = limits_err[name]
        err_sym[name] = symmetric_error_limits(np.array([lo, hi], dtype=np.float32))

    fig, axs = plt.subplots(3, 4, figsize=(20, 13), dpi=ANIM_DPI)
    title = fig.suptitle("")

    def draw(t):
        for r in range(3):
            for c in range(4):
                axs[r, c].clear()

        frame_cache = cache[t]
        for c, name in enumerate(FIELD_NAMES):
            gv, pv, ev, rel = frame_cache[name]
            vmin, vmax = limits_main[name]
            emin, emax = err_sym[name]

            axs[0, c].scatter(px, py, c=gv, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
            axs[1, c].scatter(px, py, c=pv, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
            axs[2, c].scatter(px, py, c=ev, s=4, cmap=CMAP_ERR, vmin=emin, vmax=emax)

            axs[0, c].set_title(f"GT {name}")
            axs[1, c].set_title(f"Pred {name}")
            axs[2, c].set_title(f"Error {name} (GT - Pred) | relL2={rel:.4f}")

            for r in [0, 1, 2]:
                axs[r, c].set_xlabel(labels[0])
                axs[r, c].set_ylabel(labels[1])
                axs[r, c].set_aspect("equal", adjustable="box")

        title.set_text(
            f"Sample {sample_idx} | frame {t} (chronological) | scatter | "
            f"section @{SECTION_AXIS}={target:.4f}±{hw:.4f}"
        )
        return []

    anim = FuncAnimation(fig, draw, frames=5, interval=750, blit=False, repeat=True)
    try:
        anim.save(out_path, writer=FFMpegWriter(fps=ANIM_FPS))
    except Exception:
        out_path = out_path.with_suffix(".gif")
        anim.save(out_path, writer=PillowWriter(fps=ANIM_FPS))
    plt.close(fig)
    return out_path


# =============================
# MATLAB export
# =============================
def export_matlab_bundle(sample_indices, y_true, y_pred, test_samples, test_files, raw_samples, out_dir):
    bundle = {
        "sample_indices": np.asarray(sample_indices, dtype=np.int32),
        "section_axis": SECTION_AXIS,
        "section_quantile": SECTION_QUANTILE,
        "section_band_frac": SECTION_BAND_FRAC,
    }

    for sidx in sample_indices:
        pos, yt_s, yp_s = build_aligned_sample(sidx, y_true, y_pred, test_samples, raw_samples)
        sample = test_samples[sidx]
        raw = raw_samples[sidx]
        key = f"sample_{sidx:03d}"
        bundle[f"{key}_x"] = pos.astype(np.float32)
        bundle[f"{key}_y_true"] = yt_s.astype(np.float32)
        bundle[f"{key}_y_pred"] = yp_s.astype(np.float32)
        bundle[f"{key}_idcs_airfoil"] = sample["idcs_airfoil"].astype(np.int32)
        bundle[f"{key}_file"] = test_files[sidx].name
        if "pressure" in raw:
            bundle[f"{key}_pressure"] = raw["pressure"].astype(np.float32)

    try:
        from scipy.io import savemat
        out_path = out_dir / "matlab_export_bundle.mat"
        savemat(out_path, bundle, do_compression=True)
        return out_path
    except Exception:
        out_path = out_dir / "matlab_export_bundle.npz"
        np.savez_compressed(out_path, **bundle)
        return out_path


# =============================
# Batch report
# =============================
def batch_report(y_true, y_pred, test_files, out_dir):
    B = len(y_true) if isinstance(y_true, list) else y_true.shape[0]
    rels = np.array([rel_l2_sample(get_sample(y_true, i), get_sample(y_pred, i)) for i in range(B)])
    order = np.argsort(rels)

    n = min(N_REPORT_SAMPLES, B)
    chosen = np.concatenate([
        order[: max(1, n // 3)],
        order[B // 2: B // 2 + max(1, n // 3)],
        order[-max(1, n - 2 * (n // 3)):]
    ])[:n]

    fig, axs = plt.subplots(n, 2, figsize=(12, 3 * n), dpi=FIG_DPI)
    if n == 1:
        axs = np.array([axs])

    for r, i in enumerate(chosen):
        gt = flat15_to_t53(get_sample(y_true, i))
        pr = flat15_to_t53(get_sample(y_pred, i))

        gt_m = speed_mag(gt).mean(axis=0)
        pr_m = speed_mag(pr).mean(axis=0)
        er_m = speed_mag(pr - gt).mean(axis=0)

        axs[r, 0].plot(gt_m, marker="o", label="GT mean|u|")
        axs[r, 0].plot(pr_m, marker="s", label="Pred mean|u|")
        axs[r, 0].set_title(f"sample {i} ({test_files[i].name})")
        axs[r, 0].set_xlabel("frame")
        axs[r, 0].set_ylabel("mean |u|")
        axs[r, 0].legend()

        axs[r, 1].plot(er_m, marker="o", color="tab:red")
        axs[r, 1].set_title(f"mean |Δu| per frame | relL2={rels[i]:.4f}")
        axs[r, 1].set_xlabel("frame")
        axs[r, 1].set_ylabel("mean error")

    fig.tight_layout()
    png = out_dir / "batch_report.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    csv = out_dir / "batch_report_metrics.csv"
    with open(csv, "w") as f:
        f.write("sample_idx,file,rel_l2\n")
        for i in range(B):
            f.write(f"{i},{test_files[i].name},{rels[i]:.8f}\n")

    return png, csv


# =============================
# Main
# =============================
def main():
    print("NEW SCRIPT")
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_idx", type=int, default=0, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    single_dir = OUT_DIR / "single"
    anim_dir = OUT_DIR / "anim"
    report_dir = OUT_DIR / "report"
    matlab_dir = OUT_DIR / "matlab"
    single_dir.mkdir(parents=True, exist_ok=True)
    anim_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    matlab_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions + samples once...")
    y_true, y_pred, test_samples, test_files, raw_samples = load_once()

    chosen_samples = [i for i in SINGLE_SAMPLE_INDICES if i < len(test_samples)]
    if not chosen_samples:
        raise ValueError(
            f"None of SINGLE_SAMPLE_INDICES={SINGLE_SAMPLE_INDICES} are valid for "
            f"{len(test_samples)} test samples."
        )

    print("Generating static plots...")
    for sidx in chosen_samples:
        sample_dir = single_dir / f"{sidx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        plot_2d_contour_scatter_panel(sidx, args.frame_idx, y_true, y_pred, test_samples, raw_samples, sample_dir)
        plot_2d_scatter_only_panel(sidx, args.frame_idx, y_true, y_pred, test_samples, raw_samples, sample_dir)
        plot_3d_wing_surface(sidx, args.frame_idx, y_true, y_pred, test_samples, raw_samples, sample_dir)

    ux_summary_path = plot_ux_summary_4x3(chosen_samples, args.frame_idx, y_true, y_pred, test_samples, raw_samples, single_dir)

    print("Generating animations...")
    for sidx in [i for i in ANIM_SAMPLE_INDICES if i < len(test_samples)]:
        sample_anim_dir = anim_dir / f"{sidx:03d}"
        sample_anim_dir.mkdir(parents=True, exist_ok=True)
        make_contour_animation(sidx, y_true, y_pred, test_samples, raw_samples, sample_anim_dir)
        make_scatter_animation(sidx, y_true, y_pred, test_samples, raw_samples, sample_anim_dir)

    print("Exporting MATLAB bundle...")
    matlab_path = export_matlab_bundle(chosen_samples, y_true, y_pred, test_samples, test_files, raw_samples, matlab_dir)

    print("Generating batch report...")
    preport, pcsv = batch_report(y_true, y_pred, test_files, report_dir)

    print("\nSaved outputs:")
    print(f"- static sample folders under: {single_dir}")
    print(f"- {ux_summary_path}")
    print(f"- animations under: {anim_dir}")
    print(f"- {matlab_path}")
    print(f"- {preport}")
    print(f"- {pcsv}")


if __name__ == "__main__":
    main()
"""
visualise.py

Unified visualization pipeline for Warped-IFW FNO results.

Outputs in one run:
1) 2D static contour+scatter panel (GT/Pred for |u|, ux, uy, uz)
2) 2D static scatter-only panel
3) 3D scatter comparison panel
4) 3D wing-surface panel (triangulated surface, less "bees")
5) 3D CFD-style panel: pressure on wing + y-plane ux
6) 2D contour animation (chronological frames 0..4)
7) 2D scatter animation (chronological frames 0..4)
8) batch report image + CSV

Expected files:
- PROJECT/FNO/results/y_true_points.npy   shape (B, N, 15)
- PROJECT/FNO/results/y_pred_points.npy   shape (B, N, 15)
- PROJECT/data/warped-ifw/*.npz

Requires:
- load_data.py at PROJECT root
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay

# -----------------------------
# Project paths (your structure)
# -----------------------------
THIS = Path(__file__).resolve()
VIS_DIR = THIS.parent
FNO_DIR = VIS_DIR.parent
PROJECT_DIR = FNO_DIR.parent

DATA_DIR = PROJECT_DIR / "data" / "warped-ifw"
RESULTS_DIR = FNO_DIR / "results"
OUT_DIR = VIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Split config (must match train split)
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

SECTION_AXIS = "y"         # "x" | "y" | "z"
SECTION_QUANTILE = 0.5
SECTION_BAND_FRAC = 0.015
GRID_NX = 320
GRID_NY = 190
GAUSS_SIGMA = 1.0

N_REPORT_SAMPLES = 10

# -----------------------------
# Loader imports
# -----------------------------
from load_data import load_split, load_raw_sample  # noqa: E402


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


def load_once():
    y_true = np.load(RESULTS_DIR / "y_true.npy").astype(np.float32)
    y_pred = np.load(RESULTS_DIR / "y_pred.npy").astype(np.float32)

    _, test_samples, _, test_files = load_split(
        data_dir=DATA_DIR, test_frac=TEST_FRAC, seed=SEED
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.ndim != 3 or y_true.shape[-1] != 15:
        raise ValueError(f"Expected prediction shape (B,N,15), got {y_true.shape}")

    B = y_true.shape[0]
    if B > len(test_samples):
        raise ValueError(
            f"Predictions have {B} samples but split only has {len(test_samples)} test samples."
        )

    # IMPORTANT: support tiny runs by aligning split lists to prediction batch size
    test_samples = test_samples[:B]
    test_files = test_files[:B]

    # Raw samples for pressure overlay (same order as test_files)
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
    """
    Robust 2D interpolation for contouring:
      linear -> nearest fill -> convex-hull masking -> optional smoothing.
    """
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
    # confirmed mapping: velocity_out[k] <-> pressure[k+5]
    p = raw_sample["pressure"].astype(np.float32)  # (10,N)
    return p[frame_idx + 5]


# =============================
# 2D static plots
# =============================
def plot_2d_contour_scatter_panel(sample_idx, frame_idx, y_true, y_pred, test_samples, out_dir):
    pos = test_samples[sample_idx]["x"]
    yt = flat15_to_t53(y_true[sample_idx])[:, frame_idx, :]
    yp = flat15_to_t53(y_pred[sample_idx])[:, frame_idx, :]
    rel = rel_l2_sample(y_true[sample_idx], y_pred[sample_idx])

    mask, target, hw = choose_section_mask(
        pos, axis=SECTION_AXIS, quantile=SECTION_QUANTILE, band_frac=SECTION_BAND_FRAC
    )
    i, j, labels = projected_axes(SECTION_AXIS)
    px = pos[mask, i]
    py = pos[mask, j]

    fields = [
        ("|u|", speed_mag(yt[mask]), speed_mag(yp[mask])),
        ("ux", yt[mask, 0], yp[mask, 0]),
        ("uy", yt[mask, 1], yp[mask, 1]),
        ("uz", yt[mask, 2], yp[mask, 2]),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(20, 9), dpi=FIG_DPI)

    for c, (name, gt_vals, pr_vals) in enumerate(fields):
        XI, YI, Zg = interpolate_masked(px, py, gt_vals, nx=GRID_NX, ny=GRID_NY, sigma=GAUSS_SIGMA)
        _, _, Zp = interpolate_masked(px, py, pr_vals, nx=GRID_NX, ny=GRID_NY, sigma=GAUSS_SIGMA)

        vmin = np.nanmin([Zg, Zp])
        vmax = np.nanmax([Zg, Zp])
        levels = np.linspace(vmin, vmax, 50)

        # GT
        cf0 = axs[0, c].contourf(XI, YI, Zg, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
        axs[0, c].scatter(px, py, c=gt_vals, s=2, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax, alpha=0.35)
        axs[0, c].set_title(f"GT {name}")
        fig.colorbar(cf0, ax=axs[0, c], shrink=0.82)

        # Pred
        cf1 = axs[1, c].contourf(XI, YI, Zp, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
        axs[1, c].scatter(px, py, c=pr_vals, s=2, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax, alpha=0.35)
        axs[1, c].set_title(f"Pred {name}")
        fig.colorbar(cf1, ax=axs[1, c], shrink=0.82)

        for r in [0, 1]:
            axs[r, c].set_xlabel(labels[0])
            axs[r, c].set_ylabel(labels[1])
            axs[r, c].set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Sample {sample_idx} frame {frame_idx} | contour+scatter | "
        f"section @{SECTION_AXIS}={target:.4f}±{hw:.4f} | relL2={rel:.4f}"
    )
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_2d_contour_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_2d_scatter_only_panel(sample_idx, frame_idx, y_true, y_pred, test_samples, out_dir):
    pos = test_samples[sample_idx]["x"]
    yt = flat15_to_t53(y_true[sample_idx])[:, frame_idx, :]
    yp = flat15_to_t53(y_pred[sample_idx])[:, frame_idx, :]
    rel = rel_l2_sample(y_true[sample_idx], y_pred[sample_idx])

    mask, target, hw = choose_section_mask(
        pos, axis=SECTION_AXIS, quantile=SECTION_QUANTILE, band_frac=SECTION_BAND_FRAC
    )
    i, j, labels = projected_axes(SECTION_AXIS)
    px = pos[mask, i]
    py = pos[mask, j]

    fields = [
        ("|u|", speed_mag(yt[mask]), speed_mag(yp[mask])),
        ("ux", yt[mask, 0], yp[mask, 0]),
        ("uy", yt[mask, 1], yp[mask, 1]),
        ("uz", yt[mask, 2], yp[mask, 2]),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(20, 9), dpi=FIG_DPI)

    for c, (name, gt_vals, pr_vals) in enumerate(fields):
        vmin = min(gt_vals.min(), pr_vals.min())
        vmax = max(gt_vals.max(), pr_vals.max())

        sc0 = axs[0, c].scatter(px, py, c=gt_vals, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
        sc1 = axs[1, c].scatter(px, py, c=pr_vals, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
        axs[0, c].set_title(f"GT {name}")
        axs[1, c].set_title(f"Pred {name}")

        fig.colorbar(sc0, ax=axs[0, c], shrink=0.82)
        fig.colorbar(sc1, ax=axs[1, c], shrink=0.82)

        for r in [0, 1]:
            axs[r, c].set_xlabel(labels[0])
            axs[r, c].set_ylabel(labels[1])
            axs[r, c].set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Sample {sample_idx} frame {frame_idx} | scatter-only | "
        f"section @{SECTION_AXIS}={target:.4f}±{hw:.4f} | relL2={rel:.4f}"
    )
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_2d_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================
# 3D static plots
# =============================
def plot_3d_scatter_panel(sample_idx, frame_idx, y_true, y_pred, test_samples, out_dir):
    pos = test_samples[sample_idx]["x"]
    yt = flat15_to_t53(y_true[sample_idx])[:, frame_idx, :]
    yp = flat15_to_t53(y_pred[sample_idx])[:, frame_idx, :]
    rel = rel_l2_sample(y_true[sample_idx], y_pred[sample_idx])

    mt = speed_mag(yt)
    mp = speed_mag(yp)
    me = speed_mag(yp - yt)

    vmin = min(mt.min(), mp.min())
    vmax = max(mt.max(), mp.max())
    emin, emax = np.quantile(me, [0.01, 0.99])

    fig = plt.figure(figsize=(16, 5), dpi=FIG_DPI)
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")

    sc0 = ax0.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=mt, s=1, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=mp, s=1, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=me, s=1, cmap=CMAP_ERR, vmin=emin, vmax=emax)

    ax0.set_title("GT |u|")
    ax1.set_title("Pred |u|")
    ax2.set_title("|Δu|")
    for ax in [ax0, ax1, ax2]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-60)

    fig.colorbar(sc0, ax=ax0, shrink=0.7)
    fig.colorbar(sc1, ax=ax1, shrink=0.7)
    fig.colorbar(sc2, ax=ax2, shrink=0.7)

    fig.suptitle(f"Sample {sample_idx} frame {frame_idx} | 3D scatter | relL2={rel:.4f}")
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_3d_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_3d_wing_surface(sample_idx, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir):
    """
    Cleaner geometry view:
      - triangulated wing surface mesh (from idcs_airfoil points)
      - colored by pressure
      - plus light velocity glyph/scatter overlay near section plane
    """
    s = test_samples[sample_idx]
    raw = raw_samples[sample_idx]

    pos = s["x"]
    idcs_air = s["idcs_airfoil"]
    yt = flat15_to_t53(y_true[sample_idx])[:, frame_idx, :]
    yp = flat15_to_t53(y_pred[sample_idx])[:, frame_idx, :]
    rel = rel_l2_sample(y_true[sample_idx], y_pred[sample_idx])

    p_full = pressure_out_frame(raw, frame_idx)
    p_surf = p_full[idcs_air] if idcs_air.max() < p_full.shape[0] else np.zeros_like(idcs_air, dtype=np.float32)

    surf_pts = pos[idcs_air]  # (Ns,3)
    # Triangulate in x-z (span is thin), then use y from points
    tri2 = Triangulation(surf_pts[:, 0], surf_pts[:, 2])

    pmin, pmax = np.min(p_surf), np.max(p_surf)

    fig = plt.figure(figsize=(14, 6), dpi=FIG_DPI)
    # Explicit rects instead of add_subplot: mplot3d axes report unreliable
    # bounding boxes to fig.colorbar(ax=[...]), which is what was causing the
    # colorbar to render on top of the right (Pred) panel instead of between
    # the two panels. Carving out the rects (and the colorbar's own axes)
    # by hand avoids that entirely.
    ax0 = fig.add_axes([0.02, 0.08, 0.42, 0.85], projection="3d")
    ax1 = fig.add_axes([0.53, 0.08, 0.42, 0.85], projection="3d")
    cax = fig.add_axes([0.465, 0.15, 0.015, 0.70])

    # GT and Pred panels with same wing pressure + different plane ux overlays
    for ax, u, ttl in [(ax0, yt, "GT"), (ax1, yp, "Pred")]:
        # Tri-surface wing (pressure coloring)
        surf = ax.plot_trisurf(
            surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2],
            triangles=tri2.triangles,
            cmap=CMAP_PRESS,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            alpha=0.98,
        )
        # color by triangle-average pressure
        tri_p = p_surf[tri2.triangles].mean(axis=1)
        surf.set_array(tri_p.astype(np.float32))
        surf.set_clim(pmin, pmax)

        # overlay section points colored by ux (small)
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


def plot_3d_cfd_style(sample_idx, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir):
    """
    CFD-style:
      wing pressure + y-plane ux (GT and Pred side-by-side)
    """
    s = test_samples[sample_idx]
    raw = raw_samples[sample_idx]

    pos = s["x"]
    idcs_air = s["idcs_airfoil"]
    yt = flat15_to_t53(y_true[sample_idx])[:, frame_idx, :]
    yp = flat15_to_t53(y_pred[sample_idx])[:, frame_idx, :]
    rel = rel_l2_sample(y_true[sample_idx], y_pred[sample_idx])

    p_full = pressure_out_frame(raw, frame_idx)
    p_surf = p_full[idcs_air] if idcs_air.max() < p_full.shape[0] else np.zeros_like(idcs_air, dtype=np.float32)
    pmin, pmax = p_surf.min(), p_surf.max()

    # y-plane
    y = pos[:, 1]
    y0 = np.quantile(y, 0.5)
    dy = max((y.max() - y.min()) * 0.015, 1e-6)
    m = np.abs(y - y0) <= dy
    xz = np.column_stack([pos[m, 0], pos[m, 2]])
    tri = Delaunay(xz)

    ux_gt = yt[m, 0]
    ux_pr = yp[m, 0]
    ux_min = min(np.min(ux_gt), np.min(ux_pr))
    ux_max = max(np.max(ux_gt), np.max(ux_pr))

    fig = plt.figure(figsize=(16, 6), dpi=FIG_DPI)
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    axR = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, ux, ttl in [(axL, ux_gt, "GT"), (axR, ux_pr, "Pred")]:
        # wing pressure as points
        ax.scatter(
            pos[idcs_air, 0], pos[idcs_air, 1], pos[idcs_air, 2],
            c=p_surf, s=2, cmap=CMAP_PRESS, vmin=pmin, vmax=pmax, alpha=0.95
        )

        # plane trisurf colored by ux (compat-safe)
        X = xz[:, 0]
        Z = xz[:, 1]
        Y = np.full_like(X, y0)
        tri_vals = ux[tri.simplices].mean(axis=1)

        surf = ax.plot_trisurf(
            X, Y, Z,
            triangles=tri.simplices,
            cmap=CMAP_MAIN,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            alpha=0.95,
        )
        surf.set_array(tri_vals.astype(np.float32))
        surf.set_clim(ux_min, ux_max)

        # scatter overlay on plane for point-level feel
        ax.scatter(
            pos[m, 0], pos[m, 1], pos[m, 2],
            c=ux, s=1.0, cmap=CMAP_MAIN, vmin=ux_min, vmax=ux_max, alpha=0.35
        )

        ax.set_title(f"{ttl}: wing pressure + plane u_x")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-60)

    # Pressure colorbar: anchored to the LEFT panel only, drawn on its left side,
    # so it sits on the outer-left edge of the figure instead of the centre gap.
    sm_p = plt.cm.ScalarMappable(cmap=CMAP_PRESS, norm=plt.Normalize(vmin=pmin, vmax=pmax))
    sm_p.set_array([])
    cb1 = fig.colorbar(sm_p, ax=axL, location="left", fraction=0.04, pad=0.08, shrink=0.75)
    cb1.set_label("Pressure (Pa)")

    # Velocity colorbar: anchored to the RIGHT panel only, default right side,
    # so it sits on the outer-right edge of the figure.
    sm_u = plt.cm.ScalarMappable(cmap=CMAP_MAIN, norm=plt.Normalize(vmin=ux_min, vmax=ux_max))
    sm_u.set_array([])
    cb2 = fig.colorbar(sm_u, ax=axR, location="right", fraction=0.04, pad=0.08, shrink=0.75)
    cb2.set_label("u_x (m/s)")

    fig.suptitle(f"Sample {sample_idx} frame {frame_idx} | CFD-style 3D | relL2={rel:.4f}")
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_3d_cfd_style.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================
# 2D animations
# =============================
def make_contour_animation(sample_idx, y_true, y_pred, test_samples, out_dir):
    out_path = out_dir / f"sample{sample_idx:03d}_2d_contour_animation.mp4"

    s = test_samples[sample_idx]
    pos = s["x"]
    yt = flat15_to_t53(y_true[sample_idx])
    yp = flat15_to_t53(y_pred[sample_idx])

    rel = rel_l2_sample(y_true[sample_idx], y_pred[sample_idx])

    m, target, hw = choose_section_mask(pos, axis=SECTION_AXIS, quantile=SECTION_QUANTILE, band_frac=SECTION_BAND_FRAC)
    i, j, labels = projected_axes(SECTION_AXIS)
    px = pos[m, i]
    py = pos[m, j]

    vars_order = ["|u|", "ux", "uy", "uz"]

    cache = []
    limits = {k: [np.inf, -np.inf] for k in vars_order}
    for t in range(5):
        gt = yt[:, t, :]
        pr = yp[:, t, :]
        fmap = {
            "|u|": (speed_mag(gt[m]), speed_mag(pr[m])),
            "ux": (gt[m, 0], pr[m, 0]),
            "uy": (gt[m, 1], pr[m, 1]),
            "uz": (gt[m, 2], pr[m, 2]),
        }
        fd = {}
        for vn, (gv, pv) in fmap.items():
            XI, YI, Zg = interpolate_masked(px, py, gv, GRID_NX, GRID_NY, GAUSS_SIGMA)
            _, _, Zp = interpolate_masked(px, py, pv, GRID_NX, GRID_NY, GAUSS_SIGMA)
            fd[vn] = (XI, YI, Zg, Zp)
            limits[vn][0] = min(limits[vn][0], np.nanmin([Zg, Zp]))
            limits[vn][1] = max(limits[vn][1], np.nanmax([Zg, Zp]))
        cache.append(fd)

    fig, axs = plt.subplots(2, 4, figsize=(20, 9), dpi=ANIM_DPI)
    title = fig.suptitle("")

    def draw(t):
        for r in range(2):
            for c in range(4):
                axs[r, c].clear()

        fd = cache[t]
        for c, vn in enumerate(vars_order):
            XI, YI, Zg, Zp = fd[vn]
            vmin, vmax = limits[vn]
            levels = np.linspace(vmin, vmax, 50)

            axs[0, c].contourf(XI, YI, Zg, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
            axs[1, c].contourf(XI, YI, Zp, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
            axs[0, c].set_title(f"GT {vn}")
            axs[1, c].set_title(f"Pred {vn}")

            for r in [0, 1]:
                axs[r, c].set_xlabel(labels[0])
                axs[r, c].set_ylabel(labels[1])
                axs[r, c].set_aspect("equal", adjustable="box")

        title.set_text(
            f"Sample {sample_idx} | frame {t} (chronological) | contour | "
            f"section @{SECTION_AXIS}={target:.4f}±{hw:.4f} | relL2={rel:.4f}"
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


def make_scatter_animation(sample_idx, y_true, y_pred, test_samples, out_dir):
    out_path = out_dir / f"sample{sample_idx:03d}_2d_scatter_animation.mp4"

    s = test_samples[sample_idx]
    pos = s["x"]
    yt = flat15_to_t53(y_true[sample_idx])
    yp = flat15_to_t53(y_pred[sample_idx])

    rel = rel_l2_sample(y_true[sample_idx], y_pred[sample_idx])

    m, target, hw = choose_section_mask(pos, axis=SECTION_AXIS, quantile=SECTION_QUANTILE, band_frac=SECTION_BAND_FRAC)
    i, j, labels = projected_axes(SECTION_AXIS)
    px = pos[m, i]
    py = pos[m, j]

    vars_order = ["|u|", "ux", "uy", "uz"]

    # precompute values and limits
    cache = []
    limits = {k: [np.inf, -np.inf] for k in vars_order}
    for t in range(5):
        gt = yt[:, t, :]
        pr = yp[:, t, :]
        frame = {
            "|u|": (speed_mag(gt[m]), speed_mag(pr[m])),
            "ux": (gt[m, 0], pr[m, 0]),
            "uy": (gt[m, 1], pr[m, 1]),
            "uz": (gt[m, 2], pr[m, 2]),
        }
        for vn, (gv, pv) in frame.items():
            limits[vn][0] = min(limits[vn][0], gv.min(), pv.min())
            limits[vn][1] = max(limits[vn][1], gv.max(), pv.max())
        cache.append(frame)

    fig, axs = plt.subplots(2, 4, figsize=(20, 9), dpi=ANIM_DPI)
    title = fig.suptitle("")

    def draw(t):
        for r in range(2):
            for c in range(4):
                axs[r, c].clear()

        frame = cache[t]
        for c, vn in enumerate(vars_order):
            gv, pv = frame[vn]
            vmin, vmax = limits[vn]

            axs[0, c].scatter(px, py, c=gv, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
            axs[1, c].scatter(px, py, c=pv, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
            axs[0, c].set_title(f"GT {vn}")
            axs[1, c].set_title(f"Pred {vn}")

            for r in [0, 1]:
                axs[r, c].set_xlabel(labels[0])
                axs[r, c].set_ylabel(labels[1])
                axs[r, c].set_aspect("equal", adjustable="box")

        title.set_text(
            f"Sample {sample_idx} | frame {t} (chronological) | scatter | "
            f"section @{SECTION_AXIS}={target:.4f}±{hw:.4f} | relL2={rel:.4f}"
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
# Batch report
# =============================
def batch_report(y_true, y_pred, test_files, out_dir):
    B = y_true.shape[0]
    rels = np.array([rel_l2_sample(y_true[i], y_pred[i]) for i in range(B)])
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
        gt = flat15_to_t53(y_true[i])   # (N,5,3)
        pr = flat15_to_t53(y_pred[i])

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--frame_idx", type=int, default=0, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    single_dir = OUT_DIR / "single"
    anim_dir = OUT_DIR / "anim"
    report_dir = OUT_DIR / "report"
    single_dir.mkdir(parents=True, exist_ok=True)
    anim_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions + samples once...")
    y_true, y_pred, test_samples, test_files, raw_samples = load_once()

    print("Generating static plots...")
    p2d_cs = plot_2d_contour_scatter_panel(args.sample_idx, args.frame_idx, y_true, y_pred, test_samples, single_dir)
    p2d_sc = plot_2d_scatter_only_panel(args.sample_idx, args.frame_idx, y_true, y_pred, test_samples, single_dir)
    p3d_sc = plot_3d_scatter_panel(args.sample_idx, args.frame_idx, y_true, y_pred, test_samples, single_dir)
    p3d_ws = plot_3d_wing_surface(args.sample_idx, args.frame_idx, y_true, y_pred, test_samples, raw_samples, single_dir)
    p3d_cf = plot_3d_cfd_style(args.sample_idx, args.frame_idx, y_true, y_pred, test_samples, raw_samples, single_dir)

    print("Generating animations...")
    panim_c = make_contour_animation(args.sample_idx, y_true, y_pred, test_samples, anim_dir)
    panim_s = make_scatter_animation(args.sample_idx, y_true, y_pred, test_samples, anim_dir)

    print("Generating batch report...")
    preport, pcsv = batch_report(y_true, y_pred, test_files, report_dir)

    print("\nSaved outputs:")
    print(f"- {p2d_cs}")
    print(f"- {p2d_sc}")
    print(f"- {p3d_sc}")
    print(f"- {p3d_ws}")
    print(f"- {p3d_cf}")
    print(f"- {panim_c}")
    print(f"- {panim_s}")
    print(f"- {preport}")
    print(f"- {pcsv}")
    print(f"Source sample: {test_files[args.sample_idx].name}")


if __name__ == "__main__":
    main()
"""
visualise_fno_v2.py

Robust visualisation for the Warped-IFW FNO baseline when the saved
y_true_points.npy / y_pred_points.npy have a corrupt numpy header
(shape says (85,100000,15) but the actual data is smaller).

Situation on disk
-----------------
y_true_points.npy : header=(85,100000,15), actual data=(85,31136,15)  — clean
y_pred_points.npy : header=(85,100000,15), actual data≈(84,31136,15)  — last
                    sample truncated, 84 complete samples recoverable

Strategy
--------
1. Read both files by bypassing the header reshape: open as raw bytes,
   skip the numpy header, read float32 data directly.
2. Reconstruct (B, N, 15) using the known N=31136.
3. Use only the first B_safe=84 samples (where both arrays are complete).
4. Reload the matching test-split positions via load_split(), subsample
   each sample to N=31136 points with the same RNG seed used at training
   so that positions align with saved predictions.
5. Run the full visualisation pipeline on the recovered data.

All visualisation logic is identical to visualise_fno.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay

# ---------------------------------------------------------------------------
# Paths  (edit if your layout differs)
# ---------------------------------------------------------------------------
THIS        = Path(__file__).resolve()
SCRIPT_DIR  = THIS.parent
DATA_DIR    = SCRIPT_DIR.parent / "data" / "warped-ifw"
MODEL_DIR   = SCRIPT_DIR / "trained_model"
RESULTS_DIR = SCRIPT_DIR / "results"
OUT_DIR     = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Split config  (must match what was used at training time)
# ---------------------------------------------------------------------------
TEST_FRAC   = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Known geometry of the saved arrays
# ---------------------------------------------------------------------------
N_POINTS_SAVED = 31_136   # actual points per sample in the data region
B_TRUE         = 85       # complete samples in y_true
B_PRED         = 84       # complete samples in y_pred  (last one truncated)
B_SAFE         = min(B_TRUE, B_PRED)   # = 84, used everywhere
OUT_CH         = 15

# ---------------------------------------------------------------------------
# Plot config
# ---------------------------------------------------------------------------
FIG_DPI  = 140
ANIM_DPI = 130
ANIM_FPS = 2

CMAP_MAIN  = "turbo"
CMAP_ERR   = "magma"
CMAP_PRESS = "coolwarm"

SECTION_AXIS      = "y"
SECTION_QUANTILE  = 0.5
SECTION_BAND_FRAC = 0.015
GRID_NX           = 320
GRID_NY           = 190
GAUSS_SIGMA       = 1.0

N_REPORT_SAMPLES      = 10
SINGLE_SAMPLE_INDICES = [0, 10, 20, 30]
ANIM_SAMPLE_INDICES   = [0, 10, 20, 30]

FIELD_NAMES    = ["|u|", "ux", "uy", "uz"]
COMPONENT_COLS = {"ux": 0, "uy": 1, "uz": 2}


# ===========================================================================
# Raw numpy loader — bypasses the corrupt header shape
# ===========================================================================

def _read_npy_header(f) -> Tuple[dict, int]:
    """Return (header_dict, data_offset_bytes)."""
    magic = f.read(6)
    assert magic == b"\x93NUMPY", "Not a .npy file"
    major = f.read(1)[0]
    minor = f.read(1)[0]
    if major == 1:
        header_len = struct.unpack("<H", f.read(2))[0]
        data_offset = 6 + 1 + 1 + 2 + header_len
    else:
        header_len = struct.unpack("<I", f.read(4))[0]
        data_offset = 6 + 1 + 1 + 4 + header_len
    header_str = f.read(header_len).decode("latin1")
    # safe eval of the header dict
    import ast
    header = ast.literal_eval(header_str.strip().rstrip(","))
    return header, data_offset


def load_npy_raw_float32(path: Path) -> np.ndarray:
    """
    Load a float32 .npy file ignoring the shape in the header.
    Returns a 1-D float32 array of all data elements.
    """
    with open(path, "rb") as f:
        header, offset = _read_npy_header(f)
        f.seek(offset)
        data = np.frombuffer(f.read(), dtype=np.float32).copy()
    return data


def load_saved_predictions() -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    y_true : (B_SAFE, N_POINTS_SAVED, 15)  float32
    y_pred : (B_SAFE, N_POINTS_SAVED, 15)  float32
    """
    print("Reading y_true_points.npy (bypassing header shape)...")
    raw_true = load_npy_raw_float32(RESULTS_DIR / "y_true_points.npy")
    expected_true = B_TRUE * N_POINTS_SAVED * OUT_CH
    if raw_true.size != expected_true:
        raise ValueError(
            f"y_true: expected {expected_true} elements for "
            f"({B_TRUE},{N_POINTS_SAVED},{OUT_CH}), got {raw_true.size}"
        )
    y_true_full = raw_true.reshape(B_TRUE, N_POINTS_SAVED, OUT_CH)

    print("Reading y_pred_points.npy (bypassing header shape, truncated)...")
    raw_pred = load_npy_raw_float32(RESULTS_DIR / "y_pred_points.npy")
    # Take only the first B_PRED complete samples
    n_complete_elements = B_PRED * N_POINTS_SAVED * OUT_CH
    if raw_pred.size < n_complete_elements:
        raise ValueError(
            f"y_pred: need at least {n_complete_elements} elements for "
            f"{B_PRED} complete samples, but only {raw_pred.size} available."
        )
    y_pred_full = raw_pred[:n_complete_elements].reshape(B_PRED, N_POINTS_SAVED, OUT_CH)

    # Trim both to B_SAFE
    y_true = y_true_full[:B_SAFE]
    y_pred = y_pred_full[:B_SAFE]

    print(f"  y_true shape: {y_true.shape}")
    print(f"  y_pred shape: {y_pred.shape}")
    return y_true, y_pred


# ===========================================================================
# Data loading
# ===========================================================================

def load_once_fno():
    """
    Load predictions and reload the matching test split with position
    alignment.

    Because training used subsampled point clouds (N=31136 instead of
    100000), we subsample the reloaded full-resolution samples using the
    same seed so positions align with saved predictions.

    Returns
    -------
    y_true       : list of B_SAFE (N,15) arrays
    y_pred       : list of B_SAFE (N,15) arrays
    test_samples : list of B_SAFE dicts  {x:(N,3), fx:(N,16), y:(N,15), ...}
    test_files   : list of B_SAFE Paths
    raw_samples  : list of B_SAFE raw npz dicts
    """
    from load_data import (
        load_split, load_raw_sample, subsample_sample_uniform,
    )

    y_true_arr, y_pred_arr = load_saved_predictions()
    B = B_SAFE

    y_true = [y_true_arr[i] for i in range(B)]
    y_pred = [y_pred_arr[i] for i in range(B)]

    print("Reloading geometry-disjoint test split...")
    _, test_samples, _, test_files = load_split(
        data_dir=DATA_DIR, test_frac=TEST_FRAC, seed=RANDOM_SEED,
    )

    if B > len(test_samples):
        raise ValueError(
            f"Need {B} test samples but split only has {len(test_samples)}."
        )
    test_samples = test_samples[:B]
    test_files   = test_files[:B]

    # Align point counts via subsampling
    rng = np.random.default_rng(RANDOM_SEED)
    aligned = []
    for i, s in enumerate(test_samples):
        N_full = s["x"].shape[0]
        if N_full == N_POINTS_SAVED:
            aligned.append(s)
        elif N_full > N_POINTS_SAVED:
            s_sub = subsample_sample_uniform(
                s,
                n_points_subsample=N_POINTS_SAVED,
                rng=rng,
                keep_all_airfoil=True,
            )
            if s_sub["x"].shape[0] != N_POINTS_SAVED:
                raise ValueError(
                    f"Sample {i}: subsample gave {s_sub['x'].shape[0]} pts, "
                    f"expected {N_POINTS_SAVED}."
                )
            aligned.append(s_sub)
        else:
            raise ValueError(
                f"Sample {i}: only {N_full} pts but predictions need "
                f"{N_POINTS_SAVED}."
            )

    raw_samples = [load_raw_sample(fp) for fp in test_files]
    return y_true, y_pred, aligned, test_files, raw_samples


def load_grid_spec() -> dict:
    with open(MODEL_DIR / "grid_spec.json") as f:
        d = json.load(f)
    return {
        "xyz_min": np.array(d["xyz_min"], dtype=np.float32),
        "xyz_max": np.array(d["xyz_max"], dtype=np.float32),
        "gx": d["gx"], "gy": d["gy"], "gz": d["gz"],
    }


def load_training_history() -> Optional[dict]:
    p = MODEL_DIR / "training_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_final_metrics() -> Optional[dict]:
    p = RESULTS_DIR / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ===========================================================================
# Persistence baseline (no dependency on comparison.py internals)
# ===========================================================================

def persistence_baseline_relL2(test_samples) -> float:
    """
    Repeat last observed velocity frame (fx[:,12:15]) for all 5 outputs.
    Returns mean relative L2 across the first B_SAFE samples.
    """
    rels = []
    for s in test_samples[:B_SAFE]:
        fx = s["fx"][:, :15].astype(np.float32)
        last = fx[:, 12:15]
        pred = np.concatenate([last] * 5, axis=1)
        y    = s["y"].astype(np.float32)
        num  = np.sum((pred - y) ** 2)
        den  = np.sum(y ** 2) + 1e-12
        rels.append(float(np.sqrt(num / den)))
    return float(np.mean(rels))


# ===========================================================================
# Metrics
# ===========================================================================

def relative_l2_dataset(
    y_true: List[np.ndarray], y_pred: List[np.ndarray]
) -> Tuple[float, np.ndarray]:
    per = []
    for yt, yp in zip(y_true, y_pred):
        num = np.sum((yp - yt) ** 2)
        den = np.sum(yt ** 2) + 1e-12
        per.append(float(np.sqrt(num / den)))
    per = np.array(per, dtype=np.float32)
    return float(per.mean()), per


def rel_l2_sample(yt: np.ndarray, yp: np.ndarray) -> float:
    num = np.sum((yp - yt) ** 2)
    den = np.sum(yt ** 2) + 1e-12
    return float(np.sqrt(num / den))


def field_rel_l2(gt: np.ndarray, pr: np.ndarray) -> float:
    num = np.sum((pr - gt) ** 2)
    den = np.sum(gt ** 2) + 1e-12
    return float(np.sqrt(num / den))


# ===========================================================================
# Per-component losses
# ===========================================================================

def component_losses(yt_flat: np.ndarray, yp_flat: np.ndarray) -> Dict[str, float]:
    yt = yt_flat.reshape(-1, 5, 3)
    yp = yp_flat.reshape(-1, 5, 3)

    def _rl2(a, b):
        return float(np.sqrt(np.sum((b-a)**2) / (np.sum(a**2) + 1e-12)))

    def _mae(a, b):
        return float(np.mean(np.abs(b - a)))

    out = {}
    for name, col in COMPONENT_COLS.items():
        out[f"{name}_rel_l2"] = _rl2(yt[..., col], yp[..., col])
        out[f"{name}_mae"]    = _mae(yt[..., col], yp[..., col])
    out["speed_rel_l2"] = _rl2(
        np.linalg.norm(yt, axis=-1), np.linalg.norm(yp, axis=-1)
    )
    out["speed_mae"]    = _mae(
        np.linalg.norm(yt, axis=-1), np.linalg.norm(yp, axis=-1)
    )
    out["total_rel_l2"] = _rl2(yt_flat, yp_flat)
    return out


def dataset_component_losses(
    y_true: List[np.ndarray], y_pred: List[np.ndarray]
) -> Dict[str, np.ndarray]:
    keys = None
    rows = []
    for yt, yp in zip(y_true, y_pred):
        row = component_losses(yt, yp)
        if keys is None:
            keys = list(row.keys())
        rows.append([row[k] for k in keys])
    arr = np.array(rows, dtype=np.float32)
    return {k: arr[:, i] for i, k in enumerate(keys)}


def save_component_loss_csv(
    losses_dict: Dict[str, np.ndarray],
    path: Path,
    file_names: Optional[List[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    B    = len(next(iter(losses_dict.values())))
    keys = list(losses_dict.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_idx", "file"] + keys)
        writer.writeheader()
        for i in range(B):
            row = {"sample_idx": i, "file": file_names[i] if file_names else ""}
            row.update({k: float(losses_dict[k][i]) for k in keys})
            writer.writerow(row)


# ===========================================================================
# Geometry helpers
# ===========================================================================

def flat15_to_t53(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(*arr.shape[:-1], 5, 3)


def speed_mag(u: np.ndarray) -> np.ndarray:
    return np.linalg.norm(u, axis=-1)


def frame_fields(u: np.ndarray) -> Dict[str, np.ndarray]:
    return {"|u|": speed_mag(u), "ux": u[:, 0], "uy": u[:, 1], "uz": u[:, 2]}


def symmetric_error_limits(
    values: np.ndarray, q: float = 0.99, eps: float = 1e-8
) -> Tuple[float, float]:
    vmax = float(np.quantile(np.abs(values), q))
    return -(max(vmax, eps)), max(vmax, eps)


def choose_section_mask(
    pos: np.ndarray,
    axis: str = "y",
    quantile: float = 0.5,
    band_frac: float = 0.015,
) -> Tuple[np.ndarray, float, float]:
    k    = {"x": 0, "y": 1, "z": 2}[axis]
    vals = pos[:, k]
    lo, hi = vals.min(), vals.max()
    target = float(np.quantile(vals, quantile))
    hw     = max((hi - lo) * band_frac, 1e-8)
    mask   = np.abs(vals - target) <= hw
    for factor in [2.0, 4.0]:
        if mask.sum() >= 100:
            break
        hw   *= factor
        mask  = np.abs(vals - target) <= hw
    return mask, target, hw


def projected_axes(axis: str) -> Tuple[int, int, Tuple[str, str]]:
    return {"x": (1, 2, ("y", "z")),
            "y": (0, 2, ("x", "z")),
            "z": (0, 1, ("x", "y"))}[axis]


def interpolate_masked(
    px: np.ndarray, py: np.ndarray, values: np.ndarray,
    nx: int = 320, ny: int = 190, sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi = np.linspace(px.min(), px.max(), nx)
    yi = np.linspace(py.min(), py.max(), ny)
    XI, YI = np.meshgrid(xi, yi)
    pts    = np.column_stack([px, py])
    z_lin  = griddata(pts, values, (XI, YI), method="linear")
    z_near = griddata(pts, values, (XI, YI), method="nearest")
    Z      = np.where(np.isnan(z_lin), z_near, z_lin)
    inside = (
        Delaunay(pts).find_simplex(
            np.column_stack([XI.ravel(), YI.ravel()])
        ) >= 0
    ).reshape(XI.shape)
    Z = np.where(inside, gaussian_filter(Z, sigma=sigma) if sigma > 0 else Z, np.nan)
    return XI, YI, Z


def pressure_out_frame(raw_sample: dict, frame_idx: int) -> np.ndarray:
    return raw_sample["pressure"].astype(np.float32)[frame_idx + 5]


def build_aligned_sample(
    sample_idx: int,
    y_true: List[np.ndarray],
    y_pred: List[np.ndarray],
    test_samples: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos  = test_samples[sample_idx]["x"]
    yt_s = y_true[sample_idx]
    yp_s = y_pred[sample_idx]
    if not (pos.shape[0] == yt_s.shape[0] == yp_s.shape[0]):
        raise ValueError(
            f"Sample {sample_idx}: point count mismatch "
            f"pos={pos.shape[0]} y_true={yt_s.shape[0]} y_pred={yp_s.shape[0]}"
        )
    return pos, yt_s, yp_s


# ===========================================================================
# Training curves
# ===========================================================================

def plot_training_curves(
    history: dict,
    out_dir: Path,
    final_point_relL2: Optional[float] = None,
    baseline_relL2: Optional[float] = None,
) -> Path:
    epochs = np.arange(len(history["loss"]))
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), dpi=FIG_DPI)

    axs[0].plot(epochs, history["loss"],     label="train (grid, normalised)")
    axs[0].plot(epochs, history["val_loss"], label="val  (grid, normalised)")
    if final_point_relL2 is not None:
        axs[0].axhline(
            final_point_relL2, color="tab:red", ls="--",
            label=f"point relL2 (recovered)={final_point_relL2:.4f}",
        )
    if baseline_relL2 is not None:
        axs[0].axhline(
            baseline_relL2, color="gray", ls=":",
            label=f"persistence baseline={baseline_relL2:.4f}",
        )
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("relative L2 (grid, normalised space)")
    axs[0].set_title(
        "Training curves\n"
        "(grid-space metric ≠ point-level competition metric)"
    )
    axs[0].legend(fontsize=8)

    axs[1].plot(epochs, history["lr"])
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("learning rate")
    axs[1].set_title("LR schedule")

    axs[2].plot(epochs, history["time"])
    axs[2].set_xlabel("epoch"); axs[2].set_ylabel("seconds")
    axs[2].set_title("Time per epoch")

    fig.tight_layout()
    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# Grid diagnostic
# ===========================================================================

def plot_grid_diagnostic(
    sample: dict, grid_spec: dict, out_dir: Path, slice_frac: float = 0.5
) -> Path:
    from config import points_to_grid
    Xg, Yg, Mg = points_to_grid(sample, grid_spec)
    gy = grid_spec["gy"]
    j  = int(np.clip(round(slice_frac * (gy - 1)), 0, gy - 1))

    occ    = Mg[0, :, j, :]
    target = Yg[0, :, j, :]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), dpi=FIG_DPI)
    im0 = axs[0].imshow(occ.T, origin="lower", cmap="gray", aspect="auto")
    axs[0].set_title(f"Grid occupancy | y-slice j={j}/{gy}")
    fig.colorbar(im0, ax=axs[0], shrink=0.8)

    im1 = axs[1].imshow(target.T, origin="lower", cmap=CMAP_MAIN, aspect="auto")
    axs[1].set_title("Rasterised target ux (frame 0)")
    fig.colorbar(im1, ax=axs[1], shrink=0.8)

    for ax in axs:
        ax.set_xlabel("gx"); ax.set_ylabel("gz")
    fig.tight_layout()

    out_path = out_dir / "grid_rasterization_diagnostic.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# 2-D static plots
# ===========================================================================

def _contour_row(ax_gt, ax_pr, ax_er, px, py, gt_vals, pr_vals, name, fig,
                 labels=("x", "z")):
    rel    = field_rel_l2(gt_vals, pr_vals)
    er_vals = gt_vals - pr_vals

    XI, YI, Zg = interpolate_masked(px, py, gt_vals, GRID_NX, GRID_NY, GAUSS_SIGMA)
    _,  _,  Zp = interpolate_masked(px, py, pr_vals, GRID_NX, GRID_NY, GAUSS_SIGMA)
    _,  _,  Ze = interpolate_masked(px, py, er_vals, GRID_NX, GRID_NY, GAUSS_SIGMA)

    vmin   = np.nanmin([Zg, Zp])
    vmax   = np.nanmax([Zg, Zp])
    levels = np.linspace(vmin, vmax, 50)
    emin, emax = symmetric_error_limits(np.nan_to_num(Ze, nan=0.0).ravel())
    elevels = np.linspace(emin, emax, 50)

    cf0 = ax_gt.contourf(XI, YI, Zg, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
    ax_gt.scatter(px, py, c=gt_vals, s=2, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax, alpha=0.3)
    ax_gt.set_title(f"GT {name}")
    fig.colorbar(cf0, ax=ax_gt, shrink=0.82)

    cf1 = ax_pr.contourf(XI, YI, Zp, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
    ax_pr.scatter(px, py, c=pr_vals, s=2, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax, alpha=0.3)
    ax_pr.set_title(f"Pred {name}")
    fig.colorbar(cf1, ax=ax_pr, shrink=0.82)

    cf2 = ax_er.contourf(XI, YI, Ze, levels=elevels, cmap=CMAP_ERR, alpha=0.95, extend="both")
    ax_er.scatter(px, py, c=er_vals, s=2, cmap=CMAP_ERR, vmin=emin, vmax=emax, alpha=0.3)
    ax_er.set_title(f"Error {name} | relL2={rel:.4f}")
    fig.colorbar(cf2, ax=ax_er, shrink=0.82)

    for ax in [ax_gt, ax_pr, ax_er]:
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        ax.set_aspect("equal", adjustable="box")
    return rel


def plot_2d_contour_scatter_panel(
    sample_idx, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir
) -> Path:
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples)
    yt = flat15_to_t53(yt_s)[:, frame_idx, :]
    yp = flat15_to_t53(yp_s)[:, frame_idx, :]
    gt_f = frame_fields(yt); pr_f = frame_fields(yp)

    mask, target, hw = choose_section_mask(pos, SECTION_AXIS, SECTION_QUANTILE, SECTION_BAND_FRAC)
    i, j, labels = projected_axes(SECTION_AXIS)
    px, py_ax = pos[mask, i], pos[mask, j]

    fig, axs = plt.subplots(3, 4, figsize=(20, 13), dpi=FIG_DPI)
    for c, name in enumerate(FIELD_NAMES):
        _contour_row(axs[0,c], axs[1,c], axs[2,c], px, py_ax,
                     gt_f[name][mask], pr_f[name][mask], name, fig, labels=labels)
    fig.suptitle(
        f"Sample {sample_idx} frame {frame_idx} | contour+scatter | "
        f"section @{SECTION_AXIS}={target:.4f}±{hw:.4f}"
    )
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_2d_contour_scatter.png"
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def plot_2d_scatter_only_panel(
    sample_idx, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir
) -> Path:
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples)
    yt = flat15_to_t53(yt_s)[:, frame_idx, :]
    yp = flat15_to_t53(yp_s)[:, frame_idx, :]
    gt_f = frame_fields(yt); pr_f = frame_fields(yp)

    mask, target, hw = choose_section_mask(pos, SECTION_AXIS, SECTION_QUANTILE, SECTION_BAND_FRAC)
    i, j, labels = projected_axes(SECTION_AXIS)
    px, py_ax = pos[mask, i], pos[mask, j]

    fig, axs = plt.subplots(3, 4, figsize=(20, 13), dpi=FIG_DPI)
    for c, name in enumerate(FIELD_NAMES):
        gt_v = gt_f[name][mask]; pr_v = pr_f[name][mask]; er_v = gt_v - pr_v
        rel  = field_rel_l2(gt_v, pr_v)
        vmin, vmax = min(gt_v.min(), pr_v.min()), max(gt_v.max(), pr_v.max())
        emin, emax = symmetric_error_limits(er_v)

        sc0 = axs[0,c].scatter(px, py_ax, c=gt_v, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
        axs[0,c].set_title(f"GT {name}"); fig.colorbar(sc0, ax=axs[0,c], shrink=0.82)

        sc1 = axs[1,c].scatter(px, py_ax, c=pr_v, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
        axs[1,c].set_title(f"Pred {name}"); fig.colorbar(sc1, ax=axs[1,c], shrink=0.82)

        sc2 = axs[2,c].scatter(px, py_ax, c=er_v, s=4, cmap=CMAP_ERR, vmin=emin, vmax=emax)
        axs[2,c].set_title(f"Error {name} | relL2={rel:.4f}")
        fig.colorbar(sc2, ax=axs[2,c], shrink=0.82)

        for r in range(3):
            axs[r,c].set_xlabel(labels[0]); axs[r,c].set_ylabel(labels[1])
            axs[r,c].set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Sample {sample_idx} frame {frame_idx} | scatter-only | "
        f"section @{SECTION_AXIS}={target:.4f}±{hw:.4f}"
    )
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_2d_scatter.png"
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


# ===========================================================================
# 3-D wing surface
# ===========================================================================

def plot_3d_wing_surface(
    sample_idx, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir
) -> Optional[Path]:
    s   = test_samples[sample_idx]
    raw = raw_samples[sample_idx]
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples)
    idcs_air = s["idcs_airfoil"]
    yt  = flat15_to_t53(yt_s)[:, frame_idx, :]
    yp  = flat15_to_t53(yp_s)[:, frame_idx, :]
    rel = rel_l2_sample(yt_s, yp_s)

    p_full = pressure_out_frame(raw, frame_idx)
    p_surf = (
        p_full[idcs_air]
        if idcs_air.size > 0 and idcs_air.max() < p_full.shape[0]
        else np.zeros(idcs_air.size, dtype=np.float32)
    )

    surf_pts = s["x"][idcs_air]
    if surf_pts.shape[0] < 3:
        return None

    tri2 = Triangulation(surf_pts[:, 0], surf_pts[:, 2])
    pmin, pmax = p_surf.min(), p_surf.max()

    fig = plt.figure(figsize=(14, 6), dpi=FIG_DPI)
    ax0 = fig.add_axes([0.02, 0.08, 0.42, 0.85], projection="3d")
    ax1 = fig.add_axes([0.53, 0.08, 0.42, 0.85], projection="3d")
    cax = fig.add_axes([0.465, 0.15, 0.015, 0.70])

    for ax, u, ttl in [(ax0, yt, "GT"), (ax1, yp, "Pred")]:
        surf = ax.plot_trisurf(
            surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2],
            triangles=tri2.triangles, cmap=CMAP_PRESS,
            linewidth=0.0, antialiased=False, shade=False, alpha=0.98,
        )
        surf.set_array(p_surf[tri2.triangles].mean(axis=1).astype(np.float32))
        surf.set_clim(pmin, pmax)

        y_col = pos[:, 1]
        y0    = float(np.quantile(y_col, 0.5))
        dy    = max((y_col.max() - y_col.min()) * 0.015, 1e-6)
        m     = np.abs(y_col - y0) <= dy
        ux    = u[m, 0]
        uxmin, uxmax = np.quantile(ux, [0.01, 0.99])
        ax.scatter(pos[m, 0], pos[m, 1], pos[m, 2], c=ux, s=1.2,
                   cmap=CMAP_MAIN, vmin=uxmin, vmax=uxmax, alpha=0.45)
        ax.set_title(f"{ttl}: wing surface + section ux")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-60)

    sm = plt.cm.ScalarMappable(
        cmap=CMAP_PRESS, norm=plt.Normalize(vmin=pmin, vmax=pmax)
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label("Pressure (Pa) on wing")
    fig.suptitle(f"Sample {sample_idx} frame {frame_idx} | 3D wing | relL2={rel:.4f}")
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_3d_wing_surface.png"
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


# ===========================================================================
# Per-component loss plots
# ===========================================================================

def plot_dataset_component_losses(
    losses_dict: Dict[str, np.ndarray], out_dir: Path
) -> Path:
    keys_to_plot = ["ux_rel_l2", "uy_rel_l2", "uz_rel_l2", "speed_rel_l2"]
    labels       = ["ux", "uy", "uz", "|u|"]
    data         = [losses_dict[k] for k in keys_to_plot if k in losses_dict]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=FIG_DPI)
    bp = ax.boxplot(data, labels=labels[:len(data)], patch_artist=True)
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour); patch.set_alpha(0.7)
    ax.set_ylabel("Relative L2")
    ax.set_title("Per-component loss distribution across test set")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()

    out_path = out_dir / "dataset_component_loss_boxplot.png"
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def plot_component_loss_breakdown(
    sample_idx: int, y_true, y_pred, out_dir: Path
) -> Path:
    yt = flat15_to_t53(y_true[sample_idx])
    yp = flat15_to_t53(y_pred[sample_idx])

    components = ["ux", "uy", "uz", "|u|"]
    x = np.arange(5); width = 0.2

    fig, ax = plt.subplots(figsize=(10, 4), dpi=FIG_DPI)
    for ci, comp in enumerate(components):
        losses = []
        for t in range(5):
            if comp == "|u|":
                gt_v = np.linalg.norm(yt[:, t, :], axis=-1)
                pr_v = np.linalg.norm(yp[:, t, :], axis=-1)
            else:
                col  = COMPONENT_COLS[comp]
                gt_v = yt[:, t, col]; pr_v = yp[:, t, col]
            losses.append(field_rel_l2(gt_v, pr_v))
        ax.bar(x + ci * width, losses, width, label=comp)

    ax.set_xlabel("Frame"); ax.set_ylabel("Relative L2")
    ax.set_title(f"Sample {sample_idx} — per-component relative L2 by frame")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"t={t}" for t in range(5)])
    ax.legend(); fig.tight_layout()

    out_path = out_dir / f"sample{sample_idx:03d}_component_losses.png"
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


# ===========================================================================
# Animations
# ===========================================================================

def _precompute_contour_cache(pos, yt, yp, mask, i, j):
    cache = []
    lim_main = {n: [np.inf, -np.inf] for n in FIELD_NAMES}
    lim_err  = {n: [np.inf, -np.inf] for n in FIELD_NAMES}
    px, py_ax = pos[mask, i], pos[mask, j]
    for t in range(5):
        gt_f = frame_fields(yt[:, t, :]); pr_f = frame_fields(yp[:, t, :])
        fc = {}
        for name in FIELD_NAMES:
            rel = field_rel_l2(gt_f[name], pr_f[name])
            gv  = gt_f[name][mask]; pv = pr_f[name][mask]
            ev  = (gt_f[name] - pr_f[name])[mask]
            XI, YI, Zg = interpolate_masked(px, py_ax, gv, GRID_NX, GRID_NY, GAUSS_SIGMA)
            _,  _,  Zp = interpolate_masked(px, py_ax, pv, GRID_NX, GRID_NY, GAUSS_SIGMA)
            _,  _,  Ze = interpolate_masked(px, py_ax, ev, GRID_NX, GRID_NY, GAUSS_SIGMA)
            lim_main[name][0] = min(lim_main[name][0], np.nanmin([Zg, Zp]))
            lim_main[name][1] = max(lim_main[name][1], np.nanmax([Zg, Zp]))
            fin = Ze[~np.isnan(Ze)]
            if fin.size:
                lim_err[name][0] = min(lim_err[name][0], fin.min())
                lim_err[name][1] = max(lim_err[name][1], fin.max())
            fc[name] = (XI, YI, Zg, Zp, Ze, rel)
        cache.append(fc)
    err_sym = {n: symmetric_error_limits(np.array(lim_err[n])) for n in FIELD_NAMES}
    return cache, lim_main, err_sym, px, py_ax


def make_contour_animation(
    sample_idx, y_true, y_pred, test_samples, raw_samples, out_dir
) -> Path:
    out_path = out_dir / f"sample{sample_idx:03d}_2d_contour_animation.mp4"
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples)
    yt = flat15_to_t53(yt_s); yp = flat15_to_t53(yp_s)
    mask, target, hw = choose_section_mask(pos, SECTION_AXIS, SECTION_QUANTILE, SECTION_BAND_FRAC)
    i, j, labels = projected_axes(SECTION_AXIS)
    cache, lim_main, err_sym, px, py_ax = _precompute_contour_cache(pos, yt, yp, mask, i, j)

    fig, axs = plt.subplots(3, 4, figsize=(20, 13), dpi=ANIM_DPI)
    title = fig.suptitle("")

    def draw(t):
        [ax.clear() for row in axs for ax in row]
        fc = cache[t]
        for c, name in enumerate(FIELD_NAMES):
            XI, YI, Zg, Zp, Ze, rel = fc[name]
            vmin, vmax = lim_main[name]
            levels  = np.linspace(vmin, vmax, 50)
            emin, emax = err_sym[name]
            elevels = np.linspace(emin, emax, 50)
            axs[0,c].contourf(XI, YI, Zg, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
            axs[1,c].contourf(XI, YI, Zp, levels=levels, cmap=CMAP_MAIN, alpha=0.95, extend="both")
            axs[2,c].contourf(XI, YI, Ze, levels=elevels, cmap=CMAP_ERR, alpha=0.95, extend="both")
            axs[0,c].set_title(f"GT {name}")
            axs[1,c].set_title(f"Pred {name}")
            axs[2,c].set_title(f"Error {name} | relL2={rel:.4f}")
            for r in range(3):
                axs[r,c].set_xlabel(labels[0]); axs[r,c].set_ylabel(labels[1])
                axs[r,c].set_aspect("equal", adjustable="box")
        title.set_text(
            f"Sample {sample_idx} | frame {t} | contour | "
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


def make_scatter_animation(
    sample_idx, y_true, y_pred, test_samples, raw_samples, out_dir
) -> Path:
    out_path = out_dir / f"sample{sample_idx:03d}_2d_scatter_animation.mp4"
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples)
    yt = flat15_to_t53(yt_s); yp = flat15_to_t53(yp_s)
    mask, target, hw = choose_section_mask(pos, SECTION_AXIS, SECTION_QUANTILE, SECTION_BAND_FRAC)
    i, j, labels = projected_axes(SECTION_AXIS)
    px, py_ax = pos[mask, i], pos[mask, j]

    cache = []
    lim_main = {n: [np.inf, -np.inf] for n in FIELD_NAMES}
    lim_err  = {n: [np.inf, -np.inf] for n in FIELD_NAMES}
    for t in range(5):
        gt_f = frame_fields(yt[:, t, :]); pr_f = frame_fields(yp[:, t, :])
        fc = {}
        for name in FIELD_NAMES:
            rel = field_rel_l2(gt_f[name], pr_f[name])
            gv  = gt_f[name][mask]; pv = pr_f[name][mask]; ev = gv - pv
            lim_main[name][0] = min(lim_main[name][0], gv.min(), pv.min())
            lim_main[name][1] = max(lim_main[name][1], gv.max(), pv.max())
            lim_err[name][0]  = min(lim_err[name][0], ev.min())
            lim_err[name][1]  = max(lim_err[name][1], ev.max())
            fc[name] = (gv, pv, ev, rel)
        cache.append(fc)
    err_sym = {n: symmetric_error_limits(np.array(lim_err[n])) for n in FIELD_NAMES}

    fig, axs = plt.subplots(3, 4, figsize=(20, 13), dpi=ANIM_DPI)
    title = fig.suptitle("")

    def draw(t):
        [ax.clear() for row in axs for ax in row]
        fc = cache[t]
        for c, name in enumerate(FIELD_NAMES):
            gv, pv, ev, rel = fc[name]
            vmin, vmax = lim_main[name]
            emin, emax = err_sym[name]
            axs[0,c].scatter(px, py_ax, c=gv, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
            axs[1,c].scatter(px, py_ax, c=pv, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
            axs[2,c].scatter(px, py_ax, c=ev, s=4, cmap=CMAP_ERR,  vmin=emin, vmax=emax)
            axs[0,c].set_title(f"GT {name}")
            axs[1,c].set_title(f"Pred {name}")
            axs[2,c].set_title(f"Error {name} | relL2={rel:.4f}")
            for r in range(3):
                axs[r,c].set_xlabel(labels[0]); axs[r,c].set_ylabel(labels[1])
                axs[r,c].set_aspect("equal", adjustable="box")
        title.set_text(
            f"Sample {sample_idx} | frame {t} | scatter | "
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


# ===========================================================================
# MATLAB export
# ===========================================================================

def export_matlab_bundle(
    sample_indices, y_true, y_pred, test_samples, test_files, raw_samples, out_dir
) -> Path:
    bundle: Dict = {
        "sample_indices":    np.asarray(sample_indices, dtype=np.int32),
        "section_axis":      SECTION_AXIS,
        "section_quantile":  SECTION_QUANTILE,
        "section_band_frac": SECTION_BAND_FRAC,
        "n_points_per_sample": N_POINTS_SAVED,
        "b_safe": B_SAFE,
    }
    for sidx in sample_indices:
        pos, yt_s, yp_s = build_aligned_sample(sidx, y_true, y_pred, test_samples)
        key = f"sample_{sidx:03d}"
        bundle[f"{key}_x"]            = pos.astype(np.float32)
        bundle[f"{key}_y_true"]       = yt_s.astype(np.float32)
        bundle[f"{key}_y_pred"]       = yp_s.astype(np.float32)
        bundle[f"{key}_idcs_airfoil"] = test_samples[sidx]["idcs_airfoil"].astype(np.int32)
        bundle[f"{key}_file"]         = test_files[sidx].name
        if "pressure" in raw_samples[sidx]:
            bundle[f"{key}_pressure"] = raw_samples[sidx]["pressure"].astype(np.float32)

    try:
        from scipy.io import savemat
        out_path = out_dir / "matlab_export_bundle.mat"
        savemat(out_path, bundle, do_compression=True)
    except Exception:
        out_path = out_dir / "matlab_export_bundle.npz"
        np.savez_compressed(out_path, **bundle)
    return out_path


# ===========================================================================
# Batch report
# ===========================================================================

def batch_report(
    y_true, y_pred, test_files, out_dir
) -> Tuple[Path, Path]:
    B    = len(y_true)
    rels = np.array([rel_l2_sample(y_true[i], y_pred[i]) for i in range(B)])
    order = np.argsort(rels)
    n     = min(N_REPORT_SAMPLES, B)
    chosen = np.unique(np.concatenate([
        order[:max(1, n // 3)],
        order[B // 2: B // 2 + max(1, n // 3)],
        order[-max(1, n - 2 * (n // 3)):],
    ]))[:n]

    fig, axs = plt.subplots(len(chosen), 2, figsize=(12, 3 * len(chosen)), dpi=FIG_DPI)
    if len(chosen) == 1:
        axs = axs[None, :]

    for r, i in enumerate(chosen):
        gt = flat15_to_t53(y_true[i]); pr = flat15_to_t53(y_pred[i])
        gt_m = speed_mag(gt).mean(axis=0)
        pr_m = speed_mag(pr).mean(axis=0)
        er_m = speed_mag(pr - gt).mean(axis=0)

        axs[r, 0].plot(gt_m, marker="o", label="GT mean|u|")
        axs[r, 0].plot(pr_m, marker="s", label="Pred mean|u|")
        axs[r, 0].set_title(f"sample {i} ({test_files[i].name})")
        axs[r, 0].set_xlabel("frame"); axs[r, 0].set_ylabel("mean |u|")
        axs[r, 0].legend()

        axs[r, 1].plot(er_m, marker="o", color="tab:red")
        axs[r, 1].set_title(f"mean |Δu| per frame | relL2={rels[i]:.4f}")
        axs[r, 1].set_xlabel("frame"); axs[r, 1].set_ylabel("mean error")

    fig.tight_layout()
    png = out_dir / "batch_report.png"
    fig.savefig(png, bbox_inches="tight"); plt.close(fig)

    csv_path = out_dir / "batch_report_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_idx", "file", "rel_l2"])
        writer.writeheader()
        for i in range(B):
            writer.writerow({
                "sample_idx": i,
                "file": test_files[i].name,
                "rel_l2": f"{rels[i]:.8f}",
            })
    return png, csv_path


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="FNO visualisation v2 (corrupt-header safe)")
    parser.add_argument("--frame-idx",           type=int,  default=0, choices=range(5))
    parser.add_argument("--no-animations",        action="store_true")
    parser.add_argument("--no-3d",               action="store_true")
    parser.add_argument("--no-training-curves",  action="store_true")
    parser.add_argument("--no-grid-diagnostic",  action="store_true")
    parser.add_argument("--grid-diag-sample",    type=int,  default=0)
    args = parser.parse_args()

    single_dir = OUT_DIR / "single"
    anim_dir   = OUT_DIR / "anim"
    report_dir = OUT_DIR / "report"
    matlab_dir = OUT_DIR / "matlab"
    comp_dir   = OUT_DIR / "component_losses"
    diag_dir   = OUT_DIR / "diagnostics"
    for d in [single_dir, anim_dir, report_dir, matlab_dir, comp_dir, diag_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Warped-IFW FNO Visualisation v2")
    print(f"  Recovered samples : {B_SAFE}  (of 85 saved)")
    print(f"  Points per sample : {N_POINTS_SAVED}")
    print(f"{'='*60}\n")

    print("Loading predictions and reloading test split...")
    y_true, y_pred, test_samples, test_files, raw_samples = load_once_fno()
    B = len(y_true)
    print(f"  {B} samples ready.\n")

    # ── Metrics summary ──────────────────────────────────────────────────────
    mean_rl2, per_rl2 = relative_l2_dataset(y_true, y_pred)
    print(f"Point-level relative L2 (recovered {B} samples):")
    print(f"  mean = {mean_rl2:.6f}")
    print(f"  std  = {per_rl2.std():.6f}")
    print(f"  min  = {per_rl2.min():.6f}  max = {per_rl2.max():.6f}\n")

    # ── Training curves ──────────────────────────────────────────────────────
    if not args.no_training_curves:
        history = load_training_history()
        if history is not None:
            final_metrics  = load_final_metrics()
            final_relL2    = final_metrics["relative_L2_mean"] if final_metrics else mean_rl2
            baseline_relL2 = persistence_baseline_relL2(test_samples)
            print(f"Persistence baseline relL2 = {baseline_relL2:.6f}")
            tc_path = plot_training_curves(
                history, diag_dir,
                final_point_relL2=final_relL2,
                baseline_relL2=baseline_relL2,
            )
            print(f"Training curves → {tc_path}")
        else:
            print("training_history.json not found, skipping training curves.")

    # ── Grid diagnostic ──────────────────────────────────────────────────────
    if not args.no_grid_diagnostic:
        gidx = args.grid_diag_sample
        if gidx < B:
            grid_spec = load_grid_spec()
            gd_path   = plot_grid_diagnostic(test_samples[gidx], grid_spec, diag_dir)
            print(f"Grid diagnostic → {gd_path}")
        else:
            print(f"--grid-diag-sample {gidx} out of range for {B} samples, skipping.")

    # ── Per-component losses ─────────────────────────────────────────────────
    print("\nComputing per-component losses...")
    comp_losses = dataset_component_losses(y_true, y_pred)
    print(f"\n  {'Metric':<20} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*42}")
    for k, v in comp_losses.items():
        print(f"  {k:<20} {v.mean():>10.4f} {v.std():>10.4f}")
    print()

    comp_csv = comp_dir / "component_losses_per_sample.csv"
    save_component_loss_csv(
        comp_losses, comp_csv, file_names=[f.name for f in test_files]
    )
    bp_path = plot_dataset_component_losses(comp_losses, comp_dir)
    print(f"Component loss box plot → {bp_path}")

    # ── Static plots ─────────────────────────────────────────────────────────
    chosen_samples = [i for i in SINGLE_SAMPLE_INDICES if i < B]
    if not chosen_samples:
        raise ValueError(
            f"None of SINGLE_SAMPLE_INDICES={SINGLE_SAMPLE_INDICES} are "
            f"valid for {B} samples."
        )

    print("\nGenerating static plots...")
    for sidx in chosen_samples:
        sd = single_dir / f"{sidx:03d}"
        sd.mkdir(parents=True, exist_ok=True)
        plot_2d_contour_scatter_panel(sidx, args.frame_idx, y_true, y_pred, test_samples, raw_samples, sd)
        plot_2d_scatter_only_panel(sidx, args.frame_idx, y_true, y_pred, test_samples, raw_samples, sd)
        if not args.no_3d:
            plot_3d_wing_surface(sidx, args.frame_idx, y_true, y_pred, test_samples, raw_samples, sd)
        plot_component_loss_breakdown(sidx, y_true, y_pred, sd)
        print(f"  Sample {sidx} done.")

    # ── Animations ───────────────────────────────────────────────────────────
    if not args.no_animations:
        print("\nGenerating animations...")
        for sidx in [i for i in ANIM_SAMPLE_INDICES if i < B]:
            sad = anim_dir / f"{sidx:03d}"
            sad.mkdir(parents=True, exist_ok=True)
            make_contour_animation(sidx, y_true, y_pred, test_samples, raw_samples, sad)
            make_scatter_animation(sidx, y_true, y_pred, test_samples, raw_samples, sad)
            print(f"  Sample {sidx} animations done.")
    else:
        print("Animations skipped (--no-animations).")

    # ── MATLAB export ────────────────────────────────────────────────────────
    print("\nExporting MATLAB bundle...")
    matlab_path = export_matlab_bundle(
        chosen_samples, y_true, y_pred, test_samples, test_files, raw_samples, matlab_dir
    )

    # ── Batch report ─────────────────────────────────────────────────────────
    print("Generating batch report...")
    preport, pcsv = batch_report(y_true, y_pred, test_files, report_dir)

    print("\n" + "="*60)
    print("All outputs saved:")
    print(f"  Static plots     : {single_dir}")
    print(f"  Animations       : {anim_dir}")
    print(f"  Component losses : {comp_dir}")
    print(f"  Diagnostics      : {diag_dir}")
    print(f"  MATLAB bundle    : {matlab_path}")
    print(f"  Batch report     : {preport}")
    print(f"  Metrics CSV      : {pcsv}")
    print(f"  Component CSV    : {comp_csv}")
    print("="*60)


if __name__ == "__main__":
    main()
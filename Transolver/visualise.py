"""
visualise.py

Unified visualisation pipeline for Warped-IFW / TransolverAR results.

New plots added from visualise_fno_v2.py:
  - plot_training_curves()              Training loss + LR + time per epoch
  - plot_per_sample_relL2_histogram()   Distribution of per-sample relL2
  - plot_error_vs_frame()               How error grows across output frames
  - plot_worst_best_samples()           Side-by-side worst / best predictions
  - plot_speed_magnitude_profile()      Spanwise mean-speed profile GT vs Pred
  - plot_per_frame_component_heatmap()  (frame × component) relL2 heatmap
  - plot_ux_summary_4x3()              Multi-sample ux contour summary (existing, kept)
  - plot_component_loss_breakdown()     Per-sample per-frame bar chart (existing, kept)
  - plot_dataset_component_losses()     Box plot across test set (existing, kept + mpl compat)

All existing plots are preserved unchanged.
W&B logging, position regeneration, and CSV export are unchanged.

Directory layout (unchanged):
  THIS = visualise.py
  TRANSOLVER_DIR = THIS.parent
  PROJECT_DIR    = TRANSOLVER_DIR.parent
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay

# ---------------------------------------------------------------------------
# matplotlib version shim  (boxplot: 'labels' → 'tick_labels' in mpl >= 3.9)
# ---------------------------------------------------------------------------
_MPL_VERSION = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])


def _boxplot(ax, data, tick_labels, **kwargs):
    """Version-safe wrapper around ax.boxplot()."""
    if _MPL_VERSION >= (3, 9):
        return ax.boxplot(data, tick_labels=tick_labels, **kwargs)
    return ax.boxplot(data, labels=tick_labels, **kwargs)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
THIS           = Path(__file__).resolve()
TRANSOLVER_DIR = THIS.parent
PROJECT_DIR    = TRANSOLVER_DIR.parent

DATA_DIR    = PROJECT_DIR / "data" / "warped-ifw"
RESULTS_DIR = TRANSOLVER_DIR / "results"
MODEL_DIR   = TRANSOLVER_DIR / "trained_model"
OUT_DIR     = TRANSOLVER_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot config
# ---------------------------------------------------------------------------
FIG_DPI  = 140
ANIM_DPI = 130
ANIM_FPS = 2

CMAP_MAIN  = "turbo"
CMAP_ERR   = "magma"
CMAP_PRESS = "coolwarm"

SECTION_AXIS       = "y"
SECTION_QUANTILE   = 0.5
SECTION_BAND_FRAC  = 0.015
GRID_NX            = 320
GRID_NY            = 190
GAUSS_SIGMA        = 1.0

N_REPORT_SAMPLES       = 10
SINGLE_SAMPLE_INDICES  = [0, 10, 20, 30]
ANIM_SAMPLE_INDICES    = [0, 10, 20, 30]

FIELD_NAMES    = ["|u|", "ux", "uy", "uz"]
COMPONENT_COLS = {"ux": 0, "uy": 1, "uz": 2}


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
def _import_project():
    from load_data import load_raw_sample, load_split
    from preprocessing import preprocess_samples
    from config import RunConfig
    return load_raw_sample, load_split, preprocess_samples, RunConfig


# ===========================================================================
# Position regeneration
# ===========================================================================

def _load_run_config() -> Optional[object]:
    cfg_path = MODEL_DIR / "run_config.json"
    if not cfg_path.exists():
        candidates = sorted(MODEL_DIR.glob("run_*/config.json"))
        if not candidates:
            warnings.warn(
                f"No run_config.json found under {MODEL_DIR}. "
                "Position regeneration will use default config values."
            )
            return None
        cfg_path = candidates[-1]

    _, _, _, RunConfig = _import_project()
    with open(cfg_path) as f:
        d = json.load(f)
    known = {k for k in RunConfig.__dataclass_fields__}
    return RunConfig(**{k: v for k, v in d.items() if k in known})


def regenerate_positions(force: bool = False) -> bool:
    x_path    = RESULTS_DIR / "x_points.npy"
    idcs_path = RESULTS_DIR / "idcs_airfoil.npy"

    if not force and x_path.exists() and idcs_path.exists():
        return False

    print("Regenerating x_points.npy / idcs_airfoil.npy from training config...")
    load_raw_sample, load_split, preprocess_samples, RunConfig = _import_project()

    cfg = _load_run_config()
    if cfg is None:
        cfg = RunConfig()

    _, test_samples, _, test_files = load_split(
        data_dir=DATA_DIR,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
    )

    if cfg.max_test_files is not None:
        test_samples = test_samples[: cfg.max_test_files]
        test_files   = test_files  [: cfg.max_test_files]

    test_samples, _, _ = preprocess_samples(
        test_samples, cfg, "test", cache_dir=None
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x_obj    = np.empty(len(test_samples), dtype=object)
    idcs_obj = np.empty(len(test_samples), dtype=object)
    for i, s in enumerate(test_samples):
        x_obj[i]    = s["x"].astype(np.float32)
        idcs_obj[i] = s["idcs_airfoil"].astype(np.int32)

    np.save(x_path,    x_obj,    allow_pickle=True)
    np.save(idcs_path, idcs_obj, allow_pickle=True)

    print(f"  Written {len(test_samples)} samples to {RESULTS_DIR}")
    print(f"  Sample 0: {x_obj[0].shape[0]} points")
    return True


# ===========================================================================
# Per-component loss breakdown
# ===========================================================================

def component_losses(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
) -> Dict[str, float]:
    yt = y_true_flat.reshape(-1, 5, 3)
    yp = y_pred_flat.reshape(-1, 5, 3)

    def _rel_l2(a, b):
        return float(np.sqrt(np.sum((b - a) ** 2) / (np.sum(a ** 2) + 1e-12)))

    def _mae(a, b):
        return float(np.mean(np.abs(b - a)))

    losses = {}
    for name, col in COMPONENT_COLS.items():
        losses[f"{name}_rel_l2"] = _rel_l2(yt[..., col], yp[..., col])
        losses[f"{name}_mae"]    = _mae(yt[..., col], yp[..., col])

    gt_speed = np.linalg.norm(yt, axis=-1)
    pr_speed = np.linalg.norm(yp, axis=-1)
    losses["speed_rel_l2"] = _rel_l2(gt_speed, pr_speed)
    losses["speed_mae"]    = _mae(gt_speed, pr_speed)
    losses["total_rel_l2"] = _rel_l2(y_true_flat, y_pred_flat)
    return losses


def dataset_component_losses(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
) -> Dict[str, np.ndarray]:
    keys = None
    rows = []
    for yt, yp in zip(y_true_list, y_pred_list):
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
# W&B logging helpers
# ===========================================================================

def build_wandb_log_dict(
    epoch:              int,
    train_loss:         float,
    test_loss:          float,
    train_components:   Dict[str, float],
    test_components:    Dict[str, float],
    lr:                 float,
    tf_k:               int,
    grad_norm_mean:     Optional[float] = None,
    nan_count:          Optional[int]   = None,
    full_mesh_metrics:  Optional[Dict]  = None,
) -> Dict:
    log: Dict = {
        "epoch":        epoch,
        "train/loss":   train_loss,
        "test/loss":    test_loss,
        "train/ux":     train_components.get("ux_rel_l2",    float("nan")),
        "train/uy":     train_components.get("uy_rel_l2",    float("nan")),
        "train/uz":     train_components.get("uz_rel_l2",    float("nan")),
        "train/speed":  train_components.get("speed_rel_l2", float("nan")),
        "test/ux":      test_components.get("ux_rel_l2",    float("nan")),
        "test/uy":      test_components.get("uy_rel_l2",    float("nan")),
        "test/uz":      test_components.get("uz_rel_l2",    float("nan")),
        "test/speed":   test_components.get("speed_rel_l2", float("nan")),
        "optim/lr":     lr,
        "optim/tf_k":   tf_k,
    }
    if grad_norm_mean is not None:
        log["optim/grad_norm"] = grad_norm_mean
    if nan_count is not None:
        log["optim/nan_count"] = nan_count
    if full_mesh_metrics is not None:
        log["full_mesh/loss"]  = full_mesh_metrics["relative_L2_mean"]
        log["full_mesh/nrmse"] = full_mesh_metrics["nRMSE_mean"]
    return log


def log_final_to_wandb(
    wandb_run,
    final_metrics:     Dict,
    final_comp_losses: Dict[str, np.ndarray],
    test_files:        List[Path],
) -> None:
    import wandb
    wandb_run.log({
        "final/loss":  final_metrics["relative_L2_mean"],
        "final/nrmse": final_metrics["nRMSE_mean"],
        "final/ux":    float(final_comp_losses["ux_rel_l2"].mean()),
        "final/uy":    float(final_comp_losses["uy_rel_l2"].mean()),
        "final/uz":    float(final_comp_losses["uz_rel_l2"].mean()),
        "final/speed": float(final_comp_losses["speed_rel_l2"].mean()),
    })
    B    = len(next(iter(final_comp_losses.values())))
    cols = ["sample_idx", "file", "total_rel_l2",
            "ux_rel_l2", "uy_rel_l2", "uz_rel_l2", "speed_rel_l2"]
    table = wandb.Table(columns=cols)
    for i in range(B):
        table.add_data(
            i,
            test_files[i].name if test_files else str(i),
            float(final_comp_losses["total_rel_l2"][i]),
            float(final_comp_losses["ux_rel_l2"][i]),
            float(final_comp_losses["uy_rel_l2"][i]),
            float(final_comp_losses["uz_rel_l2"][i]),
            float(final_comp_losses["speed_rel_l2"][i]),
        )
    wandb_run.log({"per_sample_table": table})


# ===========================================================================
# Data loading
# ===========================================================================

def _check_positions_consistent(y_true, x_saved) -> bool:
    for yt, xs in zip(y_true, x_saved):
        if np.asarray(yt).shape[0] != np.asarray(xs).shape[0]:
            return False
    return True


def load_once(force_regen_positions: bool = False):
    load_raw_sample, load_split, preprocess_samples, RunConfig = _import_project()

    y_true_raw = np.load(RESULTS_DIR / "y_true.npy", allow_pickle=True)
    y_pred_raw = np.load(RESULTS_DIR / "y_pred.npy", allow_pickle=True)

    def _to_list(arr):
        if arr.dtype == object:
            return [np.asarray(a, dtype=np.float32) for a in arr.tolist()]
        return [arr[i].astype(np.float32) for i in range(arr.shape[0])]

    y_true = _to_list(y_true_raw)
    y_pred = _to_list(y_pred_raw)

    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} samples, y_pred has {len(y_pred)}.")
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt.shape != yp.shape:
            raise ValueError(f"Shape mismatch at sample {i}: {yt.shape} vs {yp.shape}")
        if yt.ndim != 2 or yt.shape[-1] != 15:
            raise ValueError(f"Expected (N,15) per sample, got {yt.shape} at sample {i}")

    B = len(y_true)

    cfg = _load_run_config()
    if cfg is None:
        cfg = RunConfig()

    _, test_samples, _, test_files = load_split(
        data_dir=DATA_DIR, test_frac=cfg.test_frac, seed=cfg.seed
    )
    if cfg.max_test_files is not None:
        test_samples = test_samples[: cfg.max_test_files]
        test_files   = test_files  [: cfg.max_test_files]

    if B > len(test_samples):
        raise ValueError(
            f"Predictions have {B} samples but split has only {len(test_samples)}."
        )
    test_samples = test_samples[:B]
    test_files   = test_files  [:B]

    x_path    = RESULTS_DIR / "x_points.npy"
    idcs_path = RESULTS_DIR / "idcs_airfoil.npy"
    need_regen = force_regen_positions

    if not need_regen:
        if not x_path.exists() or not idcs_path.exists():
            print("x_points.npy / idcs_airfoil.npy missing — regenerating...")
            need_regen = True
        else:
            x_saved = np.load(x_path, allow_pickle=True)
            if not _check_positions_consistent(y_true, x_saved):
                print("WARNING: saved x_points node counts do not match y_true — regenerating...")
                need_regen = True

    if need_regen:
        regenerate_positions(force=True)

    x_saved    = np.load(x_path,    allow_pickle=True)
    idcs_saved = np.load(idcs_path, allow_pickle=True)

    for i in range(B):
        test_samples[i] = dict(test_samples[i])
        test_samples[i]["x"]            = np.asarray(x_saved[i],    dtype=np.float32)
        test_samples[i]["idcs_airfoil"] = np.asarray(idcs_saved[i], dtype=np.int32)

    raw_samples = [load_raw_sample(fp) for fp in test_files]
    return y_true, y_pred, test_samples, test_files, raw_samples


def load_training_history() -> Optional[dict]:
    """Load training_history.json if it exists."""
    p = MODEL_DIR / "training_history.json"
    if not p.exists():
        # Try per-run layout
        candidates = sorted(MODEL_DIR.glob("run_*/training_history.json"))
        if not candidates:
            return None
        p = candidates[-1]
    with open(p) as f:
        return json.load(f)


# ===========================================================================
# Geometry helpers
# ===========================================================================

def flat15_to_t53(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(*arr.shape[:-1], 5, 3)


def speed_mag(u: np.ndarray) -> np.ndarray:
    return np.linalg.norm(u, axis=-1)


def rel_l2_sample(y_true_flat: np.ndarray, y_pred_flat: np.ndarray) -> float:
    num = np.sum((y_pred_flat - y_true_flat) ** 2)
    den = np.sum(y_true_flat ** 2) + 1e-12
    return float(np.sqrt(num / den))


def field_rel_l2(gt_vals: np.ndarray, pr_vals: np.ndarray) -> float:
    num = np.sum((pr_vals - gt_vals) ** 2)
    den = np.sum(gt_vals ** 2) + 1e-12
    return float(np.sqrt(num / den))


def frame_fields(u: np.ndarray) -> Dict[str, np.ndarray]:
    return {"|u|": speed_mag(u), "ux": u[:, 0], "uy": u[:, 1], "uz": u[:, 2]}


def symmetric_error_limits(
    values: np.ndarray, q: float = 0.99, eps: float = 1e-8
) -> Tuple[float, float]:
    vmax = float(np.quantile(np.abs(values), q))
    return -(max(vmax, eps)), max(vmax, eps)


def get_sample(y, idx: int) -> np.ndarray:
    return y[idx]


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
    inside = (Delaunay(pts).find_simplex(
        np.column_stack([XI.ravel(), YI.ravel()])
    ) >= 0).reshape(XI.shape)
    if sigma and sigma > 0:
        Z = np.where(inside, gaussian_filter(Z, sigma=sigma), np.nan)
    else:
        Z = np.where(inside, Z, np.nan)
    return XI, YI, Z


def pressure_out_frame(raw_sample: dict, frame_idx: int) -> np.ndarray:
    return raw_sample["pressure"].astype(np.float32)[frame_idx + 5]


def align_sample_to_positions(
    field_flat: np.ndarray, pos: np.ndarray, raw_sample: dict
) -> np.ndarray:
    if field_flat.shape[0] == pos.shape[0]:
        return field_flat
    n = min(field_flat.shape[0], pos.shape[0])
    if abs(field_flat.shape[0] - pos.shape[0]) <= 1:
        return field_flat[:n]
    raise ValueError(
        f"Point count mismatch: field={field_flat.shape[0]}, pos={pos.shape[0]}. "
        "Run with --regen-positions to fix."
    )


def build_aligned_sample(
    sample_idx: int, y_true, y_pred, test_samples, raw_samples
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos  = test_samples[sample_idx]["x"]
    yt_s = np.asarray(get_sample(y_true, sample_idx), dtype=np.float32)
    yp_s = np.asarray(get_sample(y_pred, sample_idx), dtype=np.float32)
    raw  = raw_samples[sample_idx]
    yt_s = align_sample_to_positions(yt_s, pos, raw)
    yp_s = align_sample_to_positions(yp_s, pos, raw)
    n    = min(pos.shape[0], yt_s.shape[0], yp_s.shape[0])
    return pos[:n], yt_s[:n], yp_s[:n]


# ===========================================================================
# ── NEW PLOTS FROM FNO ──────────────────────────────────────────────────────
# ===========================================================================

def plot_training_curves(
    history: dict,
    out_dir: Path,
    final_point_relL2: Optional[float] = None,
    baseline_relL2: Optional[float] = None,
) -> Path:
    """
    Three-panel training diagnostic matching the attached style:
      Left   — train/val relative-L2 curves
      Centre — learning-rate schedule
      Right  — time per epoch

    Supports both the newer Transolver history keys and the older generic ones.
    """

    # Prefer Transolver-specific keys if present
    if "train_subsampled_relative_l2" in history:
        train_key = "train_subsampled_relative_l2"
    elif "train_loss" in history:
        train_key = "train_loss"
    elif "loss" in history:
        train_key = "loss"
    else:
        raise ValueError("No recognised training-loss key found in history.")

    if "val_subsampled_relative_l2" in history:
        val_key = "val_subsampled_relative_l2"
    elif "test_loss" in history:
        val_key = "test_loss"
    elif "val_loss" in history:
        val_key = "val_loss"
    else:
        val_key = None

    epochs = np.arange(len(history[train_key]))

    has_val  = val_key is not None and len(history[val_key]) == len(epochs)
    has_lr   = "lr" in history and len(history["lr"]) == len(epochs)
    has_time = "time" in history and len(history["time"]) == len(epochs)

    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), dpi=FIG_DPI)

    # ── left: training curves ───────────────────────────────────────────────
    axs[0].plot(epochs, history[train_key], label="train (subsampled)")
    if has_val:
        axs[0].plot(epochs, history[val_key], label="val (subsampled)")

    if final_point_relL2 is not None:
        axs[0].axhline(
            final_point_relL2,
            color="tab:red",
            ls="--",
            label=f"point relL2={final_point_relL2:.4f}",
        )

    if baseline_relL2 is not None:
        axs[0].axhline(
            baseline_relL2,
            color="gray",
            ls=":",
            label=f"persistence baseline={baseline_relL2:.4f}",
        )

    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("relative L2")
    axs[0].set_title(
        "Training curves\n"
        "(subsampled training metric ≠ final point-level metric)"
    )
    axs[0].legend(fontsize=8)

    # ── middle: LR schedule ─────────────────────────────────────────────────
    if has_lr:
        axs[1].plot(epochs, history["lr"])
        axs[1].set_xlabel("epoch")
        axs[1].set_ylabel("learning rate")
        axs[1].set_title("LR schedule")
    else:
        axs[1].set_visible(False)

    # ── right: time per epoch ───────────────────────────────────────────────
    if has_time:
        axs[2].plot(epochs, history["time"])
        axs[2].set_xlabel("epoch")
        axs[2].set_ylabel("seconds")
        axs[2].set_title("Time per epoch")
    else:
        axs[2].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_sample_relL2_histogram(
    y_true: List[np.ndarray],
    y_pred: List[np.ndarray],
    out_dir: Path,
    baseline_relL2: Optional[float] = None,
) -> Path:
    """
    Histogram of per-sample relative L2 across the test set.
    Reveals whether errors are uniformly distributed or driven by a few
    outlier geometries.  A vertical line marks the persistence baseline
    when provided.
    """
    rels = np.array([rel_l2_sample(yt, yp) for yt, yp in zip(y_true, y_pred)])

    fig, ax = plt.subplots(figsize=(8, 4), dpi=FIG_DPI)
    ax.hist(rels, bins=max(10, len(rels) // 5), color="#4C72B0",
            edgecolor="white", alpha=0.85)
    ax.axvline(rels.mean(), color="tab:red", ls="--", lw=1.6,
               label=f"mean={rels.mean():.4f}")
    ax.axvline(float(np.median(rels)), color="tab:orange", ls=":", lw=1.6,
               label=f"median={np.median(rels):.4f}")
    if baseline_relL2 is not None:
        ax.axvline(baseline_relL2, color="gray", ls="-.", lw=1.4,
                   label=f"persistence={baseline_relL2:.4f}")
    ax.set_xlabel("Relative L2")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-sample relL2 distribution  (N={len(rels)})")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "per_sample_relL2_histogram.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_error_vs_frame(
    y_true: List[np.ndarray],
    y_pred: List[np.ndarray],
    out_dir: Path,
) -> Path:
    """
    Shows how prediction error accumulates across the 5 output frames.
    Each line is one test sample (light); the bold line is the mean.
    A rising trend indicates error accumulation (autoregressive drift).
    A flat trend indicates the model predicts each frame independently.
    """
    B = len(y_true)
    per_frame = np.zeros((B, 5), dtype=np.float32)
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        yt5 = flat15_to_t53(yt)   # (N, 5, 3)
        yp5 = flat15_to_t53(yp)
        for t in range(5):
            per_frame[i, t] = field_rel_l2(
                yt5[:, t, :].ravel(), yp5[:, t, :].ravel()
            )

    frames = np.arange(5)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=FIG_DPI)
    for i in range(B):
        ax.plot(frames, per_frame[i], color="#4C72B0", alpha=0.15, lw=0.8)
    ax.plot(frames, per_frame.mean(axis=0), color="tab:red",
            lw=2.5, marker="o", label="mean")
    ax.fill_between(
        frames,
        per_frame.mean(axis=0) - per_frame.std(axis=0),
        per_frame.mean(axis=0) + per_frame.std(axis=0),
        alpha=0.2, color="tab:red", label="±1 std",
    )
    ax.set_xlabel("Output frame")
    ax.set_ylabel("Relative L2")
    ax.set_title("Prediction error vs output frame\n(rising = autoregressive drift)")
    ax.set_xticks(frames)
    ax.set_xticklabels([f"t={t}" for t in frames])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "error_vs_frame.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_worst_best_samples(
    y_true: List[np.ndarray],
    y_pred: List[np.ndarray],
    test_samples: list,
    raw_samples:  list,
    test_files:   List[Path],
    out_dir:      Path,
    frame_idx:    int = 0,
    n_each:       int = 3,
) -> Path:
    """
    Side-by-side scatter panels for the N best and N worst samples ranked
    by overall relative L2.  Each panel shows the ux cross-section so the
    visual quality difference between best and worst is immediately apparent.
    Uses scatter-only (no interpolation) for speed.
    """
    rels   = np.array([rel_l2_sample(yt, yp) for yt, yp in zip(y_true, y_pred)])
    order  = np.argsort(rels)
    best   = order[:n_each]
    worst  = order[-n_each:][::-1]
    groups = [("Best", best, "#2ca02c"), ("Worst", worst, "#d62728")]

    fig, axs = plt.subplots(
        2 * n_each, 3, figsize=(15, 4.5 * 2 * n_each), dpi=FIG_DPI
    )

    row = 0
    for group_label, indices, colour in groups:
        for sidx in indices:
            pos, yt_s, yp_s = build_aligned_sample(
                sidx, y_true, y_pred, test_samples, raw_samples
            )
            yt = flat15_to_t53(yt_s)[:, frame_idx, :]
            yp = flat15_to_t53(yp_s)[:, frame_idx, :]

            mask, target, hw = choose_section_mask(
                pos, SECTION_AXIS, SECTION_QUANTILE, SECTION_BAND_FRAC
            )
            ai, aj, labels = projected_axes(SECTION_AXIS)
            px, py_ax = pos[mask, ai], pos[mask, aj]

            gt_v = yt[mask, 0]   # ux
            pr_v = yp[mask, 0]
            er_v = gt_v - pr_v
            vmin, vmax = min(gt_v.min(), pr_v.min()), max(gt_v.max(), pr_v.max())
            emin, emax = symmetric_error_limits(er_v)
            rel = field_rel_l2(gt_v, pr_v)

            title_prefix = f"[{group_label}] s{sidx} relL2={rels[sidx]:.4f}"

            sc0 = axs[row, 0].scatter(px, py_ax, c=gt_v, s=3,
                                      cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
            axs[row, 0].set_title(f"GT ux | {title_prefix}", fontsize=8)
            fig.colorbar(sc0, ax=axs[row, 0], shrink=0.8)

            sc1 = axs[row, 1].scatter(px, py_ax, c=pr_v, s=3,
                                      cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
            axs[row, 1].set_title(f"Pred ux | {title_prefix}", fontsize=8)
            fig.colorbar(sc1, ax=axs[row, 1], shrink=0.8)

            sc2 = axs[row, 2].scatter(px, py_ax, c=er_v, s=3,
                                      cmap=CMAP_ERR, vmin=emin, vmax=emax)
            axs[row, 2].set_title(f"Error ux | relL2={rel:.4f}", fontsize=8)
            fig.colorbar(sc2, ax=axs[row, 2], shrink=0.8)

            for c in range(3):
                axs[row, c].set_xlabel(labels[0])
                axs[row, c].set_ylabel(labels[1])
                axs[row, c].set_aspect("equal", adjustable="box")
                axs[row, c].spines["bottom"].set_color(colour)
                axs[row, c].spines["top"].set_color(colour)
                axs[row, c].spines["left"].set_color(colour)
                axs[row, c].spines["right"].set_color(colour)
                for spine in axs[row, c].spines.values():
                    spine.set_linewidth(2.5)

            row += 1

    fig.suptitle(
        f"Best / Worst {n_each} samples by overall relL2 | frame {frame_idx} | ux",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = out_dir / f"worst_best_samples_frame{frame_idx}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_speed_magnitude_profile(
    y_true:       List[np.ndarray],
    y_pred:       List[np.ndarray],
    test_samples: list,
    raw_samples:  list,
    sample_indices: List[int],
    out_dir:      Path,
    frame_idx:    int = 0,
) -> Path:
    """
    Spanwise mean-speed profile: for each sample, bin points along the
    section axis and plot mean |u| GT vs Pred.  Reveals systematic
    over/under-prediction across the span.
    """
    n_samples = len(sample_indices)
    fig, axs  = plt.subplots(
        1, n_samples, figsize=(5 * n_samples, 4.5), dpi=FIG_DPI, sharey=False
    )
    if n_samples == 1:
        axs = [axs]

    axis_col = {"x": 0, "y": 1, "z": 2}[SECTION_AXIS]

    for ax, sidx in zip(axs, sample_indices):
        pos, yt_s, yp_s = build_aligned_sample(
            sidx, y_true, y_pred, test_samples, raw_samples
        )
        yt = flat15_to_t53(yt_s)[:, frame_idx, :]   # (N, 3)
        yp = flat15_to_t53(yp_s)[:, frame_idx, :]

        gt_spd = np.linalg.norm(yt, axis=-1)
        pr_spd = np.linalg.norm(yp, axis=-1)
        coord  = pos[:, axis_col]

        n_bins = 40
        edges  = np.linspace(coord.min(), coord.max(), n_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        gt_mean = np.array([
            gt_spd[(coord >= edges[k]) & (coord < edges[k + 1])].mean()
            if ((coord >= edges[k]) & (coord < edges[k + 1])).any() else np.nan
            for k in range(n_bins)
        ])
        pr_mean = np.array([
            pr_spd[(coord >= edges[k]) & (coord < edges[k + 1])].mean()
            if ((coord >= edges[k]) & (coord < edges[k + 1])).any() else np.nan
            for k in range(n_bins)
        ])

        ax.plot(centres, gt_mean, label="GT",   lw=2.0, color="tab:blue")
        ax.plot(centres, pr_mean, label="Pred", lw=2.0, color="tab:red",
                ls="--")
        ax.fill_between(centres, gt_mean, pr_mean, alpha=0.15, color="gray")
        ax.set_xlabel(f"{SECTION_AXIS} coordinate")
        ax.set_ylabel("Mean |u|")
        ax.set_title(f"Sample {sidx} | frame {frame_idx}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Spanwise mean-speed profile  (axis={SECTION_AXIS}, frame={frame_idx})"
    )
    fig.tight_layout()
    out_path = out_dir / f"speed_magnitude_profile_frame{frame_idx}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_frame_component_heatmap(
    y_true:  List[np.ndarray],
    y_pred:  List[np.ndarray],
    out_dir: Path,
) -> Path:
    """
    Heatmap of mean relative L2 across the test set, with axes:
      rows    — output frame  (t=0 … t=4)
      columns — velocity component  (ux, uy, uz, |u|)

    Immediately shows which (frame, component) combinations are hardest.
    """
    comp_names = ["ux", "uy", "uz", "|u|"]
    n_frames   = 5
    B          = len(y_true)
    grid       = np.zeros((n_frames, len(comp_names)), dtype=np.float32)

    for yt, yp in zip(y_true, y_pred):
        yt5 = flat15_to_t53(yt)   # (N, 5, 3)
        yp5 = flat15_to_t53(yp)
        for t in range(n_frames):
            for ci, cname in enumerate(comp_names):
                if cname == "|u|":
                    gv = np.linalg.norm(yt5[:, t, :], axis=-1)
                    pv = np.linalg.norm(yp5[:, t, :], axis=-1)
                else:
                    col = COMPONENT_COLS[cname]
                    gv  = yt5[:, t, col]
                    pv  = yp5[:, t, col]
                grid[t, ci] += field_rel_l2(gv, pv)

    grid /= B   # mean over samples

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=FIG_DPI)
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean relative L2")

    ax.set_xticks(range(len(comp_names)))
    ax.set_xticklabels(comp_names)
    ax.set_yticks(range(n_frames))
    ax.set_yticklabels([f"t={t}" for t in range(n_frames)])
    ax.set_xlabel("Velocity component")
    ax.set_ylabel("Output frame")
    ax.set_title("Mean relL2 heatmap: frame × component")

    # Annotate cells
    for t in range(n_frames):
        for ci in range(len(comp_names)):
            ax.text(
                ci, t, f"{grid[t, ci]:.3f}",
                ha="center", va="center", fontsize=8,
                color="white" if grid[t, ci] > grid.max() * 0.6 else "black",
            )

    fig.tight_layout()
    out_path = out_dir / "per_frame_component_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# Existing plots (unchanged logic, boxplot uses _boxplot shim)
# ===========================================================================

def plot_component_loss_breakdown(
    sample_idx: int, y_true, y_pred, out_dir: Path
) -> Path:
    yt_s = np.asarray(get_sample(y_true, sample_idx), dtype=np.float32)
    yp_s = np.asarray(get_sample(y_pred, sample_idx), dtype=np.float32)
    yt   = yt_s.reshape(-1, 5, 3)
    yp   = yp_s.reshape(-1, 5, 3)

    components = ["ux", "uy", "uz", "|u|"]
    n_frames   = 5
    x          = np.arange(n_frames)
    width      = 0.2

    fig, ax = plt.subplots(figsize=(10, 4), dpi=FIG_DPI)
    for ci, comp in enumerate(components):
        losses = []
        for t in range(n_frames):
            if comp == "|u|":
                gt_v = np.linalg.norm(yt[:, t, :], axis=-1)
                pr_v = np.linalg.norm(yp[:, t, :], axis=-1)
            else:
                col  = COMPONENT_COLS[comp]
                gt_v = yt[:, t, col]
                pr_v = yp[:, t, col]
            losses.append(field_rel_l2(gt_v, pr_v))
        ax.bar(x + ci * width, losses, width, label=comp)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Relative L2")
    ax.set_title(f"Sample {sample_idx} — per-component relative L2 by frame")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"t={t}" for t in range(n_frames)])
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / f"sample{sample_idx:03d}_component_losses.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dataset_component_losses(
    losses_dict: Dict[str, np.ndarray], out_dir: Path
) -> Path:
    keys_to_plot = ["ux_rel_l2", "uy_rel_l2", "uz_rel_l2", "speed_rel_l2"]
    tick_labels  = ["ux", "uy", "uz", "|u|"]
    data         = [losses_dict[k] for k in keys_to_plot if k in losses_dict]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=FIG_DPI)
    bp = _boxplot(ax, data, tick_labels[:len(data)], patch_artist=True)
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    ax.set_ylabel("Relative L2")
    ax.set_title("Per-component loss distribution across test set")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()

    out_path = out_dir / "dataset_component_loss_boxplot.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _contour_row(ax_gt, ax_pr, ax_er, px, py, gt_vals, pr_vals, name, fig,
                 cmap_main=CMAP_MAIN, cmap_err=CMAP_ERR, labels=("x", "z")):
    rel     = field_rel_l2(gt_vals, pr_vals)
    er_vals = gt_vals - pr_vals

    XI, YI, Zg = interpolate_masked(px, py, gt_vals, GRID_NX, GRID_NY, GAUSS_SIGMA)
    _,  _,  Zp = interpolate_masked(px, py, pr_vals, GRID_NX, GRID_NY, GAUSS_SIGMA)
    _,  _,  Ze = interpolate_masked(px, py, er_vals, GRID_NX, GRID_NY, GAUSS_SIGMA)

    vmin   = np.nanmin([Zg, Zp])
    vmax   = np.nanmax([Zg, Zp])
    levels = np.linspace(vmin, vmax, 50)
    emin, emax = symmetric_error_limits(np.nan_to_num(Ze[~np.isnan(Ze)], nan=0.0))
    elevels = np.linspace(emin, emax, 50)

    cf0 = ax_gt.contourf(XI, YI, Zg, levels=levels, cmap=cmap_main, alpha=0.95, extend="both")
    ax_gt.scatter(px, py, c=gt_vals, s=2, cmap=cmap_main, vmin=vmin, vmax=vmax, alpha=0.3)
    ax_gt.set_title(f"GT {name}"); fig.colorbar(cf0, ax=ax_gt, shrink=0.82)

    cf1 = ax_pr.contourf(XI, YI, Zp, levels=levels, cmap=cmap_main, alpha=0.95, extend="both")
    ax_pr.scatter(px, py, c=pr_vals, s=2, cmap=cmap_main, vmin=vmin, vmax=vmax, alpha=0.3)
    ax_pr.set_title(f"Pred {name}"); fig.colorbar(cf1, ax=ax_pr, shrink=0.82)

    cf2 = ax_er.contourf(XI, YI, Ze, levels=elevels, cmap=cmap_err, alpha=0.95, extend="both")
    ax_er.scatter(px, py, c=er_vals, s=2, cmap=cmap_err, vmin=emin, vmax=emax, alpha=0.3)
    ax_er.set_title(f"Error {name} | relL2={rel:.4f}"); fig.colorbar(cf2, ax=ax_er, shrink=0.82)

    for ax in [ax_gt, ax_pr, ax_er]:
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        ax.set_aspect("equal", adjustable="box")
    return rel


def plot_2d_contour_scatter_panel(
    sample_idx, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir
) -> Path:
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples)
    yt = flat15_to_t53(yt_s)[:, frame_idx, :]
    yp = flat15_to_t53(yp_s)[:, frame_idx, :]
    gt_fields = frame_fields(yt); pr_fields = frame_fields(yp)

    mask, target, hw = choose_section_mask(pos, SECTION_AXIS, SECTION_QUANTILE, SECTION_BAND_FRAC)
    i, j, labels = projected_axes(SECTION_AXIS)
    px, py_ax = pos[mask, i], pos[mask, j]

    fig, axs = plt.subplots(3, 4, figsize=(20, 13), dpi=FIG_DPI)
    for c, name in enumerate(FIELD_NAMES):
        _contour_row(axs[0,c], axs[1,c], axs[2,c], px, py_ax,
                     gt_fields[name][mask], pr_fields[name][mask], name, fig, labels=labels)
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
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples)
    yt = flat15_to_t53(yt_s)[:, frame_idx, :]
    yp = flat15_to_t53(yp_s)[:, frame_idx, :]
    gt_fields = frame_fields(yt); pr_fields = frame_fields(yp)

    mask, target, hw = choose_section_mask(pos, SECTION_AXIS, SECTION_QUANTILE, SECTION_BAND_FRAC)
    i, j, labels = projected_axes(SECTION_AXIS)
    px, py_ax = pos[mask, i], pos[mask, j]

    fig, axs = plt.subplots(3, 4, figsize=(20, 13), dpi=FIG_DPI)
    for c, name in enumerate(FIELD_NAMES):
        gt_v = gt_fields[name][mask]; pr_v = pr_fields[name][mask]
        er_v = gt_v - pr_v; rel = field_rel_l2(gt_v, pr_v)
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


def plot_ux_summary_4x3(
    sample_indices, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir
) -> Path:
    rows = len(sample_indices)
    fig, axs = plt.subplots(rows, 3, figsize=(16, 4.0 * rows), dpi=FIG_DPI)
    if rows == 1:
        axs = axs[None, :]

    for r, sidx in enumerate(sample_indices):
        pos, yt_s, yp_s = build_aligned_sample(sidx, y_true, y_pred, test_samples, raw_samples)
        yt = flat15_to_t53(yt_s)[:, frame_idx, :]
        yp = flat15_to_t53(yp_s)[:, frame_idx, :]
        mask, target, hw = choose_section_mask(pos, SECTION_AXIS, SECTION_QUANTILE, SECTION_BAND_FRAC)
        i, j, labels = projected_axes(SECTION_AXIS)
        px, py_ax = pos[mask, i], pos[mask, j]
        _contour_row(axs[r,0], axs[r,1], axs[r,2], px, py_ax,
                     yt[mask, 0], yp[mask, 0],
                     f"ux (sample {sidx})", fig, labels=labels)

    fig.suptitle(f"ux comparison summary | frame {frame_idx} | section axis {SECTION_AXIS}")
    out_path = out_dir / f"ux_summary_frame{frame_idx}_4x3.png"
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def plot_3d_wing_surface(
    sample_idx, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir
) -> Optional[Path]:
    s   = test_samples[sample_idx]
    raw = raw_samples[sample_idx]
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples)
    idcs_air = s["idcs_airfoil"]
    yt  = flat15_to_t53(yt_s)[:, frame_idx, :]
    yp  = flat15_to_t53(yp_s)[:, frame_idx, :]
    rel = rel_l2_sample(yt_s, yp_s)

    p_full = pressure_out_frame(raw, frame_idx)
    p_surf = (p_full[idcs_air]
              if idcs_air.size > 0 and idcs_air.max() < p_full.shape[0]
              else np.zeros(idcs_air.size, dtype=np.float32))

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

    sm = plt.cm.ScalarMappable(cmap=CMAP_PRESS, norm=plt.Normalize(vmin=pmin, vmax=pmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label("Pressure (Pa) on wing")
    fig.suptitle(f"Sample {sample_idx} frame {frame_idx} | 3D wing | relL2={rel:.4f}")
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_3d_wing_surface.png"
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def _precompute_contour_cache(pos, yt, yp, mask, i, j):
    cache    = []
    lim_main = {n: [np.inf, -np.inf] for n in FIELD_NAMES}
    lim_err  = {n: [np.inf, -np.inf] for n in FIELD_NAMES}
    px, py_ax = pos[mask, i], pos[mask, j]
    for t in range(5):
        gt_f = frame_fields(yt[:, t, :]); pr_f = frame_fields(yp[:, t, :])
        fc   = {}
        for name in FIELD_NAMES:
            rel = field_rel_l2(gt_f[name], pr_f[name])
            gv, pv, ev = gt_f[name][mask], pr_f[name][mask], (gt_f[name] - pr_f[name])[mask]
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
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples)
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
            axs[0,c].set_title(f"GT {name}"); axs[1,c].set_title(f"Pred {name}")
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
    pos, yt_s, yp_s = build_aligned_sample(sample_idx, y_true, y_pred, test_samples, raw_samples)
    yt = flat15_to_t53(yt_s); yp = flat15_to_t53(yp_s)
    mask, target, hw = choose_section_mask(pos, SECTION_AXIS, SECTION_QUANTILE, SECTION_BAND_FRAC)
    i, j, labels = projected_axes(SECTION_AXIS)
    px, py_ax = pos[mask, i], pos[mask, j]

    cache    = []
    lim_main = {n: [np.inf, -np.inf] for n in FIELD_NAMES}
    lim_err  = {n: [np.inf, -np.inf] for n in FIELD_NAMES}
    for t in range(5):
        gt_f = frame_fields(yt[:, t, :]); pr_f = frame_fields(yp[:, t, :])
        fc   = {}
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
            vmin, vmax = lim_main[name]; emin, emax = err_sym[name]
            axs[0,c].scatter(px, py_ax, c=gv, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
            axs[1,c].scatter(px, py_ax, c=pv, s=4, cmap=CMAP_MAIN, vmin=vmin, vmax=vmax)
            axs[2,c].scatter(px, py_ax, c=ev, s=4, cmap=CMAP_ERR,  vmin=emin, vmax=emax)
            axs[0,c].set_title(f"GT {name}"); axs[1,c].set_title(f"Pred {name}")
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


def export_matlab_bundle(
    sample_indices, y_true, y_pred, test_samples, test_files, raw_samples, out_dir
) -> Path:
    bundle: Dict = {
        "sample_indices":    np.asarray(sample_indices, dtype=np.int32),
        "section_axis":      SECTION_AXIS,
        "section_quantile":  SECTION_QUANTILE,
        "section_band_frac": SECTION_BAND_FRAC,
    }
    for sidx in sample_indices:
        pos, yt_s, yp_s = build_aligned_sample(sidx, y_true, y_pred, test_samples, raw_samples)
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


def batch_report(y_true, y_pred, test_files, out_dir) -> Tuple[Path, Path]:
    B    = len(y_true)
    rels = np.array([rel_l2_sample(get_sample(y_true, i), get_sample(y_pred, i))
                     for i in range(B)])
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
        gt = flat15_to_t53(get_sample(y_true, i))
        pr = flat15_to_t53(get_sample(y_pred, i))
        gt_m = speed_mag(gt).mean(axis=0)
        pr_m = speed_mag(pr).mean(axis=0)
        er_m = speed_mag(pr - gt).mean(axis=0)
        axs[r,0].plot(gt_m, marker="o", label="GT mean|u|")
        axs[r,0].plot(pr_m, marker="s", label="Pred mean|u|")
        axs[r,0].set_title(f"sample {i} ({test_files[i].name})")
        axs[r,0].set_xlabel("frame"); axs[r,0].set_ylabel("mean |u|")
        axs[r,0].legend()
        axs[r,1].plot(er_m, marker="o", color="tab:red")
        axs[r,1].set_title(f"mean |Δu| per frame | relL2={rels[i]:.4f}")
        axs[r,1].set_xlabel("frame"); axs[r,1].set_ylabel("mean error")

    fig.tight_layout()
    png = out_dir / "batch_report.png"
    fig.savefig(png, bbox_inches="tight"); plt.close(fig)

    csv_path = out_dir / "batch_report_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_idx", "file", "rel_l2"])
        writer.writeheader()
        for i in range(B):
            writer.writerow({"sample_idx": i, "file": test_files[i].name,
                             "rel_l2": f"{rels[i]:.8f}"})
    return png, csv_path


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="TransolverAR visualisation pipeline")
    parser.add_argument("--frame-idx",       type=int, default=0, choices=range(5))
    parser.add_argument("--regen-positions", action="store_true")
    parser.add_argument("--no-animations",   action="store_true")
    parser.add_argument("--no-3d",           action="store_true")
    parser.add_argument("--no-training-curves", action="store_true",
                        help="Skip training curves (if training_history.json is absent).")
    parser.add_argument("--wandb-run-id",    type=str, default=None)
    args = parser.parse_args()

    # ── output directories ───────────────────────────────────────────────────
    single_dir = OUT_DIR / "single"
    anim_dir   = OUT_DIR / "anim"
    report_dir = OUT_DIR / "report"
    matlab_dir = OUT_DIR / "matlab"
    comp_dir   = OUT_DIR / "component_losses"
    diag_dir   = OUT_DIR / "diagnostics"
    for d in [single_dir, anim_dir, report_dir, matlab_dir, comp_dir, diag_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────────
    print("Loading predictions and test samples...")
    y_true, y_pred, test_samples, test_files, raw_samples = load_once(
        force_regen_positions=args.regen_positions
    )
    B = len(y_true)
    print(f"  {B} test samples loaded.\n")

    chosen_samples = [i for i in SINGLE_SAMPLE_INDICES if i < B]
    if not chosen_samples:
        raise ValueError(
            f"None of SINGLE_SAMPLE_INDICES={SINGLE_SAMPLE_INDICES} are valid "
            f"for {B} test samples."
        )

    # ── training curves ──────────────────────────────────────────────────────
    if not args.no_training_curves:
        history = load_training_history()
        if history is not None:
            # Compute mean relL2 across test set for overlay
            rels = np.array([rel_l2_sample(yt, yp) for yt, yp in zip(y_true, y_pred)])
            tc_path = plot_training_curves(
                history, diag_dir,
                final_point_relL2=float(rels.mean()),
            )
            print(f"Training curves → {tc_path}")
        else:
            print("training_history.json not found — skipping training curves.")

    # ── per-component losses ─────────────────────────────────────────────────
    print("Computing per-component losses...")
    comp_losses = dataset_component_losses(y_true, y_pred)

    print(f"\n  {'Metric':<20} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*42}")
    for k, v in comp_losses.items():
        print(f"  {k:<20} {v.mean():>10.4f} {v.std():>10.4f}")
    print()

    comp_csv = comp_dir / "component_losses_per_sample.csv"
    save_component_loss_csv(comp_losses, comp_csv,
                            file_names=[f.name for f in test_files])
    print(f"  Component losses CSV: {comp_csv}")

    bp_path = plot_dataset_component_losses(comp_losses, comp_dir)
    print(f"  Component loss box plot: {bp_path}")

    # ── new diagnostic plots ─────────────────────────────────────────────────
    print("Generating diagnostic plots...")

    hist_path = plot_per_sample_relL2_histogram(y_true, y_pred, diag_dir)
    print(f"  relL2 histogram: {hist_path}")

    evf_path = plot_error_vs_frame(y_true, y_pred, diag_dir)
    print(f"  Error vs frame: {evf_path}")

    hm_path = plot_per_frame_component_heatmap(y_true, y_pred, diag_dir)
    print(f"  Frame×component heatmap: {hm_path}")

    wb_path = plot_worst_best_samples(
        y_true, y_pred, test_samples, raw_samples, test_files,
        diag_dir, frame_idx=args.frame_idx, n_each=3,
    )
    print(f"  Worst/best samples: {wb_path}")

    sp_path = plot_speed_magnitude_profile(
        y_true, y_pred, test_samples, raw_samples,
        chosen_samples, diag_dir, frame_idx=args.frame_idx,
    )
    print(f"  Speed magnitude profiles: {sp_path}")

    # ── optional W&B logging ─────────────────────────────────────────────────
    if args.wandb_run_id is not None:
        try:
            import wandb
            api = wandb.Api()
            run = api.run(args.wandb_run_id)
            for k, v in comp_losses.items():
                run.summary[f"post_eval/components/{k}"] = float(v.mean())
            run.summary.update()
            print(f"  Component losses logged to W&B run {args.wandb_run_id}")
        except Exception as e:
            print(f"  W&B logging skipped ({e})")

    # ── static plots ─────────────────────────────────────────────────────────
    print("\nGenerating static plots...")
    for sidx in chosen_samples:
        sample_dir = single_dir / f"{sidx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        plot_2d_contour_scatter_panel(sidx, args.frame_idx, y_true, y_pred,
                                      test_samples, raw_samples, sample_dir)
        plot_2d_scatter_only_panel(sidx, args.frame_idx, y_true, y_pred,
                                   test_samples, raw_samples, sample_dir)
        if not args.no_3d:
            plot_3d_wing_surface(sidx, args.frame_idx, y_true, y_pred,
                                 test_samples, raw_samples, sample_dir)
        plot_component_loss_breakdown(sidx, y_true, y_pred, sample_dir)
        print(f"  Sample {sidx} done.")

    plot_ux_summary_4x3(chosen_samples, args.frame_idx, y_true, y_pred,
                        test_samples, raw_samples, single_dir)

    # ── animations ───────────────────────────────────────────────────────────
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
        chosen_samples, y_true, y_pred,
        test_samples, test_files, raw_samples, matlab_dir
    )

    # ── batch report ─────────────────────────────────────────────────────────
    print("Generating batch report...")
    preport, pcsv = batch_report(y_true, y_pred, test_files, report_dir)

    # ── summary ──────────────────────────────────────────────────────────────
    print("\nAll outputs saved:")
    print(f"  Static plots      : {single_dir}")
    print(f"  Animations        : {anim_dir}")
    print(f"  Component losses  : {comp_dir}")
    print(f"  Diagnostics       : {diag_dir}")
    print(f"  MATLAB bundle     : {matlab_path}")
    print(f"  Batch report      : {preport}")
    print(f"  Metrics CSV       : {pcsv}")
    print(f"  Component CSV     : {comp_csv}")


if __name__ == "__main__":
    main()
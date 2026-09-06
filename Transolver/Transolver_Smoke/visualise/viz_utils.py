import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay

from load_data import load_split, load_raw_sample


def load_predictions(results_dir: Path):
    y_true = np.load(results_dir / "y_true_points.npy").astype(np.float32)
    y_pred = np.load(results_dir / "y_pred_points.npy").astype(np.float32)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.ndim != 3 or y_true.shape[-1] != 15:
        raise ValueError(f"Expected (B,N,15), got {y_true.shape}")
    return y_true, y_pred


def load_test_samples_and_files(data_dir: Path, test_frac: float, seed: int):
    _, test_samples, _, test_files = load_split(data_dir=data_dir, test_frac=test_frac, seed=seed)
    return test_samples, test_files


def load_raw_npz_for_test_files(test_files):
    return [load_raw_sample(fp) for fp in test_files]


def flat15_to_t53(arr):
    # (...,15) -> (...,5,3)
    return arr.reshape(*arr.shape[:-1], 5, 3)


def speed_mag(u):
    # (...,3) -> (...)
    return np.linalg.norm(u, axis=-1)


def rel_l2_sample(y_true_flat, y_pred_flat):
    num = np.sum((y_pred_flat - y_true_flat) ** 2)
    den = np.sum(y_true_flat ** 2) + 1e-12
    return float(np.sqrt(num / den))


def choose_section_mask(pos, axis="y", quantile=0.5, band_frac=0.015):
    axis_map = {"x": 0, "y": 1, "z": 2}
    k = axis_map[axis]
    vals = pos[:, k]
    lo, hi = vals.min(), vals.max()
    target = np.quantile(vals, quantile)
    half_band = max((hi - lo) * band_frac, 1e-8)

    mask = np.abs(vals - target) <= half_band
    # relax if too few points
    if mask.sum() < 300:
        half_band *= 2.0
        mask = np.abs(vals - target) <= half_band
    if mask.sum() < 100:
        half_band *= 2.0
        mask = np.abs(vals - target) <= half_band

    return mask, target, half_band


def projected_axes(section_axis):
    if section_axis == "x":
        return 1, 2, ("y", "z")
    if section_axis == "y":
        return 0, 2, ("x", "z")
    if section_axis == "z":
        return 0, 1, ("x", "y")
    raise ValueError("section_axis must be x|y|z")


def interpolate_to_masked_grid(
    px, py, values, nx=300, ny=180, sigma=1.0
):
    """
    Robust interpolation:
    1) linear interpolation
    2) nearest fill for holes
    3) convex-hull mask to avoid unsupported extrapolation artifacts
    4) optional mild smoothing on valid region
    """
    xi = np.linspace(px.min(), px.max(), nx)
    yi = np.linspace(py.min(), py.max(), ny)
    XI, YI = np.meshgrid(xi, yi)

    pts = np.column_stack([px, py])

    # linear
    Z_lin = griddata(pts, values, (XI, YI), method="linear")
    # nearest fill
    Z_near = griddata(pts, values, (XI, YI), method="nearest")
    Z = np.where(np.isnan(Z_lin), Z_near, Z_lin)

    # convex hull mask (via Delaunay)
    tri = Delaunay(pts)
    inside = tri.find_simplex(np.column_stack([XI.ravel(), YI.ravel()])) >= 0
    inside = inside.reshape(XI.shape)

    # Optional smoothing but only inside mask
    if sigma is not None and sigma > 0:
        Zs = gaussian_filter(Z, sigma=sigma)
        Z = np.where(inside, Zs, np.nan)
    else:
        Z = np.where(inside, Z, np.nan)

    return XI, YI, Z, inside


def pressure_out_frame(raw_sample, frame_idx):
    """
    Confirmed mapping:
      velocity_out[k] <-> pressure[k+5]
    """
    p = raw_sample["pressure"].astype(np.float32)  # (10, N)
    return p[frame_idx + 5]                        # (N,)
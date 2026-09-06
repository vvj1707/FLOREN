import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from viz_config import (
    DATA_DIR, RESULTS_DIR, OUT_DIR, TEST_FRAC, SEED, FIG_DPI,
    CMAP_MAIN, CMAP_ERR, CMAP_PRESS,
    SECTION_AXIS, SECTION_QUANTILE, SECTION_BAND_FRAC,
    GRID_NX, GRID_NY, GAUSS_SIGMA
)
from viz_utils import (
    load_predictions, load_test_samples_and_files, load_raw_npz_for_test_files,
    flat15_to_t53, speed_mag, rel_l2_sample, choose_section_mask, projected_axes,
    interpolate_to_masked_grid, pressure_out_frame
)


def plot_2d_contours(sample_idx, frame_idx, y_true, y_pred, test_samples, out_dir):
    pos = test_samples[sample_idx]["x"]  # (N,3)

    yt = flat15_to_t53(y_true[sample_idx])[:, frame_idx, :]  # (N,3)
    yp = flat15_to_t53(y_pred[sample_idx])[:, frame_idx, :]  # (N,3)

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

    fig, axs = plt.subplots(2, 4, figsize=(19, 8.5), dpi=FIG_DPI)

    for col, (name, gt_vals, pr_vals) in enumerate(fields):
        XI, YI, Zg, valid = interpolate_to_masked_grid(
            px, py, gt_vals, nx=GRID_NX, ny=GRID_NY, sigma=GAUSS_SIGMA
        )
        _, _, Zp, _ = interpolate_to_masked_grid(
            px, py, pr_vals, nx=GRID_NX, ny=GRID_NY, sigma=GAUSS_SIGMA
        )

        # Same value range for GT/Pred
        vmin = np.nanmin([Zg, Zp])
        vmax = np.nanmax([Zg, Zp])
        levels = np.linspace(vmin, vmax, 50)

        c0 = axs[0, col].contourf(XI, YI, Zg, levels=levels, cmap=CMAP_MAIN, extend="both")
        c1 = axs[1, col].contourf(XI, YI, Zp, levels=levels, cmap=CMAP_MAIN, extend="both")

        axs[0, col].set_title(f"GT {name}")
        axs[1, col].set_title(f"Pred {name}")
        fig.colorbar(c0, ax=axs[0, col], shrink=0.82)
        fig.colorbar(c1, ax=axs[1, col], shrink=0.82)

        for r in [0, 1]:
            axs[r, col].set_xlabel(labels[0])
            axs[r, col].set_ylabel(labels[1])
            axs[r, col].set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Sample {sample_idx} frame {frame_idx} | 2D contour section "
        f"@{SECTION_AXIS}={target:.4f}±{hw:.4f}"
    )
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_2d_contours.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_3d_cfd_style(sample_idx, frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir):
    """
    3D view:
    - wing surface colored by pressure
    - plane slice (default y-median) colored by u_x (GT and Pred side-by-side)
    """
    s = test_samples[sample_idx]
    raw = raw_samples[sample_idx]

    pos = s["x"]                       # (N,3)
    idcs_air = s["idcs_airfoil"]       # local indices in sampled points
    yt = flat15_to_t53(y_true[sample_idx])[:, frame_idx, :]  # (N,3)
    yp = flat15_to_t53(y_pred[sample_idx])[:, frame_idx, :]
    rel = rel_l2_sample(y_true[sample_idx], y_pred[sample_idx])

    # pressure on surface from raw sample
    # We assume local indices correspond to same points in full sample for this loader contract.
    p_full = pressure_out_frame(raw, frame_idx)   # (100000,)
    # if local point cloud is full 100k and aligned:
    p_surface = p_full[idcs_air] if idcs_air.max() < p_full.shape[0] else np.zeros_like(idcs_air, dtype=np.float32)

    # slice plane: y-median band
    y = pos[:, 1]
    y0 = np.quantile(y, 0.5)
    dy = max((y.max() - y.min()) * 0.015, 1e-6)
    m = np.abs(y - y0) <= dy

    # Triangulate plane in x-z for smoother 3D sheet rendering
    xz = np.column_stack([pos[m, 0], pos[m, 2]])
    tri = Triangulation(xz[:, 0], xz[:, 1])

    ux_gt = yt[m, 0]
    ux_pr = yp[m, 0]
    ux_min = min(np.min(ux_gt), np.min(ux_pr))
    ux_max = max(np.max(ux_gt), np.max(ux_pr))

    pmin, pmax = np.min(p_surface), np.max(p_surface)

    fig = plt.figure(figsize=(16, 6), dpi=FIG_DPI)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, ux, title in [(ax0, ux_gt, "GT"), (ax1, ux_pr, "Pred")]:
        # 1) Wing surface points (pressure)
        surf = ax.scatter(
            pos[idcs_air, 0], pos[idcs_air, 1], pos[idcs_air, 2],
            c=p_surface, s=2.0, cmap=CMAP_PRESS, vmin=pmin, vmax=pmax, alpha=0.95
        )

        # 2) Plane at y ~ y0 colored by ux
        # render as trisurf in x-z and constant y=y0
        X = xz[:, 0]
        Z = xz[:, 1]
        Y = np.full_like(X, y0)

        # facecolors from ux node values (approx by triangle mean)
        tri_vals = ux[tri.triangles].mean(axis=1)
        norm = plt.Normalize(vmin=ux_min, vmax=ux_max)
        facecolors = plt.cm.get_cmap(CMAP_MAIN)(norm(tri_vals))

        ax.plot_trisurf(
            X, Y, Z, triangles=tri.triangles,
            linewidth=0.0, antialiased=False, shade=False, facecolors=facecolors, alpha=0.95
        )

        ax.set_title(f"{title}: pressure on wing + u_x on y-plane")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-60)

    # two colorbars
    m_pressure = plt.cm.ScalarMappable(cmap=CMAP_PRESS, norm=plt.Normalize(vmin=pmin, vmax=pmax))
    m_pressure.set_array([])
    cbar1 = fig.colorbar(m_pressure, ax=[ax0, ax1], fraction=0.02, pad=0.02)
    cbar1.set_label("Pressure (Pa)")

    m_ux = plt.cm.ScalarMappable(cmap=CMAP_MAIN, norm=plt.Normalize(vmin=ux_min, vmax=ux_max))
    m_ux.set_array([])
    cbar2 = fig.colorbar(m_ux, ax=[ax0, ax1], fraction=0.02, pad=0.10)
    cbar2.set_label("u_x (m/s) on y-plane")

    fig.suptitle(f"Sample {sample_idx} frame {frame_idx} | relL2={rel:.4f}")
    out_path = out_dir / f"sample{sample_idx:03d}_frame{frame_idx}_3d_cfd_style.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--frame_idx", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR / "single"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = load_predictions(RESULTS_DIR)
    test_samples, test_files = load_test_samples_and_files(DATA_DIR, TEST_FRAC, SEED)
    raw_samples = load_raw_npz_for_test_files(test_files)

    if len(test_samples) != y_true.shape[0]:
        raise ValueError("Prediction batch size doesn't match test split size.")

    p2d = plot_2d_contours(args.sample_idx, args.frame_idx, y_true, y_pred, test_samples, out_dir)
    p3d = plot_3d_cfd_style(args.sample_idx, args.frame_idx, y_true, y_pred, test_samples, raw_samples, out_dir)

    print(f"Saved:\n- {p2d}\n- {p3d}")
    print(f"Source: {test_files[args.sample_idx].name}")


if __name__ == "__main__":
    main()
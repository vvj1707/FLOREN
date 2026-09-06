import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from viz_config import (
    DATA_DIR, RESULTS_DIR, OUT_DIR, TEST_FRAC, SEED,
    ANIM_FPS, ANIM_DPI, CMAP_MAIN,
    SECTION_AXIS, SECTION_QUANTILE, SECTION_BAND_FRAC,
    GRID_NX, GRID_NY, GAUSS_SIGMA
)
from viz_utils import (
    load_predictions, load_test_samples_and_files,
    flat15_to_t53, choose_section_mask, projected_axes,
    interpolate_to_masked_grid, rel_l2_sample, speed_mag
)


def build_frame_fields(pos, yt53, yp53):
    mask, target, hw = choose_section_mask(
        pos, axis=SECTION_AXIS, quantile=SECTION_QUANTILE, band_frac=SECTION_BAND_FRAC
    )
    i, j, labels = projected_axes(SECTION_AXIS)
    px = pos[mask, i]
    py = pos[mask, j]

    all_frames = []
    for t in range(5):
        gt = yt53[:, t, :]
        pr = yp53[:, t, :]

        var_map = {
            "|u|": (speed_mag(gt[mask]), speed_mag(pr[mask])),
            "ux": (gt[mask, 0], pr[mask, 0]),
            "uy": (gt[mask, 1], pr[mask, 1]),
            "uz": (gt[mask, 2], pr[mask, 2]),
        }

        frame_dict = {}
        XI = YI = None
        for name, (gvals, pvals) in var_map.items():
            XI, YI, Zg, _ = interpolate_to_masked_grid(
                px, py, gvals, nx=GRID_NX, ny=GRID_NY, sigma=GAUSS_SIGMA
            )
            _, _, Zp, _ = interpolate_to_masked_grid(
                px, py, pvals, nx=GRID_NX, ny=GRID_NY, sigma=GAUSS_SIGMA
            )
            frame_dict[name] = (Zg, Zp)

        all_frames.append((XI, YI, frame_dict))

    return all_frames, target, hw, labels


def make_contour_animation(sample_idx, y_true, y_pred, test_samples, out_path, fps=ANIM_FPS):
    pos = test_samples[sample_idx]["x"]
    yt53 = flat15_to_t53(y_true[sample_idx])  # (N,5,3)
    yp53 = flat15_to_t53(y_pred[sample_idx])

    rel = rel_l2_sample(y_true[sample_idx], y_pred[sample_idx])
    precomp, target, hw, labels = build_frame_fields(pos, yt53, yp53)

    vars_order = ["|u|", "ux", "uy", "uz"]

    # global color limits for stable animation
    var_limits = {}
    for vn in vars_order:
        mins, maxs = [], []
        for _, _, fd in precomp:
            Zg, Zp = fd[vn]
            mins.append(np.nanmin([Zg, Zp]))
            maxs.append(np.nanmax([Zg, Zp]))
        var_limits[vn] = (float(np.min(mins)), float(np.max(maxs)))

    fig, axs = plt.subplots(2, 4, figsize=(19, 8.5), dpi=ANIM_DPI)
    title = fig.suptitle("")

    # draw colorbars once using dummy mappables
    mappables = []
    for c, vn in enumerate(vars_order):
        vmin, vmax = var_limits[vn]
        sm = plt.cm.ScalarMappable(cmap=CMAP_MAIN, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        mappables.append(sm)
        fig.colorbar(sm, ax=axs[0, c], shrink=0.82)
        fig.colorbar(sm, ax=axs[1, c], shrink=0.82)

    def draw_frame(t):
        XI, YI, fd = precomp[t]

        for r in range(2):
            for c in range(4):
                axs[r, c].clear()

        for c, vn in enumerate(vars_order):
            Zg, Zp = fd[vn]
            vmin, vmax = var_limits[vn]
            levels = np.linspace(vmin, vmax, 50)

            axs[0, c].contourf(XI, YI, Zg, levels=levels, cmap=CMAP_MAIN, extend="both")
            axs[1, c].contourf(XI, YI, Zp, levels=levels, cmap=CMAP_MAIN, extend="both")

            axs[0, c].set_title(f"GT {vn}")
            axs[1, c].set_title(f"Pred {vn}")

            for r in [0, 1]:
                axs[r, c].set_xlabel(labels[0])
                axs[r, c].set_ylabel(labels[1])
                axs[r, c].set_aspect("equal", adjustable="box")

        title.set_text(
            f"Sample {sample_idx} | frame {t} (chronological) | "
            f"section @{SECTION_AXIS}={target:.4f}±{hw:.4f} | relL2={rel:.4f}"
        )
        return []

    anim = FuncAnimation(fig, draw_frame, frames=5, interval=700, blit=False, repeat=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".mp4":
        writer = FFMpegWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)

    anim.save(out_path, writer=writer)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(OUT_DIR / "anim" / "sample000_contours.mp4"))
    parser.add_argument("--fps", type=int, default=ANIM_FPS)
    args = parser.parse_args()

    y_true, y_pred = load_predictions(RESULTS_DIR)
    test_samples, test_files = load_test_samples_and_files(DATA_DIR, TEST_FRAC, SEED)

    if len(test_samples) != y_true.shape[0]:
        raise ValueError("Prediction batch size doesn't match test split size.")

    out_path = Path(args.out)
    make_contour_animation(args.sample_idx, y_true, y_pred, test_samples, out_path, fps=args.fps)

    print(f"Saved animation: {out_path}")
    print(f"Source: {test_files[args.sample_idx].name}")


if __name__ == "__main__":
    main()
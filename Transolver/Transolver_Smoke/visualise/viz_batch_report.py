import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from viz_config import (
    DATA_DIR, RESULTS_DIR, VIZ_DIR, TEST_FRAC, SEED,
    FIG_DPI, N_REPORT_SAMPLES
)
from viz_utils import (
    load_predictions, load_test_samples, flat15_to_t53, speed_mag,
    per_sample_rel_l2, ensure_dir
)


def make_report(y_true, y_pred, test_files, out_dir, n_samples):
    B = y_true.shape[0]
    n = min(n_samples, B)

    rels = [per_sample_rel_l2(y_true[i], y_pred[i]) for i in range(B)]
    order = np.argsort(rels)

    chosen = np.concatenate([
        order[: max(1, n // 3)],                     # best
        order[B // 2: B // 2 + max(1, n // 3)],      # mid
        order[-max(1, n - 2 * (n // 3)):]            # worst
    ])[:n]

    fig, axs = plt.subplots(n, 3, figsize=(12, 3.2 * n), dpi=FIG_DPI)
    if n == 1:
        axs = np.array([axs])

    for r, i in enumerate(chosen):
        yt = flat15_to_t53(y_true[i])   # (N,5,3)
        yp = flat15_to_t53(y_pred[i])

        mt = np.linalg.norm(yt, axis=-1).mean(axis=0)  # (5,)
        mp = np.linalg.norm(yp, axis=-1).mean(axis=0)
        me = np.linalg.norm(yp - yt, axis=-1).mean(axis=0)

        axs[r, 0].plot(mt, marker="o", label="GT")
        axs[r, 0].plot(mp, marker="s", label="Pred")
        axs[r, 0].set_title(f"[{i}] mean |u| vs frame")
        axs[r, 0].set_xlabel("frame")
        axs[r, 0].set_ylabel("mean |u|")
        axs[r, 0].legend()

        axs[r, 1].plot(me, marker="o", color="tab:red")
        axs[r, 1].set_title(f"[{i}] mean |Δu| vs frame")
        axs[r, 1].set_xlabel("frame")
        axs[r, 1].set_ylabel("mean error")

        axs[r, 2].axis("off")
        txt = (
            f"sample idx: {i}\n"
            f"file: {test_files[i].name}\n"
            f"relL2: {rels[i]:.5f}\n"
        )
        axs[r, 2].text(0.02, 0.6, txt, fontsize=10)

    fig.tight_layout()
    out_path = out_dir / "batch_report.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    # save summary csv
    csv_path = out_dir / "batch_report_metrics.csv"
    with open(csv_path, "w") as f:
        f.write("sample_idx,file,rel_l2\n")
        for i in range(B):
            f.write(f"{i},{test_files[i].name},{rels[i]:.8f}\n")

    return out_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=N_REPORT_SAMPLES)
    parser.add_argument("--out_dir", type=str, default=str(VIZ_DIR / "report"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    y_true, y_pred = load_predictions(RESULTS_DIR)
    test_samples, test_files = load_test_samples(DATA_DIR, TEST_FRAC, SEED)

    if len(test_samples) != y_true.shape[0]:
        raise ValueError("Mismatch between predictions and test split length.")

    png, csv = make_report(y_true, y_pred, test_files, out_dir, args.n_samples)
    print(f"Saved report: {png}")
    print(f"Saved metrics csv: {csv}")


if __name__ == "__main__":
    main()
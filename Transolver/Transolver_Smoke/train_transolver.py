"""
Tiny TransolverAR run for quick sanity check on learnability.
Uses a very small subset of files (~1GB equivalent) and smaller model/training budget.
"""

from pathlib import Path
import numpy as np
import wandb

from load_data import load_split
from config import (
    seed_all, normalize_fx, normalize_y,
    train_transolver_ar, predict_full_autoregressive,
    relative_l2_dataset, save_artifacts,
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WARMUP_EPOCHS, GRAD_CLIP, LR_FLOOR,
    N_LAYERS, N_HIDDEN, N_HEAD, MLP_RATIO, DROPOUT_RATE, SLICE_NUM, REF,
    SPACE_DIM, FUN_DIM, OUT_DIM, TIME_INPUT, UNIFIED_POS, GEOMETRY, DEVICE,
    TRAIN_POINTS, VAL_POINTS, TF_DECAY_EPOCHS, HARD_NOSLIP
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data/warped-ifw"
TEST_FRAC = 0.15
SEED = 42

USE_WANDB = True
WANDB_PROJECT = "warped-ifw-transolver-ar-tiny"
WANDB_ENTITY = None

# ---- tiny subset controls (~1GB scale) ----
MAX_TRAIN_FILES = 32
MAX_TEST_FILES = 8
FULL_EVAL_EVERY = 1


def main():
    seed_all(SEED)

    run = None
    if USE_WANDB:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name="TransolverAR-WarpedIFW-tiny",
            config={
                "model": "TransolverAR",
                "framework": "JAX/Flax",
                "dataset": "Warped-IFW",
                "tiny_subset": True,
                "max_train_files": MAX_TRAIN_FILES,
                "max_test_files": MAX_TEST_FILES,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_epochs": WARMUP_EPOCHS,
                "grad_clip": GRAD_CLIP,
                "lr_floor": LR_FLOOR,
                "n_layers": N_LAYERS,
                "n_hidden": N_HIDDEN,
                "n_head": N_HEAD,
                "mlp_ratio": MLP_RATIO,
                "dropout_rate": DROPOUT_RATE,
                "slice_num": SLICE_NUM,
                "ref": REF,
                "space_dim": SPACE_DIM,
                "fun_dim": FUN_DIM,
                "out_dim": OUT_DIM,
                "time_input": TIME_INPUT,
                "unified_pos": UNIFIED_POS,
                "geometry": GEOMETRY,
                "backend": DEVICE,
                "test_frac": TEST_FRAC,
                "seed": SEED,
                "train_points": TRAIN_POINTS,
                "val_points": VAL_POINTS,
                "tf_decay_epochs": TF_DECAY_EPOCHS,
                "hard_noslip": HARD_NOSLIP,
                "full_eval_every": FULL_EVAL_EVERY,
            },
        )

    print("Loading Warped-IFW samples...")
    train_samples, test_samples, train_files, test_files = load_split(
        data_dir=DATA_DIR, test_frac=TEST_FRAC, seed=SEED
    )

    # tiny file subset
    train_samples = train_samples[:MAX_TRAIN_FILES]
    test_samples = test_samples[:MAX_TEST_FILES]
    train_files = train_files[:MAX_TRAIN_FILES]
    test_files = test_files[:MAX_TEST_FILES]

    print(f"TINY RUN -> Train files: {len(train_files)}  Test files: {len(test_files)}")

    if run is not None:
        run.config.update({
            "n_train_files": len(train_files),
            "n_test_files": len(test_files),
        }, allow_val_change=True)

    print("Normalizing inputs/targets...")
    train_samples, test_samples, fx_mu, fx_std = normalize_fx(train_samples, test_samples)
    train_samples, test_samples, y_mu, y_std = normalize_y(train_samples, test_samples)

    trained = train_transolver_ar(
        train_samples=train_samples,
        test_samples=test_samples,
        seed=SEED,
        wandb_run=run,
        y_mu=y_mu,
        y_std=y_std,
        full_eval_every=FULL_EVAL_EVERY,
    )

    print("Running full-point autoregressive inference on tiny test set...")
    y_pred = predict_full_autoregressive(trained, test_samples, y_mu, y_std)

    y_true = np.stack([s["y"] for s in test_samples], axis=0) * y_std[None, None, :] + y_mu[None, None, :]

    mean_rel_l2, rel_l2_per_sample = relative_l2_dataset(y_true, y_pred)
    metrics = {
        "relative_L2_mean": float(mean_rel_l2),
        "relative_L2_std": float(rel_l2_per_sample.std()),
        "relative_L2_per_sample": rel_l2_per_sample,
    }

    print("\nTiny-run test metrics:")
    print(f"  relative_L2_mean: {metrics['relative_L2_mean']:.6f}")
    print(f"  relative_L2_std : {metrics['relative_L2_std']:.6f}")

    if run is not None:
        table = wandb.Table(columns=["sample_idx", "file", "rel_l2"])
        for i, fp in enumerate(test_files):
            table.add_data(int(i), fp.name, float(rel_l2_per_sample[i]))

        run.log({
            "final_full/relative_L2_mean": metrics["relative_L2_mean"],
            "final_full/relative_L2_std": metrics["relative_L2_std"],
            "final_full/per_sample_rel_l2": table,
        })
        run.finish()

    save_artifacts(trained, metrics, y_true, y_pred, fx_mu, fx_std, y_mu, y_std)


if __name__ == "__main__":
    main()
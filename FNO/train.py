"""
train.py

Custom training loop for Warped-IFW using scirex-zenteiq FNO on voxel grids.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from load_data import load_split
from config import (
    seed_all, build_fno, make_schedule,
    build_grid_spec, points_to_grid, trilinear_grid_to_points,
    normalize_grid_inputs, normalize_grid_targets,
    save_artifacts,
    TEST_FRAC, RANDOM_SEED, EPOCHS, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP, WARMUP_EPOCHS, LR_FLOOR,
    GRID_X, GRID_Y, GRID_Z, DEVICE
)
from loss import relative_l2_grid
from metrics import relative_l2_dataset
from comparison import evaluate_baseline

DATA_DIR = Path(__file__).resolve().parent / "data/warped-ifw"

USE_WANDB = True
WANDB_PROJECT = "warped-ifw-scirex-fno"
WANDB_ENTITY = None


def make_batches(X, Y, M, batch_size, rng, shuffle=True):
    idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), batch_size):
        sl = idx[i:i + batch_size]
        yield (
            jnp.asarray(X[sl], dtype=jnp.float32),
            jnp.asarray(Y[sl], dtype=jnp.float32),
            jnp.asarray(M[sl], dtype=jnp.float32),
        )


def main():
    seed_all(RANDOM_SEED)

    run = None
    if USE_WANDB:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name="FNO-IFW-canonical",
            config={
                "model": "scirex-FNO",
                "framework": "JAX/Flax",
                "dataset": "Warped-IFW",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_epochs": WARMUP_EPOCHS,
                "lr_floor": LR_FLOOR,
                "grad_clip": GRAD_CLIP,
                "grid": [GRID_X, GRID_Y, GRID_Z],
                "backend": DEVICE,
                "test_frac": TEST_FRAC,
                "seed": RANDOM_SEED,
            },
        )

    print("Loading samples...")
    train_samples, test_samples, train_files, test_files = load_split(
        data_dir=DATA_DIR,
        test_frac=TEST_FRAC,
        seed=RANDOM_SEED,
    )
    print(f"Train files: {len(train_files)}  Test files: {len(test_files)}")

    baseline = evaluate_baseline(test_samples)
    print(f"Persistence baseline relative_L2_mean: {baseline['relative_L2_mean']:.6f}")
    if run is not None:
        run.log({
            "baseline/relative_L2_mean": baseline["relative_L2_mean"],
            "baseline/relative_L2_std": baseline["relative_L2_std"],
            "baseline/RMSE": baseline["RMSE"],
            "baseline/MAE": baseline["MAE"],
        })

    print("Building grid spec...")
    grid_spec = build_grid_spec(train_samples + test_samples, gx=GRID_X, gy=GRID_Y, gz=GRID_Z)
    print(f"Grid: {grid_spec}")

    print("Rasterizing to grids...")
    Xtr, Ytr, Mtr = [], [], []
    for s in train_samples:
        xg, yg, mg = points_to_grid(s, grid_spec)
        Xtr.append(xg); Ytr.append(yg); Mtr.append(mg)
    Xtr = np.stack(Xtr, axis=0)
    Ytr = np.stack(Ytr, axis=0)
    Mtr = np.stack(Mtr, axis=0)

    Xte, Yte, Mte = [], [], []
    for s in test_samples:
        xg, yg, mg = points_to_grid(s, grid_spec)
        Xte.append(xg); Yte.append(yg); Mte.append(mg)
    Xte = np.stack(Xte, axis=0)
    Yte = np.stack(Yte, axis=0)
    Mte = np.stack(Mte, axis=0)

    Xall = np.concatenate([Xtr, Xte], axis=0)
    Yall = np.concatenate([Ytr, Yte], axis=0)

    Xall_n, x_mu, x_std = normalize_grid_inputs(Xtr, Xall)
    Yall_n, y_mu, y_std = normalize_grid_targets(Ytr, Yall)

    Xtr_n, Xte_n = Xall_n[:len(Xtr)], Xall_n[len(Xtr):]
    Ytr_n, Yte_n = Yall_n[:len(Ytr)], Yall_n[len(Ytr):]

    print(f"Backend: {DEVICE} ({jax.devices()[0]})")

    model = build_fno()
    key = jax.random.PRNGKey(RANDOM_SEED)
    params = model.init(key, jnp.asarray(Xtr_n[:1], dtype=jnp.float32))

    steps_per_epoch = max(1, int(np.ceil(len(Xtr_n) / BATCH_SIZE)))
    schedule = make_schedule(
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        lr_floor=LR_FLOOR,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.add_decayed_weights(WEIGHT_DECAY),
        optax.adam(schedule),
    )
    opt_state = tx.init(params)
    global_step = 0

    def loss_fn(p, xb, yb, mb):
        pred = model.apply(p, xb)
        return relative_l2_grid(pred, yb, mask=mb)

    @jax.jit
    def train_step(p, o, xb, yb, mb):
        l, g = jax.value_and_grad(loss_fn)(p, xb, yb, mb)
        upd, o = tx.update(g, o, p)
        p = optax.apply_updates(p, upd)
        return p, o, l

    @jax.jit
    def eval_step(p, xb, yb, mb):
        return loss_fn(p, xb, yb, mb)

    history = {"loss": [], "val_loss": [], "lr": [], "time": []}
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Training FNO | N_train={len(Xtr_n)} N_test={len(Xte_n)} epochs={EPOCHS}")

    for ep in range(EPOCHS):
        t0 = time.time()

        tr_losses = []
        for xb, yb, mb in make_batches(Xtr_n, Ytr_n, Mtr, BATCH_SIZE, rng, shuffle=True):
            params, opt_state, l = train_step(params, opt_state, xb, yb, mb)
            tr_losses.append(float(l))
            global_step += 1

        va_losses = []
        for xb, yb, mb in make_batches(Xte_n, Yte_n, Mte, BATCH_SIZE, rng, shuffle=False):
            l = eval_step(params, xb, yb, mb)
            va_losses.append(float(l))

        tl = float(np.mean(tr_losses))
        vl = float(np.mean(va_losses))
        lr = float(schedule(global_step))
        dt = time.time() - t0

        history["loss"].append(tl)
        history["val_loss"].append(vl)
        history["lr"].append(lr)
        history["time"].append(dt)

        print(f"[{ep:03d}] {dt:.2f}s lr={lr:.2e} train_l2={tl:.5f} val_l2={vl:.5f}")

        if run is not None:
            run.log(
                {
                    "epoch": ep,
                    "train/loss": tl,
                    "test/loss": vl,
                    "lr": lr,
                    "time_per_epoch": dt,
                    "global_step": global_step,
                },
                step=ep,
            )

    print("Point-level inference on test set...")
    y_pred_points = []
    for s in test_samples:
        Xg, _, _ = points_to_grid(s, grid_spec)
        Xg_n = Xg.copy()
        Xg_n[:16] = (Xg_n[:16] - x_mu[:, None, None, None]) / (x_std[:, None, None, None] + 1e-8)

        Yg_pred_n = np.asarray(model.apply(params, jnp.asarray(Xg_n[None], dtype=jnp.float32)))[0]
        Yg_pred = Yg_pred_n * y_std[:, None, None, None] + y_mu[:, None, None, None]

        yp = trilinear_grid_to_points(Yg_pred, s["x"], grid_spec)  # (N,15)
        y_pred_points.append(yp.astype(np.float32))

    y_pred_points = np.stack(y_pred_points, axis=0)
    y_true_points = np.stack([s["y"] for s in test_samples], axis=0).astype(np.float32)

    mean_rel_l2, per = relative_l2_dataset(y_true_points, y_pred_points)
    metrics = {
        "relative_L2_mean": float(mean_rel_l2),
        "relative_L2_std": float(per.std()),
        "relative_L2_per_sample": per,
    }

    print("\nFinal test point metrics:")
    print(f"  relative_L2_mean: {metrics['relative_L2_mean']:.6f}")
    print(f"  relative_L2_std : {metrics['relative_L2_std']:.6f}")

    if run is not None:
        run.log({
            "test/relative_L2_mean": metrics["relative_L2_mean"],
            "test/relative_L2_std": metrics["relative_L2_std"],
        })
        run.finish()

    trained = {
        "model": model,
        "params": params,
        "history": history,
        "x_mu": x_mu,
        "x_std": x_std,
        "y_mu": y_mu,
        "y_std": y_std,
        "grid_spec": grid_spec,
    }

    save_artifacts(
        trained=trained,
        metrics=metrics,
        y_true_points=y_true_points,
        y_pred_points=y_pred_points,
    )


if __name__ == "__main__":
    main()
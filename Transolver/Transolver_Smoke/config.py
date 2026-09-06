"""
TransolverAR training utilities for Warped-IFW.
Tiny-run oriented (clean terminal run, no debug/env overrides).

Key behavior:
- autoregressive rollout (5 future steps)
- training on subsampled points per sample
- scheduled teacher forcing
- hard no-slip enforcement on airfoil points
- honest W&B logging: subsampled normalized metrics are separated from
  full denormalized metrics that match visualization/output artifacts
"""

import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from operators.models.transolver_jax import Transolver
from load_data import subsample_sample_uniform

# -------------------------
# Paths
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "trained_model"
RESULTS_DIR = SCRIPT_DIR / "results"

# -------------------------
# Data constants
# -------------------------
IN_FRAMES = 5
OUT_FRAMES = 5
VEL_DIM = 3

SPACE_DIM = 3
FUN_DIM = IN_FRAMES * VEL_DIM + 1   # 15 + distance
OUT_DIM = 3                         # AR next-step output

# -------------------------
# Hyperparameters (tiny-run defaults)
# -------------------------
EPOCHS = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-6
WARMUP_EPOCHS = 5
LR_FLOOR = 1e-6
GRAD_CLIP = 1.0

N_LAYERS = 3
N_HIDDEN = 128
N_HEAD = 4
MLP_RATIO = 1
DROPOUT_RATE = 0.0
SLICE_NUM = 32
REF = 8

TIME_INPUT = False
UNIFIED_POS = False
GEOMETRY = "irregular"

RANDOM_SEED = 42
DEVICE = jax.default_backend()

# AR training settings
TRAIN_POINTS = 8_000
VAL_POINTS = 8_000
TF_DECAY_EPOCHS = 10
HARD_NOSLIP = True


def seed_all(seed=RANDOM_SEED):
    np.random.seed(seed)


def normalize_fx(train_samples, test_samples):
    fx_train = np.concatenate([s["fx"] for s in train_samples], axis=0)
    mu = fx_train.mean(axis=0).astype(np.float32)
    std = (fx_train.std(axis=0) + 1e-8).astype(np.float32)

    def apply(samples):
        out = []
        for s in samples:
            out.append({
                "x": s["x"].astype(np.float32),
                "fx": ((s["fx"] - mu) / std).astype(np.float32),
                "y": s["y"].astype(np.float32),
                "idcs_airfoil": s["idcs_airfoil"].astype(np.int32),
            })
        return out

    return apply(train_samples), apply(test_samples), mu, std


def normalize_y(train_samples, test_samples):
    y_train = np.concatenate([s["y"] for s in train_samples], axis=0)
    mu = y_train.mean(axis=0).astype(np.float32)     # (15,)
    std = (y_train.std(axis=0) + 1e-8).astype(np.float32)

    def apply(samples):
        out = []
        for s in samples:
            out.append({
                "x": s["x"].astype(np.float32),
                "fx": s["fx"].astype(np.float32),
                "y": ((s["y"] - mu) / std).astype(np.float32),
                "idcs_airfoil": s["idcs_airfoil"].astype(np.int32),
            })
        return out

    return apply(train_samples), apply(test_samples), mu, std


def build_transolver():
    return Transolver(
        space_dim=SPACE_DIM,
        n_layers=N_LAYERS,
        n_hidden=N_HIDDEN,
        dropout_rate=DROPOUT_RATE,
        n_head=N_HEAD,
        time_input=TIME_INPUT,
        mlp_ratio=MLP_RATIO,
        fun_dim=FUN_DIM,
        out_dim=OUT_DIM,
        slice_num=SLICE_NUM,
        ref=REF,
        unified_pos=UNIFIED_POS,
        geometry=GEOMETRY,
    )


def make_schedule(learning_rate, epochs, warmup_epochs, steps_per_epoch):
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    decay_steps = max(1, (epochs - warmup_epochs) * steps_per_epoch)

    warmup = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps
    )
    cosine = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=decay_steps, alpha=LR_FLOOR / learning_rate
    )
    return optax.join_schedules([warmup, cosine], [warmup_steps])


def teacher_forcing_prob(epoch: int, decay_epochs: int) -> float:
    if epoch >= decay_epochs:
        return 0.0
    return 1.0 - (epoch / max(1, decay_epochs))


def _apply_noslip(v_next: jnp.ndarray, idcs_airfoil: jnp.ndarray) -> jnp.ndarray:
    if idcs_airfoil.size == 0:
        return v_next
    return v_next.at[idcs_airfoil].set(0.0)


def _build_fx_from_window(v_window: jnp.ndarray, dist_col: jnp.ndarray) -> jnp.ndarray:
    flat = jnp.reshape(v_window, (v_window.shape[0], IN_FRAMES * VEL_DIM))  # (N,15)
    return jnp.concatenate([flat, dist_col], axis=-1)  # (N,16)


def _split_fx(sample_fx):
    v_flat = sample_fx[:, :15]
    dist = sample_fx[:, 15:16]
    v_in = jnp.reshape(v_flat, (v_flat.shape[0], IN_FRAMES, VEL_DIM))
    return v_in, dist


def relative_l2_rollout(pred_rollout: jnp.ndarray, target_rollout: jnp.ndarray) -> jnp.ndarray:
    num = jnp.sum((pred_rollout - target_rollout) ** 2)
    den = jnp.sum(target_rollout ** 2) + 1e-12
    return jnp.sqrt(num / den)


def mae_rollout(pred_rollout: jnp.ndarray, target_rollout: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(pred_rollout - target_rollout))


def relative_l2_dataset(y_true, y_pred):
    num = np.sum((y_pred - y_true) ** 2, axis=(1, 2))
    den = np.sum(y_true ** 2, axis=(1, 2)) + 1e-12
    per_sample = np.sqrt(num / den)
    return float(per_sample.mean()), per_sample


def predict_full_autoregressive_with_params(model, params, samples, y_mu, y_std):
    preds_all = []

    for s in samples:
        x = jnp.asarray(s["x"])
        fx = jnp.asarray(s["fx"])
        idcs_airfoil = jnp.asarray(s["idcs_airfoil"], dtype=jnp.int32)

        v_in, dist = _split_fx(fx)
        window = v_in

        preds = []
        for _ in range(OUT_FRAMES):
            fx_t = _build_fx_from_window(window, dist)
            pred_next = model.apply(params, x=x[None], fx=fx_t[None])[0]
            if HARD_NOSLIP:
                pred_next = _apply_noslip(pred_next, idcs_airfoil)

            preds.append(np.asarray(pred_next))
            window = jnp.concatenate([window[:, 1:, :], pred_next[:, None, :]], axis=1)

        pred_roll = np.stack(preds, axis=1)                     # (N,5,3)
        pred_flat = pred_roll.reshape(pred_roll.shape[0], -1)   # (N,15)
        pred_denorm = pred_flat * y_std[None, :] + y_mu[None, :]
        preds_all.append(pred_denorm)

    return np.stack(preds_all, axis=0)  # (B,N,15)


def train_transolver_ar(
    train_samples,
    test_samples,
    epochs=EPOCHS,
    seed=RANDOM_SEED,
    wandb_run=None,
    y_mu=None,
    y_std=None,
    full_eval_every=1,
):
    heartbeat_every = 10
    train_points = TRAIN_POINTS
    val_points = VAL_POINTS
    tf_decay_epochs = TF_DECAY_EPOCHS

    print(f"Backend: {DEVICE} ({jax.devices()[0]})")
    print(f"TRAIN_POINTS={train_points} VAL_POINTS={val_points} TF_DECAY_EPOCHS={tf_decay_epochs}")
    print(f"EPOCHS={epochs}")

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    model = build_transolver()

    rng_np = np.random.default_rng(seed)
    s0 = subsample_sample_uniform(train_samples[0], train_points, rng_np, keep_all_airfoil=True)
    x0 = jnp.asarray(s0["x"])[None]
    fx0 = jnp.asarray(s0["fx"])[None]

    t_init = time.time()
    params = model.init(init_rng, x=x0, fx=fx0)
    print(f"Model init done in {time.time() - t_init:.2f}s")

    steps_per_epoch = len(train_samples)
    schedule = make_schedule(LEARNING_RATE, epochs, WARMUP_EPOCHS, steps_per_epoch)

    tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.add_decayed_weights(WEIGHT_DECAY),
        optax.adam(schedule),
    )
    opt_state = tx.init(params)
    global_step = 0

    def rollout_loss_and_mae(p, x, fx_init, y_rollout, idcs_airfoil, tf_mask):
        v_in, dist = _split_fx(fx_init[0])
        target = jnp.reshape(y_rollout[0], (-1, OUT_FRAMES, VEL_DIM))

        preds = []
        window = v_in

        for t in range(OUT_FRAMES):
            fx_t = _build_fx_from_window(window, dist)
            pred_next = model.apply(p, x=x[0][None], fx=fx_t[None])[0]
            if HARD_NOSLIP:
                pred_next = _apply_noslip(pred_next, idcs_airfoil)

            preds.append(pred_next)

            gt_next = target[:, t, :]
            use_gt = tf_mask[t]
            next_for_window = use_gt * gt_next + (1.0 - use_gt) * pred_next
            if HARD_NOSLIP:
                next_for_window = _apply_noslip(next_for_window, idcs_airfoil)

            window = jnp.concatenate([window[:, 1:, :], next_for_window[:, None, :]], axis=1)

        pred_roll = jnp.stack(preds, axis=1)
        loss = relative_l2_rollout(pred_roll, target)
        m = mae_rollout(pred_roll, target)
        return loss, m

    @jax.jit
    def train_step(p, o, x, fx_init, y_rollout, idcs_airfoil, tf_mask):
        (loss, m), grads = jax.value_and_grad(rollout_loss_and_mae, has_aux=True)(
            p, x, fx_init, y_rollout, idcs_airfoil, tf_mask
        )
        updates, o = tx.update(grads, o, p)
        p = optax.apply_updates(p, updates)
        return p, o, loss, m

    @jax.jit
    def eval_step(p, x, fx_init, y_rollout, idcs_airfoil):
        tf_mask_eval = jnp.zeros((OUT_FRAMES,), dtype=jnp.float32)
        return rollout_loss_and_mae(p, x, fx_init, y_rollout, idcs_airfoil, tf_mask_eval)

    history = {
        "train_subsampled_relative_l2": [],
        "train_subsampled_mae": [],
        "val_subsampled_relative_l2": [],
        "val_subsampled_mae": [],
        "val_full_relative_L2_mean": [],
        "val_full_relative_L2_std": [],
        "lr": [],
        "time": [],
        "tf_prob": [],
    }

    print(
        f"Training TransolverAR | N_train={len(train_samples)} "
        f"N_test={len(test_samples)} epochs={epochs}"
    )

    first_step_timed = False

    for ep in range(epochs):
        t0 = time.time()
        tf_p = teacher_forcing_prob(ep, tf_decay_epochs)
        history["tf_prob"].append(tf_p)

        train_idx = rng_np.permutation(len(train_samples))
        t_losses, t_maes = [], []

        for i, idx in enumerate(train_idx):
            s = subsample_sample_uniform(
                train_samples[int(idx)],
                train_points,
                rng_np,
                keep_all_airfoil=True,
            )

            x = jnp.asarray(s["x"])[None]
            fx_init = jnp.asarray(s["fx"])[None]
            y_rollout = jnp.asarray(s["y"])[None]
            idcs_airfoil = jnp.asarray(s["idcs_airfoil"], dtype=jnp.int32)

            tf_mask_np = (rng_np.random(OUT_FRAMES) < tf_p).astype(np.float32)
            tf_mask = jnp.asarray(tf_mask_np)

            if not first_step_timed:
                tc = time.time()
                params, opt_state, l, m = train_step(
                    params, opt_state, x, fx_init, y_rollout, idcs_airfoil, tf_mask
                )
                print(f"First train_step (compile+execute) took {time.time() - tc:.2f}s")
                first_step_timed = True
            else:
                params, opt_state, l, m = train_step(
                    params, opt_state, x, fx_init, y_rollout, idcs_airfoil, tf_mask
                )

            t_losses.append(float(l))
            t_maes.append(float(m))
            global_step += 1

            if (i + 1) % heartbeat_every == 0 or (i + 1) == len(train_idx):
                print(f"[epoch {ep:03d}] train {i+1}/{len(train_idx)} (elapsed {time.time()-t0:.1f}s)")

        val_idx = np.arange(len(test_samples))
        v_losses, v_maes = [], []
        for j, idx in enumerate(val_idx):
            val_rng = np.random.default_rng(seed + 10_000 + int(idx))
            s = subsample_sample_uniform(
                test_samples[int(idx)],
                val_points,
                val_rng,
                keep_all_airfoil=True,
            )

            x = jnp.asarray(s["x"])[None]
            fx_init = jnp.asarray(s["fx"])[None]
            y_rollout = jnp.asarray(s["y"])[None]
            idcs_airfoil = jnp.asarray(s["idcs_airfoil"], dtype=jnp.int32)

            l, m = eval_step(params, x, fx_init, y_rollout, idcs_airfoil)
            v_losses.append(float(l))
            v_maes.append(float(m))

            if (j + 1) % max(1, heartbeat_every // 2) == 0 or (j + 1) == len(val_idx):
                print(f"[epoch {ep:03d}] eval {j+1}/{len(val_idx)}")

        tl, tm = float(np.mean(t_losses)), float(np.mean(t_maes))
        vl, vm = float(np.mean(v_losses)), float(np.mean(v_maes))
        lr = float(schedule(global_step))
        dt = time.time() - t0

        full_rel_l2_mean = None
        full_rel_l2_std = None
        if y_mu is not None and y_std is not None and (ep % full_eval_every == 0):
            y_pred_full = predict_full_autoregressive_with_params(model, params, test_samples, y_mu, y_std)
            y_true_full = (
                np.stack([s["y"] for s in test_samples], axis=0) * y_std[None, None, :] + y_mu[None, None, :]
            )
            full_rel_l2_mean, full_rel_l2_per_sample = relative_l2_dataset(y_true_full, y_pred_full)
            full_rel_l2_std = float(full_rel_l2_per_sample.std())

        history["train_subsampled_relative_l2"].append(tl)
        history["train_subsampled_mae"].append(tm)
        history["val_subsampled_relative_l2"].append(vl)
        history["val_subsampled_mae"].append(vm)
        history["val_full_relative_L2_mean"].append(None if full_rel_l2_mean is None else float(full_rel_l2_mean))
        history["val_full_relative_L2_std"].append(None if full_rel_l2_std is None else float(full_rel_l2_std))
        history["lr"].append(lr)
        history["time"].append(dt)

        msg = (
            f"[{ep:03d}] {dt:.1f}s lr={lr:.2e} tf_p={tf_p:.3f} "
            f"train_sub_l2={tl:.4f} val_sub_l2={vl:.4f}"
        )
        if full_rel_l2_mean is not None:
            msg += f" val_full_l2={full_rel_l2_mean:.4f}"
        print(msg)

        if wandb_run is not None:
            log_dict = {
                "epoch": ep,
                "train_subsampled/relative_l2": tl,
                "train_subsampled/mae": tm,
                "val_subsampled/relative_l2": vl,
                "val_subsampled/mae": vm,
                "lr": lr,
                "time_per_epoch": dt,
                "tf_prob": tf_p,
                "global_step": global_step,
                "train_points": train_points,
                "val_points": val_points,
            }
            if full_rel_l2_mean is not None:
                log_dict["val_full/relative_L2_mean"] = float(full_rel_l2_mean)
                log_dict["val_full/relative_L2_std"] = float(full_rel_l2_std)

            wandb_run.log(log_dict, step=ep)

    return {"model": model, "params": params, "history": history}


def predict_full_autoregressive(trained, samples, y_mu, y_std):
    model, params = trained["model"], trained["params"]
    return predict_full_autoregressive_with_params(model, params, samples, y_mu, y_std)


def save_artifacts(trained, metrics, y_true, y_pred, fx_mu, fx_std, y_mu, y_std):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    params_np = jax.tree_util.tree_map(np.asarray, trained["params"])
    with open(MODEL_DIR / "transolver_ar_params.pkl", "wb") as f:
        pickle.dump(params_np, f)

    np.save(MODEL_DIR / "fx_mu.npy", fx_mu)
    np.save(MODEL_DIR / "fx_std.npy", fx_std)
    np.save(MODEL_DIR / "y_mu.npy", y_mu)
    np.save(MODEL_DIR / "y_std.npy", y_std)

    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(trained["history"], f, indent=2)

    np.save(RESULTS_DIR / "y_true.npy", y_true)
    np.save(RESULTS_DIR / "y_pred.npy", y_pred)
    np.save(RESULTS_DIR / "relative_l2_per_sample.npy", metrics["relative_L2_per_sample"])

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump({
            "relative_L2_mean": float(metrics["relative_L2_mean"]),
            "relative_L2_std": float(metrics["relative_L2_std"]),
        }, f, indent=2)

    print(f"Saved model to: {MODEL_DIR}")
    print(f"Saved results to: {RESULTS_DIR}")
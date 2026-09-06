from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from load_data import load_split
from operators.models.transolver_jax import Transolver
from config import (
    CACHE_DIR,
    DEVICE,
    FUN_DIM,
    IN_FRAMES,
    MODEL_DIR,
    OUT_DIM,
    OUT_FRAMES,
    RESULTS_DIR,
    SPACE_DIM,
    VEL_DIM,
    USE_WANDB,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_arg_parser,
    config_from_args,
    normalize_fx,
    normalize_y,
    save_config,
    seed_all,
)
from preprocessing import preprocess_samples
from subsampling import subsample_sample
from validation import (
    mae_rollout,
    relative_l2_rollout,
    save_json,
    save_per_sample_metrics,
)


def build_transolver(cfg):
    return Transolver(
        space_dim=SPACE_DIM,
        n_layers=cfg.n_layers,
        n_hidden=cfg.n_hidden,
        dropout_rate=cfg.dropout_rate,
        n_head=cfg.n_head,
        time_input=cfg.time_input,
        mlp_ratio=cfg.mlp_ratio,
        fun_dim=FUN_DIM,
        out_dim=OUT_DIM,
        slice_num=cfg.slice_num,
        ref=cfg.ref,
        unified_pos=cfg.unified_pos,
        geometry=cfg.geometry,
    )


def make_schedule(cfg, steps_per_epoch):
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)
    decay_steps = max(1, (cfg.epochs - cfg.warmup_epochs) * steps_per_epoch)
    warmup = optax.linear_schedule(0.0, cfg.learning_rate, warmup_steps)
    cosine = optax.cosine_decay_schedule(
        cfg.learning_rate,
        decay_steps,
        alpha=cfg.lr_floor / cfg.learning_rate,
    )
    return optax.join_schedules([warmup, cosine], [warmup_steps])


def teacher_forcing_prob(epoch: int, cfg) -> float:
    if cfg.tf_schedule == "linear":
        if epoch >= cfg.tf_decay_epochs:
            return 0.0
        return 1.0 - (epoch / max(1, cfg.tf_decay_epochs))
    if cfg.tf_schedule == "cosine":
        if epoch >= cfg.tf_decay_epochs:
            return 0.0
        x = epoch / max(1, cfg.tf_decay_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * x))
    if cfg.tf_schedule == "piecewise":
        if epoch < cfg.tf_decay_epochs * 0.5:
            return 1.0
        if epoch >= cfg.tf_decay_epochs:
            return 0.0
        x = (epoch - cfg.tf_decay_epochs * 0.5) / max(1e-6, cfg.tf_decay_epochs * 0.5)
        return max(0.0, 1.0 - x)
    raise ValueError(cfg.tf_schedule)


def _apply_noslip(v_next: jnp.ndarray, idcs_airfoil: jnp.ndarray) -> jnp.ndarray:
    """if idcs_airfoil.size == 0:
        return v_next"""
    return v_next.at[idcs_airfoil].set(0.0)


def _build_fx_from_window(v_window: jnp.ndarray, dist_col: jnp.ndarray) -> jnp.ndarray:
    flat = jnp.reshape(v_window, (v_window.shape[0], IN_FRAMES * VEL_DIM))
    return jnp.concatenate([flat, dist_col], axis=-1)


def _split_fx(sample_fx):
    v_flat = sample_fx[:, :15]
    dist = sample_fx[:, 15:16]
    v_in = jnp.reshape(v_flat, (v_flat.shape[0], IN_FRAMES, VEL_DIM))
    return v_in, dist


def _loss_point_mask(n_points: int, idcs_airfoil: jnp.ndarray, exclude_airfoil: bool) -> jnp.ndarray | None:
    if not exclude_airfoil:
        return None
    mask = jnp.ones((n_points,), dtype=jnp.float32)
    if idcs_airfoil.size > 0:
        mask = mask.at[idcs_airfoil].set(0.0)
    return mask


def augment_sample(sample, rng: np.random.Generator, cfg):
    s = {
        "x": sample["x"].copy(),
        "fx": sample["fx"].copy(),
        "y": sample["y"].copy(),
        "idcs_airfoil": sample["idcs_airfoil"].copy(),
    }
    if cfg.aug_y_flip and rng.random() < 0.5:
        s["x"][:, 1] *= -1.0
        v_in = s["fx"][:, :15].reshape(s["fx"].shape[0], 5, 3)
        v_in[:, :, 1] *= -1.0
        s["fx"][:, :15] = v_in.reshape(s["fx"].shape[0], 15)
        y = s["y"].reshape(s["y"].shape[0], 5, 3)
        y[:, :, 1] *= -1.0
        s["y"] = y.reshape(s["y"].shape[0], 15)
    if cfg.aug_noise_std_pos > 0:
        noise = rng.normal(scale=cfg.aug_noise_std_pos, size=s["x"].shape).astype(np.float32)
        if cfg.aug_noise_skip_airfoil and s["idcs_airfoil"].size > 0:
            noise[s["idcs_airfoil"]] = 0.0
        s["x"] += noise
    if cfg.aug_noise_std_vel > 0:
        v_in = s["fx"][:, :15].reshape(s["fx"].shape[0], 5, 3)
        noise_v = rng.normal(scale=cfg.aug_noise_std_vel, size=v_in.shape).astype(np.float32)
        y_noise = rng.normal(
            scale=cfg.aug_noise_std_vel,
            size=s["y"].reshape(s["y"].shape[0], 5, 3).shape,
        ).astype(np.float32)
        if cfg.aug_noise_skip_airfoil and s["idcs_airfoil"].size > 0:
            noise_v[s["idcs_airfoil"]] = 0.0
            y_noise[s["idcs_airfoil"]] = 0.0
        v_in = v_in + noise_v
        s["fx"][:, :15] = v_in.reshape(s["fx"].shape[0], 15)
        y = s["y"].reshape(s["y"].shape[0], 5, 3) + y_noise
        s["y"] = y.reshape(s["y"].shape[0], 15)
    return s


def relative_l2_dataset_variable(y_true_list, y_pred_list):
    per_sample = []
    for yt, yp in zip(y_true_list, y_pred_list):
        num = np.sum((yp - yt) ** 2)
        den = np.sum(yt ** 2) + 1e-12
        per_sample.append(float(np.sqrt(num / den)))
    per_sample = np.asarray(per_sample, dtype=np.float32)
    return float(per_sample.mean()), per_sample


def predict_full_autoregressive_with_params(model, params, samples, y_mu, y_std, cfg):
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
            if cfg.hard_noslip:
                pred_next = _apply_noslip(pred_next, idcs_airfoil)
            preds.append(np.asarray(pred_next))
            window = jnp.concatenate([window[:, 1:, :], pred_next[:, None, :]], axis=1)
        pred_roll = np.stack(preds, axis=1)
        pred_flat = pred_roll.reshape(pred_roll.shape[0], -1)
        pred_denorm = pred_flat * y_std[None, :] + y_mu[None, :]
        if cfg.hard_noslip and s["idcs_airfoil"].size > 0:
            yp = pred_denorm.reshape(pred_denorm.shape[0], 5, 3)
            yp[s["idcs_airfoil"]] = 0.0
            pred_denorm = yp.reshape(pred_denorm.shape[0], 15)
        preds_all.append(pred_denorm.astype(np.float32))
    return preds_all


def save_artifacts(trained, metrics, y_true, y_pred, fx_mu, fx_std, y_mu, y_std, cfg):
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
    with open(MODEL_DIR / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Save ragged arrays safely when crop changes point counts across samples.
    y_true_obj = np.empty((len(y_true),), dtype=object)
    y_pred_obj = np.empty((len(y_pred),), dtype=object)
    for i, arr in enumerate(y_true):
        y_true_obj[i] = arr
    for i, arr in enumerate(y_pred):
        y_pred_obj[i] = arr
    np.save(RESULTS_DIR / "y_true.npy", y_true_obj, allow_pickle=True)
    np.save(RESULTS_DIR / "y_pred.npy", y_pred_obj, allow_pickle=True)

    # after the existing y_true_obj / y_pred_obj block
    x_obj = np.empty((len(test_samples),), dtype=object)
    idcs_obj = np.empty((len(test_samples),), dtype=object)
    for i, s in enumerate(test_samples):
        x_obj[i] = s["x"].astype(np.float32)
        idcs_obj[i] = s["idcs_airfoil"].astype(np.int32)
    np.save(RESULTS_DIR / "x_points.npy", x_obj, allow_pickle=True)
    np.save(RESULTS_DIR / "idcs_airfoil.npy", idcs_obj, allow_pickle=True)

    np.save(RESULTS_DIR / "relative_l2_per_sample.npy", metrics["relative_L2_per_sample"])
    save_json(
        {
            "relative_L2_mean": float(metrics["relative_L2_mean"]),
            "relative_L2_std": float(metrics["relative_L2_std"]),
        },
        RESULTS_DIR / "metrics.json",
    )


def train_transolver_ar(train_samples, test_samples, cfg, wandb_run=None, y_mu=None, y_std=None, test_files=None):
    heartbeat_every = 10
    print(f"Backend: {DEVICE} ({jax.devices()[0]})")
    print(f"TRAIN_POINTS={cfg.train_points} VAL_POINTS={cfg.val_points} TF_DECAY_EPOCHS={cfg.tf_decay_epochs}")
    print(f"EPOCHS={cfg.epochs}")

    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)
    rng_np = np.random.default_rng(cfg.seed)

    model = build_transolver(cfg)
    s0, _ = subsample_sample(
        train_samples[0],
        cfg.train_points,
        rng_np,
        cfg,
        cache_dir=CACHE_DIR / "subsets",
        cache_tag="init",
    )
    x0 = jnp.asarray(s0["x"])[None]
    fx0 = jnp.asarray(s0["fx"])[None]

    t_init = time.time()
    params = model.init(init_rng, x=x0, fx=fx0)
    print(f"Model init done in {time.time() - t_init:.2f}s")

    steps_per_epoch = len(train_samples)
    schedule = make_schedule(cfg, steps_per_epoch)
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.add_decayed_weights(cfg.weight_decay),
        optax.adam(schedule),
    )
    opt_state = tx.init(params)
    global_step = 0

    def rollout_loss_and_mae(p, x, fx_init, y_rollout, idcs_airfoil, tf_mask):
        v_in, dist = _split_fx(fx_init[0])
        target = jnp.reshape(y_rollout[0], (-1, OUT_FRAMES, VEL_DIM))
        point_mask = _loss_point_mask(v_in.shape[0], idcs_airfoil, cfg.exclude_airfoil_from_loss)

        preds = []
        window = v_in
        for t in range(OUT_FRAMES):
            fx_t = _build_fx_from_window(window, dist)
            pred_next = model.apply(p, x=x[0][None], fx=fx_t[None])[0]
            if cfg.hard_noslip:
                pred_next = _apply_noslip(pred_next, idcs_airfoil)
            preds.append(pred_next)
            gt_next = target[:, t, :]
            use_gt = tf_mask[t]
            next_for_window = use_gt * gt_next + (1.0 - use_gt) * pred_next
            if cfg.hard_noslip:
                next_for_window = _apply_noslip(next_for_window, idcs_airfoil)
            window = jnp.concatenate([window[:, 1:, :], next_for_window[:, None, :]], axis=1)

        pred_roll = jnp.stack(preds, axis=1)
        loss = relative_l2_rollout(pred_roll, target, mask=point_mask)
        m = mae_rollout(pred_roll, target, mask=point_mask)
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
        "crop_stats": [],
        "subsample_stats": [],
    }

    print(f"Training TransolverAR | N_train={len(train_samples)} N_test={len(test_samples)} epochs={cfg.epochs}")
    first_step_timed = False

    for ep in range(cfg.epochs):
        t0 = time.time()
        tf_p = teacher_forcing_prob(ep, cfg)
        history["tf_prob"].append(tf_p)

        train_idx = rng_np.permutation(len(train_samples))
        t_losses, t_maes = [], []
        epoch_subsample_stats = []

        for i, idx in enumerate(train_idx):
            base_sample = train_samples[int(idx)]
            aug_sample = augment_sample(base_sample, rng_np, cfg)
            s, sub_stats = subsample_sample(
                aug_sample,
                cfg.train_points,
                rng_np,
                cfg,
                cache_dir=CACHE_DIR / "subsets_train",
                cache_tag=f"train_ep{ep}",
            )
            epoch_subsample_stats.append(sub_stats)

            x = jnp.asarray(s["x"])[None]
            fx_init = jnp.asarray(s["fx"])[None]
            y_rollout = jnp.asarray(s["y"])[None]
            idcs_airfoil = jnp.asarray(s["idcs_airfoil"], dtype=jnp.int32)
            tf_mask_np = (rng_np.random(OUT_FRAMES) < tf_p).astype(np.float32)
            tf_mask = jnp.asarray(tf_mask_np)

            if not first_step_timed:
                tc = time.time()
                params, opt_state, l, m = train_step(params, opt_state, x, fx_init, y_rollout, idcs_airfoil, tf_mask)
                print(f"First train_step (compile+execute) took {time.time() - tc:.2f}s")
                first_step_timed = True
            else:
                params, opt_state, l, m = train_step(params, opt_state, x, fx_init, y_rollout, idcs_airfoil, tf_mask)

            t_losses.append(float(l))
            t_maes.append(float(m))
            global_step += 1
            if (i + 1) % heartbeat_every == 0 or (i + 1) == len(train_idx):
                print(f"[epoch {ep:03d}] train {i+1}/{len(train_idx)} (elapsed {time.time()-t0:.1f}s)")

        val_idx = np.arange(len(test_samples))
        v_losses, v_maes = [], []
        val_sub_stats = []
        for j, idx in enumerate(val_idx):
            val_rng = np.random.default_rng(cfg.seed + 10_000 + int(idx)) if cfg.deterministic_val_subsets else rng_np
            s, sub_stats = subsample_sample(
                test_samples[int(idx)],
                cfg.val_points,
                val_rng,
                cfg,
                cache_dir=CACHE_DIR / "subsets_val",
                cache_tag=f"val_fixed_{idx}",
            )
            val_sub_stats.append(sub_stats)
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

        history["train_subsampled_relative_l2"].append(tl)
        history["train_subsampled_mae"].append(tm)
        history["val_subsampled_relative_l2"].append(vl)
        history["val_subsampled_mae"].append(vm)
        history["lr"].append(lr)
        history["time"].append(dt)
        history["subsample_stats"].append({"train": epoch_subsample_stats[:10], "val": val_sub_stats[:10]})

        full_rel_l2_mean = None
        full_rel_l2_std = None
        if y_mu is not None and y_std is not None and (ep % cfg.full_eval_every == 0):
            y_pred_full = predict_full_autoregressive_with_params(model, params, test_samples, y_mu, y_std, cfg)
            y_true_full = [
                s["y"].astype(np.float32) * y_std[None, :] + y_mu[None, :]
                for s in test_samples
            ]
            full_rel_l2_mean, full_rel_l2_per_sample = relative_l2_dataset_variable(y_true_full, y_pred_full)
            full_rel_l2_std = float(full_rel_l2_per_sample.std())
            history["val_full_relative_L2_mean"].append(float(full_rel_l2_mean))
            history["val_full_relative_L2_std"].append(float(full_rel_l2_std))
            if (ep % cfg.save_val_per_sample_every == 0):
                names = [p.name for p in test_files] if test_files is not None else None
                save_per_sample_metrics(full_rel_l2_per_sample, RESULTS_DIR / f"val_per_sample_epoch_{ep:03d}.csv", names)
        else:
            history["val_full_relative_L2_mean"].append(None)
            history["val_full_relative_L2_std"].append(None)

        print(f"[{ep:03d}] {dt:.1f}s lr={lr:.2e} tf_p={tf_p:.3f} train_sub_l2={tl:.4f} val_sub_l2={vl:.4f}")
        if full_rel_l2_mean is not None:
            print(f"         full_val_relL2={full_rel_l2_mean:.4f} +/- {full_rel_l2_std:.4f}")

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
                "train_points": cfg.train_points,
                "val_points": cfg.val_points,
            }
            if full_rel_l2_mean is not None:
                log_dict["val_full/relative_L2_mean"] = float(full_rel_l2_mean)
                log_dict["val_full/relative_L2_std"] = float(full_rel_l2_std)
            wandb_run.log(log_dict, step=ep)

    return {"model": model, "params": params, "history": history}


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    seed_all(cfg.seed)
    DATA_DIR = Path(__file__).resolve().parent.parent / "data/warped-ifw"

    run = None
    if USE_WANDB:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"TransolverAR-{args.profile}",
            config=asdict(cfg),
        )

    print("Loading Warped-IFW samples...")
    train_samples, test_samples, train_files, test_files = load_split(data_dir=DATA_DIR, test_frac=cfg.test_frac, seed=cfg.seed)

    if cfg.max_train_files is not None:
        train_samples = train_samples[: cfg.max_train_files]
        train_files = train_files[: cfg.max_train_files]
    if cfg.max_test_files is not None:
        test_samples = test_samples[: cfg.max_test_files]
        test_files = test_files[: cfg.max_test_files]

    print(f"Run dataset sizes -> Train files: {len(train_files)}  Test files: {len(test_files)}")

    print("Preprocessing samples...")
    train_samples, train_crop_stats, _ = preprocess_samples(train_samples, cfg, "train", cache_dir=CACHE_DIR / "preprocessed")
    test_samples, test_crop_stats, _ = preprocess_samples(test_samples, cfg, "test", cache_dir=CACHE_DIR / "preprocessed")

    print("Normalizing inputs/targets...")
    train_samples, test_samples, fx_mu, fx_std = normalize_fx(train_samples, test_samples)
    train_samples, test_samples, y_mu, y_std = normalize_y(train_samples, test_samples)

    trained = train_transolver_ar(
        train_samples=train_samples,
        test_samples=test_samples,
        cfg=cfg,
        wandb_run=run,
        y_mu=y_mu,
        y_std=y_std,
        test_files=test_files,
    )

    print("Running full-point autoregressive inference...")
    y_pred = predict_full_autoregressive_with_params(trained["model"], trained["params"], test_samples, y_mu, y_std, cfg)
    y_true = [
        s["y"].astype(np.float32) * y_std[None, :] + y_mu[None, :]
        for s in test_samples
    ]
    mean_rel_l2, rel_l2_per_sample = relative_l2_dataset_variable(y_true, y_pred)
    metrics = {
        "relative_L2_mean": float(mean_rel_l2),
        "relative_L2_std": float(rel_l2_per_sample.std()),
        "relative_L2_per_sample": rel_l2_per_sample,
    }

    print("\nFinal test metrics:")
    print(f"  relative_L2_mean: {metrics['relative_L2_mean']:.6f}")
    print(f"  relative_L2_std : {metrics['relative_L2_std']:.6f}")

    if run is not None:
        run.log({
            "final_full/relative_L2_mean": metrics["relative_L2_mean"],
            "final_full/relative_L2_std": metrics["relative_L2_std"],
        })
        table = wandb.Table(columns=["sample_idx", "file", "rel_l2"])
        for i, v in enumerate(rel_l2_per_sample):
            table.add_data(i, test_files[i].name, float(v))
        run.log({"final_full/per_sample_rel_l2": table})
        run.finish()

    save_artifacts(trained, metrics, y_true, y_pred, fx_mu, fx_std, y_mu, y_std, cfg)
    save_json(
        {
            "train_crop_stats": train_crop_stats,
            "test_crop_stats": test_crop_stats,
        },
        RESULTS_DIR / "preprocessing_stats.json",
    )
    if args.config_out is not None:
        save_config(cfg, Path(args.config_out))


if __name__ == "__main__":
    main()

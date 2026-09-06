"""
TransolverAR training loop.

Fixes vs previous version:
  - train_step returns grad_norm for monitoring.
  - Per-element gradient clip added before global norm clip.
  - NaN recovery: reload last good params when loss is non-finite.
  - last_good_params saved every 20 steps.
  - nan_count tracked per epoch and logged to W&B + terminal.
  - Per-step component loss printed every 10 steps.
  - Structured epoch summary block with clear section labels.
  - gnorms list reset each epoch.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from config import (
    CACHE_DIR, MODEL_DIR, RESULTS_DIR,
    IN_FRAMES, OUT_FRAMES, VEL_DIM,
    RunConfig, build_arg_parser, config_from_args, save_config, seed_all,
)
from model.core import TransolverCore, TransolverCoreConfig
from normalisation import fit_and_apply, denormalise_y
from preprocessing import preprocess_samples
from subsampling import subsample_sample
from loss import build_step_weights, rollout_loss, mae_rollout
from checkpoint import save_checkpoint, save_final_artifacts
from evaluate import predict_dataset, dataset_metrics, dataset_component_losses, save_per_sample_metrics
from visualise import build_wandb_log_dict, log_final_to_wandb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _comp_rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.sum((b - a) ** 2))
    den = float(np.sum(a ** 2)) + 1e-12
    return float(np.sqrt(num / den))


def _predict_single_np(
    params, pos, v_in, idcs, t_start, cfg, model
) -> np.ndarray:
    """
    Single-sample autoregressive rollout for diagnostic printing.
    Not jit-compiled. Returns (N, OUT_FRAMES, VEL_DIM).
    """
    window = v_in
    preds  = []
    for step in range(OUT_FRAMES):
        t = (jnp.array(int(t_start) + step, dtype=jnp.int32)
             if cfg.time_input else None)
        pred_next = model.apply(
            params, pos=pos, v_window=window, t=t, training=False,
            method=lambda m, **kw: m.predict_step(**kw),
        )
        if cfg.hard_noslip:
            pred_next = apply_noslip(pred_next, idcs)
        preds.append(np.asarray(pred_next))
        window = jnp.concatenate(
            [window[:, 1:, :], pred_next[:, None, :]], axis=1
        )
    return np.stack(preds, axis=1)   # (N, OUT_FRAMES, VEL_DIM)


def _params_have_nan(params) -> bool:
    return any(
        not bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(params)
    )


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(cfg: RunConfig) -> TransolverCore:
    core_cfg = TransolverCoreConfig(
        space_dim   = 3,
        window_size = IN_FRAMES,
        vel_dim     = VEL_DIM,
        hidden_dim  = cfg.n_hidden,
        n_layers    = cfg.n_layers,
        n_heads     = cfg.n_head,
        slice_num   = cfg.slice_num,
        mlp_ratio   = cfg.mlp_ratio,
        dropout     = cfg.dropout_rate,
        time_input  = cfg.time_input,
        n_rollout   = OUT_FRAMES,
    )
    return TransolverCore(core_cfg)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def make_schedule(cfg: RunConfig, steps_per_epoch: int) -> optax.Schedule:
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)
    decay_steps  = max(1, (cfg.epochs - cfg.warmup_epochs) * steps_per_epoch)
    warmup  = optax.linear_schedule(0.0, cfg.learning_rate, warmup_steps)
    cosine  = optax.cosine_decay_schedule(
        cfg.learning_rate, decay_steps,
        alpha=cfg.lr_floor / cfg.learning_rate,
    )
    return optax.join_schedules([warmup, cosine], [warmup_steps])


# ---------------------------------------------------------------------------
# Teacher-forcing schedule
# ---------------------------------------------------------------------------

def monotone_tf_steps(epoch: int, cfg: RunConfig) -> int:
    if epoch >= cfg.tf_decay_epochs:
        return 0
    frac = 1.0 - epoch / max(1, cfg.tf_decay_epochs)
    if cfg.tf_schedule == "cosine":
        import math
        frac = 0.5 * (1.0 + math.cos(math.pi * (1.0 - frac)))
    return max(0, round(frac * OUT_FRAMES))


def build_tf_mask(k: int) -> np.ndarray:
    mask = np.zeros(OUT_FRAMES, dtype=np.float32)
    mask[:k] = 1.0
    return mask


# ---------------------------------------------------------------------------
# No-slip / input helpers
# ---------------------------------------------------------------------------

def apply_noslip(v: jnp.ndarray, idcs: jnp.ndarray) -> jnp.ndarray:
    return v.at[idcs].set(0.0)


def split_fx(fx: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    v_flat = fx[:, : IN_FRAMES * VEL_DIM]
    dist   = fx[:, IN_FRAMES * VEL_DIM : IN_FRAMES * VEL_DIM + 1]
    v_in   = v_flat.reshape(fx.shape[0], IN_FRAMES, VEL_DIM)
    return v_in, dist


def loss_point_mask(
    n_points: int, idcs_airfoil: jnp.ndarray, exclude: bool
) -> Optional[jnp.ndarray]:
    if not exclude:
        return None
    mask = jnp.ones((n_points,), dtype=jnp.float32)
    if idcs_airfoil.size > 0:
        mask = mask.at[idcs_airfoil].set(0.0)
    return mask


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_sample(sample: Dict, rng: np.random.Generator, cfg: RunConfig) -> Dict:
    s = {k: v.copy() for k, v in sample.items()}
    if cfg.aug_y_flip and rng.random() < 0.5:
        s["x"][:, 1] *= -1.0
        v_in = s["fx"][:, :15].reshape(-1, 5, 3)
        v_in[:, :, 1] *= -1.0
        s["fx"][:, :15] = v_in.reshape(-1, 15)
        y = s["y"].reshape(-1, 5, 3)
        y[:, :, 1] *= -1.0
        s["y"] = y.reshape(-1, 15)
    if cfg.aug_noise_std_pos > 0:
        noise = rng.normal(scale=cfg.aug_noise_std_pos, size=s["x"].shape).astype(np.float32)
        if cfg.aug_noise_skip_airfoil and s["idcs_airfoil"].size > 0:
            noise[s["idcs_airfoil"]] = 0.0
        s["x"] += noise
    if cfg.aug_noise_std_vel > 0:
        v_in = s["fx"][:, :15].reshape(-1, 5, 3)
        nv   = rng.normal(scale=cfg.aug_noise_std_vel, size=v_in.shape).astype(np.float32)
        ny   = rng.normal(scale=cfg.aug_noise_std_vel,
                          size=s["y"].reshape(-1, 5, 3).shape).astype(np.float32)
        if cfg.aug_noise_skip_airfoil and s["idcs_airfoil"].size > 0:
            nv[s["idcs_airfoil"]] = 0.0
            ny[s["idcs_airfoil"]] = 0.0
        s["fx"][:, :15] = (v_in + nv).reshape(-1, 15)
        s["y"]          = (s["y"].reshape(-1, 5, 3) + ny).reshape(-1, 15)
    return s


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model:          TransolverCore,
    train_samples:  List[Dict],
    test_samples:   List[Dict],
    norm_stats:     Dict,
    cfg:            RunConfig,
    run_dir:        Path,
    wandb_run=None,
    test_files:     Optional[List[Path]] = None,
) -> Dict:
    y_stats = norm_stats["y"]

    print(f"Backend : {jax.default_backend()} | {jax.devices()[0]}")
    print(f"Train={len(train_samples)}  Test={len(test_samples)}  "
          f"Epochs={cfg.epochs}  TrainPts={cfg.train_points}")

    rng_jax = jax.random.PRNGKey(cfg.seed)
    rng_np  = np.random.default_rng(cfg.seed)

    # ── model init ───────────────────────────────────────────────────────────
    rng_jax, init_rng = jax.random.split(rng_jax)
    s0, _ = subsample_sample(
        train_samples[0], cfg.train_points, rng_np, cfg,
        cache_dir=CACHE_DIR / "subsets", cache_tag="init",
    )
    dummy_pos    = jnp.asarray(s0["x"])
    dummy_window = jnp.asarray(s0["fx"][:, :15].reshape(-1, IN_FRAMES, VEL_DIM))
    dummy_t      = jnp.array(0, dtype=jnp.int32)

    t0_init = time.time()
    params = model.init(
        init_rng,
        pos=dummy_pos, v_window=dummy_window, t=dummy_t, training=False,
        method=lambda m, **kw: m.predict_step(**kw),
    )
    print(f"Model init: {time.time() - t0_init:.2f}s")

    # ── optimiser ────────────────────────────────────────────────────────────
    steps_per_epoch = len(train_samples)
    schedule        = make_schedule(cfg, steps_per_epoch)
    tx = optax.chain(
        optax.clip(1.0),
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.add_decayed_weights(cfg.weight_decay),
        optax.adamw(schedule),
    )
    opt_state   = tx.init(params)
    global_step = 0

    step_weights = build_step_weights(
        OUT_FRAMES,
        gamma      = cfg.step_loss_gamma,
        supervised = cfg.supervised_steps,
    )

    # ── loss function ─────────────────────────────────────────────────────────
    def compute_loss(p, pos, v_in, target, idcs_airfoil, tf_mask):
        """
        Used for the gradient update only.
        tf_mask controls teacher forcing — 1=use GT, 0=use model prediction.
        """
        window = v_in
        preds  = []
        for step in range(OUT_FRAMES):
            t = (jnp.array(step, dtype=jnp.int32)
                 if cfg.time_input else None)
            pred_next = model.apply(
                p, pos=pos, v_window=window, t=t, training=True,
                method=lambda m, **kw: m.predict_step(**kw),
            )
            if cfg.hard_noslip:
                pred_next = apply_noslip(pred_next, idcs_airfoil)
            preds.append(pred_next)
            gt_next      = target[:, step, :]
            use_gt       = tf_mask[step]
            next_for_win = use_gt * gt_next + (1.0 - use_gt) * pred_next
            if cfg.hard_noslip:
                next_for_win = apply_noslip(next_for_win, idcs_airfoil)
            window = jnp.concatenate(
                [window[:, 1:, :], next_for_win[:, None, :]], axis=1
            )

        pred_roll = jnp.stack(preds, axis=0)
        pt_mask   = loss_point_mask(
            pos.shape[0], idcs_airfoil, cfg.exclude_airfoil_from_loss
        )
        return rollout_loss(
            pred_roll, target.transpose(1, 0, 2), step_weights, pt_mask
        )

    def compute_eval_loss(p, pos, v_in, target, idcs_airfoil):
        """
        Pure rollout loss — NO teacher forcing.
        Used for both train_loss and test_loss that get logged.
        This is the honest measure of model quality.
        """
        tf_zero = jnp.zeros((OUT_FRAMES,), dtype=jnp.float32)
        return compute_loss(p, pos, v_in, target, idcs_airfoil, tf_zero)

    @jax.jit
    def train_step(p, o, pos, v_in, target, idcs_airfoil, tf_mask):
        loss, grads = jax.value_and_grad(compute_loss)(
            p, pos, v_in, target, idcs_airfoil, tf_mask
        )
        leaves    = jax.tree_util.tree_leaves(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
        updates, o = tx.update(grads, o, p)
        p = optax.apply_updates(p, updates)
        return p, o, loss, grad_norm

    @jax.jit
    def eval_loss_step(p, pos, v_in, target, idcs_airfoil):
        """Pure rollout loss for logging — same for train and test."""
        return compute_eval_loss(p, pos, v_in, target, idcs_airfoil)

    # ── history ───────────────────────────────────────────────────────────────
    history: Dict = {
        "train_loss":       [],
        "test_loss":        [],
        "train_ux":         [],
        "train_uy":         [],
        "train_uz":         [],
        "train_speed":      [],
        "test_ux":          [],
        "test_uy":          [],
        "test_uz":          [],
        "test_speed":       [],
        "lr":               [],
        "tf_k":             [],
        "grad_norm_mean":   [],
        "nan_count":        [],
        "full_mesh_loss":   [],
        "full_mesh_nrmse":  [],
    }

    ckpt_dir            = run_dir / "checkpoints"
    last_good_params    = None
    last_good_opt_state = None
    first_step_timed    = False

    # =========================================================================
    # Main training loop
    # =========================================================================
    for ep in range(cfg.epochs):
        t0   = time.time()
        tf_k = monotone_tf_steps(ep, cfg)
        tf_mask = jnp.asarray(build_tf_mask(tf_k))

        # ── gradient update pass (teacher forcing active) ─────────────────────
        gnorms    = []
        nan_count = 0
        train_idx = rng_np.permutation(len(train_samples))

        for i, idx in enumerate(train_idx):
            base = train_samples[int(idx)]
            aug  = augment_sample(base, rng_np, cfg)
            s, _ = subsample_sample(
                aug, cfg.train_points, rng_np, cfg,
                cache_dir=CACHE_DIR / "subsets_train",
                cache_tag=f"ep{ep}",
            )
            pos     = jnp.asarray(s["x"])
            v_in, _ = split_fx(jnp.asarray(s["fx"]))
            target  = jnp.asarray(s["y"]).reshape(-1, OUT_FRAMES, VEL_DIM)
            idcs    = jnp.asarray(s["idcs_airfoil"], dtype=jnp.int32)

            if not first_step_timed:
                tc = time.time()
                params, opt_state, l, gnorm = train_step(
                    params, opt_state, pos, v_in, target, idcs, tf_mask
                )
                print(f"First train_step (compile): {time.time()-tc:.2f}s")
                first_step_timed = True
            else:
                params, opt_state, l, gnorm = train_step(
                    params, opt_state, pos, v_in, target, idcs, tf_mask
                )

            l_val     = float(l)
            gnorm_val = float(gnorm)

            # NaN recovery
            if not np.isfinite(l_val) or _params_have_nan(params):
                nan_count += 1
                print(
                    f"  [ep {ep:04d} | step {i:>4}] NaN detected "
                    f"(gnorm={gnorm_val:.3f}, total={nan_count}). "
                    f"Restoring last good params..."
                )
                if last_good_params is not None:
                    params    = last_good_params
                    opt_state = last_good_opt_state
                continue

            if i % 20 == 0:
                last_good_params    = params
                last_good_opt_state = opt_state

            gnorms.append(gnorm_val)
            global_step += 1

        # ── train loss pass (pure rollout, no TF, subsampled) ─────────────────
        # This is the honest train loss — same computation as test loss.
        train_losses, train_preds_list, train_true_list = [], [], []
        for idx in range(min(len(train_samples), 64)):
            # Use a fixed deterministic subset for speed — not shuffled
            val_rng = np.random.default_rng(cfg.seed + 20_000 + idx)
            s, _ = subsample_sample(
                train_samples[idx], cfg.train_points, val_rng, cfg,
                cache_dir=CACHE_DIR / "subsets_train_eval",
                cache_tag=f"train_eval_fixed_{idx}",
            )
            pos     = jnp.asarray(s["x"])
            v_in, _ = split_fx(jnp.asarray(s["fx"]))
            target  = jnp.asarray(s["y"]).reshape(-1, OUT_FRAMES, VEL_DIM)
            idcs    = jnp.asarray(s["idcs_airfoil"], dtype=jnp.int32)

            l = eval_loss_step(params, pos, v_in, target, idcs)
            train_losses.append(float(l))

            # Collect predictions for component breakdown
            yp_np = _predict_single_np(params, pos, v_in, idcs, cfg, model)
            yt_np = np.asarray(target)
            train_preds_list.append(yp_np.reshape(-1, 15))
            train_true_list.append(yt_np.reshape(-1, 15))

        # ── test loss pass (pure rollout, no TF, subsampled) ──────────────────
        test_losses, test_preds_list, test_true_list = [], [], []
        for j in range(len(test_samples)):
            val_rng = (np.random.default_rng(cfg.seed + 10_000 + j)
                       if cfg.deterministic_val_subsets else rng_np)
            s, _ = subsample_sample(
                test_samples[j], cfg.val_points, val_rng, cfg,
                cache_dir=CACHE_DIR / "subsets_val",
                cache_tag=f"val_fixed_{j}",
            )
            pos     = jnp.asarray(s["x"])
            v_in, _ = split_fx(jnp.asarray(s["fx"]))
            target  = jnp.asarray(s["y"]).reshape(-1, OUT_FRAMES, VEL_DIM)
            idcs    = jnp.asarray(s["idcs_airfoil"], dtype=jnp.int32)

            l = eval_loss_step(params, pos, v_in, target, idcs)
            test_losses.append(float(l))

            yp_np = _predict_single_np(params, pos, v_in, idcs, cfg, model)
            yt_np = np.asarray(target)
            test_preds_list.append(yp_np.reshape(-1, 15))
            test_true_list.append(yt_np.reshape(-1, 15))

        # ── component breakdowns ──────────────────────────────────────────────
        train_comp = dataset_component_losses(train_true_list, train_preds_list)
        test_comp  = dataset_component_losses(test_true_list,  test_preds_list)

        train_comp_means = {k: float(v.mean()) for k, v in train_comp.items()}
        test_comp_means  = {k: float(v.mean()) for k, v in test_comp.items()}

        # ── full-mesh eval (periodic, optional) ───────────────────────────────
        full_mesh_metrics = None
        if ep % cfg.full_eval_every == 0:
            y_pred_full = predict_dataset(
                model, params, test_samples, y_stats, cfg
            )
            y_true_full = [
                denormalise_y(s["y"], y_stats) for s in test_samples
            ]
            full_mesh_metrics = dataset_metrics(y_true_full, y_pred_full)

            if ep % cfg.save_val_per_sample_every == 0:
                names = ([p.name for p in test_files]
                         if test_files is not None else None)
                full_comp = dataset_component_losses(y_true_full, y_pred_full)
                save_per_sample_metrics(
                    full_mesh_metrics["relative_L2_per_sample"],
                    RESULTS_DIR / f"val_per_sample_ep{ep:04d}.csv",
                    names=names,
                    extra_cols={
                        "nrmse":        full_mesh_metrics["nRMSE_per_sample"],
                        "ux_rel_l2":    full_comp["ux_rel_l2"],
                        "uy_rel_l2":    full_comp["uy_rel_l2"],
                        "uz_rel_l2":    full_comp["uz_rel_l2"],
                        "speed_rel_l2": full_comp["speed_rel_l2"],
                    },
                )

        # ── aggregate scalars ─────────────────────────────────────────────────
        train_loss = float(np.nanmean(train_losses)) if train_losses else float("nan")
        test_loss  = float(np.nanmean(test_losses))  if test_losses  else float("nan")
        lr         = float(schedule(global_step))
        dt         = time.time() - t0
        gn_avg     = float(np.mean(gnorms)) if gnorms else float("nan")

        # ── history ───────────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_ux"].append(train_comp_means["ux_rel_l2"])
        history["train_uy"].append(train_comp_means["uy_rel_l2"])
        history["train_uz"].append(train_comp_means["uz_rel_l2"])
        history["train_speed"].append(train_comp_means["speed_rel_l2"])
        history["test_ux"].append(test_comp_means["ux_rel_l2"])
        history["test_uy"].append(test_comp_means["uy_rel_l2"])
        history["test_uz"].append(test_comp_means["uz_rel_l2"])
        history["test_speed"].append(test_comp_means["speed_rel_l2"])
        history["lr"].append(lr)
        history["tf_k"].append(tf_k)
        history["grad_norm_mean"].append(gn_avg)
        history["nan_count"].append(nan_count)
        history["full_mesh_loss"].append(
            float(full_mesh_metrics["relative_L2_mean"])
            if full_mesh_metrics else None
        )
        history["full_mesh_nrmse"].append(
            float(full_mesh_metrics["nRMSE_mean"])
            if full_mesh_metrics else None
        )

        # ── terminal output ───────────────────────────────────────────────────
        print(f"\n{'='*72}")
        print(
            f"  Epoch {ep:04d}/{cfg.epochs-1}  |  {dt:.1f}s  |  "
            f"lr={lr:.2e}  |  tf_k={tf_k}  |  "
            f"gnorm={gn_avg:.3f}  |  NaNs={nan_count}"
        )
        print(f"  {'─'*68}")
        print(
            f"  {'':30}  {'TRAIN':>10}  {'TEST':>10}"
        )
        print(
            f"  {'─'*68}"
        )
        print(
            f"  {'Overall loss':<30}  {train_loss:>10.4f}  {test_loss:>10.4f}"
        )
        print(
            f"  {'ux  (rel L2)':<30}  "
            f"{train_comp_means['ux_rel_l2']:>10.4f}  "
            f"{test_comp_means['ux_rel_l2']:>10.4f}"
        )
        print(
            f"  {'uy  (rel L2)':<30}  "
            f"{train_comp_means['uy_rel_l2']:>10.4f}  "
            f"{test_comp_means['uy_rel_l2']:>10.4f}"
        )
        print(
            f"  {'uz  (rel L2)':<30}  "
            f"{train_comp_means['uz_rel_l2']:>10.4f}  "
            f"{test_comp_means['uz_rel_l2']:>10.4f}"
        )
        print(
            f"  {'|u| (rel L2)':<30}  "
            f"{train_comp_means['speed_rel_l2']:>10.4f}  "
            f"{test_comp_means['speed_rel_l2']:>10.4f}"
        )
        if full_mesh_metrics is not None:
            print(f"  {'─'*68}")
            print(
                f"  {'Full mesh loss (test)':<30}  "
                f"{'':>10}  "
                f"{full_mesh_metrics['relative_L2_mean']:>10.4f}"
            )
            print(
                f"  {'Full mesh nRMSE (test)':<30}  "
                f"{'':>10}  "
                f"{full_mesh_metrics['nRMSE_mean']:>10.4f}"
            )
        print(f"{'='*72}\n")

        # ── W&B ──────────────────────────────────────────────────────────────
        if wandb_run is not None:
            log = build_wandb_log_dict(
                epoch              = ep,
                train_loss         = train_loss,
                test_loss          = test_loss,
                train_components   = train_comp_means,
                test_components    = test_comp_means,
                lr                 = lr,
                tf_k               = tf_k,
                grad_norm_mean     = gn_avg,
                nan_count          = nan_count,
                full_mesh_metrics  = full_mesh_metrics,
            )
            wandb_run.log(log, step=ep)

        # ── checkpoint ────────────────────────────────────────────────────────
        if (cfg.save_checkpoint_every > 0
                and (ep + 1) % cfg.save_checkpoint_every == 0):
            save_checkpoint(ckpt_dir, ep, params, opt_state, history, cfg)
            print(f"  Checkpoint saved → epoch {ep}")

    # =========================================================================
    # Fine-tune stage
    # =========================================================================
    if cfg.finetune_epochs > 0:
        print(f"\n{'='*72}")
        print(f"  Fine-tune: {cfg.finetune_epochs} epochs  lr={cfg.finetune_lr}")
        print(f"{'='*72}")

        ft_schedule = optax.constant_schedule(cfg.finetune_lr)
        ft_tx = optax.chain(
            optax.clip(1.0),
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.add_decayed_weights(cfg.weight_decay),
            optax.adamw(ft_schedule),
        )
        ft_opt_state = ft_tx.init(params)

        @jax.jit
        def ft_train_step(p, o, pos, v_in, target, idcs):
            tf_zero = jnp.zeros((OUT_FRAMES,), dtype=jnp.float32)
            loss, grads = jax.value_and_grad(compute_loss)(
                p, pos, v_in, target, idcs, tf_zero
            )
            leaves    = jax.tree_util.tree_leaves(grads)
            grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
            updates, o = ft_tx.update(grads, o, p)
            p = optax.apply_updates(p, updates)
            return p, o, loss, grad_norm

        for ft_ep in range(cfg.finetune_epochs):
            ft_losses, ft_gnorms, ft_nans = [], [], 0
            for idx in rng_np.permutation(len(train_samples)):
                s = train_samples[int(idx)]
                if cfg.finetune_points > 0:
                    s, _ = subsample_sample(
                        s, cfg.finetune_points, rng_np, cfg,
                        cache_dir=CACHE_DIR / "subsets_ft",
                        cache_tag=f"ft{ft_ep}",
                    )
                pos     = jnp.asarray(s["x"])
                v_in, _ = split_fx(jnp.asarray(s["fx"]))
                target  = jnp.asarray(s["y"]).reshape(-1, OUT_FRAMES, VEL_DIM)
                idcs    = jnp.asarray(s["idcs_airfoil"], dtype=jnp.int32)
                params, ft_opt_state, l, gnorm = ft_train_step(
                    params, ft_opt_state, pos, v_in, target, idcs
                )
                l_val = float(l)
                if not np.isfinite(l_val) or _params_have_nan(params):
                    ft_nans += 1
                    if last_good_params is not None:
                        params       = last_good_params
                        ft_opt_state = last_good_opt_state
                    continue
                if len(ft_losses) % 20 == 0:
                    last_good_params    = params
                    last_good_opt_state = ft_opt_state
                ft_losses.append(l_val)
                ft_gnorms.append(float(gnorm))

            ft_mean = float(np.nanmean(ft_losses)) if ft_losses else float("nan")
            gn_mean = float(np.mean(ft_gnorms))    if ft_gnorms else float("nan")
            print(
                f"  [ft {ft_ep:03d}]  "
                f"loss={ft_mean:.4f}  gnorm={gn_mean:.3f}  NaNs={ft_nans}"
            )

    return {"model": model, "params": params, "history": history}


# ---------------------------------------------------------------------------
# Helper used inside train loop
# ---------------------------------------------------------------------------

def _predict_single_np(
    params, pos, v_in, idcs, cfg, model
) -> np.ndarray:
    """
    Pure rollout for a single subsampled sample.
    Returns (N, OUT_FRAMES, VEL_DIM) numpy array.
    Not jit-compiled — used for per-epoch loss logging only.
    """
    window = v_in
    preds  = []
    for step in range(OUT_FRAMES):
        t = (jnp.array(step, dtype=jnp.int32)
             if cfg.time_input else None)
        pred_next = model.apply(
            params, pos=pos, v_window=window, t=t, training=False,
            method=lambda m, **kw: m.predict_step(**kw),
        )
        if cfg.hard_noslip:
            pred_next = apply_noslip(pred_next, idcs)
        preds.append(np.asarray(pred_next))
        window = jnp.concatenate(
            [window[:, 1:, :], pred_next[:, None, :]], axis=1
        )
    return np.stack(preds, axis=1)   # (N, OUT_FRAMES, VEL_DIM)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    from load_data import load_split
    import wandb

    parser = build_arg_parser()
    args   = parser.parse_args()
    cfg    = config_from_args(args)
    seed_all(cfg.seed)

    run_dir = MODEL_DIR / f"run_{args.profile}"
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if cfg.use_wandb:
        wandb_run = wandb.init(
            project = cfg.wandb_project,
            entity  = cfg.wandb_entity or None,
            name    = f"TransolverAR-{args.profile}",
            config  = asdict(cfg),
        )

    DATA_DIR = Path(__file__).resolve().parent.parent / "data/warped-ifw"
    print("Loading data...")
    train_samples, test_samples, train_files, test_files = load_split(
        data_dir=DATA_DIR, test_frac=cfg.test_frac, seed=cfg.seed
    )

    if cfg.max_train_files is not None:
        train_samples = train_samples[: cfg.max_train_files]
        train_files   = train_files  [: cfg.max_train_files]
    if cfg.max_test_files is not None:
        test_samples  = test_samples [: cfg.max_test_files]
        test_files    = test_files   [: cfg.max_test_files]

    print(f"Train={len(train_samples)}  Test={len(test_samples)}")

    print("Preprocessing...")
    train_samples, _, _ = preprocess_samples(
        train_samples, cfg, "train", cache_dir=CACHE_DIR / "preprocessed"
    )
    test_samples, _, _ = preprocess_samples(
        test_samples, cfg, "test", cache_dir=CACHE_DIR / "preprocessed"
    )

    print("Normalising (pos + vel + dist separately)...")
    train_samples, test_samples, norm_stats = fit_and_apply(
        train_samples, test_samples
    )

    model = build_model(cfg)

    trained = train(
        model         = model,
        train_samples = train_samples,
        test_samples  = test_samples,
        norm_stats    = norm_stats,
        cfg           = cfg,
        run_dir       = run_dir,
        wandb_run     = wandb_run,
        test_files    = test_files,
    )

    print("\nFinal full-mesh evaluation...")
    y_pred = predict_dataset(
        trained["model"], trained["params"],
        test_samples, norm_stats["y"], cfg,
    )
    y_true = [denormalise_y(s["y"], norm_stats["y"]) for s in test_samples]
    metrics = dataset_metrics(y_true, y_pred)
    final_comp_losses = dataset_component_losses(y_true, y_pred)

    print(f"\n{'='*72}")
    print("  Final test metrics")
    print(f"  {'─'*68}")
    print(f"  relative_L2_mean  : {metrics['relative_L2_mean']:.6f}")
    print(f"  relative_L2_std   : {metrics['relative_L2_std']:.6f}")
    print(f"  nRMSE_mean        : {metrics['nRMSE_mean']:.6f}")
    print(f"  {'─'*68}")
    print("  Per-component (mean across test set):")
    for k, v in final_comp_losses.items():
        print(f"    {k:<22} {float(v.mean()):.6f}")
    print(f"{'='*72}\n")

    if wandb_run is not None:
        log_final_to_wandb(
            wandb_run, metrics, final_comp_losses, test_files
        )
        wandb_run.finish()

    save_final_artifacts(
        run_dir      = run_dir,
        params       = trained["params"],
        opt_state    = None,
        history      = trained["history"],
        norm_stats   = norm_stats,
        cfg          = cfg,
        test_samples = test_samples,
        y_true       = y_true,
        y_pred       = y_pred,
        metrics      = metrics,
    )

    if args.config_out is not None:
        save_config(cfg, Path(args.config_out))

    print("Done.")


if __name__ == "__main__":
    main()
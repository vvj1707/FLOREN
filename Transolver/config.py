from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
MODEL_DIR   = SCRIPT_DIR / "trained_model"
RESULTS_DIR = SCRIPT_DIR / "results"
CACHE_DIR   = SCRIPT_DIR / "cache"

# ---------------------------------------------------------------------------
# Fixed data constants  (never change between runs)
# ---------------------------------------------------------------------------
IN_FRAMES  = 5
OUT_FRAMES = 5
VEL_DIM    = 3
SPACE_DIM  = 3
# fx = [v0..v4 flat (15)] + [dist (1)]  → 16 features
FUN_DIM    = IN_FRAMES * VEL_DIM + 1
OUT_DIM    = VEL_DIM   # single-step output


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    # ---- optimisation -------------------------------------------------------
    epochs:          int   = 500
    learning_rate:   float = 2e-4
    weight_decay:    float = 1e-4
    # In RunConfig dataclass — change these three defaults:
    grad_clip: float = 0.3  # was 1.0 → tighter: per-element + global
    lr_floor: float = 1e-5  # was 1e-6 → cosine decays to 1e-5 not 1e-6
    warmup_epochs: int = 20  # was 10  → longer warmup before full lr

    # ---- model --------------------------------------------------------------
    n_layers:        int   = 6
    n_hidden:        int   = 256   # paper value
    n_head:          int   = 8
    mlp_ratio:       int   = 2     # consistent across JAX + PyTorch
    dropout_rate:    float = 0.0
    slice_num:       int   = 64
    ref:             int   = 8
    time_input:      bool  = True  # timestep conditioning now ON
    unified_pos:     bool  = False
    geometry:        str   = "irregular"

    # ---- data / run ---------------------------------------------------------
    seed:            int   = 42
    test_frac:       float = 0.20
    train_points:    int   = 20_000
    val_points:      int   = 20_000
    max_train_files: Optional[int] = None
    max_test_files:  Optional[int] = None

    # ---- teacher-forcing schedule -------------------------------------------
    # Monotone: use GT for first k steps, model output for the rest.
    # k decays linearly from OUT_FRAMES → 0 over tf_decay_epochs.
    tf_decay_epochs: int   = 250
    tf_schedule:     str   = "monotone"   # monotone | cosine | linear

    # ---- rollout loss -------------------------------------------------------
    # Pushforward: supervise only steps in supervised_steps (0-indexed).
    # Empty list  → supervise all steps (legacy behaviour).
    supervised_steps: List[int] = field(default_factory=list)
    # Per-step exponential weight  w_t = step_loss_gamma^t  (1.0 = uniform)
    step_loss_gamma:  float = 1.5

    # ---- physics ------------------------------------------------------------
    hard_noslip:              bool  = True
    exclude_airfoil_from_loss: bool = False

    # ---- preprocessing ------------------------------------------------------
    preprocess_cache_to_disk:    bool  = False
    crop_enabled:                bool  = False
    crop_max_dist:               float = 0.4
    crop_dist_feature_index:     int   = 15
    recompute_distance_if_missing: bool = True

    # ---- subsampling --------------------------------------------------------
    subsample_mode:                   str   = "uniform"
    keep_all_airfoil:                 bool  = True
    biased_sampling_near_wall_power:  float = 1.5
    biased_sampling_variance_power:   float = 1.0
    biased_sampling_mix_near_wall:    float = 0.5
    recon_k:                          int   = 4
    recon_rel_threshold:              float = 0.11
    recon_check_frames_mode:          str   = "subset"
    recon_cache_memory:               bool  = True
    recon_cache_disk:                 bool  = False
    recon_max_attempts:               int   = 12

    # ---- augmentation -------------------------------------------------------
    aug_y_flip:            bool  = False
    aug_noise_std_vel:     float = 0.0
    aug_noise_std_pos:     float = 0.0
    aug_noise_skip_airfoil: bool = True

    # ---- evaluation / checkpointing -----------------------------------------
    full_eval_every:            int  = 10
    deterministic_val_subsets:  bool = True
    save_checkpoint_every:      int  = 50   # epochs; 0 = only final
    save_val_per_sample_every:  int  = 50

    # ---- fine-tune stage ----------------------------------------------------
    # If > 0, after main training run this many epochs on full-mesh batches.
    finetune_epochs:      int   = 0
    finetune_lr:          float = 5e-5
    finetune_points:      int   = 0   # 0 = use full mesh (no subsampling)

    # ---- W&B ----------------------------------------------------------------
    use_wandb:      bool = False
    wandb_project:  str  = "transolver-ar"
    wandb_entity:   str  = ""


# ---------------------------------------------------------------------------
# Profiles  (only smoke_test and full remain)
# ---------------------------------------------------------------------------
PROFILES: Dict[str, RunConfig] = {

    # ------------------------------------------------------------------
    # smoke_test
    # Exercises every new feature on a tiny slice of data.
    # Should complete in a few minutes on a single GPU.
    # ------------------------------------------------------------------
    # smoke_test profile — update:
    "smoke_test": RunConfig(
        epochs             = 10,
        learning_rate      = 2e-4,
        warmup_epochs      = 2,
        grad_clip          = 0.3,       # was 0.5
        n_layers           = 2,
        n_hidden           = 64,
        n_head             = 4,
        mlp_ratio          = 2,
        slice_num          = 16,
        train_points       = 2_000,
        val_points         = 2_000,
        max_train_files    = 8,
        max_test_files     = 2,
        tf_decay_epochs    = 8,
        tf_schedule        = "monotone",
        supervised_steps   = [],
        step_loss_gamma    = 1.5,
        time_input         = True,
        hard_noslip        = True,
        crop_enabled       = False,
        subsample_mode     = "uniform",
        aug_y_flip         = True,
        aug_noise_std_vel  = 0.01,
        full_eval_every    = 5,
        save_checkpoint_every = 5,
        finetune_epochs    = 2,
        finetune_lr        = 5e-5,
        finetune_points    = 0,
        use_wandb          = True,
    ),

    # full profile — update:
    "full": RunConfig(
        epochs             = 500,
        learning_rate      = 2e-4,
        warmup_epochs      = 20,        # was 10
        lr_floor           = 1e-5,      # was 1e-6
        grad_clip          = 0.3,       # was 0.5
        n_layers           = 6,
        n_hidden           = 256,
        n_head             = 8,
        mlp_ratio          = 2,
        slice_num          = 64,
        train_points       = 20_000,
        val_points         = 20_000,
        max_train_files    = None,
        max_test_files     = None,
        tf_decay_epochs    = 250,
        tf_schedule        = "monotone",
        supervised_steps   = [],        # pushforward off until stable
        step_loss_gamma    = 1.5,
        time_input         = True,
        hard_noslip        = True,
        crop_enabled       = True,
        crop_max_dist      = 0.4,
        subsample_mode     = "biased",
        aug_y_flip         = True,
        aug_noise_std_vel  = 0.005,
        full_eval_every    = 10,
        save_checkpoint_every   = 50,
        save_val_per_sample_every = 50,
        finetune_epochs    = 20,
        finetune_lr        = 5e-5,
        finetune_points    = 0,
        use_wandb          = True,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seed_all(seed: int = 42) -> None:
    np.random.seed(seed)


def resolve_profile(name: str) -> RunConfig:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {sorted(PROFILES)}")
    return copy.deepcopy(PROFILES[name])


def _set_if_not_none(cfg: RunConfig, key: str, value: Any) -> None:
    if value is not None:
        setattr(cfg, key, value)


def save_config(cfg: RunConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_config(path: Path) -> RunConfig:
    with open(path) as f:
        d = json.load(f)
    return RunConfig(**d)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TransolverAR trainer")
    p.add_argument("--profile",     type=str, default="smoke_test",
                   choices=sorted(PROFILES))
    # allow any RunConfig field to be overridden from the command line
    p.add_argument("--epochs",              type=int)
    p.add_argument("--train-points",        type=int)
    p.add_argument("--val-points",          type=int)
    p.add_argument("--n-hidden",            type=int)
    p.add_argument("--n-layers",            type=int)
    p.add_argument("--n-head",              type=int)
    p.add_argument("--slice-num",           type=int)
    p.add_argument("--tf-decay-epochs",     type=int)
    p.add_argument("--tf-schedule",         type=str,
                   choices=["monotone", "cosine", "linear"])
    p.add_argument("--step-loss-gamma",     type=float)
    p.add_argument("--supervised-steps",    type=int, nargs="*")
    p.add_argument("--full-eval-every",     type=int)
    p.add_argument("--crop-enabled",        type=int, choices=[0, 1])
    p.add_argument("--crop-max-dist",       type=float)
    p.add_argument("--subsample-mode",      type=str,
                   choices=["uniform", "biased", "recon"])
    p.add_argument("--aug-y-flip",          type=int, choices=[0, 1])
    p.add_argument("--aug-noise-std-vel",   type=float)
    p.add_argument("--aug-noise-std-pos",   type=float)
    p.add_argument("--max-train-files",     type=int)
    p.add_argument("--max-test-files",      type=int)
    p.add_argument("--seed",                type=int)
    p.add_argument("--finetune-epochs",     type=int)
    p.add_argument("--finetune-lr",         type=float)
    p.add_argument("--use-wandb",           type=int, choices=[0, 1])
    p.add_argument("--config-out",          type=str, default=None)
    return p


def config_from_args(args: argparse.Namespace) -> RunConfig:
    cfg = resolve_profile(args.profile)
    _set_if_not_none(cfg, "epochs",            args.epochs)
    _set_if_not_none(cfg, "train_points",      args.train_points)
    _set_if_not_none(cfg, "val_points",        args.val_points)
    _set_if_not_none(cfg, "n_hidden",          args.n_hidden)
    _set_if_not_none(cfg, "n_layers",          args.n_layers)
    _set_if_not_none(cfg, "n_head",            args.n_head)
    _set_if_not_none(cfg, "slice_num",         args.slice_num)
    _set_if_not_none(cfg, "tf_decay_epochs",   args.tf_decay_epochs)
    _set_if_not_none(cfg, "tf_schedule",       args.tf_schedule)
    _set_if_not_none(cfg, "step_loss_gamma",   args.step_loss_gamma)
    _set_if_not_none(cfg, "full_eval_every",   args.full_eval_every)
    _set_if_not_none(cfg, "crop_max_dist",     args.crop_max_dist)
    _set_if_not_none(cfg, "subsample_mode",    args.subsample_mode)
    _set_if_not_none(cfg, "aug_noise_std_vel", args.aug_noise_std_vel)
    _set_if_not_none(cfg, "aug_noise_std_pos", args.aug_noise_std_pos)
    _set_if_not_none(cfg, "max_train_files",   args.max_train_files)
    _set_if_not_none(cfg, "max_test_files",    args.max_test_files)
    _set_if_not_none(cfg, "seed",              args.seed)
    _set_if_not_none(cfg, "finetune_epochs",   args.finetune_epochs)
    _set_if_not_none(cfg, "finetune_lr",       args.finetune_lr)
    if args.crop_enabled is not None:
        cfg.crop_enabled = bool(args.crop_enabled)
    if args.aug_y_flip is not None:
        cfg.aug_y_flip = bool(args.aug_y_flip)
    if args.use_wandb is not None:
        cfg.use_wandb = bool(args.use_wandb)
    if args.supervised_steps is not None:
        cfg.supervised_steps = args.supervised_steps
    return cfg
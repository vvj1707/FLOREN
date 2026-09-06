# `config.py`

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import numpy as np

# -------------------------
# Paths
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "trained_model"
RESULTS_DIR = SCRIPT_DIR / "results"
CACHE_DIR = SCRIPT_DIR / "cache"

# -------------------------
# Data constants
# -------------------------
IN_FRAMES = 5
OUT_FRAMES = 5
VEL_DIM = 3
SPACE_DIM = 3
FUN_DIM = IN_FRAMES * VEL_DIM + 1  # 15 + distance
OUT_DIM = 3

# -------------------------
# Model constants / defaults
# -------------------------
RANDOM_SEED = 42
DEVICE = jax.default_backend()

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

TEST_FRAC = 0.15
USE_WANDB = True
WANDB_PROJECT = "warped-ifw-transolver-ar"
WANDB_ENTITY = None

MAX_TRAIN_FILES_TINY = 32
MAX_TEST_FILES_TINY = 8


@dataclass
class RunConfig:
    # optimization
    epochs: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 1e-6
    warmup_epochs: int = 5
    lr_floor: float = 1e-6
    grad_clip: float = 1.0

    # model
    n_layers: int = N_LAYERS
    n_hidden: int = N_HIDDEN
    n_head: int = N_HEAD
    mlp_ratio: int = MLP_RATIO
    dropout_rate: float = DROPOUT_RATE
    slice_num: int = SLICE_NUM
    ref: int = REF
    time_input: bool = TIME_INPUT
    unified_pos: bool = UNIFIED_POS
    geometry: str = GEOMETRY

    # data / run
    seed: int = RANDOM_SEED
    test_frac: float = TEST_FRAC
    train_points: int = 8_000
    val_points: int = 8_000
    tf_decay_epochs: int = 10
    tf_schedule: str = "linear"  # linear|cosine|piecewise
    hard_noslip: bool = True
    exclude_airfoil_from_loss: bool = False
    full_eval_every: int = 1
    deterministic_val_subsets: bool = True

    # profiles / run shape
    tiny_subset: bool = True
    max_train_files: Optional[int] = MAX_TRAIN_FILES_TINY
    max_test_files: Optional[int] = MAX_TEST_FILES_TINY

    # preprocessing
    preprocess_in_memory: bool = True
    preprocess_cache_to_disk: bool = False
    crop_enabled: bool = False
    crop_max_dist: float = 0.4
    crop_dist_feature_index: int = 15
    recompute_distance_if_missing: bool = True

    # subsampling
    subsample_mode: str = "uniform"  # uniform|biased|recon
    keep_all_airfoil: bool = True
    biased_sampling_near_wall_power: float = 1.5
    biased_sampling_variance_power: float = 1.0
    biased_sampling_mix_near_wall: float = 0.5
    recon_check_enabled: bool = False
    recon_k: int = 4
    recon_rel_threshold: float = 0.11
    recon_check_frames_mode: str = "subset"  # subset|all|outputs_only
    recon_cache_memory: bool = True
    recon_cache_disk: bool = False
    recon_max_attempts: int = 12

    # augmentation
    aug_y_flip: bool = False
    aug_noise_std_vel: float = 0.0
    aug_noise_std_pos: float = 0.0
    aug_noise_skip_airfoil: bool = True

    # validation artifacts / diagnostics
    save_val_per_sample_every: int = 5
    log_crop_stats: bool = True
    log_noslip_error: bool = True
    log_subsample_stats: bool = True


PROFILES: Dict[str, RunConfig] = {
    "tiny": RunConfig(),
    "full": RunConfig(
        epochs=200,
        train_points=20_000,
        val_points=20_000,
        tf_decay_epochs=100,
        full_eval_every=5,
        tiny_subset=False,
        max_train_files=None,
        max_test_files=None,
        crop_enabled=True,
        crop_max_dist=0.4,
        subsample_mode="uniform",
        deterministic_val_subsets=True,
        exclude_airfoil_from_loss=False,
        aug_y_flip=True,
    ),
    "phase2": RunConfig(
        epochs=200,
        train_points=20_000,
        val_points=20_000,
        tf_decay_epochs=100,
        full_eval_every=5,
        tiny_subset=False,
        max_train_files=None,
        max_test_files=None,
        crop_enabled=True,
        crop_max_dist=0.4,
        preprocess_cache_to_disk=True,
        subsample_mode="recon",
        recon_check_enabled=True,
        recon_rel_threshold=0.11,
        recon_check_frames_mode="subset",
        recon_cache_memory=True,
        recon_cache_disk=True,
        deterministic_val_subsets=True,
        exclude_airfoil_from_loss=True,
        aug_y_flip=True,
        aug_noise_std_vel=0.0,
        aug_noise_std_pos=0.0,
    ),
    "medium_phase2": RunConfig(
        epochs=80,
        train_points=10000,
        val_points=10000,
        tf_decay_epochs=40,
        full_eval_every=5,
        tiny_subset=False,
        max_train_files=256,
        max_test_files=32,
        crop_enabled=True,
        crop_max_dist=0.4,
        preprocess_cache_to_disk=True,
        subsample_mode="recon",
        recon_check_enabled=True,
        recon_rel_threshold=0.11,
        recon_check_frames_mode="subset",
        recon_cache_memory=True,
        recon_cache_disk=True,
        deterministic_val_subsets=True,
        exclude_airfoil_from_loss=True,
        aug_y_flip=True,
    ),
}


def seed_all(seed: int = RANDOM_SEED) -> None:
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
    mu = y_train.mean(axis=0).astype(np.float32)
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


def resolve_profile(profile_name: str) -> RunConfig:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {sorted(PROFILES.keys())}")
    return copy.deepcopy(PROFILES[profile_name])


def _set_if_not_none(cfg: RunConfig, key: str, value: Any) -> None:
    if value is not None:
        setattr(cfg, key, value)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=str, default="tiny", choices=sorted(PROFILES.keys()))
    p.add_argument("--epochs", type=int)
    p.add_argument("--train-points", type=int)
    p.add_argument("--val-points", type=int)
    p.add_argument("--tf-decay-epochs", type=int)
    p.add_argument("--tf-schedule", type=str, choices=["linear", "cosine", "piecewise"])
    p.add_argument("--full-eval-every", type=int)
    p.add_argument("--crop-enabled", type=int, choices=[0, 1])
    p.add_argument("--crop-max-dist", type=float)
    p.add_argument("--subsample-mode", type=str, choices=["uniform", "biased", "recon"])
    p.add_argument("--recon-rel-threshold", type=float)
    p.add_argument("--recon-check-frames-mode", type=str, choices=["subset", "all", "outputs_only"])
    p.add_argument("--exclude-airfoil-from-loss", type=int, choices=[0, 1])
    p.add_argument("--aug-y-flip", type=int, choices=[0, 1])
    p.add_argument("--aug-noise-std-vel", type=float)
    p.add_argument("--aug-noise-std-pos", type=float)
    p.add_argument("--max-train-files", type=int)
    p.add_argument("--max-test-files", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--config-out", type=str, default=None)
    return p


def config_from_args(args: argparse.Namespace) -> RunConfig:
    cfg = resolve_profile(args.profile)
    _set_if_not_none(cfg, "epochs", args.epochs)
    _set_if_not_none(cfg, "train_points", args.train_points)
    _set_if_not_none(cfg, "val_points", args.val_points)
    _set_if_not_none(cfg, "tf_decay_epochs", args.tf_decay_epochs)
    _set_if_not_none(cfg, "tf_schedule", args.tf_schedule)
    _set_if_not_none(cfg, "full_eval_every", args.full_eval_every)
    if args.crop_enabled is not None:
        cfg.crop_enabled = bool(args.crop_enabled)
    _set_if_not_none(cfg, "crop_max_dist", args.crop_max_dist)
    _set_if_not_none(cfg, "subsample_mode", args.subsample_mode)
    _set_if_not_none(cfg, "recon_rel_threshold", args.recon_rel_threshold)
    _set_if_not_none(cfg, "recon_check_frames_mode", args.recon_check_frames_mode)
    if args.exclude_airfoil_from_loss is not None:
        cfg.exclude_airfoil_from_loss = bool(args.exclude_airfoil_from_loss)
    if args.aug_y_flip is not None:
        cfg.aug_y_flip = bool(args.aug_y_flip)
    _set_if_not_none(cfg, "aug_noise_std_vel", args.aug_noise_std_vel)
    _set_if_not_none(cfg, "aug_noise_std_pos", args.aug_noise_std_pos)
    _set_if_not_none(cfg, "max_train_files", args.max_train_files)
    _set_if_not_none(cfg, "max_test_files", args.max_test_files)
    _set_if_not_none(cfg, "seed", args.seed)
    return cfg


def save_config(cfg: RunConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
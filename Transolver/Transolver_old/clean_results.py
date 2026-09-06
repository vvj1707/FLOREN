# regenerate_positions.py
from pathlib import Path
import json
import numpy as np

from load_data import load_split
from preprocessing import preprocess_samples
from config import RunConfig, MODEL_DIR, RESULTS_DIR

DATA_DIR = Path(__file__).resolve().parent.parent / "data/warped-ifw"

with open(MODEL_DIR / "run_config.json") as f:
    cfg_dict = json.load(f)
cfg = RunConfig(**cfg_dict)

print("Reloading split with saved seed/test_frac...")
_, test_samples, _, test_files = load_split(data_dir=DATA_DIR, test_frac=cfg.test_frac, seed=cfg.seed)

if cfg.max_test_files is not None:
    test_samples = test_samples[: cfg.max_test_files]
    test_files = test_files[: cfg.max_test_files]

print("Reapplying the same crop used at training time...")
test_samples, _, _ = preprocess_samples(test_samples, cfg, "test", cache_dir=None)

x_obj = np.empty((len(test_samples),), dtype=object)
idcs_obj = np.empty((len(test_samples),), dtype=object)
for i, s in enumerate(test_samples):
    x_obj[i] = s["x"].astype(np.float32)
    idcs_obj[i] = s["idcs_airfoil"].astype(np.int32)

np.save(RESULTS_DIR / "x_points.npy", x_obj, allow_pickle=True)
np.save(RESULTS_DIR / "idcs_airfoil.npy", idcs_obj, allow_pickle=True)
print(f"Wrote x_points.npy and idcs_airfoil.npy to {RESULTS_DIR}")
print(f"Sample 0: {x_obj[0].shape[0]} points (should match your y_true[0] point count)")

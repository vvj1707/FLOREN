"""
load_data.py — Warped-IFW dataset loader + subsampling helpers for TransolverAR.

Each .npz sample contains:
    t             (10,)
    pos           (100000, 3)
    idcs_airfoil  (20000,)
    pressure      (10, 100000)
    velocity_in   (5, 100000, 3)
    velocity_out  (5, 100000, 3)

We build Transolver samples:
    x   : (N, 3)
    fx  : (N, 16) = flattened 5-step input velocity (15) + distance-to-surface (1)
    y   : (N, 15) = flattened 5-step output velocity
    idcs_airfoil : (n_airfoil_local,) indices relative to current sample points
"""

import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

N_POINTS_FULL = 100_000
N_IN_FRAMES = 5
N_OUT_FRAMES = 5
VEL_DIM = 3

FILENAME_RE = re.compile(r"^(?P<geometry_id>[^_]+)_(?P<case_id>[^-]+)-(?P<window>\d+)$")


def list_samples(data_dir: Path) -> list[Path]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")
    return files


def parse_geometry_id(path: Path) -> str:
    m = FILENAME_RE.match(path.stem)
    if m is None:
        raise ValueError(
            f"Filename {path.name!r} doesn't match expected pattern "
            "'{geometry_id}_{case_id}-{window}.npz'"
        )
    return m.group("geometry_id")


def geometry_disjoint_split(
    files: list[Path], test_frac: float = 0.15, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    by_geom: dict[str, list[Path]] = {}
    for f in files:
        by_geom.setdefault(parse_geometry_id(f), []).append(f)

    geom_ids = sorted(by_geom)
    rng = np.random.default_rng(seed)
    rng.shuffle(geom_ids)

    n_test_geoms = max(1, round(len(geom_ids) * test_frac))
    test_geoms = set(geom_ids[:n_test_geoms])
    train_geoms = set(geom_ids[n_test_geoms:])

    train_files = [f for g in sorted(train_geoms) for f in by_geom[g]]
    test_files = [f for g in sorted(test_geoms) for f in by_geom[g]]

    print(
        f"Geometry-disjoint split: {len(train_geoms)} train geometries "
        f"({len(train_files)} files), {len(test_geoms)} test geometries "
        f"({len(test_files)} files)"
    )
    return train_files, test_files


def load_raw_sample(path: Path) -> dict:
    with np.load(path) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def surface_distance(pos: np.ndarray, idcs_airfoil: np.ndarray) -> np.ndarray:
    surface_pts = pos[idcs_airfoil]
    tree = cKDTree(surface_pts)
    dist, _ = tree.query(pos, k=1)
    return dist.astype(np.float32)


def build_transolver_sample(raw: dict) -> dict:
    pos = raw["pos"].astype(np.float32)  # (100000, 3)
    vel_in = raw["velocity_in"].astype(np.float32)    # (5, 100000, 3)
    vel_out = raw["velocity_out"].astype(np.float32)  # (5, 100000, 3)
    idcs_airfoil = raw["idcs_airfoil"].astype(np.int32)

    n_points = pos.shape[0]
    if vel_in.shape[1] != n_points or vel_out.shape[1] != n_points:
        raise ValueError(
            f"Point-count mismatch: pos={n_points}, "
            f"velocity_in={vel_in.shape[1]}, velocity_out={vel_out.shape[1]}"
        )

    dist = surface_distance(pos, idcs_airfoil)  # (N,)

    # (5, N, 3) -> (N, 5, 3) -> (N, 15)
    vel_in_flat = vel_in.transpose(1, 0, 2).reshape(n_points, -1)
    vel_out_flat = vel_out.transpose(1, 0, 2).reshape(n_points, -1)

    fx = np.concatenate([vel_in_flat, dist[:, None]], axis=-1).astype(np.float32)

    return {
        "x": pos,
        "fx": fx,
        "y": vel_out_flat.astype(np.float32),
        "idcs_airfoil": idcs_airfoil,
    }


def _remap_airfoil_indices(selected_global_idx: np.ndarray, airfoil_global_idx: np.ndarray) -> np.ndarray:
    """
    Convert airfoil global indices into local indices after subsampling.
    """
    g2l = {g: i for i, g in enumerate(selected_global_idx.tolist())}
    local = [g2l[g] for g in airfoil_global_idx.tolist() if g in g2l]
    return np.asarray(local, dtype=np.int32)


def subsample_sample_uniform(
    sample: dict,
    n_points_subsample: int,
    rng: np.random.Generator,
    keep_all_airfoil: bool = True,
) -> dict:
    """
    Uniform subsample with optional guaranteed inclusion of all airfoil points.
    """
    x = sample["x"]
    fx = sample["fx"]
    y = sample["y"]
    idcs_airfoil = sample["idcs_airfoil"]

    n = x.shape[0]
    if n_points_subsample >= n:
        return sample

    if keep_all_airfoil:
        airfoil_set = np.unique(idcs_airfoil)
        n_air = len(airfoil_set)

        if n_air >= n_points_subsample:
            # If airfoil itself exceeds budget, sample airfoil only.
            selected_air = rng.choice(airfoil_set, size=n_points_subsample, replace=False)
            selected_idx = np.sort(selected_air)
        else:
            all_idx = np.arange(n, dtype=np.int32)
            non_air = np.setdiff1d(all_idx, airfoil_set, assume_unique=False)
            n_rest = n_points_subsample - n_air
            selected_rest = rng.choice(non_air, size=n_rest, replace=False)
            selected_idx = np.sort(np.concatenate([airfoil_set, selected_rest]))
    else:
        selected_idx = np.sort(rng.choice(np.arange(n), size=n_points_subsample, replace=False))

    local_airfoil = _remap_airfoil_indices(selected_idx, idcs_airfoil)

    return {
        "x": x[selected_idx],
        "fx": fx[selected_idx],
        "y": y[selected_idx],
        "idcs_airfoil": local_airfoil,
    }


def load_split(
    data_dir: Path,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[dict], list[dict], list[Path], list[Path]]:
    files = list_samples(data_dir)
    train_files, test_files = geometry_disjoint_split(files, test_frac, seed)

    print(f"Loading {len(train_files)} train samples...")
    train_samples = [build_transolver_sample(load_raw_sample(f)) for f in train_files]

    print(f"Loading {len(test_files)} test samples...")
    test_samples = [build_transolver_sample(load_raw_sample(f)) for f in test_files]

    return train_samples, test_samples, train_files, test_files
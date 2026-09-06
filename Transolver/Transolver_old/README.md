# TransolverAR Phase 1 + Phase 2 bundle for Warped-IFW

This bundle upgrades the current TransolverAR training pipeline with:

- honest validation and W&B logging
- deterministic validation subsets
- periodic full denormalized validation matching saved artifacts / visualization
- named training profiles and CLI overrides
- strict no-slip enforcement during train, inference, and save
- configurable airfoil-point loss masking
- distance crop preprocessing
- biased + reconstruction-checked subsampling
- in-memory and optional on-disk preprocessing / subset caches
- augmentation hooks with airfoil-safe noise behavior
- richer diagnostics

The code is designed as a mostly backward-compatible refactor with helper modules.

## Files

- `config.py` - defaults, named profiles, config dataclass, CLI override helper
- `preprocessing.py` - crop / reindex / cache helpers
- `subsampling.py` - uniform, biased, and reconstruction-checked subsampling
- `validation.py` - full denormalized evaluation, deterministic validation subsets, diagnostics
- `train_transolver.py` - main training entrypoint

## Expected sample format

Each sample is a dict with keys:
- `x`: `(N,3)` point positions
- `fx`: `(N,16)` input features = 15 velocity-history values + 1 distance feature
- `y`: `(N,15)` target rollout = 5 future velocity vectors flattened
- `idcs_airfoil`: `(Na,)` indices of no-slip wall / airfoil points

## Main usage

Tiny run:

```bash
python train_transolver.py --profile tiny
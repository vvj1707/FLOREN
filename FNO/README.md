# Warped-IFW FNO Directory Guide

## Overview

This directory implements a **Fourier Neural Operator (FNO)** pipeline in **JAX** for the **Warped-IFW** dataset.

At a high level, the code does the following:

1. Loads raw `.npz` CFD-style samples.
2. Converts point-cloud data into a pointwise supervised learning format.
3. Splits data by geometry so train and test geometries are disjoint.
4. Rasterizes irregular point clouds into fixed-size voxel grids.
5. Trains a 3D FNO on those grids.
6. Predicts future velocity fields on grids.
7. Interpolates predictions back to original point locations.
8. Evaluates point-level metrics.
9. Saves model artifacts and prediction arrays.
10. Produces visual diagnostics and reports.

This guide explains the **directory structure**, the **purpose of each script**, the **data flow**, and the **key implementation ideas** needed to understand and work with the project.

---

## What problem this code solves

Each raw dataset sample contains:

- point positions in 3D,
- a set of airfoil/surface point indices,
- a history of input velocities,
- future output velocities,
- pressure values over time.

The goal of the FNO pipeline is to learn a mapping:

$$
\text{past velocity history + geometric context} \rightarrow \text{future velocity field}
$$

More concretely:

- **Input per point**: 5 past velocity frames, flattened into 15 channels, plus distance-to-surface, for a total of 16 channels.
- **Grid input**: the 16 point features are rasterized into a regular voxel grid, and an extra occupancy channel is added, giving 17 input channels.
- **Output per point**: 5 future velocity frames, flattened into 15 channels.
- **Model output**: the FNO predicts 15 channels on the voxel grid, which are then interpolated back to points.

---

## Expected directory layout

A typical layout for this folder is:

```text
fno_directory/
├── comparison.py
├── config.py
├── load_data.py
├── loss.py
├── metrics.py
├── train.py
├── visualise_fno.py
├── data/
│   └── warped-ifw/
│       ├── sample1.npz
│       ├── sample2.npz
│       └── ...
├── trained_model/
│   ├── fno_params.pkl
│   ├── x_mu.npy
│   ├── x_std.npy
│   ├── y_mu.npy
│   ├── y_std.npy
│   ├── grid_spec.json
│   └── training_history.json
├── results/
│   ├── y_true_points.npy
│   ├── y_pred_points.npy
│   ├── relative_l2_per_sample.npy
│   └── metrics.json
└── outputs_fno/
    ├── single/
    ├── anim/
    ├── report/
    ├── matlab/
    ├── component_losses/
    └── diagnostics/
```

Important note:

- `trained_model/` and `results/` are created by training.
- `outputs_fno/` is created by the visualization script.
- `data/warped-ifw/` must contain the raw `.npz` samples.

---

## End-to-end pipeline summary

The pipeline can be understood in four stages:

### 1. Data loading and conversion

Handled primarily by `load_data.py`.

Raw `.npz` files are converted into a common sample format:

- `x`: point positions, shape `(N, 3)`
- `fx`: pointwise inputs, shape `(N, 16)`
- `y`: pointwise targets, shape `(N, 15)`
- `idcs_airfoil`: indices of airfoil points

### 2. Grid construction and normalization

Handled primarily by `config.py`.

Pointwise samples are rasterized onto a voxel grid:

- input grid `Xg`: shape `(17, gx, gy, gz)`
- target grid `Yg`: shape `(15, gx, gy, gz)`
- occupancy mask `Mg`: shape `(1, gx, gy, gz)`

Then input and target channels are normalized using statistics from the training set.

### 3. FNO training and evaluation

Handled by `train.py`.

The code:

- builds the model,
- trains with a custom JAX loop,
- evaluates on the test set,
- converts grid predictions back to point predictions,
- computes final point-level metrics,
- saves all artifacts.

### 4. Visualization and reporting

Handled by `visualise_fno.py`.

This script reloads saved predictions and test samples, then creates:

- training curves,
- grid diagnostics,
- per-sample static visualizations,
- animations,
- per-component loss reports,
- MATLAB export bundles,
- batch summary reports.

---

# Core data representations

Understanding the project is easiest if you first understand the data shapes.

## Raw dataset sample

Each `.npz` file contains:

- `t`: shape `(10,)`
- `pos`: shape `(100000, 3)`
- `idcs_airfoil`: shape `(20000,)`
- `pressure`: shape `(10, 100000)`
- `velocity_in`: shape `(5, 100000, 3)`
- `velocity_out`: shape `(5, 100000, 3)`

Interpretation:

- There are `100000` points in the 3D domain.
- There are 5 input time steps and 5 output time steps.
- Each velocity frame is 3D: $$u_x, u_y, u_z$$.

## Converted pointwise sample

Produced by `build_transolver_sample()` in `load_data.py`.

### `x`

```python
x.shape == (N, 3)
```

Point coordinates.

### `fx`

```python
fx.shape == (N, 16)
```

Features per point:

- channels `0..14`: flattened 5-step input velocity history,
- channel `15`: distance to nearest airfoil/surface point.

The velocity flattening is:

```text
[v_t0(3), v_t1(3), v_t2(3), v_t3(3), v_t4(3)]
```

So each point gets:

$$
5 \times 3 = 15
$$

velocity values, plus one surface distance.

### `y`

```python
y.shape == (N, 15)
```

Target future velocities, flattened in the same way:

```text
[v_t5(3), v_t6(3), v_t7(3), v_t8(3), v_t9(3)]
```

### `idcs_airfoil`

Local indices of airfoil points within the sample.

---

## Grid representation

Produced by `points_to_grid()` in `config.py`.

### Input grid `Xg`

```python
Xg.shape == (17, gx, gy, gz)
```

Channels are:

- channels `0..15`: averaged point features,
- channel `16`: occupancy.

This means:

- first 15 channels: input velocity history,
- channel 15: distance-to-surface,
- channel 16: occupancy count converted to binary occupancy.

### Target grid `Yg`

```python
Yg.shape == (15, gx, gy, gz)
```

This stores averaged future velocity channels per occupied voxel.

### Mask `Mg`

```python
Mg.shape == (1, gx, gy, gz)
```

Binary occupancy mask indicating which voxels contain at least one point.

This mask is important because the grid is sparse: most of the voxel space may be empty.

---

# File-by-file explanation

## `load_data.py`

### Purpose

This script is responsible for:

- listing dataset files,
- parsing file names to identify geometry IDs,
- creating a geometry-disjoint train/test split,
- loading raw samples,
- computing distance-to-surface,
- converting raw data into the pointwise learning format,
- optionally subsampling samples.

### Key functions

#### `list_samples(data_dir)`

Finds all `.npz` files in the dataset directory.

#### `parse_geometry_id(path)`

Extracts the geometry identifier from the filename using the pattern:

```text
{geometry_id}_{case_id}-{window}.npz
```

This is used to ensure train/test separation by geometry.

#### `geometry_disjoint_split(files, test_frac, seed)`

Groups files by geometry ID, shuffles geometry IDs, and assigns some geometries to test.

Why this matters:

- It prevents the model from seeing the same geometry in both train and test.
- It is a stronger generalization test than random file-level splitting.

#### `load_raw_sample(path)`

Loads a raw `.npz` file into a dictionary of NumPy arrays.

#### `surface_distance(pos, idcs_airfoil)`

Builds a nearest-neighbor tree over airfoil points and computes, for every point, the distance to the closest surface point.

This distance becomes an important geometric input feature.

#### `build_transolver_sample(raw)`

Converts raw dataset arrays into the common sample structure used throughout the project.

Main steps:

1. Extract positions and velocities.
2. Validate point count consistency.
3. Compute surface distance.
4. Flatten velocity histories.
5. Concatenate features.
6. Return standardized sample dictionary.

#### `subsample_sample_uniform(...)`

Uniformly subsamples points, optionally forcing inclusion of all airfoil points.

In this FNO pipeline, this helper exists but is not central to the main training path shown in `train.py`, which uses the full point clouds.

#### `load_split(data_dir, test_frac, seed)`

End-to-end loader used by training and visualization.

It:

1. finds all files,
2. creates geometry-disjoint train/test split,
3. loads train samples,
4. loads test samples,
5. returns samples and file lists.

### Why this file matters conceptually

This file defines the **canonical sample format** used by the rest of the project. Once raw CFD files are transformed into `x`, `fx`, and `y`, all later code becomes much simpler.

---

## `comparison.py`

### Purpose

Provides a simple baseline for evaluation: the **persistence baseline**.

### Core idea

The last observed velocity frame is repeated for all 5 future outputs.

If the input history is:

```text
[v_t0, v_t1, v_t2, v_t3, v_t4]
```

then the baseline predicts:

```text
[v_t4, v_t4, v_t4, v_t4, v_t4]
```

### Key functions

#### `persistence_baseline(samples)`

For each sample:

1. takes the first 15 feature channels from `fx`,
2. extracts the last observed frame,
3. concatenates it 5 times,
4. returns predictions of shape `(B, N, 15)`.

#### `evaluate_baseline(samples)`

Compares baseline predictions to true outputs using dataset-level metrics.

Returns:

- mean relative L2,
- standard deviation of relative L2 across samples,
- RMSE,
- MAE,
- per-sample relative L2 values.

### Why this file matters

It gives a sanity-check benchmark. If the learned FNO does not outperform persistence, then training or representation choices may need review.

---

## `metrics.py`

### Purpose

Defines point-level evaluation metrics.

These are applied after predictions are interpolated back from the voxel grid to the original point cloud.

### Key functions

#### `relative_l2_dataset(y_true, y_pred)`

Inputs:

```python
y_true.shape == y_pred.shape == (B, N, 15)
```

For each sample, computes:

$$
\sqrt{\frac{\sum (y_{pred} - y_{true})^2}{\sum y_{true}^2 + 10^{-12}}}
$$

Returns:

- the mean across samples,
- the per-sample values.

#### `rmse_dataset(y_true, y_pred)`

Computes the global RMSE over the entire stacked dataset.

#### `mae_dataset(y_true, y_pred)`

Computes the global MAE over the entire stacked dataset.

### Why this file matters

These metrics represent the final point-level quality of the model, which is the most meaningful measure for the original irregular-point problem.

---

## `loss.py`

### Purpose

Defines training-time losses on voxel grids.

### Key functions

#### `relative_l2_grid(pred, target, mask=None)`

Computes masked relative L2 on grid outputs.

If a mask is provided, only occupied voxels contribute.

Formula:

$$
\sqrt{\frac{\sum (pred - target)^2}{\sum target^2 + 10^{-12}}}
$$

with optional masking applied to both prediction and target.

#### `mae_grid(pred, target, mask=None)`

Computes masked mean absolute error on the grid.

### Why this file matters

The model is trained in **grid space**, not directly in point space. So these losses define the optimization objective used during training.

Important conceptual distinction:

- **Training loss** is in normalized grid space.
- **Final evaluation metric** is in point space after interpolation.

Those numbers are related, but not identical.

---

## `config.py`

### Purpose

This is the central configuration and utility module.

It contains:

- path definitions,
- hyperparameters,
- model construction,
- learning-rate schedule construction,
- grid building utilities,
- normalization functions,
- artifact saving logic.

### Main sections

#### Paths

```python
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "trained_model"
RESULTS_DIR = SCRIPT_DIR / "results"
```

This establishes where outputs are stored.

#### Dataset/split config

- `TEST_FRAC = 0.15`
- `RANDOM_SEED = 42`

#### Grid config

- `GRID_X = 64`
- `GRID_Y = 32`
- `GRID_Z = 32`

So the FNO works on a fixed 3D grid of size:

$$
64 \times 32 \times 32
$$

#### Channel config

- `IN_CH = 17`
- `OUT_CH = 15`

#### FNO hyperparameters

- `FNO_N_MODES = (16, 12, 12)`
- `FNO_HIDDEN = 48`
- `FNO_LAYERS = 4`
- `FNO_USE_GRID_EMBED = False`

#### Optimization config

- `EPOCHS = 50`
- `BATCH_SIZE = 4`
- `LEARNING_RATE = 2e-4`
- `WEIGHT_DECAY = 1e-6`
- `WARMUP_EPOCHS = 5`
- `LR_FLOOR = 1e-6`
- `GRAD_CLIP = 1.0`

### Key functions

#### `seed_all(seed)`

Seeds NumPy randomness.

#### `make_schedule(...)`

Creates a warmup + cosine decay learning-rate schedule using `optax`.

The schedule has two phases:

1. linear warmup from `0` to the base learning rate,
2. cosine decay from the base learning rate down toward `lr_floor`.

#### `build_fno()`

Constructs the FNO model using the external implementation:

```python
from operators.models.fno_jax import FNO
```

This is the heart of the learned operator.

#### `build_grid_spec(samples, gx, gy, gz)`

Computes the global spatial bounds of all samples and stores:

- `xyz_min`,
- `xyz_max`,
- grid resolution.

This spec defines how physical 3D coordinates map into voxel indices.

#### `_normalize_pos(pos, xyz_min, xyz_max)`

Normalizes physical coordinates into the unit cube.

#### `points_to_grid(sample, grid_spec)`

One of the most important functions in the whole project.

It converts a point-cloud sample into grid tensors.

Main steps:

1. normalize point coordinates,
2. map points to integer voxel indices,
3. accumulate feature sums into voxels,
4. accumulate target sums into voxels,
5. count how many points fall into each voxel,
6. divide sums by counts to get voxel averages,
7. produce occupancy mask.

This means voxel values represent **average point values** for all points that fell into each voxel.

#### `trilinear_grid_to_points(grid, points, grid_spec)`

Maps a predicted grid back to arbitrary point coordinates by trilinear interpolation.

This is essential because:

- the model predicts on a regular grid,
- but the final evaluation is performed at the original irregular points.

#### `normalize_grid_inputs(X_train, X_all)`

Normalizes channels `0..15` of the input grid using training-set statistics.

Occupancy channel `16` is left unchanged.

#### `normalize_grid_targets(Y_train, Y_all)`

Normalizes target grid channels using training-set statistics.

#### `save_artifacts(...)`

Writes model parameters, normalization arrays, grid spec, history, predictions, and metrics to disk.

Saved files include:

- `trained_model/fno_params.pkl`
- `trained_model/x_mu.npy`
- `trained_model/x_std.npy`
- `trained_model/y_mu.npy`
- `trained_model/y_std.npy`
- `trained_model/grid_spec.json`
- `trained_model/training_history.json`
- `results/y_true_points.npy`
- `results/y_pred_points.npy`
- `results/relative_l2_per_sample.npy`
- `results/metrics.json`

### Why this file matters

If `load_data.py` defines the canonical pointwise representation, then `config.py` defines the canonical **grid representation and training environment**.

---

## `train.py`

### Purpose

This is the main training and evaluation entry point.

It orchestrates the complete FNO workflow:

- data loading,
- baseline evaluation,
- grid construction,
- normalization,
- model initialization,
- training loop,
- test inference,
- point-level interpolation and metric computation,
- saving outputs.

### High-level flow

#### 1. Initialize randomness and optional Weights & Biases

The script calls `seed_all()` and optionally creates a W&B run.

#### 2. Load train/test samples

It uses `load_split()` from `load_data.py`.

This gives:

- `train_samples`
- `test_samples`
- `train_files`
- `test_files`

#### 3. Evaluate persistence baseline

It computes baseline test metrics before training the model.

This is useful for benchmarking improvement.

#### 4. Build grid specification

A single `grid_spec` is built from the spatial bounds of train and test samples.

This determines how all point clouds are voxelized.

#### 5. Rasterize all samples to voxel grids

For every sample, the script creates:

- input grid `X`,
- target grid `Y`,
- occupancy mask `M`.

#### 6. Normalize inputs and targets

Normalization statistics are computed from the training set and applied to both train and test.

#### 7. Build and initialize the FNO

The model is created by `build_fno()` and initialized with one example batch.

#### 8. Build optimizer and schedule

Optimizer chain:

1. gradient clipping,
2. weight decay,
3. Adam with scheduled learning rate.

#### 9. Train with custom JAX loop

The training loop uses:

- `loss_fn()` for masked relative L2 in grid space,
- `train_step()` compiled with `jax.jit`,
- `eval_step()` compiled with `jax.jit`.

For each epoch:

- iterate through train batches,
- update parameters,
- iterate through test batches for validation loss,
- store history,
- print and optionally log metrics.

#### 10. Run point-level inference on test set

After training, the script performs full inference sample-by-sample:

1. rasterize each test sample,
2. normalize input grid,
3. run FNO on grid,
4. denormalize output grid,
5. interpolate predicted grid back to original points.

This yields point-level predictions of shape:

```python
(B, N, 15)
```

#### 11. Compute final point-level metrics

Using `relative_l2_dataset()`.

#### 12. Save artifacts

Calls `save_artifacts()`.

### Important implementation details

#### Batch creation

`make_batches()` yields JAX arrays.

#### Masking

The loss is masked by occupancy so empty voxels do not dominate training.

#### Distinction between validation loss and final metric

Validation loss is computed on normalized grids, while final reported metrics are computed on original points after interpolation.

This is one of the most important conceptual details in the project.

### Why this file matters

This is the script you run to actually train the FNO and generate the saved outputs used by everything else.

---

## `visualise_fno.py`

### Purpose

This script is a dedicated visualization and reporting tool for the FNO pipeline.

It is intentionally separate from the Transolver visualization logic because the FNO workflow differs in several major ways.

### Why a dedicated visualization script exists

The top docstring explains the reasons clearly:

1. The FNO pipeline has different training/config conventions.
2. Test samples are full point clouds, not regenerated subsamples.
3. Saved predictions are regular stacked arrays, not ragged arrays.
4. The FNO is grid-native, so it needs grid-specific diagnostics.
5. Training losses are in normalized grid space, which should not be confused with point-level competition metrics.

### What this script produces

Inside `outputs_fno/`, it creates:

- `single/`: per-sample static plots,
- `anim/`: per-sample animations,
- `report/`: summary figures and CSVs,
- `matlab/`: `.mat` or `.npz` export bundles,
- `component_losses/`: component-wise error breakdowns,
- `diagnostics/`: training curves and grid diagnostics.

### Main functional sections

#### Data reloading

##### `load_once_fno()`

Loads:

- saved point predictions from `results/`,
- the deterministic test split from `load_split()`,
- raw `.npz` samples for pressure visualization.

It validates that shapes and point counts match.

This function is essential because visualization needs both:

- predictions,
- original geometry and raw fields.

#### Artifact loading helpers

- `load_grid_spec()`
- `load_training_history()`
- `load_final_metrics()`

These read saved artifacts from disk.

#### Grid-specific diagnostics

##### `plot_grid_diagnostic(...)`

Creates a diagnostic panel showing:

- occupancy at a selected grid slice,
- rasterized target field at the same slice.

This helps answer questions like:

- Is the voxelization too coarse?
- Is occupancy sparse or dense?
- What does the target field look like before interpolation back to points?

#### Training curves

##### `plot_training_curves(...)`

Plots:

1. train/validation grid loss,
2. learning rate schedule,
3. time per epoch.

It can also overlay:

- final point-level relative L2,
- persistence baseline relative L2.

This is especially valuable because it highlights the distinction between:

- optimization objective,
- final evaluation metric.

#### Component-wise losses

Functions such as:

- `component_losses(...)`
- `dataset_component_losses(...)`
- `save_component_loss_csv(...)`
- `plot_dataset_component_losses(...)`
- `plot_component_loss_breakdown(...)`

These break errors down by:

- `ux`,
- `uy`,
- `uz`,
- speed magnitude.

This helps diagnose which velocity component is hardest for the model.

#### Geometry and field helpers

Functions such as:

- `flat15_to_t53(...)`
- `speed_mag(...)`
- `frame_fields(...)`
- `choose_section_mask(...)`
- `interpolate_masked(...)`

These are utilities for converting flat predictions into frame/component form, slicing geometry, and generating smooth 2D contour fields.

#### Static plots

Main outputs include:

- contour + scatter panels,
- scatter-only panels,
- 3D wing surface visualizations.

These compare ground truth and predictions visually.

#### Animations

- `make_contour_animation(...)`
- `make_scatter_animation(...)`

These animate the 5 predicted future frames.

#### MATLAB export

##### `export_matlab_bundle(...)`

Exports sample subsets in a format convenient for MATLAB analysis.

#### Batch report

##### `batch_report(...)`

Builds a summary over many test samples, including CSV metrics.

### The `main()` function flow

1. Parse command-line flags.
2. Create output folders.
3. Load predictions and test data.
4. Optionally generate training curves.
5. Optionally generate grid diagnostics.
6. Compute component-wise losses.
7. Generate static plots for selected samples.
8. Optionally generate animations.
9. Export MATLAB bundle.
10. Generate batch report.

### Why this file matters

It is the main analysis/debugging/reporting tool for understanding model behavior after training.

---

# How the scripts work together

## Training-time interaction graph

```text
train.py
├── uses load_data.py to load and split data
├── uses comparison.py for persistence baseline
├── uses config.py for model config, grid conversion, normalization, saving
├── uses loss.py for training loss
└── uses metrics.py for final point-level evaluation
```

## Visualization-time interaction graph

```text
visualise_fno.py
├── uses load_data.py to reconstruct the deterministic test split
├── uses config.py for paths and grid helpers
├── uses comparison.py for baseline reference
└── uses saved artifacts from trained_model/ and results/
```

---

# Detailed data flow

## Step 1: Raw `.npz` to pointwise supervised sample

`load_data.py` converts raw arrays into:

- `x`: positions,
- `fx`: flattened velocity history + distance,
- `y`: flattened future velocities.

This is the bridge from raw CFD data to ML-ready input/output pairs.

## Step 2: Pointwise sample to grid tensors

`config.points_to_grid()` takes each sample and rasterizes it.

This is necessary because the FNO expects fixed-resolution grid-structured input.

## Step 3: Grid normalization

Training-set channel statistics are used to normalize:

- input channels `0..15`,
- all target channels.

The occupancy channel is not normalized.

## Step 4: FNO training in grid space

The FNO consumes:

```python
(B, 17, gx, gy, gz)
```

and predicts:

```python
(B, 15, gx, gy, gz)
```

Training minimizes masked relative L2 on occupied voxels.

## Step 5: Grid prediction back to points

After inference, each predicted grid is interpolated back to the original point coordinates using trilinear interpolation.

This restores predictions to the original geometry representation.

## Step 6: Point-level metrics and saved outputs

Final metrics are computed in point space and saved for analysis.

---

# Folder and artifact explanation

## `data/warped-ifw/`

Contains the raw dataset `.npz` files.

This is the only folder that must exist before running training.

## `trained_model/`

Created by `save_artifacts()`.

### `fno_params.pkl`

Serialized trained model parameters.

### `x_mu.npy`, `x_std.npy`

Input normalization statistics for channels `0..15`.

### `y_mu.npy`, `y_std.npy`

Target normalization statistics.

### `grid_spec.json`

Defines:

- global min/max coordinates,
- grid dimensions.

Needed for consistent rasterization and interpolation.

### `training_history.json`

Stores epoch-wise:

- train loss,
- validation loss,
- learning rate,
- epoch time.

## `results/`

Also created by `save_artifacts()`.

### `y_true_points.npy`

Ground-truth pointwise targets for the test set.

### `y_pred_points.npy`

Predicted pointwise outputs for the test set.

### `relative_l2_per_sample.npy`

Per-sample relative L2 errors.

### `metrics.json`

Stores final summary metrics.

Note: in the current code, this JSON includes only relative L2 mean and std.

## `outputs_fno/`

Created by `visualise_fno.py`.

### `single/`

Per-sample static plots.

### `anim/`

Animated contour/scatter outputs.

### `report/`

Batch report figure and CSV summary.

### `matlab/`

MATLAB export bundle.

### `component_losses/`

Per-component CSV and plots.

### `diagnostics/`

Training curves and grid-rasterization diagnostic images.

---

# Important design choices

## 1. Geometry-disjoint split

The split is done by geometry ID, not individual files.

Why this is important:

- avoids leakage of geometric structure,
- evaluates generalization to unseen geometries,
- is more realistic for operator-learning problems.

## 2. Distance-to-surface as an input feature

The model is not given just velocity history. It also receives the distance of each point to the airfoil/surface.

This injects geometric context into the learning problem.

## 3. Rasterization by averaging

When multiple points fall into the same voxel, their features and targets are averaged.

This is a simple and stable strategy, though it introduces discretization error.

## 4. Occupancy masking

Empty voxels should not dominate the loss, so the training objective is masked by occupied cells only.

## 5. Evaluation in point space

Even though training happens on grids, final predictions are interpolated back to the original points.

This keeps evaluation aligned with the original irregular geometry problem.

## 6. Separate visualization pipeline

The visualization script is tailored to the FNO representation and saved outputs, rather than reusing tooling designed for a point-native model.

---

# Potential sources of confusion

## Grid loss vs point metric

One of the easiest mistakes is to compare training loss directly to final point-level relative L2.

They are different because:

- training loss is computed on normalized grid tensors,
- final metric is computed on original-scale point predictions after interpolation.

A lower grid loss usually helps, but the values are not numerically equivalent.

## `trained_model` vs `trained_model_fno`

The large docstring in `visualise_fno.py` mentions names like `trained_model_fno` and `results_fno`, but the actual imported paths come from `config.py`, where the directories are:

- `trained_model`
- `results`

So the effective paths are those defined in `config.py`.

## Full point clouds vs subsampling helpers

`load_data.py` includes subsampling utilities, but the shown FNO training pipeline uses full point clouds before voxelization.

So those helpers are available but not central to the current training path.

## Pressure is not part of the prediction target

Pressure is used for visualization in the 3D wing plot, but the model predicts only future velocity channels.

---

# How to run the workflow

## 1. Prepare data

Place all `.npz` samples in:

```text
data/warped-ifw/
```

## 2. Train the model

Run:

```bash
python train.py
```

This will:

- load data,
- split train/test,
- train the FNO,
- evaluate on the test set,
- save outputs under `trained_model/` and `results/`.

## 3. Generate visualizations

Run:

```bash
python visualise_fno.py
```

Optional flags include:

```bash
python visualise_fno.py --frame-idx 0
python visualise_fno.py --no-animations
python visualise_fno.py --no-3d
python visualise_fno.py --no-training-curves
python visualise_fno.py --no-grid-diagnostic
python visualise_fno.py --grid-diag-sample 0
```

This creates outputs under `outputs_fno/`.

---

# Script-by-script quick reference

## `comparison.py`

Use when you want:

- a simple persistence baseline,
- quick sanity-check metrics.

## `config.py`

Use when you want:

- hyperparameters,
- model construction,
- voxelization utilities,
- normalization,
- save/load path definitions.

## `load_data.py`

Use when you want:

- to load raw samples,
- create geometry-disjoint splits,
- convert raw data into the project sample format.

## `loss.py`

Use when you want:

- grid-space training losses.

## `metrics.py`

Use when you want:

- point-space evaluation metrics.

## `train.py`

Use when you want:

- to train and evaluate the FNO end-to-end.

## `visualise_fno.py`

Use when you want:

- diagnostics,
- qualitative visualizations,
- reports,
- MATLAB export.

---

# Minimal mental model of the system

If you only remember one picture, remember this:

```text
Raw .npz files
    ↓
load_data.py
    ↓
Pointwise sample: x, fx, y
    ↓
config.points_to_grid()
    ↓
Voxel grids: Xg, Yg, Mg
    ↓
train.py + FNO
    ↓
Predicted voxel grid
    ↓
config.trilinear_grid_to_points()
    ↓
Pointwise prediction
    ↓
metrics.py / visualise_fno.py
```

That is the complete logic of the folder.

---

# Suggestions for future maintainers

A few improvements that would make the directory even easier to maintain:

1. Add a top-level `README.md` that links to this guide.
2. Save baseline metrics to `results/metrics.json` as well.
3. Optionally save train/test file lists for perfect reproducibility.
4. Add an inference-only script for loading `fno_params.pkl` and predicting on new data.
5. Consider vectorizing some loops in voxelization/interpolation if performance becomes limiting.
6. Harmonize naming in the `visualise_fno.py` docstring with the actual directories in `config.py`.

---

# Final summary

This FNO directory implements a **grid-based operator-learning pipeline** for predicting future 3D velocity fields from historical velocities and geometric context.

Its key ideas are:

- convert irregular point clouds into pointwise supervised samples,
- voxelize them into regular grids,
- train a 3D FNO in grid space,
- interpolate predictions back to points,
- evaluate and visualize results in the original point domain.

The most important files are:

- `load_data.py` for data conversion,
- `config.py` for model/grid utilities,
- `train.py` for end-to-end training,
- `visualise_fno.py` for post-training analysis.

If you understand:

1. the sample format `x`, `fx`, `y`,
2. the voxelization step,
3. the distinction between grid-space training and point-space evaluation,
4. the role of saved artifacts,

then you understand the core of the whole folder.

# FLOREN

**FLOREN** is a research codebase for learning and analysing unsteady flow fields on the **Warped-IFW** dataset using neural operators and point-cloud models.

The repository contains two main modelling pipelines:

- **FNO**: a voxel-grid / Fourier Neural Operator baseline in JAX
- **TransolverAR**: an autoregressive Transolver-style model that operates directly on irregular point clouds

The project is designed to support:

- reproducible training and evaluation
- geometry-disjoint generalisation experiments
- comparison against simple baselines
- checkpointing and saved run artifacts
- visualisation, diagnostics, and report generation

---

## Overview

The dataset consists of CFD-style samples containing:

- 3D point positions
- airfoil / surface point indices
- pressure histories
- input velocity histories
- future output velocities

Across the project, the learning task is to predict **5 future velocity frames** from **5 past velocity frames**, while conditioning on geometry.

Geometry is represented implicitly through:

- spatial coordinates
- distance-to-surface information
- point-cloud structure or rasterised occupancy

This repository explores two different ways of solving that problem:

### 1. FNO pipeline

The FNO pipeline converts irregular point-cloud samples into fixed voxel grids and trains a **3D Fourier Neural Operator** to predict future flow fields.

High-level flow:

1. load raw `.npz` samples
2. build pointwise features
3. split train/test by geometry
4. rasterise points to voxel grids
5. train FNO in JAX
6. interpolate predictions back to points
7. evaluate and visualise results

### 2. TransolverAR pipeline

The Transolver pipeline works **directly on irregular point clouds** and predicts future frames autoregressively.

High-level flow:

1. load raw `.npz` samples
2. build pointwise features
3. split train/test by geometry
4. optionally crop meshes
5. optionally subsample meshes
6. normalise inputs and targets
7. train Transolver autoregressively
8. evaluate on subsampled and/or full meshes
9. save artifacts and generate diagnostics

---

## Repository structure

A typical top-level layout looks like this:

```text
FLOREN/
├── data/
│   └── warped-ifw/
│       ├── *.npz
│       └── ...
├── fno/
│   ├── config.py
│   ├── load_data.py
│   ├── loss.py
│   ├── metrics.py
│   ├── comparison.py
│   ├── train.py
│   ├── visualise_fno.py
│   ├── trained_model/
│   ├── results/
│   └── outputs_fno/
├── transolver/
│   ├── checkpoint.py
│   ├── config.py
│   ├── evaluate.py
│   ├── load_data.py
│   ├── loss.py
│   ├── normalisation.py
│   ├── preprocessing.py
│   ├── subsampling.py
│   ├── train.py
│   ├── visualise.py
│   ├── model/
│   │   └── core.py
│   ├── trained_model/
│   ├── results/
│   ├── cache/
│   └── outputs/
├── docs/
│   ├── FNO_directory_guide.md
│   └── transolver_directory_guide.md
└── README.md
```

> Exact folder names may vary slightly in your local repository, but the two core pipelines and their artifact directories are organised along these lines.

---

## Data format

Each raw Warped-IFW sample contains arrays such as:

- `t`
- `pos`
- `idcs_airfoil`
- `pressure`
- `velocity_in`
- `velocity_out`

These are converted into a shared pointwise format used across the project:

- `x`: point coordinates
- `fx`: input features per point
- `y`: future rollout targets
- `idcs_airfoil`: local airfoil indices

In both pipelines:

- the first 15 input channels are the flattened 5-frame velocity history
- 1 additional input channel stores distance-to-surface
- the target contains 5 future velocity frames flattened to 15 channels

---

## Train/test split

A key design choice in FLOREN is the **geometry-disjoint split**.

Samples are grouped by geometry ID parsed from the filename, and **entire geometries** are assigned either to train or test.

This means:

- no geometry appears in both train and test
- evaluation measures generalisation to unseen geometries
- the split is stronger than random file-level splitting

---

## Main features

### Common

- Warped-IFW dataset loading
- geometry-disjoint data splitting
- pointwise supervised sample construction
- saved metrics and model artifacts
- plotting and post-hoc diagnostics

### FNO

- voxel-grid rasterisation
- 3D FNO training in JAX
- point-level interpolation from grid predictions
- persistence baseline comparison
- grid-native diagnostics

### TransolverAR

- direct irregular point-cloud modelling
- autoregressive rollout training
- teacher-forcing schedules
- optional cropping around the airfoil
- multiple subsampling policies
- checkpoint save/resume support
- optional fine-tuning on full meshes

---

## Installation

Because FLOREN is research code, exact environment setup may depend on your machine, CUDA setup, and local package versions.

A typical setup is:

```bash
git clone <your-repo-url>
cd FLOREN
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you do not yet have a `requirements.txt`, document the environment manually, including at least:

- Python version
- JAX / jaxlib version
- NumPy
- SciPy
- Optax
- Matplotlib
- Weights & Biases
- any operator / FNO dependency used by the FNO pipeline

If running on GPU, ensure your installed JAX build matches your CUDA environment.

---

## Quick start

### Train the FNO baseline

```bash
cd fno
python train.py
```

### Visualise FNO outputs

```bash
cd fno
python visualise_fno.py
```

### Train TransolverAR

```bash
cd transolver
python train.py --profile smoke_test
```

For a larger run:

```bash
python train.py --profile full
```

### Visualise TransolverAR outputs

```bash
cd transolver
python visualise.py
```

---

## Configuration

### FNO

The FNO pipeline uses constants in `config.py` for:

- grid resolution
- model width / layers / modes
- optimisation settings
- training epochs and batch size
- test split fraction

### TransolverAR

The Transolver pipeline uses a `RunConfig` dataclass and named profiles such as:

- `smoke_test`
- `full`

These control:

- optimisation
- model size
- teacher forcing
- rollout supervision
- cropping
- subsampling
- augmentation
- checkpoint intervals
- fine-tuning
- W&B logging

Many Transolver settings can be overridden from the command line.

---

## Outputs and artifacts

Training and evaluation produce artifacts such as:

- saved model parameters
- normalisation statistics
- config snapshots
- checkpoints
- prediction arrays
- per-sample metrics
- summary metrics
- visual diagnostics
- MATLAB export bundles

Typical artifact folders include:

- `trained_model/`
- `results/`
- `outputs/` or `outputs_fno/`
- `cache/`

---

## Visualisation and diagnostics

Both pipelines include post-training visualisation scripts for generating:

- training curves
- per-sample error summaries
- component-wise loss analysis
- 2D contour/scatter plots
- animations across predicted frames
- 3D wing/surface views
- batch reports
- MATLAB-compatible export bundles

These tools are intended to make model behaviour interpretable beyond a single scalar metric.

---

## Documentation

Detailed pipeline-level documentation is available in:

- `docs/FNO_directory_guide.md`
- `docs/transolver_directory_guide.md`

These guides explain:

- directory structure
- script responsibilities
- tensor shapes
- preprocessing and normalisation
- training and evaluation flow
- saved outputs

---

## Research purpose

FLOREN is intended as a platform for comparing operator-learning and point-cloud autoregressive approaches for geometry-conditioned flow prediction.

It is especially useful for questions such as:

- how well do models generalise to unseen geometries?
- what is gained by working directly on irregular meshes?
- how much accuracy is lost when rasterising to grids?
- how does autoregressive drift grow across rollout steps?
- which components of the flow field are hardest to predict?

---

## Limitations

This is research code and may include assumptions specific to:

- the Warped-IFW dataset
- filename conventions used for geometry splitting
- fixed 5-input / 5-output temporal windows
- local project layout
- hardware-specific JAX setup

Before using the repository for a new dataset or benchmark, verify:

- data format compatibility
- geometry naming conventions
- model hyperparameters
- numerical stability on your hardware

---

## Contributing

If this repository is being shared with collaborators, a useful contribution workflow is:

1. create a feature branch
2. make focused changes
3. document any config or data-format changes
4. regenerate relevant outputs if needed
5. open a pull request with a concise summary

For substantial architectural changes, update the directory guides as well as the main README.

---

## Citation

If you use this repository in academic work, add the appropriate citation information here for:

- the FLOREN project
- the Warped-IFW dataset
- the Transolver method
- the Fourier Neural Operator method

Example placeholder:

```bibtex
@misc{floren,
  title={FLOREN: Flow Learning on Warped-IFW with Neural Operators and Point-Cloud Models},
  author={Your Name},
  year={2026}
}
```

---

## License

Add your chosen license here, for example:

- MIT
- Apache-2.0
- BSD-3-Clause
- proprietary / internal research use only

---

## Contact

For questions, issues, or collaboration, add:

- maintainer name
- GitHub handle
- email or project contact route

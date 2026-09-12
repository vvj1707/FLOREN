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

## Results

### FNO vs. TransolverAR

FLOREN compares two modelling pipelines on the **Warped-IFW** geometry-conditioned unsteady-flow task:

- **FNO** — a voxel-grid baseline that rasterises irregular point clouds onto a fixed 3D grid and predicts future flow on that grid before interpolating back to points.
- **TransolverAR** — an autoregressive point-cloud model that operates directly on irregular meshes without voxelisation.

The task is to predict **5 future velocity frames from 5 past velocity frames**, using a geometry-disjoint split so that test geometries are unseen during training.

### Headline comparison

The clearest sample-level point-level relative L2 comparison available from the supplied results is:

| Metric | FNO | TransolverAR |
|---|---:|---:|
| Relative L2 — sample 0 | 0.5486 | **0.1226** |
| Relative L2 — sample 0, \|u\| error | 0.4889 | **0.0378** |
| Relative L2 — sample 0, $u_x$ error | 0.4940 | **0.0425** |
| Relative L2 — sample 0, $u_y$ error | 0.8091 | **0.1247** |
| Relative L2 — sample 0, $u_z$ error | 0.8292 | **0.0804** |

These values come from the labelled diagnostic figures for the illustrated held-out sample. They show a substantial advantage for **TransolverAR** on that sample.

### Qualitative prediction comparison

Frame-0 predictions for the same held-out sample are shown below.

| FNO — voxel-grid mapping | TransolverAR — direct point cloud |
|:---:|:---:|
| ![FNO 3D wing surface](FNO/outputs/single/000/sample000_frame0_3d_wing_surface.png) | ![Transolver 3D wing surface](Transolver/outputs/single/000/sample000_frame0_3d_wing_surface.png) |
| ![FNO 2D scatter](FNO/outputs/single/000/sample000_frame0_2d_scatter.png) | ![Transolver 2D scatter](Transolver/outputs/single/000/sample000_frame0_2d_scatter.png) |
| ![FNO 2D contour scatter](FNO/outputs/single/000/sample000_frame0_2d_contour_scatter.png) | ![Transolver 2D contour scatter](Transolver/outputs/single/000/sample000_frame0_2d_contour_scatter.png) |

The qualitative comparison highlights the difference between the two representations: **FNO** predicts through a regularised voxel-grid representation, while **TransolverAR** retains the original irregular sampling and performs autoregressive rollout directly on the mesh.

### Component-wise error analysis

#### FNO

![FNO sample component losses](FNO/outputs/single/000/sample000_component_losses.png)

![FNO dataset component loss distribution](FNO/outputs/component_losses/dataset_component_loss_boxplot.png)

#### TransolverAR

![Transolver sample component losses](Transolver/outputs/single/000/sample000_component_losses.png)

![Transolver dataset component loss distribution](Transolver/outputs/component_losses/dataset_component_loss_boxplot.png)

The dataset-level component plots show substantially higher error distributions for FNO across $u_x$, $u_y$, $u_z$, and $\|u\|$, while TransolverAR keeps the corresponding distributions lower. The remaining error for TransolverAR is largest in the $u_z$ channel among the illustrated components.

### Batch metrics

The full sample-by-sample metrics are available in the generated CSV reports:

- [FNO batch metrics](FNO/outputs/report/batch_report_metrics.csv)
- [TransolverAR batch metrics](Transolver/outputs/report/batch_report_metrics.csv)

These reports are the appropriate place to inspect individual sample rankings and relative-L2 values.

### Interpretation

The results support the design hypothesis behind FLOREN:

- **Voxel-grid mapping** provides a clean, structured operator-learning baseline and makes the irregular data easier to batch.
- **Direct point-cloud modelling** avoids rasterisation and interpolation back to the original points, preserving the irregular geometric representation.
- On the supplied sample-level qualitative and quantitative results, **TransolverAR outperforms the FNO voxel-grid baseline**.

The comparison therefore suggests that, for this geometry-conditioned flow-rollout task, the information lost through voxelisation/interpolation can be significant, while direct irregular-mesh modelling provides substantially higher predictive fidelity.

> **Note:** the reported headline values above are sample-level diagnostic values, not a dataset-wide mean. The batch CSV files should be used for a complete dataset-level statistical comparison.

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

# FLOREN (FNO vs. TransolverAR)

This page summarises the main benchmark results for **FLOREN** on the **Warped-IFW** geometry-conditioned unsteady-flow task.

We compare two modelling pipelines:

- **FNO** -- a voxel-grid baseline that rasterises irregular point clouds onto a fixed 3D grid and predicts future flow on that grid before interpolating back to points.
- **TransolverAR** -- an autoregressive point-cloud model that operates directly on irregular meshes without voxelisation.

The task is to predict **5 future velocity frames** from **5 past velocity frames**, under a **geometry-disjoint split** so test geometries are unseen during training.

## Setup

| | Value |
|---|---|
| Dataset | Warped-IFW |
| Train / test protocol | Geometry-disjoint split |
| Input per point | 5 past velocity frames $$+$$ distance-to-surface |
| Output | 5 future velocity frames |
| FNO representation | Voxelised 3D grid, then trilinear interpolation back to points |
| TransolverAR representation | Direct irregular point cloud / mesh |
| Primary reported metric | Point-level relative L2 |
| Additional diagnostics | Per-sample metrics, per-component loss plots, static visualisations |

## Prediction

Frame-0 qualitative comparison on a held-out sample.

| FNO (voxel grid mapping) | TransolverAR (direct point cloud) |
|:---:|:---:|
| ![FNO 3D wing](sandbox:/mnt/data/assistant-6uvvCxofxefuHyUYyENGnr-FNO_sample000_frame0_3d_wing_surface.png) | ![Transolver 3D wing](sandbox:/mnt/data/assistant-TLoC5NXqq4ZmGoJH1N2vn5-Transolver_sample000_frame0_3d_wing_surface.png) |
| ![FNO scatter](sandbox:/mnt/data/assistant-41A4mCADEoedvr9xmtTvR1-FNO_sample000_frame0_2d_scatter.png) | ![Transolver scatter](sandbox:/mnt/data/assistant-VvUPmn6AQKTinm6rZrhhyr-Transolver_sample000_frame0_2d_scatter.png) |
| ![FNO contour](sandbox:/mnt/data/assistant-LJuS1a44NBjPFYq23DUbAj-sample000_frame0_2d_contour_scatter.png) | ![Transolver contour](sandbox:/mnt/data/assistant-9sxjNJCh3a81CMtS8ttoPV-Transolver_sample000_frame0_2d_contour_scatter.png) |

The qualitative plots show the main difference between the two approaches: **FNO** predicts through a regularised voxel-grid representation, while **TransolverAR** keeps the original irregular sampling and rolls out autoregressively on the mesh itself.

## Accuracy and diagnostics

### Training dynamics

FNO training curves are available below. These are plotted in **normalised grid space**, so they are useful for optimisation diagnostics but are **not directly the same metric** as the final point-level relative L2 used for model comparison.

![FNO training curves](sandbox:/mnt/data/assistant-5zPuykB7AQGJKzM1nGndL3-FNO_training_curves.png)

### Per-sample / per-component diagnostics

| FNO | TransolverAR |
|:---:|:---:|
| ![FNO component losses sample](sandbox:/mnt/data/assistant-MnbS1dyoAbh8p1gJvkFypX-FNO_sample000_component_losses.png) | ![Transolver component losses sample](sandbox:/mnt/data/assistant-5wzP9kKkmwuy5sfcbtYkVh-Transolver_sample000_component_losses.png) |
| ![FNO component boxplot](sandbox:/mnt/data/assistant-4uwh9XJyu6ZymBdQ81jA6N-FNO_dataset_component_loss_boxplot.png) | ![Transolver component boxplot](sandbox:/mnt/data/assistant-X9p5qkYTNk5qugRxEh6xht-Transolver_dataset_component_loss_boxplot.png) |

For the Transolver pipeline, an additional multi-sample $$u_x$$ summary is available:

![Transolver ux summary](sandbox:/mnt/data/assistant-G541qwuUXxp3CzrPZjxBCs-Transolver_ux_summary_frame0_4x3.png)

## Summary metrics

Using the provided result artifacts, the clearest headline comparison available is the test-set **point-level relative L2**:

| Metric | FNO | TransolverAR |
|---|---:|---:|
| Relative L2 (sample 0 shown on 3D figure) | 0.5486 | **0.1226** |
| Relative L2 (sample 0, $$|u|$$ error panel) | 0.4889 | **0.0378** |
| Relative L2 (sample 0, $$u_x$$ error panel) | 0.4940 | **0.0425** |
| Relative L2 (sample 0, $$u_y$$ error panel) | 0.8091 | **0.1247** |
| Relative L2 (sample 0, $$u_z$$ error panel) | 0.8292 | **0.0804** |

These numbers come directly from the labelled diagnostic figures you provided. They show a large gap in favour of **TransolverAR** on the illustrated held-out sample.

### Dataset-level component behaviour

The dataset-level boxplots also indicate a strong difference in component-wise error distributions:

- **FNO** shows substantially larger spread and higher central error for $$u_x$$, $$u_y$$, $$u_z$$, and $$|u|$$.
- **TransolverAR** keeps all four component distributions much lower, with especially strong improvement on $$u_x$$ and $$|u|$$.
- The largest remaining error channel for TransolverAR appears to be $$u_z$$, but it is still well below the corresponding FNO distribution.

## Batch metrics files

The repository also contains per-run CSV summaries for both models:

- [FNO_batch_report_metrics.csv](sandbox:/mnt/data/assistant-GQ8V5rT3mvVBfMJzwAwbVF-FNO_batch_report_metrics.csv)
- [Transolver_batch_report_metrics.csv](sandbox:/mnt/data/assistant-4MFYQn3gJejnUJvCtLyf9c-Transolver_batch_report_metrics.csv)

These are the most appropriate tabular artifacts to inspect when you want sample-by-sample relative-L2 rankings for the two pipelines.

## Interpretation

This comparison supports the main design hypothesis behind FLOREN:

- **Voxel-grid mapping makes the problem easier to batch and train**, and provides a clean operator-learning baseline.
- However, **direct point-cloud modelling preserves geometric detail better** and avoids rasterisation / interpolation error.
- On the supplied qualitative and sample-level quantitative results, **TransolverAR clearly outperforms the FNO voxel-grid baseline**.

In other words, for this geometry-conditioned rollout task, the cost of forcing irregular CFD-style point clouds onto a fixed grid appears to be non-trivial, and the direct irregular-mesh model gives substantially better predictive fidelity.

## Notes

- The FNO training-curve overlay explicitly notes that its plotted optimisation loss is a **grid-space normalised metric**, not the final point-level competition metric.
- If you later generate final dataset-level scalar summaries such as test-set mean relative L2, standard deviation, nRMSE, runtime per epoch, or best-epoch values for both models, this page can be extended into a fuller benchmark table in exactly the same style as the Darcy comparison page.

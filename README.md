# FLOREN Results (Fourier Neural Operator vs. Transolver)

FLOREN studies **geometry-conditioned unsteady flow prediction** on the **Warped-IFW** benchmark. The task is to forecast **5 future velocity frames from 5 past velocity frames** on irregular CFD point clouds while generalising to **unseen wing geometries**.

The benchmark is motivated by aerodynamic flow modelling around a warped front-wing configuration derived from an **Imperial College / McLaren-style front-wing setup**, commonly described in the Warped-IFW dataset and GRaM competition materials as being based on a front-wing geometry developed at Imperial and inspired by a **McLaren MP4-17 / early-2000s F1 front-wing configuration**, often associated with the 2002 era. In FLOREN, each sample contains a 3D point cloud around the wing, surface-point indices, pressure, past velocity history, and future velocity targets. FLOREN compares two ways of handling that geometry:

- **Fourier Neural Operator** -- a voxel-grid baseline that rasterises the irregular point cloud onto a fixed 3D grid, predicts future flow on that grid, and interpolates predictions back to the original points.
- **Transolver** -- an autoregressive point-cloud model that operates directly on the irregular mesh / point cloud without voxelisation.

Useful references:

- [Warped-IFW dataset on Hugging Face](https://huggingface.co/datasets/gram-competition/warped-ifw)
- [GRaM Competition @ ICLR 2026](https://gram-competition.github.io/)
- [Competition / dataset paper PDF](https://raw.githubusercontent.com/mlresearch/v326/main/assets/suk26a/suk26a.pdf)

This benchmark is a **Fourier Neural Operator vs. Transolver comparison** on Warped-IFW, and the repository itself is called **FLOREN**.

## Setup

| | Value |
|---|---|
| Dataset | Warped-IFW |
| Task | Predict 5 future velocity frames from 5 past frames |
| Split | Geometry-disjoint train / test |
| Input per point | 5 past velocity frames + distance-to-surface |
| Output | 5 future velocity frames |
| Fourier Neural Operator representation | Voxelised 3D grid -> interpolate back to points |
| Transolver representation | Direct irregular mesh / point cloud |
| Primary reported metric | Point-level relative L2 |
| Additional diagnostics | Per-sample metrics, component-loss plots, static visualisations |

Reproduce the two pipelines from the repository root:

```bash
cd fno
python train.py
python visualise_fno.py

cd ../transolver
python train.py --profile full
python visualise.py
```

## Competition standing

Using the available **Transolver** dataset-level result of **0.0638 ± 0.0198** from the competition-style table provided, the model would place:

| Ranking | Model | Relative L2 error |
|---|---|---:|
| 1st | SmoothSplatNet | 0.0480 ± 0.0146 |
| 2nd | CDFDoubleGridNet | 0.0498 ± 0.0144 |
| 3rd | VRTEnsemble | 0.0510 ± 0.0151 |
| 4th | Kagent | 0.0553 ± 0.0168 |
| 5th | ResMLP | 0.0560 ± 0.0169 |
| 6th | EnsembleSpatioTemporalModels | 0.0589 ± 0.0184 |
| 7th | Transolver | 0.0638 ± 0.0198 |
| 8th | TransolverCorrector | 0.0739 ± 0.0203 |
| 9th | gEGNO | 0.0819 ± 0.0233 |
| 10th | FiniteGraphV4 | 0.0837 ± 0.0230 |
| 11th | AB-UPT | 0.0838 ± 0.0227 |
| 12th | AirFormer | 0.0850 ± 0.0227 |
| 13th | AeroChronoMixer | 0.0892 ± 0.0232 |
| 14th | Transolver Residual | 0.0896 ± 0.0244 |
| 15th | LeversTailV2Submission | 0.0970 ± 0.0230 |
| **16th** | **FLOREN** | **0.0991** |
| 17th | ImprovedMLP | 0.1002 ± 0.0238 |
| 18th | FNO3DTimeRes | 0.1224 ± 0.0218 |
| 19th | DeltaGraph | 0.1258 ± 0.0278 |
| 20th | SpatiotemporalMNO | 0.1292 ± 0.0284 |
| 21st | WaveletLatentOperator | 0.1978 ± 0.0163 |
| 22nd | PerceiverFlow | 0.2436 ± 0.0406 |
| 23rd | ZonalMoE | 0.5807 ± 0.1091 |

This places **Transolver in 7th place** on the supplied leaderboard table. If **FLOREN** achieved **0.0991** relative L2, it would place **16th**, ahead of **ImprovedMLP**, and behind **LeversTailV2Submission** in the supplied ranking.

## Prediction

Frame-0 qualitative comparison on a held-out sample:

| Fourier Neural Operator (voxel-grid mapping) | Transolver (direct point cloud) |
|:---:|:---:|
| ![FNO 3D wing](FNO/outputs/single/000/sample000_frame0_3d_wing_surface.png) | ![Transolver 3D wing](Transolver/outputs/single/000/sample000_frame0_3d_wing_surface.png) |
| ![FNO scatter](FNO/outputs/single/000/sample000_frame0_2d_scatter.png) | ![Transolver scatter](Transolver/outputs/single/000/sample000_frame0_2d_scatter.png) |
| ![FNO contour](FNO/outputs/single/000/sample000_frame0_2d_contour_scatter.png) | ![Transolver contour](Transolver/outputs/single/000/sample000_frame0_2d_contour_scatter.png) |

A multi-sample qualitative summary for Transolver is also included and provides evidence that the model works across multiple unseen airfoil / wing configurations rather than only a single held-out case:

![Transolver ux summary across multiple samples](Transolver/outputs/single/ux_summary_frame0_4x3.png)

## Accuracy

The most directly labelled quantitative comparison available from the supplied diagnostic outputs is the held-out **sample-0 point-level relative L2**:

| Metric | Fourier Neural Operator | Transolver |
|---|---:|---:|
| Relative L2 (sample 0 overall) | 0.5486 | **0.1226** |
| Relative L2 (sample 0, |u| error panel) | 0.4889 | **0.0378** |
| Relative L2 (sample 0, ux error panel) | 0.4940 | **0.0425** |
| Relative L2 (sample 0, uy error panel) | 0.8091 | **0.1247** |
| Relative L2 (sample 0, uz error panel) | 0.8292 | **0.0804** |

These are **sample-level diagnostic values**, not final dataset-wide means for both repository pipelines, but they provide the clearest like-for-like comparison available from the supplied result artifacts. For Transolver, the competition-style dataset-level result above additionally provides broader ranking context. FLOREN's own reported result is **0.0991**, which would place it **16th** on the supplied ranking.

## Accuracy diagnostics

| Fourier Neural Operator | Transolver |
|:---:|:---:|
| ![FNO training curves](FNO/outputs/diagnostics/training_curves.png) | ![Transolver ux summary](Transolver/outputs/single/ux_summary_frame0_4x3.png) |

For the Fourier Neural Operator, the training-curve panel explicitly shows that the optimisation curves are measured in **normalised grid space**, which is not the same metric as the final point-level comparison metric.

## Component-wise diagnostics

| Fourier Neural Operator | Transolver |
|:---:|:---:|
| ![FNO component losses sample](FNO/outputs/single/000/sample000_component_losses.png) | ![Transolver component losses sample](Transolver/outputs/single/000/sample000_component_losses.png) |
| ![FNO component boxplot](FNO/outputs/component_losses/dataset_component_loss_boxplot.png) | ![Transolver component boxplot](Transolver/outputs/component_losses/dataset_component_loss_boxplot.png) |

The component-loss plots show a consistent gap in favour of **Transolver**. In the supplied diagnostics, the Fourier Neural Operator exhibits substantially larger errors in **uy** and **uz**, while Transolver keeps all four reported channels lower, including **|u|**.

## Batch metrics files

The most appropriate tabular files for sample-by-sample comparison are the batch-report CSVs:

- [FNO_batch_report_metrics.csv](FNO/outputs/report/batch_report_metrics.csv)
- [Transolver_batch_report_metrics.csv](Transolver/outputs/report/batch_report_metrics.csv)

These are the best CSV artifacts to inspect when you want per-sample relative-L2 rankings for the two pipelines.

## Summary

| Metric | Fourier Neural Operator | Transolver |
|---|---|---|
| Representation | Voxelised 3D grid -> interpolate back to points | Direct irregular mesh / point cloud |
| Geometry handling | Regular grid mapping | Native irregular geometry |
| Held-out sample 0 relative L2 | 0.5486 | **0.1226** |
| Held-out sample 0 |u| relative L2 | 0.4889 | **0.0378** |
| Held-out sample 0 ux relative L2 | 0.4940 | **0.0425** |
| Held-out sample 0 uy relative L2 | 0.8091 | **0.1247** |
| Held-out sample 0 uz relative L2 | 0.8292 | **0.0804** |
| Competition-style dataset result | not supplied in matched form | **0.0638 ± 0.0198** |
| FLOREN result | not supplied in matched form | **0.0991** |
| Approx. competition rank | not directly established from supplied artifacts | **7th (Transolver), 16th (FLOREN @ 0.0991)** |
| Multi-geometry qualitative evidence | Not supplied in comparable summary form | Included via multi-sample ux summary |

Overall, the supplied FLOREN comparison indicates that **direct irregular point-cloud modelling substantially outperforms voxel-grid mapping** on this benchmark. The strongest evidence comes from the large sample-level relative-L2 gap, the lower component-wise error distributions, the multi-geometry qualitative summary, and the available competition-style ranking context.

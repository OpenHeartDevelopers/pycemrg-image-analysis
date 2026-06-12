# API Reference

The toolbox map. `pycemrg-image-analysis` is a **stateless** library: every module
below accepts in-memory data and returns in-memory data — none of it reads or
writes files (except the I/O utilities, which your orchestrator calls explicitly).
For the design rationale and the orchestration pattern that ties these together,
start with [Architecture](overview.md).

## Core: building anatomy

| Module | What it does | Page |
|---|---|---|
| Recipes | Named, ordered workflows (`get_recipe`, `list_recipes`) and the `Recipe`/`WorkflowStep` contracts | [Recipes & Workflows](recipes.md) |
| Components | Connected-component cleanup of segmentation artefacts — spatial ops + semantic wrappers | [Components](components.md) |

## Utilities: I/O & spatial queries

| Module | What it does | Page |
|---|---|---|
| I/O | Read/write images, INR ↔ NIfTI conversion, round-trip behaviour | [I/O Utilities](io.md) |
| Spatial | Voxel ↔ physical coordinate mapping, slice extraction, mesh sampling | [Spatial Queries](spatial.md) |
| Label tools | `LabelDiagnostic`, `LabelRemapper`, `check_labels` — reconcile non-standard label schemes | [Label Tools](label_tools.md) |

## Dataset prep: super-resolution training

| Module | What it does | Page |
|---|---|---|
| Augmentation | Intensity & spatial augmentation for SR training pipelines | [Augmentation](augmentation.md) |
| Metrics | Volume-comparison metrics (MSE, PSNR, SSIM, gradient error) | [Metrics](metrics.md) |

!!! info "Looking for the orchestration pattern?"
    The high-level scaffold → load → run → save sequence, the logic engines
    (`MyocardiumLogic`, `ValveLogic`, …), and the frozen `*CreationContract`
    dataclasses are documented on the [Architecture](overview.md) page.

# pycemrg-image-analysis

Python library for cardiac image segmentation post-processing, built on SimpleITK.
Maintained by the [Cardiac Electromechanics Research Group (CEMRG)](https://www.cemrg.com/) at Imperial College London.

---

## What this library does

`pycemrg-image-analysis` is a **stateless toolbox** for turning whole-heart segmentations
into analysis-ready cardiac models. It handles:

- Myocardium creation (LV, RV, LA, RA walls, aortic/pulmonary artery walls)
- Valve surface creation (mitral, tricuspid, aortic, pulmonary)
- Vein ring creation (pulmonary veins, vena cavae, LAA)
- Connected-component cleanup of segmentation artefacts
- Label diagnostics and remapping for non-standard label schemes
- Data augmentation for super-resolution training pipelines
- Mesh–image spatial queries (voxel ↔ physical coordinate mapping)
- Image quality metrics (MSE, PSNR, SSIM, gradient error)

---

## Where to go next

- **[Getting Started](getting-started/index.md)** — install the library and run your first scaffolded workflow.
- **[Tutorial: Custom Heartbuilder Pipeline](tutorials/custom_heartbuilder.md)** — a guided, command-by-command build from a fresh install.
- **[Architecture](api/overview.md)** — the stateless-toolbox philosophy and the canonical orchestration pattern.
- **[Recipes & Workflows](api/recipes.md)** — named, ordered workflows for common cardiac models.
- **[API Reference](api/index.md)** — the toolbox map: components, I/O, spatial, label tools, augmentation, metrics.
- **[CLI Reference](cli/labels.md)** — command-line label inspection and remapping.
- **[Developer Guides](guides/authoring_a_recipe.md)** — SOPs for extending the library with your own schematics and recipes.

---

## Design principles

| Principle | What it means in practice |
|---|---|
| **Stateless library** | Logic engines accept data — they never read from or write to disk |
| **Explicit contracts** | Every logic call is driven by a frozen dataclass (`*CreationContract`) |
| **Orchestrator owns I/O** | Your script handles file paths, config loading, and step sequencing |
| **No singletons / globals** | All dependencies are injected at construction time |

---

## Key domain terms

| Term | Meaning |
|---|---|
| **BP (Blood Pool)** | Cavity label (LV_BP, RV_BP, LA_BP, RA_BP) |
| **Myo (Myocardium)** | Muscle wall derived by growing outward from blood pools |
| **Semantic Map** | JSON mapping role enums → label names / integer values |
| **Recipe** | Named sequence of operations (e.g., `biventricular_basic`) |
| **Contract** | Frozen dataclass passed to a logic engine |
| **Schematic** | Pre-defined anatomy blueprint (labels + parameters + semantic map) |
| **LabelDiagnostic** | Compares image labels against a named schematic |
| **LabelRemapper** | Builds integer→integer mappings to reconcile label schemes |

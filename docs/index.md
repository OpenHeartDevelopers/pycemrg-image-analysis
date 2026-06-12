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

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Install the library and run your first scaffolded workflow.

    [:octicons-arrow-right-24: Getting Started](getting-started/index.md)

-   :material-sitemap:{ .lg .middle } __Architecture__

    ---

    The stateless-toolbox philosophy and the canonical orchestration pattern.

    [:octicons-arrow-right-24: Architecture](api/overview.md)

-   :material-book-open-variant:{ .lg .middle } __Recipes & Workflows__

    ---

    Named, ordered workflows for common cardiac models.

    [:octicons-arrow-right-24: Recipes](api/recipes.md)

-   :material-toolbox:{ .lg .middle } __API Reference__

    ---

    The toolbox map: components, I/O, spatial, label tools, augmentation, metrics.

    [:octicons-arrow-right-24: API Reference](api/index.md)

-   :material-console:{ .lg .middle } __CLI__

    ---

    Command-line label inspection and remapping.

    [:octicons-arrow-right-24: CLI Reference](cli/labels.md)

-   :material-hammer-wrench:{ .lg .middle } __Developer Guides__

    ---

    SOPs for extending the library with your own schematics and recipes.

    [:octicons-arrow-right-24: Developer Guides](guides/authoring_a_recipe.md)

</div>

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

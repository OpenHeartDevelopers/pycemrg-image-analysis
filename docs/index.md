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

## Quick start

```python
from pathlib import Path
from pycemrg_image_analysis.recipes import get_recipe
from pycemrg_image_analysis import ImageAnalysisScaffolder
from pycemrg_image_analysis.utilities import load_image, save_image

# 1. Pick a recipe
recipe = get_recipe("biventricular_basic")

# 2. Scaffold config files
scaffolder = ImageAnalysisScaffolder()
scaffolder.scaffold_components(Path("config/"), recipe.required_schematics)

# 3. Load your segmentation and run the workflow
#    (see API > Overview for the full orchestration pattern)
```

---

## Design principles

| Principle | What it means in practice |
|---|---|
| **Stateless library** | Logic engines accept data — they never read from or write to disk |
| **Explicit contracts** | Every logic call is driven by a frozen dataclass (`*CreationContract`) |
| **Orchestrator owns I/O** | Your script handles file paths, config loading, and step sequencing |
| **No singletons / globals** | All dependencies are injected at construction time |

---

## Navigation

- **[API Reference →](api/overview.md)** — Architecture guide and full module reference
- **[CLI Reference →](cli/labels.md)** — Command-line tools for label inspection and remapping
- **[Developer Guides →](guides/add_myocardium_component.md)** — SOPs for extending the library

---

## Installation

```bash
pip install pycemrg-image-analysis
# or, in editable mode from source:
pip install -e .
```

**Requires Python ≥ 3.10**

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

# Getting Started

This page gets you from a fresh install to your first scaffolded workflow, then
points you at the right next step depending on what you're doing.

## Installation

```bash
pip install pycemrg-image-analysis
# or, in editable mode from source:
pip install -e .
```

`pycemrg-image-analysis` has a hard runtime dependency on the sibling package
[`pycemrg`](https://github.com/OpenHeartDevelopers/pycemrg) (label management +
`ConfigScaffolder`). From a source checkout, install it first:

```bash
pip install -e ../pycemrg/ && pip install -e .
```

**Requires Python ≥ 3.10.**

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
#    (see Architecture for the full orchestration pattern)
```

## Your first workflow — a learning path

!!! tip "Prefer a guided, end-to-end walkthrough?"
    The [Custom Heartbuilder Pipeline](../tutorials/custom_heartbuilder.md)
    tutorial takes you from a fresh install to a project that grows a custom model
    from your own segmentation, command by command.

Follow these in order the first time through:

1. **[Architecture](../api/overview.md)** — the stateless-toolbox philosophy and
   the canonical orchestration pattern (scaffold → load → run → save). Read this
   before writing any orchestrator code.
2. **[Recipes & Workflows](../api/recipes.md)** — named, ordered workflows like
   `biventricular_basic`. The fastest way to a correct sequence of steps.
3. **[API Reference](../api/index.md)** — the toolbox map. Jump to the module you
   need: components, I/O, spatial queries, label tools, augmentation, metrics.
4. **[CLI](../cli/labels.md)** — command-line label diagnostics and remapping when
   your segmentation's label values don't match the schematic defaults.
5. **[Developer Guides](../guides/authoring_a_recipe.md)** — SOPs for extending the
   library with your own schematics and recipes.

!!! tip "Label values matter"
    Tools like 3D Slicer reset labels to sequential `1..N`. If your file's label
    values differ from the schematic defaults, walls grow from the wrong voxels
    **with no error**. Diagnose and fix this with the
    [label tools](../api/label_tools.md) or the [CLI](../cli/labels.md) before
    running a workflow.

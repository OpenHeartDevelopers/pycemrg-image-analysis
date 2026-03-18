# pycemrg-image-analysis

Python library for cardiac image segmentation post-processing, built on SimpleITK.
Maintained by the Cardiac Electromechanics Research Group (CEMRG) at Imperial College London.

## Commands

```bash
# Install in editable mode
pip install -e .

# Run all unit tests (always available)
pytest tests/unit/

# Run integration tests (requires test data)
PYCEMRG_TEST_DATA_ROOT=/path/to/data pytest tests/integration/

# Run everything
pytest
```

No Makefile, tox, or linting config. Build system is `pyproject.toml` + setuptools only.

## Architecture

```
src/pycemrg_image_analysis/
  logic/           # Stateless computation engines (MyocardiumLogic, ValveLogic, etc.)
  schematics/      # Pre-defined anatomy blueprints (LV, RV, valves, rings)
  utilities/       # Stateless data transformation (masks, filters, spatial, IO, etc.)
  recipes.py       # Pre-sequenced workflow variants (6 standard recipes)
  scaffolding.py   # Generates YAML/JSON config files for workflows
```

`logic/` engines receive contracts (frozen dataclasses from `logic/contracts.py`) and
return results. They never touch the filesystem. Orchestration happens outside this
library.

`utilities/` modules are pure functions — import and call, no class state.

## Domain Terminology

- **BP (Blood Pool):** Cavity label (LV_BP, RV_BP, LA_BP, RA_BP)
- **Myo (Myocardium):** Muscle wall derived by growing outward from blood pools
- **Semantic Map:** JSON mapping role enums → label names/integer values
- **Recipe:** Named sequence of operations (e.g., `biventricular_basic`, `four_chamber_full`)
- **Contract:** Frozen dataclass passed to a logic engine
- **Application Step:** A single mask operation (add/replace/keep) in a processing sequence
- **Label Manager:** `pycemrg` class mapping human-readable names ↔ integer voxel labels
- LV/RV = Left/Right Ventricle; LA/RA = Left/Right Atrium
- LPV/RPV = Left/Right Pulmonary Vein; SVC/IVC = Superior/Inferior Vena Cava

## Non-Obvious Constraints

**Image spacing must be correct.** `MyocardiumLogic` and `ValveLogic` use physical-space
distance maps. Wrong spacing silently produces wrong wall thicknesses.

**Application step order is critical.** Steps write to the same output array; wrong
ordering will silently overwrite anatomy. `MyocardiumRule.application_steps` list order
determines final result.

**Component cleanup function choice matters.**
- `keep_largest_component` — cleans each label independently (disconnected anatomy)
- `keep_largest_structure` — treats specified labels as one structure (removes floating debris)
Choosing wrong variant silently keeps incorrect anatomy.

**3D Slicer label resets.** Slicer often resets labels to sequential 1–N after editing.
Use `scaffolder.scaffold_components_with_mapping()` when label values differ from defaults.

**Integration tests are skipped, not failed, when data is absent.** `conftest.py`
calls `pytest.skip()` — missing `PYCEMRG_TEST_DATA_ROOT` causes silent skips.

## Dependency Notes

- `pycemrg>=0.1.0` (parent library, not in this repo) provides `LabelManager` and
  `ConfigScaffolder`. Breaking changes there cascade here.
- SimpleITK signed distance maps: thresholding with a negative lower bound captures
  regions on both sides of a surface — intentional in valve/ring creation.

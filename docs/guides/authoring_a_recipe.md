# Authoring a New Recipe

This guide walks you from a raw segmentation to a brand-new, runnable recipe.
A *recipe* is a named, ordered sequence of operations that grows anatomy out of
a label map (myocardium from blood pools, valves from intersections, vein rings,
and "push" corrections between touching walls).

By the end you will have:

- one or more component *schematics* (the source of truth for labels, parameters,
  and the operations that build a structure),
- a `Recipe` entry registered in the catalog, and
- a working run driven by `run_recipe_workflow` from
  `examples/orchestrator_patterns.py`.

If you only want to *run* an existing recipe (e.g. `biventricular_basic`), skip to
[Step 5](#step-5-run-it) and pass an existing recipe name.

---

## Mental model

Data flows in one direction, and each stage has a single owner:

```
schematic (Python dict)          <- you author this
   |  scaffolder
   v
config on disk:
  labels.yaml          (name -> integer voxel value)
  parameters.json      (name -> wall thickness, etc.)
  semantic_maps/*.json (one per "create"/"valve"/"ring" component)
   |  orchestrator loads config + segmentation
   v
logic engine                      <- stateless; never touches the filesystem
   |
   v
new segmentation image
```

Two things are worth internalising now, because they explain the rest of the guide:

1. **The engine API is not uniform across structure types.**
   - Myocardium: `MyocardiumLogic.create_from_semantic_map(image, label_manager, parameters, semantic_map)` — consumes the semantic-map dict directly. No contract, no rule.
   - Valves and rings: `ValveLogic.create_from_rule(contract)` / `RingLogic.create_from_rule(contract)` — consume a frozen contract that carries a rule.
   - Push: `MyocardiumLogic.push_structure(image, contract)` — a `PushStructureContract` built by hand, with no semantic map.

   The reference runner dispatches on `step_type` to paper over this. You do not
   need to unify it, but you do need to know which branch your new step lands in.

2. **A schematic is the source of truth.** You do not edit `labels.yaml` or the
   semantic-map JSON by hand. You author a schematic dict; the scaffolder emits
   the config. This keeps label names, integer values, and operations in one place.

---

## Step 1: Inventory your segmentation

Before writing anything, answer three questions about the input `.nrrd`/`.nii`:

**Which integer labels are present, and what anatomy does each mean?**
Map them to the project's label-name vocabulary (`LV_BP_label`, `RA_BP_label`,
`Ao_wall_label`, ...). See the domain terminology at the bottom of this tutorial.

**Is the image spacing physically correct?** Myocardium, valve, and ring logic
all use physical-space distance maps. Wrong spacing silently produces wrong wall
thicknesses, it won't show an error, just a bad result.

**Do your label *values* match the schematic defaults?** Tools like 3D Slicer
often reset labels to a sequential `1..N` after editing. If your `LA_myo_label`
is `5` in the file but the schematic template says `104`, the scaffolded
`labels.yaml` will be wrong, and the engine will grow walls from the wrong voxels.

### Diagnose label values

The quickest check is the `check_labels` convenience wrapper, which loads the
image, compares it against a schematic's expected labels, and prints a report:

```python
from pathlib import Path
from pycemrg_image_analysis.utilities.label_tools import check_labels

report = check_labels(Path("seg_input.nrrd"), "biventricular_basic")
# Prints a report; also returns a DiagnosticReport you can branch on:
if report.has_issues:
    print("Missing:", [m.label_name for m in report.missing_labels])
```

(Use `list_available_schematics()` from the same module to see valid schematic
names.) Under the hood this is `LabelDiagnostic().check_image_against_schematic(...)`
if you want the report without printing.

### Fix a mismatch

You do not edit `labels.yaml` by hand. Instead, build a `{label_name: your_int}`
mapping and let the scaffolder bake your values into the config. For the common
Slicer 1..N case, `LabelRemapper` can propose the mapping for you:

```python
from pycemrg_image_analysis.utilities.label_tools import LabelRemapper

remapper = LabelRemapper()
mapping = remapper.suggest_mapping_from_report(report)  # None if not 1..N sequential
# Or write it explicitly when the auto-suggestion can't apply:
mapping = {"LV_BP_label": 1, "LV_myo_label": 2, "RV_BP_label": 3}
```

Then pass that mapping straight to the runner (see [Step 5](#step-5-run-it)),
which scaffolds with your values via `scaffold_components_with_mapping` instead of
the schematic defaults. The mapping only needs to cover the labels that differ.

---

## Step 2: Define your component schematic(s)

A schematic is just a plain Python dict with three keys — `labels`, `parameters`,
and `semantic_map`. You do **not** need to edit the installed library to add one:
define it in your own project and inject it (next subsection). The only import you
need from the library is the role enum:

```python
# in your own project, e.g. my_schematics.py
from pycemrg_image_analysis.logic.constants import MyocardiumSemanticRole

MY_SCHEMATICS = {
    "my_new_wall": {
        "labels": {"Foo_BP_label": 9, "Foo_myo_label": 109, "Bar_BP_label": 3},
        "parameters": {"Foo_WT": 2.5},
        "semantic_map": {
            MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "Foo_BP_label",
            MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "Foo_myo_label",
            MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "Foo_WT",
            MyocardiumSemanticRole.APPLICATION_STEPS: [
                {"MODE": "REPLACE_ONLY", "RULE_LABEL_NAMES": ["Bar_BP_label"]},
            ],
        },
    },
}
```

What each field means:

- **`labels`** — every label name this component reads or writes, with its default
  integer value. The scaffolder merges these across all components in a recipe into
  one `labels.yaml`.
- **`parameters`** — numeric inputs, typically wall thickness in physical units.
- **`semantic_map`** — keyed by `MyocardiumSemanticRole` (see `logic/constants.py`):
  the source blood pool to grow from, the target label to write, the thickness
  parameter, and an ordered list of `APPLICATION_STEPS`.

### Registering your schematic

Pass your dict to the scaffolder via `extra_schematics`. It is merged over the
built-ins per instance (your entries win on a name collision), so the built-in
components remain available alongside yours:

```python
from pycemrg_image_analysis import ImageAnalysisScaffolder

scaffolder = ImageAnalysisScaffolder(extra_schematics=MY_SCHEMATICS)
scaffolder.scaffold_components(config_dir, ["lv_outflow", "my_new_wall"])
```

The runner in [Step 5](#step-5-run-it) takes the same `extra_schematics` argument
and forwards it, so a custom recipe can reference your component end to end.

> **Contributing upstream?** If you are adding a component to the library itself
> (not just your project), put the dict in the matching file under
> `src/pycemrg_image_analysis/schematics/` (`myocardium.py`, `valves.py`, or
> `rings.py`). It is picked up automatically via the `**` spread in
> `schematics/__init__.py` that builds `ALL_SCHEMATICS` — no `extra_schematics`
> needed in that case.

### Application steps and modes

Each application step is `{"MODE": <MaskOperationMode name>, "RULE_LABEL_NAMES": [...]}`.
The engine grows a wall mask from the source blood pool, then applies the steps in
order onto the working array. **Order is critical**: steps write to the same output
array, so a later step can silently overwrite anatomy a previous step placed. Common
modes (see `utilities/masks.py` / `utilities/dispatchers.py` for the full set):

- `ADD` — write the new label wherever the mask is set.
- `REPLACE_ONLY` — write only over the listed labels (`RULE_LABEL_NAMES`).
- `REPLACE_EXCEPT` — write everywhere in the mask except over the listed labels.

If you need a *new* mode, add the function in `utilities/masks.py` (with a unit
test), add the enum value to `MaskOperationMode`, and register it in
`utilities/dispatchers.py` — the single wiring point. Do not wire it ad hoc.

Valve and ring schematics follow the same shape but use `ValveSemanticRole` /
`RingSemanticRole` (structure-A/structure-B intersection for valves; source vein +
atrium myocardium for rings). See `schematics/valves.py` and `schematics/rings.py`
for concrete examples to copy.

---

## Step 3: Define the recipe

A `Recipe` is also just a dataclass you can build in your own project. The runner
in [Step 5](#step-5-run-it) accepts a `Recipe` object directly, so you do not need
to register it in the library to run it:

```python
from pycemrg_image_analysis.recipes import Recipe, WorkflowStep

MY_RECIPE = Recipe(
    name="my_recipe",
    description="LV wall plus my new wall",
    steps=[
        WorkflowStep("create", "lv_outflow"),
        WorkflowStep("create", "my_new_wall"),
        # WorkflowStep("valve", "..."), WorkflowStep("ring", "..."),
        # WorkflowStep("push", "...")  # see Step 4
    ],
    required_schematics=[
        "lv_outflow",
        "my_new_wall",
    ],
)
```

> **Contributing upstream?** To ship a recipe *with* the library, define it in
> `src/pycemrg_image_analysis/recipes.py` and add it to `RECIPE_CATALOG` so it is
> reachable by name via `get_recipe("my_recipe")`. Project-local recipes skip this
> and pass the object instead.

Two distinct lists, two distinct jobs:

- **`steps`** = execution order via `WorkflowStep(step_type, component_name)`:
    + `step_type` is one of `create`, `valve`, `ring`, `push`; 
    + `component_name` is the schematic name (or push-step key from Step 4).
- **`required_schematics`** = everything the scaffolder must emit config for. This
  is a *superset* of the create/valve/ring component names: it also includes
  parameter-only schematics such as `myo_push_steps`, which carries the labels and
  thicknesses that push steps need but has no semantic map of its own.

Order your `steps` so dependencies exist before they are used: grow a blood pool's
myocardium before a valve or push references that wall. The four-chamber recipes in
`recipes.py` are good worked examples of create-then-push-then-valve ordering.

---

## Step 4: Handle push steps (only if your recipe has them)

A push step shrinks one structure's blood pool inward where a neighbouring wall
intrudes, then relabels that shell as the pushed structure's wall. Push steps are
the one structure type with **no semantic map**: the `myo_push_steps` schematic
only supplies labels and parameters. The mapping from a push-step name to its four
contract fields lives in `_PUSH_STEP_DEFINITIONS` in
`examples/orchestrator_patterns.py`:

```python
_PUSH_STEP_DEFINITIONS = {
    "my_push_step": {
        "pusher_wall": "Ao_wall_label",     # the wall doing the pushing
        "pushed_wall": "Foo_myo_label",     # label written into the shell
        "pushed_bp":   "Foo_BP_label",      # blood pool being eroded
        "thickness_param": "Foo_WT",        # shell thickness (parameters.json)
    },
    # ... existing entries ...
}
```

To add a push step:

1. Add an entry here keyed by the step name.
2. Reference it in your recipe as `WorkflowStep("push", "my_push_step")`.
3. Make sure every label/parameter name you reference is provided by one of the
   recipe's `required_schematics` (add `myo_push_steps`, or your own schematic that
   carries those labels). If a name is missing, `build_push_contract` will raise
   when it tries to resolve it.

---

## Step 5: Run it

For a built-in recipe with default labels, pass the name:

```python
from pathlib import Path
from examples.orchestrator_patterns import run_recipe_workflow

final = run_recipe_workflow(
    "biventricular_basic",
    input_seg_path=Path("seg_input.nrrd"),
    output_dir=Path("output/"),
)
```

For the recipe and schematics you authored in Steps 2-3, pass the `Recipe` object,
your `extra_schematics`, and (if your labels differ from the defaults) the
`label_mapping` from Step 1 — all in one call:

```python
from my_schematics import MY_SCHEMATICS, MY_RECIPE  # your project module

final = run_recipe_workflow(
    MY_RECIPE,
    input_seg_path=Path("seg_input.nrrd"),
    output_dir=Path("output/"),
    label_mapping={"Foo_BP_label": 1, "Foo_myo_label": 5},  # omit if labels match
    extra_schematics=MY_SCHEMATICS,
)
```

`run_recipe_workflow` (in `examples/orchestrator_patterns.py`) scaffolds the recipe's
`required_schematics`, loads the label manager and parameters, then walks `steps` in
order, saving each intermediate result and a final `seg_final_<recipe>.nrrd`. When
`label_mapping` is given it scaffolds with `scaffold_components_with_mapping`, so the
generated `labels.yaml` carries your integer values instead of the schematic
defaults. The intermediates are useful when a step order is wrong: you can open them
in sequence and see exactly where anatomy got overwritten.

`run_recipe_workflow` is example code — copy it into your own orchestrator and adapt
it. Orchestration (file I/O, paths, logging) is intended to live in your code, not in
the library.

---

## Step 6: Verify

- **Labels present:** load the final image and confirm each target label
  (`Foo_myo_label`, valve, ring) actually appears in the array. A missing target
  usually means a later application step overwrote it, or a thickness of zero.
- **Diagnostic pass:** run `LabelDiagnostic` against your recipe's expected
  schematic to get a `DiagnosticReport` of missing/unexpected labels. If labels are
  off, `LabelRemapper.suggest_mapping_from_report()` can derive an int->int fix.
- **Add an integration test:** mirror `tests/integration/test_myocardium_no_cuts.py`.
  Scaffold your components into a `tmp_path`, run the same step loop, and assert the
  final labels exist. Integration tests are skipped (not failed) when
  `PYCEMRG_TEST_DATA_ROOT` is unset, so they cost nothing in CI without data.

---

## Reference: engine signatures

The three structure types take different *inputs*, and they cannot avoid that:
myocardium grows a wall from one blood pool, a valve is the intersection of two
structures, and a ring needs a reference image plus an atrium-myocardium trim.
Those payloads are genuinely different, so a single shared signature would just hide
the differences behind an over-general bag of arguments.

What *is* an avoidable inconsistency is the call *shape*: myocardium takes loose
positional arguments while valves and rings take a single frozen contract. Wrapping
the myocardium inputs in a contract too would unify how you call them without
pretending the payloads are the same. That is a future cleanup, not a blocker; the
runner's `step_type` dispatch absorbs the difference today.

| Step type | Engine call | Built from |
|-----------|-------------|------------|
| `create` (myocardium) | `MyocardiumLogic.create_from_semantic_map(image, label_manager, parameters, semantic_map)` | role-keyed dict from `semantic_maps/<name>.json` |
| `valve` | `ValveLogic.create_from_rule(contract)` | `ValveCreationContract` (carries `ValveRule`) |
| `ring` | `RingLogic.create_from_rule(contract)` | `RingCreationContract` (carries `RingRule` + reference image) |
| `push` | `MyocardiumLogic.push_structure(image, contract)` | `PushStructureContract` from `_PUSH_STEP_DEFINITIONS` |

See also: [add_myocardium_component.md](add_myocardium_component.md) for the deeper
mechanics of adding a single myocardium component (in-repo contribution flow).

---

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
- **LabelDiagnostic:** Compares image labels against a schematic; produces a `DiagnosticReport` (missing/unexpected).
- **LabelRemapper:** Builds int→int mapping; use `suggest_mapping_from_report()` to derive it from a `DiagnosticReport`.
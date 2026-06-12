# Create a Custom Heartbuilder Pipeline

This hands-on tutorial takes you from nothing to a project that grows a custom
cardiac model out of your own segmentation, using **your own rules**. By the end
you will have a project folder, a `labels.yaml`, and your first custom schematic.

Keep the [Authoring a Recipe](../guides/authoring_a_recipe.md) guide open on the
side. That's the mental model and reference; this page is the "type these
commands now" path.

!!! note "Optional steps"
    Steps **0.x** are optional. They help you get started quicker but are not
    compulsory, you can define your own project folder and structure, as long as
    you install the right libraries.

!!! info "Path assumption"
    This tutorial assumes your code lives under `~/code` and your project folder
    is `~/code/tutorial-custom-heart`. Adjust the paths if yours differ.

## 0.1 Prerequisites (optional)

Clone `pycemrg` and install it into an environment. This gives you the
`pycemrg init` command, which scaffolds a whole project structure for you.

=== "conda"

    ```bash
    git clone https://github.com/OpenHeartDevelopers/pycemrg.git
    cd pycemrg

    conda create -n pycemrg python=3.11 -y
    conda activate pycemrg
    pip install -e .
    ```

=== "venv"

    ```bash
    git clone https://github.com/OpenHeartDevelopers/pycemrg.git
    cd pycemrg

    python -m venv ~/.venvs/pycemrg
    source ~/.venvs/pycemrg/bin/activate
    pip install -e .
    ```

## 0.2 Create your project folder (optional)

```bash
conda activate pycemrg              # or activate your venv
pycemrg init tutorial-custom-heart --with-src --force
```

Your project is now in `~/code/tutorial-custom-heart`.

## 1. Open the reference guide

Open [Authoring a Recipe](../guides/authoring_a_recipe.md) and keep it on the
side. It has the mental model and the steps to follow to create your own heart
based on **your rules**.

From here on, assume your project folder is `~/code/tutorial-custom-heart`.

## 2. Create your environment

You can use `venv` or `conda`. Create a dedicated environment for the project:

=== "conda"

    ```bash
    conda create -n tutorial-custom-heart python=3.11 -y
    conda activate tutorial-custom-heart
    ```

=== "venv"

    ```bash
    python -m venv ~/code/tutorial-custom-heart/.venv
    source ~/code/tutorial-custom-heart/.venv/bin/activate
    ```

### 2.1 Install the `pycemrg` libraries

These are not on PyPI yet, so install them editable from clones:

```bash
cd ~/code
git clone https://github.com/OpenHeartDevelopers/pycemrg.git
git clone https://github.com/OpenHeartDevelopers/pycemrg-image-analysis.git

pip install -e ~/code/pycemrg
pip install -e ~/code/pycemrg-image-analysis
```

### 2.2 Install your own project

```bash
cd ~/code/tutorial-custom-heart
pip install -e .
```

## 3. Plan and execute

Normally you start from a segmentation with **N** labels and end up with one
with **M** labels. For example, a normal 4-chamber heart starts with ~10 labels
and ends up with 30+.

### 3.1 Your `config` folder

This holds your `labels.yaml`. Create one from a template with `pycemrg
init-labels`, which ships with your install:

```bash
pycemrg init-labels -o config/labels.yaml --num-labels 34 --num-groups 5
```

- `--num-labels`: the total number of labels you'll end up with.
- `--num-groups`: optional but helpful. For example, to collect all the
  myocardia, create a `myocardium` group and list those labels in it.

### 3.2 Edit your `labels.yaml`

!!! warning "Edit this manually, label values fail silently"
    Look at your data in ITK-SNAP or 3D Slicer and choose the labels you want.
    If your file's integer label values don't match what your schematics expect,
    structures grow from the **wrong voxels with no error**. Get this right before
    running anything.

The generated file looks like:

```yaml
labels:
  background: 0
  # --- Auto-generated placeholder labels ---
  structure_1: 1
  structure_2: 2
  structure_3: 3
  # ...

groups:
  # --- Auto-generated placeholder groups ---
  group_a:
    - structure_1
    - structure_2
  # ...
```

Modify it to match your data and the labels you want for the generated
structures. For example:

```yaml
labels:
  LV_bloodpool: 1
  LV_myo: 2
  RV_bloodpool: 3
  LA_bloodpool: 4
  # ...

groups:
  Bloodpool:
    - LV_bloodpool
    - RV_bloodpool
    - LA_bloodpool
  # ...
```

### 3.3 Create your own schematic

A schematic is a plain Python dict with three keys: `labels`, `parameters`, and
`semantic_map`. Schematics correspond to **myocardia**, **planes/valves**, and
**rings**. They define which labels are involved, which role each label plays,
and which parameters apply (wall thickness, ring thickness, …). Those relations
live in the `semantic_map`.

!!! abstract "Two files, two jobs, this is the core of the tutorial"
    Keep these separate in your head:

    - **A new *structure* is a new *schematic***, it lives in `myschematics.py`.
      A schematic answers *"what is this structure?"*: its labels, its
      wall-thickness parameter, and the roles in its `semantic_map` (which blood
      pool to grow from, which label to write). The worked example below is
      `rv_outflow`.
    - **The *rules and the order of steps* are a *recipe***, it lives in
      `myrecipes.py`. A recipe answers *"what runs, and in what sequence?"*: an
      ordered list of `WorkflowStep`s plus the `required_schematics` to scaffold.

    So the workflow is: **inspect first** (3.3.1). If a structure already exists,
    just reference it in a recipe. If it does **not** exist, author a schematic
    for it (3.3.2), then place it in a recipe (3.4).

#### 3.3.1 You don't need to define everything

Many components already exist. If you have the `pycemrg-image-analysis` source,
browse `src/pycemrg_image_analysis/schematics/`. You can also list what's
available with the `pycemrg-ima` CLI:

```console
$ pycemrg-ima inspect

Available schematic categories:
========================================

myocardium (7):
  • aortic_wall
  • la_myocardium
  • lv_outflow
  • myo_push_steps
  • pulmonary_artery
  • ra_myocardium
  • rv_myocardium

valves (4):
  • aortic_valve
  • mitral_valve
  • pulmonary_valve
  • tricuspid_valve

rings (7):
  • ivc_ring
  • laa_ring
  • lpv1_ring
  • lpv2_ring
  • rpv1_ring
  • rpv2_ring
  • svc_ring
```

#### 3.3.2 Author a new structure (a new schematic)

Scan the `myocardium` list from `pycemrg-ima inspect` above. Suppose you need an
**`rv_outflow`**, the right-ventricle analogue of the built-in `lv_outflow`. It
is *not* in the list, so it is a genuinely new structure, and a new structure is
a new schematic.

Create `src/tutorial_custom_heart/myschematics.py`:

```python
# src/tutorial_custom_heart/myschematics.py
from pycemrg_image_analysis.logic.constants import MyocardiumSemanticRole

MY_SCHEMATICS = {
    "rv_outflow": {
        # Every label this structure reads or writes, with a default value.
        "labels": {"RV_BP_label": 3, "RV_myo_label": 103},
        # Numeric inputs, here the wall thickness in mm.
        "parameters": {"RV_neck_WT": 3.5},
        # The roles: grow RV_myo_label outward from RV_BP_label by RV_neck_WT.
        "semantic_map": {
            MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME: "RV_BP_label",
            MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME: "RV_myo_label",
            MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME: "RV_neck_WT",
            MyocardiumSemanticRole.APPLICATION_STEPS: [
                {"MODE": "ADD", "RULE_LABEL_NAMES": []},
            ],
        },
    },
}
```

What each key does:

- **`labels`**: the names this structure touches and their default integer
  values. Match these to your `labels.yaml` (3.2): `RV_BP_label` is the input
  cavity you grow from; `RV_myo_label` is the wall you are creating.
- **`parameters`**: `RV_neck_WT` is how far (mm) the wall grows.
- **`semantic_map`**: the *roles* the engine needs: which label is the source
  blood pool, which is the target wall, which parameter is the thickness, and the
  ordered `APPLICATION_STEPS` that write the result. `ADD` writes the new wall
  wherever the grown mask lands. Other modes ()`REPLACE_ONLY`, `REPLACE_EXCEPT`)
  let a structure avoid overwriting its neighbours; see the
  [Authoring guide](../guides/authoring_a_recipe.md#application-steps-and-modes).

!!! tip "Cross-check label names before you scaffold"
    Run `pycemrg-ima create myocardium --labels config/labels.yaml`. It prints how
    the standard myocardium label names line up with *your* `labels.yaml` and
    flags clashes, then a value already in use, or a near-duplicate name (e.g. your
    `RV_bloodpool` vs the schematic's `RV_BP_label`). Reconcile the naming here,
    before anything is generated, so structures don't grow from the wrong voxels.

#### 3.3.3 Register your schematic (no library edits)

You do **not** edit the installed library. Inject your dict at scaffold time with
`extra_schematics=`; it is merged alongside the built-ins (yours win on a name
clash), so every standard component stays available next to `rv_outflow`:

```python
from pycemrg_image_analysis import ImageAnalysisScaffolder
from tutorial_custom_heart.myschematics import MY_SCHEMATICS

scaffolder = ImageAnalysisScaffolder(extra_schematics=MY_SCHEMATICS)
```

The reference runner takes the same `extra_schematics` argument and forwards it,
so you usually pass it there rather than constructing the scaffolder yourself
(see 3.5).

### 3.4 Define the rules and order (a recipe)

A schematic says *what* a structure is. A **recipe** says *which* steps run and
*in what order*. Create `src/tutorial_custom_heart/myrecipes.py`:

```python
# src/tutorial_custom_heart/myrecipes.py
from pycemrg_image_analysis.recipes import Recipe, WorkflowStep

MY_RECIPE = Recipe(
    name="lv_rv_outflow",
    description="LV outflow plus a custom RV outflow.",
    steps=[
        WorkflowStep("create", "lv_outflow"),   # built-in, reused as-is
        WorkflowStep("create", "rv_outflow"),   # your new schematic
    ],
    required_schematics=["lv_outflow", "rv_outflow"],
)
```

- **`steps`** is the execution order. Each `WorkflowStep(step_type,
  component_name)` has a `step_type` of `create` / `valve` / `ring` / `push`.
- **`required_schematics`** is everything the scaffolder must emit config for. It
  is a *superset* of the step component names, it also lists parameter-only
  schematics such as `myo_push_steps` that carry labels/thicknesses without a
  step of their own.

!!! warning "Order is load-bearing"
    Every step writes into the same output image, so a later step can silently
    overwrite anatomy an earlier one placed. Grow a blood pool's myocardium
    *before* any valve, ring, or push that references it.

!!! note "Need a *push* step?"
    A push (e.g. a structure that intrudes on a neighbour and must be pushed back)
    is the one step type with **no** schematic of its own, it reuses existing
    labels and is wired in the orchestrator's push registry, not in
    `myschematics.py`. Add it to your recipe as `WorkflowStep("push", "...")` and
    define its four fields per
    [Authoring guide, Step 4](../guides/authoring_a_recipe.md#step-4-handle-push-steps-only-if-your-recipe-has-them).

### 3.5 Run it

Drive the recipe with the reference runner (`run_recipe_workflow` in
`examples/orchestrator_patterns.py`, it is example code you can import or copy
into your project), passing your `Recipe` object and your schematics:

```python
from pathlib import Path
from examples.orchestrator_patterns import run_recipe_workflow
from tutorial_custom_heart.myschematics import MY_SCHEMATICS
from tutorial_custom_heart.myrecipes import MY_RECIPE

run_recipe_workflow(
    MY_RECIPE,
    input_seg_path=Path("seg_input.nrrd"),
    output_dir=Path("output/"),
    extra_schematics=MY_SCHEMATICS,
    # label_mapping={"RV_BP_label": 3, ...}  # only if your image's ints differ
)
```

The runner scaffolds the config (including your `rv_outflow`), then executes each
step in order, saving an intermediate image per step so you can see exactly where
each structure landed. The final result is written as
`output/seg_final_lv_rv_outflow.nrrd`.

## See Also

- **[Authoring a Recipe](../guides/authoring_a_recipe.md)**: the mental model and full reference this tutorial is based on.
- **[Architecture](../api/overview.md)**: the stateless-toolbox philosophy and orchestration pattern.
- **[Label Tools](../api/label_tools.md)** / **[CLI](../cli/labels.md)**: diagnose and fix mismatched label values.

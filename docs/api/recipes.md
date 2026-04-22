# API Reference: Recipes Module

## Overview

The `recipes.py` module provides pre-defined workflow recipes that specify:
- Which anatomical structures to create (components/schematics)
- The correct execution order (dependencies resolved)
- Step types (myocardium creation, valve creation, ring creation, push operations)

Recipes eliminate the need to manually track component dependencies and execution order.

---

## Core Classes

### `WorkflowStep`

Represents a single step in a workflow.

```python
@dataclass
class WorkflowStep:
    step_type: str       # "create", "valve", "ring", "push"
    component_name: str  # Schematic name (e.g., "lv_outflow", "mitral_valve")
```

**Example:**
```python
step = WorkflowStep("valve", "mitral_valve")
```

---

### `Recipe`

Defines a complete workflow with all required components and execution order.

```python
@dataclass
class Recipe:
    name: str                        # Recipe identifier
    description: str                 # Human-readable description
    steps: List[WorkflowStep]        # Ordered execution steps
    required_schematics: List[str]   # All schematics needed for scaffolding
```

**Attributes:**

- **`name`**: Unique identifier for the recipe (e.g., `"biventricular_basic"`)
- **`description`**: What the recipe creates
- **`steps`**: Ordered list of `WorkflowStep` objects defining execution sequence
- **`required_schematics`**: All schematic names needed (used for scaffolding)

**Example:**
```python
recipe = Recipe(
    name="biventricular_basic",
    description="LV + RV myocardium with mitral and tricuspid valves",
    steps=[
        WorkflowStep("create", "lv_outflow"),
        WorkflowStep("create", "rv_myocardium"),
        WorkflowStep("valve", "mitral_valve"),
        WorkflowStep("valve", "tricuspid_valve"),
    ],
    required_schematics=["lv_outflow", "rv_myocardium", "mitral_valve", "tricuspid_valve"]
)
```

---

## Available Recipes

### `biventricular_basic`

Creates LV and RV myocardium with mitral and tricuspid valves.

**Components:** 4 (lv_outflow, rv_myocardium, mitral_valve, tricuspid_valve)  
**Steps:** 4  
**Use case:** Basic biventricular model for electrophysiology simulations

---

### `four_chamber_myocardium`

Complete four-chamber myocardium including aortic and pulmonary artery walls, with push operations.

**Components:** 7 myocardium schematics  
**Steps:** 10 (includes push operations)  
**Use case:** Full cardiac anatomy without valves

---

### `four_chamber_full`

Complete four-chamber heart with all myocardium and all four valves.

**Components:** 11 (7 myocardium + 4 valves)  
**Steps:** 14  
**Use case:** Complete cardiac model for comprehensive simulations

---

### `left_atrium_with_veins`

LA myocardium with pulmonary vein rings (LPV1, LPV2, RPV1, RPV2, LAA).

**Components:** 6 (1 myocardium + 5 rings)  
**Steps:** 6  
**Use case:** Atrial fibrillation studies focusing on pulmonary vein isolation

---

### `right_atrium_with_veins`

RA myocardium with vena cava rings (SVC, IVC).

**Components:** 3 (1 myocardium + 2 rings)  
**Steps:** 3  
**Use case:** Right atrial anatomy with venous return structures

---

### `atria_full`

Both atria with all vein rings (pulmonary veins, vena cavae, LAA).

**Components:** 9 (2 myocardium + 7 rings)  
**Steps:** 9  
**Use case:** Complete atrial anatomy for bi-atrial studies

---

## Functions

### `get_recipe(name: str) -> Recipe`

Retrieve a recipe by name from the catalog.

**Parameters:**
- `name`: Recipe identifier (e.g., `"biventricular_basic"`)

**Returns:**
- `Recipe` object

**Raises:**
- `KeyError`: If recipe name not found

**Example:**
```python
from pycemrg_image_analysis.recipes import get_recipe

recipe = get_recipe("biventricular_basic")
print(f"{recipe.name}: {recipe.description}")
print(f"Steps: {len(recipe.steps)}")
```

---

### `list_recipes() -> None`

Print all available recipes with descriptions.

**Example:**
```python
from pycemrg_image_analysis.recipes import list_recipes

list_recipes()
```

**Output:**
```
Available Recipes:
================================================================================

biventricular_basic
  LV + RV myocardium with mitral and tricuspid valves
  Steps: 4
  Components: lv_outflow, rv_myocardium, mitral_valve, tricuspid_valve

four_chamber_full
  Complete four-chamber heart with all valves
  Steps: 14
  Components: lv_outflow, aortic_wall, pulmonary_artery, ... (11 total)
  
...
```

---

## Usage Patterns

### Pattern 1: Basic Recipe Execution

Use a recipe to scaffold configs and execute workflow.

```python
from pathlib import Path
from pycemrg_image_analysis.recipes import get_recipe
from pycemrg_image_analysis import ImageAnalysisScaffolder

# 1. Get recipe
recipe = get_recipe("biventricular_basic")

# 2. Scaffold all required configs at once
config_dir = Path("output/config")
scaffolder = ImageAnalysisScaffolder()
scaffolder.scaffold_components(config_dir, recipe.required_schematics)

# 3. Execute steps (see orchestrator_patterns.py for full implementation)
for step in recipe.steps:
    if step.step_type == "create":
        # Execute myocardium creation
        ...
    elif step.step_type == "valve":
        # Execute valve creation
        ...
```

---

### Pattern 2: Recipe Inspection

Examine a recipe before execution.

```python
from pycemrg_image_analysis.recipes import get_recipe

recipe = get_recipe("four_chamber_full")

print(f"Recipe: {recipe.name}")
print(f"Description: {recipe.description}")
print(f"\nExecution order:")
for i, step in enumerate(recipe.steps, 1):
    print(f"  {i}. {step.step_type:8} → {step.component_name}")

print(f"\nRequired schematics ({len(recipe.required_schematics)}):")
print(", ".join(recipe.required_schematics))
```

**Output:**
```
Recipe: four_chamber_full
Description: Complete four-chamber heart with all valves

Execution order:
  1. create   → lv_outflow
  2. create   → aortic_wall
  3. create   → pulmonary_artery
  4. push     → part_push_aorta
  ...
  14. valve    → pulmonary_valve

Required schematics (11):
lv_outflow, aortic_wall, pulmonary_artery, rv_myocardium, ...
```

---

### Pattern 3: Custom Recipe Composition

Use a recipe as a starting point, then customize.

```python
from pycemrg_image_analysis.recipes import get_recipe
from pycemrg_image_analysis.recipes import WorkflowStep

# Start with existing recipe
base_recipe = get_recipe("biventricular_basic")

# Add extra steps
custom_steps = list(base_recipe.steps) + [
    WorkflowStep("valve", "aortic_valve"),
    WorkflowStep("valve", "pulmonary_valve"),
]

# Add extra schematics
custom_schematics = list(base_recipe.required_schematics) + [
    "aortic_valve",
    "pulmonary_valve",
]

# Scaffold and execute custom workflow
scaffolder.scaffold_components(config_dir, custom_schematics)

for step in custom_steps:
    # Execute...
```

---

## Integration with Path Builders

Recipes work seamlessly with path builders for contract creation.

```python
from pathlib import Path
from pycemrg.data.labels import LabelManager
from pycemrg_image_analysis.recipes import get_recipe
from pycemrg_image_analysis.logic import MyocardiumPathBuilder
from pycemrg_image_analysis.utilities import load_image

# 1. Get recipe and scaffold
recipe = get_recipe("biventricular_basic")
scaffolder.scaffold_components(config_dir, recipe.required_schematics)

# 2. Load configuration
label_manager = LabelManager(config_path=config_dir / "labels.yaml")
with open(config_dir / "parameters.json") as f:
    parameters = json.load(f)

# 3. Initialize builder
current_seg = load_image("input.nrrd")
builder = MyocardiumPathBuilder(
    output_dir=output_dir,
    label_manager=label_manager,
    parameters=parameters,
    input_image=current_seg,
)

# 4. Execute recipe steps using builder
for step in recipe.steps:
    if step.step_type == "create":
        # Use builder to create contract
        contract = builder.build_creation_contract(f"seg_{step.component_name}")
        
        # Load rule from semantic map
        rule = load_rule_from_semantic_map(step.component_name, config_dir)
        contract = dataclasses.replace(contract, rule=rule)
        
        # Execute
        result = logic.create_from_rule(contract)
        save_image(result, contract.output_path)
        
        # Update builder's input for next step
        builder._input_image = result
        
    elif step.step_type == "valve":
        # Use builder to create valve contract
        contract = builder.build_valve_contract(f"seg_{step.component_name}")
        
        # Load rule and execute...
```

---

## Workflow: Recipe → Scaffold → Build → Execute

**Complete workflow pattern:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. SELECT RECIPE                                                    │
│    recipe = get_recipe("biventricular_basic")                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. SCAFFOLD CONFIGS                                                 │
│    scaffolder.scaffold_components(                                  │
│        config_dir,                                                  │
│        recipe.required_schematics  # All needed at once             │
│    )                                                                │
│                                                                     │
│    Creates:                                                         │
│    - config/labels.yaml                                             │
│    - config/parameters.json                                         │
│    - config/semantic_maps/*.json                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. INITIALIZE TOOLS                                                 │
│    label_manager = LabelManager(config/labels.yaml)                 │
│    parameters = load(config/parameters.json)                        │
│    builder = MyocardiumPathBuilder(...)                             │
│    logic = MyocardiumLogic()                                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. EXECUTE STEPS IN ORDER                                           │
│    for step in recipe.steps:                                        │
│        contract = builder.build_*_contract(step.component_name)     │
│        rule = load_from_semantic_map(step.component_name)           │
│        result = logic.create_from_rule(contract)                    │
│        save(result)                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Advanced: Creating Custom Recipes

Users can define their own recipes for project-specific workflows.

```python
from pycemrg_image_analysis.recipes import Recipe, WorkflowStep

# Define custom recipe
MY_CUSTOM_RECIPE = Recipe(
    name="custom_lv_focus",
    description="LV myocardium with aortic and mitral structures",
    steps=[
        WorkflowStep("create", "lv_outflow"),
        WorkflowStep("create", "aortic_wall"),
        WorkflowStep("valve", "aortic_valve"),
        WorkflowStep("valve", "mitral_valve"),
    ],
    required_schematics=[
        "lv_outflow",
        "aortic_wall",
        "aortic_valve",
        "mitral_valve",
    ]
)

# Use like any built-in recipe
scaffolder.scaffold_components(config_dir, MY_CUSTOM_RECIPE.required_schematics)

for step in MY_CUSTOM_RECIPE.steps:
    # Execute...
```

---

## Full Working Example

See `examples/orchestrator_patterns.py` for complete, runnable implementations:

- **`run_recipe_workflow()`** - Execute any recipe by name
- **`biventricular_with_specific_valves()`** - Custom composition
- **`left_atrium_with_efficient_rings()`** - Advanced pattern with optimization

**Minimal usage:**

```python
from pycemrg_image_analysis.recipes import get_recipe
from examples.orchestrator_patterns import run_recipe_workflow

result = run_recipe_workflow(
    recipe_name="biventricular_basic",
    input_seg_path=Path("input.nrrd"),
    output_dir=Path("output/")
)
```

This single function call handles:
- Recipe retrieval
- Config scaffolding
- Tool initialization
- Step execution
- Output saving

---

## Summary

**Recipes provide:**
- ✅ Predefined workflows for common tasks
- ✅ Correct execution order (dependencies handled)
- ✅ Single point of config scaffolding (`recipe.required_schematics`)
- ✅ Clear step types for dispatcher logic
- ✅ Extensible (users can define custom recipes)

**Use recipes when:**
- Building standard cardiac models
- Learning the library (recipes document best practices)
- Prototyping new workflows (start with recipe, customize as needed)

**Skip recipes when:**
- Doing exploratory single-step operations
- Building highly custom workflows from scratch
- Only need one or two components

For most orchestration tasks, starting with a recipe simplifies development significantly.
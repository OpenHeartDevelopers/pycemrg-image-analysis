# API Reference: Label Diagnostic and Remapping Tools

## Overview

The label tools help users whose segmentations have different label values than library schematics expect. This commonly occurs when:
- **Slicer resets labels to sequential 1-N** after manual corrections
- Different clinical sites use different labeling conventions
- Converting between segmentation tools with different standards

The tools provide:
1. **Diagnostic checking** - Compare your image labels to schematic expectations
2. **Auto-suggestion** - Detect sequential patterns and suggest mappings
3. **Custom scaffolding** - Generate configs with your actual label values

---

## Quick Start

```python
from pycemrg_image_analysis.utilities.label_tools import check_labels

# One-line diagnostic
report = check_labels(Path("my_seg.nrrd"), "lpv1_ring")

# If issues found, create custom mapping
from pycemrg_image_analysis import ImageAnalysisScaffolder

my_mapping = {
    "LPV1_label": 1,        # Your actual value
    "LA_myo_label": 5,      # Your actual value
    "LPV1_ring_label": 50,  # Target output value
}

scaffolder = ImageAnalysisScaffolder()
scaffolder.scaffold_components_with_mapping(
    output_dir=Path("config/"),
    component_names=["lpv1_ring"],
    label_mapping=my_mapping,
)
```

---

## Classes

### `LabelDiagnostic`

Check if image labels match schematic expectations.

#### `check_image_against_schematic(image_path, schematic_name) -> DiagnosticReport`

**Parameters:**
- `image_path`: Path to segmentation image
- `schematic_name`: Schematic to check (e.g., `"lpv1_ring"`, `"biventricular_basic"`)

**Returns:** `DiagnosticReport` with detailed mismatch information

**Example:**
```python
from pycemrg_image_analysis.utilities.label_tools import LabelDiagnostic

diagnostic = LabelDiagnostic()
report = diagnostic.check_image_against_schematic(
    Path("seg_from_slicer.nrrd"),
    "lpv1_ring"
)

if report.has_issues:
    print(f"Missing: {len(report.missing_labels)} labels")
    for m in report.missing_labels:
        print(f"  {m.label_name}: expected {m.expected_value}")
```

#### `print_report(report: DiagnosticReport) -> None`

Print human-readable diagnostic report.

**Example:**
```python
diagnostic.print_report(report)
```

**Output:**
```
======================================================================
Label Diagnostic Report
======================================================================
Image: seg_from_slicer.nrrd
Schematic: lpv1_ring

Image contains 2 unique labels (excluding 0):
  [1, 5]

Schematic expects 3 labels:

❌ ISSUES FOUND:

Missing labels (2):
  • LPV1_label: expected 8, NOT FOUND in image
  • LA_myo_label: expected 104, NOT FOUND in image

💡 SUGGESTION:
   Your image has different label values than the schematic expects.
   Use LabelRemapper to create a custom mapping...
```

---

### `DiagnosticReport`

Results of diagnostic check.

**Attributes:**
- `image_path`: Path to checked image
- `schematic_name`: Schematic checked against
- `image_labels`: Set of labels found in image
- `expected_labels`: Dict of expected {label_name: value}
- `mismatches`: List of LabelMismatch objects

**Properties:**
- `has_issues`: True if any labels missing or conflicted
- `missing_labels`: Labels expected but not found
- `ok_labels`: Labels that match expectations

**Example:**
```python
if report.has_issues:
    for mismatch in report.missing_labels:
        print(f"{mismatch.label_name} missing")
```

---

### `LabelRemapper`

Create custom label mappings for non-standard label schemes.

#### `create_mapping_from_dict(label_mapping) -> Dict[str, int]`

Validate and return a label mapping.

**Parameters:**
- `label_mapping`: Dict mapping label names to integers

**Returns:** Validated mapping dictionary

**Raises:** `ValueError` if values are not integers or negative

**Example:**
```python
from pycemrg_image_analysis.utilities.label_tools import LabelRemapper

remapper = LabelRemapper()
mapping = remapper.create_mapping_from_dict({
    "LPV1_label": 1,
    "LA_myo_label": 5,
    "LPV1_ring_label": 50,
})
```

#### `suggest_mapping_from_report(report) -> Optional[Dict[str, int]]`

Auto-suggest mapping for sequential labels (Slicer reset pattern).

**Parameters:**
- `report`: DiagnosticReport from check

**Returns:** Suggested mapping dict, or None if no clear pattern

**Example:**
```python
suggestion = remapper.suggest_mapping_from_report(report)

if suggestion:
    print("Auto-detected mapping:")
    for name, value in suggestion.items():
        print(f"  {name}: {value}")
else:
    print("Cannot auto-detect - manual mapping needed")
```

**Auto-detection criteria:**
- Image labels must be sequential (1, 2, 3, ...)
- Count must match expected label count
- If both conditions met, maps sequentially to schematic label names

---

## Scaffolder Extension

### `ImageAnalysisScaffolder.scaffold_components_with_mapping()`

Generate configs using custom label values instead of template defaults.

**Parameters:**
- `output_dir`: Where to save configs
- `component_names`: List of schematics (e.g., `["lpv1_ring"]`)
- `label_mapping`: Dict of {label_name: your_integer}
- `overwrite`: Overwrite existing files (default False)

**Example:**
```python
from pycemrg_image_analysis import ImageAnalysisScaffolder

# After Slicer reset labels to 1, 2, 3...
my_labels = {
    "LPV1_label": 1,
    "LA_myo_label": 2,
    "LPV1_ring_label": 50,
}

scaffolder = ImageAnalysisScaffolder()
scaffolder.scaffold_components_with_mapping(
    output_dir=Path("config/"),
    component_names=["lpv1_ring"],
    label_mapping=my_labels,
)
```

**Result:**
- `config/labels.yaml` contains 1, 2, 50 (not template values 8, 104, 208)
- `config/semantic_maps/lpv1_ring.json` unchanged (uses label names, not values)
- `config/parameters.json` unchanged

**Fallback behavior:** If a label name is NOT in `label_mapping`, uses schematic default value.

---

## Convenience Functions

### `check_labels(image_path, schematic_name) -> DiagnosticReport`

One-line diagnostic that prints report and returns result.

```python
from pycemrg_image_analysis.utilities.label_tools import check_labels

report = check_labels(Path("seg.nrrd"), "lpv1_ring")
# Automatically prints formatted report

if report.has_issues:
    # Handle mismatches...
```

### `list_available_schematics() -> None`

Print all schematic names organized by category.

```python
from pycemrg_image_analysis.utilities.label_tools import list_available_schematics

list_available_schematics()
```

**Output:**
```
Available Schematics:
==================================================

Myocardium:
  • aortic_wall
  • la_myocardium
  • lv_outflow
  ...

Valves:
  • aortic_valve
  • mitral_valve
  ...

Rings:
  • lpv1_ring
  • lpv2_ring
  ...
```

---

## Workflows

### Workflow 1: Quick Check Before Using Standard Scaffolding

```python
from pycemrg_image_analysis.utilities.label_tools import check_labels
from pycemrg_image_analysis import ImageAnalysisScaffolder

# Check compatibility
report = check_labels(Path("seg.nrrd"), "biventricular_basic")

if not report.has_issues:
    # Labels match - use standard scaffolding
    scaffolder = ImageAnalysisScaffolder()
    scaffolder.scaffold_components(
        output_dir=Path("config/"),
        component_names=["biventricular_basic"]
    )
else:
    print("Labels don't match - need custom mapping")
```

### Workflow 2: Auto-Detect and Apply Sequential Mapping

```python
from pycemrg_image_analysis.utilities.label_tools import (
    LabelDiagnostic, LabelRemapper
)
from pycemrg_image_analysis import ImageAnalysisScaffolder

# Diagnose
diagnostic = LabelDiagnostic()
report = diagnostic.check_image_against_schematic(
    Path("seg_from_slicer.nrrd"),
    "lpv1_ring"
)

# Try auto-suggestion
remapper = LabelRemapper()
suggestion = remapper.suggest_mapping_from_report(report)

if suggestion:
    # Use auto-detected mapping
    scaffolder = ImageAnalysisScaffolder()
    scaffolder.scaffold_components_with_mapping(
        output_dir=Path("config/"),
        component_names=["lpv1_ring"],
        label_mapping=suggestion,
    )
    print("✅ Config generated with auto-detected mapping")
else:
    print("Manual mapping required")
```

### Workflow 3: Manual Mapping for Non-Sequential Labels

```python
from pycemrg_image_analysis.utilities.label_tools import check_labels
from pycemrg_image_analysis import ImageAnalysisScaffolder

# Check what we have
report = check_labels(Path("seg.nrrd"), "lpv1_ring")

# Create manual mapping based on report
# (User looks at report and determines which image label = which role)
manual_mapping = {
    "LPV1_label": 5,        # Image has 5 for LPV1
    "LA_myo_label": 10,     # Image has 10 for LA myocardium
    "LPV1_ring_label": 100, # Choose 100 for ring output
}

# Generate configs
scaffolder = ImageAnalysisScaffolder()
scaffolder.scaffold_components_with_mapping(
    output_dir=Path("config/"),
    component_names=["lpv1_ring"],
    label_mapping=manual_mapping,
)
```

---

## Common Scenarios

### Scenario 1: Slicer Reset to 1-N

**Problem:** After manual corrections in Slicer, labels reset to 1, 2, 3, 4...

**Solution:**
```python
# Auto-suggestion should work
diagnostic = LabelDiagnostic()
report = diagnostic.check_image_against_schematic(image, schematic)

remapper = LabelRemapper()
mapping = remapper.suggest_mapping_from_report(report)
# mapping = {"LPV1_label": 1, "LA_myo_label": 2, ...}

scaffolder.scaffold_components_with_mapping(..., label_mapping=mapping)
```

### Scenario 2: Site-Specific Labeling Convention

**Problem:** Clinical site uses custom label scheme (e.g., 100-199 for chambers, 200-299 for vessels)

**Solution:**
```python
# Manual mapping
site_mapping = {
    "LV_BP_label": 101,
    "RV_BP_label": 102,
    "LA_BP_label": 103,
    "RA_BP_label": 104,
    # ...
}

scaffolder.scaffold_components_with_mapping(..., label_mapping=site_mapping)
```

### Scenario 3: Partial Mismatch

**Problem:** Some labels match, some don't

**Solution:**
```python
# Only specify mismatched labels in mapping
# Unspecified labels use schematic defaults
partial_mapping = {
    "LPV1_label": 99,  # Only this one is different
    # Rest use template values
}

scaffolder.scaffold_components_with_mapping(..., label_mapping=partial_mapping)
```

---

## Error Handling

```python
from pycemrg_image_analysis.utilities.label_tools import LabelDiagnostic

diagnostic = LabelDiagnostic()

try:
    report = diagnostic.check_image_against_schematic(
        Path("missing.nrrd"),
        "invalid_schematic"
    )
except FileNotFoundError:
    print("Image file not found")
except KeyError as e:
    print(f"Invalid schematic: {e}")
```

---

## See Also

- **Examples:** `examples/label_remapping_workflow.py`
- **Tests:** `tests/unit/test_label_tools.py`
- **Schematics:** `src/pycemrg_image_analysis/schematics/`
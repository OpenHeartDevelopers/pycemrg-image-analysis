# API Reference: Connected Component Cleanup

## Overview

Connected component cleanup functions remove floating/disconnected blobs from segmentations. Two approaches are provided based on different use cases:

1. **Per-Label Cleanup** (`keep_largest_component`) - Clean each label independently
2. **Structure Cleanup** (`keep_largest_structure`) - Treat multiple labels as one connected structure

Common use case: Neural networks (e.g., CardioForm) produce segmentations with valid anatomy plus floating garbage chunks that share labels with the main anatomy. These functions clean up such artifacts before refinement.

---

## Quick Start

```python
from pycemrg_image_analysis.utilities.postprocessing import (
    keep_largest_structure_by_name
)

# Clean entire CardioForm output (all labels)
cleaned = keep_largest_structure_by_name(cardioform_seg)

# Or clean specific multi-label structure
cleaned = keep_largest_structure_by_name(
    seg,
    label_names=["LA_BP_label", "LPV1_label", "RPV1_label"],
    label_manager=label_manager
)
```

---

## When to Use Which Function

### Use `keep_largest_component` when:
- ✅ Labels are separate anatomical structures (don't touch each other)
- ✅ Each label independently has small floating blobs
- ✅ Example: Aorta (7) has floating pieces AND LA (4) has floating pieces, but they're unrelated

### Use `keep_largest_structure` when:
- ✅ Labels form ONE connected anatomical structure
- ✅ Floating chunks share labels with main anatomy
- ✅ You want to remove anything not connected to main structure
- ✅ Example: LA + PVs form connected atrium, but there are lung chunks also labeled LA/PV

---

## Spatial Operations Layer

**Module:** `pycemrg_image_analysis.utilities.components`

These functions operate directly on `sitk.Image` objects and handle spatial connectivity using SimpleITK algorithms. Metadata (spacing, origin, direction) is preserved.

---

### `keep_largest_component()`

Keep only the largest connected component for each label independently.

**Signature:**
```python
def keep_largest_component(
    image: sitk.Image,
    label_values: List[int]
) -> sitk.Image
```

**Parameters:**
- `image`: Input segmentation image
- `label_values`: List of integer label values to process independently

**Returns:** New image with only largest component per label preserved

**Behavior:**
- Each label in `label_values` is processed separately
- For each label: finds all connected components, keeps only largest
- Labels not in `label_values` are completely untouched
- Labels with single component are unchanged (no wasted computation)
- Labels in `label_values` but not in image are silently skipped

**Example:**
```python
from pycemrg_image_analysis.utilities.components import keep_largest_component

# Label 7 has main aorta + small floating blob
# Label 4 has main LA + small floating blob
# Clean both independently
cleaned = keep_largest_component(seg_image, [7, 4])
```

**Algorithm:**
For each label:
1. Extract binary mask for that label
2. Run connected component analysis
3. Relabel components by size (largest = 1)
4. Keep only component 1
5. Restore to original label value

---

### `keep_largest_structure()`

Keep only the largest connected structure across multiple labels.

**Signature:**
```python
def keep_largest_structure(
    image: sitk.Image,
    label_values: Optional[List[int]] = None
) -> sitk.Image
```

**Parameters:**
- `image`: Input segmentation image
- `label_values`: Labels forming the structure. If `None`, uses all non-zero labels.

**Returns:** New image with only largest structure, preserving internal labels

**Behavior:**
- Treats all specified labels as ONE anatomical structure
- Finds largest connected blob considering ALL labels together
- Removes floating chunks even if they have valid label values
- Preserves all internal label values within kept structure
- Default (`label_values=None`) processes entire segmentation

**Example:**
```python
from pycemrg_image_analysis.utilities.components import keep_largest_structure

# CardioForm output has LA (4) + PVs (8,9,10,11) connected,
# plus garbage in lungs also labeled 4,8,9,10,11
cleaned = keep_largest_structure(seg, [4, 8, 9, 10, 11])
# Keeps only main LA+PV structure

# Clean entire segmentation (all labels)
cleaned = keep_largest_structure(cardioform_seg)  # label_values=None
```

**Algorithm:**
1. Create binary mask (any specified label = foreground)
2. Run connected component analysis on binary mask
3. Relabel components by size (largest = 1)
4. Multiply original image by largest component mask
5. Result: main structure with all labels, floating chunks zeroed

---

## Semantic Wrappers Layer

**Module:** `pycemrg_image_analysis.utilities.postprocessing`

These functions translate semantic label names to integer values, then delegate to the spatial operations layer. Designed for orchestrator use.

---

### `keep_largest_component_by_name()`

Keep largest component per label using semantic names.

**Signature:**
```python
def keep_largest_component_by_name(
    image: sitk.Image,
    label_names: List[str],
    label_manager: LabelManager
) -> sitk.Image
```

**Parameters:**
- `image`: Input segmentation image
- `label_names`: List of label names to process (e.g., `["Ao_BP_label", "LA_BP_label"]`)
- `label_manager`: LabelManager for name → value lookup

**Returns:** New image with only largest components per label

**Raises:** `KeyError` if label name not found in LabelManager (logged as warning, label skipped)

**Example:**
```python
from pycemrg_image_analysis.utilities.postprocessing import (
    keep_largest_component_by_name
)

cleaned = keep_largest_component_by_name(
    seg_image,
    ["Ao_BP_label", "LA_BP_label"],
    label_manager
)
```

---

### `keep_largest_structure_by_name()`

Keep largest multi-label structure using semantic names.

**Signature:**
```python
def keep_largest_structure_by_name(
    image: sitk.Image,
    label_names: Optional[List[str]] = None,
    label_manager: Optional[LabelManager] = None
) -> sitk.Image
```

**Parameters:**
- `image`: Input segmentation image
- `label_names`: List of label names forming structure. If `None`, uses all non-zero labels.
- `label_manager`: LabelManager for name → value lookup. Required if `label_names` provided.

**Returns:** New image with only largest structure

**Raises:**
- `ValueError` if `label_names` provided but `label_manager` is `None`
- `KeyError` if label name not found (logged as warning, label skipped)

**Example:**
```python
from pycemrg_image_analysis.utilities.postprocessing import (
    keep_largest_structure_by_name
)

# Clean entire segmentation (all labels)
cleaned = keep_largest_structure_by_name(cardioform_seg)

# Clean specific structure
cleaned = keep_largest_structure_by_name(
    seg_image,
    ["LA_BP_label", "LPV1_label", "LPV2_label", "RPV1_label", "RPV2_label"],
    label_manager
)
```

---

## Orchestrator Usage Patterns

### Pattern 1: Clean Entire CardioForm Output

```python
from pycemrg_image_analysis.utilities.postprocessing import (
    keep_largest_structure_by_name
)

# In AnatomyRefinementRunner.run()
cardioform_seg = load_image(subject.paths.seg_whole_heart_label4_handled)

# Clean all floating components
cleaned_seg = keep_largest_structure_by_name(cardioform_seg)

# Continue with relabeling and refinement
working_image = relabel_image(cleaned_seg, mapping)
```

---

### Pattern 2: Clean Specific Problematic Labels

```python
# Clean only known problematic labels (configurable)
labels_to_clean = ["Ao", "LA_bp", "RA_bp"]  # From config

cleaned_seg = keep_largest_structure_by_name(
    cardioform_seg,
    label_names=labels_to_clean,
    label_manager=cardioform_labels
)
```

---

### Pattern 3: Per-Label Cleanup for Independent Structures

```python
# If aorta and pulmonary artery are separate and both have floating blobs
cleaned_seg = keep_largest_component_by_name(
    seg,
    ["Ao_BP_label", "PArt_BP_label"],
    label_manager
)
```

---

### Pattern 4: Conditional Cleanup Based on Tracker Flags

```python
# Only clean if tracker indicates floating components
if subject.tracker.has_flag("FLOATING_ANATOMY"):
    labels_to_clean = subject.tracker.get_floating_labels()
    cleaned_seg = keep_largest_structure_by_name(
        cardioform_seg,
        label_names=labels_to_clean,
        label_manager=cardioform_labels
    )
else:
    cleaned_seg = cardioform_seg
```

---

## Complete Pipeline Example

```python
from pycemrg.data import LabelManager, LabelMapper
from pycemrg_image_analysis.utilities import load_image, save_image
from pycemrg_image_analysis.utilities.postprocessing import (
    keep_largest_structure_by_name,
    relabel_image
)

def run_anatomy_refinement(subject):
    """Complete pipeline with component cleanup."""
    
    # 1. Load CardioForm output
    cardioform_seg = load_image(subject.paths.seg_whole_heart_label4_handled)
    cardioform_mgr = LabelManager("config/anatomy_labels.yaml")
    
    # 2. Clean floating components
    cleaned_seg = keep_largest_structure_by_name(cardioform_seg)
    
    # 3. Translate to image-analysis standard
    image_analysis_mgr = LabelManager("config/labels.yaml")
    mapper = LabelMapper(source=cardioform_mgr, target=image_analysis_mgr)
    mapping = mapper.get_source_to_target_mapping()
    working_image = relabel_image(cleaned_seg, mapping)
    
    # 4. Continue with refinement
    working_image = run_myocardium(working_image, ...)
    working_image = run_valves(working_image, ...)
    
    # 5. Save result
    save_image(working_image, subject.paths.seg_anatomy_refined)
```

---

## Edge Cases and Behavior

### Missing Labels
```python
# Labels in list but not in image are silently skipped (no error)
cleaned = keep_largest_component(seg, [5, 99])  # 99 doesn't exist
# → Label 5 cleaned, label 99 ignored
```

### Single Component
```python
# Labels with only one component are returned unchanged
# (No wasted computation running connected components)
cleaned = keep_largest_component(seg, [5])  # Label 5 has one blob
# → Image unchanged for label 5
```

### Empty Label List
```python
# Empty list returns image unchanged
cleaned = keep_largest_component(seg, [])
# → Original image returned
```

### All Labels Default
```python
# None = all non-zero labels
cleaned = keep_largest_structure(seg)  # label_values=None
# → Uses all labels in image (excluding 0)
```

### Tie in Component Size
```python
# If two components have same size, keeps first one found
# (Determined by SimpleITK's internal ordering)
```

---

## Performance Notes

- **Per-label** (`keep_largest_component`): O(N × L) where N = image size, L = number of labels
- **Structure** (`keep_largest_structure`): O(N) - single connected component pass
- Uses SimpleITK's optimized ConnectedComponent filter (efficient for 3D medical images)
- Handles anisotropic voxels correctly via metadata preservation
- Face connectivity (not diagonal) - standard for medical segmentations

---

## Common Pitfalls

### ❌ Wrong: Using per-label for multi-label structure
```python
# Don't do this for LA+PV structure with floating chunks:
cleaned = keep_largest_component(seg, [4, 8, 9, 10, 11])
# → Cleans each label independently, won't remove chunks that are
#   the largest blob of their specific label
```

### ✅ Right: Use structure cleanup
```python
# Do this instead:
cleaned = keep_largest_structure(seg, [4, 8, 9, 10, 11])
# → Treats 4+8+9+10+11 as ONE structure, removes floating chunks
```

---

### ❌ Wrong: Forgetting to specify label_manager with names
```python
# This will raise ValueError:
cleaned = keep_largest_structure_by_name(
    seg,
    label_names=["LA_BP_label"],
    label_manager=None  # ← Error!
)
```

### ✅ Right: Provide label_manager
```python
cleaned = keep_largest_structure_by_name(
    seg,
    label_names=["LA_BP_label"],
    label_manager=label_manager
)
```

---

## See Also

- **Examples:** `examples/connected_component_cleanup.py`
- **Tests:** `tests/unit/test_components.py`
- **Related:** `remove_labels()`, `keep_labels()` for label-based cleanup without connectivity
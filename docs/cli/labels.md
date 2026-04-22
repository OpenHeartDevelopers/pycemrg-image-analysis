# Labels CLI Tool

Command-line utility for label diagnostics and remapping.

---

## Installation

No installation needed - runs directly from the repository:

```bash
python scripts/utilities/labels.py COMMAND [OPTIONS]
```

---

## Commands

### 1. `show` - Display Labels in Image

Show all unique labels with voxel counts and percentages.

**Usage:**
```bash
python scripts/utilities/labels.py show --input IMAGE
```

**Example:**
```bash
python scripts/utilities/labels.py show --input seg_from_slicer.nrrd
```

**Output:**
```
Analyzing: seg_from_slicer.nrrd
======================================================================

Image shape (Z, Y, X): (192, 512, 512)
Spacing (X, Y, Z): (0.28125, 0.28125, 0.5)
Origin (X, Y, Z): (0.0, 0.0, 0.0)

Found 4 unique labels:
Label      Voxel Count     Percentage
----------------------------------------
0          48234567           95.23%
1          1234567             2.44%  ●
2          987654              1.95%  ●
5          193456              0.38%  ●

✓ Non-zero labels: [1, 2, 5]
```

**Use this to:**
- Check which labels exist in your segmentation
- See label distribution
- Verify Slicer didn't reset your labels

---

### 2. `check` - Validate Against Schematic

Compare image labels to schematic expectations.

**Usage:**
```bash
python scripts/utilities/labels.py check --input IMAGE --schematic NAME
```

**Example:**
```bash
python scripts/utilities/labels.py check --input seg.nrrd --schematic lpv1_ring
```

**Output (when labels match):**
```
Checking: seg.nrrd
Against schematic: lpv1_ring
======================================================================

Label Diagnostic Report
======================================================================
...

✅ ALL LABELS MATCH
  • LPV1_label: 8 ✓
  • LA_myo_label: 104 ✓
  • LPV1_ring_label: 208 ✓
```

**Output (when labels don't match):**
```
❌ ISSUES FOUND:

Missing labels (2):
  • LPV1_label: expected 8, NOT FOUND in image
  • LA_myo_label: expected 104, NOT FOUND in image

💡 SUGGESTION:
   Your image has different label values than the schematic expects.
   Use LabelRemapper to create a custom mapping...
```

**Exit codes:**
- `0` - All labels match
- `1` - Issues found

**Use this to:**
- Verify compatibility before running workflows
- Identify which labels need remapping
- Automate validation in CI/CD pipelines

---

### 3. `list-schematics` - Show Available Schematics

List all schematics organized by category.

**Usage:**
```bash
python scripts/utilities/labels.py list-schematics
```

**Output:**
```
Available Schematics:
==================================================

Myocardium:
  • aortic_wall
  • la_myocardium
  • lv_outflow
  • pulmonary_artery
  • ra_myocardium
  • rv_myocardium

Valves:
  • aortic_valve
  • mitral_valve
  • pulmonary_valve
  • tricuspid_valve

Rings:
  • ivc_ring
  • laa_ring
  • lpv1_ring
  • lpv2_ring
  • rpv1_ring
  • rpv2_ring
  • svc_ring
```

**Use this to:**
- Find the correct schematic name
- See what's available in the library

---

### 4. `suggest` - Auto-Generate Mapping

Attempt to auto-detect label mapping (for sequential labels after Slicer reset).

**Usage:**
```bash
python scripts/utilities/labels.py suggest --input IMAGE --schematic NAME [--output FILE]
```

**Example:**
```bash
python scripts/utilities/labels.py suggest --input seg_from_slicer.nrrd --schematic lpv1_ring
```

**Output (when auto-detection succeeds):**
```
Analyzing: seg_from_slicer.nrrd
For schematic: lpv1_ring
======================================================================

✅ Auto-detected sequential mapping:

Copy this mapping into your code:
----------------------------------------------------------------------
mapping = {
    "LPV1_label": 1,
    "LA_myo_label": 2,
    "LPV1_ring_label": 3,
}
----------------------------------------------------------------------

Or save to file:
  Use --output FILENAME.json to save mapping
```

**With `--output` flag:**
```bash
python scripts/utilities/labels.py suggest --input seg.nrrd --schematic lpv1_ring --output my_mapping.json
```
Creates `my_mapping.json`:
```json
{
  "LPV1_label": 1,
  "LA_myo_label": 2,
  "LPV1_ring_label": 3
}
```

**Output (when auto-detection fails):**
```
⚠️  Cannot auto-detect mapping

Reasons:
  - Labels are not sequential (1, 2, 3...)
  - Label count doesn't match schematic expectations

Your image has these labels:
  [5, 10, 15]

Schematic expects these labels:
  LPV1_label: 8
  LA_myo_label: 104
  LPV1_ring_label: 208

You'll need to create a manual mapping:
----------------------------------------------------------------------
mapping = {
    "LPV1_label": ???,  # Replace ??? with your label value
    "LA_myo_label": ???,  # Replace ??? with your label value
    "LPV1_ring_label": ???,  # Replace ??? with your label value
}
----------------------------------------------------------------------
```

**Exit codes:**
- `0` - Mapping successfully suggested
- `1` - Cannot auto-detect (manual mapping needed)

**Use this to:**
- Quick fix for Slicer reset scenario
- Generate starter template for manual mapping
- Automate mapping in scripts

---

## Typical Workflows

### Workflow 1: Quick Label Inspection

```bash
# What labels are in my image?
python scripts/utilities/labels.py show --input seg.nrrd
```

---

### Workflow 2: Pre-Flight Check

```bash
# Will this work with lpv1_ring schematic?
python scripts/utilities/labels.py check --input seg.nrrd --schematic lpv1_ring

# Exit code 0 = good to go
# Exit code 1 = need remapping
```

---

### Workflow 3: Slicer Reset Recovery

```bash
# Step 1: Check what happened
python scripts/utilities/labels.py show --input seg_from_slicer.nrrd
# Output: [1, 2, 3] instead of [8, 104, 208]

# Step 2: Auto-generate mapping
python scripts/utilities/labels.py suggest \
    --input seg_from_slicer.nrrd \
    --schematic lpv1_ring \
    --output my_mapping.json

# Step 3: Use mapping in Python
python
>>> import json
>>> with open('my_mapping.json') as f:
...     mapping = json.load(f)
>>> 
>>> from pycemrg_image_analysis import ImageAnalysisScaffolder
>>> scaffolder = ImageAnalysisScaffolder()
>>> scaffolder.scaffold_components_with_mapping(
...     output_dir="config/",
...     component_names=["lpv1_ring"],
...     label_mapping=mapping
... )
```

---

### Workflow 4: Batch Validation

```bash
#!/bin/bash
# validate_all_segs.sh

for seg in data/*.nrrd; do
    echo "Checking $seg..."
    python scripts/utilities/labels.py check \
        --input "$seg" \
        --schematic biventricular_basic
    
    if [ $? -ne 0 ]; then
        echo "  ❌ FAILED: $seg"
    else
        echo "  ✅ OK: $seg"
    fi
done
```

---

## Tips

### Tip 1: Piping Output

```bash
# Save label info to file
python scripts/utilities/labels.py show --input seg.nrrd > labels_report.txt

# Count non-zero labels
python scripts/utilities/labels.py show --input seg.nrrd | grep "●" | wc -l
```

### Tip 2: Scripting with Exit Codes

```bash
# Only proceed if labels match
if python scripts/utilities/labels.py check --input seg.nrrd --schematic lpv1_ring; then
    echo "Labels match - running workflow"
    python run_workflow.py
else
    echo "Labels don't match - fix labels first"
    exit 1
fi
```

### Tip 3: Quick Schematic Lookup

```bash
# Find ring schematics
python scripts/utilities/labels.py list-schematics | grep -i ring
```

---

## Error Messages

### "File not found"
```bash
❌ Error: File not found: seg.nrrd
```
**Fix:** Check file path and extension (.nrrd, .nii, .nii.gz)

### "Invalid schematic"
```bash
❌ Error: Schematic 'invalid_name' not found
```
**Fix:** Run `list-schematics` to see valid names

### "Cannot auto-detect mapping"
```
⚠️  Cannot auto-detect mapping
```
**Fix:** Create manual mapping using the template provided in output

---

## Integration with Python API

The CLI wraps the Python API. For programmatic access, use:

```python
from pycemrg_image_analysis.utilities.label_tools import (
    LabelDiagnostic,
    LabelRemapper,
    check_labels,
)

# Equivalent to: labels.py check
diagnostic = LabelDiagnostic()
report = diagnostic.check_image_against_schematic(path, schematic)

# Equivalent to: labels.py suggest
remapper = LabelRemapper()
mapping = remapper.suggest_mapping_from_report(report)
```

---

## See Also

- **Python API:** `src/pycemrg_image_analysis/utilities/label_tools.py`
- **Examples:** `examples/label_remapping_workflow.py`
- **Tests:** `tests/unit/test_label_tools.py`
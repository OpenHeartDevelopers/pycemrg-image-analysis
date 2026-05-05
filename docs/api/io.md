# API Reference: I/O Utilities

## Overview

The I/O utilities provide functions for loading, saving, and converting medical images. This document covers the INR format conversion functions. General-purpose load/save functions are documented in the [overview](overview.md).

**Module:** `pycemrg_image_analysis.utilities.io`  
**Public import:** `pycemrg_image_analysis.utilities`

---

## INR Format

INR (`.inr`) is a binary volumetric image format used by [Inria](https://team.inria.fr/morpheme/software/). Key properties:

- Fixed-size 256-byte ASCII header (padded to the next 256-byte boundary for large headers)
- Voxel data in Fortran (x-fastest) order immediately after the header
- Carries spacing (`VX`, `VY`, `VZ`) but **no origin** — origin is always treated as `(0, 0, 0)`
- Supports integer and float scalar volumes (`VDIM=1` only)

**Supported dtypes:**

| INR `TYPE` | `PIXSIZE` | NumPy dtype |
|---|---|---|
| `unsigned fixed` | 8 | `uint8` |
| `unsigned fixed` | 16 | `uint16` |
| `unsigned fixed` | 32 | `uint32` |
| `signed fixed` | 8 | `int8` |
| `signed fixed` | 16 | `int16` |
| `signed fixed` | 32 | `int32` |
| `float` | 32 | `float32` |
| `float` | 64 | `float64` |

---

## Quick Start

```python
from pycemrg_image_analysis.utilities import convert_inr_to_image, convert_image_to_inr
from pathlib import Path

# Read an INR file → sitk.Image
image = convert_inr_to_image(Path("mesh/myocardium.inr"))

# Write a sitk.Image → INR file
convert_image_to_inr(image, Path("output/myocardium.inr"))
```

---

## Functions

### `convert_inr_to_image()`

Read an INR file and return a SimpleITK Image.

**Signature:**
```python
def convert_inr_to_image(inr_path: Path) -> sitk.Image
```

**Parameters:**
- `inr_path`: Path to the `.inr` file

**Returns:** `sitk.Image` with spacing from the header and origin fixed at `(0, 0, 0)`

**Raises:**
- `FileNotFoundError` if `inr_path` does not exist
- `ValueError` if the header contains an unsupported `TYPE` / `PIXSIZE` combination

**Notes:**
- Voxel data is read as a flat Fortran-order buffer and reshaped to `(XDIM, YDIM, ZDIM)` before being handed to SimpleITK. The axis transposition from `(x, y, z)` to the `(z, y, x)` layout that SimpleITK expects is handled internally.
- Multi-component volumes (`VDIM > 1`) are not supported.

**Example:**
```python
from pycemrg_image_analysis.utilities import convert_inr_to_image, save_image
from pathlib import Path

image = convert_inr_to_image(Path("segmentations/lv.inr"))
print(image.GetSize())     # (XDIM, YDIM, ZDIM)
print(image.GetSpacing())  # (VX, VY, VZ) from header

# Convert to NIfTI for downstream tools
save_image(image, Path("segmentations/lv.nii.gz"))
```

---

### `convert_image_to_inr()`

Write a SimpleITK Image to an INR file.

**Signature:**
```python
def convert_image_to_inr(image: sitk.Image, inr_path: Path) -> None
```

**Parameters:**
- `image`: The `sitk.Image` to serialize
- `inr_path`: Destination path for the `.inr` file (parent directories are created if missing)

**Returns:** `None`

**Raises:**
- `ValueError` if the image's pixel type has no INR equivalent (see supported dtypes table above)

**Notes:**
- The image origin is **not written** to the header. This is consistent with `convert_inr_to_image`, which always reads origin as `(0, 0, 0)`. A round-trip through INR will lose any non-zero origin.
- Voxel data is extracted in `(z, y, x)` order from SimpleITK, transposed to `(x, y, z)`, then written in Fortran order — matching the INR layout exactly.
- The header is padded with newlines to the next 256-byte boundary.

**Example:**
```python
from pycemrg_image_analysis.utilities import load_image, convert_image_to_inr
from pathlib import Path

# Load a NIfTI segmentation and export for an INR-based meshing tool
seg = load_image(Path("segmentations/lv.nii.gz"))
convert_image_to_inr(seg, Path("mesh_input/lv.inr"))
```

---

## Round-Trip Behaviour

```python
image = convert_inr_to_image(Path("original.inr"))
convert_image_to_inr(image, Path("roundtrip.inr"))
# Voxel data and spacing are preserved exactly.
# Origin is always (0, 0, 0) — not round-tripped.
```

---

## Common Pitfalls

### Wrong: Expecting origin to be preserved
```python
image = load_image(Path("scan.nii.gz"))
print(image.GetOrigin())  # e.g. (-150.0, -80.0, -200.0)

convert_image_to_inr(image, Path("scan.inr"))
recovered = convert_inr_to_image(Path("scan.inr"))
print(recovered.GetOrigin())  # (0.0, 0.0, 0.0) — origin is lost
```

### Right: Re-apply origin after loading if it matters
```python
original_origin = image.GetOrigin()
recovered = convert_inr_to_image(Path("scan.inr"))
recovered.SetOrigin(original_origin)
```

---

### Wrong: Passing a `float64` image when the meshing tool expects `uint8`
```python
# Probability map is float64 — INR supports it, but meshing tools may not
convert_image_to_inr(prob_map, Path("mesh_input/prob.inr"))
```

### Right: Cast before writing
```python
import SimpleITK as sitk

label_map = sitk.Cast(prob_map, sitk.sitkUInt8)
convert_image_to_inr(label_map, Path("mesh_input/labels.inr"))
```

---

## See Also

- **Tests:** `tests/unit/test_io.py`
- **Related:** `load_image()`, `save_image()` for NIfTI/NRRD/MHA formats

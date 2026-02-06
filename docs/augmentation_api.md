# `pycemrg_image_analysis.utilities.augmentation` API Reference

**Module:** Data augmentation for medical image analysis to increase effective dataset size.

**Convention:** 
- Intensity functions expect pre-normalized `[0, 1]` data in `(Z, Y, X)` format
- Spatial functions operate on `sitk.Image` objects with metadata

---

## Overview

This module provides two types of augmentation:

1. **Intensity Augmentation** - Modify voxel intensities (NumPy-based)
2. **Spatial Augmentation** - Modify spatial sampling (SimpleITK-based)

---

## Intensity Augmentation Functions

All intensity functions follow the same pattern:
- Input: `np.ndarray` in `(Z, Y, X)` format, normalized to `[0, 1]`
- Output: `np.ndarray` in `(Z, Y, X)` format, clipped to `[0, 1]`
- Support ROI-aware augmentation via optional `mask` parameter
- Support reproducibility via `seed` parameter

---

### `augment_brightness()`

```python
def augment_brightness(
    volume: np.ndarray,
    factor: float = 0.1,
    mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray
```

Randomly adjust image brightness by adding uniform random shift.

**Parameters:**
- `volume` (np.ndarray): Input volume in `(Z, Y, X)` format, normalized to `[0, 1]`
- `factor` (float, optional): Maximum brightness change as fraction (e.g., `0.1` = ±10%). Default: `0.1`
- `mask` (np.ndarray, optional): Binary mask in `(Z, Y, X)` format. Augmentation applied only where `mask > 0`
- `seed` (int, optional): Random seed for reproducibility

**Returns:**
- `np.ndarray`: Volume with adjusted brightness, clipped to `[0, 1]`

**Raises:**
- `ValueError`: If input not normalized to `[0, 1]` or mask shape mismatches

**Implementation:**
```
brightness_shift = uniform(-factor, +factor)
output = clip(input + brightness_shift, 0, 1)
```

**Example:**
```python
from pycemrg_image_analysis.utilities.intensity import normalize_min_max
from pycemrg_image_analysis.utilities.augmentation import augment_brightness

# Normalize volume
volume = normalize_min_max(raw_volume)  # [0, 1]

# Apply brightness augmentation
augmented = augment_brightness(volume, factor=0.1, seed=42)

# ROI-aware augmentation
myo_mask = (segmentation == myo_label)
augmented = augment_brightness(volume, factor=0.1, mask=myo_mask, seed=42)
```

---

### `augment_contrast()`

```python
def augment_contrast(
    volume: np.ndarray,
    factor: float = 0.1,
    mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray
```

Randomly adjust image contrast by scaling around mean value.

**Parameters:**
- `volume` (np.ndarray): Input volume in `(Z, Y, X)` format, normalized to `[0, 1]`
- `factor` (float, optional): Maximum contrast change as fraction. Default: `0.1`
- `mask` (np.ndarray, optional): Binary mask. If provided, mean is computed only within mask
- `seed` (int, optional): Random seed for reproducibility

**Returns:**
- `np.ndarray`: Volume with adjusted contrast, clipped to `[0, 1]`

**Raises:**
- `ValueError`: If input not normalized to `[0, 1]` or mask shape mismatches

**Implementation:**
```
contrast_factor = uniform(1 - factor, 1 + factor)
mean = compute_mean(input, mask)
output = clip(mean + contrast_factor * (input - mean), 0, 1)
```

**Example:**
```python
augmented = augment_contrast(volume, factor=0.15, seed=42)
# Contrast scaled by factor in [0.85, 1.15]
```

---

### `augment_gamma()`

```python
def augment_gamma(
    volume: np.ndarray,
    gamma_range: Tuple[float, float] = (0.8, 1.2),
    mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray
```

Apply random gamma correction (non-linear intensity transformation).

**Parameters:**
- `volume` (np.ndarray): Input volume in `(Z, Y, X)` format, normalized to `[0, 1]`
- `gamma_range` (Tuple[float, float], optional): `(min_gamma, max_gamma)` range. Values `< 1` brighten, values `> 1` darken. Default: `(0.8, 1.2)`
- `mask` (np.ndarray, optional): Binary mask for ROI-aware augmentation
- `seed` (int, optional): Random seed for reproducibility

**Returns:**
- `np.ndarray`: Gamma-corrected volume

**Raises:**
- `ValueError`: If input not normalized, `gamma_range` is invalid, or mask shape mismatches

**Implementation:**
```
gamma = uniform(gamma_range[0], gamma_range[1])
output = input ** gamma
```

**Example:**
```python
augmented = augment_gamma(volume, gamma_range=(0.7, 1.3), seed=42)
# Non-linear intensity transform
```

---

### `augment_noise()`

```python
def augment_noise(
    volume: np.ndarray,
    noise_std: float = 0.01,
    mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray
```

Add Gaussian noise to simulate scanner thermal noise.

**Parameters:**
- `volume` (np.ndarray): Input volume in `(Z, Y, X)` format, normalized to `[0, 1]`
- `noise_std` (float, optional): Standard deviation of Gaussian noise. Typical: `0.01-0.05`. Default: `0.01`
- `mask` (np.ndarray, optional): Binary mask. Noise added only where `mask > 0`
- `seed` (int, optional): Random seed for reproducibility

**Returns:**
- `np.ndarray`: Volume with added noise, clipped to `[0, 1]`

**Raises:**
- `ValueError`: If input not normalized to `[0, 1]` or mask shape mismatches

**Implementation:**
```
noise = normal(0, noise_std, shape=input.shape)
output = clip(input + noise, 0, 1)
```

**Example:**
```python
augmented = augment_noise(volume, noise_std=0.02, seed=42)
# Gaussian noise with std=0.02 added
```

---

## Spatial Augmentation Functions

### `create_slice_shifted_volumes()`

```python
def create_slice_shifted_volumes(
    img: sitk.Image,
    target_z_spacing: float,
    num_shifts: int = 16,
    preserve_extent: bool = True,
    interpolator: int = sitk.sitkLinear,
) -> List[sitk.Image]
```

Create multiple downsampled volumes by shifting the starting slice. **Standard practice for z-axis super-resolution training.**

**Parameters:**
- `img` (sitk.Image): Input high-resolution SimpleITK image
- `target_z_spacing` (float): Desired coarse z-spacing in mm (e.g., `8.0`)
- `num_shifts` (int, optional): Number of shifted versions to create. Will be clamped to `floor(target_z_spacing / original_z_spacing)`. Default: `16`
- `preserve_extent` (bool, optional): Maintain same physical extent as input (required for FAE coordinate consistency). Default: `True`
- `interpolator` (int, optional): SimpleITK interpolator. Default: `sitk.sitkLinear`

**Returns:**
- `List[sitk.Image]`: List of downsampled images, each starting at a different z-offset

**Raises:**
- `ValueError`: If `target_z_spacing <= original z-spacing` (must be downsampling)

**How It Works:**

Given high-res scan with 0.5mm z-spacing, target 8.0mm z-spacing:
- **Step size:** `8.0 / 0.5 = 16` slices
- **Shift 0:** Keeps slices `[0, 16, 32, 48, ...]`
- **Shift 1:** Keeps slices `[1, 17, 33, 49, ...]`
- **Shift 2:** Keeps slices `[2, 18, 34, 50, ...]`
- ...
- **Shift 15:** Keeps slices `[15, 31, 47, 63, ...]`

Each shifted volume sees **different anatomical z-positions** → legitimate training data diversity.

**Example:**
```python
from pycemrg_image_analysis.utilities.io import load_image, save_image
from pycemrg_image_analysis.utilities.augmentation import create_slice_shifted_volumes

# Load high-res scan: [0.33, 0.33, 0.5]mm
img = load_image("patient_001.nii.gz")

# Create 16 shifted versions at 8mm z-spacing
shifted_volumes = create_slice_shifted_volumes(
    img=img,
    target_z_spacing=8.0,
    num_shifts=16,
    preserve_extent=True  # Required for FAE training
)

# Save all shifted versions
for i, vol in enumerate(shifted_volumes):
    save_image(vol, f"train/patient_001_shift{i:02d}.nii.gz")

# Result: 16 training samples from 1 scan (16× data increase)
```

**Notes:**
- Effective dataset size increases by `num_shifts×`
- All volumes maintain consistent physical extent (when `preserve_extent=True`)
- This is **LEGITIMATE augmentation** (not artificial data)
- Only works for downsampling (cannot upsample with this method)

---

## Complete Augmentation Pipeline Example

```python
from pycemrg_image_analysis.utilities.io import load_image, save_image
from pycemrg_image_analysis.utilities.intensity import normalize_min_max
from pycemrg_image_analysis.utilities.augmentation import (
    create_slice_shifted_volumes,
    augment_brightness,
    augment_contrast,
    augment_gamma,
    augment_noise,
)
import SimpleITK as sitk
import numpy as np

# ============================================================
# Step 1: Load and normalize high-res scan
# ============================================================
img = load_image("patient_001.nii.gz")

# Normalize to [0, 1]
volume = sitk.GetArrayFromImage(img)  # (Z, Y, X)
volume_norm = normalize_min_max(volume)

img_norm = sitk.GetImageFromArray(volume_norm)
img_norm.CopyInformation(img)

# ============================================================
# Step 2: Generate slice-shifted versions (spatial augmentation)
# ============================================================
shifted_volumes = create_slice_shifted_volumes(
    img_norm,
    target_z_spacing=8.0,
    num_shifts=16,
    preserve_extent=True
)

print(f"Created {len(shifted_volumes)} slice-shifted volumes")

# ============================================================
# Step 3: Apply intensity augmentation to each shift
# ============================================================
for shift_idx, shifted_img in enumerate(shifted_volumes):
    # Convert to NumPy for intensity augmentation
    shifted_array = sitk.GetArrayFromImage(shifted_img)  # (Z, Y, X)
    
    # Create 3 intensity variants per shift
    for variant_idx in range(3):
        # Compose intensity augmentations
        augmented = shifted_array.copy()
        
        seed_base = shift_idx * 1000 + variant_idx
        
        # Apply augmentation chain
        augmented = augment_brightness(
            augmented, factor=0.1, seed=seed_base + 0
        )
        augmented = augment_contrast(
            augmented, factor=0.1, seed=seed_base + 1
        )
        augmented = augment_gamma(
            augmented, gamma_range=(0.9, 1.1), seed=seed_base + 2
        )
        augmented = augment_noise(
            augmented, noise_std=0.01, seed=seed_base + 3
        )
        
        # Convert back to SimpleITK
        augmented_img = sitk.GetImageFromArray(augmented)
        augmented_img.CopyInformation(shifted_img)
        
        # Save augmented volume
        output_path = (
            f"train/patient_001_shift{shift_idx:02d}_int{variant_idx:02d}.nii.gz"
        )
        save_image(augmented_img, output_path)

# ============================================================
# Result: 16 shifts × 3 intensity variants = 48 training volumes
# ============================================================
print("Created 48 augmented training volumes from 1 scan (48× increase)")
```

---

## ROI-Aware Augmentation Example

```python
from pycemrg_image_analysis.utilities.augmentation import (
    augment_brightness,
    augment_contrast,
    augment_noise,
)

# Load volume and segmentation
volume = normalize_min_max(raw_volume)  # (Z, Y, X), [0, 1]
segmentation = load_segmentation()  # (Z, Y, X), integer labels

# Create ROI mask (e.g., augment only myocardium)
myo_label = 500
myo_mask = (segmentation == myo_label)

# Apply intensity augmentation only to myocardium
augmented = volume.copy()
augmented = augment_brightness(augmented, factor=0.1, mask=myo_mask, seed=42)
augmented = augment_contrast(augmented, factor=0.1, mask=myo_mask, seed=43)
augmented = augment_noise(augmented, noise_std=0.02, mask=myo_mask, seed=44)

# Result: Only myocardium intensities changed, background unchanged
```

---

## Design Principles

1. **Normalized Input Required:** All intensity functions expect `[0, 1]` range for consistent augmentation magnitudes across different scans

2. **Composability:** Functions can be chained without intermediate conversions (all NumPy-based for intensity)

3. **Reproducibility:** All functions accept `seed` parameter for deterministic augmentation pipelines

4. **ROI-Aware:** Optional `mask` parameter enables targeted augmentation (e.g., augment only tissue, not background)

5. **Metadata Preservation:** Spatial augmentation maintains SimpleITK metadata (origin, spacing, direction)

6. **Legitimate Augmentation:** Slice-shift is not artificial data—it samples real anatomical positions at different z-offsets

---

## Expected Training Impact

**Without augmentation (38 scans):**
- Training data: 38 volumes
- FAE training MSE: plateaus at 9e-3 (underfitting)

**With full augmentation (38 scans):**
- Slice-shift: 16× increase → 608 volumes
- Intensity variants: 3× increase → 1,824 volumes
- FAE training MSE: target 1e-3 to 1e-4 (proper convergence)

---

## Dependencies

- `numpy`: Core array operations
- `SimpleITK`: Spatial augmentation and medical image I/O
- `pycemrg_image_analysis.utilities.artifact_simulation.downsample_volume()`: Used internally by `create_slice_shifted_volumes()`

---

## Related Utilities

- `pycemrg_image_analysis.utilities.intensity.normalize_min_max()`: Normalize volumes to `[0, 1]` before augmentation
- `pycemrg_image_analysis.utilities.artifact_simulation.downsample_volume()`: Downsample with extent preservation
- `pycemrg_image_analysis.utilities.io`: Load/save SimpleITK images
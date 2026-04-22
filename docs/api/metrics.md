# `pycemrg_image_analysis.utilities.metrics` API Reference

**Module:** Image quality metrics for volume comparison and validation.

**Convention:** All functions expect pre-normalized data in `[0, 1]` range and arrays in `(Z, Y, X)` format.

---

## Functions

### `compute_mse()`

```python
def compute_mse(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> float
```

Compute Mean Squared Error between two volumes.

**Parameters:**
- `predicted` (np.ndarray): Predicted volume in `(Z, Y, X)` format
- `ground_truth` (np.ndarray): Ground truth volume (same shape as predicted)

**Returns:**
- `float`: MSE value (scalar). Lower is better.

**Raises:**
- `ValueError`: If shapes don't match

**Example:**
```python
pred = np.random.rand(64, 128, 128)
gt = np.random.rand(64, 128, 128)
mse = compute_mse(pred, gt)
```

---

### `compute_psnr()`

```python
def compute_psnr(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    data_range: float = 1.0
) -> float
```

Compute Peak Signal-to-Noise Ratio between two volumes.

**Parameters:**
- `predicted` (np.ndarray): Predicted volume in `(Z, Y, X)` format
- `ground_truth` (np.ndarray): Ground truth volume (same shape as predicted)
- `data_range` (float, optional): Maximum possible pixel value. Default: `1.0` for normalized `[0,1]` data

**Returns:**
- `float`: PSNR in dB (higher is better). Typical range: 20-50 dB for images.

**Raises:**
- `ValueError`: If shapes don't match or MSE is zero

**Notes:**
- Formula: `PSNR = 10 * log10(data_range² / MSE)`
- Returns `inf` if volumes are identical (MSE = 0)

**Example:**
```python
pred = np.random.rand(64, 128, 128)
gt = np.random.rand(64, 128, 128)
psnr = compute_psnr(pred, gt, data_range=1.0)
print(f"PSNR: {psnr:.2f} dB")
```

---

### `compute_ssim()`

```python
def compute_ssim(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    data_range: float = 1.0,
    win_size: int = 7,
    channel_axis: int = None,
    **kwargs
) -> float
```

Compute Structural Similarity Index between two volumes using 3D SSIM.

**Parameters:**
- `predicted` (np.ndarray): Predicted volume in `(Z, Y, X)` format
- `ground_truth` (np.ndarray): Ground truth volume (same shape as predicted)
- `data_range` (float, optional): Maximum possible pixel value. Default: `1.0`
- `win_size` (int, optional): Window size for local statistics (must be odd). Default: `7`
- `channel_axis` (int, optional): Channel axis. Default: `None` for single-channel 3D volumes
- `**kwargs`: Additional arguments passed to `skimage.metrics.structural_similarity`
  - `gaussian_weights` (bool): Use Gaussian weighting
  - `sigma` (float): Standard deviation for Gaussian kernel
  - `use_sample_covariance` (bool): Use sample covariance

**Returns:**
- `float`: SSIM value in `[0, 1]` where `1` indicates perfect structural similarity
  - `>0.9`: Excellent quality
  - `0.7-0.9`: Good quality
  - `<0.7`: Poor quality

**Raises:**
- `ValueError`: If shapes don't match or `win_size` is too large for volume

**Notes:**
- Computes 3D SSIM which preserves inter-slice context
- For anisotropic spacing (e.g., 1×1×8mm), consider using smaller `win_size`
- `win_size` cannot exceed the smallest dimension of the volume

**Example:**
```python
pred = np.random.rand(64, 128, 128)
gt = np.random.rand(64, 128, 128)

# Default 3D SSIM
ssim = compute_ssim(pred, gt, data_range=1.0)

# Custom window size for thin slices
ssim = compute_ssim(pred, gt, win_size=3)

# With Gaussian weighting
ssim = compute_ssim(pred, gt, gaussian_weights=True, sigma=1.5)
```

---

### `compute_gradient_error()`

```python
def compute_gradient_error(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    axis: int = 0
) -> float
```

Compute mean absolute gradient error along specified axis. Measures edge sharpness preservation.

**Parameters:**
- `predicted` (np.ndarray): Predicted volume in `(Z, Y, X)` format
- `ground_truth` (np.ndarray): Ground truth volume (same shape as predicted)
- `axis` (int, optional): Axis along which to compute gradient. Default: `0`
  - `0` = Z-axis (depth/slice direction)
  - `1` = Y-axis (height)
  - `2` = X-axis (width)

**Returns:**
- `float`: Mean absolute gradient difference (scalar). Lower is better.

**Raises:**
- `ValueError`: If shapes don't match or axis is invalid

**Notes:**
- Uses `numpy.gradient` with default spacing (1.0 between points)
- Particularly useful for evaluating interpolation quality along the interpolation axis
- For physical spacing-aware gradients, preprocess with spacing weights

**Example:**
```python
pred = np.random.rand(64, 128, 128)
gt = np.random.rand(64, 128, 128)

# Test Z-axis gradient preservation (interpolation quality)
z_error = compute_gradient_error(pred, gt, axis=0)

# Test X and Y axes
x_error = compute_gradient_error(pred, gt, axis=2)
y_error = compute_gradient_error(pred, gt, axis=1)
```

---

### `compare_volumes()`

```python
def compare_volumes(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    data_range: float = 1.0,
    metrics: List[str] = None
) -> Dict[str, float]
```

Compute multiple quality metrics at once for efficient batch comparison.

**Parameters:**
- `predicted` (np.ndarray): Predicted volume in `(Z, Y, X)` format
- `ground_truth` (np.ndarray): Ground truth volume (same shape as predicted)
- `data_range` (float, optional): Data range for PSNR/SSIM. Default: `1.0`
- `metrics` (List[str], optional): List of metric names to compute. Default: `['mse', 'psnr', 'ssim', 'gradient']`

**Available Metrics:**
- `'mse'`: Mean Squared Error
- `'psnr'`: Peak Signal-to-Noise Ratio
- `'ssim'`: Structural Similarity Index
- `'gradient'`: Gradient error (Z-axis by default, equivalent to `gradient_z`)
- `'gradient_x'`: Gradient error along X-axis
- `'gradient_y'`: Gradient error along Y-axis
- `'gradient_z'`: Gradient error along Z-axis

**Returns:**
- `dict`: Dictionary mapping metric names to float values

**Raises:**
- `ValueError`: If unknown metric name is provided or shapes don't match

**Notes:**
- Failed metrics return `NaN` instead of raising exceptions
- Efficiently computes only requested metrics
- Default metric set covers most validation scenarios

**Example:**
```python
pred = np.random.rand(64, 128, 128)
gt = np.random.rand(64, 128, 128)

# Compute all default metrics
results = compare_volumes(pred, gt)
print(results)
# {'mse': 0.0012, 'psnr': 29.2, 'ssim': 0.94, 'gradient': 0.015}

# Compute specific metrics
results = compare_volumes(pred, gt, metrics=['mse', 'psnr', 'ssim'])
print(f"MSE: {results['mse']:.4f}, PSNR: {results['psnr']:.2f} dB")

# Include all gradient axes
results = compare_volumes(
    pred, gt,
    metrics=['mse', 'gradient_x', 'gradient_y', 'gradient_z']
)
```

---

## Usage Pattern for Interpolation Validation

```python
from pycemrg_image_analysis.utilities.metrics import compare_volumes
import numpy as np

# Load volumes (assumes orchestrator has loaded and normalized to [0,1])
ground_truth = ...  # Shape: (Z, Y, X)
interpolated = ...  # Shape: (Z', Y, X) where Z' != Z

# Compute comprehensive quality metrics
results = compare_volumes(
    interpolated,
    ground_truth,
    data_range=1.0,
    metrics=['mse', 'psnr', 'ssim', 'gradient']
)

# Evaluate quality
print(f"MSE: {results['mse']:.6f}")
print(f"PSNR: {results['psnr']:.2f} dB")
print(f"SSIM: {results['ssim']:.4f}")
print(f"Z-Gradient Error: {results['gradient']:.6f}")

# Quality thresholds (example)
if results['psnr'] > 30 and results['ssim'] > 0.9:
    print("High quality interpolation")
elif results['psnr'] > 25 and results['ssim'] > 0.8:
    print("Acceptable quality")
else:
    print("Poor quality - review parameters")
```

---

## Design Principles

1. **Pre-normalized Input:** All functions assume data is in `[0, 1]` range. Normalization is the responsibility of the orchestrator/preprocessing pipeline.

2. **Axis Convention:** All arrays must be in `(Z, Y, X)` format, consistent with `pycemrg-image-analysis` suite convention.

3. **Type Safety:** All functions return Python `float`, not NumPy scalars, for consistent API behavior.

4. **Error Handling:** Shape mismatches raise `ValueError` immediately. `compare_volumes()` catches per-metric errors and returns `NaN` for failed metrics.

5. **Stateless:** Functions have no side effects and can be called in any order.

---

## Dependencies

- `numpy`: Core array operations
- `scikit-image`: SSIM computation (`skimage.metrics.structural_similarity`)

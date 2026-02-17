# scripts/inspection/inspect_spatial_coordinates_logic.py

import numpy as np
import SimpleITK as sitk
from pycemrg_image_analysis.utilities.spatial import (
    _transform_physical_to_index_vectorized,
    _transform_physical_to_index_precise,
    sample_image_at_points,
)

size = [10, 10, 10]
img = sitk.Image(size, sitk.sitkUInt8)
img.SetSpacing([1.0, 1.0, 1.0])
img.SetOrigin([0.0, 0.0, 0.0])

arr = np.arange(np.prod(size), dtype=np.uint8).reshape(10, 10, 10)
img_from_arr = sitk.GetImageFromArray(arr)
img_from_arr.CopyInformation(img)

point = np.array([[5.5, 5.5, 5.5]])

# Check what each helper returns
idx_vec, bounds_vec = _transform_physical_to_index_vectorized(img_from_arr, point)
idx_pre, bounds_pre = _transform_physical_to_index_precise(img_from_arr, point)

continuous = img_from_arr.TransformPhysicalPointToContinuousIndex([5.5, 5.5, 5.5])

print(f"ContinuousIndex raw output: {continuous}")
print(f"Vectorized index: {idx_vec}, in_bounds: {bounds_vec}")
print(f"Precise index:    {idx_pre}, in_bounds: {bounds_pre}")

# Check what sample_image_at_points returns
_, val_fast = sample_image_at_points(img_from_arr, point, precise=False)
_, val_prec = sample_image_at_points(img_from_arr, point, precise=True)

print(f"Fast value:    {val_fast}")
print(f"Precise value: {val_prec}")

# Manual lookup for reference
x, y, z = idx_vec[0]
print(f"Manual lookup arr[z={z}, y={y}, x={x}] = {arr[z, y, x]}")
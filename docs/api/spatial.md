
## `pycemrg_image_analysis.utilities.spatial` - Coordinate Transformation Functions

Added functions for mesh-image spatial mapping (voxel-to-physical coordinate transformation).

### Quick Reference

| Function                      | Purpose                                 | Input                             | Output                                                      |
| ----------------------------- | --------------------------------------- | --------------------------------- | ----------------------------------------------------------- |
| `get_voxel_physical_bounds()` | Get physical bounding boxes for voxels  | Voxel indices `(Z,Y,X)`           | Bounds `[xmin,ymin,zmin,xmax,ymax,zmax]`, centers `(X,Y,Z)` |
| `extract_slice_voxels()`      | Extract voxel indices/values from slice | Slice index, axis, optional label | Indices `(Z,Y,X)`, values                                   |

---

### `get_voxel_physical_bounds()`

```python
def get_voxel_physical_bounds(
    image: sitk.Image,
    voxel_indices: np.ndarray,  # (N, 3) in (Z, Y, X)
) -> Tuple[np.ndarray, np.ndarray]
```

Get physical-space bounding boxes and centers for voxels. Used for spatial queries between images and meshes.

**Returns:**
- `bounds`: `(N, 6)` array in format `[xmin, ymin, zmin, xmax, ymax, zmax]` (VTK standard)
- `centers`: `(N, 3)` array of voxel centers in `(X, Y, Z)` physical coordinates

**Example:**
```python
from pycemrg_image_analysis.utilities.spatial import get_voxel_physical_bounds
import numpy as np

# Get scar voxels from segmentation
seg_array = sitk.GetArrayFromImage(seg_img)  # (Z, Y, X)
scar_indices = np.argwhere(seg_array == scar_label)  # (N, 3) in (Z, Y, X)

# Get physical bounds
bounds, centers = get_voxel_physical_bounds(seg_img, scar_indices)

# Query mesh nodes inside each voxel
for i, bbox in enumerate(bounds):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    nodes_inside = mesh.find_nodes_in_box(bbox)
```

**Notes:**
- Fully vectorized (handles millions of voxels efficiently)
- Supports rotated images via affine transformation
- Input uses NumPy `(Z, Y, X)` convention
- Output uses physical space `(X, Y, Z)` convention

---

### `extract_slice_voxels()`

```python
def extract_slice_voxels(
    image: sitk.Image,
    slice_index: int,
    slice_axis: str = 'z',
    label: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]
```

Extract voxel indices and values from a single slice. Useful for slice-by-slice processing.

**Parameters:**
- `slice_axis`: `'x'`, `'y'`, or `'z'` (default: `'z'`)
- `label`: If provided, only return voxels matching this label

**Returns:**
- `voxel_indices`: `(N, 3)` array in `(Z, Y, X)` format
- `voxel_values`: `(N,)` array of intensities/labels

**Empty Results:** Returns `(0, 3)` and `(0,)` arrays if no voxels match.

**Example:**
```python
from pycemrg_image_analysis.utilities.spatial import (
    extract_slice_voxels,
    get_voxel_physical_bounds,
)

# Extract scar voxels from Z-slice 10
scar_indices, scar_values = extract_slice_voxels(
    seg_img, slice_index=10, slice_axis='z', label=scar_label
)

if len(scar_indices) > 0:
    # Get physical bounds for spatial queries
    bounds, centers = get_voxel_physical_bounds(seg_img, scar_indices)
    print(f"Found {len(bounds)} scar voxels in slice")
else:
    print("No scar in this slice")
```

---

### Complete Workflow: CardioScar Integration

```python
import SimpleITK as sitk
import numpy as np
from pycemrg_image_analysis.utilities.io import load_image
from pycemrg_image_analysis.utilities.spatial import (
    extract_slice_voxels,
    get_voxel_physical_bounds,
)

# Load LGE image and segmentation
lge_img = load_image("lge.nii.gz")
seg_img = load_image("segmentation.nii.gz")

# Process slice-by-slice
for slice_idx in range(seg_img.GetSize()[2]):  # Z-slices
    # Extract scar voxels from this slice
    scar_indices, _ = extract_slice_voxels(
        seg_img, 
        slice_index=slice_idx, 
        slice_axis='z', 
        label=scar_label
    )
    
    if len(scar_indices) == 0:
        continue  # No scar in this slice
    
    # Get physical bounding boxes
    bounds, centers = get_voxel_physical_bounds(seg_img, scar_indices)
    
    # Query mesh nodes inside each voxel (CardioScar utilities)
    for voxel_bbox in bounds:
        xmin, ymin, zmin, xmax, ymax, zmax = voxel_bbox
        nodes_in_voxel = mesh.find_nodes_in_box(voxel_bbox)
        
        # Extract LGE intensity for nodes...
        # (CardioScar-specific processing)
```

---

### Design Notes

**Coordinate Conventions:**
- **Voxel indices** (input): `(Z, Y, X)` order (NumPy convention)
- **Physical coordinates** (output): `(X, Y, Z)` order (standard medical imaging)

**Rotation Support:**
- Uses affine transformation matrix (no resampling)
- Handles arbitrary image orientations via direction matrix
- Bounding boxes are axis-aligned in physical space

**Performance:**
- Vectorized operations (no Python loops)
- Processes 10,000+ voxels in <100ms

**Bounding Box Format:**
- Standard VTK: `[xmin, ymin, zmin, xmax, ymax, zmax]`
- Compatible with most mesh query libraries

---

### Migration from Legacy Code

**Old approach:**
```python
# Loop-based (slow)
for idx in vox_indices:
    world_coord = img.TransformIndexToPhysicalPoint(tuple(reversed(idx)))
    corners = []
    for dz, dy, dx in itertools.product([0,1], repeat=3):
        corner = img.TransformIndexToPhysicalPoint(...)
        corners.append(corner)
```

**New approach:**
```python
# Vectorized (1000× faster)
bounds, centers = get_voxel_physical_bounds(img, vox_indices)
```

### `sample_image_at_points()`
```python
def sample_image_at_points(
    image: sitk.Image,
    physical_points: np.ndarray,  # (N, 3) in (X, Y, Z)
    precise: bool = False,
) -> Tuple[np.ndarray, np.ndarray]
```

Sample image intensities at physical coordinates. Maps mesh node positions to voxel values. Points outside the image volume are silently excluded.

**Parameters:**
- `physical_points`: `(N, 3)` array in `(X, Y, Z)` physical space convention
- `precise`: If `False` (default), uses vectorized affine inverse (~1000× faster). If `True`, delegates to `TransformPhysicalPointToContinuousIndex` per point (use for debugging)

Both modes use **containment semantics** (`floor`): returns the voxel the point falls *inside*, not the nearest voxel centre.

**Returns:**
- `sampled_indices`: `(M,)` integer indices into `physical_points` — which input points were successfully sampled
- `sampled_values`: `(M,)` float array of intensity values

If no points fall inside the image, returns empty `(0,)` arrays.

**Known behaviour:** At exact voxel boundaries, fast and precise modes may disagree. Use `precise=True` for boundary-sensitive workflows.

**Example:**
```python
mesh_coords = np.array(mesh.points)  # (N, 3) in (X, Y, Z)
indices, values = sample_image_at_points(lge_image, mesh_coords)

# Map values back to full mesh
intensity_field = np.zeros(len(mesh_coords))
intensity_field[indices] = values
```

---

**Full API:** See extended `spatial.py` docstrings for detailed documentation.
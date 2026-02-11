# src/pycemrg_image_analysis/utilities/spatial.py

import logging
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def compute_target_shape(
    original_shape: Tuple[int, int, int],
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """
    Calculate target voxel shape from spacing change.

    Args:
        original_shape: Original image shape (H, W, D)
        original_spacing: Original voxel spacing (x, y, z) in mm
        target_spacing: Target voxel spacing (x, y, z) in mm

    Returns:
        Target shape (H, W, D) maintaining physical dimensions
    """
    return tuple(
        int(np.round(orig_dim * orig_space / tgt_space))
        for orig_dim, orig_space, tgt_space in zip(
            original_shape, original_spacing, target_spacing
        )
    )


def compute_actual_spacing(
    original_spacing: Tuple[float, float, float],  # (sx, sy, sz)
    original_shape: Tuple[int, int, int],  # (nx, ny, nz) ← Must be XYZ!
    target_shape: Tuple[int, int, int],  # (nx, ny, nz) ← Must be XYZ!
) -> Tuple[float, float, float]:
    """
    Calculate actual voxel spacing from shape change.

    Args:
        original_spacing: Original voxel spacing (sx, sy, sz) in mm
        original_shape: Original image shape (nx, ny, nz) in XYZ order
        target_shape: Target image shape (nx, ny, nz) in XYZ order

    Returns:
        Actual achieved spacing (sx, sy, sz) in mm
    """
    return tuple(
        orig_space * orig_dim / tgt_dim
        for orig_space, orig_dim, tgt_dim in zip(
            original_spacing, original_shape, target_shape
        )
    )


def resample_to_isotropic(image: sitk.Image, target_spacing: float = 1.0) -> sitk.Image:
    """
    Resample an image to isotropic spacing (x=y=z).
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_spacing = (target_spacing, target_spacing, target_spacing)

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


# =============================================================================
# COORDINATE TRANSFORMATION UTILITIES
# =============================================================================

def _indices_to_physical_vectorized(
    image: sitk.Image,
    indices: np.ndarray  # (N, 3) in (Z, Y, X) order
) -> np.ndarray:
    """
    Vectorized conversion of voxel indices to physical coordinates.
    
    Uses affine transformation matrix to handle rotated images correctly.
    
    Args:
        image: SimpleITK image with spacing, origin, direction metadata
        indices: (N, 3) array of voxel indices in (Z, Y, X) order
    
    Returns:
        (N, 3) array of physical coordinates in (X, Y, Z) order
    
    Note:
        This function handles arbitrary image orientations via the
        direction matrix. No assumption of axis-aligned images.
    """
    # Convert from (Z, Y, X) to (X, Y, Z) for SimpleITK convention
    indices_xyz = indices[:, [2, 1, 0]]  # Swap columns
    
    # Get image metadata
    origin = np.array(image.GetOrigin())  # (X, Y, Z)
    spacing = np.array(image.GetSpacing())  # (X, Y, Z)
    direction = np.array(image.GetDirection()).reshape(3, 3)  # 3x3 matrix
    
    # Affine transformation: Physical = Origin + Direction @ (Spacing * Index)
    # This handles rotated images correctly via the direction matrix
    scaled = indices_xyz * spacing  # Element-wise: (N, 3)
    physical = origin + (direction @ scaled.T).T  # Matrix multiply, then transpose back
    
    return physical  # (N, 3) in (X, Y, Z)


def get_voxel_physical_bounds(
    image: sitk.Image,
    voxel_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get physical-space bounding boxes and centers for voxels.
    
    Computes axis-aligned bounding boxes in physical space for a set of
    voxel indices. This is useful for spatial queries between images and
    meshes (e.g., finding which mesh nodes fall inside specific voxels).
    
    Args:
        image: SimpleITK image with spacing, origin, direction metadata
        voxel_indices: (N, 3) array of voxel indices in (Z, Y, X) order
    
    Returns:
        Tuple of (bounds, centers):
        - bounds: (N, 6) array in format [xmin, ymin, zmin, xmax, ymax, zmax]
                  Standard VTK bounding box format
        - centers: (N, 3) array of voxel centers in (X, Y, Z) physical coordinates
    
    Note:
        - Input voxel_indices use NumPy (Z, Y, X) convention
        - Output coordinates use physical space (X, Y, Z) convention
        - Handles rotated images correctly via direction matrix
        - Bounding boxes are axis-aligned in physical space (not voxel space)
    
    Example:
        >>> # Get voxels from segmentation
        >>> seg_array = sitk.GetArrayFromImage(seg_img)  # (Z, Y, X)
        >>> scar_indices = np.argwhere(seg_array == scar_label)  # (N, 3) in (Z, Y, X)
        >>> 
        >>> # Get physical bounds
        >>> bounds, centers = get_voxel_physical_bounds(seg_img, scar_indices)
        >>> 
        >>> # Query mesh nodes inside voxel bounds
        >>> for i, bbox in enumerate(bounds):
        ...     xmin, ymin, zmin, xmax, ymax, zmax = bbox
        ...     nodes_inside = mesh.query_nodes_in_box(bbox)
        ...     # Process nodes...
    
    Performance:
        Fully vectorized - processes millions of voxels efficiently.
    """
    if voxel_indices.shape[1] != 3:
        raise ValueError(
            f"voxel_indices must have shape (N, 3), got {voxel_indices.shape}"
        )
    
    # Transform voxel corner indices to physical coordinates
    # Corner (0,0,0) is the min corner, (1,1,1) is the max corner
    min_corner_indices = voxel_indices.astype(float)  # (N, 3)
    max_corner_indices = voxel_indices.astype(float) + 1.0  # (N, 3)
    
    min_corners_physical = _indices_to_physical_vectorized(image, min_corner_indices)
    max_corners_physical = _indices_to_physical_vectorized(image, max_corner_indices)
    
    # For axis-aligned bounding boxes in physical space, take min/max across corners
    # Note: If image is rotated, the "min" corner in voxel space may not be
    # the min corner in physical space, so we compute element-wise min/max
    bounds = np.column_stack([
        np.minimum(min_corners_physical[:, 0], max_corners_physical[:, 0]),  # xmin
        np.minimum(min_corners_physical[:, 1], max_corners_physical[:, 1]),  # ymin
        np.minimum(min_corners_physical[:, 2], max_corners_physical[:, 2]),  # zmin
        np.maximum(min_corners_physical[:, 0], max_corners_physical[:, 0]),  # xmax
        np.maximum(min_corners_physical[:, 1], max_corners_physical[:, 1]),  # ymax
        np.maximum(min_corners_physical[:, 2], max_corners_physical[:, 2]),  # zmax
    ])  # (N, 6)
    
    # Compute voxel centers (midpoint between corners)
    centers = (min_corners_physical + max_corners_physical) / 2.0  # (N, 3)
    
    return bounds, centers


def extract_slice_voxels(
    image: sitk.Image,
    slice_index: int,
    slice_axis: str = 'z',
    label: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract voxel indices and values from a single slice.
    
    Useful for slice-by-slice processing, visualization, or selective
    extraction of labeled regions.
    
    Args:
        image: SimpleITK image
        slice_index: Index along slice_axis (0-indexed)
        slice_axis: 'x', 'y', or 'z' (default: 'z')
        label: If provided, only return voxels matching this label value.
               If None, return all voxels in the slice.
    
    Returns:
        Tuple of (voxel_indices, voxel_values):
        - voxel_indices: (N, 3) array in (Z, Y, X) format
        - voxel_values: (N,) array of voxel intensities/labels
        
        If no voxels match the label, returns empty arrays with shapes (0, 3) and (0,).
    
    Raises:
        ValueError: If slice_axis not in ['x', 'y', 'z'] or slice_index out of bounds
    
    Example:
        >>> # Extract all voxels from Z-slice 10
        >>> indices, values = extract_slice_voxels(lge_image, slice_index=10, slice_axis='z')
        >>> print(f"Slice has {len(indices)} voxels")
        >>> 
        >>> # Extract only scar voxels (label=3) from slice
        >>> scar_indices, scar_values = extract_slice_voxels(
        ...     seg_image, slice_index=10, slice_axis='z', label=3
        ... )
        >>> 
        >>> # Get physical bounds for scar voxels
        >>> if len(scar_indices) > 0:
        ...     bounds, centers = get_voxel_physical_bounds(seg_image, scar_indices)
        >>> else:
        ...     print("No scar voxels in this slice")
    
    Note:
        - Returns voxel_indices in (Z, Y, X) convention to match NumPy arrays
        - Empty slices or no-match labels return (0, 3) and (0,) arrays (documented behavior)
    """
    # Validate slice_axis
    slice_axis = slice_axis.lower()
    if slice_axis not in ['x', 'y', 'z']:
        raise ValueError(f"slice_axis must be 'x', 'y', or 'z', got '{slice_axis}'")
    
    # Get array and validate slice_index
    array = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    shape_zyx = array.shape
    
    # Map slice_axis to array dimension
    axis_map = {'z': 0, 'y': 1, 'x': 2}
    axis_dim = axis_map[slice_axis]
    
    if slice_index < 0 or slice_index >= shape_zyx[axis_dim]:
        raise ValueError(
            f"slice_index {slice_index} out of bounds for {slice_axis}-axis "
            f"(size {shape_zyx[axis_dim]})"
        )
    
    # Extract slice
    if slice_axis == 'z':
        slice_2d = array[slice_index, :, :]  # (Y, X)
    elif slice_axis == 'y':
        slice_2d = array[:, slice_index, :]  # (Z, X)
    else:  # slice_axis == 'x'
        slice_2d = array[:, :, slice_index]  # (Z, Y)
    
    # Apply label filter if provided
    if label is not None:
        # Convert label to array dtype for comparison
        label_typed = np.array(label, dtype=array.dtype).item()
        mask = slice_2d == label_typed
    else:
        # All voxels in slice
        mask = np.ones_like(slice_2d, dtype=bool)
    
    # Get 2D indices of matching voxels
    indices_2d = np.argwhere(mask)  # (N, 2)
    
    # If no voxels match, return empty arrays
    if len(indices_2d) == 0:
        return np.empty((0, 3), dtype=int), np.empty(0, dtype=array.dtype)
    
    # Convert 2D indices to 3D indices (Z, Y, X)
    if slice_axis == 'z':
        # 2D indices are (Y, X), need to add Z
        z_coords = np.full((len(indices_2d), 1), slice_index, dtype=int)
        indices_3d = np.column_stack([z_coords, indices_2d])  # (N, 3) -> (Z, Y, X)
    elif slice_axis == 'y':
        # 2D indices are (Z, X), need to add Y
        y_coords = np.full((len(indices_2d), 1), slice_index, dtype=int)
        indices_3d = np.column_stack([
            indices_2d[:, 0:1],  # Z
            y_coords,  # Y
            indices_2d[:, 1:2],  # X
        ])
    else:  # slice_axis == 'x'
        # 2D indices are (Z, Y), need to add X
        x_coords = np.full((len(indices_2d), 1), slice_index, dtype=int)
        indices_3d = np.column_stack([
            indices_2d,  # (Z, Y)
            x_coords,  # X
        ])
    
    # Extract voxel values
    voxel_values = slice_2d[mask]
    
    return indices_3d, voxel_values
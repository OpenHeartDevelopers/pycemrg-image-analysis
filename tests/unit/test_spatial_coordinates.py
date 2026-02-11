# tests/unit/test_spatial_coordinates.py

"""
Unit tests for spatial coordinate transformation utilities.

Tests cover:
- Voxel-to-physical coordinate transformation
- Bounding box computation
- Slice extraction
- Edge cases (empty results, rotated images, out-of-bounds)
"""

import pytest
import numpy as np
import SimpleITK as sitk

from pycemrg_image_analysis.utilities.spatial import (
    get_voxel_physical_bounds,
    extract_slice_voxels,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_image():
    """Create simple 10x10x10 image with identity transform."""
    size = [10, 10, 10]  # (X, Y, Z) in SimpleITK
    spacing = [1.0, 1.0, 1.0]
    origin = [0.0, 0.0, 0.0]
    
    img = sitk.Image(size, sitk.sitkUInt8)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    # Direction is identity by default
    
    # Fill with gradient data
    arr = np.arange(np.prod(size), dtype=np.uint8).reshape(size[2], size[1], size[0])
    img_from_arr = sitk.GetImageFromArray(arr)
    img_from_arr.CopyInformation(img)
    
    return img_from_arr


@pytest.fixture
def anisotropic_image():
    """Create image with anisotropic spacing."""
    size = [64, 64, 20]  # (X, Y, Z)
    spacing = [0.5, 0.5, 2.0]  # Anisotropic in Z
    origin = [10.0, 20.0, 30.0]  # Non-zero origin
    
    img = sitk.Image(size, sitk.sitkFloat32)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    
    # Fill with label data
    arr = np.zeros((size[2], size[1], size[0]), dtype=np.float32)
    arr[5:15, 20:40, 20:40] = 3.0  # Label 3 in central region
    
    img_from_arr = sitk.GetImageFromArray(arr)
    img_from_arr.CopyInformation(img)
    
    return img_from_arr


@pytest.fixture
def rotated_image():
    """Create image with 45-degree rotation around Z-axis."""
    size = [10, 10, 10]
    spacing = [1.0, 1.0, 1.0]
    origin = [0.0, 0.0, 0.0]
    
    img = sitk.Image(size, sitk.sitkUInt8)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    
    # 45-degree rotation around Z-axis
    cos45 = np.cos(np.pi / 4)
    sin45 = np.sin(np.pi / 4)
    direction = [
        cos45, -sin45, 0.0,  # X-axis rotated
        sin45,  cos45, 0.0,  # Y-axis rotated
        0.0,    0.0,   1.0,  # Z-axis unchanged
    ]
    img.SetDirection(direction)
    
    return img


# =============================================================================
# VOXEL PHYSICAL BOUNDS TESTS
# =============================================================================

def test_single_voxel_bounds_identity(simple_image):
    """Test bounds for single voxel with identity transform."""
    # Voxel at (0, 0, 0) in (Z, Y, X)
    voxel_indices = np.array([[0, 0, 0]])
    
    bounds, centers = get_voxel_physical_bounds(simple_image, voxel_indices)
    
    # With unit spacing and zero origin:
    # Voxel (0,0,0) spans [0,1] in each dimension
    expected_bounds = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])  # [xmin, ymin, zmin, xmax, ymax, zmax]
    expected_centers = np.array([[0.5, 0.5, 0.5]])  # (X, Y, Z)
    
    assert np.allclose(bounds, expected_bounds)
    assert np.allclose(centers, expected_centers)


def test_multiple_voxels_bounds(simple_image):
    """Test bounds for multiple voxels."""
    voxel_indices = np.array([
        [0, 0, 0],  # Corner voxel
        [5, 5, 5],  # Center voxel
        [9, 9, 9],  # Opposite corner
    ])  # (Z, Y, X)
    
    bounds, centers = get_voxel_physical_bounds(simple_image, voxel_indices)
    
    assert bounds.shape == (3, 6)
    assert centers.shape == (3, 3)
    
    # Check first voxel (0,0,0) in (Z,Y,X) -> (X,Y,Z) = (0,0,0)
    assert np.allclose(bounds[0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert np.allclose(centers[0], [0.5, 0.5, 0.5])
    
    # Check center voxel (5,5,5) -> (X,Y,Z) = (5,5,5)
    assert np.allclose(bounds[1], [5.0, 5.0, 5.0, 6.0, 6.0, 6.0])
    assert np.allclose(centers[1], [5.5, 5.5, 5.5])


def test_anisotropic_spacing_bounds(anisotropic_image):
    """Test bounds with anisotropic spacing and non-zero origin."""
    # Voxel (0, 0, 0) in (Z, Y, X)
    voxel_indices = np.array([[0, 0, 0]])
    
    bounds, centers = get_voxel_physical_bounds(anisotropic_image, voxel_indices)
    
    # Spacing: (0.5, 0.5, 2.0) for (X, Y, Z)
    # Origin: (10.0, 20.0, 30.0) for (X, Y, Z)
    # Voxel (Z=0, Y=0, X=0) maps to physical (X,Y,Z) = (0,0,0)
    # Physical range: X[10.0, 10.5], Y[20.0, 20.5], Z[30.0, 32.0]
    expected_bounds = np.array([[10.0, 20.0, 30.0, 10.5, 20.5, 32.0]])
    expected_centers = np.array([[10.25, 20.25, 31.0]])
    
    assert np.allclose(bounds, expected_bounds)
    assert np.allclose(centers, expected_centers)


def test_rotated_image_bounds(rotated_image):
    """Test that bounds are computed correctly for rotated images."""
    # For rotated images, bounding boxes should still be axis-aligned
    # in physical space, but corners may swap min/max
    voxel_indices = np.array([[0, 0, 0]])
    
    bounds, centers = get_voxel_physical_bounds(rotated_image, voxel_indices)
    
    # Should return valid bounds (min < max in each dimension)
    assert bounds[0, 0] < bounds[0, 3]  # xmin < xmax
    assert bounds[0, 1] < bounds[0, 4]  # ymin < ymax
    assert bounds[0, 2] < bounds[0, 5]  # zmin < zmax
    
    # Centers should be valid coordinates
    assert centers.shape == (1, 3)
    assert np.all(np.isfinite(centers))


def test_empty_voxel_list():
    """Test with empty voxel list."""
    img = sitk.Image([10, 10, 10], sitk.sitkUInt8)
    voxel_indices = np.empty((0, 3), dtype=int)
    
    bounds, centers = get_voxel_physical_bounds(img, voxel_indices)
    
    assert bounds.shape == (0, 6)
    assert centers.shape == (0, 3)


def test_invalid_voxel_indices_shape():
    """Test that invalid voxel_indices shape raises error."""
    img = sitk.Image([10, 10, 10], sitk.sitkUInt8)
    voxel_indices = np.array([[0, 0]])  # Only 2 columns, need 3
    
    with pytest.raises(ValueError, match="must have shape"):
        get_voxel_physical_bounds(img, voxel_indices)


# =============================================================================
# EXTRACT SLICE VOXELS TESTS
# =============================================================================

def test_extract_z_slice_all_voxels(simple_image):
    """Extract all voxels from a Z-slice."""
    slice_index = 5
    
    indices, values = extract_slice_voxels(simple_image, slice_index, slice_axis='z')
    
    # Should return all voxels in slice (10x10 = 100 voxels)
    assert indices.shape == (100, 3)
    assert values.shape == (100,)
    
    # All Z coordinates should be slice_index
    assert np.all(indices[:, 0] == slice_index)
    
    # Y coordinates should range [0, 9]
    assert set(indices[:, 1]) == set(range(10))
    
    # X coordinates should range [0, 9]
    assert set(indices[:, 2]) == set(range(10))


def test_extract_y_slice_all_voxels(simple_image):
    """Extract all voxels from a Y-slice."""
    slice_index = 3
    
    indices, values = extract_slice_voxels(simple_image, slice_index, slice_axis='y')
    
    assert indices.shape == (100, 3)  # 10 Z × 10 X = 100
    assert values.shape == (100,)
    
    # All Y coordinates should be slice_index
    assert np.all(indices[:, 1] == slice_index)


def test_extract_x_slice_all_voxels(simple_image):
    """Extract all voxels from an X-slice."""
    slice_index = 7
    
    indices, values = extract_slice_voxels(simple_image, slice_index, slice_axis='x')
    
    assert indices.shape == (100, 3)  # 10 Z × 10 Y = 100
    assert values.shape == (100,)
    
    # All X coordinates should be slice_index
    assert np.all(indices[:, 2] == slice_index)


def test_extract_slice_with_label_filter(anisotropic_image):
    """Extract only voxels matching specific label."""
    slice_index = 10  # Central Z-slice where label 3 exists
    
    indices, values = extract_slice_voxels(
        anisotropic_image, slice_index, slice_axis='z', label=3
    )
    
    # Should only return label 3 voxels
    assert np.all(values == 3.0)
    
    # Should be non-empty (label 3 exists in slice 10)
    assert len(indices) > 0
    
    # All Z coordinates should be slice_index
    assert np.all(indices[:, 0] == slice_index)


def test_extract_slice_no_matching_label(anisotropic_image):
    """Extract slice with label that doesn't exist."""
    slice_index = 0  # Slice where label 3 doesn't exist
    
    indices, values = extract_slice_voxels(
        anisotropic_image, slice_index, slice_axis='z', label=3
    )
    
    # Should return empty arrays
    assert indices.shape == (0, 3)
    assert values.shape == (0,)


def test_extract_slice_invalid_axis(simple_image):
    """Test invalid slice_axis raises error."""
    with pytest.raises(ValueError, match="must be 'x', 'y', or 'z'"):
        extract_slice_voxels(simple_image, 0, slice_axis='invalid')


def test_extract_slice_out_of_bounds(simple_image):
    """Test out-of-bounds slice_index raises error."""
    with pytest.raises(ValueError, match="out of bounds"):
        extract_slice_voxels(simple_image, 100, slice_axis='z')  # Image only has 10 Z-slices
    
    with pytest.raises(ValueError, match="out of bounds"):
        extract_slice_voxels(simple_image, -1, slice_axis='z')


def test_extract_slice_case_insensitive_axis(simple_image):
    """Test that slice_axis is case-insensitive."""
    indices_lower, _ = extract_slice_voxels(simple_image, 5, slice_axis='z')
    indices_upper, _ = extract_slice_voxels(simple_image, 5, slice_axis='Z')
    
    assert np.array_equal(indices_lower, indices_upper)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_slice_extraction_to_bounds_workflow(anisotropic_image):
    """Test complete workflow: extract slice → get bounds."""
    # Extract scar voxels from slice
    slice_index = 10
    scar_indices, scar_values = extract_slice_voxels(
        anisotropic_image, slice_index, slice_axis='z', label=3
    )
    
    assert len(scar_indices) > 0, "Should find scar voxels"
    
    # Get physical bounds for scar voxels
    bounds, centers = get_voxel_physical_bounds(anisotropic_image, scar_indices)
    
    assert bounds.shape[0] == len(scar_indices)
    assert centers.shape[0] == len(scar_indices)
    
    # All bounds should be valid (min < max)
    assert np.all(bounds[:, 0] < bounds[:, 3])  # xmin < xmax
    assert np.all(bounds[:, 1] < bounds[:, 4])  # ymin < ymax
    assert np.all(bounds[:, 2] < bounds[:, 5])  # zmin < zmax


def test_vectorized_performance():
    """Test that vectorized implementation handles large voxel sets."""
    # Create large image
    size = [100, 100, 100]
    img = sitk.Image(size, sitk.sitkUInt8)
    img.SetSpacing([1.0, 1.0, 1.0])
    img.SetOrigin([0.0, 0.0, 0.0])
    
    # Create large voxel index array (10,000 voxels)
    np.random.seed(42)
    voxel_indices = np.random.randint(0, 100, size=(10000, 3))
    
    # Should complete quickly (vectorized)
    import time
    start = time.time()
    bounds, centers = get_voxel_physical_bounds(img, voxel_indices)
    elapsed = time.time() - start
    
    assert elapsed < 0.1, f"Took {elapsed:.3f}s - vectorization may be broken"
    assert bounds.shape == (10000, 6)
    assert centers.shape == (10000, 3)


def test_empty_slice_to_bounds(anisotropic_image):
    """Test workflow when slice has no matching voxels."""
    # Extract voxels with non-existent label
    indices, values = extract_slice_voxels(
        anisotropic_image, slice_index=0, slice_axis='z', label=999
    )
    
    assert len(indices) == 0
    
    # Get bounds for empty set
    bounds, centers = get_voxel_physical_bounds(anisotropic_image, indices)
    
    assert bounds.shape == (0, 6)
    assert centers.shape == (0, 3)
# tests/unit/test_components.py

"""
Unit tests for connected component cleanup functions.

Tests both keep_largest_component (per-label) and keep_largest_structure
(multi-label) functions at the components.py and postprocessing.py layers.
"""

import pytest
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from pycemrg.data import LabelManager
from pycemrg_image_analysis.utilities.components import (
    keep_largest_component,
    keep_largest_structure
)
from pycemrg_image_analysis.utilities.postprocessing import (
    keep_largest_component_by_name,
    keep_largest_structure_by_name
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_image_with_floating_blobs():
    """
    Create test image with two labels, each having multiple components.
    
    Label 5: One large blob (6 voxels) + one small blob (2 voxels)
    Label 7: One large blob (4 voxels) + one small blob (1 voxel)
    """
    array = np.array([
        [5, 5, 5, 0, 7, 7],
        [5, 5, 5, 0, 7, 7],
        [0, 0, 0, 0, 0, 0],
        [5, 5, 0, 0, 0, 7],  # Small floating blobs
    ], dtype=np.uint8)
    
    image = sitk.GetImageFromArray(array)
    image.SetSpacing([1.0, 1.0, 1.0])
    image.SetOrigin([0.0, 0.0, 0.0])
    
    return image


@pytest.fixture
def multi_label_structure_image():
    """
    Create image where labels 3 and 4 form one structure, plus floating garbage.
    
    Main structure: Labels 3 and 4 connected (left side)
    Floating garbage: Labels 3 and 4 disconnected (right side)
    """
    array = np.array([
        [3, 3, 4, 0, 0, 3],  # Main structure (3+4) | Floating 3
        [3, 4, 4, 0, 0, 4],  # Main structure (3+4) | Floating 4
        [3, 3, 4, 0, 0, 0],  # Main structure (3+4)
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    
    image = sitk.GetImageFromArray(array)
    image.SetSpacing([1.0, 1.0, 1.0])
    image.SetOrigin([0.0, 0.0, 0.0])
    
    return image


@pytest.fixture
def single_component_image():
    """Image where each label has only one component (no floating blobs)."""
    array = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    
    image = sitk.GetImageFromArray(array)
    return image


@pytest.fixture
def empty_image():
    """Image with only background (all zeros)."""
    array = np.zeros((3, 3), dtype=np.uint8)
    image = sitk.GetImageFromArray(array)
    return image


@pytest.fixture
def test_label_manager(tmp_path):
    """Create temporary label manager for name-based tests."""
    labels_yaml = tmp_path / "labels.yaml"
    labels_yaml.write_text("""
labels:
  label_3: 3
  label_4: 4
  label_5: 5
  label_7: 7
""")
    return LabelManager(config_path=labels_yaml)


# =============================================================================
# Tests: keep_largest_component (Per-Label Independent)
# =============================================================================

def test_keep_largest_component_removes_small_blobs(simple_image_with_floating_blobs):
    """Each label should keep only its largest component."""
    result = keep_largest_component(simple_image_with_floating_blobs, [5, 7])
    result_array = sitk.GetArrayFromImage(result)
    
    # Label 5: should keep large blob (6 voxels), remove small (2 voxels)
    label_5_count = np.sum(result_array == 5)
    assert label_5_count == 6, "Label 5 should have 6 voxels (large blob only)"
    
    # Label 7: should keep large blob (4 voxels), remove small (1 voxel)
    label_7_count = np.sum(result_array == 7)
    assert label_7_count == 4, "Label 7 should have 4 voxels (large blob only)"


def test_keep_largest_component_preserves_other_labels(simple_image_with_floating_blobs):
    """Labels not in the list should be completely untouched."""
    # Add another label that shouldn't be affected
    array = sitk.GetArrayFromImage(simple_image_with_floating_blobs)
    array[0, 0] = 9  # Add label 9
    modified_image = sitk.GetImageFromArray(array)
    modified_image.CopyInformation(simple_image_with_floating_blobs)
    
    # Only clean label 5
    result = keep_largest_component(modified_image, [5])
    result_array = sitk.GetArrayFromImage(result)
    
    # Label 7 should be unchanged (including its small blob)
    original_7_count = np.sum(array == 7)
    result_7_count = np.sum(result_array == 7)
    assert result_7_count == original_7_count, "Label 7 should be unchanged"
    
    # Label 9 should be unchanged
    assert result_array[0, 0] == 9, "Label 9 should be unchanged"


def test_keep_largest_component_single_component_unchanged(single_component_image):
    """Labels with single component should remain unchanged."""
    original_array = sitk.GetArrayFromImage(single_component_image)
    result = keep_largest_component(single_component_image, [1, 2])
    result_array = sitk.GetArrayFromImage(result)
    
    np.testing.assert_array_equal(result_array, original_array)


def test_keep_largest_component_missing_label_no_error(simple_image_with_floating_blobs):
    """Requesting cleanup of non-existent label should not raise."""
    result = keep_largest_component(simple_image_with_floating_blobs, [5, 99])
    result_array = sitk.GetArrayFromImage(result)
    
    # Label 5 should be cleaned
    assert np.sum(result_array == 5) == 6
    
    # Label 99 doesn't exist, no error
    assert 99 not in result_array


def test_keep_largest_component_preserves_metadata(simple_image_with_floating_blobs):
    """Metadata (spacing, origin, direction) should be preserved."""
    simple_image_with_floating_blobs.SetSpacing([1.5, 2.0, 2.5])
    simple_image_with_floating_blobs.SetOrigin([10.0, 20.0, 30.0])
    
    result = keep_largest_component(simple_image_with_floating_blobs, [5])
    
    assert result.GetSpacing() == simple_image_with_floating_blobs.GetSpacing()
    assert result.GetOrigin() == simple_image_with_floating_blobs.GetOrigin()
    assert result.GetDirection() == simple_image_with_floating_blobs.GetDirection()


def test_keep_largest_component_empty_list(simple_image_with_floating_blobs):
    """Empty label list should return unchanged image."""
    original_array = sitk.GetArrayFromImage(simple_image_with_floating_blobs)
    result = keep_largest_component(simple_image_with_floating_blobs, [])
    result_array = sitk.GetArrayFromImage(result)
    
    np.testing.assert_array_equal(result_array, original_array)


# =============================================================================
# Tests: keep_largest_structure (Multi-Label Structure)
# =============================================================================

def test_keep_largest_structure_removes_floating_chunks(multi_label_structure_image):
    """Should keep main structure (3+4 connected) and remove floating chunks."""
    result = keep_largest_structure(multi_label_structure_image, [3, 4])
    result_array = sitk.GetArrayFromImage(result)
    
    # Main structure on left (9 voxels total) should remain
    # Floating chunks on right (2 voxels) should be removed
    total_voxels = np.sum(result_array > 0)
    assert total_voxels == 9, "Only main structure should remain (9 voxels)"
    
    # Specifically, floating chunks at [0,5] and [1,5] should be zero
    assert result_array[0, 5] == 0, "Floating label 3 should be removed"
    assert result_array[1, 5] == 0, "Floating label 4 should be removed"


def test_keep_largest_structure_preserves_internal_labels(multi_label_structure_image):
    """Internal labels within kept structure should be preserved."""
    result = keep_largest_structure(multi_label_structure_image, [3, 4])
    result_array = sitk.GetArrayFromImage(result)
    
    # Main structure should still have both labels 3 and 4
    assert 3 in result_array, "Label 3 should be present in main structure"
    assert 4 in result_array, "Label 4 should be present in main structure"
    
    # Check specific positions from main structure
    assert result_array[0, 0] == 3, "Label 3 preserved in main structure"
    assert result_array[0, 2] == 4, "Label 4 preserved in main structure"


def test_keep_largest_structure_default_all_labels(multi_label_structure_image):
    """label_values=None should process all non-zero labels."""
    result = keep_largest_structure(multi_label_structure_image, label_values=None)
    result_array = sitk.GetArrayFromImage(result)
    
    # Should keep main structure, remove floating chunks
    total_voxels = np.sum(result_array > 0)
    assert total_voxels == 9, "Default should clean all labels"


def test_keep_largest_structure_single_label(simple_image_with_floating_blobs):
    """Single label in list should work (same as keep_largest_component for that label)."""
    result = keep_largest_structure(simple_image_with_floating_blobs, [5])
    result_array = sitk.GetArrayFromImage(result)
    
    # Should keep only largest blob of label 5
    assert np.sum(result_array == 5) == 6, "Should keep largest blob (6 voxels)"
    
    # Label 7 should be untouched (not in label_values)
    original_7_count = np.sum(sitk.GetArrayFromImage(simple_image_with_floating_blobs) == 7)
    result_7_count = np.sum(result_array == 7)
    assert result_7_count == original_7_count, "Label 7 should be unchanged"


def test_keep_largest_structure_empty_image(empty_image):
    """Empty image (all zeros) should return unchanged."""
    result = keep_largest_structure(empty_image)
    result_array = sitk.GetArrayFromImage(result)
    
    assert np.all(result_array == 0), "Empty image should remain empty"


def test_keep_largest_structure_missing_labels_no_error(simple_image_with_floating_blobs):
    """Specifying non-existent labels should not raise."""
    result = keep_largest_structure(simple_image_with_floating_blobs, [99, 100])
    result_array = sitk.GetArrayFromImage(result)
    
    # Original image should be returned unchanged
    original_array = sitk.GetArrayFromImage(simple_image_with_floating_blobs)
    np.testing.assert_array_equal(result_array, original_array)


def test_keep_largest_structure_preserves_unspecified_labels(multi_label_structure_image):
    """Labels not in label_values should be completely untouched."""
    # Add label 9 to image
    array = sitk.GetArrayFromImage(multi_label_structure_image)
    array[3, 3] = 9
    modified_image = sitk.GetImageFromArray(array)
    modified_image.CopyInformation(multi_label_structure_image)
    
    # Only clean structure of labels 3 and 4
    result = keep_largest_structure(modified_image, [3, 4])
    result_array = sitk.GetArrayFromImage(result)
    
    # Label 9 should be untouched
    assert result_array[3, 3] == 9, "Label 9 should be preserved"


# =============================================================================
# Tests: Name-Based Wrappers
# =============================================================================

def test_keep_largest_component_by_name(simple_image_with_floating_blobs, test_label_manager):
    """Name-based wrapper should translate names and delegate correctly."""
    result = keep_largest_component_by_name(
        simple_image_with_floating_blobs,
        ["label_5", "label_7"],
        test_label_manager
    )
    result_array = sitk.GetArrayFromImage(result)
    
    # Should behave same as integer version
    assert np.sum(result_array == 5) == 6, "Label 5 cleaned"
    assert np.sum(result_array == 7) == 4, "Label 7 cleaned"


def test_keep_largest_structure_by_name(multi_label_structure_image, test_label_manager):
    """Name-based wrapper should translate names and delegate correctly."""
    result = keep_largest_structure_by_name(
        multi_label_structure_image,
        ["label_3", "label_4"],
        test_label_manager
    )
    result_array = sitk.GetArrayFromImage(result)
    
    # Should keep main structure
    total_voxels = np.sum(result_array > 0)
    assert total_voxels == 9, "Main structure kept"


def test_keep_largest_structure_by_name_default_all_labels(multi_label_structure_image):
    """Calling with label_names=None should process all labels."""
    result = keep_largest_structure_by_name(
        multi_label_structure_image,
        label_names=None,
        label_manager=None
    )
    result_array = sitk.GetArrayFromImage(result)
    
    total_voxels = np.sum(result_array > 0)
    assert total_voxels == 9, "Should clean all labels by default"


def test_keep_largest_component_by_name_invalid_name(simple_image_with_floating_blobs, test_label_manager):
    """Invalid label name should log warning and skip (not raise)."""
    # Should not raise, just skip invalid name
    result = keep_largest_component_by_name(
        simple_image_with_floating_blobs,
        ["label_5", "invalid_label"],
        test_label_manager
    )
    result_array = sitk.GetArrayFromImage(result)
    
    # Label 5 should still be cleaned
    assert np.sum(result_array == 5) == 6


def test_keep_largest_structure_by_name_requires_manager_with_names(multi_label_structure_image):
    """Providing label_names without label_manager should raise ValueError."""
    with pytest.raises(ValueError, match="label_manager is required"):
        keep_largest_structure_by_name(
            multi_label_structure_image,
            label_names=["label_3"],
            label_manager=None
        )


def test_keep_largest_component_by_name_all_invalid_returns_unchanged(simple_image_with_floating_blobs, test_label_manager):
    """If all label names are invalid, should return image unchanged."""
    original_array = sitk.GetArrayFromImage(simple_image_with_floating_blobs)
    
    result = keep_largest_component_by_name(
        simple_image_with_floating_blobs,
        ["invalid_1", "invalid_2"],
        test_label_manager
    )
    result_array = sitk.GetArrayFromImage(result)
    
    np.testing.assert_array_equal(result_array, original_array)


# =============================================================================
# Edge Cases
# =============================================================================

def test_tie_in_component_size():
    """When two components have same size, should keep first one found."""
    # Create image with two equal-sized blobs of label 5
    array = np.array([
        [5, 5, 0, 5, 5],
        [5, 5, 0, 5, 5],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    
    image = sitk.GetImageFromArray(array)
    result = keep_largest_component(image, [5])
    result_array = sitk.GetArrayFromImage(result)
    
    # Should keep exactly 4 voxels (one of the two equal blobs)
    assert np.sum(result_array == 5) == 4


def test_3d_image():
    """Should work correctly with 3D images."""
    # Create 3D image with floating blob
    array = np.zeros((4, 4, 4), dtype=np.uint8)
    
    # Main blob (8 voxels)
    array[0:2, 0:2, 0:2] = 5
    
    # Floating blob (1 voxel)
    array[3, 3, 3] = 5
    
    image = sitk.GetImageFromArray(array)
    result = keep_largest_component(image, [5])
    result_array = sitk.GetArrayFromImage(result)
    
    # Should keep main blob (8 voxels), remove floating (1 voxel)
    assert np.sum(result_array == 5) == 8
    assert result_array[3, 3, 3] == 0, "Floating blob should be removed"


def test_diagonal_connectivity():
    """Test that diagonal connectivity is handled correctly."""
    # SimpleITK uses face connectivity by default (not diagonal)
    array = np.array([
        [5, 0, 5],
        [0, 5, 0],
        [5, 0, 5],
    ], dtype=np.uint8)
    
    image = sitk.GetImageFromArray(array)
    result = keep_largest_component(image, [5])
    result_array = sitk.GetArrayFromImage(result)
    
    # With face connectivity, center voxel is separate from corners
    # Should keep center (1 voxel) or corners (4 voxels) - corners are larger
    # Or all might be separate - SimpleITK default is face connectivity
    # So we expect at most 4 voxels (the corner group if they're connected)
    label_5_count = np.sum(result_array == 5)
    assert label_5_count <= 4, "Should not connect diagonally by default"
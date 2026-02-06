import numpy as np
import SimpleITK as sitk
import pytest

from pycemrg_image_analysis.utilities import artifact_simulation


@pytest.fixture
def sample_image() -> sitk.Image:
    """A sample 10x20x30 image with 1x1x1mm spacing."""
    size = [10, 20, 30]
    image = sitk.Image(size, sitk.sitkFloat32)
    image.SetSpacing([1.0, 1.0, 1.0])
    image.SetOrigin([0.0, 0.0, 0.0])
    # Fill with a gradient to test interpolation
    arr = np.arange(np.prod(size), dtype=np.float32).reshape(size[2], size[1], size[0])
    img_from_arr = sitk.GetImageFromArray(arr)
    img_from_arr.CopyInformation(image)
    return img_from_arr


def test_downsample_volume(sample_image):
    """Test legacy downsampling behavior (preserve_extent=False)."""
    target_spacing = (1.0, 1.0, 5.0)  # Downsample in Z

    downsampled = artifact_simulation.downsample_volume(sample_image, target_spacing)

    # Verify new spacing
    assert np.allclose(downsampled.GetSpacing(), target_spacing)

    # Verify new size is calculated correctly:
    # Original Z size = 30, Z spacing = 1.0
    # New Z spacing = 5.0
    # Ratio = 1.0 / 5.0 = 0.2
    # New Z size = 30 * 0.2 = 6
    expected_size = [10, 20, 6]
    assert downsampled.GetSize() == tuple(expected_size)

    # Verify physical properties are preserved
    assert np.allclose(downsampled.GetOrigin(), sample_image.GetOrigin())
    assert np.allclose(downsampled.GetDirection(), sample_image.GetDirection())


def test_downsample_volume_anisotropic(sample_image):
    """Test anisotropic downsampling (legacy mode)."""
    target_spacing = (2.0, 4.0, 5.0)  # Downsample in all axes

    downsampled = artifact_simulation.downsample_volume(sample_image, target_spacing)

    assert np.allclose(downsampled.GetSpacing(), target_spacing)

    # New X size = 10 * (1/2) = 5
    # New Y size = 20 * (1/4) = 5
    # New Z size = 30 * (1/5) = 6
    expected_size = [5, 5, 6]
    assert downsampled.GetSize() == tuple(expected_size)


def test_downsample_preserves_extent():
    """Test that extent-aware downsampling preserves physical dimensions."""
    # Create test image: 192 slices @ 0.5mm spacing (matching ticket example)
    size = [512, 512, 192]
    image = sitk.Image(size, sitk.sitkFloat32)
    image.SetSpacing([0.28125, 0.28125, 0.5])
    image.SetOrigin([0.0, 0.0, 0.0])

    # Fill with some data
    arr = np.random.rand(size[2], size[1], size[0]).astype(np.float32)
    img_from_arr = sitk.GetImageFromArray(arr)
    img_from_arr.CopyInformation(image)

    # Calculate original extent
    original_size = np.array(size)
    original_spacing = np.array([0.28125, 0.28125, 0.5])
    original_extents = (original_size - 1) * original_spacing

    # Z extent: (192 - 1) * 0.5 = 95.5mm
    expected_z_extent = 95.5
    assert np.isclose(original_extents[2], expected_z_extent)

    # Downsample to 8mm with extent preservation
    target_spacing = (0.28125, 0.28125, 8.0)
    downsampled = artifact_simulation.downsample_volume(
        img_from_arr, target_spacing, preserve_extent=True
    )

    # Check new extent matches within tolerance
    new_size = np.array(downsampled.GetSize())
    new_spacing = np.array(downsampled.GetSpacing())
    new_extents = (new_size - 1) * new_spacing

    # Z extent should be preserved within 1 spacing unit
    assert abs(new_extents[2] - original_extents[2]) < target_spacing[2], (
        f"Z-extent mismatch: {original_extents[2]:.1f}mm → {new_extents[2]:.1f}mm"
    )

    # All extents should be preserved
    for i, axis in enumerate(["X", "Y", "Z"]):
        extent_diff = abs(new_extents[i] - original_extents[i])
        assert extent_diff < target_spacing[i], (
            f"{axis}-extent mismatch: {original_extents[i]:.1f}mm → {new_extents[i]:.1f}mm (diff: {extent_diff:.2f}mm)"
        )

    # Expected size from ticket: 13 slices
    # Calculation: round((192-1) * 0.5 / 8.0) + 1 = round(95.5/8.0) + 1 = round(11.9375) + 1 = 12 + 1 = 13
    assert downsampled.GetSize()[2] == 13, (
        f"Expected 13 Z slices, got {downsampled.GetSize()[2]}"
    )


def test_downsample_extent_vs_legacy():
    """Compare extent-preserving vs legacy behavior on realistic case."""
    # Create image with non-integer extent ratio
    size = [128, 128, 192]
    spacing = [1.0, 1.0, 0.5]

    image = sitk.Image(size, sitk.sitkFloat32)
    image.SetSpacing(spacing)

    target_spacing = (1.0, 1.0, 8.0)

    # Legacy mode
    legacy = artifact_simulation.downsample_volume(
        image, target_spacing, preserve_extent=False
    )

    # Extent-preserving mode
    extent_preserved = artifact_simulation.downsample_volume(
        image, target_spacing, preserve_extent=True
    )

    # Calculate extents
    original_extent_z = (size[2] - 1) * spacing[2]  # 95.5mm

    legacy_size = legacy.GetSize()
    legacy_extent_z = (legacy_size[2] - 1) * target_spacing[2]

    extent_size = extent_preserved.GetSize()
    extent_extent_z = (extent_size[2] - 1) * target_spacing[2]

    # Legacy uses ceil, so: ceil(192 * 0.5 / 8.0) = ceil(12.0) = 12 slices
    # Extent: (192-1) * 0.5 = 95.5, 95.5/8 = 11.9375, round = 12, +1 = 13 slices
    assert legacy_size[2] == 12
    assert extent_size[2] == 13

    # Extent-preserving should be closer to original extent
    legacy_error = abs(legacy_extent_z - original_extent_z)
    extent_error = abs(extent_extent_z - original_extent_z)

    assert extent_error < legacy_error, (
        f"Extent preservation failed: legacy error={legacy_error:.2f}mm, extent error={extent_error:.2f}mm"
    )


def test_downsample_extent_isotropic():
    """Test extent preservation with isotropic downsampling."""
    size = [100, 100, 100]
    spacing = [1.0, 1.0, 1.0]

    image = sitk.Image(size, sitk.sitkFloat32)
    image.SetSpacing(spacing)

    target_spacing = (2.5, 2.5, 2.5)

    downsampled = artifact_simulation.downsample_volume(
        image, target_spacing, preserve_extent=True
    )

    # Original extent: (100-1) * 1.0 = 99.0mm per axis
    # New size: round(99.0 / 2.5) + 1 = round(39.6) + 1 = 40 + 1 = 41
    expected_size = [41, 41, 41]
    assert downsampled.GetSize() == tuple(expected_size)

    # Verify extent is preserved
    original_extent = (100 - 1) * 1.0
    new_extent = (41 - 1) * 2.5  # = 100.0mm

    # Should be within 1 spacing unit
    assert abs(new_extent - original_extent) < 2.5


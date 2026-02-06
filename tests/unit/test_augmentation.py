# tests/unit/test_augmentation.py

"""
Unit tests for pycemrg_image_analysis.utilities.augmentation

Tests cover:
- Input validation (normalized range)
- Mask validation and application
- Reproducibility with seeds
- Intensity augmentation correctness
- Slice-shift spatial augmentation
"""

import pytest
import numpy as np
import SimpleITK as sitk

from pycemrg_image_analysis.utilities.augmentation import (
    augment_brightness,
    augment_contrast,
    augment_gamma,
    augment_noise,
    create_slice_shifted_volumes,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def normalized_volume():
    """Small normalized volume for testing."""
    np.random.seed(42)
    volume = np.random.rand(8, 16, 16).astype(np.float32)
    return volume


@pytest.fixture
def unnormalized_volume():
    """Volume not in [0, 1] range for validation testing."""
    np.random.seed(42)
    return np.random.rand(8, 16, 16).astype(np.float32) * 1000  # CT HU range


@pytest.fixture
def binary_mask():
    """Binary ROI mask."""
    mask = np.zeros((8, 16, 16), dtype=bool)
    mask[2:6, 4:12, 4:12] = True  # Central region
    return mask


@pytest.fixture
def highres_image():
    """High-resolution SimpleITK image for slice-shift testing."""
    size = [64, 64, 96]  # 96 slices
    spacing = [1.0, 1.0, 0.5]  # 0.5mm z-spacing
    
    img = sitk.Image(size, sitk.sitkFloat32)
    img.SetSpacing(spacing)
    img.SetOrigin([0.0, 0.0, 0.0])
    
    # Fill with gradient data
    arr = np.linspace(0, 1, np.prod(size)).reshape(size[2], size[1], size[0]).astype(np.float32)
    img_from_arr = sitk.GetImageFromArray(arr)
    img_from_arr.CopyInformation(img)
    
    return img_from_arr


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

def test_brightness_rejects_unnormalized(unnormalized_volume):
    """Brightness augmentation should reject unnormalized input."""
    with pytest.raises(ValueError, match="requires normalized"):
        augment_brightness(unnormalized_volume, factor=0.1)


def test_contrast_rejects_unnormalized(unnormalized_volume):
    """Contrast augmentation should reject unnormalized input."""
    with pytest.raises(ValueError, match="requires normalized"):
        augment_contrast(unnormalized_volume, factor=0.1)


def test_gamma_rejects_unnormalized(unnormalized_volume):
    """Gamma augmentation should reject unnormalized input."""
    with pytest.raises(ValueError, match="requires normalized"):
        augment_gamma(unnormalized_volume, gamma_range=(0.8, 1.2))


def test_noise_rejects_unnormalized(unnormalized_volume):
    """Noise augmentation should reject unnormalized input."""
    with pytest.raises(ValueError, match="requires normalized"):
        augment_noise(unnormalized_volume, noise_std=0.01)


def test_gamma_rejects_invalid_range(normalized_volume):
    """Gamma should reject invalid gamma_range."""
    with pytest.raises(ValueError, match="Invalid gamma_range"):
        augment_gamma(normalized_volume, gamma_range=(1.2, 0.8))  # Reversed
    
    with pytest.raises(ValueError, match="Invalid gamma_range"):
        augment_gamma(normalized_volume, gamma_range=(0.0, 1.0))  # Zero min


# =============================================================================
# MASK VALIDATION TESTS
# =============================================================================

def test_brightness_mask_shape_mismatch(normalized_volume):
    """Brightness should reject mask with wrong shape."""
    wrong_mask = np.ones((10, 10, 10), dtype=bool)  # Different shape
    
    with pytest.raises(ValueError, match="mask shape"):
        augment_brightness(normalized_volume, factor=0.1, mask=wrong_mask)


def test_contrast_mask_shape_mismatch(normalized_volume):
    """Contrast should reject mask with wrong shape."""
    wrong_mask = np.ones((10, 10, 10), dtype=bool)
    
    with pytest.raises(ValueError, match="mask shape"):
        augment_contrast(normalized_volume, factor=0.1, mask=wrong_mask)


# =============================================================================
# BRIGHTNESS AUGMENTATION TESTS
# =============================================================================

def test_brightness_changes_intensities(normalized_volume):
    """Brightness augmentation should change intensity values."""
    augmented = augment_brightness(normalized_volume, factor=0.1, seed=42)
    
    assert not np.allclose(augmented, normalized_volume)


def test_brightness_output_range(normalized_volume):
    """Brightness output should remain in [0, 1]."""
    augmented = augment_brightness(normalized_volume, factor=0.5, seed=42)
    
    assert augmented.min() >= 0.0
    assert augmented.max() <= 1.0


def test_brightness_reproducible(normalized_volume):
    """Same seed should produce same result."""
    aug1 = augment_brightness(normalized_volume, factor=0.1, seed=42)
    aug2 = augment_brightness(normalized_volume, factor=0.1, seed=42)
    
    assert np.allclose(aug1, aug2)


def test_brightness_different_seeds(normalized_volume):
    """Different seeds should produce different results."""
    aug1 = augment_brightness(normalized_volume, factor=0.1, seed=42)
    aug2 = augment_brightness(normalized_volume, factor=0.1, seed=99)
    
    assert not np.allclose(aug1, aug2)


def test_brightness_with_mask(normalized_volume, binary_mask):
    """Brightness with mask should only change masked region."""
    augmented = augment_brightness(normalized_volume, factor=0.5, mask=binary_mask, seed=42)
    
    # Unmasked region should be unchanged
    unmasked = ~binary_mask
    assert np.allclose(augmented[unmasked], normalized_volume[unmasked])
    
    # Masked region should be different
    assert not np.allclose(augmented[binary_mask], normalized_volume[binary_mask])


def test_brightness_preserves_shape(normalized_volume):
    """Brightness should preserve volume shape."""
    augmented = augment_brightness(normalized_volume, factor=0.1, seed=42)
    assert augmented.shape == normalized_volume.shape


# =============================================================================
# CONTRAST AUGMENTATION TESTS
# =============================================================================

def test_contrast_changes_intensities(normalized_volume):
    """Contrast augmentation should change intensity values."""
    augmented = augment_contrast(normalized_volume, factor=0.2, seed=42)
    
    assert not np.allclose(augmented, normalized_volume)


def test_contrast_output_range(normalized_volume):
    """Contrast output should remain in [0, 1]."""
    augmented = augment_contrast(normalized_volume, factor=0.5, seed=42)
    
    assert augmented.min() >= 0.0
    assert augmented.max() <= 1.0


def test_contrast_reproducible(normalized_volume):
    """Same seed should produce same result."""
    aug1 = augment_contrast(normalized_volume, factor=0.2, seed=42)
    aug2 = augment_contrast(normalized_volume, factor=0.2, seed=42)
    
    assert np.allclose(aug1, aug2)


def test_contrast_with_mask(normalized_volume, binary_mask):
    """Contrast with mask should only change masked region."""
    augmented = augment_contrast(normalized_volume, factor=0.3, mask=binary_mask, seed=42)
    
    # Unmasked region should be unchanged
    unmasked = ~binary_mask
    assert np.allclose(augmented[unmasked], normalized_volume[unmasked])


# =============================================================================
# GAMMA AUGMENTATION TESTS
# =============================================================================

def test_gamma_changes_intensities(normalized_volume):
    """Gamma augmentation should change intensity values."""
    augmented = augment_gamma(normalized_volume, gamma_range=(0.7, 1.3), seed=42)
    
    assert not np.allclose(augmented, normalized_volume)


def test_gamma_output_range(normalized_volume):
    """Gamma output should remain in [0, 1]."""
    augmented = augment_gamma(normalized_volume, gamma_range=(0.5, 2.0), seed=42)
    
    assert augmented.min() >= 0.0
    assert augmented.max() <= 1.0


def test_gamma_reproducible(normalized_volume):
    """Same seed should produce same result."""
    aug1 = augment_gamma(normalized_volume, gamma_range=(0.8, 1.2), seed=42)
    aug2 = augment_gamma(normalized_volume, gamma_range=(0.8, 1.2), seed=42)
    
    assert np.allclose(aug1, aug2)


def test_gamma_with_mask(normalized_volume, binary_mask):
    """Gamma with mask should only change masked region."""
    augmented = augment_gamma(normalized_volume, gamma_range=(0.7, 1.3), mask=binary_mask, seed=42)
    
    # Unmasked region should be unchanged
    unmasked = ~binary_mask
    assert np.allclose(augmented[unmasked], normalized_volume[unmasked])


# =============================================================================
# NOISE AUGMENTATION TESTS
# =============================================================================

def test_noise_changes_intensities(normalized_volume):
    """Noise augmentation should change intensity values."""
    augmented = augment_noise(normalized_volume, noise_std=0.05, seed=42)
    
    assert not np.allclose(augmented, normalized_volume)


def test_noise_output_range(normalized_volume):
    """Noise output should remain in [0, 1]."""
    augmented = augment_noise(normalized_volume, noise_std=0.1, seed=42)
    
    assert augmented.min() >= 0.0
    assert augmented.max() <= 1.0


def test_noise_reproducible(normalized_volume):
    """Same seed should produce same result."""
    aug1 = augment_noise(normalized_volume, noise_std=0.02, seed=42)
    aug2 = augment_noise(normalized_volume, noise_std=0.02, seed=42)
    
    assert np.allclose(aug1, aug2)


def test_noise_with_mask(normalized_volume, binary_mask):
    """Noise with mask should only change masked region."""
    augmented = augment_noise(normalized_volume, noise_std=0.05, mask=binary_mask, seed=42)
    
    # Unmasked region should be unchanged
    unmasked = ~binary_mask
    assert np.allclose(augmented[unmasked], normalized_volume[unmasked])


# =============================================================================
# AUGMENTATION COMPOSITION TESTS
# =============================================================================

def test_augmentation_chain(normalized_volume):
    """Multiple augmentations should be composable."""
    result = normalized_volume.copy()
    result = augment_brightness(result, factor=0.1, seed=42)
    result = augment_contrast(result, factor=0.1, seed=43)
    result = augment_gamma(result, gamma_range=(0.9, 1.1), seed=44)
    result = augment_noise(result, noise_std=0.01, seed=45)
    
    # Result should be different from original
    assert not np.allclose(result, normalized_volume)
    
    # Result should still be in valid range
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_augmentation_with_roi_mask(normalized_volume, binary_mask):
    """Full augmentation pipeline with ROI mask."""
    result = normalized_volume.copy()
    result = augment_brightness(result, factor=0.1, mask=binary_mask, seed=42)
    result = augment_contrast(result, factor=0.1, mask=binary_mask, seed=43)
    result = augment_noise(result, noise_std=0.01, mask=binary_mask, seed=44)
    
    # Unmasked region should be unchanged
    unmasked = ~binary_mask
    assert np.allclose(result[unmasked], normalized_volume[unmasked])
    
    # Masked region should be different
    assert not np.allclose(result[binary_mask], normalized_volume[binary_mask])


# =============================================================================
# SLICE-SHIFT AUGMENTATION TESTS
# =============================================================================

def test_slice_shift_creates_multiple_volumes(highres_image):
    """Slice-shift should create requested number of volumes."""
    shifted = create_slice_shifted_volumes(
        highres_image,
        target_z_spacing=8.0,
        num_shifts=8,
    )
    
    assert len(shifted) == 8
    assert all(isinstance(vol, sitk.Image) for vol in shifted)


def test_slice_shift_spacing_correct(highres_image):
    """Slice-shift output should have correct spacing."""
    target_z_spacing = 8.0
    shifted = create_slice_shifted_volumes(
        highres_image,
        target_z_spacing=target_z_spacing,
        num_shifts=4,
    )
    
    for vol in shifted:
        spacing = vol.GetSpacing()
        assert np.isclose(spacing[2], target_z_spacing, atol=0.1)


def test_slice_shift_preserves_extent(highres_image):
    """Slice-shift with preserve_extent should maintain physical extent."""
    original_size = highres_image.GetSize()
    original_spacing = highres_image.GetSpacing()
    original_extent_z = (original_size[2] - 1) * original_spacing[2]
    
    target_z_spacing = 8.0
    shifted = create_slice_shifted_volumes(
        highres_image,
        target_z_spacing=target_z_spacing,
        num_shifts=4,
        preserve_extent=True,
    )
    
    for vol in shifted:
        new_size = vol.GetSize()
        new_spacing = vol.GetSpacing()
        new_extent_z = (new_size[2] - 1) * new_spacing[2]
        
        # Extent should be within 1 spacing unit
        assert abs(new_extent_z - original_extent_z) < target_z_spacing


def test_slice_shift_rejects_upsampling(highres_image):
    """Slice-shift should reject target_z_spacing smaller than original."""
    original_spacing = highres_image.GetSpacing()
    
    with pytest.raises(ValueError, match="must be greater than"):
        create_slice_shifted_volumes(
            highres_image,
            target_z_spacing=original_spacing[2] * 0.5,  # Trying to upsample
            num_shifts=4,
        )


def test_slice_shift_clamps_excessive_shifts(highres_image):
    """Slice-shift should clamp num_shifts to maximum possible."""
    # For 96 slices @ 0.5mm → 8mm, step = 16
    # Maximum shifts = 16
    shifted = create_slice_shifted_volumes(
        highres_image,
        target_z_spacing=8.0,
        num_shifts=100,  # Request more than possible
    )
    
    # Should get at most 16 shifts (floor(8.0 / 0.5))
    assert len(shifted) <= 16


def test_slice_shift_different_offsets():
    """Each slice-shift should sample different z-positions."""
    # Create test image with identifiable slices
    size = [32, 32, 64]
    spacing = [1.0, 1.0, 1.0]
    
    img = sitk.Image(size, sitk.sitkFloat32)
    img.SetSpacing(spacing)
    
    # Fill each slice with its index value
    arr = np.zeros((size[2], size[1], size[0]), dtype=np.float32)
    for z in range(size[2]):
        arr[z, :, :] = z / size[2]  # Normalized slice index
    
    img_from_arr = sitk.GetImageFromArray(arr)
    img_from_arr.CopyInformation(img)
    
    # Create shifted volumes
    shifted = create_slice_shifted_volumes(
        img_from_arr,
        target_z_spacing=4.0,  # Step = 4
        num_shifts=4,
    )
    
    # Extract first slice from each shifted volume
    first_slices = []
    for vol in shifted:
        arr = sitk.GetArrayFromImage(vol)
        first_slices.append(arr[0, 0, 0])  # Value of first slice
    
    # Each shifted volume should start at different position
    # (values should be different)
    assert len(set(first_slices)) == len(first_slices)


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_full_augmentation_pipeline():
    """Test complete augmentation workflow for FAE training."""
    # Create synthetic volume
    np.random.seed(42)
    volume = np.random.rand(32, 64, 64).astype(np.float32)
    
    # Create ROI mask (central region)
    mask = np.zeros_like(volume, dtype=bool)
    mask[8:24, 16:48, 16:48] = True
    
    # Apply full augmentation chain
    augmented = volume.copy()
    augmented = augment_brightness(augmented, factor=0.1, mask=mask, seed=100)
    augmented = augment_contrast(augmented, factor=0.1, mask=mask, seed=101)
    augmented = augment_gamma(augmented, gamma_range=(0.9, 1.1), mask=mask, seed=102)
    augmented = augment_noise(augmented, noise_std=0.01, mask=mask, seed=103)
    
    # Verify output validity
    assert augmented.shape == volume.shape
    assert augmented.min() >= 0.0
    assert augmented.max() <= 1.0
    
    # Verify ROI was augmented
    assert not np.allclose(augmented[mask], volume[mask])
    
    # Verify background unchanged
    assert np.allclose(augmented[~mask], volume[~mask])
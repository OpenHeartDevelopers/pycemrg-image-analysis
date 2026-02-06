# src/pycemrg_image_analysis/utilities/augmentation.py

"""
Data augmentation utilities for medical image analysis.

This module provides augmentation strategies for increasing effective dataset
size while preserving the physical validity of medical imaging data.

Sections:
- Spatial Augmentation: Operations that modify spatial sampling
- Intensity Augmentation: Operations that modify voxel intensities
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION UTILITIES (Private)
# =============================================================================

def _validate_normalized_input(volume: np.ndarray, function_name: str) -> None:
    """
    Validate that input volume is normalized to [0, 1] range.
    
    Args:
        volume: Input volume to validate
        function_name: Name of calling function (for error message)
        
    Raises:
        ValueError: If volume is not in [0, 1] range
    """
    # Small tolerance for floating point precision
    vol_min = volume.min()
    vol_max = volume.max()
    
    if vol_min < -0.01 or vol_max > 1.01:
        raise ValueError(
            f"{function_name} requires normalized [0, 1] input. "
            f"Got range [{vol_min:.3f}, {vol_max:.3f}]. "
            f"Use normalize_min_max() from utilities.intensity first."
        )


def _validate_mask_shape(volume: np.ndarray, mask: np.ndarray, function_name: str) -> None:
    """
    Validate that mask shape matches volume shape.
    
    Args:
        volume: Input volume
        mask: Binary mask
        function_name: Name of calling function (for error message)
        
    Raises:
        ValueError: If shapes don't match
    """
    if volume.shape != mask.shape:
        raise ValueError(
            f"{function_name}: mask shape {mask.shape} doesn't match volume shape {volume.shape}"
        )


def _apply_mask(
    original: np.ndarray,
    augmented: np.ndarray,
    mask: Optional[np.ndarray]
) -> np.ndarray:
    """
    Apply mask to selectively use augmented values.
    
    Args:
        original: Original volume
        augmented: Augmented volume
        mask: Binary mask (True/1 = use augmented, False/0 = use original)
        
    Returns:
        Masked volume (augmented where mask is True, original elsewhere)
    """
    if mask is None:
        return augmented
    
    # Convert mask to boolean if needed
    mask_bool = mask.astype(bool)
    
    # Use augmented values where mask is True, original elsewhere
    result = original.copy()
    result[mask_bool] = augmented[mask_bool]
    
    return result


# =============================================================================
# INTENSITY AUGMENTATION
# =============================================================================

def augment_brightness(
    volume: np.ndarray,
    factor: float = 0.1,
    mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Randomly adjust image brightness.
    
    Applies a uniform random shift to all voxel intensities. This simulates
    variations in scanner calibration or acquisition parameters.
    
    Args:
        volume: Input volume in (Z, Y, X) format, normalized to [0, 1]
        factor: Maximum brightness change as fraction (e.g., 0.1 = ±10%)
        mask: Optional binary mask in (Z, Y, X) format. If provided,
              augmentation is applied only where mask > 0
        seed: Random seed for reproducibility
    
    Returns:
        Volume with adjusted brightness in (Z, Y, X) format, clipped to [0, 1]
    
    Raises:
        ValueError: If input is not normalized to [0, 1] or mask shape mismatches
    
    Example:
        >>> volume = normalize_min_max(raw_volume)  # [0, 1]
        >>> augmented = augment_brightness(volume, factor=0.1, seed=42)
        >>> # Intensities shifted by random value in [-0.1, +0.1]
        
        >>> # ROI-aware augmentation
        >>> myo_mask = (segmentation == myo_label)
        >>> augmented = augment_brightness(volume, factor=0.1, mask=myo_mask)
        >>> # Only myocardium brightness changed, background unchanged
    """
    _validate_normalized_input(volume, "augment_brightness")
    
    if mask is not None:
        _validate_mask_shape(volume, mask, "augment_brightness")
    
    # Random brightness shift
    rng = np.random.RandomState(seed)
    brightness_shift = rng.uniform(-factor, factor)
    
    # Apply shift and clip
    augmented = np.clip(volume + brightness_shift, 0.0, 1.0)
    
    # Apply mask if provided
    return _apply_mask(volume, augmented, mask)


def augment_contrast(
    volume: np.ndarray,
    factor: float = 0.1,
    mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Randomly adjust image contrast.
    
    Scales intensities around the mean value, simulating variations in
    scanner contrast settings or tissue response.
    
    Args:
        volume: Input volume in (Z, Y, X) format, normalized to [0, 1]
        factor: Maximum contrast change as fraction (e.g., 0.1 = ±10%)
        mask: Optional binary mask in (Z, Y, X) format. If provided,
              augmentation is applied only where mask > 0
        seed: Random seed for reproducibility
    
    Returns:
        Volume with adjusted contrast in (Z, Y, X) format, clipped to [0, 1]
    
    Raises:
        ValueError: If input is not normalized to [0, 1] or mask shape mismatches
    
    Implementation:
        contrast_factor = uniform(1 - factor, 1 + factor)
        mean = compute_mean(input)
        output = clip(mean + contrast_factor * (input - mean), 0, 1)
    
    Example:
        >>> volume = normalize_min_max(raw_volume)
        >>> augmented = augment_contrast(volume, factor=0.15, seed=42)
        >>> # Contrast scaled by factor in [0.85, 1.15]
    """
    _validate_normalized_input(volume, "augment_contrast")
    
    if mask is not None:
        _validate_mask_shape(volume, mask, "augment_contrast")
    
    # Random contrast factor
    rng = np.random.RandomState(seed)
    contrast_factor = rng.uniform(1.0 - factor, 1.0 + factor)
    
    # Compute mean (use masked mean if mask provided)
    if mask is not None:
        mask_bool = mask.astype(bool)
        mean = volume[mask_bool].mean() if mask_bool.any() else volume.mean()
    else:
        mean = volume.mean()
    
    # Apply contrast adjustment and clip
    augmented = np.clip(mean + contrast_factor * (volume - mean), 0.0, 1.0)
    
    # Apply mask if provided
    return _apply_mask(volume, augmented, mask)


def augment_gamma(
    volume: np.ndarray,
    gamma_range: Tuple[float, float] = (0.8, 1.2),
    mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply random gamma correction.
    
    Gamma correction is a non-linear intensity transformation that simulates
    variations in display calibration or tissue attenuation characteristics.
    
    Args:
        volume: Input volume in (Z, Y, X) format, normalized to [0, 1]
        gamma_range: (min_gamma, max_gamma) range. Values < 1 brighten,
                     values > 1 darken. Typical range: (0.8, 1.2)
        mask: Optional binary mask in (Z, Y, X) format. If provided,
              augmentation is applied only where mask > 0
        seed: Random seed for reproducibility
    
    Returns:
        Gamma-corrected volume in (Z, Y, X) format
    
    Raises:
        ValueError: If input is not normalized to [0, 1], gamma_range is invalid,
                    or mask shape mismatches
    
    Implementation:
        gamma = uniform(gamma_range[0], gamma_range[1])
        output = input ** gamma
    
    Example:
        >>> volume = normalize_min_max(raw_volume)
        >>> augmented = augment_gamma(volume, gamma_range=(0.7, 1.3), seed=42)
        >>> # Non-linear intensity transform
    """
    _validate_normalized_input(volume, "augment_gamma")
    
    if gamma_range[0] <= 0 or gamma_range[0] >= gamma_range[1]:
        raise ValueError(
            f"Invalid gamma_range {gamma_range}. "
            f"Must satisfy: 0 < min_gamma < max_gamma"
        )
    
    if mask is not None:
        _validate_mask_shape(volume, mask, "augment_gamma")
    
    # Random gamma value
    rng = np.random.RandomState(seed)
    gamma = rng.uniform(gamma_range[0], gamma_range[1])
    
    # Apply gamma correction
    augmented = np.power(volume, gamma)
    
    # Apply mask if provided
    return _apply_mask(volume, augmented, mask)


def augment_noise(
    volume: np.ndarray,
    noise_std: float = 0.01,
    mask: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add Gaussian noise to image.
    
    Simulates thermal noise from the scanner or quantum noise from
    photon-counting detectors.
    
    Args:
        volume: Input volume in (Z, Y, X) format, normalized to [0, 1]
        noise_std: Standard deviation of Gaussian noise. Typical: 0.01-0.05
        mask: Optional binary mask in (Z, Y, X) format. If provided,
              noise is added only where mask > 0
        seed: Random seed for reproducibility
    
    Returns:
        Volume with added noise in (Z, Y, X) format, clipped to [0, 1]
    
    Raises:
        ValueError: If input is not normalized to [0, 1] or mask shape mismatches
    
    Implementation:
        noise = normal(0, noise_std, shape=input.shape)
        output = clip(input + noise, 0, 1)
    
    Example:
        >>> volume = normalize_min_max(raw_volume)
        >>> augmented = augment_noise(volume, noise_std=0.02, seed=42)
        >>> # Gaussian noise with std=0.02 added
    """
    _validate_normalized_input(volume, "augment_noise")
    
    if mask is not None:
        _validate_mask_shape(volume, mask, "augment_noise")
    
    # Generate Gaussian noise
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_std, volume.shape)
    
    # Add noise and clip
    augmented = np.clip(volume + noise, 0.0, 1.0)
    
    # Apply mask if provided
    return _apply_mask(volume, augmented, mask)


# =============================================================================
# SPATIAL AUGMENTATION
# =============================================================================

def create_slice_shifted_volumes(
    img: sitk.Image,
    target_z_spacing: float,
    num_shifts: int = 16,
    preserve_extent: bool = True,
    interpolator: int = sitk.sitkLinear,
) -> List[sitk.Image]:
    from pycemrg_image_analysis.utilities.artifact_simulation import downsample_volume
    
    original_spacing = img.GetSpacing()
    original_z_spacing = original_spacing[2]
    original_size = img.GetSize()
    original_z_size = original_size[2]
    
    # Validate downsampling only
    if target_z_spacing <= original_z_spacing:
        raise ValueError(
            f"target_z_spacing ({target_z_spacing}) must be greater than "
            f"original z-spacing ({original_z_spacing})"
        )
    
    # Calculate step size
    step = int(round(target_z_spacing / original_z_spacing))
    max_shifts = min(step, original_z_size)
    
    if num_shifts > max_shifts:
        logger.warning(
            f"Requested {num_shifts} shifts, but only {max_shifts} possible. Using {max_shifts}."
        )
        num_shifts = max_shifts
    
    # Convert to NumPy for slicing
    volume = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    
    shifted_volumes = []
    
    for shift in range(num_shifts):
        # Extract every 'step'th slice starting at 'shift'
        sliced_volume = volume[shift::step, :, :]  # NumPy array slicing
        
        if sliced_volume.shape[0] < 2:
            logger.warning(f"Shift {shift} results in <2 slices. Skipping.")
            continue
        
        # Convert back to SimpleITK image
        sliced_img = sitk.GetImageFromArray(sliced_volume)
        
        # Adjust metadata
        new_origin = list(img.GetOrigin())
        new_origin[2] += shift * original_z_spacing  # Offset start position
        
        sliced_img.SetOrigin(new_origin)
        sliced_img.SetDirection(img.GetDirection())
        sliced_img.SetSpacing((
            original_spacing[0],
            original_spacing[1],
            original_z_spacing * step  # Actual spacing between kept slices
        ))
        
        # Now downsample to target spacing (handles XY if needed)
        target_spacing = (original_spacing[0], original_spacing[1], target_z_spacing)
        downsampled = downsample_volume(
            sliced_img,
            target_spacing,
            interpolator=interpolator,
            preserve_extent=preserve_extent
        )
        
        shifted_volumes.append(downsampled)
    
    logger.info(f"Created {len(shifted_volumes)} slice-shifted volumes")
    return shifted_volumes
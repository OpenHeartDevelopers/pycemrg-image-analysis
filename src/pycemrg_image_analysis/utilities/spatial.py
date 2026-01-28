# src/pycemrg_image_analysis/utilities/spatial.py

import logging
import numpy as np
import SimpleITK as sitk
from typing import Tuple


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
        for orig_dim, orig_space, tgt_space
        in zip(original_shape, original_spacing, target_spacing)
    )


def compute_actual_spacing(
    original_spacing: Tuple[float, float, float],
    original_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """
    Calculate actual voxel spacing from shape change.
    
    Args:
        original_spacing: Original voxel spacing (x, y, z) in mm
        original_shape: Original image shape (H, W, D)
        target_shape: Target image shape (H, W, D)
    
    Returns:
        Actual achieved spacing (x, y, z) in mm
    """
    return tuple(
        orig_space * orig_dim / tgt_dim
        for orig_space, orig_dim, tgt_dim
        in zip(original_spacing, original_shape, target_shape)
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
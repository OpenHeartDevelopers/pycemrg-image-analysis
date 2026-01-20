# src/pycemrg_image_analysis/utilities/spatial.py

import logging
import numpy as np
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
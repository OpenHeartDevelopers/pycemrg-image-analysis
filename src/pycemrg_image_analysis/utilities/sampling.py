# src/pycemrg_image_analysis/utilities/sampling.py

from typing import Tuple
import numpy as np

def _pad_if_necessary(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int], # (d, h, w)
    mode: str = 'constant',
    **kwargs,
) -> np.ndarray:
    """Pads a volume if it's smaller than the target shape in any dimension."""
    current_shape = volume.shape
    padding_needed = np.maximum(0, np.array(target_shape) - current_shape)
    
    # Calculate padding for each axis (before, after)
    pad_before = padding_needed // 2
    pad_after = padding_needed - pad_before
    
    if np.any(padding_needed > 0):
        pad_width = tuple(zip(pad_before, pad_after))
        return np.pad(volume, pad_width, mode=mode, **kwargs)
        
    return volume

def extract_center_patch(
    volume: np.ndarray,
    patch_size: Tuple[int, int, int], # (d, h, w)
    pad_mode: str = 'constant',
    **pad_kwargs,
) -> np.ndarray:
    """
    Extracts a patch of a specified size from the center of a volume.

    Assumes a (Z, Y, X) or (depth, height, width) axis convention. If the
    volume is smaller than the patch_size, it will be padded first.

    Args:
        volume: The input NumPy array in (d, h, w) order.
        patch_size: The desired patch size as a tuple (d, h, w).
        pad_mode: The np.pad mode to use if padding is required.
        **pad_kwargs: Keyword arguments for np.pad (e.g., constant_values=0).

    Returns:
        The central patch as a NumPy array.
    """
    padded_volume = _pad_if_necessary(volume, patch_size, mode=pad_mode, **pad_kwargs)
    
    center = np.array(padded_volume.shape) // 2
    half_patch = np.array(patch_size) // 2
    
    start_indices = center - half_patch
    
    d, h, w = patch_size
    z0, y0, x0 = start_indices
    
    return padded_volume[z0 : z0 + d, y0 : y0 + h, x0 : x0 + w]

def extract_random_patch(
    volume: np.ndarray,
    patch_size: Tuple[int, int, int], # (d, h, w)
    rng: np.random.RandomState = None,
    pad_mode: str = 'constant',
    **pad_kwargs,
) -> np.ndarray:
    """
    Extracts a random patch of a specified size from a volume.

    Assumes a (Z, Y, X) or (depth, height, width) axis convention. If the
    volume is smaller than the patch_size, it will be padded first.

    Args:
        volume: The input NumPy array in (d, h, w) order.
        patch_size: The desired patch size as a tuple (d, h, w).
        rng: An optional seeded np.random.RandomState for reproducibility.
        pad_mode: The np.pad mode to use if padding is required.
        **pad_kwargs: Keyword arguments for np.pad (e.g., constant_values=0).

    Returns:
        A randomly selected patch as a NumPy array.
    """
    if rng is None:
        rng = np.random.RandomState()

    padded_volume = _pad_if_necessary(volume, patch_size, mode=pad_mode, **pad_kwargs)
    
    max_indices = np.array(padded_volume.shape) - np.array(patch_size)
    
    # Generate a random start index for each dimension
    start_indices = np.array([
        rng.randint(0, max_d + 1) if max_d > 0 else 0 for max_d in max_indices
    ])
    
    d, h, w = patch_size
    z0, y0, x0 = start_indices
    
    return padded_volume[z0 : z0 + d, y0 : y0 + h, x0 : x0 + w]
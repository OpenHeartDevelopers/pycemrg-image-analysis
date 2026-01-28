# src/pycemrg_image_analysis/utilities/intensity.py

import numpy as np

def clip_intensities(
    volume: np.ndarray, min_val: float, max_val: float
) -> np.ndarray:
    """
    Clamps the intensity values of a volume to a specified min/max range.

    Args:
        volume: The input NumPy array.
        min_val: The minimum value for clipping. Values below this are set to this.
        max_val: The maximum value for clipping. Values above this are set to this.

    Returns:
        A new NumPy array with clipped intensity values.
    """
    return np.clip(volume, min_val, max_val)


def normalize_min_max(volume: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Rescales the dynamic range of an array to [0.0, 1.0].

    Args:
        volume: The input NumPy array.
        epsilon: A small value to prevent division by zero if the volume is flat.

    Returns:
        A new NumPy array with values normalized to the [0.0, 1.0] range.
    """
    vol_min = volume.min()
    vol_max = volume.max()
    
    # Ensure the volume is float to handle division
    volume_float = volume.astype(np.float32)
    
    return (volume_float - vol_min) / (vol_max - vol_min + epsilon)


def normalize_percentile(
    volume: np.ndarray, p_min: float = 1.0, p_max: float = 99.0
) -> np.ndarray:
    """
    Robustly normalizes a volume to [0.0, 1.0] by clipping to percentiles.

    This method is robust to extreme outliers (e.g., metal artifacts).

    Args:
        volume: The input NumPy array.
        p_min: The lower percentile to clip to (0.0 to 100.0).
        p_max: The upper percentile to clip to (0.0 to 100.0).

    Returns:
        A new NumPy array, clipped to percentiles and normalized to [0.0, 1.0].
    """
    if not (0 <= p_min < p_max <= 100):
        raise ValueError("Percentiles must be in the range [0, 100] and p_min < p_max.")
        
    p_min_val, p_max_val = np.percentile(volume, [p_min, p_max])
    
    clipped_volume = clip_intensities(volume, p_min_val, p_max_val)
    
    # Now, normalize the clipped volume
    return normalize_min_max(clipped_volume)
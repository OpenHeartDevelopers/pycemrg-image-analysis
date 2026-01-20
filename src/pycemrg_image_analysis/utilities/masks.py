# src/pycemrg_image_analysis/utilities/masks.py

import logging 
import copy
import numpy as np
import SimpleITK as sitk

from pathlib import Path
from enum import Enum, auto

# This enum is critical for our rule-based logic
class MaskOperationMode(Enum):
    REPLACE = auto()
    REPLACE_EXCEPT = auto()
    REPLACE_ONLY = auto()
    ADD = auto()

def add_masks_replace_except(
    base_array: np.ndarray,
    mask_array: np.ndarray,
    new_mask_value: int,
    except_these_values: list[int],
) -> np.ndarray:
    """
    Applies a mask, replacing all values except those in a specified list.

    Args:
        base_array: The input image array to be modified.
        mask_array: A boolean or integer mask array. Non-zero values indicate
                    the region to apply the new value.
        new_mask_value: The new integer value to assign to the masked region.
        except_these_values: A list of integer values in the base_array that
                             should NOT be overwritten.

    Returns:
        A new NumPy array with the mask applied.
    """
    if base_array.shape != mask_array.shape:
        # A common issue in legacy code, handle it explicitly.
        raise ValueError("Input and mask arrays must have the same shape.")

    output_array = np.copy(base_array)
    
    # Create a boolean array for the mask region and for the exceptions
    mask_region = mask_array != 0
    exception_region = np.isin(output_array, except_these_values)

    # The region to update is where the mask is active AND it's not an exception
    update_region = mask_region & ~exception_region
    
    output_array[update_region] = new_mask_value
    
    return output_array
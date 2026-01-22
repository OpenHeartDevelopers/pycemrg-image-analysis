# src/pycemrg_image_analysis/utilities/masks.py

import logging 
import copy
import numpy as np
import SimpleITK as sitk

from pathlib import Path
from enum import Enum, auto

from pycemrg_image_analysis.logic.constants import ZERO_LABEL

# This enum is critical for our rule-based logic
class MaskOperationMode(Enum):
    REPLACE = auto()
    REPLACE_EXCEPT = auto()
    REPLACE_ONLY = auto()
    ADD = auto()

def add_masks(
    base_array: np.ndarray, 
    mask_array: np.ndarray, 
    new_mask_value: int,
) -> np.ndarray:
    """
    Apply a mask (mask_array) to an image array without overriding any pixels that already belong to the image array.
    Parameters:
        base_array (np.ndarray): The first input image array.
        mask_array (np.ndarray): The second input image array.
        new_mask_value: The value to assign to the pixels in base_array that are not already part of the image array.
    Returns:
        np.ndarray: The resulting image array after applying the mask.
    """
    if base_array.shape != mask_array.shape:
        # A common issue in legacy code, handle it explicitly.
        raise ValueError("Input and mask arrays must have the same shape.")
    
    output_array = np.copy(base_array)

    update_region = (mask_array != ZERO_LABEL) & (base_array == ZERO_LABEL)
    output_array[update_region] = new_mask_value

    return output_array

def add_masks_replace(
    base_array: np.ndarray, 
    mask_array: np.ndarray,
    new_mask_value: int,
) -> np.ndarray:
    
    output_array = np.copy(base_array)

    update_region = mask_array != ZERO_LABEL
    output_array[update_region] = new_mask_value
    
    return output_array

def add_masks_replace_only(
    base_array: np.ndarray, 
    mask_array: np.ndarray,
    new_mask_value: int, 
    only_override_these_values: list[int],
) -> np.ndarray:
    """
        Apply a mask to an image array, replacing the pixels that belong to the mask with a new value.

        Parameters:
            imga_array (np.ndarray): The first input image array.
            imgb_array (np.ndarray): The second input image array.
            new_mask_value: The value to assign to the pixels in imga_array that belong to the mask.
            only_override_this: The value of pixels in imga_array that should be overridden with the new_mask_value.

        Returns:
            np.ndarray: The resulting image array after applying the mask and replacing the pixels.
        """
    if base_array.shape != mask_array.shape:
        # A common issue in legacy code, handle it explicitly.
        raise ValueError("Input and mask arrays must have the same shape.")

    output_array = np.copy(base_array)

    values_to_override = set(only_override_these_values)
    values_to_override.add(0)

    mask_region = mask_array != ZERO_LABEL
    only_region = np.isin(output_array, list(values_to_override))

    update_region = mask_region & only_region

    output_array[update_region] = new_mask_value

    return output_array
    

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
    mask_region = mask_array != ZERO_LABEL
    exception_region = np.isin(output_array, except_these_values)

    # The region to update is where the mask is active AND it's not an exception
    update_region = mask_region & ~exception_region
    
    output_array[update_region] = new_mask_value
    
    return output_array

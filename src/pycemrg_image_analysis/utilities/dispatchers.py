# src/pycemrg_image_analysis/utilities/dispatchers.py

from pycemrg_image_analysis.utilities import masks
from pycemrg_image_analysis.utilities.masks import MaskOperationMode

def get_mask_operation_dispatcher() -> dict:
    """
    Returns a single, shared dispatch dictionary that maps MaskOperationMode
    Enums to their corresponding utility functions in utilities.masks.

    This provides a single point of registration for all mask application modes.
    """
    return {
        MaskOperationMode.REPLACE_EXCEPT: masks.add_masks_replace_except,
        MaskOperationMode.REPLACE_ONLY: masks.add_masks_replace_only,
        MaskOperationMode.REPLACE: masks.add_masks_replace,
        MaskOperationMode.ADD: masks.add_masks,
    }
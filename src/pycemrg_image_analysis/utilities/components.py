# src/pycemrg_image_analysis/utilities/components.py

"""
Connected component analysis utilities for segmentation cleanup.

This module provides spatial operations that require metadata awareness
(spacing, origin, direction). Unlike masks.py which operates on pure arrays,
these functions work with SimpleITK.Image objects to ensure correct handling
of anisotropic voxels and spatial relationships.

Common use cases:
- Remove floating/disconnected components from neural network outputs
- Keep only largest connected structure across multiple labels
- Clean up per-label disconnected blobs
"""

import logging
from typing import List, Optional

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def keep_largest_component(
    image: sitk.Image,
    label_values: List[int]
) -> sitk.Image:
    """
    Keep only the largest connected component for each specified label independently.
    
    Processes each label separately - finds all connected components of that label
    and retains only the largest blob. Useful for cleaning up labels that have
    small disconnected pieces.
    
    Args:
        image: Input segmentation image
        label_values: List of integer label values to process independently
        
    Returns:
        New image with only largest component per label preserved
        
    Example:
        >>> # Label 7 has main aorta + small floating blob
        >>> # Label 9 has main structure + small artifact
        >>> cleaned = keep_largest_component(seg_image, [7, 9])
        >>> # Each label now has only its largest component
        
    Note:
        - Labels not in label_values are completely untouched
        - Each label is processed independently (label 7 blobs don't affect label 9)
        - If a label has only one component, it is unchanged
        - Labels in label_values but not present in image are silently skipped
        
    See also:
        keep_largest_structure() - For treating multiple labels as one structure
    """
    array = sitk.GetArrayFromImage(image)
    output = array.copy()
    
    for label_value in label_values:
        # Skip if label not present
        if label_value not in array:
            logger.debug(f"Label {label_value} not in image, skipping")
            continue
        
        # Extract binary mask for this label
        binary_mask = (array == label_value).astype(np.uint8)
        binary_img = sitk.GetImageFromArray(binary_mask)
        binary_img.CopyInformation(image)
        
        # Connected component labeling
        component_img = sitk.ConnectedComponent(binary_img)
        
        # Relabel by size (largest = 1, second = 2, etc.)
        sorted_cc_img = sitk.RelabelComponent(component_img, sortByObjectSize=True)
        sorted_array = sitk.GetArrayFromImage(sorted_cc_img)
        
        # Keep only largest component (label 1 after sorting)
        largest_mask = sorted_array == 1
        
        # Build output: clear all instances of this label, restore only largest
        output[array == label_value] = 0
        output[largest_mask] = label_value
        
        logger.debug(f"Kept largest component for label {label_value}")
    
    result = sitk.GetImageFromArray(output)
    result.CopyInformation(image)
    
    logger.info(
        f"Processed {len(label_values)} labels for largest component "
        f"(per-label independent)"
    )
    return result


def keep_largest_structure(
    image: sitk.Image,
    label_values: Optional[List[int]] = None
) -> sitk.Image:
    """
    Keep only the largest connected structure across multiple labels.
    
    Treats specified labels as forming ONE anatomical structure. Finds the
    largest connected blob when considering ALL labels together, then removes
    any floating chunks. Preserves all internal label values within the kept
    structure.
    
    This is the solution for neural networks that segment valid anatomy plus
    garbage floating chunks that share the same labels as valid structures.
    
    Args:
        image: Input segmentation image
        label_values: Labels that form the structure. If None (default),
                     uses all non-zero labels in the image.
        
    Returns:
        New image with only largest structure, preserving all internal labels
        
    Example:
        >>> # CardioForm segments LA (4) + PVs (8) + garbage in lungs (also 4, 8)
        >>> cleaned = keep_largest_structure(seg, [4, 8])
        >>> # Keeps main LA+PV structure, removes lung chunks
        >>> # Internal labels 4 and 8 are preserved in the kept structure
        
        >>> # Clean entire segmentation (all labels)
        >>> cleaned = keep_largest_structure(cardioform_seg)
        >>> # Keeps largest multi-label blob, removes all floating chunks
        
    Algorithm:
        1. Create binary mask (any specified label = foreground)
        2. Connected component analysis on binary mask
        3. Identify largest component
        4. Multiply original image by largest component mask
        5. Result: main structure with all labels, floating chunks zeroed
        
    Note:
        - Default (label_values=None) processes ALL non-zero labels
        - Labels not in label_values are completely untouched
        - Single label is valid: keep_largest_structure(img, [7]) works
        - Different from keep_largest_component: treats labels as ONE structure,
          not independent per-label processing
        
    Use case:
        Neural networks often produce spatially disconnected regions with valid
        label values. For example, an aorta segmentation might have the main
        vessel plus small floating blobs in the lungs, all labeled as "aorta".
        This function removes such artifacts while preserving the main anatomy.
        
    See also:
        keep_largest_component() - For independent per-label processing
    """
    array = sitk.GetArrayFromImage(image)
    
    # Determine which labels to consider
    if label_values is None:
        # Use all non-zero labels in the image
        unique_labels = np.unique(array)
        label_values = [int(val) for val in unique_labels if val != 0]
        
        if not label_values:
            logger.warning("No non-zero labels found in image, returning unchanged")
            return image
        
        logger.info(f"No labels specified, using all non-zero labels: {label_values}")
    
    # Step 1: Create binary mask (any of these labels = foreground)
    mask = np.isin(array, label_values).astype(np.uint8)
    
    # Check if any voxels match
    if not np.any(mask):
        logger.warning(
            f"None of the specified labels {label_values} found in image, "
            "returning unchanged"
        )
        return image
    
    # Step 2: Connected component analysis on binary mask
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(image)
    
    conn_img = sitk.ConnectedComponent(mask_img)
    
    # Step 3: Relabel by size (largest = 1)
    sorted_conn = sitk.RelabelComponent(conn_img, sortByObjectSize=True)
    sorted_array = sitk.GetArrayFromImage(sorted_conn)
    
    # Step 4: Keep only largest component
    largest_mask = (sorted_array == 1)
    
    # Step 5: Apply to original (preserves internal labels, preserves unspecified labels)
    # Start with original array
    output = array.copy()
    
    # Zero out specified labels that are NOT in the largest component
    labels_to_clean = np.isin(array, label_values) & ~largest_mask
    output[labels_to_clean] = 0
    
    result = sitk.GetImageFromArray(output)
    result.CopyInformation(image)
    
    num_components = int(sorted_array.max())
    logger.info(
        f"Kept largest structure from {num_components} components "
        f"(multi-label structure: {label_values})"
    )
    
    return result
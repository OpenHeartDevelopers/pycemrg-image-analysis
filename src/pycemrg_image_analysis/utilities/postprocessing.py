# src/pycemrg_image_analysis/utilities/postprocessing.py

"""
Post-processing utilities for segmentation cleanup.

Provides SimpleITK.Image wrappers around array-based label operations,
handling the conversion round-trip and metadata preservation automatically.

Common use cases:
- Remove unwanted labels after workflow execution
- Extract specific structures from multi-label segmentations
- Clean up temporary scaffolding labels
"""

from dataclasses import dataclass
import logging
from typing import List, Optional

import numpy as np
import SimpleITK as sitk
from pycemrg.data import LabelManager

from pycemrg_image_analysis.utilities.masks import (
    remove_label as _remove_label_array,
    remove_labels as _remove_labels_array,
    keep_labels as _keep_labels_array,
)

from pycemrg_image_analysis.utilities.components import (
    keep_largest_component,
    keep_largest_structure,
)

logger = logging.getLogger(__name__)


def remove_label(image: sitk.Image, label: int) -> sitk.Image:
    """
    Remove a single label from image by setting it to background (0).
    
    Args:
        image: Input segmentation image
        label: Integer label value to remove
        
    Returns:
        New image with label removed (set to 0)
        
    Example:
        >>> # Remove left atrium blood pool
        >>> cleaned = remove_label(seg_image, 4)
    """
    array = sitk.GetArrayFromImage(image)
    cleaned_array = _remove_label_array(array, label)
    
    result = sitk.GetImageFromArray(cleaned_array)
    result.CopyInformation(image)  # Preserve spacing, origin, direction
    
    logger.debug(f"Removed label {label} from image")
    return result


def remove_labels(image: sitk.Image, labels: List[int]) -> sitk.Image:
    """
    Remove multiple labels from image by setting them to background (0).
    
    Args:
        image: Input segmentation image
        labels: List of integer label values to remove
        
    Returns:
        New image with labels removed (set to 0)
        
    Example:
        >>> # Remove both atria after biventricular workflow
        >>> cleaned = remove_labels(seg_image, [4, 5])  # LA, RA
    """
    array = sitk.GetArrayFromImage(image)
    cleaned_array = _remove_labels_array(array, labels)
    
    result = sitk.GetImageFromArray(cleaned_array)
    result.CopyInformation(image)
    
    logger.info(f"Removed {len(labels)} labels: {labels}")
    return result


def keep_labels(image: sitk.Image, labels_to_keep: List[int]) -> sitk.Image:
    """
    Keep only specified labels, remove all others (except background).
    
    Inverse of remove_labels - useful for extracting specific structures
    from a multi-label segmentation.
    
    Args:
        image: Input segmentation image
        labels_to_keep: List of integer label values to preserve
        
    Returns:
        New image with only kept labels, rest set to 0
        
    Example:
        >>> # Extract only ventricles from whole heart
        >>> ventricles = keep_labels(seg_image, [1, 2, 3])  # LV_BP, LV_myo, RV_BP
    """
    array = sitk.GetArrayFromImage(image)
    extracted_array = _keep_labels_array(array, labels_to_keep)
    
    result = sitk.GetImageFromArray(extracted_array)
    result.CopyInformation(image)
    
    logger.info(f"Kept {len(labels_to_keep)} labels: {labels_to_keep}")
    return result


def inspect_labels(image: sitk.Image, label_manager: LabelManager) -> dict[int, str]:
    """
    Discover all labels in image with their semantic names.
    
    Provides reverse lookup from integer values to human-readable names,
    helping orchestrators understand what structures are present in the
    segmentation after workflow execution.
    
    Args:
        image: Segmentation image to inspect
        label_manager: LabelManager for name lookup
        
    Returns:
        Dict mapping {label_value: label_name}
        
    Example:
        >>> labels_present = inspect_labels(result_image, label_manager)
        >>> print("Output contains:")
        >>> for value, name in labels_present.items():
        ...     print(f"  {value}: {name}")
        
        Output contains:
          1: LV_BP_label
          2: LV_myo_label
          4: LA_BP_label
          5: RA_BP_label
          ...
    """
    array = sitk.GetArrayFromImage(image)
    unique_labels = np.unique(array).astype(int)
    
    result = {}
    for value in unique_labels:
        if value == 0:  # Skip background
            continue
        
        try:
            name = label_manager.get_name(value)
            result[value] = name
        except KeyError:
            # Label exists in image but not in label_manager
            result[value] = f"unknown_label_{value}"
            logger.warning(f"Label {value} not found in label manager")
    
    logger.debug(f"Found {len(result)} non-zero labels in image")
    return result


def remove_labels_by_name(
    image: sitk.Image,
    label_names: List[str],
    label_manager: LabelManager
) -> sitk.Image:
    """
    Remove labels by semantic name (convenience wrapper).
    
    Looks up integer values from names, then removes them. Allows
    orchestrators to work with semantic names instead of hardcoded integers.
    
    Args:
        image: Input segmentation image
        label_names: List of label names to remove (e.g., ["LA_BP_label", "RA_BP_label"])
        label_manager: LabelManager for name → value lookup
        
    Returns:
        New image with labels removed
        
    Raises:
        KeyError: If a label name is not found in label_manager
        
    Example:
        >>> # Orchestrator works with names, not magic numbers
        >>> cleaned = remove_labels_by_name(
        ...     result_image,
        ...     ["LA_BP_label", "RA_BP_label"],
        ...     label_manager
        ... )
    """
    # Look up integer values from names
    label_values = [label_manager.get_value(name) for name in label_names]
    
    logger.info(f"Removing labels by name: {label_names} → {label_values}")
    
    # Delegate to integer-based removal
    return remove_labels(image, label_values)


def keep_labels_by_name(
    image: sitk.Image,
    label_names: List[str],
    label_manager: LabelManager
) -> sitk.Image:
    """
    Keep only specified labels by semantic name.
    
    Inverse of remove_labels_by_name - extracts structures by name.
    
    Args:
        image: Input segmentation image
        label_names: List of label names to preserve
        label_manager: LabelManager for name → value lookup
        
    Returns:
        New image with only kept labels
        
    Raises:
        KeyError: If a label name is not found in label_manager
        
    Example:
        >>> # Extract ventricles by name
        >>> ventricles = keep_labels_by_name(
        ...     whole_heart_image,
        ...     ["LV_BP_label", "LV_myo_label", "RV_BP_label", "RV_myo_label"],
        ...     label_manager
        ... )
    """
    # Look up integer values from names
    label_values = [label_manager.get_value(name) for name in label_names]
    
    logger.info(f"Keeping labels by name: {label_names} → {label_values}")
    
    # Delegate to integer-based keep
    return keep_labels(image, label_values)

def relabel_image(
    image: sitk.Image,
    label_mapping: dict[int, int]
) -> sitk.Image:
    """
    Relabel image according to mapping, handling conflicts with temporary labels.
    
    Safely swaps labels even when mappings have circular dependencies
    (e.g., 1→2, 2→1) by using temporary labels to avoid overwriting.
    
    Args:
        image: Input segmentation image
        label_mapping: Dict mapping {old_label: new_label}
        
    Returns:
        New image with labels remapped
        
    Example:
        >>> # Swap labels 1 and 2
        >>> mapping = {1: 2, 2: 1}
        >>> result = relabel_image(seg_image, mapping)
        
        >>> # Translate CardioForm to image-analysis standard
        >>> from pycemrg.data import LabelMapper
        >>> mapper = LabelMapper(cardioform_mgr, image_analysis_mgr)
        >>> mapping = mapper.get_source_to_target_mapping()
        >>> translated = relabel_image(cardioform_seg, mapping)
    """
    array = sitk.GetArrayFromImage(image)
    labels_in_image = set(np.unique(array).astype(int))
    
    # Build swap sequence with conflict resolution
    swap_ops = _get_swap_sequence(label_mapping, labels_in_image)
    
    # Apply swaps in order
    for old_label, new_label in swap_ops:
        array[array == old_label] = new_label
        logger.debug(f"Swapped label {old_label} → {new_label}")
    
    result = sitk.GetImageFromArray(array)
    result.CopyInformation(image)
    
    logger.info(f"Relabeled image with {len(label_mapping)} mappings")
    return result


def _get_swap_sequence(
    label_mapping: dict[int, int],
    labels_in_image: set[int]
) -> list[tuple[int, int]]:
    """
    Build ordered swap sequence that handles conflicts with temporary labels.
    
    When a destination label is also a source (e.g., 1→2 and 2→3),
    we first move the conflicting label to a temporary value, then
    perform the intended swaps.
    
    Args:
        label_mapping: Dict of {old_label: new_label}
        labels_in_image: Set of labels present in the image
        
    Returns:
        List of (source, target) swap operations in execution order
        
    Example conflict resolution:
        Input: {1: 2, 2: 1}  # Swap 1 and 2
        
        Without temps: BROKEN (overwrites happen)
        With temps:
          1. Move 2 → temp (e.g., 1000)
          2. Move 1 → 2
          3. Move temp → 1
        Result: Labels correctly swapped
    """
    swap_ops = []
    temp_map = {}
    temp_label_counter = max(labels_in_image) + 1 if labels_in_image else 1000
    
    old_labels = set(label_mapping.keys())
    new_labels = set(label_mapping.values())
    
    # Phase 1: Identify conflicts
    # A conflict exists when a destination label is also a source
    conflicts = new_labels & old_labels
    
    for conflict_label in conflicts:
        # Skip if it's mapping to itself
        if label_mapping.get(conflict_label) == conflict_label:
            continue
        
        # Assign temporary label
        temp_label = temp_label_counter
        temp_label_counter += 1
        temp_map[conflict_label] = temp_label
        
        logger.debug(
            f"Conflict: label {conflict_label} is both source and destination. "
            f"Using temporary {temp_label}"
        )
    
    # Phase 2: Build swap sequence
    # First, move conflicting labels to temps
    for conflict_label, temp_label in temp_map.items():
        swap_ops.append((conflict_label, temp_label))
    
    # Then, perform intended mappings
    for old_label, new_label in label_mapping.items():
        # Skip identity mappings
        if old_label == new_label:
            continue
        
        # If source was moved to temp, use temp as source
        if old_label in temp_map:
            swap_ops.append((temp_map[old_label], new_label))
        else:
            swap_ops.append((old_label, new_label))
    
    return swap_ops


def relabel_image_by_name(
    image: sitk.Image,
    label_mapping: dict[str, str],
    source_manager: LabelManager,
    target_manager: LabelManager
) -> sitk.Image:
    """
    Relabel image using semantic names instead of integers.
    
    Convenience wrapper that translates label names to integers,
    then delegates to relabel_image().
    
    Args:
        image: Input segmentation image
        label_mapping: Dict mapping {old_name: new_name}
        source_manager: LabelManager for source (old) labels
        target_manager: LabelManager for target (new) labels
        
    Returns:
        New image with labels remapped
        
    Example:
        >>> # Translate between naming conventions
        >>> mapping = {
        ...     "LV_myo": "LV_myo_label",
        ...     "LV_bp": "LV_BP_label",
        ... }
        >>> result = relabel_image_by_name(
        ...     cardioform_seg,
        ...     mapping,
        ...     cardioform_mgr,
        ...     image_analysis_mgr
        ... )
    """
    # Translate names to integers
    integer_mapping = {}
    for old_name, new_name in label_mapping.items():
        old_value = source_manager.get_value(old_name)
        new_value = target_manager.get_value(new_name)
        integer_mapping[old_value] = new_value
    
    logger.info(f"Translating {len(label_mapping)} label names to integers")
    return relabel_image(image, integer_mapping)

def keep_largest_component_by_name(
    image: sitk.Image,
    label_names: List[str],
    label_manager: LabelManager
) -> sitk.Image:
    """
    Keep only the largest connected component for specified labels (name-based).
    
    Convenience wrapper that translates semantic label names to integers,
    then delegates to components.keep_largest_component(). Processes each
    label independently.
    
    Args:
        image: Input segmentation image
        label_names: List of label names to process (e.g., ["Ao_BP_label", "LA_BP_label"])
        label_manager: LabelManager for name → value lookup
        
    Returns:
        New image with only largest components preserved per label
        
    Raises:
        KeyError: If a label name is not found in label_manager
        
    Example:
        >>> # Clean up aorta and left atrium independently
        >>> cleaned = keep_largest_component_by_name(
        ...     seg_image,
        ...     ["Ao_BP_label", "LA_BP_label"],
        ...     label_manager
        ... )
        
    See also:
        keep_largest_structure_by_name() - For treating labels as one structure
    """
    
    # Look up integer values from names
    label_values = []
    for name in label_names:
        try:
            value = label_manager.get_value(name)
            label_values.append(value)
        except KeyError:
            logger.warning(f"Label name '{name}' not found in LabelManager, skipping")
            continue
    
    if not label_values:
        logger.warning("No valid label names provided, returning image unchanged")
        return image
    
    logger.info(
        f"Cleaning largest components (per-label): {label_names} → {label_values}"
    )
    
    # Delegate to components.py (integer-based, spatial operation)
    return keep_largest_component(image, label_values)


def keep_largest_structure_by_name(
    image: sitk.Image,
    label_names: Optional[List[str]] = None,
    label_manager: Optional[LabelManager] = None
) -> sitk.Image:
    """
    Keep only the largest connected structure across multiple labels (name-based).
    
    Convenience wrapper that translates semantic label names to integers,
    then delegates to components.keep_largest_structure(). Treats all specified
    labels as forming ONE anatomical structure.
    
    Args:
        image: Input segmentation image
        label_names: List of label names forming the structure. If None,
                    uses all non-zero labels in image.
        label_manager: LabelManager for name → value lookup. Required if
                      label_names is provided.
        
    Returns:
        New image with only largest structure, preserving all internal labels
        
    Raises:
        ValueError: If label_names provided but label_manager is None
        KeyError: If a label name is not found in label_manager
        
    Example:
        >>> # Clean entire CardioForm output (all labels)
        >>> cleaned = keep_largest_structure_by_name(cardioform_seg)
        >>> # Keeps main anatomy, removes floating chunks
        
        >>> # Clean specific multi-label structure
        >>> cleaned = keep_largest_structure_by_name(
        ...     seg_image,
        ...     ["LA_BP_label", "LPV1_label", "RPV1_label", "LPV2_label", "RPV2_label"],
        ...     label_manager
        ... )
        >>> # Treats LA + all PVs as one structure
        
    Use case:
        CardioForm neural network sometimes produces spatially disconnected
        anatomy (floating aorta segments, detached atrial appendages). This
        causes downstream valve creation to fail geometrically. Cleaning
        before refinement prevents silent failures.
        
    See also:
        keep_largest_component_by_name() - For independent per-label processing
    """

    # Handle default case (all labels)
    if label_names is None:
        logger.info("No labels specified, cleaning entire segmentation (all labels)")
        return keep_largest_structure(image, label_values=None)
    
    # Explicit label list requires label_manager
    if label_manager is None:
        raise ValueError(
            "label_manager is required when label_names is specified"
        )
    
    # Look up integer values from names
    label_values = []
    for name in label_names:
        try:
            value = label_manager.get_value(name)
            label_values.append(value)
        except KeyError:
            logger.warning(f"Label name '{name}' not found in LabelManager, skipping")
            continue
    
    if not label_values:
        logger.warning("No valid label names provided, returning image unchanged")
        return image
    
    logger.info(
        f"Cleaning largest structure (multi-label): {label_names} → {label_values}"
    )
    
    # Delegate to components.py (integer-based, spatial operation)
    return keep_largest_structure(image, label_values)

@dataclass
class LabelVolumes:
    """
    Result of a label volume computation.
 
    Attributes:
        by_label: Volume per label value, keyed by integer label.
        total: Sum of all individual label volumes.
        unit: Unit of measurement, either "mL" or "mm3".
    """
    by_label: dict[int, float]
    total: float
    unit: str
 
 
def compute_label_volumes(
    image: sitk.Image,
    label_values: List[int],
    use_mm3: bool = False,
) -> LabelVolumes:
    """
    Compute the physical volume of one or more labels in a segmentation image.
 
    Voxel volume is derived from the image spacing (sx * sy * sz mm³). By
    default results are reported in mL (1 mL = 1 cm³ = 1000 mm³), which is
    the standard clinical unit. Pass use_mm3=True for raw mm³.
 
    Args:
        image: Input segmentation image. Spacing must be set correctly.
        label_values: List of integer label values to measure.
        use_mm3: If True, report volumes in mm³ instead of mL.
 
    Returns:
        LabelVolumes dataclass with per-label volumes and their total.
 
    Raises:
        ValueError: If label_values is empty.
 
    Example:
        >>> volumes = compute_label_volumes(seg_image, [1, 2, 3])
        >>> print(f"LV blood pool: {volumes.by_label[1]:.2f} mL")
        >>> print(f"Total: {volumes.total:.2f} mL")
 
        >>> # With LabelManager for readable logging
        >>> for value, vol in volumes.by_label.items():
        ...     name = label_manager.get_name(value)
        ...     print(f"  {name}: {vol:.2f} {volumes.unit}")
    """
    if not label_values:
        raise ValueError("label_values must not be empty.")
 
    spacing = image.GetSpacing()  # (sx, sy, sz) in mm
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    conversion = 1.0 if use_mm3 else 1.0 / 1000.0
    unit = "mm3" if use_mm3 else "mL"
 
    array = sitk.GetArrayFromImage(image)
 
    by_label: dict[int, float] = {}
    for label in label_values:
        voxel_count = int(np.sum(array == label))
        by_label[label] = voxel_count * voxel_volume_mm3 * conversion
        logger.debug(
            f"Label {label}: {voxel_count} voxels → {by_label[label]:.4f} {unit}"
        )
 
    total = sum(by_label.values())
    logger.info(
        f"Computed volumes for {len(label_values)} labels: "
        f"total={total:.4f} {unit}"
    )
 
    return LabelVolumes(by_label=by_label, total=total, unit=unit)
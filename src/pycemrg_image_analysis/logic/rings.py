# src/pycemrg_image_analysis/logic/rings.py

import SimpleITK as sitk
import numpy as np
import logging

from pycemrg_image_analysis.logic.contracts import RingCreationContract
from pycemrg_image_analysis.utilities import (
    filters,
    get_mask_operation_dispatcher,
    MaskOperationMode,
)

logger = logging.getLogger(__name__)


class RingLogic:
    """
    Provides stateless logic for creating vein rings.
    
    Rings are created by thresholding a distance map around a vein blood pool,
    then trimming to the atrium myocardium boundary.
    """

    def create_from_rule(self, contract: RingCreationContract) -> sitk.Image:
        """
        Generic engine for creating a ring from a rule.
        
        Algorithm:
        1. Get or compute atrium myocardium threshold (lazy evaluation)
        2. Compute distance map from vein label (using reference image)
        3. Threshold at ring_thickness → creates "mitten" around vein
        4. Apply mask with application_steps (handles replace_only rules)
        5. AND with atrium myocardium threshold → trim to myocardium only
        6. Add final ring to input segmentation
        
        Args:
            contract: RingCreationContract with all required data
            
        Returns:
            SimpleITK image with ring added to input segmentation
        """
        lm = contract.label_manager
        rule = contract.rule
        
        # --- Step 1: Get or compute atrium myocardium threshold (lazy) ---
        if contract.atrium_myocardium_threshold is not None:
            logger.debug("Using pre-computed atrium myocardium threshold")
            atrium_myo_thresh = contract.atrium_myocardium_threshold
        else:
            logger.debug("Computing atrium myocardium threshold on-demand")
            atrium_myo_label = lm.get_value(rule.atrium_myocardium_name)
            ring_thickness = contract.parameters[rule.ring_thickness_parameter_name]
            
            # Create threshold zone around atrium myocardium
            atrium_myo_distmap = filters.distance_map(
                contract.input_image, atrium_myo_label, use_image_spacing=True
            )
            atrium_myo_thresh = filters.threshold_filter(
                atrium_myo_distmap, lower=0, upper=ring_thickness, binarise=True
            )
        
        # --- Step 2: Extract rule parameters ---
        source_vein_label = lm.get_value(rule.source_vein_label_name)
        target_ring_label = lm.get_value(rule.target_ring_label_name)
        ring_thickness = contract.parameters[rule.ring_thickness_parameter_name]
        
        logger.info(
            f"Creating ring: {rule.source_vein_label_name} → {rule.target_ring_label_name}"
        )
        logger.debug(f"  Source vein label: {source_vein_label}")
        logger.debug(f"  Target ring label: {target_ring_label}")
        logger.debug(f"  Ring thickness: {ring_thickness}mm")
        
        # --- Step 3: Compute distance map from vein (using reference image) ---
        vein_distmap = filters.distance_map(
            contract.reference_image,  # Use frozen reference, not evolving segmentation
            source_vein_label,
            use_image_spacing=True
        )
        
        # --- Step 4: Threshold to create "mitten" around vein ---
        ring_mitten = filters.threshold_filter(
            vein_distmap, lower=0, upper=ring_thickness, binarise=True
        )
        
        # --- Step 5: Apply ring with application_steps rules ---
        # Convert to numpy for multi-step application
        output_array = sitk.GetArrayFromImage(contract.input_image)
        ring_array = sitk.GetArrayFromImage(ring_mitten)
        
        dispatcher = get_mask_operation_dispatcher()
        
        # If no application steps, use simple ADD
        if not rule.application_steps:
            logger.debug("No application steps - using simple ADD")
            from pycemrg_image_analysis.utilities.masks import add_masks
            output_array = add_masks(output_array, ring_array, target_ring_label)
        else:
            # Apply each step in sequence
            for step_idx, step in enumerate(rule.application_steps):
                operation_func = dispatcher.get(step.mode)
                if not operation_func:
                    raise NotImplementedError(
                        f"Application mode '{step.mode}' is not supported."
                    )
                
                logger.debug(
                    f"  Application step {step_idx + 1}/{len(rule.application_steps)}: "
                    f"{step.mode} with labels {step.rule_label_names}"
                )
                
                # Get label values for this step
                rule_label_values = [
                    lm.get_value(name) for name in step.rule_label_names
                ]
                
                # Apply the operation
                output_array = operation_func(
                    output_array, ring_array, target_ring_label, rule_label_values
                )
        
        # --- Step 6: AND with atrium myocardium threshold (trim to myocardium) ---
        logger.debug("Trimming ring to atrium myocardium boundary")
        
        # Convert output back to SimpleITK for AND operation
        output_img = sitk.GetImageFromArray(output_array)
        output_img.CopyInformation(contract.input_image)
        
        # Extract only the ring label for AND operation
        ring_only = filters.threshold_filter(
            output_img, lower=target_ring_label, upper=target_ring_label, binarise=True
        )
        
        # AND with atrium myocardium threshold
        ring_trimmed = sitk.And(ring_only, atrium_myo_thresh)
        
        # --- Step 7: Add trimmed ring back to segmentation ---
        ring_trimmed_array = sitk.GetArrayFromImage(ring_trimmed)
        
        # Replace the ring region in output with trimmed version
        # First remove untrimmed ring
        output_array[output_array == target_ring_label] = 0
        
        # Add trimmed ring
        output_array[ring_trimmed_array > 0] = target_ring_label
        
        # --- Step 8: Return final result ---
        result_img = sitk.GetImageFromArray(output_array)
        result_img.CopyInformation(contract.input_image)
        
        logger.info(f"Ring creation complete: {rule.target_ring_label_name}")
        
        return result_img
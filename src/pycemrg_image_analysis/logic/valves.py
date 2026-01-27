# src/pycemrg_image_analysis/logic/valves.py

import SimpleITK as sitk
import numpy as np

from pycemrg_image_analysis.logic.contracts import ValveCreationContract
from pycemrg_image_analysis.utilities import (
    filters,
    get_mask_operation_dispatcher, # Import the shared dispatcher
    MaskOperationMode,
)

class ValveLogic:
    """
    Provides stateless logic for creating valve structures via intersection.
    """

    def create_from_rule(
        self, contract: ValveCreationContract
    ) -> sitk.Image:
        """
        The generic engine for creating a valve from an intersection rule.
        """
        # --- Step 1: Parse the Rule ---
        # Translate human-readable names from the rule and parameter dictionary
        # into the concrete integer values and numbers needed for the operation.
        lm = contract.label_manager
        rule = contract.rule
        
        structure_a_value = lm.get_value(rule.structure_a_name)
        structure_b_value = lm.get_value(rule.structure_b_name)
        target_valve_value = lm.get_value(rule.target_valve_name)
        thickness = contract.parameters[rule.intersection_thickness_parameter_name]

        # --- Step 2: Generate the Intersection Mask (The Core Pattern) ---
        # The valve is defined as the region where Structure B is within a
        # certain physical distance (thickness) of Structure A.

        # 2a. Create a "growth zone" around Structure A by calculating a
        #     distance map and thresholding it. The zone includes pixels
        #     both inside and outside Structure A.
        dist_map_a = filters.distance_map(
            contract.input_image, structure_a_value, use_image_spacing=True
        )
        growth_zone = filters.threshold_filter(
            dist_map_a, lower=-thickness, upper=thickness, binarise=True
        )

        # 2b. Create a simple binary mask of Structure B.
        mask_b = filters.threshold_filter(
            contract.input_image, structure_b_value, structure_b_value, binarise=True
        )

        # 2c. The intersection is the boolean AND of the two masks. This leaves
        #     only the part of Structure B that falls within the growth zone of A.
        intersection_mask = sitk.And(growth_zone, mask_b)

        # --- Step 3: Apply the New Valve to the Segmentation ---
        # Use the multi-step rule engine to apply the newly created valve mask
        # back onto the original segmentation array.
        output_array = sitk.GetArrayFromImage(contract.input_image)
        mask_array = sitk.GetArrayFromImage(intersection_mask)
        dispatcher = get_mask_operation_dispatcher()

        for step in rule.application_steps:
            # (Loop logic remains the same as our last implementation)
            operation_func = dispatcher.get(step.mode)
            if not operation_func:
                raise NotImplementedError(f"Application mode '{step.mode}' is not supported.")
            
            rule_label_values = lm.get_values_from_names(step.rule_label_names)
            
            kwargs_for_utility = {}
            if step.mode == MaskOperationMode.REPLACE_EXCEPT:
                kwargs_for_utility['except_these_values'] = rule_label_values
            elif step.mode == MaskOperationMode.REPLACE_ONLY:
                kwargs_for_utility['only_override_these_values'] = rule_label_values
            
            output_array = operation_func(
                base_array=output_array, mask_array=mask_array,
                new_mask_value=target_valve_value, **kwargs_for_utility
            )

        # --- Step 4: Finalize the Output Image ---
        # Convert the final NumPy array back to a SimpleITK image, ensuring
        # it retains the same origin, spacing, and orientation as the input.
        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(contract.input_image)
        return output_image
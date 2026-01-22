# src/pycemrg_image_analysis/logic/myocardium.py

import logging
import numpy as np
import SimpleITK as sitk

from pycemrg.data.labels import LabelManager

from pycemrg_image_analysis.logic.constants import MyocardiumSemanticRole
from pycemrg_image_analysis.logic.contracts import PushStructureContract
from pycemrg_image_analysis.utilities import (
    filters,
    masks,
    MaskOperationMode,
)

class MyocardiumLogic:
    """
    Provides stateless logic for creating myocardial structures using a
    generic, rule-based engine configured by a semantic map.
    """

    # helpers
    def _get_mask_operation_dispatcher(self) -> dict:
        """
        Returns a dispatch dictionary mapping operation modes to utility functions.
        This avoids the ugly if/elif/else block.
        """
        return {
            MaskOperationMode.REPLACE_EXCEPT: masks.add_masks_replace_except,
            MaskOperationMode.REPLACE_ONLY: masks.add_masks_replace_only,
            MaskOperationMode.ADD: masks.add_masks,
            # Future utility functions will be registered here.
        }

    def create_from_semantic_map(
        self,
        input_image: sitk.Image,
        label_manager: LabelManager,
        parameters: dict[str, float],
        semantic_map: dict[MyocardiumSemanticRole, any],
    ) -> sitk.Image:
        """
        The generic engine for creating one piece of myocardium.
        """
        logging.info("1. Get names and parameters from the map")
        source_bp_name = semantic_map[MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME]
        target_myo_name = semantic_map[MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME]
        wt_param_name = semantic_map[MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME]
        
        # Get the list of application step dictionaries
        application_steps_data = semantic_map[MyocardiumSemanticRole.APPLICATION_STEPS]

        logging.info("2. Translate names to values")
        source_bp_value = label_manager.get_value(source_bp_name)
        target_myo_value = label_manager.get_value(target_myo_name)
        wall_thickness = parameters[wt_param_name]

        logging.info("3. Create the initial wall mask")
        dist_map = filters.distance_map(
            input_image, source_bp_value, use_image_spacing=True
        )
        new_wall_mask = filters.threshold_filter(
            dist_map, lower=0, upper=wall_thickness, binarise=True
        )

        logging.info("4. Apply the sequence of application steps")
        output_array = sitk.GetArrayFromImage(input_image)
        mask_array = sitk.GetArrayFromImage(new_wall_mask)
        dispatcher = self._get_mask_operation_dispatcher()
        
        # The engine now iterates through the list of steps from the SEMANTIC MAP.
        for step_data in application_steps_data:
            # Parse the current step's data
            mode_enum = MaskOperationMode[step_data["MODE"]]
            rule_label_names = step_data["RULE_LABEL_NAMES"]
            
            operation_func = dispatcher.get(mode_enum)
            if not operation_func:
                raise NotImplementedError(f"Application mode '{mode_enum}' is not supported.")
            
            rule_label_values = label_manager.get_values_from_names(rule_label_names)
            
            # This is a cleaner, more readable way to handle the different
            # keyword arguments required by the utility functions.
            kwargs_for_utility = {}
            if mode_enum == MaskOperationMode.REPLACE_EXCEPT:
                kwargs_for_utility['except_these_values'] = rule_label_values
            elif mode_enum == MaskOperationMode.REPLACE_ONLY:
                kwargs_for_utility['only_override_these_values'] = rule_label_values
            
            # The output of one step becomes the input to the next.
            output_array = operation_func(
                base_array=output_array,
                mask_array=mask_array,
                new_mask_value=target_myo_value,
                **kwargs_for_utility # sends the correct kwarg based on mode 
            )

        logging.info("5. Convert back to a SimpleITK image")
        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(input_image)
        return output_image
    
    def push_structure(
        self, input_image: sitk.Image, contract: PushStructureContract
    ) -> sitk.Image:
        """Refactors the legacy 'push_inside' workflow."""
        # This logic is a direct, stateless translation of ImageAnalysis.push_inside
        dist_map = filters.distance_map(
            input_image, contract.pusher_wall_label, use_image_spacing=True
        )
        # The threshold now correctly creates a shell
        new_pushed_wall_mask = filters.threshold_filter(
            dist_map, lower=0, upper=contract.pushed_wall_thickness, binarise=True
        )

        corrected_mask = filters.and_filter(
            image=input_image,
            mask=new_pushed_wall_mask,
            label_to_query=contract.pushed_bp_label,
            new_label_value=contract.pushed_wall_label,
        )
        
        img_array = sitk.GetArrayFromImage(input_image)

        output_array = masks.add_masks_replace(
            base_arryay=img_array, 
            mask_array=corrected_mask,
            new_mask_value=contract.pushed_wall_label
        )

        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(input_image)
        return output_image


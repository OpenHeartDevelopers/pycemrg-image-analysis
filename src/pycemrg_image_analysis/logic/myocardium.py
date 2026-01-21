# src/pycemrg_image_analysis/logic/myocardium.py

import logging

import SimpleITK as sitk
import numpy as np
from dataclasses import replace

from pycemrg_image_analysis.logic.contracts import (
    MyocardiumCreationContract,
    MyocardiumRule,
)
from pycemrg_image_analysis.utilities import (
    filters,
    masks,
    MaskOperationMode,
)


class MyocardiumLogic:
    """
    Provides stateless logic for creating myocardial structures.

    This class uses a generic, rule-based engine to execute the common
    workflow of distance mapping, thresholding, and applying a new mask.
    """

    def _create_myocardium_from_rule(
        self, contract: MyocardiumCreationContract
    ) -> sitk.Image:
        """The generic engine for creating any piece of myocardium."""
        logging.info("1. Translate names to concrete values")
        lm = contract.label_manager
        rule = contract.rule
        
        source_bp_value = lm.get_value(rule.source_bp_label_name)
        target_myo_value = lm.get_value(rule.target_myo_label_name)
        wall_thickness = contract.parameters[rule.wall_thickness_parameter_name]
        
        rule_label_values = lm.get_values_from_names(
            rule.application_rule_label_names
        )

        logging.info("2. Execute the core pattern: Distance Map -> Threshold")
        dist_map = filters.distance_map(contract.input_image, source_bp_value)
        # We binarise the threshold to get a clean mask of the new wall
        new_wall_mask = filters.threshold_filter(
            dist_map, lower=0, upper=wall_thickness, binarise=True
        )

        logging.info("3. Apply the new wall back to the main segmentation")
        # Convert images to arrays to perform mask application
        base_array = sitk.GetArrayFromImage(contract.input_image)
        mask_array = sitk.GetArrayFromImage(new_wall_mask)

        # Dispatch to the correct mask utility based on the rule's mode
        if rule.application_mode == MaskOperationMode.REPLACE_EXCEPT:
            output_array = masks.add_masks_replace_except(
                base_array=base_array,
                mask_array=mask_array,
                new_mask_value=target_myo_value,
                except_these_values=rule_label_values,
            )
        # Future application modes will be added here
        # elif rule.application_mode == MaskOperationMode.REPLACE_ONLY:
        #     ...
        else:
            raise NotImplementedError(
                f"Application mode '{rule.application_mode}' is not yet implemented."
            )

        # Convert final array back to a SimpleITK image, preserving metadata
        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(contract.input_image)
        
        return output_image

    def create_aortic_wall(
        self, contract: MyocardiumCreationContract
    ) -> sitk.Image:
        """
        Public convenience method to create the Aortic Wall.

        This method defines the specific scientific rules for the aortic wall
        and calls the generic engine to execute them.
        """
        # The "devil in the details" is defined explicitly and readably here.
        aortic_wall_rule = MyocardiumRule(
            source_bp_label_name="Ao_BP_label",
            target_myo_label_name="Ao_wall_label",
            wall_thickness_parameter_name="Ao_WT",
            application_mode=MaskOperationMode.REPLACE_EXCEPT,
            application_rule_label_names=["LV_BP_label", "LV_myo_label"],
        )
        
        # Create a new contract with this specific rule and pass to the engine
        aortic_wall_contract = replace(contract, rule=aortic_wall_rule)
        
        return self._create_myocardium_from_rule(aortic_wall_contract)
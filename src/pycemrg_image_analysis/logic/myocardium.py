# src/pycemrg_image_analysis/logic/myocardium.py

import logging 
import SimpleITK as sitk

from pycemrg.data.labels import LabelManager

from pycemrg_image_analysis.logic.constants import MyocardiumSemanticRole
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

    def create_from_semantic_map(
        self,
        input_image: sitk.Image,
        label_manager: LabelManager,
        parameters: dict[str, float],
        semantic_map: dict[MyocardiumSemanticRole, any],
    ) -> sitk.Image:
        """
        The generic engine for creating one piece of myocardium.

        Args:
            input_image: The initial segmentation image.
            label_manager: An initialized LabelManager for name-to-value translation.
            parameters: A dictionary of scientific parameters (e.g., wall thicknesses).
            semantic_map: A dictionary mapping the library's required semantic roles
                          to the user's specific names and rules for this task.

        Returns:
            A new SimpleITK image with the specified myocardium added.
        """
        logging.info("1. Use the map to get the user's custom names and rules")
        source_bp_name = semantic_map[MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME]
        target_myo_name = semantic_map[MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME]
        wt_param_name = semantic_map[MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME]
        app_mode_str = semantic_map[MyocardiumSemanticRole.APPLICATION_MODE]
        app_rule_names = semantic_map[MyocardiumSemanticRole.APPLICATION_RULE_LABEL_NAMES]

        logging.info("2. Translate names and rules to concrete values")
        source_bp_value = label_manager.get_value(source_bp_name)
        target_myo_value = label_manager.get_value(target_myo_name)
        wall_thickness = parameters[wt_param_name]
        application_mode = MaskOperationMode[app_mode_str]
        rule_label_values = label_manager.get_values_from_names(app_rule_names)

        logging.info("3. Execute the core pattern: Distance Map -> Threshold")
        dist_map = filters.distance_map(input_image, source_bp_value)
        new_wall_mask = filters.threshold_filter(
            dist_map, lower=0, upper=wall_thickness, binarise=True
        )

        logging.info("4. Apply the new wall back to the main segmentation")
        base_array = sitk.GetArrayFromImage(input_image)
        mask_array = sitk.GetArrayFromImage(new_wall_mask)

        if application_mode == MaskOperationMode.REPLACE_EXCEPT:
            output_array = masks.add_masks_replace_except(
                base_array=base_array,
                mask_array=mask_array,
                new_mask_value=target_myo_value,
                except_these_values=rule_label_values,
            )
        else:
            raise NotImplementedError(
                f"Application mode '{application_mode}' is not yet implemented."
            )

        logging.info("5. Convert back to a SimpleITK image")
        output_image = sitk.GetImageFromArray(output_array)
        output_image.CopyInformation(input_image)
        return output_image
import pytest
import json
import logging
from pathlib import Path
import SimpleITK as sitk
import numpy as np

from pycemrg.core import setup_logging
from pycemrg.data.labels import LabelManager
from pycemrg_image_analysis import ImageAnalysisScaffolder
from pycemrg_image_analysis.logic import (
    ValveLogic,
    ValveSemanticRole,
    ValveCreationContract,
    ValveRule,           # We need to import the Rule and ApplicationStep
    ApplicationStep,     # to be able to build them.
)
from pycemrg_image_analysis.utilities import save_image, load_image, MaskOperationMode


def test_all_valve_creation(tmp_path: Path, test_data_root: Path):
    ## Logging
    log_file = tmp_path / "run.log"
    setup_logging(logging.DEBUG, log_file=log_file)

    config_dir = tmp_path / "valve_config"
    output_dir = tmp_path / "valve_outputs"
    output_dir.mkdir()
    
    logging.info("1. Scaffold all components at once")
    valve_components = ["mitral_valve", "aortic_valve", "pulmonary_valve", "tricuspid_valve"]

    scaffolder = ImageAnalysisScaffolder()
    scaffolder.scaffold_components(config_dir, valve_components)
    
    logging.info("2. Load all configurations and initialize tools")
    labels_path = config_dir / "labels.yaml"
    params_path = config_dir / "parameters.json"
    semantic_map_dir = config_dir / "semantic_maps"
    input_seg_path = test_data_root / "myocardium_test" / "seg_input.nrrd"
    assert input_seg_path.exists()

    label_manager = LabelManager(config_path=labels_path)
    valve_logic = ValveLogic()
    with open(params_path, "r") as f:
        parameters = json.load(f)

    initial_image = load_image(input_seg_path)
    # We will start with a copy of the input array and add each valve to it.
    final_array = sitk.GetArrayFromImage(initial_image)

    logging.info("3. Execute the workflow for each valve component")
    for i, step_name in enumerate(valve_components): 
        logging.info(f'---- Running step {i+1}/{len(valve_components)}: {step_name}')
        map_path = semantic_map_dir / f'{step_name}.json' 
        
        with open(map_path, 'r') as f: 
            raw_map = json.load(f)
            # This is still just a dictionary
            semantic_map = {ValveSemanticRole[k]: v for k, v in raw_map.items()}

        # --- THIS IS THE KEY STEP: Building the Rule Object ---
        # The orchestrator's job is to parse the generic dictionary from the JSON
        # and construct the strongly-typed dataclass object that the library expects.
        
        # 3a. Parse the list of application steps from the map
        application_steps = [
            ApplicationStep(
                mode=MaskOperationMode[step_data["MODE"]],
                rule_label_names=step_data["RULE_LABEL_NAMES"]
            )
            for step_data in semantic_map[ValveSemanticRole.APPLICATION_STEPS]
        ]
        
        # 3b. Create the ValveRule object
        valve_rule = ValveRule(
            structure_a_name=semantic_map[ValveSemanticRole.STRUCTURE_A_NAME],
            structure_b_name=semantic_map[ValveSemanticRole.STRUCTURE_B_NAME],
            target_valve_name=semantic_map[ValveSemanticRole.TARGET_VALVE_NAME],
            intersection_thickness_parameter_name=semantic_map[ValveSemanticRole.INTERSECTION_THICKNESS_PARAMETER_NAME],
            application_steps=application_steps
        )

        # 3c. Create the final contract, now including the fully-built rule
        contract = ValveCreationContract(
            input_image=initial_image, # Always operate on the same initial image
            label_manager=label_manager, 
            parameters=parameters,
            output_path=output_dir / f"{step_name}.nrrd", # Each step has its own output path
            rule=valve_rule # Pass the constructed rule object
        )

        # 4. Call the logic engine and get the result
        result_image_with_valve = valve_logic.create_from_rule(contract)
        
        # 5. Merge the new valve into our final composite array
        result_array = sitk.GetArrayFromImage(result_image_with_valve)
        target_valve_value = label_manager.get_value(valve_rule.target_valve_name)
        # Find where the new valve is and update our final array
        final_array[result_array == target_valve_value] = target_valve_value
        
        # 6. Save intermediate step for debugging and assert
        save_image(result_image_with_valve, contract.output_path)
        assert contract.output_path.exists()
        assert target_valve_value in result_array

    # --- FINAL ASSERTION ---
    logging.info("Workflow complete. Saving final composite image.")
    final_image = sitk.GetImageFromArray(final_array)
    final_image.CopyInformation(initial_image)
    final_output_path = output_dir / "seg_with_all_valves.nrrd"
    save_image(final_image, final_output_path)

    assert final_output_path.exists() 
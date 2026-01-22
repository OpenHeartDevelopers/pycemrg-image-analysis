# tests/integration/test_myocardium_no_cuts.py

import pytest
import json
from pathlib import Path
import SimpleITK as sitk

from pycemrg.data.labels import LabelManager
from pycemrg_image_analysis import ImageAnalysisScaffolder
from pycemrg_image_analysis.logic import MyocardiumLogic, MyocardiumSemanticRole
from pycemrg_image_analysis.utilities import save_image, load_image

from pycemrg_image_analysis.logic.contracts import PushStructureContract

FULL_MYOCARDIUM_WORKFLOW = [
    {"name": "lv_outflow",       "type": "create"},
    {"name": "aortic_wall",      "type": "create"},
    {"name": "pulmonary_artery", "type": "create"},
    {"name": "pa_push_aorta",    "type": "push"},
    {"name": "rv_myocardium",    "type": "create"},
    {"name": "la_myocardium",    "type": "create"},
    {"name": "ra_myocardium",    "type": "create"},
    # missing pushing in RA, LA, PArt, RV 
]

def test_full_myocardium_workflow(test_data_root: Path, tmp_path: Path):
    """
    Integration test for the complete myocardium creation sequence,
    mimicking the legacy 'create_myocardium_refact' script.
    """
    # --- 1. ARRANGE: Scaffold all components needed for the workflow ---
    scaffolder = ImageAnalysisScaffolder()
    config_dir = tmp_path / "full_myo_config"
    output_dir = tmp_path / "full_myo_output"
    myocardium_step_names = [
        "lv_outflow", 
        "aortic_wall", 
        "pulmonary_artery", 
        "rv_myocardium",
        "la_myocardium", 
        "ra_myocardium"
    ]

    scaffolder.scaffold_components(
        output_dir=config_dir, 
        component_names=myocardium_step_names,
    )
    
    # --- 2. ARRANGE: Load all configs and initialize tools ---
    labels_path = config_dir / "labels.yaml"
    params_path = config_dir / "parameters.json"
    semantic_map_dir = config_dir / "semantic_maps" 

    # --- 3. ACT: Execute the workflow step-by-step ---
    initial_seg_path = test_data_root / "myocardium_test" / "seg_input.nrrd"
    
    assert initial_seg_path.exists()

    working_image = load_image(initial_seg_path)

    assert all([p.exists() for p in [labels_path, params_path]]), "Required config or input files are missing."

    label_manager = LabelManager(config_path=labels_path)
    myo_logic = MyocardiumLogic()

    # Load the generated files into memory
    with open(params_path, 'r') as f:
        parameters = json.load(f)

    for ix, step_name in enumerate(myocardium_step_names):
        
        semantic_map_path = semantic_map_dir / f"{step_name}.json"
        assert semantic_map_path.exists(), f"Semantic map for '{step_name}' is missing."

        with open(semantic_map_path, 'r') as f:
            raw_map = json.load(f)
            semantic_map = {MyocardiumSemanticRole[key]: value for key, value in raw_map.items()}

        # Instantiate library components

        working_image = myo_logic.create_from_semantic_map(
            input_image=working_image,
            label_manager=label_manager,
            parameters=parameters,
            semantic_map=semantic_map
        )

        # Save intermediate result for next step
        intermediate_path = output_dir / f"step{ix:02}_{step_name}.nrrd"
        save_image(working_image, intermediate_path)

        assert intermediate_path.exists(), f"Output image for step '{step_name}' was not created."

        result_array = sitk.GetArrayViewFromImage(working_image)
        target_myo_value = label_manager.get_value(semantic_map[MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME])

        assert target_myo_value in result_array, f"Target myocardium label not found in output of step '{step_name}'."
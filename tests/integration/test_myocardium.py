# tests/integration/test_myocardium.py

import json
from pathlib import Path
import SimpleITK as sitk

from pycemrg.data.labels import LabelManager
from pycemrg_image_analysis import ImageAnalysisScaffolder
from pycemrg_image_analysis.logic import MyocardiumLogic, MyocardiumSemanticRole
from pycemrg_image_analysis.utilities import save_image, load_image

# --- Test Data Setup ---
# This test requires the following files to exist in your test data directory:
#
# 1. myocardium_test/labels.yaml:
#    labels:
#      LV_BP_label: 1
#      LV_myo_label: 2
#      Ao_BP_label: 6
#      Ao_wall_label: 106
#
# 2. myocardium_test/parameters.json:
#    {
#      "Ao_WT": 2.5
#    }
#
# 3. myocardium_test/aortic_wall_map.json:
#    {
#      "SOURCE_BLOOD_POOL_NAME": "Ao_BP_label",
#      "TARGET_MYOCARDIUM_NAME": "Ao_wall_label",
#      "WALL_THICKNESS_PARAMETER_NAME": "Ao_WT",
#      "APPLICATION_MODE": "REPLACE_EXCEPT",
#      "APPLICATION_RULE_LABEL_NAMES": ["LV_BP_label", "LV_myo_label"]
#    }
#
# 4. myocardium_test/seg_input.nrrd:
#    A simple segmentation file to act as the input.

def test_create_aortic_wall_from_map(test_data_root: Path, tmp_path: Path):
    """
    Integration test demonstrating the semantic map workflow for myocardium creation.
    """
    # --- 1. ARRANGE: The Orchestrator loads ALL configuration into memory ---
    test_dir = test_data_root / "myocardium_test"
    output_dir = tmp_path / "test_myocardium"

    # The orchestrator is responsible for loading all files.
    labels_path = test_dir / "labels.yaml"
    params_path = test_dir / "parameters.json"
    semantic_map_path = test_dir / "aortic_wall_map.json"
    input_seg_path = test_dir / "seg_input.nrrd"
    
    # Pre-condition checks
    assert all([p.exists() for p in [labels_path, params_path, semantic_map_path, input_seg_path]])

    # Load into memory
    with open(params_path, 'r') as f:
        parameters = json.load(f)
        
    with open(semantic_map_path, 'r') as f:
        # We need to convert the string keys back to the Enum members
        raw_map = json.load(f)
        semantic_map = {MyocardiumSemanticRole[key]: value for key, value in raw_map.items()}

    # Instantiate library components
    label_manager = LabelManager(config_path=labels_path)
    myo_logic = MyocardiumLogic()
    input_image = load_image(input_seg_path)

    # --- 2. ACT: The Orchestrator calls the logic with pure, in-memory data ---
    output_image = myo_logic.create_from_semantic_map(
        input_image=input_image,
        label_manager=label_manager,
        parameters=parameters,
        semantic_map=semantic_map
    )

    # --- 3. ASSERT: The Orchestrator saves and verifies the output ---
    output_path = output_dir / "seg_with_aortic_wall.nrrd"
    save_image(output_image, output_path)

    assert output_path.exists()
    # A more advanced test could check that the Ao_wall_label now exists in the image
    result_array = sitk.GetArrayViewFromImage(output_image)
    ao_wall_value = label_manager.get_value("Ao_wall_label")
    assert ao_wall_value in result_array

def test_create_rv_myocardium_from_map(test_data_root: Path, tmp_path: Path):
    """
    Integration test for the RV myocardium component.
    It leverages the scaffolder to generate the necessary configuration.
    """
    # --- 1. ARRANGE: Use the scaffolder to generate the entire config ---
    # This is the payoff. A single call creates all necessary files.
    output_dir = tmp_path / "test_myocardium"
    scaffolder = ImageAnalysisScaffolder()
    config_dir = output_dir / "rv_myo_config"
    scaffolder.scaffold_components(
        output_dir=config_dir,
        component_names=["rv_myocardium"] # Specify the component we need
    )

    # --- 2. ARRANGE: The Orchestrator loads the generated configuration ---
    # Define paths to the newly created files
    labels_path = config_dir / "labels.yaml"
    params_path = config_dir / "parameters.json"
    semantic_map_path = config_dir / "semantic_maps" / "rv_myocardium.json"
    
    # We still need a starting input image from our main test data
    input_seg_path = test_data_root / "myocardium_test" / "seg_input.nrrd"
    
    assert all([p.exists() for p in [labels_path, params_path, semantic_map_path, input_seg_path]])

    # Load the generated files into memory
    with open(params_path, 'r') as f:
        parameters = json.load(f)
        
    with open(semantic_map_path, 'r') as f:
        raw_map = json.load(f)
        semantic_map = {MyocardiumSemanticRole[key]: value for key, value in raw_map.items()}

    # Instantiate library components
    label_manager = LabelManager(config_path=labels_path)
    myo_logic = MyocardiumLogic()
    input_image = load_image(input_seg_path)

    # --- 3. ACT: Call the SAME generic logic engine with the new rule ---
    output_image = myo_logic.create_from_semantic_map(
        input_image=input_image,
        label_manager=label_manager,
        parameters=parameters,
        semantic_map=semantic_map
    )

    # --- 4. ASSERT: The Orchestrator saves and verifies the output ---
    output_path = output_dir / "seg_with_rv_myocardium.nrrd"
    save_image(output_image, output_path)

    assert output_path.exists(), "Output file was not created"
    
    # Verify that the new label now exists in the output image
    result_array = sitk.GetArrayViewFromImage(output_image)
    rv_myo_value = label_manager.get_value("RV_myo_label")
    assert rv_myo_value in result_array, "RV myocardium label was not found in the output image"

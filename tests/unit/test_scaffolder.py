import pytest
import json
import yaml
from pathlib import Path

# Import the class we are testing
from pycemrg_image_analysis import ImageAnalysisScaffolder
from pycemrg_image_analysis.logic.constants import MyocardiumSemanticRole


def test_scaffold_components_single_component(tmp_path: Path):
    """
    Unit test for ImageAnalysisScaffolder.
    Verifies that it correctly creates all necessary configuration files for a
    single, known component ('aortic_wall').
    """
    # --- 1. ARRANGE ---
    # Instantiate the scaffolder and define the test parameters
    scaffolder = ImageAnalysisScaffolder()
    output_dir = tmp_path / "scaffold_output"
    component_to_scaffold = ["aortic_wall"]

    # --- 2. ACT ---
    # Run the method we are testing
    scaffolder.scaffold_components(
        output_dir=output_dir,
        component_names=component_to_scaffold
    )

    # --- 3. ASSERT ---
    # Verify that all expected files were created in the correct locations
    labels_path = output_dir / "labels.yaml"
    params_path = output_dir / "parameters.json"
    maps_dir = output_dir / "semantic_maps"
    aortic_wall_map_path = maps_dir / "aortic_wall.json"

    assert labels_path.exists(), "labels.yaml was not created"
    assert params_path.exists(), "parameters.json was not created"
    assert aortic_wall_map_path.exists(), "aortic_wall.json semantic map was not created"

    # Verify the content of the created files
    with open(labels_path, 'r') as f:
        labels_data = yaml.safe_load(f)
        # Check for a key known to be in the aortic_wall schematic
        assert "Ao_wall_label" in labels_data["labels"]
        assert labels_data["labels"]["Ao_wall_label"] == 106

    with open(params_path, 'r') as f:
        params_data = json.load(f)
        assert "Ao_WT" in params_data
        assert params_data["Ao_WT"] == 2.0

    with open(aortic_wall_map_path, 'r') as f:
        map_data = json.load(f)
        # Check that the Enum member was correctly converted to a string key
        assert map_data[MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME.name] == "Ao_BP_label"
        
        assert MyocardiumSemanticRole.APPLICATION_STEPS.name in map_data
        
        application_steps = map_data[MyocardiumSemanticRole.APPLICATION_STEPS.name]
        assert isinstance(application_steps, list)
        assert application_steps[0]["MODE"] == "REPLACE_EXCEPT"


def test_scaffold_raises_error_for_unknown_component(tmp_path: Path):
    """
    Unit test to ensure the scaffolder raises a ValueError for an unknown component.
    """
    scaffolder = ImageAnalysisScaffolder()
    
    with pytest.raises(ValueError, match="Unknown component schematic requested: 'unknown_part'"):
        scaffolder.scaffold_components(
            output_dir=tmp_path,
            component_names=["unknown_part"]
        )
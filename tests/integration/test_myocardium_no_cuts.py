import pytest
import json
import logging
from pathlib import Path
import SimpleITK as sitk

from pycemrg.core import setup_logging
from pycemrg.data.labels import LabelManager
from pycemrg_image_analysis import ImageAnalysisScaffolder
from pycemrg_image_analysis.logic import MyocardiumLogic, MyocardiumSemanticRole
from pycemrg_image_analysis.utilities import save_image, load_image
from pycemrg_image_analysis.logic.contracts import PushStructureContract


def test_full_myocardium_workflow(test_data_root: Path, tmp_path: Path):
    """
    Integration test for the complete myocardium creation sequence,
    excluding artery cuts and the complex RA push-in with SVC ring fix.
    """
    ## Logging
    log_file = tmp_path / "run.log"
    setup_logging(logging.DEBUG, log_file=log_file)

    logging.info("1. Define the full workflow sequence")
    # This list is the "master recipe" for the orchestrator.
    # Each dictionary defines a step's type and its unique name.
    WORKFLOW_STEPS = [
        {"type": "create", "name": "lv_outflow"},
        {"type": "create", "name": "aortic_wall"},
        {"type": "create", "name": "pulmonary_artery"},
        {"type": "push", "name": "part_push_aorta"},  # First push step
        {"type": "push", "name": "part_push_lv"},  # Second push step for PA
        {"type": "create", "name": "rv_myocardium"},
        {"type": "create", "name": "la_myocardium"},
        {"type": "push", "name": "la_push_aorta"},  # Push step for LA
        {"type": "create", "name": "ra_myocardium"},
        {"type": "push", "name": "rv_push_aorta"},  # Final push step
    ]

    logging.info("2. Scaffold all necessary component configs")
    scaffolder = ImageAnalysisScaffolder()
    config_dir = tmp_path / "full_myo_config"
    # Get the unique names of all 'create' steps for the scaffolder
    component_names = [
        step["name"] for step in WORKFLOW_STEPS if step["type"] == "create"
    ]
    component_names.append("myo_push_steps")  # Add the push step params
    scaffolder.scaffold_components(config_dir, component_names)

    logging.info("3. Load all configs and initialize tools")
    labels_path = config_dir / "labels.yaml"
    params_path = config_dir / "parameters.json"
    semantic_map_dir = config_dir / "semantic_maps"
    input_seg_path = test_data_root / "myocardium_test" / "seg_input.nrrd"
    assert input_seg_path.exists()

    label_manager = LabelManager(config_path=labels_path)
    myo_logic = MyocardiumLogic()
    with open(params_path, "r") as f:
        parameters = json.load(f)

    logging.info("4. ACT: Execute the workflow step-by-step")
    working_image = load_image(input_seg_path)

    for i, step in enumerate(WORKFLOW_STEPS):
        step_name = step["name"]
        step_type = step["type"]
        logging.info(
            f"--- Running Step {i + 1}/{len(WORKFLOW_STEPS)}: {step_name} ({step_type}) ---"
        )

        if step_type == "create":
            map_path = semantic_map_dir / f"{step_name}.json"
            with open(map_path, "r") as f:
                raw_map = json.load(f)
                semantic_map = {
                    MyocardiumSemanticRole[k]: v for k, v in raw_map.items()
                }

            working_image = myo_logic.create_from_semantic_map(
                working_image, label_manager, parameters, semantic_map
            )

        elif step_type == "push":
            # Here we define the contracts for each push step
            contract = None
            if step_name == "part_push_aorta":
                contract = PushStructureContract(
                    pusher_wall_label=label_manager.get_value("Ao_wall_label"),
                    pushed_wall_label=label_manager.get_value("PArt_wall_label"),
                    pushed_bp_label=label_manager.get_value("PArt_BP_label"),
                    pushed_wall_thickness=parameters["PArt_WT"],
                )
            elif step_name == "part_push_lv":
                contract = PushStructureContract(
                    pusher_wall_label=label_manager.get_value("LV_myo_label"),
                    pushed_wall_label=label_manager.get_value("PArt_wall_label"),
                    pushed_bp_label=label_manager.get_value("PArt_BP_label"),
                    pushed_wall_thickness=parameters["PArt_WT"],
                )
            elif step_name == "la_push_aorta":
                contract = PushStructureContract(
                    pusher_wall_label=label_manager.get_value("Ao_wall_label"),
                    pushed_wall_label=label_manager.get_value("LA_myo_label"),
                    pushed_bp_label=label_manager.get_value("LA_BP_label"),
                    pushed_wall_thickness=parameters["LA_WT"],
                )
            elif step_name == "rv_push_aorta":
                contract = PushStructureContract(
                    pusher_wall_label=label_manager.get_value("Ao_wall_label"),
                    pushed_wall_label=label_manager.get_value("RV_myo_label"),
                    pushed_bp_label=label_manager.get_value("RV_BP_label"),
                    pushed_wall_thickness=parameters["RV_WT"],
                )

            if contract:
                working_image = myo_logic.push_structure(working_image, contract)
            else:
                pytest.fail(f"Push step '{step_name}' is not defined in the test.")

        # Save and assert after each step
        intermediate_path = tmp_path / f"step_{i:02d}_{step_name}.nrrd"
        save_image(working_image, intermediate_path)
        assert intermediate_path.exists()

    logging.info("5. ASSERT (Final)")
    print("\n--- Workflow Complete ---")
    final_output_path = tmp_path / "seg_final_myocardium.nrrd"
    save_image(working_image, final_output_path)
    assert final_output_path.exists()

    # Final check: ensure the last created label is present
    last_myo_value = label_manager.get_value("RV_myo_label")
    result_array = sitk.GetArrayViewFromImage(working_image)
    assert last_myo_value in result_array

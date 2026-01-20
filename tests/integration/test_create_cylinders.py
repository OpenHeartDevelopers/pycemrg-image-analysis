# tests/integration/test_create_cylinders.py

import json
import logging
import numpy as np
from pathlib import Path

# --- Library Imports ---
# Notice we only import from the public-facing API defined in the __init__.py files.
# This makes the interaction clean and predictable.
from pycemrg_image_analysis.logic import SegmentationLogic, SegmentationPathBuilder
from pycemrg_image_analysis.utilities import save_image, load_image

# (Assuming a logging setup function exists in a core library)
from pycemrg.core import setup_logging


def test_create_cylinders(test_data_root: Path, tmp_path: Path):
    """
    Integration test demonstrating the canonical workflow for cylinder creation.

    This script serves as a template for an ORCHESTRATOR. It shows how to:
    1.  Manage all file paths and I/O.
    2.  Load and parse project-specific data formats.
    3.  Prepare data to meet the library's strict API requirements.
    4.  Interact with the library's high-level components (Builder, Logic).
    5.  Save the results returned by the library.
    """
    # --- 0. Setup: Logging and Output Directories ---
    # ORCHESTRATOR'S RESPONSIBILITY: Manage the execution environment, including
    # where logs and outputs should be stored. The library itself performs no logging
    # to files and never creates its own directories.
    output_dir = tmp_path / "cylinder_outputs"
    log_file = tmp_path / "test_run.log"
    setup_logging(logging.DEBUG, log_file=log_file)
    logging.info(f"Orchestrator starting. All outputs will be in: {output_dir}")

    # --- 1. Data Loading and Preparation ---
    logging.info("Arrange: Loading all necessary data from disk.")

    # ORCHESTRATOR'S RESPONSIBILITY: Define the location of all input data.
    # The library is path-agnostic and never assumes a file structure.
    cylinder_test_dir = test_data_root / "create_cylinders_test"
    origin_spacing_path = cylinder_test_dir / "geometry.json"
    points_path = cylinder_test_dir / "points.json"
    reference_seg_path = cylinder_test_dir / "seg_corrected.nrrd"

    # Pre-condition checks are a good practice for robust orchestrators.
    assert origin_spacing_path.exists(), "Missing geometry.json"
    assert points_path.exists(), "Missing points.json"
    assert reference_seg_path.exists(), "Missing reference segmentation file"

    # ORCHESTRATOR'S RESPONSIBILITY: Perform all file I/O.
    with open(origin_spacing_path, "r") as f:
        geom = json.load(f)
        origin = np.array(geom["origin"])
        spacing = np.array(geom["spacing"])

    with open(points_path, "r") as f:
        points_data = json.load(f)

    # ORCHESTRATOR'S RESPONSIBILITY: Parse and transform raw data to meet the
    # library's strict, clean API requirements.
    #
    # THE TEACHABLE MOMENT:
    # Our `points.json` file stores points as separate "SVC_1", "SVC_2" keys.
    # The library's `build_cylinder_contract` method, however, requires a single,
    # clean NumPy array of shape (3, 3). It is the orchestrator's job to perform
    # this transformation. This keeps the library simple and decoupled from the
    # specific (and potentially legacy) format of our input files.
    logging.info("Transforming raw point data into the required NumPy format.")
    svc_points = np.array(
        [points_data["SVC_1"], points_data["SVC_2"], points_data["SVC_3"]]
    )
    ivc_points = np.array(
        [points_data["IVC_1"], points_data["IVC_2"], points_data["IVC_3"]]
    )

    # The library needs image shape, so the orchestrator provides it by
    # loading a reference image.
    ref_image = load_image(reference_seg_path)
    image_shape = ref_image.GetSize()

    # ORCHESTRATOR'S RESPONSIBILITY: Define the scientific or business-logic parameters.
    SVC_PARAMS = {"radius": 10.0, "height": 20.0, "name": "SVC_cylinder"}
    IVC_PARAMS = {"radius": 12.0, "height": 25.0, "name": "IVC_cylinder"}

    # --- 2. Initialize Library Components ---
    logging.info("Arrange: Initializing library components with prepared data.")

    # We now have all the clean, in-memory data the library needs.
    logic = SegmentationLogic()
    builder = SegmentationPathBuilder(
        output_dir=output_dir,
        origin=origin,
        spacing=spacing,
        image_shape=image_shape,
    )

    # --- 3. Execute Workflow ---
    logging.info("Act: Executing the cylinder creation workflow.")

    # We can loop through our tasks, a common pattern in orchestration.
    for params, points in [(SVC_PARAMS, svc_points), (IVC_PARAMS, ivc_points)]:
        logging.info(f"Processing cylinder: {params['name']}")

        # LIBRARY INTERACTION (Step 1): Use the Builder to construct the contract.
        # This validates the inputs and prepares them for the logic engine.
        contract = builder.build_cylinder_contract(
            cylinder_name=params["name"],
            points=points,
            slicer_radius=params["radius"],
            slicer_height=params["height"],
        )

        # LIBRARY INTERACTION (Step 2): Pass the contract to the Logic engine.
        # The logic engine returns a pure, in-memory SimpleITK image object.
        cylinder_image = logic.create_cylinder(contract)

        # ORCHESTRATOR'S RESPONSIBILITY: Persist the in-memory result from the
        # library to disk at the location specified in the contract.
        logging.info(f"Saving result to {contract.output_path}")
        save_image(cylinder_image, contract.output_path)

    # --- 4. Validation ---
    logging.info("Assert: Verifying all outputs were created successfully.")

    # The orchestrator should always verify the outputs to ensure the step
    # was successful before proceeding to the next step in a pipeline.
    expected_svc_path = output_dir / f"{SVC_PARAMS['name']}.nrrd"
    expected_ivc_path = output_dir / f"{IVC_PARAMS['name']}.nrrd"

    assert expected_svc_path.exists(), f"SVC output file was not created!"
    assert expected_ivc_path.exists(), f"IVC output file was not created!"
    logging.info("Test completed successfully.")

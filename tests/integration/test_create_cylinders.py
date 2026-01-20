# tests/integration/test_create_cylinders.py

import logging
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Assuming pycemrg.core is a dependency, this is correct
from pycemrg.core import setup_logging

# Your new, clean library imports
from pycemrg_image_analysis.logic import SegmentationLogic, SegmentationPathBuilder
from pycemrg_image_analysis.utilities import save_image, load_image

# The test function needs to "ask for" the fixtures it uses as arguments.
def test_create_cylinders(test_data_root: Path, tmp_path: Path):
    """
    Integration test for the cylinder creation workflow.
    It simulates an orchestrator by creating SVC and IVC cylinder masks.
    """
    # --- 0. Setup Detailed Logging ---
    log_file = tmp_path / "test_create_cylinders.log"
    setup_logging(log_level=logging.DEBUG, log_file=log_file)
    logging.info(f"Detailed logs for this test run are in: {log_file}")
    print(f"Test outputs will be saved in: {tmp_path}") # Simple print for now

    # --- 1. Define Paths and Load All Input Data ---
    logging.info("Arrange: Loading all input data from disk.")
    
    origin_spacing_path = test_data_root / "origin_spacing.json"
    points_path = test_data_root / "points.json"
    reference_seg_path = test_data_root / "seg_corrected.nrrd"
    output_dir = tmp_path / "cylinder_outputs"

    # --- Pre-condition Checks ---
    assert origin_spacing_path.exists(), "Missing origin_spacing.json"
    assert points_path.exists(), "Missing points.json"
    assert reference_seg_path.exists(), "Missing reference segmentation file"

    # Load data from files into memory
    with open(origin_spacing_path, 'r') as f:
        geom = json.load(f)
        origin = np.array(geom['origin'])
        spacing = np.array(geom['spacing'])

    with open(points_path, 'r') as f:
        points_data = json.load(f)
        svc_points = np.array(points_data['SVC'])
        ivc_points = np.array(points_data['IVC'])

    ref_image = load_image(reference_seg_path)
    image_shape = ref_image.GetSize() # SITK GetSize() is (x,y,z), which we need

    # --- ADJUST these hard-coded parameters as needed ---
    SVC_PARAMS = {"radius": 10.0, "height": 20.0, "name": "SVC_cylinder"}
    IVC_PARAMS = {"radius": 12.0, "height": 25.0, "name": "IVC_cylinder"}

    # --- 2. Initialize Library Dependencies ---
    logging.info("Arrange: Initializing library components.")
    logic = SegmentationLogic()
    builder = SegmentationPathBuilder(
        output_dir=output_dir,
        origin=origin,
        spacing=spacing,
        image_shape=image_shape,
    )

    # We will run the logic for both SVC and IVC to be thorough
    contracts_to_process = []
    
    # --- 3. Generate the Path Contracts ---
    logging.info("Act: Building contracts for SVC and IVC.")
    svc_contract = builder.build_cylinder_contract(
        cylinder_name=SVC_PARAMS["name"],
        points=svc_points,
        slicer_radius=SVC_PARAMS["radius"],
        slicer_height=SVC_PARAMS["height"],
    )
    contracts_to_process.append(svc_contract)
    
    ivc_contract = builder.build_cylinder_contract(
        cylinder_name=IVC_PARAMS["name"],
        points=ivc_points,
        slicer_radius=IVC_PARAMS["radius"],
        slicer_height=IVC_PARAMS["height"],
    )
    contracts_to_process.append(ivc_contract)

    # --- 4. Execute the Logic Engine ---
    logging.info("Act: Executing core logic for all contracts.")
    for contract in contracts_to_process:
        logging.info(f"Creating cylinder for: {contract.output_path.name}")
        
        # Execute the logic to get an image object in memory
        cylinder_image = logic.create_cylinder(contract)
        
        # Save the resulting image to disk
        save_image(cylinder_image, contract.output_path)

    # --- 5. Validation: Assert that key output files were created ---
    logging.info("Assert: Verifying all outputs.")
    for contract in contracts_to_process:
        output_path = contract.output_path
        logging.info(f"Checking output: {output_path}")
        
        assert output_path.exists(), f"Output file was not created: {output_path}"
        
        # Optional: More detailed checks on the output image
        result_image = load_image(output_path)
        assert result_image.GetSize() == image_shape
        assert np.allclose(result_image.GetOrigin(), origin)
        assert np.sum(sitk.GetArrayViewFromImage(result_image)) > 0, "Image is empty"

    logging.info("Test completed successfully.")
# tests/integration/test_ring_creation.py

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
    RingLogic,
    RingSemanticRole,
    RingCreationContract,
    RingRule,
    ApplicationStep,
)
from pycemrg_image_analysis.utilities import save_image, load_image, MaskOperationMode, filters


def test_ring_creation_subset(tmp_path: Path, test_data_root: Path):
    """
    Integration test for ring creation workflow.
    
    Tests a representative subset of rings:
    - LPV1: Simple ring (no replace_only rules)
    - SVC: Complex ring (multiple replace_only rules, RA myocardium)
    
    Demonstrates:
    - Reference image capture before ring sequence
    - Pre-computed atrium myocardium thresholds (efficiency pattern)
    - Simple and complex application step handling
    """
    ## Logging
    log_file = tmp_path / "run.log"
    setup_logging(logging.DEBUG, log_file=log_file)

    config_dir = tmp_path / "ring_config"
    output_dir = tmp_path / "ring_outputs"
    output_dir.mkdir()
    
    logging.info("1. Scaffold ring components")
    # Test subset: one simple ring (LPV1), one complex ring (SVC)
    ring_components = ["lpv1_ring", "svc_ring"]

    scaffolder = ImageAnalysisScaffolder()
    scaffolder.scaffold_components(config_dir, ring_components)
    
    logging.info("2. Load configurations and initialize tools")
    labels_path = config_dir / "labels.yaml"
    params_path = config_dir / "parameters.json"
    semantic_map_dir = config_dir / "semantic_maps"
    
    # Input: segmentation BEFORE rings (this becomes the reference)
    input_seg_path = test_data_root / "ring_test" / "seg_before_rings.nrrd"
    assert input_seg_path.exists(), f"Reference image not found: {input_seg_path}"
    
    # Reference image for distance map calculation
    reference_seg_path = test_data_root / "ring_test" / "seg_reference.nrrd"
    assert reference_seg_path.exists(), f"Reference image not found: {reference_seg_path}"

    label_manager = LabelManager(config_path=labels_path)
    ring_logic = RingLogic()
    
    with open(params_path, "r") as f:
        parameters = json.load(f)

    # Load images
    initial_image = load_image(input_seg_path)
    reference_image = load_image(reference_seg_path)  # Frozen snapshot for distance maps
    
    # Start with copy of input segmentation
    final_array = sitk.GetArrayFromImage(initial_image)

    logging.info("3. Pre-compute atrium myocardium thresholds (efficiency pattern)")
    # Extract threshold parameters
    ring_thickness = parameters["ring_thickness"]
    
    # Compute LA myocardium threshold (for LPV1)
    la_myo_label = label_manager.get_value("LA_myo_label")
    la_myo_distmap = filters.distance_map(initial_image, la_myo_label, use_image_spacing=True)
    la_myo_thresh = filters.threshold_filter(
        la_myo_distmap, lower=0, upper=ring_thickness, binarise=True
    )
    logging.info("  LA myocardium threshold computed")
    
    # Compute RA myocardium threshold (for SVC)
    ra_myo_label = label_manager.get_value("RA_myo_label")
    ra_myo_distmap = filters.distance_map(initial_image, ra_myo_label, use_image_spacing=True)
    ra_myo_thresh = filters.threshold_filter(
        ra_myo_distmap, lower=0, upper=ring_thickness, binarise=True
    )
    logging.info("  RA myocardium threshold computed")

    logging.info("4. Execute ring creation workflow")
    
    # Map ring names to their atrium thresholds
    threshold_map = {
        "lpv1_ring": la_myo_thresh,
        "svc_ring": ra_myo_thresh,
    }
    
    for i, step_name in enumerate(ring_components):
        logging.info(f'---- Running step {i+1}/{len(ring_components)}: {step_name}')
        map_path = semantic_map_dir / f'{step_name}.json'
        
        with open(map_path, 'r') as f:
            raw_map = json.load(f)
            semantic_map = {RingSemanticRole[k]: v for k, v in raw_map.items()}

        # --- Build the RingRule object ---
        
        # Parse application steps
        application_steps = [
            ApplicationStep(
                mode=MaskOperationMode[step_data["MODE"]],
                rule_label_names=step_data["RULE_LABEL_NAMES"]
            )
            for step_data in semantic_map[RingSemanticRole.APPLICATION_STEPS]
        ]
        
        # Create RingRule
        ring_rule = RingRule(
            source_vein_label_name=semantic_map[RingSemanticRole.SOURCE_VEIN_LABEL_NAME],
            target_ring_label_name=semantic_map[RingSemanticRole.TARGET_RING_LABEL_NAME],
            ring_thickness_parameter_name=semantic_map[RingSemanticRole.RING_THICKNESS_PARAMETER_NAME],
            atrium_myocardium_name=semantic_map[RingSemanticRole.ATRIUM_MYOCARDIUM_NAME],
            application_steps=application_steps
        )

        # Create contract with pre-computed threshold and reference image
        contract = RingCreationContract(
            input_image=initial_image,  # Operate on initial image
            label_manager=label_manager,
            parameters=parameters,
            output_path=output_dir / f"{step_name}.nrrd",
            rule=ring_rule,
            reference_image=reference_image,  # Frozen snapshot for distance maps
            atrium_myocardium_threshold=threshold_map[step_name]  # Pre-computed
        )

        # Execute ring creation
        result_image_with_ring = ring_logic.create_from_rule(contract)
        
        # Merge the new ring into final composite array
        result_array = sitk.GetArrayFromImage(result_image_with_ring)
        target_ring_value = label_manager.get_value(ring_rule.target_ring_label_name)
        final_array[result_array == target_ring_value] = target_ring_value
        
        # Save intermediate step and assert
        save_image(result_image_with_ring, contract.output_path)
        assert contract.output_path.exists()
        assert target_ring_value in result_array, f"Ring label {target_ring_value} not found in result"
        
        logging.info(f"  Ring {step_name} created successfully (label {target_ring_value})")

    # --- FINAL ASSERTION ---
    logging.info("5. Validate final composite image")
    final_image = sitk.GetImageFromArray(final_array)
    final_image.CopyInformation(initial_image)
    final_output_path = output_dir / "seg_with_rings.nrrd"
    save_image(final_image, final_output_path)

    assert final_output_path.exists()
    
    # Verify both rings are present
    lpv1_ring_label = label_manager.get_value("LPV1_ring_label")
    svc_ring_label = label_manager.get_value("SVC_ring_label")
    
    assert lpv1_ring_label in final_array, "LPV1 ring not found in final image"
    assert svc_ring_label in final_array, "SVC ring not found in final image"
    
    logging.info("Ring creation test completed successfully")
    logging.info(f"  LPV1 ring voxels: {np.sum(final_array == lpv1_ring_label)}")
    logging.info(f"  SVC ring voxels: {np.sum(final_array == svc_ring_label)}")
"""
Orchestrator Patterns for Common Workflows

This module demonstrates how to compose image analysis steps
for different anatomical targets using the recipe system and
contract builders.

Each pattern is a complete, runnable function showing:
- How to scaffold configs
- How to build contracts from semantic maps
- How to execute steps in the correct order
- How to handle dependencies between steps
"""

from pathlib import Path
import json
import dataclasses
from typing import Optional

from pycemrg.data.labels import LabelManager
from pycemrg_image_analysis import ImageAnalysisScaffolder
from pycemrg_image_analysis.recipes import get_recipe
from pycemrg_image_analysis.logic import (
    MyocardiumLogic,
    MyocardiumPathBuilder,
    ValveLogic,
    RingLogic,
    MyocardiumSemanticRole,
    ValveSemanticRole,
    RingSemanticRole,
    MyocardiumRule,
    ValveRule,
    RingRule,
    ApplicationStep,
    PushStructureContract,
)
from pycemrg_image_analysis.utilities import (
    load_image,
    save_image,
    MaskOperationMode,
    filters,
)
import SimpleITK as sitk


# =============================================================================
# PATTERN 1: Recipe-Driven Workflow
# =============================================================================

def run_recipe_workflow(
    recipe_name: str,
    input_seg_path: Path,
    output_dir: Path,
    config_dir: Optional[Path] = None,
) -> sitk.Image:
    """
    Execute a pre-defined recipe workflow.
    
    This is the simplest pattern — let the recipe define what to do,
    and execute each step in order.
    
    Args:
        recipe_name: Name from RECIPE_CATALOG (e.g., "biventricular_basic")
        input_seg_path: Path to input segmentation
        output_dir: Where to save intermediate results
        config_dir: Where to generate configs (default: output_dir/config)
        
    Returns:
        Final segmentation image with all structures added
        
    Example:
        >>> result = run_recipe_workflow(
        ...     "biventricular_basic",
        ...     Path("seg_input.nrrd"),
        ...     Path("output/")
        ... )
    """
    if config_dir is None:
        config_dir = output_dir / "config"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Get recipe
    recipe = get_recipe(recipe_name)
    print(f"Running recipe: {recipe.name}")
    print(f"  {recipe.description}")
    print(f"  {len(recipe.steps)} steps")
    
    # 2. Scaffold all required configs at once
    scaffolder = ImageAnalysisScaffolder()
    scaffolder.scaffold_components(config_dir, recipe.required_schematics)
    
    # 3. Load tools
    label_manager = LabelManager(config_path=config_dir / "labels.yaml")
    with open(config_dir / "parameters.json") as f:
        parameters = json.load(f)
    
    myo_logic = MyocardiumLogic()
    valve_logic = ValveLogic()
    ring_logic = RingLogic()
    
    # 4. Execute steps
    current_seg = load_image(input_seg_path)
    
    # For rings: capture reference before starting
    reference_seg = None
    
    for i, step in enumerate(recipe.steps, 1):
        print(f"\nStep {i}/{len(recipe.steps)}: {step.step_type} - {step.component_name}")
        
        if step.step_type == "create":
            contract = build_myocardium_contract(
                step.component_name, current_seg, label_manager, 
                parameters, config_dir, output_dir
            )
            current_seg = myo_logic.create_from_rule(contract)
            save_image(current_seg, contract.output_path)
            
        elif step.step_type == "valve":
            contract = build_valve_contract(
                step.component_name, current_seg, label_manager,
                parameters, config_dir, output_dir
            )
            current_seg = valve_logic.create_from_rule(contract)
            save_image(current_seg, contract.output_path)
            
        elif step.step_type == "ring":
            # Capture reference on first ring
            if reference_seg is None:
                reference_seg = current_seg
            
            contract = build_ring_contract(
                step.component_name, current_seg, reference_seg,
                label_manager, parameters, config_dir, output_dir
            )
            current_seg = ring_logic.create_from_rule(contract)
            save_image(current_seg, contract.output_path)
            
        elif step.step_type == "push":
            contract = build_push_contract(
                step.component_name, current_seg, label_manager,
                parameters, config_dir
            )
            current_seg = myo_logic.push_structure(contract)
            save_image(current_seg, output_dir / f"{step.component_name}.nrrd")
    
    # 5. Save final result
    final_path = output_dir / f"seg_final_{recipe_name}.nrrd"
    save_image(current_seg, final_path)
    print(f"\n✅ Recipe complete. Final output: {final_path}")
    
    return current_seg


# =============================================================================
# PATTERN 2: Custom Workflow (Pick and Choose)
# =============================================================================

def biventricular_with_specific_valves(
    input_seg_path: Path,
    output_dir: Path,
    include_mitral: bool = True,
    include_tricuspid: bool = True,
    include_aortic: bool = False,
) -> sitk.Image:
    """
    Custom workflow: Choose which valves to include.
    
    Demonstrates how to compose steps manually without using
    a pre-defined recipe.
    
    Args:
        input_seg_path: Input segmentation
        output_dir: Output directory
        include_mitral: Create mitral valve
        include_tricuspid: Create tricuspid valve
        include_aortic: Create aortic valve
        
    Returns:
        Final segmentation
    """
    config_dir = output_dir / "config"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Build component list based on user choices
    components = ["lv_outflow", "rv_myocardium"]
    
    if include_mitral:
        components.append("mitral_valve")
    if include_tricuspid:
        components.append("tricuspid_valve")
    if include_aortic:
        components.extend(["aortic_wall", "aortic_valve"])
    
    print(f"Creating: {', '.join(components)}")
    
    # 2. Scaffold and execute
    scaffolder = ImageAnalysisScaffolder()
    scaffolder.scaffold_components(config_dir, components)
    
    label_manager = LabelManager(config_path=config_dir / "labels.yaml")
    with open(config_dir / "parameters.json") as f:
        parameters = json.load(f)
    
    myo_logic = MyocardiumLogic()
    valve_logic = ValveLogic()
    
    current_seg = load_image(input_seg_path)
    
    # 3. Always create myocardium first
    for component in ["lv_outflow", "rv_myocardium", "aortic_wall"]:
        if component not in components:
            continue
            
        contract = build_myocardium_contract(
            component, current_seg, label_manager,
            parameters, config_dir, output_dir
        )
        current_seg = myo_logic.create_from_rule(contract)
        save_image(current_seg, contract.output_path)
    
    # 4. Then create valves
    valve_map = {
        "mitral_valve": include_mitral,
        "tricuspid_valve": include_tricuspid,
        "aortic_valve": include_aortic,
    }
    
    for valve_name, should_create in valve_map.items():
        if not should_create:
            continue
            
        contract = build_valve_contract(
            valve_name, current_seg, label_manager,
            parameters, config_dir, output_dir
        )
        current_seg = valve_logic.create_from_rule(contract)
        save_image(current_seg, contract.output_path)
    
    final_path = output_dir / "seg_final_custom.nrrd"
    save_image(current_seg, final_path)
    
    return current_seg


# =============================================================================
# PATTERN 3: Atrial Workflow with Pre-computed Thresholds
# =============================================================================

def left_atrium_with_efficient_rings(
    input_seg_path: Path,
    output_dir: Path,
) -> sitk.Image:
    """
    Demonstrates efficient ring creation with pre-computed thresholds.
    
    Pre-computes the LA myocardium threshold once and reuses it
    for all rings, avoiding redundant distance map calculations.
    """
    config_dir = output_dir / "config"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Scaffold
    recipe = get_recipe("left_atrium_with_veins")
    scaffolder = ImageAnalysisScaffolder()
    scaffolder.scaffold_components(config_dir, recipe.required_schematics)
    
    # 2. Load tools
    label_manager = LabelManager(config_path=config_dir / "labels.yaml")
    with open(config_dir / "parameters.json") as f:
        parameters = json.load(f)
    
    myo_logic = MyocardiumLogic()
    ring_logic = RingLogic()
    
    current_seg = load_image(input_seg_path)
    
    # 3. Create LA myocardium
    la_myo_contract = build_myocardium_contract(
        "la_myocardium", current_seg, label_manager,
        parameters, config_dir, output_dir
    )
    current_seg = myo_logic.create_from_rule(la_myo_contract)
    save_image(current_seg, la_myo_contract.output_path)
    
    # 4. Pre-compute LA myocardium threshold (efficiency pattern)
    print("Pre-computing LA myocardium threshold for ring trimming...")
    la_myo_label = label_manager.get_value("LA_myo_label")
    ring_thickness = parameters["ring_thickness"]
    
    la_myo_distmap = filters.distance_map(
        current_seg, la_myo_label, use_image_spacing=True
    )
    la_myo_thresh = filters.threshold_filter(
        la_myo_distmap, lower=0, upper=ring_thickness, binarise=True
    )
    
    # 5. Capture reference for rings
    reference_seg = current_seg
    
    # 6. Create all rings with pre-computed threshold
    ring_names = ["lpv1_ring", "lpv2_ring", "rpv1_ring", "rpv2_ring", "laa_ring"]
    
    for ring_name in ring_names:
        print(f"Creating {ring_name}...")
        contract = build_ring_contract(
            ring_name, current_seg, reference_seg, label_manager,
            parameters, config_dir, output_dir,
            atrium_threshold=la_myo_thresh  # Reuse pre-computed threshold
        )
        current_seg = ring_logic.create_from_rule(contract)
        save_image(current_seg, contract.output_path)
    
    final_path = output_dir / "seg_final_la_with_rings.nrrd"
    save_image(current_seg, final_path)
    
    return current_seg


# =============================================================================
# CONTRACT BUILDER HELPERS
# =============================================================================

def build_myocardium_contract(
    component_name: str,
    current_seg: sitk.Image,
    label_manager: LabelManager,
    parameters: dict,
    config_dir: Path,
    output_dir: Path,
):
    """Build MyocardiumCreationContract from semantic map."""
    map_path = config_dir / "semantic_maps" / f"{component_name}.json"
    with open(map_path) as f:
        raw_map = json.load(f)
        semantic_map = {MyocardiumSemanticRole[k]: v for k, v in raw_map.items()}
    
    application_steps = [
        ApplicationStep(
            mode=MaskOperationMode[step["MODE"]],
            rule_label_names=step["RULE_LABEL_NAMES"]
        )
        for step in semantic_map[MyocardiumSemanticRole.APPLICATION_STEPS]
    ]
    
    rule = MyocardiumRule(
        source_bp_label_name=semantic_map[MyocardiumSemanticRole.SOURCE_BLOOD_POOL_NAME],
        target_myo_label_name=semantic_map[MyocardiumSemanticRole.TARGET_MYOCARDIUM_NAME],
        wall_thickness_parameter_name=semantic_map[MyocardiumSemanticRole.WALL_THICKNESS_PARAMETER_NAME],
        application_steps=application_steps
    )
    
    from pycemrg_image_analysis.logic import MyocardiumCreationContract
    return MyocardiumCreationContract(
        input_image=current_seg,
        label_manager=label_manager,
        parameters=parameters,
        output_path=output_dir / f"{component_name}.nrrd",
        rule=rule
    )


def build_valve_contract(
    component_name: str,
    current_seg: sitk.Image,
    label_manager: LabelManager,
    parameters: dict,
    config_dir: Path,
    output_dir: Path,
):
    """Build ValveCreationContract from semantic map."""
    map_path = config_dir / "semantic_maps" / f"{component_name}.json"
    with open(map_path) as f:
        raw_map = json.load(f)
        semantic_map = {ValveSemanticRole[k]: v for k, v in raw_map.items()}
    
    application_steps = [
        ApplicationStep(
            mode=MaskOperationMode[step["MODE"]],
            rule_label_names=step["RULE_LABEL_NAMES"]
        )
        for step in semantic_map[ValveSemanticRole.APPLICATION_STEPS]
    ]
    
    rule = ValveRule(
        structure_a_name=semantic_map[ValveSemanticRole.STRUCTURE_A_NAME],
        structure_b_name=semantic_map[ValveSemanticRole.STRUCTURE_B_NAME],
        target_valve_name=semantic_map[ValveSemanticRole.TARGET_VALVE_NAME],
        intersection_thickness_parameter_name=semantic_map[ValveSemanticRole.INTERSECTION_THICKNESS_PARAMETER_NAME],
        application_steps=application_steps
    )
    
    from pycemrg_image_analysis.logic import ValveCreationContract
    return ValveCreationContract(
        input_image=current_seg,
        label_manager=label_manager,
        parameters=parameters,
        output_path=output_dir / f"{component_name}.nrrd",
        rule=rule
    )


def build_ring_contract(
    component_name: str,
    current_seg: sitk.Image,
    reference_seg: sitk.Image,
    label_manager: LabelManager,
    parameters: dict,
    config_dir: Path,
    output_dir: Path,
    atrium_threshold: Optional[sitk.Image] = None,
):
    """Build RingCreationContract from semantic map."""
    map_path = config_dir / "semantic_maps" / f"{component_name}.json"
    with open(map_path) as f:
        raw_map = json.load(f)
        semantic_map = {RingSemanticRole[k]: v for k, v in raw_map.items()}
    
    application_steps = [
        ApplicationStep(
            mode=MaskOperationMode[step["MODE"]],
            rule_label_names=step["RULE_LABEL_NAMES"]
        )
        for step in semantic_map[RingSemanticRole.APPLICATION_STEPS]
    ]
    
    rule = RingRule(
        source_vein_label_name=semantic_map[RingSemanticRole.SOURCE_VEIN_LABEL_NAME],
        target_ring_label_name=semantic_map[RingSemanticRole.TARGET_RING_LABEL_NAME],
        ring_thickness_parameter_name=semantic_map[RingSemanticRole.RING_THICKNESS_PARAMETER_NAME],
        atrium_myocardium_name=semantic_map[RingSemanticRole.ATRIUM_MYOCARDIUM_NAME],
        application_steps=application_steps
    )
    
    from pycemrg_image_analysis.logic import RingCreationContract
    return RingCreationContract(
        input_image=current_seg,
        label_manager=label_manager,
        parameters=parameters,
        output_path=output_dir / f"{component_name}.nrrd",
        rule=rule,
        reference_image=reference_seg,
        atrium_myocardium_threshold=atrium_threshold,
    )


def build_push_contract(
    component_name: str,
    current_seg: sitk.Image,
    label_manager: LabelManager,
    parameters: dict,
    config_dir: Path,
):
    """Build PushStructureContract from semantic map."""
    # Push steps use a different pattern - parameters are in the schematic
    # but the contract is built directly, not from semantic roles
    
    # This is a simplified version - actual implementation would need
    # to map component_name to specific push operation parameters
    raise NotImplementedError(
        "Push contract building is workflow-specific. "
        "See test_myocardium_no_cuts.py for examples."
    )
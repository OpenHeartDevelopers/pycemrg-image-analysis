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

`run_recipe_workflow` is the reference runner used by the recipe-authoring
guide (docs/guides/authoring_a_recipe.md). Note the engine API is not uniform:
myocardium uses `MyocardiumLogic.create_from_semantic_map(image, label_manager,
parameters, semantic_map)`, while valves and rings use `create_from_rule(contract)`.
The dispatch in `run_recipe_workflow` reflects that asymmetry.
"""

from pathlib import Path
import json
from typing import Optional, Union, Dict

from pycemrg.data.labels import LabelManager
from pycemrg_image_analysis import ImageAnalysisScaffolder
from pycemrg_image_analysis.recipes import get_recipe, Recipe
from pycemrg_image_analysis.logic import (
    MyocardiumLogic,
    MyocardiumPathBuilder,
    ValveLogic,
    RingLogic,
    MyocardiumSemanticRole,
    ValveSemanticRole,
    RingSemanticRole,
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
    recipe: Union[str, Recipe],
    input_seg_path: Path,
    output_dir: Path,
    config_dir: Optional[Path] = None,
    label_mapping: Optional[Dict[str, int]] = None,
    extra_schematics: Optional[Dict[str, dict]] = None,
) -> sitk.Image:
    """
    Execute a recipe workflow.

    This is the simplest pattern — let the recipe define what to do,
    and execute each step in order. It works for both built-in recipes
    (passed by name) and recipes you define in your own project (passed as
    a Recipe object).

    Args:
        recipe: Either a catalog name (e.g., "biventricular_basic") or a
            Recipe object you constructed yourself.
        input_seg_path: Path to input segmentation
        output_dir: Where to save intermediate results
        config_dir: Where to generate configs (default: output_dir/config)
        label_mapping: Optional {label_name: int} overriding the schematic
            default voxel values. Use this when your segmentation's integer
            labels differ from the schematic defaults (e.g., a Slicer 1..N
            reset). When provided, configs are scaffolded with these values.
        extra_schematics: Optional {name: schematic dict} registering schematics
            defined in your own project, so a custom Recipe can reference
            components that are not built into the library.

    Returns:
        Final segmentation image with all structures added

    Example:
        >>> # Built-in recipe, default labels
        >>> result = run_recipe_workflow(
        ...     "biventricular_basic",
        ...     Path("seg_input.nrrd"),
        ...     Path("output/"),
        ... )
        >>>
        >>> # Your own recipe + your own schematics + remapped labels
        >>> result = run_recipe_workflow(
        ...     MY_RECIPE,
        ...     Path("seg_input.nrrd"),
        ...     Path("output/"),
        ...     label_mapping={"Foo_BP_label": 1, "Foo_myo_label": 5},
        ...     extra_schematics=MY_SCHEMATICS,
        ... )
    """
    if config_dir is None:
        config_dir = output_dir / "config"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve recipe (catalog name or a caller-supplied Recipe object)
    if isinstance(recipe, str):
        recipe = get_recipe(recipe)
    print(f"Running recipe: {recipe.name}")
    print(f"  {recipe.description}")
    print(f"  {len(recipe.steps)} steps")

    # 2. Scaffold all required configs at once. With a label_mapping, the
    #    scaffolded labels.yaml uses the caller's integer values instead of
    #    the schematic defaults.
    scaffolder = ImageAnalysisScaffolder(extra_schematics=extra_schematics)
    if label_mapping:
        scaffolder.scaffold_components_with_mapping(
            config_dir, recipe.required_schematics, label_mapping
        )
    else:
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
            semantic_map = load_myocardium_semantic_map(
                step.component_name, config_dir
            )
            current_seg = myo_logic.create_from_semantic_map(
                current_seg, label_manager, parameters, semantic_map
            )
            save_image(current_seg, output_dir / f"{step.component_name}.nrrd")

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
                step.component_name, label_manager, parameters
            )
            current_seg = myo_logic.push_structure(current_seg, contract)
            save_image(current_seg, output_dir / f"{step.component_name}.nrrd")
    
    # 5. Save final result
    final_path = output_dir / f"seg_final_{recipe.name}.nrrd"
    save_image(current_seg, final_path)
    print(f"\nRecipe complete. Final output: {final_path}")
    
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

        semantic_map = load_myocardium_semantic_map(component, config_dir)
        current_seg = myo_logic.create_from_semantic_map(
            current_seg, label_manager, parameters, semantic_map
        )
        save_image(current_seg, output_dir / f"{component}.nrrd")
    
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
    la_myo_map = load_myocardium_semantic_map("la_myocardium", config_dir)
    current_seg = myo_logic.create_from_semantic_map(
        current_seg, label_manager, parameters, la_myo_map
    )
    save_image(current_seg, output_dir / "la_myocardium.nrrd")
    
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

def load_myocardium_semantic_map(
    component_name: str,
    config_dir: Path,
) -> dict:
    """
    Load a scaffolded semantic map as a role-keyed dict.

    `MyocardiumLogic.create_from_semantic_map` consumes the semantic map
    directly (no contract, no rule). The on-disk JSON uses the role enum
    *names* as keys; this converts them back to `MyocardiumSemanticRole`
    members. The APPLICATION_STEPS list is left as raw dicts because the
    engine parses each step's "MODE"/"RULE_LABEL_NAMES" itself.

    Args:
        component_name: Component whose map to load (e.g., "la_myocardium").
        config_dir: Directory containing semantic_maps/<component_name>.json.

    Returns:
        dict[MyocardiumSemanticRole, Any] ready for create_from_semantic_map.
    """
    map_path = config_dir / "semantic_maps" / f"{component_name}.json"
    with open(map_path) as f:
        raw_map = json.load(f)
    return {MyocardiumSemanticRole[k]: v for k, v in raw_map.items()}


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


# Push steps have no semantic map (the "myo_push_steps" schematic only
# supplies the labels and parameters). Each push is defined here by which
# wall pushes which structure inward. The values are label/parameter *names*
# resolved against the scaffolded config at build time. To add a push step,
# add an entry here and reference its key as a WorkflowStep("push", <key>).
_PUSH_STEP_DEFINITIONS = {
    "part_push_aorta": {
        "pusher_wall": "Ao_wall_label",
        "pushed_wall": "PArt_wall_label",
        "pushed_bp": "PArt_BP_label",
        "thickness_param": "PArt_WT",
    },
    "part_push_lv": {
        "pusher_wall": "LV_myo_label",
        "pushed_wall": "PArt_wall_label",
        "pushed_bp": "PArt_BP_label",
        "thickness_param": "PArt_WT",
    },
    "la_push_aorta": {
        "pusher_wall": "Ao_wall_label",
        "pushed_wall": "LA_myo_label",
        "pushed_bp": "LA_BP_label",
        "thickness_param": "LA_WT",
    },
    "rv_push_aorta": {
        "pusher_wall": "Ao_wall_label",
        "pushed_wall": "RV_myo_label",
        "pushed_bp": "RV_BP_label",
        "thickness_param": "RV_WT",
    },
}


def build_push_contract(
    component_name: str,
    label_manager: LabelManager,
    parameters: dict,
) -> PushStructureContract:
    """
    Build a PushStructureContract for a named push step.

    Unlike create/valve/ring steps, push steps are not driven by a semantic
    map. The mapping from step name to the four contract fields lives in
    `_PUSH_STEP_DEFINITIONS`; this resolves those label/parameter names to
    concrete values for the current segmentation.

    Args:
        component_name: Push step key (e.g., "part_push_aorta").
        label_manager: Resolves label names to integer voxel values.
        parameters: Scaffolded parameters dict (wall thicknesses).

    Returns:
        PushStructureContract ready for MyocardiumLogic.push_structure().

    Raises:
        KeyError: If component_name is not a defined push step.
    """
    if component_name not in _PUSH_STEP_DEFINITIONS:
        available = ", ".join(_PUSH_STEP_DEFINITIONS)
        raise KeyError(
            f"Unknown push step '{component_name}'. "
            f"Defined push steps: {available}"
        )

    spec = _PUSH_STEP_DEFINITIONS[component_name]
    return PushStructureContract(
        pusher_wall_label=label_manager.get_value(spec["pusher_wall"]),
        pushed_wall_label=label_manager.get_value(spec["pushed_wall"]),
        pushed_bp_label=label_manager.get_value(spec["pushed_bp"]),
        pushed_wall_thickness=parameters[spec["thickness_param"]],
    )
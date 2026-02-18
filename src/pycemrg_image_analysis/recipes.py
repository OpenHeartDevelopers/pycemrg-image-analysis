# src/pycemrg_image_analysis/recipes.py

"""
Pre-defined workflow recipes for common cardiac analysis tasks.

A Recipe defines:
- Which components are needed (schematics to scaffold)
- Execution order (sequence of steps)
- Step types (create/valve/ring/push)

Use recipes to quickly set up standard workflows without manually
tracking dependencies and execution order.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class WorkflowStep:
    """Single step in a workflow."""
    step_type: str  # "create", "valve", "ring", "push"
    component_name: str
    
    def __repr__(self):
        return f"{self.step_type}:{self.component_name}"

@dataclass
class Recipe:
    """Pre-defined workflow recipe."""
    name: str
    description: str
    steps: List[WorkflowStep]
    required_schematics: List[str]
    
    def __repr__(self):
        return f"Recipe('{self.name}', {len(self.steps)} steps)"


# =============================================================================
# PRE-DEFINED RECIPES
# =============================================================================

BIVENTRICULAR_BASIC = Recipe(
    name="biventricular_basic",
    description="LV + RV myocardium with mitral and tricuspid valves",
    steps=[
        WorkflowStep("create", "lv_outflow"),
        WorkflowStep("create", "rv_myocardium"),
        WorkflowStep("valve", "mitral_valve"),
        WorkflowStep("valve", "tricuspid_valve"),
        WorkflowStep("valve", "aortic_valve"),
        WorkflowStep("valve", "pulmonary_valve"),
    ],
    required_schematics=[
        "lv_outflow",
        "rv_myocardium",
        "mitral_valve",
        "tricuspid_valve",
        "aortic_valve",
        "pulmonary_valve",
    ]
)

FOUR_CHAMBER_MYOCARDIUM = Recipe(
    name="four_chamber_myocardium",
    description="Complete four-chamber myocardium (no valves)",
    steps=[
        WorkflowStep("create", "lv_outflow"),
        WorkflowStep("create", "aortic_wall"),
        WorkflowStep("create", "pulmonary_artery"),
        WorkflowStep("push", "part_push_aorta"),
        WorkflowStep("push", "part_push_lv"),
        WorkflowStep("create", "rv_myocardium"),
        WorkflowStep("create", "la_myocardium"),
        WorkflowStep("push", "la_push_aorta"),
        WorkflowStep("create", "ra_myocardium"),
        WorkflowStep("push", "rv_push_aorta"),
    ],
    required_schematics=[
        "lv_outflow",
        "aortic_wall",
        "pulmonary_artery",
        "rv_myocardium",
        "la_myocardium",
        "ra_myocardium",
        "myo_push_steps",
    ]
)

FOUR_CHAMBER_FULL = Recipe(
    name="four_chamber_full",
    description="Complete four-chamber heart with all valves",
    steps=[
        # Myocardium
        WorkflowStep("create", "lv_outflow"),
        WorkflowStep("create", "aortic_wall"),
        WorkflowStep("create", "pulmonary_artery"),
        WorkflowStep("push", "part_push_aorta"),
        WorkflowStep("push", "part_push_lv"),
        WorkflowStep("create", "rv_myocardium"),
        WorkflowStep("create", "la_myocardium"),
        WorkflowStep("push", "la_push_aorta"),
        WorkflowStep("create", "ra_myocardium"),
        WorkflowStep("push", "rv_push_aorta"),
        # Valves
        WorkflowStep("valve", "mitral_valve"),
        WorkflowStep("valve", "tricuspid_valve"),
        WorkflowStep("valve", "aortic_valve"),
        WorkflowStep("valve", "pulmonary_valve"),
    ],
    required_schematics=[
        "lv_outflow",
        "aortic_wall",
        "pulmonary_artery",
        "rv_myocardium",
        "la_myocardium",
        "ra_myocardium",
        "myo_push_steps",
        "mitral_valve",
        "tricuspid_valve",
        "aortic_valve",
        "pulmonary_valve",
    ]
)

LEFT_ATRIUM_WITH_VEINS = Recipe(
    name="left_atrium_with_veins",
    description="LA myocardium with pulmonary vein rings",
    steps=[
        WorkflowStep("create", "la_myocardium"),
        WorkflowStep("ring", "lpv1_ring"),
        WorkflowStep("ring", "lpv2_ring"),
        WorkflowStep("ring", "rpv1_ring"),
        WorkflowStep("ring", "rpv2_ring"),
        WorkflowStep("ring", "laa_ring"),
    ],
    required_schematics=[
        "la_myocardium",
        "lpv1_ring",
        "lpv2_ring",
        "rpv1_ring",
        "rpv2_ring",
        "laa_ring",
    ]
)

RIGHT_ATRIUM_WITH_VEINS = Recipe(
    name="right_atrium_with_veins",
    description="RA myocardium with vena cava rings",
    steps=[
        WorkflowStep("create", "ra_myocardium"),
        WorkflowStep("ring", "svc_ring"),
        WorkflowStep("ring", "ivc_ring"),
    ],
    required_schematics=[
        "ra_myocardium",
        "svc_ring",
        "ivc_ring",
    ]
)

ATRIA_FULL = Recipe(
    name="atria_full",
    description="Both atria with all vein rings",
    steps=[
        # Left atrium
        WorkflowStep("create", "la_myocardium"),
        WorkflowStep("ring", "lpv1_ring"),
        WorkflowStep("ring", "lpv2_ring"),
        WorkflowStep("ring", "rpv1_ring"),
        WorkflowStep("ring", "rpv2_ring"),
        WorkflowStep("ring", "laa_ring"),
        # Right atrium
        WorkflowStep("create", "ra_myocardium"),
        WorkflowStep("ring", "svc_ring"),
        WorkflowStep("ring", "ivc_ring"),
    ],
    required_schematics=[
        "la_myocardium",
        "ra_myocardium",
        "lpv1_ring",
        "lpv2_ring",
        "rpv1_ring",
        "rpv2_ring",
        "laa_ring",
        "svc_ring",
        "ivc_ring",
    ]
)


# =============================================================================
# RECIPE CATALOG
# =============================================================================

RECIPE_CATALOG = {
    "biventricular_basic": BIVENTRICULAR_BASIC,
    "four_chamber_myocardium": FOUR_CHAMBER_MYOCARDIUM,
    "four_chamber_full": FOUR_CHAMBER_FULL,
    "left_atrium_with_veins": LEFT_ATRIUM_WITH_VEINS,
    "right_atrium_with_veins": RIGHT_ATRIUM_WITH_VEINS,
    "atria_full": ATRIA_FULL,
}


def list_recipes() -> None:
    """Print all available recipes with descriptions."""
    print("Available Recipes:")
    print("=" * 80)
    for name, recipe in RECIPE_CATALOG.items():
        print(f"\n{recipe.name}")
        print(f"  {recipe.description}")
        print(f"  Steps: {len(recipe.steps)}")
        print(f"  Components: {', '.join(recipe.required_schematics[:5])}", end="")
        if len(recipe.required_schematics) > 5:
            print(f", ... ({len(recipe.required_schematics)} total)")
        else:
            print()


def get_recipe(name: str) -> Recipe:
    """
    Get a recipe by name.
    
    Args:
        name: Recipe name (e.g., "biventricular_basic")
        
    Returns:
        Recipe object
        
    Raises:
        KeyError: If recipe name not found
    """
    if name not in RECIPE_CATALOG:
        available = ", ".join(RECIPE_CATALOG.keys())
        raise KeyError(
            f"Recipe '{name}' not found. Available recipes: {available}"
        )
    return RECIPE_CATALOG[name]
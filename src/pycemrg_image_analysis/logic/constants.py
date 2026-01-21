# src/pycemrg_image_analysis/logic/constants.py

from enum import Enum, auto

class MyocardiumSemanticRole(Enum):
    """
    Defines the abstract roles a label or parameter can play in myocardium creation.
    This is the library's internal, stable API vocabulary. The user provides a
    mapping from their custom names to these required roles.
    """
    # --- Input Data Roles ---
    # The name of the blood pool label to generate the distance map from.
    SOURCE_BLOOD_POOL_NAME = auto()
    # The name of the parameter defining the wall thickness in voxels/mm.
    WALL_THICKNESS_PARAMETER_NAME = auto()

    # --- Output Data Roles ---
    # The name of the new myocardium label to be created.
    TARGET_MYOCARDIUM_NAME = auto()
    
    # --- Rule Roles ---
    # The method to use when applying the new mask (e.g., "REPLACE_EXCEPT").
    APPLICATION_MODE = auto()
    # A list of label names to be used by the application mode rule.
    APPLICATION_RULE_LABEL_NAMES = auto()
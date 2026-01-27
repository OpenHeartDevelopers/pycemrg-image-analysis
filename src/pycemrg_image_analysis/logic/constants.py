from enum import Enum, auto

ZERO_LABEL = 0
class MyocardiumSemanticRole(Enum):
    """
    Defines the abstract roles a label or parameter can play in myocardium creation.
    This is the library's internal, stable API vocabulary. The user provides a
    mapping from their custom names to these required roles.
    """

    # --- Input Data Roles ---
    # The name of the blood pool label to generate the distance map from.
    SOURCE_BLOOD_POOL_NAME = auto()
    # The name of the parameter defining the wall thickness in millimeters.
    WALL_THICKNESS_PARAMETER_NAME = auto()

    # --- Output Data Roles ---
    # The name of the new myocardium label to be created.
    TARGET_MYOCARDIUM_NAME = auto()

    # --- Rule Roles ---
    # A list of dictionaries, where each dictionary defines a single
    # mask application step (mode, labels, etc.).
    APPLICATION_STEPS = auto()

class ValveSemanticRole(Enum):
    """
    Defines the abstract roles a label or parameter can play in valve creation.
    """
    STRUCTURE_A_NAME = auto()
    STRUCTURE_B_NAME = auto()
    TARGET_VALVE_NAME = auto()
    INTERSECTION_THICKNESS_PARAMETER_NAME = auto()
    APPLICATION_STEPS = auto()

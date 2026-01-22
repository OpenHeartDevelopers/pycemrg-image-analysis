# src/pycemrg_image_analysis/logic/__init__.py

"""
This module exposes the public API for the logic layer, including the main
logic engines and the convenience builders.
"""

from pycemrg_image_analysis.logic.builders import SegmentationPathBuilder, MyocardiumPathBuilder  
from pycemrg_image_analysis.logic.constants import MyocardiumSemanticRole, ZERO_LABEL
from pycemrg_image_analysis.logic.contracts import (
    CylinderCreationContract,
    MyocardiumRule,
    MyocardiumCreationContract,
    PushStructureContract,
)
from .segmentation import SegmentationLogic
from .myocardium import MyocardiumLogic

__all__ = [
    # Builders 
    "SegmentationPathBuilder",
    "MyocardiumPathBuilder",  
    # Constants
    "MyocardiumRule",
    "MyocardiumSemanticRole", 
    # Contracts
    "CylinderCreationContract",
    "MyocardiumCreationContract",
    "PushStructureContract",
    # Logic Engines
    "SegmentationLogic",
    "MyocardiumLogic",
]
# src/pycemrg_image_analysis/logic/contracts.py

import numpy as np
import SimpleITK as sitk

from pathlib import Path
from typing import Tuple
from enum import Enum, auto
from dataclasses import dataclass, field

from pycemrg.data.labels import LabelManager  
from pycemrg_image_analysis.utilities.masks import MaskOperationMode

@dataclass(frozen=True)
class CylinderCreationContract:
    """
    A data contract specifying all parameters needed for cylinder creation.

    This object is typically constructed by a Builder and consumed by a Logic
    class. It ensures that all necessary data is explicitly passed.
    """
    # Image metadata
    image_shape: Tuple[int, int, int]
    origin: np.ndarray
    spacing: np.ndarray

    # Cylinder geometry parameters
    points: np.ndarray
    slicer_radius: float
    slicer_height: float

    # Output specification
    output_path: Path

@dataclass(frozen=True)
class ApplicationStep:
    """Defines a single mask application operation."""
    mode: MaskOperationMode
    rule_label_names: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class MyocardiumRule:
    """Defines the full recipe for building one piece of myocardium."""
    source_bp_label_name: str
    target_myo_label_name: str
    wall_thickness_parameter_name: str
    # This now becomes a list of steps, not a single mode.
    application_steps: list[ApplicationStep]


@dataclass(frozen=True)
class CreationContract:
    """A generic base contract for 'create from map' style logic."""
    input_image: sitk.Image
    label_manager: LabelManager
    parameters: dict[str, float]
    output_path: Path

@dataclass(frozen=True)
class MyocardiumCreationContract(CreationContract):
    """Specific contract for myocardium creation, adding the rule.
    
    The single, generic contract for all myocardium creation tasks.

    This object bundles the input data (image), the specific rule for this run,
    the necessary tools (label_manager, parameters), and the output path.
    """
    rule: MyocardiumRule

@dataclass(frozen=True)
class PushStructureContract:
    """Contract for the 'push_structure' logic."""
    pusher_wall_label: int
    pushed_wall_label: int
    pushed_bp_label: int
    pushed_wall_thickness: float

@dataclass(frozen=True)
class ValveRule:
    """Defines the recipe for building one valve via intersection."""
    structure_a_name: str
    structure_b_name: str
    target_valve_name: str
    intersection_thickness_parameter_name: str
    application_steps: list[ApplicationStep]

@dataclass(frozen=True)
class ValveCreationContract(CreationContract):
    """Specific contract for valve creation, adding the rule."""
    rule: ValveRule
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
class MyocardiumRule:
    """
    A configuration object defining the scientific rules for building one
    piece of myocardium. This object is the "recipe" for a single structure.
    """
    source_bp_label_name: str
    target_myo_label_name: str
    wall_thickness_parameter_name: str
    application_mode: MaskOperationMode
    application_rule_label_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MyocardiumCreationContract:
    """
    The single, generic contract for all myocardium creation tasks.

    This object bundles the input data (image), the specific rule for this run,
    the necessary tools (label_manager, parameters), and the output path.
    """
    input_image: sitk.Image
    rule: MyocardiumRule
    label_manager: LabelManager
    parameters: dict[str, float]  # e.g., {"Ao_WT": 2.5, "RV_WT": 3.0}
    output_path: Path
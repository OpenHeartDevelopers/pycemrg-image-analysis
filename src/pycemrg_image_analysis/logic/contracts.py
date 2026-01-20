# src/pycemrg_image_analysis/logic/contracts.py

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np

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